// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Linux snapshot and pin limits read from fixture trees (TIER-E2-T02; RFC-0020 §3, §5).
//! [`ProcReader`] takes the filesystem root as a path, so these run on every OS.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use syntropy_core::Bytes;
use syntropy_tier::os::procfs::{ProcReader, ThpMode, numa_nodes};
use syntropy_tier::{CgroupMem, PinLimits, Psi};

const KIB: u64 = 1024;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// A fresh fixture root under the system temporary directory.
struct Root(PathBuf);

impl Root {
    fn new(name: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "syntropy-tier-proc-{}-{}-{name}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Root(dir)
    }

    fn put(&self, rel: &str, text: &str) -> &Self {
        let path = self.0.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
        self
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn node_meminfo(n: u16, free_kib: u64, active: u64, inactive: u64, srec: u64) -> String {
    format!(
        "Node {n} MemTotal:       33554432 kB\nNode {n} MemFree:        {free_kib} kB\n\
         Node {n} Active(file):   {active} kB\nNode {n} Inactive(file): {inactive} kB\n\
         Node {n} SReclaimable:   {srec} kB\n"
    )
}

/// A dual-node bare-metal host: no cgroup limit, PSI and vmstat readable, 8 MiB memlock, no
/// `CAP_IPC_LOCK`.
fn bare_metal() -> Root {
    let r = Root::new("bare");
    r.put(
        "proc/meminfo",
        "MemTotal:       67108864 kB\nMemFree:         1048576 kB\nMemAvailable:   41943040 kB\n",
    )
    .put(
        "sys/devices/system/node/node0/meminfo",
        &node_meminfo(0, 524288, 1048576, 2097152, 131072),
    )
    .put(
        "sys/devices/system/node/node1/meminfo",
        &node_meminfo(1, 262144, 0, 0, 0),
    )
    .put("proc/self/cgroup", "0::/user.slice/user-1000.slice/session-1.scope\n")
    .put(
        "proc/pressure/memory",
        "some avg10=2.50 avg60=1.00 avg300=0.10 total=99\nfull avg10=0.50 avg60=0.00 avg300=0.00 total=9\n",
    )
    .put("proc/vmstat", "nr_free_pages 1\nallocstall_normal 10\nallocstall_movable 5\n")
    .put(
        "proc/self/status",
        "Name:\tx\nVmLck:\t     128 kB\nVmPin:\t      64 kB\nCapEff:\t0000000000000000\n",
    )
    .put(
        "proc/self/limits",
        "Limit                     Soft Limit           Hard Limit           Units\n\
         Max locked memory         8388608              8388608              bytes\n",
    )
    .put("sys/kernel/mm/transparent_hugepage/enabled", "always [madvise] never\n");
    r
}

#[test]
fn bare_metal_snapshot() {
    let root = bare_metal();
    let reader = ProcReader::new(root.path(), Bytes(GIB));
    let s = reader.snapshot();
    assert_eq!(s.total, Bytes(64 * GIB));
    assert_eq!(s.available, Bytes(40 * GIB));
    // MemFree per node, reported only.
    assert_eq!(
        s.numa_free.as_slice(),
        &[(0, Bytes(512 * MIB)), (1, Bytes(256 * MIB))]
    );
    // node 0: 512 MiB + 1 GiB + 2 GiB + 128 MiB − 1 GiB; node 1: 256 MiB − 1 GiB floors at 0.
    assert_eq!(
        s.numa_available.as_slice(),
        &[(0, Bytes(2 * GIB + 640 * MIB)), (1, Bytes(0))]
    );
    assert_eq!(s.cgroup, None);
    assert_eq!(
        s.psi,
        Some(Psi {
            some_avg10: 2.5,
            full_avg10: 0.5
        })
    );
    assert_eq!(s.own_locked, Bytes(128 * KIB));
    assert_eq!(s.own_registered, Bytes(64 * KIB));
    assert_eq!(s.commit_headroom, None);
    // The reader took its baseline at construction and nothing changed since.
    assert_eq!(s.direct_reclaim_stalls, Some(0));
    root.put(
        "proc/vmstat",
        "allocstall_normal 13\nallocstall_movable 6\n",
    );
    assert_eq!(reader.snapshot().direct_reclaim_stalls, Some(4));

    assert_eq!(
        reader.limits(),
        PinLimits {
            memlock: Some(Bytes(8 * MIB)),
            cap_ipc_lock: false,
            io_fixed_max: Some(Bytes(8 * MIB - 64 * KIB)),
            ..PinLimits::default()
        }
    );
    assert_eq!(reader.thp(), ThpMode::Madvise);
}

#[test]
fn cap_ipc_lock_or_no_limit_is_unlimited() {
    let root = bare_metal();
    root.put(
        "proc/self/status",
        "VmLck:\t0 kB\nVmPin:\t0 kB\nCapEff:\t0000000000004000\n",
    );
    let limits = ProcReader::new(root.path(), Bytes(GIB)).limits();
    assert!(limits.cap_ipc_lock);
    assert_eq!(limits.io_fixed_max, None);

    root.put(
        "proc/self/status",
        "VmLck:\t0 kB\nVmPin:\t0 kB\nCapEff:\t0000000000000000\n",
    )
    .put(
        "proc/self/limits",
        "Max locked memory         unlimited            unlimited            bytes\n",
    );
    let limits = ProcReader::new(root.path(), Bytes(GIB)).limits();
    assert_eq!((limits.memlock, limits.io_fixed_max), (None, None));
}

#[test]
fn unreadable_limits_mean_no_lock_budget() {
    let root = Root::new("nolimits");
    let limits = ProcReader::new(root.path(), Bytes(GIB)).limits();
    assert_eq!(limits.memlock, Some(Bytes(0)));
    assert!(!limits.cap_ipc_lock);
}

/// cgroup v2 in a container: the process's own cgroup sets `memory.max`, its parent sets a
/// tighter `memory.high`. The parent binds first, so it is reported; the event counters are
/// deltas from the previous reading.
#[test]
fn cgroup_v2_reports_the_binding_level() {
    let root = bare_metal();
    root.put("proc/self/cgroup", "0::/app/worker\n")
        .put("sys/fs/cgroup/app/worker/memory.max", "17179869184\n")
        .put("sys/fs/cgroup/app/worker/memory.high", "max\n")
        .put("sys/fs/cgroup/app/worker/memory.current", "2147483648\n")
        .put(
            "sys/fs/cgroup/app/worker/memory.events",
            "low 0\nhigh 0\nmax 0\n",
        )
        .put("sys/fs/cgroup/app/memory.max", "max\n")
        .put("sys/fs/cgroup/app/memory.high", "8589934592\n")
        .put("sys/fs/cgroup/app/memory.current", "4294967296\n")
        .put("sys/fs/cgroup/app/memory.events", "low 0\nhigh 5\nmax 1\n");
    let reader = ProcReader::new(root.path(), Bytes(GIB));
    let first = reader.snapshot().cgroup.unwrap();
    assert_eq!(
        first,
        CgroupMem {
            max: None,
            high: Some(Bytes(8 * GIB)),
            current: Bytes(4 * GIB),
            high_events: 0,
            max_events: 0,
        }
    );
    root.put("sys/fs/cgroup/app/memory.events", "low 0\nhigh 8\nmax 1\n");
    let next = reader.snapshot().cgroup.unwrap();
    assert_eq!((next.high_events, next.max_events), (3, 0));

    // With the parent's high lifted, the process's own memory.max binds.
    root.put("sys/fs/cgroup/app/memory.high", "max\n");
    let own = reader.snapshot().cgroup.unwrap();
    assert_eq!(own.max, Some(Bytes(16 * GIB)));
    assert_eq!(own.current, Bytes(2 * GIB));
}

/// cgroup v1: the `memory` controller's `limit_in_bytes` and `usage_in_bytes`; no `high`.
#[test]
fn cgroup_v1_memory_controller() {
    let root = bare_metal();
    root.put(
        "proc/self/cgroup",
        "12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n0::/\n",
    )
    .put(
        "sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes",
        "8589934592\n",
    )
    .put(
        "sys/fs/cgroup/memory/docker/abc/memory.usage_in_bytes",
        "536870912\n",
    )
    .put(
        "sys/fs/cgroup/memory/memory.limit_in_bytes",
        "9223372036854771712\n",
    )
    .put("sys/fs/cgroup/memory/memory.usage_in_bytes", "1\n");
    let c = ProcReader::new(root.path(), Bytes(GIB))
        .snapshot()
        .cgroup
        .unwrap();
    assert_eq!(c.max, Some(Bytes(8 * GIB)));
    assert_eq!(c.high, None);
    assert_eq!(c.current, Bytes(512 * MIB));
}

/// Neither PSI nor vmstat readable: both signals are missing, not zero.
#[test]
fn missing_pressure_signals_are_none() {
    let root = Root::new("bare-min");
    root.put("proc/meminfo", "MemTotal: 1024 kB\nMemAvailable: 512 kB\n");
    let s = ProcReader::new(root.path(), Bytes(0)).snapshot();
    assert_eq!(s.psi, None);
    assert_eq!(s.direct_reclaim_stalls, None);
    assert!(s.numa_free.is_empty());
    assert_eq!(numa_nodes(root.path(), Bytes(0)).available.len(), 0);
}
