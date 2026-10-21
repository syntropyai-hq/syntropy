// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host memory as the OS reports it (RFC-0020 §1, §3, §5, §6). This task defines
//! [`MemSnapshot`], the fields the §6 pinning gate reads, and the live sources for two of them:
//! the per-node available estimate ([`numa_available`]) and the direct-reclaim stall delta
//! ([`ReclaimStalls`]). The `OsMemory` adapters that fill the rest come in TIER-E2-T02 (Linux),
//! TIER-E2-T03 (Windows) and later tasks.
//!
//! This is one of the two modules where RFC-0020 §1 allows `unsafe`. It holds none yet, so it
//! keeps the crate-level `deny(unsafe_code)`; every source here uses `std::fs` reads. The task
//! that adds the first `unsafe` block lifts the deny here with a scoped `allow`.

use std::path::Path;
use std::time::Instant;

use smallvec::SmallVec;
use syntropy_core::Bytes;

use crate::pressure::PressureLevel;

/// One reading of host memory, taken before every pinning step (RFC-0020 §3, §6).
///
/// Not serializable: it is a live reading stamped with an [`Instant`], read on the node that
/// took it. The residency digest (RFC-0020 §13) reports what it needs from it.
#[derive(Clone, Debug, PartialEq)]
pub struct MemSnapshot {
    /// Physical RAM.
    pub total: Bytes,
    /// `MemAvailable` / `ullAvailPhys` / free + inactive + purgeable.
    pub available: Bytes,
    /// Free memory per NUMA node. Reported only; the gate never waits on it (RFC-0020 §6).
    pub numa_free: SmallVec<[(u16, Bytes); 4]>,
    /// Per-node estimate of what the node can supply after reclaiming clean cache, with
    /// `numa_reserve` already subtracted once (RFC-0020 §5): `MemFree + Active(file) +
    /// Inactive(file) + SReclaimable − numa_reserve`, from
    /// `/sys/devices/system/node/node<n>/meminfo`. See [`numa_available`].
    pub numa_available: SmallVec<[(u16, Bytes); 4]>,
    /// The cgroup's memory limits and usage, where the process runs in one.
    pub cgroup: Option<CgroupMem>,
    /// Windows: commit limit − commit charge. `None` on every other OS, which has no commit
    /// accounting to respect. See [`commit_headroom`]. This is a plain `pub` field, so the rule
    /// holds only for values built through [`commit_headroom`], which every `OsMemory` adapter
    /// is to use.
    pub commit_headroom: Option<Bytes>,
    /// Bytes this process holds locked.
    pub own_locked: Bytes,
    /// Bytes this process holds DMA-registered.
    pub own_registered: Bytes,
    /// Memory pressure stall information from `/proc/pressure/memory` (Linux), when readable.
    pub psi: Option<Psi>,
    /// Linux: direct-reclaim stalls since the previous snapshot, the sum of the `allocstall_*`
    /// counters in `/proc/vmstat`. The pressure signal where PSI is not available (RFC-0020
    /// §6). `None` where `/proc/vmstat` is unreadable, and on the first reading, which has no
    /// previous one to subtract. See [`ReclaimStalls`].
    pub direct_reclaim_stalls: Option<u64>,
    /// The OS's own pressure level: macOS dispatch source, Windows notification.
    pub os_level: Option<PressureLevel>,
    /// When the snapshot was taken.
    pub at: Instant,
}

/// The cgroup's memory controller as of one snapshot (RFC-0020 §3, §5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CgroupMem {
    /// `memory.max` (v1: `memory.limit_in_bytes`); `None` when unlimited.
    pub max: Option<Bytes>,
    /// `memory.high`; `None` when unlimited or on v1.
    pub high: Option<Bytes>,
    /// `memory.current` (v1: `memory.usage_in_bytes`).
    pub current: Bytes,
    /// `high` events in `memory.events` since the previous snapshot.
    pub high_events: u64,
    /// `max` events in `memory.events` since the previous snapshot.
    pub max_events: u64,
}

/// Memory pressure stall information, `avg10` in percent (RFC-0020 §3, §6).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Psi {
    /// `some avg10`: share of time at least one task stalled on memory.
    pub some_avg10: f32,
    /// `full avg10`: share of time every non-idle task stalled on memory.
    pub full_avg10: f32,
}

/// A host range reserved through the OS adapter (RFC-0020 §3, supporting types).
///
/// Holds a raw pointer, so it is neither `Send` nor `Sync`; only this module hands it across
/// threads, under the adapter's own invariants.
#[derive(Debug, PartialEq, Eq)]
pub struct HostRange {
    /// The first byte of the range.
    pub base: *mut u8,
    /// The length.
    pub len: Bytes,
    /// The NUMA node the pages are bound to, if any.
    pub numa: Option<u16>,
    /// The OS's handle for the reservation, where it has one (0 otherwise).
    pub os_handle: u64,
}

// ---- commit_headroom -----------------------------------------------------------------------

/// Windows commit accounting as one reading: the commit limit and the commit charge
/// (`GlobalMemoryStatusEx`, TIER-E2-T03). Injected by the caller; this module makes no Win32
/// call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitCharge {
    /// The system commit limit.
    pub limit: Bytes,
    /// The current commit charge.
    pub charge: Bytes,
}

/// [`MemSnapshot::commit_headroom`] for this OS: `limit − charge` on Windows, and `None` on
/// every other OS whatever `reading` holds (RFC-0020 §3, §6). On Windows it is also `None`
/// when no reading is supplied: this task adds no Win32 call, so until the Windows adapter
/// (TIER-E2-T03) passes a reading, the gate's commit term is skipped rather than given an
/// invented value.
pub fn commit_headroom(reading: Option<CommitCharge>) -> Option<Bytes> {
    commit_headroom_on(cfg!(windows), reading)
}

fn commit_headroom_on(windows: bool, reading: Option<CommitCharge>) -> Option<Bytes> {
    if !windows {
        return None;
    }
    reading.map(|c| Bytes(c.limit.0.saturating_sub(c.charge.0)))
}

// ---- direct_reclaim_stalls -----------------------------------------------------------------

/// Where Linux publishes the reclaim counters.
pub const PROC_VMSTAT: &str = "/proc/vmstat";

/// The sum of every `allocstall_*` counter in a `/proc/vmstat` text (the per-zone counters of
/// Linux 4.x and later; older kernels' single `allocstall` counter also counts). `None` when
/// the text has no such counter or one fails to parse.
pub fn allocstall_total(vmstat: &str) -> Option<u64> {
    let mut total: Option<u64> = None;
    for line in vmstat.lines() {
        let mut fields = line.split_whitespace();
        let (Some(key), Some(value)) = (fields.next(), fields.next()) else {
            continue;
        };
        if key == "allocstall" || key.starts_with("allocstall_") {
            let v: u64 = value.parse().ok()?;
            total = Some(total.unwrap_or(0).checked_add(v)?);
        }
    }
    total
}

/// Turns successive `/proc/vmstat` readings into [`MemSnapshot::direct_reclaim_stalls`]: the
/// stalls since the previous reading (RFC-0020 §3, §6).
///
/// The result is `None` when the current text is unreadable or has no `allocstall` counter,
/// when there is no previous readable reading to subtract, and when the total went down (a
/// counter reset). `None` tells the gate the signal is missing; it never stands for zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReclaimStalls {
    previous: Option<u64>,
}

impl ReclaimStalls {
    /// A tracker with no baseline; the first reading returns `None`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds one reading (`None` when the file could not be read) and returns the stalls since
    /// the previous readable one.
    pub fn observe(&mut self, vmstat: Option<&str>) -> Option<u64> {
        let current = vmstat.and_then(allocstall_total);
        let delta = match (self.previous, current) {
            (Some(prev), Some(cur)) => cur.checked_sub(prev),
            _ => None,
        };
        self.previous = current;
        delta
    }

    /// Reads `path` (normally [`PROC_VMSTAT`]) and feeds it to [`ReclaimStalls::observe`].
    pub fn sample(&mut self, path: &Path) -> Option<u64> {
        let text = std::fs::read_to_string(path).ok();
        self.observe(text.as_deref())
    }
}

// ---- numa_available ------------------------------------------------------------------------

/// Where Linux publishes per-node memory.
pub const SYS_NODE_DIR: &str = "/sys/devices/system/node";

/// One node's available estimate from its `node<n>/meminfo` text, with `numa_reserve`
/// subtracted once (RFC-0020 §5): `MemFree + Active(file) + Inactive(file) + SReclaimable −
/// numa_reserve`, floored at zero. `None` when any of the four fields is missing or does not
/// parse, so a partial file never yields a low-looking estimate.
pub fn numa_available(node_meminfo: &str, numa_reserve: Bytes) -> Option<Bytes> {
    let mut fields = [None::<u64>; 4];
    const KEYS: [&str; 4] = [
        "MemFree:",
        "Active(file):",
        "Inactive(file):",
        "SReclaimable:",
    ];
    for line in node_meminfo.lines() {
        // "Node 0 MemFree:   123456 kB"
        let mut it = line.split_whitespace();
        let (Some("Node"), Some(_node), Some(key), Some(value)) =
            (it.next(), it.next(), it.next(), it.next())
        else {
            continue;
        };
        if let Some(i) = KEYS.iter().position(|k| *k == key) {
            let kib: u64 = value.parse().ok()?;
            let unit = it.next();
            if unit != Some("kB") {
                return None;
            }
            fields[i] = Some(kib.checked_mul(1024)?);
        }
    }
    let mut sum: u64 = 0;
    for f in fields {
        sum = sum.checked_add(f?)?;
    }
    Some(Bytes(sum.saturating_sub(numa_reserve.0)))
}

/// [`numa_available`] for every node under `dir` (normally [`SYS_NODE_DIR`]), sorted by node.
/// A node whose `meminfo` is unreadable or incomplete is left out; a missing directory (no
/// NUMA information, or not Linux) gives an empty list. A node absent from the list has no
/// estimate, not a zero one: the pinning gate (RFC-0020 §6, TIER-E2) must handle a missing
/// node explicitly.
pub fn read_numa_available(dir: &Path, numa_reserve: Bytes) -> SmallVec<[(u16, Bytes); 4]> {
    let mut out: SmallVec<[(u16, Bytes); 4]> = SmallVec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(node) = name
            .to_str()
            .and_then(|n| n.strip_prefix("node"))
            .and_then(|n| n.parse::<u16>().ok())
        else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(entry.path().join("meminfo")) else {
            continue;
        };
        if let Some(avail) = numa_available(&text, numa_reserve) {
            out.push((node, avail));
        }
    }
    out.sort_unstable_by_key(|(n, _)| *n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn commit_headroom_is_none_off_windows() {
        let reading = Some(CommitCharge {
            limit: Bytes(64 * GIB),
            charge: Bytes(10 * GIB),
        });
        assert_eq!(commit_headroom_on(false, reading), None);
        assert_eq!(commit_headroom_on(false, None), None);
    }

    #[test]
    fn commit_headroom_on_windows_is_limit_minus_charge() {
        let reading = CommitCharge {
            limit: Bytes(64 * GIB),
            charge: Bytes(10 * GIB),
        };
        assert_eq!(
            commit_headroom_on(true, Some(reading)),
            Some(Bytes(54 * GIB))
        );
        let over = CommitCharge {
            limit: Bytes(GIB),
            charge: Bytes(2 * GIB),
        };
        assert_eq!(commit_headroom_on(true, Some(over)), Some(Bytes(0)));
        assert_eq!(commit_headroom_on(true, None), None);
    }

    #[test]
    fn commit_headroom_follows_the_build_target() {
        let reading = Some(CommitCharge {
            limit: Bytes(2 * GIB),
            charge: Bytes(GIB),
        });
        let expected = if cfg!(windows) {
            Some(Bytes(GIB))
        } else {
            None
        };
        assert_eq!(commit_headroom(reading), expected);
    }

    const VMSTAT: &str = "nr_free_pages 100\nallocstall_dma 1\nallocstall_dma32 2\n\
                          allocstall_normal 30\nallocstall_movable 400\npgscan_kswapd 9\n";

    #[test]
    fn allocstall_total_sums_every_allocstall_counter() {
        assert_eq!(allocstall_total(VMSTAT), Some(433));
        assert_eq!(allocstall_total("allocstall 7\n"), Some(7));
        assert_eq!(allocstall_total("nr_free_pages 100\n"), None);
        assert_eq!(allocstall_total(""), None);
        assert_eq!(allocstall_total("allocstall_normal x\n"), None);
    }

    #[test]
    fn reclaim_stalls_is_a_delta_between_readings() {
        let mut r = ReclaimStalls::new();
        assert_eq!(r.observe(Some("allocstall_normal 10\n")), None);
        assert_eq!(r.observe(Some("allocstall_normal 15\n")), Some(5));
        assert_eq!(r.observe(Some("allocstall_normal 15\n")), Some(0));
    }

    #[test]
    fn reclaim_stalls_is_none_where_vmstat_is_unreadable() {
        let mut r = ReclaimStalls::new();
        assert_eq!(r.observe(Some("allocstall_normal 10\n")), None);
        assert_eq!(r.observe(None), None);
        // The unreadable reading drops the baseline: the next delta waits for a fresh one.
        assert_eq!(r.observe(Some("allocstall_normal 20\n")), None);
        assert_eq!(r.observe(Some("allocstall_normal 21\n")), Some(1));
        // A reset counter is not a negative delta.
        assert_eq!(r.observe(Some("allocstall_normal 3\n")), None);
    }

    #[test]
    fn reclaim_stalls_sample_of_a_missing_file_is_none() {
        let mut r = ReclaimStalls::new();
        let missing = Path::new("/nonexistent/syntropy-tier-test/vmstat");
        assert_eq!(r.sample(missing), None);
        assert_eq!(r.sample(missing), None);
    }

    const NODE0: &str = "Node 0 MemTotal:       65536000 kB\n\
                         Node 0 MemFree:         1000000 kB\n\
                         Node 0 MemUsed:        64536000 kB\n\
                         Node 0 Active(file):    2000000 kB\n\
                         Node 0 Inactive(file):  3000000 kB\n\
                         Node 0 SReclaimable:     500000 kB\n";

    #[test]
    fn numa_available_subtracts_the_reserve_once() {
        let sum = (1_000_000 + 2_000_000 + 3_000_000 + 500_000) * 1024;
        assert_eq!(numa_available(NODE0, Bytes(0)), Some(Bytes(sum)));
        assert_eq!(numa_available(NODE0, Bytes(GIB)), Some(Bytes(sum - GIB)));
        assert_eq!(numa_available(NODE0, Bytes(u64::MAX)), Some(Bytes(0)));
    }

    #[test]
    fn numa_available_needs_all_four_fields() {
        let partial: String = NODE0
            .lines()
            .filter(|l| !l.contains("SReclaimable"))
            .map(|l| format!("{l}\n"))
            .collect();
        assert_eq!(numa_available(&partial, Bytes(0)), None);
        assert_eq!(numa_available("", Bytes(0)), None);
    }

    fn node_meminfo(node: u16, free_kib: u64) -> String {
        format!(
            "Node {node} MemFree: {free_kib} kB\nNode {node} Active(file): 1 kB\n\
             Node {node} Inactive(file): 1 kB\nNode {node} SReclaimable: 1 kB\n"
        )
    }

    #[test]
    fn read_numa_available_reads_each_node_in_numeric_order() {
        let dir = std::env::temp_dir().join(format!("syntropy-tier-numa-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Created out of order and named so that name order (node1, node10, node2) differs
        // from numeric order: the result must not depend on readdir order.
        for n in [10u16, 2, 1] {
            let node = dir.join(format!("node{n}"));
            std::fs::create_dir_all(&node).unwrap();
            std::fs::write(node.join("meminfo"), node_meminfo(n, u64::from(n) * 1000)).unwrap();
        }
        std::fs::create_dir_all(dir.join("node3")).unwrap();
        std::fs::write(dir.join("node3").join("meminfo"), "").unwrap();
        std::fs::create_dir_all(dir.join("power")).unwrap();
        let got = read_numa_available(&dir, Bytes(0));
        std::fs::remove_dir_all(&dir).unwrap();
        let want = |n: u64| Bytes((n * 1000 + 3) * 1024);
        // node3's meminfo is incomplete, so it is left out rather than reported as zero.
        assert_eq!(
            got.as_slice(),
            &[(1, want(1)), (2, want(2)), (10, want(10))]
        );
    }

    #[test]
    fn numa_available_rejects_a_unit_other_than_kb() {
        assert_eq!(
            numa_available(&node_meminfo(0, 5), Bytes(0)),
            Some(Bytes(8 * 1024))
        );
        let mb = node_meminfo(0, 5).replace("MemFree: 5 kB", "MemFree: 5 MB");
        assert_eq!(numa_available(&mb, Bytes(0)), None);
        let bare = node_meminfo(0, 5).replace("MemFree: 5 kB", "MemFree: 5");
        assert_eq!(numa_available(&bare, Bytes(0)), None);
    }

    #[test]
    // `MemSnapshot::at` is a `std::time::Instant` (RFC-0020 §3), and `Instant::now` is the only
    // way to make one. A listed exception to the determinism lints (`xtask/src/determinism.rs`).
    #[allow(
        clippy::disallowed_methods,
        reason = "builds the `Instant` that `MemSnapshot::at` holds"
    )]
    fn mem_snapshot_carries_the_gate_fields() {
        let s = MemSnapshot {
            total: Bytes(64 * GIB),
            available: Bytes(32 * GIB),
            numa_free: SmallVec::new(),
            numa_available: SmallVec::from_slice(&[(0, Bytes(8 * GIB))]),
            cgroup: None,
            commit_headroom: commit_headroom(None),
            own_locked: Bytes(0),
            own_registered: Bytes(0),
            psi: None,
            direct_reclaim_stalls: ReclaimStalls::new().observe(None),
            os_level: None,
            at: Instant::now(),
        };
        assert_eq!(s.commit_headroom, None);
        assert_eq!(s.direct_reclaim_stalls, None);
        assert_eq!(s.numa_available[0], (0, Bytes(8 * GIB)));
    }

    #[test]
    fn read_numa_available_of_a_missing_dir_is_empty() {
        let missing = Path::new("/nonexistent/syntropy-tier-test/node");
        assert!(read_numa_available(missing, Bytes(GIB)).is_empty());
    }
}
