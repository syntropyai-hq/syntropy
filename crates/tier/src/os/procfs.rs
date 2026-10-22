// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Linux text interfaces the snapshot and the limits read (RFC-0020 §3, §5, Linux):
//! `/proc/meminfo`, `/sys/devices/system/node/node<n>/meminfo`, the cgroup v2 and v1 memory
//! files, `/proc/pressure/memory`, `/proc/vmstat`, `/proc/self/status` (`VmLck`, `VmPin`,
//! `CapEff`), `/proc/self/limits`, the THP policy and `/proc/self/numa_maps`.
//!
//! Every reader takes the filesystem root as a path, so the same code reads `/` on a live host
//! and a fixture tree in tests, on any OS. Nothing here is `unsafe`; it uses `std::fs` only.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use smallvec::SmallVec;
use syntropy_core::Bytes;
use syntropy_core::budget::{self, NodeMem};

use super::{CgroupMem, MemSnapshot, PinLimits, Psi, ReclaimStalls};

/// `CAP_IPC_LOCK` is capability 14.
const CAP_IPC_LOCK: u64 = 1 << 14;

// ---- Parsers -------------------------------------------------------------------------------

/// A `Key:   123 kB` field of a meminfo or status text, in bytes. `None` when absent, not in
/// `kB`, or not a number. A per-node meminfo's `Node <n>` prefix is skipped.
pub fn kb_field(text: &str, key: &str) -> Option<Bytes> {
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let mut first = it.next();
        if first == Some("Node") {
            it.next();
            first = it.next();
        }
        if first.and_then(|f| f.strip_suffix(':')) != Some(key) {
            continue;
        }
        let value: u64 = it.next()?.parse().ok()?;
        return match it.next() {
            Some("kB") => value.checked_mul(1024).map(Bytes),
            _ => None,
        };
    }
    None
}

/// `CapEff` from a `/proc/<pid>/status` text: whether `CAP_IPC_LOCK` is effective.
pub fn cap_ipc_lock(status: &str) -> Option<bool> {
    let hex = status
        .lines()
        .find_map(|l| l.strip_prefix("CapEff:"))?
        .trim();
    let caps = u64::from_str_radix(hex, 16).ok()?;
    Some(caps & CAP_IPC_LOCK != 0)
}

/// The soft `Max locked memory` limit from a `/proc/<pid>/limits` text. `Some(None)` when
/// unlimited; `None` when the line is missing or malformed.
pub fn memlock_soft(limits: &str) -> Option<Option<Bytes>> {
    let rest = limits
        .lines()
        .find_map(|l| l.strip_prefix("Max locked memory"))?;
    match rest.split_whitespace().next()? {
        "unlimited" => Some(None),
        n => n.parse().ok().map(|b| Some(Bytes(b))),
    }
}

/// `some avg10` and `full avg10` from a `/proc/pressure/memory` text.
pub fn psi(text: &str) -> Option<Psi> {
    let avg10 = |kind: &str| -> Option<f32> {
        let line = text.lines().find(|l| l.starts_with(kind))?;
        line.split_whitespace()
            .find_map(|f| f.strip_prefix("avg10="))?
            .parse()
            .ok()
    };
    Some(Psi {
        some_avg10: avg10("some ")?,
        full_avg10: avg10("full ")?,
    })
}

/// A cgroup limit file: `max` (v2) or a value of 2^62 or more (v1's "no limit", page-rounded
/// `LONG_MAX`) is `Some(None)`; a malformed file is `None`.
pub fn cgroup_limit(text: &str) -> Option<Option<Bytes>> {
    match text.trim() {
        "max" => Some(None),
        t => {
            let v: u64 = t.parse().ok()?;
            Some((v < 1 << 62).then_some(Bytes(v)))
        }
    }
}

/// The `high` and `max` counters of a cgroup v2 `memory.events` text.
pub fn memory_events(text: &str) -> (u64, u64) {
    let get = |key: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(key)?.strip_prefix(' '))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    };
    (get("high"), get("max"))
}

/// The transparent huge page policy (`/sys/kernel/mm/transparent_hugepage/enabled`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThpMode {
    /// `[always]`.
    Always,
    /// `[madvise]`.
    Madvise,
    /// `[never]`, or THP absent.
    Never,
}

impl ThpMode {
    /// Whether `MADV_HUGEPAGE` takes effect without privilege.
    pub fn advisable(self) -> bool {
        matches!(self, ThpMode::Always | ThpMode::Madvise)
    }
}

/// The bracketed choice of a THP `enabled` text.
pub fn thp_mode(text: &str) -> ThpMode {
    let chosen = text
        .split_whitespace()
        .find_map(|w| w.strip_prefix('[')?.strip_suffix(']'));
    match chosen {
        Some("always") => ThpMode::Always,
        Some("madvise") => ThpMode::Madvise,
        _ => ThpMode::Never,
    }
}

/// One mapping in `/proc/<pid>/numa_maps`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NumaMapsEntry {
    /// The mapping's start address.
    pub start: u64,
    /// The memory policy, such as `bind:0` or `default`.
    pub policy: String,
    /// Resident pages per node, from the `N<node>=<pages>` fields, sorted by node.
    pub pages: SmallVec<[(u16, u64); 4]>,
}

/// The `numa_maps` entry of the mapping that holds `addr`: the one with the greatest start at
/// or below it. Adjacent mappings with equal flags and policy may have merged, so the entry's
/// start can lie below `addr`; the caller checks the mapping's end in `/proc/<pid>/maps` when
/// that matters.
pub fn numa_maps_entry(text: &str, addr: u64) -> Option<NumaMapsEntry> {
    let mut best: Option<NumaMapsEntry> = None;
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let Some(start) = fields.next().and_then(|s| u64::from_str_radix(s, 16).ok()) else {
            continue;
        };
        if start > addr || best.as_ref().is_some_and(|b| b.start > start) {
            continue;
        }
        let Some(policy) = fields.next() else {
            continue;
        };
        let mut pages: SmallVec<[(u16, u64); 4]> = fields
            .filter_map(|f| {
                let (node, count) = f.strip_prefix('N')?.split_once('=')?;
                Some((node.parse().ok()?, count.parse().ok()?))
            })
            .collect();
        pages.sort_unstable();
        best = Some(NumaMapsEntry {
            start,
            policy: policy.to_owned(),
            pages,
        });
    }
    best
}

/// The end of the `/proc/<pid>/maps` mapping that starts at `start`.
pub fn maps_end(text: &str, start: u64) -> Option<u64> {
    text.lines().find_map(|l| {
        let range = l.split_whitespace().next()?;
        let (s, e) = range.split_once('-')?;
        (u64::from_str_radix(s, 16).ok()? == start)
            .then(|| u64::from_str_radix(e, 16).ok())
            .flatten()
    })
}

// ---- Readers -------------------------------------------------------------------------------

fn read(root: &Path, rel: &str) -> Option<String> {
    std::fs::read_to_string(root.join(rel)).ok()
}

/// This process's cgroup memory controller: v2 when the unified hierarchy has a memory
/// controller for the process, v1's `memory` controller otherwise. Of the process's cgroup and
/// its ancestors, the level with the least headroom, `min(max, high) − current`, is reported,
/// because that level binds first. `None` outside a cgroup with a memory limit, and when the
/// cgroup path has a `.` or `..` component (a process outside its cgroup namespace's root
/// sees such paths), so the walk never leaves the cgroup mount.
pub fn cgroup_mem(root: &Path) -> Option<CgroupMem> {
    cgroup_level(root).map(|(_, c)| c)
}

/// [`cgroup_mem`] with the directory of the level it reports, which keys the event baseline.
fn cgroup_level(root: &Path) -> Option<(PathBuf, CgroupMem)> {
    let membership = read(root, "proc/self/cgroup")?;
    let mut v1_path = None;
    let mut v2_path = None;
    for line in membership.lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if id == "0" && controllers.is_empty() {
            v2_path = Some(path.to_owned());
        } else if controllers.split(',').any(|c| c == "memory") {
            v1_path = Some(path.to_owned());
        }
    }
    let v2 = v2_path.and_then(|p| tightest(root, "sys/fs/cgroup", &p, true));
    v2.or_else(|| v1_path.and_then(|p| tightest(root, "sys/fs/cgroup/memory", &p, false)))
}

fn tightest(root: &Path, mount: &str, path: &str, v2: bool) -> Option<(PathBuf, CgroupMem)> {
    let (max_file, cur_file) = if v2 {
        ("memory.max", "memory.current")
    } else {
        ("memory.limit_in_bytes", "memory.usage_in_bytes")
    };
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    if parts.iter().any(|p| *p == ".." || *p == ".") {
        return None;
    }
    let mut best: Option<(u64, PathBuf, CgroupMem)> = None;
    for depth in (0..=parts.len()).rev() {
        let dir: PathBuf = std::iter::once(mount)
            .chain(parts[..depth].iter().copied())
            .collect();
        let dir = root.join(dir);
        let Some(max) = std::fs::read_to_string(dir.join(max_file))
            .ok()
            .and_then(|t| cgroup_limit(&t))
        else {
            continue;
        };
        let high = if v2 {
            std::fs::read_to_string(dir.join("memory.high"))
                .ok()
                .and_then(|t| cgroup_limit(&t))
                .flatten()
        } else {
            None
        };
        let Some(current) = std::fs::read_to_string(dir.join(cur_file))
            .ok()
            .and_then(|t| t.trim().parse::<u64>().ok())
        else {
            continue;
        };
        let bound = match (max, high) {
            (Some(m), Some(h)) => m.min(h),
            (Some(b), None) | (None, Some(b)) => b,
            (None, None) => continue,
        };
        let headroom = bound.0.saturating_sub(current);
        if best.as_ref().is_none_or(|(h, _, _)| headroom < *h) {
            let (high_events, max_events) = if v2 {
                std::fs::read_to_string(dir.join("memory.events"))
                    .map(|t| memory_events(&t))
                    .unwrap_or((0, 0))
            } else {
                (0, 0)
            };
            best = Some((
                headroom,
                dir.clone(),
                CgroupMem {
                    max,
                    high,
                    current: Bytes(current),
                    high_events,
                    max_events,
                },
            ));
        }
    }
    best.map(|(_, dir, c)| (dir, c))
}

/// Per-node readings, sorted by node.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NumaNodes {
    /// `MemFree` per node.
    pub free: SmallVec<[(u16, Bytes); 4]>,
    /// The available estimate per node, `numa_reserve` subtracted once.
    pub available: SmallVec<[(u16, Bytes); 4]>,
}

/// Per-node `MemFree` and the per-node available estimate, from each node's meminfo under the
/// root's `sys/devices/system/node`.
pub fn numa_nodes(root: &Path, numa_reserve: Bytes) -> NumaNodes {
    let mut free: SmallVec<[(u16, Bytes); 4]> = SmallVec::new();
    let mut avail: SmallVec<[(u16, Bytes); 4]> = SmallVec::new();
    let Ok(entries) = std::fs::read_dir(root.join("sys/devices/system/node")) else {
        return NumaNodes::default();
    };
    for entry in entries.flatten() {
        let Some(node) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.strip_prefix("node"))
            .and_then(|n| n.parse::<u16>().ok())
        else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(entry.path().join("meminfo")) else {
            continue;
        };
        let field = |k| kb_field(&text, k);
        if let Some(f) = field("MemFree") {
            free.push((node, f));
        }
        if let (Some(f), Some(a), Some(i), Some(s)) = (
            field("MemFree"),
            field("Active(file)"),
            field("Inactive(file)"),
            field("SReclaimable"),
        ) {
            let node_mem = NodeMem {
                mem_free: f,
                active_file: a,
                inactive_file: i,
                sreclaimable: s,
            };
            if let Some(est) = budget::numa_available(&node_mem, numa_reserve) {
                avail.push((node, est));
            }
        }
    }
    free.sort_unstable_by_key(|(n, _)| *n);
    avail.sort_unstable_by_key(|(n, _)| *n);
    NumaNodes {
        free,
        available: avail,
    }
}

/// `VmLck` and `VmPin` of this process, 0 where absent.
pub fn own_lock(root: &Path) -> (Bytes, Bytes) {
    let status = read(root, "proc/self/status").unwrap_or_default();
    (
        kb_field(&status, "VmLck").unwrap_or_default(),
        kb_field(&status, "VmPin").unwrap_or_default(),
    )
}

/// The pin limits of this process (RFC-0020 §3, §5 Linux).
pub fn pin_limits(root: &Path) -> PinLimits {
    let status = read(root, "proc/self/status").unwrap_or_default();
    let cap = cap_ipc_lock(&status).unwrap_or(false);
    // A missing or malformed limits file counts as the tightest reading: no lock budget.
    let memlock = read(root, "proc/self/limits")
        .and_then(|t| memlock_soft(&t))
        .unwrap_or(Some(Bytes(0)));
    let vmpin = kb_field(&status, "VmPin").unwrap_or_default();
    let io_fixed_max = match memlock {
        Some(limit) if !cap => Some(Bytes(limit.0.saturating_sub(vmpin.0))),
        _ => None,
    };
    PinLimits {
        memlock,
        cap_ipc_lock: cap,
        io_fixed_max,
        ..PinLimits::default()
    }
}

/// The THP policy of the root's kernel.
pub fn thp(root: &Path) -> ThpMode {
    read(root, "sys/kernel/mm/transparent_hugepage/enabled")
        .map(|t| thp_mode(&t))
        .unwrap_or(ThpMode::Never)
}

/// Builds [`MemSnapshot`]s from a root directory, keeping what the deltas need: the
/// `allocstall` baseline and the cgroup event counters of the previous reading.
#[derive(Debug)]
pub struct ProcReader {
    root: PathBuf,
    numa_reserve: Bytes,
    state: Mutex<ReaderState>,
}

#[derive(Debug, Default)]
struct ReaderState {
    stalls: ReclaimStalls,
    /// The raw `high` and `max` counters of the level reported last, keyed by its directory:
    /// a delta is taken only against the same level.
    events: Option<(PathBuf, u64, u64)>,
}

impl ProcReader {
    /// A reader over `root` (`/` on a live host). The cgroup event deltas and the stall delta
    /// count from this first reading.
    pub fn new(root: impl Into<PathBuf>, numa_reserve: Bytes) -> Self {
        let reader = Self {
            root: root.into(),
            numa_reserve,
            state: Mutex::new(ReaderState::default()),
        };
        let _baseline = reader.snapshot();
        reader
    }

    /// The root this reader reads.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// One snapshot. `own_locked` is `VmLck`; `own_registered` is `VmPin`, which counts pages
    /// pinned for device or I/O access (provider registrations where the driver pins through
    /// the kernel's pinning interface, io_uring fixed buffers, RDMA memory regions).
    pub fn snapshot(&self) -> MemSnapshot {
        let root = &self.root;
        let meminfo = read(root, "proc/meminfo").unwrap_or_default();
        let nodes = numa_nodes(root, self.numa_reserve);
        let (own_locked, own_registered) = own_lock(root);
        let vmstat = read(root, "proc/vmstat");
        let level = cgroup_level(root);

        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let direct_reclaim_stalls = state.stalls.observe(vmstat.as_deref());
        let cgroup = level.map(|(dir, mut c)| {
            let now = (c.high_events, c.max_events);
            let (h0, m0) = match &state.events {
                Some((d, h, m)) if *d == dir => (*h, *m),
                // A new level, or the first reading: count from here.
                _ => now,
            };
            state.events = Some((dir, now.0, now.1));
            c.high_events = now.0.saturating_sub(h0);
            c.max_events = now.1.saturating_sub(m0);
            c
        });
        if cgroup.is_none() {
            state.events = None;
        }
        drop(state);

        MemSnapshot {
            total: kb_field(&meminfo, "MemTotal").unwrap_or_default(),
            available: kb_field(&meminfo, "MemAvailable").unwrap_or_default(),
            numa_free: nodes.free,
            numa_available: nodes.available,
            cgroup,
            commit_headroom: super::commit_headroom(None),
            own_locked,
            own_registered,
            psi: read(root, "proc/pressure/memory").and_then(|t| psi(&t)),
            direct_reclaim_stalls,
            os_level: None,
            at: Instant::now(),
        }
    }

    /// The pin limits as of now.
    pub fn limits(&self) -> PinLimits {
        pin_limits(&self.root)
    }

    /// The THP policy.
    pub fn thp(&self) -> ThpMode {
        thp(&self.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn kb_fields() {
        let t = "MemTotal:       16384 kB\nMemAvailable:    8192 kB\nOdd: 5 MB\n";
        assert_eq!(kb_field(t, "MemTotal"), Some(Bytes(16 * MIB)));
        assert_eq!(kb_field(t, "MemAvailable"), Some(Bytes(8 * MIB)));
        assert_eq!(kb_field(t, "Odd"), None);
        assert_eq!(kb_field(t, "Missing"), None);
        let node = "Node 1 MemFree:   1024 kB\n";
        assert_eq!(kb_field(node, "MemFree"), Some(Bytes(MIB)));
    }

    #[test]
    fn status_fields() {
        let s = "VmLck:\t     64 kB\nVmPin:\t      0 kB\nCapEff:\t0000000000004000\n";
        assert_eq!(kb_field(s, "VmLck"), Some(Bytes(64 * 1024)));
        assert_eq!(kb_field(s, "VmPin"), Some(Bytes(0)));
        assert_eq!(cap_ipc_lock(s), Some(true));
        assert_eq!(cap_ipc_lock("CapEff:\t0000000000000000\n"), Some(false));
        assert_eq!(cap_ipc_lock("CapEff:\t000001ffffffffff\n"), Some(true));
        assert_eq!(cap_ipc_lock("nothing\n"), None);
    }

    #[test]
    fn memlock_limits() {
        let finite = "Limit                     Soft Limit           Hard Limit           Units\n\
                      Max locked memory         8388608              8388608              bytes\n";
        assert_eq!(memlock_soft(finite), Some(Some(Bytes(8 * MIB))));
        let unlimited =
            "Max locked memory         unlimited            unlimited            bytes\n";
        assert_eq!(memlock_soft(unlimited), Some(None));
        assert_eq!(memlock_soft("Max open files 1024 1024 files\n"), None);
    }

    #[test]
    fn psi_text() {
        let t = "some avg10=1.50 avg60=0.20 avg300=0.00 total=123\n\
                 full avg10=0.25 avg60=0.00 avg300=0.00 total=45\n";
        assert_eq!(
            psi(t),
            Some(Psi {
                some_avg10: 1.5,
                full_avg10: 0.25
            })
        );
        assert_eq!(psi("some avg10=1.0\n"), None);
    }

    #[test]
    fn cgroup_limits_and_events() {
        assert_eq!(cgroup_limit("max\n"), Some(None));
        assert_eq!(cgroup_limit("8589934592\n"), Some(Some(Bytes(8 << 30))));
        assert_eq!(cgroup_limit("9223372036854771712\n"), Some(None));
        assert_eq!(cgroup_limit("x"), None);
        assert_eq!(
            memory_events("low 0\nhigh 7\nmax 3\noom 0\noom_kill 0\n"),
            (7, 3)
        );
    }

    #[test]
    fn thp_modes() {
        assert_eq!(thp_mode("always [madvise] never\n"), ThpMode::Madvise);
        assert_eq!(thp_mode("[always] madvise never\n"), ThpMode::Always);
        assert_eq!(thp_mode("always madvise [never]\n"), ThpMode::Never);
        assert!(ThpMode::Madvise.advisable());
        assert!(!ThpMode::Never.advisable());
    }

    #[test]
    fn numa_maps_entries() {
        let t = "55d0c0a00000 default file=/usr/bin/x mapped=10 N0=10 kernelpagesize_kB=4\n\
                 7f0000000000 bind:0 anon=512 dirty=512 N0=512 kernelpagesize_kB=4\n\
                 7f0000400000 interleave:0-1 anon=4 dirty=4 N1=2 N0=2 kernelpagesize_kB=4\n";
        let e = numa_maps_entry(t, 0x7f00_0010_0000).unwrap();
        assert_eq!(e.start, 0x7f00_0000_0000);
        assert_eq!(e.policy, "bind:0");
        assert_eq!(e.pages.as_slice(), &[(0, 512)]);
        let e = numa_maps_entry(t, 0x7f00_0040_0000).unwrap();
        assert_eq!(e.pages.as_slice(), &[(0, 2), (1, 2)]);
        assert_eq!(numa_maps_entry(t, 0x1000), None);
        let maps = "7f0000000000-7f0000200000 rw-p 00000000 00:00 0\n";
        assert_eq!(maps_end(maps, 0x7f00_0000_0000), Some(0x7f00_0020_0000));
        assert_eq!(maps_end(maps, 0x1), None);
    }
}
