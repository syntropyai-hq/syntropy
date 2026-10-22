// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host memory as the OS reports it (RFC-0020 §1, §3, §4.2, §5, §6): [`MemSnapshot`], the
//! fields the §6 pinning gate reads, the [`OsMemory`] trait every OS adapter implements, and
//! the live sources the adapters share: the per-node available estimate ([`numa_available`])
//! and the direct-reclaim stall delta ([`ReclaimStalls`]).
//!
//! - [`procfs`] reads the Linux text interfaces (`/proc`, `/sys`) from any root directory, so
//!   the snapshot is tested against fixture trees on every OS. It holds no `unsafe`.
//! - `linux` is the Linux adapter (TIER-E2-T02): anonymous `mmap`, `mbind`, `madvise` and
//!   `mlock`, declared there because the workspace has no C bindings crate. It is the one
//!   place in this module with `unsafe`, under a scoped `allow`. Its constants and signatures
//!   are those of x86-64 and AArch64, the Linux targets ADR-0020 supports, so it is built for
//!   those two only; another Linux architecture has no adapter and no `os::linux` module.
//!
//! - `macos` holds the macOS sysctl reads `syntropy tune`'s tier side needs (TIER-E2-T13):
//!   `sysctlbyname` with no new value, under a scoped `allow`. Nothing in this crate writes a
//!   sysctl (RFC-0020 §9.1).
//!
//! The Windows adapter comes in TIER-E2-T03, WSL2 and the rest of macOS in TIER-E2-T04.

#[cfg(target_os = "macos")]
pub mod macos;
pub mod procfs;

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub mod linux;

use std::fmt;
use std::ops::Range;
use std::path::Path;
use std::time::Instant;

use smallvec::SmallVec;
use syntropy_core::budget::{self, NodeMem};
use syntropy_core::{Bytes, ErrorClass};

use crate::arena::PinClass;
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
/// numa_reserve`, floored at zero, by [`budget::numa_available`]. `None` when any of the four
/// fields is missing or does not parse, so a partial file never yields a low-looking estimate.
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
    let [mem_free, active_file, inactive_file, sreclaimable] = fields;
    let node = NodeMem {
        mem_free: Bytes(mem_free?),
        active_file: Bytes(active_file?),
        inactive_file: Bytes(inactive_file?),
        sreclaimable: Bytes(sreclaimable?),
    };
    budget::numa_available(&node, numa_reserve)
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

// ---- OsMemory ------------------------------------------------------------------------------

/// Everything OS-specific about host memory (RFC-0020 §3). One implementation per OS, plus a
/// fake. RFC-0020 §3's `pressure()` source comes with the yield ladder (RFC-0020 §11).
pub trait OsMemory: Send + Sync {
    /// One reading of host memory, taken before every pinning step.
    fn snapshot(&self) -> MemSnapshot;
    /// The OS's pinning limits as of now.
    fn limits(&self) -> PinLimits;
    /// Reserves `bytes` of anonymous memory, never file-backed. With `numa`, the pages are
    /// bound to that node before first touch; [`HostRange::numa`] is `None` when the bind
    /// failed twice (`TIER_NUMA_BIND`, RFC-0020 §15) and the range was left unbound.
    fn reserve(
        &self,
        bytes: Bytes,
        numa: Option<u16>,
        huge: HugePages,
    ) -> Result<HostRange, OsMemError>;
    /// Commits the pages of `sub` (byte offsets into `r`), so that later reads and writes do
    /// not fault (RFC-0020 §6). The bytes already there are kept.
    ///
    /// An adapter may fault pages in with atomic no-op read-modify-writes, which never discard
    /// a store. A caller that writes the range through [`HostRange::base`] must therefore not
    /// access the same bytes non-atomically while `commit` runs on them.
    fn commit(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError>;
    /// Locks `sub` against paging: `mlock`, `VirtualLock` or wiring. Refused with
    /// [`OsMemError::LockRefused`] when the OS limit does not cover it; the range then stays
    /// [`PinClass::Pageable`] (see [`lock_or_pageable`]).
    fn lock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError>;
    /// Undoes [`OsMemory::lock`] on `sub`. An adapter may leave locked a page that `sub`
    /// shares with a neighbouring locked range.
    fn unlock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError>;
    /// Returns the range to the OS.
    fn release(&self, r: HostRange);
}

/// Transparent huge pages for a host arena: the `tier.hugepages` key (RFC-0020 §14).
///
/// The key is not in the config schema yet (TIER-E1-T09 adds RFC-0020 §14's keys), so callers
/// pass the setting as this value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HugePages {
    /// Advise huge pages when the OS policy allows it without privilege (Linux THP `madvise`
    /// or `always`). The default.
    #[default]
    Auto,
    /// Never advise huge pages.
    Off,
}

/// The OS's pinning limits (RFC-0020 §3, supporting types). Fields that do not apply to the
/// OS are `None` or `false`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PinLimits {
    /// `RLIMIT_MEMLOCK` soft limit; `None` = unlimited.
    pub memlock: Option<Bytes>,
    /// `CAP_IPC_LOCK` is in the effective set.
    pub cap_ipc_lock: bool,
    /// What io_uring or IoRing registration may still take. Linux: this process's view,
    /// `RLIMIT_MEMLOCK − VmPin`, an upper bound on the per-user `io_fixed` the daemon divides
    /// (RFC-0020 §5); `None` = unlimited.
    pub io_fixed_max: Option<Bytes>,
    /// Windows: the hard working-set minimum is allowed.
    pub ws_hard_min_ok: bool,
    /// Windows: the measured fraction of RAM lockable for GPU DMA.
    pub lockable_fraction: Option<f32>,
    /// Windows WDDM: the host registration cap for helper GPUs.
    pub helper_gpu_cap: Option<Bytes>,
    /// WSL2: the `cudaHostAlloc` cap, which forces the register path (TIER-E2-T04).
    pub host_alloc_cap: Option<Bytes>,
    /// macOS: `vm.user_wire_limit` / `vm.global_user_wire_limit`.
    pub user_wire_limit: Option<Bytes>,
    /// macOS: `recommendedMaxWorkingSetSize`.
    pub gpu_wired_cap: Option<Bytes>,
}

/// Why a range was left [`PinClass::Pageable`] instead of locked: the structured finding that
/// `syntropy doctor` renders (RFC-0020 §4.2; the doctor framework is PROD-E9-T01).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageableReason {
    /// `RLIMIT_MEMLOCK` does not cover the range and `CAP_IPC_LOCK` is not held.
    MemlockLimit {
        /// The soft limit.
        limit: Bytes,
        /// What this process already holds locked (`VmLck`).
        locked: Bytes,
        /// The bytes the lock asked for.
        requested: Bytes,
    },
    /// The limit allowed it, but the lock call itself failed.
    LockFailed {
        /// The OS error number.
        errno: i32,
    },
}

impl PageableReason {
    /// A stable identifier for the finding, for logs, metrics and doctor's check.
    pub fn id(&self) -> &'static str {
        match self {
            PageableReason::MemlockLimit { .. } => "memlock_limit",
            PageableReason::LockFailed { .. } => "lock_failed",
        }
    }
}

impl fmt::Display for PageableReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PageableReason::MemlockLimit {
                limit,
                locked,
                requested,
            } => write!(
                f,
                "RLIMIT_MEMLOCK is {} bytes and {} are already locked, so {} more cannot be \
                 locked; the range stays pageable. Raise LimitMEMLOCK for the service, or grant \
                 CAP_IPC_LOCK",
                limit.0, locked.0, requested.0
            ),
            PageableReason::LockFailed { errno } => write!(
                f,
                "the lock call failed with OS error {errno}; the range stays pageable"
            ),
        }
    }
}

/// Whether locking `requested` more bytes is allowed: always with `CAP_IPC_LOCK` or no
/// memlock limit, otherwise only when `locked + requested` fits the limit (RFC-0020 §4.2,
/// Linux row). `locked` is the live `VmLck`.
pub fn lock_allowed(
    limits: &PinLimits,
    locked: Bytes,
    requested: Bytes,
) -> Result<(), PageableReason> {
    match limits.memlock {
        Some(limit) if !limits.cap_ipc_lock => {
            if locked
                .0
                .checked_add(requested.0)
                .is_some_and(|sum| sum <= limit.0)
            {
                Ok(())
            } else {
                Err(PageableReason::MemlockLimit {
                    limit,
                    locked,
                    requested,
                })
            }
        }
        _ => Ok(()),
    }
}

/// The outcome of asking to lock a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockOutcome {
    /// The range is locked.
    Locked,
    /// The range stays pageable, for this reason.
    Pageable(PageableReason),
}

impl LockOutcome {
    /// The pin class the range now has.
    pub fn pin_class(&self) -> PinClass {
        match self {
            LockOutcome::Locked => PinClass::Locked,
            LockOutcome::Pageable(_) => PinClass::Pageable,
        }
    }
}

/// [`OsMemory::lock`], with a refusal turned into [`LockOutcome::Pageable`]: a range that
/// cannot be locked is not an error, it stays `Pageable` and the reason is kept (RFC-0020
/// §4.2, ADR-0200 item 6). Other errors pass through.
pub fn lock_or_pageable(
    os: &dyn OsMemory,
    r: &HostRange,
    sub: Range<u64>,
) -> Result<LockOutcome, OsMemError> {
    match os.lock(r, sub) {
        Ok(()) => Ok(LockOutcome::Locked),
        Err(OsMemError::LockRefused(reason)) => Ok(LockOutcome::Pageable(reason)),
        Err(e) => Err(e),
    }
}

/// An [`OsMemory`] failure.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OsMemError {
    /// Reserving or committing memory failed (`TIER_HOST_RESERVE`, code 4, `Capacity`).
    HostReserve {
        /// The call that failed.
        op: &'static str,
        /// The OS error number.
        errno: i32,
    },
    /// The range could not be locked; it stays pageable.
    LockRefused(PageableReason),
    /// `munlock` or its equivalent failed.
    Unlock {
        /// The OS error number.
        errno: i32,
    },
    /// The range is not one this adapter reserved and still holds: a caller bug.
    UnknownRange,
    /// A sub-range or size the range cannot hold: a caller bug.
    OutOfRange {
        /// The range's length.
        len: Bytes,
        /// The offsets asked for.
        sub: Range<u64>,
    },
}

impl OsMemError {
    /// The error class (RFC-0001 §5.12).
    pub fn class(&self) -> ErrorClass {
        match self {
            OsMemError::HostReserve { .. } | OsMemError::LockRefused(_) => ErrorClass::Capacity,
            OsMemError::Unlock { .. }
            | OsMemError::UnknownRange
            | OsMemError::OutOfRange { .. } => ErrorClass::Internal,
        }
    }

    /// The RFC-0020 §15 code, where the error has one.
    pub fn tier_code(&self) -> Option<u16> {
        match self {
            OsMemError::HostReserve { .. } => Some(4),
            _ => None,
        }
    }
}

impl fmt::Display for OsMemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OsMemError::HostReserve { op, errno } => {
                write!(f, "TIER_HOST_RESERVE: {op} failed with OS error {errno}")
            }
            OsMemError::LockRefused(reason) => reason.fmt(f),
            OsMemError::Unlock { errno } => write!(f, "unlock failed with OS error {errno}"),
            OsMemError::UnknownRange => write!(f, "the range was not reserved by this adapter"),
            OsMemError::OutOfRange { len, sub } => write!(
                f,
                "offsets {}..{} are outside a range of {} bytes",
                sub.start, sub.end, len.0
            ),
        }
    }
}

impl std::error::Error for OsMemError {}

/// Checks that `sub` lies inside a range of `len` bytes and is not empty.
#[cfg_attr(
    not(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )),
    allow(dead_code)
)]
pub(crate) fn check_sub(len: Bytes, sub: &Range<u64>) -> Result<(), OsMemError> {
    if sub.start < sub.end && sub.end <= len.0 {
        Ok(())
    } else {
        Err(OsMemError::OutOfRange {
            len,
            sub: sub.clone(),
        })
    }
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

    // ---- lock decision and the pageable finding ----

    fn limits(memlock: Option<u64>, cap: bool) -> PinLimits {
        PinLimits {
            memlock: memlock.map(Bytes),
            cap_ipc_lock: cap,
            ..PinLimits::default()
        }
    }

    #[test]
    fn lock_allowed_only_within_memlock_or_with_the_capability() {
        let mib = 1 << 20;
        let eight = limits(Some(8 * mib), false);
        assert_eq!(lock_allowed(&eight, Bytes(6 * mib), Bytes(2 * mib)), Ok(()));
        assert_eq!(
            lock_allowed(&eight, Bytes(6 * mib), Bytes(2 * mib + 1)),
            Err(PageableReason::MemlockLimit {
                limit: Bytes(8 * mib),
                locked: Bytes(6 * mib),
                requested: Bytes(2 * mib + 1),
            })
        );
        assert!(lock_allowed(&eight, Bytes(u64::MAX), Bytes(1)).is_err());
        assert_eq!(
            lock_allowed(&limits(Some(0), true), Bytes(0), Bytes(GIB)),
            Ok(())
        );
        assert_eq!(
            lock_allowed(&limits(None, false), Bytes(0), Bytes(GIB)),
            Ok(())
        );
    }

    /// The finding `syntropy doctor` will render (PROD-E9-T01 builds doctor): a stable id and
    /// a sentence that names the limit, what is held, what was asked and the fix.
    #[test]
    fn the_pageable_reason_says_why_and_how_to_fix_it() {
        let r = PageableReason::MemlockLimit {
            limit: Bytes(8 << 20),
            locked: Bytes(0),
            requested: Bytes(64 << 20),
        };
        assert_eq!(r.id(), "memlock_limit");
        assert_eq!(
            r.to_string(),
            "RLIMIT_MEMLOCK is 8388608 bytes and 0 are already locked, so 67108864 more cannot \
             be locked; the range stays pageable. Raise LimitMEMLOCK for the service, or grant \
             CAP_IPC_LOCK"
        );
        let f = PageableReason::LockFailed { errno: 12 };
        assert_eq!(f.id(), "lock_failed");
        assert!(f.to_string().contains("OS error 12"));
    }

    /// A fake adapter whose `lock` follows [`lock_allowed`] over fixed limits.
    struct FakeOs {
        limits: PinLimits,
        locked: Bytes,
    }

    impl OsMemory for FakeOs {
        fn snapshot(&self) -> MemSnapshot {
            unimplemented!()
        }
        fn limits(&self) -> PinLimits {
            self.limits
        }
        fn reserve(&self, _: Bytes, _: Option<u16>, _: HugePages) -> Result<HostRange, OsMemError> {
            unimplemented!()
        }
        fn commit(&self, _: &HostRange, _: Range<u64>) -> Result<(), OsMemError> {
            Ok(())
        }
        fn lock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
            check_sub(r.len, &sub)?;
            lock_allowed(&self.limits, self.locked, Bytes(sub.end - sub.start))
                .map_err(OsMemError::LockRefused)
        }
        fn unlock(&self, _: &HostRange, _: Range<u64>) -> Result<(), OsMemError> {
            Ok(())
        }
        fn release(&self, _: HostRange) {}
    }

    fn range(len: u64) -> HostRange {
        HostRange {
            base: std::ptr::null_mut(),
            len: Bytes(len),
            numa: None,
            os_handle: 0,
        }
    }

    #[test]
    fn a_refused_lock_leaves_the_range_pageable_with_its_reason() {
        let os = FakeOs {
            limits: limits(Some(8 << 20), false),
            locked: Bytes(0),
        };
        let r = range(64 << 20);
        let out = lock_or_pageable(&os, &r, 0..64 << 20).unwrap();
        assert_eq!(out.pin_class(), PinClass::Pageable);
        assert!(matches!(
            out,
            LockOutcome::Pageable(PageableReason::MemlockLimit { .. })
        ));
        let ok = lock_or_pageable(&os, &r, 0..1 << 20).unwrap();
        assert_eq!(
            (ok, ok.pin_class()),
            (LockOutcome::Locked, PinClass::Locked)
        );
        // Other errors are not turned into a pin class.
        assert!(matches!(
            lock_or_pageable(&os, &r, 0..(64 << 20) + 1),
            Err(OsMemError::OutOfRange { .. })
        ));
    }

    #[test]
    fn errors_carry_their_class_and_code() {
        let e = OsMemError::HostReserve {
            op: "mmap",
            errno: 12,
        };
        assert_eq!((e.class(), e.tier_code()), (ErrorClass::Capacity, Some(4)));
        assert!(e.to_string().starts_with("TIER_HOST_RESERVE"));
        assert_eq!(OsMemError::UnknownRange.class(), ErrorClass::Internal);
        assert_eq!(HugePages::default(), HugePages::Auto);
    }
}
