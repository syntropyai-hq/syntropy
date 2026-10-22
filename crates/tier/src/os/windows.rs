// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Windows [`OsMemory`] (RFC-0020 §4.2, Windows row; §5, Windows):
//!
//! - `reserve`: `VirtualAllocExNuma(MEM_RESERVE, PAGE_READWRITE)` with the arena's node as
//!   `nndPreferred`, one reservation per lock range (the caller cuts them with
//!   [`lock_ranges`](crate::pin::lock_ranges), each at most `tier.pin_step`). A reservation
//!   takes no commit charge.
//! - `commit`: `VirtualAlloc(MEM_COMMIT)` on the pin step's pages. This is where the commit
//!   charge is taken, so the caller commits only when RFC-0020 §6's gate allows it
//!   ([`commit_if`](super::commit_if)).
//! - `lock`: raises the hard working-set minimum with `SetProcessWorkingSetSizeEx(min, max,
//!   QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)` to cover every byte the
//!   process holds locked (counted across adapters, since the minimum is per process), then
//!   `VirtualLock`s the range.
//! - `snapshot`: `GlobalMemoryStatusEx` (physical memory, and the commit limit and charge as
//!   [`MemSnapshot::commit_headroom`]), the per-node available memory, the job object's memory
//!   limits as [`MemSnapshot::cgroup`], and the low-memory resource notification as
//!   [`MemSnapshot::os_level`].
//!
//! The workspace has no Windows bindings crate, so the kernel32 functions and structures are
//! declared here, with layout checks in the tests. Every range goes through the shared
//! registry, as on Linux: calls act only on a live range with the same base, length and
//! process-wide id, and hold the registry lock across the OS call.

#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ops::Range;
use std::sync::{Mutex, MutexGuard};

use syntropy_core::Bytes;

use super::procfs::NumaNodes;
use super::registry::{Registry, Round};
use super::{
    CgroupMem, CommitCharge, HostRange, HugePages, MemSnapshot, OsMemError, OsMemory,
    PageableReason, PinLimits, commit_headroom,
};
use crate::pressure::PressureLevel;

/// The page size on x86-64 and AArch64 Windows.
const PAGE: u64 = 4096;

/// RFC-0020 §5's WDDM cap on host registration for every GPU after the first: 8 GiB.
const HELPER_GPU_CAP: Bytes = Bytes(8 << 30);

pub(crate) mod sys {
    use std::ffi::c_void;

    pub type Handle = *mut c_void;
    pub type Bool = i32;

    pub const MEM_COMMIT: u32 = 0x0000_1000;
    pub const MEM_RESERVE: u32 = 0x0000_2000;
    pub const MEM_RELEASE: u32 = 0x0000_8000;
    pub const PAGE_READWRITE: u32 = 0x04;
    pub const NUMA_NO_PREFERRED_NODE: u32 = 0xffff_ffff;

    pub const QUOTA_LIMITS_HARDWS_MIN_ENABLE: u32 = 0x1;
    pub const QUOTA_LIMITS_HARDWS_MAX_DISABLE: u32 = 0x8;

    /// `MEMORY_RESOURCE_NOTIFICATION_TYPE::LowMemoryResourceNotification`.
    pub const LOW_MEMORY_RESOURCE_NOTIFICATION: i32 = 0;

    /// `JOBOBJECTINFOCLASS::JobObjectExtendedLimitInformation`.
    pub const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;
    pub const JOB_OBJECT_LIMIT_PROCESS_MEMORY: u32 = 0x0000_0100;
    pub const JOB_OBJECT_LIMIT_JOB_MEMORY: u32 = 0x0000_0200;

    /// `MEMORYSTATUSEX`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct MemoryStatusEx {
        pub length: u32,
        pub memory_load: u32,
        pub total_phys: u64,
        pub avail_phys: u64,
        pub total_page_file: u64,
        pub avail_page_file: u64,
        pub total_virtual: u64,
        pub avail_virtual: u64,
        pub avail_extended_virtual: u64,
    }

    /// `PROCESS_MEMORY_COUNTERS_EX`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct ProcessMemoryCountersEx {
        pub cb: u32,
        pub page_fault_count: u32,
        pub peak_working_set_size: usize,
        pub working_set_size: usize,
        pub quota_peak_paged_pool_usage: usize,
        pub quota_paged_pool_usage: usize,
        pub quota_peak_non_paged_pool_usage: usize,
        pub quota_non_paged_pool_usage: usize,
        pub pagefile_usage: usize,
        pub peak_pagefile_usage: usize,
        pub private_usage: usize,
    }

    /// `JOBOBJECT_BASIC_LIMIT_INFORMATION`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct JobBasicLimit {
        pub per_process_user_time_limit: i64,
        pub per_job_user_time_limit: i64,
        pub limit_flags: u32,
        pub minimum_working_set_size: usize,
        pub maximum_working_set_size: usize,
        pub active_process_limit: u32,
        pub affinity: usize,
        pub priority_class: u32,
        pub scheduling_class: u32,
    }

    /// `IO_COUNTERS`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct IoCounters {
        pub read_operation_count: u64,
        pub write_operation_count: u64,
        pub other_operation_count: u64,
        pub read_transfer_count: u64,
        pub write_transfer_count: u64,
        pub other_transfer_count: u64,
    }

    /// `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct JobExtendedLimit {
        pub basic: JobBasicLimit,
        pub io: IoCounters,
        pub process_memory_limit: usize,
        pub job_memory_limit: usize,
        pub peak_process_memory_used: usize,
        pub peak_job_memory_used: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn GetCurrentProcess() -> Handle;
        pub fn GetLastError() -> u32;
        pub fn CloseHandle(h: Handle) -> Bool;
        pub fn VirtualAllocExNuma(
            process: Handle,
            address: *mut c_void,
            size: usize,
            allocation_type: u32,
            protect: u32,
            preferred: u32,
        ) -> *mut c_void;
        pub fn VirtualAlloc(
            address: *mut c_void,
            size: usize,
            allocation_type: u32,
            protect: u32,
        ) -> *mut c_void;
        pub fn VirtualFree(address: *mut c_void, size: usize, free_type: u32) -> Bool;
        pub fn VirtualLock(address: *mut c_void, size: usize) -> Bool;
        pub fn VirtualUnlock(address: *mut c_void, size: usize) -> Bool;
        pub fn GetProcessWorkingSetSizeEx(
            process: Handle,
            minimum: *mut usize,
            maximum: *mut usize,
            flags: *mut u32,
        ) -> Bool;
        pub fn SetProcessWorkingSetSizeEx(
            process: Handle,
            minimum: usize,
            maximum: usize,
            flags: u32,
        ) -> Bool;
        pub fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> Bool;
        pub fn K32GetProcessMemoryInfo(
            process: Handle,
            counters: *mut ProcessMemoryCountersEx,
            cb: u32,
        ) -> Bool;
        pub fn GetNumaHighestNodeNumber(highest: *mut u32) -> Bool;
        pub fn GetNumaAvailableMemoryNodeEx(node: u16, available: *mut u64) -> Bool;
        pub fn CreateMemoryResourceNotification(kind: i32) -> Handle;
        pub fn QueryMemoryResourceNotification(handle: Handle, state: *mut Bool) -> Bool;
        pub fn QueryInformationJobObject(
            job: Handle,
            class: i32,
            info: *mut c_void,
            length: u32,
            return_length: *mut u32,
        ) -> Bool;
    }
}

fn last_error() -> i32 {
    // SAFETY: GetLastError reads this thread's last-error value and touches no memory.
    unsafe { sys::GetLastError() as i32 }
}

/// The memory limits of the job object the process runs in (RFC-0020 §5, Windows).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JobMemory {
    /// `ProcessMemoryLimit`, when `JOB_OBJECT_LIMIT_PROCESS_MEMORY` is set.
    pub process_limit: Option<Bytes>,
    /// `JobMemoryLimit`, when `JOB_OBJECT_LIMIT_JOB_MEMORY` is set.
    pub job_limit: Option<Bytes>,
    /// This process's commit charge (`PrivateUsage`), which the process limit counts.
    pub process_usage: Bytes,
    /// `PeakJobMemoryUsed`: never below the job's current commit, so the headroom computed from
    /// it is conservative.
    pub job_peak: Bytes,
}

impl JobMemory {
    /// The tighter of the two limits as a cgroup record: `max` the limit and `current` its
    /// usage, so the pinning gate's cgroup term, `max − current`, is the job's headroom.
    /// `None` when neither limit is set.
    pub fn as_cgroup(&self) -> Option<CgroupMem> {
        let process = self.process_limit.map(|l| (l, self.process_usage));
        let job = self.job_limit.map(|l| (l, self.job_peak));
        let headroom = |(l, u): (Bytes, Bytes)| l.0.saturating_sub(u.0);
        let (max, current) = match (process, job) {
            (Some(p), Some(j)) => {
                if headroom(p) <= headroom(j) {
                    p
                } else {
                    j
                }
            }
            (Some(x), None) | (None, Some(x)) => x,
            (None, None) => return None,
        };
        Some(CgroupMem {
            max: Some(max),
            high: None,
            current,
            high_events: 0,
            max_events: 0,
        })
    }

    /// `job-object limit − usage`: the `cgroup_limit` input of RFC-0020 §5's Windows rule.
    pub fn headroom(&self) -> Option<Bytes> {
        self.as_cgroup()
            .map(|c| Bytes(c.max.unwrap_or_default().0.saturating_sub(c.current.0)))
    }
}

/// This process's commit charge, `PrivateUsage`.
pub fn process_commit() -> Option<Bytes> {
    let mut c = sys::ProcessMemoryCountersEx {
        cb: size_of::<sys::ProcessMemoryCountersEx>() as u32,
        ..Default::default()
    };
    // SAFETY: the pseudo-handle names this process; `c` is a writable
    // PROCESS_MEMORY_COUNTERS_EX whose size is passed as `cb`.
    let ok = unsafe { sys::K32GetProcessMemoryInfo(sys::GetCurrentProcess(), &mut c, c.cb) };
    (ok != 0).then_some(Bytes(c.private_usage as u64))
}

/// The memory limits of this process's job object, `None` when it is in no job or the job
/// cannot be queried.
pub fn job_memory() -> Option<JobMemory> {
    let mut info = sys::JobExtendedLimit::default();
    // SAFETY: a null job handle queries the job of the calling process; `info` is a writable
    // JOBOBJECT_EXTENDED_LIMIT_INFORMATION and its size is passed as the length.
    let ok = unsafe {
        sys::QueryInformationJobObject(
            std::ptr::null_mut(),
            sys::JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            (&mut info as *mut sys::JobExtendedLimit).cast(),
            size_of::<sys::JobExtendedLimit>() as u32,
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        return None;
    }
    let flags = info.basic.limit_flags;
    Some(JobMemory {
        process_limit: (flags & sys::JOB_OBJECT_LIMIT_PROCESS_MEMORY != 0)
            .then_some(Bytes(info.process_memory_limit as u64)),
        job_limit: (flags & sys::JOB_OBJECT_LIMIT_JOB_MEMORY != 0)
            .then_some(Bytes(info.job_memory_limit as u64)),
        process_usage: process_commit().unwrap_or_default(),
        job_peak: Bytes(info.peak_job_memory_used as u64),
    })
}

/// The highest NUMA node number, `None` when the call fails.
fn highest_node() -> Option<u32> {
    let mut highest = 0u32;
    // SAFETY: `highest` is a writable local.
    (unsafe { sys::GetNumaHighestNodeNumber(&mut highest) } != 0).then_some(highest)
}

/// `GlobalMemoryStatusEx`.
fn memory_status() -> Option<sys::MemoryStatusEx> {
    let mut m = sys::MemoryStatusEx {
        length: size_of::<sys::MemoryStatusEx>() as u32,
        ..Default::default()
    };
    // SAFETY: `m` is a writable MEMORYSTATUSEX whose dwLength is set, as the call requires.
    (unsafe { sys::GlobalMemoryStatusEx(&mut m) } != 0).then_some(m)
}

/// Process-wide lock accounting. The working-set minimum belongs to the process, so every
/// adapter's locked bytes are counted together and one minimum covers them all.
#[derive(Debug)]
struct WorkingSet {
    /// The minimum before the first lock, which keeps the overhead the system needs.
    base_min: Option<usize>,
    /// Bytes locked through any adapter.
    locked: u64,
    /// Whether the last hard working-set minimum request succeeded.
    hard_min_ok: bool,
}

static WORKING_SET: Mutex<WorkingSet> = Mutex::new(WorkingSet {
    base_min: None,
    locked: 0,
    hard_min_ok: true,
});

/// Lock order: the registry, then `WORKING_SET`, then an adapter's `locked_by_range`. No path
/// holds `locked_by_range` while taking `WORKING_SET`.
fn working_set() -> MutexGuard<'static, WorkingSet> {
    WORKING_SET.lock().unwrap_or_else(|p| p.into_inner())
}

fn working_set_size() -> Result<(usize, usize, u32), i32> {
    let (mut min, mut max, mut flags) = (0usize, 0usize, 0u32);
    // SAFETY: the pseudo-handle names this process; the three out-pointers are writable
    // locals.
    let ok = unsafe {
        sys::GetProcessWorkingSetSizeEx(sys::GetCurrentProcess(), &mut min, &mut max, &mut flags)
    };
    if ok != 0 {
        Ok((min, max, flags))
    } else {
        Err(last_error())
    }
}

/// Raises the hard working-set minimum to cover `locked` bytes, keeping the maximum soft.
fn cover_working_set(ws: &mut WorkingSet, locked: u64) -> Result<(), i32> {
    let (min, max, flags) = working_set_size()?;
    let base = *ws.base_min.get_or_insert(min);
    let want = base.saturating_add(locked as usize);
    if want <= min && flags & sys::QUOTA_LIMITS_HARDWS_MIN_ENABLE != 0 {
        return Ok(());
    }
    let new_min = want.max(min);
    let new_max = max.max(new_min.saturating_add(PAGE as usize * 64));
    // SAFETY: the pseudo-handle names this process; the call takes integers only.
    let ok = unsafe {
        sys::SetProcessWorkingSetSizeEx(
            sys::GetCurrentProcess(),
            new_min,
            new_max,
            sys::QUOTA_LIMITS_HARDWS_MIN_ENABLE | sys::QUOTA_LIMITS_HARDWS_MAX_DISABLE,
        )
    };
    ws.hard_min_ok = ok != 0;
    if ok != 0 { Ok(()) } else { Err(last_error()) }
}

/// The Windows host-memory adapter.
#[derive(Debug)]
pub struct WindowsMemory {
    live: Registry,
    numa_reserve: Bytes,
    /// Locked bytes per live range of this adapter, by base address, so `release` gives back
    /// what a range still held locked.
    locked_by_range: Mutex<BTreeMap<usize, u64>>,
    /// The low-memory resource notification handle, as an integer so the adapter is `Send`
    /// and `Sync`; 0 when it could not be created.
    low_memory: usize,
}

impl WindowsMemory {
    /// An adapter over the live host, with RFC-0020 §5's `numa_reserve` for the per-node
    /// estimate.
    pub fn new(numa_reserve: Bytes) -> Self {
        // SAFETY: an integer argument; the returned handle is closed in Drop.
        let handle =
            unsafe { sys::CreateMemoryResourceNotification(sys::LOW_MEMORY_RESOURCE_NOTIFICATION) };
        Self {
            live: Registry::default(),
            numa_reserve,
            locked_by_range: Mutex::new(BTreeMap::new()),
            low_memory: handle as usize,
        }
    }

    /// The page size.
    pub fn page_size(&self) -> Bytes {
        Bytes(PAGE)
    }

    /// Bytes this process holds locked through any adapter.
    pub fn locked(&self) -> Bytes {
        Bytes(working_set().locked)
    }

    fn by_range(&self) -> std::sync::MutexGuard<'_, BTreeMap<usize, u64>> {
        self.locked_by_range
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    /// Whether the low-memory resource notification is signalled. `None` without a handle.
    pub fn low_memory(&self) -> Option<bool> {
        if self.low_memory == 0 {
            return None;
        }
        let mut state: sys::Bool = 0;
        // SAFETY: the handle is the notification this adapter created and still owns; `state`
        // is a writable local.
        let ok = unsafe {
            sys::QueryMemoryResourceNotification(self.low_memory as sys::Handle, &mut state)
        };
        (ok != 0).then_some(state != 0)
    }

    fn numa_nodes(&self) -> NumaNodes {
        let mut nodes = NumaNodes::default();
        let Some(highest) = highest_node() else {
            return nodes;
        };
        for node in 0..=highest.min(u32::from(u16::MAX)) as u16 {
            let mut bytes = 0u64;
            // SAFETY: `bytes` is a writable local; an unknown node fails without writing.
            if unsafe { sys::GetNumaAvailableMemoryNodeEx(node, &mut bytes) } != 0 {
                nodes.free.push((node, Bytes(bytes)));
                nodes
                    .available
                    .push((node, Bytes(bytes.saturating_sub(self.numa_reserve.0))));
            }
        }
        nodes
    }
}

impl OsMemory for WindowsMemory {
    fn snapshot(&self) -> MemSnapshot {
        let m = memory_status().unwrap_or_default();
        let nodes = self.numa_nodes();
        let reading = (m.length != 0).then(|| CommitCharge {
            limit: Bytes(m.total_page_file),
            charge: Bytes(m.total_page_file.saturating_sub(m.avail_page_file)),
        });
        MemSnapshot {
            total: Bytes(m.total_phys),
            available: Bytes(m.avail_phys),
            numa_free: nodes.free,
            numa_available: nodes.available,
            cgroup: job_memory().and_then(|j| j.as_cgroup()),
            commit_headroom: commit_headroom(reading),
            own_locked: self.locked(),
            own_registered: Bytes(0),
            psi: None,
            direct_reclaim_stalls: None,
            os_level: self.low_memory().map(|low| {
                if low {
                    PressureLevel::High
                } else {
                    PressureLevel::Normal
                }
            }),
            at: std::time::Instant::now(),
        }
    }

    fn limits(&self) -> PinLimits {
        PinLimits {
            memlock: None,
            cap_ipc_lock: false,
            // RFC-0020 §5 grants no fixed-buffer budget off Linux (note of 2026-10-22).
            io_fixed_max: Some(Bytes(0)),
            ws_hard_min_ok: working_set().hard_min_ok,
            helper_gpu_cap: Some(HELPER_GPU_CAP),
            ..PinLimits::default()
        }
    }

    fn reserve(
        &self,
        bytes: Bytes,
        numa: Option<u16>,
        _huge: HugePages,
    ) -> Result<HostRange, OsMemError> {
        let len = bytes
            .0
            .checked_next_multiple_of(PAGE)
            .filter(|l| *l > 0)
            .and_then(|l| usize::try_from(l).ok())
            .ok_or(OsMemError::OutOfRange {
                len: bytes,
                sub: 0..bytes.0,
            })?;
        let alloc = |node: u32| {
            // SAFETY: a reservation at an address the system picks in this process; no
            // existing memory is affected and the result is checked for null.
            unsafe {
                sys::VirtualAllocExNuma(
                    sys::GetCurrentProcess(),
                    std::ptr::null_mut(),
                    len,
                    sys::MEM_RESERVE,
                    sys::PAGE_READWRITE,
                    node,
                )
            }
        };
        // A preferred node the system has is tried twice; after that, or for a node it does
        // not have, the range is reserved unbound (TIER_NUMA_BIND, RFC-0020 §15).
        let known = |n: u16| highest_node().is_some_and(|h| u32::from(n) <= h);
        let (addr, bound) = match numa.filter(|&n| known(n)) {
            Some(n) => {
                let a = alloc(u32::from(n));
                let a = if a.is_null() { alloc(u32::from(n)) } else { a };
                if a.is_null() {
                    (alloc(sys::NUMA_NO_PREFERRED_NODE), None)
                } else {
                    (a, Some(n))
                }
            }
            None => (alloc(sys::NUMA_NO_PREFERRED_NODE), None),
        };
        if addr.is_null() {
            return Err(OsMemError::HostReserve {
                op: "VirtualAllocExNuma(MEM_RESERVE)",
                errno: last_error(),
            });
        }
        let id = self.live.insert(addr as usize, len as u64);
        Ok(HostRange {
            base: addr.cast(),
            len: Bytes(len as u64),
            numa: bound,
            os_handle: id,
        })
    }

    fn commit(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
        self.live
            .with_pages(r, &sub, PAGE, Round::Outward, |addr, len| {
                // SAFETY: [addr, addr + len) lies inside a live reservation this adapter made,
                // kept reserved by the registry lock; committing changes no byte.
                let got = unsafe {
                    sys::VirtualAlloc(addr.cast(), len, sys::MEM_COMMIT, sys::PAGE_READWRITE)
                };
                if got.is_null() {
                    Err(OsMemError::HostReserve {
                        op: "VirtualAlloc(MEM_COMMIT)",
                        errno: last_error(),
                    })
                } else {
                    Ok(())
                }
            })
    }

    fn lock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
        self.live
            .with_pages(r, &sub, PAGE, Round::Outward, |addr, len| {
                let mut ws = working_set();
                let refuse = |errno| {
                    Err(OsMemError::LockRefused(PageableReason::LockFailed {
                        errno,
                    }))
                };
                let want = ws.locked + len as u64;
                if let Err(errno) = cover_working_set(&mut ws, want) {
                    return refuse(errno);
                }
                // SAFETY: [addr, addr + len) lies inside a live reservation this adapter made,
                // kept by the registry lock; VirtualLock only changes whether the committed
                // pages may leave the working set, and fails on uncommitted ones.
                if unsafe { sys::VirtualLock(addr.cast(), len) } != 0 {
                    ws.locked += len as u64;
                    *self.by_range().entry(r.base as usize).or_default() += len as u64;
                    Ok(())
                } else {
                    refuse(last_error())
                }
            })
    }

    /// Unlocks only the pages wholly inside `sub`, so a page a neighbouring locked range
    /// shares stays locked. The working-set minimum is not lowered.
    fn unlock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
        self.live
            .with_pages(r, &sub, PAGE, Round::Inward, |addr, len| {
                if len == 0 {
                    return Ok(());
                }
                // SAFETY: as for VirtualLock.
                if unsafe { sys::VirtualUnlock(addr.cast(), len) } != 0 {
                    let mut by_range = self.by_range();
                    let held = by_range.entry(r.base as usize).or_default();
                    let freed = (*held).min(len as u64);
                    *held -= freed;
                    drop(by_range);
                    let mut ws = working_set();
                    ws.locked = ws.locked.saturating_sub(freed);
                    Ok(())
                } else {
                    Err(OsMemError::Unlock {
                        errno: last_error(),
                    })
                }
            })
    }

    fn release(&self, r: HostRange) {
        self.live.remove_with(&r, || {
            // Freeing a reservation unlocks its pages.
            let held = self.by_range().remove(&(r.base as usize));
            if let Some(held) = held {
                let mut ws = working_set();
                ws.locked = ws.locked.saturating_sub(held);
            }
            // SAFETY: the range is a live reservation this adapter made, just removed from the
            // registry, with the exclusive lock held, so no other call is acting on it.
            // MEM_RELEASE with size 0 frees the whole reservation from its base.
            unsafe { sys::VirtualFree(r.base.cast::<c_void>(), 0, sys::MEM_RELEASE) };
        });
    }
}

impl Drop for WindowsMemory {
    fn drop(&mut self) {
        let held: u64 = self.by_range().values().sum();
        let mut ws = working_set();
        ws.locked = ws.locked.saturating_sub(held);
        drop(ws);
        for (base, _) in self.live.drain() {
            // SAFETY: each entry is a reservation this adapter made and never released; the
            // adapter is going away, so no call can reach it afterwards.
            unsafe { sys::VirtualFree(base as *mut c_void, 0, sys::MEM_RELEASE) };
        }
        if self.low_memory != 0 {
            // SAFETY: the notification handle this adapter created, closed once.
            unsafe { sys::CloseHandle(self.low_memory as sys::Handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The structures match the Windows SDK's 64-bit layouts.
    #[test]
    fn struct_layouts_match_the_sdk() {
        assert_eq!(size_of::<sys::MemoryStatusEx>(), 64);
        assert_eq!(size_of::<sys::ProcessMemoryCountersEx>(), 80);
        assert_eq!(size_of::<sys::JobBasicLimit>(), 64);
        assert_eq!(std::mem::offset_of!(sys::JobBasicLimit, limit_flags), 16);
        assert_eq!(std::mem::offset_of!(sys::JobBasicLimit, affinity), 48);
        assert_eq!(size_of::<sys::IoCounters>(), 48);
        assert_eq!(size_of::<sys::JobExtendedLimit>(), 144);
        assert_eq!(
            std::mem::offset_of!(sys::JobExtendedLimit, process_memory_limit),
            112
        );
    }

    #[test]
    fn job_memory_picks_the_tighter_limit() {
        let j = JobMemory {
            process_limit: Some(Bytes(1000)),
            job_limit: Some(Bytes(5000)),
            process_usage: Bytes(400),
            job_peak: Bytes(4500),
        };
        // process: 600 left; job: 500 left.
        assert_eq!(j.headroom(), Some(Bytes(500)));
        assert_eq!(j.as_cgroup().unwrap().max, Some(Bytes(5000)));
        assert_eq!(JobMemory::default().as_cgroup(), None);
    }
}
