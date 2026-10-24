// SPDX-License-Identifier: MIT OR Apache-2.0

//! The macOS [`OsMemory`] (RFC-0020 §4.2, macOS row; §5, macOS; §9):
//!
//! - `reserve`: anonymous `mmap(PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON)`, its length
//!   a multiple of the 16 KiB page and its base 16 KiB-aligned, as Metal's no-copy buffers
//!   need (§9). One memory domain, so no NUMA binding; no huge pages.
//! - `commit`: every page is touched with an atomic no-op read-modify-write on one byte, as on
//!   Linux when `MADV_POPULATE_WRITE` is refused; macOS has no populate advice.
//! - `lock`: `mlock`, which wires the pages, only while this process's wired bytes plus the
//!   request stay within the lower of `vm.user_wire_limit` and `vm.global_user_wire_limit`;
//!   otherwise [`OsMemError::LockRefused`] with [`PageableReason::WireLimit`] and the range
//!   stays `Pageable`. The adapter counts the pages it wired, page by page: a page is wired
//!   once however many lock calls cover it, and unwired only when it was wired.
//! - `snapshot`: `hw.memsize`, and free + inactive + purgeable pages from
//!   `host_statistics64(HOST_VM_INFO64)`.
//! - `limits` and [`MacMemory::wire_limits`]: the two wire limits and `iogpu.wired_limit_mb`.
//!
//! Every `sysctlbyname` call goes through one function, [`MacMemory::sysctl_read`], which
//! passes a null `newp` and a zero `newlen`: the adapter never writes a sysctl. Only
//! `syntropy tune` changes `iogpu.wired_limit_mb` (RFC-0020 §9.1). The tests replace the C
//! function with a shim that asserts this on every call.
//!
//! The workspace has no C bindings crate, so the C functions and constants are declared here,
//! with the values of macOS on AArch64, the macOS target ADR-0020 supports. Live ranges are
//! kept in the shared [`Registry`], as on the other adapters.

#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::ffi::{CStr, c_void};
use std::ops::Range;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use smallvec::SmallVec;
use syntropy_core::Bytes;

use super::registry::{Registry, Round};
use super::{
    HostRange, HugePages, MemSnapshot, OsMemError, OsMemory, PageableReason, PinLimits,
    commit_headroom,
};

/// The Apple silicon page, and the alignment Metal's no-copy buffers need (RFC-0020 §9).
pub const MAC_PAGE: Bytes = Bytes(16 << 10);

mod sys {
    use std::ffi::{c_char, c_int, c_long, c_uint, c_void};

    pub const PROT_READ: c_int = 0x1;
    pub const PROT_WRITE: c_int = 0x2;
    pub const MAP_PRIVATE: c_int = 0x0002;
    pub const MAP_ANON: c_int = 0x1000;
    /// `(void *) -1`.
    pub const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;

    /// `_SC_PAGESIZE` in the macOS C library.
    pub const SC_PAGESIZE: c_int = 29;

    pub const EINVAL: i32 = 22;

    /// `HOST_VM_INFO64` and its count in `integer_t` units: `sizeof(vm_statistics64) / 4`.
    pub const HOST_VM_INFO64: c_int = 4;
    pub const HOST_VM_INFO64_COUNT: c_uint = 38;
    pub const KERN_SUCCESS: c_int = 0;

    /// `struct vm_statistics64` from `<mach/vm_statistics.h>`, 152 bytes, 8-aligned.
    #[repr(C, align(8))]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct VmStatistics64 {
        pub free_count: u32,
        pub active_count: u32,
        pub inactive_count: u32,
        pub wire_count: u32,
        pub zero_fill_count: u64,
        pub reactivations: u64,
        pub pageins: u64,
        pub pageouts: u64,
        pub faults: u64,
        pub cow_faults: u64,
        pub lookups: u64,
        pub hits: u64,
        pub purges: u64,
        pub purgeable_count: u32,
        pub speculative_count: u32,
        pub decompressions: u64,
        pub compressions: u64,
        pub swapins: u64,
        pub swapouts: u64,
        pub compressor_page_count: u32,
        pub throttled_count: u32,
        pub external_page_count: u32,
        pub internal_page_count: u32,
        pub total_uncompressed_pages_in_compressor: u64,
    }

    /// `sysctlbyname`'s C signature.
    pub type SysctlFn = unsafe extern "C" fn(
        name: *const c_char,
        oldp: *mut c_void,
        oldlenp: *mut usize,
        newp: *mut c_void,
        newlen: usize,
    ) -> c_int;

    unsafe extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
        pub fn mlock(addr: *const c_void, len: usize) -> c_int;
        pub fn munlock(addr: *const c_void, len: usize) -> c_int;
        pub fn sysconf(name: c_int) -> c_long;
        pub fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
        pub fn mach_host_self() -> c_uint;
        pub fn mach_port_deallocate(task: c_uint, name: c_uint) -> c_int;
        pub fn host_statistics64(
            host: c_uint,
            flavor: c_int,
            info: *mut i32,
            count: *mut c_uint,
        ) -> c_int;
        pub static mach_task_self_: c_uint;
    }
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// The wire limits as one reading (RFC-0020 §5, macOS; §9.1). A field is `None` when its
/// sysctl could not be read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WireLimits {
    /// `vm.user_wire_limit`: what one user may wire.
    pub user: Option<Bytes>,
    /// `vm.global_user_wire_limit`: what all users together may wire.
    pub global: Option<Bytes>,
    /// `iogpu.wired_limit_mb` in bytes; `None` when unreadable or 0, the OS default.
    pub iogpu_wired_limit: Option<Bytes>,
}

impl WireLimits {
    /// The lower of the user and global limits; `None` when neither could be read.
    pub fn cap(&self) -> Option<Bytes> {
        match (self.user, self.global) {
            (Some(u), Some(g)) => Some(Bytes(u.0.min(g.0))),
            (one, other) => one.or(other),
        }
    }
}

/// The pages of each live range this adapter wired, by reservation id, and their total.
#[derive(Debug, Default)]
struct Wired {
    pages: BTreeMap<u64, Vec<bool>>,
    total_pages: u64,
}

/// The macOS host-memory adapter.
#[derive(Debug)]
pub struct MacMemory {
    page: u64,
    host: u32,
    sysctl: sys::SysctlFn,
    reg: Registry,
    wired: Mutex<Wired>,
}

impl MacMemory {
    /// An adapter over the live host.
    pub fn new() -> Self {
        Self::with_sysctl(sys::sysctlbyname)
    }

    /// An adapter whose sysctl reads go through `sysctl`, a function with `sysctlbyname`'s
    /// signature: the tests' shim.
    fn with_sysctl(sysctl: sys::SysctlFn) -> Self {
        // SAFETY: sysconf takes an integer name and reads no user memory.
        let page = unsafe { sys::sysconf(sys::SC_PAGESIZE) };
        let page = u64::try_from(page).ok().filter(|p| *p > 0).unwrap_or(0);
        // SAFETY: mach_host_self takes no arguments and returns a send right to the host port,
        // which `Drop` gives back.
        let host = unsafe { sys::mach_host_self() };
        Self {
            page: page.max(MAC_PAGE.0),
            host,
            sysctl,
            reg: Registry::default(),
            wired: Mutex::new(Wired::default()),
        }
    }

    /// The page size the adapter reserves, commits and wires in: 16 KiB.
    pub fn page_size(&self) -> Bytes {
        Bytes(self.page)
    }

    /// The bytes this adapter holds wired.
    pub fn wired(&self) -> Bytes {
        Bytes(self.lock_wired().total_pages * self.page)
    }

    fn lock_wired(&self) -> MutexGuard<'_, Wired> {
        self.wired.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Reads the integer sysctl `name`, 4 or 8 bytes wide. Read-only: `newp` is null and
    /// `newlen` 0. This is the one place the adapter calls `sysctlbyname`.
    fn sysctl_read(&self, name: &CStr) -> Option<u64> {
        let mut buf = [0u8; 8];
        let mut len: usize = buf.len();
        // SAFETY: `name` is NUL-terminated; `oldp` points at `len` writable bytes, and the
        // function writes at most `len` and stores the length it wrote in `len`. With a null
        // `newp` and a zero `newlen` it changes nothing.
        let rc = unsafe {
            (self.sysctl)(
                name.as_ptr(),
                buf.as_mut_ptr().cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            return None;
        }
        match len {
            4 => Some(u64::from(u32::from_ne_bytes([
                buf[0], buf[1], buf[2], buf[3],
            ]))),
            8 => Some(u64::from_ne_bytes(buf)),
            _ => None,
        }
    }

    /// The wire limits and `iogpu.wired_limit_mb`, read now.
    pub fn wire_limits(&self) -> WireLimits {
        WireLimits {
            user: self.sysctl_read(c"vm.user_wire_limit").map(Bytes),
            global: self.sysctl_read(c"vm.global_user_wire_limit").map(Bytes),
            iogpu_wired_limit: self
                .sysctl_read(c"iogpu.wired_limit_mb")
                .filter(|mb| *mb > 0)
                .and_then(|mb| mb.checked_mul(1 << 20))
                .map(Bytes),
        }
    }

    /// Free, inactive and purgeable pages from `host_statistics64`, in bytes.
    fn vm_available(&self) -> Option<u64> {
        let mut info = sys::VmStatistics64::default();
        let mut count = sys::HOST_VM_INFO64_COUNT;
        // SAFETY: `info` is a `vm_statistics64` of `HOST_VM_INFO64_COUNT` integers, 8-aligned;
        // the kernel writes at most `count` integers into it and stores how many it wrote.
        let rc = unsafe {
            sys::host_statistics64(
                self.host,
                sys::HOST_VM_INFO64,
                (&raw mut info).cast(),
                &mut count,
            )
        };
        if rc != sys::KERN_SUCCESS {
            return None;
        }
        let pages = u64::from(info.free_count)
            + u64::from(info.inactive_count)
            + u64::from(info.purgeable_count);
        pages.checked_mul(self.page)
    }

    /// The page indices `[first, end)` of the `len` bytes at `addr` in range `r`.
    fn page_ix(&self, r: &HostRange, addr: *mut u8, len: usize) -> Range<usize> {
        let first = (addr as usize - r.base as usize) / self.page as usize;
        first..first + len / self.page as usize
    }
}

/// The maximal runs of page indices in `ix` whose flag equals `want`.
fn runs(flags: &[bool], ix: Range<usize>, want: bool) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut i = ix.start;
    while i < ix.end {
        if flags[i] != want {
            i += 1;
            continue;
        }
        let start = i;
        while i < ix.end && flags[i] == want {
            i += 1;
        }
        out.push(start..i);
    }
    out
}

impl Default for MacMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Faults in every page of `[addr, addr + len)` with an atomic no-op read-modify-write on
/// each page's first byte (`fetch_or(0)`), which keeps the bytes there and never discards a
/// store another thread makes.
///
/// # Safety
///
/// `[addr, addr + len)` must be a readable and writable mapping that stays mapped for the
/// call, and no other thread may access those bytes non-atomically during it
/// ([`OsMemory::commit`] states this to callers).
unsafe fn touch(addr: *mut u8, len: usize, page: usize) {
    let mut off = 0;
    while off < len {
        // SAFETY: off < len, so the pointer is inside the mapping the caller vouches for, which
        // stays mapped for the call; a u8 is always aligned; and no non-atomic access to the
        // byte runs concurrently, as the caller guarantees.
        let byte = unsafe { AtomicU8::from_ptr(addr.add(off)) };
        byte.fetch_or(0, Ordering::Relaxed);
        off += page;
    }
}

impl OsMemory for MacMemory {
    // `MemSnapshot::at` is a `std::time::Instant` (RFC-0020 §3), stamped here on the OS side of
    // the tier manager, not in its policy (RFC-0060 §2.1 rule T1). A listed exception to the
    // determinism lints (`xtask/src/determinism.rs`).
    #[allow(
        clippy::disallowed_methods,
        reason = "stamps the OS reading's `MemSnapshot::at`, a std `Instant`"
    )]
    fn snapshot(&self) -> MemSnapshot {
        let total = Bytes(self.sysctl_read(c"hw.memsize").unwrap_or(0));
        let available = Bytes(self.vm_available().unwrap_or(0).min(total.0));
        let own = self.wired();
        MemSnapshot {
            total,
            available,
            numa_free: SmallVec::new(),
            numa_available: SmallVec::new(),
            cgroup: None,
            commit_headroom: commit_headroom(None),
            own_locked: own,
            own_registered: Bytes(0),
            psi: None,
            direct_reclaim_stalls: None,
            os_level: None,
            at: Instant::now(),
        }
    }

    /// The lower wire limit as [`PinLimits::user_wire_limit`]. `memlock` is `None`: the wire
    /// limit is what bounds `mlock` here. [`PinLimits::gpu_wired_cap`] is Metal's
    /// `recommendedMaxWorkingSetSize`, which the Metal provider reports, so it is `None` here.
    fn limits(&self) -> PinLimits {
        PinLimits {
            user_wire_limit: self.wire_limits().cap(),
            ..PinLimits::default()
        }
    }

    /// `numa` and `huge` are ignored: one memory domain, no huge pages. The range is returned
    /// with `numa = None`.
    fn reserve(
        &self,
        bytes: Bytes,
        _numa: Option<u16>,
        _huge: HugePages,
    ) -> Result<HostRange, OsMemError> {
        let len = bytes
            .0
            .checked_next_multiple_of(self.page)
            .filter(|l| *l > 0)
            .and_then(|l| usize::try_from(l).ok())
            .ok_or(OsMemError::OutOfRange {
                len: bytes,
                sub: 0..bytes.0,
            })?;
        // SAFETY: an anonymous private mapping at an address the kernel picks; no existing
        // memory is affected, and the result is checked against MAP_FAILED.
        let addr = unsafe {
            sys::mmap(
                std::ptr::null_mut(),
                len,
                sys::PROT_READ | sys::PROT_WRITE,
                sys::MAP_PRIVATE | sys::MAP_ANON,
                -1,
                0,
            )
        };
        if addr == sys::MAP_FAILED {
            return Err(OsMemError::HostReserve {
                op: "mmap",
                errno: errno(),
            });
        }
        if !(addr as usize).is_multiple_of(MAC_PAGE.0 as usize) {
            // SAFETY: unmapping the mapping just created, which nothing else references.
            unsafe { sys::munmap(addr, len) };
            return Err(OsMemError::HostReserve {
                op: "mmap (16 KiB alignment)",
                errno: sys::EINVAL,
            });
        }
        let id = self.reg.insert(addr as usize, len as u64);
        let pages = len / self.page as usize;
        self.lock_wired().pages.insert(id, vec![false; pages]);
        Ok(HostRange {
            base: addr.cast(),
            len: Bytes(len as u64),
            numa: None,
            os_handle: id,
        })
    }

    fn commit(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
        self.reg
            .with_pages(r, &sub, self.page, Round::Outward, |addr, len| {
                // SAFETY: the range is inside a live read-write mapping this adapter created,
                // kept mapped by the registry lock; concurrent non-atomic access is excluded by
                // `OsMemory::commit`'s contract.
                unsafe { touch(addr, len, self.page as usize) };
                Ok(())
            })
    }

    /// Wires the pages of `sub` that this adapter has not wired yet, after checking that the
    /// adapter's wired bytes plus those pages stay within [`WireLimits::cap`]. An unreadable
    /// limit counts as 0, so nothing is wired on a guess.
    fn lock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
        let limit = self.wire_limits().cap().unwrap_or(Bytes(0));
        self.reg
            .with_pages(r, &sub, self.page, Round::Outward, |addr, len| {
                let ix = self.page_ix(r, addr, len);
                let mut w = self.lock_wired();
                let total = w.total_pages;
                let flags = w
                    .pages
                    .get_mut(&r.os_handle)
                    .ok_or(OsMemError::UnknownRange)?;
                let new = flags[ix.clone()].iter().filter(|f| !**f).count() as u64;
                let wired = Bytes(total * self.page);
                let requested = Bytes(new * self.page);
                if wired.0.saturating_add(requested.0) > limit.0 {
                    return Err(OsMemError::LockRefused(PageableReason::WireLimit {
                        limit,
                        wired,
                        requested,
                    }));
                }
                let page = self.page as usize;
                let mut out = Ok(());
                let mut done = 0u64;
                for run in runs(flags, ix, false) {
                    let a = r.base.wrapping_add(run.start * page);
                    // SAFETY: the run lies inside a live mapping this adapter created, kept
                    // mapped by the registry lock; mlock only wires its pages.
                    if unsafe { sys::mlock(a.cast(), run.len() * page) } != 0 {
                        out = Err(OsMemError::LockRefused(PageableReason::LockFailed {
                            errno: errno(),
                        }));
                        break;
                    }
                    done += run.len() as u64;
                    flags[run].fill(true);
                }
                w.total_pages += done;
                out
            })
    }

    /// Unwires only the pages wholly inside `sub` that this adapter wired, so a page shared
    /// with a neighbouring locked sub-range stays wired, and a page wired once is unwired once.
    fn unlock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
        self.reg
            .with_pages(r, &sub, self.page, Round::Inward, |addr, len| {
                if len == 0 {
                    return Ok(());
                }
                let ix = self.page_ix(r, addr, len);
                let mut w = self.lock_wired();
                let flags = w
                    .pages
                    .get_mut(&r.os_handle)
                    .ok_or(OsMemError::UnknownRange)?;
                let page = self.page as usize;
                let mut out = Ok(());
                let mut done = 0u64;
                for run in runs(flags, ix, true) {
                    let a = r.base.wrapping_add(run.start * page);
                    // SAFETY: as for mlock.
                    if unsafe { sys::munlock(a.cast(), run.len() * page) } != 0 {
                        out = Err(OsMemError::Unlock { errno: errno() });
                        break;
                    }
                    done += run.len() as u64;
                    flags[run].fill(false);
                }
                w.total_pages -= done;
                out
            })
    }

    fn release(&self, r: HostRange) {
        self.reg.remove_with(&r, || {
            let mut w = self.lock_wired();
            if let Some(flags) = w.pages.remove(&r.os_handle) {
                w.total_pages -= flags.iter().filter(|f| **f).count() as u64;
            }
            drop(w);
            // SAFETY: the range is a live mapping this adapter created, now removed from the
            // registry. The exclusive lock is held, so no commit, lock or unlock is acting on
            // it. munmap also unwires its pages.
            unsafe { sys::munmap(r.base.cast(), r.len.0 as usize) };
        });
    }
}

impl Drop for MacMemory {
    fn drop(&mut self) {
        for (base, len) in self.reg.drain() {
            // SAFETY: each entry is a mapping this adapter created and never released; the
            // adapter is going away, so no call can reach it afterwards.
            unsafe { sys::munmap(base as *mut c_void, len as usize) };
        }
        // SAFETY: gives back the host-port send right `mach_host_self` returned in `new`;
        // `mach_task_self_` is the C library's constant for this task's port.
        unsafe { sys::mach_port_deallocate(sys::mach_task_self_, self.host) };
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::{c_char, c_int};
    use std::sync::atomic::{AtomicU64, AtomicUsize};

    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn vm_statistics64_has_the_c_layout() {
        assert_eq!(size_of::<sys::VmStatistics64>(), 152);
        assert_eq!(
            size_of::<sys::VmStatistics64>() / 4,
            sys::HOST_VM_INFO64_COUNT as usize
        );
    }

    // ---- The read-only shim ----------------------------------------------------------------

    /// Calls the shim saw, and calls that tried to write.
    static SHIM_CALLS: AtomicUsize = AtomicUsize::new(0);
    static SHIM_WRITES: AtomicUsize = AtomicUsize::new(0);
    /// The wire limit the shim reports, in bytes.
    static SHIM_WIRE: AtomicU64 = AtomicU64::new(0);

    /// A `sysctlbyname` that records every call, counts any with a non-null `newp` or a
    /// non-zero `newlen` as a write, and answers the names the adapter reads.
    unsafe extern "C" fn shim(
        name: *const c_char,
        oldp: *mut c_void,
        oldlenp: *mut usize,
        newp: *mut c_void,
        newlen: usize,
    ) -> c_int {
        SHIM_CALLS.fetch_add(1, Ordering::SeqCst);
        if !newp.is_null() || newlen != 0 {
            SHIM_WRITES.fetch_add(1, Ordering::SeqCst);
            return -1;
        }
        // SAFETY: the adapter passes a NUL-terminated name.
        let name = unsafe { CStr::from_ptr(name) }.to_str().unwrap();
        let value: Option<u64> = match name {
            "vm.user_wire_limit" | "vm.global_user_wire_limit" => {
                Some(SHIM_WIRE.load(Ordering::SeqCst))
            }
            "iogpu.wired_limit_mb" => Some(0),
            "hw.memsize" => Some(16 << 30),
            _ => None,
        };
        let Some(v) = value else { return -1 };
        // iogpu.wired_limit_mb is an int, the others 64-bit, as on the real system.
        let bytes: &[u8] = if name == "iogpu.wired_limit_mb" {
            &(v as u32).to_ne_bytes()
        } else {
            &v.to_ne_bytes()
        };
        // SAFETY: the adapter passes `oldp` with `*oldlenp` writable bytes, at least 8.
        unsafe {
            assert!(*oldlenp >= bytes.len());
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), oldp.cast(), bytes.len());
            *oldlenp = bytes.len();
        }
        0
    }

    /// Held by the tests that use the shim's statics.
    static SHIM: Mutex<()> = Mutex::new(());

    /// Criterion 2's guard: every sysctl call the adapter makes over a full cycle (snapshot,
    /// limits, reserve, commit, lock, unlock, release) passes a null `newp` and a zero
    /// `newlen`. Criterion 3's shim half: wired bytes never pass the reported limit.
    #[test]
    fn every_sysctl_call_is_a_read_and_wiring_stays_within_the_limit() {
        let _g = SHIM.lock().unwrap_or_else(|p| p.into_inner());
        SHIM_WIRE.store(4 * MIB, Ordering::SeqCst);
        let before = SHIM_CALLS.load(Ordering::SeqCst);
        let os = MacMemory::with_sysctl(shim);
        let s = os.snapshot();
        assert_eq!(s.total, Bytes(16 << 30));
        let l = os.limits();
        assert_eq!(l.user_wire_limit, Some(Bytes(4 * MIB)));
        assert_eq!(os.wire_limits().iogpu_wired_limit, None);

        let r = os.reserve(Bytes(16 * MIB), None, HugePages::Auto).unwrap();
        os.commit(&r, 0..16 * MIB).unwrap();
        // 3 MiB fits under 4 MiB.
        os.lock(&r, 0..3 * MIB).unwrap();
        assert_eq!(os.wired(), Bytes(3 * MIB));
        // Re-locking wired pages wires nothing new.
        os.lock(&r, 0..3 * MIB).unwrap();
        assert_eq!(os.wired(), Bytes(3 * MIB));
        // 2 MiB more would pass the limit: refused, nothing more wired.
        let e = os.lock(&r, 3 * MIB..5 * MIB).unwrap_err();
        assert_eq!(
            e,
            OsMemError::LockRefused(PageableReason::WireLimit {
                limit: Bytes(4 * MIB),
                wired: Bytes(3 * MIB),
                requested: Bytes(2 * MIB),
            })
        );
        assert_eq!(os.wired(), Bytes(3 * MIB));
        // Exactly up to the limit is allowed.
        os.lock(&r, 3 * MIB..4 * MIB).unwrap();
        assert_eq!(os.wired(), Bytes(4 * MIB));
        assert!(os.lock(&r, 4 * MIB..4 * MIB + 1).is_err());
        os.unlock(&r, 0..2 * MIB).unwrap();
        assert_eq!(os.wired(), Bytes(2 * MIB));
        assert_eq!(os.snapshot().own_locked, Bytes(2 * MIB));
        os.release(r);
        assert_eq!(os.wired(), Bytes(0));

        assert!(SHIM_CALLS.load(Ordering::SeqCst) > before);
        assert_eq!(SHIM_WRITES.load(Ordering::SeqCst), 0);
    }

    /// An unreadable limit counts as 0: nothing is wired on a guess.
    #[test]
    fn an_unreadable_limit_wires_nothing() {
        unsafe extern "C" fn failing(
            _: *const c_char,
            _: *mut c_void,
            _: *mut usize,
            _: *mut c_void,
            _: usize,
        ) -> c_int {
            -1
        }
        let os = MacMemory::with_sysctl(failing);
        assert_eq!(os.wire_limits(), WireLimits::default());
        assert_eq!(os.limits().user_wire_limit, None);
        let r = os.reserve(Bytes(MAC_PAGE.0), None, HugePages::Off).unwrap();
        assert!(matches!(
            os.lock(&r, 0..1),
            Err(OsMemError::LockRefused(PageableReason::WireLimit {
                limit: Bytes(0),
                ..
            }))
        ));
        os.release(r);
    }

    /// Outside the tests, the C `sysctlbyname` is named once, as the default in `new`, and
    /// called once, in `sysctl_read`, which passes a null `newp`.
    #[test]
    fn sysctlbyname_is_reached_only_through_the_read_function() {
        let src = include_str!("macos.rs");
        let code = &src[..src.find("#[cfg(test)]").unwrap()];
        assert_eq!(code.matches("sys::sysctlbyname").count(), 1);
        assert_eq!(code.matches("(self.sysctl)(").count(), 1);
    }

    #[test]
    fn the_cap_is_the_lower_limit() {
        let w = |u: Option<u64>, g: Option<u64>| WireLimits {
            user: u.map(Bytes),
            global: g.map(Bytes),
            iogpu_wired_limit: None,
        };
        assert_eq!(w(Some(5), Some(3)).cap(), Some(Bytes(3)));
        assert_eq!(w(Some(5), None).cap(), Some(Bytes(5)));
        assert_eq!(w(None, Some(7)).cap(), Some(Bytes(7)));
        assert_eq!(w(None, None).cap(), None);
    }
}
