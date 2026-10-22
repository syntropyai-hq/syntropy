// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Linux [`OsMemory`] (RFC-0020 §4.2, Linux row; §5, Linux):
//!
//! - `reserve`: anonymous `mmap(PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS |
//!   MAP_NORESERVE)`, then `madvise(MADV_DONTFORK)`, `madvise(MADV_HUGEPAGE)` when
//!   `tier.hugepages = auto` and THP is `madvise` or `always`, and `mbind(MPOL_BIND)` to the
//!   arena's node before any page is touched. A failed bind is retried once; then the range is
//!   left unbound (`TIER_NUMA_BIND`, RFC-0020 §15).
//! - `commit`: `madvise(MADV_POPULATE_WRITE)` (Linux 5.15, ADR-0020's floor, has it), with a
//!   page-by-page touch where the kernel refuses the advice.
//! - `lock`: `mlock`, only when `RLIMIT_MEMLOCK` covers it or `CAP_IPC_LOCK` is held;
//!   otherwise [`OsMemError::LockRefused`] with the reason, and the range stays `Pageable`.
//! - `snapshot` and `limits`: [`ProcReader`] over `/`.
//!
//! The workspace has no C bindings crate, so the C functions and constants are declared here.
//! The constants are the generic Linux values, the same on x86-64 and AArch64; only the
//! `mbind` syscall number differs, and `mbind` has no C library wrapper.
//!
//! Every range this adapter hands out is recorded with a reservation id, carried in
//! [`HostRange::os_handle`]. `commit`, `lock`, `unlock` and `release` act only on a range it
//! recorded, with the same base, length and id, and hold the registry lock across the system
//! call: `commit`, `lock` and `unlock` hold it shared, `release` exclusively. A range can
//! therefore not be unmapped while another call is acting on it, and a stale or forged
//! `HostRange` whose address was since reused by a new reservation does not match the new id.
//! [`HostRange`]'s fields are public, so this is what keeps those safe methods from touching
//! memory the adapter does not own.

#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::ffi::{c_ulong, c_void};
use std::ops::Range;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{RwLock, RwLockReadGuard};

use syntropy_core::Bytes;

use super::procfs::{self, ProcReader};
use super::{
    HostRange, HugePages, MemSnapshot, OsMemError, OsMemory, PageableReason, PinLimits, check_sub,
    lock_allowed,
};

mod sys {
    use std::ffi::{c_int, c_long, c_ulong, c_void};

    pub const PROT_READ: c_int = 0x1;
    pub const PROT_WRITE: c_int = 0x2;
    pub const MAP_PRIVATE: c_int = 0x02;
    pub const MAP_ANONYMOUS: c_int = 0x20;
    pub const MAP_NORESERVE: c_int = 0x4000;
    /// `(void *) -1`.
    pub const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;

    pub const MADV_DONTFORK: c_int = 10;
    pub const MADV_HUGEPAGE: c_int = 14;
    pub const MADV_POPULATE_WRITE: c_int = 23;

    pub const MPOL_BIND: c_ulong = 2;

    /// `_SC_PAGESIZE` in glibc and musl.
    pub const SC_PAGESIZE: c_int = 30;

    /// `__NR_mbind`. The module is built only for these two architectures.
    #[cfg(target_arch = "x86_64")]
    pub const SYS_MBIND: c_long = 237;
    #[cfg(target_arch = "aarch64")]
    pub const SYS_MBIND: c_long = 235;

    pub const EINTR: i32 = 4;
    pub const EINVAL: i32 = 22;

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
        pub fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
        pub fn mlock(addr: *const c_void, len: usize) -> c_int;
        pub fn munlock(addr: *const c_void, len: usize) -> c_int;
        pub fn sysconf(name: c_int) -> c_long;
        pub fn syscall(number: c_long, ...) -> c_long;
    }
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// The Linux host-memory adapter.
#[derive(Debug)]
pub struct LinuxMemory {
    reader: ProcReader,
    page: u64,
    /// Live reservations by base address.
    live: RwLock<BTreeMap<usize, Live>>,
    /// The next reservation id; 0 is never handed out.
    next_id: AtomicU64,
}

/// One live reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Live {
    len: u64,
    id: u64,
}

/// How [`LinuxMemory::with_pages`] turns a byte range into whole pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Round {
    /// Every page the range touches: commit and lock must cover each byte asked for.
    Outward,
    /// Only pages wholly inside the range: unlock must not release a page a neighbouring
    /// locked sub-range shares, because `mlock` is not counted per caller.
    Inward,
}

impl LinuxMemory {
    /// An adapter over the live host, with RFC-0020 §5's `numa_reserve` for the per-node
    /// estimate.
    pub fn new(numa_reserve: Bytes) -> Self {
        // SAFETY: sysconf takes an integer name and reads no user memory.
        let page = unsafe { sys::sysconf(sys::SC_PAGESIZE) };
        Self {
            reader: ProcReader::new("/", numa_reserve),
            page: u64::try_from(page).ok().filter(|p| *p > 0).unwrap_or(4096),
            live: RwLock::new(BTreeMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// The base page size.
    pub fn page_size(&self) -> Bytes {
        Bytes(self.page)
    }

    fn read_live(&self) -> RwLockReadGuard<'_, BTreeMap<usize, Live>> {
        self.live.read().unwrap_or_else(|p| p.into_inner())
    }

    fn is_live(live: &BTreeMap<usize, Live>, r: &HostRange) -> bool {
        live.get(&(r.base as usize))
            == Some(&Live {
                len: r.len.0,
                id: r.os_handle,
            })
    }

    /// Runs `f` on the page-aligned address and length of `sub`, rounded as `round` says,
    /// with the registry's shared lock held throughout, so `release` cannot unmap the range
    /// while `f` runs.
    fn with_pages<T>(
        &self,
        r: &HostRange,
        sub: &Range<u64>,
        round: Round,
        f: impl FnOnce(*mut u8, usize) -> Result<T, OsMemError>,
    ) -> Result<T, OsMemError> {
        check_sub(r.len, sub)?;
        let live = self.read_live();
        if !Self::is_live(&live, r) {
            return Err(OsMemError::UnknownRange);
        }
        let page = self.page;
        let (start, end) = match round {
            Round::Outward => (
                sub.start - sub.start % page,
                sub.end.div_ceil(page).saturating_mul(page).min(r.len.0),
            ),
            Round::Inward => (
                sub.start.div_ceil(page).saturating_mul(page),
                sub.end - sub.end % page,
            ),
        };
        let len = end.saturating_sub(start);
        let addr = if start < r.len.0 {
            // SAFETY: start < r.len, so the offset stays inside the live mapping that starts
            // at r.base, checked against the registry above and kept mapped by its lock.
            unsafe { r.base.add(start as usize) }
        } else {
            // Inward rounding past the end: len is 0 and the address is never dereferenced.
            r.base
        };
        let out = f(addr, len as usize);
        drop(live);
        out
    }

    /// Binds `[addr, addr + len)` to `node` with `MPOL_BIND`. `Err(errno)` on failure.
    fn bind(addr: *mut c_void, len: usize, node: u16) -> Result<(), i32> {
        let nr = sys::SYS_MBIND;
        let words = usize::from(node) / 64 + 1;
        let mut mask = vec![0 as c_ulong; words];
        mask[usize::from(node) / 64] = 1 << (node % 64);
        // The kernel reads `maxnode − 1` bits, so pass one more than the mask holds.
        let maxnode = (words * 64 + 1) as c_ulong;
        // SAFETY: mbind(addr, len, mode, nodemask, maxnode, flags) reads `maxnode − 1` bits
        // from `mask`, which holds exactly that many; `addr` and `len` are a mapping this
        // adapter created and still owns. It changes only the policy of that mapping.
        let rc = unsafe {
            sys::syscall(
                nr,
                addr as c_ulong,
                len as c_ulong,
                sys::MPOL_BIND,
                mask.as_ptr() as c_ulong,
                maxnode,
                0 as c_ulong,
            )
        };
        if rc == 0 { Ok(()) } else { Err(errno()) }
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

impl OsMemory for LinuxMemory {
    fn snapshot(&self) -> MemSnapshot {
        self.reader.snapshot()
    }

    fn limits(&self) -> PinLimits {
        self.reader.limits()
    }

    fn reserve(
        &self,
        bytes: Bytes,
        numa: Option<u16>,
        huge: HugePages,
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
                sys::MAP_PRIVATE | sys::MAP_ANONYMOUS | sys::MAP_NORESERVE,
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
        // SAFETY: [addr, addr + len) is the mapping just created; the advice changes only its
        // flags.
        if unsafe { sys::madvise(addr, len, sys::MADV_DONTFORK) } != 0 {
            let e = errno();
            // SAFETY: unmapping the mapping just created, which nothing else references.
            unsafe { sys::munmap(addr, len) };
            return Err(OsMemError::HostReserve {
                op: "madvise(MADV_DONTFORK)",
                errno: e,
            });
        }
        if huge == HugePages::Auto && self.reader.thp().advisable() {
            // Advice only: a kernel that refuses it leaves base pages, which are correct.
            // SAFETY: as above, the advice changes only the new mapping's flags.
            let _ = unsafe { sys::madvise(addr, len, sys::MADV_HUGEPAGE) };
        }
        let bound = numa.filter(|&node| {
            Self::bind(addr, len, node).is_ok() || Self::bind(addr, len, node).is_ok()
        });
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.live.write().unwrap_or_else(|p| p.into_inner()).insert(
            addr as usize,
            Live {
                len: len as u64,
                id,
            },
        );
        Ok(HostRange {
            base: addr.cast(),
            len: Bytes(len as u64),
            numa: bound,
            os_handle: id,
        })
    }

    fn commit(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
        self.with_pages(r, &sub, Round::Outward, |addr, len| {
            loop {
                // SAFETY: [addr, addr + len) lies inside a live mapping this adapter created,
                // kept mapped by the registry lock `with_pages` holds; the advice faults pages
                // in and changes no byte.
                if unsafe { sys::madvise(addr.cast(), len, sys::MADV_POPULATE_WRITE) } == 0 {
                    return Ok(());
                }
                match errno() {
                    sys::EINTR => continue,
                    sys::EINVAL => break,
                    e => {
                        return Err(OsMemError::HostReserve {
                            op: "madvise(MADV_POPULATE_WRITE)",
                            errno: e,
                        });
                    }
                }
            }
            // SAFETY: the range is inside a live read-write mapping this adapter created, kept
            // mapped by the registry lock; concurrent non-atomic access is excluded by
            // `OsMemory::commit`'s contract.
            unsafe { touch(addr, len, self.page as usize) };
            Ok(())
        })
    }

    fn lock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
        self.with_pages(r, &sub, Round::Outward, |addr, len| {
            let (locked, _) = procfs::own_lock(self.reader.root());
            lock_allowed(&self.limits(), locked, Bytes(len as u64))
                .map_err(OsMemError::LockRefused)?;
            // SAFETY: [addr, addr + len) lies inside a live mapping this adapter created, kept
            // mapped by the registry lock; mlock only changes whether its pages may be paged
            // out.
            if unsafe { sys::mlock(addr.cast(), len) } == 0 {
                Ok(())
            } else {
                Err(OsMemError::LockRefused(PageableReason::LockFailed {
                    errno: errno(),
                }))
            }
        })
    }

    /// Unlocks only the pages wholly inside `sub`: `mlock` is not counted per caller, so a
    /// page that `sub` shares with a neighbouring locked sub-range stays locked. Pin steps are
    /// page-aligned (RFC-0020 §6.1), where this is exact.
    fn unlock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
        self.with_pages(r, &sub, Round::Inward, |addr, len| {
            if len == 0 {
                return Ok(());
            }
            // SAFETY: as for mlock.
            if unsafe { sys::munlock(addr.cast(), len) } == 0 {
                Ok(())
            } else {
                Err(OsMemError::Unlock { errno: errno() })
            }
        })
    }

    fn release(&self, r: HostRange) {
        let mut live = self.live.write().unwrap_or_else(|p| p.into_inner());
        if !Self::is_live(&live, &r) {
            return;
        }
        live.remove(&(r.base as usize));
        // SAFETY: the range is a live mapping this adapter created, now removed from the
        // registry. The exclusive lock is still held, so no commit, lock or unlock is acting
        // on it, and none that starts later can find it.
        unsafe { sys::munmap(r.base.cast(), r.len.0 as usize) };
        drop(live);
    }
}

impl Drop for LinuxMemory {
    fn drop(&mut self) {
        let live = std::mem::take(self.live.get_mut().unwrap_or_else(|p| p.into_inner()));
        for (base, Live { len, .. }) in live {
            // SAFETY: each entry is a mapping this adapter created and never released; the
            // adapter is going away, so no call can reach it afterwards.
            unsafe { sys::munmap(base as *mut c_void, len as usize) };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::{c_int, c_long};

    use super::*;

    #[test]
    fn touch_keeps_the_bytes() {
        let os = LinuxMemory::new(Bytes(0));
        let page = os.page;
        let r = os.reserve(Bytes(4 * page), None, HugePages::Off).unwrap();
        // SAFETY: the range is a live read-write mapping of 4 pages that only this test uses.
        unsafe {
            r.base.write(7);
            r.base.add(page as usize).write(9);
            touch(r.base, (4 * page) as usize, page as usize);
            assert_eq!(r.base.read(), 7);
            assert_eq!(r.base.add(page as usize).read(), 9);
        }
        os.release(r);
    }

    #[test]
    fn bind_to_a_node_that_does_not_exist_fails() {
        let os = LinuxMemory::new(Bytes(0));
        let r = os.reserve(Bytes(os.page), None, HugePages::Off).unwrap();
        assert!(LinuxMemory::bind(r.base.cast(), os.page as usize, 1023).is_err());
        os.release(r);
    }

    #[test]
    fn c_int_and_c_long_are_the_lp64_sizes() {
        assert_eq!(size_of::<c_int>(), 4);
        assert_eq!(size_of::<c_long>(), 8);
        assert_eq!(size_of::<c_ulong>(), 8);
    }
}
