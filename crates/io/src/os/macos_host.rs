// SPDX-License-Identifier: MIT OR Apache-2.0

//! The macOS system calls of E1 host enumeration (RFC-0010 §2.2; PLC-E1-T03), behind safe
//! functions that `probe` calls (RFC-0001 §4.2 rule 8, note of 2026-10-22; OQ-PRC-26).
//!
//! | Function | Call |
//! |---|---|
//! | [`sysctl`] | `sysctlbyname` with no new value: a read, never a write |
//! | [`vm_statistics`] | `host_statistics64(HOST_VM_INFO64)` and the page size |
//! | [`memlock_limit`] | `getrlimit(RLIMIT_MEMLOCK)` |
//! | [`power_sources`] | IOKit `IOPSCopyPowerSourcesInfo`, `IOPSCopyPowerSourcesList`, `IOPSGetPowerSourceDescription`, `IOPSGetProvidingPowerSourceType` |
//!
//! There is no function that writes a sysctl. `iogpu.wired_limit_mb` changes only through
//! `syntropy tune` (ADR-0014, RFC-0020 §9.1), which is not built.
//!
//! Every `unsafe` block carries a `// SAFETY:` comment with its invariant (ADR-0001 item 4).

use std::ffi::{CStr, CString, c_char, c_void};
use std::io;
use std::ptr;

/// Reads the sysctl `name` with `sysctlbyname`, passing no new value, so the call cannot
/// change it. `Ok(None)` when the kernel has no such name (`ENOENT`); an empty vector when the
/// name exists with an empty value.
///
/// The bytes are the value as the kernel returns it: an integer in native byte order (4 or 8
/// bytes) or a NUL-terminated string. The caller knows which a name holds.
///
/// # Errors
///
/// `InvalidData` when `name` holds a NUL byte; otherwise the OS error when the call fails for
/// any reason but `ENOENT` (`EINVAL` maps to `InvalidInput`).
pub fn sysctl(name: &str) -> io::Result<Option<Vec<u8>>> {
    // A name with a NUL is the caller's mistake, not the kernel's `EINVAL`: a distinct kind.
    let c_name = CString::new(name)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "a sysctl name holds a NUL"))?;
    // A value may grow between the size query and the read (a string sysctl being set), so
    // retry a few times on ENOMEM.
    for _ in 0..4 {
        let mut len: libc::size_t = 0;
        // SAFETY: `c_name` is a NUL-terminated string valid for the call. A null `oldp` asks
        // only for the size, which the kernel writes to `len`, a valid `size_t`. `newp` is null
        // and `newlen` 0, so nothing is written to the sysctl.
        let rc = unsafe {
            libc::sysctlbyname(
                c_name.as_ptr(),
                ptr::null_mut(),
                &mut len,
                ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            let err = io::Error::last_os_error();
            return match err.raw_os_error() {
                Some(libc::ENOENT) => Ok(None),
                _ => Err(err),
            };
        }
        // A name whose value is empty (some sysctls in a virtual machine) has nothing to
        // read; a second call with a zero-length buffer would fail with EINVAL.
        if len == 0 {
            return Ok(Some(Vec::new()));
        }
        let mut buf = vec![0u8; len];
        let mut got = len;
        // SAFETY: `buf` is writable for `got` bytes, and the kernel writes at most `got` bytes
        // to it and the length it wrote back to `got`. `newp` is null and `newlen` 0: a read.
        let rc = unsafe {
            libc::sysctlbyname(
                c_name.as_ptr(),
                buf.as_mut_ptr().cast(),
                &mut got,
                ptr::null_mut(),
                0,
            )
        };
        if rc == 0 {
            buf.truncate(got);
            return Ok(Some(buf));
        }
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::ENOMEM) => {}
            Some(libc::ENOENT) => return Ok(None),
            _ => return Err(err),
        }
    }
    Err(io::Error::other(format!(
        "sysctl {name} kept changing size"
    )))
}

/// Page counts from `host_statistics64(HOST_VM_INFO64)`, with the page size they count in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VmStatistics {
    /// Bytes per page.
    pub page_size: u64,
    /// `free_count`.
    pub free: u64,
    /// `active_count`.
    pub active: u64,
    /// `inactive_count`.
    pub inactive: u64,
    /// `wire_count`.
    pub wired: u64,
    /// `purgeable_count`.
    pub purgeable: u64,
    /// `speculative_count`.
    pub speculative: u64,
    /// `compressor_page_count`.
    pub compressor: u64,
}

// SAFETY: these are the declarations of `<mach/mach_init.h>`, `<mach/mach_port.h>` and
// `<mach/mach_traps.h>` in libSystem, which every macOS process links. `mach_host_self` takes
// no arguments and returns a send right; it is declared safe. (`libc` marks its own
// declaration of it deprecated in favour of another crate, which RFC-0001 §4.2 does not add.)
unsafe extern "C" {
    safe fn mach_host_self() -> libc::mach_port_t;
    fn mach_port_deallocate(task: libc::mach_port_t, name: libc::mach_port_t) -> i32;
    static mach_task_self_: libc::mach_port_t;
}

/// Reads the host's virtual-memory page counts.
///
/// # Errors
///
/// When `host_statistics64` returns a Mach error, or the page size is unavailable.
pub fn vm_statistics() -> io::Result<VmStatistics> {
    // SAFETY: `libc::vm_statistics64` is a plain-data C struct, for which all-zero bytes are a
    // valid value.
    let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let host = mach_host_self();
    // SAFETY: `stats` is writable for `HOST_VM_INFO64_COUNT` integers, the count passed in
    // `count`, and the kernel writes at most `count` integers and the number it wrote back to
    // `count`. `host` is this task's send right to the host port.
    let kr = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&raw mut stats).cast(),
            &mut count,
        )
    };
    // SAFETY: `mach_task_self_` is set by libSystem before `main` and never changed; reading
    // it is a plain load. Releasing the reference `mach_host_self` added keeps the port's
    // reference count balanced; its result is not needed.
    unsafe {
        mach_port_deallocate(mach_task_self_, host);
    }
    if kr != 0 {
        return Err(io::Error::other(format!(
            "host_statistics64 returned kern_return_t {kr}"
        )));
    }
    // SAFETY: `sysconf` reads a configuration value and touches no caller memory.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size = u64::try_from(page)
        .ok()
        .filter(|p| *p > 0)
        .ok_or_else(io::Error::last_os_error)?;
    // Copy the fields out first: the struct is `packed(8)`, so no reference to a field is
    // taken.
    let s = stats;
    Ok(VmStatistics {
        page_size,
        free: u64::from(s.free_count),
        active: u64::from(s.active_count),
        inactive: u64::from(s.inactive_count),
        wired: u64::from(s.wire_count),
        purgeable: u64::from(s.purgeable_count),
        speculative: u64::from(s.speculative_count),
        compressor: u64::from(s.compressor_page_count),
    })
}

/// This process's `RLIMIT_MEMLOCK` soft limit; `None` when it is unlimited.
///
/// # Errors
///
/// When `getrlimit` fails.
pub fn memlock_limit() -> io::Result<Option<u64>> {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `lim` is a valid, writable `rlimit` for the call, and `getrlimit` writes only
    // through it.
    let rc = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut lim) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((lim.rlim_cur != libc::RLIM_INFINITY).then_some(lim.rlim_cur))
}

/// One power source's description dictionary, the keys E1 reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PowerSourceDesc {
    /// `Type`: `InternalBattery` for the system battery, `UPS` for an uninterruptible supply.
    pub kind: Option<String>,
    /// `Power Source State`: `AC Power`, `Battery Power` or `Off Line`.
    pub state: Option<String>,
    /// `Current Capacity`.
    pub current_capacity: Option<i64>,
    /// `Max Capacity`.
    pub max_capacity: Option<i64>,
    /// `Is Present`.
    pub present: Option<bool>,
}

/// What IOKit's power-source API reports.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PowerSources {
    /// `IOPSGetProvidingPowerSourceType`: `AC Power`, `Battery Power` or `UPS Power`.
    pub providing: Option<String>,
    /// Every power source, in the order IOKit lists them.
    pub sources: Vec<PowerSourceDesc>,
}

pub(super) type CFTypeRef = *const c_void;
type CFIndex = isize;
type CFTypeID = usize;

const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
const K_CF_NUMBER_SINT64_TYPE: CFIndex = 4;

// SAFETY: the declarations of `<CoreFoundation/CFBase.h>`, `CFArray.h`, `CFDictionary.h`,
// `CFString.h` and `CFNumber.h`, with the C types they use.
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: CFTypeRef);
    fn CFGetTypeID(cf: CFTypeRef) -> CFTypeID;
    safe fn CFStringGetTypeID() -> CFTypeID;
    safe fn CFNumberGetTypeID() -> CFTypeID;
    safe fn CFBooleanGetTypeID() -> CFTypeID;
    fn CFArrayGetCount(array: CFTypeRef) -> CFIndex;
    fn CFArrayGetValueAtIndex(array: CFTypeRef, idx: CFIndex) -> CFTypeRef;
    fn CFDictionaryGetValue(dict: CFTypeRef, key: CFTypeRef) -> CFTypeRef;
    fn CFStringCreateWithCString(alloc: CFTypeRef, s: *const c_char, encoding: u32) -> CFTypeRef;
    fn CFStringGetCString(s: CFTypeRef, buf: *mut c_char, size: CFIndex, encoding: u32) -> u8;
    fn CFNumberGetValue(n: CFTypeRef, ty: CFIndex, out: *mut c_void) -> u8;
    fn CFBooleanGetValue(b: CFTypeRef) -> u8;
}

// SAFETY: the declarations of `<IOKit/ps/IOPowerSources.h>`.
#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOPSCopyPowerSourcesInfo() -> CFTypeRef;
    fn IOPSCopyPowerSourcesList(blob: CFTypeRef) -> CFTypeRef;
    fn IOPSGetPowerSourceDescription(blob: CFTypeRef, ps: CFTypeRef) -> CFTypeRef;
    fn IOPSGetProvidingPowerSourceType(snapshot: CFTypeRef) -> CFTypeRef;
}

/// A CoreFoundation object this code owns (from a Create or Copy call), released on drop.
pub(super) struct Owned(pub(super) CFTypeRef);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: `self.0` came from a Create or Copy function, so this code holds one
            // reference to it, released exactly once here.
            unsafe { CFRelease(self.0) };
        }
    }
}

/// A CFString holding `s`.
pub(super) fn cf_string(s: &CStr) -> Owned {
    // SAFETY: `s` is NUL-terminated and valid for the call; a null allocator is the default
    // allocator. The result is owned (Create rule).
    Owned(unsafe { CFStringCreateWithCString(ptr::null(), s.as_ptr(), K_CF_STRING_ENCODING_UTF8) })
}

/// The UTF-8 text of `s`, which must be a live CFString or null.
///
/// # Safety
///
/// `s` is null or a valid CoreFoundation object for the duration of the call.
pub(super) unsafe fn string_value(s: CFTypeRef) -> Option<String> {
    // SAFETY: `s` is a valid object by the caller's contract; `CFGetTypeID` reads its type.
    if s.is_null() || unsafe { CFGetTypeID(s) } != CFStringGetTypeID() {
        return None;
    }
    let mut buf = [0 as c_char; 256];
    // SAFETY: `s` is a CFString, checked above. `buf` is writable for its length, which is
    // passed, and the call writes a NUL-terminated string of at most that many bytes.
    let ok = unsafe {
        CFStringGetCString(
            s,
            buf.as_mut_ptr(),
            buf.len() as CFIndex,
            K_CF_STRING_ENCODING_UTF8,
        )
    };
    if ok == 0 {
        return None;
    }
    // SAFETY: the call succeeded, so `buf` holds a NUL-terminated string.
    let text = unsafe { CStr::from_ptr(buf.as_ptr()) };
    Some(text.to_string_lossy().into_owned())
}

/// The value under `key` in the live CFDictionary `dict`, not owned.
///
/// # Safety
///
/// `dict` is a valid CFDictionary for the duration of the call.
pub(super) unsafe fn dict_get(dict: CFTypeRef, key: &CStr) -> CFTypeRef {
    let k = cf_string(key);
    if k.0.is_null() {
        return ptr::null();
    }
    // SAFETY: `dict` is a dictionary by the caller's contract and `k` a live CFString. The
    // value is borrowed from `dict` (Get rule).
    unsafe { CFDictionaryGetValue(dict, k.0) }
}

/// Reads the power sources IOKit reports.
///
/// # Errors
///
/// When IOKit returns no power-source snapshot.
pub fn power_sources() -> io::Result<PowerSources> {
    // SAFETY: takes no arguments; the snapshot is owned (Copy rule).
    let blob = Owned(unsafe { IOPSCopyPowerSourcesInfo() });
    if blob.0.is_null() {
        return Err(io::Error::other(
            "IOPSCopyPowerSourcesInfo returned no snapshot",
        ));
    }
    // SAFETY: `blob` is a live snapshot. The type string is borrowed from it (Get rule), and
    // `string_value` accepts null.
    let providing = unsafe { string_value(IOPSGetProvidingPowerSourceType(blob.0)) };
    // SAFETY: `blob` is a live snapshot; the list is owned (Copy rule).
    let list = Owned(unsafe { IOPSCopyPowerSourcesList(blob.0) });
    let mut sources = Vec::new();
    if !list.0.is_null() {
        // SAFETY: `list` is a live CFArray.
        let n = unsafe { CFArrayGetCount(list.0) };
        for i in 0..n {
            // SAFETY: `i` is in `0..count`. The element and its description are borrowed from
            // `list` and `blob`, which outlive this loop.
            let desc =
                unsafe { IOPSGetPowerSourceDescription(blob.0, CFArrayGetValueAtIndex(list.0, i)) };
            if desc.is_null() {
                continue;
            }
            // SAFETY: `desc` is a live CFDictionary for the rest of this iteration, and each
            // value read from it is checked for its type before use.
            unsafe {
                sources.push(PowerSourceDesc {
                    kind: string_value(dict_get(desc, c"Type")),
                    state: string_value(dict_get(desc, c"Power Source State")),
                    current_capacity: number_value(dict_get(desc, c"Current Capacity")),
                    max_capacity: number_value(dict_get(desc, c"Max Capacity")),
                    present: bool_value(dict_get(desc, c"Is Present")),
                });
            }
        }
    }
    Ok(PowerSources { providing, sources })
}

/// The value of the live CFNumber `n` as an `i64`.
///
/// # Safety
///
/// `n` is null or a valid CoreFoundation object for the duration of the call.
pub(super) unsafe fn number_value(n: CFTypeRef) -> Option<i64> {
    // SAFETY: `n` is a valid object by the caller's contract.
    if n.is_null() || unsafe { CFGetTypeID(n) } != CFNumberGetTypeID() {
        return None;
    }
    let mut v: i64 = 0;
    // SAFETY: `n` is a CFNumber, checked above, and `v` is a writable `i64`, the size of
    // `kCFNumberSInt64Type`.
    let ok = unsafe { CFNumberGetValue(n, K_CF_NUMBER_SINT64_TYPE, (&raw mut v).cast()) };
    (ok != 0).then_some(v)
}

/// The value of the live CFBoolean `b`.
///
/// # Safety
///
/// `b` is null or a valid CoreFoundation object for the duration of the call.
pub(super) unsafe fn bool_value(b: CFTypeRef) -> Option<bool> {
    // SAFETY: `b` is a valid object by the caller's contract.
    if b.is_null() || unsafe { CFGetTypeID(b) } != CFBooleanGetTypeID() {
        return None;
    }
    // SAFETY: `b` is a CFBoolean, checked above.
    Some(unsafe { CFBooleanGetValue(b) } != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_integer_and_string_sysctls() {
        let mem = sysctl("hw.memsize").unwrap().unwrap();
        assert_eq!(mem.len(), 8);
        assert!(u64::from_ne_bytes(mem.try_into().unwrap()) > 0);
        let brand = sysctl("machdep.cpu.brand_string").unwrap().unwrap();
        assert_eq!(brand.last(), Some(&0));
        assert_eq!(sysctl("hw.no_such_sysctl_name").unwrap(), None);
        // A NUL in the name is the caller's error, distinct from the kernel's EINVAL.
        assert_eq!(
            sysctl("bad\0name").unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn vm_statistics_counts_pages() {
        let vm = vm_statistics().unwrap();
        assert!(vm.page_size >= 4096);
        assert!(vm.free + vm.active + vm.inactive + vm.wired > 0);
    }

    #[test]
    fn power_sources_returns_a_snapshot() {
        let ps = power_sources().unwrap();
        assert!(ps.providing.is_some());
    }

    #[test]
    fn memlock_limit_reads() {
        memlock_limit().unwrap();
    }
}
