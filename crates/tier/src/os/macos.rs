// SPDX-License-Identifier: MIT OR Apache-2.0

//! macOS sysctl reads for `syntropy tune`'s tier side (RFC-0020 §9.1; TIER-E2-T13).
//!
//! Every call here is a read: `sysctlbyname` with no new value. RFC-0020 §9.1 says the probe and
//! the tier manager only read `iogpu.wired_limit_mb`. The one write is
//! [`HostOp::SetWiredLimit`](crate::tune::HostOp::SetWiredLimit), which the caller's privileged
//! runner executes. The rest of the macOS adapter, with anonymous `mmap`, `mlock` and the wire
//! limits, is TIER-E2-T04's.
//!
//! `sysctlbyname` is declared here because the workspace has no C bindings crate, as in the
//! Linux adapter.

#![allow(unsafe_code)]

use std::ffi::{CString, c_char, c_int, c_void};
use std::io;

use crate::tune::{WIRED_LIMIT_SYSCTL, WiredLimitSource};

// SAFETY: the declaration of `<sys/sysctl.h>` in libSystem, which every macOS process links.
unsafe extern "C" {
    fn sysctlbyname(
        name: *const c_char,
        oldp: *mut c_void,
        oldlenp: *mut usize,
        newp: *mut c_void,
        newlen: usize,
    ) -> c_int;
}

/// Reads the integer sysctl `name`, 4 or 8 bytes wide, without writing it.
///
/// # Errors
///
/// The OS's error, such as `ENOENT` where the sysctl does not exist on this host, or
/// `InvalidData` for a value that is not 4 or 8 bytes wide.
pub fn sysctl_u64(name: &str) -> io::Result<u64> {
    let name =
        CString::new(name).map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let mut value = [0u8; 8];
    let mut len = value.len();
    // SAFETY: `name` is NUL-terminated and outlives the call; `value` is writable for `len`
    // bytes, and the call writes at most `len` bytes there and the length it wrote into `len`.
    // The new-value pointer is null and its length 0, so the call changes nothing.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            value.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    match len {
        4 => Ok(u64::from(u32::from_ne_bytes([
            value[0], value[1], value[2], value[3],
        ]))),
        8 => Ok(u64::from_ne_bytes(value)),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("a {other}-byte sysctl value, not an integer"),
        )),
    }
}

/// The GPU wired limit, read with `sysctlbyname`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SysctlWiredLimit;

impl WiredLimitSource for SysctlWiredLimit {
    fn read_mb(&self) -> io::Result<u64> {
        sysctl_u64(WIRED_LIMIT_SYSCTL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `hw.memsize` exists on every Mac, as an 8-byte value.
    #[test]
    fn a_sysctl_reads_without_a_write() {
        let ram = sysctl_u64("hw.memsize").unwrap();
        assert!(ram >= 1 << 30, "{ram}");
        let missing = sysctl_u64("syntropy.no_such_sysctl").unwrap_err();
        assert_eq!(missing.raw_os_error(), Some(2), "ENOENT: {missing}");
        assert_eq!(
            sysctl_u64("bad\0name").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
}
