// SPDX-License-Identifier: MIT OR Apache-2.0

//! The macOS system calls of E3 storage-volume enumeration (RFC-0010 §2.2 "NVMe volume";
//! PLC-E1-T06): `statfs` for the file system and free space, then DiskArbitration's
//! description of the volume's disk for the storage protocol and the media block size, and the
//! `kern.hv_vmm_present` sysctl, since a virtual machine's disk can report no protocol.
//!
//! RFC-0010 §2.2 names IOKit's `IONVMeController` and `IOBlockStorageDevice` properties. The
//! DiskArbitration description carries the same protocol and block-size facts for a BSD disk
//! name without walking the IOKit registry, so this module needs three more declarations and
//! none for the registry. It reads only; nothing here opens the disk.

use std::ffi::{CStr, CString, c_char};
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr;

use super::macos_host::{CFTypeRef, Owned, bool_value, dict_get, number_value, string_value};
use super::volume::MacVolume;

// SAFETY: the declarations of `<DiskArbitration/DASession.h>` and `DADisk.h`.
#[link(name = "DiskArbitration", kind = "framework")]
unsafe extern "C" {
    fn DASessionCreate(allocator: CFTypeRef) -> CFTypeRef;
    fn DADiskCreateFromBSDName(
        allocator: CFTypeRef,
        session: CFTypeRef,
        name: *const c_char,
    ) -> CFTypeRef;
    fn DADiskCopyDescription(disk: CFTypeRef) -> CFTypeRef;
}

fn text(field: &[c_char]) -> String {
    // SAFETY: the kernel NUL-terminates the fixed-size name arrays of `statfs`.
    unsafe { CStr::from_ptr(field.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Describes the volume holding `path`.
///
/// # Errors
///
/// When `statfs` fails. A DiskArbitration failure leaves `protocol`, `media_block` and
/// `encrypted` unset instead, because a network or disk-image volume has no such disk.
pub fn query(path: &Path) -> io::Result<MacVolume> {
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a path holds a NUL"))?;
    // SAFETY: an all-zero `statfs` is a valid value of the plain-integer struct.
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c_path` is NUL-terminated and valid for the call; `st` is a valid `statfs`.
    if unsafe { libc::statfs(c_path.as_ptr(), &mut st) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let device = text(&st.f_mntfromname);
    let mut volume = MacVolume {
        mount: text(&st.f_mntonname),
        fs_name: text(&st.f_fstypename),
        free: st.f_bavail.saturating_mul(u64::from(st.f_bsize)),
        fs_block: st.f_bsize,
        device: device.clone(),
        protocol: None,
        media_block: None,
        encrypted: None,
        // A 4-byte integer, nonzero in a virtual machine; absent on older kernels.
        hypervisor: super::macos_host::sysctl("kern.hv_vmm_present")
            .ok()
            .flatten()
            .is_some_and(|bytes| bytes.iter().any(|&b| b != 0)),
    };
    if let Some(bsd) = device.strip_prefix("/dev/")
        && let Ok(bsd) = CString::new(bsd)
    {
        // SAFETY: a null allocator is the default; the session is owned (Create rule).
        let session = Owned(unsafe { DASessionCreate(ptr::null()) });
        if !session.0.is_null() {
            // SAFETY: `session` is live and `bsd` NUL-terminated; the disk is owned.
            let disk =
                Owned(unsafe { DADiskCreateFromBSDName(ptr::null(), session.0, bsd.as_ptr()) });
            if !disk.0.is_null() {
                // SAFETY: `disk` is live; the description is owned (Copy rule).
                let desc = Owned(unsafe { DADiskCopyDescription(disk.0) });
                if !desc.0.is_null() {
                    // SAFETY: `desc` is a live CFDictionary, and each value is checked for
                    // its type before use.
                    unsafe {
                        volume.protocol = string_value(dict_get(desc.0, c"DADeviceProtocol"));
                        volume.media_block = number_value(dict_get(desc.0, c"DAMediaBlockSize"))
                            .and_then(|n| u32::try_from(n).ok());
                        volume.encrypted = bool_value(dict_get(desc.0, c"DAMediaEncrypted"));
                    }
                }
            }
        }
    }
    Ok(volume)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describes_the_volume_of_the_temp_dir() {
        let v = query(&std::env::temp_dir()).unwrap();
        assert!(v.free > 0, "{v:?}");
        assert!(!v.fs_name.is_empty() && v.fs_block > 0, "{v:?}");
        eprintln!("{v:?}");
    }
}
