// SPDX-License-Identifier: MIT OR Apache-2.0

//! The system calls of E3 storage-volume enumeration (RFC-0010 §2.1 E3, §2.2 "NVMe volume";
//! PLC-E1-T06), behind safe functions that `probe` calls (RFC-0001 §4.2 rule 8, note of
//! 2026-10-22; OQ-PRC-26).
//!
//! | OS | Calls |
//! |---|---|
//! | Linux | `statfs`, `statx` with `STATX_DIOALIGN` (on an unnamed `O_TMPFILE` when the path is a directory, so nothing persists), through [`VolumeCalls`] |
//! | Windows | `GetVolumePathNameW`, `GetDiskFreeSpaceExW`, `GetVolumeInformationW`, `IOCTL_STORAGE_QUERY_PROPERTY` for the adapter bus type, access alignment, seek penalty and temperature (`windows_volume`) |
//! | macOS | `statfs`, then DiskArbitration's description of the volume's disk (`macos_volume`) |
//!
//! The record types below are plain data on every OS, so `probe` classifies them in code that
//! has no `unsafe` and runs on every OS. Every `unsafe` block carries a `// SAFETY:` comment with
//! its invariant (ADR-0001 item 4).

use std::io;
use std::path::Path;

/// What `statfs` reports about a Linux file system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FsStat {
    /// `f_type`, the file-system magic number.
    pub fs_type: u64,
    /// Bytes available to an unprivileged process: `f_bavail * f_bsize`.
    pub free: u64,
}

/// What `statx` reports about the file or directory on a volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatX {
    /// Major number of the device holding the file (`stx_dev_major`). 0 for a file system with
    /// no block device, such as tmpfs.
    pub dev_major: u32,
    /// Minor number (`stx_dev_minor`).
    pub dev_minor: u32,
    /// `(stx_dio_mem_align, stx_dio_offset_align)` when the kernel reports `STATX_DIOALIGN`
    /// and both are non-zero; `None` when it does not report them or reports zero, which
    /// means direct I/O is refused on the file.
    pub dio: Option<(u32, u32)>,
}

/// The raw Linux calls behind E3. [`LiveVolumeCalls`] makes them; tests inject a shim.
pub trait VolumeCalls {
    /// `statfs(path)`.
    ///
    /// # Errors
    ///
    /// The OS error, or `Unsupported` off Linux.
    fn statfs(&self, path: &Path) -> io::Result<FsStat>;

    /// `statx(path, STATX_DIOALIGN)`, and for a directory the same on an unnamed temporary
    /// file created in it.
    ///
    /// # Errors
    ///
    /// The OS error, or `Unsupported` off Linux.
    fn statx(&self, path: &Path) -> io::Result<StatX>;
}

/// The production [`VolumeCalls`].
#[derive(Clone, Copy, Debug, Default)]
pub struct LiveVolumeCalls;

#[cfg(any(target_os = "linux", target_os = "android"))]
impl VolumeCalls for LiveVolumeCalls {
    fn statfs(&self, path: &Path) -> io::Result<FsStat> {
        use std::os::unix::ffi::OsStrExt;
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a path holds a NUL"))?;
        // SAFETY: an all-zero `statfs` is a valid value of the plain-integer struct.
        let mut buf: libc::statfs = unsafe { std::mem::zeroed() };
        // SAFETY: `c_path` is NUL-terminated and valid for the call; `buf` is a valid
        // `statfs` the kernel fills.
        if unsafe { libc::statfs(c_path.as_ptr(), &mut buf) } != 0 {
            return Err(io::Error::last_os_error());
        }
        #[allow(clippy::unnecessary_cast, clippy::cast_sign_loss)]
        Ok(FsStat {
            fs_type: buf.f_type as u64,
            free: (buf.f_bavail as u64).saturating_mul(buf.f_bsize as u64),
        })
    }

    fn statx(&self, path: &Path) -> io::Result<StatX> {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        use std::os::unix::ffi::OsStrExt;
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a path holds a NUL"))?;
        let mask = libc::STATX_TYPE | libc::STATX_DIOALIGN;
        // SAFETY: an all-zero `statx` is a valid value of the plain-integer struct.
        let mut buf: libc::statx = unsafe { std::mem::zeroed() };
        // SAFETY: `c_path` is NUL-terminated and valid for the call; `buf` is a valid `statx`.
        let rc = unsafe { libc::statx(libc::AT_FDCWD, c_path.as_ptr(), 0, mask, &mut buf) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        let (dev_major, dev_minor) = (buf.stx_dev_major, buf.stx_dev_minor);
        let is_dir = u32::from(buf.stx_mode) & libc::S_IFMT == libc::S_IFDIR;
        if is_dir {
            // A directory has no direct-I/O alignment of its own. Ask a file in it: an
            // unnamed temporary file, which the kernel drops at close, so nothing persists.
            // SAFETY: as above; `O_TMPFILE` with a directory path creates no name.
            let fd = unsafe {
                libc::open(
                    c_path.as_ptr(),
                    libc::O_TMPFILE | libc::O_RDWR | libc::O_CLOEXEC,
                    0o600,
                )
            };
            if fd >= 0 {
                // SAFETY: `fd` is a new descriptor this function owns.
                let fd = unsafe { OwnedFd::from_raw_fd(fd) };
                // SAFETY: as above; `AT_EMPTY_PATH` with an empty name stats the descriptor.
                buf = unsafe { std::mem::zeroed() };
                // SAFETY: `fd` is open for the call, `c""` is NUL-terminated and `buf` is a
                // valid `statx`.
                let rc = unsafe {
                    libc::statx(
                        fd.as_raw_fd(),
                        c"".as_ptr(),
                        libc::AT_EMPTY_PATH,
                        mask,
                        &mut buf,
                    )
                };
                if rc != 0 {
                    return Ok(StatX {
                        dev_major,
                        dev_minor,
                        dio: None,
                    });
                }
            } else {
                return Ok(StatX {
                    dev_major,
                    dev_minor,
                    dio: None,
                });
            }
        }
        let reported = buf.stx_mask & libc::STATX_DIOALIGN != 0;
        let dio = (reported && buf.stx_dio_mem_align != 0 && buf.stx_dio_offset_align != 0)
            .then_some((buf.stx_dio_mem_align, buf.stx_dio_offset_align));
        Ok(StatX {
            dev_major,
            dev_minor,
            dio,
        })
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
impl VolumeCalls for LiveVolumeCalls {
    fn statfs(&self, _path: &Path) -> io::Result<FsStat> {
        Err(io::ErrorKind::Unsupported.into())
    }

    fn statx(&self, _path: &Path) -> io::Result<StatX> {
        Err(io::ErrorKind::Unsupported.into())
    }
}

/// What the Windows calls report about the volume holding a path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowsVolume {
    /// The volume's mount root, such as `C:\`.
    pub root: String,
    /// The file-system name from `GetVolumeInformationW`, such as `NTFS` or `ReFS`.
    pub fs_name: String,
    /// Bytes available to the caller (`GetDiskFreeSpaceExW`).
    pub free: u64,
    /// `STORAGE_ADAPTER_DESCRIPTOR::BusType`, when the query succeeded.
    pub bus_type: Option<u32>,
    /// `BytesPerLogicalSector` of `STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR`.
    pub logical_sector: Option<u32>,
    /// `BytesPerPhysicalSector` of the same descriptor.
    pub physical_sector: Option<u32>,
    /// `DEVICE_SEEK_PENALTY_DESCRIPTOR::IncursSeekPenalty`.
    pub seek_penalty: Option<bool>,
    /// Whether `StorageDeviceTemperatureProperty` answered.
    pub temperature_query: bool,
    /// The volume is BitLocker-protected, when readable.
    pub encrypted: Option<bool>,
}

/// What `statfs` and DiskArbitration report about the volume holding a path on macOS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacVolume {
    /// `f_mntonname`.
    pub mount: String,
    /// `f_fstypename`, such as `apfs`.
    pub fs_name: String,
    /// `f_mntfromname`, such as `/dev/disk3s1s1`.
    pub device: String,
    /// `f_bavail * f_bsize`.
    pub free: u64,
    /// `f_bsize`, the file system's block size.
    pub fs_block: u32,
    /// DiskArbitration `DADeviceProtocol`, such as `Apple Fabric` or `PCI-Express`.
    pub protocol: Option<String>,
    /// DiskArbitration `DAMediaBlockSize`.
    pub media_block: Option<u32>,
    /// DiskArbitration `DAMediaEncrypted`.
    pub encrypted: Option<bool>,
}

#[cfg(target_os = "macos")]
pub use crate::os::macos_volume::query as macos_volume;
#[cfg(windows)]
pub use crate::os::windows_volume::query as windows_volume;
