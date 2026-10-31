// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Windows system calls of E3 storage-volume enumeration (RFC-0010 §2.2 "NVMe volume";
//! PLC-E1-T06): the volume holding a path, its free space and file system, and the storage
//! properties of its device from `IOCTL_STORAGE_QUERY_PROPERTY` (adapter bus type, access
//! alignment, seek penalty, temperature).
//!
//! The volume is opened with access 0, which a standard user may do: the queries ask the
//! driver for descriptors and read no data. BitLocker status is not read here (the RFC says
//! "where readable"); `encrypted` is left unset.

use std::ffi::{OsStr, OsString};
use std::io;
use std::mem::{size_of, zeroed};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::Path;
use std::ptr;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, GetDiskFreeSpaceExW, GetVolumeInformationW,
    GetVolumePathNameW, OPEN_EXISTING,
};
use windows_sys::Win32::System::IO::DeviceIoControl;
use windows_sys::Win32::System::Ioctl::{
    DEVICE_SEEK_PENALTY_DESCRIPTOR, IOCTL_STORAGE_QUERY_PROPERTY, PropertyStandardQuery,
    STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR, STORAGE_ADAPTER_DESCRIPTOR, STORAGE_PROPERTY_QUERY,
    STORAGE_TEMPERATURE_DATA_DESCRIPTOR, StorageAccessAlignmentProperty, StorageAdapterProperty,
    StorageDeviceSeekPenaltyProperty, StorageDeviceTemperatureProperty,
};

use super::volume::WindowsVolume;

fn wide(s: &OsStr) -> Vec<u16> {
    s.encode_wide().chain(Some(0)).collect()
}

fn from_wide(buf: &[u16]) -> String {
    let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    OsString::from_wide(&buf[..end])
        .to_string_lossy()
        .into_owned()
}

/// A volume device handle, closed on drop.
struct Device(HANDLE);

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a handle `CreateFileW` returned, closed once here.
        unsafe { CloseHandle(self.0) };
    }
}

impl Device {
    /// Asks the driver for storage property `id` into `T`; `None` when the call fails.
    fn property<T: Copy>(&self, id: i32) -> Option<T> {
        let query = STORAGE_PROPERTY_QUERY {
            PropertyId: id,
            QueryType: PropertyStandardQuery,
            AdditionalParameters: [0],
        };
        // SAFETY: the descriptors are plain integer structs, valid when zeroed.
        let mut out: T = unsafe { zeroed() };
        let mut returned = 0u32;
        // SAFETY: `self.0` is an open handle. The input is `query`, whose size is passed; the
        // output is `out`, writable for `size_of::<T>()` bytes, which is passed. `returned` is
        // a valid `u32`, and no overlapped structure is used.
        let ok = unsafe {
            DeviceIoControl(
                self.0,
                IOCTL_STORAGE_QUERY_PROPERTY,
                (&raw const query).cast(),
                size_of::<STORAGE_PROPERTY_QUERY>() as u32,
                (&raw mut out).cast(),
                size_of::<T>() as u32,
                &mut returned,
                ptr::null_mut(),
            )
        };
        (ok != 0).then_some(out)
    }
}

/// Describes the volume holding `path`.
///
/// # Errors
///
/// When the volume root, its free space or its file-system name cannot be read. A device query
/// that fails leaves its field unset.
pub fn query(path: &Path) -> io::Result<WindowsVolume> {
    let name = wide(path.as_os_str());
    let mut root = [0u16; 261];
    // SAFETY: `name` is NUL-terminated; `root` is writable for the length passed.
    if unsafe { GetVolumePathNameW(name.as_ptr(), root.as_mut_ptr(), root.len() as u32) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let mut free = 0u64;
    // SAFETY: `root` is NUL-terminated; the other outputs may be null; `free` is a valid `u64`.
    if unsafe { GetDiskFreeSpaceExW(root.as_ptr(), &mut free, ptr::null_mut(), ptr::null_mut()) }
        == 0
    {
        return Err(io::Error::last_os_error());
    }
    let mut fs = [0u16; 64];
    // SAFETY: `root` is NUL-terminated; only the file-system name output is asked for, writable
    // for the length passed.
    if unsafe {
        GetVolumeInformationW(
            root.as_ptr(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            fs.as_mut_ptr(),
            fs.len() as u32,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    let root_text = from_wide(&root);
    let mut volume = WindowsVolume {
        root: root_text.clone(),
        fs_name: from_wide(&fs),
        free,
        bus_type: None,
        logical_sector: None,
        physical_sector: None,
        seek_penalty: None,
        temperature_query: false,
        encrypted: None,
    };
    // `C:\` -> `\\.\C:`, the volume device.
    let device_path = wide(OsStr::new(&format!(
        r"\\.\{}",
        root_text.trim_end_matches('\\')
    )));
    // SAFETY: `device_path` is NUL-terminated; access 0 reads no data; no security attributes
    // or template file are passed.
    let handle = unsafe {
        CreateFileW(
            device_path.as_ptr(),
            0,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            ptr::null(),
            OPEN_EXISTING,
            0,
            ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Ok(volume);
    }
    let device = Device(handle);
    volume.bus_type = device
        .property::<STORAGE_ADAPTER_DESCRIPTOR>(StorageAdapterProperty)
        .map(|d| u32::from(d.BusType));
    if let Some(a) =
        device.property::<STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR>(StorageAccessAlignmentProperty)
    {
        volume.logical_sector = Some(a.BytesPerLogicalSector);
        volume.physical_sector = Some(a.BytesPerPhysicalSector);
    }
    volume.seek_penalty = device
        .property::<DEVICE_SEEK_PENALTY_DESCRIPTOR>(StorageDeviceSeekPenaltyProperty)
        .map(|d| d.IncursSeekPenalty);
    volume.temperature_query = device
        .property::<STORAGE_TEMPERATURE_DATA_DESCRIPTOR>(StorageDeviceTemperatureProperty)
        .is_some();
    Ok(volume)
}
