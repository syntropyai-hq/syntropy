// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Windows alignment query (RFC-0021 §6).
//!
//! `GetFileInformationByHandleEx(FileStorageInfo)` reports
//! `PhysicalBytesPerSectorForPerformance`, which RFC-0021 §6 uses for offsets, lengths and
//! buffers. When that fails, the fallback opens the file's volume with no access rights and asks
//! `IOCTL_STORAGE_QUERY_PROPERTY` for `StorageAccessAlignmentProperty`, whose physical sector
//! size stands in for both.
//!
//! The workspace has no C bindings crate, so the `kernel32` functions are declared here.

use std::ffi::c_void;
use std::fs::File;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::AsRawHandle;
use std::path::Path;

use syntropy_core::resource::DioAlign;

use super::{DioQuery, DioSource};

mod sys {
    use std::ffi::c_void;

    /// `FILE_INFO_BY_HANDLE_CLASS::FileStorageInfo`.
    pub const FILE_STORAGE_INFO_CLASS: i32 = 16;
    /// `CTL_CODE(IOCTL_STORAGE_BASE, 0x500, METHOD_BUFFERED, FILE_ANY_ACCESS)`.
    pub const IOCTL_STORAGE_QUERY_PROPERTY: u32 = 0x002D_1400;
    /// `STORAGE_PROPERTY_ID::StorageAccessAlignmentProperty`.
    pub const STORAGE_ACCESS_ALIGNMENT_PROPERTY: i32 = 6;
    /// `STORAGE_QUERY_TYPE::PropertyStandardQuery`.
    pub const PROPERTY_STANDARD_QUERY: i32 = 0;
    pub const FILE_SHARE_READ: u32 = 1;
    pub const FILE_SHARE_WRITE: u32 = 2;
    pub const OPEN_EXISTING: u32 = 3;
    pub const INVALID_HANDLE_VALUE: isize = -1;

    /// `FILE_STORAGE_INFO`.
    #[repr(C)]
    #[derive(Default)]
    pub struct FileStorageInfo {
        pub logical_bytes_per_sector: u32,
        pub physical_bytes_per_sector_for_atomicity: u32,
        pub physical_bytes_per_sector_for_performance: u32,
        pub fs_effective_physical_bytes_per_sector_for_atomicity: u32,
        pub flags: u32,
        pub byte_offset_for_sector_alignment: u32,
        pub byte_offset_for_partition_alignment: u32,
    }

    /// `STORAGE_PROPERTY_QUERY`.
    #[repr(C)]
    pub struct StoragePropertyQuery {
        pub property_id: i32,
        pub query_type: i32,
        pub additional: [u8; 4],
    }

    /// `STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR`.
    #[repr(C)]
    #[derive(Default)]
    pub struct AccessAlignment {
        pub version: u32,
        pub size: u32,
        pub bytes_per_cache_line: u32,
        pub bytes_offset_for_cache_alignment: u32,
        pub bytes_per_logical_sector: u32,
        pub bytes_per_physical_sector: u32,
        pub bytes_offset_for_sector_alignment: u32,
    }

    /// `SYSTEM_INFO`.
    #[repr(C)]
    pub struct SystemInfo {
        pub processor_architecture: u16,
        pub reserved: u16,
        pub page_size: u32,
        pub min_app_address: *mut c_void,
        pub max_app_address: *mut c_void,
        pub active_processor_mask: usize,
        pub number_of_processors: u32,
        pub processor_type: u32,
        pub allocation_granularity: u32,
        pub processor_level: u16,
        pub processor_revision: u16,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn GetFileInformationByHandleEx(
            file: *mut c_void,
            class: i32,
            info: *mut c_void,
            size: u32,
        ) -> i32;
        pub fn GetVolumePathNameW(path: *const u16, out: *mut u16, len: u32) -> i32;
        pub fn GetVolumeNameForVolumeMountPointW(mount: *const u16, out: *mut u16, len: u32)
        -> i32;
        pub fn CreateFileW(
            name: *const u16,
            access: u32,
            share: u32,
            security: *mut c_void,
            disposition: u32,
            flags: u32,
            template: *mut c_void,
        ) -> *mut c_void;
        pub fn DeviceIoControl(
            device: *mut c_void,
            code: u32,
            input: *const c_void,
            input_len: u32,
            output: *mut c_void,
            output_len: u32,
            returned: *mut u32,
            overlapped: *mut c_void,
        ) -> i32;
        pub fn CloseHandle(handle: *mut c_void) -> i32;
        pub fn GetSystemInfo(info: *mut SystemInfo);
    }
}

/// See the module documentation.
pub(super) fn query(file: &File, path: &Path) -> DioQuery {
    if let Some(n) = file_storage_info(file) {
        return aligned(n, DioSource::FileStorageInfo);
    }
    match access_alignment(path) {
        Some(n) => aligned(n, DioSource::StorageAccessAlignment),
        None => DioQuery::Unknown,
    }
}

fn aligned(n: u32, source: DioSource) -> DioQuery {
    DioQuery::Aligned {
        dio: DioAlign { mem: n, offset: n },
        source,
    }
}

/// `FileStorageInfo`'s `PhysicalBytesPerSectorForPerformance`.
fn file_storage_info(file: &File) -> Option<u32> {
    let mut info = sys::FileStorageInfo::default();
    // SAFETY: `info` is a writable `FILE_STORAGE_INFO` of the size passed; the handle is open
    // for the call.
    let ok = unsafe {
        sys::GetFileInformationByHandleEx(
            file.as_raw_handle(),
            sys::FILE_STORAGE_INFO_CLASS,
            (&raw mut info).cast::<c_void>(),
            std::mem::size_of::<sys::FileStorageInfo>() as u32,
        )
    };
    (ok != 0)
        .then_some(info.physical_bytes_per_sector_for_performance)
        .filter(|&n| n > 0)
}

fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
    s.encode_wide().chain(std::iter::once(0)).collect()
}

/// The physical sector size of the volume holding `path`, from
/// `StorageAccessAlignmentProperty`.
fn access_alignment(path: &Path) -> Option<u32> {
    const LEN: usize = 1024;
    let path = wide(path.as_os_str());
    let mut mount = vec![0u16; LEN];
    // SAFETY: `path` is NUL-terminated; `mount` is writable for `LEN` UTF-16 units, as passed.
    if unsafe { sys::GetVolumePathNameW(path.as_ptr(), mount.as_mut_ptr(), LEN as u32) } == 0 {
        return None;
    }
    let mut volume = vec![0u16; LEN];
    // SAFETY: `mount` was NUL-terminated by the call above; `volume` is writable for `LEN`
    // units, as passed.
    let ok = unsafe {
        sys::GetVolumeNameForVolumeMountPointW(mount.as_ptr(), volume.as_mut_ptr(), LEN as u32)
    };
    if ok == 0 {
        return None;
    }
    // `\\?\Volume{GUID}\` names the root directory; the device is the name without the slash.
    let end = volume.iter().position(|&c| c == 0)?;
    if end == 0 {
        return None;
    }
    if volume[end - 1] == u16::from(b'\\') {
        volume[end - 1] = 0;
    }
    // SAFETY: `volume` is NUL-terminated; the other pointers are null, which CreateFileW
    // accepts for no security attributes and no template.
    let device = unsafe {
        sys::CreateFileW(
            volume.as_ptr(),
            0,
            sys::FILE_SHARE_READ | sys::FILE_SHARE_WRITE,
            std::ptr::null_mut(),
            sys::OPEN_EXISTING,
            0,
            std::ptr::null_mut(),
        )
    };
    if device as isize == sys::INVALID_HANDLE_VALUE || device.is_null() {
        return None;
    }
    let q = sys::StoragePropertyQuery {
        property_id: sys::STORAGE_ACCESS_ALIGNMENT_PROPERTY,
        query_type: sys::PROPERTY_STANDARD_QUERY,
        additional: [0; 4],
    };
    let mut out = sys::AccessAlignment::default();
    let mut returned = 0u32;
    // SAFETY: `q` is readable and `out` writable for the sizes passed; `returned` outlives the
    // synchronous call; `device` is the open handle from above.
    let ok = unsafe {
        sys::DeviceIoControl(
            device,
            sys::IOCTL_STORAGE_QUERY_PROPERTY,
            (&raw const q).cast::<c_void>(),
            std::mem::size_of::<sys::StoragePropertyQuery>() as u32,
            (&raw mut out).cast::<c_void>(),
            std::mem::size_of::<sys::AccessAlignment>() as u32,
            &raw mut returned,
            std::ptr::null_mut(),
        )
    };
    // SAFETY: `device` is a handle this function opened and closes once.
    unsafe { sys::CloseHandle(device) };
    (ok != 0)
        .then_some(out.bytes_per_physical_sector)
        .filter(|&n| n > 0)
}

/// The page size, from `GetSystemInfo`.
pub(super) fn page_size() -> u64 {
    let mut info = std::mem::MaybeUninit::<sys::SystemInfo>::zeroed();
    // SAFETY: `GetSystemInfo` fills the whole `SYSTEM_INFO` it is given, which is writable.
    unsafe { sys::GetSystemInfo(info.as_mut_ptr()) };
    // SAFETY: zero-initialised, then filled by the call; every field is plain data.
    let info = unsafe { info.assume_init() };
    match u64::from(info.page_size) {
        0 => super::FLOOR,
        n => n,
    }
}
