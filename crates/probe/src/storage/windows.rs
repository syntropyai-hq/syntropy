// SPDX-License-Identifier: MIT OR Apache-2.0

//! E3 on Windows: the volume calls (`syntropy_io::os::volume::WindowsVolume`) become a
//! `VolumeInfo`.
//!
//! Not read here, and left at their defaults: the drive's model and firmware, the controller's
//! PCI path and the chipset-lane rule (RFC-0010 §2.2 reads the SetupAPI parent chain for it), and
//! BitLocker status. They need calls PLC-E1-T06's row does not list.

use std::path::{Path, PathBuf};

use syntropy_core::resource::{DirectIoSupport, FsKind, StorageBus, TempSource, VolumeInfo};
use syntropy_core::units::Bytes;
use syntropy_io::os::volume::WindowsVolume;

use crate::error::EnumError;

/// `STORAGE_BUS_TYPE` values (`winioctl.h`).
mod bus {
    pub const ATA: u32 = 3;
    pub const USB: u32 = 7;
    pub const SATA: u32 = 11;
    pub const VIRTUAL: u32 = 14;
    pub const FILE_BACKED_VIRTUAL: u32 = 15;
    pub const NVME: u32 = 17;
}

/// Maps `raw` to a `VolumeInfo`.
///
/// Direct I/O (`FILE_FLAG_NO_BUFFERING`) needs transfers aligned to the logical sector, so
/// that size is both alignments. When the alignment query failed the volume is
/// `Unsupported`, which is "not known to work", never a measured refusal.
#[must_use]
pub fn classify(raw: &WindowsVolume) -> VolumeInfo {
    let logical = raw.logical_sector.unwrap_or(0);
    VolumeInfo {
        mount: PathBuf::from(&raw.root),
        fs: match raw.fs_name.to_ascii_uppercase().as_str() {
            "NTFS" => FsKind::Ntfs,
            "REFS" => FsKind::Refs,
            _ => FsKind::Other,
        },
        bus: match raw.bus_type {
            Some(bus::NVME) => StorageBus::Nvme,
            Some(bus::SATA | bus::ATA) => StorageBus::Sata,
            Some(bus::USB) => StorageBus::Usb,
            Some(bus::VIRTUAL | bus::FILE_BACKED_VIRTUAL) => StorageBus::Virtual,
            // SCSI (1), iSCSI (9) and SAS (10) land here too: `StorageBus` has no variant for
            // them yet (RFC-0010 §1.2), and a hosted Windows runner's system disk reports SAS.
            // Adding the variants is a follow-up to PLC-E1-T06.
            _ => StorageBus::Unknown,
        },
        pci: None,
        shared_lane: false,
        direct_io: if logical > 0 {
            DirectIoSupport::Supported {
                mem_align: logical,
                offset_align: logical,
            }
        } else {
            DirectIoSupport::Unsupported
        },
        logical_block: logical,
        physical_block: raw.physical_sector.unwrap_or(0),
        encrypted: raw.encrypted,
        free: Bytes(raw.free),
        temperature_source: if raw.temperature_query {
            TempSource::StorageQuery
        } else {
            TempSource::None
        },
        model: String::new(),
        firmware: String::new(),
    }
}

/// Describes the volume holding `path` on the live host.
///
/// # Errors
///
/// `UnsupportedOs` off Windows; `Os` when the volume cannot be read.
#[cfg(windows)]
pub fn live(path: &Path) -> Result<VolumeInfo, EnumError> {
    syntropy_io::os::volume::windows_volume(path)
        .map(|raw| classify(&raw))
        .map_err(|e| EnumError::Os {
            call: "GetVolumePathNameW / GetDiskFreeSpaceExW / GetVolumeInformationW".to_owned(),
            kind: e.kind(),
            message: e.to_string(),
        })
}

/// Describes the volume holding `path` on the live host.
///
/// # Errors
///
/// `UnsupportedOs` off Windows.
#[cfg(not(windows))]
pub fn live(_path: &Path) -> Result<VolumeInfo, EnumError> {
    Err(EnumError::UnsupportedOs)
}
