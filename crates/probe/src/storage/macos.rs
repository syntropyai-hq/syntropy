// SPDX-License-Identifier: MIT OR Apache-2.0

//! E3 on macOS: `statfs` and DiskArbitration (`syntropy_io::os::volume::MacVolume`) become a
//! `VolumeInfo`.
//!
//! RFC-0010 §2.2 names IOKit's `IONVMeController` and `IOBlockStorageDevice` properties for the
//! drive's model and firmware. This reading does not fetch them, so `model` and `firmware` stay
//! empty; the bus, block size and encryption come from DiskArbitration. macOS has no public
//! temperature interface, so the source is `None`, and no PCI path is read.

#[cfg(target_os = "macos")]
use std::io;
use std::path::{Path, PathBuf};

use syntropy_core::resource::{DirectIoSupport, FsKind, StorageBus, TempSource, VolumeInfo};
use syntropy_core::units::Bytes;
use syntropy_io::os::volume::MacVolume;

use crate::error::EnumError;

/// Maps `raw` to a `VolumeInfo`.
#[must_use]
pub fn classify(raw: &MacVolume) -> VolumeInfo {
    let bus = match raw.protocol.as_deref() {
        // Apple's internal SSDs speak NVMe over Apple's own fabric.
        Some("Apple Fabric" | "PCI-Express") => StorageBus::Nvme,
        Some("USB") => StorageBus::Usb,
        Some("SATA" | "ATA") => StorageBus::Sata,
        Some("Virtual Interface" | "Disk Image") => StorageBus::Virtual,
        _ => StorageBus::Unknown,
    };
    let block = raw.media_block.unwrap_or(raw.fs_block);
    VolumeInfo {
        mount: PathBuf::from(&raw.mount),
        fs: match raw.fs_name.as_str() {
            "apfs" => FsKind::Apfs,
            _ => FsKind::Other,
        },
        bus,
        pci: None,
        shared_lane: false,
        // Uncached reads (`F_NOCACHE`) want block-sized, block-aligned transfers.
        direct_io: if block > 0 {
            DirectIoSupport::Supported {
                mem_align: block,
                offset_align: block,
            }
        } else {
            DirectIoSupport::Unsupported
        },
        logical_block: block,
        physical_block: block,
        encrypted: raw.encrypted,
        free: Bytes(raw.free),
        temperature_source: TempSource::None,
        model: String::new(),
        firmware: String::new(),
    }
}

/// Describes the volume holding `path` on the live host.
///
/// # Errors
///
/// `UnsupportedOs` off macOS; `Os` when `statfs` fails.
#[cfg(target_os = "macos")]
pub fn live(path: &Path) -> Result<VolumeInfo, EnumError> {
    syntropy_io::os::volume::macos_volume(path)
        .map(|raw| classify(&raw))
        .map_err(|e| os_error("statfs", &e))
}

/// Describes the volume holding `path` on the live host.
///
/// # Errors
///
/// `UnsupportedOs` off macOS.
#[cfg(not(target_os = "macos"))]
pub fn live(_path: &Path) -> Result<VolumeInfo, EnumError> {
    Err(EnumError::UnsupportedOs)
}

#[cfg(target_os = "macos")]
fn os_error(call: &str, e: &io::Error) -> EnumError {
    EnumError::Os {
        call: call.to_owned(),
        kind: e.kind(),
        message: e.to_string(),
    }
}
