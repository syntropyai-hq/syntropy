// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLC-E1-T06 on Windows and macOS: the classifiers over synthetic responses, and on the matching
//! hosted runner the live system volume. The synthetic values are made up, not recorded.

use std::path::{Path, PathBuf};

use syntropy_core::resource::{DirectIoSupport, FsKind, StorageBus, TempSource};
use syntropy_io::os::volume::{MacVolume, WindowsVolume};
use syntropy_probe::EnumError;
use syntropy_probe::storage::{classify_macos, classify_windows, macos_live, windows_live};

/// `STORAGE_BUS_TYPE` SCSI (1), iSCSI (9) and SAS (10), which `StorageBus` has no variant for.
const UNMODELLED_BUS_TYPES: [u32; 3] = [1, 9, 10];

fn windows(bus: Option<u32>) -> WindowsVolume {
    WindowsVolume {
        root: r"C:\".to_owned(),
        fs_name: "NTFS".to_owned(),
        free: 123 << 30,
        bus_type: bus,
        logical_sector: Some(512),
        physical_sector: Some(4096),
        seek_penalty: Some(false),
        temperature_query: true,
        encrypted: None,
    }
}

#[test]
fn windows_nvme_reports_bus_and_alignment() {
    let v = classify_windows(&windows(Some(17)));
    assert_eq!(v.bus, StorageBus::Nvme);
    assert_eq!(v.fs, FsKind::Ntfs);
    assert_eq!(v.mount, PathBuf::from(r"C:\"));
    assert_eq!(
        v.direct_io,
        DirectIoSupport::Supported {
            mem_align: 512,
            offset_align: 512
        }
    );
    assert_eq!((v.logical_block, v.physical_block), (512, 4096));
    assert_eq!(v.temperature_source, TempSource::StorageQuery);
}

#[test]
fn windows_bus_types_map() {
    for (raw, want) in [
        (Some(11), StorageBus::Sata),
        (Some(3), StorageBus::Sata),
        (Some(7), StorageBus::Usb),
        (Some(14), StorageBus::Virtual),
        (None, StorageBus::Unknown),
    ] {
        assert_eq!(classify_windows(&windows(raw)).bus, want, "{raw:?}");
    }
}

/// SCSI, iSCSI and SAS have no `StorageBus` variant yet (RFC-0010 §1.2), so they classify as
/// `Unknown`. A follow-up to PLC-E1-T06 adds the variants and changes these expectations.
#[test]
fn windows_unmodelled_bus_types_are_unknown() {
    for raw in UNMODELLED_BUS_TYPES {
        assert_eq!(
            classify_windows(&windows(Some(raw))).bus,
            StorageBus::Unknown,
            "{raw}"
        );
    }
}

#[test]
fn windows_without_an_alignment_answer_is_not_assumed_aligned() {
    let mut raw = windows(Some(17));
    raw.logical_sector = None;
    raw.temperature_query = false;
    let v = classify_windows(&raw);
    assert_eq!(v.direct_io, DirectIoSupport::Unsupported);
    assert_eq!(v.temperature_source, TempSource::None);
}

fn mac(protocol: Option<&str>) -> MacVolume {
    MacVolume {
        mount: "/".to_owned(),
        fs_name: "apfs".to_owned(),
        device: "/dev/disk9s1".to_owned(),
        free: 1 << 36,
        fs_block: 4096,
        protocol: protocol.map(str::to_owned),
        media_block: Some(4096),
        encrypted: Some(true),
        hypervisor: false,
    }
}

#[test]
fn macos_apple_fabric_is_nvme() {
    let v = classify_macos(&mac(Some("Apple Fabric")));
    assert_eq!(v.bus, StorageBus::Nvme);
    assert_eq!(v.fs, FsKind::Apfs);
    assert_eq!(
        v.direct_io,
        DirectIoSupport::Supported {
            mem_align: 4096,
            offset_align: 4096
        }
    );
    assert_eq!(v.encrypted, Some(true));
    assert_eq!(v.temperature_source, TempSource::None);
}

#[test]
fn macos_protocols_map() {
    for (raw, want) in [
        (Some("USB"), StorageBus::Usb),
        (Some("SATA"), StorageBus::Sata),
        (Some("Disk Image"), StorageBus::Virtual),
        (Some("Thunderbolt"), StorageBus::Unknown),
        (None, StorageBus::Unknown),
    ] {
        assert_eq!(classify_macos(&mac(raw)).bus, want, "{raw:?}");
    }
}

#[test]
fn a_virtual_machines_disk_without_a_protocol_is_virtual() {
    // The hosted macOS runner's system volume: no `DADeviceProtocol`, `kern.hv_vmm_present` 1.
    let vm = |protocol: Option<&str>| {
        let mut raw = mac(protocol);
        raw.hypervisor = true;
        classify_macos(&raw).bus
    };
    assert_eq!(vm(None), StorageBus::Virtual);
    // A protocol the disk does report still decides the bus.
    assert_eq!(vm(Some("USB")), StorageBus::Usb);
}

/// The hosted macOS runner (and a developer Mac): the system volume's disk description gives a
/// block size, and the bus is known. On physical hardware the disk must report a protocol; the
/// hosted runner is a virtual machine whose disk reports none, so its bus is `Virtual`.
#[cfg(target_os = "macos")]
#[test]
fn live_macos_system_volume() {
    let raw = syntropy_io::os::volume::macos_volume(Path::new("/")).unwrap();
    eprintln!("{raw:?}");
    assert!(raw.protocol.is_some() || raw.hypervisor, "{raw:?}");
    assert!(raw.media_block.is_some(), "{raw:?}");
    let v = macos_live(Path::new("/")).unwrap();
    assert_ne!(v.bus, StorageBus::Unknown, "{v:?} from {raw:?}");
    assert_eq!(v.fs, FsKind::Apfs, "{v:?}");
    assert!(
        matches!(v.direct_io, DirectIoSupport::Supported { .. }),
        "{v:?}"
    );
    assert!(v.logical_block > 0 && v.free.0 > 0);
}

/// The hosted Windows runner: the system volume answers the bus and alignment queries and
/// reports a known bus.
#[cfg(windows)]
#[test]
fn live_windows_system_volume() {
    let raw = syntropy_io::os::volume::windows_volume(Path::new(r"C:\")).unwrap();
    eprintln!("{raw:?}");
    assert!(raw.bus_type.is_some(), "{raw:?}");
    let v = windows_live(Path::new(r"C:\")).unwrap();
    // A bus `StorageBus` cannot name yet may read `Unknown`; any other unmapped bus fails.
    if !raw
        .bus_type
        .is_some_and(|b| UNMODELLED_BUS_TYPES.contains(&b))
    {
        assert_ne!(v.bus, StorageBus::Unknown, "{v:?} from {raw:?}");
    }
    assert!(
        matches!(v.direct_io, DirectIoSupport::Supported { .. }),
        "{v:?}"
    );
    assert!(v.logical_block > 0 && v.free.0 > 0);
}

#[cfg(not(target_os = "macos"))]
#[test]
fn live_macos_is_unsupported_elsewhere() {
    assert!(matches!(
        macos_live(Path::new("/")),
        Err(EnumError::UnsupportedOs)
    ));
}

#[cfg(not(windows))]
#[test]
fn live_windows_is_unsupported_elsewhere() {
    assert!(matches!(
        windows_live(Path::new("/")),
        Err(EnumError::UnsupportedOs)
    ));
}
