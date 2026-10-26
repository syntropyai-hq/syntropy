// SPDX-License-Identifier: MIT OR Apache-2.0

//! E3 on Linux (RFC-0010 §2.2 "NVMe volume" and "Chipset-shared lane"): the `VolumeInfo` of the
//! volume holding a path.
//!
//! `statfs` and `statx(STATX_DIOALIGN)` come through [`VolumeCalls`]; everything else is read
//! from `/proc/self/mountinfo` and `/sys` through a [`HostFs`], so the same code runs on the live
//! host and on a fixture tree. No step needs root.
//!
//! **Direct I/O** is `Supported` when the kernel reports non-zero `STATX_DIOALIGN` values for a
//! file on the volume. It is `Unsupported` when the kernel reports zero, does not report them (a
//! kernel before 6.1) or the file system is memory-backed (tmpfs, ramfs), which RFC-0010 §9 lists
//! as refusing it. An old kernel therefore reads as `Unsupported`, the safe side: S1 then runs
//! buffered and labels it, rather than assuming an alignment.

use std::path::{Path, PathBuf};

use syntropy_core::resource::{DirectIoSupport, FsKind, StorageBus, TempSource, VolumeInfo};
use syntropy_core::units::Bytes;
use syntropy_io::os::volume::VolumeCalls;

use super::pci;
use crate::error::EnumError;
use crate::fs::HostFs;

/// `statfs` `f_type` magic numbers (`linux/magic.h`).
mod magic {
    pub const EXT4: u64 = 0xEF53;
    pub const XFS: u64 = 0x5846_5342;
    pub const BTRFS: u64 = 0x9123_683E;
    pub const ZFS: u64 = 0x2FC1_2FC1;
    pub const TMPFS: u64 = 0x0102_1994;
    pub const RAMFS: u64 = 0x8584_58F6;
}

fn fs_kind(magic: u64) -> FsKind {
    match magic {
        magic::EXT4 => FsKind::Ext4,
        magic::XFS => FsKind::Xfs,
        magic::BTRFS => FsKind::Btrfs,
        magic::ZFS => FsKind::Zfs,
        _ => FsKind::Other,
    }
}

/// One `/proc/self/mountinfo` row.
struct Mount {
    point: String,
    source: String,
}

fn unescape(field: &str) -> String {
    // mountinfo writes space, tab, newline and backslash as `\040`, `\011`, `\012`, `\134`.
    let mut out = String::new();
    let bytes = field.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && let Some(code) = field
                .get(i + 1..i + 4)
                .and_then(|o| u8::from_str_radix(o, 8).ok())
        {
            out.push(char::from(code));
            i += 4;
        } else {
            out.push(char::from(bytes[i]));
            i += 1;
        }
    }
    out
}

/// The mount that holds `path`: the longest mount point that is a path prefix.
fn mount_of(fs: &dyn HostFs, path: &Path) -> Result<Mount, EnumError> {
    let text = fs
        .read("proc/self/mountinfo")
        .map_err(|_| EnumError::Missing {
            path: "proc/self/mountinfo".to_owned(),
        })?;
    let mut best: Option<Mount> = None;
    for line in text.lines() {
        let fields: Vec<&str> = line.split(' ').collect();
        let Some(dash) = fields.iter().position(|f| *f == "-") else {
            continue;
        };
        if fields.len() < dash + 3 || dash < 5 {
            continue;
        }
        let point = unescape(fields[4]);
        if path.starts_with(&point) && best.as_ref().is_none_or(|b| point.len() >= b.point.len()) {
            best = Some(Mount {
                point,
                source: unescape(fields[dash + 2]),
            });
        }
    }
    best.ok_or_else(|| EnumError::Malformed {
        path: "proc/self/mountinfo".to_owned(),
        what: format!("no mount holds {}", path.display()),
    })
}

fn text(fs: &dyn HostFs, path: &str) -> Option<String> {
    fs.read(path).ok().map(|t| t.trim().to_owned())
}

fn number(fs: &dyn HostFs, path: &str) -> Result<u32, EnumError> {
    let t = fs.read(path).map_err(|_| EnumError::Missing {
        path: path.to_owned(),
    })?;
    t.trim().parse().map_err(|_| EnumError::Malformed {
        path: path.to_owned(),
        what: format!("`{}` is not a number", t.trim()),
    })
}

/// The block device name for `major:minor`, from the `DEVNAME` line of its `uevent`.
fn devname(fs: &dyn HostFs, major: u32, minor: u32) -> Option<String> {
    let uevent = fs
        .read(&format!("sys/dev/block/{major}:{minor}/uevent"))
        .ok()?;
    uevent
        .lines()
        .find_map(|l| l.strip_prefix("DEVNAME="))
        .map(|n| n.trim().to_owned())
}

/// The whole disk a block device name belongs to, and whether a dm-crypt layer lies between.
///
/// A partition is a directory inside its disk's `sys/block/<disk>`. A device-mapper device names
/// its backing devices under `slaves/`; the first is followed, and a `CRYPT-` `dm/uuid` marks
/// dm-crypt.
fn disk_of(fs: &dyn HostFs, name: &str, crypt: &mut bool) -> Option<String> {
    if fs.exists(&format!("sys/block/{name}")) {
        if text(fs, &format!("sys/block/{name}/dm/uuid")).is_some_and(|u| u.starts_with("CRYPT-")) {
            *crypt = true;
        }
        if let Ok(slaves) = fs.list(&format!("sys/block/{name}/slaves"))
            && let Some(first) = slaves.first()
        {
            return disk_of(fs, first, crypt);
        }
        return Some(name.to_owned());
    }
    fs.list("sys/block")
        .ok()?
        .into_iter()
        .find(|disk| fs.exists(&format!("sys/block/{disk}/{name}")))
}

/// `nvme0` for the namespace `nvme0n1`.
fn nvme_controller(disk: &str) -> Option<String> {
    let rest = disk.strip_prefix("nvme")?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    (!digits.is_empty() && rest[digits.len()..].starts_with('n')).then(|| format!("nvme{digits}"))
}

fn bus_of(fs: &dyn HostFs, disk: &str) -> StorageBus {
    if disk.starts_with("nvme") {
        StorageBus::Nvme
    } else if ["vd", "xvd", "loop", "zram", "ram"]
        .iter()
        .any(|p| disk.starts_with(p))
    {
        StorageBus::Virtual
    } else if disk.starts_with("sd") {
        // The SCSI layer names a SATA drive's vendor `ATA`; a USB or other drive has its own.
        match text(fs, &format!("sys/block/{disk}/device/vendor")) {
            Some(v) if v == "ATA" => StorageBus::Sata,
            _ => StorageBus::Unknown,
        }
    } else {
        StorageBus::Unknown
    }
}

/// Describes the volume holding `path`.
///
/// # Errors
///
/// `Os` when `statfs` or `statx` fails; `Missing` or `Malformed` when a file the volume's block
/// device needs is absent or does not parse. A volume with no block device (tmpfs, a network
/// file system) is `Virtual` with zero block sizes.
pub fn volume(
    fs: &dyn HostFs,
    calls: &dyn VolumeCalls,
    path: &Path,
) -> Result<VolumeInfo, EnumError> {
    let os = |call: &str, e: std::io::Error| EnumError::Os {
        call: call.to_owned(),
        kind: e.kind(),
        message: e.to_string(),
    };
    let st = calls.statfs(path).map_err(|e| os("statfs", e))?;
    let sx = calls.statx(path).map_err(|e| os("statx", e))?;
    let mount = mount_of(fs, path)?;
    let kind = fs_kind(st.fs_type);
    let memory_backed = matches!(st.fs_type, magic::TMPFS | magic::RAMFS);

    let mut info = VolumeInfo {
        mount: PathBuf::from(&mount.point),
        fs: kind,
        bus: StorageBus::Virtual,
        free: Bytes(st.free),
        direct_io: match sx.dio {
            Some((mem_align, offset_align)) if !memory_backed => DirectIoSupport::Supported {
                mem_align,
                offset_align,
            },
            _ => DirectIoSupport::Unsupported,
        },
        ..VolumeInfo::default()
    };

    // The block device: by device number, or for a file system with an anonymous device number
    // (Btrfs) by the mount source.
    let name = if sx.dev_major != 0 {
        devname(fs, sx.dev_major, sx.dev_minor)
    } else {
        mount
            .source
            .strip_prefix("/dev/")
            .filter(|n| !n.contains('/'))
            .map(str::to_owned)
    };
    let mut crypt = false;
    let Some(disk) = name.and_then(|n| disk_of(fs, &n, &mut crypt)) else {
        return Ok(info);
    };
    info.bus = bus_of(fs, &disk);
    if crypt {
        info.encrypted = Some(true);
    }
    info.logical_block = number(fs, &format!("sys/block/{disk}/queue/logical_block_size"))?;
    info.physical_block = number(fs, &format!("sys/block/{disk}/queue/physical_block_size"))?;

    if let Some(ctrl) = nvme_controller(&disk) {
        let base = format!("sys/class/nvme/{ctrl}");
        info.model = text(fs, &format!("{base}/model")).unwrap_or_default();
        info.firmware = text(fs, &format!("{base}/firmware_rev")).unwrap_or_default();
        if let Some(address) = text(fs, &format!("{base}/address")) {
            let c = pci::controller(fs, &address)?;
            info.pci = Some(c.info);
            info.shared_lane = c.shared_lane;
        }
        // The kernel's NVMe hwmon: `<controller>/hwmon<N>/temp1_input`.
        let hwmon = fs
            .list(&base)
            .unwrap_or_default()
            .into_iter()
            .any(|e| e.starts_with("hwmon") && fs.exists(&format!("{base}/{e}/temp1_input")));
        if hwmon {
            info.temperature_source = TempSource::Hwmon;
        }
    } else {
        info.model = text(fs, &format!("sys/block/{disk}/device/model")).unwrap_or_default();
        info.firmware = text(fs, &format!("sys/block/{disk}/device/rev")).unwrap_or_default();
    }
    Ok(info)
}
