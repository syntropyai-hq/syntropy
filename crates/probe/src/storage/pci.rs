// SPDX-License-Identifier: MIT OR Apache-2.0

//! The PCI side of a Linux volume (RFC-0010 §2.2, "NVMe volume" and "Chipset-shared lane"): the
//! controller's identity and links, and whether its path to the CPU passes a chipset uplink.
//!
//! Everything is read from `/sys/devices/pci*` through a [`HostFs`], by walking directory names
//! and never a symlink, so the same code runs on the live host and on a fixture tree.
//!
//! **The rule, as built.** RFC-0010 §2.2 says: walk the hierarchy from the controller to the
//! root port; if the path passes a bridge that also parents SATA, USB or NIC functions and whose
//! own link is narrower than the sum of its children, set `shared_lane` and `upstream =
//! Chipset`. The text leaves four points open, and this module reads them as follows:
//!
//! 1. *The bridges on the path* are the ancestors strictly between the root port and the
//!    controller. The root port is the CPU's own and never counts.
//! 2. *Parents* means any function below the bridge, at any depth, so a chipset upstream port
//!    parents the functions behind its downstream ports.
//! 3. *The children* whose links are summed are the non-bridge functions below the bridge.
//!    *Narrower* compares bandwidth, from each link's maximum generation and width, so a
//!    Gen4 x4 uplink under a Gen3 x4 and a Gen2 x1 endpoint is not narrower than their sum, but
//!    under a Gen4 x4 and a Gen3 x4 it is. A bridge's maximum link is used because the rule
//!    sizes the uplink, which idles at a lower current link.
//! 4. *Two daisy-chained chipset dies* need no extra case: the rule applies to each bridge on the
//!    path, and the first die's upstream port parents everything behind the second.
//!
//! A path with no bridge below the root port is `CpuRootPort`. A path with bridges, none of which
//! meets the rule, is `Switch`.

use syntropy_core::resource::{PciAddress, PciInfo, PcieLink, PcieUpstream};

use crate::error::EnumError;
use crate::fs::HostFs;

/// The controller's record and whether it shares a lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Controller {
    pub info: PciInfo,
    pub shared_lane: bool,
}

/// Parses `0000:01:00.0`.
pub(crate) fn parse_bdf(text: &str) -> Option<PciAddress> {
    let (domain, rest) = text.split_once(':')?;
    let (bus, rest) = rest.split_once(':')?;
    let (device, function) = rest.split_once('.')?;
    Some(PciAddress {
        domain: u32::from_str_radix(domain, 16).ok()?,
        bus: u8::from_str_radix(bus, 16).ok()?,
        device: u8::from_str_radix(device, 16).ok()?,
        function: function.parse().ok().filter(|f| *f < 8)?,
    })
}

fn is_bdf(name: &str) -> bool {
    parse_bdf(name).is_some() && name.len() >= 12
}

fn hex(fs: &dyn HostFs, path: &str, file: &str) -> Option<u32> {
    let text = fs.read(&format!("{path}/{file}")).ok()?;
    u32::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok()
}

/// `16.0 GT/s PCIe` -> generation.
fn gen_of(speed: &str) -> Option<u8> {
    let gt: f64 = speed.split_whitespace().next()?.parse().ok()?;
    Some(match gt as u32 {
        2 => 1,
        5 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        64 => 6,
        128 => 7,
        _ => return None,
    })
}

fn link(fs: &dyn HostFs, path: &str, speed: &str, width: &str) -> Option<PcieLink> {
    let gen_ = gen_of(fs.read(&format!("{path}/{speed}")).ok()?.trim())?;
    let width: u8 = fs
        .read(&format!("{path}/{width}"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(PcieLink { r#gen: gen_, width })
}

/// Link bandwidth in MB/s [estimate: per-lane payload rates after encoding overhead].
fn bandwidth(l: PcieLink) -> u64 {
    let per_lane = match l.r#gen {
        1 => 250,
        2 => 500,
        3 => 985,
        4 => 1969,
        5 => 3938,
        6 => 7563,
        _ => 15125,
    };
    per_lane * u64::from(l.width)
}

fn is_bridge(class: u32) -> bool {
    class >> 8 == 0x0604
}

/// SATA (0x0106), USB (0x0c03) or network (0x02) class codes.
fn is_shared_kind(class: u32) -> bool {
    matches!(class >> 8, 0x0106 | 0x0c03) || class >> 16 == 0x02
}

/// The PCI-named directories directly under `path`.
fn children(fs: &dyn HostFs, path: &str) -> Vec<String> {
    fs.list(path)
        .unwrap_or_default()
        .into_iter()
        .filter(|n| is_bdf(n))
        .collect()
}

/// The path components from a PCI root to `bdf`, the root first.
fn find(fs: &dyn HostFs, dir: &str, bdf: &str, depth: usize, out: &mut Vec<String>) -> bool {
    if depth > 12 {
        return false;
    }
    for name in children(fs, dir) {
        out.push(name.clone());
        if name == bdf || find(fs, &format!("{dir}/{name}"), bdf, depth + 1, out) {
            return true;
        }
        out.pop();
    }
    false
}

/// Sums the maximum-link bandwidth of the non-bridge functions below `dir`, and notes whether any
/// is a SATA, USB or network function.
fn endpoints(fs: &dyn HostFs, dir: &str, sum: &mut u64, shared_kind: &mut bool) {
    for name in children(fs, dir) {
        let path = format!("{dir}/{name}");
        let class = hex(fs, &path, "class").unwrap_or(0);
        if is_bridge(class) {
            endpoints(fs, &path, sum, shared_kind);
        } else {
            *shared_kind |= is_shared_kind(class);
            if let Some(l) = link(fs, &path, "max_link_speed", "max_link_width") {
                *sum += bandwidth(l);
            }
        }
    }
}

/// Describes the controller at `address` (e.g. `0000:01:00.0`).
///
/// # Errors
///
/// `Missing` when no `sys/devices/pci*` tree holds the address, or the function lacks its
/// `vendor` or `device` file.
pub(crate) fn controller(fs: &dyn HostFs, address: &str) -> Result<Controller, EnumError> {
    let bdf = parse_bdf(address).ok_or_else(|| EnumError::Malformed {
        path: "sys/class/nvme/<controller>/address".to_owned(),
        what: format!("`{address}` is not a PCI address"),
    })?;
    let missing = || EnumError::Missing {
        path: format!("sys/devices/pci*/…/{address}"),
    };
    let mut chain = Vec::new();
    let roots = fs.list("sys/devices").map_err(|_| missing())?;
    let mut root_dir = None;
    for root in roots.into_iter().filter(|r| r.starts_with("pci")) {
        let dir = format!("sys/devices/{root}");
        let mut c = Vec::new();
        if find(fs, &dir, address, 0, &mut c) {
            chain = c;
            root_dir = Some(dir);
            break;
        }
    }
    let root_dir = root_dir.ok_or_else(missing)?;
    let path_of = |upto: usize| {
        let mut p = root_dir.clone();
        for name in &chain[..=upto] {
            p.push('/');
            p.push_str(name);
        }
        p
    };
    let dev = path_of(chain.len() - 1);
    let need = |file: &str| {
        hex(fs, &dev, file).ok_or_else(|| EnumError::Missing {
            path: format!("{dev}/{file}"),
        })
    };
    let subsystem = (hex(fs, &dev, "subsystem_vendor").unwrap_or(0) << 16)
        | hex(fs, &dev, "subsystem_device").unwrap_or(0);

    // chain = [root port, bridges…, controller]; the root port is never a shared bridge.
    let mut shared_lane = false;
    let bridges = chain.len().saturating_sub(2);
    for i in 1..=bridges {
        let bridge = path_of(i);
        let (mut sum, mut kind) = (0, false);
        endpoints(fs, &bridge, &mut sum, &mut kind);
        let own = link(fs, &bridge, "max_link_speed", "max_link_width").map(bandwidth);
        if kind && own.is_some_and(|own| own < sum) {
            shared_lane = true;
        }
    }
    let upstream = if shared_lane {
        PcieUpstream::Chipset
    } else if bridges == 0 {
        PcieUpstream::CpuRootPort
    } else {
        PcieUpstream::Switch
    };
    let numa = fs
        .read(&format!("{dev}/numa_node"))
        .ok()
        .and_then(|t| t.trim().parse::<i32>().ok())
        .and_then(|n| u16::try_from(n).ok());
    Ok(Controller {
        info: PciInfo {
            bdf,
            vendor: u16::try_from(need("vendor")?).unwrap_or(0),
            device: u16::try_from(need("device")?).unwrap_or(0),
            subsystem,
            link_cur: link(fs, &dev, "current_link_speed", "current_link_width")
                .unwrap_or_default(),
            link_max: link(fs, &dev, "max_link_speed", "max_link_width").unwrap_or_default(),
            upstream,
            resizable_bar: None,
            numa,
        },
        shared_lane,
    })
}
