// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ordinal rules of RFC-0010 §1.4 (RFC-0001 §5.1), as a pure function over what
//! enumeration found.
//!
//! Devices: discrete GPUs by PCI address, then integrated GPUs, then one CPU device per NUMA
//! domain in node order, then NPUs. Tiers: one `Vram` per discrete GPU in device order; one
//! `Unified` per unified pool; one `HostPinned` and one `HostPageable` per NUMA domain; one
//! `Nvme` per volume that holds the store or a configured extra tier path.
//!
//! The result depends only on the set of inputs, never on the order enumeration listed
//! them in, so a node that re-enumerates unchanged hardware gets the same IDs.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::ids::{DeviceId, NodeId, TierId};

use super::device::{DeviceKind, PciAddress};
use super::tier::TierKind;

/// One enumerated device before it has an ordinal.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceCandidate {
    /// What kind of engine it is.
    pub kind: DeviceKind,
    /// PCI address, where the device has one.
    pub pci: Option<PciAddress>,
    /// NUMA domain. For a CPU device, the domain it covers.
    pub numa: Option<u16>,
    /// Device UUID, where the driver reports one.
    pub uuid: Option<[u8; 16]>,
    /// The name the driver reports.
    pub name: String,
    /// The device computes on a pool of memory it shares with the CPU (Apple, APUs,
    /// GB10-class), which becomes a `Unified` tier instead of `Vram`.
    pub unified: bool,
}

/// One NUMA domain's host memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HostDomain {
    /// The NUMA domain.
    pub numa: u16,
    /// The domain gets a `HostPinned` tier. False where no device copies from host RAM, such
    /// as Apple unified-memory machines (ADR-0014 capacity rules).
    pub pinned: bool,
}

/// One volume that holds the store or a configured extra tier path.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VolumeCandidate {
    /// The volume's mount point.
    pub mount: PathBuf,
    /// The volume holds the store directory.
    pub holds_store: bool,
}

/// What enumeration found on one node, in any order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Enumerated {
    /// Devices.
    pub devices: Vec<DeviceCandidate>,
    /// NUMA domains with host memory.
    pub domains: Vec<HostDomain>,
    /// Volumes for `Nvme` tiers.
    pub volumes: Vec<VolumeCandidate>,
}

/// What a tier ordinal was assigned to.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TierSlot {
    /// The VRAM of a discrete GPU.
    Vram {
        /// The GPU.
        device: DeviceId,
    },
    /// The unified pool of a GPU that shares memory with the CPU.
    Unified {
        /// The GPU.
        device: DeviceId,
    },
    /// A NUMA domain's DMA-registered RAM.
    HostPinned {
        /// The NUMA domain.
        numa: u16,
    },
    /// A NUMA domain's CPU-only RAM.
    HostPageable {
        /// The NUMA domain.
        numa: u16,
    },
    /// A storage volume.
    Nvme {
        /// The volume's mount point.
        mount: PathBuf,
    },
}

impl TierSlot {
    /// The tier kind this slot becomes.
    pub fn kind(&self) -> TierKind {
        match self {
            TierSlot::Vram { .. } => TierKind::Vram,
            TierSlot::Unified { .. } => TierKind::Unified,
            TierSlot::HostPinned { .. } => TierKind::HostPinned,
            TierSlot::HostPageable { .. } => TierKind::HostPageable,
            TierSlot::Nvme { .. } => TierKind::Nvme,
        }
    }
}

/// The IDs [`assign`] gave out, each with what it was given to, in ordinal order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ordinals {
    /// Devices, by ordinal.
    pub devices: Vec<(DeviceId, DeviceCandidate)>,
    /// Tiers, by ordinal.
    pub tiers: Vec<(TierId, TierSlot)>,
}

/// More devices or tiers than a `u16` ordinal can number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TooManyOrdinals;

impl fmt::Display for TooManyOrdinals {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("more than 65,536 devices or tiers on one node")
    }
}

impl std::error::Error for TooManyOrdinals {}

/// Position of a device kind in RFC-0010 §1.4's order.
fn kind_rank(kind: DeviceKind) -> u8 {
    match kind {
        DeviceKind::DiscreteGpu => 0,
        DeviceKind::IntegratedGpu => 1,
        DeviceKind::Cpu => 2,
        DeviceKind::Npu => 3,
    }
}

fn is_gpu(kind: DeviceKind) -> bool {
    matches!(kind, DeviceKind::DiscreteGpu | DeviceKind::IntegratedGpu)
}

/// Assigns device and tier ordinals on `node` by RFC-0010 §1.4.
///
/// Within a kind, devices are ordered by PCI address (devices without one after those with
/// one), then NUMA domain, then UUID, name and the unified flag, so the order is total and
/// equal inputs give equal IDs. Host tiers follow NUMA domain order, `HostPinned` before
/// `HostPageable` within a domain; a domain listed twice counts once and is pinned if
/// either entry is. Volumes holding the store come first, then the rest by mount point; a
/// mount point listed twice counts once.
pub fn assign(node: NodeId, found: &Enumerated) -> Result<Ordinals, TooManyOrdinals> {
    let mut devices: Vec<&DeviceCandidate> = found.devices.iter().collect();
    devices.sort_by(|a, b| {
        (
            kind_rank(a.kind),
            a.pci.is_none(),
            a.pci,
            a.numa,
            a.uuid,
            &a.name,
            a.unified,
        )
            .cmp(&(
                kind_rank(b.kind),
                b.pci.is_none(),
                b.pci,
                b.numa,
                b.uuid,
                &b.name,
                b.unified,
            ))
    });
    let devices = devices
        .into_iter()
        .enumerate()
        .map(|(i, d)| {
            let ordinal = u16::try_from(i).map_err(|_| TooManyOrdinals)?;
            Ok((DeviceId { node, ordinal }, d.clone()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut slots = Vec::new();
    for (id, d) in &devices {
        if d.kind == DeviceKind::DiscreteGpu && !d.unified {
            slots.push(TierSlot::Vram { device: *id });
        }
    }
    for (id, d) in &devices {
        if is_gpu(d.kind) && d.unified {
            slots.push(TierSlot::Unified { device: *id });
        }
    }
    let mut domains: Vec<HostDomain> = found.domains.clone();
    domains.sort_by_key(|h| (h.numa, !h.pinned));
    domains.dedup_by_key(|h| h.numa);
    for h in &domains {
        if h.pinned {
            slots.push(TierSlot::HostPinned { numa: h.numa });
        }
        slots.push(TierSlot::HostPageable { numa: h.numa });
    }
    let mut volumes: Vec<&VolumeCandidate> = found.volumes.iter().collect();
    volumes.sort_by(|a, b| (!a.holds_store, &a.mount).cmp(&(!b.holds_store, &b.mount)));
    let mut seen_mounts: Vec<&PathBuf> = Vec::new();
    for v in volumes {
        if !seen_mounts.contains(&&v.mount) {
            seen_mounts.push(&v.mount);
            slots.push(TierSlot::Nvme {
                mount: v.mount.clone(),
            });
        }
    }
    let tiers = slots
        .into_iter()
        .enumerate()
        .map(|(i, slot)| {
            let ordinal = u16::try_from(i).map_err(|_| TooManyOrdinals)?;
            Ok((TierId { node, ordinal }, slot))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Ordinals { devices, tiers })
}
