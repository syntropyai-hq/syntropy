// SPDX-License-Identifier: MIT OR Apache-2.0

//! The E1 records every OS module emits (RFC-0010 §2.1): the `NodeInfo` facts and the per-NUMA
//! host memory inputs, and the step that numbers a host's CPU devices and host tiers.

use smallvec::SmallVec;
use syntropy_core::Blake3;
use syntropy_core::ids::NodeId;
use syntropy_core::resource::caps::PowerSource;
use syntropy_core::resource::ordinal::{self, DeviceCandidate, Enumerated, HostDomain, TierSlot};
use syntropy_core::resource::{
    CpuArch, CpuTopology, DeviceClass, DeviceFeatures, DeviceInfo, DeviceInfoExt, DeviceKind,
    HostMemInfo,
};
use syntropy_core::units::Bytes;

use crate::error::EnumError;

/// The `NodeInfo` fields (RFC-0001 §5.2) whose values E1 reads. The record itself, with
/// identity, build, protocol, labels, roles and install fields owned by other RFCs, is
/// assembled by PLC-E1-T09 (Linux) and PLC-E1-T15 (Windows, macOS); its field types have no
/// owner yet (#1524 item 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeFacts {
    /// `NodeInfo::arch`.
    pub arch: CpuArch,
    /// `NodeInfo::power`.
    pub power: PowerSource,
    /// The host has a system battery (RFC-0010 §1.3 `CapabilityProfileExt::battery_present`).
    pub battery_present: bool,
    /// The kernel release, part of the profile key (RFC-0010 §5.1): Linux
    /// `/proc/sys/kernel/osrelease`, macOS `kern.osrelease`, Windows
    /// `<major>.<minor>.<build>` from `RtlGetVersion`.
    pub kernel_release: String,
}

/// One NUMA domain's host memory inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NumaHostMem {
    /// The NUMA domain.
    pub numa: u16,
    /// The domain's own RAM: Linux from its `meminfo`; Windows the whole host's
    /// `ullTotalPhys` on a single-node host and the ACPI SRAT's ranges for the node otherwise;
    /// macOS `hw.memsize` (one domain).
    pub node_total: Bytes,
    /// The inputs RFC-0020's `budget::host_usable` reads. `total` and `available` are
    /// host-wide; `numa_free` is this domain's.
    pub mem: HostMemInfo,
}

/// One NUMA domain's CPU device, before numbering.
#[derive(Clone, Debug)]
pub(crate) struct CpuDomain {
    pub numa: u16,
    pub name: String,
    pub class: DeviceClass,
    pub features: DeviceFeatures,
    pub topology: CpuTopology,
}

/// One NUMA domain with host memory.
#[derive(Clone, Debug)]
pub(crate) struct MemDomain {
    pub numa: u16,
    /// The domain gets a `HostPinned` tier (false on Apple, ADR-0014 capacity rules).
    pub pinned: bool,
}

/// Numbers the CPU devices and host tiers of a node with only these (RFC-0010 §1.4), and
/// builds the CPU `DeviceInfo` records. The ordinals are provisional, as
/// [`crate::linux::LinuxHost::cpus`] explains.
pub(crate) fn number(
    node: NodeId,
    cpus: Vec<CpuDomain>,
    mem: &[MemDomain],
) -> Result<(Vec<DeviceInfo>, Enumerated), EnumError> {
    let mut candidates = Enumerated::default();
    for c in &cpus {
        candidates.devices.push(DeviceCandidate {
            kind: DeviceKind::Cpu,
            pci: None,
            numa: Some(c.numa),
            uuid: None,
            name: c.name.clone(),
            unified: false,
        });
    }
    for m in mem {
        candidates.domains.push(HostDomain {
            numa: m.numa,
            pinned: m.pinned,
        });
    }
    let ords = ordinal::assign(node, &candidates).map_err(|_| EnumError::TooManyOrdinals)?;
    let mut out = Vec::new();
    for (id, cand) in &ords.devices {
        let numa = cand.numa.unwrap_or(0);
        let Some(c) = cpus.iter().find(|c| c.numa == numa) else {
            continue;
        };
        let local_tiers = ords
            .tiers
            .iter()
            .filter(|(_, s)| {
                matches!(s, TierSlot::HostPinned { numa: n } | TierSlot::HostPageable { numa: n } if *n == numa)
            })
            .map(|(t, _)| *t)
            .collect();
        out.push(DeviceInfo {
            id: *id,
            kind: DeviceKind::Cpu,
            name: cand.name.clone(),
            // E1's provisional class; E5 replaces it with the CPU provider's (ADR-0405
            // decision 3).
            class: c.class,
            // E5 fills the providers when the CPU provider opens the device.
            backends: SmallVec::new(),
            features: c.features,
            local_tiers,
            numa: Some(numa),
            // Hashing needs BLAKE3, which this crate does not link. The all-zero default marks
            // it unset; no task owns computing it yet (#1524 item 2).
            fingerprint: Blake3::default(),
            ext: DeviceInfoExt {
                cpu: Some(c.topology.clone()),
                ..DeviceInfoExt::default()
            },
        });
    }
    Ok((out, candidates))
}

/// A count as a `u16`, saturating.
pub(crate) fn count(n: usize) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

/// Bytes from hex text; whitespace is ignored. For recorded API responses.
pub(crate) fn parse_hex(s: &str) -> Option<Vec<u8>> {
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}
