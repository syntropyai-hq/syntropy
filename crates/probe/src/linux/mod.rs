// SPDX-License-Identifier: MIT OR Apache-2.0

//! E1 host enumeration on Linux (RFC-0010 §2.1, §2.2): CPU topology and hybrid cores, ISA
//! flags, RAM and per-NUMA memory, cgroup v1 and v2 limits, the memory-lock limit, WSL2 and
//! container detection, the io_uring sysctl, and the power source.
//!
//! Everything here reads world-readable files under `/proc` and `/sys` (plus `/dev/dxg`,
//! `/.dockerenv` and `/run/.containerenv` for existence only) through a [`HostFs`], so the
//! same code runs on the live `/` and on recorded fixture trees, and no step needs root.
//! The facts that need a system call are not read here: the AMX tile-data permission
//! request and the `io_uring_setup` check (PLC-E1-T14). Their fields say so.

mod cpu;
mod env;
mod isa;
mod mem;
mod parse;

use smallvec::SmallVec;
use syntropy_core::Blake3;
use syntropy_core::ids::NodeId;
use syntropy_core::resource::caps::PowerSource;
use syntropy_core::resource::ordinal::{self, DeviceCandidate, Enumerated, HostDomain, TierSlot};
use syntropy_core::resource::{
    CpuArch, DeviceClass, DeviceInfo, DeviceInfoExt, DeviceKind, HostMemInfo,
};
use syntropy_core::units::Bytes;

use crate::error::EnumError;
use crate::fs::HostFs;

pub use env::{CgroupVersion, ContainerKind, HostEnv, IoUringSysctl};
pub use isa::IsaSource;

/// What E1 found on one Linux host.
#[derive(Clone, Debug, PartialEq)]
pub struct LinuxHost {
    /// The `NodeInfo` facts E1 owns.
    pub node: NodeFacts,
    /// One CPU device per NUMA domain with online CPUs, in ordinal order.
    pub cpus: Vec<DeviceInfo>,
    /// One record per NUMA domain with memory, in domain order: the raw inputs of that
    /// domain's `HostPinned` and `HostPageable` tiers.
    pub host_mem: Vec<NumaHostMem>,
    /// Environment facts the tier and I/O rules read.
    pub env: HostEnv,
    /// E1's share of the ordinal input (RFC-0010 §1.4): the CPU devices and host domains.
    /// The graph builder adds the GPUs, volumes and NPUs E2–E5 find and assigns the final
    /// ordinals from the whole set.
    pub candidates: Enumerated,
}

/// The `NodeInfo` fields (RFC-0001 §5.2) whose values E1 reads. The record itself, with
/// identity, build, protocol, labels, roles and install fields owned by other RFCs, is
/// assembled by the daemon.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeFacts {
    /// `NodeInfo::arch`.
    pub arch: CpuArch,
    /// `NodeInfo::power`.
    pub power: PowerSource,
    /// The host has a system battery (RFC-0010 §1.3 `CapabilityProfileExt::battery_present`).
    pub battery_present: bool,
    /// `/proc/sys/kernel/osrelease`: part of the profile key (RFC-0010 §5.1).
    pub kernel_release: String,
}

/// One NUMA domain's host memory inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NumaHostMem {
    /// The NUMA domain.
    pub numa: u16,
    /// The domain's own RAM, from its `meminfo`.
    pub node_total: Bytes,
    /// The inputs RFC-0020's `budget::host_usable` reads. `total` and `available` are
    /// host-wide; `numa_free` is this domain's.
    pub mem: HostMemInfo,
}

/// Enumerates the host behind `fs`. `node` fills the IDs; the ordinals are those of a node
/// with only these CPU devices and host tiers (see [`LinuxHost::candidates`]).
pub fn enumerate(fs: &dyn HostFs, isa: IsaSource, node: NodeId) -> Result<LinuxHost, EnumError> {
    let cpu = cpu::facts(fs, isa)?;
    let mem = mem::facts(fs)?;
    let env = env::facts(fs)?;
    let power = env::power(fs)?;
    let kernel_release = read(fs, "proc/sys/kernel/osrelease")?.trim().to_owned();

    let domains = cpu::domains(fs, &cpu.online, &mem)?;
    let mut candidates = Enumerated::default();
    for d in &domains {
        if !d.cpus.is_empty() {
            candidates.devices.push(DeviceCandidate {
                kind: DeviceKind::Cpu,
                pci: None,
                numa: Some(d.numa),
                uuid: None,
                name: cpu.name.clone(),
                unified: false,
            });
        }
        if d.mem_total > 0 {
            candidates.domains.push(HostDomain {
                numa: d.numa,
                pinned: true,
            });
        }
    }
    let ords = ordinal::assign(node, &candidates).map_err(|_| EnumError::TooManyOrdinals)?;

    let mut cpus = Vec::new();
    for (id, cand) in &ords.devices {
        let numa = cand.numa.unwrap_or(0);
        let Some(d) = domains.iter().find(|d| d.numa == numa) else {
            continue;
        };
        let topo = cpu::topology(fs, d, &cpu)?;
        let local_tiers = ords
            .tiers
            .iter()
            .filter(|(_, s)| {
                matches!(s, TierSlot::HostPinned { numa: n } | TierSlot::HostPageable { numa: n } if *n == numa)
            })
            .map(|(t, _)| *t)
            .collect();
        cpus.push(DeviceInfo {
            id: *id,
            kind: DeviceKind::Cpu,
            name: cand.name.clone(),
            class: DeviceClass::Cpu {
                arch: cpu.arch,
                level: cpu.isa.level,
            },
            // E5 fills the providers when the CPU provider opens the device.
            backends: SmallVec::new(),
            features: cpu.isa.features,
            local_tiers,
            numa: Some(numa),
            // Hashing needs BLAKE3, which this crate does not link yet; the profile-key step
            // computes it. The all-zero default marks it unset.
            fingerprint: Blake3::default(),
            ext: DeviceInfoExt {
                cpu: Some(topo),
                ..DeviceInfoExt::default()
            },
        });
    }

    let host_mem = domains
        .iter()
        .filter(|d| d.mem_total > 0)
        .map(|d| NumaHostMem {
            numa: d.numa,
            node_total: Bytes(d.mem_total),
            mem: HostMemInfo {
                total: Bytes(mem.total),
                available: Bytes(mem.available),
                numa_free: Some(Bytes(d.mem_free)),
                cgroup_limit: mem.cgroup_headroom.map(Bytes),
                memlock_limit: mem.memlock.map(Bytes),
                windows_lockable: None,
                wsl2: env.wsl2,
                macos_wired_limit: None,
                large_pages: mem.large_pages.map(Bytes),
                // Measured by step H5 (RFC-0010 §3.2), not E1.
                lock: None,
            },
        })
        .collect();

    Ok(LinuxHost {
        node: NodeFacts {
            arch: cpu.arch,
            power: power.source,
            battery_present: power.battery_present,
            kernel_release,
        },
        cpus,
        host_mem,
        env: HostEnv {
            cgroup: mem.cgroup,
            cap_ipc_lock: mem.cap_ipc_lock,
            ..env
        },
        candidates,
    })
}

/// Enumerates the live host as the current user, reading `/` and using runtime ISA
/// detection.
#[cfg(target_os = "linux")]
pub fn enumerate_live(node: NodeId) -> Result<LinuxHost, EnumError> {
    enumerate(&crate::fs::DirFs::new("/"), IsaSource::Runtime, node)
}

/// Enumerates the live host. This module covers Linux only.
#[cfg(not(target_os = "linux"))]
pub fn enumerate_live(_node: NodeId) -> Result<LinuxHost, EnumError> {
    Err(EnumError::UnsupportedOs)
}

/// Reads a file that must exist.
fn read(fs: &dyn HostFs, path: &str) -> Result<String, EnumError> {
    fs.read(path).map_err(|e| EnumError::from_io(path, &e))
}

/// Reads a file whose absence means "not present here".
fn read_opt(fs: &dyn HostFs, path: &str) -> Result<Option<String>, EnumError> {
    match fs.read(path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(EnumError::from_io(path, &e)),
    }
}

/// Lists a directory whose absence means "none".
fn list_opt(fs: &dyn HostFs, path: &str) -> Result<Vec<String>, EnumError> {
    match fs.list(path) {
        Ok(v) => Ok(v),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(EnumError::from_io(path, &e)),
    }
}

/// Attaches `path` to a parser error.
fn at<T>(path: &str, r: Result<T, String>) -> Result<T, EnumError> {
    r.map_err(|what| EnumError::malformed(path, what))
}
