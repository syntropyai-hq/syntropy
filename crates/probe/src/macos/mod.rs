// SPDX-License-Identifier: MIT OR Apache-2.0

//! E1 host enumeration on macOS (RFC-0010 §2.1, §2.2; PLC-E1-T03): CPU topology from the
//! `hw.perflevel*` sysctls, ISA from `hw.optional.arm.FEAT_*`, RAM from `hw.memsize` and
//! `host_statistics64(HOST_VM_INFO64)`, the GPU wired limit `iogpu.wired_limit_mb` (read only),
//! `RLIMIT_MEMLOCK`, and the power source from IOKit.
//!
//! The calls go through a [`MacApi`], so the same code runs on the live host
//! (`LiveMac`, macOS only, which calls `syntropy_io::os::macos_host`) and on recorded
//! responses ([`MacFixture`]). [`MacApi::sysctl`] has `sysctlbyname`'s shape, an optional new
//! value included, so a recording fake can show that enumeration never passes one
//! (ADR-0014: Syntropy never raises the wired limit silently).
//!
//! Apple silicon has one NUMA domain and no `HostPinned` tier (RFC-0020 §9.2; ADR-0014
//! capacity rules), so E1 emits one CPU device and one `HostPageable` domain.

mod record;

use std::collections::BTreeSet;
use std::io;

use syntropy_core::ids::NodeId;
use syntropy_core::resource::caps::PowerSource;
use syntropy_core::resource::ordinal::Enumerated;
use syntropy_core::resource::{CpuArch, CpuSet, CpuTopology, DeviceClass, DeviceInfo, HostMemInfo};
use syntropy_core::units::Bytes;

use crate::error::EnumError;
use crate::host::{self, CpuDomain, MemDomain, NodeFacts, NumaHostMem};
use crate::linux::isa::{Detector, IsaSource};

pub use record::{MacCall, MacFixture, MacRecorder};

/// `host_statistics64(HOST_VM_INFO64)`'s page counts E1 reads, with their page size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VmStats {
    /// Bytes per page.
    pub page_size: u64,
    /// `free_count`.
    pub free: u64,
    /// `active_count`.
    pub active: u64,
    /// `inactive_count`.
    pub inactive: u64,
    /// `wire_count`.
    pub wired: u64,
    /// `purgeable_count`.
    pub purgeable: u64,
    /// `speculative_count`.
    pub speculative: u64,
    /// `compressor_page_count`.
    pub compressor: u64,
}

/// One IOKit power source's description, the keys E1 reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PowerSourceDesc {
    /// `Type`: `InternalBattery` for the system battery.
    pub kind: Option<String>,
    /// `Power Source State`.
    pub state: Option<String>,
    /// `Current Capacity`.
    pub current_capacity: Option<i64>,
    /// `Max Capacity`.
    pub max_capacity: Option<i64>,
    /// `Is Present`.
    pub present: Option<bool>,
}

/// IOKit's power-source snapshot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PowerSources {
    /// `IOPSGetProvidingPowerSourceType`: `AC Power`, `Battery Power` or `UPS Power`.
    pub providing: Option<String>,
    /// The power sources.
    pub sources: Vec<PowerSourceDesc>,
}

/// The macOS calls E1 makes.
pub trait MacApi {
    /// `sysctlbyname(name, old, new)`: returns the value's bytes (native-endian integer or
    /// NUL-terminated string), `None` for a name the kernel does not have. `new` is the value
    /// to write; E1 always passes `None`, and `LiveMac` refuses anything else.
    fn sysctl(&self, name: &str, new: Option<&[u8]>) -> io::Result<Option<Vec<u8>>>;
    /// `host_statistics64(HOST_VM_INFO64)`.
    fn vm_statistics(&self) -> io::Result<VmStats>;
    /// `getrlimit(RLIMIT_MEMLOCK)`'s soft limit; `None` when unlimited.
    fn memlock_limit(&self) -> io::Result<Option<u64>>;
    /// The IOKit power sources.
    fn power_sources(&self) -> io::Result<PowerSources>;
}

/// macOS environment facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacEnv {
    /// `kern.osproductversion`, e.g. `26.4.1`: Metal 4 needs 26, RDMA over TB5 26.2
    /// (RFC-0010 §2.2, Environment row).
    pub os_version: String,
}

/// What E1 found on one Mac.
#[derive(Clone, Debug, PartialEq)]
pub struct MacHost {
    /// The `NodeInfo` facts E1 owns.
    pub node: NodeFacts,
    /// The one CPU device, with provisional ordinals as
    /// [`crate::linux::LinuxHost::cpus`] explains.
    pub cpus: Vec<DeviceInfo>,
    /// The one domain's host memory inputs.
    pub host_mem: Vec<NumaHostMem>,
    /// Environment facts.
    pub env: MacEnv,
    /// E1's share of the ordinal input: the CPU device and one domain without `HostPinned`.
    pub candidates: Enumerated,
}

/// `hw.optional.arm.*` names and the `/proc/cpuinfo` names [`Detector`] checks.
const ARM_FEATURES: [(&str, &str); 7] = [
    ("hw.optional.arm.AdvSIMD", "asimd"),
    ("hw.optional.arm.FEAT_DotProd", "asimddp"),
    ("hw.optional.arm.FEAT_FP16", "asimdhp"),
    ("hw.optional.arm.FEAT_I8MM", "i8mm"),
    ("hw.optional.arm.FEAT_BF16", "bf16"),
    ("hw.optional.arm.FEAT_SVE2", "sve2"),
    ("hw.optional.arm.FEAT_SME2", "sme2"),
];

/// The GPU wired-limit sysctl. E1 reads it and never writes it (ADR-0014; RFC-0020 §9.1:
/// only `syntropy tune` changes it).
pub const WIRED_LIMIT: &str = "iogpu.wired_limit_mb";

const MIB: u64 = 1024 * 1024;

/// Enumerates the Mac behind `api`. `node` fills the IDs.
pub fn enumerate(api: &dyn MacApi, node: NodeId) -> Result<MacHost, EnumError> {
    let arch = match sysctl_int(api, "hw.optional.arm64")? {
        Some(1) => CpuArch::Aarch64,
        // Intel Macs are not a platform ADR-0020 lists.
        _ => {
            return Err(EnumError::UnsupportedArch {
                found: "x86_64 (macOS)".into(),
            });
        }
    };
    let name = need_str(api, "machdep.cpu.brand_string")?;

    let mut flags = BTreeSet::new();
    for (sysctl, cpuinfo) in ARM_FEATURES {
        if sysctl_int(api, sysctl)?.is_some_and(|v| v != 0) {
            flags.insert(cpuinfo.to_owned());
        }
    }
    let isa = Detector {
        source: IsaSource::CpuInfo,
        cpuinfo: &flags,
    }
    .isa(arch)?;

    let topology = topology(api, isa.flags)?;

    let memsize = need_int(api, "hw.memsize")?;
    let vm = api
        .vm_statistics()
        .map_err(|e| EnumError::os("host_statistics64", &e))?;
    if vm.page_size == 0 {
        return Err(EnumError::bad("host_statistics64", "page size 0"));
    }
    let pages = |n: u64| Bytes(n.saturating_mul(vm.page_size));
    // RFC-0010 §2.2: available is free + inactive + purgeable.
    let available = pages(
        vm.free
            .saturating_add(vm.inactive)
            .saturating_add(vm.purgeable),
    );
    // 0, the default, means the OS's own cap applies: `None` (RFC-0010 §1.2).
    let wired = sysctl_int(api, WIRED_LIMIT)?
        .filter(|mb| *mb > 0)
        .map(|mb| Bytes(mb.saturating_mul(MIB)));
    let memlock = api
        .memlock_limit()
        .map_err(|e| EnumError::os("getrlimit(RLIMIT_MEMLOCK)", &e))?;

    let host_mem = vec![NumaHostMem {
        numa: 0,
        node_total: Bytes(memsize),
        mem: HostMemInfo {
            total: Bytes(memsize),
            available,
            numa_free: Some(pages(vm.free)),
            // No cgroups on macOS.
            cgroup_limit: None,
            memlock_limit: memlock.map(Bytes),
            windows_lockable: None,
            wsl2: false,
            macos_wired_limit: wired,
            // Apple silicon gives user processes no large pages.
            large_pages: None,
            // Measured by step H5 (RFC-0010 §3.2), not E1.
            lock: None,
        },
    }];

    let power = power(api)?;
    let kernel_release = need_str(api, "kern.osrelease")?;
    let os_version = need_str(api, "kern.osproductversion")?;

    let (cpus, candidates) = host::number(
        node,
        vec![CpuDomain {
            numa: 0,
            name,
            class: DeviceClass::Cpu { isa: isa.level },
            features: isa.features,
            topology,
        }],
        // ADR-0014 capacity rules: Apple has no `HostPinned` tier.
        &[MemDomain {
            numa: 0,
            pinned: false,
        }],
    )?;
    Ok(MacHost {
        node: NodeFacts {
            arch,
            power: power.0,
            battery_present: power.1,
            kernel_release,
        },
        cpus,
        host_mem,
        env: MacEnv { os_version },
        candidates,
    })
}

/// The CPU topology from the performance levels. Level 0 is the fastest; on a part with more
/// than one level, the slowest level's cores are the efficiency cores, as on Linux.
///
/// The last-level cache is the sum over the core clusters of each level's highest cache
/// (`l3cachesize` where a level reports one, `l2cachesize` otherwise); a cluster is
/// `cpusperl2` (or `cpusperl3`) cores. Apple's system-level cache is not reported by any
/// sysctl, so it is not counted. `ccds` counts those clusters: the core complexes that share
/// a last-level cache. A one-level part without `hw.perflevel0.l2cachesize` uses
/// `hw.l2cachesize`, and a level without `cpusperl2` is one cluster.
fn topology(
    api: &dyn MacApi,
    isa: syntropy_core::resource::IsaFlags,
) -> Result<CpuTopology, EnumError> {
    let levels = need_int(api, "hw.nperflevels")?;
    if levels == 0 {
        return Err(EnumError::bad(
            "sysctl hw.nperflevels",
            "0 performance levels",
        ));
    }
    let mut physical_per_level = Vec::new();
    let mut llc = 0u64;
    let mut clusters = 0u64;
    for l in 0..levels {
        let key = |k: &str| format!("hw.perflevel{l}.{k}");
        let physical = need_int(api, &key("physicalcpu"))?;
        let (size, per) = match cache_int(api, &key("l3cachesize"))? {
            Some(s) if s > 0 => (s, cache_int(api, &key("cpusperl3"))?),
            _ => {
                // A one-level part (a virtual machine) may report the L2 only host-wide.
                let size = match cache_int(api, &key("l2cachesize"))? {
                    Some(s) => s,
                    None if levels == 1 => need_int(api, "hw.l2cachesize")?,
                    None => need_int(api, &key("l2cachesize"))?,
                };
                (size, cache_int(api, &key("cpusperl2"))?)
            }
        };
        let per = per.filter(|p| *p > 0).unwrap_or(physical.max(1));
        let n = physical.div_ceil(per);
        clusters += n;
        llc = llc.saturating_add(size.saturating_mul(n));
        physical_per_level.push(physical);
    }
    let physical = need_int(api, "hw.physicalcpu")?;
    let logical = need_int(api, "hw.logicalcpu")?;
    let efficiency = if levels > 1 {
        physical_per_level.last().copied().unwrap_or(0)
    } else {
        0
    };
    let to16 = |v: u64| u16::try_from(v).unwrap_or(u16::MAX);
    Ok(CpuTopology {
        numa: 0,
        physical_cores: to16(physical),
        logical_cpus: to16(logical),
        perf_cores: to16(physical.saturating_sub(efficiency)),
        efficiency_cores: to16(efficiency),
        llc: Bytes(llc),
        ccds: u8::try_from(clusters).unwrap_or(u8::MAX),
        isa,
        // AMX is x86-only.
        amx_permitted: false,
        // ADR-0012's reservation is made by the scheduler, not enumeration.
        reserved_cpus: CpuSet::default(),
    })
}

/// The power source and whether the Mac has a system battery. The Mac is on battery when
/// IOKit's providing source is `Battery Power`; the charge is the internal battery's
/// `Current Capacity` over `Max Capacity`.
fn power(api: &dyn MacApi) -> Result<(PowerSource, bool), EnumError> {
    const CALL: &str = "IOPSCopyPowerSourcesInfo";
    let ps = api.power_sources().map_err(|e| EnumError::os(CALL, &e))?;
    let battery = ps
        .sources
        .iter()
        .find(|s| s.kind.as_deref() == Some("InternalBattery") && s.present != Some(false));
    let source = if ps.providing.as_deref() == Some("Battery Power") {
        let b = battery.ok_or_else(|| EnumError::bad(CALL, "on battery power with no battery"))?;
        let (Some(cur), Some(max)) = (b.current_capacity, b.max_capacity) else {
            return Err(EnumError::bad(CALL, "battery without a capacity"));
        };
        if max <= 0 || cur < 0 {
            return Err(EnumError::bad(CALL, format!("capacity {cur}/{max}")));
        }
        let pct = (cur.saturating_mul(100) / max).clamp(0, 100);
        PowerSource::Battery {
            percent: u8::try_from(pct).unwrap_or(100),
        }
    } else {
        PowerSource::Mains
    };
    Ok((source, battery.is_some()))
}

/// Reads a sysctl. Every error, `EINVAL` included, is an error; only `ENOENT` (`Ok(None)`)
/// means the name is absent.
fn read(api: &dyn MacApi, name: &str) -> Result<Option<Vec<u8>>, EnumError> {
    api.sysctl(name, None)
        .map_err(|e| EnumError::os(format!("sysctl {name}"), &e))
}

/// A per-level cache sysctl (`hw.perflevel<N>.l2cachesize`, `l3cachesize`, `cpusperl2`,
/// `cpusperl3`). These alone may also be absent by failing with `EINVAL` or by holding an
/// empty value: macOS virtual machines report them so (the `macos-26` runner fails
/// `hw.perflevel0.l3cachesize` with `EINVAL`). [`topology`] then falls back as it documents.
fn cache_int(api: &dyn MacApi, name: &str) -> Result<Option<u64>, EnumError> {
    match api.sysctl(name, None) {
        Err(e) if e.kind() == io::ErrorKind::InvalidInput => Ok(None),
        Err(e) => Err(EnumError::os(format!("sysctl {name}"), &e)),
        Ok(None) => Ok(None),
        Ok(Some(b)) if b.is_empty() => Ok(None),
        Ok(Some(b)) => int_of(name, &b).map(Some),
    }
}

fn int_of(name: &str, b: &[u8]) -> Result<u64, EnumError> {
    decode_int(b).ok_or_else(|| {
        EnumError::bad(
            format!("sysctl {name}"),
            format!("{} bytes is not an integer", b.len()),
        )
    })
}

/// An integer sysctl: 4 or 8 native-endian bytes.
fn sysctl_int(api: &dyn MacApi, name: &str) -> Result<Option<u64>, EnumError> {
    match read(api, name)? {
        None => Ok(None),
        Some(b) => int_of(name, &b).map(Some),
    }
}

fn need_int(api: &dyn MacApi, name: &str) -> Result<u64, EnumError> {
    sysctl_int(api, name)?.ok_or_else(|| EnumError::bad(format!("sysctl {name}"), "no such name"))
}

fn need_str(api: &dyn MacApi, name: &str) -> Result<String, EnumError> {
    let b =
        read(api, name)?.ok_or_else(|| EnumError::bad(format!("sysctl {name}"), "no such name"))?;
    decode_str(&b).ok_or_else(|| EnumError::bad(format!("sysctl {name}"), "not UTF-8 text"))
}

/// A sysctl integer from its bytes. Negative 32-bit values are not expected in the names E1
/// reads; they would decode as large numbers, never as a guess.
pub(crate) fn decode_int(b: &[u8]) -> Option<u64> {
    match b.len() {
        4 => Some(u64::from(u32::from_ne_bytes(b.try_into().ok()?))),
        8 => Some(u64::from_ne_bytes(b.try_into().ok()?)),
        _ => None,
    }
}

/// A sysctl string from its bytes, without the trailing NUL.
pub(crate) fn decode_str(b: &[u8]) -> Option<String> {
    let b = b.strip_suffix(&[0]).unwrap_or(b);
    std::str::from_utf8(b).ok().map(|s| s.trim().to_owned())
}

/// The live Mac, through `syntropy_io::os::macos_host`. It has no way to write a sysctl:
/// [`MacApi::sysctl`] with a new value fails with `PermissionDenied`.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default)]
pub struct LiveMac;

#[cfg(target_os = "macos")]
impl MacApi for LiveMac {
    fn sysctl(&self, name: &str, new: Option<&[u8]>) -> io::Result<Option<Vec<u8>>> {
        if new.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "E1 never writes a sysctl",
            ));
        }
        syntropy_io::os::macos_host::sysctl(name)
    }

    fn vm_statistics(&self) -> io::Result<VmStats> {
        let v = syntropy_io::os::macos_host::vm_statistics()?;
        Ok(VmStats {
            page_size: v.page_size,
            free: v.free,
            active: v.active,
            inactive: v.inactive,
            wired: v.wired,
            purgeable: v.purgeable,
            speculative: v.speculative,
            compressor: v.compressor,
        })
    }

    fn memlock_limit(&self) -> io::Result<Option<u64>> {
        syntropy_io::os::macos_host::memlock_limit()
    }

    fn power_sources(&self) -> io::Result<PowerSources> {
        let p = syntropy_io::os::macos_host::power_sources()?;
        Ok(PowerSources {
            providing: p.providing,
            sources: p
                .sources
                .into_iter()
                .map(|s| PowerSourceDesc {
                    kind: s.kind,
                    state: s.state,
                    current_capacity: s.current_capacity,
                    max_capacity: s.max_capacity,
                    present: s.present,
                })
                .collect(),
        })
    }
}

/// Enumerates the live Mac.
#[cfg(target_os = "macos")]
pub fn enumerate_live(node: NodeId) -> Result<MacHost, EnumError> {
    enumerate(&LiveMac, node)
}

/// Enumerates the live Mac. This module's live path covers macOS only.
#[cfg(not(target_os = "macos"))]
pub fn enumerate_live(_node: NodeId) -> Result<MacHost, EnumError> {
    Err(EnumError::UnsupportedOs)
}
