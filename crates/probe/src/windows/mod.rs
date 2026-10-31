// SPDX-License-Identifier: MIT OR Apache-2.0

//! E1 host enumeration on Windows (RFC-0010 §2.1, §2.2; PLC-E1-T03): CPU topology, hybrid
//! cores, caches and NUMA masks from `GetLogicalProcessorInformationEx`, ISA from CPUID, RAM
//! from `GlobalMemoryStatusEx` and per node from `GetNumaAvailableMemoryNodeEx`, the
//! working-set ceiling obtainable without privilege, the power status, and the OS build.
//!
//! The calls go through a [`WinApi`], so the same code runs on the live host (`LiveWindows`,
//! Windows only, which calls `syntropy_io::os::windows_host`) and on recorded responses
//! ([`WinFixture`]).
//!
//! **Domains.** Each NUMA node with logical processors gets one CPU device, and each node
//! gets a `HostPinned` and a `HostPageable` tier (RFC-0010 §2.2). A node's RAM
//! (`NumaHostMem::node_total`) is `ullTotalPhys` on a single-node host; on a multi-node host
//! Windows reports no per-node total, so it comes from the ACPI SRAT's memory ranges, mapped
//! to nodes with `GetNumaProximityNodeEx`.
//!
//! **Working-set ceiling.** `HostMemInfo::windows_lockable` is the largest working-set
//! minimum this process can set without privilege (RFC-0020 §5, "Windows before H5"). The
//! probe raises the (soft) minimum by binary search in 64 MiB steps up to `ullAvailPhys` and
//! puts the original limits back, retrying once. Raising the minimum allocates nothing;
//! Windows refuses a minimum it cannot guarantee. The value is an upper bound on what a lock
//! can reach, not a lock budget: step H5's `lockable_measured` replaces it.
//!
//! **Job objects.** `HostMemInfo::cgroup_limit` is the headroom the process's job leaves
//! ([`job_headroom`]), `None` when it is in no job or the job sets no memory limit.

mod glpi;
mod record;
mod srat;

use std::collections::{BTreeMap, BTreeSet};
use std::io;

use syntropy_core::budget::NodeMem;
use syntropy_core::ids::NodeId;
use syntropy_core::resource::caps::PowerSource;
use syntropy_core::resource::ordinal::Enumerated;
use syntropy_core::resource::{CpuArch, CpuSet, CpuTopology, DeviceClass, DeviceInfo, HostMemInfo};
use syntropy_core::units::Bytes;

use crate::error::EnumError;
use crate::host::{self, CpuDomain, MemDomain, NodeFacts, NumaHostMem, count};
use crate::linux::isa::{Detector, IsaSource};

pub use record::{WinFixture, WinRecorder};

/// `GlobalMemoryStatusEx`'s physical-memory fields, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WinMemory {
    /// `ullTotalPhys`.
    pub total: u64,
    /// `ullAvailPhys`.
    pub available: u64,
}

/// `GetSystemPowerStatus`'s fields, as the call reports them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WinPower {
    /// `ACLineStatus`: 0 offline, 1 online, 255 unknown.
    pub ac_line: u8,
    /// `BatteryFlag`: 128 no system battery, 255 unknown.
    pub battery_flag: u8,
    /// `BatteryLifePercent`, 255 unknown.
    pub battery_percent: u8,
}

/// The CPU's architecture, name and ISA flags, from CPUID on the live host.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WinCpu {
    /// `x86_64` or `aarch64`.
    pub arch: String,
    /// The CPUID brand string.
    pub name: String,
    /// The features found, by their `/proc/cpuinfo` names, so one ISA rule serves every OS.
    pub flags: BTreeSet<String>,
}

/// The Windows calls E1 makes.
pub trait WinApi {
    /// The `GetLogicalProcessorInformationEx(RelationAll)` buffer.
    fn processor_information(&self) -> io::Result<Vec<u8>>;
    /// `GlobalMemoryStatusEx`.
    fn memory_status(&self) -> io::Result<WinMemory>;
    /// `GetNumaAvailableMemoryNodeEx(node)`.
    fn numa_available(&self, node: u16) -> io::Result<u64>;
    /// The ACPI SRAT (`GetSystemFirmwareTable`), `None` when the firmware has none.
    fn srat(&self) -> io::Result<Option<Vec<u8>>>;
    /// `GetNumaProximityNodeEx(proximity)`.
    fn proximity_node(&self, proximity: u32) -> Option<u16>;
    /// `GetSystemPowerStatus`.
    fn power_status(&self) -> io::Result<WinPower>;
    /// The working-set ceiling probe's result, with `upper` (`ullAvailPhys`) the highest
    /// minimum it tries.
    fn working_set_ceiling(&self, upper: u64) -> io::Result<u64>;
    /// `RtlGetVersion`: (major, minor, build).
    fn version(&self) -> io::Result<(u32, u32, u32)>;
    /// The CPU's identification and ISA flags.
    fn cpu(&self) -> io::Result<WinCpu>;
    /// The memory limits of this process's job object; `None` when it is in no job.
    fn job_memory(&self) -> io::Result<Option<WinJob>>;
}

/// A job object's memory limits, from `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WinJob {
    /// `JobMemoryLimit`, when the job sets one.
    pub job_limit: Option<u64>,
    /// `ProcessMemoryLimit`, when the job sets one.
    pub process_limit: Option<u64>,
    /// `PeakJobMemoryUsed`.
    pub peak_job: u64,
    /// `PeakProcessMemoryUsed`.
    pub peak_process: u64,
}

/// Windows environment facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowsEnv {
    /// The OS build; IoRing needs 22000 or later (RFC-0010 §2.2, Environment row).
    pub build: u32,
}

/// What E1 found on one Windows host.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowsHost {
    /// The `NodeInfo` facts E1 owns.
    pub node: NodeFacts,
    /// One CPU device per NUMA node with logical processors, with provisional ordinals as
    /// [`crate::linux::LinuxHost::cpus`] explains.
    pub cpus: Vec<DeviceInfo>,
    /// One record per NUMA node, in node order.
    pub host_mem: Vec<NumaHostMem>,
    /// Environment facts.
    pub env: WindowsEnv,
    /// E1's share of the ordinal input.
    pub candidates: Enumerated,
}

const GLPI: &str = "GetLogicalProcessorInformationEx";
/// The working-set probe's step.
pub const CEILING_STEP: u64 = 64 * 1024 * 1024;

/// Enumerates the Windows host behind `api`. `node` fills the IDs.
pub fn enumerate(api: &dyn WinApi, node: NodeId) -> Result<WindowsHost, EnumError> {
    let cpu = api.cpu().map_err(|e| EnumError::os("CPUID", &e))?;
    let arch = match cpu.arch.as_str() {
        "x86_64" => CpuArch::X86_64,
        // Windows on Arm is not an ADR-0020 platform: its ISA would need
        // `IsProcessorFeaturePresent`, which E1 does not read.
        other => {
            return Err(EnumError::UnsupportedArch {
                found: other.to_owned(),
            });
        }
    };
    let isa = Detector {
        source: IsaSource::CpuInfo,
        cpuinfo: &cpu.flags,
    }
    .isa(arch)?;

    let buf = api
        .processor_information()
        .map_err(|e| EnumError::os(GLPI, &e))?;
    let procs = glpi::parse(&buf).map_err(|w| EnumError::bad(GLPI, w))?;

    let mem = api
        .memory_status()
        .map_err(|e| EnumError::os("GlobalMemoryStatusEx", &e))?;
    let lockable = api
        .working_set_ceiling(mem.available)
        .map_err(|e| EnumError::os("SetProcessWorkingSetSizeEx", &e))?;

    let mut nodes: Vec<&glpi::Node> = procs.nodes.iter().collect();
    nodes.sort_by_key(|n| n.number);
    let totals = node_totals(api, &nodes, mem.total)?;
    let job = api
        .job_memory()
        .map_err(|e| EnumError::os("QueryInformationJobObject", &e))?;
    let cgroup_limit = job.and_then(job_headroom).map(Bytes);

    let mut cpus = Vec::new();
    let mut domains = Vec::new();
    let mut host_mem = Vec::new();
    for n in &nodes {
        let numa = u16::try_from(n.number)
            .map_err(|_| EnumError::bad(GLPI, format!("NUMA node {}", n.number)))?;
        if !n.lps.is_empty() {
            cpus.push(CpuDomain {
                numa,
                name: cpu.name.clone(),
                class: DeviceClass::Cpu { isa: isa.level },
                features: isa.features,
                topology: topology(&procs, n, numa, isa.flags),
            });
        }
        domains.push(MemDomain { numa, pinned: true });
        let free = api
            .numa_available(numa)
            .map_err(|e| EnumError::os(format!("GetNumaAvailableMemoryNodeEx({numa})"), &e))?;
        host_mem.push(NumaHostMem {
            numa,
            node_total: Bytes(totals.get(&numa).copied().unwrap_or(0)),
            mem: HostMemInfo {
                total: Bytes(mem.total),
                available: Bytes(mem.available),
                numa_free: Some(Bytes(free)),
                // RFC-0020 §5's Windows term: the job object's memory limit less its use.
                // `None`: the process is in no job, or its job sets no memory limit.
                cgroup_limit,
                // `RLIMIT_MEMLOCK` is a POSIX limit; Windows bounds locking by the working
                // set, which `windows_lockable` carries.
                memlock_limit: None,
                windows_lockable: Some(Bytes(lockable)),
                wsl2: false,
                macos_wired_limit: None,
                // Large pages need `SeLockMemoryPrivilege`, which standard users lack.
                large_pages: None,
                // Measured by step H5 (RFC-0010 §3.2), not E1.
                lock: None,
            },
            // Windows' available memory per node already counts standby pages.
            node_mem: Some(NodeMem {
                mem_free: Bytes(free),
                ..NodeMem::default()
            }),
        });
    }

    let p = api
        .power_status()
        .map_err(|e| EnumError::os("GetSystemPowerStatus", &e))?;
    let (power, battery_present) = power(p)?;
    let (major, minor, build) = api
        .version()
        .map_err(|e| EnumError::os("RtlGetVersion", &e))?;

    let (cpus, candidates) = host::number(node, cpus, &domains)?;
    Ok(WindowsHost {
        node: NodeFacts {
            arch,
            power,
            battery_present,
            kernel_release: format!("{major}.{minor}.{build}"),
        },
        cpus,
        host_mem,
        env: WindowsEnv { build },
        candidates,
    })
}

/// Each node's RAM: the whole host on one node, the SRAT's ranges otherwise.
fn node_totals(
    api: &dyn WinApi,
    nodes: &[&glpi::Node],
    total: u64,
) -> Result<BTreeMap<u16, u64>, EnumError> {
    const SRAT: &str = "GetSystemFirmwareTable(ACPI, SRAT)";
    let mut out = BTreeMap::new();
    if nodes.len() == 1 {
        out.insert(u16::try_from(nodes[0].number).unwrap_or(0), total);
        return Ok(out);
    }
    let table = api
        .srat()
        .map_err(|e| EnumError::os(SRAT, &e))?
        .ok_or_else(|| EnumError::bad(SRAT, "a multi-node host without an SRAT"))?;
    let by_domain = srat::memory_by_domain(&table).map_err(|w| EnumError::bad(SRAT, w))?;
    for (pd, bytes) in by_domain {
        let node = api.proximity_node(pd).ok_or_else(|| {
            EnumError::bad(
                "GetNumaProximityNodeEx",
                format!("no node for proximity domain {pd}"),
            )
        })?;
        let e = out.entry(node).or_insert(0u64);
        *e = e.saturating_add(bytes);
    }
    for n in nodes {
        let numa = u16::try_from(n.number).unwrap_or(u16::MAX);
        if !out.contains_key(&numa) {
            return Err(EnumError::bad(
                SRAT,
                format!("no memory range for node {numa}"),
            ));
        }
    }
    Ok(out)
}

/// One node's CPU topology. A core belongs to the node holding its logical processors.
/// On a hybrid part (`EfficiencyClass` differs between cores) the cores of the lowest class
/// are efficiency cores, as on Linux. The last-level cache is the sum of the distinct caches
/// at the highest data or unified level over the node's processors; `ccds` counts them, the
/// core complexes sharing one last-level cache.
fn topology(
    p: &glpi::Processors,
    node: &glpi::Node,
    numa: u16,
    isa: syntropy_core::resource::IsaFlags,
) -> CpuTopology {
    let cores: Vec<&glpi::Core> = p
        .cores
        .iter()
        .filter(|c| !c.lps.is_disjoint(&node.lps))
        .collect();
    let classes: BTreeSet<u8> = p.cores.iter().map(|c| c.efficiency_class).collect();
    let efficiency = if classes.len() > 1 {
        let low = classes.first().copied().unwrap_or(0);
        cores.iter().filter(|c| c.efficiency_class == low).count()
    } else {
        0
    };
    let logical: usize = cores
        .iter()
        .map(|c| c.lps.intersection(&node.lps).count())
        .sum();
    let caches: Vec<&glpi::Cache> = p
        .caches
        .iter()
        .filter(|c| !c.lps.is_disjoint(&node.lps))
        .collect();
    let top = caches.iter().map(|c| c.level).max();
    let llc: Vec<&&glpi::Cache> = caches.iter().filter(|c| Some(c.level) == top).collect();
    CpuTopology {
        numa,
        physical_cores: count(cores.len()),
        logical_cpus: count(logical),
        perf_cores: count(cores.len() - efficiency),
        efficiency_cores: count(efficiency),
        llc: Bytes(llc.iter().map(|c| c.size).sum()),
        ccds: u8::try_from(llc.len()).unwrap_or(u8::MAX),
        isa,
        // The AMX tile-data permission is a Linux request (`arch_prctl`); Windows enables the
        // state for every thread through XSAVE, and the CPU provider confirms it at E5.
        amx_permitted: false,
        reserved_cpus: CpuSet::default(),
    }
}

/// The headroom a job's memory limits leave: the tighter of `JobMemoryLimit − PeakJobMemoryUsed`
/// and `ProcessMemoryLimit − PeakProcessMemoryUsed`. Windows reports only peak commit, which is
/// at least the current one, so this never overstates the headroom. `None` when the job sets
/// no memory limit.
pub fn job_headroom(j: WinJob) -> Option<u64> {
    let job = j.job_limit.map(|l| l.saturating_sub(j.peak_job));
    let process = j.process_limit.map(|l| l.saturating_sub(j.peak_process));
    match (job, process) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// The power source and whether there is a system battery. On battery only when the AC line
/// is offline. `GetSystemPowerStatus` marks an unknown line status or battery state with 255,
/// which is an error here, not a guess.
fn power(p: WinPower) -> Result<(PowerSource, bool), EnumError> {
    const CALL: &str = "GetSystemPowerStatus";
    if p.ac_line == 255 {
        return Err(EnumError::bad(CALL, "AC line status unknown (255)"));
    }
    if p.battery_flag == 255 {
        return Err(EnumError::bad(CALL, "battery status unknown (255)"));
    }
    let battery_present = p.battery_flag != 128;
    let source = if p.ac_line == 0 {
        if p.battery_percent > 100 {
            return Err(EnumError::bad(
                "GetSystemPowerStatus",
                "on battery with an unknown charge",
            ));
        }
        PowerSource::Battery {
            percent: p.battery_percent,
        }
    } else {
        PowerSource::Mains
    };
    Ok((source, battery_present))
}

/// The largest value in `[lo, hi]`, in steps of `step` above `lo`, that `accepts` takes,
/// assuming it accepts everything below a threshold and nothing above. `lo` is known good.
pub fn ceiling_search(lo: u64, hi: u64, step: u64, mut accepts: impl FnMut(u64) -> bool) -> u64 {
    let step = step.max(1);
    if hi <= lo {
        return lo;
    }
    // Steps above `lo`: the answer is `lo + k * step` for the largest accepted k.
    let (mut good, mut bad) = (0u64, (hi - lo) / step + 1);
    while bad - good > 1 {
        let mid = good + (bad - good) / 2;
        if accepts(lo + mid * step) {
            good = mid;
        } else {
            bad = mid;
        }
    }
    lo + good * step
}

/// The live host, through `syntropy_io::os::windows_host`.
#[cfg(windows)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LiveWindows;

#[cfg(windows)]
impl WinApi for LiveWindows {
    fn processor_information(&self) -> io::Result<Vec<u8>> {
        syntropy_io::os::windows_host::logical_processor_information()
    }

    fn memory_status(&self) -> io::Result<WinMemory> {
        let m = syntropy_io::os::windows_host::memory_status()?;
        Ok(WinMemory {
            total: m.total_phys,
            available: m.avail_phys,
        })
    }

    fn numa_available(&self, node: u16) -> io::Result<u64> {
        syntropy_io::os::windows_host::numa_available_memory(node)
    }

    fn srat(&self) -> io::Result<Option<Vec<u8>>> {
        syntropy_io::os::windows_host::firmware_table(
            u32::from_be_bytes(*b"ACPI"),
            u32::from_le_bytes(*b"SRAT"),
        )
    }

    fn proximity_node(&self, proximity: u32) -> Option<u16> {
        syntropy_io::os::windows_host::numa_proximity_node(proximity)
    }

    fn power_status(&self) -> io::Result<WinPower> {
        let p = syntropy_io::os::windows_host::power_status()?;
        Ok(WinPower {
            ac_line: p.ac_line,
            battery_flag: p.battery_flag,
            battery_percent: p.battery_percent,
        })
    }

    fn working_set_ceiling(&self, upper: u64) -> io::Result<u64> {
        use syntropy_io::os::windows_host::{process_working_set, set_process_working_set};
        let orig = process_working_set()?;
        let found = ceiling_search(orig.min, upper, CEILING_STEP, |min| {
            set_process_working_set(min, min.max(orig.max)).is_ok()
        });
        // Put the original limits back; the probe must leave the process as it found it.
        // One retry, then an error naming the limits that could not be restored.
        if set_process_working_set(orig.min, orig.max).is_err()
            && let Err(e) = set_process_working_set(orig.min, orig.max)
        {
            return Err(io::Error::new(
                e.kind(),
                format!(
                    "could not restore the working-set limits min {} max {}: {e}",
                    orig.min, orig.max
                ),
            ));
        }
        Ok(found)
    }

    fn job_memory(&self) -> io::Result<Option<WinJob>> {
        Ok(
            syntropy_io::os::windows_host::job_memory()?.map(|j| WinJob {
                job_limit: j.job_limit,
                process_limit: j.process_limit,
                peak_job: j.peak_job,
                peak_process: j.peak_process,
            }),
        )
    }

    fn version(&self) -> io::Result<(u32, u32, u32)> {
        let v = syntropy_io::os::windows_host::os_version()?;
        Ok((v.major, v.minor, v.build))
    }

    fn cpu(&self) -> io::Result<WinCpu> {
        live_cpu()
    }
}

/// CPUID on x86-64: the brand string, and each feature the ISA rule checks under its
/// `/proc/cpuinfo` name. The standard library's detection covers all but LAHF-SAHF and the
/// AMX bits, which come from CPUID leaves 0x8000_0001 and 7.
#[cfg(all(windows, target_arch = "x86_64"))]
fn live_cpu() -> io::Result<WinCpu> {
    use std::arch::is_x86_feature_detected as d;
    use std::arch::x86_64::{__cpuid, __cpuid_count};

    let mut flags = BTreeSet::new();
    let mut add = |name: &str, on: bool| {
        if on {
            flags.insert(name.to_owned());
        }
    };
    add("cx16", d!("cmpxchg16b"));
    add("popcnt", d!("popcnt"));
    add("sse4_1", d!("sse4.1"));
    add("sse4_2", d!("sse4.2"));
    add("ssse3", d!("ssse3"));
    add("avx2", d!("avx2"));
    add("f16c", d!("f16c"));
    add("fma", d!("fma"));
    add("avx512f", d!("avx512f"));
    add("avx512bw", d!("avx512bw"));
    add("avx512dq", d!("avx512dq"));
    add("avx512vl", d!("avx512vl"));
    add("avx512_vnni", d!("avx512vnni"));
    add("avx512_bf16", d!("avx512bf16"));
    add("avx512_fp16", d!("avx512fp16"));
    add("avx_vnni", d!("avxvnni"));
    let max_ext = __cpuid(0x8000_0000).eax;
    if max_ext >= 0x8000_0001 {
        add("lahf_lm", __cpuid(0x8000_0001).ecx & 1 != 0);
    }
    if __cpuid(0).eax >= 7 {
        let l7 = __cpuid_count(7, 0);
        add("amx_bf16", l7.edx & (1 << 22) != 0);
        add("amx_int8", l7.edx & (1 << 25) != 0);
    }
    if max_ext < 0x8000_0004 {
        return Err(io::Error::other("CPUID has no brand string"));
    }
    let mut brand = Vec::with_capacity(48);
    for leaf in 0x8000_0002..=0x8000_0004u32 {
        let r = __cpuid(leaf);
        for w in [r.eax, r.ebx, r.ecx, r.edx] {
            brand.extend_from_slice(&w.to_le_bytes());
        }
    }
    let name = String::from_utf8_lossy(&brand)
        .trim_matches(char::from(0))
        .trim()
        .to_owned();
    Ok(WinCpu {
        arch: "x86_64".into(),
        name,
        flags,
    })
}

/// Windows on Arm is not a platform ADR-0020 lists.
#[cfg(all(windows, not(target_arch = "x86_64")))]
fn live_cpu() -> io::Result<WinCpu> {
    Ok(WinCpu {
        arch: std::env::consts::ARCH.into(),
        ..WinCpu::default()
    })
}

/// Enumerates the live host.
#[cfg(windows)]
pub fn enumerate_live(node: NodeId) -> Result<WindowsHost, EnumError> {
    enumerate(&LiveWindows, node)
}

/// Enumerates the live host. This module's live path covers Windows only.
#[cfg(not(windows))]
pub fn enumerate_live(_node: NodeId) -> Result<WindowsHost, EnumError> {
    Err(EnumError::UnsupportedOs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceiling_search_finds_the_threshold() {
        let step = 64;
        for threshold in [0u64, 63, 64, 1000, 4096, 10_000] {
            let found = ceiling_search(0, 4096, step, |v| v <= threshold);
            let want = (threshold.min(4096) / step) * step;
            assert_eq!(found, want, "threshold {threshold}");
        }
        // `lo` is kept when nothing above it is accepted, and when hi ≤ lo.
        assert_eq!(ceiling_search(200, 4096, 64, |_| false), 200);
        assert_eq!(ceiling_search(200, 100, 64, |_| true), 200);
        // Everything accepted: the last step at or below `hi`.
        assert_eq!(ceiling_search(10, 1000, 100, |_| true), 910);
    }

    #[test]
    fn job_headroom_rules() {
        let none = WinJob {
            peak_job: 5,
            peak_process: 3,
            ..WinJob::default()
        };
        assert_eq!(job_headroom(none), None);
        let job = WinJob {
            job_limit: Some(100),
            ..none
        };
        assert_eq!(job_headroom(job), Some(95));
        let both = WinJob {
            process_limit: Some(50),
            ..job
        };
        assert_eq!(job_headroom(both), Some(47));
        let over = WinJob {
            job_limit: Some(4),
            ..none
        };
        assert_eq!(job_headroom(over), Some(0));
    }

    #[test]
    fn power_rules() {
        let desk = WinPower {
            ac_line: 1,
            battery_flag: 128,
            battery_percent: 255,
        };
        assert_eq!(power(desk).unwrap(), (PowerSource::Mains, false));
        // 255 is "unknown": an error, not a value.
        for (ac_line, battery_flag) in [(255, 128), (1, 255), (255, 255)] {
            let p = WinPower {
                ac_line,
                battery_flag,
                battery_percent: 255,
            };
            assert!(power(p).is_err(), "{p:?}");
        }
        let laptop = WinPower {
            ac_line: 0,
            battery_flag: 0,
            battery_percent: 41,
        };
        assert_eq!(
            power(laptop).unwrap(),
            (PowerSource::Battery { percent: 41 }, true)
        );
        let charging = WinPower {
            ac_line: 1,
            battery_flag: 8,
            battery_percent: 90,
        };
        assert_eq!(power(charging).unwrap(), (PowerSource::Mains, true));
        assert!(
            power(WinPower {
                ac_line: 0,
                battery_flag: 0,
                battery_percent: 255
            })
            .is_err()
        );
    }
}
