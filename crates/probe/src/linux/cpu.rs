// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU facts: architecture, name, ISA, NUMA domains, cores, hybrid classification and the
//! last-level cache.

use std::collections::{BTreeMap, BTreeSet};

use syntropy_core::budget::NodeMem;
use syntropy_core::resource::{CpuArch, CpuSet, CpuTopology};
use syntropy_core::units::Bytes;

use super::isa::{self, Detector, Isa, IsaSource};
use super::mem::{self, MemFacts};
use super::{at, list_opt, parse, read, read_opt};
use crate::error::EnumError;
use crate::fs::HostFs;

const CPU: &str = "sys/devices/system/cpu";
const NODE: &str = "sys/devices/system/node";

/// Host-wide CPU facts.
#[derive(Clone, Debug)]
pub(super) struct CpuFacts {
    pub arch: CpuArch,
    pub name: String,
    pub isa: Isa,
    /// Logical CPUs the kernel has online.
    pub online: Vec<u32>,
    /// Logical CPUs that are efficiency cores, when the host is hybrid.
    pub efficiency: Option<BTreeSet<u32>>,
}

/// One NUMA domain: its online CPUs and its memory.
#[derive(Clone, Debug)]
pub(super) struct Domain {
    pub numa: u16,
    pub cpus: Vec<u32>,
    pub mem_total: u64,
    pub mem_free: u64,
    /// The reclaimable memory RFC-0020 §5's per-NUMA cap reads; `None` for a memoryless node.
    pub node_mem: Option<NodeMem>,
}

pub(super) fn facts(fs: &dyn HostFs, source: IsaSource) -> Result<CpuFacts, EnumError> {
    const INFO: &str = "proc/cpuinfo";
    let text = read(fs, INFO)?;
    let entries: Vec<_> = parse::cpuinfo(&text)
        .into_iter()
        .filter(|e| parse::field(e, "processor").is_some())
        .collect();
    let first = entries
        .first()
        .ok_or_else(|| EnumError::malformed(INFO, "no processor entries"))?;

    let arch = match source {
        IsaSource::Runtime => runtime_arch()?,
        IsaSource::CpuInfo => {
            if parse::field(first, "flags").is_some() {
                CpuArch::X86_64
            } else if parse::field(first, "Features").is_some() {
                CpuArch::Aarch64
            } else {
                return Err(EnumError::malformed(INFO, "no flags or Features line"));
            }
        }
    };
    let key = match arch {
        CpuArch::Aarch64 => "Features",
        _ => "flags",
    };
    let lists: Vec<Vec<&str>> = entries
        .iter()
        .map(|e| {
            parse::field(e, key)
                .unwrap_or("")
                .split_whitespace()
                .collect()
        })
        .collect();
    let common = isa::common_flags(&lists);
    let isa = Detector {
        source,
        cpuinfo: &common,
    }
    .isa(arch)?;

    let name = match parse::field(first, "model name") {
        Some(n) => n.to_owned(),
        None if arch == CpuArch::Aarch64 => arm_name(first)
            .ok_or_else(|| EnumError::malformed(INFO, "no model name or CPU part"))?,
        None => return Err(EnumError::malformed(INFO, "no model name")),
    };

    let online_path = format!("{CPU}/online");
    let online = at(&online_path, parse::cpulist(&read(fs, &online_path)?))?;
    if online.is_empty() {
        return Err(EnumError::malformed(&online_path, "no online CPUs"));
    }
    let efficiency = efficiency_cpus(fs, &online)?;
    Ok(CpuFacts {
        arch,
        name,
        isa,
        online,
        efficiency,
    })
}

fn runtime_arch() -> Result<CpuArch, EnumError> {
    match std::env::consts::ARCH {
        "x86_64" => Ok(CpuArch::X86_64),
        "aarch64" => Ok(CpuArch::Aarch64),
        other => Err(EnumError::UnsupportedArch {
            found: other.to_owned(),
        }),
    }
}

/// A name for an AArch64 CPU from `CPU implementer` and `CPU part`, since its cpuinfo has
/// no `model name`. Parts outside the table are named by their hex IDs.
fn arm_name(entry: &parse::CpuInfoEntry) -> Option<String> {
    let imp = parse::field(entry, "CPU implementer")?;
    let part = parse::field(entry, "CPU part")?;
    let known = match (imp, part) {
        ("0x41", "0xd03") => Some("ARM Cortex-A53"),
        ("0x41", "0xd05") => Some("ARM Cortex-A55"),
        ("0x41", "0xd08") => Some("ARM Cortex-A72"),
        ("0x41", "0xd0b") => Some("ARM Cortex-A76"),
        ("0x41", "0xd0c") => Some("ARM Neoverse-N1"),
        ("0x41", "0xd40") => Some("ARM Neoverse-V1"),
        ("0x41", "0xd49") => Some("ARM Neoverse-N2"),
        ("0x41", "0xd4f") => Some("ARM Neoverse-V2"),
        _ => None,
    };
    Some(known.map_or_else(|| format!("AArch64 {imp}:{part}"), str::to_owned))
}

/// Efficiency cores, by logical CPU. Intel hybrid parts expose an atom PMU whose `cpus` file
/// lists them. Elsewhere, when every online CPU reports `cpu_capacity` and the values
/// differ, the CPUs at the lowest capacity are the efficiency cores and the rest are
/// performance cores. `None` means a homogeneous CPU.
fn efficiency_cpus(fs: &dyn HostFs, online: &[u32]) -> Result<Option<BTreeSet<u32>>, EnumError> {
    const ATOM: &str = "sys/devices/cpu_atom/cpus";
    if let Some(text) = read_opt(fs, ATOM)? {
        let atom = at(ATOM, parse::cpulist(&text))?;
        return Ok(Some(atom.into_iter().collect()));
    }
    let mut caps = BTreeMap::new();
    for cpu in online {
        let path = format!("{CPU}/cpu{cpu}/cpu_capacity");
        match read_opt(fs, &path)? {
            Some(t) => {
                caps.insert(*cpu, at(&path, parse::number::<u32>(&t))?);
            }
            None => return Ok(None),
        }
    }
    let (Some(min), Some(max)) = (caps.values().min(), caps.values().max()) else {
        return Ok(None);
    };
    if min == max {
        return Ok(None);
    }
    Ok(Some(
        caps.iter()
            .filter(|(_, c)| *c == min)
            .map(|(cpu, _)| *cpu)
            .collect(),
    ))
}

/// NUMA domains from `/sys/devices/system/node`. Without that directory (a kernel built
/// without NUMA) the host is one domain 0 holding every online CPU and all of `/proc/meminfo`.
pub(super) fn domains(
    fs: &dyn HostFs,
    online: &[u32],
    mem: &MemFacts,
) -> Result<Vec<Domain>, EnumError> {
    let mut out = Vec::new();
    for name in list_opt(fs, NODE)? {
        let Some(n) = name.strip_prefix("node") else {
            continue;
        };
        let Ok(numa) = n.parse::<u16>() else {
            continue;
        };
        let cpus_path = format!("{NODE}/{name}/cpulist");
        let cpus = at(&cpus_path, parse::cpulist(&read(fs, &cpus_path)?))?
            .into_iter()
            .filter(|c| online.contains(c))
            .collect();
        let mi_path = format!("{NODE}/{name}/meminfo");
        let mi = read(fs, &mi_path)?;
        let mem_total = at(&mi_path, parse::meminfo_kb(&mi, "MemTotal"))?
            .ok_or_else(|| EnumError::malformed(&mi_path, "no MemTotal"))?;
        let mem_free = at(&mi_path, parse::meminfo_kb(&mi, "MemFree"))?
            .ok_or_else(|| EnumError::malformed(&mi_path, "no MemFree"))?;
        let node_mem = if mem_total > 0 {
            Some(mem::node_mem(&mi_path, &mi)?)
        } else {
            None
        };
        out.push(Domain {
            numa,
            cpus,
            mem_total,
            mem_free,
            node_mem,
        });
    }
    if out.is_empty() {
        const MEMINFO: &str = "proc/meminfo";
        out.push(Domain {
            numa: 0,
            cpus: online.to_vec(),
            mem_total: mem.total,
            mem_free: mem.free,
            node_mem: Some(mem::node_mem(MEMINFO, &read(fs, MEMINFO)?)?),
        });
    }
    out.sort_by_key(|d| d.numa);
    Ok(out)
}

/// The topology of one domain's CPU device.
pub(super) fn topology(
    fs: &dyn HostFs,
    d: &Domain,
    cpu: &CpuFacts,
    amx_permitted: bool,
) -> Result<CpuTopology, EnumError> {
    // Physical cores: logical CPUs that share a core list are one core.
    let mut cores: BTreeSet<Vec<u32>> = BTreeSet::new();
    for c in &d.cpus {
        let base = format!("{CPU}/cpu{c}/topology");
        let path = format!("{base}/core_cpus_list");
        let text = match read_opt(fs, &path)? {
            Some(t) => (path, t),
            None => {
                let old = format!("{base}/thread_siblings_list");
                let t = read(fs, &old)?;
                (old, t)
            }
        };
        cores.insert(at(&text.0, parse::cpulist(&text.1))?);
    }
    let efficiency_cores = match &cpu.efficiency {
        Some(e) => cores
            .iter()
            .filter(|core| core.iter().any(|c| e.contains(c)))
            .count(),
        None => 0,
    };
    let physical = cores.len();
    Ok(CpuTopology {
        numa: d.numa,
        physical_cores: count(physical),
        logical_cpus: count(d.cpus.len()),
        perf_cores: count(physical - efficiency_cores),
        efficiency_cores: count(efficiency_cores),
        llc: Bytes(llc(fs, &d.cpus)?),
        // Not derivable from sysfs alone: an L3 slice and a fabric link are not the same
        // unit on every part. 0 is the documented "unknown".
        ccds: 0,
        isa: cpu.isa.flags,
        // The tile-data permission request, made by the caller (PLC-E1-T14).
        amx_permitted,
        // ADR-0012's reservation is made by the scheduler, not enumeration.
        reserved_cpus: CpuSet::default(),
    })
}

fn count(n: usize) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

/// The last-level cache over a domain's CPUs: the sum of the distinct highest-level data or
/// unified caches any of them uses. 0 (unknown) when the kernel exposes no cache directory.
/// An index directory missing one of its four files is skipped: some hypervisors expose
/// partial cache entries, and a missing file there means "not reported".
fn llc(fs: &dyn HostFs, cpus: &[u32]) -> Result<u64, EnumError> {
    // (level, shared CPUs) → size.
    let mut caches: BTreeMap<(u32, Vec<u32>), u64> = BTreeMap::new();
    for c in cpus {
        let dir = format!("{CPU}/cpu{c}/cache");
        for idx in list_opt(fs, &dir)? {
            if !idx.starts_with("index") {
                continue;
            }
            let base = format!("{dir}/{idx}");
            let paths = ["type", "level", "size", "shared_cpu_list"].map(|f| format!("{base}/{f}"));
            let mut texts = Vec::with_capacity(4);
            for p in &paths {
                match read_opt(fs, p)? {
                    Some(t) => texts.push(t),
                    None => break,
                }
            }
            let [ty, level, size, shared] = texts.as_slice() else {
                continue;
            };
            if ty.trim() == "Instruction" {
                continue;
            }
            let level = at(&paths[1], parse::number::<u32>(level))?;
            let size = at(&paths[2], parse::cache_size(size))?;
            let shared = at(&paths[3], parse::cpulist(shared))?;
            caches.insert((level, shared), size);
        }
    }
    let Some(top) = caches.keys().map(|(l, _)| *l).max() else {
        return Ok(0);
    };
    Ok(caches
        .iter()
        .filter(|((l, _), _)| *l == top)
        .map(|(_, s)| *s)
        .sum())
}
