// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory facts: `/proc/meminfo`, the cgroup memory limit, the memory-lock limit and
//! `CAP_IPC_LOCK`, and transparent huge pages.

use syntropy_core::budget::NodeMem;
use syntropy_core::units::Bytes;

use super::env::CgroupVersion;
use super::{at, parse, read, read_opt};
use crate::error::EnumError;
use crate::fs::HostFs;

/// `CAP_IPC_LOCK` is capability 14.
const CAP_IPC_LOCK: u64 = 1 << 14;
const CGROUP_ROOT: &str = "sys/fs/cgroup";

/// Host-wide memory facts, in bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MemFacts {
    pub total: u64,
    pub available: u64,
    pub free: u64,
    pub cgroup: CgroupVersion,
    /// The tightest `limit − usage` over this process's cgroup and its ancestors, where a v2
    /// level's limit is `min(memory.max, memory.high)` (RFC-0020 §5); `None` when no level
    /// sets a memory limit or the memory controller is off.
    pub cgroup_headroom: Option<u64>,
    /// `RLIMIT_MEMLOCK`'s soft limit; `None` when unlimited or `CAP_IPC_LOCK` is held
    /// (`HostMemInfo::memlock_limit`'s convention).
    pub memlock: Option<u64>,
    pub cap_ipc_lock: bool,
    /// The transparent huge page size when THP is `always` or `madvise`.
    pub large_pages: Option<u64>,
}

pub(super) fn facts(fs: &dyn HostFs) -> Result<MemFacts, EnumError> {
    const MEMINFO: &str = "proc/meminfo";
    let mi = read(fs, MEMINFO)?;
    let key = |k: &str| -> Result<u64, EnumError> {
        at(MEMINFO, parse::meminfo_kb(&mi, k))?
            .ok_or_else(|| EnumError::malformed(MEMINFO, format!("no {k}")))
    };
    let total = key("MemTotal")?;
    // Present since Linux 3.14; ADR-0020's floor is 5.15, so its absence is an error, not a
    // reason to estimate it.
    let available = key("MemAvailable")?;
    let free = key("MemFree")?;

    // `/proc/self/limits` and `/proc/self/status` give the same values `getrlimit` and
    // `capget` return for this process, and every process may read its own.
    const LIMITS: &str = "proc/self/limits";
    let soft = at(LIMITS, parse::memlock_soft(&read(fs, LIMITS)?))?;
    const STATUS: &str = "proc/self/status";
    let cap_ipc_lock = at(STATUS, parse::cap_eff(&read(fs, STATUS)?))? & CAP_IPC_LOCK != 0;
    let memlock = if cap_ipc_lock { None } else { soft };

    let (cgroup, cgroup_headroom) = cgroup(fs)?;
    Ok(MemFacts {
        total,
        available,
        free,
        cgroup,
        cgroup_headroom,
        memlock,
        cap_ipc_lock,
        large_pages: thp(fs)?,
    })
}

/// The cgroup version and the memory headroom it leaves this process.
///
/// v1 is used when a v1 hierarchy carries the memory controller (including hybrid
/// setups); otherwise v2 when `/proc/self/cgroup` has a `0::` line. The process's cgroup
/// directory is its path under the mount; when that directory does not exist (a container
/// whose cgroup namespace or bind mount shows its own cgroup at the mount root), the mount
/// root is the process's cgroup. Every level from there up to the mount root is checked and
/// the smallest `limit − usage` wins, since a parent's limit binds its children.
///
/// On v2 a level's limit is `min(memory.max, memory.high)` (RFC-0020 §5): above `memory.high`
/// the kernel throttles the cgroup and reclaims hard, so the budget stops there. Either file
/// may be `max` (no limit) or absent (`memory.high` on kernels or levels without it). v1 uses
/// `memory.limit_in_bytes` alone: RFC-0020 §5's v1 term has no soft limit, and v1's
/// `memory.soft_limit_in_bytes` only guides reclaim under global pressure.
fn cgroup(fs: &dyn HostFs) -> Result<(CgroupVersion, Option<u64>), EnumError> {
    const SELF: &str = "proc/self/cgroup";
    let Some(text) = read_opt(fs, SELF)? else {
        return Ok((CgroupVersion::None, None));
    };
    let lines = at(SELF, parse::cgroup_lines(&text))?;
    let (version, mount, path, limit_file, high_file, usage_file) = if let Some(l) = lines
        .iter()
        .find(|l| l.controllers.iter().any(|c| c == "memory"))
    {
        (
            CgroupVersion::V1,
            format!("{CGROUP_ROOT}/memory"),
            &l.path,
            "memory.limit_in_bytes",
            None,
            "memory.usage_in_bytes",
        )
    } else if let Some(l) = lines.iter().find(|l| l.hierarchy == 0) {
        (
            CgroupVersion::V2,
            CGROUP_ROOT.to_owned(),
            &l.path,
            "memory.max",
            Some("memory.high"),
            "memory.current",
        )
    } else {
        return Ok((CgroupVersion::None, None));
    };

    let parts: Vec<&str> = path
        .split('/')
        .filter(|p| !p.is_empty() && *p != "." && *p != "..")
        .collect();
    let mut depth = parts.len();
    if depth > 0 && !fs.exists(&format!("{mount}/{}", parts.join("/"))) {
        depth = 0;
    }
    let mut best: Option<u64> = None;
    loop {
        let dir = if depth == 0 {
            mount.clone()
        } else {
            format!("{mount}/{}", parts[..depth].join("/"))
        };
        let lim_path = format!("{dir}/{limit_file}");
        let max = match read_opt(fs, &lim_path)? {
            Some(t) => at(&lim_path, parse::cgroup_limit(&t))?,
            None => None,
        };
        let high = match high_file {
            Some(file) => {
                let high_path = format!("{dir}/{file}");
                match read_opt(fs, &high_path)? {
                    Some(t) => at(&high_path, parse::cgroup_limit(&t))?,
                    None => None,
                }
            }
            None => None,
        };
        let limit = match (max, high) {
            (Some(m), Some(h)) => Some(m.min(h)),
            (m, h) => m.or(h),
        };
        if let Some(limit) = limit {
            let use_path = format!("{dir}/{usage_file}");
            let usage = at(&use_path, parse::number::<u64>(&read(fs, &use_path)?))?;
            let headroom = limit.saturating_sub(usage);
            best = Some(best.map_or(headroom, |b| b.min(headroom)));
        }
        if depth == 0 {
            break;
        }
        depth -= 1;
    }
    Ok((version, best))
}

/// One domain's reclaimable memory from a `meminfo` file: a node's
/// (`Node <n> Active(file): …`) or, on a kernel without NUMA, `/proc/meminfo`. Every key is
/// required: the kernel has printed them in both files since long before ADR-0020's 5.15
/// floor, so an absent one is a malformed file, not a zero.
pub(super) fn node_mem(path: &str, text: &str) -> Result<NodeMem, EnumError> {
    let key = |k: &str| -> Result<Bytes, EnumError> {
        at(path, parse::meminfo_kb(text, k))?
            .map(Bytes)
            .ok_or_else(|| EnumError::malformed(path, format!("no {k}")))
    };
    Ok(NodeMem {
        mem_free: key("MemFree")?,
        active_file: key("Active(file)")?,
        inactive_file: key("Inactive(file)")?,
        sreclaimable: key("SReclaimable")?,
    })
}

/// The THP size when the policy lets this process use huge pages without privilege.
fn thp(fs: &dyn HostFs) -> Result<Option<u64>, EnumError> {
    const ENABLED: &str = "sys/kernel/mm/transparent_hugepage/enabled";
    const SIZE: &str = "sys/kernel/mm/transparent_hugepage/hpage_pmd_size";
    let Some(t) = read_opt(fs, ENABLED)? else {
        return Ok(None);
    };
    match parse::bracketed(&t) {
        Some("always" | "madvise") => {}
        Some(_) => return Ok(None),
        None => return Err(EnumError::malformed(ENABLED, "no [selected] policy")),
    }
    match read_opt(fs, SIZE)? {
        Some(s) => Ok(Some(at(SIZE, parse::number::<u64>(&s))?)),
        None => Ok(None),
    }
}
