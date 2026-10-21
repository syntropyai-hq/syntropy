// SPDX-License-Identifier: MIT OR Apache-2.0

//! Environment facts: WSL2, containers, the io_uring sysctl, seccomp, and the power source.

use syntropy_core::resource::caps::PowerSource;

use super::{at, list_opt, parse, read, read_opt};
use crate::error::EnumError;
use crate::fs::HostFs;

/// Environment facts the tier and I/O rules read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostEnv {
    /// `/dev/dxg` exists: the host is WSL2 and its pinned-allocation cap applies
    /// (RFC-0001 §9.2, ADR-0020).
    pub wsl2: bool,
    /// The process runs in a container, when detectable.
    pub container: Option<ContainerKind>,
    /// `/proc/sys/kernel/io_uring_disabled`. This is half of RFC-0010 §2.2's "io_uring
    /// allowed" test; the other half, `io_uring_setup` succeeding, is a system call and
    /// belongs to PLC-E1-T14.
    pub io_uring: IoUringSysctl,
    /// A seccomp filter is installed on this process (`Seccomp: 2` in `/proc/self/status`).
    /// Container runtimes' default filters may block io_uring (ADR-0020), so a filtered
    /// process cannot assume it is allowed even when the sysctl is 0.
    pub seccomp_filtered: bool,
    /// Which cgroup hierarchy holds this process's memory limit.
    pub cgroup: CgroupVersion,
    /// The process holds `CAP_IPC_LOCK`, so locking is not limited by `RLIMIT_MEMLOCK`.
    pub cap_ipc_lock: bool,
}

/// A container runtime.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContainerKind {
    /// `/.dockerenv` exists, or PID 1's cgroup names docker.
    Docker,
    /// `/run/.containerenv` exists.
    Podman,
    /// PID 1's cgroup names another runtime (kubepods, containerd, lxc).
    Other,
}

/// The `kernel.io_uring_disabled` sysctl (Linux 6.6+).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoUringSysctl {
    /// The file does not exist: a kernel before the knob, where the sysctl blocks nothing.
    Absent,
    /// 0: every process may create rings.
    Enabled,
    /// 1: only members of `kernel.io_uring_group` (and `CAP_SYS_ADMIN`) may.
    GroupOnly {
        /// The group, or `None` for the kernel's "-1" (no group).
        gid: Option<u32>,
        /// This process's effective or supplementary groups include it.
        member: bool,
    },
    /// 2: no process may.
    Disabled,
}

/// Which cgroup hierarchy holds this process's memory limit.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CgroupVersion {
    /// No cgroup information, or no hierarchy this module reads.
    None,
    /// A v1 hierarchy with the memory controller.
    V1,
    /// The v2 unified hierarchy.
    V2,
}

pub(super) fn facts(fs: &dyn HostFs) -> Result<HostEnv, EnumError> {
    const STATUS: &str = "proc/self/status";
    let status = read(fs, STATUS)?;
    let seccomp_filtered = match parse::status_field(&status, "Seccomp") {
        Some(v) => at(STATUS, parse::number::<u8>(v))? == 2,
        None => false,
    };
    Ok(HostEnv {
        wsl2: fs.exists("dev/dxg"),
        container: container(fs)?,
        io_uring: io_uring(fs, &status)?,
        seccomp_filtered,
        // Filled from the memory facts by the caller.
        cgroup: CgroupVersion::None,
        cap_ipc_lock: false,
    })
}

fn container(fs: &dyn HostFs) -> Result<Option<ContainerKind>, EnumError> {
    if fs.exists(".dockerenv") {
        return Ok(Some(ContainerKind::Docker));
    }
    if fs.exists("run/.containerenv") {
        return Ok(Some(ContainerKind::Podman));
    }
    // PID 1's cgroup is world-readable. Under cgroup v1 without a cgroup namespace it names
    // the runtime.
    if let Some(t) = read_opt(fs, "proc/1/cgroup")? {
        if t.contains("/docker") {
            return Ok(Some(ContainerKind::Docker));
        }
        if ["/kubepods", "/containerd", "/lxc"]
            .iter()
            .any(|m| t.contains(m))
        {
            return Ok(Some(ContainerKind::Other));
        }
    }
    Ok(None)
}

fn io_uring(fs: &dyn HostFs, status: &str) -> Result<IoUringSysctl, EnumError> {
    const DISABLED: &str = "proc/sys/kernel/io_uring_disabled";
    const GROUP: &str = "proc/sys/kernel/io_uring_group";
    let Some(t) = read_opt(fs, DISABLED)? else {
        return Ok(IoUringSysctl::Absent);
    };
    match at(DISABLED, parse::number::<u8>(&t))? {
        0 => Ok(IoUringSysctl::Enabled),
        1 => {
            let g = at(GROUP, parse::number::<i64>(&read(fs, GROUP)?))?;
            let gid = u32::try_from(g).ok();
            let member = match gid {
                Some(gid) => at("proc/self/status", parse::groups(status))?.contains(&gid),
                None => false,
            };
            Ok(IoUringSysctl::GroupOnly { gid, member })
        }
        2 => Ok(IoUringSysctl::Disabled),
        n => Err(EnumError::malformed(DISABLED, format!("unknown value {n}"))),
    }
}

/// The power source and whether a system battery exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Power {
    pub source: PowerSource,
    pub battery_present: bool,
}

/// Reads `/sys/class/power_supply`. A battery whose `scope` is `Device` belongs to a
/// peripheral (a mouse, a UPS's HID battery) and is ignored. The host is on battery only when
/// no mains or USB supply is online and a system battery is discharging; otherwise, including
/// a host with no supplies at all, it is on mains.
pub(super) fn power(fs: &dyn HostFs) -> Result<Power, EnumError> {
    const DIR: &str = "sys/class/power_supply";
    let mut adapter_online = false;
    let mut battery_present = false;
    let mut discharging: Option<u8> = None;
    for name in list_opt(fs, DIR)? {
        let base = format!("{DIR}/{name}");
        let ty_path = format!("{base}/type");
        let Some(ty) = read_opt(fs, &ty_path)? else {
            continue;
        };
        match ty.trim() {
            "Battery" => {
                if read_opt(fs, &format!("{base}/scope"))?.is_some_and(|s| s.trim() == "Device") {
                    continue;
                }
                battery_present = true;
                let st_path = format!("{base}/status");
                if read_opt(fs, &st_path)?.is_some_and(|s| s.trim() == "Discharging") {
                    let cap_path = format!("{base}/capacity");
                    let pct = at(&cap_path, parse::number::<u8>(&read(fs, &cap_path)?))?.min(100);
                    discharging = Some(discharging.map_or(pct, |p| p.min(pct)));
                }
            }
            _ => {
                let on_path = format!("{base}/online");
                if let Some(t) = read_opt(fs, &on_path)?
                    && at(&on_path, parse::number::<u8>(&t))? != 0
                {
                    adapter_online = true;
                }
            }
        }
    }
    let source = match discharging {
        Some(percent) if !adapter_online => PowerSource::Battery { percent },
        _ => PowerSource::Mains,
    };
    Ok(Power {
        source,
        battery_present,
    })
}
