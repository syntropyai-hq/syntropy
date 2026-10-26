// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared checks of the macOS and Windows E1 tests.

#![allow(dead_code)]

use std::io::Write as _;

use syntropy_core::resource::{CpuTopology, DeviceClass, DeviceInfo, HostMemInfo};
use syntropy_probe::host::{NodeFacts, NumaHostMem};

/// The fields of E1's records that hold "unknown": a zero count or size where the type
/// documents 0 as unknown, an `Option` that must be `Some` on these hosts (the NUMA domain,
/// the topology, `numa_free`, Windows' working-set ceiling), or an empty name or release.
/// PLC-E1-T03 criterion 1 wants none.
///
/// Every `HostMemInfo` and `CpuTopology` field is named, with why its `None` or default is a
/// value and not an unknown, so a new field fails to compile here until it is classified.
/// Not checked: `DeviceInfo::fingerprint`, all zero on every OS until #1524 item 2 gives it an
/// owner.
pub fn unknown_fields(
    node: &NodeFacts,
    cpus: &[DeviceInfo],
    mem: &[NumaHostMem],
    windows: bool,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut need = |ok: bool, what: String| {
        if !ok {
            out.push(what);
        }
    };
    let NodeFacts {
        // An enum with no unknown variant.
        arch: _,
        // An enum with no unknown variant; an unknown status is an enumeration error.
        power: _,
        battery_present: _,
        kernel_release,
    } = node;
    need(
        !kernel_release.is_empty(),
        "NodeFacts::kernel_release".into(),
    );
    need(!cpus.is_empty(), "no CPU DeviceInfo".into());
    for (i, d) in cpus.iter().enumerate() {
        need(!d.name.is_empty(), format!("cpus[{i}].name"));
        need(
            matches!(d.class, DeviceClass::Cpu { .. }),
            format!("cpus[{i}].class"),
        );
        need(d.numa.is_some(), format!("cpus[{i}].numa"));
        need(!d.local_tiers.is_empty(), format!("cpus[{i}].local_tiers"));
        let Some(t) = &d.ext.cpu else {
            need(false, format!("cpus[{i}].ext.cpu"));
            continue;
        };
        let CpuTopology {
            numa,
            physical_cores,
            logical_cpus,
            perf_cores,
            efficiency_cores,
            llc,
            ccds,
            // A set of flags; empty is a value.
            isa: _,
            // False until E5's CPU provider confirms AMX.
            amx_permitted: _,
            // Made by the scheduler, not E1.
            reserved_cpus: _,
        } = t;
        need(Some(*numa) == d.numa, format!("cpus[{i}] topology numa"));
        need(*physical_cores > 0, format!("cpus[{i}] physical_cores"));
        need(
            logical_cpus >= physical_cores,
            format!("cpus[{i}] logical_cpus"),
        );
        need(*perf_cores > 0, format!("cpus[{i}] perf_cores"));
        need(
            perf_cores + efficiency_cores == *physical_cores,
            format!("cpus[{i}] perf + efficiency cores"),
        );
        need(llc.0 > 0, format!("cpus[{i}] llc"));
        need(*ccds > 0, format!("cpus[{i}] ccds"));
    }
    need(!mem.is_empty(), "no HostMemInfo".into());
    for (i, m) in mem.iter().enumerate() {
        need(m.node_total.0 > 0, format!("host_mem[{i}].node_total"));
        let HostMemInfo {
            total,
            available,
            numa_free,
            // `None` = no job-object (Windows) or cgroup limit; read on both OSes.
            cgroup_limit: _,
            // `None` = unlimited (macOS `getrlimit`); Windows has no such limit.
            memlock_limit: _,
            windows_lockable,
            wsl2,
            // `None` = the OS's default cap (`iogpu.wired_limit_mb` read as 0).
            macos_wired_limit: _,
            // `None` = no large pages without privilege, on both OSes.
            large_pages: _,
            // Measured by step H5, not E1.
            lock: _,
        } = &m.mem;
        need(total.0 > 0, format!("host_mem[{i}].total"));
        need(
            available.0 > 0 && available <= total,
            format!("host_mem[{i}].available"),
        );
        need(numa_free.is_some(), format!("host_mem[{i}].numa_free"));
        need(!*wsl2, format!("host_mem[{i}].wsl2"));
        if windows {
            need(
                windows_lockable.is_some_and(|b| b.0 > 0),
                format!("host_mem[{i}].windows_lockable"),
            );
        } else {
            need(
                windows_lockable.is_none(),
                format!("host_mem[{i}].windows_lockable"),
            );
        }
    }
    out
}

/// Writes evidence straight to the process's stderr, past the test harness's capture, so a
/// passing CI job shows it in its log.
pub fn evidence(text: &str) {
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(text.as_bytes());
    let _ = err.flush();
}
