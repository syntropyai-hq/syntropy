// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared checks of the macOS and Windows E1 tests.

#![allow(dead_code)]

use std::io::Write as _;

use syntropy_core::resource::{DeviceClass, DeviceInfo};
use syntropy_probe::host::{NodeFacts, NumaHostMem};

/// The fields of E1's records that hold "unknown": a zero count or size where the type
/// documents 0 as unknown, a missing NUMA domain or topology, an empty name or release, or
/// (Windows) no working-set ceiling. PLC-E1-T03 criterion 1 wants none.
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
    need(
        !node.kernel_release.is_empty(),
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
        match &d.ext.cpu {
            None => need(false, format!("cpus[{i}].ext.cpu")),
            Some(t) => {
                need(t.physical_cores > 0, format!("cpus[{i}] physical_cores"));
                need(
                    t.logical_cpus >= t.physical_cores,
                    format!("cpus[{i}] logical_cpus"),
                );
                need(t.perf_cores > 0, format!("cpus[{i}] perf_cores"));
                need(
                    t.perf_cores + t.efficiency_cores == t.physical_cores,
                    format!("cpus[{i}] perf + efficiency cores"),
                );
                need(t.llc.0 > 0, format!("cpus[{i}] llc"));
                need(t.ccds > 0, format!("cpus[{i}] ccds"));
            }
        }
    }
    need(!mem.is_empty(), "no HostMemInfo".into());
    for (i, m) in mem.iter().enumerate() {
        need(m.node_total.0 > 0, format!("host_mem[{i}].node_total"));
        need(m.mem.total.0 > 0, format!("host_mem[{i}].total"));
        need(
            m.mem.available.0 > 0 && m.mem.available <= m.mem.total,
            format!("host_mem[{i}].available"),
        );
        need(
            m.mem.numa_free.is_some(),
            format!("host_mem[{i}].numa_free"),
        );
        if windows {
            need(
                m.mem.windows_lockable.is_some_and(|b| b.0 > 0),
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
