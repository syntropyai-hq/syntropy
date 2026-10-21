// SPDX-License-Identifier: MIT OR Apache-2.0

//! Failure and edge cases of Linux E1 enumeration: missing files, malformed numbers, cgroup
//! `max` and v1 limits, no NUMA directory, hybrid Arm cores, batteries, the io_uring sysctl,
//! and the live host (PLC-E1-T02 criterion 2).

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use syntropy_core::ids::NodeId;
use syntropy_core::resource::caps::PowerSource;
use syntropy_core::units::Bytes;
use syntropy_probe::linux::{CgroupVersion, ContainerKind, IoUringSysctl, IsaSource, enumerate};
use syntropy_probe::{EnumError, HostFs};

const NODE: NodeId = NodeId([7; 32]);
const GIB: u64 = 1024 * 1024 * 1024;

/// A fixture tree loaded into memory, which a test then edits.
struct MemFs(BTreeMap<String, String>);

impl MemFs {
    fn fixture(scenario: &str) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/linux")
            .join(scenario);
        let mut files = BTreeMap::new();
        load(&root, &root, &mut files);
        Self(files)
    }

    fn set(&mut self, path: &str, text: &str) -> &mut Self {
        self.0.insert(path.into(), text.into());
        self
    }

    /// Removes a file, or every file under a directory.
    fn remove(&mut self, path: &str) -> &mut Self {
        let dir = format!("{path}/");
        self.0.retain(|k, _| k != path && !k.starts_with(&dir));
        self
    }

    fn run(&self) -> Result<syntropy_probe::linux::LinuxHost, EnumError> {
        enumerate(self, IsaSource::CpuInfo, NODE)
    }
}

fn load(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            load(root, &p, out);
        } else {
            let rel = p
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.insert(rel, std::fs::read_to_string(&p).unwrap());
        }
    }
}

impl HostFs for MemFs {
    fn read(&self, path: &str) -> io::Result<String> {
        self.0
            .get(path)
            .cloned()
            .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))
    }

    fn exists(&self, path: &str) -> bool {
        let dir = format!("{path}/");
        self.0.keys().any(|k| k == path || k.starts_with(&dir))
    }

    fn list(&self, path: &str) -> io::Result<Vec<String>> {
        let dir = format!("{path}/");
        let mut names: Vec<String> = self
            .0
            .keys()
            .filter_map(|k| k.strip_prefix(&dir))
            .map(|rest| rest.split('/').next().unwrap_or(rest).to_owned())
            .collect();
        if names.is_empty() {
            return Err(io::ErrorKind::NotFound.into());
        }
        names.dedup();
        Ok(names)
    }
}

fn malformed_at(r: Result<impl std::fmt::Debug, EnumError>, path: &str) {
    match r {
        Err(EnumError::Malformed { path: p, .. }) => assert_eq!(p, path),
        other => panic!("expected Malformed at {path}, got {other:?}"),
    }
}

// Missing and malformed files.

#[test]
fn missing_meminfo_is_an_error() {
    let r = MemFs::fixture("desktop").remove("proc/meminfo").run();
    assert_eq!(
        r.unwrap_err(),
        EnumError::Missing {
            path: "proc/meminfo".into()
        }
    );
}

#[test]
fn missing_mem_available_is_malformed_not_estimated() {
    let mut fs = MemFs::fixture("desktop");
    fs.set("proc/meminfo", "MemTotal: 1000 kB\nMemFree: 500 kB\n");
    malformed_at(fs.run(), "proc/meminfo");
}

#[test]
fn malformed_numbers_are_errors() {
    let mut fs = MemFs::fixture("desktop");
    fs.set(
        "proc/meminfo",
        "MemTotal: 12x4 kB\nMemFree: 1 kB\nMemAvailable: 1 kB\n",
    );
    malformed_at(fs.run(), "proc/meminfo");

    let mut fs = MemFs::fixture("desktop");
    fs.set("sys/devices/system/cpu/online", "0-");
    malformed_at(fs.run(), "sys/devices/system/cpu/online");

    let mut fs = MemFs::fixture("desktop");
    fs.set("sys/devices/system/cpu/cpu0/cache/index3/size", "lots");
    malformed_at(fs.run(), "sys/devices/system/cpu/cpu0/cache/index3/size");

    let mut fs = MemFs::fixture("docker");
    fs.set("sys/fs/cgroup/memory.max", "8G");
    malformed_at(fs.run(), "sys/fs/cgroup/memory.max");

    let mut fs = MemFs::fixture("desktop");
    fs.set("proc/self/limits", "Max locked memory  a lot  bytes\n");
    malformed_at(fs.run(), "proc/self/limits");
}

#[test]
fn missing_procfs_limit_files_are_errors_not_unlimited() {
    let r = MemFs::fixture("desktop").remove("proc/self/limits").run();
    assert_eq!(
        r.unwrap_err(),
        EnumError::Missing {
            path: "proc/self/limits".into()
        }
    );
}

#[test]
fn x86_below_v2_is_refused() {
    let mut fs = MemFs::fixture("desktop");
    let info = fs.0["proc/cpuinfo"].replace(" popcnt", "");
    fs.set("proc/cpuinfo", &info);
    assert_eq!(
        fs.run().unwrap_err(),
        EnumError::BelowBaseline {
            missing: vec!["popcnt"]
        }
    );
}

#[test]
fn a_flag_missing_on_one_cpu_is_not_claimed() {
    // ISA flags are the intersection over every processor entry.
    let mut fs = MemFs::fixture("dual-socket");
    let info = fs.0["proc/cpuinfo"].clone();
    let (head, tail) = info.split_at(info.rfind("processor\t: 31").unwrap());
    let tail = tail.replace(" avx512_fp16", "");
    fs.set("proc/cpuinfo", &format!("{head}{tail}"));
    let host = fs.run().unwrap();
    let topo = host.cpus[0].ext.cpu.as_ref().unwrap();
    assert!(
        !topo
            .isa
            .contains(syntropy_core::resource::IsaFlags::AVX512FP16)
    );
    assert!(
        topo.isa
            .contains(syntropy_core::resource::IsaFlags::AVX512BF16)
    );
}

// cgroups.

#[test]
fn cgroup_max_at_every_level_is_no_limit() {
    let host = MemFs::fixture("desktop").run().unwrap();
    assert_eq!(host.host_mem[0].mem.cgroup_limit, None);
}

#[test]
fn cgroup_parent_limit_binds_the_child() {
    let mut fs = MemFs::fixture("desktop");
    let slice = "sys/fs/cgroup/user.slice/user-1000.slice";
    let leaf =
        "sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/syntropy.scope";
    fs.set(&format!("{slice}/memory.max"), &(16 * GIB).to_string())
        .set(&format!("{slice}/memory.current"), &(10 * GIB).to_string())
        .set(&format!("{leaf}/memory.max"), &(12 * GIB).to_string())
        .set(&format!("{leaf}/memory.current"), &GIB.to_string());
    let host = fs.run().unwrap();
    // min(16 − 10, 12 − 1) GiB.
    assert_eq!(host.host_mem[0].mem.cgroup_limit, Some(Bytes(6 * GIB)));
}

#[test]
fn cgroup_usage_above_limit_leaves_zero() {
    let mut fs = MemFs::fixture("docker");
    fs.set("sys/fs/cgroup/memory.current", &(9 * GIB).to_string());
    let host = fs.run().unwrap();
    assert_eq!(host.host_mem[0].mem.cgroup_limit, Some(Bytes(0)));
}

#[test]
fn cgroup_limit_without_usage_is_an_error() {
    let r = MemFs::fixture("docker")
        .remove("sys/fs/cgroup/memory.current")
        .run();
    assert_eq!(
        r.unwrap_err(),
        EnumError::Missing {
            path: "sys/fs/cgroup/memory.current".into()
        }
    );
}

#[test]
fn cgroup_v1_limit_and_unlimited() {
    let mut fs = MemFs::fixture("docker");
    fs.remove("sys/fs/cgroup")
        .set(
            "proc/self/cgroup",
            "12:memory:/docker/0123abcd\n11:cpu,cpuacct:/docker/0123abcd\n0::/\n",
        )
        .set("sys/fs/cgroup/memory/memory.limit_in_bytes", "4294967296\n")
        .set("sys/fs/cgroup/memory/memory.usage_in_bytes", "1073741824\n");
    // /docker/0123abcd does not exist under the mount: the container sees its own cgroup at
    // the mount root.
    let host = fs.run().unwrap();
    assert_eq!(host.env.cgroup, CgroupVersion::V1);
    assert_eq!(host.host_mem[0].mem.cgroup_limit, Some(Bytes(3 * GIB)));

    fs.set(
        "sys/fs/cgroup/memory/memory.limit_in_bytes",
        "9223372036854771712\n",
    );
    assert_eq!(fs.run().unwrap().host_mem[0].mem.cgroup_limit, None);
}

#[test]
fn no_cgroup_file_means_no_limit() {
    let host = MemFs::fixture("docker")
        .remove("proc/self/cgroup")
        .run()
        .unwrap();
    assert_eq!(host.env.cgroup, CgroupVersion::None);
    assert_eq!(host.host_mem[0].mem.cgroup_limit, None);
}

// NUMA and topology.

#[test]
fn no_numa_directory_is_one_domain() {
    let host = MemFs::fixture("dual-socket")
        .remove("sys/devices/system/node")
        .run()
        .unwrap();
    assert_eq!(host.cpus.len(), 1);
    let topo = host.cpus[0].ext.cpu.as_ref().unwrap();
    assert_eq!((topo.physical_cores, topo.logical_cpus), (16, 32));
    // Both sockets' L3s.
    assert_eq!(topo.llc, Bytes(2 * 23_040 * 1024));
    assert_eq!(host.host_mem.len(), 1);
    // MemFree from /proc/meminfo.
    assert_eq!(
        host.host_mem[0].mem.numa_free,
        Some(Bytes(241_002_316 * 1024))
    );
}

#[test]
fn memoryless_node_gets_a_cpu_device_but_no_host_tiers() {
    let mut fs = MemFs::fixture("dual-socket");
    fs.set(
        "sys/devices/system/node/node1/meminfo",
        "Node 1 MemTotal: 0 kB\nNode 1 MemFree: 0 kB\n",
    );
    let host = fs.run().unwrap();
    assert_eq!(host.cpus.len(), 2);
    assert!(host.cpus[1].local_tiers.is_empty());
    assert_eq!(host.host_mem.len(), 1);
    assert_eq!(host.candidates.domains.len(), 1);
}

#[test]
fn offline_cpu_in_a_node_cpulist_is_not_counted() {
    // CPU 31 (socket 1, the sibling of CPU 15) is offline but still listed in node1's cpulist.
    let mut fs = MemFs::fixture("dual-socket");
    fs.set("sys/devices/system/cpu/online", "0-30");
    let host = fs.run().unwrap();
    let t0 = host.cpus[0].ext.cpu.clone().unwrap();
    let t1 = host.cpus[1].ext.cpu.clone().unwrap();
    assert_eq!((t0.logical_cpus, t0.physical_cores), (16, 8));
    assert_eq!((t1.logical_cpus, t1.physical_cores), (15, 8));
}

#[test]
fn thread_siblings_list_is_the_fallback() {
    let mut fs = MemFs::fixture("desktop");
    for c in 0..16 {
        let t = format!("sys/devices/system/cpu/cpu{c}/topology");
        let v = fs.0[&format!("{t}/core_cpus_list")].clone();
        fs.remove(&format!("{t}/core_cpus_list"))
            .set(&format!("{t}/thread_siblings_list"), &v);
    }
    let topo = fs.run().unwrap().cpus[0].ext.cpu.clone().unwrap();
    assert_eq!((topo.physical_cores, topo.perf_cores), (10, 6));

    fs.remove("sys/devices/system/cpu/cpu3/topology");
    assert_eq!(
        fs.run().unwrap_err(),
        EnumError::Missing {
            path: "sys/devices/system/cpu/cpu3/topology/thread_siblings_list".into()
        }
    );
}

#[test]
fn no_cache_directory_is_llc_zero() {
    let mut fs = MemFs::fixture("rpi5");
    for c in 0..4 {
        fs.remove(&format!("sys/devices/system/cpu/cpu{c}/cache"));
    }
    let topo = fs.run().unwrap().cpus[0].ext.cpu.clone().unwrap();
    assert_eq!(topo.llc, Bytes(0));
}

#[test]
fn partial_cache_entry_is_skipped() {
    // Without index3's size, the L3 entries are skipped and the L2s are the last level.
    let mut fs = MemFs::fixture("rpi5");
    for c in 0..4 {
        fs.remove(&format!("sys/devices/system/cpu/cpu{c}/cache/index3/size"));
    }
    let topo = fs.run().unwrap().cpus[0].ext.cpu.clone().unwrap();
    assert_eq!(topo.llc, Bytes(4 * 512 * 1024));
}

#[test]
fn intel_hybrid_without_the_atom_pmu_is_homogeneous() {
    let topo = MemFs::fixture("desktop")
        .remove("sys/devices/cpu_atom")
        .run()
        .unwrap()
        .cpus[0]
        .ext
        .cpu
        .clone()
        .unwrap();
    assert_eq!((topo.perf_cores, topo.efficiency_cores), (10, 0));
}

#[test]
fn arm_big_little_from_cpu_capacity() {
    let mut fs = MemFs::fixture("rpi5");
    fs.set("sys/devices/system/cpu/cpu2/cpu_capacity", "446")
        .set("sys/devices/system/cpu/cpu3/cpu_capacity", "446");
    let topo = fs.run().unwrap().cpus[0].ext.cpu.clone().unwrap();
    assert_eq!((topo.perf_cores, topo.efficiency_cores), (2, 2));

    // Three clusters: only the lowest capacity is efficiency.
    fs.set("sys/devices/system/cpu/cpu1/cpu_capacity", "800");
    let topo = fs.run().unwrap().cpus[0].ext.cpu.clone().unwrap();
    assert_eq!((topo.perf_cores, topo.efficiency_cores), (2, 2));

    // A capacity missing on one CPU: no classification.
    fs.remove("sys/devices/system/cpu/cpu0/cpu_capacity");
    let topo = fs.run().unwrap().cpus[0].ext.cpu.clone().unwrap();
    assert_eq!((topo.perf_cores, topo.efficiency_cores), (4, 0));
}

#[test]
fn unknown_arm_part_is_named_by_id() {
    let mut fs = MemFs::fixture("rpi5");
    let info = fs.0["proc/cpuinfo"].replace("0xd0b", "0xfff");
    fs.set("proc/cpuinfo", &info);
    assert_eq!(fs.run().unwrap().cpus[0].name, "AArch64 0x41:0xfff");
}

// Power.

fn laptop(fs: &mut MemFs, ac_online: &str, status: &str) {
    fs.set("sys/class/power_supply/AC/type", "Mains")
        .set("sys/class/power_supply/AC/online", ac_online)
        .set("sys/class/power_supply/BAT0/type", "Battery")
        .set("sys/class/power_supply/BAT0/scope", "System")
        .set("sys/class/power_supply/BAT0/status", status)
        .set("sys/class/power_supply/BAT0/capacity", "63");
}

#[test]
fn laptop_on_battery() {
    let mut fs = MemFs::fixture("desktop");
    laptop(&mut fs, "0", "Discharging");
    let host = fs.run().unwrap();
    assert_eq!(host.node.power, PowerSource::Battery { percent: 63 });
    assert!(host.node.battery_present);
}

#[test]
fn laptop_on_mains() {
    let mut fs = MemFs::fixture("desktop");
    laptop(&mut fs, "1", "Charging");
    let host = fs.run().unwrap();
    assert_eq!(host.node.power, PowerSource::Mains);
    assert!(host.node.battery_present);
}

#[test]
fn two_discharging_batteries_report_the_lower_charge() {
    let mut fs = MemFs::fixture("desktop");
    laptop(&mut fs, "0", "Discharging");
    fs.set("sys/class/power_supply/BAT1/type", "Battery")
        .set("sys/class/power_supply/BAT1/status", "Discharging")
        .set("sys/class/power_supply/BAT1/capacity", "41");
    assert_eq!(
        fs.run().unwrap().node.power,
        PowerSource::Battery { percent: 41 }
    );
}

#[test]
fn mains_online_wins_over_a_discharging_battery() {
    // A charger too weak for the load: the adapter is online while the battery drains.
    let mut fs = MemFs::fixture("desktop");
    laptop(&mut fs, "1", "Discharging");
    let host = fs.run().unwrap();
    assert_eq!(host.node.power, PowerSource::Mains);
    assert!(host.node.battery_present);
}

#[test]
fn malformed_battery_capacity_is_an_error() {
    let mut fs = MemFs::fixture("desktop");
    laptop(&mut fs, "0", "Discharging");
    fs.set("sys/class/power_supply/BAT0/capacity", "sixty");
    malformed_at(fs.run(), "sys/class/power_supply/BAT0/capacity");
}

// Environment.

#[test]
fn io_uring_sysctl_values() {
    let mut fs = MemFs::fixture("desktop");
    fs.set("proc/sys/kernel/io_uring_disabled", "2");
    assert_eq!(fs.run().unwrap().env.io_uring, IoUringSysctl::Disabled);

    fs.set("proc/sys/kernel/io_uring_disabled", "1")
        .set("proc/sys/kernel/io_uring_group", "27");
    assert_eq!(
        fs.run().unwrap().env.io_uring,
        IoUringSysctl::GroupOnly {
            gid: Some(27),
            member: true
        }
    );
    fs.set("proc/sys/kernel/io_uring_group", "990");
    assert_eq!(
        fs.run().unwrap().env.io_uring,
        IoUringSysctl::GroupOnly {
            gid: Some(990),
            member: false
        }
    );
    fs.set("proc/sys/kernel/io_uring_group", "-1");
    assert_eq!(
        fs.run().unwrap().env.io_uring,
        IoUringSysctl::GroupOnly {
            gid: None,
            member: false
        }
    );
    fs.set("proc/sys/kernel/io_uring_disabled", "3");
    malformed_at(fs.run(), "proc/sys/kernel/io_uring_disabled");
}

#[test]
fn containers_from_pid1_cgroup_and_podman_marker() {
    let mut fs = MemFs::fixture("desktop");
    fs.set("proc/1/cgroup", "12:memory:/docker/0123abcd\n");
    assert_eq!(fs.run().unwrap().env.container, Some(ContainerKind::Docker));
    fs.set("proc/1/cgroup", "0::/kubepods/besteffort/pod1\n");
    assert_eq!(fs.run().unwrap().env.container, Some(ContainerKind::Other));
    fs.set("run/.containerenv", "");
    assert_eq!(fs.run().unwrap().env.container, Some(ContainerKind::Podman));
}

#[test]
fn thp_never_is_no_large_pages() {
    let mut fs = MemFs::fixture("desktop");
    fs.set(
        "sys/kernel/mm/transparent_hugepage/enabled",
        "always madvise [never]",
    );
    assert_eq!(fs.run().unwrap().host_mem[0].mem.large_pages, None);
}

// Criterion 2: the live host, as the unprivileged user running the tests.

#[cfg(target_os = "linux")]
#[test]
fn live_host_enumerates_without_privilege() {
    // The criterion is about an unprivileged user, so the test refuses to pass as root: a
    // container job runs as root by default and would give false evidence.
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let field = |k: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(k))
            .unwrap_or_else(|| panic!("no {k} line in /proc/self/status"))
            .trim()
            .to_owned()
    };
    let uid = field("Uid:");
    // Real, effective, saved, filesystem.
    let euid: u32 = uid
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("Uid line {uid:?} has no effective UID"));
    let cap_eff = field("CapEff:");
    eprintln!("live enumeration as euid {euid}, CapEff {cap_eff}");
    assert_ne!(
        euid, 0,
        "this test must run as an unprivileged user; it is running as root"
    );

    let host = syntropy_probe::linux::enumerate_live(NODE)
        .unwrap_or_else(|e| panic!("live enumeration failed: {e}"));
    assert!(!host.cpus.is_empty());
    assert!(!host.host_mem.is_empty());
    for d in &host.cpus {
        let t = d.ext.cpu.as_ref().unwrap();
        assert!(t.logical_cpus >= t.physical_cores && t.physical_cores > 0);
    }
    for m in &host.host_mem {
        assert!(m.mem.total.0 > 0 && m.mem.available.0 <= m.mem.total.0);
    }
    eprintln!("{host:#?}");
}

#[cfg(not(target_os = "linux"))]
#[test]
fn live_host_is_unsupported_off_linux() {
    assert_eq!(
        syntropy_probe::linux::enumerate_live(NODE).unwrap_err(),
        EnumError::UnsupportedOs
    );
}
