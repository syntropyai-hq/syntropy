// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLC-E1-T02 acceptance criterion 1: the recorded sysfs/procfs fixtures for a desktop, a
//! dual-socket server, WSL2, Docker and a Raspberry Pi 5 produce the expected `NodeInfo`
//! facts, CPU `DeviceInfo` and `HostMemInfo`. Every expected value is written out in full.
//! The fixtures and where their values come from are in `tests/fixtures/linux/*/README.md`.

use smallvec::smallvec;
use syntropy_core::Blake3;
use syntropy_core::ids::{DeviceId, NodeId, TierId};
use syntropy_core::resource::caps::PowerSource;
use syntropy_core::resource::ordinal::{DeviceCandidate, Enumerated, HostDomain};
use syntropy_core::resource::{
    CpuArch, CpuIsa, CpuSet, CpuTopology, DeviceClass, DeviceFeatures, DeviceInfo, DeviceInfoExt,
    DeviceKind, HostMemInfo, IsaFlags,
};
use syntropy_core::units::Bytes;
use syntropy_probe::DirFs;
use syntropy_probe::linux::{
    CgroupVersion, ContainerKind, HostEnv, IoUringSysctl, IsaSource, LinuxHost, NodeFacts,
    NumaHostMem, RawCall, SysCheck, Syscalls, enumerate,
};

const NODE: NodeId = NodeId([7; 32]);
const KIB: u64 = 1024;
const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * 1024 * 1024;

/// System calls cannot be recorded with a fixture tree. Every fixture host grants both: only
/// the dual-socket server reports AMX, so it is the only one that requests the permission.
struct FixtureCalls;

impl Syscalls for FixtureCalls {
    fn request_xtiledata_permission(&self) -> RawCall {
        RawCall::Ok
    }

    fn io_uring_setup(&self) -> RawCall {
        RawCall::Ok
    }
}

fn run(scenario: &str) -> LinuxHost {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/linux/");
    enumerate(
        &DirFs::new(format!("{root}{scenario}")),
        IsaSource::CpuInfo,
        &FixtureCalls,
        NODE,
    )
    .unwrap_or_else(|e| panic!("{scenario}: {e}"))
}

fn dev(ordinal: u16) -> DeviceId {
    DeviceId {
        node: NODE,
        ordinal,
    }
}

fn tier(ordinal: u16) -> TierId {
    TierId {
        node: NODE,
        ordinal,
    }
}

fn cpu_candidate(name: &str, numa: u16) -> DeviceCandidate {
    DeviceCandidate {
        kind: DeviceKind::Cpu,
        pci: None,
        numa: Some(numa),
        uuid: None,
        name: name.into(),
        unified: false,
    }
}

#[test]
fn desktop_hybrid() {
    const NAME: &str = "12th Gen Intel(R) Core(TM) i5-12600K";
    let host = run("desktop");
    assert_eq!(
        host.node,
        NodeFacts {
            arch: CpuArch::X86_64,
            power: PowerSource::Mains,
            // The mouse battery has scope Device.
            battery_present: false,
            kernel_release: "6.8.0-45-generic".into(),
        }
    );
    assert_eq!(
        host.cpus,
        vec![DeviceInfo {
            id: dev(0),
            kind: DeviceKind::Cpu,
            name: NAME.into(),
            class: DeviceClass::Cpu { isa: CpuIsa::X86V3 },
            backends: smallvec![],
            features: DeviceFeatures::AVX2 | DeviceFeatures::AVX_VNNI,
            local_tiers: smallvec![tier(0), tier(1)],
            numa: Some(0),
            fingerprint: Blake3::default(),
            ext: DeviceInfoExt {
                cpu: Some(CpuTopology {
                    numa: 0,
                    physical_cores: 10,
                    logical_cpus: 16,
                    perf_cores: 6,
                    efficiency_cores: 4,
                    llc: Bytes(20 * MIB),
                    ccds: 0,
                    isa: IsaFlags::AVX2,
                    amx_permitted: false,
                    reserved_cpus: CpuSet::default(),
                }),
                ..DeviceInfoExt::default()
            },
        }]
    );
    assert_eq!(
        host.host_mem,
        vec![NumaHostMem {
            numa: 0,
            node_total: Bytes(32_562_044 * KIB),
            mem: HostMemInfo {
                total: Bytes(32_562_044 * KIB),
                available: Bytes(26_120_588 * KIB),
                numa_free: Some(Bytes(18_304_412 * KIB)),
                cgroup_limit: None,
                memlock_limit: Some(Bytes(8 * MIB)),
                windows_lockable: None,
                wsl2: false,
                macos_wired_limit: None,
                large_pages: Some(Bytes(2 * MIB)),
                lock: None,
            },
        }]
    );
    assert_eq!(
        host.env,
        HostEnv {
            wsl2: false,
            container: None,
            io_uring: IoUringSysctl::Enabled,
            seccomp_filtered: false,
            io_uring_setup: SysCheck::Granted,
            amx_permission: None,
            cgroup: CgroupVersion::V2,
            cap_ipc_lock: false,
        }
    );
    assert_eq!(
        host.candidates,
        Enumerated {
            devices: vec![cpu_candidate(NAME, 0)],
            domains: vec![HostDomain {
                numa: 0,
                pinned: true
            }],
            volumes: vec![],
        }
    );
}

#[test]
fn dual_socket_two_numa_domains() {
    const NAME: &str = "Intel(R) Xeon(R) Gold 5415+";
    let host = run("dual-socket");
    assert_eq!(
        host.node,
        NodeFacts {
            arch: CpuArch::X86_64,
            power: PowerSource::Mains,
            battery_present: false,
            kernel_release: "6.8.0-45-generic".into(),
        }
    );
    let features = DeviceFeatures::AVX2
        | DeviceFeatures::AVX512
        | DeviceFeatures::AVX512_VNNI
        | DeviceFeatures::AVX512_BF16
        | DeviceFeatures::AVX512_FP16
        | DeviceFeatures::AVX_VNNI;
    let isa = IsaFlags::AVX2
        | IsaFlags::AVX512F
        | IsaFlags::AVX512BW
        | IsaFlags::AVX512VL
        | IsaFlags::AVX512VNNI
        | IsaFlags::AVX512BF16
        | IsaFlags::AVX512FP16
        | IsaFlags::AMX_BF16
        | IsaFlags::AMX_INT8;
    let socket = |numa: u16, tiers: [u16; 2]| DeviceInfo {
        id: dev(numa),
        kind: DeviceKind::Cpu,
        name: NAME.into(),
        class: DeviceClass::Cpu { isa: CpuIsa::X86V4 },
        backends: smallvec![],
        features,
        local_tiers: smallvec![tier(tiers[0]), tier(tiers[1])],
        numa: Some(numa),
        fingerprint: Blake3::default(),
        ext: DeviceInfoExt {
            cpu: Some(CpuTopology {
                numa,
                physical_cores: 8,
                logical_cpus: 16,
                perf_cores: 8,
                efficiency_cores: 0,
                llc: Bytes(23_040 * KIB),
                ccds: 0,
                isa,
                amx_permitted: true,
                reserved_cpus: CpuSet::default(),
            }),
            ..DeviceInfoExt::default()
        },
    };
    assert_eq!(host.cpus, vec![socket(0, [0, 1]), socket(1, [2, 3])]);
    let mem = |numa: u16, node_total: u64, numa_free: u64| NumaHostMem {
        numa,
        node_total: Bytes(node_total * KIB),
        mem: HostMemInfo {
            total: Bytes(263_856_048 * KIB),
            available: Bytes(252_874_220 * KIB),
            numa_free: Some(Bytes(numa_free * KIB)),
            cgroup_limit: None,
            // CAP_IPC_LOCK is held, so the 8 MiB rlimit does not apply.
            memlock_limit: None,
            windows_lockable: None,
            wsl2: false,
            macos_wired_limit: None,
            large_pages: Some(Bytes(2 * MIB)),
            lock: None,
        },
    };
    assert_eq!(
        host.host_mem,
        vec![
            mem(0, 131_854_356, 120_112_508),
            mem(1, 132_001_692, 120_889_808),
        ]
    );
    assert_eq!(
        host.env,
        HostEnv {
            wsl2: false,
            container: None,
            io_uring: IoUringSysctl::Enabled,
            seccomp_filtered: false,
            io_uring_setup: SysCheck::Granted,
            amx_permission: Some(SysCheck::Granted),
            cgroup: CgroupVersion::V2,
            cap_ipc_lock: true,
        }
    );
    assert_eq!(
        host.candidates,
        Enumerated {
            devices: vec![cpu_candidate(NAME, 0), cpu_candidate(NAME, 1)],
            domains: vec![
                HostDomain {
                    numa: 0,
                    pinned: true
                },
                HostDomain {
                    numa: 1,
                    pinned: true
                },
            ],
            volumes: vec![],
        }
    );
}

#[test]
fn wsl2() {
    let host = run("wsl2");
    assert_eq!(
        host.node,
        NodeFacts {
            arch: CpuArch::X86_64,
            power: PowerSource::Mains,
            battery_present: false,
            kernel_release: "5.15.153.1-microsoft-standard-WSL2".into(),
        }
    );
    assert_eq!(
        host.cpus,
        vec![DeviceInfo {
            id: dev(0),
            kind: DeviceKind::Cpu,
            name: "AMD Ryzen 7 7800X3D 8-Core Processor".into(),
            class: DeviceClass::Cpu { isa: CpuIsa::X86V4 },
            backends: smallvec![],
            features: DeviceFeatures::AVX2
                | DeviceFeatures::AVX512
                | DeviceFeatures::AVX512_VNNI
                | DeviceFeatures::AVX512_BF16,
            local_tiers: smallvec![tier(0), tier(1)],
            numa: Some(0),
            fingerprint: Blake3::default(),
            ext: DeviceInfoExt {
                cpu: Some(CpuTopology {
                    numa: 0,
                    physical_cores: 8,
                    logical_cpus: 16,
                    perf_cores: 8,
                    efficiency_cores: 0,
                    llc: Bytes(96 * MIB),
                    ccds: 0,
                    isa: IsaFlags::AVX2
                        | IsaFlags::AVX512F
                        | IsaFlags::AVX512BW
                        | IsaFlags::AVX512VL
                        | IsaFlags::AVX512VNNI
                        | IsaFlags::AVX512BF16,
                    amx_permitted: false,
                    reserved_cpus: CpuSet::default(),
                }),
                ..DeviceInfoExt::default()
            },
        }]
    );
    assert_eq!(
        host.host_mem,
        vec![NumaHostMem {
            numa: 0,
            node_total: Bytes(32_803_456 * KIB),
            mem: HostMemInfo {
                total: Bytes(32_803_456 * KIB),
                available: Bytes(31_207_884 * KIB),
                numa_free: Some(Bytes(30_112_708 * KIB)),
                cgroup_limit: None,
                memlock_limit: Some(Bytes(64 * KIB)),
                windows_lockable: None,
                wsl2: true,
                macos_wired_limit: None,
                large_pages: Some(Bytes(2 * MIB)),
                lock: None,
            },
        }]
    );
    assert_eq!(
        host.env,
        HostEnv {
            wsl2: true,
            container: None,
            io_uring: IoUringSysctl::Absent,
            seccomp_filtered: false,
            io_uring_setup: SysCheck::Granted,
            amx_permission: None,
            cgroup: CgroupVersion::V2,
            cap_ipc_lock: false,
        }
    );
}

#[test]
fn docker_cgroup_v2_limit() {
    let host = run("docker");
    assert_eq!(
        host.node,
        NodeFacts {
            arch: CpuArch::X86_64,
            power: PowerSource::Mains,
            battery_present: false,
            kernel_release: "6.8.0-45-generic".into(),
        }
    );
    assert_eq!(
        host.cpus,
        vec![DeviceInfo {
            id: dev(0),
            kind: DeviceKind::Cpu,
            name: "AMD Ryzen 9 5950X 16-Core Processor".into(),
            class: DeviceClass::Cpu { isa: CpuIsa::X86V3 },
            backends: smallvec![],
            features: DeviceFeatures::AVX2,
            local_tiers: smallvec![tier(0), tier(1)],
            numa: Some(0),
            fingerprint: Blake3::default(),
            ext: DeviceInfoExt {
                cpu: Some(CpuTopology {
                    numa: 0,
                    physical_cores: 16,
                    logical_cpus: 32,
                    perf_cores: 16,
                    efficiency_cores: 0,
                    // Two 32 MiB L3s, one per CCD.
                    llc: Bytes(64 * MIB),
                    ccds: 0,
                    isa: IsaFlags::AVX2,
                    amx_permitted: false,
                    reserved_cpus: CpuSet::default(),
                }),
                ..DeviceInfoExt::default()
            },
        }]
    );
    assert_eq!(
        host.host_mem,
        vec![NumaHostMem {
            numa: 0,
            node_total: Bytes(65_777_452 * KIB),
            mem: HostMemInfo {
                total: Bytes(65_777_452 * KIB),
                available: Bytes(58_993_172 * KIB),
                numa_free: Some(Bytes(40_221_396 * KIB)),
                // memory.max 8 GiB − memory.current 1 GiB.
                cgroup_limit: Some(Bytes(7 * GIB)),
                memlock_limit: Some(Bytes(8 * MIB)),
                windows_lockable: None,
                wsl2: false,
                macos_wired_limit: None,
                large_pages: Some(Bytes(2 * MIB)),
                lock: None,
            },
        }]
    );
    assert_eq!(
        host.env,
        HostEnv {
            wsl2: false,
            container: Some(ContainerKind::Docker),
            io_uring: IoUringSysctl::Enabled,
            seccomp_filtered: true,
            io_uring_setup: SysCheck::Granted,
            amx_permission: None,
            cgroup: CgroupVersion::V2,
            cap_ipc_lock: false,
        }
    );
}

#[test]
fn raspberry_pi_5() {
    let host = run("rpi5");
    assert_eq!(
        host.node,
        NodeFacts {
            arch: CpuArch::Aarch64,
            power: PowerSource::Mains,
            battery_present: false,
            kernel_release: "6.6.51+rpt-rpi-2712".into(),
        }
    );
    assert_eq!(
        host.cpus,
        vec![DeviceInfo {
            id: dev(0),
            kind: DeviceKind::Cpu,
            name: "ARM Cortex-A76".into(),
            class: DeviceClass::Cpu {
                isa: CpuIsa::Arm82Dot,
            },
            backends: smallvec![],
            features: DeviceFeatures::NEON_DOTPROD | DeviceFeatures::NEON_FP16,
            local_tiers: smallvec![tier(0), tier(1)],
            numa: Some(0),
            fingerprint: Blake3::default(),
            ext: DeviceInfoExt {
                cpu: Some(CpuTopology {
                    numa: 0,
                    physical_cores: 4,
                    logical_cpus: 4,
                    perf_cores: 4,
                    efficiency_cores: 0,
                    llc: Bytes(2 * MIB),
                    ccds: 0,
                    isa: IsaFlags::NEON_DOTPROD,
                    amx_permitted: false,
                    reserved_cpus: CpuSet::default(),
                }),
                ..DeviceInfoExt::default()
            },
        }]
    );
    assert_eq!(
        host.host_mem,
        vec![NumaHostMem {
            numa: 0,
            node_total: Bytes(8_245_168 * KIB),
            mem: HostMemInfo {
                total: Bytes(8_245_168 * KIB),
                available: Bytes(7_570_784 * KIB),
                // No NUMA directory: the single domain's free memory is MemFree.
                numa_free: Some(Bytes(6_922_480 * KIB)),
                // The memory controller is off.
                cgroup_limit: None,
                // RLIMIT_MEMLOCK is unlimited.
                memlock_limit: None,
                windows_lockable: None,
                wsl2: false,
                macos_wired_limit: None,
                large_pages: Some(Bytes(32 * MIB)),
                lock: None,
            },
        }]
    );
    assert_eq!(
        host.env,
        HostEnv {
            wsl2: false,
            container: None,
            io_uring: IoUringSysctl::Enabled,
            seccomp_filtered: false,
            io_uring_setup: SysCheck::Granted,
            amx_permission: None,
            cgroup: CgroupVersion::V2,
            cap_ipc_lock: false,
        }
    );
}
