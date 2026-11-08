// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLC-E1-T03 on Windows:
//!
//! - criterion 2, fixture tests for API responses: the responses in
//!   `tests/fixtures/windows/` produce the expected `NodeFacts`, CPU `DeviceInfo` and
//!   `HostMemInfo`, every value written out; variants cover batteries and the failure cases;
//! - criterion 1, on a Windows runner the live host's records have no unknown field
//!   (`live_host_has_no_unknown_fields`, Windows only), printed to the job log with the
//!   responses they came from.

mod common;

use smallvec::smallvec;
use syntropy_core::Blake3;
use syntropy_core::budget::NodeMem;
use syntropy_core::ids::{DeviceId, NodeId, TierId};
use syntropy_core::resource::caps::PowerSource;
use syntropy_core::resource::ordinal::{DeviceCandidate, Enumerated, HostDomain};
use syntropy_core::resource::{
    CpuArch, CpuIsa, CpuSet, CpuTopology, DeviceClass, DeviceFeatures, DeviceInfo, DeviceInfoExt,
    DeviceKind, HostMemInfo, IsaFlags,
};
use syntropy_core::units::Bytes;
use syntropy_probe::EnumError;
use syntropy_probe::host::{NodeFacts, NumaHostMem};
use syntropy_probe::windows::{
    WinFixture, WinJob, WinPower, WinRecorder, WindowsEnv, WindowsHost, enumerate,
};

const NODE: NodeId = NodeId([7; 32]);
const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

fn fixture(name: &str) -> WinFixture {
    let path = format!(
        "{}/tests/fixtures/windows/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut f = WinFixture::parse(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
    if name == "windows-2025-runner" {
        // The capture predates the job-object read, so it records no `job` response. Supply
        // "in no job" here, in the test, rather than add a response the runner never gave.
        f.set_job(None);
    }
    f
}

fn run(name: &str) -> WindowsHost {
    enumerate(&fixture(name), NODE).unwrap_or_else(|e| panic!("{name}: {e}"))
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

/// `GetNumaAvailableMemoryNodeEx`'s figure as the domain's reclaimable memory.
fn node_mem(free: u64) -> Option<NodeMem> {
    Some(NodeMem {
        mem_free: Bytes(free),
        ..NodeMem::default()
    })
}

fn mem(total: u64, available: u64, free: u64, lockable: u64) -> HostMemInfo {
    HostMemInfo {
        total: Bytes(total),
        available: Bytes(available),
        numa_free: Some(Bytes(free)),
        cgroup_limit: None,
        memlock_limit: None,
        windows_lockable: Some(Bytes(lockable)),
        wsl2: false,
        macos_wired_limit: None,
        large_pages: None,
        lock: None,
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
            battery_present: false,
            kernel_release: "10.0.26100".into(),
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
                    ccds: 1,
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
            // One node: the whole host's `ullTotalPhys`.
            node_total: Bytes(34_190_000_128),
            mem: mem(34_190_000_128, 25_769_803_776, 25_769_803_776, 30 * GIB),
            node_mem: node_mem(25_769_803_776),
        }]
    );
    assert_eq!(host.env, WindowsEnv { build: 26100 });
    assert_eq!(
        host.candidates,
        Enumerated {
            devices: vec![DeviceCandidate {
                kind: DeviceKind::Cpu,
                pci: None,
                numa: Some(0),
                uuid: None,
                name: NAME.into(),
                unified: false,
            }],
            domains: vec![HostDomain {
                numa: 0,
                pinned: true
            }],
            volumes: vec![],
        }
    );
    let unknown = common::unknown_fields(&host.node, &host.cpus, &host.host_mem, true);
    assert!(unknown.is_empty(), "{unknown:?}");
}

#[test]
fn dual_socket_two_numa_nodes() {
    const NAME: &str = "AMD EPYC 9124 16-Core Processor";
    let host = run("dual-socket");
    let features = DeviceFeatures::AVX2
        | DeviceFeatures::AVX512
        | DeviceFeatures::AVX512_VNNI
        | DeviceFeatures::AVX512_BF16;
    let isa = IsaFlags::AVX2
        | IsaFlags::AVX512F
        | IsaFlags::AVX512BW
        | IsaFlags::AVX512VL
        | IsaFlags::AVX512VNNI
        | IsaFlags::AVX512BF16;
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
                physical_cores: 16,
                logical_cpus: 32,
                perf_cores: 16,
                efficiency_cores: 0,
                // Two 32 MiB L3 caches, one per core complex.
                llc: Bytes(64 * MIB),
                ccds: 2,
                isa,
                amx_permitted: false,
                reserved_cpus: CpuSet::default(),
            }),
            ..DeviceInfoExt::default()
        },
    };
    assert_eq!(host.cpus, vec![socket(0, [0, 1]), socket(1, [2, 3])]);
    let total = 256 * GIB - 3 * GIB;
    assert_eq!(
        host.host_mem,
        vec![
            NumaHostMem {
                numa: 0,
                // The SRAT's two ranges for proximity domain 0.
                node_total: Bytes(128 * GIB),
                mem: mem(total, 250 * GIB, 120 * GIB, 200 * GIB),
                node_mem: node_mem(120 * GIB),
            },
            NumaHostMem {
                numa: 1,
                node_total: Bytes(128 * GIB),
                mem: mem(total, 250 * GIB, 126 * GIB, 200 * GIB),
                node_mem: node_mem(126 * GIB),
            },
        ]
    );
    let unknown = common::unknown_fields(&host.node, &host.cpus, &host.host_mem, true);
    assert!(unknown.is_empty(), "{unknown:?}");
}

/// The responses captured on a GitHub-hosted `windows-2025` runner (see the fixture's header).
#[test]
fn windows_2025_runner_capture() {
    const NAME: &str = "AMD EPYC 7763 64-Core Processor";
    let host = run("windows-2025-runner");
    assert_eq!(
        host.node,
        NodeFacts {
            arch: CpuArch::X86_64,
            power: PowerSource::Mains,
            battery_present: false,
            kernel_release: "10.0.26100".into(),
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
            features: DeviceFeatures::AVX2,
            local_tiers: smallvec![tier(0), tier(1)],
            numa: Some(0),
            fingerprint: Blake3::default(),
            ext: DeviceInfoExt {
                cpu: Some(CpuTopology {
                    numa: 0,
                    physical_cores: 2,
                    logical_cpus: 4,
                    perf_cores: 2,
                    efficiency_cores: 0,
                    llc: Bytes(32 * MIB),
                    ccds: 1,
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
            node_total: Bytes(17_174_360_064),
            mem: mem(
                17_174_360_064,
                14_141_550_592,
                14_142_144_512,
                16_106_332_160
            ),
            node_mem: node_mem(14_142_144_512),
        }]
    );
    assert_eq!(host.env, WindowsEnv { build: 26100 });
    let unknown = common::unknown_fields(&host.node, &host.cpus, &host.host_mem, true);
    assert!(unknown.is_empty(), "{unknown:?}");
}

#[test]
fn laptop_power() {
    let mut f = fixture("desktop");
    f.set_power(WinPower {
        ac_line: 0,
        battery_flag: 0,
        battery_percent: 63,
    });
    let host = enumerate(&f, NODE).unwrap();
    assert_eq!(host.node.power, PowerSource::Battery { percent: 63 });
    assert!(host.node.battery_present);
    f.set_power(WinPower {
        ac_line: 1,
        battery_flag: 8,
        battery_percent: 100,
    });
    let host = enumerate(&f, NODE).unwrap();
    assert_eq!(host.node.power, PowerSource::Mains);
    assert!(host.node.battery_present);
}

#[test]
fn recorded_responses_round_trip() {
    for name in ["desktop", "dual-socket", "windows-2025-runner"] {
        let rec = WinRecorder::new(fixture(name));
        let first = enumerate(&rec, NODE).unwrap();
        let again = WinFixture::parse(&rec.fixture()).unwrap();
        assert_eq!(enumerate(&again, NODE).unwrap(), first, "{name}");
    }
}

/// `cgroup_limit` is the job object's headroom, and `None` only when there is no limit.
#[test]
fn job_object_limits() {
    let limit = |f: &WinFixture| enumerate(f, NODE).unwrap().host_mem[0].mem.cgroup_limit;
    // Not in a job (the fixture's `job none`).
    let mut f = fixture("desktop");
    assert_eq!(limit(&f), None);
    // In a job that sets no memory limit.
    f.set_job(Some(WinJob {
        peak_job: 3 * GIB,
        peak_process: 2 * GIB,
        ..WinJob::default()
    }));
    assert_eq!(limit(&f), None);
    // A job limit of 8 GiB with a 3 GiB peak, and a 4 GiB per-process limit with a 2 GiB peak.
    f.set_job(Some(WinJob {
        job_limit: Some(8 * GIB),
        process_limit: None,
        peak_job: 3 * GIB,
        peak_process: 2 * GIB,
    }));
    assert_eq!(limit(&f), Some(Bytes(5 * GIB)));
    f.set_job(Some(WinJob {
        job_limit: Some(8 * GIB),
        process_limit: Some(4 * GIB),
        peak_job: 3 * GIB,
        peak_process: 2 * GIB,
    }));
    assert_eq!(limit(&f), Some(Bytes(2 * GIB)));
    let rec = WinRecorder::new(f.clone());
    let first = enumerate(&rec, NODE).unwrap();
    assert!(
        rec.fixture()
            .contains("job job_limit=8589934592 process_limit=4294967296")
    );
    let again = WinFixture::parse(&rec.fixture()).unwrap();
    assert_eq!(enumerate(&again, NODE).unwrap(), first);
}

#[test]
fn failures() {
    // Windows on Arm is not an ADR-0020 platform.
    let mut f = fixture("desktop");
    f.set_cpu_arch("aarch64");
    assert_eq!(
        enumerate(&f, NODE).unwrap_err(),
        EnumError::UnsupportedArch {
            found: "aarch64".into()
        }
    );
    // An unknown power status is an error, not a value.
    let mut f = fixture("desktop");
    f.set_power(WinPower {
        ac_line: 255,
        battery_flag: 128,
        battery_percent: 255,
    });
    assert!(matches!(
        enumerate(&f, NODE),
        Err(EnumError::BadResponse { .. })
    ));
    // A multi-node host without an SRAT has no per-node total: an error, not a guess.
    let mut f = fixture("dual-socket");
    f.set_srat(None);
    assert!(matches!(
        enumerate(&f, NODE),
        Err(EnumError::BadResponse { .. })
    ));
    // Below x86-64-v2.
    let mut f = fixture("desktop");
    f.set_cpu_flags(["cx16", "popcnt", "sse4_1", "sse4_2", "ssse3"]);
    assert_eq!(
        enumerate(&f, NODE).unwrap_err(),
        EnumError::BelowBaseline {
            missing: vec!["lahf_lm"]
        }
    );
    // A response that is not in the fixture is an OS error.
    assert!(matches!(
        enumerate(&WinFixture::default(), NODE),
        Err(EnumError::Os { .. })
    ));
}

/// Criterion 1 on a Windows runner: the live records have no unknown field, and the records
/// and the responses they came from go to the job log.
#[cfg(windows)]
#[test]
fn live_host_has_no_unknown_fields() {
    let before = syntropy_io::os::windows_host::process_working_set().unwrap();
    let rec = WinRecorder::new(syntropy_probe::windows::LiveWindows);
    let host = enumerate(&rec, NODE).unwrap_or_else(|e| {
        common::evidence(&format!(
            "\n=== PLC-E1-T03 live Windows responses before the failure ===\n{}=== end ===\n",
            rec.fixture()
        ));
        panic!("live enumeration: {e}")
    });
    // The working-set probe put the original limits back.
    assert_eq!(
        syntropy_io::os::windows_host::process_working_set().unwrap(),
        before
    );
    common::evidence(&format!(
        "\n=== PLC-E1-T03 live Windows E1 records ===\n{host:#?}\n=== recorded responses (fixture format) ===\n{}=== end ===\n",
        rec.fixture()
    ));
    let unknown = common::unknown_fields(&host.node, &host.cpus, &host.host_mem, true);
    assert!(unknown.is_empty(), "unknown fields: {unknown:?}");
    // The recorded responses reproduce the live result.
    let again = WinFixture::parse(&rec.fixture()).unwrap();
    assert_eq!(enumerate(&again, NODE).unwrap(), host);
}
