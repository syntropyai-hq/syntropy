// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLC-E1-T03 on macOS:
//!
//! - criterion 2, fixture tests for captured API responses: `tests/fixtures/macos/m4-mini.txt`
//!   (recorded on an M4 Mac mini) produces the expected `NodeFacts`, CPU `DeviceInfo` and
//!   `HostMemInfo`, every value written out; variants of it cover batteries, a set wired
//!   limit and the failure cases;
//! - criterion 3, `iogpu.wired_limit_mb` is read and never written: a recording fake of the
//!   sysctl interface sees the read and no write;
//! - criterion 1, on a macOS runner the live host's records have no unknown field
//!   (`live_host_has_no_unknown_fields`, macOS only), printed to the job log.

mod common;

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
use syntropy_probe::EnumError;
use syntropy_probe::host::{NodeFacts, NumaHostMem};
use syntropy_probe::macos::{
    MacApi, MacEnv, MacFixture, MacRecorder, PowerSourceDesc, PowerSources, WIRED_LIMIT, enumerate,
};

const NODE: NodeId = NodeId([7; 32]);
const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const PAGE: u64 = 16_384;

fn fixture(name: &str) -> MacFixture {
    let path = format!(
        "{}/tests/fixtures/macos/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    MacFixture::parse(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn cpu() -> DeviceInfo {
    DeviceInfo {
        id: DeviceId {
            node: NODE,
            ordinal: 0,
        },
        kind: DeviceKind::Cpu,
        name: "Apple M4".into(),
        class: DeviceClass::Cpu {
            isa: CpuIsa::Arm92Sme2,
        },
        backends: smallvec![],
        features: DeviceFeatures::NEON_DOTPROD
            | DeviceFeatures::NEON_I8MM
            | DeviceFeatures::NEON_BF16
            | DeviceFeatures::NEON_FP16
            | DeviceFeatures::SME2,
        // One `HostPageable` tier: Apple has no `HostPinned` tier.
        local_tiers: smallvec![TierId {
            node: NODE,
            ordinal: 0
        }],
        numa: Some(0),
        fingerprint: Blake3::default(),
        ext: DeviceInfoExt {
            cpu: Some(CpuTopology {
                numa: 0,
                physical_cores: 10,
                logical_cpus: 10,
                perf_cores: 4,
                efficiency_cores: 6,
                // 16 MiB L2 on the P-cluster plus 4 MiB on the E-cluster.
                llc: Bytes(20 * MIB),
                ccds: 2,
                isa: IsaFlags::NEON_DOTPROD | IsaFlags::NEON_I8MM | IsaFlags::SME2,
                amx_permitted: false,
                reserved_cpus: CpuSet::default(),
            }),
            ..DeviceInfoExt::default()
        },
    }
}

#[test]
fn m4_mini() {
    let host = enumerate(&fixture("m4-mini"), NODE).unwrap();
    assert_eq!(
        host.node,
        NodeFacts {
            arch: CpuArch::Aarch64,
            power: PowerSource::Mains,
            battery_present: false,
            kernel_release: "25.4.0".into(),
        }
    );
    assert_eq!(host.cpus, vec![cpu()]);
    assert_eq!(
        host.host_mem,
        vec![NumaHostMem {
            numa: 0,
            node_total: Bytes(16 * GIB),
            mem: HostMemInfo {
                total: Bytes(16 * GIB),
                // free + inactive + purgeable pages.
                available: Bytes((28_015 + 319_984 + 1_476) * PAGE),
                numa_free: Some(Bytes(28_015 * PAGE)),
                cgroup_limit: None,
                memlock_limit: None,
                windows_lockable: None,
                wsl2: false,
                // 0 is the default: the OS's own cap.
                macos_wired_limit: None,
                large_pages: None,
                lock: None,
            },
        }]
    );
    assert_eq!(
        host.env,
        MacEnv {
            os_version: "26.4.1".into()
        }
    );
    assert_eq!(
        host.candidates,
        Enumerated {
            devices: vec![DeviceCandidate {
                kind: DeviceKind::Cpu,
                pci: None,
                numa: Some(0),
                uuid: None,
                name: "Apple M4".into(),
                unified: false,
            }],
            domains: vec![HostDomain {
                numa: 0,
                pinned: false
            }],
            volumes: vec![],
        }
    );
    let unknown = common::unknown_fields(&host.node, &host.cpus, &host.host_mem, false);
    assert!(unknown.is_empty(), "{unknown:?}");
}

/// Criterion 3: the wired limit is read, and no call passes a new value.
#[test]
fn wired_limit_is_read_and_never_written() {
    let mut f = fixture("m4-mini");
    f.set_sysctl(WIRED_LIMIT, Some(12_288u32.to_ne_bytes().to_vec()));
    let rec = MacRecorder::new(f);
    let host = enumerate(&rec, NODE).unwrap();
    assert_eq!(
        host.host_mem[0].mem.macos_wired_limit,
        Some(Bytes(12 * GIB))
    );
    assert!(rec.sysctl_reads().iter().any(|n| n == WIRED_LIMIT));
    assert_eq!(rec.sysctl_writes(), Vec::<String>::new());

    // The default, the name absent (an older OS), and the default fixture: still no write.
    for value in [Some(0u32.to_ne_bytes().to_vec()), None] {
        let mut f = fixture("m4-mini");
        f.set_sysctl(WIRED_LIMIT, value);
        let rec = MacRecorder::new(f);
        let host = enumerate(&rec, NODE).unwrap();
        assert_eq!(host.host_mem[0].mem.macos_wired_limit, None);
        assert!(rec.sysctl_writes().is_empty());
    }
}

/// The recording fake does see a write, so the test above would catch one.
#[test]
fn the_recorder_sees_writes() {
    let rec = MacRecorder::new(fixture("m4-mini"));
    let r = rec.sysctl(WIRED_LIMIT, Some(&8192u32.to_ne_bytes()));
    assert!(r.is_err(), "the fixture refuses writes");
    assert_eq!(rec.sysctl_writes(), vec![WIRED_LIMIT.to_owned()]);
}

#[test]
fn laptop_on_battery_and_on_mains() {
    let battery = |state: &str, providing: &str| PowerSources {
        providing: Some(providing.into()),
        sources: vec![PowerSourceDesc {
            kind: Some("InternalBattery".into()),
            state: Some(state.into()),
            current_capacity: Some(57),
            max_capacity: Some(100),
            present: Some(true),
        }],
    };
    let mut f = fixture("m4-mini");
    f.set_power(battery("Battery Power", "Battery Power"));
    let host = enumerate(&f, NODE).unwrap();
    assert_eq!(host.node.power, PowerSource::Battery { percent: 57 });
    assert!(host.node.battery_present);

    f.set_power(battery("AC Power", "AC Power"));
    let host = enumerate(&f, NODE).unwrap();
    assert_eq!(host.node.power, PowerSource::Mains);
    assert!(host.node.battery_present);

    // A UPS is not a system battery.
    let mut ups = battery("AC Power", "AC Power");
    ups.sources[0].kind = Some("UPS".into());
    f.set_power(ups);
    assert!(!enumerate(&f, NODE).unwrap().node.battery_present);

    // On battery power with no battery listed is a bad response, not a guess.
    f.set_power(PowerSources {
        providing: Some("Battery Power".into()),
        sources: vec![],
    });
    assert!(matches!(
        enumerate(&f, NODE),
        Err(EnumError::BadResponse { .. })
    ));
}

#[test]
fn recorded_responses_round_trip() {
    let rec = MacRecorder::new(fixture("m4-mini"));
    let first = enumerate(&rec, NODE).unwrap();
    let again = MacFixture::parse(&rec.fixture()).unwrap();
    assert_eq!(enumerate(&again, NODE).unwrap(), first);
}

#[test]
fn failures() {
    let mut f = fixture("m4-mini");
    f.set_sysctl("hw.memsize", None);
    assert!(matches!(
        enumerate(&f, NODE),
        Err(EnumError::BadResponse { call, .. }) if call == "sysctl hw.memsize"
    ));

    let mut f = fixture("m4-mini");
    f.set_sysctl("hw.optional.arm64", None);
    assert!(matches!(
        enumerate(&f, NODE),
        Err(EnumError::UnsupportedArch { .. })
    ));

    // An integer sysctl of the wrong width.
    let mut f = fixture("m4-mini");
    f.set_sysctl("hw.physicalcpu", Some(vec![1, 2]));
    assert!(matches!(
        enumerate(&f, NODE),
        Err(EnumError::BadResponse { .. })
    ));

    // A one-level part: no efficiency cores.
    let mut f = fixture("m4-mini");
    f.set_sysctl("hw.nperflevels", Some(1u32.to_ne_bytes().to_vec()));
    f.set_sysctl(
        "hw.perflevel0.physicalcpu",
        Some(10u32.to_ne_bytes().to_vec()),
    );
    f.set_sysctl(
        "hw.perflevel0.cpusperl2",
        Some(10u32.to_ne_bytes().to_vec()),
    );
    let t = enumerate(&f, NODE).unwrap().cpus[0]
        .ext
        .cpu
        .clone()
        .unwrap();
    assert_eq!((t.perf_cores, t.efficiency_cores, t.ccds), (10, 0, 1));
    assert_eq!(t.llc, Bytes(16 * MIB));

    // No SME2: the level drops to `Arm86I8mm`.
    let mut f = fixture("m4-mini");
    f.set_sysctl(
        "hw.optional.arm.FEAT_SME2",
        Some(0u32.to_ne_bytes().to_vec()),
    );
    assert_eq!(
        enumerate(&f, NODE).unwrap().cpus[0].class,
        DeviceClass::Cpu {
            isa: CpuIsa::Arm86I8mm
        }
    );
}

/// Criterion 1 on a macOS runner: the live records have no unknown field, the wired limit
/// is read and not written, and the records and the recorded responses go to the job log.
#[cfg(target_os = "macos")]
#[test]
fn live_host_has_no_unknown_fields() {
    let rec = MacRecorder::new(syntropy_probe::macos::LiveMac);
    let host = enumerate(&rec, NODE).unwrap_or_else(|e| panic!("live enumeration: {e}"));
    common::evidence(&format!(
        "\n=== PLC-E1-T03 live macOS E1 records ===\n{host:#?}\n=== recorded responses (fixture format) ===\n{}=== end ===\n",
        rec.fixture()
    ));
    let unknown = common::unknown_fields(&host.node, &host.cpus, &host.host_mem, false);
    assert!(unknown.is_empty(), "unknown fields: {unknown:?}");
    assert!(rec.sysctl_reads().iter().any(|n| n == WIRED_LIMIT));
    assert!(rec.sysctl_writes().is_empty());
    // The recorded responses reproduce the live result.
    let again = MacFixture::parse(&rec.fixture()).unwrap();
    assert_eq!(enumerate(&again, NODE).unwrap(), host);
}
