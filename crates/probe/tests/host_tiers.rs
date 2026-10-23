// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLC-E1-T08's acceptance criteria: `host_tiers` over E1's recorded fixtures.
//!
//! - Criterion 1, a dual-socket host gets two `HostPinned` and two `HostPageable` tiers with
//!   the right affinities: `linux_dual_socket` and `windows_dual_socket`.
//! - Criterion 2, a single-domain host gets one of each: `single_domain_hosts`.
//! - Criterion 3, one test per OS rule set: `linux_memlock`, `linux_cgroup`,
//!   `windows_lockable`, `wsl2_pinned_fraction` and `macos_has_no_host_pinned`.
//! - Criterion 4, a cgroup-limited container's `usable` follows the cgroup limit:
//!   `linux_cgroup`.
//!
//! The fixtures' memory figures are the ones `linux_fixtures.rs`, `windows_fixtures.rs` and
//! `macos_fixtures.rs` check. Every expected byte count is written out, with its derivation
//! from RFC-0020 §5 in a comment: `headroom = max(4 GiB, 0.10 · total)`, `host_usable =
//! min(available + own − headroom, cgroup + own)`, `numa_available_n = MemFree_n +
//! Active(file)_n + Inactive(file)_n + SReclaimable_n − 1 GiB`, `host_usable_n =
//! min(numa_available_n + own_n, host_usable)` and `pin_limit_n = min(pin_limit,
//! host_usable_n)`.

use smallvec::SmallVec;
use syntropy_core::budget::{HostBudgetPolicy, HostOs, NodeMem};
use syntropy_core::ids::{DeviceId, NodeId, TierId};
use syntropy_core::resource::ordinal::{self, DeviceCandidate, Enumerated};
use syntropy_core::resource::{
    BudgetSource, DeviceKind, HostPoolInfo, PciAddress, TierInfo, TierInfoExt, TierKind,
};
use syntropy_core::units::Bytes;
use syntropy_probe::host::NumaHostMem;
use syntropy_probe::linux::{self, IsaSource, RawCall, Syscalls};
use syntropy_probe::{DirFs, HostTierError, HostTierInputs, HostTiers, host_tiers};
use syntropy_probe::{macos, windows};

const NODE: NodeId = NodeId([7; 32]);
const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const GIB: u64 = 1024 * MIB;

/// Every fixture host grants both system calls (`linux_fixtures.rs` says why).
struct FixtureCalls;

impl Syscalls for FixtureCalls {
    fn request_xtiledata_permission(&self) -> RawCall {
        RawCall::Ok
    }

    fn io_uring_setup(&self) -> RawCall {
        RawCall::Ok
    }
}

/// E1's host memory records and ordinal candidates for one fixture host.
struct Host {
    mem: Vec<NumaHostMem>,
    candidates: Enumerated,
}

fn linux(scenario: &str) -> Host {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/linux/");
    let h = linux::enumerate(
        &DirFs::new(format!("{root}{scenario}")),
        IsaSource::CpuInfo,
        &FixtureCalls,
        NODE,
    )
    .unwrap_or_else(|e| panic!("{scenario}: {e}"));
    Host {
        mem: h.host_mem,
        candidates: h.candidates,
    }
}

fn windows(name: &str) -> Host {
    let path = format!(
        "{}/tests/fixtures/windows/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut f = windows::WinFixture::parse(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
    if name == "windows-2025-runner" {
        // The capture records no job-object response (`windows_fixtures.rs` says why).
        f.set_job(None);
    }
    let h = windows::enumerate(&f, NODE).unwrap_or_else(|e| panic!("{name}: {e}"));
    Host {
        mem: h.host_mem,
        candidates: h.candidates,
    }
}

fn macos(name: &str) -> Host {
    let path = format!(
        "{}/tests/fixtures/macos/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let f = macos::MacFixture::parse(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
    let h = macos::enumerate(&f, NODE).unwrap_or_else(|e| panic!("{name}: {e}"));
    Host {
        mem: h.host_mem,
        candidates: h.candidates,
    }
}

/// A discrete GPU on PCI bus `bus` whose sysfs or SetupAPI NUMA node is `numa`.
fn gpu(bus: u8, numa: Option<u16>) -> DeviceCandidate {
    DeviceCandidate {
        kind: DeviceKind::DiscreteGpu,
        pci: Some(PciAddress {
            domain: 0,
            bus,
            device: 0,
            function: 0,
        }),
        numa,
        uuid: None,
        name: format!("GPU {bus:02x}"),
        unified: false,
    }
}

/// Runs `host_tiers` on `host` with `gpus` added to its candidates, as the graph builder
/// numbers the node once E2 has found them.
fn build(
    os: HostOs,
    host: &Host,
    gpus: &[DeviceCandidate],
    own: &[(u16, Bytes)],
) -> Result<HostTiers, HostTierError> {
    let mut found = host.candidates.clone();
    found.devices.extend_from_slice(gpus);
    let ordinals = ordinal::assign(NODE, &found).unwrap();
    host_tiers(&HostTierInputs {
        os,
        host_mem: &host.mem,
        ordinals: &ordinals,
        policy: &HostBudgetPolicy::default(),
        own,
    })
}

fn dev(ordinal: u16) -> DeviceId {
    DeviceId {
        node: NODE,
        ordinal,
    }
}

fn tid(ordinal: u16) -> TierId {
    TierId {
        node: NODE,
        ordinal,
    }
}

/// The record `host_tiers` should build for one host tier of `mem`'s domain.
fn tier(
    ordinal: u16,
    kind: TierKind,
    mem: &NumaHostMem,
    usable: u64,
    affinity: &[u16],
) -> TierInfo {
    TierInfo {
        id: tid(ordinal),
        kind,
        capacity: mem.node_total,
        usable: Bytes(usable),
        affinity: affinity.iter().map(|&d| dev(d)).collect::<SmallVec<_>>(),
        pin_limit: (kind == TierKind::HostPinned).then_some(Bytes(usable)),
        volume: None,
        ext: TierInfoExt {
            numa: Some(mem.numa),
            budget_source: BudgetSource::HostRules,
            host: Some(mem.mem.clone()),
            context_reserve: None,
        },
    }
}

/// A domain's `[HostPinned, HostPageable]` pair at tier ordinals `first` and `first + 1`.
fn pair(first: u16, mem: &NumaHostMem, usable: u64, pin: u64, affinity: &[u16]) -> Vec<TierInfo> {
    vec![
        tier(first, TierKind::HostPinned, mem, pin, affinity),
        tier(first + 1, TierKind::HostPageable, mem, usable, affinity),
    ]
}

fn pool(numa: u16, first: u16, usable: u64, pin: u64) -> HostPoolInfo {
    HostPoolInfo {
        numa,
        tiers: [tid(first), tid(first + 1)],
        usable: Bytes(usable),
        pin_limit: Bytes(pin),
    }
}

/// The single-domain host's `[HostPinned, HostPageable]` usable figures.
fn single(t: &HostTiers) -> (u64, u64) {
    assert_eq!(t.tiers.len(), 2, "{t:?}");
    assert_eq!(t.pools.len(), 1, "{t:?}");
    (t.tiers[0].usable.0, t.tiers[1].usable.0)
}

/// Criterion 1 on Linux. GPUs: bus 0x17 on node 0, bus 0x65 with no NUMA node (sysfs `-1`),
/// bus 0xb1 on node 1, and bus 0xca on node 5, which has no host memory. The two without a
/// host domain attach to domain 0. Ordinals: GPUs 0–3 by PCI address, CPUs 4 and 5; `Vram`
/// tiers 0–3, then `HostPinned_0` 4, `HostPageable_0` 5, `HostPinned_1` 6, `HostPageable_1` 7.
#[test]
fn linux_dual_socket() {
    let host = linux("dual-socket");
    let gpus = [
        gpu(0x17, Some(0)),
        gpu(0x65, None),
        gpu(0xb1, Some(1)),
        gpu(0xca, Some(5)),
    ];
    let t = build(HostOs::Linux, &host, &gpus, &[]).unwrap();
    // total 263_856_048 KiB, available 252_874_220 KiB; headroom 0.10 · total =
    // 27_018_859_315; host_usable 231_924_341_965; CAP_IPC_LOCK, so pin_limit = 0.85 · total
    // = 229_660_304_179.
    // Node 0: (120_112_508 + 1_602_340 + 2_104_772 + 512_388) KiB − 1 GiB = 126_242_234_368.
    // Node 1: (120_889_808 + 1_498_116 + 1_987_204 + 498_772) KiB − 1 GiB = 126_797_131_776.
    // Both are below host_usable and pin_limit, so each domain's pair is its own estimate.
    let (n0, n1) = (126_242_234_368, 126_797_131_776);
    let mut want = pair(4, &host.mem[0], n0, n0, &[0, 1, 3, 4]);
    want.extend(pair(6, &host.mem[1], n1, n1, &[2, 5]));
    assert_eq!(t.tiers, want);
    assert_eq!(t.pools, vec![pool(0, 4, n0, n0), pool(1, 6, n1, n1)]);
}

/// Criterion 1 on Windows: the SRAT fixture's two domains. GPUs: bus 0x41 on node 1, bus 0x81
/// on node 0. Ordinals: GPUs 0 and 1, CPUs 2 and 3; `Vram` tiers 0 and 1, then the host pairs
/// at 2 and 4.
#[test]
fn windows_dual_socket() {
    let host = windows("dual-socket");
    let t = build(
        HostOs::Windows,
        &host,
        &[gpu(0x41, Some(1)), gpu(0x81, Some(0))],
        &[],
    )
    .unwrap();
    // total 253 GiB, ullAvailPhys 250 GiB; headroom 27_165_668_147; host_usable
    // 241_269_787_853; pin_limit = min(host_usable, 0.50 · total = 135_828_340_736, lockable
    // 200 GiB) = 135_828_340_736. Each node's available figure less 1 GiB: 119 GiB and
    // 125 GiB, both below pin_limit.
    let (n0, n1) = (119 * GIB, 125 * GIB);
    let mut want = pair(2, &host.mem[0], n0, n0, &[1, 2]);
    want.extend(pair(4, &host.mem[1], n1, n1, &[0, 3]));
    assert_eq!(t.tiers, want);
    assert_eq!(t.pools, vec![pool(0, 2, n0, n0), pool(1, 4, n1, n1)]);
}

/// Criterion 2: every single-domain Linux, WSL2 and Windows fixture gets one `HostPinned` and
/// one `HostPageable` tier, linked by one pool record, whose affinity is the CPU device and
/// the GPU, whatever NUMA node the GPU reports.
#[test]
fn single_domain_hosts() {
    let hosts = [
        (HostOs::Linux, "linux desktop", linux("desktop")),
        (HostOs::Linux, "linux docker", linux("docker")),
        (HostOs::Linux, "linux wsl2", linux("wsl2")),
        (HostOs::Linux, "linux rpi5", linux("rpi5")),
        (HostOs::Windows, "windows desktop", windows("desktop")),
        (
            HostOs::Windows,
            "windows runner",
            windows("windows-2025-runner"),
        ),
    ];
    for (os, name, host) in &hosts {
        for numa in [None, Some(0), Some(1)] {
            let t = build(*os, host, &[gpu(0x01, numa)], &[]).unwrap();
            let kinds: Vec<TierKind> = t.tiers.iter().map(|t| t.kind).collect();
            assert_eq!(
                kinds,
                vec![TierKind::HostPinned, TierKind::HostPageable],
                "{name}"
            );
            // GPU 0 and CPU 1; `Vram` 0, then the pair at 1 and 2.
            for tier in &t.tiers {
                assert_eq!(tier.affinity.as_slice(), &[dev(0), dev(1)], "{name}");
                assert_eq!(tier.ext.numa, Some(0), "{name}");
            }
            assert_eq!(t.pools.len(), 1, "{name}");
            assert_eq!(t.pools[0].tiers, [tid(1), tid(2)], "{name}");
            assert_eq!(t.pools[0].usable, t.tiers[1].usable, "{name}");
            assert_eq!(t.pools[0].pin_limit, t.tiers[0].usable, "{name}");
        }
    }
}

/// Criterion 3, Linux `RLIMIT_MEMLOCK`. The desktop fixture's 8 MiB rlimit, without
/// `CAP_IPC_LOCK`, is `pin_limit`; with the capability (no limit) the 0.85 fraction and
/// `host_usable` apply.
#[test]
fn linux_memlock() {
    let mut host = linux("desktop");
    // total 32_562_044 KiB, available 26_120_588 KiB; headroom 4 GiB; host_usable
    // 22_452_514_816. Node 0: (18_304_412 + 2_310_456 + 3_402_108 + 389_204) KiB − 1 GiB =
    // 23_918_186_496, above host_usable.
    let t = build(HostOs::Linux, &host, &[], &[]).unwrap();
    assert_eq!(
        t.tiers,
        pair(0, &host.mem[0], 22_452_514_816, 8 * MIB, &[0])
    );
    assert_eq!(t.pools, vec![pool(0, 0, 22_452_514_816, 8 * MIB)]);

    // 0.85 · total = 28_342_003_097, above host_usable.
    host.mem[0].mem.memlock_limit = None;
    let t = build(HostOs::Linux, &host, &[], &[]).unwrap();
    assert_eq!(single(&t), (22_452_514_816, 22_452_514_816));
}

/// Criteria 3 and 4, the Linux cgroup limit. The Docker fixture's `memory.max` 8 GiB less
/// `memory.current` 1 GiB caps `usable`, below the node estimate and `MemAvailable −
/// headroom`; `usable` follows the limit when it changes and the node estimate when there is
/// none. The 8 MiB rlimit is `pin_limit`.
#[test]
fn linux_cgroup() {
    let mut host = linux("docker");
    let t = build(HostOs::Linux, &host, &[], &[]).unwrap();
    assert_eq!(t.tiers, pair(0, &host.mem[0], 7 * GIB, 8 * MIB, &[0]));
    assert_eq!(t.pools, vec![pool(0, 0, 7 * GIB, 8 * MIB)]);

    host.mem[0].mem.cgroup_limit = Some(Bytes(3 * GIB));
    let t = build(HostOs::Linux, &host, &[], &[]).unwrap();
    assert_eq!(single(&t), (8 * MIB, 3 * GIB));

    // A limit below the rlimit bounds `pin_limit` too.
    host.mem[0].mem.cgroup_limit = Some(Bytes(4 * MIB));
    let t = build(HostOs::Linux, &host, &[], &[]).unwrap();
    assert_eq!(single(&t), (4 * MIB, 4 * MIB));

    // No limit: available 58_993_172 KiB less headroom 0.10 · 65_777_452 KiB =
    // 53_673_397_044; node 0 (40_221_396 + 2_198_004 + 3_011_552 + 402_876) KiB − 1 GiB =
    // 45_860_098_048 is lower.
    host.mem[0].mem.cgroup_limit = None;
    let t = build(HostOs::Linux, &host, &[], &[]).unwrap();
    assert_eq!(single(&t), (8 * MIB, 45_860_098_048));
}

/// Criterion 3, WSL2: Linux's rule over the VM's `MemTotal`, with `max_pinned_fraction` 0.75.
/// The fixture's 64 KiB rlimit binds first, so the fraction shows once the limit is lifted.
#[test]
fn wsl2_pinned_fraction() {
    let mut host = linux("wsl2");
    assert!(host.mem[0].mem.wsl2);
    // total 32_803_456 KiB, available 31_207_884 KiB; headroom 4 GiB; host_usable
    // 27_661_905_920, below node 0's (30_112_708 + 312_444 + 401_256 + 98_312) KiB − 1 GiB =
    // 30_593_171_456.
    let t = build(HostOs::Linux, &host, &[], &[]).unwrap();
    assert_eq!(
        t.tiers,
        pair(0, &host.mem[0], 27_661_905_920, 64 * KIB, &[0])
    );

    // 0.75 · total = 25_193_054_208, below host_usable.
    host.mem[0].mem.memlock_limit = None;
    let t = build(HostOs::Linux, &host, &[], &[]).unwrap();
    assert_eq!(single(&t), (25_193_054_208, 27_661_905_920));

    // Bare-metal Linux's 0.85 · total = 28_552_128_102 is above host_usable.
    host.mem[0].mem.wsl2 = false;
    let t = build(HostOs::Linux, &host, &[], &[]).unwrap();
    assert_eq!(single(&t), (27_661_905_920, 27_661_905_920));
}

/// Criterion 3, Windows: `pin_limit = min(host_usable, 0.50 · ullTotalPhys, lockable)`, with
/// E1's working-set ceiling as `lockable` until H5 measures one, and 0 with neither.
#[test]
fn windows_lockable() {
    let mut host = windows("desktop");
    // total 34_190_000_128, ullAvailPhys 24 GiB; headroom 4 GiB; host_usable 20 GiB, below the
    // node's 24 GiB − 1 GiB. 0.50 · total = 17_095_000_064, below the 30 GiB ceiling.
    let t = build(HostOs::Windows, &host, &[], &[]).unwrap();
    assert_eq!(
        t.tiers,
        pair(0, &host.mem[0], 20 * GIB, 17_095_000_064, &[0])
    );
    assert_eq!(t.pools, vec![pool(0, 0, 20 * GIB, 17_095_000_064)]);

    host.mem[0].mem.windows_lockable = Some(Bytes(6 * GIB));
    let t = build(HostOs::Windows, &host, &[], &[]).unwrap();
    assert_eq!(single(&t), (6 * GIB, 20 * GIB));

    host.mem[0].mem.windows_lockable = None;
    let t = build(HostOs::Windows, &host, &[], &[]).unwrap();
    assert_eq!(single(&t), (0, 20 * GIB));
}

/// Criterion 3, macOS: E1 gives the one domain no `HostPinned` slot, so the step builds one
/// `HostPageable` tier and no pool record. `usable` is RFC-0020 §5's macOS rule without the
/// working-set term, which belongs to the GPU's `Unified` tier.
#[test]
fn macos_has_no_host_pinned() {
    let host = macos("m4-mini");
    let t = build(HostOs::MacOs, &host, &[], &[]).unwrap();
    // total 16 GiB; available (28_015 + 319_984 + 1_476) · 16 KiB pages = 5_725_798_400;
    // headroom 4 GiB.
    assert_eq!(
        t.tiers,
        vec![tier(
            0,
            TierKind::HostPageable,
            &host.mem[0],
            1_430_831_104,
            &[0]
        )]
    );
    assert_eq!(t.pools, vec![]);
}

/// `own_n` is added back to the domain it is held in, and the host-wide `own` to
/// `host_usable`, so a re-plan does not count a running plan against itself.
#[test]
fn own_memory_is_added_back_per_domain() {
    let host = linux("dual-socket");
    let t = build(HostOs::Linux, &host, &[], &[(0, Bytes(4 * GIB))]).unwrap();
    // Node 0: 126_242_234_368 + 4 GiB, below host_usable + 4 GiB. Node 1 is unchanged.
    let (n0, n1) = (126_242_234_368 + 4 * GIB, 126_797_131_776);
    assert_eq!(t.pools, vec![pool(0, 0, n0, n0), pool(1, 2, n1, n1)]);
}

#[test]
fn inconsistent_inputs_are_errors() {
    // The ordinals have domain 1's tiers but E1 has no record of its memory.
    let mut host = linux("dual-socket");
    host.mem.pop();
    assert_eq!(
        build(HostOs::Linux, &host, &[], &[]),
        Err(HostTierError::NoMemory { numa: 1 })
    );

    // A Linux host's domain has a `HostPinned` slot, which the macOS rule set has not.
    let host = linux("desktop");
    assert_eq!(
        build(HostOs::MacOs, &host, &[], &[]),
        Err(HostTierError::PinnedMismatch { numa: 0 })
    );
    // And the other way round.
    let host = macos("m4-mini");
    assert_eq!(
        build(HostOs::Linux, &host, &[], &[]),
        Err(HostTierError::PinnedMismatch { numa: 0 })
    );

    // Without its per-node figure, a domain's share of a two-domain host is unknown.
    let mut host = linux("dual-socket");
    host.mem[1].node_mem = None;
    assert_eq!(
        build(HostOs::Linux, &host, &[], &[]),
        Err(HostTierError::NoNodeMemory { numa: 1 })
    );

    // A corrupt reading whose sum overflows.
    let mut host = linux("dual-socket");
    host.mem[0].node_mem = Some(NodeMem {
        mem_free: Bytes(u64::MAX),
        active_file: Bytes(1),
        ..NodeMem::default()
    });
    assert_eq!(
        build(HostOs::Linux, &host, &[], &[]),
        Err(HostTierError::NodeMemoryOverflow { numa: 0 })
    );
}
