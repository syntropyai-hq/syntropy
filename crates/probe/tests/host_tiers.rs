// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLC-E1-T08's acceptance criteria: `host_tiers` over E1's recorded fixtures.
//!
//! Linux is covered end to end: each test edits the recorded sysfs and procfs tree, runs
//! Linux E1 over it, and checks the tiers `host_tiers` builds from E1's records.
//!
//! - Criterion 1, a dual-socket host gets two `HostPinned` and two `HostPageable` tiers with
//!   the right affinities: `linux_dual_socket`.
//! - Criterion 2, a single-domain host gets one of each: `single_domain_linux_hosts`,
//!   `no_numa_directory_is_one_domain`.
//! - Criterion 3, one test per OS rule set. Linux memlock: `linux_memlock`. Linux cgroup:
//!   `cgroup_v2_container`, `cgroup_v2_memory_high`, `cgroup_v2_parent_limit`,
//!   `cgroup_v1_container`, `cgroup_on_a_two_domain_host`. WSL2: `wsl2_pinned_fraction`.
//!   Windows: `windows_lockable`. macOS: `macos_has_no_host_pinned`.
//! - Criterion 4, a cgroup-limited container's `usable` follows the cgroup limit: the cgroup
//!   tests above.
//!
//! The fixtures' memory figures are the ones `linux_fixtures.rs`, `windows_fixtures.rs` and
//! `macos_fixtures.rs` check. Every expected byte count is written out, with its derivation
//! from RFC-0020 §5 in a comment: `headroom = max(4 GiB, 0.10 · total)`, `host_usable =
//! min(available + own − headroom, cgroup + own)`, `numa_available_n = MemFree_n +
//! Active(file)_n + Inactive(file)_n + SReclaimable_n − 1 GiB`, `host_usable_n =
//! min(numa_available_n + own_n, host_usable)` and `pin_limit_n = min(pin_limit,
//! host_usable_n)`.

use std::collections::BTreeMap;
use std::io;

use smallvec::SmallVec;
use syntropy_core::budget::{HostBudgetPolicy, HostOs, NodeMem};
use syntropy_core::ids::{DeviceId, NodeId, TierId};
use syntropy_core::resource::ordinal::{self, DeviceCandidate, Enumerated};
use syntropy_core::resource::{
    BudgetSource, DeviceKind, HostPoolInfo, PciAddress, TierInfo, TierInfoExt, TierKind,
};
use syntropy_core::units::Bytes;
use syntropy_probe::host::NumaHostMem;
use syntropy_probe::linux::{self, CgroupVersion, IsaSource, RawCall, Syscalls};
use syntropy_probe::{DirFs, EnumError, HostFs, HostTierError, HostTierInputs, HostTiers};
use syntropy_probe::{host_tiers, macos, windows};

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

/// A recorded Linux fixture tree with files replaced, added or removed. A path set after a
/// removal under it is visible; the rest of the removed subtree is not.
struct Tree {
    base: DirFs,
    set: BTreeMap<String, String>,
    gone: Vec<String>,
}

impl Tree {
    fn new(scenario: &str) -> Self {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/linux/");
        Self {
            base: DirFs::new(format!("{root}{scenario}")),
            set: BTreeMap::new(),
            gone: Vec::new(),
        }
    }

    fn set(&mut self, path: &str, text: &str) -> &mut Self {
        self.set.insert(path.into(), text.into());
        self
    }

    fn remove(&mut self, path: &str) -> &mut Self {
        self.set
            .retain(|k, _| k != path && !k.starts_with(&format!("{path}/")));
        self.gone.push(path.into());
        self
    }

    fn is_gone(&self, path: &str) -> bool {
        self.gone
            .iter()
            .any(|g| path == g || path.starts_with(&format!("{g}/")))
    }

    fn enumerate(&self) -> Result<linux::LinuxHost, EnumError> {
        linux::enumerate(self, IsaSource::CpuInfo, &FixtureCalls, NODE)
    }

    fn host(&self) -> Host {
        let h = self.enumerate().unwrap();
        Host {
            mem: h.host_mem,
            candidates: h.candidates,
        }
    }
}

impl HostFs for Tree {
    fn read(&self, path: &str) -> io::Result<String> {
        match self.set.get(path) {
            Some(text) => Ok(text.clone()),
            None if self.is_gone(path) => Err(io::ErrorKind::NotFound.into()),
            None => self.base.read(path),
        }
    }

    fn exists(&self, path: &str) -> bool {
        let dir = format!("{path}/");
        self.set.keys().any(|k| k == path || k.starts_with(&dir))
            || (!self.is_gone(path) && self.base.exists(path))
    }

    fn list(&self, path: &str) -> io::Result<Vec<String>> {
        let dir = format!("{path}/");
        let mut names: Vec<String> = self
            .set
            .keys()
            .filter_map(|k| k.strip_prefix(&dir))
            .map(|rest| rest.split('/').next().unwrap_or(rest).to_owned())
            .collect();
        let base = if self.is_gone(path) {
            Err(io::ErrorKind::NotFound.into())
        } else {
            self.base.list(path)
        };
        match base {
            Ok(found) => names.extend(
                found
                    .into_iter()
                    .filter(|n| !self.is_gone(&format!("{dir}{n}"))),
            ),
            Err(e) if names.is_empty() => return Err(e),
            Err(_) => {}
        }
        names.sort();
        names.dedup();
        Ok(names)
    }
}

/// E1's host memory records and ordinal candidates for one fixture host.
struct Host {
    mem: Vec<NumaHostMem>,
    candidates: Enumerated,
}

fn linux(scenario: &str) -> Host {
    Tree::new(scenario).host()
}

fn windows(name: &str) -> Host {
    let path = format!(
        "{}/tests/fixtures/windows/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let f = windows::WinFixture::parse(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
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

/// A discrete GPU on PCI bus `bus` whose sysfs `numa_node` is `numa`.
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

/// [`build`] for a Linux host with no GPU and nothing held.
fn linux_tiers(host: &Host) -> HostTiers {
    build(HostOs::Linux, host, &[], &[]).unwrap()
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

/// A single-domain host's `(pin_limit, usable)`: its `HostPinned` and `HostPageable`
/// `usable`, after checking the pool record agrees.
fn single(t: &HostTiers) -> (u64, u64) {
    assert_eq!(t.tiers.len(), 2, "{t:?}");
    assert_eq!(t.pools.len(), 1, "{t:?}");
    let (pin, usable) = (t.tiers[0].usable.0, t.tiers[1].usable.0);
    assert_eq!((t.pools[0].pin_limit.0, t.pools[0].usable.0), (pin, usable));
    (pin, usable)
}

/// Each domain's `(pin_limit_n, host_usable_n)` from the pool records.
fn pools(t: &HostTiers) -> Vec<(u16, u64, u64)> {
    t.pools
        .iter()
        .map(|p| (p.numa, p.pin_limit.0, p.usable.0))
        .collect()
}

// The dual-socket fixture: total 263_856_048 KiB, available 252_874_220 KiB; headroom 0.10 ·
// total = 27_018_859_315; host_usable 231_924_341_965; CAP_IPC_LOCK, so pin_limit = 0.85 ·
// total = 229_660_304_179.
// Node 0: (120_112_508 + 1_602_340 + 2_104_772 + 512_388) KiB − 1 GiB = 126_242_234_368.
// Node 1: (120_889_808 + 1_498_116 + 1_987_204 + 498_772) KiB − 1 GiB = 126_797_131_776.
const DUAL_N0: u64 = 126_242_234_368;
const DUAL_N1: u64 = 126_797_131_776;

/// Criterion 1. GPUs: bus 0x17 on node 0, bus 0x65 with no NUMA node (sysfs `-1`), bus 0xb1
/// on node 1, and bus 0xca on node 5, which has no host memory. The two without a host
/// domain attach to domain 0. Ordinals: GPUs 0–3 by PCI address, CPUs 4 and 5; `Vram` tiers
/// 0–3, then `HostPinned_0` 4, `HostPageable_0` 5, `HostPinned_1` 6, `HostPageable_1` 7.
/// Both nodes' estimates are below `host_usable` and `pin_limit`, so each domain's pair is
/// its own estimate.
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
    let (n0, n1) = (DUAL_N0, DUAL_N1);
    let mut want = pair(4, &host.mem[0], n0, n0, &[0, 1, 3, 4]);
    want.extend(pair(6, &host.mem[1], n1, n1, &[2, 5]));
    assert_eq!(t.tiers, want);
    assert_eq!(t.pools, vec![pool(0, 4, n0, n0), pool(1, 6, n1, n1)]);
}

/// The per-NUMA cap reads each node's own `meminfo`: a node with little reclaimable memory
/// caps its own domain only.
#[test]
fn per_node_meminfo_caps_its_own_domain() {
    let mut tree = Tree::new("dual-socket");
    tree.set(
        "sys/devices/system/node/node1/meminfo",
        "Node 1 MemTotal:       132001692 kB\n\
         Node 1 MemFree:         10485760 kB\n\
         Node 1 Active(file):     1498116 kB\n\
         Node 1 Inactive(file):   1987204 kB\n\
         Node 1 SReclaimable:      498772 kB\n",
    );
    // Node 1: (10_485_760 + 1_498_116 + 1_987_204 + 498_772) KiB − 1 GiB = 13_743_386_624.
    let t = linux_tiers(&tree.host());
    assert_eq!(
        pools(&t),
        vec![(0, DUAL_N0, DUAL_N0), (1, 13_743_386_624, 13_743_386_624)]
    );
}

/// A node `meminfo` without one of the four terms is malformed: E1 never takes a missing
/// term as zero, which would understate the cap without saying so.
#[test]
fn node_meminfo_without_a_reclaimable_term_is_malformed() {
    const PATH: &str = "sys/devices/system/node/node1/meminfo";
    for key in ["Active(file)", "Inactive(file)", "SReclaimable"] {
        let text: String = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/linux/dual-socket/sys/devices/system/node/node1/meminfo"
        ))
        .unwrap()
        .lines()
        .filter(|l| !l.contains(key))
        .map(|l| format!("{l}\n"))
        .collect();
        let mut tree = Tree::new("dual-socket");
        tree.set(PATH, &text);
        match tree.enumerate() {
            Err(EnumError::Malformed { path, .. }) => assert_eq!(path, PATH, "{key}"),
            other => panic!("{key}: expected Malformed at {PATH}, got {other:?}"),
        }
    }
}

/// A node with CPUs and no memory gets a CPU device but no host tiers. Its CPU is in no host
/// tier's affinity, and a GPU on that node attaches to domain 0. Ordinals: GPU 0, CPUs 1 and
/// 2; `Vram` 0, then domain 0's pair at 1 and 2.
#[test]
fn memoryless_node_has_no_host_tiers() {
    let mut tree = Tree::new("dual-socket");
    tree.set(
        "sys/devices/system/node/node1/meminfo",
        "Node 1 MemTotal: 0 kB\nNode 1 MemFree: 0 kB\n",
    );
    let host = tree.host();
    let t = build(HostOs::Linux, &host, &[gpu(0x17, Some(1))], &[]).unwrap();
    assert_eq!(t.tiers, pair(1, &host.mem[0], DUAL_N0, DUAL_N0, &[0, 1]));
    assert_eq!(t.pools, vec![pool(0, 1, DUAL_N0, DUAL_N0)]);
}

/// Criterion 2: a kernel without NUMA (no `sys/devices/system/node`) is one domain, whose
/// estimate comes from `/proc/meminfo`: (241_002_316 + 3_100_456 + 4_091_976 + 1_011_160)
/// KiB − 1 GiB = 254_113_107_968, above host_usable.
#[test]
fn no_numa_directory_is_one_domain() {
    let mut tree = Tree::new("dual-socket");
    tree.remove("sys/devices/system/node");
    let host = tree.host();
    let t = linux_tiers(&host);
    assert_eq!(
        t.tiers,
        pair(0, &host.mem[0], 231_924_341_965, 229_660_304_179, &[0])
    );
    assert_eq!(single(&t), (229_660_304_179, 231_924_341_965));
}

/// Criterion 2: every single-domain Linux fixture gets one `HostPinned` and one
/// `HostPageable` tier, linked by one pool record, whose affinity is the CPU device and the
/// GPU, whatever NUMA node the GPU reports. Ordinals: GPU 0, CPU 1; `Vram` 0, then the pair
/// at 1 and 2.
#[test]
fn single_domain_linux_hosts() {
    for name in ["desktop", "docker", "wsl2", "rpi5"] {
        let host = linux(name);
        for numa in [None, Some(0), Some(1)] {
            let t = build(HostOs::Linux, &host, &[gpu(0x01, numa)], &[]).unwrap();
            let kinds: Vec<TierKind> = t.tiers.iter().map(|t| t.kind).collect();
            assert_eq!(
                kinds,
                vec![TierKind::HostPinned, TierKind::HostPageable],
                "{name}"
            );
            for tier in &t.tiers {
                assert_eq!(tier.affinity.as_slice(), &[dev(0), dev(1)], "{name}");
                assert_eq!(tier.ext.numa, Some(0), "{name}");
                assert_eq!(tier.ext.budget_source, BudgetSource::HostRules, "{name}");
            }
            assert_eq!(t.pools.len(), 1, "{name}");
            assert_eq!(t.pools[0].tiers, [tid(1), tid(2)], "{name}");
            single(&t);
        }
    }
}

/// The desktop fixture: total 32_562_044 KiB, available 26_120_588 KiB; headroom 4 GiB;
/// host_usable 22_452_514_816, below node 0's (18_304_412 + 2_310_456 + 3_402_108 + 389_204)
/// KiB − 1 GiB = 23_918_186_496. 0.85 · total = 28_342_003_097, above host_usable.
const DESKTOP_USABLE: u64 = 22_452_514_816;

/// `/proc/self/limits`' memlock row, soft then hard limit.
fn memlock_row(tree: &mut Tree, soft: &str, hard: &str) {
    let path = "proc/self/limits";
    let text: String = tree
        .read(path)
        .unwrap()
        .lines()
        .map(|l| {
            if l.starts_with("Max locked memory") {
                format!("Max locked memory         {soft:<20} {hard:<20} bytes     \n")
            } else {
                format!("{l}\n")
            }
        })
        .collect();
    tree.set(path, &text);
}

/// Criterion 3, Linux `RLIMIT_MEMLOCK` from `/proc/self/limits` and `CAP_IPC_LOCK` from
/// `/proc/self/status`. The soft limit is `pin_limit` when it binds; an unlimited soft limit
/// or the capability leaves `min(host_usable, 0.85 · total)`.
#[test]
fn linux_memlock() {
    let host = linux("desktop");
    let t = linux_tiers(&host);
    // The recorded 8 MiB soft limit, without the capability.
    assert_eq!(
        t.tiers,
        pair(0, &host.mem[0], DESKTOP_USABLE, 8 * MIB, &[0])
    );
    assert_eq!(t.pools, vec![pool(0, 0, DESKTOP_USABLE, 8 * MIB)]);

    let mut tree = Tree::new("desktop");
    memlock_row(&mut tree, "2147483648", "unlimited");
    assert_eq!(
        single(&linux_tiers(&tree.host())),
        (2 * GIB, DESKTOP_USABLE)
    );

    // A soft limit above host_usable does not raise pin_limit past it.
    memlock_row(&mut tree, "68719476736", "68719476736");
    assert_eq!(
        single(&linux_tiers(&tree.host())),
        (DESKTOP_USABLE, DESKTOP_USABLE)
    );

    memlock_row(&mut tree, "unlimited", "unlimited");
    assert_eq!(
        single(&linux_tiers(&tree.host())),
        (DESKTOP_USABLE, DESKTOP_USABLE)
    );

    // CAP_IPC_LOCK (bit 14 of CapEff) lifts the recorded 8 MiB limit.
    let mut tree = Tree::new("desktop");
    let status: String = tree
        .read("proc/self/status")
        .unwrap()
        .lines()
        .map(|l| match l.strip_prefix("CapEff:") {
            Some(_) => "CapEff:\t0000000000004000\n".to_owned(),
            None => format!("{l}\n"),
        })
        .collect();
    tree.set("proc/self/status", &status);
    assert_eq!(
        single(&linux_tiers(&tree.host())),
        (DESKTOP_USABLE, DESKTOP_USABLE)
    );
}

/// Criteria 3 and 4, a cgroup v2 container. The Docker fixture's `memory.max` 8 GiB less
/// `memory.current` 1 GiB caps `usable`, below the node estimate and `MemAvailable −
/// headroom`; `usable` follows the limit as it changes, and the node estimate when there is
/// none. The 8 MiB rlimit is `pin_limit` unless the limit is lower.
#[test]
fn cgroup_v2_container() {
    let host = linux("docker");
    let t = linux_tiers(&host);
    assert_eq!(t.tiers, pair(0, &host.mem[0], 7 * GIB, 8 * MIB, &[0]));
    assert_eq!(t.pools, vec![pool(0, 0, 7 * GIB, 8 * MIB)]);

    let mut tree = Tree::new("docker");
    tree.set("sys/fs/cgroup/memory.max", &(4 * GIB).to_string());
    assert_eq!(single(&linux_tiers(&tree.host())), (8 * MIB, 3 * GIB));

    // More in use leaves less.
    tree.set("sys/fs/cgroup/memory.current", &(3 * GIB).to_string());
    assert_eq!(single(&linux_tiers(&tree.host())), (8 * MIB, GIB));

    // A limit below the rlimit bounds `pin_limit` too.
    tree.set(
        "sys/fs/cgroup/memory.current",
        &(4 * GIB - 4 * MIB).to_string(),
    );
    assert_eq!(single(&linux_tiers(&tree.host())), (4 * MIB, 4 * MIB));

    // Usage at or above the limit leaves nothing.
    tree.set("sys/fs/cgroup/memory.current", &(5 * GIB).to_string());
    assert_eq!(single(&linux_tiers(&tree.host())), (0, 0));

    // No limit: available 58_993_172 KiB less headroom 0.10 · 65_777_452 KiB =
    // 53_673_397_044; node 0's (40_221_396 + 2_198_004 + 3_011_552 + 402_876) KiB − 1 GiB =
    // 45_860_098_048 is lower.
    tree.set("sys/fs/cgroup/memory.max", "max\n");
    assert_eq!(
        single(&linux_tiers(&tree.host())),
        (8 * MIB, 45_860_098_048)
    );
}

/// Criterion 4: cgroup v2's `memory.high` throttles before `memory.max`, so the lower of the
/// two sets the limit.
#[test]
fn cgroup_v2_memory_high() {
    let mut tree = Tree::new("docker");
    tree.set("sys/fs/cgroup/memory.high", &(5 * GIB).to_string());
    // min(8, 5) GiB − 1 GiB.
    assert_eq!(single(&linux_tiers(&tree.host())), (8 * MIB, 4 * GIB));
}

/// Criterion 4: a parent cgroup's limit binds the process's own, the tightest level's
/// headroom wins.
#[test]
fn cgroup_v2_parent_limit() {
    let mut tree = Tree::new("desktop");
    let slice = "sys/fs/cgroup/user.slice/user-1000.slice";
    let leaf =
        "sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/syntropy.scope";
    tree.set(&format!("{slice}/memory.max"), &(16 * GIB).to_string())
        .set(&format!("{slice}/memory.current"), &(10 * GIB).to_string())
        .set(&format!("{leaf}/memory.max"), &(12 * GIB).to_string())
        .set(&format!("{leaf}/memory.current"), &GIB.to_string());
    // min(16 − 10, 12 − 1) GiB.
    assert_eq!(single(&linux_tiers(&tree.host())), (8 * MIB, 6 * GIB));
}

/// Criteria 3 and 4, a cgroup v1 container: `memory.limit_in_bytes` − `memory.usage_in_bytes`
/// from the memory controller's mount, and v1's "no limit" value is no limit.
#[test]
fn cgroup_v1_container() {
    let mut tree = Tree::new("docker");
    tree.remove("sys/fs/cgroup")
        .set(
            "proc/self/cgroup",
            "12:memory:/docker/0123abcd\n11:cpu,cpuacct:/docker/0123abcd\n0::/\n",
        )
        .set("sys/fs/cgroup/memory/memory.limit_in_bytes", "4294967296\n")
        .set("sys/fs/cgroup/memory/memory.usage_in_bytes", "1073741824\n");
    assert_eq!(tree.enumerate().unwrap().env.cgroup, CgroupVersion::V1);
    let host = tree.host();
    let t = linux_tiers(&host);
    assert_eq!(t.tiers, pair(0, &host.mem[0], 3 * GIB, 8 * MIB, &[0]));
    assert_eq!(t.pools, vec![pool(0, 0, 3 * GIB, 8 * MIB)]);

    tree.set("sys/fs/cgroup/memory/memory.limit_in_bytes", "2147483648\n");
    assert_eq!(single(&linux_tiers(&tree.host())), (8 * MIB, GIB));

    // The page-rounded i64::MAX v1 reports for no limit.
    tree.set(
        "sys/fs/cgroup/memory/memory.limit_in_bytes",
        "9223372036854771712\n",
    );
    assert_eq!(
        single(&linux_tiers(&tree.host())),
        (8 * MIB, 45_860_098_048)
    );
}

/// Criterion 4 on a two-domain host: the cgroup limit bounds `host_usable`, so it caps every
/// domain, and a domain whose own estimate is lower keeps its estimate. The fixture's service
/// cgroup has `memory.max` `max`; the test sets it.
#[test]
fn cgroup_on_a_two_domain_host() {
    const SERVICE: &str = "sys/fs/cgroup/system.slice/syntropy.service";
    let mut tree = Tree::new("dual-socket");
    tree.set(&format!("{SERVICE}/memory.max"), &(97 * GIB).to_string())
        .set(&format!("{SERVICE}/memory.current"), &GIB.to_string());
    let host = tree.host();
    // host_usable = 96 GiB, below both nodes; pin_limit = host_usable.
    let t = linux_tiers(&host);
    let mut want = pair(0, &host.mem[0], 96 * GIB, 96 * GIB, &[0]);
    want.extend(pair(2, &host.mem[1], 96 * GIB, 96 * GIB, &[1]));
    assert_eq!(t.tiers, want);
    assert_eq!(
        pools(&t),
        vec![(0, 96 * GIB, 96 * GIB), (1, 96 * GIB, 96 * GIB)]
    );

    // 118 GiB = 126_701_535_232 lies between the two nodes' estimates.
    tree.set(&format!("{SERVICE}/memory.max"), &(119 * GIB).to_string());
    let t = linux_tiers(&tree.host());
    assert_eq!(
        pools(&t),
        vec![(0, DUAL_N0, DUAL_N0), (1, 118 * GIB, 118 * GIB)]
    );
}

/// `own_n` is added back to the domain it is held in, and the host-wide `own` to
/// `host_usable`, so a re-plan does not count a running plan against itself.
#[test]
fn own_memory_is_added_back_per_domain() {
    let host = linux("dual-socket");
    let t = build(HostOs::Linux, &host, &[], &[(0, Bytes(4 * GIB))]).unwrap();
    // Node 0's estimate + 4 GiB, below host_usable + 4 GiB. Node 1 is unchanged.
    let (n0, n1) = (DUAL_N0 + 4 * GIB, DUAL_N1);
    assert_eq!(t.pools, vec![pool(0, 0, n0, n0), pool(1, 2, n1, n1)]);

    // In a cgroup v2 container, `own` is added back to the cgroup headroom as well.
    let host = linux("docker");
    let t = build(HostOs::Linux, &host, &[], &[(0, Bytes(2 * GIB))]).unwrap();
    assert_eq!(single(&t), (8 * MIB, 9 * GIB));
}

/// Criterion 3, WSL2: Linux's rule over the VM's `MemTotal`, with `max_pinned_fraction`
/// 0.75. The fixture's 64 KiB rlimit binds first, so the test lifts it. total 32_803_456 KiB;
/// host_usable 27_661_905_920; 0.75 · total = 25_193_054_208.
#[test]
fn wsl2_pinned_fraction() {
    let mut host = linux("wsl2");
    assert!(host.mem[0].mem.wsl2);
    host.mem[0].mem.memlock_limit = None;
    assert_eq!(
        single(&linux_tiers(&host)),
        (25_193_054_208, 27_661_905_920)
    );
}

/// Criterion 3, Windows: E1's working-set ceiling bounds `pin_limit`. The desktop fixture:
/// host_usable 20 GiB; 0.50 · total = 17_095_000_064; the ceiling is set to 6 GiB.
#[test]
fn windows_lockable() {
    let mut host = windows("desktop");
    host.mem[0].mem.windows_lockable = Some(Bytes(6 * GIB));
    let t = build(HostOs::Windows, &host, &[], &[]).unwrap();
    assert_eq!(single(&t), (6 * GIB, 20 * GIB));
}

/// Criterion 3, macOS: E1 gives the one domain no `HostPinned` slot, so the step builds one
/// `HostPageable` tier and no pool record. total 16 GiB; available (28_015 + 319_984 +
/// 1_476) · 16 KiB pages = 5_725_798_400; headroom 4 GiB.
#[test]
fn macos_has_no_host_pinned() {
    let host = macos("m4-mini");
    let t = build(HostOs::MacOs, &host, &[], &[]).unwrap();
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

#[test]
fn inconsistent_inputs_are_errors() {
    // The ordinals have domain 1's tiers but E1 has no record of its memory.
    let mut host = linux("dual-socket");
    host.mem.pop();
    assert_eq!(
        build(HostOs::Linux, &host, &[], &[]),
        Err(HostTierError::NoMemory { numa: 1 })
    );

    // A Linux domain has a `HostPinned` slot, which the macOS rule set has not.
    assert_eq!(
        build(HostOs::MacOs, &linux("desktop"), &[], &[]),
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
