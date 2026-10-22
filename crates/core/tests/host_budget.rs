// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0020 §5's host budget rule over fixture snapshots, one or more per OS (TIER-E2-T01).
//!
//! Each fixture is the `HostMemInfo` a probe would publish for that host. The expected
//! `usable`, `pin_limit` and `io_fixed` are written out in full, with the arithmetic beside
//! them. The joint per-NUMA rows are tested in `plan_validate.rs`.

use proptest::prelude::*;
use syntropy_core::budget::*;
use syntropy_core::resource::{HostMemInfo, LockAccounting};
use syntropy_core::{Bytes, InstanceId, NodeId, TierId};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn mem(total: u64, available: u64) -> HostMemInfo {
    HostMemInfo {
        total: Bytes(total),
        available: Bytes(available),
        ..HostMemInfo::default()
    }
}

fn inputs(os: HostOs, mem: &HostMemInfo) -> HostBudgetInputs<'_> {
    HostBudgetInputs {
        os,
        mem,
        own: Bytes(0),
        working_set: None,
        user_locked_vm: Bytes(0),
    }
}

fn charged(vmlck: u64, charged: Option<bool>) -> Option<LockAccounting> {
    Some(LockAccounting {
        vmlck_before: Bytes(vmlck),
        registration_charged: charged,
        ..LockAccounting::default()
    })
}

fn budget(headroom: u64, host_usable: u64, pin_limit: u64, io_fixed: Option<u64>) -> HostBudget {
    HostBudget {
        headroom: Bytes(headroom),
        host_usable: Bytes(host_usable),
        pin_limit: Bytes(pin_limit),
        io_fixed: io_fixed.map(Bytes),
    }
}

const DEFAULT: HostBudgetPolicy = HostBudgetPolicy {
    ram_headroom: None,
    max_pinned_fraction: None,
    numa_reserve: None,
};

// ---- Linux, bare metal ------------------------------------------------------------------

/// A 64 GiB desktop with the distribution's default 8 MiB memlock limit, no `CAP_IPC_LOCK` and
/// no H5 result yet, so registration counts as charged.
#[test]
fn linux_desktop_default_memlock() {
    let mut m = mem(64 * GIB, 40 * GIB);
    m.memlock_limit = Some(Bytes(8 * MIB));
    // headroom = max(4 GiB, 0.10 · 68719476736) = 6871947673
    // host_usable = 42949672960 − 6871947673 = 36077725287
    // pin_limit = min(36077725287, 0.85 · 64 GiB = 58411555225, 8 MiB − VmLck 0) = 8388608
    // io_fixed = 8 MiB − locked_vm 0
    assert_eq!(
        host_budget(&inputs(HostOs::Linux, &m), &DEFAULT),
        budget(6_871_947_673, 36_077_725_287, 8_388_608, Some(8_388_608))
    );
}

/// The same desktop after H5 showed provider registration is not charged to the memlock
/// limit: `lockable` is unlimited, but io_uring still is (`io_fixed` per user).
#[test]
fn linux_registration_not_charged() {
    let mut m = mem(64 * GIB, 40 * GIB);
    m.memlock_limit = Some(Bytes(8 * MIB));
    m.lock = charged(MIB, Some(false));
    let mut i = inputs(HostOs::Linux, &m);
    i.user_locked_vm = Bytes(2 * MIB);
    // pin_limit = min(36077725287, 58411555225); io_fixed = 8 MiB − 2 MiB = 6291456
    assert_eq!(
        host_budget(&i, &DEFAULT),
        budget(
            6_871_947_673,
            36_077_725_287,
            36_077_725_287,
            Some(6_291_456)
        )
    );
}

/// Registration charged, with 1 MiB already locked: `lockable = RLIMIT_MEMLOCK − VmLck`.
#[test]
fn linux_registration_charged_subtracts_vmlck() {
    let mut m = mem(64 * GIB, 40 * GIB);
    m.memlock_limit = Some(Bytes(64 * MIB));
    m.lock = charged(MIB, Some(true));
    assert_eq!(
        pin_limit(&inputs(HostOs::Linux, &m), &DEFAULT),
        Bytes(66_060_288)
    );
}

/// A 128 GiB server with `CAP_IPC_LOCK`, the dedicated-mode headroom of 4 GiB and the default
/// fraction: the fraction binds `pin_limit`, and `io_fixed` is unlimited.
#[test]
fn linux_server_cap_ipc_lock() {
    let m = mem(128 * GIB, 120 * GIB);
    let policy = HostBudgetPolicy {
        ram_headroom: Some(Bytes(4 * GIB)),
        ..DEFAULT
    };
    // host_usable = 128849018880 − 4294967296 = 124554051584
    // pin_limit = min(124554051584, 0.85 · 137438953472 = 116823110451)
    assert_eq!(
        host_budget(&inputs(HostOs::Linux, &m), &policy),
        budget(4 * GIB, 124_554_051_584, 116_823_110_451, None)
    );
}

/// `own` is added back, so a re-plan of a running instance does not count its own bytes.
#[test]
fn linux_own_is_not_counted_against_itself() {
    let m = mem(64 * GIB, 20 * GIB);
    let mut i = inputs(HostOs::Linux, &m);
    i.own = Bytes(20 * GIB);
    // host_usable = 21474836480 + 21474836480 − 6871947673
    assert_eq!(host_usable(&i, &DEFAULT), Bytes(36_077_725_287));
}

/// Available memory below the headroom gives 0, never a wrapped value.
#[test]
fn linux_available_below_headroom_is_zero() {
    let m = mem(64 * GIB, 2 * GIB);
    let b = host_budget(&inputs(HostOs::Linux, &m), &DEFAULT);
    assert_eq!((b.host_usable, b.pin_limit), (Bytes(0), Bytes(0)));
}

// ---- Linux, containers ------------------------------------------------------------------

/// cgroup v2: `memory.max` 16 GiB, `memory.current` 2 GiB, so the probe publishes 14 GiB of
/// cgroup headroom, which binds.
#[test]
fn linux_cgroup_v2_memory_max() {
    let mut m = mem(64 * GIB, 40 * GIB);
    m.cgroup_limit = Some(Bytes(16 * GIB - 2 * GIB));
    m.memlock_limit = Some(Bytes(64 * MIB));
    m.lock = charged(MIB, Some(true));
    let mut i = inputs(HostOs::Linux, &m);
    i.user_locked_vm = Bytes(MIB);
    // host_usable = min(36077725287, 15032385536)
    // pin_limit = min(15032385536, 58411555225, 64 MiB − 1 MiB = 66060288)
    // io_fixed = 64 MiB − 1 MiB
    assert_eq!(
        host_budget(&i, &DEFAULT),
        budget(6_871_947_673, 15_032_385_536, 66_060_288, Some(66_060_288))
    );
}

/// In a container, `usable` follows `memory.max`: each limit below the host's available
/// memory is the budget, plus what this node's runners already hold.
#[test]
fn linux_usable_follows_the_cgroup_limit() {
    let current = 2 * GIB;
    for max in [4 * GIB, 8 * GIB, 16 * GIB, 32 * GIB] {
        let mut m = mem(64 * GIB, 40 * GIB);
        m.cgroup_limit = Some(Bytes(max - current));
        let mut i = inputs(HostOs::Linux, &m);
        assert_eq!(host_usable(&i, &DEFAULT), Bytes(max - current), "max {max}");
        i.own = Bytes(GIB);
        assert_eq!(host_usable(&i, &DEFAULT), Bytes(max - current + GIB));
    }
    // A limit above what the host has free leaves the host term binding.
    let mut m = mem(64 * GIB, 40 * GIB);
    m.cgroup_limit = Some(Bytes(60 * GIB));
    assert_eq!(
        host_usable(&inputs(HostOs::Linux, &m), &DEFAULT),
        Bytes(36_077_725_287)
    );
}

/// cgroup v1: `memory.limit_in_bytes` 8 GiB, `memory.usage_in_bytes` 512 MiB.
#[test]
fn linux_cgroup_v1_limit_in_bytes() {
    let mut m = mem(32 * GIB, 30 * GIB);
    m.cgroup_limit = Some(Bytes(8 * GIB - 512 * MIB));
    // headroom = max(4 GiB, 3435973836) = 4294967296
    // host_usable = min(32212254720 − 4294967296, 8053063680)
    // pin_limit = min(8053063680, 0.85 · 34359738368 = 29205777612); unlimited memlock
    assert_eq!(
        host_budget(&inputs(HostOs::Linux, &m), &DEFAULT),
        budget(4 * GIB, 8_053_063_680, 8_053_063_680, None)
    );
}

// ---- WSL2 -------------------------------------------------------------------------------

/// The Linux rule on the VM's `MemTotal` from `.wslconfig`, with the 0.75 default fraction.
#[test]
fn wsl2_uses_the_vm_total_and_its_fraction() {
    let mut m = mem(32 * GIB, 30 * GIB);
    m.wsl2 = true;
    m.memlock_limit = Some(Bytes(8 * MIB));
    m.lock = charged(0, Some(false));
    // host_usable = 32212254720 − 4294967296 = 27917287424
    // pin_limit = min(27917287424, 0.75 · 34359738368 = 25769803776)
    // io_fixed = 8 MiB
    assert_eq!(
        host_budget(&inputs(HostOs::Wsl2, &m), &DEFAULT),
        budget(4 * GIB, 27_917_287_424, 25_769_803_776, Some(8_388_608))
    );
}

// ---- Windows ----------------------------------------------------------------------------

/// A 128 GiB desktop: the measured lock ceiling binds below the 0.50 fraction.
#[test]
fn windows_lockable_measured_binds() {
    let mut m = mem(128 * GIB, 100 * GIB);
    m.lock = Some(LockAccounting {
        lockable_measured: Some(Bytes(60 * GIB)),
        ..LockAccounting::default()
    });
    // headroom = max(4 GiB, 13743895347)
    // host_usable = 107374182400 − 13743895347 = 93630287053
    // pin_limit = min(93630287053, 0.50 · 137438953472 = 68719476736, 64424509440)
    // io_fixed: RFC-0020 §5 has no Windows rule, so 0
    assert_eq!(
        host_budget(&inputs(HostOs::Windows, &m), &DEFAULT),
        budget(13_743_895_347, 93_630_287_053, 64_424_509_440, Some(0))
    );
}

/// Before H5 measures the lock ceiling, the fraction is the cap.
#[test]
fn windows_before_lockable_is_measured() {
    let m = mem(128 * GIB, 100 * GIB);
    assert_eq!(
        pin_limit(&inputs(HostOs::Windows, &m), &DEFAULT),
        Bytes(68_719_476_736)
    );
}

/// A job object limit of 40 GiB with 8 GiB in use binds both `usable` and `pin_limit`.
#[test]
fn windows_job_object_limit() {
    let mut m = mem(128 * GIB, 100 * GIB);
    m.cgroup_limit = Some(Bytes(32 * GIB));
    let mut i = inputs(HostOs::Windows, &m);
    i.own = Bytes(GIB);
    // host_usable = min(100 GiB + 1 GiB − 13743895347, 32 GiB + 1 GiB)
    let b = host_budget(&i, &DEFAULT);
    assert_eq!(b.host_usable, Bytes(35_433_480_192));
    assert_eq!(b.pin_limit, Bytes(35_433_480_192));
}

// ---- macOS ------------------------------------------------------------------------------

fn mac(total: u64, available: u64, working_set: u64, own: u64) -> Bytes {
    let m = mem(total, available);
    let mut i = inputs(HostOs::MacOs, &m);
    i.working_set = Some(Bytes(working_set));
    i.own = Bytes(own);
    host_usable(&i, &DEFAULT)
}

/// `min(working-set limit, RAM − headroom, available + own − headroom)`: each term binds in
/// one fixture.
#[test]
fn macos_takes_the_lowest_of_three_terms() {
    // 64 GiB: headroom 6871947673, RAM − headroom 61847529063.
    // available term = 53687091200 − 6871947673 = 46815143527 binds (working set 48 GiB).
    assert_eq!(mac(64 * GIB, 50 * GIB, 48 * GIB, 0), Bytes(46_815_143_527));
    // available term = 67645734912 − 6871947673 = 60773787239; the working set binds.
    assert_eq!(mac(64 * GIB, 63 * GIB, 48 * GIB, 0), Bytes(51_539_607_552));
    // available term = 64 GiB + 2 GiB own − headroom = 63995012711; RAM − headroom binds.
    assert_eq!(
        mac(64 * GIB, 64 * GIB, 60 * GIB, 2 * GIB),
        Bytes(61_847_529_063)
    );
}

/// No fixed percentage (ADR-0007): a 512 GiB machine may give a model far more than the 60% an
/// earlier fixed cap allowed, and the fraction applies only when the user sets it.
#[test]
fn macos_has_no_fixed_percentage() {
    // headroom = 0.10 · 549755813888 = 54975581388
    // available term = 536870912000 − 54975581388 = 481895330612 (87.7% of RAM)
    let usable = mac(512 * GIB, 500 * GIB, 480 * GIB, 0);
    assert_eq!(usable, Bytes(481_895_330_612));
    assert!(usable.0 > 512 * GIB / 100 * 75);

    let m = mem(512 * GIB, 500 * GIB);
    let mut i = inputs(HostOs::MacOs, &m);
    i.working_set = Some(Bytes(480 * GIB));
    let user = HostBudgetPolicy {
        max_pinned_fraction: Some(0.60),
        ..DEFAULT
    };
    // 0.60 · 549755813888 = 329853488332
    assert_eq!(host_usable(&i, &user), Bytes(329_853_488_332));
}

/// No `HostPinned` tier and no io_uring on macOS.
#[test]
fn macos_has_no_pin_limit() {
    let m = mem(64 * GIB, 50 * GIB);
    let mut i = inputs(HostOs::MacOs, &m);
    i.working_set = Some(Bytes(48 * GIB));
    assert_eq!(
        host_budget(&i, &DEFAULT),
        budget(6_871_947_673, 46_815_143_527, 0, Some(0))
    );
}

// ---- Tier 3 -----------------------------------------------------------------------------

/// Android and iOS: `available − headroom`, `pin_limit = 0`.
#[test]
fn mobile_available_minus_headroom() {
    let m = mem(8 * GIB, 5 * GIB);
    // headroom = max(4 GiB, 858993459) = 4 GiB
    assert_eq!(
        host_budget(&inputs(HostOs::Mobile, &m), &DEFAULT),
        budget(4 * GIB, GIB, 0, Some(0))
    );
}

// ---- Per-NUMA pools ---------------------------------------------------------------------

fn tiers(n: u16) -> [TierId; 2] {
    let node = NodeId([7; 32]);
    [
        TierId {
            node,
            ordinal: 2 * n + 1,
        },
        TierId {
            node,
            ordinal: 2 * n + 2,
        },
    ]
}

fn node_mem(free: u64, active: u64, inactive: u64, sreclaimable: u64) -> NodeMem {
    NodeMem {
        mem_free: Bytes(free),
        active_file: Bytes(active),
        inactive_file: Bytes(inactive),
        sreclaimable: Bytes(sreclaimable),
    }
}

/// A dual-socket host: node 0's estimate binds its pool, node 1's exceeds the host budget, so
/// the host budget binds there.
#[test]
fn dual_socket_pools() {
    let m = mem(256 * GIB, 200 * GIB);
    let b = host_budget(&inputs(HostOs::Linux, &m), &DEFAULT);
    // headroom = 27487790694; host_usable = 214748364800 − 27487790694 = 187260574106
    // pin_limit = min(187260574106, 0.85 · 274877906944 = 233646220902)
    assert_eq!(
        b,
        budget(27_487_790_694, 187_260_574_106, 187_260_574_106, None)
    );

    let reserve = DEFAULT.numa_reserve();
    // node 0: 20 + 30 + 40 + 2 GiB − 1 GiB = 91 GiB
    let n0 = numa_available(&node_mem(20 * GIB, 30 * GIB, 40 * GIB, 2 * GIB), reserve);
    assert_eq!(n0, Some(Bytes(91 * GIB)));
    let p0 = host_pool(0, tiers(0), &b, n0, Bytes(0));
    assert_eq!(
        (p0.usable, p0.pin_limit),
        (Bytes(91 * GIB), Bytes(91 * GIB))
    );
    // node 0 with 4 GiB already held: 95 GiB
    let p0 = host_pool(0, tiers(0), &b, n0, Bytes(4 * GIB));
    assert_eq!(p0.usable, Bytes(95 * GIB));

    // node 1: 201 GiB, above host_usable
    let n1 = numa_available(&node_mem(100 * GIB, 50 * GIB, 50 * GIB, 2 * GIB), reserve);
    let p1 = host_pool(1, tiers(1), &b, n1, Bytes(0));
    assert_eq!(p1.numa, 1);
    assert_eq!(p1.tiers, tiers(1));
    assert_eq!(p1.usable, Bytes(187_260_574_106));
}

/// `pin_limit_n = min(pin_limit, host_usable_n)`.
#[test]
fn pool_pin_limit_is_capped_by_both() {
    let mut m = mem(64 * GIB, 40 * GIB);
    m.memlock_limit = Some(Bytes(8 * MIB));
    let b = host_budget(&inputs(HostOs::Linux, &m), &DEFAULT);
    let p = host_pool(0, tiers(0), &b, Some(Bytes(MIB)), Bytes(0));
    assert_eq!((p.usable, p.pin_limit), (Bytes(MIB), Bytes(MIB)));
    let p = host_pool(0, tiers(0), &b, None, Bytes(0));
    assert_eq!(
        (p.usable, p.pin_limit),
        (Bytes(36_077_725_287), Bytes(8 * MIB))
    );
}

/// The reserve is subtracted once and floors at 0; an overflowing sum gives no estimate.
#[test]
fn numa_available_edges() {
    assert_eq!(
        numa_available(&node_mem(MIB, 0, 0, 0), Bytes(GIB)),
        Some(Bytes(0))
    );
    assert_eq!(numa_available(&node_mem(u64::MAX, 1, 0, 0), Bytes(0)), None);
}

// ---- io_fixed among runners -------------------------------------------------------------

/// Two runners share the user's `io_fixed`: the second gets only what the first leaves, and a
/// release returns it.
#[test]
fn two_runners_share_io_fixed() {
    let (r1, r2) = (InstanceId(1), InstanceId(2));
    let mut ledger = IoFixedLedger::new(Some(Bytes(8 * MIB)));
    assert_eq!(ledger.grant(r1, Bytes(6 * MIB)), Bytes(6 * MIB));
    assert_eq!(ledger.grant(r2, Bytes(6 * MIB)), Bytes(2 * MIB));
    assert_eq!(ledger.total_granted(), Bytes(8 * MIB));
    ledger.release(r1);
    assert_eq!(ledger.granted(r1), Bytes(0));
    assert_eq!(ledger.grant(r2, Bytes(6 * MIB)), Bytes(6 * MIB));
    // A smaller request shrinks a grant.
    assert_eq!(ledger.grant(r2, Bytes(MIB)), Bytes(MIB));
    assert_eq!(ledger.total_granted(), Bytes(MIB));
}

#[derive(Clone, Debug)]
enum Op {
    Grant(u64, u64),
    Release(u64),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (1u64..=2, 0u64..=32 * MIB).prop_map(|(r, w)| Op::Grant(r, w)),
        (1u64..=2).prop_map(Op::Release),
    ]
}

proptest! {
    /// Whatever two fake runners ask for, in any order, they never hold more than the user's
    /// `io_fixed` between them, each gets at most what it asked for, and a grant is all that
    /// the other runner leaves.
    #[test]
    fn two_runners_never_exceed_io_fixed(
        limit in 0u64..=16 * MIB,
        ops in proptest::collection::vec(op(), 0..64),
    ) {
        let mut ledger = IoFixedLedger::new(Some(Bytes(limit)));
        for op in ops {
            match op {
                Op::Grant(r, want) => {
                    let other = InstanceId(3 - r);
                    let left = limit - ledger.granted(other).0;
                    let got = ledger.grant(InstanceId(r), Bytes(want));
                    prop_assert_eq!(got, Bytes(want.min(left)));
                }
                Op::Release(r) => ledger.release(InstanceId(r)),
            }
            let sum = ledger.granted(InstanceId(1)).0 + ledger.granted(InstanceId(2)).0;
            prop_assert_eq!(Bytes(sum), ledger.total_granted());
            prop_assert!(sum <= limit);
        }
    }
}

/// An unlimited `io_fixed` grants every request.
#[test]
fn unlimited_io_fixed_grants_in_full() {
    let mut ledger = IoFixedLedger::new(None);
    assert_eq!(ledger.limit(), None);
    assert_eq!(
        ledger.grant(InstanceId(1), Bytes(u64::MAX)),
        Bytes(u64::MAX)
    );
    assert_eq!(ledger.grant(InstanceId(2), Bytes(GIB)), Bytes(GIB));
}

/// The ledger divides what `io_fixed` reports for the user.
#[test]
fn ledger_over_the_rule() {
    let mut m = mem(64 * GIB, 40 * GIB);
    m.memlock_limit = Some(Bytes(64 * MIB));
    let mut i = inputs(HostOs::Linux, &m);
    i.user_locked_vm = Bytes(16 * MIB);
    let mut ledger = IoFixedLedger::new(io_fixed(&i));
    assert_eq!(ledger.grant(InstanceId(1), Bytes(GIB)), Bytes(48 * MIB));
    assert_eq!(ledger.grant(InstanceId(2), Bytes(GIB)), Bytes(0));
}
