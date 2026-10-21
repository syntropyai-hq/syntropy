// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property and unit tests of the collective cost model against RFC-0032 §5.3–§5.5.
//!
//! Each property restates a formula of RFC-0032 §5.3 in F64 and checks the module against it
//! within the nanosecond of rounding. Inputs span decode partials of a few KiB to prefill
//! partials of hundreds of MiB, rates from 100 MB/s to 1 TB/s and α from 0 to 10 ms. They are
//! test inputs, not measurements.

use proptest::prelude::*;
use syntropy_core::plan::stage::CollectiveAlgo;
use syntropy_core::{Bytes, BytesPerSec, Nanos};

use super::*;

/// The formula's value in nanoseconds, rounded as the module rounds.
fn expected(nanos: f64) -> u64 {
    nanos.round() as u64
}

fn link(alpha: u64, beta: f64) -> LinkCost {
    LinkCost::network(Nanos(alpha), BytesPerSec(beta)).unwrap()
}

prop_compose! {
    fn inputs()(
        members in 2u32..=64,
        partial in 1u64..=(512 << 20),
        alpha in 0u64..=10_000_000,
        beta in 1e8f64..=1e12,
    ) -> (u32, u64, u64, f64) {
        (members, partial, alpha, beta)
    }
}

proptest! {
    /// `OneShot` on a full mesh: `α + S/β`.
    #[test]
    fn one_shot_on_a_full_mesh_is_alpha_plus_s_over_beta((n, s, a, b) in inputs()) {
        let t = one_shot(n, Bytes(s), &link(a, b), Fabric::FullMesh);
        prop_assert_eq!(t.0, expected(a as f64 + s as f64 / b * 1e9));
    }

    /// `OneShot` with the sends sharing one NIC: `α + (N−1)·S/β`.
    #[test]
    fn one_shot_on_a_shared_nic_sends_n_minus_one_partials((n, s, a, b) in inputs()) {
        let t = one_shot(n, Bytes(s), &link(a, b), Fabric::SharedNic);
        prop_assert_eq!(t.0, expected(a as f64 + f64::from(n - 1) * (s as f64 / b * 1e9)));
    }

    /// `Ring`: `2(N−1)·α + 2(N−1)/N · S/β`.
    #[test]
    fn ring_is_two_n_minus_one_steps((n, s, a, b) in inputs()) {
        let t = ring(n, Bytes(s), &link(a, b));
        let n_f = f64::from(n);
        let steps = 2.0 * (n_f - 1.0);
        prop_assert_eq!(t.0, expected(steps * a as f64 + steps / n_f * (s as f64 / b * 1e9)));
    }

    /// The choice rule: `OneShot` exactly when N and S are within the limits.
    #[test]
    fn choose_is_oneshot_exactly_within_the_limits(
        n in 0u32..=16,
        s in 0u64..=(4 << 20),
        members in 0u32..=8,
        max in 0u64..=(4 << 20),
    ) {
        let limits = OneShotLimits { members, partial: Bytes(max) };
        let want = if n <= members && s <= max { CollectiveAlgo::OneShot } else { CollectiveAlgo::Ring };
        prop_assert_eq!(choose(n, Bytes(s), limits), want);
    }

    /// `all_reduce` runs the algorithm `choose` picks, at that algorithm's time.
    #[test]
    fn all_reduce_prices_the_chosen_algorithm(
        (n, s, a, b) in inputs(),
        shared in any::<bool>(),
    ) {
        let l = link(a, b);
        let fabric = if shared { Fabric::SharedNic } else { Fabric::FullMesh };
        let r = all_reduce(n, Bytes(s), &l, fabric, OneShotLimits::default());
        prop_assert_eq!(r.algorithm, choose(n, Bytes(s), OneShotLimits::default()));
        let time = match r.algorithm {
            CollectiveAlgo::OneShot => one_shot(n, Bytes(s), &l, fabric),
            _ => ring(n, Bytes(s), &l),
        };
        prop_assert_eq!(r.time, time);
    }

    /// Host-staged intra-node all-reduces (RFC-0032 §5.5, note of 2026-10-22): `OneShot` is one
    /// copy to the host and N−1 reads back, `α + S/β_d2h + (N−1)·S/β_h2d`, whatever the fabric;
    /// each `Ring` step copies its slice both ways.
    #[test]
    fn host_staged_copies_once_out_and_reads_n_minus_one_back(
        (n, s, a, _b) in inputs(),
        d2h in 1e8f64..=1e11,
        h2d in 1e8f64..=1e11,
        shared in any::<bool>(),
    ) {
        let l = LinkCost::host_staged(Nanos(a), BytesPerSec(d2h), BytesPerSec(h2d)).unwrap();
        let fabric = if shared { Fabric::SharedNic } else { Fabric::FullMesh };
        let bytes = s as f64;
        // The module's evaluation order: seconds, then nanoseconds.
        let one = (bytes / d2h + f64::from(n - 1) * (bytes / h2d)) * 1e9;
        prop_assert_eq!(one_shot(n, Bytes(s), &l, fabric).0, expected(a as f64 + one));
        let copy = (bytes / d2h + bytes / h2d) * 1e9;
        let n_f = f64::from(n);
        let steps = 2.0 * (n_f - 1.0);
        prop_assert_eq!(ring(n, Bytes(s), &l).0, expected(steps * a as f64 + steps / n_f * copy));
    }

    /// A verified peer-to-peer pair uses the same formulas as a network link (RFC-0032 §5.5).
    #[test]
    fn peer_to_peer_is_priced_like_a_direct_link((n, s, a, b) in inputs()) {
        let p2p = LinkCost::peer_to_peer(Nanos(a), BytesPerSec(b)).unwrap();
        let net = link(a, b);
        prop_assert_eq!(ring(n, Bytes(s), &p2p), ring(n, Bytes(s), &net));
        prop_assert_eq!(
            one_shot(n, Bytes(s), &p2p, Fabric::FullMesh),
            one_shot(n, Bytes(s), &net, Fabric::FullMesh)
        );
    }

    /// α derived from a probe gives back the probe's `α_eff` at the probe partial on a full mesh
    /// (RFC-0032 §5.3, note of 2026-10-22): the size term is counted once.
    #[test]
    fn alpha_from_probe_round_trips_at_the_probe_partial(
        h_model in 1024u64..=16_384,
        b in 1e8f64..=1e12,
        extra in 0u64..=1_000_000,
    ) {
        let s0 = Bytes(h_model * 4);
        let size_term = s0.0 as f64 / b * 1e9;
        let alpha_eff = Nanos(size_term.ceil() as u64 + extra);
        let alpha = alpha_from_probe(alpha_eff, s0, BytesPerSec(b)).unwrap();
        let t = one_shot(2, s0, &link(alpha.0, b), Fabric::FullMesh);
        // Two roundings, each within half a nanosecond.
        prop_assert!(t.0.abs_diff(alpha_eff.0) <= 1, "{} vs {}", t.0, alpha_eff.0);
    }

    /// More bytes, or more members, never make an all-reduce cheaper.
    #[test]
    fn time_grows_with_bytes_and_members((n, s, a, b) in inputs(), more in 1u64..=(1 << 20)) {
        let l = link(a, b);
        prop_assert!(ring(n, Bytes(s + more), &l) >= ring(n, Bytes(s), &l));
        prop_assert!(ring(n + 1, Bytes(s), &l) >= ring(n, Bytes(s), &l));
        for fabric in [Fabric::FullMesh, Fabric::SharedNic] {
            prop_assert!(one_shot(n, Bytes(s + more), &l, fabric) >= one_shot(n, Bytes(s), &l, fabric));
        }
        prop_assert!(
            one_shot(n + 1, Bytes(s), &l, Fabric::SharedNic) >= one_shot(n, Bytes(s), &l, Fabric::SharedNic)
        );
    }

    /// Expert parallelism sends a dispatch and a combine per remote owner touched.
    #[test]
    fn ep_sends_two_messages_per_owner_touched(owners in 0u32..=1024) {
        prop_assert_eq!(ep_messages_per_layer(owners), 2 * owners);
    }
}

#[test]
fn the_choice_boundary_is_four_members_and_one_mib() {
    use CollectiveAlgo::{OneShot, Ring};
    let limits = OneShotLimits::default();
    let max = limits.partial;
    assert_eq!(max, Bytes(1_048_576), "tp.oneshot_max defaults to 1 MiB");
    assert_eq!(limits.members, 4);
    // N = 4 → 5 at a decode-sized partial.
    assert_eq!(choose(4, Bytes(28_672), limits), OneShot);
    assert_eq!(choose(5, Bytes(28_672), limits), Ring);
    // S around tp.oneshot_max at N = 4.
    assert_eq!(choose(4, Bytes(max.0 - 1), limits), OneShot);
    assert_eq!(choose(4, max, limits), OneShot);
    assert_eq!(choose(4, Bytes(max.0 + 1), limits), Ring);
    // And at N = 2, the smallest island.
    assert_eq!(choose(2, max, limits), OneShot);
    assert_eq!(choose(2, Bytes(max.0 + 1), limits), Ring);
    // A smaller configured maximum moves the boundary with it.
    let small = OneShotLimits {
        partial: Bytes(4096),
        ..limits
    };
    assert_eq!(choose(2, Bytes(4097), small), Ring);
}

/// RFC-0032 open question 3's interim rule for 3–4 node Mac full meshes: plan on `Ring` for
/// N ≥ 3 until DIST-E13-T10 measures. A member limit of 2 expresses it.
#[test]
fn a_member_limit_of_two_plans_the_ring_from_three_members() {
    let interim = OneShotLimits {
        members: 2,
        ..OneShotLimits::default()
    };
    let s0 = Bytes(7168 * 4);
    assert_eq!(choose(3, s0, interim), CollectiveAlgo::Ring);
    assert_eq!(choose(4, s0, interim), CollectiveAlgo::Ring);
    assert_eq!(choose(2, s0, interim), CollectiveAlgo::OneShot);
    assert_eq!(
        choose(3, s0, OneShotLimits::default()),
        CollectiveAlgo::OneShot
    );
}

/// The review's worked case for host-staged `OneShot` at N = 4 (illustrative inputs, not
/// measurements): 512 KiB partials, α = 10 µs and both copies at 25 GB/s. S/β is 20,971.52 ns, so
/// one copy out and three reads back give 10,000 + 4 × 20,971.52 ≈ 93,886 ns on either fabric.
#[test]
fn host_staged_one_shot_at_four_members() {
    let l = LinkCost::host_staged(Nanos(10_000), BytesPerSec(25e9), BytesPerSec(25e9)).unwrap();
    let s = Bytes(512 << 10);
    assert_eq!(one_shot(4, s, &l, Fabric::FullMesh), Nanos(93_886));
    assert_eq!(one_shot(4, s, &l, Fabric::SharedNic), Nanos(93_886));
    // N = 2 is one copy each way: 10,000 + 2 × 20,971.52.
    assert_eq!(one_shot(2, s, &l, Fabric::FullMesh), Nanos(51_943));
    // The ring's six steps each copy a quarter of S both ways: 6α + 6/4 × 2 × S/β.
    assert_eq!(ring(4, s, &l), Nanos(122_915));
}

/// ADR-0310's corrected example, revision of 2026-10-19: DeepSeek-V3 at TP = 2 with
/// α_eff = 40 µs costs 122 × 40 µs ≈ 4.9 ms per token. The probe's partial is
/// `h_model × 4 B` = 7168 × 4 B; the 25 GB/s rate is a test input. With α derived from the probe,
/// each decode all-reduce costs exactly α_eff.
#[test]
fn adr_0310s_example_is_reproduced() {
    let beta = BytesPerSec(25e9);
    let s0 = Bytes(7168 * 4);
    let alpha = alpha_from_probe(Nanos(40_000), s0, beta).unwrap();
    let l = LinkCost::network(alpha, beta).unwrap();
    let per_collective = all_reduce(2, s0, &l, Fabric::FullMesh, OneShotLimits::default());
    assert_eq!(per_collective.algorithm, CollectiveAlgo::OneShot);
    assert_eq!(per_collective.time, Nanos(40_000));
    assert_eq!(122 * per_collective.time.0, 4_880_000);
}

#[test]
fn ring_at_two_members_is_two_alphas_and_one_transfer() {
    // 2(N−1) = 2 steps of α, and 2(N−1)/N = 1 transfer of S at N = 2.
    let l = link(1_000, 1e9);
    assert_eq!(ring(2, Bytes(1_000_000), &l), Nanos(2_000 + 1_000_000));
}

#[test]
fn fewer_than_two_members_cost_nothing() {
    let l = link(5_000, 1e9);
    for n in [0, 1] {
        assert_eq!(one_shot(n, Bytes(1 << 20), &l, Fabric::SharedNic), Nanos(0));
        assert_eq!(ring(n, Bytes(1 << 20), &l), Nanos(0));
    }
}

#[test]
fn a_probe_below_its_own_transfer_time_is_refused() {
    // 28,672 B at 1 GB/s is 28.672 µs, more than the 10 µs measured.
    let error = alpha_from_probe(Nanos(10_000), Bytes(28_672), BytesPerSec(1e9)).unwrap_err();
    assert_eq!(
        error,
        CostError::ProbeBelowTransfer {
            alpha_eff: Nanos(10_000),
            transfer: Nanos(28_672),
        }
    );
    assert_eq!(
        error.to_string(),
        "the probe measured 10000 ns, less than its partial's 28672 ns transfer time at this \
         link's rate; the rate or the probe partial is wrong for the link"
    );
    // Exactly the transfer time is α = 0.
    assert_eq!(
        alpha_from_probe(Nanos(28_672), Bytes(28_672), BytesPerSec(1e9)),
        Ok(Nanos(0))
    );
}

#[test]
fn rates_must_be_finite_and_above_zero() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            LinkCost::network(Nanos(1), BytesPerSec(bad)),
            Err(CostError::InvalidRate { which: "beta", .. })
        ));
        assert!(matches!(
            LinkCost::host_staged(Nanos(1), BytesPerSec(1e9), BytesPerSec(bad)),
            Err(CostError::InvalidRate { which: "h2d", .. })
        ));
        assert!(matches!(
            LinkCost::host_staged(Nanos(1), BytesPerSec(bad), BytesPerSec(1e9)),
            Err(CostError::InvalidRate { which: "d2h", .. })
        ));
        assert!(alpha_from_probe(Nanos(1), Bytes(1), BytesPerSec(bad)).is_err());
    }
    assert_eq!(
        CostError::InvalidRate {
            which: "beta",
            value: 0.0
        }
        .to_string(),
        "beta must be a finite rate above zero, not 0 B/s"
    );
}
