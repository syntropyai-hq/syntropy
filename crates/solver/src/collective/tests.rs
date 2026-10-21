// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property and unit tests of the collective cost model against RFC-0032 §5.3–§5.5.
//!
//! Each property restates a formula of RFC-0032 §5.3 in F64 and checks the module against it
//! within the nanosecond of rounding. Inputs span decode partials of a few KiB to prefill
//! partials of hundreds of MiB, rates from 100 MB/s to 1 TB/s and α from 0 to 10 ms. They are
//! test inputs, not measurements.

use proptest::prelude::*;
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

    /// The choice rule: `OneShot` exactly when N ≤ 4 and S ≤ `tp.oneshot_max`.
    #[test]
    fn choose_is_oneshot_exactly_when_n_le_4_and_s_le_max(
        n in 0u32..=16,
        s in 0u64..=(4 << 20),
        max in 0u64..=(4 << 20),
    ) {
        let want = if n <= 4 && s <= max { Algorithm::OneShot } else { Algorithm::Ring };
        prop_assert_eq!(choose(n, Bytes(s), Bytes(max)), want);
    }

    /// `all_reduce` runs the algorithm `choose` picks, at that algorithm's time.
    #[test]
    fn all_reduce_prices_the_chosen_algorithm(
        (n, s, a, b) in inputs(),
        shared in any::<bool>(),
    ) {
        let l = link(a, b);
        let fabric = if shared { Fabric::SharedNic } else { Fabric::FullMesh };
        let r = all_reduce(n, Bytes(s), &l, fabric, ONESHOT_MAX_DEFAULT);
        prop_assert_eq!(r.algorithm, choose(n, Bytes(s), ONESHOT_MAX_DEFAULT));
        let time = match r.algorithm {
            Algorithm::OneShot => one_shot(n, Bytes(s), &l, fabric),
            Algorithm::Ring => ring(n, Bytes(s), &l),
        };
        prop_assert_eq!(r.time, time);
    }

    /// Host-staged intra-node transfers cost `S/β_d2h + S/β_h2d` (RFC-0032 §5.5, note of
    /// 2026-10-22), in both algorithms.
    #[test]
    fn host_staged_adds_both_copies(
        (n, s, a, _b) in inputs(),
        d2h in 1e8f64..=1e11,
        h2d in 1e8f64..=1e11,
    ) {
        let l = LinkCost::host_staged(Nanos(a), BytesPerSec(d2h), BytesPerSec(h2d)).unwrap();
        // The module's evaluation order: both copies in seconds, then to nanoseconds.
        let copy = (s as f64 / d2h + s as f64 / h2d) * 1e9;
        prop_assert_eq!(one_shot(n, Bytes(s), &l, Fabric::FullMesh).0, expected(a as f64 + copy));
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
    let max = ONESHOT_MAX_DEFAULT;
    assert_eq!(max, Bytes(1_048_576), "tp.oneshot_max defaults to 1 MiB");
    // N = 4 → 5 at a decode-sized partial.
    assert_eq!(choose(4, Bytes(28_672), max), Algorithm::OneShot);
    assert_eq!(choose(5, Bytes(28_672), max), Algorithm::Ring);
    // S around tp.oneshot_max at N = 4.
    assert_eq!(choose(4, Bytes(max.0 - 1), max), Algorithm::OneShot);
    assert_eq!(choose(4, max, max), Algorithm::OneShot);
    assert_eq!(choose(4, Bytes(max.0 + 1), max), Algorithm::Ring);
    // And at N = 2, the smallest island.
    assert_eq!(choose(2, max, max), Algorithm::OneShot);
    assert_eq!(choose(2, Bytes(max.0 + 1), max), Algorithm::Ring);
    // A smaller configured maximum moves the boundary with it.
    assert_eq!(choose(2, Bytes(4097), Bytes(4096)), Algorithm::Ring);
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
    let per_collective = all_reduce(2, s0, &l, Fabric::FullMesh, ONESHOT_MAX_DEFAULT);
    assert_eq!(per_collective.algorithm, Algorithm::OneShot);
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
fn a_probe_below_its_own_transfer_time_gives_zero_alpha() {
    // 28,672 B at 1 GB/s is 28.672 µs, more than the 10 µs measured.
    let alpha = alpha_from_probe(Nanos(10_000), Bytes(28_672), BytesPerSec(1e9)).unwrap();
    assert_eq!(alpha, Nanos(0));
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
