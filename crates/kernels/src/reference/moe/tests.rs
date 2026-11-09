// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-computed cases for `MoeRoute`, `MoeGrouped` and `MoeCombine`, the tie rule and the
//! combine order, and property tests for `rows_active = 0`, NaN propagation and the
//! independence of the combine from slot order.

use proptest::prelude::*;

use super::*;

const SENTINEL: f32 = -12345.0;

fn route(
    k: usize,
    renormalise: bool,
    n: usize,
    logits: &[f32],
    rows: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut ids = vec![99u32; rows * k];
    let mut w = vec![SENTINEL; rows * k];
    moe_route(
        RouteNorm::Softmax,
        k,
        renormalise,
        n,
        logits,
        rows,
        &mut ids,
        &mut w,
    )
    .unwrap();
    (ids, w)
}

#[test]
fn moe_route_by_hand() {
    // Logits [0, ln 3, 0, ln 2]: masses 1, 3, 1, 2, Z = 7.
    let (l3, l2) = (3f32.ln(), 2f32.ln());
    let logits = [0.0, l3, 0.0, l2];
    let (ids, w) = route(3, false, 4, &logits, 1);
    // Rank: expert 1, expert 3, then the tie between 0 and 2 goes to 0.
    assert_eq!(ids, [1, 3, 0]);
    let z = reduce::sum_by(4, |e| math::exp(logits[e] - l3));
    assert_eq!(w, [1.0 / z, math::exp(l2 - l3) / z, math::exp(-l3) / z]);
    assert!((w[0] - 3.0 / 7.0).abs() < 1e-6);
    // Renormalised over the three: divided by their F32Fixed sum.
    let (ids2, w2) = route(3, true, 4, &logits, 1);
    assert_eq!(ids2, ids);
    let s = reduce::sum(&w);
    assert_eq!(w2, [w[0] / s, w[1] / s, w[2] / s]);
    assert!((w2[0] - 0.5).abs() < 1e-6);
    // k = 0 writes nothing; k = n selects every expert.
    let (ids, _) = route(0, true, 4, &logits, 1);
    assert!(ids.is_empty());
    let (ids, _) = route(4, false, 4, &logits, 1);
    assert_eq!(ids, [1, 3, 0, 2]);
}

#[test]
fn moe_route_edge_cases_and_refusals() {
    // A NaN logit: every weight NaN, the selection by lowest index.
    let (ids, w) = route(2, true, 4, &[1.0, f32::NAN, 3.0, 2.0], 1);
    assert_eq!(ids, [0, 1]);
    assert!(w.iter().all(|v| v.is_nan()));
    // All −∞ likewise.
    let neg = f32::NEG_INFINITY;
    let (ids, w) = route(2, false, 3, &[neg, neg, neg], 1);
    assert_eq!(ids, [0, 1]);
    assert!(w.iter().all(|v| v.is_nan()));
    // A −∞ expert among finite ones gets weight 0 and ranks last.
    let (ids, w) = route(3, false, 3, &[neg, 0.0, 0.0], 1);
    assert_eq!(ids, [1, 2, 0]);
    assert_eq!(w[2], 0.0);
    let mut i = [0u32; 4];
    let mut v = [0.0f32; 4];
    assert!(matches!(
        moe_route(
            RouteNorm::Sigmoid,
            1,
            false,
            4,
            &[0.0; 4],
            1,
            &mut i,
            &mut v
        ),
        Err(OpError::Unsupported { .. })
    ));
    assert!(matches!(
        moe_route(
            RouteNorm::Softmax,
            5,
            false,
            4,
            &[0.0; 4],
            1,
            &mut i,
            &mut v
        ),
        Err(OpError::Len { arg: "k", .. })
    ));
    assert!(matches!(
        moe_route(
            RouteNorm::Softmax,
            2,
            false,
            4,
            &[0.0; 4],
            2,
            &mut i,
            &mut v
        ),
        Err(OpError::Len { arg: "logits", .. })
    ));
}

/// Three experts of `hidden = 2`, `inter = 1`: expert `e` has gate `[1, 0]`, up `[e + 1, 0]`
/// and down `[1, −1]ᵀ`, so on `x = [a, b]` it gives `silu(a) · (e + 1)·a` and its negative.
fn three_experts() -> Vec<[f32; 6]> {
    (0..3)
        .map(|e| [1.0, 0.0, e as f32 + 1.0, 0.0, 1.0, -1.0])
        .collect()
}

fn experts(store: &[[f32; 6]]) -> Vec<Expert<'_>> {
    store
        .iter()
        .map(|w| Expert {
            gate: Matrix::new(&w[0..2], 1, 2).unwrap(),
            up: Matrix::new(&w[2..4], 1, 2).unwrap(),
            down: Matrix::new(&w[4..6], 2, 1).unwrap(),
        })
        .collect()
}

#[test]
fn moe_grouped_by_hand() {
    let store = three_experts();
    let ex = experts(&store);
    let x = [2.0, 5.0, -1.0, 0.0];
    let ids = [2, 0, 1, 1];
    let mut y = vec![SENTINEL; 2 * 2 * 2 + 2];
    moe_grouped(GatedAct::Silu, &ex, 2, &ids, &x, 2, 2, &mut y).unwrap();
    let s2 = math::silu(2.0);
    let sm = math::silu(-1.0);
    assert_eq!(
        y,
        [
            s2 * 6.0,
            -(s2 * 6.0),
            s2 * 2.0,
            -(s2 * 2.0),
            sm * -2.0,
            -(sm * -2.0),
            sm * -2.0,
            -(sm * -2.0),
            SENTINEL,
            SENTINEL
        ]
    );
    // It is `gate_up_act` then `gemv`, exactly.
    let mut h = [0.0f32];
    gate_up_act(GatedAct::Silu, ex[2].gate, ex[2].up, &x[..2], 1, &mut h).unwrap();
    let mut want = [0.0f32; 2];
    gemv(ex[2].down, &h, 1, &mut want).unwrap();
    assert_eq!(y[..2], want);
}

#[test]
fn moe_grouped_refusals_write_nothing() {
    let store = three_experts();
    let ex = experts(&store);
    let mut y = [SENTINEL; 4];
    assert_eq!(
        moe_grouped(GatedAct::Silu, &ex, 1, &[3, 0], &[0.0; 4], 2, 2, &mut y),
        Err(OpError::RowOutOfRange { id: 3, rows: 3 })
    );
    assert_eq!(
        moe_grouped(
            GatedAct::Silu,
            &ex,
            1,
            &[0, 0],
            &[0.0; 6],
            3,
            2,
            &mut [SENTINEL; 6]
        ),
        Err(OpError::ShapeMismatch { op: "MoeGrouped" })
    );
    assert!(matches!(
        moe_grouped(GatedAct::Gelu, &ex, 1, &[0, 0], &[0.0; 4], 2, 2, &mut y),
        Err(OpError::Unsupported { .. })
    ));
    assert!(matches!(
        moe_grouped(
            GatedAct::Silu,
            &ex,
            2,
            &[0, 0, 0, 0],
            &[0.0; 4],
            2,
            2,
            &mut y
        ),
        Err(OpError::Len { arg: "y", .. })
    ));
    assert_eq!(y, [SENTINEL; 4]);
}

#[test]
fn moe_combine_by_hand() {
    // One row, hidden 1, k = 3 in slot order experts 5, 1, 3, plus one shared expert.
    let ids = [5, 1, 3];
    let w = [0.5, 0.25, 2.0];
    let routed = [8.0, 4.0, 1.0];
    let shared = [100.0];
    let mut y = [SENTINEL; 2];
    moe_combine(3, &ids, &w, &routed, 1, &shared, 1, 1, &mut y).unwrap();
    // Ascending expert index: 1 (0.25·4), 3 (2·1), 5 (0.5·8), then the shared expert.
    assert_eq!(y, [((1.0 + 2.0) + 4.0) + 100.0, SENTINEL]);
    // The order is visible in rounding: terms 1, 2^24 and −2^24 in ascending index give 0 when
    // the large ones come first, and 1 when added in slot order 1 first. Ascending index puts
    // expert 0 (2^24) first.
    let big = 16_777_216.0f32;
    let ids = [2, 0, 1];
    let routed = [1.0, big, -big];
    let mut y = [SENTINEL];
    moe_combine(3, &ids, &[1.0; 3], &routed, 0, &[], 1, 1, &mut y).unwrap();
    assert_eq!(y, [(big + -big) + 1.0]);
    assert_eq!(y, [1.0]);
    // With 2^24 + 1 rounding: experts 0 (2^24), 1 (1), 2 (−2^24) give (2^24 + 1) − 2^24 = 0.
    let ids = [1, 2, 0];
    let routed = [1.0, -big, big];
    moe_combine(3, &ids, &[1.0; 3], &routed, 0, &[], 1, 1, &mut y).unwrap();
    assert_eq!(y, [0.0]);
    // k = 0 and no shared experts: +0.
    moe_combine(0, &[], &[], &[], 0, &[], 1, 1, &mut y).unwrap();
    assert_eq!(y[0].to_bits(), 0.0f32.to_bits());
}

/// Rows of router logits with ties from a small set.
fn logits(n: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec((-4i32..4).prop_map(|v| v as f32 * 0.5), n)
}

proptest! {
    /// `rows_active = 0` writes nothing in any of the three ops.
    #[test]
    fn rows_active_zero_writes_nothing(l in logits(8)) {
        let mut ids = [99u32; 2];
        let mut w = [SENTINEL; 2];
        moe_route(RouteNorm::Softmax, 2, true, 8, &l, 0, &mut ids, &mut w).unwrap();
        prop_assert_eq!((ids, w), ([99, 99], [SENTINEL; 2]));
        let store = three_experts();
        let ex = experts(&store);
        let mut y = [SENTINEL; 4];
        moe_grouped(GatedAct::Silu, &ex, 2, &[0, 1], &l[..2], 2, 0, &mut y).unwrap();
        prop_assert_eq!(y, [SENTINEL; 4]);
        moe_combine(2, &[0, 1], &[1.0; 2], &[1.0; 4], 0, &[], 2, 0, &mut y).unwrap();
        prop_assert_eq!(y, [SENTINEL; 4]);
    }

    /// The selection is the `k` largest probabilities, ties to the lower index, in rank order,
    /// and distinct.
    #[test]
    fn route_ties_go_to_the_lower_index(l in logits(10), k in 0usize..=10) {
        let (ids, w) = route(k, false, 10, &l, 1);
        let mut want: Vec<usize> = (0..10).collect();
        want.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
        prop_assert_eq!(ids.iter().map(|&i| i as usize).collect::<Vec<_>>(), want[..k].to_vec());
        prop_assert!(w.windows(2).all(|p| p[0] >= p[1]));
    }

    /// The combine does not depend on the slot order the router wrote.
    #[test]
    fn combine_ignores_slot_order(
        perm_seed in any::<u64>(),
        w in prop::collection::vec(-2.0f32..2.0, 6),
        out in prop::collection::vec(-1e6f32..1e6, 12),
    ) {
        let ids: Vec<u32> = vec![4, 0, 7, 2, 9, 5];
        let mut order: Vec<usize> = (0..6).collect();
        let mut s = perm_seed;
        for i in (1..6).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            order.swap(i, (s >> 33) as usize % (i + 1));
        }
        let pick = |v: &[f32], width: usize| -> Vec<f32> {
            order.iter().flat_map(|&j| v[j * width..(j + 1) * width].to_vec()).collect()
        };
        let (pi, pw, po) = (
            order.iter().map(|&j| ids[j]).collect::<Vec<_>>(),
            pick(&w, 1),
            pick(&out, 2),
        );
        let (mut a, mut b) = ([0.0f32; 2], [0.0f32; 2]);
        moe_combine(6, &ids, &w, &out, 0, &[], 2, 1, &mut a).unwrap();
        moe_combine(6, &pi, &pw, &po, 0, &[], 2, 1, &mut b).unwrap();
        prop_assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
    }

    /// NaN in a router logit reaches every weight; NaN in an expert output or a weight reaches
    /// only its own row and column of the combine.
    #[test]
    fn nan_propagates(l in logits(6), at in 0usize..6, col in 0usize..2) {
        let mut l = l;
        l[at] = f32::NAN;
        let (_, w) = route(2, true, 6, &l, 1);
        prop_assert!(w.iter().all(|v| v.is_nan()));
        let mut routed = [1.0f32; 2 * 2 * 2];
        routed[col] = f32::NAN;
        let mut y = [0.0f32; 4];
        moe_combine(2, &[0, 1, 0, 1], &[0.5; 4], &routed, 0, &[], 2, 2, &mut y).unwrap();
        prop_assert!(y[col].is_nan());
        prop_assert!(!y[1 - col].is_nan() && !y[2].is_nan() && !y[3].is_nan());
        let store = three_experts();
        let ex = experts(&store);
        let mut g = [0.0f32; 4];
        moe_grouped(GatedAct::Silu, &ex, 1, &[0, 1], &[f32::NAN, 1.0, 2.0, 1.0], 2, 2, &mut g).unwrap();
        prop_assert!(g[0].is_nan() && g[1].is_nan() && !g[2].is_nan());
    }
}
