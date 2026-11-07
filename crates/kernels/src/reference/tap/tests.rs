// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-computed cases for `TapGather`, its tie rule and edge cases, and property tests for
//! `rows_tap = 0`, NaN propagation and agreement with the F64 log-sum-exp.

use proptest::prelude::*;

use super::*;

const NEG: f32 = f32::NEG_INFINITY;
const BLANK: TapRow = TapRow {
    lse: -7.0,
    argmax: 7,
    margin: -7.0,
};

fn tap(
    vocab: usize,
    k: usize,
    logits: &[f32],
    rows: &[u32],
    ids: &[u32],
) -> (Vec<f32>, Vec<TapRow>) {
    let mut lp = vec![-7.0f32; rows.len() * k];
    let mut out = vec![BLANK; rows.len()];
    tap_gather(vocab, k, logits, rows, ids, &mut lp, &mut out).unwrap();
    (lp, out)
}

#[test]
fn tap_gather_by_hand() {
    // Row 0: four equal logits, lse = ln 4. Row 1: [ln 3, 0, 0, 0]: lse = ln 6.
    let l3 = 3f32.ln();
    let logits = [0.0, 0.0, 0.0, 0.0, l3, 0.0, 0.0, 0.0];
    // Tap row 1 then row 0, with two reference IDs each.
    let (lp, out) = tap(4, 2, &logits, &[1, 0], &[0, 3, 2, 2]);
    let ln6 = math::ln(6.0);
    let ln4 = math::ln(4.0);
    assert_eq!(
        out[0].lse,
        l3 + math::ln(reduce::sum(&[
            1.0,
            math::exp(-l3),
            math::exp(-l3),
            math::exp(-l3)
        ]))
    );
    assert!((out[0].lse - ln6).abs() <= 2.0 * f32::EPSILON * ln6);
    assert_eq!(lp[0], l3 - out[0].lse);
    assert_eq!(lp[1], 0.0 - out[0].lse);
    assert_eq!(out[0].argmax, 0);
    // The margin is ln 3 − 0 through the two rounded log-probs.
    assert_eq!(out[0].margin, (l3 - out[0].lse) - (0.0 - out[0].lse));
    // Row 0: a four-way tie, argmax the lowest ID, margin +0.
    assert_eq!(out[1].lse, ln4);
    assert_eq!(lp[2..], [-ln4, -ln4]);
    assert_eq!(out[1].argmax, 0);
    assert_eq!(out[1].margin.to_bits(), 0.0f32.to_bits());
}

#[test]
fn tap_gather_edge_cases() {
    // A single-token vocabulary: lse is the logit, the log-prob 0, the margin +∞.
    let (lp, out) = tap(1, 1, &[2.5], &[0], &[0]);
    assert_eq!(
        (lp[0], out[0].lse, out[0].argmax, out[0].margin),
        (0.0, 2.5, 0, f32::INFINITY)
    );
    // All −∞: lse −∞, log-probs NaN, argmax empty, margin NaN.
    let (lp, out) = tap(2, 1, &[NEG, NEG], &[0], &[1]);
    assert_eq!(out[0].lse, NEG);
    assert!(lp[0].is_nan() && out[0].margin.is_nan());
    assert_eq!(out[0].argmax, EMPTY_ROW);
    // A NaN: everything NaN, argmax the NaN marker.
    let (lp, out) = tap(3, 1, &[0.0, f32::NAN, 1.0], &[0], &[2]);
    assert!(out[0].lse.is_nan() && lp[0].is_nan() && out[0].margin.is_nan());
    assert_eq!(out[0].argmax, NAN_ROW);
    // A +∞: lse NaN, the argmax its ID.
    let (_, out) = tap(3, 0, &[0.0, f32::INFINITY, 1.0], &[0], &[]);
    assert!(out[0].lse.is_nan());
    assert_eq!(out[0].argmax, 1);
    // A second place at −∞ gives a margin of +∞.
    let (_, out) = tap(3, 0, &[1.0, NEG, NEG], &[0], &[]);
    assert_eq!(out[0].margin, f32::INFINITY);
}

#[test]
fn tap_gather_refusals_write_nothing() {
    let logits = [0.0f32; 6];
    let mut lp = [-7.0f32; 2];
    let mut out = [BLANK; 2];
    assert_eq!(
        tap_gather(3, 1, &logits, &[0, 2], &[0, 0], &mut lp, &mut out),
        Err(OpError::RowOutOfRange { id: 2, rows: 2 })
    );
    assert_eq!(
        tap_gather(3, 1, &logits, &[0, 1], &[0, 3], &mut lp, &mut out),
        Err(OpError::RowOutOfRange { id: 3, rows: 3 })
    );
    assert!(matches!(
        tap_gather(3, 2, &logits, &[0, 1], &[0; 4], &mut lp, &mut out),
        Err(OpError::Len { arg: "lp", .. })
    ));
    assert_eq!(lp, [-7.0; 2]);
    assert_eq!(out, [BLANK; 2]);
}

proptest! {
    /// `rows_tap = 0` writes nothing.
    #[test]
    fn rows_tap_zero_writes_nothing(x in prop::collection::vec(-8.0f32..8.0, 1..20)) {
        let mut lp = [-7.0f32; 3];
        let mut out = [BLANK];
        tap_gather(x.len(), 3, &x, &[], &[], &mut lp, &mut out).unwrap();
        prop_assert_eq!(lp, [-7.0; 3]);
        prop_assert_eq!(out, [BLANK]);
    }

    /// The log-sum-exp agrees with F64's to within a few F32 roundings of its terms, and the
    /// log-probs of a row sum, through exp, to about 1.
    #[test]
    fn lse_agrees_with_f64(x in prop::collection::vec(-30.0f32..30.0, 1..300)) {
        let ids: Vec<u32> = (0..x.len() as u32).collect();
        let (lp, out) = tap(x.len(), x.len(), &x, &[0], &ids);
        let m = x.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(f64::from(b)));
        let want = m + x.iter().map(|&v| (f64::from(v) - m).exp()).sum::<f64>().ln();
        let tol = 8.0 * f64::from(f32::EPSILON) * want.abs().max(1.0) + 1e-5;
        prop_assert!((f64::from(out[0].lse) - want).abs() <= tol, "{} vs {want}", out[0].lse);
        let mass: f64 = lp.iter().map(|&v| f64::from(v).exp()).sum();
        prop_assert!((mass - 1.0).abs() < 1e-4, "{mass}");
        // The argmax is the first index of the largest logit.
        let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert_eq!(out[0].argmax as usize, x.iter().position(|&v| v == max).unwrap());
        prop_assert!(out[0].margin >= 0.0);
    }

    /// A NaN anywhere makes every output of that row NaN, and leaves other rows alone.
    #[test]
    fn nan_propagates(x in prop::collection::vec(-8.0f32..8.0, 8), at in 0usize..4) {
        let mut x = x;
        x[at] = f32::NAN;
        let (lp, out) = tap(4, 2, &x, &[0, 1], &[0, 1, 0, 1]);
        prop_assert!(out[0].lse.is_nan() && out[0].margin.is_nan());
        prop_assert!(lp[0].is_nan() && lp[1].is_nan());
        prop_assert_eq!(out[0].argmax, NAN_ROW);
        prop_assert!(!out[1].lse.is_nan() && !lp[2].is_nan());
    }
}
