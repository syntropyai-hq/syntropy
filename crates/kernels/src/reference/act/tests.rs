// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-computed cases for `Act`, and property tests for `rows_active = 0` and NaN propagation.

use proptest::prelude::*;

use super::*;

const SENTINEL: f32 = -12345.0;

#[test]
fn silu_by_hand() {
    // silu(0) = 0; silu(-0) = -0; silu(x) = x / (1 + e^-x).
    let x = [0.0, -0.0, 1.0, -1.0, 20.0, f32::INFINITY];
    let mut y = [SENTINEL; 8];
    act(ActKind::Silu, &x, 3, 2, &mut y).unwrap();
    assert_eq!(y[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(y[1].to_bits(), (-0.0f32).to_bits());
    // 1 / (1 + e^-1) = 0.731058578… and -1 / (1 + e) = -0.268941421…, to within 1 ulp.
    assert!((y[2] - 0.731_058_6).abs() <= f32::EPSILON);
    assert!((y[3] + 0.268_941_43).abs() <= f32::EPSILON);
    // e^-20 ≈ 2.06e-9 is below half an ulp of 1, so silu(20) = 20 exactly.
    assert_eq!(y[4], 20.0);
    assert_eq!(y[5], f32::INFINITY);
    assert_eq!(&y[6..], &[SENTINEL; 2]);
    for (i, &x) in x.iter().enumerate() {
        assert_eq!(y[i].to_bits(), math::silu(x).to_bits());
    }
    // A large negative input gives -0 once e^-x overflows; -inf gives NaN.
    let mut z = [SENTINEL; 2];
    act(ActKind::Silu, &[-200.0, f32::NEG_INFINITY], 2, 1, &mut z).unwrap();
    assert_eq!(z[0].to_bits(), (-0.0f32).to_bits());
    assert!(z[1].is_nan());
}

#[test]
fn gelu_is_refused_and_writes_nothing() {
    let mut y = [SENTINEL; 2];
    let e = act(ActKind::Gelu, &[1.0, 2.0], 2, 1, &mut y).unwrap_err();
    assert!(matches!(e, OpError::Unsupported { op: "Act", .. }), "{e}");
    assert_eq!(y, [SENTINEL; 2]);
}

#[test]
fn short_buffers_are_refused() {
    let mut y = [SENTINEL; 3];
    let e = act(ActKind::Silu, &[1.0; 4], 2, 2, &mut y).unwrap_err();
    assert!(matches!(e, OpError::Len { arg: "y", .. }), "{e}");
    assert_eq!(y, [SENTINEL; 3]);
}

proptest! {
    #[test]
    fn rows_active_zero_writes_nothing(x in prop::collection::vec(any::<f32>(), 0..64), width in 0usize..8) {
        let mut y = vec![SENTINEL; 64];
        act(ActKind::Silu, &x, width, 0, &mut y).unwrap();
        prop_assert!(y.iter().all(|&v| v == SENTINEL));
    }

    #[test]
    fn nan_stays_in_its_element(
        x in prop::collection::vec(-50.0f32..50.0, 12),
        at in 0usize..12,
    ) {
        let mut x = x;
        x[at] = f32::NAN;
        let mut y = vec![SENTINEL; 16];
        act(ActKind::Silu, &x, 4, 3, &mut y).unwrap();
        for (i, &v) in y[..12].iter().enumerate() {
            prop_assert_eq!(v.is_nan(), i == at);
        }
        prop_assert!(y[12..].iter().all(|&v| v == SENTINEL));
    }
}
