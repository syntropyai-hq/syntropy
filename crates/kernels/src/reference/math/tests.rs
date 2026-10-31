// SPDX-License-Identifier: MIT OR Apache-2.0

//! The software `exp` against F64, and its edge cases.

use proptest::prelude::*;

use super::*;

/// The distance from `got` to the true value `want`, in units of the F32 spacing at `want`
/// (the subnormal spacing below the normal range).
fn ulps(got: f32, want: f64) -> f64 {
    let spacing = if want.abs() < f64::from(f32::MIN_POSITIVE) {
        2f64.powi(-149)
    } else {
        2f64.powi(want.abs().log2().floor() as i32 - 23)
    };
    (f64::from(got) - want).abs() / spacing
}

#[test]
fn exp_is_within_two_ulp_of_the_true_value() {
    // Every 97th bit pattern of each sign that lies in exp's finite range.
    let mut worst = 0.0f64;
    for bits in (0..0x8000_0000u32).step_by(97) {
        for x in [f32::from_bits(bits), f32::from_bits(bits | 0x8000_0000)] {
            if !(UNDERFLOW..=88.72283).contains(&x) {
                continue;
            }
            let want = f64::from(x).exp();
            let err = ulps(exp(x), want);
            assert!(
                err <= 2.0,
                "exp({x:e}) = {:e}, true {want:e}: {err} ulp",
                exp(x)
            );
            worst = worst.max(err);
        }
    }
    assert!(worst > 0.0);
}

#[test]
fn exp_edge_cases() {
    assert_eq!(exp(0.0), 1.0);
    assert_eq!(exp(-0.0), 1.0);
    assert!(ulps(exp(1.0), std::f64::consts::E) <= 2.0);
    assert_eq!(exp(f32::INFINITY), f32::INFINITY);
    assert_eq!(exp(f32::NEG_INFINITY).to_bits(), 0);
    assert_eq!(exp(88.72284), f32::INFINITY);
    assert_eq!(exp(100.0), f32::INFINITY);
    assert_eq!(exp(-104.0).to_bits(), 0);
    // Into the subnormal range: e^-100 ≈ 3.72e-44, which is 26 or 27 units of 2^-149.
    let tiny = exp(-100.0);
    assert!(tiny > 0.0 && tiny < f32::MIN_POSITIVE, "{tiny:e}");
    assert!(ulps(tiny, (-100f64).exp()) <= 1.0);
    // The largest finite result.
    let big = exp(88.0);
    assert!(big.is_finite() && ulps(big, 88f64.exp()) <= 2.0);
    // A NaN stays NaN with its payload, made quiet.
    let nan = f32::from_bits(0x7F80_0001);
    let got = exp(nan);
    assert!(got.is_nan());
    assert_eq!(got.to_bits() & 0x003F_FFFF, 1);
}

#[test]
fn silu_edge_cases() {
    assert_eq!(silu(0.0).to_bits(), 0);
    assert_eq!(silu(-0.0).to_bits(), 0x8000_0000);
    assert_eq!(silu(f32::INFINITY), f32::INFINITY);
    assert!(silu(f32::NEG_INFINITY).is_nan());
    assert!(silu(f32::NAN).is_nan());
    // exp(100) overflows, so silu(-100) is -100 / ∞ = -0.
    assert_eq!(silu(-100.0).to_bits(), 0x8000_0000);
    // silu(x) → x for large x: exp(-20) ≈ 2e-9 is below half an ulp of 1.
    assert_eq!(silu(20.0), 20.0);
}

/// The form is the division `x / (1 + exp(−x))`, not the product `x · (1 / (1 + exp(−x)))`: the
/// two round differently on part of the range, and the grid below must hit such points.
#[test]
fn silu_is_the_written_out_division() {
    let mut product_differs = 0;
    for i in 0..10_000 {
        let x = -20.0 + i as f32 * 0.004;
        let want = x / (1.0 + exp(-x));
        assert_eq!(silu(x).to_bits(), want.to_bits(), "silu({x})");
        if (x * (1.0 / (1.0 + exp(-x)))).to_bits() != want.to_bits() {
            product_differs += 1;
        }
    }
    assert!(product_differs > 0);
}

#[test]
fn silu_by_hand() {
    // x = ln 3 makes 1 + exp(-x) = 4/3, so silu = 0.75 · ln 3 ≈ 0.823_959_2.
    let x = 3f32.ln();
    assert!((silu(x) - 0.823_959_2).abs() < 1e-6);
    // silu(2) = 2 / (1 + e^-2) ≈ 1.761_594_2; silu(-2) = silu(2) - 2.
    assert!((silu(2.0) - 1.761_594_2).abs() < 1e-6);
    assert!((silu(-2.0) + 0.238_405_84).abs() < 1e-6);
}

proptest! {
    #[test]
    fn silu_matches_f64(x in -80.0f32..80.0) {
        let want = f64::from(x) / (1.0 + (-f64::from(x)).exp());
        prop_assert!(ulps(silu(x), want) <= 4.0, "silu({x}) = {}, true {want}", silu(x));
    }
}
