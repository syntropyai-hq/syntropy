// SPDX-License-Identifier: MIT OR Apache-2.0

//! The software `sin` and `cos` against F64, and their edge cases.

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

fn check(x: f32) -> f64 {
    let (s, c) = sin_cos(x);
    let (ws, wc) = (f64::from(x).sin(), f64::from(x).cos());
    let (es, ec) = (ulps(s, ws), ulps(c, wc));
    assert!(es <= 1.0, "sin({x:e}) = {s:e}, true {ws:e}: {es} ulp");
    assert!(ec <= 1.0, "cos({x:e}) = {c:e}, true {wc:e}: {ec} ulp");
    es.max(ec)
}

#[test]
fn sin_cos_are_within_one_ulp_of_the_true_value() {
    // Every 97th finite bit pattern of each sign. The oracle is the platform's F64 sin and cos,
    // whose own error is far below an F32 ulp.
    let mut worst = 0.0f64;
    for bits in (0..0x7F80_0000u32).step_by(97) {
        for x in [f32::from_bits(bits), f32::from_bits(bits | 0x8000_0000)] {
            worst = worst.max(check(x));
        }
    }
    assert!(worst > 0.0);
}

#[test]
fn arguments_near_multiples_of_half_pi() {
    // The F32 values on each side of k·π/2 for k up to 2^20, where the reduction cancels most.
    for k in 1..=(1u32 << 20) {
        let t = f64::from(k) * FRAC_PI_2;
        let x = t as f32;
        for x in [
            x,
            f32::from_bits(x.to_bits() - 1),
            f32::from_bits(x.to_bits() + 1),
        ] {
            check(x);
        }
    }
    // A few large exponents up to f32::MAX.
    for e in [30, 50, 80, 100, 120, 127] {
        let x = 2f32.powi(e);
        check(x);
        check(f32::from_bits(x.to_bits() + 12345));
    }
    check(f32::MAX);
}

#[test]
fn edge_cases() {
    assert_eq!(sin(0.0).to_bits(), 0.0f32.to_bits());
    assert_eq!(sin(-0.0).to_bits(), (-0.0f32).to_bits());
    assert_eq!(cos(0.0), 1.0);
    assert_eq!(cos(-0.0), 1.0);
    // Below about 2^-12, sin x rounds to x and cos x to 1.
    for x in [1e-20f32, f32::MIN_POSITIVE, f32::from_bits(1), 1e-5] {
        assert_eq!(sin(x), x);
        assert_eq!(sin(-x), -x);
        assert_eq!(cos(x), 1.0);
    }
    assert_eq!(sin(std::f32::consts::FRAC_PI_2), 1.0);
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let (s, c) = sin_cos(x);
        assert!(s.is_nan() && c.is_nan());
    }
    let (s, c) = sin_cos_f64(f64::NAN);
    assert!(s.is_nan() && c.is_nan());
}

#[test]
fn sin_is_odd_and_cos_even() {
    for bits in (0..0x7F80_0000u32).step_by(100_003) {
        let x = f32::from_bits(bits);
        assert_eq!(sin(-x).to_bits(), (-sin(x)).to_bits());
        assert_eq!(cos(-x).to_bits(), cos(x).to_bits());
    }
}

#[test]
fn f64_arguments() {
    // Angles of positions above 2^20 (RFC-0042 §4.2), against the F64 oracle rounded to F32.
    for pos in [(1u64 << 20) + 1, 3_000_017, 1 << 31, u64::from(u32::MAX)] {
        for freq in [1.0f32, 0.031_622_775, 1e-4] {
            let x = pos as f64 * f64::from(freq);
            let (s, c) = sin_cos_f64(x);
            assert!(ulps(s, x.sin()) <= 1.0, "sin({x})");
            assert!(ulps(c, x.cos()) <= 1.0, "cos({x})");
        }
    }
}

#[test]
fn the_table_is_two_over_pi() {
    // The top 53 bits of the table, as an F64 in [0.5, 1), equal 2/π rounded down or to nearest.
    let top = TWO_OVER_PI[0] >> 11;
    let v = top as f64 * 2f64.powi(-53);
    assert!((v - std::f64::consts::FRAC_2_PI).abs() <= 2f64.powi(-53));
}
