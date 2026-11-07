// SPDX-License-Identifier: MIT OR Apache-2.0

//! The software natural logarithm (COMP-E1-T18), for `TapGather`'s log-sum-exp.

/// The high part of `ln 2`, with 16 significant bits, so `k · LN2_HI` is exact for every
/// exponent `k` an F32 can have (`|k| ≤ 149`).
const LN2_HI: f32 = f32::from_bits(0x3F31_7180);
/// The rest of `ln 2`, rounded to F32.
const LN2_LO: f32 = f32::from_bits(0x3717_F7D1);
/// `√2`, rounded to F32: the top of the reduced argument's range.
const SQRT2: f32 = std::f32::consts::SQRT_2;
/// The coefficients of the polynomial `R(s²)` with `ln((1 + s) / (1 − s)) ≈ 2s + s·R(s²)`,
/// a minimax fit over `|s| ≤ 0.1716`, each rounded to F32.
const LG1: f32 = f32::from_bits(0x3F2A_AAAA);
const LG2: f32 = f32::from_bits(0x3ECC_CCE1);
const LG3: f32 = f32::from_bits(0x3E91_E9EE);
const LG4: f32 = f32::from_bits(0x3E78_9E26);

/// `ln x` in F32, from IEEE operations only, like [`exp`](super::exp).
///
/// - Reduction: `x = 2^k · m` with `m` in `[√2/2, √2)`, read from the exponent bits (a
///   subnormal is first scaled by `2^25`, exactly). `f = m − 1` is exact.
/// - `ln m = 2·atanh(s)` with `s = f / (2 + f)`, written as
///   `f − (f²/2 − s·(f²/2 + R))`, where `R = z·(LG1 + w·LG3) + w·(LG2 + w·LG4)`, `z = s²` and
///   `w = z²`, so the large term `f` is added last.
/// - `k · ln 2` is added in two parts, the exact `k · LN2_HI` last.
///
/// Edge cases: NaN gives NaN (quiet, payload kept); a negative `x`, `−∞` included, gives NaN;
/// `ln(±0) = −∞`; `ln(+∞) = +∞`; `ln 1 = +0`.
#[must_use]
pub fn ln(x: f32) -> f32 {
    if x.is_nan() {
        return x + x;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x < 0.0 {
        return f32::NAN;
    }
    if x == f32::INFINITY {
        return x;
    }
    let (mut bits, mut k) = (x.to_bits(), 0i32);
    if bits < 0x0080_0000 {
        // A subnormal: `2^25` brings it into the normal range exactly.
        bits = (x * f32::from_bits((25 + 127) << 23)).to_bits();
        k = -25;
    }
    k += (bits >> 23) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    if m > SQRT2 {
        // Halving is exact.
        m *= 0.5;
        k += 1;
    }
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let r = z * (LG1 + w * LG3) + w * (LG2 + w * LG4);
    let hfsq = 0.5 * f * f;
    let kf = k as f32;
    kf * LN2_HI - ((hfsq - (s * (hfsq + r) + kf * LN2_LO)) - f)
}
