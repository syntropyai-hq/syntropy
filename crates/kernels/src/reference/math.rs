// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reference's software transcendental functions (RFC-0060 §4.1; RFC-0042 §6.2; COMP-E1-T03).
//!
//! `F32Fixed` takes `exp` from here, never from a platform library, so the reference gives the
//! same bits on every operating system and on a GPU (`RefDevice::Gpu`, COMP-E14-T23), which
//! replicates these functions operation for operation. Each is built from IEEE F32 additions,
//! multiplications and divisions, [`f32::round_ties_even`] and exponent bits, all exactly
//! specified; nothing is contracted.
//!
//! [`exp`] is within 2 ulp of the true value over its whole finite range (the unit test measures
//! it against F64 on every 97th F32 bit pattern). [`silu`] is ggml's form `x / (1 + exp(−x))`.
//! Further functions (`log`, `tanh`, `erf`) join with the ops that need them.

/// `log2(e)`, rounded to F32.
const LOG2_E: f32 = std::f32::consts::LOG2_E;
/// The high part of `ln 2`, with 15 significant bits, so `k · LN2_HI` is exact for every
/// `|k| < 2^9`.
const LN2_HI: f32 = f32::from_bits(0x3F31_7200);
/// The rest of `ln 2`, rounded to F32.
const LN2_LO: f32 = 1.428_606_8e-6;
/// Above this, `exp` overflows: the F32 just above `ln(f32::MAX)`.
const OVERFLOW: f32 = 88.722_84;
/// Below this, `exp` is below half the smallest subnormal and rounds to `+0`.
const UNDERFLOW: f32 = -103.972_08;

/// `2^k` for `−126 ≤ k ≤ 127`, from its exponent bits.
fn pow2(k: i32) -> f32 {
    debug_assert!((-126..=127).contains(&k));
    f32::from_bits(((k + 127) as u32) << 23)
}

/// `e^x` in F32, from IEEE operations only (module docs).
///
/// - Range reduction: `k = round_ties_even(x · log2 e)`, then `r = (x − k · LN2_HI) − k · LN2_LO`
///   with `|r| ≤ 0.35`.
/// - `e^r` by the degree-7 Taylor polynomial in Horner form, `1 + r(1 + r(1/2 + r(1/6 + …)))`,
///   each coefficient rounded to F32.
/// - Scaling by `2^k` in two exact power-of-two factors, `2^⌊k/2⌋` then `2^(k − ⌊k/2⌋)`, so the
///   only rounding is the last multiplication, including into the subnormal range.
///
/// Edge cases: NaN gives NaN (quiet, payload kept); `+∞` gives `+∞` and `−∞` gives `+0`;
/// `x > 88.72284` gives `+∞`; `x < −103.97208` gives `+0`; `exp(±0) = 1`.
#[must_use]
pub fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return x + x;
    }
    if x > OVERFLOW {
        return f32::INFINITY;
    }
    if x < UNDERFLOW {
        return 0.0;
    }
    let kf = (x * LOG2_E).round_ties_even();
    let r = (x - kf * LN2_HI) - kf * LN2_LO;
    const C: [f32; 8] = [
        1.0,
        1.0,
        0.5,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5040.0,
    ];
    let mut p = C[7];
    for &c in C[..7].iter().rev() {
        p = p * r + c;
    }
    // `kf` lies in [−150, 128], so both halves lie in [−75, 64].
    let k = kf as i32;
    let k1 = k.div_euclid(2);
    (p * pow2(k1)) * pow2(k - k1)
}

/// SiLU, `x / (1 + exp(−x))`, the form ggml computes, with [`exp`].
///
/// NaN gives NaN. `silu(+∞) = +∞`. `silu(−∞)` is NaN (`−∞ / +∞`), and a large negative finite
/// `x` gives `−0` once `exp(−x)` overflows. `silu(±0) = ±0`.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

#[cfg(test)]
mod tests;
