// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reference's software `sin` and `cos` (RFC-0060 §4.1; RFC-0042 §4.2's `Rope` row;
//! COMP-E1-T17).
//!
//! `F32Fixed` takes `sin` and `cos` from here, never from a platform library, for the same reason
//! it takes `exp` from [`math`](super::math): the reference must give the same bits on every
//! operating system and on a GPU (`RefDevice::Gpu`, COMP-E14-T23). They are built from IEEE F64
//! additions, multiplications and divisions, integer arithmetic and the exactly rounded
//! conversions between integers and floats, all exactly specified; nothing is contracted. Only
//! the RoPE tables call them, once per (position, frequency), so the F64 cost does not matter.
//!
//! **Method.** For an argument `x`:
//!
//! 1. **Range reduction** writes `|x| = (n + f) · π/2` with `n` an integer and `|f| ≤ 1/2`. Below
//!    `π/4` it takes `n = 0`, `f·π/2 = |x|` exactly. Above, it multiplies the binary significand
//!    of `|x|` by a 128-bit window of the bits of `2/π` (`TWO_OVER_PI`, 256 bits) in exact
//!    integer arithmetic, chosen so that the bits it drops only change `n` by multiples of 4.
//!    `n mod 4` and the fraction come out of the 192-bit product exactly, and the fraction is
//!    rounded once to F64. Any finite F32 argument is reduced this way, however large.
//! 2. **The reduced argument** is `r = f · π/2` in F64, `|r| ≤ π/4`.
//! 3. **The polynomials** are the Taylor series of `sin r` to `r^17` and of `cos r` to `r^18` in
//!    Horner form, coefficients `1/k!` rounded to F64. The first omitted term is below `2^-53`
//!    of the result on `|r| ≤ π/4`.
//! 4. **The quadrant** `n mod 4` picks `±sin r` or `±cos r`, and the F64 value is rounded once
//!    to F32, to nearest with ties to even.
//!
//! The result is within 1 ulp of the true value for every finite F32 argument (the unit test
//! measures it against F64 on every 97th F32 bit pattern, and on the arguments nearest a
//! multiple of `π/2`). Edge cases: NaN gives NaN, `±∞` gives NaN, `sin(±0) = ±0`, `cos(±0) = 1`;
//! `sin` is odd and `cos` even, bit for bit.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

/// The first 256 bits of `2/π` after the binary point, most significant word first:
/// `2/π = 0.a2f9836e4e441529 fc27… (hex)`. Computed from Machin's formula in exact integer
/// arithmetic.
pub(crate) const TWO_OVER_PI: [u64; 4] = [
    0xa2f9_836e_4e44_1529,
    0xfc27_57d1_f534_ddc0,
    0xdb62_9599_3c43_9041,
    0xfe51_63ab_debb_c561,
];

/// Bits `j0 .. j0 + 127` of `2/π` after the binary point (bit 1 is the first), as an integer
/// whose last bit is bit `j0 + 127`. Bits past [`TWO_OVER_PI`] read as zero; the reduction never
/// asks for them (`j0 ≤ 129`).
fn window(j0: u32) -> u128 {
    debug_assert!((1..=129).contains(&j0));
    let start = (j0 - 1) as usize;
    let (w, s) = (start / 64, start % 64);
    let word = |i: usize| TWO_OVER_PI.get(i).copied().unwrap_or(0);
    let hi = (u128::from(word(w)) << 64) | u128::from(word(w + 1));
    if s == 0 {
        hi
    } else {
        (hi << s) | u128::from(word(w + 2) >> (64 - s))
    }
}

/// A 192-bit unsigned integer, least significant word first.
type U192 = [u64; 3];

fn mul_64_128(m: u64, t: u128) -> U192 {
    let lo = u128::from(m) * (t as u64 as u128);
    let hi = u128::from(m) * (t >> 64);
    let mid = (lo >> 64) + (hi as u64 as u128);
    [lo as u64, mid as u64, ((hi >> 64) + (mid >> 64)) as u64]
}

fn bit(p: &U192, i: u32) -> u64 {
    if i >= 192 {
        0
    } else {
        (p[(i / 64) as usize] >> (i % 64)) & 1
    }
}

/// `p · 2^s mod 2^192`, for `s < 192`.
fn shl(p: &U192, s: u32) -> U192 {
    debug_assert!(s < 192);
    let (w, b) = ((s / 64) as usize, s % 64);
    let mut out = [0u64; 3];
    for (i, o) in out.iter_mut().enumerate().skip(w) {
        let src = i - w;
        *o = p[src] << b;
        if b > 0 && src > 0 {
            *o |= p[src - 1] >> (64 - b);
        }
    }
    out
}

/// `2^192 − p`, for `p ≠ 0`.
fn neg(p: &U192) -> U192 {
    let mut out = [!p[0], !p[1], !p[2]];
    for w in &mut out {
        let (v, carry) = w.overflowing_add(1);
        *w = v;
        if !carry {
            break;
        }
    }
    out
}

fn leading_zeros(p: &U192) -> u32 {
    if p[2] != 0 {
        p[2].leading_zeros()
    } else if p[1] != 0 {
        64 + p[1].leading_zeros()
    } else {
        128 + p[0].leading_zeros()
    }
}

/// `2^k` in F64 from its exponent bits, for `−1022 ≤ k ≤ 1023`.
fn pow2_64(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// Reduces a finite `x ≥ 0` with `x < 2^128` to `(n mod 4, r)`, `x = n·π/2 + r` with
/// `|r| ≤ π/4` (module docs, steps 1 and 2).
fn reduce(x: f64) -> (u32, f64) {
    debug_assert!(x >= 0.0 && x < 2f64.powi(128));
    if x < FRAC_PI_4 {
        return (0, x);
    }
    // x ≥ π/4 is normal: x = m · 2^e with a 53-bit m.
    let bits = x.to_bits();
    let m = (bits & ((1 << 52) - 1)) | (1 << 52);
    let e = ((bits >> 52) & 0x7ff) as i32 - 1075;
    // Bits j < e − 1 of 2/π contribute m · 2^(e − j), a multiple of 4: they cannot change
    // n mod 4 or the fraction, so the window starts at j0.
    let j0 = (e - 1).max(1);
    let p = mul_64_128(m, window(j0 as u32));
    // p · 2^(e − j0 − 127) = x · 2/π mod 4, to 2^-73 of a quadrant or better: `f` fractional
    // bits, 126 ≤ f ≤ 181.
    let f = (j0 + 127 - e) as u32;
    let mut n = (bit(&p, f) | (bit(&p, f + 1) << 1)) as u32;
    // The fraction, aligned to the top of 192 bits.
    let mut frac = shl(&p, 192 - f);
    let negative = frac[2] >> 63 == 1;
    if negative {
        // f ≥ 1/2: take f − 1 in exact arithmetic, and the next quadrant.
        n = (n + 1) & 3;
        frac = neg(&frac);
    }
    if frac == [0; 3] {
        return (n, 0.0);
    }
    let lz = leading_zeros(&frac);
    let top = shl(&frac, lz);
    let top = (u128::from(top[2]) << 64) | u128::from(top[1]);
    // `u128 as f64` rounds to nearest, ties to even. lz < 192, so the scale is a normal F64.
    let mag = (top as f64) * pow2_64(-128 - lz as i32) * FRAC_PI_2;
    (n, if negative { -mag } else { mag })
}

/// `sin r` for `|r| ≤ π/4`, by its Taylor series to `r^17` in F64.
fn sin_poly(r: f64) -> f64 {
    const C: [f64; 8] = [
        -1.0 / 6.0,
        1.0 / 120.0,
        -1.0 / 5040.0,
        1.0 / 362_880.0,
        -1.0 / 39_916_800.0,
        1.0 / 6_227_020_800.0,
        -1.0 / 1_307_674_368_000.0,
        1.0 / 355_687_428_096_000.0,
    ];
    let s = r * r;
    let mut p = C[7];
    for &c in C[..7].iter().rev() {
        p = p * s + c;
    }
    r + (r * s) * p
}

/// `cos r` for `|r| ≤ π/4`, by its Taylor series to `r^18` in F64.
fn cos_poly(r: f64) -> f64 {
    const C: [f64; 9] = [
        -1.0 / 2.0,
        1.0 / 24.0,
        -1.0 / 720.0,
        1.0 / 40_320.0,
        -1.0 / 3_628_800.0,
        1.0 / 479_001_600.0,
        -1.0 / 87_178_291_200.0,
        1.0 / 20_922_789_888_000.0,
        -1.0 / 6_402_373_705_728_000.0,
    ];
    let s = r * r;
    let mut p = C[8];
    for &c in C[..8].iter().rev() {
        p = p * s + c;
    }
    1.0 + s * p
}

/// `(sin x, cos x)` in F64 before the final rounding, for a finite `x` with `|x| < 2^128`.
fn sin_cos_wide(x: f64) -> (f64, f64) {
    let (n, r) = reduce(x.abs());
    let (s, c) = (sin_poly(r), cos_poly(r));
    let (s, c) = match n {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    };
    (if x.is_sign_negative() { -s } else { s }, c)
}

/// `(sin x, cos x)` in F32 (module docs). Both come from one range reduction.
#[must_use]
pub fn sin_cos(x: f32) -> (f32, f32) {
    if !x.is_finite() {
        // NaN stays NaN (quiet, payload kept); ±∞ gives the default NaN.
        let nan = if x.is_nan() { x + x } else { f32::NAN };
        return (nan, nan);
    }
    let (s, c) = sin_cos_wide(f64::from(x));
    (s as f32, c as f32)
}

/// `sin x` in F32 (module docs).
#[must_use]
pub fn sin(x: f32) -> f32 {
    sin_cos(x).0
}

/// `cos x` in F32 (module docs).
#[must_use]
pub fn cos(x: f32) -> f32 {
    sin_cos(x).1
}

/// `(sin x, cos x)` rounded to F32, for an F64 argument: the RoPE angle of a position above
/// `2^20`, which RFC-0042 §4.2 computes in F64. The method is the module's; NaN and `±∞` give
/// NaN.
///
/// # Panics
///
/// When `|x| ≥ 2^128`, which no RoPE angle reaches (a `u32` position times a frequency of at
/// most 1).
#[must_use]
pub fn sin_cos_f64(x: f64) -> (f32, f32) {
    if !x.is_finite() {
        let nan = if x.is_nan() { (x + x) as f32 } else { f32::NAN };
        return (nan, nan);
    }
    assert!(x.abs() < 2f64.powi(128), "sin_cos_f64 of {x:e}");
    let (s, c) = sin_cos_wide(x);
    (s as f32, c as f32)
}

#[cfg(test)]
mod tests;
