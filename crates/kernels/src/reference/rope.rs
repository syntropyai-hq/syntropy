// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OpKind::Rope`'s reference semantics, and the RoPE that `KvWrite` and the attention ops apply
//! (RFC-0040 §3.1, §3.3; RFC-0042 §4.2's `Rope` row; RFC-0043 §3.1; COMP-E1-T17).
//!
//! **The table.** A [`Rope`] gives the base `θ`, the rotated dims per head `d_rot` (even, at most
//! `d_head`; 0 for a layer without RoPE), the frequency rule and the pairing. [`RopeTable::new`]
//! computes one frequency per pair, `freq[i]` for `i < d_rot/2`, in F64 with the software `ln`
//! and `exp` below, and rounds each once to F32. `prepare` computes this table once (RFC-0043
//! §3.1); positions never enter it.
//!
//! - **`Plain`**: `freq[i] = θ^(−2i/d_rot)`, computed as `exp(−(2i/d_rot) · ln θ)`.
//! - **`Linear`**: the plain frequency divided by `factor`, so positions are divided by it.
//! - **`Llama3`**: with `wavelen = 2π / f` for the plain frequency `f`, a wavelength below
//!   `original_ctx / high_freq_factor` keeps `f`; one above `original_ctx / low_freq_factor`
//!   takes `f / factor`; between them, `s = (original_ctx / wavelen − low_freq_factor) /
//!   (high_freq_factor − low_freq_factor)` and the frequency is `(1 − s) · f / factor + s · f`.
//! - **`Yarn`**: with `c(b) = d_rot · ln(original_ctx / (2π · b)) / (2 · ln θ)`, the ramp runs
//!   from `lo = max(⌊c(beta_fast)⌋, 0)` to `hi = min(⌈c(beta_slow)⌉, d_rot − 1)` (`hi` is
//!   `lo + 0.001` when they are equal), `ramp[i] = clamp((i − lo) / (hi − lo), 0, 1)`, and the
//!   frequency is `f / factor · ramp[i] + f · (1 − ramp[i])`. Every sine and cosine is also
//!   multiplied by the attention factor: `attn_factor` when the checkpoint states it, otherwise
//!   `0.1 · ln(factor) + 1` for `factor > 1` and 1 below. These are the formulas of the
//!   publishers' reference code for the `rope_scaling` parameters RFC-0043 §5.5's note lists.
//!
//! The other rules have an attention factor of exactly 1.
//!
//! **The rotation.** For a row at position `p`, pair `i` of each head is rotated by the angle
//! `a = p · freq[i]`. RFC-0042 §4.2: the angle is one F32 multiplication for `p ≤ 2^20`, and one
//! F64 multiplication for larger `p`. Then `(s, c) = (sin a, cos a)` from
//! [`trig`], each multiplied by the attention factor in F32, and the pair `(u, w)`
//! becomes
//!
//! ```text
//! u' = u·c − w·s
//! w' = w·c + u·s
//! ```
//!
//! each product rounded once, then the sum. [`RopePairs`] says which dims pair: `Adjacent` pairs
//! `(2i, 2i+1)`; `Halves` pairs `(i, i + d_rot/2)`, the form the publishers' code writes as
//! `rotate_half`. Which one a family uses depends on how its source stores Q and K, so the family
//! module chooses it. Dims at or past `d_rot` are copied unchanged.
//!
//! **Edge cases.** Position 0 gives every pair `(s, c) = (0, 1)`, so the values come back
//! unchanged, except for the sign of a zero (`u·1 − w·0` is `+0` for `u = −0, w = −0`), and
//! except that a NaN or infinite partner makes both values of its pair NaN (`∞ · 0`). At any
//! position a NaN or an infinity in one value of a pair can make both values NaN or infinite,
//! and leaves every other pair alone.
//! With `rows_active = 0` [`rope`] writes nothing; rows past it are neither read nor written.

use super::shape::{OpError, at_least};
use super::trig;
use crate::op::RopeKind;

/// A RoPE frequency rule with every input its table needs (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeScaling {
    /// Unscaled frequencies.
    Plain,
    /// Positions divided by `factor`.
    Linear {
        /// The position scaling factor.
        factor: f32,
    },
    /// Llama-3-style frequency scaling.
    Llama3 {
        /// The frequency scaling factor.
        factor: f32,
        /// The low-frequency factor.
        low_freq_factor: f32,
        /// The high-frequency factor.
        high_freq_factor: f32,
        /// The context length the model was trained at before scaling.
        original_ctx: u32,
    },
    /// YaRN scaling.
    Yarn {
        /// The context extension factor.
        factor: f32,
        /// The context length the model was trained at before scaling.
        original_ctx: u32,
        /// The fast boundary of the interpolation ramp, in rotations.
        beta_fast: f32,
        /// The slow boundary of the interpolation ramp, in rotations.
        beta_slow: f32,
        /// The attention factor, when the checkpoint states one.
        attn_factor: Option<f32>,
    },
}

impl RopeScaling {
    /// The [`RopeKind`] this rule is.
    #[must_use]
    pub fn kind(&self) -> RopeKind {
        match self {
            Self::Plain => RopeKind::Plain,
            Self::Linear { .. } => RopeKind::Linear,
            Self::Llama3 { .. } => RopeKind::Llama3,
            Self::Yarn { .. } => RopeKind::Yarn,
        }
    }
}

/// Which dims of a head form a rotated pair (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopePairs {
    /// `(2i, 2i + 1)`.
    Adjacent,
    /// `(i, i + d_rot/2)`.
    Halves,
}

/// A layer's rotary embedding (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    /// The frequency rule.
    pub scaling: RopeScaling,
    /// The base, `θ`.
    pub theta: f32,
    /// Rotated dims per head: even, at most `d_head`; 0 for a layer without RoPE.
    pub d_rot: usize,
    /// Which dims pair.
    pub pairs: RopePairs,
}

/// A [`Rope`]'s frequencies and attention factor, computed once (module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct RopeTable {
    freqs: Vec<f32>,
    attn_factor: f32,
    pairs: RopePairs,
    kind: RopeKind,
}

/// The largest position whose angle is computed in F32 (RFC-0042 §4.2).
pub const F32_ANGLE_MAX: u32 = 1 << 20;

const OP: &str = "Rope";

fn positive(x: f32) -> bool {
    x.is_finite() && x > 0.0
}

impl RopeTable {
    /// The table of `rope` (module docs).
    ///
    /// # Errors
    ///
    /// [`OpError::Param`] when `d_rot` is odd, `θ` is not a finite value above 1, or a scaling
    /// parameter is outside its domain: a factor, frequency factor or `beta` that is not finite
    /// and positive, `high_freq_factor ≤ low_freq_factor`, an `original_ctx` of 0, or a stated
    /// attention factor that is not finite.
    pub fn new(rope: &Rope) -> Result<Self, OpError> {
        let param = |what| Err(OpError::Param { op: OP, what });
        if !rope.d_rot.is_multiple_of(2) {
            return param("d_rot is odd");
        }
        if !(rope.theta.is_finite() && rope.theta > 1.0) {
            return param("theta is not a finite value above 1");
        }
        match rope.scaling {
            RopeScaling::Plain => {}
            RopeScaling::Linear { factor } => {
                if !positive(factor) {
                    return param("the linear factor is not finite and positive");
                }
            }
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_ctx,
            } => {
                if !(positive(factor) && positive(low_freq_factor) && positive(high_freq_factor)) {
                    return param("a Llama-3 factor is not finite and positive");
                }
                if high_freq_factor <= low_freq_factor {
                    return param("high_freq_factor is not above low_freq_factor");
                }
                if original_ctx == 0 {
                    return param("original_ctx is 0");
                }
            }
            RopeScaling::Yarn {
                factor,
                original_ctx,
                beta_fast,
                beta_slow,
                attn_factor,
            } => {
                if !(positive(factor) && positive(beta_fast) && positive(beta_slow)) {
                    return param("a YaRN factor or beta is not finite and positive");
                }
                if original_ctx == 0 {
                    return param("original_ctx is 0");
                }
                if attn_factor.is_some_and(|a| !a.is_finite()) {
                    return param("the YaRN attention factor is not finite");
                }
            }
        }

        let half = rope.d_rot / 2;
        let d = rope.d_rot as f64;
        let ln_theta = ln64(f64::from(rope.theta));
        let plain = |i: usize| exp64(-((2 * i) as f64 / d) * ln_theta);
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut attn_factor = 1.0f32;
        let freqs = match rope.scaling {
            RopeScaling::Plain => (0..half).map(|i| plain(i) as f32).collect(),
            RopeScaling::Linear { factor } => (0..half)
                .map(|i| (plain(i) / f64::from(factor)) as f32)
                .collect(),
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_ctx,
            } => {
                let (factor, lo, hi) = (
                    f64::from(factor),
                    f64::from(low_freq_factor),
                    f64::from(high_freq_factor),
                );
                let ctx = f64::from(original_ctx);
                let (low_wavelen, high_wavelen) = (ctx / lo, ctx / hi);
                (0..half)
                    .map(|i| {
                        let f = plain(i);
                        let wavelen = two_pi / f;
                        let scaled = if wavelen < high_wavelen {
                            f
                        } else if wavelen > low_wavelen {
                            f / factor
                        } else {
                            let s = (ctx / wavelen - lo) / (hi - lo);
                            (1.0 - s) * f / factor + s * f
                        };
                        scaled as f32
                    })
                    .collect()
            }
            RopeScaling::Yarn {
                factor,
                original_ctx,
                beta_fast,
                beta_slow,
                attn_factor: stated,
            } => {
                let factor64 = f64::from(factor);
                let ctx = f64::from(original_ctx);
                let corr = |b: f32| (d * ln64(ctx / (two_pi * f64::from(b)))) / (2.0 * ln_theta);
                let lo = corr(beta_fast).floor().max(0.0);
                let mut hi = corr(beta_slow).ceil().min(d - 1.0);
                if lo == hi {
                    hi += 0.001;
                }
                attn_factor = match stated {
                    Some(a) => a,
                    None if factor > 1.0 => (0.1 * ln64(factor64) + 1.0) as f32,
                    None => 1.0,
                };
                (0..half)
                    .map(|i| {
                        let f = plain(i);
                        let ramp = ((i as f64 - lo) / (hi - lo)).clamp(0.0, 1.0);
                        let extrapolate = 1.0 - ramp;
                        (f / factor64 * (1.0 - extrapolate) + f * extrapolate) as f32
                    })
                    .collect()
            }
        };
        Ok(Self {
            freqs,
            attn_factor,
            pairs: rope.pairs,
            kind: rope.scaling.kind(),
        })
    }

    /// The frequency rule.
    #[must_use]
    pub fn kind(&self) -> RopeKind {
        self.kind
    }

    /// The rotated dims per head, `d_rot`.
    #[must_use]
    pub fn d_rot(&self) -> usize {
        2 * self.freqs.len()
    }

    /// `freq[i]` for each pair `i` (module docs).
    #[must_use]
    pub fn freqs(&self) -> &[f32] {
        &self.freqs
    }

    /// The factor every sine and cosine is multiplied by: 1 except under YaRN.
    #[must_use]
    pub fn attn_factor(&self) -> f32 {
        self.attn_factor
    }

    /// `(s, c)` of pair `i` at position `pos`, with the attention factor applied (module docs).
    ///
    /// # Panics
    ///
    /// When `i ≥ d_rot/2`.
    #[must_use]
    pub fn sin_cos(&self, pos: u32, i: usize) -> (f32, f32) {
        let f = self.freqs[i];
        let (s, c) = if pos <= F32_ANGLE_MAX {
            trig::sin_cos(pos as f32 * f)
        } else {
            trig::sin_cos_f64(f64::from(pos) * f64::from(f))
        };
        (s * self.attn_factor, c * self.attn_factor)
    }

    /// Rotates the first `d_rot` values of one head in place, at position `pos` (module docs).
    /// `head` is at least `d_rot` long.
    pub(crate) fn rotate(&self, pos: u32, head: &mut [f32]) {
        let half = self.freqs.len();
        debug_assert!(head.len() >= 2 * half);
        for i in 0..half {
            let (a, b) = match self.pairs {
                RopePairs::Adjacent => (2 * i, 2 * i + 1),
                RopePairs::Halves => (i, i + half),
            };
            let (s, c) = self.sin_cos(pos, i);
            let (u, w) = (head[a], head[b]);
            head[a] = u * c - w * s;
            head[b] = w * c + u * s;
        }
    }
}

/// `Rope`: the first `rows_active` rows of `x`, each `n_heads` heads of `d_head` values, rotated
/// at `positions[r]` into `y` (module docs).
///
/// # Errors
///
/// [`OpError::Param`] when the table's `d_rot` exceeds `d_head`; [`OpError::Len`] when `x` or `y`
/// holds fewer than `rows_active` rows or `positions` fewer than `rows_active` values.
pub fn rope(
    table: &RopeTable,
    n_heads: usize,
    d_head: usize,
    x: &[f32],
    positions: &[u32],
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    if table.d_rot() > d_head {
        return Err(OpError::Param {
            op: OP,
            what: "d_rot exceeds d_head",
        });
    }
    let width = n_heads.checked_mul(d_head).ok_or(OpError::Param {
        op: OP,
        what: "n_heads · d_head overflows",
    })?;
    at_least(OP, "x", x.len(), rows_active, width)?;
    at_least(OP, "y", y.len(), rows_active, width)?;
    at_least(OP, "positions", positions.len(), rows_active, 1)?;
    if width == 0 {
        return Ok(());
    }
    let n = rows_active * width;
    y[..n].copy_from_slice(&x[..n]);
    if d_head == 0 {
        return Ok(());
    }
    for (row, &pos) in y[..n].chunks_exact_mut(width).zip(positions) {
        for head in row.chunks_exact_mut(d_head) {
            table.rotate(pos, head);
        }
    }
    Ok(())
}

/// `ln x` in F64 for a finite `x > 0`, from IEEE operations only: `x = 2^k · m` with
/// `m ∈ [√½, √2)`, then `ln m = 2·atanh(s)`, `s = (m − 1)/(m + 1)`, by its series to `s^25`.
/// Within a few F64 ulp; the tables round its results to F32.
pub(crate) fn ln64(x: f64) -> f64 {
    debug_assert!(x.is_finite() && x > 0.0);
    let (mut x, mut k) = (x, 0i32);
    if x < f64::MIN_POSITIVE {
        x *= 2f64.powi(64);
        k -= 64;
    }
    let bits = x.to_bits();
    k += ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mut m = f64::from_bits((bits & ((1 << 52) - 1)) | (1023 << 52));
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        k += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut p = 1.0 / 25.0;
    for j in (0..12).rev() {
        p = p * s2 + 1.0 / f64::from(2 * j + 1);
    }
    f64::from(k) * std::f64::consts::LN_2 + 2.0 * s * p
}

/// `e^y` in F64 for `|y| ≤ 700`, from IEEE operations only: `y = k·ln 2 + r` with `|r| ≤ ln2/2`
/// (`ln 2` in two parts, the first exact times any `|k| < 2^21`), `e^r` by its Taylor series to
/// `r^14`, and `2^k` from exponent bits. Within a few F64 ulp.
pub(crate) fn exp64(y: f64) -> f64 {
    debug_assert!(y.abs() <= 700.0);
    // ln 2 = LN2_HI + LN2_LO to about 2^-86: LN2_HI is ln 2 to 32 bits, LN2_LO the rest.
    const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let k = (y * std::f64::consts::LOG2_E).round_ties_even();
    let r = (y - k * LN2_HI) - k * LN2_LO;
    let mut p = 1.0;
    for j in (1..=14).rev() {
        p = 1.0 + p * r / f64::from(j);
    }
    p * f64::from_bits(((k as i64 + 1023) as u64) << 52)
}

#[cfg(test)]
mod tests;
