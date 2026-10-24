// SPDX-License-Identifier: MIT OR Apache-2.0

//! Activation quantisation references: what a matrix product does to its activation input
//! when its `Numerics::act_in` is not `None` (RFC-0042 §4.3; ADR-0421 items 3 and 4).
//!
//! - [`quantize_q8_0`], [`quantize_q8_1`] and [`quantize_q8_k`] give ggml's `Q8_0`, `Q8_1` and
//!   `Q8_K` activation blocks, byte for byte in ggml's layouts ([`BlockQ8_0::to_bytes`] and the
//!   others). `ActQuant::Q8Block32` is the first two, which share `d` and `qs`; `Q8_1` adds
//!   the scaled sum `s`. `ActQuant::Q8K256` is `Q8_K`. [`ggml_activation_block`] gives the
//!   block ggml's CPU backend uses for each weight type.
//! - [`round_bf16`] is BF16 rounding of a matrix input (§4.3 rule b), as ggml rounds.
//! - [`fake_quant`] applies an [`ActQuant`] to one activation row and returns the values the
//!   product sees: the fake-quant step the op-level comparison inserts for an op whose
//!   activation quantisation is declared (RFC-0042 §6.4).
//!
//! **ggml's quantizers differ by ISA.** At the pinned commit, ggml has three `Q8_0` and `Q8_1`
//! quantizers ([`Q8Path`]): the portable `quantize_row_q8_{0,1}_ref` rounds `x · (1/d)` half
//! away from zero; the CPU backend's arm64 NEON path rounds it half to even; its x86 AVX and
//! AVX2 path rounds half to even and multiplies by `127/amax` in place of `1/d`, which can
//! differ from it in the last bit. They agree except where `x · id` falls on or next to a half
//! integer. `Q8_K` has one quantizer on arm64 and x86. Each path is implemented, and the golden
//! vectors in `tests/data/act_quant/` check the reference and NEON paths bit for bit against
//! ggml (`tests/ggml_act_quant.rs`). The AVX path is transcribed from the source and not yet
//! checked against vectors from an x86 host. A comparison against the ggml CPU baseline
//! uses the path that baseline ran.
//!
//! **Inputs ggml leaves undefined.** A Q8 block that holds a NaN or an infinity, or whose
//! inverse scale overflows F32 (an `amax` below about 3.7e-37), makes ggml convert an
//! out-of-range float to an integer, which C leaves undefined. Here such a block is defined:
//! a non-finite element makes `d` NaN and every `qs` 0, so the NaN reaches the product
//! (RFC-0042 §4.1 rule 6); an overflowing inverse scale gives `qs` 0 with ggml's own `d`,
//! which is 0 or −0 after rounding, so every dequantised value is 0 as it is in ggml.

use std::fmt;

use syntropy_core::{ActQuant, GgmlType};

/// Elements per `Q8_0` and `Q8_1` block: ggml's `QK8_0` and `QK8_1`.
pub const QK8: usize = 32;

/// Elements per `Q8_K` block: ggml's `QK_K`.
pub const QK_K: usize = 256;

/// E4M3's largest finite value (RFC-0042 §2.3), the divisor of a dynamic FP8 scale.
pub const E4M3_MAX: f32 = 448.0;

/// Which of ggml's `Q8_0` and `Q8_1` quantizers to reproduce (see the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Q8Path {
    /// `quantize_row_q8_0_ref` and `quantize_row_q8_1_ref` in `ggml-quants.c`: `id = 1/d`,
    /// rounded half away from zero (`roundf`). The CPU backend's path on ISAs with no
    /// path of their own.
    Reference,
    /// The CPU backend's arm64 NEON path: `id = 1/d`, rounded half to even (`vcvtnq_s32_f32`).
    Arm64Neon,
    /// The CPU backend's x86 AVX and AVX2 path: `id = 127/amax`, rounded half to even
    /// (`_mm256_round_ps` with `_MM_ROUND_NEAREST`).
    X86Avx,
}

/// One `Q8_0` block: `qs[j] · d` is element `j` (ggml's `block_q8_0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockQ8_0 {
    /// The scale, as F16 bits.
    pub d: u16,
    /// The quantised elements.
    pub qs: [i8; QK8],
}

/// One `Q8_1` block: `Q8_0` plus `s`, the F16 of `d · Σ qs` (ggml's `block_q8_1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockQ8_1 {
    /// The scale, as F16 bits.
    pub d: u16,
    /// `d · Σ qs`, as F16 bits.
    pub s: u16,
    /// The quantised elements.
    pub qs: [i8; QK8],
}

/// One `Q8_K` block: `qs[j] · d` is element `j`, with an F32 scale and the sum of each 16
/// consecutive `qs` (ggml's `block_q8_K`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ8K {
    /// The scale.
    pub d: f32,
    /// The quantised elements.
    pub qs: [i8; QK_K],
    /// `bsums[i]` is the sum of `qs[16 i .. 16 i + 16]`.
    pub bsums: [i16; QK_K / 16],
}

impl BlockQ8_0 {
    /// Bytes per block in ggml's layout.
    pub const BYTES: usize = 2 + QK8;

    /// The block in ggml's layout: `d` little-endian, then `qs`.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::BYTES] {
        let mut out = [0; Self::BYTES];
        out[..2].copy_from_slice(&self.d.to_le_bytes());
        for (o, q) in out[2..].iter_mut().zip(self.qs) {
            *o = q.to_le_bytes()[0];
        }
        out
    }

    /// The values the block holds, `qs[j] · d` in F32, as ggml dequantises it.
    #[must_use]
    pub fn dequantize(&self) -> [f32; QK8] {
        let d = f16_to_f32(self.d);
        self.qs.map(|q| f32::from(q) * d)
    }
}

impl BlockQ8_1 {
    /// Bytes per block in ggml's layout.
    pub const BYTES: usize = 4 + QK8;

    /// The block in ggml's layout: `d` and `s` little-endian, then `qs`.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::BYTES] {
        let mut out = [0; Self::BYTES];
        out[..2].copy_from_slice(&self.d.to_le_bytes());
        out[2..4].copy_from_slice(&self.s.to_le_bytes());
        for (o, q) in out[4..].iter_mut().zip(self.qs) {
            *o = q.to_le_bytes()[0];
        }
        out
    }
}

impl BlockQ8K {
    /// Bytes per block in ggml's layout.
    pub const BYTES: usize = 4 + QK_K + 2 * (QK_K / 16);

    /// The block in ggml's layout: `d` little-endian, then `qs`, then `bsums` little-endian.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::BYTES] {
        let mut out = [0; Self::BYTES];
        out[..4].copy_from_slice(&self.d.to_le_bytes());
        for (o, q) in out[4..4 + QK_K].iter_mut().zip(self.qs) {
            *o = q.to_le_bytes()[0];
        }
        for (o, s) in out[4 + QK_K..]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(self.bsums)
        {
            *o = s.to_le_bytes();
        }
        out
    }

    /// The values the block holds, `qs[j] · d` in F32, as ggml dequantises it.
    #[must_use]
    pub fn dequantize(&self) -> [f32; QK_K] {
        self.qs.map(|q| self.d * f32::from(q))
    }
}

/// The activation block ggml quantises to for one `Q8` weight type (ggml's `vec_dot_type`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlQ8Block {
    /// `Q8_0`: `ActQuant::Q8Block32`.
    Q8_0,
    /// `Q8_1`: `ActQuant::Q8Block32` with the scaled sum.
    Q8_1,
    /// `Q8_K`: `ActQuant::Q8K256`.
    Q8K,
}

impl GgmlQ8Block {
    /// The `ActQuant` a kernel declares for this block.
    #[must_use]
    pub const fn act_quant(self) -> ActQuant {
        match self {
            Self::Q8_0 | Self::Q8_1 => ActQuant::Q8Block32,
            Self::Q8K => ActQuant::Q8K256,
        }
    }
}

/// The 8-bit activation block ggml's CPU backend quantises a product's activations to for
/// weights of type `ty`, from the `vec_dot_type` column of its type traits at the pinned
/// commit (`ggml-cpu.c`). `None` for `F32`, `F16` and `BF16`, whose products take F32, F16
/// and BF16 activations.
#[must_use]
pub const fn ggml_activation_block(ty: GgmlType) -> Option<GgmlQ8Block> {
    match ty {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => None,
        GgmlType::Q8_0
        | GgmlType::Q4_0
        | GgmlType::Q5_0
        | GgmlType::IQ4_NL
        | GgmlType::Q2_0
        | GgmlType::Q1_0
        | GgmlType::MXFP4
        | GgmlType::NVFP4 => Some(GgmlQ8Block::Q8_0),
        GgmlType::Q4_1 | GgmlType::Q5_1 => Some(GgmlQ8Block::Q8_1),
        GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::IQ4_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ1_S
        | GgmlType::IQ1_M
        | GgmlType::TQ1_0
        | GgmlType::TQ2_0 => Some(GgmlQ8Block::Q8K),
        // A type added to `GgmlType` later has no 8-bit block until it is listed here, so
        // rule (b) never admits one for it by default. The tests list every variant.
        _ => None,
    }
}

/// A row whose length is not a whole number of blocks or groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowLength {
    /// The row's length.
    pub len: usize,
    /// The block or group length it must be a multiple of.
    pub block: usize,
}

impl fmt::Display for RowLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "row of {} elements is not a whole number of {}-element blocks",
            self.len, self.block
        )
    }
}

impl std::error::Error for RowLength {}

fn blocks(x: &[f32], block: usize) -> Result<std::slice::ChunksExact<'_, f32>, RowLength> {
    if x.len().is_multiple_of(block) {
        Ok(x.chunks_exact(block))
    } else {
        Err(RowLength {
            len: x.len(),
            block,
        })
    }
}

/// F16 NaN, as ggml's `ggml_compute_fp32_to_fp16` writes it.
const F16_NAN: u16 = 0x7E00;

/// `d`, the inverse scale and each `qs` of one 32-element block, shared by `Q8_0` and `Q8_1`.
/// `None` for a block with a non-finite element.
fn q8_block(x: &[f32], path: Q8Path) -> Option<(f32, [i8; QK8])> {
    if x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // ggml's `MAX(amax, fabsf(v))` over the block. Every path takes the same maximum of
    // finite values.
    let amax = x
        .iter()
        .fold(0.0f32, |m, v| if m > v.abs() { m } else { v.abs() });
    let d = amax / 127.0;
    let id = match path {
        Q8Path::Reference | Q8Path::Arm64Neon if d != 0.0 => 1.0 / d,
        Q8Path::X86Avx if amax != 0.0 => 127.0 / amax,
        _ => 0.0,
    };
    let mut qs = [0i8; QK8];
    if id.is_finite() {
        for (q, &v) in qs.iter_mut().zip(x) {
            let t = v * id;
            let r = match path {
                Q8Path::Reference => t.round(),
                Q8Path::Arm64Neon | Q8Path::X86Avx => t.round_ties_even(),
            };
            // |t| ≤ 127 plus rounding, so the conversion is exact.
            *q = r as i8;
        }
    }
    Some((d, qs))
}

/// Quantises `x` to ggml's `Q8_0` blocks by `path`'s quantizer (`ActQuant::Q8Block32`).
///
/// # Errors
///
/// [`RowLength`] when `x` is not a whole number of 32-element blocks.
pub fn quantize_q8_0(x: &[f32], path: Q8Path) -> Result<Vec<BlockQ8_0>, RowLength> {
    Ok(blocks(x, QK8)?
        .map(|b| match q8_block(b, path) {
            Some((d, qs)) => BlockQ8_0 { d: f16_bits(d), qs },
            None => BlockQ8_0 {
                d: F16_NAN,
                qs: [0; QK8],
            },
        })
        .collect())
}

/// Quantises `x` to ggml's `Q8_1` blocks by `path`'s quantizer (`ActQuant::Q8Block32` with
/// the scaled sum, for `Q4_1` and `Q5_1` weights).
///
/// # Errors
///
/// [`RowLength`] when `x` is not a whole number of 32-element blocks.
pub fn quantize_q8_1(x: &[f32], path: Q8Path) -> Result<Vec<BlockQ8_1>, RowLength> {
    Ok(blocks(x, QK8)?
        .map(|b| match q8_block(b, path) {
            Some((d, qs)) => {
                let sum: i32 = qs.iter().map(|&q| i32::from(q)).sum();
                // ggml's `sum*d`: the integer sum, at most 32 · 127 in magnitude, converts to
                // F32 exactly.
                let s = sum as f32 * d;
                BlockQ8_1 {
                    d: f16_bits(d),
                    s: f16_bits(s),
                    qs,
                }
            }
            None => BlockQ8_1 {
                d: F16_NAN,
                s: F16_NAN,
                qs: [0; QK8],
            },
        })
        .collect())
}

/// ggml's `nearest_int`: rounds `f` half to even through the F32 sum `f + 1.5 · 2^23`, for
/// `|f|` below 2^22.
fn nearest_int(f: f32) -> i32 {
    let v = f + 12_582_912.0;
    // The low 23 bits hold the integer offset by 2^22; the mask keeps the cast lossless.
    (v.to_bits() & 0x007F_FFFF) as i32 - 0x0040_0000
}

/// Quantises `x` to ggml's `Q8_K` blocks (`ActQuant::Q8K256`), as `quantize_row_q8_K_ref`,
/// which is the CPU backend's `Q8_K` quantizer on arm64 and x86 at the pinned commit.
///
/// The element of largest magnitude, the first on a tie, sets `iscale = −127/max`;
/// `qs = min(127, nearest_int(iscale · x))` and `d = 1/iscale`.
///
/// # Errors
///
/// [`RowLength`] when `x` is not a whole number of 256-element blocks.
pub fn quantize_q8_k(x: &[f32]) -> Result<Vec<BlockQ8K>, RowLength> {
    Ok(blocks(x, QK_K)?.map(q8_k_block).collect())
}

fn q8_k_block(x: &[f32]) -> BlockQ8K {
    let mut out = BlockQ8K {
        d: 0.0,
        qs: [0; QK_K],
        bsums: [0; QK_K / 16],
    };
    if x.iter().any(|v| !v.is_finite()) {
        out.d = f32::NAN;
        return out;
    }
    let (mut amax, mut max) = (0.0f32, 0.0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax == 0.0 {
        return out;
    }
    let iscale = -127.0 / max;
    if iscale.is_finite() {
        for (q, &v) in out.qs.iter_mut().zip(x) {
            // |iscale · v| ≤ 127 plus rounding, so `nearest_int` is in range and the result
            // fits an i8 after the clamp.
            *q = nearest_int(iscale * v).min(127) as i8;
        }
        for (s, chunk) in out.bsums.iter_mut().zip(out.qs.as_chunks::<16>().0) {
            // At most 16 · 128 in magnitude.
            *s = chunk.iter().map(|&q| i16::from(q)).sum();
        }
    }
    out.d = 1.0 / iscale;
    out
}

/// `x` rounded to F16, as ggml's `ggml_compute_fp32_to_fp16`: to nearest, ties to even, with
/// F16 subnormals kept, values from 65520 up to infinity, and every NaN to `sign | 0x7E00`.
#[must_use]
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let abs = b & 0x7FFF_FFFF;
    if abs > 0x7F80_0000 {
        return sign | F16_NAN;
    }
    // 65520 = (2 − 2^-11) · 2^15 is the midpoint above F16's largest finite value, a tie that
    // rounds to even, which is infinity.
    if abs >= 0x477F_F000 {
        return sign | 0x7C00;
    }
    let exp = (abs >> 23) as i32 - 127;
    let mant = abs & 0x007F_FFFF;
    let (base, rem, half) = if exp >= -14 {
        // Normal: drop 13 mantissa bits. A carry out of the mantissa steps the exponent,
        // which is the right result.
        let base = (((exp + 15) as u32) << 10) | (mant >> 13);
        (base, mant & 0x1FFF, 0x1000)
    } else if abs < 0x0080_0000 {
        // An F32 subnormal is below 2^-126, far below half of F16's smallest subnormal.
        (0, 0, 1)
    } else {
        // F16 subnormal: units of 2^-24. The value is sig · 2^(exp − 23), so it is
        // sig >> (−exp − 1) units, with the shifted-out bits as the remainder.
        let sig = u64::from(mant | 0x0080_0000);
        let shift = (-exp - 1) as u32;
        if shift >= 40 {
            (0, 0, 1)
        } else {
            let base = (sig >> shift) as u32;
            let rem = sig & ((1u64 << shift) - 1);
            let half = 1u64 << (shift - 1);
            // Compare in u64; report a remainder above, at or below half as 2, 1 or 0 of 1.
            let r = match rem.cmp(&half) {
                std::cmp::Ordering::Less => 0,
                std::cmp::Ordering::Equal => 1,
                std::cmp::Ordering::Greater => 2,
            };
            (base, r, 1)
        }
    };
    let rounded = if rem > half || (rem == half && base & 1 == 1) {
        base + 1
    } else {
        base
    };
    sign | rounded as u16
}

/// The F32 value of F16 bits `h`, exact.
#[must_use]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1F);
    let mant = u32::from(h & 0x03FF);
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: m · 2^-24, normalised for F32.
            let lead = 31 - m.leading_zeros(); // position of the top set bit, 0..=9
            let e = lead + 127 - 24;
            sign | (e << 23) | ((m << (23 - lead)) & 0x007F_FFFF)
        }
        (0x1F, 0) => sign | 0x7F80_0000,
        (0x1F, m) => sign | 0x7FC0_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// `x` rounded to BF16, as ggml's `ggml_compute_fp32_to_bf16`: to nearest, ties to even,
/// subnormals kept, and a NaN made quiet with its upper bits kept (RFC-0042 §4.3 rule b).
#[must_use]
pub fn round_bf16(x: f32) -> u16 {
    let u = x.to_bits();
    if u & 0x7FFF_FFFF > 0x7F80_0000 {
        return ((u >> 16) | 64) as u16;
    }
    // At most 0xFFFF_FFFF + 0x8000 before the shift, so add in u64.
    ((u64::from(u) + 0x7FFF + u64::from((u >> 16) & 1)) >> 16) as u16
}

/// The F32 value of BF16 bits `h`, exact.
#[must_use]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// `x` rounded to E4M3 (the OCP FN variant, RFC-0042 §2.3), to nearest with ties to even, as
/// an F32 value. A NaN stays NaN, and a magnitude that rounds above 448 becomes NaN, as a
/// cast to the FN variant gives: it has no infinity.
#[must_use]
pub fn round_e4m3(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let a = x.abs();
    // The spacing of E4M3 values at `a`: 2^(e − 3) for a normal exponent e ≥ −6, and the
    // subnormal unit 2^-9 below 2^-6.
    let e = if a >= pow2(-6) {
        ((a.to_bits() >> 23) as i32 - 127).min(8)
    } else {
        -6
    };
    let ulp = pow2(e - 3);
    // Division and multiplication by a power of two are exact here; the rounding is the only
    // inexact step.
    let r = (a / ulp).round_ties_even() * ulp;
    if r > E4M3_MAX {
        f32::NAN
    } else {
        r.copysign(x)
    }
}

/// 2^`e` for `e` in F32's normal range, built from exponent bits (RFC-0042 §4.1 rule 7).
fn pow2(e: i32) -> f32 {
    debug_assert!((-126..=127).contains(&e));
    f32::from_bits(((e + 127) as u32) << 23)
}

/// Why [`fake_quant`] cannot apply an `ActQuant`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FakeQuantError {
    /// The row is not a whole number of the scheme's blocks or groups.
    Length(RowLength),
    /// No reference exists for the scheme yet: MX FP8 and NVFP4 activations come with the
    /// rule (c) paths that use them (RFC-0042 §4.3).
    Unsupported(ActQuant),
}

impl fmt::Display for FakeQuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length(l) => l.fmt(f),
            Self::Unsupported(q) => write!(f, "no activation quantisation reference for {q:?}"),
        }
    }
}

impl std::error::Error for FakeQuantError {}

impl From<RowLength> for FakeQuantError {
    fn from(l: RowLength) -> Self {
        Self::Length(l)
    }
}

/// Applies `act` to one activation row `x` (one token's input along K) and returns the F32
/// values the matrix product multiplies: quantise, then dequantise as ggml does.
///
/// - `None`: `x` unchanged.
/// - `Bf16`: each element rounded to BF16 ([`round_bf16`]).
/// - `Q8Block32`: [`quantize_q8_0`] by `path`, dequantised (`Q8_1` holds the same values).
/// - `Q8K256`: [`quantize_q8_k`], dequantised.
/// - `Fp8Dynamic { group }`: per group of `group` elements, `s = amax / 448` in F32 and each
///   element `round_e4m3(x / s) · s`, the dynamic per-token scheme RFC-0042 §4.3 rule (a)
///   states; a group of zeros stays zero.
///
/// # Errors
///
/// [`FakeQuantError::Length`] when `x` is not a whole number of the scheme's blocks;
/// [`FakeQuantError::Unsupported`] for `Mxfp8`, `Nvfp4` and `Fp8Dynamic { group: 0 }`.
pub fn fake_quant(act: ActQuant, x: &[f32], path: Q8Path) -> Result<Vec<f32>, FakeQuantError> {
    match act {
        ActQuant::None => Ok(x.to_vec()),
        ActQuant::Bf16 => Ok(x.iter().map(|&v| bf16_to_f32(round_bf16(v))).collect()),
        ActQuant::Q8Block32 => Ok(quantize_q8_0(x, path)?
            .iter()
            .flat_map(BlockQ8_0::dequantize)
            .collect()),
        ActQuant::Q8K256 => Ok(quantize_q8_k(x)?
            .iter()
            .flat_map(BlockQ8K::dequantize)
            .collect()),
        ActQuant::Fp8Dynamic { group } if group > 0 => {
            let group = usize::from(group);
            Ok(blocks(x, group)?.flat_map(fp8_group).collect())
        }
        _ => Err(FakeQuantError::Unsupported(act)),
    }
}

fn fp8_group(x: &[f32]) -> Vec<f32> {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if x.iter().any(|v| v.is_nan()) {
        return vec![f32::NAN; x.len()];
    }
    if amax == 0.0 {
        return x.to_vec();
    }
    let s = amax / E4M3_MAX;
    x.iter().map(|&v| round_e4m3(v / s) * s).collect()
}

#[cfg(test)]
mod tests;
