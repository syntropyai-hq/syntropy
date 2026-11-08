// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reference dequantisation of GGUF weight blocks to F32 (RFC-0042 §2, §2.1; COMP-E13-T02).
//!
//! A GGUF type's value is ggml's own F32 dequantisation at the pinned commit (`third_party/
//! ggml.lock`): its `dequantize_row_*` functions and its F16 and BF16 conversions. [`dequantize`]
//! reimplements them for `F16`, `BF16`, `Q8_0`, `Q4_0` and `Q4_K`, in ggml's operation order, one
//! IEEE operation at a time:
//!
//! - **F16, BF16**: widened exactly ([`f16_to_f32`], [`bf16_to_f32`]): subnormals kept, infinities
//!   kept, a NaN made quiet with its payload kept, as ggml converts.
//! - **Q8_0**: `fl(q · d)`, `q` the signed 8-bit code and `d` the block's F16 scale.
//! - **Q4_0**: `fl((n − 8) · d)`, `n` a 4-bit code; the low nibbles give elements 0–15 and the high
//!   nibbles 16–31.
//! - **Q4_K**: per 32-element sub-block, `fl(fl(d · sc) · q) − fl(dmin · m)`, rounded again: the
//!   k-quant min term `d·sc·q − dmin·m` with `sc` and `m` the sub-block's 6-bit scale and minimum.
//!
//! Rust never contracts a multiply and an add into a fused operation, and F32 subnormals are
//! kept, so this is the two-rounding result that RFC-0042 §4.1 rule 7 makes the bit-exact class
//! (§6.4). `kernels-ggml`'s tests check it bit for bit against ggml's own functions, built with
//! contraction off, on random blocks and on zero, largest, subnormal, infinite and NaN scales.
//!
//! **Geometry.** [`block_geometry`] gives the five layouts this module decodes. Their byte counts
//! are checked against RFC-0042 §2.1's table when the crate compiles (the `const` assertions
//! below). `kernels-ggml`'s tests check them, and `syntropy_manifest`'s table for every GGUF type,
//! against ggml's type traits (`ggml_blck_size`, `ggml_type_size`).
//!
//! None of these five types has a power-of-two scale. The E8M0 scale of `MXFP4` and the other
//! GGUF types are COMP-E13-T10's (RFC-0042 §2.2).

use std::fmt;

use syntropy_core::GgmlType;

pub use super::act_quant::{bf16_to_f32, f16_to_f32};

/// A block type's geometry: the weights one block holds and the bytes it takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockGeometry {
    /// Weights per block.
    pub elems: usize,
    /// Bytes per block.
    pub bytes: usize,
}

/// Elements in a `Q8_0` and `Q4_0` block: ggml's `QK8_0` and `QK4_0`.
const QK32: usize = 32;
/// Elements in a k-quant block: ggml's `QK_K`.
const QK_K: usize = 256;
/// Bytes of an F16 scale.
const HALF: usize = 2;
/// Bytes of a `Q4_K` block's packed sub-block scales and minimums: ggml's `K_SCALE_SIZE`.
const K_SCALE_SIZE: usize = 12;

/// `block_q8_0`: `d`, then 32 signed codes.
const Q8_0_BYTES: usize = HALF + QK32;
/// `block_q4_0`: `d`, then 32 four-bit codes, two to a byte.
const Q4_0_BYTES: usize = HALF + QK32 / 2;
/// `block_q4_K`: `d`, `dmin`, the scales, then 256 four-bit codes.
const Q4_K_BYTES: usize = 2 * HALF + K_SCALE_SIZE + QK_K / 2;

/// The block geometry of a type [`dequantize`] decodes, from the layouts above: `F16`, `BF16`,
/// `Q8_0`, `Q4_0` and `Q4_K`. `None` for any other type. RFC-0042 §2.1's table for every type is
/// `syntropy_manifest::gguf::block_geometry`; `kernels-ggml`'s tests check both against ggml's
/// type traits.
#[must_use]
pub const fn block_geometry(ty: GgmlType) -> Option<BlockGeometry> {
    let (elems, bytes) = match ty {
        GgmlType::F16 | GgmlType::BF16 => (1, HALF),
        GgmlType::Q8_0 => (QK32, Q8_0_BYTES),
        GgmlType::Q4_0 => (QK32, Q4_0_BYTES),
        GgmlType::Q4_K => (QK_K, Q4_K_BYTES),
        _ => return None,
    };
    Some(BlockGeometry { elems, bytes })
}

// RFC-0042 §2.1's table for these layouts: they take the bytes it states, or the crate does not
// compile.
const _: () = {
    assert!(Q8_0_BYTES == 34);
    assert!(Q4_0_BYTES == 18);
    assert!(Q4_K_BYTES == 144);
};

/// Why [`dequantize`] refused its input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DequantError {
    /// The bytes are not a whole number of blocks.
    Length {
        /// The type.
        ty: GgmlType,
        /// The bytes given.
        bytes: usize,
    },
    /// The type has no reference dequantiser yet (COMP-E13-T10 adds the rest).
    Unsupported(GgmlType),
}

impl fmt::Display for DequantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { ty, bytes } => write!(
                f,
                "{bytes} bytes is not a whole number of {}-byte {ty:?} blocks",
                block_geometry(*ty).map_or(0, |g| g.bytes)
            ),
            Self::Unsupported(ty) => write!(f, "no reference dequantiser for {ty:?}"),
        }
    }
}

impl std::error::Error for DequantError {}

/// The F16 value in `bytes[0..2]`, little-endian.
fn half(bytes: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// ggml's `get_scale_min_k4`: sub-block `j`'s 6-bit scale and minimum from a `Q4_K` block's
/// 12 packed bytes.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn q8_0(block: &[u8], y: &mut [f32]) {
    let d = half(block);
    for (out, &q) in y.iter_mut().zip(&block[HALF..]) {
        *out = f32::from(q as i8) * d;
    }
}

fn q4_0(block: &[u8], y: &mut [f32]) {
    let d = half(block);
    let (low, high) = y.split_at_mut(QK32 / 2);
    for (j, &q) in block[HALF..].iter().enumerate() {
        low[j] = f32::from(i16::from(q & 0x0F) - 8) * d;
        high[j] = f32::from(i16::from(q >> 4) - 8) * d;
    }
}

fn q4_k(block: &[u8], y: &mut [f32]) {
    let d = half(block);
    let min = half(&block[HALF..]);
    let scales = &block[2 * HALF..2 * HALF + K_SCALE_SIZE];
    let qs = &block[2 * HALF + K_SCALE_SIZE..];
    // Four groups of 64 elements, each from 32 bytes: the low nibbles with sub-block `2g`'s
    // scale and minimum, then the high nibbles with sub-block `2g + 1`'s.
    for (g, (q, y)) in qs
        .as_chunks::<32>()
        .0
        .iter()
        .zip(y.as_chunks_mut::<64>().0.iter_mut())
        .enumerate()
    {
        let (sc, m) = scale_min_k4(2 * g, scales);
        let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
        let (sc, m) = scale_min_k4(2 * g + 1, scales);
        let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
        let (low, high) = y.split_at_mut(32);
        for l in 0..32 {
            low[l] = d1 * f32::from(q[l] & 0xF) - m1;
            high[l] = d2 * f32::from(q[l] >> 4) - m2;
        }
    }
}

/// `src`, whole blocks of `ty`, dequantised to F32 as ggml's `dequantize_row_*` (or its F16 and
/// BF16 conversion) gives them, bit for bit (module docs).
///
/// # Errors
///
/// [`DequantError::Length`] when `src` is not whole blocks; [`DequantError::Unsupported`] for a
/// type other than `F16`, `BF16`, `Q8_0`, `Q4_0` and `Q4_K`.
pub fn dequantize(ty: GgmlType, src: &[u8]) -> Result<Vec<f32>, DequantError> {
    let geometry = block_geometry(ty).ok_or(DequantError::Unsupported(ty))?;
    let mut out = vec![0.0f32; src.len() / geometry.bytes * geometry.elems];
    dequantize_into(ty, src, &mut out)?;
    Ok(out)
}

/// [`dequantize`], writing into `out` and allocating nothing, for the ops that read quantised
/// rows ([`row_gather`](super::convert::row_gather), [`convert`](super::convert::convert)).
///
/// # Errors
///
/// As [`dequantize`]; `out` is untouched then.
///
/// # Panics
///
/// When `out` does not hold exactly the elements of `src`'s blocks.
pub fn dequantize_into(ty: GgmlType, src: &[u8], out: &mut [f32]) -> Result<(), DequantError> {
    let decode: fn(&[u8], &mut [f32]) = match ty {
        GgmlType::F16 => |b, y| y[0] = half(b),
        GgmlType::BF16 => |b, y| y[0] = bf16_to_f32(u16::from_le_bytes([b[0], b[1]])),
        GgmlType::Q8_0 => q8_0,
        GgmlType::Q4_0 => q4_0,
        GgmlType::Q4_K => q4_k,
        other => return Err(DequantError::Unsupported(other)),
    };
    let geometry = block_geometry(ty).expect("every decoded type has a geometry");
    if !src.len().is_multiple_of(geometry.bytes) {
        return Err(DequantError::Length {
            ty,
            bytes: src.len(),
        });
    }
    assert_eq!(
        out.len(),
        src.len() / geometry.bytes * geometry.elems,
        "dequantize_into: the output holds a different number of elements than the blocks"
    );
    for (block, y) in src
        .chunks_exact(geometry.bytes)
        .zip(out.chunks_exact_mut(geometry.elems))
    {
        decode(block, y);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
