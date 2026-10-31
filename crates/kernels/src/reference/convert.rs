// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OpKind::RowGather` and `OpKind::Convert`'s reference semantics (RFC-0040 §3.1, §3.3;
//! COMP-E1-T03).
//!
//! Both read values stored as a GGUF type and decode them exactly: `F32` by its bits, `F16` and
//! `BF16` by exact widening, and `Q8_0`, `Q4_0` and `Q4_K` by RFC-0042 §2's reference
//! dequantisers ([`dequant`]), which are ggml's `dequantize_row_*` bit for bit. The other GGUF
//! types are refused until COMP-E13-T10 adds their dequantisers.
//!
//! - **`RowGather`** writes table row `ids[j]`, decoded to F32, as output row `j`, for each `j`:
//!   embedding and n-gram rows by token ID. It computes nothing beyond the decode, so it equals
//!   ggml's CPU `get_rows` bit for bit, NaN payloads and signed zeros included; `kernels-ggml`'s
//!   tests check that against the vendored ggml. The step's active rows are the IDs given; an ID
//!   may repeat. Every ID is checked before any row is written.
//! - **`Convert`** decodes each element to F32 as above, then encodes it as the target type:
//!   `F32` keeps the bits, `F16` and `BF16` round once to nearest, ties to even, as ggml's
//!   `fp32_to_fp16` and `fp32_to_bf16` do ([`f16_bits`], [`round_bf16`]): F16 keeps subnormals,
//!   overflows to infinity from 65520 and writes every NaN as a quiet NaN with its sign; BF16
//!   keeps a NaN's upper bits and makes it quiet. Quantising to a block type is not `Convert`'s:
//!   activation quantisation is RFC-0042 §4.3's ([`act_quant`](super::act_quant)).
//!
//! Every multi-byte value is little-endian, as GGUF stores it.

use syntropy_core::GgmlType;

use super::act_quant::{f16_bits, round_bf16};
use super::dequant::{self, DequantError, block_geometry};
use super::shape::{OpError, at_least};

/// The largest block any decoded type has: `Q4_K`'s 256 elements.
const MAX_BLOCK: usize = 256;

/// The elements and bytes of one block of `ty`, for every type this module decodes.
fn geometry(ty: GgmlType) -> Result<(usize, usize), OpError> {
    match ty {
        GgmlType::F32 => Ok((1, 4)),
        _ => block_geometry(ty)
            .map(|g| (g.elems, g.bytes))
            .ok_or(OpError::Dequant(DequantError::Unsupported(ty))),
    }
}

/// `src`, whole blocks of `ty`, decoded into `out`, which holds exactly their elements.
fn decode(ty: GgmlType, src: &[u8], out: &mut [f32]) -> Result<(), OpError> {
    if ty == GgmlType::F32 {
        for (y, b) in out.iter_mut().zip(src.as_chunks::<4>().0) {
            *y = f32::from_le_bytes(*b);
        }
        return Ok(());
    }
    dequant::dequantize_into(ty, src, out)?;
    Ok(())
}

/// `RowGather`: table row `ids[j]` of `table` (rows of `row_elems` elements of `ty`), decoded to
/// F32, as row `j` of `out`, for every `j < ids.len()` (module docs).
///
/// # Errors
///
/// [`OpError::RowLength`] when `row_elems` is zero or not whole blocks of `ty`;
/// [`OpError::Len`] when `table` is not whole rows or `out` holds fewer than `ids.len()` rows;
/// [`OpError::RowOutOfRange`] for an ID past the table; [`OpError::Dequant`] for a type with no
/// reference dequantiser. `out` is untouched then.
pub fn row_gather(
    ty: GgmlType,
    table: &[u8],
    row_elems: usize,
    ids: &[u32],
    out: &mut [f32],
) -> Result<(), OpError> {
    const OP: &str = "RowGather";
    let (block, block_bytes) = geometry(ty)?;
    if row_elems == 0 || !row_elems.is_multiple_of(block) {
        return Err(OpError::RowLength {
            ty,
            elems: row_elems,
            block,
        });
    }
    let row_bytes = row_elems / block * block_bytes;
    if !table.len().is_multiple_of(row_bytes) {
        return Err(OpError::Len {
            op: OP,
            arg: "table",
            len: table.len(),
            need: table.len().next_multiple_of(row_bytes),
        });
    }
    let rows = table.len() / row_bytes;
    at_least(OP, "out", out.len(), ids.len(), row_elems)?;
    if let Some(&id) = ids.iter().find(|&&id| id as usize >= rows) {
        return Err(OpError::RowOutOfRange { id, rows });
    }
    for (&id, out) in ids.iter().zip(out.chunks_exact_mut(row_elems)) {
        let start = id as usize * row_bytes;
        decode(ty, &table[start..start + row_bytes], out)?;
    }
    Ok(())
}

/// `Convert`: every element of `src` (whole blocks of `from`) decoded to F32 and encoded as `to`
/// into `dst`, which holds exactly as many elements of `to` (module docs).
///
/// # Errors
///
/// [`OpError::Unsupported`] when `to` is not `F32`, `F16` or `BF16`; [`OpError::Dequant`] for a
/// `from` with no reference dequantiser; [`OpError::Len`] when `src` is not whole blocks or `dst`
/// is not exactly the converted size. `dst` is untouched then.
pub fn convert(from: GgmlType, src: &[u8], to: GgmlType, dst: &mut [u8]) -> Result<(), OpError> {
    const OP: &str = "Convert";
    let encode: fn(f32, &mut [u8]) = match to {
        GgmlType::F32 => |x, d| d.copy_from_slice(&x.to_le_bytes()),
        GgmlType::F16 => |x, d| d.copy_from_slice(&f16_bits(x).to_le_bytes()),
        GgmlType::BF16 => |x, d| d.copy_from_slice(&round_bf16(x).to_le_bytes()),
        _ => {
            return Err(OpError::Unsupported {
                op: OP,
                what: "a target type other than F32, F16 and BF16",
            });
        }
    };
    let out_bytes = if to == GgmlType::F32 { 4 } else { 2 };
    let (block, block_bytes) = geometry(from)?;
    if !src.len().is_multiple_of(block_bytes) {
        return Err(OpError::Len {
            op: OP,
            arg: "src",
            len: src.len(),
            need: src.len().next_multiple_of(block_bytes),
        });
    }
    let need = src.len() / block_bytes * block * out_bytes;
    if dst.len() != need {
        return Err(OpError::Len {
            op: OP,
            arg: "dst",
            len: dst.len(),
            need,
        });
    }
    let mut buf = [0.0f32; MAX_BLOCK];
    for (src, dst) in src
        .chunks_exact(block_bytes)
        .zip(dst.chunks_exact_mut(block * out_bytes))
    {
        let values = &mut buf[..block];
        decode(from, src, values)?;
        for (&x, d) in values.iter().zip(dst.chunks_exact_mut(out_bytes)) {
            encode(x, d);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
