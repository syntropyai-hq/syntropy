// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Convert` between F32, F16 and BF16, and `RowGather` from a table of F32, F16, BF16,
//! `Q8_0`, `Q4_0` or `Q4_K` rows (RFC-0040 §3.1). Both decode exactly and compute nothing else,
//! so each equals `kernels::reference` bit for bit, NaN payloads and signed zeros included:
//!
//! - Decoding is `reference::act_quant`'s exact widening for F16 and BF16, the bits for F32,
//!   and RFC-0042 §2's reference dequantisers for the block types.
//! - `Convert`'s encoding is `reference::act_quant::f16_bits` and `round_bf16`: one rounding
//!   to nearest, ties to even.
//! - `RowGather` checks every ID before it writes any row. An ID that is negative or not below
//!   the table's rows fails the launch.

use syntropy_core::{Bytes, DType, GgmlType, QuantScheme};
use syntropy_kernels::reference::act_quant::{bf16_to_f32, f16_bits, f16_to_f32, round_bf16};
use syntropy_kernels::reference::dequant::{block_geometry, dequantize_into};
use syntropy_kernels::{Error, OpArgs, OpKey, OpKind, ShapeSig, TensorArg};

use super::{active_rows, at_level, elementwise_caps, plain_key, rows_span, run_for_level};
use crate::device::internal;
use crate::gemv::Level;
use crate::gemv::entry::packed;
use crate::launch::{HostSpan, Kernel, Launch, Plan};

/// Bytes per element of a type `Convert` takes.
fn width(t: DType) -> Option<usize> {
    match t {
        DType::F32 => Some(4),
        DType::F16 | DType::BF16 => Some(2),
        _ => None,
    }
}

/// Every element of `src` (values of `from`) decoded and encoded as `to` into `dst`, at
/// `level` (module docs). `dst` holds as many elements of `to` as `src` holds of `from`.
///
/// # Panics
///
/// When `from` or `to` is not F32, F16 or BF16, or the lengths disagree.
pub fn convert_rows(level: Level, from: DType, src: &[u8], to: DType, dst: &mut [u8]) {
    let (fw, tw) = (
        width(from).expect("Convert takes F32, F16 and BF16"),
        width(to).expect("Convert takes F32, F16 and BF16"),
    );
    assert!(
        src.len().is_multiple_of(fw) && src.len() / fw * tw == dst.len(),
        "Convert: the source and destination lengths disagree"
    );
    fn dec(t: DType) -> fn(&[u8]) -> f32 {
        match t {
            DType::F32 => |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            DType::F16 => |b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
            _ => |b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])),
        }
    }
    fn enc(t: DType) -> fn(f32, &mut [u8]) {
        match t {
            DType::F32 => |x, d| d.copy_from_slice(&x.to_le_bytes()),
            DType::F16 => |x, d| d.copy_from_slice(&f16_bits(x).to_le_bytes()),
            _ => |x, d| d.copy_from_slice(&round_bf16(x).to_le_bytes()),
        }
    }
    let (dec, enc) = (dec(from), enc(to));
    at_level(level, || {
        for (s, d) in src.chunks_exact(fw).zip(dst.chunks_exact_mut(tw)) {
            enc(dec(s), d);
        }
    });
}

/// Why [`row_gather`] refused its inputs. A refused call writes nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatherError {
    /// `ids[index]` is negative or not below the table's `rows`.
    Id {
        /// Where in `ids`.
        index: usize,
        /// The ID.
        id: i32,
        /// The table's rows.
        rows: usize,
    },
    /// The table type has no decoder here, the width is not whole blocks of it, or an argument
    /// is too short.
    Shape,
}

/// The bytes of one table row of `width` elements of `ty`, or `None` for a type with no
/// decoder or a width that is not whole blocks.
#[must_use]
pub fn table_row_bytes(ty: GgmlType, width: usize) -> Option<usize> {
    let (elems, bytes) = match ty {
        GgmlType::F32 => (1, 4),
        _ => {
            let g = block_geometry(ty)?;
            (g.elems, g.bytes)
        }
    };
    width.is_multiple_of(elems).then_some(width / elems * bytes)
}

/// Table row `ids[j]`, decoded to F32, as row `j` of `out`, for every `j` (module docs), at
/// `level`. `table` holds whole rows of `width` elements of `ty`.
///
/// # Errors
///
/// [`GatherError::Id`] for the first ID outside the table; [`GatherError::Shape`] for a type
/// with no decoder, a width that is not whole blocks, a table that is not whole rows or an
/// `out` shorter than `ids.len()` rows.
pub fn row_gather(
    level: Level,
    ty: GgmlType,
    table: &[u8],
    width: usize,
    ids: &[i32],
    out: &mut [f32],
) -> Result<(), GatherError> {
    let row = table_row_bytes(ty, width).ok_or(GatherError::Shape)?;
    if width == 0 || !table.len().is_multiple_of(row) || out.len() < ids.len() * width {
        return Err(GatherError::Shape);
    }
    let rows = table.len() / row;
    if let Some((index, &id)) = ids
        .iter()
        .enumerate()
        .find(|&(_, &id)| usize::try_from(id).map_or(true, |i| i >= rows))
    {
        return Err(GatherError::Id { index, id, rows });
    }
    at_level(level, || {
        for (&id, out) in ids.iter().zip(out.chunks_exact_mut(width)) {
            let src = &table[id as usize * row..(id as usize + 1) * row];
            if ty == GgmlType::F32 {
                for (y, b) in out.iter_mut().zip(src.as_chunks::<4>().0) {
                    *y = f32::from_le_bytes(*b);
                }
            } else {
                // The type, the width and the lengths were checked above.
                dequantize_into(ty, src, out).expect("checked table row");
            }
        }
    });
    Ok(())
}

/// The table type of a `RowGather` scheme.
fn table_type(s: QuantScheme) -> Option<GgmlType> {
    let t = match s {
        QuantScheme::Plain(DType::F32) => GgmlType::F32,
        QuantScheme::Plain(DType::F16) => GgmlType::F16,
        QuantScheme::Plain(DType::BF16) => GgmlType::BF16,
        QuantScheme::Ggml(g) => g,
        _ => return None,
    };
    matches!(
        t,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q8_0
            | GgmlType::Q4_0
            | GgmlType::Q4_K
    )
    .then_some(t)
}

fn float(s: &QuantScheme) -> Option<DType> {
    match s {
        QuantScheme::Plain(d @ (DType::F32 | DType::F16 | DType::BF16)) => Some(*d),
        QuantScheme::Ggml(GgmlType::F32) => Some(DType::F32),
        QuantScheme::Ggml(GgmlType::F16) => Some(DType::F16),
        QuantScheme::Ggml(GgmlType::BF16) => Some(DType::BF16),
        _ => None,
    }
}

/// The `Convert` kernel for `isa`'s level.
pub(crate) fn convert_kernel(isa: syntropy_kernels::CpuIsa) -> Kernel {
    Kernel {
        plan: convert_plan,
        caps: |key, shape| {
            convert_plan(key, shape)
                .map(|p| elementwise_caps(Bytes(0), p.out.unwrap_or(DType::F32)))
        },
        check: convert_check,
        run: run_for_level!(convert_run, Level::for_isa(isa)),
    }
}

/// `from` in `weight` as `Plain(from)`, `to` in `out`.
fn convert_plan(key: &OpKey, shape: &ShapeSig) -> Option<Plan> {
    if key.op != OpKind::Convert || !plain_key(key) || shape.dims.len() != 1 {
        return None;
    }
    let from = float(key.schemes.first()?)?;
    let to = float(key.schemes.last()?)?;
    (from != to).then(|| Plan {
        weight: Some(QuantScheme::Plain(from)),
        out: Some(to),
        ..Plan::for_shape(shape)
    })
}

fn plan_types(plan: &Plan) -> Result<(DType, DType), Error> {
    match (plan.weight, plan.out) {
        (Some(QuantScheme::Plain(from)), Some(to)) => Ok((from, to)),
        _ => Err(internal("Convert: the plan has no types".to_owned())),
    }
}

fn convert_check(plan: &Plan, args: &OpArgs<'_>) -> Result<(), Error> {
    let (from, to) = plan_types(plan)?;
    let n = plan.dims[0];
    let ([x], [y], [], []) = (args.inputs, args.outputs, args.weights, args.statics) else {
        return Err(internal(
            "Convert binds one input, one output, no weights and no constants".to_owned(),
        ));
    };
    let is =
        |a: &TensorArg, t| TensorArg::packed(a.buf, t, [plan.rows, n, 1, 1]).as_ref() == Some(a);
    if !is(x, from) || !is(y, to) {
        return Err(internal(format!(
            "Convert: x must be packed {from:?} [{}, {n}] and y packed {to:?} [{}, {n}]",
            plan.rows, plan.rows
        )));
    }
    Ok(())
}

fn convert_run(level: Level, l: &Launch<'_>) -> Result<(), Error> {
    const OP: &str = "Convert";
    let plan = l.plan();
    let (from, to) = plan_types(plan)?;
    let n = plan.dims[0] as usize;
    let rows = active_rows(l, OP)?;
    if rows == 0 || n == 0 {
        return Ok(());
    }
    let (Some(x), Some(y)) = (l.input(0), l.output(0)) else {
        return Err(internal(format!("{OP}: missing operands")));
    };
    let (fw, tw) = (width(from).unwrap_or(4), width(to).unwrap_or(4));
    let xs = rows_span(x, rows, n * fw, fw as u64, OP)?;
    let ys = rows_span(y, rows, n * tw, tw as u64, OP)?;
    if ys.overlaps(&xs) {
        return Err(internal(format!("{OP}: the output overlaps the input")));
    }
    // SAFETY: both spans were resolved inside live host buffers (`Launch`'s docs) and checked
    // to hold `rows · n` elements; they do not overlap, so the mutable view is the only view of
    // its bytes.
    let (src, dst) = unsafe {
        (
            std::slice::from_raw_parts(xs.addr as *const u8, xs.bytes as usize),
            std::slice::from_raw_parts_mut(ys.addr as *mut u8, ys.bytes as usize),
        )
    };
    convert_rows(level, from, src, to, dst);
    Ok(())
}

/// The `RowGather` kernel for `isa`'s level.
pub(crate) fn gather_kernel(isa: syntropy_kernels::CpuIsa) -> Kernel {
    Kernel {
        plan: gather_plan,
        caps: |key, shape| gather_plan(key, shape).map(|_| elementwise_caps(Bytes(0), DType::F32)),
        check: gather_check,
        run: run_for_level!(gather_run, Level::for_isa(isa)),
    }
}

/// `(table type, table_rows, width)` of a key and shape this kernel takes.
fn gather_shape(plan: &Plan) -> Option<(GgmlType, usize, usize)> {
    let ty = table_type(plan.weight?)?;
    let (rows, width) = (plan.dims[0] as usize, plan.dims[1] as usize);
    table_row_bytes(ty, width)?;
    Some((ty, rows, width))
}

fn gather_plan(key: &OpKey, shape: &ShapeSig) -> Option<Plan> {
    let f32_after = key
        .schemes
        .iter()
        .skip(1)
        .all(|s| float(s) == Some(DType::F32));
    if key.op != OpKind::RowGather || !plain_key(key) || !f32_after || shape.dims.len() != 2 {
        return None;
    }
    let plan = Plan::for_key(key, shape);
    (plan.dims[1] > 0 && gather_shape(&plan).is_some()).then_some(plan)
}

fn gather_check(plan: &Plan, args: &OpArgs<'_>) -> Result<(), Error> {
    let (ty, table_rows, width) = gather_shape(plan)
        .ok_or_else(|| internal("RowGather: the plan has no table".to_owned()))?;
    let ([ids], [y], [t], []) = (args.inputs, args.outputs, args.weights, args.statics) else {
        return Err(internal(
            "RowGather binds one input, one output, one weight and no constants".to_owned(),
        ));
    };
    let ids_ok = TensorArg::packed(ids.buf, DType::I32, [plan.rows, 1, 1, 1]).as_ref() == Some(ids);
    if !ids_ok || !packed(y, plan.rows, width) {
        return Err(internal(format!(
            "RowGather: ids must be packed I32 [{}] and y packed F32 [{}, {width}]",
            plan.rows, plan.rows
        )));
    }
    let need = table_row_bytes(ty, width).unwrap_or(0) as u64 * table_rows as u64;
    if t.len.0 < need {
        return Err(internal(format!(
            "RowGather: the table view holds {} bytes, the table {need}",
            t.len.0
        )));
    }
    Ok(())
}

fn gather_run(level: Level, l: &Launch<'_>) -> Result<(), Error> {
    const OP: &str = "RowGather";
    let plan = l.plan();
    let (ty, table_rows, width) =
        gather_shape(plan).ok_or_else(|| internal(format!("{OP}: the plan has no table")))?;
    let rows = active_rows(l, OP)?;
    if rows == 0 {
        return Ok(());
    }
    let (Some(ids), Some(y), Some(t)) = (l.input(0), l.output(0), l.weight(0)) else {
        return Err(internal(format!("{OP}: missing operands")));
    };
    let row = table_row_bytes(ty, width).unwrap_or(0);
    let is = rows_span(ids, rows, 4, 4, OP)?;
    let ys = rows_span(y, rows, width * 4, 4, OP)?;
    let ts = HostSpan {
        addr: t.addr,
        bytes: (row * table_rows) as u64,
    };
    if t.bytes < ts.bytes || (ts.bytes > 0 && t.addr == 0) {
        return Err(internal(format!(
            "{OP}: the table is shorter than the plan needs"
        )));
    }
    if ys.overlaps(&is) || ys.overlaps(&ts) {
        return Err(internal(format!(
            "{OP}: the output overlaps another operand"
        )));
    }
    // SAFETY: every span was resolved inside a live host buffer or weight view (`Launch`'s
    // docs) and checked above to hold what is read or written at the element's alignment; the
    // output overlaps nothing else, so its mutable view is the only view of its bytes.
    let (ids, y, table) = unsafe {
        (
            std::slice::from_raw_parts(is.addr as *const i32, rows),
            std::slice::from_raw_parts_mut(ys.addr as *mut f32, rows * width),
            if ts.bytes == 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts(ts.addr as *const u8, ts.bytes as usize)
            },
        )
    };
    row_gather(level, ty, table, width, ids, y).map_err(|e| match e {
        GatherError::Id { index, id, rows } => internal(format!(
            "{OP}: id {id} at row {index} lies outside the table's {rows} rows"
        )),
        GatherError::Shape => internal(format!("{OP}: the operands disagree with the plan")),
    })
}
