// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GateUpAct` with SiLU gating: `y[r][i] = silu(g[r][i]) · u[r][i]`, where `g` and `u` are the
//! gate and up products of the activation row (RFC-0040 §3.1; `reference::gate_up_act`).
//!
//! The weight is one matrix of `2·n_out` rows in GGUF block layout, the gate rows first and the
//! up rows after them, as the ggml provider's mapping takes it (RFC-0040 §8.2). The activation
//! rows are quantised once, as `Gemv` quantises them for the weight type. The intermediate
//! outputs are then computed [`GATE_UP_CHUNK`] at a time: the chunk's gate rows and up rows
//! each go through the level's GEMV kernels into the scratch, and the chunk's outputs are
//! `silu(g) · u` with `reference::math::silu`, one rounding for the product. Each `g` and `u`
//! is the value `Gemv` gives for that weight row bit for bit, whatever the chunk, so the output
//! is the same at every level and for every chunking.
//!
//! Against `reference::gate_up_act` the only difference is in `g` and `u`, each a `Gemv`
//! output, which the GEMV tests bound against `reference::gemv`.

use syntropy_core::Bytes;
use syntropy_kernels::reference::math::silu;
use syntropy_kernels::{Error, GatedAct, OpArgs, OpCaps, OpKey, OpKind, ShapeSig};

use super::{active_rows, plain_key, rows_span, run_for_level};
use crate::device::internal;
use crate::gemv::entry::{matmul_caps, packed, weight_shape, weight_type};
use crate::gemv::{
    ActView, GemvError, Level, WeightType, Weights, gemv, quantize_into, workspace_bytes,
};
use crate::launch::{HostSpan, Kernel, Launch, Plan};

/// Intermediate outputs computed per pass (module docs): 64 \[target\], a chosen budget that
/// bounds the scratch to `2 · rows · 64` F32 values whatever `n_out` is.
pub const GATE_UP_CHUNK: usize = 64;

/// The F32 scratch [`gate_up_act`] needs for `rows` activation rows: a gate and an up value per
/// row and chunk output.
#[must_use]
pub fn gate_up_scratch(rows: usize) -> usize {
    2 * rows * GATE_UP_CHUNK
}

/// `GateUpAct` with SiLU for every activation row of `x`, into `y` (`x.rows()` rows of
/// `w.n_out() / 2` values), with the kernels of `level` (module docs). `w` holds the gate rows
/// then the up rows; `scratch` holds at least [`gate_up_scratch`] values and is overwritten.
///
/// # Errors
///
/// [`GemvError::Len`] when `w` has an odd number of rows, or `y` or `scratch` is too short;
/// [`GemvError::Activations`] when `x` was quantised for another type or width.
pub fn gate_up_act(
    level: Level,
    w: &Weights<'_>,
    x: &ActView<'_>,
    y: &mut [f32],
    scratch: &mut [f32],
) -> Result<(), GemvError> {
    if !w.n_out().is_multiple_of(2) {
        return Err(GemvError::Len {
            arg: "weights",
            len: w.n_out(),
            need: w.n_out() + 1,
        });
    }
    if x.kind() != w.ty().act() || x.k() != w.k_in() {
        return Err(GemvError::Activations);
    }
    let (n, rows) = (w.n_out() / 2, x.rows());
    for (arg, len, need) in [
        ("y", y.len(), rows * n),
        ("scratch", scratch.len(), gate_up_scratch(rows)),
    ] {
        if len < need {
            return Err(GemvError::Len { arg, len, need });
        }
    }
    let (gate, up) = scratch[..gate_up_scratch(rows)].split_at_mut(rows * GATE_UP_CHUNK);
    for o0 in (0..n).step_by(GATE_UP_CHUNK) {
        let c = (n - o0).min(GATE_UP_CHUNK);
        let (g, u) = (&mut gate[..rows * c], &mut up[..rows * c]);
        gemv(level, &w.rows(o0..o0 + c), x, 0..c, g)?;
        gemv(level, &w.rows(n + o0..n + o0 + c), x, 0..c, u)?;
        for r in 0..rows {
            let out = &mut y[r * n + o0..r * n + o0 + c];
            for ((y, &g), &u) in out.iter_mut().zip(&g[r * c..]).zip(&u[r * c..]) {
                *y = silu(g) * u;
            }
        }
    }
    Ok(())
}

/// The `GateUpAct` kernel for `isa`'s level, lowered to what the live host runs.
pub(crate) fn kernel(isa: syntropy_kernels::CpuIsa) -> Kernel {
    Kernel {
        plan,
        caps,
        check,
        run: run_for_level!(run, Level::for_isa(isa)),
    }
}

/// `(weight type, n_out, k_in)`, for SiLU gating only: GELU and clamped SiLU are COMP-E1-T04's.
fn shape_of(key: &OpKey, shape: &ShapeSig) -> Option<(WeightType, usize, usize)> {
    if key.op
        != (OpKind::GateUpAct {
            act: GatedAct::Silu,
        })
        || !plain_key(key)
    {
        return None;
    }
    weight_shape(key, shape)
}

/// The workspace: the quantised activations, then the scratch, each on a cache line.
fn layout(ty: WeightType, rows: usize, k_in: usize) -> (usize, usize) {
    let quant = workspace_bytes(ty.act(), rows, k_in).next_multiple_of(64);
    (
        quant,
        quant + (gate_up_scratch(rows) * 4).next_multiple_of(64),
    )
}

fn plan(key: &OpKey, shape: &ShapeSig) -> Option<Plan> {
    let (ty, _, k_in) = shape_of(key, shape)?;
    let (_, total) = layout(ty, shape.rows as usize, k_in);
    Some(Plan {
        workspace: Bytes(total as u64),
        ..Plan::for_key(key, shape)
    })
}

fn caps(key: &OpKey, shape: &ShapeSig) -> Option<OpCaps> {
    let (ty, _, _) = shape_of(key, shape)?;
    Some(matmul_caps(ty, plan(key, shape)?.workspace))
}

fn check(plan: &Plan, args: &OpArgs<'_>) -> Result<(), Error> {
    let ty = plan
        .weight
        .and_then(weight_type)
        .ok_or_else(|| internal("GateUpAct: no weight scheme in the plan".to_owned()))?;
    let (n_out, k_in) = (plan.dims[0] as usize, plan.dims[1] as usize);
    let ([x], [y], [w], []) = (args.inputs, args.outputs, args.weights, args.statics) else {
        return Err(internal(
            "GateUpAct binds one input, one output, one weight and no constants".to_owned(),
        ));
    };
    if !packed(x, plan.rows, k_in) || !packed(y, plan.rows, n_out) {
        return Err(internal(format!(
            "GateUpAct: x must be packed F32 [{}, {k_in}] and y packed F32 [{}, {n_out}]",
            plan.rows, plan.rows
        )));
    }
    let need = (k_in / ty.block_elems() * ty.block_bytes()) as u64 * 2 * n_out as u64;
    if w.len.0 < need {
        return Err(internal(format!(
            "GateUpAct: the weight view holds {} bytes, the gate and up matrices {need}",
            w.len.0
        )));
    }
    Ok(())
}

fn run(level: Level, l: &Launch<'_>) -> Result<(), Error> {
    const OP: &str = "GateUpAct";
    let plan = l.plan();
    let fail = |what: String| internal(format!("{OP}: {what}"));
    let ty = plan
        .weight
        .and_then(weight_type)
        .ok_or_else(|| fail("no weight scheme".to_owned()))?;
    let (n_out, k_in) = (plan.dims[0] as usize, plan.dims[1] as usize);
    let rows = active_rows(l, OP)?;
    if rows == 0 || n_out == 0 {
        return Ok(());
    }
    let (Some(x), Some(y), Some(w), Some(ws)) =
        (l.input(0), l.output(0), l.weight(0), l.workspace())
    else {
        return Err(fail("missing operands".to_owned()));
    };
    let xs = rows_span(x, rows, k_in * 4, 4, OP)?;
    let ys = rows_span(y, rows, n_out * 4, 4, OP)?;
    let wbytes = (k_in / ty.block_elems() * ty.block_bytes()) * 2 * n_out;
    let (quant, total) = layout(ty, rows, k_in);
    if w.bytes < wbytes as u64 || ws.bytes < total as u64 || ws.addr == 0 {
        return Err(fail("an operand is shorter than the plan needs".to_owned()));
    }
    if !ws.addr.is_multiple_of(4) {
        return Err(fail("the workspace is not 4-byte aligned".to_owned()));
    }
    let (wspan, wss) = (
        HostSpan {
            addr: w.addr,
            bytes: wbytes as u64,
        },
        HostSpan {
            addr: ws.addr,
            bytes: total as u64,
        },
    );
    if ys.overlaps(&xs)
        || ys.overlaps(&wspan)
        || ys.overlaps(&wss)
        || wss.overlaps(&xs)
        || wss.overlaps(&wspan)
    {
        return Err(fail(
            "the output or the workspace overlaps another operand".to_owned(),
        ));
    }
    // SAFETY: every range was resolved inside a live host buffer or weight view (`Launch`'s
    // docs) and checked above to hold what is read or written, at the element's alignment
    // (the scratch starts `quant` bytes in, a multiple of 64, from a 4-byte-aligned base). The
    // output and the workspace overlap nothing else, and the quantised part and the scratch
    // are disjoint, so each mutable view is the only view of its bytes; the inputs are only
    // read.
    let (x, y, w, q, scratch) = unsafe {
        (
            std::slice::from_raw_parts(xs.addr as *const f32, rows * k_in),
            std::slice::from_raw_parts_mut(ys.addr as *mut f32, rows * n_out),
            std::slice::from_raw_parts(wspan.addr as *const u8, wbytes),
            std::slice::from_raw_parts_mut(ws.addr as *mut u8, quant),
            std::slice::from_raw_parts_mut(
                (ws.addr as usize + quant) as *mut f32,
                gate_up_scratch(rows),
            ),
        )
    };
    let wm = Weights::new(ty, w, 2 * n_out, k_in).map_err(|e| fail(e.to_string()))?;
    let need = workspace_bytes(ty.act(), rows, k_in);
    let view =
        quantize_into(ty.act(), x, rows, k_in, &mut q[..need]).map_err(|e| fail(e.to_string()))?;
    gate_up_act(level, &wm, &view, y, scratch).map_err(|e| fail(e.to_string()))
}
