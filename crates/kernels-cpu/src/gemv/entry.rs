// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Gemv` op on the CPU provider: the [`Kernel`] each level's dispatch table holds.
//!
//! Operands, as the ggml provider's `Gemv` mapping takes them (RFC-0040 §8.2):
//!
//! - `OpKey::schemes[0]` is the weight scheme: `Plain(F16)`, `Plain(BF16)`, or GGUF `F16`,
//!   `BF16`, `Q8_0`, `Q4_0` or `Q4_K`. Any further scheme (the activation's, the output's) is
//!   `Plain(F32)`. The layout is `SOURCE` and no fusion bit is set.
//! - The shape's dims are `[n_out, k_in]`, with `k_in` whole blocks of the weight type.
//! - Input 0 is `x`, a packed F32 `[rows, k_in]` tensor at the bucket's rows; output 0 is `y`,
//!   packed F32 `[rows, n_out]`; weight 0 is the matrix, `n_out` rows of `k_in` weights in GGUF
//!   block layout. No constants.
//! - The workspace holds the quantised activation rows ([`workspace_bytes`]).
//!
//! A launch computes the first `rows_active` rows (the whole bucket without a step descriptor)
//! and every output row, on the calling thread. Splitting the output rows across a NUMA-local
//! pool ([`partition`](super::partition)) is the pool's (PLC-E7-T07).

use syntropy_core::{AccumKind, Blake3, Bytes, DType, Numerics, QuantScheme};
use syntropy_kernels::{
    CpuIsa, Error, LayoutVersion, OpArgs, OpCaps, OpKey, OpKind, ShapeSig, TensorArg,
};

use super::act::{ActKind, quantize_into};
use super::{Level, WeightType, Weights, gemv, workspace_bytes};
use crate::device::internal;
use crate::launch::{HostSpan, Kernel, KernelFn, Launch, Plan};

/// The `Gemv` kernel for `isa`'s level, lowered to what the live host runs.
pub(crate) fn kernel(isa: CpuIsa) -> Kernel {
    Kernel {
        plan,
        caps,
        check,
        run: run_for(Level::for_isa(isa)),
    }
}

/// The weight type of a scheme this kernel takes.
fn weight_type(s: QuantScheme) -> Option<WeightType> {
    match s {
        QuantScheme::Plain(DType::F16) => Some(WeightType::F16),
        QuantScheme::Plain(DType::BF16) => Some(WeightType::Bf16),
        QuantScheme::Ggml(g) => WeightType::from_ggml(g),
        _ => None,
    }
}

/// `(weight type, n_out, k_in)` of a key and shape this kernel takes.
fn shape_of(key: &OpKey, shape: &ShapeSig) -> Option<(WeightType, usize, usize)> {
    if key.op != OpKind::Gemv || key.variant != 0 || key.layout != LayoutVersion::SOURCE {
        return None;
    }
    let (&weight, rest) = key.schemes.split_first()?;
    if rest.len() > 2 || rest.iter().any(|&s| s != QuantScheme::Plain(DType::F32)) {
        return None;
    }
    let ty = weight_type(weight)?;
    let [n_out, k_in] = shape.dims[..] else {
        return None;
    };
    let (n_out, k_in) = (n_out as usize, k_in as usize);
    k_in.is_multiple_of(ty.block_elems())
        .then_some((ty, n_out, k_in))
}

fn plan(key: &OpKey, shape: &ShapeSig) -> Option<Plan> {
    let (ty, _, k_in) = shape_of(key, shape)?;
    let ws = workspace_bytes(ty.act(), shape.rows as usize, k_in);
    Some(Plan {
        // Whole cache lines, so a workspace carved from a larger region keeps its neighbours'
        // alignment.
        workspace: Bytes(ws.next_multiple_of(64) as u64),
        ..Plan::for_key(key, shape)
    })
}

fn caps(key: &OpKey, shape: &ShapeSig) -> Option<OpCaps> {
    let (ty, _, _) = shape_of(key, shape)?;
    let plan = plan(key, shape)?;
    let (accumulate, act_in) = match ty.act() {
        ActKind::F32 => (AccumKind::F32, syntropy_core::ActQuant::None),
        // The `Q8_0` and `Q4_0` sums run over one 32-element block, `Q4_K`'s over one
        // 256-element super-block (the eight sub-block sums and their scales).
        ActKind::Q8_0 => (
            AccumKind::I32ThenF32 { max_block: 32 },
            syntropy_core::ActQuant::Q8Block32,
        ),
        ActKind::Q8K => (
            AccumKind::I32ThenF32 { max_block: 256 },
            syntropy_core::ActQuant::Q8K256,
        ),
    };
    Some(OpCaps {
        workspace: plan.workspace,
        est: None,
        layout: LayoutVersion::SOURCE,
        // A launch is one call on the stream's thread, with no allocation and no wait.
        capture_safe: true,
        // Each output's reduction order is fixed by the shape alone (the module docs of
        // `gemv`), at every level and for every row partition.
        deterministic: true,
        numerics: Numerics {
            accumulate,
            act_in,
            out: DType::F32,
            softmax_f32: false,
            state_f32: false,
        },
        // Zero is never certified: the CPU kernels' `OpCert` hashes come with their
        // certification (RFC-0040 §10.2), so until then they run uncertified.
        kernel_hash: Blake3([0; 32]),
        source_key: Blake3([0; 32]),
    })
}

/// The packed F32 `[rows, n]` tensor a slot must be, at the slot's own buffer.
fn packed(a: &TensorArg, rows: u32, n: usize) -> bool {
    u32::try_from(n)
        .is_ok_and(|n| TensorArg::packed(a.buf, DType::F32, [rows, n, 1, 1]).as_ref() == Some(a))
}

fn check(plan: &Plan, args: &OpArgs<'_>) -> Result<(), Error> {
    let ty = plan
        .weight
        .and_then(weight_type)
        .ok_or_else(|| internal("Gemv: no weight scheme in the plan".to_owned()))?;
    let (n_out, k_in) = (plan.dims[0] as usize, plan.dims[1] as usize);
    let ([x], [y], [w], []) = (args.inputs, args.outputs, args.weights, args.statics) else {
        return Err(internal(
            "Gemv binds one input, one output, one weight and no constants".to_owned(),
        ));
    };
    if !packed(x, plan.rows, k_in) || !packed(y, plan.rows, n_out) {
        return Err(internal(format!(
            "Gemv: x must be packed F32 [{}, {k_in}] and y packed F32 [{}, {n_out}]",
            plan.rows, plan.rows
        )));
    }
    let need = (k_in / ty.block_elems() * ty.block_bytes()) as u64 * n_out as u64;
    if w.len.0 < need {
        return Err(internal(format!(
            "Gemv: the weight view holds {} bytes, the matrix {need}",
            w.len.0
        )));
    }
    Ok(())
}

/// The entry point for `level`.
fn run_for(level: Level) -> KernelFn {
    match level {
        Level::Portable => |l| run(Level::Portable, l),
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 => |l| run(Level::Avx2, l),
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 => |l| run(Level::Avx512, l),
        #[cfg(target_arch = "aarch64")]
        Level::Dot => |l| run(Level::Dot, l),
        #[cfg(target_arch = "aarch64")]
        Level::I8mm => |l| run(Level::I8mm, l),
    }
}

/// Runs one launch at `level`.
fn run(level: Level, l: &Launch<'_>) -> Result<(), Error> {
    let plan = l.plan();
    let fail = |what: &str| internal(format!("Gemv: {what}"));
    let ty = plan
        .weight
        .and_then(weight_type)
        .ok_or_else(|| fail("no weight scheme"))?;
    let (n_out, k_in) = (plan.dims[0] as usize, plan.dims[1] as usize);
    let rows = l.rows_active()?.unwrap_or(plan.rows);
    if rows > plan.rows {
        return Err(fail("rows_active exceeds the bucket"));
    }
    if rows == 0 || n_out == 0 {
        return Ok(());
    }
    let rows = rows as usize;
    let (Some(x), Some(y), Some(w)) = (l.input(0), l.output(0), l.weight(0)) else {
        return Err(fail("missing operands"));
    };
    let wbytes = (k_in / ty.block_elems() * ty.block_bytes()) * n_out;
    let ws = l.workspace().unwrap_or(HostSpan { addr: 0, bytes: 0 });
    let ws_need = workspace_bytes(ty.act(), rows, k_in);
    let (xs, ys) = (
        HostSpan {
            addr: x.addr,
            bytes: (rows * k_in * 4) as u64,
        },
        HostSpan {
            addr: y.addr,
            bytes: (rows * n_out * 4) as u64,
        },
    );
    let ws = HostSpan {
        addr: ws.addr,
        bytes: ws.bytes.min(ws_need as u64),
    };
    let ws_addr_ok = ws_need == 0 || ws.addr != 0;
    if !x.addr.is_multiple_of(4) || !y.addr.is_multiple_of(4) || !ws_addr_ok {
        return Err(fail(
            "x, y and the workspace must be 4-byte aligned host memory",
        ));
    }
    if w.bytes < wbytes as u64
        || ws.bytes < ws_need as u64
        || xs.bytes > x.bytes
        || ys.bytes > y.bytes
    {
        return Err(fail("an operand is shorter than the plan needs"));
    }
    let wspan = HostSpan {
        addr: w.addr,
        bytes: wbytes as u64,
    };
    if ys.overlaps(&xs)
        || ys.overlaps(&wspan)
        || ys.overlaps(&ws)
        || ws.overlaps(&xs)
        || ws.overlaps(&wspan)
    {
        return Err(fail("the output or the workspace overlaps another operand"));
    }
    // SAFETY: every range was resolved inside a live host buffer or weight view (`Launch`'s
    // docs) and checked above to hold what is read or written, at the element's alignment.
    // The output and the workspace overlap nothing else, so the one mutable view of each is
    // the only view of its bytes; the inputs are only read.
    let (x, y, w, ws) = unsafe {
        (
            std::slice::from_raw_parts(xs.addr as *const f32, rows * k_in),
            std::slice::from_raw_parts_mut(ys.addr as *mut f32, rows * n_out),
            std::slice::from_raw_parts(wspan.addr as *const u8, wbytes),
            if ws_need == 0 {
                &mut [][..]
            } else {
                std::slice::from_raw_parts_mut(ws.addr as *mut u8, ws_need)
            },
        )
    };
    let wm = Weights::new(ty, w, n_out, k_in).map_err(|e| internal(e.to_string()))?;
    let view = quantize_into(ty.act(), x, rows, k_in, ws).map_err(|e| internal(e.to_string()))?;
    gemv(level, &wm, &view, 0..n_out, y).map_err(|e| internal(e.to_string()))
}
