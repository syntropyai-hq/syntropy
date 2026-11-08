// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `MoeGrouped` op at the `CPU` site on the CPU provider: the [`Kernel`] each level's
//! dispatch table holds (COMP-E3-T05).
//!
//! Operands:
//!
//! - `OpKey::schemes[0]` is the experts' weight scheme, GGUF `Q4_K` or `Q8_0`, for the gate, up
//!   and down matrices alike. Any further scheme is `Plain(F32)`. The layout is `SOURCE` and no
//!   fusion bit is set.
//! - The shape's dims are `[n_experts, top_k, hidden, inter]` (RFC-0040 §3.2), with `hidden`
//!   and `inter` whole blocks of the scheme.
//! - Input 0 is `x`, the normalised rows, packed F32 `[rows, hidden]`; input 1 is `ids`, the
//!   router's choices, packed I32 `[rows, top_k]`. An optional input 2 is `site`, packed I32
//!   `[n_experts, 1]`: a nonzero entry marks an expert this launch computes. Without it every
//!   expert is this site's, as on a CPU-only stage (§3.4). Output 0 is the per-(row, rank)
//!   block, packed F32 `[rows, top_k, hidden]`.
//! - Weights 0, 1 and 2 are the experts' gate, up and down matrices, each stacked by expert in
//!   GGUF's own layout (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`): expert `e`'s matrix
//!   is the `e`-th run of `[inter, hidden]` (gate, up) or `[hidden, inter]` (down) rows. No
//!   constants.
//! - The workspace is [`workspace_bytes`](super::workspace_bytes), in whole cache lines.
//!
//! A launch computes the first `rows_active` rows (the whole bucket without a step descriptor)
//! on the calling thread ([`moe_grouped`]). A slot whose expert `site` leaves out keeps its
//! value. Experts addressed per slot through `slot_addr` and the doorbell's assignment are the
//! CPU pools' integration, which binds this kernel per step.

use syntropy_core::{AccumKind, ActQuant, Blake3, Bytes, DType, GgmlType, Numerics, QuantScheme};
use syntropy_kernels::{
    CpuIsa, Error, LayoutVersion, OpArgs, OpCaps, OpKey, OpKind, ShapeSig, TensorArg,
};

use super::{Expert, MoeShape, moe_grouped, workspace_bytes};
use crate::device::internal;
use crate::gemv::{Level, WeightType, Weights};
use crate::launch::{HostSpan, HostTensor, Kernel, KernelFn, Launch, Plan};

/// The `MoeGrouped` kernel for `isa`'s level, lowered to what the live host runs.
pub(crate) fn kernel(isa: CpuIsa) -> Kernel {
    Kernel {
        plan,
        caps,
        check,
        run: run_for(Level::for_isa(isa)),
    }
}

/// The weight type of a scheme this kernel takes: the task's two schemes.
fn weight_type(s: QuantScheme) -> Option<WeightType> {
    match s {
        QuantScheme::Ggml(GgmlType::Q4_K) => Some(WeightType::Q4K),
        QuantScheme::Ggml(GgmlType::Q8_0) => Some(WeightType::Q8_0),
        _ => None,
    }
}

/// The dims as a [`MoeShape`].
fn moe_shape(dims: &[u32]) -> Option<MoeShape> {
    let [n_experts, top_k, hidden, inter] = dims[..] else {
        return None;
    };
    Some(MoeShape {
        n_experts: n_experts as usize,
        top_k: top_k as usize,
        hidden: hidden as usize,
        inter: inter as usize,
    })
}

/// `(weight type, shape)` of a key and shape this kernel takes.
fn shape_of(key: &OpKey, shape: &ShapeSig) -> Option<(WeightType, MoeShape)> {
    if key.op != OpKind::MoeGrouped || key.variant != 0 || key.layout != LayoutVersion::SOURCE {
        return None;
    }
    let (&weight, rest) = key.schemes.split_first()?;
    if rest.len() > 2 || rest.iter().any(|&s| s != QuantScheme::Plain(DType::F32)) {
        return None;
    }
    let ty = weight_type(weight)?;
    let m = moe_shape(&shape.dims)?;
    let whole =
        m.hidden.is_multiple_of(ty.block_elems()) && m.inter.is_multiple_of(ty.block_elems());
    (whole && m.n_experts > 0 && m.top_k > 0 && m.top_k <= m.n_experts).then_some((ty, m))
}

fn plan(key: &OpKey, shape: &ShapeSig) -> Option<Plan> {
    let (_, m) = shape_of(key, shape)?;
    let ws = workspace_bytes(&m, shape.rows as usize);
    Some(Plan {
        // Whole cache lines, as `Gemv`'s.
        workspace: Bytes(ws.next_multiple_of(64) as u64),
        ..Plan::for_key(key, shape)
    })
}

fn caps(key: &OpKey, shape: &ShapeSig) -> Option<OpCaps> {
    let (ty, _) = shape_of(key, shape)?;
    let plan = plan(key, shape)?;
    let (max_block, act_in) = match ty {
        WeightType::Q4K => (256, ActQuant::Q8K256),
        _ => (32, ActQuant::Q8Block32),
    };
    Some(OpCaps {
        workspace: plan.workspace,
        est: None,
        layout: LayoutVersion::SOURCE,
        // A launch is one call on the stream's thread, with no allocation and no wait.
        capture_safe: true,
        // Each output's reduction order is fixed by the shape alone, whichever rows share its
        // expert (the module docs of `moe`).
        deterministic: true,
        numerics: Numerics {
            accumulate: AccumKind::I32ThenF32 { max_block },
            act_in,
            out: DType::F32,
            softmax_f32: false,
            state_f32: false,
        },
        // Uncertified, as `Gemv`'s (RFC-0040 §10.2).
        kernel_hash: Blake3([0; 32]),
        source_key: Blake3([0; 32]),
    })
}

/// A packed tensor of `dtype` with `dims` (outermost first, unused trailing dims 1) at the
/// slot's own buffer.
fn packed(t: &TensorArg, dtype: DType, dims: [usize; 3]) -> bool {
    let [Ok(a), Ok(b), Ok(c)] = dims.map(u32::try_from) else {
        return false;
    };
    TensorArg::packed(t.buf, dtype, [a, b, c, 1]).as_ref() == Some(t)
}

/// Bytes of one expert's gate (or up) matrix and of its down matrix.
fn matrix_bytes(ty: WeightType, m: &MoeShape) -> (usize, usize) {
    let row = |k: usize| k / ty.block_elems() * ty.block_bytes();
    (m.inter * row(m.hidden), m.hidden * row(m.inter))
}

fn check(plan: &Plan, args: &OpArgs<'_>) -> Result<(), Error> {
    let ty = plan
        .weight
        .and_then(weight_type)
        .ok_or_else(|| internal("MoeGrouped: no weight scheme in the plan".to_owned()))?;
    let m = moe_shape(&plan.dims[..4])
        .ok_or_else(|| internal("MoeGrouped: no shape in the plan".to_owned()))?;
    let rows = plan.rows as usize;
    let ([x, ids, site @ ..], [out], [wg, wu, wd], []) =
        (args.inputs, args.outputs, args.weights, args.statics)
    else {
        return Err(internal(
            "MoeGrouped binds x, ids and an optional site mask, one output, three weights and \
             no constants"
                .to_owned(),
        ));
    };
    let site_ok = match site {
        [] => true,
        [s] => packed(s, DType::I32, [m.n_experts, 1, 1]),
        _ => false,
    };
    if !packed(x, DType::F32, [rows, m.hidden, 1])
        || !packed(ids, DType::I32, [rows, m.top_k, 1])
        || !packed(out, DType::F32, [rows, m.top_k, m.hidden])
        || !site_ok
    {
        return Err(internal(format!(
            "MoeGrouped: x must be packed F32 [{rows}, {}], ids packed I32 [{rows}, {}], the \
             site mask packed I32 [{}, 1] and the output packed F32 [{rows}, {}, {}]",
            m.hidden, m.top_k, m.n_experts, m.top_k, m.hidden
        )));
    }
    let (gu, d) = matrix_bytes(ty, &m);
    let (gu, d) = ((gu * m.n_experts) as u64, (d * m.n_experts) as u64);
    if wg.len.0 < gu || wu.len.0 < gu || wd.len.0 < d {
        return Err(internal(format!(
            "MoeGrouped: the weight views hold {}, {} and {} bytes, the experts {gu}, {gu} and {d}",
            wg.len.0, wu.len.0, wd.len.0
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

/// The first `bytes` of `t`, checked to lie inside it and to start at `align`.
fn span(t: &HostTensor, bytes: usize, align: u64) -> Option<HostSpan> {
    (t.addr.is_multiple_of(align) && bytes as u64 <= t.bytes).then_some(HostSpan {
        addr: t.addr,
        bytes: bytes as u64,
    })
}

/// Runs one launch at `level`.
fn run(level: Level, l: &Launch<'_>) -> Result<(), Error> {
    let plan = l.plan();
    let fail = |what: &str| internal(format!("MoeGrouped: {what}"));
    let ty = plan
        .weight
        .and_then(weight_type)
        .ok_or_else(|| fail("no weight scheme"))?;
    let m = moe_shape(&plan.dims[..4]).ok_or_else(|| fail("no shape"))?;
    let rows = l.rows_active()?.unwrap_or(plan.rows);
    if rows > plan.rows {
        return Err(fail("rows_active exceeds the bucket"));
    }
    if rows == 0 {
        return Ok(());
    }
    let rows = rows as usize;
    let (Some(x), Some(ids), Some(out)) = (l.input(0), l.input(1), l.output(0)) else {
        return Err(fail("missing operands"));
    };
    let (Some(wg), Some(wu), Some(wd)) = (l.weight(0), l.weight(1), l.weight(2)) else {
        return Err(fail("missing weights"));
    };
    let site = l.input(2);
    let (gu_bytes, d_bytes) = matrix_bytes(ty, &m);
    let ws_need = workspace_bytes(&m, rows);
    let ws = l.workspace().unwrap_or(HostSpan { addr: 0, bytes: 0 });
    let (Some(xs), Some(is), Some(os)) = (
        span(&x, rows * m.hidden * 4, 4),
        span(&ids, rows * m.top_k * 4, 4),
        span(&out, rows * m.top_k * m.hidden * 4, 4),
    ) else {
        return Err(fail(
            "an input or the output is shorter than the plan needs, or unaligned",
        ));
    };
    let ss = match site {
        None => None,
        Some(t) => Some(span(&t, m.n_experts * 4, 4).ok_or_else(|| fail("a short site mask"))?),
    };
    let weights = [
        (wg, gu_bytes * m.n_experts),
        (wu, gu_bytes * m.n_experts),
        (wd, d_bytes * m.n_experts),
    ];
    if weights.iter().any(|(w, need)| w.bytes < *need as u64) {
        return Err(fail("a weight view is shorter than the experts"));
    }
    if ws.addr == 0 || ws.bytes < ws_need as u64 {
        return Err(fail("the workspace is shorter than the plan needs"));
    }
    let ws = HostSpan {
        addr: ws.addr,
        bytes: ws_need as u64,
    };
    let wspans = weights.map(|(w, need)| HostSpan {
        addr: w.addr,
        bytes: need as u64,
    });
    let reads = [
        Some(xs),
        Some(is),
        ss,
        Some(wspans[0]),
        Some(wspans[1]),
        Some(wspans[2]),
    ];
    for written in [os, ws] {
        if reads.iter().flatten().any(|r| r.overlaps(&written)) {
            return Err(fail("the output or the workspace overlaps another operand"));
        }
    }
    if os.overlaps(&ws) {
        return Err(fail("the output overlaps the workspace"));
    }
    // SAFETY: every range was resolved inside a live host buffer or weight view (`Launch`'s
    // docs) and checked above to hold what is read or written, at the element's alignment.
    // The output and the workspace overlap nothing else, so the one mutable view of each is
    // the only view of its bytes; the inputs and weights are only read.
    let (x, ids, out, ws, site, [wg, wu, wd]) = unsafe {
        (
            std::slice::from_raw_parts(xs.addr as *const f32, rows * m.hidden),
            std::slice::from_raw_parts(is.addr as *const i32, rows * m.top_k),
            std::slice::from_raw_parts_mut(os.addr as *mut f32, rows * m.top_k * m.hidden),
            std::slice::from_raw_parts_mut(ws.addr as *mut u8, ws_need),
            ss.map(|s| std::slice::from_raw_parts(s.addr as *const i32, m.n_experts)),
            wspans.map(|w| std::slice::from_raw_parts(w.addr as *const u8, w.bytes as usize)),
        )
    };
    let expert = |e: u32| -> Option<Expert<'_>> {
        let e = e as usize;
        if site.is_some_and(|s| s[e] == 0) {
            return None;
        }
        let gu = e * gu_bytes..(e + 1) * gu_bytes;
        let dn = e * d_bytes..(e + 1) * d_bytes;
        let gate = Weights::new(ty, &wg[gu.clone()], m.inter, m.hidden).ok()?;
        let up = Weights::new(ty, &wu[gu], m.inter, m.hidden).ok()?;
        let down = Weights::new(ty, &wd[dn], m.hidden, m.inter).ok()?;
        Expert::new(gate, up, down).ok()
    };
    moe_grouped(level, &m, expert, x, ids, rows, out, ws, None).map_err(|e| internal(e.to_string()))
}
