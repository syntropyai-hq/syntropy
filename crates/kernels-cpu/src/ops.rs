// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU provider's other M0 ops: `GateUpAct`, `RowGather`, `Add`, `Mul` and `Convert`
//! (RFC-0040 §7.3; COMP-E3-T22). `Gemm` and `Logits` share the `Gemv` kernel
//! ([`crate::gemv`]'s entry).
//!
//! Each op computes the value `kernels::reference` defines, and its operands are the ggml
//! provider's mapping of the op (RFC-0040 §8.2):
//!
//! | `OpKind` | dims | operands | value |
//! |---|---|---|---|
//! | `GateUpAct { act: Silu }` | `[n_out, k_in]` | in `x [rows, k_in]`; weight `W [2·n_out, k_in]`, gate rows first, then up rows; out `[rows, n_out]` | `reference::gate_up_act`, with each product as `Gemv` computes it |
//! | `RowGather` | `[table_rows, width]` | in ids `[rows]` I32; weight `T [table_rows, width]`; out `[rows, width]` F32 | `reference::row_gather`, bit for bit |
//! | `Add`, `Mul` | `[n]` | in `a`, `b` `[rows, n]`; out `[rows, n]` | `reference::add`, `reference::mul` with `Operand::Rows`, bit for bit |
//! | `Convert` | `[n]` | in `x [rows, n]` of `schemes[0]`; out `[rows, n]` of the last scheme | `reference::convert` between F32, F16 and BF16, bit for bit |
//!
//! Every tensor is packed row-major, and every op computes the first `rows_active` rows (the
//! whole bucket without a step descriptor) and leaves the others untouched.
//!
//! **Levels.** Each op is in every level's dispatch table, at the level [`Level::for_isa`]
//! picks, so never above the live host. `GateUpAct`'s products are the GEMV kernels of that
//! level. `RowGather`, `Add`, `Mul` and `Convert` move or combine one element at a time, with
//! no reduction, so their value cannot depend on the level: each runs one body compiled for
//! the level's instructions (`at_level`: AVX2 and F16C at `X86V3`, AVX-512 at `X86V4`,
//! NEON with the dot-product instructions at `Arm82Dot` and `Arm86I8mm`), which the compiler
//! vectorises where it can. Rust never contracts a multiply and an add into a fused one, so
//! every element is one IEEE operation, rounded once, at every level.
//!
//! **Aliasing.** `Add` and `Mul` may write in place: the output may be exactly the first or
//! the second input. Any other overlap between an output and an operand is refused.

use syntropy_core::{AccumKind, ActQuant, Blake3, Bytes, DType, Numerics};
use syntropy_kernels::{Error, LayoutVersion, OpCaps, OpKey};

use crate::device::internal;
use crate::gemv::Level;
use crate::launch::{HostSpan, HostTensor, Launch, Plan};

pub(crate) mod convert;
pub(crate) mod elementwise;
pub(crate) mod gate_up;

#[cfg(test)]
mod tests;

pub use convert::{GatherError, convert_rows, row_gather, table_row_bytes};
pub use elementwise::{add, mul};
pub use gate_up::{GATE_UP_CHUNK, gate_up_act, gate_up_scratch};

/// The entry point of an op's kernel at `level`: a non-capturing closure per level, so the
/// dispatch table holds a plain function pointer.
macro_rules! run_for_level {
    ($run:path, $level:expr) => {{
        let f: $crate::launch::KernelFn = match $level {
            $crate::gemv::Level::Portable => |l| $run($crate::gemv::Level::Portable, l),
            #[cfg(target_arch = "x86_64")]
            $crate::gemv::Level::Avx2 => |l| $run($crate::gemv::Level::Avx2, l),
            #[cfg(target_arch = "x86_64")]
            $crate::gemv::Level::Avx512 => |l| $run($crate::gemv::Level::Avx512, l),
            #[cfg(target_arch = "aarch64")]
            $crate::gemv::Level::Dot => |l| $run($crate::gemv::Level::Dot, l),
            #[cfg(target_arch = "aarch64")]
            $crate::gemv::Level::I8mm => |l| $run($crate::gemv::Level::I8mm, l),
        };
        f
    }};
}
pub(crate) use run_for_level;

/// Runs `body` compiled for `level`'s instructions (module docs). The value never depends on
/// the level: `body` is the same code at each.
#[inline(always)]
pub(crate) fn at_level<R>(level: Level, body: impl FnOnce() -> R) -> R {
    match level {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: a level other than `Portable` comes only from `Level::for_isa`, which never
        // returns a level above what the live host supports, so the host has AVX2, FMA and F16C.
        Level::Avx2 => unsafe { x86_v3(body) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above; `Avx512` means the host has AVX-512 F, BW, VL and DQ, AVX2, FMA and
        // F16C.
        Level::Avx512 => unsafe { x86_v4(body) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: as above; `Dot` and `I8mm` mean the host has the dot-product instructions.
        Level::Dot | Level::I8mm => unsafe { arm_dot(body) },
        Level::Portable => body(),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
fn x86_v3<R>(body: impl FnOnce() -> R) -> R {
    body()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma,f16c")]
fn x86_v4<R>(body: impl FnOnce() -> R) -> R {
    body()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
fn arm_dot<R>(body: impl FnOnce() -> R) -> R {
    body()
}

/// Whether a key's layout and variant are the ones every op here takes.
pub(crate) fn plain_key(key: &OpKey) -> bool {
    key.variant == 0 && key.layout == LayoutVersion::SOURCE
}

/// The capabilities of an op that moves or combines elements one at a time: deterministic,
/// capture-safe, no accumulation beyond one F32 operation per element.
pub(crate) fn elementwise_caps(workspace: Bytes, out: DType) -> OpCaps {
    OpCaps {
        workspace,
        est: None,
        layout: LayoutVersion::SOURCE,
        capture_safe: true,
        deterministic: true,
        numerics: Numerics {
            accumulate: AccumKind::F32,
            act_in: ActQuant::None,
            out,
            softmax_f32: false,
            state_f32: false,
        },
        // As `Gemv`'s: the CPU kernels run uncertified until their certification.
        kernel_hash: Blake3([0; 32]),
        source_key: Blake3([0; 32]),
    }
}

/// The rows a launch computes: the step descriptor's `rows_active`, or the whole bucket.
pub(crate) fn active_rows(l: &Launch<'_>, op: &str) -> Result<usize, Error> {
    let plan: &Plan = l.plan();
    let rows = l.rows_active()?.unwrap_or(plan.rows);
    if rows > plan.rows {
        return Err(internal(format!("{op}: rows_active exceeds the bucket")));
    }
    Ok(rows as usize)
}

/// The first `rows · row_bytes` bytes of a resolved tensor, checked to lie inside it and to
/// start at a multiple of `align`.
pub(crate) fn rows_span(
    t: HostTensor,
    rows: usize,
    row_bytes: usize,
    align: u64,
    op: &str,
) -> Result<HostSpan, Error> {
    let bytes = (rows * row_bytes) as u64;
    if bytes > t.bytes || (bytes > 0 && (t.addr == 0 || !t.addr.is_multiple_of(align))) {
        return Err(internal(format!(
            "{op}: an operand is shorter than the plan needs, or misaligned"
        )));
    }
    Ok(HostSpan {
        addr: t.addr,
        bytes,
    })
}
