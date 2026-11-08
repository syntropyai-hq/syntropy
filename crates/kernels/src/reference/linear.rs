// SPDX-License-Identifier: MIT OR Apache-2.0

//! The weight-bound linear ops' reference semantics: `Gemv`, `Gemm`, `GateUpAct` and `Logits`
//! (RFC-0040 §3.1, §3.3; RFC-0042 §4.1, §4.2; COMP-E1-T03).
//!
//! Every weight is an F32 [`Matrix`] `[n_out, k_in]`, dequantised exactly from the pack by
//! RFC-0042 §2's reference dequantisers ([`dequant`](super::dequant)). Every output value is one
//! [`reduce::dot`] of a weight row with an activation row: products rounded once to F32, summed
//! in the `F32Fixed` order ([`reduce`]). The output is F32.
//!
//! - **`Gemv` and `Gemm`** compute `y[r][o] = Σ_k W[o][k] · x[r][k]` for every active row `r`.
//!   They are one function: the kinds differ in the row bucket a provider prepares (`Gemv` up to
//!   16 rows, `Gemm` above), never in the value.
//! - **Fusions** (`FUSE_BIAS`, `FUSE_RESIDUAL`, `FUSE_SCALE_OUT`, RFC-0040 §3.1) are defined as
//!   the unfused op followed by [`add`](super::elementwise::add) or
//!   [`mul`](super::elementwise::mul) on its rounded output: one more rounding each, in the order
//!   the fusion bits are listed.
//! - **`GateUpAct`** computes `y[r][i] = act(Σ_k G[i][k] · x[r][k]) · (Σ_k U[i][k] · x[r][k])`:
//!   the gate and up products are separate dots, the activation is applied to the rounded gate,
//!   and the product is rounded once. M0 defines SiLU gating ([`math::silu`]); GELU and the
//!   clamped SiLU of gpt-oss-class experts are COMP-E1-T04's, and are refused until then.
//! - **`Logits`** is the final norm and the LM head: each active row is normalised as
//!   [`rms_norm`](super::norm::rms_norm) defines it, then multiplied by the head as `Gemv`.
//!
//! Edge cases. With `rows_active = 0` nothing is written. Rows past `rows_active` are neither read
//! nor written. With `k_in = 0` every output is the empty sum, `+0.0`. A NaN in a weight row or an
//! activation row makes that output NaN (one NaN term makes the sum NaN), as does `0 · ∞`; NaN is
//! never clamped (RFC-0042 §4.1 rule 6).

use super::math;
use super::norm::Norm;
use super::reduce;
use super::shape::{Matrix, OpError, at_least};
use crate::op::GatedAct;

/// One projection of the first `rows_active` rows of `x` (`k_in` values each) into `y` (`n_out`
/// values each), as `Gemv` and `Gemm` define it (module docs).
fn project(
    op: &'static str,
    w: Matrix<'_>,
    x: &[f32],
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    let (n_out, k_in) = (w.rows(), w.cols());
    at_least(op, "x", x.len(), rows_active, k_in)?;
    at_least(op, "y", y.len(), rows_active, n_out)?;
    for r in 0..rows_active {
        let x = &x[r * k_in..(r + 1) * k_in];
        for (o, y) in y[r * n_out..(r + 1) * n_out].iter_mut().enumerate() {
            *y = reduce::dot(w.row(o), x);
        }
    }
    Ok(())
}

/// `Gemv`: `y[r][o] = Σ_k w[o][k] · x[r][k]` for `r < rows_active` (module docs).
///
/// # Errors
///
/// [`OpError::Len`] when `x` or `y` holds fewer than `rows_active` rows.
pub fn gemv(w: Matrix<'_>, x: &[f32], rows_active: usize, y: &mut [f32]) -> Result<(), OpError> {
    project("Gemv", w, x, rows_active, y)
}

/// `Gemm`: [`gemv`]'s definition, for a bucket above 16 rows (module docs).
///
/// # Errors
///
/// As [`gemv`].
pub fn gemm(w: Matrix<'_>, x: &[f32], rows_active: usize, y: &mut [f32]) -> Result<(), OpError> {
    project("Gemm", w, x, rows_active, y)
}

/// `GateUpAct`: `y[r][i] = act(gate[i] · x[r]) · (up[i] · x[r])` for `r < rows_active`, `gate`
/// and `up` both `[inter, hidden]` (module docs).
///
/// # Errors
///
/// [`OpError::ShapeMismatch`] when `gate` and `up` differ in shape; [`OpError::Len`] when `x` or
/// `y` holds fewer than `rows_active` rows; [`OpError::Unsupported`] for GELU and clamped SiLU
/// gating, which COMP-E1-T04 defines.
pub fn gate_up_act(
    act: GatedAct,
    gate: Matrix<'_>,
    up: Matrix<'_>,
    x: &[f32],
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    const OP: &str = "GateUpAct";
    let act: fn(f32) -> f32 = match act {
        GatedAct::Silu => math::silu,
        GatedAct::Gelu => {
            return Err(OpError::Unsupported {
                op: OP,
                what: "GELU gating (COMP-E1-T04)",
            });
        }
        GatedAct::ClampedSilu => {
            return Err(OpError::Unsupported {
                op: OP,
                what: "clamped SiLU gating (COMP-E1-T04)",
            });
        }
    };
    if (gate.rows(), gate.cols()) != (up.rows(), up.cols()) {
        return Err(OpError::ShapeMismatch { op: OP });
    }
    let (inter, hidden) = (gate.rows(), gate.cols());
    at_least(OP, "x", x.len(), rows_active, hidden)?;
    at_least(OP, "y", y.len(), rows_active, inter)?;
    for r in 0..rows_active {
        let x = &x[r * hidden..(r + 1) * hidden];
        for (i, y) in y[r * inter..(r + 1) * inter].iter_mut().enumerate() {
            let g = reduce::dot(gate.row(i), x);
            let u = reduce::dot(up.row(i), x);
            *y = act(g) * u;
        }
    }
    Ok(())
}

/// `Logits`: each of the first `rows_active` rows of `x` normalised by `norm`
/// ([`rms_norm`](super::norm::rms_norm)), then multiplied by `head` (`[vocab, hidden]`) into `y`
/// (`vocab` values per row). `scratch` holds one normalised row and is overwritten.
///
/// # Errors
///
/// [`OpError::Len`] when `norm`'s width is not `head`'s columns, when `x` or `y` holds fewer than
/// `rows_active` rows, or when `scratch` is shorter than a row.
pub fn logits(
    norm: Norm<'_>,
    head: Matrix<'_>,
    x: &[f32],
    rows_active: usize,
    scratch: &mut [f32],
    y: &mut [f32],
) -> Result<(), OpError> {
    const OP: &str = "Logits";
    let (vocab, hidden) = (head.rows(), head.cols());
    if norm.width() != hidden {
        return Err(OpError::Len {
            op: OP,
            arg: "norm.weight",
            len: norm.width(),
            need: hidden,
        });
    }
    at_least(OP, "x", x.len(), rows_active, hidden)?;
    at_least(OP, "y", y.len(), rows_active, vocab)?;
    at_least(OP, "scratch", scratch.len(), 1, hidden)?;
    let h = &mut scratch[..hidden];
    for r in 0..rows_active {
        norm.row(&x[r * hidden..(r + 1) * hidden], h);
        for (o, y) in y[r * vocab..(r + 1) * vocab].iter_mut().enumerate() {
            *y = reduce::dot(head.row(o), h);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
