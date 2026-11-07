// SPDX-License-Identifier: MIT OR Apache-2.0

//! The MoE ops' reference semantics: `MoeRoute` (softmax top-k), `MoeGrouped` and `MoeCombine`
//! (RFC-0040 §3.1, §3.4; RFC-0042 §4.2; RFC-0060 §4.1; COMP-E1-T18).
//!
//! - **`MoeRoute { norm: Softmax }`** takes each active row's router logits, one per expert,
//!   computed by the router's `Gemv` before it. With `m` the row max, `p[e] =
//!   fl(exp(l[e] − m) / Z)` and `Z = Σ exp(l[e] − m)` in the `F32Fixed` order. The `k` selected
//!   experts are the `k` largest `p`, ranked by `p` with ties to the **lower expert index**
//!   (RFC-0042 §4.2; RFC-0060 §4.1), written in rank order with their weights. With
//!   `renormalise` (Qwen3-MoE's `norm_topk_prob`), each weight is divided by the `F32Fixed` sum
//!   of the `k` selected `p` in rank order. Sigmoid scores, bias and grouped top-k are
//!   COMP-E3-T23's routing options and are refused here.
//! - **`MoeGrouped`** runs, for each active row `r` and each of its `k` slots `j`, the FFN of
//!   expert `e = ids[r][j]`: `y[r][j] = down_e · h` with `h = gate_up_act(gate_e, up_e, x[r])`,
//!   i.e. [`gate_up_act`] and then [`gemv`],
//!   each rounded as those functions define. The output is per (row, slot), F32. Shared experts
//!   are the same computation over the shared experts' weights, with every row listing every
//!   shared expert. Which site computes an expert (RFC-0040 §3.4's split) never changes its
//!   value here.
//! - **`MoeCombine`** sums, per row and column, the routed experts' outputs weighted by their
//!   router weights **in ascending expert index**, then the shared experts' outputs in their
//!   listed order, unweighted (RFC-0040 §3.4). Each weighted term is rounded once, and the
//!   terms are added in the `F32Fixed` order, which is sequential for up to 32 terms. The sum
//!   therefore does not depend on the slot order the router wrote or on where each expert ran.
//!   Two slots with one expert are added in slot order. The residual is added by a separate
//!   `Add`.
//!
//! Edge cases. `rows_active = 0` writes nothing, and `k = 0` gives an empty selection and an
//! all-`+0` combine. A NaN or `+∞` router logit makes `Z` NaN, so every `p` is NaN: no `p`
//! ranks above another, the selection falls to the lowest expert indices, and every weight is
//! NaN, which reaches the combined output. A row whose router logits are all `−∞` gives the
//! same result, since `l − m` is then `−∞ − −∞`, NaN. A NaN in an expert's output or in a weight
//! makes that row and column of the combine NaN.

use super::linear::{gate_up_act, gemv};
use super::math;
use super::reduce;
use super::shape::{Matrix, OpError, at_least};
use crate::op::{GatedAct, RouteNorm};

/// `MoeRoute` with `norm` over the first `rows_active` rows of `logits` (`n_experts` values
/// each): row `r`'s `k` experts into `ids[r·k..(r+1)·k]` and their weights into `weights`
/// (module docs).
///
/// # Errors
///
/// [`OpError::Unsupported`] for sigmoid scores; [`OpError::Len`] when `k > n_experts` or a
/// buffer is short.
#[allow(
    clippy::too_many_arguments,
    reason = "the op's operands and dims, as RFC-0040 §3.4 lists them"
)]
pub fn moe_route(
    norm: RouteNorm,
    k: usize,
    renormalise: bool,
    n_experts: usize,
    logits: &[f32],
    rows_active: usize,
    ids: &mut [u32],
    weights: &mut [f32],
) -> Result<(), OpError> {
    const OP: &str = "MoeRoute";
    if norm != RouteNorm::Softmax {
        return Err(OpError::Unsupported {
            op: OP,
            what: "sigmoid routing (COMP-E3-T23)",
        });
    }
    if k > n_experts {
        return Err(OpError::Len {
            op: OP,
            arg: "k",
            len: k,
            need: n_experts,
        });
    }
    at_least(OP, "logits", logits.len(), rows_active, n_experts)?;
    at_least(OP, "ids", ids.len(), rows_active, k)?;
    at_least(OP, "weights", weights.len(), rows_active, k)?;
    let mut p = vec![0.0f32; n_experts];
    let mut taken = vec![false; n_experts];
    for r in 0..rows_active {
        let l = &logits[r * n_experts..(r + 1) * n_experts];
        let m = l.iter().copied().fold(f32::NEG_INFINITY, |a, b| {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                a.max(b)
            }
        });
        let z = reduce::sum_by(n_experts, |e| math::exp(l[e] - m));
        for (p, &l) in p.iter_mut().zip(l) {
            *p = math::exp(l - m) / z;
        }
        taken.fill(false);
        let (ids, weights) = (
            &mut ids[r * k..(r + 1) * k],
            &mut weights[r * k..(r + 1) * k],
        );
        for j in 0..k {
            // The best expert not yet taken: strictly larger `p` wins, so ties (and NaN, which
            // compares false) go to the lower index.
            let mut best = (0..n_experts).find(|&e| !taken[e]).unwrap_or(0);
            for e in best + 1..n_experts {
                if !taken[e] && p[e] > p[best] {
                    best = e;
                }
            }
            taken[best] = true;
            ids[j] = best as u32;
            weights[j] = p[best];
        }
        if renormalise {
            let s = reduce::sum(weights);
            for w in weights.iter_mut() {
                *w /= s;
            }
        }
    }
    Ok(())
}

/// One expert's FFN weights, dequantised exactly: `gate` and `up` `[inter, hidden]`, `down`
/// `[hidden, inter]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Expert<'a> {
    /// The gate projection.
    pub gate: Matrix<'a>,
    /// The up projection.
    pub up: Matrix<'a>,
    /// The down projection.
    pub down: Matrix<'a>,
}

/// `MoeGrouped` with `act` gating over the first `rows_active` rows of `x` (`hidden` values
/// each): for row `r` and slot `j < k`, expert `experts[ids[r·k + j]]`'s output into
/// `y[(r·k + j)·hidden..]` (module docs).
///
/// # Errors
///
/// [`OpError::RowOutOfRange`] for an ID past `experts`; [`OpError::ShapeMismatch`] when an
/// expert's matrices do not fit `hidden` and each other; [`OpError::Len`] when a buffer is
/// short; [`OpError::Unsupported`] for gating other than SiLU, as
/// [`gate_up_act`].
#[allow(
    clippy::too_many_arguments,
    reason = "the op's operands and dims, as RFC-0040 §3.4 lists them"
)]
pub fn moe_grouped(
    act: GatedAct,
    experts: &[Expert<'_>],
    k: usize,
    ids: &[u32],
    x: &[f32],
    hidden: usize,
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    const OP: &str = "MoeGrouped";
    at_least(OP, "ids", ids.len(), rows_active, k)?;
    at_least(OP, "x", x.len(), rows_active, hidden)?;
    at_least(OP, "y", y.len(), rows_active * k, hidden)?;
    let ids = &ids[..rows_active * k];
    if let Some(&id) = ids.iter().find(|&&id| id as usize >= experts.len()) {
        return Err(OpError::RowOutOfRange {
            id,
            rows: experts.len(),
        });
    }
    let mut inter = 0;
    for &id in ids {
        let e = &experts[id as usize];
        let i = e.gate.rows();
        if (e.up.rows(), e.up.cols(), e.gate.cols()) != (i, hidden, hidden)
            || (e.down.rows(), e.down.cols()) != (hidden, i)
        {
            return Err(OpError::ShapeMismatch { op: OP });
        }
        inter = inter.max(i);
    }
    if !ids.is_empty() && act != GatedAct::Silu {
        // Refuse before writing, as `gate_up_act` would.
        return gate_up_act(act, experts[0].gate, experts[0].up, &[], 0, &mut []);
    }
    let mut h = vec![0.0f32; inter];
    for r in 0..rows_active {
        let xr = &x[r * hidden..(r + 1) * hidden];
        for j in 0..k {
            let e = &experts[ids[r * k + j] as usize];
            let h = &mut h[..e.gate.rows()];
            gate_up_act(act, e.gate, e.up, xr, 1, h)?;
            let s = (r * k + j) * hidden;
            gemv(e.down, h, 1, &mut y[s..s + hidden])?;
        }
    }
    Ok(())
}

/// `MoeCombine` over the first `rows_active` rows: row `r`'s routed outputs
/// `routed[(r·k + j)·hidden..]` weighted by `weights[r·k + j]` in ascending `ids[r·k + j]`, then
/// its `n_shared` shared outputs `shared[(r·n_shared + s)·hidden..]`, summed into
/// `y[r·hidden..]` (module docs).
///
/// # Errors
///
/// [`OpError::Len`] when a buffer is short.
#[allow(
    clippy::too_many_arguments,
    reason = "the op's operands and dims, as RFC-0040 §3.4 lists them"
)]
pub fn moe_combine(
    k: usize,
    ids: &[u32],
    weights: &[f32],
    routed: &[f32],
    n_shared: usize,
    shared: &[f32],
    hidden: usize,
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    const OP: &str = "MoeCombine";
    at_least(OP, "ids", ids.len(), rows_active, k)?;
    at_least(OP, "weights", weights.len(), rows_active, k)?;
    at_least(OP, "routed", routed.len(), rows_active * k, hidden)?;
    at_least(OP, "shared", shared.len(), rows_active * n_shared, hidden)?;
    at_least(OP, "y", y.len(), rows_active, hidden)?;
    let mut order: Vec<usize> = Vec::with_capacity(k);
    for r in 0..rows_active {
        order.clear();
        order.extend(0..k);
        // A stable sort: one expert in two slots keeps slot order.
        order.sort_by_key(|&j| ids[r * k + j]);
        for c in 0..hidden {
            y[r * hidden + c] = reduce::sum_by(k + n_shared, |t| {
                if t < k {
                    let j = order[t];
                    weights[r * k + j] * routed[(r * k + j) * hidden + c]
                } else {
                    shared[(r * n_shared + t - k) * hidden + c]
                }
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
