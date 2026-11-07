// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OpKind::Act`'s reference semantics (RFC-0040 §3.1, §3.3; COMP-E1-T17).
//!
//! `y[r][i] = act(x[r][i])` for every active row `r` and column `i`, ungated. M0 defines SiLU,
//! [`math::silu`]'s `x / (1 + exp(−x))` with the software `exp`, the activation `GateUpAct`
//! applies to its gate. GELU needs a software `tanh` or `erf`, which COMP-E1-T04 adds with
//! GELU gating; `Act(Gelu)` is refused until then.
//!
//! Edge cases: with `rows_active = 0` nothing is written, and rows past it are neither read nor
//! written. Each output depends on its own input only. NaN gives NaN; `silu(+∞) = +∞`,
//! `silu(−∞)` is NaN (`−∞ / +∞`), and a large negative finite input gives `−0`;
//! `silu(±0) = ±0`.

use super::math;
use super::shape::{OpError, at_least};
use crate::op::ActKind;

/// `Act`: `y = act(x)` over the first `rows_active` rows of `width` values (module docs).
///
/// # Errors
///
/// [`OpError::Unsupported`] for an activation other than SiLU; [`OpError::Len`] when `x` or `y`
/// holds fewer than `rows_active` rows.
pub fn act(
    kind: ActKind,
    x: &[f32],
    width: usize,
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    let f: fn(f32) -> f32 = match kind {
        ActKind::Silu => math::silu,
        ActKind::Gelu => {
            return Err(OpError::Unsupported {
                op: "Act",
                what: "GELU (COMP-E1-T04)",
            });
        }
    };
    at_least("Act", "x", x.len(), rows_active, width)?;
    at_least("Act", "y", y.len(), rows_active, width)?;
    let n = rows_active * width;
    for (y, &x) in y[..n].iter_mut().zip(&x[..n]) {
        *y = f(x);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
