// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OpKind::RmsNorm`'s reference semantics (RFC-0040 §3.1, §3.3; RFC-0042 §4.2; COMP-E1-T03).
//!
//! For each active row `x` of `n` values, with the family's weight `g` and `eps`:
//!
//! ```text
//! ss    = Σ x[i]·x[i]                 (F32Fixed order, reduce::sum_by)
//! mean  = ss / n
//! scale = 1 / sqrt(mean + eps)
//! y[i]  = (x[i] · scale) · g[i]
//! ```
//!
//! Each step is one IEEE F32 operation, rounded once; `sqrt` and the divisions are IEEE's
//! correctly rounded ones. This is ggml's operation order for its `rms_norm` followed by the
//! weight's `mul`, with the sum of squares in the `F32Fixed` order and in F32 (RFC-0042 §4.2's
//! rule for `RmsNorm`).
//!
//! Edge cases: a row of zeros gives zeros when `eps > 0`, and NaN when `eps = 0` (`scale = +∞`,
//! and `0 · ∞` is NaN); a NaN anywhere in the row makes the whole row NaN through `ss`; an
//! infinite value makes `ss = +∞` and `scale = 0`, so the row is NaN where `x` is infinite and
//! `±0` elsewhere. Every value of `g` multiplies its own column only.
//! `Logits` applies this to each row before the LM head ([`logits`](super::linear::logits)).

use super::reduce;
use super::shape::{OpError, at_least};

/// A normalisation's weight `g`, one value per column, and its `eps` (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Norm<'a> {
    /// `g`, whose length is the row width.
    pub weight: &'a [f32],
    /// `eps`, as the family defines it.
    pub eps: f32,
}

impl Norm<'_> {
    /// The row width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.weight.len()
    }

    /// One row, normalised into `y` (module docs). `x` and `y` are `width` long.
    pub(crate) fn row(&self, x: &[f32], y: &mut [f32]) {
        debug_assert!(x.len() == self.width() && y.len() == self.width());
        let ss = reduce::sum_by(x.len(), |i| x[i] * x[i]);
        // A width of zero makes `mean` NaN, but there is no value to scale then.
        let mean = ss / x.len() as f32;
        let scale = 1.0 / (mean + self.eps).sqrt();
        for ((y, &x), &g) in y.iter_mut().zip(x).zip(self.weight) {
            *y = (x * scale) * g;
        }
    }
}

/// `RmsNorm` over the first `rows_active` rows of `x` into `y`, each row `norm.width()` values
/// (module docs).
///
/// # Errors
///
/// [`OpError::Len`] when `x` or `y` holds fewer than `rows_active` rows.
pub fn rms_norm(
    norm: Norm<'_>,
    x: &[f32],
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    let n = norm.width();
    at_least("RmsNorm", "x", x.len(), rows_active, n)?;
    at_least("RmsNorm", "y", y.len(), rows_active, n)?;
    if n == 0 {
        return Ok(());
    }
    for (x, y) in x
        .chunks_exact(n)
        .zip(y.chunks_exact_mut(n))
        .take(rows_active)
    {
        norm.row(x, y);
    }
    Ok(())
}
