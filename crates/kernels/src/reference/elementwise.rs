// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OpKind::Add` and `OpKind::Mul`'s reference semantics (RFC-0040 §3.1, §3.3; COMP-E1-T03).
//!
//! `y[r][i] = a[r][i] ∘ b[r][i]` for every active row `r` and column `i`, with `∘` one IEEE F32
//! addition or multiplication, rounded once. The second operand is either a buffer of rows like
//! the first ([`Operand::Rows`], a residual or another branch's output) or one row applied to
//! every active row ([`Operand::Row`], a bias or a per-column scale).
//!
//! Edge cases: with `rows_active = 0` nothing is written, and rows past it are neither read nor
//! written. NaN propagates: `NaN ∘ x` is NaN, so is `∞ − ∞` in `Add` and `0 · ∞` in `Mul`. Signed
//! zeros follow IEEE: `+0 + −0 = +0`, `−0 + −0 = −0`.

use super::shape::{OpError, at_least};

/// The second operand of [`add`] and [`mul`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand<'a> {
    /// Rows of the same width as the first operand, at least `rows_active` of them.
    Rows(&'a [f32]),
    /// One row of the first operand's width, applied to every active row.
    Row(&'a [f32]),
}

fn apply(
    op: &'static str,
    f: fn(f32, f32) -> f32,
    a: &[f32],
    b: Operand<'_>,
    width: usize,
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    at_least(op, "a", a.len(), rows_active, width)?;
    at_least(op, "y", y.len(), rows_active, width)?;
    match b {
        Operand::Rows(b) => at_least(op, "b", b.len(), rows_active, width)?,
        Operand::Row(b) if b.len() != width => {
            return Err(OpError::Len {
                op,
                arg: "b",
                len: b.len(),
                need: width,
            });
        }
        Operand::Row(_) => {}
    }
    let n = rows_active * width;
    for (i, (y, &a)) in y[..n].iter_mut().zip(&a[..n]).enumerate() {
        let b = match b {
            Operand::Rows(b) => b[i],
            Operand::Row(b) => b[i % width],
        };
        *y = f(a, b);
    }
    Ok(())
}

/// `Add`: `y = a + b` over the first `rows_active` rows of `width` values (module docs).
///
/// # Errors
///
/// [`OpError::Len`] when `a`, `y` or a [`Operand::Rows`] holds fewer than `rows_active` rows, or
/// a [`Operand::Row`] is not `width` long.
pub fn add(
    a: &[f32],
    b: Operand<'_>,
    width: usize,
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    apply("Add", |a, b| a + b, a, b, width, rows_active, y)
}

/// `Mul`: `y = a · b` over the first `rows_active` rows of `width` values (module docs).
///
/// # Errors
///
/// As [`add`].
pub fn mul(
    a: &[f32],
    b: Operand<'_>,
    width: usize,
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    apply("Mul", |a, b| a * b, a, b, width, rows_active, y)
}

#[cfg(test)]
mod tests;
