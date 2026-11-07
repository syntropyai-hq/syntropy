// SPDX-License-Identifier: MIT OR Apache-2.0

//! `TapGather`'s reference semantics: the parity taps of RFC-0040 §3.8 (RFC-0042 §4.2;
//! COMP-E1-T18).
//!
//! For each tapped row `r` of the model's logits (the `Logits` output, before `LogitsProcess`),
//! with `k` reference token IDs:
//!
//! ```text
//! m        = max over the full vocabulary
//! lse_r    = m + ln(Σ_i exp(logits[r][i] − m))      (F32Fixed sum, math::exp, math::ln)
//! lp_r[j]  = logits[r][ids[r][j]] − lse_r            for j < k
//! argmax_r = the first index of the largest logit   (ties to the lower ID)
//! margin_r = (logits[r][argmax_r] − lse_r) − (logits[r][second_r] − lse_r)
//! ```
//!
//! `second_r` is the largest logit at an index other than `argmax_r`, ties to the lower ID, so
//! a tie for first gives a margin of `+0`. Each line is IEEE F32 operations, each rounded once.
//!
//! Edge cases. `rows_tap = 0` writes nothing. A vocabulary of one token has no second, and its
//! margin is `+∞`. A NaN anywhere in the row makes `lse`, every `lp` and the margin NaN, and the
//! argmax [`NAN_ROW`]. A row whose logits are all `−∞` has `lse = −∞`, every `lp` NaN
//! (`−∞ − −∞`), argmax [`EMPTY_ROW`] and a NaN margin. A `+∞` logit makes `m − m` NaN, so `lse` is NaN as
//! well; the argmax is the `+∞` token's ID. The argmax and the indices are RFC-0060's
//! bit-exact class.

use super::math;
use super::reduce;
use super::sample::{EMPTY_ROW, NAN_ROW};
use super::shape::{OpError, at_least};

/// One tapped row's summary (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TapRow {
    /// The log-sum-exp over the full vocabulary.
    pub lse: f32,
    /// The first index of the largest logit, or a marker.
    pub argmax: u32,
    /// The top-1 log-prob minus the top-2 log-prob.
    pub margin: f32,
}

/// `TapGather` over `logits` (rows of `vocab` values): for each tapped row index `rows[t]`, its
/// `k` reference IDs `ids[t·k..(t+1)·k]` give `lp[t·k..(t+1)·k]`, and `out[t]` its summary
/// (module docs).
///
/// # Errors
///
/// [`OpError::Len`] when `ids`, `lp` or `out` is short for `rows.len()` tapped rows;
/// [`OpError::RowOutOfRange`] for a tapped row past `logits`' rows or a reference ID past
/// `vocab`. Everything is checked before anything is written.
pub fn tap_gather(
    vocab: usize,
    k: usize,
    logits: &[f32],
    rows: &[u32],
    ids: &[u32],
    lp: &mut [f32],
    out: &mut [TapRow],
) -> Result<(), OpError> {
    const OP: &str = "TapGather";
    let n = rows.len();
    at_least(OP, "ids", ids.len(), n, k)?;
    at_least(OP, "lp", lp.len(), n, k)?;
    at_least(OP, "out", out.len(), n, 1)?;
    let have = logits.len().checked_div(vocab).unwrap_or(0);
    if let Some(&r) = rows.iter().find(|&&r| r as usize >= have) {
        return Err(OpError::RowOutOfRange { id: r, rows: have });
    }
    if let Some(&id) = ids[..n * k].iter().find(|&&id| id as usize >= vocab) {
        return Err(OpError::RowOutOfRange { id, rows: vocab });
    }
    for (t, &r) in rows.iter().enumerate() {
        let row = &logits[r as usize * vocab..(r as usize + 1) * vocab];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, |a, b| {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                a.max(b)
            }
        });
        let lse = if m == f32::NEG_INFINITY {
            m
        } else {
            m + math::ln(reduce::sum_by(vocab, |i| math::exp(row[i] - m)))
        };
        for (lp, &id) in lp[t * k..(t + 1) * k]
            .iter_mut()
            .zip(&ids[t * k..(t + 1) * k])
        {
            *lp = row[id as usize] - lse;
        }
        let argmax = super::sample::argmax_of(row);
        let margin = if argmax == NAN_ROW || argmax == EMPTY_ROW {
            f32::NAN
        } else {
            let a = argmax as usize;
            let second = (0..vocab)
                .filter(|&i| i != a)
                .fold(None, |best: Option<usize>, i| match best {
                    Some(b) if row[i] <= row[b] => Some(b),
                    _ => Some(i),
                });
            match second {
                None => f32::INFINITY,
                Some(b) => (row[a] - lse) - (row[b] - lse),
            }
        };
        out[t] = TapRow {
            lse,
            argmax,
            margin,
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests;
