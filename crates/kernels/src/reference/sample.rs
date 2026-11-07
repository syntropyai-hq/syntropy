// SPDX-License-Identifier: MIT OR Apache-2.0

//! The sampling ops' reference semantics: `LogitsProcess`, `TopKTopP`, `SampleSeeded`, `Argmax`
//! and `VerifyAccept` (RFC-0040 §3.1, §3.3; RFC-0042 §4.2; RFC-0012 §8.1, §9.5; COMP-E1-T18).
//!
//! Every op reads F32 logits, `vocab` values per row, and ties always go to the **lower token
//! ID** (RFC-0042 §4.2). Sums go through [`reduce`]'s `F32Fixed` order, and `exp` is
//! [`math::exp`].
//!
//! - **`LogitsProcess`** turns a row of model logits into the sequence's processed logits. Per
//!   token `i`, in this order, each step one IEEE operation rounded once:
//!   1. the logit bias: each `(i, b)` entry adds `b`, in the order the entries are listed;
//!   2. the repetition penalty `r` (1 is off), for tokens whose `context` count is non-zero:
//!      `v / r` when `v > 0`, else `v · r`;
//!   3. the frequency and presence penalties, for tokens whose `emitted` count `c` is non-zero:
//!      `(v − c · frequency) − presence`, with `c` converted to F32;
//!   4. the grammar mask: a token whose bit is clear becomes `−∞`, unless its value is NaN,
//!      which is never masked away (RFC-0042 §4.1 rule 6);
//!   5. the temperature `T`: `v / T` when `T > 0`. `T = 0` is greedy, which leaves the values
//!      as they are for `Argmax`.
//!
//!   RFC-0012 §8.1 names these parts but not the order in which they compose. This order puts
//!   the additive and penalty adjustments on the model's own logit scale and divides by the
//!   temperature last, so the temperature scales the processed distribution as a whole.
//! - **`TopKTopP`** keeps a token when it passes all three filters, each computed on the same
//!   row, and writes `−∞` over the rest:
//!   - top-k (`0` is off): the `k` largest values, ranked by value, ties to the lower ID;
//!   - top-p (`≥ 1` is off): with `p` the softmax of the top-k survivors, the shortest prefix of
//!     them in rank order whose sequential running sum of `p` reaches `top_p`, and at least one
//!     token;
//!   - min-p (`0` is off): `p[i] ≥ fl(min_p · p_max)`.
//!
//!   A row that holds a NaN or a `+∞` becomes all NaN, so the sampler reports it. A row with no
//!   finite value keeps its `−∞` values.
//! - **`SampleSeeded`** draws one token per row from the softmax of the row, given the row's
//!   uniform draw `u` in `[0, 1)`. The draw comes from PLC-E9-T01's counter-based generator;
//!   this function only consumes it. With `m` the row max, `e[i] = exp(v[i] − m)` and
//!   `Z = Σ e` in the `F32Fixed` order, the token is the first `i`, in ID order, whose running
//!   sum `e[0] + … + e[i]` (sequential, from `+0`) exceeds `fl(u · Z)`. If rounding leaves no such
//!   `i`, it is the last `i` with `e[i] > 0`.
//! - **`Argmax`**: the first index of the largest value.
//! - **`VerifyAccept`** applies RFC-0012 §9.5's exact acceptance to one chain of drafts. The
//!   window has `γ + 1` rows of processed logits for `γ` drafted tokens. **Greedy**: accept while
//!   `d_i` is row `i`'s `Argmax`; the next token is the `Argmax` of the first row not accepted,
//!   or of row `γ`. **Sampled, point-mass drafts**: for each `i`, take the next draw `u` and
//!   accept `d_i` when `u < p_i(d_i)`, with `p_i(d_i) = fl(e[d_i] / Z)` on row `i`. On the first
//!   rejection, the next draw samples row `i` with `d_i` removed (its logit set to `−∞`, which
//!   is the renormalised residual), and the chain stops. If every draft is accepted, the next
//!   draw samples row `γ`, the bonus token. Draws are consumed in exactly that order (RFC-0012
//!   §8.2). Stochastic drafts and trees are M1 work (PLC-E9-T05) and are not defined here.
//!
//! **Failures.** A row that cannot give a token writes a marker instead of an ID: [`NAN_ROW`]
//! when it holds a NaN, or, in the ops that take a softmax (`SampleSeeded` and sampled
//! `VerifyAccept`), a `+∞`, for which the softmax is undefined (`∞ − ∞`); and [`EMPTY_ROW`]
//! when no token is left, as when the grammar mask removes every token. `Argmax` takes a `+∞`
//! as the largest value.
//! The tail fails the sequence with a `Correctness` error on either (RFC-0042 §4.1 rule 6). It
//! never samples another token in their place.
//!
//! **Rows.** Every op but `VerifyAccept` takes `rows_active` and reads and writes only that
//! many rows (`shape`'s rules). `VerifyAccept` takes one window per call.

use super::math;
use super::reduce;
use super::shape::{OpError, at_least};

/// The marker a sampling op writes for a row that holds a NaN or a `+∞` (module docs).
pub const NAN_ROW: u32 = u32::MAX;

/// The marker a sampling op writes for a row in which no token has positive mass (module docs).
pub const EMPTY_ROW: u32 = u32::MAX - 1;

/// The repetition, presence and frequency penalties of one row, with the sequence's two count
/// tables (RFC-0012 §8.1): `context` counts the prompt and the emitted tokens, `emitted` the
/// emitted tokens only. Each table holds one count per token ID.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Penalties<'a> {
    /// The repetition penalty; 1 is off. Positive and finite.
    pub repetition: f32,
    /// The presence penalty; 0 is off. Finite.
    pub presence: f32,
    /// The frequency penalty; 0 is off. Finite.
    pub frequency: f32,
    /// Prompt and emitted counts, `vocab` of them.
    pub context: &'a [u32],
    /// Emitted counts, `vocab` of them.
    pub emitted: &'a [u32],
}

/// What `LogitsProcess` applies to one row: its sequence's sampling parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Process<'a> {
    /// The temperature; 0 is greedy and applies nothing. Finite and not negative.
    pub temperature: f32,
    /// The penalties, or `None` when every penalty is off.
    pub penalties: Option<Penalties<'a>>,
    /// The logit bias, `(token ID, bias)`, applied in this order.
    pub bias: &'a [(u32, f32)],
    /// The grammar mask: token `i` is allowed when bit `i % 32` of word `i / 32` is set. `None`
    /// allows every token.
    pub mask: Option<&'a [u32]>,
}

impl Process<'_> {
    /// No processing: temperature 1, no penalties, no bias, no mask.
    pub const NONE: Process<'static> = Process {
        temperature: 1.0,
        penalties: None,
        bias: &[],
        mask: None,
    };
}

/// `LogitsProcess` over the first `rows_active` rows of `x` into `y`, row `r` with `rows[r]`
/// (module docs).
///
/// # Errors
///
/// [`OpError::Len`] when `x`, `y` or `rows` is short, or a count table or mask does not cover
/// `vocab`; [`OpError::RowOutOfRange`] for a bias entry past `vocab`; [`OpError::Invalid`] for a
/// negative or non-finite temperature, or a penalty outside its range. Every row is checked
/// before any is written.
pub fn logits_process(
    rows: &[Process<'_>],
    vocab: usize,
    x: &[f32],
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    const OP: &str = "LogitsProcess";
    at_least(OP, "x", x.len(), rows_active, vocab)?;
    at_least(OP, "y", y.len(), rows_active, vocab)?;
    at_least(OP, "rows", rows.len(), rows_active, 1)?;
    for p in &rows[..rows_active] {
        if !(p.temperature.is_finite() && p.temperature >= 0.0) {
            return Err(OpError::Invalid {
                op: OP,
                what: "the temperature must be finite and not negative",
            });
        }
        if let Some(pen) = p.penalties {
            if !(pen.repetition.is_finite() && pen.repetition > 0.0)
                || !pen.presence.is_finite()
                || !pen.frequency.is_finite()
            {
                return Err(OpError::Invalid {
                    op: OP,
                    what: "the repetition penalty must be positive and every penalty finite",
                });
            }
            at_least(OP, "context", pen.context.len(), 1, vocab)?;
            at_least(OP, "emitted", pen.emitted.len(), 1, vocab)?;
        }
        if let Some(&(id, _)) = p.bias.iter().find(|&&(id, _)| id as usize >= vocab) {
            return Err(OpError::RowOutOfRange { id, rows: vocab });
        }
        if let Some(mask) = p.mask {
            at_least(OP, "mask", mask.len(), 1, vocab.div_ceil(32))?;
        }
    }
    for (r, p) in rows[..rows_active].iter().enumerate() {
        let (x, y) = (
            &x[r * vocab..(r + 1) * vocab],
            &mut y[r * vocab..(r + 1) * vocab],
        );
        y.copy_from_slice(x);
        for &(id, b) in p.bias {
            y[id as usize] += b;
        }
        if let Some(pen) = p.penalties {
            for (i, v) in y.iter_mut().enumerate() {
                if pen.context[i] > 0 {
                    *v = if *v > 0.0 {
                        *v / pen.repetition
                    } else {
                        *v * pen.repetition
                    };
                }
                let c = pen.emitted[i];
                if c > 0 {
                    *v = (*v - c as f32 * pen.frequency) - pen.presence;
                }
            }
        }
        if let Some(mask) = p.mask {
            for (i, v) in y.iter_mut().enumerate() {
                if mask[i / 32] & (1 << (i % 32)) == 0 && !v.is_nan() {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        if p.temperature > 0.0 {
            for v in y.iter_mut() {
                *v /= p.temperature;
            }
        }
    }
    Ok(())
}

/// The filters of `TopKTopP` for one row (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Filter {
    /// Top-k; 0 is off.
    pub top_k: u32,
    /// Top-p; 1 or more is off. Not NaN.
    pub top_p: f32,
    /// Min-p; 0 is off. In `[0, 1]`.
    pub min_p: f32,
}

impl Filter {
    /// Every filter off.
    pub const NONE: Filter = Filter {
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
    };
}

/// Whether `a` ranks before `b`: a larger value, or an equal value at a lower ID. Neither
/// value is NaN.
fn ranks_before(row: &[f32], a: usize, b: usize) -> bool {
    row[a] > row[b] || (row[a] == row[b] && a < b)
}

/// The IDs of `row` in rank order (module docs). `row` holds no NaN.
fn ranked(row: &[f32]) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..row.len()).collect();
    ids.sort_by(|&a, &b| {
        if ranks_before(row, a, b) {
            std::cmp::Ordering::Less
        } else if ranks_before(row, b, a) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    ids
}

/// Whether the softmax of `row` is undefined: a NaN, or a `+∞` (module docs).
fn undefined(row: &[f32]) -> bool {
    row.iter().any(|v| v.is_nan() || *v == f32::INFINITY)
}

/// The row max and the `F32Fixed` sum of `exp(v − max)`, for a row with no NaN and no `+∞`.
/// The max is `−∞` when no value is finite, and the sum is then `+0`.
fn softmax_parts(row: &[f32]) -> (f32, f32) {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        return (m, 0.0);
    }
    (m, reduce::sum_by(row.len(), |i| math::exp(row[i] - m)))
}

/// `TopKTopP` over the first `rows_active` rows of `x` into `y` (module docs).
///
/// # Errors
///
/// [`OpError::Len`] when `x`, `y` or `filters` is short; [`OpError::Invalid`] for a NaN `top_p`
/// or a `min_p` outside `[0, 1]`.
pub fn top_k_top_p(
    filters: &[Filter],
    vocab: usize,
    x: &[f32],
    rows_active: usize,
    y: &mut [f32],
) -> Result<(), OpError> {
    const OP: &str = "TopKTopP";
    at_least(OP, "x", x.len(), rows_active, vocab)?;
    at_least(OP, "y", y.len(), rows_active, vocab)?;
    at_least(OP, "filters", filters.len(), rows_active, 1)?;
    if filters[..rows_active]
        .iter()
        .any(|f| f.top_p.is_nan() || !(0.0..=1.0).contains(&f.min_p))
    {
        return Err(OpError::Invalid {
            op: OP,
            what: "top_p must not be NaN and min_p must lie in [0, 1]",
        });
    }
    for (r, f) in filters[..rows_active].iter().enumerate() {
        let (x, y) = (
            &x[r * vocab..(r + 1) * vocab],
            &mut y[r * vocab..(r + 1) * vocab],
        );
        if undefined(x) {
            y.fill(f32::NAN);
            continue;
        }
        let order = ranked(x);
        let mut keep = vec![true; vocab];
        let k = f.top_k as usize;
        if k > 0 && k < vocab {
            for &i in &order[k..] {
                keep[i] = false;
            }
        }
        let top: Vec<f32> = (0..vocab)
            .map(|i| if keep[i] { x[i] } else { f32::NEG_INFINITY })
            .collect();
        let (m, z) = softmax_parts(&top);
        if m != f32::NEG_INFINITY && (f.top_p < 1.0 || f.min_p > 0.0) {
            let p = |i: usize| math::exp(top[i] - m) / z;
            if f.top_p < 1.0 {
                let mut run = 0.0f32;
                let mut reached = false;
                for &i in &order {
                    if !keep[i] {
                        continue;
                    }
                    if reached {
                        keep[i] = false;
                    } else {
                        run += p(i);
                        reached = run >= f.top_p;
                    }
                }
            }
            if f.min_p > 0.0 {
                // The largest probability is the row max's.
                let threshold = f.min_p * p(order[0]);
                for (i, kept) in keep.iter_mut().enumerate() {
                    if *kept && p(i) < threshold {
                        *kept = false;
                    }
                }
            }
        }
        for (i, v) in y.iter_mut().enumerate() {
            *v = if keep[i] { x[i] } else { f32::NEG_INFINITY };
        }
    }
    Ok(())
}

/// One row's sampled token, or its marker, from the uniform draw `u` (module docs).
fn sample_row(row: &[f32], u: f32) -> u32 {
    if undefined(row) {
        return NAN_ROW;
    }
    let (m, z) = softmax_parts(row);
    if m == f32::NEG_INFINITY {
        return EMPTY_ROW;
    }
    let t = u * z;
    let mut run = 0.0f32;
    let mut last = None;
    for (i, &v) in row.iter().enumerate() {
        let e = math::exp(v - m);
        if e > 0.0 {
            last = Some(i);
        }
        run += e;
        if run > t {
            return i as u32;
        }
    }
    last.map_or(EMPTY_ROW, |i| i as u32)
}

fn check_draw(op: &'static str, u: f32) -> Result<(), OpError> {
    if (0.0..1.0).contains(&u) {
        Ok(())
    } else {
        Err(OpError::Invalid {
            op,
            what: "a uniform draw must lie in [0, 1)",
        })
    }
}

/// `SampleSeeded` over the first `rows_active` rows of `x`, row `r` with the draw `u[r]`, into
/// `out` (module docs).
///
/// # Errors
///
/// [`OpError::Len`] when `x`, `u` or `out` is short; [`OpError::Invalid`] for a draw outside
/// `[0, 1)`.
pub fn sample_seeded(
    vocab: usize,
    x: &[f32],
    u: &[f32],
    rows_active: usize,
    out: &mut [u32],
) -> Result<(), OpError> {
    const OP: &str = "SampleSeeded";
    at_least(OP, "x", x.len(), rows_active, vocab)?;
    at_least(OP, "u", u.len(), rows_active, 1)?;
    at_least(OP, "out", out.len(), rows_active, 1)?;
    for &u in &u[..rows_active] {
        check_draw(OP, u)?;
    }
    for r in 0..rows_active {
        out[r] = sample_row(&x[r * vocab..(r + 1) * vocab], u[r]);
    }
    Ok(())
}

/// One row's `Argmax`, or its marker (module docs).
pub(crate) fn argmax_of(row: &[f32]) -> u32 {
    if row.iter().any(|v| v.is_nan()) {
        return NAN_ROW;
    }
    let mut best: Option<usize> = None;
    for (i, &v) in row.iter().enumerate() {
        if v > f32::NEG_INFINITY && best.is_none_or(|b| v > row[b]) {
            best = Some(i);
        }
    }
    best.map_or(EMPTY_ROW, |i| i as u32)
}

/// `Argmax` over the first `rows_active` rows of `x` into `out` (module docs). A row whose
/// values are all `−∞` gives [`EMPTY_ROW`].
///
/// # Errors
///
/// [`OpError::Len`] when `x` or `out` is short.
pub fn argmax(vocab: usize, x: &[f32], rows_active: usize, out: &mut [u32]) -> Result<(), OpError> {
    at_least("Argmax", "x", x.len(), rows_active, vocab)?;
    at_least("Argmax", "out", out.len(), rows_active, 1)?;
    for r in 0..rows_active {
        out[r] = argmax_of(&x[r * vocab..(r + 1) * vocab]);
    }
    Ok(())
}

/// How `VerifyAccept` accepts (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Acceptance {
    /// Greedy: drafts must equal each row's `Argmax`. No draws.
    Greedy,
    /// Sampled, with point-mass drafts.
    Sampled,
}

/// What one `VerifyAccept` window gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verdict {
    /// The drafts accepted, a prefix of the chain.
    pub accepted: usize,
    /// The token after them: the bonus, the residual sample or the first non-accepted row's
    /// `Argmax`, or a marker ([`NAN_ROW`], [`EMPTY_ROW`]) when that row gives no token.
    pub next: u32,
    /// The draws consumed, from the front of `u`.
    pub draws: usize,
}

/// `VerifyAccept` over one chain: `logits` holds `drafts.len() + 1` rows of processed logits,
/// and `u` the sequence's next draws in counter order (module docs).
///
/// # Errors
///
/// [`OpError::Len`] when `logits` is short, or `u` runs out before the rules finish;
/// [`OpError::RowOutOfRange`] for a draft past `vocab`; [`OpError::Invalid`] for a draw outside
/// `[0, 1)`.
pub fn verify_accept(
    mode: Acceptance,
    vocab: usize,
    logits: &[f32],
    drafts: &[u32],
    u: &[f32],
) -> Result<Verdict, OpError> {
    const OP: &str = "VerifyAccept";
    let gamma = drafts.len();
    at_least(OP, "logits", logits.len(), gamma + 1, vocab)?;
    if let Some(&d) = drafts.iter().find(|&&d| d as usize >= vocab) {
        return Err(OpError::RowOutOfRange { id: d, rows: vocab });
    }
    let row = |i: usize| &logits[i * vocab..(i + 1) * vocab];
    if mode == Acceptance::Greedy {
        for (i, &d) in drafts.iter().enumerate() {
            let a = argmax_of(row(i));
            if a != d {
                return Ok(Verdict {
                    accepted: i,
                    next: a,
                    draws: 0,
                });
            }
        }
        return Ok(Verdict {
            accepted: gamma,
            next: argmax_of(row(gamma)),
            draws: 0,
        });
    }
    let mut used = 0;
    let mut draw = || -> Result<f32, OpError> {
        let &v = u.get(used).ok_or(OpError::Len {
            op: OP,
            arg: "u",
            len: u.len(),
            need: used + 1,
        })?;
        check_draw(OP, v)?;
        used += 1;
        Ok(v)
    };
    for (i, &d) in drafts.iter().enumerate() {
        let x = row(i);
        if undefined(x) {
            return Ok(Verdict {
                accepted: i,
                next: NAN_ROW,
                draws: used,
            });
        }
        let (m, z) = softmax_parts(x);
        let v = draw()?;
        let p = if m == f32::NEG_INFINITY {
            0.0
        } else {
            math::exp(x[d as usize] - m) / z
        };
        if v < p {
            continue;
        }
        let mut residual = x.to_vec();
        residual[d as usize] = f32::NEG_INFINITY;
        let v = draw()?;
        return Ok(Verdict {
            accepted: i,
            next: sample_row(&residual, v),
            draws: used,
        });
    }
    let v = draw()?;
    Ok(Verdict {
        accepted: gamma,
        next: sample_row(row(gamma), v),
        draws: used,
    })
}

#[cfg(test)]
mod tests;
