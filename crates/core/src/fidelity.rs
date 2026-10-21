// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fidelity metrics of the exactness contract and the `KL_K` tightness rule.
//!
//! RFC-0042 §5 defines what is measured and RFC-0060 §4.2 and §5.3–§5.4 define the
//! estimators. These are the shared definitions: the parity harness in `testkit`, the
//! executor's runtime spot check (RFC-0040 §6.2) and the warm-up canary (RFC-0060 §5.8) all
//! compute the same numbers through these functions.
//!
//! - Op level (RFC-0042 §5.1, RFC-0060 §4.2): [`rel_l2`] and [`max_abs_scaled`].
//! - End to end on full logits (RFC-0042 §5.3): [`log_softmax`], [`kl_full`] and
//!   [`top1_agrees`], with [`argmax`] breaking ties toward the lower token ID.
//! - End to end from a stored reference (RFC-0060 §5.3–§5.4): [`kl_k`], the KL coarse-grained
//!   to the reference's top-K plus one tail bucket, a lower bound on the full KL; and
//!   [`kl_tail`], the unbiased estimate of the remainder from the proportional tail sample.
//! - Aggregates (RFC-0042 §5.3): [`aggregate`] and [`aggregate_by_category`] give the mean
//!   KL, the nearest-rank p99 KL and the top-1 disagreement, pooled and per category.
//! - The tightness rule (RFC-0060 §5.4, ADR-0601 decision 2): [`check_tightness`] and
//!   [`check_tightness_by_category`].
//!
//! All arithmetic is in F64. Logits are read as F32 or F64 and widened exactly. A non-finite
//! input is an error ([`MetricError::NonFinite`]), never clamped (RFC-0060 §5.4). Sums use
//! Neumaier-compensated summation in index order, so each result is deterministic for a given
//! input.

use std::collections::BTreeMap;
use std::fmt;

/// The relative term of the tightness bound: `0.01 · KL` (RFC-0060 §5.4).
pub const TIGHTNESS_REL: f64 = 0.01;

/// The absolute term of the tightness bound, as a fraction of the class ceiling:
/// `0.01 · C_kl_mean(d)` (RFC-0060 §5.4).
pub const TIGHTNESS_CEILING_FRACTION: f64 = 0.01;

/// The constant added to `rms(r)` in [`max_abs_scaled`]'s denominator (RFC-0042 §5.1).
pub const MAX_ABS_SCALED_EPS: f64 = 1e-30;

/// The percentile the gate reads from the KL distribution (RFC-0042 §5.3).
pub const P99: f64 = 99.0;

/// Why a metric could not be computed.
///
/// Every variant is a harness error: the caller reports it and never substitutes a value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetricError {
    /// An input that needs at least one element has none.
    Empty {
        /// The input.
        what: &'static str,
    },
    /// Two inputs that must be the same length are not.
    LengthMismatch {
        /// The input that sets the length.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// An input holds NaN or an infinity where a finite value is required.
    NonFinite {
        /// The input.
        what: &'static str,
        /// The element's index, for a slice input.
        index: Option<usize>,
    },
    /// Exactly one of the reference's `log r` and the candidate's `log s` is −∞. Both are −∞
    /// only when the top-K covers the whole vocabulary.
    TailBucketMismatch,
    /// The reference has tail mass (`r > 0`) but no tail sample, so `KL_tail` has no estimate.
    EmptyTailSample,
    /// A token ID is outside the vocabulary.
    IdOutOfRange {
        /// The ID.
        id: u32,
        /// The vocabulary size.
        vocab: usize,
    },
    /// A token ID appears twice in a top-K list.
    DuplicateTopKId {
        /// The ID.
        id: u32,
    },
    /// A tail-sample ID is in the top-K; the tail sample is drawn from V∖T only.
    TailIdInTopK {
        /// The ID.
        id: u32,
    },
    /// `K` is 0; the top-K needs at least one token.
    ZeroK,
    /// A position mixes a `KL_tail` value with positions that have none.
    MixedTail,
    /// The class ceiling `C_kl_mean(d)` is negative or not finite.
    InvalidCeiling,
    /// A percentile is outside (0, 100].
    InvalidPercentile,
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricError::Empty { what } => write!(f, "{what} is empty"),
            MetricError::LengthMismatch { expected, found } => {
                write!(f, "length mismatch: expected {expected}, found {found}")
            }
            MetricError::NonFinite {
                what,
                index: Some(i),
            } => write!(f, "{what}[{i}] is not finite"),
            MetricError::NonFinite { what, index: None } => write!(f, "{what} is not finite"),
            MetricError::TailBucketMismatch => {
                f.write_str("exactly one of log r and log s is -inf")
            }
            MetricError::EmptyTailSample => {
                f.write_str("tail mass is positive but the sample is empty")
            }
            MetricError::IdOutOfRange { id, vocab } => {
                write!(f, "token id {id} is outside a vocabulary of {vocab}")
            }
            MetricError::DuplicateTopKId { id } => write!(f, "token id {id} is in the top-K twice"),
            MetricError::TailIdInTopK { id } => {
                write!(f, "tail-sample token id {id} is in the top-K")
            }
            MetricError::ZeroK => f.write_str("K is 0"),
            MetricError::MixedTail => f.write_str("some positions carry KL_tail and others do not"),
            MetricError::InvalidCeiling => f.write_str("C_kl_mean is negative or not finite"),
            MetricError::InvalidPercentile => f.write_str("percentile is outside (0, 100]"),
        }
    }
}

impl std::error::Error for MetricError {}

/// Neumaier-compensated summation in insertion order.
#[derive(Clone, Copy, Default)]
struct Sum {
    sum: f64,
    comp: f64,
}

impl Sum {
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.comp += (self.sum - t) + x;
        } else {
            self.comp += (x - t) + self.sum;
        }
        self.sum = t;
    }

    fn value(self) -> f64 {
        self.sum + self.comp
    }
}

fn finite(x: f64, what: &'static str, index: Option<usize>) -> Result<f64, MetricError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(MetricError::NonFinite { what, index })
    }
}

fn same_len(expected: usize, found: usize) -> Result<(), MetricError> {
    if expected == found {
        Ok(())
    } else {
        Err(MetricError::LengthMismatch { expected, found })
    }
}

fn widen_all<T: Copy + Into<f64>>(xs: &[T], what: &'static str) -> Result<Vec<f64>, MetricError> {
    if xs.is_empty() {
        return Err(MetricError::Empty { what });
    }
    xs.iter()
        .enumerate()
        .map(|(i, &x)| finite(x.into(), what, Some(i)))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Op level (RFC-0042 §5.1, RFC-0060 §4.2)
// ---------------------------------------------------------------------------------------------

/// `rel_l2(y, r) = ‖y − r‖₂ / ‖r‖₂`, in F64 (RFC-0042 §5.1, RFC-0060 §4.2).
///
/// `y` is the candidate and `r` the reference. When `‖r‖₂ = 0`, the result is 0 if `y` is
/// also all zero and +∞ otherwise: an exact match passes and any other output fails every
/// finite threshold.
pub fn rel_l2<Y, R>(y: &[Y], r: &[R]) -> Result<f64, MetricError>
where
    Y: Copy + Into<f64>,
    R: Copy + Into<f64>,
{
    same_len(r.len(), y.len())?;
    let y = widen_all(y, "y")?;
    let r = widen_all(r, "r")?;
    let mut diff = Sum::default();
    let mut norm = Sum::default();
    for (&yi, &ri) in y.iter().zip(&r) {
        let d = yi - ri;
        diff.add(d * d);
        norm.add(ri * ri);
    }
    let diff = diff.value().sqrt();
    let norm = norm.value().sqrt();
    if norm == 0.0 {
        return Ok(if diff == 0.0 { 0.0 } else { f64::INFINITY });
    }
    Ok(diff / norm)
}

/// `max_abs_scaled(y, r) = max_i |y_i − r_i| / (rms(r) + 1e−30)`, in F64 (RFC-0042 §5.1,
/// RFC-0060 §4.2), with `rms(r) = √(Σ r_i² / n)`.
pub fn max_abs_scaled<Y, R>(y: &[Y], r: &[R]) -> Result<f64, MetricError>
where
    Y: Copy + Into<f64>,
    R: Copy + Into<f64>,
{
    same_len(r.len(), y.len())?;
    let y = widen_all(y, "y")?;
    let r = widen_all(r, "r")?;
    let mut max_abs = 0.0f64;
    let mut sq = Sum::default();
    for (&yi, &ri) in y.iter().zip(&r) {
        max_abs = max_abs.max((yi - ri).abs());
        sq.add(ri * ri);
    }
    let rms = (sq.value() / r.len() as f64).sqrt();
    Ok(max_abs / (rms + MAX_ABS_SCALED_EPS))
}

// ---------------------------------------------------------------------------------------------
// Full vocabulary (RFC-0042 §5.3)
// ---------------------------------------------------------------------------------------------

/// Log-sum-exp over the elements `keep` selects, in F64, subtracting the maximum first.
/// Returns −∞ when `keep` selects nothing.
fn lse_where(z: &[f64], keep: impl Fn(usize) -> bool) -> f64 {
    let max = z
        .iter()
        .enumerate()
        .filter(|&(i, _)| keep(i))
        .map(|(_, &v)| v)
        .fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let mut s = Sum::default();
    for (i, &v) in z.iter().enumerate() {
        if keep(i) {
            s.add((v - max).exp());
        }
    }
    max + s.value().ln()
}

/// The log-sum-exp of a full logit row, in F64 (RFC-0060 §5.3).
pub fn logsumexp<L: Copy + Into<f64>>(logits: &[L]) -> Result<f64, MetricError> {
    let z = widen_all(logits, "logits")?;
    Ok(lse_where(&z, |_| true))
}

/// `log_softmax(logits)` over the full vocabulary, in F64 (RFC-0042 §5.3).
pub fn log_softmax<L: Copy + Into<f64>>(logits: &[L]) -> Result<Vec<f64>, MetricError> {
    let z = widen_all(logits, "logits")?;
    let lse = lse_where(&z, |_| true);
    Ok(z.into_iter().map(|v| v - lse).collect())
}

/// `KL(P_ref ‖ P_cand) = Σ_v P_ref(v) · (log P_ref(v) − log P_cand(v))` in nats, over the
/// full vocabulary in F64 (RFC-0042 §5.3).
///
/// Both rows must be finite, so every log-probability is finite. A term whose `P_ref(v)`
/// underflows to 0 contributes 0 (`0 · log 0 = 0`). The result is not clamped: rounding can
/// leave it a few ULPs below 0 when the rows are equal up to a shift.
pub fn kl_full<R, C>(ref_logits: &[R], cand_logits: &[C]) -> Result<f64, MetricError>
where
    R: Copy + Into<f64>,
    C: Copy + Into<f64>,
{
    same_len(ref_logits.len(), cand_logits.len())?;
    let lp = log_softmax(ref_logits)?;
    let lq = log_softmax(cand_logits)?;
    let mut s = Sum::default();
    for (&a, &b) in lp.iter().zip(&lq) {
        s.add(a.exp() * (a - b));
    }
    Ok(s.value())
}

/// The token with the largest logit, ties to the lower token ID (RFC-0042 §5.3).
///
/// Softmax is strictly increasing, so this is also `argmax P`.
pub fn argmax<L: Copy + Into<f64>>(logits: &[L]) -> Result<u32, MetricError> {
    let z = widen_all(logits, "logits")?;
    let mut best = 0usize;
    for (i, &v) in z.iter().enumerate().skip(1) {
        if v > z[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

/// `top1_t = [argmax P_cand = argmax P_ref]`, ties to the lower token ID (RFC-0042 §5.3).
pub fn top1_agrees<R, C>(ref_logits: &[R], cand_logits: &[C]) -> Result<bool, MetricError>
where
    R: Copy + Into<f64>,
    C: Copy + Into<f64>,
{
    same_len(ref_logits.len(), cand_logits.len())?;
    Ok(argmax(ref_logits)? == argmax(cand_logits)?)
}

// ---------------------------------------------------------------------------------------------
// Stored references (RFC-0060 §5.3–§5.4, RFC-0042 §5.3)
// ---------------------------------------------------------------------------------------------

/// One token ID and its F32 logit, as a `RefBlock` stores them (RFC-0060 §5.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdLogit {
    /// The token ID.
    pub id: u32,
    /// Its logit.
    pub logit: f32,
}

/// The reference's side of one scored position, as a `RefBlock` holds it (RFC-0060 §5.3).
#[derive(Clone, Copy, Debug)]
pub struct RefView<'a> {
    /// The top-K set T: IDs and the reference's own F32 logits.
    pub top_k: &'a [IdLogit],
    /// The tail sample S: M IDs drawn with replacement from V∖T with probability
    /// `p_ref,j / r`, with their reference logits.
    pub tail_sample: &'a [IdLogit],
    /// `lse_ref`, the log-sum-exp over the full vocabulary, in F64.
    pub lse: f64,
    /// `log r`, the log-sum-exp over V∖T minus `lse_ref`, in F64. −∞ when T is the whole
    /// vocabulary.
    pub log_r: f64,
}

/// The candidate's side of one scored position, as the tap returns it (RFC-0060 §5.5).
#[derive(Clone, Copy, Debug)]
pub struct CandView<'a> {
    /// The candidate's logits at the reference's top-K IDs, in the same order.
    pub top_k: &'a [f32],
    /// The candidate's logits at the tail-sample IDs, in the same order.
    pub tail_sample: &'a [f32],
    /// `lse_cand`, its log-sum-exp over the full vocabulary, in F64.
    pub lse: f64,
    /// `log s = lse_cand(V∖T) − lse_cand`, in F64. −∞ only when T is the whole vocabulary.
    pub log_s: f64,
}

/// The tail bucket's `(r, log r − log s)`, or `None` when T is the whole vocabulary.
fn tail_bucket(log_r: f64, log_s: f64) -> Result<Option<(f64, f64)>, MetricError> {
    let r_empty = log_r == f64::NEG_INFINITY;
    let s_empty = log_s == f64::NEG_INFINITY;
    match (r_empty, s_empty) {
        (true, true) => Ok(None),
        (false, false) => {
            let log_r = finite(log_r, "log_r", None)?;
            let log_s = finite(log_s, "log_s", None)?;
            Ok(Some((log_r.exp(), log_r - log_s)))
        }
        _ => {
            // A NaN or +∞ is reported as non-finite rather than as a mismatch.
            finite_or_neg_inf(log_r, "log_r")?;
            finite_or_neg_inf(log_s, "log_s")?;
            Err(MetricError::TailBucketMismatch)
        }
    }
}

fn finite_or_neg_inf(x: f64, what: &'static str) -> Result<(), MetricError> {
    if x.is_finite() || x == f64::NEG_INFINITY {
        Ok(())
    } else {
        Err(MetricError::NonFinite { what, index: None })
    }
}

/// `Σ_i (log p_i − log q_i)` terms over aligned reference and candidate logits, weighted by
/// `weight(log p_i)`.
fn weighted_log_ratio(
    reference: &[IdLogit],
    candidate: &[f32],
    lse_ref: f64,
    lse_cand: f64,
    what_ref: &'static str,
    what_cand: &'static str,
    weight: impl Fn(f64) -> f64,
) -> Result<f64, MetricError> {
    same_len(reference.len(), candidate.len())?;
    let mut s = Sum::default();
    for (i, (r, &y)) in reference.iter().zip(candidate).enumerate() {
        let log_p = finite(f64::from(r.logit) - lse_ref, what_ref, Some(i))?;
        let log_q = finite(f64::from(y) - lse_cand, what_cand, Some(i))?;
        s.add(weight(log_p) * (log_p - log_q));
    }
    Ok(s.value())
}

/// `KL_K = Σ_{i∈T} p_i · (log p_i − log q_i) + r · (log r − log s)` in F64 (RFC-0060 §5.4,
/// ADR-0601 decision 2), with `log p_i = z_i − lse_ref`, `log q_i = y_i − lse_cand` and
/// `r = exp(log r)`.
///
/// It is the KL between the distributions coarse-grained to T plus one tail bucket, so by
/// the data-processing inequality it is a lower bound on the full KL. When T is the whole
/// vocabulary (`log r = log s = −∞`) the bucket term is 0 (`0 · log 0 = 0`). A non-finite
/// `log s` or `log q` otherwise is an error, never clamped.
pub fn kl_k(reference: &RefView<'_>, candidate: &CandView<'_>) -> Result<f64, MetricError> {
    let lse_ref = finite(reference.lse, "lse_ref", None)?;
    let lse_cand = finite(candidate.lse, "lse_cand", None)?;
    if reference.top_k.is_empty() {
        return Err(MetricError::Empty { what: "top_k" });
    }
    let head = weighted_log_ratio(
        reference.top_k,
        candidate.top_k,
        lse_ref,
        lse_cand,
        "log_p",
        "log_q",
        f64::exp,
    )?;
    let bucket = match tail_bucket(reference.log_r, candidate.log_s)? {
        None => 0.0,
        Some((r, log_ratio)) => r * log_ratio,
    };
    Ok(head + bucket)
}

/// `KL_tail = (r / M) · Σ_{j∈S} (log p_j − log q_j) − r · (log r − log s)` in F64
/// (RFC-0060 §5.4, ADR-0601 decision 2).
///
/// With S drawn from V∖T with probability `p_j / r`, its expectation is `KL − KL_K`, so
/// `KL_K + KL_tail` is an unbiased estimate of the full KL. When T is the whole vocabulary
/// there is no remainder and the result is 0; the sample must then be empty.
pub fn kl_tail(reference: &RefView<'_>, candidate: &CandView<'_>) -> Result<f64, MetricError> {
    let lse_ref = finite(reference.lse, "lse_ref", None)?;
    let lse_cand = finite(candidate.lse, "lse_cand", None)?;
    same_len(reference.tail_sample.len(), candidate.tail_sample.len())?;
    let Some((r, log_ratio)) = tail_bucket(reference.log_r, candidate.log_s)? else {
        same_len(0, reference.tail_sample.len())?;
        return Ok(0.0);
    };
    let m = reference.tail_sample.len();
    if m == 0 {
        return Err(MetricError::EmptyTailSample);
    }
    let sum = weighted_log_ratio(
        reference.tail_sample,
        candidate.tail_sample,
        lse_ref,
        lse_cand,
        "log_p_tail",
        "log_q_tail",
        |_| 1.0,
    )?;
    Ok((r / m as f64) * sum - r * log_ratio)
}

/// A reference block's metric fields built from a full F32 logit row (RFC-0060 §5.3).
///
/// This is what the `RefBlock` writer stores and what the tightness check compares against
/// the full KL. Drawing the tail sample needs a seeded generator, so the caller supplies its
/// IDs.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredRef {
    /// Top-K by logit, descending, ties to the lower ID.
    pub top_k: Vec<IdLogit>,
    /// The tail sample, in the order given.
    pub tail_sample: Vec<IdLogit>,
    /// `lse_ref`, in F64.
    pub lse: f64,
    /// `log r`, in F64, from the log-sum-exp over V∖T (never from `1 − Σ p`).
    pub log_r: f64,
}

impl StoredRef {
    /// Builds the block's metric fields from the reference's full logits, its top-`k` and the
    /// caller's tail-sample IDs. `k` above the vocabulary size takes the whole vocabulary.
    pub fn from_logits(logits: &[f32], k: usize, tail_ids: &[u32]) -> Result<Self, MetricError> {
        if k == 0 {
            return Err(MetricError::ZeroK);
        }
        let z = widen_all(logits, "logits")?;
        let mut order: Vec<usize> = (0..z.len()).collect();
        // Descending logit, then ascending ID: a stable sort on ascending IDs keeps ties in ID order.
        order.sort_by(|&a, &b| z[b].total_cmp(&z[a]));
        order.truncate(k.min(z.len()));
        let mut in_top = vec![false; z.len()];
        for &i in &order {
            in_top[i] = true;
        }
        let lse = lse_where(&z, |_| true);
        let log_r = lse_where(&z, |i| !in_top[i]) - lse;
        let top_k = order
            .iter()
            .map(|&i| IdLogit {
                id: i as u32,
                logit: logits[i],
            })
            .collect();
        let tail_sample = tail_ids
            .iter()
            .map(|&id| {
                let i = id as usize;
                if i >= z.len() {
                    Err(MetricError::IdOutOfRange { id, vocab: z.len() })
                } else if in_top[i] {
                    Err(MetricError::TailIdInTopK { id })
                } else {
                    Ok(IdLogit {
                        id,
                        logit: logits[i],
                    })
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(StoredRef {
            top_k,
            tail_sample,
            lse,
            log_r,
        })
    }

    /// Borrows the block as the estimators read it.
    pub fn view(&self) -> RefView<'_> {
        RefView {
            top_k: &self.top_k,
            tail_sample: &self.tail_sample,
            lse: self.lse,
            log_r: self.log_r,
        }
    }
}

/// The candidate's gathered values for one stored reference, built from its full logits, as
/// a `FullLocal` tap or the `Gather` tap returns them (RFC-0060 §5.5).
#[derive(Clone, Debug, PartialEq)]
pub struct GatheredCand {
    /// Logits at the reference's top-K IDs.
    pub top_k: Vec<f32>,
    /// Logits at the reference's tail-sample IDs.
    pub tail_sample: Vec<f32>,
    /// `lse_cand`, in F64.
    pub lse: f64,
    /// `log s = lse_cand(V∖T) − lse_cand`, in F64.
    pub log_s: f64,
}

impl GatheredCand {
    /// Gathers the candidate's values at the IDs `reference` names.
    ///
    /// The IDs are validated as [`StoredRef::from_logits`] validates them: every ID must be in
    /// the vocabulary, the top-K must hold no ID twice, and no tail-sample ID may be in the
    /// top-K, since the tail sample is drawn from V∖T (RFC-0060 §5.3).
    pub fn from_logits(logits: &[f32], reference: &RefView<'_>) -> Result<Self, MetricError> {
        let z = widen_all(logits, "logits")?;
        let vocab = z.len();
        let mut in_top = vec![false; vocab];
        let check = |id: u32| {
            if (id as usize) < vocab {
                Ok(id as usize)
            } else {
                Err(MetricError::IdOutOfRange { id, vocab })
            }
        };
        for e in reference.top_k {
            let i = check(e.id)?;
            if in_top[i] {
                return Err(MetricError::DuplicateTopKId { id: e.id });
            }
            in_top[i] = true;
        }
        let top_k = reference
            .top_k
            .iter()
            .map(|e| logits[e.id as usize])
            .collect();
        let tail_sample = reference
            .tail_sample
            .iter()
            .map(|e| {
                let i = check(e.id)?;
                if in_top[i] {
                    Err(MetricError::TailIdInTopK { id: e.id })
                } else {
                    Ok(logits[i])
                }
            })
            .collect::<Result<_, _>>()?;
        let lse = lse_where(&z, |_| true);
        let log_s = lse_where(&z, |i| !in_top[i]) - lse;
        Ok(GatheredCand {
            top_k,
            tail_sample,
            lse,
            log_s,
        })
    }

    /// Borrows the values as the estimators read them.
    pub fn view(&self) -> CandView<'_> {
        CandView {
            top_k: &self.top_k,
            tail_sample: &self.tail_sample,
            lse: self.lse,
            log_s: self.log_s,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Aggregates (RFC-0042 §5.3, RFC-0060 §5.4)
// ---------------------------------------------------------------------------------------------

/// The mean of `values`, in F64.
pub fn mean(values: &[f64]) -> Result<f64, MetricError> {
    let v = widen_all(values, "values")?;
    let mut s = Sum::default();
    for x in &v {
        s.add(*x);
    }
    Ok(s.value() / v.len() as f64)
}

/// The nearest-rank percentile: the value at rank `⌈pct / 100 · n⌉` of the sorted values
/// (RFC-0042 §5.3 uses it for p99 over positions). `pct` is in (0, 100].
pub fn percentile_nearest_rank(values: &[f64], pct: f64) -> Result<f64, MetricError> {
    if !(pct > 0.0 && pct <= 100.0) {
        return Err(MetricError::InvalidPercentile);
    }
    let mut v = widen_all(values, "values")?;
    v.sort_by(f64::total_cmp);
    let n = v.len();
    // The rank is clamped to [1, n].
    let rank = ((pct * n as f64) / 100.0).ceil() as usize;
    Ok(v[rank.clamp(1, n) - 1])
}

/// The nearest-rank 99th percentile (RFC-0042 §5.3).
pub fn p99_nearest_rank(values: &[f64]) -> Result<f64, MetricError> {
    percentile_nearest_rank(values, P99)
}

/// Top-1 disagreement `1 − mean(top1_t)` (RFC-0042 §5.3).
pub fn top1_disagreement(top1: &[bool]) -> Result<f64, MetricError> {
    if top1.is_empty() {
        return Err(MetricError::Empty { what: "top1" });
    }
    let misses = top1.iter().filter(|&&t| !t).count();
    Ok(misses as f64 / top1.len() as f64)
}

/// One scored position's metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionScore {
    /// The KL the gate reads at this position: the full KL, or `KL_K` from a stored reference.
    pub kl: f64,
    /// `KL_tail` at this position, when computed from a stored reference.
    pub kl_tail: Option<f64>,
    /// `top1_t`.
    pub top1: bool,
}

/// The aggregates of a set of positions (RFC-0042 §5.3, RFC-0060 §5.4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aggregate {
    /// Scored positions.
    pub positions: usize,
    /// Mean KL.
    pub kl_mean: f64,
    /// Nearest-rank p99 KL.
    pub kl_p99: f64,
    /// Mean `KL_tail`, when every position carries one.
    pub kl_tail_mean: Option<f64>,
    /// `1 − mean(top1_t)`.
    pub top1_disagreement: f64,
}

/// Aggregates `scores` (RFC-0042 §5.3).
pub fn aggregate(scores: &[PositionScore]) -> Result<Aggregate, MetricError> {
    if scores.is_empty() {
        return Err(MetricError::Empty { what: "scores" });
    }
    let kl: Vec<f64> = scores.iter().map(|s| s.kl).collect();
    let tails: Vec<f64> = scores.iter().filter_map(|s| s.kl_tail).collect();
    let kl_tail_mean = match tails.len() {
        0 => None,
        n if n == scores.len() => Some(mean(&tails)?),
        _ => return Err(MetricError::MixedTail),
    };
    let top1: Vec<bool> = scores.iter().map(|s| s.top1).collect();
    Ok(Aggregate {
        positions: scores.len(),
        kl_mean: mean(&kl)?,
        kl_p99: p99_nearest_rank(&kl)?,
        kl_tail_mean,
        top1_disagreement: top1_disagreement(&top1)?,
    })
}

/// Pooled and per-category aggregates (RFC-0042 §5.3).
#[derive(Clone, Debug, PartialEq)]
pub struct Aggregates<C: Ord> {
    /// Every position.
    pub pooled: Aggregate,
    /// Positions per corpus category.
    pub by_category: BTreeMap<C, Aggregate>,
}

/// Aggregates `scores` pooled and per corpus category (RFC-0042 §5.3).
pub fn aggregate_by_category<C: Ord + Clone>(
    scores: &[(C, PositionScore)],
) -> Result<Aggregates<C>, MetricError> {
    let pooled: Vec<PositionScore> = scores.iter().map(|(_, s)| *s).collect();
    let mut groups: BTreeMap<C, Vec<PositionScore>> = BTreeMap::new();
    for (c, s) in scores {
        groups.entry(c.clone()).or_default().push(*s);
    }
    let by_category = groups
        .into_iter()
        .map(|(c, v)| aggregate(&v).map(|a| (c, a)))
        .collect::<Result<_, _>>()?;
    Ok(Aggregates {
        pooled: aggregate(&pooled)?,
        by_category,
    })
}

// ---------------------------------------------------------------------------------------------
// The tightness rule (RFC-0060 §5.4, RFC-0042 §5.3, ADR-0601 decision 2)
// ---------------------------------------------------------------------------------------------

/// The tightness bound `max(0.01 · KL, 0.01 · C_kl_mean(d))` (RFC-0060 §5.4).
///
/// The absolute term keeps the bound defined, and above 0, as KL → 0.
pub fn tightness_bound(kl: f64, c_kl_mean: f64) -> f64 {
    (TIGHTNESS_REL * kl).max(TIGHTNESS_CEILING_FRACTION * c_kl_mean)
}

/// One statistic's tightness comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TightnessCheck {
    /// The statistic of the exact KL from full logits.
    pub kl: f64,
    /// The same statistic of `KL_K`.
    pub kl_k: f64,
    /// `|KL − KL_K|`.
    pub diff: f64,
    /// `max(0.01 · KL, 0.01 · C_kl_mean(d))`.
    pub bound: f64,
    /// `diff ≤ bound`.
    pub pass: bool,
}

impl TightnessCheck {
    /// Compares one statistic of KL and `KL_K` against the bound (RFC-0060 §5.4).
    pub fn new(kl: f64, kl_k: f64, c_kl_mean: f64) -> Result<Self, MetricError> {
        let kl = finite(kl, "kl", None)?;
        let kl_k = finite(kl_k, "kl_k", None)?;
        if !(c_kl_mean.is_finite() && c_kl_mean >= 0.0) {
            return Err(MetricError::InvalidCeiling);
        }
        let diff = (kl - kl_k).abs();
        let bound = tightness_bound(kl, c_kl_mean);
        Ok(TightnessCheck {
            kl,
            kl_k,
            diff,
            bound,
            pass: diff <= bound,
        })
    }
}

/// The tightness rule for one set of positions, at the mean and at p99 (RFC-0060 §5.4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tightness {
    /// Mean KL against mean `KL_K`.
    pub mean: TightnessCheck,
    /// Nearest-rank p99 KL against nearest-rank p99 `KL_K`.
    pub p99: TightnessCheck,
}

impl Tightness {
    /// Both statistics are within the bound.
    pub fn pass(&self) -> bool {
        self.mean.pass && self.p99.pass
    }
}

/// Checks `|KL − KL_K| ≤ max(0.01 · KL, 0.01 · C_kl_mean(d))` at the mean and at p99 over a
/// set of positions (RFC-0060 §5.4). `kl[t]` is the exact KL from full logits and `kl_k[t]`
/// is `KL_K` at the same position; `c_kl_mean` is the dtype class's mean-KL ceiling
/// `C_kl_mean(d)` from the tolerance table (RFC-0042 §6.2).
///
/// Each statistic is taken over its own values and the two statistics are compared, so at
/// p99 the rule compares the p99 of KL with the p99 of `KL_K`: the quantity the gate reads.
pub fn check_tightness(kl: &[f64], kl_k: &[f64], c_kl_mean: f64) -> Result<Tightness, MetricError> {
    same_len(kl.len(), kl_k.len())?;
    Ok(Tightness {
        mean: TightnessCheck::new(mean(kl)?, mean(kl_k)?, c_kl_mean)?,
        p99: TightnessCheck::new(p99_nearest_rank(kl)?, p99_nearest_rank(kl_k)?, c_kl_mean)?,
    })
}

/// [`check_tightness`] per corpus category (RFC-0060 §5.4). Each row is
/// `(category, KL, KL_K)` at one position.
pub fn check_tightness_by_category<C: Ord + Clone>(
    rows: &[(C, f64, f64)],
    c_kl_mean: f64,
) -> Result<BTreeMap<C, Tightness>, MetricError> {
    let mut groups: BTreeMap<C, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for (c, kl, kl_k) in rows {
        let g = groups.entry(c.clone()).or_default();
        g.0.push(*kl);
        g.1.push(*kl_k);
    }
    groups
        .into_iter()
        .map(|(c, (kl, kl_k))| check_tightness(&kl, &kl_k, c_kl_mean).map(|t| (c, t)))
        .collect()
}

#[cfg(test)]
mod tests;
