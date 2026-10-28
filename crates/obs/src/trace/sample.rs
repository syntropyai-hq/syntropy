// SPDX-License-Identifier: MIT OR Apache-2.0

//! The head's per-step sampling decision (RFC-0053 §3.3). A hot-thread module: the
//! `hot-threads` lint step rejects `tracing` here.
//!
//! The head decides once per step, and every stage obeys the frame's flag, so a sampled step has
//! spans on every stage. A step is sampled when:
//!
//! 1. the request asked for a full trace (`TraceContext::FULL_TRACE`); or
//! 2. it is one of a sequence's first [`FIRST_DECODE_STEPS`] decode steps, which hold the cold-cache
//!    effects users ask about; or
//! 3. its [`step_hash`] falls under `obs.trace.step_sample`, 1/64 by default: `hash < ⌊p · 2⁶⁴⌋`.
//!
//! **The hash** is FNV-1a-64 (offset basis `0xcbf2_9ce4_8422_2325`, prime `0x100_0000_01b3`) over
//! the 16 trace-ID bytes, then the step index as a `u64` little-endian. It is integer arithmetic
//! only, so the same (trace ID, step) gives the same decision on every host, every build and in
//! every replay. `step` is the sequence's step counter, prefill chunks and decode steps alike; a
//! prefill chunk is sampled by rules 1 and 3 only (RFC-0053 §3.3, note of 2026-10-28).

use syntropy_core::sequence::TraceContext;

/// A sequence's first decode steps, every one sampled (RFC-0001 §13).
pub const FIRST_DECODE_STEPS: u32 = 4;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a-64 over `trace_id` then `step` as a `u64` little-endian (module docs).
#[must_use]
pub const fn step_hash(trace_id: &[u8; 16], step: u64) -> u64 {
    let mut h = FNV_OFFSET;
    let mut i = 0;
    while i < 16 {
        h ^= trace_id[i] as u64;
        h = h.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    let step = step.to_le_bytes();
    let mut j = 0;
    while j < 8 {
        h ^= step[j] as u64;
        h = h.wrapping_mul(FNV_PRIME);
        j += 1;
    }
    h
}

/// What a step is, for the sampling rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    /// A prefill chunk.
    Prefill,
    /// A decode or verify step, the sequence's `index`-th since prefill ended.
    Decode {
        /// Its index among the sequence's decode steps, from 0.
        index: u32,
    },
}

/// The head's sampler, built once from `obs.trace.step_sample`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepSampler {
    /// A step is sampled by rule 3 when its hash is below this; `u64::MAX` samples every step.
    threshold: u64,
}

impl StepSampler {
    /// The sampler for a step probability `p` in [0, 1]: `⌊p · 2⁶⁴⌋`, every step at `p ≥ 1`,
    /// none (but rules 1 and 2) at `p ≤ 0`.
    #[must_use]
    pub fn new(p: f64) -> StepSampler {
        let threshold = if p.is_nan() || p <= 0.0 {
            0
        } else if p >= 1.0 {
            u64::MAX
        } else {
            // p < 1, so the product is below 2⁶⁴ and the cast does not saturate.
            (p * 18_446_744_073_709_551_616.0) as u64
        };
        StepSampler { threshold }
    }

    /// The threshold rule 3 compares a step's hash with.
    #[must_use]
    pub fn threshold(&self) -> u64 {
        self.threshold
    }

    /// Whether step `step` of the trace `trace_id` is sampled; `full` is the request's full-trace
    /// bit. Allocates nothing and takes no lock.
    #[inline]
    #[must_use]
    pub fn sampled(&self, trace_id: &[u8; 16], step: u64, kind: StepKind, full: bool) -> bool {
        full || matches!(kind, StepKind::Decode { index } if index < FIRST_DECODE_STEPS)
            || self.threshold == u64::MAX
            || step_hash(trace_id, step) < self.threshold
    }

    /// The trace context the head puts on a step's frames, or `None` for a step that is not
    /// sampled (RFC-0053 §3.2: `ActivationFrame::trace` is set only on sampled steps). `sequence`
    /// is the sequence's context from `SequenceOpen`; `span_id` is this step's span at the head.
    #[inline]
    #[must_use]
    pub fn step_context(
        &self,
        sequence: &TraceContext,
        step: u64,
        kind: StepKind,
        span_id: [u8; 8],
    ) -> Option<TraceContext> {
        let full = sequence.full_trace();
        self.sampled(&sequence.trace_id, step, kind, full)
            .then(|| TraceContext {
                version: 0,
                trace_id: sequence.trace_id,
                span_id,
                flags: TraceContext::SAMPLED | if full { TraceContext::FULL_TRACE } else { 0 },
            })
    }
}
