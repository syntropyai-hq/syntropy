// SPDX-License-Identifier: MIT OR Apache-2.0

//! The one trace context (RFC-0053 §3.2; RFC-0031 §6.3's `TRACE` item).

use serde::{Deserialize, Serialize};

/// W3C trace context as Syntropy carries it between stages and back: version, trace-id, span-id
/// and flags, 26 bytes on the wire (RFC-0031 §6.3's `TRACE` item owns the layout; RFC-0053
/// §3.2, note of 2026-10-22). `SequenceOpen::trace`, `ActivationFrame::trace` and the `Token`
/// frame's `TRACE` items all hold this one type. Owner: RFC-0053.
///
/// Its `flags` are Syntropy's, not a W3C trace-flags byte: bit 0 [`TraceContext::SAMPLED`] and
/// bit 1 [`TraceContext::FULL_TRACE`]. Ingress keeps only bit 0 of an incoming `traceparent`,
/// and sets a returned one's flags itself (RFC-0053 §3.2, note of 2026-10-22 for #1766).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceContext {
    /// The W3C version byte; 0 today.
    pub version: u8,
    /// The trace's ID.
    pub trace_id: [u8; 16],
    /// The sender's span, the parent of the receiver's: RFC-0053 §3.2's parent span. Every
    /// stage's step span is a sibling under the sequence span, linked to its predecessor.
    pub span_id: [u8; 8],
    /// Bit 0 [`TraceContext::SAMPLED`], bit 1 [`TraceContext::FULL_TRACE`]; the others are 0.
    pub flags: u8,
}

impl TraceContext {
    /// Bit 0: this step is sampled, so every stage records its spans (RFC-0053 §3.3).
    pub const SAMPLED: u8 = 1 << 0;
    /// Bit 1: the request asked for a full trace, so every step is sampled (RFC-0053 §3.3). Set
    /// only by Syntropy's policy, never copied from a client's flags.
    pub const FULL_TRACE: u8 = 1 << 1;

    /// Whether this step is sampled.
    pub fn sampled(&self) -> bool {
        self.flags & TraceContext::SAMPLED != 0
    }

    /// Whether the request asked for a full trace.
    pub fn full_trace(&self) -> bool {
        self.flags & TraceContext::FULL_TRACE != 0
    }
}
