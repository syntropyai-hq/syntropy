// SPDX-License-Identifier: MIT OR Apache-2.0

//! W3C `traceparent` in and out at ingress (RFC-0053 §3.2).
//!
//! - **In.** [`Traceparent::parse`] reads the header as W3C Trace Context gives it: version,
//!   trace-id, parent-id and trace-flags, lowercase hex, with an all-zero ID or version `ff`
//!   refused. [`IngressTrace::start`] continues its trace-id and keeps only bit 0 (`sampled`) of
//!   its flags. Bit 1 is W3C Level 2's `random-trace-id`, which a client may set; it is kept only
//!   to be echoed, never read as Syntropy's full-trace bit.
//! - **Full traces** come only from Syntropy's policy: the request's `syntropy.trace = true` or
//!   `Syntropy-Trace: 1`, or an incoming sampled bit while `obs.trace.honor_remote_sampled` is
//!   on. So an incoming `02` never enables one, with the key set or not, and `03` behaves as `01`.
//! - **The sequence's context** ([`IngressTrace::sequence_context`]) takes its flags from that
//!   policy alone: the full-trace bit or nothing. The head still decides per step.
//! - **Out.** [`IngressTrace::response_traceparent`] is `Some` only when the request carried a
//!   valid `traceparent` (W3C §6.3). Its flags are set here, never copied from `TraceContext`:
//!   bit 0 is 1, since the root spans are always recorded; bit 1 equals the incoming bit 1, since
//!   the trace-id was continued; Syntropy's full-trace bit never leaves the node.
//!   [`IngressTrace::w3c_flags`] gives the same byte for exported spans, with bit 1 set for a
//!   trace-id ingress minted from random bytes.
//!
//! A header that does not parse is treated as absent, as W3C says: ingress starts a new trace and
//! returns no `traceparent`, because the caller took part in no trace Syntropy continued.

use syntropy_core::Entropy;
use syntropy_core::sequence::TraceContext;

/// A parsed W3C `traceparent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Traceparent {
    /// The version byte; 0 today.
    pub version: u8,
    /// The trace-id.
    pub trace_id: [u8; 16],
    /// The parent-id: the caller's span.
    pub parent_id: [u8; 8],
    /// The trace-flags byte, as sent.
    pub flags: u8,
}

/// Bit 0 of W3C trace-flags: `sampled`.
pub const W3C_SAMPLED: u8 = 1 << 0;
/// Bit 1 of W3C trace-flags (Level 2): `random-trace-id`.
pub const W3C_RANDOM_TRACE_ID: u8 = 1 << 1;

fn hex_byte(pair: &[u8]) -> Option<u8> {
    let digit = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    Some(digit(pair[0])? << 4 | digit(pair[1])?)
}

fn hex_into(field: &[u8], out: &mut [u8]) -> Option<()> {
    if field.len() != out.len() * 2 {
        return None;
    }
    for (o, pair) in out.iter_mut().zip(field.as_chunks::<2>().0) {
        *o = hex_byte(pair)?;
    }
    Some(())
}

impl Traceparent {
    /// Parses a `traceparent` header value, or `None` for one W3C says to ignore: not
    /// `vv-<32 hex>-<16 hex>-ff` in lowercase hex, version `ff`, an all-zero trace-id or
    /// parent-id, or, for version 0, anything after the flags. A later version may carry more
    /// fields after a `-`, which are ignored.
    #[must_use]
    pub fn parse(header: &str) -> Option<Traceparent> {
        let b = header.trim_matches([' ', '\t']).as_bytes();
        if b.len() < 55 || b[2] != b'-' || b[35] != b'-' || b[52] != b'-' {
            return None;
        }
        let version = hex_byte(&b[0..2])?;
        if version == 0xff || (version == 0 && b.len() != 55) || (b.len() > 55 && b[55] != b'-') {
            return None;
        }
        let mut trace_id = [0u8; 16];
        let mut parent_id = [0u8; 8];
        hex_into(&b[3..35], &mut trace_id)?;
        hex_into(&b[36..52], &mut parent_id)?;
        let flags = hex_byte(&b[53..55])?;
        if trace_id == [0; 16] || parent_id == [0; 8] {
            return None;
        }
        Some(Traceparent {
            version,
            trace_id,
            parent_id,
            flags,
        })
    }

    /// The header value: `vv-<trace-id>-<parent-id>-<flags>`, lowercase hex.
    #[must_use]
    pub fn format(&self) -> String {
        let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        format!(
            "{:02x}-{}-{}-{:02x}",
            self.version,
            hex(&self.trace_id),
            hex(&self.parent_id),
            self.flags
        )
    }
}

/// What ingress keeps of an incoming `traceparent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Incoming {
    /// Bit 0, the only bit ingress reads.
    sampled: bool,
    /// Bit 1, kept only to be echoed.
    random_trace_id: bool,
}

/// A request's trace at ingress (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IngressTrace {
    /// The trace: continued from the request's `traceparent`, or minted.
    pub trace_id: [u8; 16],
    /// The root span, `syntropy.api.request`.
    pub root_span: [u8; 8],
    /// The caller's span, the root span's parent; zero when ingress minted the trace.
    pub parent_id: [u8; 8],
    /// Whether every step is traced.
    pub full: bool,
    incoming: Option<Incoming>,
}

impl IngressTrace {
    /// The trace of a request that carried `traceparent` (the raw header, if any) and asked for
    /// a full trace or not (`trace_requested`: `syntropy.trace = true` or `Syntropy-Trace: 1`).
    /// `honor_remote_sampled` is `obs.trace.honor_remote_sampled`. New IDs come from `entropy`,
    /// never zero.
    pub fn start(
        traceparent: Option<&str>,
        trace_requested: bool,
        honor_remote_sampled: bool,
        entropy: &dyn Entropy,
    ) -> IngressTrace {
        let parsed = traceparent.and_then(Traceparent::parse);
        let incoming = parsed.map(|t| Incoming {
            sampled: t.flags & W3C_SAMPLED != 0,
            random_trace_id: t.flags & W3C_RANDOM_TRACE_ID != 0,
        });
        let trace_id = parsed.map_or_else(|| nonzero(entropy), |t| t.trace_id);
        let full = trace_requested || (honor_remote_sampled && incoming.is_some_and(|i| i.sampled));
        IngressTrace {
            trace_id,
            root_span: nonzero(entropy),
            parent_id: parsed.map_or([0; 8], |t| t.parent_id),
            full,
            incoming,
        }
    }

    /// The context ingress puts in `SequenceOpen`: this trace, the given span (`sched.prefill`'s
    /// or `sched.decode`'s), and Syntropy's flags alone: `FULL_TRACE | SAMPLED` for a full trace,
    /// else none, so the head decides each step.
    #[must_use]
    pub fn sequence_context(&self, span_id: [u8; 8]) -> TraceContext {
        TraceContext {
            version: 0,
            trace_id: self.trace_id,
            span_id,
            flags: if self.full {
                TraceContext::FULL_TRACE | TraceContext::SAMPLED
            } else {
                0
            },
        }
    }

    /// The W3C trace-flags of everything this node emits for the trace: bit 0 set; bit 1 the
    /// incoming bit 1 for a continued trace-id, and 1 for one minted from random bytes; never
    /// the full-trace bit.
    #[must_use]
    pub fn w3c_flags(&self) -> u8 {
        let random = self.incoming.is_none_or(|i| i.random_trace_id);
        W3C_SAMPLED | if random { W3C_RANDOM_TRACE_ID } else { 0 }
    }

    /// The `traceparent` to return, for the root span, only when the request carried one.
    #[must_use]
    pub fn response_traceparent(&self) -> Option<String> {
        self.incoming.map(|_| {
            Traceparent {
                version: 0,
                trace_id: self.trace_id,
                parent_id: self.root_span,
                flags: self.w3c_flags(),
            }
            .format()
        })
    }
}

/// Random bytes from `entropy`, drawn again while all zero (W3C refuses an all-zero ID). A source
/// that gives zeros 16 times running has its last byte set to 1, so this always returns.
fn nonzero<const N: usize>(entropy: &dyn Entropy) -> [u8; N] {
    let mut out = [0u8; N];
    for _ in 0..16 {
        entropy.fill(&mut out);
        if out != [0; N] {
            return out;
        }
    }
    out[N - 1] = 1;
    out
}
