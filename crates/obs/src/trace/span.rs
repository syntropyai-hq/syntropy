// SPDX-License-Identifier: MIT OR Apache-2.0

//! The span model and the hot-thread span record (RFC-0053 §3.1, §3.4). A hot-thread module: the
//! `hot-threads` lint step rejects `tracing` here.
//!
//! **Names.** Every span is `syntropy.<crate>.<operation>` (RFC-0001 §13). [`SpanName`] is the
//! closed list of RFC-0053 §3.1, and a record names one by its [`SpanNameId`], so a hot thread
//! touches no string. The daemon resolves the ID when it builds the full span.
//!
//! **The record.** RFC-0053 §3.4 lists `trace_id`, `span_id`, `parent_id`, `name`, `start`, `dur`
//! and six `u32` attributes, which is 70 bytes, not the 64 it names. [`SpanRecord`] keeps 64 bytes
//! and four attributes (RFC-0053 §3.4, note of 2026-10-28): the daemon takes a step's per-term
//! durations from its matching `StepRecord`, so a span's own attributes need at most four
//! ([`SpanName::attrs`]). The struct is `repr(C)` and 64 bytes, a build-time assertion holds it
//! there, and [`SpanRecord::encode`] writes its fields little-endian at the same offsets:
//!
//! ```text
//! 0–15   trace_id        16–23  span_id         24–31  parent_id
//! 32–39  start (u64, MonoTime nanoseconds)      40–43  dur (u32 ns)
//! 44–45  name (u16)      46–47  attr_count (u16, ≤ 4)
//! 48–63  attrs (4 × u32); unused ones are 0
//! ```

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use syntropy_core::MonoTime;

use crate::ring::RecordSink;
use crate::telemetry::{self, TelemetryRecord};

/// A span name's index in [`SpanName::ALL`], as a record carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SpanNameId(pub u16);

/// The spans of RFC-0053 §3.1.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpanName {
    /// `syntropy.api.request`, the root span, at ingress.
    Request,
    /// `syntropy.api.route`, at ingress.
    Route,
    /// `syntropy.api.place`, automatic placement's wait, at ingress.
    Place,
    /// `syntropy.api.queue`, the admission wait, at ingress.
    Queue,
    /// `syntropy.sched.prefill`, at the head.
    Prefill,
    /// `syntropy.sched.decode`, at the head.
    Decode,
    /// `syntropy.exec.step`, one stage's step.
    Step,
    /// `syntropy.net.hop`, one frame's hop between stages.
    Hop,
    /// `syntropy.cluster.recovery`, at the coordinator or the head.
    Recovery,
}

impl SpanName {
    /// Every name, at its [`SpanNameId`]. A name is appended, never moved or reused: the ID is
    /// part of the span record's encoding.
    pub const ALL: [SpanName; 9] = [
        SpanName::Request,
        SpanName::Route,
        SpanName::Place,
        SpanName::Queue,
        SpanName::Prefill,
        SpanName::Decode,
        SpanName::Step,
        SpanName::Hop,
        SpanName::Recovery,
    ];

    /// The span's name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            SpanName::Request => "syntropy.api.request",
            SpanName::Route => "syntropy.api.route",
            SpanName::Place => "syntropy.api.place",
            SpanName::Queue => "syntropy.api.queue",
            SpanName::Prefill => "syntropy.sched.prefill",
            SpanName::Decode => "syntropy.sched.decode",
            SpanName::Step => "syntropy.exec.step",
            SpanName::Hop => "syntropy.net.hop",
            SpanName::Recovery => "syntropy.cluster.recovery",
        }
    }

    /// Its ID. The match names every span, so a new one fails to compile until it has an ID.
    #[must_use]
    pub const fn id(self) -> SpanNameId {
        SpanNameId(match self {
            SpanName::Request => 0,
            SpanName::Route => 1,
            SpanName::Place => 2,
            SpanName::Queue => 3,
            SpanName::Prefill => 4,
            SpanName::Decode => 5,
            SpanName::Step => 6,
            SpanName::Hop => 7,
            SpanName::Recovery => 8,
        })
    }

    /// The span named by `id`, if any.
    #[must_use]
    pub fn from_id(id: SpanNameId) -> Option<SpanName> {
        SpanName::ALL.get(usize::from(id.0)).copied()
    }

    /// The keys of the record's `u32` attributes, in order (at most four). A request's root
    /// span carries strings too, which ingress builds without a record: [`GEN_AI_ATTRS`] and
    /// [`SYNTROPY_ATTRS`].
    #[must_use]
    pub const fn attrs(self) -> &'static [&'static str] {
        match self {
            SpanName::Request => &["gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"],
            SpanName::Route => &["syntropy.route.candidates"],
            SpanName::Place | SpanName::Queue => &[],
            SpanName::Prefill => &["syntropy.prompt_tokens", "syntropy.cached_tokens"],
            SpanName::Decode => &[
                "syntropy.tokens",
                "syntropy.windows",
                "syntropy.accepted",
                "syntropy.unattributed_ns",
            ],
            SpanName::Step => &["syntropy.stage", "syntropy.step.kind", "syntropy.step.unit"],
            SpanName::Hop => &[
                "syntropy.hop.from",
                "syntropy.hop.to",
                "syntropy.hop.transport",
                "syntropy.hop.bytes",
            ],
            SpanName::Recovery => &["syntropy.recovery.kind", "syntropy.recovery.gap_ns"],
        }
    }
}

/// The OpenTelemetry GenAI semantic-convention keys on the root span (RFC-0053 §3.1).
pub const GEN_AI_ATTRS: [&str; 5] = [
    "gen_ai.operation.name",
    "gen_ai.request.model",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.response.finish_reasons",
];

/// Syntropy's own keys on the root span (RFC-0053 §3.1).
pub const SYNTROPY_ATTRS: [&str; 4] = [
    "syntropy.fidelity",
    "syntropy.instance",
    "syntropy.plan_epoch",
    "syntropy.approx_applied",
];

/// One span as a hot thread records it (module docs). Spans never carry prompt or output text
/// or token IDs (RFC-0053 §3.1).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanRecord {
    /// The trace.
    pub trace_id: [u8; 16],
    /// This span.
    pub span_id: [u8; 8],
    /// Its parent; zero for a root span.
    pub parent_id: [u8; 8],
    /// When it started, on the node's monotonic timeline.
    pub start: MonoTime,
    /// How long it took, in nanoseconds.
    pub dur: u32,
    /// Which span it is.
    pub name: SpanNameId,
    /// How many of `attrs` are used, at most four.
    pub attr_count: u16,
    /// Its attributes, the keys [`SpanName::attrs`] gives, in order; unused ones are 0.
    pub attrs: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<SpanRecord>() == SpanRecord::BYTES);

impl SpanRecord {
    /// The bytes of one record.
    pub const BYTES: usize = 64;

    /// The record's encoding (module docs). Part of `ipc_protocol` (RFC-0001 §3.5).
    #[must_use]
    pub fn encode(&self) -> [u8; SpanRecord::BYTES] {
        let mut out = [0u8; SpanRecord::BYTES];
        out[0..16].copy_from_slice(&self.trace_id);
        out[16..24].copy_from_slice(&self.span_id);
        out[24..32].copy_from_slice(&self.parent_id);
        out[32..40].copy_from_slice(&self.start.0.to_le_bytes());
        out[40..44].copy_from_slice(&self.dur.to_le_bytes());
        out[44..46].copy_from_slice(&self.name.0.to_le_bytes());
        out[46..48].copy_from_slice(&self.attr_count.to_le_bytes());
        for (i, attr) in self.attrs.iter().enumerate() {
            out[48 + i * 4..52 + i * 4].copy_from_slice(&attr.to_le_bytes());
        }
        out
    }

    /// Decodes one record, or `None` for bytes that are not one: the wrong length, an unknown
    /// name, more than four attributes, or a non-zero unused attribute.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<SpanRecord> {
        let bytes: &[u8; SpanRecord::BYTES] = bytes.try_into().ok()?;
        let at = |from: usize, to: usize| &bytes[from..to];
        let u32_at = |i: usize| u32::from_le_bytes(at(i, i + 4).try_into().unwrap_or([0; 4]));
        let name = SpanNameId(u16::from_le_bytes([bytes[44], bytes[45]]));
        SpanName::from_id(name)?;
        let attr_count = u16::from_le_bytes([bytes[46], bytes[47]]);
        if attr_count > 4 {
            return None;
        }
        let attrs = [u32_at(48), u32_at(52), u32_at(56), u32_at(60)];
        if attrs[usize::from(attr_count)..].iter().any(|&a| a != 0) {
            return None;
        }
        Some(SpanRecord {
            trace_id: at(0, 16).try_into().ok()?,
            span_id: at(16, 24).try_into().ok()?,
            parent_id: at(24, 32).try_into().ok()?,
            start: MonoTime(u64::from_le_bytes(at(32, 40).try_into().ok()?)),
            dur: u32_at(40),
            name,
            attr_count,
            attrs,
        })
    }
}

/// A hot thread's writer of its own span ring (RFC-0053 §3.4). The ring is in process: the thread
/// is its only producer and housekeeping its only consumer, which batches the records into the
/// `telemetry` ring ([`span_batches`]). One thread owns the writer, so it takes no lock.
#[derive(Debug)]
pub struct SpanWriter<S: RecordSink> {
    sink: S,
    written: u64,
    dropped: u64,
}

impl<S: RecordSink> SpanWriter<S> {
    /// The writer of the ring `sink` feeds.
    pub fn new(sink: S) -> SpanWriter<S> {
        SpanWriter {
            sink,
            written: 0,
            dropped: 0,
        }
    }

    /// Writes `record`; whether it went in. A full ring drops and counts it (RFC-0001 §8.4 rule
    /// 8). Never allocates, formats or blocks.
    #[inline]
    pub fn record(&mut self, record: &SpanRecord) -> bool {
        if self.sink.push(&record.encode()).is_ok() {
            self.written += 1;
            true
        } else {
            self.dropped += 1;
            false
        }
    }

    /// Records written so far.
    #[must_use]
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Records dropped so far.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The sink, for tests that drain it.
    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }
}

/// Housekeeping's batching (RFC-0053 §3.4): `records` as [`TelemetryRecord::Spans`], each as many
/// records as fit a `telemetry` slot of `max_record` bytes, in order; `None` when not even one
/// record fits a slot. Not a hot-thread function.
#[must_use]
pub fn span_batches(records: &[SpanRecord], max_record: usize) -> Option<Vec<TelemetryRecord>> {
    let mut batches = Vec::new();
    let mut batch: SmallVec<[SpanRecord; 8]> = SmallVec::new();
    let mut buf = Vec::new();
    for record in records {
        batch.push(*record);
        buf.clear();
        // A span record always serializes: it holds only integers and byte arrays.
        telemetry::encode(
            &TelemetryRecord::Spans {
                records: batch.clone(),
            },
            &mut buf,
        )
        .ok()?;
        if buf.len() > max_record {
            let last = batch.pop().expect("just pushed");
            if batch.is_empty() {
                return None;
            }
            batches.push(TelemetryRecord::Spans {
                records: std::mem::take(&mut batch),
            });
            batch.push(last);
        }
    }
    if !batch.is_empty() {
        batches.push(TelemetryRecord::Spans { records: batch });
    }
    Some(batches)
}
