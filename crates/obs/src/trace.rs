// SPDX-License-Identifier: MIT OR Apache-2.0

//! Local tracing (RFC-0053 §3.1–§3.4; PROD-E8-T07).
//!
//! - [`span`]: the span model. [`SpanName`] holds the names of RFC-0053 §3.1, all
//!   `syntropy.<crate>.<operation>`, with the GenAI and Syntropy attribute keys of the root span.
//!   [`SpanRecord`] is the fixed 64-byte record hot threads write, and [`SpanWriter`] writes it into
//!   the thread's own ring without a lock or an allocation. [`span_batches`] turns drained records
//!   into [`TelemetryRecord::Spans`](crate::TelemetryRecord::Spans) for the `telemetry` ring.
//! - [`sample`]: the head's per-step decision ([`StepSampler`]). Every step of the first four
//!   decode steps, every step of a full trace, and otherwise a step whose [`step_hash`] of
//!   (trace ID, step) falls under `obs.trace.step_sample`. The hash is FNV-1a-64, so the same
//!   (trace ID, step) is sampled the same way on every host and in every replay.
//! - [`context`]: W3C `traceparent` in and out at ingress ([`Traceparent`], [`IngressTrace`]):
//!   only bit 0 of an incoming flags byte is kept, a full trace comes only from Syntropy's policy,
//!   and a returned `traceparent` has bit 0 set, echoes bit 1 and never carries the full-trace bit,
//!   and goes only to a request that carried one.
//! - [`flight`]: the daemon's bounded [`FlightRecorder`] of spans and log records.
//! - [`config`]: the `obs.flight_recorder` key and the `obs.trace.*` keys.
//!
//! [`span`]'s writer and [`sample`] run on hot threads: they allocate nothing, take no lock and use
//! no `tracing` (ADR-0012; the `hot-threads` lint step).

pub mod config;
pub mod context;
pub mod flight;
pub mod sample;
pub mod span;

pub use context::{IngressTrace, Traceparent};
pub use flight::FlightRecorder;
pub use sample::{FIRST_DECODE_STEPS, StepKind, StepSampler, fnv1a_64, step_hash};
pub use span::{SpanName, SpanNameId, SpanRecord, SpanWriter, span_batches};

#[cfg(test)]
mod tests;
