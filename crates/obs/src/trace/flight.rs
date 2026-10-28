// SPDX-License-Identifier: MIT OR Apache-2.0

//! The daemon's flight recorder (RFC-0053 §3.6, §4): the node's recent spans and log records,
//! bounded by `obs.flight_recorder` (64 MiB by default), with no content. The oldest entries go
//! first when it is full. Traces are found by trace ID, and the dashboard's trace view and
//! `syntropy logs` read from it.
//!
//! Each entry is counted at its record's encoded size, 64 bytes for a span and 48 for a log
//! record, so the bound is the records' bytes, not the heap. In memory each entry is one 72-byte
//! [`Entry`], 1.125× a span's count and 1.5× a log record's, and the deque may hold spare
//! capacity as it grows. So the heap is about 1.1–1.5× `obs.flight_recorder` for the entries
//! alone, by their mix [estimate, from the type sizes], before the deque's spare room
//! (RFC-0053 §3.4, note of 2026-10-28).

use std::collections::VecDeque;

use crate::log::LogRecord;
use crate::trace::span::SpanRecord;

/// One entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Entry {
    /// A span.
    Span(SpanRecord),
    /// A log record.
    Log(LogRecord),
}

impl Entry {
    /// The bytes it counts for.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        match self {
            Entry::Span(_) => SpanRecord::BYTES as u64,
            Entry::Log(_) => LogRecord::BYTES as u64,
        }
    }
}

/// A bounded store of the most recent spans and log records.
#[derive(Debug, Default)]
pub struct FlightRecorder {
    cap: u64,
    held: u64,
    entries: VecDeque<Entry>,
    evicted: u64,
}

impl FlightRecorder {
    /// An empty recorder of at most `cap` bytes of records.
    #[must_use]
    pub fn new(cap: u64) -> FlightRecorder {
        FlightRecorder {
            cap,
            ..FlightRecorder::default()
        }
    }

    /// Adds `entry`, evicting the oldest entries until it fits. An entry larger than the whole
    /// recorder is not kept, and counts as evicted.
    pub fn push(&mut self, entry: Entry) {
        let bytes = entry.bytes();
        if bytes > self.cap {
            self.evicted += 1;
            return;
        }
        while self.held + bytes > self.cap {
            let Some(old) = self.entries.pop_front() else {
                break;
            };
            self.held -= old.bytes();
            self.evicted += 1;
        }
        self.held += bytes;
        self.entries.push_back(entry);
    }

    /// The spans of trace `trace_id` it still holds, oldest first.
    pub fn trace(&self, trace_id: &[u8; 16]) -> impl Iterator<Item = &SpanRecord> {
        self.entries.iter().filter_map(move |e| match e {
            Entry::Span(s) if &s.trace_id == trace_id => Some(s),
            _ => None,
        })
    }

    /// Its log records, oldest first.
    pub fn logs(&self) -> impl Iterator<Item = &LogRecord> {
        self.entries.iter().filter_map(|e| match e {
            Entry::Log(l) => Some(l),
            Entry::Span(_) => None,
        })
    }

    /// The bytes it holds.
    #[must_use]
    pub fn held(&self) -> u64 {
        self.held
    }

    /// The entries it holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether it holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries evicted, or refused as too large, so far.
    #[must_use]
    pub fn evicted(&self) -> u64 {
        self.evicted
    }
}
