// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hot-thread log records (RFC-0053 §4.1). A hot-thread module: the lint of `cargo xtask ci
//! lint`'s `hot-threads` step rejects `tracing` here.
//!
//! Hot threads never format. A log line is a [`Callsite`], declared statically with its format
//! string, field names, file and line, and a [`LogRecord`] that names the callsite and carries up
//! to four `u64` arguments. A [`LogWriter`] pushes the record, 48 bytes, into the runner's `log`
//! ring, and the daemon formats it on receipt from the build's callsite table.
//!
//! Async and other non-hot code logs through the `tracing` crate's events instead, which the
//! daemon receives directly.
//!
//! **Content.** A callsite marks which of its arguments are content: token IDs, or anything
//! derived from a prompt or an output. The daemon prints those only with
//! `logging.include_content = true`, and only at `debug` or below (RFC-0053 §4.3).

use serde::{Deserialize, Serialize};
use syntropy_core::MonoTime;

use crate::ring::{RecordSink, RingClaims, RingName, RingWriter};

/// How severe a record is, most severe first (RFC-0054's `logging.level`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Level {
    /// Something failed.
    Error = 1,
    /// Something is wrong but the node goes on.
    Warn = 2,
    /// Lifecycle, plans, membership, degradations: the default.
    Info = 3,
    /// Per-request summaries.
    Debug = 4,
    /// Per-step detail, for development.
    Trace = 5,
}

impl Level {
    /// The level's name, lowercase.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Level::Error => "error",
            Level::Warn => "warn",
            Level::Info => "info",
            Level::Debug => "debug",
            Level::Trace => "trace",
        }
    }

    /// The level a byte names, if it names one.
    #[must_use]
    pub const fn from_u8(byte: u8) -> Option<Level> {
        match byte {
            1 => Some(Level::Error),
            2 => Some(Level::Warn),
            3 => Some(Level::Info),
            4 => Some(Level::Debug),
            5 => Some(Level::Trace),
            _ => None,
        }
    }
}

/// A callsite's number in its build's callsite table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CallsiteId(pub u32);

/// One hot-thread log statement, declared statically (RFC-0053 §4.1). The table of a build's
/// callsites is static too, so a record carries only the callsite's number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Callsite {
    /// Its number.
    pub id: CallsiteId,
    /// Its level.
    pub level: Level,
    /// The crate and module it is in, such as `syntropy_exec::step`.
    pub target: &'static str,
    /// The message, with `{name}` for each argument it shows, such as
    /// `"hop {hop} stalled for {micros} µs"`.
    pub format: &'static str,
    /// The arguments' names, in the order of [`LogRecord::args`]; at most four.
    pub fields: &'static [&'static str],
    /// Bit `i` set: argument `i` is content, printed only with `logging.include_content`.
    pub content: u8,
    /// How many leading arguments, with the callsite and level, identify a record for repeat
    /// suppression (RFC-0053 §4.1): two records that agree on these collapse.
    pub key_fields: u8,
    /// The source file.
    pub file: &'static str,
    /// The source line.
    pub line: u32,
}

impl Callsite {
    /// Whether argument `index` is content.
    #[must_use]
    pub const fn is_content(&self, index: usize) -> bool {
        index < 8 && self.content & (1 << index) != 0
    }
}

/// One hot-thread log record, as the `log` ring carries it (RFC-0053 §4.1). Owner: RFC-0053.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LogRecord {
    /// The callsite.
    pub callsite: CallsiteId,
    /// The level, the callsite's unless the writer raised it.
    pub level: Level,
    /// When, on the node's `Clock` timeline, from the thread's `TickSource::now()`.
    pub at: MonoTime,
    /// The arguments, in the callsite's field order; unused ones are 0.
    pub args: [u64; 4],
}

impl LogRecord {
    /// The bytes of one encoded record.
    pub const BYTES: usize = 48;

    /// The record's encoding: little-endian `callsite` (4 bytes), `level` (1), three zero bytes,
    /// `at` (8), then the four arguments (8 each). Part of `ipc_protocol` (RFC-0053 §4.1).
    #[must_use]
    pub fn encode(&self) -> [u8; LogRecord::BYTES] {
        let mut out = [0u8; LogRecord::BYTES];
        out[0..4].copy_from_slice(&self.callsite.0.to_le_bytes());
        out[4] = self.level as u8;
        out[8..16].copy_from_slice(&self.at.0.to_le_bytes());
        for (at, arg) in self.args.iter().enumerate() {
            out[16 + at * 8..24 + at * 8].copy_from_slice(&arg.to_le_bytes());
        }
        out
    }

    /// Decodes one record, or `None` for bytes that are not one: the wrong length, an unknown
    /// level or nonzero padding.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<LogRecord> {
        let bytes: &[u8; LogRecord::BYTES] = bytes.try_into().ok()?;
        if bytes[5..8] != [0, 0, 0] {
            return None;
        }
        let word = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or([0; 8]));
        Some(LogRecord {
            callsite: CallsiteId(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
            level: Level::from_u8(bytes[4])?,
            at: MonoTime(word(8)),
            args: [word(16), word(24), word(32), word(40)],
        })
    }
}

/// A hot thread's writer of the runner's `log` ring. The ring has one writer, so the runner's
/// hot threads share one `LogWriter` through their owner, as they share the ring.
#[derive(Debug)]
pub struct LogWriter<S: RecordSink> {
    ring: RingWriter<S>,
}

impl<S: RecordSink> LogWriter<S> {
    /// The writer of the `log` ring of the region `claims` covers.
    ///
    /// # Panics
    ///
    /// In a debug build, when the `log` ring already has a writer.
    pub fn claim(claims: &std::sync::Arc<RingClaims>, sink: S) -> LogWriter<S> {
        LogWriter {
            ring: RingWriter::claim(claims, RingName::Log, sink),
        }
    }

    /// Writes one record for `callsite` at `at`; whether it went in. A full ring drops and
    /// counts it. Never formats, allocates or blocks.
    #[inline]
    pub fn log(&mut self, callsite: &Callsite, at: MonoTime, args: [u64; 4]) -> bool {
        let record = LogRecord {
            callsite: callsite.id,
            level: callsite.level,
            at,
            args,
        };
        self.ring.push(&record.encode())
    }

    /// The ring's writer, for its counts.
    #[must_use]
    pub fn ring(&self) -> &RingWriter<S> {
        &self.ring
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::SinkError;

    const STALL: Callsite = Callsite {
        id: CallsiteId(7),
        level: Level::Warn,
        target: "syntropy_exec::step",
        format: "hop {hop} stalled for {micros} µs",
        fields: &["hop", "micros"],
        content: 0,
        key_fields: 1,
        file: "crates/exec/src/step.rs",
        line: 42,
    };

    #[derive(Default)]
    struct Collect(Vec<Vec<u8>>);

    impl RecordSink for Collect {
        fn push(&mut self, record: &[u8]) -> Result<(), SinkError> {
            self.0.push(record.to_vec());
            Ok(())
        }

        fn max_record(&self) -> usize {
            64
        }
    }

    #[test]
    fn records_round_trip_through_the_ring() {
        let claims = RingClaims::new();
        let mut writer = LogWriter::claim(&claims, Collect::default());
        assert!(writer.log(&STALL, MonoTime(99), [3, 1_500, 0, 0]));
        let bytes = &writer.ring().sink().0[0];
        assert_eq!(bytes.len(), LogRecord::BYTES);
        assert_eq!(
            LogRecord::decode(bytes),
            Some(LogRecord {
                callsite: STALL.id,
                level: Level::Warn,
                at: MonoTime(99),
                args: [3, 1_500, 0, 0],
            })
        );
    }

    #[test]
    fn bad_bytes_do_not_decode() {
        let good = LogRecord {
            callsite: CallsiteId(1),
            level: Level::Info,
            at: MonoTime(0),
            args: [0; 4],
        }
        .encode();
        assert!(LogRecord::decode(&good[..47]).is_none());
        let mut level = good;
        level[4] = 9;
        assert!(LogRecord::decode(&level).is_none());
        let mut padding = good;
        padding[6] = 1;
        assert!(LogRecord::decode(&padding).is_none());
    }

    #[test]
    fn levels_order_by_severity() {
        assert!(Level::Error < Level::Warn && Level::Debug < Level::Trace);
        for byte in 1..=5 {
            assert_eq!(Level::from_u8(byte).map(|level| level as u8), Some(byte));
        }
        assert!(!STALL.is_content(9) && !STALL.is_content(0));
    }
}
