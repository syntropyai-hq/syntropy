// SPDX-License-Identifier: MIT OR Apache-2.0

//! Writers of the runner's shared-memory rings (RFC-0053 §1). A hot-thread module: the lint of
//! `cargo xtask ci lint`'s `hot-threads` step rejects `tracing` here.
//!
//! **One producer per ring.** The `telemetry` ring's only producer is the drain, and each
//! dispatcher writes its `StepRecord`s into its own step ring. A [`RingWriter`] claims its ring in
//! the region's [`RingClaims`] when it is made and releases it when dropped. A debug build
//! asserts that no ring is claimed twice, so a second writer fails at once rather than racing
//! the first.
//!
//! **Lossy.** A ring that is full drops the record and counts it (RFC-0001 §8.4 rule 8): a write
//! never waits, so a full ring never slows a step. An SPSC producer cannot take back a record the
//! consumer has not read, so the record dropped is the new one, not the oldest.
//!
//! `obs` is layer 0 and cannot name `ipc`'s rings. The runner implements [`RecordSink`] for the
//! producing ends it takes from its region.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, PoisonError};

/// A ring of the runner's region, as its writers claim it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RingName {
    /// The `telemetry` ring, written by the drain.
    Telemetry,
    /// A dispatcher's step ring, written by that dispatcher.
    Step {
        /// The dispatcher.
        dispatcher: u16,
    },
    /// The `log` ring.
    Log,
}

/// Why a sink took no record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SinkError {
    /// The ring is full: the record is dropped and counted.
    Full,
    /// The record is larger than a slot.
    TooLong,
    /// The reader is gone.
    Closed,
}

/// The producing end of a shared-memory ring, as a writer sees it. The runner implements it for
/// `ipc`'s ring ends.
pub trait RecordSink {
    /// Copies `record` into the next slot, or says why not. Never blocks.
    ///
    /// # Errors
    ///
    /// [`SinkError`].
    fn push(&mut self, record: &[u8]) -> Result<(), SinkError>;

    /// The largest record a slot holds, in bytes.
    fn max_record(&self) -> usize;
}

/// The rings of one region and whether each has a writer.
#[derive(Debug, Default)]
pub struct RingClaims(Mutex<BTreeSet<RingName>>);

impl RingClaims {
    /// No ring claimed yet.
    #[must_use]
    pub fn new() -> Arc<RingClaims> {
        Arc::new(RingClaims::default())
    }

    /// Whether `ring` has a writer.
    #[must_use]
    pub fn is_claimed(&self, ring: RingName) -> bool {
        self.set().contains(&ring)
    }

    fn set(&self) -> std::sync::MutexGuard<'_, BTreeSet<RingName>> {
        // Nothing that holds the lock panics, so a poisoned set is still consistent.
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The one writer of one ring.
#[derive(Debug)]
pub struct RingWriter<S: RecordSink> {
    claims: Arc<RingClaims>,
    ring: RingName,
    sink: S,
    written: u64,
    dropped: u64,
}

impl<S: RecordSink> RingWriter<S> {
    /// The writer of `ring` in the region `claims` covers, writing into `sink`.
    ///
    /// # Panics
    ///
    /// In a debug build, when `ring` already has a writer.
    pub fn claim(claims: &Arc<RingClaims>, ring: RingName, sink: S) -> RingWriter<S> {
        let fresh = claims.set().insert(ring);
        debug_assert!(fresh, "a second writer for the {ring:?} ring");
        RingWriter {
            claims: Arc::clone(claims),
            ring,
            sink,
            written: 0,
            dropped: 0,
        }
    }

    /// Writes `record`; whether it went in. A full ring, a record larger than a slot or a reader
    /// that is gone drops it and counts it. Never blocks and never allocates.
    #[inline]
    pub fn push(&mut self, record: &[u8]) -> bool {
        match self.sink.push(record) {
            Ok(()) => {
                self.written += 1;
                true
            }
            Err(_) => {
                self.dropped += 1;
                false
            }
        }
    }

    /// The ring.
    #[must_use]
    pub fn ring(&self) -> RingName {
        self.ring
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

    /// The largest record a slot holds.
    #[must_use]
    pub fn max_record(&self) -> usize {
        self.sink.max_record()
    }

    /// The sink, for tests.
    #[must_use]
    pub fn sink(&self) -> &S {
        &self.sink
    }

    /// The sink, for tests that drain it.
    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }
}

impl<S: RecordSink> Drop for RingWriter<S> {
    fn drop(&mut self) {
        self.claims.set().remove(&self.ring);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ring of `capacity` records that nobody reads.
    #[derive(Debug, Default)]
    struct Bounded {
        records: Vec<Vec<u8>>,
        capacity: usize,
    }

    impl RecordSink for Bounded {
        fn push(&mut self, record: &[u8]) -> Result<(), SinkError> {
            if self.records.len() == self.capacity {
                return Err(SinkError::Full);
            }
            self.records.push(record.to_vec());
            Ok(())
        }

        fn max_record(&self) -> usize {
            64
        }
    }

    #[test]
    fn a_full_ring_drops_and_counts() {
        let claims = RingClaims::new();
        let mut writer = RingWriter::claim(
            &claims,
            RingName::Step { dispatcher: 0 },
            Bounded {
                capacity: 2,
                ..Bounded::default()
            },
        );
        assert!(writer.push(b"a") && writer.push(b"b"));
        assert!(!writer.push(b"c"));
        assert!(!writer.push(b"d"));
        assert_eq!((writer.written(), writer.dropped()), (2, 2));
        assert_eq!(writer.sink().records, [b"a".to_vec(), b"b".to_vec()]);
    }

    /// Each dispatcher's step ring and the telemetry ring are separate rings with one writer
    /// each; a dropped writer frees its ring.
    #[test]
    fn each_ring_has_one_writer() {
        let claims = RingClaims::new();
        let telemetry = RingWriter::claim(&claims, RingName::Telemetry, Bounded::default());
        let step0 = RingWriter::claim(
            &claims,
            RingName::Step { dispatcher: 0 },
            Bounded::default(),
        );
        let step1 = RingWriter::claim(
            &claims,
            RingName::Step { dispatcher: 1 },
            Bounded::default(),
        );
        assert!(claims.is_claimed(RingName::Telemetry));
        drop(step1);
        assert!(!claims.is_claimed(RingName::Step { dispatcher: 1 }));
        drop((telemetry, step0));
        assert!(!claims.is_claimed(RingName::Telemetry));
    }

    /// The criterion "a debug assertion fails on a second writer".
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a second writer for the Step { dispatcher: 3 } ring")]
    fn a_second_writer_fails_the_debug_assertion() {
        let claims = RingClaims::new();
        let _first = RingWriter::claim(
            &claims,
            RingName::Step { dispatcher: 3 },
            Bounded::default(),
        );
        let _second = RingWriter::claim(
            &claims,
            RingName::Step { dispatcher: 3 },
            Bounded::default(),
        );
    }
}
