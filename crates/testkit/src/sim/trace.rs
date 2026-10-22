// SPDX-License-Identifier: MIT OR Apache-2.0

//! The trace hash: BLAKE3 over every event a run processes, in order (RFC-0060 §8.2, §8.8).
//!
//! Two runs of one scenario with one seed process the same events at the same instants in the
//! same order, so their trace hashes are equal, in one process or in two. A difference means
//! non-determinism leaked into the simulated code: a `RandomState` hash map, a wall-clock read,
//! an unseeded RNG, or iteration that depends on threads (§8.8).

use std::fmt;

use syntropy_core::{Blake3, MonoTime};

/// The BLAKE3 key-derivation context of the trace hash. A change to the framing below changes
/// the version, so hashes from two framings never compare equal by accident.
const CONTEXT: &str = "syntropy 2026-10-22 sim trace v1";

/// What the trace hash records of an event: bytes that identify it, the same on every host.
///
/// Write every field that tells two events apart, in a fixed order and a fixed width, such as
/// little-endian integers. Nothing that differs from process to process may go in: no addresses,
/// and no iteration over a `RandomState` map.
pub trait TraceEvent {
    /// Appends the event's bytes to `out`.
    fn trace_bytes(&self, out: &mut Vec<u8>);
}

/// BLAKE3 over the events of a run, in the order they are processed.
///
/// Each event contributes its instant as a little-endian `u64` of nanoseconds, the length of its
/// bytes as a little-endian `u64`, then the bytes [`TraceEvent::trace_bytes`] wrote. The length
/// keeps one event's bytes from running into the next.
pub struct TraceHasher {
    hasher: blake3::Hasher,
    events: u64,
    scratch: Vec<u8>,
}

impl TraceHasher {
    /// The hash of no events.
    pub fn new() -> TraceHasher {
        TraceHasher {
            hasher: blake3::Hasher::new_derive_key(CONTEXT),
            events: 0,
            scratch: Vec::new(),
        }
    }

    /// Adds `event`, processed at `at`.
    pub fn record(&mut self, at: MonoTime, event: &impl TraceEvent) {
        self.scratch.clear();
        event.trace_bytes(&mut self.scratch);
        self.hasher.update(&at.0.to_le_bytes());
        self.hasher
            .update(&(self.scratch.len() as u64).to_le_bytes());
        self.hasher.update(&self.scratch);
        self.events += 1;
    }

    /// Events recorded so far.
    pub fn events(&self) -> u64 {
        self.events
    }

    /// The hash of the events recorded so far. Recording more afterwards is allowed.
    pub fn digest(&self) -> Blake3 {
        Blake3(*self.hasher.finalize().as_bytes())
    }
}

impl Default for TraceHasher {
    fn default() -> TraceHasher {
        TraceHasher::new()
    }
}

impl fmt::Debug for TraceHasher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TraceHasher")
            .field("events", &self.events)
            .field("digest", &Hex(&self.digest()))
            .finish()
    }
}

/// A digest in lowercase hex, as violations and test failures print it.
pub struct Hex<'a>(pub &'a Blake3);

impl fmt::Display for Hex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

impl fmt::Debug for Hex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Bytes(&'static [u8]);

    impl TraceEvent for Bytes {
        fn trace_bytes(&self, out: &mut Vec<u8>) {
            out.extend_from_slice(self.0);
        }
    }

    fn hash(events: &[(u64, &'static [u8])]) -> Blake3 {
        let mut hasher = TraceHasher::new();
        for &(at, bytes) in events {
            hasher.record(MonoTime(at), &Bytes(bytes));
        }
        hasher.digest()
    }

    #[test]
    fn the_hash_covers_order_instants_and_boundaries() {
        let base = hash(&[(1, b"ab"), (2, b"c")]);
        assert_eq!(base, hash(&[(1, b"ab"), (2, b"c")]), "deterministic");
        assert_ne!(base, hash(&[(2, b"c"), (1, b"ab")]), "order");
        assert_ne!(base, hash(&[(1, b"ab"), (3, b"c")]), "an instant");
        assert_ne!(base, hash(&[(1, b"a"), (2, b"bc")]), "where one event ends");
        assert_ne!(base, hash(&[(1, b"ab")]), "a missing event");
        assert_ne!(hash(&[]), Blake3([0; 32]));
    }

    #[test]
    fn the_framing_is_pinned() {
        // The framing and the context, by hand: a change to either changes this value, and
        // should come with a new context version.
        let mut by_hand = blake3::Hasher::new_derive_key(CONTEXT);
        by_hand.update(&7u64.to_le_bytes());
        by_hand.update(&3u64.to_le_bytes());
        by_hand.update(b"xyz");
        assert_eq!(hash(&[(7, b"xyz")]), Blake3(*by_hand.finalize().as_bytes()));
    }

    #[test]
    fn a_digest_prints_as_hex() {
        let mut digest = [0u8; 32];
        digest[0] = 0xab;
        digest[31] = 0x01;
        let text = Hex(&Blake3(digest)).to_string();
        assert_eq!(text.len(), 64);
        assert!(text.starts_with("ab00") && text.ends_with("0001"), "{text}");
        let mut hasher = TraceHasher::new();
        hasher.record(MonoTime(0), &Bytes(b""));
        assert_eq!(hasher.events(), 1);
        assert!(format!("{hasher:?}").contains("events: 1"));
    }
}
