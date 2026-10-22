// SPDX-License-Identifier: MIT OR Apache-2.0

//! Log entries and the hash chain (RFC-0032 §6.1).

use std::fmt;

use serde::{Deserialize, Serialize};
use syntropy_core::{Blake3, NodeId};
use syntropy_net::ctl::envelope::{self, DecodeError, EncodeError};

use super::event::ClusterEvent;

/// The command that produced an entry (RFC-0032 §6.3): the node it came from and that node's
/// counter. Relayed commands keep their origin. DIST-E9-T02 deduplicates by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CommandId {
    /// The node that issued the command.
    pub origin: NodeId,
    /// The node's counter, persisted with its boot counter.
    pub counter: u64,
}

/// The chain value before the first entry: BLAKE3 of the cluster key's public half, which
/// RFC-0034 §4 calls the `ClusterId` (RFC-0032 §6.1, "index 0").
///
/// DIST-E2-T06 defines `ClusterId`; until it does, the 32 bytes are passed as they are. The
/// chain hashes the same 32 bytes either way, so its vectors do not change (DIST-E9-T16).
pub fn genesis(cluster: &[u8; 32]) -> Blake3 {
    Blake3(*blake3::hash(cluster).as_bytes())
}

/// The chain value of an entry (RFC-0032 §6.1):
///
/// ```text
/// chain(i) = BLAKE3(chain(i-1) ‖ index ‖ term ‖ cause ‖ event_bytes)
/// ```
///
/// `chain(i-1)` is the previous entry's 32-byte value, or [`genesis`] for the first entry.
/// `index` and `term` are 8-byte little-endian integers, as everywhere on the wire
/// (RFC-0031). `cause` is one byte `0` when there is none, else `1` followed by the origin's
/// 32 bytes and the counter as 8 little-endian bytes. `event_bytes` are the event's CBOR exactly
/// as the coordinator encoded it, and come last, so the input needs no length for them.
pub fn chain(
    prev: &Blake3,
    index: u64,
    term: u64,
    cause: Option<&CommandId>,
    event_bytes: &[u8],
) -> Blake3 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&prev.0);
    hasher.update(&index.to_le_bytes());
    hasher.update(&term.to_le_bytes());
    match cause {
        None => {
            hasher.update(&[0]);
        }
        Some(cause) => {
            hasher.update(&[1]);
            hasher.update(&cause.origin.0);
            hasher.update(&cause.counter.to_le_bytes());
        }
    }
    hasher.update(event_bytes);
    Blake3(*hasher.finalize().as_bytes())
}

/// An entry's fields other than the event: what a log record stores beside the verbatim
/// `event_bytes` (RFC-0032 §6.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryHeader {
    /// Dense, increasing from 1.
    pub index: u64,
    /// The coordinator term that appended it (RFC-0032 §7.2).
    pub term: u64,
    /// The coordinator's wall clock at append. Informational: `apply` never reads it.
    pub at_ms: u64,
    /// The command that produced it, if any.
    pub cause: Option<CommandId>,
    /// The entry's chain value.
    pub chain: Blake3,
}

/// One entry of the cluster event log (RFC-0001 §5.10, RFC-0032 §6.1).
///
/// The coordinator builds an entry with [`LogEntry::append`], which encodes the event once. A
/// follower builds one with [`LogEntry::receive`] from the header and the bytes it was sent,
/// which it keeps as received: a decoder of an older minor drops fields it does not know, so
/// re-encoding would change the bytes, and the chain with them.
#[derive(Clone, Debug, PartialEq)]
pub struct LogEntry {
    /// Dense, increasing from 1.
    pub index: u64,
    /// The coordinator term that appended it.
    pub term: u64,
    /// The coordinator's wall clock at append. Informational: `apply` never reads it.
    pub at_ms: u64,
    /// The command that produced it; drives re-submission (RFC-0032 §6.10).
    pub cause: Option<CommandId>,
    /// The decoded view of `event_bytes`.
    pub event: ClusterEvent,
    event_bytes: Vec<u8>,
    chain: Blake3,
}

impl LogEntry {
    /// The entry the coordinator appends after the entry whose chain value is `prev`: the event
    /// is encoded once with the envelope's deterministic CBOR, and those bytes are hashed.
    pub fn append(
        prev: &Blake3,
        index: u64,
        term: u64,
        at_ms: u64,
        cause: Option<CommandId>,
        event: ClusterEvent,
    ) -> Result<Self, EncodeError> {
        let event_bytes = envelope::to_cbor(&event)?;
        let chain = chain(prev, index, term, cause.as_ref(), &event_bytes);
        Ok(LogEntry {
            index,
            term,
            at_ms,
            cause,
            event,
            event_bytes,
            chain,
        })
    }

    /// The entry a follower received after the entry whose chain value is `prev`: `event_bytes`
    /// are kept as received and decoded for the view, and the chain is recomputed over them and
    /// checked against the header's. Total over its input: any bytes give an entry or an error.
    pub fn receive(
        prev: &Blake3,
        header: &EntryHeader,
        event_bytes: Vec<u8>,
    ) -> Result<Self, ReceiveError> {
        let computed = chain(
            prev,
            header.index,
            header.term,
            header.cause.as_ref(),
            &event_bytes,
        );
        if computed != header.chain {
            return Err(ReceiveError::ChainMismatch {
                index: header.index,
                expected: header.chain,
                computed,
            });
        }
        let event = envelope::from_cbor(&event_bytes).map_err(ReceiveError::Event)?;
        Ok(LogEntry {
            index: header.index,
            term: header.term,
            at_ms: header.at_ms,
            cause: header.cause,
            event,
            event_bytes,
            chain: computed,
        })
    }

    /// The event's CBOR exactly as the coordinator encoded it. Persisted, relayed and hashed
    /// verbatim; never re-encoded.
    pub fn event_bytes(&self) -> &[u8] {
        &self.event_bytes
    }

    /// The entry's chain value: two nodes agree on the log up to this entry exactly when their
    /// values here match.
    pub fn chain(&self) -> Blake3 {
        self.chain
    }

    /// The entry's fields other than the event.
    pub fn header(&self) -> EntryHeader {
        EntryHeader {
            index: self.index,
            term: self.term,
            at_ms: self.at_ms,
            cause: self.cause,
            chain: self.chain,
        }
    }
}

/// Why a follower could not take an entry it received.
#[derive(Debug, PartialEq)]
pub enum ReceiveError {
    /// The chain recomputed over the received bytes is not the header's: the two nodes'
    /// logs diverge at or before this entry.
    ChainMismatch {
        /// The entry's index.
        index: u64,
        /// The header's chain value.
        expected: Blake3,
        /// The value computed here.
        computed: Blake3,
    },
    /// The event bytes are not an event this build can read.
    Event(DecodeError),
}

impl fmt::Display for ReceiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReceiveError::ChainMismatch { index, .. } => write!(
                f,
                "entry {index}: the chain over the received bytes does not match the header's; \
                 the logs diverge at or before it"
            ),
            ReceiveError::Event(e) => write!(f, "the entry's event does not decode: {e}"),
        }
    }
}

impl std::error::Error for ReceiveError {}
