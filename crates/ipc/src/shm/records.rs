// SPDX-License-Identifier: MIT OR Apache-2.0

//! The records of the `local_sequences` and `local_tokens` rings (RFC-0001 §3.5).
//!
//! One slot holds one record: the postcard encoding of one [`SequenceControl`] on
//! `local_sequences`, daemon to head runner, or of one [`TokenFrame`] on `local_tokens`, tail
//! runner to daemon. The types are `core`'s (RFC-0001 §5.8), so ingress, the runner and
//! `syntropy-testkit`'s fake runner (QUAL-E3-T18) share one encoding. At M0 the rings carry
//! `core` types, as RFC-0001 §5.8's note of 2026-10-23 has parity taps do; RFC-0031 §6's frame
//! bytes for sequence control arrive in M2 (DIST-E8-T05), and DIST-E11-T03 moves the rings to
//! them (RFC-0031 §1, note of 2026-10-27).
//!
//! The decoders are total: any input gives a record or a [`RecordError`], never a panic. A record
//! comes from the other process, which each side treats as possibly hostile (see [`super`]), so a
//! declared sequence length reserves no memory before its elements decode, as on the command
//! channel.
//!
//! Postcard writes a variant as its index and a struct's fields in order, with no names, so a
//! change to either type changes the encoding and bumps [`PROTOCOL`](crate::cmd::PROTOCOL).
//! Defining this encoding did not bump it: no build has shipped, and RFC-0001 §3.5 already
//! declared the rings' records as these types (RFC-0001 §18.39).

use std::fmt;

use postcard::de_flavors::Slice;
use serde::Serialize;
use serde::de::DeserializeOwned;
use syntropy_core::{SequenceControl, TokenFrame};

use crate::cmd::NoPrealloc;

/// Why a ring record did not encode or decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordError {
    /// A field's serializer failed, or the bytes are not a record of the expected type:
    /// postcard's reason.
    Malformed(postcard::Error),
    /// The record ends before the bytes do.
    Trailing {
        /// The bytes after the record.
        unread: usize,
    },
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecordError::Malformed(error) => write!(f, "the ring record is malformed: {error}"),
            RecordError::Trailing { unread } => {
                write!(f, "the ring record has {unread} bytes after it")
            }
        }
    }
}

impl std::error::Error for RecordError {}

/// Encodes `control` as one `local_sequences` record into `out`, replacing its contents. `out`
/// keeps its capacity, so a reused buffer stops allocating once it has grown to the largest
/// record.
///
/// # Errors
///
/// [`RecordError::Malformed`] if a field does not serialize.
pub fn encode_control(control: &SequenceControl, out: &mut Vec<u8>) -> Result<(), RecordError> {
    encode(control, out)
}

/// Decodes one `local_sequences` record.
///
/// # Errors
///
/// [`RecordError::Malformed`] for bytes that are not a `SequenceControl`, and
/// [`RecordError::Trailing`] for bytes left after one.
pub fn decode_control(record: &[u8]) -> Result<SequenceControl, RecordError> {
    decode(record)
}

/// Encodes `frame` as one `local_tokens` record into `out`, as [`encode_control`] does.
///
/// # Errors
///
/// [`RecordError::Malformed`] if a field does not serialize.
pub fn encode_tokens(frame: &TokenFrame, out: &mut Vec<u8>) -> Result<(), RecordError> {
    encode(frame, out)
}

/// Decodes one `local_tokens` record.
///
/// # Errors
///
/// As for [`decode_control`], for a `TokenFrame`.
pub fn decode_tokens(record: &[u8]) -> Result<TokenFrame, RecordError> {
    decode(record)
}

fn encode<T: Serialize>(value: &T, out: &mut Vec<u8>) -> Result<(), RecordError> {
    out.clear();
    let buffer = std::mem::take(out);
    *out = postcard::to_extend(value, buffer).map_err(RecordError::Malformed)?;
    Ok(())
}

fn decode<T: DeserializeOwned>(record: &[u8]) -> Result<T, RecordError> {
    let mut deserializer = postcard::Deserializer::from_flavor(NoPrealloc(Slice::new(record)));
    let value = T::deserialize(&mut deserializer).map_err(RecordError::Malformed)?;
    let rest = deserializer.finalize().map_err(RecordError::Malformed)?;
    if !rest.is_empty() {
        return Err(RecordError::Trailing { unread: rest.len() });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use smallvec::smallvec;
    use syntropy_core::{
        CancelReason, FinishReason, InstanceId, NodeId, PlanEpoch, SamplingParams, SequenceId,
        SequenceLimits, SequenceOpen, TokenEntry,
    };

    use super::*;

    fn seq(n: u32) -> SequenceId {
        SequenceId::new(&NodeId([3; 32]), 1, n)
    }

    fn open(n: u32) -> SequenceControl {
        SequenceControl::Open(SequenceOpen {
            seq: seq(n),
            session: None,
            ingress: NodeId([3; 32]),
            prompt: vec![1, 2, 3].into(),
            prefix: None,
            sampling: SamplingParams::default(),
            limits: SequenceLimits {
                max_tokens: 16,
                ..SequenceLimits::default()
            },
            exactness: Default::default(),
            priority: Default::default(),
            choice_index: None,
            prefill_chunk: 0,
            initial_window: None,
            resume: None,
            prefix_scope: Default::default(),
            session_pk: None,
            trace: None,
            parity: None,
        })
    }

    fn frame() -> TokenFrame {
        TokenFrame {
            instance: InstanceId(9),
            epoch: PlanEpoch { term: 1, seq: 2 },
            step: 7,
            entries: smallvec![TokenEntry {
                seq: seq(1),
                tokens: smallvec![10, 11],
                first_position: 3,
                logprobs: None,
                finish: Some(FinishReason::Stop),
                approx: Default::default(),
                draft: smallvec![],
                draft_tree: None,
                rng_counter: 2,
                tap: None,
            }],
        }
    }

    /// Every control variant and a token frame come back as they went in, through one reused
    /// buffer.
    #[test]
    fn records_round_trip() {
        let mut out = Vec::new();
        for control in [
            open(1),
            SequenceControl::Cancel {
                seq: seq(1),
                reason: CancelReason::ClientGone,
            },
            SequenceControl::Pause { seq: seq(1) },
            SequenceControl::Resume { seq: seq(1) },
            SequenceControl::Release {
                seq: seq(1),
                persist: None,
            },
        ] {
            encode_control(&control, &mut out).unwrap();
            assert_eq!(decode_control(&out).unwrap(), control);
        }
        encode_tokens(&frame(), &mut out).unwrap();
        assert_eq!(decode_tokens(&out).unwrap(), frame());
    }

    /// A record with bytes after it, a truncated one and a forged sequence length are refused,
    /// not decoded and not panicked on.
    #[test]
    fn bad_records_are_refused() {
        let mut out = Vec::new();
        encode_tokens(&frame(), &mut out).unwrap();
        let mut long = out.clone();
        long.push(0);
        assert_eq!(
            decode_tokens(&long),
            Err(RecordError::Trailing { unread: 1 })
        );
        assert!(matches!(
            decode_tokens(&out[..out.len() - 1]),
            Err(RecordError::Malformed(_))
        ));
        // A frame whose entry count claims 2^32 − 1 entries, followed by nothing.
        let forged = [9, 0, 2, 7, 0xff, 0xff, 0xff, 0xff, 0x0f];
        assert!(matches!(
            decode_tokens(&forged),
            Err(RecordError::Malformed(_))
        ));
        assert!(matches!(
            decode_control(&[0xff]),
            Err(RecordError::Malformed(_))
        ));
    }

    /// Decodes `input` as both record types. Each returns a record or an error; reaching the end
    /// is the check, since a panic fails the case.
    fn check_total(input: &[u8]) {
        let _ = decode_control(input);
        let _ = decode_tokens(input);
    }

    /// A valid record of each type, encoded.
    fn valid_records() -> [Vec<u8>; 2] {
        let mut control = Vec::new();
        encode_control(&open(1), &mut control).unwrap();
        let mut tokens = Vec::new();
        encode_tokens(&frame(), &mut tokens).unwrap();
        [control, tokens]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4096))]

        /// Decoding is total over arbitrary bytes.
        #[test]
        fn decoding_is_total_over_arbitrary_bytes(
            input in prop::collection::vec(any::<u8>(), 0..1024),
        ) {
            check_total(&input);
        }

        /// Decoding is total over valid records with bytes overwritten and the end cut off, which
        /// reach far deeper into the decoders than random bytes do.
        #[test]
        fn decoding_is_total_over_damaged_records(
            which in 0usize..2,
            edits in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..8),
            keep in any::<prop::sample::Index>(),
        ) {
            let mut record = valid_records()[which].clone();
            for (at, byte) in &edits {
                let at = at.index(record.len());
                record[at] = *byte;
            }
            let keep = keep.index(record.len() + 1);
            check_total(&record[..keep]);
            check_total(&record);
        }
    }
}
