// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `SeqControl` frame (RFC-0031 §6.6, DIST-E8-T05): sequence control from ingress to the
//! head, on the `Sequence` channel or, when the two share a node, the `local_sequences` ring,
//! which carries the same frame bytes (RFC-0031 §4).
//!
//! ```text
//! extension | kind u8 | body_enc u8 | reserved [6] | padding to 64
//! payload   the kind's body as CBOR, zstd-compressed when body_enc = 1
//! ```
//!
//! | `kind` | `SequenceControl` | Body |
//! |---|---|---|
//! | 1 Open | `Open`, `resume` unset | `SequenceOpen` |
//! | 2 Cancel | `Cancel` | `{ seq, reason }` |
//! | 3 Pause | `Pause` | `{ seq }` |
//! | 4 Resume | `Resume` | `{ seq }` |
//! | 5 Release | `Release` | `{ seq, persist }` |
//! | 6 ResumeOpen | `Open`, `resume` set | `SequenceOpen` |
//! | 7 Reconcile | `Reconcile` | `{ seq, emitted, rng_counter, recent }` |
//!
//! The types are `core`'s (`syntropy_core::sequence`, RFC-0032 §3.6); this codec does not
//! repeat them. A body is the variant's fields as a CBOR map, written by the control envelope's
//! deterministic encoder (RFC-0031 §9.1, [`crate::ctl::envelope`]), and so:
//! - **Unknown fields are ignored.** A newer sender's added fields reach an older decoder as map
//!   keys it skips, and a field the sender leaves out takes its default (ADR-0305 item 3).
//! - **The prompt is a byte string** of little-endian `u32` token IDs, 4 bytes a token, so a
//!   128K-token prompt is 512 KiB (`core`'s `TokenSlice` takes that form in every binary format).
//! - **`ResumeOpen` is an `Open` with `resume` set** (RFC-0031 §6.6, note of 2026-10-23): the
//!   head re-opens a migrated or recovered sequence from it (RFC-0033 §4.4). The decoder refuses
//!   an `Open` with `resume` set and a `ResumeOpen` without it.
//!
//! **Size.** A frame must fit the receiver's slot, its `ChannelSpec::slot_size` (header
//! included). The encoder writes plain CBOR when the frame fits, compresses the body with zstd
//! (level 3, as the envelope does) only when it does not, and refuses the message when neither
//! fits. The decoder refuses a frame over the slot. The ingress daemon sizes the head's sequence
//! slots for the instance's maximum context with [`sequence_slot_size`].
//!
//! **What a decoder holds** (RFC-0060 §10.2). A small compressed frame must not make the receiver
//! hold a large body, so each kind's body has a declared ceiling, before and after
//! decompression, which both ends check:
//! - every kind but `Open` and `ResumeOpen`: [`SMALL_BODY`], 1 KiB, since their bodies are small
//!   fixed maps;
//! - `Open` and `ResumeOpen`: [`open_body_limit`] of the receiver's slot, the slot plus
//!   [`INLINE_ALLOWANCE`] for a grammar sent inline.
//!
//! The encoder refuses a body over its ceiling before it compresses anything, so a sender never
//! builds a frame its receiver must refuse.
//!
//! [`encode_seq_meta`] and [`decode_seq_meta`] read and write the `SEQ_META` item that carries
//! `SequenceMeta` on a sequence's first `Activation` frame (RFC-0031 §6.3): the same CBOR, with
//! no compression and no frame of its own.

use std::fmt;

use serde::{Deserialize, Serialize};
use syntropy_core::{
    CancelReason, ErrorClass, InstanceId, PlanEpoch, SequenceControl, SequenceId, SequenceMeta,
    SequenceOpen, SessionId,
};

use super::control::code;
use super::error::FrameError;
use super::header::{self, ALIGN, Flags, Frame, FrameSpec, FrameType, HEADER_LEN};
use crate::ctl::envelope::{
    self, BodyLimit, DecodeError, EncodeError, Encoding, ZSTD_LEVEL, decode_encoded,
};

/// The extension's fields: `kind`, `body_enc` and six reserved bytes. The extension is padded
/// to 64 bytes.
pub const SEQ_CONTROL_FIXED_LEN: usize = 8;

/// The bytes of a head's sequence slot that the prompt does not take: the base header, the
/// extension and the open's other fields, the sampler's parameters, limits, trace and parity
/// spec among them \[target\]: a chosen budget. A grammar sent inline can be larger, and is
/// then compressed, or refused if it still does not fit.
pub const OPEN_ALLOWANCE: usize = 64 << 10;

/// The most tokens `Reconcile::recent` holds (RFC-0033 §4.2).
pub const MAX_RECENT: usize = 8;

/// The largest body of every kind but `Open` and `ResumeOpen`, before and after decompression
/// \[target\]. Their bodies are small fixed maps, a `Reconcile` with its 8 tokens under 100 bytes,
/// so this bounds what such a frame can make a decoder hold (RFC-0060 §10.2).
pub const SMALL_BODY: usize = 1 << 10;

/// What an `Open` body may hold beyond its receiver's slot once decompressed \[target\]: a
/// compiled grammar sent inline, the one field that can outgrow the slot's [`OPEN_ALLOWANCE`].
pub const INLINE_ALLOWANCE: usize = 1 << 20;

/// The largest `Open` or `ResumeOpen` body a receiver whose slots hold `slot_size` bytes takes,
/// before and after decompression: the slot plus [`INLINE_ALLOWANCE`], within the envelope's 64
/// MiB. The format declares this size, which is the decoder's bound under RFC-0060 §10.2.
pub fn open_body_limit(slot_size: usize) -> usize {
    slot_size
        .saturating_add(INLINE_ALLOWANCE)
        .min(envelope::MAX_BODY)
}

/// The slot size a head's `Sequence` channel needs for an instance whose context is
/// `max_context` tokens: the frame of an `Open` whose prompt fills the context, 4 bytes a token,
/// plus [`OPEN_ALLOWANCE`], rounded up to a 64-byte boundary as receive slots are. A resumed
/// open's prompt holds the emitted tokens too, which the context also bounds.
pub fn sequence_slot_size(max_context: u32) -> usize {
    // `as usize` widens: a u32 fits in usize on every target RFC-0001 supports.
    (HEADER_LEN + ALIGN + 4 * max_context as usize + OPEN_ALLOWANCE).next_multiple_of(ALIGN)
}

/// The extension's `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SeqControlKind {
    /// 1: a new sequence.
    Open = 1,
    /// 2.
    Cancel = 2,
    /// 3.
    Pause = 3,
    /// 4.
    Resume = 4,
    /// 5.
    Release = 5,
    /// 6: an open that continues a sequence after migration or recovery.
    ResumeOpen = 6,
    /// 7: what the client has seen of a sequence, during recovery.
    Reconcile = 7,
}

impl SeqControlKind {
    /// The kind `code` names, if it names one.
    pub fn from_code(code: u8) -> Option<SeqControlKind> {
        Some(match code {
            1 => SeqControlKind::Open,
            2 => SeqControlKind::Cancel,
            3 => SeqControlKind::Pause,
            4 => SeqControlKind::Resume,
            5 => SeqControlKind::Release,
            6 => SeqControlKind::ResumeOpen,
            7 => SeqControlKind::Reconcile,
            _ => return None,
        })
    }

    /// The kind's wire code.
    pub fn code(self) -> u8 {
        self as u8
    }

    /// The kind `control` travels as, or `None` for a variant a later build of `core` added.
    pub fn of(control: &SequenceControl) -> Option<SeqControlKind> {
        Some(match control {
            SequenceControl::Open(open) if open.resume.is_some() => SeqControlKind::ResumeOpen,
            SequenceControl::Open(_) => SeqControlKind::Open,
            SequenceControl::Cancel { .. } => SeqControlKind::Cancel,
            SequenceControl::Pause { .. } => SeqControlKind::Pause,
            SequenceControl::Resume { .. } => SeqControlKind::Resume,
            SequenceControl::Release { .. } => SeqControlKind::Release,
            SequenceControl::Reconcile { .. } => SeqControlKind::Reconcile,
            _ => return None,
        })
    }
}

/// The fields of a `SeqControl` frame's base header the sender chooses. `step` is 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SeqControlSpec {
    pub wire_minor: u8,
    pub flags: Flags,
    pub instance: InstanceId,
    pub epoch: PlanEpoch,
    /// The channel's sequence number, from 1.
    pub chan_seq: u64,
}

/// What a receiver accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqControlLimits {
    /// The receiver's slot size, its `ChannelSpec::slot_size`, header included: a larger frame
    /// is refused (RFC-0031 §6.1, §6.6).
    pub slot_size: usize,
    /// The largest `Open` or `ResumeOpen` body, before and after decompression. Every other
    /// kind's is at most [`SMALL_BODY`] whatever this says.
    pub body: BodyLimit,
}

impl SeqControlLimits {
    /// A slot of `slot_size` bytes, and [`open_body_limit`] of it for an open's body.
    pub fn slot(slot_size: usize) -> SeqControlLimits {
        SeqControlLimits {
            slot_size,
            body: BodyLimit::new(open_body_limit(slot_size))
                .expect("open_body_limit stays within the envelope's limit"),
        }
    }
}

/// The ceiling on `kind`'s body for a receiver with `slot_size`-byte slots.
fn body_limit(kind: SeqControlKind, slot_size: usize) -> usize {
    match kind {
        SeqControlKind::Open | SeqControlKind::ResumeOpen => open_body_limit(slot_size),
        _ => SMALL_BODY,
    }
}

/// Why a `SeqControl` frame could not be encoded or decoded.
#[non_exhaustive]
#[derive(Debug, PartialEq, Eq)]
pub enum SeqControlError {
    /// The frame failed the base header's checks, is sealed, or is of another type.
    Frame(FrameError),
    /// A received frame longer than the receiver's slot.
    OverSlot { len: usize, slot_size: usize },
    /// A message whose frame does not fit the receiver's slot even with its body compressed:
    /// `len` is the compressed frame's length.
    DoesNotFit { len: usize, slot_size: usize },
    /// A message whose body is over its kind's ceiling for the receiver's slot, `limit` bytes,
    /// before any compression: [`SMALL_BODY`], or [`open_body_limit`] for an open.
    BodyTooLarge { len: usize, limit: usize },
    /// A `kind` this build does not know.
    UnknownKind(u8),
    /// An `Open` with `resume` set, or a `ResumeOpen` without it.
    ResumeMismatch { kind: SeqControlKind },
    /// A `Reconcile` with more than [`MAX_RECENT`] recent tokens.
    TooManyRecent { len: usize },
    /// A `SequenceControl` variant this codec has no kind for: one a later `core` added.
    UnknownVariant,
    /// The body could not be read: its `body_enc`, its zstd, its CBOR or its size.
    Body(DecodeError),
    /// The body could not be written.
    Encode(EncodeError),
}

impl SeqControlError {
    /// The RFC-0001 §5.12 class. A refused frame is never `Internal` (RFC-0060 §10.2): a kind or
    /// encoding this build lacks is `Unsupported`, a frame that breaks the protocol is
    /// `InvalidRequest`, and damaged bytes are the base header's or the body's class. A message
    /// too large for the receiver's slot is `Capacity`, as the envelope's is.
    pub fn class(&self) -> ErrorClass {
        match self {
            SeqControlError::Frame(error) => error.class(),
            SeqControlError::OverSlot { .. }
            | SeqControlError::ResumeMismatch { .. }
            | SeqControlError::TooManyRecent { .. } => ErrorClass::InvalidRequest,
            SeqControlError::UnknownKind(_) => ErrorClass::Unsupported,
            SeqControlError::DoesNotFit { .. } | SeqControlError::BodyTooLarge { .. } => {
                ErrorClass::Capacity
            }
            SeqControlError::UnknownVariant => ErrorClass::Internal,
            SeqControlError::Body(error) => error.class(),
            SeqControlError::Encode(error) => error.class(),
        }
    }

    /// The RFC-0031 §12 code a receiver sends in `Close`, where one exists.
    pub fn code(&self) -> Option<u16> {
        match self {
            SeqControlError::Frame(error) => error.code(),
            SeqControlError::OverSlot { .. } => Some(code::FRAME_TOO_LARGE),
            _ => None,
        }
    }
}

impl fmt::Display for SeqControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeqControlError::Frame(error) => error.fmt(f),
            SeqControlError::OverSlot { len, slot_size } => write!(
                f,
                "a {len}-byte SeqControl frame is over the receiver's {slot_size}-byte slot"
            ),
            SeqControlError::DoesNotFit { len, slot_size } => write!(
                f,
                "the SeqControl frame takes {len} bytes compressed, over the receiver's \
                 {slot_size}-byte slot"
            ),
            SeqControlError::BodyTooLarge { len, limit } => write!(
                f,
                "a {len}-byte SeqControl body is over its kind's {limit}-byte ceiling"
            ),
            SeqControlError::UnknownKind(kind) => write!(f, "unknown SeqControl kind {kind}"),
            SeqControlError::ResumeMismatch { kind } => match kind {
                SeqControlKind::ResumeOpen => f.write_str("a ResumeOpen without `resume`"),
                _ => f.write_str("an Open with `resume` set, which only ResumeOpen carries"),
            },
            SeqControlError::TooManyRecent { len } => write!(
                f,
                "a Reconcile with {len} recent tokens, over the {MAX_RECENT} it may carry"
            ),
            SeqControlError::UnknownVariant => {
                f.write_str("a SequenceControl variant with no SeqControl kind")
            }
            SeqControlError::Body(error) => write!(f, "SeqControl body: {error}"),
            SeqControlError::Encode(error) => write!(f, "SeqControl body: {error}"),
        }
    }
}

impl std::error::Error for SeqControlError {}

impl From<FrameError> for SeqControlError {
    fn from(error: FrameError) -> SeqControlError {
        SeqControlError::Frame(error)
    }
}

/// The body of `Pause` and `Resume`.
#[derive(Serialize, Deserialize)]
struct SeqBody {
    seq: SequenceId,
}

/// The body of `Cancel`.
#[derive(Serialize, Deserialize)]
struct CancelBody {
    seq: SequenceId,
    reason: CancelReason,
}

/// The body of `Release`.
#[derive(Serialize, Deserialize)]
struct ReleaseBody {
    seq: SequenceId,
    #[serde(default)]
    persist: Option<SessionId>,
}

/// The body of `Reconcile`, RFC-0033 §11's fields, as it is written.
#[derive(Serialize)]
struct ReconcileBody {
    seq: SequenceId,
    emitted: u32,
    rng_counter: u64,
    recent: Vec<u32>,
}

/// The body of `Reconcile` as it is read: `recent` through [`Recent`], so a long array is counted
/// and refused rather than held.
#[derive(Deserialize)]
struct ReconcileBodyIn {
    seq: SequenceId,
    emitted: u32,
    rng_counter: u64,
    recent: Recent,
}

/// `Reconcile::recent` as read: the first [`MAX_RECENT`] tokens, and how many the array held.
///
/// It never allocates. Every element is read and counted, but only the first [`MAX_RECENT`] are
/// kept, so a longer array is refused without the decoder holding it. A derived `Vec<u32>` would
/// reserve room for the array's stated length and hold every token before the length could be
/// checked. The envelope's length check has already refused a length the body cannot hold, and
/// the body is under its kind's ceiling ([`SMALL_BODY`]), so counting the rest is bounded too.
///
/// RFC-0033 §4.2 and §11 give `recent` as "the last up to 8 emitted tokens". They do not say
/// that it must hold exactly `min(emitted, 8)`, so the decoder does not check that; DIST-E14-T06's
/// handshake decides what a shorter `recent` means.
struct Recent {
    tokens: [u32; MAX_RECENT],
    len: usize,
}

impl Recent {
    /// The tokens kept: the first `min(len, MAX_RECENT)`.
    fn kept(&self) -> &[u32] {
        &self.tokens[..self.len.min(MAX_RECENT)]
    }
}

impl<'de> Deserialize<'de> for Recent {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Recent, D::Error> {
        struct Tokens;

        impl<'de> serde::de::Visitor<'de> for Tokens {
            type Value = Recent;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an array of token IDs")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Recent, A::Error> {
                let mut recent = Recent {
                    tokens: [0; MAX_RECENT],
                    len: 0,
                };
                while let Some(token) = seq.next_element::<u32>()? {
                    if let Some(slot) = recent.tokens.get_mut(recent.len) {
                        *slot = token;
                    }
                    recent.len = recent.len.saturating_add(1);
                }
                Ok(recent)
            }
        }

        deserializer.deserialize_seq(Tokens)
    }
}

/// `control`'s kind and its body as deterministic CBOR.
fn body(control: &SequenceControl) -> Result<(SeqControlKind, Vec<u8>), SeqControlError> {
    let kind = SeqControlKind::of(control).ok_or(SeqControlError::UnknownVariant)?;
    let cbor = match control {
        SequenceControl::Open(open) => envelope::to_cbor(open),
        SequenceControl::Cancel { seq, reason } => envelope::to_cbor(&CancelBody {
            seq: *seq,
            reason: *reason,
        }),
        SequenceControl::Pause { seq } | SequenceControl::Resume { seq } => {
            envelope::to_cbor(&SeqBody { seq: *seq })
        }
        SequenceControl::Release { seq, persist } => envelope::to_cbor(&ReleaseBody {
            seq: *seq,
            persist: *persist,
        }),
        SequenceControl::Reconcile {
            seq,
            emitted,
            rng_counter,
            recent,
        } => {
            if recent.len() > MAX_RECENT {
                return Err(SeqControlError::TooManyRecent { len: recent.len() });
            }
            envelope::to_cbor(&ReconcileBody {
                seq: *seq,
                emitted: *emitted,
                rng_counter: *rng_counter,
                recent: recent.to_vec(),
            })
        }
        _ => return Err(SeqControlError::UnknownVariant),
    };
    Ok((kind, cbor.map_err(SeqControlError::Encode)?))
}

/// The extension, padded to 64 bytes.
fn extension(kind: SeqControlKind, encoding: Encoding) -> [u8; ALIGN] {
    let mut ext = [0; ALIGN];
    ext[0] = kind.code();
    ext[1] = encoding as u8;
    ext
}

/// Writes `control` as a `SeqControl` frame for a receiver whose slots hold `slot_size` bytes:
/// plain CBOR when the frame fits, zstd-compressed when only that fits, and
/// [`SeqControlError::DoesNotFit`] when neither does. A body over its kind's ceiling is refused
/// first, with [`SeqControlError::BodyTooLarge`].
pub fn encode_seq_control(
    control: &SequenceControl,
    spec: &SeqControlSpec,
    slot_size: usize,
) -> Result<Vec<u8>, SeqControlError> {
    let (kind, cbor) = body(control)?;
    let limit = body_limit(kind, slot_size);
    if cbor.len() > limit {
        return Err(SeqControlError::BodyTooLarge {
            len: cbor.len(),
            limit,
        });
    }
    let frame = FrameSpec {
        wire_minor: spec.wire_minor,
        ftype: FrameType::SeqControl,
        flags: spec.flags,
        instance: spec.instance,
        epoch: spec.epoch,
        step: 0,
        chan_seq: spec.chan_seq,
    };
    if header::encoded_len(ALIGN, cbor.len()) <= slot_size {
        return Ok(header::encode_to_vec(
            &frame,
            &extension(kind, Encoding::Cbor),
            &cbor,
        )?);
    }
    let compressed = zstd::bulk::compress(&cbor, ZSTD_LEVEL)
        .map_err(|error| SeqControlError::Encode(EncodeError::Compress(error.to_string())))?;
    let len = header::encoded_len(ALIGN, compressed.len());
    if len > slot_size {
        return Err(SeqControlError::DoesNotFit { len, slot_size });
    }
    Ok(header::encode_to_vec(
        &frame,
        &extension(kind, Encoding::ZstdCbor),
        &compressed,
    )?)
}

/// A body of type `T`, read as `encoding` says.
fn read<T: serde::de::DeserializeOwned>(
    encoding: Encoding,
    payload: &[u8],
    limit: BodyLimit,
) -> Result<T, SeqControlError> {
    decode_encoded(encoding, payload, limit).map_err(SeqControlError::Body)
}

impl Frame<'_> {
    /// The `SequenceControl` of an unsealed `SeqControl` frame, within `limits`.
    pub fn seq_control(
        &self,
        limits: SeqControlLimits,
    ) -> Result<SequenceControl, SeqControlError> {
        let ftype = self.header.spec.ftype;
        if ftype != FrameType::SeqControl {
            return Err(FrameError::WrongType {
                expected: FrameType::SeqControl,
                found: ftype,
            }
            .into());
        }
        if self.header.spec.flags.encrypted() {
            return Err(FrameError::Sealed(ftype).into());
        }
        // A frame's length fits in usize: the decoder found that many bytes.
        let len = self.header.frame_len() as usize;
        if len > limits.slot_size {
            return Err(SeqControlError::OverSlot {
                len,
                slot_size: limits.slot_size,
            });
        }
        let (kind, encoding) = self.seq_control_kind()?;
        let payload = self.payload;
        // The ceiling applies before and after decompression, so a small frame cannot make this
        // decoder hold a large body (RFC-0060 §10.2).
        let body = match kind {
            SeqControlKind::Open | SeqControlKind::ResumeOpen => limits.body,
            _ => BodyLimit::new(SMALL_BODY.min(limits.body.bytes()))
                .expect("SMALL_BODY is within the envelope's limit"),
        };
        Ok(match kind {
            SeqControlKind::Open | SeqControlKind::ResumeOpen => {
                let open: SequenceOpen = read(encoding, payload, body)?;
                if open.resume.is_some() != (kind == SeqControlKind::ResumeOpen) {
                    return Err(SeqControlError::ResumeMismatch { kind });
                }
                SequenceControl::Open(open)
            }
            SeqControlKind::Cancel => {
                let CancelBody { seq, reason } = read(encoding, payload, body)?;
                SequenceControl::Cancel { seq, reason }
            }
            SeqControlKind::Pause => {
                let SeqBody { seq } = read(encoding, payload, body)?;
                SequenceControl::Pause { seq }
            }
            SeqControlKind::Resume => {
                let SeqBody { seq } = read(encoding, payload, body)?;
                SequenceControl::Resume { seq }
            }
            SeqControlKind::Release => {
                let ReleaseBody { seq, persist } = read(encoding, payload, body)?;
                SequenceControl::Release { seq, persist }
            }
            SeqControlKind::Reconcile => {
                let ReconcileBodyIn {
                    seq,
                    emitted,
                    rng_counter,
                    recent,
                } = read(encoding, payload, body)?;
                if recent.len > MAX_RECENT {
                    return Err(SeqControlError::TooManyRecent { len: recent.len });
                }
                SequenceControl::Reconcile {
                    seq,
                    emitted,
                    rng_counter,
                    recent: recent.kept().iter().copied().collect(),
                }
            }
        })
    }

    /// The `kind` and `body_enc` of a `SeqControl` frame, without reading its body. They sit in
    /// the extension's first 16 bytes, which a sealed frame leaves in plaintext (RFC-0031 §6.1), so
    /// this reads a sealed frame too. The reserved bytes and the padding are ignored, so a later
    /// minor may use them.
    pub fn seq_control_kind(&self) -> Result<(SeqControlKind, Encoding), SeqControlError> {
        if self.header.spec.ftype != FrameType::SeqControl {
            return Err(FrameError::WrongType {
                expected: FrameType::SeqControl,
                found: self.header.spec.ftype,
            }
            .into());
        }
        if self.ext.len() < SEQ_CONTROL_FIXED_LEN {
            return Err(FrameError::ExtTooShort {
                ftype: FrameType::SeqControl,
                len: self.ext.len(),
                needed: SEQ_CONTROL_FIXED_LEN,
            }
            .into());
        }
        let kind = SeqControlKind::from_code(self.ext[0])
            .ok_or(SeqControlError::UnknownKind(self.ext[0]))?;
        let encoding = Encoding::from_byte(self.ext[1]).ok_or(SeqControlError::Body(
            DecodeError::UnknownEncoding(self.ext[1]),
        ))?;
        Ok((kind, encoding))
    }
}

/// `meta` as the value of a `SEQ_META` item: deterministic CBOR (RFC-0031 §6.3).
pub fn encode_seq_meta(meta: &SequenceMeta) -> Result<Vec<u8>, EncodeError> {
    envelope::to_cbor(meta)
}

/// The `SequenceMeta` a `SEQ_META` item's value holds. Unknown fields are ignored, as in a
/// `SeqControl` body.
pub fn decode_seq_meta(value: &[u8]) -> Result<SequenceMeta, DecodeError> {
    envelope::from_cbor(value)
}
