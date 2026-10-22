// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `EmbeddingResult` frame (RFC-0031 §6.14, RFC-0050 amendment A5): the pooled vector of a
//! prefill-only embedding sequence, tail → ingress on the `Tokens` channel, sent in place of
//! tokens after the sequence's last prefill chunk.
//!
//! ```text
//! extension | seq [16] | dims u32 | dtype u8 | pooling u8 | normalised u8 | reserved u8 |
//!           | prompt_tokens u32 | part_index u16 | part_count u16 | padding to 64 |
//! payload   this part's elements of the vector, dims × f32 in all, little-endian
//! ```
//!
//! One result per sequence, in one or more parts. A vector larger than the channel's slot is
//! split at element boundaries ([`EmbeddingResult::parts`]), each part a whole frame, and the
//! receiver acts on the result only when every part has arrived ([`assemble`]), as with `Token`
//! (§6.5). The part fields took the extension's `reserved[4]` at offset 28 (RFC-0031 §6.14,
//! revised 2026-10-23); `part_count = 0`, what those reserved bytes held, is refused.
//!
//! The frame type exists only on a channel that negotiated capability bit 14 `EMBEDDING`
//! ([`capability::EMBEDDING`]): the encoder refuses to write one for a channel whose negotiated
//! capabilities lack it, so a peer without it never receives one. The intersection itself is
//! DIST-E8-T13's.
//!
//! A decoded frame borrows its vector from the frame's bytes; an encoded one is written straight
//! into the caller's buffer.

use std::ops::Range;

use syntropy_core::{InstanceId, PlanEpoch, SequenceId};

use super::activation::{get16, get32, get64};
use super::error::FrameError;
use super::header::{self, ALIGN, Flags, Frame, FrameSpec, FrameType, HEADER_LEN};

/// The capability bits of RFC-0031 §10 that this crate's codecs gate on. DIST-E8-T13 adds the
/// rest of the set and its intersection.
pub mod capability {
    /// Bit 14: the peer reads `EmbeddingResult` frames (RFC-0031 §6.14, RFC-0050 amendment A5).
    pub const EMBEDDING: u64 = 1 << 14;
}

/// The extension's fields, before its padding.
pub const EMBEDDING_FIXED_LEN: usize = 32;

/// The extension as the encoder writes it: the fields, padded to 64 bytes.
const EXT_LEN: usize = EMBEDDING_FIXED_LEN.next_multiple_of(ALIGN);

/// The `dtype` of v1's vectors, F32.
pub const EMBEDDING_F32: u8 = 1;

/// Bytes in one element of a v1 vector.
const ELEMENT: usize = 4;

/// How the tail pooled the sequence's hidden states into the vector (RFC-0050 §14).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pooling {
    /// The last token's hidden state.
    LastToken,
    /// The mean over the sequence's tokens.
    Mean,
    /// The first (`CLS`) token's hidden state.
    Cls,
}

impl Pooling {
    /// The value in the extension's `pooling` byte.
    pub const fn wire(self) -> u8 {
        match self {
            Pooling::LastToken => 1,
            Pooling::Mean => 2,
            Pooling::Cls => 3,
        }
    }

    /// The pooling a `pooling` byte names, if it names one.
    pub const fn from_wire(byte: u8) -> Option<Pooling> {
        match byte {
            1 => Some(Pooling::LastToken),
            2 => Some(Pooling::Mean),
            3 => Some(Pooling::Cls),
            _ => None,
        }
    }
}

/// The header fields of an unsealed `EmbeddingResult` frame. `HAS_TRACE` must be clear, since
/// the frame has no TLV area; `ENCRYPTED` is the AEAD layer's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EmbeddingSpec {
    pub wire_minor: u8,
    pub flags: Flags,
    pub instance: InstanceId,
    pub epoch: PlanEpoch,
    pub step: u64,
    pub chan_seq: u64,
}

/// An `EmbeddingResult` frame as it travels: a sequence's whole vector, or one part of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddingResult<'a> {
    /// The embedding sequence.
    pub seq: SequenceId,
    /// The vector's length in elements, across all of its parts.
    pub dims: u32,
    pub pooling: Pooling,
    /// Whether the vector was normalised after pooling.
    pub normalised: bool,
    /// The sequence's prompt length in tokens, which the response's usage reports.
    pub prompt_tokens: u32,
    /// 0-based; a vector larger than the channel's slot travels in parts.
    pub part_index: u16,
    /// At least 1.
    pub part_count: u16,
    /// This part's elements, little-endian F32: a whole number of them, at least one, and all
    /// `dims` in a one-part result.
    pub vector: &'a [u8],
}

fn bad(field: &'static str, value: impl Into<u64>) -> FrameError {
    FrameError::BadField {
        ftype: FrameType::EmbeddingResult,
        field,
        value: value.into(),
    }
}

fn inconsistent(field: &'static str, value: u64, expected: u64) -> FrameError {
    FrameError::Inconsistent {
        ftype: FrameType::EmbeddingResult,
        field,
        value,
        expected,
    }
}

impl<'a> EmbeddingResult<'a> {
    /// Reads an unsealed `EmbeddingResult` extension and payload. Bytes after the fields, the
    /// reserved byte at offset 23 and the padding included, are ignored, so a later minor can
    /// add fields there.
    pub fn parse(ext: &'a [u8], payload: &'a [u8]) -> Result<EmbeddingResult<'a>, FrameError> {
        if ext.len() < EMBEDDING_FIXED_LEN {
            return Err(FrameError::ExtTooShort {
                ftype: FrameType::EmbeddingResult,
                len: ext.len(),
                needed: EMBEDDING_FIXED_LEN,
            });
        }
        if ext[20] != EMBEDDING_F32 {
            return Err(bad("dtype", ext[20]));
        }
        let pooling = Pooling::from_wire(ext[21]).ok_or_else(|| bad("pooling", ext[21]))?;
        let normalised = match ext[22] {
            0 => false,
            1 => true,
            other => return Err(bad("normalised", other)),
        };
        let result = EmbeddingResult {
            seq: SequenceId {
                ingress: get64(ext, 0),
                boot: get32(ext, 8),
                n: get32(ext, 12),
            },
            dims: get32(ext, 16),
            pooling,
            normalised,
            prompt_tokens: get32(ext, 24),
            part_index: get16(ext, 28),
            part_count: get16(ext, 30),
            vector: payload,
        };
        result.validate()?;
        Ok(result)
    }

    /// Checks the rules the bytes cannot express:
    /// - `dims` is at least 1;
    /// - `part_count` is at least 1, and `part_index` below it;
    /// - the vector is a whole number of elements, at least one, and at most `dims`;
    /// - a one-part result carries all `dims` elements.
    pub fn validate(&self) -> Result<(), FrameError> {
        if self.dims == 0 {
            return Err(bad("dims", 0u64));
        }
        if self.part_count == 0 {
            return Err(bad("part_count", 0u64));
        }
        if self.part_index >= self.part_count {
            return Err(bad("part_index", self.part_index));
        }
        let len = self.vector.len() as u64;
        let whole = u64::from(self.dims) * ELEMENT as u64;
        if len == 0 || !len.is_multiple_of(ELEMENT as u64) || len > whole {
            return Err(inconsistent("payload_len", len, whole));
        }
        if self.part_count == 1 && len != whole {
            return Err(inconsistent("payload_len", len, whole));
        }
        Ok(())
    }

    /// The elements this frame carries, in order.
    pub fn elements(&self) -> impl ExactSizeIterator<Item = f32> + 'a {
        self.vector
            .as_chunks::<ELEMENT>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
    }

    /// The extension's bytes: the fields, then zeroed padding.
    fn ext(&self) -> [u8; EXT_LEN] {
        let mut ext = [0u8; EXT_LEN];
        ext[0..8].copy_from_slice(&self.seq.ingress.to_le_bytes());
        ext[8..12].copy_from_slice(&self.seq.boot.to_le_bytes());
        ext[12..16].copy_from_slice(&self.seq.n.to_le_bytes());
        ext[16..20].copy_from_slice(&self.dims.to_le_bytes());
        ext[20] = EMBEDDING_F32;
        ext[21] = self.pooling.wire();
        ext[22] = u8::from(self.normalised);
        ext[24..28].copy_from_slice(&self.prompt_tokens.to_le_bytes());
        ext[28..30].copy_from_slice(&self.part_index.to_le_bytes());
        ext[30..32].copy_from_slice(&self.part_count.to_le_bytes());
        ext
    }

    /// The frame's length, header and padding included.
    pub fn frame_len(&self) -> usize {
        header::encoded_len(EXT_LEN, self.vector.len())
    }

    /// Encodes this frame, unsealed, into `out` and returns its length. `capabilities` are the
    /// channel's negotiated set (RFC-0031 §10), which must hold bit 14 `EMBEDDING`: a peer that
    /// did not advertise it never receives the frame.
    pub fn encode(
        &self,
        spec: &EmbeddingSpec,
        capabilities: u64,
        out: &mut [u8],
    ) -> Result<usize, FrameError> {
        if capabilities & capability::EMBEDDING == 0 {
            return Err(FrameError::CapabilityNotNegotiated {
                ftype: FrameType::EmbeddingResult,
                capability: capability::EMBEDDING,
            });
        }
        self.validate()?;
        if spec.flags.has_trace() {
            return Err(inconsistent("HAS_TRACE", 1, 0));
        }
        let frame = FrameSpec {
            wire_minor: spec.wire_minor,
            ftype: FrameType::EmbeddingResult,
            flags: spec.flags,
            instance: spec.instance,
            epoch: spec.epoch,
            step: spec.step,
            chan_seq: spec.chan_seq,
        };
        header::encode(&frame, &self.ext(), self.vector, out)
    }

    /// [`EmbeddingResult::encode`] into a new vector.
    pub fn encode_to_vec(
        &self,
        spec: &EmbeddingSpec,
        capabilities: u64,
    ) -> Result<Vec<u8>, FrameError> {
        let mut out = vec![0; self.frame_len()];
        let len = self.encode(spec, capabilities, &mut out)?;
        out.truncate(len);
        Ok(out)
    }

    /// The parts this whole result travels in when frames are at most `max_frame` bytes, header
    /// included: the fewest, each holding as many elements as fit beside the 64-byte header and
    /// the 64-byte extension. A vector that fits is one part. The result must be whole,
    /// `part_count = 1`, and each part is encoded with [`EmbeddingResult::encode`].
    pub fn parts(&self, max_frame: usize) -> Result<EmbeddingParts<'a>, FrameError> {
        self.validate()?;
        if self.part_count != 1 {
            return Err(bad("part_count", self.part_count));
        }
        let per_part = max_frame.saturating_sub(HEADER_LEN + EXT_LEN) / ELEMENT;
        if per_part == 0 {
            return Err(FrameError::TooLong {
                what: "embedding part",
                len: HEADER_LEN + EXT_LEN + ELEMENT,
            });
        }
        let elements = self.vector.len() / ELEMENT;
        let count =
            u16::try_from(elements.div_ceil(per_part)).map_err(|_| FrameError::TooLong {
                what: "part count",
                len: elements.div_ceil(per_part),
            })?;
        Ok(EmbeddingParts {
            whole: *self,
            per_part,
            index: 0,
            count,
        })
    }
}

/// The parts of a whole result, in order: see [`EmbeddingResult::parts`]. Its length is the part
/// count.
#[derive(Clone, Debug)]
pub struct EmbeddingParts<'a> {
    whole: EmbeddingResult<'a>,
    per_part: usize,
    index: u16,
    count: u16,
}

impl<'a> EmbeddingParts<'a> {
    /// The elements part `index` holds.
    fn elements(&self, index: u16) -> Range<usize> {
        let total = self.whole.vector.len() / ELEMENT;
        let start = usize::from(index) * self.per_part;
        start..(start + self.per_part).min(total)
    }
}

impl ExactSizeIterator for EmbeddingParts<'_> {}

impl<'a> Iterator for EmbeddingParts<'a> {
    type Item = EmbeddingResult<'a>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = usize::from(self.count - self.index);
        (left, Some(left))
    }

    fn next(&mut self) -> Option<EmbeddingResult<'a>> {
        if self.index == self.count {
            return None;
        }
        let range = self.elements(self.index);
        let part = EmbeddingResult {
            part_index: self.index,
            part_count: self.count,
            vector: &self.whole.vector[range.start * ELEMENT..range.end * ELEMENT],
            ..self.whole
        };
        self.index += 1;
        Some(part)
    }
}

/// A sequence's vector from all of its parts, which the receiver acts on only when every part
/// has arrived (RFC-0031 §6.5, §6.14). The parts may come in any order; each `part_index` must
/// appear once, every part must agree on the result's fields, and the parts must hold `dims`
/// elements in all.
pub fn assemble(parts: &[EmbeddingResult<'_>]) -> Result<Vec<f32>, FrameError> {
    let first = parts.first().ok_or(inconsistent("part_count", 0, 1))?;
    let count = first.part_count;
    if parts.len() != usize::from(count) {
        return Err(inconsistent(
            "part_count",
            parts.len() as u64,
            u64::from(count),
        ));
    }
    let mut ordered: Vec<Option<&EmbeddingResult<'_>>> = vec![None; parts.len()];
    for part in parts {
        part.validate()?;
        let agree = [
            ("seq", part.seq == first.seq),
            ("dims", part.dims == first.dims),
            ("pooling", part.pooling == first.pooling),
            ("normalised", part.normalised == first.normalised),
            ("prompt_tokens", part.prompt_tokens == first.prompt_tokens),
            ("part_count", part.part_count == count),
        ];
        if let Some((field, _)) = agree.iter().find(|(_, same)| !same) {
            return Err(bad(field, u64::from(part.part_index)));
        }
        let slot = &mut ordered[usize::from(part.part_index)];
        if slot.is_some() {
            return Err(bad("part_index", part.part_index));
        }
        *slot = Some(part);
    }
    let held: u64 = parts
        .iter()
        .map(|p| (p.vector.len() / ELEMENT) as u64)
        .sum();
    if held != u64::from(first.dims) {
        return Err(inconsistent("dims", held, u64::from(first.dims)));
    }
    Ok(ordered
        .into_iter()
        .flatten()
        .flat_map(|part| part.elements())
        .collect())
}

impl<'a> Frame<'a> {
    /// The `EmbeddingResult` of an unsealed frame. `HAS_TRACE` must be clear.
    pub fn embedding_result(&self) -> Result<EmbeddingResult<'a>, FrameError> {
        let ftype = self.header.spec.ftype;
        if ftype != FrameType::EmbeddingResult {
            return Err(FrameError::WrongType {
                expected: FrameType::EmbeddingResult,
                found: ftype,
            });
        }
        let flags = self.header.spec.flags;
        if flags.encrypted() {
            return Err(FrameError::Sealed(ftype));
        }
        if flags.has_trace() {
            return Err(inconsistent("HAS_TRACE", 1, 0));
        }
        EmbeddingResult::parse(self.ext, self.payload)
    }
}
