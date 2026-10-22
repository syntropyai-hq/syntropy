// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Collective` frame of RFC-0031 §6.8: partial sums, gathered slices and the
//! `ROUTER_BCAST` of a tensor-parallel island (RFC-0032 §5.2, §5.3).
//!
//! The extension is 32 bytes of fields, padded to 64:
//!
//! ```text
//! 0   8   step          the step; equal to the base header's
//! 8   2   layer
//! 10  1   phase         0 ATTN_ALLREDUCE | 1 FFN_ALLREDUCE | 2 ENTRY_BCAST | 3 EXIT_GATHER | 4 ROUTER_BCAST
//! 11  1   algo          0 ONE_SHOT | 1 RING_RS (reduce-scatter step) | 2 RING_AG (all-gather step)
//! 12  2   src_member
//! 14  2   n_members
//! 16  2   ring_step
//! 18  1   dtype         WireDType; F32 in exact mode (RFC-0032 §5.3)
//! 19  1   reserved
//! 20  4   elems         elements in this part
//! 24  4   elem_offset   offset of this part in the full vector (ring algorithms)
//! 28  4   rows
//! ```
//!
//! The payload of every phase but `ROUTER_BCAST` is `elems` elements of `dtype`, packed: a ring
//! step's slice of the vector starts at element `elem_offset` and need not start on a row. The
//! `ROUTER_BCAST` payload is `rows` routing rows ([`RouterRows`]), and `elems` counts the
//! experts they name. RFC-0031 §6.8's revision note of 2026-10-22 records these readings.
//!
//! [`Frame::collective`] reads a frame on a channel whose wire dtype is `wire`, and
//! [`Collective::encode`] writes one. Neither allocates.

use syntropy_core::plan::WireDType;
use syntropy_core::{InstanceId, PlanEpoch};

use super::activation::{dtype_code, dtype_from_code};
use super::error::FrameError;
use super::header::{self, ALIGN, Flags, Frame, FrameSpec, FrameType, HEADER_LEN};
use super::route::{Head, RouterRows, walk};

/// The bytes of the extension's fields; the extension is padded to 64.
pub const COLLECTIVE_FIELDS_LEN: usize = 32;

/// What a `Collective` frame carries (RFC-0031 §6.8 `phase`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CollectivePhase {
    /// The attention output's partial sums.
    AttnAllReduce,
    /// The FFN output's partial sums.
    FfnAllReduce,
    /// The stage's input, from the primary to the members.
    EntryBcast,
    /// The stage's output slices, gathered to the primary.
    ExitGather,
    /// The routed experts and weights per row, from the primary (RFC-0032 §5.2).
    RouterBcast,
}

impl CollectivePhase {
    pub fn code(self) -> u8 {
        match self {
            CollectivePhase::AttnAllReduce => 0,
            CollectivePhase::FfnAllReduce => 1,
            CollectivePhase::EntryBcast => 2,
            CollectivePhase::ExitGather => 3,
            CollectivePhase::RouterBcast => 4,
        }
    }

    pub fn from_code(code: u8) -> Option<CollectivePhase> {
        Some(match code {
            0 => CollectivePhase::AttnAllReduce,
            1 => CollectivePhase::FfnAllReduce,
            2 => CollectivePhase::EntryBcast,
            3 => CollectivePhase::ExitGather,
            4 => CollectivePhase::RouterBcast,
            _ => return None,
        })
    }
}

/// The step of the collective algorithm a frame belongs to (RFC-0031 §6.8 `algo`; the plan's
/// `CollectiveAlgo` is RFC-0032 §5.1's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WireAlgo {
    /// Each member sends its whole partial to every other member.
    OneShot,
    /// A reduce-scatter step of a ring all-reduce.
    RingReduceScatter,
    /// An all-gather step of a ring all-reduce.
    RingAllGather,
}

impl WireAlgo {
    pub fn code(self) -> u8 {
        match self {
            WireAlgo::OneShot => 0,
            WireAlgo::RingReduceScatter => 1,
            WireAlgo::RingAllGather => 2,
        }
    }

    pub fn from_code(code: u8) -> Option<WireAlgo> {
        Some(match code {
            0 => WireAlgo::OneShot,
            1 => WireAlgo::RingReduceScatter,
            2 => WireAlgo::RingAllGather,
            _ => return None,
        })
    }
}

/// The extension's fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CollectiveHeader {
    /// The step, which the base header carries too.
    pub step: u64,
    pub layer: u16,
    pub phase: CollectivePhase,
    pub algo: WireAlgo,
    /// The sending member's index in the island's member order; 0 is the primary.
    pub src_member: u16,
    pub n_members: u16,
    pub ring_step: u16,
    pub dtype: WireDType,
    /// Elements in this part; for `ROUTER_BCAST`, the experts its rows name in all.
    pub elems: u32,
    /// This part's offset in the full vector, in elements.
    pub elem_offset: u32,
    /// Token positions the vector covers; for `ROUTER_BCAST`, its rows.
    pub rows: u32,
}

/// A `Collective` payload: packed elements, or `ROUTER_BCAST`'s routing rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CollectivePayload<'a> {
    /// `elems` elements of the header's dtype.
    Elements(&'a [u8]),
    /// `rows` routing rows.
    Router(RouterRows<'a>),
}

/// The header fields of an unsealed frame on a `Collective` channel: `Collective`, `EpDispatch`
/// or `EpCombine`. The step is the extension's, and the lengths and CRCs follow from the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CollectiveSpec {
    pub wire_minor: u8,
    pub flags: Flags,
    pub instance: InstanceId,
    pub epoch: PlanEpoch,
    pub chan_seq: u64,
}

impl CollectiveSpec {
    /// The base header's fields for a frame of `ftype` at `step`. These frames carry no
    /// `TRACE`, so `HAS_TRACE` is cleared; a sealed frame is the AEAD layer's to write.
    pub(super) fn frame(&self, ftype: FrameType, step: u64) -> Result<FrameSpec, FrameError> {
        if self.flags.encrypted() {
            return Err(FrameError::BadField {
                ftype,
                field: "flags",
                value: self.flags.0.into(),
            });
        }
        Ok(FrameSpec {
            wire_minor: self.wire_minor,
            ftype,
            flags: Flags(self.flags.0 & !Flags::HAS_TRACE),
            instance: self.instance,
            epoch: self.epoch,
            step,
            chan_seq: self.chan_seq,
        })
    }
}

/// A frame's `Collective`: the extension's fields and the payload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Collective<'a> {
    pub header: CollectiveHeader,
    pub payload: CollectivePayload<'a>,
}

pub(super) fn get16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

pub(super) fn get32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

pub(super) fn get64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

fn bad(field: &'static str, value: impl Into<u64>) -> FrameError {
    FrameError::BadField {
        ftype: FrameType::Collective,
        field,
        value: value.into(),
    }
}

fn inconsistent(
    field: &'static str,
    value: impl Into<u64>,
    expected: impl Into<u64>,
) -> FrameError {
    FrameError::Inconsistent {
        ftype: FrameType::Collective,
        field,
        value: value.into(),
        expected: expected.into(),
    }
}

/// Bytes per element of the dtypes this build carries: 4 for F32, 2 for BF16. The approximate
/// containers come with DIST-E8-T11.
pub(super) fn element_len(dtype: WireDType) -> Result<usize, FrameError> {
    match dtype {
        WireDType::F32 => Ok(4),
        WireDType::BF16 => Ok(2),
        other => Err(FrameError::UnsupportedDType(other)),
    }
}

/// Reads the dtype at `code` against the channel's `wire`, as RFC-0031 §6.4 requires of every
/// frame that names one.
pub(super) fn read_dtype(
    code: u8,
    wire: WireDType,
    ftype: FrameType,
) -> Result<WireDType, FrameError> {
    let dtype = dtype_from_code(code).ok_or(FrameError::BadField {
        ftype,
        field: "dtype",
        value: code.into(),
    })?;
    if dtype != wire {
        return Err(FrameError::WireDTypeMismatch {
            expected: wire,
            found: dtype,
        });
    }
    Ok(dtype)
}

impl<'a> Collective<'a> {
    /// Reads an unsealed `Collective` extension and payload on a channel whose wire dtype is
    /// `wire` (`ChannelSpec::wire`). A frame in another dtype is refused with
    /// [`FrameError::WireDTypeMismatch`]. Bytes past the fields are reserved and ignored.
    pub fn parse(
        ext: &'a [u8],
        payload: &'a [u8],
        wire: WireDType,
    ) -> Result<Collective<'a>, FrameError> {
        if ext.len() < COLLECTIVE_FIELDS_LEN {
            return Err(FrameError::ExtTooShort {
                ftype: FrameType::Collective,
                len: ext.len(),
                needed: COLLECTIVE_FIELDS_LEN,
            });
        }
        let phase = CollectivePhase::from_code(ext[10]).ok_or(bad("phase", ext[10]))?;
        let algo = WireAlgo::from_code(ext[11]).ok_or(bad("algo", ext[11]))?;
        let dtype = read_dtype(ext[18], wire, FrameType::Collective)?;
        let header = CollectiveHeader {
            step: get64(ext, 0),
            layer: get16(ext, 8),
            phase,
            algo,
            src_member: get16(ext, 12),
            n_members: get16(ext, 14),
            ring_step: get16(ext, 16),
            dtype,
            elems: get32(ext, 20),
            elem_offset: get32(ext, 24),
            rows: get32(ext, 28),
        };
        let payload = if phase == CollectivePhase::RouterBcast {
            let (len, _) = walk(
                Head::Router,
                payload,
                header.rows as usize,
                FrameType::Collective,
            )?;
            if len != payload.len() {
                return Err(inconsistent(
                    "payload_len",
                    payload.len() as u64,
                    len as u64,
                ));
            }
            CollectivePayload::Router(RouterRows::wire(payload, header.rows as usize))
        } else {
            CollectivePayload::Elements(payload)
        };
        let collective = Collective { header, payload };
        collective.validate()?;
        Ok(collective)
    }

    /// Checks the rules the bytes cannot express:
    /// - the dtype is F32 or BF16;
    /// - at least two members, and `src_member` is one of them;
    /// - a one-shot frame has `ring_step` 0;
    /// - `ROUTER_BCAST` comes from the primary, member 0, in one shot, with routing rows: `rows`
    ///   of them, naming `elems` experts in all, at offset 0, each with as many weights as
    ///   expert IDs;
    /// - every other phase carries `elems` elements of the dtype.
    pub fn validate(&self) -> Result<(), FrameError> {
        let h = &self.header;
        let element = element_len(h.dtype)?;
        if h.n_members < 2 {
            return Err(bad("n_members", h.n_members));
        }
        if h.src_member >= h.n_members {
            return Err(inconsistent("src_member", h.src_member, h.n_members));
        }
        if h.algo == WireAlgo::OneShot && h.ring_step != 0 {
            return Err(bad("ring_step", h.ring_step));
        }
        match (h.phase, &self.payload) {
            (CollectivePhase::RouterBcast, CollectivePayload::Router(rows)) => {
                if h.algo != WireAlgo::OneShot {
                    return Err(bad("algo", h.algo.code()));
                }
                if h.src_member != 0 {
                    return Err(inconsistent("src_member", h.src_member, 0u64));
                }
                if h.elem_offset != 0 {
                    return Err(bad("elem_offset", h.elem_offset));
                }
                if rows.len() != h.rows as usize {
                    return Err(inconsistent("rows", h.rows, rows.len() as u64));
                }
                let (_, experts) = rows.measure(FrameType::Collective)?;
                if experts != u64::from(h.elems) {
                    return Err(inconsistent("elems", h.elems, experts));
                }
            }
            (CollectivePhase::RouterBcast, CollectivePayload::Elements(_)) => {
                return Err(bad("phase", h.phase.code()));
            }
            (_, CollectivePayload::Router(_)) => return Err(bad("phase", h.phase.code())),
            (_, CollectivePayload::Elements(bytes)) => {
                let expected = u64::from(h.elems) * element as u64;
                if bytes.len() as u64 != expected {
                    return Err(inconsistent("payload_len", bytes.len() as u64, expected));
                }
            }
        }
        Ok(())
    }

    fn payload_len(&self) -> Result<usize, FrameError> {
        match self.payload {
            CollectivePayload::Elements(bytes) => Ok(bytes.len()),
            CollectivePayload::Router(rows) => Ok(rows.measure(FrameType::Collective)?.0),
        }
    }

    /// The frame's length, header and padding included.
    pub fn encoded_len(&self) -> Result<usize, FrameError> {
        self.validate()?;
        Ok(header::encoded_len(
            COLLECTIVE_FIELDS_LEN,
            self.payload_len()?,
        ))
    }

    /// Writes the extension's 64 bytes into `out` and returns 64.
    pub fn write_ext(&self, out: &mut [u8]) -> Result<usize, FrameError> {
        self.validate()?;
        let available = out.len();
        let ext = out.get_mut(..ALIGN).ok_or(FrameError::BufferTooSmall {
            needed: ALIGN,
            available,
        })?;
        let h = &self.header;
        ext.fill(0);
        ext[0..8].copy_from_slice(&h.step.to_le_bytes());
        ext[8..10].copy_from_slice(&h.layer.to_le_bytes());
        ext[10] = h.phase.code();
        ext[11] = h.algo.code();
        ext[12..14].copy_from_slice(&h.src_member.to_le_bytes());
        ext[14..16].copy_from_slice(&h.n_members.to_le_bytes());
        ext[16..18].copy_from_slice(&h.ring_step.to_le_bytes());
        ext[18] = dtype_code(h.dtype).ok_or(FrameError::UnsupportedDType(h.dtype))?;
        ext[20..24].copy_from_slice(&h.elems.to_le_bytes());
        ext[24..28].copy_from_slice(&h.elem_offset.to_le_bytes());
        ext[28..32].copy_from_slice(&h.rows.to_le_bytes());
        Ok(ALIGN)
    }

    /// Encodes this frame, unsealed, into `out` and returns its length. The base header's step
    /// is the extension's.
    pub fn encode(&self, spec: &CollectiveSpec, out: &mut [u8]) -> Result<usize, FrameError> {
        let total = self.encoded_len()?;
        let frame = spec.frame(FrameType::Collective, self.header.step)?;
        if out.len() < total {
            return Err(FrameError::BufferTooSmall {
                needed: total,
                available: out.len(),
            });
        }
        let ext_len = self.write_ext(&mut out[HEADER_LEN..])?;
        match self.payload {
            CollectivePayload::Elements(bytes) => {
                header::encode_in_place(&frame, ext_len, bytes, out)
            }
            CollectivePayload::Router(rows) => {
                let len = rows.write(&mut out[HEADER_LEN + ext_len..total]);
                header::encode_written(&frame, ext_len, len, out)
            }
        }
    }

    /// [`Collective::encode`] into a new vector.
    pub fn encode_to_vec(&self, spec: &CollectiveSpec) -> Result<Vec<u8>, FrameError> {
        let mut out = vec![0; self.encoded_len()?];
        let len = self.encode(spec, &mut out)?;
        out.truncate(len);
        Ok(out)
    }
}

/// Checks what a typed read of a `Collective`-channel frame needs of the base header: the frame
/// type, no seal, no `HAS_TRACE`.
pub(super) fn check_base(frame: &Frame<'_>, ftype: FrameType) -> Result<(), FrameError> {
    let found = frame.header.spec.ftype;
    if found != ftype {
        return Err(FrameError::WrongType {
            expected: ftype,
            found,
        });
    }
    let flags = frame.header.spec.flags;
    if flags.encrypted() {
        return Err(FrameError::Sealed(ftype));
    }
    if flags.has_trace() {
        return Err(FrameError::Inconsistent {
            ftype,
            field: "HAS_TRACE",
            value: 1,
            expected: 0,
        });
    }
    Ok(())
}

/// Checks that the extension's step is the base header's.
pub(super) fn check_step(frame: &Frame<'_>, ftype: FrameType, step: u64) -> Result<(), FrameError> {
    let base = frame.header.spec.step;
    if step != base {
        return Err(FrameError::Inconsistent {
            ftype,
            field: "step",
            value: step,
            expected: base,
        });
    }
    Ok(())
}

impl<'a> Frame<'a> {
    /// The `Collective` of an unsealed frame, read on a channel whose wire dtype is `wire`. The
    /// extension's step must be the base header's, and `HAS_TRACE` must be clear.
    pub fn collective(&self, wire: WireDType) -> Result<Collective<'a>, FrameError> {
        check_base(self, FrameType::Collective)?;
        let collective = Collective::parse(self.ext, self.payload, wire)?;
        check_step(self, FrameType::Collective, collective.header.step)?;
        Ok(collective)
    }
}
