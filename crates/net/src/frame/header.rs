// SPDX-License-Identifier: MIT OR Apache-2.0

//! The 64-byte base header and the raw frame: header, extension and payload (RFC-0031 §6, §6.1,
//! §6.2).

use syntropy_core::{InstanceId, PlanEpoch};

use super::crc::{crc32c, crc32c_append};
use super::error::FrameError;

/// The first four bytes of every frame: "SYNF".
pub const MAGIC: [u8; 4] = *b"SYNF";

/// The base header's length, and the alignment of the extension and the payload.
pub const HEADER_LEN: usize = 64;

/// Extensions are padded so the payload starts 64-byte aligned.
pub const ALIGN: usize = 64;

/// The wire minor this crate writes. RFC-0031 §10 makes minors additive; the first layout is 1.
pub const WIRE_MINOR: u8 = 1;

/// The AEAD tag that ends a sealed frame's payload, counted in `payload_len` (RFC-0031 §6.1).
pub const AEAD_TAG_LEN: usize = 16;

/// A sealed frame's extension stays plaintext for its first 16 bytes, for routing; the rest is
/// sealed (RFC-0031 §6.1).
pub const SEALED_EXT_PLAINTEXT: usize = 16;

/// The frame types of RFC-0031 §6.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum FrameType {
    Activation = 0x01,
    Token = 0x02,
    SeqControl = 0x03,
    Credit = 0x04,
    KvChunk = 0x05,
    BlobData = 0x06,
    BlobRequest = 0x07,
    Collective = 0x08,
    StepDescriptor = 0x09,
    Hello = 0x0A,
    HelloAck = 0x0B,
    Close = 0x0C,
    Recover = 0x0D,
    EpDispatch = 0x0E,
    EpCombine = 0x0F,
    Ping = 0x10,
    TapIds = 0x11,
    EmbeddingResult = 0x12,
}

impl FrameType {
    /// Every frame type, in code order.
    pub const ALL: [FrameType; 18] = [
        FrameType::Activation,
        FrameType::Token,
        FrameType::SeqControl,
        FrameType::Credit,
        FrameType::KvChunk,
        FrameType::BlobData,
        FrameType::BlobRequest,
        FrameType::Collective,
        FrameType::StepDescriptor,
        FrameType::Hello,
        FrameType::HelloAck,
        FrameType::Close,
        FrameType::Recover,
        FrameType::EpDispatch,
        FrameType::EpCombine,
        FrameType::Ping,
        FrameType::TapIds,
        FrameType::EmbeddingResult,
    ];

    /// The type with wire code `code`, if this minor defines it.
    pub fn from_code(code: u8) -> Option<FrameType> {
        FrameType::ALL.into_iter().find(|t| *t as u8 == code)
    }

    /// The wire code.
    pub fn code(self) -> u8 {
        self as u8
    }

    /// Whether this is a control frame: `Hello`, `HelloAck`, `Credit`, `Ping` or `Close`. Control
    /// frames carry `chan_seq = 0` and consume no credit (ADR-0307, revised 2026-10-19).
    pub fn is_control(self) -> bool {
        matches!(
            self,
            FrameType::Hello
                | FrameType::HelloAck
                | FrameType::Credit
                | FrameType::Ping
                | FrameType::Close
        )
    }
}

/// The header's `flags` (RFC-0031 §6.1). The raw value is kept, reserved bits included, so a
/// decoded frame re-encodes byte for byte; the reserved bits carry no meaning on receive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Flags(pub u16);

impl Flags {
    /// bit0: a `TRACE` TLV is present.
    pub const HAS_TRACE: u16 = 1 << 0;
    /// bit2: the frame is AEAD-sealed (ADR-0314).
    pub const ENCRYPTED: u16 = 1 << 2;
    /// bit3: the first frame of a sequence on this edge.
    pub const FIRST_OF_SEQ: u16 = 1 << 3;
    /// bit4: the AEAD key phase, which flips on re-keying.
    pub const KEY_PHASE: u16 = 1 << 4;
    /// bit1 (formerly `RETRANSMIT`) and bits 5–15: reserved, zero on send.
    pub const RESERVED: u16 =
        !(Self::HAS_TRACE | Self::ENCRYPTED | Self::FIRST_OF_SEQ | Self::KEY_PHASE);

    pub fn has_trace(self) -> bool {
        self.0 & Self::HAS_TRACE != 0
    }

    pub fn encrypted(self) -> bool {
        self.0 & Self::ENCRYPTED != 0
    }

    pub fn first_of_seq(self) -> bool {
        self.0 & Self::FIRST_OF_SEQ != 0
    }

    /// The key phase: `false` for phase 0, `true` for phase 1.
    pub fn key_phase(self) -> bool {
        self.0 & Self::KEY_PHASE != 0
    }
}

/// The fields of the base header a sender chooses. The lengths and CRCs follow from the
/// extension and payload when the frame is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameSpec {
    pub wire_minor: u8,
    pub ftype: FrameType,
    pub flags: Flags,
    /// `InstanceId`; 0 for blob-plane frames.
    pub instance: InstanceId,
    pub epoch: PlanEpoch,
    /// Per `ActivationFrame` semantics; on a sealed control frame, `ctl_seq`; 0 where unused.
    pub step: u64,
    /// Data frames: the per-channel sequence number, from 1. Control frames: 0.
    pub chan_seq: u64,
}

/// A decoded base header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Header {
    pub spec: FrameSpec,
    /// Bytes of extension, padding included; a multiple of 64.
    pub ext_len: u32,
    pub payload_len: u32,
    /// CRC32C of the payload; 0 when there is none.
    pub payload_crc: u32,
    /// CRC32C of header bytes 0..60 followed by the extension.
    pub header_crc: u32,
}

impl Header {
    /// The whole frame's length in bytes.
    pub fn frame_len(&self) -> u64 {
        HEADER_LEN as u64 + u64::from(self.ext_len) + u64::from(self.payload_len)
    }

    /// The sealed control frame's `ctl_seq`, carried in `step`; `None` on any other frame.
    pub fn ctl_seq(&self) -> Option<u64> {
        (self.spec.ftype.is_control() && self.spec.flags.encrypted()).then_some(self.spec.step)
    }
}

/// One frame, borrowing the buffer it was decoded from: the header, the extension (padding
/// included) and the payload. On a sealed frame, the extension past its first 16 bytes and the
/// payload are ciphertext, the payload ending in the 16-byte tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame<'a> {
    pub header: Header,
    pub ext: &'a [u8],
    pub payload: &'a [u8],
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

/// The bytes a frame with this extension and payload length takes, extension padding included.
pub fn encoded_len(ext_len: usize, payload_len: usize) -> usize {
    HEADER_LEN + ext_len.next_multiple_of(ALIGN) + payload_len
}

impl<'a> Frame<'a> {
    /// Decodes the frame at the start of `buf`, which may hold more bytes after it.
    ///
    /// The checks run in the order RFC-0031 §6.1 lists for a receiver, as far as they need no
    /// channel state: `magic`, the lengths, `header_crc`, `payload_crc`; then the frame type,
    /// the extension alignment, and the control-frame rules: `chan_seq` is 0, and `step` holds
    /// `ctl_seq` (from 1) on a sealed control frame and is 0 on an unsealed one. A sealed frame's
    /// payload holds at least the AEAD tag. The negotiated minor, the channel key, `chan_seq`
    /// continuity and the slot size are the channel's to check.
    pub fn decode(buf: &'a [u8]) -> Result<Frame<'a>, FrameError> {
        Frame::decode_checked(buf, true)
    }

    /// [`Frame::decode`] without the payload CRC, for a channel that must check its key and
    /// `chan_seq` between the header's checks and the payload's (RFC-0031 §6.1, §7.2): a frame for
    /// another instance or epoch is dropped and never NACKed, whatever its payload holds. Call
    /// [`Frame::verify_payload`] before trusting the payload.
    pub fn decode_header_first(buf: &'a [u8]) -> Result<Frame<'a>, FrameError> {
        Frame::decode_checked(buf, false)
    }

    /// Checks `payload_crc` against the payload.
    pub fn verify_payload(&self) -> Result<(), FrameError> {
        let computed = if self.payload.is_empty() {
            0
        } else {
            crc32c(self.payload)
        };
        if computed == self.header.payload_crc {
            Ok(())
        } else {
            Err(FrameError::PayloadCrc {
                stated: self.header.payload_crc,
                computed,
            })
        }
    }

    fn decode_checked(buf: &'a [u8], check_payload: bool) -> Result<Frame<'a>, FrameError> {
        if buf.len() < HEADER_LEN {
            return Err(FrameError::Truncated {
                needed: HEADER_LEN as u64,
                available: buf.len(),
            });
        }
        let magic: [u8; 4] = buf[0..4].try_into().expect("4 bytes");
        if magic != MAGIC {
            return Err(FrameError::BadMagic(magic));
        }
        let ext_len = u32_at(buf, 48);
        let payload_len = u32_at(buf, 52);
        let payload_crc = u32_at(buf, 56);
        let header_crc = u32_at(buf, 60);
        let total = HEADER_LEN as u64 + u64::from(ext_len) + u64::from(payload_len);
        if (buf.len() as u64) < total {
            return Err(FrameError::Truncated {
                needed: total,
                available: buf.len(),
            });
        }
        let ext_end = HEADER_LEN + ext_len as usize;
        let ext = &buf[HEADER_LEN..ext_end];
        let payload = &buf[ext_end..ext_end + payload_len as usize];
        let computed = crc32c_append(crc32c(&buf[..60]), ext);
        if computed != header_crc {
            return Err(FrameError::HeaderCrc {
                stated: header_crc,
                computed,
            });
        }
        if check_payload {
            let computed = if payload.is_empty() {
                0
            } else {
                crc32c(payload)
            };
            if computed != payload_crc {
                return Err(FrameError::PayloadCrc {
                    stated: payload_crc,
                    computed,
                });
            }
        }
        let ftype = FrameType::from_code(buf[5]).ok_or(FrameError::UnknownFrameType(buf[5]))?;
        if !(ext_len as usize).is_multiple_of(ALIGN) {
            return Err(FrameError::ExtLenUnaligned(ext_len));
        }
        let spec = FrameSpec {
            wire_minor: buf[4],
            ftype,
            flags: Flags(u16_at(buf, 6)),
            instance: InstanceId(u64_at(buf, 8)),
            epoch: PlanEpoch {
                term: u64_at(buf, 16),
                seq: u64_at(buf, 24),
            },
            step: u64_at(buf, 32),
            chan_seq: u64_at(buf, 40),
        };
        check_spec(&spec, payload.len())?;
        Ok(Frame {
            header: Header {
                spec,
                ext_len,
                payload_len,
                payload_crc,
                header_crc,
            },
            ext,
            payload,
        })
    }

    /// Re-encodes this frame into `out`, byte for byte as it was decoded.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, FrameError> {
        encode(&self.header.spec, self.ext, self.payload, out)
    }
}

/// The rules on a frame's header fields that hold without channel state.
fn check_spec(spec: &FrameSpec, payload_len: usize) -> Result<(), FrameError> {
    let sealed = spec.flags.encrypted();
    if spec.ftype.is_control() {
        if spec.chan_seq != 0 {
            return Err(FrameError::ControlChanSeq {
                ftype: spec.ftype,
                chan_seq: spec.chan_seq,
            });
        }
        // `ctl_seq` starts at 1 under each key; an unsealed control frame leaves `step` at 0.
        if sealed == (spec.step == 0) {
            return Err(FrameError::ControlStep {
                ftype: spec.ftype,
                step: spec.step,
                sealed,
            });
        }
    }
    if sealed && payload_len < AEAD_TAG_LEN {
        return Err(FrameError::SealedWithoutTag { payload_len });
    }
    Ok(())
}

/// Encodes a frame into the start of `out` and returns its length. The extension is padded with
/// zeros to a multiple of 64; give it unpadded or already padded. `payload_crc` and
/// `header_crc` are computed here. The same rules as [`Frame::decode`] apply, so this never
/// writes a frame the decoder would refuse.
pub fn encode(
    spec: &FrameSpec,
    ext: &[u8],
    payload: &[u8],
    out: &mut [u8],
) -> Result<usize, FrameError> {
    let ext_len = ext.len().next_multiple_of(ALIGN);
    let total = check_frame(spec, ext_len, ext.len(), payload.len(), out.len())?;
    out[HEADER_LEN..HEADER_LEN + ext.len()].copy_from_slice(ext);
    out[HEADER_LEN + ext.len()..HEADER_LEN + ext_len].fill(0);
    out[HEADER_LEN + ext_len..total].copy_from_slice(payload);
    finish(spec, ext_len, &mut out[..total]);
    Ok(total)
}

/// [`encode`] for a frame whose extension the caller has already written at
/// `out[64..64 + ext_len]`, padding included, as a typed codec does to build it in place. Writes
/// the header and the payload and computes both CRCs; the extension bytes are not copied.
pub fn encode_in_place(
    spec: &FrameSpec,
    ext_len: usize,
    payload: &[u8],
    out: &mut [u8],
) -> Result<usize, FrameError> {
    let total = check_frame(spec, ext_len, ext_len, payload.len(), out.len())?;
    if !ext_len.is_multiple_of(ALIGN) {
        return Err(FrameError::ExtLenUnaligned(ext_len as u32));
    }
    out[HEADER_LEN + ext_len..total].copy_from_slice(payload);
    finish(spec, ext_len, &mut out[..total]);
    Ok(total)
}

/// [`encode`] for a frame whose extension and payload the caller has both written in place, at
/// `out[64..64 + ext_len]` (padding included) and the `payload_len` bytes after it, as a typed
/// codec does when it builds its payload too. Writes the header and computes both CRCs.
pub fn encode_assembled(
    spec: &FrameSpec,
    ext_len: usize,
    payload_len: usize,
    out: &mut [u8],
) -> Result<usize, FrameError> {
    let total = check_frame(spec, ext_len, ext_len, payload_len, out.len())?;
    if !ext_len.is_multiple_of(ALIGN) {
        return Err(FrameError::ExtLenUnaligned(ext_len as u32));
    }
    finish(spec, ext_len, &mut out[..total]);
    Ok(total)
}

/// The checks [`encode`] makes before writing: the lengths fit their fields, the header fields
/// keep the decoder's rules, and `out` holds the frame. Returns the frame's length.
fn check_frame(
    spec: &FrameSpec,
    ext_len: usize,
    ext_given: usize,
    payload_len: usize,
    available: usize,
) -> Result<usize, FrameError> {
    u32::try_from(ext_len).map_err(|_| FrameError::TooLong {
        what: "extension",
        len: ext_given,
    })?;
    u32::try_from(payload_len).map_err(|_| FrameError::TooLong {
        what: "payload",
        len: payload_len,
    })?;
    check_spec(spec, payload_len)?;
    let total = HEADER_LEN + ext_len + payload_len;
    if available < total {
        return Err(FrameError::BufferTooSmall {
            needed: total,
            available,
        });
    }
    Ok(total)
}

/// Writes the header around an extension and a payload already in `out`, then both CRCs. `out`
/// is exactly the frame; [`check_frame`] has passed.
fn finish(spec: &FrameSpec, ext_len: usize, out: &mut [u8]) {
    let ext_end = HEADER_LEN + ext_len;
    let payload_len = out.len() - ext_end;
    let payload_crc = if payload_len == 0 {
        0
    } else {
        crc32c(&out[ext_end..])
    };
    let h = &mut out[..HEADER_LEN];
    h[0..4].copy_from_slice(&MAGIC);
    h[4] = spec.wire_minor;
    h[5] = spec.ftype.code();
    h[6..8].copy_from_slice(&spec.flags.0.to_le_bytes());
    h[8..16].copy_from_slice(&spec.instance.0.to_le_bytes());
    h[16..24].copy_from_slice(&spec.epoch.term.to_le_bytes());
    h[24..32].copy_from_slice(&spec.epoch.seq.to_le_bytes());
    h[32..40].copy_from_slice(&spec.step.to_le_bytes());
    h[40..48].copy_from_slice(&spec.chan_seq.to_le_bytes());
    h[48..52].copy_from_slice(&(ext_len as u32).to_le_bytes());
    h[52..56].copy_from_slice(&(payload_len as u32).to_le_bytes());
    h[56..60].copy_from_slice(&payload_crc.to_le_bytes());
    let header_crc = crc32c_append(crc32c(&out[..60]), &out[HEADER_LEN..ext_end]);
    out[60..64].copy_from_slice(&header_crc.to_le_bytes());
}

/// The base header of a frame whose extension and payload sit elsewhere, as a zero-copy sender
/// sends them: the header, then `ext`, then `payload`, contiguous on the wire (RFC-0031 §8).
/// `ext` is the extension with its padding, a multiple of 64. Both CRCs are computed here and the
/// rules of [`encode`] apply, so the bytes are those [`encode`] would write for the same frame.
pub fn encode_header(
    spec: &FrameSpec,
    ext: &[u8],
    payload: &[u8],
) -> Result<[u8; HEADER_LEN], FrameError> {
    if !ext.len().is_multiple_of(ALIGN) {
        return Err(FrameError::ExtLenUnaligned(ext.len() as u32));
    }
    check_frame(spec, ext.len(), ext.len(), payload.len(), usize::MAX)?;
    let payload_crc = if payload.is_empty() {
        0
    } else {
        crc32c(payload)
    };
    let mut h = [0u8; HEADER_LEN];
    h[0..4].copy_from_slice(&MAGIC);
    h[4] = spec.wire_minor;
    h[5] = spec.ftype.code();
    h[6..8].copy_from_slice(&spec.flags.0.to_le_bytes());
    h[8..16].copy_from_slice(&spec.instance.0.to_le_bytes());
    h[16..24].copy_from_slice(&spec.epoch.term.to_le_bytes());
    h[24..32].copy_from_slice(&spec.epoch.seq.to_le_bytes());
    h[32..40].copy_from_slice(&spec.step.to_le_bytes());
    h[40..48].copy_from_slice(&spec.chan_seq.to_le_bytes());
    h[48..52].copy_from_slice(&(ext.len() as u32).to_le_bytes());
    h[52..56].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    h[56..60].copy_from_slice(&payload_crc.to_le_bytes());
    let header_crc = crc32c_append(crc32c(&h[..60]), ext);
    h[60..64].copy_from_slice(&header_crc.to_le_bytes());
    Ok(h)
}

/// [`encode`] into a new vector of exactly the frame's length.
pub fn encode_to_vec(spec: &FrameSpec, ext: &[u8], payload: &[u8]) -> Result<Vec<u8>, FrameError> {
    let mut out = vec![0; encoded_len(ext.len(), payload.len())];
    let len = encode(spec, ext, payload, &mut out)?;
    out.truncate(len);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(ftype: FrameType) -> FrameSpec {
        FrameSpec {
            wire_minor: WIRE_MINOR,
            ftype,
            flags: Flags::default(),
            instance: InstanceId(7),
            epoch: PlanEpoch { term: 3, seq: 9 },
            step: 0,
            chan_seq: if ftype.is_control() { 0 } else { 1 },
        }
    }

    #[test]
    fn frame_type_codes_follow_the_table() {
        for (index, ftype) in FrameType::ALL.into_iter().enumerate() {
            assert_eq!(ftype.code() as usize, index + 1);
            assert_eq!(FrameType::from_code(ftype.code()), Some(ftype));
        }
        assert_eq!(FrameType::from_code(0), None);
        assert_eq!(FrameType::from_code(0x13), None);
        let control: Vec<FrameType> = FrameType::ALL
            .into_iter()
            .filter(|t| t.is_control())
            .collect();
        assert_eq!(
            control,
            [
                FrameType::Credit,
                FrameType::Hello,
                FrameType::HelloAck,
                FrameType::Close,
                FrameType::Ping
            ]
        );
    }

    #[test]
    fn flags_name_their_bits_and_bit1_is_reserved() {
        let flags =
            Flags(Flags::HAS_TRACE | Flags::ENCRYPTED | Flags::FIRST_OF_SEQ | Flags::KEY_PHASE);
        assert!(
            flags.has_trace() && flags.encrypted() && flags.first_of_seq() && flags.key_phase()
        );
        assert_eq!(Flags::RESERVED, 0xFFE2);
        assert_eq!(Flags::RESERVED & (1 << 1), 1 << 1);
    }

    #[test]
    fn the_extension_is_padded_to_64_bytes_and_counted_in_the_header_crc() {
        let bytes = encode_to_vec(&spec(FrameType::Activation), &[1, 2, 3], &[9; 10]).unwrap();
        assert_eq!(bytes.len(), 64 + 64 + 10);
        assert_eq!(u32_at(&bytes, 48), 64);
        assert!(bytes[64 + 3..128].iter().all(|b| *b == 0));
        let frame = Frame::decode(&bytes).unwrap();
        assert_eq!(frame.ext.len(), 64);
        assert_eq!(&frame.ext[..3], &[1, 2, 3]);
        assert_eq!(frame.payload, &[9; 10]);
        let mut again = vec![0; bytes.len()];
        assert_eq!(frame.encode(&mut again), Ok(bytes.len()));
        assert_eq!(again, bytes);
        // `header_crc` covers bytes 0..60 and the extension, its padding included.
        let mut covered = bytes[..60].to_vec();
        covered.extend_from_slice(&bytes[64..128]);
        assert_eq!(u32_at(&bytes, 60), crc32c(&covered));
        let mut corrupt = bytes.clone();
        corrupt[64 + 10] ^= 1;
        assert!(matches!(
            Frame::decode(&corrupt),
            Err(FrameError::HeaderCrc { .. })
        ));
    }

    #[test]
    fn an_unaligned_extension_length_is_refused() {
        // A data frame with a 60-byte extension and a correct `header_crc`: everything holds but
        // the alignment, which `encode` would otherwise re-pad to different bytes.
        let mut bytes = encode_to_vec(&spec(FrameType::Activation), &[1, 2, 3], &[]).unwrap();
        bytes.truncate(HEADER_LEN + 60);
        bytes[48..52].copy_from_slice(&60u32.to_le_bytes());
        let header_crc = crc32c_append(crc32c(&bytes[..60]), &bytes[HEADER_LEN..]);
        bytes[60..64].copy_from_slice(&header_crc.to_le_bytes());
        assert_eq!(Frame::decode(&bytes), Err(FrameError::ExtLenUnaligned(60)));
    }

    #[test]
    fn only_a_sealed_control_frame_has_a_ctl_seq() {
        let unsealed = encode_to_vec(&spec(FrameType::Ping), &[0; 16], &[]).unwrap();
        assert_eq!(Frame::decode(&unsealed).unwrap().header.ctl_seq(), None);
        let data = encode_to_vec(&spec(FrameType::Activation), &[], &[1]).unwrap();
        assert_eq!(Frame::decode(&data).unwrap().header.ctl_seq(), None);
        let mut sealed = spec(FrameType::Ping);
        sealed.flags = Flags(Flags::ENCRYPTED);
        sealed.step = 3;
        let bytes = encode_to_vec(&sealed, &[0; 64], &[0; 16]).unwrap();
        assert_eq!(Frame::decode(&bytes).unwrap().header.ctl_seq(), Some(3));
    }

    #[test]
    fn an_empty_payload_has_a_zero_crc() {
        let bytes = encode_to_vec(&spec(FrameType::Ping), &[0; 16], &[]).unwrap();
        assert_eq!(u32_at(&bytes, 56), 0);
        assert!(Frame::decode(&bytes).is_ok());
    }

    #[test]
    fn control_frames_carry_chan_seq_zero_and_ctl_seq_only_when_sealed() {
        let mut s = spec(FrameType::Credit);
        s.chan_seq = 1;
        assert!(matches!(
            encode_to_vec(&s, &[0; 64], &[]),
            Err(FrameError::ControlChanSeq { chan_seq: 1, .. })
        ));
        let mut s = spec(FrameType::Credit);
        s.step = 4;
        assert!(matches!(
            encode_to_vec(&s, &[0; 64], &[]),
            Err(FrameError::ControlStep {
                sealed: false,
                step: 4,
                ..
            })
        ));
        let mut s = spec(FrameType::Credit);
        s.flags = Flags(Flags::ENCRYPTED);
        assert!(matches!(
            encode_to_vec(&s, &[0; 64], &[0; 16]),
            Err(FrameError::ControlStep {
                sealed: true,
                step: 0,
                ..
            })
        ));
        s.step = 1;
        let bytes = encode_to_vec(&s, &[0; 64], &[0; 16]).unwrap();
        assert_eq!(Frame::decode(&bytes).unwrap().header.ctl_seq(), Some(1));
        assert!(matches!(
            encode_to_vec(&s, &[0; 64], &[0; 15]),
            Err(FrameError::SealedWithoutTag { payload_len: 15 })
        ));
    }

    #[test]
    fn a_short_buffer_or_bad_magic_is_refused() {
        let bytes = encode_to_vec(&spec(FrameType::Activation), &[], &[1; 8]).unwrap();
        assert!(matches!(
            Frame::decode(&bytes[..63]),
            Err(FrameError::Truncated {
                needed: 64,
                available: 63
            })
        ));
        assert!(matches!(
            Frame::decode(&bytes[..bytes.len() - 1]),
            Err(FrameError::Truncated { .. })
        ));
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert_eq!(Frame::decode(&bad), Err(FrameError::BadMagic(*b"XYNF")));
        let mut out = vec![0; 10];
        assert!(matches!(
            encode(&spec(FrameType::Activation), &[], &[1; 8], &mut out),
            Err(FrameError::BufferTooSmall {
                needed: 72,
                available: 10
            })
        ));
    }

    #[test]
    fn a_frame_followed_by_more_bytes_decodes_alone() {
        let mut bytes = encode_to_vec(&spec(FrameType::Activation), &[], &[1; 8]).unwrap();
        let len = bytes.len();
        bytes.extend_from_slice(&[0xAA; 20]);
        let frame = Frame::decode(&bytes).unwrap();
        assert_eq!(frame.header.frame_len(), len as u64);
    }
}
