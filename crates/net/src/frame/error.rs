// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why a frame could not be decoded or encoded.

use std::fmt;

use syntropy_core::ErrorClass;
use syntropy_core::plan::WireDType;

use super::header::FrameType;

/// Why a frame was refused.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameError {
    /// Fewer bytes than the header, or than the frame its header declares.
    Truncated { needed: u64, available: usize },
    /// The first four bytes are not "SYNF".
    BadMagic([u8; 4]),
    /// `header_crc` does not match bytes 0..60 and the extension.
    HeaderCrc { stated: u32, computed: u32 },
    /// `payload_crc` does not match the payload, or is not 0 for an empty one.
    PayloadCrc { stated: u32, computed: u32 },
    /// `ftype` is not a type of this wire minor.
    UnknownFrameType(u8),
    /// `ext_len` is not a multiple of 64.
    ExtLenUnaligned(u32),
    /// A control frame whose `chan_seq` is not 0.
    ControlChanSeq { ftype: FrameType, chan_seq: u64 },
    /// A sealed control frame without a `ctl_seq` (from 1) in `step`, or an unsealed control
    /// frame whose `step` is not 0.
    ControlStep {
        ftype: FrameType,
        step: u64,
        sealed: bool,
    },
    /// A sealed frame whose payload is shorter than the AEAD tag.
    SealedWithoutTag { payload_len: usize },
    /// A typed message was asked of a sealed frame, which must be opened first.
    Sealed(FrameType),
    /// A typed control message was asked of a frame of another type.
    NotControl(FrameType),
    /// A typed `Activation` was asked of a frame of another type.
    NotActivation(FrameType),
    /// A typed message of one frame type was asked of a frame of another.
    WrongType {
        expected: FrameType,
        found: FrameType,
    },
    /// A control frame's extension is shorter than its fields.
    ExtTooShort {
        ftype: FrameType,
        len: usize,
        needed: usize,
    },
    /// A control message field holds a value the format does not define.
    BadField {
        ftype: FrameType,
        field: &'static str,
        value: u64,
    },
    /// A field that disagrees with what the rest of the frame gives it: a span's `row_offset`
    /// that the spans before it do not reach, or a payload that is not `n_rows` strides long.
    /// For a result assembled from parts, `value` is a part's and `expected` the first part's.
    Inconsistent {
        ftype: FrameType,
        field: &'static str,
        value: u64,
        expected: u64,
    },
    /// An `Activation` whose dtype is not its channel's `ChannelSpec::wire` (RFC-0031 §6.4).
    WireDTypeMismatch {
        expected: WireDType,
        found: WireDType,
    },
    /// A wire dtype whose container this build does not carry: `F16`, `Q8B32` and
    /// `F8E4M3B32`, which come with the `WIRE_COMPRESS` containers (DIST-E8-T11).
    UnsupportedDType(WireDType),
    /// A `Hello` or `HelloAck` credential whose length does not fit its `cred_kind`.
    CredentialLength { kind: u16, len: usize },
    /// A `Close` reason that is not UTF-8 or is longer than 256 bytes.
    BadReason { len: usize },
    /// A payload on a frame type that has none (`Credit`, `Ping`).
    UnexpectedPayload { ftype: FrameType, len: usize },
    /// An extension or payload too long for its 32-bit length field.
    TooLong { what: &'static str, len: usize },
    /// The output buffer is shorter than the frame.
    BufferTooSmall { needed: usize, available: usize },
    /// A frame whose type needs a capability (RFC-0031 §10) that the channel did not negotiate:
    /// the encoder refuses it, so a peer that did not advertise the capability never receives
    /// the frame.
    CapabilityNotNegotiated { ftype: FrameType, capability: u64 },
    /// A `Token` entry's tap block of `len` bytes was to be converted to `core`'s `LogitTap`. The
    /// block's codec is DIST-E8-T06's (RFC-0031 §6.13), so until it exists a `TokenEntryView`
    /// converts to `core`'s `TokenEntry` only without a tap.
    TapNotDecoded { len: usize },
    /// `core`'s `LogitTap` of `rows` rows was to be encoded as a `Token` entry's tap block, which
    /// has no codec before DIST-E8-T06.
    TapNotEncoded { rows: usize },
    /// A `core` `LogprobBlock` with `k = 0`, the chosen tokens' logprobs alone, was to be encoded.
    /// RFC-0031 §6.5's `logprob_k = 0` means no logprobs, so the entry has no form for it until
    /// OQ-NET-35 settles.
    ChosenLogprobsOnly { n_tokens: usize },
}

impl FrameError {
    /// The same error, naming `ftype` as the frame it was found in: the TLV reader `Activation`
    /// and `Token` frames share names its own frame otherwise.
    pub(super) fn in_frame(self, ftype: FrameType) -> FrameError {
        match self {
            FrameError::BadField { field, value, .. } => FrameError::BadField {
                ftype,
                field,
                value,
            },
            FrameError::Inconsistent {
                field,
                value,
                expected,
                ..
            } => FrameError::Inconsistent {
                ftype,
                field,
                value,
                expected,
            },
            FrameError::ExtTooShort { len, needed, .. } => {
                FrameError::ExtTooShort { ftype, len, needed }
            }
            other => other,
        }
    }

    /// The RFC-0001 §5.12 class. A refused frame is never `Internal`, as RFC-0060 §10.2 requires
    /// of every decoder of untrusted bytes: damaged or malformed bytes are `Integrity`
    /// (RFC-0031 §7.2 drops the frame and NACKs it). A frame type this minor does not define, an
    /// `Activation` in another dtype than its channel's, and a dtype container this build does
    /// not carry are `Unsupported` (RFC-0031 §12, revised 2026-10-22); a resend would carry the
    /// same bytes, so no NACK can repair them. So is a tap block this build has no codec for
    /// ([`FrameError::TapNotDecoded`]), and a logprob block the entry has no form for
    /// ([`FrameError::ChosenLogprobsOnly`]). The first two are sent in `Close` as the code
    /// [`FrameError::code`] gives. The encoder's refusal of a frame type whose capability the
    /// channel did not negotiate ([`FrameError::CapabilityNotNegotiated`]) is `Unsupported` too,
    /// with no code: the frame is never written, so nothing travels. The encoder's other
    /// failures, a buffer too small or a length too long for its field, are `Internal`.
    pub fn class(&self) -> ErrorClass {
        match self {
            FrameError::UnknownFrameType(_)
            | FrameError::WireDTypeMismatch { .. }
            | FrameError::UnsupportedDType(_)
            | FrameError::CapabilityNotNegotiated { .. }
            | FrameError::TapNotDecoded { .. }
            | FrameError::TapNotEncoded { .. }
            | FrameError::ChosenLogprobsOnly { .. } => ErrorClass::Unsupported,
            FrameError::TooLong { .. } | FrameError::BufferTooSmall { .. } => ErrorClass::Internal,
            _ => ErrorClass::Integrity,
        }
    }

    /// The same refusal, naming `to` wherever it names `from`: a `StepDescriptor` is read with
    /// the `Activation` codec, and its refusals name the `StepDescriptor`.
    pub fn retyped(self, from: FrameType, to: FrameType) -> FrameError {
        let swap = |ftype: FrameType| if ftype == from { to } else { ftype };
        match self {
            FrameError::Sealed(ftype) => FrameError::Sealed(swap(ftype)),
            FrameError::ExtTooShort { ftype, len, needed } => FrameError::ExtTooShort {
                ftype: swap(ftype),
                len,
                needed,
            },
            FrameError::BadField {
                ftype,
                field,
                value,
            } => FrameError::BadField {
                ftype: swap(ftype),
                field,
                value,
            },
            FrameError::Inconsistent {
                ftype,
                field,
                value,
                expected,
            } => FrameError::Inconsistent {
                ftype: swap(ftype),
                field,
                value,
                expected,
            },
            other => other,
        }
    }

    /// The RFC-0031 §12 code a receiver sends in `Close`, where one exists.
    pub fn code(&self) -> Option<u16> {
        match self {
            FrameError::UnknownFrameType(_) => Some(super::control::code::UNKNOWN_FRAME_TYPE),
            FrameError::TooLong { .. } => Some(super::control::code::FRAME_TOO_LARGE),
            FrameError::WireDTypeMismatch { .. } => Some(super::control::code::WIRE_DTYPE_MISMATCH),
            _ => None,
        }
    }
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameError::Truncated { needed, available } => {
                write!(
                    f,
                    "truncated frame: {needed} bytes needed, {available} available"
                )
            }
            FrameError::BadMagic(magic) => write!(f, "bad magic {magic:02x?}"),
            FrameError::HeaderCrc { stated, computed } => {
                write!(
                    f,
                    "header CRC {stated:#010x} does not match {computed:#010x}"
                )
            }
            FrameError::PayloadCrc { stated, computed } => {
                write!(
                    f,
                    "payload CRC {stated:#010x} does not match {computed:#010x}"
                )
            }
            FrameError::UnknownFrameType(code) => write!(f, "unknown frame type {code:#04x}"),
            FrameError::ExtLenUnaligned(len) => {
                write!(f, "extension length {len} is not a multiple of 64")
            }
            FrameError::ControlChanSeq { ftype, chan_seq } => {
                write!(
                    f,
                    "{ftype:?} is a control frame, but chan_seq is {chan_seq}, not 0"
                )
            }
            FrameError::ControlStep {
                ftype,
                step,
                sealed: true,
            } => write!(
                f,
                "sealed {ftype:?} needs ctl_seq from 1 in step, found {step}"
            ),
            FrameError::ControlStep {
                ftype,
                step,
                sealed: false,
            } => write!(f, "unsealed {ftype:?} has step {step}, not 0"),
            FrameError::SealedWithoutTag { payload_len } => {
                write!(
                    f,
                    "sealed frame's payload of {payload_len} bytes has no room for its tag"
                )
            }
            FrameError::Sealed(ftype) => write!(f, "{ftype:?} is sealed; open it first"),
            FrameError::NotControl(ftype) => write!(f, "{ftype:?} is not a control frame"),
            FrameError::NotActivation(ftype) => write!(f, "{ftype:?} is not an Activation frame"),
            FrameError::WrongType { expected, found } => {
                write!(f, "{found:?} is not a {expected:?} frame")
            }
            FrameError::ExtTooShort { ftype, len, needed } => {
                write!(
                    f,
                    "{ftype:?} extension of {len} bytes; its fields need {needed}"
                )
            }
            FrameError::BadField {
                ftype,
                field,
                value,
            } => write!(
                f,
                "{ftype:?} field `{field}` has the undefined value {value}"
            ),
            FrameError::Inconsistent {
                ftype,
                field,
                value,
                expected,
            } => write!(
                f,
                "{ftype:?} field `{field}` is {value}; the rest of the frame gives {expected}"
            ),
            FrameError::WireDTypeMismatch { expected, found } => write!(
                f,
                "frame dtype {found:?} is not the channel's wire dtype {expected:?}"
            ),
            FrameError::UnsupportedDType(dtype) => {
                write!(f, "wire dtype {dtype:?} is not carried by this build")
            }
            FrameError::CredentialLength { kind, len } => {
                write!(f, "credential of {len} bytes does not fit cred_kind {kind}")
            }
            FrameError::BadReason { len } => {
                write!(
                    f,
                    "Close reason of {len} bytes is not UTF-8 of at most 256 bytes"
                )
            }
            FrameError::UnexpectedPayload { ftype, len } => {
                write!(f, "{ftype:?} has no payload, but {len} bytes came")
            }
            FrameError::TooLong { what, len } => {
                write!(f, "{what} of {len} bytes is too long for its length field")
            }
            FrameError::BufferTooSmall { needed, available } => {
                write!(
                    f,
                    "output buffer of {available} bytes; the frame needs {needed}"
                )
            }
            FrameError::CapabilityNotNegotiated { ftype, capability } => write!(
                f,
                "{ftype:?} needs capability bit {}, which the channel did not negotiate",
                capability.trailing_zeros()
            ),
            FrameError::TapNotEncoded { rows } => write!(
                f,
                "a {rows}-row LogitTap has no Token tap-block codec in this build"
            ),
            FrameError::TapNotDecoded { len } => write!(
                f,
                "a Token entry's {len}-byte tap block has no LogitTap codec in this build"
            ),
            FrameError::ChosenLogprobsOnly { n_tokens } => write!(
                f,
                "a Token entry cannot carry {n_tokens} chosen-token logprobs without alternatives"
            ),
        }
    }
}

impl std::error::Error for FrameError {}
