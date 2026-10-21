// SPDX-License-Identifier: MIT OR Apache-2.0

//! The control frames: `Hello`, `HelloAck`, `Close` (RFC-0031 §6.9), `Credit` (§6.7) and `Ping`
//! (§6.2). Control frames carry `chan_seq = 0`, consume no credit and are never NACKed.
//!
//! [`Control`] is the plaintext message. On an encrypted channel the AEAD layer seals the
//! extension past its first 16 bytes and the payload, and sets `ctl_seq` in `step`; a frame is
//! opened before [`Frame::control`] reads it. Reserved fields are written as zero and ignored on
//! receive, and so are extension bytes past the fields this minor defines (RFC-0031 §10).

use syntropy_core::plan::WireDType;
use syntropy_core::{InstanceId, PlanEpoch, StageIx};

use super::error::FrameError;
use super::header::{self, ALIGN, Flags, Frame, FrameSpec, FrameType};

/// The RFC-0031 §12 codes carried in `HelloAck::status`, `Close::code` and `Error::code`.
pub mod code {
    /// `HelloAck::status` for an accepted channel.
    pub const OK: u16 = 0;
    pub const PROTO_MAJOR_MISMATCH: u16 = 0x0201;
    pub const CREDENTIAL_INVALID: u16 = 0x0202;
    pub const NOT_A_NEIGHBOUR: u16 = 0x0203;
    pub const UNKNOWN_FRAME_TYPE: u16 = 0x0204;
    pub const FRAME_TOO_LARGE: u16 = 0x0205;
    pub const WIRE_DTYPE_MISMATCH: u16 = 0x0206;
    pub const CRC_STORM: u16 = 0x0207;
    pub const ACK_TIMEOUT: u16 = 0x0208;
    pub const QP_ERROR: u16 = 0x0209;
    pub const CONNECT_TIMEOUT: u16 = 0x020A;
    pub const PEER_RESET: u16 = 0x020B;
    pub const EDGE_DEMOTED: u16 = 0x020C;
    pub const RDMA_NOT_ENABLED: u16 = 0x020D;
    pub const RDMA_OS_BUILD_MISMATCH: u16 = 0x020E;
    pub const SHM_USER_MISMATCH: u16 = 0x020F;
    /// `BlobNotFound`, `BlobBusy` and `BlobThrottled` share this code.
    pub const BLOB_UNAVAILABLE: u16 = 0x0210;
    pub const EPOCH_NOT_SERVED: u16 = 0x0211;
    pub const PROTO_RANGE_DISJOINT: u16 = 0x0212;
    pub const QP_BUDGET_EXHAUSTED: u16 = 0x0213;
}

/// The largest control extension: a `Credit` with 16 NACKs, 32 + 128 bytes, padded to 192.
pub const MAX_CONTROL_EXT: usize = 192;

/// The most NACKs one `Credit` carries.
pub const MAX_NACKS: usize = 16;

/// The longest `Close` reason, in bytes.
pub const MAX_REASON: usize = 256;

/// What a `Hello` opens: one of RFC-0031 §2's `ChannelPurpose`s, coded 1..7, or a blob stream,
/// 8 (RFC-0031 §6.9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Purpose {
    Activations = 1,
    Tokens = 2,
    Sequence = 3,
    Collective = 4,
    Replay = 5,
    Bulk = 6,
    Tap = 7,
    BlobStream = 8,
}

impl Purpose {
    const ALL: [Purpose; 8] = [
        Purpose::Activations,
        Purpose::Tokens,
        Purpose::Sequence,
        Purpose::Collective,
        Purpose::Replay,
        Purpose::Bulk,
        Purpose::Tap,
        Purpose::BlobStream,
    ];

    fn from_code(code: u8) -> Option<Purpose> {
        Purpose::ALL.into_iter().find(|p| *p as u8 == code)
    }
}

/// Which credential a `Hello` or `HelloAck` carries (RFC-0034 §5, §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum CredKind {
    /// 168 bytes.
    DataPlaneCredential = 1,
    /// 192 bytes; blob streams use it.
    AdmissionCert = 2,
}

impl CredKind {
    fn from_code(code: u16) -> Option<CredKind> {
        match code {
            1 => Some(CredKind::DataPlaneCredential),
            2 => Some(CredKind::AdmissionCert),
            _ => None,
        }
    }

    /// The credential's length in bytes.
    pub fn credential_len(self) -> usize {
        match self {
            CredKind::DataPlaneCredential => 168,
            CredKind::AdmissionCert => 192,
        }
    }
}

/// The `Hello` extension: the first frame on every channel and blob stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Hello {
    pub purpose: Purpose,
    pub wire_minor_max: u8,
    pub proto_major: u16,
    /// The capability set of RFC-0031 §10.
    pub capabilities: u64,
    pub instance: InstanceId,
    pub epoch: PlanEpoch,
    pub from_stage: StageIx,
    pub to_stage: StageIx,
    pub lane: u8,
    /// The channel's wire dtype; `None` (code 0) where the channel carries no activations.
    pub wire_dtype: Option<WireDType>,
    /// The largest frame the sender of `Hello` accepts on the reverse direction.
    pub slot_size: u32,
    /// The initial credit granted to the peer.
    pub recv_slots: u16,
    pub cred_kind: CredKind,
    /// RDMA and shared memory: from pairing; otherwise 0.
    pub channel_token: u64,
    /// On a reconnect `Hello` (RFC-0031 §7.4): the first data `chan_seq` the sender will resend,
    /// at extension offset 64.
    pub resume_from: Option<u64>,
}

/// The `HelloAck` extension: the reply to `Hello`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HelloAck {
    /// [`code::OK`], or an RFC-0031 §12 code.
    pub status: u16,
    /// The minimum of both sides' minors.
    pub wire_minor: u8,
    /// The intersection of both capability sets.
    pub capabilities: u64,
    /// On a reconnect, `limit_seq − recv_slots` is the receiver's current `ack_seq`.
    pub limit_seq: u64,
    pub slot_size: u32,
    pub recv_slots: u16,
    pub cred_kind: CredKind,
}

/// The `Close` extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Close {
    /// An RFC-0031 §12 code.
    pub code: u16,
    /// The last data `chan_seq` the closer sent, at extension offset 8.
    pub last_seq: u64,
}

/// The NACK list of a `Credit`: at most 16 `chan_seq` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Nacks {
    len: u8,
    seqs: [u64; MAX_NACKS],
}

impl Nacks {
    /// An empty list.
    pub const EMPTY: Nacks = Nacks {
        len: 0,
        seqs: [0; MAX_NACKS],
    };

    /// A list of `seqs`; `None` if there are more than 16.
    pub fn new(seqs: &[u64]) -> Option<Nacks> {
        if seqs.len() > MAX_NACKS {
            return None;
        }
        let mut nacks = Nacks::EMPTY;
        nacks.seqs[..seqs.len()].copy_from_slice(seqs);
        nacks.len = seqs.len() as u8;
        Some(nacks)
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.seqs[..self.len as usize]
    }
}

/// The `Credit` extension (RFC-0031 §6.7). It carries no payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Credit {
    /// Every frame up to this `chan_seq` was received and validated.
    pub ack_seq: u64,
    /// The sender may send frames up to this `chan_seq`; `limit_seq − ack_seq` is the number of
    /// receive slots posted.
    pub limit_seq: u64,
    /// The `chan_seq` whose arrival triggered this credit; 0 if none.
    pub echo_seq: u64,
    /// Nanoseconds between that arrival and this credit being sent.
    pub echo_delay_ns: u32,
    /// flags bit0 `SACK`: the NACK list is selective.
    pub sack: bool,
    /// flags bit1 `RESET`: the receiver lost its state; resend from `ack_seq + 1`.
    pub reset: bool,
    /// The `chan_seq` values the receiver is missing or rejected.
    pub nacks: Nacks,
}

/// The `Ping` extension: 16 bytes, `origin_ns u64, reply u8`, then padding. No payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ping {
    pub origin_ns: u64,
    pub reply: bool,
}

/// A plaintext control message: its extension fields and its payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control<'a> {
    /// The payload is the sender's credential.
    Hello {
        hello: Hello,
        credential: &'a [u8],
    },
    /// The payload is the acceptor's own credential.
    HelloAck {
        ack: HelloAck,
        credential: &'a [u8],
    },
    /// The payload is a UTF-8 reason of at most 256 bytes.
    Close {
        close: Close,
        reason: &'a str,
    },
    Credit(Credit),
    Ping(Ping),
}

/// The header fields of an unsealed control frame; `step` and `chan_seq` are 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ControlSpec {
    pub wire_minor: u8,
    pub flags: Flags,
    pub instance: InstanceId,
    pub epoch: PlanEpoch,
}

fn put16(ext: &mut [u8], at: usize, v: u16) {
    ext[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn put32(ext: &mut [u8], at: usize, v: u32) {
    ext[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(ext: &mut [u8], at: usize, v: u64) {
    ext[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn get16(ext: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([ext[at], ext[at + 1]])
}

fn get32(ext: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(ext[at..at + 4].try_into().expect("4 bytes"))
}

fn get64(ext: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(ext[at..at + 8].try_into().expect("8 bytes"))
}

/// The wire code of a dtype (RFC-0031 §6.4); 0 for none.
fn wire_dtype_code(dtype: Option<WireDType>) -> Result<u8, FrameError> {
    Ok(match dtype {
        None => 0,
        Some(WireDType::F32) => 1,
        Some(WireDType::BF16) => 2,
        Some(WireDType::F16) => 3,
        Some(WireDType::Q8B32) => 4,
        Some(WireDType::F8E4M3B32) => 5,
        Some(_) => {
            return Err(FrameError::BadField {
                ftype: FrameType::Hello,
                field: "wire_dtype",
                value: u64::MAX,
            });
        }
    })
}

fn wire_dtype_from_code(code: u8) -> Result<Option<WireDType>, FrameError> {
    Ok(match code {
        0 => None,
        1 => Some(WireDType::F32),
        2 => Some(WireDType::BF16),
        3 => Some(WireDType::F16),
        4 => Some(WireDType::Q8B32),
        5 => Some(WireDType::F8E4M3B32),
        other => {
            return Err(FrameError::BadField {
                ftype: FrameType::Hello,
                field: "wire_dtype",
                value: other.into(),
            });
        }
    })
}

fn need(ftype: FrameType, ext: &[u8], needed: usize) -> Result<(), FrameError> {
    if ext.len() < needed {
        return Err(FrameError::ExtTooShort {
            ftype,
            len: ext.len(),
            needed,
        });
    }
    Ok(())
}

fn bad(ftype: FrameType, field: &'static str, value: impl Into<u64>) -> FrameError {
    FrameError::BadField {
        ftype,
        field,
        value: value.into(),
    }
}

impl<'a> Control<'a> {
    /// The frame type.
    pub fn ftype(&self) -> FrameType {
        match self {
            Control::Hello { .. } => FrameType::Hello,
            Control::HelloAck { .. } => FrameType::HelloAck,
            Control::Close { .. } => FrameType::Close,
            Control::Credit(_) => FrameType::Credit,
            Control::Ping(_) => FrameType::Ping,
        }
    }

    /// The payload: a credential, a `Close` reason, or nothing.
    pub fn payload(&self) -> &'a [u8] {
        match self {
            Control::Hello { credential, .. } | Control::HelloAck { credential, .. } => credential,
            Control::Close { reason, .. } => reason.as_bytes(),
            Control::Credit(_) | Control::Ping(_) => &[],
        }
    }

    /// Checks the rules the extension bytes cannot express: credential lengths, the reason's
    /// length, `limit_seq ≥ ack_seq`, and `resume_from` from 1.
    pub fn validate(&self) -> Result<(), FrameError> {
        let ftype = self.ftype();
        match self {
            Control::Hello { hello, credential } => {
                if credential.len() != hello.cred_kind.credential_len() {
                    return Err(FrameError::CredentialLength {
                        kind: hello.cred_kind as u16,
                        len: credential.len(),
                    });
                }
                if hello.resume_from == Some(0) {
                    return Err(bad(ftype, "resume_from", 0u64));
                }
                wire_dtype_code(hello.wire_dtype)?;
            }
            Control::HelloAck { ack, credential } => {
                // A refusal may come before the acceptor has a credential to show.
                let refused_bare = ack.status != code::OK && credential.is_empty();
                if !refused_bare && credential.len() != ack.cred_kind.credential_len() {
                    return Err(FrameError::CredentialLength {
                        kind: ack.cred_kind as u16,
                        len: credential.len(),
                    });
                }
            }
            Control::Close { reason, .. } => {
                if reason.len() > MAX_REASON {
                    return Err(FrameError::BadReason { len: reason.len() });
                }
            }
            Control::Credit(credit) => {
                if credit.limit_seq < credit.ack_seq {
                    return Err(bad(ftype, "limit_seq", credit.limit_seq));
                }
            }
            Control::Ping(_) => {}
        }
        Ok(())
    }

    /// Writes the plaintext extension into `ext`, zero-padded, and returns its length: a multiple
    /// of 64.
    pub fn write_ext(&self, ext: &mut [u8; MAX_CONTROL_EXT]) -> Result<usize, FrameError> {
        ext.fill(0);
        let used = match self {
            Control::Hello { hello, .. } => {
                ext[0] = hello.purpose as u8;
                ext[1] = hello.wire_minor_max;
                put16(ext, 2, hello.proto_major);
                put64(ext, 8, hello.capabilities);
                put64(ext, 16, hello.instance.0);
                put64(ext, 24, hello.epoch.term);
                put64(ext, 32, hello.epoch.seq);
                put16(ext, 40, hello.from_stage.0);
                put16(ext, 42, hello.to_stage.0);
                ext[44] = hello.lane;
                ext[45] = wire_dtype_code(hello.wire_dtype)?;
                put32(ext, 48, hello.slot_size);
                put16(ext, 52, hello.recv_slots);
                put16(ext, 54, hello.cred_kind as u16);
                put64(ext, 56, hello.channel_token);
                match hello.resume_from {
                    Some(resume_from) => {
                        put64(ext, 64, resume_from);
                        72
                    }
                    None => 64,
                }
            }
            Control::HelloAck { ack, .. } => {
                put16(ext, 0, ack.status);
                ext[2] = ack.wire_minor;
                put64(ext, 8, ack.capabilities);
                put64(ext, 16, ack.limit_seq);
                put32(ext, 24, ack.slot_size);
                put16(ext, 28, ack.recv_slots);
                put16(ext, 30, ack.cred_kind as u16);
                32
            }
            Control::Close { close, .. } => {
                put16(ext, 0, close.code);
                put64(ext, 8, close.last_seq);
                16
            }
            Control::Credit(credit) => {
                put64(ext, 0, credit.ack_seq);
                put64(ext, 8, credit.limit_seq);
                put64(ext, 16, credit.echo_seq);
                put32(ext, 24, credit.echo_delay_ns);
                let nacks = credit.nacks.as_slice();
                put16(ext, 28, nacks.len() as u16);
                put16(
                    ext,
                    30,
                    u16::from(credit.sack) | (u16::from(credit.reset) << 1),
                );
                for (index, seq) in nacks.iter().enumerate() {
                    put64(ext, 32 + 8 * index, *seq);
                }
                32 + 8 * nacks.len()
            }
            Control::Ping(ping) => {
                put64(ext, 0, ping.origin_ns);
                ext[8] = u8::from(ping.reply);
                16
            }
        };
        Ok(used.next_multiple_of(ALIGN))
    }

    /// Reads a control message from a plaintext frame's extension and payload.
    pub fn parse(
        ftype: FrameType,
        ext: &[u8],
        payload: &'a [u8],
    ) -> Result<Control<'a>, FrameError> {
        let control = match ftype {
            FrameType::Hello => {
                need(ftype, ext, 64)?;
                let purpose = Purpose::from_code(ext[0]).ok_or(bad(ftype, "purpose", ext[0]))?;
                let cred = get16(ext, 54);
                let cred_kind = CredKind::from_code(cred).ok_or(bad(ftype, "cred_kind", cred))?;
                // A reconnect `Hello` is the one with `resume_from` at offset 64.
                let resume_from = if ext.len() > 64 {
                    need(ftype, ext, 72)?;
                    Some(get64(ext, 64))
                } else {
                    None
                };
                Control::Hello {
                    hello: Hello {
                        purpose,
                        wire_minor_max: ext[1],
                        proto_major: get16(ext, 2),
                        capabilities: get64(ext, 8),
                        instance: InstanceId(get64(ext, 16)),
                        epoch: PlanEpoch {
                            term: get64(ext, 24),
                            seq: get64(ext, 32),
                        },
                        from_stage: StageIx(get16(ext, 40)),
                        to_stage: StageIx(get16(ext, 42)),
                        lane: ext[44],
                        wire_dtype: wire_dtype_from_code(ext[45])?,
                        slot_size: get32(ext, 48),
                        recv_slots: get16(ext, 52),
                        cred_kind,
                        channel_token: get64(ext, 56),
                        resume_from,
                    },
                    credential: payload,
                }
            }
            FrameType::HelloAck => {
                need(ftype, ext, 32)?;
                let cred = get16(ext, 30);
                let cred_kind = CredKind::from_code(cred).ok_or(bad(ftype, "cred_kind", cred))?;
                Control::HelloAck {
                    ack: HelloAck {
                        status: get16(ext, 0),
                        wire_minor: ext[2],
                        capabilities: get64(ext, 8),
                        limit_seq: get64(ext, 16),
                        slot_size: get32(ext, 24),
                        recv_slots: get16(ext, 28),
                        cred_kind,
                    },
                    credential: payload,
                }
            }
            FrameType::Close => {
                need(ftype, ext, 16)?;
                let reason = std::str::from_utf8(payload)
                    .map_err(|_| FrameError::BadReason { len: payload.len() })?;
                Control::Close {
                    close: Close {
                        code: get16(ext, 0),
                        last_seq: get64(ext, 8),
                    },
                    reason,
                }
            }
            FrameType::Credit => {
                need(ftype, ext, 32)?;
                let n = get16(ext, 28);
                if usize::from(n) > MAX_NACKS {
                    return Err(bad(ftype, "n_nack", n));
                }
                need(ftype, ext, 32 + 8 * usize::from(n))?;
                // Filled in place: a credit is decoded for every data frame, so no allocation.
                let mut nacks = Nacks::EMPTY;
                for (index, seq) in nacks.seqs[..usize::from(n)].iter_mut().enumerate() {
                    *seq = get64(ext, 32 + 8 * index);
                }
                nacks.len = n as u8;
                let flags = get16(ext, 30);
                Control::Credit(Credit {
                    ack_seq: get64(ext, 0),
                    limit_seq: get64(ext, 8),
                    echo_seq: get64(ext, 16),
                    echo_delay_ns: get32(ext, 24),
                    sack: flags & 1 != 0,
                    reset: flags & 2 != 0,
                    nacks,
                })
            }
            FrameType::Ping => {
                need(ftype, ext, 9)?;
                let reply = match ext[8] {
                    0 => false,
                    1 => true,
                    other => return Err(bad(ftype, "reply", other)),
                };
                Control::Ping(Ping {
                    origin_ns: get64(ext, 0),
                    reply,
                })
            }
            other => return Err(FrameError::NotControl(other)),
        };
        if matches!(control, Control::Credit(_) | Control::Ping(_)) && !payload.is_empty() {
            return Err(FrameError::UnexpectedPayload {
                ftype,
                len: payload.len(),
            });
        }
        control.validate()?;
        Ok(control)
    }

    /// Encodes this message as an unsealed control frame (`chan_seq` and `step` 0) into `out`, and
    /// returns its length. A sealed control frame is built by the AEAD layer from the same
    /// extension and payload, then encoded with [`header::encode`].
    pub fn encode(&self, spec: &ControlSpec, out: &mut [u8]) -> Result<usize, FrameError> {
        self.validate()?;
        let mut ext = [0; MAX_CONTROL_EXT];
        let ext_len = self.write_ext(&mut ext)?;
        let frame = FrameSpec {
            wire_minor: spec.wire_minor,
            ftype: self.ftype(),
            flags: spec.flags,
            instance: spec.instance,
            epoch: spec.epoch,
            step: 0,
            chan_seq: 0,
        };
        header::encode(&frame, &ext[..ext_len], self.payload(), out)
    }

    /// [`Control::encode`] into a new vector.
    pub fn encode_to_vec(&self, spec: &ControlSpec) -> Result<Vec<u8>, FrameError> {
        let mut out = vec![0; header::HEADER_LEN + MAX_CONTROL_EXT + self.payload().len()];
        let len = self.encode(spec, &mut out)?;
        out.truncate(len);
        Ok(out)
    }
}

impl<'a> Frame<'a> {
    /// The control message of an unsealed control frame.
    pub fn control(&self) -> Result<Control<'a>, FrameError> {
        let ftype = self.header.spec.ftype;
        if !ftype.is_control() {
            return Err(FrameError::NotControl(ftype));
        }
        if self.header.spec.flags.encrypted() {
            return Err(FrameError::Sealed(ftype));
        }
        Control::parse(ftype, self.ext, self.payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ControlSpec {
        ControlSpec {
            wire_minor: header::WIRE_MINOR,
            flags: Flags::default(),
            instance: InstanceId(5),
            epoch: PlanEpoch { term: 2, seq: 8 },
        }
    }

    fn round_trip(control: Control<'_>) {
        let bytes = control.encode_to_vec(&spec()).unwrap();
        let frame = Frame::decode(&bytes).unwrap();
        assert_eq!(frame.header.spec.chan_seq, 0);
        assert_eq!(frame.control().unwrap(), control);
    }

    #[test]
    fn each_control_message_round_trips() {
        let credential = [7u8; 168];
        let hello = Hello {
            purpose: Purpose::Tokens,
            wire_minor_max: 1,
            proto_major: 1,
            capabilities: 0b1011,
            instance: InstanceId(5),
            epoch: PlanEpoch { term: 2, seq: 8 },
            from_stage: StageIx(1),
            to_stage: StageIx(2),
            lane: 0,
            wire_dtype: None,
            slot_size: 4096,
            recv_slots: 32,
            cred_kind: CredKind::DataPlaneCredential,
            channel_token: 0,
            resume_from: None,
        };
        round_trip(Control::Hello {
            hello,
            credential: &credential,
        });
        round_trip(Control::Hello {
            hello: Hello {
                resume_from: Some(17),
                ..hello
            },
            credential: &credential,
        });
        round_trip(Control::Close {
            close: Close {
                code: code::ACK_TIMEOUT,
                last_seq: 99,
            },
            reason: "",
        });
        round_trip(Control::Credit(Credit {
            ack_seq: 10,
            limit_seq: 42,
            echo_seq: 0,
            echo_delay_ns: 0,
            sack: false,
            reset: false,
            nacks: Nacks::new(&[11; 16]).unwrap(),
        }));
        round_trip(Control::Ping(Ping {
            origin_ns: u64::MAX,
            reply: true,
        }));
    }

    #[test]
    fn the_extension_sizes_follow_the_tables() {
        let mut ext = [0; MAX_CONTROL_EXT];
        let credit = |n: usize| {
            Control::Credit(Credit {
                ack_seq: 0,
                limit_seq: 0,
                echo_seq: 0,
                echo_delay_ns: 0,
                sack: true,
                reset: false,
                nacks: Nacks::new(&vec![1; n]).unwrap(),
            })
        };
        assert_eq!(credit(0).write_ext(&mut ext), Ok(64));
        assert_eq!(credit(4).write_ext(&mut ext), Ok(64));
        assert_eq!(credit(5).write_ext(&mut ext), Ok(128));
        assert_eq!(credit(16).write_ext(&mut ext), Ok(192));
        assert!(Nacks::new(&[1; 17]).is_none());
    }

    #[test]
    fn malformed_messages_are_refused() {
        let ext = [0u8; 64];
        assert!(matches!(
            Control::parse(FrameType::Hello, &ext, &[0; 168]),
            Err(FrameError::BadField {
                field: "purpose",
                value: 0,
                ..
            })
        ));
        let mut hello = [0u8; 64];
        hello[0] = 1;
        hello[54] = 1;
        assert!(Control::parse(FrameType::Hello, &hello, &[0; 168]).is_ok());
        assert!(matches!(
            Control::parse(FrameType::Hello, &hello, &[0; 192]),
            Err(FrameError::CredentialLength { kind: 1, len: 192 })
        ));
        hello[45] = 6;
        assert!(matches!(
            Control::parse(FrameType::Hello, &hello, &[0; 168]),
            Err(FrameError::BadField {
                field: "wire_dtype",
                value: 6,
                ..
            })
        ));
        let mut credit = [0u8; 64];
        credit[28] = 17;
        assert!(matches!(
            Control::parse(FrameType::Credit, &credit, &[]),
            Err(FrameError::BadField {
                field: "n_nack",
                value: 17,
                ..
            })
        ));
        credit[28] = 5;
        assert!(matches!(
            Control::parse(FrameType::Credit, &credit, &[]),
            Err(FrameError::ExtTooShort { needed: 72, .. })
        ));
        credit[28] = 0;
        credit[8] = 1;
        assert!(Control::parse(FrameType::Credit, &credit, &[]).is_ok());
        credit[0] = 2;
        assert!(matches!(
            Control::parse(FrameType::Credit, &credit, &[]),
            Err(FrameError::BadField {
                field: "limit_seq",
                value: 1,
                ..
            })
        ));
        let mut ping = [0u8; 64];
        ping[8] = 2;
        assert!(matches!(
            Control::parse(FrameType::Ping, &ping, &[]),
            Err(FrameError::BadField {
                field: "reply",
                value: 2,
                ..
            })
        ));
        ping[8] = 0;
        assert!(matches!(
            Control::parse(FrameType::Ping, &ping, &[1]),
            Err(FrameError::UnexpectedPayload { len: 1, .. })
        ));
        assert!(matches!(
            Control::parse(FrameType::Close, &[0; 64], &[0xFF, 0xFE]),
            Err(FrameError::BadReason { len: 2 })
        ));
        assert!(matches!(
            Control::parse(FrameType::Close, &[0; 64], &[b'a'; 257]),
            Err(FrameError::BadReason { len: 257 })
        ));
        assert!(matches!(
            Control::parse(FrameType::Activation, &[0; 64], &[]),
            Err(FrameError::NotControl(FrameType::Activation))
        ));
    }

    #[test]
    fn a_refusing_hello_ack_may_omit_the_credential() {
        let ack = HelloAck {
            status: code::PROTO_MAJOR_MISMATCH,
            wire_minor: 1,
            capabilities: 0,
            limit_seq: 0,
            slot_size: 0,
            recv_slots: 0,
            cred_kind: CredKind::DataPlaneCredential,
        };
        round_trip(Control::HelloAck {
            ack,
            credential: &[],
        });
        let accepted = HelloAck {
            status: code::OK,
            ..ack
        };
        assert!(matches!(
            Control::HelloAck {
                ack: accepted,
                credential: &[]
            }
            .validate(),
            Err(FrameError::CredentialLength { len: 0, .. })
        ));
    }

    #[test]
    fn a_sealed_frame_is_not_read_as_a_control_message() {
        let mut ext = [0; MAX_CONTROL_EXT];
        let ping = Control::Ping(Ping {
            origin_ns: 1,
            reply: false,
        });
        let ext_len = ping.write_ext(&mut ext).unwrap();
        let sealed = FrameSpec {
            wire_minor: header::WIRE_MINOR,
            ftype: FrameType::Ping,
            flags: Flags(Flags::ENCRYPTED),
            instance: InstanceId(5),
            epoch: PlanEpoch::default(),
            step: 1,
            chan_seq: 0,
        };
        let bytes = header::encode_to_vec(&sealed, &ext[..ext_len], &[0; 16]).unwrap();
        let frame = Frame::decode(&bytes).unwrap();
        assert_eq!(frame.control(), Err(FrameError::Sealed(FrameType::Ping)));
    }
}
