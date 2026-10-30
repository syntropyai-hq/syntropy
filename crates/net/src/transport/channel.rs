// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`Channel`] and the types its methods take and return (RFC-0001 §5.8, RFC-0031 §2). Owner:
//! RFC-0031.

use std::fmt;

use syntropy_core::plan::WireDType;
use syntropy_core::{Bytes, InstanceId, MonoTime, Nanos, PlanEpoch, StageIx};

use super::buffer::{ExtBuf, ExtRef, PayloadRef, PayloadSlice, PoolRegistration};
use crate::error::ErrorCode;
use crate::frame::{Flags, FrameType, Header, MAX_REASON, Purpose};

/// What a channel carries (RFC-0031 §2). RFC-0001 §5.8 lists the first six; `Tap` carries the
/// parity driver's teacher tokens and reference top-K ids (RFC-0060 amendment A4, RFC-0031
/// §6.13).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChannelPurpose {
    Activations,
    Tokens,
    Sequence,
    Collective,
    Replay,
    Bulk,
    Tap,
}

impl ChannelPurpose {
    /// Every purpose, in the order of their `Hello` codes.
    pub const ALL: [ChannelPurpose; 7] = [
        ChannelPurpose::Activations,
        ChannelPurpose::Tokens,
        ChannelPurpose::Sequence,
        ChannelPurpose::Collective,
        ChannelPurpose::Replay,
        ChannelPurpose::Bulk,
        ChannelPurpose::Tap,
    ];
}

/// The purpose's code in a `Hello` (RFC-0031 §6.9).
impl From<ChannelPurpose> for Purpose {
    fn from(purpose: ChannelPurpose) -> Purpose {
        match purpose {
            ChannelPurpose::Activations => Purpose::Activations,
            ChannelPurpose::Tokens => Purpose::Tokens,
            ChannelPurpose::Sequence => Purpose::Sequence,
            ChannelPurpose::Collective => Purpose::Collective,
            ChannelPurpose::Replay => Purpose::Replay,
            ChannelPurpose::Bulk => Purpose::Bulk,
            ChannelPurpose::Tap => Purpose::Tap,
        }
    }
}

/// A `Hello`'s purpose as a channel purpose; a blob stream is not a channel, so it has none.
impl TryFrom<Purpose> for ChannelPurpose {
    type Error = Purpose;

    fn try_from(purpose: Purpose) -> Result<ChannelPurpose, Purpose> {
        Ok(match purpose {
            Purpose::Activations => ChannelPurpose::Activations,
            Purpose::Tokens => ChannelPurpose::Tokens,
            Purpose::Sequence => ChannelPurpose::Sequence,
            Purpose::Collective => ChannelPurpose::Collective,
            Purpose::Replay => ChannelPurpose::Replay,
            Purpose::Bulk => ChannelPurpose::Bulk,
            Purpose::Tap => ChannelPurpose::Tap,
            Purpose::BlobStream => return Err(purpose),
        })
    }
}

/// The stage index that stands for the ingress daemon in a [`ChannelKey`]: a `Sequence` channel
/// runs `INGRESS → head`, and a `Tokens` channel to the ingress `tail → INGRESS` (RFC-0031 §2).
pub const INGRESS: StageIx = StageIx(0xFFFF);

/// Identifies a channel. Two channels with equal keys never coexist on one endpoint pair
/// (RFC-0031 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChannelKey {
    pub instance: InstanceId,
    /// The epoch the channel was opened for; a relink (RFC-0032 §4.4) opens new channels.
    pub epoch: PlanEpoch,
    pub purpose: ChannelPurpose,
    /// [`INGRESS`] for a `Sequence` channel.
    pub from: StageIx,
    /// [`INGRESS`] for a `Tokens` channel to the ingress.
    pub to: StageIx,
    /// Parallel channels on one edge: collective member pairs, replay streams, and the decode
    /// (0) and prefill (1) lanes of an `Activations` hop (RFC-0032 §3.2).
    pub lane: u8,
}

/// A channel's priority over others on the same connection or queue: QUIC stream priority, RDMA
/// completion-queue poll order (RFC-0031 §2, §4.2). Ordered lowest first, so `Data` is the
/// greatest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChannelPriority {
    Bulk,
    Replay,
    Collective,
    Data,
}

/// How to open one side of a channel (RFC-0031 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChannelSpec {
    pub key: ChannelKey,
    /// Where this side's receive slots come from (RFC-0031 §8).
    pub pool: PoolRegistration,
    /// The largest frame this side accepts, header included.
    pub slot_size: Bytes,
    /// Receive slots posted, which is the initial credit granted to the peer (RFC-0031 §7.1).
    pub recv_slots: u16,
    pub priority: ChannelPriority,
    /// Checked against `HopPlan::wire` (RFC-0001 §5.4); an `Activation` in another dtype is
    /// refused (RFC-0031 §6.4).
    pub wire: WireDType,
    /// Frames kept for retransmission; at least the peer's `recv_slots` (RFC-0031 §7.3).
    pub retention: u16,
}

/// A data frame to send (RFC-0031 §2). Its extension and payload already sit in frozen slots of
/// a registered pool; the channel writes the base header and the CRCs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutFrame {
    /// A data frame type; control frames are the channel's own.
    pub ftype: FrameType,
    pub step: u64,
    pub flags: Flags,
    pub ext: ExtBuf,
    pub payload: PayloadSlice,
}

/// A received data frame, in `chan_seq` order and past its CRC checks (RFC-0031 §2). Its
/// extension and payload are borrowed from the receive slot; dropping the frame returns the
/// slot, and the channel may then grant a credit for it (§7.1).
#[derive(Debug)]
pub struct InFrame {
    /// The parsed, CRC-checked base header (RFC-0031 §6.1).
    pub header: Header,
    pub ext: ExtRef,
    pub payload: PayloadRef,
}

/// The receipt for a sent data frame: its `chan_seq`, numbered from 1 on each channel.
/// [`Channel::completed`] hands it back once the peer has acknowledged the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SendTicket {
    pub chan_seq: u64,
}

/// Why [`Channel::try_send`] did not send. None of these allocates, so the hot path never does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SendError {
    /// Send credit is exhausted: the next `chan_seq` is past the peer's `limit_seq` (RFC-0031
    /// §7.1). Nothing was sent; send the frame again once credit returns. The time held counts as
    /// `credit_stall`. On a `Bulk` or `Replay` channel of a paced edge, it also means the edge's
    /// bulk pacer does not have the tokens yet (§7.5), and [`Channel::paced_until`] says when it will;
    /// that wait is not a credit stall.
    WouldBlock,
    /// The frame is larger than the peer's slot or the transport's `max_frame` (RFC-0031 §12
    /// `FrameTooLarge`).
    TooLarge,
    /// The channel cannot send this frame: a control frame type, an extension or payload outside
    /// the channel's registered pools or not frozen, or an extension whose length is not a
    /// multiple of 64.
    Invalid,
    /// The channel was closed, by this side or by the peer.
    Closed,
    /// The link failed; the backend reports the cause and starts recovery (RFC-0031 §7.4).
    Link,
}

impl fmt::Display for SendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SendError::WouldBlock => "no send credit",
            SendError::TooLarge => "the frame is larger than the peer accepts",
            SendError::Invalid => "the channel cannot send this frame",
            SendError::Closed => "the channel is closed",
            SendError::Link => "the link failed",
        })
    }
}

impl std::error::Error for SendError {}

/// Why a channel is closed; it travels in the `Close` frame (RFC-0031 §6.9).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CloseReason {
    code: Option<ErrorCode>,
    text: String,
}

impl CloseReason {
    /// A close with nothing wrong: code 0 (`OK`) and no text.
    pub const NORMAL: CloseReason = CloseReason {
        code: None,
        text: String::new(),
    };

    /// A close with an RFC-0031 §12 code, or `None` for code 0, and a reason for people. Text
    /// longer than `MAX_REASON` (256 bytes) is cut at the last character boundary that fits.
    pub fn new(code: Option<ErrorCode>, text: &str) -> CloseReason {
        let mut end = text.len().min(MAX_REASON);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        CloseReason {
            code,
            text: text[..end].to_owned(),
        }
    }

    pub fn code(&self) -> Option<ErrorCode> {
        self.code
    }

    /// The code as the `Close` frame carries it: 0 for none.
    pub fn wire_code(&self) -> u16 {
        self.code.map_or(crate::frame::code::OK, |code| code.0)
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

/// A channel's counters (RFC-0031 §2, §7). They feed the telemetry ring (RFC-0001 §13) and online
/// link refinement (RFC-0010). Bytes count whole frames: base header, extension and payload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ChannelStats {
    /// Data frames `try_send` accepted, and their bytes. Retransmits are counted apart.
    pub frames_sent: u64,
    pub bytes_sent: u64,
    /// Data frames `try_recv` handed over, and their bytes.
    pub frames_received: u64,
    pub bytes_received: u64,
    /// Data frames sent again, after a NACK, an ack timeout or a failover (RFC-0031 §7.3).
    pub retransmits: u64,
    /// `chan_seq` values this side asked the peer to send again.
    pub nacks_sent: u64,
    /// `try_send` calls refused with [`SendError::WouldBlock`].
    pub credit_stalls: u64,
    /// Time from a refused send to the credit that let it through, as the backend measures it
    /// (RFC-0001 §8.4 rule 1).
    pub credit_stall: Nanos,
    /// Round trips sampled from credit echoes (RFC-0031 §7.1), and the latest one-way α.
    pub alpha_samples: u64,
    pub alpha_last: Option<Nanos>,
    /// Frames dropped on a failed header or payload CRC (RFC-0031 §7.2).
    pub crc_header_fail: u64,
    pub crc_payload_fail: u64,
    /// Frames dropped because their instance or epoch is not the channel's (RFC-0001 invariant 7).
    pub stale_dropped: u64,
}

/// What a transport backend implements for one side of one channel. [`Channel`] calls it once
/// per method call; that one virtual call is the only cost of the indirection.
///
/// The methods follow the rules RFC-0031 §2 makes binding on every backend, which
/// [`conformance`](super::conformance) checks. The hot-path methods (`try_send`, `try_recv`,
/// `completed`, `credit`) are synchronous, never block and never allocate. A backend built on an
/// async stack (QUIC) serves them from SPSC rings to a task in the `net` domain (RFC-0031 §4.4).
pub trait ChannelBackend: Send {
    /// See [`Channel::try_send`].
    fn try_send(&mut self, frame: OutFrame) -> Result<SendTicket, SendError>;
    /// See [`Channel::try_recv`].
    fn try_recv(&mut self) -> Option<InFrame>;
    /// See [`Channel::completed`].
    fn completed(&mut self) -> Option<SendTicket>;
    /// See [`Channel::stats`].
    fn stats(&self) -> ChannelStats;
    /// See [`Channel::credit`].
    fn credit(&self) -> u32;
    /// See [`Channel::nack`].
    fn nack(&mut self, seq: u64);
    /// See [`Channel::close`].
    fn close(&mut self, reason: CloseReason);
    /// See [`Channel::paced_until`]. A backend that does not pace returns `None`.
    fn paced_until(&self) -> Option<MonoTime> {
        None
    }
}

/// One ordered, lossless, authenticated stream of data frames between two endpoints (RFC-0001
/// §5.8, RFC-0031 §2).
///
/// A concrete type over a [`ChannelBackend`], so every transport's channels, and the simulator's,
/// have one type. When an edge demotes to another transport, the channel keeps its key and its
/// owner's handle while the backend underneath is replaced, and the retained frames are sent
/// again on the new one (ADR-0011 item 7, RFC-0031 §7.4).
pub struct Channel {
    key: ChannelKey,
    backend: Box<dyn ChannelBackend>,
}

impl Channel {
    /// The channel `key` served by `backend`. Backends call this from `Transport::connect` and
    /// `Transport::accept`.
    pub fn new(key: ChannelKey, backend: Box<dyn ChannelBackend>) -> Channel {
        Channel { key, backend }
    }

    pub fn key(&self) -> &ChannelKey {
        &self.key
    }

    /// Queues a data frame whose extension and payload already sit in registered slots. The
    /// channel assigns the next `chan_seq`, from 1, writes the base header and the CRCs, and
    /// keeps a lease on both slots until the peer acknowledges the frame (RFC-0031 §7.3), so
    /// the caller may drop its own readers as soon as this returns. Returns
    /// [`SendError::WouldBlock`] when send credit is exhausted (§7.1). Never blocks and never
    /// allocates.
    pub fn try_send(&mut self, frame: OutFrame) -> Result<SendTicket, SendError> {
        self.backend.try_send(frame)
    }

    /// The next data frame, strictly in `chan_seq` order, after its header and payload CRC
    /// checks. A gap or a CRC failure never reaches the caller: the channel NACKs and waits
    /// (RFC-0031 §7.2). A frame whose instance or epoch is not the channel's is dropped and
    /// counted. The payload stays in its receive slot until the frame is dropped.
    pub fn try_recv(&mut self) -> Option<InFrame> {
        self.backend.try_recv()
    }

    /// The next sent frame the peer has **cumulatively acknowledged**, in `chan_seq` order. A
    /// local send completion is not enough: a Thunderbolt 5 send completion proves neither
    /// delivery nor integrity (RFC-0031 §2). By the time a ticket is handed back, the frame has
    /// left the retention window and the channel has released its leases on the frame's slots.
    pub fn completed(&mut self) -> Option<SendTicket> {
        self.backend.completed()
    }

    /// The channel's counters.
    pub fn stats(&self) -> ChannelStats {
        self.backend.stats()
    }

    /// Frames this side may still send before [`SendError::WouldBlock`] (RFC-0031 §2).
    pub fn credit(&self) -> u32 {
        self.backend.credit()
    }

    /// Asks the peer to send again from `seq`, after a UC loss or a CRC failure (RFC-0031 §2).
    /// A frame this side already received is never handed to the caller twice.
    pub fn nack(&mut self, seq: u64) {
        self.backend.nack(seq)
    }

    /// When the send this channel's bulk pacer held can go, on the pacer's clock: `Some` when
    /// the last `try_send` returned [`SendError::WouldBlock`] for want of the pacer's tokens
    /// (RFC-0031 §7.5). A time already past means at once. `None` otherwise, and always on a
    /// channel that is not paced.
    pub fn paced_until(&self) -> Option<MonoTime> {
        self.backend.paced_until()
    }

    /// Closes the channel gracefully once the peer has acknowledged everything sent; the reason
    /// travels in a `Close` frame (RFC-0031 §2, §6.9). Later sends return
    /// [`SendError::Closed`].
    pub fn close(&mut self, reason: CloseReason) {
        self.backend.close(reason)
    }
}

impl fmt::Debug for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Channel").field("key", &self.key).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn purposes_map_onto_hello_codes_and_back() {
        let codes: Vec<u8> = ChannelPurpose::ALL
            .into_iter()
            .map(|purpose| Purpose::from(purpose) as u8)
            .collect();
        assert_eq!(codes, [1, 2, 3, 4, 5, 6, 7]);
        for purpose in ChannelPurpose::ALL {
            assert_eq!(
                ChannelPurpose::try_from(Purpose::from(purpose)),
                Ok(purpose)
            );
        }
        assert_eq!(
            ChannelPurpose::try_from(Purpose::BlobStream),
            Err(Purpose::BlobStream)
        );
    }

    #[test]
    fn data_outranks_collective_replay_and_bulk() {
        use ChannelPriority::{Bulk, Collective, Data, Replay};
        let mut order = [Replay, Data, Bulk, Collective];
        order.sort();
        assert_eq!(order, [Bulk, Replay, Collective, Data]);
    }

    #[test]
    fn a_close_reason_fits_the_close_frame() {
        assert_eq!(CloseReason::NORMAL.wire_code(), 0);
        assert_eq!(CloseReason::NORMAL.text(), "");
        let reason = CloseReason::new(Some(ErrorCode::ACK_TIMEOUT), "no ack");
        assert_eq!((reason.wire_code(), reason.text()), (0x0208, "no ack"));
        let long = "é".repeat(200);
        let cut = CloseReason::new(None, &long);
        assert_eq!(cut.text().len(), 256);
        assert!(long.starts_with(cut.text()));
        let odd = format!("x{long}");
        assert_eq!(CloseReason::new(None, &odd).text().len(), 255);
    }
}
