// SPDX-License-Identifier: MIT OR Apache-2.0

//! The channel core: RFC-0031 §7's flow control, acknowledgement and retransmission for one
//! channel, sans I/O and the same on every transport (DIST-E7-T02; ADR-0307 items 3–5).
//!
//! A [`ChannelCore`] holds both directions of one side of a channel: the data it sends and keeps
//! until the peer acknowledges it, and the data it receives and hands over in order. It does no
//! I/O and reads no clock. The backend that wraps it (QUIC, TCP, shared memory, RDMA, or the
//! simulator's `SimTransport`) moves bytes and passes the time in:
//! - [`ChannelCore::send`] queues a data frame;
//! - [`ChannelCore::poll_transmit`] gives the next frame to put on the wire;
//! - [`ChannelCore::next_receive_slot`] lends a slot to receive into, and
//!   [`ChannelCore::on_receive`] takes the frame back once it is there;
//! - [`ChannelCore::recv`] and [`ChannelCore::completed`] give `Channel::try_recv` and
//!   `Channel::completed`;
//! - [`ChannelCore::on_tick`] runs the timers.
//!
//! **Credits** (§7.1). Data frames are numbered from 1; control frames (`Hello`, `HelloAck`,
//! `Credit`, `Ping`, `Close`) carry `chan_seq = 0`, take no credit and are never NACKed. The
//! receive pool has `recv_slots + 4` slots: credit counts only `recv_slots` of them, so the other
//! four are always there for control frames and a full data window never stops a `Ping`. The
//! credit granted is `ack_seq + recv_slots −` the frames handed over or waiting to be, which only
//! grows: it rises when the caller drops a received frame and its slot comes back.
//!
//! **Receive checks** (§6.1, §7.2), in order: the header's integrity, the channel key, then
//! `wire_minor` and the frame rules, then `chan_seq`, then the payload's CRC. A frame for another
//! instance or epoch is dropped and counted as soon as its header CRC passes, whatever else its
//! header holds, and never NACKed. A broken header NACKs the next expected frame; a broken payload
//! NACKs its own; a duplicate is dropped and the current credit sent again; a frame past a gap is
//! held in its slot, and each missing frame NACKed once per `nack_interval`.
//!
//! **Retention** (§7.3). Each sent frame keeps its 64-byte header and its leases on its extension
//! and payload slots until the peer's `ack_seq` covers it, so a resend, after a NACK or an ack
//! timeout, is byte-identical to the first send. At most `retention` frames, each at most the
//! peer's slot size, are kept.
//!
//! **The credit probe.** A credit that only grants more can be lost while the sender has nothing
//! unacknowledged, and then no ack timeout repairs it. A sender refused for credit with nothing
//! unacknowledged sends a `Ping`, with backoff, and a data receiver answers a `Ping` with its
//! current credit (RFC-0031 §7, note of 2026-10-23).

use std::collections::VecDeque;

use syntropy_core::{ErrorClass, InstanceId, MonoTime, Nanos, PlanEpoch};

use super::buffer::{BufferPool, ExtRef, PayloadRef, SlotReader, SlotWriter};
use super::channel::{
    ChannelKey, ChannelPurpose, ChannelStats, CloseReason, InFrame, SendError, SendTicket,
};
use crate::error::{Error, ErrorCode};
use crate::frame::{
    self, Close, Control, ControlSpec, Credit, Flags, Frame, FrameError, FrameSpec, FrameType,
    HEADER_LEN, Header, MAX_CONTROL_EXT, MAX_NACKS, MAX_REASON, Nacks, Ping,
};

/// Receives kept for control frames beside the `recv_slots` data receives (§7.1).
pub const CONTROL_RECEIVES: u16 = 4;

/// The largest control frame this core sends or takes: the header, the largest extension and the
/// longest payload, a `Close` reason or a credential.
pub const MAX_CONTROL_FRAME: usize = HEADER_LEN + MAX_CONTROL_EXT + MAX_REASON;

/// The floor of the ack timeout, `max(4·α_p99, 200 µs)` (§7.3).
pub const ACK_TIMEOUT_MIN: Nanos = Nanos(200_000);

/// The floor of the NACK interval, `max(2·α_p99, 100 µs)` (§7.2).
pub const NACK_INTERVAL_MIN: Nanos = Nanos(100_000);

/// Payload CRC failures within [`CRC_STORM_WINDOW`] that make a link fault, `CrcStorm` (§7.2).
pub const CRC_STORM_COUNT: u32 = 3;
/// The window of [`CRC_STORM_COUNT`]: 10 s.
pub const CRC_STORM_WINDOW: Nanos = Nanos(10_000_000_000);

/// Ack timeouts in a row that make a link fault, `AckTimeout` (§7.3, §7.4: "ack timeout × 3").
pub const ACK_TIMEOUTS_TO_FAULT: u32 = 3;

/// Ping requests whose replies wait to be sent; more are dropped, and the pinger tries again.
const MAX_PING_REPLIES: usize = 4;

/// The longest wait between a stalled sender's credit probes, in ack timeouts; a power of 2.
/// 64 \[target\]: a chosen budget, with the 1, 2, 4, … backoff, not a measured figure.
pub const PROBE_BACKOFF_MAX: u64 = 64;

/// One-way α samples the core keeps for its p99 estimate.
/// 128 \[target\]: a chosen budget, not a measured figure.
pub const ALPHA_WINDOW: usize = 128;

/// Pings this core sent whose replies it still accepts; an older one's reply gives no sample.
const PINGS_OUT: usize = 4;

/// When the receiver sends a credit (§7.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CreditCadence {
    /// After every received data frame, which also gives the sender an α sample per frame:
    /// `Activations`, `Tokens`, `Sequence` and `Tap` channels, and decode `Collective` lanes.
    EveryFrame,
    /// Every `max(1, recv_slots / 4)` frames, or when the sender's remaining credit falls below
    /// half: `Replay`, `Bulk`, and the prefill lanes of `Collective`.
    Batched,
}

impl CreditCadence {
    /// The cadence §7.1 gives a channel of `purpose`. A `Collective` prefill lane takes
    /// [`CreditCadence::Batched`], which its opener chooses.
    pub fn for_purpose(purpose: ChannelPurpose) -> CreditCadence {
        match purpose {
            ChannelPurpose::Replay | ChannelPurpose::Bulk => CreditCadence::Batched,
            _ => CreditCadence::EveryFrame,
        }
    }
}

/// How the sending direction is set up, from the peer's `Hello` or `HelloAck`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SendConfig {
    /// The peer's `recv_slots`: the initial `limit_seq`.
    pub peer_recv_slots: u16,
    /// The largest frame the peer accepts, header included.
    pub peer_slot_size: u32,
    /// Frames kept for retransmission; at least `peer_recv_slots` (§7.3).
    pub retention: u16,
}

/// One side of a channel.
#[derive(Clone, Debug)]
pub struct CoreConfig {
    pub key: ChannelKey,
    /// The negotiated wire minor: the highest this side accepts, and the one it writes.
    pub wire_minor: u8,
    pub cadence: CreditCadence,
    /// The link's α p99 before any sample, from the plan's link profile.
    pub alpha_p99: Nanos,
    /// Data frames this side accepts at once, the credit it grants; 0 if it receives no data.
    pub recv_slots: u16,
    /// The largest frame this side accepts, header included.
    pub slot_size: u32,
    /// The receive pool: at least `recv_slots + 4` slots of at least `slot_size` bytes, and at
    /// least [`MAX_CONTROL_FRAME`] bytes, for this core alone to hand out.
    pub pool: BufferPool,
    /// The sending direction, if this side sends data.
    pub send: Option<SendConfig>,
    /// Ack timeouts in a row that make a link fault: [`ACK_TIMEOUTS_TO_FAULT`], or `u32::MAX`
    /// for never, which a test of the retransmission alone uses.
    pub ack_timeouts_to_fault: u32,
    /// Payload CRC failures within [`CRC_STORM_WINDOW`] that make a link fault:
    /// [`CRC_STORM_COUNT`], or `u32::MAX` for never.
    pub crc_storm_count: u32,
}

/// A frame to put on the wire.
#[derive(Debug)]
pub enum Transmit<'a> {
    /// A data frame, new or sent again: the header, then the extension, then the payload,
    /// contiguous on the wire.
    Data {
        chan_seq: u64,
        header: &'a [u8; HEADER_LEN],
        ext: &'a [u8],
        payload: &'a [u8],
    },
    /// A control frame, whole.
    Control(&'a [u8]),
}

/// A sent data frame, kept until the peer acknowledges it.
struct Retained {
    chan_seq: u64,
    header: [u8; HEADER_LEN],
    ext: ExtRef,
    payload: PayloadRef,
    /// When it first went on the wire; `None` until then.
    sent_at: Option<MonoTime>,
    /// Whether it went on the wire again. An echo of a frame sent twice gives no α sample: its
    /// arrival cannot be matched to one of the sends.
    resent: bool,
}

impl Retained {
    fn len(&self) -> u64 {
        (HEADER_LEN + self.ext.len() + self.payload.len()) as u64
    }
}

/// The sending direction's state.
struct SendState {
    config: SendConfig,
    next_seq: u64,
    ack_seq: u64,
    limit_seq: u64,
    retained: VecDeque<Retained>,
    /// `chan_seq`s not yet sent once, in order.
    unsent: VecDeque<u64>,
    /// `chan_seq`s to send again, in the order asked.
    resend: VecDeque<u64>,
    /// The last ticket `completed` handed back; those up to `ack_seq` are due.
    completed_upto: u64,
    /// When `ack_seq` last moved, or the ack timer was last restarted.
    last_progress: MonoTime,
    timeouts: u32,
    /// Since when a send has been refused for want of credit.
    stalled_since: Option<MonoTime>,
    /// When the last credit probe went, and how many have gone since the stall began.
    last_probe: Option<MonoTime>,
    probes: u32,
}

/// A received data frame in its slot.
struct Held {
    chan_seq: u64,
    header: Header,
    reader: SlotReader,
}

/// The core of one side of one channel (see the [module](self) docs).
pub struct ChannelCore {
    key: ChannelKey,
    wire_minor: u8,
    cadence: CreditCadence,
    ctl: ControlSpec,
    pool: BufferPool,
    recv_slots: u16,
    slot_size: u32,
    ack_timeouts_to_fault: u32,
    crc_storm_count: u32,
    alpha_p99_initial: Nanos,
    alpha: [u64; ALPHA_WINDOW],
    alpha_len: usize,
    alpha_next: usize,
    // The receiving direction.
    /// The next `chan_seq` to hand over in order.
    expected: u64,
    /// Frames past a gap, by `chan_seq % recv_slots`.
    held: Vec<Option<Held>>,
    held_count: u16,
    highest_seen: u64,
    /// In order, waiting for `recv`.
    ready: VecDeque<Held>,
    /// The slots of frames handed over, until the caller drops them.
    given_out: VecDeque<u32>,
    /// The `limit_seq` last granted.
    granted: u64,
    credit_due: bool,
    frames_since_credit: u16,
    /// The data frame whose arrival the next credit echoes, and when it arrived.
    echo: Option<(u64, MonoTime)>,
    /// When each missing `chan_seq` was last NACKed, by `chan_seq % recv_slots`.
    nacked: Vec<(u64, MonoTime)>,
    nacks: Vec<u64>,
    payload_failures: VecDeque<MonoTime>,
    // The sending direction.
    send: Option<SendState>,
    // Control.
    ping_replies: VecDeque<u64>,
    closing: Option<CloseReason>,
    close_sent: bool,
    peer_close: Option<Close>,
    fault: Option<Error>,
    fault_close_sent: bool,
    /// A credit probe waits to be sent.
    probe_due: bool,
    /// The `origin_ns` of the pings this core sent, newest last: only a reply carrying one of
    /// them gives an α sample, so a peer cannot raise α and with it the ack timeout.
    pings_out: [Option<u64>; PINGS_OUT],
    ping_next: usize,
    out: [u8; MAX_CONTROL_FRAME],
    stats: ChannelStats,
}

/// A configuration the core cannot run with.
fn invalid(message: impl Into<String>) -> Error {
    Error::uncoded(ErrorClass::InvalidRequest, message)
}

/// The `(instance, epoch)` in a base header's bytes 8..32 (RFC-0031 §6.1).
fn raw_key(header: &[u8]) -> (InstanceId, PlanEpoch) {
    let word = |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().expect("8 bytes"));
    (
        InstanceId(word(8)),
        PlanEpoch {
            term: word(16),
            seq: word(24),
        },
    )
}

/// A received frame the frame rules refuse though its header is intact, as `net`'s error: with
/// the RFC-0031 §12 code `FrameError` gives it, such as `UnknownFrameType`, or with its class.
/// Never `Internal`: these are refusals of untrusted bytes (RFC-0060 §10.2).
fn refusal(error: &FrameError) -> Error {
    let message = format!("the peer sent a frame the frame rules refuse: {error}");
    match error.code() {
        Some(code) => Error::new(ErrorCode(code), message),
        None => Error::uncoded(error.class(), message),
    }
}

impl ChannelCore {
    /// The core of one side of the channel `config.key`, open: the handshake (`Hello`,
    /// `HelloAck`) has run and set `config`.
    pub fn new(config: CoreConfig) -> Result<ChannelCore, Error> {
        let need_slots = u32::from(config.recv_slots) + u32::from(CONTROL_RECEIVES);
        if config.pool.slots() < need_slots {
            return Err(invalid(format!(
                "the receive pool has {} slots; recv_slots + {CONTROL_RECEIVES} is {need_slots}",
                config.pool.slots()
            )));
        }
        let need_size = (config.slot_size as usize).max(MAX_CONTROL_FRAME);
        if config.pool.slot_size() < need_size {
            return Err(invalid(format!(
                "the receive pool's slots hold {} bytes; the channel needs {need_size}",
                config.pool.slot_size()
            )));
        }
        if let Some(send) = &config.send
            && send.retention < send.peer_recv_slots
        {
            return Err(invalid(format!(
                "retention {} is below the peer's recv_slots {} (RFC-0031 §7.3)",
                send.retention, send.peer_recv_slots
            )));
        }
        let window = usize::from(config.recv_slots.max(1));
        Ok(ChannelCore {
            key: config.key,
            wire_minor: config.wire_minor,
            cadence: config.cadence,
            ctl: ControlSpec {
                wire_minor: config.wire_minor,
                flags: Flags::default(),
                instance: config.key.instance,
                epoch: config.key.epoch,
            },
            pool: config.pool,
            recv_slots: config.recv_slots,
            slot_size: config.slot_size,
            ack_timeouts_to_fault: config.ack_timeouts_to_fault.max(1),
            crc_storm_count: config.crc_storm_count.max(1),
            alpha_p99_initial: config.alpha_p99,
            alpha: [0; ALPHA_WINDOW],
            alpha_len: 0,
            alpha_next: 0,
            expected: 1,
            held: (0..window).map(|_| None).collect(),
            held_count: 0,
            highest_seen: 0,
            ready: VecDeque::with_capacity(window),
            given_out: VecDeque::with_capacity(window),
            granted: u64::from(config.recv_slots),
            credit_due: false,
            frames_since_credit: 0,
            echo: None,
            nacked: vec![(0, MonoTime(0)); window],
            nacks: Vec::with_capacity(MAX_NACKS),
            payload_failures: VecDeque::with_capacity(CRC_STORM_COUNT as usize),
            send: config.send.map(|send| SendState {
                config: send,
                next_seq: 1,
                ack_seq: 0,
                limit_seq: u64::from(send.peer_recv_slots),
                retained: VecDeque::with_capacity(usize::from(send.retention)),
                unsent: VecDeque::with_capacity(usize::from(send.retention)),
                resend: VecDeque::with_capacity(usize::from(send.retention)),
                completed_upto: 0,
                last_progress: MonoTime(0),
                timeouts: 0,
                stalled_since: None,
                last_probe: None,
                probes: 0,
            }),
            ping_replies: VecDeque::with_capacity(MAX_PING_REPLIES),
            closing: None,
            close_sent: false,
            peer_close: None,
            fault: None,
            fault_close_sent: false,
            probe_due: false,
            pings_out: [None; PINGS_OUT],
            ping_next: 0,
            out: [0; MAX_CONTROL_FRAME],
            stats: ChannelStats::default(),
        })
    }

    pub fn key(&self) -> &ChannelKey {
        &self.key
    }

    pub fn stats(&self) -> ChannelStats {
        self.stats
    }

    /// Why the channel faulted, if it has: the backend reconnects or reports it (§7.4).
    pub fn fault(&self) -> Option<&Error> {
        self.fault.as_ref()
    }

    /// The `Close` the peer sent, if any.
    pub fn peer_close(&self) -> Option<Close> {
        self.peer_close
    }

    /// Frames this side may still send before `WouldBlock`: the peer's credit, within the
    /// retention window.
    pub fn credit(&self) -> u32 {
        self.send.as_ref().map_or(0, |send| {
            let credit = send
                .limit_seq
                .saturating_add(1)
                .saturating_sub(send.next_seq);
            let room = usize::from(send.config.retention).saturating_sub(send.retained.len());
            credit.min(room as u64) as u32
        })
    }

    /// The bytes of the frames kept for retransmission; never more than `retention ×` the peer's
    /// slot size.
    pub fn retained_bytes(&self) -> u64 {
        self.send
            .as_ref()
            .map_or(0, |send| send.retained.iter().map(Retained::len).sum())
    }

    /// The α p99 estimate: the largest of the last [`ALPHA_WINDOW`] one-way samples, or the
    /// link profile's before any sample.
    pub fn alpha_p99(&self) -> Nanos {
        if self.alpha_len == 0 {
            self.alpha_p99_initial
        } else {
            Nanos(
                self.alpha[..self.alpha_len]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0),
            )
        }
    }

    fn ack_timeout(&self) -> Nanos {
        Nanos(self.alpha_p99().0.saturating_mul(4).max(ACK_TIMEOUT_MIN.0))
    }

    fn nack_interval(&self) -> Nanos {
        Nanos(
            self.alpha_p99()
                .0
                .saturating_mul(2)
                .max(NACK_INTERVAL_MIN.0),
        )
    }

    fn sample_alpha(&mut self, one_way: Nanos) {
        self.alpha[self.alpha_next] = one_way.0;
        self.alpha_next = (self.alpha_next + 1) % ALPHA_WINDOW;
        self.alpha_len = (self.alpha_len + 1).min(ALPHA_WINDOW);
        self.stats.alpha_samples += 1;
        self.stats.alpha_last = Some(one_way);
    }

    fn raise(&mut self, error: Error) {
        if self.fault.is_none() {
            self.fault = Some(error);
        }
    }

    /// A fault the backend sees itself, outside the core's own checks: a queue pair in its
    /// error state, a lost QUIC connection or TCP reset, a link down past `link.failover_timeout`
    /// (RFC-0031 §7.4). From now on sends return [`SendError::Link`], the next
    /// [`poll_transmit`](Self::poll_transmit) gives a `Close` with the error's code, and nothing
    /// more is received. The first fault is kept.
    pub fn fail(&mut self, error: Error) {
        self.raise(error);
    }

    // ---- Sending ----

    /// Queues a data frame whose extension and payload sit in the given leases (`Channel::
    /// try_send`). Assigns the next `chan_seq` and writes the header and CRCs. Never blocks and
    /// never allocates.
    pub fn send(
        &mut self,
        ftype: FrameType,
        step: u64,
        flags: Flags,
        ext: ExtRef,
        payload: PayloadRef,
        now: MonoTime,
    ) -> Result<SendTicket, SendError> {
        if self.fault.is_some() {
            return Err(SendError::Link);
        }
        if self.closing.is_some() || self.peer_close.is_some() {
            return Err(SendError::Closed);
        }
        let Some(send) = self.send.as_mut() else {
            return Err(SendError::Invalid);
        };
        if ftype.is_control() || !ext.len().is_multiple_of(frame::ALIGN) {
            return Err(SendError::Invalid);
        }
        let len = (HEADER_LEN + ext.len() + payload.len()) as u64;
        if len > u64::from(send.config.peer_slot_size) {
            return Err(SendError::TooLarge);
        }
        let chan_seq = send.next_seq;
        // The credit, and the retention window whatever credit the peer grants: retention
        // memory stays within `retention × peer_slot_size` (§7.3).
        if chan_seq > send.limit_seq || send.retained.len() >= usize::from(send.config.retention) {
            self.stats.credit_stalls += 1;
            send.stalled_since.get_or_insert(now);
            return Err(SendError::WouldBlock);
        }
        let spec = FrameSpec {
            wire_minor: self.wire_minor,
            ftype,
            flags,
            instance: self.key.instance,
            epoch: self.key.epoch,
            step,
            chan_seq,
        };
        let header = frame::encode_header(&spec, ext.bytes(), payload.bytes())
            .map_err(|_| SendError::Invalid)?;
        send.retained.push_back(Retained {
            chan_seq,
            header,
            ext,
            payload,
            sent_at: None,
            resent: false,
        });
        send.unsent.push_back(chan_seq);
        send.next_seq += 1;
        // A stall the retention window caused ends here, not at a credit.
        if let Some(since) = send.stalled_since.take() {
            self.stats.credit_stall =
                Nanos(self.stats.credit_stall.0 + now.saturating_duration_since(since).0);
        }
        send.last_probe = None;
        send.probes = 0;
        self.stats.frames_sent += 1;
        self.stats.bytes_sent += len;
        Ok(SendTicket { chan_seq })
    }

    /// The next sent frame the peer has cumulatively acknowledged (`Channel::completed`). Its
    /// leases are already released.
    pub fn completed(&mut self) -> Option<SendTicket> {
        let send = self.send.as_mut()?;
        if send.completed_upto >= send.ack_seq {
            return None;
        }
        send.completed_upto += 1;
        Some(SendTicket {
            chan_seq: send.completed_upto,
        })
    }

    /// Asks the peer to send `seq` again (`Channel::nack`), after a loss the backend saw, such as
    /// a UC completion error. A frame already handed over is never NACKed, so never handed over
    /// twice.
    pub fn nack(&mut self, seq: u64) {
        if seq >= self.expected
            && !self.is_held(seq)
            && !self.nacks.contains(&seq)
            && self.nacks.len() < MAX_NACKS
        {
            self.nacks.push(seq);
            self.stats.nacks_sent += 1;
            self.credit_due = true;
        }
    }

    /// Closes the channel once the peer has acknowledged everything sent (`Channel::close`).
    pub fn close(&mut self, reason: CloseReason) {
        self.closing.get_or_insert(reason);
    }

    /// Takes a credit from the peer.
    fn on_credit(&mut self, credit: &Credit, now: MonoTime) {
        if credit.reset {
            self.raise(Error::new(
                ErrorCode::PEER_RESET,
                "the receiver lost its channel state",
            ));
            return;
        }
        let Some(send) = self.send.as_mut() else {
            return;
        };
        // The echo first, while the frame is still retained.
        let mut sample = None;
        if credit.echo_seq != 0
            && let Some(sent_at) = send
                .retained
                .iter()
                .find(|retained| retained.chan_seq == credit.echo_seq && !retained.resent)
                .and_then(|retained| retained.sent_at)
        {
            let rtt = now
                .saturating_duration_since(sent_at)
                .0
                .saturating_sub(u64::from(credit.echo_delay_ns));
            sample = Some(Nanos(rtt / 2));
        }
        if credit.ack_seq > send.ack_seq && credit.ack_seq < send.next_seq {
            send.ack_seq = credit.ack_seq;
            while send
                .retained
                .front()
                .is_some_and(|retained| retained.chan_seq <= send.ack_seq)
            {
                // Dropping the frame releases its leases before its ticket is handed back.
                send.retained.pop_front();
            }
            send.resend.retain(|seq| *seq > send.ack_seq);
            send.last_progress = now;
            send.timeouts = 0;
        }
        // `limit_seq − ack_seq` is at most the peer's `recv_slots` (§7.1); a peer that grants
        // more gets no more.
        let limit = credit.limit_seq.min(
            send.ack_seq
                .saturating_add(u64::from(send.config.peer_recv_slots)),
        );
        if limit > send.limit_seq {
            send.limit_seq = limit;
            if send.next_seq <= send.limit_seq
                && let Some(since) = send.stalled_since.take()
            {
                self.stats.credit_stall =
                    Nanos(self.stats.credit_stall.0 + now.saturating_duration_since(since).0);
                send.last_probe = None;
                send.probes = 0;
            }
        }
        for &seq in credit.nacks.as_slice() {
            // Only a frame already on the wire is sent again; one not yet sent goes out once.
            if seq > send.ack_seq
                && send
                    .retained
                    .iter()
                    .any(|retained| retained.chan_seq == seq && retained.sent_at.is_some())
                && !send.resend.contains(&seq)
            {
                send.resend.push_back(seq);
            }
        }
        if let Some(sample) = sample {
            self.sample_alpha(sample);
        }
    }

    // ---- Receiving ----

    /// A free slot of the receive pool to receive the next frame into, or `None` when every slot
    /// is in use. Only the core hands out its pool's slots, so it sees each one come back.
    pub fn next_receive_slot(&mut self) -> Option<SlotWriter> {
        self.reap_released();
        // Any free slot, taken without a lock (RFC-0031 §8).
        let writer = self.pool.take()?;
        // A delivered frame dropped on another thread since the scan above: its slot is free,
        // so its credit returns now, before the slot holds another frame.
        if let Some(at) = self
            .given_out
            .iter()
            .position(|slot| *slot == writer.slot())
        {
            self.given_out.remove(at);
            self.credit_due |= self.grantable() > self.granted;
        }
        Some(writer)
    }

    /// For a backend that reads a byte stream (QUIC, TCP): checks a frame's base header before
    /// its body is read, and returns the whole frame's length. A data frame must fit this side's
    /// `slot_size`, and a control frame [`MAX_CONTROL_FRAME`]. A larger frame is a protocol
    /// error: the channel faults with `FrameTooLarge` (0x0205, `InvalidRequest`) and closes. The
    /// lengths are not yet under a checked CRC, but a stream transport delivers them intact, and
    /// a stream cannot be framed past a length it does not trust.
    pub fn admit(&mut self, header: &[u8; HEADER_LEN]) -> Result<usize, Error> {
        let ext_len = u32::from_le_bytes(header[48..52].try_into().expect("4 bytes"));
        let payload_len = u32::from_le_bytes(header[52..56].try_into().expect("4 bytes"));
        let len = HEADER_LEN as u64 + u64::from(ext_len) + u64::from(payload_len);
        let control = FrameType::from_code(header[5]).is_some_and(FrameType::is_control);
        let limit = if control {
            MAX_CONTROL_FRAME as u64
        } else {
            u64::from(self.slot_size)
        };
        if len > limit {
            let error = Error::new(
                ErrorCode::FRAME_TOO_LARGE,
                format!("a {len}-byte frame on a channel whose slots hold {limit}"),
            );
            self.raise(error.clone());
            return Err(error);
        }
        Ok(len as usize)
    }

    /// Takes the frame in `slot`'s first `len` bytes (§6.1, §7.2). After a fault the channel
    /// takes nothing more: it is closing.
    pub fn on_receive(&mut self, slot: SlotReader, len: usize, now: MonoTime) {
        if self.fault.is_some() {
            return;
        }
        let bytes = &slot.bytes()[..len.min(slot.bytes().len())];
        let frame = match Frame::decode_header_first(bytes) {
            Ok(frame) => frame,
            Err(
                FrameError::HeaderCrc { .. }
                | FrameError::BadMagic(_)
                | FrameError::Truncated { .. },
            ) => {
                // The header cannot be trusted, so neither can its `chan_seq`: NACK the next one.
                self.stats.crc_header_fail += 1;
                if self.recv_slots > 0 {
                    let expected = self.expected;
                    self.schedule_nack(expected, now);
                }
                return;
            }
            Err(error) => {
                // Every other refusal comes after the header CRC passed, so the key bytes are the
                // sender's. Another channel's frame is stale, whatever else its header holds;
                // this channel's is a protocol error, which closes the channel with the code
                // `FrameError` gives, such as `UnknownFrameType`.
                if raw_key(bytes) == (self.key.instance, self.key.epoch) {
                    self.raise(refusal(&error));
                } else {
                    self.stats.stale_dropped += 1;
                }
                return;
            }
        };
        let header = frame.header;
        let spec = header.spec;
        if spec.instance != self.key.instance || spec.epoch != self.key.epoch {
            // Another instance's or epoch's frame: dropped and counted before its minor, its
            // `chan_seq` or its payload is looked at, and never NACKed (§2, §7.2; RFC-0001
            // invariant 7).
            self.stats.stale_dropped += 1;
            return;
        }
        if spec.wire_minor > self.wire_minor {
            self.raise(Error::uncoded(
                ErrorClass::InvalidRequest,
                format!(
                    "a frame of wire minor {} on a channel that negotiated {} (RFC-0031 §6.1)",
                    spec.wire_minor, self.wire_minor
                ),
            ));
            return;
        }
        if spec.ftype.is_control() {
            if frame.verify_payload().is_err() {
                // Dropped and counted, never NACKed: a control frame has no `chan_seq`.
                self.stats.crc_payload_fail += 1;
                self.count_payload_failure(now);
                return;
            }
            match frame.control() {
                Ok(control) => self.on_control(&control, now),
                Err(error) => self.raise(refusal(&error)),
            }
            return;
        }
        if header.frame_len() > u64::from(self.slot_size) {
            self.raise(Error::new(
                ErrorCode::FRAME_TOO_LARGE,
                format!(
                    "a {}-byte frame on a channel whose slots hold {}",
                    header.frame_len(),
                    self.slot_size
                ),
            ));
            return;
        }
        let seq = spec.chan_seq;
        if seq < self.expected || self.is_held(seq) {
            // A duplicate, after a resend: the sender learns the ack from a fresh credit.
            self.credit_due = true;
            return;
        }
        if seq > self.granted {
            self.raise(Error::uncoded(
                ErrorClass::InvalidRequest,
                format!(
                    "frame {seq} is past the credit granted, {} (RFC-0031 §7.1)",
                    self.granted
                ),
            ));
            return;
        }
        if frame.verify_payload().is_err() {
            self.stats.crc_payload_fail += 1;
            self.schedule_nack(seq, now);
            self.count_payload_failure(now);
            return;
        }
        let held = Held {
            chan_seq: seq,
            header,
            reader: slot,
        };
        self.highest_seen = self.highest_seen.max(seq);
        if seq == self.expected {
            self.ready.push_back(held);
            self.expected += 1;
            while let Some(next) = self.take_held(self.expected) {
                self.ready.push_back(next);
                self.expected += 1;
            }
        } else {
            let index = (seq % u64::from(self.recv_slots)) as usize;
            self.held[index] = Some(held);
            self.held_count += 1;
            self.nack_gaps(now);
        }
        self.frames_since_credit += 1;
        self.echo = Some((seq, now));
        match self.cadence {
            CreditCadence::EveryFrame => self.credit_due = true,
            CreditCadence::Batched => {
                let every = (self.recv_slots / 4).max(1);
                if self.frames_since_credit >= every || self.below_half() {
                    self.credit_due = true;
                }
            }
        }
    }

    /// The next data frame in `chan_seq` order (`Channel::try_recv`).
    pub fn recv(&mut self) -> Option<InFrame> {
        let held = self.ready.pop_front()?;
        let header = held.header;
        let ext = ExtRef::new(held.reader.clone(), HEADER_LEN as u32, header.ext_len)
            .expect("the decoder checked the lengths against the slot");
        let payload = PayloadRef::new(
            held.reader.clone(),
            HEADER_LEN as u32 + header.ext_len,
            header.payload_len,
        )
        .expect("the decoder checked the lengths against the slot");
        self.given_out.push_back(held.reader.slot());
        drop(held);
        self.stats.frames_received += 1;
        self.stats.bytes_received += header.frame_len();
        Some(InFrame {
            header,
            ext,
            payload,
        })
    }

    fn is_held(&self, seq: u64) -> bool {
        self.recv_slots > 0
            && self.held[(seq % u64::from(self.recv_slots)) as usize]
                .as_ref()
                .is_some_and(|held| held.chan_seq == seq)
    }

    fn take_held(&mut self, seq: u64) -> Option<Held> {
        if !self.is_held(seq) {
            return None;
        }
        self.held_count -= 1;
        self.held[(seq % u64::from(self.recv_slots)) as usize].take()
    }

    /// The credit this side may grant: everything above `ack_seq` up to `recv_slots`, less the
    /// frames handed over or waiting to be, whose slots are still in use.
    fn grantable(&self) -> u64 {
        let in_use = (self.ready.len() + self.given_out.len()) as u64;
        (self.expected - 1) + u64::from(self.recv_slots).saturating_sub(in_use)
    }

    /// The receiving side has read everything that has arrived and is about to wait: a
    /// [`CreditCadence::Batched`] receiver sends the credit it was holding back for a fuller
    /// batch, if any, at the next [`poll_transmit`](Self::poll_transmit). Returns whether one is
    /// now due. A sender whose frames come too sparsely to fill a batch, such as a bulk channel
    /// held by its edge's pacer, then gets its acknowledgement at once rather than at its ack
    /// timeout, and resends nothing (RFC-0031 §7.1, note of 2026-10-31). A burst read in one
    /// pass still takes one credit per batch. A backend that reads a byte stream calls it before
    /// it waits; the simulator, which delivers each frame at its arrival time, does not.
    pub fn on_idle(&mut self) -> bool {
        if self.recv_slots == 0 || self.credit_due || self.fault.is_some() {
            return false;
        }
        self.reap_released();
        if self.frames_since_credit > 0 || self.grantable() > self.granted {
            self.credit_due = true;
        }
        self.credit_due
    }

    /// Whether the sender's remaining credit has fallen below half the window.
    fn below_half(&self) -> bool {
        let remaining = self
            .granted
            .saturating_sub(self.highest_seen.max(self.expected - 1));
        remaining * 2 < u64::from(self.recv_slots)
    }

    /// Finds the slots of handed-over frames the caller has dropped; their credit returns.
    fn reap_released(&mut self) {
        let before = self.given_out.len();
        let pool = &self.pool;
        self.given_out.retain(|slot| !pool.is_free(*slot));
        if self.given_out.len() != before && self.grantable() > self.granted {
            match self.cadence {
                CreditCadence::EveryFrame => self.credit_due = true,
                CreditCadence::Batched => {
                    if self.below_half()
                        || self.grantable() - self.granted
                            >= u64::from((self.recv_slots / 4).max(1))
                    {
                        self.credit_due = true;
                    }
                }
            }
        }
    }

    /// NACKs `seq`, unless it was NACKed within the last `nack_interval`.
    fn schedule_nack(&mut self, seq: u64, now: MonoTime) {
        if self.recv_slots == 0 {
            return;
        }
        let index = (seq % u64::from(self.recv_slots)) as usize;
        let (last_seq, last_at) = self.nacked[index];
        let recent = last_seq == seq
            && now.saturating_duration_since(last_at) < self.nack_interval()
            && last_at != MonoTime(0);
        if recent || self.nacks.contains(&seq) || self.nacks.len() >= MAX_NACKS {
            return;
        }
        self.nacked[index] = (seq, now.max(MonoTime(1)));
        self.nacks.push(seq);
        self.stats.nacks_sent += 1;
        self.credit_due = true;
    }

    /// NACKs each missing `chan_seq` below the highest seen, once per `nack_interval`.
    fn nack_gaps(&mut self, now: MonoTime) {
        let mut seq = self.expected;
        while seq < self.highest_seen {
            if !self.is_held(seq) {
                self.schedule_nack(seq, now);
            }
            seq += 1;
        }
    }

    fn count_payload_failure(&mut self, now: MonoTime) {
        if self.crc_storm_count == u32::MAX {
            return;
        }
        while self
            .payload_failures
            .front()
            .is_some_and(|at| now.saturating_duration_since(*at) >= CRC_STORM_WINDOW)
        {
            self.payload_failures.pop_front();
        }
        self.payload_failures.push_back(now);
        let count = self.crc_storm_count as usize;
        if self.payload_failures.len() >= count {
            self.raise(Error::new(
                ErrorCode::CRC_STORM,
                format!(
                    "{count} payload CRC failures within {} s (RFC-0031 §7.2)",
                    CRC_STORM_WINDOW.0 / 1_000_000_000
                ),
            ));
        }
        while self.payload_failures.len() > count {
            self.payload_failures.pop_front();
        }
    }

    fn on_control(&mut self, control: &Control<'_>, now: MonoTime) {
        match control {
            Control::Credit(credit) => self.on_credit(credit, now),
            Control::Ping(ping) if !ping.reply => {
                if self.ping_replies.len() < MAX_PING_REPLIES {
                    self.ping_replies.push_back(ping.origin_ns);
                }
                // A data receiver answers with its current credit too: the ping may be a stalled
                // sender's probe for a credit that was lost.
                if self.recv_slots > 0 {
                    self.credit_due = true;
                }
            }
            Control::Ping(ping) => {
                let sent = self
                    .pings_out
                    .iter_mut()
                    .find(|out| **out == Some(ping.origin_ns));
                if let Some(out) = sent {
                    *out = None;
                    let rtt = now.0.saturating_sub(ping.origin_ns);
                    self.sample_alpha(Nanos(rtt / 2));
                }
            }
            Control::Close { close, .. } => {
                self.peer_close.get_or_insert(*close);
            }
            // The handshake ran before the core opened; a late or repeated one changes nothing.
            Control::Hello { .. } | Control::HelloAck { .. } => {}
        }
    }

    // ---- Timers and the wire ----

    /// Runs the timers: the ack timeout, which sends the oldest unacknowledged frame again
    /// (§7.3); the credit probe of a stalled sender with nothing unacknowledged; and the NACKs of
    /// frames still missing (§7.2). Also finds released slots.
    ///
    /// The probe covers what the ack timeout cannot. A credit that only grants more, sent when
    /// the receiver's caller drops a frame, may be lost while the sender has nothing
    /// unacknowledged: no later credit and no ack timeout would follow. So a sender whose
    /// `send` was refused for credit, with nothing unacknowledged, sends a `Ping` after an ack
    /// timeout, then after 2, 4, … up to [`PROBE_BACKOFF_MAX`] ack timeouts, until a credit lets
    /// it send; a data receiver answers a `Ping` with its current credit.
    pub fn on_tick(&mut self, now: MonoTime) {
        self.reap_released();
        if self.highest_seen > self.expected {
            self.nack_gaps(now);
        }
        let mut fault = false;
        if self.ack_due().is_some_and(|due| now >= due)
            && let Some(send) = self.send.as_mut()
            && let Some(oldest) = send.retained.front()
        {
            let seq = oldest.chan_seq;
            if !send.resend.contains(&seq) {
                send.resend.push_back(seq);
            }
            send.last_progress = now;
            send.timeouts += 1;
            fault = send.timeouts >= self.ack_timeouts_to_fault;
        }
        if self.probe_due_at().is_some_and(|due| now >= due)
            && let Some(send) = self.send.as_mut()
        {
            send.last_probe = Some(now);
            send.probes = send.probes.saturating_add(1);
            self.probe_due = true;
        }
        if fault {
            self.raise(Error::new(
                ErrorCode::ACK_TIMEOUT,
                "no ack within the ack timeout, repeatedly (RFC-0031 §7.3)",
            ));
        }
    }

    /// When the ack timer of the oldest frame on the wire runs out (§7.3).
    fn ack_due(&self) -> Option<MonoTime> {
        let send = self.send.as_ref()?;
        send.retained.front()?.sent_at?;
        Some(send.last_progress.saturating_add(self.ack_timeout()))
    }

    /// When a sender refused for credit, with nothing unacknowledged, sends its next probe.
    fn probe_due_at(&self) -> Option<MonoTime> {
        let send = self.send.as_ref()?;
        let since = send.stalled_since?;
        if !send.retained.is_empty() {
            return None;
        }
        let backoff = 1u64 << send.probes.min(PROBE_BACKOFF_MAX.trailing_zeros());
        let wait = Nanos(self.ack_timeout().0.saturating_mul(backoff));
        Some(send.last_probe.unwrap_or(since).saturating_add(wait))
    }

    /// When [`on_tick`](Self::on_tick) next has work, for a driver that steps on events: the
    /// ack timeout of the oldest frame on the wire, a stalled sender's next probe, or the next
    /// NACK of a missing frame, whichever comes first. A time already past means now. `None`
    /// when no timer runs, and after a fault. A received frame the caller drops is not a timer:
    /// its credit goes out at the next [`poll_transmit`](Self::poll_transmit).
    pub fn next_deadline(&self) -> Option<MonoTime> {
        if self.fault.is_some() {
            return None;
        }
        let mut next = [self.ack_due(), self.probe_due_at()]
            .into_iter()
            .flatten()
            .min();
        if self.recv_slots > 0 {
            let interval = self.nack_interval();
            for seq in self.expected..self.highest_seen {
                if self.is_held(seq) {
                    continue;
                }
                let (last_seq, last_at) = self.nacked[(seq % u64::from(self.recv_slots)) as usize];
                let due = if last_seq == seq && last_at != MonoTime(0) {
                    last_at.saturating_add(interval)
                } else {
                    MonoTime(0)
                };
                next = Some(next.map_or(due, |next| next.min(due)));
            }
        }
        next
    }

    /// The next frame to put on the wire, or `None`: control frames first (the credit, ping
    /// replies, the close), then frames sent again, then new ones.
    pub fn poll_transmit(&mut self, now: MonoTime) -> Option<Transmit<'_>> {
        if let Some(fault) = &self.fault {
            if self.fault_close_sent {
                return None;
            }
            self.fault_close_sent = true;
            let close = Close {
                code: fault.code.map_or(frame::code::OK, |code| code.0),
                last_seq: self.send.as_ref().map_or(0, |send| send.next_seq - 1),
            };
            let reason = CloseReason::new(fault.code, &fault.message);
            return self.encode_control(&Control::Close {
                close,
                reason: reason.text(),
            });
        }
        self.reap_released();
        if self.credit_due && self.recv_slots > 0 {
            self.credit_due = false;
            self.frames_since_credit = 0;
            self.granted = self.granted.max(self.grantable());
            let (echo_seq, echo_delay_ns) = match self.echo.take() {
                Some((seq, at)) => (
                    seq,
                    now.saturating_duration_since(at).0.min(u64::from(u32::MAX)) as u32,
                ),
                None => (0, 0),
            };
            let nacks = Nacks::new(&self.nacks).expect("at most MAX_NACKS");
            self.nacks.clear();
            let credit = Credit {
                ack_seq: self.expected - 1,
                limit_seq: self.granted,
                echo_seq,
                echo_delay_ns,
                sack: false,
                reset: false,
                nacks,
            };
            return self.encode_control(&Control::Credit(credit));
        }
        if let Some(origin_ns) = self.ping_replies.pop_front() {
            return self.encode_control(&Control::Ping(Ping {
                origin_ns,
                reply: true,
            }));
        }
        if self.probe_due {
            self.probe_due = false;
            return self.ping(now);
        }
        let all_acked = self
            .send
            .as_ref()
            .is_none_or(|send| send.retained.is_empty());
        if !self.close_sent
            && all_acked
            && let Some(reason) = self.closing.clone()
        {
            self.close_sent = true;
            let close = Close {
                code: reason.wire_code(),
                last_seq: self.send.as_ref().map_or(0, |send| send.next_seq - 1),
            };
            return self.encode_control(&Control::Close {
                close,
                reason: reason.text(),
            });
        }
        let send = self.send.as_mut()?;
        loop {
            let (seq, again) = match send.resend.pop_front() {
                Some(seq) => (seq, true),
                None => (send.unsent.pop_front()?, false),
            };
            // Retained frames are `ack_seq + 1 ..`, one per `chan_seq`.
            let Some(index) = seq
                .checked_sub(send.ack_seq + 1)
                .and_then(|index| usize::try_from(index).ok())
                .filter(|index| *index < send.retained.len())
            else {
                continue;
            };
            let retained = &mut send.retained[index];
            if again {
                retained.resent = true;
                self.stats.retransmits += 1;
            } else {
                retained.sent_at = Some(now);
                if index == 0 {
                    // The ack timer runs from the oldest unacknowledged frame's send (§7.3).
                    send.last_progress = now;
                }
            }
            let retained = &send.retained[index];
            return Some(Transmit::Data {
                chan_seq: seq,
                header: &retained.header,
                ext: retained.ext.bytes(),
                payload: retained.payload.bytes(),
            });
        }
    }

    /// Sends a ping, whose reply gives an α sample.
    pub fn ping(&mut self, now: MonoTime) -> Option<Transmit<'_>> {
        self.pings_out[self.ping_next] = Some(now.0);
        self.ping_next = (self.ping_next + 1) % PINGS_OUT;
        self.encode_control(&Control::Ping(Ping {
            origin_ns: now.0,
            reply: false,
        }))
    }

    fn encode_control(&mut self, control: &Control<'_>) -> Option<Transmit<'_>> {
        match control.encode(&self.ctl, &mut self.out) {
            Ok(len) => Some(Transmit::Control(&self.out[..len])),
            Err(_) => None,
        }
    }
}

impl std::fmt::Debug for ChannelCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelCore")
            .field("key", &self.key)
            .field("expected", &self.expected)
            .field("held", &self.held_count)
            .field("ready", &self.ready.len())
            .field("given_out", &self.given_out.len())
            .field("granted", &self.granted)
            .field("credit", &self.credit())
            .field("fault", &self.fault)
            .finish_non_exhaustive()
    }
}
