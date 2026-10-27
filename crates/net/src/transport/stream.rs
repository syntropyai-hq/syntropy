// SPDX-License-Identifier: MIT OR Apache-2.0

//! A channel over an ordered byte stream, such as a QUIC stream or a TCP connection: the opening
//! handshake, the `net` task that drives the channel core, and the backend a dispatcher calls
//! (RFC-0031 §2, §4.4, §6.9, §7; ADR-0012 item 4).
//!
//! ```text
//! dispatcher ──try_send──▶ out ring ──▶ net task: ChannelCore::send, poll_transmit, stream.write
//! net task: stream.read ──▶ a receive slot ──▶ ChannelCore::on_receive, recv ──▶ in ring ──▶ try_recv
//! net task: ChannelCore::completed ──▶ done ring ──▶ completed
//! ```
//!
//! - **The core lives in the net task.** A [`ChannelCore`] does no I/O and reads no clock; the
//!   task moves bytes and passes the time in. The dispatcher never awaits and never takes a lock:
//!   it pushes and pops `syntropy-spsc` rings, and reads the credit, the state and the counters
//!   from atomics the task publishes.
//! - **`try_send` decides at once.** It numbers the frame and checks it as
//!   [`ChannelCore::send`] would: closed, link fault, a control frame type, an extension that is
//!   not a multiple of 64 bytes, a frame larger than the peer's slot, and the credit. The credit
//!   it checks is the core's, published as the last `chan_seq` it allows, so the dispatcher's
//!   view lags but never runs ahead. The frame then goes to the core in order, which numbers it
//!   the same.
//! - **The one host copy on receive** is the read from the stream into a receive slot of the
//!   channel's pool (RFC-0031 §4.4, §8). The task reads a frame's 64-byte header, asks the core for
//!   its length ([`ChannelCore::admit`]), and reads the rest into the slot.
//! - **Closing.** [`Channel::close`] travels on the out ring behind the frames already sent, so
//!   they arrive first. The task finishes its half of the stream once its `Close` has gone, and
//!   ends when the peer has finished its half. A dropped [`Channel`] closes the same way.
//!
//! **The handshake** (§6.9). The side that connects sends `Hello`, with its key, wire dtype,
//! receive slots and slot size; the side that accepts answers `HelloAck` with its own, or a
//! refusal with an RFC-0031 §12 code. Each side's credential travels as the frame's payload.
//! Checking the peer's credential, the plan-neighbour and epoch checks of ADR-0308 item 2, is
//! DIST-E7-T04's; this module asks a [`ChannelAuth`] for both sides' credentials and for the
//! check, and [`UncheckedAuth`] stands in until then.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use syntropy_core::plan::WireDType;
use syntropy_core::{ErrorClass, MonoTime, Nanos};
use syntropy_spsc::{Consumer, PopError, Producer, PushError};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use super::buffer::{BufferPool, ExtRef, PayloadRef, SlotWriter};
use super::channel::{
    Channel, ChannelBackend, ChannelKey, ChannelPurpose, ChannelStats, CloseReason, InFrame,
    OutFrame, SendError, SendTicket,
};
use super::core::{
    ACK_TIMEOUTS_TO_FAULT, CONTROL_RECEIVES, CRC_STORM_COUNT, ChannelCore, CoreConfig,
    CreditCadence, MAX_CONTROL_FRAME, SendConfig, Transmit,
};
use super::domain::{DomainHandle, DomainState, wake};
use crate::error::{Error, ErrorCode};
use crate::frame::{
    self, Control, ControlSpec, CredKind, Flags, Frame, FrameType, HEADER_LEN, Hello, HelloAck,
    Purpose, WIRE_MINOR,
};

/// The protocol major this build speaks: the `1` of `syntropy/data/1` (RFC-0031 §1, §10).
pub const PROTO_MAJOR: u16 = 1;

/// How long a channel's task waits between timer runs while it is parked: the runtime timer's
/// resolution. The core's own timers, the ack timeout and the NACK interval, are at least
/// 200 µs and 100 µs (§7.2, §7.3); a reliable stream rarely needs them.
const TICK: Duration = Duration::from_millis(1);

/// What this side sends as its credential and how it checks the peer's (RFC-0034 §6,
/// ADR-0308 items 1–2). DIST-E7-T04 implements it with the runner's `DataPlaneCredential` and
/// the plan-neighbour, epoch, expiry and revocation checks.
pub trait ChannelAuth: Send + Sync + 'static {
    /// This side's credential for the channel `key`: its kind and bytes.
    fn credential(&self, key: &ChannelKey) -> (CredKind, Vec<u8>);

    /// Checks the peer's credential on the channel `key`. `peer` is the key the peer's TLS
    /// handshake proved. An error refuses the channel with that RFC-0031 §12 code.
    fn check(
        &self,
        peer: &[u8; 32],
        key: &ChannelKey,
        kind: CredKind,
        credential: &[u8],
    ) -> Result<(), ErrorCode>;
}

/// The stand-in until DIST-E7-T04: a zeroed `DataPlaneCredential` and no check. The connection's
/// TLS handshake still proves each side's endpoint key; nothing checks that the key belongs to
/// a plan neighbour, so no runner opens channels with it in a release.
#[derive(Clone, Copy, Debug, Default)]
pub struct UncheckedAuth;

impl ChannelAuth for UncheckedAuth {
    fn credential(&self, _key: &ChannelKey) -> (CredKind, Vec<u8>) {
        let kind = CredKind::DataPlaneCredential;
        (kind, vec![0; kind.credential_len()])
    }

    fn check(
        &self,
        _peer: &[u8; 32],
        _key: &ChannelKey,
        _kind: CredKind,
        _credential: &[u8],
    ) -> Result<(), ErrorCode> {
        Ok(())
    }
}

/// One side of a channel to open over a stream: its spec, less the pool registration, which the
/// transport has resolved to `pool`.
#[derive(Clone, Debug)]
pub struct StreamSpec {
    pub key: ChannelKey,
    /// This side's receive pool: at least `recv_slots + 4` slots, each of at least `slot_size`
    /// bytes and [`MAX_CONTROL_FRAME`].
    pub pool: BufferPool,
    /// The largest frame this side accepts, header included.
    pub slot_size: u32,
    /// The data frames this side accepts at once: the credit it grants. 0 on a side that only
    /// sends.
    pub recv_slots: u16,
    /// The channel's wire dtype, checked against the peer's on an `Activations` channel.
    pub wire: WireDType,
    /// Frames kept for retransmission; raised to the peer's `recv_slots` if smaller.
    pub retention: u16,
    /// The capability set this side advertises (RFC-0031 §10).
    pub capabilities: u64,
    /// The link's α p99 before any sample, from the plan's link profile.
    pub alpha_p99: Nanos,
    /// The pools frames this side sends may take their extension and payload from.
    pub send_pools: Vec<BufferPool>,
    /// The largest frame the transport carries (`TransportCaps::max_frame`).
    pub max_frame: u64,
}

/// What both sides agreed in the handshake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Agreed {
    pub wire_minor: u8,
    pub capabilities: u64,
    pub peer_recv_slots: u16,
    pub peer_slot_size: u32,
}

/// A `Hello` read from a stream, with the credential that came with it.
#[derive(Clone, Debug)]
pub struct HelloIn {
    pub hello: Hello,
    pub credential: Vec<u8>,
}

impl HelloIn {
    /// The channel the `Hello` opens, or `None` for a blob stream, which is not a channel.
    pub fn key(&self) -> Option<ChannelKey> {
        let purpose = ChannelPurpose::try_from(self.hello.purpose).ok()?;
        Some(ChannelKey {
            instance: self.hello.instance,
            epoch: self.hello.epoch,
            purpose,
            from: self.hello.from_stage,
            to: self.hello.to_stage,
            lane: self.hello.lane,
        })
    }
}

fn ctl_spec(key: &ChannelKey, wire_minor: u8) -> ControlSpec {
    ControlSpec {
        wire_minor,
        flags: Flags(0),
        instance: key.instance,
        epoch: key.epoch,
    }
}

fn wire_dtype(spec: &StreamSpec) -> Option<WireDType> {
    (spec.key.purpose == ChannelPurpose::Activations).then_some(spec.wire)
}

fn link(message: impl Into<String>) -> Error {
    Error::uncoded(ErrorClass::Link, message)
}

/// Reads one whole control frame from `r`, into a buffer of its own: the handshake's frames only,
/// before the channel's pool is in use.
async fn read_control<R: AsyncRead + Unpin>(r: &mut R) -> Result<Vec<u8>, Error> {
    let mut header = [0u8; HEADER_LEN];
    r.read_exact(&mut header)
        .await
        .map_err(|error| link(format!("the stream closed during the handshake: {error}")))?;
    let ext_len = u32::from_le_bytes(header[48..52].try_into().expect("4 bytes"));
    let payload_len = u32::from_le_bytes(header[52..56].try_into().expect("4 bytes"));
    let len = HEADER_LEN as u64 + u64::from(ext_len) + u64::from(payload_len);
    if len > MAX_CONTROL_FRAME as u64 {
        return Err(Error::new(
            ErrorCode::FRAME_TOO_LARGE,
            format!("a {len}-byte handshake frame"),
        ));
    }
    let mut frame = vec![0u8; len as usize];
    frame[..HEADER_LEN].copy_from_slice(&header);
    r.read_exact(&mut frame[HEADER_LEN..])
        .await
        .map_err(|error| link(format!("the stream closed during the handshake: {error}")))?;
    Ok(frame)
}

fn refused(error: frame::FrameError) -> Error {
    Error::uncoded(
        ErrorClass::InvalidRequest,
        format!("a malformed handshake frame: {error}"),
    )
}

/// The connecting side's handshake: sends `Hello` and reads `HelloAck` (RFC-0031 §6.9).
///
/// # Errors
///
/// The peer's refusal, with its code; a malformed or unexpected answer; the stream failing.
pub async fn offer<R, W>(
    r: &mut R,
    w: &mut W,
    spec: &StreamSpec,
    peer: &[u8; 32],
    auth: &dyn ChannelAuth,
) -> Result<Agreed, Error>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let (cred_kind, credential) = auth.credential(&spec.key);
    let hello = Control::Hello {
        hello: Hello {
            purpose: Purpose::from(spec.key.purpose),
            wire_minor_max: WIRE_MINOR,
            proto_major: PROTO_MAJOR,
            capabilities: spec.capabilities,
            instance: spec.key.instance,
            epoch: spec.key.epoch,
            from_stage: spec.key.from,
            to_stage: spec.key.to,
            lane: spec.key.lane,
            wire_dtype: wire_dtype(spec),
            slot_size: spec.slot_size,
            recv_slots: spec.recv_slots,
            cred_kind,
            channel_token: 0,
            resume_from: None,
        },
        credential: &credential,
    };
    let bytes = hello
        .encode_to_vec(&ctl_spec(&spec.key, WIRE_MINOR))
        .map_err(|error| {
            Error::uncoded(ErrorClass::Internal, format!("encoding Hello: {error}"))
        })?;
    w.write_all(&bytes)
        .await
        .map_err(|error| link(format!("sending Hello: {error}")))?;
    w.flush()
        .await
        .map_err(|error| link(format!("sending Hello: {error}")))?;

    let answer = read_control(r).await?;
    let frame = Frame::decode(&answer).map_err(refused)?;
    let Control::HelloAck { ack, credential } = frame.control().map_err(refused)? else {
        return Err(Error::uncoded(
            ErrorClass::InvalidRequest,
            "the peer did not answer Hello with HelloAck",
        ));
    };
    if ack.status != frame::code::OK {
        let message = format!("the peer refused the channel {:?}", spec.key);
        return Err(match ErrorCode(ack.status).class() {
            Some(_) => Error::new(ErrorCode(ack.status), message),
            None => Error::uncoded(ErrorClass::Unsupported, message),
        });
    }
    if ack.wire_minor > WIRE_MINOR || ack.wire_minor == 0 {
        return Err(Error::new(
            ErrorCode::PROTO_RANGE_DISJOINT,
            format!("the peer answered wire minor {}", ack.wire_minor),
        ));
    }
    let Some(kind) = ack.cred_kind else {
        return Err(Error::new(
            ErrorCode::CREDENTIAL_INVALID,
            "the peer accepted without a credential",
        ));
    };
    auth.check(peer, &spec.key, kind, credential)
        .map_err(|code| Error::new(code, "the peer's credential was refused"))?;
    Ok(Agreed {
        wire_minor: ack.wire_minor,
        capabilities: ack.capabilities & spec.capabilities,
        peer_recv_slots: ack.recv_slots,
        peer_slot_size: ack.slot_size,
    })
}

/// The accepting side's first read: the stream's `Hello`.
///
/// # Errors
///
/// A malformed first frame, or one that is not `Hello`; the stream failing.
pub async fn read_hello<R: AsyncRead + Unpin>(r: &mut R) -> Result<HelloIn, Error> {
    let bytes = read_control(r).await?;
    let frame = Frame::decode(&bytes).map_err(refused)?;
    match frame.control().map_err(refused)? {
        Control::Hello { hello, credential } => Ok(HelloIn {
            hello,
            credential: credential.to_vec(),
        }),
        _ => Err(Error::uncoded(
            ErrorClass::InvalidRequest,
            "the stream's first frame is not Hello",
        )),
    }
}

/// Checks a `Hello` against the accepting side's `spec` (RFC-0031 §6.9): the protocol major, the
/// wire minors, the wire dtype of an `Activations` channel, then the peer's credential. `Err` is
/// the §12 code the refusal carries.
pub fn check_hello(
    hello: &HelloIn,
    spec: &StreamSpec,
    peer: &[u8; 32],
    auth: &dyn ChannelAuth,
) -> Result<Agreed, ErrorCode> {
    let h = &hello.hello;
    if h.proto_major != PROTO_MAJOR {
        return Err(ErrorCode::PROTO_MAJOR_MISMATCH);
    }
    if h.wire_minor_max == 0 {
        return Err(ErrorCode::PROTO_RANGE_DISJOINT);
    }
    if let (Some(theirs), Some(ours)) = (h.wire_dtype, wire_dtype(spec))
        && theirs != ours
    {
        return Err(ErrorCode::WIRE_DTYPE_MISMATCH);
    }
    auth.check(peer, &spec.key, h.cred_kind, &hello.credential)?;
    Ok(Agreed {
        wire_minor: h.wire_minor_max.min(WIRE_MINOR),
        capabilities: h.capabilities & spec.capabilities,
        peer_recv_slots: h.recv_slots,
        peer_slot_size: h.slot_size,
    })
}

/// The accepting side's answer: `HelloAck` with `status` (OK or a §12 code). An acceptance
/// carries this side's slots and credential; a refusal carries neither.
///
/// # Errors
///
/// The stream failing.
pub async fn answer<W: AsyncWrite + Unpin>(
    w: &mut W,
    spec: &StreamSpec,
    agreed: Result<&Agreed, ErrorCode>,
    auth: &dyn ChannelAuth,
) -> Result<(), Error> {
    let (status, wire_minor, capabilities) = match agreed {
        Ok(agreed) => (frame::code::OK, agreed.wire_minor, agreed.capabilities),
        Err(code) => (code.0, WIRE_MINOR, 0),
    };
    let (cred_kind, credential) = if agreed.is_ok() {
        let (kind, bytes) = auth.credential(&spec.key);
        (Some(kind), bytes)
    } else {
        (None, Vec::new())
    };
    let ack = Control::HelloAck {
        ack: HelloAck {
            status,
            wire_minor,
            capabilities,
            limit_seq: u64::from(spec.recv_slots),
            slot_size: spec.slot_size,
            recv_slots: spec.recv_slots,
            cred_kind,
        },
        credential: &credential,
    };
    let bytes = ack
        .encode_to_vec(&ctl_spec(&spec.key, wire_minor))
        .map_err(|error| {
            Error::uncoded(ErrorClass::Internal, format!("encoding HelloAck: {error}"))
        })?;
    w.write_all(&bytes)
        .await
        .map_err(|error| link(format!("sending HelloAck: {error}")))?;
    w.flush()
        .await
        .map_err(|error| link(format!("sending HelloAck: {error}")))
}

// ---- The channel's shared state ----

const OPEN: u8 = 0;
/// The peer closed the channel, or its stream ended.
const PEER_CLOSED: u8 = 1;
/// A link fault or a protocol fault: the core's fault or a stream error.
const FAULTED: u8 = 2;

/// What the net task publishes for the dispatcher: atomics only.
#[derive(Debug, Default)]
struct Shared {
    /// The last `chan_seq` the core allows: its credit plus the frames it has taken.
    credit_end: AtomicU64,
    state: AtomicU8,
    stats: StatsCells,
}

/// [`ChannelStats`] as atomics, written by the net task. `alpha_last` holds `u64::MAX` for none.
#[derive(Debug, Default)]
struct StatsCells([AtomicU64; 13]);

impl StatsCells {
    fn store(&self, stats: &ChannelStats) {
        let values = [
            stats.frames_sent,
            stats.bytes_sent,
            stats.frames_received,
            stats.bytes_received,
            stats.retransmits,
            stats.nacks_sent,
            stats.credit_stalls,
            stats.credit_stall.0,
            stats.alpha_samples,
            stats.alpha_last.map_or(u64::MAX, |alpha| alpha.0),
            stats.crc_header_fail,
            stats.crc_payload_fail,
            stats.stale_dropped,
        ];
        for (cell, value) in self.0.iter().zip(values) {
            cell.store(value, Ordering::Relaxed);
        }
    }

    fn load(&self) -> ChannelStats {
        let v: [u64; 13] = std::array::from_fn(|i| self.0[i].load(Ordering::Relaxed));
        ChannelStats {
            frames_sent: v[0],
            bytes_sent: v[1],
            frames_received: v[2],
            bytes_received: v[3],
            retransmits: v[4],
            nacks_sent: v[5],
            credit_stalls: v[6],
            credit_stall: Nanos(v[7]),
            alpha_samples: v[8],
            alpha_last: (v[9] != u64::MAX).then_some(Nanos(v[9])),
            crc_header_fail: v[10],
            crc_payload_fail: v[11],
            stale_dropped: v[12],
        }
    }
}

/// What the dispatcher sends the net task, in order.
enum Outbound {
    Frame {
        ftype: FrameType,
        step: u64,
        flags: Flags,
        ext: ExtRef,
        payload: PayloadRef,
    },
    Nack(u64),
    Close(CloseReason),
}

// ---- The dispatcher's side ----

/// The [`ChannelBackend`] a dispatcher calls: rings and atomics, no lock (ADR-0012 item 4).
struct HotSide {
    out: Producer<Outbound>,
    inbound: Consumer<InFrame>,
    done: Consumer<SendTicket>,
    shared: Arc<Shared>,
    send_pools: Vec<BufferPool>,
    /// The largest frame the peer accepts, header included, within the transport's maximum.
    send_limit: u64,
    sends: bool,
    /// The last `chan_seq` handed out.
    assigned: u64,
    closed: bool,
    /// Refused sends, and the time from each refusal to the send that got through.
    credit_stalls: u64,
    credit_stall: u64,
    stalled_since: Option<Instant>,
}

impl HotSide {
    fn pool(&self, id: super::buffer::PoolId) -> Option<&BufferPool> {
        self.send_pools.iter().find(|pool| pool.id() == id)
    }
}

impl ChannelBackend for HotSide {
    fn try_send(&mut self, frame: OutFrame) -> Result<SendTicket, SendError> {
        if self.closed {
            return Err(SendError::Closed);
        }
        match self.shared.state.load(Ordering::Acquire) {
            OPEN => {}
            PEER_CLOSED => return Err(SendError::Closed),
            _ => return Err(SendError::Link),
        }
        if !self.sends
            || frame.ftype.is_control()
            || !(frame.ext.len as usize).is_multiple_of(frame::ALIGN)
        {
            return Err(SendError::Invalid);
        }
        let len = HEADER_LEN as u64 + u64::from(frame.ext.len) + u64::from(frame.payload.len);
        if len > self.send_limit {
            return Err(SendError::TooLarge);
        }
        if self.assigned >= self.shared.credit_end.load(Ordering::Acquire) {
            self.credit_stalls += 1;
            self.stalled_since.get_or_insert_with(Instant::now);
            return Err(SendError::WouldBlock);
        }
        let ext = self
            .pool(frame.ext.pool)
            .and_then(|pool| pool.retain_ext(frame.ext));
        let payload = self
            .pool(frame.payload.pool)
            .and_then(|pool| pool.retain_payload(frame.payload));
        let (Some(ext), Some(payload)) = (
            ext.or_else(|| frame.ext.is_empty().then(ExtRef::empty)),
            payload.or_else(|| frame.payload.is_empty().then(PayloadRef::empty)),
        ) else {
            return Err(SendError::Invalid);
        };
        match self.out.push(Outbound::Frame {
            ftype: frame.ftype,
            step: frame.step,
            flags: frame.flags,
            ext,
            payload,
        }) {
            Ok(()) => {}
            // The ring holds the credit window and more, so a full ring is a stall too.
            Err(PushError::Full(_)) => {
                self.credit_stalls += 1;
                self.stalled_since.get_or_insert_with(Instant::now);
                return Err(SendError::WouldBlock);
            }
            Err(PushError::Closed(_)) => return Err(SendError::Link),
        }
        if let Some(since) = self.stalled_since.take() {
            self.credit_stall += since.elapsed().as_nanos() as u64;
        }
        self.assigned += 1;
        Ok(SendTicket {
            chan_seq: self.assigned,
        })
    }

    fn try_recv(&mut self) -> Option<InFrame> {
        self.inbound.pop().ok()
    }

    fn completed(&mut self) -> Option<SendTicket> {
        self.done.pop().ok()
    }

    fn stats(&self) -> ChannelStats {
        let mut stats = self.shared.stats.load();
        stats.credit_stalls += self.credit_stalls;
        stats.credit_stall = Nanos(stats.credit_stall.0 + self.credit_stall);
        stats
    }

    fn credit(&self) -> u32 {
        if self.closed || self.shared.state.load(Ordering::Acquire) != OPEN {
            return 0;
        }
        let end = self.shared.credit_end.load(Ordering::Acquire);
        end.saturating_sub(self.assigned).min(u64::from(u32::MAX)) as u32
    }

    fn nack(&mut self, seq: u64) {
        // A NACK is advice: a full ring drops it, and the core's own NACKs still run.
        let _ = self.out.push(Outbound::Nack(seq));
    }

    fn close(&mut self, reason: CloseReason) {
        if self.closed {
            return;
        }
        self.closed = true;
        // The ring holds the credit window and more; if it is full the task closes the channel
        // anyway once the dispatcher drops it.
        let _ = self.out.push(Outbound::Close(reason));
    }
}

// ---- The net task ----

/// A channel opened over `r` and `w` after the handshake: the dispatcher's [`Channel`] and the
/// task that drives it, which [`Opened::spawn`] runs in the domain.
pub struct Opened<R, W> {
    pub channel: Channel,
    driver: Driver<R, W>,
}

impl<R, W> Opened<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    /// Runs the channel's task in `domain`, and returns the channel.
    pub fn spawn(self, domain: &DomainHandle) -> Channel {
        self.spawn_with(domain, ())
    }

    /// [`Opened::spawn`], keeping `keep` alive until the task ends and then dropping it: a
    /// backend's hold on what the channel uses, such as its share of a connection's window.
    pub fn spawn_with<K: Send + 'static>(self, domain: &DomainHandle, keep: K) -> Channel {
        let Opened { channel, driver } = self;
        domain.spawn(async move {
            driver.run().await;
            drop(keep);
        });
        channel
    }
}

/// Builds the channel `spec.key` over `r` and `w` once the handshake agreed `agreed`.
///
/// # Errors
///
/// A pool too small for the receive slots, or a ring that cannot be made.
pub fn open<R, W>(
    r: R,
    w: W,
    spec: StreamSpec,
    agreed: Agreed,
    domain: &DomainHandle,
) -> Result<Opened<R, W>, Error> {
    let sends = agreed.peer_recv_slots > 0;
    let retention = spec.retention.max(agreed.peer_recv_slots);
    let core = ChannelCore::new(CoreConfig {
        key: spec.key,
        wire_minor: agreed.wire_minor,
        cadence: CreditCadence::for_purpose(spec.key.purpose),
        alpha_p99: spec.alpha_p99,
        recv_slots: spec.recv_slots,
        slot_size: spec.slot_size,
        pool: spec.pool.clone(),
        send: sends.then_some(SendConfig {
            peer_recv_slots: agreed.peer_recv_slots,
            peer_slot_size: agreed.peer_slot_size,
            retention,
        }),
        ack_timeouts_to_fault: ACK_TIMEOUTS_TO_FAULT,
        crc_storm_count: CRC_STORM_COUNT,
    })?;
    let out_capacity = usize::from(retention) + 16;
    let (out_tx, out_rx) = wake::ring(out_capacity.next_power_of_two()).map_err(|error| {
        Error::uncoded(ErrorClass::Internal, format!("a channel ring: {error}"))
    })?;
    let (in_tx, in_rx) = ring(usize::from(spec.recv_slots) + usize::from(CONTROL_RECEIVES))?;
    let (done_tx, done_rx) = ring(usize::from(retention) + 1)?;
    let shared = Arc::new(Shared::default());
    let epoch = Instant::now();
    shared
        .credit_end
        .store(u64::from(core.credit()), Ordering::Release);
    shared.stats.store(&core.stats());
    let send_limit = u64::from(agreed.peer_slot_size).min(spec.max_frame);
    let hot = HotSide {
        out: out_tx,
        inbound: in_rx,
        done: done_rx,
        shared: Arc::clone(&shared),
        send_pools: spec.send_pools.clone(),
        send_limit,
        sends,
        assigned: 0,
        closed: false,
        credit_stalls: 0,
        credit_stall: 0,
        stalled_since: None,
    };
    let driver = Driver {
        core,
        r,
        w,
        out: out_rx,
        ready: None,
        inbound: in_tx,
        done: done_tx,
        shared,
        domain: Arc::clone(&domain.state),
        epoch,
        reading: Reading::Header {
            buf: [0; HEADER_LEN],
            have: 0,
        },
        held_in: None,
        held_ticket: None,
        close_sent: false,
        write_done: false,
        read_done: false,
        hot_gone: false,
        last_busy: epoch,
        last_tick: epoch,
    };
    Ok(Opened {
        channel: Channel::new(spec.key, Box::new(hot)),
        driver,
    })
}

/// A ring of at least `capacity` slots between the task and the dispatcher, which polls it.
fn ring<T>(capacity: usize) -> Result<(Producer<T>, Consumer<T>), Error> {
    syntropy_spsc::ring(capacity.next_power_of_two().max(2))
        .map_err(|error| Error::uncoded(ErrorClass::Internal, format!("a channel ring: {error}")))
}

/// Where the read of the next frame stands.
enum Reading {
    /// Reading the 64-byte header.
    Header { buf: [u8; HEADER_LEN], have: usize },
    /// The header said `len` bytes, and no receive slot was free for them.
    NeedSlot { buf: [u8; HEADER_LEN], len: usize },
    /// Reading the frame into a receive slot, the header already copied in.
    Body {
        slot: SlotWriter,
        len: usize,
        have: usize,
    },
}

struct Driver<R, W> {
    core: ChannelCore,
    r: R,
    w: W,
    out: Consumer<Outbound>,
    /// The out ring's wake, made on the domain's thread when the task starts.
    ready: Option<wake::Ready>,
    inbound: Producer<InFrame>,
    done: Producer<SendTicket>,
    shared: Arc<Shared>,
    domain: Arc<DomainState>,
    epoch: Instant,
    reading: Reading,
    /// A received frame or a ticket the dispatcher's ring had no room for yet.
    held_in: Option<InFrame>,
    held_ticket: Option<SendTicket>,
    /// This side's `Close` has gone on the wire.
    close_sent: bool,
    /// This side's half of the stream is finished.
    write_done: bool,
    /// The peer's half of the stream is finished.
    read_done: bool,
    /// The dispatcher dropped its channel.
    hot_gone: bool,
    last_busy: Instant,
    last_tick: Instant,
}

/// What one read step did.
enum ReadStep {
    /// Bytes came in; a whole frame may be ready.
    Bytes,
    /// The peer finished its half of the stream.
    Eof,
}

impl<R, W> Driver<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    fn now(&self) -> MonoTime {
        MonoTime(self.epoch.elapsed().as_nanos() as u64)
    }

    async fn run(mut self) {
        self.ready = Some(wake::Ready::new(&self.out));
        // A stream error is a link fault; the dispatcher sees it as `SendError::Link`.
        if self.drive().await.is_err() {
            self.shared.state.store(FAULTED, Ordering::Release);
        }
        self.publish();
        if !self.write_done {
            let _ = self.w.shutdown().await;
        }
    }

    async fn drive(&mut self) -> Result<(), Error> {
        loop {
            // A pass that finds work at once never awaits anything pending, so it yields here
            // when the task's share of the runtime is spent: every channel of the domain, and
            // the domain itself, keep running.
            tokio::task::consume_budget().await;
            let mut busy = self.take_outbound();
            busy |= self.transmit().await?;
            busy |= self.deliver();
            self.publish();
            if self.core.fault().is_some() {
                // The core sent its `Close` with the fault's code in `transmit`.
                self.shared.state.store(FAULTED, Ordering::Release);
                return Ok(());
            }
            if self.finished() {
                return Ok(());
            }
            let now = Instant::now();
            if busy {
                self.last_busy = now;
            }
            if now.duration_since(self.last_tick) >= TICK || busy {
                self.core.on_tick(self.now());
                self.last_tick = now;
            }
            if busy {
                continue;
            }
            self.wait(now).await?;
        }
    }

    /// Whether the channel is done: both halves of the stream finished and everything the
    /// dispatcher may still take is in its rings.
    fn finished(&self) -> bool {
        self.read_done && self.write_done && self.held_in.is_none() && self.held_ticket.is_none()
    }

    /// Moves the dispatcher's frames and commands into the core.
    fn take_outbound(&mut self) -> bool {
        let mut moved = false;
        loop {
            match self.out.pop() {
                Ok(Outbound::Frame {
                    ftype,
                    step,
                    flags,
                    ext,
                    payload,
                }) => {
                    moved = true;
                    let now = self.now();
                    if self
                        .core
                        .send(ftype, step, flags, ext, payload, now)
                        .is_err()
                    {
                        // The dispatcher checked the frame as the core does, so a refusal here
                        // means the two disagree: stop the channel rather than lose a frame.
                        self.shared.state.store(FAULTED, Ordering::Release);
                    }
                }
                Ok(Outbound::Nack(seq)) => {
                    moved = true;
                    self.core.nack(seq);
                }
                Ok(Outbound::Close(reason)) => {
                    moved = true;
                    self.core.close(reason);
                }
                Err(PopError::Empty) => return moved,
                Err(PopError::Closed) => {
                    if !self.hot_gone {
                        // A dropped channel closes as `close` does.
                        self.hot_gone = true;
                        self.core.close(CloseReason::NORMAL);
                        moved = true;
                    }
                    return moved;
                }
            }
        }
    }

    /// Writes what the core has to send. Finishes this half of the stream once the `Close` has
    /// gone, or once the peer closed and nothing is left.
    async fn transmit(&mut self) -> Result<bool, Error> {
        let mut wrote = false;
        if self.write_done {
            return Ok(false);
        }
        loop {
            let now = self.now();
            let Some(transmit) = self.core.poll_transmit(now) else {
                break;
            };
            wrote = true;
            let write = |error: std::io::Error| link(format!("writing to the stream: {error}"));
            match transmit {
                Transmit::Data {
                    header,
                    ext,
                    payload,
                    ..
                } => {
                    self.w.write_all(header).await.map_err(write)?;
                    self.w.write_all(ext).await.map_err(write)?;
                    self.w.write_all(payload).await.map_err(write)?;
                }
                Transmit::Control(bytes) => {
                    let is_close = bytes.get(5) == Some(&(FrameType::Close as u8));
                    self.w.write_all(bytes).await.map_err(write)?;
                    if is_close {
                        self.close_sent = true;
                    }
                }
            }
        }
        if wrote {
            self.w
                .flush()
                .await
                .map_err(|error| link(format!("writing to the stream: {error}")))?;
        }
        let peer_closed = self.core.peer_close().is_some() || self.read_done;
        if self.close_sent || (peer_closed && self.core.retained_bytes() == 0) {
            self.w
                .shutdown()
                .await
                .map_err(|error| link(format!("finishing the stream: {error}")))?;
            self.write_done = true;
        }
        Ok(wrote)
    }

    /// Hands received frames and completed tickets to the dispatcher's rings, as far as they have
    /// room.
    fn deliver(&mut self) -> bool {
        let mut moved = false;
        loop {
            let frame = match self.held_in.take() {
                Some(frame) => frame,
                None => match self.core.recv() {
                    Some(frame) => frame,
                    None => break,
                },
            };
            match self.inbound.push(frame) {
                Ok(()) => moved = true,
                Err(PushError::Full(frame)) => {
                    self.held_in = Some(frame);
                    break;
                }
                // The dispatcher is gone: the frame's slot comes back as it drops.
                Err(PushError::Closed(_)) => moved = true,
            }
        }
        loop {
            let ticket = match self.held_ticket.take() {
                Some(ticket) => ticket,
                None => match self.core.completed() {
                    Some(ticket) => ticket,
                    None => break,
                },
            };
            match self.done.push(ticket) {
                Ok(()) => moved = true,
                Err(PushError::Full(ticket)) => {
                    self.held_ticket = Some(ticket);
                    break;
                }
                Err(PushError::Closed(_)) => moved = true,
            }
        }
        moved
    }

    /// Publishes the credit, the state and the counters for the dispatcher.
    fn publish(&self) {
        let stats = self.core.stats();
        self.shared.credit_end.store(
            u64::from(self.core.credit()) + stats.frames_sent,
            Ordering::Release,
        );
        if self.core.peer_close().is_some() || self.read_done {
            let _ = self.shared.state.compare_exchange(
                OPEN,
                PEER_CLOSED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
        self.shared.stats.store(&stats);
    }

    /// Waits for the next thing to do: bytes on the stream, a frame from the dispatcher, the
    /// timer, or a pre-wake. While the domain is polling ([`DomainState::spinning`]) it only
    /// yields, so the next frame is picked up at once.
    async fn wait(&mut self, now: Instant) -> Result<(), Error> {
        let spinning = self.domain.spinning(self.last_busy, now);
        let Driver {
            r,
            reading,
            out,
            ready,
            domain,
            read_done,
            hot_gone,
            ..
        } = self;
        let can_read = !*read_done && !matches!(reading, Reading::NeedSlot { .. });
        // A closed ring always looks ready: once the dispatcher is gone, only the stream and the
        // timer can bring work.
        let ring_open = !*hot_gone;
        let step = tokio::select! {
            biased;
            step = read_some(r, reading), if can_read => Some(step),
            () = wait_ring(ready, out), if !spinning && ring_open => None,
            () = domain.prewoken(), if !spinning => None,
            () = tokio::time::sleep(TICK), if !spinning => None,
            () = tokio::task::yield_now(), if spinning => None,
        };
        match step {
            None => {
                self.retry_slot();
                Ok(())
            }
            Some(Ok(ReadStep::Eof)) => {
                self.read_done = true;
                Ok(())
            }
            Some(Ok(ReadStep::Bytes)) => self.advance_read(),
            Some(Err(error)) => Err(link(format!("reading the stream: {error}"))),
        }
    }

    /// After a header arrives, asks the core for the frame's length and a slot; after a body, hands
    /// the frame to the core.
    fn advance_read(&mut self) -> Result<(), Error> {
        match &mut self.reading {
            Reading::Header { buf, have } if *have == HEADER_LEN => {
                let buf = *buf;
                let len = self.core.admit(&buf)?;
                self.reading = Reading::NeedSlot { buf, len };
                self.retry_slot();
            }
            Reading::Body { len, have, .. } if *have == *len => {
                let Reading::Body { slot, len, .. } = std::mem::replace(
                    &mut self.reading,
                    Reading::Header {
                        buf: [0; HEADER_LEN],
                        have: 0,
                    },
                ) else {
                    unreachable!("matched a body above");
                };
                let now = self.now();
                self.core.on_receive(slot.freeze(), len, now);
                self.last_busy = Instant::now();
            }
            _ => {}
        }
        Ok(())
    }

    /// Takes a receive slot for a frame whose header has arrived, if one is free now.
    fn retry_slot(&mut self) {
        let Reading::NeedSlot { buf, len } = self.reading else {
            return;
        };
        let Some(mut slot) = self.core.next_receive_slot() else {
            return;
        };
        slot.bytes_mut()[..HEADER_LEN].copy_from_slice(&buf);
        self.reading = Reading::Body {
            slot,
            len,
            have: HEADER_LEN,
        };
        if len == HEADER_LEN {
            let _ = self.advance_read();
        }
    }
}

/// Waits on the out ring: on its wake once the task has made it, on the timer before.
async fn wait_ring(ready: &mut Option<wake::Ready>, out: &mut Consumer<Outbound>) {
    match ready {
        Some(ready) => ready.wait(out).await,
        None => tokio::time::sleep(TICK).await,
    }
}

/// Reads what the stream has for the frame being read. Cancel-safe: a read that loses the race
/// in [`Driver::wait`] reads nothing.
async fn read_some<R: AsyncRead + Unpin>(
    r: &mut R,
    reading: &mut Reading,
) -> std::io::Result<ReadStep> {
    let (buf, have) = match reading {
        Reading::Header { buf, have } => (&mut buf[..], have),
        Reading::Body { slot, len, have } => (&mut slot.bytes_mut()[..*len], have),
        Reading::NeedSlot { .. } => return Ok(ReadStep::Bytes),
    };
    let n = r.read(&mut buf[*have..]).await?;
    if n == 0 {
        return Ok(ReadStep::Eof);
    }
    *have += n;
    Ok(ReadStep::Bytes)
}
