// SPDX-License-Identifier: MIT OR Apache-2.0

//! The channel core, `net::transport::core` (DIST-E7-T02; RFC-0031 §2, §6.1, §7.1–§7.3;
//! ADR-0307 items 3–5).
//!
//! The tests drive two cores, a data sender and a data receiver, over a simulated wire built from
//! `net`'s public API: frames go out through `poll_transmit`, land in receive slots the core hands
//! out, and come back through `on_receive`. Time is passed in.
//! - [`sim`] runs the property tests: random loss, duplication, reordering, corruption, foreign
//!   frames and control frames, from a seed.
//! - [`conformance`] runs the channel-conformance suite over a backend that wraps the core.
//! - This file holds the shared harness and the tests of one behaviour each: the golden resend,
//!   the full window with `Hello`, `Ping` and `Close`, `FrameTooLarge`, foreign frames, the
//!   credit and its cadence, α samples, and the faults that close a channel.

mod conformance;
mod sim;

use std::path::Path;

use syntropy_core::{ErrorClass, InstanceId, MonoTime, Nanos, PlanEpoch, StageIx};
use syntropy_net::ErrorCode;
use syntropy_net::frame::{
    self, Close, Control, ControlSpec, CredKind, Credit, Flags, Frame, FrameSpec, FrameType,
    HEADER_LEN, Hello, HelloAck, Nacks, Ping, Purpose, WIRE_MINOR, code,
};
use syntropy_net::transport::core::{
    ACK_TIMEOUT_MIN, ACK_TIMEOUTS_TO_FAULT, CONTROL_RECEIVES, CRC_STORM_COUNT, MAX_CONTROL_FRAME,
    PROBE_BACKOFF_MAX,
};
use syntropy_net::transport::{
    BufferPool, ChannelCore, ChannelKey, ChannelPurpose, CloseReason, CoreConfig, CreditCadence,
    ExtRef, PayloadRef, PoolId, SendConfig, SendError, Transmit,
};

/// The key of every channel here; the instance and epoch are the golden frames' (`tests/golden/`).
const INSTANCE: InstanceId = InstanceId(0x1122_3344_5566_7788);
const EPOCH: PlanEpoch = PlanEpoch { term: 3, seq: 41 };

fn key() -> ChannelKey {
    ChannelKey {
        instance: INSTANCE,
        epoch: EPOCH,
        purpose: ChannelPurpose::Activations,
        from: StageIx(0),
        to: StageIx(1),
        lane: 0,
    }
}

/// The sizes of one channel.
#[derive(Clone, Copy, Debug)]
struct Shape {
    recv_slots: u16,
    /// The receiver's slot size, header included; a multiple of 64.
    slot_size: u32,
    retention: u16,
    cadence: CreditCadence,
}

impl Shape {
    const SMALL: Shape = Shape {
        recv_slots: 4,
        slot_size: 1024,
        retention: 4,
        cadence: CreditCadence::EveryFrame,
    };
}

/// How faults are handled: as RFC-0031 §7.2 and §7.3 say, or never, for a test of the
/// retransmission alone.
#[derive(Clone, Copy, Debug)]
struct Escalation {
    ack_timeouts: u32,
    crc_storm: u32,
}

const RFC: Escalation = Escalation {
    ack_timeouts: ACK_TIMEOUTS_TO_FAULT,
    crc_storm: CRC_STORM_COUNT,
};

const NEVER: Escalation = Escalation {
    ack_timeouts: u32::MAX,
    crc_storm: u32::MAX,
};

/// The data sender's side: no data receives, four control receives.
fn sender(shape: Shape, alpha: Nanos, escalation: Escalation) -> ChannelCore {
    ChannelCore::new(CoreConfig {
        key: key(),
        wire_minor: WIRE_MINOR,
        cadence: shape.cadence,
        alpha_p99: alpha,
        recv_slots: 0,
        slot_size: 0,
        pool: BufferPool::heap(PoolId(1), MAX_CONTROL_FRAME, u32::from(CONTROL_RECEIVES))
            .expect("a valid pool"),
        send: Some(SendConfig {
            peer_recv_slots: shape.recv_slots,
            peer_slot_size: shape.slot_size,
            retention: shape.retention,
        }),
        ack_timeouts_to_fault: escalation.ack_timeouts,
        crc_storm_count: escalation.crc_storm,
    })
    .expect("a valid sender")
}

/// The data receiver's side: `recv_slots + 4` receives of the slot size.
fn receiver(shape: Shape, alpha: Nanos, escalation: Escalation) -> ChannelCore {
    let slot = (shape.slot_size as usize).max(MAX_CONTROL_FRAME);
    let slots = u32::from(shape.recv_slots) + u32::from(CONTROL_RECEIVES);
    ChannelCore::new(CoreConfig {
        key: key(),
        wire_minor: WIRE_MINOR,
        cadence: shape.cadence,
        alpha_p99: alpha,
        recv_slots: shape.recv_slots,
        slot_size: shape.slot_size,
        pool: BufferPool::heap(PoolId(2), slot, slots).expect("a valid pool"),
        send: None,
        ack_timeouts_to_fault: escalation.ack_timeouts,
        crc_storm_count: escalation.crc_storm,
    })
    .expect("a valid receiver")
}

/// A frame's bytes on the wire.
fn wire_bytes(transmit: Transmit<'_>) -> Vec<u8> {
    match transmit {
        Transmit::Data {
            header,
            ext,
            payload,
            ..
        } => [&header[..], ext, payload].concat(),
        Transmit::Control(bytes) => bytes.to_vec(),
    }
}

/// Everything `core` has to send now.
fn drain(core: &mut ChannelCore, now: MonoTime) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while let Some(transmit) = core.poll_transmit(now) {
        out.push(wire_bytes(transmit));
    }
    out
}

/// Lands `bytes` in a receive slot `core` hands out, as a message transport does, and gives it
/// to the core.
fn deliver(core: &mut ChannelCore, bytes: &[u8], now: MonoTime) {
    let mut slot = core
        .next_receive_slot()
        .expect("a free receive slot: control frames have four of their own");
    slot.bytes_mut()[..bytes.len()].copy_from_slice(bytes);
    core.on_receive(slot.freeze(), bytes.len(), now);
}

/// A sending pool's slot holding `ext` then `payload`, as leases.
fn lease(pool: &BufferPool, ext: &[u8], payload: &[u8]) -> Option<(ExtRef, PayloadRef)> {
    let mut slot = (0..pool.slots()).find_map(|slot| pool.claim(slot))?;
    let bytes = slot.bytes_mut();
    bytes[..ext.len()].copy_from_slice(ext);
    bytes[ext.len()..ext.len() + payload.len()].copy_from_slice(payload);
    let reader = slot.freeze();
    let ext_len = ext.len() as u32;
    Some((
        ExtRef::new(reader.clone(), 0, ext_len)?,
        PayloadRef::new(reader, ext_len, payload.len() as u32)?,
    ))
}

fn send_pool(slots: u32, slot_size: usize) -> BufferPool {
    BufferPool::heap(PoolId(3), slot_size, slots).expect("a valid pool")
}

/// Sends a data frame of `payload` with no extension, step `step`.
fn send(
    core: &mut ChannelCore,
    pool: &BufferPool,
    step: u64,
    payload: &[u8],
    now: MonoTime,
) -> Result<u64, SendError> {
    let (ext, payload) = lease(pool, &[], payload).expect("a free send slot");
    core.send(
        FrameType::Activation,
        step,
        Flags::default(),
        ext,
        payload,
        now,
    )
    .map(|ticket| ticket.chan_seq)
}

fn control_spec(instance: InstanceId, epoch: PlanEpoch) -> ControlSpec {
    ControlSpec {
        wire_minor: WIRE_MINOR,
        flags: Flags::default(),
        instance,
        epoch,
    }
}

fn encode(control: Control<'_>) -> Vec<u8> {
    control
        .encode_to_vec(&control_spec(INSTANCE, EPOCH))
        .expect("a valid control frame")
}

fn credit(ack_seq: u64, limit_seq: u64, nacks: &[u64]) -> Vec<u8> {
    encode(Control::Credit(Credit {
        ack_seq,
        limit_seq,
        echo_seq: 0,
        echo_delay_ns: 0,
        sack: false,
        reset: false,
        nacks: Nacks::new(nacks).expect("at most 16"),
    }))
}

const CREDENTIAL: [u8; 168] = [7; 168];

fn hello() -> Vec<u8> {
    encode(Control::Hello {
        hello: Hello {
            purpose: Purpose::Activations,
            wire_minor_max: WIRE_MINOR,
            proto_major: 1,
            capabilities: 0,
            instance: INSTANCE,
            epoch: EPOCH,
            from_stage: StageIx(0),
            to_stage: StageIx(1),
            lane: 0,
            wire_dtype: None,
            slot_size: 1024,
            recv_slots: 4,
            cred_kind: CredKind::DataPlaneCredential,
            channel_token: 0,
            resume_from: None,
        },
        credential: &CREDENTIAL,
    })
}

fn hello_ack() -> Vec<u8> {
    encode(Control::HelloAck {
        ack: HelloAck {
            status: code::OK,
            wire_minor: WIRE_MINOR,
            capabilities: 0,
            limit_seq: 4,
            slot_size: 1024,
            recv_slots: 4,
            cred_kind: Some(CredKind::DataPlaneCredential),
        },
        credential: &CREDENTIAL,
    })
}

fn ping(origin_ns: u64) -> Vec<u8> {
    encode(Control::Ping(Ping {
        origin_ns,
        reply: false,
    }))
}

fn close(last_seq: u64) -> Vec<u8> {
    encode(Control::Close {
        close: Close {
            code: code::OK,
            last_seq,
        },
        reason: "done",
    })
}

/// The control message in `bytes`, or `None` for a data frame.
fn control_of(bytes: &[u8]) -> Option<Control<'_>> {
    let frame = Frame::decode(bytes).expect("the core sends valid frames");
    frame
        .header
        .spec
        .ftype
        .is_control()
        .then(|| frame.control().expect("a valid control frame"))
}

fn credits_in(frames: &[Vec<u8>]) -> Vec<Credit> {
    frames
        .iter()
        .filter_map(|bytes| match control_of(bytes) {
            Some(Control::Credit(credit)) => Some(credit),
            _ => None,
        })
        .collect()
}

fn at(us: u64) -> MonoTime {
    MonoTime(us * 1_000)
}

const ALPHA: Nanos = Nanos(20_000);

/// The golden frame `name` from `tests/golden/`, which `tests/frame/spec.rs` generates from
/// RFC-0031's tables.
fn golden(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/golden/{name}.hex"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let digits: String = text
        .lines()
        .map(|line| line.split('#').next().unwrap_or(""))
        .flat_map(|line| line.chars().filter(|c| !c.is_whitespace()))
        .collect();
    (0..digits.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&digits[i..i + 2], 16).expect("hex"))
        .collect()
}

/// The criterion "resends are byte-identical (golden test)". The core sends the golden
/// `data-activation` frame, then sends it again after a NACK and after an ack timeout: all three
/// are the golden bytes, header and CRCs included (RFC-0031 §7.3).
#[test]
fn resends_are_byte_identical_to_the_golden_frame() {
    let golden = golden("data-activation");
    let spec = Frame::decode(&golden)
        .expect("the golden frame")
        .header
        .spec;
    assert_eq!(
        (spec.chan_seq, spec.step, spec.flags),
        (1, 12, Flags(0b1001))
    );
    let (ext, payload) = (&golden[64..192], &golden[192..]);
    let pool = send_pool(2, 1024);
    let mut a = sender(Shape::SMALL, ALPHA, RFC);
    let (ext_ref, payload_ref) = lease(&pool, ext, payload).unwrap();
    let ticket = a
        .send(
            FrameType::Activation,
            12,
            spec.flags,
            ext_ref,
            payload_ref,
            at(0),
        )
        .unwrap();
    assert_eq!(ticket.chan_seq, 1);

    let first = drain(&mut a, at(0));
    assert_eq!(first, std::slice::from_ref(&golden), "the first send");
    assert_eq!(
        frame::encode_to_vec(&spec, ext, payload).unwrap(),
        golden,
        "the codec agrees"
    );

    // A NACK of frame 1.
    deliver(&mut a, &credit(0, 4, &[1]), at(10));
    let after_nack = drain(&mut a, at(10));
    assert_eq!(
        after_nack,
        std::slice::from_ref(&golden),
        "the resend after a NACK"
    );
    assert_eq!(a.stats().retransmits, 1);

    // No credit for an ack timeout: the oldest unacknowledged frame goes again.
    let late = at(10 + ACK_TIMEOUT_MIN.0 / 1_000);
    a.on_tick(late);
    let after_timeout = drain(&mut a, late);
    assert_eq!(
        after_timeout,
        std::slice::from_ref(&golden),
        "the resend after an ack timeout"
    );
    assert_eq!(a.stats().retransmits, 2);

    // The ack releases the leases, then `completed` hands the ticket back.
    assert_eq!(a.retained_bytes(), golden.len() as u64);
    deliver(&mut a, &credit(1, 5, &[]), late);
    assert_eq!(a.retained_bytes(), 0);
    assert!(
        pool.is_free(0),
        "the frame's send slot is free once acknowledged"
    );
    assert_eq!(a.completed().map(|t| t.chan_seq), Some(1));
    assert_eq!(a.completed(), None);
}

/// The criterion "a schedule that interleaves `Hello`, `Ping` and `Close` with a full credit
/// window still delivers every data frame exactly once". Three windows of four frames: each
/// fills the receiver's data slots, then four control frames land at once in the four control
/// receives, so none takes a data frame's receive (RFC-0031 §7.1). The sender closes with its last
/// window unacknowledged, and its `Close` goes out only once every frame is acknowledged.
#[test]
fn hello_ping_and_close_with_a_full_window_deliver_every_frame_once() {
    let shape = Shape::SMALL;
    let pool = send_pool(8, 1024);
    let mut a = sender(shape, ALPHA, RFC);
    let mut b = receiver(shape, ALPHA, RFC);
    let mut now = 0;
    let mut sent = 0u64;
    let mut delivered = Vec::new();
    for window in 0..3 {
        // Fill the window.
        while sent < 12 {
            match send(&mut a, &pool, sent, &[sent as u8; 100], at(now)) {
                Ok(seq) => {
                    sent += 1;
                    assert_eq!(seq, sent);
                }
                Err(SendError::WouldBlock) => break,
                Err(error) => panic!("send {sent}: {error:?}"),
            }
        }
        assert_eq!(a.credit(), 0, "window {window} is full");
        if window == 2 {
            a.close(CloseReason::NORMAL);
            assert_eq!(
                send(&mut a, &pool, 99, &[0; 8], at(now)),
                Err(SendError::Closed)
            );
        }
        now += 10;
        let frames = drain(&mut a, at(now));
        assert_eq!(frames.len(), 4, "four data frames and no Close yet");
        for bytes in &frames {
            deliver(&mut b, bytes, at(now));
        }
        // The receiver holds all four, so its data slots are full. Four control frames land
        // together, each in a receive of its own.
        let controls = [
            hello(),
            ping(now * 1_000),
            close(sent),
            ping(now * 1_000 + 1),
        ];
        let mut landed = Vec::new();
        for bytes in &controls {
            let mut slot = b
                .next_receive_slot()
                .expect("a control receive while every data receive is full");
            slot.bytes_mut()[..bytes.len()].copy_from_slice(bytes);
            landed.push((slot.freeze(), bytes.len()));
        }
        for (slot, len) in landed {
            b.on_receive(slot, len, at(now));
        }
        assert_eq!(b.fault(), None);
        // The receiver acknowledges all four but grants nothing new while it holds them, and
        // answers both pings.
        now += 10;
        let back = drain(&mut b, at(now));
        let credits = credits_in(&back);
        assert_eq!(
            credits.last().map(|c| (c.ack_seq, c.limit_seq)),
            Some((sent, sent))
        );
        let replies = back
            .iter()
            .filter(|bytes| {
                matches!(
                    control_of(bytes),
                    Some(Control::Ping(Ping { reply: true, .. }))
                )
            })
            .count();
        assert_eq!(replies, 2);
        for bytes in &back {
            deliver(&mut a, bytes, at(now));
        }
        assert_eq!(a.credit(), 0);
        // The caller takes and drops the four frames; their credit returns.
        while let Some(frame) = b.recv() {
            delivered.push((frame.header.spec.chan_seq, frame.payload.bytes()[0]));
        }
        now += 10;
        for bytes in drain(&mut b, at(now)) {
            deliver(&mut a, &bytes, at(now));
        }
        while a.completed().is_some() {}
    }
    // Every frame was acknowledged, so the Close goes out now.
    now += 10;
    let last = drain(&mut a, at(now));
    assert!(
        matches!(
            last.as_slice(),
            [bytes] if matches!(control_of(bytes), Some(Control::Close { close: Close { code: 0, last_seq: 12 }, .. }))
        ),
        "the Close, after the last ack"
    );
    let expected: Vec<(u64, u8)> = (1..=12).map(|seq| (seq, (seq - 1) as u8)).collect();
    assert_eq!(delivered, expected, "every frame once, in order");
    assert_eq!(b.recv().map(|f| f.header.spec.chan_seq), None);
    assert_eq!(a.fault(), None);
    assert_eq!(b.fault(), None);
}

/// The criterion "a frame larger than the slot is refused with `FrameTooLarge` (0x0205) in a
/// class RFC-0060 §10.2 allows, not §12's `Internal`".
#[test]
fn a_frame_larger_than_the_slot_is_refused_with_frame_too_large() {
    // §12's table, as revised on 2026-10-23.
    assert_eq!(ErrorCode::FRAME_TOO_LARGE.0, 0x0205);
    assert_eq!(
        ErrorCode::FRAME_TOO_LARGE.class(),
        Some(ErrorClass::InvalidRequest)
    );

    // The sender refuses a frame larger than the peer's slot.
    let shape = Shape {
        slot_size: 256,
        ..Shape::SMALL
    };
    let pool = send_pool(2, 1024);
    let mut a = sender(shape, ALPHA, RFC);
    assert_eq!(
        send(&mut a, &pool, 0, &[0; 256 - 64 + 1], at(0)),
        Err(SendError::TooLarge)
    );
    assert_eq!(send(&mut a, &pool, 0, &[0; 256 - 64], at(0)), Ok(1));

    // A receiver reading a stream checks the length before the body: a frame past its slot
    // is a protocol error that closes the channel with 0x0205.
    let mut b = receiver(shape, ALPHA, RFC);
    let big = frame::encode_to_vec(
        &FrameSpec {
            wire_minor: WIRE_MINOR,
            ftype: FrameType::Activation,
            flags: Flags::default(),
            instance: INSTANCE,
            epoch: EPOCH,
            step: 0,
            chan_seq: 1,
        },
        &[],
        &[0; 300],
    )
    .unwrap();
    let header: [u8; HEADER_LEN] = big[..HEADER_LEN].try_into().unwrap();
    let error = b.admit(&header).unwrap_err();
    assert_eq!(error.code, Some(ErrorCode::FRAME_TOO_LARGE));
    assert_eq!(error.class, ErrorClass::InvalidRequest);
    assert_ne!(error.class, ErrorClass::Internal);
    assert_eq!(b.fault(), Some(&error));
    let out = drain(&mut b, at(1));
    assert!(
        matches!(
            out.as_slice(),
            [bytes] if matches!(control_of(bytes), Some(Control::Close { close: Close { code: 0x0205, .. }, .. }))
        ),
        "a Close with 0x0205"
    );

    // A message transport lands the frame whole; one larger than the channel's slot size is
    // refused the same way, though the pool's slots could hold it.
    let mut b = receiver(shape, ALPHA, RFC);
    deliver(&mut b, &big, at(0));
    let fault = b.fault().expect("a fault");
    assert_eq!(fault.code, Some(ErrorCode::FRAME_TOO_LARGE));
    assert_eq!(fault.class, ErrorClass::InvalidRequest);
    assert!(b.recv().is_none());

    // A frame that fits is admitted, with its length.
    let mut b = receiver(shape, ALPHA, RFC);
    let small = frame::encode_to_vec(
        &FrameSpec {
            wire_minor: WIRE_MINOR,
            ftype: FrameType::Activation,
            flags: Flags::default(),
            instance: INSTANCE,
            epoch: EPOCH,
            step: 0,
            chan_seq: 1,
        },
        &[],
        &[0; 192],
    )
    .unwrap();
    let header: [u8; HEADER_LEN] = small[..HEADER_LEN].try_into().unwrap();
    assert_eq!(b.admit(&header), Ok(256));
    assert_eq!(b.fault(), None);
}

/// A data frame with a valid header CRC, of this channel's key or another's.
fn data_frame(instance: InstanceId, epoch: PlanEpoch, chan_seq: u64, payload: &[u8]) -> Vec<u8> {
    frame::encode_to_vec(
        &FrameSpec {
            wire_minor: WIRE_MINOR,
            ftype: FrameType::Activation,
            flags: Flags::default(),
            instance,
            epoch,
            step: 0,
            chan_seq,
        },
        &[],
        payload,
    )
    .unwrap()
}

/// The criterion's foreign frames, one of each kind on a clean wire: whatever their `chan_seq`
/// and payload, each is dropped before the `chan_seq` check, never surfaced or NACKed, and
/// counted as stale (RFC-0031 §2, §6.1, §7.2).
#[test]
fn foreign_frames_are_dropped_counted_and_never_nacked() {
    let shape = Shape::SMALL;
    let pool = send_pool(8, 1024);
    let mut a = sender(shape, ALPHA, RFC);
    let mut b = receiver(shape, ALPHA, RFC);
    let other_instance = InstanceId(INSTANCE.0 ^ 1);
    let next_epoch = PlanEpoch {
        term: EPOCH.term,
        seq: EPOCH.seq + 1,
    };
    let next_term = PlanEpoch {
        term: EPOCH.term + 1,
        seq: 0,
    };
    // Frame 1 arrives; then foreign frames claiming 1 (a duplicate), 2 (the next), 4 (a gap),
    // 99 (past the credit) and 0, one with a broken payload, and foreign control frames.
    send(&mut a, &pool, 0, b"one", at(0)).unwrap();
    for bytes in drain(&mut a, at(0)) {
        deliver(&mut b, &bytes, at(1));
    }
    drain(&mut b, at(1));
    let mut broken = data_frame(other_instance, EPOCH, 2, b"broken payload");
    let last = broken.len() - 1;
    broken[last] ^= 0x40;
    let foreign = [
        data_frame(other_instance, EPOCH, 1, b"dup"),
        data_frame(other_instance, EPOCH, 2, b"two?"),
        data_frame(INSTANCE, next_epoch, 2, b"two?"),
        data_frame(INSTANCE, next_term, 4, b"gap"),
        data_frame(INSTANCE, next_epoch, 99, b"past the credit"),
        broken,
        Control::Close {
            close: Close {
                code: code::OK,
                last_seq: 0,
            },
            reason: "",
        }
        .encode_to_vec(&control_spec(other_instance, EPOCH))
        .unwrap(),
        Control::Ping(Ping {
            origin_ns: 5,
            reply: false,
        })
        .encode_to_vec(&control_spec(INSTANCE, next_term))
        .unwrap(),
    ];
    for bytes in &foreign {
        deliver(&mut b, bytes, at(2));
    }
    let stats = b.stats();
    assert_eq!(stats.stale_dropped, foreign.len() as u64);
    assert_eq!(
        (
            stats.nacks_sent,
            stats.crc_payload_fail,
            stats.crc_header_fail
        ),
        (0, 0, 0)
    );
    assert_eq!(b.fault(), None);
    assert_eq!(b.peer_close(), None, "a foreign Close closes nothing");
    assert_eq!(
        drain(&mut b, at(3)),
        Vec::<Vec<u8>>::new(),
        "no credit, NACK or ping reply"
    );

    // A foreign credit acknowledging everything changes nothing on the sender.
    let foreign_credit = Control::Credit(Credit {
        ack_seq: 1,
        limit_seq: 1000,
        echo_seq: 1,
        echo_delay_ns: 0,
        sack: false,
        reset: true,
        nacks: Nacks::EMPTY,
    })
    .encode_to_vec(&control_spec(INSTANCE, next_epoch))
    .unwrap();
    deliver(&mut a, &foreign_credit, at(3));
    assert_eq!(a.stats().stale_dropped, 1);
    assert_eq!(a.completed(), None);
    assert_eq!(a.credit(), 3);
    assert_eq!(a.fault(), None);

    // The channel's own frames go on: frame 1 is the only one handed over.
    let got: Vec<(u64, Vec<u8>)> = std::iter::from_fn(|| b.recv())
        .map(|f| (f.header.spec.chan_seq, f.payload.bytes().to_vec()))
        .collect();
    assert_eq!(got, [(1, b"one".to_vec())]);
    send(&mut a, &pool, 1, b"two", at(4)).unwrap();
    for bytes in drain(&mut a, at(4)) {
        deliver(&mut b, &bytes, at(5));
    }
    let got: Vec<(u64, Vec<u8>)> = std::iter::from_fn(|| b.recv())
        .map(|f| (f.header.spec.chan_seq, f.payload.bytes().to_vec()))
        .collect();
    assert_eq!(got, [(2, b"two".to_vec())]);
}

/// Recomputes a frame's header CRC after a test changed its header.
fn reseal_header(bytes: &mut [u8]) {
    let ext_len = u32::from_le_bytes(bytes[48..52].try_into().unwrap()) as usize;
    let crc = frame::crc32c_append(frame::crc32c(&bytes[..60]), &bytes[64..64 + ext_len]);
    bytes[60..64].copy_from_slice(&crc.to_le_bytes());
}

/// `bytes` with its wire minor, frame type or `chan_seq` changed and its header CRC recomputed:
/// an intact header the frame rules refuse.
fn rewritten(
    mut bytes: Vec<u8>,
    minor: Option<u8>,
    ftype: Option<u8>,
    chan_seq: Option<u64>,
) -> Vec<u8> {
    if let Some(minor) = minor {
        bytes[4] = minor;
    }
    if let Some(ftype) = ftype {
        bytes[5] = ftype;
    }
    if let Some(chan_seq) = chan_seq {
        bytes[40..48].copy_from_slice(&chan_seq.to_le_bytes());
    }
    reseal_header(&mut bytes);
    bytes
}

/// A frame of another instance or epoch is stale whatever else its header holds: a newer wire
/// minor, a type the minor lacks, or a control frame with a `chan_seq`. The same headers on
/// this channel's key close the channel, each with its class (RFC-0031 §6.1, §6.2, §12).
#[test]
fn foreign_frames_are_stale_whatever_their_minor_or_type() {
    let other = InstanceId(INSTANCE.0 ^ 2);
    let foreign_ping = Control::Ping(Ping {
        origin_ns: 1,
        reply: false,
    })
    .encode_to_vec(&control_spec(other, EPOCH))
    .unwrap();
    let foreign = [
        rewritten(
            data_frame(other, EPOCH, 1, b"x"),
            Some(WIRE_MINOR + 1),
            None,
            None,
        ),
        rewritten(data_frame(other, EPOCH, 1, b"x"), None, Some(0x7F), None),
        rewritten(foreign_ping, None, None, Some(5)),
    ];
    let mut b = receiver(Shape::SMALL, ALPHA, RFC);
    for bytes in &foreign {
        deliver(&mut b, bytes, at(0));
    }
    assert_eq!(b.fault(), None);
    assert_eq!(b.stats().stale_dropped, 3);
    assert!(drain(&mut b, at(1)).is_empty());

    let own = data_frame(INSTANCE, EPOCH, 1, b"x");
    let cases = [
        (
            rewritten(own.clone(), Some(WIRE_MINOR + 1), None, None),
            ErrorClass::InvalidRequest,
            None,
        ),
        (
            rewritten(own.clone(), None, Some(0x7F), None),
            ErrorClass::Unsupported,
            Some(ErrorCode::UNKNOWN_FRAME_TYPE),
        ),
        (
            rewritten(ping(1), None, None, Some(5)),
            ErrorClass::Integrity,
            None,
        ),
    ];
    for (bytes, class, code) in cases {
        let mut b = receiver(Shape::SMALL, ALPHA, RFC);
        deliver(&mut b, &bytes, at(0));
        let fault = b.fault().expect("a fault");
        assert_eq!(
            (fault.class, fault.code),
            (class, code),
            "{}",
            fault.message
        );
        assert_ne!(fault.class, ErrorClass::Internal);
        assert_eq!(b.stats().stale_dropped, 0);
    }
}

/// Only a reply to a ping this core sent gives an α sample, so a peer cannot raise α, and with
/// it the ack timeout, with a forged `origin_ns`; and a credit's `limit_seq` counts no further
/// than the peer's `recv_slots` past the ack, so even `u64::MAX` is safe.
#[test]
fn a_forged_ping_reply_or_a_huge_limit_changes_nothing() {
    let mut a = sender(Shape::SMALL, ALPHA, RFC);
    let reply = |origin_ns| {
        encode(Control::Ping(Ping {
            origin_ns,
            reply: true,
        }))
    };
    deliver(&mut a, &reply(0), at(1_000_000));
    assert_eq!(a.stats().alpha_samples, 0);
    assert_eq!(a.alpha_p99(), ALPHA);
    let sent = a.ping(at(100)).map(wire_bytes).expect("a ping");
    let origin = match control_of(&sent) {
        Some(Control::Ping(ping)) => ping.origin_ns,
        other => panic!("{other:?}"),
    };
    deliver(&mut a, &reply(origin), at(140));
    assert_eq!(a.stats().alpha_last, Some(Nanos(20_000)));
    // Its reply counts once.
    deliver(&mut a, &reply(origin), at(10_000));
    assert_eq!(a.stats().alpha_samples, 1);

    deliver(&mut a, &credit(0, u64::MAX, &[]), at(2));
    assert_eq!(a.credit(), 4);
    assert_eq!(a.fault(), None);
}

/// The credit (RFC-0031 §7.1): `recv_slots` at the start, no more while the receiver holds its
/// frames, and back once it drops them; control frames take none.
#[test]
fn credit_returns_only_when_the_receiver_drops_its_frames() {
    let shape = Shape::SMALL;
    let pool = send_pool(8, 1024);
    let mut a = sender(shape, ALPHA, RFC);
    let mut b = receiver(shape, ALPHA, RFC);
    assert_eq!(a.credit(), 4);
    for step in 0..4 {
        send(&mut a, &pool, step, &[1; 32], at(0)).unwrap();
    }
    assert_eq!(
        send(&mut a, &pool, 4, &[1; 32], at(0)),
        Err(SendError::WouldBlock)
    );
    assert_eq!(a.stats().credit_stalls, 1);
    for bytes in drain(&mut a, at(0)) {
        deliver(&mut b, &bytes, at(1));
    }
    let held: Vec<_> = std::iter::from_fn(|| b.recv()).collect();
    assert_eq!(held.len(), 4);
    // A ping takes no credit and grants none; the receiver answers it with its credit.
    deliver(&mut b, &ping(1), at(1));
    let back = drain(&mut b, at(2));
    let credits = credits_in(&back);
    assert_eq!(
        credits
            .iter()
            .map(|c| (c.ack_seq, c.limit_seq))
            .collect::<Vec<_>>(),
        [(4, 4)]
    );
    for bytes in &back {
        deliver(&mut a, bytes, at(3));
    }
    assert_eq!(a.credit(), 0);
    assert_eq!(
        std::iter::from_fn(|| a.completed())
            .map(|t| t.chan_seq)
            .collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
    // Two frames dropped, two credits back.
    let mut held = held;
    let kept = held.split_off(2);
    drop(held);
    let back = drain(&mut b, at(4));
    assert_eq!(credits_in(&back).last().map(|c| c.limit_seq), Some(6));
    for bytes in &back {
        deliver(&mut a, bytes, at(5));
    }
    assert_eq!(a.credit(), 2);
    assert!(a.stats().credit_stall.0 > 0, "the stall's time is counted");
    drop(kept);
}

/// The `Batched` cadence (RFC-0031 §7.1): a credit every `max(1, recv_slots / 4)` frames, or when
/// the sender's remaining credit falls below half, where `EveryFrame` sends one per frame.
#[test]
fn batched_credits_come_every_quarter_window() {
    for (cadence, expected) in [(CreditCadence::EveryFrame, 8), (CreditCadence::Batched, 4)] {
        let shape = Shape {
            recv_slots: 8,
            retention: 8,
            cadence,
            ..Shape::SMALL
        };
        let pool = send_pool(10, 1024);
        let mut a = sender(shape, ALPHA, RFC);
        let mut b = receiver(shape, ALPHA, RFC);
        for step in 0..8 {
            send(&mut a, &pool, step, &[2; 16], at(0)).unwrap();
        }
        let mut credits = 0;
        for bytes in drain(&mut a, at(0)) {
            deliver(&mut b, &bytes, at(1));
            credits += credits_in(&drain(&mut b, at(1))).len();
            while b.recv().is_some() {}
        }
        assert_eq!(credits, expected, "{cadence:?}");
        assert_eq!(
            CreditCadence::for_purpose(ChannelPurpose::Bulk),
            CreditCadence::Batched
        );
        assert_eq!(
            CreditCadence::for_purpose(ChannelPurpose::Activations),
            CreditCadence::EveryFrame
        );
    }
}

/// α samples come from credit echoes (RFC-0031 §7.1): `rtt = now − sent_at − echo_delay`, and
/// α is half of it. A frame sent twice gives none, since its echo cannot be matched to a send.
#[test]
fn alpha_comes_from_credit_echoes_and_never_from_a_resent_frame() {
    let shape = Shape::SMALL;
    let pool = send_pool(8, 1024);
    let mut a = sender(shape, ALPHA, RFC);
    let mut b = receiver(shape, ALPHA, RFC);
    send(&mut a, &pool, 0, b"x", at(0)).unwrap();
    let out = drain(&mut a, at(100));
    deliver(&mut b, &out[0], at(130));
    // The receiver sends its credit 5 µs after the frame landed.
    let back = drain(&mut b, at(135));
    let echo = credits_in(&back)[0];
    assert_eq!((echo.echo_seq, echo.echo_delay_ns), (1, 5_000));
    deliver(&mut a, &back[0], at(165));
    // 65 µs round trip less the 5 µs delay: α is 30 µs.
    assert_eq!(a.stats().alpha_samples, 1);
    assert_eq!(a.stats().alpha_last, Some(Nanos(30_000)));
    assert_eq!(a.alpha_p99(), Nanos(30_000));

    // Frame 2 is NACKed and sent again; the echo of it gives no sample.
    send(&mut a, &pool, 1, b"y", at(200)).unwrap();
    drain(&mut a, at(200));
    deliver(&mut a, &credit(1, 5, &[2]), at(210));
    let again = drain(&mut a, at(210));
    deliver(&mut b, &again[0], at(240));
    let back = drain(&mut b, at(240));
    assert_eq!(credits_in(&back)[0].echo_seq, 2);
    deliver(&mut a, &back[0], at(270));
    assert_eq!(a.stats().alpha_samples, 1);
    assert_eq!(a.completed().map(|t| t.chan_seq), Some(1));
    assert_eq!(a.completed().map(|t| t.chan_seq), Some(2));
}

/// Retention bounds memory whatever the peer grants (RFC-0031 §7.3): a credit past the peer's
/// `recv_slots` still leaves the sender at `retention × slot_size` retained bytes.
#[test]
fn a_peer_granting_too_much_still_meets_the_retention_window() {
    let shape = Shape::SMALL;
    let pool = send_pool(8, 1024);
    let mut a = sender(shape, ALPHA, RFC);
    deliver(&mut a, &credit(0, 100, &[]), at(0));
    assert_eq!(a.credit(), 4, "no more than recv_slots past the ack");
    for step in 0..4 {
        send(&mut a, &pool, step, &[4; 1024 - 64], at(0)).unwrap();
    }
    assert_eq!(
        send(&mut a, &pool, 4, &[4; 1024 - 64], at(0)),
        Err(SendError::WouldBlock)
    );
    assert_eq!(a.retained_bytes(), 4 * 1024);
    deliver(&mut a, &credit(1, 100, &[]), at(1));
    assert_eq!(send(&mut a, &pool, 4, &[4; 1024 - 64], at(1)), Ok(5));
    assert_eq!(a.retained_bytes(), 4 * 1024);
}

/// A peer that grants more than its `recv_slots` past the ack gets no more than `recv_slots`
/// of credit, even where the retention window would allow more (RFC-0031 §7.1): with
/// `retention` 8 above `recv_slots` 4, only the `limit_seq` cap holds the sender at 4.
#[test]
fn the_limit_seq_cap_holds_where_retention_would_allow_more() {
    let shape = Shape {
        recv_slots: 4,
        slot_size: 1024,
        retention: 8,
        cadence: CreditCadence::EveryFrame,
    };
    let pool = send_pool(16, 1024);
    let mut a = sender(shape, ALPHA, RFC);
    deliver(&mut a, &credit(0, u64::MAX, &[]), at(0));
    assert_eq!(a.credit(), 4);
    for step in 0..4 {
        send(&mut a, &pool, step, &[1; 16], at(0)).unwrap();
    }
    assert_eq!(
        send(&mut a, &pool, 4, &[1; 16], at(0)),
        Err(SendError::WouldBlock)
    );
}

/// A credit that only grants more is lost while the sender has nothing unacknowledged: the
/// sender, refused for credit, probes with a `Ping` after an ack timeout, then after 2 and 4 ack
/// timeouts, and the receiver answers with its current credit. A sender that was never refused
/// does not probe.
#[test]
fn a_stalled_sender_probes_for_a_lost_credit() {
    let shape = Shape::SMALL;
    let pool = send_pool(8, 1024);
    let mut a = sender(shape, ALPHA, RFC);
    let mut b = receiver(shape, ALPHA, RFC);
    for step in 0..4 {
        send(&mut a, &pool, step, &[3; 16], at(0)).unwrap();
    }
    for bytes in drain(&mut a, at(0)) {
        deliver(&mut b, &bytes, at(1));
    }
    let held: Vec<_> = std::iter::from_fn(|| b.recv()).collect();
    for bytes in drain(&mut b, at(2)) {
        deliver(&mut a, &bytes, at(3));
    }
    assert_eq!(
        (a.credit(), a.retained_bytes()),
        (0, 0),
        "all acknowledged, no credit"
    );
    // Nothing refused yet: no probe, however long.
    a.on_tick(at(10_000));
    assert!(drain(&mut a, at(10_000)).is_empty());
    assert_eq!(
        send(&mut a, &pool, 4, &[3; 16], at(10_000)),
        Err(SendError::WouldBlock)
    );
    // The caller drops its frames; the credit that grants four more is lost.
    drop(held);
    let lost = credits_in(&drain(&mut b, at(10_001)));
    assert_eq!(lost.iter().map(|c| c.limit_seq).collect::<Vec<_>>(), [8]);
    // Probes after 1, 2, 4, … ack timeouts, the wait doubling up to its cap of
    // PROBE_BACKOFF_MAX (64) ack timeouts; the first seven answers are lost too.
    let timeout = ACK_TIMEOUT_MIN.0 / 1_000;
    let mut probes = Vec::new();
    for us in 10_000..10_000 + 256 * timeout {
        a.on_tick(at(us));
        for bytes in drain(&mut a, at(us)) {
            assert!(matches!(
                control_of(&bytes),
                Some(Control::Ping(Ping { reply: false, .. }))
            ));
            probes.push(us - 10_000);
            deliver(&mut b, &bytes, at(us));
            let answer = drain(&mut b, at(us));
            assert_eq!(credits_in(&answer).last().map(|c| c.limit_seq), Some(8));
            if probes.len() == 8 {
                for bytes in &answer {
                    deliver(&mut a, bytes, at(us));
                }
            }
        }
    }
    // The gaps are 1, 2, 4, 8, 16, 32, then 64 and 64 again at the cap; no probe after the
    // credit arrives.
    let gaps: Vec<u64> = [0]
        .iter()
        .chain(&probes)
        .zip(&probes)
        .map(|(last, next)| (next - last) / timeout)
        .collect();
    assert_eq!(gaps, [1, 2, 4, 8, 16, 32, 64, 64]);
    assert_eq!(probes.last(), Some(&(191 * timeout)));
    assert_eq!(PROBE_BACKOFF_MAX, 64);
    assert_eq!(a.credit(), 4);
    assert_eq!(send(&mut a, &pool, 4, &[3; 16], at(20_000)), Ok(5));
}

/// RFC-0031 §7.3: three ack timeouts in a row are a link fault, `AckTimeout` (0x0208, `Link`).
/// The channel sends `Close` with the code and refuses further sends.
#[test]
fn three_ack_timeouts_in_a_row_are_a_link_fault() {
    let pool = send_pool(4, 1024);
    let mut a = sender(Shape::SMALL, ALPHA, RFC);
    send(&mut a, &pool, 0, b"lost", at(0)).unwrap();
    drain(&mut a, at(0));
    let timeout = ACK_TIMEOUT_MIN.0 / 1_000;
    for n in 1..=3 {
        let now = at(n * timeout);
        a.on_tick(now);
        if n < 3 {
            assert_eq!(a.fault(), None, "after {n} timeouts");
            assert_eq!(drain(&mut a, now).len(), 1, "a resend after timeout {n}");
        }
    }
    let fault = a.fault().expect("a fault after three timeouts");
    assert_eq!(fault.code, Some(ErrorCode::ACK_TIMEOUT));
    assert_eq!(fault.class, ErrorClass::Link);
    let out = drain(&mut a, at(3 * timeout));
    assert!(matches!(
        out.as_slice(),
        [bytes] if matches!(control_of(bytes), Some(Control::Close { close: Close { code: 0x0208, last_seq: 1 }, .. }))
    ));
    assert_eq!(
        send(&mut a, &pool, 1, b"more", at(1_000)),
        Err(SendError::Link)
    );

    // An ack between timeouts starts the count again.
    let mut a = sender(Shape::SMALL, ALPHA, RFC);
    for step in 0..2 {
        send(&mut a, &pool, step, b"x", at(0)).unwrap();
    }
    drain(&mut a, at(0));
    a.on_tick(at(timeout));
    a.on_tick(at(2 * timeout));
    deliver(&mut a, &credit(1, 5, &[]), at(2 * timeout + 1));
    a.on_tick(at(3 * timeout + 1));
    a.on_tick(at(4 * timeout + 1));
    assert_eq!(a.fault(), None);
}

/// A frame whose payload CRC fails: its header, then one flipped payload bit.
fn corrupt_payload(seq: u64) -> Vec<u8> {
    let mut bytes = data_frame(INSTANCE, EPOCH, seq, b"payload");
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    bytes
}

/// RFC-0031 §7.2: a payload CRC failure drops the frame and NACKs it; three on one channel within
/// 10 s are a link fault, `CrcStorm` (0x0207, `Link`), and three spread wider are not.
#[test]
fn three_payload_crc_failures_within_ten_seconds_are_a_crc_storm() {
    let shape = Shape::SMALL;
    let mut b = receiver(shape, ALPHA, RFC);
    deliver(&mut b, &corrupt_payload(1), at(0));
    let back = drain(&mut b, at(1));
    assert_eq!(credits_in(&back)[0].nacks.as_slice(), [1]);
    assert_eq!(b.stats().crc_payload_fail, 1);
    deliver(&mut b, &corrupt_payload(1), MonoTime(5_000_000_000));
    assert_eq!(b.fault(), None);
    deliver(&mut b, &corrupt_payload(1), MonoTime(9_999_000_000));
    let fault = b.fault().expect("a CRC storm");
    assert_eq!(fault.code, Some(ErrorCode::CRC_STORM));
    assert_eq!(fault.class, ErrorClass::Link);

    let mut b = receiver(shape, ALPHA, RFC);
    for seconds in [0, 6, 12, 18] {
        deliver(
            &mut b,
            &corrupt_payload(1),
            MonoTime(seconds * 1_000_000_000),
        );
    }
    assert_eq!(b.fault(), None, "never three within 10 s");
    assert_eq!(b.stats().crc_payload_fail, 4);
}

/// A header CRC failure drops the frame and NACKs the next expected one (RFC-0031 §7.2), and a
/// credit with `RESET` is a link fault, `PeerReset` (§7.4).
#[test]
fn a_broken_header_nacks_the_next_frame_and_a_reset_credit_faults() {
    let mut b = receiver(Shape::SMALL, ALPHA, RFC);
    let mut bytes = data_frame(INSTANCE, EPOCH, 1, b"x");
    bytes[20] ^= 0x10;
    deliver(&mut b, &bytes, at(0));
    assert_eq!(b.stats().crc_header_fail, 1);
    let back = drain(&mut b, at(1));
    assert_eq!(credits_in(&back)[0].nacks.as_slice(), [1]);

    let mut a = sender(Shape::SMALL, ALPHA, RFC);
    let reset = encode(Control::Credit(Credit {
        ack_seq: 0,
        limit_seq: 4,
        echo_seq: 0,
        echo_delay_ns: 0,
        sack: false,
        reset: true,
        nacks: Nacks::EMPTY,
    }));
    deliver(&mut a, &reset, at(0));
    let fault = a.fault().expect("a fault");
    assert_eq!(fault.code, Some(ErrorCode::PEER_RESET));
    assert_eq!(fault.class, ErrorClass::Link);
}

/// A late `Hello` or `HelloAck` changes nothing: the handshake ran before the core opened.
#[test]
fn a_late_hello_or_hello_ack_changes_nothing() {
    let pool = send_pool(4, 1024);
    let mut a = sender(Shape::SMALL, ALPHA, RFC);
    let mut b = receiver(Shape::SMALL, ALPHA, RFC);
    deliver(&mut a, &hello_ack(), at(0));
    deliver(&mut b, &hello(), at(0));
    assert_eq!((a.fault(), b.fault()), (None, None));
    assert_eq!(a.credit(), 4);
    send(&mut a, &pool, 0, b"after", at(1)).unwrap();
    for bytes in drain(&mut a, at(1)) {
        deliver(&mut b, &bytes, at(2));
    }
    assert_eq!(b.recv().map(|f| f.header.spec.chan_seq), Some(1));
}

/// The core refuses a receive pool without the four control receives, and a retention window
/// below the peer's `recv_slots` (RFC-0031 §7.1, §7.3).
#[test]
fn the_core_refuses_too_few_receives_or_too_short_a_retention() {
    let shape = Shape::SMALL;
    let short_pool = CoreConfig {
        key: key(),
        wire_minor: WIRE_MINOR,
        cadence: shape.cadence,
        alpha_p99: ALPHA,
        recv_slots: 4,
        slot_size: 1024,
        pool: BufferPool::heap(PoolId(9), 1024, 7).unwrap(),
        send: None,
        ack_timeouts_to_fault: ACK_TIMEOUTS_TO_FAULT,
        crc_storm_count: CRC_STORM_COUNT,
    };
    let error = ChannelCore::new(short_pool.clone()).unwrap_err();
    assert_eq!(error.class, ErrorClass::InvalidRequest);
    let short_retention = CoreConfig {
        pool: BufferPool::heap(PoolId(9), 1024, 8).unwrap(),
        send: Some(SendConfig {
            peer_recv_slots: 4,
            peer_slot_size: 1024,
            retention: 3,
        }),
        ..short_pool
    };
    assert!(ChannelCore::new(short_retention).is_err());
}
