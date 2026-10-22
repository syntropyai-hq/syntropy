// SPDX-License-Identifier: MIT OR Apache-2.0

//! The property tests: a data sender and a data receiver over a simulated wire that loses,
//! duplicates, delays, reorders and corrupts frames, all drawn from a seed (SplitMix64), so a
//! failing case replays exactly. The criteria they cover:
//! - received frames equal sent frames, exactly once and in order, and every ticket comes back
//!   once, in order, for random loss, duplication, reordering and corruption schedules;
//! - retention memory never exceeds `retention × slot_size`;
//! - a schedule that drops credits only still delivers exactly once;
//! - a schedule that also injects frames whose `(instance, epoch)` is not the channel key's
//!   still delivers the channel's frames exactly once and in order, and every foreign frame is
//!   counted as stale; the receiver's NACKs and CRC counters are checked in `main.rs`, on a
//!   clean wire where they can be attributed;
//! - `Hello`, `Ping` and `Close` landing at random times change nothing delivered.
//!
//! Ack timeouts and CRC storms never escalate here ([`super::NEVER`]): escalation ends a channel
//! by design (RFC-0031 §7.4), and these tests are of the retransmission. `main.rs` tests the
//! escalation.

use std::collections::VecDeque;

use proptest::prelude::*;
use syntropy_core::{InstanceId, MonoTime, Nanos, PlanEpoch};
use syntropy_net::frame::{
    Close, Control, Credit, Flags, FrameSpec, FrameType, Nacks, WIRE_MINOR, code, encode_to_vec,
};
use syntropy_net::transport::{BufferPool, CreditCadence, InFrame, SendError};

use super::{
    EPOCH, INSTANCE, NEVER, Shape, close, control_spec, deliver, hello, hello_ack, lease, ping,
    receiver, send_pool, sender, wire_bytes,
};

/// SplitMix64: the whole schedule comes from one seed.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in `0..n`; `n` is at least 1.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// True with probability `per_mille / 1000`.
    fn chance(&mut self, per_mille: u32) -> bool {
        self.below(1000) < u64::from(per_mille)
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

/// What the wire and the injector do, each in parts per thousand.
#[derive(Clone, Copy, Debug, Default)]
pub struct Faults {
    /// Frames lost, each way.
    pub loss: u32,
    /// Frames from the receiver lost on top of `loss`: credits and ping replies.
    pub credit_loss: u32,
    /// Frames delivered twice.
    pub duplicate: u32,
    /// Frames with one bit flipped, which a CRC always catches.
    pub corrupt: u32,
    /// The largest extra delay, in ticks, on top of one tick: frames overtake each other.
    pub jitter: u64,
    /// Ticks on which a foreign frame lands at each side.
    pub foreign: u32,
    /// Ticks on which a `Hello`, `Ping`, `Close` or `HelloAck` of the channel lands.
    pub control: u32,
    /// Received frames the caller keeps for a while before dropping them.
    pub hold: u32,
}

/// One tick of simulated time.
const TICK: u64 = 5_000;

/// The ticks a run may take before it counts as stuck.
const MAX_TICKS: u64 = 400_000;

/// Frames in flight on one direction of the wire.
#[derive(Default)]
struct Wire {
    in_flight: Vec<(u64, u64, Vec<u8>)>,
    order: u64,
}

impl Wire {
    fn push(
        &mut self,
        rng: &mut SplitMix,
        faults: &Faults,
        tick: u64,
        extra_loss: u32,
        bytes: Vec<u8>,
    ) {
        if rng.chance(faults.loss) || rng.chance(extra_loss) {
            return;
        }
        let copies = if rng.chance(faults.duplicate) { 2 } else { 1 };
        for _ in 0..copies {
            let mut bytes = bytes.clone();
            if rng.chance(faults.corrupt) {
                let bit = rng.below(bytes.len() as u64 * 8) as usize;
                bytes[bit / 8] ^= 1 << (bit % 8);
            }
            let due = tick + 1 + rng.below(faults.jitter + 1);
            self.in_flight.push((due, self.order, bytes));
            self.order += 1;
        }
    }

    /// The frames due by `tick`, in arrival order.
    fn due(&mut self, tick: u64) -> Vec<Vec<u8>> {
        let mut due: Vec<(u64, u64, Vec<u8>)> = Vec::new();
        let mut index = 0;
        while index < self.in_flight.len() {
            if self.in_flight[index].0 <= tick {
                due.push(self.in_flight.swap_remove(index));
            } else {
                index += 1;
            }
        }
        due.sort_by_key(|(at, order, _)| (*at, *order));
        due.into_iter().map(|(_, _, bytes)| bytes).collect()
    }
}

/// A frame the caller sends.
#[derive(Clone, Debug)]
struct Sent {
    step: u64,
    ext: Vec<u8>,
    payload: Vec<u8>,
}

/// What a run saw.
#[derive(Debug)]
pub struct Outcome {
    pub ticks: u64,
    pub retransmits: u64,
    pub foreign: (u64, u64),
}

fn now(tick: u64) -> MonoTime {
    MonoTime(tick * TICK)
}

/// A frame of another `(instance, epoch)`: data with any `chan_seq`, sometimes with a broken
/// payload, or a control frame that would change the channel if it were taken.
fn foreign_frame(rng: &mut SplitMix, frames: u64) -> Vec<u8> {
    let (instance, epoch) = match rng.below(3) {
        0 => (InstanceId(INSTANCE.0 ^ (1 + rng.below(1 << 20))), EPOCH),
        1 => (
            INSTANCE,
            PlanEpoch {
                term: EPOCH.term,
                seq: EPOCH.seq + 1 + rng.below(3),
            },
        ),
        _ => (
            INSTANCE,
            PlanEpoch {
                term: EPOCH.term.saturating_sub(1),
                seq: EPOCH.seq,
            },
        ),
    };
    match rng.below(4) {
        0 | 1 => {
            let len = 1 + rng.below(64) as usize;
            let payload = rng.bytes(len);
            let mut bytes = encode_to_vec(
                &FrameSpec {
                    wire_minor: WIRE_MINOR,
                    ftype: FrameType::Activation,
                    flags: Flags::default(),
                    instance,
                    epoch,
                    step: rng.below(100),
                    chan_seq: 1 + rng.below(frames + 8),
                },
                &[],
                &payload,
            )
            .expect("a valid frame");
            if rng.chance(300) {
                let last = bytes.len() - 1;
                bytes[last] ^= 0x80;
            }
            bytes
        }
        2 => Control::Credit(Credit {
            ack_seq: rng.below(frames + 1),
            limit_seq: frames + 100,
            echo_seq: 0,
            echo_delay_ns: 0,
            sack: false,
            reset: rng.chance(500),
            nacks: Nacks::new(&[1 + rng.below(frames)]).expect("one NACK"),
        })
        .encode_to_vec(&control_spec(instance, epoch))
        .expect("a valid credit"),
        _ => Control::Close {
            close: Close {
                code: code::ACK_TIMEOUT,
                last_seq: 0,
            },
            reason: "foreign",
        }
        .encode_to_vec(&control_spec(instance, epoch))
        .expect("a valid close"),
    }
}

/// Runs `frames` frames of random sizes through a channel of `shape` over a wire with
/// `faults`, and checks every criterion above on every tick.
pub fn run(seed: u64, shape: Shape, frames: usize, faults: Faults) -> Result<Outcome, String> {
    let mut rng = SplitMix(seed);
    // The link profile's α before any sample: the wire's own delay.
    let alpha = Nanos((1 + faults.jitter) * TICK);
    let mut a = sender(shape, alpha, NEVER);
    let mut b = receiver(shape, alpha, NEVER);
    let pool: BufferPool = send_pool(u32::from(shape.retention) + 2, shape.slot_size as usize);
    let largest = shape.slot_size as usize - 64;
    let sent: Vec<Sent> = (0..frames as u64)
        .map(|step| {
            let ext = match rng.below(3) {
                0 => Vec::new(),
                n => rng.bytes(64 * n as usize),
            };
            let ext = if ext.len() > largest { Vec::new() } else { ext };
            let payload_len = rng.below((largest - ext.len()) as u64 + 1) as usize;
            Sent {
                step,
                ext,
                payload: rng.bytes(payload_len),
            }
        })
        .collect();
    let bound = u64::from(shape.retention) * u64::from(shape.slot_size);
    let (mut ab, mut ba) = (Wire::default(), Wire::default());
    let mut next = 0;
    let mut received = 0;
    let mut tickets = 0u64;
    let mut kept: VecDeque<(u64, InFrame)> = VecDeque::new();
    let mut foreign = (0u64, 0u64);
    let mut done_at = None;
    for tick in 1..=MAX_TICKS {
        let t = now(tick);
        // The sending caller: as many frames as the credit and the send pool allow. It tries
        // whatever its credit, as a stage does, so a refused send marks the stall.
        while next < frames {
            let frame = &sent[next];
            let Some((ext, payload)) = lease(&pool, &frame.ext, &frame.payload) else {
                break;
            };
            match a.send(
                FrameType::Activation,
                frame.step,
                Flags::default(),
                ext,
                payload,
                t,
            ) {
                Ok(ticket) if ticket.chan_seq == next as u64 + 1 => next += 1,
                Ok(ticket) => {
                    return Err(format!("frame {next} got ticket {}", ticket.chan_seq));
                }
                Err(SendError::WouldBlock) => break,
                Err(error) => return Err(format!("send of frame {next}: {error:?}")),
            }
        }
        // Frames from elsewhere land directly, so each is counted exactly.
        if rng.chance(faults.foreign) {
            deliver(&mut b, &foreign_frame(&mut rng, frames as u64), t);
            foreign.0 += 1;
        }
        if rng.chance(faults.foreign) {
            deliver(&mut a, &foreign_frame(&mut rng, frames as u64), t);
            foreign.1 += 1;
        }
        if rng.chance(faults.control) {
            let bytes = match rng.below(4) {
                0 => hello(),
                1 => ping(t.0),
                2 => close(next as u64),
                _ => ping(t.0 + 1),
            };
            deliver(&mut b, &bytes, t);
        }
        if rng.chance(faults.control) {
            let bytes = if rng.chance(500) {
                hello_ack()
            } else {
                ping(t.0)
            };
            deliver(&mut a, &bytes, t);
        }
        // Both sides' timers, then the wire.
        a.on_tick(t);
        b.on_tick(t);
        while let Some(transmit) = a.poll_transmit(t) {
            ab.push(&mut rng, &faults, tick, 0, wire_bytes(transmit));
        }
        while let Some(transmit) = b.poll_transmit(t) {
            ba.push(
                &mut rng,
                &faults,
                tick,
                faults.credit_loss,
                wire_bytes(transmit),
            );
        }
        for bytes in ab.due(tick) {
            deliver(&mut b, &bytes, t);
        }
        for bytes in ba.due(tick) {
            deliver(&mut a, &bytes, t);
        }
        // The receiving caller: every frame in order, some kept for a while.
        while kept.front().is_some_and(|(until, _)| *until <= tick) {
            kept.pop_front();
        }
        while let Some(frame) = b.recv() {
            let Some(expected) = sent.get(received) else {
                return Err(format!(
                    "frame {} handed over after all {frames}",
                    frame.header.spec.chan_seq
                ));
            };
            let spec = frame.header.spec;
            if spec.chan_seq != received as u64 + 1
                || spec.step != expected.step
                || (spec.instance, spec.epoch) != (INSTANCE, EPOCH)
                || frame.ext.bytes() != expected.ext.as_slice()
                || frame.payload.bytes() != expected.payload.as_slice()
            {
                return Err(format!(
                    "frame {received} came as chan_seq {}, step {}, {} ext and {} payload bytes",
                    spec.chan_seq,
                    spec.step,
                    frame.ext.len(),
                    frame.payload.len()
                ));
            }
            received += 1;
            if rng.chance(faults.hold) {
                kept.push_back((tick + 1 + rng.below(50), frame));
            }
        }
        while let Some(ticket) = a.completed() {
            tickets += 1;
            if ticket.chan_seq != tickets {
                return Err(format!(
                    "ticket {} came back as {}",
                    tickets, ticket.chan_seq
                ));
            }
        }
        if tickets > received as u64 {
            return Err(format!(
                "{tickets} tickets back for {received} frames handed over"
            ));
        }
        if a.retained_bytes() > bound {
            return Err(format!(
                "{} bytes retained, over retention × slot_size = {bound}",
                a.retained_bytes()
            ));
        }
        if let Some(fault) = a.fault().or(b.fault()) {
            return Err(format!("the channel faulted: {fault}"));
        }
        // Once everything is through, a few more ticks: a late duplicate must not show.
        if received == frames && tickets == frames as u64 {
            let at = *done_at.get_or_insert(tick);
            if tick >= at + 2 * (faults.jitter + 2) {
                let stats = (a.stats(), b.stats());
                if stats.1.stale_dropped != foreign.0 || stats.0.stale_dropped != foreign.1 {
                    return Err(format!(
                        "stale frames counted {} and {}, injected {} and {}",
                        stats.1.stale_dropped, stats.0.stale_dropped, foreign.0, foreign.1
                    ));
                }
                if stats.0.frames_sent != frames as u64 || stats.1.frames_received != frames as u64
                {
                    return Err(format!(
                        "{} sent and {} received of {frames}",
                        stats.0.frames_sent, stats.1.frames_received
                    ));
                }
                return Ok(Outcome {
                    ticks: tick,
                    retransmits: stats.0.retransmits,
                    foreign,
                });
            }
        }
    }
    Err(format!(
        "stuck after {MAX_TICKS} ticks: {next} sent, {received} received, {tickets} tickets back"
    ))
}

fn shapes() -> impl Strategy<Value = Shape> {
    (
        1u16..=8,
        0u16..=4,
        prop::sample::select(vec![256u32, 512, 1024]),
        any::<bool>(),
    )
        .prop_map(|(recv_slots, extra, slot_size, batched)| Shape {
            recv_slots,
            slot_size,
            retention: recv_slots + extra,
            cadence: if batched {
                CreditCadence::Batched
            } else {
                CreditCadence::EveryFrame
            },
        })
}

proptest! {
    /// The criterion "for random loss, duplication, reordering and corruption schedules,
    /// received frames equal sent frames, exactly once and in order", with the retention bound
    /// checked on every tick, and `Hello`, `Ping` and `Close` landing at random.
    #[test]
    fn random_faults_deliver_every_frame_once_and_in_order(
        seed in any::<u64>(),
        shape in shapes(),
        frames in 1usize..=48,
        loss in 0u32..=200,
        duplicate in 0u32..=200,
        corrupt in 0u32..=200,
        jitter in 0u64..=20,
        control in 0u32..=100,
        hold in 0u32..=300,
    ) {
        let faults = Faults { loss, duplicate, corrupt, jitter, control, hold, ..Faults::default() };
        let outcome = run(seed, shape, frames, faults);
        prop_assert!(outcome.is_ok(), "seed {seed:#x}, {shape:?}, {faults:?}: {}", outcome.unwrap_err());
    }

    /// The criterion "a schedule that drops credits only still delivers exactly once": every
    /// data frame arrives, but most credits do not, so the sender learns its acks from the
    /// credits a duplicate provokes after an ack timeout (RFC-0031 §7.2, §7.3).
    #[test]
    fn dropping_only_credits_still_delivers_exactly_once(
        seed in any::<u64>(),
        shape in shapes(),
        frames in 1usize..=48,
        credit_loss in 300u32..=900,
    ) {
        let faults = Faults { credit_loss, ..Faults::default() };
        let outcome = run(seed, shape, frames, faults);
        prop_assert!(outcome.is_ok(), "seed {seed:#x}, {shape:?}, {faults:?}: {}", outcome.unwrap_err());
    }

    /// The criterion "a schedule that also injects frames whose `(instance, epoch)` is not the
    /// channel key's still delivers the channel's frames exactly once and in order", with every
    /// foreign frame counted as stale, at both sides.
    #[test]
    fn foreign_frames_change_nothing_delivered(
        seed in any::<u64>(),
        shape in shapes(),
        frames in 1usize..=48,
        loss in 0u32..=150,
        corrupt in 0u32..=150,
        jitter in 0u64..=10,
        foreign in 50u32..=500,
    ) {
        let faults = Faults { loss, corrupt, jitter, foreign, ..Faults::default() };
        let outcome = run(seed, shape, frames, faults);
        prop_assert!(outcome.is_ok(), "seed {seed:#x}, {shape:?}, {faults:?}: {}", outcome.unwrap_err());
    }
}

/// A heavy fixed schedule: every fault at once, one window, many frames.
#[test]
fn every_fault_at_once_through_a_one_frame_window() {
    let shape = Shape {
        recv_slots: 1,
        slot_size: 512,
        retention: 1,
        cadence: CreditCadence::EveryFrame,
    };
    let faults = Faults {
        loss: 200,
        credit_loss: 200,
        duplicate: 200,
        corrupt: 200,
        jitter: 20,
        foreign: 200,
        control: 100,
        hold: 200,
    };
    for seed in 0..16 {
        let outcome = run(seed, shape, 64, faults).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        // 64 frames through one slot: at least a tick out and a tick back each.
        assert!(outcome.ticks >= 128, "seed {seed}: {outcome:?}");
        assert!(outcome.retransmits > 0, "seed {seed}: {outcome:?}");
        assert!(outcome.foreign.0 > 0 && outcome.foreign.1 > 0);
    }
}

/// With credits only dropped, the frames get through by ack timeouts and duplicates: some are
/// sent again, and none is handed over twice.
#[test]
fn a_credit_only_schedule_resends_on_ack_timeouts() {
    let faults = Faults {
        credit_loss: 800,
        ..Faults::default()
    };
    let outcome = run(7, super::Shape::SMALL, 32, faults).expect("delivery");
    assert!(outcome.retransmits > 0, "{outcome:?}");
}

/// The wire does what it is told: duplicates, flips one bit, loses, and reorders.
#[test]
fn the_wire_injects_what_it_is_told() {
    let mut rng = SplitMix(1);
    let mut wire = Wire::default();
    let faults = Faults {
        duplicate: 1000,
        ..Faults::default()
    };
    wire.push(&mut rng, &faults, 0, 0, vec![1, 2, 3]);
    assert_eq!(wire.due(1), [vec![1, 2, 3], vec![1, 2, 3]]);
    let faults = Faults {
        corrupt: 1000,
        ..Faults::default()
    };
    wire.push(&mut rng, &faults, 0, 0, vec![0; 8]);
    let got = wire.due(1);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].iter().map(|b| b.count_ones()).sum::<u32>(), 1);
    let faults = Faults {
        loss: 1000,
        ..Faults::default()
    };
    wire.push(&mut rng, &faults, 0, 0, vec![9]);
    assert!(wire.due(10).is_empty());
    let faults = Faults {
        jitter: 5,
        ..Faults::default()
    };
    for n in 0..64u8 {
        wire.push(&mut rng, &faults, 0, 0, vec![n]);
    }
    let order: Vec<u8> = wire.due(10).into_iter().map(|b| b[0]).collect();
    assert_eq!(order.len(), 64);
    assert!(
        order.windows(2).any(|w| w[0] > w[1]),
        "some frames overtake others"
    );
}

/// The schedule that found the lost window update: one receive slot, corruption, and a caller
/// that keeps frames, so a credit that only grants more can be lost while the sender has nothing
/// unacknowledged. The stalled sender's probe repairs it.
#[test]
fn a_lost_window_update_found_by_a_seed_is_repaired() {
    let shape = Shape {
        recv_slots: 1,
        slot_size: 256,
        retention: 1,
        cadence: CreditCadence::EveryFrame,
    };
    let faults = Faults {
        corrupt: 42,
        jitter: 7,
        hold: 133,
        ..Faults::default()
    };
    run(507_643_826_620_405_991, shape, 16, faults).expect("delivery");
}
