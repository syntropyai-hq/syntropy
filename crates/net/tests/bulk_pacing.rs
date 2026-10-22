// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bulk pacing in the simulator (DIST-E7-T13; RFC-0031 §7.5; RFC-0001 §8.4 rule 6; ADR-0011
//! item 8): a bulk transfer beside a live data channel on one simulated 2.5 GbE edge, paced by
//! `net::transport::pacing::BulkPacer` under `SimTransport`.
//!
//! The edge is RFC-0060 §8.3's `lan_2_5g`, whose links share their β between every stream in
//! send order, so bulk frames delay the data channel's. The data channel is decode-like: one
//! 16 KiB frame every 2 ms [target]. Bulk goes in 4 KiB frames [target], close to the packet
//! interleaving a real QUIC connection gets. Every figure here comes from the simulator's model,
//! so each is an [estimate] of the rig, never a measurement.

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use syntropy_core::ids::{Endpoint, LinkId, LinkKind, NetMedium};
use syntropy_core::plan::WireDType;
use syntropy_core::policy::Fraction;
use syntropy_core::{Bytes, InstanceId, MonoTime, Nanos, NodeId, PlanEpoch, StageIx};
use syntropy_net::Error;
use syntropy_net::frame::{Flags, FrameType};
use syntropy_net::transport::core::MAX_CONTROL_FRAME;
use syntropy_net::transport::pacing::{BulkPacer, PacingConfig, WINDOW};
use syntropy_net::transport::{
    BufferPool, Channel, ChannelKey, ChannelPriority, ChannelPurpose, ChannelSpec, ExtBuf,
    OutFrame, PoolId, RemoteEndpoint, SendError, Transport,
};
use syntropy_testkit::sim::{FAILOVER_LAN, LinkModel, SimClock, SimEntropy, SimFabric, SimNet};

const A: NodeId = NodeId([1; 32]);
const B: NodeId = NodeId([2; 32]);

/// The data channel's frame and period [target]: a decode step's activations.
const DATA_FRAME: u32 = 16 * 1024;
const DATA_PERIOD: u64 = 2_000_000;

/// The bulk channel's frame [target].
const BULK_FRAME: u32 = 4 * 1024;

/// The idle phase that measures the data channel's idle p99 [target].
const IDLE: u64 = 3_000_000_000;

fn link(from: NodeId, to: NodeId) -> LinkId {
    LinkId {
        src: Endpoint::Nic {
            node: from,
            ordinal: 0,
        },
        dst: Endpoint::Nic {
            node: to,
            ordinal: 0,
        },
        kind: LinkKind::Network(NetMedium::Ethernet),
        path: 0,
    }
}

fn key(purpose: ChannelPurpose, lane: u8) -> ChannelKey {
    ChannelKey {
        instance: InstanceId(7),
        epoch: PlanEpoch { term: 1, seq: 1 },
        purpose,
        from: StageIx(0),
        to: StageIx(1),
        lane,
    }
}

fn ready<T>(future: impl Future<Output = T>) -> Option<T> {
    let mut future = pin!(future);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(value) => Some(value),
        Poll::Pending => None,
    }
}

/// Opens `key` from A to B, B receiving with `recv_slots` slots of `slot` bytes.
fn open(
    net: &SimNet,
    key: ChannelKey,
    recv_slots: u16,
    slot: usize,
    pools: u32,
) -> Result<(Channel, Channel), Error> {
    let (a, b) = (net.transport(A), net.transport(B));
    let control = BufferPool::heap(PoolId(pools), MAX_CONTROL_FRAME, 4).expect("a pool");
    let receive = BufferPool::heap(
        PoolId(pools + 1),
        slot.max(MAX_CONTROL_FRAME),
        u32::from(recv_slots) + 4,
    )
    .expect("a pool");
    let spec = |pool, recv_slots, slot: usize, retention| ChannelSpec {
        key,
        pool,
        slot_size: Bytes(slot as u64),
        recv_slots,
        priority: ChannelPriority::Data,
        wire: WireDType::F32,
        retention,
    };
    let sending = spec(a.register(&control)?, 0, 0, recv_slots);
    let receiving = spec(b.register(&receive)?, recv_slots, slot, 0);
    let peer = RemoteEndpoint {
        node: B,
        key: [0; 32],
        addrs: Vec::new(),
        relay: None,
    };
    let mut connect = pin!(a.connect(&peer, sending));
    let mut cx = Context::from_waker(Waker::noop());
    assert!(connect.as_mut().poll(&mut cx).is_pending());
    let receiver = ready(b.accept(receiving)).expect("the connect waits")?;
    let Poll::Ready(sender) = connect.as_mut().poll(&mut cx) else {
        panic!("paired");
    };
    Ok((sender?, receiver))
}

/// Sends a frame of `len` payload bytes from `pool`'s next free slot.
fn send_one(
    channel: &mut Channel,
    pool: &BufferPool,
    step: u64,
    len: u32,
) -> Result<u64, SendError> {
    let mut slot = (0..pool.slots())
        .find_map(|slot| pool.claim(slot))
        .expect("a free send slot");
    slot.bytes_mut()[..len as usize].fill(step as u8);
    let reader = slot.freeze();
    let result = channel.try_send(OutFrame {
        ftype: FrameType::Activation,
        step,
        flags: Flags::default(),
        ext: ExtBuf::EMPTY,
        payload: reader.payload(0, len).expect("in the slot"),
    });
    drop(reader);
    result.map(|ticket| ticket.chan_seq)
}

/// The nearest-rank p99 of `samples`.
fn p99(samples: &[Nanos]) -> Nanos {
    let mut sorted: Vec<u64> = samples.iter().map(|sample| sample.0).collect();
    sorted.sort_unstable();
    let rank = (sorted.len() * 99).div_ceil(100).max(1);
    Nanos(sorted[rank - 1])
}

/// What one run saw.
#[derive(Debug, PartialEq, Eq)]
struct Run {
    /// The data channel's α p99 on the idle edge, and under bulk.
    idle_p99: Nanos,
    bulk_p99: Nanos,
    /// The p99 of each 250 ms window under bulk.
    windows: Vec<Nanos>,
    /// When the bulk transfer started and ended, and its bytes.
    bulk_start: u64,
    bulk_end: u64,
    bulk_bytes: u64,
    /// The pacer at the end, and its ceiling.
    rate: u64,
    ceiling: u64,
    halvings: u64,
    increases: u64,
    /// Each window's bulk bytes, from the pacer's count.
    rates: Vec<u64>,
}

/// The scenario, from `seed`: an idle phase, then `bulk` bytes of bulk paced against the idle
/// p99 times `ratio`, beside the data channel throughout.
fn run(seed: u64, bulk: u64, idle_scale: f64) -> Run {
    let clock = SimClock::new();
    let model = LinkModel::preset("lan_2_5g").expect("a preset");
    let beta = model.beta;
    let mut fabric = SimFabric::new();
    fabric.add_link(link(A, B), model.clone());
    fabric.add_link(link(B, A), model);
    let net = SimNet::new(fabric, clock.handle(), SimEntropy::new(seed), FAILOVER_LAN);
    net.route(A, B, link(A, B));
    net.route(B, A, link(B, A));
    let mut clock = clock;
    let pool = BufferPool::heap(PoolId(1), 64 * 1024, 128).expect("a pool");
    net.transport(A).register(&pool).expect("registered");
    let (mut data, mut data_rx) =
        open(&net, key(ChannelPurpose::Activations, 0), 4, 64 * 1024, 10).expect("data");
    let (mut bulk_tx, mut bulk_rx) =
        open(&net, key(ChannelPurpose::Bulk, 1), 64, 8 * 1024, 20).expect("bulk");
    net.watch_alpha(A, B);

    let now = |clock: &SimClock| syntropy_core::Clock::now(clock).0;
    let mut next_data = 0u64;
    let mut step = 0u64;
    let mut bulk_sent = 0u64;
    let mut bulk_start = None;
    let mut idle_p99 = None;
    let mut rates = Vec::new();
    let mut last_window = 0u64;
    let mut last_bytes = 0u64;
    let mut iterations = 0u64;
    loop {
        iterations += 1;
        assert!(iterations < 50_000_000, "no progress");
        let t = now(&clock);
        // The data channel: a frame every period, when it has credit.
        if t >= next_data {
            if data.credit() > 0 {
                step += 1;
                send_one(&mut data, &pool, step, DATA_FRAME).expect("data credit");
            }
            next_data += DATA_PERIOD;
        }
        // After the idle phase, pace the edge and start the bulk transfer.
        if idle_p99.is_none() && t >= IDLE {
            let samples: Vec<Nanos> = net.alpha_log(A, B).iter().map(|s| s.1).collect();
            let measured = p99(&samples);
            idle_p99 = Some(measured);
            net.pace(
                A,
                B,
                PacingConfig {
                    beta,
                    idle_p99: Nanos((measured.0 as f64 * idle_scale) as u64),
                    p99_ratio: 1.2,
                    floor: Fraction::new(0.05).unwrap(),
                    burst: 64 * 1024,
                },
            );
            bulk_start = Some(t);
            last_window = t;
        }
        // The bulk sender: as much as the pacer and the credit allow.
        let mut bulk_wait = None;
        if bulk_start.is_some() && bulk_sent < bulk {
            loop {
                match send_one(&mut bulk_tx, &pool, 0, BULK_FRAME) {
                    Ok(_) => {
                        bulk_sent += u64::from(BULK_FRAME);
                        if bulk_sent >= bulk {
                            break;
                        }
                    }
                    Err(SendError::WouldBlock) => {
                        let at = net.pacing_ready_at(A, B, u64::from(BULK_FRAME) + 64);
                        bulk_wait = (at.0 > t).then_some(at.0);
                        break;
                    }
                    Err(error) => panic!("bulk send: {error:?}"),
                }
            }
        }
        // The receivers drop what they get; the senders take back their tickets.
        while data_rx.try_recv().is_some() {}
        while bulk_rx.try_recv().is_some() {}
        while data.completed().is_some() {}
        while bulk_tx.completed().is_some() {}
        // Each window's bulk rate, from the pacer's byte count.
        if let Some(pacer) = net.pacer(A, B)
            && t >= last_window + WINDOW.0
        {
            let bytes = pacer.stats().bytes;
            rates.push(bytes - last_bytes);
            last_bytes = bytes;
            last_window += WINDOW.0;
        }
        let done = bulk_sent >= bulk && bulk_tx.stats().frames_sent == bulk_tx_acked(&bulk_tx);
        if done {
            break;
        }
        // The next event: the network's, the data channel's period, or the pacer.
        let next = [net.next_deadline().map(|d| d.0), Some(next_data), bulk_wait]
            .into_iter()
            .flatten()
            .min()
            .expect("the data channel always has a next frame");
        clock.advance_to(MonoTime(next.max(t)));
        net.step();
    }
    let bulk_start = bulk_start.expect("started");
    let end = now(&clock);
    let log = net.alpha_log(A, B);
    let under: Vec<Nanos> = log
        .iter()
        .filter(|(at, _)| at.0 >= bulk_start)
        .map(|(_, alpha)| *alpha)
        .collect();
    let mut windows = Vec::new();
    let mut start = bulk_start;
    while start < end {
        let window: Vec<Nanos> = log
            .iter()
            .filter(|(at, _)| at.0 >= start && at.0 < start + WINDOW.0)
            .map(|(_, alpha)| *alpha)
            .collect();
        if !window.is_empty() {
            windows.push(p99(&window));
        }
        start += WINDOW.0;
    }
    let pacer: BulkPacer = net.pacer(A, B).expect("paced");
    let stats = pacer.stats();
    Run {
        idle_p99: idle_p99.expect("measured"),
        bulk_p99: p99(&under),
        windows,
        bulk_start,
        bulk_end: end,
        bulk_bytes: bulk_sent,
        rate: pacer.rate(),
        ceiling: pacer.ceiling(),
        halvings: stats.halvings,
        increases: stats.increases,
        rates,
    }
}

/// Whether every bulk frame sent is acknowledged: the tickets came back.
fn bulk_tx_acked(channel: &Channel) -> u64 {
    // `completed` has been drained each step, so acked frames are those no longer credited
    // against: the sent count once the credit window is whole again.
    let stats = channel.stats();
    if channel.credit() == 64 {
        stats.frames_sent
    } else {
        u64::MAX
    }
}

/// The bound, as the pacer computes it: `ceil(1.2 × idle p99)`.
fn bound(run: &Run) -> Nanos {
    Nanos((run.idle_p99.0 as f64 * 1.2).ceil() as u64)
}

/// RFC-0031 §7.5 with the ceiling (note of 2026-10-23), seed 1, 1.25 GiB [target] beside the
/// data channel on `lan_2_5g`. The rate climbs from the floor until one window's p99 breaks
/// 1.2 × idle; it halves, climbs back to its ceiling two steps below the breach, and stays
/// there. Every window but the one the halving answers is within the bound, and the bulk
/// transfer runs at most of β meanwhile [estimate].
#[test]
fn aimd_settles_under_its_ceiling_and_every_other_window_keeps_the_bound() {
    let run = run(1, 5 << 28, 1.0);
    let bound = bound(&run);
    let over: Vec<usize> = run
        .windows
        .iter()
        .enumerate()
        .filter(|(_, p99)| **p99 > bound)
        .map(|(at, _)| at)
        .collect();
    assert_eq!(run.halvings, 1, "{run:?}");
    assert_eq!(
        over.len(),
        1,
        "one window over the bound, the one the halving answers: {run:?}"
    );
    // It settles at its ceiling, two steps below the rate that broke the bound: the last two
    // windows carry the same bytes, at most of β.
    assert_eq!(run.rate, run.ceiling, "{run:?}");
    let settled = &run.rates[run.rates.len() - 2..];
    assert_eq!(settled[0], settled[1], "{:?}", run.rates);
    let beta = 2.35e9 / 8.0;
    let steady = settled[0] as f64 / (WINDOW.0 as f64 / 1e9);
    assert!(steady >= 0.85 * beta, "steady {steady} B/s");
}

/// RFC-0031 §7.5's floor: a data channel whose bound can never be met halves the rate every
/// window, and the transfer still moves at 5 % of β [estimate].
#[test]
fn the_floor_holds_when_the_bound_cannot_be_met() {
    // An idle p99 a hundredth of the real one: every window breaks the bound.
    let run = run(2, 32 << 20, 0.01);
    assert_eq!(run.increases, 0, "{run:?}");
    let floor = (2.35e9 / 8.0 * 0.05f64).round() as u64;
    assert_eq!(run.rate, floor);
    let seconds = (run.bulk_end - run.bulk_start) as f64 / 1e9;
    let throughput = run.bulk_bytes as f64 / seconds;
    // The floor, give or take the burst over the run.
    assert!(
        throughput >= floor as f64 * 0.98,
        "{throughput} B/s against {floor}"
    );
    assert!(
        throughput <= floor as f64 * 1.1,
        "{throughput} B/s against {floor}"
    );
}

/// A seed replays the paced run exactly: the same α samples, windows, rates and end; another
/// seed gives another run.
#[test]
fn a_seed_replays_the_paced_run() {
    let first = run(7, 96 << 20, 1.0);
    assert_eq!(run(7, 96 << 20, 1.0), first);
    assert_ne!(run(8, 96 << 20, 1.0), first);
}

/// The rig criterion's shape in the simulator: a 10 GiB-class transfer [target] keeps the data
/// channel's p99 over the whole transfer within 1.2 × idle. Takes about a minute and a half in a
/// debug build, so it is run by hand: `cargo test -p syntropy-net --test bulk_pacing -- --ignored`.
#[test]
#[ignore = "a minute and a half; run with --ignored"]
fn a_ten_gib_transfer_keeps_the_transfer_p99_within_the_bound() {
    let run = run(1, 10 << 30, 1.0);
    eprintln!(
        "ten GiB, seed 1: idle p99 {} ns, p99 over the transfer {} ns ({:.3} ×), {:.1} s, {} halvings",
        run.idle_p99.0,
        run.bulk_p99.0,
        run.bulk_p99.0 as f64 / run.idle_p99.0 as f64,
        (run.bulk_end - run.bulk_start) as f64 / 1e9,
        run.halvings
    );
    assert!(run.bulk_p99 <= bound(&run), "{run:?}");
}
