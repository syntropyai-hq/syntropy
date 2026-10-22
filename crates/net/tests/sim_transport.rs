// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's `SimTransport` (QUAL-E8-T04; RFC-0060 §8.1, §8.3): RFC-0031's `Transport` and
//! `Channel` over the simulated fabric, built on the channel core (DIST-E7-T02).
//!
//! - The channel-conformance suite passes on it.
//! - A sender that exhausts its credit gets `WouldBlock`, and the wait is recorded as
//!   `credit_stall`, to the nanosecond of virtual time.
//! - Frames in flight on a link that goes down are dropped, and the channel reports `Link` after
//!   the failure timeout, at both ends. A link back up before then loses nothing.

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use syntropy_core::ids::{Endpoint, LinkId, LinkKind, NetMedium};
use syntropy_core::plan::WireDType;
use syntropy_core::{Bytes, BytesPerSec, ErrorClass, InstanceId, MonoTime, Nanos, NodeId};
use syntropy_core::{PlanEpoch, StageIx};
use syntropy_net::Error;
use syntropy_net::frame::{Flags, FrameType};
use syntropy_net::transport::conformance::{self, Fabric, MAX_FRAME, SEND_SLOT, SEND_SLOTS};
use syntropy_net::transport::core::MAX_CONTROL_FRAME;
use syntropy_net::transport::{
    BufferPool, Channel, ChannelKey, ChannelPriority, ChannelPurpose, ChannelSpec, ExtBuf,
    OutFrame, PoolId, RemoteEndpoint, SendError, Transport,
};
use syntropy_testkit::sim::{
    Dist, FAILOVER_LAN, LinkModel, SimClock, SimEntropy, SimFabric, SimNet, SimTransport,
};

const A: NodeId = NodeId([1; 32]);
const B: NodeId = NodeId([2; 32]);

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

/// A link with a fixed one-way delay and no loss, so virtual times can be checked exactly.
fn fixed(alpha: Nanos) -> LinkModel {
    LinkModel {
        alpha: Dist::Const(alpha),
        beta: BytesPerSec(1.25e9),
        jitter: Dist::Const(Nanos(0)),
        loss: 0.0,
        reorder: 0.0,
        mtu: 1_500,
        up: true,
        per_packet_overhead: Nanos(0),
    }
}

/// Two nodes, A and B, joined both ways by links of `model`, on a network whose clock the test
/// owns.
struct Two {
    clock: SimClock,
    net: SimNet,
    a: SimTransport,
    b: SimTransport,
}

impl Two {
    fn new(model: LinkModel, seed: u64) -> Two {
        let clock = SimClock::new();
        let mut fabric = SimFabric::new();
        fabric.add_link(link(A, B), model.clone());
        fabric.add_link(link(B, A), model);
        let net = SimNet::new(fabric, clock.handle(), SimEntropy::new(seed), FAILOVER_LAN);
        net.route(A, B, link(A, B));
        net.route(B, A, link(B, A));
        Two {
            a: net.transport(A),
            b: net.transport(B),
            clock,
            net,
        }
    }

    /// Advances the clock by `by` and moves the network.
    fn run(&mut self, by: Nanos) {
        self.clock.advance(by);
        self.net.step();
    }

    /// Advances in steps of `step` until `done` or `limit` passes.
    fn run_until(&mut self, step: Nanos, limit: Nanos, mut done: impl FnMut() -> bool) -> bool {
        let end = self.clock_now().0 + limit.0;
        while self.clock_now().0 < end {
            if done() {
                return true;
            }
            self.run(step);
        }
        done()
    }

    fn clock_now(&self) -> MonoTime {
        syntropy_core::Clock::now(&self.clock)
    }
}

/// Polls a future that the simulated network completes without waiting on anything else.
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

fn key(lane: u8) -> ChannelKey {
    ChannelKey {
        instance: InstanceId(7),
        epoch: PlanEpoch { term: 1, seq: 1 },
        purpose: ChannelPurpose::Activations,
        from: StageIx(0),
        to: StageIx(1),
        lane,
    }
}

fn remote(node: NodeId) -> RemoteEndpoint {
    RemoteEndpoint {
        node,
        key: [0; 32],
        addrs: Vec::new(),
        relay: None,
    }
}

/// Opens `key` from A to B: A sends, B receives with `recv_slots` slots of `slot_size` bytes.
/// Pools are numbered from `pools`.
fn open(
    two: &Two,
    key: ChannelKey,
    recv_slots: u16,
    slot_size: usize,
    pools: u32,
) -> Result<(Channel, Channel), Error> {
    let control = BufferPool::heap(PoolId(pools), MAX_CONTROL_FRAME, 4).expect("a pool");
    let receive = BufferPool::heap(
        PoolId(pools + 1),
        slot_size.max(MAX_CONTROL_FRAME),
        u32::from(recv_slots) + 4,
    )
    .expect("a pool");
    let spec = |pool, recv_slots, slot_size: usize, retention| ChannelSpec {
        key,
        pool,
        slot_size: Bytes(slot_size as u64),
        recv_slots,
        priority: ChannelPriority::Data,
        wire: WireDType::F32,
        retention,
    };
    let sending = spec(two.a.register(&control)?, 0, 0, recv_slots);
    let receiving = spec(two.b.register(&receive)?, recv_slots, slot_size, 0);
    let peer = remote(B);
    let mut connect = pin!(two.a.connect(&peer, sending));
    let mut cx = Context::from_waker(Waker::noop());
    assert!(
        connect.as_mut().poll(&mut cx).is_pending(),
        "the accept is not called yet"
    );
    let receiver = ready(two.b.accept(receiving)).expect("the connect is waiting")?;
    let Poll::Ready(sender) = connect.as_mut().poll(&mut cx) else {
        panic!("the accept paired the connect");
    };
    Ok((sender?, receiver))
}

/// The suite's fabric: one `SimNet` of two nodes, each check's channel opened through the
/// transports, and 50 µs of virtual time per pump.
struct Sim {
    two: Two,
    send_pool: BufferPool,
    opened: u32,
    hold: bool,
}

impl Fabric for Sim {
    fn open(&mut self, key: ChannelKey, recv_slots: u16) -> Result<(Channel, Channel), Error> {
        self.opened += 1;
        open(&self.two, key, recv_slots, MAX_FRAME, 100 + 2 * self.opened)
    }

    fn send_pool(&self) -> &BufferPool {
        &self.send_pool
    }

    fn pump(&mut self) {
        // Held delivery stops time on the whole network: nothing in flight moves.
        if !self.hold {
            self.two.run(Nanos(50_000));
        }
    }

    fn hold_delivery(&mut self, hold: bool) -> bool {
        self.hold = hold;
        true
    }
}

/// The criterion "the `Transport` conformance tests from DIST pass on `SimTransport`".
#[test]
fn the_conformance_suite_passes_on_sim_transport() {
    for seed in [1, 2, 3] {
        let two = Two::new(LinkModel::preset("lan_2_5g").expect("a preset"), seed);
        let send_pool = BufferPool::heap(PoolId(1), SEND_SLOT, SEND_SLOTS).expect("a pool");
        two.a.register(&send_pool).expect("registered");
        let mut sim = Sim {
            two,
            send_pool,
            opened: 0,
            hold: false,
        };
        let report = conformance::run(&mut sim);
        assert!(report.passed(), "seed {seed}: {report}");
        assert!(report.skipped.is_empty(), "seed {seed}: {report}");
        assert_eq!(report.to_string(), "12 of 12 rules passed");
    }
}

/// Sends an `Activation` frame whose payload is `len` bytes of `step`, from `pool`'s next free
/// slot; the slot's reader is dropped when the send returns, as a caller may.
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

/// Every frame `channel` has now, as (`chan_seq`, step), each dropped once read.
fn received(channel: &mut Channel) -> Vec<(u64, u64)> {
    std::iter::from_fn(|| channel.try_recv())
        .map(|frame| (frame.header.spec.chan_seq, frame.header.spec.step))
        .collect()
}

/// A two-node network with one channel from A to B over fixed 100 µs links, and A's send pool.
fn one_channel(recv_slots: u16) -> (Two, Channel, Channel, BufferPool) {
    let two = Two::new(fixed(Nanos(100_000)), 9);
    let pool = BufferPool::heap(PoolId(1), 1024, 16).expect("a pool");
    two.a.register(&pool).expect("registered");
    let (sender, receiver) = open(&two, key(0), recv_slots, 1024, 10).expect("opened");
    (two, sender, receiver, pool)
}

/// The criterion "a sender that exhausts credit gets `WouldBlock` and records `credit_stall`".
/// The receiver keeps both frames of a two-slot window, so the third send is refused; once it
/// drops them, its credit arrives one link delay later, and the stall is that wait, exactly.
#[test]
fn a_sender_out_of_credit_gets_would_block_and_records_the_stall() {
    let (mut two, mut sender, mut receiver, pool) = one_channel(2);
    assert_eq!(sender.credit(), 2);
    assert_eq!(send_one(&mut sender, &pool, 1, 100), Ok(1));
    assert_eq!(send_one(&mut sender, &pool, 2, 100), Ok(2));
    assert_eq!(
        send_one(&mut sender, &pool, 3, 100),
        Err(SendError::WouldBlock)
    );
    let stalled_at = two.clock_now();
    assert_eq!(sender.stats().credit_stalls, 1);
    // Both frames arrive; the receiver keeps them, so no credit comes back.
    two.run(Nanos(1_000_000));
    let held: Vec<_> = std::iter::from_fn(|| receiver.try_recv()).collect();
    assert_eq!(held.len(), 2);
    two.run(Nanos(1_000_000));
    assert_eq!(sender.credit(), 0);
    assert_eq!(sender.stats().credit_stall, Nanos(0), "not yet released");
    // The receiver drops them at t; its credit, a 128-byte frame, lands 100 µs + ⌈128 B / β⌉
    // later, and the stall ends there.
    drop(held);
    two.run(Nanos(0));
    let released_at = two.clock_now();
    assert!(two.run_until(Nanos(10_000), Nanos(10_000_000), || sender.credit() == 2));
    let transfer = (128.0f64 * 1e9 / 1.25e9).ceil() as u64;
    let expected = released_at.0 - stalled_at.0 + 100_000 + transfer;
    assert_eq!(sender.stats().credit_stall, Nanos(expected));
    assert_eq!(send_one(&mut sender, &pool, 3, 100), Ok(3));
}

/// The criterion "frames in flight on a downed link are dropped and the channel reports `Link`".
/// The link goes down with two frames on it: neither arrives, the channel keeps its key and
/// takes sends until the failure timeout, then both ends report a `Link` error.
#[test]
fn frames_in_flight_on_a_downed_link_are_dropped_and_the_channel_reports_link() {
    let (mut two, mut sender, mut receiver, pool) = one_channel(4);
    assert_eq!(send_one(&mut sender, &pool, 1, 100), Ok(1));
    assert_eq!(send_one(&mut sender, &pool, 2, 100), Ok(2));
    assert_eq!(two.net.in_flight(), 2);
    two.net
        .with_fabric(|fabric| fabric.set_up(&link(A, B), false));
    let down_at = two.clock_now();
    two.run(Nanos(1_000_000));
    assert!(
        received(&mut receiver).is_empty(),
        "both frames were dropped"
    );
    // Before the failure timeout the channel is still open.
    two.run(Nanos(FAILOVER_LAN.0 - 2_000_000));
    assert_eq!(two.net.fault(A, &key(0)), None);
    assert_eq!(send_one(&mut sender, &pool, 3, 100), Ok(3));
    // At the failure timeout both ends fail with a `Link` error.
    let net = two.net.clone();
    assert!(two.run_until(Nanos(100_000), Nanos(10_000_000), || {
        two_fault(&net, A).is_some()
    }));
    assert!(two.clock_now().0 - down_at.0 >= FAILOVER_LAN.0);
    for node in [A, B] {
        let fault = two_fault(&two.net, node).expect("a fault at both ends");
        assert_eq!(fault.class, ErrorClass::Link, "{fault}");
    }
    assert_eq!(send_one(&mut sender, &pool, 4, 100), Err(SendError::Link));
    assert!(received(&mut receiver).is_empty());
}

fn two_fault(net: &SimNet, node: NodeId) -> Option<Error> {
    net.fault(node, &key(0))
}

/// A link back up before the failure timeout loses nothing: the cores send the dropped frames
/// again, and each arrives once, in order.
#[test]
fn a_link_back_up_before_the_timeout_loses_nothing() {
    let (mut two, mut sender, mut receiver, pool) = one_channel(4);
    for step in 1..=3 {
        send_one(&mut sender, &pool, step, 100).expect("sent");
    }
    two.net
        .with_fabric(|fabric| fabric.set_up(&link(A, B), false));
    two.run(Nanos(500_000_000));
    assert!(received(&mut receiver).is_empty());
    two.net
        .with_fabric(|fabric| fabric.set_up(&link(A, B), true));
    let mut got = Vec::new();
    assert!(two.run_until(Nanos(1_000_000), Nanos(5_000_000_000), || {
        got.extend(received(&mut receiver));
        got.len() >= 3
    }));
    two.run(Nanos(100_000_000));
    got.extend(received(&mut receiver));
    assert_eq!(got, [(1, 1), (2, 2), (3, 3)], "each frame once, in order");
    assert_eq!(two.net.fault(A, &key(0)), None);
    assert!(sender.stats().retransmits > 0);
}

/// A split one way only, from A to B, fails the channel at both ends too: B hears nothing of
/// what A sends.
#[test]
fn a_one_way_split_fails_both_ends() {
    let (mut two, mut sender, _receiver, pool) = one_channel(4);
    send_one(&mut sender, &pool, 1, 100).expect("sent");
    two.net
        .with_fabric(|fabric| fabric.split(A, B, syntropy_testkit::sim::Split::OneWay));
    assert!(two.net.with_fabric(|fabric| fabric.is_up(&link(B, A))));
    let net = two.net.clone();
    assert!(two.run_until(Nanos(10_000_000), Nanos(3_000_000_000), || {
        two_fault(&net, B).is_some()
    }));
    assert!(two_fault(&two.net, A).is_some());
}

/// A connect with no link up to the peer fails with `ConnectTimeout`, at both ends.
#[test]
fn a_connect_with_no_link_fails() {
    let two = Two::new(fixed(Nanos(100_000)), 4);
    two.net
        .with_fabric(|fabric| fabric.set_up(&link(A, B), false));
    let error = open(&two, key(5), 2, 1024, 50).expect_err("no link");
    assert_eq!(error.code, Some(syntropy_net::ErrorCode::CONNECT_TIMEOUT));
    assert_eq!(error.class, ErrorClass::Link);
}

/// The transport reports QUIC's capabilities, and a send names only pools registered with it.
#[test]
fn the_transport_is_quic_like_and_sends_only_registered_pools() {
    let (_two, mut sender, _receiver, _pool) = one_channel(2);
    let stray = BufferPool::heap(PoolId(77), 1024, 2).expect("a pool");
    assert_eq!(
        send_one(&mut sender, &stray, 1, 100),
        Err(SendError::Invalid)
    );
    let two = Two::new(fixed(Nanos(1)), 1);
    let caps = two.a.caps();
    assert!(caps.reliable && caps.encrypted && !caps.zero_copy_recv);
    assert_eq!(caps.max_frame, Bytes(64 << 20));
    assert_eq!(two.a.kind(), syntropy_core::resource::TransportKind::Quic);
    assert_eq!(two.a.node(), A);
}

/// What one run saw: each delivery's virtual time, `chan_seq` and step, the sender's
/// retransmits, and when the last frame arrived.
#[derive(Debug, PartialEq, Eq)]
struct Run {
    deliveries: Vec<(u64, u64, u64)>,
    retransmits: u64,
    end: u64,
}

/// 60 frames through a four-slot window over a lossy, jittery `wifi6_home` link, from `seed`.
fn replay(seed: u64) -> Run {
    let mut model = LinkModel::preset("wifi6_home").expect("a preset");
    model.loss = 0.05;
    model.jitter = Dist::Uniform {
        lo: Nanos(0),
        hi: Nanos(2_000_000),
    };
    let mut two = Two::new(model, seed);
    let pool = BufferPool::heap(PoolId(1), 1024, 16).expect("a pool");
    two.a.register(&pool).expect("registered");
    let (mut sender, mut receiver) = open(&two, key(0), 4, 1024, 10).expect("opened");
    let mut next = 1;
    let mut deliveries = Vec::new();
    let limit = two.clock_now().0 + 60_000_000_000;
    while deliveries.len() < 60 && two.clock_now().0 < limit {
        while next <= 60 && sender.credit() > 0 {
            send_one(&mut sender, &pool, next, 200).expect("credit to send");
            next += 1;
        }
        two.run(Nanos(100_000));
        let now = two.clock_now().0;
        while let Some(frame) = receiver.try_recv() {
            deliveries.push((now, frame.header.spec.chan_seq, frame.header.spec.step));
        }
        while sender.completed().is_some() {}
    }
    Run {
        deliveries,
        retransmits: sender.stats().retransmits,
        end: two.clock_now().0,
    }
}

/// A seed replays one run exactly (RFC-0060 §8): the same seed gives the same deliveries at the
/// same virtual times and the same retransmits, and another seed gives another run. Each run
/// delivers all 60 frames once, in order.
#[test]
fn a_seed_replays_exactly() {
    let in_order = |run: &Run| {
        let order: Vec<(u64, u64)> = run.deliveries.iter().map(|d| (d.1, d.2)).collect();
        order == (1..=60).map(|seq| (seq, seq)).collect::<Vec<_>>()
    };
    let first = replay(7);
    assert!(in_order(&first), "every frame once, in order: {first:?}");
    // The run takes the lossy paths: timers fire and frames go again.
    assert!(first.retransmits > 0, "{first:?}");
    assert_eq!(replay(7), first);
    let other = replay(8);
    assert!(in_order(&other), "{other:?}");
    assert_ne!(other, first);
}
