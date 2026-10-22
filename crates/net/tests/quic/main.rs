// SPDX-License-Identifier: MIT OR Apache-2.0

//! The QUIC backend (DIST-E7-T03; RFC-0031 §4) between two endpoints on this host: the
//! channel-conformance suite over real QUIC streams, the connection receive windows, the two
//! connections per peer, the LAN initial window, the path kind, and a timeout told from a
//! refusal.
//!
//! Each endpoint runs in a `net` domain of its own, as two runners would. The tests dial over
//! loopback with no relay, so they need no network beyond this host. The run over 2.5 GbE between
//! two nodes is a lab check, not one of these.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use syntropy_core::plan::WireDType;
use syntropy_core::resource::TransportKind;
use syntropy_core::{Bytes, ErrorClass, InstanceId, NodeId, PlanEpoch, SequenceId, StageIx};
use syntropy_net::ErrorCode;
use syntropy_net::endpoint::Config as EndpointConfig;
use syntropy_net::frame::{
    Activation, ActivationHeader, ActivationKind, Flags, FrameType, RowLayout, SideItems, Span,
    Spans, TreeArea,
};
use syntropy_net::transport::conformance::{self, Fabric, MAX_FRAME, Rule, SEND_SLOT, SEND_SLOTS};
use syntropy_net::transport::core::{CONTROL_RECEIVES, MAX_CONTROL_FRAME};
use syntropy_net::transport::domain::NetDomain;
use syntropy_net::transport::quic::{QuicConfig, QuicSettings, QuicTransport, Role, WindowReport};
use syntropy_net::transport::stream::UncheckedAuth;
use syntropy_net::transport::{
    BufferPool, Channel, ChannelKey, ChannelPriority, ChannelPurpose, ChannelSpec, OutFrame,
    PoolId, RemoteEndpoint, SendError, SlotReader, Transport,
};

/// A receive slot: the suite's largest frame and any control frame, rounded up to 64 bytes.
const SLOT: usize = (if MAX_FRAME > MAX_CONTROL_FRAME {
    MAX_FRAME
} else {
    MAX_CONTROL_FRAME
})
.next_multiple_of(64);

/// A bound socket as a peer on this host dials it: an unspecified IP becomes loopback.
fn local(addr: SocketAddr) -> SocketAddr {
    match addr.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => {
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), addr.port())
        }
        IpAddr::V6(ip) if ip.is_unspecified() => {
            SocketAddr::new(Ipv6Addr::LOCALHOST.into(), addr.port())
        }
        _ => addr,
    }
}

/// Two endpoints, each in a domain of its own, and a runtime the test thread drives their
/// futures on.
struct Pair {
    runtime: tokio::runtime::Runtime,
    a: QuicTransport,
    b: QuicTransport,
    // Dropped after the transports, which run in them.
    _domains: [NetDomain; 2],
    next_pool: u32,
}

fn settings() -> QuicSettings {
    QuicSettings {
        connect_timeout: Some(Duration::from_secs(5)),
        ..QuicSettings::default()
    }
}

fn config(settings: QuicSettings) -> QuicConfig {
    QuicConfig {
        endpoint: EndpointConfig::default(),
        settings,
        auth: Arc::new(UncheckedAuth),
        capabilities: 0,
    }
}

impl Pair {
    fn new() -> Pair {
        Pair::with(settings(), settings())
    }

    fn with(a_settings: QuicSettings, b_settings: QuicSettings) -> Pair {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a test runtime");
        let domains = [
            NetDomain::start("syntropy-net-a").expect("domain a"),
            NetDomain::start("syntropy-net-b").expect("domain b"),
        ];
        let (a, b) = runtime.block_on(async {
            let a = QuicTransport::bind(domains[0].handle(), [0xA1; 32], config(a_settings));
            let b = QuicTransport::bind(domains[1].handle(), [0xB2; 32], config(b_settings));
            let (a, b) = tokio::join!(a, b);
            (a.expect("bind a"), b.expect("bind b"))
        });
        Pair {
            runtime,
            a,
            b,
            _domains: domains,
            next_pool: 1,
        }
    }

    /// `b` as `a` dials it: its key and its loopback addresses, no relay.
    fn remote_b(&self) -> RemoteEndpoint {
        RemoteEndpoint {
            node: NodeId(self.b.node_id()),
            key: self.b.node_id(),
            addrs: self.b.bound_addrs().into_iter().map(local).collect(),
            relay: None,
        }
    }

    /// A fresh receive pool of `slots` slots, registered with `transport`.
    fn pool(
        &mut self,
        on_a: bool,
        slots: u32,
    ) -> (BufferPool, syntropy_net::transport::PoolRegistration) {
        let id = PoolId(self.next_pool);
        self.next_pool += 1;
        let pool = BufferPool::heap(id, SLOT, slots).expect("a receive pool");
        let transport = if on_a { &self.a } else { &self.b };
        let registration = transport.register(&pool).expect("register");
        (pool, registration)
    }

    /// Opens `key` from `a` (the sender) to `b` (the receiver, with `recv_slots` slots).
    fn open(
        &mut self,
        key: ChannelKey,
        recv_slots: u16,
    ) -> Result<(Channel, Channel), syntropy_net::Error> {
        let (_, a_pool) = self.pool(true, u32::from(CONTROL_RECEIVES));
        let (_, b_pool) = self.pool(false, u32::from(recv_slots) + u32::from(CONTROL_RECEIVES));
        let spec = |pool, recv_slots| ChannelSpec {
            key,
            pool,
            slot_size: Bytes(SLOT as u64),
            recv_slots,
            priority: ChannelPriority::Data,
            wire: WireDType::F32,
            retention: recv_slots.max(4),
        };
        let remote = self.remote_b();
        let (sender, receiver) = self.runtime.block_on(async {
            tokio::join!(
                self.a.connect(&remote, spec(a_pool, 0)),
                self.b.accept(spec(b_pool, recv_slots)),
            )
        });
        Ok((sender?, receiver?))
    }

    fn windows(transport: &QuicTransport) -> Vec<WindowReport> {
        let mut windows = transport.connection_windows();
        windows.sort_by_key(|w| (w.role == Role::Bulk, w.channels));
        windows
    }
}

/// The suite's fabric: channels from `a` to `b`, with a send pool registered with `a`.
struct QuicFabric {
    pair: Pair,
    send_pool: BufferPool,
}

impl QuicFabric {
    fn new() -> QuicFabric {
        let pair = Pair::new();
        let send_pool = BufferPool::heap(PoolId(0), SEND_SLOT, SEND_SLOTS).expect("a send pool");
        pair.a.register(&send_pool).expect("register the send pool");
        QuicFabric { pair, send_pool }
    }
}

impl Fabric for QuicFabric {
    fn open(
        &mut self,
        key: ChannelKey,
        recv_slots: u16,
    ) -> Result<(Channel, Channel), syntropy_net::Error> {
        self.pair.open(key, recv_slots)
    }

    fn send_pool(&self) -> &BufferPool {
        &self.send_pool
    }

    fn register(
        &mut self,
        pool: &BufferPool,
    ) -> Option<Result<syntropy_net::transport::PoolRegistration, syntropy_net::Error>> {
        Some(self.pair.a.register(pool))
    }

    /// Frames, credits and acknowledgements move on their own; the suite waits a little.
    fn pump(&mut self) {
        std::thread::sleep(Duration::from_micros(500));
    }
}

fn key(purpose: ChannelPurpose, lane: u8) -> ChannelKey {
    ChannelKey {
        instance: InstanceId(0x5359_4e46),
        epoch: PlanEpoch { term: 3, seq: 9 },
        purpose,
        from: StageIx(0),
        to: StageIx(1),
        lane,
    }
}

#[test]
fn the_channel_conformance_suite_passes_over_quic_on_loopback() {
    let mut fabric = QuicFabric::new();
    let report = conformance::run(&mut fabric);
    assert!(report.failures.is_empty(), "{:#?}", report.failures);
    // QUIC delivers what it has sent; the fabric cannot hold it in flight.
    assert_eq!(report.skipped, [Rule::CompletedAfterAck]);

    // Every channel the suite opened rode the one latency connection. Each check drops its
    // channels, so once their tasks end the receive window is back at its floor: it covers the
    // open channels only (RFC-0031 §4.3). The sums at each open are the next test's.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let latency = loop {
        let windows = Pair::windows(&fabric.pair.b);
        assert_eq!(windows.len(), 1, "{windows:#?}");
        if windows[0].channels == 0 || std::time::Instant::now() > deadline {
            break windows[0];
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(latency.role, Role::Latency);
    assert_eq!(latency.peer, fabric.pair.a.node_id());
    assert_eq!(
        (latency.channels, latency.credited, latency.window),
        (0, 0, QuicSettings::receive_window_floor())
    );
    assert!(!latency.over_cap);

    // Each path has a controller. Whether it took the LAN window depends on the first round
    // trip measured, which on a loaded host can reach 2 ms even on loopback; the controller's
    // unit tests pin that rule. Never below noq-proto's 12,000-byte default.
    let windows = fabric.pair.a.congestion_windows(&fabric.pair.b.node_id());
    assert!(!windows.is_empty());
    assert!(
        windows.iter().all(|w| *w >= 12_000),
        "congestion windows {windows:?}"
    );

    // A direct path, reported as `Quic`, never `QuicRelay` (RFC-0031 §4.1).
    assert_eq!(
        fabric.pair.a.path_kind(&fabric.pair.b.node_id()),
        Some(TransportKind::Quic)
    );
}

#[test]
fn the_receive_window_follows_the_credited_sum_at_each_channel_open() {
    let mut pair = Pair::new();
    let per_slot = SLOT as u64;
    let control = u64::from(CONTROL_RECEIVES) * MAX_CONTROL_FRAME as u64;
    let mut expected = 0;
    let mut open = Vec::new();
    for (lane, slots) in [(0u8, 4u16), (1, 16), (2, 64)] {
        open.push(
            pair.open(key(ChannelPurpose::Activations, lane), slots)
                .expect("open"),
        );
        expected += u64::from(slots) * per_slot + control;
        let windows = Pair::windows(&pair.b);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].credited, expected, "after lane {lane}");
        assert_eq!(windows[0].window, expected, "after lane {lane}");
    }
}

#[test]
fn replay_and_bulk_channels_take_a_second_connection() {
    let mut pair = Pair::new();
    let _decode = pair
        .open(key(ChannelPurpose::Activations, 0), 4)
        .expect("open");
    let _replay = pair.open(key(ChannelPurpose::Replay, 0), 4).expect("open");
    let _bulk = pair.open(key(ChannelPurpose::Bulk, 0), 4).expect("open");
    let windows = Pair::windows(&pair.b);
    let roles: Vec<(Role, usize)> = windows.iter().map(|w| (w.role, w.channels)).collect();
    assert_eq!(roles, [(Role::Latency, 1), (Role::Bulk, 2)], "{windows:#?}");
    let dialled = Pair::windows(&pair.a);
    assert_eq!(dialled.len(), 2, "{dialled:#?}");
}

#[test]
fn a_slot_larger_than_max_frame_is_refused_at_open() {
    let small = QuicSettings {
        max_frame: 256,
        ..settings()
    };
    let mut pair = Pair::with(small, small);
    let error = pair
        .open(key(ChannelPurpose::Activations, 0), 4)
        .expect_err("a slot over net.max_frame");
    assert_eq!(error.class, ErrorClass::InvalidRequest, "{error}");
}

#[test]
fn a_dial_that_outlasts_the_connect_timeout_is_connect_timeout() {
    let short = QuicSettings {
        connect_timeout: Some(Duration::from_millis(300)),
        ..settings()
    };
    let mut pair = Pair::with(short, settings());
    // `b`'s key at an address where nothing answers.
    let unused = std::net::UdpSocket::bind("127.0.0.1:0").expect("a free port");
    let silent = RemoteEndpoint {
        addrs: vec![unused.local_addr().expect("its address")],
        ..pair.remote_b()
    };
    let (_, registration) = pair.pool(true, 4);
    let spec = ChannelSpec {
        key: key(ChannelPurpose::Activations, 0),
        pool: registration,
        slot_size: Bytes(SLOT as u64),
        recv_slots: 0,
        priority: ChannelPriority::Data,
        wire: WireDType::F32,
        retention: 4,
    };
    let error = pair
        .runtime
        .block_on(pair.a.connect(&silent, spec))
        .expect_err("nothing answers");
    assert_eq!(error.code, Some(ErrorCode::CONNECT_TIMEOUT), "{error}");
    assert_eq!(error.class, ErrorClass::Link);
    drop(unused);
}

/// A channel `a` opens and `b` never accepts, with the same default settings on both sides, as a
/// cluster runs: `b` refuses it with `NotANeighbour` (`Auth`) within `a`'s wait, so `a` learns
/// that the plans disagree rather than timing out (`Link`) and reconnecting (review of #1826, B1).
#[test]
fn a_channel_nobody_accepts_is_refused_with_its_code_not_timed_out() {
    // A dial to loopback addresses is a LAN link (2 s); one that also names an address outside
    // the LAN is a WAN link (10 s), though the loopback path still connects. TEST-NET-3 is
    // unroutable.
    let wan_hint: SocketAddr = "203.0.113.1:9".parse().expect("an address");
    for (class, extra, wait) in [
        ("LAN", None, Duration::from_secs(2)),
        ("WAN", Some(wan_hint), Duration::from_secs(10)),
    ] {
        let mut pair = Pair::with(QuicSettings::default(), QuicSettings::default());
        let (_, registration) = pair.pool(true, 4);
        let spec = ChannelSpec {
            key: key(ChannelPurpose::Activations, 7),
            pool: registration,
            slot_size: Bytes(SLOT as u64),
            recv_slots: 0,
            priority: ChannelPriority::Data,
            wire: WireDType::F32,
            retention: 4,
        };
        let mut remote = pair.remote_b();
        remote.addrs.extend(extra);
        let started = std::time::Instant::now();
        let error = pair
            .runtime
            .block_on(pair.a.connect(&remote, spec))
            .expect_err("b never accepts the channel");
        let took = started.elapsed();
        assert_eq!(
            error.code,
            Some(ErrorCode::NOT_A_NEIGHBOUR),
            "{class}: {error}"
        );
        assert_eq!(error.class, ErrorClass::Auth, "{class}");
        // Refused after the hold, half the shorter connect timeout, and before the connector's
        // own wait ran out.
        let hold = QuicSettings::default().early_hold();
        assert!(
            took >= hold && took < wait,
            "{class}: refused after {took:?}"
        );
    }
}

/// The connection receive window covers the open channels only: it falls back when channels
/// close, and rises again when one opens (RFC-0031 §4.3; review of #1826, N1).
#[test]
fn closing_a_channel_lowers_the_receive_window_again() {
    let mut pair = Pair::new();
    let per_channel = 4 * SLOT as u64 + u64::from(CONTROL_RECEIVES) * MAX_CONTROL_FRAME as u64;
    let floor = QuicSettings::receive_window_floor();
    let wait_for = |pair: &Pair, channels: usize| {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let windows = Pair::windows(&pair.b);
            if windows.len() == 1 && windows[0].channels == channels {
                return windows[0];
            }
            assert!(
                std::time::Instant::now() < deadline,
                "never reached {channels} open channels: {windows:#?}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    for round in 0..3u8 {
        let channels = pair
            .open(key(ChannelPurpose::Activations, round), 4)
            .expect("open");
        let open = wait_for(&pair, 1);
        assert_eq!(
            (open.credited, open.window),
            (per_channel, per_channel),
            "round {round}"
        );
        drop(channels);
        let closed = wait_for(&pair, 0);
        assert_eq!(
            (closed.credited, closed.window),
            (0, floor),
            "round {round}"
        );
    }
    // The opening side gives its share back too.
    let dialled = Pair::windows(&pair.a);
    assert_eq!(dialled.len(), 1);
    assert_eq!(dialled[0].channels, 0, "{dialled:#?}");
}

/// A one-row `Activation` frame in a free slot of `pool`: its extension at offset 0, then its
/// payload, with the slot's first reader, which the caller holds.
fn activation_frame(pool: &BufferPool) -> (SlotReader, OutFrame) {
    let spans = [Span {
        seq: SequenceId {
            ingress: 1,
            boot: 1,
            n: 1,
        },
        first_position: 0,
        n_positions: 1,
        row_offset: 0,
        tree: None,
    }];
    let payload = [0x5Au8; 64];
    let chunk = Activation {
        header: ActivationHeader {
            from_stage: StageIx(0),
            to_stage: StageIx(1),
            kind: ActivationKind::Decode,
            layout: RowLayout {
                dtype: WireDType::F32,
                streams: 1,
                hidden: 16,
            },
            n_rows: 1,
            part_index: 0,
            part_count: 1,
        },
        spans: Spans::new(&spans),
        tree: TreeArea::new(&[]),
        side: SideItems::new(&[]),
        payload: &payload,
    };
    let mut writer = pool.take().expect("a free send slot");
    let ext_len = chunk
        .write_ext(writer.bytes_mut())
        .expect("the extension fits");
    writer.bytes_mut()[ext_len..ext_len + payload.len()].copy_from_slice(&payload);
    let reader = writer.freeze();
    let frame = OutFrame {
        ftype: FrameType::Activation,
        step: 1,
        flags: Flags(Flags::FIRST_OF_SEQ),
        ext: reader.ext(0, ext_len as u32).expect("in the slot"),
        payload: reader
            .payload(ext_len as u32, payload.len() as u32)
            .expect("in the slot"),
    };
    (reader, frame)
}

/// RFC-0031 §8's note of 2026-10-22, through a channel: a descriptor whose slot was freed and
/// handed out again since is refused with `SendError::Invalid`, never leased or sent. The same
/// holds for a stale `ExtBuf`.
#[test]
fn a_stale_descriptor_is_refused_with_invalid_and_never_sent() {
    let mut fabric = QuicFabric::new();
    let (mut sender, mut receiver) = fabric
        .open(key(ChannelPurpose::Activations, 0), 4)
        .expect("open");
    let pool = fabric.send_pool().clone();

    // The holder drops the slot's last reader; the slot is claimed and frozen again.
    let (reader, stale) = activation_frame(&pool);
    let slot = reader.slot();
    drop(reader);
    let refrozen = pool.claim(slot).expect("the freed slot").freeze();
    assert_eq!(sender.try_send(stale), Err(SendError::Invalid));

    // A fresh payload with the stale extension is refused too.
    let (fresh_reader, fresh) = activation_frame(&pool);
    let stale_ext = OutFrame {
        ext: stale.ext,
        ..fresh
    };
    assert_eq!(sender.try_send(stale_ext), Err(SendError::Invalid));

    // Nothing was sent: the first frame the channel takes is chan_seq 1, and it arrives.
    assert_eq!(sender.try_send(fresh).map(|t| t.chan_seq), Ok(1));
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let frame = loop {
        if let Some(frame) = receiver.try_recv() {
            break frame;
        }
        assert!(std::time::Instant::now() < deadline, "the frame never came");
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(frame.header.spec.chan_seq, 1);
    assert!(receiver.try_recv().is_none());
    drop((refrozen, fresh_reader));
}

/// RFC-0031 §8's receive-path row for QUIC: the frame lands in a slot of the receiver's
/// registered pool, one host copy from the transport's decrypt buffer, and the consumer reads it
/// there. The sender sends from its own registered slot, which it keeps until the frame is
/// acknowledged (the conformance suite's `Retention` rule), so no copy is made before TLS reads
/// it.
#[test]
fn a_frame_lands_in_a_registered_receive_slot_after_one_host_copy() {
    let mut fabric = QuicFabric::new();
    let (mut sender, mut receiver) = fabric
        .open(key(ChannelPurpose::Activations, 1), 4)
        .expect("open");
    // `Pair::open` makes the sender's pool, then the receiver's.
    let receive_pool = PoolId(fabric.pair.next_pool - 1);
    let pool = fabric.send_pool().clone();
    let (reader, frame) = activation_frame(&pool);
    let send_slot = reader.slot();
    let ticket = sender.try_send(frame).expect("sent");
    drop(reader);

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let received = loop {
        if let Some(frame) = receiver.try_recv() {
            break frame;
        }
        assert!(std::time::Instant::now() < deadline, "the frame never came");
        std::thread::sleep(Duration::from_millis(1));
    };
    let (pool_id, slot) = received.payload.slot().expect("a payload in a slot");
    assert_eq!(
        pool_id, receive_pool,
        "landed in the receiver's registered pool"
    );
    assert_eq!(
        received.ext.slot(),
        Some((receive_pool, slot)),
        "one slot per frame"
    );
    assert_eq!(received.payload.bytes(), &[0x5A; 64]);

    // The sender sent from its slot and kept it leased until the peer acknowledged the frame:
    // the conformance suite's `Retention` rule holds the frame back to check that, which a live
    // link cannot. Here the slot is free once the ticket is back.
    let completed = loop {
        if let Some(done) = sender.completed() {
            break done;
        }
        assert!(std::time::Instant::now() < deadline, "no ticket");
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(completed, ticket);
    assert!(
        pool.claim(send_slot).is_some(),
        "released once acknowledged"
    );
    drop(received);
}
