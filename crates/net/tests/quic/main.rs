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
use syntropy_core::{Bytes, ErrorClass, InstanceId, NodeId, PlanEpoch, StageIx};
use syntropy_net::ErrorCode;
use syntropy_net::endpoint::Config as EndpointConfig;
use syntropy_net::transport::conformance::{self, Fabric, MAX_FRAME, Rule, SEND_SLOT, SEND_SLOTS};
use syntropy_net::transport::core::{CONTROL_RECEIVES, MAX_CONTROL_FRAME};
use syntropy_net::transport::domain::NetDomain;
use syntropy_net::transport::quic::{QuicConfig, QuicSettings, QuicTransport, Role, WindowReport};
use syntropy_net::transport::stream::UncheckedAuth;
use syntropy_net::transport::{
    BufferPool, Channel, ChannelKey, ChannelPriority, ChannelPurpose, ChannelSpec, PoolId,
    RemoteEndpoint, Transport,
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

    // Every channel the suite opened rode the one latency connection, and its receive window is
    // the sum of their credits (RFC-0031 §4.3). The receiver's side holds the data credits.
    let windows = Pair::windows(&fabric.pair.b);
    assert_eq!(windows.len(), 1, "{windows:#?}");
    let latency = windows[0];
    assert_eq!(latency.role, Role::Latency);
    assert_eq!(latency.peer, fabric.pair.a.node_id());
    let per_channel = u64::from(conformance::RECV_SLOTS) * SLOT as u64
        + u64::from(CONTROL_RECEIVES) * MAX_CONTROL_FRAME as u64;
    assert_eq!(latency.channels, Rule::ALL.len());
    assert_eq!(latency.credited, per_channel * Rule::ALL.len() as u64);
    assert_eq!(latency.window, latency.credited);
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
