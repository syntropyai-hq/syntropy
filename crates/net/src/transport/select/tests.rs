// SPDX-License-Identifier: MIT OR Apache-2.0

//! DIST-E7-T11's runtime fallback, over test transports whose connects refuse, hang or succeed.

use std::future::{Future, pending, ready};
use std::pin::pin;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use syntropy_core::plan::WireDType;
use syntropy_core::resource::{LinkCaps, TransportKind, link_cost};
use syntropy_core::{
    BytesPerSec, DeviceId, Endpoint, InstanceId, LinkKind, MeasureSource, Measured, NetMedium,
    NodeId, PlanEpoch, RdmaKind, StageIx,
};

use super::*;
use crate::error::ErrorCode;
use crate::transport::{
    BufferPool, ChannelBackend, ChannelKey, ChannelPriority, ChannelPurpose, ChannelStats,
    CloseReason, InFrame, OutFrame, PoolId, PoolRegistration, SendError, SendTicket, TransportCaps,
};

const TB5: TransportKind = TransportKind::Rdma(RdmaKind::Thunderbolt5);
const TIMEOUT: Duration = Duration::from_secs(2);
const DECODE: Bytes = Bytes(32_768);

/// What a test transport's connect does.
#[derive(Clone)]
enum Connect {
    /// Fails at once with this error.
    Refuse(Error),
    /// Never completes.
    Hang,
    /// Opens a channel.
    Accept,
}

struct TestTransport {
    kind: TransportKind,
    connect: Connect,
    tries: AtomicUsize,
}

impl TestTransport {
    fn new(kind: TransportKind, connect: Connect) -> TestTransport {
        TestTransport {
            kind,
            connect,
            tries: AtomicUsize::new(0),
        }
    }

    fn tries(&self) -> usize {
        self.tries.load(Ordering::Relaxed)
    }
}

impl Transport for TestTransport {
    fn kind(&self) -> TransportKind {
        self.kind
    }

    fn caps(&self) -> TransportCaps {
        TransportCaps {
            zero_copy_recv: false,
            device_memory_recv: false,
            max_frame: Bytes(64 << 20),
            needs_registration: false,
            encrypted: true,
            reliable: true,
            max_channels_per_peer: u16::MAX,
            inline_max: 0,
        }
    }

    fn register(&self, pool: &BufferPool) -> Result<PoolRegistration, Error> {
        let _ = pool;
        Ok(registration(self.kind))
    }

    async fn connect(&self, remote: &RemoteEndpoint, spec: ChannelSpec) -> Result<Channel, Error> {
        let _ = remote;
        self.tries.fetch_add(1, Ordering::Relaxed);
        assert_eq!(
            spec.pool.transport, self.kind,
            "the spec's pool is registered with it"
        );
        match self.connect.clone() {
            Connect::Refuse(error) => Err(error),
            Connect::Hang => pending().await,
            Connect::Accept => Ok(Channel::new(spec.key, Box::new(NullBackend))),
        }
    }

    async fn accept(&self, spec: ChannelSpec) -> Result<Channel, Error> {
        let _ = spec;
        pending().await
    }
}

/// A channel backend that carries nothing: the fallback only opens channels.
struct NullBackend;

impl ChannelBackend for NullBackend {
    fn try_send(&mut self, frame: OutFrame) -> Result<SendTicket, SendError> {
        let _ = frame;
        Err(SendError::WouldBlock)
    }
    fn try_recv(&mut self) -> Option<InFrame> {
        None
    }
    fn completed(&mut self) -> Option<SendTicket> {
        None
    }
    fn stats(&self) -> ChannelStats {
        ChannelStats::default()
    }
    fn credit(&self) -> u32 {
        0
    }
    fn nack(&mut self, seq: u64) {
        let _ = seq;
    }
    fn close(&mut self, reason: CloseReason) {
        let _ = reason;
    }
}

fn registration(kind: TransportKind) -> PoolRegistration {
    PoolRegistration {
        pool: PoolId(0),
        transport: kind,
        handle: 0,
    }
}

fn spec_for(transport: &dyn AnyTransport) -> Result<ChannelSpec, Error> {
    Ok(ChannelSpec {
        key: ChannelKey {
            instance: InstanceId(1),
            epoch: PlanEpoch { term: 1, seq: 1 },
            purpose: ChannelPurpose::Activations,
            from: StageIx(0),
            to: StageIx(1),
            lane: 0,
        },
        pool: registration(transport.kind()),
        slot_size: Bytes(36 << 10),
        recv_slots: 4,
        priority: ChannelPriority::Data,
        wire: WireDType::F32,
        retention: 4,
    })
}

fn remote() -> RemoteEndpoint {
    RemoteEndpoint {
        node: NodeId([2; 32]),
        key: [0; 32],
        addrs: Vec::new(),
        relay: None,
    }
}

fn link(medium: NetMedium) -> LinkId {
    let end = |n: u8| {
        Endpoint::Device(DeviceId {
            node: NodeId([n; 32]),
            ordinal: 0,
        })
    };
    LinkId {
        src: end(1),
        dst: end(2),
        kind: LinkKind::Network(medium),
        path: 0,
    }
}

fn measured<T>(p50: T, p99: T, source: MeasureSource) -> Measured<T> {
    Measured {
        p50,
        p99,
        samples: 200,
        at_unix_ms: 0,
        source,
    }
}

/// Probed costs: α in µs at p50 and p99, β in GB/s.
fn caps(
    medium: NetMedium,
    transport: TransportKind,
    alpha_us: (u64, u64),
    beta_gbps: f64,
) -> LinkCaps {
    LinkCaps {
        link: link(medium),
        transport,
        alpha: measured(
            Nanos(alpha_us.0 * 1_000),
            Nanos(alpha_us.1 * 1_000),
            MeasureSource::Startup,
        ),
        beta: measured(
            BytesPerSec(beta_gbps * 1e9),
            BytesPerSec(beta_gbps * 1e9),
            MeasureSource::Startup,
        ),
        alpha_eff: None,
        loss: None,
    }
}

/// A Thunderbolt 5 RDMA hop between two nodes that also share 10 GbE (QUIC and TCP) and, on one
/// host, shared memory with no probe result.
fn candidates() -> Vec<LinkCaps> {
    let mut shm = caps(
        NetMedium::Loopback,
        TransportKind::SharedMemory,
        (2, 5),
        10.0,
    );
    shm.alpha = Measured::default();
    vec![
        caps(NetMedium::Rdma(RdmaKind::Thunderbolt5), TB5, (15, 25), 2.5),
        caps(NetMedium::Ethernet, TransportKind::Quic, (150, 300), 1.18),
        caps(NetMedium::Ethernet, TransportKind::Tcp, (160, 900), 1.18),
        shm,
    ]
}

fn choice(candidates: &[LinkCaps]) -> HopChoice<'_> {
    HopChoice {
        link: link(NetMedium::Rdma(RdmaKind::Thunderbolt5)),
        planned: TB5,
        candidates,
        frame: DECODE,
        objective: Objective::Latency,
        p99_weight: None,
        connect_timeout: TIMEOUT,
    }
}

/// Runs a future that every test transport and clock here completes, or abandons, at once.
fn run<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("the fallback waited on a future that never wakes"),
    }
}

/// A clock whose every sleep has already passed, which records how long each was.
struct Clock(Mutex<Vec<Duration>>);

impl Clock {
    fn new() -> Clock {
        Clock(Mutex::new(Vec::new()))
    }

    fn sleep(&self) -> impl Fn(Duration) -> std::future::Ready<()> + '_ {
        |d| {
            self.0.lock().unwrap().push(d);
            ready(())
        }
    }

    fn sleeps(&self) -> Vec<Duration> {
        self.0.lock().unwrap().clone()
    }
}

/// The task's criterion: a plan naming RDMA on a host without RDMA connects on QUIC, and returns
/// the demotion with QUIC's cost.
#[test]
fn a_plan_naming_rdma_on_a_host_without_rdma_connects_on_quic() {
    let candidates = candidates();
    let quic = TestTransport::new(TransportKind::Quic, Connect::Accept);
    let tcp = TestTransport::new(TransportKind::Tcp, Connect::Accept);
    let clock = Clock::new();
    let hop = run(connect_hop(
        &choice(&candidates),
        &[&tcp, &quic],
        &remote(),
        spec_for,
        clock.sleep(),
    ))
    .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(hop.transport, TransportKind::Quic);
    assert_eq!(hop.channel.key().instance, InstanceId(1));
    let quic_cost = link_cost(&candidates[1], DECODE, &Objective::Latency, None);
    assert_eq!(
        hop.demoted,
        Some((
            Demotion {
                edge: link(NetMedium::Rdma(RdmaKind::Thunderbolt5)),
                from: TB5,
                to: TransportKind::Quic,
            },
            quic_cost,
        ))
    );
    // 150 µs + 32,768 B at 1.18 GB/s + 0.25 · 150 µs + 25 µs.
    assert_eq!(quic_cost, Nanos(150_000 + 27_769 + 37_500 + 25_000));
    assert_eq!((quic.tries(), tcp.tries()), (1, 0));
}

/// An RDMA backend that refuses (RDMA not enabled) or never connects gives way to QUIC; the hang
/// waits `link.connect_timeout`.
#[test]
fn an_rdma_backend_that_refuses_or_hangs_gives_way_to_quic() {
    let candidates = candidates();
    for (connect, sleeps) in [
        (
            Connect::Refuse(Error::new(ErrorCode::RDMA_NOT_ENABLED, "RDMA is off")),
            vec![TIMEOUT, TIMEOUT],
        ),
        (Connect::Hang, vec![TIMEOUT, TIMEOUT]),
    ] {
        let rdma = TestTransport::new(TB5, connect);
        let quic = TestTransport::new(TransportKind::Quic, Connect::Accept);
        let clock = Clock::new();
        let hop = run(connect_hop(
            &choice(&candidates),
            &[&rdma, &quic],
            &remote(),
            spec_for,
            clock.sleep(),
        ))
        .unwrap_or_else(|e| panic!("{e:?}"));
        assert_eq!(hop.transport, TransportKind::Quic);
        assert_eq!(hop.demoted.map(|(d, _)| d.to), Some(TransportKind::Quic));
        assert_eq!((rdma.tries(), quic.tries()), (1, 1));
        assert_eq!(
            clock.sleeps(),
            sleeps,
            "each attempt waits link.connect_timeout"
        );
    }
}

/// A planned transport that connects is used, with no demotion and no other attempt.
#[test]
fn a_planned_transport_that_connects_is_kept() {
    let candidates = candidates();
    let rdma = TestTransport::new(TB5, Connect::Accept);
    let quic = TestTransport::new(TransportKind::Quic, Connect::Accept);
    let clock = Clock::new();
    let hop = run(connect_hop(
        &choice(&candidates),
        &[&quic, &rdma],
        &remote(),
        spec_for,
        clock.sleep(),
    ))
    .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(hop.transport, TB5);
    assert_eq!(hop.demoted, None);
    assert_eq!((rdma.tries(), quic.tries()), (1, 0));
}

/// After the planned transport, measured candidates go by cost: in throughput mode TCP's lower
/// cost puts it before QUIC; in latency mode its long tail puts it after. Shared memory, with no
/// probe result, comes last despite its rank.
#[test]
fn the_next_candidate_is_the_cheapest_measured_one() {
    let candidates = candidates();
    let throughput = Objective::Throughput {
        per_sequence_floor_tps: 1.0,
        target_concurrency: None,
        ttft_ceiling: None,
    };
    let mut cheaper_tcp = candidates.clone();
    cheaper_tcp[2].alpha = measured(Nanos(140_000), Nanos(900_000), MeasureSource::Startup);
    for (objective, first) in [
        (Objective::Latency, TransportKind::Quic),
        (throughput, TransportKind::Tcp),
    ] {
        let shm = TestTransport::new(TransportKind::SharedMemory, Connect::Hang);
        let quic = TestTransport::new(TransportKind::Quic, Connect::Hang);
        let tcp = TestTransport::new(TransportKind::Tcp, Connect::Hang);
        let clock = Clock::new();
        let hop_choice = HopChoice {
            objective,
            ..choice(&cheaper_tcp)
        };
        let error = run(connect_hop(
            &hop_choice,
            &[&shm, &quic, &tcp],
            &remote(),
            spec_for,
            clock.sleep(),
        ))
        .err()
        .expect("every transport hangs");
        let tried: Vec<TransportKind> = error.attempts.iter().map(|(kind, _)| *kind).collect();
        let second = if first == TransportKind::Quic {
            TransportKind::Tcp
        } else {
            TransportKind::Quic
        };
        assert_eq!(tried, [TB5, first, second, TransportKind::SharedMemory]);
        assert_eq!(error.attempts[0].1, AttemptFailure::Unavailable);
        assert!(
            error.attempts[1..]
                .iter()
                .all(|(_, failure)| *failure == AttemptFailure::TimedOut)
        );
    }
}

/// When nothing connects, the error lists every attempt and why it failed, and a spec that cannot
/// be built counts as a failed attempt.
#[test]
fn no_transport_connecting_reports_every_attempt() {
    let candidates = candidates();
    let quic = TestTransport::new(
        TransportKind::Quic,
        Connect::Refuse(Error::new(ErrorCode::CONNECT_TIMEOUT, "no route")),
    );
    let tcp = TestTransport::new(TransportKind::Tcp, Connect::Accept);
    let clock = Clock::new();
    let no_tcp_spec = |t: &dyn AnyTransport| {
        if t.kind() == TransportKind::Tcp {
            Err(Error::new(ErrorCode::FRAME_TOO_LARGE, "no pool"))
        } else {
            spec_for(t)
        }
    };
    let error = run(connect_hop(
        &choice(&candidates),
        &[&quic, &tcp],
        &remote(),
        no_tcp_spec,
        clock.sleep(),
    ))
    .err()
    .expect("nothing connects");
    assert_eq!(
        error.attempts,
        [
            (TB5, AttemptFailure::Unavailable),
            (
                TransportKind::Quic,
                AttemptFailure::Failed(Error::new(ErrorCode::CONNECT_TIMEOUT, "no route"))
            ),
            (
                TransportKind::Tcp,
                AttemptFailure::Failed(Error::new(ErrorCode::FRAME_TOO_LARGE, "no pool"))
            ),
            (TransportKind::SharedMemory, AttemptFailure::Unavailable),
        ]
    );
    assert_eq!(tcp.tries(), 0, "no connect without a spec");
    assert_eq!(
        error.class(),
        ErrorClass::Link,
        "every candidate failed: the edge is down"
    );
}

/// An `Auth` refusal or a cancellation ends the fallback, from the connect or from building the
/// spec: no other transport is tried, and the error keeps the class (RFC-0001 §10.1 never
/// retries `Auth` automatically).
#[test]
fn an_auth_refusal_or_a_cancellation_ends_the_fallback() {
    let candidates = candidates();
    for error in [
        Error::new(ErrorCode::CREDENTIAL_INVALID, "bad credential"),
        Error::new(ErrorCode::NOT_A_NEIGHBOUR, "not a neighbour"),
        Error::uncoded(ErrorClass::Cancelled, "the caller left"),
    ] {
        let rdma = TestTransport::new(TB5, Connect::Refuse(error.clone()));
        let quic = TestTransport::new(TransportKind::Quic, Connect::Accept);
        let clock = Clock::new();
        let failure = run(connect_hop(
            &choice(&candidates),
            &[&rdma, &quic],
            &remote(),
            spec_for,
            clock.sleep(),
        ))
        .err()
        .expect("the fallback stops");
        assert_eq!(
            failure.attempts,
            [(TB5, AttemptFailure::Failed(error.clone()))]
        );
        assert_eq!(failure.class(), error.class);
        assert_eq!(quic.tries(), 0, "no other transport is tried");

        let rdma = TestTransport::new(TB5, Connect::Accept);
        let quic = TestTransport::new(TransportKind::Quic, Connect::Accept);
        let refused = |t: &dyn AnyTransport| {
            if t.kind() == TB5 {
                Err(error.clone())
            } else {
                spec_for(t)
            }
        };
        let failure = run(connect_hop(
            &choice(&candidates),
            &[&rdma, &quic],
            &remote(),
            refused,
            clock.sleep(),
        ))
        .err()
        .expect("the fallback stops");
        assert_eq!(
            failure.attempts,
            [(TB5, AttemptFailure::Failed(error.clone()))]
        );
        assert_eq!(failure.class(), error.class);
        assert_eq!((rdma.tries(), quic.tries()), (0, 0));
    }
}

/// The planned transport goes first even when another candidate costs less: a plan naming QUIC
/// on a hop that also has Thunderbolt 5 RDMA keeps QUIC when it connects, and moves to RDMA, the
/// cheapest other candidate, only when QUIC fails.
#[test]
fn the_planned_transport_goes_first_even_when_another_costs_less() {
    let candidates = candidates();
    let quic_planned = HopChoice {
        link: link(NetMedium::Ethernet),
        planned: TransportKind::Quic,
        ..choice(&candidates)
    };
    let rdma = TestTransport::new(TB5, Connect::Accept);
    let quic = TestTransport::new(TransportKind::Quic, Connect::Accept);
    let clock = Clock::new();
    let hop = run(connect_hop(
        &quic_planned,
        &[&rdma, &quic],
        &remote(),
        spec_for,
        clock.sleep(),
    ))
    .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(hop.transport, TransportKind::Quic);
    assert_eq!(hop.demoted, None);
    assert_eq!((quic.tries(), rdma.tries()), (1, 0));

    let rdma = TestTransport::new(TB5, Connect::Accept);
    let quic = TestTransport::new(
        TransportKind::Quic,
        Connect::Refuse(Error::new(ErrorCode::CONNECT_TIMEOUT, "no route")),
    );
    let hop = run(connect_hop(
        &quic_planned,
        &[&rdma, &quic],
        &remote(),
        spec_for,
        clock.sleep(),
    ))
    .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(hop.transport, TB5);
    assert_eq!(
        hop.demoted.map(|(d, _)| (d.from, d.to)),
        Some((TransportKind::Quic, TB5))
    );
    assert_eq!((quic.tries(), rdma.tries()), (1, 1));
}

/// A configured `link.p99_weight` ranks the candidates and prices the demotion: with no weight on
/// the tail, TCP's lower median puts it before QUIC; at the default, its long tail puts it after.
#[test]
fn a_configured_p99_weight_ranks_the_candidates_and_prices_the_demotion() {
    let mut candidates = candidates();
    candidates[2].alpha = measured(Nanos(140_000), Nanos(900_000), MeasureSource::Startup);
    let (quic_caps, tcp_caps) = (candidates[1].clone(), candidates[2].clone());
    let cost = |c: &LinkCaps, weight| link_cost(c, DECODE, &Objective::Latency, weight);
    assert!(cost(&tcp_caps, Some(0.0)) < cost(&quic_caps, Some(0.0)));
    assert!(cost(&quic_caps, None) < cost(&tcp_caps, None));
    for (weight, expected, caps) in [
        (Some(0.0), TransportKind::Tcp, &tcp_caps),
        (None, TransportKind::Quic, &quic_caps),
    ] {
        let quic = TestTransport::new(TransportKind::Quic, Connect::Accept);
        let tcp = TestTransport::new(TransportKind::Tcp, Connect::Accept);
        let clock = Clock::new();
        let weighted = HopChoice {
            p99_weight: weight,
            ..choice(&candidates)
        };
        let hop = run(connect_hop(
            &weighted,
            &[&quic, &tcp],
            &remote(),
            spec_for,
            clock.sleep(),
        ))
        .unwrap_or_else(|e| panic!("{e:?}"));
        assert_eq!(hop.transport, expected, "p99_weight {weight:?}");
        assert_eq!(hop.demoted.map(|(_, c)| c), Some(cost(caps, weight)));
    }
}

/// `Transport::cost` is `core`'s `link_cost` at `link.p99_weight`'s default.
#[test]
fn transport_cost_is_cores_link_cost() {
    let quic = TestTransport::new(TransportKind::Quic, Connect::Accept);
    let wifi = caps(NetMedium::WiFi, TransportKind::Quic, (4_000, 250_000), 0.05);
    for size in [Bytes(128), DECODE, Bytes(16 << 20)] {
        for objective in [
            Objective::Latency,
            Objective::Throughput {
                per_sequence_floor_tps: 1.0,
                target_concurrency: None,
                ttft_ceiling: None,
            },
        ] {
            assert_eq!(
                quic.cost(&wifi, size, objective),
                link_cost(&wifi, size, &objective, None)
            );
        }
    }
}
