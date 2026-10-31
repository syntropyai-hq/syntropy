// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bulk pacing in the real backends (DIST-E7-T25; RFC-0031 §7.5, §4.2, §5.3; RFC-0001 §8.4
//! rule 6): two `QuicTransport`s, then two `TcpTransport`s, over loopback, each with a data
//! channel and a `Bulk` channel from A to B.
//!
//! - A paced edge's bulk sends never pass the pacer's rate plus its burst.
//! - The data channel's α samples reach the pacer, and a halving set from them lowers the rate.
//! - With pacing off, bulk sends are not held.
//!
//! The pacer reads a [`SimClock`] the test advances by hand, so what the pacer lets through does
//! not depend on how fast this host runs: the frames cross real loopback sockets, but the rate,
//! the windows and the stamps of the α samples are on the test's clock. The α values themselves
//! are real round trips, so the tests that look at them set bounds no loopback sample can miss.
//! The rig check of ADR-0011 item 8's bound, a 10 GB blob on a 2.5 GbE edge beside a live ring,
//! is not one of these.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use syntropy_core::clock::Clock;
use syntropy_core::plan::WireDType;
use syntropy_core::policy::Fraction;
use syntropy_core::{Bytes, BytesPerSec, InstanceId, MonoTime, Nanos, NodeId, PlanEpoch, StageIx};
use syntropy_net::endpoint::Config as EndpointConfig;
use syntropy_net::frame::{CredKind, Flags, FrameType, HEADER_LEN};
use syntropy_net::transport::core::{CONTROL_RECEIVES, MAX_CONTROL_FRAME};
use syntropy_net::transport::domain::NetDomain;
use syntropy_net::transport::pacing::{BulkPacer, PacedEdges, PacingConfig, WINDOW};
use syntropy_net::transport::quic::{QuicConfig, QuicSettings, QuicTransport};
use syntropy_net::transport::stream::{ChannelAuth, UncheckedAuth};
use syntropy_net::transport::tcp::{TcpConfig, TcpTransport};
use syntropy_net::transport::{
    BufferPool, Channel, ChannelKey, ChannelPriority, ChannelPurpose, ChannelSpec, ExtBuf,
    OutFrame, PoolId, PoolRegistration, RemoteEndpoint, SendError, Transport,
};
use syntropy_net::{Error, ErrorCode};
use syntropy_testkit::sim::SimClock;

/// A bulk frame's payload, and the bytes the pacer counts for it, header included.
const PAYLOAD: u32 = 4096;
const FRAME: u64 = HEADER_LEN as u64 + PAYLOAD as u64;

/// A receive slot: a bulk frame and any control frame, rounded up to 64 bytes.
const SLOT: usize = (if FRAME as usize > MAX_CONTROL_FRAME {
    FRAME as usize
} else {
    MAX_CONTROL_FRAME
})
.next_multiple_of(64);

/// The bulk channel's receive slots, so credit rarely holds a send in these tests.
const BULK_SLOTS: u16 = 64;

/// The edge: β of 1 MB/s, so the floor is 50 kB/s and a step 50 kB/s \[target\], with a 16 KiB
/// burst, four frames.
const BETA: u64 = 1_000_000;
const STEP: u64 = BETA / 20;
const BURST: u64 = 16 * 1024;

/// One test at a time, a test workaround for #2005: a stream channel faults when its `net`
/// domain's thread is starved for about 12 ms with no α sample, and these tests each run two
/// domains and send without pause, so several at once on a small host can starve one. Remove it
/// once #2005 is fixed.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The longest a test waits on the loopback sockets for credit or a sample.
const PATIENCE: Duration = Duration::from_secs(20);

fn config(idle_p99: Nanos) -> PacingConfig {
    PacingConfig {
        beta: BytesPerSec(BETA as f64),
        idle_p99,
        p99_ratio: 1.2,
        floor: Fraction::new(0.05).expect("5 %"),
        burst: BURST,
    }
}

fn key(purpose: ChannelPurpose) -> ChannelKey {
    ChannelKey {
        instance: InstanceId(0x25),
        epoch: PlanEpoch { term: 1, seq: 1 },
        purpose,
        from: StageIx(0),
        to: StageIx(1),
        lane: 0,
    }
}

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

/// Two endpoints of one backend, A and B, each in a `net` domain of its own.
trait Pair {
    fn runtime(&self) -> &tokio::runtime::Runtime;
    /// A's pacing of its peer edges.
    fn pacing(&self) -> &PacedEdges;
    /// B's key, by which A's pacing names the edge to it.
    fn peer(&self) -> [u8; 32];
    fn remote_b(&self) -> RemoteEndpoint;
    fn connect_a(
        &self,
        remote: &RemoteEndpoint,
        spec: ChannelSpec,
    ) -> impl Future<Output = Result<Channel, Error>>;
    fn accept_b(&self, spec: ChannelSpec) -> impl Future<Output = Result<Channel, Error>>;
    fn register(&self, on_a: bool, pool: &BufferPool) -> PoolRegistration;
}

struct Quic {
    runtime: tokio::runtime::Runtime,
    a: QuicTransport,
    b: QuicTransport,
    _domains: [NetDomain; 2],
}

impl Quic {
    fn new() -> Quic {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a test runtime");
        let domains = [
            NetDomain::start("syntropy-net-a").expect("domain a"),
            NetDomain::start("syntropy-net-b").expect("domain b"),
        ];
        let config = || QuicConfig {
            endpoint: EndpointConfig::default(),
            settings: QuicSettings {
                connect_timeout: Some(Duration::from_secs(5)),
                ..QuicSettings::default()
            },
            auth: Arc::new(UncheckedAuth),
            capabilities: 0,
        };
        let (a, b) = runtime.block_on(async {
            let a = QuicTransport::bind(domains[0].handle(), [0xA1; 32], config());
            let b = QuicTransport::bind(domains[1].handle(), [0xB2; 32], config());
            let (a, b) = tokio::join!(a, b);
            (a.expect("bind a"), b.expect("bind b"))
        });
        Quic {
            runtime,
            a,
            b,
            _domains: domains,
        }
    }
}

impl Pair for Quic {
    fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }

    fn pacing(&self) -> &PacedEdges {
        self.a.pacing()
    }

    fn peer(&self) -> [u8; 32] {
        self.b.node_id()
    }

    fn remote_b(&self) -> RemoteEndpoint {
        RemoteEndpoint {
            node: NodeId(self.b.node_id()),
            key: self.b.node_id(),
            addrs: self.b.bound_addrs().into_iter().map(local).collect(),
            relay: None,
        }
    }

    fn connect_a(
        &self,
        remote: &RemoteEndpoint,
        spec: ChannelSpec,
    ) -> impl Future<Output = Result<Channel, Error>> {
        self.a.connect(remote, spec)
    }

    fn accept_b(&self, spec: ChannelSpec) -> impl Future<Output = Result<Channel, Error>> {
        self.b.accept(spec)
    }

    fn register(&self, on_a: bool, pool: &BufferPool) -> PoolRegistration {
        let transport = if on_a { &self.a } else { &self.b };
        transport.register(pool).expect("register")
    }
}

/// Credentials that admit every peer: runner credentials are DIST-E7-T04's.
struct AnyPeer;

impl ChannelAuth for AnyPeer {
    fn credential(&self, _: &ChannelKey) -> (CredKind, Vec<u8>) {
        let kind = CredKind::DataPlaneCredential;
        (kind, vec![0; kind.credential_len()])
    }

    fn check(&self, _: &[u8; 32], _: &ChannelKey, _: CredKind, _: &[u8]) -> Result<(), ErrorCode> {
        Ok(())
    }
}

struct Tcp {
    runtime: tokio::runtime::Runtime,
    a: TcpTransport,
    b: TcpTransport,
    _domains: [NetDomain; 2],
}

impl Tcp {
    fn new() -> Tcp {
        Tcp::with_alpha(Nanos(1_000_000))
    }

    /// A pair whose channels start from an α p99 of `alpha_p99`, so their first ack timeout is
    /// four times it (RFC-0031 §7.3).
    fn with_alpha(alpha_p99: Nanos) -> Tcp {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("a runtime");
        let domains = [
            NetDomain::start("syntropy-net-tcp-a").expect("domain a"),
            NetDomain::start("syntropy-net-tcp-b").expect("domain b"),
        ];
        let config = |seed: u8| TcpConfig {
            seed: [seed; 32],
            auth: Arc::new(AnyPeer),
            capabilities: 0,
            alpha_p99,
            handshake_timeout: Duration::from_secs(5),
        };
        let bind = |domain: &NetDomain, seed| {
            runtime
                .block_on(TcpTransport::bind(
                    domain.handle(),
                    config(seed),
                    SocketAddr::from(([127, 0, 0, 1], 0)),
                ))
                .expect("a listener on loopback")
        };
        let a = bind(&domains[0], 0xA1);
        let b = bind(&domains[1], 0xB2);
        b.allowed().allow(a.public_key());
        Tcp {
            runtime,
            a,
            b,
            _domains: domains,
        }
    }
}

impl Pair for Tcp {
    fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }

    fn pacing(&self) -> &PacedEdges {
        self.a.pacing()
    }

    fn peer(&self) -> [u8; 32] {
        self.b.public_key()
    }

    fn remote_b(&self) -> RemoteEndpoint {
        RemoteEndpoint {
            node: NodeId(self.b.public_key()),
            key: self.b.public_key(),
            addrs: vec![self.b.local_addr()],
            relay: None,
        }
    }

    fn connect_a(
        &self,
        remote: &RemoteEndpoint,
        spec: ChannelSpec,
    ) -> impl Future<Output = Result<Channel, Error>> {
        self.a.connect(remote, spec)
    }

    fn accept_b(&self, spec: ChannelSpec) -> impl Future<Output = Result<Channel, Error>> {
        self.b.accept(spec)
    }

    fn register(&self, on_a: bool, pool: &BufferPool) -> PoolRegistration {
        let transport = if on_a { &self.a } else { &self.b };
        transport.register(pool).expect("register")
    }
}

/// A's data channel and bulk channel to B, B's ends of them, and A's send pool.
struct Edge<P> {
    pair: P,
    data: Channel,
    data_in: Channel,
    bulk: Channel,
    bulk_in: Channel,
    send_pool: BufferPool,
}

impl<P: Pair> Edge<P> {
    fn open(pair: P) -> Edge<P> {
        Edge::open_with(pair, ChannelPurpose::Bulk)
    }

    /// The edge with its bulk channel of `bulk`, `Bulk` or `Replay`.
    fn open_with(pair: P, bulk: ChannelPurpose) -> Edge<P> {
        let send_pool = BufferPool::heap(PoolId(1), PAYLOAD as usize, 512).expect("a send pool");
        pair.register(true, &send_pool);
        let mut next_pool = 2;
        let mut open = |purpose: ChannelPurpose, recv_slots: u16| {
            let mut pool = |on_a: bool, slots: u32| {
                let pool = BufferPool::heap(PoolId(next_pool), SLOT, slots).expect("a pool");
                next_pool += 1;
                pair.register(on_a, &pool)
            };
            let a_pool = pool(true, u32::from(CONTROL_RECEIVES));
            let b_pool = pool(false, u32::from(recv_slots) + u32::from(CONTROL_RECEIVES));
            let spec = |pool: PoolRegistration, recv_slots| ChannelSpec {
                key: key(purpose),
                pool,
                slot_size: Bytes(SLOT as u64),
                recv_slots,
                priority: if matches!(purpose, ChannelPurpose::Bulk | ChannelPurpose::Replay) {
                    ChannelPriority::Bulk
                } else {
                    ChannelPriority::Data
                },
                wire: WireDType::F32,
                retention: recv_slots.max(4),
            };
            let remote = pair.remote_b();
            let (sender, receiver) = pair.runtime().block_on(async {
                tokio::join!(
                    pair.connect_a(&remote, spec(a_pool, 0)),
                    pair.accept_b(spec(b_pool, recv_slots)),
                )
            });
            (sender.expect("A's end"), receiver.expect("B's end"))
        };
        let (data, data_in) = open(ChannelPurpose::Activations, 8);
        let (bulk, bulk_in) = open(bulk, BULK_SLOTS);
        Edge {
            pair,
            data,
            data_in,
            bulk,
            bulk_in,
            send_pool,
        }
    }

    /// Drops what B has received, so its slots come back and its credit flows to A.
    fn drain(&mut self) {
        while self.data_in.try_recv().is_some() {}
        while self.bulk_in.try_recv().is_some() {}
        while self.data.completed().is_some() {}
        while self.bulk.completed().is_some() {}
    }

    /// One frame on `data` or the bulk channel, from the send pool's next free slot.
    fn send(&mut self, bulk: bool) -> Result<(), SendError> {
        let deadline = Instant::now() + PATIENCE;
        let mut slot = loop {
            if let Some(slot) =
                (0..self.send_pool.slots()).find_map(|slot| self.send_pool.claim(slot))
            {
                break slot;
            }
            assert!(Instant::now() < deadline, "no send slot came back");
            self.drain();
            std::thread::sleep(Duration::from_millis(1));
        };
        slot.bytes_mut()[..PAYLOAD as usize].fill(0x5A);
        let reader = slot.freeze();
        let frame = OutFrame {
            ftype: FrameType::Activation,
            step: 0,
            flags: Flags::default(),
            ext: ExtBuf::EMPTY,
            payload: reader.payload(0, PAYLOAD).expect("in the slot"),
        };
        let channel = if bulk { &mut self.bulk } else { &mut self.data };
        channel.try_send(frame).map(|_| ())
    }

    /// Sends bulk frames until the pacer holds one, and returns how many went and when the held
    /// one can go. A send held for credit alone is tried again once B's frames are drained, and
    /// one held until a time not after `now`, the pacer's clock, at once: the `net` task held
    /// the edge for an α sample.
    fn send_until_paced(&mut self, at_most: usize, now: MonoTime) -> (usize, Option<MonoTime>) {
        let deadline = Instant::now() + PATIENCE;
        let mut sent = 0;
        while sent < at_most {
            match self.send(true) {
                Ok(()) => sent += 1,
                Err(SendError::WouldBlock) => match self.bulk.paced_until() {
                    Some(when) if when > now => return (sent, Some(when)),
                    Some(_) => assert!(Instant::now() < deadline, "the edge stayed held"),
                    None => {
                        assert!(Instant::now() < deadline, "credit never came back");
                        self.drain();
                        std::thread::sleep(Duration::from_millis(1));
                    }
                },
                Err(error) => panic!("bulk send: {error}"),
            }
        }
        (sent, None)
    }

    /// One frame on the data channel, which only its credit may hold.
    fn send_data(&mut self) {
        let deadline = Instant::now() + PATIENCE;
        loop {
            match self.send(false) {
                Ok(()) => return,
                Err(SendError::WouldBlock) => {
                    assert_eq!(
                        self.data.paced_until(),
                        None,
                        "the data channel is not paced"
                    );
                    assert!(Instant::now() < deadline, "credit never came back");
                    self.drain();
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("data send: {error}"),
            }
        }
    }

    fn pacer(&self) -> BulkPacer {
        self.pair
            .pacing()
            .pacer(&self.pair.peer())
            .expect("the edge is paced")
    }
}

fn clock() -> (SimClock, Arc<dyn Clock>) {
    let clock = SimClock::new();
    let handle: Arc<dyn Clock> = Arc::new(clock.handle());
    (clock, handle)
}

/// Over twelve windows of the test's clock, in steps of 10 ms, the bytes the bulk channel sent
/// never pass the burst plus the integral of the pacer's rate; each send the pacer holds names
/// a later time, at which the next one goes; and the data channel is never held.
fn paces_at_the_rate_plus_the_burst(pair: impl Pair) {
    let mut edge = Edge::open(pair);
    let (mut clock, handle) = clock();
    // An idle p99 of a second: no loopback sample breaks the bound, so the rate grows a step a
    // window, whether or not samples come.
    edge.pair
        .pacing()
        .pace(&edge.pair.peer(), config(Nanos(1_000_000_000)), handle);
    // A step's tokens, at most 6,000 bytes here, and a frame's leftover fit in the burst, so
    // none is lost to the bucket's cap and the edge can send all it is allowed.
    let steps = 25;
    let step = Nanos(WINDOW.0 / steps);
    let (mut sent, mut allowed) = (0u64, BURST as u128 * 1_000_000_000);
    for window in 0..12u64 {
        let rate = edge.pacer().rate();
        assert_eq!(
            rate,
            (BETA / 20 + window * STEP).min(BETA),
            "window {window}"
        );
        for _ in 0..steps {
            let (frames, held) = edge.send_until_paced(1_000, clock.now());
            sent += frames as u64 * FRAME;
            let held = held.expect("the pacer holds the bulk channel");
            assert!(held > clock.now(), "a held send names a later time");
            assert!(
                u128::from(sent) * 1_000_000_000 <= allowed,
                "window {window}: {sent} bytes against {} allowed",
                allowed / 1_000_000_000
            );
            // The data channel is not paced.
            edge.send_data();
            clock.advance(step);
            allowed += u128::from(rate) * u128::from(step.0);
        }
    }
    // The bucket's own count agrees, and the edge sent all it allowed but the last step's tokens
    // and a frame's leftover.
    assert_eq!(edge.pacer().stats().bytes, sent);
    assert!(
        u128::from(sent + BURST) * 1_000_000_000 >= allowed,
        "{sent} bytes of {}",
        allowed / 1_000_000_000
    );
}

/// The data channel's α samples reach the pacer, stamped on its clock, and a window whose p99 is
/// above the bound halves the rate.
fn alpha_samples_halve_the_rate(pair: impl Pair) {
    let mut edge = Edge::open(pair);
    let (mut clock, handle) = clock();
    // An idle p99 of zero: every sample above 0 ns breaks the bound.
    edge.pair
        .pacing()
        .pace(&edge.pair.peer(), config(Nanos(0)), handle);
    // Ten windows with no sample raise the rate from the floor by ten steps.
    clock.advance(Nanos(10 * WINDOW.0));
    let before = edge.pacer();
    assert_eq!(before.rate(), BETA / 20 + 10 * STEP);
    assert_eq!(before.stats().halvings, 0);
    assert_eq!(before.stats().last_p99, None, "no sample came while idle");
    // Data frames in the eleventh window; each credit echo is an α sample.
    let deadline = Instant::now() + PATIENCE;
    let mut frames = 0;
    while edge.data.stats().alpha_samples < 8 {
        assert!(Instant::now() < deadline, "no α sample came");
        if frames < 64 && edge.send(false).is_ok() {
            frames += 1;
        }
        edge.drain();
        std::thread::sleep(Duration::from_millis(1));
    }
    let samples = edge.data.stats().alpha_samples;
    clock.advance(WINDOW);
    let after = edge.pacer();
    assert!(
        after.stats().last_p99.is_some_and(|p99| p99.0 > 0),
        "the window's p99 came from the data channel's {samples} samples: {:?}",
        after.stats()
    );
    assert_eq!(after.stats().halvings, 1, "{:?}", after.stats());
    assert_eq!(after.rate(), before.rate() / 2);
    assert!(after.rate() < before.rate());
}

/// An edge that is not paced, or no longer, holds no bulk send: many times the burst goes at one
/// instant of the clock, held only by credit.
fn unpaced_bulk_is_not_held(pair: impl Pair) {
    let mut edge = Edge::open(pair);
    let frames = (4 * BURST / FRAME) as usize;
    let (sent, held) = edge.send_until_paced(frames, MonoTime(0));
    assert_eq!((sent, held), (frames, None), "not paced");
    assert!(edge.pair.pacing().pacer(&edge.pair.peer()).is_none());
    // Paced, it holds the bulk channel within the burst; unpaced again, it lets it go.
    let (_clock, handle) = clock();
    edge.pair
        .pacing()
        .pace(&edge.pair.peer(), config(Nanos(1_000_000_000)), handle);
    let (sent, held) = edge.send_until_paced(frames, MonoTime(0));
    assert!(
        held.is_some() && sent as u64 * FRAME <= BURST + FRAME,
        "{sent} frames"
    );
    edge.pair.pacing().unpace(&edge.pair.peer());
    let (sent, held) = edge.send_until_paced(frames, MonoTime(0));
    assert_eq!((sent, held), (frames, None), "unpaced");
    assert_eq!(edge.bulk.paced_until(), None);
}

/// A `Replay` channel is paced as a `Bulk` one is: at one instant of the clock it sends within
/// the burst, then the pacer holds it until a later time.
fn replay_is_paced(pair: impl Pair) {
    let mut edge = Edge::open_with(pair, ChannelPurpose::Replay);
    let (_clock, handle) = clock();
    edge.pair
        .pacing()
        .pace(&edge.pair.peer(), config(Nanos(1_000_000_000)), handle);
    let (sent, held) = edge.send_until_paced((4 * BURST / FRAME) as usize, MonoTime(0));
    assert_eq!(sent as u64, BURST / FRAME, "the burst's whole frames go");
    assert!(held.is_some_and(|when| when > MonoTime(0)), "{held:?}");
    assert_eq!(edge.bulk.paced_until(), held);
}

/// The stream driver sends a batched receiver's held-back credit when it goes idle (RFC-0031
/// §7.1, note of 2026-10-31): one bulk frame, a sixteenth of a batch, is acknowledged without a
/// resend, well before its 400 ms ack timeout. Without it, the frame is acknowledged only after
/// the ack timeout sends it again.
#[test]
fn tcp_an_idle_receiver_acknowledges_a_sparse_bulk_frame() {
    let _serial = serial();
    let mut edge = Edge::open(Tcp::with_alpha(Nanos(100_000_000)));
    edge.send(true).expect("one bulk frame");
    let start = Instant::now();
    let deadline = start + PATIENCE;
    while edge.bulk.completed().is_none() {
        assert!(
            Instant::now() < deadline,
            "the frame was never acknowledged"
        );
        while edge.bulk_in.try_recv().is_some() {}
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        edge.bulk.stats().retransmits,
        0,
        "acknowledged without a resend"
    );
    assert!(
        start.elapsed() < Duration::from_millis(400),
        "{:?}",
        start.elapsed()
    );
}

#[test]
fn quic_replay_is_paced() {
    let _serial = serial();
    replay_is_paced(Quic::new());
}

#[test]
fn tcp_replay_is_paced() {
    let _serial = serial();
    replay_is_paced(Tcp::new());
}

#[test]
fn quic_paces_at_the_rate_plus_the_burst() {
    let _serial = serial();
    paces_at_the_rate_plus_the_burst(Quic::new());
}

#[test]
fn quic_alpha_samples_halve_the_rate() {
    let _serial = serial();
    alpha_samples_halve_the_rate(Quic::new());
}

#[test]
fn quic_unpaced_bulk_is_not_held() {
    let _serial = serial();
    unpaced_bulk_is_not_held(Quic::new());
}

#[test]
fn tcp_paces_at_the_rate_plus_the_burst() {
    let _serial = serial();
    paces_at_the_rate_plus_the_burst(Tcp::new());
}

#[test]
fn tcp_alpha_samples_halve_the_rate() {
    let _serial = serial();
    alpha_samples_halve_the_rate(Tcp::new());
}

#[test]
fn tcp_unpaced_bulk_is_not_held() {
    let _serial = serial();
    unpaced_bulk_is_not_held(Tcp::new());
}
