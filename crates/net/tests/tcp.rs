// SPDX-License-Identifier: MIT OR Apache-2.0

//! The TCP fallback backend (DIST-E7-T05; RFC-0031 §5.3, ADR-0011 item 2) over loopback.
//!
//! - **The criterion.** With UDP blocked, a two-node ring forms over TCP and passes the
//!   channel-conformance suite. A hosted runner cannot apply a firewall rule unprivileged, so the
//!   block is a stand-in: node B's UDP port is a socket that never reads, so every datagram to it
//!   is dropped, as a `DROP` rule drops it. A QUIC dial from A to B then fails, and A opens its
//!   channel to B over TCP; B does the same back to A. Both directions of the ring pass the suite.
//! - **Identity.** A connector that pins the wrong key, and a connector whose key the acceptor
//!   does not allow, are both refused with an `Auth` error; so is a credential the check refuses.

use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
use std::time::Duration;

use syntropy_core::plan::WireDType;
use syntropy_core::{Bytes, ErrorClass, InstanceId, Nanos, NodeId, PlanEpoch, SequenceId, StageIx};
use syntropy_net::Error;
use syntropy_net::endpoint::{self, Endpoint, NodeAddr};
use syntropy_net::error::ErrorCode;
use syntropy_net::frame::{
    Activation, ActivationHeader, ActivationKind, Flags, FrameType, RowLayout, SideItems, Span,
    Spans, TreeArea,
};
use syntropy_net::frame::{CredKind, Hello, HelloAck};
use syntropy_net::transport::conformance::{self, Fabric, MAX_FRAME, Rule, SEND_SLOT, SEND_SLOTS};
use syntropy_net::transport::core::{CONTROL_RECEIVES, MAX_CONTROL_FRAME};
use syntropy_net::transport::tcp::{
    Credential, CredentialCheck, MAX_HANDSHAKES_PER_ADDRESS, TLS_TIMEOUT, TcpConfig, TcpTransport,
};
use syntropy_net::transport::{
    BufferPool, Channel, ChannelKey, ChannelPriority, ChannelPurpose, ChannelSpec, OutFrame,
    PoolId, RemoteEndpoint, SlotReader, Transport,
};
use tokio::io::AsyncReadExt;

/// A check that admits every credential, or refuses each with `refuse`: runner credentials and
/// their checks are DIST-E7-T04's, so these tests stand in for them.
struct StubCheck {
    refuse: Option<ErrorCode>,
}

impl CredentialCheck for StubCheck {
    fn hello(&self, _: &Hello, _: CredKind, _: &[u8], _: &[u8; 32]) -> Result<(), ErrorCode> {
        self.refuse.map_or(Ok(()), Err)
    }

    fn hello_ack(
        &self,
        _: &HelloAck,
        _: CredKind,
        _: &[u8],
        _: &[u8; 32],
    ) -> Result<(), ErrorCode> {
        Ok(())
    }
}

fn config(seed: u8, refuse: Option<ErrorCode>) -> TcpConfig {
    TcpConfig {
        seed: [seed; 32],
        credential: Credential {
            kind: CredKind::DataPlaneCredential,
            bytes: vec![seed; CredKind::DataPlaneCredential.credential_len()],
        },
        check: Arc::new(StubCheck { refuse }),
        alpha_p99: Nanos(1_000_000),
        handshake_timeout: Duration::from_secs(5),
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("a runtime")
}

fn bind(rt: &tokio::runtime::Runtime, seed: u8, refuse: Option<ErrorCode>) -> TcpTransport {
    bind_with(rt, config(seed, refuse))
}

fn bind_with(rt: &tokio::runtime::Runtime, config: TcpConfig) -> TcpTransport {
    rt.block_on(TcpTransport::bind(
        config,
        SocketAddr::from(([127, 0, 0, 1], 0)),
    ))
    .expect("a listener on loopback")
}

fn remote(to: &TcpTransport) -> RemoteEndpoint {
    RemoteEndpoint {
        node: NodeId(to.public_key()),
        key: to.public_key(),
        addrs: vec![to.local_addr()],
        relay: None,
    }
}

/// The largest frame a receive slot holds: the suite's largest, rounded to the slot alignment.
fn slot_size() -> usize {
    MAX_FRAME.next_multiple_of(64)
}

fn spec(
    key: ChannelKey,
    transport: &TcpTransport,
    pool: &BufferPool,
    recv_slots: u16,
) -> ChannelSpec {
    ChannelSpec {
        key,
        pool: transport.register(pool).expect("registration"),
        slot_size: Bytes(if recv_slots == 0 {
            MAX_CONTROL_FRAME as u64
        } else {
            slot_size() as u64
        }),
        recv_slots,
        priority: ChannelPriority::Data,
        wire: WireDType::F32,
        retention: recv_slots.max(4),
    }
}

/// Channels from `from` to `to` over TCP, the suite's frames in `from`'s send pool.
struct Tcp<'a> {
    rt: &'a tokio::runtime::Runtime,
    from: &'a TcpTransport,
    to: &'a TcpTransport,
    send_pool: BufferPool,
    next_pool: u32,
}

impl<'a> Tcp<'a> {
    fn new(
        rt: &'a tokio::runtime::Runtime,
        from: &'a TcpTransport,
        to: &'a TcpTransport,
        id: u32,
    ) -> Self {
        let send_pool = BufferPool::heap(PoolId(id), SEND_SLOT, SEND_SLOTS).expect("a pool");
        from.register(&send_pool).expect("registration");
        Tcp {
            rt,
            from,
            to,
            send_pool,
            next_pool: id + 1,
        }
    }

    fn pool(&mut self, slot_size: usize, slots: u32) -> BufferPool {
        let pool = BufferPool::heap(PoolId(self.next_pool), slot_size, slots).expect("a pool");
        self.next_pool += 1;
        pool
    }
}

impl Fabric for Tcp<'_> {
    fn open(&mut self, key: ChannelKey, recv_slots: u16) -> Result<(Channel, Channel), Error> {
        let sender_pool = self.pool(MAX_CONTROL_FRAME, u32::from(CONTROL_RECEIVES));
        let receiver_pool = self.pool(
            slot_size().max(MAX_CONTROL_FRAME),
            u32::from(recv_slots) + u32::from(CONTROL_RECEIVES),
        );
        let sending = spec(key, self.from, &sender_pool, 0);
        let receiving = spec(key, self.to, &receiver_pool, recv_slots);
        let remote = remote(self.to);
        let (sender, receiver) = self.rt.block_on(async {
            tokio::join!(
                self.from.connect(&remote, sending),
                self.to.accept(receiving)
            )
        });
        Ok((sender?, receiver?))
    }

    fn send_pool(&self) -> &BufferPool {
        &self.send_pool
    }

    fn register(
        &mut self,
        pool: &BufferPool,
    ) -> Option<Result<syntropy_net::transport::PoolRegistration, Error>> {
        Some(self.from.register(pool))
    }

    fn pump(&mut self) {
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn key(from: u16, to: u16) -> ChannelKey {
    ChannelKey {
        instance: InstanceId(0x7cb),
        epoch: PlanEpoch { term: 2, seq: 9 },
        purpose: ChannelPurpose::Activations,
        from: StageIx(from),
        to: StageIx(to),
        lane: 0,
    }
}

/// Runs the suite from `from` to `to` and asserts every rule it can check passed. The fabric
/// cannot hold delivery on a real link, so `CompletedAfterAck` is the one rule skipped.
fn passes_the_suite(rt: &tokio::runtime::Runtime, from: &TcpTransport, to: &TcpTransport, id: u32) {
    let mut fabric = Tcp::new(rt, from, to, id);
    let report = conformance::run(&mut fabric);
    assert!(report.passed(), "{report}");
    assert_eq!(report.skipped, [Rule::CompletedAfterAck], "{report}");
}

/// Node B's UDP port with UDP "blocked": a socket bound there that never reads, so every datagram
/// sent to it is dropped, as a firewall's `DROP` rule drops it, and a QUIC handshake gets no
/// answer.
fn block_udp(at: SocketAddr) -> UdpSocket {
    UdpSocket::bind(at).expect("the TCP listener's port, free for UDP")
}

/// The criterion: with UDP blocked, QUIC dials between the two nodes fail, and a two-node ring
/// forms over TCP and passes the channel-conformance suite in both directions.
#[test]
fn with_udp_blocked_a_two_node_ring_forms_over_tcp_and_passes_the_suite() {
    let rt = runtime();
    let a = bind(&rt, 1, None);
    let b = bind(&rt, 2, None);
    a.allowed().allow(b.public_key());
    b.allowed().allow(a.public_key());
    let _blocked_a = block_udp(a.local_addr());
    let _blocked_b = block_udp(b.local_addr());

    // QUIC first, as ADR-0011's order has it; it fails in both directions.
    for (from_seed, to) in [(1u8, &b), (2, &a)] {
        let failed = rt.block_on(async {
            let quic = Endpoint::bind([from_seed; 32], endpoint::Config::default())
                .await
                .expect("a QUIC endpoint");
            let peer = NodeAddr::new(to.public_key())
                .expect("a valid key")
                .with_direct_addr(to.local_addr());
            let dial = tokio::time::timeout(
                Duration::from_secs(3),
                quic.connect(peer, b"syntropy/data/1"),
            )
            .await;
            quic.close().await;
            !matches!(dial, Ok(Ok(_)))
        });
        assert!(failed, "a QUIC dial reached a node whose UDP is blocked");
    }

    // The fallback: the ring's two hops over TCP.
    passes_the_suite(&rt, &a, &b, 100);
    passes_the_suite(&rt, &b, &a, 200);
    // Its slots are far under the kernel's buffer caps, so every socket got 2 × slot_size.
    assert_eq!(a.stats().buffers_autotuned, 0);
    assert_eq!(b.stats().buffers_autotuned, 0);
}

/// A connector that pins another key than the acceptor's is refused by its own TLS check.
#[test]
fn a_connector_refuses_an_acceptor_whose_key_it_did_not_dial_for() {
    let rt = runtime();
    let a = bind(&rt, 3, None);
    let b = bind(&rt, 4, None);
    b.allowed().allow(a.public_key());
    let mut fabric = Tcp::new(&rt, &a, &b, 300);
    let pool = fabric.pool(MAX_CONTROL_FRAME, u32::from(CONTROL_RECEIVES));
    let mut wrong = remote(&b);
    wrong.key = [0x99; 32];
    let error = rt
        .block_on(a.connect(&wrong, spec(key(0, 1), &a, &pool, 0)))
        .unwrap_err();
    assert_eq!(error.class, ErrorClass::Auth, "{error}");
    assert_eq!(error.code, Some(ErrorCode::CREDENTIAL_INVALID));
}

/// An acceptor refuses a connector whose key is not one it allowed, during TLS, so no channel
/// opens on either side.
#[test]
fn an_acceptor_refuses_a_key_it_did_not_allow() {
    let rt = runtime();
    let a = bind(&rt, 5, None);
    let b = bind(&rt, 6, None);
    let mut fabric = Tcp::new(&rt, &a, &b, 400);
    let sender_pool = fabric.pool(MAX_CONTROL_FRAME, u32::from(CONTROL_RECEIVES));
    let receiver_pool = fabric.pool(slot_size().max(MAX_CONTROL_FRAME), 8);
    let error = rt
        .block_on(a.connect(&remote(&b), spec(key(0, 1), &a, &sender_pool, 0)))
        .unwrap_err();
    assert!(
        matches!(error.class, ErrorClass::Auth | ErrorClass::Link),
        "{error}"
    );
    let accepted = rt.block_on(async {
        tokio::time::timeout(
            Duration::from_millis(300),
            b.accept(spec(key(0, 1), &b, &receiver_pool, 4)),
        )
        .await
    });
    assert!(
        accepted.is_err(),
        "the acceptor opened a channel for a key it never allowed"
    );

    // Allowed, the same peer connects.
    b.allowed().allow(a.public_key());
    let to_b = remote(&b);
    let (sender, receiver) = rt.block_on(async {
        tokio::join!(
            a.connect(&to_b, spec(key(0, 2), &a, &sender_pool, 0)),
            b.accept(spec(key(0, 2), &b, &receiver_pool, 4))
        )
    });
    assert_eq!(*sender.unwrap().key(), key(0, 2));
    assert_eq!(*receiver.unwrap().key(), key(0, 2));
}

/// A credential the acceptor's check refuses is answered with a refusing `HelloAck`, whose code
/// the connector reports.
#[test]
fn a_refused_credential_is_reported_with_its_code() {
    let rt = runtime();
    let a = bind(&rt, 7, None);
    let b = bind(&rt, 8, Some(ErrorCode::NOT_A_NEIGHBOUR));
    b.allowed().allow(a.public_key());
    let mut fabric = Tcp::new(&rt, &a, &b, 500);
    let pool = fabric.pool(MAX_CONTROL_FRAME, u32::from(CONTROL_RECEIVES));
    let error = rt
        .block_on(a.connect(&remote(&b), spec(key(0, 1), &a, &pool, 0)))
        .unwrap_err();
    assert_eq!(error.code, Some(ErrorCode::NOT_A_NEIGHBOUR), "{error}");
    assert_eq!(error.class, ErrorClass::Auth);
}

/// A connector tries each of the endpoint's addresses in turn: a dead first address does not stop
/// the channel opening on the second.
#[test]
fn a_connector_falls_through_to_the_next_address() {
    let rt = runtime();
    let a = bind(&rt, 9, None);
    let b = bind(&rt, 10, None);
    b.allowed().allow(a.public_key());
    let mut fabric = Tcp::new(&rt, &a, &b, 600);
    let sender_pool = fabric.pool(MAX_CONTROL_FRAME, u32::from(CONTROL_RECEIVES));
    let receiver_pool = fabric.pool(slot_size().max(MAX_CONTROL_FRAME), 8);
    // A port that was just free: nothing listens there now.
    let dead = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let mut via_dead = remote(&b);
    via_dead.addrs.insert(0, dead);
    let (sender, receiver) = rt.block_on(async {
        tokio::join!(
            a.connect(&via_dead, spec(key(1, 2), &a, &sender_pool, 0)),
            b.accept(spec(key(1, 2), &b, &receiver_pool, 4))
        )
    });
    assert!(sender.is_ok(), "{:?}", sender.err());
    assert!(receiver.is_ok());
}

/// A connection that arrived while its key was allowed, and waits for `accept`, is refused if
/// the key is forbidden before `accept` answers it: the acceptor checks the key and the
/// credential again.
#[test]
fn a_key_forbidden_while_its_connection_waits_is_refused() {
    let rt = runtime();
    let a = bind(&rt, 11, None);
    let b = bind(&rt, 12, None);
    b.allowed().allow(a.public_key());
    let mut fabric = Tcp::new(&rt, &a, &b, 700);
    let sender_pool = fabric.pool(MAX_CONTROL_FRAME, u32::from(CONTROL_RECEIVES));
    let receiver_pool = fabric.pool(slot_size().max(MAX_CONTROL_FRAME), 8);
    let to_b = remote(&b);
    let a_key = a.public_key();
    let (sender, receiver) = rt.block_on(async {
        tokio::join!(
            a.connect(&to_b, spec(key(2, 3), &a, &sender_pool, 0)),
            async {
                // The Hello has arrived by now and waits for this accept.
                tokio::time::sleep(Duration::from_millis(300)).await;
                b.allowed().forbid(&a_key);
                b.accept(spec(key(2, 3), &b, &receiver_pool, 4)).await
            }
        )
    });
    let refused = receiver.unwrap_err();
    assert_eq!(
        refused.code,
        Some(ErrorCode::CREDENTIAL_INVALID),
        "{refused}"
    );
    let error = sender.unwrap_err();
    assert_eq!(error.code, Some(ErrorCode::CREDENTIAL_INVALID), "{error}");
    assert_eq!(error.class, ErrorClass::Auth);
}

/// A connection whose `accept` does not come within the handshake timeout is dropped, so a peer
/// cannot leave connections parked indefinitely.
#[test]
fn a_connection_left_waiting_past_the_handshake_timeout_is_dropped() {
    let rt = runtime();
    let a = bind(&rt, 13, None);
    let b = bind_with(
        &rt,
        TcpConfig {
            handshake_timeout: Duration::from_millis(300),
            ..config(14, None)
        },
    );
    b.allowed().allow(a.public_key());
    let mut fabric = Tcp::new(&rt, &a, &b, 800);
    let sender_pool = fabric.pool(MAX_CONTROL_FRAME, u32::from(CONTROL_RECEIVES));
    let receiver_pool = fabric.pool(slot_size().max(MAX_CONTROL_FRAME), 8);
    let to_b = remote(&b);
    let (sender, receiver) = rt.block_on(async {
        tokio::join!(
            a.connect(&to_b, spec(key(4, 5), &a, &sender_pool, 0)),
            async {
                tokio::time::sleep(Duration::from_millis(800)).await;
                tokio::time::timeout(
                    Duration::from_millis(300),
                    b.accept(spec(key(4, 5), &b, &receiver_pool, 4)),
                )
                .await
            }
        )
    });
    assert!(
        receiver.is_err(),
        "an accept found a connection past its timeout"
    );
    let error = sender.unwrap_err();
    assert_eq!(error.class, ErrorClass::Link, "{error}");
}

/// One source address holds at most `MAX_HANDSHAKES_PER_ADDRESS` handshakes: a connection past
/// that is closed at once, while the listener keeps accepting.
#[test]
fn one_address_cannot_hold_every_handshake() {
    let rt = runtime();
    let b = bind(&rt, 15, None);
    let addr = b.local_addr();
    rt.block_on(async {
        let mut silent = Vec::new();
        for _ in 0..MAX_HANDSHAKES_PER_ADDRESS {
            silent.push(tokio::net::TcpStream::connect(addr).await.unwrap());
        }
        // The listener has taken them all into handshakes by now.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let mut extra = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut byte = [0u8; 1];
        let read = tokio::time::timeout(Duration::from_millis(500), extra.read(&mut byte)).await;
        assert!(
            matches!(read, Ok(Ok(0) | Err(_))),
            "a connection past the address's cap was kept: {read:?}"
        );
        drop(silent);
    });
}

/// A connection that never starts TLS is let go after `TLS_TIMEOUT`, well before the whole
/// handshake's timeout.
#[test]
fn a_connection_that_never_starts_tls_is_let_go() {
    let rt = runtime();
    let b = bind_with(
        &rt,
        TcpConfig {
            handshake_timeout: Duration::from_secs(30),
            ..config(16, None)
        },
    );
    let addr = b.local_addr();
    rt.block_on(async {
        let mut silent = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut byte = [0u8; 1];
        let read = tokio::time::timeout(TLS_TIMEOUT * 2, silent.read(&mut byte)).await;
        assert!(
            matches!(read, Ok(Ok(0) | Err(_))),
            "a silent connection was held past the TLS timeout: {read:?}"
        );
    });
}

/// Writes `line` to the process's stderr, past the test harness's capture, so CI logs show it
/// for a passing test too: which caps a runner had, and whether it skipped the check.
#[cfg(target_os = "linux")]
fn report(line: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "tcp autotuning test: {line}");
}

/// A channel whose receiver's slots are too large for the kernel's unprivileged buffer cap opens
/// on sockets left to autotuning: the acceptor's, sized for its own slot, and the connector's
/// send buffer, sized for the peer's slot from `HelloAck`. Both ends count it.
///
/// The slot is sized from this host's own caps, read from `/proc/sys/net/core`, which differ
/// between hosts (hosted runners do not all keep Linux's 212 992-byte default), so that twice it
/// is over both. Each end counts the socket before `connect` or `accept` returns, so the counts
/// are final once both have. Only a host where `/proc/sys` cannot be read, or whose caps would
/// need slots over 64 MiB, skips the check, and says so in the log; the unit test in
/// `transport::tcp` checks the backend reads the same caps.
#[cfg(target_os = "linux")]
#[test]
fn a_slot_over_the_kernel_cap_is_left_to_autotuning_and_counted() {
    let read = |name: &str| -> Option<usize> {
        std::fs::read_to_string(format!("/proc/sys/net/core/{name}"))
            .ok()?
            .trim()
            .parse()
            .ok()
    };
    let (Some(send_cap), Some(recv_cap)) = (read("wmem_max"), read("rmem_max")) else {
        report("SKIPPED: /proc/sys/net/core cannot be read on this host");
        return;
    };
    // Twice the slot is over the larger cap, so neither buffer can be asked for.
    let big = (send_cap.max(recv_cap) / 2 + 64).next_multiple_of(64);
    if big > 64 << 20 {
        report(&format!(
            "SKIPPED: caps of {send_cap} and {recv_cap} bytes need {big}-byte slots"
        ));
        return;
    }
    report(&format!(
        "buffer caps: wmem_max {send_cap}, rmem_max {recv_cap}; slot {big}"
    ));
    let rt = runtime();
    let a = bind(&rt, 17, None);
    let b = bind(&rt, 18, None);
    b.allowed().allow(a.public_key());
    let mut fabric = Tcp::new(&rt, &a, &b, 900);
    // Three channels, so a count that lagged its `connect` or `accept` would show.
    for lane in 0..3u16 {
        let sender_pool = fabric.pool(MAX_CONTROL_FRAME, u32::from(CONTROL_RECEIVES));
        let receiver_pool = fabric.pool(big, 2 + u32::from(CONTROL_RECEIVES));
        let sending = spec(key(6, 7 + lane), &a, &sender_pool, 0);
        let mut receiving = spec(key(6, 7 + lane), &b, &receiver_pool, 2);
        receiving.slot_size = Bytes(big as u64);
        let to_b = remote(&b);
        let (sender, receiver) =
            rt.block_on(async { tokio::join!(a.connect(&to_b, sending), b.accept(receiving)) });
        assert!(sender.is_ok() && receiver.is_ok());
        let opened = u64::from(lane) + 1;
        assert_eq!(
            a.stats().buffers_autotuned,
            opened,
            "the connector, channel {opened}"
        );
        assert_eq!(
            b.stats().buffers_autotuned,
            opened,
            "the acceptor, channel {opened}"
        );
    }
}

/// The listener requires the data ALPN: a client that completes TLS with an allowed key but no
/// ALPN is dropped before any `Hello` is read, and one that offers another ALPN does not get
/// through TLS. Neither opens a channel.
#[test]
fn a_connection_without_the_data_alpn_opens_no_channel() {
    use syntropy_net::transport::tcp::tls;
    let rt = runtime();
    let b = bind(&rt, 20, None);
    let client_seed = [21u8; 32];
    let client_key = bind(&rt, 21, None).public_key();
    b.allowed().allow(client_key);
    let addr = b.local_addr();
    for alpn in [Vec::new(), vec![b"syntropy/data/2".to_vec()]] {
        let mut config = tls::client_config(&client_seed, b.public_key()).unwrap();
        config.alpn_protocols = alpn.clone();
        let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
        let dropped = rt.block_on(async {
            let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
            let Ok(mut stream) = connector.connect(tls::server_name(), tcp).await else {
                return true;
            };
            let mut byte = [0u8; 1];
            matches!(
                tokio::time::timeout(Duration::from_secs(1), stream.read(&mut byte)).await,
                Ok(Ok(0) | Err(_))
            )
        });
        assert!(dropped, "a connection offering {alpn:?} was kept");
    }
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
    let payload = [0xA5u8; 64];
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

/// RFC-0031 §8's receive-path row for TCP: the frame lands in a slot of the receiver's
/// registered pool, one host copy from the TLS stream, and the consumer reads it there. The
/// sender sends from its own registered slot, which it keeps until the frame is acknowledged
/// (the conformance suite's `Retention` rule); it is free once the ticket is back.
#[test]
fn a_frame_lands_in_a_registered_receive_slot_after_one_host_copy() {
    let rt = runtime();
    let a = bind(&rt, 1, None);
    let b = bind(&rt, 2, None);
    a.allowed().allow(b.public_key());
    b.allowed().allow(a.public_key());
    let mut fabric = Tcp::new(&rt, &a, &b, 300);
    let (mut sender, mut receiver) = fabric.open(key(0, 1), 4).expect("open");
    // `Tcp::open` makes the sender's pool, then the receiver's.
    let receive_pool = PoolId(fabric.next_pool - 1);
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
    assert_eq!(received.payload.bytes(), &[0xA5; 64]);
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
