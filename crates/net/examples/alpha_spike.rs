// SPDX-License-Identifier: MIT OR Apache-2.0

//! The measurement harness of the DIST-E7-T16 spike: QUIC against TCP α, the cost of a credit
//! per frame, and QUIC stream against datagram delivery (RFC-0031 §3, §4.3, §7.1, open questions
//! 2, 3 and 5; OQ-NET-29). A spike harness, not a product path: it runs by hand between two hosts.
//!
//! ```text
//! host A:  cargo run --release -p syntropy-net --example alpha_spike -- serve [--udp-port 47600] [--tcp-port 47601]
//! host B:  cargo run --release -p syntropy-net --example alpha_spike -- probe --server <ip> --quic <key>@<port> --echo <key>@<port> [--samples 1000] [--period-ms 10]
//! one host: cargo run --release -p syntropy-net --example alpha_spike -- local
//! ```
//!
//! `serve` prints the two `--quic` and `--echo` values `probe` takes. What `probe` measures, each
//! as a series of samples with its p50, p90, p99 and maximum in µs:
//! - **`channel.<quic|tcp>.<size>`**: α as the channel core takes it (RFC-0031 §7.1): half the
//!   round trip from a data frame's send to the credit that echoes it, less the receiver's echo
//!   delay. One `Activations` frame of `size` bytes of payload is sent every `--period-ms`, and the
//!   next waits for the previous frame's sample, so the channel never queues.
//! - **`udp.<size>`**, and **`udp.<size>.credit`**: half the round trip of a kernel UDP echo,
//!   the floor under both transports. With `.credit` the echo server sends a 64-byte datagram, as
//!   a credit is, before each echo: the extra packet a credit per frame puts on the air (open
//!   question 3), without the channel's own work.
//! - **`quic.stream.<size>`** and **`quic.datagram.<size>`**: half the round trip of an echo on
//!   one QUIC connection, over one bidirectional stream and over datagrams (OQ-NET-29). Datagram
//!   sizes stop below the path's datagram limit.
//!
//! The rate is low by design, about 1.6 MB/s at 16 KiB every 10 ms, so a run on a shared LAN or
//! Wi-Fi floods nothing. Every figure it prints is a measurement of the hosts and link it ran on,
//! and only that.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::Arc;
use std::time::{Duration, Instant};

use iroh::endpoint::{RelayMode, presets};
use iroh::{EndpointAddr, PublicKey, SecretKey};
use syntropy_core::plan::WireDType;
use syntropy_core::{Bytes, InstanceId, Nanos, NodeId, PlanEpoch, StageIx};
use syntropy_net::endpoint::Config as EndpointConfig;
use syntropy_net::frame::{Flags, FrameType, HEADER_LEN};
use syntropy_net::transport::core::{CONTROL_RECEIVES, MAX_CONTROL_FRAME};
use syntropy_net::transport::domain::NetDomain;
use syntropy_net::transport::quic::{QuicConfig, QuicSettings, QuicTransport};
use syntropy_net::transport::stream::UncheckedAuth;
use syntropy_net::transport::tcp::{TcpConfig, TcpTransport};
use syntropy_net::transport::{
    BufferPool, Channel, ChannelKey, ChannelPriority, ChannelPurpose, ChannelSpec, ExtBuf,
    OutFrame, PoolId, RemoteEndpoint, SendError, Transport,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// The payload sizes of the channel runs: a token-sized frame and a decode activation's.
const CHANNEL_SIZES: [u32; 2] = [64, 16 * 1024];
/// The sizes of the UDP and QUIC echo runs; 1 KiB stays under the QUIC datagram limit.
const ECHO_SIZES: [usize; 2] = [64, 1024];
/// A receive slot: the largest frame, rounded up to 64 bytes.
const SLOT: usize = (16 * 1024 + HEADER_LEN + MAX_CONTROL_FRAME).next_multiple_of(64);
const RECV_SLOTS: u16 = 8;
/// The application protocol of the echo endpoint.
const ECHO_ALPN: &[u8] = b"syntropy-spike/echo/1";
/// The seeds of the four endpoints: test keys, never a deployed identity.
const SERVER_QUIC_SEED: [u8; 32] = [0x51; 32];
const SERVER_TCP_SEED: [u8; 32] = [0x52; 32];
const SERVER_ECHO_SEED: [u8; 32] = [0x53; 32];
const PROBE_SEED: [u8; 32] = [0x61; 32];
/// How long one sample may take before it counts as lost.
const SAMPLE_TIMEOUT: Duration = Duration::from_secs(1);

type Error = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|at| args.get(at + 1))
            .cloned()
    };
    let number = |name: &str, default: u64| -> Result<u64, Error> {
        flag(name).map_or(Ok(default), |value| Ok(value.parse()?))
    };
    let options = Options {
        udp_port: u16::try_from(number("--udp-port", 47600)?)?,
        tcp_port: u16::try_from(number("--tcp-port", 47601)?)?,
        samples: usize::try_from(number("--samples", 1000)?)?,
        period: Duration::from_millis(number("--period-ms", 10)?),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    match args.first().map(String::as_str) {
        Some("serve") => serve(&runtime, &options, IpAddr::V4(Ipv4Addr::UNSPECIFIED), None),
        Some("probe") => {
            let server: IpAddr = flag("--server").ok_or("--server <ip>")?.parse()?;
            let quic = parse_peer(&flag("--quic").ok_or("--quic <key>@<port>")?)?;
            let echo = parse_peer(&flag("--echo").ok_or("--echo <key>@<port>")?)?;
            probe(&runtime, &options, server, quic, echo)
        }
        Some("local") => {
            let (tx, rx) = std::sync::mpsc::channel();
            let server_options = options.clone();
            std::thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .expect("a runtime");
                let _ = serve(
                    &runtime,
                    &server_options,
                    IpAddr::V4(Ipv4Addr::LOCALHOST),
                    Some(tx),
                );
            });
            let (quic, echo) = rx.recv()?;
            probe(
                &runtime,
                &options,
                IpAddr::V4(Ipv4Addr::LOCALHOST),
                quic,
                echo,
            )
        }
        _ => Err("usage: alpha_spike serve | probe --server <ip> --quic <key>@<port> --echo <key>@<port> | local".into()),
    }
}

#[derive(Clone, Debug)]
struct Options {
    udp_port: u16,
    tcp_port: u16,
    samples: usize,
    period: Duration,
}

/// An endpoint key and a UDP port, as `serve` prints them.
type Peer = ([u8; 32], u16);

fn parse_peer(text: &str) -> Result<Peer, Error> {
    let (key, port) = text.split_once('@').ok_or("<key>@<port>")?;
    if key.len() != 64 {
        return Err("a key is 64 hex digits".into());
    }
    let mut bytes = [0u8; 32];
    for (at, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&key[2 * at..2 * at + 2], 16)?;
    }
    Ok((bytes, port.parse()?))
}

fn hex(key: &[u8; 32]) -> String {
    key.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn public_key(seed: &[u8; 32]) -> [u8; 32] {
    ed25519_dalek::SigningKey::from_bytes(seed)
        .verifying_key()
        .to_bytes()
}

fn key(purpose: ChannelPurpose, lane: u8) -> ChannelKey {
    ChannelKey {
        instance: InstanceId(0x7116),
        epoch: PlanEpoch { term: 1, seq: 1 },
        purpose,
        from: StageIx(0),
        to: StageIx(1),
        lane,
    }
}

fn spec(
    key: ChannelKey,
    pool: syntropy_net::transport::PoolRegistration,
    recv_slots: u16,
) -> ChannelSpec {
    ChannelSpec {
        key,
        pool,
        slot_size: Bytes(SLOT as u64),
        recv_slots,
        priority: ChannelPriority::Data,
        wire: WireDType::F32,
        retention: recv_slots.max(4),
    }
}

/// The port of the first IPv4 socket among `addrs`.
fn port_of(addrs: &[SocketAddr]) -> Result<u16, Error> {
    addrs
        .iter()
        .find(|addr| addr.is_ipv4())
        .map(SocketAddr::port)
        .ok_or_else(|| "no IPv4 socket".into())
}

// ---- serve ----

fn serve(
    runtime: &tokio::runtime::Runtime,
    options: &Options,
    ip: IpAddr,
    ready: Option<std::sync::mpsc::Sender<(Peer, Peer)>>,
) -> Result<(), Error> {
    let domain = NetDomain::start("alpha-spike-serve")?;
    let quic = runtime.block_on(QuicTransport::bind(
        domain.handle(),
        SERVER_QUIC_SEED,
        QuicConfig {
            endpoint: EndpointConfig::default(),
            settings: QuicSettings::default(),
            auth: Arc::new(UncheckedAuth),
            capabilities: 0,
        },
    ))?;
    let tcp = Arc::new(runtime.block_on(TcpTransport::bind(
        domain.handle(),
        tcp_config(SERVER_TCP_SEED),
        SocketAddr::new(ip, options.tcp_port),
    ))?);
    tcp.allowed().allow(public_key(&PROBE_SEED));
    let echo = runtime.block_on(
        iroh::Endpoint::builder(presets::Minimal)
            .relay_mode(RelayMode::Disabled)
            .secret_key(SecretKey::from_bytes(&SERVER_ECHO_SEED))
            .alpns(vec![ECHO_ALPN.to_vec()])
            .bind(),
    )?;
    let udp = UdpSocket::bind(SocketAddr::new(ip, options.udp_port))?;
    let quic_peer = (quic.node_id(), port_of(&quic.bound_addrs())?);
    let echo_peer = (*echo.id().as_bytes(), port_of(&echo.bound_sockets())?);
    println!(
        "serving: --quic {}@{} --echo {}@{} (udp {}, tcp {})",
        hex(&quic_peer.0),
        quic_peer.1,
        hex(&echo_peer.0),
        echo_peer.1,
        options.udp_port,
        options.tcp_port
    );
    // Accept every channel the probe opens, and drain each on a thread of its own.
    for (lane, _) in CHANNEL_SIZES.iter().enumerate() {
        let lane = lane as u8;
        let pool = BufferPool::heap(
            PoolId(100 + u32::from(lane)),
            SLOT,
            u32::from(RECV_SLOTS + CONTROL_RECEIVES),
        )?;
        let registration = quic.register(&pool)?;
        let channel = spec(
            key(ChannelPurpose::Activations, lane),
            registration,
            RECV_SLOTS,
        );
        let quic = quic.clone();
        runtime.spawn(async move {
            if let Ok(channel) = quic.accept(channel).await {
                drain(channel);
            }
        });
        let pool = BufferPool::heap(
            PoolId(200 + u32::from(lane)),
            SLOT,
            u32::from(RECV_SLOTS + CONTROL_RECEIVES),
        )?;
        let registration = tcp.register(&pool)?;
        let channel = spec(
            key(ChannelPurpose::Activations, lane),
            registration,
            RECV_SLOTS,
        );
        let tcp = Arc::clone(&tcp);
        runtime.spawn(async move {
            if let Ok(channel) = tcp.accept(channel).await {
                drain(channel);
            }
        });
    }
    runtime.spawn(echo_server(echo));
    std::thread::spawn(move || udp_echo(udp));
    if let Some(ready) = ready {
        let _ = ready.send((quic_peer, echo_peer));
    }
    // Serve until killed; the transports run in the domain and the runtime.
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}

fn tcp_config(seed: [u8; 32]) -> TcpConfig {
    TcpConfig {
        seed,
        auth: Arc::new(UncheckedAuth),
        capabilities: 0,
        alpha_p99: Nanos(1_000_000),
        handshake_timeout: Duration::from_secs(5),
    }
}

/// Drops every frame the channel delivers, so its slots, and the sender's credit, come back.
fn drain(mut channel: Channel) {
    std::thread::spawn(move || {
        loop {
            if channel.try_recv().is_none() {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    });
}

/// Echoes each datagram; one whose first byte is 1 is preceded by a 64-byte datagram, as a
/// credit would be.
fn udp_echo(udp: UdpSocket) {
    let mut buf = vec![0u8; 65536];
    // The credit stand-in starts with 2, so the probe never takes it for the echo.
    let credit = [2u8; 64];
    while let Ok((len, from)) = udp.recv_from(&mut buf) {
        if buf[0] == 1 {
            let _ = udp.send_to(&credit, from);
        }
        let _ = udp.send_to(&buf[..len], from);
    }
}

/// Echoes datagrams, and each bidirectional stream's length-prefixed messages, on every
/// connection to the echo endpoint.
async fn echo_server(endpoint: iroh::Endpoint) {
    while let Some(incoming) = endpoint.accept().await {
        tokio::spawn(async move {
            let Ok(connection) = incoming.await else {
                return;
            };
            let datagrams = connection.clone();
            tokio::spawn(async move {
                while let Ok(datagram) = datagrams.read_datagram().await {
                    if datagrams.send_datagram(datagram).is_err() {
                        return;
                    }
                }
            });
            while let Ok((mut send, mut recv)) = connection.accept_bi().await {
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    loop {
                        let Ok(len) = recv.read_u32().await else {
                            return;
                        };
                        let len = len as usize;
                        if len > buf.len()
                            || AsyncReadExt::read_exact(&mut recv, &mut buf[..len])
                                .await
                                .is_err()
                        {
                            return;
                        }
                        if send.write_u32(len as u32).await.is_err()
                            || send.write_all(&buf[..len]).await.is_err()
                        {
                            return;
                        }
                    }
                });
            }
        });
    }
}

// ---- probe ----

fn probe(
    runtime: &tokio::runtime::Runtime,
    options: &Options,
    server: IpAddr,
    quic_peer: Peer,
    echo_peer: Peer,
) -> Result<(), Error> {
    println!(
        "probe: {} samples a series, one every {:?}; figures are µs",
        options.samples, options.period
    );
    println!("series n lost p50 p90 p99 max");
    // Kernel UDP: the floor, then with a credit-sized datagram before each echo.
    let udp = UdpSocket::bind(SocketAddr::new(
        if server.is_ipv4() {
            IpAddr::V4(Ipv4Addr::UNSPECIFIED)
        } else {
            IpAddr::V6(std::net::Ipv6Addr::UNSPECIFIED)
        },
        0,
    ))?;
    udp.set_read_timeout(Some(SAMPLE_TIMEOUT))?;
    let to = SocketAddr::new(server, options.udp_port);
    for size in ECHO_SIZES {
        for credit in [false, true] {
            let series = udp_series(&udp, to, size, credit, options);
            let label = format!("udp.{size}{}", if credit { ".credit" } else { "" });
            report(&label, &series);
        }
    }
    // The transports' channels: α from credit echoes.
    let domain = NetDomain::start("alpha-spike-probe")?;
    let quic = runtime.block_on(QuicTransport::bind(
        domain.handle(),
        PROBE_SEED,
        QuicConfig {
            endpoint: EndpointConfig::default(),
            settings: QuicSettings::default(),
            auth: Arc::new(UncheckedAuth),
            capabilities: 0,
        },
    ))?;
    let tcp = runtime.block_on(TcpTransport::bind(
        domain.handle(),
        tcp_config(PROBE_SEED),
        SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
    ))?;
    let send_pool = BufferPool::heap(PoolId(1), SLOT, 64)?;
    quic.register(&send_pool)?;
    tcp.register(&send_pool)?;
    let quic_remote = RemoteEndpoint {
        node: NodeId(quic_peer.0),
        key: quic_peer.0,
        addrs: vec![SocketAddr::new(server, quic_peer.1)],
        relay: None,
    };
    let tcp_remote = RemoteEndpoint {
        node: NodeId(public_key(&SERVER_TCP_SEED)),
        key: public_key(&SERVER_TCP_SEED),
        addrs: vec![SocketAddr::new(server, options.tcp_port)],
        relay: None,
    };
    for (lane, size) in CHANNEL_SIZES.into_iter().enumerate() {
        for (name, transport) in [("quic", 0), ("tcp", 1)] {
            let control = BufferPool::heap(
                PoolId(10 + 2 * lane as u32 + transport),
                SLOT,
                u32::from(CONTROL_RECEIVES),
            )?;
            let channel_key = key(ChannelPurpose::Activations, lane as u8);
            let channel = if transport == 0 {
                let registration = quic.register(&control)?;
                runtime.block_on(quic.connect(&quic_remote, spec(channel_key, registration, 0)))?
            } else {
                let registration = tcp.register(&control)?;
                runtime.block_on(tcp.connect(&tcp_remote, spec(channel_key, registration, 0)))?
            };
            let series = channel_series(channel, &send_pool, size, options);
            report(&format!("channel.{name}.{size}"), &series);
        }
    }
    // QUIC: one stream against datagrams, on one connection.
    runtime.block_on(async {
        let endpoint = iroh::Endpoint::builder(presets::Minimal)
            .relay_mode(RelayMode::Disabled)
            .secret_key(SecretKey::from_bytes(&PROBE_SEED))
            .bind()
            .await?;
        let addr = EndpointAddr::new(PublicKey::from_bytes(&echo_peer.0)?)
            .with_ip_addr(SocketAddr::new(server, echo_peer.1));
        let connection = endpoint.connect(addr, ECHO_ALPN).await?;
        let (mut send, mut recv) = connection.open_bi().await?;
        for size in ECHO_SIZES {
            let payload = vec![0x5a; size];
            let mut back = vec![0u8; size];
            let mut series = Series::default();
            for _ in 0..options.samples {
                let started = Instant::now();
                send.write_u32(size as u32).await?;
                send.write_all(&payload).await?;
                let echoed = tokio::time::timeout(SAMPLE_TIMEOUT, async {
                    let len = recv.read_u32().await? as usize;
                    AsyncReadExt::read_exact(&mut recv, &mut back[..len]).await
                })
                .await;
                series.take(started, echoed.is_ok_and(|read| read.is_ok()));
                tokio::time::sleep(options.period).await;
            }
            report(&format!("quic.stream.{size}"), &series);
            let mut series = Series::default();
            let limit = connection.max_datagram_size().unwrap_or(0);
            if size > limit {
                println!(
                    "quic.datagram.{size} skipped: over the path's datagram limit, {limit} bytes"
                );
                continue;
            }
            for _ in 0..options.samples {
                let started = Instant::now();
                connection.send_datagram(payload.clone().into())?;
                let echoed = tokio::time::timeout(SAMPLE_TIMEOUT, connection.read_datagram()).await;
                series.take(started, echoed.is_ok_and(|read| read.is_ok()));
                tokio::time::sleep(options.period).await;
            }
            report(&format!("quic.datagram.{size}"), &series);
        }
        connection.close(0u32.into(), b"done");
        endpoint.close().await;
        Ok::<(), Error>(())
    })?;
    Ok(())
}

/// One-way samples, in ns, and how many timed out.
#[derive(Default)]
struct Series {
    samples: Vec<u64>,
    lost: usize,
    /// The first send error other than `WouldBlock`, if any: a channel that faulted says so.
    error: Option<SendError>,
    /// A channel's data frames sent again, at the end of its series.
    retransmits: Option<u64>,
}

impl Series {
    /// Half the round trip since `started`, or a loss.
    fn take(&mut self, started: Instant, answered: bool) {
        if answered {
            self.samples.push(started.elapsed().as_nanos() as u64 / 2);
        } else {
            self.lost += 1;
        }
    }
}

fn udp_series(
    udp: &UdpSocket,
    to: SocketAddr,
    size: usize,
    credit: bool,
    options: &Options,
) -> Series {
    let mut payload = vec![0x5a; size];
    payload[0] = u8::from(credit);
    let mut buf = vec![0u8; 65536];
    let mut series = Series::default();
    for _ in 0..options.samples {
        let started = Instant::now();
        let mut answered = udp.send_to(&payload, to).is_ok();
        // The echo is the datagram of the payload's size; a credit before it is skipped.
        while answered {
            match udp.recv_from(&mut buf) {
                Ok((len, _)) if len == size && buf[0] != 2 => break,
                Ok(_) => {}
                Err(_) => answered = false,
            }
        }
        series.take(started, answered);
        std::thread::sleep(options.period);
    }
    series
}

/// Sends one frame of `size` bytes every period, each after the last one's α sample, and
/// collects the samples the channel core takes from the credit echoes.
fn channel_series(mut channel: Channel, pool: &BufferPool, size: u32, options: &Options) -> Series {
    let mut series = Series::default();
    let mut seen = channel.stats().alpha_samples;
    for step in 0..options.samples as u64 {
        let deadline = Instant::now() + SAMPLE_TIMEOUT;
        let sent = loop {
            let Some(mut slot) = (0..pool.slots()).find_map(|slot| pool.claim(slot)) else {
                if Instant::now() > deadline {
                    break false;
                }
                std::thread::sleep(Duration::from_micros(100));
                continue;
            };
            slot.bytes_mut()[..size as usize].fill(step as u8);
            let reader = slot.freeze();
            let frame = OutFrame {
                ftype: FrameType::Activation,
                step,
                flags: Flags::default(),
                ext: ExtBuf::EMPTY,
                payload: reader.payload(0, size).expect("in the slot"),
            };
            match channel.try_send(frame) {
                Ok(_) => break true,
                Err(SendError::WouldBlock) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_micros(100));
                }
                Err(error) => {
                    series.error.get_or_insert(error);
                    break false;
                }
            }
        };
        let mut answered = false;
        while sent && Instant::now() < deadline {
            let stats = channel.stats();
            if stats.alpha_samples != seen {
                seen = stats.alpha_samples;
                if let Some(alpha) = stats.alpha_last {
                    series.samples.push(alpha.0);
                    answered = true;
                }
                break;
            }
            std::thread::yield_now();
        }
        if !answered {
            series.lost += 1;
        }
        while channel.completed().is_some() {}
        std::thread::sleep(options.period);
    }
    series.retransmits = Some(channel.stats().retransmits);
    series
}

fn report(label: &str, series: &Series) {
    let mut sorted = series.samples.clone();
    sorted.sort_unstable();
    let rank = |q: usize| -> String {
        if sorted.is_empty() {
            return "-".into();
        }
        let at = (sorted.len() * q).div_ceil(100).max(1) - 1;
        format!("{:.1}", sorted[at] as f64 / 1000.0)
    };
    let max = sorted
        .last()
        .map_or("-".into(), |max| format!("{:.1}", *max as f64 / 1000.0));
    println!(
        "{label} {} {} {} {} {} {max}",
        sorted.len(),
        series.lost,
        rank(50),
        rank(90),
        rank(99)
    );
    if let Some(error) = series.error {
        println!("  {label}: the first send error was \"{error}\"");
    }
    if let Some(retransmits) = series.retransmits.filter(|count| *count > 0) {
        println!("  {label}: {retransmits} data frames sent again");
    }
}
