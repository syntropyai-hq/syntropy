// SPDX-License-Identifier: MIT OR Apache-2.0

//! The TCP fallback backend (RFC-0031 §5.3; ADR-0011 item 2; DIST-E7-T05): the last backend of
//! ADR-0011's order, for an edge whose QUIC dials fail while TCP connects, because UDP is blocked.
//!
//! - **Identity.** TLS 1.3 authenticates both ends with the Ed25519 key form iroh uses, a raw
//!   public key ([`tls`]). The connector accepts only the key in its [`RemoteEndpoint`]; the
//!   acceptor only the keys its owner allowed ([`TcpTransport::allowed`]), the plan's neighbours.
//! - **One TCP connection per channel**, so no two channels share a byte stream and neither waits
//!   on the other's lost segments.
//! - **Socket options.** `TCP_NODELAY` is on. `SO_SNDBUF` and `SO_RCVBUF` are at least twice the
//!   channel's `slot_size`, set before the connector's `connect`. On Linux, `TCP_QUICKACK` is
//!   re-armed after every read, since the kernel clears it.
//! - **Framing** is RFC-0031 §6's, as on a QUIC stream: frames back to back on the stream, each
//!   read by its base header's lengths. Credits, acknowledgements and retransmission are the
//!   [`ChannelCore`]'s (§7), as on every backend.
//!
//! **Opening a channel.** The connector dials each of the endpoint's addresses in turn, runs the
//! TLS handshake, and sends `Hello`; the acceptor's listener runs the TLS handshake, reads the
//! `Hello`, checks its credential, and holds the connection until [`Transport::accept`] asks for
//! its channel key, then answers `HelloAck` (§6.9). The credential is the owner's to mint and to
//! check ([`Credential`], [`CredentialCheck`]): runner credentials and their verification are
//! DIST-E7-T04's, so this backend carries the bytes and calls the check.
//!
//! **The driver.** Each channel has two tasks on the `net` domain's tokio runtime: one reads
//! frames off the stream into the core's receive slots, one writes what the core has to send and
//! runs its timers every [`TICK`]. The [`Channel`]'s methods and the two tasks share the core
//! under a mutex, held only while the core runs, never across I/O.
//!
//! The mutex is never on a dispatcher's path. A dispatcher cannot call a [`Channel`]: `exec` and
//! `sched` never depend on `net` (RFC-0001 §4.2 rule 3, which `cargo xtask check-deps` enforces,
//! `COMPUTE_MUST_NOT_REACH` in `xtask/src/check_deps/policy.rs`), so they hand frames to the
//! runner over SPSC rings, as the [module](super) docs say. The runner's code that drains those
//! rings calls the channel's methods on the `net` domain, which is one single-threaded tokio
//! runtime (RFC-0031 §4.4). There, the methods and the channel's tasks take turns on one
//! thread, so the lock is never contended; it is there for a runtime with more threads, such as
//! the tests'.

pub mod tls;

use std::io;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use syntropy_core::resource::TransportKind;
use syntropy_core::{Bytes, ErrorClass, MonoTime, Nanos};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::{TcpListener, TcpSocket, TcpStream};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, oneshot};
use tokio::task::JoinHandle;
use tokio_rustls::{TlsAcceptor, TlsConnector, TlsStream};

use self::tls::AllowedKeys;
use super::{
    BufferPool, Channel, ChannelBackend, ChannelCore, ChannelKey, ChannelPurpose, ChannelSpec,
    ChannelStats, CloseReason, CoreConfig, CreditCadence, InFrame, OutFrame, PoolRegistration,
    RemoteEndpoint, SendConfig, SendError, SendTicket, Transport, TransportCaps,
};
use crate::error::{Error, ErrorCode};
use crate::frame::{
    Control, ControlSpec, CredKind, Flags, Frame, HEADER_LEN, Hello, HelloAck, WIRE_MINOR, code,
};
use crate::transport::core::{ACK_TIMEOUTS_TO_FAULT, CRC_STORM_COUNT, MAX_CONTROL_FRAME, Transmit};

/// The protocol major every `Hello` carries (RFC-0031 §10).
pub const PROTO_MAJOR: u16 = 1;

/// The largest frame on this backend: `net.max_frame`'s default, 64 MiB (RFC-0031 §2).
pub const MAX_FRAME: u64 = 64 << 20;

/// How often a channel's writer runs the core's timers when nothing wakes it sooner \[target\]:
/// the ack timeout's floor is 200 µs and tokio's timer resolution 1 ms.
pub const TICK: Duration = Duration::from_millis(1);

/// The bytes a writer gathers from the core before it writes them out \[target\].
const FLUSH_AT: usize = 256 << 10;

/// Connections the listener takes through TLS and `Hello` at once \[target\]. A connection that
/// arrives while every one is held is closed at once, and the listener keeps accepting.
pub const MAX_HANDSHAKES: usize = 64;

/// Of those, the most from one source address \[target\], so one host's slow or silent
/// connections cannot hold every handshake.
pub const MAX_HANDSHAKES_PER_ADDRESS: usize = 8;

/// The longest the TLS handshake may take, before the peer has proved its key \[target\]:
/// shorter than the whole handshake's timeout, so an unauthenticated connection is let go
/// soonest. TLS 1.3 takes one round trip and a bit.
pub const TLS_TIMEOUT: Duration = Duration::from_secs(2);

/// How long the listener waits after an `accept` fails, such as when the process is out of file
/// descriptors, before it tries again \[target\].
const ACCEPT_BACKOFF: Duration = Duration::from_millis(10);

/// A credential this endpoint presents in its `Hello` or `HelloAck` (RFC-0031 §6.9): a runner's
/// data-plane credential, or a daemon's admission certificate. Its bytes are `kind`'s length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Credential {
    pub kind: CredKind,
    pub bytes: Vec<u8>,
}

/// How an endpoint checks a peer's credential once TLS has proved the peer holds `peer`'s key:
/// that the credential is signed for that key, names this channel's stage, instance and epoch,
/// is not expired and not revoked (RFC-0034 §6.3; DIST-E7-T04 builds the checks). A refusal is
/// the RFC-0031 §12 code the acceptor sends in `HelloAck`, or the connector reports.
pub trait CredentialCheck: Send + Sync {
    /// The acceptor's check of the connector's `Hello`.
    fn hello(
        &self,
        hello: &Hello,
        kind: CredKind,
        credential: &[u8],
        peer: &[u8; 32],
    ) -> Result<(), ErrorCode>;

    /// The connector's check of the acceptor's `HelloAck`.
    fn hello_ack(
        &self,
        ack: &HelloAck,
        kind: CredKind,
        credential: &[u8],
        peer: &[u8; 32],
    ) -> Result<(), ErrorCode>;
}

/// A [`TcpTransport`]'s settings.
#[derive(Clone)]
pub struct TcpConfig {
    /// This endpoint's Ed25519 seed: the runner key, or the node key for a daemon.
    pub seed: [u8; 32],
    /// The credential this endpoint presents.
    pub credential: Credential,
    /// How it checks the peer's.
    pub check: Arc<dyn CredentialCheck>,
    /// A link's α p99 before any sample, from the plan's link profile.
    pub alpha_p99: Nanos,
    /// The longest a dial, a TLS handshake or a `Hello` exchange may take.
    pub handshake_timeout: Duration,
}

impl std::fmt::Debug for TcpConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpConfig")
            .field("credential", &self.credential.kind)
            .field("alpha_p99", &self.alpha_p99)
            .field("handshake_timeout", &self.handshake_timeout)
            .finish_non_exhaustive()
    }
}

/// The monotonic time a transport's cores run on, from the transport's start, never 0.
#[derive(Clone, Copy, Debug)]
struct Clock(Instant);

impl Clock {
    fn now(self) -> MonoTime {
        let nanos = u64::try_from(self.0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        MonoTime(nanos.saturating_add(1))
    }
}

/// A connection whose `Hello` has arrived, its credential checked, held for the `accept` of its
/// key, at most the handshake timeout. `accept` checks the key and the credential again before
/// it answers, since either may have stopped passing while the connection waited.
struct Arrived {
    hello: Hello,
    credential: Vec<u8>,
    peer: [u8; 32],
    stream: TlsStream<TcpStream>,
    at: Instant,
}

/// Drops the arrived connections that waited longer than `timeout`, closing them.
fn expire(arrived: &mut Vec<(ChannelKey, Arrived)>, timeout: Duration) {
    arrived.retain(|(_, waiting)| waiting.at.elapsed() < timeout);
}

/// Arrived connections and waiting accepts, matched by channel key.
#[derive(Default)]
struct Rendezvous {
    arrived: Vec<(ChannelKey, Arrived)>,
    waiting: Vec<(ChannelKey, oneshot::Sender<Arrived>)>,
}

struct Inner {
    config: TcpConfig,
    /// This endpoint's public key.
    key: [u8; 32],
    allowed: AllowedKeys,
    pools: Mutex<Vec<BufferPool>>,
    rendezvous: Mutex<Rendezvous>,
    /// One permit per handshake in progress, [`MAX_HANDSHAKES`] in all.
    handshakes: Arc<Semaphore>,
    /// Handshakes in progress by source address.
    per_address: Mutex<Vec<(IpAddr, usize)>>,
    local_addr: SocketAddr,
    clock: Clock,
}

/// Locks `mutex`, past a panic of another holder: the state it guards stays consistent between
/// calls, so a poisoned lock is still usable.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The TCP fallback transport: one listener for the channels this endpoint accepts, and a dial
/// per channel it connects.
pub struct TcpTransport {
    inner: Arc<Inner>,
    listener: JoinHandle<()>,
}

impl std::fmt::Debug for TcpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpTransport")
            .field("local_addr", &self.inner.local_addr)
            .finish_non_exhaustive()
    }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        self.listener.abort();
    }
}

fn link(code: ErrorCode, message: impl Into<String>) -> Error {
    Error::new(code, message)
}

fn tls_error(error: rustls::Error) -> Error {
    Error::uncoded(ErrorClass::Internal, format!("TLS configuration: {error}"))
}

impl TcpTransport {
    /// Listens on `addr` for the channels this endpoint accepts. Call it inside the `net`
    /// domain's tokio runtime, which runs the listener and every channel's tasks.
    pub async fn bind(config: TcpConfig, addr: SocketAddr) -> Result<TcpTransport, Error> {
        if config.credential.bytes.len() != config.credential.kind.credential_len() {
            return Err(Error::uncoded(
                ErrorClass::InvalidRequest,
                format!(
                    "a {}-byte credential of kind {:?}",
                    config.credential.bytes.len(),
                    config.credential.kind
                ),
            ));
        }
        let listener = TcpListener::bind(addr).await.map_err(|error| {
            Error::uncoded(ErrorClass::Link, format!("TCP listen on {addr}: {error}"))
        })?;
        let local_addr = listener
            .local_addr()
            .map_err(|error| Error::uncoded(ErrorClass::Link, error.to_string()))?;
        let key = ed25519_dalek::SigningKey::from_bytes(&config.seed)
            .verifying_key()
            .to_bytes();
        let allowed = AllowedKeys::default();
        let acceptor = TlsAcceptor::from(Arc::new(
            tls::server_config(&config.seed, allowed.clone()).map_err(tls_error)?,
        ));
        let inner = Arc::new(Inner {
            config,
            key,
            allowed,
            pools: Mutex::new(Vec::new()),
            rendezvous: Mutex::new(Rendezvous::default()),
            handshakes: Arc::new(Semaphore::new(MAX_HANDSHAKES)),
            per_address: Mutex::new(Vec::new()),
            local_addr,
            clock: Clock(Instant::now()),
        });
        let listener = tokio::spawn(listen(Arc::clone(&inner), listener, acceptor));
        Ok(TcpTransport { inner, listener })
    }

    /// The address the listener is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.inner.local_addr
    }

    /// This endpoint's Ed25519 public key, which its peers pin.
    pub fn public_key(&self) -> [u8; 32] {
        self.inner.key
    }

    /// The keys the listener accepts connections from: the plan's neighbours, which the owner
    /// keeps current.
    pub fn allowed(&self) -> &AllowedKeys {
        &self.inner.allowed
    }

    /// The pool `registration` names, if it was registered here.
    fn pool(&self, registration: PoolRegistration) -> Result<BufferPool, Error> {
        lock(&self.inner.pools)
            .iter()
            .find(|pool| pool.id() == registration.pool)
            .cloned()
            .ok_or_else(|| {
                Error::uncoded(
                    ErrorClass::InvalidRequest,
                    format!(
                        "pool {:?} is not registered with this transport",
                        registration.pool
                    ),
                )
            })
    }

    /// Opens the channel `spec.key` to `remote` over the first of its addresses that connects.
    async fn dial(&self, remote: &RemoteEndpoint, spec: ChannelSpec) -> Result<Channel, Error> {
        let pool = self.pool(spec.pool)?;
        let config = &self.inner.config;
        let connector = TlsConnector::from(Arc::new(
            tls::client_config(&config.seed, remote.key).map_err(tls_error)?,
        ));
        let mut last = link(ErrorCode::CONNECT_TIMEOUT, "the endpoint has no address");
        for addr in &remote.addrs {
            let attempt = tokio::time::timeout(
                config.handshake_timeout,
                connect_one(&connector, *addr, &spec, config, remote.key),
            )
            .await;
            match attempt {
                Ok(Ok((stream, ack))) => {
                    return self.open(stream, &spec, pool, ack_send(&ack, &spec), ack.wire_minor);
                }
                Ok(Err(error)) if error.class == ErrorClass::Auth => return Err(error),
                Ok(Err(error)) => last = error,
                Err(_) => {
                    last = link(
                        ErrorCode::CONNECT_TIMEOUT,
                        format!(
                            "no handshake with {addr} within {:?}",
                            config.handshake_timeout
                        ),
                    )
                }
            }
        }
        Err(last)
    }

    /// Answers the `Hello` of an arrived connection for `spec` and opens the channel.
    async fn answer(&self, spec: ChannelSpec, arrived: Arrived) -> Result<Channel, Error> {
        let Arrived {
            hello,
            credential,
            peer,
            mut stream,
            ..
        } = arrived;
        let pool = self.pool(spec.pool)?;
        let config = &self.inner.config;
        // The key may have been forbidden, or the credential stopped passing, since the `Hello`
        // arrived.
        let refusal = if self.inner.allowed.allows(&peer) {
            config
                .check
                .hello(&hello, hello.cred_kind, &credential, &peer)
                .err()
        } else {
            Some(ErrorCode::CREDENTIAL_INVALID)
        };
        if let Some(code) = refusal {
            refuse(&mut stream, &hello, code.0).await;
            return Err(Error::new(
                code,
                "the peer's key or credential no longer passes",
            ));
        }
        let tcp = stream.get_ref().0;
        size_buffers(tcp, spec.slot_size);
        let wire_minor = WIRE_MINOR.min(hello.wire_minor_max);
        let mismatch = matches!(hello.wire_dtype, Some(wire) if wire != spec.wire);
        let status = if mismatch {
            code::WIRE_DTYPE_MISMATCH
        } else {
            code::OK
        };
        let ack = HelloAck {
            status,
            wire_minor,
            capabilities: 0,
            limit_seq: u64::from(spec.recv_slots),
            slot_size: slot_size(&spec),
            recv_slots: spec.recv_slots,
            cred_kind: Some(config.credential.kind),
        };
        let frame = Control::HelloAck {
            ack,
            credential: &config.credential.bytes,
        }
        .encode_to_vec(&control_spec(&spec))
        .map_err(|error| Error::uncoded(ErrorClass::Internal, error.to_string()))?;
        tokio::time::timeout(config.handshake_timeout, async {
            stream.write_all(&frame).await?;
            stream.flush().await
        })
        .await
        .map_err(|_| {
            link(
                ErrorCode::CONNECT_TIMEOUT,
                "the HelloAck did not go out in time",
            )
        })?
        .map_err(|error| link(ErrorCode::PEER_RESET, error.to_string()))?;
        if mismatch {
            return Err(Error::new(
                ErrorCode::WIRE_DTYPE_MISMATCH,
                format!(
                    "the peer's Hello names {:?}, the channel {:?}",
                    hello.wire_dtype, spec.wire
                ),
            ));
        }
        let send = (hello.recv_slots > 0).then(|| SendConfig {
            peer_recv_slots: hello.recv_slots,
            peer_slot_size: hello.slot_size,
            retention: spec.retention.max(hello.recv_slots),
        });
        self.open(stream, &spec, pool, send, wire_minor)
    }

    /// The channel over `stream`, its handshakes done: the core, and the two tasks that drive it.
    fn open(
        &self,
        stream: TlsStream<TcpStream>,
        spec: &ChannelSpec,
        pool: BufferPool,
        send: Option<SendConfig>,
        wire_minor: u8,
    ) -> Result<Channel, Error> {
        let core = ChannelCore::new(CoreConfig {
            key: spec.key,
            wire_minor,
            cadence: CreditCadence::for_purpose(spec.key.purpose),
            alpha_p99: self.inner.config.alpha_p99,
            recv_slots: spec.recv_slots,
            slot_size: slot_size(spec),
            pool,
            send,
            ack_timeouts_to_fault: ACK_TIMEOUTS_TO_FAULT,
            crc_storm_count: CRC_STORM_COUNT,
        })?;
        #[cfg(target_os = "linux")]
        let fd = std::os::fd::AsRawFd::as_raw_fd(stream.get_ref().0);
        let shared = Arc::new(Mutex::new(Shared {
            core,
            link: None,
            dropped: false,
        }));
        let wake = Arc::new(Notify::new());
        let clock = self.inner.clock;
        let (reader, writer) = tokio::io::split(stream);
        let read = tokio::spawn(read_frames(
            reader,
            Arc::clone(&shared),
            Arc::clone(&wake),
            clock,
            #[cfg(target_os = "linux")]
            fd,
        ));
        tokio::spawn(write_frames(
            writer,
            Arc::clone(&shared),
            Arc::clone(&wake),
            clock,
        ));
        let pools = lock(&self.inner.pools).clone();
        Ok(Channel::new(
            spec.key,
            Box::new(TcpChannel {
                shared,
                wake,
                clock,
                pools,
                read,
            }),
        ))
    }
}

/// What a connector's sending direction is, from the acceptor's `HelloAck`.
fn ack_send(ack: &HelloAck, spec: &ChannelSpec) -> Option<SendConfig> {
    (ack.recv_slots > 0).then(|| SendConfig {
        peer_recv_slots: ack.recv_slots,
        peer_slot_size: ack.slot_size,
        retention: spec.retention.max(ack.recv_slots),
    })
}

/// The largest frame `spec`'s side accepts, as the handshake and the core take it.
fn slot_size(spec: &ChannelSpec) -> u32 {
    u32::try_from(spec.slot_size.0).unwrap_or(u32::MAX)
}

/// The control frames' header fields for `spec`'s channel.
fn control_spec(spec: &ChannelSpec) -> ControlSpec {
    ControlSpec {
        wire_minor: WIRE_MINOR,
        flags: Flags::default(),
        instance: spec.key.instance,
        epoch: spec.key.epoch,
    }
}

/// `SO_SNDBUF` and `SO_RCVBUF` at twice the slot size at least (RFC-0031 §5.3). The kernel may
/// round them; a refusal leaves its default, which still carries the frames.
fn size_buffers(tcp: &TcpStream, slot_size: Bytes) {
    let size = usize::try_from(slot_size.0.saturating_mul(2)).unwrap_or(usize::MAX);
    let socket = socket2::SockRef::from(tcp);
    let _ = socket.set_send_buffer_size(size);
    let _ = socket.set_recv_buffer_size(size);
}

/// Whether the purpose's channels carry activations, so `Hello` names their wire dtype.
fn carries_activations(purpose: ChannelPurpose) -> bool {
    matches!(
        purpose,
        ChannelPurpose::Activations | ChannelPurpose::Replay | ChannelPurpose::Collective
    )
}

/// Dials `addr`, runs the TLS handshake against `expected`, sends `Hello` and reads `HelloAck`.
async fn connect_one(
    connector: &TlsConnector,
    addr: SocketAddr,
    spec: &ChannelSpec,
    config: &TcpConfig,
    expected: [u8; 32],
) -> Result<(TlsStream<TcpStream>, HelloAck), Error> {
    let socket = if addr.is_ipv4() {
        TcpSocket::new_v4()
    } else {
        TcpSocket::new_v6()
    }
    .map_err(|error| link(ErrorCode::CONNECT_TIMEOUT, error.to_string()))?;
    let size = u32::try_from(spec.slot_size.0.saturating_mul(2)).unwrap_or(u32::MAX);
    let _ = socket.set_send_buffer_size(size);
    let _ = socket.set_recv_buffer_size(size);
    let tcp = socket.connect(addr).await.map_err(|error| {
        link(
            ErrorCode::CONNECT_TIMEOUT,
            format!("TCP connect to {addr}: {error}"),
        )
    })?;
    tcp.set_nodelay(true)
        .map_err(|error| link(ErrorCode::PEER_RESET, error.to_string()))?;
    let mut stream = TlsStream::Client(connector.connect(tls::server_name(), tcp).await.map_err(
        |error| {
            Error::new(
                ErrorCode::CREDENTIAL_INVALID,
                format!("TLS with {addr}: {error}"),
            )
        },
    )?);
    let hello = Hello {
        purpose: spec.key.purpose.into(),
        wire_minor_max: WIRE_MINOR,
        proto_major: PROTO_MAJOR,
        capabilities: 0,
        instance: spec.key.instance,
        epoch: spec.key.epoch,
        from_stage: spec.key.from,
        to_stage: spec.key.to,
        lane: spec.key.lane,
        wire_dtype: carries_activations(spec.key.purpose).then_some(spec.wire),
        slot_size: slot_size(spec),
        recv_slots: spec.recv_slots,
        cred_kind: config.credential.kind,
        channel_token: 0,
        resume_from: None,
    };
    let frame = Control::Hello {
        hello,
        credential: &config.credential.bytes,
    }
    .encode_to_vec(&control_spec(spec))
    .map_err(|error| Error::uncoded(ErrorClass::Internal, error.to_string()))?;
    stream
        .write_all(&frame)
        .await
        .map_err(|error| link(ErrorCode::PEER_RESET, error.to_string()))?;
    stream
        .flush()
        .await
        .map_err(|error| link(ErrorCode::PEER_RESET, error.to_string()))?;
    let bytes = read_control(&mut stream).await?;
    let frame = Frame::decode(&bytes).map_err(refused)?;
    let Control::HelloAck { ack, credential } = frame.control().map_err(refused)? else {
        return Err(Error::uncoded(
            ErrorClass::InvalidRequest,
            format!("the peer answered Hello with {:?}", frame.header.spec.ftype),
        ));
    };
    if ack.status != code::OK {
        return Err(Error::new(
            ErrorCode(ack.status),
            format!(
                "the peer refused the channel with status {:#06x}",
                ack.status
            ),
        ));
    }
    let kind = ack.cred_kind.ok_or_else(|| {
        Error::new(
            ErrorCode::CREDENTIAL_INVALID,
            "a HelloAck without a credential",
        )
    })?;
    config
        .check
        .hello_ack(&ack, kind, credential, &expected)
        .map_err(|code| Error::new(code, "the acceptor's credential was refused"))?;
    Ok((stream, ack))
}

/// A frame the handshake refused, as `net`'s error: never `Internal`.
fn refused(error: crate::frame::FrameError) -> Error {
    match error.code() {
        Some(code) => Error::new(ErrorCode(code), error.to_string()),
        None => Error::uncoded(error.class(), error.to_string()),
    }
}

/// Reads one control frame whole, refusing one longer than a control frame may be before it reads
/// past the header.
async fn read_control(stream: &mut (impl AsyncRead + Unpin)) -> Result<Vec<u8>, Error> {
    let mut frame = vec![0; HEADER_LEN];
    stream
        .read_exact(&mut frame)
        .await
        .map_err(|error| link(ErrorCode::PEER_RESET, error.to_string()))?;
    let ext_len = u32::from_le_bytes(frame[48..52].try_into().expect("4 bytes"));
    let payload_len = u32::from_le_bytes(frame[52..56].try_into().expect("4 bytes"));
    let len = HEADER_LEN as u64 + u64::from(ext_len) + u64::from(payload_len);
    if len > MAX_CONTROL_FRAME as u64 {
        return Err(Error::new(
            ErrorCode::FRAME_TOO_LARGE,
            format!("a {len}-byte handshake frame"),
        ));
    }
    frame.resize(len as usize, 0);
    stream
        .read_exact(&mut frame[HEADER_LEN..])
        .await
        .map_err(|error| link(ErrorCode::PEER_RESET, error.to_string()))?;
    Ok(frame)
}

/// The listener: each connection runs its handshakes on a task of its own, so a slow or silent
/// peer never holds up another.
async fn listen(inner: Arc<Inner>, listener: TcpListener, acceptor: TlsAcceptor) {
    loop {
        let (tcp, from) = match listener.accept().await {
            Ok(accepted) => accepted,
            Err(_) => {
                tokio::time::sleep(ACCEPT_BACKOFF).await;
                continue;
            }
        };
        // Over either cap the connection is closed at once and the listener goes on accepting,
        // so slow or silent peers can neither hold every handshake nor stop the listener.
        if let Some(slot) = Handshake::start(&inner, from.ip()) {
            tokio::spawn(arrive(Arc::clone(&inner), tcp, acceptor.clone(), slot));
        }
    }
}

/// One handshake in progress: a permit of [`MAX_HANDSHAKES`] and a count against its source
/// address, both given back when it ends.
struct Handshake {
    inner: Arc<Inner>,
    address: IpAddr,
    _permit: OwnedSemaphorePermit,
}

impl Handshake {
    /// A handshake for a connection from `address`, or `None` when a cap is reached.
    fn start(inner: &Arc<Inner>, address: IpAddr) -> Option<Handshake> {
        let permit = Arc::clone(&inner.handshakes).try_acquire_owned().ok()?;
        let mut counts = lock(&inner.per_address);
        match counts.iter_mut().find(|(known, _)| *known == address) {
            Some((_, count)) if *count >= MAX_HANDSHAKES_PER_ADDRESS => return None,
            Some((_, count)) => *count += 1,
            None => counts.push((address, 1)),
        }
        Some(Handshake {
            inner: Arc::clone(inner),
            address,
            _permit: permit,
        })
    }
}

impl Drop for Handshake {
    fn drop(&mut self) {
        let mut counts = lock(&self.inner.per_address);
        if let Some(at) = counts.iter().position(|(known, _)| *known == self.address) {
            counts[at].1 -= 1;
            if counts[at].1 == 0 {
                counts.swap_remove(at);
            }
        }
    }
}

/// One arriving connection: TLS, then its `Hello`, its credential checked, then on to the
/// `accept` of its key. A connection that fails is dropped, after a refusing `HelloAck` where
/// the `Hello` was read.
async fn arrive(inner: Arc<Inner>, tcp: TcpStream, acceptor: TlsAcceptor, slot: Handshake) {
    let timeout = inner.config.handshake_timeout;
    let arrived = tokio::time::timeout(timeout, async {
        tcp.set_nodelay(true).ok()?;
        let tls = tokio::time::timeout(TLS_TIMEOUT.min(timeout), acceptor.accept(tcp));
        let mut stream = TlsStream::Server(tls.await.ok()?.ok()?);
        let peer = {
            let (_, connection) = stream.get_ref();
            let certs = connection.peer_certificates()?;
            tls::spki_key(certs.first()?)?
        };
        let bytes = read_control(&mut stream).await.ok()?;
        let frame = Frame::decode(&bytes).ok()?;
        let Ok(Control::Hello { hello, credential }) = frame.control() else {
            return None;
        };
        let refusal = if hello.proto_major != PROTO_MAJOR {
            Some(code::PROTO_MAJOR_MISMATCH)
        } else {
            inner
                .config
                .check
                .hello(&hello, hello.cred_kind, credential, &peer)
                .err()
                .map(|code| code.0)
        };
        if let Some(status) = refusal {
            refuse(&mut stream, &hello, status).await;
            return None;
        }
        Some(Arrived {
            hello,
            credential: credential.to_vec(),
            peer,
            stream,
            at: Instant::now(),
        })
    })
    .await;
    drop(slot);
    let Ok(Some(arrived)) = arrived else {
        return;
    };
    let Ok(key) = channel_key(&arrived.hello) else {
        return;
    };
    let mut rendezvous = lock(&inner.rendezvous);
    // An accept waiting for this key takes it; one whose caller went away is skipped.
    let mut arrived = Some(arrived);
    while let Some(at) = rendezvous.waiting.iter().position(|(k, _)| *k == key) {
        let (_, sender) = rendezvous.waiting.remove(at);
        match sender.send(arrived.take().expect("still held")) {
            Ok(()) => return,
            Err(back) => arrived = Some(back),
        }
    }
    if let Some(arrived) = arrived {
        // A second connection for a key replaces the first, which a reconnect would do.
        rendezvous.arrived.retain(|(k, _)| *k != key);
        expire(&mut rendezvous.arrived, timeout);
        rendezvous.arrived.push((key, arrived));
    }
}

/// Sends a refusing `HelloAck`, best effort, and closes.
async fn refuse(stream: &mut TlsStream<TcpStream>, hello: &Hello, status: u16) {
    let ack = HelloAck {
        status,
        wire_minor: WIRE_MINOR.min(hello.wire_minor_max),
        capabilities: 0,
        limit_seq: 0,
        slot_size: 0,
        recv_slots: 0,
        cred_kind: None,
    };
    let spec = ControlSpec {
        wire_minor: WIRE_MINOR,
        flags: Flags::default(),
        instance: hello.instance,
        epoch: hello.epoch,
    };
    if let Ok(frame) = (Control::HelloAck {
        ack,
        credential: &[],
    })
    .encode_to_vec(&spec)
    {
        let _ = stream.write_all(&frame).await;
        let _ = stream.shutdown().await;
    }
}

/// The channel key a `Hello` names.
fn channel_key(hello: &Hello) -> Result<ChannelKey, ()> {
    Ok(ChannelKey {
        instance: hello.instance,
        epoch: hello.epoch,
        purpose: ChannelPurpose::try_from(hello.purpose).map_err(|_| ())?,
        from: hello.from_stage,
        to: hello.to_stage,
        lane: hello.lane,
    })
}

impl Transport for TcpTransport {
    fn kind(&self) -> TransportKind {
        TransportKind::Tcp
    }

    fn caps(&self) -> TransportCaps {
        TransportCaps {
            zero_copy_recv: false,
            device_memory_recv: false,
            max_frame: Bytes(MAX_FRAME),
            needs_registration: false,
            encrypted: true,
            reliable: true,
            max_channels_per_peer: u16::MAX,
            inline_max: 0,
        }
    }

    /// Records the pool, so a channel's frames may name it. TCP needs no registration with the
    /// OS, so the handle is 0. Register every pool before opening the channels that use it.
    fn register(&self, pool: &BufferPool) -> Result<PoolRegistration, Error> {
        let mut pools = lock(&self.inner.pools);
        if !pools.iter().any(|known| known.same_pool(pool)) {
            pools.push(pool.clone());
        }
        Ok(PoolRegistration {
            pool: pool.id(),
            transport: TransportKind::Tcp,
            handle: 0,
        })
    }

    /// Dials `remote`'s addresses in turn and opens the channel on the first that completes the
    /// handshakes. The acceptor's TLS key must be `remote.key`.
    async fn connect(&self, remote: &RemoteEndpoint, spec: ChannelSpec) -> Result<Channel, Error> {
        self.dial(remote, spec).await
    }

    async fn accept(&self, spec: ChannelSpec) -> Result<Channel, Error> {
        let waiting = {
            let mut rendezvous = lock(&self.inner.rendezvous);
            expire(&mut rendezvous.arrived, self.inner.config.handshake_timeout);
            match rendezvous.arrived.iter().position(|(k, _)| *k == spec.key) {
                Some(at) => Ok(rendezvous.arrived.remove(at).1),
                None => {
                    let (sender, receiver) = oneshot::channel();
                    rendezvous.waiting.push((spec.key, sender));
                    Err(receiver)
                }
            }
        };
        let arrived = match waiting {
            Ok(arrived) => arrived,
            Err(receiver) => receiver.await.map_err(|_| {
                Error::uncoded(
                    ErrorClass::Link,
                    "the listener stopped before the peer connected",
                )
            })?,
        };
        self.answer(spec, arrived).await
    }
}

/// A channel's core and its link state, shared by the channel's methods and its two tasks.
struct Shared {
    core: ChannelCore,
    /// Why the connection failed, once it has.
    link: Option<Error>,
    /// The channel was dropped: the writer closes the stream.
    dropped: bool,
}

/// One side of a TCP channel.
struct TcpChannel {
    shared: Arc<Mutex<Shared>>,
    wake: Arc<Notify>,
    clock: Clock,
    /// The pools registered when the channel opened, which its frames' slots come from.
    pools: Vec<BufferPool>,
    read: JoinHandle<()>,
}

impl Drop for TcpChannel {
    fn drop(&mut self) {
        lock(&self.shared).dropped = true;
        self.wake.notify_one();
        self.read.abort();
    }
}

impl ChannelBackend for TcpChannel {
    fn try_send(&mut self, frame: OutFrame) -> Result<SendTicket, SendError> {
        let ext = self
            .pools
            .iter()
            .find_map(|pool| pool.retain_ext(frame.ext))
            .ok_or(SendError::Invalid)?;
        let payload = self
            .pools
            .iter()
            .find_map(|pool| pool.retain_payload(frame.payload))
            .ok_or(SendError::Invalid)?;
        let now = self.clock.now();
        let ticket = {
            let mut shared = lock(&self.shared);
            if shared.link.is_some() {
                return Err(SendError::Link);
            }
            shared
                .core
                .send(frame.ftype, frame.step, frame.flags, ext, payload, now)?
        };
        self.wake.notify_one();
        Ok(ticket)
    }

    fn try_recv(&mut self) -> Option<InFrame> {
        lock(&self.shared).core.recv()
    }

    fn completed(&mut self) -> Option<SendTicket> {
        lock(&self.shared).core.completed()
    }

    fn stats(&self) -> ChannelStats {
        lock(&self.shared).core.stats()
    }

    fn credit(&self) -> u32 {
        lock(&self.shared).core.credit()
    }

    fn nack(&mut self, seq: u64) {
        lock(&self.shared).core.nack(seq);
        self.wake.notify_one();
    }

    fn close(&mut self, reason: CloseReason) {
        lock(&self.shared).core.close(reason);
        self.wake.notify_one();
    }
}

/// Records a failed connection, once.
fn fail(shared: &Mutex<Shared>, error: io::Error) {
    let mut shared = lock(shared);
    if shared.link.is_none() {
        shared.link = Some(link(ErrorCode::PEER_RESET, error.to_string()));
    }
}

/// Re-arms `TCP_QUICKACK`, which Linux clears after the delayed-ack decision it overrides
/// (RFC-0031 §5.3). Best effort: a refusal costs only a delayed ack.
#[cfg(target_os = "linux")]
fn quickack(fd: std::os::fd::RawFd) {
    // SAFETY: `fd` is the channel's TCP socket, which the reading half of its stream owns; this
    // runs on the reading task while it holds that half, so the descriptor is open, and the
    // borrow ends before the call returns.
    let socket = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };
    let _ = socket2::SockRef::from(&socket).set_tcp_quickack(true);
}

/// The reading task: each frame's header, then its body straight into a receive slot the core
/// lends, then the frame to the core.
async fn read_frames(
    mut reader: ReadHalf<TlsStream<TcpStream>>,
    shared: Arc<Mutex<Shared>>,
    wake: Arc<Notify>,
    clock: Clock,
    #[cfg(target_os = "linux")] fd: std::os::fd::RawFd,
) {
    let mut header = [0u8; HEADER_LEN];
    loop {
        if let Err(error) = reader.read_exact(&mut header).await {
            fail(&shared, error);
            wake.notify_one();
            return;
        }
        #[cfg(target_os = "linux")]
        quickack(fd);
        let admitted = lock(&shared).core.admit(&header);
        let Ok(len) = admitted else {
            // The core faulted on an oversized frame; the writer sends its `Close`.
            wake.notify_one();
            return;
        };
        // Every receive slot in use means the caller holds the frames; one comes back when it
        // drops them.
        let mut slot = loop {
            if let Some(slot) = lock(&shared).core.next_receive_slot() {
                break slot;
            }
            tokio::time::sleep(TICK).await;
        };
        slot.bytes_mut()[..HEADER_LEN].copy_from_slice(&header);
        if let Err(error) = reader
            .read_exact(&mut slot.bytes_mut()[HEADER_LEN..len])
            .await
        {
            fail(&shared, error);
            wake.notify_one();
            return;
        }
        #[cfg(target_os = "linux")]
        quickack(fd);
        let now = clock.now();
        lock(&shared).core.on_receive(slot.freeze(), len, now);
        wake.notify_one();
    }
}

/// Appends a transmit's bytes, contiguous as on the wire.
fn append(out: &mut Vec<u8>, transmit: Transmit<'_>) {
    match transmit {
        Transmit::Data {
            header,
            ext,
            payload,
            ..
        } => {
            out.extend_from_slice(header);
            out.extend_from_slice(ext);
            out.extend_from_slice(payload);
        }
        Transmit::Control(frame) => out.extend_from_slice(frame),
    }
}

/// The writing task: whenever woken, and every [`TICK`], runs the core's timers and writes what
/// the core has to send. It ends, closing the stream, once the channel is dropped or the
/// connection fails.
async fn write_frames(
    mut writer: WriteHalf<TlsStream<TcpStream>>,
    shared: Arc<Mutex<Shared>>,
    wake: Arc<Notify>,
    clock: Clock,
) {
    let mut out = Vec::new();
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            () = wake.notified() => {}
            _ = tick.tick() => {}
        }
        let (more, finished) = {
            let mut guard = lock(&shared);
            let now = clock.now();
            guard.core.on_tick(now);
            let mut more = false;
            while let Some(transmit) = guard.core.poll_transmit(now) {
                append(&mut out, transmit);
                if out.len() >= FLUSH_AT {
                    more = true;
                    break;
                }
            }
            (more, guard.dropped || guard.link.is_some())
        };
        if !out.is_empty() {
            let written = async {
                writer.write_all(&out).await?;
                writer.flush().await
            }
            .await;
            out.clear();
            if let Err(error) = written {
                fail(&shared, error);
                return;
            }
        }
        if finished {
            let _ = writer.shutdown().await;
            return;
        }
        if more {
            wake.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_never_reads_zero_and_never_goes_back() {
        let clock = Clock(Instant::now());
        let a = clock.now();
        let b = clock.now();
        assert!(a.0 >= 1 && b >= a);
    }

    #[test]
    fn a_hello_names_its_channel_key() {
        let hello = Hello {
            purpose: ChannelPurpose::Tokens.into(),
            wire_minor_max: WIRE_MINOR,
            proto_major: PROTO_MAJOR,
            capabilities: 0,
            instance: syntropy_core::InstanceId(5),
            epoch: syntropy_core::PlanEpoch { term: 1, seq: 2 },
            from_stage: syntropy_core::StageIx(3),
            to_stage: crate::transport::INGRESS,
            lane: 1,
            wire_dtype: None,
            slot_size: 4096,
            recv_slots: 8,
            cred_kind: CredKind::DataPlaneCredential,
            channel_token: 0,
            resume_from: None,
        };
        let key = channel_key(&hello).unwrap();
        assert_eq!(key.purpose, ChannelPurpose::Tokens);
        assert_eq!(key.to, crate::transport::INGRESS);
        assert_eq!((key.from, key.lane), (syntropy_core::StageIx(3), 1));
        let blob = Hello {
            purpose: crate::frame::Purpose::BlobStream,
            ..hello
        };
        assert!(channel_key(&blob).is_err());
    }

    /// `SO_SNDBUF` and `SO_RCVBUF` come out at least twice the slot, for a slot whose double is
    /// within the kernel's unprivileged cap (Linux's `net.core.wmem_max` and `rmem_max`, 208 KiB
    /// by default).
    #[tokio::test]
    async fn buffers_are_at_least_twice_the_slot() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (client, accepted) = tokio::join!(TcpStream::connect(addr), listener.accept());
        let (_client, (server, _)) = (client.unwrap(), accepted.unwrap());
        let slot = Bytes(48 << 10);
        size_buffers(&server, slot);
        let socket = socket2::SockRef::from(&server);
        let want = 2 * (48 << 10);
        assert!(socket.send_buffer_size().unwrap() >= want);
        assert!(socket.recv_buffer_size().unwrap() >= want);
    }

    /// The reading task's re-arm sets `TCP_QUICKACK` again after the kernel clears it.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn quickack_is_rearmed() {
        use std::os::fd::AsRawFd;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (client, accepted) = tokio::join!(TcpStream::connect(addr), listener.accept());
        let (_client, (server, _)) = (client.unwrap(), accepted.unwrap());
        let socket = socket2::SockRef::from(&server);
        socket.set_tcp_quickack(false).unwrap();
        assert!(!socket.tcp_quickack().unwrap());
        quickack(server.as_raw_fd());
        assert!(socket.tcp_quickack().unwrap());
    }

    #[test]
    fn only_activation_channels_name_a_wire_dtype() {
        assert!(carries_activations(ChannelPurpose::Activations));
        assert!(carries_activations(ChannelPurpose::Replay));
        assert!(!carries_activations(ChannelPurpose::Tokens));
        assert!(!carries_activations(ChannelPurpose::Sequence));
    }
}
