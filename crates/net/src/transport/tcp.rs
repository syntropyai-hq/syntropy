// SPDX-License-Identifier: MIT OR Apache-2.0

//! The TCP fallback backend (RFC-0031 §5.3; ADR-0011 item 2; DIST-E7-T05): the last backend of
//! ADR-0011's order, for an edge whose QUIC dials fail while TCP connects, because UDP is blocked.
//!
//! - **Identity.** TLS 1.3 authenticates both ends with the Ed25519 key form iroh uses, a raw
//!   public key ([`tls`]). The connector accepts only the key in its [`RemoteEndpoint`]; the
//!   acceptor only the keys its owner allowed ([`TcpTransport::allowed`]), the plan's neighbours.
//!   Both require the data ALPN, `syntropy/data/1`, once TLS is done.
//! - **One TCP connection per channel**, so no two channels share a byte stream and neither waits
//!   on the other's lost segments.
//! - **Socket options.** `TCP_NODELAY` is on. `SO_SNDBUF` and `SO_RCVBUF` are at least twice the
//!   larger `slot_size` of the channel's two sides where the OS allows it (the connector's
//!   receive buffer before `connect`, from its own slot), and otherwise left to the kernel's
//!   autotuning, which [`TcpStats::buffers_autotuned`] counts: on Linux an unprivileged request
//!   over `net.core.wmem_max` or `rmem_max` is clamped, and turns autotuning off. On Linux,
//!   `TCP_QUICKACK` is re-armed after every read off the socket (`QuickAck`), since the kernel
//!   clears it.
//!
//! **The channel** is [`stream`]'s, as on a QUIC stream: its handshake (`Hello`, `HelloAck`,
//! §6.9), and the `net` task that drives the [`ChannelCore`](super::ChannelCore), with the
//! dispatcher's side served from SPSC rings and atomics, with no lock (§4.4). This module opens
//! the TCP connection and its TLS session, and hands the stream over. The credentials and their
//! check are the owner's [`ChannelAuth`] (DIST-E7-T04;
//! [`UncheckedAuth`](super::stream::UncheckedAuth) until then).
//!
//! **Opening a channel.** The connector dials each of the endpoint's addresses in turn, runs
//! TLS, and offers `Hello`. The acceptor's listener runs TLS, reads the `Hello`, refuses it at
//! once if its protocol major or its credential fails, and hands the connection to the
//! [`Transport::accept`] of its channel key. One that arrives first is held for
//! [`TcpConfig::early_hold`], half the handshake timeout, then refused with `NotANeighbour`,
//! within the connector's own wait, as the QUIC backend does (RFC-0031 §4.3): so an acceptor
//! calls `accept` within `early_hold` of the peer's `Hello`. A second connection for a held key
//! replaces the first, as a reconnect would. `accept` checks the peer's key against the allowed
//! set again, and the whole `Hello` with its credential, since either may have stopped passing
//! while the connection waited.
//!
//! **The listener** takes at most [`MAX_HANDSHAKES`] connections through TLS and `Hello` at once,
//! and [`MAX_HANDSHAKES_PER_ADDRESS`] from one source; a connection over either is closed at
//! once, and the listener keeps accepting. TLS has [`TLS_TIMEOUT`], shorter than the handshake's.

pub mod tls;

use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::Duration;

use syntropy_core::resource::TransportKind;
use syntropy_core::{Bytes, ErrorClass, Nanos};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpSocket, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};
use tokio::task::JoinHandle;
use tokio_rustls::{TlsAcceptor, TlsConnector, TlsStream};

use self::tls::AllowedKeys;
use super::domain::DomainHandle;
use super::stream::{self, ChannelAuth, HelloIn, StreamSpec};
use super::{
    BufferPool, Channel, ChannelKey, ChannelSpec, PoolId, PoolRegistration, RemoteEndpoint,
    Transport, TransportCaps,
};
use crate::error::{Error, ErrorCode};

/// The largest frame on this backend: `net.max_frame`'s default, 64 MiB (RFC-0031 §2).
pub const MAX_FRAME: u64 = 64 << 20;

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

/// Connections held for a channel nobody accepts yet; more are refused \[target\].
const MAX_EARLY: usize = 64;

/// A [`TcpTransport`]'s settings.
#[derive(Clone)]
pub struct TcpConfig {
    /// This endpoint's Ed25519 seed: the runner key, or the node key for a daemon.
    pub seed: [u8; 32],
    /// Credentials for each channel's `Hello` and `HelloAck`, and the check of the peer's.
    pub auth: Arc<dyn ChannelAuth>,
    /// The capability set this side advertises in `Hello` (RFC-0031 §10).
    pub capabilities: u64,
    /// A link's α p99 before any sample, from the plan's link profile.
    pub alpha_p99: Nanos,
    /// The longest a dial, its TLS handshake and its `Hello` exchange may take together.
    pub handshake_timeout: Duration,
}

impl TcpConfig {
    /// How long a connection whose `Hello` has arrived waits for the `accept` of its channel:
    /// half the handshake timeout. It is then refused with `NotANeighbour`, within the
    /// connector's own wait, so the connector learns that the plans disagree (`Auth`) rather
    /// than timing out (`Link`). An acceptor calls `accept` within it of the peer's `Hello`.
    pub fn early_hold(&self) -> Duration {
        self.handshake_timeout / 2
    }
}

impl std::fmt::Debug for TcpConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpConfig")
            .field("capabilities", &self.capabilities)
            .field("alpha_p99", &self.alpha_p99)
            .field("handshake_timeout", &self.handshake_timeout)
            .finish_non_exhaustive()
    }
}

/// A [`TcpTransport`]'s counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TcpStats {
    /// Sockets whose send or receive buffer was left to the kernel's autotuning, because 2 ×
    /// `slot_size` was over the kernel's unprivileged cap (RFC-0031 §5.3, note of 2026-10-23).
    pub buffers_autotuned: u64,
}

#[derive(Debug, Default)]
struct Counters {
    buffers_autotuned: AtomicU64,
}

/// A TCP stream that re-arms `TCP_QUICKACK` after every read off it on Linux, since the kernel
/// clears the flag after the delayed-ack decision it overrides (RFC-0031 §5.3); elsewhere a
/// plain TCP stream. TLS runs over it, so every read the session makes off the socket re-arms
/// it. A refused re-arm costs only a delayed ack.
struct QuickAck(TcpStream);

impl QuickAck {
    /// The TCP stream, for its socket options.
    fn tcp(&self) -> &TcpStream {
        &self.0
    }

    fn rearm(&self) {
        #[cfg(target_os = "linux")]
        {
            let _ = socket2::SockRef::from(&self.0).set_tcp_quickack(true);
        }
    }
}

impl AsyncRead for QuickAck {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let before = buf.filled().len();
        let polled = Pin::new(&mut this.0).poll_read(cx, buf);
        if matches!(polled, Poll::Ready(Ok(()))) && buf.filled().len() > before {
            this.rearm();
        }
        polled
    }
}

impl AsyncWrite for QuickAck {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().0).poll_write(cx, buf)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().0).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.0.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().0).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().0).poll_shutdown(cx)
    }
}

/// A channel's TLS session over its TCP connection.
type Tls = TlsStream<QuickAck>;

/// A connection whose `Hello` has arrived, held for the `accept` of its key.
struct Early {
    stream: Tls,
    hello: HelloIn,
    /// The key the peer's TLS handshake proved.
    peer: [u8; 32],
}

/// An `accept` waiting for its channel's connection.
struct Waiter {
    spec: StreamSpec,
    reply: oneshot::Sender<Result<Channel, Error>>,
}

/// Early connections and waiting accepts, matched by channel key.
#[derive(Default)]
struct Routes {
    /// Each early connection with its arrival number, so its hold's end removes it and not a
    /// later connection that replaced it.
    early: HashMap<ChannelKey, (u64, Early)>,
    /// The next arrival number.
    arrivals: u64,
    waiters: HashMap<ChannelKey, Waiter>,
}

impl Routes {
    /// The `accept` waiting for `key`, if its caller is still there: one that went away is
    /// dropped.
    fn waiter(&mut self, key: &ChannelKey) -> Option<Waiter> {
        self.waiters
            .remove(key)
            .filter(|waiter| !waiter.reply.is_closed())
    }
}

struct Inner {
    config: TcpConfig,
    /// This endpoint's public key.
    key: [u8; 32],
    allowed: AllowedKeys,
    pools: Mutex<Vec<BufferPool>>,
    routes: Mutex<Routes>,
    /// One permit per handshake in progress, [`MAX_HANDSHAKES`] in all.
    handshakes: Arc<Semaphore>,
    /// Handshakes in progress by source ([`source`]).
    per_address: Mutex<Vec<(IpAddr, usize)>>,
    stats: Counters,
    local_addr: SocketAddr,
    domain: DomainHandle,
}

/// Locks `mutex`, past a panic of another holder: the state it guards stays consistent between
/// calls, so a poisoned lock is still usable.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The TCP fallback transport: one listener for the channels this endpoint accepts, and a dial
/// per channel it connects, both in the `net` domain.
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
    /// Listens on `addr` for the channels this endpoint accepts, in `domain`, which runs the
    /// listener and every channel's task.
    ///
    /// # Errors
    ///
    /// The listener's bind error, or a TLS configuration the seed cannot make.
    pub async fn bind(
        domain: &DomainHandle,
        config: TcpConfig,
        addr: SocketAddr,
    ) -> Result<TcpTransport, Error> {
        let task_domain = domain.clone();
        domain
            .spawn(async move {
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
                    routes: Mutex::new(Routes::default()),
                    handshakes: Arc::new(Semaphore::new(MAX_HANDSHAKES)),
                    per_address: Mutex::new(Vec::new()),
                    stats: Counters::default(),
                    local_addr,
                    domain: task_domain.clone(),
                });
                let listener = task_domain.spawn(listen(Arc::clone(&inner), listener, acceptor));
                Ok(TcpTransport { inner, listener })
            })
            .await
            .map_err(|error| {
                Error::uncoded(ErrorClass::Link, format!("the net domain stopped: {error}"))
            })?
    }

    /// The address the listener is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.inner.local_addr
    }

    /// This endpoint's Ed25519 public key, which its peers pin.
    pub fn public_key(&self) -> [u8; 32] {
        self.inner.key
    }

    /// The transport's counters.
    pub fn stats(&self) -> TcpStats {
        TcpStats {
            buffers_autotuned: self.inner.stats.buffers_autotuned.load(Ordering::Relaxed),
        }
    }

    /// The keys the listener accepts connections from: the plan's neighbours, which the owner
    /// keeps current.
    pub fn allowed(&self) -> &AllowedKeys {
        &self.inner.allowed
    }
}

impl Inner {
    /// The stream driver's spec for `spec`, its pool resolved.
    fn stream_spec(&self, spec: &ChannelSpec) -> Result<StreamSpec, Error> {
        let pools = lock(&self.pools).clone();
        let pool = pools
            .iter()
            .find(|pool| pool.id() == spec.pool.pool)
            .cloned()
            .ok_or_else(|| {
                Error::uncoded(
                    ErrorClass::InvalidRequest,
                    format!(
                        "pool {:?} is not registered with this transport",
                        spec.pool.pool
                    ),
                )
            })?;
        let slot_size = u32::try_from(spec.slot_size.0)
            .ok()
            .filter(|size| u64::from(*size) <= MAX_FRAME)
            .ok_or_else(|| {
                Error::uncoded(
                    ErrorClass::InvalidRequest,
                    format!(
                        "a {}-byte slot is larger than net.max_frame, {MAX_FRAME}",
                        spec.slot_size.0
                    ),
                )
            })?;
        Ok(StreamSpec {
            key: spec.key,
            pool,
            slot_size,
            recv_slots: spec.recv_slots,
            wire: spec.wire,
            retention: spec.retention,
            capabilities: self.config.capabilities,
            alpha_p99: self.config.alpha_p99,
            send_pools: pools,
            max_frame: MAX_FRAME,
        })
    }

    /// Counts a socket left to autotuning.
    fn autotuned(&self) {
        self.stats.buffers_autotuned.fetch_add(1, Ordering::Relaxed);
    }

    /// Dials `addr` for the channel `spec`, runs TLS against `remote.key`, offers `Hello`, and
    /// opens the channel.
    async fn dial(
        &self,
        addr: SocketAddr,
        remote: &RemoteEndpoint,
        spec: StreamSpec,
    ) -> Result<Channel, Error> {
        let socket = if addr.is_ipv4() {
            TcpSocket::new_v4()
        } else {
            TcpSocket::new_v6()
        }
        .map_err(|error| link(ErrorCode::CONNECT_TIMEOUT, error.to_string()))?;
        // The receive buffer goes before `connect`, since the window scale is fixed in the SYN.
        // It follows this side's slot; the send buffer follows the peer's, which `HelloAck`
        // gives.
        let mut autotuned = false;
        match buffer_request(Bytes(u64::from(spec.slot_size)), buffer_caps().1) {
            Some(size) => {
                let _ = socket.set_recv_buffer_size(u32::try_from(size).unwrap_or(u32::MAX));
            }
            None => autotuned = true,
        }
        let tcp = socket.connect(addr).await.map_err(|error| {
            link(
                ErrorCode::CONNECT_TIMEOUT,
                format!("TCP connect to {addr}: {error}"),
            )
        })?;
        tcp.set_nodelay(true)
            .map_err(|error| link(ErrorCode::PEER_RESET, error.to_string()))?;
        let connector = TlsConnector::from(Arc::new(
            tls::client_config(&self.config.seed, remote.key).map_err(tls_error)?,
        ));
        let tls = TlsStream::Client(
            connector
                .connect(tls::server_name(), QuickAck(tcp))
                .await
                .map_err(|error| {
                    Error::new(
                        ErrorCode::CREDENTIAL_INVALID,
                        format!("TLS with {addr}: {error}"),
                    )
                })?,
        );
        if !tls::agreed_alpn(tls.get_ref().1) {
            return Err(Error::new(
                ErrorCode::PROTO_MAJOR_MISMATCH,
                format!("{addr} did not agree on the data ALPN"),
            ));
        }
        let (mut r, mut w) = tokio::io::split(tls);
        let agreed = stream::offer(
            &mut r,
            &mut w,
            &spec,
            &remote.key,
            self.config.auth.as_ref(),
        )
        .await?;
        // The send buffer, now that the peer's slot is known.
        let tls = r.unsplit(w);
        let largest = spec.slot_size.max(agreed.peer_slot_size);
        match buffer_request(Bytes(u64::from(largest)), buffer_caps().0) {
            Some(size) => {
                let _ = socket2::SockRef::from(tls.get_ref().0.tcp()).set_send_buffer_size(size);
            }
            None => autotuned = true,
        }
        if autotuned {
            self.autotuned();
        }
        let (r, w) = tokio::io::split(tls);
        Ok(stream::open(r, w, spec, agreed, &self.domain)?.spawn(&self.domain))
    }

    /// The acceptor's end of the handshake on `early`'s connection, then the channel. The key is
    /// checked against the allowed set again, and the credential, since either may have stopped
    /// passing while the connection waited.
    async fn complete(&self, early: Early, spec: StreamSpec) -> Result<Channel, Error> {
        let Early {
            mut stream,
            hello,
            peer,
        } = early;
        let auth = self.config.auth.as_ref();
        let agreed = if self.allowed.allows(&peer) {
            stream::check_hello(&hello, &spec, &peer, auth)
        } else {
            Err(ErrorCode::CREDENTIAL_INVALID)
        };
        let largest = spec.slot_size.max(hello.hello.slot_size);
        if size_buffers(stream.get_ref().0.tcp(), Bytes(u64::from(largest))) {
            self.autotuned();
        }
        stream::answer(
            &mut stream,
            &spec,
            agreed.as_ref().map_err(|code| *code),
            auth,
        )
        .await?;
        let agreed = agreed
            .map_err(|code| Error::new(code, format!("refused the channel {:?}", spec.key)))?;
        let (r, w) = tokio::io::split(stream);
        Ok(stream::open(r, w, spec, agreed, &self.domain)?.spawn(&self.domain))
    }

    /// Hands an arrived connection to the `accept` waiting for its key, or holds it for one, for
    /// [`TcpConfig::early_hold`] at most.
    async fn route(&self, early: Early) {
        let Some(key) = early.hello.key() else {
            // A blob stream is not a channel.
            return;
        };
        // The protocol major and the credential are checked on arrival, so a connection that
        // fails either is refused at once instead of being held; `accept` checks them again.
        let auth = self.config.auth.as_ref();
        let hello = &early.hello;
        let refused = if hello.hello.proto_major != stream::PROTO_MAJOR {
            Err(ErrorCode::PROTO_MAJOR_MISMATCH)
        } else {
            auth.check(&early.peer, &key, hello.hello.cred_kind, &hello.credential)
        };
        if let Err(code) = refused {
            let mut stream = early.stream;
            let _ = stream::answer(&mut stream, &placeholder_spec(&key), Err(code), auth).await;
            return;
        }
        /// What to do with the connection, decided under the routing lock and done after it.
        enum Next {
            Complete(Waiter, Early),
            /// Held, with its arrival number, and the connection it replaced, if any, which is
            /// closed once the lock is released.
            Hold(u64, Option<Early>),
            Refuse(Early),
        }
        let next = {
            let mut routes = lock(&self.routes);
            match routes.waiter(&key) {
                Some(waiter) => Next::Complete(waiter, early),
                // A second connection for a key replaces the first, as a reconnect would.
                None if routes.early.len() < MAX_EARLY || routes.early.contains_key(&key) => {
                    let arrival = routes.arrivals;
                    routes.arrivals += 1;
                    let replaced = routes.early.insert(key, (arrival, early));
                    Next::Hold(arrival, replaced.map(|(_, early)| early))
                }
                None => Next::Refuse(early),
            }
        };
        let refuse = match next {
            Next::Complete(waiter, early) => {
                let result = self.complete(early, waiter.spec).await;
                let _ = waiter.reply.send(result);
                return;
            }
            Next::Refuse(early) => Some(early),
            Next::Hold(arrival, replaced) => {
                drop(replaced);
                // Refuse the early connection if nobody accepts its channel in time: well within
                // the connector's own wait, so it gets the refusal, not a timeout.
                tokio::time::sleep(self.config.early_hold()).await;
                let mut routes = lock(&self.routes);
                match routes.early.get(&key) {
                    Some((held, _)) if *held == arrival => {
                        routes.early.remove(&key).map(|(_, early)| early)
                    }
                    _ => None,
                }
            }
        };
        if let Some(Early { mut stream, .. }) = refuse {
            let _ = stream::answer(
                &mut stream,
                &placeholder_spec(&key),
                Err(ErrorCode::NOT_A_NEIGHBOUR),
                self.config.auth.as_ref(),
            )
            .await;
        }
    }
}

/// The spec a refusal is written with: only its key is read.
fn placeholder_spec(key: &ChannelKey) -> StreamSpec {
    StreamSpec {
        key: *key,
        pool: BufferPool::heap(PoolId(0), 64, 1).expect("a one-slot pool of 64 bytes"),
        slot_size: 0,
        recv_slots: 0,
        wire: syntropy_core::plan::WireDType::F32,
        retention: 0,
        capabilities: 0,
        alpha_p99: Nanos(0),
        send_pools: Vec::new(),
        max_frame: 0,
    }
}

/// The largest `SO_SNDBUF` and `SO_RCVBUF` an unprivileged process may ask for: Linux's
/// `net.core.wmem_max` and `net.core.rmem_max`, read once. `None` where the cap is not known,
/// so the request is made as asked. The values differ between hosts: hosted CI runners do not
/// all keep the 212 992-byte default.
fn buffer_caps() -> (Option<usize>, Option<usize>) {
    #[cfg(target_os = "linux")]
    {
        static CAPS: std::sync::OnceLock<(Option<usize>, Option<usize>)> =
            std::sync::OnceLock::new();
        *CAPS.get_or_init(|| {
            let read = |name: &str| {
                std::fs::read_to_string(format!("/proc/sys/net/core/{name}"))
                    .ok()?
                    .trim()
                    .parse()
                    .ok()
            };
            (read("wmem_max"), read("rmem_max"))
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        (None, None)
    }
}

/// The buffer size to request for a channel whose slots hold `slot_size` bytes, against the
/// kernel's `cap`: 2 × `slot_size` where it fits (RFC-0031 §5.3), or `None` to leave the kernel
/// its autotuning. On Linux, a request over the cap would be clamped to it and would also turn
/// autotuning off, so a large slot would get a smaller buffer than not asking.
fn buffer_request(slot_size: Bytes, cap: Option<usize>) -> Option<usize> {
    let want = usize::try_from(slot_size.0.saturating_mul(2)).unwrap_or(usize::MAX);
    match cap {
        Some(cap) if want > cap => None,
        _ => Some(want),
    }
}

/// The send and receive buffer sizes to request for `slot_size`, and whether either was left to
/// autotuning.
fn buffer_plan(slot_size: Bytes) -> (Option<usize>, Option<usize>, bool) {
    let (send_cap, recv_cap) = buffer_caps();
    let send = buffer_request(slot_size, send_cap);
    let recv = buffer_request(slot_size, recv_cap);
    (send, recv, send.is_none() || recv.is_none())
}

/// `SO_SNDBUF` and `SO_RCVBUF` on a connected socket, as [`buffer_plan`] gives them for the
/// channel's larger slot, either side's. The kernel may round them; a refusal leaves its
/// default, which still carries the frames. Returns whether either was left to autotuning.
fn size_buffers(tcp: &TcpStream, slot_size: Bytes) -> bool {
    let (send, recv, autotuned) = buffer_plan(slot_size);
    let socket = socket2::SockRef::from(tcp);
    if let Some(size) = send {
        let _ = socket.set_send_buffer_size(size);
    }
    if let Some(size) = recv {
        let _ = socket.set_recv_buffer_size(size);
    }
    autotuned
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
        if let Some(slot) = Handshake::start(&inner, source(from.ip())) {
            tokio::spawn(arrive(Arc::clone(&inner), tcp, acceptor.clone(), slot));
        }
    }
}

/// What a handshake counts against for [`MAX_HANDSHAKES_PER_ADDRESS`]: an IPv4 address, or an
/// IPv6 address's /64, since one host is commonly given a whole /64. An IPv4-mapped IPv6 address
/// counts as its IPv4 address.
fn source(ip: IpAddr) -> IpAddr {
    match ip.to_canonical() {
        IpAddr::V6(v6) => {
            let [a, b, c, d, ..] = v6.segments();
            IpAddr::V6(std::net::Ipv6Addr::new(a, b, c, d, 0, 0, 0, 0))
        }
        v4 => v4,
    }
}

/// One handshake in progress: a permit of [`MAX_HANDSHAKES`] and a count against its source,
/// both given back when it ends.
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

/// One arriving connection: TLS, the data ALPN, then its `Hello`, then on to the `accept` of its
/// key. A connection that fails any of them is dropped.
async fn arrive(inner: Arc<Inner>, tcp: TcpStream, acceptor: TlsAcceptor, slot: Handshake) {
    let timeout = inner.config.handshake_timeout;
    let arrived = tokio::time::timeout(timeout, async {
        tcp.set_nodelay(true).ok()?;
        let tls = tokio::time::timeout(TLS_TIMEOUT.min(timeout), acceptor.accept(QuickAck(tcp)));
        let mut stream = TlsStream::Server(tls.await.ok()?.ok()?);
        // A connector that offered no ALPN completes TLS without one; it is not a data channel.
        if !tls::agreed_alpn(stream.get_ref().1) {
            return None;
        }
        let peer = {
            let (_, connection) = stream.get_ref();
            let certs = connection.peer_certificates()?;
            tls::spki_key(certs.first()?)?
        };
        let hello = stream::read_hello(&mut stream).await.ok()?;
        Some(Early {
            stream,
            hello,
            peer,
        })
    })
    .await;
    drop(slot);
    if let Ok(Some(early)) = arrived {
        inner.route(early).await;
    }
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
        let inner = Arc::clone(&self.inner);
        let remote = remote.clone();
        self.inner
            .domain
            .spawn(async move {
                let spec = inner.stream_spec(&spec)?;
                let timeout = inner.config.handshake_timeout;
                let mut last = link(ErrorCode::CONNECT_TIMEOUT, "the endpoint has no address");
                for addr in &remote.addrs {
                    match tokio::time::timeout(timeout, inner.dial(*addr, &remote, spec.clone()))
                        .await
                    {
                        Ok(Ok(channel)) => return Ok(channel),
                        Ok(Err(error)) if error.class == ErrorClass::Auth => return Err(error),
                        Ok(Err(error)) => last = error,
                        Err(_) => {
                            last = link(
                                ErrorCode::CONNECT_TIMEOUT,
                                format!("no handshake with {addr} within {timeout:?}"),
                            )
                        }
                    }
                }
                Err(last)
            })
            .await
            .map_err(|error| {
                Error::uncoded(ErrorClass::Link, format!("the net domain stopped: {error}"))
            })?
    }

    /// Opens the channel `spec.key` on the connection that names it: one held since it arrived,
    /// or the next. The wait runs in the caller's task, so a caller that stops waiting, such as
    /// on its own timeout, leaves no `accept` registered, and may call again.
    async fn accept(&self, spec: ChannelSpec) -> Result<Channel, Error> {
        let spec = self.inner.stream_spec(&spec)?;
        let (reply, answer) = oneshot::channel();
        let early = {
            let mut routes = lock(&self.inner.routes);
            if let Some(waiter) = routes.waiter(&spec.key) {
                routes.waiters.insert(spec.key, waiter);
                return Err(Error::uncoded(
                    ErrorClass::InvalidRequest,
                    format!("the channel {:?} is already being accepted", spec.key),
                ));
            }
            match routes.early.remove(&spec.key) {
                Some((_, early)) => Some(early),
                None => {
                    routes.waiters.insert(
                        spec.key,
                        Waiter {
                            spec: spec.clone(),
                            reply,
                        },
                    );
                    None
                }
            }
        };
        let stopped = |error: tokio::task::JoinError| {
            Error::uncoded(ErrorClass::Link, format!("the net domain stopped: {error}"))
        };
        match early {
            Some(early) => {
                let inner = Arc::clone(&self.inner);
                self.inner
                    .domain
                    .spawn(async move { inner.complete(early, spec).await })
                    .await
                    .map_err(stopped)?
            }
            None => answer.await.map_err(|_| {
                Error::uncoded(
                    ErrorClass::Link,
                    "the transport closed before the channel opened",
                )
            })?,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(
            !size_buffers(&server, slot),
            "96 KiB fits under the default caps"
        );
        let socket = socket2::SockRef::from(&server);
        let want = 2 * (48 << 10);
        assert!(socket.send_buffer_size().unwrap() >= want);
        assert!(socket.recv_buffer_size().unwrap() >= want);
    }

    /// 2 × `slot_size` is requested where the cap allows it, and left to autotuning where it does
    /// not, or where the cap is not known.
    #[test]
    fn a_buffer_is_requested_only_under_the_cap() {
        let slot = Bytes(100 << 10);
        assert_eq!(buffer_request(slot, None), Some(200 << 10));
        assert_eq!(buffer_request(slot, Some(200 << 10)), Some(200 << 10));
        assert_eq!(buffer_request(slot, Some((200 << 10) - 1)), None);
        assert_eq!(
            buffer_request(Bytes(u64::MAX), Some(usize::MAX)),
            Some(usize::MAX)
        );
    }

    /// On Linux, whose caps are known, a slot whose double is over them is left to autotuning,
    /// and the plan says so.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_slot_over_the_linux_cap_is_left_to_autotuning() {
        // Linux always exposes the caps, on hosted runners too, so this test never skips: the
        // backend must read the same values as this test, and a reader that returns `None`
        // fails here.
        let read = |name: &str| -> usize {
            std::fs::read_to_string(format!("/proc/sys/net/core/{name}"))
                .unwrap_or_else(|error| panic!("net.core.{name}: {error}"))
                .trim()
                .parse()
                .unwrap_or_else(|error| panic!("net.core.{name}: {error}"))
        };
        let (send_cap, recv_cap) = (read("wmem_max"), read("rmem_max"));
        assert_eq!(buffer_caps(), (Some(send_cap), Some(recv_cap)));
        let large = Bytes(u64::try_from(send_cap.max(recv_cap)).unwrap());
        let (send, recv, autotuned) = buffer_plan(large);
        assert_eq!((send, recv, autotuned), (None, None, true));
        let small = Bytes(u64::try_from(send_cap.min(recv_cap) / 4).unwrap());
        assert!(!buffer_plan(small).2);
    }

    /// One host's handshakes are counted together: an IPv6 /64 as one source, and an IPv4-mapped
    /// address as its IPv4 address.
    #[test]
    fn a_handshake_counts_against_its_hosts_source() {
        use std::net::{Ipv4Addr, Ipv6Addr};
        let v4 = IpAddr::V4(Ipv4Addr::new(192, 0, 2, 7));
        assert_eq!(source(v4), v4);
        assert_eq!(
            source(IpAddr::V6(Ipv4Addr::new(192, 0, 2, 7).to_ipv6_mapped())),
            v4
        );
        let a = IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 1, 2, 3, 4, 5, 6));
        let b = IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 1, 2, 0xffff, 0, 0, 1));
        let other = IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 1, 3, 3, 4, 5, 6));
        assert_eq!(source(a), source(b));
        assert_ne!(source(a), source(other));
    }

    /// A read through [`QuickAck`] sets `TCP_QUICKACK` again after the kernel clears it.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn quickack_is_rearmed_after_a_read() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (client, accepted) = tokio::join!(TcpStream::connect(addr), listener.accept());
        let (mut client, (server, _)) = (client.unwrap(), accepted.unwrap());
        socket2::SockRef::from(&server)
            .set_tcp_quickack(false)
            .unwrap();
        assert!(!socket2::SockRef::from(&server).tcp_quickack().unwrap());
        let mut reader = QuickAck(server);
        client.write_all(b"x").await.unwrap();
        let mut byte = [0u8; 1];
        reader.read_exact(&mut byte).await.unwrap();
        assert!(socket2::SockRef::from(reader.tcp()).tcp_quickack().unwrap());
    }
}
