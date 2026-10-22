// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`QuicTransport`]: the data plane's QUIC backend on iroh (RFC-0031 §4; ADR-0308).
//!
//! - **Endpoints** (§4.1). A daemon binds its endpoint with the node key and a runner with its
//!   runner key: the same type, bound in the runner's or the daemon's `net` domain
//!   ([`NetDomain`](super::super::domain::NetDomain)). Dialling hands iroh every address the
//!   [`RemoteEndpoint`] names, and its relay; iroh races the paths and keeps the best.
//!   [`QuicTransport::path_kind`] reports the selected one as `Quic` or `QuicRelay`.
//! - **Two connections per peer** (§4.2, ADR-0308 as revised 2026-10-19). The latency connection
//!   carries every channel but `Replay` and `Bulk`, which take the bulk connection, so a replay
//!   burst never sits in the decode stream's congestion window. Both speak `syntropy/data/1`
//!   ([`DATA_ALPN`]). The side that opens a channel dials the connection if neither side has one
//!   yet; a connection the peer dialled is reused once its first channel shows its kind.
//! - **One bidirectional stream per channel.** The opener sends `Hello` on it and the acceptor
//!   answers `HelloAck` ([`stream`](super::super::stream)); the acceptor routes each new stream
//!   by the key in its `Hello`, so it never needs to know which connection is which. The stream's
//!   priority is the channel's, with the decode lane of an `Activations` hop above its prefill
//!   lane.
//! - **Windows** (§4.3). Each connection's stream receive window is twice `net.max_frame`, twice
//!   the largest slot any of its channels can have. Its connection receive window is the sum of
//!   the credited bytes, `recv_slots × slot_size` plus room for control frames, over the channels
//!   opened on it, set with `Connection::set_receive_window` before each channel's handshake.
//!   [`QuicTransport::connection_windows`] shows each one.
//! - **Early streams.** A stream whose `Hello` comes before its channel is accepted is held for
//!   [`QuicSettings::early_hold`], half the shorter connect timeout, then refused with
//!   `NotANeighbour`: within the connector's wait on the same settings, so it learns that the plans
//!   disagree (`Auth`) rather than timing out (`Link`), which would make it reconnect.
//! - **Timeouts.** A dial or a handshake that outlasts `link.connect_timeout` fails with
//!   `ConnectTimeout` (0x020A). A refusal, such as an unaccepted ALPN or a `HelloAck` with a
//!   §12 status, keeps its own error: a refusal is `Link` with no code, or the status's code.
//!
//! The connect and accept futures are `Send` and may run on any runtime: they hand the work to the
//! domain's thread and wait for its answer.
//!
//! **`quic.max_conn_window`** (RFC-0031 §4.3, note of 2026-10-23 from this task) is a sanity
//! bound, 64 MiB by default: when the credited sum passes it the window still follows the sum, so
//! credits never stall, and [`WindowReport::over_cap`] shows it. `net` has no logger yet
//! (RFC-0053); the report is what the runner's telemetry reads.
//!
//! **Windows.** The runtime cannot wait on a `syntropy-spsc` ring's event handle there, so a
//! parked channel's task re-checks its rings every 1 ms
//! ([`PARKED_POLL`](crate::transport::domain::PARKED_POLL)); while live sequences exist it
//! busy-polls, as on other platforms. DIST-E7-T24 adds a wait bridge.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use iroh::endpoint::{ConnectOptions, Connection, RecvStream, SendStream};
use syntropy_core::resource::TransportKind;
use syntropy_core::{ErrorClass, Nanos};
use tokio::sync::{OnceCell, oneshot};

use super::settings::{LinkClass, QuicSettings, varint};
use crate::endpoint::{self, NodeAddr, PathKind};
use crate::error::{Error, ErrorCode};
use crate::transport::buffer::{BufferPool, PoolRegistration};
use crate::transport::core::{CONTROL_RECEIVES, MAX_CONTROL_FRAME};
use crate::transport::domain::DomainHandle;
use crate::transport::stream::{self, ChannelAuth, HelloIn, StreamSpec};
use crate::transport::{
    Channel, ChannelKey, ChannelPriority, ChannelPurpose, ChannelSpec, RemoteEndpoint, Transport,
    TransportCaps,
};

/// The data plane's application protocol: sequence and stage channels (RFC-0031 §1).
pub const DATA_ALPN: &[u8] = b"syntropy/data/1";

/// The α p99 a channel's core starts from before its first sample, until the plan's link profile
/// reaches the transport: 1 ms, so its first ack timeout is 4 ms (§7.3).
pub const INITIAL_ALPHA_P99: Nanos = Nanos(1_000_000);

/// The early streams held for a channel nobody accepts yet; more are refused.
const MAX_EARLY: usize = 64;

/// Which of a peer's two connections a channel takes (RFC-0031 §4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// Decode and prefill lanes, ring-return tokens, collectives, sequence and tap channels.
    Latency,
    /// `Replay` and `Bulk` channels: replay, KV migration and tensor-group reads.
    Bulk,
}

impl Role {
    /// The connection a channel of `purpose` takes.
    pub fn of(purpose: ChannelPurpose) -> Role {
        match purpose {
            ChannelPurpose::Replay | ChannelPurpose::Bulk => Role::Bulk,
            _ => Role::Latency,
        }
    }
}

/// A channel stream's QUIC priority: its class, with the decode lane of an `Activations` hop
/// above the prefill lane (RFC-0031 §4.2). Higher is sent first.
pub fn stream_priority(key: &ChannelKey, priority: ChannelPriority) -> i32 {
    let decode_lane = key.purpose == ChannelPurpose::Activations && key.lane == 0;
    (priority as i32) * 2 + i32::from(decode_lane)
}

/// The link class `link.*`'s defaults use for a peer: LAN when every direct address the peer
/// gave is loopback, link-local or private, WAN otherwise, a relay-only peer among them.
pub fn link_class(remote: &RemoteEndpoint) -> LinkClass {
    let local = |addr: &SocketAddr| match addr.ip() {
        IpAddr::V4(ip) => ip.is_loopback() || ip.is_private() || ip.is_link_local(),
        IpAddr::V6(ip) => ip.is_loopback() || ip.is_unique_local() || ip.is_unicast_link_local(),
    };
    if !remote.addrs.is_empty() && remote.addrs.iter().all(local) {
        LinkClass::Lan
    } else {
        LinkClass::Wan
    }
}

/// How to bind a [`QuicTransport`].
#[derive(Clone)]
pub struct QuicConfig {
    /// The endpoint's relays and direct paths. The data ALPN is added to what it accepts.
    pub endpoint: endpoint::Config,
    /// RFC-0031 §4.3's values.
    pub settings: QuicSettings,
    /// Credentials for each channel's `Hello` and `HelloAck`.
    pub auth: Arc<dyn ChannelAuth>,
    /// The capability set this side advertises in `Hello` (RFC-0031 §10).
    pub capabilities: u64,
}

impl std::fmt::Debug for QuicConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuicConfig")
            .field("endpoint", &self.endpoint)
            .field("settings", &self.settings)
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

/// One connection's receive window, as [`QuicTransport::connection_windows`] shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowReport {
    /// The peer's endpoint key.
    pub peer: [u8; 32],
    pub role: Role,
    /// The channels opened on it.
    pub channels: usize,
    /// The bytes their credits allow, control frames included.
    pub credited: u64,
    /// The connection receive window set: never below `credited`.
    pub window: u64,
    /// The window passed `quic.max_conn_window`.
    pub over_cap: bool,
}

/// A connection to a peer, shared by its channels.
struct Conn {
    conn: Connection,
    /// Set when this side dials it, or by the first channel the peer opens on it.
    role: OnceLock<Role>,
    peer: [u8; 32],
    window: Mutex<Window>,
}

#[derive(Default)]
struct Window {
    channels: usize,
    credited: u64,
    window: u64,
    over_cap: bool,
}

/// A stream whose `Hello` arrived before anyone accepted its channel.
struct Early {
    conn: Arc<Conn>,
    send: SendStream,
    recv: RecvStream,
    hello: HelloIn,
}

/// An `accept` waiting for its channel's stream.
struct Waiter {
    spec: StreamSpec,
    reply: oneshot::Sender<Result<Channel, Error>>,
}

/// Each peer's connections by kind, each dialled at most once at a time.
type Conns = HashMap<([u8; 32], Role), Arc<OnceCell<Arc<Conn>>>>;

#[derive(Default)]
struct Routes {
    waiters: HashMap<ChannelKey, Waiter>,
    early: HashMap<ChannelKey, Early>,
}

struct Inner {
    domain: DomainHandle,
    endpoint: endpoint::Endpoint,
    settings: QuicSettings,
    auth: Arc<dyn ChannelAuth>,
    capabilities: u64,
    pools: Mutex<Vec<BufferPool>>,
    conns: Mutex<Conns>,
    routes: Mutex<Routes>,
}

/// The QUIC backend of RFC-0031 §4. See the [module](self).
#[derive(Clone)]
pub struct QuicTransport {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for QuicTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuicTransport")
            .field("endpoint", &self.inner.endpoint)
            .finish_non_exhaustive()
    }
}

/// Locks a control-path map; a panic while holding one is a bug that already failed a test.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn link(message: impl Into<String>) -> Error {
    Error::uncoded(ErrorClass::Link, message)
}

impl QuicTransport {
    /// Binds an endpoint whose key is the public half of `secret_key`, in `domain`, accepting
    /// data channels.
    ///
    /// # Errors
    ///
    /// The endpoint's bind error.
    pub async fn bind(
        domain: &DomainHandle,
        secret_key: [u8; 32],
        config: QuicConfig,
    ) -> Result<QuicTransport, Error> {
        let task_domain = domain.clone();
        domain
            .spawn(async move {
                let QuicConfig {
                    endpoint: endpoint_config,
                    settings,
                    auth,
                    capabilities,
                } = config;
                let accept = endpoint_config.accept_alpn(DATA_ALPN);
                // Accepted connections take the WAN idle timeout; QUIC uses the smaller of the
                // two sides', so a dialler on a LAN path brings its 2 s.
                let transport = settings.transport_config(LinkClass::Wan);
                let endpoint = endpoint::Endpoint::bind_with(secret_key, accept, move |builder| {
                    builder.transport_config(transport)
                })
                .await?;
                let inner = Arc::new(Inner {
                    domain: task_domain.clone(),
                    endpoint,
                    settings,
                    auth,
                    capabilities,
                    pools: Mutex::new(Vec::new()),
                    conns: Mutex::new(HashMap::new()),
                    routes: Mutex::new(Routes::default()),
                });
                task_domain.spawn(accept_connections(Arc::downgrade(&inner)));
                Ok(QuicTransport { inner })
            })
            .await
            .map_err(|error| link(format!("the net domain stopped: {error}")))?
    }

    /// The endpoint's key, which peers dial.
    pub fn node_id(&self) -> [u8; 32] {
        self.inner.endpoint.node_id()
    }

    /// The endpoint's address as iroh knows it: its direct addresses and its relay.
    pub fn addr(&self) -> NodeAddr {
        self.inner.endpoint.addr()
    }

    /// The local sockets of the endpoint's direct paths.
    pub fn bound_addrs(&self) -> Vec<SocketAddr> {
        self.inner.endpoint.bound_addrs()
    }

    /// The kind of the selected path of the latency connection to `peer`: `Quic` when direct,
    /// `QuicRelay` through a relay (RFC-0031 §4.1). `None` with no connection or no path.
    pub fn path_kind(&self, peer: &[u8; 32]) -> Option<TransportKind> {
        let conn = self.inner.existing(peer, Role::Latency)?;
        let paths = conn.conn.paths();
        let selected = paths.iter().find(|path| path.is_selected())?;
        Some(TransportKind::from(if selected.is_relay() {
            PathKind::Relay
        } else {
            PathKind::Direct
        }))
    }

    /// Each connection's receive window and what set it (RFC-0031 §4.3).
    pub fn connection_windows(&self) -> Vec<WindowReport> {
        let conns: Vec<Arc<Conn>> = lock(&self.inner.conns)
            .values()
            .filter_map(|cell| cell.get().cloned())
            .collect();
        conns
            .iter()
            .map(|conn| {
                let window = lock(&conn.window);
                WindowReport {
                    peer: conn.peer,
                    role: conn.role.get().copied().unwrap_or(Role::Latency),
                    channels: window.channels,
                    credited: window.credited,
                    window: window.window,
                    over_cap: window.over_cap,
                }
            })
            .collect()
    }

    /// The congestion window of each path of the latency connection to `peer`, as its
    /// controller reports it: what `quic.lan_initial_window` set, or the default.
    pub fn congestion_windows(&self, peer: &[u8; 32]) -> Vec<u64> {
        let Some(conn) = self.inner.existing(peer, Role::Latency) else {
            return Vec::new();
        };
        let paths = conn.conn.paths();
        paths
            .iter()
            .filter_map(|path| conn.conn.congestion_state(path.id()))
            .map(|controller| controller.window())
            .collect()
    }

    /// Closes the endpoint and every connection on it.
    pub async fn close(&self) {
        let inner = Arc::clone(&self.inner);
        let _ = self
            .inner
            .domain
            .spawn(async move { inner.endpoint.close().await })
            .await;
    }
}

impl Inner {
    fn existing(&self, peer: &[u8; 32], role: Role) -> Option<Arc<Conn>> {
        let cell = lock(&self.conns).get(&(*peer, role)).cloned()?;
        let conn = cell.get()?.clone();
        conn.conn.close_reason().is_none().then_some(conn)
    }

    /// The receive pool and send pools of a channel spec, resolved from the registrations.
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
        let slot_size = spec.slot_size.0;
        if slot_size > self.settings.max_frame {
            return Err(Error::uncoded(
                ErrorClass::InvalidRequest,
                format!(
                    "a {slot_size}-byte slot is larger than net.max_frame, {}",
                    self.settings.max_frame
                ),
            ));
        }
        Ok(StreamSpec {
            key: spec.key,
            pool,
            slot_size: slot_size as u32,
            recv_slots: spec.recv_slots,
            wire: spec.wire,
            retention: spec.retention,
            capabilities: self.capabilities,
            alpha_p99: INITIAL_ALPHA_P99,
            send_pools: pools,
            max_frame: self.settings.max_frame,
        })
    }

    /// The connection of kind `role` to `remote`: one either side already has, else a new one,
    /// dialled once however many channels ask for it at the same time.
    async fn connection(
        self: &Arc<Self>,
        remote: &RemoteEndpoint,
        role: Role,
    ) -> Result<Arc<Conn>, Error> {
        loop {
            let cell = Arc::clone(
                lock(&self.conns)
                    .entry((remote.key, role))
                    .or_insert_with(|| Arc::new(OnceCell::new())),
            );
            let conn = cell
                .get_or_try_init(|| self.dial(remote, role))
                .await?
                .clone();
            if conn.conn.close_reason().is_none() {
                return Ok(conn);
            }
            // A connection that has closed is replaced, unless another caller already did.
            let mut conns = lock(&self.conns);
            if conns
                .get(&(remote.key, role))
                .is_some_and(|current| Arc::ptr_eq(current, &cell))
            {
                conns.remove(&(remote.key, role));
            }
        }
    }

    async fn dial(
        self: &Arc<Self>,
        remote: &RemoteEndpoint,
        role: Role,
    ) -> Result<Arc<Conn>, Error> {
        let class = link_class(remote);
        let mut addr = NodeAddr::new(remote.key)?;
        for direct in &remote.addrs {
            addr = addr.with_direct_addr(*direct);
        }
        if let Some(relay) = &remote.relay {
            addr = addr.with_relay_url(relay.clone());
        }
        let options =
            ConnectOptions::new().with_transport_config(self.settings.transport_config(class));
        let timeout = self.settings.connect_timeout(class);
        let dial = async {
            let connecting = self
                .endpoint
                .inner
                .connect_with_opts(addr.inner, DATA_ALPN, options)
                .await
                .map_err(|error| link(format!("cannot connect: {error}")))?;
            connecting
                .await
                .map_err(|error| link(format!("cannot connect: {error}")))
        };
        let conn = match tokio::time::timeout(timeout, dial).await {
            Ok(result) => result?,
            Err(_) => {
                return Err(Error::new(
                    ErrorCode::CONNECT_TIMEOUT,
                    format!("no connection within {timeout:?}"),
                ));
            }
        };
        let conn = self.adopt(conn, Some(role), remote.key);
        // The peer may open its own channels to this side on the connection.
        tokio::spawn(accept_streams(Arc::downgrade(self), Arc::clone(&conn)));
        Ok(conn)
    }

    /// A connection this side holds, of kind `role` if it is known yet.
    fn adopt(&self, conn: Connection, role: Option<Role>, peer: [u8; 32]) -> Arc<Conn> {
        let conn = Arc::new(Conn {
            conn,
            role: OnceLock::new(),
            peer,
            window: Mutex::new(Window::default()),
        });
        if let Some(role) = role {
            let _ = conn.role.set(role);
        }
        set_window(&conn, &self.settings, 0);
        conn
    }

    /// Adds a channel's credited bytes to its connection and sets the connection receive window
    /// to the new sum (RFC-0031 §4.3).
    fn credit_channel(&self, conn: &Conn, spec: &StreamSpec) {
        let credited = u64::from(spec.recv_slots) * u64::from(spec.slot_size)
            + u64::from(CONTROL_RECEIVES) * MAX_CONTROL_FRAME as u64;
        set_window(conn, &self.settings, credited);
    }

    /// Hands an accepted stream to the `accept` waiting for its key, or holds it for one.
    async fn route(self: Arc<Self>, conn: Arc<Conn>, send: SendStream, recv: RecvStream) {
        let mut recv = recv;
        let timeout = self.settings.connect_timeout(LinkClass::Wan);
        let hold = self.settings.early_hold();
        let hello = match tokio::time::timeout(timeout, stream::read_hello(&mut recv)).await {
            Ok(Ok(hello)) => hello,
            // A stream that never says `Hello`, or says something else, is dropped.
            _ => return,
        };
        let Some(key) = hello.key() else {
            return;
        };
        // The first channel on a connection the peer dialled shows which kind it is, so this
        // side's own channels to the peer can reuse it.
        self.remember(&conn, Role::of(key.purpose));
        let early = Early {
            conn,
            send,
            recv,
            hello,
        };
        /// What to do with the stream, decided under the routing lock and done after it.
        enum Next {
            Complete(Waiter, Early),
            Hold,
            Refuse(Early),
        }
        let next = {
            let mut routes = lock(&self.routes);
            match routes.waiters.remove(&key) {
                Some(waiter) => Next::Complete(waiter, early),
                None if routes.early.len() < MAX_EARLY && !routes.early.contains_key(&key) => {
                    routes.early.insert(key, early);
                    Next::Hold
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
            Next::Hold => {
                // Refuse the early stream if nobody accepts its channel in time: well within the
                // connector's own wait, so it gets the refusal, not a timeout.
                tokio::time::sleep(hold).await;
                lock(&self.routes).early.remove(&key)
            }
        };
        if let Some(Early { mut send, .. }) = refuse {
            let _ = stream::answer(
                &mut send,
                &placeholder_spec(&key),
                Err(ErrorCode::NOT_A_NEIGHBOUR),
                self.auth.as_ref(),
            )
            .await;
        }
    }

    /// Records a connection the peer dialled as this side's connection of kind `role` to it, if
    /// this side has none. Its first channel decides its kind.
    fn remember(&self, conn: &Arc<Conn>, role: Role) {
        if *conn.role.get_or_init(|| role) != role {
            return;
        }
        let mut conns = lock(&self.conns);
        let cell = conns
            .entry((conn.peer, role))
            .or_insert_with(|| Arc::new(OnceCell::new()));
        if cell.get().is_none() {
            let _ = cell.set(Arc::clone(conn));
        }
    }

    /// The acceptor's end of the handshake on `early`'s stream, then the channel.
    async fn complete(&self, early: Early, spec: StreamSpec) -> Result<Channel, Error> {
        let Early {
            conn,
            mut send,
            recv,
            hello,
        } = early;
        let peer = conn.peer;
        let agreed = stream::check_hello(&hello, &spec, &peer, self.auth.as_ref());
        self.credit_channel(&conn, &spec);
        stream::answer(
            &mut send,
            &spec,
            agreed.as_ref().map_err(|code| *code),
            self.auth.as_ref(),
        )
        .await?;
        let agreed = agreed
            .map_err(|code| Error::new(code, format!("refused the channel {:?}", spec.key)))?;
        let opened = stream::open(recv, send, spec, agreed, &self.domain)?;
        Ok(opened.spawn(&self.domain))
    }
}

/// Sets a connection's receive window after `added` more credited bytes.
fn set_window(conn: &Conn, settings: &QuicSettings, added: u64) {
    let mut window = lock(&conn.window);
    if added > 0 {
        window.channels += 1;
        window.credited = window.credited.saturating_add(added);
    }
    let set = settings.connection_window(window.credited);
    window.window = set.window;
    window.over_cap = set.over_cap;
    conn.conn.set_receive_window(varint(set.window));
}

/// The spec a refusal is written with: only its key is read.
fn placeholder_spec(key: &ChannelKey) -> StreamSpec {
    StreamSpec {
        key: *key,
        pool: BufferPool::heap(crate::transport::PoolId(0), 64, 1)
            .expect("a one-slot pool of 64 bytes"),
        slot_size: 0,
        recv_slots: 0,
        wire: syntropy_core::plan::WireDType::F32,
        retention: 0,
        capabilities: 0,
        alpha_p99: INITIAL_ALPHA_P99,
        send_pools: Vec::new(),
        max_frame: 0,
    }
}

/// Accepts the endpoint's incoming connections for as long as the transport lives.
async fn accept_connections(inner: Weak<Inner>) {
    loop {
        let Some(strong) = inner.upgrade() else {
            return;
        };
        let endpoint = strong.endpoint.inner.clone();
        drop(strong);
        let Some(incoming) = endpoint.accept().await else {
            return;
        };
        let inner = inner.clone();
        tokio::spawn(async move {
            let Ok(conn) = incoming.await else {
                return;
            };
            // The endpoint is the transport's own, but it accepts what its configuration names;
            // only data connections carry channels.
            if conn.alpn() != DATA_ALPN {
                return;
            }
            let Some(strong) = inner.upgrade() else {
                return;
            };
            let peer = *conn.remote_id().as_bytes();
            // Its kind is learnt from its first channel.
            let conn = strong.adopt(conn, None, peer);
            drop(strong);
            accept_streams(inner, conn).await;
        });
    }
}

/// Accepts the streams the peer opens on `conn`, and routes each by its `Hello`.
async fn accept_streams(inner: Weak<Inner>, conn: Arc<Conn>) {
    loop {
        let Ok((send, recv)) = conn.conn.accept_bi().await else {
            return;
        };
        let Some(strong) = inner.upgrade() else {
            return;
        };
        tokio::spawn(strong.route(Arc::clone(&conn), send, recv));
    }
}

impl Transport for QuicTransport {
    fn kind(&self) -> TransportKind {
        TransportKind::Quic
    }

    fn caps(&self) -> TransportCaps {
        TransportCaps {
            zero_copy_recv: false,
            device_memory_recv: false,
            max_frame: syntropy_core::Bytes(self.inner.settings.max_frame),
            needs_registration: false,
            encrypted: true,
            reliable: true,
            max_channels_per_peer: u16::MAX,
            inline_max: 0,
        }
    }

    fn register(&self, pool: &BufferPool) -> Result<PoolRegistration, Error> {
        let mut pools = lock(&self.inner.pools);
        if pools.iter().any(|known| known.id() == pool.id()) {
            return Err(Error::uncoded(
                ErrorClass::InvalidRequest,
                format!("pool {:?} is already registered", pool.id()),
            ));
        }
        pools.push(pool.clone());
        Ok(PoolRegistration {
            pool: pool.id(),
            transport: TransportKind::Quic,
            handle: 0,
        })
    }

    async fn connect(&self, remote: &RemoteEndpoint, spec: ChannelSpec) -> Result<Channel, Error> {
        let inner = Arc::clone(&self.inner);
        let remote = remote.clone();
        self.inner
            .domain
            .spawn(async move {
                let spec_in = inner.stream_spec(&spec)?;
                let role = Role::of(spec.key.purpose);
                let conn = inner.connection(&remote, role).await?;
                let (mut send, mut recv) = conn
                    .conn
                    .open_bi()
                    .await
                    .map_err(|error| link(format!("cannot open a stream: {error}")))?;
                let _ = send.set_priority(stream_priority(&spec.key, spec.priority));
                inner.credit_channel(&conn, &spec_in);
                let class = link_class(&remote);
                let timeout = inner.settings.connect_timeout(class);
                let offer = stream::offer(
                    &mut recv,
                    &mut send,
                    &spec_in,
                    &remote.key,
                    inner.auth.as_ref(),
                );
                let agreed = match tokio::time::timeout(timeout, offer).await {
                    Ok(agreed) => agreed?,
                    Err(_) => {
                        return Err(Error::new(
                            ErrorCode::CONNECT_TIMEOUT,
                            format!("no HelloAck within {timeout:?}"),
                        ));
                    }
                };
                let opened = stream::open(recv, send, spec_in, agreed, &inner.domain)?;
                Ok(opened.spawn(&inner.domain))
            })
            .await
            .map_err(|error| link(format!("the net domain stopped: {error}")))?
    }

    async fn accept(&self, spec: ChannelSpec) -> Result<Channel, Error> {
        let inner = Arc::clone(&self.inner);
        self.inner
            .domain
            .spawn(async move {
                let spec_in = inner.stream_spec(&spec)?;
                let (reply, answer) = oneshot::channel();
                let early = {
                    let mut routes = lock(&inner.routes);
                    if routes.waiters.contains_key(&spec.key) {
                        return Err(Error::uncoded(
                            ErrorClass::InvalidRequest,
                            format!("the channel {:?} is already being accepted", spec.key),
                        ));
                    }
                    match routes.early.remove(&spec.key) {
                        Some(early) => Some(early),
                        None => {
                            routes.waiters.insert(
                                spec.key,
                                Waiter {
                                    spec: spec_in.clone(),
                                    reply,
                                },
                            );
                            None
                        }
                    }
                };
                match early {
                    Some(early) => inner.complete(early, spec_in).await,
                    None => answer
                        .await
                        .map_err(|_| link("the transport closed before the channel opened"))?,
                }
            })
            .await
            .map_err(|error| link(format!("the net domain stopped: {error}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_and_bulk_take_the_bulk_connection() {
        for purpose in ChannelPurpose::ALL {
            let expected = match purpose {
                ChannelPurpose::Replay | ChannelPurpose::Bulk => Role::Bulk,
                _ => Role::Latency,
            };
            assert_eq!(Role::of(purpose), expected, "{purpose:?}");
        }
    }

    #[test]
    fn the_decode_lane_outranks_the_prefill_lane_and_data_outranks_the_rest() {
        let key = |purpose, lane| ChannelKey {
            instance: syntropy_core::InstanceId(1),
            epoch: syntropy_core::PlanEpoch { term: 1, seq: 1 },
            purpose,
            from: syntropy_core::StageIx(0),
            to: syntropy_core::StageIx(1),
            lane,
        };
        let decode = stream_priority(&key(ChannelPurpose::Activations, 0), ChannelPriority::Data);
        let prefill = stream_priority(&key(ChannelPurpose::Activations, 1), ChannelPriority::Data);
        let collective = stream_priority(
            &key(ChannelPurpose::Collective, 0),
            ChannelPriority::Collective,
        );
        let replay = stream_priority(&key(ChannelPurpose::Replay, 0), ChannelPriority::Replay);
        let bulk = stream_priority(&key(ChannelPurpose::Bulk, 0), ChannelPriority::Bulk);
        assert!(decode > prefill && prefill > collective && collective > replay);
        assert!(replay > bulk);
    }

    #[test]
    fn a_peer_with_only_local_addresses_is_on_the_lan() {
        let remote = |addrs: &[&str]| RemoteEndpoint {
            node: syntropy_core::NodeId([1; 32]),
            key: [1; 32],
            addrs: addrs
                .iter()
                .map(|a| a.parse().expect("an address"))
                .collect(),
            relay: None,
        };
        assert_eq!(link_class(&remote(&["192.168.1.4:7600"])), LinkClass::Lan);
        assert_eq!(
            link_class(&remote(&["10.0.0.2:7600", "[fe80::1]:7600", "127.0.0.1:1"])),
            LinkClass::Lan
        );
        assert_eq!(link_class(&remote(&["[fd00::7]:7600"])), LinkClass::Lan);
        assert_eq!(
            link_class(&remote(&["192.168.1.4:7600", "203.0.113.9:7600"])),
            LinkClass::Wan
        );
        // A relay-only peer is not known to be on the LAN.
        assert_eq!(link_class(&remote(&[])), LinkClass::Wan);
    }
}
