// SPDX-License-Identifier: MIT OR Apache-2.0

//! A minimal iroh endpoint (DIST-E7-T23, ADR-0011): bind with a node key the caller supplies,
//! connect to a node ID with an application protocol, and report the kind and round-trip time of
//! the path a connection uses.
//!
//! This is what the M1a field probe (DIST-E4-T15) needs, on the iroh version the daemon will use.
//! DIST-E7-T03's QUIC backend builds on the same pin. `net` is the only crate that links iroh
//! (RFC-0001 §4.2 rule 6), so no iroh type appears in this module's public API. Nothing here
//! creates a runtime: every `async` function runs in the caller's tokio context.
//!
//! Relays are one of three choices ([`Relays`]):
//!
//! - [`Relays::Public`]: iroh's public relays, with iroh's online address lookup, so a bare node
//!   ID resolves. These are for internal and alpha builds only, until DIST-E7-T22's built-in set
//!   ships: no public release points at them (RFC-0030 §6.4).
//! - [`Relays::Url`]: one relay at a given [`RelayUrl`], with no online lookup.
//! - [`Relays::Disabled`]: no relay.
//!
//! [`Config::direct_paths`] set to `false` removes the endpoint's direct (IP) transports, so every
//! connection goes through a relay: the forced relay the probe reports as relayed.
//!
//! A [`Connection`] also carries unreliable datagrams (RFC 9221), the way RFC-0030 §5.2's
//! heartbeats travel; the field probe times its echoes over them.
//!
//! With the `test-relay` feature, [`test_relay`] runs a relay server on this host for tests of the
//! crates above `net`, which rule 6 keeps from naming iroh. It is a test-only feature (RFC-0001
//! §4.2 rule 10): no release turns it on.

use std::fmt;
use std::net::SocketAddr;
use std::str::FromStr;
use std::time::Duration;

use iroh::endpoint::{Builder, SendDatagramError, presets};
use iroh::{EndpointAddr, PublicKey, RelayMode, SecretKey, TransportAddr};

/// The kind of path a connection's packets take (RFC-0030 §5, `PathKind`): `Direct` or `Relay`,
/// as iroh reports the selected path. DIST-E4-T01's membership core and DIST-E7-T03's backend use
/// this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PathKind {
    /// A direct IP path between the two endpoints.
    Direct,
    /// A path through a relay server.
    Relay,
}

/// The URL of a relay server, such as `https://relay.example.net`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RelayUrl(iroh::RelayUrl);

impl FromStr for RelayUrl {
    type Err = EndpointError;

    fn from_str(url: &str) -> Result<Self, Self::Err> {
        iroh::RelayUrl::from_str(url)
            .map(Self)
            .map_err(|error| EndpointError::InvalidRelayUrl(error.to_string()))
    }
}

impl fmt::Display for RelayUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Which relays an endpoint uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Relays {
    /// iroh's public relays, with iroh's online address lookup. Internal and alpha builds only,
    /// until DIST-E7-T22's built-in relay set ships (RFC-0030 §6.4). Only this documentation limits
    /// it now; DIST-E7-T22's relay set replaces it, and that task's release check should cover it.
    Public,
    /// One relay, with no online address lookup.
    Url(RelayUrl),
    /// No relay: direct paths only.
    Disabled,
}

/// How to bind an [`Endpoint`]. Start from [`Config::default`], which has no relay, direct paths
/// on and no accepted protocol, so it connects but accepts nothing, and set fields with the
/// builder methods. New fields come with defaults, so a caller is unaffected when one is added.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// The relays the endpoint uses.
    pub relays: Relays,
    /// Whether the endpoint has direct (IP) transports. With `false`, every connection goes
    /// through a relay.
    pub direct_paths: bool,
    /// The application protocols (ALPNs) the endpoint accepts on incoming connections, preferred
    /// first. Outgoing connections name theirs in [`Endpoint::connect`].
    pub accept_alpns: Vec<Vec<u8>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            relays: Relays::Disabled,
            direct_paths: true,
            accept_alpns: Vec::new(),
        }
    }
}

impl Config {
    /// Sets the relays.
    pub fn relays(mut self, relays: Relays) -> Self {
        self.relays = relays;
        self
    }

    /// Sets whether the endpoint has direct (IP) transports.
    pub fn direct_paths(mut self, direct_paths: bool) -> Self {
        self.direct_paths = direct_paths;
        self
    }

    /// Adds one accepted application protocol, after any already set.
    pub fn accept_alpn(mut self, alpn: impl Into<Vec<u8>>) -> Self {
        self.accept_alpns.push(alpn.into());
        self
    }

    /// Replaces the accepted application protocols.
    pub fn accept_alpns(mut self, alpns: Vec<Vec<u8>>) -> Self {
        self.accept_alpns = alpns;
        self
    }
}

/// Why an endpoint operation failed. The message comes from iroh and names no secret.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EndpointError {
    /// The 32 bytes are not a valid public key.
    InvalidNodeId,
    /// The relay URL does not parse.
    InvalidRelayUrl(String),
    /// [`Relays::Disabled`] with `direct_paths: false` leaves the endpoint no transport.
    NoTransport,
    /// Binding the endpoint failed.
    Bind(String),
    /// Connecting to a peer failed.
    Connect(String),
    /// Accepting a connection failed.
    Accept(String),
    /// The endpoint is closed.
    Closed,
    /// A datagram cannot be sent: it is larger than the path takes, or the peer or this endpoint
    /// does not accept datagrams.
    Datagram(String),
}

impl fmt::Display for EndpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNodeId => write!(f, "not a valid node ID"),
            Self::InvalidRelayUrl(error) => write!(f, "invalid relay URL: {error}"),
            Self::NoTransport => {
                write!(
                    f,
                    "no relay and no direct paths: the endpoint has no transport"
                )
            }
            Self::Bind(error) => write!(f, "cannot bind the endpoint: {error}"),
            Self::Connect(error) => write!(f, "cannot connect: {error}"),
            Self::Accept(error) => write!(f, "cannot accept a connection: {error}"),
            Self::Closed => write!(f, "the endpoint is closed"),
            Self::Datagram(error) => write!(f, "cannot send the datagram: {error}"),
        }
    }
}

impl std::error::Error for EndpointError {}

/// Where to reach a node: its ID, and optionally relay URLs and direct socket addresses. With
/// neither, only [`Relays::Public`]'s online lookup can find the node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeAddr {
    inner: EndpointAddr,
}

impl NodeAddr {
    /// A node by its ID, the 32 bytes of its public key.
    pub fn new(node_id: [u8; 32]) -> Result<Self, EndpointError> {
        let key = PublicKey::from_bytes(&node_id).map_err(|_| EndpointError::InvalidNodeId)?;
        Ok(Self {
            inner: EndpointAddr::new(key),
        })
    }

    /// This address with a relay the node is reachable through.
    pub fn with_relay_url(self, url: RelayUrl) -> Self {
        Self {
            inner: self.inner.with_relay_url(url.0),
        }
    }

    /// This address with a direct socket address of the node.
    pub fn with_direct_addr(self, addr: SocketAddr) -> Self {
        Self {
            inner: self.inner.with_ip_addr(addr),
        }
    }

    /// The node's ID.
    pub fn node_id(&self) -> [u8; 32] {
        *self.inner.id.as_bytes()
    }

    /// The direct socket addresses in this address.
    pub fn direct_addrs(&self) -> Vec<SocketAddr> {
        self.inner.ip_addrs().copied().collect()
    }

    /// The relay URLs in this address.
    pub fn relay_urls(&self) -> Vec<RelayUrl> {
        self.inner.relay_urls().cloned().map(RelayUrl).collect()
    }
}

/// A bound iroh endpoint. Its `Debug` shows the node ID only: iroh's own carries the host's
/// addresses, which must stay out of logs and of the field probe's report (DIST-E4-T15). A clone
/// is another handle to the same endpoint.
#[derive(Clone)]
pub struct Endpoint {
    inner: iroh::Endpoint,
}

/// A node ID as lowercase hex, for `Debug`.
struct NodeIdHex([u8; 32]);

impl fmt::Debug for NodeIdHex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

impl fmt::Debug for Endpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Endpoint")
            .field("node_id", &NodeIdHex(self.node_id()))
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Connection")
            .field("remote_node_id", &NodeIdHex(self.remote_node_id()))
            .field("alpn", &String::from_utf8_lossy(&self.alpn()))
            .finish_non_exhaustive()
    }
}

impl Endpoint {
    /// Binds an endpoint whose node ID is the public half of `secret_key`. The caller owns the
    /// key: this module neither generates nor stores one.
    pub async fn bind(secret_key: [u8; 32], config: Config) -> Result<Self, EndpointError> {
        Self::bind_with(secret_key, config, |builder| builder).await
    }

    /// [`bind`](Self::bind), with a last change to the builder. Tests use it to trust a test
    /// relay's self-signed certificate.
    async fn bind_with(
        secret_key: [u8; 32],
        config: Config,
        last: impl FnOnce(Builder) -> Builder,
    ) -> Result<Self, EndpointError> {
        let mut builder = match config.relays {
            Relays::Public => iroh::Endpoint::builder(presets::N0),
            Relays::Url(url) => {
                iroh::Endpoint::builder(presets::Minimal).relay_mode(RelayMode::custom([url.0]))
            }
            Relays::Disabled => {
                if !config.direct_paths {
                    return Err(EndpointError::NoTransport);
                }
                iroh::Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled)
            }
        };
        if !config.direct_paths {
            builder = builder.clear_ip_transports();
        }
        let builder = builder
            .secret_key(SecretKey::from_bytes(&secret_key))
            .alpns(config.accept_alpns);
        let inner = last(builder)
            .bind()
            .await
            .map_err(|error| EndpointError::Bind(error.to_string()))?;
        Ok(Self { inner })
    }

    /// This endpoint's node ID.
    pub fn node_id(&self) -> [u8; 32] {
        *self.inner.id().as_bytes()
    }

    /// This endpoint's address as iroh currently knows it: its node ID, its relay once connected
    /// to one, and the direct addresses it has found.
    pub fn addr(&self) -> NodeAddr {
        NodeAddr {
            inner: self.inner.addr(),
        }
    }

    /// The local sockets the endpoint's direct transports are bound to.
    pub fn bound_addrs(&self) -> Vec<SocketAddr> {
        self.inner.bound_sockets()
    }

    /// Waits until the endpoint is reachable: connected to its relay, if it has one.
    pub async fn online(&self) {
        self.inner.online().await;
    }

    /// Connects to `peer` with the application protocol `alpn`, which the peer must accept.
    pub async fn connect(&self, peer: NodeAddr, alpn: &[u8]) -> Result<Connection, EndpointError> {
        let inner = self
            .inner
            .connect(peer.inner, alpn)
            .await
            .map_err(|error| EndpointError::Connect(error.to_string()))?;
        Ok(Connection { inner })
    }

    /// Accepts the next incoming connection, on any of the endpoint's accepted protocols;
    /// [`Connection::alpn`] says which.
    pub async fn accept(&self) -> Result<Connection, EndpointError> {
        let incoming = self.inner.accept().await.ok_or(EndpointError::Closed)?;
        let inner = incoming
            .await
            .map_err(|error| EndpointError::Accept(error.to_string()))?;
        Ok(Connection { inner })
    }

    /// Closes the endpoint and every connection on it.
    pub async fn close(&self) {
        self.inner.close().await;
    }
}

/// An open connection to a peer. Its `Debug` shows the peer's node ID and the protocol only, for
/// the same reason as [`Endpoint`]'s.
#[derive(Clone)]
pub struct Connection {
    /// The iroh connection, for the transport code in this crate (DIST-E7-T03) to open streams on.
    pub(crate) inner: iroh::endpoint::Connection,
}

impl Connection {
    /// The peer's node ID.
    pub fn remote_node_id(&self) -> [u8; 32] {
        *self.inner.remote_id().as_bytes()
    }

    /// The application protocol the connection speaks.
    pub fn alpn(&self) -> Vec<u8> {
        self.inner.alpn().to_vec()
    }

    /// The kind of the path the connection currently sends on, or `None` if it has no selected
    /// path. It is read at each call, so it follows a move from a relay to a direct path.
    pub fn path_kind(&self) -> Option<PathKind> {
        let paths = self.inner.paths();
        let selected = paths.iter().find(|path| path.is_selected())?;
        Some(if selected.is_relay() {
            PathKind::Relay
        } else {
            PathKind::Direct
        })
    }

    /// A sample of the round-trip time on the selected path: QUIC's current estimate, or `None`
    /// if the connection has no selected path. Read it again for another sample.
    pub fn rtt(&self) -> Option<Duration> {
        let paths = self.inner.paths();
        let selected = paths.iter().find(|path| path.is_selected())?;
        Some(selected.rtt())
    }

    /// The address of the selected path when it is direct, or `None` when the connection is
    /// relayed or has no selected path. The field probe uses it only to find which local interface
    /// the path leaves from, and never records it.
    pub fn remote_direct_addr(&self) -> Option<SocketAddr> {
        let paths = self.inner.paths();
        let selected = paths.iter().find(|path| path.is_selected())?;
        match selected.remote_addr() {
            TransportAddr::Ip(addr) => Some(*addr),
            _ => None,
        }
    }

    /// Sends `data` as one unreliable, unordered datagram. It may be lost or reordered, and it
    /// must fit [`max_datagram_size`](Self::max_datagram_size).
    pub fn send_datagram(&self, data: &[u8]) -> Result<(), EndpointError> {
        self.inner
            .send_datagram(data.to_vec().into())
            .map_err(|error| match error {
                SendDatagramError::ConnectionLost(_) => EndpointError::Closed,
                other => EndpointError::Datagram(other.to_string()),
            })
    }

    /// Waits for the next datagram from the peer. Fails with [`EndpointError::Closed`] once the
    /// connection is closed.
    pub async fn read_datagram(&self) -> Result<Vec<u8>, EndpointError> {
        self.inner
            .read_datagram()
            .await
            .map(|data| data.to_vec())
            .map_err(|_| EndpointError::Closed)
    }

    /// The largest datagram [`send_datagram`](Self::send_datagram) takes now, or `None` if the peer
    /// does not accept datagrams. It follows the path's MTU estimate, and is a little over a
    /// kilobyte at least when datagrams are accepted.
    pub fn max_datagram_size(&self) -> Option<usize> {
        self.inner.max_datagram_size()
    }

    /// Closes the connection.
    pub fn close(&self) {
        self.inner.close(0u8.into(), b"closed");
    }
}

/// A relay server on this host, for tests of the crates above `net` (the `test-relay` feature).
/// Rule 6 keeps those crates from naming iroh, so they reach iroh's test relay through here.
#[cfg(feature = "test-relay")]
pub mod test_relay {
    use iroh::tls::CaTlsConfig;

    use super::{Config, Endpoint, EndpointError, RelayUrl};

    /// A running relay server. It stops when dropped.
    pub struct TestRelay {
        url: RelayUrl,
        _server: Box<dyn std::any::Any + Send + Sync>,
    }

    impl std::fmt::Debug for TestRelay {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TestRelay")
                .field("url", &self.url)
                .finish_non_exhaustive()
        }
    }

    impl TestRelay {
        /// Starts a relay server on a loopback port, with a self-signed certificate.
        pub async fn start() -> Result<Self, EndpointError> {
            let (_map, url, server) = iroh::test_utils::run_relay_server()
                .await
                .map_err(|error| EndpointError::Bind(error.to_string()))?;
            Ok(Self {
                url: RelayUrl(url),
                _server: Box::new(server),
            })
        }

        /// The relay's URL, for [`super::Relays::Url`].
        pub fn url(&self) -> RelayUrl {
            self.url.clone()
        }
    }

    /// [`Endpoint::bind`], trusting any relay certificate, so the endpoint accepts the test relay's
    /// self-signed one. For tests only: it skips the relay's certificate check.
    pub async fn bind_trusting_any_relay(
        secret_key: [u8; 32],
        config: Config,
    ) -> Result<Endpoint, EndpointError> {
        Endpoint::bind_with(secret_key, config, |builder| {
            builder.ca_tls_config(CaTlsConfig::insecure_skip_verify())
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    use iroh::tls::CaTlsConfig;

    use super::*;

    const DATA: &[u8] = b"syntropy/test-data/0";
    const CTL: &[u8] = b"syntropy/test-ctl/0";

    fn key(byte: u8) -> [u8; 32] {
        [byte; 32]
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

    fn hex(bytes: &[u8; 32]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// `shown` names none of `addrs`: neither an IP address nor a `:port`, the form a socket
    /// address takes (a bare number could match digits inside a node ID's hex).
    fn assert_no_addresses(shown: &str, addrs: &[SocketAddr]) {
        for addr in addrs {
            assert!(
                !shown.contains(&addr.ip().to_string()),
                "{shown} shows {}",
                addr.ip()
            );
            assert!(
                !shown.contains(&format!(":{}", addr.port())),
                "{shown} shows port {}",
                addr.port()
            );
        }
    }

    /// The server's address as a client on this host dials it directly.
    fn direct_addr(server: &Endpoint) -> NodeAddr {
        let mut peer = NodeAddr::new(server.node_id()).unwrap();
        for addr in server.bound_addrs() {
            peer = peer.with_direct_addr(local(addr));
        }
        assert!(!peer.direct_addrs().is_empty());
        peer
    }

    /// Each accepted connection's peer node ID and protocol, once the server task ends.
    type Served = tokio::task::JoinHandle<Result<Vec<([u8; 32], Vec<u8>)>, EndpointError>>;

    /// Accepts `count` connections, holds each until its client closes it, and returns each
    /// one's peer and protocol.
    fn serve(server: Endpoint, count: usize) -> Served {
        tokio::spawn(async move {
            let mut seen = Vec::new();
            for _ in 0..count {
                let conn = server.accept().await?;
                seen.push((conn.remote_node_id(), conn.alpn()));
                conn.inner.closed().await;
            }
            server.close().await;
            Ok(seen)
        })
    }

    #[tokio::test]
    async fn two_endpoints_on_one_host_connect_by_node_id_on_a_direct_path() {
        // No relay and no online lookup: the test hands the client the server's direct address,
        // so it needs no network beyond loopback.
        let server = Endpoint::bind(key(1), Config::default().accept_alpn(DATA))
            .await
            .unwrap();
        let peer = direct_addr(&server);
        let peer_addrs = peer.direct_addrs();
        let server_id = server.node_id();
        let served = serve(server, 1);

        let client = Endpoint::bind(key(2), Config::default()).await.unwrap();
        let conn = tokio::time::timeout(Duration::from_secs(30), client.connect(peer, DATA))
            .await
            .expect("connects within 30 s")
            .unwrap();
        assert_eq!(conn.remote_node_id(), server_id);
        assert_eq!(conn.alpn(), DATA);
        assert_eq!(conn.path_kind(), Some(PathKind::Direct));
        for shown in [format!("{client:?}"), format!("{conn:?}")] {
            assert!(shown.contains(&hex(&server_id)) || shown.contains(&hex(&client.node_id())));
            assert_no_addresses(&shown, &peer_addrs);
        }
        let samples: Vec<Duration> = (0..3).filter_map(|_| conn.rtt()).collect();
        assert_eq!(samples.len(), 3, "an RTT sample per read");
        assert!(samples.iter().all(|rtt| *rtt > Duration::ZERO));

        conn.close();
        client.close().await;
        assert_eq!(
            served.await.unwrap().unwrap(),
            [(client.node_id(), DATA.to_vec())]
        );
    }

    /// Accepts one connection and sends back every datagram it reads, until the client closes.
    fn echo(server: Endpoint) -> tokio::task::JoinHandle<usize> {
        tokio::spawn(async move {
            let conn = server.accept().await.unwrap();
            let mut echoed = 0;
            while let Ok(data) = conn.read_datagram().await {
                if conn.send_datagram(&data).is_ok() {
                    echoed += 1;
                }
            }
            server.close().await;
            echoed
        })
    }

    /// Sends `payload` until its echo comes back, and returns the echo. Datagrams may be lost,
    /// so each try waits 1 s; ten tries bound the test.
    async fn echoed(conn: &Connection, payload: &[u8]) -> Vec<u8> {
        for _ in 0..10 {
            conn.send_datagram(payload).unwrap();
            if let Ok(back) =
                tokio::time::timeout(Duration::from_secs(1), conn.read_datagram()).await
            {
                return back.unwrap();
            }
        }
        panic!("no echo in ten tries");
    }

    #[tokio::test]
    async fn datagrams_echo_on_a_direct_path() {
        let server = Endpoint::bind(key(12), Config::default().accept_alpn(DATA))
            .await
            .unwrap();
        let peer = direct_addr(&server);
        let peer_addrs = peer.direct_addrs();
        let echoing = echo(server);
        let client = Endpoint::bind(key(13), Config::default()).await.unwrap();
        let conn = client.connect(peer, DATA).await.unwrap();

        // The selected path's address is one the client was given, and datagrams fit a heartbeat
        // body of RFC-0030 §5.2 (1,150 bytes).
        let addr = conn
            .remote_direct_addr()
            .expect("a direct path has an address");
        assert!(
            peer_addrs.contains(&addr),
            "{addr} is not one of {peer_addrs:?}"
        );
        let max = conn
            .max_datagram_size()
            .expect("the peer accepts datagrams");
        assert!(max >= 1_150, "max datagram size {max}");
        for payload in [b"beat 1".to_vec(), vec![7; 1_150]] {
            assert_eq!(echoed(&conn, &payload).await, payload);
        }
        // Larger than any QUIC packet. The size limit itself moves with the path MTU estimate, so
        // one byte over a size read earlier may fit by the time it is sent.
        let refused = conn.send_datagram(&vec![0; 64 * 1024]);
        assert!(
            matches!(refused, Err(EndpointError::Datagram(_))),
            "{refused:?}"
        );

        conn.close();
        client.close().await;
        assert!(echoing.await.unwrap() >= 2);
        assert_eq!(conn.send_datagram(b"late"), Err(EndpointError::Closed));
    }

    #[tokio::test]
    async fn one_endpoint_accepts_each_protocol_and_reports_which() {
        // Two accepted protocols on one server; a client connects with each in turn.
        let server = Endpoint::bind(key(7), Config::default().accept_alpn(CTL).accept_alpn(DATA))
            .await
            .unwrap();
        let peer = direct_addr(&server);
        let served = serve(server, 2);

        let client = Endpoint::bind(key(8), Config::default()).await.unwrap();
        for alpn in [CTL, DATA] {
            let conn = client.connect(peer.clone(), alpn).await.unwrap();
            assert_eq!(conn.alpn(), alpn);
            conn.close();
        }
        client.close().await;
        let seen: Vec<Vec<u8>> = served
            .await
            .unwrap()
            .unwrap()
            .into_iter()
            .map(|(_, a)| a)
            .collect();
        assert_eq!(seen, [CTL.to_vec(), DATA.to_vec()]);
    }

    #[tokio::test]
    async fn a_protocol_the_peer_does_not_accept_fails_to_connect() {
        // The server accepts only DATA. Its accept loop runs, so the handshake reaches it and the
        // protocol mismatch is refused on both sides.
        let server = Endpoint::bind(key(9), Config::default().accept_alpn(DATA))
            .await
            .unwrap();
        let peer = direct_addr(&server);
        let accepting = tokio::spawn(async move {
            let result = server.accept().await.map(|conn| conn.alpn());
            server.close().await;
            result
        });
        let client = Endpoint::bind(key(10), Config::default()).await.unwrap();
        let result = tokio::time::timeout(Duration::from_secs(30), client.connect(peer, CTL))
            .await
            .expect("fails within 30 s");
        assert!(
            matches!(result, Err(EndpointError::Connect(_))),
            "{result:?}"
        );
        let accepted = tokio::time::timeout(Duration::from_secs(30), accepting)
            .await
            .expect("the server's accept ends within 30 s")
            .unwrap();
        assert!(
            matches!(accepted, Err(EndpointError::Accept(_))),
            "{accepted:?}"
        );
        client.close().await;
    }

    #[tokio::test]
    async fn a_forced_relay_is_reported_as_relayed() {
        // A relay server on this host, with direct paths disabled on both endpoints, so the only
        // path is through the relay. The test relay's certificate is self-signed.
        let (_map, relay_url, _relay) = iroh::test_utils::run_relay_server().await.unwrap();
        let url: RelayUrl = relay_url.to_string().parse().unwrap();
        let trust = |builder: Builder| builder.ca_tls_config(CaTlsConfig::insecure_skip_verify());
        let relayed = Config::default()
            .relays(Relays::Url(url.clone()))
            .direct_paths(false);

        let server = Endpoint::bind_with(key(3), relayed.clone().accept_alpn(DATA), trust)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(30), server.online())
            .await
            .expect("the server reaches the relay within 30 s");
        assert!(
            server.bound_addrs().is_empty(),
            "no direct transport is bound"
        );
        let peer = NodeAddr::new(server.node_id()).unwrap().with_relay_url(url);
        assert!(peer.direct_addrs().is_empty());
        let served = serve(server, 1);

        let client = Endpoint::bind_with(key(4), relayed, trust).await.unwrap();
        let conn = tokio::time::timeout(Duration::from_secs(30), client.connect(peer, DATA))
            .await
            .expect("connects through the relay within 30 s")
            .unwrap();
        assert_eq!(conn.path_kind(), Some(PathKind::Relay));
        assert!(conn.rtt().is_some_and(|rtt| rtt > Duration::ZERO));
        assert_eq!(
            conn.remote_direct_addr(),
            None,
            "a relayed path has no direct address"
        );

        conn.close();
        client.close().await;
        served.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn datagrams_echo_through_a_relay() {
        let (_map, relay_url, _relay) = iroh::test_utils::run_relay_server().await.unwrap();
        let url: RelayUrl = relay_url.to_string().parse().unwrap();
        let trust = |builder: Builder| builder.ca_tls_config(CaTlsConfig::insecure_skip_verify());
        let relayed = Config::default()
            .relays(Relays::Url(url.clone()))
            .direct_paths(false);
        let server = Endpoint::bind_with(key(14), relayed.clone().accept_alpn(DATA), trust)
            .await
            .unwrap();
        server.online().await;
        let peer = NodeAddr::new(server.node_id()).unwrap().with_relay_url(url);
        let echoing = echo(server);
        let client = Endpoint::bind_with(key(15), relayed, trust).await.unwrap();
        let conn = client.connect(peer, DATA).await.unwrap();
        assert_eq!(conn.path_kind(), Some(PathKind::Relay));
        assert_eq!(echoed(&conn, b"beat").await, b"beat");
        conn.close();
        client.close().await;
        assert!(echoing.await.unwrap() >= 1);
    }

    #[tokio::test]
    async fn no_relay_and_no_direct_paths_is_refused() {
        let error = Endpoint::bind(key(5), Config::default().direct_paths(false))
            .await
            .unwrap_err();
        assert_eq!(error, EndpointError::NoTransport);
    }

    #[tokio::test]
    async fn an_endpoints_debug_shows_its_node_id_and_no_address() {
        let endpoint = Endpoint::bind(key(11), Config::default()).await.unwrap();
        let bound: Vec<SocketAddr> = endpoint.bound_addrs();
        assert!(!bound.is_empty());
        let shown = format!("{endpoint:?}");
        assert_eq!(
            shown,
            format!("Endpoint {{ node_id: {}, .. }}", hex(&endpoint.node_id()))
        );
        assert_no_addresses(&shown, &bound);
        endpoint.close().await;
    }

    #[test]
    fn the_default_config_connects_directly_and_accepts_nothing() {
        let config = Config::default();
        assert_eq!(config.relays, Relays::Disabled);
        assert!(config.direct_paths);
        assert!(config.accept_alpns.is_empty());
        let config = config.accept_alpn(DATA).accept_alpns(vec![CTL.to_vec()]);
        assert_eq!(config.accept_alpns, [CTL.to_vec()]);
    }

    #[test]
    fn addresses_and_relay_urls_validate_their_parts() {
        let node = SecretKey::from_bytes(&key(6)).public();
        let url: RelayUrl = "https://relay.example.net".parse().unwrap();
        let addr = NodeAddr::new(*node.as_bytes())
            .unwrap()
            .with_relay_url(url.clone())
            .with_direct_addr("192.0.2.1:7777".parse().unwrap());
        assert_eq!(addr.node_id(), *node.as_bytes());
        assert_eq!(
            addr.direct_addrs(),
            ["192.0.2.1:7777".parse::<SocketAddr>().unwrap()]
        );
        assert_eq!(addr.relay_urls(), std::slice::from_ref(&url));
        assert_eq!(url.to_string().parse::<RelayUrl>().unwrap(), url);
        assert!(matches!(
            "not a url".parse::<RelayUrl>(),
            Err(EndpointError::InvalidRelayUrl(_))
        ));
    }
}
