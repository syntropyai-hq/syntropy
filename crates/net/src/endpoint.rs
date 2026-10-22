// SPDX-License-Identifier: MIT OR Apache-2.0

//! A minimal iroh endpoint (DIST-E7-T23, ADR-0011): bind with a node key the caller supplies,
//! connect to a node ID, and report the kind and round-trip time of the path a connection uses.
//!
//! This is what the M1a field probe (DIST-E4-T15) needs, on the iroh version the daemon will use.
//! DIST-E7-T03's QUIC backend builds on the same pin. `net` is the only crate that links iroh
//! (RFC-0001 §4.2 rule 6), so no iroh type appears in this module's public API.
//!
//! Relays are one of three choices ([`Relays`]):
//!
//! - [`Relays::Public`]: iroh's public relays, with iroh's online address lookup, so a bare node
//!   ID resolves. These are for internal and alpha builds only, until DIST-E7-T22's built-in set
//!   ships: no public release points at them (RFC-0030 §6.4).
//! - [`Relays::Url`]: one relay at a given URL, with no online lookup.
//! - [`Relays::Disabled`]: no relay.
//!
//! [`Config::direct_paths`] set to `false` removes the endpoint's direct (IP) transports, so every
//! connection goes through a relay: the forced relay the probe reports as relayed.

use std::fmt;
use std::net::SocketAddr;
use std::str::FromStr;
use std::time::Duration;

use iroh::endpoint::{Builder, presets};
use iroh::{EndpointAddr, PublicKey, RelayMode, RelayUrl, SecretKey};

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

/// Which relays an endpoint uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Relays {
    /// iroh's public relays, with iroh's online address lookup. Internal and alpha builds only,
    /// until DIST-E7-T22's built-in relay set ships (RFC-0030 §6.4).
    Public,
    /// One relay at this URL, such as `https://relay.example.net`, with no online address lookup.
    Url(String),
    /// No relay: direct paths only.
    Disabled,
}

/// How to bind an [`Endpoint`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// The relays the endpoint uses.
    pub relays: Relays,
    /// Whether the endpoint has direct (IP) transports. With `false`, every connection goes
    /// through a relay.
    pub direct_paths: bool,
    /// The application protocol both sides speak, offered when connecting and accepted when
    /// listening.
    pub alpn: Vec<u8>,
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
}

impl fmt::Display for EndpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNodeId => write!(f, "not a valid node ID"),
            Self::InvalidRelayUrl(error) => write!(f, "invalid relay URL: {error}"),
            Self::NoTransport => write!(
                f,
                "no relay and no direct paths: the endpoint has no transport"
            ),
            Self::Bind(error) => write!(f, "cannot bind the endpoint: {error}"),
            Self::Connect(error) => write!(f, "cannot connect: {error}"),
            Self::Accept(error) => write!(f, "cannot accept a connection: {error}"),
            Self::Closed => write!(f, "the endpoint is closed"),
        }
    }
}

impl std::error::Error for EndpointError {}

/// Where to reach a node: its ID, and optionally a relay URL and direct socket addresses. With
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
    pub fn with_relay_url(self, url: &str) -> Result<Self, EndpointError> {
        Ok(Self {
            inner: self.inner.with_relay_url(parse_relay_url(url)?),
        })
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
    pub fn relay_urls(&self) -> Vec<String> {
        self.inner.relay_urls().map(ToString::to_string).collect()
    }
}

fn parse_relay_url(url: &str) -> Result<RelayUrl, EndpointError> {
    RelayUrl::from_str(url).map_err(|error| EndpointError::InvalidRelayUrl(error.to_string()))
}

/// A bound iroh endpoint.
#[derive(Debug)]
pub struct Endpoint {
    inner: iroh::Endpoint,
    alpn: Vec<u8>,
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
        let mut builder = match &config.relays {
            Relays::Public => iroh::Endpoint::builder(presets::N0),
            Relays::Url(url) => iroh::Endpoint::builder(presets::Minimal)
                .relay_mode(RelayMode::custom([parse_relay_url(url)?])),
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
            .alpns(vec![config.alpn.clone()]);
        let inner = last(builder)
            .bind()
            .await
            .map_err(|error| EndpointError::Bind(error.to_string()))?;
        Ok(Self {
            inner,
            alpn: config.alpn,
        })
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

    /// Connects to `peer` with this endpoint's protocol.
    pub async fn connect(&self, peer: NodeAddr) -> Result<Connection, EndpointError> {
        let inner = self
            .inner
            .connect(peer.inner, &self.alpn)
            .await
            .map_err(|error| EndpointError::Connect(error.to_string()))?;
        Ok(Connection { inner })
    }

    /// Accepts the next incoming connection.
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

/// An open connection to a peer.
#[derive(Clone, Debug)]
pub struct Connection {
    inner: iroh::endpoint::Connection,
}

impl Connection {
    /// The peer's node ID.
    pub fn remote_node_id(&self) -> [u8; 32] {
        *self.inner.remote_id().as_bytes()
    }

    /// The kind of the path the connection currently sends on, or `None` if it has no selected
    /// path.
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

    /// Closes the connection.
    pub fn close(&self) {
        self.inner.close(0u8.into(), b"closed");
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    use iroh::tls::CaTlsConfig;

    use super::*;

    const ALPN: &[u8] = b"syntropy/test-endpoint/0";

    fn key(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn config(relays: Relays, direct_paths: bool) -> Config {
        Config {
            relays,
            direct_paths,
            alpn: ALPN.to_vec(),
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

    /// Accepts one connection and holds it open until the client closes it.
    fn serve(server: Endpoint) -> tokio::task::JoinHandle<Result<[u8; 32], EndpointError>> {
        tokio::spawn(async move {
            let conn = server.accept().await?;
            let peer = conn.remote_node_id();
            conn.inner.closed().await;
            server.close().await;
            Ok(peer)
        })
    }

    #[tokio::test]
    async fn two_endpoints_on_one_host_connect_by_node_id_on_a_direct_path() {
        // No relay and no online lookup: the test hands the client the server's direct address,
        // so it needs no network beyond loopback.
        let server = Endpoint::bind(key(1), config(Relays::Disabled, true))
            .await
            .unwrap();
        let mut peer = NodeAddr::new(server.node_id()).unwrap();
        for addr in server.bound_addrs() {
            peer = peer.with_direct_addr(local(addr));
        }
        assert!(!peer.direct_addrs().is_empty());
        let server_id = server.node_id();
        let served = serve(server);

        let client = Endpoint::bind(key(2), config(Relays::Disabled, true))
            .await
            .unwrap();
        let conn = tokio::time::timeout(Duration::from_secs(30), client.connect(peer))
            .await
            .expect("connects within 30 s")
            .unwrap();
        assert_eq!(conn.remote_node_id(), server_id);
        assert_eq!(conn.path_kind(), Some(PathKind::Direct));
        let samples: Vec<Duration> = (0..3).filter_map(|_| conn.rtt()).collect();
        assert_eq!(samples.len(), 3, "an RTT sample per read");
        assert!(samples.iter().all(|rtt| *rtt > Duration::ZERO));

        conn.close();
        client.close().await;
        assert_eq!(served.await.unwrap().unwrap(), client.node_id());
    }

    #[tokio::test]
    async fn a_forced_relay_is_reported_as_relayed() {
        // A relay server on this host, with direct paths disabled on both endpoints, so the only
        // path is through the relay. The test relay's certificate is self-signed.
        let (_map, relay_url, _relay) = iroh::test_utils::run_relay_server().await.unwrap();
        let url = relay_url.to_string();
        let trust = |builder: Builder| builder.ca_tls_config(CaTlsConfig::insecure_skip_verify());

        let server = Endpoint::bind_with(key(3), config(Relays::Url(url.clone()), false), trust)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(30), server.online())
            .await
            .expect("the server reaches the relay within 30 s");
        assert!(
            server.bound_addrs().is_empty(),
            "no direct transport is bound"
        );
        let peer = NodeAddr::new(server.node_id())
            .unwrap()
            .with_relay_url(&url)
            .unwrap();
        assert!(peer.direct_addrs().is_empty());
        let served = serve(server);

        let client = Endpoint::bind_with(key(4), config(Relays::Url(url), false), trust)
            .await
            .unwrap();
        let conn = tokio::time::timeout(Duration::from_secs(30), client.connect(peer))
            .await
            .expect("connects through the relay within 30 s")
            .unwrap();
        assert_eq!(conn.path_kind(), Some(PathKind::Relay));
        assert!(conn.rtt().is_some_and(|rtt| rtt > Duration::ZERO));

        conn.close();
        client.close().await;
        served.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn no_relay_and_no_direct_paths_is_refused() {
        let error = Endpoint::bind(key(5), config(Relays::Disabled, false))
            .await
            .unwrap_err();
        assert_eq!(error, EndpointError::NoTransport);
    }

    #[test]
    fn addresses_validate_their_parts() {
        let node = SecretKey::from_bytes(&key(6)).public();
        let addr = NodeAddr::new(*node.as_bytes())
            .unwrap()
            .with_relay_url("https://relay.example.net")
            .unwrap()
            .with_direct_addr("192.0.2.1:7777".parse().unwrap());
        assert_eq!(addr.node_id(), *node.as_bytes());
        assert_eq!(
            addr.direct_addrs(),
            ["192.0.2.1:7777".parse::<SocketAddr>().unwrap()]
        );
        assert_eq!(addr.relay_urls().len(), 1);
        assert!(matches!(
            NodeAddr::new(*node.as_bytes())
                .unwrap()
                .with_relay_url("not a url"),
            Err(EndpointError::InvalidRelayUrl(_))
        ));
    }
}
