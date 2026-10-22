// SPDX-License-Identifier: MIT OR Apache-2.0

//! The data plane's `Transport` trait and its `Channel` (RFC-0001 §5.8, RFC-0031 §2; ADR-0011
//! item 2). Owner: RFC-0031.
//!
//! This module is the specification every backend builds on: QUIC (DIST-E7-T03), TCP
//! (DIST-E7-T05), shared memory (DIST-E7-T06), RDMA (DIST-E7-T07, DIST-E7-T08), and the
//! simulator's `SimTransport` (QUIC-equivalent semantics over `SimFabric`, QUAL-E8-T04). The
//! backends here are [`quic`] and [`tcp`], the TCP fallback, which both run their channels on
//! [`stream`]'s driver. [`conformance`] is RFC-0031 §2's binding semantics as a suite every
//! backend runs.
//!
//! - A [`Transport`] opens channels: [`Transport::connect`] and [`Transport::accept`] are
//!   asynchronous, and run in the runner's `net` domain, never on a hot thread.
//! - A [`Channel`] is concrete. Its hot-path methods are synchronous and never block; it calls
//!   the backend through [`ChannelBackend`].
//! - Frames name their extension and payload by descriptors into registered slots
//!   ([`buffer`]), so no frame is copied on the way in or out except where RFC-0031 §8 says.
//! - [`core`] is RFC-0031 §7's flow control, acknowledgement and retransmission, sans I/O:
//!   every backend wraps one [`ChannelCore`] per channel side (DIST-E7-T02).
//!
//! `exec` and `sched` never see this module: they hand frames to the runner over SPSC rings,
//! and only the runner and the daemon depend on `net` (RFC-0001 §4.2 rule 3).

pub mod buffer;
mod channel;
pub mod conformance;
pub mod core;
pub mod domain;
pub mod quic;
pub mod stream;
pub mod tcp;

use std::future::Future;
use std::net::SocketAddr;

use syntropy_core::resource::TransportKind;
use syntropy_core::{Bytes, NodeId};

pub use self::core::{ChannelCore, CoreConfig, CreditCadence, SendConfig, Transmit};
pub use buffer::{
    BufferPool, ExtBuf, ExtRef, HeapMemory, PayloadRef, PayloadSlice, PoolError, PoolId,
    PoolMemory, PoolRegistration, SLOT_ALIGN, SlotReader, SlotWriter,
};
pub use channel::{
    Channel, ChannelBackend, ChannelKey, ChannelPriority, ChannelPurpose, ChannelSpec,
    ChannelStats, CloseReason, INGRESS, InFrame, OutFrame, SendError, SendTicket,
};

use crate::endpoint::{PathKind, RelayUrl};
use crate::error::Error;

/// What a transport can do (RFC-0031 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransportCaps {
    /// Frames land in a registered slot with no host copy.
    pub zero_copy_recv: bool,
    /// Frames can land in device memory (GPUDirect RDMA, a `Unified` import).
    pub device_memory_recv: bool,
    /// The largest frame: 16,773,120 bytes for RDMA over Thunderbolt 5, and `net.max_frame`
    /// (64 MiB by default) for the others. A larger prefill chunk is split into parts (RFC-0031
    /// §6.3).
    pub max_frame: Bytes,
    /// Pools must be registered before use: RDMA, shared memory.
    pub needs_registration: bool,
    /// True for QUIC and TCP; per ADR-0314 for RDMA; false for shared memory.
    pub encrypted: bool,
    /// The transport itself delivers reliably: RC queue pairs, QUIC, TCP. False for UC queue
    /// pairs (Thunderbolt 5), where the channel's acknowledgement is the reliability layer.
    pub reliable: bool,
    /// The most channels to one peer: 4 for Thunderbolt 5 (ADR-0011 item 6); `u16::MAX` where the
    /// transport sets no cap.
    pub max_channels_per_peer: u16,
    /// RDMA's inline-send limit for small frames, in bytes; 0 if none.
    pub inline_max: u16,
}

/// Where to reach a peer's endpoint (RFC-0031 §2).
///
/// RFC-0031 §2 gives two more fields, each added by the task that builds its type: the RDMA
/// pairing (DIST-E7-T07) and the shared-memory offer (DIST-E7-T06).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RemoteEndpoint {
    pub node: NodeId,
    /// The endpoint's Ed25519 public key: a runner key, or the node key for a daemon.
    pub key: [u8; 32],
    /// Every known direct address, best first; iroh races them (RFC-0031 §4.1).
    pub addrs: Vec<SocketAddr>,
    /// The peer's relay, for a path when no direct one connects (RFC-0031 §4.1, §5.4).
    pub relay: Option<RelayUrl>,
}

/// The transport kind of a QUIC path: `Quic` when direct, `QuicRelay` through a relay, so plans
/// and metrics never mistake a relayed edge for a direct one (RFC-0031 §2, §4.1).
impl From<PathKind> for TransportKind {
    fn from(path: PathKind) -> TransportKind {
        match path {
            PathKind::Direct => TransportKind::Quic,
            PathKind::Relay => TransportKind::QuicRelay,
        }
    }
}

/// A data-plane transport backend (ADR-0011). The solver picks one per hop from measured α
/// (`HopPlan::transport`). Owner: RFC-0031.
///
/// `connect` and `accept` return futures that are `Send`, so the `net` domain can run them on
/// its runtime. Implementations write them as `async fn`. Because of those methods the trait is
/// not object-safe; code that picks a transport at run time is generic over it, or wraps it.
///
/// RFC-0031 §2's `cost` method, the per-frame cost from the link profile, comes with link
/// selection (DIST-E7-T11), which defines it with RFC-0010's `LinkCaps`.
pub trait Transport: Send + Sync {
    /// `SharedMemory`, `Rdma(_)`, `Quic`, `QuicRelay` or `Tcp`. `QuicRelay` marks a relayed
    /// path, so plans and metrics never mistake it for a direct one.
    fn kind(&self) -> TransportKind;

    /// Zero-copy receive, device-memory receive, maximum frame size, whether buffers must be
    /// registered, whether the channel is encrypted.
    fn caps(&self) -> TransportCaps;

    /// Registers a buffer pool: RDMA memory regions, shared-memory segments. Pools come from the
    /// tier manager's pinned arena (RFC-0001 §9.4). Each pool is registered once per transport.
    fn register(&self, pool: &BufferPool) -> Result<PoolRegistration, Error>;

    /// Opens the channel `spec.key` to `remote`: a QUIC handshake, an RDMA queue-pair setup.
    fn connect(
        &self,
        remote: &RemoteEndpoint,
        spec: ChannelSpec,
    ) -> impl Future<Output = Result<Channel, Error>> + Send;

    /// Accepts the channel `spec.key` from the peer that connects it.
    fn accept(&self, spec: ChannelSpec) -> impl Future<Output = Result<Channel, Error>> + Send;

    /// Re-establishes a broken channel with the same key and sends its unacknowledged frames
    /// again (RFC-0031 §7.4). A backend without it returns an `Unsupported` error.
    fn reconnect(&self, channel: &mut Channel) -> impl Future<Output = Result<(), Error>> + Send {
        let _ = channel;
        async { Err(Error::unsupported()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_direct_path_is_quic_and_a_relayed_one_quic_relay() {
        assert_eq!(TransportKind::from(PathKind::Direct), TransportKind::Quic);
        assert_eq!(
            TransportKind::from(PathKind::Relay),
            TransportKind::QuicRelay
        );
    }

    #[test]
    fn a_remote_endpoint_names_its_relay() {
        let relay: RelayUrl = "https://relay.example.net".parse().expect("a relay URL");
        let remote = RemoteEndpoint {
            node: NodeId([7; 32]),
            key: [7; 32],
            addrs: vec![SocketAddr::from(([192, 0, 2, 1], 7600))],
            relay: Some(relay.clone()),
        };
        assert_eq!(remote.relay.as_ref(), Some(&relay));
        assert!(relay.to_string().starts_with("https://relay.example.net"));
    }
}
