// SPDX-License-Identifier: MIT OR Apache-2.0

//! Link topology records: RFC-0001 §5.2's `LinkInfo` and RFC-0031 §2's `TransportKind`.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::iface::IfaceInfo;
use crate::ids::{LinkId, RdmaKind};

/// One directed link and the transports usable on it (RFC-0001 §5.2). Owner: RFC-0010.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkInfo {
    /// The link's identity.
    pub id: LinkId,
    /// Transports usable on this link, best first.
    #[serde(default)]
    pub transports: SmallVec<[TransportKind; 3]>,
    /// For a network link, the interface it leaves through: name, addresses, MTU
    /// (RFC-0030 §7.1). `None` for other links.
    #[serde(default)]
    pub interface: Option<IfaceInfo>,
    /// The link is metered.
    #[serde(default)]
    pub metered: bool,
}

/// A transport that carries frames over a link (RFC-0031 §2, which adds `QuicRelay` to
/// RFC-0001 §5.8's list). Owner: RFC-0031.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TransportKind {
    /// Shared memory between two nodes on one host.
    SharedMemory,
    /// Remote direct memory access.
    Rdma(RdmaKind),
    /// QUIC on a direct path.
    Quic,
    /// QUIC through a relay, kept apart from direct paths in plans, metrics and the dashboard.
    QuicRelay,
    /// TCP.
    Tcp,
}
