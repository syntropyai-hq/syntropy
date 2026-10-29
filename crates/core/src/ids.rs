// SPDX-License-Identifier: MIT OR Apache-2.0

//! Identifiers of nodes, devices, tiers and links, and the digest and version types the
//! resource graph uses (RFC-0001 §5.1).
//!
//! RFC-0010 §1.4 owns the ordinal rules for [`DeviceId`] and [`TierId`];
//! [`crate::resource::ordinal`] implements them.

use serde::{Deserialize, Serialize};

/// Persistent node identity: the node's Ed25519 public key. Generated at first start and
/// stored with the secret key in the daemon's state directory (mode 0600). Owner: RFC-0034.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub [u8; 32]);

impl NodeId {
    /// The first 8 bytes, read big-endian. Used where a compact prefix is enough (frame
    /// headers, `SequenceId`). Never an identity on its own.
    pub fn short(&self) -> u64 {
        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&self.0[..8]);
        u64::from_be_bytes(prefix)
    }
}

/// A compute engine on a node. Ordinals are assigned by the probe in a fixed order:
/// discrete GPUs by PCI address, then integrated GPUs, then one CPU device per NUMA domain,
/// then NPUs (RFC-0010 §1.4). Stable across restarts while the hardware is unchanged.
/// Owner: RFC-0010.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DeviceId {
    /// The node the device belongs to.
    pub node: NodeId,
    /// Position in the node's device order.
    pub ordinal: u16,
}

/// A memory or storage level on a node. Owner: RFC-0010 (discovery), RFC-0020 (semantics).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TierId {
    /// The node the tier belongs to.
    pub node: NodeId,
    /// Position in the node's tier order (RFC-0010 §1.4).
    pub ordinal: u16,
}

/// A directed, measured path between two endpoints. Directed because Wi-Fi and other
/// asymmetric links differ by direction. `path` separates parallel paths of one kind (two
/// NICs, several Thunderbolt cables). Owner: RFC-0010 (measurement), RFC-0031 (network kinds).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LinkId {
    /// Where the link starts.
    pub src: Endpoint,
    /// Where the link ends.
    pub dst: Endpoint,
    /// What carries the bytes.
    pub kind: LinkKind,
    /// Separates parallel paths of one kind between the same endpoints.
    pub path: u8,
}

/// One end of a [`LinkId`] (RFC-0001 §5.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Endpoint {
    /// A compute device.
    Device(DeviceId),
    /// A memory or storage tier.
    Tier(TierId),
    /// A network interface, numbered as RFC-0030 §7.1's `IfaceInfo::ordinal`.
    Nic {
        /// The node that owns the interface.
        node: NodeId,
        /// The interface's ordinal on that node.
        ordinal: u16,
    },
}

/// What a [`LinkId`] runs over (RFC-0001 §5.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LinkKind {
    /// A device and its own tier.
    MemoryBus,
    /// Host and device, including Thunderbolt PCIe tunnels.
    Pcie,
    /// An `Nvme` tier and the host.
    Storage,
    /// Node to node.
    Network(NetMedium),
}

/// The physical or logical medium of a network link (RFC-0001 §5.1; RFC-0031 owns network
/// kinds).
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum NetMedium {
    /// Wired Ethernet. The serde default, filling the field in records that predate it.
    #[default]
    Ethernet,
    /// Wi-Fi.
    WiFi,
    /// IP over a Thunderbolt or USB4 cable.
    ThunderboltIp,
    /// Remote direct memory access.
    Rdma(RdmaKind),
    /// An overlay network (a VPN tunnel).
    Overlay,
    /// Two nodes on one host.
    Loopback,
}

/// The RDMA fabric of a [`NetMedium::Rdma`] link (RFC-0001 §5.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RdmaKind {
    /// RDMA over Converged Ethernet, version 2.
    RoceV2,
    /// InfiniBand.
    InfiniBand,
    /// RDMA over Thunderbolt 5.
    Thunderbolt5,
}

/// A BLAKE3 digest (RFC-0001 §5.1). The default is all zeros, which no real digest is
/// expected to equal; it only fills a field a record predates.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Blake3(pub [u8; 32]);

/// Version of the resource graph; increases by one per `GraphUpdated` (RFC-0010 §1.4).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct GraphVersion(pub u64);

/// One committed ring of stages serving one model. Allocated by the coordinator, never
/// reused (RFC-0001 §5.1). Owner: RFC-0011.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InstanceId(pub u64);

/// Content identity of a model variant (RFC-0001 §5.1, §5.3; ADR-0411). Owner: RFC-0041.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub Blake3);

/// A decoder layer's index. MTP modules take indices after the last decoder layer
/// (RFC-0001 §5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerIx(pub u16);

/// A routed expert's index within its layer (RFC-0001 §5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExpertIx(pub u16);

/// A stage's position in the ring; 0 is the head (RFC-0001 §5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StageIx(pub u16);

/// Index into `TensorCensus::groups`, the solver's placement unit (RFC-0001 §5.1, §5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TensorGroupId(pub u32);

/// BLAKE3 of one chunk's bytes (RFC-0001 §5.6; RFC-0041 §1, §6). Known after the chunk's first
/// fetch, or at registration for a local file (RFC-0041 §6.3). Owner: RFC-0041.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ChunkId(pub Blake3);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_is_the_big_endian_prefix() {
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        bytes[8] = 0xff;
        assert_eq!(NodeId(bytes).short(), 0x0102_0304_0506_0708);
    }
}
