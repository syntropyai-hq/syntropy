// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph assembly records (RFC-0010 §1.4): the node inventory that carries device, tier and
//! link records to the coordinator, and the delta `GraphUpdated` carries.

use serde::{Deserialize, Serialize};

use crate::ids::{Blake3, LinkId, NodeId, TierId};
use crate::units::Bytes;

use super::device::DeviceInfo;
use super::link::LinkInfo;
use super::tier::{HostPoolInfo, TierInfo};

/// One node's device, tier and link records (RFC-0010 §1.4, amendment A1). Carried in
/// `NodeInfo::inventory` on `Admit`, and in `CapabilityProfileExt::inventory` after a
/// hardware change. Every field has a serde default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeInventory {
    /// Devices, in ordinal order.
    pub devices: Vec<DeviceInfo>,
    /// Tiers, in ordinal order.
    pub tiers: Vec<TierInfo>,
    /// Links whose source is on this node.
    pub links: Vec<LinkInfo>,
    /// One host pool record per NUMA domain (RFC-0010 §2.2).
    pub host_pools: Vec<HostPoolInfo>,
}

/// What changed in one `GraphUpdated` (RFC-0010 §1.4). Every field has a serde default.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphDelta {
    /// Nodes admitted.
    pub nodes_added: Vec<NodeId>,
    /// Nodes that left.
    pub nodes_removed: Vec<NodeId>,
    /// Profiles replaced: node, profile version, profile digest.
    pub profiles: Vec<(NodeId, u64, Blake3)>,
    /// Ledgers changed.
    pub ledgers: Vec<(NodeId, LedgerSummary)>,
    /// Links changed.
    pub links: Vec<(LinkId, LinkChange)>,
    /// Why the graph changed.
    pub cause: GraphChangeCause,
}

/// A node's device ledger as the coordinator sees it (RFC-0001 §5.10 `ReportLedger`).
///
/// RFC-0001 names the type without fields. The minimal reading is the capacity running
/// instances hold per tier, which is what `ResourceGraph::reservations` needs; RFC-0030 and
/// RFC-0033 may add fields with serde defaults.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LedgerSummary {
    /// Bytes held per tier.
    pub held: Vec<(TierId, Bytes)>,
}

/// How a link changed (RFC-0010 §1.4).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LinkChange {
    /// The link is new.
    Added,
    /// The link is gone.
    Removed,
    /// The link's usable transports changed.
    TransportsChanged,
}

/// Why the graph changed (RFC-0010 §1.4).
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum GraphChangeCause {
    /// A node joined or left. The serde default, as the first listed variant.
    #[default]
    Membership,
    /// A startup profile arrived.
    StartupProfile,
    /// A model probe finished.
    ModelProbe,
    /// Measured costs drifted past the threshold.
    Drift,
    /// A device ledger changed.
    Ledger,
}
