// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph assembly records (RFC-0010 §1.4): the node inventory that carries device, tier and
//! link records to the coordinator, and the delta `GraphUpdated` carries.

use serde::{Deserialize, Serialize};

use crate::ids::{Blake3, LinkId, NodeId, TierId};
use crate::units::Bytes;

use super::device::DeviceInfo;
use super::graph::Reservation;
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
/// RFC-0001 names the type without fields. `held` is the capacity running instances hold per
/// tier; `entries` says which instance and epoch hold it (PLC-E1-T18; RFC-0010 §1.4, note of
/// 2026-10-29). [`LedgerSummary::reservations`] turns both into `ResourceGraph::reservations`.
/// RFC-0030 and RFC-0033 may add fields with serde defaults.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LedgerSummary {
    /// Bytes held per tier.
    pub held: Vec<(TierId, Bytes)>,
    /// Who holds them: one entry per instance, epoch and tier, each with a holder. A node that
    /// does not report entries sends none, and its `held` bytes stay unattributed. An empty
    /// list is not encoded, so a ledger without entries keeps the bytes it had before the field.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub entries: Vec<Reservation>,
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
