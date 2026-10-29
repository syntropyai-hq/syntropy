// SPDX-License-Identifier: MIT OR Apache-2.0

//! The resource graph (RFC-0001 §5.2) and the capacity running instances hold in it
//! (PLC-E1-T18). Owner: RFC-0010 (assembly, §1.4), RFC-0001 (the record).
//!
//! RFC-0001 names `Reservation` without fields. Its form is a maintainer decision, drafted in
//! RFC-0010 §1.4's note of 2026-10-29 with four options; this module builds the recommended
//! default, (a). A reservation is keyed by tier and, when the ledger says so, by the instance
//! and plan epoch that hold it. RFC-0001's per-device dimension is dropped: a host tier's
//! `affinity` lists several devices, so a tier's bytes do not split by device.
//!
//! The coordinator builds the graph from the event log (RFC-0010 §1.4). This module only holds
//! the record and the pure rules over it: how a node's [`LedgerSummary`] becomes reservations,
//! and how much of a tier a plan may use.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ids::{GraphVersion, InstanceId, NodeId, TierId};
use crate::plan::PlanEpoch;
use crate::units::Bytes;

use super::caps::CapabilityProfile;
use super::device::DeviceInfo;
use super::inventory::LedgerSummary;
use super::link::LinkInfo;
use super::node::NodeInfo;
use super::tier::TierInfo;

/// The cluster's resources as the solver sees them (RFC-0001 §5.2).
///
/// `profiles` is a map in memory and a list of `(NodeId, CapabilityProfile)` pairs in node
/// order on the wire. A `NodeId` is a byte array, which JSON cannot use as a map key, and the
/// graph is served as JSON (RFC-0050's `GET /v1/syntropy/cluster/graph`). RFC-0001 §5.2's note
/// of 2026-10-29 records this.
///
/// Every list keeps the order its builder gave it, so one graph built from the same records in
/// the same order encodes to the same bytes through one encoder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResourceGraph {
    /// Bumped on any change the solver can observe: membership, profile drift past the
    /// threshold (RFC-0001 §7.4), ledger changes.
    pub version: GraphVersion,
    /// The admitted nodes.
    pub nodes: Vec<NodeInfo>,
    /// Every node's devices.
    pub devices: Vec<DeviceInfo>,
    /// Every node's tiers.
    pub tiers: Vec<TierInfo>,
    /// Every link.
    pub links: Vec<LinkInfo>,
    /// Each node's latest capability profile.
    #[serde(default, with = "profile_pairs")]
    pub profiles: BTreeMap<NodeId, CapabilityProfile>,
    /// Capacity held by running instances. The solver treats it as unavailable, except the
    /// reservation of the plan a break-before-make transition replaces ([`Self::available`]).
    #[serde(default)]
    pub reservations: Vec<Reservation>,
}

impl ResourceGraph {
    /// The tier's record, if the graph has it.
    #[must_use]
    pub fn tier(&self, id: TierId) -> Option<&TierInfo> {
        self.tiers.iter().find(|t| t.id == id)
    }

    /// The bytes of `tier` a plan may use: the tier's `usable` less every reservation on it,
    /// saturating at zero. `exempt` names the holder whose reservations do not count:
    ///
    /// - a break-before-make re-plan of an instance passes the holder of the plan it replaces,
    ///   the instance and the epoch it runs, since that plan's runners stop first;
    /// - a make-before-break re-plan passes `None`, since the old epoch's runners hold their
    ///   bytes until the new epoch is routable (RFC-0011 §11.2 `transition`; RFC-0032 §4.2);
    /// - a first plan passes `None`.
    ///
    /// A reservation with no holder is never exempt, and neither is one whose epoch differs
    /// from `exempt`'s, such as an orphan runner's (RFC-0032 §7.5). `None` when the graph has
    /// no such tier.
    #[must_use]
    pub fn available(&self, tier: TierId, exempt: Option<Holder>) -> Option<Bytes> {
        let usable = self.tier(tier)?.usable.0;
        let held = self
            .reservations
            .iter()
            .filter(|r| r.tier == tier && (r.holder.is_none() || r.holder != exempt))
            .fold(0u64, |sum, r| sum.saturating_add(r.bytes.0));
        Some(Bytes(usable.saturating_sub(held)))
    }
}

/// Bytes of one tier held by a running instance, or by something the ledger cannot attribute
/// (RFC-0010 §1.4, note of 2026-10-29, option (a)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Reservation {
    /// The tier, which names its node.
    pub tier: TierId,
    /// Who holds the bytes. `None`: bytes in the ledger's `held` that no entry accounts for,
    /// from a node that does not report entries. Such a reservation is never exempt.
    #[serde(default)]
    pub holder: Option<Holder>,
    /// The bytes held.
    pub bytes: Bytes,
}

/// The instance and plan epoch that hold a [`Reservation`]. Two epochs of one instance hold
/// separate reservations while a make-before-break transition runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Holder {
    /// The running instance.
    pub instance: InstanceId,
    /// The plan epoch its runners run.
    pub epoch: PlanEpoch,
}

/// Why a [`LedgerSummary`] could not become reservations.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerError {
    /// `held` names the tier more than once.
    DuplicateTier(TierId),
    /// Two entries name the same holder and tier.
    DuplicateEntry(TierId, Holder),
    /// An entry has no holder. Entries attribute bytes; the unattributed remainder is derived.
    NoHolder(TierId),
    /// The entries for the tier add up to more than `held` says the node holds.
    EntriesExceedHeld {
        /// The tier.
        tier: TierId,
        /// What `held` says.
        held: Bytes,
        /// What the entries add up to, saturating.
        entries: Bytes,
    },
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let at = |t: &TierId| format!("tier {} of node {:016x}", t.ordinal, t.node.short());
        match self {
            LedgerError::DuplicateTier(t) => write!(f, "{} is held more than once", at(t)),
            LedgerError::DuplicateEntry(t, h) => write!(
                f,
                "{} has two entries for instance {} at epoch {}.{}",
                at(t),
                h.instance.0,
                h.epoch.term,
                h.epoch.seq
            ),
            LedgerError::NoHolder(t) => write!(f, "{} has an entry with no holder", at(t)),
            LedgerError::EntriesExceedHeld {
                tier,
                held,
                entries,
            } => write!(
                f,
                "{}: entries add up to {} bytes, above the {} bytes held",
                at(tier),
                entries.0,
                held.0
            ),
        }
    }
}

impl std::error::Error for LedgerError {}

impl LedgerSummary {
    /// The reservations this ledger stands for: every entry, plus one reservation with no
    /// holder for each tier whose `held` bytes the entries do not all account for. A ledger
    /// without entries, as from a node that does not report them, yields one such reservation
    /// per tier it holds. Tiers held at zero bytes yield nothing. The result is sorted by tier,
    /// then holder, with the unattributed reservation first, so it does not depend on the order
    /// the node listed its tiers or entries in.
    ///
    /// The checks run here, not on decode, so a graph can carry a ledger as reported. A caller
    /// that gets an error must fail closed: dropping the node's ledger would make its held
    /// capacity look free. The tiers are the reporting node's own; nothing here checks that
    /// `TierId::node` is the node that reported the ledger, so the coordinator does (PLC-E1-T11).
    ///
    /// # Errors
    ///
    /// [`LedgerError`] when `held` names a tier twice, an entry has no holder, two entries
    /// share a holder and tier, or a tier's entries exceed its `held` bytes. An entry for a
    /// tier `held` does not name counts against zero bytes held.
    pub fn reservations(&self) -> Result<Vec<Reservation>, LedgerError> {
        let mut held: BTreeMap<TierId, u64> = BTreeMap::new();
        for &(tier, bytes) in &self.held {
            if held.insert(tier, bytes.0).is_some() {
                return Err(LedgerError::DuplicateTier(tier));
            }
        }
        let mut attributed: BTreeMap<TierId, u128> = BTreeMap::new();
        let mut entries: BTreeMap<(TierId, Holder), Bytes> = BTreeMap::new();
        for entry in &self.entries {
            let holder = entry.holder.ok_or(LedgerError::NoHolder(entry.tier))?;
            if entries.insert((entry.tier, holder), entry.bytes).is_some() {
                return Err(LedgerError::DuplicateEntry(entry.tier, holder));
            }
            *attributed.entry(entry.tier).or_default() += u128::from(entry.bytes.0);
        }
        for (&tier, &sum) in &attributed {
            let cap = held.get(&tier).copied().unwrap_or(0);
            if sum > u128::from(cap) {
                return Err(LedgerError::EntriesExceedHeld {
                    tier,
                    held: Bytes(cap),
                    entries: Bytes(u64::try_from(sum).unwrap_or(u64::MAX)),
                });
            }
        }
        let mut out: Vec<Reservation> = entries
            .into_iter()
            .filter(|(_, bytes)| bytes.0 > 0)
            .map(|((tier, holder), bytes)| Reservation {
                tier,
                holder: Some(holder),
                bytes,
            })
            .collect();
        for (&tier, &bytes) in &held {
            // The check above keeps each tier's entries within its `held` bytes, so `rest` is
            // at most `bytes`.
            let rest = u128::from(bytes) - attributed.get(&tier).copied().unwrap_or(0);
            if rest > 0 {
                out.push(Reservation {
                    tier,
                    holder: None,
                    bytes: Bytes(u64::try_from(rest).unwrap_or(bytes)),
                });
            }
        }
        out.sort();
        Ok(out)
    }
}

/// `ResourceGraph::profiles` as a list of pairs in node order.
mod profile_pairs {
    use std::collections::BTreeMap;

    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{CapabilityProfile, NodeId};

    pub(super) fn serialize<S: Serializer>(
        map: &BTreeMap<NodeId, CapabilityProfile>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let pairs: Vec<(&NodeId, &CapabilityProfile)> = map.iter().collect();
        pairs.serialize(serializer)
    }

    /// A node listed twice is refused, since which profile holds would depend on list order, and
    /// so is a profile listed under a node other than its own.
    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<NodeId, CapabilityProfile>, D::Error> {
        let pairs = Vec::<(NodeId, CapabilityProfile)>::deserialize(deserializer)?;
        let mut map = BTreeMap::new();
        for (node, profile) in pairs {
            if profile.node != node {
                return Err(D::Error::custom(format!(
                    "the profile listed under node {:016x} is node {:016x}'s",
                    node.short(),
                    profile.node.short()
                )));
            }
            if map.insert(node, profile).is_some() {
                return Err(D::Error::custom(format!(
                    "node {:016x} has two capability profiles",
                    node.short()
                )));
            }
        }
        Ok(map)
    }
}
