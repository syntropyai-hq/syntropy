// SPDX-License-Identifier: MIT OR Apache-2.0

//! The payload types of the base events that RFC-0001 §5.10 names and no task had built.
//! DIST-E9-T01 defines them here (`dependency-fixes.md`, "DIST-E9-T01's payload types");
//! `NodeInfo` is a stand-in that DIST-E9-T03 replaces.
//!
//! Each holds what [`apply`](super::apply) needs to do what RFC-0032 §6.5 says for its event.
//! They follow the envelope's evolution rule (ADR-0305 item 3): a decoder ignores fields it
//! does not know. So an event a later build encodes with a fuller type still decodes here, and
//! its `event_bytes`, which the chain hashes, are kept verbatim either way (RFC-0032 §6.1).
//!
//! A payload whose shape no RFC fixes yet is an [`Opaque`]: the owner's CBOR value, kept
//! whole, so this crate settles none of its fields.

use std::ops::Range;

use serde::{Deserialize, Serialize};
use syntropy_core::policy::{ConfigValue, KeyPath};
use syntropy_core::{
    Blake3, Bytes, InstanceId, LayerIx, NodeId, PlacementPlan, PlanDigest, PlanEpoch,
};

/// A payload value whose type another task defines, kept as the CBOR value it was encoded as.
/// Its owner's type replaces it; a value encoded by that type decodes here unchanged.
///
/// It stands in for an `Option<T>` too, with `null` for `None`, which is how serde encodes it:
/// an `Option<Opaque>` could not tell `None` from `Some(null)` once encoded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Opaque(pub ciborium::Value);

impl Default for Opaque {
    fn default() -> Self {
        Opaque::null()
    }
}

impl Opaque {
    /// An empty value: CBOR `null`.
    pub const fn null() -> Self {
        Opaque(ciborium::Value::Null)
    }
}

/// A node's description (RFC-0001 §5.2). Stand-in: only `id`, the field `apply` reads.
/// DIST-E9-T03 replaces it with RFC-0001 §5.2's type (PLC-E1-T09); until then, the fields a
/// full description carries are dropped here on decode and kept in `event_bytes`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeInfo {
    /// The node.
    pub id: NodeId,
}

/// A runner's state without its fields, as the log records it (RFC-0001 §5.11): `core`'s, since
/// DIST-E1-T05 put `RunnerState` there. Re-exported here, with its variants in the same order, so
/// the log's encoding is unchanged.
pub use syntropy_core::RunnerStateKind;

/// A model manifest's digest (RFC-0041): `RegisterModel` fetches the body by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ManifestDigest(pub Blake3);

/// A committed plan's body (RFC-0001 §5.10): inline when its RFC-0011 §11.2 compact encoding is
/// at most 64 KiB, otherwise by digest, with the body on the blob plane. The shell fetches and
/// verifies a by-digest body before `validate()` and `apply()` see the entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PlanBody {
    /// The plan itself.
    Inline(Box<PlacementPlan>),
    /// The plan's digest and the size of its compact encoding.
    ByDigest {
        /// The plan's content identity.
        digest: PlanDigest,
        /// The compact encoding's size.
        len: Bytes,
    },
}

/// What logs and the dashboard show of a committed plan (RFC-0001 §5.10): the instance, the
/// epoch, and each stage's node and layer range. RFC-0011 owns the fields and may add to them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanSummary {
    /// The instance the plan serves.
    pub instance: InstanceId,
    /// The plan's epoch.
    pub epoch: PlanEpoch,
    /// Each stage's node and the layers it holds, in ring order.
    pub stages: Vec<StageSummary>,
}

/// One stage of a [`PlanSummary`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageSummary {
    /// The node that hosts the stage.
    pub node: NodeId,
    /// The layers it holds.
    pub layers: Range<LayerIx>,
}

/// Why an instance stopped being routable (RFC-0001 §5.10). No RFC lists its variants yet, so
/// the value is kept whole.
pub type UnroutableReason = Opaque;

/// One instance's workload report (RFC-0001 §5.10), which the solver reads when it re-solves.
/// The report's parts belong to other RFCs: `experts` to RFC-0023 (TIER-E10-T09's hooks), the
/// rest to RFC-0012. Each part is kept whole until its owner defines it; RFC-0001's optional
/// parts are `null` when absent.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkloadReport {
    /// Expert hit curves, their union, per-tier hit rates and the balancer fit (RFC-0023 §13).
    pub experts: Opaque,
    /// Acceptance per drafter, greedy and sampled (RFC-0012 §11).
    pub acceptance: Vec<Opaque>,
    /// Context-length statistics (RFC-0012 §11).
    pub context: Opaque,
    /// Concurrency statistics (RFC-0012 §11).
    pub concurrency: Opaque,
}

/// The digest of the chunk set a node holds for a model (RFC-0041). RFC-0041 has not fixed its
/// form, so the value is kept whole.
pub type ChunkSetDigest = Opaque;

/// Who proposed a configuration patch (RFC-0054 §2.3): node, layer, file path, and per key the
/// (index, term) of the value the proposer last saw. RFC-0054 has not fixed its fields, so the
/// value is kept whole.
pub type ProposalOrigin = Opaque;

/// A change to the cluster-scoped configuration keys (RFC-0001 §5.10, RFC-0054 §2.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfigPatch {
    /// The keys set, with their values, in order.
    pub set: Vec<(KeyPath, ConfigValue)>,
    /// The keys removed.
    pub unset: Vec<KeyPath>,
    /// Who proposed it.
    pub origin: ProposalOrigin,
}
