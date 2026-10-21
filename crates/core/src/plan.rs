// SPDX-License-Identifier: MIT OR Apache-2.0

//! The placement plan (RFC-0001 §5.4): the solver's output for one instance, and the deltas
//! runners adopt in place. RFC-0011 §2 and §12.3 own the semantics and define the supporting
//! types; the RFCs named on each type own its contents.
//!
//! [`PlacementPlan::validate`] checks RFC-0060 §3.2's P-solver-1 invariants.

pub mod budget;
pub mod cost;
pub mod delta;
pub mod speculative;
pub mod stage;
pub mod validate;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ids::{
    Blake3, DeviceId, GraphVersion, InstanceId, LinkId, ModelId, NodeId, TensorGroupId, TierId,
};
use crate::policy::{ExactnessPolicy, Objective};
use crate::units::Bytes;

pub use budget::*;
pub use cost::*;
pub use delta::*;
pub use speculative::*;
pub use stage::*;
pub use validate::*;

/// Total order over plans in a cluster (RFC-0001 §5.4). `term` is the coordinator lease term
/// that committed the plan; `seq` increases within a term. Ordering is lexicographic, so a
/// plan from a newer coordinator always supersedes one from an older coordinator. Terms embed
/// a 24-bit node tag, so the two sides of a network split never mint the same epoch
/// (RFC-0032 §7.2). Owner: RFC-0011 (semantics), RFC-0032 (terms and election).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct PlanEpoch {
    /// The committing coordinator's lease term.
    pub term: u64,
    /// The plan's sequence number within the term.
    pub seq: u64,
}

/// A plan's content identity: the BLAKE3 digest of its RFC-0011 §11.2 compact encoding.
/// `PlanCommitted` carries it, and a plan whose encoding exceeds 64 KiB travels by it over the
/// blob plane (RFC-0011 amendment B2, RFC-0032 §6.5).
///
/// `core` holds the type only. The encoding and the hash are computed by the crate that owns
/// the compact encoding, which links the BLAKE3 implementation; `core` links no hasher.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct PlanDigest(pub Blake3);

/// The solver's output for one instance (RFC-0001 §5.4). Immutable once committed.
/// Owner: RFC-0011, which adds `PlacementPlanExt` (PLC-E4-T01).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacementPlan {
    /// Where the plan sits in the cluster's order of plans.
    pub epoch: PlanEpoch,
    /// The instance the plan serves.
    pub instance: InstanceId,
    /// The model variant it places.
    pub model: ModelId,
    /// The resource graph it was solved against.
    pub graph: GraphVersion,
    /// Digest of the `InstancePolicy` it was solved under (RFC-0001 §11.3).
    pub policy: Blake3,
    /// What the solve optimised (RFC-0001 §11.3).
    pub objective: Objective,
    /// `Exact` unless the user opted in (RFC-0001 §11.3).
    pub exactness: ExactnessPolicy,
    /// Ring order. `stages[0]` is the head (embeddings, drafting); the last is the tail (LM
    /// head, sampler). The tail's successor is the head. A one-stage plan is both.
    pub stages: Vec<StageAssignment>,
    /// Exactly one entry per tensor group in the census.
    pub tensors: Vec<TierAssignment>,
    /// Expert-cache budgets per device and tier.
    pub expert_cache: Vec<ExpertCacheBudget>,
    /// KV budgets per stage and tier.
    pub kv: Vec<KvBudget>,
    /// Prefetch windows between tiers.
    pub prefetch: Vec<PrefetchWindow>,
    /// Scratch per device: activations, workspaces, captured graphs.
    pub scratch: Vec<(DeviceId, Bytes)>,
    /// Speculative decoding, when the solver chose a window above zero.
    pub speculative: Option<SpeculativeConfig>,
    /// Sequence, chunk and micro-batch limits (RFC-0011 §2).
    pub batching: BatchingLimits,
    /// The cost model's prediction (RFC-0011 §2). Runners compare live timings with it to
    /// detect drift (RFC-0001 §7.4).
    pub predicted: PredictedCost,
    /// Inputs that were `Prior` or `Derived` rather than measured. Shown in the dashboard.
    pub assumptions: Vec<Assumption>,
}

/// Where a tensor group lives and which device computes it (RFC-0001 §5.4).
/// Owner: RFC-0011 (assignment), RFC-0020 (home-tier semantics).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierAssignment {
    /// The census group.
    pub group: TensorGroupId,
    /// The single node responsible for this group in the cluster.
    pub owner: NodeId,
    /// The tier holding the authoritative resident copy on the owner. Expert-cache copies in
    /// faster tiers are runtime state (`ResidencyMap`), not plan state.
    pub home: TierId,
    /// Which device computes the group, and where it streams from `min_tokens` up.
    pub compute: ComputeSite,
    /// Deliberate replicas of hot data on other nodes. Rare, and never accidental.
    pub replicas: SmallVec<[NodeId; 2]>,
}

/// The devices that compute a tensor group (RFC-0001 §5.4). Owner: RFC-0011.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeSite {
    /// Device that executes the group at decode and small batch sizes: a device in the home
    /// tier's affinity set (execute-in-place). For groups homed on `Nvme`, the device that
    /// computes the group from its staging tier (ADR-0006 item 6).
    pub decode: DeviceId,
    /// From `min_tokens` up, stream the group to another device and compute it there. `None`
    /// when streaming never pays on this hardware.
    pub batched: Option<StreamSite>,
}

/// The streaming site of a [`ComputeSite`] (RFC-0001 §5.4). Owner: RFC-0011.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamSite {
    /// The device that computes the streamed group.
    pub device: DeviceId,
    /// The link the group streams over.
    pub via: LinkId,
    /// The measured n* (cost model §3.2): the smallest pass, in tokens, at which streaming
    /// beats computing in place.
    pub min_tokens: u32,
}
