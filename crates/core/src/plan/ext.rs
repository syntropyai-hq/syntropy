// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0011 §2's additions to the placement plan: how the plan is entered, how it was solved,
//! whether it is degraded, what it moves and what it relies on. Owner: RFC-0011.
//!
//! Every field has a serde default, so a plan serialized before these additions still loads.
//! A default is the conservative reading: break-before-make, uncertified, nothing certified at
//! run time, no degradation recorded.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ids::{Blake3, NodeId, TensorGroupId};
use crate::resource::TierKind;
use crate::units::{Bytes, Nanos};

/// Added to `PlacementPlan` (RFC-0011 §2). Owner: RFC-0011.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlacementPlanExt {
    /// How the plan replaces the one before it (RFC-0001 §7.3).
    pub transition: TransitionStrategy,
    /// Digest of the `PlanExplanation`, fetched over the bulk plane (RFC-0011 §14).
    pub explanation: Option<Blake3>,
    /// How the solve went.
    pub solve: SolveStats,
    /// Why the plan is degraded, if it is (RFC-0011 §17).
    pub degraded: Option<DegradedReason>,
    /// Bytes the transition must fetch or re-pack.
    pub move_bytes: Bytes,
    /// What the plan relies on, from the conformance manifest (RFC-0060 §6).
    pub certification: PlanCertification,
    /// RDMA queue pairs the plan takes per node.
    pub rdma_qps: SmallVec<[(NodeId, u16); 8]>,
    /// Groups stored once per reader layout instead of one shared layout (RFC-0040 §5.4;
    /// RFC-0011 §11.2).
    pub per_site_layout: Vec<TensorGroupId>,
    /// Each measured node's WAN rate class, ⌊log2⌋ of its origin download rate in MB/s as the
    /// solve read it; unmeasured nodes have no entry (RFC-0041 §8.3; RFC-0011 §11.2).
    pub wan_rate_class: SmallVec<[(NodeId, u8); 8]>,
}

/// How a new plan replaces the old one (RFC-0001 §7.3). Owner: RFC-0001.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransitionStrategy {
    /// The new runners load while the old epoch serves, when the device ledgers have room for
    /// both.
    MakeBeforeBreak,
    /// The old epoch drains first; the usual case, because plans fill memory. The default.
    #[default]
    BreakBeforeMake,
}

/// Which method produced the plan (RFC-0011 §9.5).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolvePath {
    /// The MILP proved its gap.
    Milp,
    /// The MILP's best incumbent when a limit stopped it.
    MilpIncumbent,
    /// The DP with greedy in-node placement, the production method. The default.
    #[default]
    Dp,
}

/// How a solve went (RFC-0011 §2, §9). Owner: RFC-0011.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolveStats {
    /// Which method won.
    pub path: SolvePath,
    /// The relative MIP gap at the end; 0 for the DP, which proves none.
    pub gap: f32,
    /// MIP branch-and-bound nodes explored.
    pub mip_nodes: u64,
    /// Simplex iterations.
    pub simplex_iterations: u64,
    /// Wall-clock time of the solve.
    pub wall: Nanos,
}

/// Why a plan is degraded (RFC-0011 §17). Owner: RFC-0011.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DegradedReason {
    /// Dense bytes on NVMe: the first pass was infeasible (constraint C10).
    DenseOnNvme,
    /// One node holds two stages (C2 relaxed for that node), used only in recovery.
    MultiStageOnNode,
    /// A survivor absorbed a lost stage (RFC-0033 §6.4).
    AbsorbedStage,
}

/// What a plan relies on (RFC-0011 §2; RFC-0060 §6). Owner: RFC-0011.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlanCertification {
    /// How the plan's parts are certified.
    pub status: CertStatus,
    /// The run-time values certified: the runtime may use these and nothing beyond them
    /// (RFC-0012 §10.5).
    pub runtime: CertifiedRuntime,
    /// The certificates used.
    pub relies_on: Vec<CertRef>,
}

/// How a plan's parts are certified (RFC-0060 §6). Owner: RFC-0060.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CertStatus {
    /// Every part has its certificate.
    Certified,
    /// Some part is certified by pairwise inference.
    Derived,
    /// Some part has none. The default.
    #[default]
    Uncertified,
}

/// The run-time values a plan's certification allows (RFC-0012 §10.5). Each default allows the
/// least.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CertifiedRuntime {
    /// Whether prefix reuse is certified.
    pub prefix_reuse: bool,
    /// The tiers KV may spill to.
    pub kv_spill_tiers: SmallVec<[TierKind; 3]>,
    /// The most concurrent sequences.
    pub max_concurrency: u32,
    /// The longest verify window.
    pub max_window: u16,
}

/// One certificate a plan relies on (RFC-0060 §6): its kind and its ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CertRef {
    /// The kind.
    pub kind: CertKind,
    /// The certificate's ID.
    pub id: Blake3,
}

/// The certificate kinds a plan names (RFC-0060 §6).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CertKind {
    /// An op on a device class (`OpCert`).
    Op,
    /// A wire format for a family and phase (`WireCert`, ADR-0019 decision 5).
    Wire,
    /// A family module (`FamilyCert`, RFC-0043 §9.4).
    Family,
    /// A device path (`PathCert`).
    Path,
}
