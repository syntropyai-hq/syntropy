// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a slow-phase solve reads (RFC-0011 §2).

use serde::{Deserialize, Serialize};
use syntropy_core::Nanos;
use syntropy_core::plan::PlacementPlan;
use syntropy_core::policy::InstancePolicy;
use syntropy_core::replan::ReplanReason;
use syntropy_core::resource::ResourceGraph;
use syntropy_manifest::census::ModelManifest;

use crate::workload::WorkloadStats;

/// Everything a slow-phase solve reads, borrowed from the coordinator's `ClusterState`
/// (RFC-0011 §2).
#[derive(Clone, Debug)]
pub struct SolveInput<'a> {
    /// The cluster's resources.
    pub graph: &'a ResourceGraph,
    /// The model's census.
    pub manifest: &'a ModelManifest,
    /// The instance's policy: objective, exactness, pins and overrides (RFC-0001 §11.3).
    pub policy: &'a InstancePolicy,
    /// The plan in force; present on re-plans.
    pub current: Option<&'a PlacementPlan>,
    /// Learned hit curves, union curves and tier hit rates.
    pub workload: &'a WorkloadStats,
    /// Why the solve runs.
    pub reason: SolveReason,
    /// The solve's work and time limits.
    pub limits: SolveLimits,
}

/// Why a solve runs (RFC-0011 §2, §12.1). RFC-0011 §2 also lists `Repair { constraints }`,
/// which carries RFC-0033's `RepairConstraints`; PLC-E4-T15 adds it with `solve_repair`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolveReason {
    /// The model is loading (`EnsureInstance`).
    Load,
    /// The coordinator asked for a new plan (RFC-0033 §6.1).
    Replan(ReplanReason),
    /// A budget-only change, solved in place (RFC-0011 §12.3).
    Delta,
}

/// The default of `solver.max_nodes`: candidate nodes kept after scoring (RFC-0011 §16).
pub const MAX_NODES_DEFAULT: u16 = 8;

/// The default of `solver.safety_time_limit`: 2 s (RFC-0011 §16; SC-31).
pub const SAFETY_TIME_LIMIT_DEFAULT: Nanos = Nanos(2_000_000_000);

/// The default of `solver.mip_rel_gap`: 1% (RFC-0011 §16).
pub const MIP_REL_GAP_DEFAULT: f64 = 0.01;

/// A solve's limits (RFC-0011 §2, §9.2, §16). The defaults are RFC-0011 §16's.
///
/// Spike S1 sets the two deterministic work limits so that they bind at no more than half of
/// `safety_time_limit` (OQ-PL-01). Until it does, their defaults are the type's maximum, no
/// limit. Only the MILP reads them, and no build has the MILP until §9.5's gate passes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolveLimits {
    /// Candidate nodes kept after scoring: `solver.max_nodes`.
    pub max_nodes: u16,
    /// MIP branch-and-bound nodes: `solver.mip_node_limit`.
    pub mip_node_limit: u32,
    /// Simplex iterations: `solver.simplex_iteration_limit`.
    pub simplex_iteration_limit: u64,
    /// The wall-clock safety net: `solver.safety_time_limit`. A repair solve's `Resolve` gets
    /// what remains of RFC-0011 §12.4's 2 s.
    pub safety_time_limit: Nanos,
    /// The relative gap the MILP stops at: `solver.mip_rel_gap`.
    pub mip_rel_gap: f64,
}

impl Default for SolveLimits {
    fn default() -> SolveLimits {
        SolveLimits {
            max_nodes: MAX_NODES_DEFAULT,
            mip_node_limit: u32::MAX,
            simplex_iteration_limit: u64::MAX,
            safety_time_limit: SAFETY_TIME_LIMIT_DEFAULT,
            mip_rel_gap: MIP_REL_GAP_DEFAULT,
        }
    }
}
