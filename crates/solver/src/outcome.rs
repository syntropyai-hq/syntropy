// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a solve and the evaluator return (RFC-0011 §2, §11.3, §14, §17).

use serde::{Deserialize, Serialize};
use syntropy_core::plan::{
    Assumption, DegradedReason, PlacementPlan, PlanEpoch, PredictedCost, SolveStats,
};
use syntropy_core::policy::PlacementPin;

/// The result of a slow-phase solve (RFC-0011 §2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolveOutcome {
    /// The new plan. `None` when no plan fits (see `infeasible`) or the current plan stays.
    pub plan: Option<PlacementPlan>,
    /// A re-plan did not pass hysteresis and payback (RFC-0011 §12.2), so the current plan
    /// stays.
    pub keep_current: bool,
    /// Why the plan is what it is (RFC-0011 §14).
    pub explanation: PlanExplanation,
    /// Why no plan fits, when none does (RFC-0011 §17).
    pub infeasible: Option<InfeasibilityReport>,
    /// Which path won, its gap, nodes, iterations and wall time.
    pub stats: SolveStats,
}

/// Why a plan is what it is (RFC-0011 §14). Owner: RFC-0011.
///
/// RFC-0011 §14 also lists the headline, shadow prices and the reasons per group, device and
/// stage, the bottlenecks and the alternatives. PLC-E6-T06 adds them on one node, and
/// PLC-E6-T07 across the cluster.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlanExplanation {
    /// The plan explained.
    pub epoch: PlanEpoch,
    /// The unmeasured inputs and their effect (RFC-0011 §12.3).
    pub assumptions: Vec<Assumption>,
}

/// Why no plan fits (RFC-0011 §14.1, §17). Owner: RFC-0011.
///
/// RFC-0011 §14.1 also lists the shortfall per node and tier, the binding constraints, the
/// smallest fitting sibling and the extra memory that would make the variant fit. PLC-E4-T02
/// adds the shortfall per tier for one device, and PLC-E6-T08 the rest.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InfeasibilityReport {
    /// Pins that cannot all hold; never dropped silently (ADR-0008 item 8).
    pub conflicting_pins: Vec<PlacementPin>,
    /// A degraded plan that would fit, such as dense bytes on NVMe.
    pub degraded_option: Option<DegradedReason>,
}

/// What the evaluator finds for a plan (RFC-0011 §11.3): every term of its prediction,
/// recomputed from the plan's integral assignments. Owner: RFC-0011.
///
/// PLC-E4-T06 adds each tier's predicted peak residency on every node and the violations that
/// rounding left.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Evaluation {
    /// The plan's prediction.
    pub predicted: PredictedCost,
}
