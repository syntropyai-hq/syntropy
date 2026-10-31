// SPDX-License-Identifier: MIT OR Apache-2.0

//! The placement solver: filter-then-score front end, the DP with greedy in-node placement,
//! per-request routing and plan diffing.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0011. Like the rest of the solver it is pure: no
//! I/O, clocks or threads (RFC-0001 §4.2 rule 5). It holds:
//!
//! - RFC-0011 §2's inputs and outputs: [`SolveInput`], [`SolveLimits`], [`WorkloadStats`] with
//!   its [`HitCurve`]s and [`UnionCurve`]s, and [`SolveOutcome`];
//! - the plan's prediction and RFC-0011 §2's plan additions, which live in `core` beside
//!   `PlacementPlan` and are re-exported here: [`PredictedCost`] with its per-stage
//!   [`StageCost`], [`BatchingLimits`], [`PlanDelta`], [`Assumption`] and [`PlacementPlanExt`];
//! - the entry points [`solve`], [`solve_delta`], [`route`] and [`evaluate`]. Their bodies are
//!   not built yet: each returns [`SolveError::NotBuilt`] with the task that builds it;
//! - the collective cost model ([`collective`], RFC-0032 §5.3–§5.5), which prices all-reduces
//!   for tensor-parallel placement nodes.

#![forbid(unsafe_code)]

pub mod collective;
mod error;
mod input;
mod outcome;
mod route;
mod workload;

pub use error::{Entry, SolveError};
pub use input::{
    MAX_NODES_DEFAULT, MIP_REL_GAP_DEFAULT, SAFETY_TIME_LIMIT_DEFAULT, SolveInput, SolveLimits,
    SolveReason,
};
pub use outcome::{Evaluation, InfeasibilityReport, PlanExplanation, SolveOutcome};
pub use route::{RouteDecision, RouteRequest, RoutingView};
pub use workload::{HitCurve, UnionCurve, WorkloadStats};

pub use syntropy_core::plan::{
    AssumedInput, Assumption, BatchingLimits, CertKind, CertRef, CertStatus, CertifiedRuntime,
    DegradedReason, DeltaItem, Jitter, MemoryPurpose, PlacementPlanExt, PlanCertification,
    PlanDelta, PredictedCost, ResidentBytes, SolvePath, SolveStats, StageCost, StageTerm,
    TransitionStrategy,
};

use syntropy_core::plan::PlacementPlan;

/// Solves a placement: the slow phase of RFC-0011 §1, run by the coordinator between requests.
///
/// # Errors
///
/// [`SolveError::NotBuilt`] for now. PLC-E4-T02 builds the M0 rule for one device, and
/// PLC-E4-T07 and PLC-E4-T08 the DP.
pub fn solve(input: &SolveInput<'_>) -> Result<SolveOutcome, SolveError> {
    let _ = input;
    Err(SolveError::NotBuilt {
        entry: Entry::Solve,
        task: "PLC-E4-T02",
    })
}

/// Re-optimises the budgets of `current` with its structure fixed (RFC-0011 §12.3): KV and
/// expert-cache shares, miss splits, speculative settings and batching. `Ok(None)` when the
/// delta solve cannot improve on `current`.
///
/// # Errors
///
/// [`SolveError::NotBuilt`] for now. PLC-E6-T03 builds the delta solve.
pub fn solve_delta(
    input: &SolveInput<'_>,
    current: &PlacementPlan,
) -> Result<Option<PlanDelta>, SolveError> {
    let _ = (input, current);
    Err(SolveError::NotBuilt {
        entry: Entry::SolveDelta,
        task: "PLC-E6-T03",
    })
}

/// Routes one request to an instance, on the ingress node, from its replica of cluster state
/// (RFC-0011 §13). It never moves weights.
///
/// # Errors
///
/// [`SolveError::NotBuilt`] for now. PLC-E6-T19 builds routing between one node's instances,
/// and PLC-E6-T05 across the cluster.
pub fn route(view: &RoutingView<'_>, request: &RouteRequest) -> Result<RouteDecision, SolveError> {
    let _ = (view, request);
    Err(SolveError::NotBuilt {
        entry: Entry::Route,
        task: "PLC-E6-T19",
    })
}

/// Recomputes every term of `plan`'s prediction from its integral assignments, with the
/// coefficients the DP and the MILP use (RFC-0011 §11.3). The evaluator is the single source of
/// a plan's numbers.
///
/// # Errors
///
/// [`SolveError::NotBuilt`] for now. PLC-E4-T02 builds the single-tier evaluator for one
/// device, and PLC-E4-T06 the full one.
pub fn evaluate(plan: &PlacementPlan, input: &SolveInput<'_>) -> Result<Evaluation, SolveError> {
    let _ = (plan, input);
    Err(SolveError::NotBuilt {
        entry: Entry::Evaluate,
        task: "PLC-E4-T02",
    })
}

#[cfg(test)]
mod tests;
