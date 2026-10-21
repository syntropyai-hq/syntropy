// SPDX-License-Identifier: MIT OR Apache-2.0

//! Plan deltas: what a runner adopts in place, without reloading (RFC-0001 §5.4,
//! RFC-0011 §12.3). The solver crate re-exports [`DeltaItem`] (PLC-E4-T01).

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ids::{DeviceId, NodeId, StageIx};

use super::PlanEpoch;
use super::budget::{ExpertCacheBudget, KvBudget, PrefetchWindow};
use super::cost::{BatchingLimits, PredictedCost};
use super::speculative::SpeculativeConfig;
use super::stage::HopPlan;

/// Budget and link changes between two epochs of one instance (RFC-0001 §5.4). Any other
/// change to layers, owners or home tiers is a new shard and goes through drain-and-replace
/// (RFC-0001 §7.3). Owner: RFC-0011 (RFC-0032 for `Relink` and `ExtendShard`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanDelta {
    /// The epoch the delta applies to.
    pub from: PlanEpoch,
    /// The epoch it produces.
    pub to: PlanEpoch,
    /// The changes, applied together.
    pub changes: Vec<DeltaItem>,
}

/// One change in a [`PlanDelta`] (RFC-0011 §12.3). Every variant except `ExtendShard` leaves
/// the runner's shard unchanged. `Relink` and `ExtendShard` are RFC-0032's (§4.1); the delta
/// solve never emits them. Owner: RFC-0011.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DeltaItem {
    /// A device's expert-cache budget: bytes, per-layer split, miss split.
    ExpertCache {
        /// The device.
        device: DeviceId,
        /// The new budget.
        budget: ExpertCacheBudget,
    },
    /// A stage's KV budgets.
    Kv {
        /// The stage.
        stage: StageIx,
        /// The new budgets, one per tier.
        budgets: SmallVec<[KvBudget; 3]>,
    },
    /// A prefetch window.
    Prefetch {
        /// The new window.
        window: PrefetchWindow,
    },
    /// The speculative setup; `None` turns speculation off.
    Speculative {
        /// The new setup.
        config: Option<SpeculativeConfig>,
    },
    /// The batching limits.
    Batching {
        /// The new limits.
        limits: BatchingLimits,
    },
    /// The new prediction. Always present in a delta.
    Predicted {
        /// The new prediction.
        cost: PredictedCost,
    },
    /// Same shard; a different neighbour, hop transport, relay or stage index (RFC-0032
    /// §4.4). The runner stays `Ready { degraded: Relinking }` while it applies.
    Relink {
        /// The stage's new index.
        stage: StageIx,
        /// The previous stage's node.
        prev: NodeId,
        /// The next stage's node.
        next: NodeId,
        /// The new outbound hop.
        hop: HopPlan,
    },
    /// The runner keeps its shard and stage and loads the listed stages' layers: a survivor
    /// absorbing a lost stage after a repair solve (RFC-0011 §12.4, RFC-0033 §7.2). It
    /// reports `Ready { degraded: Absorbing }` until they are resident.
    ExtendShard {
        /// The stages whose layers it takes on; layers and tiers come from the new plan.
        stages: SmallVec<[StageIx; 2]>,
    },
}
