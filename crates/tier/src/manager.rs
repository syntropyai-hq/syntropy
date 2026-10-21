// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `TierManager` façade and the commands the housekeeping thread posts to each shard
//! (RFC-0020 §1, §3, §12). This task defines [`TierCommand`]; the façade, `TierShard` and the
//! SPSC command ring come in TIER-E1-T02 and TIER-E1-T06.

#![forbid(unsafe_code)]

use std::ops::Range;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use syntropy_core::{Bytes, PlanDelta};

use crate::arena::ArenaRef;
use crate::pressure::PressureLevel;

/// A command from the housekeeping thread to a shard's dispatcher, applied between steps
/// (RFC-0020 §1, §3, §12).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TierCommand {
    /// A pinning step finished for `range` of the arena (RFC-0020 §6).
    PinStep {
        /// The arena.
        arena: ArenaRef,
        /// Byte range within the arena.
        range: Range<u64>,
    },
    /// Unpin `range` of the arena.
    Unpin {
        /// The arena.
        arena: ArenaRef,
        /// Byte range within the arena.
        range: Range<u64>,
    },
    /// Unmap segments of an elastic arena (RFC-0020 §8.4).
    ShrinkElastic {
        /// The arena.
        arena: ArenaRef,
        /// Segments to unmap.
        segments: u32,
    },
    /// Map segments into an elastic arena (RFC-0020 §8.4).
    GrowElastic {
        /// The arena.
        arena: ArenaRef,
        /// Segments to map.
        segments: u32,
    },
    /// Give memory back under pressure (RFC-0020 §11.4).
    Yield {
        /// The pressure level that triggered the yield.
        level: PressureLevel,
        /// Bytes to give back.
        reclaim: Bytes,
    },
    /// Adopt a plan delta in place (RFC-0001 §5.4, RFC-0020 §12).
    ApplyDelta {
        /// The delta, shared with the other shards.
        #[serde(with = "arc_delta")]
        delta: Arc<PlanDelta>,
    },
}

/// Serde for `Arc<PlanDelta>` by value, without serde's `rc` feature. Each decode makes a new
/// `Arc`; sharing is not preserved across a round trip.
mod arc_delta {
    use std::sync::Arc;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use syntropy_core::PlanDelta;

    pub(super) fn serialize<S: Serializer>(d: &Arc<PlanDelta>, s: S) -> Result<S::Ok, S::Error> {
        PlanDelta::serialize(d, s)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Arc<PlanDelta>, D::Error> {
        PlanDelta::deserialize(d).map(Arc::new)
    }
}
