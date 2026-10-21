// SPDX-License-Identifier: MIT OR Apache-2.0

//! The policy objects a [`crate::plan::PlacementPlan`] records (RFC-0001 §11.3).
//!
//! Only the two a plan carries are here, in RFC-0001 §11.3's shape: [`Objective`] and
//! [`ExactnessPolicy`] with its [`ApproxKnobs`]. RFC-0054 §4 owns them; the task that builds
//! the rest of the policy objects (PROD-E1-T04) adds fields and variants with serde defaults.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

/// What the solver and scheduler optimise (RFC-0001 §11.3). Default: `Latency` (D2).
/// Owner: RFC-0054.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Objective {
    /// Minimise t_tok, then TTFT.
    #[default]
    Latency,
    /// Maximise the sum of tokens per second, subject to a per-sequence floor.
    Throughput {
        /// The slowest rate one sequence may see, in tokens per second.
        per_sequence_floor_tps: f32,
    },
}

/// Exact by default (D3, ADR-0019). `Approximate` lists the knobs that may be used
/// (RFC-0001 §11.3). Owner: RFC-0054.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExactnessPolicy {
    /// Output matches the reference within ADR-0019's tolerances.
    #[default]
    Exact,
    /// The listed knobs may be used.
    Approximate {
        /// The knobs in effect.
        knobs: ApproxKnobs,
    },
}

bitflags! {
    /// The approximate knobs, named per ADR-0019 §6, at RFC-0001 §11.3's bit positions. Each
    /// knob carries a published KL and top-1 cost.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct ApproxKnobs: u32 {
        /// Skip low-weight routed experts.
        const EXPERT_SKIP        = 1 << 0;
        /// Defer missed experts to a later step.
        const EXPERT_DEFER       = 1 << 1;
        /// Compute missed experts at a lower precision.
        const MISS_LOW_PRECISION = 1 << 2;
        /// Route to fewer experts than the model's top-k.
        const TOPK_REDUCE        = 1 << 3;
        /// Store KV in a lossy dtype.
        const KV_LOSSY           = 1 << 4;
        /// Send activations in a compressed wire dtype.
        const WIRE_COMPRESS      = 1 << 5;
        /// Approximate attention.
        const APPROX_ATTENTION   = 1 << 6;
        /// Lenient speculative acceptance.
        const SPEC_LENIENT       = 1 << 7;
    }
}
