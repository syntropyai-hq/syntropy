// SPDX-License-Identifier: MIT OR Apache-2.0

//! The policy objects (RFC-0001 §11.3), with the fields RFC-0054 §4 adds. Owner: RFC-0054.
//!
//! - This module: [`Objective`], [`ExactnessPolicy`] with [`ApproxKnobs`] and [`ApproxParams`],
//!   the [`ApproxFlags`] alias and [`Fidelity`].
//! - [`resource`]: [`ResourcePolicy`], the node's own policy, and [`ResourceOverlay`], an
//!   instance's narrowing of it, with [`NetworkScope`] (RFC-0054 §4.3, §4.4).
//! - [`instance`]: [`InstancePolicy`] and its parts, with [`InstancePolicy::canonical_bytes`]
//!   (RFC-0054 §4.5, §6.2, §6.3).
//! - [`value`]: the field types `config` parses text into and validates (OQ-PRD-27):
//!   [`Reserve`], [`Fraction`], the selectors, [`KeyPath`] and [`ConfigValue`].
//!
//! Every type is plain data. Fields added after RFC-0001 carry serde defaults, so a record
//! written before them still decodes.

pub mod instance;
pub mod resource;
pub mod value;

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::plan::{KvDType, WireDType};
use crate::quant::QuantScheme;
use crate::units::Nanos;

pub use instance::*;
pub use resource::*;
pub use value::*;

/// What the solver and scheduler optimise (RFC-0001 §11.3, RFC-0054 §4.1). Default: `Latency`
/// (D2). Owner: RFC-0054 (type), RFC-0011 (objective function), RFC-0012 (scheduling
/// behaviour).
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
        /// The sequences the plan is sized for; `None` lets the solver choose (RFC-0054 §4.1).
        /// Unset, it is absent from the canonical form only: a `PlacementPlan` carries this
        /// type over the IPC in postcard (RFC-0001 §18.20).
        #[serde(default, skip_serializing_if = "crate::canon::absent_if_none")]
        target_concurrency: Option<u16>,
        /// Admission rejects a request whose predicted TTFT is above this (RFC-0054 §4.1).
        #[serde(default, skip_serializing_if = "crate::canon::absent_if_none")]
        ttft_ceiling: Option<Nanos>,
    },
}

/// Exact by default (D3, ADR-0019). `Approximate` lists the knobs that may be used
/// (RFC-0001 §11.3) and their parameters (RFC-0054 §4.2). Owner: RFC-0054.
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
        /// Each knob's parameters; a knob without them takes RFC-0042's default.
        #[serde(default)]
        params: ApproxParams,
    },
}

bitflags! {
    /// The approximate knobs, named per ADR-0019 §6, at RFC-0001 §11.3's bit positions. Each
    /// knob carries a published KL and top-1 cost. RFC-0054 §4.2 gives each knob's scope of
    /// effect: [`ApproxKnobs::PER_SEQUENCE`] or [`ApproxKnobs::PER_INSTANCE`].
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct ApproxKnobs: u32 {
        /// Skip low-weight routed experts. Per sequence.
        const EXPERT_SKIP        = 1 << 0;
        /// Defer missed experts to a later step. Per sequence.
        const EXPERT_DEFER       = 1 << 1;
        /// Compute missed experts at a lower precision. Per instance.
        const MISS_LOW_PRECISION = 1 << 2;
        /// Route to fewer experts than the model's top-k. Per sequence.
        const TOPK_REDUCE        = 1 << 3;
        /// Store KV in a lossy dtype. Per instance.
        const KV_LOSSY           = 1 << 4;
        /// Send activations in a compressed wire dtype. Per instance.
        const WIRE_COMPRESS      = 1 << 5;
        /// Approximate attention. Per instance.
        const APPROX_ATTENTION   = 1 << 6;
        /// Lenient speculative acceptance. Per sequence.
        const SPEC_LENIENT       = 1 << 7;
    }
}

impl ApproxKnobs {
    /// The knobs a single sequence can use without changing the plan (RFC-0054 §4.2).
    pub const PER_SEQUENCE: ApproxKnobs = ApproxKnobs::EXPERT_SKIP
        .union(ApproxKnobs::EXPERT_DEFER)
        .union(ApproxKnobs::TOPK_REDUCE)
        .union(ApproxKnobs::SPEC_LENIENT);

    /// The knobs that change the plan, and so every sequence of the instance (RFC-0054 §4.2).
    pub const PER_INSTANCE: ApproxKnobs = ApproxKnobs::MISS_LOW_PRECISION
        .union(ApproxKnobs::KV_LOSSY)
        .union(ApproxKnobs::WIRE_COMPRESS)
        .union(ApproxKnobs::APPROX_ATTENTION);
}

/// The knobs that changed a token, on `TokenEntry::approx` (RFC-0001 §5.8): the same bit
/// layout as [`ApproxKnobs`] (RFC-0054 §4.2, amendment A2).
pub type ApproxFlags = ApproxKnobs;

/// Each knob's parameters (RFC-0054 §4.2). The parameter types belong to RFC-0042.
///
/// Only the knobs whose parameter types `core` already holds have a field here; each M3 knob
/// task adds its own (`expert_skip`, `expert_defer`, `topk_reduce`, `approx_attention`,
/// `spec_lenient`). `None` takes RFC-0042's default for the knob. Unset fields are left out of
/// the canonical form, so adding a field leaves earlier policies' digests unchanged. Every other
/// format writes them: a `PlacementPlan` carries this type over the IPC in postcard, which cannot
/// skip a field (RFC-0001 §3.5, §18.20; [`canon::absent_if_none`](crate::canon::absent_if_none)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ApproxParams {
    /// `MISS_LOW_PRECISION`: the scheme of the lower-precision expert copies.
    #[serde(default, skip_serializing_if = "crate::canon::absent_if_none")]
    pub miss_low_precision: Option<QuantScheme>,
    /// `KV_LOSSY`: the KV dtype.
    #[serde(default, skip_serializing_if = "crate::canon::absent_if_none")]
    pub kv_lossy: Option<KvDType>,
    /// `WIRE_COMPRESS`: the activation wire dtype.
    #[serde(default, skip_serializing_if = "crate::canon::absent_if_none")]
    pub wire_compress: Option<WireDType>,
}

/// What a request asks for, or what an instance gives a request that states none
/// (RFC-0001 §11.3, RFC-0054 §7). Default: `Exact` (D3). Owner: RFC-0054.
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum Fidelity {
    /// Output within ADR-0019's tolerances.
    #[default]
    Exact,
    /// Approximate, if the instance policy and the cluster also allow it (RFC-0054 §7).
    Approximate,
}
