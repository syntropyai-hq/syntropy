// SPDX-License-Identifier: MIT OR Apache-2.0

//! The learned workload statistics a solve reads instead of priors once they exist
//! (RFC-0011 §2). RFC-0023 §13 owns the curves' contents and produces them per MoE layer.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use syntropy_core::{Bytes, LayerIx, MeasureSource, Measured, TierId};

/// Learned statistics the solver uses instead of priors once they exist (RFC-0011 §2).
///
/// An empty list means nothing is learned yet, and the solver uses the priors RFC-0011 §2 gives
/// for each field. RFC-0011 §2 also lists `acceptance`, `context` and `concurrency`, RFC-0012
/// §11's `AcceptanceReport`, `ContextStats` and `ConcurrencyStats`; PLC-E9-T19 defines those
/// types and adds the fields. Until then the solver uses their priors: acceptance 0.7
/// (ADR-0017 item 5), context p90 `solver.context_target` and an uncached prompt p90 of 4096
/// tokens (RFC-0011 §7.1). Build one from [`Default`] and set its fields.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkloadStats {
    /// Per MoE layer: the hit fraction of routed reads the runtime cache policy achieves as a
    /// function of resident bytes in fast memory, H_l(c), where c counts the VRAM and RAM caches
    /// as exclusive sets for NVMe-homed experts (RFC-0023 §3, §13). Prior: H_l(c) = c / pool_l
    /// (uniform routing, cost model §3.3).
    pub hit_curves: Vec<HitCurve>,
    /// Per MoE layer: expected distinct experts for one pass of T tokens (RFC-0023 §13). Prior:
    /// the uniform-routing formula E · (1 − (1 − k/E)^T) (cost model §3.9).
    pub union: Vec<UnionCurve>,
    /// Measured hit rate per expert-cache tier and layer (`ExpertCacheDigest::tier_hit`),
    /// compared with the curve at that tier's budget (RFC-0011 §12.1; RFC-0010 §7.1).
    pub tier_hit: Vec<(TierId, LayerIx, Measured<f32>)>,
}

/// H_l(c), one MoE layer's hit curve: the hit fraction of the layer's routed reads the cache
/// policy achieves with c bytes of the layer's experts resident in fast memory (VRAM ∪ RAM
/// cache, exclusive sets; RFC-0023 §3, §13). Concave, piecewise linear and non-decreasing, with
/// H(0) = 0 and H(pool) = 1. Owner: RFC-0023 §13.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HitCurve {
    /// The layer.
    pub layer: LayerIx,
    /// The bytes of all the layer's experts, where H reaches 1.
    pub pool: Bytes,
    /// The curve's breakpoints: resident bytes and the hit fraction there.
    pub points: SmallVec<[(Bytes, f32); 9]>,
    /// Where the curve came from: `Prior` until a runner reports it.
    pub source: MeasureSource,
}

/// U_l(T), one MoE layer's union curve: the expected distinct experts of the layer one pass of
/// T tokens touches (RFC-0023 §13). Owner: RFC-0023 §13.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnionCurve {
    /// The layer.
    pub layer: LayerIx,
    /// E_l, the layer's routed experts.
    pub experts: u16,
    /// k_l, the experts each token selects.
    pub top_k: u16,
    /// T consecutive tokens of one sequence: verify windows and drafts. T ∈ {1, 2, 4, …, 128}.
    pub window: SmallVec<[(u16, f32); 8]>,
    /// T tokens from different sequences: micro-batches in throughput mode.
    pub batch: SmallVec<[(u16, f32); 8]>,
    /// Where the curve came from.
    pub source: MeasureSource,
}
