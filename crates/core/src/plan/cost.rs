// SPDX-License-Identifier: MIT OR Apache-2.0

//! The plan's prediction, batching limits and assumptions (RFC-0011 §2, §12.3).
//! Owner: RFC-0011.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ids::{DeviceId, LayerIx, LinkId, TierId};
use crate::measured::MeasureSource;
use crate::quant::QuantScheme;
use crate::units::Nanos;

/// The cost model's prediction for a plan, in the units the runners' telemetry reports
/// (RFC-0011 §2; RFC-0001 §13, §14). Owner: RFC-0011.
///
/// RFC-0011 §2 also gives it `stages: Vec<StageCost>`, keyed by RFC-0053's `CostTerm`.
/// `CostTerm` lives in `syntropy-obs`, which `core` does not depend on, so the per-stage
/// breakdown is left to PLC-E4-T01, which adds it with a serde default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictedCost {
    /// Per accepted token, speculation included.
    pub t_tok: Nanos,
    /// One ring traversal at the plan's verify window.
    pub t_pass: Nanos,
    /// Expected tokens per pass, `(1 − a^(γ+1)) / (1 − a)`.
    pub tokens_per_pass: f32,
    /// TTFT per uncached prompt length in tokens; by default 1K, 4K and 32K.
    pub ttft: SmallVec<[(u32, Nanos); 4]>,
    /// TTFT at the cap's reference length L_ttft (RFC-0011 §7.1).
    pub ttft_ref: (u32, Nanos),
    /// The TTFT cap the solve was held to. `None` in M1 (reported only) and in throughput
    /// mode.
    pub ttft_cap: Option<Nanos>,
    /// Present when the ring has a `WiFi` or `Overlay` edge.
    pub jitter: Option<Jitter>,
    /// The longest context the plan holds KV for (RFC-0011 §11.2).
    pub context_length: u32,
    /// t_pass(T) for T = 1 ..= `SpeculativeConfig::max_window` + 1, for RFC-0012's window
    /// controller.
    pub pass_by_window: SmallVec<[(u16, Nanos); 9]>,
    /// Tokens per second at the plan's batching, in throughput mode.
    pub throughput: Option<f32>,
}

/// The jitter of a ring with a wireless or overlay hop (RFC-0011 §2, constraint C14).
/// Owner: RFC-0011.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Jitter {
    /// Predicted stall rate.
    pub stall_rate: f32,
    /// Predicted p99 token time.
    pub t_tok_p99: Nanos,
}

/// Sequence, chunk and micro-batch limits (RFC-0011 §2). Owner: RFC-0011.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchingLimits {
    /// KV admission ceiling for the instance.
    pub max_sequences: u32,
    /// Tokens per prefill chunk (RFC-0011 §11.2).
    pub prefill_chunk: u32,
    /// Throughput mode: sequences per stage step. 1 in latency mode.
    pub micro_batch: u16,
    /// Captured prefill buckets (RFC-0012 §2).
    pub prefill_ladder: SmallVec<[u32; 5]>,
}

/// A solver input that was not measured (RFC-0011 §12.3). Owner: RFC-0011.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Assumption {
    /// Which input.
    pub input: AssumedInput,
    /// `Prior` or `Derived`.
    pub source: MeasureSource,
    /// Change in predicted t_tok if the input were 20% worse.
    pub sensitivity: Nanos,
}

/// The input an [`Assumption`] names (RFC-0011 §12.3, which lists these as examples).
/// Owner: RFC-0011.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssumedInput {
    /// A quant kernel's effective bandwidth on a device reading a tier.
    QuantBw {
        /// The computing device.
        device: DeviceId,
        /// The tier it reads.
        tier: TierId,
        /// The quant scheme.
        scheme: QuantScheme,
    },
    /// A MoE layer's hit curve.
    HitCurve {
        /// The layer.
        layer: LayerIx,
    },
    /// A link's p99 latency.
    LinkP99 {
        /// The link.
        link: LinkId,
    },
}
