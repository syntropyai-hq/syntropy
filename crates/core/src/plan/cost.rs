// SPDX-License-Identifier: MIT OR Apache-2.0

//! The plan's prediction, batching limits and assumptions (RFC-0011 §2, §12.3).
//! Owner: RFC-0011.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ids::{DeviceId, LayerIx, LinkId, StageIx, TierId};
use crate::measured::MeasureSource;
use crate::quant::QuantScheme;
use crate::units::{Bytes, Nanos};

/// The cost model's prediction for a plan, in the units the runners' telemetry reports
/// (RFC-0011 §2; RFC-0001 §13, §14). Owner: RFC-0011.
///
/// RFC-0011 §2's `stages: Vec<StageCost>` is keyed by RFC-0053's `CostTerm`, which lives in
/// `syntropy-obs`, a crate `core` does not depend on. So [`StageCost`] is keyed by [`StageTerm`],
/// which mirrors `CostTerm` variant for variant with the same discriminants, and `obs` converts
/// between the two (RFC-0011 §2, note of 2026-10-31; PLC-E4-T01).
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
    /// The prediction per stage, in ring order. Empty in a plan serialized before PLC-E4-T01.
    #[serde(default)]
    pub stages: Vec<StageCost>,
}

/// A cost term of a stage's step time: RFC-0053 §2.2's `CostTerm`, mirrored in `core` so plans
/// can carry it (RFC-0011 §2). The discriminants are `CostTerm`'s, and `syntropy_obs` converts
/// each to and from it; a test there holds the two lists to one another. Owner: RFC-0053.
#[repr(u8)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StageTerm {
    /// Dense weights read from VRAM.
    DenseVram = 0,
    /// Dense weights read from unified memory.
    DenseUnified = 1,
    /// Dense weights computed on the host CPU.
    DenseHostCpu = 2,
    /// Expert-cache hits computed on the GPU.
    ExpertHitGpu = 3,
    /// Experts computed on the CPU, including the in-place share of misses.
    ExpertHitCpu = 4,
    /// Expert misses copied to the computing device.
    ExpertMissCopied = 5,
    /// Waiting for an NVMe demand read.
    NvmeDemandWait = 6,
    /// KV reads.
    KvRead = 7,
    /// Row lookups: embeddings and other tables.
    RowLookup = 8,
    /// Device switches within a stage.
    Switch = 9,
    /// Collectives of tensor-parallel blocks.
    Collective = 10,
    /// Sending the activation to the next stage.
    HopSend = 11,
    /// Waiting for the previous stage's activation.
    HopWait = 12,
    /// Waiting for send credit.
    CreditStall = 13,
    /// Fixed per-step cost.
    Fixed = 14,
    /// Sampling.
    Sampler = 15,
    /// Applying a constraint mask.
    ConstraintMask = 16,
    /// Drafting.
    Draft = 17,
}

impl StageTerm {
    /// Every term, in discriminant order.
    pub const ALL: [StageTerm; 18] = [
        StageTerm::DenseVram,
        StageTerm::DenseUnified,
        StageTerm::DenseHostCpu,
        StageTerm::ExpertHitGpu,
        StageTerm::ExpertHitCpu,
        StageTerm::ExpertMissCopied,
        StageTerm::NvmeDemandWait,
        StageTerm::KvRead,
        StageTerm::RowLookup,
        StageTerm::Switch,
        StageTerm::Collective,
        StageTerm::HopSend,
        StageTerm::HopWait,
        StageTerm::CreditStall,
        StageTerm::Fixed,
        StageTerm::Sampler,
        StageTerm::ConstraintMask,
        StageTerm::Draft,
    ];

    /// The term with discriminant `code`, if any.
    #[must_use]
    pub fn from_code(code: u8) -> Option<StageTerm> {
        Self::ALL.get(usize::from(code)).copied()
    }

    /// Its discriminant, `CostTerm`'s for the same term.
    #[must_use]
    pub fn code(self) -> u8 {
        self as u8
    }
}

/// One stage's predicted step (RFC-0011 §2). Owner: RFC-0011.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageCost {
    /// The stage.
    pub stage: StageIx,
    /// Critical-path attributions that sum to the stage's step time: where sites run in parallel
    /// (a MoE block's GPU, CPU and NVMe work, C8; KV's GPU and CPU-split parts, C7), the block's
    /// time goes to the binding site's term.
    pub terms: SmallVec<[(StageTerm, Nanos); 18]>,
    /// Each non-binding site of every parallel block, under its own term, summed over the step's
    /// blocks; not part of the step time.
    pub overlapped: SmallVec<[(StageTerm, Nanos); 4]>,
    /// The largest predicted per-layer CPU in-place time; RFC-0040 derives
    /// `kernels.spin_kernel_max_us` from it.
    pub cpu_layer_max: Nanos,
    /// Predicted read bytes per token per tier (SC-11).
    pub bytes: SmallVec<[(TierId, Bytes); 6]>,
    /// Predicted peak resident bytes per tier, by purpose.
    pub resident: SmallVec<[ResidentBytes; 6]>,
    /// One prefill chunk of [`BatchingLimits::prefill_chunk`] tokens.
    pub prefill_per_chunk: Nanos,
}

impl StageCost {
    /// The step time: the sum of [`terms`](Self::terms), saturating.
    #[must_use]
    pub fn step(&self) -> Nanos {
        Nanos(
            self.terms
                .iter()
                .fold(0u64, |sum, (_, t)| sum.saturating_add(t.0)),
        )
    }
}

/// A tier's predicted peak resident bytes for one stage (RFC-0011 §2). Owner: RFC-0011.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidentBytes {
    /// The tier.
    pub tier: TierId,
    /// Held before this plan: other instances, the old epoch.
    pub before: Bytes,
    /// Held under this plan.
    pub after: Bytes,
    /// `after`, by purpose.
    pub by_purpose: SmallVec<[(MemoryPurpose, Bytes); 8]>,
}

/// What resident bytes are for: RFC-0001 §9.1 and §9.3's ten purposes, the plan's view of a
/// tier's budget. Owner: RFC-0001. The tier manager's arena purposes (RFC-0020 §2) are a finer
/// taxonomy; whoever fills [`ResidentBytes::by_purpose`] from arenas maps them onto these.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryPurpose {
    /// The device context's own reserve.
    ContextReserve,
    /// Activations, workspaces and captured graphs.
    Scratch,
    /// Dense weights.
    DenseWeights,
    /// KV in use.
    Kv,
    /// KV reserved for admission.
    KvReserved,
    /// The expert cache.
    ExpertCache,
    /// Staging buffers.
    Staging,
    /// Network buffer pools.
    NetPools,
    /// Activation history for replay.
    History,
    /// Headroom kept free.
    Headroom,
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
