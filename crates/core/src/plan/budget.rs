// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory budgets in a plan (RFC-0001 §5.4): the expert cache (RFC-0023 §2), KV
//! (RFC-0022 §2, §5.3) and prefetch windows (RFC-0021).

use serde::{Deserialize, Serialize};

use crate::ids::{DeviceId, StageIx, TierId};
use crate::units::Bytes;

/// An expert-cache budget on one device and tier (RFC-0001 §5.4). Owner: RFC-0023 (§2 there
/// defines [`CachePolicy`] and [`MissSplit`], which RFC-0011 emits).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExpertCacheBudget {
    /// The device the cache serves.
    pub device: DeviceId,
    /// `Vram` or `Unified` over a RAM home. Over an `Nvme` home the RAM cache is two budgets:
    /// a copyable `HostPinned` part and an in-place `HostPageable` part (RFC-0020 §2).
    pub tier: TierId,
    /// The cache's size.
    pub bytes: Bytes,
    /// Per-layer budgets. Uneven by design.
    pub per_layer: Option<Vec<Bytes>>,
    /// Admission and eviction policy plus parameters.
    pub policy: CachePolicy,
    /// Starting split of misses between in-place CPU compute and a PCIe copy to the GPU. The
    /// runtime rebalances it from measured costs.
    pub miss_split: MissSplit,
}

/// An expert cache's admission and eviction policy (RFC-0023 §2; parameter defaults in
/// RFC-0023 §15). Owner: RFC-0023.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum CachePolicy {
    /// Decayed least-frequently-used.
    DecayedLfu {
        /// Per-step decay of the frequency counts.
        decay: f32,
        /// Steps between adaptations.
        adapt_every_steps: u16,
        /// Factor by which a candidate's decayed count must exceed the resident's it replaces.
        hysteresis: f32,
        /// Most replacements per adaptation.
        max_replace: u16,
        /// Run the shadow-LRU guard, which keeps the hit rate at or above LRU's (RFC-0023 §5.3).
        guard: bool,
    },
    /// Least-recently-used: the baseline and the guard fallback.
    Lru,
    /// Fill from the profile once and never replace (deterministic runs).
    Static,
}

/// The expert-miss balancer's starting point (RFC-0023 §2, §8). Owner: RFC-0023.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MissSplit {
    /// Fraction of missed bytes copied over PCIe at batch 1.
    pub copied_share: f32,
    /// Whether the runtime refits the split.
    pub mode: MissSplitMode,
}

/// Whether the miss split is refitted at run time (RFC-0023 §2). Owner: RFC-0023.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MissSplitMode {
    /// Refitted from measured costs. The default.
    #[default]
    Fitted,
    /// Frozen for the run (deterministic runs, tests; ADR-0019 item 8).
    Fixed,
}

/// A KV budget for one stage on one tier (RFC-0001 §5.4). Owner: RFC-0022.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvBudget {
    /// The stage whose layers' KV it holds.
    pub stage: StageIx,
    /// The tier holding it.
    pub tier: TierId,
    /// Its size.
    pub bytes: Bytes,
    /// Lossless unless the plan is approximate.
    pub dtype: KvDType,
    /// Receives pages evicted from `tier`.
    pub spill: Option<TierId>,
    /// Who attends pages spilled to host RAM (RFC-0022 §5.3). Chosen by the solver from
    /// RFC-0022 §5.4's two cost terms.
    #[serde(default)]
    pub spill_compute: SpillCompute,
}

/// KV storage dtype (RFC-0022 §2). Owner: RFC-0022.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvDType {
    /// The reference's KV dtype. Exact. The default.
    #[default]
    Native,
    /// Approximate: int8 with one F16 scale per block.
    Q8 {
        /// Elements per scale.
        block: u16,
    },
    /// Approximate: int8 K and 4-bit V after a Hadamard rotation.
    K8V4 {
        /// Elements per scale.
        block: u16,
        /// Whether the rotation is applied.
        rotate: bool,
    },
}

/// Who attends KV pages spilled to host RAM (RFC-0022 §5.3). Owner: RFC-0022.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpillCompute {
    /// Host pages are attended by the CPU for passes of fewer than `max_rows` query rows,
    /// and streamed for larger passes. `max_rows` starts at the plan's T*_kv and is refined
    /// online from live timings.
    CpuSplit {
        /// The largest pass, in query rows, the CPU attends.
        max_rows: u16,
    },
    /// Host pages are always streamed to the GPU. The serde default, filling the field in
    /// records that predate it: it is the unmeasured default for MLA and every kind with
    /// `g_eff ≥ 32` (RFC-0022 §5.3), and it is correct for every kind.
    #[default]
    GpuStream,
}

/// A prefetch window between two tiers (RFC-0001 §5.4). Owner: RFC-0021 (I/O), RFC-0023
/// (expert prediction feeding it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefetchWindow {
    /// The tier read from.
    pub from: TierId,
    /// The tier written to.
    pub to: TierId,
    /// Bytes in flight at most; bounded by what `to` can hold (RFC-0001 §8.4).
    pub max_inflight: Bytes,
    /// How many layers ahead the window reads.
    pub lookahead_layers: u16,
}
