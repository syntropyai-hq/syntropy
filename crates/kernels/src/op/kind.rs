// SPDX-License-Identifier: MIT OR Apache-2.0

//! The op kinds of RFC-0040 §3.1, the parameter enums they carry, each kind's dims signature
//! (RFC-0040 §3.2) and its profiling class (RFC-0001 §5.2 `KernelClass`).

use serde::{Deserialize, Serialize};
use syntropy_core::resource::KernelClass;

pub use syntropy_core::resource::AttnKind;

/// Every kind of op a provider can implement (RFC-0040 §3.1). Owner: RFC-0040.
///
/// The list is closed per release: a family module (RFC-0043) uses only these kinds, and a
/// new kind needs a reference implementation, a ggml mapping or an explicit "native only"
/// mark, and parity tests (RFC-0040 §3.1, §11.4). It is still `#[non_exhaustive]`, as
/// RFC-0001 §5 makes every enum, because a later release may add a kind under that rule;
/// within one release no crate may use a kind that is not listed here.
///
/// Each kind has a dims signature ([`OpKind::dims`]) and, where RFC-0040 §3.1 says so, a
/// profiling class ([`OpKind::kernel_class`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OpKind {
    // ---- Weight-bound linear algebra ----
    /// Matrix-vector product, rows ≤ 16: decode, verify windows, small batches.
    Gemv,
    /// Matrix-matrix product, rows > 16: prefill chunks, large batches.
    Gemm,
    /// Fused gate and up projections with a gated activation.
    GateUpAct {
        /// The gating activation.
        act: GatedAct,
    },
    /// Router logits to top-k expert ids and weights.
    MoeRoute {
        /// How router logits become weights.
        norm: RouteNorm,
    },
    /// The residency-aware grouped expert FFN (RFC-0040 §3.4).
    MoeGrouped,
    /// Weighted sum of per-expert outputs in ascending expert index, plus shared experts.
    MoeCombine,
    // ---- Attention and sequence mixers ----
    /// At most 16 query rows per sequence against paged KV.
    AttnDecode(AttnKind),
    /// A causal chunk against paged KV; for `Bidirectional`, the whole input with no mask.
    AttnPrefill(AttnKind),
    /// Attention over a subset of pages, returning `(o, m, l)` unnormalised (RFC-0040 §3.6).
    AttnPartial(AttnKind),
    /// Exact log-sum-exp merge of partials in a fixed order (RFC-0040 §3.6).
    AttnMerge,
    /// Indexer scores and top-k block or token selection.
    AttnIndex(IndexKind),
    /// Appends K/V or latents into KV pages, with RoPE applied where the family says.
    KvWrite(AttnKind),
    /// One-token recurrent state update.
    ScanDecode(ScanKind),
    /// Chunked parallel scan over a prefill chunk.
    ScanPrefill(ScanKind),
    /// Short convolution in Mamba- and DeltaNet-class layers.
    CausalConv1d,
    // ---- Normalisation and elementwise ----
    /// RMS normalisation.
    RmsNorm,
    /// Layer normalisation.
    LayerNorm,
    /// Rotary position embedding.
    Rope(RopeKind),
    /// An ungated activation.
    Act(ActKind),
    /// Elementwise add.
    Add,
    /// Elementwise multiply.
    Mul,
    /// Multiply by a scalar.
    Scale,
    /// Hyper-connection mixing over residual streams (RFC-0043 §3.3 `HyperConn`).
    HyperMix,
    // ---- Tables, head and sampling ----
    /// Embedding and n-gram rows by id.
    RowGather,
    /// Reserved (RFC-0043 §2.4): encoder rows written over placeholder rows. Only its
    /// `kernels::reference` semantics exist in v1; no provider is asked to implement it.
    EmbedInject,
    /// Embedding epilogue: one pooled row per sequence, optional L2 norm (RFC-0043 §5.4).
    Pool(PoolKind),
    /// Final norm plus LM head: a `Gemv`/`Gemm` with a vocabulary-sized output.
    Logits,
    /// Temperature, penalties, grammar mask, logit bias.
    LogitsProcess,
    /// Top-k and top-p filtering.
    TopKTopP,
    /// Sampling with a seeded generator.
    SampleSeeded,
    /// Greedy selection.
    Argmax,
    /// Exact acceptance over a verify window or tree (RFC-0012).
    VerifyAccept,
    // ---- Data movement ----
    /// Dtype conversion, including dequantisation.
    Convert,
    /// Row gather for KV compaction, migration and prefix-cache copy-on-write.
    GatherRows,
    /// Row scatter for KV compaction, migration and prefix-cache copy-on-write.
    ScatterRows,
    /// Layer-first device pages to page-first host or NVMe records (RFC-0022 §3; RFC-0040 §5.5).
    KvPack,
    /// Page-first host or NVMe records back to layer-first device pages (RFC-0040 §5.5).
    KvUnpack,
    /// Sums per-site partial outputs in a fixed order.
    PartialReduce,
    /// First node of every step: applies posted slot-table and KV block-table deltas
    /// (RFC-0040 §3.4).
    ApplyTableDeltas,
    /// Writes the device clock into a slot of a host-mapped timing buffer (RFC-0040 §3.7).
    Timestamp,
    /// Parity taps: candidate log-probs at reference top-K ids, logsumexp, argmax, margin
    /// (RFC-0040 §3.8).
    TapGather,
    /// Parity variant only: copies tapped rows' residual stream at a layer boundary
    /// (RFC-0040 §3.8).
    HiddenTap,
}

/// The gating activation of [`OpKind::GateUpAct`] (RFC-0040 §3.1).
///
/// RFC-0040 names the type without variants. These are the gated activations RFC-0043 §3.3
/// lists for `DenseFfn`: SiLU-gated, GELU-gated and clamped SiLU-gated.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GatedAct {
    /// SiLU-gated.
    Silu,
    /// GELU-gated.
    Gelu,
    /// SiLU-gated with clamped inputs.
    ClampedSilu,
}

/// How [`OpKind::MoeRoute`] turns router logits into weights (RFC-0040 §3.1).
///
/// RFC-0040 names the type without variants. These are the two RFC-0042 §4 and RFC-0043
/// §3.3 name; per-expert bias and group-limited top-k are routing parameters, not kinds.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RouteNorm {
    /// Softmax over the router logits.
    Softmax,
    /// Sigmoid of each router logit.
    Sigmoid,
}

/// The rotary embedding of [`OpKind::Rope`] (RFC-0040 §3.1).
///
/// RFC-0040 names the type without variants. These are the RoPE variants RFC-0043 §3.3
/// lists by frequency rule; partial rotary and per-layer bases are parameters.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RopeKind {
    /// Unscaled frequencies.
    Plain,
    /// Linear position scaling.
    Linear,
    /// YaRN scaling.
    Yarn,
    /// Llama-3-style frequency scaling.
    Llama3,
}

/// The activation of [`OpKind::Act`] (RFC-0040 §3.1).
///
/// RFC-0040 names the type without variants. These are the activations RFC-0043 §3.3's FFN
/// rows name, ungated.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ActKind {
    /// SiLU.
    Silu,
    /// GELU.
    Gelu,
}

/// The pooling of [`OpKind::Pool`] (RFC-0040 §3.1; RFC-0043 §5.4).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PoolKind {
    /// Mean of each sequence's rows.
    Mean,
    /// The first row of each sequence.
    Cls,
    /// The last row of each sequence.
    LastToken,
}

/// The selection of [`OpKind::AttnIndex`] (RFC-0040 §3.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum IndexKind {
    /// Top-k blocks.
    BlockTopK,
    /// Top-k tokens.
    TokenTopK,
    /// Top-k compressed entries.
    CompressedTopK,
}

/// The recurrence of [`OpKind::ScanDecode`] and [`OpKind::ScanPrefill`] (RFC-0040 §3.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ScanKind {
    /// Gated DeltaNet.
    GatedDeltaNet,
    /// KDA.
    Kda,
    /// Mamba-2.
    Mamba2,
}

/// Dims of attention over paged GQA-class KV (RFC-0040 §3.2, `AttnDecode(Gqa)`).
const ATTN_GQA: &[&str] = &["n_q_heads", "n_kv_heads", "d_head", "page_tokens"];
/// Dims of attention over MLA latents (RFC-0040 §3.2, `AttnDecode(Mla)`).
const ATTN_MLA: &[&str] = &["n_heads", "d_latent", "d_rope", "page_tokens"];
/// Dims of attention over compressed entries.
const ATTN_COMPRESSED: &[&str] = &["n_heads", "d_entry", "page_tokens"];
/// Dims of an elementwise op over `n` elements per row.
const ELEMENTWISE: &[&str] = &["n"];
/// Dims of an op over vocabulary-sized rows.
const VOCAB: &[&str] = &["n_vocab"];

impl OpKind {
    /// The names of `ShapeSig::dims`, in order, for this kind (RFC-0040 §3.2: "op-specific
    /// static dims, in a fixed order per `OpKind`"). Every kind has between one and six.
    ///
    /// RFC-0040 §3.2 gives the signatures of `Gemv`/`Gemm`, `MoeGrouped`, `AttnDecode(Gqa)`,
    /// `AttnDecode(Mla)` and `ScanDecode`, and these are transcribed from it. The others are
    /// this crate's, built the same way from the op's description in §3.1; the reference
    /// semantics of each op (`kernels::reference`) may refine them before the list is frozen.
    #[must_use]
    pub fn dims(&self) -> &'static [&'static str] {
        match self {
            Self::Gemv | Self::Gemm | Self::GateUpAct { .. } => &["n_out", "k_in"],
            Self::MoeRoute { .. } => &["n_experts", "top_k", "hidden"],
            Self::MoeGrouped => &["n_experts", "top_k", "hidden", "inter"],
            Self::MoeCombine => &["top_k", "n_shared", "hidden"],
            Self::AttnDecode(k) | Self::AttnPrefill(k) | Self::AttnPartial(k) => attn_dims(*k),
            Self::AttnMerge => &["n_q_heads", "d_v", "n_parts"],
            Self::AttnIndex(_) => &["n_heads", "d_head", "top_k", "page_tokens"],
            Self::KvWrite(k) => match k {
                AttnKind::Mla => &["d_latent", "d_rope", "page_tokens"],
                AttnKind::Compressed => &["d_entry", "page_tokens"],
                _ => &["n_kv_heads", "d_head", "page_tokens"],
            },
            Self::ScanDecode(_) | Self::ScanPrefill(_) => &["n_heads", "d_k", "d_v"],
            Self::CausalConv1d => &["channels", "width"],
            Self::RmsNorm | Self::LayerNorm | Self::Pool(_) | Self::HiddenTap => &["hidden"],
            Self::Rope(_) => &["n_heads", "d_head", "d_rot"],
            Self::Act(_) | Self::Add | Self::Mul | Self::Scale | Self::Convert => ELEMENTWISE,
            Self::HyperMix => &["n_streams", "hidden"],
            Self::RowGather => &["table_rows", "width"],
            Self::EmbedInject | Self::GatherRows | Self::ScatterRows => &["width"],
            Self::Logits => &["n_vocab", "hidden"],
            Self::LogitsProcess
            | Self::TopKTopP
            | Self::SampleSeeded
            | Self::Argmax
            | Self::VerifyAccept => VOCAB,
            Self::KvPack | Self::KvUnpack => &["n_layers", "page_bytes"],
            Self::PartialReduce => &["n_parts", "n"],
            Self::ApplyTableDeltas => &["max_deltas"],
            Self::Timestamp => &["n_slots"],
            Self::TapGather => &["n_vocab", "top_k"],
        }
    }

    /// The class this kind profiles as (RFC-0040 §3.1; RFC-0001 §5.2 `KernelClass`), or
    /// `None` for a kind RFC-0040 §3.1 maps to no class.
    ///
    /// `Gemv`, `GateUpAct` and `Logits` profile as `DenseGemv`; `MoeGrouped` as `ExpertGemv`;
    /// `AttnDecode` as `Attention(kind)`; `RowGather` as `RowGather`; `Convert`, `GatherRows`
    /// and `ScatterRows` as `Copy`.
    #[must_use]
    pub fn kernel_class(&self) -> Option<KernelClass> {
        match self {
            Self::Gemv | Self::GateUpAct { .. } | Self::Logits => Some(KernelClass::DenseGemv),
            Self::MoeGrouped => Some(KernelClass::ExpertGemv),
            Self::AttnDecode(k) => Some(KernelClass::Attention(*k)),
            Self::RowGather => Some(KernelClass::RowGather),
            Self::Convert | Self::GatherRows | Self::ScatterRows => Some(KernelClass::Copy),
            // Listed in full, so that a new kind needs a decision here.
            Self::Gemm
            | Self::MoeRoute { .. }
            | Self::MoeCombine
            | Self::AttnPrefill(_)
            | Self::AttnPartial(_)
            | Self::AttnMerge
            | Self::AttnIndex(_)
            | Self::KvWrite(_)
            | Self::ScanDecode(_)
            | Self::ScanPrefill(_)
            | Self::CausalConv1d
            | Self::RmsNorm
            | Self::LayerNorm
            | Self::Rope(_)
            | Self::Act(_)
            | Self::Add
            | Self::Mul
            | Self::Scale
            | Self::HyperMix
            | Self::EmbedInject
            | Self::Pool(_)
            | Self::LogitsProcess
            | Self::TopKTopP
            | Self::SampleSeeded
            | Self::Argmax
            | Self::VerifyAccept
            | Self::KvPack
            | Self::KvUnpack
            | Self::PartialReduce
            | Self::ApplyTableDeltas
            | Self::Timestamp
            | Self::TapGather
            | Self::HiddenTap => None,
        }
    }
}

/// Dims of `AttnDecode`, `AttnPrefill` and `AttnPartial` for one attention kind.
fn attn_dims(kind: AttnKind) -> &'static [&'static str] {
    match kind {
        AttnKind::Mla => ATTN_MLA,
        AttnKind::Compressed => ATTN_COMPRESSED,
        _ => ATTN_GQA,
    }
}
