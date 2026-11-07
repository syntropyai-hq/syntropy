// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `qwen_moe` family: GQA decoders whose every FFN is a routed MoE with softmax top-k
//! routing and no shared expert (RFC-0043 §5.5; ADR-0003 decisions 1, 2 and 4).
//!
//! It accepts Qwen3-MoE checkpoints, such as Qwen3-30B-A3B, from GGUF or safetensors, and builds
//! every layer from the `Gqa` and `Moe` blocks, with `Embed` in the prologue and `LmHead` in the
//! epilogue (RFC-0043 §3).
//!
//! - **Attention is `llama`'s.** The attention block, RoPE, tied embeddings and their refusals
//!   are those of `llama`'s Qwen3 variant: QK-norm, no projection bias. This module reads them
//!   with `llama::parse_attention`, classifies them with `llama::attention_tables`, and gets
//!   the block parameters from [`llama::gqa_params`], so the two families cannot drift apart.
//! - **Hyperparameters.** The routed experts (`num_experts`), the experts per token
//!   (`num_experts_per_tok`, at most `num_experts`), each expert's intermediate dim
//!   (`moe_intermediate_size`) and whether the top-k weights are renormalised
//!   (`norm_topk_prob`), read the same way from GGUF and `config.json`. `config.json` leaves
//!   `norm_topk_prob` false when it is absent, as the publisher's configuration class does. GGUF
//!   states it only as `{arch}.expert_weights_norm`, which Qwen3-MoE conversions do not write;
//!   such a file reads as renormalising, the setting every published Qwen3-MoE checkpoint uses.
//! - **Fail closed.** A layout the blocks cannot compute is `HyperParamInvalid`, never ignored:
//!   dense layers among the MoE ones (`decoder_sparse_step` other than 1, or a non-empty
//!   `mlp_only_layers`), shared experts, routing other than softmax, and everything `llama`
//!   refuses (a sliding window in effect, partial rotary, an activation other than SiLU, …).
//! - **Classification.** safetensors stores one tensor per expert and projection
//!   (`mlp.experts.{E}.gate_proj.weight`); GGUF stores each projection of every expert of a
//!   layer in one tensor, the expert index outermost (`blk.{L}.ffn_gate_exps.weight`), which the
//!   census slices (RFC-0041 §2.1). The router is `mlp.gate.weight` or `ffn_gate_inp.weight`.
//!   The pre-FFN norm is a tensor of the `Moe` block, so it joins the layer's router group
//!   (RFC-0041 §3.1). An expert index at or past `num_experts`, a wrong shape, and any tensor
//!   with no rule, such as a shared expert's or a router bias, are `Unknown`.
//! - **KV.** `llama`'s: one `PerToken` entry of `2 · n_kv_heads · d_head` elements per layer, in
//!   the dtype RFC-0042 §4.2's rule gives for the source (COMP-E13-T06).
//!
//! The chat format is COMP-E17-T04's. [`fixture_spec`] gives the `FamilySpec` the synthetic
//! fixtures are generated from (RFC-0043 §9.1).

use std::sync::OnceLock;

use syntropy_kernels::RopeKind;
use syntropy_manifest::hparams::{HParamValue, HyperParams, N_EXPERTS, TOP_K};
use syntropy_manifest::kv::{KvFormula, KvSource};

use super::llama::{self, Variant, attention_tables, float, gguf, refuse, rule, st, to_f32};
use super::{Family, FamilyError, Key, MetaView};
use crate::blocks::MoeParams;
use crate::classify::{
    BlockSel, ClassRule, ClassSel, Classification, Matcher, RuleTable, ShapeDim,
};
use crate::ir::{Block, OpSet, TensorRole};
use crate::spec::{AttentionSpec, FamilySpec, LayerKind, MoeSpec, VariantSpec};

/// The family tag (RFC-0043 §5.1), frozen.
pub const TAG: &str = "qwen_moe";

/// The module path the frozen list binds [`TAG`] to.
pub const MODULE: &str = module_path!();

/// The architecture names it accepts: the GGUF `general.architecture` value, the `config.json`
/// `architectures` entry and its `model_type`.
const NAMES: &[&str] = &["qwen3moe", "Qwen3MoeForCausalLM", "qwen3_moe"];

// ---- Hyperparameter names in `HyperParams::extra` ---------------------------------------------
//
// The attention's are `llama`'s ([`llama::N_HEADS`], …); the expert counts are the shared
// [`N_EXPERTS`] and [`TOP_K`], which the classification engine and the census read.

/// Each expert's intermediate dim.
pub const D_EXPERT: &str = "d_expert";
/// Whether the top-k routing weights are renormalised to sum to 1 (`Bool`).
pub const RENORMALIZE: &str = "renormalize";

// ---- Metadata keys -----------------------------------------------------------------------------

const K_EXPERTS: Key = Key::new("{arch}.expert_count", "num_experts");
const K_TOP_K: Key = Key::new("{arch}.expert_used_count", "num_experts_per_tok");
const K_D_EXPERT: Key = Key::new("{arch}.expert_feed_forward_length", "moe_intermediate_size");
const K_NORM_TOPK: Key = Key::new("{arch}.expert_weights_norm", "norm_topk_prob");
const K_SPARSE_STEP: Key = Key::config("decoder_sparse_step");
const K_MLP_ONLY: Key = Key::config("mlp_only_layers");
const K_SHARED: Key = Key::gguf("{arch}.expert_shared_count");
const K_SHARED_FFN: Key = Key::new(
    "{arch}.expert_shared_feed_forward_length",
    "shared_expert_intermediate_size",
);
const K_SHARED_N: Key = Key::config("num_shared_experts");
const K_SHARED_N_OLD: Key = Key::config("n_shared_experts");
const K_GATING: Key = Key::gguf("{arch}.expert_gating_func");
const K_SCORING: Key = Key::config("scoring_func");

/// GGUF's `expert_gating_func` value for softmax.
const GGUF_GATING_SOFTMAX: u64 = 1;

// ---- Classification rules ----------------------------------------------------------------------

use ShapeDim::Hp;

const MOE: BlockSel = BlockSel::Layer(Block::Moe);

/// GGUF names of the `Moe` block, shapes innermost first: the experts stacked, expert outermost.
const GGUF_MOE: &[ClassRule] = &[
    rule(
        "blk.{L}.ffn_norm.weight",
        MOE,
        TensorRole::Norm,
        ClassSel::DenseEveryToken,
        &[Hp("hidden")],
    ),
    rule(
        "blk.{L}.ffn_gate_inp.weight",
        MOE,
        TensorRole::Router,
        ClassSel::Router,
        &[Hp("hidden"), Hp(N_EXPERTS)],
    ),
    rule(
        "blk.{L}.ffn_gate_exps.weight",
        MOE,
        TensorRole::Gate,
        ClassSel::ExpertsStacked,
        &[Hp("hidden"), Hp(D_EXPERT), Hp(N_EXPERTS)],
    ),
    rule(
        "blk.{L}.ffn_up_exps.weight",
        MOE,
        TensorRole::Up,
        ClassSel::ExpertsStacked,
        &[Hp("hidden"), Hp(D_EXPERT), Hp(N_EXPERTS)],
    ),
    rule(
        "blk.{L}.ffn_down_exps.weight",
        MOE,
        TensorRole::Down,
        ClassSel::ExpertsStacked,
        &[Hp(D_EXPERT), Hp("hidden"), Hp(N_EXPERTS)],
    ),
];

/// safetensors names of the `Moe` block, shapes outermost first: one tensor per expert.
const ST_MOE: &[ClassRule] = &[
    rule(
        "model.layers.{L}.post_attention_layernorm.weight",
        MOE,
        TensorRole::Norm,
        ClassSel::DenseEveryToken,
        &[Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.mlp.gate.weight",
        MOE,
        TensorRole::Router,
        ClassSel::Router,
        &[Hp(N_EXPERTS), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.mlp.experts.{E}.gate_proj.weight",
        MOE,
        TensorRole::Gate,
        ClassSel::Expert,
        &[Hp(D_EXPERT), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.mlp.experts.{E}.up_proj.weight",
        MOE,
        TensorRole::Up,
        ClassSel::Expert,
        &[Hp(D_EXPERT), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.mlp.experts.{E}.down_proj.weight",
        MOE,
        TensorRole::Down,
        ClassSel::Expert,
        &[Hp("hidden"), Hp(D_EXPERT)],
    ),
];

/// The rule tables, tied or untied: `llama`'s Qwen3 attention (QK-norm, no bias) and the `Moe`
/// block's.
fn tables(tied: bool) -> Vec<RuleTable> {
    let mut t = attention_tables(true, false, tied);
    t.extend([gguf(GGUF_MOE), st(ST_MOE)]);
    t
}

// ---- The family --------------------------------------------------------------------------------

/// The `qwen_moe` family module.
#[derive(Debug)]
pub struct QwenMoe {
    /// One matcher untied and one tied ([`llama::TIED_EMBEDDINGS`]), compiled on first use.
    matchers: [OnceLock<Matcher>; 2],
}

/// The registered instance.
pub static QWEN_MOE: QwenMoe = QwenMoe {
    matchers: [const { OnceLock::new() }; 2],
};

impl QwenMoe {
    fn matcher(&self, hp: &HyperParams) -> &Matcher {
        let tied = llama::flag(hp, llama::TIED_EMBEDDINGS);
        self.matchers[usize::from(tied)].get_or_init(|| {
            Matcher::new(&tables(tied)).expect("the qwen_moe rules compile, as its tests check")
        })
    }
}

/// Refuses a checkpoint whose names are not this family's, as resolution would.
fn check_architecture(meta: &MetaView<'_>) -> Result<(), FamilyError> {
    let names = meta.architectures()?;
    if names.iter().any(|n| NAMES.contains(n)) {
        Ok(())
    } else {
        Err(FamilyError::UnknownArchitecture {
            names: names.iter().map(|&n| n.to_owned()).collect(),
            supported: vec![TAG],
        })
    }
}

/// The expert hyperparameters, as `extra` values, after the refusals of the module docs.
fn parse_experts(meta: &MetaView<'_>) -> Result<Vec<(&'static str, HParamValue)>, FamilyError> {
    let n_experts = meta.require_u64(K_EXPERTS, 1..=u64::from(u16::MAX))?;
    let top_k = meta.require_u64(K_TOP_K, 1..=n_experts)?;
    let d_expert = meta.require_u64(K_D_EXPERT, 1..=1 << 24)?;
    let renormalize = match meta.bool(K_NORM_TOPK)? {
        Some(v) => v,
        None => matches!(meta, MetaView::Gguf(_)),
    };

    if meta.u64_or(K_SPARSE_STEP, 1, 0..=u64::MAX)? != 1 {
        return Err(refuse(
            meta,
            K_SPARSE_STEP,
            "dense layers between the MoE layers are not part of this family, whose every FFN is \
             a `Moe` block",
        ));
    }
    if meta.u64s(K_MLP_ONLY)?.is_some_and(|l| !l.is_empty()) {
        return Err(refuse(
            meta,
            K_MLP_ONLY,
            "dense layers among the MoE layers are not part of this family, whose every FFN is a \
             `Moe` block",
        ));
    }
    for key in [K_SHARED, K_SHARED_FFN, K_SHARED_N, K_SHARED_N_OLD] {
        if meta.u64(key)?.is_some_and(|n| n > 0) {
            return Err(refuse(
                meta,
                key,
                "a shared expert is not part of this family's `Moe` block",
            ));
        }
    }
    if let Some(g) = meta.u64(K_GATING)?
        && g != GGUF_GATING_SOFTMAX
    {
        return Err(refuse(
            meta,
            K_GATING,
            format!("gating function {g} is not softmax, the only routing of this family"),
        ));
    }
    if let Some(s) = meta.str(K_SCORING)?
        && s != "softmax"
    {
        return Err(refuse(
            meta,
            K_SCORING,
            format!("the scoring function `{s}` is not softmax, the only routing of this family"),
        ));
    }
    Ok(vec![
        (N_EXPERTS, HParamValue::U64(n_experts)),
        (TOP_K, HParamValue::U64(top_k)),
        (D_EXPERT, HParamValue::U64(d_expert)),
        (RENORMALIZE, HParamValue::Bool(renormalize)),
    ])
}

impl Family for QwenMoe {
    fn name(&self) -> &'static str {
        TAG
    }

    fn accepts(&self) -> &'static [&'static str] {
        NAMES
    }

    fn version(&self) -> u32 {
        1
    }

    fn parse_hparams(&self, meta: &MetaView<'_>) -> Result<HyperParams, FamilyError> {
        check_architecture(meta)?;
        let mut hp = llama::parse_attention(meta, Variant::Qwen3)?;
        for (k, v) in parse_experts(meta)? {
            hp = hp.with(k, v);
        }
        Ok(hp)
    }

    fn classify(&self, name: &str, shape: &[u64], hp: &HyperParams) -> Classification {
        self.matcher(hp).classify(name, shape, hp)
    }

    fn kv_formula(&self, hp: &HyperParams, source: KvSource) -> KvFormula {
        llama::gqa_kv_formula(hp, source)
    }

    fn required_ops(&self, hp: &HyperParams) -> OpSet {
        let mut ops = OpSet::new();
        if let Some(p) = llama::gqa_params(hp) {
            ops.extend(p.ops());
        }
        if let Some(p) = moe_params(hp) {
            ops.extend(p.ops());
        }
        ops.extend(llama::embed_params(hp).ops());
        ops.extend(llama::head_params(hp).ops());
        ops
    }
}

// ---- Block parameters --------------------------------------------------------------------------

/// The `Moe` block's parameters for every layer; `None` when `hp` lacks one.
#[must_use]
pub fn moe_params(hp: &HyperParams) -> Option<MoeParams> {
    Some(MoeParams {
        experts: u16::try_from(hp.count(N_EXPERTS)?).ok()?,
        top_k: u16::try_from(hp.count(TOP_K)?).ok()?,
        expert_inter: u32::try_from(hp.count(D_EXPERT)?).ok()?,
        renormalize: match hp.extra.get(RENORMALIZE) {
            Some(HParamValue::Bool(b)) => *b,
            _ => return None,
        },
        eps: to_f32(float(hp, llama::RMS_EPS)?),
    })
}

// ---- Fixture spec ------------------------------------------------------------------------------

const MOE_LAYERS: &[LayerKind] = &[LayerKind::GqaMoe];

/// What the `nano` and `micro` fixtures of `qwen_moe` must cover (RFC-0043 §9.1; RFC-0060
/// §12.1): the one form the family has, Qwen3-30B-A3B's.
static SPEC: FamilySpec = FamilySpec {
    family: TAG,
    variants: &[VariantSpec {
        name: "qwen3moe",
        gguf_architecture: "qwen3moe",
        architecture: "Qwen3MoeForCausalLM",
        model_type: "qwen3_moe",
        layers: MOE_LAYERS,
        attention: AttentionSpec {
            qk_norm: true,
            qkv_bias: false,
            rope: RopeKind::Plain,
        },
        moe: Some(MoeSpec {
            experts: 128,
            top_k: 8,
            shared: 0,
            latent: false,
            router_bias: false,
        }),
        tied_embeddings: false,
        mtp_modules: 0,
    }],
};

/// The family's `FamilySpec` (RFC-0043 §9.1 `fixture_spec`).
#[must_use]
pub fn fixture_spec() -> &'static FamilySpec {
    &SPEC
}

#[cfg(test)]
mod tests;
