// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `llama` family: dense GQA decoders (RFC-0043 §5.5; ADR-0003 decisions 1, 2 and 4).
//!
//! It accepts Llama 3.x, Mistral dense and Qwen2 and Qwen3 dense checkpoints, from GGUF or
//! safetensors, and builds every layer from the `Gqa` and `DenseFfn` blocks, with `Embed` in
//! the prologue and `LmHead` in the epilogue (RFC-0043 §3).
//!
//! - **Variants.** The architecture name picks the variant. Qwen3 normalises Q and K per head
//!   (QK-norm). Qwen2 has a bias on the Q, K and V projections. Llama and Mistral have
//!   neither. The variant is stored as two flags, [`QK_NORM`] and [`QKV_BIAS`], and the
//!   classification rules for the norm and bias tensors apply only where the flag is set, so a
//!   tensor the variant does not have is `Unknown`.
//! - **Hyperparameters.** [`Llama::parse_hparams`] reads the same values from GGUF and from
//!   `config.json`, so one model gives equal [`HyperParams`] from either source (RFC-0043
//!   §5.2). Floats are kept at F32 precision, which is what GGUF stores and the blocks take.
//!   Tied embeddings are the exception: GGUF has no key for them, so a GGUF source reads as
//!   untied, and a GGUF file with no `output.weight` is tied by RFC-0043 §4.2's first rule.
//! - **RoPE scaling.** Plain, linear, Llama-3-style and YaRN from `config.json`'s
//!   `rope_scaling`; plain and linear from GGUF. Any other rule, and YaRN from GGUF, is
//!   `HyperParamInvalid`. GGUF stores Llama-3-style scaling as a `rope_freqs.weight` table,
//!   which no rule classifies, so such a file is refused as `Unknown`.
//! - **Fail closed.** A setting the blocks cannot compute is `HyperParamInvalid`, never ignored:
//!   a sliding window in effect, a bias on the O projection or the FFN, partial rotary, an
//!   activation other than SiLU, and `config.json`'s newer `rope_parameters` layout, which this
//!   module does not read yet.
//! - **Tied embeddings.** With `tie_word_embeddings` set, the head reads the token table, as
//!   the publisher's reference code does. A checkpoint that also stores a head tensor, as
//!   Qwen3-0.6B does, has it classified `Ignored` with [`TIED_REASON`], so it is never fetched.
//! - **KV.** One `PerToken` entry of `2 · n_kv_heads · d_head` elements per layer, in the dtype
//!   RFC-0042 §4.2's rule gives for the source (COMP-E13-T06).
//!
//! The chat format is COMP-E17-T04's, which adds the `ChatFormat` type. [`fixture_spec`] gives
//! the `FamilySpec` the synthetic fixtures are generated from (RFC-0043 §9.1); the trait method
//! that returns it joins [`Family`] with the generator (QUAL-E7-T01).

use std::sync::OnceLock;

use syntropy_kernels::{GatedAct, RopeKind};
use syntropy_manifest::config::ConfigFile;
use syntropy_manifest::hparams::{HParamValue, HyperParams};
use syntropy_manifest::kv::{KvDTypeRule, KvFormula, KvSource, LayerKv};

use super::{Family, FamilyError, Key, MetaView};
use crate::blocks::{
    DenseFfnParams, EmbedParams, GqaParams, LmHeadParams, QkvLayout, RopeParams, RopeScaling,
};
use crate::classify::{
    BlockSel, ClassRule, ClassSel, Classification, DimOrder, Matcher, RuleTable, ShapeDim,
};
use crate::ir::{Block, OpSet, TensorRole};
use crate::spec::{AttentionSpec, FamilySpec, LayerKind, VariantSpec};

/// The family tag (RFC-0043 §5.1), frozen.
pub const TAG: &str = "llama";

/// The module path the frozen list binds [`TAG`] to.
pub const MODULE: &str = module_path!();

/// The architecture names it accepts, with the variant each selects: GGUF
/// `general.architecture` values, `config.json` `architectures` entries and `model_type`
/// values.
const ACCEPTED: &[(&str, Variant)] = &[
    ("llama", Variant::Llama),
    ("LlamaForCausalLM", Variant::Llama),
    ("mistral", Variant::Llama),
    ("MistralForCausalLM", Variant::Llama),
    ("qwen2", Variant::Qwen2),
    ("Qwen2ForCausalLM", Variant::Qwen2),
    ("qwen3", Variant::Qwen3),
    ("Qwen3ForCausalLM", Variant::Qwen3),
];

const NAMES: &[&str] = &[
    "llama",
    "LlamaForCausalLM",
    "mistral",
    "MistralForCausalLM",
    "qwen2",
    "Qwen2ForCausalLM",
    "qwen3",
    "Qwen3ForCausalLM",
];

/// What the architecture name says about the attention block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    /// Llama and Mistral: no QK-norm, no projection bias.
    Llama,
    /// Qwen2: a bias on the Q, K and V projections.
    Qwen2,
    /// Qwen3: Q and K RMS-normalised per head.
    Qwen3,
}

// ---- Hyperparameter names in `HyperParams::extra` ---------------------------------------------

/// Query heads.
pub const N_HEADS: &str = "n_heads";
/// KV heads.
pub const N_KV_HEADS: &str = "n_kv_heads";
/// Head dim.
pub const D_HEAD: &str = "d_head";
/// The FFN's intermediate dim.
pub const D_FFN: &str = "d_ffn";
/// The RMS norm epsilon (`F64`, at F32 precision).
pub const RMS_EPS: &str = "rms_eps";
/// The RoPE base θ (`F64`, at F32 precision).
pub const ROPE_THETA: &str = "rope_theta";
/// The RoPE rule (`Str`): `none`, `linear`, `llama3` or `yarn`.
pub const ROPE_SCALING: &str = "rope_scaling";
/// The RoPE scaling factor (`F64`), for every rule but `none`.
pub const ROPE_FACTOR: &str = "rope_factor";
/// Llama-3-style scaling's low-frequency factor (`F64`).
pub const ROPE_LOW_FREQ_FACTOR: &str = "rope_low_freq_factor";
/// Llama-3-style scaling's high-frequency factor (`F64`).
pub const ROPE_HIGH_FREQ_FACTOR: &str = "rope_high_freq_factor";
/// The pre-scaling context length (`U64`), for Llama-3-style and YaRN scaling.
pub const ROPE_ORIGINAL_CTX: &str = "rope_original_ctx";
/// YaRN's fast ramp boundary (`F64`).
pub const ROPE_BETA_FAST: &str = "rope_beta_fast";
/// YaRN's slow ramp boundary (`F64`).
pub const ROPE_BETA_SLOW: &str = "rope_beta_slow";
/// YaRN's attention factor (`F64`), present only when the checkpoint states one.
pub const ROPE_ATTN_FACTOR: &str = "rope_attn_factor";
/// The trained context length (`U64`).
pub const MAX_CTX: &str = "max_ctx";
/// Whether Q and K are RMS-normalised per head (`Bool`).
pub const QK_NORM: &str = "qk_norm";
/// Whether the Q, K and V projections have a bias (`Bool`).
pub const QKV_BIAS: &str = "qkv_bias";
/// Whether the head reads the token table (`Bool`).
pub const TIED_EMBEDDINGS: &str = "tied_embeddings";

/// Why a stored head tensor is not fetched when the embeddings are tied.
pub const TIED_REASON: &str =
    "tied embeddings: the head reads the token table, as the reference does";

// ---- Metadata keys -----------------------------------------------------------------------------

const K_LAYERS: Key = Key::new("{arch}.block_count", "num_hidden_layers");
const K_HIDDEN: Key = Key::new("{arch}.embedding_length", "hidden_size");
const K_VOCAB: Key = Key::new("{arch}.vocab_size", "vocab_size");
const K_HEADS: Key = Key::new("{arch}.attention.head_count", "num_attention_heads");
const K_KV_HEADS: Key = Key::new("{arch}.attention.head_count_kv", "num_key_value_heads");
const K_D_HEAD: Key = Key::new("{arch}.attention.key_length", "head_dim");
const K_D_HEAD_V: Key = Key::gguf("{arch}.attention.value_length");
const K_FFN: Key = Key::new("{arch}.feed_forward_length", "intermediate_size");
const K_EPS: Key = Key::new("{arch}.attention.layer_norm_rms_epsilon", "rms_norm_eps");
const K_THETA: Key = Key::new("{arch}.rope.freq_base", "rope_theta");
const K_CTX: Key = Key::new("{arch}.context_length", "max_position_embeddings");
const K_ROPE_DIMS: Key = Key::gguf("{arch}.rope.dimension_count");
const K_PARTIAL_ROTARY: Key = Key::config("partial_rotary_factor");
const K_ACT: Key = Key::config("hidden_act");
const K_ATTN_BIAS: Key = Key::config("attention_bias");
const K_MLP_BIAS: Key = Key::config("mlp_bias");
const K_TIED: Key = Key::config("tie_word_embeddings");
const K_WINDOW: Key = Key::new("{arch}.attention.sliding_window", "sliding_window");
const K_USE_WINDOW: Key = Key::config("use_sliding_window");
const K_GGUF_TOKENS: &str = "tokenizer.ggml.tokens";

const K_GGUF_SCALING: Key = Key::gguf("{arch}.rope.scaling.type");
const K_GGUF_FACTOR: Key = Key::gguf("{arch}.rope.scaling.factor");
const K_GGUF_SCALE_LINEAR: Key = Key::gguf("{arch}.rope.scale_linear");
const K_RS_TYPE: Key = Key::config("rope_scaling.rope_type");
const K_RS_TYPE_OLD: Key = Key::config("rope_scaling.type");
const K_RS_FACTOR: Key = Key::config("rope_scaling.factor");
const K_RS_LOW: Key = Key::config("rope_scaling.low_freq_factor");
const K_RS_HIGH: Key = Key::config("rope_scaling.high_freq_factor");
const K_RS_ORIGINAL: Key = Key::config("rope_scaling.original_max_position_embeddings");
const K_RS_BETA_FAST: Key = Key::config("rope_scaling.beta_fast");
const K_RS_BETA_SLOW: Key = Key::config("rope_scaling.beta_slow");
const K_RS_ATTN: Key = Key::config("rope_scaling.attention_factor");
const K_RS_MSCALE: Key = Key::config("rope_scaling.mscale");
const K_RS_MSCALE_ALL: Key = Key::config("rope_scaling.mscale_all_dim");
const K_RS_TRUNCATE: Key = Key::config("rope_scaling.truncate");

/// The default RoPE base where the metadata gives none.
const DEFAULT_THETA: f64 = 10_000.0;
/// YaRN's ramp boundaries where the configuration gives none.
const DEFAULT_BETA_FAST: f64 = 32.0;
const DEFAULT_BETA_SLOW: f64 = 1.0;

// ---- Classification rules ----------------------------------------------------------------------

const fn rule(
    pattern: &'static str,
    block: BlockSel,
    role: TensorRole,
    class: ClassSel,
    shape: &'static [ShapeDim],
) -> ClassRule {
    ClassRule {
        pattern,
        block,
        role,
        class,
        shape,
        aux_of: None,
    }
}

use ShapeDim::{Any, Hp, Mul};

const GQA: BlockSel = BlockSel::Layer(Block::Gqa);
const FFN: BlockSel = BlockSel::Layer(Block::DenseFfn);
const DENSE: ClassSel = ClassSel::DenseEveryToken;
const Q_WIDTH: ShapeDim = Mul(N_HEADS, D_HEAD);
const KV_WIDTH: ShapeDim = Mul(N_KV_HEADS, D_HEAD);
const TIED: ClassSel = ClassSel::Ignored {
    reason: TIED_REASON,
};

/// GGUF names, shapes innermost first (ggml's `ne`).
const GGUF_BASE: &[ClassRule] = &[
    rule(
        "token_embd.weight",
        BlockSel::Prologue,
        TensorRole::Embed,
        ClassSel::RowLookup,
        &[Hp("hidden"), Hp("vocab")],
    ),
    rule(
        "blk.{L}.attn_norm.weight",
        GQA,
        TensorRole::Norm,
        DENSE,
        &[Hp("hidden")],
    ),
    rule(
        "blk.{L}.attn_q.weight",
        GQA,
        TensorRole::Q,
        DENSE,
        &[Hp("hidden"), Q_WIDTH],
    ),
    rule(
        "blk.{L}.attn_k.weight",
        GQA,
        TensorRole::K,
        DENSE,
        &[Hp("hidden"), KV_WIDTH],
    ),
    rule(
        "blk.{L}.attn_v.weight",
        GQA,
        TensorRole::V,
        DENSE,
        &[Hp("hidden"), KV_WIDTH],
    ),
    rule(
        "blk.{L}.attn_output.weight",
        GQA,
        TensorRole::O,
        DENSE,
        &[Q_WIDTH, Hp("hidden")],
    ),
    rule(
        "blk.{L}.ffn_norm.weight",
        FFN,
        TensorRole::Norm,
        DENSE,
        &[Hp("hidden")],
    ),
    rule(
        "blk.{L}.ffn_gate.weight",
        FFN,
        TensorRole::Gate,
        DENSE,
        &[Hp("hidden"), Hp(D_FFN)],
    ),
    rule(
        "blk.{L}.ffn_up.weight",
        FFN,
        TensorRole::Up,
        DENSE,
        &[Hp("hidden"), Hp(D_FFN)],
    ),
    rule(
        "blk.{L}.ffn_down.weight",
        FFN,
        TensorRole::Down,
        DENSE,
        &[Hp(D_FFN), Hp("hidden")],
    ),
    rule(
        "output_norm.weight",
        BlockSel::Epilogue,
        TensorRole::Norm,
        DENSE,
        &[Hp("hidden")],
    ),
];
const GGUF_HEAD: &[ClassRule] = &[rule(
    "output.weight",
    BlockSel::Epilogue,
    TensorRole::Head,
    ClassSel::Head,
    &[Hp("hidden"), Hp("vocab")],
)];
const GGUF_HEAD_TIED: &[ClassRule] = &[rule(
    "output.weight",
    BlockSel::Epilogue,
    TensorRole::Head,
    TIED,
    &[Hp("hidden"), Hp("vocab")],
)];
const GGUF_QK_NORM: &[ClassRule] = &[
    rule(
        "blk.{L}.attn_q_norm.weight",
        GQA,
        TensorRole::Norm,
        DENSE,
        &[Hp(D_HEAD)],
    ),
    rule(
        "blk.{L}.attn_k_norm.weight",
        GQA,
        TensorRole::Norm,
        DENSE,
        &[Hp(D_HEAD)],
    ),
];
const GGUF_BIAS: &[ClassRule] = &[
    rule("blk.{L}.attn_q.bias", GQA, TensorRole::Q, DENSE, &[Q_WIDTH]),
    rule(
        "blk.{L}.attn_k.bias",
        GQA,
        TensorRole::K,
        DENSE,
        &[KV_WIDTH],
    ),
    rule(
        "blk.{L}.attn_v.bias",
        GQA,
        TensorRole::V,
        DENSE,
        &[KV_WIDTH],
    ),
];

/// safetensors names, shapes outermost first.
const ST_BASE: &[ClassRule] = &[
    rule(
        "model.embed_tokens.weight",
        BlockSel::Prologue,
        TensorRole::Embed,
        ClassSel::RowLookup,
        &[Hp("vocab"), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.input_layernorm.weight",
        GQA,
        TensorRole::Norm,
        DENSE,
        &[Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.self_attn.q_proj.weight",
        GQA,
        TensorRole::Q,
        DENSE,
        &[Q_WIDTH, Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.self_attn.k_proj.weight",
        GQA,
        TensorRole::K,
        DENSE,
        &[KV_WIDTH, Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.self_attn.v_proj.weight",
        GQA,
        TensorRole::V,
        DENSE,
        &[KV_WIDTH, Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.self_attn.o_proj.weight",
        GQA,
        TensorRole::O,
        DENSE,
        &[Hp("hidden"), Q_WIDTH],
    ),
    rule(
        "model.layers.{L}.self_attn.rotary_emb.inv_freq",
        GQA,
        TensorRole::Other(0),
        ClassSel::Ignored {
            reason: "a buffer the reference recomputes, not a weight",
        },
        &[Any],
    ),
    rule(
        "model.layers.{L}.post_attention_layernorm.weight",
        FFN,
        TensorRole::Norm,
        DENSE,
        &[Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.mlp.gate_proj.weight",
        FFN,
        TensorRole::Gate,
        DENSE,
        &[Hp(D_FFN), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.mlp.up_proj.weight",
        FFN,
        TensorRole::Up,
        DENSE,
        &[Hp(D_FFN), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.mlp.down_proj.weight",
        FFN,
        TensorRole::Down,
        DENSE,
        &[Hp("hidden"), Hp(D_FFN)],
    ),
    rule(
        "model.norm.weight",
        BlockSel::Epilogue,
        TensorRole::Norm,
        DENSE,
        &[Hp("hidden")],
    ),
];
const ST_HEAD: &[ClassRule] = &[rule(
    "lm_head.weight",
    BlockSel::Epilogue,
    TensorRole::Head,
    ClassSel::Head,
    &[Hp("vocab"), Hp("hidden")],
)];
const ST_HEAD_TIED: &[ClassRule] = &[rule(
    "lm_head.weight",
    BlockSel::Epilogue,
    TensorRole::Head,
    TIED,
    &[Hp("vocab"), Hp("hidden")],
)];
const ST_QK_NORM: &[ClassRule] = &[
    rule(
        "model.layers.{L}.self_attn.q_norm.weight",
        GQA,
        TensorRole::Norm,
        DENSE,
        &[Hp(D_HEAD)],
    ),
    rule(
        "model.layers.{L}.self_attn.k_norm.weight",
        GQA,
        TensorRole::Norm,
        DENSE,
        &[Hp(D_HEAD)],
    ),
];
const ST_BIAS: &[ClassRule] = &[
    rule(
        "model.layers.{L}.self_attn.q_proj.bias",
        GQA,
        TensorRole::Q,
        DENSE,
        &[Q_WIDTH],
    ),
    rule(
        "model.layers.{L}.self_attn.k_proj.bias",
        GQA,
        TensorRole::K,
        DENSE,
        &[KV_WIDTH],
    ),
    rule(
        "model.layers.{L}.self_attn.v_proj.bias",
        GQA,
        TensorRole::V,
        DENSE,
        &[KV_WIDTH],
    ),
];

/// The rule tables for one combination of the variant flags.
fn tables(qk_norm: bool, bias: bool, tied: bool) -> Vec<RuleTable> {
    let gguf = |rules| RuleTable {
        order: DimOrder::InnermostFirst,
        rules,
    };
    let st = |rules| RuleTable {
        order: DimOrder::OutermostFirst,
        rules,
    };
    let mut t = vec![
        gguf(GGUF_BASE),
        gguf(if tied { GGUF_HEAD_TIED } else { GGUF_HEAD }),
        st(ST_BASE),
        st(if tied { ST_HEAD_TIED } else { ST_HEAD }),
    ];
    if qk_norm {
        t.extend([gguf(GGUF_QK_NORM), st(ST_QK_NORM)]);
    }
    if bias {
        t.extend([gguf(GGUF_BIAS), st(ST_BIAS)]);
    }
    t
}

// ---- The family --------------------------------------------------------------------------------

/// The `llama` family module.
#[derive(Debug)]
pub struct Llama {
    /// One matcher per combination of [`QK_NORM`], [`QKV_BIAS`] and [`TIED_EMBEDDINGS`],
    /// compiled on first use.
    matchers: [OnceLock<Matcher>; 8],
}

/// The registered instance.
pub static LLAMA: Llama = Llama {
    matchers: [const { OnceLock::new() }; 8],
};

fn flag(hp: &HyperParams, key: &str) -> bool {
    matches!(hp.extra.get(key), Some(HParamValue::Bool(true)))
}

fn float(hp: &HyperParams, key: &str) -> Option<f64> {
    match hp.extra.get(key) {
        Some(HParamValue::F64(v)) => Some(*v),
        _ => None,
    }
}

/// `v` at F32 precision, as GGUF stores it and the blocks take it.
fn f32_precision(v: f64) -> f64 {
    f64::from(v as f32)
}

fn to_f32(v: f64) -> f32 {
    v as f32
}

impl Llama {
    fn matcher(&self, hp: &HyperParams) -> &Matcher {
        let (q, b, t) = (
            flag(hp, QK_NORM),
            flag(hp, QKV_BIAS),
            flag(hp, TIED_EMBEDDINGS),
        );
        let i = usize::from(q) | usize::from(b) << 1 | usize::from(t) << 2;
        self.matchers[i].get_or_init(|| {
            Matcher::new(&tables(q, b, t)).expect("the llama rules compile, as its tests check")
        })
    }
}

/// The key's name in `meta`, for an error message.
fn name(meta: &MetaView<'_>, key: Key) -> String {
    meta.key_name(key)
        .ok()
        .flatten()
        .unwrap_or_else(|| "a key this source does not carry".to_owned())
}

fn variant(meta: &MetaView<'_>) -> Result<Variant, FamilyError> {
    let names = meta.architectures()?;
    names
        .iter()
        .find_map(|n| ACCEPTED.iter().find(|(a, _)| a == n).map(|&(_, v)| v))
        .ok_or_else(|| FamilyError::UnknownArchitecture {
            names: names.iter().map(|&n| n.to_owned()).collect(),
            supported: vec![TAG],
        })
}

/// A float that, when present, must be in `range`; absent gives `None`.
fn opt_f64(
    meta: &MetaView<'_>,
    key: Key,
    range: std::ops::RangeInclusive<f64>,
) -> Result<Option<f64>, FamilyError> {
    match meta.f64(key)? {
        None => Ok(None),
        Some(_) => meta.require_f64(key, range).map(Some),
    }
}

/// Refuses `key` with `reason`.
fn refuse(meta: &MetaView<'_>, key: Key, reason: impl Into<String>) -> FamilyError {
    FamilyError::invalid(name(meta, key), reason)
}

/// The vocabulary size: `{arch}.vocab_size`, or the GGUF token list's length when the file
/// does not state it.
fn vocab(meta: &MetaView<'_>) -> Result<u64, FamilyError> {
    let range = 1..=u64::from(u32::MAX);
    if meta.u64(K_VOCAB)?.is_some() {
        return meta.require_u64(K_VOCAB, range);
    }
    if let MetaView::Gguf(m) = meta {
        let tokens = m.array(K_GGUF_TOKENS).map_err(|e| {
            FamilyError::invalid(
                K_GGUF_TOKENS,
                format!("holds {}, expected an array", e.found),
            )
        })?;
        if let Some(a) = tokens {
            let n = u64::try_from(a.len()).unwrap_or(u64::MAX);
            if range.contains(&n) {
                return Ok(n);
            }
            return Err(FamilyError::invalid(
                K_GGUF_TOKENS,
                format!("{n} tokens is outside {}..={}", range.start(), range.end()),
            ));
        }
    }
    meta.require_u64(K_VOCAB, range)
}

/// The RoPE rule, as `extra` values.
fn rope_scaling(
    meta: &MetaView<'_>,
    max_ctx: u64,
) -> Result<Vec<(&'static str, HParamValue)>, FamilyError> {
    let factor = |key| meta.require_f64(key, 1.0..=1e6).map(f32_precision);
    let s = |v: &str| HParamValue::Str(v.to_owned());
    let f = HParamValue::F64;
    match meta {
        MetaView::Gguf(_) => {
            let kind = meta.str(K_GGUF_SCALING)?;
            match kind.as_deref() {
                None | Some("none") => {
                    if meta.f64(K_GGUF_SCALE_LINEAR)?.is_some() {
                        let k = factor(K_GGUF_SCALE_LINEAR)?;
                        Ok(vec![(ROPE_SCALING, s("linear")), (ROPE_FACTOR, f(k))])
                    } else {
                        Ok(vec![(ROPE_SCALING, s("none"))])
                    }
                }
                Some("linear") => Ok(vec![
                    (ROPE_SCALING, s("linear")),
                    (ROPE_FACTOR, f(factor(K_GGUF_FACTOR)?)),
                ]),
                Some(other) => Err(refuse(
                    meta,
                    K_GGUF_SCALING,
                    format!("the RoPE rule `{other}` is not read from GGUF"),
                )),
            }
        }
        MetaView::Config(c) => {
            if c.contains(ConfigFile::Config, "rope_parameters") {
                return Err(FamilyError::invalid(
                    "config.json: rope_parameters",
                    "this layout of the RoPE settings is not read yet",
                ));
            }
            if !c.contains(ConfigFile::Config, "rope_scaling") {
                return Ok(vec![(ROPE_SCALING, s("none"))]);
            }
            let kind = match meta.str(K_RS_TYPE)? {
                Some(k) => k,
                None => meta.str(K_RS_TYPE_OLD)?.ok_or_else(|| {
                    refuse(
                        meta,
                        K_RS_TYPE,
                        "required when `rope_scaling` is set, and missing",
                    )
                })?,
            };
            match kind.as_str() {
                "default" => Ok(vec![(ROPE_SCALING, s("none"))]),
                "linear" => Ok(vec![
                    (ROPE_SCALING, s("linear")),
                    (ROPE_FACTOR, f(factor(K_RS_FACTOR)?)),
                ]),
                "llama3" => {
                    let low = meta.require_f64(K_RS_LOW, 0.0..=1e6).map(f32_precision)?;
                    let high = meta.require_f64(K_RS_HIGH, 0.0..=1e6).map(f32_precision)?;
                    if low <= 0.0 || high <= low {
                        return Err(refuse(
                            meta,
                            K_RS_HIGH,
                            format!(
                                "needs 0 < low_freq_factor < high_freq_factor, got {low} and {high}"
                            ),
                        ));
                    }
                    let original = meta.require_u64(K_RS_ORIGINAL, 1..=u64::from(u32::MAX))?;
                    Ok(vec![
                        (ROPE_SCALING, s("llama3")),
                        (ROPE_FACTOR, f(factor(K_RS_FACTOR)?)),
                        (ROPE_LOW_FREQ_FACTOR, f(low)),
                        (ROPE_HIGH_FREQ_FACTOR, f(high)),
                        (ROPE_ORIGINAL_CTX, HParamValue::U64(original)),
                    ])
                }
                "yarn" => {
                    for key in [K_RS_MSCALE, K_RS_MSCALE_ALL] {
                        if meta.f64(key)?.is_some() {
                            return Err(refuse(meta, key, "this YaRN form is not computed"));
                        }
                    }
                    if meta.bool(K_RS_TRUNCATE)? == Some(false) {
                        return Err(refuse(
                            meta,
                            K_RS_TRUNCATE,
                            "untruncated YaRN is not computed",
                        ));
                    }
                    let original = meta.u64_or(K_RS_ORIGINAL, max_ctx, 1..=u64::from(u32::MAX))?;
                    let beta_fast =
                        opt_f64(meta, K_RS_BETA_FAST, 0.0..=1e6)?.unwrap_or(DEFAULT_BETA_FAST);
                    let beta_slow =
                        opt_f64(meta, K_RS_BETA_SLOW, 0.0..=1e6)?.unwrap_or(DEFAULT_BETA_SLOW);
                    let mut v = vec![
                        (ROPE_SCALING, s("yarn")),
                        (ROPE_FACTOR, f(factor(K_RS_FACTOR)?)),
                        (ROPE_ORIGINAL_CTX, HParamValue::U64(original)),
                        (ROPE_BETA_FAST, f(f32_precision(beta_fast))),
                        (ROPE_BETA_SLOW, f(f32_precision(beta_slow))),
                    ];
                    if let Some(a) = opt_f64(meta, K_RS_ATTN, 0.0..=1e6)? {
                        v.push((ROPE_ATTN_FACTOR, f(f32_precision(a))));
                    }
                    Ok(v)
                }
                other => Err(refuse(
                    meta,
                    K_RS_TYPE,
                    format!("the RoPE rule `{other}` is not computed"),
                )),
            }
        }
    }
}

impl Family for Llama {
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
        let variant = variant(meta)?;
        let n_layers = meta.require_u64(K_LAYERS, 1..=u64::from(u16::MAX))?;
        let hidden = meta.require_u64(K_HIDDEN, 1..=1 << 20)?;
        let vocab = vocab(meta)?;
        let n_heads = meta.require_u64(K_HEADS, 1..=u64::from(u16::MAX))?;
        let n_kv_heads = meta.u64_or(K_KV_HEADS, n_heads, 1..=n_heads)?;
        if n_heads % n_kv_heads != 0 {
            return Err(refuse(
                meta,
                K_KV_HEADS,
                format!("{n_kv_heads} KV heads do not divide {n_heads} query heads"),
            ));
        }
        let d_head = match meta.u64(K_D_HEAD)? {
            Some(_) => meta.require_u64(K_D_HEAD, 1..=u64::from(u16::MAX))?,
            None if hidden % n_heads == 0 => hidden / n_heads,
            None => {
                return Err(refuse(
                    meta,
                    K_HEADS,
                    format!(
                        "{n_heads} heads do not divide the hidden size {hidden}, and no head dim is given"
                    ),
                ));
            }
        };
        if let Some(v) = meta.u64(K_D_HEAD_V)?
            && v != d_head
        {
            return Err(refuse(
                meta,
                K_D_HEAD_V,
                format!("V heads of {v} differ from K heads of {d_head}"),
            ));
        }
        let d_ffn = meta.require_u64(K_FFN, 1..=1 << 24)?;
        let eps = meta.require_f64(K_EPS, f64::MIN_POSITIVE..=1.0)?;
        let theta = match meta.f64(K_THETA)? {
            Some(_) => meta.require_f64(K_THETA, 1.0..=1e12)?,
            None => DEFAULT_THETA,
        };
        let max_ctx = meta.require_u64(K_CTX, 1..=u64::from(u32::MAX))?;

        // Settings the blocks cannot compute.
        if let Some(dims) = meta.u64(K_ROPE_DIMS)?
            && dims != d_head
        {
            return Err(refuse(
                meta,
                K_ROPE_DIMS,
                format!("partial rotary ({dims} of {d_head} dims) is not part of this family"),
            ));
        }
        if let Some(p) = meta.f64(K_PARTIAL_ROTARY)?
            && p != 1.0
        {
            return Err(refuse(
                meta,
                K_PARTIAL_ROTARY,
                format!("partial rotary ({p}) is not part of this family"),
            ));
        }
        if let Some(act) = meta.str(K_ACT)?
            && act != "silu"
        {
            return Err(refuse(
                meta,
                K_ACT,
                format!("the activation `{act}` is not part of this family, which is SiLU-gated"),
            ));
        }
        if variant != Variant::Qwen2 && meta.bool(K_ATTN_BIAS)? == Some(true) {
            return Err(refuse(
                meta,
                K_ATTN_BIAS,
                "a bias on the O projection is not part of the `Gqa` block",
            ));
        }
        if meta.bool(K_MLP_BIAS)? == Some(true) {
            return Err(refuse(
                meta,
                K_MLP_BIAS,
                "a bias in the FFN is not part of the `DenseFfn` block",
            ));
        }
        let window_in_effect = match variant {
            Variant::Qwen2 | Variant::Qwen3 => meta.bool(K_USE_WINDOW)? == Some(true),
            Variant::Llama => meta.u64(K_WINDOW)?.is_some(),
        } || matches!(meta, MetaView::Gguf(_))
            && meta.u64(K_WINDOW)?.is_some();
        if window_in_effect {
            return Err(refuse(
                meta,
                K_WINDOW,
                "a sliding window needs the `SlidingGqa` block, which this family does not use",
            ));
        }
        let tied = meta.bool(K_TIED)?.unwrap_or(false);
        let rope = rope_scaling(meta, max_ctx)?;

        let narrow = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
        let mut hp = HyperParams::new(
            u16::try_from(n_layers).unwrap_or(u16::MAX),
            narrow(hidden),
            narrow(vocab),
        )
        .with(N_HEADS, HParamValue::U64(n_heads))
        .with(N_KV_HEADS, HParamValue::U64(n_kv_heads))
        .with(D_HEAD, HParamValue::U64(d_head))
        .with(D_FFN, HParamValue::U64(d_ffn))
        .with(RMS_EPS, HParamValue::F64(f32_precision(eps)))
        .with(ROPE_THETA, HParamValue::F64(f32_precision(theta)))
        .with(MAX_CTX, HParamValue::U64(max_ctx))
        .with(QK_NORM, HParamValue::Bool(variant == Variant::Qwen3))
        .with(QKV_BIAS, HParamValue::Bool(variant == Variant::Qwen2))
        .with(TIED_EMBEDDINGS, HParamValue::Bool(tied));
        for (k, v) in rope {
            hp = hp.with(k, v);
        }
        Ok(hp)
    }

    fn classify(&self, name: &str, shape: &[u64], hp: &HyperParams) -> Classification {
        self.matcher(hp).classify(name, shape, hp)
    }

    fn kv_formula(&self, hp: &HyperParams, source: KvSource) -> KvFormula {
        let kv_heads = hp
            .count(N_KV_HEADS)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0);
        let d_head = hp
            .count(D_HEAD)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0);
        let mut kv = KvFormula::new(
            vec![LayerKv::gqa(kv_heads, d_head, None); usize::from(hp.n_layers)],
            KvDTypeRule::source_default(source),
        );
        // RFC-0042 §4.2's rule with no declared cache and no family requirement: the source's
        // dtype on every layer. It cannot fail on these inputs; the line above is its value.
        let rule = KvDTypeRule {
            source,
            declared_cache: None,
            family: None,
            family_layers: Vec::new(),
        };
        if let Ok(dtype) = rule.reference() {
            kv.dtype = dtype;
        }
        kv
    }

    fn required_ops(&self, hp: &HyperParams) -> OpSet {
        let mut ops = OpSet::new();
        if let Some(p) = gqa_params(hp) {
            ops.extend(p.ops());
        }
        ops.extend(ffn_params(hp).ops());
        ops.extend(embed_params(hp).ops());
        ops.extend(head_params(hp).ops());
        ops
    }
}

// ---- Block parameters --------------------------------------------------------------------------

/// The RoPE rule `hp` states; `None` for an unknown rule or a missing parameter.
#[must_use]
pub fn rope_params(hp: &HyperParams) -> Option<RopeParams> {
    let factor = || float(hp, ROPE_FACTOR).map(to_f32);
    let original = || {
        hp.count(ROPE_ORIGINAL_CTX)
            .and_then(|n| u32::try_from(n).ok())
    };
    let scaling = match hp.extra.get(ROPE_SCALING) {
        Some(HParamValue::Str(s)) => match s.as_str() {
            "none" => RopeScaling::Plain,
            "linear" => RopeScaling::Linear { factor: factor()? },
            "llama3" => RopeScaling::Llama3 {
                factor: factor()?,
                low_freq_factor: to_f32(float(hp, ROPE_LOW_FREQ_FACTOR)?),
                high_freq_factor: to_f32(float(hp, ROPE_HIGH_FREQ_FACTOR)?),
                original_ctx: original()?,
            },
            "yarn" => RopeScaling::Yarn {
                factor: factor()?,
                original_ctx: original()?,
                beta_fast: to_f32(float(hp, ROPE_BETA_FAST)?),
                beta_slow: to_f32(float(hp, ROPE_BETA_SLOW)?),
                attn_factor: float(hp, ROPE_ATTN_FACTOR).map(to_f32),
            },
            _ => return None,
        },
        _ => return None,
    };
    Some(RopeParams {
        scaling,
        theta: to_f32(float(hp, ROPE_THETA)?),
        dims: u16::try_from(hp.count(D_HEAD)?).ok()?,
    })
}

/// The `Gqa` block's parameters for every layer; `None` when `hp` lacks one.
#[must_use]
pub fn gqa_params(hp: &HyperParams) -> Option<GqaParams> {
    let u16_of = |k| u16::try_from(hp.count(k)?).ok();
    Some(GqaParams {
        heads: u16_of(N_HEADS)?,
        kv_heads: u16_of(N_KV_HEADS)?,
        head_dim: u16_of(D_HEAD)?,
        qkv: QkvLayout::Separate {
            qk_norm: flag(hp, QK_NORM),
        },
        bias: flag(hp, QKV_BIAS),
        rope: rope_params(hp)?,
        scale: None,
        softcap: None,
        eps: to_f32(float(hp, RMS_EPS)?),
    })
}

/// The `DenseFfn` block's parameters: SiLU-gated.
#[must_use]
pub fn ffn_params(hp: &HyperParams) -> DenseFfnParams {
    DenseFfnParams {
        inter: hp
            .count(D_FFN)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0),
        act: GatedAct::Silu,
        eps: float(hp, RMS_EPS).map_or(0.0, to_f32),
    }
}

/// The `Embed` block's parameters: no scale.
#[must_use]
pub fn embed_params(hp: &HyperParams) -> EmbedParams {
    EmbedParams {
        hidden: hp.hidden,
        vocab: hp.vocab,
        scale: None,
    }
}

/// The `LmHead` block's parameters: no soft-cap.
#[must_use]
pub fn head_params(hp: &HyperParams) -> LmHeadParams {
    LmHeadParams {
        vocab: hp.vocab,
        softcap: None,
        eps: float(hp, RMS_EPS).map_or(0.0, to_f32),
    }
}

// ---- Fixture spec ------------------------------------------------------------------------------

const DENSE_LAYERS: &[LayerKind] = &[LayerKind::GqaDense];

/// What the `nano` and `micro` fixtures of `llama` must cover (RFC-0043 §9.1; RFC-0060
/// §12.1): one variant per attention form and RoPE rule, so every classification rule and
/// every RoPE rule is exercised.
static SPEC: FamilySpec = FamilySpec {
    family: TAG,
    variants: &[
        VariantSpec {
            name: "llama3",
            gguf_architecture: "llama",
            architecture: "LlamaForCausalLM",
            model_type: "llama",
            layers: DENSE_LAYERS,
            attention: AttentionSpec {
                qk_norm: false,
                qkv_bias: false,
                rope: RopeKind::Llama3,
            },
            moe: None,
            tied_embeddings: false,
            mtp_modules: 0,
        },
        VariantSpec {
            name: "llama-linear",
            gguf_architecture: "llama",
            architecture: "LlamaForCausalLM",
            model_type: "llama",
            layers: DENSE_LAYERS,
            attention: AttentionSpec {
                qk_norm: false,
                qkv_bias: false,
                rope: RopeKind::Linear,
            },
            moe: None,
            tied_embeddings: false,
            mtp_modules: 0,
        },
        VariantSpec {
            name: "qwen2",
            gguf_architecture: "qwen2",
            architecture: "Qwen2ForCausalLM",
            model_type: "qwen2",
            layers: DENSE_LAYERS,
            attention: AttentionSpec {
                qk_norm: false,
                qkv_bias: true,
                rope: RopeKind::Yarn,
            },
            moe: None,
            tied_embeddings: true,
            mtp_modules: 0,
        },
        VariantSpec {
            name: "qwen3",
            gguf_architecture: "qwen3",
            architecture: "Qwen3ForCausalLM",
            model_type: "qwen3",
            layers: DENSE_LAYERS,
            attention: AttentionSpec {
                qk_norm: true,
                qkv_bias: false,
                rope: RopeKind::Plain,
            },
            moe: None,
            tied_embeddings: true,
            mtp_modules: 0,
        },
    ],
};

/// The family's `FamilySpec` (RFC-0043 §9.1 `fixture_spec`).
#[must_use]
pub fn fixture_spec() -> &'static FamilySpec {
    &SPEC
}

#[cfg(test)]
mod tests;
