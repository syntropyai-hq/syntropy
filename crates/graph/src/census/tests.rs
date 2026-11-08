// SPDX-License-Identifier: MIT OR Apache-2.0

//! The census builder against a test-only family, on synthetic headers in both formats. Every
//! expected count and byte total is worked out here from the shapes, independently of the
//! builder. The `llama` and `qwen_moe` families' golden censuses on the recorded real headers
//! are COMP-E16-T03's and COMP-E16-T04's.

use std::sync::OnceLock;

use syntropy_core::DType;
use syntropy_manifest::census::{ClassKind, SourceFormat, TensorClass};
use syntropy_manifest::config::HfConfig;
use syntropy_manifest::gguf::{GgufHeader, Limits};
use syntropy_manifest::hparams::{HParamValue, HyperParams, N_EXPERTS, TOP_K};
use syntropy_manifest::identity::{FileDigest, SourceFile};
use syntropy_manifest::kv::{KvDTypeRule, KvFormula, KvSource, LayerKv};
use syntropy_manifest::range::{RangeError, RangeSource};
use syntropy_manifest::safetensors::side::SourceRole;
use syntropy_manifest::safetensors::{SafetensorsHeader, parse_header};

use super::*;
use crate::classify::{ClassRule, ClassSel, DimOrder, Matcher, RuleTable, ShapeDim};
use crate::families::{Family, FamilyError, Key, MetaView, Registration, Registry};
use crate::ir::OpSet;

use ShapeDim::{Hp, Mul};

// ---- The test family -----------------------------------------------------------------------------

/// Dimensions of the synthetic checkpoints.
const LAYERS: u64 = 2;
const HIDDEN: u64 = 8;
const VOCAB: u64 = 32;
const HEADS: u64 = 2;
const KV_HEADS: u64 = 1;
const D_HEAD: u64 = 4;
const D_FFN: u64 = 6;
const EXPERTS: u64 = 4;
const TOP: u64 = 2;

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

const GQA: BlockSel = BlockSel::Layer(Block::Gqa);
const MOE: BlockSel = BlockSel::Layer(Block::Moe);

/// safetensors names, outermost dimension first: per-expert tensors and an LM head.
const SAFETENSORS: &[ClassRule] = &[
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
        ClassSel::DenseEveryToken,
        &[Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.self_attn.q_proj.weight",
        GQA,
        TensorRole::Q,
        ClassSel::DenseEveryToken,
        &[Mul("n_heads", "d_head"), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.self_attn.k_proj.weight",
        GQA,
        TensorRole::K,
        ClassSel::DenseEveryToken,
        &[Mul("n_kv_heads", "d_head"), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.self_attn.v_proj.weight",
        GQA,
        TensorRole::V,
        ClassSel::DenseEveryToken,
        &[Mul("n_kv_heads", "d_head"), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.self_attn.o_proj.weight",
        GQA,
        TensorRole::O,
        ClassSel::DenseEveryToken,
        &[Hp("hidden"), Mul("n_heads", "d_head")],
    ),
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
        &[Hp("d_ffn"), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.mlp.experts.{E}.up_proj.weight",
        MOE,
        TensorRole::Up,
        ClassSel::Expert,
        &[Hp("d_ffn"), Hp("hidden")],
    ),
    rule(
        "model.layers.{L}.mlp.experts.{E}.down_proj.weight",
        MOE,
        TensorRole::Down,
        ClassSel::Expert,
        &[Hp("hidden"), Hp("d_ffn")],
    ),
    rule(
        "model.norm.weight",
        BlockSel::Epilogue,
        TensorRole::Norm,
        ClassSel::DenseEveryToken,
        &[Hp("hidden")],
    ),
    rule(
        "lm_head.weight",
        BlockSel::Epilogue,
        TensorRole::Head,
        ClassSel::Head,
        &[Hp("vocab"), Hp("hidden")],
    ),
    rule(
        "visual.blocks.{N}.attn.qkv.weight",
        BlockSel::Encoder,
        TensorRole::Qkv,
        ClassSel::Encoder {
            tower: syntropy_manifest::census::TowerIx(0),
        },
        &[ShapeDim::Any, ShapeDim::Any],
    ),
];

/// GGUF names, innermost dimension first: stacked experts, and no `output.weight`, so the
/// embedding is tied.
const GGUF: &[ClassRule] = &[
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
        ClassSel::DenseEveryToken,
        &[Hp("hidden")],
    ),
    rule(
        "blk.{L}.attn_q.weight",
        GQA,
        TensorRole::Q,
        ClassSel::DenseEveryToken,
        &[Hp("hidden"), Mul("n_heads", "d_head")],
    ),
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
        &[Hp("hidden"), Hp("d_ffn"), Hp(N_EXPERTS)],
    ),
    rule(
        "blk.{L}.ffn_up_exps.weight",
        MOE,
        TensorRole::Up,
        ClassSel::ExpertsStacked,
        &[Hp("hidden"), Hp("d_ffn"), Hp(N_EXPERTS)],
    ),
    rule(
        "blk.{L}.ffn_down_exps.weight",
        MOE,
        TensorRole::Down,
        ClassSel::ExpertsStacked,
        &[Hp("d_ffn"), Hp("hidden"), Hp(N_EXPERTS)],
    ),
    rule(
        "output_norm.weight",
        BlockSel::Epilogue,
        TensorRole::Norm,
        ClassSel::DenseEveryToken,
        &[Hp("hidden")],
    ),
];

struct TestMoe {
    tag: &'static str,
    accepts: &'static [&'static str],
}

impl TestMoe {
    fn matcher() -> &'static Matcher {
        static M: OnceLock<Matcher> = OnceLock::new();
        M.get_or_init(|| {
            Matcher::new(&[
                RuleTable {
                    order: DimOrder::OutermostFirst,
                    rules: SAFETENSORS,
                },
                RuleTable {
                    order: DimOrder::InnermostFirst,
                    rules: GGUF,
                },
            ])
            .expect("the test rules compile")
        })
    }
}

const K_LAYERS: Key = Key::new("{arch}.block_count", "num_hidden_layers");
const K_HIDDEN: Key = Key::new("{arch}.embedding_length", "hidden_size");
const K_VOCAB: Key = Key::new("{arch}.vocab_size", "vocab_size");
const K_HEADS: Key = Key::new("{arch}.attention.head_count", "num_attention_heads");
const K_KV_HEADS: Key = Key::new("{arch}.attention.head_count_kv", "num_key_value_heads");
const K_D_HEAD: Key = Key::new("{arch}.attention.key_length", "head_dim");
const K_D_FFN: Key = Key::new("{arch}.expert_feed_forward_length", "moe_intermediate_size");
const K_EXPERTS: Key = Key::new("{arch}.expert_count", "num_experts");
const K_TOP_K: Key = Key::new("{arch}.expert_used_count", "num_experts_per_tok");

impl Family for TestMoe {
    fn name(&self) -> &'static str {
        self.tag
    }
    fn accepts(&self) -> &'static [&'static str] {
        self.accepts
    }
    fn version(&self) -> u32 {
        1
    }
    fn parse_hparams(&self, meta: &MetaView<'_>) -> Result<HyperParams, FamilyError> {
        let get = |key| meta.require_u64(key, 1..=1 << 20);
        let narrow = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
        Ok(HyperParams::new(
            u16::try_from(get(K_LAYERS)?).unwrap_or(u16::MAX),
            narrow(get(K_HIDDEN)?),
            narrow(get(K_VOCAB)?),
        )
        .with("n_heads", HParamValue::U64(get(K_HEADS)?))
        .with("n_kv_heads", HParamValue::U64(get(K_KV_HEADS)?))
        .with("d_head", HParamValue::U64(get(K_D_HEAD)?))
        .with("d_ffn", HParamValue::U64(get(K_D_FFN)?))
        .with(N_EXPERTS, HParamValue::U64(get(K_EXPERTS)?))
        .with(TOP_K, HParamValue::U64(get(K_TOP_K)?)))
    }
    fn classify(&self, name: &str, shape: &[u64], hp: &HyperParams) -> Classification {
        Self::matcher().classify(name, shape, hp)
    }
    fn kv_formula(&self, hp: &HyperParams, source: KvSource) -> KvFormula {
        let elems = 2 * hp.count("n_kv_heads").unwrap_or(0) * hp.count("d_head").unwrap_or(0);
        KvFormula::new(
            vec![
                LayerKv::PerToken {
                    elems_per_token: u32::try_from(elems).unwrap_or(u32::MAX),
                    window: None,
                    latent: false,
                };
                usize::from(hp.n_layers)
            ],
            KvDTypeRule::source_default(source),
        )
    }
    fn required_ops(&self, _: &HyperParams) -> OpSet {
        OpSet::new()
    }
}

static MOE_FAMILY: TestMoe = TestMoe {
    tag: "census_moe",
    accepts: &["testmoe", "TestMoeForCausalLM"],
};
static OTHER_FAMILY: TestMoe = TestMoe {
    tag: "census_other",
    accepts: &["othermoe"],
};
static REGISTRY: Registry = Registry::new(&[
    Registration {
        module: "census::tests::moe",
        family: &MOE_FAMILY,
    },
    Registration {
        module: "census::tests::other",
        family: &OTHER_FAMILY,
    },
]);

// ---- Synthetic sources -----------------------------------------------------------------------------

const CONFIG: &str = r#"{
    "architectures": ["TestMoeForCausalLM"],
    "model_type": "testmoe",
    "num_hidden_layers": 2, "hidden_size": 8, "vocab_size": 32,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
    "moe_intermediate_size": 6, "num_experts": 4, "num_experts_per_tok": 2
}"#;

/// The safetensors checkpoint's tensors: (name, shape), BF16.
fn st_tensors() -> Vec<(String, Vec<u64>)> {
    let mut t = vec![("model.embed_tokens.weight".to_owned(), vec![VOCAB, HIDDEN])];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        t.push((format!("{p}.input_layernorm.weight"), vec![HIDDEN]));
        t.push((
            format!("{p}.self_attn.q_proj.weight"),
            vec![HEADS * D_HEAD, HIDDEN],
        ));
        t.push((
            format!("{p}.self_attn.k_proj.weight"),
            vec![KV_HEADS * D_HEAD, HIDDEN],
        ));
        t.push((
            format!("{p}.self_attn.v_proj.weight"),
            vec![KV_HEADS * D_HEAD, HIDDEN],
        ));
        t.push((
            format!("{p}.self_attn.o_proj.weight"),
            vec![HIDDEN, HEADS * D_HEAD],
        ));
        t.push((format!("{p}.post_attention_layernorm.weight"), vec![HIDDEN]));
        t.push((format!("{p}.mlp.gate.weight"), vec![EXPERTS, HIDDEN]));
        for e in 0..EXPERTS {
            t.push((
                format!("{p}.mlp.experts.{e}.gate_proj.weight"),
                vec![D_FFN, HIDDEN],
            ));
            t.push((
                format!("{p}.mlp.experts.{e}.up_proj.weight"),
                vec![D_FFN, HIDDEN],
            ));
            t.push((
                format!("{p}.mlp.experts.{e}.down_proj.weight"),
                vec![HIDDEN, D_FFN],
            ));
        }
    }
    t.push(("model.norm.weight".to_owned(), vec![HIDDEN]));
    t.push(("lm_head.weight".to_owned(), vec![VOCAB, HIDDEN]));
    t
}

/// One safetensors shard holding `tensors` as BF16, contiguous in order.
fn shard(tensors: &[(String, Vec<u64>)]) -> SafetensorsHeader {
    let mut json = String::from("{");
    let mut at = 0u64;
    for (i, (name, shape)) in tensors.iter().enumerate() {
        let len = shape.iter().product::<u64>() * 2;
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            r#""{name}":{{"dtype":"BF16","shape":{shape:?},"data_offsets":[{at},{}]}}"#,
            at + len
        ));
        at += len;
    }
    json.push('}');
    let file_len = 8 + json.len() as u64 + at;
    parse_header(json.as_bytes(), file_len).expect("a valid synthetic header")
}

fn files() -> Vec<SourceFile> {
    vec![SourceFile {
        role: SourceRole::Weights,
        digest: FileDigest([7; 32]),
        len: 1,
    }]
}

fn bf16() -> KvSource {
    KvSource::Safetensors {
        compute: Some(DType::BF16),
        quantized: false,
    }
}

fn st_census(tensors: &[(String, Vec<u64>)], config: &str) -> Result<ModelManifest, CensusError> {
    let shards = [shard(tensors)];
    let config = HfConfig::parse(config.as_bytes(), None).expect("valid JSON");
    let files = files();
    build_in(
        &REGISTRY,
        &Sources {
            meta: MetaView::Config(&config),
            weights: Weights::Safetensors(&shards),
            kv_source: bf16(),
            tokenizer: None,
            files: &files,
        },
    )
}

/// A GGUF v3 file of the given metadata and F32 tensors, laid out with alignment 32.
fn gguf_file(kvs: &[(&str, u32)], arch: &str, tensors: &[(String, Vec<u64>)]) -> Vec<u8> {
    fn s(out: &mut Vec<u8>, v: &str) {
        out.extend_from_slice(&(v.len() as u64).to_le_bytes());
        out.extend_from_slice(v.as_bytes());
    }
    let pad = |n: u64| n.div_ceil(32) * 32;
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64 + 1).to_le_bytes());
    s(&mut out, "general.architecture");
    out.extend_from_slice(&8u32.to_le_bytes());
    s(&mut out, arch);
    for (k, v) in kvs {
        s(&mut out, k);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    let mut at = 0u64;
    for (name, ne) in tensors {
        s(&mut out, name);
        out.extend_from_slice(&(ne.len() as u32).to_le_bytes());
        for d in ne {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&at.to_le_bytes());
        at = pad(at + ne.iter().product::<u64>() * 4);
    }
    let data_start = pad(out.len() as u64);
    out.resize(usize::try_from(data_start + at).unwrap(), 0);
    out
}

struct Bytes(Vec<u8>);

impl RangeSource for Bytes {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        let start = usize::try_from(offset).unwrap();
        buf.copy_from_slice(&self.0[start..start + buf.len()]);
        Ok(())
    }
}

/// The GGUF checkpoint's tensors, innermost dimension first, with no `output.weight`.
fn gguf_tensors() -> Vec<(String, Vec<u64>)> {
    let mut t = vec![("token_embd.weight".to_owned(), vec![HIDDEN, VOCAB])];
    for l in 0..LAYERS {
        t.push((format!("blk.{l}.attn_norm.weight"), vec![HIDDEN]));
        t.push((
            format!("blk.{l}.attn_q.weight"),
            vec![HIDDEN, HEADS * D_HEAD],
        ));
        t.push((format!("blk.{l}.ffn_norm.weight"), vec![HIDDEN]));
        t.push((
            format!("blk.{l}.ffn_gate_inp.weight"),
            vec![HIDDEN, EXPERTS],
        ));
        t.push((
            format!("blk.{l}.ffn_gate_exps.weight"),
            vec![HIDDEN, D_FFN, EXPERTS],
        ));
        t.push((
            format!("blk.{l}.ffn_up_exps.weight"),
            vec![HIDDEN, D_FFN, EXPERTS],
        ));
        t.push((
            format!("blk.{l}.ffn_down_exps.weight"),
            vec![D_FFN, HIDDEN, EXPERTS],
        ));
    }
    t.push(("output_norm.weight".to_owned(), vec![HIDDEN]));
    t
}

fn gguf_census(tensors: &[(String, Vec<u64>)]) -> Result<ModelManifest, CensusError> {
    let kvs = [
        ("testmoe.block_count", LAYERS as u32),
        ("testmoe.embedding_length", HIDDEN as u32),
        ("testmoe.vocab_size", VOCAB as u32),
        ("testmoe.attention.head_count", HEADS as u32),
        ("testmoe.attention.head_count_kv", KV_HEADS as u32),
        ("testmoe.attention.key_length", D_HEAD as u32),
        ("testmoe.expert_feed_forward_length", D_FFN as u32),
        ("testmoe.expert_count", EXPERTS as u32),
        ("testmoe.expert_used_count", TOP as u32),
    ];
    let file = Bytes(gguf_file(&kvs, "testmoe", tensors));
    let header = GgufHeader::read(&file, &Limits::default()).expect("a valid synthetic GGUF");
    let files = files();
    build_in(
        &REGISTRY,
        &Sources {
            meta: MetaView::Gguf(&header.metadata),
            weights: Weights::Gguf(&header),
            kv_source: KvSource::Gguf,
            tokenizer: None,
            files: &files,
        },
    )
}

fn elems(shape: &[u64]) -> u64 {
    shape.iter().product()
}

// ---- Golden census ---------------------------------------------------------------------------------

/// The safetensors checkpoint: group counts, class totals, `read_per_token` and the KV formula,
/// each worked out here from the shapes alone.
#[test]
fn a_safetensors_census_groups_and_totals_as_section_3_1_says() {
    let m = st_census(&st_tensors(), CONFIG).unwrap();
    assert_eq!(m.family, "census_moe");
    assert_eq!(m.format, SourceFormat::Safetensors);

    // Bytes, BF16: two per element.
    let b = |shape: &[u64]| elems(shape) * 2;
    let embed = b(&[VOCAB, HIDDEN]);
    let mixer = b(&[HIDDEN])
        + b(&[HEADS * D_HEAD, HIDDEN])
        + 2 * b(&[KV_HEADS * D_HEAD, HIDDEN])
        + b(&[HIDDEN, HEADS * D_HEAD]);
    let router = b(&[HIDDEN]) + b(&[EXPERTS, HIDDEN]);
    let expert = 3 * b(&[D_FFN, HIDDEN]);
    let head = b(&[HIDDEN]) + b(&[VOCAB, HIDDEN]);

    // Groups: 1 lookup; per layer a mixer, a router (with the pre-FFN norm) and 4 experts;
    // 1 head (with the final norm).
    let count = |kind| {
        m.census
            .groups
            .iter()
            .filter(|g| g.class.kind() == kind)
            .count()
    };
    assert_eq!(m.census.groups.len(), 1 + 2 * (1 + 1 + 4) + 1);
    assert_eq!(count(ClassKind::RowLookup), 1);
    assert_eq!(count(ClassKind::DenseEveryToken), 2);
    assert_eq!(count(ClassKind::Router), 2);
    assert_eq!(count(ClassKind::Expert), 8);
    assert_eq!(count(ClassKind::Head), 1);
    assert!(
        m.census.groups.iter().all(|g| g.shared_with.is_none()),
        "nothing is tied"
    );

    let totals = &m.census.totals;
    assert_eq!(totals.get(ClassKind::RowLookup).bytes.0, embed);
    assert_eq!(
        totals.get(ClassKind::RowLookup).read_per_token.0,
        embed / VOCAB,
        "one row"
    );
    assert_eq!(totals.get(ClassKind::DenseEveryToken).bytes.0, 2 * mixer);
    assert_eq!(
        totals.get(ClassKind::DenseEveryToken).read_per_token.0,
        2 * mixer
    );
    assert_eq!(totals.get(ClassKind::Router).bytes.0, 2 * router);
    assert_eq!(totals.get(ClassKind::Expert).bytes.0, 2 * EXPERTS * expert);
    // The uniform prior: each expert reads top_k / n_experts of its bytes.
    assert_eq!(
        totals.get(ClassKind::Expert).read_per_token.0,
        2 * EXPERTS * (expert * TOP / EXPERTS)
    );
    assert_eq!(totals.get(ClassKind::Head).bytes.0, head);
    assert_eq!(totals.get(ClassKind::Head).read_per_token.0, head);

    // Every record is one tensor; Σ census bytes equals the source's tensor bytes.
    let source: u64 = st_tensors().iter().map(|(_, s)| b(s)).sum();
    assert_eq!(m.census.tensors.len(), st_tensors().len());
    assert_eq!(m.census.record_bytes(), source);

    // KV: 2 · n_kv_heads · d_head per token per layer, BF16 from the BF16 source.
    assert_eq!(
        m.kv,
        KvFormula::new(
            vec![
                LayerKv::PerToken {
                    elems_per_token: (2 * KV_HEADS * D_HEAD) as u32,
                    window: None,
                    latent: false
                };
                LAYERS as usize
            ],
            DType::BF16
        )
    );
    assert!(m.ignored.is_empty());
    assert!(m.tokenizer.is_none());
}

/// The GGUF checkpoint: stacked experts are sliced into one record each, and with no
/// `output.weight` the embedding is tied: one record, a lookup and a head group naming each
/// other.
#[test]
fn a_gguf_census_slices_experts_and_ties_the_embedding() {
    let m = gguf_census(&gguf_tensors()).unwrap();
    assert_eq!(m.format, SourceFormat::Gguf { version: 3 });
    let f = |shape: &[u64]| elems(shape) * 4;

    // Records: 1 embedding, per layer 4 dense ones and 3 stacked tensors × 4 experts, 1 norm.
    assert_eq!(m.census.tensors.len(), 1 + 2 * (4 + 3 * 4) + 1);
    let slices: Vec<&str> = m
        .census
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("blk.0.ffn_gate_exps"))
        .map(|t| t.name.as_str())
        .collect();
    assert_eq!(slices.len(), 4);

    let lookup = m
        .census
        .groups
        .iter()
        .find(|g| g.class == TensorClass::RowLookup)
        .unwrap();
    let head = m
        .census
        .groups
        .iter()
        .find(|g| g.class == TensorClass::Head)
        .unwrap();
    assert_eq!(lookup.shared_with, Some(head.id));
    assert_eq!(head.shared_with, Some(lookup.id));
    let embed_id = lookup.tensors[0];
    assert!(
        head.tensors.contains(&embed_id),
        "one record in both groups"
    );
    assert_eq!(head.bytes.0, f(&[HIDDEN]) + f(&[HIDDEN, VOCAB]));
    assert_eq!(lookup.read_per_token.0, f(&[HIDDEN, VOCAB]) / VOCAB);

    // An expert group holds its slice of each stacked tensor.
    let expert_bytes = 2 * f(&[HIDDEN, D_FFN]) + f(&[D_FFN, HIDDEN]);
    let experts: Vec<&TensorGroup> = m
        .census
        .groups
        .iter()
        .filter(|g| g.class.kind() == ClassKind::Expert)
        .collect();
    assert_eq!(experts.len(), 8);
    assert!(
        experts
            .iter()
            .all(|g| g.tensors.len() == 3 && g.bytes.0 == expert_bytes)
    );
    assert!(
        experts
            .iter()
            .all(|g| g.read_per_token.0 == expert_bytes * TOP / EXPERTS)
    );

    // Σ census bytes, each record once, equals the file's tensor bytes, though the tied
    // embedding is in two groups.
    let source: u64 = gguf_tensors().iter().map(|(_, s)| f(s)).sum();
    assert_eq!(m.census.record_bytes(), source);
    let grouped: u64 = m.census.groups.iter().map(|g| g.bytes.0).sum();
    assert_eq!(grouped, source + f(&[HIDDEN, VOCAB]));
    assert_eq!(m.kv.dtype, KvFormula::new(Vec::new(), DType::F16).dtype);
}

/// `ModelId` is the v1 formula over the family tag and the files, and the manifest stores and
/// reads back.
#[test]
fn the_manifest_has_its_model_id_and_round_trips() {
    let m = st_census(&st_tensors(), CONFIG).unwrap();
    assert_eq!(m.id, model_id("census_moe", &files()).unwrap());
    let back = ModelManifest::decode(&m.encode().unwrap()).unwrap();
    assert_eq!(back, m);
}

/// A tower tensor the family classifies `Ignored` is listed with its reason, never placed, and
/// counted by the byte check.
#[test]
fn ignored_tensors_are_listed_and_never_placed() {
    let mut tensors = st_tensors();
    tensors.push(("visual.blocks.0.attn.qkv.weight".into(), vec![12, 4]));
    let m = st_census(&tensors, CONFIG).unwrap();
    assert_eq!(m.ignored.len(), 1);
    assert_eq!(m.ignored[0].0, "visual.blocks.0.attn.qkv.weight");
    assert!(
        m.census
            .tensors
            .iter()
            .all(|t| !t.name.starts_with("visual."))
    );
}

// ---- Failing closed ---------------------------------------------------------------------------------

/// An architecture no family accepts is `UnknownFamily` (1), naming it and the supported tags.
#[test]
fn an_unknown_family_fails_closed() {
    let config = CONFIG
        .replace("TestMoeForCausalLM", "MysteryForCausalLM")
        .replace("\"testmoe\"", "\"mystery\"");
    let error = st_census(&st_tensors(), &config).unwrap_err();
    assert_eq!(error.code(), Some(1));
    assert_eq!(error.class(), ErrorClass::Unsupported);
    let CensusError::UnknownFamily { names, supported } = error else {
        panic!("{error:?}")
    };
    assert_eq!(names, ["MysteryForCausalLM", "mystery"]);
    assert_eq!(supported, ["census_moe", "census_other"]);
}

/// Names that resolve to two families are refused, not resolved by order.
#[test]
fn an_ambiguous_family_fails_closed() {
    let config = CONFIG.replace("\"testmoe\"", "\"othermoe\"");
    let error = st_census(&st_tensors(), &config).unwrap_err();
    assert_eq!(error.code(), Some(1));
    let CensusError::AmbiguousFamily { families, .. } = error else {
        panic!("{error:?}")
    };
    assert_eq!(families, ["census_moe", "census_other"]);
}

/// A tensor no rule matches is `UnknownTensor` (3), naming it.
#[test]
fn an_unknown_tensor_fails_closed() {
    let mut tensors = st_tensors();
    tensors.push(("model.layers.0.mystery.weight".into(), vec![HIDDEN]));
    let error = st_census(&tensors, CONFIG).unwrap_err();
    assert_eq!(error.code(), Some(3));
    assert_eq!(
        error,
        CensusError::UnknownTensor {
            names: vec!["model.layers.0.mystery.weight".into()]
        }
    );
}

/// A known name with the wrong shape is `UnknownTensor` (3) too.
#[test]
fn a_shape_mismatch_fails_closed() {
    let mut tensors = st_tensors();
    let q = tensors
        .iter_mut()
        .find(|(n, _)| n == "model.layers.1.self_attn.q_proj.weight")
        .unwrap();
    q.1 = vec![HEADS * D_HEAD + 1, HIDDEN];
    let error = st_census(&tensors, CONFIG).unwrap_err();
    assert_eq!(
        error,
        CensusError::UnknownTensor {
            names: vec!["model.layers.1.self_attn.q_proj.weight".into()]
        }
    );
}

/// Missing tensors are `HeaderInvalid` (6): a layer with none, an expert without one of its
/// siblings' tensors, a layer short of experts, and no embedding.
#[test]
fn a_missing_tensor_fails_closed() {
    let without = |gone: &dyn Fn(&str) -> bool| -> CensusError {
        let tensors: Vec<_> = st_tensors().into_iter().filter(|(n, _)| !gone(n)).collect();
        st_census(&tensors, CONFIG).unwrap_err()
    };
    type Gone<'a> = &'a dyn Fn(&str) -> bool;
    let cases: [(&str, Gone<'_>, &str); 4] = [
        (
            "a layer",
            &|n| n.starts_with("model.layers.1."),
            "decoder layer 1",
        ),
        (
            "one expert tensor",
            &|n| n == "model.layers.0.mlp.experts.2.up_proj.weight",
            "expert 2",
        ),
        (
            "an expert",
            &|n| n.starts_with("model.layers.1.mlp.experts.3."),
            "3 routed experts",
        ),
        (
            "the embedding",
            &|n| n == "model.embed_tokens.weight",
            "no token embedding",
        ),
    ];
    for (what, gone, says) in cases {
        let error = without(gone);
        assert_eq!(error.code(), Some(6), "{what}: {error}");
        assert!(error.to_string().contains(says), "{what}: {error}");
    }
}

/// A stacked expert tensor whose expert dimension disagrees with the hyperparameters fails its
/// shape rule; one whose bytes do not slice would be `HeaderInvalid`.
#[test]
fn a_stacked_tensor_with_the_wrong_expert_count_fails_closed() {
    let mut tensors = gguf_tensors();
    let gate = tensors
        .iter_mut()
        .find(|(n, _)| n == "blk.0.ffn_gate_exps.weight")
        .unwrap();
    gate.1 = vec![HIDDEN, D_FFN, EXPERTS + 1];
    let error = gguf_census(&tensors).unwrap_err();
    assert_eq!(error.code(), Some(3), "{error}");
}

/// A safetensors type with no scheme here is `UnknownQuant` (4).
#[test]
fn an_unsupported_stored_type_fails_closed() {
    let json = r#"{"model.embed_tokens.weight":{"dtype":"F8_E4M3","shape":[32,8],"data_offsets":[0,256]}}"#;
    let shards = [parse_header(json.as_bytes(), 8 + json.len() as u64 + 256).unwrap()];
    let config = HfConfig::parse(CONFIG.as_bytes(), None).unwrap();
    let files = files();
    let error = build_in(
        &REGISTRY,
        &Sources {
            meta: MetaView::Config(&config),
            weights: Weights::Safetensors(&shards),
            kv_source: bf16(),
            tokenizer: None,
            files: &files,
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), Some(4), "{error}");
}

/// A MoE checkpoint whose `top_k` the prior cannot use, here more than `n_experts`, is refused,
/// as one with none is: the expert prior would otherwise be wrong, or zero.
#[test]
fn an_unusable_top_k_fails_closed() {
    let config = CONFIG.replace("\"num_experts_per_tok\": 2", "\"num_experts_per_tok\": 5");
    let error = st_census(&st_tensors(), &config).unwrap_err();
    assert_eq!(error.code(), Some(6), "{error}");
    assert!(error.to_string().contains("top_k"), "{error}");
}
