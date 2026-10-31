// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the `qwen_moe` family against COMP-E16-T04's acceptance criteria and RFC-0043 §4,
//! §5.3 and §5.5.
//!
//! - **Golden census.** The family's classification of Qwen3-30B-A3B and the manifest the census
//!   builder (COMP-E10-T07) makes from it are compared with `golden/qwen3-30b-a3b.txt`. The
//!   checkpoint has 18,867 tensors, so the golden file folds the layer and expert indices out
//!   of each line (`model.layers.{L}.mlp.experts.{E}.up_proj.weight`) and counts the lines
//!   that are then equal; the test checks apart that every layer and every expert is present.
//!   In CI it runs on a configuration and a safetensors header *derived* here from
//!   Qwen3-30B-A3B's published `config.json`: names and shapes only, written by this file, with
//!   no file of the checkpoint copied. With `SYNTROPY_QWEN3_30B_A3B_DIR` set to a directory of
//!   the recorded files, fetched and verified with `cargo xtask fixtures` at the revision
//!   QUAL-E7-T06 pinned, it also runs on the recorded `config.json`, `generation_config.json`
//!   and the 16 shard headers (`model-0000N-of-00016.safetensors.header`); unset, that test
//!   prints that it did not run. `SYNTROPY_UPDATE_SNAPSHOTS=1` regenerates the golden file, as
//!   for `llama`.
//! - **KV dtype.** F16 for a GGUF source and BF16 for a BF16 safetensors source, by
//!   COMP-E13-T06's rule.
//! - **Fail closed.** Unknown architectures, missing and out-of-range expert counts, layouts the
//!   `Moe` block cannot compute, an expert index past the last, wrong shapes, and tensors with
//!   no rule, such as a shared expert's.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use syntropy_core::{DType, ExpertIx, LayerIx, TensorGroupId};
use syntropy_kernels::{AttnKind, GatedAct, OpKind, RouteNorm};
use syntropy_manifest::census::{ModelManifest, TensorClass};
use syntropy_manifest::config::HfConfig;
use syntropy_manifest::identity::{FileDigest, SourceFile};
use syntropy_manifest::kv::{KvDTypeRule, LayerKv};
use syntropy_manifest::safetensors::side::SourceRole;
use syntropy_manifest::safetensors::{SafetensorsHeader, parse_header, read_header_default};

use super::*;
use crate::census::{self, CensusError, Sources, Weights};
use crate::classify::{TensorPlan, classify_tensors};
use crate::families::llama::tests::{Bytes, Kv, header, safetensors_header, update_golden, with};
use crate::families::llama::{D_HEAD, LLAMA, N_HEADS, N_KV_HEADS, QK_NORM, QKV_BIAS};
use crate::families::{FROZEN_TAGS, families};

// ---- Fixtures -----------------------------------------------------------------------------------

/// A `config.json` for Qwen3-30B-A3B, derived from the published one at revision
/// `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`: the keys the family reads, with their values.
const CONFIG: &[u8] = br#"{
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "num_hidden_layers": 48,
    "hidden_size": 2048,
    "vocab_size": 151936,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "norm_topk_prob": true,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "rope_scaling": null,
    "max_position_embeddings": 40960,
    "sliding_window": null,
    "use_sliding_window": false,
    "attention_bias": false,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16"
}"#;
const GOLDEN: &str = include_str!("golden/qwen3-30b-a3b.txt");
/// The directory of the recorded Qwen3-30B-A3B files, for the opt-in test.
const DIR_VAR: &str = "SYNTROPY_QWEN3_30B_A3B_DIR";
/// Each shard's size on the Hub at the pinned revision, which its recorded header describes.
const SHARD_BYTES: [u64; 16] = [
    3_999_417_504,
    3_999_974_192,
    3_997_360_832,
    3_999_975_056,
    3_999_975_400,
    3_999_975_400,
    3_999_975_472,
    3_997_362_064,
    3_999_975_408,
    3_999_975_400,
    3_999_975_408,
    3_987_400_496,
    3_997_353_632,
    3_999_975_400,
    3_999_975_400,
    1_087_928_584,
];
const LAYERS: u16 = 48;
const EXPERTS: u16 = 128;
/// Qwen3-30B-A3B's tensors: per layer 7 of attention, the router and the pre-FFN norm, and 3 per expert;
/// and the embedding, final norm and head.
const TENSORS: usize = 3 + 48 * (9 + 3 * 128);

fn config() -> HfConfig {
    HfConfig::parse(CONFIG, None).unwrap()
}

/// Qwen3-30B-A3B's safetensors tensors, derived from its configuration: (name, shape outermost
/// first), sorted as a safetensors header lists them. Every tensor is BF16.
fn derived_tensors() -> Vec<(String, Vec<u64>)> {
    let (hidden, vocab, q, kv, d_head, inter) = (2048, 151_936, 32 * 128, 4 * 128, 128, 768);
    let mut t: Vec<(String, Vec<u64>)> = vec![
        ("lm_head.weight".into(), vec![vocab, hidden]),
        ("model.embed_tokens.weight".into(), vec![vocab, hidden]),
        ("model.norm.weight".into(), vec![hidden]),
    ];
    for l in 0..LAYERS {
        for (n, shape) in [
            ("input_layernorm.weight", vec![hidden]),
            ("mlp.gate.weight", vec![u64::from(EXPERTS), hidden]),
            ("post_attention_layernorm.weight", vec![hidden]),
            ("self_attn.k_norm.weight", vec![d_head]),
            ("self_attn.k_proj.weight", vec![kv, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, q]),
            ("self_attn.q_norm.weight", vec![d_head]),
            ("self_attn.q_proj.weight", vec![q, hidden]),
            ("self_attn.v_proj.weight", vec![kv, hidden]),
        ] {
            t.push((format!("model.layers.{l}.{n}"), shape));
        }
        for e in 0..EXPERTS {
            for (n, shape) in [
                ("down_proj.weight", vec![hidden, inter]),
                ("gate_proj.weight", vec![inter, hidden]),
                ("up_proj.weight", vec![inter, hidden]),
            ] {
                t.push((format!("model.layers.{l}.mlp.experts.{e}.{n}"), shape));
            }
        }
    }
    t.sort();
    t
}

fn plan(hp: &HyperParams, tensors: &[(String, Vec<u64>)]) -> TensorPlan {
    classify_tensors(
        &QWEN_MOE,
        hp,
        tensors.iter().map(|(n, s)| (n.as_str(), s.as_slice())),
    )
}

/// A small `config.json` of Qwen3-MoE, with `fields` added or replacing the base's.
fn config_with(fields: &str) -> HfConfig {
    config_named("Qwen3MoeForCausalLM", "qwen3_moe", fields)
}

fn config_named(arch: &str, model_type: &str, fields: &str) -> HfConfig {
    let base = [
        ("num_hidden_layers", "2"),
        ("hidden_size", "64"),
        ("vocab_size", "256"),
        ("num_attention_heads", "4"),
        ("num_key_value_heads", "2"),
        ("head_dim", "16"),
        ("moe_intermediate_size", "32"),
        ("num_experts", "4"),
        ("num_experts_per_tok", "2"),
        ("norm_topk_prob", "true"),
        ("rms_norm_eps", "1e-6"),
        ("max_position_embeddings", "4096"),
    ];
    let mut json = format!(r#"{{"architectures": ["{arch}"], "model_type": "{model_type}""#);
    for (k, v) in base {
        if !fields.contains(&format!("\"{k}\"")) {
            write!(json, r#", "{k}": {v}"#).unwrap();
        }
    }
    if !fields.is_empty() {
        write!(json, ", {fields}").unwrap();
    }
    json.push('}');
    HfConfig::parse(json.as_bytes(), None).unwrap()
}

fn parse(c: &HfConfig) -> Result<HyperParams, FamilyError> {
    QWEN_MOE.parse_hparams(&MetaView::Config(c))
}

fn code3(c: &HfConfig, key: &str) {
    let e = parse(c).unwrap_err();
    assert_eq!(e.code(), 3, "{e}");
    assert!(e.to_string().contains(key), "{e} does not name {key}");
}

/// Qwen3-30B-A3B's metadata as a GGUF converter writes it.
fn gguf_kvs() -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("qwen3moe")),
        ("qwen3moe.block_count", Kv::U32(48)),
        ("qwen3moe.context_length", Kv::U32(40960)),
        ("qwen3moe.embedding_length", Kv::U32(2048)),
        ("qwen3moe.feed_forward_length", Kv::U32(6144)),
        ("qwen3moe.attention.head_count", Kv::U32(32)),
        ("qwen3moe.attention.head_count_kv", Kv::U32(4)),
        ("qwen3moe.rope.freq_base", Kv::F32(1_000_000.0)),
        ("qwen3moe.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
        ("qwen3moe.attention.key_length", Kv::U32(128)),
        ("qwen3moe.attention.value_length", Kv::U32(128)),
        ("qwen3moe.expert_count", Kv::U32(128)),
        ("qwen3moe.expert_used_count", Kv::U32(8)),
        ("qwen3moe.expert_feed_forward_length", Kv::U32(768)),
        ("qwen3moe.vocab_size", Kv::U32(151_936)),
    ]
}

fn gguf_hp(kvs: &[(&'static str, Kv)]) -> Result<HyperParams, FamilyError> {
    let h = header(kvs, &[]);
    QWEN_MOE.parse_hparams(&MetaView::Gguf(&h.metadata))
}

// ---- Golden census --------------------------------------------------------------------------------

/// `name` with the number after `layers.`, `blk.` or `experts.` replaced by `{L}` or `{E}`.
fn fold(name: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for part in name.split('.') {
        let folded = match out.last() {
            Some(&("layers" | "blk")) if part.parse::<u32>().is_ok() => "{L}",
            Some(&"experts") if part.parse::<u32>().is_ok() => "{E}",
            _ => part,
        };
        out.push(folded);
    }
    out.join(".")
}

/// A class with its layer and expert indices folded out.
fn fold_class(c: TensorClass) -> String {
    match c {
        TensorClass::Expert { .. } => "Expert".into(),
        TensorClass::Router { .. } => "Router".into(),
        other => format!("{other:?}"),
    }
}

fn fold_layer(l: Option<LayerIx>) -> &'static str {
    if l.is_some() { "{L}" } else { "-" }
}

/// Appends `lines` under `title`, each distinct line once with its count.
fn counted(out: &mut String, title: &str, lines: impl IntoIterator<Item = String>) {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for l in lines {
        *counts.entry(l).or_default() += 1;
    }
    writeln!(
        out,
        "# {title} ({} lines, {} distinct)",
        counts.values().sum::<usize>(),
        counts.len()
    )
    .unwrap();
    for (line, n) in counts {
        writeln!(out, "{n} × {line}").unwrap();
    }
}

/// The family's result and the census builder's manifest, folded (module docs): every field
/// llama's golden file holds, except the source extents and `ModelId`, which depend on the
/// file's bytes.
fn render(hp: &HyperParams, plan: &TensorPlan, kv: &KvFormula, m: &ModelManifest) -> String {
    let mut out = String::new();
    writeln!(out, "# hyperparameters").unwrap();
    writeln!(
        out,
        "n_layers={} hidden={} vocab={} residual_streams={}",
        hp.n_layers, hp.hidden, hp.vocab, hp.residual_streams
    )
    .unwrap();
    for (k, v) in &hp.extra {
        writeln!(out, "{k}={v:?}").unwrap();
    }
    writeln!(out, "# kv").unwrap();
    writeln!(out, "layers={} first={:?}", kv.layers.len(), kv.layers[0]).unwrap();
    writeln!(
        out,
        "all_equal={}",
        kv.layers.iter().all(|l| *l == kv.layers[0])
    )
    .unwrap();
    writeln!(out, "dtype={:?}", kv.dtype).unwrap();
    counted(
        &mut out,
        "fetch",
        plan.fetch.iter().map(|(name, c)| {
            let Classification::Classified {
                class,
                layer,
                role,
                stacked_experts,
                block,
            } = c
            else {
                unreachable!("fetch holds classified tensors only")
            };
            format!(
                "{} class={} layer={} role={role:?} block={block:?} stacked={stacked_experts:?}",
                fold(name),
                fold_class(*class),
                fold_layer(*layer)
            )
        }),
    );
    writeln!(out, "# ignored ({})", plan.ignored.len()).unwrap();
    for (name, reason) in &plan.ignored {
        writeln!(out, "{name}: {reason}").unwrap();
    }
    writeln!(out, "# unknown ({})", plan.unknown.len()).unwrap();
    for name in &plan.unknown {
        writeln!(out, "{name}").unwrap();
    }

    writeln!(out, "# manifest").unwrap();
    writeln!(out, "family={} format={:?}", m.family, m.format).unwrap();
    counted(
        &mut out,
        "records",
        m.census.tensors.iter().map(|r| {
            format!(
                "{} class={} layer={} shape={:?} quant={:?} bytes={:?} aux={}",
                fold(&r.name),
                fold_class(r.class),
                fold_layer(r.layer),
                r.shape,
                r.quant,
                r.bytes,
                r.aux.len()
            )
        }),
    );
    let name = |id| {
        m.census
            .tensors
            .iter()
            .find(|t| t.id == id)
            .map_or("?".into(), |t| fold(&t.name))
    };
    let group_key = |id: TensorGroupId| {
        m.census
            .groups
            .iter()
            .find(|g| g.id == id)
            .map(|g| format!("{}@{}", fold_class(g.class), fold_layer(g.layer)))
    };
    counted(
        &mut out,
        "groups",
        m.census.groups.iter().map(|g| {
            let mut members: Vec<String> = g.tensors.iter().map(|&t| name(t)).collect();
            members.sort_unstable();
            format!(
                "{}@{} bytes={:?} read_per_token={:?} shared_with={:?} tensors={members:?}",
                fold_class(g.class),
                fold_layer(g.layer),
                g.bytes,
                g.read_per_token,
                g.shared_with.and_then(group_key)
            )
        }),
    );
    writeln!(out, "# manifest ignored ({})", m.ignored.len()).unwrap();
    for (n, reason) in &m.ignored {
        writeln!(out, "{n}: {reason}").unwrap();
    }
    writeln!(out, "# totals").unwrap();
    for (kind, total) in &m.census.totals.by_class {
        writeln!(out, "{kind:?}: {total:?}").unwrap();
    }
    out
}

/// Resolves, parses and classifies a Qwen3-30B-A3B checkpoint from `config` and its shard
/// `headers`, builds its manifest with the census builder (COMP-E10-T07), and checks both
/// against the golden file. `rewrite` lets this run regenerate it.
fn check_census(config: &HfConfig, headers: &[SafetensorsHeader], rewrite: bool) {
    let tensors: Vec<(String, Vec<u64>)> = headers
        .iter()
        .flat_map(|h| &h.tensors)
        .map(|t| (t.name.clone(), t.shape.clone()))
        .collect();
    assert_eq!(tensors.len(), TENSORS, "Qwen3-30B-A3B's tensor count");
    let meta = MetaView::Config(config);
    let family = families().resolve(&meta).unwrap();
    assert_eq!(family.name(), TAG);
    let hp = family.parse_hparams(&meta).unwrap();
    let plan = plan(&hp, &tensors);
    let kv_source = KvSource::Safetensors {
        compute: Some(DType::BF16),
        quantized: false,
    };
    let kv = family.kv_formula(&hp, kv_source);
    let files: Vec<SourceFile> = headers
        .iter()
        .map(|h| SourceFile {
            role: SourceRole::Weights,
            digest: FileDigest([0; 32]),
            len: h.file_len,
        })
        .collect();
    let manifest = census::build(&Sources {
        meta,
        weights: Weights::Safetensors(headers),
        kv_source,
        tokenizer: None,
        files: &files,
    })
    .unwrap();
    assert_eq!(manifest.family, TAG);
    assert_eq!(
        manifest.hparams, hp,
        "the builder's hyperparameters are the family's"
    );
    assert_eq!(manifest.kv, kv, "the builder's KV formula is the family's");
    let actual = render(&hp, &plan, &kv, &manifest);
    if rewrite && update_golden() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/qwen_moe/golden/qwen3-30b-a3b.txt");
        std::fs::write(path, &actual).unwrap();
        return;
    }
    assert_eq!(
        actual,
        GOLDEN.replace("\r\n", "\n"),
        "the census differs from the golden file; the actual census is:\n{actual}"
    );

    // The facts the golden file holds, stated on their own so a regenerated file cannot
    // quietly change them.
    assert!(plan.unknown.is_empty(), "{:?}", plan.unknown);
    assert!(plan.ignored.is_empty(), "untied: the head is fetched");
    assert_eq!(plan.fetch.len(), TENSORS);
    assert_eq!((hp.n_layers, hp.hidden, hp.vocab), (48, 2048, 151_936));
    assert_eq!(hp.count(N_HEADS), Some(32));
    assert_eq!(hp.count(N_KV_HEADS), Some(4));
    assert_eq!(hp.count(D_HEAD), Some(128));
    assert_eq!(hp.count(N_EXPERTS), Some(128));
    assert_eq!(hp.count(TOP_K), Some(8));
    assert_eq!(hp.count(D_EXPERT), Some(768));
    assert_eq!(hp.extra.get(RENORMALIZE), Some(&HParamValue::Bool(true)));
    assert!(llama::flag(&hp, QK_NORM) && !llama::flag(&hp, QKV_BIAS));
    assert!(!llama::flag(&hp, llama::TIED_EMBEDDINGS));
    assert_eq!(kv.layers, vec![LayerKv::gqa(4, 128, None); 48]);

    // Every (layer, expert) has its own group of gate, up and down; each layer has one router
    // group holding the router and the pre-FFN norm, and one attention group.
    let groups = &manifest.census.groups;
    let experts: BTreeSet<(u16, u16)> = groups
        .iter()
        .filter_map(|g| match g.class {
            TensorClass::Expert { layer, id } => {
                assert_eq!(g.tensors.len(), 3);
                Some((layer.0, id.0))
            }
            _ => None,
        })
        .collect();
    let all: BTreeSet<(u16, u16)> = (0..LAYERS)
        .flat_map(|l| (0..EXPERTS).map(move |e| (l, e)))
        .collect();
    assert_eq!(experts, all);
    let routers: BTreeSet<u16> = groups
        .iter()
        .filter_map(|g| match g.class {
            TensorClass::Router { layer } => {
                assert_eq!(g.tensors.len(), 2, "the router and the pre-FFN norm");
                Some(layer.0)
            }
            _ => None,
        })
        .collect();
    assert_eq!(routers, (0..LAYERS).collect());
    assert_eq!(groups.len(), 1 + 48 * (1 + 1 + 128) + 1);
    for g in groups {
        if let TensorClass::Expert { .. } = g.class {
            assert_eq!(g.read_per_token.0, g.bytes.0 * 8 / 128, "top_k / n_experts");
        }
    }
    let head = groups
        .iter()
        .find(|g| g.class == TensorClass::Head)
        .unwrap();
    assert_eq!(head.shared_with, None, "untied");
    assert_eq!(manifest.census.tensors.len(), TENSORS);
    assert!(manifest.ignored.is_empty());
}

/// COMP-E16-T04's golden census, on headers derived from Qwen3-30B-A3B's configuration: the
/// checkpoint resolves to `qwen_moe` through this build's registry from its metadata alone,
/// every tensor is classified, and the result matches the golden file.
#[test]
fn golden_census_on_qwen3_30b_a3b_derived_headers() {
    let header = read_header_default(safetensors_header(&derived_tensors())).unwrap();
    check_census(&config(), std::slice::from_ref(&header), true);
}

/// COMP-E16-T04's golden census on the recorded files (QUAL-E7-T06), opt-in:
/// `SYNTROPY_QWEN3_30B_A3B_DIR` names a directory holding Qwen3-30B-A3B's `config.json`,
/// `generation_config.json` and its 16 shard headers, `model-0000N-of-00016.safetensors.header`,
/// at revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, verified against
/// `testdata/fixtures.toml`'s BLAKE3 with `cargo xtask fixtures`. No weight is read: each header
/// is parsed against its shard's size on the Hub, and must describe the whole shard. The census
/// must match the same golden file as the derived header's.
#[test]
fn golden_census_on_the_recorded_qwen3_30b_a3b_headers() {
    let Some(dir) = std::env::var_os(DIR_VAR) else {
        eprintln!(
            "NOT RUN: {DIR_VAR} is not set; the census on the recorded Qwen3-30B-A3B headers \
             needs the fixtures (see this module's documentation)"
        );
        return;
    };
    let dir = Path::new(&dir);
    let read = |f: &str| std::fs::read(dir.join(f)).unwrap();
    let config =
        HfConfig::parse(&read("config.json"), Some(&read("generation_config.json"))).unwrap();
    let headers: Vec<SafetensorsHeader> = SHARD_BYTES
        .iter()
        .enumerate()
        .map(|(i, &len)| {
            let bytes = read(&format!("model-{:05}-of-00016.safetensors.header", i + 1));
            let n = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).unwrap();
            assert_eq!(
                bytes.len(),
                8 + n,
                "the recorded range is the header exactly"
            );
            let h = parse_header(&bytes[8..], len).unwrap();
            let end = h.tensors.iter().map(|t| t.offset + t.len).max().unwrap();
            assert_eq!(end, len, "shard {}'s header covers the whole shard", i + 1);
            h
        })
        .collect();
    check_census(&config, &headers, false);
    eprintln!("ran on {}", dir.display());
}

// ---- KV formula and dtype -----------------------------------------------------------------------

/// `KvFormula::dtype` is F16 for a GGUF source and BF16 for a BF16 safetensors source, by
/// COMP-E13-T06's rule, with the same per-layer entries.
#[test]
fn kv_dtype_is_f16_for_gguf_and_bf16_for_bf16_safetensors() {
    let hp = parse(&config()).unwrap();
    let gguf = QWEN_MOE.kv_formula(&hp, KvSource::Gguf);
    let st = QWEN_MOE.kv_formula(
        &hp,
        KvSource::Safetensors {
            compute: Some(DType::BF16),
            quantized: false,
        },
    );
    for l in [0, 23, 47] {
        assert_eq!(gguf.dtype.of(LayerIx(l)), DType::F16);
        assert_eq!(st.dtype.of(LayerIx(l)), DType::BF16);
    }
    assert!(gguf.dtype.layers.is_empty() && st.dtype.layers.is_empty());
    assert_eq!(gguf.layers, st.layers);
    assert_eq!(gguf.layers.len(), 48);
    assert_eq!(
        gguf.layers[0],
        LayerKv::PerToken {
            elems_per_token: 2 * 4 * 128,
            window: None,
            latent: false,
        }
    );
    for source in [
        KvSource::Gguf,
        KvSource::Safetensors {
            compute: Some(DType::F16),
            quantized: false,
        },
        KvSource::Safetensors {
            compute: None,
            quantized: true,
        },
    ] {
        let rule = KvDTypeRule {
            source,
            declared_cache: None,
            family: None,
            family_layers: Vec::new(),
        };
        assert_eq!(
            QWEN_MOE.kv_formula(&hp, source).dtype,
            rule.reference().unwrap()
        );
    }
}

// ---- Both sources ---------------------------------------------------------------------------------

/// The same model from GGUF and from safetensors gives equal hyperparameters (RFC-0043 §5.2).
#[test]
fn gguf_and_config_give_equal_hyperparameters() {
    assert_eq!(gguf_hp(&gguf_kvs()).unwrap(), parse(&config()).unwrap());
}

/// `norm_topk_prob` maps to `renormalize`: as stated, false when `config.json` leaves it out,
/// and true when GGUF does.
#[test]
fn norm_topk_prob_maps_to_renormalize() {
    let renorm = |hp: HyperParams| moe_params(&hp).unwrap().renormalize;
    assert!(renorm(
        parse(&config_with(r#""norm_topk_prob": true"#)).unwrap()
    ));
    assert!(!renorm(
        parse(&config_with(r#""norm_topk_prob": false"#)).unwrap()
    ));
    let without = String::from_utf8(CONFIG.to_vec())
        .unwrap()
        .replace("\"norm_topk_prob\": true,", "");
    assert!(!renorm(
        parse(&HfConfig::parse(without.as_bytes(), None).unwrap()).unwrap()
    ));
    assert!(renorm(gguf_hp(&gguf_kvs()).unwrap()));
    let off = with(
        &gguf_kvs(),
        &[("qwen3moe.expert_weights_norm", Kv::Bool(false))],
    );
    assert!(!renorm(gguf_hp(&off).unwrap()));
    // A value of the wrong type is refused, not read as either.
    code3(&config_with(r#""norm_topk_prob": 1"#), "norm_topk_prob");
}

/// GGUF's stacked expert tensors, router and norm classify; the census slices each stacked
/// tensor into one record per expert and groups them per expert.
#[test]
fn gguf_stacked_experts_classify_and_slice() {
    let hp = gguf_hp(&gguf_kvs()).unwrap();
    let stacked = |name: &str, shape: &[u64]| QWEN_MOE.classify(name, shape, &hp);
    for (name, shape, role) in [
        (
            "blk.0.ffn_gate_exps.weight",
            [2048u64, 768, 128],
            TensorRole::Gate,
        ),
        (
            "blk.47.ffn_up_exps.weight",
            [2048, 768, 128],
            TensorRole::Up,
        ),
        (
            "blk.3.ffn_down_exps.weight",
            [768, 2048, 128],
            TensorRole::Down,
        ),
    ] {
        let layer = name[4..].split('.').next().unwrap().parse().unwrap();
        assert_eq!(
            stacked(name, &shape),
            Classification::Classified {
                class: TensorClass::Expert {
                    layer: LayerIx(layer),
                    id: ExpertIx(0),
                },
                layer: Some(LayerIx(layer)),
                role,
                stacked_experts: Some(128),
                block: BlockSel::Layer(Block::Moe),
            },
            "{name}"
        );
        // Fewer experts than the hyperparameters name, or the expert axis not outermost.
        let mut fewer = shape;
        fewer[2] = 127;
        assert_eq!(stacked(name, &fewer), Classification::Unknown);
        let mut moved = shape;
        moved.rotate_right(1);
        assert_eq!(stacked(name, &moved), Classification::Unknown);
    }
    assert!(matches!(
        stacked("blk.0.ffn_gate_inp.weight", &[2048, 128]),
        Classification::Classified {
            class: TensorClass::Router { layer: LayerIx(0) },
            role: TensorRole::Router,
            ..
        }
    ));
    assert_eq!(
        stacked("blk.0.ffn_gate_inp.weight", &[128, 2048]),
        Classification::Unknown
    );
    assert!(matches!(
        stacked("blk.0.ffn_norm.weight", &[2048]),
        Classification::Classified {
            class: TensorClass::DenseEveryToken,
            block: BlockSel::Layer(Block::Moe),
            ..
        }
    ));
    for name in [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_gate_inp_shexp.weight",
        "blk.0.exp_probs_b.bias",
        "blk.48.ffn_gate_exps.weight",
    ] {
        assert_eq!(
            stacked(name, &[2048, 768, 128]),
            Classification::Unknown,
            "{name}"
        );
    }

    // A two-layer, four-expert GGUF through the census builder.
    let kvs = with(
        &gguf_kvs(),
        &[
            ("qwen3moe.block_count", Kv::U32(2)),
            ("qwen3moe.embedding_length", Kv::U32(64)),
            ("qwen3moe.attention.head_count", Kv::U32(4)),
            ("qwen3moe.attention.head_count_kv", Kv::U32(2)),
            ("qwen3moe.attention.key_length", Kv::U32(16)),
            ("qwen3moe.attention.value_length", Kv::U32(16)),
            ("qwen3moe.expert_count", Kv::U32(4)),
            ("qwen3moe.expert_used_count", Kv::U32(2)),
            ("qwen3moe.expert_feed_forward_length", Kv::U32(32)),
            ("qwen3moe.vocab_size", Kv::U32(256)),
        ],
    );
    let per_layer: &[(&str, &[u64])] = &[
        ("attn_norm.weight", &[64]),
        ("attn_q.weight", &[64, 64]),
        ("attn_k.weight", &[64, 32]),
        ("attn_v.weight", &[64, 32]),
        ("attn_output.weight", &[64, 64]),
        ("attn_q_norm.weight", &[16]),
        ("attn_k_norm.weight", &[16]),
        ("ffn_norm.weight", &[64]),
        ("ffn_gate_inp.weight", &[64, 4]),
        ("ffn_gate_exps.weight", &[64, 32, 4]),
        ("ffn_up_exps.weight", &[64, 32, 4]),
        ("ffn_down_exps.weight", &[32, 64, 4]),
    ];
    let mut names: Vec<(String, Vec<u64>)> = vec![
        ("token_embd.weight".into(), vec![64, 256]),
        ("output_norm.weight".into(), vec![64]),
        ("output.weight".into(), vec![64, 256]),
    ];
    for l in 0..2 {
        for (n, s) in per_layer {
            names.push((format!("blk.{l}.{n}"), s.to_vec()));
        }
    }
    let tensors: Vec<(&str, &[u64])> = names
        .iter()
        .map(|(n, s)| (n.as_str(), s.as_slice()))
        .collect();
    let bytes = crate::families::llama::tests::gguf(&kvs, &tensors);
    let h = syntropy_manifest::gguf::GgufHeader::read(
        Bytes(bytes.clone()),
        &syntropy_manifest::gguf::Limits::default(),
    )
    .unwrap();
    let files = [SourceFile {
        role: SourceRole::Weights,
        digest: FileDigest([0; 32]),
        len: bytes.len() as u64,
    }];
    let m = census::build(&Sources {
        meta: MetaView::Gguf(&h.metadata),
        weights: Weights::Gguf(&h),
        kv_source: KvSource::Gguf,
        tokenizer: None,
        files: &files,
    })
    .unwrap();
    assert_eq!(m.family, TAG);
    let experts: Vec<_> = m
        .census
        .groups
        .iter()
        .filter(|g| matches!(g.class, TensorClass::Expert { .. }))
        .collect();
    assert_eq!(experts.len(), 2 * 4);
    for g in &experts {
        assert_eq!(g.tensors.len(), 3, "gate, up and down of one expert");
        assert_eq!(
            g.bytes.0,
            3 * 64 * 32 * 4,
            "one expert's slice of each, in F32"
        );
        assert_eq!(g.read_per_token.0, g.bytes.0 * 2 / 4);
    }
    assert_eq!(
        m.census
            .groups
            .iter()
            .filter(|g| matches!(g.class, TensorClass::Router { .. }))
            .map(|g| g.tensors.len())
            .collect::<Vec<_>>(),
        [2, 2],
        "the router and the pre-FFN norm, per layer"
    );
    assert_eq!(m.kv.dtype.default, DType::F16);
}

// ---- Fail closed --------------------------------------------------------------------------------

#[test]
fn other_architectures_are_code_1() {
    for (arch, model_type) in [
        ("Qwen3ForCausalLM", "qwen3"),
        ("Qwen2MoeForCausalLM", "qwen2_moe"),
        ("MixtralForCausalLM", "mixtral"),
    ] {
        let e = parse(&config_named(arch, model_type, "")).unwrap_err();
        assert_eq!(e.code(), 1, "{arch}");
    }
    let e = gguf_hp(&with(
        &gguf_kvs(),
        &[("general.architecture", Kv::Str("qwen2moe"))],
    ))
    .unwrap_err();
    assert_eq!(e.code(), 1);
    // A configuration whose names point at both families is refused by the census as
    // ambiguous, never resolved by order.
    let c = config_named("Qwen3MoeForCausalLM", "qwen3", "");
    let e = census::build(&Sources {
        meta: MetaView::Config(&c),
        weights: Weights::Safetensors(&[]),
        kv_source: KvSource::Gguf,
        tokenizer: None,
        files: &[],
    })
    .unwrap_err();
    assert!(matches!(e, CensusError::AmbiguousFamily { .. }), "{e}");
    assert_eq!(e.code(), Some(1));
}

#[test]
fn missing_or_out_of_range_expert_counts_are_code_3() {
    for key in [
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "rms_norm_eps",
    ] {
        let json = String::from_utf8(CONFIG.to_vec()).unwrap();
        let without = json.replace(&format!("\"{key}\""), &format!("\"renamed_{key}\""));
        assert_ne!(without, json);
        code3(&HfConfig::parse(without.as_bytes(), None).unwrap(), key);
    }
    code3(&config_with(r#""num_experts": 0"#), "num_experts");
    code3(&config_with(r#""num_experts": 65536"#), "num_experts");
    code3(
        &config_with(r#""num_experts_per_tok": 0"#),
        "num_experts_per_tok",
    );
    code3(
        &config_with(r#""num_experts_per_tok": 5"#),
        "num_experts_per_tok",
    );
    code3(
        &config_with(r#""moe_intermediate_size": 0"#),
        "moe_intermediate_size",
    );
    // top_k = E is the edge that is allowed.
    let hp = parse(&config_with(r#""num_experts_per_tok": 4"#)).unwrap();
    assert_eq!(hp.count(TOP_K), Some(4));
    for key in [
        "qwen3moe.expert_count",
        "qwen3moe.expert_used_count",
        "qwen3moe.expert_feed_forward_length",
    ] {
        let kvs: Vec<_> = gguf_kvs().into_iter().filter(|(k, _)| *k != key).collect();
        let e = gguf_hp(&kvs).unwrap_err();
        assert_eq!(e.code(), 3);
        assert!(e.to_string().contains(key), "{e}");
    }
    let e = gguf_hp(&with(
        &gguf_kvs(),
        &[("qwen3moe.expert_used_count", Kv::U32(129))],
    ))
    .unwrap_err();
    assert!(e.to_string().contains("expert_used_count"), "{e}");
}

/// Layouts the `Moe` block cannot compute are refused, never ignored; the forms Qwen3-MoE
/// configurations carry with their default values parse.
#[test]
fn layouts_the_moe_block_cannot_compute_are_code_3() {
    code3(
        &config_with(r#""decoder_sparse_step": 2"#),
        "decoder_sparse_step",
    );
    code3(
        &config_with(r#""decoder_sparse_step": 0"#),
        "decoder_sparse_step",
    );
    code3(&config_with(r#""mlp_only_layers": [0]"#), "mlp_only_layers");
    code3(
        &config_with(r#""shared_expert_intermediate_size": 512"#),
        "shared_expert_intermediate_size",
    );
    code3(
        &config_with(r#""num_shared_experts": 1"#),
        "num_shared_experts",
    );
    code3(&config_with(r#""n_shared_experts": 2"#), "n_shared_experts");
    code3(&config_with(r#""scoring_func": "sigmoid""#), "scoring_func");
    // `llama`'s refusals hold here too: they are the same code.
    code3(
        &config_with(r#""use_sliding_window": true, "sliding_window": 4096"#),
        "sliding_window",
    );
    code3(&config_with(r#""attention_bias": true"#), "attention_bias");
    code3(&config_with(r#""hidden_act": "gelu""#), "hidden_act");
    code3(
        &config_with(r#""partial_rotary_factor": 0.5"#),
        "partial_rotary_factor",
    );
    code3(
        &config_with(r#""rope_scaling": {"rope_type": "dynamic", "factor": 2.0}"#),
        "rope_type",
    );
    parse(&config_with(
        r#""decoder_sparse_step": 1, "mlp_only_layers": [], "shared_expert_intermediate_size": 0,
           "scoring_func": "softmax", "use_sliding_window": false, "sliding_window": null"#,
    ))
    .unwrap();
    for (key, value) in [
        ("qwen3moe.expert_shared_count", Kv::U32(1)),
        ("qwen3moe.expert_shared_feed_forward_length", Kv::U32(512)),
        ("qwen3moe.expert_gating_func", Kv::U32(2)),
        ("qwen3moe.attention.sliding_window", Kv::U32(4096)),
    ] {
        let e = gguf_hp(&with(&gguf_kvs(), &[(key, value)])).unwrap_err();
        assert_eq!(e.code(), 3, "{key}");
        assert!(e.to_string().contains(key), "{e}");
    }
    gguf_hp(&with(
        &gguf_kvs(),
        &[
            ("qwen3moe.expert_shared_count", Kv::U32(0)),
            ("qwen3moe.expert_gating_func", Kv::U32(1)),
        ],
    ))
    .unwrap();
}

/// An expert index past the last, a wrong shape, a layer past the last, and a tensor no rule
/// has are `Unknown`; a planted one refuses the census as `UnknownTensor` (3).
#[test]
fn expert_indices_shapes_and_strangers_fail_closed() {
    let hp = parse(&config()).unwrap();
    let tensors: Vec<_> = derived_tensors()
        .into_iter()
        .filter(|(n, _)| {
            !n.starts_with("model.layers.")
                || n.starts_with("model.layers.0.")
                || n.starts_with("model.layers.47.")
        })
        .collect();
    for (name, shape) in &tensors {
        assert!(
            matches!(
                QWEN_MOE.classify(name, shape, &hp),
                Classification::Classified { .. }
            ),
            "{name}"
        );
        for d in 0..shape.len() {
            let mut wrong = shape.clone();
            wrong[d] += 1;
            assert_eq!(
                QWEN_MOE.classify(name, &wrong, &hp),
                Classification::Unknown,
                "{name} {wrong:?}"
            );
        }
        let mut transposed = shape.clone();
        transposed.reverse();
        if transposed != *shape {
            assert_eq!(
                QWEN_MOE.classify(name, &transposed, &hp),
                Classification::Unknown,
                "{name} transposed"
            );
        }
    }
    for (name, shape) in [
        (
            "model.layers.0.mlp.experts.128.gate_proj.weight",
            [768u64, 2048],
        ),
        (
            "model.layers.0.mlp.experts.01.gate_proj.weight",
            [768, 2048],
        ),
        ("model.layers.48.mlp.experts.0.up_proj.weight", [768, 2048]),
        ("model.layers.48.mlp.gate.weight", [128, 2048]),
        (
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            [768, 2048],
        ),
        ("model.layers.0.mlp.shared_expert_gate.weight", [1, 2048]),
        ("model.layers.0.mlp.gate.bias", [128, 1]),
        ("model.layers.0.mlp.gate_proj.weight", [6144, 2048]),
        (
            "model.layers.0.mlp.experts.0.gate_up_proj.weight",
            [1536, 2048],
        ),
    ] {
        assert_eq!(
            QWEN_MOE.classify(name, &shape, &hp),
            Classification::Unknown,
            "{name}"
        );
    }
    // A planted shared expert in the header refuses the census, named.
    let mut planted = derived_tensors();
    planted.push((
        "model.layers.0.mlp.shared_expert.gate_proj.weight".into(),
        vec![768, 2048],
    ));
    planted.sort();
    let header = read_header_default(safetensors_header(&planted)).unwrap();
    let c = config();
    let files = [SourceFile {
        role: SourceRole::Weights,
        digest: FileDigest([0; 32]),
        len: header.file_len,
    }];
    let e = census::build(&Sources {
        meta: MetaView::Config(&c),
        weights: Weights::Safetensors(std::slice::from_ref(&header)),
        kv_source: KvSource::Gguf,
        tokenizer: None,
        files: &files,
    })
    .unwrap_err();
    assert_eq!(
        e,
        CensusError::UnknownTensor {
            names: vec!["model.layers.0.mlp.shared_expert.gate_proj.weight".into()]
        }
    );
    // A layer missing one expert is refused too.
    let missing: Vec<_> = derived_tensors()
        .into_iter()
        .filter(|(n, _)| !n.starts_with("model.layers.5.mlp.experts.77."))
        .collect();
    let header = read_header_default(safetensors_header(&missing)).unwrap();
    let e = census::build(&Sources {
        meta: MetaView::Config(&c),
        weights: Weights::Safetensors(std::slice::from_ref(&header)),
        kv_source: KvSource::Gguf,
        tokenizer: None,
        files: &files,
    })
    .unwrap_err();
    assert_eq!(e.code(), Some(6), "{e}");
}

// ---- Registry, ops, spec ----------------------------------------------------------------------

/// `qwen_moe` is registered under its frozen tag, and no architecture name it accepts is one
/// `llama` accepts, so resolution never depends on registration order.
#[test]
fn the_registry_binds_qwen_moe_to_its_frozen_tag() {
    families().check(FROZEN_TAGS).unwrap();
    assert_eq!(families().tags(), ["llama", "qwen_moe"]);
    assert_eq!(MODULE, "syntropy_graph::families::qwen_moe");
    let frozen = FROZEN_TAGS.iter().find(|t| t.tag == TAG).unwrap();
    assert_eq!(frozen.module, MODULE);
    assert_eq!(
        QWEN_MOE.accepts(),
        ["qwen3moe", "Qwen3MoeForCausalLM", "qwen3_moe"]
    );
    for name in QWEN_MOE.accepts() {
        assert!(!LLAMA.accepts().contains(name), "{name} is also llama's");
        assert_eq!(families().resolve_names(&[name]).unwrap().name(), TAG);
    }
    for name in LLAMA.accepts() {
        assert_eq!(families().resolve_names(&[name]).unwrap().name(), "llama");
    }
}

/// Both matchers, untied and tied, compile.
#[test]
fn every_rule_table_combination_compiles() {
    for tied in [false, true] {
        Matcher::new(&tables(tied)).unwrap();
    }
}

/// A tied configuration ignores a stored head; untied, it is the head.
#[test]
fn tied_embeddings_ignore_a_stored_head() {
    let untied = parse(&config_with("")).unwrap();
    let tied = parse(&config_with(r#""tie_word_embeddings": true"#)).unwrap();
    assert!(matches!(
        QWEN_MOE.classify("lm_head.weight", &[256, 64], &untied),
        Classification::Classified {
            class: TensorClass::Head,
            ..
        }
    ));
    assert_eq!(
        QWEN_MOE.classify("lm_head.weight", &[256, 64], &tied),
        Classification::Ignored {
            reason: llama::TIED_REASON
        }
    );
}

/// `required_ops` is the union of its blocks' ops (RFC-0040 §11.1): `Gqa`, `Moe` with softmax
/// routing, `Embed` and `LmHead`, and no dense FFN.
#[test]
fn required_ops_are_the_blocks_ops() {
    let hp = parse(&config()).unwrap();
    let ops = QWEN_MOE.required_ops(&hp);
    let mut expected = llama::gqa_params(&hp).unwrap().ops();
    expected.extend(moe_params(&hp).unwrap().ops());
    expected.extend(llama::embed_params(&hp).ops());
    expected.extend(llama::head_params(&hp).ops());
    assert_eq!(ops, expected);
    for op in [
        OpKind::RowGather,
        OpKind::RmsNorm,
        OpKind::Gemv,
        OpKind::Gemm,
        OpKind::KvWrite(AttnKind::Gqa),
        OpKind::AttnDecode(AttnKind::Gqa),
        OpKind::AttnPrefill(AttnKind::Gqa),
        OpKind::MoeRoute {
            norm: RouteNorm::Softmax,
        },
        OpKind::MoeGrouped,
        OpKind::MoeCombine,
        OpKind::Add,
        OpKind::Logits,
    ] {
        assert!(ops.contains(&op), "{op:?}");
    }
    assert!(
        !ops.contains(&OpKind::GateUpAct {
            act: GatedAct::Silu
        }),
        "no dense FFN"
    );
    assert!(!ops.contains(&OpKind::Scale), "no embedding scale");
}

/// Qwen3-30B-A3B's `Moe` and `Gqa` parameters come from its hyperparameters; a missing one gives
/// none.
#[test]
fn block_parameters_follow_the_hyperparameters() {
    let hp = parse(&config()).unwrap();
    let m = moe_params(&hp).unwrap();
    assert_eq!(
        (m.experts, m.top_k, m.expert_inter, m.renormalize),
        (128, 8, 768, true)
    );
    assert_eq!(m.eps, 1e-6_f32);
    let g = llama::gqa_params(&hp).unwrap();
    assert_eq!((g.heads, g.kv_heads, g.head_dim), (32, 4, 128));
    assert_eq!(g.qkv, crate::blocks::QkvLayout::Separate { qk_norm: true });
    assert!(!g.bias);
    assert_eq!(g.rope.theta, 1_000_000.0);
    let mut bad = hp.clone();
    bad.extra.remove(RENORMALIZE);
    assert_eq!(moe_params(&bad), None);
    let mut bad = hp;
    bad.extra.remove(TOP_K);
    assert_eq!(moe_params(&bad), None);
}

/// The `FamilySpec` has one variant, Qwen3-30B-A3B's form, which resolves to `qwen_moe` by its
/// own names and parses to the attention and MoE features it states.
#[test]
fn the_fixture_spec_states_the_one_variant() {
    let spec = fixture_spec();
    assert_eq!(spec.family, TAG);
    let [v] = spec.variants else {
        panic!("one variant")
    };
    for n in [v.gguf_architecture, v.architecture, v.model_type] {
        assert_eq!(families().resolve_names(&[n]).unwrap().name(), TAG, "{n}");
    }
    assert_eq!(v.layers, [LayerKind::GqaMoe]);
    assert_eq!(
        v.moe,
        Some(MoeSpec {
            experts: 128,
            top_k: 8,
            shared: 0,
            latent: false,
            router_bias: false,
        })
    );
    assert_eq!(
        v.attention,
        AttentionSpec {
            qk_norm: true,
            qkv_bias: false,
            rope: RopeKind::Plain,
        }
    );
    assert!(!v.tied_embeddings);
    assert_eq!(v.mtp_modules, 0);
    let hp = parse(&config()).unwrap();
    let m = moe_params(&hp).unwrap();
    let s = v.moe.unwrap();
    assert_eq!((m.experts, m.top_k), (s.experts, s.top_k));
    assert_eq!(llama::flag(&hp, QK_NORM), v.attention.qk_norm);
    assert_eq!(llama::flag(&hp, QKV_BIAS), v.attention.qkv_bias);
    assert_eq!(llama::flag(&hp, llama::TIED_EMBEDDINGS), v.tied_embeddings);
}
