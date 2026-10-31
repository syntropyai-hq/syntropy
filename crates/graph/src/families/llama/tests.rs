// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the `llama` family against COMP-E16-T03's acceptance criteria and RFC-0043 §4,
//! §5.3 and §5.5.
//!
//! - **Golden census.** The family's classification of Qwen3-0.6B and the manifest the census
//!   builder (COMP-E10-T07) makes from it are compared with `golden/qwen3-0.6b.txt` twice. In CI it runs on a configuration and a safetensors header *derived* here from
//!   Qwen3-0.6B's published `config.json`: names and shapes only, written by this file, with
//!   no file of the checkpoint copied. With `SYNTROPY_QWEN3_0_6B_DIR` set to a directory of
//!   the real files, fetched and verified with `cargo xtask fixtures` at the revision
//!   QUAL-E7-T06 pinned, it also runs on the recorded `config.json`,
//!   `generation_config.json` and `model.safetensors` header; unset, that test prints that it
//!   did not run. To regenerate the golden file after an intended change, run the tests with
//!   `SYNTROPY_UPDATE_SNAPSHOTS=1` and review the diff. The variable is refused when `CI` is
//!   set.
//! - **KV dtype.** F16 for a GGUF source and BF16 for a BF16 safetensors source, by
//!   COMP-E13-T06's rule.
//! - **Fail closed.** Unknown architectures, missing and out-of-range hyperparameters,
//!   settings the blocks cannot compute, names that match with the wrong shape, tensors the
//!   variant does not have, and a `kv_dtype` that narrows the reference's.

use std::fmt::Write as _;
use std::path::Path;

use syntropy_core::TensorGroupId;
use syntropy_core::{DType, LayerIx};
use syntropy_kernels::{AttnKind, GatedAct, OpKind};
use syntropy_manifest::census::{ModelManifest, TensorClass, TensorId};
use syntropy_manifest::config::HfConfig;
use syntropy_manifest::gguf::{GgufHeader, Limits};
use syntropy_manifest::identity::{FileDigest, SourceFile};
use syntropy_manifest::kv::KvDTypeOverride;
use syntropy_manifest::range::{FileSource, RangeError, RangeSource};
use syntropy_manifest::safetensors::SafetensorsHeader;
use syntropy_manifest::safetensors::read_header_default;
use syntropy_manifest::safetensors::side::SourceRole;

use super::*;
use crate::census::{self, Sources, Weights};
use crate::classify::{TensorPlan, classify_tensors};
use crate::families::{FROZEN_TAGS, Registry, families};

// ---- Fixtures -----------------------------------------------------------------------------------

/// A `config.json` for Qwen3-0.6B, derived from the published one at revision
/// `c1899de289a04d12100db370d81485cdf75e47ca`: the keys the family reads, with their values.
const QWEN3_CONFIG: &[u8] = br#"{
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "num_hidden_layers": 28,
    "hidden_size": 1024,
    "vocab_size": 151936,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000,
    "rope_scaling": null,
    "max_position_embeddings": 40960,
    "sliding_window": null,
    "use_sliding_window": false,
    "attention_bias": false,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16"
}"#;
const QWEN3_GOLDEN: &str = include_str!("golden/qwen3-0.6b.txt");
/// The directory of the real Qwen3-0.6B files, for the opt-in test.
const QWEN3_DIR_VAR: &str = "SYNTROPY_QWEN3_0_6B_DIR";

fn qwen3_config() -> HfConfig {
    HfConfig::parse(QWEN3_CONFIG, None).unwrap()
}

/// Qwen3-0.6B's safetensors tensors, derived from its configuration: (name, shape outermost
/// first), in the sorted order a safetensors header lists them. Every tensor is BF16.
fn qwen3_derived_tensors() -> Vec<(String, Vec<u64>)> {
    let (hidden, vocab, q, kv, d_head, ffn) = (1024, 151_936, 16 * 128, 8 * 128, 128, 3072);
    let mut t: Vec<(String, Vec<u64>)> = vec![
        ("lm_head.weight".into(), vec![vocab, hidden]),
        ("model.embed_tokens.weight".into(), vec![vocab, hidden]),
        ("model.norm.weight".into(), vec![hidden]),
    ];
    for l in 0..28 {
        for (n, shape) in [
            ("input_layernorm.weight", vec![hidden]),
            ("mlp.down_proj.weight", vec![hidden, ffn]),
            ("mlp.gate_proj.weight", vec![ffn, hidden]),
            ("mlp.up_proj.weight", vec![ffn, hidden]),
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
    }
    t.sort();
    t
}

/// A safetensors file holding only `tensors`' header: reads past the header fail, so a census
/// that reads it reads no weight.
struct HeaderOnly {
    header: Vec<u8>,
    len: u64,
}

impl RangeSource for HeaderOnly {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        let start = usize::try_from(offset).unwrap();
        let end = start + buf.len();
        assert!(
            end <= self.header.len(),
            "read {start}..{end} past the header"
        );
        buf.copy_from_slice(&self.header[start..end]);
        Ok(())
    }
}

/// The 8-byte length and the JSON header of a BF16 safetensors file of `tensors`.
fn safetensors_header(tensors: &[(String, Vec<u64>)]) -> HeaderOnly {
    let mut json = String::from(r#"{"__metadata__":{"format":"pt"}"#);
    let mut at = 0u64;
    for (name, shape) in tensors {
        let bytes = shape.iter().product::<u64>() * 2;
        let dims: Vec<String> = shape.iter().map(u64::to_string).collect();
        write!(
            json,
            r#","{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{at},{}]}}"#,
            dims.join(","),
            at + bytes
        )
        .unwrap();
        at += bytes;
    }
    json.push('}');
    let mut header = (json.len() as u64).to_le_bytes().to_vec();
    header.extend_from_slice(json.as_bytes());
    let len = header.len() as u64 + at;
    HeaderOnly { header, len }
}

/// The derived header, read back through the safetensors reader.
fn qwen3_tensors() -> Vec<(String, Vec<u64>)> {
    let source = safetensors_header(&qwen3_derived_tensors());
    read_header_default(&source)
        .unwrap()
        .tensors
        .into_iter()
        .map(|t| (t.name, t.shape))
        .collect()
}

fn plan(hp: &HyperParams, tensors: &[(String, Vec<u64>)]) -> TensorPlan {
    classify_tensors(
        &LLAMA,
        hp,
        tensors.iter().map(|(n, s)| (n.as_str(), s.as_slice())),
    )
}

/// A small `config.json` of architecture `arch`, with `fields` added or replacing the base's.
fn config_with(arch: &str, model_type: &str, fields: &str) -> HfConfig {
    let base = [
        ("num_hidden_layers", "2"),
        ("hidden_size", "64"),
        ("vocab_size", "256"),
        ("num_attention_heads", "4"),
        ("num_key_value_heads", "2"),
        ("intermediate_size", "96"),
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
    LLAMA.parse_hparams(&MetaView::Config(c))
}

fn code3(c: &HfConfig, key: &str) {
    let e = parse(c).unwrap_err();
    assert_eq!(e.code(), 3, "{e}");
    assert!(e.to_string().contains(key), "{e} does not name {key}");
}

// ---- A GGUF writer ------------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Kv {
    U32(u32),
    F32(f32),
    Str(&'static str),
}

/// A GGUF v3 header of `kvs` and F32 `tensors` (ggml's `ne`), with no data section.
fn gguf(kvs: &[(&str, Kv)], tensors: &[(&str, &[u64])]) -> Vec<u8> {
    fn s(out: &mut Vec<u8>, v: &str) {
        out.extend_from_slice(&(v.len() as u64).to_le_bytes());
        out.extend_from_slice(v.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs {
        s(&mut out, k);
        match v {
            Kv::U32(x) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
            }
            Kv::F32(x) => {
                out.extend_from_slice(&6u32.to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
            }
            Kv::Str(x) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                s(&mut out, x);
            }
        }
    }
    let mut at = 0u64;
    for (name, ne) in tensors {
        s(&mut out, name);
        out.extend_from_slice(&(ne.len() as u32).to_le_bytes());
        for d in *ne {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes()); // F32
        out.extend_from_slice(&at.to_le_bytes());
        at += (ne.iter().product::<u64>() * 4).div_ceil(32) * 32;
    }
    let data_start = (out.len() as u64).div_ceil(32) * 32;
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

fn header(kvs: &[(&str, Kv)], tensors: &[(&str, &[u64])]) -> GgufHeader {
    GgufHeader::read(Bytes(gguf(kvs, tensors)), &Limits::default()).unwrap()
}

/// Qwen3-0.6B's metadata as a GGUF converter writes it.
fn qwen3_gguf_kvs() -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("qwen3")),
        ("qwen3.block_count", Kv::U32(28)),
        ("qwen3.context_length", Kv::U32(40960)),
        ("qwen3.embedding_length", Kv::U32(1024)),
        ("qwen3.feed_forward_length", Kv::U32(3072)),
        ("qwen3.attention.head_count", Kv::U32(16)),
        ("qwen3.attention.head_count_kv", Kv::U32(8)),
        ("qwen3.rope.freq_base", Kv::F32(1_000_000.0)),
        ("qwen3.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
        ("qwen3.attention.key_length", Kv::U32(128)),
        ("qwen3.attention.value_length", Kv::U32(128)),
        ("qwen3.vocab_size", Kv::U32(151_936)),
    ]
}

fn with(kvs: &[(&'static str, Kv)], extra: &[(&'static str, Kv)]) -> Vec<(&'static str, Kv)> {
    let mut v: Vec<_> = kvs
        .iter()
        .filter(|(k, _)| !extra.iter().any(|(e, _)| e == k))
        .copied()
        .collect();
    v.extend_from_slice(extra);
    v
}

// ---- Golden census --------------------------------------------------------------------------------

/// The census as the family gives it, one line per tensor, for the golden file.
fn render(hp: &HyperParams, plan: &TensorPlan, kv: &KvFormula) -> String {
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
    writeln!(out, "# fetch ({})", plan.fetch.len()).unwrap();
    let mut fetch = plan.fetch.clone();
    fetch.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, c) in &fetch {
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
        writeln!(
            out,
            "{name} class={class:?} layer={layer:?} role={role:?} block={block:?} stacked={stacked_experts:?}"
        )
        .unwrap();
    }
    writeln!(out, "# ignored ({})", plan.ignored.len()).unwrap();
    for (name, reason) in &plan.ignored {
        writeln!(out, "{name}: {reason}").unwrap();
    }
    writeln!(out, "# unknown ({})", plan.unknown.len()).unwrap();
    for name in &plan.unknown {
        writeln!(out, "{name}").unwrap();
    }
    out
}

/// The census builder's manifest (COMP-E10-T07), without the parts that depend on the file's
/// bytes or digest (`ModelId`, source extents): records, groups, ignored tensors and totals.
fn render_manifest(m: &ModelManifest) -> String {
    let mut out = String::new();
    writeln!(out, "# manifest").unwrap();
    writeln!(out, "family={} format={:?}", m.family, m.format).unwrap();
    let name = |id: TensorId| {
        m.census
            .tensors
            .iter()
            .find(|t| t.id == id)
            .map_or("?", |t| t.name.as_str())
    };
    let group_key = |id: TensorGroupId| {
        m.census
            .groups
            .iter()
            .find(|g| g.id == id)
            .map(|g| format!("{:?}@{:?}", g.class, g.layer))
    };
    writeln!(out, "# records ({})", m.census.tensors.len()).unwrap();
    let mut records: Vec<_> = m.census.tensors.iter().collect();
    records.sort_by(|a, b| a.name.cmp(&b.name));
    for r in records {
        writeln!(
            out,
            "{} class={:?} layer={:?} shape={:?} quant={:?} bytes={:?} aux={}",
            r.name,
            r.class,
            r.layer,
            r.shape,
            r.quant,
            r.bytes,
            r.aux.len()
        )
        .unwrap();
    }
    writeln!(out, "# groups ({})", m.census.groups.len()).unwrap();
    for g in &m.census.groups {
        let mut members: Vec<&str> = g.tensors.iter().map(|&t| name(t)).collect();
        members.sort_unstable();
        writeln!(
            out,
            "{:?}@{:?} bytes={:?} read_per_token={:?} shared_with={:?} tensors={members:?}",
            g.class,
            g.layer,
            g.bytes,
            g.read_per_token,
            g.shared_with.and_then(group_key)
        )
        .unwrap();
    }
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

/// Whether this run rewrites the golden file: only when `SYNTROPY_UPDATE_SNAPSHOTS` is set,
/// and never under CI.
fn update_golden() -> bool {
    match (
        std::env::var_os("SYNTROPY_UPDATE_SNAPSHOTS"),
        std::env::var_os("CI"),
    ) {
        (Some(_), Some(_)) => {
            panic!("SYNTROPY_UPDATE_SNAPSHOTS is set under CI; goldens are only rewritten locally")
        }
        (update, _) => update.is_some(),
    }
}

/// Resolves, parses and classifies a Qwen3-0.6B checkpoint from `config` and its safetensors
/// `header`, builds its manifest with the census builder (COMP-E10-T07), and checks both
/// against the golden file. `rewrite` lets this run regenerate it.
fn check_qwen3_census(config: &HfConfig, header: &SafetensorsHeader, rewrite: bool) {
    let tensors: Vec<(String, Vec<u64>)> = header
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.shape.clone()))
        .collect();
    let tensors = &tensors[..];
    let meta = MetaView::Config(config);
    let family = families().resolve(&meta).unwrap();
    assert_eq!(family.name(), "llama");
    let hp = family.parse_hparams(&meta).unwrap();
    assert_eq!(tensors.len(), 311, "Qwen3-0.6B's tensor count");
    let plan = plan(&hp, tensors);
    let kv = family.kv_formula(
        &hp,
        KvSource::Safetensors {
            compute: Some(DType::BF16),
            quantized: false,
        },
    );
    let files = [SourceFile {
        role: SourceRole::Weights,
        digest: FileDigest([0; 32]),
        len: header.file_len,
    }];
    let manifest = census::build(&Sources {
        meta,
        weights: Weights::Safetensors(std::slice::from_ref(header)),
        kv_source: KvSource::Safetensors {
            compute: Some(DType::BF16),
            quantized: false,
        },
        tokenizer: None,
        files: &files,
    })
    .unwrap();
    assert_eq!(manifest.family, "llama");
    assert_eq!(
        manifest.hparams, hp,
        "the builder's hyperparameters are the family's"
    );
    assert_eq!(manifest.kv, kv, "the builder's KV formula is the family's");
    let actual = render(&hp, &plan, &kv) + &render_manifest(&manifest);
    if rewrite && update_golden() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/llama/golden/qwen3-0.6b.txt");
        std::fs::write(path, &actual).unwrap();
        return;
    }
    assert_eq!(
        actual,
        QWEN3_GOLDEN.replace("\r\n", "\n"),
        "the census differs from the golden file; the actual census is:\n{actual}"
    );

    // The facts the golden file holds, stated on their own so a regenerated file cannot
    // quietly change them.
    assert!(plan.unknown.is_empty(), "{:?}", plan.unknown);
    assert_eq!(plan.ignored, [("lm_head.weight".to_owned(), TIED_REASON)]);
    assert_eq!(plan.fetch.len(), 2 + 28 * 11);
    assert_eq!(hp.n_layers, 28);
    assert_eq!((hp.hidden, hp.vocab), (1024, 151_936));
    assert_eq!(hp.count(N_HEADS), Some(16));
    assert_eq!(hp.count(N_KV_HEADS), Some(8));
    assert_eq!(
        hp.count(D_HEAD),
        Some(128),
        "Qwen3 states head_dim apart from hidden"
    );
    assert_eq!(hp.count(D_FFN), Some(3072));
    assert!(flag(&hp, QK_NORM) && !flag(&hp, QKV_BIAS) && flag(&hp, TIED_EMBEDDINGS));
    assert_eq!(kv.layers, vec![LayerKv::gqa(8, 128, None); 28]);
    // RFC-0060 §7.2 check 1's facts, as the builder gives them: every tensor recorded or
    // ignored, the head and embedding as `Head` and `RowLookup`, and the tied head sharing
    // the embedding's record.
    assert_eq!(manifest.census.tensors.len(), 310);
    assert_eq!(manifest.ignored.len(), 1);
    let head = manifest
        .census
        .groups
        .iter()
        .find(|g| g.class == TensorClass::Head)
        .unwrap();
    let lookup = manifest
        .census
        .groups
        .iter()
        .find(|g| g.class == TensorClass::RowLookup)
        .unwrap();
    assert_eq!(
        head.shared_with,
        Some(lookup.id),
        "tied: the head reads the token table"
    );
    assert_eq!(lookup.shared_with, Some(head.id));
    assert_eq!(manifest.census.groups.len(), 1 + 28 * 2 + 1);
    let embed = plan
        .fetch
        .iter()
        .find(|(n, _)| n == "model.embed_tokens.weight")
        .unwrap();
    assert!(matches!(
        embed.1,
        Classification::Classified {
            class: TensorClass::RowLookup,
            ..
        }
    ));
}

/// COMP-E16-T03's first criterion, on headers derived from Qwen3-0.6B's configuration: the
/// checkpoint resolves to `llama` through this build's registry from its metadata alone, every
/// tensor is classified, the stored head is ignored because the embeddings are tied, and the
/// whole result matches the golden file.
#[test]
fn golden_census_on_qwen3_0_6b_derived_headers() {
    let header = read_header_default(safetensors_header(&qwen3_derived_tensors())).unwrap();
    check_qwen3_census(&qwen3_config(), &header, true);
}

/// COMP-E16-T03's first criterion on the recorded files themselves (QUAL-E7-T06), opt-in:
/// `SYNTROPY_QWEN3_0_6B_DIR` names a directory holding Qwen3-0.6B's `config.json`,
/// `generation_config.json` and `model.safetensors` at revision
/// `c1899de289a04d12100db370d81485cdf75e47ca`, verified against `testdata/fixtures.toml`'s
/// BLAKE3 with `cargo xtask fixtures`. Only the header of `model.safetensors` is read. The
/// census must match the same golden file as the derived headers'.
#[test]
fn golden_census_on_the_recorded_qwen3_0_6b_fixtures() {
    let Some(dir) = std::env::var_os(QWEN3_DIR_VAR) else {
        eprintln!(
            "NOT RUN: {QWEN3_DIR_VAR} is not set; the census on the recorded Qwen3-0.6B fixtures \
             needs the real files (see this module's documentation)"
        );
        return;
    };
    let dir = Path::new(&dir);
    let read = |f: &str| std::fs::read(dir.join(f)).unwrap();
    let config =
        HfConfig::parse(&read("config.json"), Some(&read("generation_config.json"))).unwrap();
    let source = FileSource::open(dir.join("model.safetensors")).unwrap();
    assert_eq!(source.len(), 1_503_300_328, "fixtures.toml's byte count");
    let header = read_header_default(&source).unwrap();
    let end = header
        .tensors
        .iter()
        .map(|t| t.offset + t.len)
        .max()
        .unwrap();
    assert_eq!(end, source.len(), "the header covers the whole file");
    check_qwen3_census(&config, &header, false);
    eprintln!("ran on {}", dir.display());
}

// ---- KV formula and dtype -----------------------------------------------------------------------

/// COMP-E16-T03's second criterion: `KvFormula::dtype` is F16 for a GGUF source and BF16 for a
/// BF16 safetensors source, by COMP-E13-T06's rule, with the same per-layer entries.
#[test]
fn kv_dtype_is_f16_for_gguf_and_bf16_for_bf16_safetensors() {
    let hp = parse(&qwen3_config()).unwrap();
    let gguf = LLAMA.kv_formula(&hp, KvSource::Gguf);
    let st = LLAMA.kv_formula(
        &hp,
        KvSource::Safetensors {
            compute: Some(DType::BF16),
            quantized: false,
        },
    );
    for l in [0, 13, 27] {
        assert_eq!(gguf.dtype.of(LayerIx(l)), DType::F16);
        assert_eq!(st.dtype.of(LayerIx(l)), DType::BF16);
    }
    assert!(gguf.dtype.layers.is_empty() && st.dtype.layers.is_empty());
    assert_eq!(gguf.layers, st.layers);
    assert_eq!(
        gguf.layers[0],
        LayerKv::PerToken {
            elems_per_token: 2 * 8 * 128,
            window: None,
            latent: false,
        }
    );
    assert!(gguf.aux.is_empty());
    // The formula is the rule's, step 3, for every source the rule knows.
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
            LLAMA.kv_formula(&hp, source).dtype,
            rule.reference().unwrap()
        );
    }
}

/// An instance `kv_dtype` that would store KV below the reference's dtype is refused, and one
/// that widens it is applied: the family states no requirement that blocks either.
#[test]
fn a_kv_dtype_below_the_reference_is_refused() {
    let source = KvSource::Safetensors {
        compute: Some(DType::BF16),
        quantized: false,
    };
    let rule = KvDTypeRule {
        source,
        declared_cache: None,
        family: None,
        family_layers: Vec::new(),
    };
    let hp = parse(&qwen3_config()).unwrap();
    assert_eq!(
        LLAMA.kv_formula(&hp, source).dtype.default,
        rule.reference().unwrap().default
    );
    let e = rule.resolve(Some(KvDTypeOverride::F16)).unwrap_err();
    assert_eq!(e.name(), "kv_dtype_narrows");
    assert_eq!(
        rule.resolve(Some(KvDTypeOverride::F32)).unwrap().default,
        DType::F32
    );
}

// ---- Both sources ---------------------------------------------------------------------------------

/// The same model from GGUF and from safetensors gives equal hyperparameters (RFC-0043 §5.2),
/// floats included, except tied embeddings, which GGUF has no key for.
#[test]
fn gguf_and_config_give_equal_hyperparameters() {
    let from_config = parse(&qwen3_config()).unwrap();
    let h = header(&qwen3_gguf_kvs(), &[]);
    let mut from_gguf = LLAMA.parse_hparams(&MetaView::Gguf(&h.metadata)).unwrap();
    assert_eq!(
        from_gguf.extra.get(TIED_EMBEDDINGS),
        Some(&HParamValue::Bool(false))
    );
    from_gguf = from_gguf.with(TIED_EMBEDDINGS, HParamValue::Bool(true));
    assert_eq!(from_gguf, from_config);
}

/// GGUF names of every Qwen3 tensor classify, with the same classes as the safetensors names;
/// a GGUF with no vocabulary key takes it from the token list's length.
#[test]
fn gguf_names_classify_like_safetensors_names() {
    let h = header(&qwen3_gguf_kvs(), &[]);
    let hp = LLAMA.parse_hparams(&MetaView::Gguf(&h.metadata)).unwrap();
    let per_layer: &[(&str, &[u64])] = &[
        ("attn_norm.weight", &[1024]),
        ("attn_q.weight", &[1024, 2048]),
        ("attn_k.weight", &[1024, 1024]),
        ("attn_v.weight", &[1024, 1024]),
        ("attn_output.weight", &[2048, 1024]),
        ("attn_q_norm.weight", &[128]),
        ("attn_k_norm.weight", &[128]),
        ("ffn_norm.weight", &[1024]),
        ("ffn_gate.weight", &[1024, 3072]),
        ("ffn_up.weight", &[1024, 3072]),
        ("ffn_down.weight", &[3072, 1024]),
    ];
    let mut tensors: Vec<(String, Vec<u64>)> = vec![
        ("token_embd.weight".into(), vec![1024, 151_936]),
        ("output_norm.weight".into(), vec![1024]),
        ("output.weight".into(), vec![1024, 151_936]),
    ];
    for l in [0, 27] {
        for (n, s) in per_layer {
            tensors.push((format!("blk.{l}.{n}"), s.to_vec()));
        }
    }
    let p = plan(&hp, &tensors);
    assert!(p.unknown.is_empty(), "{:?}", p.unknown);
    assert!(p.ignored.is_empty());
    assert_eq!(p.fetch.len(), tensors.len());
    let class = |name: &str, p: &TensorPlan| {
        p.fetch
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| *c)
            .unwrap()
    };
    let st = plan(&hp, &qwen3_tensors()[..]);
    for (g, s) in [
        ("token_embd.weight", "model.embed_tokens.weight"),
        ("output_norm.weight", "model.norm.weight"),
        (
            "blk.27.attn_q.weight",
            "model.layers.27.self_attn.q_proj.weight",
        ),
        (
            "blk.0.attn_k_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        ),
        (
            "blk.0.ffn_down.weight",
            "model.layers.0.mlp.down_proj.weight",
        ),
        (
            "blk.0.ffn_norm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ),
    ] {
        assert_eq!(class(g, &p), class(s, &st), "{g} and {s}");
    }
    assert!(matches!(
        class("output.weight", &p),
        Classification::Classified {
            class: TensorClass::Head,
            ..
        }
    ));

    // No vocabulary key: the token list gives it.
    let kvs: Vec<_> = qwen3_gguf_kvs()
        .into_iter()
        .filter(|(k, _)| *k != "qwen3.vocab_size")
        .collect();
    let h = header(&kvs, &[]);
    let e = LLAMA
        .parse_hparams(&MetaView::Gguf(&h.metadata))
        .unwrap_err();
    assert_eq!(e.code(), 3);
    assert!(e.to_string().contains("qwen3.vocab_size"), "{e}");
}

/// A GGUF that stores Llama-3-style scaling as a `rope_freqs.weight` table is refused: the
/// table has no rule, so the census refuses the load rather than drop the scaling.
#[test]
fn a_gguf_rope_frequency_table_is_unknown() {
    let kvs = with(
        &qwen3_gguf_kvs(),
        &[("general.architecture", Kv::Str("llama"))],
    );
    let kvs: Vec<_> = kvs
        .into_iter()
        .map(|(k, v)| (k.strip_prefix("qwen3").map_or(k, |_| leak(k)), v))
        .collect();
    let h = header(&kvs, &[]);
    let hp = LLAMA.parse_hparams(&MetaView::Gguf(&h.metadata)).unwrap();
    assert!(!flag(&hp, QK_NORM));
    assert_eq!(
        LLAMA.classify("rope_freqs.weight", &[64], &hp),
        Classification::Unknown
    );
}

/// `qwen3.x` as `llama.x`, for a test that rewrites the architecture.
fn leak(k: &'static str) -> &'static str {
    Box::leak(format!("llama{}", &k["qwen3".len()..]).into_boxed_str())
}

// ---- Variants -----------------------------------------------------------------------------------

/// The architecture picks the variant: QK-norm for Qwen3, QKV bias for Qwen2, neither for
/// Llama and Mistral. A tensor the variant does not have is `Unknown`.
#[test]
fn variants_set_qk_norm_and_qkv_bias_and_gate_their_tensors() {
    for (arch, model_type, qk_norm, bias) in [
        ("LlamaForCausalLM", "llama", false, false),
        ("MistralForCausalLM", "mistral", false, false),
        ("Qwen2ForCausalLM", "qwen2", false, true),
        ("Qwen3ForCausalLM", "qwen3", true, false),
    ] {
        let hp = parse(&config_with(arch, model_type, "")).unwrap();
        assert_eq!(flag(&hp, QK_NORM), qk_norm, "{arch}");
        assert_eq!(flag(&hp, QKV_BIAS), bias, "{arch}");
        let q_norm = LLAMA.classify("model.layers.0.self_attn.q_norm.weight", &[16], &hp);
        assert_eq!(q_norm == Classification::Unknown, !qk_norm, "{arch}");
        let q_bias = LLAMA.classify("model.layers.0.self_attn.q_proj.bias", &[64], &hp);
        assert_eq!(q_bias == Classification::Unknown, !bias, "{arch}");
        let k_bias = LLAMA.classify("blk.1.attn_k.bias", &[32], &hp);
        assert_eq!(k_bias == Classification::Unknown, !bias, "{arch}");
        let g = gqa_params(&hp).unwrap();
        assert_eq!(g.qkv, QkvLayout::Separate { qk_norm }, "{arch}");
        assert_eq!(g.bias, bias, "{arch}");
    }
}

/// Untied, the head is classified `Head`; tied, a stored head is ignored and never fetched.
#[test]
fn tied_embeddings_ignore_a_stored_head() {
    let untied = parse(&config_with("LlamaForCausalLM", "llama", "")).unwrap();
    let tied = parse(&config_with(
        "LlamaForCausalLM",
        "llama",
        r#""tie_word_embeddings": true"#,
    ))
    .unwrap();
    for (name, shape) in [
        ("lm_head.weight", [256u64, 64]),
        ("output.weight", [64, 256]),
    ] {
        assert!(matches!(
            LLAMA.classify(name, &shape, &untied),
            Classification::Classified {
                class: TensorClass::Head,
                role: TensorRole::Head,
                block: BlockSel::Epilogue,
                ..
            }
        ));
        assert_eq!(
            LLAMA.classify(name, &shape, &tied),
            Classification::Ignored {
                reason: TIED_REASON
            }
        );
        // The shape is checked either way.
        assert_eq!(
            LLAMA.classify(name, &[shape[0], shape[1] + 1], &tied),
            Classification::Unknown
        );
    }
}

// ---- RoPE scaling -------------------------------------------------------------------------------

#[test]
fn rope_scaling_rules_parse_into_block_parameters() {
    // No scaling, the default base.
    let hp = parse(&config_with("LlamaForCausalLM", "llama", "")).unwrap();
    let r = rope_params(&hp).unwrap();
    assert_eq!(r.scaling, RopeScaling::Plain);
    assert_eq!((r.theta, r.dims), (10_000.0, 16));

    // Llama-3-style, as Llama-3.2-1B's configuration states it.
    let hp = parse(&config_with(
        "LlamaForCausalLM",
        "llama",
        r#""rope_theta": 500000.0, "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0,
           "low_freq_factor": 1.0, "original_max_position_embeddings": 8192,
           "rope_type": "llama3"}"#,
    ))
    .unwrap();
    assert_eq!(
        rope_params(&hp).unwrap().scaling,
        RopeScaling::Llama3 {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_ctx: 8192,
        }
    );
    assert_eq!(rope_params(&hp).unwrap().theta, 500_000.0);

    // YaRN with the defaults, and the original context from `max_position_embeddings`.
    let hp = parse(&config_with(
        "Qwen2ForCausalLM",
        "qwen2",
        r#""max_position_embeddings": 12288, "rope_scaling": {"type": "yarn", "factor": 4.0}"#,
    ))
    .unwrap();
    assert_eq!(
        rope_params(&hp).unwrap().scaling,
        RopeScaling::Yarn {
            factor: 4.0,
            original_ctx: 12_288,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: None,
        }
    );
    let hp = parse(&config_with(
        "Qwen3ForCausalLM",
        "qwen3",
        r#""rope_scaling": {"rope_type": "yarn", "factor": 4.0,
           "original_max_position_embeddings": 32768, "beta_fast": 16, "beta_slow": 2,
           "attention_factor": 1.5}"#,
    ))
    .unwrap();
    assert_eq!(
        rope_params(&hp).unwrap().scaling,
        RopeScaling::Yarn {
            factor: 4.0,
            original_ctx: 32_768,
            beta_fast: 16.0,
            beta_slow: 2.0,
            attn_factor: Some(1.5),
        }
    );

    // Linear, and `default`, which is no scaling.
    let hp = parse(&config_with(
        "LlamaForCausalLM",
        "llama",
        r#""rope_scaling": {"rope_type": "linear", "factor": 2.0}"#,
    ))
    .unwrap();
    assert_eq!(
        rope_params(&hp).unwrap().scaling,
        RopeScaling::Linear { factor: 2.0 }
    );
    let hp = parse(&config_with(
        "LlamaForCausalLM",
        "llama",
        r#""rope_scaling": {"rope_type": "default"}"#,
    ))
    .unwrap();
    assert_eq!(rope_params(&hp).unwrap().scaling, RopeScaling::Plain);

    // GGUF: linear by type and factor, and by the older key.
    let h = header(
        &with(
            &qwen3_gguf_kvs(),
            &[
                ("qwen3.rope.scaling.type", Kv::Str("linear")),
                ("qwen3.rope.scaling.factor", Kv::F32(4.0)),
            ],
        ),
        &[],
    );
    let hp = LLAMA.parse_hparams(&MetaView::Gguf(&h.metadata)).unwrap();
    assert_eq!(
        rope_params(&hp).unwrap().scaling,
        RopeScaling::Linear { factor: 4.0 }
    );
    let h = header(
        &with(
            &qwen3_gguf_kvs(),
            &[("qwen3.rope.scale_linear", Kv::F32(2.0))],
        ),
        &[],
    );
    let hp = LLAMA.parse_hparams(&MetaView::Gguf(&h.metadata)).unwrap();
    assert_eq!(
        rope_params(&hp).unwrap().scaling,
        RopeScaling::Linear { factor: 2.0 }
    );
}

#[test]
fn rope_rules_it_does_not_compute_are_code_3() {
    let rs = |body: &str| {
        config_with(
            "LlamaForCausalLM",
            "llama",
            &format!(r#""rope_scaling": {{{body}}}"#),
        )
    };
    code3(&rs(r#""rope_type": "dynamic", "factor": 2.0"#), "rope_type");
    code3(
        &rs(r#""rope_type": "longrope", "factor": 2.0"#),
        "rope_type",
    );
    code3(&rs(r#""factor": 2.0"#), "rope_type");
    code3(&rs(r#""rope_type": "linear""#), "rope_scaling.factor");
    code3(
        &rs(r#""rope_type": "linear", "factor": 0.5"#),
        "rope_scaling.factor",
    );
    code3(
        &rs(
            r#""rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
              "original_max_position_embeddings": 8192"#,
        ),
        "high_freq_factor",
    );
    code3(
        &rs(
            r#""rope_type": "llama3", "factor": 8.0, "low_freq_factor": 4.0,
              "high_freq_factor": 1.0, "original_max_position_embeddings": 8192"#,
        ),
        "high_freq_factor",
    );
    code3(
        &rs(
            r#""rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
              "high_freq_factor": 4.0"#,
        ),
        "original_max_position_embeddings",
    );
    code3(
        &rs(r#""rope_type": "yarn", "factor": 40.0, "mscale": 1.0"#),
        "rope_scaling.mscale",
    );
    code3(
        &rs(r#""rope_type": "yarn", "factor": 40.0, "mscale_all_dim": 1.0"#),
        "mscale_all_dim",
    );
    code3(
        &rs(r#""rope_type": "yarn", "factor": 4.0, "truncate": false"#),
        "truncate",
    );
    code3(
        &config_with(
            "LlamaForCausalLM",
            "llama",
            r#""rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}"#,
        ),
        "rope_parameters",
    );
    let h = header(
        &with(
            &qwen3_gguf_kvs(),
            &[
                ("qwen3.rope.scaling.type", Kv::Str("yarn")),
                ("qwen3.rope.scaling.factor", Kv::F32(4.0)),
            ],
        ),
        &[],
    );
    let e = LLAMA
        .parse_hparams(&MetaView::Gguf(&h.metadata))
        .unwrap_err();
    assert_eq!(e.code(), 3);
    assert!(e.to_string().contains("qwen3.rope.scaling.type"), "{e}");
}

// ---- Fail closed --------------------------------------------------------------------------------

#[test]
fn unknown_architectures_are_code_1() {
    for (arch, model_type) in [
        ("Qwen3MoeForCausalLM", "qwen3_moe"),
        ("GemmaForCausalLM", "gemma"),
        ("LLaMAForCausalLM", "Llama"),
    ] {
        let c = config_with(arch, model_type, "");
        let e = families().resolve(&MetaView::Config(&c)).err().unwrap();
        assert_eq!(e.code(), 1, "{arch}");
        assert!(e.to_string().contains("supported families: llama"), "{e}");
        let e = parse(&c).unwrap_err();
        assert_eq!(e.code(), 1, "{arch}");
    }
    let h = header(
        &with(
            &qwen3_gguf_kvs(),
            &[("general.architecture", Kv::Str("qwen3moe"))],
        ),
        &[],
    );
    let e = families()
        .resolve(&MetaView::Gguf(&h.metadata))
        .err()
        .unwrap();
    assert_eq!(e.code(), 1);
}

#[test]
fn missing_or_out_of_range_hyperparameters_are_code_3() {
    // Each required key of the recorded configuration, renamed away in turn.
    for key in [
        "num_hidden_layers",
        "hidden_size",
        "vocab_size",
        "num_attention_heads",
        "intermediate_size",
        "rms_norm_eps",
        "max_position_embeddings",
    ] {
        let json = String::from_utf8(QWEN3_CONFIG.to_vec()).unwrap();
        let without = json.replace(&format!("\"{key}\""), &format!("\"renamed_{key}\""));
        assert_ne!(without, json);
        let c = HfConfig::parse(without.as_bytes(), None).unwrap();
        code3(&c, key);
    }
    code3(
        &config_with("LlamaForCausalLM", "llama", r#""num_key_value_heads": 3"#),
        "num_key_value_heads",
    );
    code3(
        &config_with("LlamaForCausalLM", "llama", r#""num_key_value_heads": 8"#),
        "num_key_value_heads",
    );
    code3(
        &config_with("LlamaForCausalLM", "llama", r#""num_hidden_layers": 0"#),
        "num_hidden_layers",
    );
    code3(
        &config_with("LlamaForCausalLM", "llama", r#""rms_norm_eps": 0"#),
        "rms_norm_eps",
    );
    code3(
        &config_with("LlamaForCausalLM", "llama", r#""hidden_size": "64""#),
        "hidden_size",
    );
    // 64 does not split into 3 heads, and no head dim is given; with one, it parses.
    code3(
        &config_with(
            "LlamaForCausalLM",
            "llama",
            r#""num_attention_heads": 3, "num_key_value_heads": 1"#,
        ),
        "num_attention_heads",
    );
    let hp = parse(&config_with(
        "LlamaForCausalLM",
        "llama",
        r#""num_attention_heads": 3, "num_key_value_heads": 1, "head_dim": 32"#,
    ))
    .unwrap();
    assert_eq!(hp.count(D_HEAD), Some(32));
    let h = header(
        &with(
            &qwen3_gguf_kvs(),
            &[("qwen3.attention.value_length", Kv::U32(64))],
        ),
        &[],
    );
    let e = LLAMA
        .parse_hparams(&MetaView::Gguf(&h.metadata))
        .unwrap_err();
    assert!(e.to_string().contains("value_length"), "{e}");
}

/// Settings the blocks cannot compute are refused, never ignored.
#[test]
fn settings_the_blocks_cannot_compute_are_code_3() {
    let llama = |f| config_with("LlamaForCausalLM", "llama", f);
    code3(&llama(r#""sliding_window": 4096"#), "sliding_window");
    code3(
        &config_with("MistralForCausalLM", "mistral", r#""sliding_window": 4096"#),
        "sliding_window",
    );
    code3(&llama(r#""attention_bias": true"#), "attention_bias");
    code3(
        &config_with("Qwen3ForCausalLM", "qwen3", r#""attention_bias": true"#),
        "attention_bias",
    );
    code3(&llama(r#""mlp_bias": true"#), "mlp_bias");
    code3(&llama(r#""hidden_act": "gelu""#), "hidden_act");
    code3(
        &llama(r#""partial_rotary_factor": 0.5"#),
        "partial_rotary_factor",
    );
    code3(
        &config_with(
            "Qwen2ForCausalLM",
            "qwen2",
            r#""use_sliding_window": true, "sliding_window": 4096"#,
        ),
        "sliding_window",
    );
    // Qwen's configurations carry a window that is off, and Qwen2's own bias flag is moot.
    parse(&config_with(
        "Qwen2ForCausalLM",
        "qwen2",
        r#""use_sliding_window": false, "sliding_window": 32768, "attention_bias": true"#,
    ))
    .unwrap();
    parse(&llama(
        r#""sliding_window": null, "attention_bias": false, "mlp_bias": false, "hidden_act": "silu""#,
    ))
    .unwrap();
    let h = header(
        &with(
            &qwen3_gguf_kvs(),
            &[("qwen3.rope.dimension_count", Kv::U32(64))],
        ),
        &[],
    );
    let e = LLAMA
        .parse_hparams(&MetaView::Gguf(&h.metadata))
        .unwrap_err();
    assert!(e.to_string().contains("qwen3.rope.dimension_count"), "{e}");
    let h = header(
        &with(
            &qwen3_gguf_kvs(),
            &[("qwen3.attention.sliding_window", Kv::U32(4096))],
        ),
        &[],
    );
    let e = LLAMA
        .parse_hparams(&MetaView::Gguf(&h.metadata))
        .unwrap_err();
    assert!(e.to_string().contains("sliding_window"), "{e}");
}

/// A name that matches with the wrong shape, a layer past the last, or a name no rule has is
/// `Unknown`, so the load is refused.
#[test]
fn shape_mismatches_and_strangers_are_unknown() {
    let hp = parse(&qwen3_config()).unwrap();
    let tensors = qwen3_tensors();
    for (name, shape) in &tensors {
        if name == "lm_head.weight" {
            continue;
        }
        for d in 0..shape.len() {
            let mut wrong = shape.clone();
            wrong[d] += 1;
            assert_eq!(
                LLAMA.classify(name, &wrong, &hp),
                Classification::Unknown,
                "{name} {wrong:?}"
            );
        }
        let mut transposed = shape.clone();
        transposed.reverse();
        if transposed != *shape {
            assert_eq!(
                LLAMA.classify(name, &transposed, &hp),
                Classification::Unknown,
                "{name} transposed"
            );
        }
        let mut extra = shape.clone();
        extra.push(1);
        assert_eq!(LLAMA.classify(name, &extra, &hp), Classification::Unknown);
    }
    for name in [
        "model.layers.28.self_attn.q_proj.weight",
        "model.layers.01.self_attn.q_proj.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "blk.28.attn_q.weight",
        "rope_freqs.weight",
    ] {
        assert_eq!(
            LLAMA.classify(name, &[2048, 1024], &hp),
            Classification::Unknown,
            "{name}"
        );
    }
    // A planted stranger in the recorded header is reported, not dropped.
    let mut planted = tensors.clone();
    planted.push((
        "model.layers.0.mlp.shared_expert.gate_proj.weight".into(),
        vec![3072, 1024],
    ));
    let p = plan(&hp, &planted);
    assert_eq!(
        p.unknown,
        ["model.layers.0.mlp.shared_expert.gate_proj.weight"]
    );
}

// ---- Registry, ops, spec ----------------------------------------------------------------------

#[test]
fn the_registry_binds_llama_to_its_frozen_tag() {
    families().check(FROZEN_TAGS).unwrap();
    assert_eq!(families().tags(), ["llama"]);
    assert_eq!(MODULE, "syntropy_graph::families::llama");
    let frozen = FROZEN_TAGS.iter().find(|t| t.tag == TAG).unwrap();
    assert_eq!(frozen.module, MODULE);
    for name in LLAMA.accepts() {
        assert!(
            ACCEPTED.iter().any(|(a, _)| a == name),
            "{name} has no variant"
        );
        let f = families().resolve_names(&[name]).unwrap();
        assert_eq!(f.name(), TAG);
    }
    assert_eq!(LLAMA.accepts().len(), ACCEPTED.len());
}

/// Every combination of the variant flags compiles to a matcher.
#[test]
fn every_rule_table_combination_compiles() {
    for i in 0..8 {
        Matcher::new(&tables(i & 1 != 0, i & 2 != 0, i & 4 != 0)).unwrap();
    }
}

/// `required_ops` is the union of its blocks' ops (RFC-0040 §11.1), in both forms.
#[test]
fn required_ops_are_the_blocks_ops() {
    let hp = parse(&qwen3_config()).unwrap();
    let ops = LLAMA.required_ops(&hp);
    let mut expected = gqa_params(&hp).unwrap().ops();
    expected.extend(ffn_params(&hp).ops());
    expected.extend(embed_params(&hp).ops());
    expected.extend(head_params(&hp).ops());
    assert_eq!(ops, expected);
    for op in [
        OpKind::RowGather,
        OpKind::RmsNorm,
        OpKind::Gemv,
        OpKind::Gemm,
        OpKind::KvWrite(AttnKind::Gqa),
        OpKind::AttnDecode(AttnKind::Gqa),
        OpKind::AttnPrefill(AttnKind::Gqa),
        OpKind::GateUpAct {
            act: GatedAct::Silu,
        },
        OpKind::Add,
        OpKind::Logits,
    ] {
        assert!(ops.contains(&op), "{op:?}");
    }
    assert!(!ops.contains(&OpKind::Scale), "no embedding scale");
}

/// Qwen3-0.6B's block parameters come from its hyperparameters.
#[test]
fn block_parameters_follow_the_hyperparameters() {
    let hp = parse(&qwen3_config()).unwrap();
    let g = gqa_params(&hp).unwrap();
    assert_eq!((g.heads, g.kv_heads, g.head_dim), (16, 8, 128));
    assert_eq!(g.qkv, QkvLayout::Separate { qk_norm: true });
    assert!(!g.bias);
    assert_eq!(g.rope.theta, 1_000_000.0);
    assert_eq!(g.rope.dims, 128);
    assert_eq!(g.eps, 1e-6_f32);
    assert_eq!((g.scale, g.softcap), (None, None));
    let f = ffn_params(&hp);
    assert_eq!((f.inter, f.act), (3072, GatedAct::Silu));
    assert_eq!(embed_params(&hp).scale, None);
    assert_eq!(head_params(&hp).softcap, None);
    // A RoPE rule it does not know gives no parameters.
    let bad = hp.with(ROPE_SCALING, HParamValue::Str("dynamic".into()));
    assert_eq!(gqa_params(&bad), None);
}

/// Each `FamilySpec` variant resolves to `llama` by its own architecture names, and parsing a
/// configuration of that variant gives the attention features the spec states.
#[test]
fn the_fixture_spec_covers_every_variant() {
    let spec = fixture_spec();
    assert_eq!(spec.family, TAG);
    let names: Vec<_> = spec.variants.iter().map(|v| v.name).collect();
    let mut unique = names.clone();
    unique.dedup();
    assert_eq!(names, unique);
    let mut ropes = Vec::new();
    for v in spec.variants {
        for n in [v.gguf_architecture, v.architecture, v.model_type] {
            assert_eq!(families().resolve_names(&[n]).unwrap().name(), TAG, "{n}");
        }
        assert_eq!(v.layers, [crate::spec::LayerKind::GqaDense]);
        assert_eq!((v.moe, v.mtp_modules), (None, 0));
        let hp = parse(&config_with(v.architecture, v.model_type, "")).unwrap();
        assert_eq!(flag(&hp, QK_NORM), v.attention.qk_norm, "{}", v.name);
        assert_eq!(flag(&hp, QKV_BIAS), v.attention.qkv_bias, "{}", v.name);
        ropes.push(v.attention.rope);
    }
    for rope in [
        RopeKind::Plain,
        RopeKind::Linear,
        RopeKind::Llama3,
        RopeKind::Yarn,
    ] {
        assert!(ropes.contains(&rope), "no variant uses {rope:?}");
    }
    assert!(spec.variants.iter().any(|v| v.tied_embeddings));
    assert!(spec.variants.iter().any(|v| !v.tied_embeddings));
    assert!(spec.variants.iter().any(|v| v.attention.qk_norm));
    assert!(spec.variants.iter().any(|v| v.attention.qkv_bias));
}

/// Classification reads no tensor byte: the source serves the header only and fails any read
/// past it, and the census runs from it alone.
#[test]
fn the_census_needs_no_weights() {
    let hp = parse(&qwen3_config()).unwrap();
    let source = safetensors_header(&qwen3_derived_tensors());
    let header = read_header_default(&source).unwrap();
    assert_eq!(header.data_start, source.header.len() as u64);
    let tensors: Vec<_> = header
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.shape.clone()))
        .collect();
    assert!(plan(&hp, &tensors).unknown.is_empty());
}

#[test]
fn registry_new_with_llama_resolves_from_gguf() {
    static ONE: [crate::families::Registration; 1] = [crate::families::Registration {
        module: MODULE,
        family: &LLAMA,
    }];
    let h = header(&qwen3_gguf_kvs(), &[]);
    let f = Registry::new(&ONE)
        .resolve(&MetaView::Gguf(&h.metadata))
        .unwrap();
    assert_eq!(f.name(), TAG);
}
