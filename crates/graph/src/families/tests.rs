// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::ops::Range;
use std::sync::OnceLock;

use syntropy_core::ErrorClass;
use syntropy_manifest::census::TowerIx;
use syntropy_manifest::config::HfConfig;
use syntropy_manifest::gguf::{GgufHeader, Limits};
use syntropy_manifest::hparams::{HParamValue, HyperParams};
use syntropy_manifest::kv::{KvDTypeRule, KvFormula, KvSource, LayerKv};
use syntropy_manifest::range::{HeaderReader, RangeError, RangeSource};
use syntropy_manifest::tokenizer::{TokenizerSources, TokenizerSpec};

use super::*;
use crate::classify::{
    BlockSel, ClassRule, ClassSel, DimOrder, MULTIMODAL_REASON, Matcher, RuleTable, ShapeDim,
    classify_tensors,
};
use crate::ir::{Block, TensorRole};

// ---- A test family ----------------------------------------------------------------------------

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

use ShapeDim::{Hp, Mul};

const GGUF_RULES: &[ClassRule] = &[
    rule(
        "token_embd.weight",
        BlockSel::Prologue,
        TensorRole::Embed,
        ClassSel::RowLookup,
        &[Hp("hidden"), Hp("vocab")],
    ),
    rule(
        "blk.{L}.attn_q.weight",
        BlockSel::Layer(Block::Gqa),
        TensorRole::Q,
        ClassSel::DenseEveryToken,
        &[Hp("hidden"), Mul("n_heads", "d_head")],
    ),
    rule(
        "output.weight",
        BlockSel::Epilogue,
        TensorRole::Head,
        ClassSel::Head,
        &[Hp("hidden"), Hp("vocab")],
    ),
    rule(
        "v.blk.{N}.attn_q.weight",
        BlockSel::Encoder,
        TensorRole::Q,
        ClassSel::Encoder { tower: TowerIx(0) },
        &[Hp("v_hidden"), Hp("v_hidden")],
    ),
];

/// A family with a few rules and five hyperparameters, standing in for the real modules.
struct Toy {
    tag: &'static str,
    accepts: &'static [&'static str],
    rules: OnceLock<Matcher>,
}

impl Toy {
    const fn new(tag: &'static str, accepts: &'static [&'static str]) -> Self {
        Self {
            tag,
            accepts,
            rules: OnceLock::new(),
        }
    }
}

const N_LAYERS: Key = Key::new("{arch}.block_count", "num_hidden_layers");
const HIDDEN: Key = Key::new("{arch}.embedding_length", "hidden_size");
const VOCAB: Key = Key::new("{arch}.vocab_size", "vocab_size");
const N_HEADS: Key = Key::new("{arch}.attention.head_count", "num_attention_heads");
const ROPE_THETA: Key = Key::new("{arch}.rope.freq_base", "rope_theta");
const V_HIDDEN: Key = Key::new(
    "{arch}.vision.embedding_length",
    "vision_config.hidden_size",
);

impl Family for Toy {
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
        let n_layers = meta.require_u64(N_LAYERS, 1..=u64::from(u16::MAX))?;
        let hidden = meta.require_u64(HIDDEN, 1..=1 << 20)?;
        let vocab = meta.require_u64(VOCAB, 1..=u64::from(u32::MAX))?;
        let n_heads = meta.require_u64(N_HEADS, 1..=hidden)?;
        if hidden % n_heads != 0 {
            return Err(FamilyError::invalid(
                meta.key_name(N_HEADS)?.unwrap_or_default(),
                format!("{n_heads} heads do not divide the hidden size {hidden}"),
            ));
        }
        let rope_theta = meta.require_f64(ROPE_THETA, 1.0..=1e12)?;
        let v_hidden = meta.u64_or(V_HIDDEN, 0, 0..=1 << 20)?;
        let narrow = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
        Ok(HyperParams::new(
            u16::try_from(n_layers).unwrap_or(u16::MAX),
            narrow(hidden),
            narrow(vocab),
        )
        .with("n_heads", HParamValue::U64(n_heads))
        .with("d_head", HParamValue::U64(hidden / n_heads))
        .with("rope_theta", HParamValue::F64(rope_theta))
        .with("v_hidden", HParamValue::U64(v_hidden)))
    }
    fn classify(&self, name: &str, shape: &[u64], hp: &HyperParams) -> Classification {
        self.rules
            .get_or_init(|| {
                Matcher::new(&[RuleTable {
                    order: DimOrder::InnermostFirst,
                    rules: GGUF_RULES,
                }])
                .expect("the test rules compile")
            })
            .classify(name, shape, hp)
    }
    fn kv_formula(&self, hp: &HyperParams, source: KvSource) -> KvFormula {
        // GQA with every head a KV head: `2 · n_heads · d_head` elements per token and layer.
        let elems_per_token =
            u32::try_from(2 * hp.count("n_heads").unwrap_or(0) * hp.count("d_head").unwrap_or(0))
                .unwrap_or(u32::MAX);
        let layer = LayerKv::PerToken {
            elems_per_token,
            window: None,
            latent: false,
        };
        KvFormula::new(
            vec![layer; usize::from(hp.n_layers)],
            KvDTypeRule::source_default(source),
        )
    }
    fn required_ops(&self, _: &HyperParams) -> OpSet {
        OpSet::new()
    }
}

static TOY: Toy = Toy::new("toy", &["toy", "ToyForCausalLM"]);
static TOY_TOO: Toy = Toy::new("toy", &["toy-again"]);
static TOY2: Toy = Toy::new("toy2", &["toy2", "ToyForCausalLM"]);
static TOY3: Toy = Toy::new("toy3", &["toy3"]);

const HERE: &str = module_path!();

static ONE: [Registration; 1] = [Registration {
    module: HERE,
    family: &TOY,
}];
static TWO_SAME_TAG: [Registration; 2] = [
    Registration {
        module: HERE,
        family: &TOY,
    },
    Registration {
        module: HERE,
        family: &TOY_TOO,
    },
];
static SHARED_NAME: [Registration; 2] = [
    Registration {
        module: HERE,
        family: &TOY,
    },
    Registration {
        module: "syntropy_graph::families::toy2",
        family: &TOY2,
    },
];
static THREE: [Registration; 2] = [
    Registration {
        module: HERE,
        family: &TOY,
    },
    Registration {
        module: "syntropy_graph::families::toy3",
        family: &TOY3,
    },
];

const TEST_TAGS: &[FrozenTag] = &[
    FrozenTag {
        tag: "toy",
        module: HERE,
    },
    FrozenTag {
        tag: "toy2",
        module: "syntropy_graph::families::toy2",
    },
    FrozenTag {
        tag: "toy3",
        module: "syntropy_graph::families::toy3",
    },
];

// ---- A GGUF writer ------------------------------------------------------------------------------

enum Kv {
    U32(u32),
    F32(f32),
    Str(&'static str),
}

/// A GGUF v3 file of `kvs` and F32 `tensors`, laid out as ggml lays them (alignment 32). Each
/// tensor's extent in the file comes back beside it.
fn gguf(kvs: &[(&str, Kv)], tensors: &[(&str, &[u64])]) -> (Vec<u8>, Vec<(String, Range<u64>)>) {
    fn s(out: &mut Vec<u8>, v: &str) {
        out.extend_from_slice(&(v.len() as u64).to_le_bytes());
        out.extend_from_slice(v.as_bytes());
    }
    let pad = |n: u64| n.div_ceil(32) * 32;
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
    let mut offsets = Vec::new();
    let mut at = 0u64;
    for (name, ne) in tensors {
        s(&mut out, name);
        out.extend_from_slice(&(ne.len() as u32).to_le_bytes());
        for d in *ne {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes()); // F32
        out.extend_from_slice(&at.to_le_bytes());
        let bytes = ne.iter().product::<u64>() * 4;
        offsets.push((at, bytes));
        at = pad(at + bytes);
    }
    let data_start = pad(out.len() as u64);
    out.resize(usize::try_from(data_start + at).unwrap(), 0xA5);
    let extents = tensors
        .iter()
        .zip(offsets)
        .map(|((name, _), (off, len))| {
            ((*name).to_owned(), data_start + off..data_start + off + len)
        })
        .collect();
    (out, extents)
}

/// A source over bytes that records every range read.
struct Recording {
    bytes: Vec<u8>,
    reads: RefCell<Vec<Range<u64>>>,
}

impl RangeSource for Recording {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        let end = offset + buf.len() as u64;
        let range = usize::try_from(offset).unwrap()..usize::try_from(end).unwrap();
        buf.copy_from_slice(&self.bytes[range]);
        self.reads.borrow_mut().push(offset..end);
        Ok(())
    }
}

fn toy_kvs(arch: &'static str) -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str(arch)),
        ("toy.block_count", Kv::U32(2)),
        ("toy.embedding_length", Kv::U32(64)),
        ("toy.vocab_size", Kv::U32(256)),
        ("toy.attention.head_count", Kv::U32(4)),
        ("toy.rope.freq_base", Kv::F32(10_000.0)),
        ("toy.vision.embedding_length", Kv::U32(32)),
    ]
}

const TOY_CONFIG: &[u8] = br#"{
    "architectures": ["ToyForCausalLM"],
    "model_type": "toy",
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "vocab_size": 256,
    "num_attention_heads": 4,
    "rope_theta": 10000.0,
    "vision_config": {"hidden_size": 32}
}"#;

fn header(kvs: &[(&str, Kv)]) -> GgufHeader {
    let (bytes, _) = gguf(kvs, &[]);
    let source = Recording {
        bytes,
        reads: RefCell::new(Vec::new()),
    };
    GgufHeader::read(&source, &Limits::default()).unwrap()
}

fn config(json: &str) -> HfConfig {
    HfConfig::parse(json.as_bytes(), None).unwrap()
}

// ---- Resolution ---------------------------------------------------------------------------------

#[test]
fn an_unknown_architecture_fails_with_code_1() {
    let registry = Registry::new(&ONE);
    let h = header(&[("general.architecture", Kv::Str("mystery"))]);
    let e = registry
        .resolve(&MetaView::Gguf(&h.metadata))
        .err()
        .unwrap();
    assert_eq!(e.code(), 1);
    assert_eq!(e.class(), ErrorClass::Unsupported);
    assert_eq!(
        e,
        FamilyError::UnknownArchitecture {
            names: vec!["mystery".into()],
            supported: vec!["toy"],
        }
    );
    let msg = e.to_string();
    assert!(msg.starts_with("UnknownArchitecture (Graph 1)"), "{msg}");
    assert!(msg.contains("mystery") && msg.contains("toy"), "{msg}");

    // From config.json, every name is tried and named.
    let c = config(r#"{"architectures": ["MysteryForCausalLM"], "model_type": "mystery"}"#);
    let e = registry.resolve(&MetaView::Config(&c)).err().unwrap();
    assert_eq!(e.code(), 1);
    assert!(
        matches!(&e, FamilyError::UnknownArchitecture { names, .. } if names == &["MysteryForCausalLM", "mystery"])
    );

    // Metadata that names no architecture at all.
    let h = header(&[]);
    let e = registry
        .resolve(&MetaView::Gguf(&h.metadata))
        .err()
        .unwrap();
    assert_eq!(e.code(), 1);
    assert!(e.to_string().contains("names no architecture"), "{e}");
    let c = config("{}");
    assert_eq!(
        registry
            .resolve(&MetaView::Config(&c))
            .err()
            .unwrap()
            .code(),
        1
    );
}

#[test]
fn this_builds_registry_refuses_architectures_no_family_accepts() {
    let h = header(&toy_kvs("toy"));
    let e = families()
        .resolve(&MetaView::Gguf(&h.metadata))
        .err()
        .unwrap();
    assert_eq!(e.code(), 1);
    assert!(e.to_string().contains("supported families: llama"), "{e}");
}

#[test]
fn known_architectures_resolve_from_either_source() {
    let registry = Registry::new(&ONE);
    let h = header(&toy_kvs("toy"));
    assert_eq!(
        registry
            .resolve(&MetaView::Gguf(&h.metadata))
            .unwrap()
            .name(),
        "toy"
    );
    let c = config(r#"{"architectures": ["ToyForCausalLM"]}"#);
    assert_eq!(
        registry.resolve(&MetaView::Config(&c)).unwrap().name(),
        "toy"
    );
    // `model_type` is tried after every `architectures` entry.
    let c = config(r#"{"architectures": ["Other"], "model_type": "toy"}"#);
    assert_eq!(
        registry.resolve(&MetaView::Config(&c)).unwrap().name(),
        "toy"
    );
    assert_eq!(registry.get("toy").unwrap().version(), 1);
    assert!(registry.get("llama").is_none());
}

#[test]
fn a_mistyped_architecture_key_is_code_3() {
    let registry = Registry::new(&ONE);
    let h = header(&[("general.architecture", Kv::U32(7))]);
    let e = registry
        .resolve(&MetaView::Gguf(&h.metadata))
        .err()
        .unwrap();
    assert_eq!(e.code(), 3);
    let c = config(r#"{"architectures": "ToyForCausalLM"}"#);
    let e = registry.resolve(&MetaView::Config(&c)).err().unwrap();
    assert_eq!(e.code(), 3);
    assert!(e.to_string().contains("config.json: architectures"), "{e}");
}

// ---- Hyperparameters ----------------------------------------------------------------------------

#[test]
fn both_sources_give_equal_hyperparameters() {
    let h = header(&toy_kvs("toy"));
    let c = HfConfig::parse(TOY_CONFIG, None).unwrap();
    let from_gguf = TOY.parse_hparams(&MetaView::Gguf(&h.metadata)).unwrap();
    let from_hf = TOY.parse_hparams(&MetaView::Config(&c)).unwrap();
    assert_eq!(from_gguf, from_hf);
    assert_eq!(from_gguf.n_layers, 2);
    assert_eq!(from_gguf.count("d_head"), Some(16));
}

#[test]
fn a_config_missing_a_required_field_is_code_3() {
    let c = config(r#"{"model_type": "toy", "hidden_size": 64, "vocab_size": 256}"#);
    let e = TOY.parse_hparams(&MetaView::Config(&c)).unwrap_err();
    assert_eq!(e.code(), 3);
    assert_eq!(e.class(), ErrorClass::Unsupported);
    assert_eq!(
        e,
        FamilyError::invalid("config.json: num_hidden_layers", "required, and missing")
    );
    let mut kvs = toy_kvs("toy");
    kvs.retain(|(k, _)| *k != "toy.attention.head_count");
    let h = header(&kvs);
    let e = TOY.parse_hparams(&MetaView::Gguf(&h.metadata)).unwrap_err();
    assert_eq!(
        e,
        FamilyError::invalid("toy.attention.head_count", "required, and missing")
    );
    assert!(
        e.to_string().starts_with("HyperParamInvalid (Graph 3)"),
        "{e}"
    );
}

#[test]
fn a_hyperparameter_out_of_range_is_code_3() {
    const BASE: &[(&str, &str)] = &[
        ("model_type", r#""toy""#),
        ("num_hidden_layers", "2"),
        ("hidden_size", "64"),
        ("vocab_size", "256"),
        ("num_attention_heads", "4"),
        ("rope_theta", "10000.0"),
        ("vision_config", r#"{"hidden_size": 32}"#),
    ];
    let cases = [
        ("num_hidden_layers", "0", "config.json: num_hidden_layers"),
        (
            "num_hidden_layers",
            "70000",
            "config.json: num_hidden_layers",
        ),
        (
            "num_attention_heads",
            "128",
            "config.json: num_attention_heads",
        ),
        (
            "num_attention_heads",
            "5",
            "config.json: num_attention_heads",
        ),
        ("rope_theta", "0.5", "config.json: rope_theta"),
        ("rope_theta", "-1", "config.json: rope_theta"),
        (
            "vision_config",
            r#"{"hidden_size": 2000000}"#,
            "config.json: vision_config.hidden_size",
        ),
    ];
    for (field, value, key) in cases {
        let body: Vec<String> = BASE
            .iter()
            .map(|&(k, v)| format!("\"{k}\": {}", if k == field { value } else { v }))
            .collect();
        let c = config(&format!("{{{}}}", body.join(", ")));
        let e = TOY.parse_hparams(&MetaView::Config(&c)).unwrap_err();
        assert_eq!(e.code(), 3, "{field} = {value}: {e}");
        assert!(
            matches!(&e, FamilyError::HyperParamInvalid { key: k, .. } if k == key),
            "{field} = {value}: {e}"
        );
    }
    // NaN is never in range.
    let mut kvs = toy_kvs("toy");
    kvs.retain(|(k, _)| *k != "toy.rope.freq_base");
    kvs.push(("toy.rope.freq_base", Kv::F32(f32::NAN)));
    let h = header(&kvs);
    assert_eq!(
        TOY.parse_hparams(&MetaView::Gguf(&h.metadata))
            .unwrap_err()
            .code(),
        3
    );
}

#[test]
fn malformed_metadata_is_refused() {
    // A value of the wrong type, from each source.
    let mut kvs = toy_kvs("toy");
    kvs.retain(|(k, _)| *k != "toy.block_count");
    kvs.push(("toy.block_count", Kv::Str("two")));
    let h = header(&kvs);
    let e = TOY.parse_hparams(&MetaView::Gguf(&h.metadata)).unwrap_err();
    assert_eq!(e.code(), 3);
    assert!(
        matches!(&e, FamilyError::HyperParamInvalid { key, reason } if key == "toy.block_count" && reason.contains("Str")),
        "{e}"
    );
    let c = config(r#"{"num_hidden_layers": 2.5}"#);
    let e = TOY.parse_hparams(&MetaView::Config(&c)).unwrap_err();
    assert_eq!(e.code(), 3);
    assert!(e.to_string().contains("a non-negative integer"), "{e}");
    // A GGUF `{arch}` key with no architecture to read it under.
    let h = header(&[("toy.block_count", Kv::U32(2))]);
    let e = TOY.parse_hparams(&MetaView::Gguf(&h.metadata)).unwrap_err();
    assert_eq!(e.code(), 3);
    assert!(e.to_string().contains("general.architecture"), "{e}");
    // A config.json that is not a JSON object never reaches a family.
    let e = HfConfig::parse(b"[]", None).unwrap_err();
    assert_eq!(e.class(), ErrorClass::Integrity);
    assert!(HfConfig::parse(br#"{"hidden_size": 1, "hidden_size": 2}"#, None).is_err());
}

#[test]
fn keys_absent_from_a_source_read_as_absent() {
    let c = config(r#"{"a": [1, 2]}"#);
    let view = MetaView::Config(&c);
    assert_eq!(view.u64(Key::gguf("toy.block_count")).unwrap(), None);
    assert_eq!(view.u64s(Key::config("a")).unwrap(), Some(vec![1, 2]));
    assert_eq!(view.u64_or(Key::config("b"), 7, 0..=10).unwrap(), 7);
    let e = view.require_u64(Key::gguf("x"), 0..=1).unwrap_err();
    assert!(e.to_string().contains("does not carry"), "{e}");
    let h = header(&toy_kvs("toy"));
    let view = MetaView::Gguf(&h.metadata);
    assert_eq!(view.u64(Key::config("num_hidden_layers")).unwrap(), None);
    assert_eq!(
        view.str(Key::gguf("general.architecture"))
            .unwrap()
            .as_deref(),
        Some("toy")
    );
    assert_eq!(
        view.key_name(Key::generation("{arch}.x", "eos_token_id"))
            .unwrap()
            .as_deref(),
        Some("toy.x")
    );
}

// ---- The frozen tag rule ------------------------------------------------------------------------

#[test]
fn the_registry_holds_families_to_the_frozen_list() {
    assert_eq!(Registry::new(&ONE).check(TEST_TAGS), Ok(()));
    assert_eq!(Registry::new(&THREE).check(TEST_TAGS), Ok(()));
    // This build's registry against the checked-in list.
    assert_eq!(families().check(FROZEN_TAGS), Ok(()));

    assert_eq!(
        Registry::new(&ONE).check(&TEST_TAGS[1..]),
        Err(RegistryError::Unlisted {
            tag: "toy",
            module: HERE,
        })
    );
    let moved = [FrozenTag {
        tag: "toy",
        module: "syntropy_graph::families::elsewhere",
    }];
    let e = Registry::new(&ONE).check(&moved).unwrap_err();
    assert_eq!(
        e,
        RegistryError::Reassigned {
            tag: "toy",
            listed: "syntropy_graph::families::elsewhere",
            module: HERE,
        }
    );
    assert!(e.to_string().contains("never reassigned"), "{e}");
    assert_eq!(
        Registry::new(&TWO_SAME_TAG).check(TEST_TAGS),
        Err(RegistryError::DuplicateTag("toy"))
    );
    assert_eq!(
        Registry::new(&SHARED_NAME).check(TEST_TAGS),
        Err(RegistryError::SharedArchitecture {
            name: "ToyForCausalLM",
            tags: ["toy", "toy2"],
        })
    );
}

#[test]
fn the_frozen_list_names_each_tag_and_module_once() {
    let twice_tag = [TEST_TAGS[0], TEST_TAGS[1], TEST_TAGS[0]];
    assert_eq!(
        Registry::new(&[]).check(&twice_tag),
        Err(RegistryError::DuplicateListedTag("toy"))
    );
    let twice_module = [
        TEST_TAGS[0],
        FrozenTag {
            tag: "toy_renamed",
            module: HERE,
        },
    ];
    assert_eq!(
        Registry::new(&[]).check(&twice_module),
        Err(RegistryError::DuplicateListedModule(HERE))
    );
}

// ---- Classification before any fetch -------------------------------------------------------------

/// Classification reads only the header, which is read in full before it runs, and the plan
/// it gives never names an ignored tensor, so a fetch driven by the plan never reads one. The
/// fetch path itself does not exist yet; [`TensorPlan::fetch`] is the planning output that
/// will drive it, and this test plays the fetch by reading each planned tensor's extent.
#[test]
fn vision_tower_tensors_are_ignored_and_never_fetched() {
    let tensors: &[(&str, &[u64])] = &[
        ("token_embd.weight", &[64, 256]),
        ("blk.0.attn_q.weight", &[64, 64]),
        ("blk.1.attn_q.weight", &[64, 64]),
        ("output.weight", &[64, 256]),
        ("v.blk.0.attn_q.weight", &[32, 32]),
        ("v.blk.1.attn_q.weight", &[32, 32]),
    ];
    let (bytes, extents) = gguf(&toy_kvs("toy"), tensors);
    let source = Recording {
        bytes,
        reads: RefCell::new(Vec::new()),
    };

    // 1. The header, through the one header-read routine, starting small so it reads only
    //    what it must.
    let mut reader = HeaderReader::new(&source, 1 << 30).with_initial_read(64);
    let header = GgufHeader::read_from(&mut reader, &Limits::default()).unwrap();
    let header_reads = source.reads.borrow().len();

    // 2. Resolution, hyperparameters and classification read no bytes.
    let meta = MetaView::Gguf(&header.metadata);
    let family = Registry::new(&ONE).resolve(&meta).unwrap();
    let hp = family.parse_hparams(&meta).unwrap();
    let shapes: Vec<(String, Vec<u64>)> = header
        .tensors()
        .map(|t| (t.name.to_owned(), t.ne().to_vec()))
        .collect();
    let plan = classify_tensors(
        family,
        &hp,
        shapes.iter().map(|(n, s)| (n.as_str(), s.as_slice())),
    );
    assert_eq!(
        source.reads.borrow().len(),
        header_reads,
        "classification issued a read"
    );
    assert!(plan.unknown.is_empty(), "{:?}", plan.unknown);
    assert_eq!(
        plan.ignored,
        [
            ("v.blk.0.attn_q.weight".to_owned(), MULTIMODAL_REASON),
            ("v.blk.1.attn_q.weight".to_owned(), MULTIMODAL_REASON),
        ]
    );
    assert_eq!(plan.fetch.len(), 4);

    // 3. The fetch the plan drives: every planned tensor's extent, and nothing else.
    let extent = |name: &str| extents.iter().find(|(n, _)| n == name).unwrap().1.clone();
    for (name, _) in &plan.fetch {
        let r = extent(name);
        let mut buf = vec![0; usize::try_from(r.end - r.start).unwrap()];
        source.read_at(r.start, &mut buf).unwrap();
    }

    let reads = source.reads.borrow();
    for (name, _) in &plan.ignored {
        let r = extent(name);
        for read in reads.iter() {
            assert!(
                read.end <= r.start || read.start >= r.end,
                "read {read:?} touches ignored tensor {name} at {r:?}"
            );
        }
    }
}

/// A family's `kv_formula` takes the source, so the reference's KV dtype follows RFC-0042 §4.2's
/// rule: F16 for GGUF, the compute dtype for a BF16 safetensors checkpoint.
#[test]
fn kv_formula_takes_the_source() {
    let hp = HyperParams::new(2, 64, 32)
        .with("n_heads", HParamValue::U64(4))
        .with("d_head", HParamValue::U64(16));
    let gguf = TOY.kv_formula(&hp, KvSource::Gguf);
    assert_eq!(gguf.layers.len(), 2);
    assert_eq!(
        gguf,
        KvFormula::new(
            vec![
                LayerKv::PerToken {
                    elems_per_token: 128,
                    window: None,
                    latent: false,
                };
                2
            ],
            syntropy_core::DType::F16,
        )
    );
    let bf16 = TOY.kv_formula(
        &hp,
        KvSource::Safetensors {
            compute: Some(syntropy_core::DType::BF16),
            quantized: false,
        },
    );
    assert_eq!(
        bf16.dtype,
        KvFormula::new(Vec::new(), syntropy_core::DType::BF16).dtype
    );
}

/// The default `tokenizer_spec` is the manifest crate's reading of the side files.
#[test]
fn the_default_tokenizer_spec_reads_the_side_files() {
    macro_rules! side {
        ($file:literal) => {
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../manifest/tests/data/tokenizer/llama-like/",
                $file
            ))
        };
    }
    let sources = || TokenizerSources {
        tokenizer_json: side!("tokenizer.json"),
        tokenizer_config: Some(side!("tokenizer_config.json")),
        generation_config: Some(side!("generation_config.json")),
        chat_template_file: None,
    };
    assert_eq!(
        TOY.tokenizer_spec(sources()),
        TokenizerSpec::from_sources(sources())
    );
    assert!(TOY.tokenizer_spec(sources()).is_ok());
}
