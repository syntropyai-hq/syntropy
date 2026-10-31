// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use syntropy_graph::classify::{BlockSel, Classification};
use syntropy_graph::families::llama;
use syntropy_graph::families::{Family, families};
use syntropy_graph::ir::Block;
use syntropy_graph::spec::{FamilySpec, LayerKind};
use syntropy_manifest::census::ModelManifest;

use super::*;

/// The seed the golden digests are for.
const SEED: u64 = 0x5EED;

/// A fresh directory under the system temp directory, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p =
            std::env::temp_dir().join(format!("syntropy-fixtures-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn gen_llama(seed: u64, dir: &Path) -> Vec<Fixture> {
    generate(
        llama::fixture_spec(),
        SizeClass::Nano,
        seed,
        &[QuantClass::F32],
        dir,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// Every file of every fixture, by its path below the output directory, with its bytes.
fn files(fixtures: &[Fixture], root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for f in fixtures {
        let mut entries: Vec<_> = std::fs::read_dir(&f.dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        entries.sort();
        for p in entries {
            let rel = p
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.insert(rel, std::fs::read(&p).unwrap());
        }
    }
    out
}

/// The BLAKE3 digest of every `llama` fixture file for [`SEED`]. A change to the generator
/// that changes any byte changes these; regenerate with `SYNTROPY_PRINT_FIXTURE_DIGESTS=1`
/// and say why in the change.
const LLAMA_DIGESTS: &[(&str, &str)] = &[
    (
        "llama-llama-linear/config.json",
        "533aa069066fc29077426755e9861d385c4db29949f2a679a23d5463d3edd3e4",
    ),
    (
        "llama-llama-linear/model-f32.gguf",
        "875412383b41d1ca26f392c61fda831d9840c7dcdbeaec9e4b0fc9cd02f32261",
    ),
    (
        "llama-llama-linear/model.safetensors",
        "129f3e6060dbf50ba7562913f65836c0ca37ccaa8999b9de7bc3e33e770bf22e",
    ),
    (
        "llama-llama-linear/tokenizer.json",
        "9b6b3c8be1d7fff8448ba1287395c1a66b91ee0f4daade9ab6bc076d93212571",
    ),
    (
        "llama-llama-linear/tokenizer_config.json",
        "4d13cf3c18d94e5650fe0eca93f8fe2bb753be300c4d8738b79063f98db5fd6f",
    ),
    (
        "llama-llama3/config.json",
        "0a34954d0b2618404e64a58c61722fddf258af1efa4b0e0d6a8bd88eea99b789",
    ),
    (
        "llama-llama3/model.safetensors",
        "96ca47aea5d040027e188bd9b61dffb7b8c771044b048f3c9a7854d2fd7ed385",
    ),
    (
        "llama-llama3/tokenizer.json",
        "9b6b3c8be1d7fff8448ba1287395c1a66b91ee0f4daade9ab6bc076d93212571",
    ),
    (
        "llama-llama3/tokenizer_config.json",
        "4d13cf3c18d94e5650fe0eca93f8fe2bb753be300c4d8738b79063f98db5fd6f",
    ),
    (
        "llama-qwen2/config.json",
        "9df8de19533362b9fc48c9e20df4b10abdd4a8daed295582be4bc7cc7432394d",
    ),
    (
        "llama-qwen2/model.safetensors",
        "f7a68cd8010ce5942d30879c72c677549e582a5c2292b05448711176db674922",
    ),
    (
        "llama-qwen2/tokenizer.json",
        "9b6b3c8be1d7fff8448ba1287395c1a66b91ee0f4daade9ab6bc076d93212571",
    ),
    (
        "llama-qwen2/tokenizer_config.json",
        "4d13cf3c18d94e5650fe0eca93f8fe2bb753be300c4d8738b79063f98db5fd6f",
    ),
    (
        "llama-qwen3/config.json",
        "543b934b067e23d7008171f27f98961f7167560d7896504dafa4fcc738000bf1",
    ),
    (
        "llama-qwen3/model-f32.gguf",
        "80dcbafa5d9b6a4df21978125a721351a7bf23eb7b2bca4692a4e38c6f573972",
    ),
    (
        "llama-qwen3/model.safetensors",
        "2f90e067b0510823d697728d6ae5b921398ab3a2597b4dc335ee5fb83dd40758",
    ),
    (
        "llama-qwen3/tokenizer.json",
        "9b6b3c8be1d7fff8448ba1287395c1a66b91ee0f4daade9ab6bc076d93212571",
    ),
    (
        "llama-qwen3/tokenizer_config.json",
        "4d13cf3c18d94e5650fe0eca93f8fe2bb753be300c4d8738b79063f98db5fd6f",
    ),
];

#[test]
fn a_seed_gives_byte_identical_files() {
    let (a, b) = (TempDir::new("same-a"), TempDir::new("same-b"));
    let fa = files(&gen_llama(SEED, &a.0), &a.0);
    let fb = files(&gen_llama(SEED, &b.0), &b.0);
    assert_eq!(fa.keys().collect::<Vec<_>>(), fb.keys().collect::<Vec<_>>());
    for (k, v) in &fa {
        assert!(fb[k] == *v, "{k} differs between two runs of one seed");
    }
    let c = TempDir::new("other-seed");
    let fc = files(&gen_llama(SEED + 1, &c.0), &c.0);
    assert_ne!(
        fa["llama-qwen3/model.safetensors"], fc["llama-qwen3/model.safetensors"],
        "another seed gives other weights"
    );
}

#[test]
fn the_files_match_their_golden_digests() {
    let dir = TempDir::new("golden");
    let got: Vec<(String, String)> = files(&gen_llama(SEED, &dir.0), &dir.0)
        .into_iter()
        .map(|(k, v)| (k, blake3::hash(&v).to_hex().to_string()))
        .collect();
    if std::env::var_os("SYNTROPY_PRINT_FIXTURE_DIGESTS").is_some() {
        for (k, h) in &got {
            println!("    (\"{k}\", \"{h}\"),");
        }
    }
    let want: Vec<(String, String)> = LLAMA_DIGESTS
        .iter()
        .map(|(k, h)| ((*k).to_owned(), (*h).to_owned()))
        .collect();
    assert_eq!(got, want, "the generated bytes changed; see LLAMA_DIGESTS");
}

/// The blocks of each decoder layer, by the family's own classification of every tensor.
fn layer_blocks(family: &dyn Family, m: &ModelManifest) -> BTreeMap<u16, BTreeSet<String>> {
    let mut out: BTreeMap<u16, BTreeSet<String>> = BTreeMap::new();
    for t in &m.census.tensors {
        match family.classify(&t.name, &t.shape, &m.hparams) {
            Classification::Classified {
                block: BlockSel::Layer(b),
                layer: Some(l),
                ..
            } => {
                out.entry(l.0).or_default().insert(format!("{b:?}"));
            }
            Classification::Classified { .. } | Classification::Ignored { .. } => {}
            Classification::Unknown => panic!("{} is unclassified", t.name),
        }
    }
    out
}

fn kind_blocks(kind: LayerKind) -> BTreeSet<String> {
    let blocks = match kind {
        LayerKind::GqaDense => [Block::Gqa, Block::DenseFfn],
        LayerKind::GqaMoe => [Block::Gqa, Block::Moe],
        other => panic!("no blocks listed for {other:?}"),
    };
    blocks.iter().map(|b| format!("{b:?}")).collect()
}

/// Checks AC2 and AC3 for every fixture of `spec`: each file's census classifies every
/// tensor, and each layer holds its pattern's blocks, so every layer kind is present.
fn check_structure(spec: &FamilySpec, fixtures: &[Fixture]) {
    let family = families().get(spec.family).expect("a registered family");
    assert_eq!(fixtures.len(), spec.variants.len());
    for (f, v) in fixtures.iter().zip(spec.variants) {
        assert_eq!(f.variant, v.name);
        let original = original(spec.family, v, SizeClass::Nano, SEED).unwrap();
        let mut manifests = vec![("safetensors", f.manifest.clone())];
        for (_, path) in &f.gguf {
            manifests.push(("gguf", census_gguf(path).unwrap_or_else(|e| panic!("{e}"))));
        }
        for (format, m) in &manifests {
            assert_eq!(m.family, spec.family, "{} {format}", v.name);
            // Every written tensor is in the census or ignored; none is unclassified, or the
            // census would have refused the file.
            let placed = m.census.tensors.len() + m.ignored.len();
            assert_eq!(placed, original.tensors.len(), "{} {format}", v.name);
            let blocks = layer_blocks(family, m);
            assert_eq!(
                blocks.len(),
                original.dims.layers as usize,
                "{} {format}",
                v.name
            );
            for (l, got) in &blocks {
                let kind = v.layers[usize::from(*l) % v.layers.len()];
                assert_eq!(*got, kind_blocks(kind), "{} {format} layer {l}", v.name);
            }
            let names: BTreeSet<_> = m.census.tensors.iter().map(|t| t.name.as_str()).collect();
            let has = |s: &str| names.iter().any(|n| n.contains(s));
            assert_eq!(has("q_norm"), v.attention.qk_norm, "{} {format}", v.name);
            assert_eq!(
                has("q_proj.bias") || has("attn_q.bias"),
                v.attention.qkv_bias,
                "{} {format}",
                v.name
            );
        }
        for kind in v.layers {
            assert!(
                (0..original.dims.layers as usize).any(|l| v.layers[l % v.layers.len()] == *kind),
                "{} lacks {kind:?}",
                v.name
            );
        }
    }
}

#[test]
fn every_llama_fixture_is_classified_and_structure_true() {
    let dir = TempDir::new("llama-structure");
    let fixtures = gen_llama(SEED, &dir.0);
    check_structure(llama::fixture_spec(), &fixtures);
    // GGUF can state the plain and linear rules only; the safetensors original exists for all.
    let with_gguf: Vec<_> = fixtures
        .iter()
        .filter(|f| !f.gguf.is_empty())
        .map(|f| f.variant)
        .collect();
    assert_eq!(with_gguf, ["llama-linear", "qwen3"]);
    assert!(fixtures.iter().all(|f| f.safetensors.is_some()));
    // Tied variants store no head; the family ties it to the token table.
    for f in &fixtures {
        let v = llama::fixture_spec()
            .variants
            .iter()
            .find(|v| v.name == f.variant)
            .unwrap();
        let head = f
            .manifest
            .census
            .tensors
            .iter()
            .any(|t| t.name == "lm_head.weight");
        assert_eq!(head, !v.tied_embeddings, "{}", v.name);
        assert_eq!(f.manifest.hparams.vocab, tokenizer::VOCAB);
        assert!(f.manifest.tokenizer.is_some(), "{}", v.name);
    }
}

#[test]
fn gguf_and_safetensors_hold_the_same_weights() {
    let dir = TempDir::new("same-weights");
    let fixtures = gen_llama(SEED, &dir.0);
    for f in fixtures.iter().filter(|f| !f.gguf.is_empty()) {
        let gguf = census_gguf(&f.gguf[0].1).unwrap();
        let st = &f.manifest;
        assert_eq!(gguf.hparams.n_layers, st.hparams.n_layers);
        assert_eq!(gguf.hparams.hidden, st.hparams.hidden);
        assert_eq!(gguf.hparams.vocab, st.hparams.vocab);
        assert_eq!(gguf.kv.layers.len(), st.kv.layers.len());
        assert_eq!(
            gguf.census.tensors.iter().map(|t| t.bytes.0).sum::<u64>(),
            st.census.tensors.iter().map(|t| t.bytes.0).sum::<u64>(),
            "{}",
            f.variant
        );
    }
}

#[test]
fn llama_gguf_q_and_k_rows_interleave_each_head_halves() {
    let t = model::Tensor {
        gguf: String::new(),
        hf: String::new(),
        shape: vec![8, 1],
        init: model::Init::Unit,
        gguf_rope_heads: Some(2),
        data: (0..8).map(|x| x as f32).collect(),
    };
    // Head 0 holds rows 0–3: halves [0, 1] and [2, 3] interleave to 0, 2, 1, 3.
    assert_eq!(t.gguf_data(), [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
}

#[test]
fn the_logarithm_is_the_platform_one_to_an_ulp_and_draws_are_standard() {
    let mut n = Normals::new(1);
    for i in 1..10_000u32 {
        let x = f64::from(i) / 10_000.0;
        let (got, want) = (normal::ln(x), x.ln());
        assert!(
            (got - want).abs() <= 2.0 * f64::EPSILON * want.abs().max(1.0),
            "ln({x}) = {got}, want {want}"
        );
    }
    let draws: Vec<f64> = (0..200_000).map(|_| n.draw()).collect();
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    let var = draws.iter().map(|z| z * z).sum::<f64>() / draws.len() as f64 - mean * mean;
    assert!(mean.abs() < 0.01, "mean {mean}");
    assert!((var - 1.0).abs() < 0.02, "variance {var}");
}

#[test]
fn the_tokenizer_spells_every_byte_once() {
    let t = tokenizer::tokens();
    assert_eq!(t.len(), tokenizer::VOCAB as usize);
    let distinct: BTreeSet<_> = t.iter().collect();
    assert_eq!(distinct.len(), t.len());
    assert_eq!(t[usize::from(b'A')], "A");
    assert_eq!(t[0], "\u{100}");
    assert_eq!(t[usize::from(b' ')], "\u{120}");
}

#[test]
fn unbuilt_size_classes_and_quant_classes_are_refused() {
    let dir = TempDir::new("refused");
    let e = generate(llama::fixture_spec(), SizeClass::Micro, SEED, &[], &dir.0).unwrap_err();
    assert!(matches!(e, FixtureError::Unsupported(_)), "{e}");
}
