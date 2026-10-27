// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prefix keys with BLAKE3 (TIER-E7-T01; RFC-0022 §7.1): `KvFormula::kv_settings`, the
//! `kv_dtype` override's effect on the root key, and golden keys that pin the chain's bytes
//! across runs and builds. `core`'s tests check the same chain with a stand-in hash.

use syntropy_core::sequence::PrefixScope;
use syntropy_core::{Blake3, ModelId, PrefixKeyChain, PrefixScopeMode};

use super::dtype::{KvDTypeOverride, KvDTypeRule, KvSource};
use super::*;

fn blake3(bytes: &[u8]) -> Blake3 {
    Blake3(*blake3::hash(bytes).as_bytes())
}

fn hex(b: &Blake3) -> String {
    b.0.iter().map(|x| format!("{x:02x}")).collect()
}

fn model() -> ModelId {
    ModelId(blake3(b"a test model"))
}

/// A four-layer formula: two global GQA layers and two with a 4096-token window.
fn formula(dtype: KvDTypeSpec) -> KvFormula {
    KvFormula {
        layers: vec![
            LayerKv::gqa(8, 128, None),
            LayerKv::gqa(8, 128, Some(4096)),
            LayerKv::gqa(8, 128, None),
            LayerKv::gqa(8, 128, Some(4096)),
        ],
        aux: Vec::new(),
        dtype,
        layout_version: 1,
    }
}

fn rope() -> Vec<RopeSettings> {
    vec![RopeSettings {
        base: 500_000.0,
        scaling: Some("llama3".into()),
        factors: [("factor".to_owned(), 8.0)].into_iter().collect(),
        original_context: Some(8192),
    }]
}

fn settings_of(f: &KvFormula) -> KvSettings {
    f.kv_settings(
        64,
        vec![KvDType::Native],
        rope(),
        ApproxKnobs::empty(),
        None,
    )
}

fn root(s: &KvSettings) -> Blake3 {
    PrefixKeyChain::new(
        blake3,
        model(),
        s.hash_with(blake3),
        &PrefixScope::default(),
        PrefixScopeMode::Principal,
        s.page_tokens,
    )
    .last()
    .0
}

#[test]
fn the_formula_fills_its_part_of_the_settings() {
    let f = formula(KvDTypeSpec {
        default: DType::F16,
        layers: vec![(LayerIx(2), DType::F32), (LayerIx(0), DType::F16)],
    });
    let s = settings_of(&f);
    assert_eq!(s.dtype, DType::F16);
    assert_eq!(
        s.dtype_layers,
        vec![(LayerIx(2), DType::F32)],
        "normal form"
    );
    assert_eq!(s.layout_version, 1);
    assert_eq!(s.windows, vec![None, Some(4096), None, Some(4096)]);
    assert_eq!(s.compression, vec![None; 4]);
    let mut c = f.clone();
    c.layers.push(LayerKv::Compressed {
        elems_per_entry: 576,
        tokens_per_entry: 4,
        window: Some(128),
        selected: None,
        latent: true,
    });
    c.layers.push(LayerKv::Fixed {
        elems_per_seq: 1024,
    });
    let s = settings_of(&c);
    assert_eq!(s.windows[4..], [Some(128), None]);
    assert_eq!(s.compression[4..], [Some(4), None]);
}

/// The `kv_dtype` override (COMP-E13-T06) changes the root key: it changes the resolved
/// `KvDTypeSpec`, which `KvFormula::dtype` holds and `kv_settings_hash` covers.
#[test]
fn the_kv_dtype_override_changes_the_root_key() {
    let rule = KvDTypeRule::for_source(KvSource::Gguf);
    let reference = formula(rule.resolve(None).unwrap());
    let forced = formula(rule.resolve(Some(KvDTypeOverride::Bf16)).unwrap());
    let wider = formula(rule.resolve(Some(KvDTypeOverride::F32)).unwrap());
    let same = formula(rule.resolve(Some(KvDTypeOverride::F16)).unwrap());
    let r = root(&settings_of(&reference));
    assert_ne!(root(&settings_of(&forced)), r);
    assert_ne!(root(&settings_of(&wider)), r);
    assert_ne!(root(&settings_of(&forced)), root(&settings_of(&wider)));
    assert_eq!(
        root(&settings_of(&same)),
        r,
        "the reference's own dtype changes nothing"
    );
}

/// The formula's other KV-affecting parts, through the formula, with BLAKE3.
#[test]
fn the_formula_s_other_settings_change_the_root_key() {
    let base = formula(KvDTypeSpec::uniform(DType::F16));
    let r = root(&settings_of(&base));
    let mut layout = base.clone();
    layout.layout_version = 2;
    let mut window = base.clone();
    window.layers[1] = LayerKv::gqa(8, 128, Some(1024));
    let mut compressed = base.clone();
    compressed.layers[2] = LayerKv::Compressed {
        elems_per_entry: 2048,
        tokens_per_entry: 4,
        window: None,
        selected: None,
        latent: false,
    };
    let mut ratio = compressed.clone();
    ratio.layers[2] = LayerKv::Compressed {
        elems_per_entry: 2048,
        tokens_per_entry: 128,
        window: None,
        selected: None,
        latent: false,
    };
    for (name, f) in [
        ("layout version", &layout),
        ("window", &window),
        ("compression", &compressed),
    ] {
        assert_ne!(root(&settings_of(f)), r, "{name}");
    }
    assert_ne!(root(&settings_of(&ratio)), root(&settings_of(&compressed)));
    let p = base.kv_settings(
        128,
        vec![KvDType::Native],
        rope(),
        ApproxKnobs::empty(),
        None,
    );
    assert_ne!(root(&p), r, "P");
    let knob = base.kv_settings(
        64,
        vec![KvDType::Q8 { block: 32 }],
        rope(),
        ApproxKnobs::KV_LOSSY,
        Some(KvDType::Q8 { block: 32 }),
    );
    assert_ne!(root(&knob), r, "kv_lossy");
    let mut scaled = rope();
    scaled[0].factors.insert("factor".into(), 4.0);
    let s = base.kv_settings(
        64,
        vec![KvDType::Native],
        scaled,
        ApproxKnobs::empty(),
        None,
    );
    assert_ne!(root(&s), r, "RoPE scaling");
}

/// Golden keys: BLAKE3 over the bytes this build writes, pinned, so a change to any part of the
/// chain or of `KvSettings`' encoding, on any host, fails here. Two principals share the first
/// two pages of a span that ends at token 150 and part from page 2.
#[test]
fn golden_keys_are_stable_across_runs() {
    let s = settings_of(&formula(KvDTypeSpec::uniform(DType::F16)));
    let tokens: Vec<u32> = (0..64 * 3).collect();
    let keys = |p: u8| {
        let scope = PrefixScope {
            principal: Some([p; 16]),
            shared_until: Some(150),
        };
        let mut chain = PrefixKeyChain::new(
            blake3,
            model(),
            s.hash_with(blake3),
            &scope,
            PrefixScopeMode::Principal,
            64,
        );
        let root = chain.last().0;
        let pages: Vec<String> = chain
            .push_tokens(&tokens, &[])
            .iter()
            .map(|k| hex(&k.0))
            .collect();
        (hex(&root), pages)
    };
    let (root_a, a) = keys(1);
    let (root_b, b) = keys(2);
    assert_eq!(root_a, root_b);
    assert_eq!(a[..2], b[..2]);
    assert_ne!(a[2], b[2]);
    // kv_settings_hash, k_root, principal 1's three pages, then principal 2's page 2.
    let got = [
        hex(&s.hash_with(blake3)),
        root_a,
        a[0].clone(),
        a[1].clone(),
        a[2].clone(),
        b[2].clone(),
    ];
    assert_eq!(got, GOLDEN, "a prefix key changed");
}

/// This build's BLAKE3 values, pinned when the chain was written (TIER-E7-T01). A change means
/// every cached prefix and saved session key changes: a new `PREFIX_DOMAIN` or settings domain
/// version, not a re-pin.
const GOLDEN: [&str; 6] = [
    "c101e9cd7667d3319ecd19b7adede8ed0fb0b13e5ec734828d628b175a7cd443",
    "c11b3fc645a62f37ab3f36490cdb4fe317118404d01182a522847c72a8ad58ec",
    "1c21687e0803900f722d66210ce3588c48742acb4d177633674ccb9a1ae1e9ad",
    "fa6dc55a01f153193fb9becc23fa2e64623d4a128680525c171f685113be5435",
    "a374015ca8cf5bdbd44227f473d6ebfcd2b1b1b2cfe02ce1b77da4deca874871",
    "248f9500a86322c5959a876c0a0e2dbc8a21e5e3e17277d6b66c6f1554178d5d",
];
