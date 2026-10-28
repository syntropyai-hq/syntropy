// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prefix keys with BLAKE3 (TIER-E7-T01; RFC-0022 §7.1): `KvFormula::kv_settings`, the
//! `kv_dtype` override's effect on the root key, and golden keys that pin the chain's bytes
//! across runs and builds. `core`'s tests check the same chain with a stand-in hash.

use syntropy_core::policy::ExactnessPolicy;
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
        ApproxParams::default(),
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

/// `kv_settings` carries every RFC-0022 §7.1 field from its inputs. The destructure names every
/// field of `KvSettings`, so a field added there fails to compile here until it is filled and
/// checked, and cannot be left out of the hash unnoticed.
#[test]
fn kv_settings_carries_every_field() {
    let f = KvFormula {
        layers: vec![
            LayerKv::gqa(8, 128, Some(512)),
            LayerKv::Compressed {
                elems_per_entry: 576,
                tokens_per_entry: 8,
                window: Some(64),
                selected: None,
                latent: true,
            },
        ],
        aux: Vec::new(),
        dtype: KvDTypeSpec {
            default: DType::BF16,
            layers: vec![(LayerIx(1), DType::F32)],
        },
        layout_version: 7,
    };
    let stored = vec![KvDType::Q8 { block: 32 }, KvDType::Native];
    let lossy = ApproxParams {
        kv_lossy: Some(KvDType::Q8 { block: 32 }),
        ..ApproxParams::default()
    };
    let KvSettings {
        page_tokens,
        stored: got_stored,
        dtype,
        dtype_layers,
        layout_version,
        rope: got_rope,
        windows,
        compression,
        knobs,
        params,
    } = f.kv_settings(32, stored.clone(), rope(), ApproxKnobs::KV_LOSSY, lossy);
    assert_eq!(page_tokens, 32);
    assert_eq!(got_stored, stored);
    assert_eq!(dtype, DType::BF16);
    assert_eq!(dtype_layers, vec![(LayerIx(1), DType::F32)]);
    assert_eq!(layout_version, 7);
    assert_eq!(got_rope, rope());
    assert_eq!(windows, vec![Some(512), Some(64)]);
    assert_eq!(compression, vec![None, Some(8)]);
    assert_eq!(knobs, ApproxKnobs::KV_LOSSY);
    assert_eq!(params, lossy);
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
        ApproxParams::default(),
    );
    assert_ne!(root(&p), r, "P");
    let knob = base.kv_settings(
        64,
        vec![KvDType::Q8 { block: 32 }],
        rope(),
        ApproxKnobs::KV_LOSSY,
        ApproxParams {
            kv_lossy: Some(KvDType::Q8 { block: 32 }),
            ..ApproxParams::default()
        },
    );
    assert_ne!(root(&knob), r, "kv_lossy");
    let mut scaled = rope();
    scaled[0].factors.insert("factor".into(), 4.0);
    let s = base.kv_settings(
        64,
        vec![KvDType::Native],
        scaled,
        ApproxKnobs::empty(),
        ApproxParams::default(),
    );
    assert_ne!(root(&s), r, "RoPE scaling");
}

/// An approximate sequence's own knob changes the root, with BLAKE3 (review B1 on #1942).
#[test]
fn a_sequence_s_knob_changes_the_root_key() {
    let s = settings_of(&formula(KvDTypeSpec::uniform(DType::F16)));
    let exact = root(&s.with_sequence(&ExactnessPolicy::Exact));
    let skip = root(&s.with_sequence(&ExactnessPolicy::Approximate {
        knobs: ApproxKnobs::EXPERT_SKIP,
        params: ApproxParams::default(),
    }));
    assert_eq!(exact, root(&s));
    assert_ne!(skip, exact);
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

/// This build's BLAKE3 values, pinned when the chain was written (TIER-E7-T01) and again when
/// `KvSettings::kv_lossy` became `params` (the review of #1942). A change means
/// every cached prefix and saved session key changes: a new `PREFIX_DOMAIN` or settings domain
/// version, not a re-pin.
const GOLDEN: [&str; 6] = [
    "76d3eabe4367c5ff3aeb1a6e418bac3bb774786d8a4075405e32e2c5fd0f3592",
    "09839aebee2a4dcf7a119f28ef8186b27fa1ccf5dc3d8b1a9e0e1d2a069cb466",
    "7c4b7a51c2c10712d96c72942c2c1c7942b0b85a9c4788b7081e3ed3d487a4aa",
    "1edf259eb3771c7195e6ec91625c30e78c81bd2777939b5db19e85528b496a0b",
    "369d20340f30b09176274cd5e367a9ab6eb2414df063ee077b85bc91cc364119",
    "77ee803d7d2c1e61b0faafbcd2fb13e7df06afb1fd69f0ad7cb7a812e04236b9",
];
