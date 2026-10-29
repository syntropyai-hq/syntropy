// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the prefix keys (TIER-E7-T01; RFC-0022 §7.1).
//!
//! `core` links no hasher, so these use two stand-ins: [`record`], which returns its input's
//! length and keeps the bytes, to check exactly what is hashed; and [`mix`], a deterministic
//! 256-bit mixer, for the properties. The BLAKE3 values, golden keys included, are checked in
//! `manifest`, which links BLAKE3.

use std::cell::RefCell;

use proptest::prelude::*;

use super::*;

/// A deterministic stand-in for BLAKE3: four FNV-1a-64 lanes with different seeds.
fn mix(bytes: &[u8]) -> Blake3 {
    let mut out = [0u8; 32];
    for (lane, chunk) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (lane as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        chunk.copy_from_slice(&h.to_le_bytes());
    }
    Blake3(out)
}

/// A named edit of the settings.
type Change = (&'static str, Box<dyn Fn(&mut KvSettings)>);

fn model() -> ModelId {
    ModelId(Blake3([7; 32]))
}

fn settings() -> KvSettings {
    KvSettings {
        page_tokens: 64,
        stored: vec![KvDType::Native, KvDType::Native],
        dtype: DType::F16,
        dtype_layers: vec![(LayerIx(3), DType::F32)],
        layout_version: 1,
        rope: vec![RopeSettings {
            base: 500_000.0,
            scaling: Some("llama3".into()),
            factors: [
                ("factor".to_owned(), 8.0),
                ("low_freq_factor".to_owned(), 1.0),
            ]
            .into_iter()
            .collect(),
            original_context: Some(8192),
        }],
        windows: vec![None, Some(4096), None, Some(4096)],
        compression: vec![None, None, None, None],
        knobs: ApproxKnobs::empty(),
        params: ApproxParams::default(),
    }
}

fn root(s: &KvSettings, scope: &PrefixScope, mode: PrefixScopeMode) -> PrefixKey {
    PrefixKeyChain::new(mix, model(), s.hash_with(mix), scope, mode, s.page_tokens).last()
}

fn principal(p: u8, shared_until: Option<u32>) -> PrefixScope {
    PrefixScope {
        principal: Some([p; 16]),
        shared_until,
    }
}

fn tokens(n: u32, seed: u32) -> Vec<u32> {
    (0..n)
        .map(|i| i.wrapping_mul(2_654_435_761) ^ seed)
        .collect()
}

/// Every KV-affecting setting RFC-0022 §7.1 lists changes `kv_settings_hash`, and so the root
/// key. The criterion's list: the dtype (with the `kv_dtype` override, which changes it), the
/// layout version, P, RoPE scaling, the window, the compression ratio and a KV knob.
#[test]
fn every_kv_affecting_setting_changes_the_root_key() {
    let base = settings();
    let scope = PrefixScope::default();
    let r0 = root(&base, &scope, PrefixScopeMode::Principal);
    let changes: Vec<Change> = vec![
        (
            "dtype F16 → BF16 (a kv_dtype override)",
            Box::new(|s| s.dtype = DType::BF16),
        ),
        (
            "a per-layer dtype",
            Box::new(|s| s.dtype_layers.push((LayerIx(1), DType::BF16))),
        ),
        (
            "a per-layer dtype's value",
            Box::new(|s| s.dtype_layers[0].1 = DType::BF16),
        ),
        ("layout version", Box::new(|s| s.layout_version = 2)),
        ("P", Box::new(|s| s.page_tokens = 128)),
        (
            "a stored KvDType",
            Box::new(|s| s.stored[1] = KvDType::Q8 { block: 32 }),
        ),
        (
            "a page class more",
            Box::new(|s| s.stored.push(KvDType::Native)),
        ),
        ("RoPE base", Box::new(|s| s.rope[0].base = 10_000.0)),
        (
            "RoPE scaling type",
            Box::new(|s| s.rope[0].scaling = Some("yarn".into())),
        ),
        ("no RoPE scaling", Box::new(|s| s.rope[0].scaling = None)),
        (
            "a RoPE scaling factor",
            Box::new(|s| {
                s.rope[0].factors.insert("factor".into(), 4.0);
            }),
        ),
        (
            "a RoPE factor more",
            Box::new(|s| {
                s.rope[0].factors.insert("beta_fast".into(), 32.0);
            }),
        ),
        (
            "RoPE original context",
            Box::new(|s| s.rope[0].original_context = Some(4096)),
        ),
        (
            "a second RoPE set",
            Box::new(|s| {
                let r = s.rope[0].clone();
                s.rope.push(r);
            }),
        ),
        ("a window", Box::new(|s| s.windows[1] = Some(1024))),
        ("a window added", Box::new(|s| s.windows[0] = Some(4096))),
        (
            "a compression ratio",
            Box::new(|s| s.compression[2] = Some(4)),
        ),
        (
            "the kv_lossy knob",
            Box::new(|s| {
                s.knobs = ApproxKnobs::KV_LOSSY;
                s.params.kv_lossy = Some(KvDType::Q8 { block: 32 });
            }),
        ),
        (
            "the approx_attention knob",
            Box::new(|s| s.knobs = ApproxKnobs::APPROX_ATTENTION),
        ),
        // The knobs that change K and V through the forward pass (review B1 on #1942).
        (
            "the expert_skip knob",
            Box::new(|s| s.knobs = ApproxKnobs::EXPERT_SKIP),
        ),
        (
            "the expert_defer knob",
            Box::new(|s| s.knobs = ApproxKnobs::EXPERT_DEFER),
        ),
        (
            "the topk_reduce knob",
            Box::new(|s| s.knobs = ApproxKnobs::TOPK_REDUCE),
        ),
        (
            "the miss_low_precision knob",
            Box::new(|s| s.knobs = ApproxKnobs::MISS_LOW_PRECISION),
        ),
        (
            "the wire_compress knob",
            Box::new(|s| s.knobs = ApproxKnobs::WIRE_COMPRESS),
        ),
    ];
    let mut seen = vec![r0];
    for (name, change) in &changes {
        let mut s = base.clone();
        change(&mut s);
        let r = root(&s, &scope, PrefixScopeMode::Principal);
        assert_ne!(r, r0, "{name}");
        assert!(!seen.contains(&r), "{name} collides with an earlier change");
        seen.push(r);
    }
    // kv_lossy's parameter counts too, while the knob is on.
    let lossy = |block| {
        let mut s = base.clone();
        s.knobs = ApproxKnobs::KV_LOSSY;
        s.params.kv_lossy = Some(KvDType::Q8 { block });
        root(&s, &scope, PrefixScopeMode::Principal)
    };
    assert_ne!(lossy(32), lossy(64));
}

/// Settings that store the same KV give the same key: the normal form.
/// Every knob but `SPEC_LENIENT` is a KV knob, a knob added later included; each knob's
/// parameter counts while the knob is on and not while it is off.
#[test]
fn every_forward_pass_knob_and_its_parameter_counts() {
    assert_eq!(
        KV_KNOBS,
        ApproxKnobs::all().difference(ApproxKnobs::SPEC_LENIENT)
    );
    assert!(!KV_KNOBS.contains(ApproxKnobs::SPEC_LENIENT));
    let base = settings();
    let scope = PrefixScope::default();
    let root_of = |s: &KvSettings| root(s, &scope, PrefixScopeMode::Principal);
    let with = |knobs, params: ApproxParams| {
        let mut s = base.clone();
        s.knobs = knobs;
        s.params = params;
        root_of(&s)
    };
    let mlp = |scheme| ApproxParams {
        miss_low_precision: Some(scheme),
        ..ApproxParams::default()
    };
    let wire = |dtype| ApproxParams {
        wire_compress: Some(dtype),
        ..ApproxParams::default()
    };
    let q4 = crate::quant::QuantScheme::Ggml(crate::dtype::GgmlType::Q4_0);
    let q8 = crate::quant::QuantScheme::Ggml(crate::dtype::GgmlType::Q8_0);
    assert_ne!(
        with(ApproxKnobs::MISS_LOW_PRECISION, mlp(q4)),
        with(ApproxKnobs::MISS_LOW_PRECISION, mlp(q8))
    );
    assert_ne!(
        with(
            ApproxKnobs::WIRE_COMPRESS,
            wire(crate::plan::WireDType::F16)
        ),
        with(
            ApproxKnobs::WIRE_COMPRESS,
            wire(crate::plan::WireDType::Q8B32)
        )
    );
    // Off, a parameter is not in the key.
    assert_eq!(with(ApproxKnobs::empty(), mlp(q4)), root_of(&base));
    assert_eq!(
        with(ApproxKnobs::empty(), wire(crate::plan::WireDType::F16)),
        root_of(&base)
    );
}

/// A sequence's own knobs, from `SequenceOpen::exactness`, join the instance's: an approximate
/// sequence on an exact instance gets other keys than an exact sequence (review B1 on #1942).
#[test]
fn an_approximate_sequence_on_an_exact_instance_gets_other_keys() {
    let instance = settings();
    let scope = PrefixScope::default();
    let root_of = |s: &KvSettings| root(s, &scope, PrefixScopeMode::Principal);
    let exact = instance.with_sequence(&ExactnessPolicy::Exact);
    assert_eq!(exact, instance);
    for knob in [
        ApproxKnobs::EXPERT_SKIP,
        ApproxKnobs::TOPK_REDUCE,
        ApproxKnobs::EXPERT_DEFER,
    ] {
        let seq = instance.with_sequence(&ExactnessPolicy::Approximate {
            knobs: knob,
            params: ApproxParams::default(),
        });
        assert_eq!(seq.knobs, knob);
        assert_ne!(root_of(&seq), root_of(&exact), "{knob:?}");
    }
    // A sequence's per-sequence knobs add to the instance's; its per-instance parameter does not
    // replace the instance's, which the pages are stored under (review N1 on #1942).
    let mut lossy = instance.clone();
    lossy.knobs = ApproxKnobs::KV_LOSSY;
    lossy.params.kv_lossy = Some(KvDType::Q8 { block: 32 });
    let seq = lossy.with_sequence(&ExactnessPolicy::Approximate {
        knobs: ApproxKnobs::EXPERT_SKIP,
        params: ApproxParams {
            kv_lossy: Some(KvDType::Q8 { block: 64 }),
            ..ApproxParams::default()
        },
    });
    assert_eq!(seq.knobs, ApproxKnobs::KV_LOSSY | ApproxKnobs::EXPERT_SKIP);
    assert_eq!(seq.params.kv_lossy, Some(KvDType::Q8 { block: 32 }));
}

/// A sequence's per-instance knobs and parameters leave its root as the instance's: the plan
/// applies those knobs to every sequence or to none (RFC-0054 §4.2), so the pages are stored as
/// the instance says (review N1 on #1942).
#[test]
fn a_sequence_s_per_instance_knobs_and_parameters_do_not_change_the_root() {
    let scope = PrefixScope::default();
    let root_of = |s: &KvSettings| root(s, &scope, PrefixScopeMode::Principal);
    let per_instance = ExactnessPolicy::Approximate {
        knobs: ApproxKnobs::PER_INSTANCE,
        params: ApproxParams {
            miss_low_precision: Some(crate::quant::QuantScheme::Ggml(
                crate::dtype::GgmlType::Q4_0,
            )),
            kv_lossy: Some(KvDType::Q8 { block: 64 }),
            wire_compress: Some(crate::plan::WireDType::F16),
        },
    };

    // On an exact instance, they leave the exact root.
    let exact = settings();
    let seq = exact.with_sequence(&per_instance);
    assert_eq!(seq, exact);
    assert_eq!(root_of(&seq), root_of(&exact));

    // On an instance with those knobs on, the instance's parameters stay.
    let mut lossy = settings();
    lossy.knobs = ApproxKnobs::PER_INSTANCE;
    lossy.params = ApproxParams {
        miss_low_precision: Some(crate::quant::QuantScheme::Ggml(
            crate::dtype::GgmlType::Q8_0,
        )),
        kv_lossy: Some(KvDType::Q8 { block: 32 }),
        wire_compress: Some(crate::plan::WireDType::Q8B32),
    };
    let seq = lossy.with_sequence(&per_instance);
    assert_eq!(seq, lossy);
    assert_eq!(root_of(&seq), root_of(&lossy));

    // With a per-sequence knob as well, only that knob is added.
    let mut mixed = per_instance;
    if let ExactnessPolicy::Approximate { knobs, .. } = &mut mixed {
        *knobs |= ApproxKnobs::EXPERT_SKIP;
    }
    let seq = lossy.with_sequence(&mixed);
    assert_eq!(
        seq.knobs,
        ApproxKnobs::PER_INSTANCE | ApproxKnobs::EXPERT_SKIP
    );
    assert_eq!(seq.params, lossy.params);
    assert_ne!(root_of(&seq), root_of(&lossy));
}

#[test]
fn what_does_not_change_kv_does_not_change_the_key() {
    let base = settings();
    let scope = PrefixScope::default();
    let r0 = root(&base, &scope, PrefixScopeMode::Principal);
    let same: Vec<Change> = vec![
        (
            "spec_lenient, the one knob that does not change KV",
            Box::new(|s| s.knobs = ApproxKnobs::SPEC_LENIENT),
        ),
        (
            "kv_lossy's dtype with the knob off",
            Box::new(|s| s.params.kv_lossy = Some(KvDType::Q8 { block: 32 })),
        ),
        (
            "a per-layer dtype equal to the default",
            Box::new(|s| s.dtype_layers.push((LayerIx(0), DType::F16))),
        ),
        (
            "a second entry for a layer",
            Box::new(|s| s.dtype_layers.push((LayerIx(3), DType::BF16))),
        ),
        (
            "factors inserted in another order",
            Box::new(|s| {
                s.rope[0].factors = [
                    ("low_freq_factor".to_owned(), 1.0),
                    ("factor".to_owned(), 8.0),
                ]
                .into_iter()
                .collect();
            }),
        ),
    ];
    for (name, change) in &same {
        let mut s = base.clone();
        change(&mut s);
        assert_eq!(root(&s, &scope, PrefixScopeMode::Principal), r0, "{name}");
    }
    // Exceptions given out of order.
    let mut a = base.clone();
    a.dtype_layers = vec![(LayerIx(5), DType::BF16), (LayerIx(3), DType::F32)];
    let mut b = base.clone();
    b.dtype_layers = vec![(LayerIx(3), DType::F32), (LayerIx(5), DType::BF16)];
    assert_eq!(a.canonical_bytes(), b.canonical_bytes());
}

/// The bytes hashed are exactly RFC-0022 §7.1's, for text.
#[test]
fn the_chain_hashes_rfc_0022_s_bytes() {
    let inputs = RefCell::new(Vec::<Vec<u8>>::new());
    let record = |bytes: &[u8]| {
        inputs.borrow_mut().push(bytes.to_vec());
        mix(bytes)
    };
    let settings_hash = Blake3([9; 32]);
    let p = 4;
    let mut chain = PrefixKeyChain::new(
        record,
        model(),
        settings_hash,
        &principal(0xAB, Some(5)),
        PrefixScopeMode::Principal,
        p,
    );
    let root = chain.last();
    let k0 = chain.push_page(&[1, 2, 3, 4], &[]);
    let k1 = chain.push_page(&[5, 6, 7, 8], &[]);
    let inputs = inputs.into_inner();
    assert_eq!(inputs.len(), 3);
    // k_root: the domain, the ModelId, the settings hash, and an empty shared_salt.
    let mut want = b"syntropy-kv-prefix-v1".to_vec();
    want.extend([7; 32]);
    want.extend([9; 32]);
    assert_eq!(inputs[0], want);
    // Page 0 is wholly inside the shared span (5 tokens → 1 whole page): no salt.
    let mut want = root.0.0.to_vec();
    for t in 1u32..=4 {
        want.extend(t.to_le_bytes());
    }
    assert_eq!(inputs[1], want);
    // Page 1 straddles the breakpoint, so it is private: the scope salt comes first.
    let mut want = k0.0.0.to_vec();
    want.extend([0xAB; 16]);
    for t in 5u32..=8 {
        want.extend(t.to_le_bytes());
    }
    assert_eq!(inputs[2], want);
    assert_eq!(k1, PrefixKey(mix(&inputs[2])));
}

/// The kv_settings_hash bytes: the domain, then `canon`'s encoding of the normal form. Pinned, so
/// a change to the encoding, which would change every key, fails here.
#[test]
fn the_settings_bytes_are_pinned() {
    let bytes = settings().canonical_bytes();
    assert!(bytes.starts_with(b"syntropy/kv-settings/v1"));
    assert_eq!(
        &bytes[KV_SETTINGS_DOMAIN.len()..],
        canon::to_bytes(&settings().normalized()).unwrap()
    );
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        hex, PINNED_SETTINGS_HEX,
        "the canonical encoding of KvSettings changed"
    );
}

/// `settings()`'s bytes: the domain, then a struct (`0b`) of 10 fields by name, `compression`
/// first and `windows` last; enums by variant name (`F16`, `Native`); `knobs` as its `u32` bits;
/// `params` as a struct of no fields while every knob is off.
const PINNED_SETTINGS_HEX: &str = concat!(
    "73796e74726f70792f6b762d73657474696e67732f76310b0a00000000000000070b000000000000",
    "00636f6d7072657373696f6e0904000000000000000000000007050000000000000064747970650c",
    "07030000000000000046313600070c0000000000000064747970655f6c6179657273090100000000",
    "0000000902000000000000000303000000000000000c070300000000000000463332000705000000",
    "000000006b6e6f6273030000000000000000070e000000000000006c61796f75745f76657273696f",
    "6e030100000000000000070b00000000000000706167655f746f6b656e7303400000000000000007",
    "0600000000000000706172616d730b0000000000000000070400000000000000726f706509010000",
    "00000000000b040000000000000007040000000000000062617365060000000080841e4107070000",
    "0000000000666163746f72730a0200000000000000070600000000000000666163746f7206000000",
    "0000002040070f000000000000006c6f775f667265715f666163746f7206000000000000f03f0710",
    "000000000000006f726967696e616c5f636f6e746578740103002000000000000007070000000000",
    "00007363616c696e67010706000000000000006c6c616d613307060000000000000073746f726564",
    "0902000000000000000c0706000000000000004e6174697665000c0706000000000000004e617469",
    "76650007070000000000000077696e646f7773090400000000000000000103001000000000000000",
    "01030010000000000000",
);

#[test]
fn two_principals_share_the_span_and_part_after_it() {
    let s = settings();
    let h = s.hash_with(mix);
    let toks = tokens(64 * 5, 1);
    let keys = |scope: PrefixScope, mode| {
        PrefixKeyChain::new(mix, model(), h, &scope, mode, 64).push_tokens(&toks, &[])
    };
    // A breakpoint at token 150: pages 0 and 1 (tokens 0..128) are shared; page 2 straddles it.
    let a = keys(principal(1, Some(150)), PrefixScopeMode::Principal);
    let b = keys(principal(2, Some(150)), PrefixScopeMode::Principal);
    assert_eq!(a.len(), 5);
    assert_eq!(a[..2], b[..2], "inside the shared span");
    for i in 2..5 {
        assert_ne!(a[i], b[i], "page {i}, after the span");
    }
    // The local principal is a scope of its own.
    let local = keys(
        PrefixScope {
            principal: None,
            shared_until: Some(150),
        },
        PrefixScopeMode::Principal,
    );
    assert_eq!(local[..2], a[..2]);
    assert_ne!(local[2], a[2]);
    // No breakpoint: nothing shared, from the first page.
    let a0 = keys(principal(1, None), PrefixScopeMode::Principal);
    let b0 = keys(principal(2, None), PrefixScopeMode::Principal);
    assert_ne!(a0[0], b0[0]);
    // A breakpoint on a page boundary shares exactly the pages before it.
    let a2 = keys(principal(1, Some(128)), PrefixScopeMode::Principal);
    let b2 = keys(principal(2, Some(128)), PrefixScopeMode::Principal);
    assert_eq!(a2[..2], b2[..2]);
    assert_ne!(a2[2], b2[2]);
    // Instance scope: one cache for every client, and apart from principal scope's keys.
    let ia = keys(principal(1, None), PrefixScopeMode::Instance);
    let ib = keys(principal(2, Some(150)), PrefixScopeMode::Instance);
    assert_eq!(ia, ib);
    assert_ne!(ia[0], a[0]);
}

#[test]
fn a_different_image_parts_the_keys_from_its_first_page() {
    let s = settings();
    let h = s.hash_with(mix);
    // A placeholder run at tokens 100..230: pages 1, 2 and 3 hold some of it.
    let toks = tokens(64 * 5, 3);
    let image = |byte| MediaSpan {
        start: 100,
        len: 130,
        hash: Blake3([byte; 32]),
    };
    let keys = |media: &[MediaSpan]| {
        PrefixKeyChain::new(
            mix,
            model(),
            h,
            &PrefixScope::default(),
            PrefixScopeMode::Principal,
            64,
        )
        .push_tokens(&toks, media)
    };
    let plain = keys(&[]);
    let one = keys(&[image(1)]);
    let two = keys(&[image(2)]);
    assert_eq!(plain[0], one[0], "page 0 holds no placeholder");
    assert_eq!(one[0], two[0]);
    for i in 1..5 {
        assert_ne!(one[i], two[i], "page {i}");
        assert_ne!(one[i], plain[i], "page {i}");
    }
    // A span that ends where a page starts does not reach it; an empty span reaches nothing.
    let before = MediaSpan {
        start: 0,
        len: 64,
        hash: Blake3([5; 32]),
    };
    let empty = MediaSpan {
        start: 64,
        len: 0,
        hash: Blake3([5; 32]),
    };
    let edge = keys(&[before, empty]);
    assert_ne!(edge[0], plain[0]);
    let only_before = keys(&[before]);
    assert_eq!(edge[1..], only_before[1..]);
}

/// Media records go by absolute start, whatever order the caller lists the spans in, and each
/// is a fixed 40 bytes after the page's `4·P` token bytes.
#[test]
fn media_records_have_a_fixed_order_and_width() {
    let inputs = RefCell::new(Vec::<Vec<u8>>::new());
    let record = |bytes: &[u8]| {
        inputs.borrow_mut().push(bytes.to_vec());
        mix(bytes)
    };
    let a = MediaSpan {
        start: 2,
        len: 1,
        hash: Blake3([0xA; 32]),
    };
    let b = MediaSpan {
        start: 1,
        len: 1,
        hash: Blake3([0xB; 32]),
    };
    let page = [10, 11, 12, 13];
    let key = |spans: &[MediaSpan]| {
        let mut chain = PrefixKeyChain::new(
            &record,
            model(),
            Blake3([9; 32]),
            &PrefixScope::default(),
            PrefixScopeMode::Instance,
            4,
        );
        chain.push_page(&page, spans)
    };
    assert_eq!(key(&[a, b]), key(&[b, a]));
    let inputs = inputs.into_inner();
    let page_input = &inputs[1];
    // 32 bytes of k_root, no scope salt (instance scope), 16 bytes of tokens, two 40-byte records.
    assert_eq!(page_input.len(), 32 + 4 * 4 + 2 * 40);
    let records = &page_input[32 + 16..];
    assert_eq!(
        &records[..8],
        &[1, 0, 0, 0, 1, 0, 0, 0],
        "b, which starts first, comes first"
    );
    assert_eq!(&records[8..40], &[0xB; 32]);
    assert_eq!(&records[40..48], &[2, 0, 0, 0, 1, 0, 0, 0]);
}

/// A media span listed twice counts once (review N2 on #1942).
#[test]
fn a_media_span_listed_twice_counts_once() {
    let s = settings();
    let h = s.hash_with(mix);
    let toks = tokens(64, 6);
    let span = MediaSpan {
        start: 3,
        len: 10,
        hash: Blake3([4; 32]),
    };
    let key = |media: &[MediaSpan]| {
        PrefixKeyChain::new(
            mix,
            model(),
            h,
            &PrefixScope::default(),
            PrefixScopeMode::Principal,
            64,
        )
        .push_tokens(&toks, media)
    };
    assert_eq!(key(&[span, span]), key(&[span]));
    assert_ne!(key(&[span]), key(&[]));
}

#[test]
fn the_salt_constants_differ() {
    assert_ne!(LOCAL_SCOPE_SALT, INSTANCE_SHARED_SALT);
    // Both are ASCII text, unlike the hashed bytes of a principal's salt.
    assert!(
        LOCAL_SCOPE_SALT
            .iter()
            .chain(&INSTANCE_SHARED_SALT)
            .all(u8::is_ascii)
    );
    // The local principal and a principal whose salt is the instance constant are two scopes.
    let s = settings();
    let h = s.hash_with(mix);
    let toks = tokens(64, 5);
    let key = |scope: PrefixScope| {
        PrefixKeyChain::new(mix, model(), h, &scope, PrefixScopeMode::Principal, 64)
            .push_tokens(&toks, &[])
    };
    let local = key(PrefixScope {
        principal: None,
        shared_until: None,
    });
    let other = key(PrefixScope {
        principal: Some(INSTANCE_SHARED_SALT),
        shared_until: None,
    });
    assert_ne!(local, other);
}

#[test]
fn a_partial_page_has_no_key_and_the_chain_resumes() {
    let s = settings();
    let h = s.hash_with(mix);
    let toks = tokens(64 * 3 + 10, 4);
    let new = || {
        PrefixKeyChain::new(
            mix,
            model(),
            h,
            &PrefixScope::default(),
            PrefixScopeMode::Principal,
            64,
        )
    };
    let mut whole = new();
    let all = whole.push_tokens(&toks, &[]);
    assert_eq!(all.len(), 3);
    assert_eq!(whole.next_page(), 3);
    assert_eq!(whole.last(), all[2]);
    // The same keys page by page, as the head keys each page when it seals.
    let mut paged = new();
    for (i, page) in toks.as_chunks::<64>().0.iter().enumerate() {
        assert_eq!(paged.push_page(page, &[]), all[i]);
    }
}

#[test]
#[should_panic(expected = "only a full page has a key")]
fn a_short_page_is_refused() {
    let s = settings();
    let mut chain = PrefixKeyChain::new(
        mix,
        model(),
        s.hash_with(mix),
        &PrefixScope::default(),
        PrefixScopeMode::Principal,
        64,
    );
    chain.push_page(&[1, 2, 3], &[]);
}

proptest! {
    /// Identical inputs give identical keys: two chains built apart agree on every page.
    #[test]
    fn identical_inputs_give_identical_keys(
        toks in prop::collection::vec(any::<u32>(), 0..600),
        p in prop::sample::select(vec![1u32, 16, 64]),
        who in any::<Option<[u8; 16]>>(),
        until in any::<Option<u16>>(),
    ) {
        let mut s = settings();
        s.page_tokens = p;
        let scope = PrefixScope { principal: who, shared_until: until.map(u32::from) };
        let run = || {
            PrefixKeyChain::new(mix, model(), s.clone().hash_with(mix), &scope, PrefixScopeMode::Principal, p)
                .push_tokens(&toks, &[])
        };
        let first = run();
        prop_assert_eq!(first.len(), toks.len() / p as usize);
        prop_assert_eq!(first, run());
    }
}
