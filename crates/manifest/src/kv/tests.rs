// SPDX-License-Identifier: MIT OR Apache-2.0

//! `KvFormula`'s tests (TIER-E6-T01).
//!
//! - The cost model's KV-per-token figures (RFC-0022 §2's examples): Llama-3.3-70B 320 KiB and
//!   DeepSeek-V3 68.6 KiB, each derived from the model's published `config.json` values, read
//!   through [`HfConfig`].
//! - Every `LayerKv` kind, the auxiliary cache, the lossy dtypes, the serde defaults and the
//!   formula check.
//! - Properties: the bytes add up over a split of the layers, grow with the tokens, and a step
//!   never reads more than the sequence stores.

use proptest::prelude::*;

use super::*;
use crate::config::{ConfigFile, HfConfig};

const KIB: u64 = 1024;

/// A count from a `config.json` excerpt.
fn count(config: &HfConfig, key: &str) -> u32 {
    let value = config.u64(ConfigFile::Config, key).unwrap().unwrap();
    u32::try_from(value).unwrap()
}

/// The keys of `meta-llama/Llama-3.3-70B-Instruct`'s published `config.json` that size its KV.
/// The publisher's repository is gated; these values were read on 2026-10-28 from a public copy
/// of the file, `unsloth/Llama-3.3-70B-Instruct` at revision
/// `99cd0d2c829e92a67c844f9144c2509632e5c87f`.
const LLAMA_3_3_70B: &str = r#"{
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "hidden_size": 8192,
    "head_dim": 128,
    "torch_dtype": "bfloat16"
}"#;

/// The keys of `deepseek-ai/DeepSeek-V3`'s published `config.json` (revision
/// `e815299b0bcbac849fa540c768ef21845365c9eb`, read 2026-10-28) that size its KV.
const DEEPSEEK_V3: &str = r#"{
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "num_nextn_predict_layers": 1,
    "torch_dtype": "bfloat16"
}"#;

fn llama() -> KvFormula {
    let config = HfConfig::parse(LLAMA_3_3_70B.as_bytes(), None).unwrap();
    // GQA: each layer keeps K and V, each `n_kv_heads · head_dim` elements, per token.
    let layer = LayerKv::gqa(
        count(&config, "num_key_value_heads"),
        count(&config, "head_dim"),
        None,
    );
    KvFormula::new(
        vec![layer; count(&config, "num_hidden_layers") as usize],
        DType::BF16,
    )
}

fn deepseek() -> KvFormula {
    let config = HfConfig::parse(DEEPSEEK_V3.as_bytes(), None).unwrap();
    // MLA: each layer keeps one latent of `kv_lora_rank` elements and one decoupled RoPE key of
    // `qk_rope_head_dim` elements per token, shared by all 128 heads. The cost model's figure
    // counts the 61 decoder layers; the MTP module (`num_nextn_predict_layers`) is not one.
    let layer = LayerKv::mla(
        count(&config, "kv_lora_rank"),
        count(&config, "qk_rope_head_dim"),
    );
    KvFormula::new(
        vec![layer; count(&config, "num_hidden_layers") as usize],
        DType::BF16,
    )
}

fn all(formula: &KvFormula) -> Range<LayerIx> {
    LayerIx(0)..LayerIx(formula.layers.len() as u16)
}

/// Llama-3.3-70B: 80 layers × 2 (K, V) × 8 KV heads × 128 × 2 bytes (BF16) = 327,680 bytes
/// = 320 KiB per token.
#[test]
fn llama_3_3_70b_keeps_320_kib_per_token() {
    let f = llama();
    assert_eq!(f.layers.len(), 80);
    assert_eq!(f.layers[0], LayerKv::gqa(8, 128, None));
    assert_eq!(f.stored(all(&f), 1, KvDType::Native), Bytes(320 * KIB));
    // A step at context 1 reads the one token; at 8192 tokens it reads them all.
    assert_eq!(
        f.read_per_step(all(&f), 1, KvDType::Native),
        Bytes(320 * KIB)
    );
    assert_eq!(
        f.stored(all(&f), 8192, KvDType::Native),
        Bytes(8192 * 320 * KIB)
    );
    assert_eq!(
        f.read_per_step(all(&f), 8192, KvDType::Native),
        Bytes(8192 * 320 * KIB)
    );
}

/// DeepSeek-V3: 61 layers × (512 + 64) elements × 2 bytes (BF16) = 70,272 bytes = 68.625 KiB,
/// the cost model's 68.6 KiB per token.
#[test]
fn deepseek_v3_keeps_68_6_kib_per_token() {
    let f = deepseek();
    assert_eq!(f.layers.len(), 61);
    let per_token = f.stored(all(&f), 1, KvDType::Native);
    assert_eq!(per_token, Bytes(70_272));
    assert_eq!(format!("{:.1}", per_token.0 as f64 / KIB as f64), "68.6");
    assert_eq!(f.read_per_step(all(&f), 1, KvDType::Native), per_token);
}

/// A sliding-window layer keeps and reads at most its window; a recurrent layer its state,
/// once, from the first token on.
#[test]
fn windows_cap_tokens_and_state_is_counted_once() {
    let f = KvFormula::new(
        vec![
            LayerKv::gqa(8, 64, Some(128)),
            LayerKv::Fixed {
                elems_per_seq: 1 << 20,
            },
        ],
        DType::BF16,
    );
    let window = LayerIx(0)..LayerIx(1);
    let state = LayerIx(1)..LayerIx(2);
    let per_token = 2 * 8 * 64 * 2;
    assert_eq!(
        f.stored(window.clone(), 100, KvDType::Native),
        Bytes(100 * per_token)
    );
    assert_eq!(
        f.stored(window.clone(), 5000, KvDType::Native),
        Bytes(128 * per_token)
    );
    assert_eq!(
        f.read_per_step(window, 5000, KvDType::Native),
        Bytes(128 * per_token)
    );
    for tokens in [0, 1, 1_000_000] {
        assert_eq!(
            f.stored(state.clone(), tokens, KvDType::Native),
            Bytes(2 << 20)
        );
        assert_eq!(
            f.read_per_step(state.clone(), tokens, KvDType::Native),
            Bytes(2 << 20)
        );
    }
}

/// A sparse layer stores every token and reads `selected` of them, plus its indexer's keys over
/// the whole context, in the indexer's own dtype.
#[test]
fn a_sparse_layer_reads_its_selection_and_the_indexer_keys() {
    let mut f = KvFormula::new(
        vec![
            LayerKv::Sparse {
                elems_per_token: 576,
                selected: 2048,
                latent: true,
            },
            LayerKv::mla(512, 64),
        ],
        DType::BF16,
    );
    f.aux.push(AuxKv {
        layer: LayerIx(0),
        kind: AuxKind::SparseIndexer,
        elems_per_token: 128,
        dtype: DType::F8E4M3,
    });
    let sparse = LayerIx(0)..LayerIx(1);
    let context = 100_000;
    let indexer = context * 128;
    assert_eq!(
        f.stored(sparse.clone(), context, KvDType::Native),
        Bytes(context * 576 * 2 + indexer)
    );
    assert_eq!(
        f.read_per_step(sparse.clone(), context, KvDType::Native),
        Bytes(2048 * 576 * 2 + indexer)
    );
    // Short contexts read what there is.
    assert_eq!(
        f.read_per_step(sparse, 10, KvDType::Native),
        Bytes(10 * 576 * 2 + 10 * 128)
    );
    // The indexer belongs to layer 0 only.
    let other = LayerIx(1)..LayerIx(2);
    assert_eq!(f.stored(other, 10, KvDType::Native), Bytes(10 * 576 * 2));
}

/// A compressed layer forms one entry per `r` tokens, when the entry's last token is
/// processed (OQ-MF-12), keeps a raw window beside them, and reads all formed entries or its
/// selection.
#[test]
fn a_compressed_layer_counts_formed_entries_and_its_window() {
    let f = KvFormula::new(
        vec![
            LayerKv::Compressed {
                elems_per_entry: 512,
                tokens_per_entry: 4,
                window: Some(128),
                selected: Some(1024),
                latent: true,
            },
            LayerKv::Compressed {
                elems_per_entry: 512,
                tokens_per_entry: 128,
                window: Some(128),
                selected: None,
                latent: true,
            },
        ],
        DType::BF16,
    );
    let entry = 512 * 2;
    let r4 = LayerIx(0)..LayerIx(1);
    let r128 = LayerIx(1)..LayerIx(2);
    // 3 tokens: no entry formed yet, 3 raw.
    assert_eq!(f.stored(r4.clone(), 3, KvDType::Native), Bytes(3 * entry));
    // 65,536 tokens: 16,384 entries at r = 4, 512 at r = 128, each with a raw window of 128.
    let n = 65_536;
    assert_eq!(
        f.stored(r4.clone(), n, KvDType::Native),
        Bytes((16_384 + 128) * entry)
    );
    assert_eq!(
        f.stored(r128.clone(), n, KvDType::Native),
        Bytes((512 + 128) * entry)
    );
    // r = 4 reads its 1,024 selected entries; r = 128 reads all 512.
    assert_eq!(
        f.read_per_step(r4, n, KvDType::Native),
        Bytes((1024 + 128) * entry)
    );
    assert_eq!(
        f.read_per_step(r128, n, KvDType::Native),
        Bytes((512 + 128) * entry)
    );
}

/// RFC-0022 §9's lossy forms against BF16: `Q8 { 32 }` ≈ 0.53×, `K8V4 { 32 }` ≈ 0.40×.
/// Recurrent state and indexer keys keep their dtype.
#[test]
fn lossy_dtypes_shrink_entries_but_not_state_or_indexer_keys() {
    let f = llama();
    let native = f.stored(all(&f), 1, KvDType::Native).0;
    let q8 = f.stored(all(&f), 1, KvDType::Q8 { block: 32 }).0;
    let k8v4 = f
        .stored(
            all(&f),
            1,
            KvDType::K8V4 {
                block: 32,
                rotate: true,
            },
        )
        .0;
    // Per layer, 2,048 elements: Q8 is 2,048 bytes and 64 F16 scales = 2,176 bytes; K8V4 is
    // 1,024 + 64 bytes of K and 512 + 64 bytes of V = 1,664 bytes; BF16 is 4,096 bytes.
    assert_eq!(q8, 80 * 2176);
    assert_eq!(k8v4, 80 * 1664);
    assert_eq!(format!("{:.2}", q8 as f64 / native as f64), "0.53");
    assert_eq!(format!("{:.2}", k8v4 as f64 / native as f64), "0.41");

    let mut state = KvFormula::new(
        vec![LayerKv::Fixed {
            elems_per_seq: 1000,
        }],
        DType::F32,
    );
    state.aux.push(AuxKv {
        layer: LayerIx(0),
        kind: AuxKind::SparseIndexer,
        elems_per_token: 64,
        dtype: DType::BF16,
    });
    let one = LayerIx(0)..LayerIx(1);
    assert_eq!(
        state.stored(one.clone(), 10, KvDType::Q8 { block: 32 }),
        state.stored(one, 10, KvDType::Native)
    );
}

/// Per-layer dtype exceptions, sub-byte widths rounded up per entry, and a range past the last
/// layer clamped.
#[test]
fn per_layer_dtypes_and_ranges() {
    let mut f = KvFormula::new(vec![LayerKv::gqa(1, 3, None); 4], DType::BF16);
    f.dtype.layers.push((LayerIx(2), DType::F32));
    f.dtype.layers.push((LayerIx(3), DType::F4E2M1));
    assert_eq!(f.dtype.of(LayerIx(0)), DType::BF16);
    assert_eq!(f.dtype.of(LayerIx(2)), DType::F32);
    // 6 elements: 12 bytes in BF16, 24 in F32, 3 in FP4.
    assert_eq!(
        f.stored(LayerIx(0)..LayerIx(4), 1, KvDType::Native),
        Bytes(12 + 12 + 24 + 3)
    );
    assert_eq!(
        f.stored(LayerIx(2)..LayerIx(99), 2, KvDType::Native),
        Bytes(2 * (24 + 3))
    );
    assert_eq!(
        f.stored(LayerIx(9)..LayerIx(99), 2, KvDType::Native),
        Bytes(0)
    );
    assert_eq!(
        f.stored(LayerIx(3)..LayerIx(1), 2, KvDType::Native),
        Bytes(0)
    );
}

/// A record written before `aux`, `dtype` and `layout_version` existed reads with their
/// defaults (RFC-0022 §2).
#[test]
fn the_added_fields_have_serde_defaults() {
    let old = r#"{"layers":[{"PerToken":{"elems_per_token":2048,"window":null}},{"Fixed":{"elems_per_seq":7}},{"Sparse":{"elems_per_token":10,"selected":2}}]}"#;
    let f: KvFormula = serde_json::from_str(old).unwrap();
    assert_eq!(
        f.layers,
        vec![
            LayerKv::gqa(8, 128, None),
            LayerKv::Fixed { elems_per_seq: 7 },
            LayerKv::Sparse {
                elems_per_token: 10,
                selected: 2,
                latent: false,
            },
        ]
    );
    assert!(f.aux.is_empty());
    assert_eq!(f.dtype, KvDTypeSpec::uniform(DType::BF16));
    assert_eq!(f.layout_version, 0);
    let mut new = deepseek();
    new.layout_version = 3;
    new.dtype.layers.push((LayerIx(5), DType::F16));
    let back: KvFormula = serde_json::from_str(&serde_json::to_string(&new).unwrap()).unwrap();
    assert_eq!(back, new);
}

#[test]
fn the_check_refuses_an_empty_ratio_and_unknown_layers() {
    assert_eq!(llama().check(), Ok(()));
    let mut f = KvFormula::new(
        vec![
            LayerKv::mla(512, 64),
            LayerKv::Compressed {
                elems_per_entry: 512,
                tokens_per_entry: 0,
                window: Some(128),
                selected: None,
                latent: false,
            },
        ],
        DType::BF16,
    );
    assert_eq!(
        f.check(),
        Err(KvFormulaError::NoTokensPerEntry { layer: LayerIx(1) })
    );
    // Unchecked, the empty ratio forms no entries rather than dividing by zero.
    assert_eq!(
        f.stored(LayerIx(1)..LayerIx(2), 1000, KvDType::Native),
        Bytes(128 * 1024)
    );
    f.layers.pop();
    f.aux.push(AuxKv {
        layer: LayerIx(1),
        kind: AuxKind::SparseIndexer,
        elems_per_token: 1,
        dtype: DType::BF16,
    });
    assert_eq!(
        f.check(),
        Err(KvFormulaError::NoSuchLayer { layer: LayerIx(1) })
    );
    f.aux.clear();
    f.dtype.layers.push((LayerIx(4), DType::F16));
    assert_eq!(
        f.check(),
        Err(KvFormulaError::NoSuchLayer { layer: LayerIx(4) })
    );
}

/// `K8V4` sizes a latent, which is no K/V pair, as `Q8` (OQ-IO-30's default): an MLA layer, and
/// `Sparse` and `Compressed` layers whose family marks them latent (DSA- and V4-class).
#[test]
fn k8v4_sizes_a_latent_as_q8() {
    let f = deepseek();
    let k8v4 = KvDType::K8V4 {
        block: 32,
        rotate: true,
    };
    let q8 = KvDType::Q8 { block: 32 };
    // 576 elements: 576 bytes and 18 F16 scales = 612 bytes per layer, against 1,152 in BF16.
    assert_eq!(f.stored(all(&f), 1, k8v4), Bytes(61 * 612));
    assert_eq!(f.stored(all(&f), 1, k8v4), f.stored(all(&f), 1, q8));
    let other = KvFormula::new(
        vec![
            LayerKv::Sparse {
                elems_per_token: 576,
                selected: 2048,
                latent: true,
            },
            LayerKv::Compressed {
                elems_per_entry: 512,
                tokens_per_entry: 4,
                window: Some(128),
                selected: None,
                latent: true,
            },
        ],
        DType::BF16,
    );
    assert_eq!(
        other.stored(all(&other), 1000, k8v4),
        other.stored(all(&other), 1000, q8)
    );
    // A K/V pair still splits: GQA is smaller under K8V4 than under Q8.
    let g = llama();
    assert!(g.stored(all(&g), 1, k8v4) < g.stored(all(&g), 1, q8));
}

/// A `Sparse` or `Compressed` layer that stores K/V pairs, as block-sparse and QSA-class layers
/// do "as `Gqa`" (RFC-0043 §3.1), splits under `K8V4` as GQA does; only a latent is `Q8`.
#[test]
fn k8v4_splits_sparse_and_compressed_pairs() {
    let k8v4 = KvDType::K8V4 {
        block: 32,
        rotate: true,
    };
    let q8 = KvDType::Q8 { block: 32 };
    // Block-sparse GQA: 2 × 8 KV heads × 128 = 2,048 elements per token. K8V4: 1,024 + 64 bytes
    // of K and 512 + 64 bytes of V = 1,664 bytes.
    let sparse = |latent| LayerKv::Sparse {
        elems_per_token: 2048,
        selected: 2048,
        latent,
    };
    let compressed = |latent| LayerKv::Compressed {
        elems_per_entry: 2048,
        tokens_per_entry: 4,
        window: Some(128),
        selected: None,
        latent,
    };
    for (pair, latent) in [
        (sparse(false), sparse(true)),
        (compressed(false), compressed(true)),
    ] {
        let pair = KvFormula::new(vec![pair], DType::BF16);
        let latent = KvFormula::new(vec![latent], DType::BF16);
        let one = LayerIx(0)..LayerIx(1);
        // 1,000 tokens: 1,000 sparse entries, or 250 compressed entries plus 128 raw.
        let entries = pair.stored(one.clone(), 1000, KvDType::Native).0 / 4096;
        assert_eq!(pair.stored(one.clone(), 1000, k8v4), Bytes(entries * 1664));
        assert_eq!(
            latent.stored(one.clone(), 1000, k8v4),
            latent.stored(one.clone(), 1000, q8)
        );
        assert_eq!(latent.stored(one.clone(), 1000, q8), Bytes(entries * 2176));
        // A step at 1,000 tokens reads the same entries: all 1,000 sparse ones (2,048 may be
        // selected), or every compressed entry plus the window. Read and store follow one rule.
        let read = pair.read_per_step(one.clone(), 1000, KvDType::Native).0 / 4096;
        assert_eq!(read, entries);
        assert_eq!(
            pair.read_per_step(one.clone(), 1000, k8v4),
            Bytes(read * 1664)
        );
        assert_eq!(
            latent.read_per_step(one.clone(), 1000, k8v4),
            latent.read_per_step(one.clone(), 1000, q8)
        );
        assert_eq!(latent.read_per_step(one, 1000, q8), Bytes(read * 2176));
    }
}

/// Products are taken in `u128` and saturate at `u64::MAX` bytes; nothing wraps.
#[test]
fn extreme_sizes_saturate_rather_than_wrap() {
    let wide = LayerKv::PerToken {
        elems_per_token: u32::MAX,
        window: None,
        latent: false,
    };
    let f = KvFormula::new(vec![wide; 1 << 16], DType::F64);
    let every = LayerIx(0)..LayerIx(u16::MAX);
    for dtype in [KvDType::Native, KvDType::Q8 { block: 1 }] {
        assert_eq!(f.stored(every.clone(), u64::MAX, dtype), Bytes(u64::MAX));
        assert_eq!(
            f.read_per_step(every.clone(), u64::MAX, dtype),
            Bytes(u64::MAX)
        );
    }
    // At the edge: one layer of u32::MAX F64 elements is 34,359,738,360 bytes per token, so
    // 2^29 tokens need 2^64 − 2^32 bytes, which fits, and 2^30 tokens do not.
    let one = LayerIx(0)..LayerIx(1);
    let per_token = u128::from(u32::MAX) * 8;
    assert_eq!(
        u128::from(f.stored(one.clone(), 1 << 29, KvDType::Native).0),
        per_token << 29
    );
    assert_eq!(f.stored(one, 1 << 30, KvDType::Native), Bytes(u64::MAX));
    let state = KvFormula::new(
        vec![LayerKv::Fixed {
            elems_per_seq: u64::MAX,
        }],
        DType::F64,
    );
    assert_eq!(
        state.stored(LayerIx(0)..LayerIx(1), 0, KvDType::Native),
        Bytes(u64::MAX)
    );
}

fn layer_kv() -> impl Strategy<Value = LayerKv> {
    prop_oneof![
        (1u32..4096, prop::option::of(1u32..512), any::<bool>()).prop_map(
            |(elems_per_token, window, latent)| LayerKv::PerToken {
                elems_per_token,
                window,
                latent,
            }
        ),
        (0u64..1 << 24).prop_map(|elems_per_seq| LayerKv::Fixed { elems_per_seq }),
        (1u32..4096, 0u32..4096, any::<bool>()).prop_map(|(elems_per_token, selected, latent)| {
            LayerKv::Sparse {
                elems_per_token,
                selected,
                latent,
            }
        }),
        (
            1u32..4096,
            1u16..256,
            prop::option::of(0u32..512),
            prop::option::of(0u32..4096),
            any::<bool>(),
        )
            .prop_map(
                |(elems_per_entry, tokens_per_entry, window, selected, latent)| {
                    LayerKv::Compressed {
                        elems_per_entry,
                        tokens_per_entry,
                        window,
                        selected,
                        latent,
                    }
                }
            ),
    ]
}

fn kv_dtype() -> impl Strategy<Value = KvDType> {
    prop_oneof![
        Just(KvDType::Native),
        (1u16..128).prop_map(|block| KvDType::Q8 { block }),
        (1u16..128, any::<bool>()).prop_map(|(block, rotate)| KvDType::K8V4 { block, rotate }),
    ]
}

fn formula() -> impl Strategy<Value = KvFormula> {
    (
        prop::collection::vec(layer_kv(), 1..12),
        prop_oneof![
            Just(DType::BF16),
            Just(DType::F16),
            Just(DType::F32),
            Just(DType::F8E4M3)
        ],
        prop::collection::vec((0u16..12, 1u32..256), 0..3),
    )
        .prop_map(|(layers, dtype, aux)| {
            let n = layers.len() as u16;
            let mut f = KvFormula::new(layers, dtype);
            f.aux = aux
                .into_iter()
                .map(|(layer, elems_per_token)| AuxKv {
                    layer: LayerIx(layer % n),
                    kind: AuxKind::SparseIndexer,
                    elems_per_token,
                    dtype: DType::BF16,
                })
                .collect();
            f
        })
}

/// `KvFormula::stored` or `KvFormula::read_per_step`.
type Size = fn(&KvFormula, Range<LayerIx>, u64, KvDType) -> Bytes;

proptest! {
    /// Splitting the layers splits the bytes; more tokens never store less; a step never reads
    /// more than the sequence stores.
    #[test]
    fn bytes_add_up_grow_and_bound_the_read(
        f in formula(),
        cut in 0u16..12,
        tokens in 0u64..1 << 20,
        more in 0u64..1 << 20,
        dtype in kv_dtype(),
    ) {
        prop_assert_eq!(f.check(), Ok(()));
        let n = f.layers.len() as u16;
        let cut = LayerIx(cut % (n + 1));
        let whole = LayerIx(0)..LayerIx(n);
        let both: [Size; 2] =
            [KvFormula::stored, KvFormula::read_per_step];
        for bytes in both {
            prop_assert_eq!(
                bytes(&f, whole.clone(), tokens, dtype).0,
                bytes(&f, LayerIx(0)..cut, tokens, dtype).0 + bytes(&f, cut..LayerIx(n), tokens, dtype).0
            );
        }
        prop_assert!(f.stored(whole.clone(), tokens + more, dtype) >= f.stored(whole.clone(), tokens, dtype));
        prop_assert!(f.read_per_step(whole.clone(), tokens, dtype) <= f.stored(whole, tokens, dtype));
    }
}
