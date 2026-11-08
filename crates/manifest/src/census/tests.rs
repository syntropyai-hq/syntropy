// SPDX-License-Identifier: MIT OR Apache-2.0

use syntropy_core::DType;

use super::*;
use crate::hparams::HParamValue;
use crate::kv::LayerKv;
use crate::tokenizer::{SpecialTokens, TemplateSource};

fn group(id: u32, class: TensorClass, bytes: u64, read: u64) -> TensorGroup {
    TensorGroup {
        id: TensorGroupId(id),
        class,
        layer: None,
        tensors: vec![TensorId(id)],
        bytes: Bytes(bytes),
        read_per_token: Bytes(read),
        shared_with: None,
    }
}

fn manifest() -> ModelManifest {
    let extent = SourceExtent {
        file: 0,
        offset: 64,
        len: 128,
    };
    let record = TensorRecord {
        id: TensorId(0),
        name: "blk.0.attn_q.weight".into(),
        class: TensorClass::DenseEveryToken,
        layer: Some(LayerIx(0)),
        shape: vec![8, 8],
        quant: QuantScheme::Plain(DType::F16),
        bytes: Bytes(128),
        source: extent,
        aux: Vec::new(),
        hash: None,
    };
    let groups = vec![group(0, TensorClass::DenseEveryToken, 128, 128)];
    let specials = SpecialTokens {
        eos: Some(2),
        end_of_turn: vec![7, 2, 7],
        ..SpecialTokens::default()
    };
    ModelManifest {
        id: ModelId(Blake3([3; 32])),
        family: "llama".into(),
        format: SourceFormat::Gguf { version: 3 },
        hparams: HyperParams::new(1, 8, 16).with("rope_theta", HParamValue::F64(1e4)),
        census: TensorCensus {
            tensors: vec![record],
            totals: ClassTotals::of(&groups),
            groups,
        },
        kv: KvFormula::new(
            vec![LayerKv::PerToken {
                elems_per_token: 16,
                window: None,
                latent: false,
            }],
            DType::F16,
        ),
        tokenizer: Some(TokenizerSpec::new(
            b"{}".to_vec(),
            specials,
            true,
            false,
            Some("{{ x }}".into()),
            TemplateSource::Shipped,
        )),
        ignored: vec![("v.blk.0.attn_q.weight".into(), "multimodal".into())],
    }
}

/// A manifest stored as postcard plus zstd reads back equal, the tokenizer's ID included.
#[test]
fn a_stored_manifest_reads_back_equal() {
    let original = manifest();
    let stored = original.encode().unwrap();
    assert!(stored.starts_with(MANIFEST_TAG));
    let back = ModelManifest::decode(&stored).unwrap();
    assert_eq!(back, original);
    assert_eq!(
        back.tokenizer.as_ref().map(TokenizerSpec::id),
        original.tokenizer.as_ref().map(TokenizerSpec::id)
    );
}

/// The stored form is compressed: a manifest of many alike records is far smaller stored.
#[test]
fn the_stored_form_is_compressed() {
    let mut m = manifest();
    let record = m.census.tensors[0].clone();
    m.census.tensors = (0..2_000)
        .map(|i| TensorRecord {
            id: TensorId(i),
            ..record.clone()
        })
        .collect();
    let plain = postcard::to_stdvec(&m).unwrap();
    let stored = m.encode().unwrap();
    assert!(
        stored.len() * 4 < plain.len(),
        "{} of {}",
        stored.len(),
        plain.len()
    );
}

/// Anything but the stored form is refused, with code 6, never decoded on trust.
#[test]
fn a_damaged_manifest_is_refused() {
    let stored = manifest().encode().unwrap();
    assert!(ModelManifest::decode(&stored[1..]).is_err(), "no tag");
    let mut flipped = stored.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 0xff;
    assert!(ModelManifest::decode(&flipped).is_err(), "a damaged stream");
    let mut not_postcard = MANIFEST_TAG.to_vec();
    not_postcard.extend(zstd::bulk::compress(&[0xff; 7], 3).unwrap());
    let error = ModelManifest::decode(&not_postcard).unwrap_err();
    assert_eq!(error.code(), 6);
}

/// Totals sum the groups per class, in class order.
#[test]
fn totals_sum_groups_per_class() {
    let groups = [
        group(0, TensorClass::RowLookup, 100, 10),
        group(1, TensorClass::DenseEveryToken, 50, 50),
        group(2, TensorClass::DenseEveryToken, 25, 25),
        group(
            3,
            TensorClass::Expert {
                layer: LayerIx(0),
                id: ExpertIx(1),
            },
            40,
            5,
        ),
    ];
    let totals = ClassTotals::of(&groups);
    assert_eq!(
        totals.get(ClassKind::DenseEveryToken),
        ClassTotal {
            bytes: Bytes(75),
            read_per_token: Bytes(75)
        }
    );
    assert_eq!(totals.get(ClassKind::Expert).read_per_token, Bytes(5));
    assert_eq!(totals.get(ClassKind::Head), ClassTotal::default());
    let kinds: Vec<ClassKind> = totals.by_class.iter().map(|(k, _)| *k).collect();
    assert_eq!(
        kinds,
        [
            ClassKind::DenseEveryToken,
            ClassKind::Expert,
            ClassKind::RowLookup
        ]
    );
}

/// A stream that decompresses past the cap is refused, read as a stream rather than allocated
/// up front.
#[test]
fn a_manifest_past_the_cap_is_refused() {
    let big = vec![0u8; MAX_MANIFEST_BYTES + 1];
    let mut stored = MANIFEST_TAG.to_vec();
    stored.extend(zstd::bulk::compress(&big, 3).unwrap());
    let error = ModelManifest::decode(&stored).unwrap_err();
    assert_eq!(error.code(), 6);
    assert!(error.reason.contains("past"), "{error}");
}
