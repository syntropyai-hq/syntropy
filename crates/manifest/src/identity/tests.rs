// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of `FileDigest` and the v1 `ModelId` (COMP-E10-T06; RFC-0041 §1).
//!
//! The synthetic GGUF files come from [`crate::gguf::synth`]. The golden IDs are those of the
//! checked-in fixtures under `tests/data/`, in `tests/golden/model_ids.txt`. A change that moves
//! one fails `golden_ids_of_the_fixtures_do_not_change`; a deliberate change to the formula is a
//! new formula version (RFC-0041 §1), not a re-bless. Regenerate the file only for a new fixture,
//! with `SYNTROPY_BLESS=1 cargo test -p syntropy-manifest identity`.

use std::path::{Path, PathBuf};

use proptest::prelude::*;
use syntropy_core::GgmlType;

use super::*;
use crate::FileSource;
use crate::gguf::Value;
use crate::gguf::synth::Synth;
use crate::hparams::HyperParams;

/// A byte buffer as a [`RangeSource`].
struct Mem(Vec<u8>);

impl RangeSource for Mem {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        crate::range::check_bounds(offset, buf.len() as u64, self.len())?;
        let at = offset as usize;
        buf.copy_from_slice(&self.0[at..at + buf.len()]);
        Ok(())
    }
}

fn hex(id: &ModelId) -> String {
    id.0.0.iter().map(|b| format!("{b:02x}")).collect()
}

fn weights(bytes: &[u8]) -> SourceFile {
    SourceFile {
        role: SourceRole::Weights,
        digest: FileDigest::of_bytes(bytes),
        len: bytes.len() as u64,
    }
}

fn gguf_id(tag: &str, bytes: &[u8]) -> ModelId {
    model_id(tag, &[weights(bytes)]).unwrap()
}

#[test]
fn sha256_matches_the_published_vectors() {
    // FIPS 180-2, appendix B.1, and the empty message.
    assert_eq!(
        FileDigest::of_bytes(b"abc").to_hex(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        FileDigest::of_bytes(b"").to_hex(),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn a_streamed_digest_equals_the_whole_buffer_s() {
    // Longer than one read, and not a multiple of it.
    let bytes: Vec<u8> = (0..(READ_SIZE * 2 + 12_345))
        .map(|i| (i % 251) as u8)
        .collect();
    assert_eq!(
        FileDigest::of_source(&Mem(bytes.clone())).unwrap(),
        FileDigest::of_bytes(&bytes)
    );
    assert_eq!(
        FileDigest::of_source(&Mem(Vec::new())).unwrap(),
        FileDigest::of_bytes(b"")
    );
}

#[test]
fn a_hub_object_id_parses_to_the_same_digest() {
    let local = FileDigest::of_bytes(b"abc");
    let listed = "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD";
    assert_eq!(FileDigest::from_hex(listed), Some(local));
    assert_eq!(FileDigest::from_hex(&local.to_hex()), Some(local));
    let mixed = format!("{}{}", &listed[..32], &local.to_hex()[32..]);
    assert_eq!(FileDigest::from_hex(&mixed), Some(local));
}

#[test]
fn from_hex_takes_exactly_64_hex_digits() {
    let good = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert!(FileDigest::from_hex(good).is_some());
    let refused = [
        // `from_str_radix` would take a sign on a pair.
        format!("+{}", &good[1..]),
        format!("{}+{}", &good[..10], &good[11..]),
        format!("-{}", &good[1..]),
        // Whitespace, inside or around.
        format!(" {}", &good[1..]),
        format!("{} ", &good[..63]),
        format!("{}\n", &good[..63]),
        format!("{}\t{}", &good[..20], &good[21..]),
        format!(" {good}"),
        format!("{good}\n"),
        // Wrong lengths: 63, 65, empty, and the Hub's 40-digit SHA-1 of a Git-stored file.
        good[..63].to_owned(),
        format!("{good}0"),
        String::new(),
        good[..40].to_owned(),
        // Non-hex, and a prefix.
        format!("{}zz", &good[..62]),
        format!("{}g", &good[..63]),
        format!("0x{}", &good[2..]),
        // A multi-byte character that keeps the byte length at 64.
        format!("{}é", &good[..62]),
    ];
    for text in &refused {
        assert_eq!(FileDigest::from_hex(text), None, "{text:?}");
    }
}

#[test]
fn renaming_a_local_file_keeps_its_id() {
    let dir = std::env::temp_dir().join(format!("syntropy-identity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = sample().build();
    let first = dir.join("model.gguf");
    std::fs::write(&first, &bytes).unwrap();
    let before = FileDigest::of_source(&FileSource::open(&first).unwrap()).unwrap();
    let second = dir.join("renamed-Q4_K_M.gguf");
    std::fs::rename(&first, &second).unwrap();
    let after = FileDigest::of_source(&FileSource::open(&second).unwrap()).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(before, after);
    assert_eq!(before, FileDigest::of_bytes(&bytes));
    let len = bytes.len() as u64;
    let id = |digest| {
        model_id(
            "llama",
            &[SourceFile {
                role: SourceRole::Weights,
                digest,
                len,
            }],
        )
        .unwrap()
    };
    assert_eq!(id(before), id(after));
}

#[test]
fn the_formula_writes_the_documented_bytes() {
    let a = weights(b"weights");
    let config = SourceFile {
        role: SourceRole::Config,
        digest: FileDigest::of_bytes(b"{}"),
        len: 2,
    };
    let mut expected = Vec::new();
    expected.extend_from_slice(b"syntropy/model-id/v1");
    expected.extend_from_slice(&5u64.to_le_bytes());
    expected.extend_from_slice(b"llama");
    expected.push(1);
    expected.extend_from_slice(&a.digest.0);
    expected.extend_from_slice(&7u64.to_le_bytes());
    expected.push(3);
    expected.extend_from_slice(&config.digest.0);
    expected.extend_from_slice(&2u64.to_le_bytes());
    let id = model_id("llama", &[config, a]).unwrap();
    assert_eq!(id.0.0, *blake3::hash(&expected).as_bytes());
}

#[test]
fn files_outside_the_formula_s_roles_are_not_hashed() {
    let w = weights(b"weights");
    let base = model_id("qwen_moe", &[w]).unwrap();
    for role in [
        SourceRole::Tokenizer,
        SourceRole::TokenizerConfig,
        SourceRole::SpecialTokensMap,
    ] {
        let extra = SourceFile {
            role,
            digest: FileDigest::of_bytes(b"tokenizer"),
            len: 9,
        };
        assert!(!role.in_model_id());
        assert_eq!(model_id("qwen_moe", &[w, extra]).unwrap(), base, "{role:?}");
    }
    for role in [
        SourceRole::WeightIndex,
        SourceRole::Config,
        SourceRole::GenerationConfig,
    ] {
        let extra = SourceFile {
            role,
            digest: FileDigest::of_bytes(b"{}"),
            len: 2,
        };
        assert!(role.in_model_id());
        assert_ne!(model_id("qwen_moe", &[w, extra]).unwrap(), base, "{role:?}");
    }
}

/// Every role, in code order. `every_role_is_listed` fails to compile when a role is added and not
/// listed here.
const ALL_ROLES: [SourceRole; 7] = [
    SourceRole::Weights,
    SourceRole::WeightIndex,
    SourceRole::Config,
    SourceRole::GenerationConfig,
    SourceRole::Tokenizer,
    SourceRole::TokenizerConfig,
    SourceRole::SpecialTokensMap,
];

#[test]
fn every_role_is_listed() {
    fn listed(role: SourceRole) -> bool {
        match role {
            SourceRole::Weights
            | SourceRole::WeightIndex
            | SourceRole::Config
            | SourceRole::GenerationConfig
            | SourceRole::Tokenizer
            | SourceRole::TokenizerConfig
            | SourceRole::SpecialTokensMap => ALL_ROLES.contains(&role),
        }
    }
    assert!(ALL_ROLES.into_iter().all(listed));
}

#[test]
fn role_codes_and_in_model_id_agree_and_codes_are_fixed() {
    let codes: Vec<(SourceRole, Option<u8>)> =
        ALL_ROLES.into_iter().map(|r| (r, role_code(r))).collect();
    for (role, code) in &codes {
        assert_eq!(role.in_model_id(), code.is_some(), "{role:?}");
    }
    // The v1 codes; changing one changes every ModelId.
    assert_eq!(
        codes,
        [
            (SourceRole::Weights, Some(1)),
            (SourceRole::WeightIndex, Some(2)),
            (SourceRole::Config, Some(3)),
            (SourceRole::GenerationConfig, Some(4)),
            (SourceRole::Tokenizer, None),
            (SourceRole::TokenizerConfig, None),
            (SourceRole::SpecialTokensMap, None),
        ]
    );
}

#[test]
fn an_id_needs_a_tag_and_a_weights_file() {
    let w = weights(b"weights");
    assert_eq!(model_id("", &[w]), Err(IdentityError::NoFamilyTag));
    assert_eq!(model_id("llama", &[]), Err(IdentityError::NoWeights));
    let config = SourceFile {
        role: SourceRole::Config,
        digest: FileDigest::of_bytes(b"{}"),
        len: 2,
    };
    assert_eq!(model_id("llama", &[config]), Err(IdentityError::NoWeights));
}

/// The same file under a patched `HyperParams` that gained a field: what the census derives is
/// no input, so the ID of the same files is the same whichever type a release derives.
#[test]
fn adding_a_field_to_hyperparams_does_not_change_the_id() {
    /// `HyperParams` as a later release might extend it.
    #[derive(Clone, Debug, PartialEq)]
    struct PatchedHyperParams {
        base: HyperParams,
        sliding_window: u32,
    }

    /// What a census holds after reading a model: its derived values and the files it read.
    struct Read<H> {
        _hparams: H,
        files: Vec<SourceFile>,
    }

    fn id_of<H>(read: &Read<H>) -> ModelId {
        model_id("llama", &read.files).unwrap()
    }

    let bytes = sample().build();
    let files = vec![weights(&bytes)];
    let today = Read {
        _hparams: HyperParams::new(2, 64, 32),
        files: files.clone(),
    };
    let patched = Read {
        _hparams: PatchedHyperParams {
            base: HyperParams::new(2, 64, 32),
            sliding_window: 4096,
        },
        files,
    };
    assert_eq!(id_of(&today), id_of(&patched));
    assert_eq!(id_of(&today), gguf_id("llama", &bytes));
}

fn sample() -> Synth {
    Synth::new(3)
        .kv(
            "general.architecture",
            Value::Str(b"llama".as_slice().into()),
        )
        .kv("llama.block_count", Value::U32(2))
        .kv("llama.embedding_length", Value::U32(64))
        .tensor("token_embd.weight", &[64, 32], GgmlType::F16)
        .tensor("blk.0.attn_q.weight", &[64, 64], GgmlType::Q8_0)
}

prop_compose! {
    fn synth_strategy()(
        blocks in 1u32..=64,
        width in 1u64..=4,
        n_tensors in 1usize..=3,
        ty in prop::sample::select(vec![GgmlType::F32, GgmlType::F16, GgmlType::Q8_0, GgmlType::Q4_0]),
    ) -> Synth {
        let mut s = Synth::new(3)
            .kv("general.architecture", Value::Str(b"llama".as_slice().into()))
            .kv("llama.block_count", Value::U32(blocks));
        for t in 0..n_tensors {
            s = s.tensor(&format!("blk.{t}.ffn_up.weight"), &[32 * width, 2], ty);
        }
        s
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Any one tensor byte flipped gives another ID.
    #[test]
    fn changing_a_tensor_byte_changes_the_id(
        s in synth_strategy(),
        pick in any::<prop::sample::Index>(),
        flip in 1u8..=255,
    ) {
        let bytes = s.build();
        let data_start = (s.header().len() as u64).next_multiple_of(s.alignment());
        let (offset, len) = s.layout()[pick.index(s.tensors.len())];
        let at = usize::try_from(data_start + offset).unwrap() + pick.index(len as usize);
        let mut changed = bytes.clone();
        changed[at] ^= flip;
        prop_assert_ne!(gguf_id("llama", &bytes), gguf_id("llama", &changed));
    }

    /// A different value for a hyperparameter key gives another ID.
    #[test]
    fn changing_a_hyperparameter_changes_the_id(s in synth_strategy(), delta in 1u32..1000) {
        let bytes = s.build();
        let mut other = s.clone();
        for (k, v) in &mut other.kvs {
            if k == "llama.block_count"
                && let Value::U32(n) = v
            {
                *n = n.wrapping_add(delta);
            }
        }
        prop_assert_ne!(gguf_id("llama", &bytes), gguf_id("llama", &other.build()));
    }

    /// The order the files are given in does not matter; the family tag does.
    #[test]
    fn file_order_does_not_matter_and_the_tag_does(
        a in synth_strategy(),
        b in synth_strategy(),
        tags in prop::sample::subsequence(vec!["llama", "gemma", "qwen_moe", "glm", "bert"], 2),
    ) {
        let (fa, fb) = (weights(&a.build()), weights(&b.build()));
        let config = SourceFile { role: SourceRole::Config, digest: FileDigest::of_bytes(b"{}"), len: 2 };
        prop_assert_eq!(
            model_id(tags[0], &[fa, fb, config]).unwrap(),
            model_id(tags[0], &[config, fb, fa]).unwrap()
        );
        prop_assert_ne!(
            model_id(tags[0], &[fa, fb]).unwrap(),
            model_id(tags[1], &[fa, fb]).unwrap()
        );
    }
}

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn digest_of(name: &str) -> SourceFile {
    let path = data(name);
    let source = FileSource::open(&path).unwrap();
    let len = source.len();
    let digest = FileDigest::of_source(&source).unwrap();
    let role = match name.rsplit('/').next().unwrap() {
        "config.json" => SourceRole::Config,
        "model.safetensors.index.json" => SourceRole::WeightIndex,
        _ => SourceRole::Weights,
    };
    SourceFile { role, digest, len }
}

/// The fixture models: a name, a family tag, and the files the census would read.
const MODELS: &[(&str, &str, &[&str])] = &[
    ("gguf/tiny-v3", "llama", &["gguf/tiny-v3.gguf"]),
    (
        "gguf/tiny-split",
        "llama",
        &[
            "gguf/tiny-split-00001-of-00002.gguf",
            "gguf/tiny-split-00002-of-00002.gguf",
        ],
    ),
    (
        "safetensors/tiny",
        "llama",
        &["safetensors/tiny.safetensors", "safetensors/config.json"],
    ),
    (
        "safetensors/sharded",
        "llama",
        &[
            "safetensors/model-00001-of-00002.safetensors",
            "safetensors/model-00002-of-00002.safetensors",
            "safetensors/model.safetensors.index.json",
            "safetensors/config.json",
        ],
    ),
    (
        "safetensors/dtypes",
        "llama",
        &["safetensors/dtypes.safetensors"],
    ),
];

fn golden_text() -> String {
    let mut out = String::from(
        "# SPDX-License-Identifier: MIT OR Apache-2.0\n\
         # Golden FileDigests and v1 ModelIds of the fixtures under tests/data (RFC-0041 §1).\n\
         # Checked by identity::tests; a changed line is a changed formula or fixture, not a re-bless.\n",
    );
    let mut files: Vec<&str> = MODELS
        .iter()
        .flat_map(|(_, _, f)| f.iter().copied())
        .collect();
    files.sort_unstable();
    files.dedup();
    for name in files {
        let f = digest_of(name);
        out.push_str(&format!("file {name} {} {}\n", f.digest, f.len));
    }
    for (name, tag, files) in MODELS {
        let files: Vec<SourceFile> = files.iter().map(|f| digest_of(f)).collect();
        let id = model_id(tag, &files).unwrap();
        out.push_str(&format!("model {name} {tag} {}\n", hex(&id)));
    }
    out
}

#[test]
fn golden_ids_of_the_fixtures_do_not_change() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/model_ids.txt");
    let now = golden_text();
    if std::env::var_os("SYNTROPY_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &now).unwrap();
    }
    let checked_in = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        checked_in, now,
        "a fixture's FileDigest or ModelId changed; see the module docs"
    );
}
