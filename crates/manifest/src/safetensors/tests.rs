// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the safetensors reader (COMP-E10-T04; RFC-0041 §2.2, §10).
//!
//! The fixtures under `tests/data/safetensors/` are synthetic: [`fixtures`] writes every one
//! of them, and each `.hubrec` is the recording of the requests the reader makes against its
//! file. Nothing comes from a real checkpoint or from the network. To regenerate them after
//! changing the generator or the read pattern, run the tests with `SYNTROPY_BLESS=1`.
//!
//! The `fuzz_*` tests are structured fuzzing with proptest, standing in for the
//! `safetensors_header` cargo-fuzz target of RFC-0060 §10.1 until the fuzz workspace exists.
//! They assert §10.1's oracle and §10.2's properties: no panic, a classified error (never
//! `Internal`), and on acceptance offsets inside the file, no overlap and lengths that match
//! the dtype and shape.

use std::path::PathBuf;

use proptest::prelude::*;
use syntropy_core::{DType, ErrorClass};

use super::index::{INDEX_FILE, read_index, read_sharded};
use super::side::{SourceRole, classify, source_files};
use super::*;
use crate::range::replay::{Recorder, Recording, Replay};
use crate::range::{FileSource, HEADER_MAX_BYTES_DEFAULT, RangeError, RangeSource};

// ---- An in-memory source and a writer ----------------------------------------------------------

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

/// One tensor entry as a test writes it: name, dtype, shape, data_offsets.
type Entry<'a> = (&'a str, &'a str, &'a [u64], [u64; 2]);

/// A file: the JSON header built from `entries` and `metadata`, then `data_len` bytes of data,
/// byte `i` of the data being `i mod 251`.
fn file(entries: &[Entry<'_>], metadata: &[(&str, &str)], data_len: u64) -> Vec<u8> {
    let mut header = String::from("{");
    let mut first = true;
    let mut sep = |h: &mut String| {
        if !first {
            h.push(',');
        }
        first = false;
    };
    if !metadata.is_empty() {
        sep(&mut header);
        header.push_str("\"__metadata__\":{");
        let kv: Vec<String> = metadata
            .iter()
            .map(|(k, v)| format!("{k:?}:{v:?}"))
            .collect();
        header.push_str(&kv.join(","));
        header.push('}');
    }
    for (name, dtype, shape, [b, e]) in entries {
        sep(&mut header);
        header.push_str(&format!(
            "{name:?}:{{\"dtype\":{dtype:?},\"shape\":{shape:?},\"data_offsets\":[{b},{e}]}}"
        ));
    }
    header.push('}');
    raw(header.as_bytes(), data_len)
}

/// A file with the header text `header`, padded with spaces to a multiple of 8 as writers do.
fn raw(header: &[u8], data_len: u64) -> Vec<u8> {
    let mut h = header.to_vec();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend((0..data_len).map(|i| (i % 251) as u8));
    out
}

fn parse(bytes: Vec<u8>) -> Result<SafetensorsHeader, SafetensorsError> {
    read_header(Mem(bytes), HEADER_MAX_BYTES_DEFAULT)
}

fn code(r: Result<SafetensorsHeader, SafetensorsError>) -> Option<u16> {
    r.unwrap_err().code()
}

// ---- Container validation ---------------------------------------------------------------------

#[test]
fn reads_a_valid_file() {
    let bytes = file(
        &[
            ("a", "F32", &[2, 3], [0, 24]),
            ("b", "F16", &[4], [24, 32]),
            ("c", "BF16", &[], [32, 34]),
            ("empty", "F32", &[0, 5], [34, 34]),
        ],
        &[("format", "pt")],
        34,
    );
    let h = parse(bytes.clone()).unwrap();
    assert_eq!(h.file_len, bytes.len() as u64);
    assert_eq!(h.data_start, bytes.len() as u64 - 34);
    assert_eq!(h.metadata, vec![("format".to_owned(), "pt".to_owned())]);
    let a = h.tensor("a").unwrap();
    assert_eq!(
        (a.dtype, a.shape.clone(), a.offset, a.len),
        (DType::F32, vec![2, 3], h.data_start, 24)
    );
    assert_eq!(h.tensor("b").unwrap().dtype, DType::F16);
    let c = h.tensor("c").unwrap();
    assert_eq!(
        (c.dtype, c.len, c.offset),
        (DType::BF16, 2, h.data_start + 32)
    );
    assert_eq!(h.tensor("empty").unwrap().len, 0);
    assert_eq!(
        h.tensors
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["a", "b", "c", "empty"]
    );
}

#[test]
fn overlapping_offsets_are_header_invalid() {
    let r = parse(file(
        &[("a", "F32", &[4], [0, 16]), ("b", "F32", &[4], [8, 24])],
        &[],
        24,
    ));
    let e = r.unwrap_err();
    assert_eq!((e.code(), e.class()), (Some(6), ErrorClass::Integrity));
    assert!(e.to_string().contains("overlap"), "{e}");
    // The same span twice.
    assert_eq!(
        code(parse(file(
            &[("a", "F16", &[2], [0, 4]), ("b", "F16", &[2], [0, 4])],
            &[],
            4
        ))),
        Some(6)
    );
    // Order in the header does not hide it.
    assert_eq!(
        code(parse(file(
            &[("b", "F32", &[2], [4, 12]), ("a", "F32", &[2], [0, 8])],
            &[],
            12
        ))),
        Some(6)
    );
    // An empty tensor inside another holds no byte, so it does not overlap.
    assert!(
        parse(file(
            &[("a", "F32", &[4], [0, 16]), ("e", "F32", &[0], [8, 8])],
            &[],
            16
        ))
        .is_ok()
    );
}

#[test]
fn out_of_range_offsets_are_header_invalid() {
    // Past the data section's end.
    assert_eq!(
        code(parse(file(&[("a", "F32", &[4], [0, 16])], &[], 12))),
        Some(6)
    );
    // begin > end.
    assert_eq!(
        code(parse(file(&[("a", "F32", &[0], [8, 4])], &[], 12))),
        Some(6)
    );
    // Offsets near u64::MAX.
    let max = u64::MAX;
    assert_eq!(
        code(parse(file(&[("a", "F32", &[1], [max - 3, max])], &[], 12))),
        Some(6)
    );
}

#[test]
fn a_length_that_does_not_match_the_shape_is_header_invalid() {
    let e = parse(file(&[("a", "F32", &[2, 3], [0, 20])], &[], 24)).unwrap_err();
    assert_eq!(e.code(), Some(6));
    assert!(e.to_string().contains("needs 24"), "{e}");
    // An element count that overflows.
    let big = u64::MAX / 2;
    assert_eq!(
        code(parse(file(&[("a", "F16", &[big, 4], [0, 8])], &[], 8))),
        Some(6)
    );
}

#[test]
fn other_dtypes_are_unsupported_container() {
    for dtype in ["F8_E4M3", "I32", "BOOL", "F64", "Q4_0", "f32"] {
        let e = parse(file(&[("a", dtype, &[1], [0, 4])], &[], 4)).unwrap_err();
        assert_eq!(
            (e.code(), e.class()),
            (Some(2), ErrorClass::Unsupported),
            "{dtype}"
        );
        assert!(e.to_string().contains(dtype));
    }
}

#[test]
fn the_header_size_is_bounded() {
    // Declared length past 100 MiB.
    let mut bytes = (HEADER_MAX + 1).to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}      ");
    let e = parse(bytes).unwrap_err();
    assert_eq!((e.code(), e.class()), (Some(5), ErrorClass::Integrity));
    // Past store.header_max_bytes.
    let small = file(&[("a", "F32", &[1], [0, 4])], &[], 4);
    let e = read_header(Mem(small.clone()), 16).unwrap_err();
    assert_eq!(e.code(), Some(5));
    // Past the file's end.
    let mut bytes = 1000u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    assert_eq!(code(parse(bytes)), Some(6));
    // Shorter than the length word.
    assert_eq!(code(parse(vec![1, 2, 3])), Some(6));
    // Only the first 8 bytes and the header are read: two requests, no data byte.
    let rec = Recorder::new(Mem(small.clone()), "mem");
    read_header(&rec, HEADER_MAX_BYTES_DEFAULT).unwrap();
    let r = rec.recording();
    assert_eq!(r.responses.len(), 2);
    let header_end = small.len() as u64 - 4;
    assert!(
        r.responses
            .iter()
            .all(|(s, b)| s + b.len() as u64 <= header_end)
    );
}

#[test]
fn malformed_headers_are_header_invalid() {
    let cases: &[&[u8]] = &[
        b"",
        b"[]",
        b"{\"a\":1}",
        b"{\"a\":{\"dtype\":\"F32\",\"shape\":[1]}}",
        b"{\"a\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,0]}}",
        b"{\"a\":{\"dtype\":\"F32\",\"shape\":[1.5],\"data_offsets\":[0,0]}}",
        b"{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4,8]}}",
        b"{\"a\":{\"dtype\":7,\"shape\":[1],\"data_offsets\":[0,4]}}",
        b"{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4],\"extra\":1}}",
        b"{\"a\":{\"dtype\":\"F32\",\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,4]}}",
        b"{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
        b"{\"__metadata__\":{\"k\":1}}",
        b"{\"__metadata__\":[]}",
        b"{\"a\":",
        b"{} trailing",
        b"\xff\xfe",
    ];
    for c in cases {
        let e = parse(raw(c, 4)).unwrap_err();
        assert_eq!(
            (e.code(), e.class()),
            (Some(6), ErrorClass::Integrity),
            "{:?}: {e}",
            String::from_utf8_lossy(c)
        );
    }
    // A header of only spaces (the length word counts the padding) is not a JSON object.
    assert_eq!(code(parse(raw(b"        ", 0))), Some(6));
}

// ---- Sharded index and side files -------------------------------------------------------------

fn index_text(map: &[(&str, &str)]) -> Vec<u8> {
    let kv: Vec<String> = map.iter().map(|(t, f)| format!("{t:?}:{f:?}")).collect();
    format!(
        "{{\"metadata\":{{\"total_size\":40}},\"weight_map\":{{{}}}}}",
        kv.join(",")
    )
    .into_bytes()
}

fn shard_set() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (
            "s1.safetensors",
            file(
                &[("x", "F32", &[2], [0, 8]), ("y", "F16", &[4], [8, 16])],
                &[],
                16,
            ),
        ),
        (
            "s2.safetensors",
            file(&[("z", "BF16", &[2, 6], [0, 24])], &[], 24),
        ),
    ]
}

fn read_set(
    index: Vec<u8>,
    shards: Vec<(&'static str, Vec<u8>)>,
) -> Result<index::ShardedHeaders, SafetensorsError> {
    read_sharded(
        Mem(index),
        |name| {
            shards
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, b)| Mem(b.clone()))
                .ok_or_else(|| SafetensorsError::invalid(format!("no shard {name}")))
        },
        HEADER_MAX_BYTES_DEFAULT,
    )
}

#[test]
fn a_sharded_set_parses_and_locates_tensors() {
    let index = index_text(&[
        ("x", "s1.safetensors"),
        ("y", "s1.safetensors"),
        ("z", "s2.safetensors"),
    ]);
    let set = read_set(index, shard_set()).unwrap();
    assert_eq!(set.index.total_size, Some(40));
    assert_eq!(set.index.shards, ["s1.safetensors", "s2.safetensors"]);
    let (f, t) = set.locate("z").unwrap();
    assert_eq!((f, t.dtype, t.len), ("s2.safetensors", DType::BF16, 24));
    assert!(set.locate("w").is_none());
}

#[test]
fn an_index_that_disagrees_with_its_shards_is_header_invalid() {
    let bad_sets: [&[(&str, &str)]; 4] = [
        // y missing from the index.
        &[("x", "s1.safetensors"), ("z", "s2.safetensors")],
        // y placed in the wrong shard.
        &[
            ("x", "s1.safetensors"),
            ("y", "s2.safetensors"),
            ("z", "s2.safetensors"),
        ],
        // w indexed but held by no shard.
        &[
            ("x", "s1.safetensors"),
            ("y", "s1.safetensors"),
            ("z", "s2.safetensors"),
            ("w", "s2.safetensors"),
        ],
        // A path, not a file name.
        &[("x", "../s1.safetensors")],
    ];
    for map in bad_sets {
        let e = read_set(index_text(map), shard_set()).unwrap_err();
        assert_eq!(e.code(), Some(6), "{map:?}: {e}");
    }
    // The same tensor in two shards.
    let mut shards = shard_set();
    shards[1].1 = file(
        &[("x", "F32", &[2], [0, 8]), ("z", "BF16", &[2, 6], [8, 32])],
        &[],
        32,
    );
    let e = read_set(
        index_text(&[
            ("x", "s1.safetensors"),
            ("y", "s1.safetensors"),
            ("z", "s2.safetensors"),
        ]),
        shards,
    )
    .unwrap_err();
    assert_eq!(e.code(), Some(6));
    assert!(e.to_string().contains("in both"), "{e}");
    // Malformed index files.
    for text in [
        &b"{}"[..],
        b"{\"weight_map\":[]}",
        b"{\"weight_map\":{\"x\":1}}",
        b"{\"weight_map\":{},\"other\":1}",
        b"{\"metadata\":{\"total_size\":-1},\"weight_map\":{}}",
        b"not json",
    ] {
        let e = read_index(Mem(text.to_vec()), HEADER_MAX_BYTES_DEFAULT).unwrap_err();
        assert_eq!(e.code(), Some(6), "{}", String::from_utf8_lossy(text));
    }
    let e = read_index(Mem(vec![b' '; 64]), 32).unwrap_err();
    assert_eq!(e.code(), Some(5));
}

#[test]
fn shard_names_must_be_plain_file_names() {
    use super::index::is_plain_file_name;
    for ok in ["model-00001-of-00002.safetensors", "a.safetensors"] {
        assert!(is_plain_file_name(ok), "{ok}");
    }
    for bad in ["", ".", "..", "a/b", "..\\a", "/etc/x", "c:x", "a\0b"] {
        assert!(!is_plain_file_name(bad), "{bad:?}");
    }
}

#[test]
fn side_files_are_source_files_with_roles() {
    assert_eq!(classify("config.json"), Some(SourceRole::Config));
    assert_eq!(classify(INDEX_FILE), Some(SourceRole::WeightIndex));
    assert_eq!(
        classify("model-00001-of-00002.safetensors"),
        Some(SourceRole::Weights)
    );
    assert_eq!(classify(".safetensors"), None);
    assert_eq!(classify("README.md"), None);
    assert!(SourceRole::GenerationConfig.in_model_id());
    assert!(!SourceRole::Tokenizer.in_model_id());
    assert!(!SourceRole::TokenizerConfig.in_model_id());
    assert!(!SourceRole::SpecialTokensMap.in_model_id());
    let files = source_files([
        ("tokenizer.json", 9),
        ("README.md", 1),
        ("config.json", 3),
        ("b.safetensors", 7),
        ("a.safetensors", 5),
        ("special_tokens_map.json", 2),
        ("tokenizer_config.json", 4),
        ("generation_config.json", 6),
        (INDEX_FILE, 8),
    ]);
    let got: Vec<(&str, SourceRole)> = files.iter().map(|f| (f.name.as_str(), f.role)).collect();
    assert_eq!(
        got,
        [
            ("a.safetensors", SourceRole::Weights),
            ("b.safetensors", SourceRole::Weights),
            (INDEX_FILE, SourceRole::WeightIndex),
            ("config.json", SourceRole::Config),
            ("generation_config.json", SourceRole::GenerationConfig),
            ("tokenizer.json", SourceRole::Tokenizer),
            ("tokenizer_config.json", SourceRole::TokenizerConfig),
            ("special_tokens_map.json", SourceRole::SpecialTokensMap),
        ]
    );
}

// ---- Fixtures: local files and their recorded Hub responses -----------------------------------

const DIR: &str = "tests/data/safetensors";
const SINGLE: &str = "tiny.safetensors";
const SHARD1: &str = "model-00001-of-00002.safetensors";
const SHARD2: &str = "model-00002-of-00002.safetensors";

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(DIR)
        .join(name)
}

/// Every fixture file, by name. All synthetic.
fn fixtures() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (
            SINGLE,
            file(
                &[
                    ("model.embed.weight", "BF16", &[8, 4], [0, 64]),
                    ("model.norm.weight", "F32", &[4], [64, 80]),
                    ("model.head.weight", "F16", &[8, 4], [80, 144]),
                    ("model.empty", "F32", &[0], [144, 144]),
                ],
                &[("format", "pt"), ("origin", "synthetic")],
                144,
            ),
        ),
        (
            SHARD1,
            file(
                &[
                    ("layers.0.attn.weight", "BF16", &[4, 4], [0, 32]),
                    ("layers.0.norm.weight", "F32", &[4], [32, 48]),
                ],
                &[("format", "pt")],
                48,
            ),
        ),
        (
            SHARD2,
            file(
                &[
                    ("layers.1.attn.weight", "BF16", &[4, 4], [0, 32]),
                    ("layers.1.norm.weight", "F32", &[4], [32, 48]),
                    ("head.weight", "F16", &[2, 4], [48, 64]),
                ],
                &[("format", "pt")],
                64,
            ),
        ),
        (
            INDEX_FILE,
            format!(
                "{{\n  \"metadata\": {{\"total_size\": 112}},\n  \"weight_map\": {{\n    \
                 \"head.weight\": \"{SHARD2}\",\n    \"layers.0.attn.weight\": \"{SHARD1}\",\n    \
                 \"layers.0.norm.weight\": \"{SHARD1}\",\n    \"layers.1.attn.weight\": \
                 \"{SHARD2}\",\n    \"layers.1.norm.weight\": \"{SHARD2}\"\n  }}\n}}\n"
            )
            .into_bytes(),
        ),
        (
            "config.json",
            b"{\"architectures\": [\"SyntheticForTest\"], \"hidden_size\": 4, \"num_hidden_layers\": 2}\n"
                .to_vec(),
        ),
    ]
}

/// The files that get a recording: everything the reader reads by range.
const RECORDED: [&str; 4] = [SINGLE, SHARD1, SHARD2, INDEX_FILE];

/// Runs the reader over `name` through a recorder and returns the recording.
fn record(name: &str) -> Recording {
    let file = FileSource::open(data(name)).unwrap();
    let rec = Recorder::new(&file, format!("synthetic:{name}"));
    if name == INDEX_FILE {
        read_index(&rec, HEADER_MAX_BYTES_DEFAULT).unwrap();
    } else {
        read_header(&rec, HEADER_MAX_BYTES_DEFAULT).unwrap();
    }
    rec.recording()
}

fn hubrec(name: &str) -> String {
    format!("{name}.hubrec")
}

#[test]
fn fixtures_are_the_generator_output_and_their_recordings() {
    if std::env::var_os("SYNTROPY_BLESS").is_some() {
        std::fs::create_dir_all(data("")).unwrap();
        for (name, bytes) in fixtures() {
            std::fs::write(data(name), bytes).unwrap();
        }
        for name in RECORDED {
            let text = format!(
                "# SPDX-License-Identifier: MIT OR Apache-2.0\n# Synthetic: made by safetensors::tests, not recorded from the network. See src/range/replay.rs for the format.\n{}",
                record(name).to_text()
            );
            std::fs::write(data(&hubrec(name)), text).unwrap();
        }
    }
    for (name, bytes) in fixtures() {
        assert_eq!(std::fs::read(data(name)).unwrap(), bytes, "{name}");
    }
    for name in RECORDED {
        let text = std::fs::read_to_string(data(&hubrec(name))).unwrap();
        assert_eq!(Recording::parse(&text).unwrap(), record(name), "{name}");
    }
}

fn replay(name: &str) -> Replay {
    Replay::parse(&std::fs::read_to_string(data(&hubrec(name))).unwrap()).unwrap()
}

#[test]
fn a_local_file_and_its_recorded_hub_fixture_give_identical_results() {
    let local = read_header(
        FileSource::open(data(SINGLE)).unwrap(),
        HEADER_MAX_BYTES_DEFAULT,
    );
    let hub = read_header(replay(SINGLE), HEADER_MAX_BYTES_DEFAULT);
    assert!(local.is_ok());
    assert_eq!(local, hub);
    assert_eq!(local.unwrap().tensors.len(), 4);
}

#[test]
fn the_sharded_fixture_parses_locally_and_from_the_hub_alike() {
    let local = read_sharded(
        FileSource::open(data(INDEX_FILE)).unwrap(),
        |name| FileSource::open(data(name)).map_err(SafetensorsError::from),
        HEADER_MAX_BYTES_DEFAULT,
    )
    .unwrap();
    let hub = read_sharded(
        replay(INDEX_FILE),
        |name| {
            let path = data(&hubrec(name));
            let text = std::fs::read_to_string(path)
                .map_err(|e| SafetensorsError::invalid(e.to_string()))?;
            Replay::parse(&text).map_err(SafetensorsError::from)
        },
        HEADER_MAX_BYTES_DEFAULT,
    )
    .unwrap();
    assert_eq!(local, hub);
    assert_eq!(local.index.shards, [SHARD2, SHARD1]);
    assert_eq!(local.index.total_size, Some(112));
    let (f, t) = local.locate("layers.0.attn.weight").unwrap();
    assert_eq!(
        (f, t.dtype, t.shape.as_slice()),
        (SHARD1, DType::BF16, &[4, 4][..])
    );
    let total: u64 = local
        .shards
        .iter()
        .flat_map(|(_, h)| h.tensors.iter().map(|t| t.len))
        .sum();
    assert_eq!(total, 112);
    // The directory's source files, with the side file.
    let names: Vec<(String, u64)> = fixtures()
        .into_iter()
        .filter(|(n, _)| *n != SINGLE)
        .map(|(n, b)| (n.to_owned(), b.len() as u64))
        .collect();
    let files = source_files(names.iter().map(|(n, l)| (n.as_str(), *l)));
    assert_eq!(files.len(), 4);
    assert_eq!(files[3].role, SourceRole::Config);
}

// ---- Structured fuzzing (proptest) -------------------------------------------------------------

/// RFC-0060 §10.1's oracle over an accepted header.
fn oracle(h: &SafetensorsHeader) {
    let mut spans: Vec<(u64, u64)> = Vec::new();
    for t in &h.tensors {
        assert!(t.offset >= h.data_start);
        assert!(t.offset + t.len <= h.file_len);
        let numel: u64 = t.shape.iter().product();
        assert_eq!(numel * u64::from(t.dtype.bits() / 8), t.len);
        if t.len > 0 {
            spans.push((t.offset, t.offset + t.len));
        }
    }
    spans.sort_unstable();
    for w in spans.windows(2) {
        assert!(w[0].1 <= w[1].0, "overlap {w:?}");
    }
}

/// RFC-0060 §10.2: a rejection is classified, never `Internal`.
fn classified(e: &SafetensorsError) {
    assert!(e.code().is_some(), "{e:?}");
    assert_ne!(e.class(), ErrorClass::Internal, "{e:?}");
}

fn check(bytes: Vec<u8>) {
    match parse(bytes) {
        Ok(h) => oracle(&h),
        Err(e) => classified(&e),
    }
}

prop_compose! {
    /// A tensor entry with small, possibly inconsistent, fields.
    fn entry()(
        dtype in prop::sample::select(vec!["F32", "F16", "BF16", "F8_E4M3", "I32"]),
        shape in prop::collection::vec(0u64..6, 0..4),
        begin in 0u64..96,
        len in 0u64..64,
        exact in any::<bool>(),
    ) -> (String, Vec<u64>, u64, u64) {
        let size = match dtype { "F32" | "I32" => 4, "F8_E4M3" => 1, _ => 2 };
        let numel: u64 = shape.iter().product();
        let len = if exact { numel * size } else { len };
        (dtype.to_owned(), shape, begin, begin + len)
    }
}

/// A header of up to 6 entries with names that may repeat, and a data section that may be
/// short.
fn structured() -> impl Strategy<Value = Vec<u8>> {
    (
        prop::collection::vec((0u8..8, entry()), 0..6),
        0u64..160,
        any::<bool>(),
    )
        .prop_map(|(entries, data_len, meta)| {
            let names: Vec<String> = entries.iter().map(|(n, _)| format!("t{n}")).collect();
            let owned: Vec<(String, Vec<u64>, [u64; 2], String)> = entries
                .into_iter()
                .map(|(_, (d, s, b, e))| (d, s, [b, e], String::new()))
                .collect();
            let es: Vec<Entry<'_>> = owned
                .iter()
                .zip(&names)
                .map(|((d, s, o, _), n)| (n.as_str(), d.as_str(), s.as_slice(), *o))
                .collect();
            let md: &[(&str, &str)] = if meta { &[("k", "v")] } else { &[] };
            file(&es, md, data_len)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    /// Arbitrary bytes never panic, and every rejection is classified.
    #[test]
    fn fuzz_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..256)) {
        check(bytes);
    }

    /// An arbitrary header text behind a correct length word.
    #[test]
    fn fuzz_arbitrary_header_text(text in "\\{[\"a-z0-9_:,\\[\\]{} .-]{0,120}\\}", data in 0u64..64) {
        check(raw(text.as_bytes(), data));
    }

    /// Structured headers with consistent and inconsistent offsets, lengths, dtypes and
    /// names: accepted ones satisfy the oracle; rejected ones are classified.
    #[test]
    fn fuzz_structured(bytes in structured()) {
        check(bytes);
    }

    /// One byte of a valid file flipped: still no panic, still classified.
    #[test]
    fn fuzz_mutated_fixture(at in 0usize..400, to in any::<u8>()) {
        let mut bytes = fixtures().remove(0).1;
        let i = at % bytes.len();
        bytes[i] = to;
        check(bytes);
    }

    /// The index parser on arbitrary text: no panic; a rejection is HeaderInvalid.
    #[test]
    fn fuzz_index(text in "\\{[\"a-z0-9_:,\\[\\]{} ./-]{0,120}\\}") {
        if let Err(e) = read_index(Mem(text.into_bytes()), HEADER_MAX_BYTES_DEFAULT) {
            prop_assert_eq!(e.code(), Some(6));
        }
    }
}

#[test]
fn errors_from_reads_keep_their_class() {
    let e = SafetensorsError::from(RangeError::NotRecorded { offset: 0, len: 8 });
    assert_eq!(e.class(), ErrorClass::Internal);
    let e = SafetensorsError::from(RangeError::OutOfRange {
        offset: 0,
        len: 8,
        source_len: 4,
    });
    assert_eq!(e.code(), Some(6));
}
