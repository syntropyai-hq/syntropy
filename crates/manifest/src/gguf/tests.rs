// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the GGUF reader (COMP-E10-T02; RFC-0041 §2.1, §10).
//!
//! Every input is made by the generator in [`super::synth`] or written field by field with its
//! byte writer; nothing comes from a real checkpoint or the network. The checked-in fixtures
//! under `tests/data/gguf/` are the generator's output and the recording of the reader's
//! requests against it ([`fixtures`]); regenerate them with `SYNTROPY_BLESS=1 cargo test -p
//! syntropy-manifest`.

use proptest::prelude::*;
use syntropy_core::{ErrorClass, GgmlType, QuantScheme};

use super::synth::{Synth, TensorSpec, W};
use super::*;
use crate::range::HeaderReader;

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

fn parse(bytes: Vec<u8>) -> Result<GgufHeader, GgufError> {
    GgufHeader::read(Mem(bytes), &Limits::DEFAULT)
}

fn parse_with(bytes: Vec<u8>, limits: &Limits) -> Result<GgufHeader, GgufError> {
    GgufHeader::read(Mem(bytes), limits)
}

fn code(r: Result<GgufHeader, GgufError>) -> u16 {
    let e = r.expect_err("the reader accepted a file it must refuse");
    e.code()
        .unwrap_or_else(|| panic!("error without a code: {e}"))
}

fn str_val(s: &str) -> Value {
    Value::Str(s.as_bytes().into())
}

/// A small model with every value type and a few tensors.
fn sample(version: u32) -> Synth {
    Synth::new(version)
        .kv("general.architecture", str_val("fixture"))
        .kv("fixture.block_count", Value::U32(2))
        .kv("fixture.context_length", Value::U64(4096))
        .kv("fixture.rope.freq_base", Value::F32(10_000.0))
        .kv("fixture.eps", Value::F64(1e-6))
        .kv("fixture.flag", Value::Bool(true))
        .kv("fixture.u8", Value::U8(7))
        .kv("fixture.i8", Value::I8(-7))
        .kv("fixture.u16", Value::U16(700))
        .kv("fixture.i16", Value::I16(-700))
        .kv("fixture.i32", Value::I32(-70_000))
        .kv("fixture.i64", Value::I64(-7_000_000_000))
        .kv(
            "tokenizer.ggml.tokens",
            Value::Array(Array::Str(vec![
                b"<s>"[..].into(),
                b"</s>"[..].into(),
                b"\xff"[..].into(),
            ])),
        )
        .kv(
            "tokenizer.ggml.scores",
            Value::Array(Array::F32(vec![0.0, -1.5, 2.25])),
        )
        .kv(
            "tokenizer.ggml.token_type",
            Value::Array(Array::I32(vec![3, 3, 6])),
        )
        .kv(
            "fixture.bools",
            Value::Array(Array::Bool(vec![true, false])),
        )
        .kv("fixture.empty", Value::Array(Array::U64(vec![])))
        .tensor("token_embd.weight", &[64, 3], GgmlType::F16)
        .tensor("blk.0.attn_q.weight", &[256, 4], GgmlType::Q4_K)
        .tensor("blk.0.ffn_gate_exps.weight", &[32, 2, 4], GgmlType::Q8_0)
        .tensor("output_norm.weight", &[64], GgmlType::F32)
}

/// What the reader must report for `s`'s tensors, given the data section's start.
fn expected_tensors(s: &Synth, data_start: u64, file: u16) -> Vec<GgufTensor> {
    s.tensors
        .iter()
        .zip(s.layout())
        .map(|(t, (offset, bytes))| GgufTensor {
            name: t.name.clone(),
            ne: t.ne.clone(),
            ggml_type: t.ggml_type,
            offset,
            start: data_start + offset,
            bytes,
            file,
        })
        .collect()
}

fn check_matches(s: &Synth, h: &GgufHeader) {
    assert_eq!(h.version, s.version);
    assert_eq!(h.alignment, s.alignment());
    let header_len = s.header().len() as u64;
    assert_eq!(h.header_bytes, header_len);
    assert_eq!(h.data_start, pad(header_len, s.alignment()).unwrap());
    assert_eq!(h.tensors, expected_tensors(s, h.data_start, 0));
    let kvs: Vec<(String, Value)> = h
        .metadata
        .iter()
        .map(|(k, v)| (k.to_owned(), v.clone()))
        .collect();
    assert_eq!(kvs, s.kvs);
}

// ---- Versions and endianness -----------------------------------------------------------------

#[test]
fn v2_and_v3_files_parse() {
    for version in [2, 3] {
        let s = sample(version);
        let h = parse(s.build()).unwrap();
        check_matches(&s, &h);
        assert_eq!(h.alignment, DEFAULT_ALIGNMENT);
        assert_eq!(h.tensors[1].scheme(), QuantScheme::Ggml(GgmlType::Q4_K));
        assert_eq!(h.tensors[1].bytes, 4 * 144);
        assert_eq!(h.tensors[2].elements(), 32 * 2 * 4);
    }
}

#[test]
fn v1_big_endian_and_unknown_versions_fail_with_code_2() {
    let mut w = W::default();
    w.preamble(1, 0, 0);
    let e = parse(w.0).unwrap_err();
    assert_eq!(e, GgufError::UnsupportedContainer(Unsupported::Version1));
    assert_eq!((e.code(), e.class()), (Some(2), ErrorClass::Unsupported));

    // A big-endian file writes the magic as bytes and every number big-endian.
    for version in [1u32, 2, 3] {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&version.to_be_bytes());
        b.extend_from_slice(&0u64.to_be_bytes());
        b.extend_from_slice(&0u64.to_be_bytes());
        let e = parse(b).unwrap_err();
        assert_eq!(
            e,
            GgufError::UnsupportedContainer(Unsupported::BigEndian { version })
        );
        assert_eq!(e.code(), Some(2));
    }

    for version in [0u32, 4, 0x0102_0304, u32::MAX] {
        let mut w = W::default();
        w.preamble(version, 0, 0);
        assert_eq!(
            parse(w.0).unwrap_err(),
            GgufError::UnsupportedContainer(Unsupported::UnknownVersion(version))
        );
    }
}

#[test]
fn input_without_the_magic_is_header_invalid() {
    for bytes in [
        b"GGUG\x03\0\0\0".to_vec(),
        b"FUGG".to_vec(),
        vec![],
        b"GG".to_vec(),
    ] {
        assert_eq!(code(parse(bytes)), 6);
    }
}

// ---- Limits ----------------------------------------------------------------------------------

fn limit_of(r: Result<GgufHeader, GgufError>) -> (Limit, u64, u64) {
    match r {
        Err(GgufError::HeaderLimit { limit, value, max }) => (limit, value, max),
        other => panic!("expected HeaderLimit, got {other:?}"),
    }
}

#[test]
fn each_default_limit_is_enforced_with_code_5_before_reading() {
    let d = Limits::DEFAULT;
    assert_eq!(
        (
            d.tensors,
            d.kv_pairs,
            d.string_bytes,
            d.array_len,
            d.header_bytes
        ),
        (1 << 20, 1 << 16, 64 << 20, 1 << 24, 1 << 30)
    );

    // Each file is a few bytes long but declares a count past the limit, so the reader must
    // refuse it from the count alone.
    let mut w = W::default();
    w.preamble(3, (1 << 20) + 1, 0);
    assert_eq!(
        limit_of(parse(w.0)),
        (Limit::Tensors, (1 << 20) + 1, 1 << 20)
    );

    let mut w = W::default();
    w.preamble(3, 0, (1 << 16) + 1);
    assert_eq!(
        limit_of(parse(w.0)),
        (Limit::KvPairs, (1 << 16) + 1, 1 << 16)
    );

    // A key one byte longer than 64 MiB.
    let mut w = W::default();
    w.preamble(3, 0, 1).u64((64 << 20) + 1);
    assert_eq!(
        limit_of(parse(w.0)),
        (Limit::StringBytes, (64 << 20) + 1, 64 << 20)
    );

    // A string value one byte longer than 64 MiB.
    let mut w = W::default();
    w.preamble(3, 0, 1)
        .str("k")
        .u32(ValueType::Str.id())
        .u64((64 << 20) + 1);
    assert_eq!(limit_of(parse(w.0)).0, Limit::StringBytes);

    // An array of 2²⁴ + 1 elements.
    let mut w = W::default();
    w.preamble(3, 0, 1)
        .str("k")
        .u32(ValueType::Array.id())
        .u32(ValueType::U8.id())
        .u64((1 << 24) + 1);
    assert_eq!(
        limit_of(parse(w.0)),
        (Limit::ArrayLen, (1 << 24) + 1, 1 << 24)
    );

    // A tensor name one byte longer than 64 MiB.
    let mut w = W::default();
    w.preamble(3, 1, 0).u64((64 << 20) + 1);
    assert_eq!(limit_of(parse(w.0)).0, Limit::StringBytes);
}

#[test]
fn limits_admit_their_bound_and_refuse_one_more() {
    let tight = Limits {
        tensors: 2,
        kv_pairs: 2,
        string_bytes: 8,
        array_len: 3,
        header_bytes: 1 << 20,
    };
    let base = || Synth::new(3);

    let two = base()
        .tensor("a", &[4], GgmlType::F32)
        .tensor("b", &[4], GgmlType::F32);
    parse_with(two.build(), &tight).unwrap();
    let three = two.clone().tensor("c", &[4], GgmlType::F32);
    assert_eq!(
        limit_of(parse_with(three.build(), &tight)).0,
        Limit::Tensors
    );

    let kv2 = base().kv("a", Value::U8(1)).kv("b", Value::U8(2));
    parse_with(kv2.build(), &tight).unwrap();
    let kv3 = kv2.kv("c", Value::U8(3));
    assert_eq!(limit_of(parse_with(kv3.build(), &tight)).0, Limit::KvPairs);

    parse_with(base().kv("12345678", Value::U8(1)).build(), &tight).unwrap();
    assert_eq!(
        limit_of(parse_with(
            base().kv("123456789", Value::U8(1)).build(),
            &tight
        )),
        (Limit::StringBytes, 9, 8)
    );
    assert_eq!(
        limit_of(parse_with(
            base().kv("k", str_val("123456789")).build(),
            &tight
        ))
        .0,
        Limit::StringBytes
    );
    assert_eq!(
        limit_of(parse_with(
            base().tensor("123456789", &[4], GgmlType::F32).build(),
            &tight
        ))
        .0,
        Limit::StringBytes
    );

    parse_with(
        base()
            .kv("a", Value::Array(Array::U32(vec![1, 2, 3])))
            .build(),
        &tight,
    )
    .unwrap();
    assert_eq!(
        limit_of(parse_with(
            base()
                .kv("a", Value::Array(Array::U32(vec![1, 2, 3, 4])))
                .build(),
            &tight
        )),
        (Limit::ArrayLen, 4, 3)
    );
    assert_eq!(
        limit_of(parse_with(
            base()
                .kv("a", Value::Array(Array::Str(vec![b"x"[..].into(); 4])))
                .build(),
            &tight
        ))
        .0,
        Limit::ArrayLen
    );
}

#[test]
fn the_header_cap_bounds_the_whole_header_and_never_reaches_the_data() {
    let s = sample(3);
    let header_len = s.header().len() as u64;
    let file = s.build();
    assert!(
        file.len() as u64 > header_len + 64,
        "the sample has a data section"
    );

    // A cap exactly at the end of the tensor-info table admits the file: the parser never
    // asks for a byte of the data section, and the cap bounds every request the header
    // reader makes as well as every read.
    let exact = Limits {
        header_bytes: header_len,
        ..Limits::DEFAULT
    };
    let h = parse_with(file.clone(), &exact).unwrap();
    check_matches(&s, &h);

    let short = Limits {
        header_bytes: header_len - 1,
        ..Limits::DEFAULT
    };
    let e = parse_with(file.clone(), &short).unwrap_err();
    assert!(matches!(
        e,
        GgufError::HeaderLimit {
            limit: Limit::HeaderBytes,
            max,
            ..
        } if max == header_len - 1
    ));
    assert_eq!((e.code(), e.class()), (Some(5), ErrorClass::Integrity));

    // The header reader's own cap counts too.
    let mut reader = HeaderReader::new(Mem(file), header_len - 1);
    assert_eq!(
        GgufHeader::read_from(&mut reader, &Limits::DEFAULT)
            .unwrap_err()
            .code(),
        Some(5)
    );
}

#[test]
fn header_reads_are_incremental_and_coalesced() {
    // A header of about 40 KiB read with a 1 KiB first request takes a handful of doubling
    // requests, not one per field.
    let mut s = Synth::new(3);
    for i in 0..1000 {
        s = s.kv(&format!("fixture.key_{i:04}"), Value::U32(i));
    }
    let file = s.build();
    let mut reader =
        HeaderReader::new(Mem(file), Limits::DEFAULT.header_bytes).with_initial_read(1024);
    let h = GgufHeader::read_from(&mut reader, &Limits::DEFAULT).unwrap();
    assert_eq!(h.metadata.len(), 1000);
    assert!(
        (2..=8).contains(&reader.requests()),
        "{} requests",
        reader.requests()
    );
}

// ---- Alignment -------------------------------------------------------------------------------

#[test]
fn general_alignment_sets_the_data_start_and_tensor_offsets() {
    for align in [1u32, 8, 64, 4096] {
        let s = Synth::new(3)
            .kv(ALIGNMENT_KEY, Value::U32(align))
            .tensor("a", &[3], GgmlType::F32)
            .tensor("b", &[5], GgmlType::F16);
        let h = parse(s.build()).unwrap();
        check_matches(&s, &h);
        assert_eq!(h.alignment, u64::from(align));
        assert_eq!(h.data_start % u64::from(align), 0);
        assert_eq!(h.tensors[1].offset, pad(12, u64::from(align)).unwrap());
    }
}

#[test]
fn a_bad_alignment_is_header_invalid() {
    for v in [
        Value::U32(0),
        Value::U32(3),
        Value::U32(48),
        Value::U64(32),
        Value::I32(32),
        str_val("32"),
    ] {
        // The header alone: the generator cannot lay out data at a bad alignment.
        let s = Synth::new(3).kv(ALIGNMENT_KEY, v.clone());
        assert_eq!(code(parse(s.header())), 6, "{v:?}");
    }
}

// ---- The ggml type mapping -------------------------------------------------------------------

#[test]
fn every_ggml_type_maps_to_its_scheme_with_its_size() {
    let mut s = Synth::new(3);
    for (i, t) in GgmlType::ALL.into_iter().enumerate() {
        let (block, _) = block_geometry(t).unwrap();
        s = s.tensor(&format!("t{i}"), &[block * 2, 3], t);
    }
    let h = parse(s.build()).unwrap();
    check_matches(&s, &h);
    for (t, want) in h.tensors.iter().zip(GgmlType::ALL) {
        assert_eq!(t.scheme(), QuantScheme::Ggml(want));
        assert_eq!(t.bytes, block_geometry(want).unwrap().1 * 2 * 3);
    }
}

#[test]
fn unknown_and_refused_type_ids_fail_with_code_4_naming_them() {
    for id in [4u32, 9, 15, 24, 26, 28, 31, 43, 1000] {
        let mut w = W::default();
        w.preamble(3, 1, 0).tensor_info("w", &[32], id, 0);
        let e = parse(w.0).unwrap_err();
        assert_eq!(
            e,
            GgufError::UnknownQuant {
                tensor: "w".into(),
                type_id: id
            }
        );
        assert_eq!((e.code(), e.class()), (Some(4), ErrorClass::Unsupported));
        let msg = e.to_string();
        assert!(msg.contains(&format!("ID {id}")), "{msg}");
        if let Some(name) = ggml_type_name(id) {
            assert!(msg.contains(&name), "{msg}");
        }
    }
}

// ---- Malformed headers -----------------------------------------------------------------------

/// A header with one tensor-info entry written by hand, followed by `data` zero bytes.
fn one_tensor(ne: &[u64], type_id: u32, offset: u64, data: usize) -> Vec<u8> {
    let mut w = W::default();
    w.preamble(3, 1, 0).tensor_info("w", ne, type_id, offset);
    let start = pad(w.0.len() as u64, 32).unwrap() as usize;
    w.0.resize(start + data, 0);
    w.0
}

#[test]
fn malformed_headers_are_header_invalid() {
    let f32_id = ggml_type_id(GgmlType::F32).unwrap();
    let q4k = ggml_type_id(GgmlType::Q4_K).unwrap();
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("duplicate key", {
            Synth::new(3)
                .kv("a", Value::U8(1))
                .kv("a", Value::U8(2))
                .build()
        }),
        ("unknown value type", {
            let mut w = W::default();
            w.preamble(3, 0, 1).str("k").u32(13).u32(0);
            w.0
        }),
        ("nested array", {
            let mut w = W::default();
            w.preamble(3, 0, 1)
                .str("k")
                .u32(ValueType::Array.id())
                .u32(ValueType::Array.id())
                .u64(0);
            w.0
        }),
        ("array of unknown type", {
            let mut w = W::default();
            w.preamble(3, 0, 1)
                .str("k")
                .u32(ValueType::Array.id())
                .u32(99)
                .u64(0);
            w.0
        }),
        ("bool of 2", {
            let mut w = W::default();
            w.preamble(3, 0, 1).str("k").u32(ValueType::Bool.id()).u8(2);
            w.0
        }),
        ("bool array element of 255", {
            let mut w = W::default();
            w.preamble(3, 0, 1)
                .str("k")
                .u32(ValueType::Array.id())
                .u32(ValueType::Bool.id())
                .u64(2)
                .u8(1)
                .u8(255);
            w.0
        }),
        ("key not UTF-8", {
            let mut w = W::default();
            w.preamble(3, 0, 1).str(b"\xc3\x28").u32(0).u8(0);
            w.0
        }),
        ("tensor name not UTF-8", {
            let mut w = W::default();
            w.preamble(3, 1, 0)
                .str(b"\xff")
                .u32(1)
                .u64(4)
                .u32(f32_id)
                .u64(0);
            w.0
        }),
        ("duplicate tensor", {
            Synth::new(3)
                .tensor("a", &[4], GgmlType::F32)
                .tensor("a", &[4], GgmlType::F32)
                .build()
        }),
        (
            "five dimensions",
            one_tensor(&[1, 1, 1, 1, 1], f32_id, 0, 4),
        ),
        ("negative dimension", one_tensor(&[u64::MAX], f32_id, 0, 4)),
        ("partial block", one_tensor(&[255], q4k, 0, 1024)),
        (
            "element count past 2^63",
            one_tensor(&[1 << 32, 1 << 31], f32_id, 0, 0),
        ),
        ("misaligned offset", one_tensor(&[4], f32_id, 16, 64)),
        ("first tensor not at 0", one_tensor(&[4], f32_id, 32, 64)),
        (
            "data past the end of the file",
            one_tensor(&[64], f32_id, 0, 255),
        ),
        ("overlap", {
            let mut w = W::default();
            w.preamble(3, 2, 0)
                .tensor_info("a", &[16], f32_id, 0)
                .tensor_info("b", &[16], f32_id, 32);
            w.0.resize(4096, 0);
            w.0
        }),
        ("gap", {
            let mut w = W::default();
            w.preamble(3, 2, 0)
                .tensor_info("a", &[16], f32_id, 0)
                .tensor_info("b", &[16], f32_id, 128);
            w.0.resize(4096, 0);
            w.0
        }),
        ("out of order", {
            let mut w = W::default();
            w.preamble(3, 2, 0)
                .tensor_info("a", &[16], f32_id, 64)
                .tensor_info("b", &[16], f32_id, 0);
            w.0.resize(4096, 0);
            w.0
        }),
        ("counts larger than the file", {
            let mut w = W::default();
            w.preamble(3, 5, 5);
            w.0
        }),
    ];
    for (what, bytes) in cases {
        let e = parse(bytes).expect_err(what);
        assert!(
            matches!(e, GgufError::HeaderInvalid { .. }),
            "{what}: {e:?}"
        );
        assert_eq!(
            (e.code(), e.class()),
            (Some(6), ErrorClass::Integrity),
            "{what}"
        );
    }

    // The control: the same shapes laid out correctly parse.
    parse(one_tensor(&[64], f32_id, 0, 256)).unwrap();
    parse(one_tensor(&[256], q4k, 0, 144)).unwrap();
    parse(one_tensor(&[], f32_id, 0, 4)).unwrap();
    parse(one_tensor(&[0, 7], f32_id, 0, 0)).unwrap();
}

#[test]
fn every_truncation_of_a_valid_file_is_header_invalid() {
    let s = sample(3);
    let file = s.build();
    let header_len = s.header().len();
    for cut in 0..file.len() {
        let e = parse(file[..cut].to_vec()).expect_err("a truncated file parsed");
        assert_eq!(e.code(), Some(6), "cut at {cut}: {e}");
        if cut >= header_len {
            // Past the header the reader still refuses: the tensors' bytes are missing.
            assert!(
                matches!(e, GgufError::HeaderInvalid { offset, .. } if offset == header_len as u64)
            );
        }
    }
    parse(file).unwrap();
}

// ---- MetaView --------------------------------------------------------------------------------

#[test]
fn meta_view_accessors_widen_exactly_and_refuse_mismatches() {
    let h = parse(sample(3).build()).unwrap();
    let m = &h.metadata;
    assert_eq!(m.architecture().unwrap(), Some("fixture"));
    assert_eq!(m.u32("fixture.block_count").unwrap(), Some(2));
    assert_eq!(m.u64("fixture.context_length").unwrap(), Some(4096));
    assert_eq!(m.u64("fixture.u8").unwrap(), Some(7));
    assert_eq!(m.u64("fixture.u16").unwrap(), Some(700));
    assert_eq!(m.i64("fixture.i8").unwrap(), Some(-7));
    assert_eq!(m.i64("fixture.i64").unwrap(), Some(-7_000_000_000));
    assert_eq!(m.i64("fixture.block_count").unwrap(), Some(2));
    assert_eq!(m.f64("fixture.rope.freq_base").unwrap(), Some(10_000.0));
    assert_eq!(m.f64("fixture.eps").unwrap(), Some(1e-6));
    assert_eq!(m.bool("fixture.flag").unwrap(), Some(true));
    assert_eq!(m.array("tokenizer.ggml.scores").unwrap().unwrap().len(), 3);
    assert!(m.array("fixture.empty").unwrap().unwrap().is_empty());
    assert_eq!(
        m.array("tokenizer.ggml.tokens")
            .unwrap()
            .unwrap()
            .element_type(),
        ValueType::Str
    );
    assert_eq!(m.u64("absent").unwrap(), None);
    assert_eq!(m.str("absent").unwrap(), None);

    let e = m.u64("fixture.i8").unwrap_err();
    assert_eq!(e.key, "fixture.i8");
    assert!(e.to_string().contains("-7"), "{e}");
    assert!(m.u32("fixture.i64").is_err());
    assert!(m.u64("general.architecture").is_err());
    assert!(m.f64("fixture.block_count").is_err());
    assert!(m.str("fixture.flag").is_err());
    assert!(m.bool("fixture.u8").is_err());
    assert!(m.array("fixture.u8").is_err());

    let s = Synth::new(3)
        .kv("big", Value::U64(u64::MAX))
        .kv("bad", Value::Str(b"\xff"[..].into()));
    let h = parse(s.build()).unwrap();
    assert!(h.metadata.i64("big").is_err());
    assert!(h.metadata.u32("big").is_err());
    assert_eq!(h.metadata.u64("big").unwrap(), Some(u64::MAX));
    assert!(h.metadata.str("bad").is_err());
    assert_eq!(h.metadata.get("bad"), Some(&Value::Str(b"\xff"[..].into())));
}

// ---- Split files -----------------------------------------------------------------------------

#[test]
fn split_names_parse_and_format() {
    let n = SplitName::parse("model-q4-00002-of-00003.gguf").unwrap();
    assert_eq!(
        n,
        SplitName {
            prefix: "model-q4".into(),
            no: 2,
            count: 3
        }
    );
    assert_eq!(
        n.file_names(),
        [
            "model-q4-00001-of-00003.gguf",
            "model-q4-00002-of-00003.gguf",
            "model-q4-00003-of-00003.gguf"
        ]
    );
    for bad in [
        "model.gguf",
        "model-00001-of-00003.bin",
        "model-00000-of-00003.gguf",
        "model-00004-of-00003.gguf",
        "model-0001-of-00003.gguf",
        "model-00001-af-00003.gguf",
        "-00001-of-00003.gguf",
        "00001-of-00003.gguf",
        "model-0000a-of-00003.gguf",
        "model-99999-of-70000.gguf",
        "é-00001-of-00001.gguf\u{0}",
    ] {
        assert_eq!(SplitName::parse(bad), None, "{bad}");
    }
    assert!(SplitName::parse("é-00001-of-00001.gguf").is_some());
}

/// A split set of `count` files with `per` tensors each.
fn split_set(count: u16, per: usize) -> Vec<Synth> {
    let total = i32::from(count) * per as i32;
    (0..count)
        .map(|i| {
            let mut s = Synth::new(3)
                .kv(SPLIT_NO, Value::U16(i))
                .kv(SPLIT_COUNT, Value::U16(count))
                .kv(SPLIT_TENSORS_COUNT, Value::I32(total));
            if i == 0 {
                s = s.kv("general.architecture", str_val("fixture"));
            }
            for j in 0..per {
                s = s.tensor(&format!("blk.{i}.t{j}.weight"), &[32, 2], GgmlType::Q8_0);
            }
            s
        })
        .collect()
}

#[test]
fn a_split_set_reads_every_header_and_tags_each_tensor_with_its_file() {
    let set = split_set(3, 2);
    let m = GgufModel::read(set.iter().map(|s| Mem(s.build())), &Limits::DEFAULT).unwrap();
    assert_eq!(m.files().len(), 3);
    assert_eq!(m.metadata().architecture().unwrap(), Some("fixture"));
    let tensors: Vec<&GgufTensor> = m.tensors().collect();
    assert_eq!(tensors.len(), 6);
    for (i, (s, h)) in set.iter().zip(m.files()).enumerate() {
        let want = expected_tensors(s, h.data_start, i as u16);
        assert_eq!(h.tensors, want);
    }

    // A single file without split keys is a set of one.
    let one = GgufModel::read([Mem(sample(3).build())], &Limits::DEFAULT).unwrap();
    assert_eq!(one.tensors().count(), 4);
    assert!(one.tensors().all(|t| t.file == 0));
}

#[test]
fn split_sets_that_do_not_agree_are_header_invalid() {
    let build =
        |set: &[Synth]| GgufModel::read(set.iter().map(|s| Mem(s.build())), &Limits::DEFAULT);
    let is_6 = |r: Result<GgufModel, GgufError>, what: &str| {
        let e = r.expect_err(what);
        assert_eq!(
            (e.code(), e.class()),
            (Some(6), ErrorClass::Integrity),
            "{what}: {e}"
        );
    };

    let set = split_set(3, 2);
    is_6(build(&set[..2]), "a missing split");
    is_6(
        build(&[set[1].clone(), set[0].clone(), set[2].clone()]),
        "out of order",
    );
    is_6(
        build(&[set[0].clone(), set[0].clone(), set[2].clone()]),
        "a split twice",
    );
    is_6(
        build(&[sample(3), sample(3)]),
        "two files without split keys",
    );
    is_6(build(&[]), "no files");

    let mut dup = split_set(2, 1);
    dup[1].tensors[0].name = dup[0].tensors[0].name.clone();
    is_6(build(&dup), "a name in two splits");

    let mut wrong_total = split_set(2, 1);
    wrong_total[1].kvs[2].1 = Value::I32(3);
    is_6(build(&wrong_total), "split.tensors.count");

    let mut wrong_type = split_set(2, 1);
    wrong_type[0].kvs[0].1 = str_val("0");
    is_6(build(&wrong_type), "split.no as a string");

    // An error in one file names the file.
    let mut bad = split_set(2, 1);
    bad[1].version = 1;
    match build(&bad).unwrap_err() {
        GgufError::InFile { file, error } => {
            assert_eq!(file, 1);
            assert_eq!(error.code(), Some(2));
        }
        other => panic!("{other:?}"),
    }
}

// ---- Recorded Hub fixtures -------------------------------------------------------------------

mod fixtures {
    use std::path::PathBuf;

    use super::*;
    use crate::range::FileSource;
    use crate::range::replay::{Recorder, Recording, Replay};

    /// The first request reads 64 bytes, so each fixture header takes several requests.
    const INITIAL: u64 = 64;

    fn data(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/gguf")
            .join(name)
    }

    /// Each fixture file and the generator that makes it.
    fn files() -> Vec<(String, Vec<u8>)> {
        let mut out = vec![("tiny-v3.gguf".to_owned(), sample(3).build())];
        let set = split_set(2, 2);
        let name = SplitName {
            prefix: "tiny-split".into(),
            no: 1,
            count: 2,
        };
        for (n, s) in name.file_names().into_iter().zip(&set) {
            out.push((n, s.build()));
        }
        out
    }

    fn reader<S: RangeSource>(s: S) -> HeaderReader<S> {
        HeaderReader::new(s, Limits::DEFAULT.header_bytes).with_initial_read(INITIAL)
    }

    fn record(name: &str) -> Recording {
        let file = FileSource::open(data(name)).unwrap();
        let recorder = Recorder::new(&file, format!("synthetic:{name}"));
        GgufHeader::read_from(&mut reader(&recorder), &Limits::DEFAULT).unwrap();
        recorder.recording()
    }

    fn rec_name(name: &str) -> String {
        format!("{name}.hubrec")
    }

    fn replay(name: &str) -> Replay {
        Replay::parse(&std::fs::read_to_string(data(&rec_name(name))).unwrap()).unwrap()
    }

    #[test]
    fn fixtures_are_the_generator_output_and_its_recording() {
        let bless = std::env::var_os("SYNTROPY_BLESS").is_some();
        for (name, bytes) in files() {
            if bless {
                std::fs::create_dir_all(data("")).unwrap();
                std::fs::write(data(&name), &bytes).unwrap();
                let text = format!(
                    "# SPDX-License-Identifier: MIT OR Apache-2.0\n# Synthetic: made by gguf::tests, not recorded from the network. See src/range/replay.rs for the format.\n{}",
                    record(&name).to_text()
                );
                std::fs::write(data(&rec_name(&name)), text).unwrap();
            }
            assert_eq!(std::fs::read(data(&name)).unwrap(), bytes, "{name}");
            assert_eq!(replay(&name).recording(), &record(&name), "{name}");
        }
    }

    #[test]
    fn the_reader_gives_identical_results_from_a_local_file_and_its_hub_fixture() {
        for (name, _) in files() {
            let file = FileSource::open(data(&name)).unwrap();
            let hub = replay(&name);
            let mut a = reader(&file);
            let mut b = reader(&hub);
            let from_file = GgufHeader::read_from(&mut a, &Limits::DEFAULT).unwrap();
            let from_hub = GgufHeader::read_from(&mut b, &Limits::DEFAULT).unwrap();
            assert_eq!(from_file, from_hub, "{name}");
            assert_eq!(a.requests(), b.requests(), "{name}");
            assert!(
                a.requests() > 1,
                "{name}: the fixture should take several requests"
            );
        }

        // The split set too, through the set reader.
        let names = &files()[1..];
        let mut local: Vec<_> = names
            .iter()
            .map(|(n, _)| reader(FileSource::open(data(n)).unwrap()))
            .collect();
        let mut hub: Vec<_> = names.iter().map(|(n, _)| reader(replay(n))).collect();
        let a = GgufModel::read_from(&mut local, &Limits::DEFAULT).unwrap();
        let b = GgufModel::read_from(&mut hub, &Limits::DEFAULT).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.tensors().count(), 4);
    }
}

// ---- Structured fuzzing ----------------------------------------------------------------------

fn value_strategy() -> impl Strategy<Value = Value> {
    let bytes = || proptest::collection::vec(any::<u8>(), 0..12);
    let finite32 = || -1.0e6f32..1.0e6;
    let finite64 = || -1.0e12f64..1.0e12;
    let n = || 0usize..6;
    prop_oneof![
        any::<u8>().prop_map(Value::U8),
        any::<i8>().prop_map(Value::I8),
        any::<u16>().prop_map(Value::U16),
        any::<i16>().prop_map(Value::I16),
        any::<u32>().prop_map(Value::U32),
        any::<i32>().prop_map(Value::I32),
        finite32().prop_map(Value::F32),
        any::<bool>().prop_map(Value::Bool),
        bytes().prop_map(|b| Value::Str(b.into())),
        any::<u64>().prop_map(Value::U64),
        any::<i64>().prop_map(Value::I64),
        finite64().prop_map(Value::F64),
        proptest::collection::vec(any::<u8>(), n()).prop_map(|v| Value::Array(Array::U8(v))),
        proptest::collection::vec(any::<i16>(), n()).prop_map(|v| Value::Array(Array::I16(v))),
        proptest::collection::vec(any::<u32>(), n()).prop_map(|v| Value::Array(Array::U32(v))),
        proptest::collection::vec(finite32(), n()).prop_map(|v| Value::Array(Array::F32(v))),
        proptest::collection::vec(any::<bool>(), n()).prop_map(|v| Value::Array(Array::Bool(v))),
        proptest::collection::vec(bytes(), n())
            .prop_map(|v| Value::Array(Array::Str(v.into_iter().map(Into::into).collect()))),
        proptest::collection::vec(any::<i64>(), n()).prop_map(|v| Value::Array(Array::I64(v))),
        proptest::collection::vec(finite64(), n()).prop_map(|v| Value::Array(Array::F64(v))),
    ]
}

fn tensor_strategy() -> impl Strategy<Value = (GgmlType, Vec<u64>)> {
    (
        proptest::sample::select(GgmlType::ALL.to_vec()),
        1u64..4,
        proptest::collection::vec(1u64..4, 0..4),
    )
        .prop_map(|(t, blocks, rest)| {
            let (block, _) = block_geometry(t).unwrap();
            let mut ne = vec![block * blocks];
            ne.extend(rest);
            (t, ne)
        })
}

prop_compose! {
    fn synth_strategy()(
        version in 2u32..4,
        align in proptest::option::of(proptest::sample::select(vec![1u32, 8, 32, 64, 256])),
        kvs in proptest::collection::vec(value_strategy(), 0..8),
        tensors in proptest::collection::vec(tensor_strategy(), 0..6),
    ) -> Synth {
        let mut s = Synth::new(version);
        if let Some(a) = align {
            s = s.kv(ALIGNMENT_KEY, Value::U32(a));
        }
        for (i, v) in kvs.into_iter().enumerate() {
            s = s.kv(&format!("fuzz.key.{i}"), v);
        }
        s.tensors = tensors
            .into_iter()
            .enumerate()
            .map(|(i, (ggml_type, ne))| TensorSpec { name: format!("fuzz.t{i}"), ne, ggml_type })
            .collect();
        s
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Any file the generator writes parses back to exactly what it wrote.
    #[test]
    fn generated_files_parse_to_what_was_written(s in synth_strategy()) {
        let h = parse(s.build()).unwrap();
        check_matches(&s, &h);
    }

    /// Hostile input: random edits to a valid file never panic, and every refusal carries
    /// one of the reader's codes.
    #[test]
    fn mutated_files_never_panic_and_refuse_with_a_code(
        s in synth_strategy(),
        edits in proptest::collection::vec((any::<proptest::sample::Index>(), any::<u8>()), 1..6),
        cut in proptest::option::of(any::<proptest::sample::Index>()),
    ) {
        let mut b = s.build();
        let header_len = s.header().len();
        for (at, byte) in edits {
            let i = at.index(header_len);
            b[i] = byte;
        }
        if let Some(c) = cut {
            let len = c.index(b.len() + 1);
            b.truncate(len);
        }
        if let Err(e) = parse(b) {
            prop_assert!(matches!(e.code(), Some(2 | 4 | 5 | 6)), "{e}");
        }
    }

    /// Hostile input: arbitrary bytes after a valid preamble never panic.
    #[test]
    fn random_bytes_never_panic(
        version in 2u32..4,
        n_tensors in 0u64..8,
        n_kv in 0u64..8,
        tail in proptest::collection::vec(any::<u8>(), 0..512),
    ) {
        let mut w = W::default();
        w.preamble(version, n_tensors, n_kv).bytes(&tail);
        if let Err(e) = parse(w.0) {
            prop_assert!(matches!(e.code(), Some(2 | 4 | 5 | 6)), "{e}");
        }
    }
}

// ---- Differential corpus ---------------------------------------------------------------------

/// Writes a corpus for a differential check against ggml's own GGUF reader, which this crate
/// does not link (only `syntropy-kernels-ggml` links ggml). Run with
/// `SYNTROPY_GGUF_CORPUS=<dir> cargo test -p syntropy-manifest -- --ignored write_differential_corpus`.
/// For each case it writes `case-NNNN.gguf` and `case-NNNN.txt`, the reader's verdict in
/// [`dump`]'s format.
#[test]
#[ignore = "writes a corpus for an out-of-tree differential check"]
fn write_differential_corpus() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

    let Some(dir) = std::env::var_os("SYNTROPY_GGUF_CORPUS") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut runner = TestRunner::new_with_rng(
        Config::default(),
        TestRng::from_seed(RngAlgorithm::ChaCha, &[7; 32]),
    );
    let strategy = (
        synth_strategy(),
        proptest::collection::vec((any::<proptest::sample::Index>(), any::<u8>()), 0..3),
    );
    for i in 0..2000 {
        let (s, edits) = strategy.new_tree(&mut runner).unwrap().current();
        let mut b = s.build();
        // Half the cases are mutated, to compare verdicts on malformed input too.
        if i % 2 == 1 {
            let header_len = s.header().len();
            for (at, byte) in edits {
                let j = at.index(header_len);
                b[j] = byte;
            }
        }
        std::fs::write(dir.join(format!("case-{i:04}.gguf")), &b).unwrap();
        std::fs::write(dir.join(format!("case-{i:04}.txt")), dump(&b)).unwrap();
    }
}

/// The reader's verdict on `bytes` in the differential format: `OK` and one line per tensor
/// (name, ggml type ID, `ne` padded to four dimensions with 1, data offset), or `REJECT`, the
/// code and the message.
fn dump(bytes: &[u8]) -> String {
    let mut out = String::new();
    match parse(bytes.to_vec()) {
        Ok(h) => {
            out.push_str("OK\n");
            for t in &h.tensors {
                let mut ne = t.ne.clone();
                ne.resize(4, 1);
                out.push_str(&format!(
                    "{} {} {} {} {} {} {}\n",
                    t.name,
                    ggml_type_id(t.ggml_type).unwrap(),
                    ne[0],
                    ne[1],
                    ne[2],
                    ne[3],
                    t.offset
                ));
            }
        }
        Err(e) => out.push_str(&format!("REJECT {} {e}\n", e.code().unwrap_or(0))),
    }
    out
}

/// The reverse direction: dumps every `.gguf` file in `SYNTROPY_GGUF_DUMP` (for example files
/// written by ggml's own GGUF writer) to a `.txt` next to it, in [`dump`]'s format. Run with
/// `SYNTROPY_GGUF_DUMP=<dir> cargo test -p syntropy-manifest -- --ignored dump_gguf_files`.
#[test]
#[ignore = "dumps files for an out-of-tree differential check"]
fn dump_gguf_files() {
    let Some(dir) = std::env::var_os("SYNTROPY_GGUF_DUMP") else {
        return;
    };
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "gguf") {
            let out = dump(&std::fs::read(&path).unwrap());
            std::fs::write(path.with_extension("txt"), out).unwrap();
        }
    }
}
