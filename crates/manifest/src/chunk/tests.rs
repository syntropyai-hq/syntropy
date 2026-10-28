// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of chunk boundaries and the table format (COMP-E12-T01; RFC-0041 §6.1, §6.2).
//!
//! The files are made by the crate's synthetic GGUF writer ([`crate::gguf::synth`]) or written as
//! safetensors headers here. [`HeaderOnly`] serves a file's real header and a fill byte for every
//! byte after it, so two of them with different fills differ in nothing but their data: equal
//! chunks from both show the boundaries depend on the header alone.

use proptest::prelude::*;
use syntropy_core::GgmlType;

use super::*;
use crate::gguf::synth::Synth;
use crate::gguf::{Limits, Value};
use crate::range::{RangeError, RangeSource, check_bounds};
use crate::safetensors::parse_header;
use crate::slice::{gguf_experts, safetensors_experts};

/// A file's header, then `fill` for every byte after it.
struct HeaderOnly {
    header: Vec<u8>,
    len: u64,
    fill: u8,
}

impl RangeSource for HeaderOnly {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        check_bounds(offset, buf.len() as u64, self.len)?;
        for (i, b) in buf.iter_mut().enumerate() {
            *b = self
                .header
                .get(offset as usize + i)
                .copied()
                .unwrap_or(self.fill);
        }
        Ok(())
    }
}

/// A whole file in memory.
struct Mem(Vec<u8>);

impl RangeSource for Mem {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        check_bounds(offset, buf.len() as u64, self.len())?;
        buf.copy_from_slice(&self.0[offset as usize..offset as usize + buf.len()]);
        Ok(())
    }
}

/// The file `s` would build, as its length: the data section's start plus its last tensor's end.
fn file_len(s: &Synth) -> u64 {
    let data_start = (s.header().len() as u64).next_multiple_of(s.alignment());
    data_start + s.layout().last().map_or(0, |(o, b)| o + b)
}

fn header_only(s: &Synth, fill: u8) -> GgufHeader {
    GgufHeader::read(
        HeaderOnly {
            header: s.header(),
            len: file_len(s),
            fill,
        },
        &Limits::DEFAULT,
    )
    .unwrap()
}

/// `⌈a / b⌉`.
fn ceil_div(a: u64, b: u64) -> u64 {
    a.div_ceil(b)
}

/// Checks that `chunks` tile `[0, file_len)` and that each forced boundary in `cuts` is a chunk's
/// start; returns `Σ ⌈run / 4 MiB⌉` over the runs `cuts` makes, computed here.
fn check_tiling(chunks: &[Chunk], file_len: u64, data_start: u64, cuts: &[u64]) -> u64 {
    let mut at = 0;
    for c in chunks {
        assert_eq!(c.offset, at, "chunks are contiguous");
        assert!(c.len > 0 && u64::from(c.len) <= CHUNK_BYTES, "{c:?}");
        assert_eq!(c.header, c.offset < data_start, "{c:?}");
        at = c.end();
    }
    assert_eq!(at, file_len, "chunks cover the file");
    let mut cuts: Vec<u64> = cuts
        .iter()
        .copied()
        .chain([0, data_start, file_len])
        .collect();
    cuts.sort_unstable();
    cuts.dedup();
    for cut in &cuts[..cuts.len() - 1] {
        assert!(
            chunks.iter().any(|c| c.offset == *cut),
            "no chunk starts at {cut}"
        );
    }
    cuts.windows(2)
        .map(|r| ceil_div(r[1] - r[0], CHUNK_BYTES))
        .sum()
}

prop_compose! {
    /// A synthetic GGUF of dense tensors up to about 12 MiB and one stacked expert tensor.
    fn synth_strategy()(
        alignment in prop::sample::select(vec![32u32, 64, 256, 4096]),
        dense in prop::collection::vec((1u64..=48, 1u64..=256, 0usize..3), 1..5),
        n_expert in 1u64..=6,
        expert_rows in 1u64..=4096,
        experts_at in 0usize..5,
    ) -> (Synth, u16) {
        let types = [GgmlType::F32, GgmlType::F16, GgmlType::Q8_0];
        let mut s = Synth::new(3).kv("general.alignment", Value::U32(alignment));
        for (i, &(cols, rows, t)) in dense.iter().enumerate() {
            if i == experts_at {
                s = s.tensor("blk.0.ffn_up_exps.weight", &[256, expert_rows, n_expert], GgmlType::F32);
            }
            s = s.tensor(&format!("t{i}.weight"), &[cols * 256, rows], types[t]);
        }
        if experts_at >= dense.len() {
            s = s.tensor("blk.0.ffn_up_exps.weight", &[256, expert_rows, n_expert], GgmlType::F32);
        }
        (s, n_expert as u16)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Boundaries depend only on the header; chunks tile the file; the record count is
    /// `Σ over boundary runs of ⌈run / 4 MiB⌉`; every tensor and expert start is a boundary.
    #[test]
    fn chunks_come_from_the_header_and_tile_the_file((s, n_expert) in synth_strategy()) {
        let header = header_only(&s, 0x00);
        let stack = header.tensors().find(|t| t.name == "blk.0.ffn_up_exps.weight").unwrap();
        let experts = gguf_experts(&stack, n_expert).unwrap();
        let chunks = gguf_chunks(&header, &experts).unwrap();
        // The forced boundaries, from the tensors and slices directly, not from `gguf_starts`.
        let cuts: Vec<u64> = header
            .tensors()
            .map(|t| t.start)
            .chain(experts.iter().map(|e| e.extent.offset))
            .collect();
        let expected = check_tiling(&chunks, header.file_len, header.data_start, &cuts);
        prop_assert_eq!(chunks.len() as u64, expected);
        // The same header over other data bytes gives the same chunks.
        let other = header_only(&s, 0xFF);
        let stack = other
            .tensors()
            .find(|t| t.name == "blk.0.ffn_up_exps.weight")
            .unwrap();
        let other_experts = gguf_experts(&stack, n_expert).unwrap();
        prop_assert_eq!(gguf_chunks(&other, &other_experts).unwrap(), chunks.clone());
        // The table holds one record per chunk and reads back.
        let table = ChunkTable::new(FileDigest([0; 32]), header.file_len, &chunks);
        prop_assert_eq!(table.records.len(), chunks.len());
        prop_assert_eq!(ChunkTable::decode(&table.encode()).unwrap(), table);
    }
}

proptest! {
    // Each case builds two whole files of up to 12.5 MiB, so it crosses the 4 MiB step.
    #![proptest_config(ProptestConfig::with_cases(12))]

    /// A tensor of identical bytes in two files gives identical chunks: the same lengths and the
    /// same bytes, so the same ChunkIds. Its bytes are a whole number of alignment units, so no
    /// padding follows it in either file.
    #[test]
    fn identical_tensors_give_identical_chunks(
        alignment in prop::sample::select(vec![32u32, 256, 4096]),
        rows in 1u64..=800,
        before_a in prop::collection::vec(1u64..=8, 0..3),
        before_b in prop::collection::vec(1u64..=8, 0..3),
        after in any::<bool>(),
    ) {
        let build = |before: &[u64]| {
            let mut s = Synth::new(3).kv("general.alignment", Value::U32(alignment));
            for (i, &cols) in before.iter().enumerate() {
                s = s.tensor(&format!("pre{i}.weight"), &[cols * 32, 3], GgmlType::F16);
            }
            // 4096 × 4 bytes per row: a multiple of every alignment drawn.
            s = s.tensor("shared.weight", &[4096, rows], GgmlType::F32);
            if after {
                s = s.tensor("post.weight", &[64, 2], GgmlType::F32);
            }
            s
        };
        let ids = |s: &Synth| {
            let bytes = s.build();
            let header = GgufHeader::read(Mem(bytes.clone()), &Limits::DEFAULT).unwrap();
            let shared = header.tensors().find(|t| t.name == "shared.weight").unwrap();
            let (from, to) = (shared.start, shared.start + shared.bytes);
            gguf_chunks(&header, &[])
                .unwrap()
                .into_iter()
                .filter(|c| c.offset >= from && c.end() <= to)
                .map(|c| (c.len, *blake3::hash(&bytes[c.offset as usize..c.end() as usize]).as_bytes()))
                .collect::<Vec<_>>()
        };
        let (a, b) = (ids(&build(&before_a)), ids(&build(&before_b)));
        prop_assert_eq!(a.len() as u64, ceil_div(4096 * 4 * rows, CHUNK_BYTES));
        prop_assert_eq!(a, b);
    }
}

#[test]
fn a_run_is_cut_every_4_mib_from_its_own_start() {
    let mib = 1 << 20;
    // Header 0..100, data from 128, tensors at 128 and 128 + 9 MiB, file end at 128 + 10 MiB.
    let chunks = chunk_file(128 + 10 * mib, 128, [128 + 9 * mib, 128]).unwrap();
    let got: Vec<(u64, u32, bool)> = chunks.iter().map(|c| (c.offset, c.len, c.header)).collect();
    assert_eq!(
        got,
        vec![
            (0, 128, true),
            (128, 4 << 20, false),
            (128 + 4 * mib, 4 << 20, false),
            (128 + 8 * mib, 1 << 20, false),
            (128 + 9 * mib, 1 << 20, false),
        ]
    );
    // Padding before the next start stays with the chunk before it.
    let padded = chunk_file(4096 + 300, 64, [64, 4096]).unwrap();
    assert_eq!(
        padded.iter().map(|c| (c.offset, c.len)).collect::<Vec<_>>(),
        vec![(0, 64), (64, 4032), (4096, 300)]
    );
}

#[test]
fn edge_files_and_bad_starts() {
    assert_eq!(chunk_file(0, 0, []).unwrap(), vec![]);
    assert_eq!(
        chunk_file(40, 40, [40]).unwrap(),
        vec![Chunk {
            offset: 0,
            len: 40,
            header: true
        }],
        "a header-only file, and an empty tensor at the end"
    );
    assert_eq!(
        chunk_file(10, 20, []),
        Err(ChunkError::DataPastEnd {
            data_start: 20,
            file_len: 10
        })
    );
    let err = chunk_file(100, 20, [10]).unwrap_err();
    assert_eq!(
        err,
        ChunkError::StartOutOfRange {
            start: 10,
            data_start: 20,
            file_len: 100
        }
    );
    assert_eq!(err.code(), 6);
    assert!(chunk_file(100, 20, [101]).is_err());
}

/// A safetensors checkpoint with a stacked expert tensor and its stacked scale tensor. Each
/// expert's auxiliary extent is its slice of the scale tensor (RFC-0041 §2.1, §2.2), so each
/// expert's weight slice and scale slice start a chunk.
#[test]
fn safetensors_boundaries_include_experts_and_auxiliary_extents() {
    let experts = 4u64;
    let expert_bytes = 3 * (1 << 20);
    let stack_len = experts * expert_bytes;
    let rows = expert_bytes / (512 * 4);
    let json = format!(
        "{{\"model.layers.0.mlp.experts.down_proj\":{{\"dtype\":\"F32\",\"shape\":[{experts},{rows},512],\
         \"data_offsets\":[0,{stack_len}]}},\
         \"model.layers.0.mlp.experts.down_proj_scale\":{{\"dtype\":\"F32\",\"shape\":[{experts}],\
         \"data_offsets\":[{stack_len},{}]}}}}",
        stack_len + 4 * experts
    );
    let file_len = 8 + json.len() as u64 + stack_len + 4 * experts;
    let header = parse_header(json.as_bytes(), file_len).unwrap();
    let stack = header
        .tensor("model.layers.0.mlp.experts.down_proj")
        .unwrap();
    let scale = header
        .tensor("model.layers.0.mlp.experts.down_proj_scale")
        .unwrap();
    let mut slices = safetensors_experts(stack, 0, experts as u16).unwrap();
    slices.extend(safetensors_experts(scale, 0, experts as u16).unwrap());
    let chunks = safetensors_chunks(&header, &slices).unwrap();
    let cuts: Vec<u64> = header
        .tensors
        .iter()
        .map(|t| t.offset)
        .chain(slices.iter().map(|e| e.extent.offset))
        .collect();
    let expected = check_tiling(&chunks, file_len, header.data_start, &cuts);
    assert_eq!(chunks.len() as u64, expected);
    // One header chunk; each 3 MiB expert one chunk; each expert's 4-byte scale slice one chunk.
    assert_eq!(chunks.len(), 1 + 4 + 4);
    for e in 0..experts {
        let at = scale.offset + 4 * e;
        let chunk = chunks.iter().find(|c| c.offset == at).unwrap();
        assert_eq!(chunk.len, 4, "expert {e}'s scale");
    }
}

/// The table's bytes, written out field by field from RFC-0041 §6.2: the 64-byte header, then
/// three 48-byte records (a header chunk, a committed chunk with witnesses, a committed chunk a
/// keyholder verified).
#[test]
fn the_table_encoding_is_golden() {
    let table = ChunkTable {
        digest: FileDigest([0x11; 32]),
        file_len: 4_194_404,
        records: vec![
            ChunkRecord {
                chunk: Chunk {
                    offset: 0,
                    len: 64,
                    header: true,
                },
                committed: None,
            },
            ChunkRecord {
                chunk: Chunk {
                    offset: 64,
                    len: 4_194_304,
                    header: false,
                },
                committed: Some((Blake3([0xAB; 32]), EvidenceForm::Witnesses)),
            },
            ChunkRecord {
                chunk: Chunk {
                    offset: 4_194_368,
                    len: 36,
                    header: false,
                },
                committed: Some((Blake3([0xCD; 32]), EvidenceForm::KeyholderVerified)),
            },
        ],
    };
    let golden = [
        // Header: magic, version 1, reserved, FileDigest, file length 0x400064, 3 entries.
        "53594e43484e4b31",
        "01000000",
        "00000000",
        &"11".repeat(32),
        "6400400000000000",
        "0300000000000000",
        // Record 0: offset 0, length 64, flags 0b10 (header region), no ID.
        "0000000000000000",
        "40000000",
        "02000000",
        &"00".repeat(32),
        // Record 1: offset 64, length 0x400000, flags 1 | 2 << 2 (committed, Witnesses).
        "4000000000000000",
        "00004000",
        "09000000",
        &"ab".repeat(32),
        // Record 2: offset 0x400040, length 36, flags 1 | 3 << 2 (committed, KeyholderVerified).
        "4000400000000000",
        "24000000",
        "0d000000",
        &"cd".repeat(32),
    ]
    .concat();
    let bytes = table.encode();
    assert_eq!(bytes.len(), HEADER_BYTES + 3 * RECORD_BYTES);
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, golden);
    assert_eq!(ChunkTable::decode(&bytes).unwrap(), table);
    // An origin-hashes commit is evidence form 1, and a provisional commit sets bit 0 only.
    for (form, flags) in [
        (EvidenceForm::OriginHashes, 5u32),
        (EvidenceForm::Provisional, 1),
    ] {
        let r = ChunkRecord {
            chunk: Chunk {
                offset: 0,
                len: 1,
                header: false,
            },
            committed: Some((Blake3([1; 32]), form)),
        };
        assert_eq!(r.encode()[12..16], flags.to_le_bytes());
        assert_eq!(ChunkRecord::decode(&r.encode()).unwrap(), r);
    }
}

#[test]
fn a_bad_table_is_refused() {
    let chunks = chunk_file(1000, 100, [100, 400]).unwrap();
    let good = ChunkTable::new(FileDigest([2; 32]), 1000, &chunks).encode();
    assert!(ChunkTable::decode(&good).is_ok());
    let edit = |at: usize, byte: u8| {
        let mut b = good.clone();
        b[at] = byte;
        ChunkTable::decode(&b)
    };
    // Each refusal names its reason.
    let bad = |r: Result<ChunkTable, ChunkError>, reason: &'static str| {
        assert_eq!(r, Err(ChunkError::BadTable(reason)), "{reason}");
    };
    bad(edit(0, b'X'), "wrong magic");
    bad(edit(8, 2), "unknown version");
    bad(edit(13, 1), "the reserved word is not zero");
    bad(
        ChunkTable::decode(&good[..good.len() - 1]),
        "the length disagrees with the entry count",
    );
    bad(ChunkTable::decode(&good[..10]), "shorter than its header");
    let mut huge = good.clone();
    huge[56..64].copy_from_slice(&u64::MAX.to_le_bytes());
    bad(
        ChunkTable::decode(&huge),
        "the length disagrees with the entry count",
    );
    bad(
        edit(HEADER_BYTES + 12, 0x10),
        "a record sets an undefined flag bit",
    );
    bad(
        edit(HEADER_BYTES + 12, 0x04),
        "an uncommitted record has an evidence form or an ID",
    );
    bad(
        edit(HEADER_BYTES + 20, 0x01),
        "an uncommitted record has an evidence form or an ID",
    );
    let mut zero = good.clone();
    zero[HEADER_BYTES + 8..HEADER_BYTES + 12].copy_from_slice(&0u32.to_le_bytes());
    bad(ChunkTable::decode(&zero), "a record has length 0");
    bad(
        edit(HEADER_BYTES + RECORD_BYTES, 1),
        "the records do not tile the file",
    );
    let mut past = good.clone();
    let last = HEADER_BYTES + 2 * RECORD_BYTES;
    past[last + 8..last + 12].copy_from_slice(&1000u32.to_le_bytes());
    bad(ChunkTable::decode(&past), "a record runs past the file");
    let mut short = good.clone();
    short[48..56].copy_from_slice(&1001u64.to_le_bytes());
    bad(
        ChunkTable::decode(&short),
        "the records do not cover the file",
    );
}

/// A 5 MiB record in a 12 MiB file tiles it, ends inside it and is otherwise well formed: only the
/// 4 MiB bound refuses it (review N1 on #1945).
#[test]
fn a_record_over_4_mib_is_refused_even_when_it_tiles_the_file() {
    let mib = 1u32 << 20;
    let records = [5 * mib, 4 * mib, 3 * mib];
    let mut at = 0u64;
    let table = ChunkTable {
        digest: FileDigest([3; 32]),
        file_len: 12 << 20,
        records: records
            .iter()
            .map(|&len| {
                let chunk = Chunk {
                    offset: at,
                    len,
                    header: false,
                };
                at += u64::from(len);
                ChunkRecord {
                    chunk,
                    committed: None,
                }
            })
            .collect(),
    };
    assert_eq!(
        ChunkTable::decode(&table.encode()),
        Err(ChunkError::BadTable("a record is longer than 4 MiB"))
    );
    // Exactly 4 MiB is a chunk.
    let ok = chunk_file(12 << 20, 0, []).unwrap();
    assert_eq!(ok.len(), 3);
    assert!(
        ChunkTable::decode(&ChunkTable::new(FileDigest([3; 32]), 12 << 20, &ok).encode()).is_ok()
    );
}

/// A decoded table is used only if it matches the chunks computed from the header (review N2 on
/// #1945).
#[test]
fn a_table_matches_only_the_chunks_it_was_built_from() {
    let chunks = chunk_file(10 << 20, 300, [300, 5 << 20]).unwrap();
    let table =
        ChunkTable::decode(&ChunkTable::new(FileDigest([4; 32]), 10 << 20, &chunks).encode())
            .unwrap();
    assert!(table.matches(&chunks));
    let mut moved = chunks.clone();
    moved[1].len -= 1;
    moved[2].offset -= 1;
    assert!(!table.matches(&moved), "another boundary");
    let mut flag = chunks.clone();
    flag[0].header = false;
    assert!(!table.matches(&flag), "another header flag");
    assert!(!table.matches(&chunks[..chunks.len() - 1]), "a chunk fewer");
    // Other starts give other chunks, so a table from them does not match.
    let other = chunk_file(10 << 20, 300, [300, 6 << 20]).unwrap();
    assert!(!table.matches(&other));
}

// ---- One pass, two hashes (COMP-E11-T14; RFC-0041 §4.3 step 2) ----------------------------------

/// A file in memory that records every read it serves.
struct Reads {
    bytes: Vec<u8>,
    reads: std::sync::Mutex<Vec<(u64, usize)>>,
}

impl RangeSource for Reads {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        check_bounds(offset, buf.len() as u64, self.len())?;
        self.reads.lock().unwrap().push((offset, buf.len()));
        buf.copy_from_slice(&self.bytes[offset as usize..offset as usize + buf.len()]);
        Ok(())
    }
}

fn reads(bytes: Vec<u8>) -> Reads {
    Reads {
        bytes,
        reads: std::sync::Mutex::new(Vec::new()),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// The pass gives the file's SHA-256 and each chunk's BLAKE3, the same as hashing them apart,
    /// and reads every byte exactly once, front to back.
    #[test]
    fn one_read_gives_the_sha_256_and_every_chunk_s_blake3(
        len in 0usize..(9 << 20),
        data_start in 0usize..300,
        cuts in prop::collection::vec(0usize..(9 << 20), 0..12),
        seed in any::<u8>(),
    ) {
        let data_start = data_start.min(len) as u64;
        let bytes: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(167).wrapping_add(seed)).collect();
        let starts: Vec<u64> = cuts.iter().map(|&c| (c.min(len) as u64).max(data_start)).collect();
        let chunks = chunk_file(len as u64, data_start, starts).unwrap();
        let source = reads(bytes.clone());
        let mut seen = 0u64;
        let (digest, ids) = hash_chunks(&source, &chunks, |n| seen += n).unwrap();
        prop_assert_eq!(digest, FileDigest::of_bytes(&bytes));
        prop_assert_eq!(seen, len as u64);
        prop_assert_eq!(ids.len(), chunks.len());
        for (c, id) in chunks.iter().zip(&ids) {
            let want = blake3::hash(&bytes[c.offset as usize..c.end() as usize]);
            prop_assert_eq!(id.0, *want.as_bytes(), "{:?}", c);
        }
        // One sequential pass: each read starts where the last ended, none is over 1 MiB.
        let log = source.reads.lock().unwrap().clone();
        let mut at = 0u64;
        for (offset, n) in log {
            prop_assert_eq!(offset, at);
            prop_assert!(n <= 1 << 20 && n > 0);
            at += n as u64;
        }
        prop_assert_eq!(at, len as u64);
    }
}

#[test]
fn chunks_that_do_not_tile_the_file_are_refused() {
    let source = reads(vec![7; 1000]);
    let whole = chunk_file(1000, 100, [100]).unwrap();
    assert!(hash_chunks(&source, &whole, |_| {}).is_ok());
    let mut gap = whole.clone();
    gap[1].offset += 1;
    let short = &whole[..whole.len() - 1];
    let mut zero = whole.clone();
    zero.push(Chunk {
        offset: 1000,
        len: 0,
        header: false,
    });
    for (name, chunks) in [
        ("a gap", gap.as_slice()),
        ("short", short),
        ("empty", &[][..]),
        ("a zero chunk", zero.as_slice()),
    ] {
        assert_eq!(
            hash_chunks(&source, chunks, |_| {}),
            Err(HashError::NotTiled),
            "{name}"
        );
    }
    // An empty file has no chunks, and hashes to the SHA-256 of nothing.
    let empty = reads(Vec::new());
    let (digest, ids) = hash_chunks(&empty, &[], |_| {}).unwrap();
    assert_eq!(digest, FileDigest::of_bytes(b""));
    assert!(ids.is_empty());
}

/// A read that fails is reported, and the chunk hasher stops.
#[test]
fn a_failed_read_is_reported() {
    struct Fails(u64);
    impl RangeSource for Fails {
        fn len(&self) -> u64 {
            self.0
        }
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
            if offset >= 2 << 20 {
                return Err(RangeError::Io {
                    kind: std::io::ErrorKind::Other,
                    message: "the disk went away".into(),
                });
            }
            buf.fill(1);
            Ok(())
        }
    }
    let chunks = chunk_file(5 << 20, 0, []).unwrap();
    let err = hash_chunks(&Fails(5 << 20), &chunks, |_| {}).unwrap_err();
    assert!(
        matches!(err, HashError::Read(RangeError::Io { .. })),
        "{err:?}"
    );
}
