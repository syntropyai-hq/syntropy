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

/// A safetensors checkpoint with a stacked expert tensor and its scale tensor, an auxiliary
/// extent: each tensor's and each expert's start is a boundary.
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
    let slices = safetensors_experts(stack, 0, experts as u16).unwrap();
    let chunks = safetensors_chunks(&header, &slices).unwrap();
    let cuts: Vec<u64> = header
        .tensors
        .iter()
        .map(|t| t.offset)
        .chain(slices.iter().map(|e| e.extent.offset))
        .collect();
    let expected = check_tiling(&chunks, file_len, header.data_start, &cuts);
    assert_eq!(chunks.len() as u64, expected);
    // One header chunk; each 3 MiB expert one chunk; the scale one chunk.
    assert_eq!(chunks.len(), 1 + 4 + 1);
    let scale = header
        .tensor("model.layers.0.mlp.experts.down_proj_scale")
        .unwrap();
    assert_eq!(chunks.last().unwrap().offset, scale.offset);
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
    let bad = |r: Result<ChunkTable, ChunkError>, why: &str| {
        assert_eq!(
            r,
            Err(ChunkError::BadTable(match r.clone() {
                Err(ChunkError::BadTable(w)) => w,
                other => panic!("{why}: {other:?}"),
            })),
            "{why}"
        );
    };
    bad(edit(0, b'X'), "magic");
    bad(edit(8, 2), "version");
    bad(edit(13, 1), "reserved");
    bad(ChunkTable::decode(&good[..good.len() - 1]), "length");
    bad(ChunkTable::decode(&good[..10]), "short");
    bad(edit(HEADER_BYTES + 12, 0x10), "an undefined flag bit");
    bad(edit(HEADER_BYTES + 12, 0x04), "evidence without a commit");
    bad(edit(HEADER_BYTES + 20, 0x01), "an ID without a commit");
    let mut zero = good.clone();
    zero[HEADER_BYTES + 8..HEADER_BYTES + 12].copy_from_slice(&0u32.to_le_bytes());
    bad(ChunkTable::decode(&zero), "a zero length");
    bad(edit(HEADER_BYTES + RECORD_BYTES, 1), "a gap");
    let mut short = good.clone();
    short[48..56].copy_from_slice(&1001u64.to_le_bytes());
    bad(ChunkTable::decode(&short), "records short of the file");
}
