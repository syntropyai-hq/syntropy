// SPDX-License-Identifier: MIT OR Apache-2.0

//! Chunk boundaries and the chunk-table format (RFC-0041 §6.1, §6.2; ADR-0016 decision 1;
//! COMP-E12-T01).
//!
//! **Boundaries** are a pure function of a file's header, so every node computes the same chunks
//! without talking to anyone ([`chunk_file`]):
//!
//! 1. a boundary at offset 0 and at the start of the data section;
//! 2. a boundary at every tensor start, every auxiliary-extent start and every expert-slice start.
//!    Scale, zero-point and global-scale tensors are tensors of the file, so their starts are
//!    tensor starts; [`gguf_starts`] and [`safetensors_starts`] add the expert slices that
//!    [`slice`](mod@crate::slice) computes;
//! 3. inside a run between two such boundaries, a boundary every [`CHUNK_BYTES`] (4 MiB),
//!    counted from the run's start;
//! 4. padding between tensors belongs to the preceding chunk, so chunks tile the file exactly.
//!
//! A run is cut at the next forced boundary, and padding lies before the next tensor's start,
//! so rule 4 follows from rules 2 and 3. A shard therefore maps to whole chunks with no
//! over-fetch, and identical tensor bytes in two files give identical chunks.
//!
//! **The table** (`tables/<sha256>.chunks`) is a 64-byte header followed by one 48-byte record
//! per chunk, every integer little-endian ([`ChunkTable::encode`]):
//!
//! ```text
//! header   0–7    magic "SYNCHNK1"
//!          8–11   version (u32), 1
//!          12–15  reserved, zero
//!          16–47  FileDigest (SHA-256)
//!          48–55  file length (u64)
//!          56–63  entry count (u64)
//! record   0–7    offset (u64)
//!          8–11   length (u32)
//!          12–15  flags (u32): bit 0 ID committed; bit 1 header region; bits 2–3 evidence form
//!                 (0 provisional, 1 OriginHashes, 2 Witnesses, 3 KeyholderVerified); the rest zero
//!          16–47  ChunkId (BLAKE3), zero until committed
//! ```
//!
//! RFC-0041 §6.2 names the header's fields and their order but not their widths. The widths
//! above, and the reserved word that aligns the digest, are recorded in its note of 2026-10-28.

use std::fmt;

use syntropy_core::Blake3;

use crate::gguf::GgufHeader;
use crate::identity::FileDigest;
use crate::safetensors::SafetensorsHeader;
use crate::slice::ExpertSlice;

/// `chunk_bytes`: the most bytes in one chunk, and the step inside a run (ADR-0016 decision 1).
pub const CHUNK_BYTES: u64 = 4 << 20;

/// The table's magic number.
pub const MAGIC: [u8; 8] = *b"SYNCHNK1";

/// The table format's version.
pub const VERSION: u32 = 1;

/// The table header's length in bytes.
pub const HEADER_BYTES: usize = 64;

/// One record's length in bytes.
pub const RECORD_BYTES: usize = 48;

/// One chunk of a file: where it is and whether it lies in the header region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Chunk {
    /// Its first byte, as an offset into the file.
    pub offset: u64,
    /// Its length: at least 1 and at most [`CHUNK_BYTES`].
    pub len: u32,
    /// Whether it lies before the data section's start.
    pub header: bool,
}

impl Chunk {
    /// One past its last byte.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.offset + u64::from(self.len)
    }
}

/// Why a file's boundaries or a table cannot be built or read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChunkError {
    /// The data section starts past the end of the file.
    DataPastEnd {
        /// The data section's start.
        data_start: u64,
        /// The file's length.
        file_len: u64,
    },
    /// A forced boundary lies before the data section or past the end of the file.
    StartOutOfRange {
        /// The boundary.
        start: u64,
        /// The data section's start.
        data_start: u64,
        /// The file's length.
        file_len: u64,
    },
    /// A table's bytes are not a valid v1 table.
    BadTable(&'static str),
}

impl ChunkError {
    /// The `Formats` code RFC-0041 §10 gives it: 6 (`HeaderInvalid`). A file's header that puts
    /// a tensor outside its data section is invalid, and so is a table that is not a v1 table.
    #[must_use]
    pub fn code(&self) -> u16 {
        6
    }
}

impl fmt::Display for ChunkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataPastEnd {
                data_start,
                file_len,
            } => write!(
                f,
                "the data section starts at {data_start}, past the file's end at {file_len}"
            ),
            Self::StartOutOfRange {
                start,
                data_start,
                file_len,
            } => write!(
                f,
                "a tensor or expert starts at {start}, outside the data section \
                 [{data_start}, {file_len})"
            ),
            Self::BadTable(why) => write!(f, "not a v1 chunk table: {why}"),
        }
    }
}

impl std::error::Error for ChunkError {}

/// The chunks of a file of `file_len` bytes whose data section starts at `data_start`, with a
/// forced boundary at each of `starts` (the module docs' rules). `starts` may be in any order
/// and repeat; a start equal to `file_len` (an empty tensor at the end) forces nothing.
///
/// # Errors
///
/// [`ChunkError::DataPastEnd`], and [`ChunkError::StartOutOfRange`] for a start before
/// `data_start` or past `file_len`.
pub fn chunk_file(
    file_len: u64,
    data_start: u64,
    starts: impl IntoIterator<Item = u64>,
) -> Result<Vec<Chunk>, ChunkError> {
    if data_start > file_len {
        return Err(ChunkError::DataPastEnd {
            data_start,
            file_len,
        });
    }
    let mut cuts = vec![0, data_start];
    for start in starts {
        if start < data_start || start > file_len {
            return Err(ChunkError::StartOutOfRange {
                start,
                data_start,
                file_len,
            });
        }
        cuts.push(start);
    }
    cuts.push(file_len);
    cuts.sort_unstable();
    cuts.dedup();
    let mut chunks = Vec::new();
    for run in cuts.windows(2) {
        let (from, to) = (run[0], run[1]);
        let mut at = from;
        while at < to {
            let len = (to - at).min(CHUNK_BYTES);
            chunks.push(Chunk {
                offset: at,
                len: u32::try_from(len).expect("at most CHUNK_BYTES"),
                header: at < data_start,
            });
            at += len;
        }
    }
    Ok(chunks)
}

/// The forced boundaries of a GGUF file: each tensor's start, and the start of each of
/// `experts`, the slices of its stacked expert tensors.
pub fn gguf_starts<'a>(
    header: &'a GgufHeader,
    experts: &'a [ExpertSlice],
) -> impl Iterator<Item = u64> + 'a {
    header
        .tensors()
        .map(|t| t.start)
        .chain(experts.iter().map(|e| e.extent.offset))
}

/// The chunks of a GGUF file (RFC-0041 §6.1), with `experts` its stacked tensors' slices.
///
/// # Errors
///
/// As [`chunk_file`].
pub fn gguf_chunks(header: &GgufHeader, experts: &[ExpertSlice]) -> Result<Vec<Chunk>, ChunkError> {
    chunk_file(
        header.file_len,
        header.data_start,
        gguf_starts(header, experts),
    )
}

/// The forced boundaries of a safetensors file: each tensor's start, and the start of each of
/// `experts`, the slices of its stacked expert tensors.
pub fn safetensors_starts<'a>(
    header: &'a SafetensorsHeader,
    experts: &'a [ExpertSlice],
) -> impl Iterator<Item = u64> + 'a {
    header
        .tensors
        .iter()
        .map(|t| t.offset)
        .chain(experts.iter().map(|e| e.extent.offset))
}

/// The chunks of a safetensors file (RFC-0041 §6.1), with `experts` its stacked tensors' slices.
///
/// # Errors
///
/// As [`chunk_file`].
pub fn safetensors_chunks(
    header: &SafetensorsHeader,
    experts: &[ExpertSlice],
) -> Result<Vec<Chunk>, ChunkError> {
    chunk_file(
        header.file_len,
        header.data_start,
        safetensors_starts(header, experts),
    )
}

/// How a committed chunk ID was evidenced (RFC-0041 §6.3; RFC-0034 §10.1): bits 2–3 of a
/// record's flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EvidenceForm {
    /// None yet: a node may use it, and no peer packs from it.
    #[default]
    Provisional,
    /// The origin's chunk hashes.
    OriginHashes,
    /// Two witnesses that each read the bytes.
    Witnesses,
    /// A keyholder's signature.
    KeyholderVerified,
}

impl EvidenceForm {
    fn bits(self) -> u32 {
        match self {
            Self::Provisional => 0,
            Self::OriginHashes => 1,
            Self::Witnesses => 2,
            Self::KeyholderVerified => 3,
        }
    }

    fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Self::Provisional,
            1 => Self::OriginHashes,
            2 => Self::Witnesses,
            _ => Self::KeyholderVerified,
        }
    }
}

/// One 48-byte record of a chunk table (RFC-0041 §6.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkRecord {
    /// Where the chunk is.
    pub chunk: Chunk,
    /// Its `ChunkId` once committed, with the evidence it was committed with. `None` until
    /// then, written as zero.
    pub committed: Option<(Blake3, EvidenceForm)>,
}

const FLAG_COMMITTED: u32 = 1;
const FLAG_HEADER: u32 = 1 << 1;
const EVIDENCE_SHIFT: u32 = 2;
const FLAGS_KNOWN: u32 = FLAG_COMMITTED | FLAG_HEADER | (3 << EVIDENCE_SHIFT);

impl ChunkRecord {
    /// The record's 48 bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; RECORD_BYTES] {
        let mut out = [0u8; RECORD_BYTES];
        out[0..8].copy_from_slice(&self.chunk.offset.to_le_bytes());
        out[8..12].copy_from_slice(&self.chunk.len.to_le_bytes());
        let mut flags = 0;
        if self.chunk.header {
            flags |= FLAG_HEADER;
        }
        if let Some((id, evidence)) = self.committed {
            flags |= FLAG_COMMITTED | (evidence.bits() << EVIDENCE_SHIFT);
            out[16..48].copy_from_slice(&id.0);
        }
        out[12..16].copy_from_slice(&flags.to_le_bytes());
        out
    }

    /// Reads a record.
    ///
    /// # Errors
    ///
    /// [`ChunkError::BadTable`] for a flag bit RFC-0041 §6.2 does not define, a zero length, or
    /// an uncommitted record with an evidence form or a non-zero ID.
    pub fn decode(bytes: &[u8; RECORD_BYTES]) -> Result<ChunkRecord, ChunkError> {
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"));
        let offset = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes"));
        let len = word(8);
        let flags = word(12);
        if flags & !FLAGS_KNOWN != 0 {
            return Err(ChunkError::BadTable("a record sets an undefined flag bit"));
        }
        if len == 0 {
            return Err(ChunkError::BadTable("a record has length 0"));
        }
        let id: [u8; 32] = bytes[16..48].try_into().expect("32 bytes");
        let evidence = EvidenceForm::from_bits(flags >> EVIDENCE_SHIFT);
        let committed = if flags & FLAG_COMMITTED != 0 {
            Some((Blake3(id), evidence))
        } else {
            if evidence != EvidenceForm::Provisional || id != [0; 32] {
                return Err(ChunkError::BadTable(
                    "an uncommitted record has an evidence form or an ID",
                ));
            }
            None
        };
        Ok(ChunkRecord {
            chunk: Chunk {
                offset,
                len,
                header: flags & FLAG_HEADER != 0,
            },
            committed,
        })
    }
}

/// A file's chunk table (RFC-0041 §6.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkTable {
    /// The file's SHA-256.
    pub digest: FileDigest,
    /// The file's length.
    pub file_len: u64,
    /// One record per chunk, in file order.
    pub records: Vec<ChunkRecord>,
}

impl ChunkTable {
    /// A table of `chunks`, none committed yet.
    #[must_use]
    pub fn new(digest: FileDigest, file_len: u64, chunks: &[Chunk]) -> ChunkTable {
        ChunkTable {
            digest,
            file_len,
            records: chunks
                .iter()
                .map(|&chunk| ChunkRecord {
                    chunk,
                    committed: None,
                })
                .collect(),
        }
    }

    /// The table's bytes: the 64-byte header, then the records.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + RECORD_BYTES * self.records.len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&self.digest.0);
        out.extend_from_slice(&self.file_len.to_le_bytes());
        out.extend_from_slice(&(self.records.len() as u64).to_le_bytes());
        for record in &self.records {
            out.extend_from_slice(&record.encode());
        }
        out
    }

    /// Reads a table, and checks that its records tile the file it names.
    ///
    /// # Errors
    ///
    /// [`ChunkError::BadTable`] for a wrong magic or version, a non-zero reserved word, a length
    /// that disagrees with the entry count, a bad record, or records that do not tile the file
    /// from offset 0 in order.
    pub fn decode(bytes: &[u8]) -> Result<ChunkTable, ChunkError> {
        let header = bytes
            .get(..HEADER_BYTES)
            .ok_or(ChunkError::BadTable("shorter than its header"))?;
        if header[0..8] != MAGIC {
            return Err(ChunkError::BadTable("wrong magic"));
        }
        let u64_at = |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().expect("8"));
        if u32::from_le_bytes(header[8..12].try_into().expect("4")) != VERSION {
            return Err(ChunkError::BadTable("unknown version"));
        }
        if header[12..16] != [0; 4] {
            return Err(ChunkError::BadTable("the reserved word is not zero"));
        }
        let digest = FileDigest(header[16..48].try_into().expect("32"));
        let file_len = u64_at(48);
        let count = u64_at(56);
        let body = &bytes[HEADER_BYTES..];
        if u64::try_from(body.len()).ok() != count.checked_mul(RECORD_BYTES as u64) {
            return Err(ChunkError::BadTable(
                "the length disagrees with the entry count",
            ));
        }
        let mut records = Vec::with_capacity(body.len() / RECORD_BYTES);
        let mut at = 0u64;
        for raw in body.as_chunks::<RECORD_BYTES>().0 {
            let record = ChunkRecord::decode(raw)?;
            if record.chunk.offset != at {
                return Err(ChunkError::BadTable("the records do not tile the file"));
            }
            at = record.chunk.end();
            records.push(record);
        }
        if at != file_len {
            return Err(ChunkError::BadTable("the records do not cover the file"));
        }
        Ok(ChunkTable {
            digest,
            file_len,
            records,
        })
    }
}

#[cfg(test)]
mod tests;
