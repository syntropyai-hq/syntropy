// SPDX-License-Identifier: MIT OR Apache-2.0

//! Chunk boundaries and the chunk-table format (RFC-0041 §6.1, §6.2; ADR-0016 decision 1;
//! COMP-E12-T01).
//!
//! **Boundaries** are a pure function of a file's header, so every node computes the same chunks
//! without talking to anyone ([`chunk_file`]):
//!
//! 1. a boundary at offset 0 and at the start of the data section;
//! 2. a boundary at every tensor start, every auxiliary-extent start and every expert-slice start.
//!    The auxiliary extent of an unstacked weight (its scale, zero-point or global-scale tensor)
//!    is a tensor of the file, so its start is a tensor start. A stacked expert tensor's auxiliary
//!    tensor is stacked too (MXFP4's `W_blocks` and `W_scales`), and expert `e`'s auxiliary extent
//!    is that tensor's `e`-th slice (RFC-0041 §2.1, §2.2), which starts inside it. So the caller
//!    passes the slices of every stacked tensor, weights and auxiliary tensors alike, and
//!    [`gguf_starts`] and [`safetensors_starts`] add them to the tensor starts;
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
use crate::range::{RangeError, RangeSource};
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

/// The domain [`ChunkTable::shape_digest`] hashes first.
pub const SHAPE_DOMAIN: &[u8] = b"syntropy/chunk-table-shape/v1";

/// A chunk's `ChunkId`: the BLAKE3 of its bytes (RFC-0041 §6.2; ADR-0016 decision 1).
#[must_use]
pub fn chunk_id(bytes: &[u8]) -> Blake3 {
    Blake3(*blake3::hash(bytes).as_bytes())
}

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
    /// [`ChunkError::BadTable`] for a flag bit RFC-0041 §6.2 does not define, a length of 0 or
    /// above [`CHUNK_BYTES`], or an uncommitted record with an evidence form or a non-zero ID.
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
        // A table is read from disk or a peer: no record may claim more than one chunk's bytes.
        if u64::from(len) > CHUNK_BYTES {
            return Err(ChunkError::BadTable("a record is longer than 4 MiB"));
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
    /// from offset 0 in order. A table that decodes says only that it is well formed; a reader
    /// that uses its boundaries checks them against the chunks it computes from the file's
    /// header ([`gguf_chunks`], [`safetensors_chunks`]), which are the source of truth, with
    /// [`ChunkTable::matches`].
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
            at = at
                .checked_add(u64::from(record.chunk.len))
                .filter(|&end| end <= file_len)
                .ok_or(ChunkError::BadTable("a record runs past the file"))?;
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

    /// The digest of the table's shape: its `FileDigest`, its file length, and each record's
    /// offset, length and header flag, in order, after [`SHAPE_DOMAIN`]. IDs and evidence are
    /// not in it, so a table that commits more IDs, or stronger evidence for them, keeps its
    /// digest. A sparse blob's inventory records it, so its bitmap is read only against a table
    /// whose chunk `i` is the same range of the same file (RFC-0041 §4.1, §4.2).
    #[must_use]
    pub fn shape_digest(&self) -> Blake3 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(SHAPE_DOMAIN);
        hasher.update(&self.digest.0);
        hasher.update(&self.file_len.to_le_bytes());
        hasher.update(&(self.records.len() as u64).to_le_bytes());
        for record in &self.records {
            let c = record.chunk;
            hasher.update(&c.offset.to_le_bytes());
            hasher.update(&c.len.to_le_bytes());
            hasher.update(&[u8::from(c.header)]);
        }
        Blake3(*hasher.finalize().as_bytes())
    }

    /// Whether the table's records are exactly `chunks`: the same offsets, lengths and header
    /// flags, in order. A table read from disk or from a peer is used only if it matches the
    /// chunks computed from the file's own header.
    #[must_use]
    pub fn matches(&self, chunks: &[Chunk]) -> bool {
        self.records.len() == chunks.len()
            && self.records.iter().zip(chunks).all(|(r, c)| r.chunk == *c)
    }
}

/// The bytes one read of [`hash_chunks`] takes.
const READ_BYTES: usize = 1 << 20;

/// Read buffers in flight between the reading thread and the chunk hasher.
const BUFFERS: usize = 4;

/// Why [`hash_chunks`] could not hash a file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HashError {
    /// The chunks do not tile the source from offset 0 to its end.
    NotTiled,
    /// A read failed.
    Read(RangeError),
}

impl fmt::Display for HashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotTiled => f.write_str("the chunks do not tile the file"),
            Self::Read(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for HashError {}

/// The file's SHA-256 and each chunk's BLAKE3, its `ChunkId`, from one sequential read of
/// `source` (RFC-0041 §4.3 step 2). `chunks` must tile the source, as [`chunk_file`] gives them.
///
/// The file is read once, front to back, in 1 MiB reads. The calling thread feeds each read to
/// SHA-256, which is a sequential chain, and hands the buffer to a second thread that feeds the
/// same bytes to one BLAKE3 hasher per chunk, cutting at the chunk boundaries. So the two hashes
/// overlap: the pass takes about as long as the slower of the read, SHA-256 and BLAKE3, not their
/// sum. `on_read` is called on the calling thread with each read's length once SHA-256 has it.
///
/// # Errors
///
/// [`HashError::NotTiled`] for chunks that do not tile the source, and [`HashError::Read`] for a
/// read that fails.
pub fn hash_chunks<S: RangeSource>(
    source: &S,
    chunks: &[Chunk],
    mut on_read: impl FnMut(u64),
) -> Result<(FileDigest, Vec<Blake3>), HashError> {
    use sha2::Digest as _;
    use std::sync::mpsc::sync_channel;

    let len = source.len();
    let mut at = 0u64;
    for c in chunks {
        if c.offset != at || c.len == 0 {
            return Err(HashError::NotTiled);
        }
        at = c.end();
    }
    if at != len {
        return Err(HashError::NotTiled);
    }

    // Full buffers go to the chunk hasher; it sends each one back when it has hashed it.
    let (full_tx, full_rx) = sync_channel::<(Vec<u8>, usize)>(BUFFERS);
    let (free_tx, free_rx) = sync_channel::<Vec<u8>>(BUFFERS);
    for _ in 0..BUFFERS {
        free_tx
            .send(vec![0u8; READ_BYTES])
            .expect("the channel holds every buffer");
    }
    std::thread::scope(|s| {
        let hasher = s.spawn(move || {
            let mut ids = Vec::with_capacity(chunks.len());
            let mut chunk = 0usize;
            let mut left = chunks.first().map_or(0, |c| u64::from(c.len));
            let mut current = blake3::Hasher::new();
            while let Ok((buf, n)) = full_rx.recv() {
                let mut bytes = &buf[..n];
                while !bytes.is_empty() {
                    let take = usize::try_from(left.min(bytes.len() as u64)).unwrap_or(0);
                    current.update(&bytes[..take]);
                    bytes = &bytes[take..];
                    left -= take as u64;
                    if left == 0 {
                        ids.push(Blake3(*current.finalize().as_bytes()));
                        current = blake3::Hasher::new();
                        chunk += 1;
                        left = chunks.get(chunk).map_or(0, |c| u64::from(c.len));
                    }
                }
                // The reader may have stopped early; a closed channel is not an error here.
                let _ = free_tx.send(buf);
            }
            ids
        });
        let mut sha = sha2::Sha256::new();
        let mut at = 0u64;
        let mut result = Ok(());
        while at < len {
            let Ok(mut buf) = free_rx.recv() else { break };
            let n = usize::try_from((len - at).min(READ_BYTES as u64)).expect("at most 1 MiB");
            if let Err(e) = source.read_at(at, &mut buf[..n]) {
                result = Err(HashError::Read(e));
                break;
            }
            sha.update(&buf[..n]);
            at += n as u64;
            on_read(n as u64);
            if full_tx.send((buf, n)).is_err() {
                break;
            }
        }
        drop(full_tx);
        let ids = hasher.join().expect("the chunk hasher does not panic");
        result?;
        Ok((FileDigest(sha.finalize().into()), ids))
    })
}

#[cfg(test)]
mod tests;
