// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sparse source blobs, their inventories, and chunk-file mode (RFC-0041 §4.2, §4.4;
//! COMP-E11-T02).
//!
//! A source file that a node holds only in part is a **sparse blob**, `blobs/<sha256>.blob`: a
//! file of the source's full length, created sparse, into which verified chunks are written at
//! their original offsets. Holes cost no disk. The **inventory**, `blobs/<sha256>.inv`, says which
//! chunks are valid, and every read checks it first: a range that touches a chunk the inventory
//! does not hold is refused with [`BlobError::NotHeld`], so a hole is never read.
//!
//! # Start: the sparse probe
//!
//! [`Blobs::open`] runs at a writable root's start. It creates `staging/sparse-probe`, marks it
//! sparse, extends it to [`PROBE_BYTES`] (1 GiB) and reads the bytes it holds on disk. Then:
//!
//! 1. It writes [`PROBE_WRITE_BYTES`] in the middle, and checks that the write allocated about
//!    what it wrote, not the gap below it.
//! 2. It punches those bytes out again, and checks that the punch freed at least half of them.
//! 3. It writes the same bytes [`PROBE_NEAR_GAP_BYTES`] (8 MiB) into the file. A volume that
//!    zero-fills a small gap below a write allocates the gap as well. APFS does: on the hosted
//!    `macos-26` runner, a 4 KiB write 8 MiB into a file extended to 64 MiB held 8,404,992 bytes,
//!    and one 1 MiB in held 1,064,960, while one 32 MiB in held 16,384 and 1 MiB written 512 MiB
//!    into a 1 GiB file held 1,048,576 [measured, hosted macos-26, APFS, commit 318a01c]. On such
//!    a volume the probe punches the gap and checks that it frees.
//!
//! A volume that passes is [`StorageMode::Sparse`], or [`StorageMode::SparseFillsGaps`] when step
//! 3 filled the gap and the punch freed it. Then every chunk write punches the unheld runs just
//! before and after its chunk, once the chunk's bytes and bit are synced
//! (punching the 8 MiB gap of that 64 MiB file left 16,384 bytes held, same source). A volume
//! that fails a step is in **chunk-file mode** ([`StorageMode::ChunkFile`]), with the reason
//! ([`NoSparse`]): exFAT and FAT32 refuse the sparse flag on Windows and allocate the whole
//! extension on Linux and macOS, and HFS+ allocates it too. The probe file is removed before
//! `open` returns.
//!
//! What the probe found is written to `v1/storage-mode.json` (under a temporary name, synced,
//! renamed and its directory synced), so an offline `syntropy doctor` reads it with
//! [`read_storage_mode`] without opening the root for writing (STO-005, PROD-E9-T02).
//!
//! # Chunk-file mode
//!
//! On a volume without sparse files, each chunk is its own file, `chunks/<id[0..2]>/<id>`, named
//! by its `ChunkId` (the BLAKE3 of its bytes). There is no `.blob` file, only the inventory, which
//! also records each held chunk's ID. A read that spans chunks reads each chunk file in turn. Two
//! chunks with the same bytes, in one file or in two, share one chunk file, so evicting a chunk
//! deletes its file only when no inventory on the root still holds that ID. The mode is slower,
//! and `syntropy doctor` flags it (PROD-E9-T02, through [`Blobs::mode`]).
//!
//! # The inventory
//!
//! A fixed-size file, every integer little-endian, updated in place:
//!
//! ```text
//! header   0–7    magic "SYNCINV1"
//!          8–11   version (u32), 1
//!          12     mode: 0 sparse blob, 1 chunk files
//!          13     complete: 1 once the whole file's SHA-256 matched its FileDigest
//!          14–15  reserved, zero
//!          16–47  the chunk table's shape digest (ChunkTable::shape_digest)
//!          48–55  chunk count (u64)
//!          56–63  reserved, zero
//! bitmap   ⌈count / 8⌉ bytes: chunk i is held when bit (i mod 8) of byte ⌊i / 8⌋ is set
//! IDs      chunk-file mode only: count × 32 bytes, the ChunkId of chunk i's file
//! ```
//!
//! The shape digest covers the file's digest, its length and every chunk's boundaries, but not
//! the chunk IDs, so the bitmap stays valid while IDs are committed (RFC-0041 §6.3). An inventory
//! is read only against a table of the same shape.
//!
//! Every change is synced before the step it guards:
//!
//! - **Writing a chunk:** the bytes are written and synced (the blob, or the chunk file under a
//!   temporary name, renamed into place and its directory synced), in chunk-file mode the ID is
//!   written and synced, and only then is the chunk's bit set and synced. A crash before the bit
//!   leaves bytes the inventory does not claim, which a later write overwrites.
//!   Writing a chunk that is already held clears the complete flag and the chunk's bit, synced,
//!   before anything else.
//! - **Evicting a chunk** (garbage collection, §4.4): the bit is cleared and synced first, then the
//!   chunk's range is punched ([`syntropy_io::os::punch_hole`]: `fallocate(FALLOC_FL_PUNCH_HOLE)`
//!   on Linux, `F_PUNCHHOLE` on macOS, `FSCTL_SET_ZERO_DATA` on Windows), or its chunk file
//!   deleted. A crash between the two leaves blocks the inventory no longer claims, which
//!   [`Blob::reclaim`] frees.
//!
//! When every chunk is held, [`Blob::complete`] hashes the whole file and compares it with the
//! table's `FileDigest` (ADR-0016 decision 5). A match marks the inventory complete, and the blob
//! is then byte-identical to the origin's file. A mismatch is `FileDigestMismatch` (8): every
//! chunk is evicted, so nothing of the file is served, and the caller revokes its table (§6.3)
//! and refetches.
//!
//! Two leftovers of a crash in chunk-file mode are not freed here: a chunk file renamed into place
//! whose ID was never written to the inventory, and a `chunks/<xx>/<id>.part` file whose write
//! did not finish. No inventory names either, so COMP-E11-T09's garbage collection must sweep
//! `chunks/` for them, as it sweeps unnamed extents.
//!
//! Disk use is counted in allocated blocks, not file lengths ([`Blobs::disk_usage`],
//! [`Blob::allocated_bytes`]).
//!
//! Writes go through the standard library's buffered positional writes, each synced. RFC-0041
//! §4.2 names the direct-I/O write path of ADR-0013 decision 1, which `syntropy-io` does not
//! offer for writes yet.

use std::collections::BTreeSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use serde::{Deserialize, Serialize};
use syntropy_core::{Blake3, ErrorClass};
use syntropy_manifest::chunk::{ChunkTable, chunk_id};
use syntropy_manifest::{FileDigest, RangeError, RangeSource};

use crate::{Layout, WritableRoot};

/// The length the start-up probe extends its file to: 1 GiB (RFC-0041 §4.2).
pub const PROBE_BYTES: u64 = 1 << 30;

/// The bytes the probe writes and then punches out again: 1 MiB.
pub const PROBE_WRITE_BYTES: u64 = 1 << 20;

/// Where the probe's second write goes: 8 MiB into the file, past a gap small enough that APFS
/// zero-fills it (module docs).
pub const PROBE_NEAR_GAP_BYTES: u64 = 8 << 20;

/// The inventory's magic.
pub const INVENTORY_MAGIC: [u8; 8] = *b"SYNCINV1";

/// The inventory format version.
pub const INVENTORY_VERSION: u32 = 1;

/// The inventory header's length in bytes.
pub const INVENTORY_HEADER_BYTES: usize = 64;

const COMPLETE_AT: u64 = 13;

/// How a root stores source chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlobMode {
    /// One sparse file per source file, at the source's offsets.
    Sparse,
    /// One file per chunk, named by its `ChunkId`.
    ChunkFiles,
}

impl BlobMode {
    fn byte(self) -> u8 {
        match self {
            BlobMode::Sparse => 0,
            BlobMode::ChunkFiles => 1,
        }
    }

    fn from_byte(b: u8) -> Option<BlobMode> {
        match b {
            0 => Some(BlobMode::Sparse),
            1 => Some(BlobMode::ChunkFiles),
            _ => None,
        }
    }
}

/// What the start-up probe found (module docs).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "reason", rename_all = "snake_case")]
pub enum StorageMode {
    /// The volume has sparse files: new source files are sparse blobs.
    Sparse,
    /// The volume has sparse files, but a write past a small gap zero-fills the gap, as APFS
    /// does: new source files are sparse blobs, and each chunk write then punches the unheld
    /// ranges beside the chunk (module docs).
    SparseFillsGaps {
        /// The gap the probe wrote past.
        gap: u64,
        /// The bytes that write allocated beyond what it wrote.
        filled: u64,
    },
    /// It does not, and why: new source files are stored one file per chunk.
    ChunkFile(NoSparse),
}

impl StorageMode {
    /// The mode new blobs on the root are created in.
    #[must_use]
    pub fn blob_mode(&self) -> BlobMode {
        match self {
            StorageMode::Sparse | StorageMode::SparseFillsGaps { .. } => BlobMode::Sparse,
            StorageMode::ChunkFile(_) => BlobMode::ChunkFiles,
        }
    }
}

impl fmt::Display for StorageMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageMode::Sparse => f.write_str("sparse blobs"),
            StorageMode::SparseFillsGaps { gap, filled } => write!(
                f,
                "sparse blobs; a write past a {gap}-byte gap allocated {filled} bytes more than it \
                 wrote, so each chunk write punches the holes beside it"
            ),
            StorageMode::ChunkFile(why) => write!(
                f,
                "chunk-file mode, one file per chunk, which is slower: the volume has no \
                 sparse files ({why})"
            ),
        }
    }
}

/// Why the probe found no sparse-file support.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NoSparse {
    /// The volume refused to mark the probe sparse, as Windows does on FAT32 and exFAT.
    SetSparseRefused {
        /// The OS error's text.
        message: String,
    },
    /// Extending the probe failed for want of space, quota or file size, which no volume with
    /// sparse files needs.
    ExtendRefused {
        /// The OS error's text.
        message: String,
    },
    /// Extending the probe allocated its blocks.
    ExtendAllocates {
        /// The length it was extended to.
        len: u64,
        /// The bytes it then held on disk.
        allocated: u64,
    },
    /// The volume refused to punch a hole.
    PunchRefused {
        /// The OS error's text.
        message: String,
    },
    /// A write far past the end of the data allocated the gap below it too.
    WriteFillsGap {
        /// The gap.
        gap: u64,
        /// The bytes written.
        written: u64,
        /// The bytes the write allocated.
        allocated: u64,
    },
    /// A write past a small gap filled it, and punching the gap did not free it.
    GapPunchKeptBlocks {
        /// The gap.
        gap: u64,
        /// The bytes the write allocated beyond what it wrote.
        filled: u64,
        /// The bytes still held beyond the write after the punch.
        after: u64,
    },
    /// A punch succeeded but freed less than half of the bytes written.
    PunchKeptBlocks {
        /// The bytes held before the punch.
        before: u64,
        /// The bytes held after it.
        after: u64,
    },
}

impl fmt::Display for NoSparse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NoSparse::SetSparseRefused { message } => {
                write!(f, "marking a file sparse failed: {message}")
            }
            NoSparse::ExtendRefused { message } => {
                write!(f, "extending a sparse file failed: {message}")
            }
            NoSparse::ExtendAllocates { len, allocated } => write!(
                f,
                "extending a file to {len} bytes allocated {allocated} bytes on disk"
            ),
            NoSparse::PunchRefused { message } => {
                write!(f, "punching a hole failed: {message}")
            }
            NoSparse::WriteFillsGap {
                gap,
                written,
                allocated,
            } => write!(
                f,
                "writing {written} bytes past a {gap}-byte gap allocated {allocated} bytes"
            ),
            NoSparse::GapPunchKeptBlocks { gap, filled, after } => write!(
                f,
                "a write past a {gap}-byte gap filled {filled} bytes of it, and punching the gap \
                 left {after} bytes allocated"
            ),
            NoSparse::PunchKeptBlocks { before, after } => write!(
                f,
                "punching a hole left {after} of {before} bytes allocated"
            ),
        }
    }
}

/// Bytes on disk: what files hold in allocated blocks, and their lengths.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskUsage {
    /// Bytes in allocated blocks: the figure the store accounts (RFC-0041 §4.2).
    pub allocated: u64,
    /// The files' lengths, which for sparse blobs exceed what they hold.
    pub length: u64,
}

impl std::ops::AddAssign for DiskUsage {
    fn add_assign(&mut self, other: DiskUsage) {
        self.allocated = self.allocated.saturating_add(other.allocated);
        self.length = self.length.saturating_add(other.length);
    }
}

/// Why a blob operation failed.
#[non_exhaustive]
#[derive(Debug)]
pub enum BlobError {
    /// A read touches a chunk the inventory does not hold. Holes are never read.
    NotHeld {
        /// The file.
        digest: FileDigest,
        /// The first such chunk's index.
        chunk: usize,
    },
    /// A chunk index past the table's end.
    NoSuchChunk {
        /// The index.
        chunk: usize,
        /// The table's chunk count.
        count: usize,
    },
    /// A read past the file's end.
    OutOfRange {
        /// The read's offset.
        offset: u64,
        /// The read's length.
        len: u64,
        /// The file's length.
        file_len: u64,
    },
    /// Bytes given for a chunk are not the chunk's length.
    WrongLength {
        /// The chunk's index.
        chunk: usize,
        /// Its length in the table.
        expected: u32,
        /// The bytes given.
        got: usize,
    },
    /// `ChunkMismatch`, `(Formats, 7)`: bytes given for a chunk fail its committed `ChunkId`.
    ChunkMismatch {
        /// The chunk's index.
        chunk: usize,
        /// The committed ID.
        expected: Blake3,
        /// The BLAKE3 of the bytes given.
        actual: Blake3,
    },
    /// `FileDigestMismatch`, `(Formats, 8)`: a complete blob fails its `FileDigest`. Every
    /// chunk has been evicted.
    FileDigestMismatch {
        /// The table's digest.
        expected: FileDigest,
        /// The SHA-256 of the assembled bytes.
        actual: FileDigest,
    },
    /// An inventory, blob or chunk file is not what the inventory format allows, or does not
    /// match the table it is read against.
    BadInventory {
        /// The file.
        path: PathBuf,
        /// What is wrong.
        why: &'static str,
    },
    /// The file's blob is already open through another handle.
    Busy {
        /// The file.
        digest: FileDigest,
    },
    /// A file-system operation failed.
    Io {
        /// What was being done, such as `"punch"`.
        op: &'static str,
        /// The path.
        path: PathBuf,
        /// The OS error.
        source: io::Error,
    },
}

impl BlobError {
    fn io(op: &'static str, path: &Path, source: io::Error) -> BlobError {
        BlobError::Io {
            op,
            path: path.to_owned(),
            source,
        }
    }

    /// The `Formats` code RFC-0041 §10 gives it: 7 `ChunkMismatch`, 8 `FileDigestMismatch`, or
    /// `None` for an error outside that table.
    #[must_use]
    pub fn code(&self) -> Option<u16> {
        match self {
            BlobError::ChunkMismatch { .. } => Some(7),
            BlobError::FileDigestMismatch { .. } => Some(8),
            _ => None,
        }
    }

    /// Its class (RFC-0001 §10).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            // A full disk or quota is `Capacity`, as RFC-0041 §10's `NoSpace` is, not corruption.
            BlobError::Io { source, .. }
                if matches!(
                    source.kind(),
                    io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded
                ) =>
            {
                ErrorClass::Capacity
            }
            BlobError::ChunkMismatch { .. }
            | BlobError::FileDigestMismatch { .. }
            | BlobError::BadInventory { .. }
            | BlobError::Io { .. } => ErrorClass::Integrity,
            BlobError::NotHeld { .. }
            | BlobError::NoSuchChunk { .. }
            | BlobError::OutOfRange { .. }
            | BlobError::WrongLength { .. } => ErrorClass::InvalidRequest,
            BlobError::Busy { .. } => ErrorClass::Internal,
        }
    }
}

impl fmt::Display for BlobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlobError::NotHeld { digest, chunk } => write!(
                f,
                "chunk {chunk} of {} is not held, so it cannot be read",
                digest.to_hex()
            ),
            BlobError::NoSuchChunk { chunk, count } => {
                write!(f, "chunk {chunk} does not exist; the file has {count}")
            }
            BlobError::OutOfRange {
                offset,
                len,
                file_len,
            } => write!(
                f,
                "a read of {len} bytes at {offset} passes the file's end at {file_len}"
            ),
            BlobError::WrongLength {
                chunk,
                expected,
                got,
            } => write!(f, "chunk {chunk} is {expected} bytes, not {got}"),
            BlobError::ChunkMismatch {
                chunk,
                expected,
                actual,
            } => write!(
                f,
                "chunk {chunk}'s bytes hash to {}, not its committed ID {}",
                hex(&actual.0),
                hex(&expected.0)
            ),
            BlobError::FileDigestMismatch { expected, actual } => write!(
                f,
                "the assembled file hashes to {}, not its digest {}; its chunks were evicted",
                actual.to_hex(),
                expected.to_hex()
            ),
            BlobError::BadInventory { path, why } => {
                write!(f, "{} cannot be used: {why}", path.display())
            }
            BlobError::Busy { digest } => {
                write!(f, "the blob of {} is already open", digest.to_hex())
            }
            BlobError::Io { op, path, source } => {
                write!(f, "cannot {op} {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for BlobError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BlobError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The OS calls the probe and the blobs make, so a test can stand in for a volume this host does
/// not have.
pub(crate) trait SparseOps: Sync {
    fn set_sparse(&self, file: &File) -> io::Result<()>;
    fn punch_hole(&self, file: &File, offset: u64, len: u64) -> io::Result<()>;
    fn allocated_bytes(&self, file: &File) -> io::Result<u64>;
}

/// The host's own calls, through `syntropy_io::os`.
pub(crate) struct OsSparse;

impl SparseOps for OsSparse {
    fn set_sparse(&self, file: &File) -> io::Result<()> {
        syntropy_io::os::set_sparse(file)
    }

    fn punch_hole(&self, file: &File, offset: u64, len: u64) -> io::Result<()> {
        syntropy_io::os::punch_hole(file, offset, len)
    }

    fn allocated_bytes(&self, file: &File) -> io::Result<u64> {
        syntropy_io::os::allocated_bytes(file)
    }
}

/// Runs the probe at `path` (module docs) and removes the file again.
pub(crate) fn probe(path: &Path, ops: &dyn SparseOps) -> Result<StorageMode, BlobError> {
    // A probe left by a start that crashed.
    remove_if_present(path).map_err(|e| BlobError::io("remove", path, e))?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| BlobError::io("create", path, e))?;
    let found = probe_file(&file, ops);
    drop(file);
    let removed = fs::remove_file(path).map_err(|e| BlobError::io("remove", path, e));
    let mode = found.map_err(|e| BlobError::io("probe", path, e))?;
    removed?;
    Ok(mode)
}

fn probe_file(file: &File, ops: &dyn SparseOps) -> io::Result<StorageMode> {
    let refuse = |why| Ok(StorageMode::ChunkFile(why));
    if let Err(e) = ops.set_sparse(file) {
        return refuse(NoSparse::SetSparseRefused {
            message: e.to_string(),
        });
    }
    if let Err(e) = file.set_len(PROBE_BYTES) {
        return match e.kind() {
            io::ErrorKind::StorageFull
            | io::ErrorKind::QuotaExceeded
            | io::ErrorKind::FileTooLarge => refuse(NoSparse::ExtendRefused {
                message: e.to_string(),
            }),
            _ => Err(e),
        };
    }
    file.sync_all()?;
    let allocated = ops.allocated_bytes(file)?;
    if allocated >= PROBE_BYTES / 2 {
        return refuse(NoSparse::ExtendAllocates {
            len: PROBE_BYTES,
            allocated,
        });
    }
    // Bytes no file system compresses or deduplicates away, so the punch's effect shows.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let bytes: Vec<u8> = (0..PROBE_WRITE_BYTES)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[0]
        })
        .collect();
    // A write far into the extension must allocate about what it wrote, not the gap below it.
    let at = PROBE_BYTES / 2;
    write_all_at(file, at, &bytes)?;
    file.sync_all()?;
    let before = ops.allocated_bytes(file)?;
    let grew = before.saturating_sub(allocated);
    if grew > 2 * PROBE_WRITE_BYTES {
        return refuse(NoSparse::WriteFillsGap {
            gap: at,
            written: PROBE_WRITE_BYTES,
            allocated: grew,
        });
    }
    if let Err(e) = ops.punch_hole(file, at, PROBE_WRITE_BYTES) {
        return refuse(NoSparse::PunchRefused {
            message: e.to_string(),
        });
    }
    file.sync_all()?;
    let after = ops.allocated_bytes(file)?;
    if before.saturating_sub(after) < PROBE_WRITE_BYTES / 2 {
        return refuse(NoSparse::PunchKeptBlocks { before, after });
    }
    // A write a short way past the file's start: APFS zero-fills a gap this small below a write
    // (module docs). If it does, the filled gap must punch out again.
    write_all_at(file, PROBE_NEAR_GAP_BYTES, &bytes)?;
    file.sync_all()?;
    let near = ops.allocated_bytes(file)?;
    let grew = near.saturating_sub(after);
    if grew <= 2 * PROBE_WRITE_BYTES {
        return Ok(StorageMode::Sparse);
    }
    if let Err(e) = ops.punch_hole(file, 0, PROBE_NEAR_GAP_BYTES) {
        return refuse(NoSparse::PunchRefused {
            message: e.to_string(),
        });
    }
    file.sync_all()?;
    let punched = ops.allocated_bytes(file)?;
    if punched.saturating_sub(after) > 2 * PROBE_WRITE_BYTES {
        return refuse(NoSparse::GapPunchKeptBlocks {
            gap: PROBE_NEAR_GAP_BYTES,
            filled: grew.saturating_sub(PROBE_WRITE_BYTES),
            after: punched.saturating_sub(after),
        });
    }
    Ok(StorageMode::SparseFillsGaps {
        gap: PROBE_NEAR_GAP_BYTES,
        filled: grew.saturating_sub(PROBE_WRITE_BYTES),
    })
}

/// The format version of `v1/storage-mode.json`.
pub const STORAGE_MODE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct StorageModeRecord {
    version: u32,
    storage: StorageMode,
}

/// Writes what the probe found to `v1/storage-mode.json`, replacing the last start's.
fn write_storage_mode(layout: &Layout, mode: &StorageMode) -> Result<(), BlobError> {
    let record = StorageModeRecord {
        version: STORAGE_MODE_VERSION,
        storage: mode.clone(),
    };
    let mut bytes = serde_json::to_vec_pretty(&record)
        .map_err(io::Error::other)
        .map_err(|e| BlobError::io("encode", &layout.storage_mode(), e))?;
    bytes.push(b'\n');
    write_synced(&layout.storage_mode(), &bytes)
}

/// What the last start-up probe of the root at `layout` found, read without the writer lock, so
/// an offline `doctor` can show it (STO-005). `Ok(None)` when no writable start has probed the
/// root yet.
///
/// # Errors
///
/// [`BlobError::Io`] when the file cannot be read, and [`BlobError::BadInventory`] when it is not
/// a version-1 record.
pub fn read_storage_mode(layout: &Layout) -> Result<Option<StorageMode>, BlobError> {
    let path = layout.storage_mode();
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(BlobError::io("read", &path, e)),
    };
    let record: StorageModeRecord = serde_json::from_slice(&bytes).map_err(|_| {
        bad(
            &path,
            "the storage-mode record is not valid JSON of its format",
        )
    })?;
    if record.version != STORAGE_MODE_VERSION {
        return Err(bad(&path, "the storage-mode record's version is not 1"));
    }
    Ok(Some(record.storage))
}

/// The source blobs of one writable root.
pub struct Blobs<'r> {
    root: &'r WritableRoot,
    mode: StorageMode,
    ops: &'static dyn SparseOps,
    /// Held while a chunk file is created and its inventory bit set, and while one is checked
    /// for other holders and deleted, so the two never interleave.
    chunk_files: Mutex<()>,
    /// The files whose blob is open through a [`Blob`].
    open: Mutex<BTreeSet<FileDigest>>,
}

impl fmt::Debug for Blobs<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Blobs")
            .field("root", &self.root.path())
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl<'r> Blobs<'r> {
    /// Runs the sparse probe on `root`'s volume and opens its blobs (module docs).
    ///
    /// # Errors
    ///
    /// [`BlobError::Io`] when the probe file cannot be created, probed or removed.
    pub fn open(root: &'r WritableRoot) -> Result<Blobs<'r>, BlobError> {
        Blobs::open_with(root, &OsSparse)
    }

    /// A root's blobs as if its probe had found `mode`, for tests of a volume this host is not.
    #[cfg(test)]
    pub(crate) fn open_as(
        root: &'r WritableRoot,
        ops: &'static dyn SparseOps,
        mode: StorageMode,
    ) -> Blobs<'r> {
        Blobs {
            root,
            mode,
            ops,
            chunk_files: Mutex::new(()),
            open: Mutex::new(BTreeSet::new()),
        }
    }

    pub(crate) fn open_with(
        root: &'r WritableRoot,
        ops: &'static dyn SparseOps,
    ) -> Result<Blobs<'r>, BlobError> {
        let mode = probe(&root.layout().sparse_probe(), ops)?;
        write_storage_mode(root.layout(), &mode)?;
        Ok(Blobs {
            root,
            mode,
            ops,
            chunk_files: Mutex::new(()),
            open: Mutex::new(BTreeSet::new()),
        })
    }

    /// What the probe found, which `doctor` reports.
    pub fn mode(&self) -> &StorageMode {
        &self.mode
    }

    /// The root.
    pub fn root(&self) -> &WritableRoot {
        self.root
    }

    /// Opens the blob of the file `table` describes, creating it empty in the root's mode when
    /// the root has none. An existing blob keeps the mode it was created in.
    ///
    /// # Errors
    ///
    /// - [`BlobError::Busy`] when the file's blob is already open.
    /// - [`BlobError::BadInventory`] when its inventory is malformed or of another table's shape,
    ///   or its sparse blob is missing or of the wrong length. [`Blobs::remove`] clears it.
    /// - [`BlobError::Io`] when a file cannot be created, read or synced.
    pub fn blob(&self, table: &ChunkTable) -> Result<Blob<'_>, BlobError> {
        let digest = table.digest;
        if !lock(&self.open).insert(digest) {
            return Err(BlobError::Busy { digest });
        }
        let opened = self.open_blob(table);
        if opened.is_err() {
            lock(&self.open).remove(&digest);
        }
        opened
    }

    fn open_blob(&self, table: &ChunkTable) -> Result<Blob<'_>, BlobError> {
        let layout = self.root.layout();
        let inv_path = layout.inventory(&table.digest.0);
        let data_path = layout.blob(&table.digest.0);
        let bytes = match fs::read(&inv_path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(BlobError::io("read", &inv_path, e)),
        };
        let inventory = match bytes {
            Some(bytes) => {
                let inventory = Inventory::decode(&bytes).map_err(|why| bad(&inv_path, why))?;
                inventory.check(table).map_err(|why| bad(&inv_path, why))?;
                inventory
            }
            None => self.create(table, &inv_path, &data_path)?,
        };
        let data = match inventory.mode {
            BlobMode::Sparse => {
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&data_path)
                    .map_err(|e| match e.kind() {
                        io::ErrorKind::NotFound => bad(&data_path, "the sparse blob is missing"),
                        _ => BlobError::io("open", &data_path, e),
                    })?;
                let len = file
                    .metadata()
                    .map_err(|e| BlobError::io("read", &data_path, e))?
                    .len();
                if len != table.file_len {
                    return Err(bad(&data_path, "the sparse blob is not the file's length"));
                }
                Some(file)
            }
            BlobMode::ChunkFiles => None,
        };
        let inv_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&inv_path)
            .map_err(|e| BlobError::io("open", &inv_path, e))?;
        Ok(Blob {
            blobs: self,
            table: table.clone(),
            inventory,
            inv_path,
            inv_file,
            data_path,
            data,
        })
    }

    /// Creates an empty blob in the root's mode: the sparse file first, then the inventory, whose
    /// presence makes the blob exist.
    fn create(
        &self,
        table: &ChunkTable,
        inv_path: &Path,
        data_path: &Path,
    ) -> Result<Inventory, BlobError> {
        let mode = self.mode.blob_mode();
        // A blob file without an inventory is left by a create that crashed.
        remove_if_present(data_path).map_err(|e| BlobError::io("remove", data_path, e))?;
        if mode == BlobMode::Sparse {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(data_path)
                .map_err(|e| BlobError::io("create", data_path, e))?;
            self.ops
                .set_sparse(&file)
                .map_err(|e| BlobError::io("mark sparse", data_path, e))?;
            file.set_len(table.file_len)
                .map_err(|e| BlobError::io("extend", data_path, e))?;
            file.sync_all()
                .map_err(|e| BlobError::io("sync", data_path, e))?;
        }
        let inventory = Inventory::empty(mode, table);
        write_synced(inv_path, &inventory.encode())?;
        Ok(inventory)
    }

    /// Deletes the blob and inventory of `digest`, and every chunk file only it held.
    ///
    /// # Errors
    ///
    /// [`BlobError::Busy`] when the blob is open, and [`BlobError::Io`] when a file cannot be
    /// read or deleted.
    pub fn remove(&self, digest: FileDigest) -> Result<(), BlobError> {
        // Held to the end, so the blob cannot be opened while it is removed. Nothing takes this
        // lock while holding `chunk_files`, so taking both here cannot deadlock.
        let open = lock(&self.open);
        if open.contains(&digest) {
            return Err(BlobError::Busy { digest });
        }
        let layout = self.root.layout();
        let inv_path = layout.inventory(&digest.0);
        let ids = match fs::read(&inv_path) {
            Ok(bytes) => Inventory::decode(&bytes)
                .map(|inv| inv.held_ids())
                .unwrap_or_default(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => BTreeSet::new(),
            Err(e) => return Err(BlobError::io("read", &inv_path, e)),
        };
        let _guard = lock(&self.chunk_files);
        remove_if_present(&inv_path).map_err(|e| BlobError::io("remove", &inv_path, e))?;
        let data_path = layout.blob(&digest.0);
        remove_if_present(&data_path).map_err(|e| BlobError::io("remove", &data_path, e))?;
        sync_dir(&layout.blobs_dir())?;
        let held = self.held_chunk_ids(None)?;
        for id in ids.difference(&held) {
            let path = layout.chunk_file(id);
            remove_if_present(&path).map_err(|e| BlobError::io("remove", &path, e))?;
        }
        Ok(())
    }

    /// The bytes the root's blobs, inventories and chunk files hold on disk, and their lengths.
    ///
    /// # Errors
    ///
    /// [`BlobError::Io`] when a directory cannot be listed or a file's size read.
    pub fn disk_usage(&self) -> Result<DiskUsage, BlobError> {
        let layout = self.root.layout();
        let mut usage = DiskUsage::default();
        usage += self.dir_usage(&layout.blobs_dir())?;
        let chunks = layout.chunks_dir();
        for dir in list(&chunks)? {
            if dir.is_dir() {
                usage += self.dir_usage(&dir)?;
            }
        }
        Ok(usage)
    }

    fn dir_usage(&self, dir: &Path) -> Result<DiskUsage, BlobError> {
        let mut usage = DiskUsage::default();
        for path in list(dir)? {
            let file = match File::open(&path) {
                Ok(file) => file,
                // Removed since the listing.
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(BlobError::io("open", &path, e)),
            };
            let meta = file
                .metadata()
                .map_err(|e| BlobError::io("read", &path, e))?;
            if !meta.is_file() {
                continue;
            }
            usage += DiskUsage {
                allocated: self
                    .ops
                    .allocated_bytes(&file)
                    .map_err(|e| BlobError::io("read", &path, e))?,
                length: meta.len(),
            };
        }
        Ok(usage)
    }

    /// The chunk IDs every chunk-file inventory on the root holds, leaving out `except`'s.
    fn held_chunk_ids(&self, except: Option<&Path>) -> Result<BTreeSet<[u8; 32]>, BlobError> {
        let mut held = BTreeSet::new();
        for path in list(&self.root.layout().blobs_dir())? {
            if path.extension().is_none_or(|x| x != "inv") || Some(path.as_path()) == except {
                continue;
            }
            let bytes = match fs::read(&path) {
                Ok(bytes) => bytes,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(BlobError::io("read", &path, e)),
            };
            // A malformed inventory might name any chunk file, so nothing is deleted while one
            // is on the root.
            let inventory = Inventory::decode(&bytes).map_err(|why| bad(&path, why))?;
            held.extend(inventory.held_ids());
        }
        Ok(held)
    }
}

/// An open source blob: its table, its inventory and its files (module docs).
pub struct Blob<'b> {
    blobs: &'b Blobs<'b>,
    table: ChunkTable,
    inventory: Inventory,
    inv_path: PathBuf,
    inv_file: File,
    data_path: PathBuf,
    /// The sparse blob, in sparse mode.
    data: Option<File>,
}

impl fmt::Debug for Blob<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Blob")
            .field("digest", &self.table.digest)
            .field("mode", &self.inventory.mode)
            .field("held", &self.held_count())
            .field("chunks", &self.chunk_count())
            .finish_non_exhaustive()
    }
}

impl Drop for Blob<'_> {
    fn drop(&mut self) {
        lock(&self.blobs.open).remove(&self.table.digest);
    }
}

impl Blob<'_> {
    /// The file's digest.
    pub fn digest(&self) -> FileDigest {
        self.table.digest
    }

    /// The file's length.
    pub fn len(&self) -> u64 {
        self.table.file_len
    }

    /// Whether the file is empty.
    pub fn is_empty(&self) -> bool {
        self.table.file_len == 0
    }

    /// The table the blob was opened with.
    pub fn table(&self) -> &ChunkTable {
        &self.table
    }

    /// How this blob stores its chunks.
    pub fn mode(&self) -> BlobMode {
        self.inventory.mode
    }

    /// The file's chunk count.
    pub fn chunk_count(&self) -> usize {
        self.table.records.len()
    }

    /// Whether chunk `index` is held.
    pub fn is_held(&self, index: usize) -> bool {
        self.inventory.is_held(index)
    }

    /// How many chunks are held.
    pub fn held_count(&self) -> usize {
        self.inventory.held_count()
    }

    /// Whether the whole file is held and matched its `FileDigest` ([`Blob::complete`]).
    pub fn is_complete(&self) -> bool {
        self.inventory.complete
    }

    /// Writes chunk `index`'s verified `bytes` and marks it held (module docs). Returns the
    /// chunk's `ChunkId`, the BLAKE3 of the bytes, for the caller's pending batch (RFC-0041 §6.3).
    ///
    /// A chunk that is already held is evicted first: the blob's complete flag and the chunk's bit
    /// are cleared and synced, then the new bytes are written as for any chunk, so a rewrite with
    /// other bytes, such as a refetch after corruption, is never read half-written and never
    /// leaves the blob marked complete over bytes that were not hashed. In chunk-file mode the
    /// replaced chunk's file is deleted when no inventory holds it any more.
    ///
    /// # Errors
    ///
    /// - [`BlobError::NoSuchChunk`] or [`BlobError::WrongLength`] for an index or length the table
    ///   does not have.
    /// - [`BlobError::ChunkMismatch`] when the table commits an ID the bytes fail. Nothing is
    ///   written.
    /// - [`BlobError::Io`] when a write or sync fails.
    pub fn write_chunk(&mut self, index: usize, bytes: &[u8]) -> Result<Blake3, BlobError> {
        let record = *self.record(index)?;
        if bytes.len() != record.chunk.len as usize {
            return Err(BlobError::WrongLength {
                chunk: index,
                expected: record.chunk.len,
                got: bytes.len(),
            });
        }
        let id = chunk_id(bytes);
        if let Some((expected, _)) = record.committed
            && expected != id
        {
            return Err(BlobError::ChunkMismatch {
                chunk: index,
                expected,
                actual: id,
            });
        }
        // A held chunk is evicted before it is written again: its bit, and the complete flag,
        // are cleared and synced first, so a crash during the new write never leaves a bit over
        // bytes that are neither the old chunk nor the new one.
        let replaced = self.is_held(index);
        if replaced {
            self.set_complete(false)?;
            self.set_held(index, false)?;
        }
        if let Some(file) = &self.data {
            write_all_at(file, record.chunk.offset, bytes)
                .and_then(|()| file.sync_data())
                .map_err(|e| BlobError::io("write", &self.data_path, e))?;
            self.set_held(index, true)?;
            if matches!(self.blobs.mode, StorageMode::SparseFillsGaps { .. }) {
                // Only now, with the chunk's bytes and bit synced: every range punched is
                // unheld, so a crash at any point leaves at worst zeros the inventory does not
                // claim. A failed punch costs only space, which `reclaim` frees later.
                let _ = self.punch_beside(index);
            }
            return Ok(id);
        }
        let blobs = self.blobs;
        let _guard = lock(&blobs.chunk_files);
        let old = self.inventory.ids[index];
        self.write_chunk_file(&id.0, bytes)?;
        self.inventory.ids[index] = id.0;
        let at = self.inventory.id_offset(index);
        write_all_at(&self.inv_file, at, &id.0)
            .and_then(|()| self.inv_file.sync_data())
            .map_err(|e| BlobError::io("write", &self.inv_path, e))?;
        self.set_held(index, true)?;
        // The chunk file the replaced bytes were in, when nothing else holds it any more.
        if replaced && old != id.0 {
            self.delete_unheld_chunk_files(&[old])?;
        }
        Ok(id)
    }

    /// Writes `chunks/<id[0..2]>/<id>` unless it is there: under a temporary name, synced,
    /// renamed into place, and its directory synced.
    fn write_chunk_file(&self, id: &[u8; 32], bytes: &[u8]) -> Result<(), BlobError> {
        let layout = self.blobs.root.layout();
        let path = layout.chunk_file(id);
        match fs::metadata(&path) {
            // A file at the final name was renamed there whole.
            Ok(meta) if meta.len() == bytes.len() as u64 => return Ok(()),
            Ok(_) => return Err(bad(&path, "a chunk file is not its chunk's length")),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(BlobError::io("read", &path, e)),
        }
        let dir = path.parent().expect("a chunk file has a directory");
        if !dir.is_dir() {
            fs::create_dir_all(dir).map_err(|e| BlobError::io("create directory", dir, e))?;
            sync_dir(&layout.chunks_dir())?;
            sync_dir(layout.base())?;
        }
        let tmp = path.with_extension("part");
        write_synced_at(&tmp, bytes)?;
        fs::rename(&tmp, &path).map_err(|e| BlobError::io("rename", &path, e))?;
        sync_dir(dir)
    }

    /// Reads `buf.len()` bytes at `offset`. Every chunk the range touches must be held: a hole
    /// is never read.
    ///
    /// # Errors
    ///
    /// - [`BlobError::OutOfRange`] for a range past the file's end.
    /// - [`BlobError::NotHeld`] naming the first chunk in the range that is not held. Nothing is
    ///   read.
    /// - [`BlobError::Io`] or [`BlobError::BadInventory`] when a file cannot be read.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), BlobError> {
        let len = buf.len() as u64;
        let end = offset
            .checked_add(len)
            .filter(|&end| end <= self.table.file_len)
            .ok_or(BlobError::OutOfRange {
                offset,
                len,
                file_len: self.table.file_len,
            })?;
        if len == 0 {
            return Ok(());
        }
        let chunks = self.chunks_in(offset..end);
        if let Some(missing) = chunks.clone().find(|&i| !self.is_held(i)) {
            return Err(BlobError::NotHeld {
                digest: self.table.digest,
                chunk: missing,
            });
        }
        match &self.data {
            Some(file) => read_exact_at(file, offset, buf)
                .map_err(|e| BlobError::io("read", &self.data_path, e)),
            None => {
                for i in chunks {
                    let chunk = self.table.records[i].chunk;
                    let from = offset.max(chunk.offset);
                    let to = end.min(chunk.end());
                    let out = &mut buf[usize::try_from(from - offset).expect("within buf")
                        ..usize::try_from(to - offset).expect("within buf")];
                    self.read_chunk_file(i, from - chunk.offset, out)?;
                }
                Ok(())
            }
        }
    }

    fn read_chunk_file(&self, index: usize, at: u64, out: &mut [u8]) -> Result<(), BlobError> {
        let path = self
            .blobs
            .root
            .layout()
            .chunk_file(&self.inventory.ids[index]);
        let file = File::open(&path).map_err(|e| BlobError::io("open", &path, e))?;
        let len = file
            .metadata()
            .map_err(|e| BlobError::io("read", &path, e))?
            .len();
        if len != u64::from(self.table.records[index].chunk.len) {
            return Err(bad(&path, "a chunk file is not its chunk's length"));
        }
        read_exact_at(&file, at, out).map_err(|e| BlobError::io("read", &path, e))
    }

    /// Chunk `index`'s bytes.
    ///
    /// # Errors
    ///
    /// As [`Blob::read_at`], and [`BlobError::NoSuchChunk`].
    pub fn read_chunk(&self, index: usize) -> Result<Vec<u8>, BlobError> {
        let chunk = self.record(index)?.chunk;
        let mut out = vec![0; chunk.len as usize];
        self.read_at(chunk.offset, &mut out)?;
        Ok(out)
    }

    /// The parts of `range`, clipped to the file, that lie in held chunks, in order and with
    /// adjacent parts merged. A read-around reader reads these and nothing else.
    pub fn held_runs(&self, range: Range<u64>) -> Vec<Range<u64>> {
        let end = range.end.min(self.table.file_len);
        if range.start >= end {
            return Vec::new();
        }
        let mut runs: Vec<Range<u64>> = Vec::new();
        for i in self.chunks_in(range.start..end) {
            if !self.is_held(i) {
                continue;
            }
            let chunk = self.table.records[i].chunk;
            let part = range.start.max(chunk.offset)..end.min(chunk.end());
            match runs.last_mut() {
                Some(last) if last.end == part.start => last.end = part.end,
                _ => runs.push(part),
            }
        }
        runs
    }

    /// Evicts chunk `index`: clears its bit, then frees its blocks (module docs). Evicting a chunk
    /// that is not held only frees its blocks again.
    ///
    /// # Errors
    ///
    /// [`BlobError::NoSuchChunk`], and [`BlobError::Io`] when the inventory cannot be written or
    /// the punch or delete fails. The bit is cleared before either, so the chunk is never read
    /// again whatever fails after.
    pub fn evict(&mut self, index: usize) -> Result<(), BlobError> {
        self.record(index)?;
        self.set_complete(false)?;
        self.set_held(index, false)?;
        self.free(&[index])
    }

    /// Punches the runs of unheld chunks just before and just after chunk `index`, back to the
    /// nearest held chunk or the file's ends: the gap a volume that fills gaps
    /// ([`StorageMode::SparseFillsGaps`]) allocated when the chunk was written. The ranges are
    /// whole chunks that are not held, and `syntropy_io::os::punch_hole` rounds them inward to
    /// whole blocks where the OS punches only blocks, so no byte of a held chunk changes.
    fn punch_beside(&self, index: usize) -> Result<(), BlobError> {
        let Some(file) = &self.data else {
            return Ok(());
        };
        let records = &self.table.records;
        let mut first = index;
        while first > 0 && !self.is_held(first - 1) {
            first -= 1;
        }
        let mut last = index + 1;
        while last < records.len() && !self.is_held(last) {
            last += 1;
        }
        let before = records[first].chunk.offset..records[index].chunk.offset;
        let after = records[index].chunk.end()..records[last - 1].chunk.end();
        for run in [before, after] {
            if run.start < run.end {
                self.blobs
                    .ops
                    .punch_hole(file, run.start, run.end - run.start)
                    .map_err(|e| BlobError::io("punch", &self.data_path, e))?;
            }
        }
        file.sync_data()
            .map_err(|e| BlobError::io("sync", &self.data_path, e))
    }

    /// Frees the blocks of every chunk that is not held: the ranges a crash between clearing a
    /// bit and punching left allocated, or chunk files no inventory holds any more.
    ///
    /// # Errors
    ///
    /// [`BlobError::Io`] when a punch or delete fails.
    pub fn reclaim(&mut self) -> Result<(), BlobError> {
        let unheld: Vec<usize> = (0..self.chunk_count())
            .filter(|&i| !self.is_held(i))
            .collect();
        self.free(&unheld)
    }

    /// Frees the blocks of `chunks`, none of them held: one punch per run of adjacent chunks, or
    /// each chunk file no inventory on the root still holds.
    fn free(&self, chunks: &[usize]) -> Result<(), BlobError> {
        match &self.data {
            Some(file) => {
                let mut runs: Vec<Range<u64>> = Vec::new();
                for &i in chunks {
                    let chunk = self.table.records[i].chunk;
                    match runs.last_mut() {
                        Some(last) if last.end == chunk.offset => last.end = chunk.end(),
                        _ => runs.push(chunk.offset..chunk.end()),
                    }
                }
                for run in runs {
                    self.blobs
                        .ops
                        .punch_hole(file, run.start, run.end - run.start)
                        .map_err(|e| BlobError::io("punch", &self.data_path, e))?;
                }
                Ok(())
            }
            None => {
                let _guard = lock(&self.blobs.chunk_files);
                let ids: Vec<[u8; 32]> = chunks.iter().map(|&i| self.inventory.ids[i]).collect();
                self.delete_unheld_chunk_files(&ids)
            }
        }
    }

    /// Deletes each chunk file of `ids` that no inventory on the root holds, this one included.
    /// The caller holds `chunk_files`.
    fn delete_unheld_chunk_files(&self, ids: &[[u8; 32]]) -> Result<(), BlobError> {
        let mut held = self.blobs.held_chunk_ids(Some(&self.inv_path))?;
        held.extend(self.inventory.held_ids());
        let layout = self.blobs.root.layout();
        for id in ids {
            if *id == [0; 32] || held.contains(id) {
                continue;
            }
            let path = layout.chunk_file(id);
            remove_if_present(&path).map_err(|e| BlobError::io("remove", &path, e))?;
        }
        Ok(())
    }

    /// Hashes the whole file and compares it with the table's `FileDigest` (module docs). On a
    /// match the inventory is marked complete and the digest returned.
    ///
    /// # Errors
    ///
    /// - [`BlobError::NotHeld`] naming the first chunk not held.
    /// - [`BlobError::FileDigestMismatch`] when the bytes fail the digest. Every chunk has then
    ///   been evicted.
    /// - [`BlobError::Io`] when a read or write fails.
    pub fn complete(&mut self) -> Result<FileDigest, BlobError> {
        if let Some(missing) = (0..self.chunk_count()).find(|&i| !self.is_held(i)) {
            return Err(BlobError::NotHeld {
                digest: self.table.digest,
                chunk: missing,
            });
        }
        let actual = FileDigest::of_source(&HeldSource(self)).map_err(|e| BlobError::Io {
            op: "hash",
            path: self.data_path.clone(),
            source: io::Error::other(e.to_string()),
        })?;
        if actual == self.table.digest {
            self.set_complete(true)?;
            return Ok(actual);
        }
        // The flag first: a complete flag over a bitmap that is not full is a malformed
        // inventory, which could not be opened again for the refetch.
        self.set_complete(false)?;
        let all: Vec<usize> = (0..self.chunk_count()).collect();
        for &i in &all {
            self.inventory.set(i, false);
        }
        let bitmap = self.inventory.bitmap.clone();
        write_all_at(&self.inv_file, INVENTORY_HEADER_BYTES as u64, &bitmap)
            .and_then(|()| self.inv_file.sync_data())
            .map_err(|e| BlobError::io("write", &self.inv_path, e))?;
        self.free(&all)?;
        Err(BlobError::FileDigestMismatch {
            expected: self.table.digest,
            actual,
        })
    }

    /// The bytes this blob holds on disk: its sparse file and inventory, or its inventory and
    /// the distinct chunk files it holds.
    ///
    /// # Errors
    ///
    /// [`BlobError::Io`] when a file's size cannot be read.
    pub fn allocated_bytes(&self) -> Result<u64, BlobError> {
        let ops = self.blobs.ops;
        let size = |file: &File, path: &Path| {
            ops.allocated_bytes(file)
                .map_err(|e| BlobError::io("read", path, e))
        };
        let mut total = size(&self.inv_file, &self.inv_path)?;
        match &self.data {
            Some(file) => total += size(file, &self.data_path)?,
            None => {
                for id in self.inventory.held_ids() {
                    let path = self.blobs.root.layout().chunk_file(&id);
                    let file = File::open(&path).map_err(|e| BlobError::io("open", &path, e))?;
                    total += size(&file, &path)?;
                }
            }
        }
        Ok(total)
    }

    fn record(&self, index: usize) -> Result<&syntropy_manifest::chunk::ChunkRecord, BlobError> {
        self.table.records.get(index).ok_or(BlobError::NoSuchChunk {
            chunk: index,
            count: self.chunk_count(),
        })
    }

    /// The indices of the chunks that overlap `range`, which is not empty and within the file.
    fn chunks_in(&self, range: Range<u64>) -> Range<usize> {
        let records = &self.table.records;
        let first = records.partition_point(|r| r.chunk.end() <= range.start);
        let last = records.partition_point(|r| r.chunk.offset < range.end);
        first..last
    }

    fn set_held(&mut self, index: usize, held: bool) -> Result<(), BlobError> {
        self.inventory.set(index, held);
        let (at, byte) = self.inventory.bitmap_byte(index);
        write_all_at(&self.inv_file, at, &[byte])
            .and_then(|()| self.inv_file.sync_data())
            .map_err(|e| BlobError::io("write", &self.inv_path, e))
    }

    fn set_complete(&mut self, complete: bool) -> Result<(), BlobError> {
        if self.inventory.complete == complete {
            return Ok(());
        }
        self.inventory.complete = complete;
        write_all_at(&self.inv_file, COMPLETE_AT, &[u8::from(complete)])
            .and_then(|()| self.inv_file.sync_data())
            .map_err(|e| BlobError::io("write", &self.inv_path, e))
    }
}

/// A complete blob as a [`RangeSource`], for hashing.
struct HeldSource<'a, 'b>(&'a Blob<'b>);

impl RangeSource for HeldSource<'_, '_> {
    fn len(&self) -> u64 {
        self.0.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        self.0.read_at(offset, buf).map_err(|e| RangeError::Io {
            kind: io::ErrorKind::Other,
            message: e.to_string(),
        })
    }
}

/// An inventory in memory (module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Inventory {
    mode: BlobMode,
    complete: bool,
    shape: Blake3,
    count: usize,
    bitmap: Vec<u8>,
    /// Chunk-file mode only: each chunk's ID, zero until written.
    ids: Vec<[u8; 32]>,
}

impl Inventory {
    fn empty(mode: BlobMode, table: &ChunkTable) -> Inventory {
        let count = table.records.len();
        Inventory {
            mode,
            complete: false,
            shape: table.shape_digest(),
            count,
            bitmap: vec![0; count.div_ceil(8)],
            ids: match mode {
                BlobMode::Sparse => Vec::new(),
                BlobMode::ChunkFiles => vec![[0; 32]; count],
            },
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(INVENTORY_HEADER_BYTES + self.bitmap.len() + 32 * self.ids.len());
        out.extend_from_slice(&INVENTORY_MAGIC);
        out.extend_from_slice(&INVENTORY_VERSION.to_le_bytes());
        out.push(self.mode.byte());
        out.push(u8::from(self.complete));
        out.extend_from_slice(&[0; 2]);
        out.extend_from_slice(&self.shape.0);
        out.extend_from_slice(&(self.count as u64).to_le_bytes());
        out.extend_from_slice(&[0; 8]);
        out.extend_from_slice(&self.bitmap);
        for id in &self.ids {
            out.extend_from_slice(id);
        }
        out
    }

    fn decode(bytes: &[u8]) -> Result<Inventory, &'static str> {
        let header = bytes
            .get(..INVENTORY_HEADER_BYTES)
            .ok_or("the inventory is shorter than its header")?;
        if header[0..8] != INVENTORY_MAGIC {
            return Err("the inventory's magic is wrong");
        }
        if u32::from_le_bytes(header[8..12].try_into().expect("4 bytes")) != INVENTORY_VERSION {
            return Err("the inventory's version is not 1");
        }
        let mode = BlobMode::from_byte(header[12]).ok_or("the inventory's mode is undefined")?;
        let complete = match header[13] {
            0 => false,
            1 => true,
            _ => return Err("the inventory's complete flag is not 0 or 1"),
        };
        if header[14..16] != [0; 2] || header[56..64] != [0; 8] {
            return Err("the inventory's reserved bytes are not zero");
        }
        let shape = Blake3(header[16..48].try_into().expect("32 bytes"));
        let count = usize::try_from(u64::from_le_bytes(
            header[48..56].try_into().expect("8 bytes"),
        ))
        .map_err(|_| "the inventory's chunk count is too large")?;
        let bitmap_len = count.div_ceil(8);
        let ids_len = match mode {
            BlobMode::Sparse => 0,
            BlobMode::ChunkFiles => count
                .checked_mul(32)
                .ok_or("the chunk count is too large")?,
        };
        let body = &bytes[INVENTORY_HEADER_BYTES..];
        if Some(body.len()) != bitmap_len.checked_add(ids_len) {
            return Err("the inventory's length disagrees with its chunk count");
        }
        let bitmap = body[..bitmap_len].to_vec();
        if count % 8 != 0 && bitmap.last().is_some_and(|&b| b >> (count % 8) != 0) {
            return Err("the inventory sets a bit past its last chunk");
        }
        let ids: Vec<[u8; 32]> = body[bitmap_len..].as_chunks::<32>().0.to_vec();
        let inventory = Inventory {
            mode,
            complete,
            shape,
            count,
            bitmap,
            ids,
        };
        if complete && inventory.held_count() != count {
            return Err("the inventory is complete but does not hold every chunk");
        }
        if mode == BlobMode::ChunkFiles
            && (0..count).any(|i| inventory.is_held(i) && inventory.ids[i] == [0; 32])
        {
            return Err("the inventory holds a chunk file with no ID");
        }
        Ok(inventory)
    }

    /// Whether this inventory is for `table`'s shape.
    fn check(&self, table: &ChunkTable) -> Result<(), &'static str> {
        if self.count != table.records.len() {
            return Err("the inventory's chunk count is not the table's");
        }
        if self.shape != table.shape_digest() {
            return Err("the inventory is for another chunk table's shape");
        }
        Ok(())
    }

    fn is_held(&self, index: usize) -> bool {
        self.bitmap
            .get(index / 8)
            .is_some_and(|b| b & (1 << (index % 8)) != 0)
    }

    fn held_count(&self) -> usize {
        self.bitmap.iter().map(|b| b.count_ones() as usize).sum()
    }

    fn set(&mut self, index: usize, held: bool) {
        let bit = 1 << (index % 8);
        if held {
            self.bitmap[index / 8] |= bit;
        } else {
            self.bitmap[index / 8] &= !bit;
        }
    }

    /// The file offset of chunk `index`'s bitmap byte, and its value.
    fn bitmap_byte(&self, index: usize) -> (u64, u8) {
        (
            (INVENTORY_HEADER_BYTES + index / 8) as u64,
            self.bitmap[index / 8],
        )
    }

    fn id_offset(&self, index: usize) -> u64 {
        (INVENTORY_HEADER_BYTES + self.bitmap.len() + 32 * index) as u64
    }

    /// The IDs of the chunk files this inventory holds.
    fn held_ids(&self) -> BTreeSet<[u8; 32]> {
        match self.mode {
            BlobMode::Sparse => BTreeSet::new(),
            BlobMode::ChunkFiles => (0..self.count)
                .filter(|&i| self.is_held(i))
                .map(|i| self.ids[i])
                .collect(),
        }
    }
}

fn bad(path: &Path, why: &'static str) -> BlobError {
    BlobError::BadInventory {
        path: path.to_owned(),
        why,
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn remove_if_present(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

/// The entries of `dir`, or none when it does not exist.
fn list(dir: &Path) -> Result<Vec<PathBuf>, BlobError> {
    match fs::read_dir(dir) {
        Ok(entries) => entries
            .map(|e| e.map(|e| e.path()))
            .collect::<io::Result<_>>()
            .map_err(|e| BlobError::io("list", dir, e)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(BlobError::io("list", dir, e)),
    }
}

fn sync_dir(dir: &Path) -> Result<(), BlobError> {
    syntropy_io::os::sync_dir(dir).map_err(|e| BlobError::io("sync", dir, e))
}

/// Writes `bytes` to `path` under a temporary name, syncs it, renames it into place and syncs
/// the directory.
fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), BlobError> {
    let tmp = path.with_extension("tmp");
    write_synced_at(&tmp, bytes)?;
    fs::rename(&tmp, path).map_err(|e| BlobError::io("rename", path, e))?;
    sync_dir(path.parent().expect("a store file has a directory"))
}

/// Creates or truncates `path`, writes `bytes` and syncs it.
fn write_synced_at(path: &Path, bytes: &[u8]) -> Result<(), BlobError> {
    let file = File::create(path).map_err(|e| BlobError::io("create", path, e))?;
    write_all_at(&file, 0, bytes)
        .and_then(|()| file.sync_all())
        .map_err(|e| BlobError::io("write", path, e))
}

#[cfg(unix)]
fn write_all_at(file: &File, offset: u64, bytes: &[u8]) -> io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, bytes, offset)
}

#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn write_all_at(file: &File, mut offset: u64, mut bytes: &[u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt as _;
    while !bytes.is_empty() {
        match file.seek_write(bytes, offset) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => {
                bytes = &bytes[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt as _;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
