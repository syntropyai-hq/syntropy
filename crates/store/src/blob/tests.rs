// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of sparse blobs, inventories and chunk-file mode (COMP-E11-T02; RFC-0041 §4.2, §4.4).
//!
//! Sparse mode runs on the host's own volume: the system temporary directory, which is ext4 or
//! tmpfs on Linux, APFS on macOS and NTFS on Windows. A volume without sparse files is a stand-in
//! [`SparseOps`] that behaves as exFAT does on each OS, since making a real exFAT volume needs
//! root. The probe and everything after it run unchanged against the stand-in.

use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;
use syntropy_core::{Clock, MonoTime};
use syntropy_manifest::FileSource;
use syntropy_manifest::chunk::{CHUNK_BYTES, Chunk, EvidenceForm, chunk_file};
use syntropy_manifest::identity::{SourceFile, model_id};
use syntropy_manifest::safetensors::side::SourceRole;

use super::*;
use crate::local::{LocalFile, LocalIndex};

const MIB: u64 = 1 << 20;

/// A fresh directory with a writable root in it, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "syntropy-store-blob-{tag}-{}-{n}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn root(&self) -> WritableRoot {
        WritableRoot::open(self.0.join("root")).unwrap()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// exFAT or FAT32 as Linux and macOS present it: the sparse flag is not needed, an extension
/// allocates every block, and holes cannot be punched.
struct FatUnix;

impl SparseOps for FatUnix {
    fn set_sparse(&self, _file: &File) -> io::Result<()> {
        Ok(())
    }

    fn punch_hole(&self, _file: &File, _offset: u64, _len: u64) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }

    fn allocated_bytes(&self, file: &File) -> io::Result<u64> {
        // Every byte of the file, rounded up to a 32 KiB cluster.
        Ok(file.metadata()?.len().div_ceil(32 << 10) * (32 << 10))
    }
}

/// exFAT or FAT32 as Windows presents it: `FSCTL_SET_SPARSE` fails with "incorrect function".
struct FatWindows;

impl SparseOps for FatWindows {
    fn set_sparse(&self, _file: &File) -> io::Result<()> {
        Err(io::Error::other("Incorrect function. (os error 1)"))
    }

    fn punch_hole(&self, _file: &File, _offset: u64, _len: u64) -> io::Result<()> {
        Err(io::Error::other("Incorrect function. (os error 1)"))
    }

    fn allocated_bytes(&self, file: &File) -> io::Result<u64> {
        FatUnix.allocated_bytes(file)
    }
}

/// A volume with sparse extension but no hole punching.
struct NoPunch;

impl SparseOps for NoPunch {
    fn set_sparse(&self, file: &File) -> io::Result<()> {
        OsSparse.set_sparse(file)
    }

    fn punch_hole(&self, _file: &File, _offset: u64, _len: u64) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }

    fn allocated_bytes(&self, file: &File) -> io::Result<u64> {
        OsSparse.allocated_bytes(file)
    }
}

/// A volume whose punch succeeds and frees nothing.
struct PunchKeeps;

impl SparseOps for PunchKeeps {
    fn set_sparse(&self, file: &File) -> io::Result<()> {
        OsSparse.set_sparse(file)
    }

    fn punch_hole(&self, _file: &File, _offset: u64, _len: u64) -> io::Result<()> {
        Ok(())
    }

    fn allocated_bytes(&self, file: &File) -> io::Result<u64> {
        OsSparse.allocated_bytes(file)
    }
}

/// `len` bytes with every value below 0x80, from `seed`, so a canary byte of 0xC5 never occurs.
fn source_bytes(len: u64, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[3] & 0x7f
        })
        .collect()
}

const CANARY: u8 = 0xC5;

/// The table of a file of `bytes` with `chunks`.
fn table_of(bytes: &[u8], chunks: &[Chunk]) -> ChunkTable {
    ChunkTable::new(FileDigest::of_bytes(bytes), bytes.len() as u64, chunks)
}

fn chunk_bytes<'a>(bytes: &'a [u8], chunk: &Chunk) -> &'a [u8] {
    &bytes[usize::try_from(chunk.offset).unwrap()..usize::try_from(chunk.end()).unwrap()]
}

/// Writes every chunk of `bytes` into `blob`, in an order that is not the file's.
fn fill(blob: &mut Blob<'_>, bytes: &[u8]) {
    let n = blob.chunk_count();
    let records = blob.table().records.clone();
    for k in 0..n {
        // A stride coprime with any n: every index once, out of order.
        let i = (k * 7 + 3) % n;
        let i = if n.is_multiple_of(7) { k } else { i };
        blob.write_chunk(i, chunk_bytes(bytes, &records[i].chunk))
            .unwrap();
    }
}

// ---- The probe ----------------------------------------------------------------------------------

/// The host's temporary volume has sparse files on every supported OS, so a root there makes
/// sparse blobs, and the probe file is gone afterwards.
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
#[test]
fn the_probe_finds_sparse_files_on_the_hosts_volume() {
    let scratch = Scratch::new("probe");
    let root = scratch.root();
    let blobs = Blobs::open(&root).unwrap();
    // APFS fills a small gap below a write and frees it when punched; ext4, tmpfs and NTFS
    // leave the gap a hole (module docs).
    if cfg!(target_os = "macos") {
        assert!(
            matches!(blobs.mode(), StorageMode::SparseFillsGaps { gap, filled }
                if *gap == PROBE_NEAR_GAP_BYTES && *filled > 2 * PROBE_WRITE_BYTES),
            "{}",
            blobs.mode()
        );
    } else {
        assert_eq!(blobs.mode(), &StorageMode::Sparse, "{}", blobs.mode());
    }
    assert_eq!(blobs.mode().blob_mode(), BlobMode::Sparse);
    assert!(
        !root.layout().sparse_probe().exists(),
        "the probe file was left"
    );
}

/// A volume whose allocation the probe sees step by step: each `allocated_bytes` call returns
/// the next figure, so the probe's decisions can be checked for volumes this host is not.
struct Scripted {
    figures: &'static [u64],
    next: AtomicUsize,
}

impl Scripted {
    const fn new(figures: &'static [u64]) -> Scripted {
        Scripted {
            figures,
            next: AtomicUsize::new(0),
        }
    }
}

impl SparseOps for Scripted {
    fn set_sparse(&self, _file: &File) -> io::Result<()> {
        Ok(())
    }

    fn punch_hole(&self, _file: &File, _offset: u64, _len: u64) -> io::Result<()> {
        Ok(())
    }

    fn allocated_bytes(&self, _file: &File) -> io::Result<u64> {
        let i = self.next.fetch_add(1, Ordering::Relaxed);
        Ok(self.figures[i.min(self.figures.len() - 1)])
    }
}

// The figures, in the order the probe reads them: after the 64 MiB extension, after the near
// write, after punching its gap (only when it filled), after the 1 GiB extension, after the far
// write, after punching it.

/// The near write fills nothing; the far one allocates the 512 MiB below it.
static FILLS_FAR: Scripted = Scripted::new(&[0, MIB, MIB, 513 * MIB]);
/// APFS as measured: the near write holds 9 MiB and its punched gap frees; the far write holds
/// 1 MiB, and punching it frees it.
static FILLS_NEAR: Scripted = Scripted::new(&[0, 9 * MIB, MIB, MIB, 2 * MIB, MIB]);
/// The near write fills its gap, and the punch frees nothing.
static FILLS_NEAR_KEPT: Scripted = Scripted::new(&[0, 9 * MIB, 9 * MIB]);
/// Neither write fills.
static FILLS_NOTHING: Scripted = Scripted::new(&[0, MIB, MIB, 2 * MIB, MIB]);

/// The probe checks that a write allocates about what it wrote (the review of #1965, B3).
#[test]
fn the_probe_checks_that_a_write_allocates_what_it_wrote() {
    let scratch = Scratch::new("scripted");
    let root = scratch.root();
    let mode = |ops: &'static dyn SparseOps| Blobs::open_with(&root, ops).unwrap().mode().clone();
    assert_eq!(
        mode(&FILLS_FAR),
        StorageMode::ChunkFile(NoSparse::WriteFillsGap {
            gap: PROBE_BYTES / 2 - PROBE_NEAR_GAP_BYTES - PROBE_WRITE_BYTES,
            written: PROBE_WRITE_BYTES,
            allocated: 512 * MIB,
        })
    );
    assert_eq!(
        mode(&FILLS_NEAR),
        StorageMode::SparseFillsGaps {
            gap: PROBE_NEAR_GAP_BYTES,
            filled: 8 * MIB,
        }
    );
    assert_eq!(
        mode(&FILLS_NEAR_KEPT),
        StorageMode::ChunkFile(NoSparse::GapPunchKeptBlocks {
            gap: PROBE_NEAR_GAP_BYTES,
            filled: 8 * MIB,
            after: 9 * MIB,
        })
    );
    assert_eq!(mode(&FILLS_NOTHING), StorageMode::Sparse);
    assert!(!root.layout().sparse_probe().exists());
}

/// On a volume that fills gaps, a chunk write punches the unheld runs beside it, and no byte of
/// a held chunk changes, including chunks whose ends share a block with the punched range.
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
#[test]
fn a_write_on_a_gap_filling_volume_punches_beside_its_chunk_and_keeps_held_bytes() {
    let scratch = Scratch::new("straddle");
    let root = scratch.root();
    let blobs = Blobs::open_as(
        &root,
        &OsSparse,
        StorageMode::SparseFillsGaps {
            gap: PROBE_NEAR_GAP_BYTES,
            filled: 8 * MIB,
        },
    );
    // Boundaries that fall inside blocks on every file system here.
    let len = 3 * MIB + 5_000;
    let bytes = source_bytes(len, 31);
    let chunks = chunk_file(len, 0, [10_001, 700_003, 1_400_005, 2_100_007]).unwrap();
    assert_eq!(chunks.len(), 5);
    let table = table_of(&bytes, &chunks);
    let mut blob = blobs.blob(&table).unwrap();
    // Canary bytes where chunks 2 and 4 would be, then chunks 1 and 3: each write punches the
    // unheld runs beside it, which share a block with a held chunk at every boundary.
    plant_canary(&mut blob, 2);
    plant_canary(&mut blob, 4);
    blob.write_chunk(1, chunk_bytes(&bytes, &chunks[1]))
        .unwrap();
    blob.write_chunk(3, chunk_bytes(&bytes, &chunks[3]))
        .unwrap();
    for i in [1, 3] {
        assert_eq!(
            blob.read_chunk(i).unwrap(),
            chunk_bytes(&bytes, &chunks[i]),
            "chunk {i}"
        );
    }
    blob.write_chunk(0, chunk_bytes(&bytes, &chunks[0]))
        .unwrap();
    blob.write_chunk(2, chunk_bytes(&bytes, &chunks[2]))
        .unwrap();
    blob.write_chunk(4, chunk_bytes(&bytes, &chunks[4]))
        .unwrap();
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            blob.read_chunk(i).unwrap(),
            chunk_bytes(&bytes, chunk),
            "chunk {i}"
        );
    }
    assert_eq!(blob.complete().unwrap(), table.digest);
}

/// An exFAT volume switches the root to chunk-file mode, on each OS's behaviour, with a reason
/// `doctor` can show; the probe file is removed either way.
#[test]
fn an_exfat_volume_switches_the_root_to_chunk_file_mode() {
    let scratch = Scratch::new("exfat");
    let root = scratch.root();

    let blobs = Blobs::open_with(&root, &FatUnix).unwrap();
    match blobs.mode() {
        StorageMode::ChunkFile(NoSparse::ExtendAllocates { len, allocated }) => {
            assert_eq!(*len, PROBE_BYTES);
            assert_eq!(*allocated, PROBE_BYTES);
        }
        other => panic!("expected chunk-file mode, got {other:?}"),
    }
    assert_eq!(blobs.mode().blob_mode(), BlobMode::ChunkFiles);
    assert_eq!(
        blobs.mode().to_string(),
        "chunk-file mode, one file per chunk, which is slower: the volume has no sparse files \
         (extending a file to 1073741824 bytes allocated 1073741824 bytes on disk)"
    );
    assert!(!root.layout().sparse_probe().exists());
    drop(blobs);

    let blobs = Blobs::open_with(&root, &FatWindows).unwrap();
    assert!(
        matches!(
            blobs.mode(),
            StorageMode::ChunkFile(NoSparse::SetSparseRefused { message })
                if message.contains("Incorrect function")
        ),
        "{:?}",
        blobs.mode()
    );
    assert!(!root.layout().sparse_probe().exists());
}

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
#[test]
fn a_volume_that_cannot_punch_or_free_is_chunk_file_mode() {
    let scratch = Scratch::new("nopunch");
    let root = scratch.root();
    let blobs = Blobs::open_with(&root, &NoPunch).unwrap();
    assert!(
        matches!(
            blobs.mode(),
            StorageMode::ChunkFile(NoSparse::PunchRefused { .. })
        ),
        "{:?}",
        blobs.mode()
    );
    drop(blobs);
    let blobs = Blobs::open_with(&root, &PunchKeeps).unwrap();
    match blobs.mode() {
        // APFS fills the near write's gap, and this volume's punch cannot free it.
        StorageMode::ChunkFile(NoSparse::GapPunchKeptBlocks { filled, .. })
            if cfg!(target_os = "macos") =>
        {
            assert!(*filled > 2 * PROBE_WRITE_BYTES, "{filled}");
        }
        StorageMode::ChunkFile(NoSparse::PunchKeptBlocks { before, after }) => {
            assert!(before >= &PROBE_WRITE_BYTES, "{before}");
            assert_eq!(before, after);
        }
        other => panic!("expected the punch to be found wanting, got {other:?}"),
    }
}

/// A probe file left by a start that crashed does not stop the next one.
#[test]
fn a_stale_probe_file_is_replaced() {
    let scratch = Scratch::new("stale");
    let root = scratch.root();
    fs::write(root.layout().sparse_probe(), b"left over").unwrap();
    let blobs = Blobs::open_with(&root, &FatUnix).unwrap();
    assert_eq!(blobs.mode().blob_mode(), BlobMode::ChunkFiles);
    assert!(!root.layout().sparse_probe().exists());
}

// ---- Holes are never read (RFC-0041 Testing: "Holes are never read") ----------------------------

/// A file of `len` bytes with chunk boundaries forced at `starts` (as tensor starts would be),
/// and a held subset; then canary bytes planted wherever a chunk is not held.
fn partial_blob<'b>(blobs: &'b Blobs<'_>, bytes: &[u8], starts: &[u64], held: &[bool]) -> Blob<'b> {
    let len = bytes.len() as u64;
    let chunks = chunk_file(len, 0, starts.iter().copied()).unwrap();
    let table = table_of(bytes, &chunks);
    let mut blob = blobs.blob(&table).unwrap();
    for (i, chunk) in chunks.iter().enumerate() {
        if held[i % held.len()] {
            blob.write_chunk(i, chunk_bytes(bytes, chunk)).unwrap();
        } else {
            plant_canary(&mut blob, i);
        }
    }
    blob
}

/// Puts canary bytes where chunk `index` would be, as a crash or a partial punch can leave
/// them: in the sparse file at the chunk's range, or in chunk-file mode a chunk file of canary
/// bytes whose ID the inventory records but whose bit is clear (a crash after the ID's write).
fn plant_canary(blob: &mut Blob<'_>, index: usize) {
    let chunk = blob.table.records[index].chunk;
    let canary = vec![CANARY; chunk.len as usize];
    match &blob.data {
        Some(file) => write_all_at(file, chunk.offset, &canary).unwrap(),
        None => {
            let id = chunk_id(&canary);
            blob.write_chunk_file(&id.0, &canary).unwrap();
            blob.inventory.ids[index] = id.0;
            let at = blob.inventory.id_offset(index);
            write_all_at(&blob.inv_file, at, &id.0).unwrap();
        }
    }
}

fn check_reads(blob: &Blob<'_>, bytes: &[u8], reads: &[(u64, u64)]) -> Result<(), TestCaseError> {
    let len = bytes.len() as u64;
    for &(a, b) in reads {
        let (offset, n) = (a % len, b % (len / 2 + 1));
        let n = n.min(len - offset);
        let range = offset..offset + n;
        // A plain read: all held bytes, or refused.
        let mut buf = vec![0u8; usize::try_from(n).unwrap()];
        let all_held = n == 0 || blob.chunks_in(range.clone()).all(|i| blob.is_held(i));
        match blob.read_at(offset, &mut buf) {
            Ok(()) => {
                prop_assert!(all_held, "a read over a hole succeeded");
                prop_assert!(!buf.contains(&CANARY), "a canary byte was read");
                prop_assert_eq!(
                    &buf[..],
                    &bytes[usize::try_from(offset).unwrap()..][..buf.len()]
                );
            }
            Err(BlobError::NotHeld { chunk, .. }) => {
                prop_assert!(!all_held);
                prop_assert!(!blob.is_held(chunk));
                prop_assert!(buf.iter().all(|&b| b == 0), "a refused read wrote bytes");
            }
            Err(e) => return Err(TestCaseError::fail(e.to_string())),
        }
        // A read-around: the held runs inside the range, each read, nothing else.
        let mut covered = 0u64;
        for run in blob.held_runs(range.clone()) {
            prop_assert!(run.start >= range.start && run.end <= range.end);
            let mut part = vec![0u8; usize::try_from(run.end - run.start).unwrap()];
            blob.read_at(run.start, &mut part)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert!(!part.contains(&CANARY), "a read-around read a canary byte");
            prop_assert_eq!(
                &part[..],
                &bytes[usize::try_from(run.start).unwrap()..usize::try_from(run.end).unwrap()]
            );
            covered += run.end - run.start;
        }
        let held_in_range: u64 = blob
            .table
            .records
            .iter()
            .enumerate()
            .filter(|(i, _)| blob.is_held(*i))
            .map(|(_, r)| {
                let c = r.chunk;
                range
                    .end
                    .min(c.end())
                    .saturating_sub(range.start.max(c.offset))
            })
            .sum();
        prop_assert_eq!(covered, held_in_range, "the runs miss held bytes");
    }
    Ok(())
}

proptest! {
    // Each case syncs every chunk it writes, several times in chunk-file mode, so the counts
    // are kept low enough for a hosted runner's disk.
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Random partial inventories and random read and read-around requests against a sparse
    /// blob whose holes hold a canary pattern: the canary never appears in any output.
    #[test]
    fn holes_are_never_read_from_a_sparse_blob(
        len in 1u64..96 << 10,
        starts in proptest::collection::vec(any::<u64>(), 0..24),
        held in proptest::collection::vec(any::<bool>(), 1..32),
        reads in proptest::collection::vec((any::<u64>(), any::<u64>()), 1..24),
        seed in any::<u64>(),
    ) {
        holes_are_never_read(&OsSparse, len, &starts, &held, &reads, seed)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// The same in chunk-file mode, with canary chunk files a crash could leave.
    #[test]
    fn holes_are_never_read_in_chunk_file_mode(
        len in 1u64..96 << 10,
        starts in proptest::collection::vec(any::<u64>(), 0..24),
        held in proptest::collection::vec(any::<bool>(), 1..32),
        reads in proptest::collection::vec((any::<u64>(), any::<u64>()), 1..24),
        seed in any::<u64>(),
    ) {
        holes_are_never_read(&FatUnix, len, &starts, &held, &reads, seed)?;
    }
}

fn holes_are_never_read(
    ops: &'static dyn SparseOps,
    len: u64,
    starts: &[u64],
    held: &[bool],
    reads: &[(u64, u64)],
    seed: u64,
) -> Result<(), TestCaseError> {
    let scratch = Scratch::new("holes");
    let root = scratch.root();
    let blobs = Blobs::open_with(&root, ops).unwrap();
    let bytes = source_bytes(len, seed);
    let starts: Vec<u64> = starts.iter().map(|s| s % len).collect();
    let blob = partial_blob(&blobs, &bytes, &starts, held);
    check_reads(&blob, &bytes, reads)?;
    // The same after a restart, from the synced inventory alone.
    drop(blob);
    drop(blobs);
    let blobs = Blobs::open_with(&root, ops).unwrap();
    let chunks = chunk_file(len, 0, starts.iter().copied()).unwrap();
    let blob = blobs.blob(&table_of(&bytes, &chunks)).unwrap();
    for i in 0..chunks.len() {
        prop_assert_eq!(blob.is_held(i), held[i % held.len()]);
    }
    check_reads(&blob, &bytes, reads)
}

// ---- The same identity as a local path ----------------------------------------------------------

/// A clock far after any file a test writes, so no stamp is racy.
struct LaterClock(std::sync::atomic::AtomicU64);

impl Clock for LaterClock {
    fn now(&self) -> MonoTime {
        MonoTime(self.0.fetch_add(1_000_000, Ordering::Relaxed))
    }

    fn unix_ms(&self) -> u64 {
        4_102_444_800_000
    }
}

/// A file assembled in a blob gives the same `FileDigest` and `ModelId` as the same file
/// indexed through a local path: on whichever OS this runs, in sparse and chunk-file mode.
fn assembled_identity(ops: &'static dyn SparseOps, mode: BlobMode) {
    let scratch = Scratch::new("identity");
    let root = scratch.root();
    // Three 4 MiB steps, a forced boundary inside, a header region and a short tail.
    let len = 3 * CHUNK_BYTES + 777_777;
    let bytes = source_bytes(len, 42);
    let chunks = chunk_file(len, 4096, [4096, 5 * MIB + 3]).unwrap();
    assert!(chunks.len() >= 5);

    let local_path = scratch.0.join("model.gguf");
    fs::write(&local_path, &bytes).unwrap();
    let mut index = LocalIndex::open(&root).unwrap();
    let report = index
        .index(
            "llama",
            &[LocalFile {
                path: local_path.clone(),
                role: SourceRole::Weights,
                chunks: Some(chunks.clone()),
            }],
            &LaterClock(std::sync::atomic::AtomicU64::new(0)),
            &|_| {},
        )
        .unwrap();
    let local = report.model.files[0].digest;

    let blobs = Blobs::open_with(&root, ops).unwrap();
    assert_eq!(blobs.mode().blob_mode(), mode);
    let mut blob = blobs.blob(&table_of(&bytes, &chunks)).unwrap();
    assert_eq!(blob.mode(), mode);
    fill(&mut blob, &bytes);
    let assembled = blob.complete().unwrap();
    assert!(blob.is_complete());

    assert_eq!(assembled, local, "FileDigest");
    let id = model_id(
        "llama",
        &[SourceFile {
            role: SourceRole::Weights,
            digest: assembled,
            len,
        }],
    )
    .unwrap();
    assert_eq!(id, report.model.model_id, "ModelId");

    if mode == BlobMode::Sparse {
        // The blob itself, read through a local path, is the file.
        let path = root.layout().blob(&assembled.0);
        let source = FileSource::open(&path).unwrap();
        assert_eq!(source.len(), len);
        assert_eq!(FileDigest::of_source(&source).unwrap(), local);
    }
    // And a restart keeps the completion.
    drop(blob);
    let blob = blobs.blob(&table_of(&bytes, &chunks)).unwrap();
    assert!(blob.is_complete());
    assert_eq!(blob.read_chunk(2).unwrap(), chunk_bytes(&bytes, &chunks[2]));
}

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
#[test]
fn a_file_assembled_in_a_sparse_blob_has_the_local_paths_identity() {
    assembled_identity(&OsSparse, BlobMode::Sparse);
}

#[test]
fn a_file_assembled_in_chunk_files_has_the_local_paths_identity() {
    assembled_identity(&FatUnix, BlobMode::ChunkFiles);
}

// ---- Eviction punches holes; disk use is allocated blocks ---------------------------------------

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
#[test]
fn evicting_chunks_punches_their_holes_and_keeps_the_rest() {
    let scratch = Scratch::new("evict");
    let root = scratch.root();
    let blobs = Blobs::open(&root).unwrap();
    let len = 8 * CHUNK_BYTES;
    let bytes = source_bytes(len, 7);
    let chunks = chunk_file(len, 0, []).unwrap();
    assert_eq!(chunks.len(), 8);
    let table = table_of(&bytes, &chunks);
    let mut blob = blobs.blob(&table).unwrap();
    fill(&mut blob, &bytes);
    blob.complete().unwrap();
    let full = blob.allocated_bytes().unwrap();
    assert!(full >= len, "{full} bytes held for {len} written");

    for i in [1, 2, 6] {
        blob.evict(i).unwrap();
    }
    assert!(!blob.is_complete(), "an eviction keeps the blob complete");
    let after = blob.allocated_bytes().unwrap();
    assert!(
        full - after >= 3 * CHUNK_BYTES - 128 * 1024,
        "{full} bytes held before evicting 12 MiB, {after} after"
    );
    assert_eq!(
        fs::metadata(root.layout().blob(&table.digest.0))
            .unwrap()
            .len(),
        len
    );
    for (i, chunk) in chunks.iter().enumerate() {
        let held = ![1, 2, 6].contains(&i);
        assert_eq!(blob.is_held(i), held);
        match blob.read_chunk(i) {
            Ok(got) => {
                assert!(held);
                assert_eq!(got, chunk_bytes(&bytes, chunk));
            }
            Err(BlobError::NotHeld { chunk: missing, .. }) => {
                assert!(!held);
                assert_eq!(missing, i);
            }
            Err(e) => panic!("{e}"),
        }
    }
    // The cleared bits were synced: a restart sees them.
    drop(blob);
    let blob = blobs.blob(&table).unwrap();
    assert_eq!(blob.held_count(), 5);
    assert!(!blob.is_complete());
}

/// A crash between clearing a bit and punching leaves blocks allocated, which `reclaim` frees.
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
#[test]
fn reclaim_frees_what_a_crash_before_the_punch_left() {
    let scratch = Scratch::new("reclaim");
    let root = scratch.root();
    let blobs = Blobs::open(&root).unwrap();
    let len = 4 * CHUNK_BYTES;
    let bytes = source_bytes(len, 9);
    let chunks = chunk_file(len, 0, []).unwrap();
    let mut blob = blobs.blob(&table_of(&bytes, &chunks)).unwrap();
    fill(&mut blob, &bytes);
    // The first step of an eviction, then the crash.
    blob.set_held(3, false).unwrap();
    let before = blob.allocated_bytes().unwrap();
    blob.reclaim().unwrap();
    let after = blob.allocated_bytes().unwrap();
    assert!(
        before - after >= CHUNK_BYTES - 128 * 1024,
        "{before} → {after}"
    );
    assert_eq!(blob.read_chunk(0).unwrap(), chunk_bytes(&bytes, &chunks[0]));
}

/// A 64 MiB sparse blob holding one 4 MiB chunk is accounted at what it holds, not its length.
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
#[test]
fn disk_use_is_counted_in_allocated_blocks() {
    let scratch = Scratch::new("usage");
    let root = scratch.root();
    let blobs = Blobs::open(&root).unwrap();
    let len = 16 * CHUNK_BYTES;
    let bytes = source_bytes(len, 11);
    let chunks = chunk_file(len, 0, []).unwrap();
    let mut blob = blobs.blob(&table_of(&bytes, &chunks)).unwrap();
    blob.write_chunk(5, chunk_bytes(&bytes, &chunks[5]))
        .unwrap();
    let usage = blobs.disk_usage().unwrap();
    assert!(usage.length >= len, "{usage:?}");
    assert!(usage.allocated >= CHUNK_BYTES, "{usage:?}");
    assert!(usage.allocated < 2 * CHUNK_BYTES, "{usage:?}");
    assert_eq!(blob.allocated_bytes().unwrap(), usage.allocated);
}

#[test]
fn chunk_file_mode_accounts_its_chunk_files() {
    let scratch = Scratch::new("usage-cf");
    let root = scratch.root();
    let blobs = Blobs::open_with(&root, &OsSparseCounting).unwrap();
    let len = 3 * MIB;
    let bytes = source_bytes(len, 12);
    let chunks = chunk_file(len, 0, [MIB, 2 * MIB]).unwrap();
    let mut blob = blobs.blob(&table_of(&bytes, &chunks)).unwrap();
    blob.write_chunk(1, chunk_bytes(&bytes, &chunks[1]))
        .unwrap();
    let usage = blobs.disk_usage().unwrap();
    assert!(
        !root.layout().blob(&blob.digest().0).exists(),
        "chunk-file mode made a blob"
    );
    // The inventory and one chunk file.
    assert_eq!(
        usage.length,
        MIB + fs::metadata(&blob.inv_path).unwrap().len()
    );
    assert_eq!(usage.allocated, blob.allocated_bytes().unwrap());
}

/// Chunk-file mode on the host's real allocation counts: the stand-in refuses the punch, so the
/// probe picks chunk files, and sizes are the volume's own.
struct OsSparseCounting;

impl SparseOps for OsSparseCounting {
    fn set_sparse(&self, file: &File) -> io::Result<()> {
        OsSparse.set_sparse(file)
    }

    fn punch_hole(&self, _file: &File, _offset: u64, _len: u64) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }

    fn allocated_bytes(&self, file: &File) -> io::Result<u64> {
        // On a host without the call, the length stands in.
        OsSparse
            .allocated_bytes(file)
            .or_else(|_| Ok(file.metadata()?.len()))
    }
}

// ---- Chunk files are shared by ID ---------------------------------------------------------------

#[test]
fn a_chunk_file_shared_by_two_files_is_deleted_only_when_neither_holds_it() {
    let scratch = Scratch::new("shared");
    let root = scratch.root();
    let blobs = Blobs::open_with(&root, &FatUnix).unwrap();
    // Two files whose second chunk has the same bytes, and a third file whose two chunks are
    // the same bytes as each other.
    let common = source_bytes(1000, 5);
    let a = [source_bytes(500, 1), common.clone()].concat();
    let b = [source_bytes(700, 2), common.clone()].concat();
    let c = [common.clone(), common.clone()].concat();
    let table = |bytes: &[u8], first: u64| {
        table_of(bytes, &chunk_file(bytes.len() as u64, 0, [first]).unwrap())
    };
    let (ta, tb, tc) = (table(&a, 500), table(&b, 700), table(&c, 1000));
    let shared = root.layout().chunk_file(&chunk_id(&common).0);

    let mut blob_a = blobs.blob(&ta).unwrap();
    fill(&mut blob_a, &a);
    let mut blob_b = blobs.blob(&tb).unwrap();
    fill(&mut blob_b, &b);
    let mut blob_c = blobs.blob(&tc).unwrap();
    fill(&mut blob_c, &c);
    assert!(shared.is_file());

    blob_a.evict(1).unwrap();
    assert!(shared.is_file(), "b and c still hold it");
    drop(blob_b);
    blobs.remove(tb.digest).unwrap();
    assert!(shared.is_file(), "c still holds it");
    blob_c.evict(0).unwrap();
    assert!(shared.is_file(), "c's chunk 1 still holds it");
    assert_eq!(blob_c.read_chunk(1).unwrap(), common);
    blob_c.evict(1).unwrap();
    assert!(!shared.exists(), "nothing holds it");
    // a's own chunk stays.
    assert_eq!(blob_a.read_chunk(0).unwrap(), &a[..500]);
}

// ---- Integrity -----------------------------------------------------------------------------------

#[test]
fn bytes_that_fail_a_committed_id_are_refused_and_not_written() {
    for ops in [&OsSparse as &'static dyn SparseOps, &FatUnix] {
        let scratch = Scratch::new("mismatch");
        let root = scratch.root();
        let blobs = Blobs::open_with(&root, ops).unwrap();
        let bytes = source_bytes(3000, 3);
        let chunks = chunk_file(3000, 0, [1000, 2000]).unwrap();
        let mut table = table_of(&bytes, &chunks);
        table.records[1].committed = Some((chunk_id(&bytes[1000..2000]), EvidenceForm::Witnesses));
        let mut blob = blobs.blob(&table).unwrap();
        let mut wrong = bytes[1000..2000].to_vec();
        wrong[17] ^= 1;
        let error = blob.write_chunk(1, &wrong).unwrap_err();
        assert_eq!(error.code(), Some(7), "{error}");
        assert_eq!(error.class(), ErrorClass::Integrity);
        assert!(!blob.is_held(1));
        assert!(matches!(
            blob.write_chunk(1, &bytes[1000..1999]),
            Err(BlobError::WrongLength {
                chunk: 1,
                expected: 1000,
                got: 999
            })
        ));
        assert!(matches!(
            blob.write_chunk(3, &bytes[..10]),
            Err(BlobError::NoSuchChunk { chunk: 3, count: 3 })
        ));
        assert_eq!(
            blob.write_chunk(1, &bytes[1000..2000]).unwrap(),
            chunk_id(&bytes[1000..2000])
        );
    }
}

/// A blob whose assembled bytes fail the `FileDigest` serves nothing afterwards.
#[test]
fn a_complete_blob_that_fails_its_digest_is_cleared() {
    for ops in [&OsSparse as &'static dyn SparseOps, &FatUnix] {
        let scratch = Scratch::new("digest");
        let root = scratch.root();
        let blobs = Blobs::open_with(&root, ops).unwrap();
        let bytes = source_bytes(3000, 4);
        let chunks = chunk_file(3000, 0, [1000, 2000]).unwrap();
        let mut table = table_of(&bytes, &chunks);
        table.digest = FileDigest([0xAB; 32]);
        let mut blob = blobs.blob(&table).unwrap();
        fill(&mut blob, &bytes);
        let error = blob.complete().unwrap_err();
        assert_eq!(error.code(), Some(8), "{error}");
        assert!(matches!(
            error,
            BlobError::FileDigestMismatch { actual, .. } if actual == FileDigest::of_bytes(&bytes)
        ));
        assert_eq!(blob.held_count(), 0);
        assert!(!blob.is_complete());
        assert!(matches!(
            blob.read_at(0, &mut [0; 10]),
            Err(BlobError::NotHeld { chunk: 0, .. })
        ));
        drop(blob);
        assert_eq!(blobs.blob(&table).unwrap().held_count(), 0);
    }
}

#[test]
fn completing_needs_every_chunk() {
    let scratch = Scratch::new("incomplete");
    let root = scratch.root();
    let blobs = Blobs::open_with(&root, &OsSparse).unwrap();
    let bytes = source_bytes(3000, 6);
    let chunks = chunk_file(3000, 0, [1000, 2000]).unwrap();
    let mut blob = blobs.blob(&table_of(&bytes, &chunks)).unwrap();
    blob.write_chunk(0, &bytes[..1000]).unwrap();
    blob.write_chunk(2, &bytes[2000..]).unwrap();
    assert!(matches!(
        blob.complete(),
        Err(BlobError::NotHeld { chunk: 1, .. })
    ));
    assert!(matches!(
        blob.read_at(2990, &mut [0; 20]),
        Err(BlobError::OutOfRange {
            offset: 2990,
            len: 20,
            file_len: 3000
        })
    ));
}

// ---- The inventory -------------------------------------------------------------------------------

#[test]
fn an_inventory_round_trips_and_its_bytes_are_pinned() {
    let chunks = chunk_file(10 << 20, 300, [300, 5 << 20]).unwrap();
    let table = ChunkTable::new(FileDigest([4; 32]), 10 << 20, &chunks);
    let mut inv = Inventory::empty(BlobMode::ChunkFiles, &table);
    inv.set(1, true);
    inv.ids[1] = [7; 32];
    let bytes = inv.encode();
    assert_eq!(bytes.len(), INVENTORY_HEADER_BYTES + 1 + 32 * chunks.len());
    assert_eq!(Inventory::decode(&bytes).unwrap(), inv);
    assert_eq!(&bytes[..8], b"SYNCINV1");
    assert_eq!(bytes[8..12], 1u32.to_le_bytes());
    assert_eq!(bytes[12], 1, "mode");
    assert_eq!(bytes[13], 0, "complete");
    assert_eq!(bytes[16..48], table.shape_digest().0);
    assert_eq!(bytes[48..56], (chunks.len() as u64).to_le_bytes());
    assert_eq!(bytes[64], 0b10, "bit 1 of byte 0");
    assert_eq!(bytes[65 + 32..65 + 64], [7; 32]);

    let sparse = Inventory::empty(BlobMode::Sparse, &table);
    assert_eq!(
        sparse.encode().len(),
        INVENTORY_HEADER_BYTES + chunks.len().div_ceil(8)
    );
}

/// A change that makes an inventory's bytes invalid.
type Corrupt = fn(&mut Vec<u8>);

#[test]
fn a_malformed_inventory_is_refused() {
    let chunks = chunk_file(10 << 20, 300, [300, 5 << 20]).unwrap();
    let table = ChunkTable::new(FileDigest([4; 32]), 10 << 20, &chunks);
    let good = Inventory::empty(BlobMode::ChunkFiles, &table).encode();
    let cases: [(&str, Corrupt); 9] = [
        ("magic", |b| b[0] = b'X'),
        ("version", |b| b[8] = 2),
        ("mode", |b| b[12] = 2),
        ("complete flag", |b| b[13] = 2),
        ("reserved", |b| b[15] = 1),
        ("length", |b| {
            b.pop();
        }),
        ("bit past the end", |b| b[64] = 0x80),
        ("complete but partial", |b| b[13] = 1),
        ("held without an ID", |b| b[64] = 1),
    ];
    for (what, corrupt) in cases {
        let mut bytes = good.clone();
        corrupt(&mut bytes);
        assert!(Inventory::decode(&bytes).is_err(), "{what} was accepted");
    }
    assert!(Inventory::decode(&good[..10]).is_err());
}

/// An inventory written for another shape of table is not read against this one.
#[test]
fn an_inventory_of_another_shape_is_not_used() {
    let scratch = Scratch::new("shape");
    let root = scratch.root();
    let blobs = Blobs::open_with(&root, &OsSparse).unwrap();
    let bytes = source_bytes(3000, 8);
    let table = table_of(&bytes, &chunk_file(3000, 0, [1000, 2000]).unwrap());
    drop(blobs.blob(&table).unwrap());
    let moved = table_of(&bytes, &chunk_file(3000, 0, [1000, 2001]).unwrap());
    assert!(matches!(
        blobs.blob(&moved),
        Err(BlobError::BadInventory { why, .. }) if why.contains("shape")
    ));
    // `remove` clears it, and the new shape gets a fresh blob.
    blobs.remove(table.digest).unwrap();
    assert_eq!(blobs.blob(&moved).unwrap().held_count(), 0);
}

#[test]
fn a_blob_opens_through_one_handle_at_a_time() {
    let scratch = Scratch::new("busy");
    let root = scratch.root();
    let blobs = Blobs::open_with(&root, &OsSparse).unwrap();
    let bytes = source_bytes(100, 10);
    let table = table_of(&bytes, &chunk_file(100, 0, []).unwrap());
    let blob = blobs.blob(&table).unwrap();
    assert!(matches!(blobs.blob(&table), Err(BlobError::Busy { .. })));
    assert!(matches!(
        blobs.remove(table.digest),
        Err(BlobError::Busy { .. })
    ));
    drop(blob);
    drop(blobs.blob(&table).unwrap());
}

// ---- Review of #1965: re-verification, rewrites, the recorded mode, the error class ------------

/// Corrupts one byte of chunk `index` where it is stored: in the sparse file, or in its chunk
/// file, keeping the length.
fn corrupt(blob: &Blob<'_>, index: usize) {
    let chunk = blob.table.records[index].chunk;
    match &blob.data {
        Some(file) => {
            let mut byte = [0u8];
            read_exact_at(file, chunk.offset, &mut byte).unwrap();
            write_all_at(file, chunk.offset, &[byte[0] ^ 0xff]).unwrap();
        }
        None => {
            let path = blob
                .blobs
                .root
                .layout()
                .chunk_file(&blob.inventory.ids[index]);
            let mut bytes = fs::read(&path).unwrap();
            bytes[0] ^= 0xff;
            fs::write(&path, bytes).unwrap();
        }
    }
}

/// B1: a blob that completed once and then fails a re-verification is cleared, flag first, so
/// it still opens and can be refetched.
#[test]
fn a_failed_reverification_clears_the_complete_flag_and_the_blob_reopens() {
    for ops in [&OsSparse as &'static dyn SparseOps, &FatUnix] {
        let scratch = Scratch::new("reverify");
        let root = scratch.root();
        let blobs = Blobs::open_with(&root, ops).unwrap();
        let bytes = source_bytes(3000, 21);
        let table = table_of(&bytes, &chunk_file(3000, 0, [1000, 2000]).unwrap());
        let mut blob = blobs.blob(&table).unwrap();
        fill(&mut blob, &bytes);
        blob.complete().unwrap();
        assert!(blob.is_complete());

        corrupt(&blob, 1);
        let error = blob.complete().unwrap_err();
        assert_eq!(error.code(), Some(8), "{error}");
        assert!(!blob.is_complete());
        assert_eq!(blob.held_count(), 0);
        drop(blob);

        // The inventory on disk is valid, and the refetch can go ahead.
        let mut blob = blobs.blob(&table).unwrap();
        assert!(!blob.is_complete());
        assert_eq!(blob.held_count(), 0);
        fill(&mut blob, &bytes);
        assert_eq!(blob.complete().unwrap(), table.digest);
    }
}

/// B2: rewriting a held chunk evicts it first. The complete flag goes, the new bytes are what
/// is read, a restart sees the same, and in chunk-file mode the replaced bytes' file is freed.
///
/// The order is what makes a crash safe. In both modes the chunk's bit and the complete flag
/// are cleared and synced before any new byte is written. A crash during the rewrite therefore
/// leaves the chunk unheld, so the torn range is never read, and leaves the blob not complete, so
/// no digest is claimed over bytes that were not hashed. A crash after the new bytes and their
/// ID are synced but before the bit is set leaves the same state. In chunk-file mode the old file
/// is deleted only after the new bit is set; a crash before that leaves an orphaned chunk file,
/// which COMP-E11-T09's sweep frees (module docs).
#[test]
fn rewriting_a_held_chunk_evicts_it_first() {
    for ops in [&OsSparse as &'static dyn SparseOps, &FatUnix] {
        let scratch = Scratch::new("rewrite");
        let root = scratch.root();
        let blobs = Blobs::open_with(&root, ops).unwrap();
        let bytes = source_bytes(3000, 22);
        let table = table_of(&bytes, &chunk_file(3000, 0, [1000, 2000]).unwrap());
        let mut blob = blobs.blob(&table).unwrap();
        fill(&mut blob, &bytes);
        blob.complete().unwrap();
        let old_file = root.layout().chunk_file(&chunk_id(&bytes[1000..2000]).0);

        // The state a crash right after the eviction step leaves: unheld, not complete.
        blob.set_complete(false).unwrap();
        blob.set_held(1, false).unwrap();
        drop(blob);
        let mut blob = blobs.blob(&table).unwrap();
        assert!(!blob.is_held(1) && !blob.is_complete());
        blob.write_chunk(1, &bytes[1000..2000]).unwrap();
        blob.complete().unwrap();

        // A rewrite with other bytes.
        let other = vec![0x11u8; 1000];
        blob.write_chunk(1, &other).unwrap();
        assert!(!blob.is_complete(), "the complete flag survived a rewrite");
        assert!(blob.is_held(1));
        assert_eq!(blob.read_chunk(1).unwrap(), other);
        if blob.mode() == BlobMode::ChunkFiles {
            assert!(!old_file.exists(), "the replaced chunk file leaked");
            assert!(root.layout().chunk_file(&chunk_id(&other).0).is_file());
        }
        drop(blob);
        let mut blob = blobs.blob(&table).unwrap();
        assert!(!blob.is_complete());
        assert_eq!(blob.held_count(), 3);
        assert_eq!(blob.read_chunk(1).unwrap(), other);
        // Its digest now fails, and the blob is cleared, flag first.
        assert_eq!(blob.complete().unwrap_err().code(), Some(8));
    }
}

/// N1: the probe's outcome is recorded under `v1/` and reads back without the writer lock.
#[test]
fn the_probe_outcome_is_recorded_for_an_offline_doctor() {
    let scratch = Scratch::new("mode-record");
    let root = scratch.root();
    assert_eq!(read_storage_mode(root.layout()).unwrap(), None);
    drop(Blobs::open_with(&root, &FatUnix).unwrap());
    let found = read_storage_mode(root.layout()).unwrap().unwrap();
    assert!(matches!(
        found,
        StorageMode::ChunkFile(NoSparse::ExtendAllocates { .. })
    ));
    let text = fs::read_to_string(root.layout().storage_mode()).unwrap();
    assert!(text.contains("\"version\": 1"), "{text}");
    assert!(text.contains("\"mode\": \"chunk_file\""), "{text}");
    assert!(text.contains("\"kind\": \"extend_allocates\""), "{text}");
    // A later start replaces it.
    drop(Blobs::open_with(&root, &FatWindows).unwrap());
    assert!(matches!(
        read_storage_mode(root.layout()).unwrap(),
        Some(StorageMode::ChunkFile(NoSparse::SetSparseRefused { .. }))
    ));
    // Read from a read-only open of the same root, as an offline doctor would.
    let reader = crate::ReadOnlyRoot::open(root.path()).unwrap();
    assert!(read_storage_mode(reader.layout()).unwrap().is_some());
    fs::write(root.layout().storage_mode(), b"{\"version\": 2}").unwrap();
    assert!(matches!(
        read_storage_mode(root.layout()),
        Err(BlobError::BadInventory { .. })
    ));
}

/// N3: a full disk is `Capacity`, other OS errors `Integrity`.
#[test]
fn a_full_disk_is_a_capacity_error() {
    let io = |kind: io::ErrorKind| BlobError::Io {
        op: "write",
        path: PathBuf::from("/x"),
        source: kind.into(),
    };
    assert_eq!(io(io::ErrorKind::StorageFull).class(), ErrorClass::Capacity);
    assert_eq!(
        io(io::ErrorKind::QuotaExceeded).class(),
        ErrorClass::Capacity
    );
    assert_eq!(
        io(io::ErrorKind::PermissionDenied).class(),
        ErrorClass::Integrity
    );
}
