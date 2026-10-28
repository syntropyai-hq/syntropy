// SPDX-License-Identifier: MIT OR Apache-2.0

//! Local models, indexed in place (RFC-0041 §4.3; COMP-E11-T03, COMP-E11-T14).
//!
//! A local path is indexed, never copied or modified. [`LocalIndex::index`] takes the files the
//! census read, each with its role and its chunks (RFC-0041 §6.1), and:
//!
//! 1. stamps each file with its (device, inode, size, mtime), and closes it again;
//! 2. reuses the `FileDigest` of a file whose stamp matches the record of an earlier index and
//!    was not racy (below), and whose chunk table is on disk, matches its chunks and holds every
//!    chunk's ID, so a later start reads nothing more;
//! 3. hashes every other file in one multi-threaded pass. SHA-256 is a sequential chain, so each
//!    file is hashed whole on one thread, and up to `min(cores, files)` files are hashed side by
//!    side, as the `t_index` bound assumes. A thread opens its file when it starts on it, checks
//!    the stamp through that handle, hashes, and stamps it again, so at most one file per thread
//!    is open. The same read gives each chunk's BLAKE3, its `ChunkId`
//!    ([`hash_chunks`](syntropy_manifest::chunk::hash_chunks)), and the file's chunk table is
//!    written to `tables/<sha256>.chunks` with every ID set, so a local source's table is complete
//!    when the model is registered (RFC-0041 §4.3 step 2). Its IDs are this node's own, evidence
//!    form provisional: in a cluster they become a witness only as COMP-E11-T17 says;
//! 4. computes the v1 `ModelId` (RFC-0041 §1) and writes `local/<model-id>.sources`, which
//!    survives a restart.
//!
//! [`LocalIndex::check`] compares a model's files with their record. A change to the device,
//! inode, size or mtime of any of them is [`LocalError::SourceChanged`], RFC-0041 §10's
//! `LocalSourceChanged` (14, `Integrity`). Its action is to re-index, which yields a new
//! `ModelId` for the changed bytes.
//!
//! # Racy stamps
//!
//! An mtime has the file system's granularity: a coarse-clock tick on Linux, 1 s on HFS+, 2 s on
//! FAT32. A write in the same tick as the file's previous write leaves the stamp unchanged. If it
//! lands while the file is hashed, after the hasher has read the range it changes, the check
//! after hashing passes and the stamp is paired with a digest of bytes the file no longer holds.
//! So, as version control systems do with their "racily clean" entries, a file whose mtime is
//! less than [`RACY_MARGIN_MS`] (2 s, the coarsest supported granularity) before the clock's
//! wall time at the start of the pass is recorded as **racy**. A racy stamp is never reused: the
//! next pass hashes that file again, and by then a later write would have moved its mtime. The
//! wall time comes from the [`Clock`] the pass is given (rule T1).
//!
//! Past that, the stamp is a check against accidental change, not against an adversary: a file
//! rewritten in place with its size and mtime set back keeps its record, and its old digest is
//! reused. RFC-0041 §4.3 accepts this, since the file is the user's own and bytes are copied out
//! of it at `Loading`. [`LocalIndex::check`] compares stamps only, so it cannot see a same-tick
//! write to a racy file either; the next `index` does.
//!
//! A racy stamp also keeps the file's chunk table from being reused: the file is read again, and
//! its table rewritten from the bytes it holds then. A local file's evidence in a cluster is
//! COMP-E11-T17.
//!
//! # The record
//!
//! `local/<model-id>.sources` is one JSON object, written to a temporary name, synced and renamed
//! into place, and the directory is synced:
//!
//! ```text
//! { "version": 1, "model_id": "<64 hex>", "family": "<frozen tag>",
//!   "files": [ { "path": "<absolute path>", "role": "weights", "sha256": "<64 hex>",
//!                "size": 123, "device": 1, "inode": 2, "mtime_secs": 1, "mtime_nanos": 0,
//!                "racy": false } ] }
//! ```
//!
//! Paths are absolute and canonical. A path that is not valid Unicode cannot be written in this
//! form, so a model with one is indexed but not recorded, and is hashed again at the next start.
//! A record that cannot be read is ignored and named in [`LocalIndex::unreadable`]; it is replaced
//! when that model is next indexed.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use syntropy_core::{BytesPerSec, Clock, ErrorClass, ModelId, Nanos};
use syntropy_manifest::chunk::{
    Chunk, ChunkTable, EvidenceForm, HashError, chunk_file, hash_chunks,
};
use syntropy_manifest::identity::{IdentityError, SourceFile, model_id};
use syntropy_manifest::safetensors::side::SourceRole;
use syntropy_manifest::{FileDigest, FileSource, RangeError, RangeSource as _};

use crate::{Layout, WritableRoot};

/// One file of a local model, as the census found it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalFile {
    /// Where it is. A relative path is resolved against the current directory.
    pub path: PathBuf,
    /// What it is to the census.
    pub role: SourceRole,
    /// Its chunks (RFC-0041 §6.1), as the census computes them from its header: `gguf_chunks` or
    /// `safetensors_chunks` with the slices of its stacked tensors. `None` for a side file with
    /// no tensors, such as `config.json`, which is all header region: `chunk_file(len, len, [])`.
    pub chunks: Option<Vec<Chunk>>,
}

impl LocalFile {
    /// A file with no chunks of its own: a side file, all header region.
    #[must_use]
    pub fn side(path: impl Into<PathBuf>, role: SourceRole) -> LocalFile {
        LocalFile {
            path: path.into(),
            role,
            chunks: None,
        }
    }
}

/// A file's modification time, as seconds and nanoseconds from the Unix epoch. Before the
/// epoch, `secs` is negative and `nanos` counts forward from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mtime {
    /// Whole seconds.
    pub secs: i64,
    /// Nanoseconds past `secs`, below 10⁹.
    pub nanos: u32,
}

/// How close to the pass's start, in milliseconds, a file's mtime may be before its stamp is
/// racy: 2 s, FAT32's granularity and the coarsest of the supported file systems.
pub const RACY_MARGIN_MS: i64 = 2_000;

impl Mtime {
    /// Milliseconds from the Unix epoch, rounded down.
    #[must_use]
    pub fn unix_ms(self) -> i128 {
        i128::from(self.secs) * 1_000 + i128::from(self.nanos / 1_000_000)
    }

    fn of(time: SystemTime) -> Mtime {
        match time.duration_since(UNIX_EPOCH) {
            Ok(d) => Mtime {
                secs: i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
                nanos: d.subsec_nanos(),
            },
            Err(before) => {
                let d = before.duration();
                let secs = i64::try_from(d.as_secs()).unwrap_or(i64::MAX);
                if d.subsec_nanos() == 0 {
                    Mtime {
                        secs: -secs,
                        nanos: 0,
                    }
                } else {
                    Mtime {
                        secs: -secs - 1,
                        nanos: 1_000_000_000 - d.subsec_nanos(),
                    }
                }
            }
        }
    }
}

/// What the index records of a file to tell whether it changed (RFC-0041 §4.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileStamp {
    /// The device, or the volume serial number on Windows.
    pub device: u64,
    /// The inode, or the file index on Windows.
    pub inode: u64,
    /// The length in bytes.
    pub size: u64,
    /// The modification time.
    pub mtime: Mtime,
}

impl FileStamp {
    /// The stamp of the file `file` is open on, read through that handle.
    ///
    /// # Errors
    ///
    /// The OS error when the file's metadata or identity cannot be read.
    pub fn of(file: &File) -> std::io::Result<FileStamp> {
        let metadata = file.metadata()?;
        let id = syntropy_io::os::file_identity(file)?;
        Ok(FileStamp {
            device: id.device,
            inode: id.inode,
            size: metadata.len(),
            mtime: Mtime::of(metadata.modified()?),
        })
    }
}

/// One file of an indexed model: where it is, what it is, its digest and its stamp.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRecord {
    /// Its absolute, canonical path.
    pub path: PathBuf,
    /// Its role.
    pub role: SourceRole,
    /// Its SHA-256.
    pub digest: FileDigest,
    /// Its stamp when it was hashed.
    pub stamp: FileStamp,
    /// Whether the stamp was racy: its mtime was less than [`RACY_MARGIN_MS`] before the pass
    /// began. A racy stamp is never reused (module docs).
    pub racy: bool,
}

/// A local model as the index holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalModel {
    /// Its v1 `ModelId`.
    pub model_id: ModelId,
    /// The family tag `model_id` hashes.
    pub family: String,
    /// Its files, in the order they were given.
    pub files: Vec<SourceRecord>,
}

/// How far an indexing pass has got, given after each read of a file being hashed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    /// Files this pass hashes.
    pub files_total: usize,
    /// Of those, the files fully hashed.
    pub files_done: usize,
    /// Bytes this pass hashes.
    pub bytes_total: u64,
    /// Of those, the bytes hashed.
    pub bytes_done: u64,
}

/// What an indexing pass did.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexReport {
    /// The model, as recorded.
    pub model: LocalModel,
    /// The length of each file this pass hashed, in the order given. Files whose record matched
    /// are not in it.
    pub hashed: Vec<u64>,
    /// Files whose digest came from their record, unread.
    pub reused: usize,
    /// Files recorded with a racy stamp, which the next pass hashes again.
    pub racy: usize,
    /// Threads that hashed files: `min(cores, files hashed)`.
    pub threads: usize,
    /// The pass's wall time on the clock it was given, from the first stamp to the record's
    /// rename.
    pub elapsed: Nanos,
    /// Whether the record on disk holds this model: written by this pass, or already there
    /// unchanged. It is not when a path is not valid Unicode.
    pub recorded: bool,
}

impl IndexReport {
    /// The bytes this pass hashed.
    #[must_use]
    pub fn hashed_bytes(&self) -> u64 {
        self.hashed.iter().sum()
    }

    /// This pass's time set against RFC-0041 §4.3's lower bound for the files it hashed.
    #[must_use]
    pub fn against(&self, rates: &IndexRates) -> TIndex {
        TIndex {
            files: self.hashed.len(),
            bytes: self.hashed_bytes(),
            measured: self.elapsed,
            bound_secs: t_index_bound(&self.hashed, rates),
            rates: *rates,
        }
    }
}

/// The rates RFC-0041 §4.3's `t_index` bound takes. Each comes from a measurement or an
/// estimate, and whoever reports the bound labels which.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexRates {
    /// `W_read`: the volume's sequential read rate.
    pub w_read: BytesPerSec,
    /// `R_sha`: one core's SHA-256 rate.
    pub r_sha: BytesPerSec,
    /// The cores available to the pass.
    pub cores: usize,
}

/// `t_index = max(Σ bytes / W_read, max_f bytes_f / R_sha, Σ bytes / (min(cores, files) · R_sha))`
/// in seconds (RFC-0041 §4.3), for files of lengths `files`. No files take no time. A rate that
/// is not positive gives infinity.
#[must_use]
pub fn t_index_bound(files: &[u64], rates: &IndexRates) -> f64 {
    if files.is_empty() {
        return 0.0;
    }
    let rate = |r: BytesPerSec| if r.0 > 0.0 { r.0 } else { 0.0 };
    let (w_read, r_sha) = (rate(rates.w_read), rate(rates.r_sha));
    let total = files.iter().sum::<u64>() as f64;
    let largest = files.iter().copied().max().unwrap_or(0) as f64;
    let lanes = rates.cores.clamp(1, files.len()) as f64;
    (total / w_read)
        .max(largest / r_sha)
        .max(total / (lanes * r_sha))
}

/// A pass's time against its `t_index` bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TIndex {
    /// Files hashed.
    pub files: usize,
    /// Bytes hashed.
    pub bytes: u64,
    /// The pass's wall time.
    pub measured: Nanos,
    /// The bound, in seconds.
    pub bound_secs: f64,
    /// The rates the bound used.
    pub rates: IndexRates,
}

impl TIndex {
    /// The measured time in seconds.
    #[must_use]
    pub fn measured_secs(&self) -> f64 {
        self.measured.0 as f64 / 1e9
    }

    /// The measured time over the bound: 1 is at the bound. `None` when the bound is 0.
    #[must_use]
    pub fn ratio(&self) -> Option<f64> {
        (self.bound_secs > 0.0).then(|| self.measured_secs() / self.bound_secs)
    }
}

impl fmt::Display for TIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "hashed {} files, {} bytes, in {:.3} s; t_index bound {:.3} s at W_read {:.2} GB/s, \
             R_sha {:.2} GB/s, {} cores",
            self.files,
            self.bytes,
            self.measured_secs(),
            self.bound_secs,
            self.rates.w_read.0 / 1e9,
            self.rates.r_sha.0 / 1e9,
            self.rates.cores,
        )?;
        if let Some(r) = self.ratio() {
            write!(f, "; {r:.2}× the bound")?;
        }
        Ok(())
    }
}

/// Why a local model could not be indexed or checked.
#[non_exhaustive]
#[derive(Debug)]
pub enum LocalError {
    /// A file's (device, inode, size, mtime) differs from its record, or it changed while it
    /// was hashed, or it is gone (`now` is `None`): RFC-0041 §10's `LocalSourceChanged`.
    SourceChanged {
        /// The file.
        path: PathBuf,
        /// Its stamp in the record, or when hashing began.
        was: FileStamp,
        /// Its stamp now, if it can still be read.
        now: Option<FileStamp>,
    },
    /// No model with this ID is indexed.
    NotIndexed(ModelId),
    /// The `ModelId` cannot be computed: an empty family tag, or no weights file.
    Identity(IdentityError),
    /// A file's chunks do not tile it from offset 0 to its end.
    ChunksNotTiled {
        /// The file.
        path: PathBuf,
    },
    /// A file could not be opened, stamped or read.
    Io {
        /// What was being done, such as `"hash"`.
        op: &'static str,
        /// The path.
        path: PathBuf,
        /// The OS error's text.
        message: String,
    },
}

impl LocalError {
    /// The `Formats` code RFC-0041 §10 gives it: 14 `LocalSourceChanged`, or `None` for an error
    /// outside that table.
    #[must_use]
    pub fn code(&self) -> Option<u16> {
        match self {
            LocalError::SourceChanged { .. } => Some(14),
            _ => None,
        }
    }

    /// Its class (RFC-0001 §10). A file that cannot be read is `Integrity`, as `manifest`'s
    /// `RangeError::Io` is: the source is treated as unreadable.
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            LocalError::SourceChanged { .. } | LocalError::Io { .. } => ErrorClass::Integrity,
            LocalError::NotIndexed(_) | LocalError::Identity(_) => ErrorClass::InvalidRequest,
            // The census computed the chunks: a bug there, not in the file.
            LocalError::ChunksNotTiled { .. } => ErrorClass::Internal,
        }
    }

    fn io(op: &'static str, path: &Path, e: impl fmt::Display) -> LocalError {
        LocalError::Io {
            op,
            path: path.to_owned(),
            message: e.to_string(),
        }
    }
}

impl fmt::Display for LocalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LocalError::SourceChanged { path, now, .. } => {
                write!(f, "local model file {} ", path.display())?;
                if now.is_some() {
                    f.write_str("changed since it was indexed")?;
                } else {
                    f.write_str("is gone since it was indexed")?;
                }
                f.write_str("; index it again, which gives the model a new ID")
            }
            LocalError::NotIndexed(id) => write!(f, "no local model {} is indexed", hex(&id.0.0)),
            LocalError::Identity(e) => write!(f, "cannot compute the model ID: {e}"),
            LocalError::ChunksNotTiled { path } => {
                write!(f, "the chunks given for {} do not tile it", path.display())
            }
            LocalError::Io { op, path, message } => {
                write!(f, "cannot {op} {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for LocalError {}

/// The local index of one writable store root.
#[derive(Debug)]
pub struct LocalIndex<'r> {
    root: &'r WritableRoot,
    models: BTreeMap<ModelId, LocalModel>,
    unreadable: Vec<PathBuf>,
}

impl<'r> LocalIndex<'r> {
    /// Reads every record in `root`'s `local/` directory.
    ///
    /// # Errors
    ///
    /// [`LocalError::Io`] when the directory cannot be listed. A record that cannot be read is
    /// not an error; [`unreadable`](Self::unreadable) names it.
    pub fn open(root: &'r WritableRoot) -> Result<LocalIndex<'r>, LocalError> {
        let dir = root.layout().local_dir();
        let entries = std::fs::read_dir(&dir).map_err(|e| LocalError::io("list", &dir, e))?;
        let mut models = BTreeMap::new();
        let mut unreadable = Vec::new();
        let mut paths: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "sources"))
            .collect();
        paths.sort();
        for path in paths {
            match read_record(&path) {
                Some(model) => {
                    models.insert(model.model_id, model);
                }
                None => unreadable.push(path),
            }
        }
        Ok(LocalIndex {
            root,
            models,
            unreadable,
        })
    }

    /// The indexed model `id`, if any.
    #[must_use]
    pub fn get(&self, id: &ModelId) -> Option<&LocalModel> {
        self.models.get(id)
    }

    /// Every indexed model, by ID.
    pub fn models(&self) -> impl Iterator<Item = &LocalModel> {
        self.models.values()
    }

    /// Records in `local/` that could not be read when the index was opened.
    #[must_use]
    pub fn unreadable(&self) -> &[PathBuf] {
        &self.unreadable
    }

    /// The chunk table of the source file whose SHA-256 is `digest`, from
    /// `tables/<sha256>.chunks` (RFC-0041 §4.1, §6.2).
    ///
    /// # Errors
    ///
    /// [`LocalError::Io`] when the table cannot be read or is not a valid table of that file.
    pub fn chunk_table(&self, digest: FileDigest) -> Result<ChunkTable, LocalError> {
        let path = self.root.layout().chunk_table(&digest.0);
        let bytes = std::fs::read(&path).map_err(|e| LocalError::io("read", &path, e))?;
        let table = ChunkTable::decode(&bytes).map_err(|e| LocalError::io("read", &path, e))?;
        if table.digest != digest {
            return Err(LocalError::io(
                "read",
                &path,
                "the table names another file's digest",
            ));
        }
        Ok(table)
    }

    /// Whether the file `digest` has a table on disk that matches `chunks` and holds every
    /// chunk's ID: what lets a later start skip the file.
    fn table_is_complete(&self, digest: FileDigest, chunks: &[Chunk]) -> bool {
        self.chunk_table(digest)
            .is_ok_and(|t| t.matches(chunks) && t.records.iter().all(|r| r.committed.is_some()))
    }

    /// Checks every file of model `id` against its record. It compares stamps only, so a
    /// same-tick write to a file recorded as racy passes it; the next [`index`](Self::index)
    /// hashes such a file again (module docs).
    ///
    /// # Errors
    ///
    /// - [`LocalError::SourceChanged`] for the first file whose stamp differs, or that is gone.
    /// - [`LocalError::NotIndexed`] when no model `id` is indexed.
    pub fn check(&self, id: &ModelId) -> Result<(), LocalError> {
        let model = self.models.get(id).ok_or(LocalError::NotIndexed(*id))?;
        for file in &model.files {
            let now = File::open(&file.path)
                .ok()
                .and_then(|f| FileStamp::of(&f).ok());
            if now != Some(file.stamp) {
                return Err(LocalError::SourceChanged {
                    path: file.path.clone(),
                    was: file.stamp,
                    now,
                });
            }
        }
        Ok(())
    }

    /// Indexes the model of family `family_tag` made of `files`: reuses each digest whose record
    /// still matches, hashes the rest, and records the result (module docs). `progress` is called
    /// from the hashing threads after each read. `clock` times the pass, and its wall time at
    /// the start decides which stamps are racy.
    ///
    /// # Errors
    ///
    /// - [`LocalError::Io`] when a file cannot be opened, stamped or read, or the record cannot
    ///   be written.
    /// - [`LocalError::SourceChanged`] when a file changed while it was hashed.
    /// - [`LocalError::Identity`] for an empty tag or no weights file.
    pub fn index(
        &mut self,
        family_tag: &str,
        files: &[LocalFile],
        clock: &dyn Clock,
        progress: &(dyn Fn(Progress) + Sync),
    ) -> Result<IndexReport, LocalError> {
        let start = clock.now();
        // Taken before any stamp, so a pass longer than the margin cannot let a racy file
        // through.
        let racy_after = i128::from(clock.unix_ms()) - i128::from(RACY_MARGIN_MS);

        // Stamp every file, and find the ones whose record still matches and was not racy. Each
        // file is closed again at once; the hashing threads reopen the ones they hash.
        // Each path's record, a settled one in preference to a racy one when two models name it.
        let mut known: BTreeMap<&Path, &SourceRecord> = BTreeMap::new();
        for f in self.models.values().flat_map(|m| m.files.iter()) {
            let entry = known.entry(f.path.as_path()).or_insert(f);
            if entry.racy && !f.racy {
                *entry = f;
            }
        }
        let mut opened = Vec::with_capacity(files.len());
        for file in files {
            let path = std::fs::canonicalize(&file.path)
                .map_err(|e| LocalError::io("resolve", &file.path, e))?;
            let stamp = {
                let handle = File::open(&path).map_err(|e| LocalError::io("open", &path, e))?;
                FileStamp::of(&handle).map_err(|e| LocalError::io("stamp", &path, e))?
            };
            let chunks = match &file.chunks {
                Some(chunks) => chunks.clone(),
                None => {
                    chunk_file(stamp.size, stamp.size, []).expect("a file is all header region")
                }
            };
            if !tiles(&chunks, stamp.size) {
                return Err(LocalError::ChunksNotTiled { path });
            }
            // A settled record is reused only with its complete chunk table, matching these
            // chunks; a racy one never (module docs).
            let cached = known
                .get(path.as_path())
                .filter(|r| !r.racy && r.stamp == stamp)
                .filter(|r| self.table_is_complete(r.digest, &chunks))
                .map(|r| r.digest);
            opened.push(Opened {
                path,
                role: file.role,
                stamp,
                racy: stamp.mtime.unix_ms() >= racy_after,
                chunks,
                digest: cached,
            });
        }

        // Hash the files with no matching record, one file per thread at a time.
        let todo: Vec<usize> = (0..opened.len())
            .filter(|&i| opened[i].digest.is_none())
            .collect();
        let hashed: Vec<u64> = todo.iter().map(|&i| opened[i].stamp.size).collect();
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        let threads = cores.min(todo.len());
        let digests = hash_files(&opened, &todo, threads, self.root.layout(), progress)?;
        for (&i, digest) in todo.iter().zip(digests) {
            opened[i].digest = Some(digest);
        }

        let records: Vec<SourceRecord> = opened
            .into_iter()
            .map(|o| SourceRecord {
                path: o.path,
                role: o.role,
                digest: o.digest.expect("every file has a digest by now"),
                stamp: o.stamp,
                racy: o.racy,
            })
            .collect();
        let racy = records.iter().filter(|r| r.racy).count();
        let sources: Vec<SourceFile> = records
            .iter()
            .map(|r| SourceFile {
                role: r.role,
                digest: r.digest,
                len: r.stamp.size,
            })
            .collect();
        let id = model_id(family_tag, &sources).map_err(LocalError::Identity)?;
        let model = LocalModel {
            model_id: id,
            family: family_tag.to_owned(),
            files: records,
        };

        // A record naming one of these paths with another stamp or digest is stale: drop it. A
        // racy record whose file was hashed again goes this way when the bytes had changed.
        let stale: Vec<ModelId> = self
            .models
            .values()
            .filter(|m| m.model_id != id)
            .filter(|m| {
                m.files.iter().any(|old| {
                    model.files.iter().any(|new| {
                        new.path == old.path && (new.stamp != old.stamp || new.digest != old.digest)
                    })
                })
            })
            .map(|m| m.model_id)
            .collect();
        for old in stale {
            self.models.remove(&old);
            let path = self.root.layout().local_sources(&old.0.0);
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(LocalError::io("remove", &path, e)),
            }
        }

        // A record that already says this is left alone, so a start that reuses every digest
        // writes nothing.
        let unchanged = self.models.get(&id) == Some(&model)
            && self.root.layout().local_sources(&id.0.0).is_file();
        let recorded = unchanged || self.write_record(&model)?;
        self.models.insert(id, model.clone());
        Ok(IndexReport {
            model,
            reused: files.len() - todo.len(),
            racy,
            hashed,
            threads,
            elapsed: clock.now().saturating_duration_since(start),
            recorded,
        })
    }

    /// Writes `model`'s record atomically. Returns `false`, writing nothing, when a path is not
    /// valid Unicode.
    fn write_record(&self, model: &LocalModel) -> Result<bool, LocalError> {
        let Some(text) = encode(model) else {
            return Ok(false);
        };
        let layout = self.root.layout();
        let path = layout.local_sources(&model.model_id.0.0);
        let tmp = layout
            .local_dir()
            .join(format!(".{}.sources.tmp", hex(&model.model_id.0.0)));
        let mut file = File::create(&tmp).map_err(|e| LocalError::io("create", &tmp, e))?;
        file.write_all(text.as_bytes())
            .and_then(|()| file.sync_all())
            .map_err(|e| LocalError::io("write", &tmp, e))?;
        drop(file);
        std::fs::rename(&tmp, &path).map_err(|e| LocalError::io("rename", &path, e))?;
        let dir = layout.local_dir();
        syntropy_io::os::sync_dir(&dir).map_err(|e| LocalError::io("sync", &dir, e))?;
        Ok(true)
    }
}

/// A file stamped for indexing.
struct Opened {
    path: PathBuf,
    role: SourceRole,
    stamp: FileStamp,
    racy: bool,
    chunks: Vec<Chunk>,
    digest: Option<FileDigest>,
}

/// Hashes `opened[i]` for each `i` in `todo` on `threads` threads, each taking the next file
/// when it finishes one. Returns the digests in `todo`'s order.
fn hash_files(
    opened: &[Opened],
    todo: &[usize],
    threads: usize,
    layout: &Layout,
    progress: &(dyn Fn(Progress) + Sync),
) -> Result<Vec<FileDigest>, LocalError> {
    let files_total = todo.len();
    let bytes_total = todo.iter().map(|&i| opened[i].stamp.size).sum();
    let next = AtomicUsize::new(0);
    let files_done = AtomicUsize::new(0);
    let bytes_done = AtomicU64::new(0);
    let results: Vec<Mutex<Option<Result<FileDigest, LocalError>>>> =
        todo.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&i) = todo.get(k) else { break };
                    let result = hash_one(&opened[i], layout, |n| {
                        let done = bytes_done.fetch_add(n, Ordering::Relaxed) + n;
                        progress(Progress {
                            files_total,
                            files_done: files_done.load(Ordering::Relaxed),
                            bytes_total,
                            bytes_done: done,
                        });
                    });
                    let failed = result.is_err();
                    *results[k].lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
                    files_done.fetch_add(1, Ordering::Relaxed);
                    if failed {
                        // Stop handing out files; the others finish the file they hold.
                        next.store(todo.len(), Ordering::Relaxed);
                        break;
                    }
                }
            });
        }
    });
    if files_total > 0 {
        progress(Progress {
            files_total,
            files_done: files_done.load(Ordering::Relaxed),
            bytes_total,
            bytes_done: bytes_done.load(Ordering::Relaxed),
        });
    }
    // Files are handed out in order and none after a failure, so with no error every slot is
    // filled; with one, the first error in order is returned.
    let mut digests = Vec::with_capacity(todo.len());
    for slot in results {
        match slot.into_inner().unwrap_or_else(|e| e.into_inner()) {
            Some(Ok(digest)) => digests.push(digest),
            Some(Err(e)) => return Err(e),
            None => {}
        }
    }
    Ok(digests)
}

/// Opens one stamped file, checks through the new handle that it is the file that was stamped,
/// hashes it and its chunks in one pass, checks that its stamp did not change meanwhile, and
/// writes its chunk table under `layout`.
fn hash_one(
    file: &Opened,
    layout: &Layout,
    on_read: impl FnMut(u64),
) -> Result<FileDigest, LocalError> {
    let handle = File::open(&file.path).map_err(|e| LocalError::io("open", &file.path, e))?;
    if FileStamp::of(&handle).ok() != Some(file.stamp) {
        return Err(changed(file, &handle));
    }
    let source = FileSource::from_file(handle).map_err(|e| range_error(&file.path, e))?;
    if source.len() != file.stamp.size {
        return Err(changed(file, source.file()));
    }
    let (digest, ids) = hash_chunks(&source, &file.chunks, on_read).map_err(|e| match e {
        HashError::Read(RangeError::OutOfRange { .. }) => changed(file, source.file()),
        HashError::Read(e) => range_error(&file.path, e),
        HashError::NotTiled => LocalError::ChunksNotTiled {
            path: file.path.clone(),
        },
    })?;
    if FileStamp::of(source.file()).ok() != Some(file.stamp) {
        return Err(changed(file, source.file()));
    }
    let mut table = ChunkTable::new(digest, file.stamp.size, &file.chunks);
    for (record, id) in table.records.iter_mut().zip(ids) {
        record.committed = Some((id, EvidenceForm::Provisional));
    }
    write_table(layout, &table)?;
    Ok(digest)
}

/// Whether `chunks` tile `[0, len)`.
fn tiles(chunks: &[Chunk], len: u64) -> bool {
    let mut at = 0u64;
    for c in chunks {
        if c.offset != at || c.len == 0 {
            return false;
        }
        at = c.end();
    }
    at == len
}

/// Writes `table` under a temporary name, syncs it, renames it into place and syncs the
/// directory. A table names its file by digest, so two models that share a file share its table.
fn write_table(layout: &Layout, table: &ChunkTable) -> Result<(), LocalError> {
    let tables = layout.tables_dir();
    let path = layout.chunk_table(&table.digest.0);
    // A per-thread temporary name: two hashing threads may write tables at once.
    let tmp = tables.join(format!(
        ".{}.{:?}.chunks.tmp",
        table.digest.to_hex(),
        std::thread::current().id()
    ));
    let mut file = File::create(&tmp).map_err(|e| LocalError::io("create", &tmp, e))?;
    file.write_all(&table.encode())
        .and_then(|()| file.sync_all())
        .map_err(|e| LocalError::io("write", &tmp, e))?;
    drop(file);
    std::fs::rename(&tmp, &path).map_err(|e| LocalError::io("rename", &path, e))?;
    syntropy_io::os::sync_dir(&tables).map_err(|e| LocalError::io("sync", &tables, e))
}

fn changed(file: &Opened, handle: &File) -> LocalError {
    LocalError::SourceChanged {
        path: file.path.clone(),
        was: file.stamp,
        now: FileStamp::of(handle).ok(),
    }
}

fn range_error(path: &Path, e: RangeError) -> LocalError {
    LocalError::io("hash", path, e)
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The record's version.
const VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordText {
    version: u32,
    model_id: String,
    family: String,
    files: Vec<FileText>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileText {
    path: String,
    role: String,
    sha256: String,
    size: u64,
    device: u64,
    inode: u64,
    mtime_secs: i64,
    mtime_nanos: u32,
    racy: bool,
}

fn encode(model: &LocalModel) -> Option<String> {
    let files = model
        .files
        .iter()
        .map(|f| {
            Some(FileText {
                path: f.path.to_str()?.to_owned(),
                role: f.role.name().to_owned(),
                sha256: f.digest.to_hex(),
                size: f.stamp.size,
                device: f.stamp.device,
                inode: f.stamp.inode,
                mtime_secs: f.stamp.mtime.secs,
                mtime_nanos: f.stamp.mtime.nanos,
                racy: f.racy,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let text = RecordText {
        version: VERSION,
        model_id: hex(&model.model_id.0.0),
        family: model.family.clone(),
        files,
    };
    serde_json::to_string_pretty(&text).ok().map(|mut s| {
        s.push('\n');
        s
    })
}

/// Reads one record, or `None` if it is not a valid v1 record whose name, contents and
/// `ModelId` agree.
fn read_record(path: &Path) -> Option<LocalModel> {
    let text = std::fs::read_to_string(path).ok()?;
    decode(&text, path.file_stem()?.to_str()?)
}

fn decode(text: &str, stem: &str) -> Option<LocalModel> {
    let record: RecordText = serde_json::from_str(text).ok()?;
    if record.version != VERSION || record.model_id != stem {
        return None;
    }
    let files = record
        .files
        .into_iter()
        .map(|f| {
            if f.mtime_nanos >= 1_000_000_000 {
                return None;
            }
            Some(SourceRecord {
                path: PathBuf::from(f.path),
                role: SourceRole::from_name(&f.role)?,
                digest: FileDigest::from_hex(&f.sha256)?,
                stamp: FileStamp {
                    device: f.device,
                    inode: f.inode,
                    size: f.size,
                    mtime: Mtime {
                        secs: f.mtime_secs,
                        nanos: f.mtime_nanos,
                    },
                },
                racy: f.racy,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let sources: Vec<SourceFile> = files
        .iter()
        .map(|f| SourceFile {
            role: f.role,
            digest: f.digest,
            len: f.stamp.size,
        })
        .collect();
    let id = model_id(&record.family, &sources).ok()?;
    if hex(&id.0.0) != record.model_id {
        return None;
    }
    Some(LocalModel {
        model_id: id,
        family: record.family,
        files,
    })
}

#[cfg(test)]
mod tests;
