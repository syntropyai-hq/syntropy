// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the local index (COMP-E11-T03; RFC-0041 §4.3, §10).
//!
//! Every file is written by the test into a fresh directory. A "restart" is a new
//! [`LocalIndex`] opened on the same root, which reads only what the last one recorded.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use syntropy_core::{Clock, MonoTime};

use super::*;
use syntropy_core::Blake3;
use syntropy_manifest::chunk::{Chunk, EvidenceForm};

/// A clock whose monotonic time moves on 1 ms at each reading, at a fixed virtual wall time.
struct StepClock {
    mono: AtomicU64,
    unix_ms: u64,
}

/// 2100-01-01T00:00:00Z: a wall time long after any file a test writes, so no stamp is racy.
const LATER_MS: u64 = 4_102_444_800_000;

impl StepClock {
    /// At [`LATER_MS`].
    fn new() -> StepClock {
        StepClock::at(LATER_MS)
    }

    /// At wall time `unix_ms`.
    fn at(unix_ms: u64) -> StepClock {
        StepClock {
            mono: AtomicU64::new(0),
            unix_ms,
        }
    }
}

impl Clock for StepClock {
    fn now(&self) -> MonoTime {
        MonoTime(self.mono.fetch_add(1_000_000, Ordering::Relaxed))
    }

    fn unix_ms(&self) -> u64 {
        self.unix_ms
    }
}

/// `path`'s mtime in milliseconds from the epoch.
fn mtime_ms(path: &Path) -> u64 {
    let stamp = FileStamp::of(&File::open(path).unwrap()).unwrap();
    u64::try_from(stamp.mtime.unix_ms()).unwrap()
}

/// A fresh directory, removed on drop, with a store root at `root/` and model files beside it.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "syntropy-store-local-{tag}-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("models")).unwrap();
        Scratch(dir)
    }

    fn root(&self) -> WritableRoot {
        WritableRoot::open(self.0.join("root")).unwrap()
    }

    fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join("models").join(name);
        std::fs::write(&path, bytes).unwrap();
        std::fs::canonicalize(path).unwrap()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn bytes(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn weights(path: &Path) -> LocalFile {
    LocalFile {
        path: path.to_owned(),
        role: SourceRole::Weights,
        chunks: None,
    }
}

fn quiet(_: Progress) {}

fn expected_id(tag: &str, files: &[(SourceRole, &[u8])]) -> ModelId {
    let sources: Vec<SourceFile> = files
        .iter()
        .map(|(role, b)| SourceFile {
            role: *role,
            digest: FileDigest::of_bytes(b),
            len: b.len() as u64,
        })
        .collect();
    model_id(tag, &sources).unwrap()
}

#[test]
fn indexing_gives_the_v1_model_id_and_records_it() {
    let s = Scratch::new("basic");
    let a = bytes(1, 3 << 20);
    let b = bytes(2, (1 << 20) + 7);
    let cfg = br#"{"architectures":["LlamaForCausalLM"]}"#.to_vec();
    let files = [
        weights(&s.file("model-00001-of-00002.safetensors", &a)),
        weights(&s.file("model-00002-of-00002.safetensors", &b)),
        LocalFile {
            path: s.file("config.json", &cfg),
            role: SourceRole::Config,
            chunks: None,
        },
        LocalFile {
            path: s.file("tokenizer.json", b"{}"),
            role: SourceRole::Tokenizer,
            chunks: None,
        },
    ];
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    let report = index
        .index("llama", &files, &StepClock::new(), &quiet)
        .unwrap();
    let want = expected_id(
        "llama",
        &[
            (SourceRole::Weights, &a),
            (SourceRole::Weights, &b),
            (SourceRole::Config, &cfg),
        ],
    );
    assert_eq!(report.model.model_id, want);
    assert_eq!(report.hashed.len(), 4);
    assert_eq!(report.reused, 0);
    assert!(report.recorded);
    assert!(report.threads >= 1 && report.threads <= 4);
    assert!(report.elapsed.0 > 0);
    let path = root.layout().local_sources(&want.0.0);
    assert!(path.is_file(), "{}", path.display());
    assert_eq!(index.get(&want), Some(&report.model));
    index.check(&want).unwrap();
}

#[test]
fn a_second_start_reads_the_record_and_hashes_nothing() {
    let s = Scratch::new("restart");
    let data = bytes(3, 2 << 20);
    let path = s.file("model.gguf", &data);
    let first = {
        let root = s.root();
        let mut index = LocalIndex::open(&root).unwrap();
        index
            .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
            .unwrap()
    };
    // The root is closed; a new process opens it.
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    assert_eq!(index.get(&first.model.model_id), Some(&first.model));
    let calls = AtomicUsize::new(0);
    let second = index
        .index("llama", &[weights(&path)], &StepClock::new(), &|_| {
            calls.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    assert_eq!(second.model.model_id, first.model.model_id);
    assert_eq!(second.hashed, Vec::<u64>::new());
    assert_eq!(second.reused, 1);
    assert_eq!(second.threads, 0);
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "no progress: nothing was read"
    );
    // Nor was the unchanged record written again: its mtime, set by hand, stays.
    let record = root.layout().local_sources(&first.model.model_id.0.0);
    let marker = UNIX_EPOCH + std::time::Duration::from_secs(1);
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(&record)
        .unwrap();
    f.set_modified(marker).unwrap();
    drop(f);
    let third = index
        .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
        .unwrap();
    assert!(third.recorded);
    assert_eq!(
        std::fs::metadata(&record).unwrap().modified().unwrap(),
        marker
    );
}

/// Proof that the second start does not read the file: its bytes are rewritten in place with the
/// same length and the mtime set back, which leaves the stamp unchanged, and the index still
/// gives the first digest. (This is also the stamp's documented limit.)
#[test]
fn a_matching_stamp_is_trusted_without_reading_the_file() {
    let s = Scratch::new("trusted");
    let data = bytes(4, 1 << 20);
    let path = s.file("model.gguf", &data);
    let first = {
        let root = s.root();
        let mut index = LocalIndex::open(&root).unwrap();
        index
            .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
            .unwrap()
    };
    let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
    {
        let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all(&bytes(99, 1 << 20)).unwrap();
        f.set_modified(mtime).unwrap();
        f.sync_all().unwrap();
    }
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    let second = index
        .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
        .unwrap();
    assert_eq!(second.model.files[0].digest, FileDigest::of_bytes(&data));
    assert_eq!(second.model.model_id, first.model.model_id);
}

#[test]
fn modifying_a_file_is_local_source_changed_and_reindexing_gives_a_new_id() {
    let s = Scratch::new("changed");
    let a = bytes(5, 1 << 20);
    let b = bytes(6, 1 << 20);
    let pa = s.file("model-00001-of-00002.gguf", &a);
    let pb = s.file("model-00002-of-00002.gguf", &b);
    let files = [weights(&pa), weights(&pb)];
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    let first = index
        .index("llama", &files, &StepClock::new(), &quiet)
        .unwrap();
    let old = first.model.model_id;

    // A user edits one file: one byte more.
    let mut edited = b.clone();
    edited.push(0);
    std::fs::write(&pb, &edited).unwrap();

    let err = index.check(&old).unwrap_err();
    assert_eq!(err.code(), Some(14));
    assert_eq!(err.class(), ErrorClass::Integrity);
    match &err {
        LocalError::SourceChanged { path, was, now } => {
            assert_eq!(path, &pb);
            assert_eq!(was.size, b.len() as u64);
            assert_eq!(now.map(|n| n.size), Some(edited.len() as u64));
        }
        other => panic!("expected SourceChanged, got {other:?}"),
    }
    assert!(err.to_string().contains("index it again"), "{err}");

    // The next use re-indexes: only the changed file is hashed, and the ID is new.
    let second = index
        .index("llama", &files, &StepClock::new(), &quiet)
        .unwrap();
    let new = second.model.model_id;
    assert_ne!(new, old);
    assert_eq!(
        new,
        expected_id(
            "llama",
            &[(SourceRole::Weights, &a), (SourceRole::Weights, &edited)]
        )
    );
    assert_eq!(second.hashed, vec![edited.len() as u64]);
    assert_eq!(second.reused, 1);
    index.check(&new).unwrap();

    // The stale record is gone, from memory and from disk.
    assert_eq!(index.get(&old), None);
    assert!(!root.layout().local_sources(&old.0.0).exists());
    assert!(root.layout().local_sources(&new.0.0).is_file());
    drop(index);
    let reopened = LocalIndex::open(&root).unwrap();
    assert_eq!(reopened.models().count(), 1);
}

#[test]
fn a_replaced_or_removed_file_is_local_source_changed() {
    let s = Scratch::new("replaced");
    let data = bytes(7, 4096);
    let path = s.file("model.gguf", &data);
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    let id = index
        .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
        .unwrap()
        .model
        .model_id;

    // The same bytes under a new inode: written beside it and renamed over it.
    let tmp = s.file("model.gguf.new", &data);
    std::fs::rename(&tmp, &path).unwrap();
    let err = index.check(&id).unwrap_err();
    assert_eq!(err.code(), Some(14));

    std::fs::remove_file(&path).unwrap();
    match index.check(&id).unwrap_err() {
        LocalError::SourceChanged { now: None, .. } => {}
        other => panic!("expected SourceChanged with no stamp, got {other:?}"),
    }
}

#[test]
fn a_file_that_changes_while_it_is_hashed_is_refused() {
    let s = Scratch::new("racing");
    let path = s.file("model.gguf", &bytes(8, 3 << 20));
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    let touched = AtomicBool::new(false);
    let err = index
        .index("llama", &[weights(&path)], &StepClock::new(), &|_| {
            if !touched.swap(true, Ordering::Relaxed) {
                let mut f = std::fs::OpenOptions::new()
                    .append(true)
                    .open(&path)
                    .unwrap();
                f.write_all(b"more").unwrap();
            }
        })
        .unwrap_err();
    assert_eq!(err.code(), Some(14), "{err:?}");
    assert_eq!(index.models().count(), 0);
}

#[test]
fn files_are_hashed_side_by_side_and_progress_reaches_the_total() {
    let s = Scratch::new("parallel");
    let datas: Vec<Vec<u8>> = (0..6)
        .map(|i| bytes(10 + i, (2 << 20) + i as usize))
        .collect();
    let files: Vec<LocalFile> = datas
        .iter()
        .enumerate()
        .map(|(i, d)| weights(&s.file(&format!("part-{i}.gguf"), d)))
        .collect();
    let total: u64 = datas.iter().map(|d| d.len() as u64).sum();
    let seen = Mutex::new(Vec::new());
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    let report = index
        .index("llama", &files, &StepClock::new(), &|p| {
            seen.lock().unwrap().push(p);
        })
        .unwrap();
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    assert_eq!(report.threads, cores.min(6));
    assert_eq!(report.hashed_bytes(), total);
    let seen = seen.into_inner().unwrap();
    let last = *seen.last().unwrap();
    assert_eq!(
        last,
        Progress {
            files_total: 6,
            files_done: 6,
            bytes_total: total,
            bytes_done: total,
        }
    );
    assert!(
        seen.iter()
            .all(|p| p.bytes_done <= total && p.files_total == 6)
    );
    let refs: Vec<(SourceRole, &[u8])> = datas
        .iter()
        .map(|d| (SourceRole::Weights, d.as_slice()))
        .collect();
    assert_eq!(report.model.model_id, expected_id("llama", &refs));
}

#[test]
fn another_spelling_of_a_path_finds_the_same_record() {
    let s = Scratch::new("spelling");
    let path = s.file("model.gguf", &bytes(20, 4096));
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    index
        .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
        .unwrap();
    // `models/./model.gguf` and `models/../models/model.gguf` name the same file.
    let dir = path.parent().unwrap();
    for spelling in [
        dir.join(".").join("model.gguf"),
        dir.join("..").join("models").join("model.gguf"),
    ] {
        let again = index
            .index("llama", &[weights(&spelling)], &StepClock::new(), &quiet)
            .unwrap();
        assert_eq!(again.reused, 1, "{}", spelling.display());
        assert_eq!(again.model.files[0].path, path);
    }
}

#[test]
fn a_bad_record_is_ignored_and_named() {
    let s = Scratch::new("bad");
    let path = s.file("model.gguf", &bytes(21, 4096));
    let root = s.root();
    let id = LocalIndex::open(&root)
        .unwrap()
        .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
        .unwrap()
        .model
        .model_id;
    let good = std::fs::read_to_string(root.layout().local_sources(&id.0.0)).unwrap();

    let local = root.layout().local_dir();
    // Not JSON; a valid record under another model's name; a record whose digest does not give
    // its ID; an unknown field; an unknown role.
    std::fs::write(
        local.join(format!("{}.sources", "0".repeat(64))),
        "not json",
    )
    .unwrap();
    std::fs::write(local.join(format!("{}.sources", "1".repeat(64))), &good).unwrap();
    let wrong_digest = good.replacen(
        &FileDigest::of_bytes(&bytes(21, 4096)).to_hex(),
        &"2".repeat(64),
        1,
    );
    std::fs::write(
        local.join(format!("{}.sources", "3".repeat(64))),
        &wrong_digest,
    )
    .unwrap();
    let extra = good.replacen("\"version\": 1,", "\"version\": 1, \"extra\": 0,", 1);
    std::fs::write(local.join(format!("{}.sources", "4".repeat(64))), &extra).unwrap();
    let role = good.replacen("\"weights\"", "\"lora\"", 1);
    std::fs::write(local.join(format!("{}.sources", "5".repeat(64))), &role).unwrap();

    let index = LocalIndex::open(&root).unwrap();
    assert_eq!(index.models().count(), 1);
    assert!(index.get(&id).is_some());
    assert_eq!(index.unreadable().len(), 5, "{:?}", index.unreadable());
}

#[test]
fn the_record_round_trips() {
    let model = LocalModel {
        model_id: expected_id("gemma", &[(SourceRole::Weights, b"w")]),
        family: "gemma".into(),
        files: vec![SourceRecord {
            path: PathBuf::from("/models/m.gguf"),
            role: SourceRole::Weights,
            digest: FileDigest::of_bytes(b"w"),
            stamp: FileStamp {
                device: 66,
                inode: 1 << 40,
                size: 1,
                mtime: Mtime {
                    secs: -5,
                    nanos: 999_999_999,
                },
            },
            racy: true,
        }],
    };
    let text = encode(&model).unwrap();
    assert_eq!(
        decode(&text, &hex(&model.model_id.0.0)),
        Some(model.clone())
    );
    assert_eq!(decode(&text, &"0".repeat(64)), None);
}

/// A file written just before the pass starts has a racy stamp: a same-tick write after it would
/// leave the stamp unchanged. It is hashed again at the next start, which finds such a write.
#[test]
fn a_racy_stamp_is_hashed_again_at_the_next_start() {
    let s = Scratch::new("racy");
    let data = bytes(30, 1 << 20);
    let path = s.file("model.gguf", &data);
    let written = mtime_ms(&path);
    let root = s.root();

    // The pass starts 500 ms after the file's last write, inside the 2 s margin.
    let first = LocalIndex::open(&root)
        .unwrap()
        .index(
            "llama",
            &[weights(&path)],
            &StepClock::at(written + 500),
            &quiet,
        )
        .unwrap();
    assert_eq!(first.racy, 1);
    assert!(first.model.files[0].racy);

    // A same-tick rewrite at the same size: the bytes change, the stamp does not.
    let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
    let edited = bytes(31, 1 << 20);
    {
        let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all(&edited).unwrap();
        f.set_modified(mtime).unwrap();
    }
    assert_eq!(
        FileStamp::of(&File::open(&path).unwrap()).unwrap(),
        first.model.files[0].stamp
    );

    // The next start, later: the racy record is not reused, so the new bytes are hashed.
    let mut index = LocalIndex::open(&root).unwrap();
    assert!(index.get(&first.model.model_id).unwrap().files[0].racy);
    let second = index
        .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
        .unwrap();
    assert_eq!(second.hashed, vec![edited.len() as u64]);
    assert_eq!(second.racy, 0);
    assert_eq!(second.model.files[0].digest, FileDigest::of_bytes(&edited));
    assert_ne!(second.model.model_id, first.model.model_id);

    // Now settled, the record is reused.
    let third = index
        .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
        .unwrap();
    assert_eq!((third.hashed.len(), third.reused), (0, 1));

    // Exactly at the margin is still racy; one millisecond past it is not.
    let edge = |offset: u64| {
        let s = Scratch::new("racy-edge");
        let path = s.file("model.gguf", &data);
        let root = s.root();
        let clock = StepClock::at(mtime_ms(&path) + offset);
        LocalIndex::open(&root)
            .unwrap()
            .index("llama", &[weights(&path)], &clock, &quiet)
            .unwrap()
            .racy
    };
    assert_eq!(edge(RACY_MARGIN_MS as u64), 1);
    assert_eq!(edge(RACY_MARGIN_MS as u64 + 1), 0);
}

/// A change that keeps the size, a new mtime or a new inode, is caught by `index` itself, not only
/// by `check`.
#[test]
fn index_hashes_a_same_size_change_and_a_new_inode_again() {
    let s = Scratch::new("same-size");
    let data = bytes(40, 1 << 20);
    let path = s.file("model.gguf", &data);
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    let first = index
        .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
        .unwrap();
    let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();

    // Same size, new bytes, a new mtime.
    let edited = bytes(41, 1 << 20);
    {
        let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all(&edited).unwrap();
        f.set_modified(mtime + std::time::Duration::from_secs(3))
            .unwrap();
    }
    let second = index
        .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
        .unwrap();
    assert_eq!(second.hashed, vec![edited.len() as u64]);
    assert_eq!(second.model.files[0].digest, FileDigest::of_bytes(&edited));
    assert_ne!(second.model.model_id, first.model.model_id);

    // The same bytes and mtime under a new inode: hashed again, the same ID.
    let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
    let copy = s.file("model.gguf.copy", &edited);
    File::options()
        .write(true)
        .open(&copy)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
    std::fs::rename(&copy, &path).unwrap();
    let third = index
        .index("llama", &[weights(&path)], &StepClock::new(), &quiet)
        .unwrap();
    assert_ne!(
        third.model.files[0].stamp.inode,
        second.model.files[0].stamp.inode
    );
    assert_eq!(third.hashed, vec![edited.len() as u64]);
    assert_eq!(third.model.model_id, second.model.model_id);
}

#[test]
fn mtimes_before_the_epoch_count_forward_from_the_second() {
    use std::time::Duration;
    // Offsets in whole 100 ns, the resolution of Windows' `SystemTime`.
    let at = |d: Duration| Mtime::of(UNIX_EPOCH - d);
    assert_eq!(at(Duration::new(5, 0)), Mtime { secs: -5, nanos: 0 });
    assert_eq!(
        at(Duration::new(5, 100)),
        Mtime {
            secs: -6,
            nanos: 999_999_900
        }
    );
    assert_eq!(
        Mtime::of(UNIX_EPOCH + Duration::new(7, 300)),
        Mtime {
            secs: 7,
            nanos: 300
        }
    );
    assert_eq!(
        Mtime {
            secs: 7,
            nanos: 300
        }
        .unix_ms(),
        7_000
    );
    assert_eq!(
        Mtime {
            secs: -6,
            nanos: 999_999_900
        }
        .unix_ms(),
        -5_001
    );
}

/// RFC-0041 §4.3's two worked cases [estimates there; here only the formula's arithmetic].
#[test]
fn the_t_index_bound_matches_the_rfc_s_worked_cases() {
    // Nine 45 GB split files at W_read 6 GB/s and R_sha 2 GB/s on 16 cores: drive-bound, 67.5 s.
    let rates = IndexRates {
        w_read: BytesPerSec(6e9),
        r_sha: BytesPerSec(2e9),
        cores: 16,
    };
    let split = vec![45_000_000_000u64; 9];
    assert!((t_index_bound(&split, &rates) - 67.5).abs() < 1e-9);
    // One unsplit 120 GB file at R_sha 2.5 GB/s: 48 s whatever the drive.
    let fast_drive = IndexRates {
        w_read: BytesPerSec(14e9),
        r_sha: BytesPerSec(2.5e9),
        cores: 16,
    };
    assert!((t_index_bound(&[120_000_000_000], &fast_drive) - 48.0).abs() < 1e-9);
    // Few cores: the third term. Four 10 GB files on two cores at 2 GB/s, a fast drive: 10 s.
    let two = IndexRates {
        w_read: BytesPerSec(100e9),
        r_sha: BytesPerSec(2e9),
        cores: 2,
    };
    assert!((t_index_bound(&[10_000_000_000; 4], &two) - 10.0).abs() < 1e-9);
    assert_eq!(t_index_bound(&[], &rates), 0.0);
    let zero = IndexRates {
        w_read: BytesPerSec(0.0),
        ..rates
    };
    assert!(t_index_bound(&[1], &zero).is_infinite());
}

#[test]
fn a_report_is_set_against_its_bound() {
    let report = IndexReport {
        model: LocalModel {
            model_id: expected_id("llama", &[(SourceRole::Weights, b"w")]),
            family: "llama".into(),
            files: Vec::new(),
        },
        hashed: vec![4_000_000_000, 2_000_000_000],
        reused: 1,
        racy: 0,
        threads: 2,
        elapsed: Nanos(3_000_000_000),
        recorded: true,
    };
    let rates = IndexRates {
        w_read: BytesPerSec(3e9),
        r_sha: BytesPerSec(2e9),
        cores: 8,
    };
    let t = report.against(&rates);
    assert_eq!((t.files, t.bytes), (2, 6_000_000_000));
    assert!((t.bound_secs - 2.0).abs() < 1e-9);
    assert!((t.ratio().unwrap() - 1.5).abs() < 1e-9);
    assert_eq!(
        t.to_string(),
        "hashed 2 files, 6000000000 bytes, in 3.000 s; t_index bound 2.000 s at W_read 3.00 GB/s, \
         R_sha 2.00 GB/s, 8 cores; 1.50× the bound"
    );
}

#[test]
fn an_empty_tag_or_no_weights_is_refused() {
    let s = Scratch::new("identity");
    let path = s.file("config.json", b"{}");
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    let cfg = LocalFile {
        path,
        role: SourceRole::Config,
        chunks: None,
    };
    let err = index
        .index("llama", &[cfg], &StepClock::new(), &quiet)
        .unwrap_err();
    assert!(
        matches!(err, LocalError::Identity(IdentityError::NoWeights)),
        "{err:?}"
    );
    assert_eq!(err.code(), None);
    let missing = weights(&s.0.join("models/absent.gguf"));
    let err = index
        .index("llama", &[missing], &StepClock::new(), &quiet)
        .unwrap_err();
    assert!(
        matches!(err, LocalError::Io { op: "resolve", .. }),
        "{err:?}"
    );
}

// ---- Chunk tables (COMP-E11-T14; RFC-0041 §4.3 step 2) ------------------------------------------

/// A weights file with a header region and two tensors: chunks as the census would give them.
fn weights_with_chunks(path: &Path, len: u64) -> LocalFile {
    LocalFile {
        path: path.to_owned(),
        role: SourceRole::Weights,
        chunks: Some(syntropy_manifest::chunk::chunk_file(len, 100, [100, len / 2 + 3]).unwrap()),
    }
}

/// Checks that file `bytes`'s table is complete: it matches `chunks`, and every record holds
/// the BLAKE3 of its chunk's bytes, computed here apart from the indexer.
fn assert_table_complete(index: &LocalIndex<'_>, bytes: &[u8], chunks: &[Chunk]) {
    let table = index.chunk_table(FileDigest::of_bytes(bytes)).unwrap();
    assert_eq!(table.file_len, bytes.len() as u64);
    assert!(table.matches(chunks));
    for record in &table.records {
        let c = record.chunk;
        let want = blake3::hash(&bytes[c.offset as usize..c.end() as usize]);
        assert_eq!(
            record.committed,
            Some((Blake3(*want.as_bytes()), EvidenceForm::Provisional)),
            "{c:?}"
        );
    }
}

#[test]
fn a_local_source_s_chunk_table_is_complete_at_registration() {
    let s = Scratch::new("tables");
    let data = bytes(50, (9 << 20) + 17);
    let cfg = br#"{"architectures":["LlamaForCausalLM"]}"#.to_vec();
    let w = weights_with_chunks(&s.file("model.safetensors", &data), data.len() as u64);
    let files = [
        w.clone(),
        LocalFile::side(s.file("config.json", &cfg), SourceRole::Config),
    ];
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    let report = index
        .index("llama", &files, &StepClock::new(), &quiet)
        .unwrap();
    assert_eq!(report.hashed.len(), 2);
    // The weights file: 1 header chunk, then each run cut every 4 MiB.
    let chunks = w.chunks.unwrap();
    assert_table_complete(&index, &data, &chunks);
    assert!(chunks.len() > 3, "{} chunks", chunks.len());
    // A side file is all header region.
    let side =
        syntropy_manifest::chunk::chunk_file(cfg.len() as u64, cfg.len() as u64, []).unwrap();
    assert_table_complete(&index, &cfg, &side);
    assert!(
        index
            .chunk_table(FileDigest::of_bytes(&cfg))
            .unwrap()
            .records[0]
            .chunk
            .header
    );
    // On disk where RFC-0041 §4.1 puts it.
    assert!(
        root.layout()
            .chunk_table(&FileDigest::of_bytes(&data).0)
            .is_file()
    );
}

/// A later start reuses a file only with its table: a missing or mismatched table means the
/// file is read again, and the table restored.
#[test]
fn a_later_start_needs_the_table_as_well_as_the_stamp() {
    let s = Scratch::new("table-reuse");
    let data = bytes(51, (5 << 20) + 9);
    let path = s.file("model.gguf", &data);
    let w = weights_with_chunks(&path, data.len() as u64);
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    index
        .index("llama", &[w.clone()], &StepClock::new(), &quiet)
        .unwrap();
    // The table is there: nothing is read.
    let again = index
        .index("llama", &[w.clone()], &StepClock::new(), &quiet)
        .unwrap();
    assert_eq!((again.hashed.len(), again.reused), (0, 1));
    // The table is gone: the file is read again, and the table comes back.
    let table = root.layout().chunk_table(&FileDigest::of_bytes(&data).0);
    std::fs::remove_file(&table).unwrap();
    let mut reopened = LocalIndex::open(&root).unwrap();
    let rebuilt = reopened
        .index("llama", &[w.clone()], &StepClock::new(), &quiet)
        .unwrap();
    assert_eq!(rebuilt.hashed.len(), 1);
    assert_table_complete(&reopened, &data, w.chunks.as_ref().unwrap());
    // The census gives other boundaries: the old table does not match, so it is rebuilt too.
    let mut moved = w.clone();
    moved.chunks =
        Some(syntropy_manifest::chunk::chunk_file(data.len() as u64, 100, [100]).unwrap());
    let rechunked = reopened
        .index("llama", &[moved.clone()], &StepClock::new(), &quiet)
        .unwrap();
    assert_eq!(rechunked.hashed.len(), 1);
    assert_table_complete(&reopened, &data, moved.chunks.as_ref().unwrap());
}

/// The racily-clean rule covers the table: a racy stamp is read again, and its table rewritten
/// from the bytes the file holds then.
#[test]
fn a_racy_stamp_rewrites_the_chunk_table() {
    let s = Scratch::new("racy-table");
    let data = bytes(52, 1 << 20);
    let path = s.file("model.gguf", &data);
    let written = mtime_ms(&path);
    let w = weights_with_chunks(&path, data.len() as u64);
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    let first = index
        .index("llama", &[w.clone()], &StepClock::at(written + 500), &quiet)
        .unwrap();
    assert_eq!(first.racy, 1);
    // A same-tick rewrite at the same size.
    let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
    let edited = bytes(53, 1 << 20);
    {
        let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all(&edited).unwrap();
        f.set_modified(mtime).unwrap();
    }
    let second = index
        .index("llama", &[w.clone()], &StepClock::new(), &quiet)
        .unwrap();
    assert_eq!(second.hashed.len(), 1);
    assert_table_complete(&index, &edited, w.chunks.as_ref().unwrap());
}

#[test]
fn chunks_that_do_not_tile_the_file_are_refused() {
    let s = Scratch::new("bad-chunks");
    let data = bytes(54, 4096);
    let path = s.file("model.gguf", &data);
    let root = s.root();
    let mut index = LocalIndex::open(&root).unwrap();
    let mut w = weights_with_chunks(&path, data.len() as u64);
    w.chunks.as_mut().unwrap().pop();
    let err = index
        .index("llama", &[w], &StepClock::new(), &quiet)
        .unwrap_err();
    assert!(matches!(err, LocalError::ChunksNotTiled { .. }), "{err:?}");
    assert_eq!(err.class(), ErrorClass::Internal);
    assert_eq!(err.code(), None);
}
