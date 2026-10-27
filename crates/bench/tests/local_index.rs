// SPDX-License-Identifier: MIT OR Apache-2.0

//! The local index's time against RFC-0041 §4.3's bound,
//! `t_index = max(Σ bytes / W_read, max_f bytes_f / R_sha, Σ bytes / (min(cores, files) · R_sha))`
//! (COMP-E11-T03).
//!
//! An ignored timing test, not part of any suite: a timing on a shared CI runner proves nothing.
//! Run it on a quiet machine with an optimised build, `--ignored --nocapture`, under a memory cap.
//! It writes about 1.5 GiB under Cargo's temporary directory and removes it.
//!
//! The rates are measured here, on the machine that runs it:
//! - `R_sha` is one thread's SHA-256 rate over a 256 MiB buffer in memory, the median of 3 runs.
//! - `W_read` is the rate of one sequential read of the files just written, so it is the page
//!   cache's rate, not the drive's. Dropping the page cache needs privileges this test does not
//!   take. The printed bound is then the bound for files already in memory.
//!
//! The figures go in a PR, labelled `[measured]` with the commit and rig, never in the docs.
//! `bench` reads the wall clock here because it is outside the determinism lints, which keep it
//! out of `store` (RFC-0060 §2.1 rule T1); the index itself takes the clock it is given.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use syntropy_core::{BytesPerSec, Clock, MonoTime};
use syntropy_manifest::FileDigest;
use syntropy_manifest::safetensors::side::SourceRole;
use syntropy_store::WritableRoot;
use syntropy_store::local::{IndexRates, LocalFile, LocalIndex};

const MIB: usize = 1 << 20;

/// The wall clock, as a [`Clock`] for the index.
struct WallClock(Instant);

impl Clock for WallClock {
    fn now(&self) -> MonoTime {
        MonoTime(u64::try_from(self.0.elapsed().as_nanos()).unwrap_or(u64::MAX))
    }

    fn unix_ms(&self) -> u64 {
        0
    }
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_file(path: &Path, len: usize, seed: u8) {
    let block: Vec<u8> = (0..MIB)
        .map(|i| (i as u8).wrapping_mul(131).wrapping_add(seed))
        .collect();
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        out.extend_from_slice(&block[..MIB.min(len - out.len())]);
    }
    std::fs::write(path, out).unwrap();
}

/// One thread's SHA-256 rate over a buffer in memory, the median of 3 runs.
fn r_sha() -> BytesPerSec {
    let buf: Vec<u8> = (0..256 * MIB).map(|i| (i % 251) as u8).collect();
    let mut runs: Vec<f64> = (0..3)
        .map(|_| {
            let start = Instant::now();
            std::hint::black_box(FileDigest::of_bytes(std::hint::black_box(&buf)));
            buf.len() as f64 / start.elapsed().as_secs_f64()
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    BytesPerSec(runs[1])
}

/// One sequential read of `files`, 1 MiB at a time.
fn w_read(files: &[PathBuf]) -> BytesPerSec {
    let mut buf = vec![0u8; MIB];
    let mut total = 0u64;
    let start = Instant::now();
    for path in files {
        let mut f = std::fs::File::open(path).unwrap();
        loop {
            let n = f.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            total += n as u64;
        }
    }
    BytesPerSec(total as f64 / start.elapsed().as_secs_f64())
}

fn run(tag: &str, sizes: &[usize]) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("bench-local-index-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let scratch = Scratch(dir);
    let paths: Vec<PathBuf> = sizes
        .iter()
        .enumerate()
        .map(|(i, &len)| {
            let path = scratch.0.join("models").join(format!("part-{i}.gguf"));
            write_file(&path, len, i as u8);
            path
        })
        .collect();
    let rates = IndexRates {
        w_read: w_read(&paths),
        r_sha: r_sha(),
        cores: std::thread::available_parallelism().map_or(1, usize::from),
    };
    let root = WritableRoot::open(scratch.0.join("root")).unwrap();
    let mut index = LocalIndex::open(&root).unwrap();
    let files: Vec<LocalFile> = paths
        .iter()
        .map(|p| LocalFile {
            path: p.clone(),
            role: SourceRole::Weights,
        })
        .collect();
    let clock = WallClock(Instant::now());
    let report = index.index("llama", &files, &clock, &|_| {}).unwrap();
    assert_eq!(report.hashed.len(), sizes.len());
    let t = report.against(&rates);
    println!("{tag}: {t} [measured; W_read is the page cache's rate]");
    println!("{tag}: {} hashing threads", report.threads);
    assert!(t.ratio().is_some_and(f64::is_finite));

    // A second start reads nothing.
    drop(index);
    let mut index = LocalIndex::open(&root).unwrap();
    let start = clock.now();
    let again = index.index("llama", &files, &clock, &|_| {}).unwrap();
    let secs = clock.now().saturating_duration_since(start).0 as f64 / 1e9;
    assert_eq!((again.hashed.len(), again.reused), (0, sizes.len()));
    println!(
        "{tag}: second start: 0 bytes hashed, {} files reused, {secs:.4} s [measured]",
        again.reused
    );
}

#[test]
#[ignore = "a timing: run by hand, optimised, on a quiet machine"]
fn split_files_index_against_the_t_index_bound() {
    run("split", &[256 * MIB; 4]);
}

#[test]
#[ignore = "a timing: run by hand, optimised, on a quiet machine"]
fn one_unsplit_file_is_bound_by_one_core() {
    run("unsplit", &[512 * MIB]);
}
