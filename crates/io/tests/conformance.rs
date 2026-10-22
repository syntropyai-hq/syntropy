// SPDX-License-Identifier: MIT OR Apache-2.0

//! The conformance harness (`syntropy_io::conformance`) against engines whose behaviour is known
//! (TIER-E5-T33). A correct engine built on positional `std::fs` reads and writes passes it over a
//! generated file. The same engine with one fault fails it, and the failure names the operation
//! the fault hit.

#![cfg(any(unix, windows))]
#![deny(clippy::undocumented_unsafe_blocks)]

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use syntropy_io::conformance::{self, DestKind, Op, Options, Problem};
use syntropy_io::{
    BufIndex, Bytes, CancelOutcome, CompletionBuf, DioAlign, EngineCaps, EngineKind, FileIx,
    HostRegion, IoDestKind, IoEngine, IoError, IoErrorCode, IoOp, IoTicket, Nanos, OpenMode,
    ReadCompletion, SubmitBatch, SyncMode, Wait,
};

/// One fault, hitting the `n`th operation the engine is given (counting from 0, reads and writes
/// together, in submission order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    /// Flips one byte of the read's destination.
    CorruptRead(u64),
    /// Reports half the operation's length.
    Short(u64),
    /// Completes the operation with its ticket plus this offset.
    WrongTicket(u64, u64),
    /// Changes the first byte past the read's destination.
    Overrun(u64),
    /// Flips one byte of what the write puts in the file.
    CorruptWrite(u64),
    /// Never completes the operation.
    Drop(u64),
    /// Completes the operation with `IO_SHORT_READ`.
    Error(u64),
}

/// An engine that performs each operation with positional `std::fs` calls when it is submitted
/// and completes it on the next `reap`.
struct StdEngine {
    files: Vec<File>,
    bufs: Vec<HostRegion>,
    done: Vec<ReadCompletion>,
    writes: bool,
    dio: DioAlign,
    fault: Fault,
    seen: u64,
    log: Option<Arc<Mutex<Vec<Op>>>>,
}

impl StdEngine {
    fn new() -> Self {
        StdEngine {
            files: Vec::new(),
            bufs: Vec::new(),
            done: Vec::new(),
            writes: true,
            dio: DioAlign {
                mem: 4096,
                offset: 4096,
            },
            fault: Fault::None,
            seen: 0,
            log: None,
        }
    }

    fn with_fault(fault: Fault) -> Self {
        StdEngine {
            fault,
            ..StdEngine::new()
        }
    }

    fn logging(mut self, log: &Arc<Mutex<Vec<Op>>>) -> Self {
        self.log = Some(Arc::clone(log));
        self
    }

    /// Performs `op` and returns its completion, or `None` when the fault drops it.
    fn perform(&mut self, op: &IoOp) -> Option<ReadCompletion> {
        let nth = self.seen;
        self.seen += 1;
        let extent = op.extent();
        let write = matches!(op, IoOp::Write(_));
        if let Some(log) = &self.log {
            let dest = match op.dest().kind() {
                IoDestKind::Fixed { .. } => DestKind::Fixed,
                IoDestKind::Raw { .. } => DestKind::Raw,
            };
            log.lock().unwrap().push(Op {
                ticket: op.ticket(),
                write,
                dest,
                extent,
            });
        }
        if self.fault == Fault::Drop(nth) {
            return None;
        }
        let ticket = match self.fault {
            Fault::WrongTicket(n, by) if n == nth => IoTicket(op.ticket().0 + by),
            _ => op.ticket(),
        };
        let result = match self.fault {
            Fault::Error(n) if n == nth => Err(IoError::new(IoErrorCode::ShortRead)),
            _ => self.transfer(op, nth),
        };
        let result = result.map(|n| match self.fault {
            Fault::Short(n_) if n_ == nth => Bytes(n.0 / 2),
            _ => n,
        });
        Some(ReadCompletion {
            ticket,
            result,
            latency: Nanos(0),
        })
    }

    fn transfer(&mut self, op: &IoOp, nth: u64) -> Result<Bytes, IoError> {
        let extent = op.extent();
        let len = usize::try_from(extent.len.0).unwrap();
        let addr = match op.dest().kind() {
            IoDestKind::Raw { addr } => addr,
            IoDestKind::Fixed { buf, offset } => match self.bufs.get(usize::from(buf.0)) {
                Some(region) if offset + extent.len.0 <= region.len.0 => region.addr + offset,
                _ => return Err(IoError::new(IoErrorCode::Misaligned)),
            },
        };
        // RFC-0021 §6, as the I/O thread checks it: memory and file offsets and the length.
        let (mem, off) = (u64::from(self.dio.mem), u64::from(self.dio.offset));
        if !(addr.is_multiple_of(mem)
            && extent.offset.is_multiple_of(off)
            && extent.len.0.is_multiple_of(off))
        {
            return Err(IoError::new(IoErrorCode::Misaligned));
        }
        let file = self
            .files
            .get(usize::from(extent.file.0))
            .ok_or(IoError::new(IoErrorCode::ExtentBad))?;
        match op {
            IoOp::Read(_) => {
                // SAFETY: the destination's contract (`IoDest::raw`, `IoDest::fixed`, the latter
                // checked against the registered region above) gives this engine `len` bytes at
                // `addr`, touched by nothing else until the completion is reaped.
                let dest = unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, len) };
                read_at(file, dest, extent.offset)
                    .map_err(|_| IoError::new(IoErrorCode::ShortRead))?;
                if self.fault == Fault::CorruptRead(nth) {
                    dest[len / 2] ^= 0x01;
                }
                if self.fault == Fault::Overrun(nth) {
                    // SAFETY: not an engine's right in general. The harness documents that each
                    // destination is followed by a guard unit it owns and touches only after the
                    // round's completions are reaped, so this byte is inside its allocation and
                    // nothing reads or writes it meanwhile.
                    unsafe { *(addr as *mut u8).add(len) ^= 0xFF };
                }
            }
            IoOp::Write(w) => {
                // SAFETY: as for a read: the source's contract gives this engine `len` bytes at
                // `addr`, which nothing writes until the completion is reaped.
                let src = unsafe { std::slice::from_raw_parts(addr as *const u8, len) };
                let mut bytes = src.to_vec();
                if self.fault == Fault::CorruptWrite(nth) {
                    bytes[len / 2] ^= 0x01;
                }
                write_at(file, &bytes, extent.offset)
                    .map_err(|_| IoError::new(IoErrorCode::NoSpace))?;
                match w.sync {
                    SyncMode::None => Ok(()),
                    SyncMode::DataSync => file.sync_data(),
                    SyncMode::FullSync => file.sync_all(),
                }
                .map_err(|_| IoError::new(IoErrorCode::NoSpace))?;
            }
        }
        Ok(extent.len)
    }
}

impl IoEngine for StdEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::PreadPool
    }

    fn caps(&self) -> EngineCaps {
        EngineCaps {
            fixed_buffers: false,
            fixed_files: false,
            writes: self.writes,
            cancel: false,
            max_depth: 64,
            direct: false,
            dio: self.dio,
        }
    }

    fn open(&mut self, path: &Path, mode: OpenMode) -> Result<FileIx, IoError> {
        let file = match mode {
            OpenMode::Read => File::open(path),
            OpenMode::ReadWrite => OpenOptions::new().read(true).write(true).open(path),
            OpenMode::Create => OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path),
        }
        .map_err(|_| IoError::new(IoErrorCode::ExtentBad))?;
        self.files.push(file);
        Ok(FileIx(u16::try_from(self.files.len() - 1).unwrap()))
    }

    unsafe fn register_buffers(
        &mut self,
        regions: &[HostRegion],
    ) -> Result<Vec<BufIndex>, IoError> {
        let first = self.bufs.len();
        self.bufs.extend_from_slice(regions);
        Ok((first..self.bufs.len())
            .map(|i| BufIndex(u16::try_from(i).unwrap()))
            .collect())
    }

    fn submit(&mut self, batch: &mut SubmitBatch) -> Result<u32, IoError> {
        let n = batch.len();
        for op in batch.pending() {
            if let Some(c) = self.perform(op) {
                self.done.push(c);
            }
        }
        batch.consume(n);
        Ok(u32::try_from(n).unwrap())
    }

    fn reap(&mut self, out: &mut CompletionBuf, _wait: Wait) -> u32 {
        let n = out.remaining().min(self.done.len());
        for c in self.done.drain(..n) {
            out.push(c).unwrap();
        }
        u32::try_from(n).unwrap()
    }

    fn cancel(&mut self, _ticket: IoTicket) -> CancelOutcome {
        CancelOutcome::NotFound
    }

    fn in_flight(&self) -> (u32, Bytes) {
        let bytes = self.done.iter().map(|c| c.result.map_or(0, |b| b.0)).sum();
        (u32::try_from(self.done.len()).unwrap(), Bytes(bytes))
    }
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        match file.seek_read(&mut buf[done..], offset + done as u64)? {
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => done += n,
        }
    }
    Ok(())
}

#[cfg(windows)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        done += file.seek_write(&buf[done..], offset + done as u64)?;
    }
    Ok(())
}

/// A generated file to read and a scratch path to write, both removed on drop.
struct Files {
    file: PathBuf,
    scratch: PathBuf,
}

impl Files {
    /// A generated file of `len` bytes for the test `name`.
    fn new(name: &str, len: u64) -> Self {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let files = Files {
            file: dir.join(format!("syntropy-io-conformance-{name}-{pid}")),
            scratch: dir.join(format!("syntropy-io-conformance-{name}-{pid}.scratch")),
        };
        conformance::write_test_file(&files.file, len, 0x7e57).unwrap();
        files
    }
}

impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.file);
        let _ = std::fs::remove_file(&self.scratch);
    }
}

const FILE_LEN: u64 = 4 << 20;

/// The operations of one run of `engine` with `seed`, and its result.
fn logged_run(
    engine: StdEngine,
    files: &Files,
    seed: u64,
) -> (Vec<Op>, Result<conformance::Report, conformance::Failure>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let result = conformance::check(
        engine.logging(&log),
        &files.file,
        &files.scratch,
        &Options::new(seed),
    );
    let ops = log.lock().unwrap().clone();
    (ops, result)
}

#[test]
fn a_correct_engine_passes_with_fixed_and_raw_destinations_and_writes() {
    let files = Files::new("pass", FILE_LEN);
    let seed = conformance::seed_from_env();
    let (ops, result) = logged_run(StdEngine::new(), &files, seed);
    let report = result.unwrap_or_else(|failure| panic!("{failure}"));
    // 8 rounds of 32 reads, 32 writes and their 32 read-backs, in 16 KiB units.
    assert_eq!(report.seed, seed);
    assert_eq!(report.unit, 16384);
    assert_eq!((report.reads, report.writes), (8 * 32 + 32, 32));
    assert_eq!(ops.len(), 8 * 32 + 2 * 32);
    for write in [false, true] {
        for dest in [DestKind::Fixed, DestKind::Raw] {
            assert!(
                ops.iter().any(|op| op.write == write && op.dest == dest),
                "no {} with {dest:?} memory",
                if write { "write" } else { "read" }
            );
        }
    }
    let read_bytes: u64 = ops
        .iter()
        .filter(|o| !o.write)
        .map(|o| o.extent.len.0)
        .sum();
    assert_eq!(report.bytes_read, read_bytes);
    // The file's first unit and its last whole one are read in the first round.
    assert_eq!(ops[0].extent.offset, 0);
    assert_eq!(ops[1].extent.offset + ops[1].extent.len.0, FILE_LEN);
    assert!(!files.scratch.exists(), "the scratch file is removed");
}

#[test]
fn an_engine_without_writes_is_checked_for_reads_only() {
    let files = Files::new("no-writes", FILE_LEN);
    let engine = StdEngine {
        writes: false,
        ..StdEngine::new()
    };
    let (ops, result) = logged_run(engine, &files, 3);
    let report = result.unwrap_or_else(|failure| panic!("{failure}"));
    assert_eq!((report.reads, report.writes), (8 * 32, 0));
    assert!(ops.iter().all(|op| !op.write));
    assert!(!files.scratch.exists(), "the scratch file is never created");
}

#[test]
fn the_unit_rises_to_the_engines_alignment() {
    let files = Files::new("unit", FILE_LEN);
    let engine = StdEngine {
        dio: DioAlign {
            mem: 4096,
            offset: 65536,
        },
        ..StdEngine::new()
    };
    // The engine refuses a misaligned operation, so passing means every one was aligned.
    let (ops, result) = logged_run(engine, &files, 4);
    let report = result.unwrap_or_else(|failure| panic!("{failure}"));
    assert_eq!(report.unit, 65536);
    assert!(
        ops.iter()
            .all(|op| op.extent.offset.is_multiple_of(65536)
                && op.extent.len.0.is_multiple_of(65536))
    );
}

#[test]
fn a_seed_repeats_its_operations() {
    let files = Files::new("seed", FILE_LEN);
    let (first, a) = logged_run(StdEngine::new(), &files, 0x5eed);
    let (again, b) = logged_run(StdEngine::new(), &files, 0x5eed);
    let (other, c) = logged_run(StdEngine::new(), &files, 0x5eee);
    assert!(a.is_ok() && b.is_ok() && c.is_ok());
    assert_eq!(first, again);
    assert_ne!(first, other);
}

/// Runs the engine with `fault` and returns the problem, the operation the fault hit, and the
/// failure's text.
fn failing_run(name: &str, fault: Fault, nth: u64) -> (Problem, Op, String) {
    let files = Files::new(name, FILE_LEN);
    let seed = 0xfa17;
    let (ops, result) = logged_run(StdEngine::with_fault(fault), &files, seed);
    let failure = result.expect_err("the fault is caught");
    assert_eq!(failure.seed, seed);
    let text = failure.to_string();
    assert!(text.contains("seed 0xfa17"), "{text}");
    let hit = ops[usize::try_from(nth).unwrap()];
    assert!(
        text.contains(&hit.to_string()),
        "{text} does not name {hit}"
    );
    (failure.problem, hit, text)
}

#[test]
fn a_corrupted_byte_fails_naming_the_extent() {
    let (problem, hit, _) = failing_run("corrupt", Fault::CorruptRead(5), 5);
    let Problem::Bytes { op, at, .. } = problem else {
        panic!("{problem:?}");
    };
    assert_eq!(op, hit);
    assert_eq!(at, hit.extent.offset + hit.extent.len.0 / 2);
}

#[test]
fn a_short_completion_fails_naming_the_extent() {
    let (problem, hit, _) = failing_run("short", Fault::Short(7), 7);
    assert_eq!(
        problem,
        Problem::Short {
            op: hit,
            transferred: Bytes(hit.extent.len.0 / 2)
        }
    );
}

#[test]
fn a_wrong_ticket_fails_naming_the_extent_left_without_a_completion() {
    let (problem, hit, _) = failing_run("ticket", Fault::WrongTicket(9, 1000), 9);
    assert_eq!(
        problem,
        Problem::WrongTicket {
            found: IoTicket(hit.ticket.0 + 1000),
            missing: hit
        }
    );
}

#[test]
fn a_ticket_of_another_operation_fails_as_a_second_completion() {
    // Operation 3 completes as operation 4, so whichever comes second is a duplicate: 4 itself.
    let (problem, _, _) = failing_run("ticket-reused", Fault::WrongTicket(3, 1), 4);
    assert!(matches!(problem, Problem::Duplicate(_)), "{problem:?}");
}

#[test]
fn a_write_past_the_destination_fails_naming_the_extent() {
    let (problem, hit, _) = failing_run("overrun", Fault::Overrun(2), 2);
    assert_eq!(problem, Problem::Overrun { op: hit, past: 0 });
}

#[test]
fn a_corrupted_write_fails_naming_the_extent() {
    // Operations 0..256 are the reads; 256 + 3 is the fourth write.
    let nth = 8 * 32 + 3;
    let (problem, hit, _) = failing_run("corrupt-write", Fault::CorruptWrite(nth), nth);
    assert!(hit.write);
    let Problem::Written { op, at, .. } = problem else {
        panic!("{problem:?}");
    };
    assert_eq!(op, hit);
    assert_eq!(at, hit.extent.offset + hit.extent.len.0 / 2);
}

#[test]
fn a_lost_operation_fails_naming_the_extent() {
    let (problem, hit, _) = failing_run("lost", Fault::Drop(11), 11);
    assert_eq!(problem, Problem::Lost(hit));
}

#[test]
fn an_error_completion_fails_naming_the_extent() {
    let (problem, hit, _) = failing_run("error", Fault::Error(13), 13);
    assert_eq!(
        problem,
        Problem::Failed {
            op: hit,
            error: IoError::new(IoErrorCode::ShortRead)
        }
    );
}

#[test]
fn a_file_smaller_than_one_unit_is_refused() {
    let files = Files::new("small", 1000);
    let result = conformance::check(
        StdEngine::new(),
        &files.file,
        &files.scratch,
        &Options::new(1),
    );
    assert_eq!(
        result.map_err(|f| f.problem),
        Err(Problem::FileTooSmall {
            len: 1000,
            unit: 16384
        })
    );
}

#[test]
fn a_generated_file_depends_only_on_its_seed_and_length() {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let paths: Vec<_> = (0..3)
        .map(|i| dir.join(format!("syntropy-io-conformance-gen{i}-{pid}")))
        .collect();
    for (path, seed) in paths.iter().zip([1, 1, 2]) {
        conformance::write_test_file(path, 100_000, seed).unwrap();
    }
    let read = |p: &PathBuf| std::fs::read(p).unwrap();
    assert_eq!(read(&paths[0]).len(), 100_000);
    assert_eq!(read(&paths[0]), read(&paths[1]));
    assert_ne!(read(&paths[0]), read(&paths[2]));
    for path in &paths {
        let _ = std::fs::remove_file(path);
    }
}
