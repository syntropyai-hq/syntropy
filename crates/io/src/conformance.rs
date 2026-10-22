// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engine conformance harness (RFC-0021 Testing, "Every engine returns the right bytes").
//!
//! [`check`] takes any [`IoEngine`] and a file. It reads random aligned extents of the file into
//! fixed and raw destinations and compares each one with a buffered `std::fs` read of the same
//! extent, which goes through the page cache. When the engine reports
//! [`EngineCaps::writes`](crate::EngineCaps::writes), it then writes random aligned extents to a
//! scratch file and reads them back, through `std::fs` and through the engine.
//!
//! Every engine's tests run it over a file that [`write_test_file`] generates, so an engine can be
//! checked without the fixture files. TIER-E5-T04 runs it over the nano fixture files against
//! every engine the CI host has.
//!
//! # What a run checks
//!
//! - Each operation completes exactly once, with its own ticket and its whole length.
//! - A read's destination holds the file's bytes, and a write's extent holds the source's bytes,
//!   read back both ways.
//! - No operation touches memory past its destination: each destination is followed by a guard
//!   unit the harness owns, which must keep its fill.
//! - Nothing is left in flight once every completion has been reaped.
//!
//! Extents and destinations are aligned to a unit of at least [`Options::unit`], raised to the
//! file's direct-I/O alignment ([`IoEngine::file_dio`], else
//! [`EngineCaps::dio`](crate::EngineCaps::dio)), so every request meets RFC-0021 §6.
//! Even-numbered operations use a registered (fixed) buffer, and odd-numbered ones raw memory.
//!
//! # Seeds
//!
//! A run is fixed by [`Options::seed`]: the same seed, file and unit give the same extents,
//! destinations and data. [`check`] prints the seed to standard error when it starts, and every
//! [`Failure`] names it. [`seed_from_env`] reads a seed from [`SEED_VAR`], or picks a fresh one, so
//! a failed run can be repeated.

use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use syntropy_core::Entropy;
use syntropy_core::resource::EngineKind;
use syntropy_core::{Bytes, ExtentRef, FileIx, HostRegion, Nanos};

use crate::batch::{CompletionBuf, SubmitBatch};
use crate::engine::{BufIndex, IoEngine, OpenMode, Wait};
use crate::error::{IoError, IoErrorCode};
use crate::os::OsEntropy;
use crate::request::{IoDest, IoOp, IoPriority, IoTicket, ReadRequest, SyncMode, WriteRequest};

/// The environment variable [`seed_from_env`] reads: a seed in decimal, or in hex after `0x`.
pub const SEED_VAR: &str = "SYNTROPY_IO_CONFORMANCE_SEED";

/// How much a run does, and its seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// Fixes the extents, destinations and data.
    pub seed: u64,
    /// Rounds of reads. Each round submits [`Options::extents`] reads together.
    pub rounds: u32,
    /// Reads per round, and the number of writes.
    pub extents: u32,
    /// The longest extent, in units.
    pub max_units: u32,
    /// The smallest unit, in bytes. The run raises it to the file's direct-I/O alignment and
    /// rounds it up to a power of two.
    pub unit: u64,
    /// How long one round may take before the run fails.
    pub timeout: Duration,
}

impl Options {
    /// The defaults for `seed`: 8 rounds of 32 reads, then 32 writes, each 1 to 4 units of at
    /// least 16 KiB (macOS's alignment, which covers 4 KiB elsewhere), with 60 s a round.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Options {
            seed,
            rounds: 8,
            extents: 32,
            max_units: 4,
            unit: 16 << 10,
            timeout: Duration::from_secs(60),
        }
    }
}

/// What a passing run did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Report {
    /// The run's seed.
    pub seed: u64,
    /// The engine checked.
    pub engine: EngineKind,
    /// The alignment unit the run used, in bytes.
    pub unit: u64,
    /// Reads checked against `std::fs`, the write phase's read-back included.
    pub reads: u32,
    /// Writes checked; zero when the engine does not write.
    pub writes: u32,
    /// Bytes those reads transferred.
    pub bytes_read: u64,
    /// Bytes those writes transferred.
    pub bytes_written: u64,
}

/// The memory an operation reads into or writes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DestKind {
    /// A registered buffer ([`IoDest::fixed`]).
    Fixed,
    /// Raw memory ([`IoDest::raw`]).
    Raw,
}

/// One operation the harness submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Op {
    /// Its ticket.
    pub ticket: IoTicket,
    /// A write; otherwise a read.
    pub write: bool,
    /// The memory it used.
    pub dest: DestKind,
    /// The file range it covered.
    pub extent: ExtentRef,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (what, to, mem) = match (self.write, self.dest) {
            (false, DestKind::Fixed) => ("read", "of", "into a fixed buffer"),
            (false, DestKind::Raw) => ("read", "of", "into raw memory"),
            (true, DestKind::Fixed) => ("write", "to", "from a fixed buffer"),
            (true, DestKind::Raw) => ("write", "to", "from raw memory"),
        };
        write!(
            f,
            "ticket {}: {what} {to} file {} at offset {}, {} bytes, {mem}",
            self.ticket.0, self.extent.file.0, self.extent.offset, self.extent.len.0
        )
    }
}

/// A run that failed: the seed that repeats it, the engine and the problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// The run's seed.
    pub seed: u64,
    /// The engine checked.
    pub engine: EngineKind,
    /// What went wrong.
    pub problem: Problem,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} failed I/O conformance (seed {:#x}; {SEED_VAR}={:#x} repeats the run): {}",
            self.engine, self.seed, self.seed, self.problem
        )
    }
}

impl std::error::Error for Failure {}

/// What went wrong in a failed run. Every problem with an operation names it, extent included.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Problem {
    /// The engine could not open a file.
    Open {
        /// The file.
        path: PathBuf,
        /// The engine's error.
        error: IoError,
    },
    /// The engine refused to register the harness's fixed buffer.
    Register(IoError),
    /// The file holds less than one unit, so there is nothing to read.
    FileTooSmall {
        /// The file's length.
        len: u64,
        /// The unit.
        unit: u64,
    },
    /// A `std::fs` call failed: reading the reference bytes, or a written extent.
    Reference {
        /// The operation whose bytes were being read, if any.
        op: Option<Op>,
        /// The error, as text.
        error: String,
    },
    /// [`IoEngine::submit`] returned an error.
    Submit(IoError),
    /// [`SubmitBatch::push`] refused an operation. A bug in the harness.
    Push(Op),
    /// An operation completed with an error.
    Failed {
        /// The operation.
        op: Op,
        /// Its error.
        error: IoError,
    },
    /// An operation completed with a length other than its extent's.
    Short {
        /// The operation.
        op: Op,
        /// The bytes the completion reported.
        transferred: Bytes,
    },
    /// A byte differs from what the file or the source holds.
    Bytes {
        /// The read whose destination differs.
        op: Op,
        /// The file offset of the first byte that differs.
        at: u64,
        /// What it should hold.
        expected: u8,
        /// What it holds.
        found: u8,
    },
    /// A written extent reads back through `std::fs` with a byte its source did not hold.
    Written {
        /// The write.
        op: Op,
        /// The file offset of the first byte that differs.
        at: u64,
        /// The source's byte.
        expected: u8,
        /// The byte on disk.
        found: u8,
    },
    /// An operation changed memory past the end of its destination.
    Overrun {
        /// The operation.
        op: Op,
        /// How far past the end the first changed byte is.
        past: u64,
    },
    /// A completion carried a ticket that names no operation in flight, and this operation never
    /// completed.
    WrongTicket {
        /// The ticket the completion carried.
        found: IoTicket,
        /// The operation left without a completion.
        missing: Op,
    },
    /// A completion carried a ticket that names no operation in flight, after every operation had
    /// completed.
    UnknownTicket(IoTicket),
    /// An operation completed twice.
    Duplicate(Op),
    /// An operation never completed, and the engine reports nothing in flight.
    Lost(Op),
    /// An operation had not completed when the round's time ran out.
    Timeout {
        /// The first operation still waiting.
        op: Op,
        /// The round's time.
        after: Duration,
    },
    /// The engine reports operations in flight after every completion was reaped.
    InFlight {
        /// Operations it reports.
        ops: u32,
        /// Their bytes.
        bytes: Bytes,
    },
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Problem::Open { path, error } => {
                write!(f, "the engine could not open {}: {error}", path.display())
            }
            Problem::Register(error) => {
                write!(f, "the engine refused to register a fixed buffer: {error}")
            }
            Problem::FileTooSmall { len, unit } => write!(
                f,
                "the file holds {len} bytes, less than one {unit}-byte unit"
            ),
            Problem::Reference {
                op: Some(op),
                error,
            } => {
                write!(f, "{op}: reading it through std::fs failed: {error}")
            }
            Problem::Reference { op: None, error } => write!(f, "std::fs failed: {error}"),
            Problem::Submit(error) => write!(f, "submit failed: {error}"),
            Problem::Push(op) => write!(f, "{op}: the batch refused it"),
            Problem::Failed { op, error } => write!(f, "{op}: failed with {error}"),
            Problem::Short { op, transferred } => write!(
                f,
                "{op}: completed with {} bytes, not {}",
                transferred.0, op.extent.len.0
            ),
            Problem::Bytes {
                op,
                at,
                expected,
                found,
            } => write!(
                f,
                "{op}: the byte for file offset {at} is {found:#04x}, not {expected:#04x}"
            ),
            Problem::Written {
                op,
                at,
                expected,
                found,
            } => write!(
                f,
                "{op}: file offset {at} reads back through std::fs as {found:#04x}, not \
                 {expected:#04x}"
            ),
            Problem::Overrun { op, past } => write!(
                f,
                "{op}: changed memory {past} bytes past the end of its destination"
            ),
            Problem::WrongTicket { found, missing } => write!(
                f,
                "a completion carried ticket {}, which names no operation in flight, and \
                 {missing} never completed",
                found.0
            ),
            Problem::UnknownTicket(ticket) => write!(
                f,
                "a completion carried ticket {}, which names no operation in flight",
                ticket.0
            ),
            Problem::Duplicate(op) => write!(f, "{op}: completed twice"),
            Problem::Lost(op) => write!(
                f,
                "{op}: never completed, and the engine reports nothing in flight"
            ),
            Problem::Timeout { op, after } => {
                write!(f, "{op}: not completed after {} ms", after.as_millis())
            }
            Problem::InFlight { ops, bytes } => write!(
                f,
                "the engine reports {ops} operations ({} bytes) in flight after every completion \
                 was reaped",
                bytes.0
            ),
        }
    }
}

/// Runs the conformance checks on `engine`: reads of `file`, then, when the engine writes, writes
/// to `scratch`, which it creates, replacing any file there, and removes at the end.
///
/// The engine is dropped before the run returns, so its files are closed and the memory it
/// registered is still valid while it drops (the contract of [`IoEngine::register_buffers`]).
///
/// Prints the seed to standard error when it starts.
///
/// # Errors
///
/// The first problem found, with the seed and the operation it concerns.
pub fn check<E: IoEngine>(
    engine: E,
    file: &Path,
    scratch: &Path,
    opts: &Options,
) -> Result<Report, Failure> {
    let kind = engine.kind();
    eprintln!(
        "I/O conformance: {kind:?}, seed {:#x} ({SEED_VAR}={:#x} repeats this run)",
        opts.seed, opts.seed
    );
    let fail = |problem| Failure {
        seed: opts.seed,
        engine: kind,
        problem,
    };
    let writes = engine.caps().writes;
    let result = run(engine, file, writes.then_some(scratch), opts).map_err(fail);
    if writes {
        // The engine is gone, so the file is closed; Windows removes no open file.
        let _ = std::fs::remove_file(scratch);
    }
    result.map(|counts| Report {
        seed: opts.seed,
        engine: kind,
        unit: counts.unit,
        reads: counts.reads,
        writes: counts.writes,
        bytes_read: counts.bytes_read,
        bytes_written: counts.bytes_written,
    })
}

/// The seed in [`SEED_VAR`], or a fresh one from the OS random source when it is unset.
///
/// # Panics
///
/// If [`SEED_VAR`] is set to anything but a decimal number or `0x` and a hex number: a test run
/// that names a seed must get it.
#[must_use]
pub fn seed_from_env() -> u64 {
    match std::env::var(SEED_VAR) {
        Ok(text) => parse_seed(&text)
            .unwrap_or_else(|| panic!("{SEED_VAR}={text:?} is not a decimal or 0x hex number")),
        Err(_) => {
            let mut bytes = [0u8; 8];
            OsEntropy::new().fill(&mut bytes);
            u64::from_le_bytes(bytes)
        }
    }
}

/// A seed in decimal, or in hex after `0x`.
fn parse_seed(text: &str) -> Option<u64> {
    let text = text.trim();
    match text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => text.parse().ok(),
    }
}

/// Writes `len` bytes generated from `seed` to `path`, replacing any file there: the file the
/// engine tasks run [`check`] over. The same seed and length give the same bytes.
///
/// # Errors
///
/// Any error from creating or writing the file.
pub fn write_test_file(path: &Path, len: u64, seed: u64) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    let mut rng = Rng::new(seed, Stream::File);
    let mut chunk = [0u8; 64 << 10];
    let mut left = len;
    while left > 0 {
        let n = usize::try_from(left.min(chunk.len() as u64)).unwrap_or(chunk.len());
        rng.fill(&mut chunk[..n]);
        out.write_all(&chunk[..n])?;
        left -= n as u64;
    }
    out.flush()
}

/// What a passing run counted.
struct Counts {
    unit: u64,
    reads: u32,
    writes: u32,
    bytes_read: u64,
    bytes_written: u64,
}

/// The engine and the memory it uses. Fields drop in declaration order, so the engine drops
/// before the memory it registered.
struct Rig<E> {
    engine: E,
    fixed: Mem,
    raw: Mem,
    buf: BufIndex,
    stride: u64,
}

/// One fill for memory no operation has written. Rounds alternate between two, so a read that
/// leaves a destination as the last round left it is caught.
const FILLS: [u8; 2] = [0xA5, 0x5A];

fn run<E: IoEngine>(
    mut engine: E,
    path: &Path,
    scratch: Option<&Path>,
    opts: &Options,
) -> Result<Counts, Problem> {
    let open = |engine: &mut E, path: &Path, mode| {
        engine.open(path, mode).map_err(|error| Problem::Open {
            path: path.to_owned(),
            error,
        })
    };
    let file = open(&mut engine, path, OpenMode::Read)?;
    let out = match scratch {
        Some(scratch) => Some((open(&mut engine, scratch, OpenMode::Create)?, scratch)),
        None => None,
    };
    let unit = [Some(file), out.map(|(f, _)| f)]
        .into_iter()
        .flatten()
        .map(|f| {
            let dio = engine.file_dio(f).unwrap_or(engine.caps().dio);
            u64::from(dio.mem.max(dio.offset))
        })
        .fold(opts.unit.max(1), u64::max)
        .next_power_of_two();
    let file_len = std::fs::metadata(path)
        .map_err(|e| Problem::Reference {
            op: None,
            error: e.to_string(),
        })?
        .len();
    let file_units = file_len / unit;
    if file_units == 0 {
        return Err(Problem::FileTooSmall {
            len: file_len,
            unit,
        });
    }
    let max_units = u64::from(opts.max_units.max(1)).min(file_units);
    // Each slot holds the longest extent and is followed by one guard unit. Operation `i` takes
    // slot `i / 2` of its kind of memory.
    let stride = (max_units + 1) * unit;
    let slots = u64::from(opts.extents).div_ceil(2).max(1);
    let mut rig = Rig {
        engine,
        fixed: Mem::new(slots * stride, unit),
        raw: Mem::new(slots * stride, unit),
        buf: BufIndex(0),
        stride,
    };
    let region = rig.fixed.region();
    // SAFETY: `rig.fixed` is memory this function owns, valid for reads and writes over its
    // whole length. It is a field of `rig` declared after `engine`, so it stays allocated until
    // the engine has dropped, and nothing but the engine's fixed destinations touches it while
    // operations are in flight (see `Mem`).
    let bufs = unsafe { rig.engine.register_buffers(&[region]) }.map_err(Problem::Register)?;
    rig.buf = *bufs
        .first()
        .ok_or(Problem::Register(IoError::new(IoErrorCode::RegisterFailed)))?;

    let mut counts = Counts {
        unit,
        reads: 0,
        writes: 0,
        bytes_read: 0,
        bytes_written: 0,
    };
    let mut reference = File::open(path).map_err(|e| Problem::Reference {
        op: None,
        error: e.to_string(),
    })?;
    let mut rng = Rng::new(opts.seed, Stream::Reads);
    let per_round = u64::from(opts.extents);
    let mut ticket = 1;
    for round in 0..opts.rounds {
        let ops: Vec<Op> = (0..per_round)
            .map(|i| {
                let units = 1 + rng.below(max_units);
                let at = match (round, i) {
                    // The file's first unit and its last whole one are always read.
                    (0, 0) => 0,
                    (0, 1) => file_units - units,
                    _ => rng.below(file_units - units + 1),
                };
                Op {
                    ticket: IoTicket(ticket + i),
                    write: false,
                    dest: dest_kind(i),
                    extent: ExtentRef {
                        file,
                        offset: at * unit,
                        len: Bytes(units * unit),
                    },
                }
            })
            .collect();
        ticket += per_round;
        let fill = FILLS[round as usize % 2];
        rig.fill_all(fill);
        rig.drive(&ops, opts.timeout, &[])?;
        for (i, op) in ops.iter().enumerate() {
            let want = read_reference(&mut reference, op)?;
            rig.compare(i as u64, op, &want, fill)?;
            counts.reads += 1;
            counts.bytes_read += op.extent.len.0;
        }
    }

    if let Some((out, scratch)) = out {
        write_phase(&mut rig, out, scratch, opts, ticket, max_units, &mut counts)?;
    }
    Ok(counts)
}

/// Writes random aligned extents to `out`, then reads them back through `std::fs` and through
/// the engine.
fn write_phase<E: IoEngine>(
    rig: &mut Rig<E>,
    out: FileIx,
    scratch: &Path,
    opts: &Options,
    first_ticket: u64,
    max_units: u64,
    counts: &mut Counts,
) -> Result<(), Problem> {
    let unit = counts.unit;
    let n = u64::from(opts.extents);
    let mut rng = Rng::new(opts.seed, Stream::Writes);
    // Each write gets its own place in the file, in a shuffled order, with a gap after it.
    let mut places: Vec<u64> = (0..n).collect();
    for i in (1..places.len()).rev() {
        let j = usize::try_from(rng.below(i as u64 + 1)).unwrap_or(0);
        places.swap(i, j);
    }
    let syncs = [SyncMode::None, SyncMode::DataSync, SyncMode::FullSync];
    let ops: Vec<Op> = (0..n)
        .map(|i| Op {
            ticket: IoTicket(first_ticket + i),
            write: true,
            dest: dest_kind(i),
            extent: ExtentRef {
                file: out,
                offset: places[i as usize] * (max_units + 1) * unit,
                len: Bytes((1 + rng.below(max_units)) * unit),
            },
        })
        .collect();
    let fill = FILLS[0];
    rig.fill_all(fill);
    let mut sources = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        let mut data = vec![0u8; len_usize(op)];
        rng.fill(&mut data);
        rig.slot_mut(i as u64, op.dest, data.len())
            .copy_from_slice(&data);
        sources.push(data);
    }
    rig.drive(&ops, opts.timeout, &syncs)?;
    for (i, op) in ops.iter().enumerate() {
        rig.check_guard(i as u64, op, fill)?;
        counts.writes += 1;
        counts.bytes_written += op.extent.len.0;
    }

    // Back through std::fs.
    let mut disk = File::open(scratch).map_err(|e| Problem::Reference {
        op: None,
        error: e.to_string(),
    })?;
    for (op, data) in ops.iter().zip(&sources) {
        let found = read_reference(&mut disk, op)?;
        if let Some(at) = first_difference(&found, data) {
            return Err(Problem::Written {
                op: *op,
                at: op.extent.offset + at as u64,
                expected: data[at],
                found: found[at],
            });
        }
    }

    // Back through the engine, each into the other kind of memory than it was written from.
    let back: Vec<Op> = ops
        .iter()
        .enumerate()
        .map(|(i, op)| Op {
            ticket: IoTicket(first_ticket + n + i as u64),
            write: false,
            dest: match op.dest {
                DestKind::Fixed => DestKind::Raw,
                DestKind::Raw => DestKind::Fixed,
            },
            extent: op.extent,
        })
        .collect();
    let fill = FILLS[1];
    rig.fill_all(fill);
    rig.drive(&back, opts.timeout, &[])?;
    for (i, (op, data)) in back.iter().zip(&sources).enumerate() {
        rig.compare(i as u64, op, data, fill)?;
        counts.reads += 1;
        counts.bytes_read += op.extent.len.0;
    }
    Ok(())
}

/// Even-numbered operations use the fixed buffer, odd-numbered ones raw memory.
const fn dest_kind(i: u64) -> DestKind {
    if i.is_multiple_of(2) {
        DestKind::Fixed
    } else {
        DestKind::Raw
    }
}

fn len_usize(op: &Op) -> usize {
    usize::try_from(op.extent.len.0).unwrap_or(usize::MAX)
}

/// The extent's bytes, read through `std::fs` and the page cache.
fn read_reference(file: &mut File, op: &Op) -> Result<Vec<u8>, Problem> {
    let mut bytes = vec![0u8; len_usize(op)];
    file.seek(SeekFrom::Start(op.extent.offset))
        .and_then(|_| file.read_exact(&mut bytes))
        .map_err(|e| Problem::Reference {
            op: Some(*op),
            error: e.to_string(),
        })?;
    Ok(bytes)
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

impl<E: IoEngine> Rig<E> {
    /// Where operation `i`'s slot starts in its kind of memory: slot `i / 2`. Even and odd
    /// operations use different kinds, so no two operations of a round share a slot.
    fn slot_offset(&self, i: u64) -> u64 {
        (i / 2) * self.stride
    }

    fn mem(&self, kind: DestKind) -> &Mem {
        match kind {
            DestKind::Fixed => &self.fixed,
            DestKind::Raw => &self.raw,
        }
    }

    fn slot_mut(&mut self, i: u64, kind: DestKind, len: usize) -> &mut [u8] {
        let offset = self.slot_offset(i);
        let mem = match kind {
            DestKind::Fixed => &mut self.fixed,
            DestKind::Raw => &mut self.raw,
        };
        mem.bytes_mut(offset, len as u64)
    }

    fn fill_all(&mut self, fill: u8) {
        let len = self.fixed.len;
        self.fixed.bytes_mut(0, len).fill(fill);
        let len = self.raw.len;
        self.raw.bytes_mut(0, len).fill(fill);
    }

    /// The destination for operation `i`, `op`'s length long.
    fn dest(&mut self, i: u64, op: &Op) -> IoDest {
        let offset = self.slot_offset(i);
        let len = op.extent.len;
        assert!(
            offset + self.stride <= self.mem(op.dest).len,
            "slot past the end of the memory"
        );
        match op.dest {
            // SAFETY: the range lies inside the fixed buffer, which `rig.engine` registered at
            // `self.buf` and which stays allocated until the engine drops (see `Rig`). The
            // destination never leaves this module: it is submitted only to `self.engine`, so no
            // other engine in the process sees it. Each in-flight operation has its own slot, and
            // the harness touches no slot until every completion of the round has been reaped.
            DestKind::Fixed => unsafe { IoDest::fixed(self.buf, offset, len) },
            // SAFETY: the range lies inside `self.raw`'s allocation, which outlives the engine
            // (see `Rig`); its slot is this operation's alone, and the harness touches it only
            // after every completion of the round has been reaped.
            DestKind::Raw => unsafe { IoDest::raw(self.raw.addr() + offset, len) },
        }
    }

    /// Submits every operation and reaps until each has completed once, in full. Write `i`
    /// syncs as `syncs[i % syncs.len()]`, or not at all when `syncs` is empty.
    fn drive(&mut self, ops: &[Op], timeout: Duration, syncs: &[SyncMode]) -> Result<(), Problem> {
        let Some(first) = ops.first() else {
            return Ok(());
        };
        let base = first.ticket.0;
        let mut batch = SubmitBatch::with_capacity(ops.len());
        for (i, op) in ops.iter().enumerate() {
            let dest = self.dest(i as u64, op);
            let io = if op.write {
                IoOp::Write(WriteRequest {
                    ticket: op.ticket,
                    extent: op.extent,
                    src: dest,
                    priority: IoPriority::Background,
                    sync: syncs
                        .get(i % syncs.len().max(1))
                        .copied()
                        .unwrap_or(SyncMode::None),
                })
            } else {
                IoOp::Read(ReadRequest {
                    ticket: op.ticket,
                    extent: op.extent,
                    dest,
                    priority: IoPriority::Demand,
                    useful_until: None,
                })
            };
            batch.push(io).map_err(|_| Problem::Push(*op))?;
        }
        let mut out = CompletionBuf::with_capacity(ops.len());
        let mut done = vec![false; ops.len()];
        let mut left = ops.len();
        let mut stray = None;
        let mut idle = 0;
        let start = Instant::now();
        while left > 0 {
            if !batch.is_empty() {
                self.engine.submit(&mut batch).map_err(Problem::Submit)?;
            }
            let reaped = self.engine.reap(&mut out, Wait::Until(Nanos(1_000_000)));
            for c in out.drain() {
                let index = c
                    .ticket
                    .0
                    .checked_sub(base)
                    .and_then(|i| usize::try_from(i).ok())
                    .filter(|&i| i < ops.len());
                let Some(i) = index else {
                    stray.get_or_insert(c.ticket);
                    continue;
                };
                let op = ops[i];
                if done[i] {
                    return Err(Problem::Duplicate(op));
                }
                done[i] = true;
                left -= 1;
                match c.result {
                    Err(error) => return Err(Problem::Failed { op, error }),
                    Ok(n) if n != op.extent.len => {
                        return Err(Problem::Short { op, transferred: n });
                    }
                    Ok(_) => {}
                }
            }
            if left == 0 {
                break;
            }
            let missing = || ops[done.iter().position(|d| !d).unwrap_or(0)];
            // Three quiet polls in a row with nothing in flight: what is left will never come.
            idle = if reaped == 0 && batch.is_empty() && self.engine.in_flight().0 == 0 {
                idle + 1
            } else {
                0
            };
            if idle >= 3 {
                return Err(match stray {
                    Some(found) => Problem::WrongTicket {
                        found,
                        missing: missing(),
                    },
                    None => Problem::Lost(missing()),
                });
            }
            if start.elapsed() > timeout {
                return Err(Problem::Timeout {
                    op: missing(),
                    after: timeout,
                });
            }
        }
        if let Some(ticket) = stray {
            return Err(Problem::UnknownTicket(ticket));
        }
        // A last poll: a completion past the last one expected is a ticket reused or repeated.
        self.engine.reap(&mut out, Wait::None);
        if let Some(c) = out.drain().next() {
            let index = c
                .ticket
                .0
                .checked_sub(base)
                .and_then(|i| usize::try_from(i).ok());
            return Err(match index.and_then(|i| ops.get(i)) {
                Some(op) => Problem::Duplicate(*op),
                None => Problem::UnknownTicket(c.ticket),
            });
        }
        match self.engine.in_flight() {
            (0, _) => Ok(()),
            (ops, bytes) => Err(Problem::InFlight { ops, bytes }),
        }
    }

    /// Checks read `op` in slot `i` against `want`, and the slot's guard against `fill`.
    fn compare(&mut self, i: u64, op: &Op, want: &[u8], fill: u8) -> Result<(), Problem> {
        let offset = self.slot_offset(i);
        let got = self.mem(op.dest).bytes(offset, want.len() as u64);
        if let Some(at) = first_difference(got, want) {
            return Err(Problem::Bytes {
                op: *op,
                at: op.extent.offset + at as u64,
                expected: want[at],
                found: got[at],
            });
        }
        self.check_guard(i, op, fill)
    }

    /// Checks that the memory after slot `i`'s operation, up to the next slot, kept its fill.
    fn check_guard(&self, i: u64, op: &Op, fill: u8) -> Result<(), Problem> {
        let start = self.slot_offset(i) + op.extent.len.0;
        let end = self.slot_offset(i) + self.stride;
        let guard = self.mem(op.dest).bytes(start, end - start);
        match guard.iter().position(|&b| b != fill) {
            Some(past) => Err(Problem::Overrun {
                op: *op,
                past: past as u64,
            }),
            None => Ok(()),
        }
    }
}

/// Memory the harness owns: `len` bytes aligned to `align`, inside one allocation that never
/// moves or grows.
///
/// The harness touches it, through [`Mem::bytes`] and [`Mem::bytes_mut`], only when no
/// operation is in flight. While operations are in flight, only the engine touches it, each
/// operation in its own slot.
struct Mem {
    // Never read directly: it keeps the allocation `base` points into alive.
    _buf: Vec<u8>,
    base: *mut u8,
    len: u64,
}

impl Mem {
    fn new(len: u64, align: u64) -> Self {
        let len_usize = usize::try_from(len).expect("conformance memory fits the address space");
        let align_usize = usize::try_from(align).expect("alignment fits the address space");
        let mut buf = vec![0u8; len_usize + align_usize];
        let start = buf.as_ptr().align_offset(align_usize);
        // `as_mut_ptr` materialises no reference to the elements, so `base` stays valid while the
        // engine writes through addresses derived from it.
        let base = buf.as_mut_ptr().wrapping_add(start);
        Mem {
            _buf: buf,
            base,
            len,
        }
    }

    fn addr(&self) -> u64 {
        self.base as u64
    }

    fn region(&self) -> HostRegion {
        HostRegion {
            addr: self.addr(),
            len: Bytes(self.len),
        }
    }

    fn bytes(&self, offset: u64, len: u64) -> &[u8] {
        assert!(offset + len <= self.len, "slot past the end of the memory");
        // SAFETY: `offset .. offset + len` lies inside the `len` bytes from `base`, which are in
        // `_buf`'s allocation. The harness calls this only when no operation is in flight (see
        // `Mem`), so nothing writes the range while the slice lives.
        unsafe { std::slice::from_raw_parts(self.base.add(offset as usize), len as usize) }
    }

    fn bytes_mut(&mut self, offset: u64, len: u64) -> &mut [u8] {
        assert!(offset + len <= self.len, "slot past the end of the memory");
        // SAFETY: as for `bytes`; `&mut self` keeps the harness from holding another slice of
        // this memory at the same time.
        unsafe { std::slice::from_raw_parts_mut(self.base.add(offset as usize), len as usize) }
    }
}

/// Which of a seed's independent streams a generator draws.
#[derive(Clone, Copy)]
enum Stream {
    Reads = 1,
    Writes = 2,
    File = 3,
}

/// SplitMix64: small, seedable and well mixed, so nearby seeds give unrelated runs.
struct Rng(u64);

impl Rng {
    fn new(seed: u64, stream: Stream) -> Self {
        Rng(seed ^ (stream as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n`, for `n > 0`. The modulo bias is below 2^-40 for the counts used here.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn fill(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let word = self.next().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seed_reads_in_decimal_or_hex() {
        assert_eq!(parse_seed("42"), Some(42));
        assert_eq!(parse_seed(" 0x2a "), Some(42));
        assert_eq!(parse_seed("0X2A"), Some(42));
        assert_eq!(parse_seed("0x"), None);
        assert_eq!(parse_seed("forty-two"), None);
        assert_eq!(parse_seed("-1"), None);
    }

    #[test]
    fn streams_of_one_seed_differ_and_repeat() {
        let draw = |stream| {
            let mut rng = Rng::new(7, stream);
            [rng.next(), rng.next(), rng.next()]
        };
        assert_eq!(draw(Stream::Reads), draw(Stream::Reads));
        assert_ne!(draw(Stream::Reads), draw(Stream::Writes));
        assert_ne!(draw(Stream::Writes), draw(Stream::File));
    }

    #[test]
    fn memory_is_aligned_and_sized() {
        for align in [4096, 16384, 65536] {
            let mem = Mem::new(3 * align, align);
            assert_eq!(mem.addr() % align, 0);
            assert_eq!(mem.bytes(0, 3 * align).len() as u64, 3 * align);
        }
    }

    #[test]
    fn every_problem_names_its_operation() {
        let op = Op {
            ticket: IoTicket(6),
            write: false,
            dest: DestKind::Fixed,
            extent: ExtentRef {
                file: FileIx(0),
                offset: 49152,
                len: Bytes(32768),
            },
        };
        let named = "ticket 6: read of file 0 at offset 49152, 32768 bytes, into a fixed buffer";
        assert_eq!(op.to_string(), named);
        for problem in [
            Problem::Bytes {
                op,
                at: 50000,
                expected: 1,
                found: 2,
            },
            Problem::Short {
                op,
                transferred: Bytes(16384),
            },
            Problem::WrongTicket {
                found: IoTicket(1006),
                missing: op,
            },
            Problem::Overrun { op, past: 0 },
            Problem::Lost(op),
            Problem::Duplicate(op),
        ] {
            assert!(problem.to_string().contains(named), "{problem}");
        }
        let failure = Failure {
            seed: 0x5eed,
            engine: EngineKind::PreadPool,
            problem: Problem::Lost(op),
        };
        assert!(failure.to_string().contains("seed 0x5eed"), "{failure}");
        assert!(
            failure
                .to_string()
                .contains("SYNTROPY_IO_CONFORMANCE_SEED=0x5eed"),
            "{failure}"
        );
    }
}
