// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Windows engines against real files (RFC-0021 §7 and Testing; TIER-E5-T08): IoRing, and
//! IOCP with and without helper submitters, each through the conformance harness
//! (`syntropy_io::conformance`, TIER-E5-T33), with IoRing's writes also sent through IOCP as on a
//! ring that lacks them.
//!
//! On a host without IoRing, its tests say so and check nothing. They write that, the Windows
//! build and what `QueryIoRingCapabilities` reports to standard error, uncaptured, and to the CI
//! job's summary when there is one (`GITHUB_STEP_SUMMARY`), so a hosted run shows which engines
//! it covered.

#![cfg(windows)]
#![deny(clippy::undocumented_unsafe_blocks)]

use std::io::Write as _;
use std::path::PathBuf;

use syntropy_io::conformance::{self, Options, Report};
use syntropy_io::windows::{
    IOCP_THREADS_DEFAULT, IoRingEngine, IoRingOptions, IocpEngine, IocpOptions, ioring_support,
};
use syntropy_io::{
    BufIndex, Bytes, CancelOutcome, CompletionBuf, ExtentRef, FileIx, HostRegion, IoDest, IoEngine,
    IoErrorCode, IoOp, IoPriority, IoTicket, Nanos, OpenMode, ReadCompletion, ReadRequest,
    SubmitBatch, Wait,
};

const UNIT: u64 = 16384;

/// Writes `line` to standard error and, in a CI job, to its summary. The test harness captures
/// `eprintln!`, so the line goes to the stderr handle itself, which it does not, and shows in the
/// job's log as well.
fn summary(line: &str) {
    let _ = writeln!(std::io::stderr(), "windows_engines: {line}");
    if let Some(path) = std::env::var_os("GITHUB_STEP_SUMMARY")
        && let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(path)
    {
        let _ = writeln!(file, "- {line}");
    }
}

/// A generated file and a scratch path, removed on drop.
struct Files {
    file: PathBuf,
    scratch: PathBuf,
}

impl Files {
    fn new(name: &str, len: u64) -> Self {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let files = Files {
            file: dir.join(format!("syntropy-io-win-{name}-{pid}")),
            scratch: dir.join(format!("syntropy-io-win-{name}-{pid}.scratch")),
        };
        conformance::write_test_file(&files.file, len, 0x5711).unwrap();
        files
    }
}

impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.file);
        let _ = std::fs::remove_file(&self.scratch);
    }
}

/// The conformance harness over a generated 8 MiB file, on the engine `make` builds on the
/// harness's thread.
fn conformance_run<E, F>(make: F, name: &str) -> Report
where
    E: IoEngine + 'static,
    F: FnOnce() -> E + Send + 'static,
{
    let files = Files::new(name, 8 << 20);
    let seed = conformance::seed_from_env();
    let report = conformance::check(make, &files.file, &files.scratch, &Options::new(seed))
        .unwrap_or_else(|failure| panic!("{failure}"));
    assert!(report.reads > 0 && report.writes > 0, "{report:?}");
    summary(&format!(
        "{name}: conformance passed ({:?}, seed {:#x}, {} reads, {} writes, {}-byte unit)",
        report.engine, report.seed, report.reads, report.writes, report.unit
    ));
    report
}

/// An IoRing engine, or `None` after reporting that the host has none.
fn ring(options: IoRingOptions, test: &str) -> Option<IoRingEngine> {
    match IoRingEngine::new(options) {
        Ok(engine) => Some(engine),
        Err(e) if e.code == IoErrorCode::EngineUnavailable => {
            summary(&format!(
                "{test}: IoRing unavailable ({e}); checked nothing"
            ));
            None
        }
        Err(e) => panic!("{e}"),
    }
}

fn iocp(helpers: u32) -> IocpEngine {
    IocpEngine::new(IocpOptions { depth: 64, helpers }).unwrap()
}

#[test]
fn the_host_reports_its_windows_build_and_ioring_support() {
    let ver = std::process::Command::new("cmd")
        .args(["/C", "ver"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|e| format!("`ver` failed: {e}"));
    summary(&format!("Windows: {ver}"));
    match ioring_support() {
        Ok(support) => summary(&format!(
            "QueryIoRingCapabilities and IsIoRingOpSupported: {support:?}"
        )),
        Err(e) => summary(&format!("IoRing: unavailable ({e})")),
    }
}

#[test]
fn iocp_passes_the_conformance_harness() {
    let report = conformance_run(|| iocp(0), "iocp");
    assert_eq!(report.engine, syntropy_io::EngineKind::Iocp);
}

#[test]
fn iocp_with_helper_submitters_passes_the_conformance_harness() {
    conformance_run(|| iocp(IOCP_THREADS_DEFAULT - 1), "iocp-helpers");
}

#[test]
fn ioring_passes_the_conformance_harness() {
    let test = "ioring";
    let Some(engine) = ring(IoRingOptions::default(), test) else {
        return;
    };
    let via = if engine.writes_via_iocp() {
        "IOCP"
    } else {
        "the ring"
    };
    summary(&format!("{test}: writes go through {via}"));
    // The harness builds its own engine on its thread; this one only showed the write path.
    drop(engine);
    let report = conformance_run(
        || IoRingEngine::new(IoRingOptions::default()).unwrap(),
        test,
    );
    assert_eq!(report.engine, syntropy_io::EngineKind::IoRing);
}

#[test]
fn ioring_with_writes_through_iocp_passes_the_conformance_harness() {
    let test = "ioring-iocp-writes";
    let Some(engine) = ring(
        IoRingOptions {
            writes_via_iocp: true,
        },
        test,
    ) else {
        return;
    };
    assert!(engine.writes_via_iocp());
    drop(engine);
    let options = IoRingOptions {
        writes_via_iocp: true,
    };
    conformance_run(move || IoRingEngine::new(options).unwrap(), test);
}

#[test]
fn ioring_sends_writes_to_iocp_exactly_when_the_ring_lacks_them() {
    let Some(engine) = ring(IoRingOptions::default(), "ioring-write-path") else {
        return;
    };
    let support = engine.support();
    assert_eq!(engine.writes_via_iocp(), !(support.write && support.flush));
}

/// One 16 KiB block, aligned for unbuffered I/O.
#[repr(C, align(16384))]
#[derive(Clone)]
struct Block([u8; UNIT as usize]);

/// Memory the tests own. Each test makes at most one destination per range at a time and keeps
/// the memory alive until every completion is reaped and the engine is dropped.
struct Mem {
    blocks: Vec<Block>,
}

impl Mem {
    fn new(blocks: usize, fill: u8) -> Self {
        Mem {
            blocks: vec![Block([fill; UNIT as usize]); blocks],
        }
    }

    fn base(&mut self) -> u64 {
        self.blocks.as_mut_ptr() as u64
    }

    fn bytes(&self) -> Vec<u8> {
        self.blocks.iter().flat_map(|b| b.0).collect()
    }

    fn raw(&mut self, block: u64, blocks: u64) -> IoDest {
        assert!(block + blocks <= self.blocks.len() as u64);
        let addr = self.base() + block * UNIT;
        // SAFETY: the range lies in `blocks`, which outlives the request; see `Mem`.
        unsafe { IoDest::raw(addr, Bytes(blocks * UNIT)) }
    }

    fn region(&mut self) -> HostRegion {
        HostRegion {
            addr: self.base(),
            len: Bytes(self.blocks.len() as u64 * UNIT),
        }
    }
}

fn read(ticket: u64, file: FileIx, offset: u64, dest: IoDest) -> IoOp {
    let len = dest.len();
    IoOp::Read(ReadRequest {
        ticket: IoTicket(ticket),
        extent: ExtentRef { file, offset, len },
        dest,
        priority: IoPriority::Demand,
        useful_until: None,
    })
}

/// Submits `ops`, cancels `cancel` once it is submitted, and reaps until all have completed.
fn run(
    engine: &mut dyn IoEngine,
    ops: Vec<IoOp>,
    cancel: Option<IoTicket>,
) -> (Vec<ReadCompletion>, Option<CancelOutcome>) {
    let want = ops.len();
    let mut batch = SubmitBatch::with_capacity(want.max(1));
    for op in ops {
        batch.push(op).unwrap();
    }
    let mut out = CompletionBuf::with_capacity(64);
    let mut done = Vec::new();
    let mut outcome = None;
    for _ in 0..100_000 {
        if !batch.is_empty() {
            engine.submit(&mut batch).unwrap();
        }
        if batch.is_empty() && outcome.is_none() {
            outcome = cancel.map(|ticket| engine.cancel(ticket));
        }
        engine.reap(&mut out, Wait::Until(Nanos(1_000_000)));
        done.extend(out.drain());
        if done.len() == want {
            assert_eq!(engine.in_flight(), (0, Bytes(0)));
            done.sort_by_key(|c| c.ticket);
            return (done, outcome);
        }
    }
    panic!("only {} of {want} operations completed", done.len());
}

/// Runs `test` against IOCP and, where the host has it, IoRing.
fn each_engine(name: &str, test: impl Fn(&mut dyn IoEngine)) {
    test(&mut iocp(0));
    if let Some(mut engine) = ring(IoRingOptions::default(), name) {
        test(&mut engine);
    }
}

#[test]
fn a_fixed_destination_outside_the_registered_buffers_is_refused_untouched() {
    let files = Files::new("refused", 4 * UNIT);
    each_engine("refused", |engine| {
        let file = engine.open(&files.file, OpenMode::Read).unwrap();
        let mut mem = Mem::new(4, 0xEE);
        // SAFETY: `mem` outlives every operation, which completes before `each_engine` drops
        // the engine, and nothing else touches it meanwhile.
        let bufs = unsafe { engine.register_buffers(&[mem.region()]) }.unwrap();
        // SAFETY: neither range names memory: the first is an index no engine registered, and
        // the second lies past the registered region. The engine must refuse both untouched.
        let (unknown, past) = unsafe {
            (
                IoDest::fixed(BufIndex(bufs[0].0 + 1), 0, Bytes(UNIT)),
                IoDest::fixed(bufs[0], 4 * UNIT, Bytes(UNIT)),
            )
        };
        let (done, _) = run(
            engine,
            vec![read(1, file, 0, unknown), read(2, file, 0, past)],
            None,
        );
        for c in &done {
            assert_eq!(
                c.result.map_err(|e| e.code),
                Err(IoErrorCode::Misaligned),
                "{c:?}"
            );
        }
        assert!(mem.bytes().iter().all(|&b| b == 0xEE), "memory was touched");
    });
}

#[test]
fn a_read_past_the_end_of_the_file_is_short() {
    let files = Files::new("eof", 3 * UNIT);
    let reference = std::fs::read(&files.file).unwrap();
    each_engine("eof", |engine| {
        let file = engine.open(&files.file, OpenMode::Read).unwrap();
        let mut mem = Mem::new(2, 0);
        let (done, _) = run(engine, vec![read(1, file, 2 * UNIT, mem.raw(0, 2))], None);
        assert_eq!(done[0].result, Ok(Bytes(UNIT)), "{:?}", done[0]);
        assert!(mem.bytes()[..UNIT as usize] == reference[2 * UNIT as usize..]);
    });
}

#[test]
fn a_cancelled_read_still_completes_once() {
    let files = Files::new("cancel", 64 * UNIT);
    each_engine("cancel", |engine| {
        let file = engine.open(&files.file, OpenMode::Read).unwrap();
        assert_eq!(engine.cancel(IoTicket(99)), CancelOutcome::NotFound);
        let mut mem = Mem::new(64, 0);
        let ops = (0..64)
            .map(|i| read(i, file, i * UNIT, mem.raw(i, 1)))
            .collect();
        let (done, outcome) = run(engine, ops, Some(IoTicket(63)));
        // Already in the kernel: it completes normally, or as cancelled. Or it has completed and
        // been reaped already, and is not found.
        assert!(
            matches!(
                outcome,
                Some(CancelOutcome::TooLate | CancelOutcome::NotFound)
            ),
            "{outcome:?}"
        );
        assert_eq!(done.len(), 64);
        let last = done[63].result.map_err(|e| e.code);
        assert!(
            matches!(last, Ok(Bytes(UNIT)) | Err(IoErrorCode::Cancelled)),
            "{last:?}"
        );
    });
}

#[test]
fn ioring_makes_a_registration_asked_for_in_flight_once_the_ring_is_idle() {
    let test = "ioring-late-register";
    let files = Files::new("late-register", 4 * UNIT);
    let reference = std::fs::read(&files.file).unwrap();
    // Declared before the engine, so they outlive it.
    let mut first = Mem::new(1, 0);
    let mut late = Mem::new(1, 0);
    let Some(mut engine) = ring(IoRingOptions::default(), test) else {
        return;
    };
    if !engine.support().register_buffers {
        summary(&format!(
            "{test}: the ring cannot register buffers; checked nothing"
        ));
        return;
    }
    let file = engine.open(&files.file, OpenMode::Read).unwrap();
    // SAFETY: `first` and `late` outlive the engine, which is dropped before them, and only
    // the engine's operations touch them while it lives.
    unsafe { engine.register_buffers(&[first.region()]) }.unwrap();
    assert!(engine.caps().fixed_buffers);
    // A read in flight, so the next registration has to wait.
    let mut batch = SubmitBatch::with_capacity(1);
    batch.push(read(1, file, 0, first.raw(0, 1))).unwrap();
    engine.submit(&mut batch).unwrap();
    // SAFETY: as above.
    let bufs = unsafe { engine.register_buffers(&[late.region()]) }.unwrap();
    assert!(
        !engine.caps().fixed_buffers,
        "the registration waits while the read is in flight"
    );
    // The reap that returns the read finds the ring idle, and registers.
    let mut out = CompletionBuf::with_capacity(4);
    for _ in 0..10_000 {
        engine.reap(&mut out, Wait::Until(Nanos(1_000_000)));
        if !out.is_empty() {
            break;
        }
    }
    assert_eq!(out.drain().count(), 1);
    assert!(
        engine.caps().fixed_buffers,
        "registered once the ring is idle"
    );
    // The late region now serves a fixed read.
    // SAFETY: `bufs[0]` is `late`, one block, registered on this engine only.
    let dest = unsafe { IoDest::fixed(bufs[0], 0, Bytes(UNIT)) };
    let (done, _) = run(&mut engine, vec![read(2, file, UNIT, dest)], None);
    assert_eq!(done[0].result, Ok(Bytes(UNIT)), "{:?}", done[0]);
    assert!(late.bytes() == reference[UNIT as usize..2 * UNIT as usize]);
    drop(engine);
    summary(&format!(
        "{test}: a registration asked for in flight was made once idle"
    ));
}
