// SPDX-License-Identifier: MIT OR Apache-2.0

//! The thread-pool engines against real files (RFC-0021 §4.2, §5, Testing): `PreadPool` on
//! Linux and Windows, `MacNoCache` on macOS.
//!
//! The engine this host has runs through the engine conformance harness
//! (`syntropy_io::conformance`, TIER-E5-T33) with one worker and with several, and on Linux also
//! in the flagged buffered mode on tmpfs (TIER-E5-T03 criterion 1). The other tests check the
//! same claim directly (random aligned extents of a fixture file, read into fixed and raw
//! destinations, match the bytes read through the page cache, and writes read back), the
//! per-file direct-I/O alignment the engine reports, cancellation, refused destinations, the I/O
//! loop driving the engine, and, on Linux, that a file on tmpfs selects the flagged buffered
//! mode.

#![cfg(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "macos",
    windows
))]
#![deny(clippy::undocumented_unsafe_blocks)]

use std::path::{Path, PathBuf};

use syntropy_io::conformance::{self, Options, Report};
use syntropy_io::{
    BufIndex, Bytes, CancelOutcome, CompletionBuf, ExtentRef, FileIx, HostRegion, IoClient, IoDest,
    IoEngine, IoErrorCode, IoLoop, IoOp, IoPriority, IoTicket, LoopConfig, Nanos, OpenMode, Poll,
    ReadCompletion, ReadRequest, SubmitBatch, SyncMode, Wait, WriteRequest,
};

#[cfg(not(target_os = "macos"))]
type Engine = syntropy_io::pool::PreadPool;
#[cfg(target_os = "macos")]
type Engine = syntropy_io::pool::MacNoCache;

fn engine(threads: usize) -> Engine {
    Engine::new(threads).unwrap()
}

const BLOCK: u64 = 16384;

/// How long a test waits for completions before it fails [target]. A loaded runner may take
/// seconds; a wait ends as soon as the completions arrive, so the budget costs nothing when
/// they do.
const DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// One block, aligned for direct I/O on every OS (RFC-0021 §6: 4 KiB, 16 KiB on macOS).
#[repr(C, align(16384))]
#[derive(Clone)]
struct Block([u8; BLOCK as usize]);

/// Memory the tests own, as the tier manager owns slots. Each test makes at most one
/// destination per byte range at a time and keeps the memory alive until every completion has
/// been reaped and every engine using it is dropped.
struct Mem {
    blocks: Vec<Block>,
}

impl Mem {
    fn new(blocks: usize) -> Self {
        Self {
            blocks: vec![Block([0; BLOCK as usize]); blocks],
        }
    }

    fn base(&mut self) -> u64 {
        self.blocks.as_mut_ptr() as u64
    }

    fn len(&self) -> u64 {
        self.blocks.len() as u64 * BLOCK
    }

    fn bytes(&self, offset: u64, len: u64) -> &[u8] {
        // SAFETY: the range lies in `blocks`, and no operation is in flight when a test reads.
        let all = unsafe {
            std::slice::from_raw_parts(self.blocks.as_ptr().cast::<u8>(), self.len() as usize)
        };
        &all[offset as usize..(offset + len) as usize]
    }

    fn raw(&mut self, offset: u64, len: u64) -> IoDest {
        assert!(offset + len <= self.len());
        let addr = self.base() + offset;
        // SAFETY: the range lies in `blocks`, which outlives the request; see `Mem`.
        unsafe { IoDest::raw(addr, Bytes(len)) }
    }

    fn region(&mut self) -> HostRegion {
        HostRegion {
            addr: self.base(),
            len: Bytes(self.len()),
        }
    }
}

fn fixed(buf: BufIndex, offset: u64, len: u64) -> IoDest {
    // SAFETY: every caller passes a range of a `Mem` registered at `buf` on the one engine the
    // test uses, alive until its completions are reaped; see `Mem`.
    unsafe { IoDest::fixed(buf, offset, Bytes(len)) }
}

/// A fixture file of `len` deterministic bytes in `dir`.
fn fixture(dir: &Path, name: &str, len: usize) -> (PathBuf, Vec<u8>) {
    let mut x = 0x9e37_79b9_7f4a_7c15_u64 ^ len as u64;
    let bytes: Vec<u8> = (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect();
    let path = dir.join(format!("syntropy-io-pool-{name}-{}", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    (path, bytes)
}

/// A small deterministic generator for the random extents.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
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

/// Submits every operation and reaps until each has completed.
fn run(engine: &mut Engine, ops: Vec<IoOp>) -> Vec<ReadCompletion> {
    let want = ops.len();
    let mut batch = SubmitBatch::with_capacity(want.max(1));
    for op in ops {
        batch.push(op).unwrap();
    }
    let mut out = CompletionBuf::with_capacity(64);
    let mut done = Vec::new();
    let start = std::time::Instant::now();
    while start.elapsed() < DEADLINE {
        if !batch.is_empty() {
            engine.submit(&mut batch).unwrap();
        }
        engine.reap(&mut out, Wait::Until(Nanos(1_000_000)));
        done.extend(out.drain());
        if done.len() == want {
            assert_eq!(engine.in_flight(), (0, Bytes(0)));
            done.sort_by_key(|c| c.ticket);
            return done;
        }
    }
    panic!(
        "only {} of {want} operations completed within {DEADLINE:?}",
        done.len()
    );
}

#[test]
fn random_aligned_extents_match_the_page_cache() {
    let (path, _) = fixture(&std::env::temp_dir(), "conformance", 4 << 20);
    let mut e = engine(4);
    let file = e.open(&path, OpenMode::Read).unwrap();
    // The reference: the same file read through the page cache.
    let reference = std::fs::read(&path).unwrap();
    let mut mem = Mem::new(64);
    // SAFETY: `mem` outlives the engine, which is dropped first at the end of the test.
    let bufs = unsafe { e.register_buffers(&[mem.region()]) }.unwrap();
    let mut rng = Rng(0x5eed);
    for round in 0..8 {
        // 32 extents a round, each 1 to 4 blocks at a block-aligned offset, half into the
        // registered buffer and half into raw memory, each at its own place in `mem`.
        let mut ops = Vec::new();
        let mut expect = Vec::new();
        for i in 0..32u64 {
            let blocks = 1 + rng.below(2);
            let len = blocks * BLOCK;
            let offset = rng.below(reference.len() as u64 / BLOCK - blocks) * BLOCK;
            let at = i * 2 * BLOCK;
            let dest = if i % 2 == 0 {
                fixed(bufs[0], at, len)
            } else {
                mem.raw(at, len)
            };
            ops.push(read(round * 100 + i, file, offset, dest));
            expect.push((at, offset, len));
        }
        let done = run(&mut e, ops);
        for (c, (at, offset, len)) in done.iter().zip(&expect) {
            assert_eq!(c.result, Ok(Bytes(*len)), "ticket {}", c.ticket.0);
            assert!(
                mem.bytes(*at, *len) == &reference[*offset as usize..(*offset + *len) as usize],
                "ticket {}: bytes differ from the page cache",
                c.ticket.0
            );
        }
    }
    drop(e);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn writes_read_back() {
    let path = std::env::temp_dir().join(format!("syntropy-io-pool-write-{}", std::process::id()));
    let mut e = engine(2);
    let file = e.open(&path, OpenMode::Create).unwrap();
    let mut mem = Mem::new(8);
    for (i, b) in mem.blocks.iter_mut().enumerate() {
        b.0.fill(i as u8 + 1);
    }
    let syncs = [SyncMode::None, SyncMode::DataSync, SyncMode::FullSync];
    let ops = (0..8u64)
        .map(|i| {
            IoOp::Write(WriteRequest {
                ticket: IoTicket(i),
                extent: ExtentRef {
                    file,
                    offset: i * BLOCK,
                    len: Bytes(BLOCK),
                },
                src: mem.raw(i * BLOCK, BLOCK),
                priority: IoPriority::Background,
                sync: syncs[i as usize % 3],
            })
        })
        .collect();
    let done = run(&mut e, ops);
    assert!(
        done.iter().all(|c| c.result == Ok(Bytes(BLOCK))),
        "{done:?}"
    );
    let on_disk = std::fs::read(&path).unwrap();
    assert_eq!(on_disk.len() as u64, 8 * BLOCK);
    assert!(on_disk == mem.bytes(0, 8 * BLOCK), "written bytes differ");

    // And back through the engine, into fresh memory.
    let mut back = Mem::new(8);
    let ops = (0..8u64)
        .map(|i| read(100 + i, file, i * BLOCK, back.raw(i * BLOCK, BLOCK)))
        .collect();
    let done = run(&mut e, ops);
    assert!(done.iter().all(|c| c.result == Ok(Bytes(BLOCK))));
    assert!(back.bytes(0, 8 * BLOCK) == mem.bytes(0, 8 * BLOCK));
    drop(e);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_read_past_the_end_of_the_file_is_short() {
    let (path, bytes) = fixture(&std::env::temp_dir(), "short", (BLOCK + BLOCK / 4) as usize);
    let mut e = engine(1);
    let file = e.open(&path, OpenMode::Read).unwrap();
    let mut mem = Mem::new(2);
    let done = run(&mut e, vec![read(1, file, 0, mem.raw(0, 2 * BLOCK))]);
    // The tail is a quarter block; with direct I/O the OS may round the end of file up to its
    // block, so only "short, with the bytes it reports" is portable.
    let Ok(Bytes(n)) = done[0].result else {
        panic!("{:?}", done[0].result);
    };
    assert!((BLOCK..2 * BLOCK).contains(&n), "{n}");
    let n = n.min(bytes.len() as u64);
    assert!(mem.bytes(0, n) == &bytes[..n as usize]);
    drop(e);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_fixed_destination_outside_the_registered_buffers_is_refused_untouched() {
    let (path, _) = fixture(&std::env::temp_dir(), "refuse", BLOCK as usize);
    let mut e = engine(1);
    let file = e.open(&path, OpenMode::Read).unwrap();
    let mut mem = Mem::new(2);
    // SAFETY: `mem` outlives the engine.
    let bufs = unsafe { e.register_buffers(&[mem.region()]) }.unwrap();
    let ops = vec![
        // A buffer index nothing registered.
        read(1, file, 0, fixed(BufIndex(bufs[0].0 + 1), 0, BLOCK)),
        // Past the end of the one registered buffer.
        read(2, file, 0, fixed(bufs[0], 2 * BLOCK, BLOCK)),
        // A file index nothing opened.
        read(3, FileIx(file.0 + 1), 0, mem.raw(0, BLOCK)),
    ];
    let done = run(&mut e, ops);
    for c in &done {
        assert_eq!(
            c.result.map_err(|e| e.code),
            Err(IoErrorCode::Misaligned),
            "{c:?}"
        );
    }
    assert!(mem.bytes(0, 2 * BLOCK).iter().all(|&b| b == 0));
    drop(e);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn cancel_stops_an_unstarted_read_or_says_too_late() {
    let (path, bytes) = fixture(&std::env::temp_dir(), "cancel", 4 * BLOCK as usize);
    let mut e = engine(2);
    let file = e.open(&path, OpenMode::Read).unwrap();
    assert_eq!(e.cancel(IoTicket(7)), CancelOutcome::NotFound);
    let mut mem = Mem::new(4);
    let mut batch = SubmitBatch::with_capacity(2);
    batch.push(read(1, file, 0, mem.raw(0, BLOCK))).unwrap();
    batch
        .push(read(2, file, BLOCK, mem.raw(BLOCK, BLOCK)))
        .unwrap();
    assert_eq!(e.submit(&mut batch).unwrap(), 2);
    let outcome = e.cancel(IoTicket(2));
    assert!(
        matches!(outcome, CancelOutcome::Cancelled | CancelOutcome::TooLate),
        "{outcome:?}"
    );
    let mut out = CompletionBuf::with_capacity(4);
    let mut done = Vec::new();
    let start = std::time::Instant::now();
    while done.len() < 2 && start.elapsed() < DEADLINE {
        e.reap(&mut out, Wait::Until(Nanos(1_000_000)));
        done.extend(out.drain());
    }
    assert_eq!(
        done.len(),
        2,
        "only {} of 2 completed within {DEADLINE:?}",
        done.len()
    );
    done.sort_by_key(|c| c.ticket);
    assert_eq!(done[0].result, Ok(Bytes(BLOCK)));
    match outcome {
        CancelOutcome::Cancelled => {
            assert_eq!(
                done[1].result.map_err(|e| e.code),
                Err(IoErrorCode::Cancelled)
            );
        }
        _ => {
            assert_eq!(done[1].result, Ok(Bytes(BLOCK)));
            assert!(mem.bytes(BLOCK, BLOCK) == &bytes[BLOCK as usize..2 * BLOCK as usize]);
        }
    }
    assert_eq!(e.cancel(IoTicket(2)), CancelOutcome::NotFound);
    drop(e);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_engine_reports_its_kind_and_depth() {
    let e = engine(3);
    let caps = e.caps();
    assert_eq!(caps.max_depth, 3);
    assert!(caps.writes && caps.cancel && !caps.fixed_buffers && !caps.fixed_files);
    assert!(caps.direct, "nothing opened yet, so nothing refused");
    #[cfg(not(target_os = "macos"))]
    assert_eq!(e.kind(), syntropy_io::EngineKind::PreadPool);
    #[cfg(target_os = "macos")]
    assert_eq!(e.kind(), syntropy_io::EngineKind::MacNoCache);
}

/// Polls `io` until `want` completions have reached `client` and the loop is idle, or until
/// `deadline` passes, and returns the completions that arrived. It ends as soon as they have.
fn drive(
    io: &mut IoLoop<Engine>,
    client: &mut IoClient,
    want: usize,
    deadline: std::time::Duration,
) -> Vec<ReadCompletion> {
    let mut done = Vec::new();
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        let p = io.poll_once();
        while let Ok(c) = client.poll() {
            done.push(c);
        }
        if done.len() >= want && p == Poll::Idle {
            break;
        }
        if p != Poll::Busy {
            std::thread::yield_now();
        }
    }
    done
}

#[test]
fn the_io_loop_drives_the_engine() {
    let (path, bytes) = fixture(&std::env::temp_dir(), "loop", 1 << 20);
    let mut e = engine(4);
    let file = e.open(&path, OpenMode::Read).unwrap();
    let mut mem = Mem::new(64);
    // SAFETY: `mem` outlives the loop, which owns the engine and is dropped first.
    let bufs = unsafe { e.register_buffers(&[mem.region()]) }.unwrap();
    let config = LoopConfig {
        chunk: Bytes(4 * BLOCK),
        spin: std::time::Duration::ZERO,
        ..LoopConfig::new(1)
    };
    let (mut io, mut clients) = IoLoop::new(e, &config).unwrap();
    let client: &mut IoClient = &mut clients[0];
    // Eight requests of eight blocks: each is split into two chunks of four.
    for i in 0..8u64 {
        let offset = (i * 3 % 8) * 8 * BLOCK;
        client
            .submit(read(
                i,
                file,
                offset,
                fixed(bufs[0], i * 8 * BLOCK, 8 * BLOCK),
            ))
            .unwrap();
    }
    let done = drive(&mut io, client, 8, DEADLINE);
    assert_eq!(
        done.len(),
        8,
        "only {} of 8 completed within {DEADLINE:?}",
        done.len()
    );
    for c in &done {
        let i = c.ticket.0;
        assert_eq!(c.result, Ok(Bytes(8 * BLOCK)));
        let offset = (i * 3 % 8) * 8 * BLOCK;
        assert!(
            mem.bytes(i * 8 * BLOCK, 8 * BLOCK)
                == &bytes[offset as usize..(offset + 8 * BLOCK) as usize]
        );
    }
    drop(io);
    let _ = std::fs::remove_file(&path);
}

/// A completion that never arrives fails the wait at the deadline, not after a fixed count of
/// polls. The loop is asked for nine completions after eight reads, with a deadline of 300 ms
/// instead of the 30 s a real wait gets, and only the eight real ones come back.
#[test]
fn a_missing_completion_fails_the_wait_within_the_deadline() {
    let (path, _) = fixture(&std::env::temp_dir(), "loop-missing", 1 << 20);
    let mut e = engine(2);
    let file = e.open(&path, OpenMode::Read).unwrap();
    let mut mem = Mem::new(16);
    // SAFETY: `mem` outlives the loop, which owns the engine and is dropped first.
    let bufs = unsafe { e.register_buffers(&[mem.region()]) }.unwrap();
    let config = LoopConfig {
        spin: std::time::Duration::ZERO,
        ..LoopConfig::new(1)
    };
    let (mut io, mut clients) = IoLoop::new(e, &config).unwrap();
    let client: &mut IoClient = &mut clients[0];
    for i in 0..8u64 {
        client
            .submit(read(i, file, 0, fixed(bufs[0], i * BLOCK, BLOCK)))
            .unwrap();
    }
    let deadline = std::time::Duration::from_millis(300);
    let start = std::time::Instant::now();
    let done = drive(&mut io, client, 9, deadline);
    let took = start.elapsed();
    assert_eq!(done.len(), 8, "the eight submitted reads complete");
    assert!(took >= deadline, "gave up early: {took:?}");
    assert!(
        took < deadline + std::time::Duration::from_secs(10),
        "did not stop at the deadline: {took:?}"
    );
    drop(io);
    let _ = std::fs::remove_file(&path);
}

/// RFC-0021 §4.2 and Testing ("a tmpfs-backed run exercises buffered mode"): a file on tmpfs is
/// opened buffered, the volume is flagged, and reads still return the right bytes.
///
/// The tmpfs directory is `SYNTROPY_TMPFS_DIR`, or `/dev/shm`, which is tmpfs on glibc and musl
/// systems and in containers.
#[cfg(target_os = "linux")]
#[test]
fn a_tmpfs_backed_run_selects_buffered_mode_and_flags_the_volume() {
    let dir = std::env::var_os("SYNTROPY_TMPFS_DIR")
        .map_or_else(|| PathBuf::from("/dev/shm"), PathBuf::from);
    assert!(
        dir.is_dir(),
        "{} is missing; set SYNTROPY_TMPFS_DIR to a tmpfs mount",
        dir.display()
    );
    let (path, bytes) = fixture(&dir, "tmpfs", 8 * BLOCK as usize);
    let mut e = engine(2);
    assert!(e.caps().direct);
    let file = e.open(&path, OpenMode::Read).unwrap();
    let caps = e.caps();
    assert!(!caps.direct, "a tmpfs file must select buffered mode");
    let reason = e.buffered_reason().expect("the volume is flagged buffered");
    assert_eq!(reason.code, IoErrorCode::DirectRefused);
    // What the probe copies into the resource graph (RFC-0010 `StorageCapsExt::buffered`).
    let flagged = syntropy_core::resource::StorageCapsExt {
        buffered: !caps.direct,
        ..Default::default()
    };
    assert!(flagged.buffered);
    let mut mem = Mem::new(8);
    let ops = (0..8u64)
        .map(|i| read(i, file, (7 - i) * BLOCK, mem.raw(i * BLOCK, BLOCK)))
        .collect();
    let done = run(&mut e, ops);
    for c in &done {
        let i = c.ticket.0;
        assert_eq!(c.result, Ok(Bytes(BLOCK)));
        let off = ((7 - i) * BLOCK) as usize;
        assert!(mem.bytes(i * BLOCK, BLOCK) == &bytes[off..off + BLOCK as usize]);
    }
    eprintln!(
        "tmpfs run: {} -> direct={} reason={reason}",
        dir.display(),
        caps.direct
    );
    drop(e);
    let _ = std::fs::remove_file(&path);
}

/// Reports, without asserting, which mode the temp directory's file system gave: direct on a
/// disk file system, buffered on tmpfs or one that refuses `O_DIRECT`.
#[test]
fn report_the_mode_the_temp_directory_selects() {
    let (path, _) = fixture(&std::env::temp_dir(), "mode", BLOCK as usize);
    let mut e = engine(1);
    e.open(&path, OpenMode::Read).unwrap();
    eprintln!(
        "{}: direct={} reason={:?}",
        std::env::temp_dir().display(),
        e.caps().direct,
        e.buffered_reason()
    );
    drop(e);
    let _ = std::fs::remove_file(&path);
}

/// The conformance harness (TIER-E5-T33) over a generated 8 MiB file in `dir`, on an engine with
/// `threads` workers built on the harness's thread.
fn conformance_in(dir: &Path, name: &str, threads: usize) -> Report {
    let pid = std::process::id();
    let file = dir.join(format!("syntropy-io-pool-conf-{name}-{pid}"));
    let scratch = dir.join(format!("syntropy-io-pool-conf-{name}-{pid}.scratch"));
    conformance::write_test_file(&file, 8 << 20, 0x7003).unwrap();
    let seed = conformance::seed_from_env();
    let result = conformance::check(
        move || engine(threads),
        &file,
        &scratch,
        &Options::new(seed),
    );
    let _ = std::fs::remove_file(&file);
    let _ = std::fs::remove_file(&scratch);
    let report = result.unwrap_or_else(|failure| panic!("{failure}"));
    #[cfg(not(target_os = "macos"))]
    assert_eq!(report.engine, syntropy_io::EngineKind::PreadPool);
    #[cfg(target_os = "macos")]
    assert_eq!(report.engine, syntropy_io::EngineKind::MacNoCache);
    assert!(report.reads > 0 && report.writes > 0, "{report:?}");
    eprintln!(
        "{name}: conformance passed ({:?}, seed {:#x}, {} reads, {} writes, {}-byte unit)",
        report.engine, report.seed, report.reads, report.writes, report.unit
    );
    report
}

/// TIER-E5-T03 criterion 1, with a single worker: every operation runs in submission order.
#[test]
fn the_conformance_harness_passes_with_one_worker() {
    conformance_in(&std::env::temp_dir(), "one", 1);
}

/// TIER-E5-T03 criterion 1, with several workers: operations complete out of order.
#[test]
fn the_conformance_harness_passes_with_several_workers() {
    conformance_in(&std::env::temp_dir(), "several", 4);
}

/// TIER-E5-T03 criterion 1 in the flagged buffered mode: the harness passes on a tmpfs file,
/// which the engine opens buffered (RFC-0021 §4.2). The directory is chosen as in
/// `a_tmpfs_backed_run_selects_buffered_mode_and_flags_the_volume`.
#[cfg(target_os = "linux")]
#[test]
fn the_conformance_harness_passes_in_buffered_mode_on_tmpfs() {
    let dir = std::env::var_os("SYNTROPY_TMPFS_DIR")
        .map_or_else(|| PathBuf::from("/dev/shm"), PathBuf::from);
    assert!(
        dir.is_dir(),
        "{} is missing; set SYNTROPY_TMPFS_DIR to a tmpfs mount",
        dir.display()
    );
    conformance_in(&dir, "tmpfs", 4);
}

/// The engine reports each open file's direct-I/O alignment as `align::query` gives it
/// (RFC-0021 §6, TIER-E5-T05's `IoEngine::file_dio`), and nothing for a file it never opened.
#[test]
fn file_dio_reports_what_the_os_gives_for_each_file() {
    let (path, _) = fixture(&std::env::temp_dir(), "dio", BLOCK as usize);
    let mut e = engine(1);
    let file = e.open(&path, OpenMode::Read).unwrap();
    let want = syntropy_io::align::query(&std::fs::File::open(&path).unwrap(), &path).dio();
    assert_eq!(e.file_dio(file), Some(want));
    assert_eq!(e.file_dio(FileIx(file.0 + 1)), None);
    #[cfg(target_os = "macos")]
    assert_eq!(want.mem, 16384);
    eprintln!("{}: file_dio {want:?}", std::env::temp_dir().display());
    drop(e);
    let _ = std::fs::remove_file(&path);
}
