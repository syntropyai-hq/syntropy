// SPDX-License-Identifier: MIT OR Apache-2.0

//! The alignment contract of RFC-0021 §6 (TIER-E5-T05).
//!
//! - Every extent the pack places passes the pre-submission check under any `DioAlign` an OS
//!   reports. The pack builder (RFC-0041) does not exist yet, so the property runs on
//!   [`RequestAlign::pack_extent`], §6's pack rule (ADR-0005 item 3) as this crate states it.
//! - A misaligned request never reaches the engine: a shim engine records every operation it is
//!   given, and every recorded operation passes the check.
//! - A row or read-around window is aligned, covers its bytes, and reads at most
//!   `len + 2A − 2` bytes.

#![deny(clippy::undocumented_unsafe_blocks)]

use std::path::Path;
use std::time::Duration;

use proptest::prelude::*;
use syntropy_io::align::RequestAlign;
use syntropy_io::{
    BufIndex, Bytes, CancelOutcome, CompletionBuf, DioAlign, EngineCaps, EngineKind, ExtentRef,
    FileIx, HostRegion, IoDest, IoEngine, IoError, IoErrorCode, IoLoop, IoOp, IoPriority, IoTicket,
    LoopConfig, OpenMode, Poll, ReadCompletion, ReadRequest, SubmitBatch, SyncMode, Wait,
    WriteRequest,
};

/// An engine that records what it is given and completes it in full, touching no memory.
struct Recorder {
    dio: DioAlign,
    /// What `open` reports for the file it opens, when set.
    opened: Option<DioAlign>,
    open_dio: Option<DioAlign>,
    log: Vec<(ExtentRef, IoDest)>,
    in_flight: Vec<(IoTicket, u64)>,
}

impl Recorder {
    fn new(dio: DioAlign) -> Self {
        Self {
            dio,
            opened: None,
            open_dio: None,
            log: Vec::new(),
            in_flight: Vec::new(),
        }
    }
}

impl IoEngine for Recorder {
    fn kind(&self) -> EngineKind {
        EngineKind::PreadPool
    }

    fn caps(&self) -> EngineCaps {
        EngineCaps {
            fixed_buffers: true,
            fixed_files: false,
            writes: true,
            cancel: false,
            max_depth: 16,
            direct: true,
            dio: self.dio,
        }
    }

    fn open(&mut self, _: &Path, _: OpenMode) -> Result<FileIx, IoError> {
        self.opened = self.open_dio;
        Ok(FileIx(0))
    }

    fn file_dio(&self, file: FileIx) -> Option<DioAlign> {
        if file == FileIx(0) { self.opened } else { None }
    }

    unsafe fn register_buffers(&mut self, _: &[HostRegion]) -> Result<Vec<BufIndex>, IoError> {
        Ok(Vec::new())
    }

    fn submit(&mut self, batch: &mut SubmitBatch) -> Result<u32, IoError> {
        let n = (16 - self.in_flight.len()).min(batch.len());
        for op in &batch.pending()[..n] {
            // Only the kind and length are recorded; the shim never touches the memory.
            let dest = match op.dest().kind() {
                syntropy_io::IoDestKind::Fixed { buf, offset } => fixed(buf.0, offset, 0),
                syntropy_io::IoDestKind::Raw { addr } => raw(addr, 0),
            };
            self.log.push((op.extent(), dest));
            self.in_flight.push((op.ticket(), op.extent().len.0));
        }
        batch.consume(n);
        Ok(n as u32)
    }

    fn reap(&mut self, out: &mut CompletionBuf, _: Wait) -> u32 {
        let mut n = 0;
        while out.remaining() > 0
            && let Some((ticket, len)) = self.in_flight.pop()
        {
            out.push(ReadCompletion {
                ticket,
                result: Ok(Bytes(len)),
                latency: syntropy_io::Nanos(1),
            })
            .unwrap();
            n += 1;
        }
        n
    }

    fn cancel(&mut self, _: IoTicket) -> CancelOutcome {
        CancelOutcome::NotFound
    }

    fn in_flight(&self) -> (u32, Bytes) {
        let bytes = self.in_flight.iter().map(|f| f.1).sum();
        (self.in_flight.len() as u32, Bytes(bytes))
    }
}

fn fixed(buf: u16, offset: u64, len: u64) -> IoDest {
    // SAFETY: only `Recorder` sees these destinations, and it never touches memory through
    // them, so no real memory is named.
    unsafe { IoDest::fixed(BufIndex(buf), offset, Bytes(len)) }
}

fn raw(addr: u64, len: u64) -> IoDest {
    // SAFETY: as for `fixed`: `Recorder` never dereferences a destination.
    unsafe { IoDest::raw(addr, Bytes(len)) }
}

fn ext(offset: u64, len: u64) -> ExtentRef {
    ExtentRef {
        file: FileIx(0),
        offset,
        len: Bytes(len),
    }
}

/// A value an OS may report for one `DioAlign` field: 0 (nothing reported) or a power of two
/// from 1 byte to 64 KiB. `statx` reports values as small as 4 for memory and 512 for offsets;
/// Windows sector sizes are 512 or 4096; macOS is 16 KiB.
fn dio_value() -> impl Strategy<Value = u32> {
    prop_oneof![Just(0u32), (0u32..=16).prop_map(|k| 1 << k)]
}

fn dio() -> impl Strategy<Value = DioAlign> {
    (dio_value(), dio_value()).prop_map(|(mem, offset)| DioAlign { mem, offset })
}

fn page() -> impl Strategy<Value = u64> {
    prop_oneof![Just(4096u64), Just(16384), Just(65536)]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Criterion 1: any extent the pack places passes the check under any `DioAlign` the OS
    /// reports, read into memory the tier manager aligned to `M`.
    #[test]
    fn every_pack_extent_passes_the_check(
        dio in dio(),
        page in page(),
        header in 0u64..1 << 20,
        units in prop::collection::vec(1u64..=8 << 20, 1..48),
        slot in 0u64..64,
    ) {
        let a = RequestAlign::from_dio(dio, page);
        let mut cursor = header;
        for (i, len) in units.iter().copied().enumerate() {
            let e = a.pack_extent(FileIx(0), cursor, len).unwrap();
            prop_assert!(e.offset >= cursor, "units never overlap");
            prop_assert!(e.len.0 >= len && e.len.0 - len < a.offset(), "padding is under one A");
            cursor = e.offset + e.len.0;
            for dest in [fixed(0, slot * a.mem(), e.len.0), raw((slot + 1) * a.mem(), e.len.0)] {
                let op = IoOp::Read(ReadRequest {
                    ticket: IoTicket(i as u64),
                    extent: e,
                    dest,
                    priority: IoPriority::Load,
                    useful_until: None,
                });
                prop_assert_eq!(a.check(&op), Ok(()), "unit {} {:?} under {:?}", i, e, dio);
            }
        }
    }

    /// Rows and read-around windows: aligned, covering, and at most `len + 2A − 2` bytes.
    #[test]
    fn windows_are_aligned_and_bounded(
        dio in dio(),
        offset in 0u64..1 << 40,
        len in 1u64..1 << 24,
    ) {
        let a = RequestAlign::from_dio(dio, 4096);
        let w = a.window(ext(offset, len)).unwrap();
        prop_assert!(a.extent_ok(&w.extent));
        prop_assert_eq!(w.extent.offset + w.skip, offset);
        prop_assert!(w.skip + len <= w.extent.len.0);
        prop_assert!(w.extent.len.0 <= len + 2 * a.offset() - 2);
    }

    /// Criterion 2: through the I/O loop, a request that fails the check completes with
    /// `IO_MISALIGNED` and the engine never sees it; every operation the engine sees passes.
    #[test]
    fn misaligned_requests_never_reach_the_engine(
        dio in dio(),
        reqs in prop::collection::vec(
            (0u64..4096, 1u64..64, 0u64..4096, any::<bool>(), any::<bool>()),
            1..40,
        ),
    ) {
        let config = LoopConfig {
            page: Bytes(4096),
            spin: Duration::ZERO,
            ..LoopConfig::new(1)
        };
        let (mut io, mut clients) = IoLoop::new(Recorder::new(dio), &config).unwrap();
        let a = io.alignment(FileIx(0));
        let mut expect_bad = 0u64;
        let mut expect_bytes = 0u64;
        let mut want = Vec::new();
        for (t, (off, blocks, dest_off, use_raw, write)) in reqs.iter().copied().enumerate() {
            // Offsets and lengths in 512-byte sectors, so some requests are aligned and some
            // are not under each reported value.
            let extent = ext(off * 512, blocks * 512);
            let len = extent.len.0;
            let dest = if use_raw { raw(dest_off * 512, len) } else { fixed(0, dest_off * 512, len) };
            let ticket = IoTicket(t as u64);
            let op = if write {
                IoOp::Write(WriteRequest {
                    ticket,
                    extent,
                    src: dest,
                    priority: IoPriority::Background,
                    sync: SyncMode::None,
                })
            } else {
                IoOp::Read(ReadRequest {
                    ticket,
                    extent,
                    dest,
                    priority: IoPriority::Demand,
                    useful_until: None,
                })
            };
            let ok = a.check(&op).is_ok();
            if ok { expect_bytes += len } else { expect_bad += 1 }
            want.push((t as u64, if ok { Ok(Bytes(len)) } else { Err(IoError::new(IoErrorCode::Misaligned)) }));
            clients[0].submit(op).unwrap();
        }
        let mut got = Vec::new();
        for _ in 0..100_000 {
            let idle = io.poll_once() == Poll::Idle;
            while let Ok(c) = clients[0].poll() {
                got.push((c.ticket.0, c.result));
            }
            if idle && got.len() == want.len() {
                break;
            }
        }
        got.sort_by_key(|g| g.0);
        prop_assert_eq!(&got, &want);
        prop_assert_eq!(io.misaligned(), expect_bad);
        let seen: u64 = io.engine().log.iter().map(|(e, _)| e.len.0).sum();
        prop_assert_eq!(seen, expect_bytes, "every aligned byte, and only those, was submitted");
        for (e, dest) in &io.engine().log {
            prop_assert!(a.extent_ok(e) && a.dest_ok(dest), "the engine saw {:?} under {:?}", e, a);
        }
    }
}

/// The same, spelled out once: under a 4 KiB volume, a 512-byte-aligned read is rejected and an
/// aligned one goes through, and the counter shows the one rejection.
#[test]
fn a_misaligned_read_is_rejected_and_counted() {
    let config = LoopConfig {
        page: Bytes(4096),
        spin: Duration::ZERO,
        ..LoopConfig::new(1)
    };
    let dio = DioAlign {
        mem: 512,
        offset: 512,
    };
    let (mut io, mut clients) = IoLoop::new(Recorder::new(dio), &config).unwrap();
    for (t, offset) in [(1u64, 512u64), (2, 8192)] {
        clients[0]
            .submit(IoOp::Read(ReadRequest {
                ticket: IoTicket(t),
                extent: ext(offset, 4096),
                dest: fixed(0, 0, 4096),
                priority: IoPriority::Demand,
                useful_until: None,
            }))
            .unwrap();
    }
    while io.poll_once() != Poll::Idle {}
    let mut got: Vec<_> = std::iter::from_fn(|| clients[0].poll().ok())
        .map(|c| (c.ticket.0, c.result))
        .collect();
    got.sort_by_key(|g| g.0);
    assert_eq!(
        got,
        [
            (1, Err(IoError::new(IoErrorCode::Misaligned))),
            (2, Ok(Bytes(4096)))
        ]
    );
    assert_eq!(io.misaligned(), 1);
    assert_eq!(io.engine().log.len(), 1);
    assert_eq!(io.engine().log[0].0, ext(8192, 4096));
}

/// The alignment a file reports at `open`, after the loop is built, is the one its requests are
/// checked against: an 8 KiB volume refuses a read at offset 4096, which the loop's 4 KiB
/// default would pass, before the engine sees it.
#[test]
fn a_files_alignment_from_open_governs_its_requests() {
    let config = LoopConfig {
        page: Bytes(4096),
        spin: Duration::ZERO,
        ..LoopConfig::new(1)
    };
    let mut engine = Recorder::new(DioAlign { mem: 0, offset: 0 });
    engine.open_dio = Some(DioAlign {
        mem: 8192,
        offset: 8192,
    });
    let (mut io, mut clients) = IoLoop::new(engine, &config).unwrap();
    assert_eq!(io.alignment(FileIx(0)).offset(), 4096, "nothing opened yet");
    let file = io
        .engine_mut()
        .open(Path::new("pack"), OpenMode::Read)
        .unwrap();
    assert_eq!(io.alignment(file).offset(), 8192);
    assert_eq!(io.alignment(file).mem(), 8192);
    for (t, offset) in [(1u64, 4096u64), (2, 16384)] {
        clients[0]
            .submit(IoOp::Read(ReadRequest {
                ticket: IoTicket(t),
                extent: ext(offset, 8192),
                dest: fixed(0, 0, 8192),
                priority: IoPriority::Demand,
                useful_until: None,
            }))
            .unwrap();
    }
    while io.poll_once() != Poll::Idle {}
    let mut got: Vec<_> = std::iter::from_fn(|| clients[0].poll().ok())
        .map(|c| (c.ticket.0, c.result))
        .collect();
    got.sort_by_key(|g| g.0);
    assert_eq!(
        got,
        [
            (1, Err(IoError::new(IoErrorCode::Misaligned))),
            (2, Ok(Bytes(8192)))
        ]
    );
    assert_eq!(io.misaligned(), 1);
    let seen: Vec<_> = io.engine().log.iter().map(|(e, _)| *e).collect();
    assert_eq!(
        seen,
        [ext(16384, 8192)],
        "the 4096-offset read never reached the engine"
    );
}
