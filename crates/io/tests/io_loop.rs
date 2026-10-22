// SPDX-License-Identifier: MIT OR Apache-2.0

//! The I/O thread loop against a fake engine (TIER-E5-T02; RFC-0021 §3.1, §3.3).
//!
//! The fake engine keeps a "file" and its "registered buffers" as its own byte vectors and
//! resolves each `Fixed` destination against them, so no test names real memory. It logs every
//! operation it accepts and can reap them newest first, shorten every transfer, or fail a submit.
//!
//! `CountingAlloc` counts each thread's allocations, so each test counts only its own thread.

#![deny(clippy::undocumented_unsafe_blocks)]

use std::path::Path;
use std::thread;
use std::time::Duration;

use syntropy_io::{
    BufIndex, Bytes, CancelOutcome, CompletionBuf, DioAlign, EngineCaps, EngineKind, ExtentRef,
    FileIx, HostRegion, IoClient, IoDest, IoDestKind, IoEngine, IoError, IoErrorCode, IoLoop, IoOp,
    IoPriority, IoTicket, LoopConfig, OpenMode, Poll, ReadCompletion, ReadRequest, SubmitBatch,
    SyncMode, Wait, WriteRequest,
};
use syntropy_testkit::guards::{CountingAlloc, alloc_count_this_thread};

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

const FILE_LEN: usize = 4 << 20;
const BUF_LEN: usize = 1 << 20;

/// An operation the fake engine accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Logged {
    extent: ExtentRef,
    dest: IoDestKind,
    dest_len: Bytes,
    priority: IoPriority,
    write: bool,
}

#[derive(Clone, Copy, Debug)]
struct InFlight {
    ticket: IoTicket,
    op: Logged,
}

struct FakeEngine {
    depth: u32,
    file: Vec<u8>,
    bufs: Vec<Vec<u8>>,
    in_flight: Vec<InFlight>,
    log: Vec<Logged>,
    /// Reap the newest operation first, so chunks complete out of order.
    reverse: bool,
    /// Shorten every transfer by this many bytes.
    short_by: u64,
    /// Fail the next submit with this error.
    fail_submit: Option<IoError>,
}

impl FakeEngine {
    fn new(depth: u32, bufs: usize) -> Self {
        Self {
            depth,
            // A byte pattern that differs at every offset a test reads.
            file: (0..FILE_LEN).map(|i| (i % 251) as u8).collect(),
            bufs: vec![vec![0; BUF_LEN]; bufs],
            in_flight: Vec::with_capacity(depth as usize),
            log: Vec::with_capacity(1 << 16),
            reverse: false,
            short_by: 0,
            fail_submit: None,
        }
    }

    fn file_range(&self, offset: u64, len: u64) -> &[u8] {
        &self.file[offset as usize..(offset + len) as usize]
    }

    fn buf_range(&self, buf: u16, offset: u64, len: u64) -> &[u8] {
        &self.bufs[usize::from(buf)][offset as usize..(offset + len) as usize]
    }
}

impl IoEngine for FakeEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::PreadPool
    }

    fn caps(&self) -> EngineCaps {
        EngineCaps {
            fixed_buffers: true,
            fixed_files: false,
            writes: true,
            cancel: false,
            max_depth: self.depth,
            direct: true,
            dio: DioAlign { mem: 0, offset: 0 },
        }
    }

    fn open(&mut self, _: &Path, _: OpenMode) -> Result<FileIx, IoError> {
        Ok(FileIx(0))
    }

    unsafe fn register_buffers(&mut self, _: &[HostRegion]) -> Result<Vec<BufIndex>, IoError> {
        Ok(Vec::new())
    }

    fn submit(&mut self, batch: &mut SubmitBatch) -> Result<u32, IoError> {
        if let Some(e) = self.fail_submit.take() {
            return Err(e);
        }
        let room = self.depth as usize - self.in_flight.len();
        let n = room.min(batch.len());
        for op in &batch.pending()[..n] {
            let logged = Logged {
                extent: op.extent(),
                dest: op.dest().kind(),
                dest_len: op.dest().len(),
                priority: op.priority(),
                write: matches!(op, IoOp::Write(_)),
            };
            self.log.push(logged);
            self.in_flight.push(InFlight {
                ticket: op.ticket(),
                op: logged,
            });
        }
        batch.consume(n);
        Ok(n as u32)
    }

    fn reap(&mut self, out: &mut CompletionBuf, _: Wait) -> u32 {
        let mut n = 0;
        while out.remaining() > 0 && !self.in_flight.is_empty() {
            let f = if self.reverse {
                self.in_flight.pop().unwrap()
            } else {
                self.in_flight.remove(0)
            };
            let IoDestKind::Fixed { buf, offset } = f.op.dest else {
                panic!("the tests use fixed destinations only");
            };
            let (start, len) = (f.op.extent.offset as usize, f.op.extent.len.0 as usize);
            let b = &mut self.bufs[usize::from(buf.0)][offset as usize..offset as usize + len];
            if f.op.write {
                self.file[start..start + len].copy_from_slice(b);
            } else {
                b.copy_from_slice(&self.file[start..start + len]);
            }
            out.push(ReadCompletion {
                ticket: f.ticket,
                result: Ok(Bytes(f.op.extent.len.0.saturating_sub(self.short_by))),
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
        let bytes = self.in_flight.iter().map(|f| f.op.extent.len.0).sum();
        (self.in_flight.len() as u32, Bytes(bytes))
    }
}

/// A read of `len` file bytes at `offset` into buffer `buf` at `buf_offset`.
fn read(ticket: u64, offset: u64, len: u64, buf: u16, buf_offset: u64, p: IoPriority) -> IoOp {
    IoOp::Read(ReadRequest {
        ticket: IoTicket(ticket),
        extent: ExtentRef {
            file: FileIx(0),
            offset,
            len: Bytes(len),
        },
        dest: fixed(buf, buf_offset, len),
        priority: p,
        useful_until: None,
    })
}

fn fixed(buf: u16, offset: u64, len: u64) -> IoDest {
    // SAFETY: only `FakeEngine` resolves these destinations, against byte vectors it owns, and
    // it checks every range with slice indexing. No real memory is named, so no range can be
    // touched by anything else.
    unsafe { IoDest::fixed(BufIndex(buf), offset, Bytes(len)) }
}

fn config(producers: usize, chunk: u64) -> LoopConfig {
    LoopConfig {
        chunk: Bytes(chunk),
        spin: Duration::ZERO,
        ..LoopConfig::new(producers)
    }
}

/// Polls until nothing is queued or in flight.
fn settle(io: &mut IoLoop<FakeEngine>) {
    for _ in 0..100_000 {
        if io.poll_once() == Poll::Idle {
            return;
        }
    }
    panic!("the loop never went idle");
}

fn drain(client: &mut IoClient) -> Vec<ReadCompletion> {
    let mut out = Vec::new();
    while let Ok(c) = client.poll() {
        out.push(c);
    }
    out
}

fn tickets_and_results(cs: &[ReadCompletion]) -> Vec<(u64, Result<Bytes, IoError>)> {
    let mut v: Vec<_> = cs.iter().map(|c| (c.ticket.0, c.result)).collect();
    v.sort_by_key(|(t, _)| *t);
    v
}

#[test]
fn completions_return_to_the_right_producer_ring() {
    let (mut io, mut clients) = IoLoop::new(FakeEngine::new(8, 3), &config(3, 1 << 20)).unwrap();
    // Every producer uses the same tickets: the loop routes by producer, not by ticket.
    for (p, client) in clients.iter_mut().enumerate() {
        for t in 0..6u64 {
            let offset = (p as u64 * 64 + t * 8) * 4096;
            client
                .submit(read(
                    t,
                    offset,
                    4096,
                    p as u16,
                    t * 8192,
                    IoPriority::Demand,
                ))
                .unwrap();
        }
    }
    settle(&mut io);
    for (p, client) in clients.iter_mut().enumerate() {
        let got = drain(client);
        assert_eq!(
            tickets_and_results(&got),
            (0..6).map(|t| (t, Ok(Bytes(4096)))).collect::<Vec<_>>(),
            "producer {p}"
        );
        for t in 0..6u64 {
            let offset = (p as u64 * 64 + t * 8) * 4096;
            assert_eq!(
                io.engine().buf_range(p as u16, t * 8192, 4096),
                io.engine().file_range(offset, 4096),
                "producer {p}, ticket {t}: the bytes landed in its own destination"
            );
        }
    }
}

#[test]
fn chunks_of_one_request_reassemble() {
    const CHUNK: u64 = 64 << 10;
    let mut engine = FakeEngine::new(16, 1);
    engine.reverse = true;
    let (mut io, mut clients) = IoLoop::new(engine, &config(1, CHUNK)).unwrap();
    let len = 3 * CHUNK + 4096;
    let offset = 40960;
    clients[0]
        .submit(read(7, offset, len, 0, 8192, IoPriority::Load))
        .unwrap();
    settle(&mut io);
    // Four chunks of at most 64 KiB, contiguous in the file and in the destination.
    let log = &io.engine().log;
    let shape: Vec<_> = log
        .iter()
        .map(|l| (l.extent.offset, l.extent.len.0, l.dest, l.dest_len.0))
        .collect();
    let want: Vec<_> = (0..4)
        .map(|i| {
            let n = if i == 3 { 4096 } else { CHUNK };
            (
                offset + i * CHUNK,
                n,
                IoDestKind::Fixed {
                    buf: BufIndex(0),
                    offset: 8192 + i * CHUNK,
                },
                n,
            )
        })
        .collect();
    assert_eq!(shape, want);
    // Reaped newest first, they still make one completion with the whole length.
    let got = drain(&mut clients[0]);
    assert_eq!(tickets_and_results(&got), [(7, Ok(Bytes(len)))]);
    assert_eq!(
        io.engine().buf_range(0, 8192, len),
        io.engine().file_range(offset, len)
    );
}

#[test]
fn adjacent_row_reads_coalesce_into_one_read() {
    let (mut io, mut clients) = IoLoop::new(FakeEngine::new(64, 2), &config(1, 1 << 20)).unwrap();
    // 20 adjacent rows, adjacent in the file and in one buffer: 16 coalesce, then 4.
    for t in 0..20u64 {
        clients[0]
            .submit(read(
                t,
                (1 << 20) + t * 4096,
                4096,
                0,
                t * 4096,
                IoPriority::Demand,
            ))
            .unwrap();
    }
    settle(&mut io);
    let lens: Vec<_> = io.engine().log.iter().map(|l| l.extent.len.0).collect();
    assert_eq!(lens, [16 * 4096, 4 * 4096]);
    let got = drain(&mut clients[0]);
    assert_eq!(
        tickets_and_results(&got),
        (0..20).map(|t| (t, Ok(Bytes(4096)))).collect::<Vec<_>>()
    );
    assert_eq!(
        io.engine().buf_range(0, 0, 20 * 4096),
        io.engine().file_range(1 << 20, 20 * 4096)
    );

    // Not coalesced: a gap in the file, a gap in the destination, another buffer, another
    // class, a destination longer than its extent, or past the chunk size.
    io.engine_mut().log.clear();
    let p = IoPriority::PrefetchNear;
    for op in [
        read(100, 0, 4096, 0, 0, p),
        read(101, 8192, 4096, 0, 4096, p),   // file gap
        read(102, 12288, 4096, 0, 12288, p), // destination gap
        read(103, 16384, 4096, 1, 16384, p), // another buffer
        read(104, 20480, 4096, 1, 20480, IoPriority::Demand), // another class
    ] {
        clients[0].submit(op).unwrap();
    }
    settle(&mut io);
    assert_eq!(io.engine().log.len(), 5);
    assert_eq!(drain(&mut clients[0]).len(), 5);

    io.engine_mut().log.clear();
    // The previous read's destination is longer than its extent, so the next file bytes would
    // not land after it.
    clients[0]
        .submit(IoOp::Read(ReadRequest {
            ticket: IoTicket(200),
            extent: ExtentRef {
                file: FileIx(0),
                offset: 0,
                len: Bytes(4096),
            },
            dest: fixed(0, 0, 8192),
            priority: p,
            useful_until: None,
        }))
        .unwrap();
    clients[0]
        .submit(read(201, 4096, 4096, 0, 8192, p))
        .unwrap();
    settle(&mut io);
    assert_eq!(io.engine().log.len(), 2);
    assert_eq!(drain(&mut clients[0]).len(), 2);
}

#[test]
fn coalescing_stops_at_the_chunk_size() {
    let (mut io, mut clients) = IoLoop::new(FakeEngine::new(64, 1), &config(1, 8192)).unwrap();
    for t in 0..4u64 {
        clients[0]
            .submit(read(t, t * 4096, 4096, 0, t * 4096, IoPriority::Demand))
            .unwrap();
    }
    settle(&mut io);
    let lens: Vec<_> = io.engine().log.iter().map(|l| l.extent.len.0).collect();
    assert_eq!(lens, [8192, 8192]);
    assert_eq!(drain(&mut clients[0]).len(), 4);
}

#[test]
fn the_highest_class_is_submitted_first() {
    let (mut io, mut clients) = IoLoop::new(FakeEngine::new(1, 1), &config(1, 1 << 20)).unwrap();
    // Posted lowest class first, and far apart so nothing coalesces.
    for (t, p) in IoPriority::ALL.iter().rev().enumerate() {
        let t = t as u64;
        clients[0]
            .submit(read(t, t * 65536, 4096, 0, t * 65536, *p))
            .unwrap();
    }
    settle(&mut io);
    let order: Vec<_> = io.engine().log.iter().map(|l| l.priority).collect();
    assert_eq!(order, IoPriority::ALL);
    assert_eq!(drain(&mut clients[0]).len(), 5);
}

#[test]
fn writes_are_submitted_whole() {
    let (mut io, mut clients) = IoLoop::new(FakeEngine::new(4, 1), &config(1, 4096)).unwrap();
    io.engine_mut().bufs[0][..16384].fill(0xAB);
    clients[0]
        .submit(IoOp::Write(WriteRequest {
            ticket: IoTicket(9),
            extent: ExtentRef {
                file: FileIx(0),
                offset: 0,
                len: Bytes(16384),
            },
            src: fixed(0, 0, 16384),
            priority: IoPriority::Background,
            sync: SyncMode::DataSync,
        }))
        .unwrap();
    settle(&mut io);
    assert_eq!(io.engine().log.len(), 1, "a write is never chunked");
    assert!(io.engine().log[0].write);
    assert_eq!(
        tickets_and_results(&drain(&mut clients[0])),
        [(9, Ok(Bytes(16384)))]
    );
    assert!(io.engine().file_range(0, 16384).iter().all(|b| *b == 0xAB));
}

#[test]
fn failed_and_short_operations_fail_their_requests() {
    let (mut io, mut clients) = IoLoop::new(FakeEngine::new(8, 1), &config(1, 1 << 20)).unwrap();
    // An extent longer than its destination never reaches the engine.
    clients[0]
        .submit(IoOp::Read(ReadRequest {
            ticket: IoTicket(1),
            extent: ExtentRef {
                file: FileIx(0),
                offset: 0,
                len: Bytes(8192),
            },
            dest: fixed(0, 0, 4096),
            priority: IoPriority::Demand,
            useful_until: None,
        }))
        .unwrap();
    settle(&mut io);
    assert!(io.engine().log.is_empty());
    assert_eq!(
        tickets_and_results(&drain(&mut clients[0])),
        [(1, Err(IoError::new(IoErrorCode::Misaligned)))]
    );

    // A short transfer of a coalesced read: the bytes go to the first row, the second is short.
    io.engine_mut().short_by = 512;
    clients[0]
        .submit(read(2, 0, 4096, 0, 0, IoPriority::Demand))
        .unwrap();
    clients[0]
        .submit(read(3, 4096, 4096, 0, 4096, IoPriority::Demand))
        .unwrap();
    settle(&mut io);
    assert_eq!(
        tickets_and_results(&drain(&mut clients[0])),
        [
            (2, Ok(Bytes(4096))),
            (3, Err(IoError::new(IoErrorCode::ShortRead)))
        ]
    );
    io.engine_mut().short_by = 0;

    // A submit error fails every operation still in the batch, and each of its requests.
    io.engine_mut().fail_submit = Some(IoError::new(IoErrorCode::NoSpace));
    clients[0]
        .submit(read(4, 65536, 4096, 0, 65536, IoPriority::Demand))
        .unwrap();
    clients[0]
        .submit(read(5, 131072, 8192, 0, 131072, IoPriority::Load))
        .unwrap();
    settle(&mut io);
    assert_eq!(
        tickets_and_results(&drain(&mut clients[0])),
        [
            (4, Err(IoError::new(IoErrorCode::NoSpace))),
            (5, Err(IoError::new(IoErrorCode::NoSpace)))
        ]
    );
}

#[test]
fn a_chunk_failing_fails_the_whole_request_once() {
    // One request in three chunks, the middle one short: one completion, an error.
    let mut engine = FakeEngine::new(1, 1);
    engine.short_by = 0;
    let (mut io, mut clients) = IoLoop::new(engine, &config(1, 4096)).unwrap();
    clients[0]
        .submit(read(1, 0, 12288, 0, 0, IoPriority::Demand))
        .unwrap();
    io.poll_once(); // the first chunk is submitted and reaped
    io.engine_mut().short_by = 100;
    io.poll_once(); // the second, short
    io.engine_mut().short_by = 0;
    settle(&mut io);
    let got = drain(&mut clients[0]);
    assert_eq!(
        tickets_and_results(&got),
        [(1, Err(IoError::new(IoErrorCode::ShortRead)))]
    );
}

#[test]
fn a_producer_that_does_not_take_its_completions_is_not_drained() {
    let cfg = LoopConfig {
        completion_ring: 2,
        ..config(1, 1 << 20)
    };
    let (mut io, mut clients) = IoLoop::new(FakeEngine::new(8, 1), &cfg).unwrap();
    for t in 0..4u64 {
        clients[0]
            .submit(read(t, t * 65536, 4096, 0, t * 65536, IoPriority::Demand))
            .unwrap();
    }
    settle(&mut io);
    // Only two fit its completion ring, so only two were taken.
    assert_eq!(io.engine().log.len(), 2);
    assert_eq!(drain(&mut clients[0]).len(), 2);
    settle(&mut io);
    assert_eq!(drain(&mut clients[0]).len(), 2);
}

/// One round of steady-state traffic: 16 coalescing rows and one chunked read.
fn round(client: &mut IoClient, io: &mut IoLoop<FakeEngine>, out: &mut Vec<ReadCompletion>) {
    for t in 0..16u64 {
        client
            .submit(read(t, t * 4096, 4096, 0, t * 4096, IoPriority::Demand))
            .unwrap();
    }
    client
        .submit(read(
            16,
            1 << 20,
            256 << 10,
            0,
            512 << 10,
            IoPriority::PrefetchNear,
        ))
        .unwrap();
    settle(io);
    while let Ok(c) = client.poll() {
        out.push(c);
    }
}

#[test]
fn the_loop_allocates_nothing_in_steady_state() {
    let (mut io, mut clients) = IoLoop::new(FakeEngine::new(16, 1), &config(1, 64 << 10)).unwrap();
    let mut out = Vec::with_capacity(64);
    round(&mut clients[0], &mut io, &mut out);
    assert_eq!(out.len(), 17);
    let before = alloc_count_this_thread();
    for _ in 0..1_000 {
        out.clear();
        io.engine_mut().log.clear();
        round(&mut clients[0], &mut io, &mut out);
        assert_eq!(out.len(), 17);
    }
    assert_eq!(alloc_count_this_thread() - before, 0);
}

/// The loop on its own thread: it parks when idle, the doorbells wake it, it allocates nothing
/// while it runs, and it returns once every producer has gone.
#[test]
fn a_parked_loop_wakes_on_the_doorbell_and_allocates_nothing() {
    const ROUNDS: u64 = 500;
    let (io, clients) = IoLoop::new(FakeEngine::new(8, 2), &config(2, 64 << 10)).unwrap();
    let looper = thread::spawn(move || {
        let mut io = io;
        // The thread's handle exists before the count starts.
        drop(thread::current());
        let before = alloc_count_this_thread();
        io.run();
        (alloc_count_this_thread() - before, io)
    });
    let producers: Vec<_> = clients
        .into_iter()
        .map(|mut client| {
            thread::spawn(move || {
                let p = client.index() as u16;
                for t in 0..ROUNDS {
                    let offset = (u64::from(p) * 512 + t % 64) * 4096;
                    client
                        .submit(read(
                            t,
                            offset,
                            4096,
                            p,
                            (t % 64) * 4096,
                            IoPriority::Demand,
                        ))
                        .unwrap();
                    let c = client
                        .completions()
                        .pop_wait(Duration::ZERO)
                        .expect("the loop answers");
                    assert_eq!((c.ticket, c.result), (IoTicket(t), Ok(Bytes(4096))));
                    // Now and then the loop runs dry and parks.
                    if t % 16 == 0 {
                        thread::sleep(Duration::from_micros(200));
                    }
                }
            })
        })
        .collect();
    for p in producers {
        p.join().unwrap();
    }
    let (allocations, io) = looper.join().unwrap();
    assert_eq!(allocations, 0, "the loop allocated while running");
    assert!(
        io.parks() > 0,
        "the loop never parked, so the doorbells went untested"
    );
    assert_eq!(io.engine().log.len(), 2 * ROUNDS as usize);
}
