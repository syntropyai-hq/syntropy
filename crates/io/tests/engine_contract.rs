// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `IoEngine` contract seen from outside the crate, as `FakeIo` and the engines will see it,
//! and the no-allocation property of `SubmitBatch` and `CompletionBuf` (RFC-0021 §2, goal 6).
//!
//! A counting allocator wraps the system allocator for this test binary. It counts per thread,
//! so tests running in parallel do not see each other's allocations.

#![deny(clippy::undocumented_unsafe_blocks)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::Path;

use syntropy_io::{
    BufIndex, Bytes, CancelOutcome, CompletionBuf, DioAlign, EngineCaps, EngineKind, ExtentRef,
    FileIx, HostRegion, IoDest, IoEngine, IoError, IoErrorCode, IoOp, IoPriority, IoTicket, Nanos,
    OpenMode, ReadCompletion, ReadRequest, SubmitBatch, SyncMode, Wait, WriteRequest,
};

struct CountingAlloc;

thread_local! {
    // A const initialiser and a type without `Drop` keep this from allocating on first use.
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

// SAFETY: every method forwards to `System` with the caller's arguments unchanged, so `System`'s
// guarantees hold. Counting touches only a thread-local `Cell<u64>`, which neither allocates nor
// unwinds.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, which `System` shares.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: `ptr` came from this allocator, which got it from `System`; the caller upholds
        // the rest of `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator, which got it from `System`, with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Adds one to this thread's count. `try_with`, because a panic inside the allocator aborts.
fn count() {
    let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocations_on_this_thread() -> u64 {
    ALLOCATIONS.with(Cell::get)
}

/// An engine that completes every operation in full on the next `reap`, touching no memory.
/// Its in-flight list is allocated once, at construction, like a real engine's ring.
struct ImmediateEngine {
    in_flight: Vec<IoOp>,
    depth: usize,
}

impl ImmediateEngine {
    fn new(depth: usize) -> Self {
        Self {
            in_flight: Vec::with_capacity(depth),
            depth,
        }
    }
}

impl IoEngine for ImmediateEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::PreadPool
    }

    fn caps(&self) -> EngineCaps {
        EngineCaps {
            fixed_buffers: false,
            fixed_files: false,
            writes: true,
            cancel: true,
            max_depth: u32::try_from(self.depth).unwrap(),
            direct: true,
            dio: DioAlign {
                mem: 4096,
                offset: 4096,
            },
        }
    }

    fn open(&mut self, _path: &Path, _mode: OpenMode) -> Result<FileIx, IoError> {
        Ok(FileIx(0))
    }

    fn register_buffers(&mut self, regions: &[HostRegion]) -> Result<Vec<BufIndex>, IoError> {
        (0..regions.len())
            .map(|i| {
                u16::try_from(i)
                    .map(BufIndex)
                    .map_err(|_| IoError::new(IoErrorCode::RegisterFailed))
            })
            .collect()
    }

    fn submit(&mut self, batch: &mut SubmitBatch) -> Result<u32, IoError> {
        let room = self.depth - self.in_flight.len();
        let n = room.min(batch.len());
        self.in_flight.extend_from_slice(&batch.pending()[..n]);
        batch.consume(n);
        Ok(u32::try_from(n).unwrap())
    }

    fn reap(&mut self, out: &mut CompletionBuf, _wait: Wait) -> u32 {
        let n = out.remaining().min(self.in_flight.len());
        for op in self.in_flight.drain(..n) {
            let c = ReadCompletion {
                ticket: op.ticket(),
                result: Ok(Bytes(op.extent().len)),
                latency: Nanos(0),
            };
            out.push(c)
                .expect("reap appends at most `remaining` completions");
        }
        u32::try_from(n).unwrap()
    }

    fn cancel(&mut self, ticket: IoTicket) -> CancelOutcome {
        match self.in_flight.iter().position(|op| op.ticket() == ticket) {
            Some(i) => {
                self.in_flight.remove(i);
                CancelOutcome::Cancelled
            }
            None => CancelOutcome::NotFound,
        }
    }

    fn in_flight(&self) -> (u32, Bytes) {
        let bytes = self.in_flight.iter().map(|op| op.extent().len).sum();
        (u32::try_from(self.in_flight.len()).unwrap(), Bytes(bytes))
    }
}

fn read(ticket: u64, priority: IoPriority) -> IoOp {
    IoOp::Read(ReadRequest {
        ticket: IoTicket(ticket),
        extent: ExtentRef {
            file: 0,
            offset: ticket * 65536,
            len: 65536,
        },
        dest: IoDest::Raw {
            addr: 0x10_0000 + ticket * 65536,
        },
        priority,
        useful_until: None,
    })
}

fn write(ticket: u64) -> IoOp {
    IoOp::Write(WriteRequest {
        ticket: IoTicket(ticket),
        extent: ExtentRef {
            file: 1,
            offset: ticket * 4096,
            len: 4096,
        },
        src: IoDest::Fixed {
            buf: BufIndex(0),
            offset: 0,
        },
        priority: IoPriority::Background,
        sync: SyncMode::DataSync,
    })
}

#[test]
fn submit_accepts_a_prefix_and_leaves_the_rest() {
    let mut engine = ImmediateEngine::new(2);
    let mut batch = SubmitBatch::with_capacity(8);
    for t in 0..5 {
        batch.push(read(t, IoPriority::Demand)).unwrap();
    }

    assert_eq!(engine.submit(&mut batch), Ok(2));
    let left: Vec<_> = batch.pending().iter().map(IoOp::ticket).collect();
    assert_eq!(left, [IoTicket(2), IoTicket(3), IoTicket(4)]);
    assert_eq!(engine.in_flight(), (2, Bytes(2 * 65536)));

    // Depth is full: nothing more is accepted until a completion is reaped.
    assert_eq!(engine.submit(&mut batch), Ok(0));
    assert_eq!(batch.len(), 3);
}

#[test]
fn reap_appends_no_more_than_the_buffer_holds() {
    let mut engine = ImmediateEngine::new(4);
    let mut batch = SubmitBatch::with_capacity(4);
    for t in 0..4 {
        batch.push(read(t, IoPriority::PrefetchNear)).unwrap();
    }
    assert_eq!(engine.submit(&mut batch), Ok(4));

    let mut out = CompletionBuf::with_capacity(3);
    assert_eq!(engine.reap(&mut out, Wait::None), 3);
    assert!(out.is_full());
    let tickets: Vec<_> = out.drain().map(|c| c.ticket).collect();
    assert_eq!(tickets, [IoTicket(0), IoTicket(1), IoTicket(2)]);

    // The completion that did not fit stays with the engine for the next call.
    assert_eq!(engine.reap(&mut out, Wait::Until(Nanos(100_000))), 1);
    assert_eq!(out.as_slice()[0].ticket, IoTicket(3));
    assert_eq!(out.as_slice()[0].result, Ok(Bytes(65536)));
    assert_eq!(engine.in_flight(), (0, Bytes(0)));
}

#[test]
fn cancel_reports_its_outcome() {
    let mut engine = ImmediateEngine::new(4);
    let mut batch = SubmitBatch::with_capacity(4);
    batch.push(read(7, IoPriority::PrefetchFar)).unwrap();
    engine.submit(&mut batch).unwrap();
    assert_eq!(engine.cancel(IoTicket(7)), CancelOutcome::Cancelled);
    assert_eq!(engine.cancel(IoTicket(7)), CancelOutcome::NotFound);
}

#[test]
fn an_engine_moves_onto_its_thread_as_a_trait_object() {
    let mut engine: Box<dyn IoEngine> = Box::new(ImmediateEngine::new(2));
    let regions = [
        HostRegion {
            addr: 0x20_0000,
            len: Bytes(1 << 30),
        },
        HostRegion {
            addr: 0x4020_0000,
            len: Bytes(1 << 30),
        },
    ];
    assert_eq!(
        engine.register_buffers(&regions),
        Ok(vec![BufIndex(0), BufIndex(1)])
    );
    assert_eq!(
        engine.open(Path::new("model.pack"), OpenMode::Read),
        Ok(FileIx(0))
    );
    assert_eq!(engine.kind(), EngineKind::PreadPool);

    let handle = std::thread::spawn(move || {
        let mut batch = SubmitBatch::with_capacity(1);
        batch.push(write(1)).unwrap();
        engine.submit(&mut batch).unwrap();
        let mut out = CompletionBuf::with_capacity(1);
        engine.reap(&mut out, Wait::None);
        out.as_slice()[0]
    });
    let c = handle.join().unwrap();
    assert_eq!(c.ticket, IoTicket(1));
    assert_eq!(c.result, Ok(Bytes(4096)));
}

/// The I/O loop's steady state: push chunks, submit, reap, drain, including pushes into a full
/// batch and reaps into a full buffer. After the buffers and the engine are built at `Warming`,
/// none of it allocates.
#[test]
fn steady_state_submit_and_reap_allocate_nothing() {
    let mut engine = ImmediateEngine::new(16);
    let mut batch = SubmitBatch::with_capacity(32);
    let mut out = CompletionBuf::with_capacity(8);
    let mut reaped = 0u64;
    let mut rejected = 0u64;

    let before = allocations_on_this_thread();
    for round in 0..1_000u64 {
        for i in 0..40 {
            let t = round * 40 + i;
            let op = if i % 10 == 9 {
                write(t)
            } else {
                read(t, IoPriority::ALL[(i % 5) as usize])
            };
            if batch.push(op).is_err() {
                rejected += 1;
            }
        }
        while !batch.is_empty() || engine.in_flight().0 > 0 {
            engine.submit(&mut batch).unwrap();
            engine.reap(&mut out, Wait::None);
            for c in out.drain() {
                assert!(c.result.is_ok());
                reaped += 1;
            }
        }
    }
    let after = allocations_on_this_thread();

    assert_eq!(after - before, 0, "the steady-state loop allocated");
    // 40 pushes per round into a batch of 32 that is empty at the start of each round.
    assert_eq!(rejected, 1_000 * 8);
    assert_eq!(reaped, 1_000 * 32);
}

/// The counter works: a growing `Vec` is seen.
#[test]
fn the_counting_allocator_sees_allocations() {
    let before = allocations_on_this_thread();
    let v: Vec<u64> = (0..1024).collect();
    assert_eq!(v.len(), 1024);
    assert!(allocations_on_this_thread() > before);
}
