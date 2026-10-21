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
    OpenMode, PushError, ReadCompletion, ReadRequest, SubmitBatch, SyncMode, Wait, WriteRequest,
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

/// One 4 KiB page, aligned the way RFC-0021 §6 aligns destinations.
#[repr(C, align(4096))]
struct Page([u8; 4096]);

const PAGE: u64 = 4096;

/// Memory the tests own, handed out one page per slot, as the tier manager hands out slots.
///
/// Each test makes at most one destination per page at a time: a page's earlier destination has
/// been dropped unsubmitted, or its request's completion reaped, before the next is made. Each
/// test keeps the pool alive until every destination and engine using it is gone.
struct Pool {
    pages: Vec<Page>,
}

impl Pool {
    fn new(slots: usize) -> Self {
        Self {
            pages: (0..slots).map(|_| Page([0; 4096])).collect(),
        }
    }

    fn base(&mut self) -> u64 {
        self.pages.as_mut_ptr() as u64
    }

    fn slot(&mut self, slot: usize) -> IoDest {
        assert!(slot < self.pages.len());
        let addr = self.base() + slot as u64 * PAGE;
        // SAFETY: the page lies in this pool's allocation, which outlives every destination and
        // engine that uses it; each test makes at most one destination per page at a time (see
        // `Pool`); and nothing else reads or writes the pages.
        unsafe { IoDest::raw(addr, Bytes(PAGE)) }
    }

    fn region(&mut self) -> HostRegion {
        HostRegion {
            addr: self.base(),
            len: Bytes(self.pages.len() as u64 * PAGE),
        }
    }
}

/// An engine that completes every operation in full on the next `reap` and touches no memory.
/// Its in-flight list is allocated once, at construction, like a real engine's ring.
struct ImmediateEngine {
    in_flight: Vec<(IoTicket, Bytes)>,
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

    unsafe fn register_buffers(
        &mut self,
        regions: &[HostRegion],
    ) -> Result<Vec<BufIndex>, IoError> {
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
        for op in &batch.pending()[..n] {
            self.in_flight.push((op.ticket(), op.extent().len));
        }
        batch.consume(n);
        Ok(u32::try_from(n).unwrap())
    }

    fn reap(&mut self, out: &mut CompletionBuf, _wait: Wait) -> u32 {
        let n = out.remaining().min(self.in_flight.len());
        for (ticket, len) in self.in_flight.drain(..n) {
            let c = ReadCompletion {
                ticket,
                result: Ok(len),
                latency: Nanos(0),
            };
            out.push(c)
                .expect("reap appends at most `remaining` completions");
        }
        u32::try_from(n).unwrap()
    }

    fn cancel(&mut self, ticket: IoTicket) -> CancelOutcome {
        match self.in_flight.iter().position(|(t, _)| *t == ticket) {
            Some(i) => {
                self.in_flight.remove(i);
                CancelOutcome::Cancelled
            }
            None => CancelOutcome::NotFound,
        }
    }

    fn in_flight(&self) -> (u32, Bytes) {
        let bytes = self.in_flight.iter().map(|(_, len)| len.0).sum();
        (u32::try_from(self.in_flight.len()).unwrap(), Bytes(bytes))
    }
}

fn extent(file: u16, ticket: u64) -> ExtentRef {
    ExtentRef {
        file: FileIx(file),
        offset: ticket * PAGE,
        len: Bytes(PAGE),
    }
}

fn read(ticket: u64, priority: IoPriority, dest: IoDest) -> IoOp {
    IoOp::Read(ReadRequest {
        ticket: IoTicket(ticket),
        extent: extent(0, ticket),
        dest,
        priority,
        useful_until: None,
    })
}

fn write(ticket: u64, src: IoDest) -> IoOp {
    IoOp::Write(WriteRequest {
        ticket: IoTicket(ticket),
        extent: extent(1, ticket),
        src,
        priority: IoPriority::Background,
        sync: SyncMode::DataSync,
    })
}

#[test]
fn submit_accepts_a_prefix_and_leaves_the_rest() {
    let mut pool = Pool::new(5);
    let mut engine = ImmediateEngine::new(2);
    let mut batch = SubmitBatch::with_capacity(8);
    for t in 0..5 {
        let dest = pool.slot(t);
        batch
            .push(read(t as u64, IoPriority::Demand, dest))
            .unwrap();
    }

    assert_eq!(engine.submit(&mut batch), Ok(2));
    let left: Vec<_> = batch.pending().iter().map(IoOp::ticket).collect();
    assert_eq!(left, [IoTicket(2), IoTicket(3), IoTicket(4)]);
    assert_eq!(engine.in_flight(), (2, Bytes(2 * PAGE)));

    // Depth is full: nothing more is accepted until a completion is reaped.
    assert_eq!(engine.submit(&mut batch), Ok(0));
    assert_eq!(batch.len(), 3);
}

#[test]
fn reap_appends_no_more_than_the_buffer_holds() {
    let mut pool = Pool::new(4);
    let mut engine = ImmediateEngine::new(4);
    let mut batch = SubmitBatch::with_capacity(4);
    for t in 0..4 {
        let dest = pool.slot(t);
        batch
            .push(read(t as u64, IoPriority::PrefetchNear, dest))
            .unwrap();
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
    assert_eq!(out.as_slice()[0].result, Ok(Bytes(PAGE)));
    assert_eq!(engine.in_flight(), (0, Bytes(0)));
}

#[test]
fn cancel_reports_its_outcome() {
    let mut pool = Pool::new(1);
    let mut engine = ImmediateEngine::new(4);
    let mut batch = SubmitBatch::with_capacity(4);
    let dest = pool.slot(0);
    batch.push(read(7, IoPriority::PrefetchFar, dest)).unwrap();
    engine.submit(&mut batch).unwrap();
    assert_eq!(engine.cancel(IoTicket(7)), CancelOutcome::Cancelled);
    assert_eq!(engine.cancel(IoTicket(7)), CancelOutcome::NotFound);
}

#[test]
fn an_engine_moves_onto_its_thread_as_a_trait_object() {
    let mut pool = Pool::new(2);
    let mut engine: Box<dyn IoEngine> = Box::new(ImmediateEngine::new(2));
    let regions = [pool.region()];
    // SAFETY: the pool owns the region and outlives the engine, which is dropped inside the
    // thread joined below; nothing else uses the pool's memory until then.
    let bufs = unsafe { engine.register_buffers(&regions) };
    assert_eq!(bufs, Ok(vec![BufIndex(0)]));
    assert_eq!(
        engine.open(Path::new("model.pack"), OpenMode::Read),
        Ok(FileIx(0))
    );
    assert_eq!(engine.kind(), EngineKind::PreadPool);

    // SAFETY: of the engines in this test process, only `engine` registers buffers, and its
    // buffer 0 is the pool's two pages, so page 1 lies inside it. The pool outlives the engine,
    // and nothing else uses page 1 while `src` or its request exists.
    let src = unsafe { IoDest::fixed(BufIndex(0), PAGE, Bytes(PAGE)) };
    let handle = std::thread::spawn(move || {
        let mut batch = SubmitBatch::with_capacity(1);
        batch.push(write(1, src)).unwrap();
        engine.submit(&mut batch).unwrap();
        let mut out = CompletionBuf::with_capacity(1);
        engine.reap(&mut out, Wait::None);
        out.as_slice()[0]
    });
    let c = handle.join().unwrap();
    drop(pool);
    assert_eq!(c.ticket, IoTicket(1));
    assert_eq!(c.result, Ok(Bytes(PAGE)));
}

/// The review's case: code without `unsafe` takes a 4 KiB slot the tier manager made and names a
/// 1 GiB extent for it. The push refuses it, so the engine never sees it.
#[test]
fn a_destination_bounds_the_transfer() {
    let mut pool = Pool::new(1);
    let mut engine = ImmediateEngine::new(4);
    let mut batch = SubmitBatch::with_capacity(4);
    let slot = pool.slot(0);
    let overrun = IoOp::Read(ReadRequest {
        ticket: IoTicket(1),
        extent: ExtentRef {
            file: FileIx(0),
            offset: 0,
            len: Bytes(1 << 30),
        },
        dest: slot,
        priority: IoPriority::Demand,
        useful_until: None,
    });
    let refused = batch.push(overrun).unwrap_err();
    assert!(matches!(refused, PushError::ExceedsDest(_)));
    assert!(batch.is_empty());
    assert_eq!(engine.submit(&mut batch), Ok(0));
    assert_eq!(engine.in_flight(), (0, Bytes(0)));

    // The same slot, taken back from the refused request, carries a one-page read.
    let IoOp::Read(request) = refused.into_op() else {
        unreachable!("a read was pushed")
    };
    batch
        .push(read(2, IoPriority::Demand, request.dest))
        .unwrap();
    assert_eq!(engine.submit(&mut batch), Ok(1));
    assert_eq!(engine.in_flight(), (1, Bytes(PAGE)));
}

/// The I/O loop's steady state: push chunks, submit, reap, drain, including pushes into a full
/// batch and reaps into a full buffer. After the buffers, the engine and the memory are set up at
/// `Warming`, none of it allocates.
#[test]
fn steady_state_submit_and_reap_allocate_nothing() {
    const BATCH: usize = 32;
    const PUSHES: usize = 40;
    let mut pool = Pool::new(PUSHES);
    let mut engine = ImmediateEngine::new(16);
    let mut batch = SubmitBatch::with_capacity(BATCH);
    let mut out = CompletionBuf::with_capacity(8);
    let mut reaped = 0u64;
    let mut rejected = 0u64;

    let before = allocations_on_this_thread();
    for round in 0..1_000u64 {
        // One page per push in a round, and every round drains before the next, so each page has
        // one destination at a time.
        for i in 0..PUSHES {
            let t = round * 40 + i as u64;
            let dest = pool.slot(i);
            let op = if i % 10 == 9 {
                write(t, dest)
            } else {
                read(t, IoPriority::ALL[i % 5], dest)
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
