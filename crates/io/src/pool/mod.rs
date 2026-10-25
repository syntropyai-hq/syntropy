// SPDX-License-Identifier: MIT OR Apache-2.0

//! The thread-pool engines: [`PreadPool`] on Linux and Windows (RFC-0021 §4.2; ADR-0013
//! item 3), and [`MacNoCache`] and [`MacMapped`] on macOS (RFC-0021 §5; ADR-0013 item 5).
//!
//! All are one pool: the I/O thread keeps its queues and depth rules (RFC-0021 §3) and hands
//! each operation to one of `threads` blocking worker threads over that worker's own SPSC ring.
//! The worker runs the positional read or write and posts the result on its own SPSC ring back.
//! No multi-producer queue is used. The thread count is the engine's depth
//! ([`EngineCaps::max_depth`]), so the engine sends an operation only to a worker with nothing
//! outstanding.
//!
//! What differs per OS sits behind the crate-private `Backend` trait: how a file is opened
//! (`O_DIRECT`, `FILE_FLAG_NO_BUFFERING`, or `F_NOCACHE` and `F_RDAHEAD`, and for `MacMapped` a
//! read-only mapping), how a read and a write are issued, and what a buffered fallback does after a read (`POSIX_FADV_DONTNEED` on
//! Linux). A file whose direct open or first aligned read is refused is opened buffered, and the
//! engine reports [`EngineCaps::direct`] `false`: the `buffered` volume flag of RFC-0021 §4.2,
//! which the probe copies into `StorageCapsExt::buffered`. [`PreadPool::buffered_reason`],
//! [`MacNoCache::buffered_reason`] and [`MacMapped::buffered_reason`] say why.
//!
//! **Cancellation** (RFC-0021 §4.2) removes an operation a worker has not started: it completes
//! with `IO_CANCELLED` and no system call. An operation already in the kernel completes
//! normally, and [`IoEngine::cancel`] says `TooLate`.
//!
//! **Waiting.** [`IoEngine::reap`] with [`Wait::Until`] parks the I/O thread on one wake word
//! that every worker checks after it posts a result, the same pattern as the I/O loop's
//! doorbells (RFC-0021 §3.1 note of 2026-10-22).
//!
//! Every table, ring and queue is allocated when the engine is made, so `submit` and `reap`
//! allocate nothing (RFC-0021 goal 6). Opening a file and registering buffers happen at
//! `Loading` and may allocate.
//!
//! **Alignment** (RFC-0021 §6). Each backend asks [`align::query`](crate::align::query) for a
//! file's direct-I/O alignment when it opens it (`statx(STATX_DIOALIGN)`, `FileStorageInfo`, the
//! 16 KiB macOS page), and [`IoEngine::file_dio`] reports it per file. [`EngineCaps::dio`] stays
//! zero, §6's 4096-byte floor, because the alignment belongs to each file, not to the engine.
//!
//! Not here: pinning the workers to the I/O core set is the runner's.

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
mod mapped;
#[cfg(windows)]
mod windows;

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub use linux::PreadPool;
#[cfg(target_os = "macos")]
pub use macos::MacNoCache;
#[cfg(target_os = "macos")]
pub use mapped::{MacMapped, RESIDENT_POLL, RESIDENT_WAIT_MAX};
#[cfg(windows)]
pub use windows::PreadPool;

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering, fence};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

use syntropy_core::resource::{DioAlign, EngineKind};
use syntropy_core::{Bytes, FileIx, HostRegion, Nanos};
use syntropy_spsc::{Consumer, Producer};

use crate::batch::{CompletionBuf, SubmitBatch};
use crate::engine::{BufIndex, CancelOutcome, EngineCaps, IoEngine, OpenMode, Wait};
use crate::error::{IoError, IoErrorCode};
use crate::request::{IoDestKind, IoOp, IoTicket, ReadCompletion, SyncMode};

/// `io.pool_threads` on Linux and Windows: 16 \[target\] (RFC-0021 §15).
pub const POOL_THREADS_DEFAULT: usize = 16;

/// `io.pool_threads_macos`: 8 \[target\] (RFC-0021 §15).
pub const POOL_THREADS_MACOS_DEFAULT: usize = 8;

/// The most worker threads one pool runs.
pub const POOL_THREADS_MAX: usize = 256;

/// The most destinations one worker operation carries: a gathered `preadv` on macOS
/// (RFC-0021 §5). The same bound as the I/O loop's coalescing, [`crate::MAX_COALESCE`].
pub(crate) const MAX_SEGS: usize = crate::io_thread::MAX_COALESCE;

/// Messages one worker's ring holds. A worker has at most one operation outstanding (the
/// thread count is the depth), so the rest of the room is for file handles sent at `open`.
const WORKER_RING: usize = 8;

/// The size and alignment of the probe read that checks direct I/O at open (RFC-0021 §4.2):
/// one 4 KiB block at offset 0, which meets every logical block size up to 4 KiB.
#[cfg(not(target_os = "macos"))]
pub(crate) const PROBE: usize = 4096;

/// Worker operation states, one per slot.
const QUEUED: u8 = 0;
const RUNNING: u8 = 1;
const CANCELLED: u8 = 2;

/// One destination range of a worker operation, already resolved to an address.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Seg {
    /// First byte.
    pub addr: u64,
    /// Bytes to transfer.
    pub len: u64,
}

/// A file as a backend opened it: the handles the workers use, and whether direct I/O holds.
pub(crate) struct Opened<H> {
    /// One handle that every worker shares, or one per worker, in worker order.
    pub handles: Vec<Arc<H>>,
    /// `None` for direct I/O; for the flagged buffered mode, the `IO_DIRECT_REFUSED` error that
    /// caused it (RFC-0021 §4.2, §16).
    pub refused: Option<IoError>,
    /// The file's direct-I/O alignment from [`align::query`](crate::align::query), zero when
    /// the OS reported none (RFC-0021 §6).
    pub dio: DioAlign,
}

/// What one OS supplies to the pool.
pub(crate) trait Backend: Send + Sync + 'static {
    /// An open file as a worker uses it.
    type Handle: Send + Sync + 'static;

    /// The engine kind the pool reports.
    const KIND: EngineKind;

    /// Contiguous reads in a batch are gathered into one vectored read until it holds at least
    /// this many bytes; 0 turns gathering off.
    const GATHER_TO: u64;

    /// Opens `path` for `workers` worker threads, direct where the file system allows it.
    fn open(
        &self,
        path: &Path,
        mode: OpenMode,
        workers: usize,
    ) -> Result<Opened<Self::Handle>, IoError>;

    /// Reads from `offset` into `segs`, in order, as one contiguous file range; returns the
    /// bytes read, which may be short.
    ///
    /// # Safety
    ///
    /// Each segment is memory valid for writes over its whole length that nothing else reads
    /// or writes until the call returns: the [`IoDest`](crate::IoDest) contract of the request
    /// it came from, which holds until that request's completion is reaped.
    unsafe fn read(&self, h: &Self::Handle, offset: u64, segs: &[Seg]) -> Result<u64, IoError>;

    /// Writes `seg` at `offset`; returns the bytes written, which may be short.
    ///
    /// # Safety
    ///
    /// `seg` is memory valid for reads over its whole length that nothing writes until the
    /// call returns, as for [`Backend::read`].
    unsafe fn write(&self, h: &Self::Handle, offset: u64, seg: Seg) -> Result<u64, IoError>;

    /// Makes written data durable per `mode`.
    fn sync(&self, h: &Self::Handle, mode: SyncMode) -> Result<(), IoError>;
}

/// An I/O error code for an OS error number from a read, write or sync (RFC-0021 §16): a
/// refused alignment is `IO_MISALIGNED`, a full volume `IO_NO_SPACE`, and anything else
/// `IO_SHORT_READ`, which the I/O thread retries once before `IO_EXTENT_BAD`.
pub(crate) fn transfer_error(os: i32, misaligned: i32, no_space: &[i32]) -> IoError {
    let code = if os == misaligned {
        IoErrorCode::Misaligned
    } else if no_space.contains(&os) {
        IoErrorCode::NoSpace
    } else {
        IoErrorCode::ShortRead
    };
    IoError::with_os_error(code, os)
}

/// The error for a file that cannot be opened at all. RFC-0021 §16 has no code for it; the
/// file's extents cannot be read, which is `IO_EXTENT_BAD`'s action (rebuild, never serve), so
/// that code carries the OS error, except a full volume, which is `IO_NO_SPACE`.
pub(crate) fn open_error(e: &std::io::Error, no_space: &[i32]) -> IoError {
    match e.raw_os_error() {
        Some(os) if no_space.contains(&os) => IoError::with_os_error(IoErrorCode::NoSpace, os),
        Some(os) => IoError::with_os_error(IoErrorCode::ExtentBad, os),
        None => IoError::new(IoErrorCode::ExtentBad),
    }
}

/// A page-aligned scratch block for the open-time probe read.
#[cfg(not(target_os = "macos"))]
pub(crate) struct ProbeBuf {
    ptr: *mut u8,
}

#[cfg(not(target_os = "macos"))]
impl ProbeBuf {
    const LAYOUT: std::alloc::Layout = match std::alloc::Layout::from_size_align(PROBE, PROBE) {
        Ok(l) => l,
        Err(_) => panic!("PROBE is a power of two"),
    };

    /// Allocates the block, or `None` if the allocator refuses.
    pub(crate) fn new() -> Option<Self> {
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { std::alloc::alloc(Self::LAYOUT) };
        (!ptr.is_null()).then_some(Self { ptr })
    }

    /// The block as one segment: owned by this value, valid for writes, used by nothing else.
    pub(crate) fn seg(&self) -> Seg {
        Seg {
            addr: self.ptr as u64,
            len: PROBE as u64,
        }
    }
}

#[cfg(not(target_os = "macos"))]
impl Drop for ProbeBuf {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc` with this layout and is freed once.
        unsafe { std::alloc::dealloc(self.ptr, Self::LAYOUT) }
    }
}

/// What a worker runs.
#[derive(Clone, Copy, Debug)]
enum JobKind {
    Read,
    Write(SyncMode),
}

/// One operation for a worker: plain data, so the ring moves it without allocating.
#[derive(Clone, Copy, Debug)]
struct Job {
    slot: u32,
    file: FileIx,
    offset: u64,
    kind: JobKind,
    segs: [Seg; MAX_SEGS],
    nsegs: u8,
    at: Instant,
}

// A job is moved through a preallocated ring slot; boxing it would allocate per operation.
#[allow(clippy::large_enum_variant)]
enum Msg<H> {
    /// The handle for the next file index.
    AddFile(Arc<H>),
    Job(Job),
}

/// A worker's result for one slot.
#[derive(Clone, Copy, Debug)]
struct Done {
    slot: u32,
    result: Result<u64, IoError>,
    latency: Nanos,
}

/// The wake word the I/O thread parks on in a blocking reap.
#[derive(Debug, Default)]
struct Bell {
    waiting: AtomicBool,
    thread: OnceLock<Thread>,
}

impl Bell {
    /// A worker's half: after posting a result, wakes the I/O thread if it is parked.
    fn ring(&self) {
        fence(Ordering::SeqCst);
        if self.waiting.load(Ordering::Relaxed)
            && self.waiting.swap(false, Ordering::Relaxed)
            && let Some(t) = self.thread.get()
        {
            t.unpark();
        }
    }
}

struct Worker<H> {
    tx: Producer<Msg<H>>,
    rx: Consumer<Done>,
    /// Operations sent and not yet taken back.
    outstanding: u32,
    join: Option<JoinHandle<()>>,
}

/// The engine's record of one slot in use.
#[derive(Clone, Copy, Debug)]
struct Slot {
    tickets: [IoTicket; MAX_SEGS],
    lens: [u64; MAX_SEGS],
    /// Operations in the slot; 0 when the slot is free.
    n: u8,
}

/// The pool engine over one backend.
pub(crate) struct Pool<B: Backend> {
    backend: Arc<B>,
    workers: Vec<Worker<B::Handle>>,
    bell: Arc<Bell>,
    states: Arc<[AtomicU8]>,
    slots: Vec<Slot>,
    free: Vec<u32>,
    /// Completions taken from the workers and not yet reaped.
    ready: VecDeque<ReadCompletion>,
    /// The most operations held at once, in slots or in `ready`.
    ops_cap: usize,
    in_flight: u32,
    in_flight_bytes: u64,
    /// Each open file's direct-I/O alignment, by [`FileIx`].
    files: Vec<DioAlign>,
    regions: Vec<HostRegion>,
    refused: Option<IoError>,
    next_worker: usize,
}

impl<B: Backend> Pool<B> {
    /// Starts `threads` workers over `backend`.
    ///
    /// # Panics
    ///
    /// If `threads` is 0 or above [`POOL_THREADS_MAX`].
    pub(crate) fn new(backend: B, threads: usize) -> Result<Self, IoError> {
        assert!(
            (1..=POOL_THREADS_MAX).contains(&threads),
            "a pool runs 1 to {POOL_THREADS_MAX} threads, not {threads}"
        );
        let backend = Arc::new(backend);
        let bell = Arc::new(Bell::default());
        let states: Arc<[AtomicU8]> = (0..threads).map(|_| AtomicU8::new(QUEUED)).collect();
        let mut pool = Self {
            backend,
            workers: Vec::with_capacity(threads),
            bell,
            states,
            slots: vec![
                Slot {
                    tickets: [IoTicket(0); MAX_SEGS],
                    lens: [0; MAX_SEGS],
                    n: 0,
                };
                threads
            ],
            free: (0..threads as u32).rev().collect(),
            ready: VecDeque::with_capacity(threads * MAX_SEGS),
            ops_cap: threads * MAX_SEGS,
            in_flight: 0,
            in_flight_bytes: 0,
            files: Vec::new(),
            regions: Vec::new(),
            refused: None,
            next_worker: 0,
        };
        for i in 0..threads {
            let unavailable = |_| IoError::new(IoErrorCode::EngineUnavailable);
            let (tx, job_rx) = syntropy_spsc::ring(WORKER_RING).map_err(|_| unavailable(()))?;
            let (done_tx, rx) = syntropy_spsc::ring(WORKER_RING).map_err(|_| unavailable(()))?;
            let (backend, bell, states) = (
                Arc::clone(&pool.backend),
                Arc::clone(&pool.bell),
                Arc::clone(&pool.states),
            );
            let join = thread::Builder::new()
                .name(format!("syntropy-io-pool-{i}"))
                .spawn(move || work(&*backend, job_rx, done_tx, &bell, &states))
                .map_err(|e| {
                    e.raw_os_error()
                        .map_or(IoError::new(IoErrorCode::EngineUnavailable), |os| {
                            IoError::with_os_error(IoErrorCode::EngineUnavailable, os)
                        })
                })?;
            pool.workers.push(Worker {
                tx,
                rx,
                outstanding: 0,
                join: Some(join),
            });
        }
        Ok(pool)
    }

    /// Why the engine is in the flagged buffered mode, or `None` while every file it opened is
    /// direct.
    pub(crate) fn buffered_reason(&self) -> Option<IoError> {
        self.refused
    }

    /// The address `len` bytes of `dest` start at, or `None` for a `Fixed` destination outside
    /// the buffers this engine registered.
    fn resolve(&self, kind: IoDestKind, len: u64) -> Option<u64> {
        match kind {
            IoDestKind::Raw { addr } => Some(addr),
            IoDestKind::Fixed { buf, offset } => {
                let r = self.regions.get(usize::from(buf.0))?;
                let end = offset.checked_add(len)?;
                (end <= r.len.0).then(|| r.addr + offset)
            }
        }
    }

    /// Completes an operation the engine refuses before any worker sees it.
    fn refuse(&mut self, ticket: IoTicket, error: IoError) {
        self.ready.push_back(ReadCompletion {
            ticket,
            result: Err(error),
            latency: Nanos(0),
        });
        self.in_flight += 1;
    }

    /// The worker with nothing outstanding to send the next operation to, round-robin.
    fn idle_worker(&mut self) -> Option<usize> {
        let n = self.workers.len();
        (0..n)
            .map(|k| (self.next_worker + k) % n)
            .find(|&w| self.workers[w].outstanding == 0)
    }

    /// Takes every result the workers have posted into `ready`. Returns how many it took.
    fn collect(&mut self) -> usize {
        let mut took = 0;
        for w in 0..self.workers.len() {
            while let Ok(done) = self.workers[w].rx.pop() {
                self.workers[w].outstanding -= 1;
                took += 1;
                let slot = done.slot as usize;
                let s = self.slots[slot];
                let mut left = done.result;
                self.in_flight_bytes -= s.lens[..usize::from(s.n)].iter().sum::<u64>();
                for k in 0..usize::from(s.n) {
                    let result = match &mut left {
                        Ok(n) => {
                            let got = (*n).min(s.lens[k]);
                            *n -= got;
                            Ok(Bytes(got))
                        }
                        Err(e) => Err(*e),
                    };
                    self.ready.push_back(ReadCompletion {
                        ticket: s.tickets[k],
                        result,
                        latency: done.latency,
                    });
                }
                self.slots[slot].n = 0;
                self.free.push(done.slot);
            }
        }
        took
    }

    /// Moves completions from `ready` into `out`.
    fn deliver(&mut self, out: &mut CompletionBuf) -> u32 {
        let mut n = 0;
        while out.remaining() > 0 {
            let Some(c) = self.ready.pop_front() else {
                break;
            };
            self.in_flight -= 1;
            let _ = out.push(c);
            n += 1;
        }
        n
    }
}

impl<B: Backend> IoEngine for Pool<B> {
    fn kind(&self) -> EngineKind {
        B::KIND
    }

    fn caps(&self) -> EngineCaps {
        EngineCaps {
            fixed_buffers: false,
            fixed_files: false,
            writes: true,
            cancel: true,
            max_depth: self.workers.len() as u32,
            direct: self.refused.is_none(),
            dio: DioAlign { mem: 0, offset: 0 },
        }
    }

    fn open(&mut self, path: &Path, mode: OpenMode) -> Result<FileIx, IoError> {
        let ix = u16::try_from(self.files.len())
            .map_err(|_| IoError::new(IoErrorCode::RegisterFailed))?;
        let opened = self.backend.open(path, mode, self.workers.len())?;
        assert!(
            !opened.handles.is_empty(),
            "a backend returns at least one handle"
        );
        for w in 0..self.workers.len() {
            let mut msg = Msg::AddFile(Arc::clone(&opened.handles[w % opened.handles.len()]));
            // The worker takes handles ahead of any job, so the ring frees up at once.
            loop {
                match self.workers[w].tx.push(msg) {
                    Ok(()) => break,
                    Err(syntropy_spsc::PushError::Full(m)) => {
                        msg = m;
                        thread::yield_now();
                    }
                    Err(syntropy_spsc::PushError::Closed(_)) => {
                        return Err(IoError::new(IoErrorCode::EngineUnavailable));
                    }
                }
            }
        }
        self.files.push(opened.dio);
        if self.refused.is_none() {
            self.refused = opened.refused;
        }
        Ok(FileIx(ix))
    }

    fn file_dio(&self, file: FileIx) -> Option<DioAlign> {
        self.files.get(usize::from(file.0)).copied()
    }

    unsafe fn register_buffers(
        &mut self,
        regions: &[HostRegion],
    ) -> Result<Vec<BufIndex>, IoError> {
        let first = self.regions.len();
        if first + regions.len() > usize::from(u16::MAX) + 1 {
            return Err(IoError::new(IoErrorCode::RegisterFailed));
        }
        self.regions.extend_from_slice(regions);
        Ok((first..first + regions.len())
            .map(|i| BufIndex(i as u16))
            .collect())
    }

    fn submit(&mut self, batch: &mut SubmitBatch) -> Result<u32, IoError> {
        let mut accepted = 0u32;
        while let Some(op) = batch.pending().first() {
            if self.in_flight as usize >= self.ops_cap {
                break;
            }
            let (ticket, extent) = (op.ticket(), op.extent());
            let len = extent.len.0;
            if usize::from(extent.file.0) >= self.files.len() {
                self.refuse(ticket, IoError::new(IoErrorCode::Misaligned));
                batch.consume(1);
                accepted += 1;
                continue;
            }
            let Some(addr) = self.resolve(op.dest().kind(), len) else {
                self.refuse(ticket, IoError::new(IoErrorCode::Misaligned));
                batch.consume(1);
                accepted += 1;
                continue;
            };
            let kind = match op {
                IoOp::Read(_) => JobKind::Read,
                IoOp::Write(w) => JobKind::Write(w.sync),
            };
            if self.free.is_empty() {
                break;
            }
            let Some(w) = self.idle_worker() else {
                break;
            };
            let mut job = Job {
                slot: 0,
                file: extent.file,
                offset: extent.offset,
                kind,
                segs: [Seg::default(); MAX_SEGS],
                nsegs: 1,
                at: Instant::now(),
            };
            job.segs[0] = Seg { addr, len };
            let mut slot = Slot {
                tickets: [IoTicket(0); MAX_SEGS],
                lens: [0; MAX_SEGS],
                n: 1,
            };
            slot.tickets[0] = ticket;
            slot.lens[0] = len;
            let mut total = len;
            // Gather the reads that continue this one into one vectored read (RFC-0021 §5).
            if matches!(kind, JobKind::Read) && B::GATHER_TO > 0 {
                let mut end = extent.offset + len;
                for next in &batch.pending()[1..] {
                    let n = usize::from(job.nsegs);
                    if total >= B::GATHER_TO
                        || n == MAX_SEGS
                        || self.in_flight as usize + n + 1 > self.ops_cap
                    {
                        break;
                    }
                    let IoOp::Read(r) = next else { break };
                    if r.extent.file != extent.file || r.extent.offset != end {
                        break;
                    }
                    let Some(a) = self.resolve(r.dest.kind(), r.extent.len.0) else {
                        break;
                    };
                    job.segs[n] = Seg {
                        addr: a,
                        len: r.extent.len.0,
                    };
                    slot.tickets[n] = r.ticket;
                    slot.lens[n] = r.extent.len.0;
                    job.nsegs += 1;
                    slot.n += 1;
                    total += r.extent.len.0;
                    end += r.extent.len.0;
                }
            }
            let id = self.free.pop().expect("checked above");
            job.slot = id;
            self.states[id as usize].store(QUEUED, Ordering::Release);
            match self.workers[w].tx.push(Msg::Job(job)) {
                Ok(()) => {}
                Err(_) => {
                    self.free.push(id);
                    return Err(IoError::new(IoErrorCode::EngineUnavailable));
                }
            }
            self.slots[id as usize] = slot;
            self.workers[w].outstanding += 1;
            self.next_worker = (w + 1) % self.workers.len();
            self.in_flight += u32::from(slot.n);
            self.in_flight_bytes += total;
            batch.consume(usize::from(slot.n));
            accepted += u32::from(slot.n);
        }
        Ok(accepted)
    }

    fn reap(&mut self, out: &mut CompletionBuf, wait: Wait) -> u32 {
        self.collect();
        let n = self.deliver(out);
        if n > 0 || out.remaining() == 0 || self.in_flight == 0 {
            return n;
        }
        let Wait::Until(t) = wait else {
            return 0;
        };
        let deadline = Instant::now() + Duration::from_nanos(t.0);
        // A worker unparks the thread bound here; on any other thread the park below just times
        // out. One engine stays on one I/O thread, so that is rare.
        self.bell.thread.get_or_init(thread::current);
        loop {
            self.bell.waiting.store(true, Ordering::Relaxed);
            fence(Ordering::SeqCst);
            if self.collect() > 0 {
                self.bell.waiting.store(false, Ordering::Relaxed);
                return self.deliver(out);
            }
            let now = Instant::now();
            if now >= deadline {
                self.bell.waiting.store(false, Ordering::Relaxed);
                return 0;
            }
            thread::park_timeout(deadline - now);
            self.bell.waiting.store(false, Ordering::Relaxed);
        }
    }

    fn cancel(&mut self, ticket: IoTicket) -> CancelOutcome {
        for (i, s) in self.slots.iter().enumerate() {
            let n = usize::from(s.n);
            if !s.tickets[..n].contains(&ticket) {
                continue;
            }
            // A gathered read carries other requests too: it runs, and the I/O thread discards
            // this one's completion.
            if n > 1 {
                return CancelOutcome::TooLate;
            }
            return match self.states[i].compare_exchange(
                QUEUED,
                CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => CancelOutcome::Cancelled,
                Err(_) => CancelOutcome::TooLate,
            };
        }
        if self.ready.iter().any(|c| c.ticket == ticket) {
            return CancelOutcome::TooLate;
        }
        CancelOutcome::NotFound
    }

    fn in_flight(&self) -> (u32, Bytes) {
        (self.in_flight, Bytes(self.in_flight_bytes))
    }
}

impl<B: Backend> Drop for Pool<B> {
    fn drop(&mut self) {
        // Operations no worker has started are skipped; one in the kernel finishes. Closing the
        // rings stops each worker once it has drained its ring, and joining means no worker
        // touches memory after the engine is gone.
        for s in self.states.iter() {
            let _ = s.compare_exchange(QUEUED, CANCELLED, Ordering::AcqRel, Ordering::Acquire);
        }
        let mut joins = Vec::with_capacity(self.workers.len());
        for w in self.workers.drain(..) {
            let Worker { tx, rx, join, .. } = w;
            drop(tx);
            drop(rx);
            joins.extend(join);
        }
        for j in joins {
            let _ = j.join();
        }
    }
}

/// One worker thread: runs each job from its ring and posts the result back.
fn work<B: Backend>(
    backend: &B,
    mut jobs: Consumer<Msg<B::Handle>>,
    mut done: Producer<Done>,
    bell: &Bell,
    states: &[AtomicU8],
) {
    let mut files: Vec<Arc<B::Handle>> = Vec::new();
    while let Some(msg) = jobs.pop_wait(Duration::ZERO) {
        let job = match msg {
            Msg::AddFile(h) => {
                files.push(h);
                continue;
            }
            Msg::Job(job) => job,
        };
        let state = &states[job.slot as usize];
        let result = if state
            .compare_exchange(QUEUED, RUNNING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            Err(IoError::new(IoErrorCode::Cancelled))
        } else {
            let h = &files[usize::from(job.file.0)];
            let segs = &job.segs[..usize::from(job.nsegs)];
            match job.kind {
                // SAFETY: each segment came from a request's `IoDest`, resolved and bounds-checked
                // by `Pool::submit`; its contract reserves the range for this operation until the
                // request's completion is reaped, which is after this result is posted.
                JobKind::Read => unsafe { backend.read(h, job.offset, segs) },
                JobKind::Write(sync) => {
                    // SAFETY: as for the read: the source range is reserved for this operation.
                    let wrote = unsafe { backend.write(h, job.offset, segs[0]) };
                    match (wrote, sync) {
                        (Ok(n), SyncMode::None) => Ok(n),
                        (Ok(n), mode) => backend.sync(h, mode).map(|()| n),
                        (Err(e), _) => Err(e),
                    }
                }
            }
        };
        let latency = Nanos(u64::try_from(job.at.elapsed().as_nanos()).unwrap_or(u64::MAX));
        // The engine sends a worker one job at a time, so the result ring always has room; it
        // is closed only when the engine is dropping, and then nobody wants the result.
        let posted = done.push(Done {
            slot: job.slot,
            result,
            latency,
        });
        if posted.is_err() && done.is_closed() {
            continue;
        }
        bell.ring();
    }
}

/// Implements [`IoEngine`] for a newtype over [`Pool`] by forwarding every method.
macro_rules! forward_engine {
    ($t:ty) => {
        impl $crate::engine::IoEngine for $t {
            fn kind(&self) -> syntropy_core::resource::EngineKind {
                self.pool.kind()
            }
            fn caps(&self) -> $crate::engine::EngineCaps {
                self.pool.caps()
            }
            fn open(
                &mut self,
                path: &std::path::Path,
                mode: $crate::engine::OpenMode,
            ) -> Result<syntropy_core::FileIx, $crate::error::IoError> {
                self.pool.open(path, mode)
            }
            fn file_dio(
                &self,
                file: syntropy_core::FileIx,
            ) -> Option<syntropy_core::resource::DioAlign> {
                self.pool.file_dio(file)
            }
            unsafe fn register_buffers(
                &mut self,
                regions: &[syntropy_core::HostRegion],
            ) -> Result<Vec<$crate::engine::BufIndex>, $crate::error::IoError> {
                // SAFETY: the caller upholds this method's contract, which the pool's shares.
                unsafe { self.pool.register_buffers(regions) }
            }
            fn submit(
                &mut self,
                batch: &mut $crate::batch::SubmitBatch,
            ) -> Result<u32, $crate::error::IoError> {
                self.pool.submit(batch)
            }
            fn reap(
                &mut self,
                out: &mut $crate::batch::CompletionBuf,
                wait: $crate::engine::Wait,
            ) -> u32 {
                self.pool.reap(out, wait)
            }
            fn cancel(
                &mut self,
                ticket: $crate::request::IoTicket,
            ) -> $crate::engine::CancelOutcome {
                self.pool.cancel(ticket)
            }
            fn in_flight(&self) -> (u32, syntropy_core::Bytes) {
                self.pool.in_flight()
            }
        }
    };
}
#[allow(unused_imports)]
pub(crate) use forward_engine;
