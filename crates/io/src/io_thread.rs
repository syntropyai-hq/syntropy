// SPDX-License-Identifier: MIT OR Apache-2.0

//! The I/O thread loop (RFC-0021 §3.1, §3.3; ADR-0012 items 2, 4 and 5).
//!
//! One [`IoLoop`] runs on each volume's I/O thread and drives that thread's [`IoEngine`]. Each
//! producer (a device dispatcher, housekeeping, the loader) holds an [`IoClient`]: a request ring
//! into the loop, a completion ring back, and a doorbell word that wakes a parked loop. Every
//! ring is a `syntropy-spsc` ring, so fan-in is a set of SPSC rings that the one consumer polls,
//! with no lock (ADR-0012 item 4).
//!
//! Each pass of [`IoLoop::run`]:
//! 1. drains the producers' rings round-robin into five fixed-capacity queues, one per
//!    [`IoPriority`](crate::IoPriority). A read whose extent and `Fixed` destination both continue the read at the
//!    back of its queue is coalesced into it, up to the chunk size (§3.3);
//! 2. splits each queued read longer than the chunk size into chunks with
//!    [`IoDest::split_at`](crate::IoDest::split_at), and admits chunks to the [`SubmitBatch`] highest priority first,
//!    while the engine's in-flight count stays below [`EngineCaps::max_depth`];
//! 3. submits the batch and reaps completions without waiting;
//! 4. when every chunk of a request has completed, posts one [`ReadCompletion`] on the ring of
//!    the producer that sent it, with the producer's own ticket;
//! 5. with nothing in flight and every ring empty, spins for the spin budget and then parks on
//!    the doorbells. With work in flight but nothing it can submit, it reaps with
//!    [`Wait::Until`], at most [`REAP_WAIT_MAX`].
//!
//! Every buffer, queue and table is allocated in [`IoLoop::new`], at `Warming`, so a pass
//! allocates nothing (RFC-0021 goal 6; ADR-0012 item 6).
//!
//! Not built here: RFC-0021 §3.2's depth and priority rules beyond "highest priority first"
//! (`qd_target`, the prefetch byte cap, the `Background` share) and the `useful_until` drops are
//! TIER-E5-T07; the one retry of a short transfer (§16) is TIER-E5-T15, so a short transfer
//! completes with `IO_SHORT_READ` here; pinning the thread to a core is the runner's.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};
use std::sync::{Arc, OnceLock};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

use syntropy_core::{Bytes, ExtentRef, FileIx, Nanos};
use syntropy_spsc::{CapacityError, Consumer, PopError, Producer, PushError};

use crate::align::{self, RequestAlign};
use crate::batch::{CompletionBuf, SubmitBatch};
use crate::engine::{EngineCaps, IoEngine, Wait};
use crate::error::{IoError, IoErrorCode};
use crate::request::{IoOp, IoTicket, ReadCompletion, ReadRequest};

/// `io.chunk`: the largest read the loop submits as one operation, 4 MiB \[target\]
/// (RFC-0021 §3.1; the probe tunes it, §10).
pub const CHUNK_DEFAULT: Bytes = Bytes(4 << 20);

/// The longest the loop blocks in [`IoEngine::reap`] before it checks its rings again:
/// 100 µs \[target\] (RFC-0021 §3.1).
pub const REAP_WAIT_MAX: Nanos = Nanos(100_000);

/// The most requests one coalesced read carries. RFC-0021 §3.3 expects about 16 rows per token
/// for the surveyed n-gram table.
pub const MAX_COALESCE: usize = 16;

/// The sizes and budgets of one I/O thread's loop, fixed at `Warming`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopConfig {
    /// How many producers, one [`IoClient`] each.
    pub producers: usize,
    /// Slots in each producer's request ring; a power of two.
    pub request_ring: usize,
    /// Slots in each producer's completion ring; a power of two. It also caps the producer's
    /// requests the loop holds at once, so a completion always has a slot.
    pub completion_ring: usize,
    /// Requests the loop holds at once across all producers, queued or in flight.
    pub max_requests: usize,
    /// Operations the [`SubmitBatch`] holds.
    pub batch: usize,
    /// Completions the [`CompletionBuf`] takes from one reap.
    pub completions: usize,
    /// `io.chunk` (RFC-0021 §3.1): the largest read submitted as one operation. The loop
    /// rounds it down to a multiple of the volume's `A` and `M` (RFC-0021 §6), and up to that
    /// multiple if it is smaller, so every chunk of an aligned request is aligned.
    pub chunk: Bytes,
    /// The OS page size, part of RFC-0021 §6's memory alignment `M`.
    pub page: Bytes,
    /// `io.spin` (RFC-0021 §3.1): how long an idle loop spins before it parks; 50 µs, or
    /// 10 µs on battery (ADR-0012 item 5).
    pub spin: Duration,
    /// How long a loop with work in flight and nothing to submit blocks in
    /// [`IoEngine::reap`]; at most [`REAP_WAIT_MAX`].
    pub reap_wait: Nanos,
}

impl LoopConfig {
    /// The defaults for `producers` producers: rings of 256 slots \[target\], 1024 requests
    /// \[target\], batches and reaps of 64 \[target\], [`CHUNK_DEFAULT`], this host's page
    /// size, the 50 µs spin and [`REAP_WAIT_MAX`].
    #[must_use]
    pub fn new(producers: usize) -> Self {
        Self {
            producers,
            request_ring: 256,
            completion_ring: 256,
            max_requests: 1024,
            batch: 64,
            completions: 64,
            chunk: CHUNK_DEFAULT,
            page: Bytes(align::os_page_size()),
            spin: syntropy_spsc::SPIN_DEFAULT,
            reap_wait: REAP_WAIT_MAX,
        }
    }
}

/// Why [`IoLoop::new`] refused a [`LoopConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigError {
    /// No producers, or more than 65,536.
    Producers(usize),
    /// A ring capacity that is not a power of two.
    Ring(CapacityError),
    /// A zero chunk size, request count, batch or completion buffer.
    Zero(&'static str),
    /// A size too large for the loop's tables: more than 2^32 requests or operation slots.
    TooLarge(&'static str),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Producers(n) => write!(f, "{n} producers; 1 to 65536 are allowed"),
            ConfigError::Ring(e) => write!(f, "{e}"),
            ConfigError::Zero(what) => write!(f, "`{what}` must be at least 1"),
            ConfigError::TooLarge(what) => write!(f, "`{what}` is too large"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<CapacityError> for ConfigError {
    fn from(e: CapacityError) -> Self {
        ConfigError::Ring(e)
    }
}

/// One producer's doorbell word and completion count, on a cache line of their own so producers
/// don't share one.
#[repr(align(128))]
#[derive(Debug, Default)]
struct Word {
    /// Set by a loop about to park; cleared by the producer that unparks it.
    waiting: AtomicBool,
    /// Completions the producer has taken from its completion ring, ever. The loop compares it
    /// with the completions it has posted, so it never holds more of the producer's requests
    /// than the ring has free slots for.
    popped: AtomicU64,
}

/// The doorbells of one loop: one word per producer ring, and the loop's thread to unpark
/// (RFC-0021 §3.1; ADR-0012 item 5).
///
/// **Never a lost wake-up.** Before it parks, the loop sets every word, issues a sequentially
/// consistent fence, and drains the rings once more. A producer, after its push, issues the same
/// fence and reads its word. Of the two fences one comes first, so either the loop's drain sees
/// the push or the producer sees its word set and unparks the loop. A producer clears its word
/// when it unparks the loop, so a busy loop, which never sets the words, costs a producer one
/// fence and one load per request and no system call.
///
/// A producer also rings after it takes a completion: the slot it frees may be what the loop's
/// drain was waiting for, and the same fence pair keeps that wake-up from being lost.
#[derive(Debug)]
struct Doorbell {
    words: Box<[Word]>,
    thread: OnceLock<Thread>,
}

impl Doorbell {
    /// Called by producer `index` after a push, after taking a completion, and on its close.
    fn ring(&self, index: usize) {
        fence(Ordering::SeqCst);
        let word = &self.words[index].waiting;
        if word.load(Ordering::Relaxed)
            && word.swap(false, Ordering::AcqRel)
            && let Some(thread) = self.thread.get()
        {
            thread.unpark();
        }
    }

    fn arm(&self) {
        for word in &self.words {
            word.waiting.store(true, Ordering::Relaxed);
        }
        fence(Ordering::SeqCst);
    }

    fn disarm(&self) {
        for word in &self.words {
            word.waiting.store(false, Ordering::Relaxed);
        }
    }
}

/// A producer's end of an I/O thread: requests in, completions out (RFC-0021 §1).
///
/// `Send`, so it moves to the producer's thread. A producer posts a request and carries on; it
/// finds the completion when it next polls ([`IoClient::poll`]), or waits for it
/// ([`IoClient::wait`]). Every completion is taken through these two methods, which tell the
/// loop a slot is free. Dropping the client closes its request ring.
pub struct IoClient {
    /// `Some` until the client is dropped.
    requests: Option<Producer<IoOp>>,
    completions: Consumer<ReadCompletion>,
    bell: Arc<Doorbell>,
    index: usize,
}

impl IoClient {
    /// Posts a request and rings the doorbell. Never blocks or allocates.
    ///
    /// # Errors
    ///
    /// The ring's [`PushError`], with the request handed back: `Full` when the loop has not
    /// drained this ring yet, `Closed` when the loop is gone.
    pub fn submit(&mut self, op: IoOp) -> Result<(), PushError<IoOp>> {
        let ring = self
            .requests
            .as_mut()
            .expect("the request ring lives until the client is dropped");
        ring.push(op)?;
        self.bell.ring(self.index);
        Ok(())
    }

    /// The next completion, if one is ready. Never blocks.
    ///
    /// # Errors
    ///
    /// `Empty` when none is ready; `Closed` once the loop is gone and every completion has been
    /// taken.
    pub fn poll(&mut self) -> Result<ReadCompletion, PopError> {
        let completion = self.completions.pop()?;
        self.took_one();
        Ok(completion)
    }

    /// Waits for the next completion: spins for `spin`, then parks this thread until the loop
    /// posts one ([`Consumer::wait`]). `None` once the loop is gone and every completion has been
    /// taken.
    ///
    /// # Panics
    ///
    /// If this client has already waited on another thread, as [`Consumer::wait`] does.
    pub fn wait(&mut self, spin: Duration) -> Option<ReadCompletion> {
        let completion = self.completions.pop_wait(spin)?;
        self.took_one();
        Some(completion)
    }

    /// Counts a taken completion and rings the doorbell: the freed slot may let a parked loop
    /// take more of this producer's requests.
    fn took_one(&self) {
        self.bell.words[self.index]
            .popped
            .fetch_add(1, Ordering::Release);
        self.bell.ring(self.index);
    }

    /// This producer's index, as the loop numbers its rings.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }
}

impl fmt::Debug for IoClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IoClient")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl Drop for IoClient {
    fn drop(&mut self) {
        // Close the ring first, then ring, so a parked loop wakes and sees the close.
        drop(self.requests.take());
        self.bell.ring(self.index);
    }
}

/// What one [`IoLoop::poll_once`] pass left behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Poll {
    /// Chunks are queued and the engine has room for them.
    Busy,
    /// Chunks are in flight, and nothing more can be submitted until some complete.
    Blocked,
    /// Nothing is queued or in flight.
    Idle,
}

/// The loop's side of one producer.
struct Port {
    requests: Consumer<IoOp>,
    completions: Producer<ReadCompletion>,
    /// The producer's requests taken from its ring and not yet completed.
    outstanding: usize,
    /// Completions posted on the producer's ring, ever. Less [`Word::popped`], the ones still in
    /// the ring.
    posted: u64,
    /// Whether the request ring may still deliver requests.
    open: bool,
}

/// A request the loop holds, from intake until its completion is posted.
#[derive(Clone, Copy, Debug)]
struct Req {
    port: u16,
    ticket: IoTicket,
    /// Bytes transferred so far.
    done: u64,
    /// Queue entries and operations in flight that still carry part of the request.
    parts: u32,
    /// The first error any part met.
    error: Option<IoError>,
    start: Instant,
}

/// The requests one queue entry or operation carries, each with its byte count, in extent
/// order. One entry for a chunk; up to [`MAX_COALESCE`] for coalesced rows.
#[derive(Clone, Copy, Debug)]
struct Members {
    n: u8,
    req: [u32; MAX_COALESCE],
    len: [u64; MAX_COALESCE],
}

impl Members {
    const FREE: Members = Members {
        n: 0,
        req: [0; MAX_COALESCE],
        len: [0; MAX_COALESCE],
    };

    fn one(req: u32, len: u64) -> Self {
        let mut m = Members::FREE;
        m.n = 1;
        m.req[0] = req;
        m.len[0] = len;
        m
    }

    fn push(&mut self, req: u32, len: u64) {
        let i = usize::from(self.n);
        self.req[i] = req;
        self.len[i] = len;
        self.n += 1;
    }
}

/// A request, or the rest of one, waiting in a priority queue.
struct Pending {
    op: IoOp,
    members: Members,
}

/// One I/O thread's loop over its engine (RFC-0021 §3.1).
pub struct IoLoop<E> {
    engine: E,
    caps: EngineCaps,
    ports: Box<[Port]>,
    bell: Arc<Doorbell>,
    /// One queue per [`IoPriority`](crate::IoPriority), indexed by its discriminant.
    queues: [VecDeque<Pending>; 5],
    reqs: Vec<Req>,
    free_reqs: Vec<u32>,
    /// The requests each operation slot carries; `n == 0` for a free slot. An operation's
    /// ticket, as the engine sees it, is its slot index.
    ops: Vec<Members>,
    free_ops: Vec<u32>,
    batch: SubmitBatch,
    done: CompletionBuf,
    /// `io.chunk` as configured. Each request's chunks are this rounded to its file's `A` and
    /// `M` ([`RequestAlign::chunk`]).
    chunk: u64,
    /// The OS page size, part of RFC-0021 §6's `M`.
    page: u64,
    /// Requests rejected before submission with `IO_MISALIGNED` (RFC-0021 §6, §16).
    misaligned: u64,
    spin: Duration,
    reap_wait: Nanos,
    /// The port the next drain starts at, so no producer is always drained first.
    next_port: usize,
    /// How many times the loop has parked.
    parks: u64,
}

impl<E: IoEngine> IoLoop<E> {
    /// Builds the loop and one [`IoClient`] per producer, allocating every ring, queue and
    /// table now.
    ///
    /// # Errors
    ///
    /// [`ConfigError`] for a producer count outside 1 to 65,536, a ring capacity that is not a
    /// power of two, or a zero size.
    pub fn new(engine: E, config: &LoopConfig) -> Result<(Self, Vec<IoClient>), ConfigError> {
        if config.producers == 0 || config.producers > usize::from(u16::MAX) + 1 {
            return Err(ConfigError::Producers(config.producers));
        }
        for (value, what) in [
            (config.chunk.0, "chunk"),
            (config.max_requests as u64, "max_requests"),
            (config.batch as u64, "batch"),
            (config.completions as u64, "completions"),
        ] {
            if value == 0 {
                return Err(ConfigError::Zero(what));
            }
        }
        let caps = engine.caps();
        let bell = Arc::new(Doorbell {
            words: (0..config.producers).map(|_| Word::default()).collect(),
            thread: OnceLock::new(),
        });
        let mut ports = Vec::with_capacity(config.producers);
        let mut clients = Vec::with_capacity(config.producers);
        for index in 0..config.producers {
            let (req_tx, req_rx) = syntropy_spsc::ring(config.request_ring)?;
            let (done_tx, done_rx) = syntropy_spsc::ring(config.completion_ring)?;
            ports.push(Port {
                requests: req_rx,
                completions: done_tx,
                outstanding: 0,
                posted: 0,
                open: true,
            });
            clients.push(IoClient {
                requests: Some(req_tx),
                completions: done_rx,
                bell: bell.clone(),
                index,
            });
        }
        // Each request is in at most one queue entry, so a queue of `max_requests` never
        // fills, and a coalesced or chunked request never needs more.
        let queues = std::array::from_fn(|_| VecDeque::with_capacity(config.max_requests));
        let now = Instant::now();
        let idle = Req {
            port: 0,
            ticket: IoTicket(0),
            done: 0,
            parts: 0,
            error: None,
            start: now,
        };
        let op_slots = caps.max_depth as usize + config.batch;
        let max_requests = u32::try_from(config.max_requests)
            .map_err(|_| ConfigError::TooLarge("max_requests"))?;
        let op_count = u32::try_from(op_slots).map_err(|_| ConfigError::TooLarge("batch"))?;
        let looper = IoLoop {
            engine,
            caps,
            ports: ports.into_boxed_slice(),
            bell,
            queues,
            reqs: vec![idle; config.max_requests],
            free_reqs: (0..max_requests).rev().collect(),
            ops: vec![Members::FREE; op_slots],
            free_ops: (0..op_count).rev().collect(),
            batch: SubmitBatch::with_capacity(config.batch),
            done: CompletionBuf::with_capacity(config.completions),
            chunk: config.chunk.0,
            page: config.page.0,
            misaligned: 0,
            spin: config.spin,
            reap_wait: Nanos(config.reap_wait.0.min(REAP_WAIT_MAX.0)),
            next_port: 0,
            parks: 0,
        };
        Ok((looper, clients))
    }

    /// The engine.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// The engine, mutably: for `open` and `register_buffers` at `Loading`, and for tests.
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    /// RFC-0021 §6's `A` and `M` that a request on `file` is checked against now: the
    /// alignment the engine recorded for that file at `open` ([`IoEngine::file_dio`]), or else
    /// the engine's current [`EngineCaps::dio`], with this host's page size. Read at each
    /// request, never fixed when the loop is built, because files open after that (§6: "per
    /// open file").
    #[must_use]
    pub fn alignment(&self, file: FileIx) -> RequestAlign {
        let dio = self
            .engine
            .file_dio(file)
            .unwrap_or_else(|| self.engine.caps().dio);
        RequestAlign::from_dio(dio, self.page)
    }

    /// How many requests the loop has rejected with `IO_MISALIGNED` before submission: each one
    /// a producer bug (RFC-0021 §6, §16).
    #[must_use]
    pub fn misaligned(&self) -> u64 {
        self.misaligned
    }

    /// How many times the loop has parked on its doorbells.
    #[must_use]
    pub fn parks(&self) -> u64 {
        self.parks
    }

    /// Runs the loop until every producer has dropped its client and nothing is queued or in
    /// flight. Parks the calling thread while idle, so it is always called from the same
    /// thread: the I/O thread.
    ///
    /// # Panics
    ///
    /// If the loop has parked on another thread before: the doorbells unpark that thread.
    pub fn run(&mut self) {
        loop {
            match self.poll_once() {
                Poll::Busy => {}
                Poll::Blocked => {
                    // Blocking inside the kernel is fine on this thread (RFC-0021 §3.1).
                    self.reap(Wait::Until(self.reap_wait));
                }
                Poll::Idle => {
                    if self.ports.iter().all(|p| !p.open) {
                        return;
                    }
                    self.idle_wait();
                }
            }
        }
    }

    /// One pass of the loop without waiting: drain the rings, admit chunks, submit, reap, and
    /// post the completions of finished requests.
    pub fn poll_once(&mut self) -> Poll {
        self.drain();
        self.admit();
        let accepted = self.submit();
        self.reap(Wait::None);
        let (in_flight, _) = self.engine.in_flight();
        let queued = !self.batch.is_empty() || self.queues.iter().any(|q| !q.is_empty());
        if in_flight == 0 && !queued {
            Poll::Idle
        } else if queued && accepted > 0 && self.has_room(in_flight) {
            Poll::Busy
        } else {
            // Nothing more fits, or the engine accepted nothing this pass (its own queue may be
            // full): wait in `reap` rather than spin.
            Poll::Blocked
        }
    }

    /// Whether another chunk may join the batch.
    fn has_room(&self, in_flight: u32) -> bool {
        (in_flight as usize + self.batch.len()) < self.caps.max_depth as usize
            && !self.batch.is_full()
            && !self.free_ops.is_empty()
    }

    /// Spins for the spin budget, then parks on the doorbells until a producer pushes or closes.
    fn idle_wait(&mut self) {
        // The real clock: it decides only when this thread parks, never what is submitted.
        let start = Instant::now();
        let mut spins: u32 = 0;
        loop {
            if self.drain() > 0 {
                return;
            }
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(64) && start.elapsed() >= self.spin {
                break;
            }
        }
        let me = thread::current();
        let bound = self.bell.thread.get_or_init(|| me.clone());
        assert_eq!(
            bound.id(),
            me.id(),
            "an I/O loop parks on the thread that first parked it"
        );
        self.bell.arm();
        if self.drain() == 0 {
            // `park` may return early; the caller's next pass simply finds nothing.
            self.parks += 1;
            thread::park();
        }
        self.bell.disarm();
    }

    /// Moves requests from the producers' rings into the priority queues, starting at a
    /// different producer each time. Returns the requests taken plus the rings newly found
    /// closed.
    ///
    /// Each ring is drained in one visit, until the producer's requests in the loop plus its
    /// completions not yet taken fill its completion ring, so a completion always has a slot.
    /// The rows a token needs, posted together, stay together and can coalesce (RFC-0021 §3.3).
    fn drain(&mut self) -> usize {
        let count = self.ports.len();
        let mut progress = 0;
        for k in 0..count {
            let p = (self.next_port + k) % count;
            loop {
                // Completions posted and not yet taken still fill the producer's ring.
                let popped = self.bell.words[p].popped.load(Ordering::Acquire);
                let port = &mut self.ports[p];
                let unpopped =
                    usize::try_from(port.posted.wrapping_sub(popped)).unwrap_or(usize::MAX);
                if !port.open
                    || port.outstanding.saturating_add(unpopped) >= port.completions.capacity()
                    || self.free_reqs.is_empty()
                {
                    break;
                }
                match port.requests.pop() {
                    Ok(op) => {
                        port.outstanding += 1;
                        progress += 1;
                        self.intake(p, op);
                    }
                    Err(PopError::Empty) => break,
                    Err(PopError::Closed) => {
                        port.open = false;
                        progress += 1;
                        break;
                    }
                }
            }
        }
        self.next_port = (self.next_port + 1) % count;
        progress
    }

    /// Takes one request into its queue, coalescing it into the queue's last read when both its
    /// extent and its destination continue that read.
    fn intake(&mut self, port: usize, op: IoOp) {
        let now = Instant::now();
        let len = op.extent().len.0;
        let r = self
            .free_reqs
            .pop()
            .expect("drain checks for a free request");
        self.reqs[r as usize] = Req {
            port: u16::try_from(port).expect("at most 65536 producers"),
            ticket: op.ticket(),
            done: 0,
            parts: 1,
            error: None,
            start: now,
        };
        let align = self.alignment(op.extent().file);
        if len > op.dest().len().0 || align.check(&op).is_err() {
            // A request that breaks RFC-0021 §6's alignment never reaches the engine, so never
            // the kernel: it is a bug, rejected and counted (`IO_MISALIGNED`, §16). So is an
            // extent longer than its destination, which `SubmitBatch::push` would refuse
            // (RFC-0021 §2); §16 gives that no code of its own, and `IO_MISALIGNED` is the other
            // request-shape bug of class `Internal`.
            self.misaligned += 1;
            let req = &mut self.reqs[r as usize];
            req.error = Some(IoError::new(IoErrorCode::Misaligned));
            req.parts = 0;
            self.finish(r);
            return;
        }
        let queue = &mut self.queues[op.priority() as usize];
        let op = match (queue.back(), op) {
            (Some(back), IoOp::Read(next))
                if can_coalesce(back, &next, align.chunk(self.chunk)) =>
            {
                let back = queue.pop_back().expect("checked above");
                let IoOp::Read(prev) = back.op else {
                    unreachable!("can_coalesce takes only reads")
                };
                match prev.dest.join(next.dest) {
                    Ok(dest) => {
                        let mut members = back.members;
                        members.push(r, len);
                        queue.push_back(Pending {
                            op: IoOp::Read(ReadRequest {
                                extent: ExtentRef {
                                    len: Bytes(prev.extent.len.0 + len),
                                    ..prev.extent
                                },
                                dest,
                                ..prev
                            }),
                            members,
                        });
                        return;
                    }
                    Err((prev_dest, next_dest)) => {
                        // `can_coalesce` checked adjacency, so this does not happen; keep both.
                        debug_assert!(false, "can_coalesce and IoDest::join disagree");
                        queue.push_back(Pending {
                            op: IoOp::Read(ReadRequest {
                                dest: prev_dest,
                                ..prev
                            }),
                            members: back.members,
                        });
                        IoOp::Read(ReadRequest {
                            dest: next_dest,
                            ..next
                        })
                    }
                }
            }
            (_, op) => op,
        };
        queue.push_back(Pending {
            op,
            members: Members::one(r, len),
        });
    }

    /// The next operation to submit, highest priority first: the front of the first non-empty
    /// queue, or its first chunk when it is a read longer than the chunk size.
    fn next_chunk(&mut self) -> Option<(IoOp, Members)> {
        let file = self
            .queues
            .iter()
            .find(|q| !q.is_empty())?
            .front()?
            .op
            .extent()
            .file;
        // Chunks at a multiple of the file's `A` and `M` keep an aligned request aligned.
        let chunk = self.alignment(file).chunk(self.chunk);
        let queue = self.queues.iter_mut().find(|q| !q.is_empty())?;
        let front = queue.pop_front().expect("found non-empty");
        let len = front.op.extent().len.0;
        let IoOp::Read(read) = front.op else {
            // Writes are submitted whole: their sync mode applies to the whole extent.
            return Some((front.op, front.members));
        };
        if len <= chunk {
            return Some((IoOp::Read(read), front.members));
        }
        // Only a single request is longer than a chunk: coalescing stops at the chunk size.
        let r = front.members.req[0];
        let (head, rest) = match read.dest.split_at(Bytes(chunk)) {
            Ok(halves) => halves,
            Err(_) => unreachable!("intake checked that the extent fits its destination"),
        };
        let extent = read.extent;
        queue.push_front(Pending {
            op: IoOp::Read(ReadRequest {
                ticket: read.ticket,
                extent: ExtentRef {
                    file: extent.file,
                    offset: extent.offset + chunk,
                    len: Bytes(len - chunk),
                },
                dest: rest,
                priority: read.priority,
                useful_until: read.useful_until,
            }),
            members: Members::one(r, len - chunk),
        });
        self.reqs[r as usize].parts += 1;
        Some((
            IoOp::Read(ReadRequest {
                ticket: read.ticket,
                extent: ExtentRef {
                    file: extent.file,
                    offset: extent.offset,
                    len: Bytes(chunk),
                },
                dest: head,
                priority: read.priority,
                useful_until: read.useful_until,
            }),
            Members::one(r, chunk),
        ))
    }

    /// Fills the batch with chunks while the engine has depth for them.
    fn admit(&mut self) {
        let (in_flight, _) = self.engine.in_flight();
        while self.has_room(in_flight) {
            let Some((mut op, members)) = self.next_chunk() else {
                break;
            };
            let slot = self
                .free_ops
                .pop()
                .expect("has_room checks for a free slot");
            self.ops[slot as usize] = members;
            // Chunks of an aligned request at a multiple of `A` and `M`, and coalesced aligned
            // reads, stay aligned. The check runs again here, at the last step before the
            // engine, in case the file's alignment changed after intake.
            if self.alignment(op.extent().file).check(&op).is_err() {
                self.misaligned += 1;
                self.complete_op(slot, Err(IoError::new(IoErrorCode::Misaligned)));
                continue;
            }
            let ticket = IoTicket(u64::from(slot));
            match &mut op {
                IoOp::Read(r) => r.ticket = ticket,
                IoOp::Write(w) => w.ticket = ticket,
            }
            if self.batch.push(op).is_err() {
                // Neither refusal can happen: intake checked the length, has_room the space.
                self.complete_op(slot, Err(IoError::new(IoErrorCode::Misaligned)));
            }
        }
    }

    /// Submits the batch, returning how many operations left it. What the engine does not
    /// accept stays for the next pass; on an engine error, every operation left in the batch
    /// fails with it.
    fn submit(&mut self) -> usize {
        if self.batch.is_empty() {
            return 0;
        }
        let before = self.batch.len();
        if let Err(e) = self.engine.submit(&mut self.batch) {
            for i in 0..self.batch.len() {
                let slot = self.batch.pending()[i].ticket().0;
                self.complete_op(slot_index(slot), Err(e));
            }
            self.batch.clear();
        }
        before - self.batch.len()
    }

    /// Reaps completions and settles each operation's requests.
    fn reap(&mut self, wait: Wait) {
        self.engine.reap(&mut self.done, wait);
        for i in 0..self.done.len() {
            let c = self.done.as_slice()[i];
            self.complete_op(slot_index(c.ticket.0), c.result);
        }
        self.done.clear();
    }

    /// Settles operation `slot`: credits each request it carried, in extent order, and posts a
    /// request's completion once its last part is done.
    fn complete_op(&mut self, slot: u32, result: Result<Bytes, IoError>) {
        let Some(members) = self.ops.get_mut(slot as usize) else {
            debug_assert!(false, "the engine completed an unknown ticket {slot}");
            return;
        };
        if members.n == 0 {
            debug_assert!(false, "the engine completed a free ticket {slot}");
            return;
        }
        let m = std::mem::replace(members, Members::FREE);
        self.free_ops.push(slot);
        let mut left = match result {
            Ok(Bytes(n)) => n,
            Err(_) => 0,
        };
        for k in 0..usize::from(m.n) {
            let r = m.req[k];
            let req = &mut self.reqs[r as usize];
            match result {
                Ok(_) => {
                    let got = left.min(m.len[k]);
                    left -= got;
                    req.done += got;
                    if got < m.len[k] {
                        // The one retry of RFC-0021 §16 is TIER-E5-T15's.
                        req.error
                            .get_or_insert(IoError::new(IoErrorCode::ShortRead));
                    }
                }
                Err(e) => {
                    req.error.get_or_insert(e);
                }
            }
            req.parts -= 1;
            if req.parts == 0 {
                self.finish(r);
            }
        }
    }

    /// Posts request `r`'s completion on its producer's ring and frees it.
    fn finish(&mut self, r: u32) {
        let req = self.reqs[r as usize];
        let completion = ReadCompletion {
            ticket: req.ticket,
            result: req.error.map_or(Ok(Bytes(req.done)), Err),
            latency: Nanos(u64::try_from(req.start.elapsed().as_nanos()).unwrap_or(u64::MAX)),
        };
        let port = &mut self.ports[usize::from(req.port)];
        port.outstanding -= 1;
        match port.completions.push(completion) {
            Ok(()) => port.posted += 1,
            // A producer that has gone no longer wants its completions.
            Err(PushError::Closed(_)) => {}
            Err(PushError::Full(_)) => {
                // `drain` takes a request only while the producer's requests in the loop plus
                // its completions not yet taken leave a slot, and the producer can take a
                // completion only through `IoClient`, which counts it. So this is a bug here,
                // never something a producer can cause.
                unreachable!("the loop held more requests than the completion ring has slots")
            }
        }
        self.free_reqs.push(r);
    }
}

impl<E> fmt::Debug for IoLoop<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IoLoop")
            .field("producers", &self.ports.len())
            .field("chunk", &self.chunk)
            .field(
                "queued",
                &self.queues.iter().map(VecDeque::len).sum::<usize>(),
            )
            .finish_non_exhaustive()
    }
}

/// An operation's ticket as the slot index it was given in [`IoLoop::admit`]. A ticket that
/// does not fit is out of range, and `complete_op` ignores it.
fn slot_index(ticket: u64) -> u32 {
    u32::try_from(ticket).unwrap_or(u32::MAX)
}

/// Whether `next` continues the read at the back of its queue: the same file, the next bytes of
/// it, and the next bytes of the same registered buffer, within the chunk size and
/// [`MAX_COALESCE`] requests (RFC-0021 §3.3).
fn can_coalesce(back: &Pending, next: &ReadRequest, chunk: u64) -> bool {
    use crate::request::IoDestKind::Fixed;
    let IoOp::Read(prev) = &back.op else {
        return false;
    };
    let (
        Fixed { buf, offset },
        Fixed {
            buf: next_buf,
            offset: next_offset,
        },
    ) = (prev.dest.kind(), next.dest.kind())
    else {
        return false;
    };
    let prev_len = prev.extent.len.0;
    usize::from(back.members.n) < MAX_COALESCE
        && prev.useful_until == next.useful_until
        && prev.extent.file == next.extent.file
        && prev.extent.offset.checked_add(prev_len) == Some(next.extent.offset)
        // The bytes land where the file continues only if the previous read fills its
        // destination exactly.
        && prev.dest.len().0 == prev_len
        && buf == next_buf
        && offset.checked_add(prev_len) == Some(next_offset)
        && prev_len
            .checked_add(next.extent.len.0)
            .is_some_and(|total| total <= chunk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_errors_are_reported() {
        let ok = LoopConfig::new(1);
        assert_eq!(ok.chunk, Bytes(4 << 20));
        assert_eq!(ok.spin, Duration::from_micros(50));
        assert_eq!(ok.reap_wait, Nanos(100_000));
        for (config, want) in [
            (LoopConfig::new(0), ConfigError::Producers(0)),
            (LoopConfig::new(65_537), ConfigError::Producers(65_537)),
            (
                LoopConfig {
                    chunk: Bytes(0),
                    ..LoopConfig::new(1)
                },
                ConfigError::Zero("chunk"),
            ),
        ] {
            assert_eq!(check(&config), Err(want));
        }
        let bad_ring = LoopConfig {
            request_ring: 3,
            ..LoopConfig::new(1)
        };
        assert!(matches!(check(&bad_ring), Err(ConfigError::Ring(_))));
        assert_eq!(
            ConfigError::Zero("batch").to_string(),
            "`batch` must be at least 1"
        );
    }

    /// `IoLoop::new` over an engine with no depth, for the config checks.
    fn check(config: &LoopConfig) -> Result<(), ConfigError> {
        IoLoop::new(NoEngine, config).map(|_| ())
    }

    struct NoEngine;

    impl IoEngine for NoEngine {
        fn kind(&self) -> syntropy_core::resource::EngineKind {
            syntropy_core::resource::EngineKind::PreadPool
        }
        fn caps(&self) -> EngineCaps {
            EngineCaps {
                fixed_buffers: false,
                fixed_files: false,
                writes: false,
                cancel: false,
                max_depth: 1,
                direct: true,
                dio: syntropy_core::resource::DioAlign { mem: 0, offset: 0 },
            }
        }
        fn open(
            &mut self,
            _: &std::path::Path,
            _: crate::OpenMode,
        ) -> Result<syntropy_core::FileIx, IoError> {
            Err(IoError::new(IoErrorCode::EngineUnavailable))
        }
        unsafe fn register_buffers(
            &mut self,
            _: &[syntropy_core::HostRegion],
        ) -> Result<Vec<crate::BufIndex>, IoError> {
            Ok(Vec::new())
        }
        fn submit(&mut self, _: &mut SubmitBatch) -> Result<u32, IoError> {
            Ok(0)
        }
        fn reap(&mut self, _: &mut CompletionBuf, _: Wait) -> u32 {
            0
        }
        fn cancel(&mut self, _: IoTicket) -> crate::CancelOutcome {
            crate::CancelOutcome::NotFound
        }
        fn in_flight(&self) -> (u32, Bytes) {
            (0, Bytes(0))
        }
    }

    #[test]
    fn doorbell_wakes_only_an_armed_loop() {
        let bell = Doorbell {
            words: (0..2).map(|_| Word::default()).collect(),
            thread: OnceLock::new(),
        };
        // Not armed: ringing changes nothing.
        bell.ring(0);
        assert!(!bell.words[0].waiting.load(Ordering::Relaxed));
        bell.arm();
        assert!(bell.words.iter().all(|w| w.waiting.load(Ordering::Relaxed)));
        // Ringing clears only the producer's own word; with no thread bound, nothing unparks.
        bell.ring(1);
        assert!(bell.words[0].waiting.load(Ordering::Relaxed));
        assert!(!bell.words[1].waiting.load(Ordering::Relaxed));
        bell.disarm();
        assert!(
            bell.words
                .iter()
                .all(|w| !w.waiting.load(Ordering::Relaxed))
        );
        assert_eq!(std::mem::align_of::<Word>(), 128);
    }

    #[test]
    fn io_priority_discriminants_index_the_queues() {
        use crate::request::IoPriority;
        for (i, p) in IoPriority::ALL.iter().enumerate() {
            assert_eq!(*p as usize, i);
        }
    }
}
