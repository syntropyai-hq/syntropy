// SPDX-License-Identifier: MIT OR Apache-2.0

//! `IocpEngine` (RFC-0021 §7; ADR-0013 item 4): overlapped `ReadFile` and `WriteFile` on files
//! associated with one completion port, reaped with `GetQueuedCompletionStatusEx`.
//!
//! Each operation in flight owns one slot of a table allocated once, at construction. The slot
//! starts with the operation's `OVERLAPPED`, so a completion packet's `OVERLAPPED` pointer leads
//! back to the slot. The table never moves, and it is freed only once the kernel holds none of
//! its `OVERLAPPED`s.
//!
//! **Helper submitters.** One thread issuing overlapped calls usually reaches the depth a volume
//! needs. When the probe finds it submission-bound, the engine is built with up to
//! `io.iocp_threads − 1` helpers ([`IocpOptions::helpers`]; 4 threads in all by default,
//! [`IOCP_THREADS_DEFAULT`]). The I/O thread then fills each slot and hands it to a helper over
//! an SPSC ring, round robin; the helper issues the call, and a call that fails at once is
//! posted to the port with `PostQueuedCompletionStatus`, so every completion reaches the I/O
//! thread the same way.
//!
//! **Writes with a sync mode** are flushed with `FlushFileBuffers` when their completion is
//! reaped, before the completion is returned. The flush blocks the I/O thread for its duration;
//! such writes are `Background` pack and session writes (RFC-0021 §13).

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::os::windows::io::AsRawHandle;
use std::path::Path;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use syntropy_core::resource::{DioAlign, EngineKind};
use syntropy_core::{Bytes, FileIx, HostRegion};
use syntropy_spsc::{Consumer, Producer, SPIN_DEFAULT};
use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_IO_PENDING, GetLastError, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Storage::FileSystem::{FlushFileBuffers, ReadFile, WriteFile};
use windows_sys::Win32::System::IO::{
    CancelIoEx, CreateIoCompletionPort, GetOverlappedResult, GetQueuedCompletionStatusEx,
    OVERLAPPED, OVERLAPPED_0, OVERLAPPED_0_0, OVERLAPPED_ENTRY, PostQueuedCompletionStatus,
};

use super::common::{self, MAX_CALL, Regions, WinFile};
use crate::batch::{CompletionBuf, SubmitBatch};
use crate::engine::{BufIndex, CancelOutcome, EngineCaps, IoEngine, OpenMode, Wait};
use crate::error::{IoError, IoErrorCode};
use crate::request::{IoOp, IoTicket, ReadCompletion, SyncMode};

/// The `io.iocp_threads` default: the I/O thread and three helpers (ADR-0013 item 4).
pub const IOCP_THREADS_DEFAULT: u32 = 4;

/// How an [`IocpEngine`] is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IocpOptions {
    /// The most operations in flight at once.
    pub depth: u32,
    /// Helper threads that issue the calls, for an I/O thread the probe found submission-bound;
    /// zero issues every call on the I/O thread. At most `io.iocp_threads − 1`.
    pub helpers: u32,
}

impl Default for IocpOptions {
    /// 64 in flight and no helpers.
    fn default() -> Self {
        IocpOptions {
            depth: 64,
            helpers: 0,
        }
    }
}

/// A completion key that marks a packet a helper posted for a call that failed at once. The low
/// 32 bits hold the Win32 error. Files are associated with key 0.
const FAILED: usize = 1 << (usize::BITS - 1);

/// How long dropping the engine waits for the kernel to finish with its operations.
const DRAIN: Duration = Duration::from_secs(30);

/// One operation in flight. `#[repr(C)]` with the `OVERLAPPED` first, so the kernel's pointer to
/// it is a pointer to the slot. The kernel writes the `OVERLAPPED` while the operation is in
/// flight, so it sits in an `UnsafeCell`, and the engine reads it only once the packet is
/// dequeued.
#[repr(C)]
struct Slot {
    overlapped: UnsafeCell<OVERLAPPED>,
    ticket: IoTicket,
    file: u16,
    len: u64,
    write: bool,
    sync: SyncMode,
    issued: Option<Instant>,
    busy: bool,
}

impl Slot {
    fn idle() -> Self {
        Slot {
            overlapped: UnsafeCell::new(OVERLAPPED::default()),
            ticket: IoTicket(0),
            file: 0,
            len: 0,
            write: false,
            sync: SyncMode::None,
            issued: None,
            busy: false,
        }
    }
}

/// A call for a helper to issue: plain integers, so it can cross threads.
struct Job {
    overlapped: usize,
    handle: usize,
    addr: usize,
    len: u32,
    write: bool,
}

struct Helper {
    jobs: Option<Producer<Job>>,
    thread: Option<JoinHandle<()>>,
}

/// The IOCP engine. See the module documentation.
pub struct IocpEngine {
    port: HANDLE,
    files: Vec<WinFile>,
    regions: Regions,
    slots: Box<[Slot]>,
    free: Vec<u32>,
    /// Completions of operations that failed before reaching the kernel, returned by the next
    /// `reap`.
    ready: VecDeque<ReadCompletion>,
    entries: Box<[OVERLAPPED_ENTRY]>,
    busy: u32,
    busy_bytes: u64,
    helpers: Vec<Helper>,
    next_helper: usize,
}

// SAFETY: the port and file handles are process-wide and usable from any thread. The slot table
// is touched by the thread that owns the engine and, through the `OVERLAPPED` pointers it hands
// over, by the kernel and the helpers; moving the engine to another thread moves none of it.
unsafe impl Send for IocpEngine {}

impl IocpEngine {
    /// A completion port and its slot table, with `options.helpers` helper threads.
    ///
    /// # Errors
    ///
    /// [`IoErrorCode::EngineUnavailable`] if the port, a ring or a helper thread cannot be made.
    pub fn new(options: IocpOptions) -> Result<Self, IoError> {
        let unavailable = |os: Option<i32>| match os {
            Some(os) => IoError::with_os_error(IoErrorCode::EngineUnavailable, os),
            None => IoError::new(IoErrorCode::EngineUnavailable),
        };
        let depth = options.depth.max(1);
        // SAFETY: a new port, not associated with any file; every argument is a plain value.
        let port =
            unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, std::ptr::null_mut(), 0, 1) };
        if port.is_null() {
            // SAFETY: reads this thread's last error, which the failed call set.
            return Err(unavailable(Some(unsafe { GetLastError() } as i32)));
        }
        let mut engine = IocpEngine {
            port,
            files: Vec::new(),
            regions: Regions::default(),
            slots: (0..depth).map(|_| Slot::idle()).collect(),
            free: (0..depth).rev().collect(),
            ready: VecDeque::with_capacity(depth as usize),
            entries: (0..depth).map(|_| OVERLAPPED_ENTRY::default()).collect(),
            busy: 0,
            busy_bytes: 0,
            helpers: Vec::new(),
            next_helper: 0,
        };
        for _ in 0..options.helpers {
            // At most `depth` jobs are in flight, so a ring of `depth` slots never fills.
            let capacity = (depth as usize).next_power_of_two();
            let (jobs, queue) =
                syntropy_spsc::ring::<Job>(capacity).map_err(|_| unavailable(None))?;
            let port = engine.port as usize;
            let thread = std::thread::Builder::new()
                .name("syntropy-iocp-submit".into())
                .spawn(move || helper(queue, port))
                .map_err(|e| unavailable(e.raw_os_error()))?;
            engine.helpers.push(Helper {
                jobs: Some(jobs),
                thread: Some(thread),
            });
        }
        Ok(engine)
    }

    /// Accepts `op` into a free slot, with `file` as its file, and issues it or hands it to a
    /// helper. Returns `false`, accepting nothing, when every slot is busy.
    pub(super) fn accept(&mut self, op: &IoOp, file: FileIx) -> bool {
        let Some(&ix) = self.free.last() else {
            return false;
        };
        let issued = Instant::now();
        let extent = op.extent();
        let len = extent.len.0.min(MAX_CALL);
        let Some(f) = self.files.get(usize::from(file.0)) else {
            self.refuse(op.ticket(), IoError::new(IoErrorCode::ExtentBad), issued);
            return true;
        };
        let Some(addr) = self.regions.address(op.dest().kind(), len) else {
            self.refuse(op.ticket(), IoError::new(IoErrorCode::Misaligned), issued);
            return true;
        };
        let handle = f.file.as_raw_handle() as usize;
        let (write, sync) = match op {
            IoOp::Read(_) => (false, SyncMode::None),
            IoOp::Write(w) => (true, w.sync),
        };
        self.free.pop();
        let slot = &mut self.slots[ix as usize];
        *slot = Slot {
            overlapped: UnsafeCell::new(OVERLAPPED {
                Anonymous: OVERLAPPED_0 {
                    Anonymous: OVERLAPPED_0_0 {
                        Offset: extent.offset as u32,
                        OffsetHigh: (extent.offset >> 32) as u32,
                    },
                },
                ..OVERLAPPED::default()
            }),
            ticket: op.ticket(),
            file: file.0,
            len,
            write,
            sync,
            issued: Some(issued),
            busy: true,
        };
        let mut job = Job {
            overlapped: slot.overlapped.get() as usize,
            handle,
            addr: addr as usize,
            len: len as u32,
            write,
        };
        self.busy += 1;
        self.busy_bytes += len;
        if !self.helpers.is_empty() {
            let n = self.helpers.len();
            let helper = &mut self.helpers[self.next_helper % n];
            self.next_helper = self.next_helper.wrapping_add(1);
            // A ring holds `depth` jobs and at most `depth` are in flight, so a push fails only
            // if the helper is gone; the call is then issued here.
            if let Some(ring) = helper.jobs.as_mut() {
                match ring.push(job) {
                    Ok(()) => return true,
                    Err(e) => job = e.into_inner(),
                }
            }
        }
        // SAFETY: `job.overlapped` is this slot's `OVERLAPPED`, which stays allocated and is not
        // reused until its completion is reaped (see `Slot`). `job.addr .. + job.len` is memory
        // the operation's destination contract gives this engine until then, checked against
        // the registered regions for a `Fixed` destination. `job.handle` is open for as long as
        // the engine is.
        if let Err(code) = unsafe { issue(&job) } {
            let result = common::failed(code, 0);
            let c = self.complete(ix as usize, result);
            self.ready.push_back(c);
        }
        true
    }

    /// Completes an operation that never reached the kernel.
    fn refuse(&mut self, ticket: IoTicket, error: IoError, issued: Instant) {
        self.ready
            .push_back(common::completion(ticket, Err(error), Some(issued)));
    }

    /// Frees slot `ix` and returns its completion.
    fn complete(&mut self, ix: usize, result: Result<Bytes, IoError>) -> ReadCompletion {
        let slot = &mut self.slots[ix];
        let len = slot.len;
        slot.busy = false;
        let c = common::completion(slot.ticket, result, slot.issued.take());
        self.busy -= 1;
        self.busy_bytes = self.busy_bytes.saturating_sub(len);
        self.free.push(ix as u32);
        c
    }

    /// The slot an `OVERLAPPED` pointer from the port belongs to, if it is one of this engine's.
    fn slot_of(&self, overlapped: *mut OVERLAPPED) -> Option<usize> {
        let base = self.slots.as_ptr() as usize;
        let at = (overlapped as usize).checked_sub(base)?;
        let ix = at / size_of::<Slot>();
        (at % size_of::<Slot>() == 0 && ix < self.slots.len() && self.slots[ix].busy).then_some(ix)
    }

    /// The result of the completed operation in slot `ix`, `bytes` long by the packet.
    fn result_of(&self, ix: usize, bytes: u32) -> Result<Bytes, IoError> {
        let slot = &self.slots[ix];
        let Some(file) = self.files.get(usize::from(slot.file)) else {
            return Err(IoError::new(IoErrorCode::ExtentBad));
        };
        let handle = file.file.as_raw_handle() as HANDLE;
        let mut transferred = 0u32;
        // SAFETY: the slot's `OVERLAPPED` belongs to a completed operation on `handle`; with
        // `bWait` false the call only reads the status the kernel left in it.
        let ok = unsafe { GetOverlappedResult(handle, slot.overlapped.get(), &mut transferred, 0) };
        if ok == 0 {
            // SAFETY: reads this thread's last error, which the failed call set.
            return common::failed(unsafe { GetLastError() }, u64::from(bytes));
        }
        let result = Ok(Bytes(u64::from(transferred)));
        if slot.write && slot.sync != SyncMode::None {
            // SAFETY: `handle` is open; the flush touches no memory of this process.
            if unsafe { FlushFileBuffers(handle) } == 0 {
                // SAFETY: as above.
                return common::failed(unsafe { GetLastError() }, 0);
            }
        }
        result
    }

    /// Takes up to `out.remaining()` packets from the port, waiting up to `ms` for the first.
    fn drain_port(&mut self, out: &mut CompletionBuf, ms: u32) -> u32 {
        let max = out.remaining().min(self.entries.len());
        if max == 0 || self.busy == 0 {
            return 0;
        }
        let mut got = 0u32;
        // SAFETY: `entries` has room for `max` entries, and the port is open.
        let ok = unsafe {
            GetQueuedCompletionStatusEx(
                self.port,
                self.entries.as_mut_ptr(),
                max as u32,
                &mut got,
                ms,
                0,
            )
        };
        if ok == 0 {
            return 0;
        }
        let mut n = 0;
        for i in 0..got as usize {
            let entry = self.entries[i];
            let Some(ix) = self.slot_of(entry.lpOverlapped) else {
                continue;
            };
            let result = if entry.lpCompletionKey & FAILED != 0 {
                common::failed((entry.lpCompletionKey & 0xFFFF_FFFF) as u32, 0)
            } else {
                self.result_of(ix, entry.dwNumberOfBytesTransferred)
            };
            let c = self.complete(ix, result);
            if out.push(c).is_ok() {
                n += 1;
            }
        }
        n
    }
}

/// Issues `job`'s call. `Ok` once it is in the kernel, whether it completed at once or is
/// pending: an overlapped handle posts a packet either way.
///
/// # Safety
///
/// `job.overlapped` points to an `OVERLAPPED` that stays valid and untouched until its packet is
/// dequeued; `job.addr .. job.addr + job.len` is memory valid for the transfer until then; and
/// `job.handle` is an open overlapped handle associated with the engine's port.
unsafe fn issue(job: &Job) -> Result<(), u32> {
    let handle = job.handle as HANDLE;
    let overlapped = job.overlapped as *mut OVERLAPPED;
    // SAFETY: the caller's contract.
    let ok = unsafe {
        if job.write {
            WriteFile(
                handle,
                job.addr as *const u8,
                job.len,
                std::ptr::null_mut(),
                overlapped,
            )
        } else {
            ReadFile(
                handle,
                job.addr as *mut u8,
                job.len,
                std::ptr::null_mut(),
                overlapped,
            )
        }
    };
    if ok != 0 {
        return Ok(());
    }
    // SAFETY: reads this thread's last error, which the failed call set.
    match unsafe { GetLastError() } {
        ERROR_IO_PENDING => Ok(()),
        code => Err(code),
    }
}

/// A helper's loop: issue each job until the engine drops its ring.
fn helper(mut jobs: Consumer<Job>, port: usize) {
    while let Some(job) = jobs.pop_wait(SPIN_DEFAULT) {
        // SAFETY: the I/O thread built `job` from a slot it reserved, as `issue` requires, and
        // does not touch the slot again until its packet is dequeued.
        if let Err(code) = unsafe { issue(&job) } {
            // SAFETY: the port is open while any helper runs (the engine joins them before
            // closing it), and the `OVERLAPPED` is the job's own.
            unsafe {
                PostQueuedCompletionStatus(
                    port as HANDLE,
                    0,
                    FAILED | code as usize,
                    job.overlapped as *const OVERLAPPED,
                )
            };
        }
    }
}

impl IoEngine for IocpEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Iocp
    }

    fn caps(&self) -> EngineCaps {
        EngineCaps {
            fixed_buffers: false,
            fixed_files: false,
            writes: true,
            cancel: true,
            max_depth: self.slots.len() as u32,
            direct: self.files.iter().all(|f| f.direct),
            dio: common::widest(self.files.iter().map(|f| f.dio)),
        }
    }

    fn open(&mut self, path: &Path, mode: OpenMode) -> Result<FileIx, IoError> {
        let ix = u16::try_from(self.files.len())
            .map_err(|_| IoError::new(IoErrorCode::RegisterFailed))?;
        let file = common::open(path, mode)?;
        let handle = file.file.as_raw_handle() as HANDLE;
        // SAFETY: `handle` is an open overlapped file handle and the port is open; key 0 marks
        // the packets of calls that reached the kernel.
        let port = unsafe { CreateIoCompletionPort(handle, self.port, 0, 0) };
        if port.is_null() {
            // SAFETY: reads this thread's last error, which the failed call set.
            let os = unsafe { GetLastError() } as i32;
            return Err(IoError::with_os_error(IoErrorCode::EngineUnavailable, os));
        }
        self.files.push(file);
        Ok(FileIx(ix))
    }

    fn file_dio(&self, file: FileIx) -> Option<DioAlign> {
        self.files.get(usize::from(file.0)).map(|f| f.dio)
    }

    unsafe fn register_buffers(
        &mut self,
        regions: &[HostRegion],
    ) -> Result<Vec<BufIndex>, IoError> {
        // IOCP has no registered buffers: the regions are kept to resolve `Fixed` destinations.
        self.regions.add(regions)
    }

    fn submit(&mut self, batch: &mut SubmitBatch) -> Result<u32, IoError> {
        let mut accepted = 0;
        for op in batch.pending() {
            if !self.accept(op, op.extent().file) {
                break;
            }
            accepted += 1;
        }
        batch.consume(accepted);
        Ok(accepted as u32)
    }

    fn reap(&mut self, out: &mut CompletionBuf, wait: Wait) -> u32 {
        let mut n = 0;
        while out.remaining() > 0 {
            let Some(c) = self.ready.pop_front() else {
                break;
            };
            if out.push(c).is_ok() {
                n += 1;
            }
        }
        let ms = if n > 0 { 0 } else { common::millis(wait) };
        n + self.drain_port(out, ms)
    }

    fn cancel(&mut self, ticket: IoTicket) -> CancelOutcome {
        let Some(slot) = self.slots.iter().find(|s| s.busy && s.ticket == ticket) else {
            return CancelOutcome::NotFound;
        };
        if let Some(file) = self.files.get(usize::from(slot.file)) {
            let handle = file.file.as_raw_handle() as HANDLE;
            // SAFETY: `handle` is open and the `OVERLAPPED` is a busy slot's. A call a helper
            // has not issued yet is not found, and completes normally.
            unsafe { CancelIoEx(handle, slot.overlapped.get()) };
        }
        CancelOutcome::TooLate
    }

    fn in_flight(&self) -> (u32, Bytes) {
        let ready = self.ready.len() as u32;
        (self.busy + ready, Bytes(self.busy_bytes))
    }
}

impl Drop for IocpEngine {
    fn drop(&mut self) {
        // Dropping a helper's ring ends its loop once it has issued every job already queued.
        for helper in &mut self.helpers {
            helper.jobs = None;
            if let Some(thread) = helper.thread.take() {
                let _ = thread.join();
            }
        }
        for slot in self.slots.iter().filter(|s| s.busy) {
            if let Some(file) = self.files.get(usize::from(slot.file)) {
                // SAFETY: as in `cancel`.
                unsafe { CancelIoEx(file.file.as_raw_handle() as HANDLE, slot.overlapped.get()) };
            }
        }
        let deadline = Instant::now() + DRAIN;
        let mut sink = CompletionBuf::with_capacity(self.slots.len());
        while self.busy > 0 && Instant::now() < deadline {
            self.drain_port(&mut sink, 100);
            sink.clear();
        }
        if self.busy > 0 {
            // The kernel may still write these `OVERLAPPED`s, so they must stay allocated.
            std::mem::forget(std::mem::take(&mut self.slots));
        }
        // SAFETY: the port is this engine's own, and no helper is left to post to it.
        unsafe { CloseHandle(self.port) };
    }
}
