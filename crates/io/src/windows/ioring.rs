// SPDX-License-Identifier: MIT OR Apache-2.0

//! `IoRingEngine` (RFC-0021 §7; ADR-0013 item 4): Windows I/O rings, on Windows 11 and later.
//!
//! - `QueryIoRingCapabilities` gives the highest version and the queue limits; the ring is
//!   created with that version and 256 submission and 512 completion entries, or the limits if
//!   they are lower. `IsIoRingOpSupported` is checked for every operation used.
//! - Files are registered with `BuildIoRingRegisterFileHandles` and registered regions with
//!   `BuildIoRingRegisterBuffers`, both while nothing is in flight, so each re-registration of
//!   the whole table leaves no operation holding an old index. One asked for while operations are
//!   in flight is made by the first `reap` that finds the ring idle; until then operations use raw
//!   handles and addresses. A registration that fails leaves them on raw handles and addresses
//!   for good (`IO_REGISTER_FAILED` continues unregistered).
//! - Reads are `BuildIoRingReadFile`, submitted with `SubmitIoRing` without waiting and reaped
//!   with `PopIoRingCompletion`. Where the ring supports it, a completion event set with
//!   `SetIoRingCompletionEvent` is what `reap` blocks on.
//! - Writes are `BuildIoRingWriteFile`, followed by `BuildIoRingFlushFile` for a write with a
//!   sync mode; the write completes once its flush has. When the ring lacks the write or the
//!   flush operation, writes go to an [`IocpEngine`] this engine owns (one engine for reads, one
//!   for writes), which opens every writable file itself.
//! - Cancellation is `BuildIoRingCancelRequest`.

use std::collections::VecDeque;
use std::os::windows::io::AsRawHandle;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use syntropy_core::resource::{DioAlign, EngineKind};
use syntropy_core::{Bytes, FileIx, HostRegion};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::Storage::FileSystem::{
    FILE_FLUSH_DATA, FILE_FLUSH_DEFAULT, FILE_FLUSH_MODE, FILE_WRITE_FLAGS, FILE_WRITE_FLAGS_NONE,
    HIORING, IORING_BUFFER_INFO, IORING_BUFFER_REF, IORING_BUFFER_REF_0, IORING_CAPABILITIES,
    IORING_CQE, IORING_CREATE_FLAGS, IORING_FEATURE_SET_COMPLETION_EVENT, IORING_HANDLE_REF,
    IORING_HANDLE_REF_0, IORING_OP_CANCEL, IORING_OP_CODE, IORING_OP_FLUSH, IORING_OP_READ,
    IORING_OP_REGISTER_BUFFERS, IORING_OP_REGISTER_FILES, IORING_OP_WRITE, IORING_REF_RAW,
    IORING_REF_REGISTERED, IORING_REGISTERED_BUFFER, IORING_SQE_FLAGS, IORING_VERSION,
};
use windows_sys::Win32::System::Threading::WaitForSingleObject;
use windows_sys::core::{BOOL, HRESULT};

use super::common::{self, MAX_CALL, Regions, WinFile, sys};
use super::iocp::{IocpEngine, IocpOptions};
use crate::batch::{CompletionBuf, SubmitBatch};
use crate::engine::{BufIndex, CancelOutcome, EngineCaps, IoEngine, OpenMode, Wait};
use crate::error::{IoError, IoErrorCode};
use crate::request::{IoOp, IoTicket, ReadCompletion, SyncMode};

/// The ring's submission entries (RFC-0021 §7), or the limit `QueryIoRingCapabilities` reports.
const SUBMISSION_ENTRIES: u32 = 256;
/// The ring's completion entries (RFC-0021 §7), or the limit.
const COMPLETION_ENTRIES: u32 = 512;
/// The size of one registered buffer: `IORING_BUFFER_INFO::Length` is 32 bits, so a larger
/// region is registered in pieces of this size, and an operation that crosses a piece's end uses
/// its raw address.
const PIECE: u64 = 1 << 31;
/// `SubmitIoRing` returns this when the ring is full.
const SUBMISSION_QUEUE_FULL: HRESULT = 0x8046_0002_u32 as HRESULT;
/// `PopIoRingCompletion` returns this when no completion is waiting.
const S_FALSE: HRESULT = 1;
/// `E_NOTIMPL`.
const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as HRESULT;
/// HRESULT_FROM_WIN32's facility.
const FACILITY_WIN32: u32 = 0x8007_0000;

/// The kind of entry a completion's user data names; the low 32 bits are a slot index.
const TAG_OP: usize = 0;
const TAG_FLUSH: usize = 1 << 32;
const TAG_CANCEL: usize = 2 << 32;
const TAG_REGISTER: usize = 3 << 32;
const TAG_MASK: usize = 0xFF << 32;

/// How long dropping the engine, or a registration, waits for the ring.
const DRAIN: Duration = Duration::from_secs(30);

type QueryCaps = unsafe extern "system" fn(*mut IORING_CAPABILITIES) -> HRESULT;
type Create = unsafe extern "system" fn(
    IORING_VERSION,
    IORING_CREATE_FLAGS,
    u32,
    u32,
    *mut HIORING,
) -> HRESULT;
type Close = unsafe extern "system" fn(HIORING) -> HRESULT;
type OpSupported = unsafe extern "system" fn(HIORING, IORING_OP_CODE) -> BOOL;
type RegisterFiles = unsafe extern "system" fn(HIORING, u32, *const HANDLE, usize) -> HRESULT;
type RegisterBuffers =
    unsafe extern "system" fn(HIORING, u32, *const IORING_BUFFER_INFO, usize) -> HRESULT;
type Read = unsafe extern "system" fn(
    HIORING,
    IORING_HANDLE_REF,
    IORING_BUFFER_REF,
    u32,
    u64,
    usize,
    IORING_SQE_FLAGS,
) -> HRESULT;
type Write = unsafe extern "system" fn(
    HIORING,
    IORING_HANDLE_REF,
    IORING_BUFFER_REF,
    u32,
    u64,
    FILE_WRITE_FLAGS,
    usize,
    IORING_SQE_FLAGS,
) -> HRESULT;
type Flush = unsafe extern "system" fn(
    HIORING,
    IORING_HANDLE_REF,
    FILE_FLUSH_MODE,
    usize,
    IORING_SQE_FLAGS,
) -> HRESULT;
type Cancel = unsafe extern "system" fn(HIORING, IORING_HANDLE_REF, usize, usize) -> HRESULT;
type Submit = unsafe extern "system" fn(HIORING, u32, u32, *mut u32) -> HRESULT;
type Pop = unsafe extern "system" fn(HIORING, *mut IORING_CQE) -> HRESULT;
type SetEvent = unsafe extern "system" fn(HIORING, HANDLE) -> HRESULT;

/// The IoRing functions, resolved at run time.
struct Api {
    query: QueryCaps,
    create: Create,
    close: Close,
    op_supported: OpSupported,
    register_files: RegisterFiles,
    register_buffers: RegisterBuffers,
    read: Read,
    cancel: Cancel,
    submit: Submit,
    pop: Pop,
    set_event: SetEvent,
    /// In `kernel32.dll` from Windows 11 22H2.
    write: Option<Write>,
    flush: Option<Flush>,
}

/// The IoRing functions, or `None` on a Windows without them.
fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    // SAFETY: `load` casts each function to the type windows-sys and `ioringapi.h` give it.
    API.get_or_init(|| unsafe { load() }).as_ref()
}

/// Resolves the IoRing functions.
///
/// # Safety
///
/// Each symbol is cast to the signature `ioringapi.h` declares for it; the cast is sound only
/// because these names are those functions.
unsafe fn load() -> Option<Api> {
    let ring_dll = common::wide("api-ms-win-core-ioring-l1-1-0.dll");
    // SAFETY: a NUL-terminated name, no file handle, and a documented flag.
    let ring = unsafe {
        sys::LoadLibraryExW(
            ring_dll.as_ptr(),
            std::ptr::null_mut(),
            sys::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    };
    if ring.is_null() {
        return None;
    }
    let kernel32_dll = common::wide("kernel32.dll");
    // SAFETY: a NUL-terminated name; kernel32 is loaded in every process.
    let kernel32 = unsafe { sys::GetModuleHandleW(kernel32_dll.as_ptr()) };
    macro_rules! resolve {
        ($module:expr, $name:literal, $ty:ty) => {
            symbol($module, concat!($name, "\0").as_bytes()).map(|f| {
                // SAFETY: the caller's contract: `$name` is the function `$ty` describes.
                unsafe { std::mem::transmute::<sys::Proc, $ty>(f) }
            })
        };
    }
    Some(Api {
        query: resolve!(ring, "QueryIoRingCapabilities", QueryCaps)?,
        create: resolve!(ring, "CreateIoRing", Create)?,
        close: resolve!(ring, "CloseIoRing", Close)?,
        op_supported: resolve!(ring, "IsIoRingOpSupported", OpSupported)?,
        register_files: resolve!(ring, "BuildIoRingRegisterFileHandles", RegisterFiles)?,
        register_buffers: resolve!(ring, "BuildIoRingRegisterBuffers", RegisterBuffers)?,
        read: resolve!(ring, "BuildIoRingReadFile", Read)?,
        cancel: resolve!(ring, "BuildIoRingCancelRequest", Cancel)?,
        submit: resolve!(ring, "SubmitIoRing", Submit)?,
        pop: resolve!(ring, "PopIoRingCompletion", Pop)?,
        set_event: resolve!(ring, "SetIoRingCompletionEvent", SetEvent)?,
        write: resolve!(kernel32, "BuildIoRingWriteFile", Write),
        flush: resolve!(kernel32, "BuildIoRingFlushFile", Flush),
    })
}

/// `name`, NUL-terminated, in `module`, or `None` when the module or the name is missing.
fn symbol(module: *mut std::ffi::c_void, name: &[u8]) -> Option<sys::Proc> {
    debug_assert_eq!(name.last(), Some(&0));
    if module.is_null() {
        return None;
    }
    // SAFETY: `module` is a loaded module and `name` is NUL-terminated.
    unsafe { sys::GetProcAddress(module, name.as_ptr()) }
}

/// What this host's IoRing supports: `QueryIoRingCapabilities` and `IsIoRingOpSupported` on a
/// ring made to ask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoRingSupport {
    /// The highest `IORING_VERSION`.
    pub version: i32,
    /// The most submission entries a ring may have.
    pub max_submission: u32,
    /// The most completion entries a ring may have.
    pub max_completion: u32,
    /// `IORING_FEATURE_FLAGS`.
    pub feature_flags: i32,
    /// `IORING_FEATURE_SET_COMPLETION_EVENT`.
    pub completion_event: bool,
    /// The operations, as `IsIoRingOpSupported` reports them.
    pub read: bool,
    /// Writes.
    pub write: bool,
    /// Flushes.
    pub flush: bool,
    /// Cancellation.
    pub cancel: bool,
    /// File registration.
    pub register_files: bool,
    /// Buffer registration.
    pub register_buffers: bool,
}

/// What this host's IoRing supports.
///
/// # Errors
///
/// [`IoErrorCode::EngineUnavailable`] when the host has no IoRing.
pub fn ioring_support() -> Result<IoRingSupport, IoError> {
    let ring = Ring::new()?;
    Ok(ring.support)
}

/// A ring and what it supports.
struct Ring {
    api: &'static Api,
    handle: HIORING,
    support: IoRingSupport,
    /// The submission and completion entries it was created with.
    entries: (u32, u32),
}

impl Ring {
    fn new() -> Result<Self, IoError> {
        let unavailable = |hr: Option<HRESULT>| match hr {
            Some(hr) => IoError::with_os_error(IoErrorCode::EngineUnavailable, hr),
            None => IoError::new(IoErrorCode::EngineUnavailable),
        };
        let api = api().ok_or(unavailable(None))?;
        let mut caps = IORING_CAPABILITIES::default();
        // SAFETY: `caps` is a valid out-pointer.
        let hr = unsafe { (api.query)(&mut caps) };
        if hr < 0 {
            return Err(unavailable(Some(hr)));
        }
        let mut handle: HIORING = std::ptr::null_mut();
        let limit = |want: u32, max: u32| if max == 0 { want } else { want.min(max) };
        let entries = (
            limit(SUBMISSION_ENTRIES, caps.MaxSubmissionQueueSize),
            limit(COMPLETION_ENTRIES, caps.MaxCompletionQueueSize),
        );
        // SAFETY: plain values and a valid out-pointer.
        let hr = unsafe {
            (api.create)(
                caps.MaxVersion,
                IORING_CREATE_FLAGS::default(),
                entries.0,
                entries.1,
                &mut handle,
            )
        };
        if hr < 0 || handle.is_null() {
            return Err(unavailable(Some(hr)));
        }
        // SAFETY: `handle` is the ring just created.
        let op = |code| unsafe { (api.op_supported)(handle, code) } != 0;
        let support = IoRingSupport {
            version: caps.MaxVersion,
            max_submission: caps.MaxSubmissionQueueSize,
            max_completion: caps.MaxCompletionQueueSize,
            feature_flags: caps.FeatureFlags,
            completion_event: caps.FeatureFlags & IORING_FEATURE_SET_COMPLETION_EVENT != 0,
            read: op(IORING_OP_READ),
            write: api.write.is_some() && op(IORING_OP_WRITE),
            flush: api.flush.is_some() && op(IORING_OP_FLUSH),
            cancel: op(IORING_OP_CANCEL),
            register_files: op(IORING_OP_REGISTER_FILES),
            register_buffers: op(IORING_OP_REGISTER_BUFFERS),
        };
        let ring = Ring {
            api,
            handle,
            support,
            entries,
        };
        if !support.read {
            return Err(unavailable(None));
        }
        Ok(ring)
    }

    /// Submits every entry built so far, without waiting.
    fn submit(&self) -> HRESULT {
        let mut submitted = 0;
        // SAFETY: the ring is open, and `submitted` is a valid out-pointer.
        unsafe { (self.api.submit)(self.handle, 0, 0, &mut submitted) }
    }

    /// The next completion, if one is waiting.
    fn pop(&self) -> Option<IORING_CQE> {
        let mut cqe = IORING_CQE::default();
        // SAFETY: the ring is open, and `cqe` is a valid out-pointer.
        let hr = unsafe { (self.api.pop)(self.handle, &mut cqe) };
        (hr != S_FALSE && hr >= 0).then_some(cqe)
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: the ring is this value's own; `IoRingEngine::drop` has waited for every
        // operation on it, or left it open by clearing `handle`.
        unsafe { (self.api.close)(self.handle) };
    }
}

/// How an [`IoRingEngine`] is built.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IoRingOptions {
    /// Send writes to the IOCP engine even where the ring supports them, as a host whose ring
    /// lacks them does. For tests of that path.
    pub writes_via_iocp: bool,
}

/// A file the ring engine opened.
struct RingFile {
    win: WinFile,
    /// The writer's index for the same file, when writes go through IOCP.
    writer: Option<FileIx>,
}

/// One operation in flight. The ring names it by its index in the user data.
#[derive(Clone, Copy)]
struct Slot {
    ticket: IoTicket,
    file: u16,
    len: u64,
    write: bool,
    sync: SyncMode,
    /// Set once the write has completed and its flush is in flight: the bytes it wrote.
    flushing: Option<u64>,
    issued: Option<Instant>,
    busy: bool,
}

const IDLE: Slot = Slot {
    ticket: IoTicket(0),
    file: 0,
    len: 0,
    write: false,
    sync: SyncMode::None,
    flushing: None,
    issued: None,
    busy: false,
};

/// The IoRing engine. See the module documentation.
pub struct IoRingEngine {
    files: Vec<RingFile>,
    /// The registered file table; `files[i]` is entry `i` when `files_registered`.
    handles: Vec<HANDLE>,
    files_registered: bool,
    regions: Regions,
    /// The registered buffer table, and the first entry of each region in it.
    buffers: Vec<IORING_BUFFER_INFO>,
    first_piece: Vec<u32>,
    buffers_registered: bool,
    /// A registration was skipped because operations were in flight; `reap` retries it once the
    /// ring is idle.
    files_pending: bool,
    buffers_pending: bool,
    slots: Vec<Slot>,
    free: Vec<u32>,
    ready: VecDeque<ReadCompletion>,
    busy: u32,
    busy_bytes: u64,
    /// Entries built but not yet taken by a successful `SubmitIoRing`.
    unsubmitted: bool,
    event: Option<HANDLE>,
    writer: Option<IocpEngine>,
    /// Declared last among the ring's users, so it closes after `drop` has drained it.
    ring: Ring,
}

// SAFETY: the ring, event and file handles are process-wide and usable from any thread; the
// registration tables are plain data the engine owns. Moving the engine moves none of what the
// kernel holds.
unsafe impl Send for IoRingEngine {}

impl IoRingEngine {
    /// A ring with the highest version this host supports.
    ///
    /// # Errors
    ///
    /// [`IoErrorCode::EngineUnavailable`] when the host has no IoRing, or the ring cannot read;
    /// the IOCP engine is the fallback.
    pub fn new(options: IoRingOptions) -> Result<Self, IoError> {
        let ring = Ring::new()?;
        let support = ring.support;
        let event = if support.completion_event {
            // SAFETY: no attributes or name; an auto-reset event, initially clear.
            let event = unsafe { sys::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
            // SAFETY: the ring and the event are both open.
            (!event.is_null() && unsafe { (ring.api.set_event)(ring.handle, event) } >= 0)
                .then_some(event)
        } else {
            None
        };
        let writer = if options.writes_via_iocp || !(support.write && support.flush) {
            Some(IocpEngine::new(IocpOptions::default())?)
        } else {
            None
        };
        // Every operation in flight may also have a cancellation's completion queued, so the
        // completion queue holds two per operation.
        let depth = ring.entries.0.min(ring.entries.1 / 2).max(1);
        Ok(IoRingEngine {
            files: Vec::new(),
            handles: Vec::new(),
            files_registered: false,
            regions: Regions::default(),
            buffers: Vec::new(),
            first_piece: Vec::new(),
            buffers_registered: false,
            files_pending: false,
            buffers_pending: false,
            slots: vec![IDLE; depth as usize],
            free: (0..depth).rev().collect(),
            ready: VecDeque::with_capacity(depth as usize),
            busy: 0,
            busy_bytes: 0,
            unsubmitted: false,
            event,
            writer,
            ring,
        })
    }

    /// What the ring supports.
    #[must_use]
    pub fn support(&self) -> IoRingSupport {
        self.ring.support
    }

    /// Whether writes go to the IOCP engine.
    #[must_use]
    pub fn writes_via_iocp(&self) -> bool {
        self.writer.is_some()
    }

    /// Builds entry `build`, submits it, and waits until its completion, tagged `TAG_REGISTER`,
    /// arrives. Only while nothing is in flight.
    fn register(&mut self, build: impl FnOnce(&Ring) -> HRESULT) -> bool {
        debug_assert_eq!(self.busy, 0);
        if build(&self.ring) < 0 {
            return false;
        }
        if self.ring.submit() < 0 {
            self.unsubmitted = true;
        }
        let deadline = Instant::now() + DRAIN;
        while Instant::now() < deadline {
            while let Some(cqe) = self.ring.pop() {
                if cqe.UserData & TAG_MASK == TAG_REGISTER {
                    return cqe.ResultCode >= 0;
                }
            }
            if self.unsubmitted && self.ring.submit() >= 0 {
                self.unsubmitted = false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        false
    }

    /// Registers every open file, when nothing is in flight. Otherwise, or if the ring refuses,
    /// operations use raw handles.
    fn register_files(&mut self) {
        self.files_registered = false;
        self.files_pending = false;
        if !self.ring.support.register_files {
            return;
        }
        if self.busy > 0 {
            self.files_pending = true;
            return;
        }
        self.handles = self
            .files
            .iter()
            .map(|f| f.win.file.as_raw_handle() as HANDLE)
            .collect();
        let (n, table) = (self.handles.len() as u32, self.handles.as_ptr());
        // SAFETY: `table` holds `n` open handles and stays allocated, unchanged, until the
        // registration has completed: `register` waits for it.
        self.files_registered = self.register(|ring| unsafe {
            (ring.api.register_files)(ring.handle, n, table, TAG_REGISTER)
        });
    }

    /// Makes a registration that was skipped while operations were in flight, once none is.
    fn retry_registration(&mut self) {
        if self.busy > 0 {
            return;
        }
        if self.files_pending {
            self.register_files();
        }
        if self.buffers_pending {
            self.register_buffers_with_ring();
        }
    }

    /// Registers every region, in pieces of at most [`PIECE`] bytes, when nothing is in flight.
    fn register_buffers_with_ring(&mut self) {
        self.buffers_registered = false;
        self.buffers_pending = false;
        if !self.ring.support.register_buffers {
            return;
        }
        if self.busy > 0 {
            self.buffers_pending = true;
            return;
        }
        self.buffers.clear();
        self.first_piece.clear();
        for region in self.regions.all() {
            self.first_piece.push(self.buffers.len() as u32);
            let mut at = 0;
            while at < region.len.0 {
                let len = (region.len.0 - at).min(PIECE);
                self.buffers.push(IORING_BUFFER_INFO {
                    Address: (region.addr + at) as *mut std::ffi::c_void,
                    Length: len as u32,
                });
                at += len;
            }
        }
        let (n, table) = (self.buffers.len() as u32, self.buffers.as_ptr());
        // SAFETY: `table` holds `n` regions the caller of `register_buffers` vouched for until
        // the engine drops, and stays allocated until the registration completes.
        self.buffers_registered = self.register(|ring| unsafe {
            (ring.api.register_buffers)(ring.handle, n, table, TAG_REGISTER)
        });
    }

    fn file_ref(&self, file: u16) -> Option<IORING_HANDLE_REF> {
        let f = self.files.get(usize::from(file))?;
        Some(if self.files_registered {
            IORING_HANDLE_REF {
                Kind: IORING_REF_REGISTERED,
                Handle: IORING_HANDLE_REF_0 {
                    Index: u32::from(file),
                },
            }
        } else {
            IORING_HANDLE_REF {
                Kind: IORING_REF_RAW,
                Handle: IORING_HANDLE_REF_0 {
                    Handle: f.win.file.as_raw_handle() as HANDLE,
                },
            }
        })
    }

    /// The buffer reference for `len` bytes of a destination, or `None` for a `Fixed`
    /// destination outside the registered regions.
    fn buffer_ref(&self, op: &IoOp, len: u64) -> Option<IORING_BUFFER_REF> {
        let raw = |addr: u64| IORING_BUFFER_REF {
            Kind: IORING_REF_RAW,
            Buffer: IORING_BUFFER_REF_0 {
                Address: addr as *mut std::ffi::c_void,
            },
        };
        let kind = op.dest().kind();
        let addr = self.regions.address(kind, len)?;
        if let crate::request::IoDestKind::Fixed { buf, offset } = kind
            && self.buffers_registered
            && len > 0
        {
            let piece = offset / PIECE;
            if (offset + len - 1) / PIECE == piece {
                let first = self.first_piece[usize::from(buf.0)];
                return Some(IORING_BUFFER_REF {
                    Kind: IORING_REF_REGISTERED,
                    Buffer: IORING_BUFFER_REF_0 {
                        IndexAndOffset: IORING_REGISTERED_BUFFER {
                            BufferIndex: first + piece as u32,
                            Offset: (offset % PIECE) as u32,
                        },
                    },
                });
            }
        }
        Some(raw(addr))
    }

    fn refuse(&mut self, ticket: IoTicket, error: IoError, issued: Instant) {
        self.ready
            .push_back(common::completion(ticket, Err(error), Some(issued)));
    }

    fn complete(&mut self, ix: usize, result: Result<Bytes, IoError>) -> ReadCompletion {
        let slot = self.slots[ix];
        self.slots[ix] = IDLE;
        self.busy -= 1;
        self.busy_bytes = self.busy_bytes.saturating_sub(slot.len);
        self.free.push(ix as u32);
        common::completion(slot.ticket, result, slot.issued)
    }

    /// Submits what is built; a failure leaves it for the next call.
    fn flush_queue(&mut self) {
        self.unsubmitted = self.ring.submit() < 0;
    }

    /// Takes completions from the ring until `out` is full or none is waiting. A write with a
    /// sync mode gets its flush here, and completes when the flush does.
    fn pop_ring(&mut self, out: &mut CompletionBuf, dropping: bool) -> u32 {
        let mut n = 0;
        while out.remaining() > 0 {
            let Some(cqe) = self.ring.pop() else {
                break;
            };
            let ix = cqe.UserData & 0xFFFF_FFFF;
            let tag = cqe.UserData & TAG_MASK;
            if (tag != TAG_OP && tag != TAG_FLUSH) || ix >= self.slots.len() || !self.slots[ix].busy
            {
                continue;
            }
            let result = result_of(&cqe);
            let slot = self.slots[ix];
            let result = match (tag, slot.flushing) {
                (TAG_OP, _) if slot.write && slot.sync != SyncMode::None && !dropping => {
                    if let Ok(Bytes(bytes)) = result {
                        if self.start_flush(ix, bytes) {
                            continue;
                        }
                        Err(IoError::new(IoErrorCode::ShortRead))
                    } else {
                        result
                    }
                }
                (TAG_FLUSH, Some(bytes)) => result.map(|_| Bytes(bytes)),
                _ => result,
            };
            let c = self.complete(ix, result);
            if out.push(c).is_ok() {
                n += 1;
            }
        }
        n
    }

    /// Builds and submits the flush that follows slot `ix`'s write of `bytes`.
    fn start_flush(&mut self, ix: usize, bytes: u64) -> bool {
        let slot = self.slots[ix];
        let (Some(flush), Some(file)) = (self.ring.api.flush, self.file_ref(slot.file)) else {
            return false;
        };
        let mode = match slot.sync {
            SyncMode::DataSync => FILE_FLUSH_DATA,
            _ => FILE_FLUSH_DEFAULT,
        };
        // SAFETY: the ring is open and `file` names one of this engine's open files.
        let hr = unsafe { flush(self.ring.handle, file, mode, ix | TAG_FLUSH, 0) };
        if hr < 0 {
            return false;
        }
        self.slots[ix].flushing = Some(bytes);
        self.flush_queue();
        true
    }
}

/// A ring completion's result: the bytes moved, or the error. End of file is a short read.
fn result_of(cqe: &IORING_CQE) -> Result<Bytes, IoError> {
    let hr = cqe.ResultCode;
    let bytes = cqe.Information as u64;
    if hr >= 0 {
        return Ok(Bytes(bytes));
    }
    let hr = hr as u32;
    if hr & 0xFFFF_0000 == FACILITY_WIN32 {
        return common::failed(hr & 0xFFFF, bytes);
    }
    Err(IoError::with_os_error(IoErrorCode::ShortRead, hr as i32))
}

impl IoEngine for IoRingEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::IoRing
    }

    fn caps(&self) -> EngineCaps {
        EngineCaps {
            fixed_buffers: self.buffers_registered,
            fixed_files: self.files_registered,
            writes: true,
            cancel: self.ring.support.cancel,
            max_depth: self.slots.len() as u32,
            direct: self.files.iter().all(|f| f.win.direct),
            dio: common::widest(self.files.iter().map(|f| f.win.dio)),
        }
    }

    fn open(&mut self, path: &Path, mode: OpenMode) -> Result<FileIx, IoError> {
        let ix = u16::try_from(self.files.len())
            .map_err(|_| IoError::new(IoErrorCode::RegisterFailed))?;
        let win = common::open(path, mode)?;
        // The writer opens the file the ring has just opened (or created), so it never
        // truncates it again.
        let writer = match (&mut self.writer, mode) {
            (Some(writer), OpenMode::ReadWrite | OpenMode::Create) => {
                Some(writer.open(path, OpenMode::ReadWrite)?)
            }
            _ => None,
        };
        self.files.push(RingFile { win, writer });
        self.register_files();
        Ok(FileIx(ix))
    }

    fn file_dio(&self, file: FileIx) -> Option<DioAlign> {
        self.files.get(usize::from(file.0)).map(|f| f.win.dio)
    }

    unsafe fn register_buffers(
        &mut self,
        regions: &[HostRegion],
    ) -> Result<Vec<BufIndex>, IoError> {
        let indexes = self.regions.add(regions)?;
        if let Some(writer) = self.writer.as_mut() {
            // SAFETY: the caller's contract, passed on; the writer is dropped with this engine.
            // It numbers regions in the same order, so the indexes agree.
            unsafe { writer.register_buffers(regions) }?;
        }
        self.register_buffers_with_ring();
        Ok(indexes)
    }

    fn submit(&mut self, batch: &mut SubmitBatch) -> Result<u32, IoError> {
        let mut accepted = 0;
        let mut built = false;
        for op in batch.pending() {
            let issued = Instant::now();
            let extent = op.extent();
            let write = matches!(op, IoOp::Write(_));
            if write && let Some(writer) = self.writer.as_mut() {
                match self
                    .files
                    .get(usize::from(extent.file.0))
                    .and_then(|f| f.writer)
                {
                    Some(file) => {
                        if !writer.accept(op, file) {
                            break;
                        }
                    }
                    // A write to a file opened for reading only.
                    None => self.refuse(op.ticket(), IoError::new(IoErrorCode::ExtentBad), issued),
                }
                accepted += 1;
                continue;
            }
            let Some(&ix) = self.free.last() else {
                break;
            };
            let len = extent.len.0.min(MAX_CALL);
            let Some(file) = self.file_ref(extent.file.0) else {
                self.refuse(op.ticket(), IoError::new(IoErrorCode::ExtentBad), issued);
                accepted += 1;
                continue;
            };
            let Some(buffer) = self.buffer_ref(op, len) else {
                self.refuse(op.ticket(), IoError::new(IoErrorCode::Misaligned), issued);
                accepted += 1;
                continue;
            };
            let user = ix as usize | TAG_OP;
            let ring = self.ring.handle;
            // SAFETY: the ring is open; `file` names an open file of this engine; `buffer` names
            // memory the operation's destination contract gives the engine until its completion
            // is reaped, checked against the registered regions for a `Fixed` destination.
            let hr = unsafe {
                match (op, self.ring.api.write) {
                    (IoOp::Write(_), Some(write)) => write(
                        ring,
                        file,
                        buffer,
                        len as u32,
                        extent.offset,
                        FILE_WRITE_FLAGS_NONE,
                        user,
                        0,
                    ),
                    // No writer means the ring writes, so `write` was resolved.
                    (IoOp::Write(_), None) => E_NOTIMPL,
                    (IoOp::Read(_), _) => {
                        (self.ring.api.read)(ring, file, buffer, len as u32, extent.offset, user, 0)
                    }
                }
            };
            if hr == SUBMISSION_QUEUE_FULL {
                break;
            }
            accepted += 1;
            if hr < 0 {
                let error = IoError::with_os_error(IoErrorCode::ShortRead, hr);
                self.refuse(op.ticket(), error, issued);
                continue;
            }
            self.free.pop();
            let (write, sync) = match op {
                IoOp::Write(w) => (true, w.sync),
                IoOp::Read(_) => (false, SyncMode::None),
            };
            self.slots[ix as usize] = Slot {
                ticket: op.ticket(),
                file: extent.file.0,
                len,
                write,
                sync,
                flushing: None,
                issued: Some(issued),
                busy: true,
            };
            self.busy += 1;
            self.busy_bytes += len;
            built = true;
        }
        batch.consume(accepted);
        if built || self.unsubmitted {
            self.flush_queue();
        }
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
        if self.unsubmitted {
            self.flush_queue();
        }
        n += self.pop_ring(out, false);
        self.retry_registration();
        if let Some(writer) = self.writer.as_mut() {
            n += writer.reap(out, Wait::None);
        }
        if n > 0 || out.remaining() == 0 {
            return n;
        }
        let ms = common::millis(wait);
        if self.busy > 0 {
            if let Some(event) = self.event
                && ms > 0
            {
                // SAFETY: the event is open; the ring signals it when a completion arrives, and
                // an auto-reset event set since the last pop is still set, so none is missed.
                unsafe { WaitForSingleObject(event, ms) };
                n += self.pop_ring(out, false);
                self.retry_registration();
            }
        } else if let Some(writer) = self.writer.as_mut() {
            n += writer.reap(out, wait);
        }
        n
    }

    fn cancel(&mut self, ticket: IoTicket) -> CancelOutcome {
        if let Some(ix) = self.slots.iter().position(|s| s.busy && s.ticket == ticket) {
            let slot = self.slots[ix];
            if self.ring.support.cancel
                && let Some(file) = self.file_ref(slot.file)
            {
                let target = ix
                    | if slot.flushing.is_some() {
                        TAG_FLUSH
                    } else {
                        TAG_OP
                    };
                // SAFETY: the ring is open and `file` names the operation's open file.
                let hr = unsafe {
                    (self.ring.api.cancel)(self.ring.handle, file, target, ix | TAG_CANCEL)
                };
                if hr >= 0 {
                    self.flush_queue();
                }
            }
            return CancelOutcome::TooLate;
        }
        match self.writer.as_mut() {
            Some(writer) => writer.cancel(ticket),
            None => CancelOutcome::NotFound,
        }
    }

    fn in_flight(&self) -> (u32, Bytes) {
        let (writer_ops, writer_bytes) = self
            .writer
            .as_ref()
            .map_or((0, Bytes(0)), IoEngine::in_flight);
        (
            self.busy + self.ready.len() as u32 + writer_ops,
            Bytes(self.busy_bytes + writer_bytes.0),
        )
    }
}

impl Drop for IoRingEngine {
    fn drop(&mut self) {
        let busy: Vec<usize> = (0..self.slots.len())
            .filter(|&ix| self.slots[ix].busy)
            .collect();
        for ix in busy {
            if let (true, Some(file)) =
                (self.ring.support.cancel, self.file_ref(self.slots[ix].file))
            {
                let target = ix
                    | if self.slots[ix].flushing.is_some() {
                        TAG_FLUSH
                    } else {
                        TAG_OP
                    };
                // SAFETY: as in `cancel`.
                unsafe { (self.ring.api.cancel)(self.ring.handle, file, target, ix | TAG_CANCEL) };
            }
        }
        self.flush_queue();
        let deadline = Instant::now() + DRAIN;
        let mut sink = CompletionBuf::with_capacity(self.slots.len().max(1));
        while self.busy > 0 && Instant::now() < deadline {
            if self.unsubmitted {
                self.flush_queue();
            }
            if self.pop_ring(&mut sink, true) == 0 {
                std::thread::sleep(Duration::from_millis(1));
            }
            sink.clear();
        }
        if self.busy > 0 {
            // The kernel may still use the ring's operations and their memory: keep the ring
            // open and its tables allocated rather than free them under it.
            std::mem::forget(std::mem::take(&mut self.handles));
            std::mem::forget(std::mem::take(&mut self.buffers));
            self.ring.handle = std::ptr::null_mut();
        }
        if let Some(event) = self.event {
            // SAFETY: the event is this engine's own.
            unsafe { CloseHandle(event) };
        }
    }
}
