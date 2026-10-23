// SPDX-License-Identifier: MIT OR Apache-2.0

//! `PreadPool` on Windows (RFC-0021 §4.2, §7; ADR-0013 items 3 and 4).
//!
//! Windows' own engines, IoRing and IOCP, are TIER-E5-T08; this pool is the portable fallback.
//! Files are opened with `FILE_FLAG_NO_BUFFERING`. Synchronous I/O on one handle is serialised
//! by the I/O manager, so each worker opens the file itself and reads through its own handle
//! with a positional `ReadFile` (an `OVERLAPPED` carrying the offset). A file whose unbuffered
//! open, or whose first aligned read, fails with `ERROR_INVALID_PARAMETER` is opened buffered,
//! and the engine reports [`EngineCaps::direct`](crate::EngineCaps::direct) `false`. Windows has
//! no per-range cache drop, so buffered mode is only flagged.
//!
//! The workspace has no C bindings crate, so the two `kernel32` functions are declared here.

use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::io::AsRawHandle;
use std::path::Path;
use std::sync::Arc;

use syntropy_core::resource::EngineKind;

use super::{Backend, Opened, Pool, ProbeBuf, Seg, open_error, transfer_error};
use crate::align;
use crate::engine::OpenMode;
use crate::error::{IoError, IoErrorCode};
use crate::request::SyncMode;

mod sys {
    use std::ffi::c_void;

    pub const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;

    pub const ERROR_HANDLE_EOF: i32 = 38;
    pub const ERROR_HANDLE_DISK_FULL: i32 = 39;
    pub const ERROR_INVALID_PARAMETER: i32 = 87;
    pub const ERROR_DISK_FULL: i32 = 112;

    #[repr(C)]
    pub struct Overlapped {
        pub internal: usize,
        pub internal_high: usize,
        pub offset: u32,
        pub offset_high: u32,
        pub event: *mut c_void,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn ReadFile(
            file: *mut c_void,
            buf: *mut c_void,
            len: u32,
            read: *mut u32,
            overlapped: *mut Overlapped,
        ) -> i32;
        pub fn WriteFile(
            file: *mut c_void,
            buf: *const c_void,
            len: u32,
            written: *mut u32,
            overlapped: *mut Overlapped,
        ) -> i32;
    }
}

const NO_SPACE: [i32; 2] = [sys::ERROR_DISK_FULL, sys::ERROR_HANDLE_DISK_FULL];

/// The largest transfer one call issues: 1 GiB, a multiple of every sector size. A longer
/// operation returns short, which the I/O thread handles (RFC-0021 §16).
const MAX_CALL: u64 = 1 << 30;

fn last_error() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

fn overlapped(offset: u64) -> sys::Overlapped {
    sys::Overlapped {
        internal: 0,
        internal_high: 0,
        offset: offset as u32,
        offset_high: (offset >> 32) as u32,
        event: std::ptr::null_mut(),
    }
}

/// One positional read into `seg`: the bytes read, 0 at end of file.
///
/// # Safety
///
/// `seg` is valid for writes and reserved for the call.
unsafe fn read_at(file: &File, offset: u64, seg: Seg) -> Result<u64, i32> {
    let mut ov = overlapped(offset);
    let mut got = 0u32;
    let len = seg.len.min(MAX_CALL) as u32;
    // SAFETY: the caller guarantees `seg`; `ov` and `got` outlive the synchronous call, and the
    // handle is open for it.
    let ok = unsafe {
        sys::ReadFile(
            file.as_raw_handle(),
            seg.addr as *mut c_void,
            len,
            &raw mut got,
            &raw mut ov,
        )
    };
    if ok != 0 {
        return Ok(u64::from(got));
    }
    match last_error() {
        sys::ERROR_HANDLE_EOF => Ok(0),
        e => Err(e),
    }
}

/// An open file, one handle per worker.
#[derive(Debug)]
pub(crate) struct WinFile {
    file: File,
}

/// The Windows backend.
#[derive(Debug, Default)]
pub(crate) struct Windows;

fn options(mode: OpenMode, create: bool, unbuffered: bool) -> OpenOptions {
    let mut o = OpenOptions::new();
    o.read(true);
    match mode {
        OpenMode::Read => {}
        OpenMode::ReadWrite => {
            o.write(true);
        }
        OpenMode::Create => {
            o.write(true);
            if create {
                o.create(true).truncate(true);
            }
        }
    }
    if unbuffered {
        o.custom_flags(sys::FILE_FLAG_NO_BUFFERING);
    }
    o
}

impl Backend for Windows {
    type Handle = WinFile;
    const KIND: EngineKind = EngineKind::PreadPool;
    const GATHER_TO: u64 = 0;

    fn open(
        &self,
        path: &Path,
        mode: OpenMode,
        workers: usize,
    ) -> Result<Opened<WinFile>, IoError> {
        let refused = match options(mode, true, true).open(path) {
            Ok(file) => {
                let probe = ProbeBuf::new().and_then(|buf| {
                    // SAFETY: the block is owned by `buf` and used by nothing else.
                    unsafe { read_at(&file, 0, buf.seg()) }.err()
                });
                match probe {
                    Some(os) if os == sys::ERROR_INVALID_PARAMETER => {
                        Some(IoError::with_os_error(IoErrorCode::DirectRefused, os))
                    }
                    _ => {
                        let dio = align::query(&file, path).dio();
                        let mut handles = Vec::with_capacity(workers);
                        handles.push(Arc::new(WinFile { file }));
                        for _ in 1..workers {
                            let file = options(mode, false, true)
                                .open(path)
                                .map_err(|e| open_error(&e, &NO_SPACE))?;
                            handles.push(Arc::new(WinFile { file }));
                        }
                        return Ok(Opened {
                            handles,
                            refused: None,
                            dio,
                        });
                    }
                }
            }
            Err(e) if e.raw_os_error() == Some(sys::ERROR_INVALID_PARAMETER) => Some(
                IoError::with_os_error(IoErrorCode::DirectRefused, sys::ERROR_INVALID_PARAMETER),
            ),
            Err(e) => return Err(open_error(&e, &NO_SPACE)),
        };
        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let file = options(mode, w == 0, false)
                .open(path)
                .map_err(|e| open_error(&e, &NO_SPACE))?;
            handles.push(Arc::new(WinFile { file }));
        }
        let dio = align::query(&handles[0].file, path).dio();
        Ok(Opened {
            handles,
            refused,
            dio,
        })
    }

    unsafe fn read(&self, h: &WinFile, offset: u64, segs: &[Seg]) -> Result<u64, IoError> {
        let mut total = 0;
        for seg in segs {
            // SAFETY: the caller guarantees each segment is valid for writes and reserved.
            let n = unsafe { read_at(&h.file, offset + total, *seg) }
                .map_err(|os| transfer_error(os, sys::ERROR_INVALID_PARAMETER, &NO_SPACE))?;
            total += n;
            if n < seg.len {
                break;
            }
        }
        Ok(total)
    }

    unsafe fn write(&self, h: &WinFile, offset: u64, seg: Seg) -> Result<u64, IoError> {
        let mut ov = overlapped(offset);
        let mut put = 0u32;
        let len = seg.len.min(MAX_CALL) as u32;
        // SAFETY: the caller guarantees `seg` is valid for reads and reserved; `ov` and `put`
        // outlive the synchronous call, and the handle is open for it.
        let ok = unsafe {
            sys::WriteFile(
                h.file.as_raw_handle(),
                seg.addr as *const c_void,
                len,
                &raw mut put,
                &raw mut ov,
            )
        };
        if ok != 0 {
            Ok(u64::from(put))
        } else {
            Err(transfer_error(
                last_error(),
                sys::ERROR_INVALID_PARAMETER,
                &NO_SPACE,
            ))
        }
    }

    fn sync(&self, h: &WinFile, mode: SyncMode) -> Result<(), IoError> {
        let r = match mode {
            SyncMode::None => Ok(()),
            // `FlushFileBuffers` for both (RFC-0021 §2).
            SyncMode::DataSync | SyncMode::FullSync => h.file.sync_all(),
        };
        r.map_err(|e| {
            transfer_error(
                e.raw_os_error().unwrap_or(0),
                sys::ERROR_INVALID_PARAMETER,
                &NO_SPACE,
            )
        })
    }
}

/// The portable fallback engine: a pool of blocking positional-I/O threads with unbuffered
/// handles (RFC-0021 §4.2, §7). See the [module](crate::pool) for the shape.
pub struct PreadPool {
    pool: Pool<Windows>,
}

impl PreadPool {
    /// Starts a pool of `threads` workers; [`POOL_THREADS_DEFAULT`](super::POOL_THREADS_DEFAULT)
    /// is `io.pool_threads`' default. The thread count is the engine's depth.
    ///
    /// # Errors
    ///
    /// `IO_ENGINE_UNAVAILABLE` if a worker thread cannot be started.
    ///
    /// # Panics
    ///
    /// If `threads` is 0 or above [`POOL_THREADS_MAX`](super::POOL_THREADS_MAX).
    pub fn new(threads: usize) -> Result<Self, IoError> {
        Ok(Self {
            pool: Pool::new(Windows, threads)?,
        })
    }

    /// Why the engine runs in the flagged buffered mode (RFC-0021 §4.2), or `None` while every
    /// file is unbuffered.
    #[must_use]
    pub fn buffered_reason(&self) -> Option<IoError> {
        self.pool.buffered_reason()
    }
}

impl std::fmt::Debug for PreadPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreadPool")
            .field("buffered_reason", &self.buffered_reason())
            .finish_non_exhaustive()
    }
}

super::forward_engine!(PreadPool);
