// SPDX-License-Identifier: MIT OR Apache-2.0

//! What both Windows engines share: opening a file, the Win32 error mapping, the registered
//! host regions, and the few `kernel32` functions this crate declares itself.

use std::fs::{File, OpenOptions};
use std::os::windows::fs::OpenOptionsExt;
use std::path::Path;
use std::time::Instant;

use syntropy_core::resource::DioAlign;
use syntropy_core::{Bytes, HostRegion, Nanos};
use windows_sys::Win32::Foundation::{
    ERROR_DISK_FULL, ERROR_HANDLE_DISK_FULL, ERROR_HANDLE_EOF, ERROR_INVALID_PARAMETER,
    ERROR_OPERATION_ABORTED,
};
use windows_sys::Win32::Storage::FileSystem::{
    FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE,
};

use crate::align;
use crate::engine::{BufIndex, OpenMode, Wait};
use crate::error::{IoError, IoErrorCode};
use crate::request::{IoDestKind, IoTicket, ReadCompletion};

/// The largest transfer one call issues: 1 GiB, a multiple of every sector size, and inside the
/// 32-bit lengths of `ReadFile` and IoRing. A longer operation completes short, which the I/O
/// thread resubmits (RFC-0021 §16).
pub(super) const MAX_CALL: u64 = 1 << 30;

/// The `kernel32` functions that windows-sys gates behind features this crate does not enable
/// (`Win32_Security`, `Win32_System_LibraryLoader`).
pub(super) mod sys {
    use std::ffi::c_void;

    /// `LoadLibraryExW`: look only in the system directory.
    pub const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x0000_0800;

    /// A function `GetProcAddress` returned, before it is cast to its real type.
    pub type Proc = unsafe extern "system" fn() -> isize;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn CreateEventW(
            attributes: *const c_void,
            manual_reset: i32,
            initial_state: i32,
            name: *const u16,
        ) -> *mut c_void;
        pub fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        pub fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        pub fn GetProcAddress(module: *mut c_void, name: *const u8) -> Option<Proc>;
    }
}

/// A file one engine opened.
pub(super) struct WinFile {
    pub file: File,
    pub dio: DioAlign,
    /// Opened with `FILE_FLAG_NO_BUFFERING`.
    pub direct: bool,
}

/// Opens `path` for `mode`, unbuffered and overlapped, or overlapped only where the file system
/// refuses unbuffered opens. Both read and write sharing are allowed, so the IOCP engine that
/// takes an IoRing engine's writes can open the same file.
pub(super) fn open(path: &Path, mode: OpenMode) -> Result<WinFile, IoError> {
    let attempt = |flags: u32| {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .custom_flags(flags);
        match mode {
            OpenMode::Read => {}
            OpenMode::ReadWrite => {
                options.write(true);
            }
            OpenMode::Create => {
                options.write(true).create(true).truncate(true);
            }
        }
        options.open(path)
    };
    let (file, direct) = match attempt(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED) {
        Ok(file) => (file, true),
        Err(e) if e.raw_os_error() == Some(ERROR_INVALID_PARAMETER as i32) => {
            (attempt(FILE_FLAG_OVERLAPPED).map_err(open_error)?, false)
        }
        Err(e) => return Err(open_error(e)),
    };
    let dio = align::query(&file, path).dio();
    Ok(WinFile { file, dio, direct })
}

/// A file that cannot be opened: RFC-0021 §16 has no code for it, so it is `IO_EXTENT_BAD` with
/// the OS error, as on the other engines.
pub(super) fn open_error(e: std::io::Error) -> IoError {
    match e.raw_os_error() {
        Some(os) => IoError::with_os_error(IoErrorCode::ExtentBad, os),
        None => IoError::new(IoErrorCode::ExtentBad),
    }
}

/// The result of a transfer that failed with Win32 error `code`. End of file is a short
/// transfer of `bytes`, which the I/O thread handles (RFC-0021 §16).
pub(super) fn failed(code: u32, bytes: u64) -> Result<Bytes, IoError> {
    let kind = match code {
        ERROR_HANDLE_EOF => return Ok(Bytes(bytes)),
        ERROR_INVALID_PARAMETER => IoErrorCode::Misaligned,
        ERROR_DISK_FULL | ERROR_HANDLE_DISK_FULL => IoErrorCode::NoSpace,
        ERROR_OPERATION_ABORTED => IoErrorCode::Cancelled,
        _ => IoErrorCode::ShortRead,
    };
    Err(IoError::with_os_error(kind, code as i32))
}

/// The host regions registered with an engine, in registration order.
#[derive(Default)]
pub(super) struct Regions(Vec<HostRegion>);

impl Regions {
    /// Records `regions` and returns their indexes.
    pub fn add(&mut self, regions: &[HostRegion]) -> Result<Vec<BufIndex>, IoError> {
        let first = self.0.len();
        let end = first + regions.len();
        if u16::try_from(end.saturating_sub(1)).is_err() {
            return Err(IoError::new(IoErrorCode::RegisterFailed));
        }
        self.0.extend_from_slice(regions);
        Ok((first..end)
            .map(|i| BufIndex(u16::try_from(i).unwrap_or(u16::MAX)))
            .collect())
    }

    pub fn get(&self, buf: BufIndex) -> Option<HostRegion> {
        self.0.get(usize::from(buf.0)).copied()
    }

    pub fn all(&self) -> &[HostRegion] {
        &self.0
    }

    /// The address of `len` bytes of a destination, or `None` for a `Fixed` destination outside
    /// the registered regions, whose memory the engine must not touch (`IoEngine::submit`).
    pub fn address(&self, kind: IoDestKind, len: u64) -> Option<u64> {
        match kind {
            IoDestKind::Raw { addr } => Some(addr),
            IoDestKind::Fixed { buf, offset } => {
                let region = self.get(buf)?;
                (offset.checked_add(len)? <= region.len.0).then(|| region.addr + offset)
            }
        }
    }
}

/// How long a `reap` may block, in the milliseconds the Win32 waits take. Shorter waits round
/// down to a poll: the I/O thread passes at most 100 µs (RFC-0021 §3.1), and rounding up would
/// park it for a whole timer tick.
pub(super) fn millis(wait: Wait) -> u32 {
    match wait {
        Wait::None => 0,
        Wait::Until(Nanos(ns)) => u32::try_from(ns / 1_000_000).unwrap_or(u32::MAX - 1),
    }
}

/// A completion for `ticket`, with the time since `issued`.
pub(super) fn completion(
    ticket: IoTicket,
    result: Result<Bytes, IoError>,
    issued: Option<Instant>,
) -> ReadCompletion {
    let latency = issued.map_or(0, |t| {
        u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX)
    });
    ReadCompletion {
        ticket,
        result,
        latency: Nanos(latency),
    }
}

/// `s` as a NUL-terminated UTF-16 string.
pub(super) fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// The larger of each field of the alignments of `files`, zero when there are none.
pub(super) fn widest(files: impl Iterator<Item = DioAlign>) -> DioAlign {
    files.fold(DioAlign { mem: 0, offset: 0 }, |a, b| DioAlign {
        mem: a.mem.max(b.mem),
        offset: a.offset.max(b.offset),
    })
}
