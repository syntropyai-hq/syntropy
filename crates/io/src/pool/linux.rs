// SPDX-License-Identifier: MIT OR Apache-2.0

//! `PreadPool` on Linux (RFC-0021 §4.2; ADR-0013 item 3).
//!
//! Files are opened `O_DIRECT`. A file falls back to buffered access, and the engine reports
//! [`EngineCaps::direct`](crate::EngineCaps::direct) `false`, when any of these holds:
//! - the file lives on tmpfs or ramfs (`fstatfs`). Since Linux 6.6 tmpfs accepts `O_DIRECT`, but
//!   its pages are the page cache, so no read there is direct; RFC-0021 §4.2 names tmpfs as a
//!   buffered case, and this check keeps it one on every kernel;
//! - `open` with `O_DIRECT` returns `EINVAL`;
//! - the first aligned read, one 4 KiB block at offset 0 into a 4 KiB-aligned block, returns
//!   `EINVAL` (some FUSE file systems, WSL2's drvfs).
//!
//! A buffered file follows every read with `posix_fadvise(POSIX_FADV_DONTNEED)` on its range,
//! so the reads do not fill the page cache.
//!
//! The workspace has no C bindings crate, so the C functions and constants are declared here.
//! They are those of glibc and musl on x86-64 and AArch64, the Linux targets ADR-0020 supports;
//! only `O_DIRECT` differs between the two.

use std::ffi::{c_int, c_void};
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use syntropy_core::resource::EngineKind;

use super::{Backend, Opened, Pool, ProbeBuf, Seg, open_error, transfer_error};
use crate::engine::OpenMode;
use crate::error::{IoError, IoErrorCode};
use crate::request::SyncMode;

mod sys {
    use std::ffi::{c_int, c_void};

    #[cfg(target_arch = "x86_64")]
    pub const O_DIRECT: c_int = 0o40000;
    #[cfg(target_arch = "aarch64")]
    pub const O_DIRECT: c_int = 0o200000;

    pub const POSIX_FADV_DONTNEED: c_int = 4;

    pub const TMPFS_MAGIC: i64 = 0x0102_1994;
    pub const RAMFS_MAGIC: i64 = 0x8584_58f6;

    pub const EINTR: i32 = 4;
    pub const EINVAL: i32 = 22;
    pub const ENOSPC: i32 = 28;
    pub const EDQUOT: i32 = 122;

    #[repr(C)]
    pub struct IoVec {
        pub base: *mut c_void,
        pub len: usize,
    }

    /// `struct statfs`, of which only `f_type`, the first field (`__fsword_t`, a `long`), is
    /// read. The real struct is 120 bytes on both targets; this one is larger, so the kernel
    /// never writes past it.
    #[repr(C)]
    pub struct StatFs {
        pub f_type: i64,
        pub rest: [u64; 31],
    }

    unsafe extern "C" {
        pub fn pread(fd: c_int, buf: *mut c_void, count: usize, offset: i64) -> isize;
        pub fn preadv(fd: c_int, iov: *const IoVec, iovcnt: c_int, offset: i64) -> isize;
        pub fn pwrite(fd: c_int, buf: *const c_void, count: usize, offset: i64) -> isize;
        pub fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
        pub fn fstatfs(fd: c_int, buf: *mut StatFs) -> c_int;
    }
}

const NO_SPACE: [i32; 2] = [sys::ENOSPC, sys::EDQUOT];

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// An open file and whether it is in buffered mode.
#[derive(Debug)]
pub(crate) struct PosixFile {
    file: File,
    buffered: bool,
}

/// The Linux backend.
#[derive(Debug, Default)]
pub(crate) struct Linux;

fn options(mode: OpenMode, create: bool) -> OpenOptions {
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
    o
}

/// Whether `file` lives on a RAM-backed file system, where no read is direct.
fn ram_backed(file: &File) -> bool {
    let mut st = sys::StatFs {
        f_type: 0,
        rest: [0; 31],
    };
    // SAFETY: `st` is a writable buffer larger than the kernel's `struct statfs`, and the fd is
    // open for the duration of the call.
    let rc = unsafe { sys::fstatfs(file.as_raw_fd(), &raw mut st) };
    rc == 0 && (st.f_type == sys::TMPFS_MAGIC || st.f_type == sys::RAMFS_MAGIC)
}

/// The OS error of an aligned direct read of the file's first block, if it fails.
fn probe(file: &File) -> Option<i32> {
    let buf = ProbeBuf::new()?;
    let seg = buf.seg();
    loop {
        // SAFETY: the block is owned by `buf`, valid for `seg.len` bytes and used by nothing
        // else; the fd is open for the call.
        let n = unsafe {
            sys::pread(
                file.as_raw_fd(),
                seg.addr as *mut c_void,
                seg.len as usize,
                0,
            )
        };
        if n >= 0 {
            return None;
        }
        match errno() {
            sys::EINTR => continue,
            os => return Some(os),
        }
    }
}

impl Backend for Linux {
    type Handle = PosixFile;
    const KIND: EngineKind = EngineKind::PreadPool;
    const GATHER_TO: u64 = 0;

    fn open(
        &self,
        path: &Path,
        mode: OpenMode,
        _workers: usize,
    ) -> Result<Opened<PosixFile>, IoError> {
        let refused = match options(mode, true).custom_flags(sys::O_DIRECT).open(path) {
            Ok(file) => {
                let refused = if ram_backed(&file) {
                    Some(IoError::new(IoErrorCode::DirectRefused))
                } else {
                    probe(&file)
                        .filter(|&os| os == sys::EINVAL)
                        .map(|os| IoError::with_os_error(IoErrorCode::DirectRefused, os))
                };
                if refused.is_none() {
                    return Ok(Opened {
                        handles: vec![std::sync::Arc::new(PosixFile {
                            file,
                            buffered: false,
                        })],
                        refused: None,
                    });
                }
                refused
            }
            Err(e) if e.raw_os_error() == Some(sys::EINVAL) => Some(IoError::with_os_error(
                IoErrorCode::DirectRefused,
                sys::EINVAL,
            )),
            Err(e) => return Err(open_error(&e, &NO_SPACE)),
        };
        // Buffered. A `Create` file already exists by now, unless the direct open itself failed.
        let file = options(mode, true)
            .open(path)
            .map_err(|e| open_error(&e, &NO_SPACE))?;
        Ok(Opened {
            handles: vec![std::sync::Arc::new(PosixFile {
                file,
                buffered: true,
            })],
            refused,
        })
    }

    unsafe fn read(&self, h: &PosixFile, offset: u64, segs: &[Seg]) -> Result<u64, IoError> {
        let fd = h.file.as_raw_fd();
        let off = i64::try_from(offset).map_err(|_| IoError::new(IoErrorCode::Misaligned))?;
        let n = loop {
            let n = if let [seg] = segs {
                // SAFETY: the caller guarantees the segment is valid for writes and reserved.
                unsafe { sys::pread(fd, seg.addr as *mut c_void, seg.len as usize, off) }
            } else {
                let mut iov = [const {
                    sys::IoVec {
                        base: std::ptr::null_mut(),
                        len: 0,
                    }
                }; super::MAX_SEGS];
                for (v, s) in iov.iter_mut().zip(segs) {
                    *v = sys::IoVec {
                        base: s.addr as *mut c_void,
                        len: s.len as usize,
                    };
                }
                // SAFETY: as for `pread`, for each segment; `iov` holds `segs.len()` entries.
                unsafe { sys::preadv(fd, iov.as_ptr(), segs.len() as c_int, off) }
            };
            if n >= 0 {
                break n as u64;
            }
            match errno() {
                sys::EINTR => continue,
                os => return Err(transfer_error(os, sys::EINVAL, &NO_SPACE)),
            }
        };
        if h.buffered {
            let len: u64 = segs.iter().map(|s| s.len).sum();
            // Advice only: a failure leaves pages cached, which costs memory, not correctness.
            // SAFETY: plain integers; the fd is open for the call.
            let _ = unsafe { sys::posix_fadvise(fd, off, len as i64, sys::POSIX_FADV_DONTNEED) };
        }
        Ok(n)
    }

    unsafe fn write(&self, h: &PosixFile, offset: u64, seg: Seg) -> Result<u64, IoError> {
        let fd = h.file.as_raw_fd();
        let off = i64::try_from(offset).map_err(|_| IoError::new(IoErrorCode::Misaligned))?;
        loop {
            // SAFETY: the caller guarantees the segment is valid for reads and reserved.
            let n = unsafe { sys::pwrite(fd, seg.addr as *const c_void, seg.len as usize, off) };
            if n >= 0 {
                return Ok(n as u64);
            }
            match errno() {
                sys::EINTR => continue,
                os => return Err(transfer_error(os, sys::EINVAL, &NO_SPACE)),
            }
        }
    }

    fn sync(&self, h: &PosixFile, mode: SyncMode) -> Result<(), IoError> {
        let r = match mode {
            SyncMode::None => Ok(()),
            SyncMode::DataSync => h.file.sync_data(),
            SyncMode::FullSync => h.file.sync_all(),
        };
        r.map_err(|e| transfer_error(e.raw_os_error().unwrap_or(0), sys::EINVAL, &NO_SPACE))
    }
}

/// The universal fallback engine: a pool of blocking `pread`/`pwrite` threads with direct I/O
/// (RFC-0021 §4.2; ADR-0013 item 3). See the [module](crate::pool) for the shape.
pub struct PreadPool {
    pool: Pool<Linux>,
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
            pool: Pool::new(Linux, threads)?,
        })
    }

    /// Why the engine runs in the flagged buffered mode (RFC-0021 §4.2): the
    /// `IO_DIRECT_REFUSED` error of the first file it opened buffered, with the OS error where
    /// one caused it. `None` while every file is direct.
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
