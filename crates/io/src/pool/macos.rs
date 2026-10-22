// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MacNoCache` (RFC-0021 §5; ADR-0013 item 5).
//!
//! Each file is opened, then `fcntl(F_NOCACHE, 1)` is set on it before the engine hands it to
//! any worker, so before the first read: `F_NOCACHE` does not purge pages already cached
//! (ADR-0013 context). `fcntl(F_RDAHEAD, 0)` then turns read-ahead off. A pool of
//! `io.pool_threads_macos` (8) threads issues `pread`, or `preadv` for a gathered read: reads in
//! a batch that continue one another in the file are gathered into one vectored read until it
//! holds at least 32 KiB ([`MIN_REQUEST`]), so small rows reach the drive as requests of at
//! least that size where they are contiguous (RFC-0021 §5).
//!
//! If `F_NOCACHE` is refused, the file stays open buffered and the engine reports
//! [`EngineCaps::direct`](crate::EngineCaps::direct) `false`, as RFC-0021 §4.2 does for a refused
//! `O_DIRECT`. A refused `F_RDAHEAD` changes only speed and is ignored.
//!
//! The system calls go through the crate-private `MacSys` trait, so a test can record their
//! order against the real calls; the engine uses `RealSys`. The workspace has no C bindings
//! crate, so the C functions and constants are declared here.

use std::ffi::{c_int, c_void};
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;

use syntropy_core::resource::EngineKind;

use super::{Backend, MAX_SEGS, Opened, Pool, Seg, open_error, transfer_error};
use crate::engine::OpenMode;
use crate::error::{IoError, IoErrorCode};
use crate::request::SyncMode;

/// The smallest request `MacNoCache` aims to issue: 32 KiB \[target\] (RFC-0021 §5;
/// ADR-0013 item 5).
pub const MIN_REQUEST: u64 = 32 << 10;

pub(crate) mod sys {
    use std::ffi::{c_int, c_void};

    pub const F_RDAHEAD: c_int = 45;
    pub const F_NOCACHE: c_int = 48;

    pub const EINTR: i32 = 4;
    pub const EINVAL: i32 = 22;
    pub const ENOSPC: i32 = 28;
    pub const EDQUOT: i32 = 69;

    #[repr(C)]
    pub struct IoVec {
        pub base: *mut c_void,
        pub len: usize,
    }

    unsafe extern "C" {
        pub fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
        pub fn pread(fd: c_int, buf: *mut c_void, count: usize, offset: i64) -> isize;
        pub fn preadv(fd: c_int, iov: *const IoVec, iovcnt: c_int, offset: i64) -> isize;
        pub fn pwrite(fd: c_int, buf: *const c_void, count: usize, offset: i64) -> isize;
    }
}

const NO_SPACE: [i32; 2] = [sys::ENOSPC, sys::EDQUOT];

/// The system calls `MacNoCache` makes on an open file.
pub(crate) trait MacSys: Send + Sync + 'static {
    /// `fcntl(fd, cmd, arg)`; the OS error on failure.
    fn fcntl(&self, fd: c_int, cmd: c_int, arg: c_int) -> Result<(), i32>;

    /// `pread` for one segment, `preadv` for several: the bytes read, or the OS error.
    ///
    /// # Safety
    ///
    /// As for [`Backend::read`].
    unsafe fn read(&self, fd: c_int, offset: i64, segs: &[Seg]) -> Result<u64, i32>;

    /// `pwrite`: the bytes written, or the OS error.
    ///
    /// # Safety
    ///
    /// As for [`Backend::write`].
    unsafe fn write(&self, fd: c_int, offset: i64, seg: Seg) -> Result<u64, i32>;
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// The real system calls.
#[derive(Debug, Default)]
pub(crate) struct RealSys;

impl MacSys for RealSys {
    fn fcntl(&self, fd: c_int, cmd: c_int, arg: c_int) -> Result<(), i32> {
        // SAFETY: `F_NOCACHE` and `F_RDAHEAD` take an integer argument and touch no user
        // memory; the fd is the caller's open file.
        let rc = unsafe { sys::fcntl(fd, cmd, arg) };
        if rc == -1 { Err(errno()) } else { Ok(()) }
    }

    unsafe fn read(&self, fd: c_int, offset: i64, segs: &[Seg]) -> Result<u64, i32> {
        loop {
            let n = if let [seg] = segs {
                // SAFETY: the caller guarantees the segment is valid for writes and reserved.
                unsafe { sys::pread(fd, seg.addr as *mut c_void, seg.len as usize, offset) }
            } else {
                let mut iov = [const {
                    sys::IoVec {
                        base: std::ptr::null_mut(),
                        len: 0,
                    }
                }; MAX_SEGS];
                for (v, s) in iov.iter_mut().zip(segs) {
                    *v = sys::IoVec {
                        base: s.addr as *mut c_void,
                        len: s.len as usize,
                    };
                }
                // SAFETY: as for `pread`, for each segment; `iov` holds `segs.len()` entries.
                unsafe { sys::preadv(fd, iov.as_ptr(), segs.len() as c_int, offset) }
            };
            if n >= 0 {
                return Ok(n as u64);
            }
            match errno() {
                sys::EINTR => continue,
                os => return Err(os),
            }
        }
    }

    unsafe fn write(&self, fd: c_int, offset: i64, seg: Seg) -> Result<u64, i32> {
        loop {
            // SAFETY: the caller guarantees the segment is valid for reads and reserved.
            let n = unsafe { sys::pwrite(fd, seg.addr as *const c_void, seg.len as usize, offset) };
            if n >= 0 {
                return Ok(n as u64);
            }
            match errno() {
                sys::EINTR => continue,
                os => return Err(os),
            }
        }
    }
}

/// The macOS backend over a set of system calls.
#[derive(Debug, Default)]
pub(crate) struct Mac<S> {
    pub(crate) sys: S,
}

impl<S: MacSys> Backend for Mac<S> {
    type Handle = File;
    const KIND: EngineKind = EngineKind::MacNoCache;
    const GATHER_TO: u64 = MIN_REQUEST;

    fn open(&self, path: &Path, mode: OpenMode, _workers: usize) -> Result<Opened<File>, IoError> {
        let mut o = OpenOptions::new();
        o.read(true);
        match mode {
            OpenMode::Read => {}
            OpenMode::ReadWrite => {
                o.write(true);
            }
            OpenMode::Create => {
                o.write(true).create(true).truncate(true);
            }
        }
        let file = o.open(path).map_err(|e| open_error(&e, &NO_SPACE))?;
        let fd = file.as_raw_fd();
        // Before any worker has the file, so before its first read.
        let refused = self
            .sys
            .fcntl(fd, sys::F_NOCACHE, 1)
            .err()
            .map(|os| IoError::with_os_error(IoErrorCode::DirectRefused, os));
        let _ = self.sys.fcntl(fd, sys::F_RDAHEAD, 0);
        Ok(Opened {
            handles: vec![Arc::new(file)],
            refused,
        })
    }

    unsafe fn read(&self, h: &File, offset: u64, segs: &[Seg]) -> Result<u64, IoError> {
        let off = i64::try_from(offset).map_err(|_| IoError::new(IoErrorCode::Misaligned))?;
        // SAFETY: the caller's guarantee, passed on unchanged.
        unsafe { self.sys.read(h.as_raw_fd(), off, segs) }
            .map_err(|os| transfer_error(os, sys::EINVAL, &NO_SPACE))
    }

    unsafe fn write(&self, h: &File, offset: u64, seg: Seg) -> Result<u64, IoError> {
        let off = i64::try_from(offset).map_err(|_| IoError::new(IoErrorCode::Misaligned))?;
        // SAFETY: the caller's guarantee, passed on unchanged.
        unsafe { self.sys.write(h.as_raw_fd(), off, seg) }
            .map_err(|os| transfer_error(os, sys::EINVAL, &NO_SPACE))
    }

    fn sync(&self, h: &File, mode: SyncMode) -> Result<(), IoError> {
        let r = match mode {
            SyncMode::None => Ok(()),
            SyncMode::DataSync => h.sync_data(),
            SyncMode::FullSync => h.sync_all(),
        };
        r.map_err(|e| transfer_error(e.raw_os_error().unwrap_or(0), sys::EINVAL, &NO_SPACE))
    }
}

/// The macOS engine that reads around the unified buffer cache: `F_NOCACHE` set before the first
/// read, read-ahead off, a `pread`/`preadv` pool and gathered requests of at least 32 KiB
/// (RFC-0021 §5; ADR-0013 item 5). See the [module](crate::pool) for the pool's shape.
pub struct MacNoCache {
    pool: Pool<Mac<RealSys>>,
}

impl MacNoCache {
    /// Starts a pool of `threads` workers;
    /// [`POOL_THREADS_MACOS_DEFAULT`](super::POOL_THREADS_MACOS_DEFAULT) is
    /// `io.pool_threads_macos`' default. The thread count is the engine's depth.
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
            pool: Pool::new(Mac { sys: RealSys }, threads)?,
        })
    }

    /// Why the engine runs buffered: `IO_DIRECT_REFUSED` with the OS error of the refused
    /// `F_NOCACHE`. `None` while every file has it set.
    #[must_use]
    pub fn buffered_reason(&self) -> Option<IoError> {
        self.pool.buffered_reason()
    }
}

impl std::fmt::Debug for MacNoCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacNoCache")
            .field("buffered_reason", &self.buffered_reason())
            .finish_non_exhaustive()
    }
}

super::forward_engine!(MacNoCache);

#[cfg(test)]
mod tests {
    //! The order of system calls, recorded by a shim that forwards each call to the real one.

    use std::sync::Mutex;

    use super::*;
    use crate::{
        Bytes, CompletionBuf, ExtentRef, IoDest, IoEngine, IoOp, IoPriority, IoTicket, Nanos,
        ReadRequest, SubmitBatch, Wait,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Call {
        Fcntl {
            fd: c_int,
            cmd: c_int,
            arg: c_int,
        },
        Read {
            fd: c_int,
            offset: i64,
            segs: usize,
            len: u64,
        },
    }

    /// Forwards to [`RealSys`] and records each call, in order. `refuse_nocache` makes
    /// `F_NOCACHE` fail with `EINVAL` without calling the OS.
    #[derive(Default)]
    struct Recorder {
        calls: Arc<Mutex<Vec<Call>>>,
        refuse_nocache: bool,
    }

    impl MacSys for Recorder {
        fn fcntl(&self, fd: c_int, cmd: c_int, arg: c_int) -> Result<(), i32> {
            self.calls
                .lock()
                .unwrap()
                .push(Call::Fcntl { fd, cmd, arg });
            if self.refuse_nocache && cmd == sys::F_NOCACHE {
                return Err(sys::EINVAL);
            }
            RealSys.fcntl(fd, cmd, arg)
        }

        unsafe fn read(&self, fd: c_int, offset: i64, segs: &[Seg]) -> Result<u64, i32> {
            let len = segs.iter().map(|s| s.len).sum();
            self.calls.lock().unwrap().push(Call::Read {
                fd,
                offset,
                segs: segs.len(),
                len,
            });
            // SAFETY: the caller's guarantee, passed on unchanged.
            unsafe { RealSys.read(fd, offset, segs) }
        }

        unsafe fn write(&self, fd: c_int, offset: i64, seg: Seg) -> Result<u64, i32> {
            // SAFETY: the caller's guarantee, passed on unchanged.
            unsafe { RealSys.write(fd, offset, seg) }
        }
    }

    #[repr(C, align(16384))]
    struct Block([u8; 16384]);

    fn fixture(name: &str, len: usize) -> (std::path::PathBuf, Vec<u8>) {
        let bytes: Vec<u8> = (0..len).map(|i| (i * 7 + i / 4096) as u8).collect();
        let path = std::env::temp_dir().join(format!(
            "syntropy-io-macnocache-{name}-{}",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        (path, bytes)
    }

    fn read_op(ticket: u64, offset: u64, len: u64, addr: u64) -> IoOp {
        IoOp::Read(ReadRequest {
            ticket: IoTicket(ticket),
            extent: ExtentRef {
                file: syntropy_core::FileIx(0),
                offset,
                len: Bytes(len),
            },
            // SAFETY: each test passes a range of its own `Block`s, alive until every completion
            // is reaped, with one destination per range at a time.
            dest: unsafe { IoDest::raw(addr, Bytes(len)) },
            priority: IoPriority::Demand,
            useful_until: None,
        })
    }

    fn reap_all(pool: &mut Pool<Mac<Recorder>>, want: usize) -> Vec<crate::ReadCompletion> {
        let mut out = CompletionBuf::with_capacity(64);
        let mut got = Vec::new();
        for _ in 0..10_000 {
            pool.reap(&mut out, Wait::Until(Nanos(1_000_000)));
            got.extend(out.drain());
            if got.len() == want {
                return got;
            }
        }
        panic!("only {} of {want} completions arrived", got.len());
    }

    #[test]
    fn f_nocache_and_f_rdahead_are_set_before_any_read() {
        let (path, bytes) = fixture("order", 64 << 10);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let sys = Recorder {
            calls: Arc::clone(&calls),
            refuse_nocache: false,
        };
        let mut pool = Pool::new(Mac { sys }, 2).unwrap();
        pool.open(&path, OpenMode::Read).unwrap();
        assert!(pool.caps().direct, "F_NOCACHE was accepted");
        let mut blocks: Vec<Block> = (0..2).map(|_| Block([0; 16384])).collect();
        let base = blocks.as_mut_ptr() as u64;
        let mut batch = SubmitBatch::with_capacity(4);
        batch.push(read_op(1, 16384, 16384, base)).unwrap();
        batch.push(read_op(2, 0, 16384, base + 16384)).unwrap();
        assert_eq!(pool.submit(&mut batch).unwrap(), 2);
        let done = reap_all(&mut pool, 2);
        assert!(done.iter().all(|c| c.result == Ok(Bytes(16384))));
        assert_eq!(&blocks[0].0[..], &bytes[16384..32768]);
        assert_eq!(&blocks[1].0[..], &bytes[..16384]);
        drop(pool);

        let calls = calls.lock().unwrap().clone();
        let Some(Call::Fcntl { fd, cmd, arg }) = calls.first().copied() else {
            panic!("the first call is not an fcntl: {calls:?}");
        };
        assert_eq!((cmd, arg), (sys::F_NOCACHE, 1), "{calls:?}");
        assert_eq!(
            calls[1],
            Call::Fcntl {
                fd,
                cmd: sys::F_RDAHEAD,
                arg: 0
            },
            "{calls:?}"
        );
        let reads = calls[2..]
            .iter()
            .filter(|c| matches!(c, Call::Read { fd: f, .. } if *f == fd))
            .count();
        assert_eq!(reads, 2, "{calls:?}");
        assert!(
            calls[2..].iter().all(|c| matches!(c, Call::Read { .. })),
            "{calls:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn contiguous_small_reads_are_gathered_to_at_least_32_kib() {
        let (path, bytes) = fixture("gather", 64 << 10);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let sys = Recorder {
            calls: Arc::clone(&calls),
            refuse_nocache: false,
        };
        let mut pool = Pool::new(Mac { sys }, 2).unwrap();
        pool.open(&path, OpenMode::Read).unwrap();
        let mut blocks: Vec<Block> = (0..3).map(|_| Block([0; 16384])).collect();
        let base = blocks.as_mut_ptr() as u64;
        // Twelve 4 KiB rows, contiguous in the file: eight make 32 KiB, then four more.
        let mut batch = SubmitBatch::with_capacity(16);
        for i in 0..12u64 {
            batch
                .push(read_op(i, 4096 * i, 4096, base + 4096 * i))
                .unwrap();
        }
        assert_eq!(pool.submit(&mut batch).unwrap(), 12);
        let done = reap_all(&mut pool, 12);
        assert!(done.iter().all(|c| c.result == Ok(Bytes(4096))));
        drop(pool);
        let got: Vec<u8> = blocks.iter().flat_map(|b| b.0).take(12 * 4096).collect();
        assert_eq!(got, bytes[..12 * 4096]);

        let reads: Vec<(usize, u64)> = calls
            .lock()
            .unwrap()
            .iter()
            .filter_map(|c| match c {
                Call::Read { segs, len, .. } => Some((*segs, *len)),
                Call::Fcntl { .. } => None,
            })
            .collect();
        assert_eq!(reads, [(8, MIN_REQUEST), (4, 16384)]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_refused_f_nocache_flags_the_volume_buffered() {
        let (path, _) = fixture("refused", 4096);
        let sys = Recorder {
            calls: Arc::default(),
            refuse_nocache: true,
        };
        let mut pool = Pool::new(Mac { sys }, 1).unwrap();
        pool.open(&path, OpenMode::Read).unwrap();
        assert!(!pool.caps().direct);
        assert_eq!(
            pool.buffered_reason(),
            Some(IoError::with_os_error(
                IoErrorCode::DirectRefused,
                sys::EINVAL
            ))
        );
        let _ = std::fs::remove_file(&path);
    }
}
