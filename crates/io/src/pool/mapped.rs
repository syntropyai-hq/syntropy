// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MacMapped` (RFC-0021 §5; ADR-0013 item 5; ADR-0007 item 6).
//!
//! A file opened for reading is mapped read-only and whole (`mmap(PROT_READ, MAP_SHARED)`), and
//! the mapping is marked `madvise(MADV_RANDOM)`. A read runs on a pool thread in three steps:
//! `madvise(MADV_WILLNEED)` on the pages the extent covers; `mincore` polled every 50 µs
//! ([`RESIDENT_POLL`]) until every page is resident; a copy from the mapping into the
//! destination slot. `madvise(MADV_DONTNEED)` then releases the range from the mapping. The
//! clean file pages that remain are the OS's cache. The engine never counts them, and it has no
//! way to report them: a completion reports only the bytes copied into the slot, which the tier
//! manager owns (ADR-0007 item 6). `MacMapped` is a read mechanism, not a cache.
//!
//! Choices RFC-0021 §5 leaves open:
//! - **The wait is bounded.** If the pages are not all resident after [`RESIDENT_WAIT_MAX`], or
//!   `mincore` fails, the read is a `pread` of the file instead. A copy that faults a page in and
//!   meets an I/O error raises `SIGBUS`; a `pread` returns the error, which the I/O thread
//!   handles as RFC-0021 §16 says.
//! - **Reads past the end of the file are short.** The mapping covers the file's length when it
//!   is opened. A read from or past that end returns 0 bytes, and one that crosses it returns
//!   the bytes before it, as `pread` does. A pack file does not change once it is committed
//!   (RFC-0041), so the length holds. A file truncated while it is mapped raises `SIGBUS` on the
//!   pages it lost.
//! - **Writes go through `F_NOCACHE`,** as in `MacNoCache` (RFC-0021 §13: writes use the same
//!   engines and never the page cache). A file opened for writing is not mapped, because it
//!   grows while it is written; its reads are `pread`s on the `F_NOCACHE` handle too. An empty
//!   file is not mapped either (`mmap` refuses a length of zero), and every read of it returns 0
//!   bytes.
//! - **A refused mapping** fails `open` with `IO_ENGINE_UNAVAILABLE` and the OS error: the next
//!   engine, `MacNoCache`, reads that file (RFC-0021 §16).
//! - **No gathering.** The copy costs the same per byte at any request size, so contiguous reads
//!   are not joined, unlike `MacNoCache`.
//! - `MADV_DONTNEED` on a range drops the mapping's pages there, including a page another
//!   worker's read shares at an edge. That read then faults the page in again from the cache: it
//!   costs time, never correctness.
//!
//! The mapping calls go through the crate-private `MapSys` trait, so a test can record their
//! order against the real calls; the engine uses `RealMap`. The `F_NOCACHE` calls are
//! `MacNoCache`'s own `MacSys`.

use std::ffi::c_int;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use syntropy_core::resource::EngineKind;

use super::macos::{Mac, MacSys, RealSys};
use super::{Backend, Opened, Pool, Seg};
use crate::align;
use crate::engine::OpenMode;
use crate::error::{IoError, IoErrorCode};
use crate::request::SyncMode;

/// How often a read polls `mincore` while it waits for its pages: 50 µs \[target\] (RFC-0021 §5).
pub const RESIDENT_POLL: Duration = Duration::from_micros(50);

/// How long a read waits for its pages before it reads with `pread` instead: 20 ms \[target\],
/// a chosen budget. A 4 MiB chunk takes about 2 ms at 2 GB/s \[estimate\]: 4 MiB / 2 GB/s.
pub const RESIDENT_WAIT_MAX: Duration = Duration::from_millis(20);

/// Pages one `mincore` call checks: its result vector lives on the worker's stack, so a read
/// allocates nothing. 256 pages are one 4 MiB chunk at the 16 KiB page of Apple silicon.
const MINCORE_PAGES: usize = 256;

/// The mapping calls `MacMapped` makes.
pub(crate) trait MapSys: Send + Sync + 'static {
    /// `mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0)`: the mapping's address, or the OS error.
    fn map(&self, fd: c_int, len: usize) -> Result<usize, i32>;

    /// `madvise(addr, len, advice)`; the OS error on failure.
    ///
    /// # Safety
    ///
    /// `[addr, addr + len)` lies inside a live mapping this engine made of a file, read-only and
    /// shared, so no advice changes the bytes it holds.
    unsafe fn advise(&self, addr: usize, len: usize, advice: c_int) -> Result<(), i32>;

    /// `mincore(addr, len, vec)`, which writes one byte per page of the range into `vec`; the OS
    /// error on failure.
    ///
    /// # Safety
    ///
    /// As for [`MapSys::advise`]; `vec` has at least one byte per page of the range.
    unsafe fn residency(&self, addr: usize, len: usize, vec: &mut [u8]) -> Result<(), i32>;
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// The real mapping calls.
#[derive(Debug, Default)]
pub(crate) struct RealMap;

impl MapSys for RealMap {
    fn map(&self, fd: c_int, len: usize) -> Result<usize, i32> {
        // SAFETY: a new read-only mapping at an address the kernel picks overlaps no memory the
        // program uses; the fd is the caller's open file.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            Err(errno())
        } else {
            Ok(p as usize)
        }
    }

    unsafe fn advise(&self, addr: usize, len: usize, advice: c_int) -> Result<(), i32> {
        // SAFETY: the caller guarantees the range is inside a shared read-only file mapping,
        // where advice changes residency and never contents.
        let rc = unsafe { libc::madvise(addr as *mut libc::c_void, len, advice) };
        if rc == -1 { Err(errno()) } else { Ok(()) }
    }

    unsafe fn residency(&self, addr: usize, len: usize, vec: &mut [u8]) -> Result<(), i32> {
        // SAFETY: the caller guarantees the range is mapped and `vec` holds a byte per page.
        let rc = unsafe {
            libc::mincore(
                addr as *const libc::c_void,
                len,
                vec.as_mut_ptr().cast::<libc::c_char>(),
            )
        };
        if rc == -1 { Err(errno()) } else { Ok(()) }
    }
}

/// A whole-file mapping, unmapped on drop.
#[derive(Debug)]
struct Region {
    /// The mapping's first byte, as an address: the workers share it, and only `read` turns it
    /// back into a pointer.
    base: usize,
    /// The file's length when it was mapped. The mapping covers it rounded up to a page.
    len: u64,
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: `base` and `len` are the mapping `MapSys::map` returned, and the last handle
        // to it is going: every worker that copied from it has been joined or dropped its
        // handle, so nothing reads it after this.
        unsafe { libc::munmap(self.base as *mut libc::c_void, self.len as usize) };
    }
}

/// An open file as the workers use it: the file, and its mapping when it was opened for reading
/// and is not empty.
#[derive(Debug)]
pub(crate) struct MappedFile {
    file: std::fs::File,
    map: Option<Region>,
}

/// The `MacMapped` backend over a set of mapping calls and `MacNoCache`'s file calls.
#[derive(Debug)]
pub(crate) struct Mapped<M, S> {
    pub(crate) map: M,
    /// Opens files and runs `pread`, `pwrite` and sync, as `MacNoCache` does.
    pub(crate) nocache: Mac<S>,
    /// The OS page size: the unit of `madvise` and `mincore`.
    pub(crate) page: u64,
}

impl<M: MapSys, S: MacSys> Mapped<M, S> {
    /// Polls `mincore` until every page of `[addr, addr + len)` is resident. `false` if the wait
    /// runs past [`RESIDENT_WAIT_MAX`] or `mincore` fails.
    ///
    /// # Safety
    ///
    /// As for [`MapSys::advise`]; `addr` and `len` are multiples of the page size.
    unsafe fn wait_resident(&self, addr: usize, len: usize) -> bool {
        let page = self.page as usize;
        let deadline = Instant::now() + RESIDENT_WAIT_MAX;
        let mut vec = [0u8; MINCORE_PAGES];
        // Pages before `done` were all resident at their last check.
        let mut done = 0;
        loop {
            while done < len {
                let span = (len - done).min(MINCORE_PAGES * page);
                let pages = span / page;
                // SAFETY: a sub-range of the caller's range, page-aligned; `vec` holds
                // `MINCORE_PAGES` bytes, at least one per page of `span`.
                if unsafe { self.map.residency(addr + done, span, &mut vec) }.is_err() {
                    return false;
                }
                let resident = vec[..pages]
                    .iter()
                    .take_while(|&&v| c_int::from(v) & libc::MINCORE_INCORE != 0)
                    .count();
                done += resident * page;
                if resident < pages {
                    break;
                }
            }
            if done >= len {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(RESIDENT_POLL);
        }
    }
}

impl<M: MapSys, S: MacSys> Backend for Mapped<M, S> {
    type Handle = MappedFile;
    const KIND: EngineKind = EngineKind::MacMapped;
    const GATHER_TO: u64 = 0;

    fn open(
        &self,
        path: &Path,
        mode: OpenMode,
        workers: usize,
    ) -> Result<Opened<MappedFile>, IoError> {
        let opened = self.nocache.open(path, mode, workers)?;
        let Opened {
            handles,
            refused,
            dio,
        } = opened;
        let file = Arc::into_inner(handles.into_iter().next().expect("one handle"))
            .expect("no worker has the handle yet");
        let len = file
            .metadata()
            .map_err(|e| super::open_error(&e, &[]))?
            .len();
        let (map, refused) = if mode == OpenMode::Read && len > 0 {
            let size =
                usize::try_from(len).map_err(|_| IoError::new(IoErrorCode::EngineUnavailable))?;
            let base = self
                .map
                .map(file.as_raw_fd(), size)
                .map_err(|os| IoError::with_os_error(IoErrorCode::EngineUnavailable, os))?;
            let region = Region { base, len };
            // A refused `MADV_RANDOM` changes only speed.
            // SAFETY: the whole mapping just made, shared and read-only.
            let _ = unsafe { self.map.advise(base, size, libc::MADV_RANDOM) };
            // A mapped file is read through the mapping, so a refused `F_NOCACHE` does not make
            // it the flagged buffered mode; it affects only the `pread` after a timed-out wait.
            (Some(region), None)
        } else {
            (None, refused)
        };
        Ok(Opened {
            handles: vec![Arc::new(MappedFile { file, map })],
            refused,
            dio,
        })
    }

    unsafe fn read(&self, h: &MappedFile, offset: u64, segs: &[Seg]) -> Result<u64, IoError> {
        let Some(map) = &h.map else {
            // SAFETY: the caller's guarantee, passed on unchanged.
            return unsafe { self.nocache.read(&h.file, offset, segs) };
        };
        let want: u64 = segs.iter().map(|s| s.len).sum();
        let end = offset.saturating_add(want).min(map.len);
        if offset >= end {
            return Ok(0);
        }
        // The pages the bytes `[offset, end)` lie on. The mapping covers `map.len` rounded up to
        // a page, so the last one is inside it.
        let first = offset / self.page * self.page;
        let last = end.div_ceil(self.page) * self.page;
        let (addr, span) = (map.base + first as usize, (last - first) as usize);
        // A refused `MADV_WILLNEED` changes only speed.
        // SAFETY: `[addr, addr + span)` is inside the mapping, which `h` keeps alive.
        let _ = unsafe { self.map.advise(addr, span, libc::MADV_WILLNEED) };
        // SAFETY: as above; `addr` and `span` are page multiples, since `base` is.
        let result = if unsafe { self.wait_resident(addr, span) } {
            let mut src = map.base + offset as usize;
            let mut left = end - offset;
            for s in segs {
                let n = s.len.min(left);
                // SAFETY: the source lies in `[offset, end)`, inside the mapping, and is
                // readable; the destination is the caller's segment, valid for writes over
                // `s.len >= n` bytes and used by nothing else until the call returns. A
                // read-only mapping and tier-manager memory do not overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(src as *const u8, s.addr as *mut u8, n as usize);
                }
                src += n as usize;
                left -= n;
                if left == 0 {
                    break;
                }
            }
            Ok(end - offset)
        } else {
            // SAFETY: the caller's guarantee, passed on unchanged.
            unsafe { self.nocache.read(&h.file, offset, segs) }
        };
        // After the copy: the bytes are in the slot, so the mapping lets its pages go. A refused
        // `MADV_DONTNEED` leaves them to the OS, which can reclaim clean file pages any time.
        // SAFETY: the range advised above.
        let _ = unsafe { self.map.advise(addr, span, libc::MADV_DONTNEED) };
        result
    }

    unsafe fn write(&self, h: &MappedFile, offset: u64, seg: Seg) -> Result<u64, IoError> {
        // SAFETY: the caller's guarantee, passed on unchanged.
        unsafe { self.nocache.write(&h.file, offset, seg) }
    }

    fn sync(&self, h: &MappedFile, mode: SyncMode) -> Result<(), IoError> {
        self.nocache.sync(&h.file, mode)
    }
}

/// The macOS engine that reads through a read-only mapping and copies into owned slots:
/// `MADV_WILLNEED` on request, a copy once `mincore` shows the pages resident, `MADV_DONTNEED`
/// after (RFC-0021 §5; ADR-0013 item 5). It never counts the page cache as residency (ADR-0007
/// item 6). RFC-0010's probe chooses it over [`MacNoCache`](super::MacNoCache) per volume by
/// measurement (RFC-0021 §10). See the [module](crate::pool) for the pool's shape.
pub struct MacMapped {
    pool: Pool<Mapped<RealMap, RealSys>>,
}

impl MacMapped {
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
        let backend = Mapped {
            map: RealMap,
            nocache: Mac { sys: RealSys },
            page: align::os_page_size(),
        };
        Ok(Self {
            pool: Pool::new(backend, threads)?,
        })
    }

    /// Why the engine runs buffered: `IO_DIRECT_REFUSED` with the OS error of a refused
    /// `F_NOCACHE` on a file opened for writing. `None` while every such file has it set. Reads
    /// through the mapping are this engine's design, not the flagged buffered mode.
    #[must_use]
    pub fn buffered_reason(&self) -> Option<IoError> {
        self.pool.buffered_reason()
    }
}

impl std::fmt::Debug for MacMapped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacMapped")
            .field("buffered_reason", &self.buffered_reason())
            .finish_non_exhaustive()
    }
}

super::forward_engine!(MacMapped);

#[cfg(test)]
mod tests {
    //! The order of the mapping calls, recorded by a shim that forwards each call to the real
    //! one, and the residency check of TIER-E5-T09: once a read completes, no page of the
    //! mapping is still advised in, and the completion reports only the bytes in the slot.

    use std::sync::Mutex;

    use super::*;
    use crate::{
        Bytes, CompletionBuf, ExtentRef, IoDest, IoEngine, IoOp, IoPriority, IoTicket, Nanos,
        ReadRequest, SubmitBatch, Wait, WriteRequest,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Call {
        Map {
            len: usize,
        },
        Advise {
            addr: usize,
            len: usize,
            advice: c_int,
        },
        Mincore {
            addr: usize,
            len: usize,
        },
    }

    /// Forwards to [`RealMap`] and records each call, in order. `refuse_map` makes `mmap` fail
    /// with `ENODEV` without calling the OS; `never_resident` reports every page absent, so each
    /// wait times out.
    #[derive(Default)]
    struct Recorder {
        calls: Arc<Mutex<Vec<Call>>>,
        refuse_map: bool,
        never_resident: bool,
    }

    const ENODEV: i32 = 19;

    impl MapSys for Recorder {
        fn map(&self, fd: c_int, len: usize) -> Result<usize, i32> {
            self.calls.lock().unwrap().push(Call::Map { len });
            if self.refuse_map {
                return Err(ENODEV);
            }
            RealMap.map(fd, len)
        }

        unsafe fn advise(&self, addr: usize, len: usize, advice: c_int) -> Result<(), i32> {
            self.calls
                .lock()
                .unwrap()
                .push(Call::Advise { addr, len, advice });
            // SAFETY: the caller's guarantee, passed on unchanged.
            unsafe { RealMap.advise(addr, len, advice) }
        }

        unsafe fn residency(&self, addr: usize, len: usize, vec: &mut [u8]) -> Result<(), i32> {
            self.calls.lock().unwrap().push(Call::Mincore { addr, len });
            // SAFETY: the caller's guarantee, passed on unchanged.
            unsafe { RealMap.residency(addr, len, vec) }?;
            if self.never_resident {
                vec.fill(0);
            }
            Ok(())
        }
    }

    type TestPool = Pool<Mapped<Recorder, RealSys>>;

    fn pool(threads: usize, rec: Recorder) -> TestPool {
        let backend = Mapped {
            map: rec,
            nocache: Mac { sys: RealSys },
            page: align::os_page_size(),
        };
        Pool::new(backend, threads).unwrap()
    }

    fn recorder() -> (Recorder, Arc<Mutex<Vec<Call>>>) {
        let rec = Recorder::default();
        let calls = Arc::clone(&rec.calls);
        (rec, calls)
    }

    #[repr(C, align(16384))]
    #[derive(Clone)]
    struct Block([u8; 16384]);

    fn fixture(name: &str, len: usize) -> (std::path::PathBuf, Vec<u8>) {
        let bytes: Vec<u8> = (0..len).map(|i| (i * 13 + i / 4096) as u8).collect();
        let path = std::env::temp_dir().join(format!(
            "syntropy-io-macmapped-{name}-{}",
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

    /// Submits `ops` and reaps until each has completed; the completions in ticket order.
    fn run(pool: &mut TestPool, ops: Vec<IoOp>) -> Vec<crate::ReadCompletion> {
        let want = ops.len();
        let mut batch = SubmitBatch::with_capacity(want.max(1));
        for op in ops {
            batch.push(op).unwrap();
        }
        let mut out = CompletionBuf::with_capacity(64);
        let mut got = Vec::new();
        for _ in 0..100_000 {
            if !batch.is_empty() {
                pool.submit(&mut batch).unwrap();
            }
            pool.reap(&mut out, Wait::Until(Nanos(1_000_000)));
            got.extend(out.drain());
            if got.len() == want {
                got.sort_by_key(|c| c.ticket);
                return got;
            }
        }
        panic!("only {} of {want} completions arrived", got.len());
    }

    fn contents(blocks: &[Block]) -> Vec<u8> {
        blocks.iter().flat_map(|b| b.0).collect()
    }

    /// The `(addr, len)` of every `advice` call, in order.
    fn advised(calls: &[Call], advice: c_int) -> Vec<(usize, usize)> {
        calls
            .iter()
            .filter_map(|c| match *c {
                Call::Advise {
                    addr,
                    len,
                    advice: a,
                } if a == advice => Some((addr, len)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn open_maps_the_whole_file_read_only_and_random() {
        let (path, bytes) = fixture("open", 100_000);
        let (rec, calls) = recorder();
        let mut pool = pool(1, rec);
        pool.open(&path, OpenMode::Read).unwrap();
        assert_eq!(pool.kind(), EngineKind::MacMapped);
        assert!(pool.caps().direct && pool.caps().writes);
        let calls = calls.lock().unwrap().clone();
        assert_eq!(calls[0], Call::Map { len: bytes.len() });
        let Call::Advise { len, advice, .. } = calls[1] else {
            panic!("{calls:?}");
        };
        assert_eq!((len, advice), (bytes.len(), libc::MADV_RANDOM));
        assert_eq!(calls.len(), 2, "{calls:?}");
        drop(pool);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_read_is_willneed_then_mincore_then_dontneed_on_its_pages() {
        let page = align::os_page_size();
        let (path, bytes) = fixture("order", 8 * page as usize);
        let (rec, calls) = recorder();
        let mut pool = pool(1, rec);
        pool.open(&path, OpenMode::Read).unwrap();
        calls.lock().unwrap().clear();
        let mut blocks = vec![Block([0; 16384]); 2];
        let base = blocks.as_mut_ptr() as u64;
        // Two pages from page 3.
        let done = run(&mut pool, vec![read_op(1, 3 * page, 2 * page, base)]);
        assert_eq!(done[0].result, Ok(Bytes(2 * page)));
        let p = page as usize;
        assert!(contents(&blocks)[..2 * p] == bytes[3 * p..5 * p]);
        drop(pool);

        let calls = calls.lock().unwrap().clone();
        let Some(Call::Advise { addr, len, advice }) = calls.first().copied() else {
            panic!("{calls:?}");
        };
        assert_eq!((len, advice), (2 * p, libc::MADV_WILLNEED), "{calls:?}");
        let last = calls.len() - 1;
        assert_eq!(
            calls[last],
            Call::Advise {
                addr,
                len,
                advice: libc::MADV_DONTNEED
            },
            "{calls:?}"
        );
        assert!(
            calls[1..last]
                .iter()
                .all(|c| matches!(c, Call::Mincore { addr: a, len: l } if *a == addr && *l == len)),
            "{calls:?}"
        );
        assert!(last >= 2, "mincore was polled: {calls:?}");
        let _ = std::fs::remove_file(&path);
    }

    /// TIER-E5-T09 criterion 2, the engine's half: once every read has completed, every range
    /// advised in has been released, so the mapping holds no page as the engine's own, and each
    /// completion reports exactly the bytes copied into its slot. RFC-0020 §3 keys a home's
    /// `Loading` state to the read's ticket, so the residency map (TIER-E1-T04) learns of a read
    /// only from its completion, and a mapped page never reaches it.
    #[test]
    fn no_mapped_page_stays_advised_in_after_its_read() {
        let page = align::os_page_size();
        let (path, bytes) = fixture("residency", 64 * page as usize);
        let (rec, calls) = recorder();
        let mut pool = pool(4, rec);
        pool.open(&path, OpenMode::Read).unwrap();
        calls.lock().unwrap().clear();
        let n = 16u64;
        let mut blocks = vec![Block([0; 16384]); n as usize];
        let base = blocks.as_mut_ptr() as u64;
        // Unaligned extents one page apart on four workers: each covers two pages and shares one
        // with the next, so reads that share a page run at once.
        let ops = (0..n)
            .map(|i| read_op(i, i * page + 777, 16384, base + i * 16384))
            .collect();
        let done = run(&mut pool, ops);
        assert_eq!(pool.in_flight(), (0, Bytes(0)));
        for c in &done {
            let i = c.ticket.0 as usize;
            assert_eq!(c.result, Ok(Bytes(16384)), "{c:?}");
            let off = i * page as usize + 777;
            assert!(blocks[i].0[..] == bytes[off..off + 16384]);
        }
        drop(pool);
        let calls = calls.lock().unwrap().clone();
        let mut willneed = advised(&calls, libc::MADV_WILLNEED);
        let mut dontneed = advised(&calls, libc::MADV_DONTNEED);
        assert_eq!(willneed.len(), n as usize);
        willneed.sort_unstable();
        dontneed.sort_unstable();
        assert_eq!(willneed, dontneed, "every range advised in is released");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pages_that_never_arrive_are_read_with_pread_and_still_released() {
        let page = align::os_page_size();
        let (path, bytes) = fixture("timeout", 4 * page as usize);
        let rec = Recorder {
            never_resident: true,
            ..Recorder::default()
        };
        let calls = Arc::clone(&rec.calls);
        let mut pool = pool(1, rec);
        pool.open(&path, OpenMode::Read).unwrap();
        let mut blocks = vec![Block([0; 16384]); 1];
        let base = blocks.as_mut_ptr() as u64;
        let started = Instant::now();
        let done = run(&mut pool, vec![read_op(1, page, 16384, base)]);
        assert!(started.elapsed() >= RESIDENT_WAIT_MAX);
        assert_eq!(done[0].result, Ok(Bytes(16384)));
        let p = page as usize;
        assert!(contents(&blocks)[..16384] == bytes[p..p + 16384]);
        drop(pool);
        let calls = calls.lock().unwrap().clone();
        let polls = calls
            .iter()
            .filter(|c| matches!(c, Call::Mincore { .. }))
            .count();
        assert!(polls > 1, "the wait polled: {polls}");
        assert_eq!(
            advised(&calls, libc::MADV_WILLNEED),
            advised(&calls, libc::MADV_DONTNEED)
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reads_at_and_across_the_end_of_the_file_are_short() {
        let page = align::os_page_size();
        let len = 2 * page as usize + 1000;
        let (path, bytes) = fixture("eof", len);
        let (rec, _) = recorder();
        let mut pool = pool(2, rec);
        pool.open(&path, OpenMode::Read).unwrap();
        let mut blocks = vec![Block([0xAA; 16384]); 3];
        let base = blocks.as_mut_ptr() as u64;
        let at = 2 * page;
        let done = run(
            &mut pool,
            vec![
                // Across the end: 1000 bytes are left.
                read_op(1, at, 16384, base),
                // At the end, and past it.
                read_op(2, len as u64, 16384, base + 16384),
                read_op(3, len as u64 + 5 * page, 16384, base + 2 * 16384),
            ],
        );
        assert_eq!(done[0].result, Ok(Bytes(1000)));
        assert_eq!(done[1].result, Ok(Bytes(0)));
        assert_eq!(done[2].result, Ok(Bytes(0)));
        assert!(blocks[0].0[..1000] == bytes[at as usize..]);
        assert!(
            blocks[0].0[1000..].iter().all(|&b| b == 0xAA),
            "past the bytes read"
        );
        assert!(blocks[1..].iter().all(|b| b.0.iter().all(|&x| x == 0xAA)));
        drop(pool);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unaligned_extent_is_copied_exactly() {
        let (path, bytes) = fixture("unaligned", 200_000);
        let (rec, _) = recorder();
        let mut pool = pool(1, rec);
        pool.open(&path, OpenMode::Read).unwrap();
        let mut blocks = vec![Block([0; 16384]); 1];
        let base = blocks.as_mut_ptr() as u64;
        let done = run(&mut pool, vec![read_op(1, 12_345, 5_000, base + 3)]);
        assert_eq!(done[0].result, Ok(Bytes(5_000)));
        let got = contents(&blocks);
        assert!(got[3..5_003] == bytes[12_345..17_345]);
        assert!(got[..3].iter().chain(&got[5_003..]).all(|&b| b == 0));
        drop(pool);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_empty_file_is_not_mapped_and_reads_nothing() {
        let (path, _) = fixture("empty", 0);
        let (rec, calls) = recorder();
        let mut pool = pool(1, rec);
        pool.open(&path, OpenMode::Read).unwrap();
        let mut blocks = vec![Block([0; 16384]); 1];
        let base = blocks.as_mut_ptr() as u64;
        let done = run(&mut pool, vec![read_op(1, 0, 16384, base)]);
        assert_eq!(done[0].result, Ok(Bytes(0)));
        drop(pool);
        assert!(calls.lock().unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_refused_mapping_makes_the_engine_unavailable_for_the_file() {
        let (path, _) = fixture("refused", 16384);
        let rec = Recorder {
            refuse_map: true,
            ..Recorder::default()
        };
        let mut pool = pool(1, rec);
        assert_eq!(
            pool.open(&path, OpenMode::Read),
            Err(IoError::with_os_error(
                IoErrorCode::EngineUnavailable,
                ENODEV
            ))
        );
        assert_eq!(pool.file_dio(syntropy_core::FileIx(0)), None);
        drop(pool);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_file_fails_to_open() {
        let (rec, calls) = recorder();
        let mut pool = pool(1, rec);
        let path = std::env::temp_dir().join("syntropy-io-macmapped-no-such-file");
        let err = pool.open(&path, OpenMode::Read).unwrap_err();
        assert_eq!(err.code, IoErrorCode::ExtentBad);
        assert!(calls.lock().unwrap().is_empty());
    }

    #[test]
    fn a_file_opened_for_writing_is_written_and_read_without_a_mapping() {
        let path = std::env::temp_dir().join(format!(
            "syntropy-io-macmapped-write-{}",
            std::process::id()
        ));
        let (rec, calls) = recorder();
        let mut pool = pool(2, rec);
        pool.open(&path, OpenMode::Create).unwrap();
        let mut src = vec![Block([0; 16384]); 2];
        src[0].0.fill(0x11);
        src[1].0.fill(0x22);
        let base = src.as_mut_ptr() as u64;
        let writes = (0..2u64)
            .map(|i| {
                IoOp::Write(WriteRequest {
                    ticket: IoTicket(i),
                    extent: ExtentRef {
                        file: syntropy_core::FileIx(0),
                        offset: i * 16384,
                        len: Bytes(16384),
                    },
                    // SAFETY: a range of `src`, alive until the completion is reaped.
                    src: unsafe { IoDest::raw(base + i * 16384, Bytes(16384)) },
                    priority: IoPriority::Background,
                    sync: SyncMode::DataSync,
                })
            })
            .collect();
        let done = run(&mut pool, writes);
        assert!(
            done.iter().all(|c| c.result == Ok(Bytes(16384))),
            "{done:?}"
        );
        let mut back = vec![Block([0; 16384]); 2];
        let at = back.as_mut_ptr() as u64;
        let done = run(
            &mut pool,
            vec![
                read_op(10, 0, 16384, at),
                read_op(11, 16384, 16384, at + 16384),
            ],
        );
        assert!(done.iter().all(|c| c.result == Ok(Bytes(16384))));
        assert!(contents(&back) == contents(&src));
        drop(pool);
        assert!(calls.lock().unwrap().is_empty(), "no mapping calls");
        assert_eq!(std::fs::read(&path).unwrap(), contents(&src));
        let _ = std::fs::remove_file(&path);
    }
}
