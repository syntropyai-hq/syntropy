// SPDX-License-Identifier: MIT OR Apache-2.0

//! The alignment contract of RFC-0021 §6 (ADR-0005 items 3 and 4).
//!
//! - [`query`] asks the OS what direct I/O needs for one open file: `statx(STATX_DIOALIGN)`,
//!   falling back to the block device's logical block size, on Linux;
//!   `GetFileInformationByHandleEx(FileStorageInfo)`, falling back to
//!   `IOCTL_STORAGE_QUERY_PROPERTY` (`StorageAccessAlignmentProperty`), on Windows; the 16 KiB
//!   VM page on macOS, since `F_NOCACHE` has no alignment rule of its own.
//! - [`RequestAlign`] turns that [`DioAlign`] into §6's `A` (file offsets and lengths) and `M`
//!   (memory), and checks a request against them before submission. A request that fails is a
//!   bug, `IO_MISALIGNED`; the I/O loop rejects it before it reaches the kernel and counts it.
//! - [`RequestAlign::pack_extent`] is §6's pack rule, which RFC-0041's pack builder follows: a
//!   unit starts at a multiple of `A_pack` and its length is padded to one.
//! - [`RequestAlign::window`] is the aligned block range that covers a byte range. Rows
//!   (`RowLookup`) are read that way, and so is an extent that is not packed (read-around).
//!   [`ReadAroundStats`] counts the payload and window bytes of each volume's read-arounds, and
//!   [`ReadAroundStats::report`] writes their ratio into the volume's capability profile
//!   ([`StorageCapsExt::read_around_efficiency`]).
//!
//! Every function here is plain arithmetic except [`query`] and [`os_page_size`], which call the
//! OS. The workspace has no C bindings crate, so those calls are declared by hand in the per-OS
//! modules, like the engines' own.

use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use syntropy_core::resource::{DioAlign, StorageCapsExt};
use syntropy_core::{Bytes, ExtentRef};

use crate::error::{IoError, IoErrorCode};
use crate::request::{IoDest, IoDestKind, IoOp};

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod linux;
#[cfg(windows)]
mod windows;

/// §6's floor for both `A` and `M`: 4 KiB.
pub const FLOOR: u64 = 4096;

/// The alignment Syntropy uses on macOS, where `F_NOCACHE` has no rule of its own: the 16 KiB VM
/// page of Apple silicon (RFC-0021 §6).
pub const MACOS_ALIGN: u32 = 16 << 10;

/// Where a [`DioQuery::Aligned`] value came from (RFC-0021 §6's table).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DioSource {
    /// Linux `statx(STATX_DIOALIGN)`: `stx_dio_mem_align` and `stx_dio_offset_align`
    /// (Linux 6.1 and later).
    Statx,
    /// Linux fallback: the block device's `queue/logical_block_size`, for both.
    LogicalBlockSize,
    /// Windows `FileStorageInfo`: `PhysicalBytesPerSectorForPerformance`, for both.
    FileStorageInfo,
    /// Windows fallback: `StorageAccessAlignmentProperty`'s physical sector size, for both.
    StorageAccessAlignment,
    /// macOS: [`MACOS_ALIGN`] for both.
    MacPage,
}

/// What the OS says direct I/O on one file needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DioQuery {
    /// The file's direct-I/O alignment, and where it came from.
    Aligned {
        /// The alignment.
        dio: DioAlign,
        /// Its source.
        source: DioSource,
    },
    /// The file system reports that it does not support direct I/O (`statx` returns zero for
    /// both values). RFC-0021 §6 sends such a file to buffered mode (§4.2).
    Unsupported,
    /// Nothing answered. [`RequestAlign`]'s 4 KiB floor still applies.
    Unknown,
}

impl DioQuery {
    /// The alignment to put in [`EngineCaps::dio`](crate::EngineCaps::dio): the reported value,
    /// or zero (the floor) for [`Unsupported`](DioQuery::Unsupported) and
    /// [`Unknown`](DioQuery::Unknown).
    #[must_use]
    pub const fn dio(self) -> DioAlign {
        match self {
            DioQuery::Aligned { dio, .. } => dio,
            DioQuery::Unsupported | DioQuery::Unknown => DioAlign { mem: 0, offset: 0 },
        }
    }
}

/// Asks the OS for the direct-I/O alignment of `file`, which was opened from `path`
/// (RFC-0021 §6). Engines call it in `open` and report the result in
/// [`EngineCaps::dio`](crate::EngineCaps::dio).
///
/// Never fails: an OS that answers nothing gives [`DioQuery::Unknown`].
#[must_use]
pub fn query(file: &File, path: &Path) -> DioQuery {
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    {
        let _ = path;
        linux::query(file)
    }
    #[cfg(windows)]
    {
        windows::query(file, path)
    }
    #[cfg(target_os = "macos")]
    {
        let _ = (file, path);
        DioQuery::Aligned {
            dio: DioAlign {
                mem: MACOS_ALIGN,
                offset: MACOS_ALIGN,
            },
            source: DioSource::MacPage,
        }
    }
    #[cfg(not(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        windows,
        target_os = "macos"
    )))]
    {
        let _ = (file, path);
        DioQuery::Unknown
    }
}

/// The OS page size, which §6's `M` includes. [`FLOOR`] where the OS does not answer.
#[must_use]
pub fn os_page_size() -> u64 {
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    {
        linux::page_size()
    }
    #[cfg(target_os = "macos")]
    {
        unix_page_size(29) // _SC_PAGESIZE on macOS
    }
    #[cfg(windows)]
    {
        windows::page_size()
    }
    #[cfg(not(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        windows,
        target_os = "macos"
    )))]
    {
        FLOOR
    }
}

/// `sysconf(name)` as a page size, or [`FLOOR`] if it fails.
#[cfg(any(
    target_os = "macos",
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn unix_page_size(name: std::ffi::c_int) -> u64 {
    unsafe extern "C" {
        fn sysconf(name: std::ffi::c_int) -> std::ffi::c_long;
    }
    // SAFETY: `sysconf` takes an integer and returns one; it touches no caller memory.
    let n = unsafe { sysconf(name) };
    u64::try_from(n).ok().filter(|&n| n > 0).unwrap_or(FLOOR)
}

/// The least common multiple of `a` and `b`, saturating at `u64::MAX`. Both are positive.
fn lcm(a: u64, b: u64) -> u64 {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    (a / x).saturating_mul(b)
}

/// RFC-0021 §6's `A` and `M` for one volume: what every request's file range and memory must
/// be aligned to.
///
/// `A = max(4096, offset)` and `M = max(4096, mem, OS page size)`. Every value an OS reports is
/// a power of two, for which the maximum is also the least common multiple; for any other
/// value this type takes the least common multiple, so a request aligned to `A` is aligned to
/// every part of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RequestAlign {
    offset: u64,
    mem: u64,
}

impl RequestAlign {
    /// The floor alone: 4 KiB for both.
    pub const FLOOR: RequestAlign = RequestAlign {
        offset: FLOOR,
        mem: FLOOR,
    };

    /// `A` and `M` for a volume whose files report `dio`, on a host whose page size is `page`.
    /// A zero in `dio` or `page` means "not reported", and the floor stands in for it.
    #[must_use]
    pub fn from_dio(dio: DioAlign, page: u64) -> Self {
        let offset = lcm(FLOOR, u64::from(dio.offset).max(1));
        let mem = lcm(lcm(FLOOR, u64::from(dio.mem).max(1)), page.max(1));
        Self { offset, mem }
    }

    /// `A`: file offsets and lengths are multiples of it.
    #[must_use]
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// `M`: memory addresses are multiples of it.
    #[must_use]
    pub const fn mem(self) -> u64 {
        self.mem
    }

    /// Whether an extent's offset and length are multiples of `A`.
    #[must_use]
    pub const fn extent_ok(self, extent: &ExtentRef) -> bool {
        extent.offset.is_multiple_of(self.offset) && extent.len.0.is_multiple_of(self.offset)
    }

    /// Whether a destination starts at a multiple of `M`.
    ///
    /// A `Raw` destination is checked by its address. A `Fixed` destination is checked by its
    /// offset inside the registered buffer; the buffer's own start is aligned to `M` by the tier
    /// manager, which registers it (RFC-0020; RFC-0021 §6, "what the tier manager guarantees").
    #[must_use]
    pub fn dest_ok(self, dest: &IoDest) -> bool {
        match dest.kind() {
            IoDestKind::Fixed { offset, .. } => offset.is_multiple_of(self.mem),
            IoDestKind::Raw { addr } => addr.is_multiple_of(self.mem),
        }
    }

    /// The pre-submission check of RFC-0021 §6: the extent is aligned to `A` and the memory to
    /// `M`.
    ///
    /// # Errors
    ///
    /// `IO_MISALIGNED` when either is not.
    pub fn check(self, op: &IoOp) -> Result<(), IoError> {
        if self.extent_ok(&op.extent()) && self.dest_ok(op.dest()) {
            Ok(())
        } else {
            Err(IoError::new(IoErrorCode::Misaligned))
        }
    }

    /// `n` rounded down to a multiple of both `A` and `M`, and at least that multiple: a chunk
    /// size that keeps every chunk of an aligned request aligned, in the file and in memory.
    #[must_use]
    pub fn chunk(self, n: u64) -> u64 {
        let unit = lcm(self.offset, self.mem);
        let down = n - n % unit;
        if down == 0 { unit } else { down }
    }

    /// Where the pack places a unit of `len` bytes whose predecessor ends at `cursor`
    /// (RFC-0021 §6, "what the pack guarantees", items 1 and 2; ADR-0005 item 3): it starts at
    /// the first multiple of `A_pack` at or after `cursor`, and its length is padded to a
    /// multiple of `A_pack`. `A_pack` is this `A`, which is 4 KiB or the volume's larger `A` at
    /// pack-build time. The next unit's cursor is the returned extent's end.
    ///
    /// `None` if the placement passes `u64::MAX`.
    #[must_use]
    pub fn pack_extent(
        self,
        file: syntropy_core::FileIx,
        cursor: u64,
        len: u64,
    ) -> Option<ExtentRef> {
        let offset = round_up(cursor, self.offset)?;
        let padded = round_up(len, self.offset)?;
        offset.checked_add(padded)?;
        Some(ExtentRef {
            file,
            offset,
            len: Bytes(padded),
        })
    }

    /// The aligned block range `[⌊off/A⌋·A, ⌈(off+len)/A⌉·A)` covering `extent` (RFC-0021 §6):
    /// how a row is read into a `RowCache` slot, and how an extent that is not packed is read
    /// around into an `IoStaging` slot.
    ///
    /// `None` if the range passes `u64::MAX`.
    #[must_use]
    pub fn window(self, extent: ExtentRef) -> Option<Window> {
        let start = extent.offset - extent.offset % self.offset;
        let end = round_up(extent.offset.checked_add(extent.len.0)?, self.offset)?;
        Some(Window {
            extent: ExtentRef {
                file: extent.file,
                offset: start,
                len: Bytes(end - start),
            },
            skip: extent.offset - start,
            payload: extent.len,
        })
    }
}

/// `n` rounded up to a multiple of `a`, or `None` on overflow.
const fn round_up(n: u64, a: u64) -> Option<u64> {
    match n % a {
        0 => Some(n),
        r => n.checked_add(a - r),
    }
}

/// An aligned read that covers a smaller byte range: a row, or an extent read around
/// (RFC-0021 §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Window {
    /// The aligned extent to read. It passes [`RequestAlign::extent_ok`].
    pub extent: ExtentRef,
    /// Where the wanted bytes start inside the window.
    pub skip: u64,
    /// How many bytes are wanted.
    pub payload: Bytes,
}

impl Window {
    /// The wanted bytes, out of the window's bytes once they are read. `None` if `staged` is
    /// shorter than the window.
    #[must_use]
    pub fn payload_in<'a>(&self, staged: &'a [u8]) -> Option<&'a [u8]> {
        let start = usize::try_from(self.skip).ok()?;
        let len = usize::try_from(self.payload.0).ok()?;
        staged.get(start..start.checked_add(len)?)
    }

    /// The read efficiency `len / window_len` (RFC-0021 §6); 1 for an empty window.
    #[must_use]
    pub fn efficiency(&self) -> f64 {
        if self.extent.len.0 == 0 {
            1.0
        } else {
            self.payload.0 as f64 / self.extent.len.0 as f64
        }
    }
}

/// One volume's read-around bytes: the payload wanted and the windows read (RFC-0021 §6;
/// ADR-0005 item 4).
///
/// Shared by whoever reads around on the volume (the store's source reads, COMP-E11-T08) and
/// read by whoever builds the volume's capability profile, so the counters are atomics and
/// [`ReadAroundStats::record`] takes `&self`. It never allocates.
#[derive(Debug, Default)]
pub struct ReadAroundStats {
    payload: AtomicU64,
    window: AtomicU64,
    reads: AtomicU64,
}

impl ReadAroundStats {
    /// Empty counters.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            payload: AtomicU64::new(0),
            window: AtomicU64::new(0),
            reads: AtomicU64::new(0),
        }
    }

    /// Counts one read-around.
    pub fn record(&self, w: &Window) {
        self.payload.fetch_add(w.payload.0, Ordering::Relaxed);
        self.window.fetch_add(w.extent.len.0, Ordering::Relaxed);
        self.reads.fetch_add(1, Ordering::Relaxed);
    }

    /// Read-arounds counted so far.
    #[must_use]
    pub fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    /// Payload bytes over window bytes, over every read-around so far; `None` before the first.
    #[must_use]
    pub fn efficiency(&self) -> Option<f32> {
        let window = self.window.load(Ordering::Relaxed);
        let payload = self.payload.load(Ordering::Relaxed);
        (window > 0).then(|| (payload as f64 / window as f64) as f32)
    }

    /// Writes the efficiency into the volume's capability profile
    /// ([`StorageCapsExt::read_around_efficiency`]), so the solver sees the cost (RFC-0021 §6).
    /// Leaves the field as it is before the first read-around.
    pub fn report(&self, caps: &mut StorageCapsExt) {
        if let Some(e) = self.efficiency() {
            caps.read_around_efficiency = Some(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syntropy_core::FileIx;

    fn ext(offset: u64, len: u64) -> ExtentRef {
        ExtentRef {
            file: FileIx(0),
            offset,
            len: Bytes(len),
        }
    }

    #[test]
    fn a_and_m_follow_section_6() {
        let a = RequestAlign::from_dio(DioAlign { mem: 0, offset: 0 }, 0);
        assert_eq!(a, RequestAlign::FLOOR);
        let a = RequestAlign::from_dio(
            DioAlign {
                mem: 4,
                offset: 512,
            },
            4096,
        );
        assert_eq!((a.offset(), a.mem()), (4096, 4096));
        let a = RequestAlign::from_dio(
            DioAlign {
                mem: MACOS_ALIGN,
                offset: MACOS_ALIGN,
            },
            16384,
        );
        assert_eq!((a.offset(), a.mem()), (16384, 16384));
        // A 64 KiB-page kernel raises M only.
        let a = RequestAlign::from_dio(
            DioAlign {
                mem: 512,
                offset: 512,
            },
            65536,
        );
        assert_eq!((a.offset(), a.mem()), (4096, 65536));
        // Not a power of two: the least common multiple.
        let a = RequestAlign::from_dio(
            DioAlign {
                mem: 0,
                offset: 6144,
            },
            4096,
        );
        assert_eq!(a.offset(), 12288);
    }

    #[test]
    fn a_row_reads_as_its_aligned_block_range() {
        let a = RequestAlign::FLOOR;
        // A 1.4 KB row across a block boundary reads two blocks (RFC-0021 §Performance).
        let w = a.window(ext(4096 * 3 - 700, 1400)).unwrap();
        assert_eq!(w.extent, ext(4096 * 2, 8192));
        assert_eq!(w.skip, 4096 - 700);
        assert!(a.extent_ok(&w.extent));
        let staged: Vec<u8> = (0..8192u32).map(|i| i as u8).collect();
        assert_eq!(w.payload_in(&staged).unwrap(), &staged[3396..3396 + 1400]);
        assert!(w.payload_in(&staged[..4000]).is_none());
        // An aligned extent is its own window.
        let w = a.window(ext(8192, 4096)).unwrap();
        assert_eq!(
            (w.extent, w.skip, w.efficiency()),
            (ext(8192, 4096), 0, 1.0)
        );
        assert!(a.window(ext(u64::MAX - 10, 20)).is_none());
    }

    #[test]
    fn read_around_efficiency_reaches_the_profile() {
        let stats = ReadAroundStats::new();
        let mut caps = StorageCapsExt::default();
        stats.report(&mut caps);
        assert_eq!(caps.read_around_efficiency, None, "nothing read around yet");
        let a = RequestAlign::FLOOR;
        // 1 KiB wanted from one 4 KiB block, then 6 KiB from three blocks: 7 of 16 KiB.
        stats.record(&a.window(ext(100, 1024)).unwrap());
        stats.record(&a.window(ext(8192 + 3072, 6144)).unwrap());
        assert_eq!(stats.reads(), 2);
        stats.report(&mut caps);
        assert_eq!(caps.read_around_efficiency, Some(7.0 / 16.0));
    }

    #[test]
    fn chunks_stay_aligned() {
        let a = RequestAlign::from_dio(
            DioAlign {
                mem: 0,
                offset: 65536,
            },
            4096,
        );
        assert_eq!(a.chunk(4 << 20), 4 << 20);
        assert_eq!(a.chunk(100_000), 65536);
        assert_eq!(a.chunk(1000), 65536);
    }

    #[test]
    fn this_hosts_query_answers() {
        let path = std::env::temp_dir().join(format!("syntropy-align-{}", std::process::id()));
        std::fs::write(&path, [0u8; 4096]).unwrap();
        let file = File::open(&path).unwrap();
        let q = query(&file, &path);
        eprintln!("alignment query on {}: {q:?}", path.display());
        // A broken FFI must not pass: on every OS with a query, a file on a real volume gets a
        // concrete answer. `Unknown` is right only on Linux, for a file with no block device
        // under it (tmpfs, ramfs) whose file system fills no `STATX_DIOALIGN`.
        #[cfg(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ))]
        if q == DioQuery::Unknown {
            assert!(
                !linux::on_block_device(&file),
                "Unknown for a file on a block device"
            );
        }
        let _ = std::fs::remove_file(&path);
        #[cfg(target_os = "macos")]
        assert_eq!(
            q,
            DioQuery::Aligned {
                dio: DioAlign {
                    mem: MACOS_ALIGN,
                    offset: MACOS_ALIGN
                },
                source: DioSource::MacPage
            }
        );
        #[cfg(windows)]
        assert!(
            matches!(q, DioQuery::Aligned { .. }),
            "Windows reports sector sizes for every local volume: {q:?}"
        );
        if let DioQuery::Aligned { dio, .. } = q {
            assert!(
                dio.offset > 0 || dio.mem > 0,
                "an Aligned answer is non-zero"
            );
            assert!(dio.offset == 0 || dio.offset.is_power_of_two());
            assert!(dio.mem == 0 || dio.mem.is_power_of_two());
        }
        let page = os_page_size();
        assert!(page.is_power_of_two() && page >= 4096, "page size {page}");
    }
}
