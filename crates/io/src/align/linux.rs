// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Linux alignment query (RFC-0021 §6).
//!
//! `statx` with `STATX_DIOALIGN` (Linux 6.1 and later) reports `stx_dio_mem_align` and
//! `stx_dio_offset_align` for the file. When the kernel or the file system does not fill that
//! field, the fallback is the logical block size of the block device that holds the file,
//! read from sysfs (`/sys/dev/block/<major>:<minor>/queue/logical_block_size`, or the parent
//! device's for a partition). `statx` is called through `syscall`, so the query does not
//! depend on the C library's own `statx` wrapper, which older glibc and musl releases lack.

use std::ffi::{c_char, c_int, c_long, c_void};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;

use syntropy_core::resource::DioAlign;

use super::{DioQuery, DioSource};

mod sys {
    use std::ffi::c_long;

    #[cfg(target_arch = "x86_64")]
    pub const SYS_STATX: c_long = 332;
    #[cfg(target_arch = "aarch64")]
    pub const SYS_STATX: c_long = 291;

    pub const AT_EMPTY_PATH: c_long = 0x1000;
    pub const STATX_DIOALIGN: u32 = 0x2000;

    pub const SC_PAGESIZE: std::ffi::c_int = 30;

    /// `struct statx` from `<linux/stat.h>`: 256 bytes, of which this reads `stx_mask` and the
    /// two direct-I/O fields at bytes 152 and 156.
    #[repr(C)]
    pub struct Statx {
        pub stx_mask: u32,
        pub head: [u32; 37],
        pub stx_dio_mem_align: u32,
        pub stx_dio_offset_align: u32,
        pub tail: [u64; 12],
    }

    const _: () = assert!(std::mem::size_of::<Statx>() == 256);
    const _: () = assert!(std::mem::offset_of!(Statx, stx_dio_mem_align) == 152);
    const _: () = assert!(std::mem::offset_of!(Statx, stx_dio_offset_align) == 156);

    unsafe extern "C" {
        pub fn syscall(num: c_long, ...) -> c_long;
    }
}

/// See the module documentation.
pub(super) fn query(file: &File) -> DioQuery {
    if let Some(q) = statx_dio(file) {
        return q;
    }
    match logical_block_size(file) {
        Some(n) => DioQuery::Aligned {
            dio: DioAlign { mem: n, offset: n },
            source: DioSource::LogicalBlockSize,
        },
        None => DioQuery::Unknown,
    }
}

/// `statx(STATX_DIOALIGN)`'s answer, or `None` when the call fails or leaves the field unset.
fn statx_dio(file: &File) -> Option<DioQuery> {
    let mut st = sys::Statx {
        stx_mask: 0,
        head: [0; 37],
        stx_dio_mem_align: 0,
        stx_dio_offset_align: 0,
        tail: [0; 12],
    };
    let empty: [c_char; 1] = [0];
    // SAFETY: `statx(fd, "", AT_EMPTY_PATH, mask, buf)` reads the NUL-terminated empty path and
    // writes at most `sizeof(struct statx)`, 256 bytes, into `st`, which is that size and
    // writable; the fd is open for the call. Every argument is passed as a `long`, as the
    // syscall ABI reads them.
    let rc = unsafe {
        sys::syscall(
            sys::SYS_STATX,
            c_long::from(file.as_raw_fd()),
            empty.as_ptr(),
            sys::AT_EMPTY_PATH,
            c_long::from(sys::STATX_DIOALIGN),
            (&raw mut st).cast::<c_void>(),
        )
    };
    if rc != 0 || st.stx_mask & sys::STATX_DIOALIGN == 0 {
        return None;
    }
    if st.stx_dio_mem_align == 0 && st.stx_dio_offset_align == 0 {
        // The file system answered: no direct I/O here.
        return Some(DioQuery::Unsupported);
    }
    Some(DioQuery::Aligned {
        dio: DioAlign {
            mem: st.stx_dio_mem_align,
            offset: st.stx_dio_offset_align,
        },
        source: DioSource::Statx,
    })
}

/// The logical block size of the block device holding `file`, from sysfs.
fn logical_block_size(file: &File) -> Option<u32> {
    let dev = file.metadata().ok()?.dev();
    // The C library's `major` and `minor` encoding of `dev_t`.
    let major = ((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff);
    let minor = (dev & 0xff) | ((dev >> 12) & !0xff);
    let base = format!("/sys/dev/block/{major}:{minor}");
    // A whole device has `queue/`; a partition's is its parent's.
    [
        format!("{base}/queue/logical_block_size"),
        format!("{base}/../queue/logical_block_size"),
    ]
    .iter()
    .find_map(|p| std::fs::read_to_string(p).ok())
    .and_then(|s| s.trim().parse::<u32>().ok())
    .filter(|&n| n > 0)
}

/// The page size, from `sysconf(_SC_PAGESIZE)`.
pub(super) fn page_size() -> u64 {
    super::unix_page_size(sys::SC_PAGESIZE as c_int)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fallback on its own, for the file system `TMPDIR` names: a power of two when a block
    /// device answers, `None` on file systems with no block device (tmpfs, overlay over one).
    #[test]
    fn the_logical_block_size_fallback_answers_or_declines() {
        let path = std::env::temp_dir().join(format!("syntropy-lbs-{}", std::process::id()));
        std::fs::write(&path, [0u8; 512]).unwrap();
        let file = File::open(&path).unwrap();
        let n = logical_block_size(&file);
        eprintln!(
            "logical block size for {}: {n:?}; statx: {:?}",
            path.display(),
            statx_dio(&file)
        );
        let _ = std::fs::remove_file(&path);
        assert!(n.is_none_or(u32::is_power_of_two));
    }
}
