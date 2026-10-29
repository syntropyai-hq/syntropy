// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sparse files: marking a file sparse, punching a hole in it, and the bytes it holds on disk,
//! for the store's sparse source blobs (RFC-0041 §4.2, §4.4; COMP-E11-T02).
//!
//! `store` may not use `unsafe` (RFC-0001 §4.2 rule 8), and each of these needs a system call on
//! some OS, so they live here behind safe functions:
//!
//! - [`set_sparse`] marks a file sparse. Windows needs `FSCTL_SET_SPARSE` before a file can have
//!   holes. On Linux and macOS a file extended with `ftruncate` is sparse already, so it does
//!   nothing there.
//! - [`punch_hole`] frees the disk blocks under a byte range: `fallocate` with
//!   `FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE` on Linux, `F_PUNCHHOLE` on macOS and
//!   `FSCTL_SET_ZERO_DATA` on Windows. The file keeps its length, and no byte outside the range
//!   changes. Bytes inside it read as zero afterwards, except on macOS, where `F_PUNCHHOLE` takes
//!   only whole blocks of the file system: the range is shrunk to the blocks that lie wholly
//!   inside it, and a partial block at either end keeps its bytes. The store never reads a
//!   punched range, whatever it holds: the inventory bitmap says which ranges are valid.
//! - [`allocated_bytes`] is the bytes the file holds on disk, which for a sparse file is less
//!   than its length: `st_blocks × 512` on Unix, and the allocation size that
//!   `GetFileInformationByHandleEx(FileStandardInfo)` reports on Windows.
//!
//! Other targets return [`io::ErrorKind::Unsupported`].

use std::fs::File;
use std::io;

/// Marks `file` sparse, so that a range never written, or punched with [`punch_hole`], holds no
/// disk blocks. `file` must be open for writing.
///
/// # Errors
///
/// The OS error, such as [`io::ErrorKind::Unsupported`] or an invalid-function error on a
/// Windows volume without sparse files (FAT32, exFAT).
pub fn set_sparse(file: &File) -> io::Result<()> {
    imp::set_sparse(file)
}

/// Frees the disk blocks under `[offset, offset + len)` of `file`, which must be open for
/// writing, and on Windows marked with [`set_sparse`]. The file's length does not change, and no
/// byte outside the range changes. See the module documentation for what the range reads as
/// afterwards.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`] for a range past what the OS can address, and the OS error,
/// such as [`io::ErrorKind::Unsupported`] on a file system without hole punching.
pub fn punch_hole(file: &File, offset: u64, len: u64) -> io::Result<()> {
    if len == 0 {
        return Ok(());
    }
    let end = offset
        .checked_add(len)
        .filter(|&end| i64::try_from(end).is_ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "range past i64::MAX"))?;
    imp::punch_hole(file, offset, end)
}

/// The bytes `file` holds on disk: its allocated blocks, not its length.
///
/// # Errors
///
/// The OS error when the file's information cannot be read.
pub fn allocated_bytes(file: &File) -> io::Result<u64> {
    imp::allocated_bytes(file)
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
fn unix_allocated(file: &File) -> io::Result<u64> {
    use std::os::unix::fs::MetadataExt;
    // `st_blocks` counts 512-byte units on Linux and macOS, whatever the file system's block.
    Ok(file.metadata()?.blocks().saturating_mul(512))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod imp {
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;

    pub(super) fn set_sparse(_file: &File) -> io::Result<()> {
        Ok(())
    }

    /// `end` fits in `i64`: [`super::punch_hole`] checked it.
    pub(super) fn punch_hole(file: &File, offset: u64, end: u64) -> io::Result<()> {
        let start = libc::off_t::try_from(offset).expect("checked by the caller");
        let len = libc::off_t::try_from(end - offset).expect("checked by the caller");
        // SAFETY: `file` keeps its descriptor open for the call, which reads only its scalar
        // arguments.
        let rc = unsafe {
            libc::fallocate(
                file.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                start,
                len,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    pub(super) fn allocated_bytes(file: &File) -> io::Result<u64> {
        super::unix_allocated(file)
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;

    pub(super) fn set_sparse(_file: &File) -> io::Result<()> {
        Ok(())
    }

    /// `end` fits in `i64`: [`super::punch_hole`] checked it.
    pub(super) fn punch_hole(file: &File, offset: u64, end: u64) -> io::Result<()> {
        let mut fs = std::mem::MaybeUninit::<libc::statfs>::uninit();
        // SAFETY: `file` keeps its descriptor open for the call, and `fs` is a live output of
        // the struct's size.
        if unsafe { libc::fstatfs(file.as_raw_fd(), fs.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call succeeded, so it filled the whole struct.
        let block = u64::from(unsafe { fs.assume_init() }.f_bsize).max(1);
        // `F_PUNCHHOLE` takes whole blocks: shrink the range to the blocks inside it.
        let first = offset.div_ceil(block) * block;
        let last = end / block * block;
        if first >= last {
            return Ok(());
        }
        let hole = libc::fpunchhole_t {
            fp_flags: 0,
            reserved: 0,
            fp_offset: libc::off_t::try_from(first).expect("checked by the caller"),
            fp_length: libc::off_t::try_from(last - first).expect("checked by the caller"),
        };
        // SAFETY: `file` keeps its descriptor open for the call, and `hole` is a live input of
        // the type `F_PUNCHHOLE` reads.
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_PUNCHHOLE, &raw const hole) };
        if rc == -1 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    pub(super) fn allocated_bytes(file: &File) -> io::Result<u64> {
        super::unix_allocated(file)
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::windows::io::AsRawHandle;

    use windows_sys::Win32::Storage::FileSystem::{
        FILE_STANDARD_INFO, FileStandardInfo, GetFileInformationByHandleEx,
    };
    use windows_sys::Win32::System::IO::DeviceIoControl;
    use windows_sys::Win32::System::Ioctl::{
        FILE_ZERO_DATA_INFORMATION, FSCTL_SET_SPARSE, FSCTL_SET_ZERO_DATA,
    };

    /// `DeviceIoControl` with `input`, no output and no overlapped I/O.
    fn control<T>(file: &File, code: u32, input: Option<&T>) -> io::Result<()> {
        let (ptr, len) = match input {
            Some(input) => (
                std::ptr::from_ref(input).cast::<c_void>(),
                u32::try_from(size_of::<T>()).expect("a small struct"),
            ),
            None => (std::ptr::null(), 0),
        };
        let mut returned = 0u32;
        // SAFETY: `file` keeps its handle open for the call; `ptr` is null with length 0, or
        // points at a live `T` of `len` bytes that the call only reads; there is no output
        // buffer, `returned` is a live output, and the call is synchronous (no OVERLAPPED).
        let ok = unsafe {
            DeviceIoControl(
                file.as_raw_handle().cast(),
                code,
                ptr,
                len,
                std::ptr::null_mut(),
                0,
                &raw mut returned,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    pub(super) fn set_sparse(file: &File) -> io::Result<()> {
        // No input buffer: the file is marked sparse.
        control::<()>(file, FSCTL_SET_SPARSE, None)
    }

    /// `end` fits in `i64`: [`super::punch_hole`] checked it.
    pub(super) fn punch_hole(file: &File, offset: u64, end: u64) -> io::Result<()> {
        let range = FILE_ZERO_DATA_INFORMATION {
            FileOffset: i64::try_from(offset).expect("checked by the caller"),
            BeyondFinalZero: i64::try_from(end).expect("checked by the caller"),
        };
        control(file, FSCTL_SET_ZERO_DATA, Some(&range))
    }

    pub(super) fn allocated_bytes(file: &File) -> io::Result<u64> {
        let mut info = std::mem::MaybeUninit::<FILE_STANDARD_INFO>::uninit();
        // SAFETY: `file` keeps its handle open for the call, and `info` is a live output of the
        // size passed.
        let ok = unsafe {
            GetFileInformationByHandleEx(
                file.as_raw_handle().cast(),
                FileStandardInfo,
                info.as_mut_ptr().cast(),
                u32::try_from(size_of::<FILE_STANDARD_INFO>()).expect("a small struct"),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call succeeded, so it filled the whole struct.
        let info = unsafe { info.assume_init() };
        Ok(u64::try_from(info.AllocationSize).unwrap_or(0))
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    windows
)))]
mod imp {
    use std::fs::File;
    use std::io;

    pub(super) fn set_sparse(_file: &File) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }

    pub(super) fn punch_hole(_file: &File, _offset: u64, _end: u64) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }

    pub(super) fn allocated_bytes(_file: &File) -> io::Result<u64> {
        Err(io::ErrorKind::Unsupported.into())
    }
}

#[cfg(all(
    test,
    any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        windows
    )
))]
mod tests {
    use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
    use std::path::PathBuf;

    use super::*;

    /// A fresh directory under the system's temporary directory, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let dir = std::env::temp_dir()
                .join(format!("syntropy-io-sparse-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const MIB: u64 = 1 << 20;

    /// A sparse file of `len` bytes, open for reading and writing.
    fn sparse_file(path: &std::path::Path, len: u64) -> File {
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        set_sparse(&file).unwrap();
        file.set_len(len).unwrap();
        file
    }

    fn write_at(mut file: &File, offset: u64, bytes: &[u8]) {
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
    }

    fn read_at(mut file: &File, offset: u64, len: usize) -> Vec<u8> {
        let mut out = vec![0; len];
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.read_exact(&mut out).unwrap();
        out
    }

    /// A file extended past what it holds keeps its length and holds less than it.
    #[test]
    fn an_extended_sparse_file_holds_less_than_its_length() {
        let dir = Scratch::new("extend");
        let file = sparse_file(&dir.0.join("f"), 64 * MIB);
        write_at(&file, 8 * MIB, &[7u8; 4096]);
        assert_eq!(file.metadata().unwrap().len(), 64 * MIB);
        let held = allocated_bytes(&file).unwrap();
        assert!(held > 0, "the written block is held");
        assert!(held < 8 * MIB, "{held} bytes held for 4 KiB written");
    }

    /// Punching frees the blocks under the range, keeps the length, and leaves every byte
    /// outside the range as it was, with ranges that do not start or end on a block.
    #[test]
    fn a_punched_range_frees_its_blocks_and_nothing_outside_it_changes() {
        let dir = Scratch::new("punch");
        let len = 16 * MIB;
        let file = sparse_file(&dir.0.join("f"), len);
        let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8 + 1).collect();
        write_at(&file, 0, &bytes);
        let before = allocated_bytes(&file).unwrap();
        assert!(before >= len, "{before} bytes held for {len} written");

        // An unaligned range: 4 MiB + 100 to 12 MiB - 100.
        let (start, end) = (4 * MIB + 100, 12 * MIB - 100);
        punch_hole(&file, start, end - start).unwrap();
        file.sync_all().unwrap();
        assert_eq!(file.metadata().unwrap().len(), len);
        let after = allocated_bytes(&file).unwrap();
        assert!(
            after <= before - 7 * MIB,
            "{before} bytes held before the punch, {after} after"
        );
        let now = read_at(&file, 0, usize::try_from(len).unwrap());
        let (s, e) = (
            usize::try_from(start).unwrap(),
            usize::try_from(end).unwrap(),
        );
        assert_eq!(now[..s], bytes[..s], "bytes before the range changed");
        assert_eq!(now[e..], bytes[e..], "bytes after the range changed");
        // Inside the range, every whole block reads as zero on every OS.
        let block = 64 * 1024;
        let first = start.div_ceil(block) * block;
        let last = end / block * block;
        let inner = &now[usize::try_from(first).unwrap()..usize::try_from(last).unwrap()];
        assert!(inner.iter().all(|&b| b == 0), "a punched block is not zero");
    }

    /// TEMPORARY, for the review of #1965: how APFS allocates a file extended with `ftruncate`
    /// and written past a gap. It fails on purpose so the hosted macOS job prints the table.
    #[cfg(target_os = "macos")]
    #[test]
    fn apfs_allocation_table() {
        let dir = Scratch::new("apfs");
        let mut rows = Vec::new();
        let mut n = 0;
        let mut case = |what: &str, f: &dyn Fn(&File)| {
            n += 1;
            let file = File::options()
                .read(true)
                .write(true)
                .create_new(true)
                .open(dir.0.join(format!("f{n}")))
                .unwrap();
            f(&file);
            file.sync_all().unwrap();
            rows.push(format!(
                "{what}: len {} allocated {}",
                file.metadata().unwrap().len(),
                allocated_bytes(&file).unwrap()
            ));
        };
        let kib4 = [7u8; 4096];
        case("A set_len 64M, write 4K @ 8M", &|f| {
            f.set_len(64 * MIB).unwrap();
            write_at(f, 8 * MIB, &kib4);
        });
        case("B write 4K @ 8M past EOF, then set_len 64M", &|f| {
            write_at(f, 8 * MIB, &kib4);
            f.set_len(64 * MIB).unwrap();
        });
        case("C set_len 64M, write 4K @ 32M", &|f| {
            f.set_len(64 * MIB).unwrap();
            write_at(f, 32 * MIB, &kib4);
        });
        case("D set_len 64M, write 4K @ 1M", &|f| {
            f.set_len(64 * MIB).unwrap();
            write_at(f, MIB, &kib4);
        });
        case("E as A, then punch [0, 8M)", &|f| {
            f.set_len(64 * MIB).unwrap();
            write_at(f, 8 * MIB, &kib4);
            punch_hole(f, 0, 8 * MIB).unwrap();
        });
        case("F set_len 1G, write 1M @ 512M", &|f| {
            f.set_len(1 << 30).unwrap();
            write_at(f, 512 * MIB, &vec![9u8; 1 << 20]);
        });
        case("G set_len 64M, write 4K @ 8M, then 4K @ 32M", &|f| {
            f.set_len(64 * MIB).unwrap();
            write_at(f, 8 * MIB, &kib4);
            write_at(f, 32 * MIB, &kib4);
        });
        case("H write 4K @ 0, set_len 64M, write 4K @ 32M", &|f| {
            write_at(f, 0, &kib4);
            f.set_len(64 * MIB).unwrap();
            write_at(f, 32 * MIB, &kib4);
        });
        case(
            "I set_len 64M, write 4K @ 8M without sync between, then punch [0, 8M)",
            &|f| {
                f.set_len(64 * MIB).unwrap();
                let mut g = f;
                g.seek(SeekFrom::Start(8 * MIB)).unwrap();
                g.write_all(&kib4).unwrap();
                punch_hole(f, 0, 8 * MIB).unwrap();
            },
        );
        let mut fs = std::mem::MaybeUninit::<libc::statfs>::uninit();
        let probe = File::open(&dir.0).unwrap();
        // SAFETY: test only; `fs` is a live output of the struct's size.
        let rc = unsafe { libc::fstatfs(std::os::fd::AsRawFd::as_raw_fd(&probe), fs.as_mut_ptr()) };
        assert_eq!(rc, 0);
        // SAFETY: the call succeeded.
        let fs = unsafe { fs.assume_init() };
        let name: String = fs
            .f_fstypename
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8 as char)
            .collect();
        rows.push(format!("fs {name}, f_bsize {}", fs.f_bsize));
        panic!("APFS allocation table:\n{}", rows.join("\n"));
    }

    #[test]
    fn an_empty_punch_does_nothing_and_a_huge_one_is_refused() {
        let dir = Scratch::new("edges");
        let file = sparse_file(&dir.0.join("f"), MIB);
        write_at(&file, 0, &[9u8; 4096]);
        punch_hole(&file, 0, 0).unwrap();
        assert_eq!(read_at(&file, 0, 4096), vec![9u8; 4096]);
        let error = punch_hole(&file, u64::MAX - 1, 2).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }
}
