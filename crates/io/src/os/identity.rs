// SPDX-License-Identifier: MIT OR Apache-2.0

//! A file's identity on its volume, for the local index's (device, inode, size, mtime) record
//! (RFC-0041 §4.3; COMP-E11-T03).
//!
//! `store` may not use `unsafe` (RFC-0001 §4.2 rule 8), and the Windows identity needs a system
//! call, so [`file_identity`] lives here behind a safe function:
//!
//! - On Unix it is the `st_dev` and `st_ino` of the open file.
//! - On Windows it is the volume serial number and the 64-bit file index that
//!   `GetFileInformationByHandle` reports for the open handle. On NTFS the index is unique on its
//!   volume. On ReFS a file ID is 128 bits and the 64-bit index may not be unique, so there the
//!   size and modification time in the same record do more of the work.
//!
//! Other targets return [`io::ErrorKind::Unsupported`].

use std::fs::File;
use std::io;

/// Which file an open file is: its volume and its number on that volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileIdentity {
    /// The device (Unix `st_dev`) or volume serial number (Windows).
    pub device: u64,
    /// The inode (Unix `st_ino`) or file index (Windows).
    pub inode: u64,
}

/// The identity of the file `file` is open on. Two handles on one file, or on two hard links to
/// it, give the same identity; two different files on one volume give different ones.
///
/// # Errors
///
/// The OS error when the file's information cannot be read, and
/// [`io::ErrorKind::Unsupported`] on a target with neither Unix nor Windows file identities.
pub fn file_identity(file: &File) -> io::Result<FileIdentity> {
    imp::file_identity(file)
}

#[cfg(unix)]
mod imp {
    use std::fs::File;
    use std::io;
    use std::os::unix::fs::MetadataExt;

    use super::FileIdentity;

    pub(super) fn file_identity(file: &File) -> io::Result<FileIdentity> {
        let metadata = file.metadata()?;
        Ok(FileIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
}

#[cfg(windows)]
mod imp {
    use std::fs::File;
    use std::io;
    use std::os::windows::io::AsRawHandle;

    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
    };

    use super::FileIdentity;

    pub(super) fn file_identity(file: &File) -> io::Result<FileIdentity> {
        let mut info = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::uninit();
        // SAFETY: `file` keeps its handle open for the call, and `info` is a live output of the
        // struct's size.
        if unsafe { GetFileInformationByHandle(file.as_raw_handle().cast(), info.as_mut_ptr()) }
            == 0
        {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call succeeded, so it filled the whole struct.
        let info = unsafe { info.assume_init() };
        Ok(FileIdentity {
            device: u64::from(info.dwVolumeSerialNumber),
            inode: u64::from(info.nFileIndexHigh) << 32 | u64::from(info.nFileIndexLow),
        })
    }
}

#[cfg(not(any(unix, windows)))]
mod imp {
    use std::fs::File;
    use std::io;

    use super::FileIdentity;

    pub(super) fn file_identity(_file: &File) -> io::Result<FileIdentity> {
        Err(io::ErrorKind::Unsupported.into())
    }
}

#[cfg(all(test, any(unix, windows)))]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "syntropy-io-identity-{}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn one_file_has_one_identity_and_two_files_two() {
        let dir = scratch("basic");
        let (a, b) = (dir.join("a"), dir.join("b"));
        std::fs::write(&a, b"one").unwrap();
        std::fs::write(&b, b"two").unwrap();
        let first = file_identity(&File::open(&a).unwrap()).unwrap();
        let again = file_identity(&File::open(&a).unwrap()).unwrap();
        let other = file_identity(&File::open(&b).unwrap()).unwrap();
        assert_eq!(first, again);
        assert_eq!(first.device, other.device);
        assert_ne!(first.inode, other.inode);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_hard_link_is_the_same_file_and_a_copy_is_not() {
        let dir = scratch("links");
        let (a, link, copy) = (dir.join("a"), dir.join("link"), dir.join("copy"));
        std::fs::write(&a, b"bytes").unwrap();
        std::fs::hard_link(&a, &link).unwrap();
        std::fs::copy(&a, &copy).unwrap();
        let id = |p: &PathBuf| file_identity(&File::open(p).unwrap()).unwrap();
        assert_eq!(id(&a), id(&link));
        assert_ne!(id(&a), id(&copy));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
