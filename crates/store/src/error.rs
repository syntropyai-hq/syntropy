// SPDX-License-Identifier: MIT OR Apache-2.0

//! Errors from opening store roots.

use std::fmt;
use std::io;
use std::path::PathBuf;

/// Why a store root could not be opened (RFC-0041 §4.1).
#[non_exhaustive]
#[derive(Debug)]
pub enum StoreError {
    /// Another process holds the root's writer lock, so this one cannot write to the root.
    WriterLocked {
        /// The root.
        root: PathBuf,
        /// The holder's process ID, as it recorded in the lock file, when it could be read.
        holder_pid: Option<u32>,
    },
    /// A path that must be a directory exists and is something else.
    NotADirectory {
        /// The path.
        path: PathBuf,
    },
    /// A read-only root has no `v1/` directory, so it holds no store.
    NoLayout {
        /// The root.
        root: PathBuf,
    },
    /// The same directory is listed more than once among the writable and read-only roots.
    DuplicateRoot {
        /// The directory, resolved.
        path: PathBuf,
    },
    /// A file-system operation on a root failed.
    Io {
        /// What was being done, such as `"create directory"`.
        op: &'static str,
        /// The path it was done to.
        path: PathBuf,
        /// The error the OS returned.
        source: io::Error,
    },
}

impl StoreError {
    pub(crate) fn io(op: &'static str, path: impl Into<PathBuf>, source: io::Error) -> Self {
        StoreError::Io {
            op,
            path: path.into(),
            source,
        }
    }
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::WriterLocked { root, holder_pid } => {
                write!(
                    f,
                    "store root {} is already open for writing by another process",
                    root.display()
                )?;
                if let Some(pid) = holder_pid {
                    write!(f, " (pid {pid})")?;
                }
                f.write_str("; a store root has one writer process at a time")
            }
            StoreError::NotADirectory { path } => {
                write!(f, "{} is not a directory", path.display())
            }
            StoreError::NoLayout { root } => write!(
                f,
                "read-only store root {} has no v1/ directory",
                root.display()
            ),
            StoreError::DuplicateRoot { path } => write!(
                f,
                "store root {} is listed more than once in store.dirs and store.readonly_dirs",
                path.display()
            ),
            StoreError::Io { op, path, source } => {
                write!(f, "cannot {op} {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StoreError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
