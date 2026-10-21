// SPDX-License-Identifier: MIT OR Apache-2.0

//! The filesystem view enumeration reads through.
//!
//! Enumeration only reads files, so one trait covers the live `/` and a recorded fixture
//! tree. Paths are relative to the view's root and use `/` separators, e.g. `proc/meminfo`.

use std::io;
use std::path::PathBuf;

/// Read-only access to a host's `/proc`, `/sys` and the few other paths E1 reads.
pub trait HostFs {
    /// The whole file as UTF-8 text.
    fn read(&self, path: &str) -> io::Result<String>;
    /// Whether anything exists at `path`.
    fn exists(&self, path: &str) -> bool;
    /// The names of a directory's entries, sorted.
    fn list(&self, path: &str) -> io::Result<Vec<String>>;
}

/// A [`HostFs`] rooted at a directory: `/` for the live host, or a fixture tree.
#[derive(Clone, Debug)]
pub struct DirFs {
    root: PathBuf,
}

impl DirFs {
    /// A view whose `proc/meminfo` is `<root>/proc/meminfo`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path(&self, path: &str) -> PathBuf {
        self.root.join(path)
    }
}

impl HostFs for DirFs {
    fn read(&self, path: &str) -> io::Result<String> {
        std::fs::read_to_string(self.path(path))
    }

    fn exists(&self, path: &str) -> bool {
        self.path(path).exists()
    }

    fn list(&self, path: &str) -> io::Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(self.path(path))? {
            names.push(entry?.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        Ok(names)
    }
}
