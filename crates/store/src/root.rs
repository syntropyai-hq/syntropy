// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opening store roots (RFC-0041 §4.1, ADR-0016 decision 7).
//!
//! `store.dirs` lists writable roots and `store.readonly_dirs` lists roots that are only read,
//! such as a shared NAS mount. A [`WritableRoot`] holds the root's writer lock for as long as
//! it lives, so one process at a time writes to a root. A [`ReadOnlyRoot`] never writes, never
//! creates and takes no lock. [`StoreRoots`] opens both lists together.
//!
//! # The writer lock
//!
//! The lock is an exclusive OS advisory lock (`flock` on Linux and macOS, `LockFileEx` on
//! Windows, through [`File::try_lock`]) on `<root>/v1/locks/writer.lock`. The OS releases it
//! when the holder closes the file, which happens when its [`WritableRoot`] is dropped or when
//! the process exits or is killed, so a crashed daemon never leaves a root locked. The file is
//! opened close-on-exec (the standard library's default), so a child the holder spawns does not
//! inherit the lock. A process that finds the lock held fails at once with
//! [`StoreError::WriterLocked`]; it never waits.
//!
//! The holder writes its process ID into the lock file, and a refused process reports it. On
//! Windows the lock also blocks reading the file, so the ID is not reported there.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::error::StoreError;
use crate::layout::Layout;

/// A writable store root, holding the root's writer lock until it is dropped.
///
/// Opening one creates the root directory if it is missing, then the layout's fixed
/// directories ([`Layout::fixed_dirs`]). Paths that take a caller-supplied name, such as
/// `packs/<device-class>/`, are created by the code that writes them.
#[derive(Debug)]
pub struct WritableRoot {
    layout: Layout,
    /// Holds the advisory lock; closing it releases the lock.
    _lock: File,
}

impl WritableRoot {
    /// Opens the root at `path` for writing.
    ///
    /// # Errors
    ///
    /// - [`StoreError::WriterLocked`] when another process holds the root's writer lock.
    /// - [`StoreError::NotADirectory`] when `path`, or a directory of its layout, exists and is
    ///   not a directory.
    /// - [`StoreError::Io`] when a directory cannot be created, or the lock file cannot be
    ///   opened, locked or written.
    pub fn open(path: impl AsRef<Path>) -> Result<WritableRoot, StoreError> {
        let root = prepare_writable(path.as_ref())?;
        WritableRoot::open_resolved(root)
    }

    /// Takes the lock on a root that [`prepare_writable`] resolved, then creates the layout.
    fn open_resolved(root: PathBuf) -> Result<WritableRoot, StoreError> {
        let layout = Layout::new(root);
        // `locks/` must exist before the lock can be taken. Every other directory is created
        // under the lock, by its holder only.
        ensure_dir(layout.base())?;
        ensure_dir(&layout.locks_dir())?;

        let lock_path = layout.writer_lock();
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            // The holder's ID must survive a refused open of the same file.
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| StoreError::io("open", &lock_path, e))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(StoreError::WriterLocked {
                    root: layout.root().to_owned(),
                    holder_pid: read_holder_pid(&lock_path),
                });
            }
            Err(TryLockError::Error(e)) => return Err(StoreError::io("lock", &lock_path, e)),
        }
        record_holder(&lock).map_err(|e| StoreError::io("write", &lock_path, e))?;

        for dir in layout.fixed_dirs() {
            ensure_dir(&dir)?;
        }
        Ok(WritableRoot {
            layout,
            _lock: lock,
        })
    }

    /// The root's layout. Its root path is the resolved form of the path it was opened with.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The root directory, resolved.
    pub fn path(&self) -> &Path {
        self.layout.root()
    }
}

/// A read-only store root.
///
/// Opening one only reads metadata: it checks that the root and its `v1/` directory exist and
/// are directories, and resolves the root's path. It creates nothing, writes nothing and takes
/// no lock, so a root on a read-only mount, or one another node's daemon writes to, can be
/// opened. Reads through its [`Layout`] are the caller's.
#[derive(Debug)]
pub struct ReadOnlyRoot {
    layout: Layout,
}

impl ReadOnlyRoot {
    /// Opens the root at `path` for reading.
    ///
    /// # Errors
    ///
    /// - [`StoreError::NoLayout`] when the root has no `v1/` directory.
    /// - [`StoreError::NotADirectory`] when `path` or `<path>/v1` is not a directory.
    /// - [`StoreError::Io`] when `path` does not exist or cannot be read.
    pub fn open(path: impl AsRef<Path>) -> Result<ReadOnlyRoot, StoreError> {
        let root = resolve_readonly(path.as_ref())?;
        ReadOnlyRoot::open_resolved(root)
    }

    fn open_resolved(root: PathBuf) -> Result<ReadOnlyRoot, StoreError> {
        let layout = Layout::new(root);
        match fs::metadata(layout.base()) {
            Ok(meta) if meta.is_dir() => Ok(ReadOnlyRoot { layout }),
            Ok(_) => Err(StoreError::NotADirectory {
                path: layout.base().to_owned(),
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Err(StoreError::NoLayout {
                root: layout.root().to_owned(),
            }),
            Err(e) => Err(StoreError::io("read", layout.base(), e)),
        }
    }

    /// The root's layout. Its root path is the resolved form of the path it was opened with.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The root directory, resolved.
    pub fn path(&self) -> &Path {
        self.layout.root()
    }
}

/// A node's store roots: the writable roots of `store.dirs` and the read-only roots of
/// `store.readonly_dirs`, each list in configuration order.
#[derive(Debug)]
pub struct StoreRoots {
    writable: Vec<WritableRoot>,
    readonly: Vec<ReadOnlyRoot>,
}

impl StoreRoots {
    /// Opens every writable root, taking each one's writer lock, and every read-only root.
    ///
    /// Paths are resolved first, and a directory listed twice, in either list or in both, is
    /// refused before any lock is taken. On any error, the locks already taken are released.
    ///
    /// # Errors
    ///
    /// [`StoreError::DuplicateRoot`], or any error of [`WritableRoot::open`] and
    /// [`ReadOnlyRoot::open`].
    pub fn open<W, R>(writable: W, readonly: R) -> Result<StoreRoots, StoreError>
    where
        W: IntoIterator,
        W::Item: AsRef<Path>,
        R: IntoIterator,
        R::Item: AsRef<Path>,
    {
        let writable: Vec<PathBuf> = writable
            .into_iter()
            .map(|p| prepare_writable(p.as_ref()))
            .collect::<Result<_, _>>()?;
        let readonly: Vec<PathBuf> = readonly
            .into_iter()
            .map(|p| resolve_readonly(p.as_ref()))
            .collect::<Result<_, _>>()?;
        let all: Vec<&PathBuf> = writable.iter().chain(&readonly).collect();
        for (i, path) in all.iter().enumerate() {
            if all[..i].contains(path) {
                return Err(StoreError::DuplicateRoot {
                    path: (*path).clone(),
                });
            }
        }
        Ok(StoreRoots {
            writable: writable
                .into_iter()
                .map(WritableRoot::open_resolved)
                .collect::<Result<_, _>>()?,
            readonly: readonly
                .into_iter()
                .map(ReadOnlyRoot::open_resolved)
                .collect::<Result<_, _>>()?,
        })
    }

    /// The writable roots, in `store.dirs` order. New data goes to the first one with enough
    /// free space (RFC-0041 §4.1).
    pub fn writable(&self) -> &[WritableRoot] {
        &self.writable
    }

    /// The read-only roots, in `store.readonly_dirs` order.
    pub fn readonly(&self) -> &[ReadOnlyRoot] {
        &self.readonly
    }

    /// Every root's layout, writable roots first, for lookups that search all roots.
    pub fn layouts(&self) -> impl Iterator<Item = &Layout> {
        self.writable
            .iter()
            .map(WritableRoot::layout)
            .chain(self.readonly.iter().map(ReadOnlyRoot::layout))
    }
}

/// Creates a writable root's directory when it is missing, and resolves its path.
fn prepare_writable(path: &Path) -> Result<PathBuf, StoreError> {
    ensure_dir(path)?;
    fs::canonicalize(path).map_err(|e| StoreError::io("resolve", path, e))
}

/// Checks that a read-only root is a directory, and resolves its path, without writing.
fn resolve_readonly(path: &Path) -> Result<PathBuf, StoreError> {
    let meta = fs::metadata(path).map_err(|e| StoreError::io("read", path, e))?;
    if !meta.is_dir() {
        return Err(StoreError::NotADirectory {
            path: path.to_owned(),
        });
    }
    fs::canonicalize(path).map_err(|e| StoreError::io("resolve", path, e))
}

/// Creates `path` and its missing parents, or checks that it is already a directory.
fn ensure_dir(path: &Path) -> Result<(), StoreError> {
    match fs::metadata(path) {
        Ok(meta) if meta.is_dir() => Ok(()),
        Ok(_) => Err(StoreError::NotADirectory {
            path: path.to_owned(),
        }),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(path).map_err(|e| StoreError::io("create directory", path, e))
        }
        Err(e) => Err(StoreError::io("read", path, e)),
    }
}

/// Replaces the lock file's contents with this process's ID. Called only by the lock holder.
fn record_holder(mut lock: &File) -> io::Result<()> {
    lock.set_len(0)?;
    writeln!(lock, "{}", std::process::id())
}

/// The process ID the lock's holder recorded, if the file can be read and holds one. The
/// holder may not have written it yet, and on Windows its lock blocks the read.
fn read_holder_pid(lock_path: &Path) -> Option<u32> {
    let mut text = String::new();
    File::open(lock_path).ok()?.read_to_string(&mut text).ok()?;
    text.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fresh directory under the system temporary directory, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> TempDir {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("syntropy-store-unit-{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            TempDir(path)
        }

        fn join(&self, p: &str) -> PathBuf {
            self.0.join(p)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn opening_a_writable_root_creates_the_layout_and_records_the_holder() {
        let tmp = TempDir::new();
        let root = WritableRoot::open(tmp.join("a/b/models")).unwrap();
        for dir in root.layout().fixed_dirs() {
            assert!(dir.is_dir(), "{} missing", dir.display());
        }
        let pid = fs::read_to_string(root.layout().writer_lock()).unwrap();
        assert_eq!(pid, format!("{}\n", std::process::id()));
        assert_eq!(
            read_holder_pid(&root.layout().writer_lock()),
            Some(std::process::id())
        );
    }

    #[test]
    fn a_writable_root_can_be_reopened_after_its_holder_is_dropped() {
        let tmp = TempDir::new();
        let path = tmp.join("models");
        drop(WritableRoot::open(&path).unwrap());
        let again = WritableRoot::open(&path).unwrap();
        assert_eq!(again.path(), fs::canonicalize(&path).unwrap());
    }

    #[test]
    fn a_file_is_not_a_writable_root() {
        let tmp = TempDir::new();
        let path = tmp.join("file");
        fs::write(&path, b"x").unwrap();
        assert!(matches!(
            WritableRoot::open(&path),
            Err(StoreError::NotADirectory { path: p }) if p == path
        ));
        fs::create_dir(tmp.join("r")).unwrap();
        fs::write(tmp.join("r/v1"), b"x").unwrap();
        assert!(matches!(
            WritableRoot::open(tmp.join("r")),
            Err(StoreError::NotADirectory { .. })
        ));
    }

    #[test]
    fn a_read_only_root_needs_an_existing_v1_directory() {
        let tmp = TempDir::new();
        let missing = tmp.join("missing");
        match ReadOnlyRoot::open(&missing) {
            Err(StoreError::Io { source, path, .. }) => {
                assert_eq!(source.kind(), io::ErrorKind::NotFound);
                assert_eq!(path, missing);
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
        fs::create_dir(tmp.join("empty")).unwrap();
        assert!(matches!(
            ReadOnlyRoot::open(tmp.join("empty")),
            Err(StoreError::NoLayout { .. })
        ));
        assert!(
            !tmp.join("empty/v1").exists(),
            "a read-only open created v1/"
        );
        fs::write(tmp.join("file"), b"x").unwrap();
        assert!(matches!(
            ReadOnlyRoot::open(tmp.join("file")),
            Err(StoreError::NotADirectory { .. })
        ));
    }

    #[test]
    fn a_read_only_root_opens_beside_its_writer() {
        let tmp = TempDir::new();
        let writer = WritableRoot::open(tmp.join("shared")).unwrap();
        let reader = ReadOnlyRoot::open(tmp.join("shared")).unwrap();
        assert_eq!(reader.layout(), writer.layout());
    }

    #[test]
    fn store_roots_open_both_lists_in_order() {
        let tmp = TempDir::new();
        drop(WritableRoot::open(tmp.join("ro")).unwrap());
        let roots = StoreRoots::open([tmp.join("w1"), tmp.join("w2")], [tmp.join("ro")]).unwrap();
        let names: Vec<_> = roots
            .layouts()
            .map(|l| l.root().file_name().unwrap().to_str().unwrap().to_owned())
            .collect();
        assert_eq!(names, ["w1", "w2", "ro"]);
        assert_eq!(roots.writable().len(), 2);
        assert_eq!(roots.readonly().len(), 1);
    }

    #[test]
    fn a_root_listed_twice_is_refused_before_any_lock() {
        let tmp = TempDir::new();
        drop(WritableRoot::open(tmp.join("ro")).unwrap());
        let cases: [(Vec<PathBuf>, Vec<PathBuf>); 3] = [
            (vec![tmp.join("w"), tmp.join("x/../w")], vec![]),
            (vec![tmp.join("ro")], vec![tmp.join("ro")]),
            (vec![], vec![tmp.join("ro"), tmp.join("./ro")]),
        ];
        for (w, r) in cases {
            assert!(
                matches!(
                    StoreRoots::open(&w, &r),
                    Err(StoreError::DuplicateRoot { .. })
                ),
                "{w:?} {r:?}"
            );
        }
        // Nothing stayed locked.
        StoreRoots::open([tmp.join("w"), tmp.join("ro")], [] as [PathBuf; 0]).unwrap();
    }

    #[test]
    fn errors_say_what_failed() {
        let locked = StoreError::WriterLocked {
            root: PathBuf::from("/data/models"),
            holder_pid: Some(4242),
        };
        assert_eq!(
            locked.to_string(),
            "store root /data/models is already open for writing by another process \
             (pid 4242); a store root has one writer process at a time"
        );
        let locked = StoreError::WriterLocked {
            root: PathBuf::from("/data/models"),
            holder_pid: None,
        };
        assert_eq!(
            locked.to_string(),
            "store root /data/models is already open for writing by another process; \
             a store root has one writer process at a time"
        );
    }
}
