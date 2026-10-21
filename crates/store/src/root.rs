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
//! Windows, through [`File::try_lock`]) on `<root>/v1/locks/writer.lock`. Dropping the
//! [`WritableRoot`] unlocks it. When the process exits or is killed, the OS releases it, so a
//! crashed daemon never leaves a root locked; on Windows that release can lag the exit
//! slightly. The file is opened close-on-exec (the standard library's default), so a child the
//! holder spawns does not inherit the lock. A process that finds the lock held fails at once
//! with [`StoreError::WriterLocked`]; it never waits.
//!
//! The lock file stays empty. On Windows the lock is a mandatory byte-range lock, so no other
//! handle could read it. The holder writes its process ID to `locks/writer.pid` instead, which
//! is never locked, and a refused process reports that ID. The ID is only a diagnostic: writing
//! or reading it never decides whether a root opens. A holder that cannot write the file
//! removes it instead, so a refusal reports no ID rather than an earlier holder's. The ID can
//! still name a previous holder in the moment between a new holder taking the lock and writing
//! the file, or when the file can be neither written nor removed.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
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
    /// Holds the advisory lock, which [`Drop`] releases.
    lock: File,
}

impl Drop for WritableRoot {
    fn drop(&mut self) {
        // Closing the file would release the lock too, but Windows documents that a lock left
        // on a closed handle is freed only when the system gets to it. Unlock explicitly. A
        // failure leaves the release to the close that follows.
        let _ = self.lock.unlock();
    }
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
    ///   opened or locked.
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
            // Nothing is ever written to the lock file; a refused open must not touch it.
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| StoreError::io("open", &lock_path, e))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(StoreError::WriterLocked {
                    root: layout.root().to_owned(),
                    holder_pid: read_holder_pid(&layout.writer_pid()),
                });
            }
            Err(TryLockError::Error(e)) => return Err(StoreError::io("lock", &lock_path, e)),
        }
        // From here on, `root` releases the lock on every return.
        let root = WritableRoot { layout, lock };
        record_holder(&root.layout.writer_pid());

        for dir in root.layout.fixed_dirs() {
            ensure_dir(&dir)?;
        }
        Ok(root)
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

/// Records this process's ID in `writer.pid`. Called only by the lock holder. Best effort: the
/// ID is a diagnostic, so a failure leaves the root open. When the write fails, for example on
/// a `writer.pid` that is not writable, the file still holds an earlier holder's ID, which may
/// be a dead or reused process. Removing it makes a refusal report no ID instead.
fn record_holder(pid_path: &Path) {
    if fs::write(pid_path, format!("{}\n", std::process::id())).is_ok() {
        return;
    }
    if fs::remove_file(pid_path).is_err() {
        remove_read_only_file(pid_path);
    }
}

/// Windows refuses to delete a file with the read-only attribute. Clear the attribute, then
/// delete the file. Best effort.
#[cfg(windows)]
#[allow(
    clippy::permissions_set_readonly_false,
    reason = "on Windows this clears only the read-only attribute; it widens no access mode"
)]
fn remove_read_only_file(path: &Path) {
    let Ok(meta) = fs::symlink_metadata(path) else {
        return;
    };
    if !meta.is_file() {
        return;
    }
    let mut perms = meta.permissions();
    perms.set_readonly(false);
    if fs::set_permissions(path, perms).is_ok() {
        let _ = fs::remove_file(path);
    }
}

/// On Unix a file's own mode never blocks deleting it, so there is nothing more to try.
#[cfg(not(windows))]
fn remove_read_only_file(_path: &Path) {}

/// The process ID the lock's holder recorded in `writer.pid`, if the file can be read and holds
/// one. `None` when it is missing, unreadable or malformed, or the holder has not written it yet.
fn read_holder_pid(pid_path: &Path) -> Option<u32> {
    fs::read_to_string(pid_path).ok()?.trim().parse().ok()
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
        // The ID goes to the unlocked `writer.pid`; the locked file stays empty, so nothing
        // ever has to read it (on Windows no other handle could).
        let pid = fs::read_to_string(root.layout().writer_pid()).unwrap();
        assert_eq!(pid, format!("{}\n", std::process::id()));
        assert_eq!(
            read_holder_pid(&root.layout().writer_pid()),
            Some(std::process::id())
        );
        assert_eq!(fs::metadata(root.layout().writer_lock()).unwrap().len(), 0);
    }

    /// Holder A exits and leaves its ID in `writer.pid`. The file is then made read-only, so
    /// holder B cannot overwrite it. A refusal while B holds the root must not name A. This is
    /// the review's reproduction, in one process (see `the_holder_id_is_best_effort`).
    #[test]
    fn a_holder_that_cannot_record_its_id_removes_the_stale_one() {
        let tmp = TempDir::new();
        let path = tmp.join("models");
        let pid_path = WritableRoot::open(&path).unwrap().layout().writer_pid();
        let stale = u32::MAX;
        assert_ne!(stale, std::process::id());
        fs::write(&pid_path, format!("{stale}\n")).unwrap();
        let mut perms = fs::metadata(&pid_path).unwrap().permissions();
        perms.set_readonly(true);
        fs::set_permissions(&pid_path, perms).unwrap();
        // A privileged user (root on Unix) can still write a read-only file. Then holder B's
        // write succeeds and the stale ID is overwritten instead of removed.
        let write_blocked = OpenOptions::new().write(true).open(&pid_path).is_err();

        let held = WritableRoot::open(&path).unwrap();
        let reported = match WritableRoot::open(&path) {
            Err(StoreError::WriterLocked { holder_pid, .. }) => holder_pid,
            other => panic!("expected WriterLocked, got {other:?}"),
        };
        assert_ne!(
            reported,
            Some(stale),
            "the refusal named the earlier holder"
        );
        if write_blocked {
            assert_eq!(reported, None);
            assert!(!pid_path.exists(), "the stale writer.pid was left in place");
        } else {
            assert_eq!(reported, Some(std::process::id()));
        }
        drop(held);
    }

    /// A second handle in the same process conflicts with the first on every OS (`flock` locks
    /// belong to the open file, `LockFileEx` locks to the handle), which lets this test reach
    /// the refusal path without a child process. The two-process criterion is tested in
    /// `tests/writer_lock.rs`.
    #[test]
    fn the_holder_id_is_best_effort() {
        let tmp = TempDir::new();
        let path = tmp.join("models");
        let held = WritableRoot::open(&path).unwrap();
        let pid_path = held.layout().writer_pid();
        let holder_pid = |path: &Path| match WritableRoot::open(path) {
            Err(StoreError::WriterLocked { holder_pid, .. }) => holder_pid,
            other => panic!("expected WriterLocked, got {other:?}"),
        };
        assert_eq!(holder_pid(&path), Some(std::process::id()));
        fs::write(&pid_path, b"not a pid\n").unwrap();
        assert_eq!(holder_pid(&path), None);
        fs::remove_file(&pid_path).unwrap();
        assert_eq!(holder_pid(&path), None);
        // A missing or unwritable `writer.pid` never stops a root from opening.
        drop(held);
        fs::create_dir(&pid_path).unwrap();
        let reopened = WritableRoot::open(&path).unwrap();
        assert!(pid_path.is_dir());
        drop(reopened);
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
