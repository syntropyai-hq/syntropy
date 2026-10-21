// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opening a read-only store root writes nothing to it (RFC-0041 §4.1).
//!
//! The root is populated by a writer, then made unwritable (on Unix) and snapshotted. It is
//! opened read-only, alone and beside a writable root, and read through its layout. The
//! snapshot afterwards must equal the one before: no entry added, removed, resized or modified.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use common::TempDir;
use syntropy_store::{Name, ReadOnlyRoot, StoreRoots, WritableRoot};

#[derive(Debug, PartialEq, Eq)]
struct Entry {
    path: PathBuf,
    is_dir: bool,
    len: u64,
    modified: SystemTime,
}

/// Every entry under `root`, the root included, sorted by path.
fn snapshot(root: &Path) -> Vec<Entry> {
    fn walk(path: &Path, out: &mut Vec<Entry>) {
        let meta = fs::symlink_metadata(path).unwrap();
        out.push(Entry {
            path: path.to_owned(),
            is_dir: meta.is_dir(),
            len: meta.len(),
            modified: meta.modified().unwrap(),
        });
        if meta.is_dir() {
            for entry in fs::read_dir(path).unwrap() {
                walk(&entry.unwrap().path(), out);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Makes every entry under `root` unwritable, or writable again, for its owner.
#[cfg(unix)]
fn set_writable(root: &Path, writable: bool) {
    use std::os::unix::fs::PermissionsExt;
    for entry in snapshot(root) {
        let mode = match (entry.is_dir, writable) {
            (true, true) => 0o755,
            (true, false) => 0o555,
            (false, true) => 0o644,
            (false, false) => 0o444,
        };
        fs::set_permissions(&entry.path, fs::Permissions::from_mode(mode)).unwrap();
    }
}

#[test]
fn opening_a_read_only_root_writes_nothing_to_it() {
    let tmp = TempDir::new("ro");
    let shared = tmp.join("shared");
    let demo = Name::new("demo").unwrap();
    {
        let writer = WritableRoot::open(&shared).unwrap();
        fs::write(writer.layout().local_ref(&demo), b"model id\n").unwrap();
    }
    #[cfg(unix)]
    set_writable(&shared, false);
    let before = snapshot(&shared);

    let direct = ReadOnlyRoot::open(&shared).unwrap();
    let roots = StoreRoots::open([tmp.join("rw")], [&shared]).unwrap();
    for layout in [direct.layout(), roots.readonly()[0].layout()] {
        assert_eq!(fs::read(layout.local_ref(&demo)).unwrap(), b"model id\n");
    }
    drop(roots);
    drop(direct);

    let after = snapshot(&shared);
    #[cfg(unix)]
    set_writable(&shared, true);
    assert_eq!(before, after);
    // The writable root beside it was created and locked as usual.
    assert!(tmp.join("rw/v1/locks/writer.lock").is_file());
}
