// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CA's files: an owner-only directory, and files replaced whole, never left half written.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

/// Creates `dir`, and its parents, owner-only: 0700 on Unix. On Windows the state directory's
/// owner-only DACL (DIST-E2-T01) is inherited.
pub(super) fn private_dir(dir: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        builder.mode(0o700);
        builder.create(dir)?;
        // An existing directory keeps its mode under `create`, so narrow it.
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(not(unix))]
    builder.create(dir)?;
    Ok(())
}

/// Writes `bytes` to `path` so that `path` always holds either its old contents or all of
/// `bytes`: a new owner-only file (0600 on Unix) beside it, flushed to disk, renamed over
/// `path`, then the directory flushed. A failure leaves `path` as it was and removes the new
/// file.
pub(super) fn write_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other("no directory"))?;
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::other("no file name"))?;
    let temp = dir.join(format!(".{}.new", name.to_string_lossy()));
    // A file left by a write that crashed is stale; `create_new` below needs it gone.
    match fs::remove_file(&temp) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)?;
        sync_dir(dir)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

/// Flushes `dir`'s entries, so a rename in it survives a crash. Windows has no directory handle
/// to flush; its rename is durable once it returns.
fn sync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// The file at `path`, or `None` if there is none.
pub(super) fn read_optional(path: &Path) -> io::Result<Option<Vec<u8>>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("syntropy-api-files-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_private_write_replaces_the_file_whole_and_leaves_no_temporary() {
        let dir = scratch("write").join("tls");
        private_dir(&dir).unwrap();
        let path = dir.join("ca.json");
        write_private(&path, b"first").unwrap();
        write_private(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
        let names: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(names, ["ca.json"]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&path), 0o600);
            assert_eq!(mode(&dir), 0o700);
        }
        assert_eq!(read_optional(&dir.join("absent")).unwrap(), None);
        fs::remove_dir_all(dir.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_stale_temporary_from_a_crash_is_replaced() {
        let dir = scratch("stale");
        private_dir(&dir).unwrap();
        let path = dir.join("ca.json");
        fs::write(dir.join(".ca.json.new"), b"half").unwrap();
        write_private(&path, b"whole").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"whole");
        assert!(!dir.join(".ca.json.new").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
