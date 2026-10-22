// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CA's files: an owner-only directory, and files replaced whole, never left half written.

use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

/// Creates `dir`, and its parents, owner-only: 0700 on Unix. On Windows the state directory's
/// owner-only DACL (DIST-E2-T01) is inherited.
///
/// An existing `dir` must be a directory itself. A link, even to a directory, or anything else
/// is refused before anything is changed, so a link planted in the state directory cannot get
/// another directory narrowed or written into.
pub(super) fn private_dir(dir: &Path) -> io::Result<()> {
    match fs::symlink_metadata(dir) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let mut builder = fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            builder.create(dir)?;
        }
        Err(e) => return Err(e),
        Ok(_) => {}
    }
    // `create` accepts a link to a directory, so look again, without following one.
    let found = fs::symlink_metadata(dir)?;
    if !found.is_dir() {
        return Err(io::Error::other("a link or not a directory; refused"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // An existing directory keeps its mode under `create`, so narrow it, through a handle
        // checked to be the directory just looked at: a link swapped in meanwhile is not
        // followed.
        let handle = File::open(dir)?;
        same_file(&found, &handle.metadata()?)?;
        handle.set_permissions(fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Refuses `opened` unless it is the file `found` described: a path swapped for a link between
/// the look and the open. Windows has no inode to compare; there the look alone decides.
fn same_file(found: &Metadata, opened: &Metadata) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if (found.dev(), found.ino()) != (opened.dev(), opened.ino()) {
            return Err(io::Error::other("replaced while being opened; refused"));
        }
    }
    #[cfg(not(unix))]
    let _ = (found, opened);
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

/// The owner-only file at `path`, or `None` if there is none. It must be a regular file: a
/// link, even to a regular file, or anything else is refused, not followed. A file whose mode
/// was widened is narrowed to 0600 again on Unix, as [`private_dir`] narrows the directory.
pub(super) fn read_private(path: &Path) -> io::Result<Option<Vec<u8>>> {
    let found = match fs::symlink_metadata(path) {
        Ok(found) => found,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    if !found.is_file() {
        return Err(io::Error::other("a link or not a regular file; refused"));
    }
    let mut file = File::open(path)?;
    same_file(&found, &file.metadata()?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if found.permissions().mode() & 0o777 != 0o600 {
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(Some(bytes))
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
        assert_eq!(read_private(&dir.join("absent")).unwrap(), None);
        fs::remove_dir_all(dir.parent().unwrap()).unwrap();
    }

    /// Review probe N1: a `tls` that is a link to another directory, or not a directory, is
    /// refused before anything is narrowed or written, and a file that is a link is not read.
    #[cfg(unix)]
    #[test]
    fn links_and_other_kinds_of_file_are_refused_not_followed() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let root = scratch("links");
        let elsewhere = root.join("elsewhere");
        fs::create_dir_all(&elsewhere).unwrap();
        fs::set_permissions(&elsewhere, fs::Permissions::from_mode(0o755)).unwrap();
        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
        let tls = root.join("tls");
        symlink(&elsewhere, &tls).unwrap();
        assert!(private_dir(&tls).is_err());
        assert_eq!(mode(&elsewhere), 0o755, "the link's target was narrowed");
        fs::remove_file(&tls).unwrap();
        fs::write(&tls, b"a file").unwrap();
        assert!(private_dir(&tls).is_err());
        fs::remove_file(&tls).unwrap();
        private_dir(&tls).unwrap();
        assert_eq!(mode(&tls), 0o700);
        let secret = elsewhere.join("secret");
        fs::write(&secret, b"not the CA's").unwrap();
        symlink(&secret, tls.join("ca.json")).unwrap();
        assert!(read_private(&tls.join("ca.json")).is_err());
        assert!(
            read_private(&tls).is_err(),
            "a directory is not a regular file"
        );
        fs::remove_dir_all(&root).unwrap();
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
