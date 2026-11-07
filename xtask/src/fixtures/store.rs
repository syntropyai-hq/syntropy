// SPDX-License-Identifier: MIT OR Apache-2.0

//! The store: keys, the [`Store`] trait and the local-directory backend. The S3 backend is
//! [`super::s3`].
//!
//! Every object is stored under the lowercase hex BLAKE3 of its bytes, below one of the five
//! prefixes of RFC-0060 §12.3. A licence file sits beside the object it covers, as
//! `<hash>.licence`. Each object is written [`Access::Public`] or [`Access::Private`]: the
//! access rule of ADR-0603 decision 8, which [`super::access`] decides per fixture.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::{hash_file, hash_reader};

/// The top-level prefixes of the bucket (QUAL-E7-T05). Only `Fixtures` has a command yet; the
/// others are used by the tasks that store references, kernel caches, results and artefacts.
#[expect(
    dead_code,
    reason = "Refs, Kcache, Results and Artefacts are used by later tasks"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prefix {
    Fixtures,
    Refs,
    Kcache,
    Results,
    Artefacts,
}

impl Prefix {
    pub fn dir(self) -> &'static str {
        match self {
            Prefix::Fixtures => "fixtures",
            Prefix::Refs => "refs",
            Prefix::Kcache => "kcache",
            Prefix::Results => "results",
            Prefix::Artefacts => "artefacts",
        }
    }
}

/// The name of one object: a prefix and a BLAKE3 in lowercase hex, optionally the licence file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key {
    prefix: Prefix,
    hex: String,
    licence: bool,
}

impl Key {
    /// The object stored under `hex`. A hash that is not 64 lowercase hex digits is refused, so a
    /// key can never name a path outside its prefix.
    pub fn new(prefix: Prefix, hex: &str) -> Result<Key, Error> {
        let valid = hex.len() == 64 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
        if !valid {
            return Err(Error::InvalidKey(hex.to_owned()));
        }
        Ok(Key {
            prefix,
            hex: hex.to_owned(),
            licence: false,
        })
    }

    /// The licence file beside this object.
    pub fn licence(&self) -> Key {
        Key {
            licence: true,
            ..self.clone()
        }
    }

    pub fn hex(&self) -> &str {
        &self.hex
    }

    /// Whether this is a licence file's key.
    pub fn is_licence(&self) -> bool {
        self.licence
    }

    fn file_name(&self) -> String {
        if self.licence {
            format!("{}.licence", self.hex)
        } else {
            self.hex.clone()
        }
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.prefix.dir(), self.file_name())
    }
}

/// What can go wrong with the store.
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// The bytes read do not hash to the key, or are not the recorded length. Fatal: the caller
    /// must not use them.
    Integrity {
        key: String,
        expected: String,
        actual: String,
    },
    NotFound(String),
    /// The object is larger than the cache cap.
    TooLarge {
        bytes: u64,
        cap: u64,
    },
    /// Publishing needs a non-empty licence file.
    LicenceMissing(String),
    InvalidKey(String),
    /// The store refused the request: an anonymous read of a private object, or a write
    /// without the lab credential.
    Denied(String),
    /// A request that failed for any other reason: the network, a server error, a reply the
    /// client did not expect. Never read as "absent".
    Remote(String),
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Integrity {
                key,
                expected,
                actual,
            } => {
                write!(f, "Integrity: {key} should be {expected} but is {actual}")
            }
            Error::NotFound(key) => write!(f, "{key} is not in the store"),
            Error::TooLarge { bytes, cap } => {
                write!(
                    f,
                    "an object of {bytes} bytes cannot fit a cache capped at {cap} bytes"
                )
            }
            Error::LicenceMissing(why) => {
                write!(f, "refusing to publish without a licence file: {why}")
            }
            Error::InvalidKey(key) => write!(f, "`{key}` is not a lowercase hex BLAKE3"),
            Error::Denied(what) => write!(f, "access denied: {what}"),
            Error::Remote(what) => write!(f, "store request failed: {what}"),
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

/// Who may read an object (ADR-0603 decision 8): the bucket is private, and public read is
/// granted object by object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// Anyone may read it: a project-synthetic or permissively licensed fixture, or the licence
    /// file of one.
    Public,
    /// Only the lab credential may read it.
    Private,
}

/// Where objects live: [`LocalStore`], or [`super::s3::S3Store`] over the project bucket.
pub trait Store {
    /// The object's bytes. [`Error::NotFound`] when it is absent, [`Error::Denied`] when this
    /// client may not read it.
    fn open(&self, key: &Key) -> Result<Box<dyn Read>, Error>;
    /// Writes the object whole or not at all, readable as `access` says. An object's bytes must
    /// hash to its key ([`Error::Integrity`] otherwise); a licence file's need not.
    fn put(&self, key: &Key, data: &mut dyn Read, access: Access) -> Result<(), Error>;
    /// Whether the object is there. A request that fails is an error, never `false`.
    fn exists(&self, key: &Key) -> Result<bool, Error>;
}

/// Makes each temporary name unique within the process; the process ID makes it unique across
/// processes.
static NEXT_TMP: AtomicU64 = AtomicU64::new(0);

/// A temporary name beside `path`, unique to this call: `<name>.<pid>.<n>.part`.
pub(crate) fn temp_beside(path: &Path) -> PathBuf {
    let n = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    path.with_file_name(format!("{name}.{}.{n}.part", std::process::id()))
}

/// Copies `data` into `out`, returning the BLAKE3 of what was copied and its length.
pub(crate) fn copy_hashing(data: &mut dyn Read, out: &mut dyn Write) -> io::Result<(String, u64)> {
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut total = 0u64;
    loop {
        let n = match data.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        hasher.update(&buf[..n]);
        out.write_all(&buf[..n])?;
        total += n as u64;
    }
    Ok((hasher.finalize().to_hex().to_string(), total))
}

/// The integrity rule for a write: an object's bytes hash to its key; a licence file is named by
/// the object it covers, so its own hash is not checked.
pub(crate) fn check_put(key: &Key, actual: &str) -> Result<(), Error> {
    if key.is_licence() || actual == key.hex() {
        Ok(())
    } else {
        Err(Error::Integrity {
            key: key.to_string(),
            expected: key.hex().to_owned(),
            actual: actual.to_owned(),
        })
    }
}

/// A store in a directory: `<root>/<prefix>/<name>`.
pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    pub fn new(root: PathBuf) -> LocalStore {
        LocalStore { root }
    }

    fn path(&self, key: &Key) -> PathBuf {
        self.root.join(key.prefix.dir()).join(key.file_name())
    }
}

impl Store for LocalStore {
    fn open(&self, key: &Key) -> Result<Box<dyn Read>, Error> {
        match File::open(self.path(key)) {
            Ok(file) => Ok(Box::new(file)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Err(Error::NotFound(key.to_string())),
            Err(e) => Err(e.into()),
        }
    }

    /// Copies into a temporary name unique to this call, hashing as it copies, and renames it
    /// into place only when the bytes hash to the key. Two writers of one key never share a
    /// temporary file. A local directory has no readers to restrict, so `access` is not stored.
    fn put(&self, key: &Key, data: &mut dyn Read, _access: Access) -> Result<(), Error> {
        let path = self.path(key);
        fs::create_dir_all(path.parent().expect("a key path has a parent"))?;
        let tmp = temp_beside(&path);
        let written = File::create_new(&tmp)
            .and_then(|mut f| {
                let r = copy_hashing(data, &mut f)?;
                f.sync_all()?;
                Ok(r)
            })
            .map_err(Error::from)
            .and_then(|(actual, _)| check_put(key, &actual));
        if let Err(e) = written {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        fs::rename(&tmp, &path).inspect_err(|_| {
            let _ = fs::remove_file(&tmp);
        })?;
        Ok(())
    }

    fn exists(&self, key: &Key) -> Result<bool, Error> {
        match fs::metadata(self.path(key)) {
            Ok(m) => Ok(m.is_file()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}

/// Reads `key` to the end and checks it against the key's hash and `bytes`. Used by `verify`.
pub fn verify(store: &dyn Store, key: &Key, bytes: u64) -> Result<(), Error> {
    let mut reader = store.open(key)?;
    let (actual, len) = hash_reader(&mut reader, bytes)?;
    check(key, bytes, &actual, len)
}

/// The integrity rule shared by `verify` and the cache: the hash and the length must both match.
pub(crate) fn check(key: &Key, bytes: u64, actual: &str, len: u64) -> Result<(), Error> {
    if actual != key.hex() {
        return Err(Error::Integrity {
            key: key.to_string(),
            expected: key.hex().to_owned(),
            actual: actual.to_owned(),
        });
    }
    if len != bytes {
        return Err(Error::Integrity {
            key: key.to_string(),
            expected: format!("{bytes} bytes"),
            actual: format!("{len} bytes"),
        });
    }
    Ok(())
}

/// Publishes `file` under its BLAKE3 with `licence_file` beside it, both readable as `access`
/// says, returning the hash and length. A missing or empty licence file is refused before
/// anything is written. The licence goes in first, so an object is never visible without its
/// licence. An object already in the store under that hash is kept as it is, not read again;
/// `verify` is what checks the store.
pub fn publish(
    store: &dyn Store,
    prefix: Prefix,
    file: &Path,
    licence_file: &Path,
    access: Access,
) -> Result<(String, u64), Error> {
    let licence_len = fs::metadata(licence_file)
        .map_err(|e| Error::LicenceMissing(format!("{}: {e}", licence_file.display())))?
        .len();
    if licence_len == 0 {
        return Err(Error::LicenceMissing(format!(
            "{} is empty",
            licence_file.display()
        )));
    }
    let (hash, bytes) = hash_file(file)?;
    let key = Key::new(prefix, &hash)?;
    if !store.exists(&key.licence())? {
        store.put(&key.licence(), &mut File::open(licence_file)?, access)?;
    }
    if !store.exists(&key)? {
        store.put(&key, &mut File::open(file)?, access)?;
    }
    Ok((hash, bytes))
}
