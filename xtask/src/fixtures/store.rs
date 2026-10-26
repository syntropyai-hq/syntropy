// SPDX-License-Identifier: MIT OR Apache-2.0

//! The store: keys, the [`Store`] trait and the local-directory backend.
//!
//! Every object is stored under the lowercase hex BLAKE3 of its bytes, below one of the five
//! prefixes of RFC-0060 §12.3. A licence file sits beside the object it covers, as
//! `<hash>.licence`.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

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
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

/// Where objects live. The S3 backend is a separate task.
pub trait Store {
    fn open(&self, key: &Key) -> Result<Box<dyn Read>, Error>;
    /// Writes the object whole or not at all.
    fn put(&self, key: &Key, data: &mut dyn Read) -> Result<(), Error>;
    fn exists(&self, key: &Key) -> bool;
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

    fn put(&self, key: &Key, data: &mut dyn Read) -> Result<(), Error> {
        let path = self.path(key);
        fs::create_dir_all(path.parent().expect("a key path has a parent"))?;
        let tmp = path.with_extension("part");
        let written = File::create(&tmp).and_then(|mut f| io::copy(data, &mut f));
        if let Err(e) = written {
            let _ = fs::remove_file(&tmp);
            return Err(e.into());
        }
        fs::rename(&tmp, &path)?;
        Ok(())
    }

    fn exists(&self, key: &Key) -> bool {
        self.path(key).is_file()
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

/// Publishes `file` under its BLAKE3 with `licence_file` beside it, returning the hash and
/// length. A missing or empty licence file is refused before anything is written. The licence
/// goes in first, so an object is never visible without its licence. An object already in the
/// store under that hash is kept as it is, not read again; `verify` is what checks the store.
pub fn publish(
    store: &dyn Store,
    prefix: Prefix,
    file: &Path,
    licence_file: &Path,
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
    if !store.exists(&key.licence()) {
        store.put(&key.licence(), &mut File::open(licence_file)?)?;
    }
    if !store.exists(&key) {
        store.put(&key, &mut File::open(file)?)?;
    }
    Ok((hash, bytes))
}
