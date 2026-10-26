// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runner-local cache: a directory of verified objects, least recently used first out, never
//! larger than its cap (RFC-0060 §12.3).
//!
//! Recency is the file's modification time. Every hit and every insert stamps the file with a time
//! later than any other entry's, so the order is exact even on a file system with coarse clocks.
//! A hit is hashed again before it is used, because the cache directory is outside the store's
//! control: a cached file that no longer matches is deleted and fetched again.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::hash_file;
use super::store::{Error, Key, Store, check};

pub struct Cache {
    dir: PathBuf,
    cap: u64,
}

impl Cache {
    /// Opens (creating) the cache in `dir` with a cap of `cap` bytes. Half-written files from an
    /// earlier crash are removed.
    pub fn open(dir: PathBuf, cap: u64) -> Result<Cache, Error> {
        fs::create_dir_all(&dir)?;
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "part") {
                fs::remove_file(path)?;
            }
        }
        Ok(Cache { dir, cap })
    }

    /// The bytes the entries occupy.
    pub fn size(&self) -> Result<u64, Error> {
        Ok(self.entries()?.iter().map(|e| e.bytes).sum())
    }

    fn entries(&self) -> Result<Vec<Entry>, Error> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_file() {
                entries.push(Entry {
                    path: entry.path(),
                    bytes: meta.len(),
                    used: meta.modified()?,
                });
            }
        }
        entries.sort_by(|a, b| (a.used, &a.path).cmp(&(b.used, &b.path)));
        Ok(entries)
    }

    /// Stamps `path` as the most recently used entry.
    fn stamp(&self, path: &Path) -> Result<(), Error> {
        let newest = self.entries()?.last().map(|e| e.used);
        let now = SystemTime::now();
        let time = match newest {
            Some(newest) if newest >= now => newest + Duration::from_millis(1),
            _ => now,
        };
        File::options().write(true).open(path)?.set_modified(time)?;
        Ok(())
    }

    /// The path of `key` in the cache, fetching it from `store` if it is not there. The bytes are
    /// checked against the key's BLAKE3 and against `bytes` before the path is returned; a
    /// mismatch is [`Error::Integrity`] and leaves nothing in the cache.
    pub fn fetch(&self, store: &dyn Store, key: &Key, bytes: u64) -> Result<PathBuf, Error> {
        let path = self.dir.join(key.hex());
        if path.is_file() {
            let ok = hash_file(&path).map(|(h, n)| check(key, bytes, &h, n).is_ok());
            if ok.unwrap_or(false) {
                self.stamp(&path)?;
                return Ok(path);
            }
            fs::remove_file(&path)?;
        }
        if bytes > self.cap {
            return Err(Error::TooLarge {
                bytes,
                cap: self.cap,
            });
        }
        // Make room first, so the cache holds at most `cap` bytes while the download runs.
        let mut entries = self.entries()?.into_iter();
        let mut total = self.size()?;
        while total + bytes > self.cap {
            let victim = entries
                .next()
                .expect("an empty cache has room for bytes <= cap");
            fs::remove_file(&victim.path)?;
            total -= victim.bytes;
        }
        let part = path.with_extension("part");
        let result = self.download(store, key, bytes, &part);
        if let Err(e) = result {
            let _ = fs::remove_file(&part);
            return Err(e);
        }
        fs::rename(&part, &path)?;
        self.stamp(&path)?;
        Ok(path)
    }

    fn download(&self, store: &dyn Store, key: &Key, bytes: u64, part: &Path) -> Result<(), Error> {
        let mut reader = store.open(key)?.take(bytes.saturating_add(1));
        let mut file = File::create(part)?;
        let mut hasher = blake3::Hasher::new();
        let mut buf = vec![0u8; 1 << 16];
        let mut len = 0u64;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            file.write_all(&buf[..n])?;
            len += n as u64;
        }
        check(key, bytes, hasher.finalize().to_hex().as_str(), len)
    }
}

struct Entry {
    path: PathBuf,
    bytes: u64,
    used: SystemTime,
}
