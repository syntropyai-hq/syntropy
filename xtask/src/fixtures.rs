// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask fixtures`: the fixture store client (QUAL-E7-T05, RFC-0060 §12.3, ADR-0603).
//!
//! Large test data never enters git. It lives in a content-addressed store under its BLAKE3, and
//! `testdata/fixtures.toml` holds the manifest. This module has three parts:
//! - [`manifest`]: the `testdata/fixtures.toml` schema and the in-place edit `import` makes;
//! - [`store`]: the [`store::Store`] trait, the local-directory backend, and publishing an object
//!   with its licence file beside it;
//! - [`cache`]: the runner-local LRU cache with a size cap, which verifies every object before it
//!   hands out a path.
//!
//! A hash mismatch is fatal ([`store::Error::Integrity`]): bytes that are not the expected ones
//! are never used. The S3 backend and the public-read access rule need the bucket, which is a
//! Maintainer action, and are a separate task.

pub mod cache;
pub mod manifest;
pub mod store;

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use self::cache::Cache;
use self::manifest::Manifest;
use self::store::{Error, Key, LocalStore, Prefix};

/// What a `fixtures` subcommand does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Fetch the named fixtures (all when none is named) into the cache, verified.
    Fetch,
    /// Publish a file with its licence file and fill `bytes` and `blake3` in the manifest.
    Import,
    /// Check every fixture with a recorded hash against the store, without the cache.
    Verify,
}

impl Action {
    /// The subcommand as typed.
    pub fn parse(word: &str) -> Option<Action> {
        match word {
            "fetch" => Some(Action::Fetch),
            "import" => Some(Action::Import),
            "verify" => Some(Action::Verify),
            _ => None,
        }
    }
}

/// The options of a `fixtures` subcommand, as parsed from the command line.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Args {
    /// `testdata/fixtures.toml`, else the one in this checkout.
    pub manifest: Option<String>,
    /// The store's root directory (the local-directory backend).
    pub store: Option<String>,
    /// The cache directory (`fetch`).
    pub cache: Option<String>,
    /// The cache cap in bytes (`fetch`).
    pub cap: Option<String>,
    /// One fixture by name (`fetch`, `verify`; required by `import`).
    pub name: Option<String>,
    /// The file to import.
    pub file: Option<String>,
    /// The licence file to publish beside the imported file.
    pub licence_file: Option<String>,
}

/// Runs `action`. `checkout` locates the default manifest. Progress goes to `out`.
pub fn run(
    action: Action,
    args: &Args,
    checkout: &Path,
    out: &mut dyn Write,
) -> Result<(), String> {
    let manifest_path = args
        .manifest
        .as_ref()
        .map_or_else(|| checkout.join("testdata/fixtures.toml"), PathBuf::from);
    let store = LocalStore::new(PathBuf::from(need(&args.store, "--store")?));
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest =
        Manifest::parse(&text).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let selected = |name: &str| args.name.as_deref().is_none_or(|n| n == name);
    if let Some(name) = &args.name
        && action != Action::Import
        && !manifest.fixtures.iter().any(|f| &f.name == name)
    {
        return Err(format!("no fixture named `{name}` in the manifest"));
    }
    match action {
        Action::Fetch => {
            let cap: u64 = need(&args.cap, "--cap")?
                .parse()
                .map_err(|_| "--cap is a number of bytes".to_owned())?;
            let cache = Cache::open(PathBuf::from(need(&args.cache, "--cache")?), cap)
                .map_err(|e| e.to_string())?;
            for f in manifest.fixtures.iter().filter(|f| selected(&f.name)) {
                let key = f.key().map_err(|e| format!("{}: {e}", f.name))?;
                let path = cache
                    .fetch(&store, &key, f.bytes)
                    .map_err(|e| format!("{}: {e}", f.name))?;
                writeln!(out, "{}: verified, {}", f.name, path.display()).map_err(io_err)?;
            }
        }
        Action::Verify => {
            for f in manifest.fixtures.iter().filter(|f| selected(&f.name)) {
                let key = f.key().map_err(|e| format!("{}: {e}", f.name))?;
                store::verify(&store, &key, f.bytes).map_err(|e| format!("{}: {e}", f.name))?;
                writeln!(out, "{}: ok", f.name).map_err(io_err)?;
            }
        }
        Action::Import => {
            let name = need(&args.name, "--name")?;
            let file = need(&args.file, "--file")?;
            let licence = need(&args.licence_file, "--licence-file")?;
            let (hash, bytes) = store::publish(
                &store,
                Prefix::Fixtures,
                Path::new(&file),
                Path::new(&licence),
            )
            .map_err(|e| e.to_string())?;
            let updated = manifest::fill(&text, &name, &hash, bytes)?;
            std::fs::write(&manifest_path, updated)
                .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
            writeln!(out, "{name}: {bytes} bytes, blake3 {hash}").map_err(io_err)?;
        }
    }
    Ok(())
}

fn need(value: &Option<String>, option: &str) -> Result<String, String> {
    value.clone().ok_or_else(|| format!("missing {option}"))
}

fn io_err(e: io::Error) -> String {
    e.to_string()
}

/// BLAKE3 of everything `reader` yields and its length, reading at most `limit + 1` bytes so a
/// longer stream is seen as longer without being read to the end.
pub(crate) fn hash_reader(reader: &mut dyn Read, limit: u64) -> io::Result<(String, u64)> {
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut total = 0u64;
    let mut reader = reader.take(limit.saturating_add(1));
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hasher.finalize().to_hex().to_string(), total))
}

pub(crate) fn hash_file(path: &Path) -> io::Result<(String, u64)> {
    let mut file = File::open(path)?;
    hash_reader(&mut file, u64::MAX - 1)
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Error {
        Error::Io(e.to_string())
    }
}

/// A fixture's store key, for the manifest module.
pub(crate) fn fixture_key(hex: &str) -> Result<Key, Error> {
    Key::new(Prefix::Fixtures, hex)
}
#[cfg(test)]
mod tests;
