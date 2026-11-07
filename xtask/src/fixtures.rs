// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask fixtures`: the fixture store client (QUAL-E7-T05, RFC-0060 §12.3, ADR-0603).
//!
//! Large test data never enters git. It lives in a content-addressed store under its BLAKE3, and
//! `testdata/fixtures.toml` holds the manifest. This module has three parts:
//! - [`manifest`]: the `testdata/fixtures.toml` schema and the in-place edit `import` makes;
//! - [`store`]: the [`store::Store`] trait, the local-directory backend, and publishing an object
//!   with its licence file beside it;
//! - [`s3`]: the S3 backend over the project bucket, anonymous or with the lab credential, and
//!   [`sigv4`], the request signing it uses (QUAL-E7-T15);
//! - [`access`]: the bucket's access rule, which decides per fixture whether its objects are
//!   publicly readable (ADR-0603 decision 8);
//! - [`cache`]: the runner-local LRU cache with a size cap, which verifies every object before it
//!   hands out a path;
//! - [`lint`]: `lint`'s `fixtures` step, the licence rule and pinned sources (QUAL-E7-T06).
//!
//! A hash mismatch is fatal ([`store::Error::Integrity`]): bytes that are not the expected ones
//! are never used. `--store` names a directory, or the bucket as `s3+https://…` ([`s3`]). The
//! bucket itself, its policy and the lab credential are Maintainer actions.

pub mod access;
pub mod cache;
pub mod lint;
pub mod manifest;
pub mod s3;
pub mod sigv4;
pub mod store;

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use self::cache::Cache;
use self::manifest::Manifest;
use self::s3::{S3Config, S3Store};
use self::store::{Error, Key, LocalStore, Prefix, Store};

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
    /// The store: a directory (the local-directory backend), or the bucket as
    /// `s3+https://<host>/<bucket>` ([`s3`]).
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

/// Runs `action`. `checkout` locates the default manifest. Progress goes to `out`. The S3
/// region and the lab credential come from the process environment.
pub fn run(
    action: Action,
    args: &Args,
    checkout: &Path,
    out: &mut dyn Write,
) -> Result<(), String> {
    run_with_env(action, args, checkout, out, &|k| std::env::var(k).ok())
}

/// [`run`], reading the environment through `env`.
pub fn run_with_env(
    action: Action,
    args: &Args,
    checkout: &Path,
    out: &mut dyn Write,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<(), String> {
    let manifest_path = args
        .manifest
        .as_ref()
        .map_or_else(|| checkout.join("testdata/fixtures.toml"), PathBuf::from);
    let store = open_store(&need(&args.store, "--store")?, env)?;
    let store = store.as_ref();
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
                if skip_unimported(f, args, out)? {
                    continue;
                }
                let key = f.key().map_err(|e| format!("{}: {e}", f.name))?;
                let path = cache
                    .fetch(store, &key, f.bytes)
                    .map_err(|e| format!("{}: {e}", f.name))?;
                writeln!(out, "{}: verified, {}", f.name, path.display()).map_err(io_err)?;
            }
        }
        Action::Verify => {
            for f in manifest.fixtures.iter().filter(|f| selected(&f.name)) {
                if skip_unimported(f, args, out)? {
                    continue;
                }
                let key = f.key().map_err(|e| format!("{}: {e}", f.name))?;
                store::verify(store, &key, f.bytes).map_err(|e| format!("{}: {e}", f.name))?;
                writeln!(out, "{}: ok", f.name).map_err(io_err)?;
            }
        }
        Action::Import => {
            let name = need(&args.name, "--name")?;
            let file = need(&args.file, "--file")?;
            let licence = need(&args.licence_file, "--licence-file")?;
            // The entry is checked before anything is published: it must exist, and the edit
            // that records the object must succeed for this file's hash and length.
            let entry = manifest
                .fixtures
                .iter()
                .find(|f| f.name == name)
                .ok_or_else(|| format!("no fixture named `{name}` in the manifest"))?;
            let (hash, bytes) = hash_file(Path::new(&file)).map_err(|e| format!("{file}: {e}"))?;
            let updated = manifest::fill(&text, &name, &hash, bytes)?;
            Manifest::parse(&updated).map_err(|e| format!("the filled manifest: {e}"))?;
            let access = access::access(entry);
            let (published, len) = store::publish(
                store,
                Prefix::Fixtures,
                Path::new(&file),
                Path::new(&licence),
                access,
            )
            .map_err(|e| e.to_string())?;
            if (published.as_str(), len) != (hash.as_str(), bytes) {
                return Err(format!("{file} changed while it was imported"));
            }
            write_atomically(&manifest_path, updated.as_bytes())
                .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
            writeln!(out, "{name}: {bytes} bytes, blake3 {hash}, {access:?}").map_err(io_err)?;
        }
    }
    Ok(())
}

/// Whether `fetch` or `verify` over the whole manifest passes over `f`: a `lab` fixture recorded but
/// not yet imported has no `blake3`, such as a checkpoint's weights that wait for the lab store
/// (QUAL-E7-T06). A `nightly` fixture is never skipped: the `fixtures` lint step requires it to be
/// imported, and a run over the manifest fails on one that is not. Named with `--name`, an
/// un-imported fixture is an error, so a run never uses an empty hash.
fn skip_unimported(
    f: &manifest::Fixture,
    args: &Args,
    out: &mut dyn Write,
) -> Result<bool, String> {
    if f.tier == manifest::Tier::Lab && f.blake3.is_empty() && args.name.is_none() {
        writeln!(out, "{}: not imported yet, skipped", f.name).map_err(io_err)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// The store `--store` names: the bucket for an `s3+` URL, else a directory. `env` reads the
/// region and the lab credential.
pub fn open_store(
    spec: &str,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Box<dyn Store>, String> {
    match S3Config::parse(spec, env) {
        Some(config) => Ok(Box::new(S3Store::new(config?)?)),
        None => Ok(Box::new(LocalStore::new(PathBuf::from(spec)))),
    }
}

/// Replaces `path` with `bytes` in one rename: a temporary file beside it, unique to this call,
/// written and synced first. A reader sees the old file or the new one, never a part.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = store::temp_beside(path);
    let written = File::create_new(&tmp).and_then(|mut f| {
        f.write_all(bytes)?;
        f.sync_all()
    });
    let renamed = written.and_then(|()| std::fs::rename(&tmp, path));
    if renamed.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    renamed
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
mod fake_s3;
#[cfg(test)]
mod s3_tests;
#[cfg(test)]
mod tests;
