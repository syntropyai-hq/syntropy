// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reading a fixture's bytes from the runner's fixture cache (RFC-0060 §12.3).
//!
//! `cargo xtask fixtures fetch` fills the cache: each object is a file named by its BLAKE3 in
//! lowercase hex. This reader looks a fixture up in `testdata/fixtures.toml`, applies the
//! licence rule, and hashes the cached bytes again before it hands them out, so a test never
//! runs on bytes the manifest does not name.
//!
//! The manifest is read with a scanner for the flat form `fixtures.toml` uses (`[[fixture]]`
//! tables of strings, integers, booleans, string lists and one inline `source` table), so the
//! crate needs no TOML parser. `cargo xtask fixtures` owns the schema; a line this scanner does
//! not understand is an error, never skipped.

use std::fmt;
use std::path::{Path, PathBuf};

/// The licences the licence rule accepts for an ungated nightly fixture (ADR-0603 decision 4;
/// the same list as `cargo xtask fixtures`' lint step).
pub const OSI_LICENCES: [&str; 8] = [
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsl-1.0",
    "isc",
    "mpl-2.0",
    "zlib",
];

/// One `[[fixture]]` entry, the fields this reader uses.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fixture {
    pub name: String,
    /// Empty when the fixture is recorded but not imported.
    pub blake3: String,
    pub bytes: u64,
    pub hub: String,
    pub revision: String,
    pub file: String,
    pub licence: String,
    pub gated: bool,
}

impl Fixture {
    /// Whether only a runner with the lab credential may read it: a gated fixture, or one
    /// under a licence the rule does not accept for public runners (RFC-0060 §12.2, §12.3).
    pub fn needs_lab(&self) -> bool {
        self.gated || !OSI_LICENCES.contains(&self.licence.as_str())
    }
}

/// Why a fixture's bytes are not handed out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadError {
    /// No entry of that name in the manifest.
    Unknown(String),
    /// The entry has no BLAKE3 yet.
    NotImported(String),
    /// The entry is gated or not OSI-licensed, and this runner has no lab credential.
    LabOnly(String),
    /// The cache has no object for it.
    Missing { name: String, path: PathBuf },
    /// The cached bytes are not the ones the manifest names.
    Integrity { name: String, detail: String },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Unknown(n) => write!(f, "{n}: no such fixture in testdata/fixtures.toml"),
            LoadError::NotImported(n) => write!(f, "{n}: recorded but not imported yet"),
            LoadError::LabOnly(n) => write!(
                f,
                "{n}: gated or not OSI-licensed; read only with the lab credential \
                 (SYNTROPY_FIXTURE_LAB=1)"
            ),
            LoadError::Missing { name, path } => write!(
                f,
                "{name}: not in the cache ({}); run `cargo xtask fixtures fetch --name {name}` \
                 with --store, --cache and --cap",
                path.display()
            ),
            LoadError::Integrity { name, detail } => write!(f, "{name}: {detail}"),
        }
    }
}

/// The fixtures of a `fixtures.toml` text.
pub fn parse_manifest(text: &str) -> Result<Vec<Fixture>, String> {
    let mut out: Vec<Fixture> = Vec::new();
    // The table open: none yet, a `[[fixture]]`, or another kind (`[[reference]]`, …), whose
    // keys are passed over.
    let mut table: Option<bool> = None;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let at = |what: &str| format!("fixtures.toml:{}: {what}", i + 1);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            let fixture = line == "[[fixture]]";
            if fixture {
                out.push(Fixture::default());
            }
            table = Some(fixture);
            continue;
        }
        match table {
            None => return Err(at("a key outside any table")),
            Some(false) => continue,
            Some(true) => {}
        }
        let f = out.last_mut().expect("a [[fixture]] table is open");
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| at("not `key = value`"))?;
        let (key, value) = (key.trim(), value.trim());
        match key {
            "name" => f.name = string(value).ok_or_else(|| at("name is not a string"))?,
            "blake3" => f.blake3 = string(value).ok_or_else(|| at("blake3 is not a string"))?,
            "licence" => f.licence = string(value).ok_or_else(|| at("licence is not a string"))?,
            "bytes" => f.bytes = value.parse().map_err(|_| at("bytes is not an integer"))?,
            "gated" => {
                f.gated = match value {
                    "true" => true,
                    "false" => false,
                    _ => return Err(at("gated is not a boolean")),
                }
            }
            "source" => {
                let inner = value
                    .strip_prefix('{')
                    .and_then(|v| v.strip_suffix('}'))
                    .ok_or_else(|| at("source is not an inline table"))?;
                for pair in inner.split(',') {
                    let (k, v) = pair
                        .split_once('=')
                        .ok_or_else(|| at("a source field is not `key = value`"))?;
                    let v = string(v.trim()).ok_or_else(|| at("a source field is not a string"))?;
                    match k.trim() {
                        "hub" => f.hub = v,
                        "revision" => f.revision = v,
                        "file" => f.file = v,
                        _ => {}
                    }
                }
            }
            "tier" | "used_by" | "quant" | "quantised_from" => {}
            other => return Err(at(&format!("unknown key `{other}`"))),
        }
    }
    Ok(out)
}

/// A basic TOML string with no escapes, which is all `fixtures.toml` writes.
fn string(v: &str) -> Option<String> {
    let s = v.strip_prefix('"')?.strip_suffix('"')?;
    (!s.contains(['"', '\\'])).then(|| s.to_owned())
}

/// The fixture named `name`.
pub fn find<'a>(manifest: &'a [Fixture], name: &str) -> Result<&'a Fixture, LoadError> {
    manifest
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| LoadError::Unknown(name.to_owned()))
}

/// The bytes of `f` from the cache directory `cache`, after the licence rule and a fresh BLAKE3
/// and length check. `lab` says whether this runner holds the lab credential.
pub fn load(f: &Fixture, cache: &Path, lab: bool) -> Result<Vec<u8>, LoadError> {
    if f.blake3.is_empty() {
        return Err(LoadError::NotImported(f.name.clone()));
    }
    if f.needs_lab() && !lab {
        return Err(LoadError::LabOnly(f.name.clone()));
    }
    let valid = f.blake3.len() == 64
        && f.blake3
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if !valid {
        return Err(LoadError::Integrity {
            name: f.name.clone(),
            detail: format!("blake3 {:?} is not 64 lowercase hex digits", f.blake3),
        });
    }
    let path = cache.join(&f.blake3);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(LoadError::Missing {
                name: f.name.clone(),
                path,
            });
        }
        Err(e) => {
            return Err(LoadError::Integrity {
                name: f.name.clone(),
                detail: format!("{}: {e}", path.display()),
            });
        }
    };
    let actual = blake3::hash(&bytes).to_hex().to_string();
    if bytes.len() as u64 != f.bytes || actual != f.blake3 {
        return Err(LoadError::Integrity {
            name: f.name.clone(),
            detail: format!(
                "the cache holds {} bytes with blake3 {actual}; the manifest names {} bytes with \
                 blake3 {}",
                bytes.len(),
                f.bytes,
                f.blake3
            ),
        });
    }
    Ok(bytes)
}
