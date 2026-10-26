// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `testdata/fixtures.toml` schema (RFC-0060 §12.3) and the edit `import` makes to it.

use serde::Deserialize;

use super::store::{Error, Key};

/// The manifest: every `[[fixture]]`. `[[reference]]` tables are read by their own tasks and kept
/// here as opaque values.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    #[serde(default, rename = "fixture")]
    pub fixtures: Vec<Fixture>,
    #[expect(dead_code, reason = "read by the reference tasks of QUAL-E5")]
    #[serde(default, rename = "reference")]
    pub references: Vec<toml::Value>,
}

/// One `[[fixture]]`. `source`, `licence`, `gated` and `used_by` are checked by the schema and
/// read by the fixture lints (QUAL-E7-T06), not by the store client.
#[expect(
    dead_code,
    reason = "schema fields read by the fixture lints of QUAL-E7-T06"
)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub name: String,
    /// The content address, lowercase hex; empty until `import` fills it.
    #[serde(default)]
    pub blake3: String,
    /// The length in bytes; filled by `import`.
    #[serde(default)]
    pub bytes: u64,
    pub source: toml::Value,
    pub licence: String,
    pub gated: bool,
    #[serde(default)]
    pub used_by: Vec<String>,
}

impl Fixture {
    /// The store key. A fixture that was never imported has none, which is an error here so that
    /// `fetch` and `verify` never run against an empty hash.
    pub fn key(&self) -> Result<Key, Error> {
        super::fixture_key(&self.blake3)
    }
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Manifest, String> {
        let manifest: Manifest = toml::from_str(text).map_err(|e| e.to_string())?;
        for (i, f) in manifest.fixtures.iter().enumerate() {
            if manifest.fixtures[..i].iter().any(|g| g.name == f.name) {
                return Err(format!("fixture `{}` appears twice", f.name));
            }
        }
        Ok(manifest)
    }
}

/// `text` with `blake3` and `bytes` set in the `[[fixture]]` named `name`, every other byte
/// unchanged. The entry must exist: import fills a record someone wrote with its source and
/// licence, it does not invent one.
pub fn fill(text: &str, name: &str, hash: &str, bytes: u64) -> Result<String, String> {
    let manifest = Manifest::parse(text)?;
    if !manifest.fixtures.iter().any(|f| f.name == name) {
        return Err(format!("no fixture named `{name}` in the manifest"));
    }
    let mut out = String::with_capacity(text.len() + 80);
    let mut block = String::new();
    let mut done = (false, false);
    // A block runs from one table header to the next, so key order inside it does not matter.
    for line in text.split_inclusive('\n') {
        if line.trim().starts_with('[') {
            out.push_str(&fill_block(&block, name, hash, bytes, &mut done));
            block.clear();
        }
        block.push_str(line);
    }
    out.push_str(&fill_block(&block, name, hash, bytes, &mut done));
    if done != (true, true) {
        return Err(format!(
            "fixture `{name}` must have `blake3` and `bytes` lines to fill"
        ));
    }
    Ok(out)
}

fn fill_block(block: &str, name: &str, hash: &str, bytes: u64, done: &mut (bool, bool)) -> String {
    let is_target = block.trim_start().starts_with("[[fixture]]")
        && block
            .lines()
            .any(|l| key_of(l.trim()) == Some("name") && value_of(l.trim()) == Some(name));
    if !is_target {
        return block.to_owned();
    }
    let mut out = String::new();
    for line in block.split_inclusive('\n') {
        match key_of(line.trim()) {
            Some("blake3") => {
                out.push_str(&replace_value(line, &format!("\"{hash}\"")));
                done.0 = true;
            }
            Some("bytes") => {
                out.push_str(&replace_value(line, &bytes.to_string()));
                done.1 = true;
            }
            _ => out.push_str(line),
        }
    }
    out
}

fn key_of(line: &str) -> Option<&str> {
    line.split_once('=').map(|(k, _)| k.trim())
}

/// The string value of a `key = "value"` line, without a trailing comment.
fn value_of(line: &str) -> Option<&str> {
    let (_, v) = line.split_once('=')?;
    let v = v.trim();
    let rest = v.strip_prefix('"')?;
    rest.split_once('"').map(|(s, _)| s)
}

/// `line` with the text after `=` replaced by `value`, keeping the alignment before `=` and a
/// trailing comment.
fn replace_value(line: &str, value: &str) -> String {
    let (key, rest) = line
        .split_once('=')
        .expect("caller matched a key = value line");
    let comment = rest
        .find('#')
        .map(|i| format!("  {}", rest[i..].trim_end()))
        .unwrap_or_default();
    let newline = if line.ends_with('\n') { "\n" } else { "" };
    format!("{key}= {value}{comment}{newline}")
}
