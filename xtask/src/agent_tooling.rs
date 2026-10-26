// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ci/agent-tooling.lock`, the pin of the agent runtime, plug-ins and tool servers, and the
//! checks that use it (ADR-0624 decision 5; RFC-0062 §12.5; QUAL-E1-T24).
//!
//! The lock holds one `[[tool]]` table per tool, the way Actions are pinned by SHA:
//!
//! ```toml
//! [[tool]]
//! name = "example-server"       # lowercase letters, digits, `-`, `_`, `.`
//! kind = "toolserver"           # or "runtime", "plugin"
//! version = "1.2.3"
//! digest = "sha256:<64 hex digits>"
//! about = "what it is for"
//!
//! [[tool]]
//! name = "other"
//! kind = "plugin"
//! about = "..."
//! planned = "QUAL-E1-T24"       # no version or digest yet: a maintainer pins it
//! ```
//!
//! A tool is either pinned (version and digest) or planned (neither). No digest is written down
//! that nobody has verified, so a tool whose digest is not known stays planned, and a run that
//! uses it is refused: unpinned tooling is tooling that differs from the lock.
//!
//! Three steps of the `agent-tooling` job, and one of `dco`, use it:
//! - `lock`: the file is well formed;
//! - `verify`: with `--observed <file>`, the tools a run is about to start with are exactly the
//!   pinned ones, by version and digest. Anything else is refused, so a run does not start;
//! - `record`: with `--provenance <file>`, the verified tooling is written as JSON, for the run's
//!   provenance record (RFC-0062 §6.3);
//! - `agent-tooling` in `dco`: no commit message names a pinned tool with its version, or holds
//!   a digest. Tooling versions belong in provenance, not in commit messages (ADR-0021 decision 1).
//!
//! The observed file has the same `[[tool]]` tables with `name`, `version` and `digest`. The
//! lock and the observed file come from the base branch's checkout and the agent runner, never
//! from the PR under check.

use std::collections::BTreeMap;

use crate::contrib::{self, Commit};
use crate::run::Context;

/// The lock, relative to the workspace root.
pub const LOCK: &str = "ci/agent-tooling.lock";

const KINDS: &[&str] = &["runtime", "plugin", "toolserver"];

/// One tool of the lock.
#[derive(Debug, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    pub kind: String,
    /// `None` for a planned tool.
    pub pin: Option<Pin>,
    pub planned: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Pin {
    pub version: String,
    pub digest: String,
}

fn text<'t>(table: &'t toml::Table, key: &str, what: &str) -> Result<Option<&'t str>, String> {
    match table.get(key) {
        None => Ok(None),
        Some(toml::Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(format!("{what}: `{key}` must be a string")),
    }
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"-_.".contains(&b))
}

fn valid_version(version: &str) -> bool {
    !version.is_empty()
        && version
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.+".contains(&b))
}

fn valid_digest(digest: &str) -> bool {
    digest.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    })
}

fn tables<'t>(value: &'t toml::Table, what: &str) -> Result<Vec<&'t toml::Table>, String> {
    let Some(tools) = value.get("tool") else {
        return Ok(Vec::new());
    };
    let toml::Value::Array(tools) = tools else {
        return Err(format!("{what}: `tool` must be an array of tables"));
    };
    tools
        .iter()
        .map(|t| {
            t.as_table()
                .ok_or_else(|| format!("{what}: each `tool` must be a table"))
        })
        .collect()
}

/// Parses and checks the lock.
pub fn parse(source: &str) -> Result<Vec<Tool>, String> {
    let top: toml::Table = source.parse().map_err(|e: toml::de::Error| {
        let at = e
            .span()
            .map_or(String::new(), |s| format!(" at byte {}", s.start));
        format!("{LOCK}: not valid TOML{at}: {}", e.message())
    })?;
    let mut problems = Vec::new();
    let mut tools: Vec<Tool> = Vec::new();
    for (i, table) in tables(&top, LOCK)?.into_iter().enumerate() {
        let what = format!("{LOCK}: tool {}", i + 1);
        let get = |key: &str, problems: &mut Vec<String>| match text(table, key, &what) {
            Ok(v) => v,
            Err(e) => {
                problems.push(e);
                None
            }
        };
        for key in table.keys() {
            if !["name", "kind", "version", "digest", "about", "planned"].contains(&key.as_str()) {
                problems.push(format!("{what}: unknown key `{key}`"));
            }
        }
        let name = get("name", &mut problems);
        let kind = get("kind", &mut problems);
        let version = get("version", &mut problems);
        let digest = get("digest", &mut problems);
        let planned = get("planned", &mut problems);
        if get("about", &mut problems).is_none_or(str::is_empty) {
            problems.push(format!("{what}: `about` is required"));
        }
        let Some(name) = name.filter(|n| valid_name(n)) else {
            problems.push(format!(
                "{what}: `name` must be lowercase letters, digits, `-`, `_` or `.`"
            ));
            continue;
        };
        let what = format!("{LOCK}: tool `{name}`");
        if tools.iter().any(|t| t.name == name) {
            problems.push(format!("{what}: listed twice"));
        }
        let kind = kind.unwrap_or("");
        if !KINDS.contains(&kind) {
            problems.push(format!(
                "{what}: `kind` must be one of {}",
                KINDS.join(", ")
            ));
        }
        let pin = match (version, digest, planned) {
            (Some(version), Some(digest), None) => {
                if !valid_version(version) {
                    problems.push(format!("{what}: `version` is not a plain version string"));
                }
                if !valid_digest(digest) {
                    problems.push(format!(
                        "{what}: `digest` must be `sha256:` and 64 lowercase hex digits"
                    ));
                }
                Some(Pin {
                    version: version.to_owned(),
                    digest: digest.to_owned(),
                })
            }
            (None, None, Some(task)) if !task.is_empty() => None,
            _ => {
                problems.push(format!(
                    "{what}: give both `version` and `digest`, or neither with `planned` \
                     naming the task that pins it"
                ));
                None
            }
        };
        tools.push(Tool {
            name: name.to_owned(),
            kind: kind.to_owned(),
            pin,
            planned: planned.map(str::to_owned),
        });
    }
    if problems.is_empty() {
        Ok(tools)
    } else {
        Err(problems.join("\n"))
    }
}

/// Parses an observed file: `name`, `version` and `digest` per tool.
pub fn parse_observed(source: &str) -> Result<BTreeMap<String, Pin>, String> {
    let top: toml::Table = source.parse().map_err(|e: toml::de::Error| {
        format!("observed tooling: not valid TOML: {}", e.message())
    })?;
    let mut observed = BTreeMap::new();
    for table in tables(&top, "observed tooling")? {
        let what = "observed tooling";
        let field = |key: &str| {
            text(table, key, what)?.ok_or_else(|| format!("{what}: a tool lacks `{key}`"))
        };
        let name = field("name")?;
        let pin = Pin {
            version: field("version")?.to_owned(),
            digest: field("digest")?.to_owned(),
        };
        if observed.insert(name.to_owned(), pin).is_some() {
            return Err(format!("{what}: `{name}` is listed twice"));
        }
    }
    Ok(observed)
}

/// What differs between the tooling a run would start with and the lock. Empty means it may start.
pub fn verify(lock: &[Tool], observed: &BTreeMap<String, Pin>) -> Vec<String> {
    let mut problems = Vec::new();
    for (name, seen) in observed {
        match lock.iter().find(|t| &t.name == name) {
            None => problems.push(format!("`{name}` is not in {LOCK}")),
            Some(Tool { pin: None, .. }) => {
                problems.push(format!("`{name}` is in {LOCK} but not pinned yet"));
            }
            Some(Tool { pin: Some(pin), .. }) => {
                if pin.version != seen.version {
                    problems.push(format!(
                        "`{name}` is version {}; the lock pins {}",
                        seen.version, pin.version
                    ));
                }
                if pin.digest != seen.digest {
                    problems.push(format!(
                        "`{name}` has digest {}; the lock pins {}",
                        seen.digest, pin.digest
                    ));
                }
            }
        }
    }
    for tool in lock.iter().filter(|t| t.pin.is_some()) {
        if !observed.contains_key(&tool.name) {
            problems.push(format!(
                "`{}` is pinned but the run does not have it",
                tool.name
            ));
        }
    }
    problems
}

/// The provenance record of the tooling: the pinned tools, as JSON.
pub fn provenance(lock: &[Tool]) -> String {
    let tools: Vec<serde_json::Value> = lock
        .iter()
        .filter_map(|t| {
            let pin = t.pin.as_ref()?;
            Some(serde_json::json!({
                "name": t.name,
                "kind": t.kind,
                "version": pin.version,
                "digest": pin.digest,
            }))
        })
        .collect();
    let mut out = serde_json::to_string_pretty(&serde_json::json!({ "agent_tooling": tools }))
        .expect("a JSON value serialises");
    out.push('\n');
    out
}

/// Whether `message` names the tool together with its pinned version: the name, then any of
/// ` @-_:=/v`, then the version, which does not run on into more version characters. A plain
/// version number, or a name alone, does not match.
fn names_with_version(message: &str, name: &str, version: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    let version = version.to_ascii_lowercase();
    let word = |b: u8| b.is_ascii_alphanumeric() || b"-_.".contains(&b);
    let bytes = lower.as_bytes();
    let mut from = 0;
    while let Some(at) = lower[from..].find(name) {
        let start = from + at;
        let end = start + name.len();
        from = start + 1;
        if start > 0 && word(bytes[start - 1]) {
            continue;
        }
        let rest = lower[end..].trim_start_matches(|c| " @-_:=/v".contains(c));
        if rest.len() == lower.len() - end {
            continue;
        }
        if let Some(after) = rest.strip_prefix(&version) {
            // A closing full stop ends a sentence; `.0` or `-rc1` makes it another version.
            let sentence_end = after.strip_prefix('.').is_some_and(|rest| {
                !rest
                    .bytes()
                    .next()
                    .is_some_and(|b| b.is_ascii_alphanumeric())
            });
            if sentence_end || !after.bytes().next().is_some_and(word) {
                return true;
            }
        }
    }
    false
}

/// The commits whose message holds a pinned tool's version or digest.
pub fn commit_problems(lock: &[Tool], commits: &[Commit]) -> Vec<String> {
    let mut problems = Vec::new();
    for commit in commits {
        let short = commit.sha.get(..12).unwrap_or(&commit.sha);
        for tool in lock {
            let Some(pin) = &tool.pin else { continue };
            let digest = pin.digest.to_ascii_lowercase();
            if commit.message.to_ascii_lowercase().contains(&digest)
                || names_with_version(&commit.message, &tool.name, &pin.version)
            {
                problems.push(format!(
                    "commit {short}: the message names the version of agent tooling `{}`; \
                     tooling versions go in the run's provenance, not in commit messages \
                     (ADR-0021 decision 1)",
                    tool.name
                ));
            }
        }
    }
    problems
}

fn load(context: &Context) -> Result<Vec<Tool>, String> {
    let path = context.root.join(LOCK);
    let source = std::fs::read_to_string(&path).map_err(|e| format!("{LOCK}: {e}"))?;
    parse(&source)
}

fn report(problems: Vec<String>) -> Result<(), String> {
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// `cargo xtask ci agent-tooling`, step `lock`.
pub fn lock_step(context: &Context) -> Result<(), String> {
    load(context).map(|_| ())
}

/// Step `verify`: with `--observed`, the run's tooling must be the lock's.
pub fn verify_step(context: &Context) -> Result<(), String> {
    let lock = load(context)?;
    let Some(path) = context.args.get("--observed") else {
        return Ok(());
    };
    let source = std::fs::read_to_string(path).map_err(|e| format!("--observed {path}: {e}"))?;
    let problems = verify(&lock, &parse_observed(&source)?);
    report(
        problems
            .into_iter()
            .map(|p| format!("refusing to start: {p}"))
            .collect(),
    )
}

/// Step `record`: with `--provenance`, write the pinned tooling there, unless `verify` refuses.
pub fn record_step(context: &Context) -> Result<(), String> {
    let Some(path) = context.args.get("--provenance") else {
        return Ok(());
    };
    // A run that would be refused records nothing.
    verify_step(context)?;
    let lock = load(context)?;
    std::fs::write(path, provenance(&lock)).map_err(|e| format!("--provenance {path}: {e}"))
}

/// The `dco` job's step: no commit message holds a pinned tooling version.
pub fn commits_step(context: &Context) -> Result<(), String> {
    let lock = load(context)?;
    let path = context
        .args
        .get("--commits")
        .ok_or_else(|| "`--commits` is required".to_owned())?;
    let source = std::fs::read_to_string(path).map_err(|e| format!("--commits {path}: {e}"))?;
    report(commit_problems(&lock, &contrib::parse_commits(&source)?))
}

#[cfg(test)]
mod tests;
