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
//! - `agent-tooling` in `dco`: no commit message, and not the PR's title or body (the squash
//!   commit on `main`), names a tool of the lock on a line with a version, names a pinned tool
//!   and holds its version, or holds a pinned digest. Tooling versions belong in provenance, not
//!   in commit messages (ADR-0021 decision 1). The lock is the base branch's, so a tool the PR
//!   itself adds to the lock is not known to the check until it merges.
//!
//! The observed file has the same `[[tool]]` tables with `name`, `version` and `digest`. The
//! lock and the observed file come from the base branch's checkout and the agent runner, never
//! from the PR under check.

use std::collections::BTreeMap;

use crate::contrib::{self, Commit, PullRequest};
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

fn word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"-_.".contains(&b)
}

/// Whether `line` (lower case) names the tool: the name, not inside a longer word. A `-` or `_`
/// after it counts as an end only before a version (`tool-1.2.3`), and a `.` only at a sentence
/// end, so `tool-ext` and `tool.conf` are other names.
fn names_tool(line: &str, name: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(at) = line[from..].find(name) {
        let start = from + at;
        let end = start + name.len();
        from = start + 1;
        if start > 0 && word(bytes[start - 1]) {
            continue;
        }
        let rest = &bytes[end..];
        let digit_at = |i: usize| rest.get(i).is_some_and(u8::is_ascii_digit);
        let ends = match rest.first() {
            None => true,
            Some(b'-' | b'_') => digit_at(1) || (rest.get(1) == Some(&b'v') && digit_at(2)),
            Some(b'.') => !rest.get(1).is_some_and(u8::is_ascii_alphanumeric),
            Some(b) => !b.is_ascii_alphanumeric(),
        };
        if ends {
            return true;
        }
    }
    false
}

/// Whether `line` holds a version token: digits and at least one `.digits` group, with or
/// without a leading `v`, not inside a longer word or number, and not a section number (`§12.5`).
fn has_version_token(line: &str) -> bool {
    let bytes = line.as_bytes();
    (0..bytes.len()).any(|i| {
        let digits = if bytes[i] == b'v' { i + 1 } else { i };
        if !bytes.get(digits).is_some_and(u8::is_ascii_digit) {
            return false;
        }
        // `§` is two bytes in UTF-8, ending in 0xa7.
        if i > 0 && ((word(bytes[i - 1]) && !b"-_".contains(&bytes[i - 1])) || bytes[i - 1] == 0xa7)
        {
            return false;
        }
        let mut at = digits;
        while bytes.get(at).is_some_and(u8::is_ascii_digit) {
            at += 1;
        }
        bytes.get(at) == Some(&b'.') && bytes.get(at + 1).is_some_and(u8::is_ascii_digit)
    })
}

/// Whether `text` (lower case) holds `version` as a whole token, with or without a `v`: not run
/// on into more version characters, though a closing full stop may end a sentence.
fn holds_version(text: &str, version: &str) -> bool {
    let bare = version.strip_prefix('v').unwrap_or(version);
    let bytes = text.as_bytes();
    let mut from = 0;
    while let Some(at) = text[from..].find(bare) {
        let start = from + at;
        let end = start + bare.len();
        from = start + 1;
        let before = match start.checked_sub(1).map(|i| bytes[i]) {
            Some(b'v') => start.checked_sub(2).map(|i| bytes[i]),
            other => other,
        };
        if before.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'.') {
            continue;
        }
        let after = &bytes[end..];
        let runs_on = match after.first() {
            None => false,
            Some(b'.') => after.get(1).is_some_and(u8::is_ascii_alphanumeric),
            Some(&b) => word(b),
        };
        if !runs_on {
            return true;
        }
    }
    false
}

/// The tools whose version `text` holds, by name. A message holds a tool's version when:
/// - a line names a tool of the lock, pinned or planned, and holds any version token, which
///   catches the version a bump PR introduces, since the lock read is the base branch's;
/// - the message names a pinned tool anywhere and holds its pinned version anywhere;
/// - it holds a pinned tool's digest, with or without `sha256:`.
///
/// A plain version number, or a name alone, does not match.
fn tools_named_with_version<'l>(lock: &'l [Tool], text: &str) -> Vec<&'l str> {
    let lower = text.to_ascii_lowercase();
    lock.iter()
        .filter(|tool| {
            let on_a_line = lower
                .lines()
                .any(|line| names_tool(line, &tool.name) && has_version_token(line));
            let pinned = tool.pin.as_ref().is_some_and(|pin| {
                let digest = pin.digest.to_ascii_lowercase();
                let hex = digest.strip_prefix("sha256:").unwrap_or(&digest);
                lower.contains(hex)
                    || (names_tool(&lower, &tool.name)
                        && holds_version(&lower, &pin.version.to_ascii_lowercase()))
            });
            on_a_line || pinned
        })
        .map(|tool| tool.name.as_str())
        .collect()
}

fn problem(what: &str, tool: &str) -> String {
    format!(
        "{what} names the version of agent tooling `{tool}`; tooling versions go in the run's \
         provenance, not in commit messages (ADR-0021 decision 1)"
    )
}

/// The commits whose message holds a tool's version or a pinned digest.
pub fn commit_problems(lock: &[Tool], commits: &[Commit]) -> Vec<String> {
    let mut problems = Vec::new();
    for commit in commits {
        let short = commit.sha.get(..12).unwrap_or(&commit.sha);
        for tool in tools_named_with_version(lock, &commit.message) {
            problems.push(problem(&format!("commit {short}: the message"), tool));
        }
    }
    problems
}

/// The same check on the PR's title and body, which become the squash commit on `main`
/// (`ci/repo-settings/repository.json`: `PR_TITLE` and `PR_BODY`).
pub fn pull_request_problems(lock: &[Tool], pr: &PullRequest) -> Vec<String> {
    let mut problems = Vec::new();
    for (what, text) in [("the PR title", &pr.title), ("the PR body", &pr.body)] {
        for tool in tools_named_with_version(lock, text) {
            problems.push(problem(what, tool));
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
        // Said aloud, so a passing step is not read as a verified run.
        eprintln!("agent-tooling: verify: no `--observed` given, so no run's tooling was checked");
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
    // The record is of what the run has, so it needs the run's own list; the lock alone would
    // record tooling nobody observed.
    if context.args.get("--observed").is_none() {
        return Err(
            "`--provenance` needs `--observed`: the record is of the observed tooling".to_owned(),
        );
    }
    // A run that would be refused records nothing.
    verify_step(context)?;
    let lock = load(context)?;
    std::fs::write(path, provenance(&lock)).map_err(|e| format!("--provenance {path}: {e}"))
}

/// The `dco` job's step: no commit message, and not the PR's title or body, holds a tooling
/// version.
pub fn commits_step(context: &Context) -> Result<(), String> {
    let lock = load(context)?;
    let read = |flag: &str| {
        let path = context
            .args
            .get(flag)
            .ok_or_else(|| format!("`{flag}` is required"))?;
        std::fs::read_to_string(path).map_err(|e| format!("{flag} {path}: {e}"))
    };
    let pr = contrib::parse_pull_request(&read("--event")?)?;
    let mut problems = pull_request_problems(&lock, &pr);
    problems.extend(commit_problems(
        &lock,
        &contrib::parse_commits(&read("--commits")?)?,
    ));
    report(problems)
}

#[cfg(test)]
mod tests;
