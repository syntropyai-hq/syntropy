// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ci/agent-instructions.toml`, the list of every file agents load as instructions or
//! configuration, and `cargo xtask instructions`, the way agent tooling loads them
//! (RFC-0062 §12.3; ADR-0624 decisions 3 and 5; QUAL-E1-T19).
//!
//! A change to a listed file changes how every later change is written or reviewed, so it needs
//! a maintainer's approval. The list holds one `[[file]]` table per entry:
//!
//! ```toml
//! [[file]]
//! path = "AGENTS.md"         # root-relative with `/` separators; a directory ends in `/`
//! kind = "instructions"      # or "configuration"
//! about = "the rules every agent follows"
//! planned = "QUAL-E1-T24"    # optional: the task that creates it; until then it may be missing
//! ```
//!
//! The `agent-instructions` step of `lint` checks the list: its keys and values, that every entry
//! that is not planned exists, that the list names itself, and that `.github/CODEOWNERS` gives
//! every listed path a maintainer as its owner.
//!
//! `cargo xtask instructions --show <path>` prints one listed file and refuses any other path:
//! agent tooling loads instructions through it, so an unlisted file is never loaded as one.
//! `--list` prints the entries. With `--ref <rev>`, the list and the file both come from that
//! commit, so a run that trusts `main` reads `main`'s list and files whatever the working tree
//! holds: a PR can neither add its own file to the list nor change a listed file's text.

use std::path::Path;
use std::process::Command;

use crate::codeowners::{self, CODEOWNERS, ROSTER};
use crate::licences::toml;
use crate::roster::Roster;
use crate::run::Context;

/// Where the list lives, relative to the workspace root.
pub const LIST: &str = "ci/agent-instructions.toml";

/// The keys an entry may have; `path`, `kind` and `about` are required.
const KEYS: &[&str] = &["path", "kind", "about", "planned"];

/// One listed file or directory.
#[derive(Debug, PartialEq, Eq)]
pub struct Listed {
    pub line: usize,
    pub path: String,
    pub kind: String,
    pub about: String,
    /// The task that creates it, while it does not exist yet.
    pub planned: Option<String>,
}

impl Listed {
    /// Whether the entry covers `path`: the same file, or a file under a listed directory.
    pub fn covers(&self, path: &str) -> bool {
        match self.path.strip_suffix('/') {
            Some(dir) => path.strip_prefix(dir).is_some_and(|rest| {
                rest.strip_prefix('/')
                    .is_some_and(|name| !name.is_empty() && !name.ends_with('/'))
            }),
            None => self.path == path,
        }
    }
}

/// Parses the list's text. Every problem stops it, so a half-read list is never used.
pub fn parse(text: &str) -> Result<Vec<Listed>, Vec<String>> {
    let tables = toml::tables(text, "file").map_err(|error| vec![format!("{LIST}:{error}")])?;
    let mut entries = Vec::new();
    let mut problems = Vec::new();
    for table in &tables {
        let at = format!("{LIST}:{}", table.line);
        for (key, _) in &table.entries {
            if !KEYS.contains(&key.as_str()) {
                problems.push(format!(
                    "{at}: unknown key `{key}`; keys are {}",
                    KEYS.join(", ")
                ));
            }
        }
        let required = |key: &str, problems: &mut Vec<String>| match table.str(key) {
            Some(value) if !value.trim().is_empty() => value.to_owned(),
            _ => {
                problems.push(format!("{at}: `{key}` is missing or empty"));
                String::new()
            }
        };
        let path = required("path", &mut problems);
        let kind = required("kind", &mut problems);
        let about = required("about", &mut problems);
        if !path.is_empty() && !well_formed(&path) {
            problems.push(format!(
                "{at}: `{path}` is not a root-relative path with `/` separators"
            ));
        }
        if !kind.is_empty() && kind != "instructions" && kind != "configuration" {
            problems.push(format!(
                "{at}: `kind` is `{kind}`; it is `instructions` or `configuration`"
            ));
        }
        let planned = table.str("planned").map(str::to_owned);
        if let Some(task) = &planned
            && !is_task_id(task)
        {
            problems.push(format!("{at}: `planned` is `{task}`, not a task ID"));
        }
        entries.push(Listed {
            line: table.line,
            path,
            kind,
            about,
            planned,
        });
    }
    for (index, entry) in entries.iter().enumerate() {
        if entries[..index]
            .iter()
            .any(|other| other.path == entry.path)
        {
            problems.push(format!(
                "{LIST}:{}: `{}` is listed twice",
                entry.line, entry.path
            ));
        }
    }
    if problems.is_empty() {
        Ok(entries)
    } else {
        Err(problems)
    }
}

/// A root-relative path: `/`-separated names, none empty, `.` or `..`, no backslash, not
/// starting with `-`. A directory ends in one `/`.
pub fn well_formed(path: &str) -> bool {
    let body = path.strip_suffix('/').unwrap_or(path);
    !body.is_empty()
        && !path.starts_with('-')
        && !path.contains('\\')
        && body
            .split('/')
            .all(|name| !name.is_empty() && name != "." && name != "..")
}

/// `QUAL-E1-T24` and the like.
fn is_task_id(id: &str) -> bool {
    let parts: Vec<&str> = id.split('-').collect();
    let digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    matches!(parts.as_slice(), [area, epic, task]
        if ["PLC", "TIER", "DIST", "COMP", "PROD", "QUAL"].contains(area)
            && epic.strip_prefix('E').is_some_and(digits)
            && task.strip_prefix('T').is_some_and(|n| n.len() == 2 && digits(n)))
}

/// The `agent-instructions` step of `lint`: the list is well formed, its entries exist unless
/// planned, it names itself, and CODEOWNERS gives every entry a maintainer.
pub fn step(context: &Context) -> Result<(), String> {
    let root = &context.root;
    let read = |relative: &str| {
        std::fs::read_to_string(root.join(relative)).map_err(|error| format!("{relative}: {error}"))
    };
    let entries = parse(&read(LIST)?).map_err(|problems| problems.join("\n"))?;
    let rules = codeowners::parse(&read(CODEOWNERS)?).map_err(|problems| problems.join("\n"))?;
    let roster = Roster::parse(&read(ROSTER)?).map_err(|error| format!("{ROSTER}: {error}"))?;
    let problems = check(&entries, |path| exists(root, path), &rules, &roster);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// Whether `path` exists under `root` as what it says it is: a directory when it ends in `/`,
/// else a regular file. A symbolic link is neither.
fn exists(root: &Path, path: &str) -> bool {
    let full = root.join(path.trim_end_matches('/'));
    std::fs::symlink_metadata(&full).is_ok_and(|meta| {
        if path.ends_with('/') {
            meta.is_dir()
        } else {
            meta.is_file()
        }
    })
}

/// Every problem with `entries`, given which paths exist and the CODEOWNERS `rules`.
pub fn check(
    entries: &[Listed],
    exists: impl Fn(&str) -> bool,
    rules: &[codeowners::Rule],
    roster: &Roster,
) -> Vec<String> {
    let mut problems = Vec::new();
    if !entries.iter().any(|entry| entry.path == LIST) {
        problems.push(format!(
            "{LIST} does not list itself; it is agent configuration, and changing it needs a \
             maintainer too"
        ));
    }
    for entry in entries {
        let at = format!("{LIST}:{}: `{}`", entry.line, entry.path);
        match (&entry.planned, exists(&entry.path)) {
            (None, false) => problems.push(format!(
                "{at} does not exist; list it with `planned = \"<task>\"` until its task adds it"
            )),
            (Some(task), true) => problems.push(format!(
                "{at} exists, so it is no longer planned by {task}; drop its `planned` key"
            )),
            _ => {}
        }
        // A directory's owner is the rule that covers a file inside it.
        let probe = if entry.path.ends_with('/') {
            format!("{}probe", entry.path)
        } else {
            entry.path.clone()
        };
        let owned = codeowners::owning_rule(rules, &probe).is_some_and(|rule| {
            rule.owners.iter().any(|owner| {
                owner
                    .strip_prefix('@')
                    .is_some_and(|login| roster.is_maintainer(login))
            })
        });
        if !owned {
            problems.push(format!(
                "{at} has no CODEOWNERS rule that names a maintainer; every listed file needs \
                 a maintainer's approval to change (RFC-0062 §12.3)"
            ));
        }
    }
    problems
}

/// Where `cargo xtask instructions` reads from.
#[derive(Debug)]
pub enum Source<'a> {
    /// The working tree under the workspace root.
    Tree(&'a Path),
    /// A commit of the repository at the workspace root: the list and every file come from it.
    Commit { root: &'a Path, rev: &'a str },
}

/// The text of `path` from `source`: a regular file, never a directory or a symbolic link.
fn read(source: &Source<'_>, path: &str) -> Result<String, String> {
    match source {
        Source::Tree(root) => {
            let full = root.join(path);
            let meta = std::fs::symlink_metadata(&full).map_err(|e| format!("{path}: {e}"))?;
            if !meta.is_file() {
                return Err(format!("{path} is not a regular file"));
            }
            std::fs::read_to_string(&full).map_err(|e| format!("{path}: {e}"))
        }
        Source::Commit { root, rev } => {
            let commit = git(
                root,
                &[
                    "rev-parse",
                    "--verify",
                    "--quiet",
                    &format!("{rev}^{{commit}}"),
                ],
            )
            .map_err(|_| format!("`{rev}` is not a commit in this repository"))?;
            let commit = commit.trim();
            // `<mode> <type> <object>\t<path>`: only a regular file's blob is read.
            let entry = git(root, &["ls-tree", commit, "--", path])?;
            let (meta, _) = entry.split_once('\t').unwrap_or(("", ""));
            let fields: Vec<&str> = meta.split(' ').collect();
            match fields.as_slice() {
                [mode, "blob", object] if *mode == "100644" || *mode == "100755" => {
                    git(root, &["cat-file", "blob", object])
                }
                [] | [""] => Err(format!("{path} does not exist at {rev}")),
                _ => Err(format!("{path} is not a regular file at {rev}")),
            }
        }
    }
}

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|e| format!("could not start `git`: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "`git {}` failed: {}",
            args.join(" "),
            stderr.lines().next().unwrap_or("").trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|_| "git printed text that is not UTF-8".to_owned())
}

/// The listed entries, read from `source`.
pub fn load(source: &Source<'_>) -> Result<Vec<Listed>, String> {
    parse(&read(source, LIST)?).map_err(|problems| problems.join("\n"))
}

/// The text of `path` if the list from the same `source` covers it; otherwise why it is refused.
pub fn show(source: &Source<'_>, path: &str) -> Result<String, String> {
    if !well_formed(path) || path.ends_with('/') {
        return Err(format!(
            "refused: `{path}` is not a root-relative file path; agents load instructions only \
             from the files {LIST} lists (RFC-0062 §12.3)"
        ));
    }
    let entries = load(source)?;
    if !entries.iter().any(|entry| entry.covers(path)) {
        let at = match source {
            Source::Tree(_) => "the working tree".to_owned(),
            Source::Commit { rev, .. } => format!("`{rev}`"),
        };
        return Err(format!(
            "refused: `{path}` is not listed in {LIST} at {at}; agents load instructions only \
             from listed files (RFC-0062 §12.3)"
        ));
    }
    read(source, path)
}

#[cfg(test)]
mod tests;
