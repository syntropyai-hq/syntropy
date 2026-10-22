// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask ci dep-diff`: a new direct dependency needs a maintainer's approval
//! (RFC-0062 §12.3, §11.4; ADR-0624 decision 5; QUAL-E1-T19).
//!
//! The job compares two checkouts of the workspace: `--base`, the PR's base commit, and
//! `--head`, the PR merged into it. A new direct dependency is one of:
//! - an external crate, named with its source, that some manifest in the head declares as a
//!   normal, build or dev dependency and no manifest in the base does. A version bump of a crate
//!   the base already uses is not new; the same name from another source, such as a git fork, is;
//! - a vendored component in `third_party/licences.toml`, by its `upstream`, that the base does
//!   not list;
//! - a setting that changes where dependencies come from, added or changed: a `[patch]` or
//!   `[replace]` table in the root manifest, or a `[source]`, `[registries]`, `[registry]`,
//!   `[patch]` or `[replace]` table in any `.cargo/config.toml`. A patch or a source replacement
//!   swaps a crate the workspace already uses for other code, so it needs the same approval. A
//!   configuration file this check cannot read with certainty, because it uses `include` or holds
//!   a multi-line string, counts on every PR while it stays.
//!
//! Path dependencies between workspace members are not new dependencies: `check-deps` judges
//! those edges (RFC-0001 §4.2).
//!
//! Its two steps:
//! - **`dep-diff`** lists what is new and fails unless a maintainer has approved it: a PR
//!   comment with the line `dep-diff approved <head SHA>`, naming the PR's current head commit in
//!   full. It counts only when a maintainer of `--roster` posted it from their own account and
//!   never edited it, so a push after it, or an edit, needs a new one. The PR's data comes from
//!   `--event`, `--comments` (`GET /issues/{n}/comments`, every page) and `--roster`; without
//!   them, as in a local run, nothing can approve, and a new dependency fails.
//! - **`registry`** looks up each new crates.io dependency and prints its first-publish date, its
//!   first recorded publisher and its owners, for the maintainer's review. It fails when the name
//!   does not resolve on crates.io, which catches a typosquatted or invented name, and when the
//!   registry cannot be reached. A git dependency has no registry entry; its URL is printed.
//!   `--registry <dir>` reads recorded responses instead: `<name>.json` and `<name>-owners.json`,
//!   a missing file meaning the name does not resolve.
//!
//! `pr-meta.yml` runs the job from a checkout of the base branch, so the PR cannot change how it
//! is judged. It reads the head's manifests with `cargo metadata --no-deps`, which resolves and
//! builds nothing.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use crate::licences::manifest;
use crate::registry::cargo;
use crate::roster::Roster;
use crate::run::Context;

/// The source cargo records for a crates.io dependency.
const CRATES_IO: &str = "registry+https://github.com/rust-lang/crates.io-index";

/// What the registry check sends as its User-Agent, as crates.io asks.
const USER_AGENT: &str = "syntropy-xtask (https://github.com/syntropyai-hq/syntropy)";

/// An external crate, as a manifest names it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Crate {
    pub name: String,
    /// cargo's source string: `registry+…` or `git+…`.
    pub source: String,
}

/// What the head adds over the base.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct NewDependencies {
    /// Each new crate, with the manifests that declare it: `<package> (<kind>)`.
    pub crates: BTreeMap<Crate, Vec<String>>,
    /// Each new vendored component's `upstream`.
    pub vendored: Vec<String>,
    /// Each setting that changes where dependencies come from, added or changed: a `[patch]` or
    /// `[replace]` table in the root manifest, or a source, registry or patch table in a
    /// `.cargo/config.toml`, as `<file>: <line>`.
    pub sources: Vec<String>,
}

impl NewDependencies {
    pub fn is_empty(&self) -> bool {
        self.crates.is_empty() && self.vendored.is_empty() && self.sources.is_empty()
    }

    /// One line per new dependency.
    pub fn lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = self
            .crates
            .iter()
            .map(|(krate, users)| {
                format!(
                    "new crate `{}` from {}, in {}",
                    krate.name,
                    shown_source(&krate.source),
                    users.join(", ")
                )
            })
            .collect();
        lines.extend(
            self.vendored
                .iter()
                .map(|upstream| format!("new vendored component from {upstream}")),
        );
        lines.extend(
            self.sources
                .iter()
                .map(|setting| format!("new dependency source setting in {setting}")),
        );
        lines
    }
}

fn shown_source(source: &str) -> String {
    if source == CRATES_IO {
        "crates.io".to_owned()
    } else {
        format!("`{source}`")
    }
}

/// The external direct dependencies a `cargo metadata --no-deps` document declares: each crate
/// with the members that name it.
pub fn declared(metadata: &str) -> Result<BTreeMap<Crate, Vec<String>>, String> {
    let value: Value =
        serde_json::from_str(metadata).map_err(|e| format!("cargo metadata: not JSON: {e}"))?;
    let packages = value
        .get("packages")
        .and_then(Value::as_array)
        .ok_or("cargo metadata: no `packages` array")?;
    let mut crates: BTreeMap<Crate, Vec<String>> = BTreeMap::new();
    for package in packages {
        let name = package
            .get("name")
            .and_then(Value::as_str)
            .ok_or("cargo metadata: a package has no name")?;
        let dependencies = package
            .get("dependencies")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("cargo metadata: `{name}` has no `dependencies` array"))?;
        for dependency in dependencies {
            // A path dependency has no source; it is a workspace edge, not an external crate.
            let Some(source) = dependency.get("source").and_then(Value::as_str) else {
                continue;
            };
            let crate_name = dependency
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("cargo metadata: a dependency of `{name}` has no name"))?;
            let kind = dependency
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("normal");
            let user = format!("{name} ({kind})");
            let users = crates
                .entry(Crate {
                    name: crate_name.to_owned(),
                    source: source.to_owned(),
                })
                .or_default();
            if !users.contains(&user) {
                users.push(user);
            }
        }
    }
    Ok(crates)
}

/// `cargo metadata --no-deps` for the workspace under `root`, run from `cwd`, so cargo reads no
/// configuration from `root` when that is the PR's checkout.
fn metadata(root: &Path, cwd: &Path) -> Result<String, String> {
    let manifest = root.join("Cargo.toml");
    let output = Command::new(cargo())
        .args([
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--offline",
        ])
        .arg("--manifest-path")
        .arg(&manifest)
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("could not start cargo: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "cargo metadata on {} failed: {}",
            manifest.display(),
            stderr
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("")
                .trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|_| "cargo metadata printed non-UTF-8 text".into())
}

/// The vendored components `third_party/licences.toml` under `root` lists, by `upstream`.
fn vendored(root: &Path) -> Result<BTreeSet<String>, String> {
    manifest::load(root)
        .map(|entries| entries.into_iter().map(|entry| entry.upstream).collect())
        .map_err(|problems| problems.join("\n"))
}

/// What `head` adds over `base`.
pub fn compare(
    base: &BTreeMap<Crate, Vec<String>>,
    head: &BTreeMap<Crate, Vec<String>>,
    base_vendored: &BTreeSet<String>,
    head_vendored: &BTreeSet<String>,
) -> NewDependencies {
    NewDependencies {
        crates: head
            .iter()
            .filter(|(krate, _)| !base.contains_key(*krate))
            .map(|(krate, users)| (krate.clone(), users.clone()))
            .collect(),
        vendored: head_vendored.difference(base_vendored).cloned().collect(),
        sources: Vec::new(),
    }
}

/// The tables of a Cargo manifest or configuration file that decide where dependencies come
/// from: in `Cargo.toml`, `[patch.*]` and `[replace]`; in `.cargo/config.toml`, those and
/// `[source.*]`, `[registries.*]`, `[registry]` and `include`, and the same keys written dotted
/// (`source.crates-io.replace-with = …`). A patch or a source replacement swaps a crate the
/// workspace already uses for other code without adding a dependency, so it needs the same
/// approval.
const MANIFEST_TABLES: &[&str] = &["patch", "replace"];
const CONFIG_TABLES: &[&str] = &[
    "patch",
    "replace",
    "source",
    "registries",
    "registry",
    "include",
];

/// Whether a TOML key path, a table header's contents or the text before a `=`, starts with one
/// of `tables`. The answer fails closed: a first segment that is quoted, so it could spell a
/// table name with escapes, or that is not a plain bare key, counts as one.
fn names_a_table(path: &str, tables: &[&str]) -> bool {
    let path = path.trim();
    if path.starts_with('"') || path.starts_with('\'') {
        return true;
    }
    let first = path
        .split(|c: char| c == '.' || c.is_whitespace())
        .next()
        .unwrap_or("");
    let bare = !first.is_empty()
        && first
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    !bare || tables.contains(&first)
}

/// The setting lines of `text` under `tables`, each as `[<header>] <line>` or `<line>` for a
/// key at the top, with comments and surrounding space removed. A file with a multi-line string
/// is read whole instead, as one setting that names its contents' digest, because such a string
/// could hold a line that looks like a table header and hide the lines after it.
pub fn source_settings(text: &str, tables: &[&str]) -> BTreeSet<String> {
    let mut settings = BTreeSet::new();
    if text.contains("\"\"\"") || text.contains("'''") {
        settings.insert(format!(
            "the whole file, which holds a multi-line string (digest {:016x})",
            fnv1a(text)
        ));
        return settings;
    }
    // `None` before any header, where top-level keys sit; `Some(None)` under another table.
    let mut header: Option<Option<String>> = None;
    for line in text.lines() {
        let line = crate::licences::toml::strip_comment(line).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            let name = line
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim()
                .to_owned();
            let matching = names_a_table(&name, tables).then_some(name);
            if let Some(name) = &matching {
                settings.insert(format!("[{name}]"));
            }
            header = Some(matching);
            continue;
        }
        match &header {
            Some(Some(name)) => {
                settings.insert(format!("[{name}] {line}"));
            }
            Some(None) => {}
            None => {
                let key = line.split('=').next().unwrap_or("");
                if names_a_table(key, tables) {
                    settings.insert(line.to_owned());
                }
            }
        }
    }
    settings
}

/// FNV-1a, 64 bits: a digest of a file compared whole, so a change shows without printing it.
fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// What a walk of a tree finds: every cargo configuration file, and every symbolic link to a
/// directory or named `.cargo`, which cargo follows and the walk does not.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Found {
    pub configs: Vec<String>,
    pub links: Vec<String>,
}

/// Every `.cargo/config.toml` and `.cargo/config` under `root`, root-relative, except under
/// `skip` (the head checkout, when it lies inside the base), `target/` and `.git/`. Test
/// fixtures' files are read too: telling them apart would rest on text the PR controls.
fn cargo_configs(root: &Path, skip: Option<&Path>) -> Result<Found, String> {
    let relative = |path: &Path| {
        path.strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    };
    let mut found = Found::default();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
            let path = entry.path();
            let name = entry.file_name();
            if skip.is_some_and(|skip| path == skip) || name == "target" || name == ".git" {
                continue;
            }
            let kind = entry
                .file_type()
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if kind.is_symlink() {
                if name == ".cargo" || path.is_dir() {
                    found.links.push(relative(&path));
                }
                continue;
            }
            if !kind.is_dir() {
                continue;
            }
            if name == ".cargo" {
                for file in ["config.toml", "config"] {
                    if path.join(file).is_file() {
                        found.configs.push(relative(&path.join(file)));
                    }
                }
            } else {
                pending.push(path);
            }
        }
    }
    found.configs.sort();
    found.links.sort();
    Ok(found)
}

/// Why a cargo configuration file cannot be read with certainty, if it cannot: it uses `include`,
/// which loads settings from other files, or it holds a multi-line string, which could hide a
/// table header. Such a file counts as a new setting on every PR while it stays.
pub fn unreadable(text: &str) -> Option<&'static str> {
    if text.contains("\"\"\"") || text.contains("\'\'\'") {
        return Some("it holds a multi-line string, which could hide a table header");
    }
    let includes = text.lines().any(|line| {
        let line = crate::licences::toml::strip_comment(line).trim();
        let path = match line.strip_prefix('[') {
            Some(header) => header.trim_matches(|c| c == '[' || c == ']'),
            None => line.split('=').next().unwrap_or(""),
        };
        !line.is_empty() && names_a_table(path, &["include"])
    });
    includes
        .then_some("it uses `include`, which loads settings from files this check does not follow")
}

/// Each source setting of the tree at `root`, as `<file>: <setting>`.
fn sources(root: &Path, skip: Option<&Path>) -> Result<BTreeSet<String>, String> {
    let read = |relative: &str| {
        std::fs::read_to_string(root.join(relative)).map_err(|e| format!("{relative}: {e}"))
    };
    let mut all: BTreeSet<String> = source_settings(&read("Cargo.toml")?, MANIFEST_TABLES)
        .into_iter()
        .map(|setting| format!("Cargo.toml: {setting}"))
        .collect();
    let found = cargo_configs(root, skip)?;
    for link in found.links {
        all.insert(format!(
            "{link}: a symbolic link to a directory, which cargo follows and this check does not"
        ));
    }
    for file in found.configs {
        let text = read(&file)?;
        all.extend(
            source_settings(&text, CONFIG_TABLES)
                .into_iter()
                .map(|setting| format!("{file}: {setting}")),
        );
    }
    Ok(all)
}

/// The new dependencies of the context's `--head` over its `--base`.
fn new_dependencies(context: &Context) -> Result<NewDependencies, String> {
    // Absolute, because cargo runs in the base and would read a relative path from there.
    let dir = |option: &str| {
        let given = context
            .args
            .get(option)
            .ok_or_else(|| format!("`{option}` is required"))?;
        std::path::absolute(given).map_err(|e| format!("{option} {given}: {e}"))
    };
    let base = dir("--base")?;
    let head = dir("--head")?;
    let base_crates = declared(&metadata(&base, &base)?)?;
    let head_crates = declared(&metadata(&head, &base)?)?;
    let mut new = compare(
        &base_crates,
        &head_crates,
        &vendored(&base)?,
        &vendored(&head)?,
    );
    let base_sources = sources(&base, head.starts_with(&base).then_some(head.as_path()))?;
    new.sources = sources(&head, None)?
        .difference(&base_sources)
        .cloned()
        .collect();
    // A file the check cannot read with certainty counts on every PR, whatever the base holds.
    for file in cargo_configs(&head, None)?.configs {
        let text = std::fs::read_to_string(head.join(&file)).map_err(|e| format!("{file}: {e}"))?;
        if let Some(why) = unreadable(&text) {
            new.sources
                .push(format!("{file}, compared on every PR because {why}"));
        }
    }
    Ok(new)
}

/// The line a maintainer's PR comment holds to approve the new dependencies of one commit:
/// `dep-diff approved <the head commit's full SHA>`, on a line of its own.
pub const APPROVAL: &str = "dep-diff approved";

/// A comment of `GET /issues/{n}/comments`, reduced to what approval reads.
#[derive(Debug, PartialEq, Eq)]
pub struct Comment {
    pub login: String,
    pub body: String,
    pub created_at: String,
    pub updated_at: String,
    /// Posted through a GitHub App, so no person posted it directly.
    pub via_app: bool,
}

/// The comments in `text`, one or more JSON arrays as `gh api --paginate` prints pages.
pub fn parse_comments(text: &str) -> Result<Vec<Comment>, String> {
    let mut comments = Vec::new();
    for page in serde_json::Deserializer::from_str(text).into_iter::<Value>() {
        let page = page.map_err(|e| format!("comments: not JSON: {e}"))?;
        let items = page
            .as_array()
            .ok_or("comments: a page is not a JSON array")?;
        for (i, item) in items.iter().enumerate() {
            let field = |path: &str| {
                item.pointer(path)
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| format!("comments[{i}]: `{path}` is missing"))
            };
            comments.push(Comment {
                login: field("/user/login")?,
                body: field("/body").unwrap_or_default(),
                created_at: field("/created_at")?,
                updated_at: field("/updated_at")?,
                via_app: item
                    .get("performed_via_github_app")
                    .is_some_and(|app| !app.is_null()),
            });
        }
    }
    Ok(comments)
}

/// The SHAs `body` approves: each line that reads exactly `dep-diff approved <40 hex digits>`.
pub fn approved_shas(body: &str) -> Vec<&str> {
    body.lines()
        .filter_map(|line| line.trim().strip_prefix(APPROVAL))
        .filter_map(|rest| rest.strip_prefix(' '))
        .filter(|sha| sha.len() == 40 && sha.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')))
        .collect()
}

/// The maintainer whose comment approves `head`, or why no comment does. A comment counts when a
/// maintainer of `roster` posted it from their own account, never edited it, and it names `head`
/// itself: one naming an earlier commit is stale, because a push may have added more.
pub fn approving_comment(
    comments: &[Comment],
    head: &str,
    roster: &Roster,
) -> Result<String, String> {
    let mut refused = Vec::new();
    for comment in comments {
        let shas = approved_shas(&comment.body);
        if shas.is_empty() {
            continue;
        }
        let who = &comment.login;
        if comment.via_app {
            refused.push(format!("`{who}` posted one through a GitHub App"));
        } else if roster.is_agent(who) {
            refused.push(format!("`{who}`, who posted one, is an agent identity"));
        } else if !roster.is_maintainer(who) {
            refused.push(format!("`{who}`, who posted one, is not a maintainer"));
        } else if comment.updated_at != comment.created_at {
            refused.push(format!(
                "`{who}` edited theirs, and an edited approval does not count; post a new one"
            ));
        } else if !shas.contains(&head) {
            refused.push(format!(
                "`{who}`'s names {}, not the head commit; a push after an approval needs a new one",
                shas.join(", ")
            ));
        } else {
            return Ok(who.clone());
        }
    }
    Err(if refused.is_empty() {
        format!("no maintainer has commented `{APPROVAL} {head}`")
    } else {
        format!(
            "no comment `{APPROVAL} {head}` counts: {}",
            refused.join("; ")
        )
    })
}

/// The PR's head commit, from its event payload.
fn head_sha(event: &str) -> Result<String, String> {
    let value: Value = serde_json::from_str(event).map_err(|e| format!("event: not JSON: {e}"))?;
    let pr = value.get("pull_request").unwrap_or(&value);
    pr.pointer("/head/sha")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "event: `head.sha` is missing".to_owned())
}

/// The PR data that can carry an approval, all or none of it given.
const PR_DATA: &[&str] = &["--event", "--comments", "--roster"];

/// Who approved the new dependencies, or why nobody has.
fn approval(context: &Context) -> Result<String, String> {
    let given: Vec<&str> = PR_DATA
        .iter()
        .copied()
        .filter(|option| context.args.get(option).is_some())
        .collect();
    if given.is_empty() {
        return Err(
            "no PR data was given (`--event`, `--comments`, `--roster`), so nothing can approve \
             them"
                .to_owned(),
        );
    }
    if given.len() != PR_DATA.len() {
        return Err(format!(
            "approval reads {} together, but only {} were given",
            PR_DATA.join(", "),
            given.join(", ")
        ));
    }
    let read = |option: &str| {
        let path = context.args.get(option).unwrap_or_default();
        std::fs::read_to_string(path).map_err(|e| format!("{option} {path}: {e}"))
    };
    let event = read("--event")?;
    let roster = Roster::parse(&read("--roster")?).map_err(|e| format!("roster: {e}"))?;
    let head = head_sha(&event)?;
    let comments = parse_comments(&read("--comments")?)?;
    approving_comment(&comments, &head, &roster)
        .map(|login| format!("approved by `{login}`, in a comment naming {head}"))
}

/// The `dep-diff` step.
pub fn step(context: &Context) -> Result<(), String> {
    let new = new_dependencies(context)?;
    if new.is_empty() {
        eprintln!("dep-diff: no new direct dependency or vendored component");
        return Ok(());
    }
    for line in new.lines() {
        eprintln!("dep-diff: {line}");
    }
    match approval(context) {
        Ok(approved) => {
            eprintln!("dep-diff: {approved}");
            Ok(())
        }
        Err(reason) => {
            let count = new.lines().len();
            let (what, needs) = if count == 1 {
                ("dependency", "needs")
            } else {
                ("dependencies", "need")
            };
            Err(format!(
                "{count} new direct {what} {needs} a maintainer's approval (RFC-0062 §12.3): \
                 {reason}. A maintainer approves by commenting `{APPROVAL} <head SHA>` on the \
                 PR, and a later push needs a new comment"
            ))
        }
    }
}

/// What the registry says about a crate.
#[derive(Debug, PartialEq, Eq)]
pub enum Lookup {
    Found {
        crate_json: String,
        owners_json: String,
    },
    Missing,
}

/// Where registry answers come from.
pub trait Registry {
    fn lookup(&mut self, name: &str) -> Result<Lookup, String>;
}

/// Recorded answers in a directory.
pub struct Recorded(pub PathBuf);

impl Registry for Recorded {
    fn lookup(&mut self, name: &str) -> Result<Lookup, String> {
        let read = |file: String| std::fs::read_to_string(self.0.join(&file));
        match (
            read(format!("{name}.json")),
            read(format!("{name}-owners.json")),
        ) {
            (Ok(crate_json), Ok(owners_json)) => Ok(Lookup::Found {
                crate_json,
                owners_json,
            }),
            (Err(_), _) => Ok(Lookup::Missing),
            (Ok(_), Err(e)) => Err(format!("recorded owners of `{name}`: {e}")),
        }
    }
}

/// crates.io's API, through `curl`.
pub struct CratesIo;

impl CratesIo {
    /// The body and status of a GET.
    fn get(url: &str) -> Result<(u16, String), String> {
        let output = Command::new("curl")
            .args([
                "--silent",
                "--show-error",
                "--proto",
                "=https",
                "--max-time",
                "30",
            ])
            .args(["--header", &format!("User-Agent: {USER_AGENT}")])
            .args(["--write-out", "\n%{http_code}", url])
            .output()
            .map_err(|e| format!("could not start curl: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "could not reach crates.io: {}",
                stderr.lines().next().unwrap_or("").trim()
            ));
        }
        let text = String::from_utf8_lossy(&output.stdout).into_owned();
        let (body, status) = text.rsplit_once('\n').unwrap_or(("", &text));
        let status = status
            .trim()
            .parse()
            .map_err(|_| format!("curl printed no HTTP status for {url}"))?;
        Ok((status, body.to_owned()))
    }
}

impl Registry for CratesIo {
    fn lookup(&mut self, name: &str) -> Result<Lookup, String> {
        let base = format!("https://crates.io/api/v1/crates/{name}");
        match CratesIo::get(&base)? {
            (200, crate_json) => match CratesIo::get(&format!("{base}/owners"))? {
                (200, owners_json) => Ok(Lookup::Found {
                    crate_json,
                    owners_json,
                }),
                (status, _) => Err(format!(
                    "crates.io answered {status} for the owners of `{name}`"
                )),
            },
            (404, _) => Ok(Lookup::Missing),
            (status, _) => Err(format!("crates.io answered {status} for `{name}`")),
        }
    }
}

/// A crate name crates.io could hold, so it is safe in a URL.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.starts_with(|c: char| c.is_ascii_alphabetic())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// One line for the maintainer: first publish, first recorded publisher, owners.
pub fn describe(name: &str, crate_json: &str, owners_json: &str) -> Result<String, String> {
    let value: Value =
        serde_json::from_str(crate_json).map_err(|e| format!("`{name}`: not JSON: {e}"))?;
    let created = value
        .pointer("/crate/created_at")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("`{name}`: crates.io gave no `crate.created_at`"))?;
    let date = |at: &str| at.split('T').next().unwrap_or(at).to_owned();
    let mut versions: Vec<&Value> = value
        .get("versions")
        .and_then(Value::as_array)
        .map(|versions| versions.iter().collect())
        .unwrap_or_default();
    versions.sort_by_key(|v| v.get("created_at").and_then(Value::as_str).unwrap_or(""));
    let login = |v: &Value| {
        v.pointer("/published_by/login")
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    let num = |v: &Value| {
        v.get("num")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_owned()
    };
    let publisher = match versions.first() {
        Some(first) => match login(first) {
            Some(by) => format!("by `{by}` (version {})", num(first)),
            None => match versions.iter().find(|v| login(v).is_some()) {
                Some(v) => format!(
                    "publisher not recorded; the first recorded one is `{}` (version {}, {})",
                    login(v).unwrap_or_default(),
                    num(v),
                    date(v.get("created_at").and_then(Value::as_str).unwrap_or(""))
                ),
                None => "no publisher recorded".to_owned(),
            },
        },
        None => "no versions listed".to_owned(),
    };
    let owners: Value =
        serde_json::from_str(owners_json).map_err(|e| format!("`{name}` owners: not JSON: {e}"))?;
    let owners: Vec<String> = owners
        .get("users")
        .and_then(Value::as_array)
        .map(|users| {
            users
                .iter()
                .filter_map(|u| u.get("login").and_then(Value::as_str))
                .map(|login| format!("`{login}`"))
                .collect()
        })
        .unwrap_or_default();
    Ok(format!(
        "`{name}` on crates.io: first published {}, {publisher}; owners {}",
        date(created),
        if owners.is_empty() {
            "none listed".to_owned()
        } else {
            owners.join(", ")
        }
    ))
}

/// Every problem the registry check finds for `new`, printing what it learns.
pub fn check_registry(new: &NewDependencies, registry: &mut dyn Registry) -> Vec<String> {
    let mut problems = Vec::new();
    for krate in new.crates.keys() {
        if krate.source != CRATES_IO {
            if krate.source.starts_with("git+") {
                eprintln!(
                    "registry: `{}` comes from git, {}; it has no registry entry to check, so \
                     the maintainer reviews that repository",
                    krate.name, krate.source
                );
            } else {
                problems.push(format!(
                    "`{}` comes from `{}`, which is not crates.io; the registry check covers \
                     crates.io only",
                    krate.name, krate.source
                ));
            }
            continue;
        }
        if !valid_name(&krate.name) {
            problems.push(format!(
                "`{}` is not a crate name crates.io accepts",
                krate.name
            ));
            continue;
        }
        match registry.lookup(&krate.name) {
            Ok(Lookup::Found {
                crate_json,
                owners_json,
            }) => match describe(&krate.name, &crate_json, &owners_json) {
                Ok(line) => eprintln!("registry: {line}"),
                Err(problem) => problems.push(problem),
            },
            Ok(Lookup::Missing) => problems.push(format!(
                "`{}` does not resolve on crates.io: no crate of that name exists, so the name is \
                 misspelt, invented or not yet published (ADR-0624 decision 5)",
                krate.name
            )),
            Err(problem) => problems.push(problem),
        }
    }
    problems
}

/// The `registry` step.
pub fn registry_step(context: &Context) -> Result<(), String> {
    let new = new_dependencies(context)?;
    if new.crates.is_empty() {
        eprintln!("registry: no new crate to look up");
        return Ok(());
    }
    let mut registry: Box<dyn Registry> = match context.args.get("--registry") {
        Some(dir) => Box::new(Recorded(PathBuf::from(dir))),
        None => Box::new(CratesIo),
    };
    let problems = check_registry(&new, registry.as_mut());
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

#[cfg(test)]
mod tests;
