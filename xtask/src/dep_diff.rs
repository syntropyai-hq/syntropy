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
//!   not list.
//!
//! Path dependencies between workspace members are not new dependencies: `check-deps` judges
//! those edges (RFC-0001 §4.2).
//!
//! Its two steps:
//! - **`dep-diff`** lists what is new and fails unless a maintainer has approved it, by an
//!   approving review on the PR's current head commit, or by applying the `dependency-approved`
//!   label (its label history read as for `semver-major-approved`, RFC-0062 §8.3). The PR's data
//!   comes from `--event`, `--reviews` (`GET /pulls/{n}/reviews`, every page), `--label-events`
//!   (`GET /issues/{n}/events` with `--include`, every page) and `--roster`; without them, as in a
//!   local run, nothing can approve, and a new dependency fails.
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

use crate::contrib::{applied_by_maintainer, parse_label_events, parse_pull_request};
use crate::licences::manifest;
use crate::registry::cargo;
use crate::roster::Roster;
use crate::run::Context;

/// The label a maintainer applies to approve a new dependency.
pub const APPROVAL_LABEL: &str = "dependency-approved";

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
}

impl NewDependencies {
    pub fn is_empty(&self) -> bool {
        self.crates.is_empty() && self.vendored.is_empty()
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
    }
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
    Ok(compare(
        &base_crates,
        &head_crates,
        &vendored(&base)?,
        &vendored(&head)?,
    ))
}

/// A review of `GET /pulls/{n}/reviews`, reduced to what approval reads.
#[derive(Debug, PartialEq, Eq)]
pub struct Review {
    pub login: String,
    /// `APPROVED`, `CHANGES_REQUESTED`, `COMMENTED`, `DISMISSED` or `PENDING`.
    pub state: String,
    /// The commit the review was submitted on.
    pub commit: String,
}

/// The reviews in `text`, one or more JSON arrays as `gh api --paginate` prints pages, in the
/// order GitHub lists them, which is the order they were submitted.
pub fn parse_reviews(text: &str) -> Result<Vec<Review>, String> {
    let mut reviews = Vec::new();
    for page in serde_json::Deserializer::from_str(text).into_iter::<Value>() {
        let page = page.map_err(|e| format!("reviews: not JSON: {e}"))?;
        let items = page
            .as_array()
            .ok_or("reviews: a page is not a JSON array")?;
        for (i, item) in items.iter().enumerate() {
            let field = |path: &str| {
                item.pointer(path)
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| format!("reviews[{i}]: `{path}` is missing"))
            };
            reviews.push(Review {
                login: field("/user/login")?,
                state: field("/state")?,
                // A pending review has no commit yet; it approves nothing.
                commit: field("/commit_id").unwrap_or_default(),
            });
        }
    }
    Ok(reviews)
}

/// The maintainer whose approving review stands on `head`, if one does. A reviewer's latest
/// review that is not a plain comment is their verdict, as GitHub reads it; an approval of an
/// earlier commit does not cover later pushes.
pub fn approving_review(reviews: &[Review], head: &str, roster: &Roster) -> Option<String> {
    let mut verdicts: BTreeMap<&str, &Review> = BTreeMap::new();
    for review in reviews {
        if review.state != "COMMENTED" && review.state != "PENDING" {
            verdicts.insert(review.login.as_str(), review);
        }
    }
    verdicts
        .into_iter()
        .find(|(login, review)| {
            review.state == "APPROVED" && review.commit == head && roster.is_maintainer(login)
        })
        .map(|(login, _)| login.to_owned())
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
const PR_DATA: &[&str] = &["--event", "--reviews", "--label-events", "--roster"];

/// Who approved the new dependencies, or why nobody has.
fn approval(context: &Context) -> Result<String, String> {
    let given: Vec<&str> = PR_DATA
        .iter()
        .copied()
        .filter(|option| context.args.get(option).is_some())
        .collect();
    if given.is_empty() {
        return Err(
            "no PR data was given (`--event`, `--reviews`, `--label-events`, `--roster`), so \
             nothing can approve them"
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
    let reviews = parse_reviews(&read("--reviews")?)?;
    if let Some(login) = approving_review(&reviews, &head, &roster) {
        return Ok(format!("approved by `{login}`, in a review of {head}"));
    }
    let pr = parse_pull_request(&event)?;
    let events = parse_label_events(&read("--label-events")?)?;
    match applied_by_maintainer(APPROVAL_LABEL, &pr.labels, &events, &roster) {
        Ok(login) => Ok(format!(
            "approved by `{login}`, who applied `{APPROVAL_LABEL}`"
        )),
        Err(label) => Err(format!(
            "no maintainer has approved them: no maintainer's approving review on the head \
             commit {head}, and {label}"
        )),
    }
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
                 {reason}. A maintainer approves with an approving review on the head commit, or \
                 by applying `{APPROVAL_LABEL}`"
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
