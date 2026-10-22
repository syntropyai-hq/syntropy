// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask ci changes --base <rev> --head <rev>`: the affected-job set of RFC-0062 §2.2
//! (ADR-0620 decision 3; QUAL-E1-T07).
//!
//! The rules live in `ci/jobs.toml`. Each changed path falls in one class, tried in this order:
//! - **workspace**: a path every crate builds or runs with, such as `Cargo.lock`, the toolchain
//!   pin, `xtask` (which defines every job's steps) or `pr.yml`. It counts as a change to every
//!   crate;
//! - **crate**: a path inside a workspace member's directory. It counts as a change to that
//!   crate;
//! - **inputs**: a path outside every crate that some crates read, such as `numerics/` for
//!   `kernels`. It counts as a change to those crates;
//! - **no crate**: a path no crate's build or tests read, such as `docs/` and `plan/`. It
//!   changes no crate, so only the jobs that run whatever changed run;
//! - **unclassified**: any other path. It counts as a change to every crate, so a path the rules
//!   do not know fails closed.
//!
//! Each job has one rule:
//! - `always`: it runs on every change (`lint`);
//! - `closure`: it runs when the reverse-dependency closure of the changed crates is not empty,
//!   or, with `crates`, when the closure includes one of them. Hosted jobs use it, so a change to
//!   `core` runs every hosted test;
//! - `direct`: it runs when a changed path matches one of its `paths`, or when `Cargo.lock`
//!   changes the entry of a package in its `lock_deps`. The closure plays no part. Hardware jobs
//!   use it, so a `core`-only change does not occupy the lab.
//!
//! A job marked `planned` has no suite or rig yet: its rule is evaluated and reported, but a
//! hosted one is not listed, and `ci-gate` does not hold a same-repository PR for a hardware one.
//!
//! The crate graph is read from the manifests as TOML at both revisions, and the closure is taken
//! over the union of the two graphs, so an edge the PR adds or removes counts. Nothing from the PR
//! runs: the diff and the PR's files are read through git, and no cargo command reads the PR's
//! tree, whose cargo configuration could otherwise start a program in this job.
//!
//! `pr.yml` runs this command from a checkout of the base branch, built with no cache, as it runs
//! `ci-gate` (RFC-0062 §2.2, note 2026-10-22), so a PR cannot make it list nothing. The job
//! appends `jobs` (the hosted jobs to run) and `hw_required` (the hardware jobs whose `hw/<job>`
//! statuses `ci-gate` waits for) to `--github-output`.
//!
//! `lint`'s `jobs` step checks the rules against the golden suite in `ci/jobs-goldens.toml`:
//! each recorded diff must give exactly its expected sets, every rule and pattern must be met by
//! some diff, and the suite must record the digest of the `ci/jobs.toml` it was checked against,
//! so a change to the rules without a change to the goldens fails.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use serde::Deserialize;

use crate::run::Context;

/// The rules, relative to the workspace root.
pub const CONFIG: &str = "ci/jobs.toml";

/// The golden suite the rules are checked against.
pub const GOLDENS: &str = "ci/jobs-goldens.toml";

/// The workflow whose jobs the rules cover.
pub const WORKFLOW: &str = ".github/workflows/pr.yml";

/// The jobs of [`WORKFLOW`] that are not themselves listed: `changes` decides, `ci-gate` judges.
const UNLISTED: &[&str] = &["changes", "ci-gate"];

/// Workspace packages are named `syntropy-<crate>`; the rules use `<crate>`.
const PREFIX: &str = "syntropy-";

// ---------------------------------------------------------------------------------------------
// Path patterns.

/// A path pattern: segments separated by `/`, where `**` matches any number of whole segments,
/// none included, and `*` matches any run of characters within one segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glob {
    text: String,
    segments: Vec<String>,
}

impl Glob {
    pub fn parse(text: &str) -> Result<Glob, String> {
        let bad = |why: &str| Err(format!("`{text}`: {why}"));
        if text.is_empty() {
            return bad("an empty pattern");
        }
        if text.starts_with('/') || text.ends_with('/') || text.contains("//") {
            return bad("a pattern is a relative path with no empty segment");
        }
        if text.contains(['\\', '?', '[', ']', '{', '}', '!']) {
            return bad("only `*` and `**` are special");
        }
        let segments: Vec<String> = text.split('/').map(str::to_owned).collect();
        if segments.iter().any(|s| s == "." || s == "..") {
            return bad("`.` and `..` are not segments of a path");
        }
        if segments.iter().any(|s| s.contains("**") && s != "**") {
            return bad("`**` stands alone as a segment");
        }
        Ok(Glob {
            text: text.to_owned(),
            segments,
        })
    }

    pub fn matches(&self, path: &str) -> bool {
        let parts: Vec<&str> = path.split('/').collect();
        matches_segments(&self.segments, &parts)
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }
}

fn matches_segments(globs: &[String], parts: &[&str]) -> bool {
    match globs.split_first() {
        None => parts.is_empty(),
        Some((glob, rest)) if glob == "**" => {
            (0..=parts.len()).any(|skip| matches_segments(rest, &parts[skip..]))
        }
        Some((glob, rest)) => parts.split_first().is_some_and(|(part, after)| {
            crate::codeowners::glob_matches(glob, part) && matches_segments(rest, after)
        }),
    }
}

// ---------------------------------------------------------------------------------------------
// The rules.

/// Where a job runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Runner {
    /// A GitHub-hosted runner: a job of `pr.yml`.
    Hosted,
    /// A lab rig, through the dispatcher: a `hw/<job>` status `ci-gate` waits for.
    Lab,
}

/// When a job runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum When {
    Always,
    /// When the closure is not empty, or includes one of `crates`.
    Closure {
        crates: Option<BTreeSet<String>>,
    },
    /// When a changed path matches one of `paths`, or `Cargo.lock` changes one of `lock_deps`.
    Direct {
        paths: Vec<Glob>,
        lock_deps: BTreeSet<String>,
    },
}

/// One job's rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    pub name: String,
    pub runner: Runner,
    pub when: When,
    /// The task that adds the job's suite or rig, while it has none.
    pub planned: Option<String>,
}

/// `ci/jobs.toml`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub workspace: Vec<Glob>,
    pub no_crate: Vec<Glob>,
    /// Paths outside every crate, each with the crates that read it.
    pub inputs: Vec<(Glob, BTreeSet<String>)>,
    /// The accounts whose `hw/<job>` statuses `ci-gate` accepts: the lab dispatcher's.
    pub status_creators: BTreeSet<String>,
    /// By name.
    pub jobs: Vec<Job>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    paths: RawPaths,
    #[serde(default)]
    inputs: BTreeMap<String, Vec<String>>,
    hardware: RawHardware,
    job: BTreeMap<String, RawJob>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPaths {
    workspace: Vec<String>,
    no_crate: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHardware {
    status_creators: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawJob {
    runner: String,
    when: String,
    crates: Option<Vec<String>>,
    #[serde(default)]
    paths: Vec<String>,
    #[serde(default)]
    lock_deps: Vec<String>,
    planned: Option<String>,
}

impl Config {
    /// Parses `ci/jobs.toml`, with every problem found.
    pub fn parse(text: &str) -> Result<Config, Vec<String>> {
        let raw: RawConfig = toml::from_str(text).map_err(|e| vec![format!("{CONFIG}: {e}")])?;
        let mut problems = Vec::new();
        let workspace = globs(&raw.paths.workspace, "paths.workspace", &mut problems);
        let no_crate = globs(&raw.paths.no_crate, "paths.no_crate", &mut problems);
        let mut inputs = Vec::new();
        for (pattern, crates) in &raw.inputs {
            let glob = globs(std::slice::from_ref(pattern), "inputs", &mut problems);
            if crates.is_empty() {
                problems.push(format!("{CONFIG}: inputs: `{pattern}` names no crate"));
            }
            if let Some(glob) = glob.into_iter().next() {
                inputs.push((glob, crates.iter().cloned().collect()));
            }
        }
        let mut jobs = Vec::new();
        for (name, job) in &raw.job {
            let problem = |why: String| format!("{CONFIG}: job `{name}`: {why}");
            if !is_job_name(name) {
                problems.push(problem(
                    "a job name is lowercase words joined by `-`".into(),
                ));
            }
            let runner = match job.runner.as_str() {
                "hosted" => Runner::Hosted,
                "lab" => Runner::Lab,
                other => {
                    problems.push(problem(format!(
                        "runner `{other}` is not `hosted` or `lab`"
                    )));
                    continue;
                }
            };
            if job.crates.is_some() && job.when != "closure" {
                problems.push(problem("`crates` goes with `when = \"closure\"`".into()));
            }
            if (!job.paths.is_empty() || !job.lock_deps.is_empty()) && job.when != "direct" {
                problems.push(problem(
                    "`paths` and `lock_deps` go with `when = \"direct\"`".into(),
                ));
            }
            let when = match job.when.as_str() {
                "always" => When::Always,
                "closure" => When::Closure {
                    crates: job.crates.as_ref().map(|c| c.iter().cloned().collect()),
                },
                "direct" => {
                    if job.paths.is_empty() && job.lock_deps.is_empty() {
                        problems.push(problem("a direct rule needs `paths` or `lock_deps`".into()));
                    }
                    When::Direct {
                        paths: globs(&job.paths, &format!("job `{name}`"), &mut problems),
                        lock_deps: job.lock_deps.iter().cloned().collect(),
                    }
                }
                other => {
                    problems.push(problem(format!(
                        "`when = \"{other}\"` is not `always`, `closure` or `direct`"
                    )));
                    continue;
                }
            };
            if runner == Runner::Lab {
                if !name.starts_with("hw-") {
                    problems.push(problem(
                        "a lab job is named `hw-<rig>`, the status context after `hw/`".into(),
                    ));
                }
                if !matches!(when, When::Direct { .. }) {
                    problems.push(problem(
                        "a lab job uses a direct rule (RFC-0062 §2.2), not the closure".into(),
                    ));
                }
            }
            if let Some(task) = &job.planned
                && !is_task_ref(task)
            {
                problems.push(problem(format!(
                    "`planned = \"{task}\"` is not a task ID such as `QUAL-E2-T02`"
                )));
            }
            jobs.push(Job {
                name: name.clone(),
                runner,
                when,
                planned: job.planned.clone(),
            });
        }
        if problems.is_empty() {
            Ok(Config {
                workspace,
                no_crate,
                inputs,
                status_creators: raw.hardware.status_creators.into_iter().collect(),
                jobs,
            })
        } else {
            Err(problems)
        }
    }

    /// Reads and parses `ci/jobs.toml` under `root`.
    pub fn load(root: &Path) -> Result<Config, String> {
        let text =
            std::fs::read_to_string(root.join(CONFIG)).map_err(|e| format!("{CONFIG}: {e}"))?;
        Config::parse(&text).map_err(|problems| problems.join("\n"))
    }

    pub fn job(&self, name: &str) -> Option<&Job> {
        self.jobs.iter().find(|job| job.name == name)
    }

    /// Every crate the rules name that `workspace` does not have.
    pub fn unknown_crates(&self, workspace: &Workspace) -> Vec<String> {
        let known: BTreeSet<&str> = workspace.members.values().map(String::as_str).collect();
        let mut problems = Vec::new();
        for (glob, crates) in &self.inputs {
            for name in crates.iter().filter(|c| !known.contains(c.as_str())) {
                problems.push(format!(
                    "{CONFIG}: inputs: `{}` names `{name}`, which is not a workspace crate",
                    glob.as_str()
                ));
            }
        }
        for job in &self.jobs {
            if let When::Closure {
                crates: Some(crates),
            } = &job.when
            {
                for name in crates.iter().filter(|c| !known.contains(c.as_str())) {
                    problems.push(format!(
                        "{CONFIG}: job `{}` names `{name}`, which is not a workspace crate",
                        job.name
                    ));
                }
            }
        }
        problems
    }
}

/// Parses each pattern of `list`, adding a problem for each bad one.
fn globs(list: &[String], what: &str, problems: &mut Vec<String>) -> Vec<Glob> {
    list.iter()
        .filter_map(|text| match Glob::parse(text) {
            Ok(glob) => Some(glob),
            Err(error) => {
                problems.push(format!("{CONFIG}: {what}: {error}"));
                None
            }
        })
        .collect()
}

fn is_job_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--")
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// A task ID, or several of one epic written `QUAL-E3-T09/T10`.
fn is_task_ref(text: &str) -> bool {
    let mut parts = text.split('/');
    let Some(first) = parts.next() else {
        return false;
    };
    let digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    let fields: Vec<&str> = first.split('-').collect();
    let [area, epic, task] = fields.as_slice() else {
        return false;
    };
    let task_number = |t: &str| t.strip_prefix('T').is_some_and(digits);
    ["PLC", "TIER", "DIST", "COMP", "PROD", "QUAL"].contains(area)
        && epic.strip_prefix('E').is_some_and(digits)
        && task_number(task)
        && parts.all(task_number)
}

// ---------------------------------------------------------------------------------------------
// The crate graph.

/// Files at one revision.
pub trait Tree {
    /// The file at `path`, `/`-separated from the root, or `None` when there is none.
    fn read(&self, path: &str) -> Result<Option<String>, String>;
}

/// The files of a checkout.
pub struct Files<'a>(pub &'a Path);

impl Tree for Files<'_> {
    fn read(&self, path: &str) -> Result<Option<String>, String> {
        match std::fs::read_to_string(self.0.join(path)) {
            Ok(text) => Ok(Some(text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("{path}: {e}")),
        }
    }
}

/// The files of a commit, read through git from the repository at `root`, never checked out.
pub struct Git<'a> {
    pub root: &'a Path,
    pub rev: &'a str,
}

impl Tree for Git<'_> {
    fn read(&self, path: &str) -> Result<Option<String>, String> {
        let spec = format!("{}:{path}", self.rev);
        let exists = Command::new("git")
            .current_dir(self.root)
            .args(["cat-file", "-e", &spec])
            .output()
            .map_err(|e| format!("could not start git: {e}"))?;
        if !exists.status.success() {
            return Ok(None);
        }
        let shown = git(self.root, &["cat-file", "blob", &spec])?;
        String::from_utf8(shown)
            .map(Some)
            .map_err(|_| format!("{spec}: not UTF-8"))
    }
}

/// Runs git in `root` and returns its standard output, or why it failed.
fn git(root: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .map_err(|e| format!("could not start git: {e}"))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(format!(
            "`git {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

/// The workspace members and the edges between them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Workspace {
    /// Each member's directory, `/`-separated from the root, with its crate name.
    pub members: BTreeMap<String, String>,
    /// Each crate, with the crates that depend on it, by any kind of dependency.
    pub dependents: BTreeMap<String, BTreeSet<String>>,
}

impl Workspace {
    /// Reads the members' manifests in `tree`. A member's dependency on another member is an
    /// edge however it is declared: by path, through `workspace = true`, or by package name.
    pub fn read(tree: &dyn Tree) -> Result<Workspace, String> {
        let root_text = tree
            .read("Cargo.toml")?
            .ok_or_else(|| "the root `Cargo.toml` is missing".to_owned())?;
        let root: toml::Table =
            toml::from_str(&root_text).map_err(|e| format!("Cargo.toml: {e}"))?;
        let workspace = root
            .get("workspace")
            .and_then(toml::Value::as_table)
            .ok_or_else(|| "Cargo.toml has no `[workspace]` table".to_owned())?;
        let mut dirs = Vec::new();
        for member in workspace
            .get("members")
            .and_then(toml::Value::as_array)
            .ok_or_else(|| "Cargo.toml: `workspace.members` is not an array".to_owned())?
        {
            let dir = member
                .as_str()
                .ok_or_else(|| "Cargo.toml: a workspace member is not a string".to_owned())?;
            if dir.contains(['*', '?', '[']) {
                return Err(format!(
                    "Cargo.toml: member `{dir}` is a glob; list each member by its directory"
                ));
            }
            dirs.push(normalise(dir).ok_or_else(|| format!("Cargo.toml: member `{dir}`"))?);
        }
        // `workspace.dependencies` entries, which members inherit with `workspace = true`.
        let shared = workspace
            .get("dependencies")
            .and_then(toml::Value::as_table)
            .cloned()
            .unwrap_or_default();

        let mut members = BTreeMap::new();
        let mut manifests = Vec::new();
        for dir in &dirs {
            let path = format!("{dir}/Cargo.toml");
            let text = tree
                .read(&path)?
                .ok_or_else(|| format!("member `{dir}` has no {path}"))?;
            let manifest: toml::Table =
                toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
            let package = manifest
                .get("package")
                .and_then(|p| p.get("name"))
                .and_then(toml::Value::as_str)
                .ok_or_else(|| format!("{path}: `package.name` is missing"))?
                .to_owned();
            members.insert(dir.clone(), short_name(&package));
            manifests.push((dir.clone(), package, manifest));
        }
        let by_package: BTreeMap<&str, &str> = manifests
            .iter()
            .map(|(dir, package, _)| (package.as_str(), members[dir].as_str()))
            .collect();

        let mut dependents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (dir, _, manifest) in &manifests {
            let user = &members[dir];
            for (key, entry) in dependency_entries(manifest) {
                let mut targets = BTreeSet::new();
                let mut names = vec![key.as_str()];
                if let Some(package) = entry.get("package").and_then(toml::Value::as_str) {
                    names.push(package);
                }
                if let Some(path) = entry.get("path").and_then(toml::Value::as_str)
                    && let Some(target) = normalise(&format!("{dir}/{path}"))
                    && let Some(name) = members.get(&target)
                {
                    targets.insert(name.clone());
                }
                if entry.get("workspace").and_then(toml::Value::as_bool) == Some(true)
                    && let Some(inherited) = shared.get(&key)
                {
                    if let Some(package) = inherited.get("package").and_then(toml::Value::as_str) {
                        names.push(package);
                    }
                    if let Some(path) = inherited.get("path").and_then(toml::Value::as_str)
                        && let Some(target) = normalise(path)
                        && let Some(name) = members.get(&target)
                    {
                        targets.insert(name.clone());
                    }
                }
                for name in names {
                    if let Some(target) = by_package.get(name) {
                        targets.insert((*target).to_owned());
                    }
                }
                for target in targets.into_iter().filter(|t| t != user) {
                    dependents.entry(target).or_default().insert(user.clone());
                }
            }
        }
        Ok(Workspace {
            members,
            dependents,
        })
    }

    /// Both graphs' members and edges.
    pub fn union(mut self, other: Workspace) -> Workspace {
        self.members.extend(other.members);
        for (target, users) in other.dependents {
            self.dependents.entry(target).or_default().extend(users);
        }
        self
    }

    /// The crate whose directory holds `path`: the deepest one, so `crates/kernels-cuda/` is not
    /// taken for `crates/kernels/`.
    pub fn crate_of(&self, path: &str) -> Option<&str> {
        self.members
            .iter()
            .filter(|(dir, _)| {
                path.strip_prefix(dir.as_str())
                    .is_some_and(|r| r.starts_with('/'))
            })
            .max_by_key(|(dir, _)| dir.len())
            .map(|(_, name)| name.as_str())
    }

    /// `seeds` and every crate that depends on one of them, directly or not.
    pub fn closure(&self, seeds: &BTreeSet<String>) -> BTreeSet<String> {
        let mut closure = seeds.clone();
        let mut work: Vec<String> = seeds.iter().cloned().collect();
        while let Some(name) = work.pop() {
            for user in self.dependents.get(&name).into_iter().flatten() {
                if closure.insert(user.clone()) {
                    work.push(user.clone());
                }
            }
        }
        closure
    }

    /// Every crate.
    pub fn all(&self) -> BTreeSet<String> {
        self.members.values().cloned().collect()
    }
}

fn short_name(package: &str) -> String {
    package.strip_prefix(PREFIX).unwrap_or(package).to_owned()
}

/// Every entry of every dependency table of a manifest: `[dependencies]`, `[dev-dependencies]`
/// and `[build-dependencies]`, and the same under each `[target.<cfg>]`. A plain version string
/// yields an empty table.
fn dependency_entries(manifest: &toml::Table) -> Vec<(String, toml::Table)> {
    const KINDS: &[&str] = &[
        "dependencies",
        "dev-dependencies",
        "build-dependencies",
        "dev_dependencies",
        "build_dependencies",
    ];
    let mut tables: Vec<&toml::Table> = KINDS
        .iter()
        .filter_map(|kind| manifest.get(*kind).and_then(toml::Value::as_table))
        .collect();
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        for target in targets.values().filter_map(toml::Value::as_table) {
            tables.extend(
                KINDS
                    .iter()
                    .filter_map(|kind| target.get(*kind).and_then(toml::Value::as_table)),
            );
        }
    }
    tables
        .into_iter()
        .flat_map(|table| {
            table
                .iter()
                .map(|(key, value)| (key.clone(), value.as_table().cloned().unwrap_or_default()))
        })
        .collect()
}

/// `path` with `.` and `..` segments resolved, `/`-separated, or `None` when it leaves the root.
fn normalise(path: &str) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            part => parts.push(part),
        }
    }
    Some(parts.join("/"))
}

// ---------------------------------------------------------------------------------------------
// The affected-job set.

/// The class of one changed path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Class {
    /// Counts as a change to every crate; the pattern it matched.
    Workspace(String),
    Crate(String),
    /// Read by these crates; the pattern it matched.
    Input(String, BTreeSet<String>),
    NoCrate,
    /// Known to no rule, so counted as a change to every crate.
    Unclassified,
}

/// What `changes` decided, and why.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Affected {
    /// Each changed path with its class, in the diff's order.
    pub paths: Vec<(String, Class)>,
    /// The packages whose `Cargo.lock` entries changed.
    pub lock: BTreeSet<String>,
    /// The crates the paths change, before the closure.
    pub changed: BTreeSet<String>,
    pub closure: BTreeSet<String>,
    /// The hosted jobs to run, by name.
    pub jobs: Vec<String>,
    /// The hardware jobs whose statuses `ci-gate` waits for, by name.
    pub hw_required: Vec<String>,
    /// Planned jobs whose rule matched, with the task that adds each.
    pub planned: Vec<(String, String)>,
    /// Why each matched job matched.
    pub reasons: BTreeMap<String, String>,
}

/// Applies `config` to the changed `paths` and the `Cargo.lock` packages in `lock`.
pub fn affected(
    config: &Config,
    workspace: &Workspace,
    paths: &[String],
    lock: &BTreeSet<String>,
) -> Affected {
    let mut affected = Affected {
        lock: lock.clone(),
        ..Affected::default()
    };
    let mut everything = false;
    for path in paths {
        let class = classify(config, workspace, path);
        match &class {
            Class::Workspace(_) | Class::Unclassified => everything = true,
            Class::Crate(name) => {
                affected.changed.insert(name.clone());
            }
            Class::Input(_, crates) => affected.changed.extend(crates.iter().cloned()),
            Class::NoCrate => {}
        }
        affected.paths.push((path.clone(), class));
    }
    if everything {
        affected.changed = workspace.all();
    }
    affected.closure = workspace.closure(&affected.changed);

    for job in &config.jobs {
        let reason = match &job.when {
            When::Always => Some("runs on every change".to_owned()),
            When::Closure { crates: None } => (!affected.closure.is_empty())
                .then(|| format!("the closure holds {} crates", affected.closure.len())),
            When::Closure {
                crates: Some(crates),
            } => {
                let hit: Vec<&str> = crates
                    .iter()
                    .filter(|c| affected.closure.contains(*c))
                    .map(String::as_str)
                    .collect();
                (!hit.is_empty()).then(|| format!("the closure includes {}", hit.join(", ")))
            }
            When::Direct {
                paths: globs,
                lock_deps,
            } => paths
                .iter()
                .find_map(|path| {
                    globs
                        .iter()
                        .find(|glob| glob.matches(path))
                        .map(|glob| format!("{path} matches `{}`", glob.as_str()))
                })
                .or_else(|| {
                    lock.iter()
                        .find(|package| lock_deps.contains(*package))
                        .map(|package| format!("Cargo.lock changes `{package}`"))
                }),
        };
        let Some(reason) = reason else {
            continue;
        };
        affected.reasons.insert(job.name.clone(), reason);
        match (job.runner, &job.planned) {
            (Runner::Hosted, Some(task)) => affected.planned.push((job.name.clone(), task.clone())),
            (Runner::Hosted, None) => affected.jobs.push(job.name.clone()),
            (Runner::Lab, planned) => {
                affected.hw_required.push(job.name.clone());
                if let Some(task) = planned {
                    affected.planned.push((job.name.clone(), task.clone()));
                }
            }
        }
    }
    affected
}

/// The class of `path`, by the order of the module documentation.
pub fn classify(config: &Config, workspace: &Workspace, path: &str) -> Class {
    if let Some(glob) = config.workspace.iter().find(|g| g.matches(path)) {
        return Class::Workspace(glob.as_str().to_owned());
    }
    if let Some(name) = workspace.crate_of(path) {
        return Class::Crate(name.to_owned());
    }
    let mut readers = BTreeSet::new();
    let mut first = None;
    for (glob, crates) in config.inputs.iter().filter(|(g, _)| g.matches(path)) {
        first.get_or_insert_with(|| glob.as_str().to_owned());
        readers.extend(crates.iter().cloned());
    }
    if let Some(pattern) = first {
        return Class::Input(pattern, readers);
    }
    if config.no_crate.iter().any(|g| g.matches(path)) {
        return Class::NoCrate;
    }
    Class::Unclassified
}

/// The packages whose `Cargo.lock` entries differ between two lock files: added, removed, or
/// with another version, source or checksum. A missing file has no packages.
pub fn lock_changes(base: Option<&str>, head: Option<&str>) -> Result<BTreeSet<String>, String> {
    let entries = |text: Option<&str>, side: &str| -> Result<BTreeSet<Vec<String>>, String> {
        let Some(text) = text else {
            return Ok(BTreeSet::new());
        };
        let lock: toml::Table =
            toml::from_str(text).map_err(|e| format!("Cargo.lock at the {side}: {e}"))?;
        let mut entries = BTreeSet::new();
        for package in lock
            .get("package")
            .and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
        {
            let field = |name: &str| {
                package
                    .get(name)
                    .and_then(toml::Value::as_str)
                    .unwrap_or_default()
                    .to_owned()
            };
            entries.insert(vec![
                field("name"),
                field("version"),
                field("source"),
                field("checksum"),
            ]);
        }
        Ok(entries)
    };
    let base = entries(base, "base")?;
    let head = entries(head, "head")?;
    Ok(base
        .symmetric_difference(&head)
        .map(|entry| entry[0].clone())
        .collect())
}

/// The paths that differ between the trees of two commits, without rename detection, so a moved
/// file counts at both its old and its new path.
pub fn changed_paths(root: &Path, base: &str, head: &str) -> Result<Vec<String>, String> {
    let out = git(
        root,
        &[
            "diff",
            "--no-renames",
            "--no-ext-diff",
            "--name-only",
            "-z",
            base,
            head,
            "--",
        ],
    )?;
    out.split(|b| *b == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            String::from_utf8(path.to_vec()).map_err(|_| "a changed path is not UTF-8".to_owned())
        })
        .collect()
}

/// A JSON array of names, as the job outputs carry them.
pub fn json_list(names: &[String]) -> String {
    serde_json::to_string(names).expect("a list of strings serialises")
}

/// The report `changes` prints: each class with its paths, the closure and each job's verdict.
pub fn report(affected: &Affected, config: &Config) -> Vec<String> {
    let mut lines = Vec::new();
    let mut by_class: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for (path, class) in &affected.paths {
        let key = match class {
            Class::Workspace(pattern) => format!("every crate (matches `{pattern}`)"),
            Class::Crate(name) => format!("crate `{name}`"),
            Class::Input(pattern, crates) => format!(
                "read by {} (matches `{pattern}`)",
                crates.iter().cloned().collect::<Vec<_>>().join(", ")
            ),
            Class::NoCrate => "no crate".to_owned(),
            Class::Unclassified => "every crate: no rule knows the path".to_owned(),
        };
        by_class.entry(key).or_default().push(path);
    }
    for (class, paths) in &by_class {
        // Unclassified paths are listed in full: each is a rule to add.
        if class.starts_with("every crate: no rule") || paths.len() <= 5 {
            lines.push(format!("{class}: {}", paths.join(", ")));
        } else {
            lines.push(format!(
                "{class}: {}, and {} more",
                paths[..3].join(", "),
                paths.len() - 3
            ));
        }
    }
    if !affected.lock.is_empty() {
        lines.push(format!(
            "Cargo.lock changes: {}",
            affected.lock.iter().cloned().collect::<Vec<_>>().join(", ")
        ));
    }
    lines.push(format!(
        "closure: {}",
        if affected.closure.is_empty() {
            "empty".to_owned()
        } else {
            affected
                .closure
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        }
    ));
    for job in &config.jobs {
        let verdict = match (affected.reasons.get(&job.name), &job.planned) {
            (Some(reason), Some(task)) if job.runner == Runner::Hosted => {
                format!("would run ({reason}), but it is planned: {task}")
            }
            (Some(reason), Some(task)) => {
                format!("required ({reason}); planned: {task}")
            }
            (Some(reason), None) if job.runner == Runner::Lab => format!("required ({reason})"),
            (Some(reason), None) => format!("runs ({reason})"),
            (None, _) => "not affected".to_owned(),
        };
        lines.push(format!("{}: {verdict}", job.name));
    }
    lines
}

/// `cargo xtask ci changes --base <rev> --head <rev> [--github-output <file>]`.
pub fn step(context: &Context) -> Result<(), String> {
    let base = context
        .args
        .get("--base")
        .ok_or_else(|| "`--base` is required".to_owned())?;
    let head = context
        .args
        .get("--head")
        .ok_or_else(|| "`--head` is required".to_owned())?;
    let root = &context.root;
    for rev in [base, head] {
        git(
            root,
            &[
                "rev-parse",
                "--verify",
                "--quiet",
                &format!("{rev}^{{commit}}"),
            ],
        )
        .map_err(|_| format!("`{rev}` is not a commit in this repository"))?;
    }
    let config = Config::load(root)?;
    let base_tree = Git { root, rev: base };
    let head_tree = Git { root, rev: head };
    let base_graph = Workspace::read(&base_tree).map_err(|e| format!("the base: {e}"))?;
    let unknown = config.unknown_crates(&base_graph);
    if !unknown.is_empty() {
        return Err(unknown.join("\n"));
    }
    let graph =
        base_graph.union(Workspace::read(&head_tree).map_err(|e| format!("the head: {e}"))?);
    let paths = changed_paths(root, base, head)?;
    let lock = if paths.iter().any(|p| p == "Cargo.lock") {
        lock_changes(
            base_tree.read("Cargo.lock")?.as_deref(),
            head_tree.read("Cargo.lock")?.as_deref(),
        )?
    } else {
        BTreeSet::new()
    };
    let affected = affected(&config, &graph, &paths, &lock);
    eprintln!(
        "changes: {} paths differ between {base} and {head}",
        paths.len()
    );
    for line in report(&affected, &config) {
        eprintln!("changes: {line}");
    }
    let outputs = format!(
        "jobs={}\nhw_required={}\n",
        json_list(&affected.jobs),
        json_list(&affected.hw_required)
    );
    print!("{outputs}");
    if let Some(file) = context.args.get("--github-output") {
        std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(file)
            .and_then(|mut f| f.write_all(outputs.as_bytes()))
            .map_err(|e| format!("--github-output {file}: {e}"))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The golden suite.

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GoldenFile {
    /// `fnv1a:<16 hex digits>` of `ci/jobs.toml` with LF line ends.
    jobs_toml: String,
    golden: Vec<Golden>,
}

/// One recorded diff and the sets it must give.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Golden {
    pub name: String,
    /// Where the diff comes from: a PR and commit, or `constructed` with the reason.
    pub source: String,
    pub paths: Vec<String>,
    /// Packages whose `Cargo.lock` entries change.
    #[serde(default)]
    pub lock: Vec<String>,
    pub jobs: Vec<String>,
    pub hw_required: Vec<String>,
    /// Planned jobs whose rule matches.
    #[serde(default)]
    pub planned: Vec<String>,
}

/// The digest the golden suite records for `ci/jobs.toml`.
pub fn digest(config_text: &str) -> String {
    format!(
        "fnv1a:{:016x}",
        crate::dep_diff::fnv1a(&config_text.replace("\r\n", "\n"))
    )
}

/// The jobs of a workflow file: the keys two spaces in under its top-level `jobs:`.
pub fn workflow_jobs(text: &str) -> Vec<String> {
    let mut jobs = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        let line = line.trim_end();
        if !line.starts_with(' ') && !line.is_empty() && !line.starts_with('#') {
            inside = line == "jobs:";
            continue;
        }
        if inside
            && let Some(key) = line.strip_prefix("  ")
            && !key.starts_with(' ')
            && !key.starts_with('#')
            && let Some(name) = key.strip_suffix(':')
        {
            jobs.push(name.to_owned());
        }
    }
    jobs
}

/// Every problem with the rules in `config_text` against the goldens in `goldens_text`, the
/// crate graph `workspace` and the jobs of `pr.yml` in `workflow_text`.
pub fn check(
    config_text: &str,
    goldens_text: &str,
    workspace: &Workspace,
    workflow_text: &str,
) -> Vec<String> {
    let config = match Config::parse(config_text) {
        Ok(config) => config,
        Err(problems) => return problems,
    };
    let mut problems = config.unknown_crates(workspace);
    let file: GoldenFile = match toml::from_str(goldens_text) {
        Ok(file) => file,
        Err(e) => {
            problems.push(format!("{GOLDENS}: {e}"));
            return problems;
        }
    };

    let expected = digest(config_text);
    if file.jobs_toml != expected {
        problems.push(format!(
            "{CONFIG} changed since {GOLDENS} was checked against it: re-check every golden, \
             add one for each new rule or pattern, then set `jobs_toml = \"{expected}\"`"
        ));
    }

    // Each golden gives exactly its sets.
    let mut names = BTreeSet::new();
    let mut seen_jobs = BTreeSet::new();
    let mut unclassified = false;
    for golden in &file.golden {
        if !names.insert(golden.name.as_str()) {
            problems.push(format!("{GOLDENS}: golden `{}` appears twice", golden.name));
        }
        if golden.source.trim().is_empty() {
            problems.push(format!("{GOLDENS}: golden `{}` has no source", golden.name));
        }
        let lock: BTreeSet<String> = golden.lock.iter().cloned().collect();
        let got = affected(&config, workspace, &golden.paths, &lock);
        let planned: Vec<String> = got.planned.iter().map(|(name, _)| name.clone()).collect();
        for (what, want, got) in [
            ("jobs", &golden.jobs, &got.jobs),
            ("hw_required", &golden.hw_required, &got.hw_required),
            ("planned", &golden.planned, &planned),
        ] {
            let want: BTreeSet<&String> = want.iter().collect();
            let got: BTreeSet<&String> = got.iter().collect();
            if want != got {
                problems.push(format!(
                    "{GOLDENS}: golden `{}`: {what} is {:?}, expected {:?}",
                    golden.name,
                    got.into_iter().collect::<Vec<_>>(),
                    want.into_iter().collect::<Vec<_>>(),
                ));
            }
        }
        seen_jobs.extend(golden.jobs.iter().cloned());
        seen_jobs.extend(golden.hw_required.iter().cloned());
        seen_jobs.extend(golden.planned.iter().cloned());
        unclassified |= got.paths.iter().any(|(_, c)| *c == Class::Unclassified);
    }

    // One diff per rule: every job and every pattern is met by some golden.
    let all_paths: Vec<&String> = file.golden.iter().flat_map(|g| &g.paths).collect();
    let mut uncovered = |glob: &Glob, what: &str| {
        if !all_paths.iter().any(|path| glob.matches(path)) {
            problems.push(format!(
                "{GOLDENS}: no golden has a path that `{}` ({what}) matches",
                glob.as_str()
            ));
        }
    };
    for glob in &config.workspace {
        uncovered(glob, "paths.workspace");
    }
    for glob in &config.no_crate {
        uncovered(glob, "paths.no_crate");
    }
    for (glob, _) in &config.inputs {
        uncovered(glob, "inputs");
    }
    // A direct job's patterns, and its lock dependencies, are held to more below: each needs a
    // golden that depends on it alone.
    for job in &config.jobs {
        if !seen_jobs.contains(&job.name) {
            problems.push(format!(
                "{GOLDENS}: no golden runs or requires `{}`",
                job.name
            ));
        }
    }
    if !unclassified {
        problems.push(format!(
            "{GOLDENS}: no golden has a path no rule knows, so failing closed is untested"
        ));
    }

    // Each part of a rule is pinned: some golden depends on it alone, so dropping it from
    // `ci/jobs.toml` changes that golden's sets. A golden met by two of a job's patterns would
    // still require the job with either one gone.
    for job in &config.jobs {
        if let When::Direct { paths, lock_deps } = &job.when {
            // The indices of the job's patterns the golden meets, and its lock dependencies.
            let met = |golden: &Golden| -> (Vec<usize>, Vec<String>) {
                let globs = (0..paths.len())
                    .filter(|&i| golden.paths.iter().any(|path| paths[i].matches(path)))
                    .collect();
                let locks = golden
                    .lock
                    .iter()
                    .filter(|p| lock_deps.contains(*p))
                    .cloned()
                    .collect();
                (globs, locks)
            };
            for (i, glob) in paths.iter().enumerate() {
                let pinned = file.golden.iter().any(|golden| {
                    let (globs, locks) = met(golden);
                    globs == [i] && locks.is_empty()
                });
                if !pinned {
                    problems.push(format!(
                        "{GOLDENS}: no golden meets `{}` alone among `{}`'s patterns, so \
                         dropping it would go unseen",
                        glob.as_str(),
                        job.name
                    ));
                }
            }
            for package in lock_deps {
                let pinned = file.golden.iter().any(|golden| {
                    let (globs, locks) = met(golden);
                    globs.is_empty() && locks == [package.clone()]
                });
                if !pinned {
                    problems.push(format!(
                        "{GOLDENS}: no golden changes `{package}` alone among `{}`'s lock \
                         dependencies and paths, so dropping it would go unseen",
                        job.name
                    ));
                }
            }
        }
    }

    // The hosted rules and `pr.yml`'s jobs name the same jobs.
    let hosted: BTreeSet<&str> = config
        .jobs
        .iter()
        .filter(|job| job.runner == Runner::Hosted)
        .map(|job| job.name.as_str())
        .collect();
    let workflow: BTreeSet<String> = workflow_jobs(workflow_text)
        .into_iter()
        .filter(|job| !UNLISTED.contains(&job.as_str()))
        .collect();
    for job in hosted.iter().filter(|job| !workflow.contains(**job)) {
        problems.push(format!(
            "{CONFIG}: hosted job `{job}` is not a job of {WORKFLOW}"
        ));
    }
    for job in workflow.iter().filter(|job| !hosted.contains(job.as_str())) {
        problems.push(format!(
            "{WORKFLOW}: job `{job}` has no rule in {CONFIG}, so `changes` never lists it"
        ));
    }
    problems
}

/// The `jobs` step of `lint`: [`check`] on the checkout.
pub fn goldens_step(context: &Context) -> Result<(), String> {
    let read = |path: &str| {
        std::fs::read_to_string(context.root.join(path)).map_err(|e| format!("{path}: {e}"))
    };
    let workspace = Workspace::read(&Files(&context.root))?;
    let problems = check(
        &read(CONFIG)?,
        &read(GOLDENS)?,
        &workspace,
        &read(WORKFLOW)?,
    );
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

#[cfg(test)]
mod tests;
