// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask workflows`: the workflow security lint of RFC-0062 §12.4 (QUAL-E1-T09), over
//! every workflow in `.github/workflows/` and every composite action under `.github/actions/`.
//!
//! | Check | What fails |
//! |---|---|
//! | Read-only top level | a workflow whose top-level `permissions:` is missing, `write-all`, or grants any scope `write`; a job may widen its own |
//! | Pinned actions | a `uses:` that is not a local path (`./…`), an action or reusable workflow at a 40-hex commit SHA, or a `docker://` image at a `sha256:` digest (§11.1) |
//! | No `pull_request_target` | a workflow triggered by it, in any form of `on:` |
//! | No lab label | a job outside `lab.yml` whose `runs-on` names a lab label (§3.2) |
//! | No PR checkout beside a secret | a job that checks out a PR ref and holds a secret, an environment or a write token |
//!
//! **Lab labels.** A label is a lab label, compared without case, when it is `self-hosted` or
//! `syntropy-lab`; starts with `lab-` (the per-run `lab-run-<uuid>`), or `r-` (a rig ID such as
//! `R-ADA`, RFC-0061 §6.2); or is one of §3.2's rig labels: a backend and architecture label
//! (`cuda-…`, `metal-…`, `hip-…`, `vulkan-…`) or a feature (`rdma-…`, `nvme-…`, `roce`,
//! `unified`). A runner `group` counts as a label. A `runs-on` computed by an expression is read
//! through the job's matrix when it is `${{ matrix.<key> }}`, and otherwise fails, because the
//! labels it yields cannot be checked.
//!
//! **PR checkouts.** A step checks out a PR ref when it is `actions/checkout` and either its `ref`
//! names one (`github.event.pull_request.head…`, `…merge_commit_sha`, `github.head_ref`,
//! `refs/pull/`, `github.event.workflow_run.head…`), or the workflow runs on `pull_request`,
//! `pull_request_target` or `merge_group` and the `ref` is unset, `github.sha` or `github.ref`,
//! which there name the PR's merge commit. A `run` script that fetches `refs/pull/` or runs
//! `gh pr checkout` checks one out too. A job holds a secret when it, or the workflow's `env`,
//! names `secrets.` other than `secrets.GITHUB_TOKEN`, when it names an `environment`, or when it
//! passes `secrets: inherit`. It holds a write token when its own `permissions`, or the
//! workflow's where it sets none, grant any scope `write`.
//!
//! A file that does not parse, or whose shape the checks cannot read, fails: the lint never
//! passes a workflow it could not check.

use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use yaml_rust2::{Yaml, YamlLoader};

/// The one workflow that may name a lab label (RFC-0062 §3.2).
pub const LAB_WORKFLOW: &str = "lab.yml";

/// Lab labels matched exactly, without case.
const LAB_LABELS: &[&str] = &["self-hosted", "syntropy-lab", "roce", "unified"];

/// Prefixes of lab labels, matched without case.
const LAB_PREFIXES: &[&str] = &[
    "lab-", "r-", "cuda-", "metal-", "hip-", "vulkan-", "rdma-", "nvme-",
];

/// Triggers under which a checkout with no `ref`, or `github.sha` or `github.ref`, gets the PR's
/// code.
const PR_TRIGGERS: &[&str] = &["pull_request", "pull_request_target", "merge_group"];

/// Expression fragments that name a PR's code in a checkout's `ref`.
const PR_REFS: &[&str] = &[
    "github.event.pull_request.head",
    "github.event.pull_request.merge_commit_sha",
    "github.head_ref",
    "refs/pull/",
    "github.event.workflow_run.head",
];

/// One problem: the file, where in it, and what is wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    /// The file, relative to the root.
    pub file: String,
    /// The place in the file, such as `jobs.build.runs-on`.
    pub at: String,
    /// What is wrong, and the rule.
    pub message: String,
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.at.is_empty() {
            write!(f, "{}: {}", self.file, self.message)
        } else {
            write!(f, "{}: {}: {}", self.file, self.at, self.message)
        }
    }
}

/// Collects one file's findings.
struct Report<'a> {
    file: &'a str,
    findings: Vec<Finding>,
}

impl Report<'_> {
    fn add(&mut self, at: impl Into<String>, message: impl Into<String>) {
        self.findings.push(Finding {
            file: self.file.to_owned(),
            at: at.into(),
            message: message.into(),
        });
    }
}

/// The value at `key` of a mapping, if `node` is one and has it.
fn get<'y>(node: &'y Yaml, key: &str) -> Option<&'y Yaml> {
    match node {
        Yaml::Hash(map) => map.get(&Yaml::String(key.to_owned())),
        _ => None,
    }
}

/// A scalar as text: strings, and numbers and booleans as written.
fn text(node: &Yaml) -> Option<String> {
    match node {
        Yaml::String(s) | Yaml::Real(s) => Some(s.clone()),
        Yaml::Integer(n) => Some(n.to_string()),
        Yaml::Boolean(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Every scalar under `node`, as text.
fn scalars(node: &Yaml, out: &mut Vec<String>) {
    match node {
        Yaml::Array(items) => items.iter().for_each(|item| scalars(item, out)),
        Yaml::Hash(map) => map.iter().for_each(|(key, value)| {
            scalars(key, out);
            scalars(value, out);
        }),
        other => out.extend(text(other)),
    }
}

/// The one document of a YAML file, or why there is not one.
fn parse(source: &str) -> Result<Yaml, String> {
    let mut docs = YamlLoader::load_from_str(source).map_err(|error| error.to_string())?;
    match docs.len() {
        1 => Ok(docs.remove(0)),
        0 => Err("the file is empty".to_owned()),
        n => Err(format!("the file holds {n} YAML documents, not one")),
    }
}

/// Whether a `permissions:` value grants any scope `write`, or what is wrong with its shape.
fn writes(permissions: &Yaml) -> Result<bool, String> {
    match permissions {
        Yaml::String(s) if s == "read-all" => Ok(false),
        Yaml::String(s) if s == "write-all" => Ok(true),
        Yaml::Hash(scopes) => {
            let mut write = false;
            for (scope, level) in scopes {
                let scope = text(scope).unwrap_or_default();
                match text(level).as_deref() {
                    Some("read" | "none") => {}
                    Some("write") => write = true,
                    _ => {
                        return Err(format!(
                            "`{scope}` must be `read`, `write` or `none`, and is {level:?}"
                        ));
                    }
                }
            }
            Ok(write)
        }
        other => Err(format!(
            "must be `read-all`, `write-all` or a map of scopes, and is {other:?}"
        )),
    }
}

/// Why a `uses:` value is not pinned, if it is not.
fn unpinned(uses: &str) -> Option<String> {
    if uses.contains("${{") {
        return Some("is computed by an expression, so its pin cannot be checked".to_owned());
    }
    if uses.starts_with("./") {
        return None;
    }
    if let Some(image) = uses.strip_prefix("docker://") {
        let pinned = image.rsplit_once("@sha256:").is_some_and(|(_, digest)| {
            digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit())
        });
        return (!pinned).then(|| "must name its image by `@sha256:` digest".to_owned());
    }
    match uses.rsplit_once('@') {
        Some((_, rev)) if rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()) => None,
        Some((_, rev)) => Some(format!(
            "is pinned to `{rev}`, not to a full 40-hex commit SHA (RFC-0062 §11.1)"
        )),
        None => Some("names no version; pin it to a full 40-hex commit SHA".to_owned()),
    }
}

/// Checks every `uses:` of `steps`.
fn check_steps_pinned(report: &mut Report<'_>, at: &str, steps: Option<&Yaml>) {
    let Some(steps) = steps else { return };
    let Yaml::Array(steps) = steps else {
        report.add(at, "`steps` is not a list");
        return;
    };
    for (n, step) in steps.iter().enumerate() {
        if let Some(uses) = get(step, "uses") {
            match text(uses) {
                Some(uses) => {
                    if let Some(problem) = unpinned(&uses) {
                        report.add(format!("{at}[{n}].uses"), format!("`{uses}` {problem}"));
                    }
                }
                None => report.add(format!("{at}[{n}].uses"), "is not a string"),
            }
        }
    }
}

/// The workflow's triggers, as names.
fn triggers(report: &mut Report<'_>, workflow: &Yaml) -> Vec<String> {
    // YAML 1.1 reads a bare `on` as `true`; the 1.2 loader keeps it a string, as GitHub does.
    let Some(on) = get(workflow, "on") else {
        report.add("on", "the workflow has no `on:` triggers");
        return Vec::new();
    };
    match on {
        Yaml::String(event) => vec![event.clone()],
        Yaml::Array(events) => events.iter().filter_map(text).collect(),
        Yaml::Hash(events) => events.keys().filter_map(text).collect(),
        other => {
            report.add("on", format!("cannot be read: {other:?}"));
            Vec::new()
        }
    }
}

fn is_lab_label(label: &str) -> bool {
    let label = label.trim().to_ascii_lowercase();
    LAB_LABELS.contains(&label.as_str()) || LAB_PREFIXES.iter().any(|p| label.starts_with(p))
}

/// The matrix key a `${{ matrix.<key> }}` expression names, if it is exactly that.
fn matrix_key(label: &str) -> Option<&str> {
    let inner = label.trim().strip_prefix("${{")?.strip_suffix("}}")?.trim();
    let key = inner.strip_prefix("matrix.")?;
    key.bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        .then_some(key)
}

/// Every value the matrix gives `key`, or why they cannot be read.
fn matrix_values(job: &Yaml, key: &str) -> Result<Vec<String>, String> {
    let matrix = get(job, "strategy")
        .and_then(|s| get(s, "matrix"))
        .ok_or("the job has no `strategy.matrix`")?;
    if !matches!(matrix, Yaml::Hash(_)) {
        return Err("the matrix is computed, so its labels cannot be read".to_owned());
    }
    let mut values = Vec::new();
    if let Some(axis) = get(matrix, key) {
        scalars(axis, &mut values);
    }
    if let Some(include) = get(matrix, "include") {
        let Yaml::Array(entries) = include else {
            return Err("`include` is computed, so its labels cannot be read".to_owned());
        };
        for entry in entries {
            if let Some(value) = get(entry, key) {
                scalars(value, &mut values);
            }
        }
    }
    if values.is_empty() {
        return Err(format!("the matrix gives `{key}` no value"));
    }
    if values.iter().any(|v| v.contains("${{")) {
        return Err(format!("the matrix computes `{key}`"));
    }
    Ok(values)
}

/// The labels `runs-on` names, read through the matrix where needed.
fn runs_on_labels(report: &mut Report<'_>, at: &str, job: &Yaml, runs_on: &Yaml) -> Vec<String> {
    let mut named = Vec::new();
    match runs_on {
        Yaml::Hash(_) => {
            for key in ["labels", "group"] {
                if let Some(value) = get(runs_on, key) {
                    scalars(value, &mut named);
                }
            }
        }
        other => scalars(other, &mut named),
    }
    let mut labels = Vec::new();
    for label in named {
        if !label.contains("${{") {
            labels.push(label);
        } else if let Some(key) = matrix_key(&label) {
            match matrix_values(job, key) {
                Ok(values) => labels.extend(values),
                Err(why) => report.add(at, format!("`{label}`: {why}")),
            }
        } else {
            report.add(
                at,
                format!("`{label}` is computed, so the labels it yields cannot be checked"),
            );
        }
    }
    labels
}

/// Whether a step checks out a PR ref.
fn checks_out_pr(step: &Yaml, pr_event: bool) -> bool {
    if let Some(run) = get(step, "run").and_then(text)
        && (run.contains("refs/pull/") || run.contains("gh pr checkout"))
    {
        return true;
    }
    let Some(uses) = get(step, "uses").and_then(text) else {
        return false;
    };
    if !uses.starts_with("actions/checkout@") {
        return false;
    }
    let reference = get(step, "with")
        .and_then(|with| get(with, "ref"))
        .and_then(text);
    match reference {
        Some(r) if PR_REFS.iter().any(|p| r.contains(p)) => true,
        Some(r) => {
            let r: String = r.chars().filter(|c| !c.is_whitespace()).collect();
            pr_event && (r == "${{github.sha}}" || r == "${{github.ref}}")
        }
        None => pr_event,
    }
}

/// Why a job holds a secret, if it does.
fn holds_secret(job: &Yaml, workflow_env: Option<&Yaml>) -> Option<String> {
    if get(job, "environment").is_some() {
        return Some("it names an environment, whose secrets it receives".to_owned());
    }
    if get(job, "secrets").and_then(text).as_deref() == Some("inherit") {
        return Some("it passes `secrets: inherit`".to_owned());
    }
    let mut all = Vec::new();
    scalars(job, &mut all);
    if let Some(env) = workflow_env {
        scalars(env, &mut all);
    }
    all.iter()
        .flat_map(|s| s.match_indices("secrets.").map(move |(i, _)| &s[i..]))
        .find(|named| !named.starts_with("secrets.GITHUB_TOKEN"))
        .map(|named| {
            let name: String = named
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.')
                .collect();
            format!("it reads `{name}`")
        })
}

/// Checks one workflow file, named `name` (its file name).
pub fn check_workflow(file: &str, name: &str, source: &str) -> Vec<Finding> {
    let mut report = Report {
        file,
        findings: Vec::new(),
    };
    let workflow = match parse(source) {
        Ok(workflow) => workflow,
        Err(error) => {
            report.add("", format!("does not parse: {error}"));
            return report.findings;
        }
    };
    if !matches!(workflow, Yaml::Hash(_)) {
        report.add("", "is not a map");
        return report.findings;
    }

    // Read-only at the top (RFC-0062 §12.4).
    let top_writes = match get(&workflow, "permissions") {
        None => {
            report.add(
                "permissions",
                "the workflow sets no top-level `permissions:`; set it read-only, such as \
                 `contents: read`, and widen per job",
            );
            true
        }
        Some(permissions) => match writes(permissions) {
            Ok(false) => false,
            Ok(true) => {
                report.add(
                    "permissions",
                    "the top-level `permissions:` grants write; it must be read-only, with \
                     writes granted per job (RFC-0062 §12.4)",
                );
                true
            }
            Err(why) => {
                report.add("permissions", why);
                true
            }
        },
    };

    let on = triggers(&mut report, &workflow);
    if on.iter().any(|event| event == "pull_request_target") {
        report.add(
            "on",
            "the workflow uses `pull_request_target`, which RFC-0062 §12.4 forbids",
        );
    }
    let pr_event = on.iter().any(|event| PR_TRIGGERS.contains(&event.as_str()));
    let workflow_env = get(&workflow, "env");
    let lab = name == LAB_WORKFLOW;

    let Some(jobs) = get(&workflow, "jobs") else {
        report.add("jobs", "the workflow has no jobs");
        return report.findings;
    };
    let Yaml::Hash(jobs) = jobs else {
        report.add("jobs", "is not a map");
        return report.findings;
    };
    for (id, job) in jobs {
        let id = text(id).unwrap_or_default();
        let at = format!("jobs.{id}");
        if !matches!(job, Yaml::Hash(_)) {
            report.add(&at, "is not a map");
            continue;
        }

        // Pinned actions, and pinned reusable workflows.
        if let Some(uses) = get(job, "uses") {
            match text(uses) {
                Some(uses) => {
                    if let Some(problem) = unpinned(&uses) {
                        report.add(format!("{at}.uses"), format!("`{uses}` {problem}"));
                    }
                }
                None => report.add(format!("{at}.uses"), "is not a string"),
            }
        }
        check_steps_pinned(&mut report, &format!("{at}.steps"), get(job, "steps"));

        // Lab labels only in `lab.yml`.
        if let Some(runs_on) = get(job, "runs-on") {
            let runs_at = format!("{at}.runs-on");
            let labels = runs_on_labels(&mut report, &runs_at, job, runs_on);
            if !lab {
                for label in labels.iter().filter(|l| is_lab_label(l)) {
                    report.add(
                        &runs_at,
                        format!(
                            "names the lab label `{label}`; only `{LAB_WORKFLOW}` may reach a \
                             rig (RFC-0062 §3.2)"
                        ),
                    );
                }
            }
        }

        // No PR checkout in a job that holds a secret or a write token.
        let pr_step = match get(job, "steps") {
            Some(Yaml::Array(steps)) => steps.iter().position(|s| checks_out_pr(s, pr_event)),
            _ => None,
        };
        if let Some(n) = pr_step {
            let job_writes = match get(job, "permissions") {
                Some(permissions) => writes(permissions).unwrap_or(true),
                None => top_writes,
            };
            let holds = holds_secret(job, workflow_env)
                .or_else(|| job_writes.then(|| "its token can write".to_owned()));
            if let Some(why) = holds {
                report.add(
                    format!("{at}.steps[{n}]"),
                    format!(
                        "checks out a PR ref in a job that holds a secret or a write token: \
                         {why} (RFC-0062 §12.4)"
                    ),
                );
            }
        }
    }
    report.findings
}

/// Checks one composite action: its steps' `uses:` are pinned.
pub fn check_action(file: &str, source: &str) -> Vec<Finding> {
    let mut report = Report {
        file,
        findings: Vec::new(),
    };
    match parse(source) {
        Ok(action) => {
            let runs = get(&action, "runs");
            check_steps_pinned(
                &mut report,
                "runs.steps",
                runs.and_then(|r| get(r, "steps")),
            );
        }
        Err(error) => report.add("", format!("does not parse: {error}")),
    }
    report.findings
}

/// The YAML files directly in `dir`, sorted.
fn yaml_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    match fs::read_dir(dir) {
        Ok(entries) => {
            for entry in entries {
                let path = entry?.path();
                let yaml = path.extension().is_some_and(|e| e == "yml" || e == "yaml");
                if yaml && path.is_file() {
                    files.push(path);
                }
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    files.sort();
    Ok(files)
}

/// The composite actions under `dir`: each `action.yml` or `action.yaml` below it, sorted.
fn action_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_owned()];
    while let Some(dir) = pending.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        for entry in entries {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else if path
                .file_name()
                .is_some_and(|n| n == "action.yml" || n == "action.yaml")
            {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

/// The workflow files under `root`, which `actionlint` checks too.
pub fn workflow_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    yaml_files(&root.join(".github/workflows"))
}

/// Runs the lint on `root` and reports to `log`. Returns whether it passed.
pub fn run(root: &Path, log: &mut dyn Write) -> io::Result<bool> {
    let shown = |path: &Path| {
        path.strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    };
    let workflows = workflow_files(root)?;
    let actions = action_files(&root.join(".github/actions"))?;
    let mut findings = Vec::new();
    for path in &workflows {
        let file = shown(path);
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        match fs::read_to_string(path) {
            Ok(source) => findings.extend(check_workflow(&file, &name, &source)),
            Err(error) => findings.push(Finding {
                file,
                at: String::new(),
                message: format!("cannot be read: {error}"),
            }),
        }
    }
    for path in &actions {
        let file = shown(path);
        match fs::read_to_string(path) {
            Ok(source) => findings.extend(check_action(&file, &source)),
            Err(error) => findings.push(Finding {
                file,
                at: String::new(),
                message: format!("cannot be read: {error}"),
            }),
        }
    }
    for finding in &findings {
        writeln!(log, "workflows: {finding}")?;
    }
    if findings.is_empty() {
        writeln!(
            log,
            "workflows: PASS: {} workflows and {} composite actions meet RFC-0062 §12.4",
            workflows.len(),
            actions.len()
        )?;
    } else {
        writeln!(
            log,
            "workflows: FAIL: {} problems in the workflows and actions under {} (RFC-0062 §12.4)",
            findings.len(),
            root.join(".github").display()
        )?;
    }
    Ok(findings.is_empty())
}

#[cfg(test)]
mod tests;
