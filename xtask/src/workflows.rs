// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask workflows`: the workflow security lint of RFC-0062 §12.4 (QUAL-E1-T09), over
//! every workflow in `.github/workflows/`, every action under `.github/actions/`, and every local
//! action (`uses: ./…`) a workflow or action reaches.
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
//! **PR checkouts.** The rule allows only what it knows is safe. A step that is `actions/checkout`
//! (in any case) checks out a PR ref unless its `ref` is one of these, compared without case or
//! whitespace:
//! - a literal branch, tag or SHA with no `pull/` in it;
//! - an expression naming the base: `github.event.pull_request.base.sha` or `.base.ref`,
//!   `github.base_ref`, `github.event.merge_group.base_sha`, or
//!   `github.event.repository.default_branch`;
//! - unset, empty, `github.sha` or `github.ref`, but only when the workflow is not triggered by
//!   `pull_request`, `pull_request_target`, `merge_group` or `workflow_call`. Under those, these
//!   name code the PR controls. A called workflow cannot see what calls it, so `workflow_call`
//!   counts.
//!
//! Any other expression counts as a PR checkout on every trigger. A `run` script that names
//! `refs/pull/`, runs `gh pr checkout`, or runs `git fetch`, `checkout`, `pull`, `switch` or
//! `worktree` with `pull/` in it counts too. So does a step that uses a local composite action
//! one of whose steps counts.
//!
//! **Secrets.** A job holds a secret when any expression in it, or in the workflow's `env`, uses
//! the `secrets` context, in any case and any form (`secrets.X`, `secrets['X']`,
//! `toJSON(secrets)`), other than `GITHUB_TOKEN`. That includes `if:` values, which GitHub reads
//! as expressions without braces. It also holds one when it names an `environment` or passes
//! `secrets: inherit`. It holds a write token when its own `permissions`, or the workflow's where
//! it sets none, grant any scope `write`.
//!
//! **Limits.** A file cannot show these, so the lint does not check them:
//! - whether a 40-hex `@` is a commit, rather than a tag or branch of that name;
//! - a fork of `actions/checkout` under another name;
//! - a ref that reaches a checkout through `env` or a step's outputs, unless it is an expression
//!   in the `ref` itself.
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

/// Triggers under which a checkout with no `ref`, or `github.sha` or `github.ref`, gets code a
/// pull request controls. A called workflow (`workflow_call`) cannot see its caller's trigger, so
/// it counts.
const PR_TRIGGERS: &[&str] = &[
    "pull_request",
    "pull_request_target",
    "merge_group",
    "workflow_call",
];

/// Checkout refs, as expressions without whitespace and in lower case, that name the base
/// branch or the repository's own code under any trigger.
const BASE_REFS: &[&str] = &[
    "${{github.event.pull_request.base.sha}}",
    "${{github.event.pull_request.base.ref}}",
    "${{github.base_ref}}",
    "${{github.event.merge_group.base_sha}}",
    "${{github.event.repository.default_branch}}",
];

/// How deep a chain of local composite actions is followed.
const LOCAL_DEPTH: usize = 8;

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

/// Whether `uses` names `actions/checkout`, in any case and with or without a trailing slash.
fn is_checkout(uses: &str) -> bool {
    let name = uses.split('@').next().unwrap_or_default().trim();
    name.trim_end_matches('/')
        .eq_ignore_ascii_case("actions/checkout")
}

/// Whether a step checks out code a pull request controls, by itself: see the module docs.
fn checks_out_pr(step: &Yaml, pr_event: bool) -> bool {
    if let Some(run) = get(step, "run").and_then(text) {
        let run = run.to_ascii_lowercase();
        let git = [
            "git fetch",
            "git checkout",
            "git pull",
            "git switch",
            "git worktree",
        ]
        .iter()
        .any(|command| run.contains(command));
        if run.contains("refs/pull/")
            || run.contains("gh pr checkout")
            || (git && run.contains("pull/"))
        {
            return true;
        }
    }
    let Some(uses) = get(step, "uses").and_then(text) else {
        return false;
    };
    if !is_checkout(&uses) {
        return false;
    }
    let reference: String = get(step, "with")
        .and_then(|with| get(with, "ref"))
        .and_then(text)
        .unwrap_or_default()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    if reference.is_empty() {
        return pr_event;
    }
    if reference.contains("${{") {
        if BASE_REFS.contains(&reference.as_str()) {
            return false;
        }
        let own = reference == "${{github.sha}}" || reference == "${{github.ref}}";
        return pr_event || !own;
    }
    reference.contains("pull/")
}

/// Whether a step checks out a PR ref, itself or through the local composite action it uses,
/// followed through `actions` (which reads a local action by its `uses:` path) to `depth`.
fn step_checks_out_pr(
    step: &Yaml,
    pr_event: bool,
    actions: &dyn Fn(&str) -> Option<String>,
    depth: usize,
) -> bool {
    if checks_out_pr(step, pr_event) {
        return true;
    }
    let Some(uses) = get(step, "uses").and_then(text) else {
        return false;
    };
    if !uses.starts_with("./") || depth == 0 {
        return false;
    }
    let Some(action) = actions(&uses).and_then(|source| parse(&source).ok()) else {
        return false;
    };
    match get(&action, "runs").and_then(|runs| get(runs, "steps")) {
        Some(Yaml::Array(steps)) => steps
            .iter()
            .any(|step| step_checks_out_pr(step, pr_event, actions, depth - 1)),
        _ => false,
    }
}

/// The expressions under `node`: each `${{ … }}` in a scalar, and each `if:` value whole, which
/// GitHub reads as an expression without braces.
fn expressions(node: &Yaml, out: &mut Vec<String>) {
    match node {
        Yaml::Array(items) => items.iter().for_each(|item| expressions(item, out)),
        Yaml::Hash(map) => {
            for (key, value) in map {
                expressions(key, out);
                if text(key).as_deref() == Some("if") {
                    out.extend(text(value));
                }
                expressions(value, out);
            }
        }
        other => {
            let Some(scalar) = text(other) else { return };
            let mut rest = scalar.as_str();
            while let Some(open) = rest.find("${{") {
                let inside = &rest[open + 3..];
                let close = inside.find("}}").unwrap_or(inside.len());
                out.push(inside[..close].to_owned());
                rest = &inside[close..];
            }
        }
    }
}

/// The first use of the `secrets` context in `expression` other than `GITHUB_TOKEN`, in any case
/// and any form: `secrets.X`, `secrets['X']`, `toJSON(secrets)`.
fn reads_secret(expression: &str) -> Option<String> {
    // ASCII lower-casing keeps every byte offset, so offsets in `lower` index `expression` too.
    let lower = expression.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let word = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'-';
    let mut from = 0;
    while let Some(at) = lower[from..].find("secrets").map(|i| i + from) {
        let end = at + "secrets".len();
        from = end;
        let inside_a_name = at > 0 && (word(bytes[at - 1]) || bytes[at - 1] == b'.');
        if inside_a_name || (end < bytes.len() && word(bytes[end])) {
            continue;
        }
        let rest: String = lower[end..]
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        let token = [".github_token", "['github_token']", "[\"github_token\"]"]
            .iter()
            .any(|form| {
                rest.strip_prefix(form)
                    .is_some_and(|after| after.bytes().next().is_none_or(|b| !word(b)))
            });
        if token {
            continue;
        }
        let named: String = expression[at..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || "_-.[]'\"".contains(*c))
            .collect();
        return Some(named);
    }
    None
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
    expressions(job, &mut all);
    if let Some(env) = workflow_env {
        expressions(env, &mut all);
    }
    all.iter()
        .find_map(|expression| reads_secret(expression))
        .map(|named| format!("it reads `{named}`"))
}

/// Checks one workflow file, named `name` (its file name). `actions` reads a local action by its
/// `uses:` path (`./…`), so a job's composite actions are followed when looking for a PR checkout.
pub fn check_workflow(
    file: &str,
    name: &str,
    source: &str,
    actions: &dyn Fn(&str) -> Option<String>,
) -> Vec<Finding> {
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
            Some(Yaml::Array(steps)) => steps
                .iter()
                .position(|s| step_checks_out_pr(s, pr_event, actions, LOCAL_DEPTH)),
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

/// Checks one action: a composite action's steps' `uses:` are pinned, and a Docker action's
/// `docker://` image is pinned by digest.
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
            if let Some(image) = runs.and_then(|r| get(r, "image")).and_then(text)
                && image.starts_with("docker://")
                && let Some(problem) = unpinned(&image)
            {
                report.add("runs.image", format!("`{image}` {problem}"));
            }
        }
        Err(error) => report.add("", format!("does not parse: {error}")),
    }
    report.findings
}

/// Every `uses:` under `node` that names a local action: `./…`, but not a reusable workflow
/// (`.yml` or `.yaml`).
fn local_uses(node: &Yaml, out: &mut Vec<String>) {
    match node {
        Yaml::Array(items) => items.iter().for_each(|item| local_uses(item, out)),
        Yaml::Hash(map) => {
            for (key, value) in map {
                if text(key).as_deref() == Some("uses")
                    && let Some(uses) = text(value)
                    && uses.starts_with("./")
                    && !(uses.ends_with(".yml") || uses.ends_with(".yaml"))
                {
                    out.push(uses);
                }
                local_uses(value, out);
            }
        }
        _ => {}
    }
}

/// The action file of the local action `uses` (`./dir`) under `root`, if it has one.
fn local_action(root: &Path, uses: &str) -> Option<PathBuf> {
    let dir = root.join(uses.trim_start_matches("./"));
    ["action.yml", "action.yaml"]
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
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
    let read = |path: &Path| fs::read_to_string(path);
    let workflows = workflow_files(root)?;
    let mut actions = action_files(&root.join(".github/actions"))?;
    let mut findings = Vec::new();
    let unreadable = |file: String, error: io::Error| Finding {
        file,
        at: String::new(),
        message: format!("cannot be read: {error}"),
    };

    // Every local action a workflow or an action reaches is checked too, wherever it sits.
    let mut queue: Vec<(String, String)> = Vec::new();
    let queue_uses = |path: &Path, queue: &mut Vec<(String, String)>| {
        if let Ok(yaml) = read(path)
            .map_err(|_| ())
            .and_then(|s| parse(&s).map_err(|_| ()))
        {
            let mut uses = Vec::new();
            local_uses(&yaml, &mut uses);
            queue.extend(uses.into_iter().map(|u| (shown(path), u)));
        }
    };
    for path in workflows.iter().chain(actions.clone().iter()) {
        queue_uses(path, &mut queue);
    }
    while let Some((from, uses)) = queue.pop() {
        match local_action(root, &uses) {
            Some(path) if !actions.contains(&path) => {
                queue_uses(&path, &mut queue);
                actions.push(path);
            }
            Some(_) => {}
            None => findings.push(Finding {
                file: from,
                at: String::new(),
                message: format!("uses `{uses}`, a local action with no action.yml or action.yaml"),
            }),
        }
    }
    actions.sort();
    actions.dedup();

    let resolve = |uses: &str| local_action(root, uses).and_then(|path| read(&path).ok());
    for path in &workflows {
        let file = shown(path);
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        match read(path) {
            Ok(source) => findings.extend(check_workflow(&file, &name, &source, &resolve)),
            Err(error) => findings.push(unreadable(file, error)),
        }
    }
    for path in &actions {
        let file = shown(path);
        match read(path) {
            Ok(source) => findings.extend(check_action(&file, &source)),
            Err(error) => findings.push(unreadable(file, error)),
        }
    }
    for finding in &findings {
        writeln!(log, "workflows: {finding}")?;
    }
    if findings.is_empty() {
        writeln!(
            log,
            "workflows: PASS: {} workflows and {} actions meet RFC-0062 §12.4",
            workflows.len(),
            actions.len()
        )?;
    } else {
        writeln!(
            log,
            "workflows: FAIL: {} problems in the workflows and actions under {} (RFC-0062 §12.4)",
            findings.len(),
            root.display()
        )?;
    }
    Ok(findings.is_empty())
}

#[cfg(test)]
mod tests;
