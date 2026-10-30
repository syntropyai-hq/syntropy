// SPDX-License-Identifier: MIT OR Apache-2.0

//! The weekly self-check of the affected-job rules (RFC-0062 "Testing and validation" item 2,
//! §2.2; QUAL-E1-T12).
//!
//! `ci/jobs.toml` names paths, and a file move can leave a pattern that matches nothing, which the
//! golden suite does not see because its paths are recorded. This check builds the diff the weekly
//! bot PR would carry from the tracked files of the checkout: one file under each pattern of each
//! direct rule. It then applies the rules to that diff and fails, naming the rule and the pattern,
//! when
//!
//! - a pattern of a rule matches no tracked file (a removed or moved path);
//! - a file under a pattern does not select its job, alone in its own diff: a lab job must be
//!   required, a hosted job must run, and a planned hosted job must be listed as planned;
//! - the combined diff does not require exactly the lab direct jobs, or does not run exactly the
//!   hosted jobs the rules give it, hosted direct jobs included.
//!
//! A pattern with no tracked file under a rule whose job is `planned` is reported and passes: the
//! suite or rig the rule is for does not exist yet, and its task adds the paths.
//!
//! The scheduled workflow that opens the draft PR is a separate task (QUAL-E1-T35): it needs a
//! write token, which only a maintainer can issue. This module is the check that workflow runs.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use crate::changes::{self, Config, Glob, Job, Runner, When, Workspace};
use crate::run::Context;

/// What the check decided.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// The diff of the self-check PR: one tracked file per pattern, with the rule it stands for.
    pub diff: Vec<(String, String)>,
    /// Patterns of planned rules that no tracked file meets yet.
    pub notices: Vec<String>,
    pub problems: Vec<String>,
}

fn direct_rules(config: &Config) -> impl Iterator<Item = (&Job, &Vec<Glob>)> {
    config.jobs.iter().filter_map(|job| match &job.when {
        When::Direct { paths, .. } => Some((job, paths)),
        _ => None,
    })
}

/// Whether `affected` selects `job`: a lab job is required, a planned hosted job is listed as
/// planned, and any other hosted job runs.
fn selects(affected: &changes::Affected, job: &Job) -> bool {
    match (job.runner, &job.planned) {
        (Runner::Lab, _) => affected.hw_required.contains(&job.name),
        (Runner::Hosted, Some(_)) => affected.planned.iter().any(|(name, _)| *name == job.name),
        (Runner::Hosted, None) => affected.jobs.contains(&job.name),
    }
}

/// The crate that owns `path` for the closure: its member, or the readers of the input it is.
fn owners(config: &Config, workspace: &Workspace, path: &str) -> BTreeSet<String> {
    if let Some(name) = workspace.crate_of(path) {
        return BTreeSet::from([name.to_owned()]);
    }
    config
        .inputs
        .iter()
        .filter(|(glob, _)| glob.matches(path))
        .flat_map(|(_, crates)| crates.iter().cloned())
        .collect()
}

/// Applies the check to the tracked `files` of a checkout.
pub fn check(config: &Config, workspace: &Workspace, files: &[String]) -> Outcome {
    check_with(config, workspace, files, &changes::affected)
}

/// The rules engine as the check calls it: [`changes::affected`], or a stand-in in tests.
type Rules = dyn Fn(&Config, &Workspace, &[String], &BTreeSet<String>) -> changes::Affected;

/// [`check`] over the rules engine `apply`.
fn check_with(config: &Config, workspace: &Workspace, files: &[String], apply: &Rules) -> Outcome {
    let mut out = Outcome::default();
    let mut required = BTreeSet::new();
    let mut hosted_direct = BTreeSet::new();
    let mut picked = BTreeSet::new();
    for (job, patterns) in direct_rules(config) {
        let mut met = false;
        for pattern in patterns {
            let Some(file) = files.iter().find(|f| pattern.matches(f)) else {
                let what = format!(
                    "rule `{}`: pattern `{}` matches no tracked file",
                    job.name,
                    pattern.as_str()
                );
                match &job.planned {
                    Some(task) => out.notices.push(format!("{what}; planned by {task}")),
                    None => out.problems.push(what),
                }
                continue;
            };
            met = true;
            let alone = apply(
                config,
                workspace,
                std::slice::from_ref(file),
                &BTreeSet::new(),
            );
            if !selects(&alone, job) {
                out.problems.push(format!(
                    "rule `{}`: `{file}`, under pattern `{}`, does not select `{}`",
                    job.name,
                    pattern.as_str(),
                    job.name
                ));
            }
            out.diff.push((file.clone(), job.name.clone()));
            picked.insert(file.clone());
        }
        match (met, job.runner, &job.planned) {
            (false, ..) => {}
            (true, Runner::Lab, _) => {
                required.insert(job.name.clone());
            }
            (true, Runner::Hosted, None) => {
                hosted_direct.insert(job.name.clone());
            }
            // A planned hosted job does not run; the per-file check saw it listed as planned.
            (true, Runner::Hosted, Some(_)) => {}
        }
    }

    let paths: Vec<String> = picked.into_iter().collect();
    let combined = apply(config, workspace, &paths, &BTreeSet::new());
    let got: BTreeSet<String> = combined.hw_required.iter().cloned().collect();
    for name in required.difference(&got) {
        out.problems.push(format!(
            "rule `{name}`: the combined diff does not require it"
        ));
    }
    for name in got.difference(&required) {
        out.problems.push(format!(
            "the combined diff requires `{name}`, which no checked rule expects"
        ));
    }

    let mut closure = BTreeSet::new();
    for path in &paths {
        closure.extend(owners(config, workspace, path));
    }
    let closure = workspace.closure(&closure);
    let expected: BTreeSet<String> = config
        .jobs
        .iter()
        .filter(|job| job.runner == Runner::Hosted && job.planned.is_none())
        .filter(|job| match &job.when {
            When::Always => true,
            When::Closure { crates: None } => !closure.is_empty(),
            When::Closure {
                crates: Some(crates),
            } => crates.iter().any(|c| closure.contains(c)),
            When::Direct { .. } => hosted_direct.contains(&job.name),
        })
        .map(|job| job.name.clone())
        .collect();
    let ran: BTreeSet<String> = combined.jobs.iter().cloned().collect();
    for name in expected.difference(&ran) {
        out.problems.push(format!(
            "hosted job `{name}` did not run on the combined diff"
        ));
    }
    for name in ran.difference(&expected) {
        out.problems.push(format!(
            "hosted job `{name}` ran on the combined diff and no rule expects it"
        ));
    }
    out
}

fn tracked_files(root: &Path) -> Result<Vec<String>, String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(["ls-files", "-z"])
        .output()
        .map_err(|e| format!("could not start git: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "`git ls-files` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .split('\0')
        .filter(|f| !f.is_empty())
        .map(str::to_owned)
        .collect())
}

/// `cargo xtask ci self-check`.
pub fn step(context: &Context) -> Result<(), String> {
    let root = &context.root;
    let config = Config::load(root)?;
    let workspace = Workspace::read(&changes::Files(root))?;
    let files = tracked_files(root)?;
    let out = check(&config, &workspace, &files);
    for (file, rule) in &out.diff {
        eprintln!("self-check: {file} (rule `{rule}`)");
    }
    for notice in &out.notices {
        eprintln!("self-check: notice: {notice}");
    }
    if out.problems.is_empty() {
        eprintln!(
            "self-check: {} files, one per pattern of each direct rule, give the expected jobs",
            out.diff.len()
        );
        Ok(())
    } else {
        Err(out.problems.join("\n"))
    }
}

#[cfg(test)]
mod tests;
