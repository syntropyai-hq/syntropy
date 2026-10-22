// SPDX-License-Identifier: MIT OR Apache-2.0

//! `pr-meta.yml`'s jobs that compare the PR's files with the base's (issue #1702).
//!
//! Such a job, `dep-diff` today and `properties-removal` once it lands, reads the PR from its
//! merge commit, `github.sha`, which is the PR merged into main's tip. Its base must be that
//! merge commit's first parent. `pull_request.base.sha` can be older, and then whatever main
//! gained since would be charged to the PR, as #1681's `spsc` dependencies were on #1682.

const WORKFLOW: &str = include_str!("../../.github/workflows/pr-meta.yml");
const STALE_BASE: &str = include_str!("fixtures/pr_meta/stale-base.yml");

/// The jobs of a workflow: each key two spaces in under `jobs:`, with the lines that follow it.
/// Line endings are normalised first: a Windows checkout has CRLF.
fn jobs(workflow: &str) -> Vec<(String, String)> {
    let workflow = workflow.replace("\r\n", "\n");
    let start = workflow.find("\njobs:\n").expect("a `jobs:` section") + "\njobs:\n".len();
    let mut jobs: Vec<(String, String)> = Vec::new();
    for line in workflow[start..].lines() {
        let key = line
            .strip_prefix("  ")
            .filter(|rest| !rest.starts_with(' ') && !rest.starts_with('#'))
            .and_then(|rest| rest.strip_suffix(':'));
        match (key, jobs.last_mut()) {
            (Some(name), _) => jobs.push((name.to_owned(), String::new())),
            (None, Some((_, text))) => {
                text.push_str(line);
                text.push('\n');
            }
            (None, None) => {}
        }
    }
    jobs
}

/// The steps of a job: each `- ` item of its `steps:` list, with the lines that follow it.
fn steps(job: &str) -> Vec<String> {
    let mut steps: Vec<String> = Vec::new();
    let mut in_steps = false;
    for line in job.lines() {
        if line == "    steps:" {
            in_steps = true;
        } else if !in_steps {
            continue;
        } else if line.starts_with("      - ") {
            steps.push(format!("{line}\n"));
        } else if let Some(step) = steps.last_mut() {
            step.push_str(line);
            step.push('\n');
        }
    }
    steps
}

/// Whether a job compares the PR's files with the base's: it passes the PR checkout as `--head`.
fn compares_with_the_pr(job: &str) -> bool {
    job.contains(r#"--head "$GITHUB_WORKSPACE/pr""#)
}

/// A job's text without its comment lines. Only what runs counts: comments may name the old
/// pattern to explain the new one.
fn uncommented(job: &str) -> String {
    job.lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .map(|line| format!("{line}\n"))
        .collect()
}

/// What is wrong with how a comparing job pairs its base and head, one line each.
fn problems(job: &str) -> Vec<&'static str> {
    let job = uncommented(job);
    let job = job.as_str();
    let mut problems = Vec::new();
    if job.contains("pull_request.base.sha") {
        problems
            .push("checks out pull_request.base.sha, which can be older than the merge ref's main");
    }
    if job.matches("ref: ${{ github.sha }}").count() != 2 {
        problems
            .push("does not check out the merge commit, github.sha, for both the base and the PR");
    }
    if job.matches("fetch-depth: 2").count() != 2 {
        problems.push("does not fetch both checkouts two commits deep, so HEAD^1 is missing");
    }
    // The root holds the PR's tree until the move, so nothing from the root may run before it:
    // no local action (`setup-rust` would run the PR's copy) and no cargo.
    let steps = steps(job);
    let moved = steps
        .iter()
        .position(|step| step.contains("run: git checkout --detach --quiet HEAD^1"));
    let runs_the_root = steps
        .iter()
        .position(|step| step.contains("uses: ./") || step.contains("cargo"));
    match (moved, runs_the_root) {
        (Some(moved), Some(runs)) if moved < runs => {}
        _ => problems.push(
            "does not move to the merge commit's first parent before a local action or cargo runs",
        ),
    }
    if !job.contains(r#"test "$(git -C pr rev-parse HEAD^1)" = "$(git rev-parse HEAD)""#) {
        problems.push("does not check that the base is the head's first parent");
    }
    problems
}

#[test]
fn every_comparing_job_takes_its_base_from_the_merge_commit() {
    let comparing: Vec<(String, String)> = jobs(WORKFLOW)
        .into_iter()
        .filter(|(_, job)| compares_with_the_pr(job))
        .collect();
    let names: Vec<&str> = comparing.iter().map(|(name, _)| name.as_str()).collect();
    assert!(names.contains(&"dep-diff"), "comparing jobs: {names:?}");
    for (name, job) in &comparing {
        assert_eq!(problems(job), Vec::<&str>::new(), "job `{name}`");
    }
}

/// The `dep-diff` job's steps, without comment lines.
fn dep_diff_steps() -> Vec<String> {
    let (_, job) = jobs(WORKFLOW)
        .into_iter()
        .find(|(name, _)| name == "dep-diff")
        .expect("a `dep-diff` job");
    steps(&uncommented(&job))
}

/// A job made of these steps.
fn job_of(steps: &[String]) -> String {
    format!("    steps:\n{}", steps.concat())
}

/// The index of the one step that contains `text`.
fn step_with(steps: &[String], text: &str) -> usize {
    let found: Vec<usize> = (0..steps.len())
        .filter(|&i| steps[i].contains(text))
        .collect();
    assert_eq!(found.len(), 1, "steps containing {text:?}: {found:?}");
    found[0]
}

#[test]
fn moving_to_the_base_after_a_local_action_is_caught() {
    let steps = dep_diff_steps();
    assert_eq!(problems(&job_of(&steps)), Vec::<&str>::new());
    // The move placed after `setup-rust`, which would then run the PR's copy of the action.
    let mut mutated = steps.clone();
    let step = mutated.remove(step_with(
        &steps,
        "run: git checkout --detach --quiet HEAD^1",
    ));
    let setup = step_with(&mutated, "uses: ./.github/actions/setup-rust");
    mutated.insert(setup + 1, step);
    assert!(
        problems(&job_of(&mutated)).contains(
            &"does not move to the merge commit's first parent before a local action or cargo runs"
        ),
        "{:?}",
        problems(&job_of(&mutated))
    );
}

#[test]
fn the_jobs_that_only_read_pr_metadata_are_not_comparing_jobs() {
    // `dco` and `pr-title` read the PR's commits, title and labels, not its files, so the age of
    // their base checkout cannot charge anything to the PR.
    for (name, job) in jobs(WORKFLOW) {
        if name == "dco" || name == "pr-title" {
            assert!(!compares_with_the_pr(&job), "job `{name}`");
        }
    }
}

#[test]
fn the_stale_base_pattern_is_caught() {
    // The `dep-diff` job before issue #1702's fix.
    let stale = jobs(STALE_BASE);
    let (name, job) = stale.first().expect("the fixture's job");
    assert_eq!(name, "dep-diff");
    assert!(compares_with_the_pr(job));
    let found = problems(job);
    assert!(
        found.contains(
            &"checks out pull_request.base.sha, which can be older than the merge ref's main"
        ),
        "{found:?}"
    );
    assert!(
        found.contains(
            &"does not move to the merge commit's first parent before a local action or cargo runs"
        ),
        "{found:?}"
    );
}
