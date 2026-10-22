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
fn jobs(workflow: &str) -> Vec<(String, String)> {
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

/// Whether a job compares the PR's files with the base's: it passes the PR checkout as `--head`.
fn compares_with_the_pr(job: &str) -> bool {
    job.contains(r#"--head "$GITHUB_WORKSPACE/pr""#)
}

/// What is wrong with how a comparing job pairs its base and head, one line each.
fn problems(job: &str) -> Vec<&'static str> {
    // Only what runs counts: comment lines may name the old pattern to explain the new one.
    let job: String = job
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .map(|line| format!("{line}\n"))
        .collect();
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
    match (
        job.find("run: git checkout --detach --quiet HEAD^1"),
        job.find("cargo xtask"),
    ) {
        (Some(base), Some(xtask)) if base < xtask => {}
        _ => problems
            .push("does not move to the merge commit's first parent before cargo xtask runs"),
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
        found.contains(&"does not move to the merge commit's first parent before cargo xtask runs"),
        "{found:?}"
    );
}
