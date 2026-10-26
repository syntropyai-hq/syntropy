// SPDX-License-Identifier: MIT OR Apache-2.0

//! The workflow security lint's rules (RFC-0062 §12.4), each on a small inline workflow, and the
//! lint over the fixture trees under `tests/fixtures/workflows/` and over this repository.

use std::path::Path;

use super::*;

const SHA: &str = "3d3c42e5aac5ba805825da76410c181273ba90b1";

/// A workflow that meets every rule: read-only, a pinned checkout, a hosted runner.
fn clean() -> String {
    format!(
        "on: [push]\n\
         permissions:\n  contents: read\n\
         jobs:\n  build:\n    runs-on: ubuntu-24.04\n    steps:\n      \
         - uses: actions/checkout@{SHA}\n      - run: cargo xtask ci lint\n"
    )
}

/// The findings' `at` and the first words of each message, for compact assertions.
fn found(file: &str, source: &str) -> Vec<(String, String)> {
    check_workflow("w.yml", file, source, &|_| None)
        .into_iter()
        .map(|f| (f.at, f.message))
        .collect()
}

fn assert_clean(file: &str, source: &str) {
    let findings = found(file, source);
    assert!(findings.is_empty(), "{findings:#?}\n{source}");
}

/// Exactly one finding, at `at`, whose message contains `needle`.
fn assert_one(file: &str, source: &str, at: &str, needle: &str) {
    let findings = found(file, source);
    assert_eq!(findings.len(), 1, "{findings:#?}\n{source}");
    assert_eq!(findings[0].0, at, "{findings:#?}");
    assert!(findings[0].1.contains(needle), "{findings:#?}");
}

#[test]
fn a_clean_workflow_passes() {
    assert_clean("w.yml", &clean());
}

#[test]
fn the_top_level_permissions_must_be_read_only() {
    let base = clean();
    let without = base.replace("permissions:\n  contents: read\n", "");
    assert_one("w.yml", &without, "permissions", "sets no top-level");
    let write_all = base.replace(
        "permissions:\n  contents: read\n",
        "permissions: write-all\n",
    );
    assert_one("w.yml", &write_all, "permissions", "grants write");
    let one_write = base.replace("  contents: read\n", "  contents: read\n  issues: write\n");
    assert_one("w.yml", &one_write, "permissions", "grants write");
    let bad_level = base.replace("  contents: read\n", "  contents: admin\n");
    assert_one(
        "w.yml",
        &bad_level,
        "permissions",
        "must be `read`, `write` or `none`",
    );
    assert_clean(
        "w.yml",
        &base.replace(
            "permissions:\n  contents: read\n",
            "permissions: read-all\n",
        ),
    );
    assert_clean(
        "w.yml",
        &base.replace("  contents: read\n", "  contents: read\n  packages: none\n"),
    );
}

#[test]
fn a_job_may_widen_its_own_permissions() {
    let source = clean().replace(
        "    runs-on: ubuntu-24.04\n",
        "    runs-on: ubuntu-24.04\n    permissions:\n      actions: write\n",
    );
    assert_clean("w.yml", &source);
}

#[test]
fn every_action_is_pinned_by_full_sha() {
    let base = clean();
    let checkout = format!("actions/checkout@{SHA}");
    for (uses, needle) in [
        ("actions/checkout@v4", "not to a full 40-hex commit SHA"),
        ("actions/checkout@main", "not to a full 40-hex commit SHA"),
        (
            "actions/checkout@3d3c42e",
            "not to a full 40-hex commit SHA",
        ),
        ("actions/checkout", "names no version"),
        ("${{ matrix.action }}", "computed by an expression"),
        ("docker://alpine:3.20", "`@sha256:` digest"),
    ] {
        assert_one(
            "w.yml",
            &base.replace(&checkout, uses),
            "jobs.build.steps[0].uses",
            needle,
        );
    }
    let digest = format!("docker://alpine@sha256:{}", "a".repeat(64));
    assert_clean("w.yml", &base.replace(&checkout, &digest));
    assert_clean(
        "w.yml",
        &base.replace(&checkout, "./.github/actions/setup-rust"),
    );
}

#[test]
fn a_reusable_workflow_is_pinned_too() {
    let tagged = "on: [push]\npermissions:\n  contents: read\njobs:\n  call:\n    \
                  uses: octo/repo/.github/workflows/ci.yml@v1\n";
    assert_one(
        "w.yml",
        tagged,
        "jobs.call.uses",
        "not to a full 40-hex commit SHA",
    );
    let local = tagged.replace(
        "octo/repo/.github/workflows/ci.yml@v1",
        "./.github/workflows/ci.yml",
    );
    assert_clean("w.yml", &local);
}

#[test]
fn pull_request_target_is_refused_in_every_form_of_on() {
    let base = clean();
    for on in [
        "on: pull_request_target\n",
        "on: [push, pull_request_target]\n",
        "on:\n  pull_request_target:\n    types: [opened]\n",
    ] {
        let source = base.replace("on: [push]\n", on);
        assert_one("w.yml", &source, "on", "pull_request_target");
    }
}

#[test]
fn a_workflow_without_triggers_or_jobs_fails() {
    assert_one(
        "w.yml",
        &clean().replace("on: [push]\n", ""),
        "on",
        "no `on:` triggers",
    );
    let no_jobs = "on: [push]\npermissions:\n  contents: read\n";
    assert_one("w.yml", no_jobs, "jobs", "no jobs");
}

#[test]
fn lab_labels_are_refused_outside_lab_yml() {
    let base = clean();
    for runs_on in [
        "[self-hosted, syntropy-lab]",
        "self-hosted",
        "[ubuntu-24.04, R-ADA]",
        "lab-run-6f1c",
        "[self-hosted, cuda-sm89]",
        "[metal-m4]",
        "{ group: syntropy-lab }",
        "{ labels: [rdma-tb5] }",
    ] {
        let source = base.replace("runs-on: ubuntu-24.04", &format!("runs-on: {runs_on}"));
        let findings = found("pr.yml", &source);
        assert!(
            !findings.is_empty()
                && findings
                    .iter()
                    .all(|(at, m)| at == "jobs.build.runs-on" && m.contains("lab label")),
            "{runs_on}: {findings:#?}"
        );
        assert_clean("lab.yml", &source);
    }
}

#[test]
fn hosted_runner_labels_pass() {
    for runs_on in [
        "ubuntu-24.04",
        "macos-26",
        "windows-2025",
        "[ubuntu-24.04-arm]",
    ] {
        let source = clean().replace("runs-on: ubuntu-24.04", &format!("runs-on: {runs_on}"));
        assert_clean("pr.yml", &source);
    }
}

#[test]
fn a_matrix_runs_on_is_read_through_the_matrix() {
    let job = |matrix: &str| {
        clean().replace(
            "    runs-on: ubuntu-24.04\n",
            &format!("    strategy:\n      matrix:\n{matrix}    runs-on: ${{{{ matrix.os }}}}\n"),
        )
    };
    let hosted = job("        os: [ubuntu-24.04, macos-26, windows-2025]\n");
    assert_clean("pr.yml", &hosted);
    let included =
        job("        include:\n          - os: ubuntu-24.04\n          - os: self-hosted\n");
    assert_one(
        "pr.yml",
        &included,
        "jobs.build.runs-on",
        "lab label `self-hosted`",
    );
    let missing = job("        arch: [x64]\n");
    assert_one(
        "pr.yml",
        &missing,
        "jobs.build.runs-on",
        "gives `os` no value",
    );
    let computed = job("        os: ${{ fromJSON(needs.plan.outputs.os) }}\n");
    assert_one("pr.yml", &computed, "jobs.build.runs-on", "computes `os`");
    let other = clean().replace("runs-on: ubuntu-24.04", "runs-on: ${{ inputs.runner }}");
    assert_one("pr.yml", &other, "jobs.build.runs-on", "is computed");
}

/// A job on `pull_request` that checks out the PR and also reads a secret.
fn pr_checkout_job(extra_job: &str, step: &str) -> String {
    format!(
        "on:\n  pull_request:\n\
         permissions:\n  contents: read\n\
         jobs:\n  build:\n    runs-on: ubuntu-24.04\n{extra_job}    steps:\n      {step}\n      \
         - run: make\n"
    )
}

#[test]
fn a_pr_checkout_beside_a_secret_or_a_write_token_fails() {
    let checkout = format!("- uses: actions/checkout@{SHA}");
    let secret_env = "    env:\n      TOKEN: ${{ secrets.DEPLOY_KEY }}\n";
    assert_one(
        "w.yml",
        &pr_checkout_job(secret_env, &checkout),
        "jobs.build.steps[0]",
        "it reads `secrets.DEPLOY_KEY`",
    );
    let write = "    permissions:\n      contents: write\n";
    assert_one(
        "w.yml",
        &pr_checkout_job(write, &checkout),
        "jobs.build.steps[0]",
        "its token can write",
    );
    let environment = "    environment: release\n";
    assert_one(
        "w.yml",
        &pr_checkout_job(environment, &checkout),
        "jobs.build.steps[0]",
        "names an environment",
    );
    let gh_checkout = "- run: gh pr checkout 12";
    assert_one(
        "w.yml",
        &pr_checkout_job(secret_env, gh_checkout),
        "jobs.build.steps[0]",
        "secret or a write token",
    );
    let fetch = "- run: git fetch origin refs/pull/12/head";
    assert_one(
        "w.yml",
        &pr_checkout_job(secret_env, fetch),
        "jobs.build.steps[0]",
        "secret or a write token",
    );
}

#[test]
fn a_pr_checkout_with_only_the_read_only_token_passes() {
    let checkout = format!("- uses: actions/checkout@{SHA}");
    let github_token = "    env:\n      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n";
    assert_clean("w.yml", &pr_checkout_job(github_token, &checkout));
    assert_clean("w.yml", &pr_checkout_job("", &checkout));
}

#[test]
fn a_checkout_of_the_base_on_pull_request_is_not_a_pr_checkout() {
    let base = format!(
        "- uses: actions/checkout@{SHA}\n        with:\n          \
         ref: ${{{{ github.event.pull_request.base.sha }}}}"
    );
    let secret_env = "    env:\n      TOKEN: ${{ secrets.DEPLOY_KEY }}\n";
    assert_clean("w.yml", &pr_checkout_job(secret_env, &base));
}

#[test]
fn a_named_pr_ref_counts_on_any_trigger() {
    let source = format!(
        "on:\n  workflow_run:\n    workflows: [pr]\n    types: [completed]\n\
         permissions:\n  contents: read\n\
         jobs:\n  report:\n    runs-on: ubuntu-24.04\n    secrets: inherit\n    steps:\n      \
         - uses: actions/checkout@{SHA}\n        with:\n          \
         ref: ${{{{ github.event.workflow_run.head_sha }}}}\n"
    );
    assert_one("w.yml", &source, "jobs.report.steps[0]", "secrets: inherit");
    // On `push`, a checkout with no `ref` gets the pushed commit, not a PR.
    let push = source
        .replace(
            "  workflow_run:\n    workflows: [pr]\n    types: [completed]\n",
            "  push:\n",
        )
        .replace(
            "        with:\n          ref: ${{ github.event.workflow_run.head_sha }}\n",
            "",
        );
    assert_clean("w.yml", &push);
}

#[test]
fn a_secret_in_the_workflow_env_counts_for_every_job() {
    let checkout = format!("- uses: actions/checkout@{SHA}");
    let source = pr_checkout_job("", &checkout).replace(
        "jobs:\n",
        "env:\n  KEY: ${{ secrets.SIGNING_KEY }}\njobs:\n",
    );
    assert_one(
        "w.yml",
        &source,
        "jobs.build.steps[0]",
        "it reads `secrets.SIGNING_KEY`",
    );
}

#[test]
fn a_file_that_cannot_be_read_fails() {
    assert_one("w.yml", "on: [push\n", "", "does not parse");
    assert_one("w.yml", "", "", "the file is empty");
    assert_one("w.yml", "a: 1\n---\nb: 2\n", "", "2 YAML documents");
    assert_one("w.yml", "- a list\n", "", "is not a map");
    let jobs_list = "on: [push]\npermissions:\n  contents: read\njobs: [build]\n";
    assert_one("w.yml", jobs_list, "jobs", "is not a map");
}

#[test]
fn a_composite_actions_steps_are_pinned() {
    let action =
        |uses: &str| format!("name: a\nruns:\n  using: composite\n  steps:\n    - uses: {uses}\n");
    let findings = check_action("a/action.yml", &action("actions/cache@v4"));
    assert_eq!(findings.len(), 1, "{findings:#?}");
    assert_eq!(findings[0].at, "runs.steps[0].uses");
    assert!(check_action("a/action.yml", &action(&format!("actions/cache@{SHA}"))).is_empty());
    assert!(check_action("a/action.yml", "runs: [\n").len() == 1);
}

/// The lint over one fixture tree: whether it passed, and its report.
fn lint_fixture(name: &str) -> (bool, String) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/workflows")
        .join(name);
    let mut log = Vec::new();
    let pass = run(&root, &mut log).expect("the fixture is readable");
    (pass, String::from_utf8(log).expect("the report is UTF-8"))
}

/// The task's criterion: each fixture workflow fails the lint for its one fault, and the clean
/// fixture, with a workflow and a composite action, passes.
#[test]
fn each_acceptance_fixture_fails_for_its_fault() {
    for (fixture, needle) in [
        ("write-all", "grants write"),
        ("tag-pinned", "is pinned to `v4`"),
        ("pull-request-target", "pull_request_target"),
        ("lab-label", "names the lab label `self-hosted`"),
        (
            "pr-checkout-secret",
            "checks out a PR ref in a job that holds a secret",
        ),
        (
            "local-action-unpinned",
            "ci/act/action.yml: runs.steps[0].uses",
        ),
    ] {
        let (pass, log) = lint_fixture(fixture);
        assert!(!pass, "{fixture}: {log}");
        assert!(log.contains(needle), "{fixture}: {log}");
        assert!(log.contains("workflows: FAIL"), "{fixture}: {log}");
    }
    let (pass, log) = lint_fixture("clean");
    assert!(pass, "{log}");
    assert!(
        log.contains("workflows: PASS: 1 workflows and 1 actions"),
        "{log}"
    );
}

/// This repository's workflows and composite actions meet the rules.
#[test]
fn this_repository_passes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut log = Vec::new();
    let pass = run(&root, &mut log).expect("the checkout is readable");
    let log = String::from_utf8(log).expect("the report is UTF-8");
    assert!(pass, "{log}");
}

/// Every form of the `secrets` context counts, in any case, `if:` values included; only
/// `GITHUB_TOKEN` does not.
#[test]
fn every_form_of_the_secrets_context_counts() {
    let checkout = format!("- uses: actions/checkout@{SHA}");
    for (env, named) in [
        ("    env:\n      ALL: ${{ toJSON(secrets) }}\n", "secrets"),
        (
            "    env:\n      KEY: ${{ secrets['DEPLOY'] }}\n",
            "secrets['DEPLOY']",
        ),
        (
            "    env:\n      KEY: ${{ SECRETS.DEPLOY }}\n",
            "SECRETS.DEPLOY",
        ),
        ("    if: secrets.DEPLOY != ''\n", "secrets.DEPLOY"),
        (
            "    env:\n      KEY: ${{ format('{0}', secrets.X) }}\n",
            "secrets.X",
        ),
    ] {
        assert_one(
            "w.yml",
            &pr_checkout_job(env, &checkout),
            "jobs.build.steps[0]",
            &format!("it reads `{named}`"),
        );
    }
    for env in [
        "    env:\n      T: ${{ secrets['GITHUB_TOKEN'] }}\n",
        "    env:\n      T: ${{ Secrets.github_token }}\n",
        "    env:\n      T: ${{ github.event.secrets_url }}\n",
    ] {
        assert_clean("w.yml", &pr_checkout_job(env, &checkout));
    }
}

/// A called workflow cannot see its caller's trigger, so a default checkout there is a PR
/// checkout.
#[test]
fn a_called_workflow_counts_as_a_pr_trigger() {
    let source = format!(
        "on:\n  workflow_call:\n    secrets:\n      token:\n        required: true\n\
         permissions:\n  contents: read\n\
         jobs:\n  build:\n    runs-on: ubuntu-24.04\n    steps:\n      \
         - uses: actions/checkout@{SHA}\n      - run: make\n        env:\n          \
         TOKEN: ${{{{ secrets.token }}}}\n"
    );
    assert_one(
        "w.yml",
        &source,
        "jobs.build.steps[0]",
        "it reads `secrets.token`",
    );
}

/// Only refs known to be safe pass; anything else beside a secret fails.
#[test]
fn only_known_safe_checkout_refs_pass() {
    let secret_env = "    env:\n      TOKEN: ${{ secrets.DEPLOY_KEY }}\n";
    let with_ref =
        |r: &str| format!("- uses: actions/checkout@{SHA}\n        with:\n          ref: {r}");
    for unsafe_step in [
        with_ref("''"),
        with_ref("${{ inputs.ref }}"),
        with_ref("${{ github.event.after }}"),
        with_ref("${{ github.SHA }}"),
        with_ref("${{ github.event['pull_request'].head.sha }}"),
        with_ref("refs/pull/7/merge"),
        format!("- uses: Actions/Checkout@{SHA}"),
        format!("- uses: actions/checkout/@{SHA}"),
        "- run: git fetch origin pull/7/head".to_owned(),
    ] {
        let findings = found("w.yml", &pr_checkout_job(secret_env, &unsafe_step));
        assert!(
            findings
                .iter()
                .any(|(at, m)| at == "jobs.build.steps[0]" && m.contains("checks out a PR ref")),
            "{unsafe_step}: {findings:#?}"
        );
    }
    for safe_step in [
        with_ref("main"),
        with_ref(SHA),
        with_ref("${{ github.base_ref }}"),
        with_ref("${{ github.event.repository.default_branch }}"),
    ] {
        assert_clean("w.yml", &pr_checkout_job(secret_env, &safe_step));
    }
    // On `push`, the event's own commit is the repository's code.
    let push = pr_checkout_job(secret_env, &with_ref("${{ github.sha }}"))
        .replace("on:\n  pull_request:\n", "on:\n  push:\n");
    assert_clean("w.yml", &push);
}

/// A step that uses a local composite action which checks out the PR counts, through a chain of
/// actions.
#[test]
fn a_pr_checkout_inside_a_local_action_counts() {
    let secret_env = "    env:\n      TOKEN: ${{ secrets.DEPLOY_KEY }}\n";
    let source = pr_checkout_job(secret_env, "- uses: ./ci/outer");
    let outer = "runs:\n  using: composite\n  steps:\n    - uses: ./ci/inner\n".to_owned();
    let inner =
        format!("runs:\n  using: composite\n  steps:\n    - uses: actions/checkout@{SHA}\n");
    let resolve = |uses: &str| match uses {
        "./ci/outer" => Some(outer.clone()),
        "./ci/inner" => Some(inner.clone()),
        _ => None,
    };
    let findings = check_workflow("w.yml", "w.yml", &source, &resolve);
    assert_eq!(findings.len(), 1, "{findings:#?}");
    assert_eq!(findings[0].at, "jobs.build.steps[0]");
    let harmless = |uses: &str| {
        (uses == "./ci/outer")
            .then(|| "runs:\n  using: composite\n  steps:\n    - run: make\n".to_owned())
    };
    assert!(check_workflow("w.yml", "w.yml", &source, &harmless).is_empty());
}

/// A Docker action's `docker://` image is pinned by digest.
#[test]
fn a_docker_actions_image_is_pinned() {
    let action = |image: &str| format!("runs:\n  using: docker\n  image: {image}\n");
    let findings = check_action("a/action.yml", &action("docker://alpine:3.20"));
    assert_eq!(findings.len(), 1, "{findings:#?}");
    assert_eq!(findings[0].at, "runs.image");
    let digest = format!("docker://alpine@sha256:{}", "b".repeat(64));
    assert!(check_action("a/action.yml", &action(&digest)).is_empty());
    assert!(check_action("a/action.yml", &action("Dockerfile")).is_empty());
}

/// Review triggers, and any trigger not known to check out the repository's own code, count as a
/// pull request's.
#[test]
fn only_known_triggers_make_a_default_checkout_safe() {
    let secret_env = "    env:\n      TOKEN: ${{ secrets.NPM_TOKEN }}\n";
    let checkout = format!("- uses: actions/checkout@{SHA}");
    for on in [
        "  pull_request_review:\n",
        "  pull_request_review_comment:\n",
        "  some_future_event:\n",
    ] {
        let source = pr_checkout_job(secret_env, &checkout).replace("  pull_request:\n", on);
        assert_one(
            "w.yml",
            &source,
            "jobs.build.steps[0]",
            "checks out a PR ref",
        );
    }
    for on in ["  push:\n", "  issue_comment:\n", "  workflow_run:\n"] {
        let source = pr_checkout_job(secret_env, &checkout).replace("  pull_request:\n", on);
        assert_clean("w.yml", &source);
    }
}

/// A checkout of another repository counts, whatever its ref: the `workflow_run` fork case.
#[test]
fn a_checkout_of_another_repository_counts() {
    let secret_env = "    env:\n      TOKEN: ${{ secrets.NPM_TOKEN }}\n";
    let with = |repository: &str| {
        format!(
            "- uses: actions/checkout@{SHA}\n        with:\n          repository: {repository}\n          ref: main"
        )
    };
    for repository in [
        "${{ github.event.workflow_run.head_repository.full_name }}",
        "someone/fork",
    ] {
        let source = pr_checkout_job(secret_env, &with(repository))
            .replace("  pull_request:\n", "  push:\n");
        assert_one(
            "w.yml",
            &source,
            "jobs.build.steps[0]",
            "checks out a PR ref",
        );
    }
    for repository in ["${{ github.repository }}", "SyntropyAI-HQ/syntropy"] {
        let source = pr_checkout_job(secret_env, &with(repository))
            .replace("  pull_request:\n", "  push:\n");
        assert_clean("w.yml", &source);
    }
}

/// git and gh options between the program and its subcommand do not hide a PR fetch.
#[test]
fn git_options_do_not_hide_a_pr_fetch() {
    let secret_env = "    env:\n      TOKEN: ${{ secrets.NPM_TOKEN }}\n";
    for step in [
        "- run: git -c protocol.version=2 fetch origin pull/7/head",
        "- run: git -C repo --no-pager fetch origin pull/7/head",
        "- run: gh -R owner/repo pr checkout 7",
    ] {
        let source =
            pr_checkout_job(secret_env, step).replace("  pull_request:\n", "  issue_comment:\n");
        assert_one(
            "w.yml",
            &source,
            "jobs.build.steps[0]",
            "checks out a PR ref",
        );
    }
}

/// A `}}` inside a quoted string does not end the expression.
#[test]
fn a_quoted_closing_brace_does_not_hide_a_secret() {
    let checkout = format!("- uses: actions/checkout@{SHA}");
    let env = "    env:\n      KEY: ${{ format('}}{0}', secrets.NPM_TOKEN) }}\n";
    assert_one(
        "w.yml",
        &pr_checkout_job(env, &checkout),
        "jobs.build.steps[0]",
        "it reads `secrets.NPM_TOKEN`",
    );
    let mut ends = Vec::new();
    expressions(
        &Yaml::String("${{ 'it''s }}' }} tail".to_owned()),
        &mut ends,
    );
    assert_eq!(ends, [" 'it''s }}' ".to_owned()]);
}
