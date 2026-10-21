// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask ci dep-diff` run end to end on the fixture workspaces in
//! `tests/fixtures/dep-diff/` (QUAL-E1-T19). Each `head-*` changes `base` in one way, and `pr/`
//! and `registry/` hold the recorded PR data and crates.io answers the job reads.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dep-diff")
}

/// The PR data options for the `reviews` file in `pr/`.
fn pr(reviews: &str) -> Vec<String> {
    let at = |file: &str| fixtures().join("pr").join(file).display().to_string();
    vec![
        "--event".into(),
        at("event.json"),
        "--reviews".into(),
        at(reviews),
        "--roster".into(),
        at("roster.toml"),
    ]
}

/// The exit status and the log of `dep-diff` on `head` against `base`, with `extra` options and
/// the recorded registry.
fn run(head: &str, extra: &[String]) -> (Option<i32>, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["ci", "dep-diff", "--base"])
        .arg(fixtures().join("base"))
        .arg("--head")
        .arg(fixtures().join(head))
        .arg("--registry")
        .arg(fixtures().join("registry"))
        .args(extra)
        .output()
        .expect("xtask runs");
    let log = String::from_utf8(output.stderr).expect("UTF-8 output");
    (output.status.code(), log)
}

#[test]
fn a_new_dev_dependency_fails_until_a_maintainer_approves() {
    // No PR data, as in a local run.
    let (code, log) = run("head-dev-dependency", &[]);
    assert_eq!(code, Some(1), "{log}");
    assert!(
        log.contains("dep-diff: new crate `proptest` from crates.io, in fixture-a (dev)"),
        "{log}"
    );
    assert!(log.contains("dep-diff failed"), "{log}");
    // The registry step passes: the name resolves, and its publisher is printed.
    assert!(log.contains("registry passed"), "{log}");
    assert!(
        log.contains(
            "registry: `proptest` on crates.io: first published 2017-06-18, publisher not \
             recorded; the first recorded one is `example-publisher` (version 0.9.2, \
             2019-03-24); owners `example-publisher`, `github:example:publish`"
        ),
        "{log}"
    );

    // PR data with no approval, one on an earlier commit, a contributor's, and a withdrawn one.
    for reviews in [
        "reviews-none.json",
        "reviews-stale.json",
        "reviews-contributor.json",
        "reviews-withdrawn.json",
    ] {
        let (code, log) = run("head-dev-dependency", &pr(reviews));
        assert_eq!(code, Some(1), "{reviews}: {log}");
        assert!(
            log.contains("no maintainer has approved them in a review of the head commit"),
            "{reviews}: {log}"
        );
    }

    // A maintainer's approving review on the head commit passes it.
    let (code, log) = run("head-dev-dependency", &pr("reviews-approved.json"));
    assert_eq!(code, Some(0), "{log}");
    assert!(
        log.contains(
            "dep-diff: approved by `alice`, in a review of 1111111111111111111111111111111111111111"
        ),
        "{log}"
    );
}

#[test]
fn a_patch_or_a_source_replacement_needs_approval_too() {
    let (code, log) = run("head-patch", &[]);
    assert_eq!(code, Some(1), "{log}");
    assert!(
        log.contains(
            "dep-diff: new dependency source setting in Cargo.toml: [patch.crates-io] serde = { \
             git = \"https://example.invalid/serde-fork.git\" }"
        ),
        "{log}"
    );
    let (code, log) = run("head-source-replacement", &[]);
    assert_eq!(code, Some(1), "{log}");
    assert!(
        log.contains(
            "dep-diff: new dependency source setting in crates/a/.cargo/config.toml: \
             [source.crates-io] replace-with = \"mirror\""
        ),
        "{log}"
    );
    assert!(
        log.contains("[source.mirror] registry = \"https://example.invalid/index\""),
        "{log}"
    );
    // Approval covers them as it covers a new crate.
    let (code, log) = run("head-patch", &pr("reviews-approved.json"));
    assert_eq!(code, Some(0), "{log}");
}

#[test]
fn a_version_bump_of_an_existing_dependency_passes() {
    let (code, log) = run("head-bump", &[]);
    assert_eq!(code, Some(0), "{log}");
    assert!(
        log.contains("dep-diff: no new direct dependency or vendored component"),
        "{log}"
    );
    assert!(log.contains("registry: no new crate to look up"), "{log}");
}

#[test]
fn a_new_vendored_component_fails() {
    let (code, log) = run("head-vendored", &[]);
    assert_eq!(code, Some(1), "{log}");
    assert!(
        log.contains("dep-diff: new vendored component from https://example.invalid/fixture-c"),
        "{log}"
    );
}

#[test]
fn a_crate_name_that_does_not_exist_fails_the_registry_check() {
    let approved = pr("reviews-approved.json");
    let (code, log) = run("head-typo", &approved);
    assert_eq!(code, Some(1), "{log}");
    // Approval does not excuse it: dep-diff passes, and the registry step fails.
    assert!(log.contains("dep-diff passed"), "{log}");
    assert!(
        log.contains("`serde_jsno` does not resolve on crates.io"),
        "{log}"
    );
    assert!(log.contains("FAIL (1 of 2 failed: registry)"), "{log}");
}

#[test]
fn a_git_dependency_is_new_and_has_no_registry_entry() {
    let (code, log) = run("head-git", &[]);
    assert_eq!(code, Some(1), "{log}");
    assert!(
        log.contains("new crate `fixture-build-helper` from `git+https://example.invalid"),
        "{log}"
    );
    assert!(log.contains("fixture-a (build)"), "{log}");
    assert!(log.contains("it has no registry entry to check"), "{log}");
    assert!(log.contains("registry passed"), "{log}");
}

#[test]
fn approval_reads_all_of_the_pr_data_or_none() {
    let at = |file: &str| fixtures().join("pr").join(file).display().to_string();
    let (code, log) = run(
        "head-dev-dependency",
        &[
            "--event".into(),
            at("event.json"),
            "--roster".into(),
            at("roster.toml"),
        ],
    );
    assert_eq!(code, Some(1), "{log}");
    assert!(
        log.contains("but only --event, --roster were given"),
        "{log}"
    );
}
