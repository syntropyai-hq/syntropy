// SPDX-License-Identifier: MIT OR Apache-2.0

//! The exit codes `pr.yml` relies on (QUAL-E1-T06, QUAL-E1-T07). Its `ci-gate` job runs the base
//! branch's `cargo xtask ci ci-gate`, and tells "the base has no `ci-gate` job" from "the gate
//! failed" by the exit code alone: 2 for a usage error, which an unknown job is, and 1 for a
//! failed step.

use std::path::Path;
use std::process::Command;

fn xtask(args: &[&str]) -> (Option<i32>, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .output()
        .expect("xtask runs");
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn recorded(name: &str) -> String {
    fixture(&format!("needs-{name}.json"))
}

fn fixture(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/ci-gate")
        .join(name)
        .display()
        .to_string()
}

#[test]
fn an_unknown_job_is_a_usage_error() {
    let (code, log) = xtask(&["ci", "no-such-job"]);
    assert_eq!(code, Some(2), "{log}");
    assert!(log.contains("unknown job `no-such-job`"), "{log}");
}

#[test]
fn ci_gate_without_its_input_is_a_usage_error() {
    let (code, log) = xtask(&["ci", "ci-gate"]);
    assert_eq!(code, Some(2), "{log}");
}

#[test]
fn ci_gate_exits_0_when_it_passes_and_1_when_it_fails() {
    let event = fixture("event-pull-request.json");
    let statuses = fixture("statuses-none.json");
    let gate = |needs: &str| {
        xtask(&[
            "ci",
            "ci-gate",
            "--needs",
            needs,
            "--event",
            &event,
            "--statuses",
            &statuses,
        ])
    };
    let (code, log) = gate(&recorded("docs-only"));
    assert_eq!(code, Some(0), "{log}");
    let (code, log) = gate(&recorded("fixture-failed"));
    assert_eq!(code, Some(1), "{log}");
    // An unreadable input is a failed step, not a usage error.
    let (code, log) = gate("/no/such/file.json");
    assert_eq!(code, Some(1), "{log}");
}

#[test]
fn ci_gate_without_an_event_is_a_usage_error() {
    // `pr.yml` calls a base with `ci/jobs.toml` with `--event`, and a base without it, which
    // predates QUAL-E1-T07, with `--needs` alone. On a base that has the hardware clause the old
    // form is refused, so it never passes without the clause.
    let (code, log) = xtask(&["ci", "ci-gate", "--needs", &recorded("docs-only")]);
    assert_eq!(code, Some(2), "{log}");
}

#[test]
fn changes_without_its_revisions_is_a_usage_error() {
    let (code, log) = xtask(&["ci", "changes"]);
    assert_eq!(code, Some(2), "{log}");
    let (code, log) = xtask(&["ci", "changes", "--base", "HEAD"]);
    assert_eq!(code, Some(2), "{log}");
}

#[test]
fn dep_diff_without_its_checkouts_is_a_usage_error() {
    // `pr-meta.yml` reads exit code 2 from the base's `dep-diff` as "the base has no such job".
    let (code, log) = xtask(&["ci", "dep-diff"]);
    assert_eq!(code, Some(2), "{log}");
}
