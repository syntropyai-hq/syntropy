// SPDX-License-Identifier: MIT OR Apache-2.0

//! The exit codes `pr.yml` relies on (QUAL-E1-T06). Its `ci-gate` job runs the base branch's
//! `cargo xtask ci ci-gate`, and tells "the base has no `ci-gate` job" from "the gate failed"
//! by the exit code alone: 2 for a usage error, which an unknown job is, and 1 for a failed step.

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
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/ci-gate")
        .join(format!("needs-{name}.json"))
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
    let (code, log) = xtask(&["ci", "ci-gate", "--needs", &recorded("fixture-skipped")]);
    assert_eq!(code, Some(0), "{log}");
    let (code, log) = xtask(&["ci", "ci-gate", "--needs", &recorded("fixture-failed")]);
    assert_eq!(code, Some(1), "{log}");
    // An unreadable input is a failed step, not a usage error.
    let (code, log) = xtask(&["ci", "ci-gate", "--needs", "/no/such/file.json"]);
    assert_eq!(code, Some(1), "{log}");
}

#[test]
fn dep_diff_without_its_checkouts_is_a_usage_error() {
    // `pr-meta.yml` reads exit code 2 from the base's `dep-diff` as "the base has no such job".
    let (code, log) = xtask(&["ci", "dep-diff"]);
    assert_eq!(code, Some(2), "{log}");
}
