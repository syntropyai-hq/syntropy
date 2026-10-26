// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask ci freeze` run end to end on the fixture logs in `tests/fixtures/freeze/`
//! (QUAL-E17-T04). The fixtures stand in for the event log `rc:start` will write (QUAL-E17-T05).

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/freeze")
        .join(name)
}

fn run(events: &str, pr: Option<&str>, out: Option<&Path>) -> (Option<i32>, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .args(["ci", "freeze", "--events"])
        .arg(fixture(events));
    if let Some(pr) = pr {
        command.arg("--pr").arg(fixture(pr));
    }
    if let Some(out) = out {
        command.arg("--durations-out").arg(out);
    }
    let output = command.output().expect("xtask runs");
    let log = String::from_utf8_lossy(&output.stderr).into_owned();
    (output.status.code(), log)
}

#[test]
fn a_frozen_main_holds_an_unlabelled_pr() {
    let (code, log) = run("frozen.json", Some("pr-plain.json"), None);
    assert_eq!(code, Some(1), "{log}");
    assert!(log.contains("PR #11 held"), "{log}");
}

#[test]
fn a_frozen_main_lets_an_rc_fix_pr_merge() {
    let (code, log) = run("frozen.json", Some("pr-fix.json"), None);
    assert_eq!(code, Some(0), "{log}");
}

#[test]
fn a_lifted_freeze_holds_nothing() {
    let (code, log) = run("lifted.json", Some("pr-plain.json"), None);
    assert_eq!(code, Some(0), "{log}");
}

#[test]
fn durations_are_written_for_the_cost_report() {
    let out = std::env::temp_dir().join(format!("freeze-durations-{}.json", std::process::id()));
    let (code, log) = run("frozen.json", None, Some(&out));
    assert_eq!(code, Some(0), "{log}");
    let text = std::fs::read_to_string(&out).expect("durations written");
    std::fs::remove_file(&out).ok();
    let windows: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(windows[0]["seconds"], 3600);
    assert_eq!(windows[0]["lifted_by"], "publish");
    assert!(windows[1]["end"].is_null());
}
