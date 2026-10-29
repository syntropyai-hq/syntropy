// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask test runner-build-id`: the build-ID check through the real runner (QUAL-E16-T29;
//! RFC-0001 §3.1; ADR-0015 decision 1).
//!
//! 1. It builds `syntropy-cli` with `SYNTROPY_BUILD_COMMIT` set to another commit, which only
//!    `cli`'s build script reads, and copies that `syntropy` binary into a fresh directory under
//!    `target/xtask/`, removed on every path: a runner of another build.
//! 2. It builds again without the variable, whatever step 1 did, so the binary left behind is the
//!    checkout's own. The copy must report the other commit in `--version --json`, so the build
//!    IDs differ by the commit.
//! 3. It runs `crates/cli/tests/runner_build_id.rs`'s ignored test with the copy's path in
//!    `SYNTROPY_OTHER_BUILD_RUNNER`. The test is the daemon: through `daemon`'s real spawn path it
//!    spawns `syntropy __runner` from the copy, which sends its own `build_id::id()` in `Hello`,
//!    and checks that the daemon refuses and kills it; then it spawns the checkout's own binary,
//!    whose `Hello` passes. The suite requires that test to have run and passed, so a renamed
//!    test cannot pass by running nothing.
//!
//! The second build recompiles `syntropy-cli` alone (the `cache-hit` suite checks that). This
//! suite runs locally; no hosted job runs it yet, which is the maintainers' call.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cache_hit::build;
use crate::registry::cargo;
use crate::run::Context;

/// The commit the other build claims: not a real one, and never this checkout's.
const OTHER_COMMIT: &str = "b111d1d0b111d1d0b111d1d0b111d1d0b111d1d0";

/// The one test the suite runs.
const TEST: &str = "a_runner_of_another_build_is_refused_and_this_build_passes";

/// A directory of this run's own under the checkout's `target/xtask/`, made with an exclusive
/// create and removed when dropped, on every path.
struct Scratch(PathBuf);

impl Scratch {
    fn new(root: &Path) -> Result<Scratch, String> {
        let parent = root.join("target").join("xtask");
        fs::create_dir_all(&parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        let dir = parent.join(format!("runner-build-id-{}", std::process::id()));
        // `create_dir` fails if the path exists, so a planted file or link is never reused.
        fs::create_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(Scratch(dir))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The suite's one step.
pub fn suite_step(context: &Context) -> Result<(), String> {
    let root = &context.root;
    let scratch = Scratch::new(root)?;
    let copy = scratch
        .0
        .join(format!("syntropy{}", std::env::consts::EXE_SUFFIX));
    // The other build and its copy; the restoring build runs whatever they did.
    let other = build(root, Some(OTHER_COMMIT))
        .map_err(|e| format!("the other build failed: {e}"))
        .and_then(|lines| executable(&lines))
        .and_then(|built| {
            fs::copy(&built, &copy)
                .map(|_| ())
                .map_err(|e| format!("{}: {e}", copy.display()))
        });
    let restored = build(root, None).map_err(|e| format!("the restoring build failed: {e}"));
    other?;
    restored?;
    // The copy must be the other commit's build, so the test's mismatch is the commit's.
    let reported = reported_commit(&copy)?;
    if reported != OTHER_COMMIT {
        return Err(format!(
            "the other build reports commit {reported}, not {OTHER_COMMIT}"
        ));
    }
    let output = Command::new(cargo())
        .current_dir(root)
        .args([
            "test",
            "--locked",
            "--package",
            "syntropy-cli",
            "--test",
            "runner_build_id",
            "--",
            "--ignored",
            "--exact",
            TEST,
        ])
        .env("SYNTROPY_OTHER_BUILD_RUNNER", &copy)
        .env_remove("SYNTROPY_BUILD_COMMIT")
        .output()
        .map_err(|e| format!("cargo test: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    print!("{stdout}");
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        return Err(format!(
            "the runner build-ID test failed: {}",
            output.status
        ));
    }
    // A renamed or gated test would run nothing and still exit 0.
    if !stdout.contains(&format!("test {TEST} ... ok")) || !stdout.contains("1 passed") {
        return Err(format!("`{TEST}` did not run and pass"));
    }
    println!(
        "runner-build-id: a runner built from commit {OTHER_COMMIT} was refused and killed at Hello; this build's runner passed"
    );
    Ok(())
}

/// The `syntropy` executable among a build's `--message-format=json` lines.
fn executable(lines: &[String]) -> Result<PathBuf, String> {
    lines
        .iter()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|m| m["reason"] == "compiler-artifact" && m["target"]["name"] == "syntropy")
        .find_map(|m| m["executable"].as_str().map(PathBuf::from))
        .ok_or_else(|| "cargo named no `syntropy` executable".to_owned())
}

/// The commit `binary --version --json` reports.
fn reported_commit(binary: &Path) -> Result<String, String> {
    let output = Command::new(binary)
        .args(["--version", "--json"])
        .output()
        .map_err(|e| format!("{}: {e}", binary.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} --version --json: {}",
            binary.display(),
            output.status
        ));
    }
    let body: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("{} --version --json: {e}", binary.display()))?;
    body["commit"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| "--version --json printed no commit".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_executable_is_the_syntropy_binary() {
        let lines = [
            r#"{"reason":"compiler-artifact","target":{"name":"syntropy_cli"},"executable":null}"#
                .to_owned(),
            r#"{"reason":"compiler-artifact","target":{"name":"syntropy"},"executable":"/t/debug/syntropy"}"#
                .to_owned(),
            r#"{"reason":"build-finished","success":true}"#.to_owned(),
        ];
        assert_eq!(
            executable(&lines).unwrap(),
            PathBuf::from("/t/debug/syntropy")
        );
        assert!(executable(&lines[..1]).is_err());
    }
}
