// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask test runner-build-id`: the build-ID check through the real runner (QUAL-E16-T29;
//! RFC-0001 §3.1; ADR-0015 decision 1).
//!
//! 1. It builds `syntropy-cli` with `SYNTROPY_BUILD_COMMIT` set to another commit, which only
//!    `cli`'s build script reads, and copies that `syntropy` binary aside: a runner of another
//!    build.
//! 2. It builds again without the variable, so the binary left behind is the checkout's own.
//! 3. It runs `crates/cli/tests/runner_build_id.rs`'s ignored test with the copy's path in
//!    `SYNTROPY_OTHER_BUILD_RUNNER`. The test is the daemon: through `daemon`'s real spawn path it
//!    spawns `syntropy __runner` from the copy, which sends its own `build_id::id()` in `Hello`,
//!    and checks that the daemon refuses and kills it; then it spawns the checkout's own binary,
//!    whose `Hello` passes.
//!
//! The second build recompiles `syntropy-cli` alone (the `cache-hit` suite checks that). This
//! suite runs locally; no hosted job runs it yet, which is the maintainers' call.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use crate::cache_hit::build;
use crate::registry::cargo;
use crate::run::Context;

/// The commit the other build claims: not a real one, and never this checkout's.
const OTHER_COMMIT: &str = "b111d1d0b111d1d0b111d1d0b111d1d0b111d1d0";

/// The suite's one step.
pub fn suite_step(context: &Context) -> Result<(), String> {
    let root = &context.root;
    let lines =
        build(root, Some(OTHER_COMMIT)).map_err(|e| format!("the other build failed: {e}"))?;
    let built = executable(&lines)?;
    let copy = std::env::temp_dir().join(format!(
        "syntropy-other-build-{}{}",
        std::process::id(),
        std::env::consts::EXE_SUFFIX
    ));
    let copied = fs::copy(&built, &copy).map_err(|e| format!("{}: {e}", copy.display()));
    // Leave a binary of the checkout's own commit behind, whatever happens next.
    let restored = build(root, None).map_err(|e| format!("the restoring build failed: {e}"));
    copied?;
    restored?;
    let status = Command::new(cargo())
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
            "a_runner_of_another_build_is_refused_and_this_build_passes",
        ])
        .env("SYNTROPY_OTHER_BUILD_RUNNER", &copy)
        .env_remove("SYNTROPY_BUILD_COMMIT")
        .status()
        .map_err(|e| format!("cargo test: {e}"));
    fs::remove_file(&copy).ok();
    let status = status?;
    if !status.success() {
        return Err(format!("the runner build-ID test failed: {status}"));
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
