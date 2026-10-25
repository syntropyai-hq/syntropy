// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask test loom`: the loom models of RFC-0060 §3.4 (QUAL-E3-T09), run under loom's model
//! checker.
//!
//! The models sit next to the code they check, under `#[cfg(loom)]`: the SPSC rings' in
//! `syntropy-spsc`, and the IPC region's `local_tokens` and `local_sequences` rings' in
//! `syntropy-ipc`. The suite runs those two crates' unit tests with `--cfg loom` added to the
//! compiler flags, with `LOOM_MAX_PREEMPTIONS=3` unless the caller set a bound, and in a target
//! directory of its own, `loom/` under the usual one, since other compiler flags would otherwise
//! rebuild the whole workspace build in place. It then checks that the models the task names
//! passed, so a build that left loom out cannot pass by running none of them.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::run::Context;

/// The crates that hold loom models.
pub const PACKAGES: &[&str] = &["syntropy-spsc", "syntropy-ipc"];

/// The preemption bound QUAL-E3-T09 sets, unless `LOOM_MAX_PREEMPTIONS` is set already.
const PREEMPTIONS: &str = "3";

/// The models that must pass, by test path: the three-thread models of each ring, and the two
/// that find a relaxed publishing store planted in each ring. They exist only under `--cfg loom`
/// or run only under it as models.
pub const REQUIRED: &[&str] = &[
    "ring::tests::values_cross_a_three_thread_pipeline_in_order",
    "ring::tests::the_models_find_a_relaxed_publishing_store",
    "shared::tests::the_models_find_a_relaxed_publishing_store_in_a_shared_ring",
    "shm::loom_model::the_local_rings_carry_records_both_ways_across_three_threads",
];

/// The variables the suite's cargo run sets, from the caller's environment `var`.
#[derive(Debug, PartialEq, Eq)]
pub struct Environment {
    /// `RUSTFLAGS`, or `CARGO_ENCODED_RUSTFLAGS` when the caller set that, which cargo then reads
    /// instead, with `--cfg loom` added.
    pub flags: (&'static str, String),
    /// `CARGO_TARGET_DIR`: `loom/` under the caller's target directory.
    pub target_dir: PathBuf,
    /// `LOOM_MAX_PREEMPTIONS`.
    pub preemptions: String,
}

/// The suite's environment, for a workspace at `root` and a caller whose variables `var` reads.
pub fn environment(root: &Path, var: impl Fn(&str) -> Option<String>) -> Environment {
    let flags = match var("CARGO_ENCODED_RUSTFLAGS") {
        Some(encoded) if !encoded.is_empty() => (
            "CARGO_ENCODED_RUSTFLAGS",
            format!("{encoded}\u{1f}--cfg\u{1f}loom"),
        ),
        _ => match var("RUSTFLAGS") {
            Some(flags) if !flags.trim().is_empty() => ("RUSTFLAGS", format!("{flags} --cfg loom")),
            _ => ("RUSTFLAGS", "--cfg loom".to_string()),
        },
    };
    let target = var("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from);
    Environment {
        flags,
        target_dir: target.join("loom"),
        preemptions: var("LOOM_MAX_PREEMPTIONS").unwrap_or_else(|| PREEMPTIONS.to_string()),
    }
}

/// The test paths in `stdout` that passed, from libtest's `test <path> ... ok` lines.
pub fn passed(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("test ")?.strip_suffix(" ... ok"))
        .map(str::trim)
        .collect()
}

/// The models of [`REQUIRED`] that `stdout` does not show passing.
pub fn missing(stdout: &str) -> Vec<&'static str> {
    let passed = passed(stdout);
    REQUIRED
        .iter()
        .copied()
        .filter(|model| !passed.contains(model))
        .collect()
}

/// The suite step: the two crates' unit tests under loom, then the check that the named models
/// passed.
pub fn suite_step(context: &Context) -> Result<(), String> {
    let environment = environment(&context.root, |name| std::env::var(name).ok());
    let mut command = Command::new(crate::registry::cargo());
    command.args(["test", "--locked", "--lib", "--color", "never"]);
    for package in PACKAGES {
        command.args(["--package", package]);
    }
    // Plain text, so the result lines read the same whatever CI sets.
    command
        .args(["--", "--color", "never"])
        .env(environment.flags.0, &environment.flags.1)
        .env("CARGO_TARGET_DIR", &environment.target_dir)
        .env("LOOM_MAX_PREEMPTIONS", &environment.preemptions)
        .env("CARGO_TERM_COLOR", "never")
        .current_dir(&context.root);
    eprintln!(
        "loom: {}={:?} CARGO_TARGET_DIR={} LOOM_MAX_PREEMPTIONS={} cargo test --lib -p {}",
        environment.flags.0,
        environment.flags.1,
        environment.target_dir.display(),
        environment.preemptions,
        PACKAGES.join(" -p ")
    );
    let output = command
        .output()
        .map_err(|e| format!("cannot start cargo: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    print!("{stdout}");
    if !output.status.success() {
        return Err(format!("the loom models failed ({})", output.status));
    }
    match missing(&stdout).as_slice() {
        [] => Ok(()),
        models => Err(format!(
            "these models did not pass, so loom may not have been on: {}",
            models.join(", ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars<'a>(set: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            set.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    #[test]
    fn a_bare_environment_gets_the_cfg_its_own_target_and_a_bound_of_three() {
        let root = Path::new("/w");
        assert_eq!(
            environment(root, vars(&[])),
            Environment {
                flags: ("RUSTFLAGS", "--cfg loom".to_string()),
                target_dir: PathBuf::from("/w/target/loom"),
                preemptions: "3".to_string(),
            }
        );
    }

    #[test]
    fn the_callers_flags_target_and_bound_are_kept() {
        let set = [
            ("RUSTFLAGS", "-D warnings"),
            ("CARGO_TARGET_DIR", "/cache/t"),
            ("LOOM_MAX_PREEMPTIONS", "2"),
        ];
        let environment = environment(Path::new("/w"), vars(&set));
        assert_eq!(
            environment.flags,
            ("RUSTFLAGS", "-D warnings --cfg loom".to_string())
        );
        assert_eq!(environment.target_dir, PathBuf::from("/cache/t/loom"));
        assert_eq!(environment.preemptions, "2");
    }

    #[test]
    fn encoded_flags_win_as_they_do_for_cargo() {
        let set = [
            ("CARGO_ENCODED_RUSTFLAGS", "-D\u{1f}warnings"),
            ("RUSTFLAGS", "-O"),
        ];
        assert_eq!(
            environment(Path::new("/w"), vars(&set)).flags,
            (
                "CARGO_ENCODED_RUSTFLAGS",
                "-D\u{1f}warnings\u{1f}--cfg\u{1f}loom".to_string()
            )
        );
    }

    #[test]
    fn a_run_without_the_models_is_caught() {
        let all: String = REQUIRED
            .iter()
            .map(|model| format!("test {model} ... ok\n"))
            .collect();
        assert!(missing(&all).is_empty());
        let without_loom = "test ring::tests::values_cross_threads_in_order ... ok\n\
                            test ring::tests::values_cross_a_three_thread_pipeline_in_order ... ok\n";
        assert_eq!(missing(without_loom), REQUIRED[1..].to_vec());
        let ignored = format!("test {} ... ignored\n", REQUIRED[0]);
        assert!(missing(&ignored).contains(&REQUIRED[0]));
    }
}
