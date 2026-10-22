// SPDX-License-Identifier: MIT OR Apache-2.0

//! The property-test framework's two runtime promises, on a throwaway crate (QUAL-E3-T05;
//! RFC-0060 §3.1).
//!
//! A named property runs on proptest's default configuration (`xtask/src/properties.rs`). The
//! fixture crate does the same, with the workspace's proptest version and features, resolved
//! offline from a copy of the workspace's `Cargo.lock`. Two of its tests show what the framework
//! relies on:
//!
//! - a planted failing property shrinks to its minimal case, and its seed is written to the
//!   crate's `proptest-regressions/` file. A second run replays that seed instead of saving a new
//!   one;
//! - the default configuration's case count is `PROPTEST_CASES`, and 256 when it is unset.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const MANIFEST: &str = r#"[package]
name = "planted"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dev-dependencies]
proptest = { version = "1.12.0", default-features = false, features = ["std"] }
"#;

const LIB: &str = r#"#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest::test_runner::{Config, TestRunner};

    proptest! {
        /// Planted: fails for every `x` from 1000 up, so the minimal failing case is 1000.
        #[test]
        fn planted_failure(x in 0u32..1_000_000) {
            prop_assert!(x < 1000);
        }
    }

    /// Prints how many cases the default configuration runs.
    #[test]
    fn default_case_count() {
        let mut runner = TestRunner::new(Config::default());
        let count = std::cell::Cell::new(0u32);
        runner
            .run(&any::<u8>(), |_| {
                count.set(count.get() + 1);
                Ok(())
            })
            .unwrap();
        println!("cases run: {}", count.get());
    }
}
"#;

/// Writes the fixture into its own directory for `name`, so tests that run at once do not share
/// one.
fn fixture(name: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("properties")
        .join(name);
    fs::create_dir_all(root.join("src")).expect("create the fixture");
    fs::write(root.join("Cargo.toml"), MANIFEST).expect("write");
    fs::write(root.join("src/lib.rs"), LIB).expect("write");
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    fs::copy(workspace.join("Cargo.lock"), root.join("Cargo.lock")).expect("copy the lock");
    // Every run starts with no saved seeds.
    let _ = fs::remove_dir_all(root.join("proptest-regressions"));
    root
}

/// `cargo test --lib -- <filter>` in the fixture, with `PROPTEST_CASES` set to `cases` or unset.
fn test(root: &Path, filter: &str, cases: Option<&str>) -> (Output, String) {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(cargo);
    command
        .args([
            "test",
            "--offline",
            "--lib",
            "--color",
            "never",
            "--",
            filter,
        ])
        .args(["--exact", "--nocapture", "--color", "never"])
        .current_dir(root)
        .env("CARGO_TARGET_DIR", root.join("target"))
        // Plain text whatever CI sets (`setup-rust` sets `CARGO_TERM_COLOR=always`).
        .env("CARGO_TERM_COLOR", "never");
    match cases {
        Some(cases) => command.env("PROPTEST_CASES", cases),
        None => command.env_remove("PROPTEST_CASES"),
    };
    let output = command.output().expect("cargo runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output, text)
}

/// The `cc` lines of the fixture's regression file.
fn seeds(root: &Path) -> Vec<String> {
    let file = root.join("proptest-regressions/lib.txt");
    let text = fs::read_to_string(&file)
        .unwrap_or_else(|e| panic!("no regression file at {}: {e}", file.display()));
    text.lines()
        .filter(|line| line.starts_with("cc "))
        .map(str::to_owned)
        .collect()
}

#[test]
fn a_failing_case_shrinks_and_its_seed_is_saved_and_replayed() {
    let root = fixture("planted");

    let (first, log) = test(&root, "tests::planted_failure", Some("256"));
    assert!(
        !first.status.success(),
        "the planted property passed:\n{log}"
    );
    assert!(
        log.contains("minimal failing input: x = 1000"),
        "it did not shrink to x = 1000:\n{log}"
    );
    let saved = seeds(&root);
    assert_eq!(saved.len(), 1, "{saved:?}");
    let (seed, comment) = saved[0].split_once(" # ").expect("a seed and a comment");
    let hex = seed.strip_prefix("cc ").expect("`cc <seed>`");
    assert!(
        hex.len() >= 32 && hex.chars().all(|c| c.is_ascii_hexdigit()),
        "{seed}"
    );
    assert_eq!(comment, "shrinks to x = 1000");

    // The second run replays the saved seed first, fails on it, and saves nothing new.
    let (second, log) = test(&root, "tests::planted_failure", Some("256"));
    assert!(!second.status.success(), "{log}");
    assert!(log.contains("minimal failing input: x = 1000"), "{log}");
    assert_eq!(
        seeds(&root),
        saved,
        "the saved seed was not the one that failed"
    );
}

#[test]
fn the_environment_sets_the_case_count() {
    let root = fixture("cases");
    for (cases, expected) in [(Some("37"), "37"), (Some("16384"), "16384"), (None, "256")] {
        let (output, log) = test(&root, "tests::default_case_count", cases);
        assert!(output.status.success(), "{log}");
        assert!(
            log.contains(&format!("cases run: {expected}\n")),
            "PROPTEST_CASES={cases:?} did not run {expected} cases:\n{log}"
        );
    }
}

/// A registry holding `ids`, or none.
fn checkout(name: &str, ids: Option<&[&str]>) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("properties-removal")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("ci")).expect("create the checkout");
    if let Some(ids) = ids {
        let tables: String = ids
            .iter()
            .map(|id| {
                format!(
                    "[[property]]\nid = \"{id}\"\ncrate = \"core\"\nowner = \"QUAL-E3-T05\"\n\n"
                )
            })
            .collect();
        fs::write(dir.join("ci/properties.toml"), tables).expect("write the registry");
    }
    dir
}

/// `cargo xtask ci properties-removal` on `base` and `head`, with a PR whose description is `body`.
fn removal(base: &Path, head: &Path, body: &str) -> (Option<i32>, String) {
    let event = base.with_extension("event.json");
    let payload = serde_json::json!({
        "pull_request": { "title": "test: x", "body": body, "labels": [], "commits": 1 }
    });
    fs::write(&event, payload.to_string()).expect("write the event");
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["ci", "properties-removal", "--base"])
        .arg(base)
        .arg("--head")
        .arg(head)
        .arg("--event")
        .arg(&event)
        .output()
        .expect("xtask runs");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.code(), log)
}

/// The `pr-meta` `properties-removal` check (RFC-0060 §3.2): a removal needs a reason in the PR
/// description, and nothing else does.
#[test]
fn removing_a_property_needs_a_reason_in_the_pr_description() {
    let base = checkout("base", Some(&["P-core-1", "P-bench-1"]));
    let head = checkout("head", Some(&["P-bench-1"]));

    let (code, log) = removal(&base, &head, "Drops a property.");
    assert_eq!(code, Some(1), "{log}");
    assert!(
        log.contains(
            "P-core-1 was removed, and the PR description has no line `Removes P-core-1: <reason>`"
        ),
        "{log}"
    );

    let (code, log) = removal(
        &base,
        &head,
        "Drops a property.\n\nRemoves P-core-1: it moved\n",
    );
    assert_eq!(code, Some(0), "{log}");
    assert!(
        log.contains("removed with a reason: [\"P-core-1\"]"),
        "{log}"
    );

    // A PR that deletes the registry removes every property.
    let gone = checkout("gone", None);
    let (code, log) = removal(&base, &gone, "Removes P-core-1: moved\n");
    assert_eq!(code, Some(1), "{log}");
    assert!(log.contains("P-bench-1 was removed"), "{log}");

    // A base with no registry yet, as before this task, removes nothing.
    let none = checkout("none", None);
    let (code, log) = removal(&none, &head, "");
    assert_eq!(code, Some(0), "{log}");
    assert!(log.contains("no property was removed"), "{log}");
}
