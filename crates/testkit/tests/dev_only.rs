// SPDX-License-Identifier: MIT OR Apache-2.0

//! `SimEntropy` cannot be named from a non-dev dependency path (QUAL-E3-T02; RFC-0060 §2.1
//! rule T2, RFC-0001 §4.2 rule 9).
//!
//! Rule 9 lets a production crate take `testkit` as a dev-dependency only, and
//! `cargo xtask check-deps` fails any other edge to it (xtask's `rule-9` fixture). This test
//! checks what a dev-dependency still allows. It builds a throwaway crate, the probe, with
//! `syntropy-testkit` as its only dependency, a dev-dependency, and one library function that
//! names `SimEntropy`. The same source compiles when built for the crate's tests and fails when
//! built as the library a production build links.
//!
//! The probe lives under cargo's temporary target directory, with its own target directory and
//! a copy of the workspace's `Cargo.lock`, so it resolves offline to the versions the workspace
//! uses. Its first run compiles `testkit` and its dependencies once; later runs reuse that.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Output;

/// The probe's manifest. `TESTKIT` is replaced by the path of this crate.
const MANIFEST: &str = r#"[package]
name = "dev-only-probe"
version = "0.0.0"
edition = "2024"
publish = false

# A workspace of its own, not a member of the one it sits in.
[workspace]

[dev-dependencies]
syntropy-testkit = { path = "TESTKIT" }
"#;

/// The probe's library: production code that names `SimEntropy`.
const LIB: &str = "\
pub fn production_entropy() -> syntropy_testkit::sim::SimEntropy {
    syntropy_testkit::sim::SimEntropy::new(1)
}
";

/// Writes the probe and returns its directory.
fn probe() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dev-only-probe");
    fs::create_dir_all(dir.join("src")).expect("create the probe");
    let testkit = Path::new(env!("CARGO_MANIFEST_DIR"));
    // A TOML basic string, so a Windows path's backslashes are escaped.
    let path = testkit.display().to_string().replace('\\', "\\\\");
    fs::write(dir.join("Cargo.toml"), MANIFEST.replace("TESTKIT", &path)).expect("write");
    fs::write(dir.join("src/lib.rs"), LIB).expect("write");
    fs::copy(testkit.join("../../Cargo.lock"), dir.join("Cargo.lock")).expect("copy the lock");
    dir
}

/// `cargo check` on the probe with `args`.
fn check(dir: &Path, args: &[&str]) -> Output {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    std::process::Command::new(cargo)
        .args(["check", "--offline", "--quiet", "--message-format=short"])
        .args(args)
        .current_dir(dir)
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .output()
        .expect("cargo runs")
}

#[test]
fn sim_entropy_is_named_only_from_test_builds() {
    let dir = probe();

    // Built for its unit tests, the library sees its dev-dependencies, so the name resolves.
    // This also shows the path `syntropy_testkit::sim::SimEntropy` is right. (`--tests` alone:
    // adding `--lib` would check the plain library too.)
    let tests = check(&dir, &["--tests"]);
    let stderr = String::from_utf8_lossy(&tests.stderr);
    assert!(tests.status.success(), "the test build failed:\n{stderr}");

    // Built as a library, it does not, and the only errors are that the crate is not linked.
    let library = check(&dir, &["--lib"]);
    let stderr = String::from_utf8_lossy(&library.stderr);
    assert!(!library.status.success(), "the library built:\n{stderr}");
    let errors: Vec<&str> = stderr
        .lines()
        .filter(|line| line.contains("error"))
        .filter(|line| !line.starts_with("error: could not compile"))
        .collect();
    assert!(!errors.is_empty(), "no errors:\n{stderr}");
    for error in &errors {
        assert!(
            error.contains("error[E0433]") && error.contains("unlinked crate `syntropy_testkit`"),
            "an error other than the missing crate: {error}\n{stderr}"
        );
    }
}
