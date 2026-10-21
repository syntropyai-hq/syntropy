// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask build --profile <p>` run end to end on the fixture workspaces in
//! `tests/fixtures/` (QUAL-E3-T04).
//!
//! `profiles` carries the workspace's own `[profile.*]` sections, which a test keeps identical
//! to the root `Cargo.toml`'s. So a profile build of the fixture shows what the real profiles
//! build. `profiles-broken` breaks them: its `test-release` has no debug assertions, and its
//! release crate turns a test-only feature on. Each fixture is built into its own directory under
//! cargo's temporary target directory.

use std::path::Path;
use std::process::Command;

fn fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The exit status and the `build:` lines of `cargo xtask build --profile <profile>` on
/// `fixture`.
fn build(fixture: &str, profile: &str) -> (Option<i32>, Vec<String>) {
    let target = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("build-profiles")
        .join(fixture);
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["build", "--profile", profile, "--manifest-path"])
        .arg(fixtures().join(fixture).join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", target)
        .output()
        .expect("xtask runs");
    let log = String::from_utf8(output.stderr).expect("UTF-8 output");
    let lines = log
        .lines()
        .filter_map(|line| line.strip_prefix("build: "))
        .map(str::to_owned)
        .collect();
    (output.status.code(), lines)
}

/// The text of `manifest` from its first `[profile.` section to the end.
fn profile_sections(manifest: &Path) -> String {
    let text = std::fs::read_to_string(manifest).expect("read the manifest");
    let start = text
        .lines()
        .position(|line| line.starts_with("[profile."))
        .expect("a [profile.*] section");
    text.lines().skip(start).collect::<Vec<_>>().join("\n")
}

#[test]
fn the_fixture_profiles_are_the_workspace_profiles() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../Cargo.toml");
    let workspace = profile_sections(&root);
    for section in [
        "[profile.release]",
        "[profile.test-release]",
        "[profile.chaos]",
    ] {
        assert!(workspace.contains(section), "the workspace lacks {section}");
    }
    assert_eq!(
        profile_sections(&fixtures().join("profiles/Cargo.toml")),
        workspace,
        "copy the root Cargo.toml's [profile.*] sections into tests/fixtures/profiles/Cargo.toml"
    );
}

#[test]
fn each_profile_builds_with_its_test_features_and_debug_assertions() {
    let cases: [(&str, &[&str], &str); 4] = [
        (
            "dev",
            &["cargo build --locked --profile dev --workspace --features \
               fault-injection,debug-taps,guards"],
            "PASS: 5 units of workspace crates built in `dev`, debug assertions on in every one; \
             test features on: `exec/debug-taps`, `obs/debug-taps`, `obs/fault-injection` and \
             `obs/guards`",
        ),
        (
            "test-release",
            &[
                "cargo build --locked --profile test-release --workspace --features \
               fault-injection,debug-taps,guards",
            ],
            "PASS: 5 units of workspace crates built in `test-release`, debug assertions on in \
             every one; test features on: `exec/debug-taps`, `obs/debug-taps`, \
             `obs/fault-injection` and `obs/guards`",
        ),
        (
            "chaos",
            &[
                "cargo build --locked --profile chaos --workspace --exclude syntropy-testkit \
               --exclude syntropy-xtask --features fault-injection",
            ],
            "PASS: 3 units of workspace crates built in `chaos`, debug assertions off in every \
             one; test features on: `obs/fault-injection`",
        ),
        (
            "release",
            &["cargo build --locked --profile release --package syntropy-cli"],
            "PASS: 3 units of workspace crates built in `release`, debug assertions off in every \
             one; test features on: none",
        ),
    ];
    for (profile, commands, verdict) in cases {
        let (code, lines) = build("profiles", profile);
        assert_eq!(code, Some(0), "{profile}: {lines:#?}");
        let ran: Vec<&str> = lines
            .iter()
            .filter_map(|line| line.strip_prefix("cargo "))
            .collect();
        let expected: Vec<&str> = commands
            .iter()
            .map(|command| command.strip_prefix("cargo ").unwrap())
            .collect();
        assert_eq!(ran, expected, "{profile}: {lines:#?}");
        assert_eq!(lines.last().map(String::as_str), Some(verdict), "{profile}");
    }
}

#[test]
fn a_build_that_differs_from_its_profile_fails() {
    let (code, lines) = build("profiles-broken", "test-release");
    assert_eq!(code, Some(1), "{lines:#?}");
    assert!(
        lines.iter().any(|line| line
            == "`cli` (bin `syntropy`) was built with debug assertions off; the `test-release` \
                profile has them on"),
        "{lines:#?}"
    );
    assert_eq!(
        lines.last().map(String::as_str),
        Some("FAIL: the build differs from the `test-release` profile in 5 ways"),
        "{lines:#?}"
    );

    let (code, lines) = build("profiles-broken", "release");
    assert_eq!(code, Some(1), "{lines:#?}");
    assert_eq!(
        lines
            .iter()
            .filter(|line| !line.starts_with("profile ") && !line.starts_with("cargo "))
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "`obs` (lib `syntropy_obs`) was built with `guards`, which the `release` profile \
             does not turn on",
            "FAIL: the build differs from the `release` profile in 1 way",
        ]
    );
}
