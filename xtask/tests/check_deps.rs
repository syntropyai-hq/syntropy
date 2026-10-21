// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask check-deps` run end to end on the fixture workspaces in `tests/fixtures/`
//! (QUAL-E1-T03).
//!
//! `pass` keeps every rule of RFC-0001 §4.2 while using the patterns the rules allow. Each
//! `rule-N` breaks rule N only, and must fail with exactly that rule's findings. The fixtures
//! depend on stand-ins in `fixtures/externals/` for external crates, and carry their own
//! `Cargo.lock` and `ci/layers.toml`, so they resolve offline. A fixture with a core binary is
//! built into its own directory under cargo's temporary target directory.

use std::path::Path;
use std::process::Command;

/// The exit status and the `check-deps:` lines of a run on `fixture`.
fn check(fixture: &str) -> (Option<i32>, Vec<String>) {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let target = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("check-deps")
        .join(fixture);
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .arg("check-deps")
        .arg("--manifest-path")
        .arg(fixtures.join(fixture).join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", target)
        .output()
        .expect("xtask runs");
    let log = String::from_utf8(output.stderr).expect("UTF-8 output");
    let lines = log
        .lines()
        .filter_map(|line| line.strip_prefix("check-deps: "))
        .map(str::to_owned)
        .collect();
    (output.status.code(), lines)
}

/// The finding lines: those that start with a rule label, without the "not run" notes.
fn findings(lines: &[String]) -> Vec<&str> {
    lines
        .iter()
        .map(String::as_str)
        .filter(|line| {
            let finding = line.starts_with("rule ") || line.starts_with("layer table: ");
            finding && !line.contains(" not run: ")
        })
        .collect()
}

/// Runs `fixture` and checks that it fails with findings of `rule` only, one containing each of
/// `expected`, and no others.
fn fails_with(fixture: &str, rule: u8, expected: &[&str]) {
    let (code, lines) = check(fixture);
    let found = findings(&lines);
    let label = format!("rule {rule}: ");
    assert_eq!(code, Some(1), "{fixture}: {lines:#?}");
    assert!(
        found.iter().all(|line| line.starts_with(&label)),
        "{fixture} broke another rule: {found:#?}"
    );
    assert_eq!(found.len(), expected.len(), "{fixture}: {found:#?}");
    for text in expected {
        assert!(
            found.iter().any(|line| line.contains(text)),
            "{fixture}: no finding contains `{text}`: {found:#?}"
        );
    }
    let verdict = format!("({})", label.trim_end_matches(": "));
    assert!(
        lines
            .iter()
            .any(|line| line.starts_with("FAIL: ") && line.ends_with(&verdict)),
        "{fixture}: {lines:#?}"
    );
}

#[test]
fn the_pass_fixture_passes() {
    let (code, lines) = check("pass");
    assert_eq!(code, Some(0), "{lines:#?}");
    assert_eq!(findings(&lines), Vec::<&str>::new());
    let verdict = lines.iter().find(|line| line.starts_with("PASS: "));
    if cfg!(target_os = "linux") {
        assert_eq!(
            verdict.map(String::as_str),
            Some("PASS: the layer table and rules 1-10 of RFC-0001 §4.2"),
            "{lines:#?}"
        );
    } else {
        assert!(verdict.is_some(), "{lines:#?}");
    }
}

#[test]
fn rule_1_core_depends_on_a_workspace_crate_and_a_foreign_external() {
    fails_with(
        "rule-1",
        1,
        &[
            "`core` has a normal dependency on the workspace crate `obs`",
            "`core` has a normal dependency on `tokio`",
        ],
    );
}

#[test]
fn rule_2_edges_must_go_to_lower_layers() {
    fails_with(
        "rule-2",
        2,
        &[
            "`manifest` (layer 1) has a normal dependency on `solver` (layer 2)",
            "`kernels` (layer 1) has a dev dependency on `kernels-cpu` (layer 2)",
            "`xtask` is a tool outside the crate graph",
        ],
    );
}

#[test]
fn rule_3_compute_crates_reach_no_network_crate_or_tokio() {
    fails_with(
        "rule-3",
        3,
        &[
            "exec → net;",
            "io → tokio;",
            "kernels-cpu → tokio-shim → tokio;",
        ],
    );
}

#[test]
fn rule_3_cites_adr_0012_for_tokio_in_io_and_providers() {
    let (_, lines) = check("rule-3");
    let adr: Vec<&str> = findings(&lines)
        .into_iter()
        .filter(|line| line.ends_with("(and ADR-0012 §6)"))
        .collect();
    assert_eq!(adr.len(), 2, "{adr:#?}");
}

#[test]
fn rule_4_the_daemon_links_no_provider() {
    fails_with(
        "rule-4",
        4,
        &["daemon → kernels-loader;", "daemon → exec → kernels-cpu;"],
    );
}

#[test]
fn rule_5_the_solver_reaches_no_io() {
    fails_with("rule-5", 5, &["solver → net;", "solver → tokio;"]);
}

#[test]
fn rule_6_each_external_system_has_one_crate() {
    fails_with(
        "rule-6",
        6,
        &[
            "`store` reaches `iroh`, a binding of iroh: store → iroh; only `net` uses iroh",
            "`kernels-cpu` declares `links = \"ggml\"`",
        ],
    );
}

#[cfg(target_os = "linux")]
#[test]
fn rule_7_the_core_binary_links_no_vendor_runtime() {
    fails_with(
        "rule-7",
        7,
        &[
            "needs `libcuda.so` from the CUDA runtime",
            "CUDA runtime symbols in its dynamic symbol table: cuInit;",
        ],
    );
}

/// The two libraries the review of the first version of rule 7 linked past its denylist. The
/// allow-list rejects them however they are named.
#[cfg(target_os = "linux")]
#[test]
fn rule_7_rejects_vendor_libraries_by_allow_list() {
    fails_with(
        "rule-7-cufft",
        7,
        &[
            "needs `libcufft.so` from the CUDA runtime",
            "CUDA runtime symbols in its dynamic symbol table: cufftPlan1d;",
        ],
    );
    fails_with(
        "rule-7-nccl",
        7,
        &[
            "needs `libnccl.so` from the CUDA runtime",
            "CUDA runtime symbols in its dynamic symbol table: ncclCommInitRank;",
        ],
    );
}

#[cfg(not(target_os = "linux"))]
#[test]
fn rule_7_is_reported_as_not_run_off_linux() {
    for fixture in ["rule-7", "rule-7-cufft", "rule-7-nccl"] {
        let (code, lines) = check(fixture);
        assert_eq!(code, Some(0), "{fixture}: {lines:#?}");
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("rule 7 not run: ")),
            "{fixture}: {lines:#?}"
        );
    }
}

#[test]
fn rule_8_crates_outside_the_allow_list_forbid_unsafe_code() {
    fails_with(
        "rule-8",
        8,
        &[
            "`graph` does not set",
            "`solver` does not set",
            "`text` does not set",
            "`config` does not set",
            "crates/bench/src/bin/run.rs (bin target `run`)",
            // `testkit` may hold `unsafe` only in its one allowed `GlobalAlloc` impl.
            "`testkit`: crates/testkit/src/other.rs:6: uses `unsafe` outside the one allowed impl",
        ],
    );
}

#[test]
fn rule_9_dev_only_crates_are_only_dev_dependencies() {
    fails_with(
        "rule-9",
        9,
        &[
            "`runner` has a normal dependency on `testkit`",
            "`store` has a build dependency on `bench`",
        ],
    );
}

#[test]
fn rule_10_release_builds_carry_no_test_features() {
    // `testkit` turns on `obs/guards`, but only as a dev-dependency of `cli`, which no release
    // build includes. Its default feature turning on `obs/guards` is a finding all the same, and so
    // is the `obs/fault-injection` it reaches through `obs`'s own `extra` (#1556).
    fails_with(
        "rule-10",
        10,
        &[
            "the release build of `cli` enables `obs/fault-injection`",
            "the release build of `kernels-vulkan` enables `obs/debug-taps`",
            "`testkit` turns on `obs/guards` by default (default → helpers → syntropy-obs/guards)",
            "`testkit` turns on `obs/fault-injection` by default (default → helpers → \
             syntropy-obs/extra → fault-injection)",
        ],
    );
}

#[test]
fn every_rule_has_a_fixture() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for rule in 1..=10 {
        let manifest = fixtures.join(format!("rule-{rule}")).join("Cargo.toml");
        assert!(manifest.is_file(), "{}", manifest.display());
    }
}

#[test]
fn a_missing_manifest_fails() {
    let (code, lines) = check("no-such-fixture");
    assert_eq!(code, Some(1), "{lines:#?}");
    assert!(
        lines
            .iter()
            .any(|line| line.starts_with("FAIL: `cargo metadata` failed")),
        "{lines:#?}"
    );
}
