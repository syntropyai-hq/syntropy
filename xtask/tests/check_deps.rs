// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask check-deps` run end to end on the fixture workspaces in `tests/fixtures/`
//! (QUAL-E1-T03).
//!
//! `pass` keeps every rule of RFC-0001 §4.2 while using the patterns the rules allow. Each
//! `rule-N` breaks rule N only, and must fail with exactly that rule's findings. The fixtures
//! depend on stand-ins in `fixtures/externals/` for external crates, and carry their own
//! `Cargo.lock` and `ci/layers.toml`, so they resolve offline. A fixture with a core binary is
//! built into its own directory under cargo's temporary target directory. Rule 7 reads the core
//! binary of the host, so its fixtures are per OS: `rule-7`, `rule-7-cufft` and `rule-7-nccl` run
//! on Linux, `rule-7-windows-*` on Windows and `rule-7-macos-*` on macOS (QUAL-E1-T28).

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
    // Rule 7 runs on every OS (QUAL-E1-T28). Rule 10's symbol-table half reads ELF, so it runs on
    // Linux only.
    let expected = if cfg!(target_os = "linux") {
        "PASS: the layer table and rules 1-10 of RFC-0001 §4.2"
    } else {
        "PASS: the layer table and rules 1-10 of RFC-0001 §4.2, except rule 10"
    };
    assert_eq!(verdict.map(String::as_str), Some(expected), "{lines:#?}");
    assert!(
        !lines
            .iter()
            .any(|line| line.starts_with("rule 7 not run: ")),
        "{lines:#?}"
    );
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

/// On Windows, a DLL in the import directory that is not a system DLL fails rule 7.
#[cfg(windows)]
#[test]
fn rule_7_on_windows_an_imported_vendor_dll_fails() {
    fails_with(
        "rule-7-windows-import",
        7,
        &[
            "imports `nvcuda.dll` from the CUDA runtime",
            "CUDA runtime symbols in its import name tables: cuInit;",
        ],
    );
}

/// On Windows, a DLL in the delay-load directory counts as an import does.
#[cfg(windows)]
#[test]
fn rule_7_on_windows_a_delay_loaded_vendor_dll_fails() {
    fails_with(
        "rule-7-windows-delay",
        7,
        &[
            "delay-loads `vulkan-1.dll` from the Vulkan runtime",
            "Vulkan runtime symbols in its import name tables: vkCreateInstance;",
        ],
    );
}

/// On macOS, a library outside the system ones fails rule 7.
#[cfg(target_os = "macos")]
#[test]
fn rule_7_on_macos_a_vendor_dylib_fails() {
    fails_with(
        "rule-7-macos-vulkan",
        7,
        &[
            "loads `@rpath/libvulkan.1.dylib` from the Vulkan runtime",
            "Vulkan runtime symbols in its undefined symbols: vkCreateInstance;",
        ],
    );
}

/// On macOS, the Metal provider is linked into the core, so `Metal.framework` and the system
/// frameworks beneath its binding pass rule 7.
#[cfg(target_os = "macos")]
#[test]
fn rule_7_on_macos_metal_and_system_frameworks_pass() {
    let (code, lines) = check("rule-7-macos-metal");
    assert_eq!(code, Some(0), "{lines:#?}");
    assert_eq!(findings(&lines), Vec::<&str>::new());
    assert!(
        !lines
            .iter()
            .any(|line| line.starts_with("rule 7 not run: ")),
        "{lines:#?}"
    );
}

/// Each fixture that builds for one OS only names that OS in its manifest's first line, and has
/// a test above that runs it there.
#[test]
fn each_os_fixture_of_rule_7_names_its_os() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for (fixture, os) in [
        ("rule-7-windows-import", "Windows"),
        ("rule-7-windows-delay", "Windows"),
        ("rule-7-macos-vulkan", "macOS"),
        ("rule-7-macos-metal", "macOS"),
    ] {
        let manifest = std::fs::read_to_string(fixtures.join(fixture).join("Cargo.toml")).unwrap();
        let first = manifest.lines().next().unwrap_or_default();
        assert!(first.contains(&format!("on {os},")), "{fixture}: {first}");
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

// `testkit` denies `unsafe` rather than forbidding it, so an `allow` in a file the scope check
// does not read would turn it back on. Each way to compile such a file is refused.

#[test]
fn rule_8_a_path_module_outside_src_is_refused() {
    fails_with(
        "rule-8-path",
        8,
        &["`testkit`: crates/testkit/src/lib.rs:8: sets a module `path` in an attribute"],
    );
}

#[test]
fn rule_8_include_of_a_file_that_is_not_rs_is_refused() {
    fails_with(
        "rule-8-include",
        8,
        &["`testkit`: crates/testkit/src/lib.rs:9: names `include`"],
    );
}

#[test]
fn rule_8_a_build_script_and_its_generated_code_are_refused() {
    fails_with(
        "rule-8-build-script",
        8,
        &[
            "`testkit` has a build script, crates/testkit/build.rs",
            "`testkit`: crates/testkit/src/lib.rs:9: names `include`",
        ],
    );
}

#[test]
fn rule_8_a_local_macro_is_refused() {
    fails_with(
        "rule-8-macro",
        8,
        &["`testkit`: crates/testkit/src/lib.rs:9: defines a declarative macro"],
    );
}

#[test]
fn rule_8_a_bin_target_outside_src_is_read() {
    fails_with(
        "rule-8-bin",
        8,
        &[
            "`testkit`: crates/testkit/tools/helper.rs:5: sets a lint level for `unsafe_code`",
            "`testkit`: crates/testkit/tools/helper.rs:8: uses `unsafe` outside the one allowed impl",
        ],
    );
}

#[test]
fn rule_8_a_file_rustc_reads_from_its_second_line_is_refused() {
    fails_with(
        "rule-8-shebang",
        8,
        &["`testkit`: crates/testkit/src/sneaky.rs:1: starts with `#!`"],
    );
}

#[test]
fn rule_8_a_leading_allow_after_the_deny_is_refused() {
    fails_with(
        "rule-8-root-allow",
        8,
        &[
            "`testkit`: crates/testkit/src/lib.rs:6: sets `unsafe_code` in the crate root's leading attributes",
        ],
    );
}

#[test]
fn rule_8_a_file_whose_extension_is_rs_in_another_case_is_read() {
    fails_with(
        "rule-8-upper-ext",
        8,
        &[
            "`testkit`: crates/testkit/src/upper.RS:1: has an extension that is `rs` only in another case",
            "`testkit`: crates/testkit/src/upper.RS:5: sets a lint level for `unsafe_code`",
            "`testkit`: crates/testkit/src/upper.RS:8: uses `unsafe` outside the one allowed impl",
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
    // is the `obs/fault-injection` it reaches through `obs`'s own `extra` (#1556). On Linux, the
    // core binary's symbol table holds the fault point `cli` calls (QUAL-E9-T01).
    let mut expected = vec![
        "the release build of `cli` enables `obs/fault-injection`",
        "the release build of `kernels-vulkan` enables `obs/debug-taps`",
        "`testkit` turns on `obs/guards` by default (default → helpers → syntropy-obs/guards)",
        "`testkit` turns on `obs/fault-injection` by default (default → helpers → \
         syntropy-obs/extra → fault-injection)",
    ];
    if cfg!(target_os = "linux") {
        expected.push(
            "the core binary, built with its release features, holds 1 symbol of \
             `syntropy_obs::faults::inject`, which only `fault-injection` compiles: ",
        );
    }
    fails_with("rule-10", 10, &expected);
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
