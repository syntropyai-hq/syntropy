// SPDX-License-Identifier: MIT OR Apache-2.0

//! The licence checks of QUAL-E1-T04 run end to end: `cargo xtask deny`, `spdx`, `third-party`
//! and `notices`, each the command a `lint` step runs, pointed at a fixture.
//!
//! The cargo-deny fixtures in `tests/fixtures/licences/` are workspaces whose `.cargo/config.toml`
//! serves crates.io, another registry and a git repository from stand-in directories, so they
//! resolve offline and cargo-deny sees each stand-in as coming from that source. They are checked
//! against this checkout's `deny.toml`, the policy under test. The SPDX and vendored-file
//! fixtures are built in a scratch directory by each test, so no file in the repository breaks
//! the checks the repository's own `lint` runs.
//!
//! The cargo-deny tests run on Linux, where `lint` runs, and need cargo-deny installed. Locally
//! they pass with a notice on stderr when it is missing; in CI, where `CI` is set, a missing
//! cargo-deny fails them, so they never pass there without running.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/licences")
}

fn cargo() -> PathBuf {
    std::env::var_os("CARGO").map_or_else(|| PathBuf::from("cargo"), PathBuf::from)
}

/// Runs this crate's xtask binary with `args` and returns its exit status and output: stderr,
/// where xtask writes, then stdout, where cargo-deny prints its summary line.
fn xtask(args: &[&str]) -> (Option<i32>, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .output()
        .expect("xtask runs");
    let mut log = String::from_utf8(output.stderr).expect("UTF-8 output");
    log.push_str(&String::from_utf8(output.stdout).expect("UTF-8 output"));
    (output.status.code(), log)
}

/// An empty directory for one test, under cargo's temporary target directory.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("licences")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes `text` to `rel` under `root`, creating directories.
fn write(root: &Path, rel: &str, text: &str) {
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// Copies the tree at `from` into `to`.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            if entry.file_name() != "target" {
                copy_tree(&entry.path(), &target);
            }
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// The lines of `log` that start with `prefix`, without it.
fn lines<'a>(log: &'a str, prefix: &str) -> Vec<&'a str> {
    log.lines().filter_map(|l| l.strip_prefix(prefix)).collect()
}

// ---------------------------------------------------------------------------------------------
// cargo-deny

/// Whether the cargo-deny tests can run here. See the module docs.
#[cfg(target_os = "linux")]
fn cargo_deny_installed() -> bool {
    let installed = Command::new(cargo())
        .args(["deny", "--version"])
        .output()
        .is_ok_and(|output| output.status.success());
    let in_ci = std::env::var_os("CI").is_some_and(|v| !v.is_empty() && v != "false");
    assert!(
        installed || !in_ci,
        "cargo-deny is not installed, and CI must run these tests; install the version \
         `cargo xtask deny` pins"
    );
    if !installed {
        eprintln!("skipped: cargo-deny is not installed, so the cargo-deny fixtures did not run");
    }
    installed
}

/// Runs `cargo xtask deny` on the fixture workspace `name`.
#[cfg(target_os = "linux")]
fn deny(name: &str) -> (Option<i32>, String) {
    let manifest = fixtures().join(name).join("Cargo.toml");
    xtask(&["deny", "--manifest-path", manifest.to_str().unwrap()])
}

/// Runs the fixture `name` and checks that it fails the deny step with each of `expected` in the
/// log.
#[cfg(target_os = "linux")]
fn deny_fails(name: &str, expected: &[&str]) {
    if !cargo_deny_installed() {
        return;
    }
    let (code, log) = deny(name);
    assert_eq!(code, Some(1), "{name}: {log}");
    assert!(
        lines(&log, "deny: FAIL: licenses, bans, sources").len() == 1,
        "{name}: {log}"
    );
    for text in expected {
        assert!(log.contains(text), "{name}: no `{text}` in {log}");
    }
}

#[cfg(target_os = "linux")]
#[test]
fn a_scratch_crate_depending_on_a_gpl_crate_fails_the_deny_step() {
    deny_fails(
        "deny-gpl",
        &[
            "error[rejected]: failed to satisfy license requirements",
            "license = \"GPL-3.0-only\"",
            "bans ok, licenses FAILED, sources ok",
        ],
    );
}

#[cfg(target_os = "linux")]
#[test]
fn openssl_sys_is_banned() {
    deny_fails(
        "deny-openssl",
        &[
            "error[banned]: crate 'openssl-sys = 0.9.0' is explicitly banned",
            "bans FAILED, licenses ok, sources ok",
        ],
    );
}

#[cfg(target_os = "linux")]
#[test]
fn a_crate_from_a_registry_deny_toml_does_not_allow_fails() {
    deny_fails(
        "deny-registry",
        &[
            "error[source-not-allowed]: detected 'registry' source not explicitly allowed",
            "elsewhere-crate 1.0.0 sparse+https://registry.example.invalid/index/",
            "bans ok, licenses ok, sources FAILED",
        ],
    );
}

#[cfg(target_os = "linux")]
#[test]
fn a_crate_from_a_git_repository_deny_toml_does_not_allow_fails() {
    deny_fails(
        "deny-git",
        &[
            "error[source-not-allowed]: detected 'git' source not explicitly allowed",
            "bans ok, licenses ok, sources FAILED",
        ],
    );
}

#[cfg(target_os = "linux")]
#[test]
fn an_mpl_crate_without_a_per_crate_exception_fails() {
    deny_fails(
        "deny-mpl",
        &[
            "error[rejected]: failed to satisfy license requirements",
            "license = \"MPL-2.0\"",
            "bans ok, licenses FAILED, sources ok",
        ],
    );
}

#[cfg(target_os = "linux")]
#[test]
fn a_cdla_permissive_crate_passes_the_deny_step() {
    if !cargo_deny_installed() {
        return;
    }
    let (code, log) = deny("deny-cdla");
    assert_eq!(code, Some(0), "{log}");
    assert!(log.contains("bans ok, licenses ok, sources ok"), "{log}");
    assert_eq!(
        lines(&log, "deny: PASS: "),
        ["licenses, bans, sources (ADR-0018 §3, RFC-0062 §11.2)"],
        "{log}"
    );
}

#[test]
fn every_deny_fixture_resolves_offline_from_its_lock_file() {
    for name in [
        "deny-gpl",
        "deny-openssl",
        "deny-registry",
        "deny-git",
        "deny-mpl",
        "deny-cdla",
        "notices",
    ] {
        let dir = fixtures().join(name);
        assert!(dir.join("Cargo.lock").is_file(), "{name}");
        assert!(dir.join(".cargo/config.toml").is_file(), "{name}");
        let output = Command::new(cargo())
            .args(["metadata", "--format-version", "1", "--locked", "--offline"])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

// ---------------------------------------------------------------------------------------------
// SPDX headers

const HEADER: &str = "SPDX-License-Identifier: MIT OR Apache-2.0";

#[test]
fn a_source_file_without_the_spdx_line_fails() {
    let root = scratch("spdx-fail");
    // One file per source type ADR-0018 §1 names, without the header.
    let bad = [
        "crates/core/src/lib.rs",
        "crates/kernels-cuda/build.rs",
        "crates/kernels-cuda/kernels/gemm.cu",
        "crates/kernels-cuda/kernels/tile.cuh",
        "crates/kernels-hip/kernels/gemm.hip",
        "crates/kernels-metal/shaders/gemm.metal",
        "crates/kernels-vulkan/shaders/gemm.slang",
        "crates/kernels-ggml/CMakeLists.txt",
        "packaging/install.sh",
    ];
    for rel in bad {
        write(&root, rel, "int x;\n");
    }
    // A header that names another licence, and one below the first line.
    write(
        &root,
        "crates/obs/src/lib.rs",
        "// SPDX-License-Identifier: MIT\n",
    );
    write(
        &root,
        "crates/obs/src/late.rs",
        &format!("//! Docs first.\n// {HEADER}\n"),
    );
    // Files that pass, or that the check skips.
    write(&root, "crates/obs/src/good.rs", &format!("// {HEADER}\n"));
    write(&root, "tools/run.sh", &format!("#!/bin/sh\n# {HEADER}\n"));
    write(&root, "target/debug/build/out.rs", "generated\n");
    write(&root, "third_party/ggml/src/ggml.c", "/* upstream */\n");
    write(
        &root,
        "crates/kernels-cuda/vendor/cute/tensor.hpp",
        "// upstream\n",
    );
    write(&root, "README.md", "no header needed\n");

    let (code, log) = xtask(&["spdx", "--root", root.to_str().unwrap()]);
    assert_eq!(code, Some(1), "{log}");
    let mut failed: Vec<&str> = lines(&log, "spdx: ")
        .into_iter()
        .filter(|line| !line.starts_with("FAIL: "))
        .map(|line| line.split_once(": ").unwrap().0)
        .collect();
    failed.sort();
    let mut expected: Vec<&str> = bad.to_vec();
    expected.extend(["crates/obs/src/late.rs", "crates/obs/src/lib.rs"]);
    expected.sort();
    assert_eq!(failed, expected, "{log}");
    assert!(
        log.contains("spdx: crates/obs/src/lib.rs: has `SPDX-License-Identifier: MIT`"),
        "{log}"
    );
    assert!(
        log.contains("spdx: FAIL: 11 of 13 source files lack"),
        "{log}"
    );
}

#[test]
fn source_files_with_the_spdx_line_pass() {
    let root = scratch("spdx-pass");
    for rel in [
        "crates/kernels-cuda/build.rs",
        "crates/kernels-cuda/kernels/gemm.cu",
        "crates/kernels-metal/shaders/gemm.metal",
        "crates/kernels-vulkan/shaders/gemm.slang",
    ] {
        write(&root, rel, &format!("// {HEADER}\n"));
    }
    write(
        &root,
        "crates/kernels-ggml/CMakeLists.txt",
        &format!("# {HEADER}\n"),
    );
    let (code, log) = xtask(&["spdx", "--root", root.to_str().unwrap()]);
    assert_eq!(code, Some(0), "{log}");
    assert!(
        log.contains("spdx: PASS: 5 source files start with"),
        "{log}"
    );
}

// ---------------------------------------------------------------------------------------------
// third_party/licences.toml

/// A vendored tree: two files and a licence under `third_party/tiny/`, a header and its licence
/// under a provider's `vendor/`, and a lock file the manifest needs no entry for.
fn vendored_tree(name: &str) -> PathBuf {
    let root = scratch(name);
    write(
        &root,
        "third_party/tiny/tiny.c",
        "int tiny(void) { return 1; }\n",
    );
    write(&root, "third_party/tiny/tiny.h", "int tiny(void);\n");
    write(
        &root,
        "third_party/tiny/LICENSE",
        "Copyright (c) 2026 Tiny authors\nMIT terms.\n",
    );
    write(&root, "third_party/tiny.lock", "commit = \"abc\"\n");
    write(
        &root,
        "crates/kernels-cuda/vendor/cute/tensor.hpp",
        "// upstream header\n",
    );
    write(
        &root,
        "crates/kernels-cuda/vendor/cute/LICENSE.txt",
        "Copyright (c) 2026 Cute authors\nBSD-3-Clause terms.\n",
    );
    root
}

/// A `[[file]]` entry.
fn entry(path: &str, license: &str, notice: &str) -> String {
    format!(
        "[[file]]\npath = \"{path}\"\nupstream = \"https://example.org/upstream\"\n\
         version = \"1.2.3\"\nlicense = \"{license}\"\nnotice = \"{notice}\"\n\n"
    )
}

/// The manifest that lists every file of [`vendored_tree`] individually.
fn full_manifest() -> String {
    let tiny = "third_party/tiny/LICENSE";
    let cute = "crates/kernels-cuda/vendor/cute/LICENSE.txt";
    format!(
        "# SPDX-License-Identifier: MIT OR Apache-2.0\n\n{}{}{}{}{}",
        entry("third_party/tiny/tiny.c", "MIT", tiny),
        entry("third_party/tiny/tiny.h", "MIT", tiny),
        entry("third_party/tiny/LICENSE", "MIT", tiny),
        entry(
            "crates/kernels-cuda/vendor/cute/tensor.hpp",
            "BSD-3-Clause",
            cute
        ),
        entry(
            "crates/kernels-cuda/vendor/cute/LICENSE.txt",
            "BSD-3-Clause",
            cute
        ),
    )
}

fn third_party(root: &Path) -> (Option<i32>, String) {
    xtask(&["third-party", "--root", root.to_str().unwrap()])
}

#[test]
fn a_manifest_listing_every_vendored_file_passes() {
    let root = vendored_tree("third-party-pass");
    write(&root, "third_party/licences.toml", &full_manifest());
    let (code, log) = third_party(&root);
    assert_eq!(code, Some(0), "{log}");
    assert!(log.contains("third-party: PASS: 5 vendored files"), "{log}");
}

#[test]
fn a_file_under_third_party_without_an_entry_fails() {
    let root = vendored_tree("third-party-missing");
    write(&root, "third_party/licences.toml", &full_manifest());
    write(&root, "third_party/tiny/extra.c", "int extra;\n");
    write(&root, "crates/kernels-hip/vendor/rocm/x.h", "int x;\n");
    let (code, log) = third_party(&root);
    assert_eq!(code, Some(1), "{log}");
    assert_eq!(
        lines(&log, "third-party: ")
            .into_iter()
            .filter(|l| !l.starts_with("FAIL: "))
            .collect::<Vec<_>>(),
        [
            "crates/kernels-hip/vendor/rocm/x.h: not listed in third_party/licences.toml \
             (RFC-0062 §12.2)",
            "third_party/tiny/extra.c: not listed in third_party/licences.toml (RFC-0062 §12.2)",
        ],
        "{log}"
    );
}

#[test]
fn a_vendored_tree_listed_as_one_directory_entry_fails() {
    let root = vendored_tree("third-party-directory");
    let manifest = format!(
        "{}{}{}",
        entry("third_party/tiny", "MIT", "third_party/tiny/LICENSE"),
        entry(
            "crates/kernels-cuda/vendor/cute/tensor.hpp",
            "BSD-3-Clause",
            "crates/kernels-cuda/vendor/cute/LICENSE.txt"
        ),
        entry(
            "crates/kernels-cuda/vendor/cute/LICENSE.txt",
            "BSD-3-Clause",
            "crates/kernels-cuda/vendor/cute/LICENSE.txt"
        ),
    );
    write(&root, "third_party/licences.toml", &manifest);
    let (code, log) = third_party(&root);
    assert_eq!(code, Some(1), "{log}");
    assert!(
        log.contains(
            "third-party: third_party/licences.toml:1: `third_party/tiny` is a directory; list \
             each vendored file in it individually (ADR-0018 §3)"
        ),
        "{log}"
    );
    for file in ["LICENSE", "tiny.c", "tiny.h"] {
        assert!(
            log.contains(&format!(
                "third-party: third_party/tiny/{file}: not listed in third_party/licences.toml; \
                 `third_party/tiny` is listed as a directory, but each vendored file needs its \
                 own entry (ADR-0018 §3)"
            )),
            "{file}: {log}"
        );
    }
    assert!(log.contains("third-party: FAIL: 4 problems"), "{log}");
}

#[test]
fn an_entry_with_a_denied_licence_or_a_stale_path_fails() {
    let root = vendored_tree("third-party-licence");
    let manifest = full_manifest().replacen("license = \"MIT\"", "license = \"GPL-2.0-only\"", 1)
        + &entry("third_party/gone.c", "MIT", "third_party/tiny/LICENSE")
        + &entry(
            "third_party/tiny/tiny.c",
            "MIT",
            "third_party/tiny/NO-LICENSE",
        );
    write(&root, "third_party/licences.toml", &manifest);
    let (code, log) = third_party(&root);
    assert_eq!(code, Some(1), "{log}");
    for expected in [
        "`third_party/tiny/tiny.c` is under `GPL-2.0-only`, which the allow list in deny.toml \
         does not meet",
        "`third_party/gone.c` does not exist",
        "`third_party/tiny/tiny.c` is listed again; its entry is at line 3",
        "the notice file `third_party/tiny/NO-LICENSE` of `third_party/tiny/tiny.c` does not exist",
    ] {
        assert!(log.contains(expected), "{expected}: {log}");
    }
}

// ---------------------------------------------------------------------------------------------
// THIRD-PARTY-NOTICES

/// A copy of the fixtures in which the `notices` workspace has a vendored tree and a manifest.
fn notices_fixture(name: &str) -> PathBuf {
    let root = scratch(name);
    copy_tree(&fixtures(), &root);
    let ws = root.join("notices");
    write(
        &ws,
        "third_party/tiny/tiny.c",
        "int tiny(void) { return 1; }\n",
    );
    write(
        &ws,
        "third_party/tiny/LICENSE",
        "Copyright (c) 2026 Tiny authors\nMIT terms.\n",
    );
    let tiny = "third_party/tiny/LICENSE";
    write(
        &ws,
        "third_party/licences.toml",
        &format!(
            "{}{}",
            entry("third_party/tiny/tiny.c", "MIT", tiny),
            entry("third_party/tiny/LICENSE", "MIT", tiny),
        ),
    );
    ws
}

/// The third-party packages of `cli`'s release build, as `cargo tree` resolves them: normal and
/// build edges, every platform. An oracle independent of `cargo xtask notices`.
fn release_packages(ws: &Path) -> BTreeSet<String> {
    let output = Command::new(cargo())
        .args([
            "tree",
            "--locked",
            "--offline",
            "-e",
            "normal,build",
            "--target",
            "all",
        ])
        .args(["-p", "syntropy-cli", "--prefix", "none", "--format", "{p}"])
        .current_dir(ws)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            let name = words.next()?;
            let version = words.next()?.strip_prefix('v')?;
            (!name.starts_with("syntropy-")).then(|| format!("{name} {version}"))
        })
        .collect()
}

/// The `== title ==` section of `text`, up to the next section.
fn section<'a>(text: &'a str, title: &str) -> &'a str {
    let start = text
        .find(&format!("== {title}"))
        .unwrap_or_else(|| panic!("no section `{title}` in {text}"));
    let body = &text[start..];
    let body = &body[body.find('\n').unwrap() + 1..];
    match body.find("\n== ") {
        Some(end) => &body[..end],
        None => body,
    }
}

#[test]
fn the_notices_list_the_release_build_and_every_manifest_entry() {
    let ws = notices_fixture("notices-pass");
    let out = ws.join("THIRD-PARTY-NOTICES");
    let manifest = ws.join("Cargo.toml");
    let (code, log) = xtask(&[
        "notices",
        "--manifest-path",
        manifest.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "{log}");
    let text = std::fs::read_to_string(&out).unwrap();

    // Every package of the release build, and nothing else: not the dev-dependency, and not
    // the crate only the non-release `bench` uses.
    let crates = section(&text, "Rust crates");
    let listed: BTreeSet<String> = crates
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with(' '))
        .map(str::to_owned)
        .collect();
    let expected = release_packages(&ws);
    assert_eq!(
        expected,
        [
            "apache-crate 1.0.0",
            "cdla-crate 1.0.0",
            "mit-crate 1.0.0",
            "mpl-crate 2.1.0"
        ]
        .map(String::from)
        .into(),
        "the cargo tree oracle changed"
    );
    assert_eq!(listed, expected, "{text}");
    assert!(
        !text.contains("dev-crate") && !text.contains("bench-crate"),
        "{text}"
    );

    // Every manifest entry.
    let files = section(&text, "Vendored files (2)");
    for path in ["third_party/tiny/tiny.c", "third_party/tiny/LICENSE"] {
        assert!(
            files.contains(&format!("{path}\n  Licence: MIT\n")),
            "{files}"
        );
    }
    assert!(
        text.contains(
            "-- third_party/tiny/LICENSE (for third_party/tiny/tiny.c, third_party/tiny/LICENSE) \
             --\n\nCopyright (c) 2026 Tiny authors"
        ),
        "{text}"
    );
    assert!(
        log.contains("notices: PASS: 4 crates of the release build (cli) and 2 vendored files"),
        "{log}"
    );
}

#[test]
fn the_notices_print_where_mpl_source_is_and_carry_apache_notice_files() {
    let ws = notices_fixture("notices-mpl-apache");
    let out = ws.join("THIRD-PARTY-NOTICES");
    let manifest = ws.join("Cargo.toml");
    let (code, log) = xtask(&[
        "notices",
        "--manifest-path",
        manifest.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "{log}");
    let text = std::fs::read_to_string(&out).unwrap();
    let mpl = section(&text, "MPL-2.0 source code");
    assert!(
        mpl.contains(
            "mpl-crate 2.1.0\n  https://crates.io/crates/mpl-crate/2.1.0\n  \
             https://static.crates.io/crates/mpl-crate/mpl-crate-2.1.0.crate\n  \
             Repository: https://example.org/mpl-crate\n"
        ),
        "{mpl}"
    );
    assert!(!mpl.contains("apache-crate"), "{mpl}");
    let notices = section(&text, "NOTICE files");
    let shipped = std::fs::read_to_string(fixtures().join("registry/apache-crate/NOTICE")).unwrap();
    assert!(
        notices.contains(&format!(
            "-- apache-crate 1.0.0: NOTICE --\n\n{}",
            shipped.trim_end()
        )),
        "{notices}"
    );
    let licences = section(&text, "Licence texts");
    assert!(
        licences.contains("-- mit-crate 1.0.0: LICENSE-MIT --\n\nCopyright (c) 2026 The mit-crate"),
        "{licences}"
    );
}

#[test]
fn the_notices_fail_when_the_manifest_misses_a_vendored_file() {
    let ws = notices_fixture("notices-uncovered");
    write(&ws, "third_party/tiny/extra.c", "int extra;\n");
    let manifest = ws.join("Cargo.toml");
    let (code, log) = xtask(&["notices", "--manifest-path", manifest.to_str().unwrap()]);
    assert_eq!(code, Some(1), "{log}");
    assert!(
        log.contains("notices: third_party/tiny/extra.c: not listed in third_party/licences.toml"),
        "{log}"
    );
    assert!(log.contains("notices: FAIL: 1 problem;"), "{log}");
}
