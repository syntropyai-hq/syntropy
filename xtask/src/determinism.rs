// SPDX-License-Identifier: MIT OR Apache-2.0

//! The determinism lints of RFC-0060 §2.1 (rule T1) and §2.4 (rule T3), QUAL-E3-T03.
//!
//! Clippy enforces the rules. `disallowed-types` bans `HashMap`, `HashSet`, `RandomState` and the
//! OS-seeded generator types, and `disallowed-methods` bans clock reads and unseeded random
//! generators, as `ci/clippy-determinism.toml` lists them. Clippy reads its config from each
//! crate's own directory, so every crate in [`COVERED`] holds a copy as its `clippy.toml`, and the
//! `clippy` step of `cargo xtask ci lint` applies it to the crate's library, binaries, tests and
//! benches. `xtask/tests/determinism.rs` lints a fixture to show that each banned item fails.
//!
//! This module is the `determinism` step that follows. It checks what clippy cannot see:
//!
//! - every covered crate's `clippy.toml` is identical to `ci/clippy-determinism.toml`, and no
//!   `.clippy.toml` sits beside it, since clippy would read that file instead;
//! - no covered crate's `[lints]` table, and not the workspace's, sets the lints, and
//!   `.cargo/config.toml` does not point clippy at another config with `CLIPPY_CONF_DIR`;
//! - code in a covered crate names one of the lints, or a group that holds them (`clippy::all`,
//!   `clippy::style`), only in a file of [`EXCEPTIONS`], and only for the lints that entry lists.
//!   Every exception must still be in use;
//! - in `exec` and `obs`, only the `TickSource` implementation may lift `disallowed_methods`
//!   (rule T1), so an exception there must be the file that implements `TickSource`.

use std::path::{Path, PathBuf};

use crate::run::Context;

/// The clippy config every covered crate copies, relative to the workspace root.
pub const SOURCE: &str = "ci/clippy-determinism.toml";

/// The crates the lints cover, by directory under `crates/`, each with the rule that covers it.
///
/// Rule T1 names `cluster`, `solver`, `sched`, `tier`, `store`, `api` and the runner's lifecycle
/// machine. QUAL-E3-T03 adds `exec` and `obs`. Rule T3 covers the crates a protocol core or the
/// solver reaches. Those are the crates that hold the cores (`cluster`, and `daemon` for the
/// supervisor's restart policy, RFC-0001 §2.3 invariant 12) and the solver, and the crates their
/// code calls into: `core` and `manifest`. The cores are sans-I/O, so their shells' crates, such as
/// `net`, are not reached. A lint covers a whole crate, so a crate that rule T1 names only in part
/// is covered whole, and its other parts are listed in [`EXCEPTIONS`] where they need the clock.
pub const COVERED: &[(&str, &str)] = &[
    (
        "core",
        "rule T3: every protocol core and the solver reach it",
    ),
    (
        "obs",
        "QUAL-E3-T03: hot-thread timing goes through a TickSource",
    ),
    ("manifest", "rule T3: the solver reads the model manifest"),
    ("solver", "rules T1 and T3"),
    ("tier", "rule T1: the tier manager's policy"),
    ("store", "rule T1: the store's scheduling"),
    (
        "exec",
        "QUAL-E3-T03: hot-thread timing goes through a TickSource",
    ),
    (
        "cluster",
        "rules T1 and T3: the membership, election, log, transition and recovery cores",
    ),
    ("sched", "rule T1"),
    ("api", "rule T1: ingress routing"),
    ("runner", "rule T1: the runner's lifecycle machine"),
    ("daemon", "rule T3: the supervisor's restart-policy core"),
];

/// A file that may lift some of the lints, and why.
#[derive(Debug)]
pub struct Exception {
    /// Relative to the workspace root, with `/` separators.
    pub file: &'static str,
    /// The lints it may name, such as `disallowed_types`.
    pub lints: &'static [&'static str],
    pub why: &'static str,
}

/// The files of covered crates that may lift a lint, each with a scoped `#[allow]` and a reason.
pub const EXCEPTIONS: &[Exception] = &[
    Exception {
        file: "crates/core/src/det_hash.rs",
        lints: &["disallowed_types"],
        why: "the `DetHashMap` and `DetHashSet` aliases, rule T3's replacements, name the std types",
    },
    Exception {
        file: "crates/tier/src/os.rs",
        lints: &["disallowed_methods"],
        why: "the OS-facing module, not part of tier's policy: `MemSnapshot::at` is a std \
              `Instant` (RFC-0020 §3)",
    },
    Exception {
        file: "crates/store/tests/writer_lock.rs",
        lints: &["disallowed_methods"],
        why: "an integration test that waits in real time for the OS to free a file lock",
    },
];

/// The lints and lint groups that turn the determinism lints off when allowed.
const LINTS: &[&str] = &["disallowed_types", "disallowed_methods", "all", "style"];

/// Crates where only the `TickSource` implementation may read the clock (rule T1).
const TICK_SOURCE_CRATES: &[&str] = &["exec", "obs"];

/// The text that marks the file implementing `TickSource`.
const TICK_SOURCE_IMPL: &str = "TickSource for ";

/// The `determinism` step of `cargo xtask ci lint`.
pub fn step(context: &Context) -> Result<(), String> {
    let problems = check(&context.root, COVERED, EXCEPTIONS);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// Every problem in the checkout at `root`, given the covered crates and the exceptions.
pub fn check(root: &Path, covered: &[(&str, &str)], exceptions: &[Exception]) -> Vec<String> {
    let mut problems = Vec::new();
    let source = match std::fs::read_to_string(root.join(SOURCE)) {
        Ok(text) => text,
        Err(error) => return vec![format!("{SOURCE}: {error}")],
    };
    let mut files = Vec::new();
    for &(name, _) in covered {
        let dir = root.join("crates").join(name);
        if !dir.join("Cargo.toml").is_file() {
            problems.push(format!(
                "crates/{name}: a covered crate that is not in the workspace"
            ));
            continue;
        }
        let config = std::fs::read_to_string(dir.join("clippy.toml")).ok();
        problems.extend(config_problems(
            name,
            config.as_deref(),
            dir.join(".clippy.toml").exists(),
            &source,
        ));
        let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap_or_default();
        problems.extend(manifest_problems(
            &format!("crates/{name}/Cargo.toml"),
            &manifest,
            "lints",
        ));
        for path in rust_files(&dir) {
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            files.push((relative(root, &path), text));
        }
    }
    let workspace = std::fs::read_to_string(root.join("Cargo.toml")).unwrap_or_default();
    problems.extend(manifest_problems(
        "Cargo.toml",
        &workspace,
        "workspace.lints",
    ));
    let cargo_config = std::fs::read_to_string(root.join(".cargo/config.toml")).unwrap_or_default();
    if cargo_config.contains("CLIPPY_CONF_DIR") {
        problems.push(
            ".cargo/config.toml: sets CLIPPY_CONF_DIR, which would replace every covered crate's \
             clippy.toml"
                .to_owned(),
        );
    }
    problems.extend(source_problems(&files, covered, exceptions));
    problems
}

/// A covered crate's own clippy config against the source.
fn config_problems(
    name: &str,
    config: Option<&str>,
    dot_config: bool,
    source: &str,
) -> Vec<String> {
    let mut problems = Vec::new();
    match config {
        None => problems.push(format!(
            "crates/{name}/clippy.toml: missing; copy {SOURCE} there (a covered crate)"
        )),
        Some(text) if text != source => problems.push(format!(
            "crates/{name}/clippy.toml: differs from {SOURCE}; copy it again"
        )),
        Some(_) => {}
    }
    if dot_config {
        problems.push(format!(
            "crates/{name}/.clippy.toml: clippy would read it instead of clippy.toml; remove it"
        ));
    }
    problems
}

/// Keys of `section`'s `clippy` table in a manifest that set one of [`LINTS`]: `[lints.clippy]`
/// in a crate, `[workspace.lints.clippy]` in the workspace, or an inline `clippy = { ... }`.
fn manifest_problems(file: &str, text: &str, section: &str) -> Vec<String> {
    let table = format!("{section}.clippy");
    let mut problems = Vec::new();
    let mut current = String::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if let Some(header) = line.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
            current = header.trim().to_owned();
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().trim_matches('"').replace('-', "_");
        let names_a_lint = if current == table {
            LINTS.contains(&key.as_str())
        } else if current == section && key == "clippy" {
            LINTS.iter().any(|lint| {
                value
                    .replace('-', "_")
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .any(|word| word == *lint)
            })
        } else {
            false
        };
        if names_a_lint {
            problems.push(format!(
                "{file}:{}: sets a determinism lint in [{current}] (`{}`); the lints of {SOURCE} \
                 stay on in covered crates",
                index + 1,
                line
            ));
        }
    }
    problems
}

/// Lint attributes in the covered crates' Rust files against the exceptions.
fn source_problems(
    files: &[(String, String)],
    covered: &[(&str, &str)],
    exceptions: &[Exception],
) -> Vec<String> {
    let mut problems = Vec::new();
    let mut used = vec![false; exceptions.len()];
    for (file, text) in files {
        let exception = exceptions.iter().position(|e| e.file == file);
        for (index, line) in text.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            for lint in LINTS.iter().filter(|lint| names_lint(line, lint)) {
                match exception {
                    Some(at) if exceptions[at].lints.contains(lint) => used[at] = true,
                    _ => problems.push(format!(
                        "{file}:{}: names `clippy::{lint}`, which lifts a determinism lint (RFC-0060 \
                         §2.1, §2.4); only a listed exception in xtask/src/determinism.rs may",
                        index + 1
                    )),
                }
            }
        }
    }
    for (exception, used) in exceptions.iter().zip(used) {
        let crate_name = exception
            .file
            .strip_prefix("crates/")
            .and_then(|rest| rest.split('/').next())
            .unwrap_or_default();
        if !covered.iter().any(|&(name, _)| name == crate_name) {
            problems.push(format!(
                "{}: an exception outside the covered crates",
                exception.file
            ));
            continue;
        }
        if !used {
            problems.push(format!(
                "{}: a listed exception ({}) that no longer names {:?}; remove it from \
                 xtask/src/determinism.rs",
                exception.file, exception.why, exception.lints
            ));
        }
        let lifts_methods = exception.lints.contains(&"disallowed_methods")
            || exception.lints.contains(&"all")
            || exception.lints.contains(&"style");
        if lifts_methods && TICK_SOURCE_CRATES.contains(&crate_name) {
            let text = files
                .iter()
                .find(|(file, _)| file == exception.file)
                .map(|(_, text)| text.as_str())
                .unwrap_or_default();
            if !text.contains(TICK_SOURCE_IMPL) {
                problems.push(format!(
                    "{}: in `{crate_name}` only the `TickSource` implementation may read the clock \
                     (rule T1), and this file does not implement it",
                    exception.file
                ));
            }
        }
    }
    problems
}

/// Whether `line` names `clippy::<lint>` as a whole word.
fn names_lint(line: &str, lint: &str) -> bool {
    let needle = format!("clippy::{lint}");
    line.match_indices(&needle).any(|(at, _)| {
        let next = line[at + needle.len()..].chars().next();
        !next.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

/// Every `.rs` file under `dir`, skipping `target` directories, sorted.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if entry.file_name() != "target" {
                    pending.push(path);
                }
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// `path` relative to `root`, with `/` separators on every OS.
fn relative(root: &Path, path: &Path) -> String {
    let inside = path.strip_prefix(root).unwrap_or(path);
    inside
        .components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check_deps::layers::Layers;

    fn root() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
    }

    fn file(path: &str, text: &str) -> (String, String) {
        (path.to_owned(), text.to_owned())
    }

    const EXEC_ONLY: &[(&str, &str)] = &[("exec", "test"), ("solver", "test")];

    #[test]
    fn the_checkout_passes() {
        assert_eq!(check(root(), COVERED, EXCEPTIONS), Vec::<String>::new());
    }

    #[test]
    fn every_covered_crate_is_in_the_layer_table() {
        let layers = Layers::load(root()).unwrap();
        for &(name, _) in COVERED {
            assert!(layers.place(name).is_some(), "{name}");
        }
    }

    #[test]
    fn the_crates_rule_t1_names_and_exec_and_obs_are_covered() {
        for name in [
            "cluster", "solver", "sched", "tier", "store", "api", "runner", "exec", "obs",
        ] {
            assert!(COVERED.iter().any(|&(n, _)| n == name), "{name}");
        }
    }

    #[test]
    fn a_missing_or_changed_copy_of_the_config_fails() {
        assert_eq!(
            config_problems("exec", Some("x"), false, "x"),
            Vec::<String>::new()
        );
        let missing = config_problems("exec", None, false, "x");
        assert!(
            missing[0].contains("crates/exec/clippy.toml: missing"),
            "{missing:?}"
        );
        let changed = config_problems("exec", Some("y"), false, "x");
        assert!(changed[0].contains("differs from"), "{changed:?}");
        let shadowed = config_problems("exec", Some("x"), true, "x");
        assert!(shadowed[0].contains(".clippy.toml"), "{shadowed:?}");
    }

    #[test]
    fn a_manifest_that_allows_the_lints_fails() {
        let crate_table =
            "[package]\nname = \"x\"\n\n[lints.clippy]\ndisallowed-methods = \"allow\"\n";
        let found = manifest_problems("crates/exec/Cargo.toml", crate_table, "lints");
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].starts_with("crates/exec/Cargo.toml:5: "),
            "{found:?}"
        );

        let inline = "[lints]\nclippy = { all = \"allow\" }\n";
        assert_eq!(manifest_problems("m", inline, "lints").len(), 1);

        let workspace = "[workspace.lints.clippy]\ndisallowed_types = \"allow\"\n";
        assert_eq!(
            manifest_problems("m", workspace, "workspace.lints").len(),
            1
        );

        let other =
            "[lints.clippy]\nneedless_range_loop = \"allow\"\n[dependencies]\nall = \"1\"\n";
        assert_eq!(manifest_problems("m", other, "lints"), Vec::<String>::new());
    }

    #[test]
    fn an_allow_outside_the_exceptions_fails() {
        let files = [file(
            "crates/exec/src/step.rs",
            "#[allow(\n    clippy::disallowed_methods,\n    reason = \"no\"\n)]\nfn f() {}\n",
        )];
        let found = source_problems(&files, EXEC_ONLY, &[]);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].starts_with("crates/exec/src/step.rs:2: names `clippy::disallowed_methods`")
        );
    }

    #[test]
    fn groups_that_hold_the_lints_count_and_comments_do_not() {
        let files = [file(
            "crates/solver/src/lib.rs",
            "#![allow(clippy::all)]\n#[expect(clippy::style)]\n// clippy::disallowed_types\n\
             #[allow(clippy::all_caps_not_a_lint)]\n",
        )];
        let found = source_problems(&files, EXEC_ONLY, &[]);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found[0].contains(":1: names `clippy::all`"), "{found:?}");
        assert!(found[1].contains(":2: names `clippy::style`"), "{found:?}");
    }

    #[test]
    fn an_exception_covers_its_own_lints_only() {
        let exceptions = [Exception {
            file: "crates/solver/src/map.rs",
            lints: &["disallowed_types"],
            why: "test",
        }];
        let files = [file(
            "crates/solver/src/map.rs",
            "#[allow(clippy::disallowed_types)]\ntype M = ();\n#[allow(clippy::disallowed_methods)]\n",
        )];
        let found = source_problems(&files, EXEC_ONLY, &exceptions);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].contains(":3: names `clippy::disallowed_methods`"),
            "{found:?}"
        );
    }

    #[test]
    fn an_unused_exception_or_one_outside_the_covered_crates_fails() {
        let exceptions = [
            Exception {
                file: "crates/solver/src/map.rs",
                lints: &["disallowed_types"],
                why: "test",
            },
            Exception {
                file: "crates/net/src/lib.rs",
                lints: &["disallowed_methods"],
                why: "test",
            },
        ];
        let files = [file("crates/solver/src/map.rs", "fn f() {}\n")];
        let found = source_problems(&files, EXEC_ONLY, &exceptions);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found[0].contains("no longer names"), "{found:?}");
        assert!(found[1].contains("outside the covered crates"), "{found:?}");
    }

    #[test]
    fn in_exec_only_the_tick_source_implementation_may_read_the_clock() {
        let allow = "#[allow(clippy::disallowed_methods, reason = \"the tick source\")]\n";
        let exceptions = [Exception {
            file: "crates/exec/src/tick.rs",
            lints: &["disallowed_methods"],
            why: "test",
        }];
        let tick =
            format!("impl TickSource for Os {{\n{allow}fn ticks(&self) -> u64 {{ 0 }}\n}}\n");
        let files = [file("crates/exec/src/tick.rs", &tick)];
        assert_eq!(
            source_problems(&files, EXEC_ONLY, &exceptions),
            Vec::<String>::new()
        );

        let elsewhere = [file("crates/exec/src/tick.rs", allow)];
        let found = source_problems(&elsewhere, EXEC_ONLY, &exceptions);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].contains("only the `TickSource` implementation"),
            "{found:?}"
        );
    }

    #[test]
    fn names_lint_matches_whole_names() {
        assert!(names_lint("#[allow(clippy::all)]", "all"));
        assert!(names_lint("clippy::all,", "all"));
        assert!(!names_lint("clippy::all_caps", "all"));
        assert!(!names_lint("clippy::alloc_instead_of_core", "all"));
    }

    #[test]
    fn paths_are_relative_with_forward_slashes() {
        let root = Path::new("w");
        let path = root.join("crates").join("exec").join("src").join("lib.rs");
        assert_eq!(relative(root, &path), "crates/exec/src/lib.rs");
    }
}
