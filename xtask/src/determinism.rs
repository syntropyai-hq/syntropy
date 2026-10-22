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
//! - no covered crate's `[lints]` table, and not the workspace's, sets the lints;
//! - the repository's cargo config (`.cargo/config.toml`, `.cargo/config`), parsed as TOML, does
//!   not reach clippy another way: no `CLIPPY_CONF_DIR` or `RUSTFLAGS` key (as in `[env]`), no
//!   compiler wrapper, and no `rustflags` at any depth that set a lint level (`-A`, `-W`,
//!   `--allow`, `--warn`, `--cap-lints`), which reach clippy-driver because nothing in CI sets
//!   `RUSTFLAGS`;
//! - an attribute in a covered crate that sets a level below `deny` (`allow`, `expect`, `warn`,
//!   `force_warn`) names one of the lints, a group that holds them (`warnings`, `clippy::all`,
//!   `clippy::style`) or one of their old names only in a file of [`EXCEPTIONS`], and only for the
//!   lints that entry lists. An in-source level beats the command line's `-D warnings`: a bare
//!   `#![allow(warnings)]` turns the lints off, and `#![warn(warnings)]` makes them warnings. The
//!   attributes are found from the file's Rust tokens, so no comment or literal can hide one.
//!   Every exception must still be in use;
//! - in `exec` and `obs`, only the `TickSource` implementation may lift `disallowed_methods`
//!   (rule T1), so an exception there must be the file that implements `TickSource`.

use std::path::{Path, PathBuf};

use proc_macro2::{Delimiter, TokenStream, TokenTree};

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
    /// The lints it may name, as an attribute writes them, such as `clippy::disallowed_types`.
    pub lints: &'static [&'static str],
    pub why: &'static str,
}

/// The files of covered crates that may lift a lint, each with a scoped `#[allow]` and a reason.
pub const EXCEPTIONS: &[Exception] = &[
    Exception {
        file: "crates/core/src/det_hash.rs",
        lints: &["clippy::disallowed_types"],
        why: "the `DetHashMap` and `DetHashSet` aliases, rule T3's replacements, name the std types",
    },
    Exception {
        file: "crates/tier/src/os.rs",
        lints: &["clippy::disallowed_methods"],
        why: "the OS-facing module, not part of tier's policy: `MemSnapshot::at` is a std \
              `Instant` (RFC-0020 §3)",
    },
    Exception {
        file: "crates/store/tests/writer_lock.rs",
        lints: &["clippy::disallowed_methods"],
        why: "an integration test that waits in real time for the OS to free a file lock",
    },
];

/// The names that turn the determinism lints off when a lowering attribute ([`LOWERING`]) names
/// them: the two lints, the groups that hold them (`warnings` holds every warn-level lint), and
/// the lints' names before clippy renamed them, which it still honours.
const LIFTING: &[&str] = &[
    "clippy::disallowed_types",
    "clippy::disallowed_methods",
    "clippy::disallowed_type",
    "clippy::disallowed_method",
    "clippy::all",
    "clippy::style",
    "warnings",
];

/// The keys of a manifest's `[lints.clippy]` table that would set the lints' level.
const MANIFEST_KEYS: &[&str] = &[
    "disallowed_types",
    "disallowed_methods",
    "disallowed_type",
    "disallowed_method",
    "all",
    "style",
];

/// Of [`LIFTING`], the names that lift `disallowed_methods`, the clock ban.
const LIFTING_METHODS: &[&str] = &[
    "clippy::disallowed_methods",
    "clippy::disallowed_method",
    "clippy::all",
    "clippy::style",
    "warnings",
];

/// The repository's cargo config files, which reach every cargo run from the root.
const CARGO_CONFIGS: &[&str] = &[".cargo/config.toml", ".cargo/config"];

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
    for config in CARGO_CONFIGS {
        if let Ok(text) = std::fs::read_to_string(root.join(config)) {
            problems.extend(cargo_config_problems(config, &text));
        }
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

/// Keys of `section`'s `clippy` table in a manifest that set one of [`MANIFEST_KEYS`]: `[lints.clippy]`
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
            MANIFEST_KEYS.contains(&key.as_str())
        } else if current == section && key == "clippy" {
            MANIFEST_KEYS.iter().any(|lint| {
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
        let levels = match lint_levels(text) {
            Ok(levels) => levels,
            Err(error) => {
                problems.push(format!("{file}: {error}"));
                continue;
            }
        };
        for LintLevel { level, lint, line } in levels {
            if !LIFTING.contains(&lint.as_str()) {
                continue;
            }
            match exception {
                Some(at) if exceptions[at].lints.contains(&lint.as_str()) => used[at] = true,
                _ => problems.push(format!(
                    "{file}:{line}: `{level}({lint})` lifts a determinism lint (RFC-0060 §2.1, \
                     §2.4); only a listed exception in xtask/src/determinism.rs may"
                )),
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
        let lifts_methods = exception
            .lints
            .iter()
            .any(|lint| LIFTING_METHODS.contains(lint));
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

/// The lint levels below `deny` that an attribute can set. `allow` and `expect` silence a lint,
/// and `warn` on the `warnings` group turns `-D warnings` back into warnings.
const LOWERING: &[&str] = &["allow", "expect", "warn", "force_warn"];

/// A lint-level attribute: the level, one lint it names (`clippy::all`, `warnings`) and its line.
#[derive(Debug, PartialEq, Eq)]
pub struct LintLevel {
    pub level: String,
    pub lint: String,
    pub line: usize,
}

/// Every lint a lowering attribute in `text` names, found from the file's Rust tokens as the
/// compiler sees them, so comments, string and char literals and raw strings cannot hide one:
/// `#[...]`, `#![...]`, `cfg_attr(pred, ...)` at any depth, and attributes inside macro bodies.
/// Doc comments are `doc` attributes and name nothing. A file that does not tokenize is an error.
pub fn lint_levels(text: &str) -> Result<Vec<LintLevel>, String> {
    let stream: TokenStream = text
        .parse()
        .map_err(|error| format!("does not tokenize as Rust: {error}"))?;
    let mut found = Vec::new();
    walk(stream, &mut found);
    Ok(found)
}

/// Finds the attributes in `stream` and in every group inside it.
fn walk(stream: TokenStream, found: &mut Vec<LintLevel>) {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    for (index, token) in tokens.iter().enumerate() {
        let TokenTree::Group(group) = token else {
            continue;
        };
        if group.delimiter() == Delimiter::Bracket && after_hash(&tokens, index) {
            attribute(group.stream(), found);
        }
        walk(group.stream(), found);
    }
}

/// Whether the token before `index` is `#`, or `#` then `!`.
fn after_hash(tokens: &[TokenTree], index: usize) -> bool {
    let punct = |at: Option<usize>, c: char| {
        at.and_then(|at| tokens.get(at))
            .is_some_and(|t| matches!(t, TokenTree::Punct(p) if p.as_char() == c))
    };
    punct(index.checked_sub(1), '#')
        || (punct(index.checked_sub(1), '!') && punct(index.checked_sub(2), '#'))
}

/// One attribute's tokens: a lint level with its list, or `cfg_attr` with its attributes.
fn attribute(stream: TokenStream, found: &mut Vec<LintLevel>) {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    let (Some(TokenTree::Ident(name)), Some(TokenTree::Group(args))) =
        (tokens.first(), tokens.get(1))
    else {
        return;
    };
    if args.delimiter() != Delimiter::Parenthesis {
        return;
    }
    let name = name.to_string();
    if name == "cfg_attr" {
        for item in split_commas(args.stream()).into_iter().skip(1) {
            attribute(item, found);
        }
        return;
    }
    if !LOWERING.contains(&name.as_str()) {
        return;
    }
    for item in split_commas(args.stream()) {
        let tokens: Vec<TokenTree> = item.into_iter().collect();
        // `reason = "..."` names no lint.
        if tokens
            .iter()
            .any(|t| matches!(t, TokenTree::Punct(p) if p.as_char() == '='))
        {
            continue;
        }
        let Some(first) = tokens.first() else {
            continue;
        };
        let lint: String = tokens
            .iter()
            .map(|t| t.to_string().trim_start_matches("r#").to_owned())
            .collect();
        found.push(LintLevel {
            level: name.clone(),
            lint,
            line: first.span().start().line,
        });
    }
}

/// `stream` split at its top-level commas.
fn split_commas(stream: TokenStream) -> Vec<TokenStream> {
    let mut items = vec![TokenStream::new()];
    for token in stream {
        if matches!(&token, TokenTree::Punct(p) if p.as_char() == ',') {
            items.push(TokenStream::new());
        } else if let Some(last) = items.last_mut() {
            last.extend([token]);
        }
    }
    items.retain(|item| !item.is_empty());
    items
}

/// The flags in a `rustflags` value that set a lint level clippy would obey.
fn lint_level_flags(value: &toml::Value) -> Vec<String> {
    let flags: Vec<String> = match value {
        toml::Value::String(text) => text.split_whitespace().map(str::to_owned).collect(),
        toml::Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map_or_else(|| format!("{item:?}"), str::to_owned)
            })
            .collect(),
        other => vec![format!("{other:?}")],
    };
    flags
        .into_iter()
        .filter(|flag| {
            let flag = flag.trim();
            ["-A", "-W", "--allow", "--warn", "--cap-lints"]
                .iter()
                .any(|level| flag.starts_with(level))
                || flag.contains("crate-attr")
        })
        .collect()
}

/// Ways a repository cargo config file would reach clippy around the per-crate configs, read
/// from the parsed TOML, so inline tables, dotted and quoted keys all count. A file that does not
/// parse is a problem: cargo would refuse it, and the check cannot vouch for it.
fn cargo_config_problems(file: &str, text: &str) -> Vec<String> {
    let table: toml::Table = match toml::from_str(text) {
        Ok(table) => table,
        Err(error) => return vec![format!("{file}: does not parse as TOML: {error}")],
    };
    let mut problems = Vec::new();
    config_walk(file, "", &table, &mut problems);
    problems
}

fn config_walk(file: &str, path: &str, table: &toml::Table, problems: &mut Vec<String>) {
    for (key, value) in table {
        let full = if path.is_empty() {
            key.clone()
        } else {
            format!("{path}.{key}")
        };
        for variable in ["CLIPPY_CONF_DIR", "RUSTFLAGS"] {
            if key.contains(variable) {
                problems.push(format!(
                    "{file}: `{full}` sets {variable}, which would change how clippy lints every \
                     covered crate"
                ));
            }
        }
        if key == "rustc-wrapper" || key == "rustc-workspace-wrapper" {
            problems.push(format!(
                "{file}: sets `{full}`; a wrapper can rewrite clippy's flags in every covered crate"
            ));
        }
        if key == "rustflags" {
            for flag in lint_level_flags(value) {
                problems.push(format!(
                    "{file}: `{full}` passes `{flag}`, which sets a lint level clippy would obey; \
                     the lints of {SOURCE} stay on in covered crates"
                ));
            }
        }
        if let toml::Value::Table(inner) = value {
            config_walk(file, &full, inner, problems);
        }
    }
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
            found[0].starts_with("crates/exec/src/step.rs:2: `allow(clippy::disallowed_methods)`"),
            "{found:?}"
        );
    }

    #[test]
    fn groups_that_hold_the_lints_count_and_comments_do_not() {
        let files = [file(
            "crates/solver/src/lib.rs",
            "#![allow(clippy::all)]\n#[expect(clippy::style)]\n// #[allow(clippy::disallowed_types)]\n\
             #[allow(clippy::all_caps_not_a_lint)]\n#![deny(warnings)]\n#![forbid(clippy::all)]\n",
        )];
        let found = source_problems(&files, EXEC_ONLY, &[]);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(found[0].contains(":1: `allow(clippy::all)`"), "{found:?}");
        assert!(
            found[1].contains(":2: `expect(clippy::style)`"),
            "{found:?}"
        );
    }

    /// Review B1, and `warn(warnings)`, which turns `-D warnings` back into warnings.
    #[test]
    fn lowering_warnings_or_an_old_lint_name_fails() {
        let files = [file(
            "crates/exec/src/lib.rs",
            "#![forbid(unsafe_code)]\n#![allow(warnings)]\n#[expect(warnings)]\nfn f() {}\n\
             #![allow(renamed_and_removed_lints, clippy::disallowed_method)]\n\
             #[allow(clippy :: disallowed_type)]\n\
             #[cfg_attr(test, allow(warnings))]\n#![warn(warnings)]\n#![force_warn(clippy::style)]\n",
        )];
        let found = source_problems(&files, EXEC_ONLY, &[]);
        for expected in [
            ":2: `allow(warnings)`",
            ":3: `expect(warnings)`",
            ":5: `allow(clippy::disallowed_method)`",
            ":6: `allow(clippy::disallowed_type)`",
            ":7: `allow(warnings)`",
            ":8: `warn(warnings)`",
            ":9: `force_warn(clippy::style)`",
        ] {
            assert!(
                found.iter().any(|line| line.contains(expected)),
                "{expected}: {found:#?}"
            );
        }
        assert_eq!(found.len(), 7, "{found:#?}");
    }

    fn levels(text: &str) -> Vec<(String, String, usize)> {
        lint_levels(text)
            .unwrap()
            .into_iter()
            .map(|l| (l.level, l.lint, l.line))
            .collect()
    }

    /// The attributes come from Rust's tokens: no literal or comment can hide or fake one.
    #[test]
    fn lint_levels_come_from_tokens_not_text() {
        let text = r##"#[allow(
    clippy::disallowed_methods,
    reason = "a reason (with, commas) \" and ) here"
)]
fn allow(x: u8) {}
const Q: char = '"';
fn f<'a>(x: &'a str) -> &'a str { x } // a trailing " quote
#[allow(warnings)]
fn g() {}
const R: &str = r#"#[allow(clippy::all)] " still raw"#;
/* #[allow(warnings)] in a block comment */
/// #[allow(warnings)] in a doc comment
#![cfg_attr(feature = "f", expect(clippy::style, warnings))]
#[cfg_attr(all(), cfg_attr(test, allow(clippy::disallowed_types)))]
macro_rules! m { () => { #[allow(clippy::disallowed_methods)] fn h() {} } }
let s = "#[allow(warnings)]"; foo.allow(warnings);
"##;
        assert_eq!(
            levels(text),
            [
                ("allow".into(), "clippy::disallowed_methods".into(), 2),
                ("allow".into(), "warnings".into(), 8),
                ("expect".into(), "clippy::style".into(), 13),
                ("expect".into(), "warnings".into(), 13),
                ("allow".into(), "clippy::disallowed_types".into(), 14),
                ("allow".into(), "clippy::disallowed_methods".into(), 15),
            ]
        );
    }

    #[test]
    fn a_file_that_does_not_tokenize_fails_closed() {
        let files = [file(
            "crates/exec/src/lib.rs",
            "fn f() { \"unterminated }\n",
        )];
        let found = source_problems(&files, EXEC_ONLY, &[]);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].starts_with("crates/exec/src/lib.rs: does not tokenize as Rust"),
            "{found:?}"
        );
    }

    /// Review B2: rustflags in the repository's cargo config reach clippy-driver. The config is
    /// parsed as TOML, so inline tables and quoted or dotted keys count too.
    #[test]
    fn a_cargo_config_that_reaches_clippy_fails() {
        let real = std::fs::read_to_string(root().join(".cargo/config.toml")).unwrap();
        assert_eq!(
            cargo_config_problems(".cargo/config.toml", &real),
            Vec::<String>::new()
        );
        let fine = "[build]\nrustflags = [\"-C\", \"target-cpu=native\"]\n";
        assert_eq!(
            cargo_config_problems(".cargo/config.toml", fine),
            Vec::<String>::new()
        );
        for (text, expected) in [
            (
                "[build]\nrustflags = [\"-Aclippy::disallowed_methods\"]\n",
                "`build.rustflags` passes `-Aclippy::disallowed_methods`",
            ),
            (
                "[build]\nrustflags = \"-A warnings\"\n",
                "`build.rustflags` passes `-A`",
            ),
            (
                "build = { rustflags = [\"-Awarnings\"] }\n",
                "`build.rustflags` passes `-Awarnings`",
            ),
            (
                "[build]\n\"rustflags\" = [\"--cap-lints\", \"allow\"]\n",
                "`build.rustflags` passes `--cap-lints`",
            ),
            (
                "[target.'cfg(all())']\nrustflags = [\n  \"--cap-lints\",\n  \"allow\",\n]\n",
                "`target.cfg(all()).rustflags` passes `--cap-lints`",
            ),
            (
                "target.x86_64-unknown-linux-gnu.rustflags = [\"--allow=warnings\"]\n",
                "`target.x86_64-unknown-linux-gnu.rustflags` passes `--allow=warnings`",
            ),
            (
                "[target.x86_64-unknown-linux-gnu]\nrustflags = [\"-W\", \"clippy::all\"]\n",
                "passes `-W`",
            ),
            (
                "[profile.dev]\nrustflags = [\"-Zcrate-attr=allow(warnings)\"]\n",
                "`profile.dev.rustflags` passes `-Zcrate-attr=allow(warnings)`",
            ),
            (
                "[env]\nRUSTFLAGS = \"-Awarnings\"\n",
                "`env.RUSTFLAGS` sets RUSTFLAGS",
            ),
            (
                "[env]\nCARGO_ENCODED_RUSTFLAGS = { value = \"x\", force = true }\n",
                "`env.CARGO_ENCODED_RUSTFLAGS` sets RUSTFLAGS",
            ),
            (
                "[env]\nCLIPPY_CONF_DIR = \"/x\"\n",
                "`env.CLIPPY_CONF_DIR` sets CLIPPY_CONF_DIR",
            ),
            (
                "build = { rustc-wrapper = \"w\" }\n",
                "sets `build.rustc-wrapper`",
            ),
            (
                "[build]\nrustc-workspace-wrapper = \"w\"\n",
                "sets `build.rustc-workspace-wrapper`",
            ),
            ("[build\nrustflags = 1\n", "does not parse as TOML"),
        ] {
            let found = cargo_config_problems(".cargo/config.toml", text);
            assert!(
                found.iter().any(|line| line.contains(expected)),
                "{expected}: {found:#?}"
            );
        }
    }

    #[test]
    fn an_exception_covers_its_own_lints_only() {
        let exceptions = [Exception {
            file: "crates/solver/src/map.rs",
            lints: &["clippy::disallowed_types"],
            why: "test",
        }];
        let files = [file(
            "crates/solver/src/map.rs",
            "#[allow(clippy::disallowed_types)]\ntype M = ();\n#[allow(clippy::disallowed_methods)]\n",
        )];
        let found = source_problems(&files, EXEC_ONLY, &exceptions);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].contains(":3: `allow(clippy::disallowed_methods)`"),
            "{found:?}"
        );
    }

    #[test]
    fn an_unused_exception_or_one_outside_the_covered_crates_fails() {
        let exceptions = [
            Exception {
                file: "crates/solver/src/map.rs",
                lints: &["clippy::disallowed_types"],
                why: "test",
            },
            Exception {
                file: "crates/net/src/lib.rs",
                lints: &["clippy::disallowed_methods"],
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
            lints: &["clippy::disallowed_methods"],
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
    fn paths_are_relative_with_forward_slashes() {
        let root = Path::new("w");
        let path = root.join("crates").join("exec").join("src").join("lib.rs");
        assert_eq!(relative(root, &path), "crates/exec/src/lib.rs");
    }
}
