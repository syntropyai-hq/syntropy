// SPDX-License-Identifier: MIT OR Apache-2.0

//! The determinism lints of RFC-0060 §2.1 (rule T1) and §2.4 (rule T3), QUAL-E3-T03.
//!
//! Clippy enforces the rules. `disallowed-types` bans `HashMap`, `HashSet`, `RandomState`, the
//! OS-seeded generator types and the production clock, entropy and tick sources,
//! `syntropy_io::os::SystemClock`, `OsEntropy` and `SystemTicks`, and `disallowed-methods` bans
//! clock reads and unseeded random generators, as `ci/clippy-determinism.toml` lists them. Only a
//! binary's wiring builds the production sources: the CLI, which the lints do not cover, and the
//! wiring files of `runner` and `daemon`, which join [`EXCEPTIONS`] for
//! `clippy::disallowed_types` when they land (#1763). Clippy reads its config from each crate's
//! own directory, so every crate in [`COVERED`] holds a copy as its `clippy.toml`, and the
//! `clippy` step of `cargo xtask ci lint` applies it to the crate's library, binaries, tests and
//! benches. `xtask/tests/determinism.rs` lints a fixture to show that each banned item fails.
//!
//! This module is the `determinism` step that follows. It checks what clippy cannot see:
//!
//! - every covered crate's `clippy.toml` is identical to `ci/clippy-determinism.toml`, and no
//!   `.clippy.toml` sits beside it, since clippy would read that file instead;
//! - no covered crate's `[lints]` table, and not the workspace's, sets the lints;
//! - the repository's cargo config (`.cargo/config.toml`, `.cargo/config`), parsed as TOML, holds
//!   only the `cargo xtask` alias and, for the Linux triples of [`LINKER_TRIPLES`], `linker` set
//!   to [`PROVIDER_LINKER`] and nothing else (COMP-E9-T19). Much else in it reaches
//!   clippy-driver: lint-level `rustflags` (CI sets `RUSTFLAGS` only for the loom suite's own test
//!   build, never for clippy), `[env]`, compiler wrappers, and an alias that shadows
//!   `cargo clippy`. A linker sees only link arguments, after the lints have run;
//! - an attribute in a covered crate that sets a level below `deny` (`allow`, `expect`, `warn`,
//!   `force_warn`) names one of the lints, a group that holds them (`warnings`, `clippy::all`,
//!   `clippy::style`) or one of their old names only in a file of [`EXCEPTIONS`], and only for the
//!   lints that entry lists. An in-source level beats the command line's `-D warnings`: a bare
//!   `#![allow(warnings)]` turns the lints off, and `#![warn(warnings)]` makes them warnings. The
//!   levels are found from the file's Rust tokens, so no comment or literal can hide one, and in
//!   macro bodies and arguments too; a lint name a macro substitutes (`allow($lint)`, `#[$attr]`)
//!   fails, because it cannot be read. Every exception must still be in use;
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
        file: "crates/tier/src/os/procfs.rs",
        lints: &["clippy::disallowed_methods"],
        why: "the Linux procfs adapter, on the OS side of tier, not its policy: it stamps \
              `MemSnapshot::at`, a std `Instant` (RFC-0020 §3)",
    },
    Exception {
        file: "crates/tier/src/os/windows.rs",
        lints: &["clippy::disallowed_methods"],
        why: "the Windows adapter, on the OS side of tier, not its policy: it stamps \
              `MemSnapshot::at`, a std `Instant` (RFC-0020 §3)",
    },
    Exception {
        file: "crates/tier/tests/commit_gate.rs",
        lints: &["clippy::disallowed_methods"],
        why: "an integration test whose scripted OS builds the std `Instant` that \
              `MemSnapshot::at` holds",
    },
    Exception {
        file: "crates/core/src/canon.rs",
        lints: &["clippy::disallowed_types"],
        why: "a unit test feeds the canonical encoder a `RandomState` `HashMap` to show that \
              map order does not reach the encoding",
    },
    Exception {
        file: "crates/store/tests/writer_lock.rs",
        lints: &["clippy::disallowed_methods"],
        why: "an integration test that waits in real time for the OS to free a file lock",
    },
    Exception {
        file: "crates/tier/src/os/macos.rs",
        lints: &["clippy::disallowed_methods"],
        why: "the macOS adapter, on the OS side of tier, not its policy: it stamps \
              `MemSnapshot::at`, a std `Instant` (RFC-0020 §3)",
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
            if lint.contains('$') {
                problems.push(format!(
                    "{file}:{line}: `{level}` takes `{lint}` from a macro, so the lint it names \
                     cannot be checked; name the lint where the attribute is written (RFC-0060 \
                     §2.1, §2.4)"
                ));
                continue;
            }
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

/// Every lint a lowering level names in `text`, found from the file's Rust tokens as the
/// compiler sees them, so comments, string and char literals and raw strings cannot hide one.
///
/// A level followed by its list counts wherever it appears: in `#[...]` and `#![...]`, inside
/// `cfg_attr(pred, ...)` at any depth, in a macro body, and in a macro's arguments, where
/// `m!(allow(warnings))` could hand the attribute to a macro defined elsewhere. Lint names built
/// from a macro's metavariables (`allow($lint)`, `#[$attr]`) are reported as `$...`, because the
/// name they expand to cannot be read here. Doc comments are `doc` attributes and name nothing. A
/// file that does not tokenize is an error.
pub fn lint_levels(text: &str) -> Result<Vec<LintLevel>, String> {
    let stream: TokenStream = text
        .parse()
        .map_err(|error| format!("does not tokenize as Rust: {error}"))?;
    let mut found = Vec::new();
    walk(stream, &mut found);
    Ok(found)
}

/// Finds lint levels in `stream` and in every group inside it.
fn walk(stream: TokenStream, found: &mut Vec<LintLevel>) {
    let tokens: Vec<TokenTree> = stream.into_iter().collect();
    for (index, token) in tokens.iter().enumerate() {
        match token {
            TokenTree::Ident(name) if LOWERING.contains(&name.to_string().as_str()) => {
                if let Some(TokenTree::Group(args)) = tokens.get(index + 1)
                    && args.delimiter() == Delimiter::Parenthesis
                {
                    level_list(&name.to_string(), args.stream(), found);
                }
            }
            TokenTree::Group(group) => {
                // An attribute whose whole content is a metavariable: `#[$attr]`.
                if group.delimiter() == Delimiter::Bracket
                    && after_hash(&tokens, index)
                    && let Some(TokenTree::Punct(dollar)) = group.stream().into_iter().next()
                    && dollar.as_char() == '$'
                {
                    found.push(LintLevel {
                        level: "attribute".to_owned(),
                        lint: group.stream().to_string().replace(' ', ""),
                        line: dollar.span().start().line,
                    });
                }
                walk(group.stream(), found);
            }
            _ => {}
        }
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

/// The lints of one level's list, `allow(a, b, reason = "...")`.
fn level_list(level: &str, list: TokenStream, found: &mut Vec<LintLevel>) {
    for item in split_commas(list) {
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
            level: level.to_owned(),
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

/// The only key the repository's cargo config may hold: the `cargo xtask` alias that every CI job
/// calls (`.cargo/config.toml`). Its value is pinned by [`xtask_alias_problem`].
const XTASK_ALIAS: &str = "xtask";

/// The linker driver the cargo config may set: the wrapper that puts a provider library's export
/// under its ELF version node (RFC-0040 §10.4; COMP-E9-T19).
const PROVIDER_LINKER: &str = "ci/link/provider-cc";

/// The targets whose `linker` the cargo config may set to [`PROVIDER_LINKER`].
const LINKER_TRIPLES: [&str; 2] = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"];

/// Whether `value`, the `xtask` alias, is `run`, then `--package syntropy-xtask`, `--quiet`,
/// `--locked` and `--target-dir <dir>` in any order, then `--` at the end, and nothing else.
///
/// Any other token could reach clippy through the runner: `--config env.RUSTFLAGS=...` or
/// `--config env.CLIPPY_ARGS=...` set what the lint steps inherit, and `--manifest-path` or `-Z`
/// change what runs. The `--target-dir` value is a plain relative path inside the checkout.
fn xtask_alias_problem(value: &toml::Value) -> Option<String> {
    let Some(text) = value.as_str() else {
        return Some("is not a string".to_owned());
    };
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let (Some(&"run"), Some(&"--")) = (tokens.first(), tokens.last()) else {
        return Some("must start with `run` and end with `--`".to_owned());
    };
    let mut package = false;
    let mut rest = tokens[1..tokens.len() - 1].iter();
    while let Some(&token) = rest.next() {
        match token {
            "--quiet" | "--locked" => {}
            "--package" => {
                if package || rest.next() != Some(&"syntropy-xtask") {
                    return Some("`--package` must name `syntropy-xtask`, once".to_owned());
                }
                package = true;
            }
            "--target-dir" => {
                let plain = |dir: &str| {
                    !dir.is_empty()
                        && !dir.starts_with(['-', '/', '\\'])
                        && !dir.contains("..")
                        && !dir.contains(':')
                        && dir
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '_' | '-'))
                };
                if !rest.next().is_some_and(|&dir| plain(dir)) {
                    return Some("`--target-dir` needs a plain relative path".to_owned());
                }
            }
            other => {
                return Some(format!(
                    "passes `{other}`; only `--package syntropy-xtask`, `--quiet`, `--locked` \
                     and `--target-dir <dir>` may come between `run` and `--`"
                ));
            }
        }
    }
    if package {
        None
    } else {
        Some("does not run `--package syntropy-xtask`".to_owned())
    }
}

/// A repository cargo config file against an allowlist: the `xtask` alias and nothing else.
///
/// Cargo reads this file for every run from the root, and much of it can change how clippy lints
/// the covered crates: `rustflags` that set a lint level (`-A`, `--cap-lints`, `--force-warn`),
/// `[env]` entries that reach rustc and clippy-driver (`CLIPPY_ARGS` with `force = true`),
/// compiler wrappers, a different `rustc`, and an alias that shadows `cargo clippy` itself, an
/// external subcommand that an alias overrides. An allowlist closes them all, including ones not
/// listed here; a change to this file is CI configuration, so the check changes with it. The file
/// is parsed as TOML, so inline tables and quoted or dotted keys count. A file that does not parse
/// is a problem: the check cannot vouch for it.
fn cargo_config_problems(file: &str, text: &str) -> Vec<String> {
    let table: toml::Table = match toml::from_str(text) {
        Ok(table) => table,
        Err(error) => return vec![format!("{file}: does not parse as TOML: {error}")],
    };
    let mut problems = Vec::new();
    let refuse = |key: &str| {
        format!(
            "{file}: `{key}` is outside the allowlist (only `alias.{XTASK_ALIAS}` and the Linux \
             `target.<triple>.linker`): the repository's cargo config reaches every clippy run of \
             the covered crates"
        )
    };
    for (key, value) in &table {
        if key == "target" {
            let Some(targets) = value.as_table() else {
                problems.push(refuse(key));
                continue;
            };
            for (triple, settings) in targets {
                let full = format!("target.{triple}");
                let Some(settings) = settings
                    .as_table()
                    .filter(|_| LINKER_TRIPLES.contains(&triple.as_str()))
                else {
                    problems.push(refuse(&full));
                    continue;
                };
                for (name, setting) in settings {
                    if name != "linker" {
                        problems.push(refuse(&format!("{full}.{name}")));
                    } else if setting.as_str() != Some(PROVIDER_LINKER) {
                        problems.push(format!(
                            "{file}: `{full}.linker` must be `{PROVIDER_LINKER}`, the provider \
                             libraries' link wrapper"
                        ));
                    }
                }
            }
            continue;
        }
        let Some(aliases) = value.as_table().filter(|_| key == "alias") else {
            problems.push(refuse(key));
            continue;
        };
        for (name, command) in aliases {
            let full = format!("alias.{name}");
            if name != XTASK_ALIAS {
                problems.push(refuse(&full));
            } else if let Some(problem) = xtask_alias_problem(command) {
                problems.push(format!(
                    "{file}: `{full}` {problem}; every CI job reaches xtask through it, so its \
                     flags are pinned"
                ));
            }
        }
    }
    problems
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

    /// The levels in `text` that name a lifting lint or a macro-built one.
    fn levels(text: &str) -> Vec<(String, String, usize)> {
        lint_levels(text)
            .unwrap()
            .into_iter()
            .filter(|l| LIFTING.contains(&l.lint.as_str()) || l.lint.contains('$'))
            .map(|l| (l.level, l.lint, l.line))
            .collect()
    }

    /// The levels come from Rust's tokens: no literal or comment can hide or fake one.
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
let s = "#[allow(warnings)]";
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

    /// A macro can carry the attribute or build the lint's name; both are found or refused.
    #[test]
    fn lint_levels_through_macros_are_found_or_refused() {
        let text = "macro_rules! a { ($l:ident) => { #[allow($l)] fn f() {} } }\n\
                    macro_rules! b { ($m:meta) => { #[$m] fn g() {} } }\n\
                    a!(warnings);\nb!(allow(warnings));\nother::m!(expect(clippy::all));\n";
        assert_eq!(
            levels(text),
            [
                ("allow".into(), "$l".into(), 1),
                ("attribute".into(), "$m".into(), 2),
                ("allow".into(), "warnings".into(), 4),
                ("expect".into(), "clippy::all".into(), 5),
            ]
        );
        let files = [file("crates/exec/src/lib.rs", text)];
        let found = source_problems(&files, EXEC_ONLY, &[]);
        for expected in [
            ":1: `allow` takes `$l` from a macro",
            ":2: `attribute` takes `$m` from a macro",
            ":4: `allow(warnings)` lifts",
            ":5: `expect(clippy::all)` lifts",
        ] {
            assert!(
                found.iter().any(|line| line.contains(expected)),
                "{expected}: {found:#?}"
            );
        }
        assert_eq!(found.len(), 4, "{found:#?}");
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

    /// Review B2 and its follow-ups: the repository's cargo config holds the `xtask` alias and
    /// the Linux provider linker and nothing else, read as TOML, so no table, key form or flag
    /// gets past it.
    #[test]
    fn the_cargo_config_holds_only_the_xtask_alias_and_the_linker() {
        let real = std::fs::read_to_string(root().join(".cargo/config.toml")).unwrap();
        assert_eq!(
            cargo_config_problems(".cargo/config.toml", &real),
            Vec::<String>::new()
        );
        let alias = "[alias]\nxtask = \"run --quiet --locked --package syntropy-xtask --\"\n";
        assert_eq!(
            cargo_config_problems(".cargo/config", alias),
            Vec::<String>::new()
        );
        for (text, expected) in [
            (
                "[build]\nrustflags = [\"-Aclippy::disallowed_methods\"]\n",
                "`build` is outside the allowlist",
            ),
            (
                "[build]\nrustflags = [\"-C\", \"target-cpu=native\"]\n",
                "`build` is outside the allowlist",
            ),
            (
                "build = { rustflags = [\"--force-warn\", \"clippy::disallowed_methods\"] }\n",
                "`build` is outside the allowlist",
            ),
            (
                "[target.'cfg(all())']\nrustflags = [\"--cap-lints\", \"allow\"]\n",
                "`target.cfg(all())` is outside the allowlist",
            ),
            (
                "[target.x86_64-unknown-linux-gnu]\nrustflags = [\"--cap-lints\", \"allow\"]\n",
                "`target.x86_64-unknown-linux-gnu.rustflags` is outside the allowlist",
            ),
            (
                "[target.x86_64-unknown-linux-gnu]\nlinker = \"clippy-driver\"\n",
                "`target.x86_64-unknown-linux-gnu.linker` must be `ci/link/provider-cc`",
            ),
            (
                "[target.x86_64-pc-windows-msvc]\nlinker = \"ci/link/provider-cc\"\n",
                "`target.x86_64-pc-windows-msvc` is outside the allowlist",
            ),
            ("target = \"x\"\n", "`target` is outside the allowlist"),
            (
                "[env]\nCLIPPY_ARGS = { value = \"\", force = true }\n",
                "`env` is outside the allowlist",
            ),
            (
                "[alias]\nclippy = \"run --package other --\"\n",
                "`alias.clippy` is outside the allowlist",
            ),
            (
                "alias = { fmt = \"check\" }\n",
                "`alias.fmt` is outside the allowlist",
            ),
            (
                "[alias]\nxtask = \"run --quiet --package other --\"\n",
                "`alias.xtask` `--package` must name `syntropy-xtask`, once",
            ),
            (
                "[alias]\nxtask = \"run --quiet --locked --package syntropy-xtask --target-dir \
                 target/xtask --config env.RUSTFLAGS='-Aclippy::disallowed_methods' --\"\n",
                "`alias.xtask` passes `--config`",
            ),
            (
                "[alias]\nxtask = \"run --package syntropy-xtask --manifest-path o/Cargo.toml --\"\n",
                "`alias.xtask` passes `--manifest-path`",
            ),
            (
                "[alias]\nxtask = \"run --package syntropy-xtask -Zunstable-options --\"\n",
                "`alias.xtask` passes `-Zunstable-options`",
            ),
            (
                "[alias]\nxtask = \"run --package syntropy-xtask --target-dir ../elsewhere --\"\n",
                "`--target-dir` needs a plain relative path",
            ),
            (
                "[alias]\nxtask = \"run --package syntropy-xtask\"\n",
                "must start with `run` and end with `--`",
            ),
            (
                "[alias]\nxtask = \"run --quiet --locked --\"\n",
                "does not run `--package syntropy-xtask`",
            ),
            (
                "[alias]\nxtask = [\"run\", \"--package\", \"syntropy-xtask\", \"--\"]\n",
                "`alias.xtask` is not a string",
            ),
            ("alias = \"x\"\n", "`alias` is outside the allowlist"),
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
