// SPDX-License-Identifier: MIT OR Apache-2.0

//! The property-test framework of RFC-0060 §3.1 and §3.2 (QUAL-E3-T05).
//!
//! A named property has a stable ID (`P-<crate>-<n>`) from RFC-0060 §3.2's table, and
//! `ci/properties.toml` lists every one that has tests, with its crate and the task that owns it.
//! Its tests are the test functions in that crate whose names start with the ID in snake case
//! (`p_core_1_` for P-core-1). They run on proptest's default configuration, which sets the rest:
//!
//! - **Case count.** proptest reads `PROPTEST_CASES`, and uses 256 when it is unset, which is
//!   every PR run. Nightly runs `cargo xtask test properties --cases 16384`, which runs each
//!   listed property with `PROPTEST_CASES` set.
//! - **Regressions.** A failing case shrinks, and its seed goes to a `proptest-regressions/`
//!   file that is committed. For a test under `src/`, proptest writes
//!   `crates/<crate>/proptest-regressions/<path under src>.txt`. For one in an integration-test
//!   directory with a `main.rs`, `tests/<suite>/`, it writes
//!   `crates/<crate>/tests/proptest-regressions/<file>.txt`. A single-file integration test has
//!   neither, and proptest would write beside it with a warning, so a property may not live in
//!   one.
//!
//! The `properties` step of `cargo xtask ci lint` checks:
//! - every entry against RFC-0060 §3.2's table, the workspace and the epic files;
//! - that each property has tests, in an allowed place, in files that set no `proptest_config`
//!   of their own (which could fix the case count or move the regression file);
//! - that every test function named like a property belongs to a listed one.
//!
//! The `properties-removal` job checks that a property the PR removes from the registry has a
//! reason in the PR description, on a line `Removes P-x-n: <reason>` (RFC-0060 §3.2). It reads the
//! description, so like the other checks that judge one (`dco`, `pr-title`, `dep-diff`) it runs in
//! `pr-meta.yml`, from the base branch, again on every edit, and not in `lint`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::licences::toml::{self, Value};
use crate::run::Context;

/// The registry, relative to the workspace root.
pub const REGISTRY: &str = "ci/properties.toml";

/// The RFC whose §3.2 table holds every property ID.
const RFC: &str = "plan/rfcs/RFC-0060-testing-and-correctness.md";

/// The directory of the epic files, where every owning task has its row.
const EPICS: &str = "plan/implementation/epics";

/// The case count of a PR run: proptest's own default (RFC-0060 §1.1).
pub const PR_CASES: u32 = 256;

/// One registry entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Property {
    pub id: String,
    pub crate_name: String,
    pub owner: String,
    /// The line of its `[[property]]` header.
    pub line: usize,
}

impl Property {
    /// The prefix of its test functions' names: `p_core_1_` for P-core-1.
    pub fn prefix(&self) -> String {
        prefix(&self.id)
    }
}

/// `P-core-1` as `p_core_1_`.
pub fn prefix(id: &str) -> String {
    format!("{}_", id.to_ascii_lowercase().replace('-', "_"))
}

/// Whether `id` has the form `P-<letters>-<n>`, with `n` a positive number.
pub fn valid_id(id: &str) -> bool {
    let mut parts = id.split('-');
    let (Some("P"), Some(word), Some(number), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    !word.is_empty()
        && word.chars().all(|c| c.is_ascii_lowercase())
        && !number.is_empty()
        && !number.starts_with('0')
        && number.chars().all(|c| c.is_ascii_digit())
}

/// Whether `id` has the form of a task ID, `QUAL-E3-T05`.
fn valid_task(id: &str) -> bool {
    let parts: Vec<&str> = id.split('-').collect();
    let [area, epic, task] = parts.as_slice() else {
        return false;
    };
    let digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    ["PLC", "TIER", "DIST", "COMP", "PROD", "QUAL"].contains(area)
        && epic.strip_prefix('E').is_some_and(digits)
        && task
            .strip_prefix('T')
            .is_some_and(|n| n.len() == 2 && digits(n))
}

/// Reads the registry. Each problem starts with `ci/properties.toml:<line>`.
pub fn parse(text: &str) -> Result<Vec<Property>, Vec<String>> {
    let tables = toml::tables(text, "property").map_err(|e| vec![format!("{REGISTRY}:{e}")])?;
    let mut properties = Vec::new();
    let mut problems = Vec::new();
    for table in tables {
        let at = format!("{REGISTRY}:{}", table.line);
        for (key, value) in &table.entries {
            if !matches!(key.as_str(), "id" | "crate" | "owner") {
                problems.push(format!(
                    "{at}: unknown key `{key}`; expected id, crate and owner"
                ));
            }
            if !matches!(value, Value::Str(_)) {
                problems.push(format!("{at}: `{key}` is not a string"));
            }
        }
        let field = |key: &str| table.str(key).map(str::to_owned);
        match (field("id"), field("crate"), field("owner")) {
            (Some(id), Some(crate_name), Some(owner)) => properties.push(Property {
                id,
                crate_name,
                owner,
                line: table.line,
            }),
            _ => problems.push(format!("{at}: a property needs id, crate and owner")),
        }
    }
    if problems.is_empty() {
        Ok(properties)
    } else {
        Err(problems)
    }
}

/// The property IDs in RFC-0060 §3.2's table: the first cell of each row that starts `| P-`.
pub fn table_ids(rfc: &str) -> BTreeSet<String> {
    rfc.lines()
        .filter_map(|line| line.strip_prefix("| P-"))
        .filter_map(|rest| rest.split('|').next())
        .map(|id| format!("P-{}", id.trim()))
        .filter(|id| valid_id(id))
        .collect()
}

/// Problems with the entries themselves.
pub fn registry_problems(
    properties: &[Property],
    table: &BTreeSet<String>,
    crates: &BTreeSet<String>,
    epics: &str,
) -> Vec<String> {
    let mut problems = Vec::new();
    let mut seen = BTreeSet::new();
    for property in properties {
        let at = format!("{REGISTRY}:{}: {}", property.line, property.id);
        if !valid_id(&property.id) {
            problems.push(format!("{at}: not a property ID of the form P-<crate>-<n>"));
        } else if !table.contains(&property.id) {
            problems.push(format!("{at}: not in RFC-0060 §3.2's table"));
        }
        if !seen.insert(property.id.as_str()) {
            problems.push(format!("{at}: listed twice"));
        }
        if !crates.contains(&property.crate_name) {
            problems.push(format!(
                "{at}: `{}` is not a crate under crates/",
                property.crate_name
            ));
        }
        if !valid_task(&property.owner) {
            problems.push(format!("{at}: owner `{}` is not a task ID", property.owner));
        } else if !epics.contains(&format!("| {} |", property.owner)) {
            problems.push(format!(
                "{at}: owner `{}` is not a task in the epic files",
                property.owner
            ));
        }
    }
    problems
}

/// The test functions named like a property in `text`, by the ID their names give, with the
/// line each is on. `fn p_core_1_x(` is P-core-1's.
pub fn property_fns(text: &str) -> Vec<(String, usize)> {
    let mut found = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let mut rest = line;
        while let Some(at) = rest.find("fn p_") {
            let before = rest[..at].chars().next_back();
            let name: String = rest[at + 3..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            rest = &rest[at + 3..];
            if before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
                continue;
            }
            let parts: Vec<&str> = name.splitn(4, '_').collect();
            if let ["p", word, number, _] = parts.as_slice() {
                let id = format!("P-{word}-{number}");
                if valid_id(&id) {
                    found.push((id, index + 1));
                }
            }
        }
    }
    found
}

/// Problems with where the properties' tests are. `files` are the Rust files under `crates/`,
/// relative to the root with `/` separators; `suites` are the `crates/<c>/tests/<suite>`
/// directories that hold a `main.rs`.
pub fn source_problems(
    properties: &[Property],
    files: &[(String, String)],
    suites: &BTreeSet<String>,
) -> Vec<String> {
    let listed: BTreeMap<&str, &Property> = properties.iter().map(|p| (p.id.as_str(), p)).collect();
    let mut problems = Vec::new();
    let mut with_tests = BTreeSet::new();
    for (file, text) in files {
        let found = property_fns(text);
        if found.is_empty() {
            continue;
        }
        let crate_name = file
            .strip_prefix("crates/")
            .and_then(|rest| rest.split('/').next())
            .unwrap_or_default();
        let mut file_holds_a_listed_property = false;
        for (id, line) in &found {
            match listed.get(id.as_str()) {
                None => problems.push(format!(
                    "{file}:{line}: a test named like property {id}, which {REGISTRY} does not \
                     list"
                )),
                Some(property) if property.crate_name != crate_name => problems.push(format!(
                    "{file}:{line}: a test of {id}, which {REGISTRY} puts in `{}`",
                    property.crate_name
                )),
                Some(_) => {
                    with_tests.insert(id.clone());
                    file_holds_a_listed_property = true;
                }
            }
        }
        if !file_holds_a_listed_property {
            continue;
        }
        if !allowed_place(file, crate_name, suites) {
            problems.push(format!(
                "{file}: holds a property, but is neither under src/ nor in a tests/<suite>/ \
                 directory with a main.rs, so proptest has no proptest-regressions/ directory \
                 to save a failing seed in"
            ));
        }
        if let Some((index, _)) = text
            .lines()
            .enumerate()
            .find(|(_, line)| line.contains("proptest_config"))
        {
            problems.push(format!(
                "{file}:{}: sets a proptest_config; a file with a property runs on the default \
                 configuration, so PROPTEST_CASES sets its case count",
                index + 1
            ));
        }
    }
    for property in properties {
        if !with_tests.contains(&property.id) {
            problems.push(format!(
                "{REGISTRY}:{}: {} has no test function named `{}...` in crates/{}",
                property.line,
                property.id,
                property.prefix(),
                property.crate_name
            ));
        }
    }
    problems
}

/// Whether a property in `file` gets a `proptest-regressions/` directory from proptest.
fn allowed_place(file: &str, crate_name: &str, suites: &BTreeSet<String>) -> bool {
    let root = format!("crates/{crate_name}/");
    let Some(inside) = file.strip_prefix(&root) else {
        return false;
    };
    if inside.starts_with("src/") {
        return true;
    }
    let mut parts = inside.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some("tests"), Some(suite), Some(_)) => suites.contains(&format!("{root}tests/{suite}")),
        _ => false,
    }
}

/// Properties in `base` that are gone from `head`, each needing a `Removes <id>: <reason>` line
/// in `body`, the PR description.
pub fn removal_problems(
    base: &BTreeSet<String>,
    head: &BTreeSet<String>,
    body: &str,
) -> Vec<String> {
    let reasons: BTreeSet<String> = body
        .lines()
        .filter_map(|line| {
            let line = line.trim().trim_start_matches(['-', '*']).trim_start();
            let rest = line.strip_prefix("Removes ")?;
            let (id, reason) = rest.split_once(':')?;
            (!reason.trim().is_empty()).then(|| id.trim().to_owned())
        })
        .collect();
    base.difference(head)
        .filter(|id| !reasons.contains(*id))
        .map(|id| {
            format!(
                "{REGISTRY}: {id} was removed, and the PR description has no line \
                 `Removes {id}: <reason>` (RFC-0060 §3.2)"
            )
        })
        .collect()
}

/// The `properties` step of `cargo xtask ci lint`: the registry against the code, RFC-0060 and
/// the epic files. Removals are the `properties-removal` job's, which reads the PR description.
pub fn step(context: &Context) -> Result<(), String> {
    let root = &context.root;
    let mut problems = Vec::new();
    let registry = read(root, REGISTRY)?;
    let properties = match parse(&registry) {
        Ok(properties) => properties,
        Err(found) => return Err(found.join("\n")),
    };
    let crates = crate_names(root);
    let epics = epics_text(root)?;
    problems.extend(registry_problems(
        &properties,
        &table_ids(&read(root, RFC)?),
        &crates,
        &epics,
    ));
    let (files, suites) = sources(root, &crates);
    problems.extend(source_problems(&properties, &files, &suites));
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// The property IDs in the registry of the checkout at `dir`: none if it has no registry.
pub fn registry_ids(dir: &Path) -> Result<BTreeSet<String>, String> {
    let path = dir.join(REGISTRY);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeSet::new()),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let properties =
        parse(&text).map_err(|found| format!("{}: {}", path.display(), found.join("; ")))?;
    Ok(properties.into_iter().map(|p| p.id).collect())
}

/// `cargo xtask ci properties-removal --base <dir> --head <dir> --event <file>`: every property
/// in the base checkout's registry that the head checkout's lacks needs a `Removes <ID>: <reason>`
/// line in the PR description (RFC-0060 §3.2). `pr-meta.yml` runs it from the base branch, on
/// every edit of the description too, with the PR checked out as data.
pub fn removal_step(context: &Context) -> Result<(), String> {
    let dir = |option: &str| {
        context
            .args
            .get(option)
            .map(PathBuf::from)
            .ok_or_else(|| format!("`{option}` is required"))
    };
    let base = registry_ids(&dir("--base")?)?;
    let head = registry_ids(&dir("--head")?)?;
    let event_path = context.args.get("--event").ok_or("`--event` is required")?;
    let event = std::fs::read_to_string(event_path)
        .map_err(|error| format!("--event {event_path}: {error}"))?;
    let body = crate::contrib::parse_pull_request(&event)?.body;
    let removed: Vec<&String> = base.difference(&head).collect();
    let problems = removal_problems(&base, &head, &body);
    if problems.is_empty() {
        if removed.is_empty() {
            eprintln!("properties-removal: no property was removed");
        } else {
            eprintln!("properties-removal: removed with a reason: {removed:?}");
        }
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// The suite step of `cargo xtask test properties [--cases <n>]`: each listed property's tests,
/// with `PROPTEST_CASES` set to n (256 by default).
pub fn suite_step(context: &Context) -> Result<(), String> {
    let cases = match context.args.get("--cases") {
        None => PR_CASES,
        Some(text) => text
            .parse::<u32>()
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| format!("--cases `{text}` is not a positive number"))?,
    };
    let properties = parse(&read(&context.root, REGISTRY)?).map_err(|found| found.join("\n"))?;
    let mut problems = Vec::new();
    for property in &properties {
        let prefix = property.prefix();
        let package = format!("syntropy-{}", property.crate_name);
        eprintln!(
            "properties: {} at {cases} cases: cargo test -p {package} --locked -- {prefix}",
            property.id
        );
        // Plain text, so the result lines read the same whatever CI sets.
        let output = Command::new(crate::registry::cargo())
            .args([
                "test", "-p", &package, "--locked", "--color", "never", "--", &prefix,
            ])
            .args(["--color", "never"])
            .env("PROPTEST_CASES", cases.to_string())
            .env("CARGO_TERM_COLOR", "never")
            .current_dir(&context.root)
            .output()
            .map_err(|e| format!("cannot start cargo: {e}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        print!("{stdout}");
        let passed = stdout
            .lines()
            .filter(|line| line.starts_with("test ") && line.ends_with(" ... ok"))
            .filter(|line| {
                let name = line["test ".len()..line.len() - " ... ok".len()].trim();
                name.rsplit("::")
                    .next()
                    .is_some_and(|last| last.starts_with(&prefix))
            })
            .count();
        if !output.status.success() {
            problems.push(format!("{}: failed at {cases} cases", property.id));
        } else if passed == 0 {
            problems.push(format!("{}: no test named `{prefix}...` ran", property.id));
        } else {
            eprintln!(
                "properties: {} passed: {passed} tests at {cases} cases",
                property.id
            );
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

fn read(root: &Path, relative: &str) -> Result<String, String> {
    std::fs::read_to_string(root.join(relative)).map_err(|e| format!("{relative}: {e}"))
}

/// The directories under `crates/` that hold a `Cargo.toml`.
fn crate_names(root: &Path) -> BTreeSet<String> {
    let Ok(entries) = std::fs::read_dir(root.join("crates")) else {
        return BTreeSet::new();
    };
    entries
        .flatten()
        .filter(|entry| entry.path().join("Cargo.toml").is_file())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect()
}

/// Every epic file, joined.
fn epics_text(root: &Path) -> Result<String, String> {
    let dir = root.join(EPICS);
    let entries = std::fs::read_dir(&dir).map_err(|e| format!("{EPICS}: {e}"))?;
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
        .collect();
    paths.sort();
    let mut text = String::new();
    for path in paths {
        text.push_str(&std::fs::read_to_string(&path).map_err(|e| format!("{EPICS}: {e}"))?);
        text.push('\n');
    }
    Ok(text)
}

/// The Rust files of every crate, and the `tests/<suite>` directories that hold a `main.rs`.
fn sources(root: &Path, crates: &BTreeSet<String>) -> (Vec<(String, String)>, BTreeSet<String>) {
    let mut files = Vec::new();
    let mut suites = BTreeSet::new();
    for name in crates {
        let dir = root.join("crates").join(name);
        let mut pending = vec![dir.clone()];
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
                    let text = std::fs::read_to_string(&path).unwrap_or_default();
                    files.push((relative(root, &path), text));
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir(dir.join("tests")) {
            for entry in entries.flatten() {
                if entry.path().join("main.rs").is_file() {
                    suites.insert(relative(root, &entry.path()));
                }
            }
        }
    }
    files.sort();
    (files, suites)
}

/// `path` relative to `root`, with `/` separators on every OS.
fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests;
