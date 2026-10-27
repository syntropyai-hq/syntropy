// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0054 §8's validation beyond layering: the file-format version, the cross-field rules of
//! phase 4, the host checks of phase 5, and [`load_and_validate`], which runs phases 1–5 and
//! keeps every error. Phase 3, each value against its key's type and range, runs as layering
//! assigns the value ([`crate::ResolvedConfig::resolve`]).

use std::fmt;
use std::path::Path;

use crate::code::Code;
use crate::diagnostic::{Problem, render_all};
use crate::discovery::{Os, Scan};
use crate::env::Env;
use crate::origin::Origin;
use crate::registry::Registry;
use crate::resolve::{ConfigFile, Flags, Proposal, Rejected, ResolvedConfig, read_and_resolve};
use crate::schema::{KeySchema, Scope};

/// The key whose unusable path is an error, not a warning (RFC-0054 §8 phase 5).
const STATE_DIR_KEY: &str = "node.state_dir";

/// The newest `config_version` this binary reads (RFC-0054 §1, §8).
pub const CONFIG_VERSION: i64 = 1;

/// What is wrong with a file's `config_version`, as a code and a message, if anything.
///
/// - No `config_version` is version 1.
/// - A version above [`CONFIG_VERSION`] is rejected, naming both versions. In normal use that
///   happens only after an update rollback, which restores the pre-update copy (RFC-0052 §7.4).
/// - A version below 1, or a value that is not an integer, is rejected.
pub fn config_version_problem(value: Option<&toml::Value>) -> Option<(Code, String)> {
    let found = match value {
        None => return None,
        Some(toml::Value::Integer(n)) => *n,
        Some(_) => {
            return Some((
                Code::WrongType,
                "`config_version` must be an integer, such as `config_version = 1`".to_owned(),
            ));
        }
    };
    if found > CONFIG_VERSION {
        return Some((
            Code::OutOfRange,
            format!(
                "this file has `config_version = {found}`, but this Syntropy reads version \
                 {CONFIG_VERSION} and older; it was written by a newer release"
            ),
        ));
    }
    if found < 1 {
        return Some((
            Code::OutOfRange,
            format!("`config_version = {found}` is not a version; versions start at 1"),
        ));
    }
    None
}

/// What a cross-field or environment rule reads: the registry, the resolved configuration, and
/// the host.
pub struct RuleInput<'a> {
    /// The sections.
    pub registry: &'a Registry,
    /// The node's resolved values and its proposals, including the `policy.*` tables.
    pub config: &'a ResolvedConfig,
    /// The host, for phase 5.
    pub environment: &'a dyn Environment,
}

/// What a rule found.
#[derive(Debug, Default)]
pub struct Findings {
    /// Errors: the configuration is rejected.
    pub errors: Vec<Problem>,
    /// Warnings: it loads, and they are printed.
    pub warnings: Vec<Problem>,
}

/// The validation phase a rule belongs to (RFC-0054 §8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    /// Phase 4: rules across keys.
    CrossField,
    /// Phase 5: the host the configuration names.
    Environment,
}

/// A rule of phase 4 or 5: it reads resolved values by path and reports what is wrong. The
/// built-in rules run first; a domain task adds its own through [`load_and_validate`], as
/// PROD-E5-T02 does for the API bind and TLS rules.
#[derive(Clone, Copy)]
pub struct Rule {
    /// A name for tests and logs.
    pub name: &'static str,
    /// When it runs.
    pub phase: Phase,
    /// The check.
    pub check: fn(&RuleInput<'_>, &mut Findings),
}

impl fmt::Debug for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rule")
            .field("name", &self.name)
            .field("phase", &self.phase)
            .finish()
    }
}

/// The rules every load runs, before any a caller adds.
pub const BUILTIN_RULES: &[Rule] = &[
    Rule {
        name: "knob_without_approximate",
        phase: Phase::CrossField,
        check: knob_without_approximate,
    },
    Rule {
        name: "paths_usable",
        phase: Phase::Environment,
        check: paths_usable,
    },
    Rule {
        name: "secret_file_private",
        phase: Phase::Environment,
        check: secret_file_private,
    },
];

/// The host as phase 5 sees it (RFC-0054 §8). [`HostEnvironment`] answers from the standard
/// library; the daemon's implementation, which `serve` (PROD-E6-T02) supplies, also answers
/// direct I/O through `io` and interface names through `net`, which `config` cannot depend on.
pub trait Environment {
    /// Whether `path`, or the directory it would be created in, can be used as `expected`.
    /// - [`PathUse::Directory`]: an existing path is a writable directory, and a missing path's
    ///   nearest existing ancestor is a writable directory it can be created in.
    /// - [`PathUse::Any`]: an existing directory is writable, an existing file is readable,
    ///   and a missing path's nearest existing ancestor is a writable directory.
    ///
    /// The error says why not.
    fn usable(&self, path: &Path, expected: PathUse) -> Result<(), String>;

    /// Whether the file at `path` is readable by its group or others; `None` where the system
    /// has no such permission bits, or the file cannot be read.
    fn readable_by_others(&self, path: &Path) -> Option<bool>;

    /// Whether the volume holding `path` supports direct I/O; `None` if not known.
    fn direct_io(&self, path: &Path) -> Option<bool> {
        let _ = path;
        None
    }

    /// The node's network interface names; `None` if not known.
    fn interfaces(&self) -> Option<Vec<String>> {
        None
    }
}

/// What a path key names, for [`Environment::usable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathUse {
    /// A directory the node writes in, such as `node.state_dir`.
    Directory,
    /// A file or a directory.
    Any,
}

impl PathUse {
    /// What the key at `path` names: a directory when its last segment is `dir` or `dirs`, or
    /// ends in `_dir` or `_dirs`, as `node.state_dir` does; anything otherwise. A key that names
    /// a directory follows that naming, so phase 5 checks it as one.
    pub fn of(path: &str) -> PathUse {
        let name = path.rsplit('.').next().unwrap_or(path);
        let directory =
            ["dir", "dirs"].contains(&name) || name.ends_with("_dir") || name.ends_with("_dirs");
        if directory {
            PathUse::Directory
        } else {
            PathUse::Any
        }
    }
}

/// [`Environment`] from the standard library alone.
///
/// Checking that a directory is writable has a side effect: [`Environment::usable`] creates
/// an empty file, `.syntropy-write-check-<pid>`, in each existing directory a path key names,
/// or in the nearest existing ancestor of a missing one, and removes it at once. The standard
/// library has no check that writes nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostEnvironment;

impl Environment for HostEnvironment {
    fn usable(&self, path: &Path, expected: PathUse) -> Result<(), String> {
        let mut probe = path;
        loop {
            match std::fs::metadata(probe) {
                Ok(meta) if meta.is_dir() => return writable_dir(probe),
                Ok(_) if probe == path && expected == PathUse::Directory => {
                    return Err(String::from("it is a file, not a directory"));
                }
                Ok(_) if probe == path => {
                    return std::fs::File::open(probe)
                        .map(|_| ())
                        .map_err(|e| format!("cannot read it: {e}"));
                }
                Ok(_) => return Err(format!("`{}` is a file, not a directory", probe.display())),
                Err(_) => match probe.parent() {
                    Some(parent) if !parent.as_os_str().is_empty() => probe = parent,
                    _ => return Err("no part of the path exists".to_owned()),
                },
            }
        }
    }

    fn readable_by_others(&self, path: &Path) -> Option<bool> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::metadata(path)
                .ok()
                .map(|meta| meta.permissions().mode() & 0o044 != 0)
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            None
        }
    }
}

/// Whether this process can create a file in `dir`: it creates an empty one, with a name no
/// other file has (`create_new`), and removes it.
fn writable_dir(dir: &Path) -> Result<(), String> {
    let probe = dir.join(format!(".syntropy-write-check-{}", std::process::id()));
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .map_err(|e| format!("cannot write to `{}`: {e}", dir.display()))?;
    let _ = std::fs::remove_file(&probe);
    Ok(())
}

/// Phase 4: an approximate knob, or `default_fidelity = "approximate"`, needs
/// `fidelity = "approximate"` (RFC-0054 §5.4, §8). Each `[policy."<selector>"]` table is read
/// alone, with `[policy.default]`'s `fidelity` counting for every table. Merging the tables by
/// Rule 3 is PROD-E1-T08's.
fn knob_without_approximate(input: &RuleInput<'_>, findings: &mut Findings) {
    let tables: Vec<&Proposal> = input
        .config
        .proposals()
        .iter()
        .filter(|p| p.key.is_none())
        .collect();
    let approximate = |table: &toml::Value| {
        table.get("fidelity").and_then(toml::Value::as_str) == Some("approximate")
    };
    let default_approximate = tables
        .iter()
        .find(|p| p.path == "policy.default")
        .is_some_and(|p| approximate(p.raw_value()));
    for table in tables {
        let mut enabled: Vec<String> = Vec::new();
        let value = table.raw_value();
        if let Some(toml::Value::Table(knobs)) = value.get("approximate") {
            for (knob, value) in knobs {
                if value != &toml::Value::Boolean(false) {
                    enabled.push(format!("approximate.{knob}"));
                }
            }
        }
        if value.get("default_fidelity").and_then(toml::Value::as_str) == Some("approximate") {
            enabled.push(String::from("default_fidelity = \"approximate\""));
        }
        if enabled.is_empty() || approximate(value) || default_approximate {
            continue;
        }
        findings.errors.push(
            Problem::new(
                Code::KnobWithoutApproximate,
                Some(table.origin.clone()),
                format!(
                    "`{}` turns on {}, but its `fidelity` is not \"approximate\"; a model is \
                     approximate only where its policy says so (RFC-0054 §5.4)",
                    table.path,
                    enabled.join(" and ")
                ),
            )
            .about(format!("{}.fidelity", table.path))
            .hint(Some(String::from(
                "add `fidelity = \"approximate\"` to this table, or to `[policy.default]`",
            ))),
        );
    }
}

/// Whether a key's type is a path or holds paths.
fn path_typed(key: &KeySchema) -> bool {
    key.ty.contains("PathBuf")
}

/// The paths a value holds: a string, or the strings of an array.
fn paths_in(value: &toml::Value) -> Vec<&str> {
    match value {
        toml::Value::String(s) => vec![s.as_str()],
        toml::Value::Array(items) => items.iter().filter_map(toml::Value::as_str).collect(),
        _ => Vec::new(),
    }
}

/// Phase 5: every path a node key names exists, or can be created, and can be used, as a
/// writable directory where the key names one ([`PathUse::of`]). An unusable `node.state_dir`
/// is an error; any other path is a warning (RFC-0054 §8).
fn paths_usable(input: &RuleInput<'_>, findings: &mut Findings) {
    for resolved in input.config.iter() {
        if resolved.key.scope != Scope::Node || !path_typed(resolved.key) {
            continue;
        }
        let Some(value) = resolved.value() else {
            continue;
        };
        for path in paths_in(value) {
            let expected = PathUse::of(resolved.key.path);
            if let Err(why) = input.environment.usable(Path::new(path), expected) {
                let problem = Problem::new(
                    Code::PathUnusable,
                    Some(resolved.origin.clone()),
                    format!(
                        "`{}` names `{path}`, which cannot be used: {why}",
                        resolved.key.path
                    ),
                )
                .about(resolved.key.path);
                if resolved.key.path == STATE_DIR_KEY {
                    findings.errors.push(problem);
                } else {
                    findings.warnings.push(problem);
                }
            }
        }
    }
}

/// Whether a value holds a secret written inline, rather than as `{ file = … }` or
/// `{ env = … }`.
fn holds_inline_secret(value: &toml::Value) -> bool {
    match value {
        toml::Value::String(_) => true,
        toml::Value::Table(table) => {
            let reference =
                table.len() == 1 && (table.contains_key("file") || table.contains_key("env"));
            !reference && table.values().any(holds_inline_secret)
        }
        toml::Value::Array(items) => items.iter().any(holds_inline_secret),
        _ => false,
    }
}

/// Phase 5: a config file that holds a secret inline is not readable by its group or others
/// (a warning on Unix; RFC-0054 §8, §10.1). The file and variable forms are not flagged.
fn secret_file_private(input: &RuleInput<'_>, findings: &mut Findings) {
    let mut flagged: Vec<&Path> = Vec::new();
    let sources = input
        .config
        .iter()
        .filter(|r| r.key.holds_secret)
        .filter_map(|r| Some((r.key.path, r.raw_value()?, &r.origin)))
        .chain(
            input
                .config
                .proposals()
                .iter()
                .filter(|p| p.key.is_some_and(|k| k.holds_secret))
                .map(|p| (p.path.as_str(), p.raw_value(), &p.origin)),
        );
    for (key, value, origin) in sources {
        let Origin::File { path, .. } = origin else {
            continue;
        };
        if !holds_inline_secret(value) || flagged.contains(&path.as_path()) {
            continue;
        }
        if input.environment.readable_by_others(path) == Some(true) {
            flagged.push(path);
            findings.warnings.push(
                Problem::uncoded(
                    Some(origin.clone()),
                    format!(
                        "`{}` holds the secret `{key}`, and its group or others can read it",
                        path.display()
                    ),
                )
                .about(key)
                .hint(Some(String::from(
                    "restrict the file (`chmod 600`), or move the secret to `{ file = … }` or \
                     `{ env = … }`",
                ))),
            );
        }
    }
}

/// One load and validation: the files read, and what they resolved to or why not. It keeps the
/// files' text so every problem renders with its source line.
#[derive(Debug)]
pub struct Checked {
    /// The files of the config layer, in merge order.
    pub files: Vec<ConfigFile>,
    /// The configuration, with every warning, or every error and warning found.
    pub outcome: Result<ResolvedConfig, Rejected>,
}

impl Checked {
    /// Every error, in the order found.
    pub fn errors(&self) -> &[Problem] {
        match &self.outcome {
            Ok(_) => &[],
            Err(rejected) => &rejected.errors,
        }
    }

    /// Every warning, in the order found.
    pub fn warnings(&self) -> &[Problem] {
        match &self.outcome {
            Ok(config) => config.warnings(),
            Err(rejected) => &rejected.warnings,
        }
    }

    /// Every error, then every warning, in RFC-0054 §8's format, never showing a secret.
    ///
    /// The text quotes the configuration's own sources: a value from a file, the environment or
    /// a flag, and an unknown variable's name, any of which can hold a control character such as
    /// ESC. A caller that prints it to a terminal escapes it first. The CLI's `config show` and
    /// `config check` do, line by line, and so must the daemon's `serve` startup report
    /// (PROD-E6-T02). This crate is layer 0 and leaves the escaping to its callers, which share
    /// `syntropy_api::printable`.
    pub fn render(&self, registry: &Registry) -> String {
        render_all(self.errors(), self.warnings(), &self.files, registry)
    }
}

/// Reads the files `scan` found, resolves them with the environment and flags, and validates the
/// result: RFC-0054 §8's phases 1–5, which `serve` and `config check` run (PROD-E6-T02,
/// PROD-E6-T23). It never stops at the first error: phases 1–3 report together, and phases 4 and
/// 5 run once those pass, so a start-up prints every error it can find.
///
/// `computed` supplies the computed defaults, as for [`ResolvedConfig::resolve_with`]. `rules`
/// are the callers' own phase 4 and 5 rules, run after [`BUILTIN_RULES`].
#[allow(clippy::too_many_arguments)]
pub fn load_and_validate(
    registry: &Registry,
    os: Os,
    scan: &Scan,
    env: &Env,
    flags: &Flags,
    computed: impl Fn(&'static KeySchema) -> Option<toml::Value>,
    rules: &[Rule],
    environment: &dyn Environment,
) -> Checked {
    let (files, outcome) = read_and_resolve(registry, os, scan, env, flags, computed);
    let outcome = outcome.and_then(|config| validate(registry, config, rules, environment));
    Checked { files, outcome }
}

/// Runs phases 4 and 5 over a resolved configuration: the built-in rules, then `rules`, phase by
/// phase.
pub fn validate(
    registry: &Registry,
    mut config: ResolvedConfig,
    rules: &[Rule],
    environment: &dyn Environment,
) -> Result<ResolvedConfig, Rejected> {
    let mut findings = Findings::default();
    let mut all: Vec<&Rule> = BUILTIN_RULES.iter().chain(rules).collect();
    all.sort_by_key(|rule| rule.phase);
    {
        let input = RuleInput {
            registry,
            config: &config,
            environment,
        };
        for rule in all {
            (rule.check)(&input, &mut findings);
        }
    }
    if findings.errors.is_empty() {
        config.push_warnings(findings.warnings);
        Ok(config)
    } else {
        let mut warnings = config.warnings().to_vec();
        warnings.extend(findings.warnings);
        Err(Rejected {
            errors: findings.errors,
            warnings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_or_current_versions_pass() {
        assert_eq!(config_version_problem(None), None);
        assert_eq!(config_version_problem(Some(&toml::Value::Integer(1))), None);
    }

    #[test]
    fn a_newer_version_names_both() {
        let (code, message) = config_version_problem(Some(&toml::Value::Integer(2))).unwrap();
        assert_eq!(code, Code::OutOfRange);
        assert!(message.contains("config_version = 2"), "{message}");
        assert!(message.contains("reads version 1"), "{message}");
    }

    #[test]
    fn a_directory_key_needs_a_directory() {
        let dir = std::env::temp_dir().join(format!("syntropy-usable-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let file = dir.join("f");
        std::fs::write(&file, "").expect("write");
        let host = HostEnvironment;
        assert!(host.usable(&file, PathUse::Directory).is_err());
        assert_eq!(host.usable(&file, PathUse::Any), Ok(()));
        assert_eq!(host.usable(&dir, PathUse::Directory), Ok(()));
        assert_eq!(
            host.usable(&dir.join("new/sub"), PathUse::Directory),
            Ok(())
        );
        assert!(
            host.usable(&file.join("under-a-file"), PathUse::Directory)
                .is_err()
        );
        assert_eq!(std::fs::read_dir(&dir).expect("read dir").count(), 1);
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn directory_keys_are_named_as_directories() {
        let named = PathUse::of;
        assert_eq!(named("node.state_dir"), PathUse::Directory);
        assert_eq!(named("store.dirs"), PathUse::Directory);
        assert_eq!(named("store.cache_dirs"), PathUse::Directory);
        assert_eq!(named("api.tls.cert"), PathUse::Any);
        assert_eq!(named("node.dirty"), PathUse::Any);
    }

    #[test]
    fn other_values_are_rejected() {
        let wrong = config_version_problem(Some(&toml::Value::String("1".into())));
        assert_eq!(wrong.map(|(code, _)| code), Some(Code::WrongType));
        let zero = config_version_problem(Some(&toml::Value::Integer(0)));
        assert_eq!(zero.map(|(code, _)| code), Some(Code::OutOfRange));
    }
}
