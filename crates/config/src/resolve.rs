// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rule 1 of RFC-0054 §2.2: node keys resolve through
//! `default → config file (orders 1–4 of §1) → environment → CLI`, later layers winning, and
//! every resolved value carries its [`Origin`].
//!
//! [`ResolvedConfig::resolve`] is a pure function of the registry and the layers, so the
//! precedence is tested without a file system. [`ConfigFile::read_all`] reads the files a
//! [`Scan`] found.
//!
//! What a node's layers hold for other scopes is kept, not resolved:
//! - a cluster key, or a `[policy."<selector>"]` table, is a *proposal* (§2.3). It is listed in
//!   [`ResolvedConfig::proposals`], and a cluster key's effective value stays its default here;
//!   the event log supplies it once the cluster scope is wired.
//! - an instance key outside a `policy.*` table, or any request key, is an error.
//!
//! This module reports what layering itself finds: syntax errors with line and column, unknown
//! keys, keys in the wrong scope, unreadable files and variables, each with its RFC-0054 §8
//! code. It also runs §8's phase 3 as it goes: each value is checked against its key's type and
//! range when a layer sets it, so phases 1–3 report in one list. Phases 4 and 5 are in
//! [`crate::validate`], and the rendering in [`crate::diagnostic`].

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use serde::de::DeserializeOwned;
use toml::Value;

use crate::code::{Code, suggest};
use crate::diagnostic::Problem;
use crate::discovery::{HOME_VAR, Mode, Os, Scan, locate};
use crate::env::{self, Env, FIXED, OTHER_KNOWN};
use crate::origin::Origin;
use crate::registry::Registry;
use crate::schema::{ConfigSection, DefaultSpec, DefaultValue, Deprecated, KeySchema, Scope};
use crate::text;
use crate::validate::config_version_problem;
use crate::value::ValueError;

/// The key `SYNTROPY_HOME` sets (RFC-0054 §2.4).
const STATE_DIR_KEY: &str = "node.state_dir";
/// The file-format version key every file may carry (RFC-0054 §1).
const CONFIG_VERSION_KEY: &str = "config_version";
/// The top-level table that holds the model policy tables (RFC-0054 §5.4).
const POLICY_TABLE: &str = "policy";

/// A config file's path and text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigFile {
    /// The file.
    pub path: PathBuf,
    /// Its text, without a leading byte-order mark.
    pub text: String,
}

impl ConfigFile {
    /// A file of `text`. A leading byte-order mark is dropped (RFC-0054 §1).
    pub fn new(path: impl Into<PathBuf>, text: impl Into<String>) -> Self {
        let mut text = text.into();
        if text.starts_with('\u{feff}') {
            text.drain(..'\u{feff}'.len_utf8());
        }
        Self {
            path: path.into(),
            text,
        }
    }

    /// Reads every file `scan` found, in merge order. Returns every file that cannot be read
    /// or is not UTF-8.
    pub fn read_all(scan: &Scan) -> Result<Vec<ConfigFile>, Vec<Problem>> {
        let (files, unreadable) = Self::read_each(scan);
        if unreadable.is_empty() {
            Ok(files)
        } else {
            Err(unreadable.into_iter().map(|(_, problem)| problem).collect())
        }
    }

    /// Reads every file `scan` found, in merge order: the files that could be read, and for
    /// each one that cannot be read or is not UTF-8, its path and the problem.
    fn read_each(scan: &Scan) -> (Vec<ConfigFile>, Vec<(PathBuf, Problem)>) {
        let mut files = Vec::new();
        let mut unreadable = Vec::new();
        for path in scan.files() {
            match std::fs::read(&path) {
                Ok(bytes) => match String::from_utf8(bytes) {
                    Ok(text) => files.push(ConfigFile::new(path, text)),
                    Err(_) => {
                        let problem = Problem::new(
                            Code::PathUnusable,
                            None,
                            format!("{}: the file is not UTF-8", path.display()),
                        );
                        unreadable.push((path, problem));
                    }
                },
                Err(e) => {
                    let problem = Problem::new(
                        Code::PathUnusable,
                        None,
                        format!("{}: cannot read the file: {e}", path.display()),
                    );
                    unreadable.push((path, problem));
                }
            }
        }
        (files, unreadable)
    }
}

/// Reads the files `scan` found and resolves the ones that can be read, so a file that cannot
/// be read is reported beside every other file's errors, not alone (RFC-0054 §8: every error,
/// not only the first). Returns the files read, for rendering, and the outcome. The errors stay
/// in merge order: each file's in turn, a file that cannot be read in its place, then the
/// environment's and the flags'.
pub(crate) fn read_and_resolve(
    registry: &Registry,
    os: Os,
    scan: &Scan,
    env: &Env,
    flags: &Flags,
    computed: impl Fn(&'static KeySchema) -> Option<Value>,
) -> (Vec<ConfigFile>, Result<ResolvedConfig, Rejected>) {
    let (files, unreadable) = ConfigFile::read_each(scan);
    let layers = Layers {
        os,
        files: &files,
        env,
        flags,
    };
    let outcome = ResolvedConfig::resolve_with(registry, layers, computed);
    if unreadable.is_empty() {
        return (files, outcome);
    }
    let (errors, warnings) = match outcome {
        Ok(config) => (Vec::new(), config.warnings().to_vec()),
        Err(rejected) => (rejected.errors, rejected.warnings),
    };
    let order = scan.files();
    let place = |path: &Path| order.iter().position(|p| p == path).unwrap_or(order.len());
    let mut errors: Vec<(usize, Problem)> = unreadable
        .into_iter()
        .map(|(path, problem)| (place(&path), problem))
        .chain(errors.into_iter().map(|problem| {
            let at = match &problem.at {
                Some(Origin::File { path, .. }) => place(path),
                _ => order.len(),
            };
            (at, problem)
        }))
        .collect();
    errors.sort_by_key(|(at, _)| *at);
    let errors = errors.into_iter().map(|(_, problem)| problem).collect();
    (files, Err(Rejected { errors, warnings }))
}

/// One value set on the command line.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FlagValue {
    path: String,
    text: String,
    flag: String,
}

/// The CLI layer: `--set key=value` (repeatable, any key) and the values of the dedicated flags
/// RFC-0051 §2.3 lists, in the order given. A later flag for the same key wins. `--config` is
/// not here: it chooses the files, through [`locate`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    values: Vec<FlagValue>,
}

impl Flags {
    /// No flags.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one `--set` argument, `key=value`. The key is a dotted path of bare TOML keys; the
    /// value is read as the environment's values are. Whether the key exists is checked when
    /// the layers resolve, and an unknown key is an error there (RFC-0054 §2.4).
    pub fn set(&mut self, arg: &str) -> Result<(), Problem> {
        let at = Some(Origin::Cli {
            flag: String::from("--set"),
        });
        // The messages never echo the argument: any part of it may be a secret. A well-formed
        // key is named later, when it resolves.
        let Some((key, value)) = arg.split_once('=') else {
            return Err(Problem::new(
                Code::Syntax,
                at,
                "an argument has no `=`: `--set` takes `key=value`",
            ));
        };
        let key = key.trim();
        let well_formed = !key.is_empty()
            && key
                .split('.')
                .all(|s| !s.is_empty() && crate::schema::is_bare_key(s));
        if !well_formed {
            // Not echoed either: text before the first `=` that is not a key may be a value
            // written with the wrong separator (`--set 'k:Bearer …=x'`).
            return Err(Problem::new(
                Code::Syntax,
                at,
                "the text before `=` is not a key: `--set` takes a dotted path of bare keys, \
                 such as `resource.ram_headroom`, and cannot set a `policy.*` table",
            ));
        }
        self.values.push(FlagValue {
            path: key.to_owned(),
            text: value.to_owned(),
            flag: format!("--set {key}"),
        });
        Ok(())
    }

    /// Adds the value of a dedicated flag, such as `--log-level debug` for `logging.level`.
    /// `text` is read as an environment value is.
    pub fn push(
        &mut self,
        flag: impl Into<String>,
        path: impl Into<String>,
        text: impl Into<String>,
    ) {
        self.values.push(FlagValue {
            path: path.into(),
            text: text.into(),
            flag: flag.into(),
        });
    }

    /// Whether no flag sets a value.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Everything Rule 1 reads, in precedence order after the defaults.
#[derive(Clone, Copy, Debug)]
pub struct Layers<'a> {
    /// The OS, for `SYNTROPY_HOME`'s path rules.
    pub os: Os,
    /// The files of the config layer, in merge order: [`Scan::files`].
    pub files: &'a [ConfigFile],
    /// The environment snapshot.
    pub env: &'a Env,
    /// The command-line values.
    pub flags: &'a Flags,
}

/// One key's effective value.
#[derive(Clone)]
pub struct Resolved {
    /// The key.
    pub key: &'static KeySchema,
    /// The value. `None` for an `Option` key left unset, or a computed default the caller did
    /// not supply. Private, because a secret-typed key's value is a secret outside `Secret`:
    /// see [`Resolved::value`].
    value: Option<Value>,
    /// The layer that set it.
    pub origin: Origin,
}

impl Resolved {
    /// The value, for a key whose type holds no secret. A secret-typed key gives `None` here:
    /// its value leaves only inside `Secret`, through [`ResolvedConfig::section`], or redacted,
    /// through [`Resolved::shown`] (RFC-0054 §10.1).
    pub fn value(&self) -> Option<&Value> {
        if text::is_secret(self.key) {
            None
        } else {
            self.value.as_ref()
        }
    }

    /// Whether a layer or the default gave the key a value.
    pub fn is_set(&self) -> bool {
        self.value.is_some()
    }

    /// The value, a secret-typed key's included, for the crate's own checks, which never print
    /// it, such as whether a secret is written inline (RFC-0054 §8 phase 5).
    pub(crate) fn raw_value(&self) -> Option<&Value> {
        self.value.as_ref()
    }

    /// The value as `config show` prints it: inline TOML, with secrets redacted.
    pub fn shown(&self) -> Option<String> {
        self.value.as_ref().map(|v| text::shown(self.key, v))
    }
}

/// Keys compare by path: the registry holds one schema per path.
impl PartialEq for Resolved {
    fn eq(&self, other: &Self) -> bool {
        self.key.path == other.key.path && self.value == other.value && self.origin == other.origin
    }
}

/// Redacts secrets, as `config show` does.
impl fmt::Debug for Resolved {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Resolved")
            .field("key", &self.key.path)
            .field("value", &self.shown())
            .field("origin", &self.origin)
            .finish()
    }
}

/// A cluster value a node's file, environment or flags propose (RFC-0054 §2.3). The proposal
/// itself is submitted once the cluster scope is wired.
#[derive(Clone)]
pub struct Proposal {
    /// The key as written, with selectors quoted: `allow_approximate`, `policy."qwen3*"`.
    pub path: String,
    /// The cluster key, or `None` for a `policy.*` table.
    pub key: Option<&'static KeySchema>,
    /// The proposed value. Private for the same reason as [`Resolved`]'s.
    value: Value,
    /// The value as `config show` prints it, computed with the registry when the proposal is
    /// made, so a `policy.*` table is redacted entry by entry.
    shown: String,
    /// Whether `shown` redacts anything.
    redacted: bool,
    /// The layer that set it.
    pub origin: Origin,
}

impl Proposal {
    /// A cluster key's proposal.
    fn cluster(key: &'static KeySchema, value: Value, origin: Origin) -> Self {
        Self {
            path: key.path.to_owned(),
            key: Some(key),
            shown: text::shown(key, &value),
            redacted: text::is_secret(key),
            value,
            origin,
        }
    }

    /// A `policy."<selector>"` table's proposal. Each entry is redacted unless it is an
    /// instance key whose type holds no secret ([`text::policy_shown`]).
    fn policy(path: String, registry: &Registry, value: Value, origin: Origin) -> Self {
        let (shown, redacted) = text::policy_shown(registry, &value);
        Self {
            path,
            key: None,
            value,
            shown,
            redacted,
            origin,
        }
    }

    /// The proposed value, unless any of it is redacted in `config show`: a secret-typed
    /// cluster key, or a `policy.*` table with a secret-typed or unknown entry (see
    /// [`Resolved::value`]).
    pub fn value(&self) -> Option<&Value> {
        if self.redacted {
            None
        } else {
            Some(&self.value)
        }
    }

    /// The proposed value, a secret's included, for the crate's own checks (see
    /// [`Resolved::raw_value`]).
    pub(crate) fn raw_value(&self) -> &Value {
        &self.value
    }

    /// The value as `config show` prints it, with secrets redacted.
    pub fn shown(&self) -> String {
        self.shown.clone()
    }
}

impl PartialEq for Proposal {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.value == other.value && self.origin == other.origin
    }
}

/// Redacts secrets, as `config show` does.
impl fmt::Debug for Proposal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Proposal")
            .field("path", &self.path)
            .field("value", &self.shown())
            .field("origin", &self.origin)
            .finish()
    }
}

/// Why the layers did not resolve: every error, and the warnings found on the way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejected {
    /// The errors, in the order found: files in merge order, then the environment, then flags.
    pub errors: Vec<Problem>,
    /// The warnings.
    pub warnings: Vec<Problem>,
}

/// The node's effective configuration: one immutable snapshot per process (RFC-0001 §11.2).
/// A reload builds a new one and swaps it in whole, through [`SharedConfig`].
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedConfig {
    /// Every node and cluster key of the registry, by path.
    values: BTreeMap<&'static str, Resolved>,
    proposals: Vec<Proposal>,
    files: Vec<PathBuf>,
    warnings: Vec<Problem>,
}

/// A section that could not be built from the resolved values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectionError {
    /// The section's type.
    pub section: &'static str,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for SectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "config section `{}`: {}", self.section, self.message)
    }
}

impl std::error::Error for SectionError {}

impl ResolvedConfig {
    /// Resolves `layers` against `registry` with Rule 1. Computed defaults stay unset; see
    /// [`ResolvedConfig::resolve_with`].
    pub fn resolve(registry: &Registry, layers: Layers<'_>) -> Result<Self, Rejected> {
        Self::resolve_with(registry, layers, |_| None)
    }

    /// Resolves `layers`, asking `computed` for the value of each computed default no layer
    /// overrides. The daemon supplies them from the probed hardware; a key it returns `None`
    /// for stays unset, with `Origin::Default { computed }` naming the rule.
    pub fn resolve_with(
        registry: &Registry,
        layers: Layers<'_>,
        computed: impl Fn(&'static KeySchema) -> Option<Value>,
    ) -> Result<Self, Rejected> {
        let mut r = Resolution::new(registry);
        for file in layers.files {
            r.file(file);
        }
        r.env(layers.os, layers.env);
        r.flags(layers.flags);
        if !r.errors.is_empty() {
            return Err(Rejected {
                errors: r.errors,
                warnings: r.warnings,
            });
        }
        let mut values = BTreeMap::new();
        for key in registry.keys() {
            if !matches!(key.scope, Scope::Node | Scope::Cluster) {
                continue;
            }
            let resolved = match r.set.remove(key.path) {
                Some((value, origin)) => Resolved {
                    key,
                    value: Some(value),
                    origin,
                },
                None => default_of(key, &computed),
            };
            values.insert(key.path, resolved);
        }
        Ok(Self {
            values,
            proposals: r.proposals.into_values().collect(),
            files: layers.files.iter().map(|f| f.path.clone()).collect(),
            warnings: r.warnings,
        })
    }

    /// The effective value of the node or cluster key at `path`.
    pub fn get(&self, path: &str) -> Option<&Resolved> {
        self.values.get(path)
    }

    /// Every node and cluster key, in path order.
    pub fn iter(&self) -> impl Iterator<Item = &Resolved> {
        self.values.values()
    }

    /// The cluster values this node's layers propose, in path order.
    pub fn proposals(&self) -> &[Proposal] {
        &self.proposals
    }

    /// The config files read, in merge order.
    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }

    /// The warnings found while loading, such as an unknown `SYNTROPY_*` variable.
    pub fn warnings(&self) -> &[Problem] {
        &self.warnings
    }

    /// Adds the warnings of the later validation phases.
    pub(crate) fn push_warnings(&mut self, warnings: Vec<Problem>) {
        self.warnings.extend(warnings);
    }

    /// Builds section `S` from the effective values. A key of another scope in `S` takes its
    /// fixed default. A range check runs on the result.
    pub fn section<S: ConfigSection>(&self) -> Result<S, SectionError> {
        let schema = &S::SCHEMA;
        let error = |message: String| SectionError {
            section: schema.type_name,
            message,
        };
        let mut table = toml::Table::new();
        for key in schema.keys {
            let value = match self.values.get(key.path) {
                Some(resolved) => resolved.value.clone(),
                None => match key.default {
                    DefaultSpec::Value(v) => Some(to_value(&v)),
                    DefaultSpec::Unset | DefaultSpec::Computed(_) => None,
                },
            };
            match value {
                Some(value) => {
                    table.insert(key.name.to_owned(), value);
                }
                None if matches!(key.default, DefaultSpec::Computed(_)) => {
                    return Err(error(format!(
                        "`{}` has a computed default that was not supplied",
                        key.path
                    )));
                }
                None => {}
            }
        }
        let section: S = match deserialize(Value::Table(table.clone())) {
            Ok(section) => section,
            Err(message) => return Err(error(withhold_secrets::<S>(table, message))),
        };
        let range_errors = section.check_ranges();
        if let Some(first) = range_errors.first() {
            // A range error's text can quote the value, so a secret key's is withheld too.
            let message = match schema.keys.iter().find(|k| k.path == first.key) {
                Some(key) if key.holds_secret => {
                    format!(
                        "`{}`: out of range; its value is withheld because it holds a secret",
                        key.path
                    )
                }
                _ => first.to_string(),
            };
            return Err(error(message));
        }
        Ok(section)
    }
}

/// The error to report when section `S` does not deserialize from `table`. serde's message can
/// quote the value it rejected (`invalid type: string "…"`), and for a key that holds a secret
/// the value is the secret (RFC-0054 §10.1, G7). So when the section has secret-typed keys, it
/// deserializes again with each of them at its schema default:
/// - if that still fails, a key that holds no secret is at fault, and its message is shown;
/// - otherwise a secret-typed key is, and the message names the section's secret-typed keys
///   with no serde text.
fn withhold_secrets<S: ConfigSection>(table: toml::Table, message: String) -> String {
    let secrets: Vec<&KeySchema> = S::SCHEMA.keys.iter().filter(|k| k.holds_secret).collect();
    if secrets.is_empty() {
        return message;
    }
    let mut without = table;
    for key in &secrets {
        match key.default {
            DefaultSpec::Value(v) => {
                without.insert(key.name.to_owned(), to_value(&v));
            }
            DefaultSpec::Unset | DefaultSpec::Computed(_) => {
                without.remove(key.name);
            }
        }
    }
    let paths: Vec<String> = secrets.iter().map(|k| format!("`{}`", k.path)).collect();
    match deserialize::<S>(Value::Table(without)) {
        Err(other) => other,
        Ok(_) => format!(
            "{}: a value has the wrong type or shape; it is withheld because the key holds a \
             secret",
            paths.join(", ")
        ),
    }
}

fn deserialize<T: DeserializeOwned>(value: Value) -> Result<T, String> {
    value
        .try_into()
        .map_err(|e: toml::de::Error| e.message().to_owned())
}

fn default_of(
    key: &'static KeySchema,
    computed: &impl Fn(&'static KeySchema) -> Option<Value>,
) -> Resolved {
    let (value, rule) = match key.default {
        DefaultSpec::Value(v) => (Some(to_value(&v)), None),
        DefaultSpec::Unset => (None, None),
        DefaultSpec::Computed(rule) => (computed(key), Some(rule)),
    };
    Resolved {
        key,
        value,
        origin: Origin::Default { computed: rule },
    }
}

/// A schema default as a TOML value.
fn to_value(value: &DefaultValue) -> Value {
    match *value {
        DefaultValue::Bool(b) => Value::Boolean(b),
        DefaultValue::Int(i) => Value::Integer(i),
        DefaultValue::Float(x) => Value::Float(x),
        DefaultValue::Str(s) => Value::String(s.to_owned()),
        DefaultValue::Array(items) => Value::Array(items.iter().map(to_value).collect()),
        DefaultValue::Table(entries) => Value::Table(
            entries
                .iter()
                .map(|(k, v)| ((*k).to_owned(), to_value(v)))
                .collect(),
        ),
    }
}

/// Reads an environment or flag value: a TOML value, or the text itself as a string where it
/// does not parse as one. A key whose type is a string or a path always takes the text as
/// written, unless it is a quoted TOML string, so `SYNTROPY_NODE__NAME=42` names the node `42`
/// (RFC-0054 §2.4: "a bare string without quotes is accepted for string-typed keys").
fn read_text(key: &KeySchema, text: &str) -> Value {
    let quoted = text.starts_with('"') || text.starts_with('\'');
    if is_string_typed(key) && !quoted {
        return Value::String(text.to_owned());
    }
    text.parse::<Value>()
        .unwrap_or_else(|_| Value::String(text.to_owned()))
}

fn is_string_typed(key: &KeySchema) -> bool {
    let ty = key.ty.replace(' ', "");
    let inner = ty
        .strip_prefix("Option<")
        .and_then(|t| t.strip_suffix('>'))
        .unwrap_or(&ty);
    matches!(
        inner,
        "String" | "PathBuf" | "std::path::PathBuf" | "std::string::String"
    )
}

/// The layers as they resolve: the last value set per key, proposals, errors and warnings.
struct Resolution<'r> {
    registry: &'r Registry,
    set: BTreeMap<&'static str, (Value, Origin)>,
    proposals: BTreeMap<String, Proposal>,
    errors: Vec<Problem>,
    warnings: Vec<Problem>,
}

impl<'r> Resolution<'r> {
    fn new(registry: &'r Registry) -> Self {
        Self {
            registry,
            set: BTreeMap::new(),
            proposals: BTreeMap::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Whether some registered key lives under `path`.
    fn is_table(&self, path: &str) -> bool {
        self.registry.keys().any(|k| {
            k.path.len() > path.len()
                && k.path.starts_with(path)
                && k.path.as_bytes()[path.len()] == b'.'
        })
    }

    /// Records `value` for `key` from a layer, by the key's scope. `None` means the scope does
    /// not belong in a node's layers; the caller reports it.
    fn assign(&mut self, key: &'static KeySchema, value: Value, origin: Origin) -> Option<()> {
        if matches!(key.scope, Scope::Node | Scope::Cluster)
            && let Err(error) = (key.check)(&value)
        {
            self.errors.push(value_problem(key, &error, origin.clone()));
        }
        match key.scope {
            Scope::Node => {
                self.set.insert(key.path, (value, origin));
                Some(())
            }
            Scope::Cluster => {
                self.proposals
                    .insert(key.path.to_owned(), Proposal::cluster(key, value, origin));
                Some(())
            }
            Scope::Instance | Scope::Request => None,
        }
    }

    fn file(&mut self, file: &ConfigFile) {
        let lines = LineIndex::new(&file.text);
        let syntax = |e: toml::de::Error, problems: &mut Vec<Problem>| {
            let span = e.span();
            let (line, column) = span
                .as_ref()
                .map(|s| lines.line_column(s.start))
                .unwrap_or((1, 1));
            let width = span
                .and_then(|s| file.text.get(s))
                .map_or(1, |t| t.chars().take_while(|&c| c != '\n').count() as u32);
            problems.push(
                Problem::new(
                    Code::Syntax,
                    Some(Origin::File {
                        path: file.path.clone(),
                        line,
                    }),
                    // Only the message: the library's rendered error quotes the source line.
                    format!("TOML syntax: {}", e.message().trim_end()),
                )
                .spanning(column, width),
            );
        };
        let spans = match toml::de::DeTable::parse(&file.text) {
            Ok(table) => key_offsets(table.get_ref()),
            Err(e) => return syntax(e, &mut self.errors),
        };
        let table: toml::Table = match toml::from_str(&file.text) {
            Ok(table) => table,
            Err(e) => return syntax(e, &mut self.errors),
        };
        let at = |segments: &[String]| Origin::File {
            path: file.path.clone(),
            line: spans
                .get(segments)
                .map_or(1, |&offset| lines.line_column(offset).0),
        };
        let mut segments = Vec::new();
        self.walk(&table, &mut segments, &at);
    }

    fn walk(
        &mut self,
        table: &toml::Table,
        segments: &mut Vec<String>,
        at: &dyn Fn(&[String]) -> Origin,
    ) {
        for (name, value) in table {
            segments.push(name.clone());
            let top = segments.len() == 1;
            if top && name == CONFIG_VERSION_KEY {
                // Not a resolved key; a file newer than this binary is rejected (RFC-0054 §8).
                if let Some((code, message)) = config_version_problem(Some(value)) {
                    self.errors.push(
                        Problem::new(code, Some(at(segments)), message).about(CONFIG_VERSION_KEY),
                    );
                }
            } else if top && name == POLICY_TABLE {
                self.policy(value, segments, at);
            } else if segments.iter().all(|s| crate::schema::is_bare_key(s)) {
                let path = segments.join(".");
                if let Some(key) = self.registry.key(&path) {
                    if self.assign(key, value.clone(), at(segments)).is_none() {
                        self.errors.push(
                            Problem::new(Code::ScopeMismatch, Some(at(segments)), misplaced(key))
                                .about(&path),
                        );
                    }
                } else if value.is_table() && self.is_table(&path) {
                    if let Value::Table(inner) = value {
                        self.walk(inner, segments, at);
                    }
                } else if let Some(deprecated) = self.registry.deprecated(&path) {
                    self.deprecated(deprecated, value.clone(), at(segments));
                } else {
                    self.errors.push(
                        Problem::new(
                            Code::UnknownKey,
                            Some(at(segments)),
                            format!("unknown key `{path}`"),
                        )
                        .about(&path)
                        .hint(did_you_mean(self.registry, &path)),
                    );
                }
            } else {
                let written = segments
                    .iter()
                    .map(|s| text::Key(s).to_string())
                    .collect::<Vec<_>>()
                    .join(".");
                self.errors.push(
                    Problem::new(
                        Code::UnknownKey,
                        Some(at(segments)),
                        format!("unknown key `{written}`"),
                    )
                    .about(written),
                );
            }
            segments.pop();
        }
    }

    /// `[policy."<selector>"]` tables are cluster storage for instance policies (RFC-0054
    /// §2.1): each one is a proposal, checked against the instance keys when policies resolve.
    fn policy(
        &mut self,
        value: &Value,
        segments: &mut Vec<String>,
        at: &dyn Fn(&[String]) -> Origin,
    ) {
        let Value::Table(selectors) = value else {
            self.errors.push(
                Problem::new(
                    Code::WrongType,
                    Some(at(segments)),
                    "`policy` must be a table of model policies, such as `[policy.\"qwen3*\"]`",
                )
                .about(POLICY_TABLE),
            );
            return;
        };
        for (selector, body) in selectors {
            segments.push(selector.clone());
            if let Value::Table(table) = body {
                let path = format!("{POLICY_TABLE}.{}", text::Key(selector));
                self.policy_keys(table, &path, segments, &mut Vec::new(), at);
                self.proposals.insert(
                    path.clone(),
                    Proposal::policy(path, self.registry, body.clone(), at(segments)),
                );
            } else {
                let path = format!("{POLICY_TABLE}.{}", text::Key(selector));
                self.errors.push(
                    Problem::new(
                        Code::WrongType,
                        Some(at(segments)),
                        format!("`{path}` must be a table of instance keys"),
                    )
                    .about(path),
                );
            }
            segments.pop();
        }
    }

    fn env(&mut self, os: Os, env: &Env) {
        for (name, value) in env.syntropy_vars() {
            let Some(name) = name.to_str() else {
                self.warnings.push(Problem::new(
                    Code::UnknownKey,
                    None,
                    format!(
                        "ignored an environment variable whose name starts with `SYNTROPY_` but \
                         is not UTF-8: {}",
                        name.to_string_lossy()
                    ),
                ));
                continue;
            };
            if FIXED.contains(&name) || OTHER_KNOWN.contains(&name) {
                continue;
            }
            let at = Origin::Env {
                var: name.to_owned(),
            };
            let path = env::path_for(name);
            let key = path.as_deref().and_then(|path| self.registry.key(path));
            let Some(key) = key else {
                if let Some(deprecated) = path.as_deref().and_then(|p| self.registry.deprecated(p))
                {
                    if let Some(text) = value.to_str() {
                        let value = text
                            .parse::<Value>()
                            .unwrap_or_else(|_| Value::String(text.to_owned()));
                        self.deprecated(deprecated, value, at);
                    }
                    continue;
                }
                let hint = path
                    .as_deref()
                    .and_then(|p| suggest(self.registry, p))
                    .map(|p| format!("did you mean `{}`?", env::var_for(p)));
                let mut problem = Problem::new(
                    Code::UnknownKey,
                    Some(at),
                    format!(
                        "unknown variable: `{name}` names no config key{}",
                        path.as_deref()
                            .map(|p| format!(" (`{p}`)"))
                            .unwrap_or_default()
                    ),
                )
                .hint(hint);
                if let Some(path) = path {
                    problem = problem.about(path);
                }
                self.warnings.push(problem);
                continue;
            };
            if matches!(key.scope, Scope::Instance | Scope::Request) {
                self.warnings.push(
                    Problem::new(
                        Code::ScopeMismatch,
                        Some(at),
                        format!(
                            "ignored: `{}` is a {} key, and environment variables set node keys",
                            key.path, key.scope
                        ),
                    )
                    .about(key.path),
                );
                continue;
            }
            let Some(text) = value.to_str() else {
                self.errors.push(
                    Problem::new(Code::WrongType, Some(at), "the value is not valid UTF-8")
                        .about(key.path),
                );
                continue;
            };
            let value = read_text(key, text);
            self.assign(key, value, at);
        }
        // `SYNTROPY_HOME` sets the state directory unless its own variable is set too.
        let own_var = env::var_for(STATE_DIR_KEY);
        if env.get_nonempty(HOME_VAR).is_some() && env.get_nonempty(&own_var).is_none() {
            let state = locate(os, Mode::User, env, None).dirs.state;
            if let (Some(key), Some(state)) = (self.registry.key(STATE_DIR_KEY), state) {
                let value = Value::String(state.to_string_lossy().into_owned());
                self.assign(
                    key,
                    value,
                    Origin::Env {
                        var: String::from(HOME_VAR),
                    },
                );
            }
        }
    }

    fn flags(&mut self, flags: &Flags) {
        for flag in &flags.values {
            let at = Origin::Cli {
                flag: flag.flag.clone(),
            };
            let Some(key) = self.registry.key(&flag.path) else {
                if let Some(deprecated) = self.registry.deprecated(&flag.path) {
                    let value = flag
                        .text
                        .parse::<Value>()
                        .unwrap_or_else(|_| Value::String(flag.text.clone()));
                    self.deprecated(deprecated, value, at);
                    continue;
                }
                self.errors.push(
                    Problem::new(
                        Code::UnknownKey,
                        Some(at),
                        format!("unknown key `{}`", flag.path),
                    )
                    .about(&flag.path)
                    .hint(did_you_mean(self.registry, &flag.path)),
                );
                continue;
            };
            let value = read_text(key, &flag.text);
            if self.assign(key, value, at.clone()).is_none() {
                self.errors.push(
                    Problem::new(Code::ScopeMismatch, Some(at), misplaced(key)).about(key.path),
                );
            }
        }
    }
}

impl Resolution<'_> {
    /// RFC-0054 §8's phases 2 and 3 inside one `[policy."<selector>"]` table: every key is a
    /// known instance key, and its value fits. Each table is checked alone; merging them by Rule 3
    /// is PROD-E1-T08's.
    fn policy_keys(
        &mut self,
        table: &toml::Table,
        policy: &str,
        segments: &mut Vec<String>,
        relative: &mut Vec<String>,
        at: &dyn Fn(&[String]) -> Origin,
    ) {
        for (name, value) in table {
            segments.push(name.clone());
            relative.push(name.clone());
            let key_path = relative.join(".");
            let relative_written = relative
                .iter()
                .map(|s| text::Key(s).to_string())
                .collect::<Vec<_>>()
                .join(".");
            let written = format!("{policy}.{relative_written}");
            // As in `walk`: a segment no key could have, such as `"pin.layers"`, is one unknown
            // key, never the dotted path its text spells.
            if !relative.iter().all(|s| crate::schema::is_bare_key(s)) {
                self.errors.push(
                    Problem::new(
                        Code::UnknownKey,
                        Some(at(segments)),
                        format!("unknown instance key `{relative_written}` in `{policy}`"),
                    )
                    .about(&written),
                );
                segments.pop();
                relative.pop();
                continue;
            }
            match self.registry.key(&key_path) {
                Some(key) if key.scope == Scope::Instance => {
                    if let Err(error) = (key.check)(value) {
                        self.errors
                            .push(value_problem(key, &error, at(segments)).about(&written));
                    }
                }
                Some(key) => self.errors.push(
                    Problem::new(
                        Code::ScopeMismatch,
                        Some(at(segments)),
                        format!(
                            "`{key_path}` is a {} key, and a policy table holds instance keys",
                            key.scope
                        ),
                    )
                    .about(&written),
                ),
                None if value.is_table() && self.is_table(&key_path) => {
                    if let Value::Table(inner) = value {
                        self.policy_keys(inner, policy, segments, relative, at);
                    }
                }
                None => self.errors.push(
                    Problem::new(
                        Code::UnknownKey,
                        Some(at(segments)),
                        format!("unknown instance key `{key_path}` in `{policy}`"),
                    )
                    .about(&written)
                    .hint(did_you_mean(self.registry, &key_path)),
                ),
            }
            segments.pop();
            relative.pop();
        }
    }

    /// A deprecated key a layer set: a warning naming the replacement, and the value goes to the
    /// replacement (RFC-0054 §8). A key removed outright is ignored, with the warning.
    fn deprecated(&mut self, deprecated: &'static Deprecated, value: Value, origin: Origin) {
        self.warnings.push(
            Problem::uncoded(Some(origin.clone()), deprecated.warning()).about(deprecated.path),
        );
        if let Some(key) = deprecated
            .replacement
            .and_then(|new| self.registry.key(new))
            && self.assign(key, value, origin.clone()).is_none()
        {
            self.errors.push(
                Problem::new(Code::ScopeMismatch, Some(origin), misplaced(key)).about(key.path),
            );
        }
    }
}

/// "did you mean `…`?" for an unknown key, when a registered key is close.
fn did_you_mean(registry: &Registry, path: &str) -> Option<String> {
    suggest(registry, path).map(|close| format!("did you mean `{close}`?"))
}

/// A value that failed phase 3. A secret-typed key's message is fixed, since an error about a
/// value inside it could quote the value.
fn value_problem(key: &'static KeySchema, error: &ValueError, origin: Origin) -> Problem {
    let message = if text::is_secret(key) {
        format!(
            "`{}` does not fit its type, `{}`; the value is not shown, since the key holds a secret",
            key.path, key.ty
        )
    } else {
        format!("`{}`: {}", key.path, error.message())
    };
    Problem::new(Code::from(error.kind()), Some(origin), message).about(key.path)
}

/// Why a key of `key.scope` cannot be set from a node's file, environment or flags.
fn misplaced(key: &KeySchema) -> String {
    match key.scope {
        Scope::Instance => format!(
            "`{}` is an instance key: set it in a `[policy.\"<model>\"]` table, or when loading \
             the model",
            key.path
        ),
        _ => format!(
            "`{}` is a {} key: a request sets it, not the node's configuration",
            key.path, key.scope
        ),
    }
}

/// The byte offset of every key in a parsed document, by key path. Arrays are not entered: a
/// key inside an array of tables belongs to the array's key.
fn key_offsets(table: &toml::de::DeTable<'_>) -> BTreeMap<Vec<String>, usize> {
    fn walk(
        table: &toml::de::DeTable<'_>,
        prefix: &mut Vec<String>,
        out: &mut BTreeMap<Vec<String>, usize>,
    ) {
        for (key, value) in table.iter() {
            prefix.push(key.get_ref().to_string());
            out.entry(prefix.clone()).or_insert(key.span().start);
            if let toml::de::DeValue::Table(inner) = value.get_ref() {
                walk(inner, prefix, out);
            }
            prefix.pop();
        }
    }
    let mut out = BTreeMap::new();
    walk(table, &mut Vec::new(), &mut out);
    out
}

/// Byte offsets to 1-based lines and columns.
struct LineIndex<'a> {
    text: &'a str,
    starts: Vec<usize>,
}

impl<'a> LineIndex<'a> {
    fn new(text: &'a str) -> Self {
        let starts = std::iter::once(0)
            .chain(text.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        Self { text, starts }
    }

    fn line_column(&self, offset: usize) -> (u32, u32) {
        let offset = offset.min(self.text.len());
        let line = self.starts.partition_point(|&s| s <= offset).max(1);
        let start = self.starts[line - 1];
        let column = self
            .text
            .get(start..offset)
            .map_or(1, |s| s.chars().count() + 1);
        (
            u32::try_from(line).unwrap_or(u32::MAX),
            u32::try_from(column).unwrap_or(u32::MAX),
        )
    }
}

/// The node's current [`ResolvedConfig`], replaced whole by a reload (RFC-0054 §9.1).
///
/// A reader takes the current snapshot with [`SharedConfig::load`] and keeps it as long as it
/// needs. A reload builds a new snapshot and stores it with [`SharedConfig::replace`], one atomic
/// pointer store. Readers take no locks, and no reader sees a mix of two snapshots.
pub struct SharedConfig {
    current: ArcSwap<ResolvedConfig>,
}

impl SharedConfig {
    /// Holds `config` as the current snapshot.
    pub fn new(config: ResolvedConfig) -> Self {
        Self {
            current: ArcSwap::from_pointee(config),
        }
    }

    /// The current snapshot.
    pub fn load(&self) -> Arc<ResolvedConfig> {
        self.current.load_full()
    }

    /// Makes `config` the current snapshot and returns the one it replaced.
    pub fn replace(&self, config: ResolvedConfig) -> Arc<ResolvedConfig> {
        self.current.swap(Arc::new(config))
    }
}

/// Redacts secrets, as `config show` does.
impl fmt::Debug for SharedConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedConfig")
            .field("current", &self.load())
            .finish()
    }
}

/// Reads the files `scan` found and resolves them with the environment and flags: the whole
/// load a daemon or `config check` performs. A file that cannot be read is an error, like a
/// syntax error, and the files that can be read are still resolved, so their errors are
/// reported with it.
pub fn load(
    registry: &Registry,
    os: Os,
    scan: &Scan,
    env: &Env,
    flags: &Flags,
) -> Result<ResolvedConfig, Rejected> {
    read_and_resolve(registry, os, scan, env, flags, |_| None).1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A map of secrets beside a plain key, for [`withhold_secrets`].
    #[derive(Debug, serde::Deserialize, crate::ConfigSection)]
    #[config(section = "p", scope = Node, owner = "RFC-0054")]
    #[allow(dead_code)]
    struct Mixed {
        /// Headers, each a secret.
        #[config(reload = Hot, default = {})]
        headers: BTreeMap<String, crate::Secret<String>>,
        /// A count.
        #[config(reload = Hot, default = 1)]
        count: u32,
    }

    /// `section()` withholds a wrong-shaped secret's value if one reaches it, though phase 3
    /// rejects such a value at load: it reads the section again with each secret-typed key at
    /// its default, and names those keys without serde's text. A fault in a plain key keeps
    /// serde's message.
    #[test]
    fn a_wrong_shaped_secret_is_withheld_from_a_section_error() {
        let table = |text: &str| -> toml::Table { text.parse().expect("a table") };
        let bad = table("headers = \"canary-u1\"\ncount = 2\n");
        let serde = deserialize::<Mixed>(Value::Table(bad.clone())).unwrap_err();
        assert!(
            serde.contains("canary-u1"),
            "serde quotes the value: {serde}"
        );
        let message = withhold_secrets::<Mixed>(bad, serde);
        assert!(!message.contains("canary"), "{message}");
        assert!(message.contains("`p.headers`"), "{message}");
        assert!(
            message.contains("withheld because the key holds a secret"),
            "{message}"
        );

        let plain = table("headers = {}\ncount = \"many\"\n");
        let serde = deserialize::<Mixed>(Value::Table(plain.clone())).unwrap_err();
        let message = withhold_secrets::<Mixed>(plain, serde);
        assert!(message.contains("invalid type"), "{message}");
        assert!(!message.contains("withheld"), "{message}");
    }

    #[test]
    fn line_index_counts_lines_and_columns() {
        let index = LineIndex::new("a = 1\n\nb = \"é\"x\n");
        assert_eq!(index.line_column(0), (1, 1));
        assert_eq!(index.line_column(4), (1, 5));
        assert_eq!(index.line_column(6), (2, 1));
        assert_eq!(index.line_column(7), (3, 1));
        // After the two-byte `é`, the column counts characters.
        assert_eq!(index.line_column(14), (3, 7));
    }

    #[test]
    fn a_byte_order_mark_is_dropped() {
        assert_eq!(ConfigFile::new("/x", "\u{feff}a = 1").text, "a = 1");
    }

    #[test]
    fn set_needs_key_equals_value() {
        let mut flags = Flags::new();
        assert!(flags.set("resource.ram_headroom=16GiB").is_ok());
        assert!(flags.set("a.b=").is_ok());
        for bad in [
            "novalue",
            "=x",
            "a..b=1",
            "policy.\"q*\".objective=latency",
            " =1",
        ] {
            assert!(flags.set(bad).is_err(), "{bad}");
        }
        // A value written with the wrong separator is never echoed.
        let problem = flags.set("api.token:hunter2-canary").unwrap_err();
        assert!(!problem.to_string().contains("hunter2"), "{problem}");
    }
}
