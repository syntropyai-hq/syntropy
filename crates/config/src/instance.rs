// SPDX-License-Identifier: MIT OR Apache-2.0

//! An instance's typed section, built from its policy (RFC-0054 §5.4; PROD-E1-T24).
//!
//! A domain crate declares its instance keys as a [`ConfigSection`], such as the scheduler's
//! `sched.*`. An instance's policy carries their values: the `<prefix>` table of a `policy.*`
//! table, or entries of `InstancePolicy::domain` by key path. [`InstanceSection::from_policy`]
//! and [`InstanceSection::from_domain`] build the section from them, so a domain crate never
//! parses its own table:
//!
//! - **phase 2** (RFC-0054 §8): every key the table sets is a key of the section, else it is
//!   `unknown_key` with the edit-distance suggestion; a key of the section whose scope is not
//!   `instance` is `scope_mismatch`;
//! - **phase 3**: each value fits its key's type, unit and range, through the key's own check
//!   (`wrong_type`, `bad_unit`, `out_of_range`, and so on);
//! - a key the table leaves out takes its fixed default, or its computed default from the
//!   `computed` function the caller passes, since the caller owns the inputs the rule needs,
//!   such as the instance's stage count. A computed default the caller does not supply leaves
//!   an `Option` key `None`.
//!
//! Every problem in the policy is reported, not only the first, and none quotes a value of a
//! secret-typed key. A computed default the caller gets wrong (none for a key that is not an
//! `Option`, or one that does not fit its key) is the caller's error, a [`SectionError`], not
//! the policy's. [`InstanceSection::changes`] lists the keys two builds differ in, each with its
//! reload class (§9.1), which is what a reload acts on (§9.3).
//!
//! A section that shares its prefix with another section (`SectionSchema::shared`) reads only
//! its own keys from the table: the others belong to the other section.

use std::collections::BTreeMap;
use std::fmt;
use std::hash::{BuildHasher, RandomState};
use std::sync::OnceLock;

use syntropy_core::policy::{ConfigValue, KeyPath};

use crate::Value;
use crate::code::{Code, suggest_among};
use crate::diagnostic::Problem;
use crate::resolve::{SectionError, deserialize, to_value, value_problem, withhold_secrets};
use crate::schema::{ConfigSection, DefaultSpec, KeySchema, Reload, Scope};

/// Section `S` of one instance, with what each of its keys took.
#[derive(Clone, PartialEq)]
pub struct InstanceSection<S> {
    section: S,
    /// Each key's effective value by full path: the policy's, else its default.
    values: BTreeMap<&'static str, Kept>,
}

/// What a build keeps of a key's effective value, for [`InstanceSection::changes`].
#[derive(Clone, PartialEq)]
enum Kept {
    /// The value, or `None` for a key left unset, such as a computed default the caller did not
    /// supply.
    Plain(Option<Value>),
    /// A secret-typed key's value is not kept, so no copy of it outlives the section's own
    /// zeroing `Secret` (RFC-0054 §10.1). Only a fingerprint is, under a key drawn once per
    /// process: it tells two builds apart, is never shown, and cannot be checked against a guess
    /// outside the process.
    Secret(Option<u64>),
}

impl Kept {
    fn of(key: &KeySchema, value: Option<Value>) -> Self {
        if key.holds_secret {
            Kept::Secret(value.as_ref().map(fingerprint))
        } else {
            Kept::Plain(value)
        }
    }
}

/// A secret value's fingerprint: SipHash of its `Debug` form, keyed once per process. `Debug`
/// rather than the TOML text, because `toml`'s `Display` is a feature `config` does not turn on.
fn fingerprint(value: &Value) -> u64 {
    static KEY: OnceLock<RandomState> = OnceLock::new();
    KEY.get_or_init(RandomState::new)
        .hash_one(format!("{value:?}"))
}

/// The section, and each key's value with a secret-typed key's written `[redacted]`.
impl<S: fmt::Debug> fmt::Debug for InstanceSection<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Shown<'a>(&'a Kept);
        impl fmt::Debug for Shown<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.0 {
                    Kept::Plain(value) => value.fmt(f),
                    Kept::Secret(Some(_)) => f.write_str("[redacted]"),
                    Kept::Secret(None) => f.write_str("None"),
                }
            }
        }
        let values: BTreeMap<&str, Shown<'_>> = self
            .values
            .iter()
            .map(|(path, kept)| (*path, Shown(kept)))
            .collect();
        f.debug_struct("InstanceSection")
            .field("section", &self.section)
            .field("values", &values)
            .finish()
    }
}

/// A key whose effective value differs between two builds of a section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Change {
    /// The key's full path, such as `sched.max_queued`.
    pub key: &'static str,
    /// When the change takes effect (RFC-0054 §9.1).
    pub reload: Reload,
}

/// Why an instance's section was not built.
#[derive(Clone, Debug, PartialEq)]
pub enum InstanceSectionError {
    /// The policy's values break RFC-0054 §8 phases 2 and 3: every problem, each with its code.
    Rejected(Vec<Problem>),
    /// The caller's computed defaults do not make a section: one is missing for a key that is
    /// not an `Option`, or one does not fit its key.
    Section(SectionError),
}

impl std::fmt::Display for InstanceSectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstanceSectionError::Rejected(problems) => {
                let shown: Vec<String> = problems.iter().map(ToString::to_string).collect();
                write!(f, "{}", shown.join("; "))
            }
            InstanceSectionError::Section(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for InstanceSectionError {}

impl<S: ConfigSection> InstanceSection<S> {
    /// Builds `S` from `table`, the `<prefix>` table of an instance's policy, such as its
    /// `sched` table. `computed` gives the value of each computed default the table leaves out,
    /// or `None` to leave the key unset.
    ///
    /// # Errors
    ///
    /// [`InstanceSectionError::Rejected`] with every problem in `table`, or
    /// [`InstanceSectionError::Section`] when `computed` does not make a section.
    pub fn from_policy(
        table: &Value,
        computed: impl Fn(&'static KeySchema) -> Option<Value>,
    ) -> Result<Self, InstanceSectionError> {
        let schema = &S::SCHEMA;
        let Value::Table(table) = table else {
            let problem = Problem::new(
                Code::WrongType,
                None,
                format!("`{}` must be a table of its keys", schema.prefix),
            )
            .about(schema.prefix);
            return Err(InstanceSectionError::Rejected(vec![problem]));
        };
        let mut problems = Vec::new();
        let mut set = BTreeMap::new();
        for (name, value) in table {
            let path = full_path(schema.prefix, name);
            match schema.key(name) {
                // Another section owns the rest of a shared prefix.
                None if schema.shared => {}
                None => problems.push(
                    Problem::new(
                        Code::UnknownKey,
                        None,
                        format!("unknown key `{path}` in the `{}` section", schema.prefix),
                    )
                    .about(&path)
                    .hint(
                        suggest_among(schema.keys, &path)
                            .map(|close| format!("did you mean `{close}`?")),
                    ),
                ),
                Some(key) if key.scope != Scope::Instance => problems.push(
                    Problem::new(
                        Code::ScopeMismatch,
                        None,
                        format!(
                            "`{}` is a {} key, and an instance's policy holds instance keys",
                            key.path, key.scope
                        ),
                    )
                    .about(key.path),
                ),
                Some(key) => match (key.check)(value) {
                    Ok(()) => {
                        set.insert(key.name, value.clone());
                    }
                    Err(error) => problems.push(value_problem(key, &error, None)),
                },
            }
        }
        if !problems.is_empty() {
            return Err(InstanceSectionError::Rejected(problems));
        }
        Self::assemble(set, computed)
    }

    /// Builds `S` from the entries of `domain`, an `InstancePolicy::domain`, whose key paths
    /// lie under the section's prefix, as [`InstanceSection::from_policy`] builds it from the
    /// same keys in a table.
    ///
    /// An entry at the prefix itself, such as `sched`, holds a table of the section's keys and
    /// is read as one.
    ///
    /// # Errors
    ///
    /// As [`InstanceSection::from_policy`]. An entry deeper than one of the section's keys,
    /// such as `sched.queue.max`, is an unknown key, and an entry at the prefix that is not a
    /// table is the wrong type.
    pub fn from_domain(
        domain: &BTreeMap<KeyPath, ConfigValue>,
        computed: impl Fn(&'static KeySchema) -> Option<Value>,
    ) -> Result<Self, InstanceSectionError> {
        let prefix = S::SCHEMA.prefix;
        let wrong_type = |path: &str, message: String| {
            InstanceSectionError::Rejected(vec![
                Problem::new(Code::WrongType, None, message).about(path),
            ])
        };
        let mut table = toml::Table::new();
        for (path, value) in domain {
            let path = path.as_str();
            let name = if prefix.is_empty() {
                Some(path)
            } else if path == prefix {
                None
            } else {
                path.strip_prefix(prefix)
                    .and_then(|rest| rest.strip_prefix('.'))
            };
            if name.is_none() && path != prefix {
                continue;
            }
            let Some(value) = toml_value(value) else {
                return Err(wrong_type(
                    path,
                    format!("`{path}` holds a kind of value this build cannot read"),
                ));
            };
            match (name, value) {
                (Some(name), value) => {
                    table.insert(name.to_owned(), value);
                }
                (None, Value::Table(entries)) => table.extend(entries),
                (None, _) => {
                    return Err(wrong_type(
                        path,
                        format!("`{prefix}` must be a table of its keys"),
                    ));
                }
            }
        }
        Self::from_policy(&Value::Table(table), computed)
    }

    /// The section from the values that passed, with every default filled in.
    fn assemble(
        mut set: BTreeMap<&'static str, Value>,
        computed: impl Fn(&'static KeySchema) -> Option<Value>,
    ) -> Result<Self, InstanceSectionError> {
        let schema = &S::SCHEMA;
        let error = |message: String| {
            InstanceSectionError::Section(SectionError {
                section: schema.type_name,
                message,
            })
        };
        let mut table = toml::Table::new();
        let mut values = BTreeMap::new();
        let mut unsupplied = Vec::new();
        for key in schema.keys {
            let value = match set.remove(key.name) {
                Some(value) => Some(value),
                None => match key.default {
                    DefaultSpec::Value(v) => Some(to_value(&v)),
                    DefaultSpec::Unset => None,
                    DefaultSpec::Computed(_) => {
                        let value = computed(key);
                        match &value {
                            Some(v) => {
                                if (key.check)(v).is_err() {
                                    return Err(error(format!(
                                        "the computed default given for `{}` does not fit its \
                                         type or range",
                                        key.path
                                    )));
                                }
                            }
                            None => unsupplied.push(key.path),
                        }
                        value
                    }
                },
            };
            if let Some(value) = &value {
                table.insert(key.name.to_owned(), value.clone());
            }
            values.insert(key.path, Kept::of(key, value));
        }
        let section: S = deserialize(Value::Table(table.clone())).map_err(|message| {
            // Every value passed its key's check, so a computed default left out of a key that
            // is not an `Option` is the usual cause.
            if unsupplied.is_empty() {
                error(withhold_secrets::<S>(table, message))
            } else {
                error(format!(
                    "{}; computed defaults not supplied: {}",
                    withhold_secrets::<S>(table, message),
                    unsupplied
                        .iter()
                        .map(|path| format!("`{path}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }
        })?;
        Ok(Self { section, values })
    }

    /// The section.
    pub fn section(&self) -> &S {
        &self.section
    }

    /// The section, by value.
    pub fn into_section(self) -> S {
        self.section
    }

    /// The effective value of the key at `path`: the policy's, else its default. `None` for a
    /// key left unset, a secret-typed key, whose value is not kept (read it from the section),
    /// or a path that is not one of the section's keys.
    pub fn value(&self, path: &str) -> Option<&Value> {
        match self.values.get(path)? {
            Kept::Plain(value) => value.as_ref(),
            Kept::Secret(_) => None,
        }
    }

    /// The keys whose effective values differ between `self` and `newer`, in path order, each
    /// with its reload class (RFC-0054 §9.1). Identical builds give none.
    pub fn changes(&self, newer: &Self) -> Vec<Change> {
        S::SCHEMA
            .keys
            .iter()
            .filter(|key| self.values.get(key.path) != newer.values.get(key.path))
            .map(|key| {
                let change = Change {
                    key: key.path,
                    reload: key.reload,
                };
                (key.path, change)
            })
            .collect::<BTreeMap<_, _>>()
            .into_values()
            .collect()
    }

    /// The strictest reload class among [`InstanceSection::changes`]: what applying `newer`
    /// takes, or `None` when nothing changed.
    pub fn reload_needed(&self, newer: &Self) -> Option<Reload> {
        self.changes(newer).iter().map(|change| change.reload).max()
    }
}

impl<S> std::ops::Deref for InstanceSection<S> {
    type Target = S;

    fn deref(&self) -> &S {
        &self.section
    }
}

fn full_path(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}

/// `value` as TOML, or `None` for a kind of value this build does not know. A date-time whose
/// text does not parse becomes a string, which the key's own check then rejects as the wrong
/// type.
fn toml_value(value: &ConfigValue) -> Option<Value> {
    Some(match value {
        ConfigValue::Bool(b) => Value::Boolean(*b),
        ConfigValue::Integer(i) => Value::Integer(*i),
        ConfigValue::Float(x) => Value::Float(*x),
        ConfigValue::String(s) => Value::String(s.clone()),
        ConfigValue::Datetime(text) => text
            .parse()
            .map_or_else(|_| Value::String(text.clone()), Value::Datetime),
        ConfigValue::Array(items) => {
            Value::Array(items.iter().map(toml_value).collect::<Option<_>>()?)
        }
        ConfigValue::Table(entries) => Value::Table(
            entries
                .iter()
                .map(|(name, value)| Some((name.clone(), toml_value(value)?)))
                .collect::<Option<_>>()?,
        ),
        // `ConfigValue` is non-exhaustive: a kind added later is refused, not guessed at.
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
