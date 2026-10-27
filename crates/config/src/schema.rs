// SPDX-License-Identifier: MIT OR Apache-2.0

//! The schema as data: what `#[derive(ConfigSection)]` records about each section and key
//! (RFC-0054 §10.3).
//!
//! A domain crate declares its section once, as a serde struct with `#[config(...)]` attributes.
//! The derive turns the attributes into a [`SectionSchema`] constant, so the loader, the
//! validators, the reference docs and the JSON Schema all read one definition (ADR-0500
//! decision 2).

use std::fmt;

use serde::de::DeserializeOwned;

use crate::value::ValueError;

/// Where a key's value comes from (RFC-0054 §2.1, RFC-0001 §11.1). Every key has exactly one
/// scope, and it never changes within a `config_version`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    /// The node's config file, environment and CLI: `default < file < env < CLI`.
    Node,
    /// The event log only. A node's file proposes a value; it never overrides one (§2.3).
    Cluster,
    /// `InstancePolicy` on `EnsureInstance`, with defaults from the `policy.*` tables (§2.2
    /// Rule 3).
    Instance,
    /// The API request. Request keys only narrow what the instance allows (§7).
    Request,
}

impl Scope {
    /// The scope's name as RFC-0054 writes it: `node`, `cluster`, `instance` or `request`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Scope::Node => "node",
            Scope::Cluster => "cluster",
            Scope::Instance => "instance",
            Scope::Request => "request",
        }
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// When a changed value takes effect (RFC-0054 §9.1, RFC-0001 §11.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reload {
    /// On `SIGHUP`, `syntropy config reload` or the reload API, by one atomic pointer store.
    Hot,
    /// When the affected instances adopt a new plan epoch.
    Instance,
    /// When the daemon restarts.
    Node,
}

impl Reload {
    /// The class's name as RFC-0054 writes it: `hot`, `instance` or `node`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Reload::Hot => "hot",
            Reload::Instance => "instance",
            Reload::Node => "node",
        }
    }
}

impl fmt::Display for Reload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Who may change a key held in the event log (RFC-0054 §2.3 step 10, RFC-0034 §8.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Authority {
    /// The operator of any member that holds `STAGE_HOST`.
    Member,
    /// A patch that sets the key carries a keyholder's signature.
    Keyholder,
}

impl Authority {
    /// The authority's name as RFC-0054 writes it: `member` or `keyholder`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Authority::Member => "member",
            Authority::Keyholder => "keyholder",
        }
    }
}

impl fmt::Display for Authority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A key's default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DefaultSpec {
    /// A fixed value.
    Value(DefaultValue),
    /// No value. Only an `Option` key may start unset; [`ConfigSection`]'s default check
    /// rejects it anywhere else.
    Unset,
    /// Computed when the configuration resolves, from the probed hardware or from other keys.
    /// The text is the rule as `config show` prints it in `Origin::Default { computed }`, such as
    /// `"max(6GiB, 10% of RAM)"` (RFC-0054 §2.2).
    Computed(&'static str),
}

/// A value written in a `#[config(...)]` attribute: a default, or a range bound. It has the
/// shape of a TOML value, and its [`Display`](fmt::Display) form is the TOML a user would write.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DefaultValue {
    /// `true` or `false`.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A float.
    Float(f64),
    /// A string, in the unit syntax of RFC-0054 §3 where the key's type has one (`"16GiB"`).
    Str(&'static str),
    /// An array.
    Array(&'static [DefaultValue]),
    /// An inline table, in written order.
    Table(&'static [(&'static str, DefaultValue)]),
}

impl DefaultValue {
    /// Deserializes the value into `T`, the way the loader deserializes a value from a file.
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T, ValueError> {
        T::deserialize(crate::default::ValueDeserializer::new(self))
            .map_err(|e| ValueError::wrong_type(e.to_string()))
    }
}

impl fmt::Display for DefaultValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DefaultValue::Bool(b) => write!(f, "{b}"),
            DefaultValue::Int(i) => write!(f, "{i}"),
            // `{:?}` keeps the decimal point (`1.0`), which TOML needs to read a float.
            DefaultValue::Float(x) => write!(f, "{x:?}"),
            DefaultValue::Str(s) => write_toml_string(f, s),
            DefaultValue::Array(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("]")
            }
            DefaultValue::Table(entries) => {
                if entries.is_empty() {
                    return f.write_str("{}");
                }
                f.write_str("{ ")?;
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    if is_bare_key(key) {
                        f.write_str(key)?;
                    } else {
                        write_toml_string(f, key)?;
                    }
                    write!(f, " = {value}")?;
                }
                f.write_str(" }")
            }
        }
    }
}

/// Writes a TOML basic string, escaping what TOML requires and every control and
/// [`crate::chars::invisible`] character, as TOML's `\uXXXX` or `\UXXXXXXXX`: the text holds none
/// of them raw, so it is safe on a terminal, and it parses back to `s`.
pub(crate) fn write_toml_string(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
    f.write_str("\"")?;
    for c in s.chars() {
        match c {
            '"' => f.write_str("\\\"")?,
            '\\' => f.write_str("\\\\")?,
            '\n' => f.write_str("\\n")?,
            '\t' => f.write_str("\\t")?,
            '\r' => f.write_str("\\r")?,
            // Controls and invisible characters as TOML escapes, so printed TOML holds none raw
            // and still parses back to the same string.
            c if c.is_control() || crate::chars::invisible(c) => {
                if u32::from(c) <= 0xFFFF {
                    write!(f, "\\u{:04X}", u32::from(c))?;
                } else {
                    write!(f, "\\U{:08X}", u32::from(c))?;
                }
            }
            c => write!(f, "{c}")?,
        }
    }
    f.write_str("\"")
}

/// Whether TOML accepts `key` unquoted.
pub(crate) fn is_bare_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// One key, as `#[derive(ConfigSection)]` records it. The derive builds these; code reads them.
#[derive(Clone, Copy, Debug)]
pub struct KeySchema {
    /// The field name, which is the last segment of `path`.
    pub name: &'static str,
    /// The full dotted path, such as `resource.ram_headroom`. Instance keys are written
    /// relative to a `policy.*` table, such as `objective`.
    pub path: &'static str,
    /// The field's doc comment, which the reference page and `config get --help` print.
    pub doc: &'static str,
    /// The field's Rust type as written, such as `Option<Duration>`.
    pub ty: &'static str,
    /// Where the key's value comes from.
    pub scope: Scope,
    /// When a change takes effect.
    pub reload: Reload,
    /// The owning RFC, such as `RFC-0020`.
    pub owner: &'static str,
    /// The default.
    pub default: DefaultSpec,
    /// The smallest accepted value, if the key has one.
    pub min: Option<DefaultValue>,
    /// The largest accepted value, if the key has one.
    pub max: Option<DefaultValue>,
    /// The cluster feature level that introduced the key (RFC-0032 §6.4). Required for cluster
    /// keys, optional for instance keys, and `None` for node and request keys.
    pub introduced_at: Option<u32>,
    /// Who may change the key in the event log. `Some` for cluster and instance keys, `None`
    /// for node and request keys.
    pub authority: Option<Authority>,
    /// The environment variable that sets a node key, such as
    /// `SYNTROPY_RESOURCE__RAM_HEADROOM` (RFC-0054 §2.4). `None` for other scopes.
    pub env: Option<&'static str>,
    /// Whether the key's type holds a secret anywhere inside it: `<T as HoldsSecret>::HOLDS_SECRET`
    /// ([`HoldsSecret`](crate::HoldsSecret)), through options, maps, arrays and nested structs.
    /// `config show`, flag handling and the diagnostics redact by this flag, never by the
    /// type's name (RFC-0054 §10.1).
    pub holds_secret: bool,
    /// Checks a value for this key, as a file, the environment or a flag gives it: it must
    /// deserialize into the field's type and lie in the key's range (RFC-0054 §8 phase 3). The
    /// error's kind gives its code, such as `bad_unit`.
    pub check: fn(&toml::Value) -> Result<(), ValueError>,
    /// Whether two values for this key give the same value. Both are read as the field's
    /// type, through its `serde(with)` path if it has one, and compared with `==` where the
    /// type implements `PartialEq`, so `"300us"` and `"300000ns"` are one duration. A type
    /// without `PartialEq`, such as a `Secret`, compares values as written, and so does a
    /// value that does not read.
    pub same: fn(&toml::Value, &toml::Value) -> bool,
}

/// One section, as `#[derive(ConfigSection)]` records it.
#[derive(Clone, Copy, Debug)]
pub struct SectionSchema {
    /// The Rust type that declares the section.
    pub type_name: &'static str,
    /// The table the keys live in, such as `resource` or `api.auth`; empty for keys at the top
    /// level of the file (or of a `policy.*` table).
    pub prefix: &'static str,
    /// `false`: the section owns the whole `prefix.*` subtree. `true`: it owns only its own
    /// keys, inside a table another section owns (RFC-0054 §5.1 gives `net.relay_mode` to
    /// RFC-0030 and the other `net.*` keys to RFC-0031). A top-level section is always shared.
    pub shared: bool,
    /// The owning RFC.
    pub owner: &'static str,
    /// The scope keys take unless they declare their own.
    pub scope: Scope,
    /// The struct's doc comment.
    pub doc: &'static str,
    /// The keys, in field order.
    pub keys: &'static [KeySchema],
    /// Keys a release renamed or removed, which the section still accepts with a warning
    /// (RFC-0054 §8, RFC-0062 §10.1).
    pub deprecated: &'static [Deprecated],
    /// Deserializes every fixed default into its field's type and checks it against the key's
    /// range. [`ConfigSection::check_defaults`] calls it.
    pub check_defaults: fn() -> Vec<KeyError>,
}

impl SectionSchema {
    /// The key named `name` (the field name, not the full path).
    pub fn key(&self, name: &str) -> Option<&'static KeySchema> {
        self.keys.iter().find(|k| k.name == name)
    }
}

/// A key a release renamed or removed (RFC-0054 §8). A file, variable or flag that sets it loads
/// with a warning naming the replacement, and its value goes to the replacement. The record is
/// removed by the release that closes RFC-0062 §10.1's window ([`Deprecated::closes_in`]);
/// from then on the key is unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deprecated {
    /// The old full path.
    pub path: &'static str,
    /// The full path of the key that replaced it, or `None` if it was removed outright.
    pub replacement: Option<&'static str>,
    /// The release that deprecated it, as `major.minor`.
    pub since: &'static str,
}

impl Deprecated {
    /// The first release, as `(major, minor)`, that no longer accepts the key: one minor
    /// release before 1.0 and one major release after (RFC-0062 §10.1). `None` if `since` is
    /// not `major.minor`, which the derive rejects.
    pub fn closes_in(&self) -> Option<(u32, u32)> {
        let (major, minor) = self.since.split_once('.')?;
        let (major, minor): (u32, u32) = (major.parse().ok()?, minor.parse().ok()?);
        Some(if major == 0 {
            (0, minor + 1)
        } else {
            (major + 1, 0)
        })
    }

    /// The warning a load prints for the key.
    pub fn warning(&self) -> String {
        match self.replacement {
            Some(new) => format!(
                "`{}` is deprecated since {}; use `{new}`, which now takes its value",
                self.path, self.since
            ),
            None => format!(
                "`{}` is deprecated since {} and has no effect; remove it",
                self.path, self.since
            ),
        }
    }
}

/// A configuration section: one struct of keys under one prefix (RFC-0001 §11.2). Implement it
/// with `#[derive(ConfigSection)]`, never by hand.
///
/// ```
/// use serde::Deserialize;
/// use syntropy_config::{ByteSize, ConfigSection, Duration};
///
/// /// Example settings.
/// #[derive(Deserialize, ConfigSection)]
/// #[config(section = "example", scope = Node, owner = "RFC-0054")]
/// struct ExampleSection {
///     /// How much memory the example keeps.
///     #[config(reload = Hot, default = "1GiB", min = "64MiB")]
///     keep: ByteSize,
///     /// How long the example waits.
///     #[config(reload = Node, default = "30s")]
///     wait: Duration,
/// }
///
/// let key = ExampleSection::SCHEMA.key("keep").unwrap();
/// assert_eq!(key.path, "example.keep");
/// assert_eq!(key.env, Some("SYNTROPY_EXAMPLE__KEEP"));
/// assert!(ExampleSection::check_defaults().is_empty());
/// ```
pub trait ConfigSection: DeserializeOwned + 'static {
    /// The section's schema.
    const SCHEMA: SectionSchema;

    /// Checks every key that declares `min` or `max` against its range (RFC-0054 §8 phase 3).
    /// Returns one error per key out of range, or none.
    fn check_ranges(&self) -> Vec<KeyError>;

    /// Deserializes every fixed default into its field's type, checks it against the key's
    /// range, and checks that `default = unset` sits only on an `Option` key. A domain crate's
    /// tests call this, so a default that does not parse fails its build's tests rather than a
    /// user's start-up.
    fn check_defaults() -> Vec<KeyError> {
        (Self::SCHEMA.check_defaults)()
    }
}

/// A key whose value, default or range is invalid.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyError {
    /// The key's full path.
    pub key: &'static str,
    /// What is wrong.
    pub error: ValueError,
}

impl fmt::Display for KeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`: {}", self.key, self.error)
    }
}

impl std::error::Error for KeyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_print_as_toml() {
        let cases = [
            (DefaultValue::Bool(false), "false"),
            (DefaultValue::Int(-3), "-3"),
            (DefaultValue::Float(1.0), "1.0"),
            (DefaultValue::Float(0.015625), "0.015625"),
            (DefaultValue::Str("16GiB"), "\"16GiB\""),
            (DefaultValue::Str("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\""),
            (
                DefaultValue::Array(&[
                    DefaultValue::Str("127.0.0.1:7433"),
                    DefaultValue::Str("[::1]:7433"),
                ]),
                "[\"127.0.0.1:7433\", \"[::1]:7433\"]",
            ),
            (DefaultValue::Table(&[]), "{}"),
            (
                DefaultValue::Table(&[
                    ("interactive", DefaultValue::Str("6tok/s")),
                    ("odd key", DefaultValue::Int(1)),
                ]),
                "{ interactive = \"6tok/s\", \"odd key\" = 1 }",
            ),
        ];
        for (value, toml) in cases {
            assert_eq!(value.to_string(), toml);
        }
    }

    #[test]
    fn names_match_rfc_0054() {
        assert_eq!(Scope::Instance.to_string(), "instance");
        assert_eq!(Reload::Hot.to_string(), "hot");
        assert_eq!(Authority::Keyholder.to_string(), "keyholder");
    }
}
