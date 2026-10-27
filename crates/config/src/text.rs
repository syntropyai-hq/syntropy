// SPDX-License-Identifier: MIT OR Apache-2.0

//! Values and key paths printed as TOML, so a value `config show` prints can be pasted back
//! into a file (RFC-0054 §3).

use std::fmt;

use toml::Value;

use crate::registry::Registry;
use crate::schema::{KeySchema, is_bare_key, write_toml_string};
use crate::value::{REDACTED, SecretSource};

/// `value` as inline TOML.
pub(crate) struct Inline<'a>(pub(crate) &'a Value);

impl fmt::Display for Inline<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Value::String(s) => write_toml_string(f, s),
            Value::Integer(i) => write!(f, "{i}"),
            Value::Float(x) if x.is_nan() => f.write_str("nan"),
            Value::Float(x) if x.is_infinite() => {
                f.write_str(if *x > 0.0 { "inf" } else { "-inf" })
            }
            // `{:?}` keeps the decimal point (`1.0`), which TOML needs to read a float.
            Value::Float(x) => write!(f, "{x:?}"),
            Value::Boolean(b) => write!(f, "{b}"),
            Value::Datetime(d) => write!(f, "{d}"),
            Value::Array(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", Inline(item))?;
                }
                f.write_str("]")
            }
            Value::Table(table) => {
                if table.is_empty() {
                    return f.write_str("{}");
                }
                f.write_str("{ ")?;
                for (i, (key, value)) in table.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{} = {}", Key(key), Inline(value))?;
                }
                f.write_str(" }")
            }
        }
    }
}

/// One key segment, quoted when TOML needs it.
pub(crate) struct Key<'a>(pub(crate) &'a str);

impl fmt::Display for Key<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if is_bare_key(self.0) {
            f.write_str(self.0)
        } else {
            write_toml_string(f, self.0)
        }
    }
}

/// Whether a key's type holds a secret anywhere inside it, as the derive recorded it
/// ([`KeySchema::holds_secret`]).
pub(crate) fn is_secret(key: &KeySchema) -> bool {
    key.holds_secret
}

/// Whether a key's type is exactly `SecretSource`, optional or not. Its file and variable forms
/// (`{ file = … }`, `{ env = … }`) name where the secret is, not the secret, so they may print.
/// Any other spelling, such as an alias, fails this check and is redacted whole.
fn is_secret_source(key: &KeySchema) -> bool {
    if !key.holds_secret {
        return false;
    }
    let ty: String = key.ty.chars().filter(|c| !c.is_whitespace()).collect();
    let inner = ty
        .strip_prefix("Option<")
        .and_then(|t| t.strip_suffix('>'))
        .unwrap_or(&ty);
    inner.rsplit("::").next() == Some("SecretSource")
}

/// A key's value as `config show` prints it (RFC-0054 §10.1, G7).
/// - A `SecretSource` key prints its file or variable form as written, and an inline value as
///   `[redacted]`.
/// - Any other key whose type holds a secret prints as `[redacted]` whole, whatever its value's
///   shape: in a table of secrets, an entry named `env` or `file` is a secret, not a source.
/// - Everything else prints as inline TOML.
pub(crate) fn shown(key: &KeySchema, value: &Value) -> String {
    if is_secret_source(key) {
        return match value.clone().try_into::<SecretSource>() {
            Ok(source) => source.to_string(),
            Err(_) => String::from(REDACTED),
        };
    }
    if is_secret(key) {
        return String::from(REDACTED);
    }
    Inline(value).to_string()
}

/// A `policy."<selector>"` table as `config show` prints it, and whether anything in it is
/// redacted. The table's entries are instance keys, written relative to the table (RFC-0054
/// §5.4):
/// - an entry that is an instance key whose type holds no secret prints as written;
/// - an entry whose key holds a secret ([`KeySchema::holds_secret`]) prints `[redacted]`;
/// - an entry that no instance key names prints `[redacted]` too, because an unknown key may be
///   a misspelt secret. Validating the table is the policy tables' task.
///
/// A table that groups instance keys, such as `approximate`, is walked entry by entry.
pub(crate) fn policy_shown(registry: &Registry, value: &Value) -> (String, bool) {
    let Value::Table(table) = value else {
        return (String::from(REDACTED), true);
    };
    let mut redacted = false;
    let text = policy_table(registry, table, "", &mut redacted);
    (text, redacted)
}

fn policy_table(
    registry: &Registry,
    table: &toml::Table,
    prefix: &str,
    redacted: &mut bool,
) -> String {
    if table.is_empty() {
        return String::from("{}");
    }
    let entries: Vec<String> = table
        .iter()
        .map(|(name, value)| {
            let path = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}.{name}")
            };
            let key = registry
                .key(&path)
                .filter(|k| k.scope == crate::Scope::Instance);
            let under = |k: &KeySchema| {
                k.scope == crate::Scope::Instance
                    && k.path.len() > path.len()
                    && k.path.starts_with(&path)
                    && k.path.as_bytes()[path.len()] == b'.'
            };
            let shown = match (key, value) {
                (Some(key), value) if !key.holds_secret => Inline(value).to_string(),
                (None, Value::Table(inner)) if registry.keys().any(under) => {
                    policy_table(registry, inner, &path, redacted)
                }
                _ => {
                    *redacted = true;
                    String::from(REDACTED)
                }
            };
            format!("{} = {shown}", Key(name))
        })
        .collect();
    format!("{{ {} }}", entries.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prints_inline_toml_that_parses_back() {
        let text =
            r#"{ a = "x\"y", "b c" = [1, 2.5, true], d = { e = inf }, f = 1979-05-27T07:32:00Z }"#;
        let value: Value = text.parse().expect("parses");
        let printed = Inline(&value).to_string();
        let again: Value = printed.parse().expect("prints parseable TOML");
        assert_eq!(again, value);
        assert_eq!(Inline(&Value::Float(1.0)).to_string(), "1.0");
        assert_eq!(Inline(&Value::Float(f64::NAN)).to_string(), "nan");
        assert_eq!(Key("qwen3*").to_string(), "\"qwen3*\"");
        assert_eq!(Key("ram_headroom").to_string(), "ram_headroom");
    }

    /// Whether `text` holds a control (but a tab) or an invisible character raw.
    fn has_unsafe(text: &str) -> bool {
        text.chars()
            .any(|c| (c.is_control() && c != '\t') || crate::chars::invisible(c))
    }

    /// A string with ESC, a zero-width space, a right-to-left override, a soft hyphen and a tag
    /// character prints as a TOML string holding none of them raw, which parses back to the
    /// original, in a value, an array, a table and a quoted key (review of #1923, N1).
    #[test]
    fn controls_and_invisible_characters_print_as_toml_escapes() {
        let hostile = "a\u{1b}[2Jb\u{200B}c\u{202E}d\u{AD}e\u{E0041}f\u{7}";
        let string = Value::String(hostile.to_owned());
        let shown = Inline(&string).to_string();
        assert_eq!(
            shown,
            r#""a\u001B[2Jb\u200Bc\u202Ed\u00ADe\U000E0041f\u0007""#
        );
        let mut table = toml::Table::new();
        table.insert(hostile.to_owned(), Value::Array(vec![string.clone()]));
        let table = Value::Table(table);
        for value in [string, table] {
            let shown = Inline(&value).to_string();
            assert!(!has_unsafe(&shown), "{shown:?}");
            let back: toml::Table = format!("v = {shown}").parse().expect("valid TOML");
            assert_eq!(back["v"], value, "{shown}");
        }
        assert!(!has_unsafe(&Key(hostile).to_string()));
    }

    /// A secret's file and variable forms print as TOML too, not Rust's `{:?}`.
    #[test]
    fn a_secret_source_prints_as_toml() {
        let source = SecretSource::File("/run/secrets/a\u{200B}b".into());
        let shown = source.to_string();
        assert_eq!(shown, r#"{ file = "/run/secrets/a\u200Bb" }"#);
        let back: toml::Table = format!("v = {shown}").parse().expect("valid TOML");
        assert_eq!(back["v"]["file"].as_str(), Some("/run/secrets/a\u{200B}b"));
        let shown = SecretSource::Env("V\u{202E}".into()).to_string();
        assert_eq!(shown, r#"{ env = "V\u202E" }"#);
    }
}
