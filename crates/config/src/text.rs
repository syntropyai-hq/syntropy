// SPDX-License-Identifier: MIT OR Apache-2.0

//! Values and key paths printed as TOML, so a value `config show` prints can be pasted back
//! into a file (RFC-0054 §3).

use std::fmt;

use toml::Value;

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

/// Whether a key holds a secret anywhere in its Rust type as the section writes it:
/// `Secret<…>`, `SecretSource`, or a type built from them, such as
/// `BTreeMap<String, Secret<String>>`.
pub(crate) fn is_secret(key: &KeySchema) -> bool {
    key.ty.contains("Secret")
}

/// Whether a key's type is exactly `SecretSource`, optional or not. Its file and variable forms
/// (`{ file = … }`, `{ env = … }`) name where the secret is, not the secret, so they may print.
fn is_secret_source(key: &KeySchema) -> bool {
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
}
