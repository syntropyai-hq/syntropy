// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Secret<T>` and the forms a secret is written in (RFC-0054 §3, §10.1).

use std::fmt;
use std::path::PathBuf;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use zeroize::Zeroize;

use super::{ValueError, ValueErrorKind};

/// What a secret prints as, everywhere.
pub const REDACTED: &str = "[redacted]";

/// A secret value: an API key, a Hub token, an exporter header (RFC-0054 §10.1).
///
/// - `Debug`, `Display` and `Serialize` all produce `[redacted]`, so a secret never reaches
///   `config show`, a log, the event log or a debug bundle by accident.
/// - The value is read only through [`Secret::expose_for`], which names why, so every read is
///   findable with a search.
/// - The value's memory is zeroed on drop, through `zeroize`.
pub struct Secret<T: Zeroize>(T);

impl<T: Zeroize> Secret<T> {
    /// Wraps `value`.
    pub fn new(value: T) -> Self {
        Self(value)
    }

    /// The value, for the purpose `why` names, such as `"otlp authorization header"`.
    pub fn expose_for(&self, why: &'static str) -> &T {
        let _ = why;
        &self.0
    }
}

impl<T: Zeroize> Drop for Secret<T> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl<T: Zeroize + Clone> Clone for Secret<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Zeroize> fmt::Debug for Secret<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

impl<T: Zeroize> fmt::Display for Secret<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

/// Serializes as `[redacted]`: the value itself leaves only through [`Secret::expose_for`].
impl<T: Zeroize> Serialize for Secret<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(REDACTED)
    }
}

impl<'de, T: Zeroize + Deserialize<'de>> Deserialize<'de> for Secret<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(Secret)
    }
}

/// A secret as written in a config file: the value itself (`"Bearer …"`), a file that holds it
/// (`{ file = "/run/secrets/otlp" }`) or an environment variable (`{ env = "MY_VAR" }`)
/// (RFC-0054 §3, §10.1). Containers and service managers should use the file form.
///
/// Reading the file or variable is deferred to [`SecretSource::resolve`], so loading a config
/// never fails on a secret that the node does not use yet, and a rotated file is read fresh.
#[derive(Clone, Debug)]
pub enum SecretSource {
    /// The value, written in the file.
    Inline(Secret<String>),
    /// A file whose contents, without one trailing newline, are the value.
    File(PathBuf),
    /// An environment variable whose value is the secret.
    Env(String),
}

impl SecretSource {
    /// Reads the secret.
    pub fn resolve(&self) -> Result<Secret<String>, ValueError> {
        let unavailable = |why: String| ValueError::new(ValueErrorKind::SecretUnavailable, why);
        match self {
            SecretSource::Inline(secret) => Ok(secret.clone()),
            SecretSource::File(path) => {
                let mut text = std::fs::read_to_string(path).map_err(|e| {
                    unavailable(format!(
                        "cannot read the secret file {}: {e}",
                        path.display()
                    ))
                })?;
                if text.ends_with('\n') {
                    text.pop();
                    if text.ends_with('\r') {
                        text.pop();
                    }
                }
                Ok(Secret::new(text))
            }
            SecretSource::Env(var) => std::env::var(var)
                .map(Secret::new)
                .map_err(|e| unavailable(format!("cannot read the secret variable `{var}`: {e}"))),
        }
    }
}

/// Prints the file and variable forms as written, and an inline value as `[redacted]`.
impl fmt::Display for SecretSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecretSource::Inline(_) => f.write_str(REDACTED),
            SecretSource::File(path) => write!(f, "{{ file = {:?} }}", path.display().to_string()),
            SecretSource::Env(var) => write!(f, "{{ env = {var:?} }}"),
        }
    }
}

/// Serializes the file and variable forms as tables, and an inline value as `[redacted]`.
impl Serialize for SecretSource {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            SecretSource::Inline(secret) => secret.serialize(serializer),
            SecretSource::File(path) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("file", path)?;
                map.end()
            }
            SecretSource::Env(var) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("env", var)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for SecretSource {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(SourceVisitor)
    }
}

struct SourceVisitor;

impl<'de> Visitor<'de> for SourceVisitor {
    type Value = SecretSource;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a secret: a string, { file = \"/path\" } or { env = \"VAR\" }")
    }

    fn visit_str<E: de::Error>(self, s: &str) -> Result<SecretSource, E> {
        Ok(SecretSource::Inline(Secret::new(s.to_owned())))
    }

    fn visit_string<E: de::Error>(self, s: String) -> Result<SecretSource, E> {
        Ok(SecretSource::Inline(Secret::new(s)))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<SecretSource, A::Error> {
        const FIELDS: &[&str] = &["file", "env"];
        let key: String = map
            .next_key()?
            .ok_or_else(|| de::Error::custom("a secret table needs `file` or `env`"))?;
        let source = match key.as_str() {
            "file" => SecretSource::File(map.next_value()?),
            "env" => SecretSource::Env(map.next_value()?),
            other => return Err(de::Error::unknown_field(other, FIELDS)),
        };
        if map.next_key::<String>()?.is_some() {
            return Err(de::Error::custom(
                "a secret table has exactly one of `file` or `env`",
            ));
        }
        Ok(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_prints_the_value() {
        let secret = Secret::new("hunter2".to_owned());
        assert_eq!(format!("{secret}"), REDACTED);
        assert_eq!(format!("{secret:?}"), REDACTED);
        assert_eq!(serde_json::to_string(&secret).unwrap(), "\"[redacted]\"");
        assert_eq!(secret.expose_for("test"), "hunter2");

        let source: SecretSource = serde_json::from_str("\"hunter2\"").unwrap();
        assert_eq!(source.to_string(), REDACTED);
        assert!(!format!("{source:?}").contains("hunter2"));
        assert_eq!(serde_json::to_string(&source).unwrap(), "\"[redacted]\"");
    }

    #[test]
    fn reads_the_three_forms() {
        let file =
            std::env::temp_dir().join(format!("syntropy-config-secret-{}", std::process::id()));
        std::fs::write(&file, "from-file\n").unwrap();
        let json = serde_json::to_string(&serde_json::json!({ "file": file })).unwrap();
        let source: SecretSource = serde_json::from_str(&json).unwrap();
        assert_eq!(source.resolve().unwrap().expose_for("test"), "from-file");
        std::fs::remove_file(&file).unwrap();
        assert_eq!(
            source.resolve().unwrap_err().kind(),
            ValueErrorKind::SecretUnavailable
        );

        // PATH is set in every test environment, so it stands in for a secret variable.
        let source: SecretSource = serde_json::from_str(r#"{"env": "PATH"}"#).unwrap();
        assert_eq!(
            source.resolve().unwrap().expose_for("test"),
            &std::env::var("PATH").unwrap()
        );
        assert_eq!(source.to_string(), "{ env = \"PATH\" }");
        assert_eq!(serde_json::to_string(&source).unwrap(), r#"{"env":"PATH"}"#);

        let source: SecretSource = serde_json::from_str("\"inline\"").unwrap();
        assert_eq!(source.resolve().unwrap().expose_for("test"), "inline");
    }

    #[test]
    fn rejects_other_tables() {
        for json in [
            r#"{}"#,
            r#"{"path": "/x"}"#,
            r#"{"file": "/x", "env": "Y"}"#,
            "1",
        ] {
            assert!(
                serde_json::from_str::<SecretSource>(json).is_err(),
                "{json}"
            );
        }
    }
}
