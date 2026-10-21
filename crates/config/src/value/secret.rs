// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Secret<T>` and the forms a secret is written in (RFC-0054 §3, §10.1).

use std::fmt;
use std::path::PathBuf;

use serde::de::{self, EnumAccess, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use zeroize::Zeroize;

use super::{ValueError, ValueErrorKind};

/// What a secret prints as, everywhere.
pub const REDACTED: &str = "[redacted]";

/// The error for a value that is not one of a secret's forms. It never includes the value: a
/// secret written without quotes, such as an all-digit token, parses as a number, and serde's
/// own message would copy it into the load error and the logs.
const NOT_A_SECRET: &str = "a secret must be a string, { file = \"…\" } or { env = \"…\" }; the value given is none of \
     these, and is not shown";

/// The error for a secret that does not deserialize as its type. `T`'s own error is dropped,
/// since it can quote the value, such as `u8`'s "invalid value: integer …".
const SECRET_WRONG_FORM: &str =
    "a secret must be a value of its key's type; the value given is not, and is not shown";

/// The error for a key in a secret table other than `file` and `env`. The key is not shown,
/// in case the secret was written as one.
const NOT_A_SECRET_TABLE: &str =
    "a secret table has one key, `file` or `env`; this one is neither, and is not shown";

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

/// A value that does not fit `T` fails with a fixed message: `T`'s own error can quote it.
impl<'de, T: Zeroize + Deserialize<'de>> Deserialize<'de> for Secret<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer)
            .map(Secret)
            .map_err(|_| de::Error::custom(SECRET_WRONG_FORM))
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
            SecretSource::Env(var) => std::env::var(var).map(Secret::new).map_err(|e| {
                unavailable(format!(
                    "cannot read the secret variable `{var}`: {}",
                    describe_var_error(&e)
                ))
            }),
        }
    }
}

/// Why a variable could not be read, without its value: `VarError::NotUnicode`'s own `Display`
/// prints the value's bytes, which here are the secret.
fn describe_var_error(error: &std::env::VarError) -> &'static str {
    match error {
        std::env::VarError::NotPresent => "it is not set",
        std::env::VarError::NotUnicode(_) => "its value is not valid UTF-8",
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
        let key: String = map
            .next_key()?
            .ok_or_else(|| de::Error::custom("a secret table needs `file` or `env`"))?;
        let source = match key.as_str() {
            "file" => SecretSource::File(map.next_value()?),
            "env" => SecretSource::Env(map.next_value()?),
            _ => return Err(de::Error::custom(NOT_A_SECRET_TABLE)),
        };
        if map.next_key::<String>()?.is_some() {
            return Err(de::Error::custom(
                "a secret table has exactly one of `file` or `env`",
            ));
        }
        Ok(source)
    }

    // Every other form a value can take fails with one fixed message, never serde's default
    // `invalid_type`, which quotes the value.

    fn visit_bool<E: de::Error>(self, _: bool) -> Result<SecretSource, E> {
        Err(E::custom(NOT_A_SECRET))
    }

    fn visit_i64<E: de::Error>(self, _: i64) -> Result<SecretSource, E> {
        Err(E::custom(NOT_A_SECRET))
    }

    fn visit_i128<E: de::Error>(self, _: i128) -> Result<SecretSource, E> {
        Err(E::custom(NOT_A_SECRET))
    }

    fn visit_u64<E: de::Error>(self, _: u64) -> Result<SecretSource, E> {
        Err(E::custom(NOT_A_SECRET))
    }

    fn visit_u128<E: de::Error>(self, _: u128) -> Result<SecretSource, E> {
        Err(E::custom(NOT_A_SECRET))
    }

    fn visit_f64<E: de::Error>(self, _: f64) -> Result<SecretSource, E> {
        Err(E::custom(NOT_A_SECRET))
    }

    fn visit_bytes<E: de::Error>(self, _: &[u8]) -> Result<SecretSource, E> {
        Err(E::custom(NOT_A_SECRET))
    }

    fn visit_none<E: de::Error>(self) -> Result<SecretSource, E> {
        Err(E::custom(NOT_A_SECRET))
    }

    fn visit_some<D: Deserializer<'de>>(self, _: D) -> Result<SecretSource, D::Error> {
        Err(de::Error::custom(NOT_A_SECRET))
    }

    fn visit_unit<E: de::Error>(self) -> Result<SecretSource, E> {
        Err(E::custom(NOT_A_SECRET))
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(self, _: D) -> Result<SecretSource, D::Error> {
        Err(de::Error::custom(NOT_A_SECRET))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, _: A) -> Result<SecretSource, A::Error> {
        Err(de::Error::custom(NOT_A_SECRET))
    }

    fn visit_enum<A: EnumAccess<'de>>(self, _: A) -> Result<SecretSource, A::Error> {
        Err(de::Error::custom(NOT_A_SECRET))
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
    fn a_variable_error_never_holds_the_value() {
        assert_eq!(
            describe_var_error(&std::env::VarError::NotPresent),
            "it is not set"
        );
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let value = std::ffi::OsString::from_vec(b"hunter2\xff".to_vec());
            let error = std::env::VarError::NotUnicode(value);
            // The standard library's message would carry the value.
            assert!(error.to_string().contains("hunter2"));
            assert!(!describe_var_error(&error).contains("hunter2"));
        }
        let source = SecretSource::Env("SYNTROPY_CONFIG_TEST_UNSET_SECRET_VARIABLE".to_owned());
        let err = source.resolve().unwrap_err();
        assert!(err.message().ends_with("it is not set"), "{err}");
    }

    use serde::de::IntoDeserializer;
    use serde::de::value::{Error as ValueDeError, MapDeserializer, SeqDeserializer};

    /// What a rejected secret's error says, for `SecretSource` and for `Secret<T>`.
    fn source_error<'de, D: Deserializer<'de, Error = ValueDeError>>(d: D) -> String {
        SecretSource::deserialize(d).unwrap_err().to_string()
    }

    fn secret_error<'de, T, D>(d: D) -> String
    where
        T: Zeroize + Deserialize<'de> + fmt::Debug,
        D: Deserializer<'de, Error = ValueDeError>,
    {
        Secret::<T>::deserialize(d).unwrap_err().to_string()
    }

    /// The error names the accepted forms and none of the value's text.
    fn assert_hides(message: &str, value: &str) {
        assert!(
            !message.contains(value),
            "the error shows the value: {message}"
        );
        assert!(
            message.contains("a secret must be"),
            "not the fixed message: {message}"
        );
    }

    #[test]
    fn an_integer_secret_error_hides_the_value() {
        let digits = |m: &str| m.chars().any(|c| c.is_ascii_digit());
        for message in [
            source_error(987_654_321i64.into_deserializer()),
            source_error(987_654_321u64.into_deserializer()),
            secret_error::<String, _>(987_654_321i64.into_deserializer()),
        ] {
            assert_hides(&message, "987654321");
            assert!(!digits(&message), "the error holds digits: {message}");
        }
    }

    #[test]
    fn a_float_secret_error_hides_the_value() {
        for message in [
            source_error(0.123_456_789f64.into_deserializer()),
            secret_error::<String, _>(0.123_456_789f64.into_deserializer()),
        ] {
            assert_hides(&message, "123456789");
            assert!(!message.chars().any(|c| c.is_ascii_digit()), "{message}");
        }
    }

    #[test]
    fn a_bool_secret_error_hides_the_value() {
        for message in [
            source_error(true.into_deserializer()),
            secret_error::<String, _>(true.into_deserializer()),
        ] {
            assert_hides(&message, "true");
        }
    }

    #[test]
    fn an_array_secret_error_hides_the_value() {
        let strings = || SeqDeserializer::<_, ValueDeError>::new(["seq-secret-7731"].into_iter());
        let bytes = || SeqDeserializer::<_, ValueDeError>::new([300u64, 7731].into_iter());
        for message in [
            source_error(strings()),
            secret_error::<String, _>(strings()),
            // A byte secret whose element is out of range: `u8`'s error would print `300`.
            secret_error::<Vec<u8>, _>(bytes()),
        ] {
            assert_hides(&message, "seq-secret-7731");
            assert!(
                !message.contains("300") && !message.contains("7731"),
                "{message}"
            );
        }
    }

    #[test]
    fn a_table_key_is_not_echoed() {
        let table = MapDeserializer::<_, ValueDeError>::new([("tok-3141-key", "x")].into_iter());
        let message = source_error(table);
        assert!(
            !message.contains("tok-3141-key"),
            "the error shows the key: {message}"
        );
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
