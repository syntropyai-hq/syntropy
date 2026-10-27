// SPDX-License-Identifier: MIT OR Apache-2.0

//! A strict JSON value for the safetensors header and the shard index.
//!
//! `serde_json::Value` keeps the last of two equal keys and drops the first without a word.
//! A header that names one tensor twice, or gives one tensor two `data_offsets`, would then
//! parse as something its writer may not have meant. [`Json`] keeps objects as ordered key
//! lists and refuses a repeated key, so such a header fails instead. Numbers are a
//! non-negative integer that fits `u64`, or any other number as the `f64` `serde_json` reads,
//! which `config.json`'s fractional hyperparameters need ([`crate::config`]).
//! Nesting depth is bounded by `serde_json`'s recursion limit (128).

use std::fmt;

use serde::Deserialize;
use serde::de::{self, MapAccess, SeqAccess, Visitor};

/// A JSON value with ordered, duplicate-free objects.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    /// A non-negative integer that fits `u64`.
    Uint(u64),
    /// Any other number: negative, fractional or too large, as `serde_json` reads it.
    OtherNumber(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Parses `bytes` as one JSON value, with only whitespace after it.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut de = serde_json::Deserializer::from_slice(bytes);
        let value = Json::deserialize(&mut de).map_err(|e| e.to_string())?;
        de.end().map_err(|e| e.to_string())?;
        Ok(value)
    }

    /// The kind of value, for messages.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "a boolean",
            Json::Uint(_) | Json::OtherNumber(_) => "a number",
            Json::Str(_) => "a string",
            Json::Array(_) => "an array",
            Json::Object(_) => "an object",
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(JsonVisitor)
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Json;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Json, E> {
        Ok(Json::Bool(v))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Json, E> {
        Ok(Json::Uint(v))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Json, E> {
        Ok(u64::try_from(v).map_or(Json::OtherNumber(v as f64), Json::Uint))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Json, E> {
        Ok(Json::OtherNumber(v))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Json, E> {
        Ok(Json::Str(v.to_owned()))
    }

    fn visit_string<E: de::Error>(self, v: String) -> Result<Json, E> {
        Ok(Json::Str(v))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = seq.next_element::<Json>()? {
            out.push(v);
        }
        Ok(Json::Array(out))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        let mut out: Vec<(String, Json)> = Vec::new();
        // An ordered set (RFC-0060 §2.4 rule T3), though only membership reads it.
        let mut seen = std::collections::BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate key {key:?}")));
            }
            let value = map.next_value::<Json>()?;
            out.push((key, value));
        }
        Ok(Json::Object(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_values_and_refuses_duplicate_keys() {
        assert_eq!(
            Json::parse(br#"{"a": [1, -2, 1.5, "x", null, true]}  "#).unwrap(),
            Json::Object(vec![(
                "a".into(),
                Json::Array(vec![
                    Json::Uint(1),
                    Json::OtherNumber(-2.0),
                    Json::OtherNumber(1.5),
                    Json::Str("x".into()),
                    Json::Null,
                    Json::Bool(true),
                ])
            )])
        );
        let e = Json::parse(br#"{"a": 1, "a": 2}"#).unwrap_err();
        assert!(e.contains("duplicate key"), "{e}");
        assert!(Json::parse(br#"{"a": {"b": 1, "b": 1}}"#).is_err());
        assert!(Json::parse(b"{} x").is_err());
        assert_eq!(
            Json::parse(b"18446744073709551616").unwrap(),
            Json::OtherNumber(18_446_744_073_709_551_616.0)
        );
    }

    #[test]
    fn deep_nesting_is_an_error_not_a_crash() {
        let deep = "[".repeat(100_000);
        assert!(Json::parse(deep.as_bytes()).is_err());
    }
}
