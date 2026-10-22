// SPDX-License-Identifier: MIT OR Apache-2.0

//! A serde deserializer over [`DefaultValue`], so a default written in an attribute
//! deserializes through the same `Deserialize` impl as a value read from a file.

use serde::de::value::{Error, MapDeserializer, SeqDeserializer};
use serde::de::{self, IntoDeserializer, Visitor};
use serde::forward_to_deserialize_any;

use crate::schema::DefaultValue;

/// Deserializes one [`DefaultValue`].
pub struct ValueDeserializer<'a> {
    value: &'a DefaultValue,
}

impl<'a> ValueDeserializer<'a> {
    pub(crate) fn new(value: &'a DefaultValue) -> Self {
        Self { value }
    }
}

impl<'de, 'a> de::Deserializer<'de> for ValueDeserializer<'a> {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match *self.value {
            DefaultValue::Bool(b) => visitor.visit_bool(b),
            DefaultValue::Int(i) => visitor.visit_i64(i),
            DefaultValue::Float(x) => visitor.visit_f64(x),
            DefaultValue::Str(s) => visitor.visit_str(s),
            DefaultValue::Array(items) => {
                let mut seq = SeqDeserializer::new(items.iter().map(ValueDeserializer::new));
                let value = visitor.visit_seq(&mut seq)?;
                seq.end()?;
                Ok(value)
            }
            DefaultValue::Table(entries) => {
                let mut map = MapDeserializer::new(
                    entries.iter().map(|(k, v)| (*k, ValueDeserializer::new(v))),
                );
                let value = visitor.visit_map(&mut map)?;
                map.end()?;
                Ok(value)
            }
        }
    }

    /// A written value is always present; `default = unset` never reaches a deserializer.
    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_some(self)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_newtype_struct(self)
    }

    /// Unit variants are written as strings (`default = "personal"`), as in TOML.
    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        match *self.value {
            DefaultValue::Str(s) => visitor.visit_enum(s.into_deserializer()),
            _ => self.deserialize_any(visitor),
        }
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf unit unit_struct seq tuple tuple_struct map struct identifier
        ignored_any
    }
}

impl<'de, 'a> IntoDeserializer<'de, Error> for ValueDeserializer<'a> {
    type Deserializer = Self;

    fn into_deserializer(self) -> Self {
        self
    }
}

/// Deserializes "no value": an `Option` becomes `None`, and anything else is an error. The
/// default check uses it for `default = unset`.
pub(crate) struct UnsetDeserializer;

impl<'de> de::Deserializer<'de> for UnsetDeserializer {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, _visitor: V) -> Result<V::Value, Error> {
        Err(de::Error::custom(
            "`default = unset` needs an `Option` key: this key always has a value",
        ))
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_none()
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf unit unit_struct newtype_struct seq tuple tuple_struct map struct enum
        identifier ignored_any
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(rename_all = "kebab-case")]
    enum Mode {
        Personal,
        UnderPressure,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Pair {
        interactive: String,
        background: String,
    }

    #[test]
    fn deserializes_every_shape() {
        assert_eq!(DefaultValue::Bool(true).deserialize::<bool>(), Ok(true));
        assert_eq!(DefaultValue::Int(64).deserialize::<u32>(), Ok(64));
        assert_eq!(DefaultValue::Int(1).deserialize::<f32>(), Ok(1.0));
        assert_eq!(DefaultValue::Float(0.5).deserialize::<f64>(), Ok(0.5));
        assert_eq!(
            DefaultValue::Str("x").deserialize::<Option<String>>(),
            Ok(Some("x".to_owned()))
        );
        assert_eq!(
            DefaultValue::Str("under-pressure").deserialize::<Mode>(),
            Ok(Mode::UnderPressure)
        );
        assert_eq!(
            DefaultValue::Str("personal").deserialize::<Mode>(),
            Ok(Mode::Personal)
        );
        assert_eq!(
            DefaultValue::Array(&[DefaultValue::Str("a"), DefaultValue::Str("b")])
                .deserialize::<Vec<String>>(),
            Ok(vec!["a".to_owned(), "b".to_owned()])
        );
        assert_eq!(
            DefaultValue::Table(&[])
                .deserialize::<BTreeMap<String, String>>()
                .map(|m| m.len()),
            Ok(0)
        );
        assert_eq!(
            DefaultValue::Table(&[
                ("interactive", DefaultValue::Str("6tok/s")),
                ("background", DefaultValue::Str("1tok/s")),
            ])
            .deserialize::<Pair>(),
            Ok(Pair {
                interactive: "6tok/s".to_owned(),
                background: "1tok/s".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_the_wrong_shape() {
        assert!(DefaultValue::Str("x").deserialize::<u32>().is_err());
        assert!(DefaultValue::Int(-1).deserialize::<u32>().is_err());
        assert!(DefaultValue::Str("other").deserialize::<Mode>().is_err());
    }

    #[test]
    fn unset_fits_only_an_option() {
        assert_eq!(Option::<u32>::deserialize(UnsetDeserializer), Ok(None));
        assert!(u32::deserialize(UnsetDeserializer).is_err());
    }
}
