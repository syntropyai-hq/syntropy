// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Reserve`: memory kept back from a tier, as a floor, a share and an optional cap
//! (RFC-0054 §3).

use std::fmt;
use std::str::FromStr;

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use syntropy_core::Bytes;

use super::bytes::ByteSizeVisitor;
use super::fraction::FractionVisitor;
use super::{ByteSize, Fraction, ValueError, serde_error};

/// Memory kept back from a tier: `"8GiB"`, `"10%"`, `{ min = "6GiB", fraction = 0.10 }` or
/// `{ min = "6GiB", fraction = 0.10, max = "32GiB" }` (RFC-0054 §3).
///
/// It resolves against the tier's physical capacity as
/// `clamp(max(min, fraction × capacity), .., max)` ([`Reserve::resolve`]).
///
/// - `"8GiB"` is `{ min = "8GiB", fraction = 0 }`; `"10%"` is `{ min = "0B", fraction = 0.1 }`.
/// - In a table, each field is optional: `min` and `fraction` default to zero, `max` to none.
/// - `max` below `min` is rejected.
/// - Output uses the short forms where they fit and the table form otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reserve {
    /// The floor.
    pub min: Bytes,
    /// The share of capacity, in [0, 1].
    pub fraction: f32,
    /// The cap, if any.
    pub max: Option<Bytes>,
}

impl Reserve {
    /// A reserve of `min` bytes.
    pub const fn bytes(min: Bytes) -> Self {
        Self {
            min,
            fraction: 0.0,
            max: None,
        }
    }

    /// A reserve with every field, checked.
    pub fn new(min: Bytes, fraction: Fraction, max: Option<Bytes>) -> Result<Self, ValueError> {
        let reserve = Self {
            min,
            fraction: fraction.get(),
            max,
        };
        reserve.validate()?;
        Ok(reserve)
    }

    /// Checks that `fraction` lies in [0, 1] and `max`, if set, is at least `min`.
    pub fn validate(&self) -> Result<(), ValueError> {
        Fraction::new(self.fraction)?;
        if let Some(max) = self.max
            && max < self.min
        {
            return Err(ValueError::out_of_range(format!(
                "a reserve's max ({}) is below its min ({})",
                ByteSize(max),
                ByteSize(self.min)
            )));
        }
        Ok(())
    }

    /// The bytes kept back from a tier of `capacity`: `clamp(max(min, fraction × capacity), ..,
    /// max)`. The share rounds up to a whole byte, since it is memory kept back.
    pub fn resolve(&self, capacity: Bytes) -> Bytes {
        let share = (f64::from(self.fraction) * capacity.0 as f64).ceil();
        // `as` saturates; a share above u64::MAX cannot exceed any real capacity anyway.
        let wanted = self.min.max(Bytes(share as u64));
        match self.max {
            Some(max) => wanted.min(max),
            None => wanted,
        }
    }
}

impl fmt::Display for Reserve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // A fraction outside [0, 1] (set through the public field, unchecked) prints as the
        // table form with its raw value, which `from_str` then rejects.
        let fraction = Fraction::new(self.fraction).ok();
        match (self.min, self.fraction, self.max, fraction) {
            (min, 0.0, None, _) => write!(f, "{}", ByteSize(min)),
            (Bytes(0), _, None, Some(fraction)) => f.write_str(&fraction.to_percent_string()),
            (min, raw, max, _) => {
                write!(f, "{{ min = \"{}\", fraction = {raw}", ByteSize(min))?;
                if let Some(max) = max {
                    write!(f, ", max = \"{}\"", ByteSize(max))?;
                }
                f.write_str(" }")
            }
        }
    }
}

impl FromStr for Reserve {
    type Err = ValueError;

    fn from_str(s: &str) -> Result<Self, ValueError> {
        let trimmed = s.trim();
        let reserve = if trimmed.starts_with('{') {
            parse_table(trimmed)?
        } else if trimmed.ends_with('%') {
            Reserve {
                min: Bytes(0),
                fraction: trimmed.parse::<Fraction>()?.get(),
                max: None,
            }
        } else {
            Reserve::bytes(trimmed.parse::<ByteSize>()?.bytes())
        };
        reserve.validate()?;
        Ok(reserve)
    }
}

/// Parses the inline-table form `{ min = "6GiB", fraction = 0.10, max = "32GiB" }`, the way
/// TOML writes it, so the printed table form parses back.
fn parse_table(s: &str) -> Result<Reserve, ValueError> {
    let bad = |why: &str| {
        ValueError::bad_unit(format!(
            "`{s}` is not a reserve: {why}; write {{ min = \"6GiB\", fraction = 0.10, max = \
             \"32GiB\" }}, with any of the three fields"
        ))
    };
    let inner = s
        .strip_prefix('{')
        .and_then(|rest| rest.strip_suffix('}'))
        .ok_or_else(|| bad("the table is not closed"))?;
    let mut table = Table::default();
    for entry in inner.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            if inner.trim().is_empty() {
                continue;
            }
            return Err(bad("an entry is empty"));
        }
        let (key, value) = entry
            .split_once('=')
            .ok_or_else(|| bad(&format!("`{entry}` is not `key = value`")))?;
        let value = value.trim();
        let text = match value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
            Some(quoted) if !quoted.contains(['"', '\\']) => Text::Quoted(quoted),
            Some(_) => return Err(bad("a string holds a quote or backslash")),
            None => Text::Bare(value),
        };
        table.set(key.trim(), text).map_err(|why| bad(&why))?;
    }
    Ok(table.finish())
}

enum Text<'a> {
    Quoted(&'a str),
    Bare(&'a str),
}

#[derive(Default)]
struct Table {
    min: Option<Bytes>,
    fraction: Option<f32>,
    max: Option<Bytes>,
}

impl Table {
    fn set(&mut self, key: &str, text: Text<'_>) -> Result<(), String> {
        let size = |text: Text<'_>| -> Result<Bytes, String> {
            match text {
                Text::Quoted(s) => s.parse::<ByteSize>().map(ByteSize::bytes),
                Text::Bare(s) => s
                    .parse::<u64>()
                    .map(Bytes)
                    .map_err(|_| ValueError::bad_unit(format!("`{s}` is not a size"))),
            }
            .map_err(|e| e.to_string())
        };
        let duplicate = || format!("`{key}` is set twice");
        match key {
            "min" if self.min.is_none() => self.min = Some(size(text)?),
            "max" if self.max.is_none() => self.max = Some(size(text)?),
            "fraction" if self.fraction.is_none() => {
                let (Text::Quoted(s) | Text::Bare(s)) = text;
                self.fraction = Some(s.parse::<Fraction>().map_err(|e| e.to_string())?.get());
            }
            "min" | "max" | "fraction" => return Err(duplicate()),
            _ => return Err(format!("`{key}` is not a field")),
        }
        Ok(())
    }

    fn finish(self) -> Reserve {
        Reserve {
            min: self.min.unwrap_or(Bytes(0)),
            fraction: self.fraction.unwrap_or(0.0),
            max: self.max,
        }
    }
}

impl Serialize for Reserve {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.max.is_none() && (self.fraction == 0.0 || self.min == Bytes(0)) {
            return serializer.collect_str(self);
        }
        let len = if self.max.is_some() { 3 } else { 2 };
        let mut map = serializer.serialize_map(Some(len))?;
        map.serialize_entry("min", &ByteSize(self.min))?;
        let fraction = Fraction::new(self.fraction).map_err(serde::ser::Error::custom)?;
        map.serialize_entry("fraction", &fraction)?;
        if let Some(max) = self.max {
            map.serialize_entry("max", &ByteSize(max))?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for Reserve {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ReserveVisitor)
    }
}

struct ReserveVisitor;

impl<'de> Visitor<'de> for ReserveVisitor {
    type Value = Reserve;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "a reserve: a size (\"8GiB\"), a percentage (\"10%\") or a table { min, fraction, max }",
        )
    }

    fn visit_u64<E: de::Error>(self, n: u64) -> Result<Reserve, E> {
        Ok(Reserve::bytes(Bytes(n)))
    }

    fn visit_i64<E: de::Error>(self, n: i64) -> Result<Reserve, E> {
        ByteSizeVisitor
            .visit_i64(n)
            .map(|size| Reserve::bytes(size.bytes()))
    }

    fn visit_str<E: de::Error>(self, s: &str) -> Result<Reserve, E> {
        s.parse().map_err(serde_error)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Reserve, A::Error> {
        struct Size(ByteSize);
        impl<'de> Deserialize<'de> for Size {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                d.deserialize_any(ByteSizeVisitor).map(Size)
            }
        }
        struct Share(Fraction);
        impl<'de> Deserialize<'de> for Share {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                d.deserialize_any(FractionVisitor).map(Share)
            }
        }
        let mut table = Table::default();
        while let Some(key) = map.next_key::<String>()? {
            let duplicate = || serde_error(ValueError::bad_unit(format!("`{key}` is set twice")));
            match key.as_str() {
                "min" if table.min.is_none() => {
                    table.min = Some(map.next_value::<Size>()?.0.bytes())
                }
                "max" if table.max.is_none() => {
                    table.max = Some(map.next_value::<Size>()?.0.bytes())
                }
                "fraction" if table.fraction.is_none() => {
                    table.fraction = Some(map.next_value::<Share>()?.0.get());
                }
                "min" | "max" | "fraction" => return Err(duplicate()),
                other => {
                    return Err(de::Error::unknown_field(other, &["min", "fraction", "max"]));
                }
            }
        }
        let reserve = table.finish();
        reserve.validate().map_err(serde_error)?;
        Ok(reserve)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    const GIB: u64 = 1 << 30;

    fn reserve(min: u64, fraction: f32, max: Option<u64>) -> Reserve {
        Reserve {
            min: Bytes(min),
            fraction,
            max: max.map(Bytes),
        }
    }

    #[test]
    fn parses_the_rfc_forms() {
        assert_eq!("8GiB".parse(), Ok(reserve(8 * GIB, 0.0, None)));
        assert_eq!("10%".parse(), Ok(reserve(0, 0.1, None)));
        assert_eq!(
            "{ min = \"6GiB\", fraction = 0.10 }".parse(),
            Ok(reserve(6 * GIB, 0.1, None))
        );
        assert_eq!(
            "{ min = \"6GiB\", fraction = 0.10, max = \"32GiB\" }".parse(),
            Ok(reserve(6 * GIB, 0.1, Some(32 * GIB)))
        );
        assert_eq!("{}".parse(), Ok(reserve(0, 0.0, None)));
        let json = |s: &str| serde_json::from_str::<Reserve>(s).ok();
        assert_eq!(
            json(r#"{"min": "6GiB", "fraction": 0.1, "max": "32GiB"}"#),
            Some(reserve(6 * GIB, 0.1, Some(32 * GIB)))
        );
        assert_eq!(json(r#"{"fraction": "10%"}"#), Some(reserve(0, 0.1, None)));
        assert_eq!(json("4096"), Some(reserve(4096, 0.0, None)));
    }

    #[test]
    fn rejects_bad_forms() {
        for s in [
            "",
            "8",
            "110%",
            "{ min = \"6GiB\"",
            "{ min = \"6GiB\", min = \"1GiB\" }",
            "{ mni = \"6GiB\" }",
            "{ min = \"32GiB\", max = \"6GiB\" }",
            "{ min = \"6GiB\",, fraction = 0.1 }",
        ] {
            assert!(s.parse::<Reserve>().is_err(), "{s:?} should not parse");
        }
        assert!(serde_json::from_str::<Reserve>(r#"{"min": "1GiB", "extra": 1}"#).is_err());
        assert!(serde_json::from_str::<Reserve>(r#"{"min": "8GiB", "max": "1GiB"}"#).is_err());
    }

    #[test]
    fn resolves_as_the_rfc_formula() {
        let personal = reserve(6 * GIB, 0.1, None);
        assert_eq!(personal.resolve(Bytes(16 * GIB)), Bytes(6 * GIB));
        // 0.1f32 is 0.100000001490116…, so 10% of 128 GiB is 204.8 bytes above a decimal
        // tenth, rounded up.
        assert_eq!(personal.resolve(Bytes(128 * GIB)), Bytes(13_743_895_552));
        let capped = reserve(6 * GIB, 0.1, Some(8 * GIB));
        assert_eq!(capped.resolve(Bytes(1024 * GIB)), Bytes(8 * GIB));
    }

    #[test]
    fn prints_short_forms_where_they_fit() {
        assert_eq!(reserve(8 * GIB, 0.0, None).to_string(), "8GiB");
        assert_eq!(reserve(0, 0.1, None).to_string(), "10%");
        assert_eq!(
            reserve(6 * GIB, 0.1, None).to_string(),
            "{ min = \"6GiB\", fraction = 0.1 }"
        );
        assert_eq!(
            reserve(6 * GIB, 0.1, Some(32 * GIB)).to_string(),
            "{ min = \"6GiB\", fraction = 0.1, max = \"32GiB\" }"
        );
        let invalid = reserve(0, 1.5, None);
        assert_eq!(invalid.to_string(), "{ min = \"0B\", fraction = 1.5 }");
        assert!(invalid.to_string().parse::<Reserve>().is_err());
    }

    fn any_reserve() -> impl Strategy<Value = Reserve> {
        (
            any::<u64>(),
            0.0f32..=1.0,
            proptest::option::of(any::<u64>()),
        )
            .prop_map(|(min, fraction, max)| reserve(min, fraction, max.map(|m| m.max(min))))
    }

    proptest! {
        #[test]
        fn printed_form_parses_back(r in any_reserve()) {
            let back = r.to_string().parse::<Reserve>().unwrap();
            prop_assert_eq!(back.min, r.min);
            prop_assert_eq!(back.fraction.to_bits(), (r.fraction + 0.0).to_bits());
            prop_assert_eq!(back.max, r.max);
            let json = serde_json::to_string(&r).unwrap();
            let back = serde_json::from_str::<Reserve>(&json).unwrap();
            prop_assert_eq!(back.min, r.min);
            prop_assert_eq!(back.fraction.to_bits(), (r.fraction + 0.0).to_bits());
            prop_assert_eq!(back.max, r.max);
        }
    }
}
