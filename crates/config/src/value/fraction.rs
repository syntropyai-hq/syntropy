// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Fraction`'s forms: a share of something, in [0, 1], written as a number (`0.2`) or a
//! percentage (`"20%"`) (RFC-0054 §3). The type is `syntropy-core`'s.
//!
//! Output is the shortest decimal that reads back as the same `f32`, such as `0.2`. A
//! percentage is converted by moving the decimal point, so `"20%"` and `0.20` are the same
//! value. Negative zero is stored as zero.

use std::fmt;

use serde::de::{self, Visitor};
use serde::{Deserializer, Serializer};
use syntropy_core::policy::Fraction;

use super::form::Form;
use super::{ValueError, serde_error, split_decimal, split_number};

/// `value` as a [`Fraction`], or the error a key reports for a value outside [0, 1].
pub(crate) fn fraction(value: f32) -> Result<Fraction, ValueError> {
    Fraction::new(value).ok_or_else(|| {
        ValueError::out_of_range(format!(
            "{value} is not a fraction: it must lie between 0 and 1, or 0% and 100%"
        ))
    })
}

/// The share as a percentage string, such as `20%`, that parses back to the same value.
pub(crate) fn percent_string(share: Fraction) -> String {
    shift_point_right(&share.get().to_string(), 2) + "%"
}

impl Form for Fraction {
    fn deserialize_form<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(FractionVisitor)
    }

    fn serialize_form<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.get())
    }

    fn parse_text(s: &str) -> Result<Self, ValueError> {
        let bad = || {
            ValueError::bad_unit(format!(
                "`{s}` is not a fraction: write a number between 0 and 1, such as 0.2, or a \
                 percentage, such as \"20%\""
            ))
        };
        let trimmed = s.trim();
        let (number, rest) = split_number(trimmed);
        let Some((int, frac)) = split_decimal(number) else {
            return Err(bad());
        };
        let decimal = match rest {
            "" if frac.is_empty() => int.to_owned(),
            "" => format!("{int}.{frac}"),
            "%" => shift_point_left(int, frac, 2),
            _ => return Err(bad()),
        };
        let value: f32 = decimal.parse().map_err(|_| bad())?;
        fraction(value)
    }

    fn fmt_text(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

/// `int.frac` divided by 10^places, as decimal text.
fn shift_point_left(int: &str, frac: &str, places: usize) -> String {
    let padded = format!("{:0>width$}", int, width = places + 1);
    let (head, tail) = padded.split_at(padded.len() - places);
    format!("{head}.{tail}{frac}")
}

/// The decimal text `number` multiplied by 10^places, as decimal text. `number` is what
/// `f32`'s `Display` prints: digits with at most one point, never an exponent.
fn shift_point_right(number: &str, places: usize) -> String {
    let (int, frac) = number.split_once('.').unwrap_or((number, ""));
    let frac = format!("{frac:0<places$}");
    let (moved, rest) = frac.split_at(places);
    let int = format!("{int}{moved}");
    let int = int.trim_start_matches('0');
    let int = if int.is_empty() { "0" } else { int };
    if rest.is_empty() {
        int.to_owned()
    } else {
        format!("{int}.{rest}")
    }
}

pub(super) struct FractionVisitor;

impl Visitor<'_> for FractionVisitor {
    type Value = Fraction;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a number between 0 and 1, or a percentage such as \"20%\"")
    }

    fn visit_f64<E: de::Error>(self, x: f64) -> Result<Fraction, E> {
        // The shortest text of `x` read as f32 rounds once, as `parse_text` does.
        Fraction::parse_text(&x.to_string()).map_err(serde_error)
    }

    fn visit_f32<E: de::Error>(self, x: f32) -> Result<Fraction, E> {
        fraction(x).map_err(serde_error)
    }

    fn visit_i64<E: de::Error>(self, n: i64) -> Result<Fraction, E> {
        match n {
            0 => Ok(Fraction::ZERO),
            1 => Ok(Fraction::ONE),
            _ => Err(serde_error(ValueError::out_of_range(format!(
                "{n} is not a fraction: it must lie between 0 and 1, or 0% and 100%"
            )))),
        }
    }

    fn visit_u64<E: de::Error>(self, n: u64) -> Result<Fraction, E> {
        self.visit_i64(i64::try_from(n).unwrap_or(i64::MAX))
    }

    fn visit_str<E: de::Error>(self, s: &str) -> Result<Fraction, E> {
        Fraction::parse_text(s).map_err(serde_error)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::value::form::{Shown, parse};

    /// Reads JSON the way a key with `serde(with = "syntropy_config::form")` does.
    fn json(s: &str) -> Option<f32> {
        let mut de = serde_json::Deserializer::from_str(s);
        Fraction::deserialize_form(&mut de).ok().map(Fraction::get)
    }

    #[test]
    fn parses_numbers_and_percentages() {
        assert_eq!(parse::<Fraction>("0.2").map(Fraction::get), Ok(0.2));
        assert_eq!(parse::<Fraction>("20%").map(Fraction::get), Ok(0.2));
        assert_eq!(parse::<Fraction>("12.5%").map(Fraction::get), Ok(0.125));
        assert_eq!(parse::<Fraction>("100%"), Ok(Fraction::ONE));
        assert_eq!(parse::<Fraction>("0.5%").map(Fraction::get), Ok(0.005));
        assert_eq!(parse::<Fraction>("1"), Ok(Fraction::ONE));
        assert_eq!(json("0.1"), Some(0.1));
        assert_eq!(json("0"), Some(0.0));
        assert_eq!(json("\"10%\""), Some(0.1));
    }

    #[test]
    fn rejects_values_outside_zero_to_one() {
        for s in [
            "1.1", "101%", "-0.1", "", "%", "20 %%", "0.2x", "nan", "inf",
        ] {
            assert!(parse::<Fraction>(s).is_err(), "{s:?} should not parse");
        }
        assert!(json("2").is_none());
        assert!(fraction(f32::NAN).is_err());
        assert_eq!(fraction(-0.0).map(|f| f.get().to_bits()), Ok(0));
    }

    #[test]
    fn prints_the_shortest_decimal() {
        let shown = |x: f32| Shown(&fraction(x).unwrap()).to_string();
        assert_eq!(shown(0.2), "0.2");
        assert_eq!(shown(0.015625), "0.015625");
        assert_eq!(shown(1.0), "1");
        assert_eq!(percent_string(fraction(0.2).unwrap()), "20%");
        assert_eq!(percent_string(fraction(0.015625).unwrap()), "1.5625%");
        assert_eq!(percent_string(Fraction::ONE), "100%");
        assert_eq!(percent_string(Fraction::ZERO), "0%");
    }

    proptest! {
        #[test]
        fn printed_form_parses_back(x in 0.0f32..=1.0) {
            let f = fraction(x).unwrap();
            let back = parse::<Fraction>(&Shown(&f).to_string()).unwrap();
            prop_assert_eq!(back.get().to_bits(), f.get().to_bits());
            let back = parse::<Fraction>(&percent_string(f)).unwrap();
            prop_assert_eq!(back.get().to_bits(), f.get().to_bits());
            let mut out = Vec::new();
            f.serialize_form(&mut serde_json::Serializer::new(&mut out)).unwrap();
            let back = json(std::str::from_utf8(&out).unwrap()).unwrap();
            prop_assert_eq!(back.to_bits(), f.get().to_bits());
        }
    }
}
