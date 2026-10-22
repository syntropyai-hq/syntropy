// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Fraction`: a share of something, in [0, 1] (RFC-0054 §3).

use std::fmt;
use std::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{ValueError, serde_error, split_decimal, split_number};

/// A share in [0, 1], written as a number (`0.2`) or a percentage (`"20%"`) (RFC-0054 §3).
///
/// Output is the shortest decimal that reads back as the same `f32`, such as `0.2`. A
/// percentage is converted by moving the decimal point, so `"20%"` and `0.20` are the same
/// value. Negative zero is stored as zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Fraction(f32);

impl Fraction {
    /// Zero.
    pub const ZERO: Fraction = Fraction(0.0);
    /// One.
    pub const ONE: Fraction = Fraction(1.0);

    /// `value`, if it lies in [0, 1].
    pub fn new(value: f32) -> Result<Self, ValueError> {
        if (0.0..=1.0).contains(&value) {
            // `+ 0.0` turns -0.0 into 0.0.
            Ok(Self(value + 0.0))
        } else {
            Err(ValueError::out_of_range(format!(
                "{value} is not a fraction: it must lie between 0 and 1, or 0% and 100%"
            )))
        }
    }

    /// The share as a number in [0, 1].
    pub const fn get(self) -> f32 {
        self.0
    }

    /// The share as a percentage string, such as `20%`, that parses back to the same value.
    pub fn to_percent_string(self) -> String {
        shift_point_right(&self.0.to_string(), 2) + "%"
    }
}

impl fmt::Display for Fraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for Fraction {
    type Err = ValueError;

    fn from_str(s: &str) -> Result<Self, ValueError> {
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
        Self::new(value)
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

impl Serialize for Fraction {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for Fraction {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(FractionVisitor)
    }
}

pub(super) struct FractionVisitor;

impl Visitor<'_> for FractionVisitor {
    type Value = Fraction;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a number between 0 and 1, or a percentage such as \"20%\"")
    }

    fn visit_f64<E: de::Error>(self, x: f64) -> Result<Fraction, E> {
        // The shortest text of `x` read as f32 rounds once, as `from_str` does.
        x.to_string().parse().map_err(serde_error)
    }

    fn visit_f32<E: de::Error>(self, x: f32) -> Result<Fraction, E> {
        Fraction::new(x).map_err(serde_error)
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
        s.parse().map_err(serde_error)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn parses_numbers_and_percentages() {
        assert_eq!("0.2".parse::<Fraction>().map(Fraction::get), Ok(0.2));
        assert_eq!("20%".parse::<Fraction>().map(Fraction::get), Ok(0.2));
        assert_eq!("12.5%".parse::<Fraction>().map(Fraction::get), Ok(0.125));
        assert_eq!("100%".parse::<Fraction>(), Ok(Fraction::ONE));
        assert_eq!("0.5%".parse::<Fraction>().map(Fraction::get), Ok(0.005));
        assert_eq!("1".parse::<Fraction>(), Ok(Fraction::ONE));
        let json = |s: &str| serde_json::from_str::<Fraction>(s).ok().map(Fraction::get);
        assert_eq!(json("0.1"), Some(0.1));
        assert_eq!(json("0"), Some(0.0));
        assert_eq!(json("\"10%\""), Some(0.1));
    }

    #[test]
    fn rejects_values_outside_zero_to_one() {
        for s in [
            "1.1", "101%", "-0.1", "", "%", "20 %%", "0.2x", "nan", "inf",
        ] {
            assert!(s.parse::<Fraction>().is_err(), "{s:?} should not parse");
        }
        assert!(serde_json::from_str::<Fraction>("2").is_err());
        assert!(Fraction::new(f32::NAN).is_err());
        assert_eq!(Fraction::new(-0.0).map(|f| f.get().to_bits()), Ok(0));
    }

    #[test]
    fn prints_the_shortest_decimal() {
        assert_eq!(Fraction::new(0.2).unwrap().to_string(), "0.2");
        assert_eq!(Fraction::new(0.015625).unwrap().to_string(), "0.015625");
        assert_eq!(Fraction::ONE.to_string(), "1");
        assert_eq!(Fraction::new(0.2).unwrap().to_percent_string(), "20%");
        assert_eq!(
            Fraction::new(0.015625).unwrap().to_percent_string(),
            "1.5625%"
        );
        assert_eq!(Fraction::ONE.to_percent_string(), "100%");
        assert_eq!(Fraction::ZERO.to_percent_string(), "0%");
    }

    proptest! {
        #[test]
        fn printed_form_parses_back(x in 0.0f32..=1.0) {
            let f = Fraction::new(x).unwrap();
            let back = f.to_string().parse::<Fraction>().unwrap();
            prop_assert_eq!(back.get().to_bits(), f.get().to_bits());
            let back = f.to_percent_string().parse::<Fraction>().unwrap();
            prop_assert_eq!(back.get().to_bits(), f.get().to_bits());
            let json = serde_json::to_string(&f).unwrap();
            let back = serde_json::from_str::<Fraction>(&json).unwrap();
            prop_assert_eq!(back.get().to_bits(), f.get().to_bits());
        }
    }
}
