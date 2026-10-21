// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Rate`: tokens or bytes per second (RFC-0054 §3).

use std::fmt;
use std::str::FromStr;

use syntropy_core::BytesPerSec;

use super::{ValueError, serde_via_str, split_decimal, split_number};

/// Byte units, as for `ByteSize`, followed by `/s`.
const BYTE_UNITS: &[(&str, f64)] = &[
    ("B", 1.0),
    ("KiB", 1024.0),
    ("MiB", 1_048_576.0),
    ("GiB", 1_073_741_824.0),
    ("TiB", 1_099_511_627_776.0),
    ("KB", 1e3),
    ("kB", 1e3),
    ("MB", 1e6),
    ("GB", 1e9),
    ("TB", 1e12),
];

/// Units tried on output, largest first. Byte rates print in SI, as links and drives quote
/// them.
const BYTE_OUT: &[(&str, f64)] = &[("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)];

/// A rate: `"5tok/s"` (tokens per second) or `"2.5GB/s"` (bytes per second) (RFC-0054 §3).
///
/// - Byte rates take the units of `ByteSize`, IEC or SI.
/// - Output for a byte rate is the largest SI unit in which the value prints exactly and is at
///   least 1, else bytes per second.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Rate {
    /// Tokens per second.
    Tokens(f32),
    /// Bytes per second.
    Bytes(BytesPerSec),
}

impl Rate {
    /// Tokens per second, if this is a token rate.
    pub fn tokens_per_sec(self) -> Option<f32> {
        match self {
            Rate::Tokens(t) => Some(t),
            Rate::Bytes(_) => None,
        }
    }

    /// Bytes per second, if this is a byte rate.
    pub fn bytes_per_sec(self) -> Option<BytesPerSec> {
        match self {
            Rate::Bytes(b) => Some(b),
            Rate::Tokens(_) => None,
        }
    }
}

impl fmt::Display for Rate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Rate::Tokens(t) => write!(f, "{t}tok/s"),
            Rate::Bytes(BytesPerSec(x)) => {
                for &(unit, scale) in BYTE_OUT {
                    let amount = x / scale;
                    if amount >= 1.0 {
                        let text = amount.to_string();
                        if scaled(&text, scale) == Some(x) {
                            return write!(f, "{text}{unit}/s");
                        }
                    }
                }
                write!(f, "{x}B/s")
            }
        }
    }
}

/// The decimal `text` times `scale`, as the parser computes it.
fn scaled(text: &str, scale: f64) -> Option<f64> {
    text.parse::<f64>().ok().map(|amount| amount * scale)
}

impl FromStr for Rate {
    type Err = ValueError;

    fn from_str(s: &str) -> Result<Self, ValueError> {
        let bad = |why: &str| {
            ValueError::bad_unit(format!(
                "`{s}` is not a rate: {why}; write a number and a unit per second, such as \
                 \"5tok/s\" or \"2.5GB/s\""
            ))
        };
        let (number, unit) = split_number(s.trim());
        if split_decimal(number).is_none() {
            return Err(bad("the number is missing or malformed"));
        }
        let Some(unit) = unit.strip_suffix("/s") else {
            return Err(bad("the unit must end in `/s`"));
        };
        let finite = |x: f64| {
            if x.is_finite() {
                Ok(x)
            } else {
                Err(ValueError::out_of_range(format!("`{s}` is too large")))
            }
        };
        if unit == "tok" {
            let tokens: f32 = number.parse().map_err(|_| bad("malformed number"))?;
            return finite(f64::from(tokens)).map(|_| Rate::Tokens(tokens));
        }
        let Some(&(_, scale)) = BYTE_UNITS.iter().find(|(name, _)| *name == unit) else {
            return Err(bad(&format!("`{unit}/s` is not a unit")));
        };
        let bytes = scaled(number, scale).ok_or_else(|| bad("malformed number"))?;
        finite(bytes).map(|b| Rate::Bytes(BytesPerSec(b)))
    }
}

serde_via_str!(Rate, "a rate such as \"5tok/s\" or \"2.5GB/s\"");

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn parses_the_rfc_forms() {
        assert_eq!("5tok/s".parse(), Ok(Rate::Tokens(5.0)));
        assert_eq!("2.5GB/s".parse(), Ok(Rate::Bytes(BytesPerSec(2.5e9))));
        assert_eq!(
            "1GiB/s".parse(),
            Ok(Rate::Bytes(BytesPerSec(1_073_741_824.0)))
        );
        assert_eq!("0.5tok/s".parse(), Ok(Rate::Tokens(0.5)));
    }

    #[test]
    fn rejects_bad_forms() {
        for s in [
            "",
            "5",
            "5tok",
            "5tokens/s",
            "GB/s",
            "-1tok/s",
            "5XB/s",
            "1e3tok/s",
        ] {
            assert!(s.parse::<Rate>().is_err(), "{s:?} should not parse");
        }
    }

    #[test]
    fn prints_the_largest_exact_unit() {
        assert_eq!(Rate::Tokens(5.0).to_string(), "5tok/s");
        assert_eq!(Rate::Bytes(BytesPerSec(2.5e9)).to_string(), "2.5GB/s");
        assert_eq!(Rate::Bytes(BytesPerSec(512.0)).to_string(), "512B/s");
        assert_eq!(Rate::Bytes(BytesPerSec(0.0)).to_string(), "0B/s");
    }

    proptest! {
        #[test]
        fn token_rates_parse_back(t in 0.0f32..1e9) {
            let rate = Rate::Tokens(t);
            prop_assert_eq!(rate.to_string().parse::<Rate>(), Ok(rate));
        }

        #[test]
        fn byte_rates_parse_back(x in 0.0f64..1e15) {
            let rate = Rate::Bytes(BytesPerSec(x));
            prop_assert_eq!(rate.to_string().parse::<Rate>(), Ok(rate));
            let json = serde_json::to_string(&rate).unwrap();
            prop_assert_eq!(serde_json::from_str::<Rate>(&json).unwrap(), rate);
        }
    }
}
