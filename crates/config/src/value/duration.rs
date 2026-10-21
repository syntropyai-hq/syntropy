// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Duration`: a time span written with units (RFC-0054 §3).

use std::fmt;
use std::str::FromStr;

use syntropy_core::Nanos;

use super::{ValueError, serde_via_str};

const NS: u64 = 1;
const US: u64 = 1_000;
const MS: u64 = 1_000_000;
const S: u64 = 1_000_000_000;
const M: u64 = 60 * S;
const H: u64 = 60 * M;
const D: u64 = 24 * H;

/// Units, largest first. Input must list them in this order, each at most once.
const UNITS: &[(&str, u64)] = &[
    ("d", D),
    ("h", H),
    ("m", M),
    ("s", S),
    ("ms", MS),
    ("us", US),
    ("ns", NS),
];

/// A time span written with units: `"250ms"`, `"10s"`, `"5m"`, `"2h"`, `"7d"`, or compound,
/// largest unit first, `"1m30s"` (RFC-0054 §3). A bare number is never a duration.
///
/// - Units are `d`, `h`, `m`, `s`, `ms`, `us` and `ns`. RFC-0054 §3 lists `ms` to `d`; `us`
///   and `ns` exist so that every [`Nanos`] value prints in a form that parses back.
/// - Each component is a whole number. `"1.5s"` is rejected; write `"1s500ms"`.
/// - Output is compound with zero components left out (`"1m30s"`), and `"0s"` for zero.
/// - `"never"`, where a key allows it, is an `Option<Duration>` of `None`: see [`never`](mod@never).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration(pub Nanos);

impl Duration {
    /// Zero.
    pub const ZERO: Duration = Duration(Nanos(0));

    /// `n` nanoseconds.
    pub const fn from_nanos(n: u64) -> Self {
        Self(Nanos(n))
    }

    /// `n` milliseconds.
    ///
    /// # Panics
    ///
    /// If the span exceeds 2^64 - 1 nanoseconds. In a `const` item that is a compile error.
    pub const fn from_millis(n: u64) -> Self {
        match n.checked_mul(MS) {
            Some(nanos) => Self(Nanos(nanos)),
            None => panic!("the duration is longer than 2^64 - 1 nanoseconds"),
        }
    }

    /// `n` seconds.
    ///
    /// # Panics
    ///
    /// If the span exceeds 2^64 - 1 nanoseconds. In a `const` item that is a compile error.
    pub const fn from_secs(n: u64) -> Self {
        match n.checked_mul(S) {
            Some(nanos) => Self(Nanos(nanos)),
            None => panic!("the duration is longer than 2^64 - 1 nanoseconds"),
        }
    }

    /// The span as the core unit type.
    pub const fn nanos(self) -> Nanos {
        self.0
    }
}

impl From<Nanos> for Duration {
    fn from(nanos: Nanos) -> Self {
        Self(nanos)
    }
}

impl From<Duration> for Nanos {
    fn from(d: Duration) -> Self {
        d.0
    }
}

impl From<Duration> for std::time::Duration {
    fn from(d: Duration) -> Self {
        std::time::Duration::from_nanos(d.0.0)
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rest = self.0.0;
        if rest == 0 {
            return f.write_str("0s");
        }
        for &(unit, scale) in UNITS {
            let count = rest / scale;
            if count > 0 {
                write!(f, "{count}{unit}")?;
                rest %= scale;
            }
        }
        Ok(())
    }
}

impl FromStr for Duration {
    type Err = ValueError;

    fn from_str(s: &str) -> Result<Self, ValueError> {
        let bad = |why: String| {
            ValueError::bad_unit(format!(
                "`{s}` is not a duration: {why}; write whole numbers with units, largest first, \
                 such as \"250ms\", \"10s\" or \"1m30s\""
            ))
        };
        if s.is_empty() {
            return Err(bad("it is empty".to_owned()));
        }
        let mut rest = s;
        let mut total: u64 = 0;
        // Index into UNITS of the smallest unit seen so far; the next must be smaller.
        let mut next_allowed = 0;
        while !rest.is_empty() {
            let digits_end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            if digits_end == 0 {
                return Err(bad(format!("expected a number at `{rest}`")));
            }
            let (digits, after) = rest.split_at(digits_end);
            let unit_end = after
                .find(|c: char| !c.is_ascii_alphabetic())
                .unwrap_or(after.len());
            let (unit, after) = after.split_at(unit_end);
            if unit.is_empty() {
                return Err(bad(format!("`{digits}` has no unit")));
            }
            let Some(index) = UNITS.iter().position(|(name, _)| *name == unit) else {
                return Err(bad(format!("`{unit}` is not a unit")));
            };
            if index < next_allowed {
                return Err(bad(format!("`{unit}` is out of order or repeated")));
            }
            next_allowed = index + 1;
            let overflow =
                || ValueError::out_of_range(format!("`{s}` is longer than 2^64 - 1 nanoseconds"));
            let count: u64 = digits.parse().map_err(|_| overflow())?;
            let part = count.checked_mul(UNITS[index].1).ok_or_else(overflow)?;
            total = total.checked_add(part).ok_or_else(overflow)?;
            rest = after;
        }
        Ok(Self(Nanos(total)))
    }
}

serde_via_str!(
    Duration,
    "a duration such as \"250ms\", \"10s\" or \"1m30s\""
);

/// Serde helpers for an `Option<Duration>` key that allows `"never"`, which is `None`
/// (RFC-0054 §3). Use as `#[serde(with = "syntropy_config::never", default)]`.
pub mod never {
    use serde::{Deserialize, Deserializer, Serializer};

    use super::Duration;

    /// The word that means no duration.
    pub const NEVER: &str = "never";

    /// Prints `None` as `"never"` and a duration in its usual form.
    pub fn serialize<S: Serializer>(
        value: &Option<Duration>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            None => serializer.serialize_str(NEVER),
            Some(d) => serializer.collect_str(d),
        }
    }

    /// Reads `"never"` as `None` and anything else as a duration.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Duration>, D::Error> {
        let s = String::deserialize(deserializer)?;
        parse(&s).map_err(serde::de::Error::custom)
    }

    /// Parses `"never"` as `None` and anything else as a duration.
    pub fn parse(s: &str) -> Result<Option<Duration>, super::ValueError> {
        if s == NEVER {
            Ok(None)
        } else {
            s.parse().map(Some)
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn parses_the_rfc_forms() {
        assert_eq!("250ms".parse(), Ok(Duration::from_millis(250)));
        assert_eq!("10s".parse(), Ok(Duration::from_secs(10)));
        assert_eq!("5m".parse(), Ok(Duration::from_secs(300)));
        assert_eq!("2h".parse(), Ok(Duration::from_secs(7200)));
        assert_eq!("7d".parse(), Ok(Duration::from_secs(7 * 86_400)));
        assert_eq!("1m30s".parse(), Ok(Duration::from_secs(90)));
        assert_eq!("1s500ms".parse(), Ok(Duration::from_millis(1500)));
    }

    #[test]
    fn rejects_bad_forms() {
        for s in [
            "", "10", "s", "1.5s", "30s1m", "1s1s", "10 s", "10sec", "-1s", "1m30",
        ] {
            assert!(s.parse::<Duration>().is_err(), "{s:?} should not parse");
        }
        assert!("214749d".parse::<Duration>().is_err());
        assert!(serde_json::from_str::<Duration>("10").is_err());
    }

    #[test]
    fn prints_compound_forms() {
        assert_eq!(Duration::from_millis(250).to_string(), "250ms");
        assert_eq!(Duration::from_secs(90).to_string(), "1m30s");
        assert_eq!(Duration::ZERO.to_string(), "0s");
        assert_eq!(Duration::from_nanos(86_400 * S + 1).to_string(), "1d1ns");
    }

    #[test]
    fn constructors_panic_rather_than_wrap() {
        assert_eq!(
            Duration::from_secs(u64::MAX / S).nanos(),
            Nanos(u64::MAX / S * S)
        );
        assert!(std::panic::catch_unwind(|| Duration::from_secs(u64::MAX / S + 1)).is_err());
        assert!(std::panic::catch_unwind(|| Duration::from_millis(u64::MAX)).is_err());
    }

    #[test]
    fn never_is_none() {
        assert_eq!(never::parse("never"), Ok(None));
        assert_eq!(never::parse("2m"), Ok(Some(Duration::from_secs(120))));
        assert!(never::parse("Never").is_err());
    }

    proptest! {
        #[test]
        fn printed_form_parses_back(n in any::<u64>()) {
            let d = Duration::from_nanos(n);
            prop_assert_eq!(d.to_string().parse::<Duration>(), Ok(d));
            let json = serde_json::to_string(&d).unwrap();
            prop_assert_eq!(serde_json::from_str::<Duration>(&json).unwrap(), d);
        }
    }
}
