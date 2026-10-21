// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ByteSize`: a size written with a unit (RFC-0054 §3).

use std::fmt;
use std::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use syntropy_core::Bytes;

use super::{ValueError, split_decimal, split_number};

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
const TIB: u64 = 1 << 40;

/// Units accepted on input. IEC and SI are kept distinct: `1GiB` is 2³⁰ bytes, `1GB` is 10⁹.
/// SI kilo is accepted as `KB` and `kB`.
const UNITS: &[(&str, u64)] = &[
    ("B", 1),
    ("KiB", KIB),
    ("MiB", MIB),
    ("GiB", GIB),
    ("TiB", TIB),
    ("KB", 1_000),
    ("kB", 1_000),
    ("MB", 1_000_000),
    ("GB", 1_000_000_000),
    ("TB", 1_000_000_000_000),
];

/// Units used on output, largest first. Output is always IEC.
const IEC_OUT: &[(&str, u64)] = &[("TiB", TIB), ("GiB", GIB), ("MiB", MIB), ("KiB", KIB)];

/// A size in bytes, written with a unit: `"16GiB"`, `"512MiB"`, `"1.5TiB"`, `"20GB"`, or a
/// TOML integer meaning bytes (RFC-0054 §3, ADR-0500 decision 6).
///
/// - IEC units (`KiB`…`TiB`) and SI units (`KB`…`TB`) are both accepted, and `B` for bytes.
/// - A fractional amount (`"1.5TiB"`) rounds to the nearest byte, halves away from zero.
/// - Output is the largest IEC unit that divides the size exactly, else bytes: 20 GB prints as
///   `19531250KiB`, and 1000 bytes as `1000B`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteSize(pub Bytes);

impl ByteSize {
    /// `n` bytes.
    pub const fn b(n: u64) -> Self {
        Self(Bytes(n))
    }

    /// `n` KiB.
    pub const fn kib(n: u64) -> Self {
        Self(Bytes(n * KIB))
    }

    /// `n` MiB.
    pub const fn mib(n: u64) -> Self {
        Self(Bytes(n * MIB))
    }

    /// `n` GiB.
    pub const fn gib(n: u64) -> Self {
        Self(Bytes(n * GIB))
    }

    /// `n` TiB.
    pub const fn tib(n: u64) -> Self {
        Self(Bytes(n * TIB))
    }

    /// The size as the core unit type.
    pub const fn bytes(self) -> Bytes {
        self.0
    }
}

impl From<Bytes> for ByteSize {
    fn from(bytes: Bytes) -> Self {
        Self(bytes)
    }
}

impl From<ByteSize> for Bytes {
    fn from(size: ByteSize) -> Self {
        size.0
    }
}

impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0.0;
        if n != 0 {
            for &(unit, scale) in IEC_OUT {
                if n.is_multiple_of(scale) {
                    return write!(f, "{}{unit}", n / scale);
                }
            }
        }
        write!(f, "{n}B")
    }
}

impl FromStr for ByteSize {
    type Err = ValueError;

    fn from_str(s: &str) -> Result<Self, ValueError> {
        parse_scaled(s, UNITS).map(|n| Self(Bytes(n)))
    }
}

/// Parses `<number><unit>` against `units`, rounding a fraction to the nearest whole count.
pub(super) fn parse_scaled(s: &str, units: &[(&str, u64)]) -> Result<u64, ValueError> {
    let bad = |why: &str| {
        ValueError::bad_unit(format!(
            "`{s}` is not a size: {why}; write a number and a unit, such as \"16GiB\", \
             \"512MiB\" or \"20GB\""
        ))
    };
    let (number, unit) = split_number(s.trim());
    let Some((int, frac)) = split_decimal(number) else {
        return Err(bad("the number is missing or malformed"));
    };
    if unit.is_empty() {
        return Err(bad("the unit is missing"));
    }
    let Some(&(_, scale)) = units.iter().find(|(name, _)| *name == unit) else {
        return Err(bad(&format!("`{unit}` is not a unit")));
    };
    // digits × scale / 10^frac, in u128 so nothing in range overflows, then rounded.
    let too_big = || ValueError::out_of_range(format!("`{s}` is larger than 2^64 - 1 bytes"));
    if int.len() + frac.len() > 30 {
        return Err(too_big());
    }
    let digits: u128 = format!("{int}{frac}")
        .parse()
        .map_err(|_| bad("malformed"))?;
    let divisor = 10u128.pow(frac.len() as u32);
    let scaled = digits * u128::from(scale);
    let rounded = (scaled + divisor / 2) / divisor;
    u64::try_from(rounded).map_err(|_| too_big())
}

impl Serialize for ByteSize {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ByteSize {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ByteSizeVisitor)
    }
}

pub(super) struct ByteSizeVisitor;

impl Visitor<'_> for ByteSizeVisitor {
    type Value = ByteSize;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a size such as \"16GiB\", or an integer number of bytes")
    }

    fn visit_u64<E: de::Error>(self, n: u64) -> Result<ByteSize, E> {
        Ok(ByteSize::b(n))
    }

    fn visit_i64<E: de::Error>(self, n: i64) -> Result<ByteSize, E> {
        u64::try_from(n)
            .map(ByteSize::b)
            .map_err(|_| E::custom(format!("a size cannot be negative: {n}")))
    }

    fn visit_str<E: de::Error>(self, s: &str) -> Result<ByteSize, E> {
        s.parse().map_err(E::custom)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn parses_the_rfc_forms() {
        assert_eq!("16GiB".parse(), Ok(ByteSize::gib(16)));
        assert_eq!("512MiB".parse(), Ok(ByteSize::mib(512)));
        assert_eq!("1.5TiB".parse(), Ok(ByteSize::gib(1536)));
        assert_eq!("20GB".parse(), Ok(ByteSize::b(20_000_000_000)));
        assert_eq!("8kB".parse(), Ok(ByteSize::b(8_000)));
        assert_eq!("1000B".parse(), Ok(ByteSize::b(1000)));
        assert_eq!("16 GiB".parse(), Ok(ByteSize::gib(16)));
        assert_eq!(
            serde_json::from_str::<ByteSize>("4096").ok(),
            Some(ByteSize::kib(4))
        );
    }

    #[test]
    fn rounds_fractions_to_the_nearest_byte() {
        // 0.1 GiB = 107374182.4 bytes.
        assert_eq!("0.1GiB".parse(), Ok(ByteSize::b(107_374_182)));
        assert_eq!("1.5B".parse(), Ok(ByteSize::b(2)));
    }

    #[test]
    fn rejects_bad_forms() {
        for s in [
            "", "GiB", "16", "16gib", "16XB", "-1GiB", "1..5GiB", "1.GiB", ".5GiB",
        ] {
            assert!(s.parse::<ByteSize>().is_err(), "{s:?} should not parse");
        }
        assert!("17000000TiB".parse::<ByteSize>().is_err());
        assert!(serde_json::from_str::<ByteSize>("-4").is_err());
    }

    #[test]
    fn prints_the_largest_exact_iec_unit() {
        assert_eq!(ByteSize::gib(16).to_string(), "16GiB");
        assert_eq!(ByteSize::gib(1536).to_string(), "1536GiB");
        assert_eq!(ByteSize::b(20_000_000_000).to_string(), "19531250KiB");
        assert_eq!(ByteSize::b(1000).to_string(), "1000B");
        assert_eq!(ByteSize::b(0).to_string(), "0B");
    }

    proptest! {
        #[test]
        fn printed_form_parses_back(n in any::<u64>()) {
            let size = ByteSize::b(n);
            prop_assert_eq!(size.to_string().parse::<ByteSize>(), Ok(size));
            let json = serde_json::to_string(&size).unwrap();
            prop_assert_eq!(serde_json::from_str::<ByteSize>(&json).unwrap(), size);
        }
    }
}
