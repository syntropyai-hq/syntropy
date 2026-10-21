// SPDX-License-Identifier: MIT OR Apache-2.0

//! The value types of RFC-0054 §3: units, selectors, regular expressions and secrets.
//!
//! Every type parses the forms §3 accepts, through [`FromStr`](std::str::FromStr) and serde,
//! and prints one canonical form through [`Display`](std::fmt::Display). The printed form parses
//! back to the same value, so any value `config show` prints can be pasted back into a file
//! (§3, last paragraph). Secrets are the exception: they never print.

use std::fmt;

mod bytes;
mod duration;
mod fraction;
mod range;
mod rate;
mod regex;
mod reserve;
mod secret;
mod selector;

pub use self::bytes::ByteSize;
pub use self::duration::{Duration, never};
pub use self::fraction::Fraction;
pub use self::range::InRange;
pub use self::rate::Rate;
pub use self::regex::{
    MAX_PATTERN_LEN as REGEX_MAX_PATTERN_LEN, Regex, SIZE_LIMIT as REGEX_SIZE_LIMIT,
};
pub use self::reserve::Reserve;
pub use self::secret::{REDACTED, Secret, SecretSource};
pub use self::selector::{
    DeviceApi, DeviceSelector, LayerRange, MIN_ID_PREFIX, ModelSelector, NodeSelector, TierSelector,
};

/// Why a value was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValueErrorKind {
    /// The value has the wrong TOML type or shape for the key (RFC-0054 §8 code 2,
    /// `wrong_type`).
    WrongType,
    /// The value lies outside the key's range (code 4, `out_of_range`).
    OutOfRange,
    /// The unit syntax of §3 is wrong (code 5, `bad_unit`).
    BadUnit,
    /// A node, device, tier, layer or model selector does not parse, or is not allowed where
    /// it appears (code 6, `bad_selector`).
    BadSelector,
    /// A regular expression does not compile, or exceeds the size limits (code 7,
    /// `bad_regex`).
    BadRegex,
    /// A secret's file or environment variable cannot be read.
    SecretUnavailable,
}

/// A rejected value: what kind of error, and a message that says what was wrong and what is
/// accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueError {
    kind: ValueErrorKind,
    message: String,
}

impl ValueError {
    /// An error of `kind` with `message`.
    pub fn new(kind: ValueErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub(crate) fn wrong_type(message: impl Into<String>) -> Self {
        Self::new(ValueErrorKind::WrongType, message)
    }

    pub(crate) fn out_of_range(message: impl Into<String>) -> Self {
        Self::new(ValueErrorKind::OutOfRange, message)
    }

    pub(crate) fn bad_unit(message: impl Into<String>) -> Self {
        Self::new(ValueErrorKind::BadUnit, message)
    }

    pub(crate) fn bad_selector(message: impl Into<String>) -> Self {
        Self::new(ValueErrorKind::BadSelector, message)
    }

    pub(crate) fn bad_regex(message: impl Into<String>) -> Self {
        Self::new(ValueErrorKind::BadRegex, message)
    }

    /// What kind of error this is.
    pub fn kind(&self) -> ValueErrorKind {
        self.kind
    }

    /// The message, without the kind.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ValueError {}

/// Implements `Deserialize` for a type through its `FromStr`, accepting only a string, and
/// `Serialize` through its `Display`.
macro_rules! serde_via_str {
    ($ty:ty, $expecting:literal) => {
        impl serde::Serialize for $ty {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.collect_str(self)
            }
        }

        impl<'de> serde::Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct V;
                impl serde::de::Visitor<'_> for V {
                    type Value = $ty;
                    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        f.write_str($expecting)
                    }
                    fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<$ty, E> {
                        s.parse().map_err(E::custom)
                    }
                }
                deserializer.deserialize_str(V)
            }
        }
    };
}
pub(crate) use serde_via_str;

/// Parses a non-negative decimal number with an optional fraction (`12`, `1.5`, `.5` is not
/// accepted), returning the digits without the point and the count of fraction digits.
pub(crate) fn split_decimal(number: &str) -> Option<(&str, &str)> {
    let (int, frac) = match number.split_once('.') {
        Some((int, frac)) => (int, frac),
        None => (number, ""),
    };
    let digits = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    if int.is_empty() || !digits(int) || !digits(frac) || (number.contains('.') && frac.is_empty())
    {
        return None;
    }
    Some((int, frac))
}

/// Splits `s` into its leading number (digits and at most one `.`) and the rest, with
/// whitespace between them dropped.
pub(crate) fn split_number(s: &str) -> (&str, &str) {
    let end = s
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(s.len());
    (&s[..end], s[end..].trim_start())
}
