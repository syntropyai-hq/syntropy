// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size digests, written in JSON as lowercase hex.
//!
//! Lowercase is the only accepted form, so one digest has one spelling and a key compares
//! equal exactly when its text does.

use std::fmt;
use std::str::FromStr;

use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize, Serializer};

/// A string that is not `2 × N` lowercase hex digits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HexError {
    expected_digits: usize,
}

impl fmt::Display for HexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "expected {} lowercase hex digits", self.expected_digits)
    }
}

impl std::error::Error for HexError {}

fn encode(bytes: &[u8], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    bytes.iter().try_for_each(|b| write!(f, "{b:02x}"))
}

fn nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

/// Decodes exactly `2 × N` lowercase hex digits.
fn decode<const N: usize>(s: &str) -> Result<[u8; N], HexError> {
    let err = HexError {
        expected_digits: 2 * N,
    };
    let digits = s.as_bytes();
    if digits.len() != 2 * N {
        return Err(err);
    }
    let mut out = [0u8; N];
    let (pairs, _) = digits.as_chunks::<2>();
    for (byte, &[hi, lo]) in out.iter_mut().zip(pairs) {
        let (Some(hi), Some(lo)) = (nibble(hi), nibble(lo)) else {
            return Err(err);
        };
        *byte = (hi << 4) | lo;
    }
    Ok(out)
}

macro_rules! hex_digest {
    ($(#[$doc:meta])* $name:ident, $len:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub [u8; $len]);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                encode(&self.0, f)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self)
            }
        }

        impl FromStr for $name {
            type Err = HexError;

            fn from_str(s: &str) -> Result<Self, HexError> {
                decode(s).map(Self)
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.collect_str(self)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct HexVisitor;

                impl Visitor<'_> for HexVisitor {
                    type Value = $name;

                    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        write!(f, "a string of {} lowercase hex digits", 2 * $len)
                    }

                    fn visit_str<E: de::Error>(self, s: &str) -> Result<$name, E> {
                        s.parse().map_err(E::custom)
                    }
                }

                deserializer.deserialize_str(HexVisitor)
            }
        }
    };
}

hex_digest!(
    /// A BLAKE3 digest (32 bytes), written as 64 lowercase hex digits.
    Blake3,
    32
);

hex_digest!(
    /// A git commit ID (20 bytes), written as 40 lowercase hex digits.
    GitSha,
    20
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_and_rejects_other_spellings() {
        let sha = GitSha(
            *b"\x00\x01\x7f\x80\xff\x10\x20\x30\x40\x50\x60\x70\x80\x90\xa0\xb0\xc0\xd0\xe0\xf0",
        );
        let text = sha.to_string();
        assert_eq!(text, "00017f80ff102030405060708090a0b0c0d0e0f0");
        assert_eq!(text.parse::<GitSha>(), Ok(sha));
        assert!(text.to_uppercase().parse::<GitSha>().is_err());
        assert!(text[..38].parse::<GitSha>().is_err());
        assert!(format!("{text}00").parse::<GitSha>().is_err());
        assert!("zz".repeat(20).parse::<GitSha>().is_err());
    }
}
