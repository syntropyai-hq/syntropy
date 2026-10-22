// SPDX-License-Identifier: MIT OR Apache-2.0

//! Session and sequence identifiers (RFC-0001 §5.9).
//!
//! [`SessionId`] names a conversation or agent context whose KV state may persist and be
//! resumed. Its string form, `ses_` plus 26 characters of lowercase Crockford base32, is what
//! clients see (RFC-0050 §9.3). [`SequenceId`] names one generation stream and is unique across
//! the cluster without coordination.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::entropy::Entropy;
use crate::ids::NodeId;

/// A conversation or agent context whose KV state may persist and be resumed: 128 random bits
/// minted by the ingress node (RFC-0001 §5.9). Owner: RFC-0022.
///
/// Clients see it as an opaque string, `ses_` followed by the 16 bytes in lowercase Crockford
/// base32 without padding (RFC-0050 §9.3), through [`fmt::Display`] and [`FromStr`]. Every value
/// has exactly one string form, and the strings sort in the same order as the bytes. In
/// records and on the wire it is the 16 bytes, like every other ID in this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub [u8; 16]);

/// One generation stream inside a session, or standalone (RFC-0001 §5.9). Owner: RFC-0032.
///
/// Unique cluster-wide without coordination: the ingress node's short ID, its boot counter and
/// a per-boot counter. The choices of an `n > 1` request are separate sequences with distinct
/// IDs, and a sequence keeps its ID across migration and recovery. With a generated seed it is
/// the second component of the sampler key (ADR-0107).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SequenceId {
    /// The ingress node's short ID ([`NodeId::short`]).
    pub ingress: u64,
    /// The ingress node's boot counter.
    pub boot: u32,
    /// The sequence's number within that boot.
    pub n: u32,
}

impl SequenceId {
    /// The `n`th sequence that `ingress` starts in boot `boot`.
    #[must_use]
    pub fn new(ingress: &NodeId, boot: u32, n: u32) -> Self {
        Self {
            ingress: ingress.short(),
            boot,
            n,
        }
    }
}

/// The prefix of [`SessionId`]'s string form.
const PREFIX: &str = "ses_";

/// Lowercase Crockford base32: the digits, then the letters without `i`, `l`, `o` and `u`. The
/// alphabet is in ASCII order, so encoded strings sort as their bytes do.
const ALPHABET: &[u8; 32] = b"0123456789abcdefghjkmnpqrstvwxyz";

/// Characters in the base32 part: 128 bits in 5-bit groups, the last group padded with two
/// zero bits.
const DIGITS: usize = 26;

impl SessionId {
    /// A new session ID: 16 bytes from `entropy`.
    #[must_use]
    pub fn mint(entropy: &impl Entropy) -> Self {
        let mut bytes = [0u8; 16];
        entropy.fill(&mut bytes);
        Self(bytes)
    }
}

/// `ses_` and 26 lowercase Crockford base32 characters, most significant bits first.
impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = u128::from_be_bytes(self.0);
        let mut out = [0u8; DIGITS];
        for (i, slot) in out.iter_mut().enumerate() {
            // Digits 0–24 hold five bits each, from the top; digit 25 holds the low three bits
            // and two zero padding bits.
            let group = if i < DIGITS - 1 {
                (value >> (123 - 5 * i)) & 0x1f
            } else {
                (value & 0b111) << 2
            };
            *slot = ALPHABET[group as usize];
        }
        f.write_str(PREFIX)?;
        // Every byte comes from `ALPHABET`, which is ASCII.
        f.write_str(std::str::from_utf8(&out).map_err(|_| fmt::Error)?)
    }
}

/// Why a string is not a [`SessionId`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseSessionIdError {
    /// The string does not start with `ses_`.
    MissingPrefix,
    /// The part after `ses_` is not 26 characters long.
    Length {
        /// Its length in bytes.
        found: usize,
    },
    /// A character is not in the lowercase Crockford alphabet. Upper case and Crockford's
    /// aliases (`i`, `l`, `o`) are refused, so each ID has one string form.
    Character {
        /// Its byte offset in the whole string.
        at: usize,
    },
    /// The last character sets one of the two padding bits, which no 16-byte value does.
    Padding,
}

impl fmt::Display for ParseSessionIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPrefix => write!(f, "a session ID starts with `{PREFIX}`"),
            Self::Length { found } => write!(
                f,
                "a session ID has {DIGITS} characters after `{PREFIX}`, not {found}"
            ),
            Self::Character { at } => write!(
                f,
                "character {at} of the session ID is not lowercase Crockford base32"
            ),
            Self::Padding => write!(f, "the session ID's last character sets padding bits"),
        }
    }
}

impl std::error::Error for ParseSessionIdError {}

impl FromStr for SessionId {
    type Err = ParseSessionIdError;

    /// Parses the form [`fmt::Display`] writes, and only that form.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let body = s
            .strip_prefix(PREFIX)
            .ok_or(ParseSessionIdError::MissingPrefix)?;
        if body.len() != DIGITS {
            return Err(ParseSessionIdError::Length { found: body.len() });
        }
        let mut value: u128 = 0;
        for (i, byte) in body.bytes().enumerate() {
            let digit =
                ALPHABET
                    .iter()
                    .position(|&c| c == byte)
                    .ok_or(ParseSessionIdError::Character {
                        at: PREFIX.len() + i,
                    })?;
            if i == DIGITS - 1 {
                // The last digit carries the value's three low bits and two padding bits.
                if digit & 0b11 != 0 {
                    return Err(ParseSessionIdError::Padding);
                }
                value = (value << 3) | (digit >> 2) as u128;
            } else {
                value = (value << 5) | digit as u128;
            }
        }
        Ok(Self(value.to_be_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;

    use super::*;

    fn session(bytes: [u8; 16]) -> SessionId {
        SessionId(bytes)
    }

    #[test]
    fn known_strings() {
        assert_eq!(
            session([0; 16]).to_string(),
            "ses_00000000000000000000000000"
        );
        assert_eq!(
            session([0xff; 16]).to_string(),
            "ses_zzzzzzzzzzzzzzzzzzzzzzzzzw"
        );
        // 1 in the lowest bit: the last digit is 1 << 2.
        let mut one = [0u8; 16];
        one[15] = 1;
        assert_eq!(session(one).to_string(), "ses_00000000000000000000000004");
        // The top five bits set: the first digit is 31.
        let mut top = [0u8; 16];
        top[0] = 0xf8;
        assert_eq!(session(top).to_string(), "ses_z0000000000000000000000000");
        // RFC 4648's test vector "foobar" padded to 16 bytes, in Crockford's alphabet.
        let mut foobar = [0u8; 16];
        foobar[..6].copy_from_slice(b"foobar");
        assert_eq!(
            session(foobar).to_string(),
            "ses_csqpyrk1e80000000000000000"
        );
        for id in [[0; 16], [0xff; 16], one, top, foobar] {
            assert_eq!(id, session(id).to_string().parse::<SessionId>().unwrap().0);
        }
    }

    #[test]
    fn bad_strings_are_refused() {
        let good = session([0x5a; 16]).to_string();
        assert_eq!(
            "".parse::<SessionId>(),
            Err(ParseSessionIdError::MissingPrefix)
        );
        assert_eq!(
            good.replacen("ses_", "SES_", 1).parse::<SessionId>(),
            Err(ParseSessionIdError::MissingPrefix)
        );
        assert_eq!(
            good[4..].parse::<SessionId>(),
            Err(ParseSessionIdError::MissingPrefix)
        );
        assert_eq!(
            "ses_".parse::<SessionId>(),
            Err(ParseSessionIdError::Length { found: 0 })
        );
        assert_eq!(
            good[..good.len() - 1].parse::<SessionId>(),
            Err(ParseSessionIdError::Length { found: 25 })
        );
        assert_eq!(
            format!("{good}0").parse::<SessionId>(),
            Err(ParseSessionIdError::Length { found: 27 })
        );
        // Upper case, Crockford's aliases, `u`, padding and non-ASCII are refused.
        for (bad, at) in [
            ('A', 4),
            ('i', 5),
            ('l', 10),
            ('o', 17),
            ('u', 28),
            ('=', 4),
        ] {
            let mut s = good.clone().into_bytes();
            s[at] = bad as u8;
            let s = String::from_utf8(s).unwrap();
            assert_eq!(
                s.parse::<SessionId>(),
                Err(ParseSessionIdError::Character { at }),
                "{s}"
            );
        }
        assert_eq!(
            good.to_uppercase()
                .replacen("SES_", "ses_", 1)
                .parse::<SessionId>(),
            Err(ParseSessionIdError::Character { at: 4 })
        );
        // A multi-byte character makes the byte length wrong or lands on a bad byte.
        let accented = format!("ses_é{}", &good[6..]);
        assert_eq!(accented.len(), good.len());
        assert_eq!(
            accented.parse::<SessionId>(),
            Err(ParseSessionIdError::Character { at: 4 })
        );
    }

    #[test]
    fn padding_bits_must_be_zero() {
        let zero = "ses_00000000000000000000000000";
        for last in ['1', '2', '3', '5', 'z'] {
            let s = format!("{}{last}", &zero[..zero.len() - 1]);
            assert_eq!(
                s.parse::<SessionId>(),
                Err(ParseSessionIdError::Padding),
                "{s}"
            );
        }
        assert!(
            format!("{}4", &zero[..zero.len() - 1])
                .parse::<SessionId>()
                .is_ok()
        );
    }

    #[test]
    fn errors_display_without_the_input() {
        let e = ParseSessionIdError::Length { found: 3 };
        assert_eq!(
            e.to_string(),
            "a session ID has 26 characters after `ses_`, not 3"
        );
        let boxed: Box<dyn std::error::Error> = Box::new(ParseSessionIdError::Padding);
        assert!(boxed.source().is_none());
    }

    #[test]
    fn mint_takes_sixteen_bytes() {
        struct Fixed;
        impl Entropy for Fixed {
            fn fill(&self, buf: &mut [u8]) {
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = i as u8 + 1;
                }
            }
        }
        let id = SessionId::mint(&Fixed);
        assert_eq!(id.0, core::array::from_fn(|i| i as u8 + 1));
    }

    #[test]
    fn sequence_id_takes_the_short_node_id() {
        let mut node = [0u8; 32];
        node[..8].copy_from_slice(&0x0102_0304_0506_0708u64.to_be_bytes());
        node[8] = 0xff;
        let seq = SequenceId::new(&NodeId(node), 7, 9);
        assert_eq!(
            seq,
            SequenceId {
                ingress: 0x0102_0304_0506_0708,
                boot: 7,
                n: 9,
            }
        );
    }

    #[test]
    fn serde_encodings() {
        let seq = SequenceId {
            ingress: 1,
            boot: 2,
            n: 3,
        };
        assert_eq!(
            serde_json::to_value(seq).unwrap(),
            serde_json::json!({ "ingress": 1, "boot": 2, "n": 3 })
        );
        let mut bytes = [0u8; 16];
        bytes[0] = 9;
        assert_eq!(
            serde_json::to_value(SessionId(bytes)).unwrap(),
            serde_json::json!([9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        );
    }

    fn sequence_id() -> impl Strategy<Value = SequenceId> {
        (any::<u64>(), any::<u32>(), any::<u32>()).prop_map(|(ingress, boot, n)| SequenceId {
            ingress,
            boot,
            n,
        })
    }

    proptest! {
        #[test]
        fn sequence_id_round_trips(seq in sequence_id()) {
            let json = serde_json::to_string(&seq).unwrap();
            prop_assert_eq!(serde_json::from_str::<SequenceId>(&json).unwrap(), seq);
        }

        #[test]
        fn session_id_round_trips_through_serde(bytes in any::<[u8; 16]>()) {
            let id = SessionId(bytes);
            let json = serde_json::to_string(&id).unwrap();
            prop_assert_eq!(serde_json::from_str::<SessionId>(&json).unwrap(), id);
        }

        /// Distinct (`ingress`, `boot`, `n`) triples give distinct IDs, and distinct encodings.
        #[test]
        fn distinct_triples_give_distinct_sequence_ids(
            a in sequence_id(),
            b in sequence_id(),
            node in any::<[u8; 32]>(),
            boot in any::<u32>(),
            n in any::<u32>(),
        ) {
            let triple = |s: &SequenceId| (s.ingress, s.boot, s.n);
            prop_assert_eq!(triple(&a) == triple(&b), a == b);
            prop_assert_eq!(
                triple(&a) == triple(&b),
                serde_json::to_string(&a).unwrap() == serde_json::to_string(&b).unwrap()
            );
            // Neighbours in each component differ from the original.
            let node = NodeId(node);
            let s = SequenceId::new(&node, boot, n);
            let mut seen = BTreeSet::from([s]);
            for t in [
                SequenceId { ingress: s.ingress ^ 1, ..s },
                SequenceId { boot: boot.wrapping_add(1), ..s },
                SequenceId { n: n.wrapping_add(1), ..s },
                SequenceId { boot: n, n: boot, ..s },
            ] {
                if (t.boot, t.n) == (s.boot, s.n) && t.ingress == s.ingress {
                    continue; // `boot == n`: the swap is the same triple.
                }
                prop_assert!(seen.insert(t), "{:?} collides", t);
            }
        }

        /// The string form round-trips, has one fixed length, and sorts as the bytes do.
        #[test]
        fn session_id_string_round_trips(a in any::<[u8; 16]>(), b in any::<[u8; 16]>()) {
            let (a, b) = (SessionId(a), SessionId(b));
            let (sa, sb) = (a.to_string(), b.to_string());
            prop_assert_eq!(sa.len(), 30);
            prop_assert!(sa.bytes().skip(4).all(|c| ALPHABET.contains(&c)));
            prop_assert_eq!(sa.parse::<SessionId>(), Ok(a));
            prop_assert_eq!(sa.cmp(&sb), a.cmp(&b));
        }

        /// Any string of the right shape either parses back to itself or is refused for
        /// padding: the string form is one-to-one.
        #[test]
        fn every_canonical_string_parses_to_itself(digits in prop::collection::vec(0usize..32, 26)) {
            let body: String = digits.iter().map(|&d| ALPHABET[d] as char).collect();
            let s = format!("ses_{body}");
            match s.parse::<SessionId>() {
                Ok(id) => prop_assert_eq!(id.to_string(), s),
                Err(e) => {
                    prop_assert_eq!(e, ParseSessionIdError::Padding);
                    prop_assert!(digits[25] & 0b11 != 0);
                }
            }
        }
    }
}
