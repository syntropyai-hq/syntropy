// SPDX-License-Identifier: MIT OR Apache-2.0

//! Teacher-forced parity taps (RFC-0060 §5.5, amendment A4).
//!
//! The parity driver opens a sequence on a real instance with a [`ParityTapSpec`]
//! (`SequenceOpen::parity`, RFC-0001 §5.8). At the tail, the spec's [`ForceMode`] replaces
//! sampling with the teacher token. The tail reports what it measured in a [`LogitTap`] on each
//! `TokenEntry` (`TokenEntry::tap`). RFC-0012 §8.5 defines what the scheduler and the tail do with
//! these types, RFC-0031 §6.5 and §6.13 their wire layout, RFC-0050 §13.6 the parity endpoint,
//! and RFC-0060 §5.3–§5.4 the arithmetic that consumes a tap.
//!
//! The types live in `core` because the tail in `exec`, the parity endpoint in `api` and the
//! driver all build against one definition, and RFC-0001 §4.2 rule 3 keeps compute crates such as
//! `exec` off `net` (RFC-0060 §5.5, "Crate").
//!
//! **Non-finite values are never clamped** (RFC-0060 §5.4). A tap's floats keep their value
//! through serde. In a human-readable format such as JSON, a non-finite value is written as the
//! string `"nan"`, `"inf"` or `"-inf"` (RFC-0050 §13.6), and a finite one as a number with
//! round-trip precision. In a binary format each float is written as itself, so its IEEE bit
//! pattern survives (RFC-0031 §6.5).

use serde::{Deserialize, Serialize};

/// Names the teacher stream of one parity run (RFC-0060 §5.5).
///
/// The driver's daemon chooses it per parity run. The tail treats it as opaque. The teacher
/// tokens themselves arrive in `TapIds` frames for the sequence, which carry its `SequenceId`
/// (RFC-0031 §6.13), so this ID needs no wire field of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TeacherStreamId(pub u64);

/// A teacher-forced parity sequence: which teacher stream, how the tail forces it and what it
/// taps (RFC-0060 §5.5). Carried as `SequenceOpen::parity` (RFC-0001 §5.8).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParityTapSpec {
    /// The corpus tokens, delivered to the tail and to the head's drafter if there is one.
    pub stream: TeacherStreamId,
    /// How the tail replaces sampling with the teacher tokens.
    pub mode: ForceMode,
    /// What the tail returns per scored position.
    pub tap: TapMode,
}

/// How the tail forces the teacher tokens (RFC-0060 §5.5, RFC-0012 §8.5). In every mode the token
/// the tail sends around the ring is the teacher token, so each stage sees service-shaped traffic.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ForceMode {
    /// Positions `[0, n)` as chunked prefill with the plan's chunk size.
    Prefill,
    /// Positions `[0, p)` as prefill, then `[p, n)` one decode step per position.
    Decode {
        /// `p`: the positions run as prefill before decode starts.
        prefill_len: u32,
    },
    /// As [`ForceMode::Decode`], with verify windows of `gamma + 1` positions. A synthetic drafter
    /// proposes the teacher token with probability `accept` and a different token otherwise, so
    /// windows exercise full acceptance, partial acceptance and KV truncation. `seed` fixes the
    /// pattern.
    Verify {
        /// The positions run as prefill before the first verify window.
        prefill_len: u32,
        /// The draft length γ of each window.
        gamma: u16,
        /// The probability that the synthetic drafter proposes the teacher token.
        accept: f32,
        /// Keys the drafter's draws, separately from the sampling stream (RFC-0012 §8.5).
        seed: u64,
    },
}

/// What the tail returns per scored position (RFC-0060 §5.5).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TapMode {
    /// The tail writes full F32 logits into a shared-memory region the driver maps. Only when the
    /// driver runs on the tail's host.
    FullLocal,
    /// The driver streams the reference top-K ids and tail-sample ids (RFC-0060 §5.3) for upcoming
    /// positions to the tail over a `Tap` channel. The tail returns, per position, the candidate's
    /// logits at those ids, its logsumexp and its logsumexp over V∖T (both computed in F64), its
    /// argmax and its top-2 margin.
    Gather {
        /// Reference top-K ids per position, the set T.
        k: u16,
        /// Tail-sample ids per position, the set S drawn from V∖T.
        tail: u16,
    },
}

/// The tap block of one `TokenEntry`: one [`TapRow`] per tapped position, which is every
/// position of a verify window or prefill chunk (`TokenEntry::tap`, RFC-0001 §5.8; the tap block
/// of RFC-0031 §6.5).
///
/// Every row's [`TapValues`] has the entry's mode: [`TapValues::Gather`] with `k` and `tail`
/// values for [`TapMode::Gather`], or [`TapValues::FullLocal`] with `k = tail = 0`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogitTap {
    /// Reference top-K ids per position (the set T); 0 for `FullLocal`.
    pub k: u16,
    /// Tail-sample ids per position (the set S); 0 for `FullLocal`.
    pub tail: u16,
    /// The tapped positions, in position order.
    pub rows: Vec<TapRow>,
}

/// What the tail measured at one tapped position (RFC-0031 §6.5's per-position record). The
/// parity endpoint streams one `TapRow` per scored position (RFC-0050 §13.6).
///
/// Values are the candidate's raw F32 logits, not log-probabilities. The driver forms
/// `log q_i = y_i − lse` and `log s = lse_complement − lse` in F64 (RFC-0060 §5.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TapRow {
    /// The sequence position.
    pub pos: u32,
    /// The candidate's argmax token id; ties go to the lower id.
    pub argmax: u32,
    /// The candidate's top-1 minus top-2 logit.
    #[serde(with = "tap_float")]
    pub margin: f32,
    /// The candidate's logsumexp over the full vocabulary, computed in F64.
    #[serde(with = "tap_float")]
    pub lse: f64,
    /// The candidate's logsumexp over V∖T, computed in F64 directly, never as `1 − Σ p`. 0 for
    /// `FullLocal`, where the driver computes it from the row.
    #[serde(with = "tap_float")]
    pub lse_complement: f64,
    /// The logits, or where the row is.
    pub values: TapValues,
}

/// A tapped position's logits (RFC-0031 §6.5).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TapValues {
    /// [`TapMode::Gather`]: the candidate's F32 logits at the ids the driver sent, each set in the
    /// order sent (RFC-0031 §6.13).
    Gather {
        /// At the `k` reference top-K ids, the set T.
        #[serde(with = "tap_float::seq")]
        top: Vec<f32>,
        /// At the `tail` tail-sample ids, the set S.
        #[serde(with = "tap_float::seq")]
        tail: Vec<f32>,
    },
    /// [`TapMode::FullLocal`]: the row's offset in the shared-memory region of full F32 logits that
    /// the driver maps. These are the row offsets of RFC-0012 §8.5.
    FullLocal {
        /// Byte offset of the row in the region.
        shm_offset: u64,
    },
}

/// Serde for the tap's floats: in a human-readable format, a non-finite value is the string
/// `"nan"`, `"inf"` or `"-inf"` (RFC-0050 §13.6); everywhere else, the float as itself.
mod tap_float {
    use std::fmt;
    use std::marker::PhantomData;

    use serde::de::{self, DeserializeOwned, Visitor};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// A float type a tap carries.
    pub(super) trait TapFloat: Copy + Serialize + DeserializeOwned {
        fn name(self) -> Option<&'static str>;
        fn from_name(name: &str) -> Option<Self>;
        /// A JSON number, which a self-describing format hands over as an `f64`.
        fn from_f64(value: f64) -> Self;
    }

    macro_rules! tap_float {
        ($t:ty) => {
            impl TapFloat for $t {
                fn name(self) -> Option<&'static str> {
                    if self.is_nan() {
                        Some("nan")
                    } else if self == <$t>::INFINITY {
                        Some("inf")
                    } else if self == <$t>::NEG_INFINITY {
                        Some("-inf")
                    } else {
                        None
                    }
                }
                fn from_name(name: &str) -> Option<Self> {
                    match name {
                        "nan" => Some(<$t>::NAN),
                        "inf" => Some(<$t>::INFINITY),
                        "-inf" => Some(<$t>::NEG_INFINITY),
                        _ => None,
                    }
                }
                #[allow(clippy::cast_possible_truncation)]
                fn from_f64(value: f64) -> Self {
                    value as $t
                }
            }
        };
    }
    tap_float!(f32);
    tap_float!(f64);

    /// One float on the wire.
    struct Wire<F>(F);

    impl<F: TapFloat> Serialize for Wire<F> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            match self.0.name() {
                Some(name) if serializer.is_human_readable() => serializer.serialize_str(name),
                _ => self.0.serialize(serializer),
            }
        }
    }

    impl<'de, F: TapFloat> Deserialize<'de> for Wire<F> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            if deserializer.is_human_readable() {
                deserializer
                    .deserialize_any(NumberOrName(PhantomData))
                    .map(Wire)
            } else {
                F::deserialize(deserializer).map(Wire)
            }
        }
    }

    struct NumberOrName<F>(PhantomData<F>);

    impl<F: TapFloat> Visitor<'_> for NumberOrName<F> {
        type Value = F;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(r#"a number, or "nan", "inf" or "-inf""#)
        }

        fn visit_f64<E: de::Error>(self, value: f64) -> Result<F, E> {
            Ok(F::from_f64(value))
        }

        #[allow(clippy::cast_precision_loss)]
        fn visit_i64<E: de::Error>(self, value: i64) -> Result<F, E> {
            Ok(F::from_f64(value as f64))
        }

        #[allow(clippy::cast_precision_loss)]
        fn visit_u64<E: de::Error>(self, value: u64) -> Result<F, E> {
            Ok(F::from_f64(value as f64))
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<F, E> {
            F::from_name(value).ok_or_else(|| E::invalid_value(de::Unexpected::Str(value), &self))
        }
    }

    pub(super) fn serialize<F: TapFloat, S: Serializer>(
        value: &F,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        Wire(*value).serialize(serializer)
    }

    pub(super) fn deserialize<'de, F: TapFloat, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<F, D::Error> {
        Wire::deserialize(deserializer).map(|wire| wire.0)
    }

    /// The same for a sequence of floats.
    pub(super) mod seq {
        use serde::{Deserialize, Deserializer, Serializer};

        use super::{TapFloat, Wire};

        pub(in super::super) fn serialize<F: TapFloat, S: Serializer>(
            values: &[F],
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(values.iter().map(|value| Wire(*value)))
        }

        pub(in super::super) fn deserialize<'de, F: TapFloat, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Vec<F>, D::Error> {
            let wires = Vec::<Wire<F>>::deserialize(deserializer)?;
            Ok(wires.into_iter().map(|wire| wire.0).collect())
        }
    }
}
