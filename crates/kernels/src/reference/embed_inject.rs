// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OpKind::EmbedInject`'s reference semantics (RFC-0043 §2.4; RFC-0040 §3.1; COMP-E15-T11).
//!
//! `EmbedInject` is a reserved multimodal hook: a prologue node that writes an encoder's output
//! rows over the embedding rows at the positions of the input's placeholder tokens. No v1 family
//! uses it, and `ir::validate` rule 6 refuses a graph that does; no provider implements it. Its
//! semantics are defined now so that the op's meaning is fixed before the IR freezes.
//!
//! The op moves values and computes none, so it is exact: the result is the same in every
//! [`RefMode`](super::RefMode), and every row it does not write keeps its bits, NaN payloads and
//! signed zeros included. Where the placeholder positions come from (the request's placeholder
//! token IDs, RFC-0022 §7.1) is the multimodal family's to define after 1.0; here they are an
//! argument.

use std::fmt;

/// Why [`embed_inject`] refused its inputs. It writes nothing when it refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmbedInjectError {
    /// A row width of zero.
    ZeroWidth,
    /// The embedding rows are not a whole number of rows of `width`.
    RaggedRows {
        /// The values given.
        len: usize,
        /// The row width.
        width: usize,
    },
    /// The encoder's output does not hold exactly one row of `width` per position.
    EncodedRows {
        /// The values given.
        len: usize,
        /// The positions given.
        positions: usize,
        /// The row width.
        width: usize,
    },
    /// A position past the last embedding row.
    PositionOutOfRange {
        /// The position.
        position: u32,
        /// The embedding rows.
        rows: usize,
    },
    /// Positions that do not strictly increase: a repeated or out-of-order placeholder.
    PositionsNotIncreasing {
        /// The index in `positions` of the first one not above its predecessor.
        at: usize,
    },
}

impl fmt::Display for EmbedInjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWidth => write!(f, "EmbedInject: the row width is zero"),
            Self::RaggedRows { len, width } => write!(
                f,
                "EmbedInject: {len} embedding values are not whole rows of {width}"
            ),
            Self::EncodedRows {
                len,
                positions,
                width,
            } => write!(
                f,
                "EmbedInject: {len} encoder values are not one row of {width} for each of \
                 {positions} positions"
            ),
            Self::PositionOutOfRange { position, rows } => write!(
                f,
                "EmbedInject: position {position} is past the {rows} embedding rows"
            ),
            Self::PositionsNotIncreasing { at } => write!(
                f,
                "EmbedInject: position {at} does not follow its predecessor in increasing order"
            ),
        }
    }
}

impl std::error::Error for EmbedInjectError {}

/// `EmbedInject` (module docs): writes row `i` of `encoded` over row `positions[i]` of `rows`,
/// for every `i`, and leaves every other row of `rows` unchanged.
///
/// - `rows`: the embedding rows of one sequence's tokens, row-major, `width` values each;
/// - `encoded`: the encoder's output rows, one per position, `width` values each;
/// - `positions`: the row of each placeholder token, strictly increasing.
///
/// # Errors
///
/// [`EmbedInjectError`] when the shapes disagree or a position is out of range or out of order.
/// `rows` is untouched then.
pub fn embed_inject(
    rows: &mut [f32],
    width: usize,
    encoded: &[f32],
    positions: &[u32],
) -> Result<(), EmbedInjectError> {
    if width == 0 {
        return Err(EmbedInjectError::ZeroWidth);
    }
    if !rows.len().is_multiple_of(width) {
        return Err(EmbedInjectError::RaggedRows {
            len: rows.len(),
            width,
        });
    }
    if positions.len().checked_mul(width) != Some(encoded.len()) {
        return Err(EmbedInjectError::EncodedRows {
            len: encoded.len(),
            positions: positions.len(),
            width,
        });
    }
    let n_rows = rows.len() / width;
    for (at, &position) in positions.iter().enumerate() {
        if position as usize >= n_rows {
            return Err(EmbedInjectError::PositionOutOfRange {
                position,
                rows: n_rows,
            });
        }
        if at > 0 && position <= positions[at - 1] {
            return Err(EmbedInjectError::PositionsNotIncreasing { at });
        }
    }
    for (row, &position) in encoded.chunks_exact(width).zip(positions) {
        let start = position as usize * width;
        rows[start..start + width].copy_from_slice(row);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn encoder_rows_replace_the_placeholder_rows_only() {
        // Four rows of width 2; placeholders at rows 1 and 3.
        let mut rows = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        embed_inject(&mut rows, 2, &[10.0, 11.0, 30.0, 31.0], &[1, 3]).unwrap();
        assert_eq!(rows, [0.0, 0.5, 10.0, 11.0, 2.0, 2.5, 30.0, 31.0]);
    }

    /// The op moves values: a NaN payload, a signed zero and an infinity pass bit for bit, and
    /// rows it does not write keep theirs.
    #[test]
    fn values_move_bit_for_bit() {
        let nan = f32::from_bits(0x7fc0_1234);
        let mut rows = vec![-0.0, nan, 1.0, 2.0];
        embed_inject(&mut rows, 2, &[f32::NEG_INFINITY, -0.0], &[1]).unwrap();
        let bits: Vec<u32> = rows.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            bits,
            [
                (-0.0f32).to_bits(),
                0x7fc0_1234,
                f32::NEG_INFINITY.to_bits(),
                (-0.0f32).to_bits()
            ]
        );
    }

    #[test]
    fn no_positions_change_nothing() {
        let mut rows = vec![1.0, 2.0, 3.0];
        embed_inject(&mut rows, 3, &[], &[]).unwrap();
        assert_eq!(rows, [1.0, 2.0, 3.0]);
    }

    /// Each malformed input is refused, and `rows` is left as it was.
    #[test]
    fn malformed_inputs_are_refused_and_write_nothing() {
        let cases: [(usize, &[f32], &[u32], EmbedInjectError); 5] = [
            (0, &[], &[], EmbedInjectError::ZeroWidth),
            (
                4,
                &[],
                &[],
                EmbedInjectError::RaggedRows { len: 6, width: 4 },
            ),
            (
                2,
                &[9.0],
                &[0],
                EmbedInjectError::EncodedRows {
                    len: 1,
                    positions: 1,
                    width: 2,
                },
            ),
            (
                2,
                &[9.0, 9.0],
                &[3],
                EmbedInjectError::PositionOutOfRange {
                    position: 3,
                    rows: 3,
                },
            ),
            (
                2,
                &[9.0, 9.0, 8.0, 8.0],
                &[2, 2],
                EmbedInjectError::PositionsNotIncreasing { at: 1 },
            ),
        ];
        for (width, encoded, positions, expected) in cases {
            let mut rows = vec![1.0f32; 6];
            assert_eq!(
                embed_inject(&mut rows, width, encoded, positions),
                Err(expected)
            );
            assert_eq!(rows, [1.0f32; 6]);
        }
        let mut rows = vec![1.0f32; 6];
        assert!(embed_inject(&mut rows, 2, &[9.0, 9.0, 8.0, 8.0], &[2, 1]).is_err());
        assert_eq!(rows, [1.0f32; 6]);
    }

    proptest! {
        /// Against a direct transcription of the definition: row r is `encoded`'s row i when
        /// `positions[i] == r`, and its old value otherwise.
        #[test]
        fn matches_the_definition(
            width in 1usize..6,
            n_rows in 0usize..12,
            mask in any::<u16>(),
            seed in any::<u32>(),
        ) {
            let rows: Vec<f32> = (0..n_rows * width).map(|i| i as f32 + 0.25).collect();
            let positions: Vec<u32> = (0..n_rows)
                .filter(|r| mask >> r & 1 == 1)
                .map(|r| r as u32)
                .collect();
            let encoded: Vec<f32> = (0..positions.len() * width)
                .map(|i| -(i as f32) - f32::from_bits(seed % 1000 + 1))
                .collect();
            let mut got = rows.clone();
            embed_inject(&mut got, width, &encoded, &positions).unwrap();
            for r in 0..n_rows {
                let expect: &[f32] = match positions.iter().position(|&p| p as usize == r) {
                    Some(i) => &encoded[i * width..(i + 1) * width],
                    None => &rows[r * width..(r + 1) * width],
                };
                let got_bits: Vec<u32> = got[r * width..(r + 1) * width].iter().map(|v| v.to_bits()).collect();
                let expect_bits: Vec<u32> = expect.iter().map(|v| v.to_bits()).collect();
                prop_assert_eq!(got_bits, expect_bits, "row {}", r);
            }
        }
    }
}
