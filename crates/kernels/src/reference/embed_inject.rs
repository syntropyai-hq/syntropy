// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OpKind::EmbedInject`'s reference semantics (RFC-0043 §2.4; RFC-0040 §3.1; COMP-E15-T11).
//!
//! `EmbedInject` is a reserved multimodal hook: a prologue node that writes an encoder's output
//! rows over the embedding rows at the positions of the input's placeholder tokens. No v1 family
//! uses it, and `ir::validate` rule 6 refuses a graph that does; no provider implements it. Its
//! semantics are defined now so that the op's meaning is fixed before the IR freezes.
//!
//! The op moves values and computes none, so it is exact. [`embed_inject`] is generic over the
//! element type, so it serves both [`RefMode`](super::RefMode)s, F32 storage for `F32Fixed` and
//! F64 for `F64`, with the same result, and every row it does not write keeps its bits, NaN
//! payloads and signed zeros included. Where the placeholder positions come from (the
//! request's placeholder token IDs, RFC-0022 §7.1) is the multimodal family's to define after
//! 1.0; here they are an argument.

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
    /// A position given twice: two encoder rows for one placeholder, which has no meaning.
    DuplicatePosition {
        /// The position.
        position: u32,
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
            Self::DuplicatePosition { position } => {
                write!(f, "EmbedInject: position {position} is given twice")
            }
        }
    }
}

impl std::error::Error for EmbedInjectError {}

/// `EmbedInject` (module docs): writes row `i` of `encoded` over row `positions[i]` of `rows`,
/// for every `i`, and leaves every other row of `rows` unchanged.
///
/// - `rows`: the embedding rows of one sequence's tokens, row-major, `width` values each;
/// - `encoded`: the encoder's output rows, one per position, `width` values each;
/// - `positions`: the row of each placeholder token, in the encoder's row order. They need not
///   increase, since a family may encode its inputs in another order than they appear, but each
///   names a different row.
///
/// A `width` of zero is refused even with no rows: a row has at least one value.
///
/// # Errors
///
/// [`EmbedInjectError`] when the shapes disagree, a position is out of range or a position is
/// given twice. `rows` is untouched then.
pub fn embed_inject<T: Copy>(
    rows: &mut [T],
    width: usize,
    encoded: &[T],
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
    let mut seen = std::collections::BTreeSet::new();
    for &position in positions {
        if position as usize >= n_rows {
            return Err(EmbedInjectError::PositionOutOfRange {
                position,
                rows: n_rows,
            });
        }
        if !seen.insert(position) {
            return Err(EmbedInjectError::DuplicatePosition { position });
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
                EmbedInjectError::DuplicatePosition { position: 2 },
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
    }

    /// Positions in another order than the rows: each encoder row still goes to its own
    /// position.
    #[test]
    fn positions_may_come_in_any_order() {
        let mut rows = vec![0.0f32; 6];
        embed_inject(&mut rows, 2, &[9.0, 9.5, 8.0, 8.5], &[2, 0]).unwrap();
        assert_eq!(rows, [8.0, 8.5, 0.0, 0.0, 9.0, 9.5]);
    }

    /// The F64 reference mode stores F64; the same move serves it.
    #[test]
    fn f64_rows_move_the_same_way() {
        let mut rows = vec![0.0f64, 1.0, 2.0, 3.0];
        embed_inject(&mut rows, 2, &[7.0f64, 7.5], &[1]).unwrap();
        assert_eq!(rows, [0.0, 1.0, 7.0, 7.5]);
    }

    proptest! {
        /// Against a direct transcription of the definition, with arbitrary bit patterns (NaN
        /// payloads, infinities and signed zeros among them) and positions in any order: row r
        /// is `encoded`'s row i when `positions[i] == r`, and its old bits otherwise.
        #[test]
        fn matches_the_definition(
            width in 1usize..6,
            n_rows in 0usize..12,
            picks in proptest::sample::subsequence((0u32..12).collect::<Vec<_>>(), 0..12),
            shuffle in any::<u64>(),
            bits in proptest::collection::vec(any::<u32>(), 144),
        ) {
            let mut positions: Vec<u32> = picks.into_iter().filter(|&p| (p as usize) < n_rows).collect();
            // A deterministic shuffle, so positions arrive out of order. Its slight bias does not
            // matter: any order is valid input.
            let len = positions.len();
            for i in (1..len).rev() {
                let j = (shuffle.rotate_left(i as u32) % (i as u64 + 1)) as usize;
                positions.swap(i, j);
            }
            let rows: Vec<f32> = (0..n_rows * width).map(|i| f32::from_bits(bits[i % 144])).collect();
            let encoded: Vec<f32> = (0..positions.len() * width)
                .map(|i| f32::from_bits(bits[(i * 7 + 3) % 144]))
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

        /// Any refused input leaves `rows` exactly as it was.
        #[test]
        fn a_refused_input_writes_nothing(
            width in 0usize..4,
            n_values in 0usize..16,
            encoded_len in 0usize..16,
            positions in proptest::collection::vec(0u32..8, 0..6),
        ) {
            let rows: Vec<f32> = (0..n_values).map(|i| i as f32).collect();
            let encoded: Vec<f32> = (0..encoded_len).map(|i| -(i as f32) - 1.0).collect();
            let mut got = rows.clone();
            if embed_inject(&mut got, width, &encoded, &positions).is_err() {
                prop_assert_eq!(got, rows);
            }
        }
    }
}
