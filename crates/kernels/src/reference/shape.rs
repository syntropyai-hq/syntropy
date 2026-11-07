// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shapes the reference ops take, and why an op refuses its inputs (COMP-E1-T03).
//!
//! **Rows.** Every op that works on token rows takes `rows_active`, the step's active row count
//! (RFC-0040 §3.2, §3.5). Its activation buffers may hold more rows than that, the bucket's
//! padded rows; an op reads and writes only the first `rows_active` rows and leaves the rest
//! untouched. With `rows_active = 0` an op writes nothing, whatever its buffers hold. Rows are
//! row-major and contiguous.
//!
//! **Refusals.** An op checks every shape before it writes, so a refused call leaves its output
//! untouched.

use std::fmt;

use syntropy_core::GgmlType;

use super::dequant::DequantError;

/// Why a reference op refused its inputs. It writes nothing when it refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpError {
    /// A matrix's values are not `rows · cols`.
    MatrixShape {
        /// The values given.
        len: usize,
        /// The rows.
        rows: usize,
        /// The columns.
        cols: usize,
    },
    /// An argument has the wrong number of values for the op.
    Len {
        /// The op.
        op: &'static str,
        /// The argument.
        arg: &'static str,
        /// The values given.
        len: usize,
        /// The values the op needs: at least this many for an activation or output buffer,
        /// exactly this many for a weight.
        need: usize,
    },
    /// Two matrices that must have one shape do not.
    ShapeMismatch {
        /// The op.
        op: &'static str,
    },
    /// The op does not define this variant in this milestone.
    Unsupported {
        /// The op.
        op: &'static str,
        /// The variant, and the task that adds it.
        what: &'static str,
    },
    /// A row of a table type is not a whole number of blocks.
    RowLength {
        /// The table type.
        ty: GgmlType,
        /// The row's elements.
        elems: usize,
        /// The type's block.
        block: usize,
    },
    /// A row ID past the table.
    RowOutOfRange {
        /// The ID.
        id: u32,
        /// The table's rows.
        rows: usize,
    },
    /// The bytes could not be decoded.
    Dequant(DequantError),
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatrixShape { len, rows, cols } => {
                write!(f, "{len} values are not a {rows} × {cols} matrix")
            }
            Self::Len { op, arg, len, need } => {
                write!(f, "{op}: `{arg}` holds {len} values, the op needs {need}")
            }
            Self::ShapeMismatch { op } => write!(f, "{op}: the matrices differ in shape"),
            Self::Unsupported { op, what } => write!(f, "{op}: {what} is not defined yet"),
            Self::RowLength { ty, elems, block } => write!(
                f,
                "a row of {elems} elements is not whole {block}-element {ty:?} blocks"
            ),
            Self::RowOutOfRange { id, rows } => {
                write!(f, "row ID {id} is past the table's {rows} rows")
            }
            Self::Dequant(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for OpError {}

impl From<DequantError> for OpError {
    fn from(e: DequantError) -> Self {
        Self::Dequant(e)
    }
}

/// A row-major F32 matrix: a weight, dequantised exactly (RFC-0042 §2), `rows` rows of `cols`
/// values. A projection's weight has one row per output and one column per input, as GGUF and
/// safetensors store it (`[n_out, k_in]`, RFC-0040 §3.2's dims).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix<'a> {
    data: &'a [f32],
    rows: usize,
    cols: usize,
}

impl<'a> Matrix<'a> {
    /// `data` as `rows` rows of `cols` values.
    ///
    /// # Errors
    ///
    /// [`OpError::MatrixShape`] when `data` does not hold exactly `rows · cols` values.
    pub fn new(data: &'a [f32], rows: usize, cols: usize) -> Result<Self, OpError> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(OpError::MatrixShape {
                len: data.len(),
                rows,
                cols,
            });
        }
        Ok(Self { data, rows, cols })
    }

    /// The rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row `i`.
    ///
    /// # Panics
    ///
    /// When `i ≥ rows`.
    #[must_use]
    pub fn row(&self, i: usize) -> &'a [f32] {
        assert!(i < self.rows, "row {i} of {}", self.rows);
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

/// Checks that `buf` holds at least `rows · width` values.
pub(crate) fn at_least(
    op: &'static str,
    arg: &'static str,
    buf: usize,
    rows: usize,
    width: usize,
) -> Result<(), OpError> {
    let need = rows.checked_mul(width).ok_or(OpError::Len {
        op,
        arg,
        len: buf,
        need: usize::MAX,
    })?;
    if buf < need {
        return Err(OpError::Len {
            op,
            arg,
            len: buf,
            need,
        });
    }
    Ok(())
}
