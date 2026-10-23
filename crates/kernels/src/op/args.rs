// SPDX-License-Identifier: MIT OR Apache-2.0

//! What one launch binds (RFC-0040 §3.5): tensors, weight views, the step descriptor, the
//! workspace and per-op constants.
//!
//! Nothing here changes from step to step. Values that do live in the step descriptor
//! ([`crate::step`]), which `OpArgs::step` points at, so one captured graph serves every step.

use syntropy_core::{BufferView, Bytes, DType};

use crate::memory::BufferSlice;

/// The buffers and constants one launch binds (RFC-0040 §3.5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpArgs<'a> {
    /// Activations, KV page pools and states the op reads.
    pub inputs: &'a [TensorArg],
    /// What the op writes.
    pub outputs: &'a [TensorArg],
    /// Weights, as the tier manager's `view` of each lease returns them (RFC-0020 §3). Their
    /// layout must equal the op key's `layout`.
    pub weights: &'a [BufferView],
    /// The device-resident copy of the step descriptor.
    pub step: BufferSlice,
    /// Scratch, when the op's `OpCaps::workspace` is not zero (RFC-0040 §5.3).
    pub workspace: Option<BufferSlice>,
    /// Per-op constants fixed at `prepare`: eps, rope theta, scales.
    pub statics: &'a [Scalar],
}

impl OpArgs<'_> {
    /// No bindings: empty operand, weight and constant lists, an empty step slice and no
    /// workspace. A provider that replays a recorded launch, whose arguments were bound at
    /// capture, passes it.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            inputs: &[],
            outputs: &[],
            weights: &[],
            step: BufferSlice {
                buffer: 0,
                offset: Bytes(0),
                bytes: Bytes(0),
            },
            workspace: None,
            statics: &[],
        }
    }
}

/// One tensor operand (RFC-0040 §3.5).
///
/// `dims` and `strides` have four entries; an operand of lower rank sets its trailing dims to
/// 1. Strides count elements, not bytes, so they mean the same for every dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorArg {
    /// The bytes holding the tensor.
    pub buf: BufferSlice,
    /// The element type.
    pub dtype: DType,
    /// Extent of each dimension.
    pub dims: [u32; 4],
    /// Distance in elements between consecutive indices of each dimension.
    pub strides: [u32; 4],
}

/// A per-op constant (RFC-0040 §3.5).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    /// A float constant: eps, rope theta, a scale.
    F32(f32),
    /// A signed integer constant.
    I32(i32),
    /// An unsigned integer constant: a count or an index.
    U32(u32),
}

impl TensorArg {
    /// A densely packed tensor with `dims[0]` the outermost dimension and `dims[3]` the
    /// innermost: `strides[3]` is 1 and each stride is the product of the extents inside it.
    /// `None` when a stride does not fit in a `u32`.
    #[must_use]
    pub fn packed(buf: BufferSlice, dtype: DType, dims: [u32; 4]) -> Option<Self> {
        let mut strides = [1u32; 4];
        for d in (0..3).rev() {
            strides[d] = strides[d + 1].checked_mul(dims[d + 1])?;
        }
        Some(Self {
            buf,
            dtype,
            dims,
            strides,
        })
    }

    /// The bytes the tensor spans from the start of `buf`: one past its last element,
    /// rounded up to a whole byte for sub-byte dtypes. 0 when any dimension is empty;
    /// `u64::MAX` when the span does not fit in a `u64`.
    #[must_use]
    pub fn span_bytes(&self) -> u64 {
        if self.dims.contains(&0) {
            return 0;
        }
        let last: u128 = self
            .dims
            .iter()
            .zip(self.strides)
            .map(|(&d, s)| u128::from(d - 1) * u128::from(s))
            .sum();
        let bits = (last + 1) * u128::from(self.dtype.bits());
        u64::try_from(bits.div_ceil(8)).unwrap_or(u64::MAX)
    }

    /// Whether every element lies inside `buf`.
    #[must_use]
    pub fn fits(&self) -> bool {
        self.span_bytes() <= self.buf.bytes.0
    }
}
