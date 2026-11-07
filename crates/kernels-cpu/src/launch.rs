// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a CPU kernel launch carries to the stream that runs it (RFC-0040 §3.5, §7.3).
//!
//! - **The plan.** `prepare` asks the op's kernel for a [`Plan`]: the bucket's rows, the
//!   shape's dims, the weight scheme and the workspace it needs. The plan is plain data, and
//!   every launch of the prepared op carries a copy.
//! - **The operands.** `launch` copies the [`OpArgs`] into a fixed-size form of buffer
//!   handles, which is what a capture records. When the launch is pushed onto a ring, at once
//!   or at each replay of a capture, every handle is resolved to a host address and checked
//!   against its buffer, as a copy's are: a buffer freed since the capture fails the replay
//!   instead of being read or written.
//! - **The kernel** gets a [`Launch`]: the plan and the resolved operands.
//!
//! Bindings are held by value, so a launch allocates nothing. Their size caps the operands one
//! launch binds: [`MAX_TENSORS`] inputs and outputs together, [`MAX_WEIGHTS`] weight views and
//! [`MAX_STATICS`] constants. `launch` refuses more. A ring item is then 640 B \[measured:
//! `size_of`, debug, aarch64 macOS, the `a_ring_item_stays_small` test\], so a stream's ring of
//! 4,096 items takes 2.5 MiB.

use std::mem::offset_of;

use syntropy_core::{Bytes, DType, QuantScheme, ViewKind};
use syntropy_kernels::step::StepDesc;
use syntropy_kernels::{BufferSlice, Error, OpArgs, OpKey, Scalar, ShapeSig, TensorArg};

use crate::device::internal;

/// Inputs and outputs one launch binds, together: 6 \[target\], a chosen budget. The M0 ops
/// take at most four (RFC-0040 §3.1's operand lists).
pub const MAX_TENSORS: usize = 6;

/// Weight views one launch binds: 4 \[target\], a chosen budget.
pub const MAX_WEIGHTS: usize = 4;

/// Per-op constants one launch binds: 4 \[target\], a chosen budget.
pub const MAX_STATICS: usize = 4;

/// What `prepare` fixed for one prepared op. Plain data, copied into each launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The bucket: the most token rows a launch computes (`ShapeSig::rows`).
    pub rows: u32,
    /// The shape's dims, in the order `OpKind::dims` names; unused entries are 0.
    pub dims: [u32; 6],
    /// The weight scheme, `OpKey::schemes[0]`, for ops with weights.
    pub weight: Option<QuantScheme>,
    /// The output's element type, for ops whose key fixes one other than F32: `Convert`'s
    /// target.
    pub out: Option<DType>,
    /// The scratch every launch must bind (RFC-0040 §5.3).
    pub workspace: Bytes,
}

impl Plan {
    /// A plan for `shape` with no weight and no workspace.
    #[must_use]
    pub fn for_shape(shape: &ShapeSig) -> Self {
        let mut dims = [0; 6];
        for (d, &s) in dims.iter_mut().zip(&shape.dims) {
            *d = s;
        }
        Self {
            rows: shape.rows,
            dims,
            weight: None,
            out: None,
            workspace: Bytes(0),
        }
    }

    /// The plan for `key` at `shape`: [`for_shape`](Self::for_shape) with the key's weight
    /// scheme.
    #[must_use]
    pub fn for_key(key: &OpKey, shape: &ShapeSig) -> Self {
        Self {
            weight: key.schemes.first().copied(),
            ..Self::for_shape(shape)
        }
    }
}

/// An input or output resolved to host memory: the [`TensorArg`] with its buffer's address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostTensor {
    /// The address of the tensor's first element.
    pub addr: u64,
    /// The bytes the tensor spans from `addr` ([`TensorArg::span_bytes`]).
    pub bytes: u64,
    /// The element type.
    pub dtype: DType,
    /// Extent of each dimension, outermost first.
    pub dims: [u32; 4],
    /// Distance in elements between consecutive indices of each dimension.
    pub strides: [u32; 4],
}

impl HostTensor {
    /// Whether the tensor is densely packed with these `dims` and this `dtype`.
    #[must_use]
    pub fn is_packed(&self, dtype: DType, dims: [u32; 4]) -> bool {
        let probe = BufferSlice {
            buffer: 0,
            offset: Bytes(0),
            bytes: Bytes(self.bytes),
        };
        TensorArg::packed(probe, dtype, dims).is_some_and(|t| {
            t.dtype == self.dtype && t.dims == self.dims && t.strides == self.strides
        })
    }

    /// The host range the tensor spans.
    #[must_use]
    pub fn span(&self) -> HostSpan {
        HostSpan {
            addr: self.addr,
            bytes: self.bytes,
        }
    }
}

/// A resolved host range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostSpan {
    /// The first byte.
    pub addr: u64,
    /// The length.
    pub bytes: u64,
}

impl HostSpan {
    /// Whether the two ranges share a byte.
    #[must_use]
    pub fn overlaps(&self, other: &HostSpan) -> bool {
        self.bytes > 0
            && other.bytes > 0
            && self.addr < other.addr.saturating_add(other.bytes)
            && other.addr < self.addr.saturating_add(self.bytes)
    }
}

/// One launch's operands, in a fixed-size form: `T` for an input or output and `S` for a
/// buffer range. A capture keeps them as handles ([`TensorArg`], [`BufferSlice`]); a ring holds
/// them resolved ([`HostTensor`], [`HostSpan`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bindings<T, S> {
    /// The inputs, then the outputs.
    tensors: [Option<T>; MAX_TENSORS],
    inputs: u8,
    outputs: u8,
    /// The weight views, checked to be host-addressable, as host ranges.
    weights: [Option<HostSpan>; MAX_WEIGHTS],
    step: Option<S>,
    workspace: Option<S>,
    statics: [Option<Scalar>; MAX_STATICS],
}

/// The operands as `launch` received them.
pub(crate) type Unresolved = Bindings<TensorArg, BufferSlice>;

/// The operands with every buffer resolved to host memory.
pub(crate) type Resolved = Bindings<HostTensor, HostSpan>;

impl Unresolved {
    /// `args` in the fixed-size form. Every weight view must be host-addressable.
    pub(crate) fn from_args(args: &OpArgs<'_>) -> Result<Self, Error> {
        let n = args.inputs.len() + args.outputs.len();
        if n > MAX_TENSORS || args.weights.len() > MAX_WEIGHTS || args.statics.len() > MAX_STATICS {
            return Err(internal(format!(
                "a launch binds {} tensors, {} weights and {} constants; a CPU launch takes at \
                 most {MAX_TENSORS}, {MAX_WEIGHTS} and {MAX_STATICS}",
                n,
                args.weights.len(),
                args.statics.len()
            )));
        }
        let mut tensors = [None; MAX_TENSORS];
        for (t, a) in tensors
            .iter_mut()
            .zip(args.inputs.iter().chain(args.outputs))
        {
            *t = Some(*a);
        }
        let mut weights = [None; MAX_WEIGHTS];
        for (w, a) in weights.iter_mut().zip(args.weights) {
            if !matches!(a.kind, ViewKind::Host | ViewKind::Unified) || a.addr == 0 {
                return Err(internal(
                    "a weight view is not host memory, which a CPU kernel reads".to_owned(),
                ));
            }
            *w = Some(HostSpan {
                addr: a.addr,
                bytes: a.len.0,
            });
        }
        let mut statics = [None; MAX_STATICS];
        for (s, a) in statics.iter_mut().zip(args.statics) {
            *s = Some(*a);
        }
        Ok(Self {
            tensors,
            inputs: args.inputs.len() as u8,
            outputs: args.outputs.len() as u8,
            weights,
            step: (args.step.bytes.0 > 0).then_some(args.step),
            workspace: args.workspace,
            statics,
        })
    }

    /// The operands with each buffer range resolved by `range` to its host address. Each
    /// tensor must lie inside its range.
    pub(crate) fn resolve(
        &self,
        mut range: impl FnMut(BufferSlice) -> Result<u64, Error>,
    ) -> Result<Resolved, Error> {
        let mut tensors = [None; MAX_TENSORS];
        for (out, t) in tensors.iter_mut().zip(&self.tensors) {
            if let Some(t) = t {
                if !t.fits() {
                    return Err(internal(format!(
                        "a tensor spans {} bytes past its {}-byte slice",
                        t.span_bytes(),
                        t.buf.bytes.0
                    )));
                }
                *out = Some(HostTensor {
                    addr: range(t.buf)?,
                    bytes: t.span_bytes(),
                    dtype: t.dtype,
                    dims: t.dims,
                    strides: t.strides,
                });
            }
        }
        let span = |s: BufferSlice, range: &mut dyn FnMut(BufferSlice) -> Result<u64, Error>| {
            range(s).map(|addr| HostSpan {
                addr,
                bytes: s.bytes.0,
            })
        };
        Ok(Resolved {
            tensors,
            inputs: self.inputs,
            outputs: self.outputs,
            weights: self.weights,
            step: self.step.map(|s| span(s, &mut range)).transpose()?,
            workspace: self.workspace.map(|s| span(s, &mut range)).transpose()?,
            statics: self.statics,
        })
    }
}

/// What a CPU kernel runs with: its prepared op's plan and its resolved operands.
///
/// Every address names host memory of a buffer the device knows (or, for a weight, a view the
/// tier manager handed out), checked to cover the range when the launch reached the ring. The
/// tier manager frees such memory only between steps, once the streams using it have drained
/// (RFC-0001 §9.1), so the memory stays valid while the kernel runs.
#[derive(Clone, Copy, Debug)]
pub struct Launch<'a> {
    plan: &'a Plan,
    args: &'a Resolved,
}

impl<'a> Launch<'a> {
    pub(crate) fn new(plan: &'a Plan, args: &'a Resolved) -> Self {
        Self { plan, args }
    }

    /// The prepared op's plan.
    #[must_use]
    pub fn plan(&self) -> &Plan {
        self.plan
    }

    /// Input `i`.
    #[must_use]
    pub fn input(&self, i: usize) -> Option<HostTensor> {
        (i < usize::from(self.args.inputs))
            .then(|| self.args.tensors[i])
            .flatten()
    }

    /// How many inputs the launch binds.
    #[must_use]
    pub fn inputs(&self) -> usize {
        usize::from(self.args.inputs)
    }

    /// Output `i`.
    #[must_use]
    pub fn output(&self, i: usize) -> Option<HostTensor> {
        (i < usize::from(self.args.outputs))
            .then(|| self.args.tensors[usize::from(self.args.inputs) + i])
            .flatten()
    }

    /// How many outputs the launch binds.
    #[must_use]
    pub fn outputs(&self) -> usize {
        usize::from(self.args.outputs)
    }

    /// Weight view `i`, which is host or unified memory, as a host range.
    #[must_use]
    pub fn weight(&self, i: usize) -> Option<HostSpan> {
        self.args.weights.get(i).copied().flatten()
    }

    /// The workspace, if the launch binds one.
    #[must_use]
    pub fn workspace(&self) -> Option<HostSpan> {
        self.args.workspace
    }

    /// Constant `i`.
    #[must_use]
    pub fn statics(&self, i: usize) -> Option<Scalar> {
        self.args.statics.get(i).copied().flatten()
    }

    /// The step descriptor's `rows_active`, or `None` when the launch binds no descriptor (an
    /// empty `step` slice), in which case the kernel computes the whole bucket, as the ggml
    /// provider does.
    ///
    /// # Errors
    ///
    /// A descriptor too short to hold the field.
    pub fn rows_active(&self) -> Result<Option<u32>, Error> {
        const AT: usize = offset_of!(StepDesc, rows_active);
        let Some(step) = self.args.step else {
            return Ok(None);
        };
        if step.bytes < (AT + 4) as u64 {
            return Err(internal(format!(
                "a {}-byte step descriptor has no rows_active",
                step.bytes
            )));
        }
        // SAFETY: the range was resolved from a live buffer of this device that covers
        // `step.bytes` bytes from `step.addr` (the type's docs), and the field's four bytes
        // lie inside it. The host writes the descriptor before the launch's step starts.
        let value = unsafe { std::ptr::read_unaligned((step.addr as usize + AT) as *const u32) };
        Ok(Some(value))
    }
}

/// The kernel for one op kind: what `query`, `prepare`, `launch` and the stream's runner call.
#[derive(Clone, Copy, Debug)]
pub struct Kernel {
    /// The plan for a key and shape, or `None` when the kernel does not support them, which
    /// makes `prepare` fail with `OpUnsupported`.
    pub plan: fn(&OpKey, &ShapeSig) -> Option<Plan>,
    /// The capabilities `query` reports, or `None`.
    pub caps: fn(&OpKey, &ShapeSig) -> Option<syntropy_kernels::OpCaps>,
    /// Checks a launch's operands against the plan when `launch` is called, so a mismatch is
    /// the caller's error and never reaches the stream, where an error is sticky.
    pub check: fn(&Plan, &OpArgs<'_>) -> Result<(), Error>,
    /// Runs the op on the calling thread.
    pub run: KernelFn,
}

/// One CPU kernel's entry point: it runs the op on the calling thread over the launch's
/// resolved operands.
pub type KernelFn = fn(&Launch<'_>) -> Result<(), Error>;

impl Kernel {
    /// A stand-in kernel, for tests: every key at every shape, no workspace, no capabilities,
    /// no operand checks.
    #[must_use]
    pub fn stand_in(run: KernelFn) -> Self {
        Self {
            plan: |key, shape| Some(Plan::for_key(key, shape)),
            caps: |_, _| None,
            check: |_, _| Ok(()),
            run,
        }
    }
}

#[cfg(test)]
mod tests {
    /// A ring slot holds a launch's operands by value; this bounds what that costs per stream.
    #[test]
    fn a_ring_item_stays_small() {
        let item = size_of::<crate::stream::Item>();
        eprintln!("ring item: {item} bytes");
        assert!(item <= 640, "{item}");
    }
}
