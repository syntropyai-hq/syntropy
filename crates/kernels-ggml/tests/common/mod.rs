// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the mapping tests share: host buffers behind handles, op keys and shapes, and launches.

#![allow(dead_code, reason = "each test binary uses part of it")]
#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline (COMP-E13-T15)"
)]

use std::cell::RefCell;

use smallvec::SmallVec;
use syntropy_core::{BufferView, Bytes, DType, QuantScheme, ViewKind};
use syntropy_kernels::{
    BufferSlice, Error, LayoutVersion, OpArgs, OpKey, OpKind, PreparedOp, Scalar, ShapeClass,
    ShapeSig, TensorArg,
};
use syntropy_kernels_ggml::{CpuOps, HostMemory, HostRange};

/// Host buffers, each 64-byte aligned, named by their index.
#[derive(Default)]
pub struct Buffers {
    bufs: RefCell<Vec<Vec<[u8; 64]>>>,
}

impl Buffers {
    /// A new buffer holding `bytes`, padded to whole 64-byte chunks; its handle.
    pub fn add(&self, bytes: &[u8]) -> u64 {
        let mut v = vec![[0u8; 64]; bytes.len().div_ceil(64).max(1)];
        v.as_flattened_mut()[..bytes.len()].copy_from_slice(bytes);
        let mut bufs = self.bufs.borrow_mut();
        bufs.push(v);
        (bufs.len() - 1) as u64
    }

    /// A new buffer of `len` bytes, every byte `fill`.
    pub fn filled(&self, len: usize, fill: u8) -> u64 {
        self.add(&vec![fill; len])
    }

    /// The first `len` bytes of buffer `h`.
    pub fn bytes(&self, h: u64, len: usize) -> Vec<u8> {
        self.bufs.borrow()[h as usize].as_flattened()[..len].to_vec()
    }

    /// A host view of buffer `h`, as a weight.
    pub fn view(&self, h: u64) -> BufferView {
        let r = self.range(h).unwrap();
        BufferView {
            kind: ViewKind::Host,
            addr: r.addr,
            len: Bytes(r.len),
        }
    }

    /// The whole of buffer `h`, as a slice.
    pub fn slice(&self, h: u64) -> BufferSlice {
        BufferSlice {
            buffer: h,
            offset: Bytes(0),
            bytes: Bytes(self.range(h).unwrap().len),
        }
    }
}

impl HostMemory for Buffers {
    fn range(&self, handle: u64) -> Option<HostRange> {
        let bufs = self.bufs.borrow();
        let b = bufs.get(usize::try_from(handle).ok()?)?;
        Some(HostRange {
            addr: b.as_ptr() as u64,
            len: (b.len() * 64) as u64,
        })
    }
}

/// An op key in the source layout with no fusions.
pub fn key(op: OpKind, schemes: &[QuantScheme]) -> OpKey {
    OpKey {
        op,
        schemes: SmallVec::from_slice(schemes),
        layout: LayoutVersion::SOURCE,
        variant: 0,
    }
}

pub const F32: QuantScheme = QuantScheme::Plain(DType::F32);

/// A decode-bucket shape (one row).
pub fn decode(dims: &[u32]) -> ShapeSig {
    ShapeSig::new(ShapeClass::Decode, dims)
}

/// A batch-bucket shape of `rows` rows.
pub fn batch(rows: u32, dims: &[u32]) -> ShapeSig {
    ShapeSig::new(ShapeClass::Batch { rows }, dims)
}

/// A dense `[rows, n]` tensor over the whole of buffer `h`.
pub fn tensor(b: &Buffers, h: u64, dtype: DType, rows: u32, n: u32) -> TensorArg {
    let buf = BufferSlice {
        bytes: Bytes(u64::from(rows) * u64::from(n) * u64::from(dtype.bits()) / 8),
        ..b.slice(h)
    };
    TensorArg::packed(buf, dtype, [rows, n, 1, 1]).unwrap()
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// One prepared op and the buffers of its launch.
pub struct Launch {
    pub inputs: Vec<TensorArg>,
    pub outputs: Vec<TensorArg>,
    pub weights: Vec<BufferView>,
    pub statics: Vec<Scalar>,
    pub workspace: Option<BufferSlice>,
    pub step: BufferSlice,
}

impl Launch {
    /// A launch of `op` over these operands, with a workspace of the size it declares.
    pub fn new(
        b: &Buffers,
        op: &PreparedOp,
        inputs: Vec<TensorArg>,
        weights: Vec<BufferView>,
        outputs: Vec<TensorArg>,
        statics: Vec<Scalar>,
    ) -> Self {
        let workspace =
            (op.workspace.0 > 0).then(|| b.slice(b.filled(op.workspace.0 as usize, 0xA5)));
        Self {
            inputs,
            outputs,
            weights,
            statics,
            workspace,
            step: OpArgs::none().step,
        }
    }

    pub fn args(&self) -> OpArgs<'_> {
        OpArgs {
            inputs: &self.inputs,
            outputs: &self.outputs,
            weights: &self.weights,
            step: self.step,
            workspace: self.workspace,
            statics: &self.statics,
        }
    }

    /// Launches `op` with these operands.
    pub fn run(&self, ops: &CpuOps, op: &PreparedOp, b: &Buffers) -> Result<(), Error> {
        // SAFETY: every operand names a buffer of `b`, which lives for the call and is touched
        // by nothing else; the weight views are views of `b`'s buffers.
        unsafe { ops.launch(op, &self.args(), b) }
    }
}

/// A deterministic generator for test data (SplitMix64).
pub struct Gen(pub u64);

impl Gen {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }

    pub fn floats(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }

    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}
