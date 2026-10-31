// SPDX-License-Identifier: MIT OR Apache-2.0

//! `kernels::reference`: the F32 reference semantics of the ops, and the reference numerics'
//! modes and devices (RFC-0040 §1, §3.3; RFC-0060 §4.1; ADR-0019 decision 1).
//!
//! An op's function here is its definition (RFC-0040 §3.3): F32 inputs, weights dequantised
//! exactly (RFC-0042 §2), F32 accumulation in the fixed order of [`RefMode::F32Fixed`]. Each
//! module's docs state the op's mathematical definition, its edge cases (`rows_active = 0`,
//! padded rows, NaN and infinities) and its accumulation contract. `graph::reference` interprets
//! the IR with these functions, and `testkit`'s `RefExecutor` drives it.
//!
//! - [`reduce`] holds the one implementation of the `F32Fixed` reduction order, which every op
//!   here and `graph::reference` use, and the `F64` mode's compensated sum.
//! - [`math`] holds the software transcendental functions the reference uses instead of a
//!   platform library: `exp` and SiLU so far.
//! - [`linear`] holds `Gemv`, `Gemm`, `GateUpAct` and `Logits`; [`norm`] holds `RmsNorm`;
//!   [`elementwise`] holds `Add` and `Mul`; [`convert`](mod@convert) holds `RowGather` and
//!   `Convert`. [`shape`] holds the [`Matrix`] they take and the [`OpError`] they refuse with.
//!   These are COMP-E1-T03's ops. Attention, RoPE and `Act` follow in COMP-E1-T17, and the
//!   sampling, tap and MoE ops in COMP-E1-T18.
//!
//! These types live here, under the F32 reference semantics of RFC-0001 §4.1, because both
//! `graph::reference` (the interpreter) and the GPU reference in `kernels-cuda` (COMP-E14-T23)
//! read them, and those two crates are peers in layer 2.
//!
//! [`act_quant`] holds the activation quantisation references of RFC-0042 §4.3: ggml's 8-bit
//! activation blocks, BF16 rounding of matrix inputs, and the fake-quant step of the op-level
//! comparison (RFC-0042 §6.4).
//!
//! [`dequant`] holds the reference dequantisers of RFC-0042 §2.1 for `F16`, `BF16`, `Q8_0`, `Q4_0`
//! and `Q4_K`, bit-exact against ggml's, and the GGUF block geometry table.
//!
//! [`embed_inject`] holds the reserved `EmbedInject` op's semantics (RFC-0043 §2.4), defined so
//! that its meaning is fixed before any provider or family uses it.

pub mod act_quant;
pub mod convert;
pub mod dequant;
pub mod elementwise;
pub mod embed_inject;
pub mod linear;
pub mod math;
pub mod norm;
pub mod reduce;
pub mod shape;

pub use convert::{convert, row_gather};
pub use elementwise::{Operand, add, mul};
pub use linear::{gate_up_act, gemm, gemv, logits};
pub use norm::{Norm, rms_norm};
pub use shape::{Matrix, OpError};

/// The reference's numeric mode (RFC-0060 §4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefMode {
    /// The ADR-0019 reference: F32 storage and accumulation; every reduction is pairwise over
    /// fixed 32-element blocks in index order ([`reduce`] spells the order out); softmax
    /// subtracts the row max.
    F32Fixed,
    /// High-precision reference for op-level conformance and for calibrating `F32Fixed`: F64
    /// storage, Neumaier-compensated summation, exp and log in F64.
    F64,
}

/// Where `F32Fixed` runs (RFC-0060 §4.1, ADR-0601 decision 7). The GPU path uses the same
/// reduction order and the reference's software transcendentals, and must be bit-identical to
/// the CPU path on nano and micro fixtures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefDevice {
    /// The CPU.
    Cpu,
    /// The GPU with this index.
    Gpu(u32),
}
