// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reference numerics' modes and devices (RFC-0060 §4.1; ADR-0019 decision 1).
//!
//! The reference forward pass runs in one of two numeric modes, on the CPU or on one GPU. These
//! types live here, under the F32 reference semantics of RFC-0001 §4.1, because both
//! `graph::reference` (the interpreter) and the GPU reference in `kernels-cuda` (COMP-E14-T23)
//! read them, and those two crates are peers in layer 2. The op semantics themselves come with
//! later tasks.
//!
//! [`act_quant`] holds the activation quantisation references of RFC-0042 §4.3: ggml's 8-bit
//! activation blocks, BF16 rounding of matrix inputs, and the fake-quant step of the op-level
//! comparison (RFC-0042 §6.4).

pub mod act_quant;

/// The reference's numeric mode (RFC-0060 §4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefMode {
    /// The ADR-0019 reference: F32 storage and accumulation; every reduction is pairwise over
    /// fixed 32-element blocks in index order; softmax subtracts the row max.
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
