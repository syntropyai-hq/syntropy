// SPDX-License-Identifier: MIT OR Apache-2.0

//! A kernel's declared numerics: accumulation, activation quantisation and output type.
//!
//! RFC-0042 §1.1 defines [`ActQuant`] and §4 defines [`Numerics`] and [`AccumKind`]. Each
//! kernel declares a `Numerics` in its `OpCaps` (RFC-0040 §3.2), and selection rejects any
//! kernel whose declaration breaks RFC-0042 §4's rules (ADR-0421 item 6). The rules
//! themselves are checked elsewhere; these are the declarations.

use crate::dtype::DType;

/// Activation quantisation a matrix product performs on its input (RFC-0042 §1.1, §4.3).
///
/// Whether a variant is allowed in exact mode depends on the checkpoint and the weight
/// scheme (RFC-0042 §4.3 rules a–c); this type only names it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ActQuant {
    /// No activation quantisation: the product reads its input unchanged.
    None,
    /// BF16 rounding of the matrix input (§4.3 rule b).
    Bf16,
    /// 8-bit blocks of 32 elements with one F16 scale, as ggml's `Q8_0`/`Q8_1` layouts
    /// (§4.3 rule b).
    Q8Block32,
    /// 8-bit blocks of 256 elements, as ggml's `Q8_K` layout (§4.3 rule b).
    Q8K256,
    /// Dynamic FP8 (E4M3) per token, in groups of `group` elements along K, with
    /// scale = amax / 448: the scheme FP8 checkpoints declare as
    /// `activation_scheme: dynamic` (§4.3 rule a).
    Fp8Dynamic {
        /// Elements per scale group along K.
        group: u16,
    },
    /// MX FP8 activations (§4.3 rule c unless the checkpoint declares them).
    Mxfp8,
    /// NVFP4 activations (§4.3 rule c unless the checkpoint declares them).
    Nvfp4,
}

/// How a kernel accumulates its dot products (RFC-0042 §4, §4.1 rules 1–2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AccumKind {
    /// F32 accumulators (§4.1 rule 1).
    F32,
    /// Integer dot products accumulated exactly in I32 within one scale block, then
    /// converted to F32 and multiplied by the block's scales (§4.1 rule 2).
    I32ThenF32 {
        /// The most products the kernel sums in I32 before converting to F32.
        max_block: u32,
    },
    /// F64 accumulators.
    F64,
}

/// A kernel's numerics declaration (RFC-0042 §4; ADR-0421 item 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Numerics {
    /// How dot products accumulate.
    pub accumulate: AccumKind,
    /// What the op does to its activation input (§4.3).
    pub act_in: ActQuant,
    /// The output type: F32, BF16 or F16, with F16 only where RFC-0042 §4.2 allows it.
    pub out: DType,
    /// Attention: the running max and sum are kept in F32.
    pub softmax_f32: bool,
    /// Scans: the recurrent state is kept in F32.
    pub state_f32: bool,
}
