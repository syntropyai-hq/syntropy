// SPDX-License-Identifier: MIT OR Apache-2.0

//! The accumulation contract: the static check of each kernel's [`Numerics`] against its
//! op's rule, and the dynamic detector (RFC-0042 §4, §4.1, §4.2; ADR-0421 items 1, 2, 6).
//!
//! **Static check.** Each kernel declares a [`Numerics`] in the [`OpCaps`] its `query`
//! returns (RFC-0040 §3.2). [`rule`] gives each [`OpKind`]'s rule as a [`NumericsRule`], and
//! [`check`] holds a declaration to it. RFC-0040 §6.2 condition 2 makes it a selection
//! condition: [`check_caps`] is what selection calls, and a kernel that fails it is never a
//! candidate. [`check_registry`] runs the check over every key of an [`OpRegistry`], and
//! reports a key with no declaration as [`NumericsViolation::Undeclared`].
//!
//! FP16 and BF16 accumulators (§4.1 rule 1) cannot be declared at all. RFC-0042 §4 gives
//! [`AccumKind`] three variants, `F32`, `I32ThenF32` and `F64`, and no other:
//!
//! ```compile_fail
//! let _ = syntropy_core::AccumKind::F16;
//! ```
//!
//! ```compile_fail
//! let _ = syntropy_core::AccumKind::BF16;
//! ```
//!
//! **Dynamic detector.** A kernel can declare F32 and still accumulate in FP16, or flush F32
//! subnormals to zero (§4.1 rules 1 and 5). [`detect_dot`] runs a dot-product kernel on the
//! [`detector_cases`], inputs on which either fault fails grossly while F32 accumulation in
//! any order is exact, and rejects it on the first case it misses. COMP-E13-T13 adds the
//! per-provider checks: the fast-math flag audit and the bit-exact list of RFC-0042 §6.4.
//!
//! The rule for a matrix product's `act_in` depends on the checkpoint and the weight scheme
//! (RFC-0042 §4.3 rules a–c), not on the op alone, so COMP-E13-T03 checks it at selection.
//! Here an op that is not a matrix product must not quantise its input at all.

use std::fmt;

use syntropy_core::{AccumKind, ActQuant, DType, Numerics};

use crate::op::{OpCaps, OpKey, OpKind, OpRegistry, RouteNorm};

/// The longest scale block an integer dot product may sum in I32 (RFC-0042 §4.1 rule 2).
///
/// No scale block in RFC-0042 §2 is longer than 256 products. An 8-bit product is below
/// 2^15 in magnitude, so I32 stays exact for sums of fewer than 2^16 products, and 256 is
/// well inside that.
pub const MAX_I32_BLOCK: u32 = 256;

/// The output types of a matrix product whose next op may take a BF16 input (RFC-0042 §4.2,
/// §4.3 rule b).
const OUT_MATRIX: &[DType] = &[DType::F32, DType::BF16];
/// F32 only: the residual stream, logits and every op §4.2 gives no narrower output.
const OUT_F32: &[DType] = &[DType::F32];
/// The model's KV dtypes (RFC-0042 §4.2, "The model's KV dtype"): F32, BF16, F16, or FP8
/// where the checkpoint declares an FP8 cache.
const OUT_KV: &[DType] = &[
    DType::F32,
    DType::BF16,
    DType::F16,
    DType::F8E4M3,
    DType::F8E5M2,
];
/// Ops that output ids or indices as well as F32 values.
const OUT_INDEX: &[DType] = &[DType::F32, DType::I32, DType::I64];

/// One op's accumulation rule (RFC-0042 §4.1, §4.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericsRule {
    /// The op is a matrix product (`Gemv`, `Gemm`, `GateUpAct`, `MoeGrouped`, `Logits`):
    /// it may accumulate integer dot products in I32 within a scale block (§4.1 rule 2), and
    /// its `act_in` follows §4.3. Every other op accumulates in F32 or F64 and declares
    /// `act_in` [`ActQuant::None`].
    pub matrix_product: bool,
    /// The output types §4.2 allows.
    pub out: &'static [DType],
    /// The op must keep its softmax max and sum in F32 (§4.2: attention, `MoeRoute` with
    /// softmax, and the sampling ops' softmax over F32 logits).
    pub softmax_f32: bool,
    /// The op must keep its recurrent state in F32 (§4.2: `ScanDecode`, `ScanPrefill`).
    pub state_f32: bool,
}

impl NumericsRule {
    /// The weakest declaration that meets the rule: I32 within a 256-product block on a
    /// matrix product and F32 elsewhere, no activation quantisation, the first output type
    /// listed, and F32 softmax or state only where required.
    #[must_use]
    pub fn minimal(&self) -> Numerics {
        Numerics {
            accumulate: if self.matrix_product {
                AccumKind::I32ThenF32 {
                    max_block: MAX_I32_BLOCK,
                }
            } else {
                AccumKind::F32
            },
            act_in: ActQuant::None,
            out: self.out[0],
            softmax_f32: self.softmax_f32,
            state_f32: self.state_f32,
        }
    }
}

/// The rule `op` must meet (RFC-0042 §4.2).
///
/// Every kind is listed by name, so a new kind needs a decision here.
#[must_use]
pub fn rule(op: &OpKind) -> NumericsRule {
    let base = NumericsRule {
        matrix_product: false,
        out: OUT_F32,
        softmax_f32: false,
        state_f32: false,
    };
    match op {
        // §4.2: "outputs F32. BF16 outputs only where the next op is a matrix product whose
        // input may be BF16". `MoeGrouped` feeds `MoeCombine` and `Logits` feeds
        // `LogitsProcess`, so neither has a matrix product next.
        OpKind::Gemv | OpKind::Gemm | OpKind::GateUpAct { .. } => NumericsRule {
            matrix_product: true,
            out: OUT_MATRIX,
            ..base
        },
        OpKind::MoeGrouped | OpKind::Logits => NumericsRule {
            matrix_product: true,
            ..base
        },
        // §4.2: running max and sum in F32; split-KV partials merged in F32.
        OpKind::AttnDecode(_) | OpKind::AttnPrefill(_) | OpKind::AttnPartial(_) => NumericsRule {
            softmax_f32: true,
            ..base
        },
        OpKind::AttnMerge => NumericsRule {
            softmax_f32: true,
            ..base
        },
        // §4.2: logits, softmax or sigmoid, and top-k all in F32.
        OpKind::MoeRoute { norm } => NumericsRule {
            out: OUT_INDEX,
            softmax_f32: matches!(norm, RouteNorm::Softmax),
            ..base
        },
        // §4.2: F32 logits; RFC-0040 §3.8: `TapGather` follows the F32 softmax rule.
        OpKind::TopKTopP | OpKind::SampleSeeded | OpKind::VerifyAccept | OpKind::TapGather => {
            NumericsRule {
                out: OUT_INDEX,
                softmax_f32: true,
                ..base
            }
        }
        OpKind::Argmax | OpKind::AttnIndex(_) => NumericsRule {
            out: OUT_INDEX,
            ..base
        },
        // §4.2: state and gates in F32.
        OpKind::ScanDecode(_) | OpKind::ScanPrefill(_) => NumericsRule {
            state_f32: true,
            ..base
        },
        // §4.2: rounded once to the model's KV dtype. The data-movement ops carry KV pages.
        OpKind::KvWrite(_)
        | OpKind::GatherRows
        | OpKind::ScatterRows
        | OpKind::KvPack
        | OpKind::KvUnpack => NumericsRule {
            out: OUT_KV,
            ..base
        },
        // Dtype conversion: the output type is the op's purpose.
        OpKind::Convert => NumericsRule {
            out: &DType::ALL,
            ..base
        },
        OpKind::CausalConv1d
        | OpKind::RmsNorm
        | OpKind::LayerNorm
        | OpKind::Rope(_)
        | OpKind::Act(_)
        | OpKind::Add
        | OpKind::Mul
        | OpKind::Scale
        | OpKind::HyperMix
        | OpKind::MoeCombine
        | OpKind::RowGather
        | OpKind::EmbedInject
        | OpKind::Pool(_)
        | OpKind::LogitsProcess
        | OpKind::PartialReduce
        | OpKind::ApplyTableDeltas
        | OpKind::Timestamp
        | OpKind::HiddenTap => base,
    }
}

/// How a [`Numerics`] declaration breaks its op's rule (RFC-0042 §4.1, §4.2).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumericsViolation {
    /// The kernel declares no `Numerics` (ADR-0421 item 6).
    Undeclared,
    /// `I32ThenF32` on an op that is not a matrix product (§4.1 rule 2, §4.2).
    IntegerAccumulation,
    /// `I32ThenF32` with a block of zero or more than [`MAX_I32_BLOCK`] products (§4.1
    /// rule 2).
    IntegerBlock {
        /// The declared block.
        max_block: u32,
    },
    /// Activation quantisation on an op that is not a matrix product (§4.3).
    ActivationQuant(ActQuant),
    /// An output type §4.2 does not allow for the op.
    Output(DType),
    /// `softmax_f32` false where §4.2 requires the softmax max and sum in F32.
    SoftmaxNotF32,
    /// `state_f32` false where §4.2 requires the scan state in F32.
    StateNotF32,
}

impl fmt::Display for NumericsViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Undeclared => write!(f, "no Numerics declaration"),
            Self::IntegerAccumulation => {
                write!(f, "I32 accumulation on an op that is not a matrix product")
            }
            Self::IntegerBlock { max_block } => write!(
                f,
                "I32 accumulation over {max_block} products; the block must be 1 to {MAX_I32_BLOCK}"
            ),
            Self::ActivationQuant(q) => {
                write!(f, "activation quantisation {q:?} outside a matrix product")
            }
            Self::Output(dtype) => write!(f, "output type {dtype:?} not allowed for the op"),
            Self::SoftmaxNotF32 => write!(f, "softmax max and sum not kept in F32"),
            Self::StateNotF32 => write!(f, "scan state not kept in F32"),
        }
    }
}

impl std::error::Error for NumericsViolation {}

/// Holds `numerics` to `op`'s [`rule`] (RFC-0042 §4.1, §4.2; RFC-0040 §6.2 condition 2).
///
/// # Errors
///
/// The first [`NumericsViolation`] found, in the order the variants are declared.
pub fn check(op: &OpKind, numerics: &Numerics) -> Result<(), NumericsViolation> {
    let rule = rule(op);
    match numerics.accumulate {
        AccumKind::I32ThenF32 { .. } if !rule.matrix_product => {
            return Err(NumericsViolation::IntegerAccumulation);
        }
        AccumKind::I32ThenF32 { max_block } if max_block == 0 || max_block > MAX_I32_BLOCK => {
            return Err(NumericsViolation::IntegerBlock { max_block });
        }
        _ => {}
    }
    if !rule.matrix_product && numerics.act_in != ActQuant::None {
        return Err(NumericsViolation::ActivationQuant(numerics.act_in));
    }
    if !rule.out.contains(&numerics.out) {
        return Err(NumericsViolation::Output(numerics.out));
    }
    if rule.softmax_f32 && !numerics.softmax_f32 {
        return Err(NumericsViolation::SoftmaxNotF32);
    }
    if rule.state_f32 && !numerics.state_f32 {
        return Err(NumericsViolation::StateNotF32);
    }
    Ok(())
}

/// [`check`] on the `numerics` a provider's `query` returned for `key` (RFC-0040 §6.2
/// condition 2): a kernel that fails it is never a candidate.
///
/// # Errors
///
/// As [`check`].
pub fn check_caps(key: &OpKey, caps: &OpCaps) -> Result<(), NumericsViolation> {
    check(&key.op, &caps.numerics)
}

/// Runs [`check`] over every key of `registry`, with `declared` giving each key's
/// declaration, or `None` when the kernel declares none.
///
/// # Errors
///
/// Every failing key with its [`NumericsViolation`], in `OpKey` order.
pub fn check_registry(
    registry: &OpRegistry,
    mut declared: impl FnMut(&OpKey) -> Option<Numerics>,
) -> Result<(), Vec<(OpKey, NumericsViolation)>> {
    let failures: Vec<_> = registry
        .iter()
        .filter_map(|key| {
            let result = match declared(key) {
                Some(numerics) => check(&key.op, &numerics),
                None => Err(NumericsViolation::Undeclared),
            };
            result.err().map(|violation| (key.clone(), violation))
        })
        .collect();
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

/// One input of the dynamic detector: a dot product whose F32 result is exact in any
/// summation order, and on which FP16 accumulation or flush-to-zero fails grossly.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct DetectorCase {
    /// What the case catches.
    pub name: &'static str,
    /// The left operand.
    pub a: Vec<f32>,
    /// The right operand, the same length as `a`.
    pub b: Vec<f32>,
    /// The exact dot product, computed in F64 and exactly representable in F32.
    pub expected: f32,
}

impl DetectorCase {
    fn new(name: &'static str, a: Vec<f32>, b: Vec<f32>) -> Self {
        let exact: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        // Every case is built so that this conversion is exact; the tests check it.
        let expected = exact as f32;
        Self {
            name,
            a,
            b,
            expected,
        }
    }
}

/// The largest relative error [`detect_dot`] accepts: 2^-8 \[target\], a chosen threshold.
///
/// F32 accumulation in any order gets every case exactly, so it passes with no margin
/// needed. Every fault the detector looks for misses by 50 % or more (see
/// [`detector_cases`]), so the threshold sits far from both.
pub const DETECTOR_REL_TOL: f32 = 1.0 / 256.0;

/// The detector's inputs (RFC-0042 §4.1 rules 1 and 5).
///
/// - `f16_overflow`: 4096 products of 32 × 32 sum to 2^22. The total exceeds FP16's largest
///   finite value, 65504, so an FP16 accumulator returns infinity in any summation order,
///   sequential or a tree.
/// - `f16_swamping`: 1 plus 4096 products of 2^-12 × 1 sums to 2. Each 2^-12 is a quarter of
///   FP16's unit in the last place at 1 (2^-10), and a thirty-second of BF16's (2^-7), so a
///   sequential FP16 or BF16 accumulator that starts at 1 rounds every addition away and
///   returns 1. A pairwise tree sums most of the small terms before they meet the 1, so this
///   case catches sequential accumulation only.
/// - `f16_large_terms`: 2048 plus 2048 products of 1 × 1 sums to 4096. FP16's spacing at
///   2048 is 2, so each `+ 1` rounds back to even and a sequential accumulator stays at 2048.
///   Like `f16_swamping`, it catches sequential accumulation only.
/// - `f32_subnormal_inputs`: 1024 products of 2^-130 × 1 sum to 2^-120. An input of 2^-130
///   is an F32 subnormal, so a kernel that treats subnormal inputs as zero returns 0.
/// - `f32_subnormal_products`: 1024 products of 2^-70 × 2^-70 sum to 2^-130. Each product,
///   2^-140, is an F32 subnormal, so a kernel that flushes subnormal results returns 0.
///
/// FP16 accumulation in any order also fails both subnormal cases, because FP16's smallest
/// subnormal is 2^-24: its partial sums of these values round to 0, as a flush would.
#[must_use]
pub fn detector_cases() -> Vec<DetectorCase> {
    let p = pow2;
    let mut swamp_a = vec![1.0];
    swamp_a.extend(std::iter::repeat_n(p(-12), 4096));
    let mut large_a = vec![2048.0];
    large_a.extend(std::iter::repeat_n(1.0, 2048));
    vec![
        DetectorCase::new("f16_overflow", vec![32.0; 4096], vec![32.0; 4096]),
        DetectorCase::new("f16_swamping", swamp_a, vec![1.0; 4097]),
        DetectorCase::new("f16_large_terms", large_a, vec![1.0; 2049]),
        DetectorCase::new("f32_subnormal_inputs", vec![p(-130); 1024], vec![1.0; 1024]),
        DetectorCase::new(
            "f32_subnormal_products",
            vec![p(-70); 1024],
            vec![p(-70); 1024],
        ),
    ]
}

/// 2^`e` for `e` in F32's range, normal or subnormal, built from its bits (RFC-0042 §4.1
/// rule 7). `powi` computes 2^-130 as 1 / 2^130, which overflows, and returns zero.
fn pow2(e: i32) -> f32 {
    assert!((-149..=127).contains(&e), "2^{e} is outside F32's range");
    let bits = if e >= -126 {
        u32::try_from(e + 127).expect("in range") << 23
    } else {
        1u32 << u32::try_from(e + 149).expect("in range")
    };
    f32::from_bits(bits)
}

/// A case the kernel under test missed.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectorFailure {
    /// The case's name ([`DetectorCase::name`]).
    pub case: &'static str,
    /// The exact result.
    pub expected: f32,
    /// What the kernel returned.
    pub got: f32,
}

impl fmt::Display for DetectorFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "accumulation detector case {}: expected {:e}, kernel returned {:e}",
            self.case, self.expected, self.got
        )
    }
}

impl std::error::Error for DetectorFailure {}

/// Runs the dot-product kernel `dot` on every [`detector_cases`] input (RFC-0042 §4.1 rules
/// 1 and 5).
///
/// # Errors
///
/// The first case whose result is not within [`DETECTOR_REL_TOL`] of the exact value,
/// including a NaN or infinite result.
pub fn detect_dot(mut dot: impl FnMut(&[f32], &[f32]) -> f32) -> Result<(), DetectorFailure> {
    for case in detector_cases() {
        let got = dot(&case.a, &case.b);
        let bound = DETECTOR_REL_TOL * case.expected.abs();
        // A NaN or infinite `got` makes `within` false.
        let within = (got - case.expected).abs() <= bound;
        if !within {
            return Err(DetectorFailure {
                case: case.name,
                expected: case.expected,
                got,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
