// SPDX-License-Identifier: MIT OR Apache-2.0

//! The activation quantisation rule for matrix products at selection (RFC-0042 §4.3 rules
//! a–c; ADR-0421 items 3–5).
//!
//! [`check_act`] holds a candidate's `Numerics::act_in` to the rule for one op on one
//! weight tensor, which [`ActContext`] describes:
//! - **(a) Declared by the checkpoint.** When [`ActContext::declared`] is set, the kernel must
//!   apply exactly that quantisation, which the reference also applies. A kernel that skips
//!   it, such as a W8A16 kernel on a declared-W8A8 op, or that applies another, is rejected.
//! - **(b) ggml-equivalent.** With nothing declared, BF16 rounding of the input is allowed,
//!   and so is an 8-bit block when the weights run through ggml in the calibration baseline:
//!   GGUF integer-coded weights, and safetensors tensors the census found transcribable
//!   ([`ActContext::transcription`], RFC-0042 §3.5). The block must be the one ggml uses
//!   for that type ([`ggml_activation_block`]).
//! - **(c) Certified.** Anything else (undeclared FP8, MX FP8 or FP4 activations, or an
//!   8-bit block on a tensor with no ggml baseline) needs a certificate:
//!   the candidate's `kernel_hash` in [`ActContext::certified`]. Without one it is a candidate
//!   only in approximate mode with the `act_lowp` knob on ([`ActContext::act_lowp`], RFC-0042
//!   §10).
//!
//! The caller resolves certificates: it knows the family, the dtype class and the shape class,
//! and lists the kernel hashes the release's `ConformanceManifest` certifies for them
//! (RFC-0060 §6.4 `PathCert`). COMP-E14-T06 wires the manifest to this list.
//!
//! An op that is not a matrix product must not quantise its input at all, which
//! [`check`](super::check) already enforces; this rule passes it through.

use std::fmt;

use syntropy_core::{ActQuant, Blake3, CtScheme, GgmlType, QuantScheme};

use super::rule;
use crate::op::{OpCaps, OpKey};
use crate::reference::act_quant::ggml_activation_block;

/// What the activation rule needs to know about one matrix product's weight tensor and the
/// run (RFC-0042 §4.3).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ActContext {
    /// The activation quantisation the checkpoint declares for this op (rule a): the FP8
    /// config's `activation_scheme: dynamic`, or compressed-tensors `input_activations`.
    pub declared: Option<ActQuant>,
    /// The ggml type the census found the tensor exactly transcribable to (RFC-0042 §3.5),
    /// for a safetensors tensor. Ignored for a GGUF tensor, whose type is its own.
    pub transcription: Option<GgmlType>,
    /// The kernel hashes (`OpCaps::kernel_hash`) a certificate allows for this op under rule
    /// (c), resolved by the caller for the model's family, dtype class and shape class.
    pub certified: Vec<Blake3>,
    /// Approximate mode with the `act_lowp` knob on (RFC-0042 §10): a kernel rule (c) would
    /// reject is admitted, and labelled by the caller. False in exact mode.
    pub act_lowp: bool,
}

/// Which part of the rule admitted a candidate's `act_in`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActBasis {
    /// The op is not a matrix product, or the kernel reads its input unchanged.
    Unquantised,
    /// Rule (a): the kernel applies the checkpoint's declared quantisation.
    Declared,
    /// Rule (b): BF16 rounding of the input.
    Bf16,
    /// Rule (b): the 8-bit block ggml uses for this ggml type.
    GgmlBlock(GgmlType),
    /// Rule (c): a certificate allows the kernel.
    Certified,
    /// Approximate mode with `act_lowp` on; nothing in exact mode allows the kernel.
    ActLowp,
}

/// How a candidate's `act_in` breaks RFC-0042 §4.3.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActViolation {
    /// Rule (a): the checkpoint declares `declared` and the kernel applies `applied`.
    NotDeclared {
        /// What the checkpoint declares.
        declared: ActQuant,
        /// What the kernel applies.
        applied: ActQuant,
    },
    /// Rule (b) does not cover the 8-bit block, because the tensor has no ggml baseline
    /// (not GGUF, and not transcribable), and no certificate allows it.
    NotTranscribable(ActQuant),
    /// Rule (b) does not cover the 8-bit block, because ggml uses another for the type, or
    /// none, and no certificate allows it.
    WrongBlock {
        /// The weights' ggml type.
        ggml: GgmlType,
        /// What the kernel applies.
        applied: ActQuant,
    },
    /// Rule (c): FP8, MX FP8 or FP4 activations the checkpoint does not declare, and no
    /// certificate allows them.
    Uncertified(ActQuant),
    /// The census's transcription is not one RFC-0042 §3.5 lists for the weight scheme.
    BadTranscription {
        /// The weight scheme.
        scheme: QuantScheme,
        /// The claimed ggml type.
        ggml: GgmlType,
    },
}

impl fmt::Display for ActViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotDeclared { declared, applied } => write!(
                f,
                "the checkpoint declares activations {declared:?}; the kernel applies {applied:?}"
            ),
            Self::NotTranscribable(q) => write!(
                f,
                "{q:?} activations on a tensor with no exact ggml transcription, without a certificate"
            ),
            Self::WrongBlock { ggml, applied } => write!(
                f,
                "{applied:?} activations on {ggml:?} weights, which ggml does not quantise that way, without a certificate"
            ),
            Self::Uncertified(q) => write!(
                f,
                "undeclared {q:?} activations without a certificate in exact mode"
            ),
            Self::BadTranscription { scheme, ggml } => write!(
                f,
                "{scheme:?} has no exact transcription to {ggml:?} (RFC-0042 §3.5)"
            ),
        }
    }
}

impl std::error::Error for ActViolation {}

/// The ggml type the calibration baseline runs a weight of `scheme` as: the GGUF type itself,
/// or the census's `transcription` when RFC-0042 §3.5 allows it for the scheme.
///
/// §3.5's table: OCP MXFP4 with 32-element blocks to `MXFP4`; symmetric `PackedInt` of 4 bits
/// to `Q4_0` and of 8 bits to `Q8_0`, with a group size that is a multiple of 32. The census
/// checks the per-tensor condition on the scales; this checks the pairing.
///
/// # Errors
///
/// [`ActViolation::BadTranscription`] for a pairing §3.5 does not list.
pub fn baseline_type(
    scheme: QuantScheme,
    transcription: Option<GgmlType>,
) -> Result<Option<GgmlType>, ActViolation> {
    if let QuantScheme::Ggml(ty) = scheme {
        return Ok(Some(ty));
    }
    let Some(ggml) = transcription else {
        return Ok(None);
    };
    let listed = match scheme {
        QuantScheme::Mxfp4 { block: 32 } => ggml == GgmlType::MXFP4,
        QuantScheme::CompressedTensors(CtScheme::PackedInt {
            bits,
            group,
            symmetric: true,
            ..
        }) if group % 32 == 0 => {
            (bits == 4 && ggml == GgmlType::Q4_0) || (bits == 8 && ggml == GgmlType::Q8_0)
        }
        _ => false,
    };
    if listed {
        Ok(Some(ggml))
    } else {
        Err(ActViolation::BadTranscription { scheme, ggml })
    }
}

/// Holds `caps.numerics.act_in` to RFC-0042 §4.3 for `key` on the tensor `ctx` describes.
/// The weight scheme is `key.schemes[0]`.
///
/// # Errors
///
/// The [`ActViolation`] that rejects the candidate.
pub fn check_act(key: &OpKey, caps: &OpCaps, ctx: &ActContext) -> Result<ActBasis, ActViolation> {
    let applied = caps.numerics.act_in;
    if !rule(&key.op).matrix_product {
        return Ok(ActBasis::Unquantised);
    }
    // Rule (a): the declared quantisation is part of the reference, so a kernel must apply
    // it and nothing else. No certificate replaces it.
    if let Some(declared) = ctx.declared {
        return if applied == declared {
            Ok(ActBasis::Declared)
        } else {
            Err(ActViolation::NotDeclared { declared, applied })
        };
    }
    let violation = match applied {
        ActQuant::None => return Ok(ActBasis::Unquantised),
        ActQuant::Bf16 => return Ok(ActBasis::Bf16),
        ActQuant::Q8Block32 | ActQuant::Q8K256 => {
            let scheme = key.schemes.first().copied();
            let baseline = match scheme {
                Some(scheme) => baseline_type(scheme, ctx.transcription)?,
                None => None,
            };
            match baseline {
                Some(ggml) => {
                    if ggml_activation_block(ggml).map(|b| b.act_quant()) == Some(applied) {
                        return Ok(ActBasis::GgmlBlock(ggml));
                    }
                    ActViolation::WrongBlock { ggml, applied }
                }
                None => ActViolation::NotTranscribable(applied),
            }
        }
        // Undeclared FP8, MX FP8 and FP4 activations, and any variant added later.
        _ => ActViolation::Uncertified(applied),
    };
    if ctx.certified.contains(&caps.kernel_hash) {
        Ok(ActBasis::Certified)
    } else if ctx.act_lowp {
        Ok(ActBasis::ActLowp)
    } else {
        Err(violation)
    }
}

#[cfg(test)]
mod tests;
