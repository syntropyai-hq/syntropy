// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the activation quantisation rule against RFC-0042 §4.3 rules a–c and §3.5
//! (COMP-E13-T03 criteria 2 and 3).

use smallvec::smallvec;
use syntropy_core::{Bytes, DType, Numerics, ScaleGranularity};

use super::*;
use crate::op::tests::all_kinds;
use crate::op::{LayoutVersion, OpKind};

const FP8_BLOCK: QuantScheme = QuantScheme::Fp8 {
    format: DType::F8E4M3,
    scale: ScaleGranularity::Block {
        rows: 128,
        cols: 128,
    },
};

const FP8_DYNAMIC: ActQuant = ActQuant::Fp8Dynamic { group: 128 };

fn packed_int(bits: u8, group: u16, symmetric: bool) -> QuantScheme {
    QuantScheme::CompressedTensors(CtScheme::PackedInt {
        bits,
        group,
        symmetric,
        zero_point: !symmetric,
    })
}

fn key(op: OpKind, weights: QuantScheme) -> OpKey {
    OpKey {
        op,
        schemes: smallvec![
            weights,
            QuantScheme::Plain(DType::F32),
            QuantScheme::Plain(DType::F32)
        ],
        layout: LayoutVersion::SOURCE,
        variant: 0,
    }
}

fn gemm(weights: QuantScheme) -> OpKey {
    key(OpKind::Gemm, weights)
}

/// A kernel that applies `act_in`, with `hash` as its `kernel_hash`.
fn kernel(key: &OpKey, act_in: ActQuant, hash: u8) -> OpCaps {
    OpCaps {
        workspace: Bytes(0),
        est: None,
        layout: LayoutVersion::SOURCE,
        capture_safe: true,
        deterministic: true,
        numerics: Numerics {
            act_in,
            ..rule(&key.op).minimal()
        },
        kernel_hash: Blake3([hash; 32]),
        source_key: Blake3([0; 32]),
    }
}

fn exact() -> ActContext {
    ActContext::default()
}

fn certifying(hashes: &[u8]) -> ActContext {
    ActContext {
        certified: hashes.iter().map(|&h| Blake3([h; 32])).collect(),
        ..ActContext::default()
    }
}

// ---- Criterion 2: undeclared FP8 activations ---------------------------------------------

#[test]
fn undeclared_fp8_activations_are_rejected_in_exact_mode() {
    let k = gemm(FP8_BLOCK);
    for act in [FP8_DYNAMIC, ActQuant::Mxfp8, ActQuant::Nvfp4] {
        let w8a8 = kernel(&k, act, 7);
        assert_eq!(
            check_act(&k, &w8a8, &exact()),
            Err(ActViolation::Uncertified(act))
        );
        // A certificate for another kernel does not help.
        assert_eq!(
            check_act(&k, &w8a8, &certifying(&[8])),
            Err(ActViolation::Uncertified(act))
        );
    }
}

#[test]
fn a_certificate_allows_undeclared_fp8_activations() {
    let k = gemm(FP8_BLOCK);
    let w8a8 = kernel(&k, FP8_DYNAMIC, 7);
    assert_eq!(
        check_act(&k, &w8a8, &certifying(&[3, 7])),
        Ok(ActBasis::Certified)
    );
}

#[test]
fn approximate_mode_with_act_lowp_admits_them_and_says_so() {
    let k = gemm(FP8_BLOCK);
    let w8a8 = kernel(&k, ActQuant::Nvfp4, 7);
    let lowp = ActContext {
        act_lowp: true,
        ..ActContext::default()
    };
    assert_eq!(check_act(&k, &w8a8, &lowp), Ok(ActBasis::ActLowp));
}

#[test]
fn widening_and_bf16_rounding_need_no_certificate() {
    let k = gemm(FP8_BLOCK);
    assert_eq!(
        check_act(&k, &kernel(&k, ActQuant::None, 1), &exact()),
        Ok(ActBasis::Unquantised)
    );
    assert_eq!(
        check_act(&k, &kernel(&k, ActQuant::Bf16, 1), &exact()),
        Ok(ActBasis::Bf16)
    );
}

// ---- Rule (a): declared by the checkpoint -------------------------------------------------

#[test]
fn a_declared_scheme_must_be_applied_exactly() {
    let k = gemm(FP8_BLOCK);
    let declared = ActContext {
        declared: Some(FP8_DYNAMIC),
        ..ActContext::default()
    };
    assert_eq!(
        check_act(&k, &kernel(&k, FP8_DYNAMIC, 1), &declared),
        Ok(ActBasis::Declared)
    );
    // W8A16 skips the declared quantisation; another group size or scheme is another model.
    for applied in [
        ActQuant::None,
        ActQuant::Bf16,
        ActQuant::Fp8Dynamic { group: 64 },
        ActQuant::Mxfp8,
    ] {
        let want = Err(ActViolation::NotDeclared {
            declared: FP8_DYNAMIC,
            applied,
        });
        assert_eq!(check_act(&k, &kernel(&k, applied, 1), &declared), want);
        // No certificate replaces the declared step, and neither does `act_lowp`.
        let anyway = ActContext {
            certified: vec![Blake3([1; 32])],
            act_lowp: true,
            ..declared.clone()
        };
        assert_eq!(check_act(&k, &kernel(&k, applied, 1), &anyway), want);
    }
}

// ---- Criterion 3 and rule (b): 8-bit blocks and transcription -----------------------------

#[test]
fn eight_bit_activations_on_a_non_transcribable_packed_int_need_a_path_cert() {
    // Asymmetric INT4 group 32: never transcribable (RFC-0042 §3.5).
    let asym = gemm(packed_int(4, 32, false));
    let q8 = kernel(&asym, ActQuant::Q8Block32, 9);
    assert_eq!(
        check_act(&asym, &q8, &exact()),
        Err(ActViolation::NotTranscribable(ActQuant::Q8Block32))
    );
    assert_eq!(
        check_act(&asym, &q8, &certifying(&[9])),
        Ok(ActBasis::Certified)
    );
    // Symmetric INT4 whose scales the census found do not fit F16: not transcribable either.
    let sym = gemm(packed_int(4, 32, true));
    let q8 = kernel(&sym, ActQuant::Q8Block32, 9);
    assert_eq!(
        check_act(&sym, &q8, &exact()),
        Err(ActViolation::NotTranscribable(ActQuant::Q8Block32))
    );
    // The default for such tensors, widening with F32 activations, needs nothing.
    assert_eq!(
        check_act(&asym, &kernel(&asym, ActQuant::None, 9), &exact()),
        Ok(ActBasis::Unquantised)
    );
}

#[test]
fn a_transcribable_tensor_takes_its_ggml_types_block() {
    let transcribed = |ggml| ActContext {
        transcription: Some(ggml),
        ..ActContext::default()
    };
    for (scheme, ggml) in [
        (packed_int(4, 32, true), GgmlType::Q4_0),
        (packed_int(4, 128, true), GgmlType::Q4_0),
        (packed_int(8, 64, true), GgmlType::Q8_0),
        (QuantScheme::Mxfp4 { block: 32 }, GgmlType::MXFP4),
    ] {
        let k = gemm(scheme);
        assert_eq!(
            check_act(&k, &kernel(&k, ActQuant::Q8Block32, 1), &transcribed(ggml)),
            Ok(ActBasis::GgmlBlock(ggml)),
            "{scheme:?}"
        );
        // ggml uses Q8_0 for every one of these, so Q8_K blocks are not ggml-equivalent.
        assert_eq!(
            check_act(&k, &kernel(&k, ActQuant::Q8K256, 1), &transcribed(ggml)),
            Err(ActViolation::WrongBlock {
                ggml,
                applied: ActQuant::Q8K256
            })
        );
    }
}

#[test]
fn a_transcription_section_3_5_does_not_list_is_refused() {
    for (scheme, ggml) in [
        (packed_int(4, 32, false), GgmlType::Q4_0),
        (packed_int(4, 16, true), GgmlType::Q4_0),
        (packed_int(4, 32, true), GgmlType::Q8_0),
        (packed_int(8, 32, true), GgmlType::Q4_0),
        (FP8_BLOCK, GgmlType::Q8_0),
        (QuantScheme::Mxfp4 { block: 32 }, GgmlType::Q4_0),
        (
            QuantScheme::Nvfp4 {
                block: 16,
                scale: DType::F8E4M3,
            },
            GgmlType::NVFP4,
        ),
        (QuantScheme::Plain(DType::BF16), GgmlType::BF16),
    ] {
        let k = gemm(scheme);
        let ctx = ActContext {
            transcription: Some(ggml),
            ..ActContext::default()
        };
        assert_eq!(
            check_act(&k, &kernel(&k, ActQuant::Q8Block32, 1), &ctx),
            Err(ActViolation::BadTranscription { scheme, ggml }),
            "{scheme:?} as {ggml:?}"
        );
    }
}

#[test]
fn gguf_weights_take_the_block_ggml_uses_for_their_type() {
    for ggml in GgmlType::ALL {
        let k = gemm(QuantScheme::Ggml(ggml));
        let want = ggml_activation_block(ggml).map(|b| b.act_quant());
        for applied in [ActQuant::Q8Block32, ActQuant::Q8K256] {
            let got = check_act(&k, &kernel(&k, applied, 1), &exact());
            if want == Some(applied) {
                assert_eq!(got, Ok(ActBasis::GgmlBlock(ggml)), "{ggml:?}");
            } else {
                assert_eq!(
                    got,
                    Err(ActViolation::WrongBlock { ggml, applied }),
                    "{ggml:?} {applied:?}"
                );
            }
        }
        // A census transcription is ignored for a GGUF tensor: its type is its own.
        let ctx = ActContext {
            transcription: Some(GgmlType::Q4_0),
            ..ActContext::default()
        };
        assert!(check_act(&k, &kernel(&k, ActQuant::Bf16, 1), &ctx).is_ok());
    }
    // Plain safetensors weights have no ggml baseline for 8-bit blocks.
    let plain = gemm(QuantScheme::Plain(DType::BF16));
    assert_eq!(
        check_act(&plain, &kernel(&plain, ActQuant::Q8Block32, 1), &exact()),
        Err(ActViolation::NotTranscribable(ActQuant::Q8Block32))
    );
}

#[test]
fn every_matrix_product_is_held_to_the_rule() {
    let mut matrix = 0;
    for op in all_kinds() {
        let k = key(op, FP8_BLOCK);
        let mut c = kernel(&k, FP8_DYNAMIC, 1);
        if rule(&op).matrix_product {
            matrix += 1;
            assert_eq!(
                check_act(&k, &c, &exact()),
                Err(ActViolation::Uncertified(FP8_DYNAMIC)),
                "{op:?}"
            );
        } else {
            // `check` rejects quantisation outside a matrix product; this rule leaves it.
            c.numerics.act_in = ActQuant::None;
            assert_eq!(check_act(&k, &c, &exact()), Ok(ActBasis::Unquantised));
        }
    }
    let listed = all_kinds()
        .iter()
        .filter(|op| {
            matches!(
                op,
                OpKind::Gemv
                    | OpKind::Gemm
                    | OpKind::GateUpAct { .. }
                    | OpKind::MoeGrouped
                    | OpKind::Logits
            )
        })
        .count();
    assert_eq!(matrix, listed);
}
