// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the accumulation contract against RFC-0042 §4, §4.1 and §4.2 (COMP-E13-T05).

use smallvec::smallvec;
use syntropy_core::{AccumKind, ActQuant, DType, Numerics, QuantScheme};

use super::*;
use crate::op::tests::all_kinds;
use crate::op::{AttnKind, LayoutVersion, OpKey, OpKind, OpRegistry, RouteNorm, ScanKind};

fn key(op: OpKind) -> OpKey {
    OpKey {
        op,
        schemes: smallvec![QuantScheme::Plain(DType::F32); 3],
        layout: LayoutVersion::SOURCE,
        variant: 0,
    }
}

fn f32_decl() -> Numerics {
    Numerics {
        accumulate: AccumKind::F32,
        act_in: ActQuant::None,
        out: DType::F32,
        softmax_f32: true,
        state_f32: true,
    }
}

// ---- The rule table -----------------------------------------------------------------------

fn is_matrix(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::Gemv
            | OpKind::Gemm
            | OpKind::GateUpAct { .. }
            | OpKind::MoeGrouped
            | OpKind::Logits
    )
}

fn needs_softmax(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::AttnDecode(_)
            | OpKind::AttnPrefill(_)
            | OpKind::AttnPartial(_)
            | OpKind::AttnMerge
            | OpKind::MoeRoute {
                norm: RouteNorm::Softmax
            }
            | OpKind::TopKTopP
            | OpKind::SampleSeeded
            | OpKind::VerifyAccept
            | OpKind::TapGather
    )
}

fn needs_state(op: &OpKind) -> bool {
    matches!(op, OpKind::ScanDecode(_) | OpKind::ScanPrefill(_))
}

/// RFC-0042 §4.2 allows F16 output only for the model's KV dtype, and `Convert` outputs any
/// type.
fn allows_f16(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::KvWrite(_)
            | OpKind::GatherRows
            | OpKind::ScatterRows
            | OpKind::KvPack
            | OpKind::KvUnpack
            | OpKind::Convert
    )
}

/// §4.2: BF16 outputs only where the next op may take a BF16 input, plus the KV dtypes.
fn allows_bf16(op: &OpKind) -> bool {
    matches!(op, OpKind::Gemv | OpKind::Gemm | OpKind::GateUpAct { .. }) || allows_f16(op)
}

/// The table test: for every `OpKind`, the rule's flags are the ones RFC-0042 §4.2 names, its
/// minimal declaration passes, and each declaration one step weaker fails with the
/// violation that names the step.
#[test]
fn every_kind_accepts_its_minimal_declaration_and_rejects_weaker_ones() {
    let kinds = all_kinds();
    assert_eq!(kinds.len(), 85, "all_kinds lists every kind and parameter");
    for op in &kinds {
        let r = rule(op);
        assert_eq!(r.matrix_product, is_matrix(op), "{op:?} matrix_product");
        assert_eq!(r.softmax_f32, needs_softmax(op), "{op:?} softmax_f32");
        assert_eq!(r.state_f32, needs_state(op), "{op:?} state_f32");
        assert!(r.out.contains(&DType::F32), "{op:?} must allow F32 output");
        assert_eq!(
            r.out.contains(&DType::F16),
            allows_f16(op),
            "{op:?} F16 out"
        );
        assert_eq!(
            r.out.contains(&DType::BF16),
            allows_bf16(op),
            "{op:?} BF16 out"
        );

        let min = r.minimal();
        assert_eq!(check(op, &min), Ok(()), "{op:?} minimal {min:?}");

        // Stronger accumulators always pass.
        for accumulate in [AccumKind::F32, AccumKind::F64] {
            assert_eq!(check(op, &Numerics { accumulate, ..min }), Ok(()), "{op:?}");
        }
        // Every allowed output type passes.
        for &out in r.out {
            assert_eq!(
                check(op, &Numerics { out, ..min }),
                Ok(()),
                "{op:?} {out:?}"
            );
        }

        // I32 accumulation: within a block of 1 to 256 on a matrix product, never elsewhere.
        let i32_block = |max_block| Numerics {
            accumulate: AccumKind::I32ThenF32 { max_block },
            ..min
        };
        if r.matrix_product {
            assert_eq!(check(op, &i32_block(1)), Ok(()));
            assert_eq!(check(op, &i32_block(32)), Ok(()));
            assert_eq!(
                check(op, &i32_block(257)),
                Err(NumericsViolation::IntegerBlock { max_block: 257 })
            );
            assert_eq!(
                check(op, &i32_block(0)),
                Err(NumericsViolation::IntegerBlock { max_block: 0 })
            );
        } else {
            assert_eq!(
                check(op, &i32_block(32)),
                Err(NumericsViolation::IntegerAccumulation),
                "{op:?}"
            );
            let q = Numerics {
                act_in: ActQuant::Q8Block32,
                ..min
            };
            assert_eq!(
                check(op, &q),
                Err(NumericsViolation::ActivationQuant(ActQuant::Q8Block32)),
                "{op:?}"
            );
        }

        // Output types outside the rule fail.
        for out in DType::ALL {
            if !r.out.contains(&out) {
                assert_eq!(
                    check(op, &Numerics { out, ..min }),
                    Err(NumericsViolation::Output(out)),
                    "{op:?} {out:?}"
                );
            }
        }

        if r.softmax_f32 {
            let weak = Numerics {
                softmax_f32: false,
                ..min
            };
            assert_eq!(check(op, &weak), Err(NumericsViolation::SoftmaxNotF32));
        }
        if r.state_f32 {
            let weak = Numerics {
                state_f32: false,
                ..min
            };
            assert_eq!(check(op, &weak), Err(NumericsViolation::StateNotF32));
        }
    }
}

#[test]
fn named_examples_from_rfc_0042_section_4_2() {
    let gqa = OpKind::AttnDecode(AttnKind::Gqa);
    let no_softmax = Numerics {
        softmax_f32: false,
        ..f32_decl()
    };
    assert_eq!(
        check(&gqa, &no_softmax),
        Err(NumericsViolation::SoftmaxNotF32)
    );

    let scan = OpKind::ScanDecode(ScanKind::GatedDeltaNet);
    let no_state = Numerics {
        state_f32: false,
        ..f32_decl()
    };
    assert_eq!(check(&scan, &no_state), Err(NumericsViolation::StateNotF32));

    let f16_out = Numerics {
        out: DType::F16,
        ..f32_decl()
    };
    assert_eq!(
        check(&OpKind::Gemv, &f16_out),
        Err(NumericsViolation::Output(DType::F16))
    );
    assert_eq!(check(&OpKind::KvWrite(AttnKind::Gqa), &f16_out), Ok(()));
    let bf16_out = Numerics {
        out: DType::BF16,
        ..f32_decl()
    };
    assert_eq!(check(&OpKind::Gemm, &bf16_out), Ok(()));
    assert_eq!(
        check(&OpKind::Logits, &bf16_out),
        Err(NumericsViolation::Output(DType::BF16))
    );
    // Sigmoid routing has no softmax to keep in F32.
    let sigmoid = OpKind::MoeRoute {
        norm: RouteNorm::Sigmoid,
    };
    assert_eq!(check(&sigmoid, &no_softmax), Ok(()));
}

#[test]
fn check_caps_uses_the_declared_numerics() {
    let mut caps = OpCaps {
        workspace: syntropy_core::Bytes(0),
        est: None,
        layout: LayoutVersion::SOURCE,
        capture_safe: true,
        deterministic: true,
        numerics: f32_decl(),
        kernel_hash: syntropy_core::Blake3::default(),
        source_key: syntropy_core::Blake3::default(),
    };
    let k = key(OpKind::AttnPrefill(AttnKind::Gqa));
    assert_eq!(check_caps(&k, &caps), Ok(()));
    caps.numerics.softmax_f32 = false;
    assert_eq!(check_caps(&k, &caps), Err(NumericsViolation::SoftmaxNotF32));
}

#[test]
fn check_registry_reports_every_failing_and_undeclared_key() {
    let mut registry = OpRegistry::new();
    for op in all_kinds() {
        registry.register(key(op)).expect("unique keys");
    }
    assert_eq!(
        check_registry(&registry, |k| Some(rule(&k.op).minimal())),
        Ok(())
    );

    let undeclared = key(OpKind::RmsNorm);
    let weak = key(OpKind::ScanPrefill(ScanKind::Mamba2));
    let failures = check_registry(&registry, |k| {
        if *k == undeclared {
            None
        } else if *k == weak {
            Some(Numerics {
                state_f32: false,
                ..rule(&k.op).minimal()
            })
        } else {
            Some(rule(&k.op).minimal())
        }
    })
    .expect_err("two keys fail");
    let mut expected = vec![
        (undeclared, NumericsViolation::Undeclared),
        (weak, NumericsViolation::StateNotF32),
    ];
    expected.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(failures, expected);
}

#[test]
fn violation_display_names_the_problem() {
    assert_eq!(
        NumericsViolation::IntegerBlock { max_block: 512 }.to_string(),
        "I32 accumulation over 512 products; the block must be 1 to 256"
    );
    assert_eq!(
        NumericsViolation::Output(DType::F16).to_string(),
        "output type F16 not allowed for the op"
    );
    assert_eq!(
        NumericsViolation::Undeclared.to_string(),
        "no Numerics declaration"
    );
}

// ---- The dynamic detector -----------------------------------------------------------------

/// Rounds to the nearest FP16 value, ties to even, overflowing to infinity.
fn round_f16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let min_normal = 2f32.powi(-14);
    let r = if x.abs() < min_normal {
        // FP16 subnormals are multiples of 2^-24.
        let q = 2f32.powi(-24);
        (x / q).round_ties_even() * q
    } else {
        let bits = x.to_bits();
        let lsb = (bits >> 13) & 1;
        f32::from_bits((bits + 0x0fff + lsb) & !0x1fff)
    };
    if r.abs() > 65504.0 {
        f32::INFINITY.copysign(x)
    } else {
        r
    }
}

/// Rounds to the nearest BF16 value, ties to even.
fn round_bf16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits((bits + 0x7fff + lsb) & !0xffff)
}

fn flush(x: f32) -> f32 {
    if x.is_subnormal() { 0.0 } else { x }
}

/// The F32 dot product in index order: the stand-in for `kernels::reference`, which
/// COMP-E1-T03 builds, with the same F32 accumulation and no contraction.
fn f32_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |acc, (&x, &y)| acc + x * y)
}

/// The same in reverse order: reduction order is free within F32 (§4.1 rule 3).
fn f32_dot_reversed(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).rev().fold(0.0, |acc, (&x, &y)| acc + x * y)
}

/// Pairwise summation, as a tree-reducing GPU kernel would do it.
fn f32_dot_pairwise(a: &[f32], b: &[f32]) -> f32 {
    fn sum(v: &[f32]) -> f32 {
        if v.len() <= 1 {
            v.first().copied().unwrap_or(0.0)
        } else {
            let (l, r) = v.split_at(v.len() / 2);
            sum(l) + sum(r)
        }
    }
    let products: Vec<f32> = a.iter().zip(b).map(|(&x, &y)| x * y).collect();
    sum(&products)
}

/// The deliberately FP16-accumulating variant: F32 products, FP16 accumulator.
fn f16_accumulating_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0, |acc, (&x, &y)| round_f16(acc + x * y))
}

fn bf16_accumulating_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0, |acc, (&x, &y)| round_bf16(acc + x * y))
}

/// Treats subnormal inputs as zero and flushes subnormal results.
fn ftz_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0, |acc, (&x, &y)| flush(acc + flush(flush(x) * flush(y))))
}

/// Flushes subnormal results only, keeping subnormal inputs.
fn flush_results_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0, |acc, (&x, &y)| flush(acc + flush(x * y)))
}

#[test]
fn pow2_builds_normal_and_subnormal_powers() {
    assert_eq!(pow2(0), 1.0);
    assert_eq!(pow2(-12), 1.0 / 4096.0);
    assert_eq!(pow2(-126), f32::MIN_POSITIVE);
    assert_eq!(pow2(127), 2f32.powi(127));
    assert_eq!(pow2(-130).to_bits(), 1 << 19);
    assert_eq!(pow2(-149).to_bits(), 1);
    assert!(pow2(-130).is_subnormal());
}

#[test]
fn rounding_helpers_match_known_values() {
    assert_eq!(round_f16(1.0 + 2f32.powi(-12)), 1.0);
    assert_eq!(round_f16(1.0 + 2f32.powi(-10)), 1.0 + 2f32.powi(-10));
    assert_eq!(round_f16(2049.0), 2048.0);
    assert_eq!(round_f16(2051.0), 2052.0);
    assert_eq!(round_f16(65504.0), 65504.0);
    assert_eq!(round_f16(65520.0), f32::INFINITY);
    assert_eq!(round_f16(2f32.powi(-24)), 2f32.powi(-24));
    assert_eq!(round_f16(2f32.powi(-26)), 0.0);
    assert_eq!(round_bf16(1.0 + 2f32.powi(-9)), 1.0);
    assert_eq!(round_bf16(1.0 + 2f32.powi(-7)), 1.0 + 2f32.powi(-7));
}

#[test]
fn every_case_has_an_exact_f32_result_and_matching_lengths() {
    let cases = detector_cases();
    assert_eq!(cases.len(), 5);
    for case in &cases {
        assert_eq!(case.a.len(), case.b.len(), "{}", case.name);
        let exact: f64 = case
            .a
            .iter()
            .zip(&case.b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        assert_eq!(
            f64::from(case.expected),
            exact,
            "{} exact in F32",
            case.name
        );
        // Every F32 order gets it bit for bit.
        assert_eq!(f32_dot(&case.a, &case.b), case.expected, "{}", case.name);
        assert_eq!(
            f32_dot_reversed(&case.a, &case.b),
            case.expected,
            "{}",
            case.name
        );
        assert_eq!(
            f32_dot_pairwise(&case.a, &case.b),
            case.expected,
            "{}",
            case.name
        );
    }
    assert!(cases[3].a[0].is_subnormal(), "subnormal inputs");
    assert!(
        (cases[4].a[0] * cases[4].b[0]).is_subnormal(),
        "subnormal products"
    );
}

#[test]
fn detector_passes_f32_accumulation_in_any_order() {
    assert_eq!(detect_dot(f32_dot), Ok(()));
    assert_eq!(detect_dot(f32_dot_reversed), Ok(()));
    assert_eq!(detect_dot(f32_dot_pairwise), Ok(()));
    let f64_dot = |a: &[f32], b: &[f32]| {
        let s: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        s as f32
    };
    assert_eq!(detect_dot(f64_dot), Ok(()));
}

/// Criterion 1: a deliberately FP16-accumulating variant is rejected. It cannot declare FP16
/// accumulation (the `compile_fail` examples in the module docs), so it declares F32, which
/// the static check accepts, and the detector rejects it.
#[test]
fn detector_rejects_an_f16_accumulating_variant_that_declares_f32() {
    let declared = rule(&OpKind::Gemv).minimal();
    let declared = Numerics {
        accumulate: AccumKind::F32,
        ..declared
    };
    assert_eq!(check(&OpKind::Gemv, &declared), Ok(()));

    let err = detect_dot(f16_accumulating_dot).expect_err("FP16 accumulation is caught");
    assert_eq!(err.case, "f16_overflow");
    assert_eq!(err.got, f32::INFINITY);
    assert_eq!(err.expected, 4_194_304.0);
    assert_eq!(
        err.to_string(),
        "accumulation detector case f16_overflow: expected 4.194304e6, kernel returned inf"
    );

    // Each FP16 case catches it on its own, not only the first.
    for case in detector_cases()
        .iter()
        .filter(|c| c.name.starts_with("f16_"))
    {
        let got = f16_accumulating_dot(&case.a, &case.b);
        let rel = ((got - case.expected) / case.expected).abs();
        assert!(rel.is_nan() || rel >= 0.5, "{}: {got}", case.name);
    }
}

#[test]
fn detector_rejects_bf16_accumulation() {
    let err = detect_dot(bf16_accumulating_dot).expect_err("BF16 accumulation is caught");
    assert!(err.case.starts_with("f16_"), "{err}");
    let swamping = &detector_cases()[1];
    assert_eq!(swamping.name, "f16_swamping");
    assert_eq!(bf16_accumulating_dot(&swamping.a, &swamping.b), 1.0);
}

#[test]
fn detector_rejects_flush_to_zero() {
    for variant in [ftz_dot as fn(&[f32], &[f32]) -> f32, flush_results_dot] {
        let err = detect_dot(variant).expect_err("flushing is caught");
        assert_eq!(err.case, "f32_subnormal_inputs");
        assert_eq!(err.got, 0.0);
        // Each subnormal case catches it on its own.
        for case in detector_cases()
            .iter()
            .filter(|c| c.name.starts_with("f32_"))
        {
            assert_eq!(variant(&case.a, &case.b), 0.0, "{}", case.name);
        }
    }
}

#[test]
fn detector_rejects_nan() {
    let err = detect_dot(|_, _| f32::NAN).expect_err("NaN fails");
    assert!(err.got.is_nan());
}
