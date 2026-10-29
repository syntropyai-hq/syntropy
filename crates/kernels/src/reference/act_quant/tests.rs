// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the activation quantisation references that need no ggml output: the conversions
//! against their definitions, the layouts, the inputs ggml leaves undefined, and the
//! fake-quant step (COMP-E13-T03). `tests/ggml_act_quant.rs` checks them against ggml.

use proptest::prelude::*;
use syntropy_core::{ActQuant, GgmlType};

use super::*;

// ---- F16, BF16 and E4M3 ------------------------------------------------------------------

#[test]
fn every_f16_value_round_trips() {
    for h in 0..=u16::MAX {
        let x = f16_to_f32(h);
        if x.is_nan() {
            assert_eq!(f16_bits(x), (h & 0x8000) | 0x7E00, "NaN {h:#06x}");
        } else {
            assert_eq!(f16_bits(x), h, "{h:#06x} = {x:e}");
        }
    }
}

#[test]
fn f16_decoding_matches_its_definition() {
    assert_eq!(f16_to_f32(0x3C00), 1.0);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);
    assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2f32.powi(-24));
    assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
    assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
}

#[test]
fn f16_rounding_boundaries() {
    // 65520 is the tie above 65504; it goes to even, infinity. Just below stays finite.
    assert_eq!(f16_bits(65520.0), 0x7C00);
    assert_eq!(f16_bits(f32::from_bits(65520f32.to_bits() - 1)), 0x7BFF);
    // Half of the smallest subnormal is a tie to zero; just above rounds up.
    assert_eq!(f16_bits(2f32.powi(-25)), 0x0000);
    assert_eq!(
        f16_bits(f32::from_bits(2f32.powi(-25).to_bits() + 1)),
        0x0001
    );
    // 1.5 subnormal units is a tie between 1 and 2: to 2. 2.5 units: to 2.
    assert_eq!(f16_bits(1.5 * 2f32.powi(-24)), 0x0002);
    assert_eq!(f16_bits(2.5 * 2f32.powi(-24)), 0x0002);
    // An F32 subnormal is zero in F16, with its sign.
    assert_eq!(f16_bits(f32::from_bits(0x8000_0001)), 0x8000);
    // The carry out of the largest subnormal reaches the smallest normal.
    assert_eq!(f16_bits(1023.5 * 2f32.powi(-24)), 0x0400);
}

/// The F16 value nearest to `x`, ties to even, by search in F64: the definition `f16_bits`
/// must meet for finite `x` below 65520.
fn nearest_f16(x: f32) -> u16 {
    let target = f64::from(x.abs());
    // Finite non-negative F16 values are ordered like their bits: find the largest at or
    // below the target.
    let (mut lo, mut hi) = (0u16, 0x7BFF);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if f64::from(f16_to_f32(mid)) <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let (below, above) = (lo, (lo + 1).min(0x7BFF));
    let db = target - f64::from(f16_to_f32(below));
    let da = f64::from(f16_to_f32(above)) - target;
    let pick = if above == below || db < da || (db == da && below & 1 == 0) {
        below
    } else {
        above
    };
    pick | if x.is_sign_negative() { 0x8000 } else { 0 }
}

proptest! {
    #[test]
    fn f16_rounds_to_nearest_even(bits in 0u32..0x477F_F000, neg in any::<bool>()) {
        let x = f32::from_bits(bits | if neg { 0x8000_0000 } else { 0 });
        prop_assert_eq!(f16_bits(x), nearest_f16(x));
    }

    #[test]
    fn bf16_rounds_to_nearest_even(bits in any::<u32>()) {
        let x = f32::from_bits(bits);
        prop_assume!(!x.is_nan());
        let h = round_bf16(x);
        let back = bf16_to_f32(h);
        if back.is_finite() {
            // Within half a BF16 unit, and on a tie the result is even.
            let lo = f32::from_bits(bits & 0xFFFF_0000);
            let hi = f32::from_bits((bits & 0xFFFF_0000) + 0x1_0000);
            let (dl, dh) = (f64::from(x) - f64::from(lo), f64::from(hi) - f64::from(x));
            let want = if dl.abs() < dh.abs() || (dl.abs() == dh.abs() && (bits >> 16) & 1 == 0) {
                lo
            } else {
                hi
            };
            prop_assert_eq!(back.to_bits(), want.to_bits());
        }
    }
}

#[test]
fn bf16_keeps_nan_quiet_and_subnormals() {
    assert_eq!(round_bf16(f32::from_bits(0x7F80_0001)), 0x7FC0);
    assert_eq!(round_bf16(f32::from_bits(0xFFC0_0000)), 0xFFC0);
    assert_eq!(round_bf16(f32::INFINITY), 0x7F80);
    // F32's largest finite value is above BF16's largest by more than half a unit.
    assert_eq!(round_bf16(f32::MAX), 0x7F80);
    // A subnormal is rounded, not flushed.
    assert_eq!(round_bf16(f32::from_bits(0x0001_8000)), 0x0002);
    for h in 0..=u16::MAX {
        let x = bf16_to_f32(h);
        if !x.is_nan() {
            assert_eq!(round_bf16(x), h);
        }
    }
}

/// Every E4M3 (FN) code's value: bias 7, subnormal unit 2^-9, 0x7F and 0xFF NaN.
fn e4m3_value(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((code >> 3) & 0x0F);
    let mant = f32::from(code & 0x07);
    if code & 0x7F == 0x7F {
        return f32::NAN;
    }
    if exp == 0 {
        sign * mant * 2f32.powi(-9)
    } else {
        sign * (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
    }
}

#[test]
fn e4m3_rounding_keeps_every_code_and_ties_to_even() {
    let mut values: Vec<f32> = (0u8..=0x7E).map(e4m3_value).collect();
    for &v in &values {
        assert_eq!(round_e4m3(v), v);
        assert_eq!(round_e4m3(-v).to_bits(), (-v).to_bits());
    }
    assert_eq!(*values.last().expect("codes"), E4M3_MAX);
    values.dedup();
    for pair in values.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let mid = (a + b) / 2.0;
        // The code of `a` is even exactly when its index is; the tie goes to the even one.
        let a_even = values.iter().position(|&v| v == a).expect("listed") % 2 == 0;
        assert_eq!(round_e4m3(mid), if a_even { a } else { b }, "tie {a} {b}");
        assert_eq!(round_e4m3(f32::from_bits(mid.to_bits() - 1)), a);
        assert_eq!(round_e4m3(f32::from_bits(mid.to_bits() + 1)), b);
    }
    // 464 is the tie between 448 and the next step, 480, which E4M3 FN does not have: 448
    // is even and wins. Above it the cast gives NaN.
    assert_eq!(round_e4m3(464.0), 448.0);
    assert!(round_e4m3(464.5).is_nan());
    assert!(round_e4m3(f32::NAN).is_nan());
}

// ---- Layouts and the activation block per weight type ------------------------------------

#[test]
fn block_sizes_are_ggmls() {
    // ggml's static asserts: sizeof(block_q8_0) = 2 + 32, block_q8_1 = 4 + 32, block_q8_K =
    // 4 + 256 + 16 · 2.
    assert_eq!(BlockQ8_0::BYTES, 34);
    assert_eq!(BlockQ8_1::BYTES, 36);
    assert_eq!(BlockQ8K::BYTES, 292);
}

#[test]
fn each_weight_type_has_ggmls_activation_block() {
    use GgmlQ8Block::{Q8_0, Q8_1, Q8K};
    // The `vec_dot_type` column of ggml-cpu.c's type traits at the pinned commit, in
    // `GgmlType::ALL` order.
    let expected = [
        (GgmlType::F32, None),
        (GgmlType::F16, None),
        (GgmlType::BF16, None),
        (GgmlType::Q8_0, Some(Q8_0)),
        (GgmlType::Q4_0, Some(Q8_0)),
        (GgmlType::Q4_1, Some(Q8_1)),
        (GgmlType::Q5_0, Some(Q8_0)),
        (GgmlType::Q5_1, Some(Q8_1)),
        (GgmlType::Q2_K, Some(Q8K)),
        (GgmlType::Q3_K, Some(Q8K)),
        (GgmlType::Q4_K, Some(Q8K)),
        (GgmlType::Q5_K, Some(Q8K)),
        (GgmlType::Q6_K, Some(Q8K)),
        (GgmlType::IQ4_NL, Some(Q8_0)),
        (GgmlType::IQ4_XS, Some(Q8K)),
        (GgmlType::IQ3_XXS, Some(Q8K)),
        (GgmlType::IQ3_S, Some(Q8K)),
        (GgmlType::IQ2_XXS, Some(Q8K)),
        (GgmlType::IQ2_XS, Some(Q8K)),
        (GgmlType::IQ2_S, Some(Q8K)),
        (GgmlType::IQ1_S, Some(Q8K)),
        (GgmlType::IQ1_M, Some(Q8K)),
        (GgmlType::TQ1_0, Some(Q8K)),
        (GgmlType::TQ2_0, Some(Q8K)),
        (GgmlType::Q2_0, Some(Q8_0)),
        (GgmlType::Q1_0, Some(Q8_0)),
        (GgmlType::MXFP4, Some(Q8_0)),
        (GgmlType::NVFP4, Some(Q8_0)),
    ];
    assert_eq!(
        expected.len(),
        GgmlType::ALL.len(),
        "a new GgmlType needs a row"
    );
    for ((ty, block), listed) in expected.into_iter().zip(GgmlType::ALL) {
        assert_eq!(ty, listed);
        assert_eq!(ggml_activation_block(ty), block, "{ty:?}");
    }
    assert_eq!(Q8_0.act_quant(), ActQuant::Q8Block32);
    assert_eq!(Q8_1.act_quant(), ActQuant::Q8Block32);
    assert_eq!(Q8K.act_quant(), ActQuant::Q8K256);
}

// ---- The quantizers on inputs the vectors do not cover -------------------------------------

#[test]
fn a_block_with_a_non_finite_element_is_nan() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut x = vec![0.5f32; 256];
        x[40] = bad;
        let b0 = quantize_q8_0(&x, Q8Path::Reference).expect("whole blocks");
        assert_eq!(b0[1].d, 0x7E00);
        assert_eq!(b0[1].qs, [0; QK8]);
        assert!(b0[1].dequantize().iter().all(|v| v.is_nan()));
        // The other blocks are untouched.
        assert_ne!(b0[0].d, 0x7E00);
        let b1 = quantize_q8_1(&x, Q8Path::Arm64Neon).expect("whole blocks");
        assert_eq!((b1[1].d, b1[1].s), (0x7E00, 0x7E00));
        let k = quantize_q8_k(&x).expect("whole blocks");
        assert!(k[0].d.is_nan());
        assert!(k[0].dequantize().iter().all(|v| v.is_nan()));
    }
}

#[test]
fn an_overflowing_inverse_scale_gives_zeros() {
    // amax = 1e-38: d = amax/127 is below 1/F32::MAX, so 1/d overflows.
    for path in [Q8Path::Reference, Q8Path::Arm64Neon, Q8Path::X86Avx] {
        let mut x = vec![0.0f32; 32];
        x[3] = 1.0e-38;
        x[4] = -0.5e-38;
        let b = quantize_q8_0(&x, path).expect("one block");
        assert_eq!(b[0].d, 0, "{path:?}");
        assert_eq!(b[0].qs, [0; QK8]);
        assert!(b[0].dequantize().iter().all(|&v| v == 0.0));
    }
    let mut x = vec![0.0f32; 256];
    x[3] = -1.0e-38;
    let k = quantize_q8_k(&x).expect("one block");
    // ggml's d = 1/iscale with iscale = −127/max = +infinity.
    assert_eq!(k[0].d.to_bits(), 0.0f32.to_bits());
    assert_eq!(k[0].qs, [0; QK_K]);
}

#[test]
fn a_ragged_row_is_an_error() {
    assert_eq!(
        quantize_q8_0(&[0.0; 33], Q8Path::Reference).err(),
        Some(RowLength { len: 33, block: 32 })
    );
    assert_eq!(
        quantize_q8_k(&[0.0; 32]).err(),
        Some(RowLength {
            len: 32,
            block: 256
        })
    );
}

/// From the source: the AVX path differs from NEON only through `127/amax` against
/// `1/(amax/127)`. Where the two inverse scales are equal, so are the blocks. No x86 vectors
/// check the AVX path against ggml yet (`tests/data/act_quant/README.md`).
#[test]
fn the_avx_path_is_the_neon_path_when_the_inverse_scales_agree() {
    let mut rng = 0x1234_5678_9ABC_DEF0u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let (mut same, mut differ) = (0, 0);
    for scale in [1.0e-3f32, 0.7, 3.0, 1.0e4] {
        for _ in 0..200 {
            let x: Vec<f32> = (0..32).map(|_| next() * scale).collect();
            let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let neon = quantize_q8_0(&x, Q8Path::Arm64Neon).expect("one block");
            let avx = quantize_q8_0(&x, Q8Path::X86Avx).expect("one block");
            if 127.0 / amax == 1.0 / (amax / 127.0) {
                assert_eq!(neon, avx);
                same += 1;
            } else {
                differ += 1;
            }
        }
    }
    assert!(same > 0 && differ > 0, "same {same}, differ {differ}");
}

/// From the source: the AVX path multiplies by `127/amax`, the NEON path by `1/(amax/127)`.
/// Here the two inverse scales differ by one ulp, and that moves the second element across a
/// rounding boundary: 1.4999999 on the AVX path, 1.5 on the NEON path.
#[test]
fn the_avx_path_multiplies_by_127_over_amax() {
    let amax = f32::from_bits(0x3F80_000F);
    let v = f32::from_bits(0x3C41_831C);
    assert_ne!(127.0 / amax, 1.0 / (amax / 127.0));
    let mut x = [0.0f32; 32];
    x[0] = amax;
    x[1] = v;
    let avx = &quantize_q8_0(&x, Q8Path::X86Avx).expect("one block")[0];
    let neon = &quantize_q8_0(&x, Q8Path::Arm64Neon).expect("one block")[0];
    assert_eq!(avx.d, neon.d);
    assert_eq!((avx.qs[0], avx.qs[1]), (127, 1));
    assert_eq!((neon.qs[0], neon.qs[1]), (127, 2));
}

// ---- The fake-quant step -------------------------------------------------------------------

#[test]
fn fake_quant_gives_the_dequantised_values() {
    let x: Vec<f32> = (0..256).map(|i| (i as f32 - 100.0) / 7.0).collect();
    assert_eq!(
        fake_quant(ActQuant::None, &x, Q8Path::Reference).expect("none"),
        x
    );
    let bf = fake_quant(ActQuant::Bf16, &x, Q8Path::Reference).expect("bf16");
    for (a, b) in x.iter().zip(&bf) {
        assert_eq!(*b, bf16_to_f32(round_bf16(*a)));
    }
    let q = fake_quant(ActQuant::Q8Block32, &x, Q8Path::Reference).expect("q8_0");
    let blocks = quantize_q8_0(&x, Q8Path::Reference).expect("q8_0");
    let want: Vec<f32> = blocks.iter().flat_map(BlockQ8_0::dequantize).collect();
    assert_eq!(q, want);
    let k = fake_quant(ActQuant::Q8K256, &x, Q8Path::Reference).expect("q8_k");
    assert_eq!(k, quantize_q8_k(&x).expect("q8_k")[0].dequantize().to_vec());
    // Each Q8 value is within half a step of its input, plus what rounding d to F16 (a
    // relative 2^-11) moves 127 steps by.
    for (b, (xs, qs)) in blocks.iter().zip(x.chunks(32).zip(q.chunks(32))) {
        let d = f16_to_f32(b.d);
        for (a, v) in xs.iter().zip(qs) {
            assert!((a - v).abs() <= d * (0.5 + 127.0 / 2048.0), "{a} {v} {d}");
        }
    }
}

#[test]
fn fp8_dynamic_fake_quant_scales_each_group() {
    let mut x: Vec<f32> = (0..256)
        .map(|i| ((i * 37) % 101) as f32 / 13.0 - 3.5)
        .collect();
    x[200..].fill(0.0);
    let q =
        fake_quant(ActQuant::Fp8Dynamic { group: 128 }, &x, Q8Path::Reference).expect("two groups");
    for (xs, qs) in x.chunks(128).zip(q.chunks(128)) {
        let amax = xs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let s = amax / E4M3_MAX;
        for (a, v) in xs.iter().zip(qs) {
            assert_eq!(*v, round_e4m3(a / s) * s);
            // E4M3 keeps 3 mantissa bits: a relative step of at most 2^-3 at normal values.
            assert!((a - v).abs() <= a.abs() / 16.0 + s * 2f32.powi(-9));
        }
        // The group's maximum is exact: it maps to 448.
        let at = xs.iter().position(|v| v.abs() == amax).expect("max");
        assert_eq!(qs[at].abs(), round_e4m3(E4M3_MAX) * s);
    }
    let zeros = fake_quant(
        ActQuant::Fp8Dynamic { group: 32 },
        &[0.0; 64],
        Q8Path::Reference,
    )
    .expect("zeros");
    assert!(zeros.iter().all(|&v| v == 0.0));
}

#[test]
fn fake_quant_reports_what_it_cannot_do() {
    let x = [0.0f32; 64];
    for act in [
        ActQuant::Mxfp8,
        ActQuant::Nvfp4,
        ActQuant::Fp8Dynamic { group: 0 },
    ] {
        assert_eq!(
            fake_quant(act, &x, Q8Path::Reference),
            Err(FakeQuantError::Unsupported(act))
        );
    }
    assert_eq!(
        fake_quant(ActQuant::Q8K256, &x, Q8Path::Reference),
        Err(FakeQuantError::Length(RowLength {
            len: 64,
            block: 256
        }))
    );
    assert_eq!(
        fake_quant(ActQuant::Fp8Dynamic { group: 48 }, &x, Q8Path::Reference),
        Err(FakeQuantError::Length(RowLength { len: 64, block: 48 }))
    );
}
