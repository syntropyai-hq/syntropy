// SPDX-License-Identifier: MIT OR Apache-2.0

//! The COMP-E3-T22 ops against `kernels::reference`, at every level this host runs.
//!
//! - `Gemm` (and `Logits`, the same kernel): prefill row counts above 16 give, at every level,
//!   the portable kernels' bits, the bits of the same rows computed 8 at a time as `Gemv`, and
//!   values within the GEMV module's derived bound of `reference::gemm`.
//! - `GateUpAct`: every level and every chunking gives the same bits; the output is exactly
//!   `silu(g) · u` with `g` and `u` the level's `Gemv` outputs; those are within the derived
//!   bound of `reference::gemv`; and `reference::gate_up_act` is exactly the same composition of
//!   reference products, so the two differ only through `g` and `u`.
//! - `RowGather`, `Add`, `Mul` and `Convert`: bit for bit equal to the reference at every
//!   level, on random values and on NaN payloads, infinities, signed zeros, subnormals and
//!   rounding ties; refusals write nothing.

use proptest::prelude::*;
use syntropy_core::{DType, GgmlType};
use syntropy_kernels::GatedAct;
use syntropy_kernels::reference::act_quant::{f16_bits, round_bf16};
use syntropy_kernels::reference::math::silu;
use syntropy_kernels::reference::{
    Matrix, Operand, add as ref_add, convert as ref_convert, gate_up_act as ref_gate_up,
    gemm as ref_gemm, gemv as ref_gemv, mul as ref_mul, row_gather as ref_row_gather,
};

use super::*;
use crate::gemv::act::ActKind;
use crate::gemv::tests::{
    Rng, TYPES, activations, assert_same, check_against_reference, levels, run, seen, weights,
    widths,
};
use crate::gemv::{Activations, GemvError, WeightType, Weights, gemv};

/// Values that exercise IEEE edge cases: signed zeros, infinities, NaNs with payloads and
/// signs, subnormals, the F16 overflow threshold, and values halfway between F16 and BF16
/// neighbours.
fn specials() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7F80_0001),
        f32::from_bits(0xFFC1_2345),
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(0x0000_1234),
        65504.0,
        65519.99,
        65520.0,
        6.0e-8,
        2.98e-8,
        1.0 + f32::EPSILON * 1024.0,
        1.0 + 2f32.powi(-8),
        -(1.0 + 3.0 * 2f32.powi(-8)),
        f32::MAX,
        -f32::MAX,
        1.0,
        -3.5,
    ]
}

/// Bit-equal (NaN payloads included).
fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    let g: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    let w: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
    assert_eq!(g, w, "{what}");
}

#[test]
fn gemm_equals_gemv_rows_and_the_reference() {
    let mut rng = Rng(21);
    for ty in TYPES {
        let k = widths(ty)[1];
        let n_out = 11;
        for rows in [17, 24, 33, 64] {
            let w = weights(&mut rng, ty, n_out, k);
            let x = activations(&mut rng, rows, k);
            let want = run(Level::Portable, ty, &w, n_out, k, &x, rows);
            for level in levels() {
                let got = run(level, ty, &w, n_out, k, &x, rows);
                assert_same(&got, &want, &format!("Gemm {ty:?} rows {rows} {level:?}"));
                // The same rows, 8 at a time as decode GEMV calls.
                for r0 in (0..rows).step_by(8) {
                    let r = (rows - r0).min(8);
                    let part = run(level, ty, &w, n_out, k, &x[r0 * k..(r0 + r) * k], r);
                    assert_same(
                        &part,
                        &want[r0 * n_out..(r0 + r) * n_out],
                        &format!("Gemm {ty:?} rows {r0}+{r} {level:?}"),
                    );
                }
            }
            check_against_reference(ty, &w, n_out, k, &x, rows, &want);
            // `reference::gemm` is `reference::gemv`'s value, which the bound above used.
            let wd = syntropy_kernels::reference::dequant::dequantize(ty.ggml(), &w).unwrap();
            let xs = seen(ty, &x, k);
            let (mut a, mut b) = (vec![0.0f32; rows * n_out], vec![0.0f32; rows * n_out]);
            let m = Matrix::new(&wd, n_out, k).unwrap();
            ref_gemm(m, &xs, rows, &mut a).unwrap();
            ref_gemv(m, &xs, rows, &mut b).unwrap();
            assert_bits(&a, &b, "reference Gemm vs Gemv");
        }
    }
}

/// `GateUpAct` at `level` over a fresh scratch.
fn gate_up(
    level: Level,
    ty: WeightType,
    w: &[u8],
    n: usize,
    k: usize,
    x: &[f32],
    rows: usize,
) -> Vec<f32> {
    let wm = Weights::new(ty, w, 2 * n, k).unwrap();
    let mut a = Activations::new();
    a.quantize(ty.act(), x, rows, k).unwrap();
    let mut y = vec![f32::from_bits(0x7FC0_BEEF); rows * n];
    let mut scratch = vec![0.0f32; gate_up_scratch(rows)];
    gate_up_act(level, &wm, &a.view(), &mut y, &mut scratch).unwrap();
    y
}

#[test]
fn gate_up_act_is_silu_of_the_gemv_products() {
    let mut rng = Rng(22);
    for ty in TYPES {
        let k = widths(ty)[1];
        // Below one chunk, exactly one, and a partial last chunk.
        for (n, rows) in [(5, 1), (64, 3), (100, 8), (130, 17)] {
            let w = weights(&mut rng, ty, 2 * n, k);
            let x = activations(&mut rng, rows, k);
            let want = gate_up(Level::Portable, ty, &w, n, k, &x, rows);
            // `silu(g) · u` with the portable `Gemv` outputs of the gate and up rows.
            let g = run(Level::Portable, ty, &w[..w.len() / 2], n, k, &x, rows);
            let u = run(Level::Portable, ty, &w[w.len() / 2..], n, k, &x, rows);
            let composed: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
            assert_same(&want, &composed, &format!("{ty:?} n {n} rows {rows}"));
            for level in levels() {
                let got = gate_up(level, ty, &w, n, k, &x, rows);
                assert_same(&got, &want, &format!("{ty:?} n {n} rows {rows} {level:?}"));
            }
            // Each product is within the GEMV bound of the reference product.
            check_against_reference(ty, &w[..w.len() / 2], n, k, &x, rows, &g);
            check_against_reference(ty, &w[w.len() / 2..], n, k, &x, rows, &u);
            // And the reference `GateUpAct` is the same composition of reference products.
            let wd = syntropy_kernels::reference::dequant::dequantize(ty.ggml(), &w).unwrap();
            let xs = seen(ty, &x, k);
            let (gate, up) = wd.split_at(n * k);
            let (gm, um) = (
                Matrix::new(gate, n, k).unwrap(),
                Matrix::new(up, n, k).unwrap(),
            );
            let mut yr = vec![0.0f32; rows * n];
            ref_gate_up(GatedAct::Silu, gm, um, &xs, rows, &mut yr).unwrap();
            let (mut gr, mut ur) = (vec![0.0f32; rows * n], vec![0.0f32; rows * n]);
            ref_gemv(gm, &xs, rows, &mut gr).unwrap();
            ref_gemv(um, &xs, rows, &mut ur).unwrap();
            let rc: Vec<f32> = gr.iter().zip(&ur).map(|(&g, &u)| silu(g) * u).collect();
            assert_bits(&yr, &rc, &format!("reference GateUpAct {ty:?}"));
        }
    }
}

#[test]
fn gate_up_act_refusals() {
    let ty = WeightType::Q8_0;
    let w = vec![0u8; 3 * 34];
    let odd = Weights::new(ty, &w, 3, 32).unwrap();
    let mut a = Activations::new();
    a.quantize(ty.act(), &[0.0; 32], 1, 32).unwrap();
    let mut y = [7.0f32; 2];
    let mut s = vec![0.0f32; gate_up_scratch(1)];
    assert!(matches!(
        gate_up_act(Level::Portable, &odd, &a.view(), &mut y, &mut s),
        Err(GemvError::Len { arg: "weights", .. })
    ));
    let even = Weights::new(ty, &w[..2 * 34], 2, 32).unwrap();
    assert!(matches!(
        gate_up_act(Level::Portable, &even, &a.view(), &mut y, &mut s[..1]),
        Err(GemvError::Len { arg: "scratch", .. })
    ));
    assert!(matches!(
        gate_up_act(Level::Portable, &even, &a.view(), &mut [], &mut s),
        Err(GemvError::Len { arg: "y", .. })
    ));
    a.quantize(ActKind::F32, &[0.0; 32], 1, 32).unwrap();
    assert_eq!(
        gate_up_act(Level::Portable, &even, &a.view(), &mut y, &mut s),
        Err(GemvError::Activations)
    );
    assert_eq!(y, [7.0; 2]);
}

/// A table of `rows` rows of `width` elements of `ty`, with special values in the float types.
fn table(rng: &mut Rng, ty: GgmlType, rows: usize, width: usize) -> Vec<u8> {
    let sp = specials();
    let val = |rng: &mut Rng, i: usize| {
        if i.is_multiple_of(5) {
            sp[(rng.next() as usize) % sp.len()]
        } else {
            rng.spread()
        }
    };
    let n = rows * width;
    match ty {
        GgmlType::F32 => (0..n).flat_map(|i| val(rng, i).to_le_bytes()).collect(),
        GgmlType::F16 => (0..n)
            .flat_map(|i| f16_bits(val(rng, i)).to_le_bytes())
            .collect(),
        GgmlType::BF16 => (0..n)
            .flat_map(|i| round_bf16(val(rng, i)).to_le_bytes())
            .collect(),
        _ => {
            let wt = WeightType::from_ggml(ty).unwrap();
            weights(rng, wt, rows, width)
        }
    }
}

const TABLE_TYPES: [GgmlType; 6] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::BF16,
    GgmlType::Q8_0,
    GgmlType::Q4_0,
    GgmlType::Q4_K,
];

#[test]
fn row_gather_equals_the_reference() {
    let mut rng = Rng(23);
    for ty in TABLE_TYPES {
        let width = if ty == GgmlType::Q4_K { 512 } else { 64 };
        let rows = 9;
        let t = table(&mut rng, ty, rows, width);
        // Repeats, the first and the last row.
        let ids = [3i32, 0, 8, 3, 3, 5, 8, 1];
        let uids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let mut want = vec![0.0f32; ids.len() * width];
        ref_row_gather(ty, &t, width, &uids, &mut want).unwrap();
        for level in levels() {
            let mut got = vec![f32::from_bits(0x7FC0_BEEF); ids.len() * width];
            row_gather(level, ty, &t, width, &ids, &mut got).unwrap();
            assert_bits(&got, &want, &format!("{ty:?} {level:?}"));
        }
    }
}

#[test]
fn row_gather_refusals_write_nothing() {
    let t = vec![0u8; 4 * 34];
    for ids in [&[0, 1, 4][..], &[2, -1], &[i32::MIN], &[i32::MAX]] {
        let mut out = vec![7.0f32; ids.len() * 32];
        let e = row_gather(Level::Portable, GgmlType::Q8_0, &t, 32, ids, &mut out).unwrap_err();
        assert!(
            matches!(e, GatherError::Id { rows: 4, .. }),
            "{ids:?}: {e:?}"
        );
        assert!(out.iter().all(|&v| v == 7.0));
    }
    let mut out = vec![7.0f32; 64];
    for (ty, width, t) in [
        (GgmlType::Q4_1, 32, &t[..]),
        (GgmlType::Q8_0, 16, &t[..]),
        (GgmlType::Q8_0, 32, &t[..33]),
        (GgmlType::Q8_0, 0, &t[..]),
    ] {
        assert_eq!(
            row_gather(Level::Portable, ty, t, width, &[0], &mut out),
            Err(GatherError::Shape),
            "{ty:?} {width}"
        );
    }
    assert_eq!(
        row_gather(
            Level::Portable,
            GgmlType::Q8_0,
            &t,
            32,
            &[0, 1, 2],
            &mut out
        ),
        Err(GatherError::Shape)
    );
    assert!(out.iter().all(|&v| v == 7.0));
}

/// Random values with one special value in every three.
fn mixed(rng: &mut Rng, n: usize) -> Vec<f32> {
    let sp = specials();
    (0..n)
        .map(|i| {
            if i.is_multiple_of(3) {
                sp[(rng.next() as usize) % sp.len()]
            } else {
                rng.spread() * 1e3
            }
        })
        .collect()
}

#[test]
fn add_and_mul_equal_the_reference() {
    let mut rng = Rng(24);
    for n in [0, 1, 7, 16, 33, 1000] {
        let (a, b) = (mixed(&mut rng, n), mixed(&mut rng, n));
        let (mut wa, mut wm) = (vec![0.0f32; n], vec![0.0f32; n]);
        ref_add(&a, Operand::Rows(&b), n, 1, &mut wa).unwrap();
        ref_mul(&a, Operand::Rows(&b), n, 1, &mut wm).unwrap();
        for level in levels() {
            let mut y = vec![0.0f32; n];
            add(level, &a, &b, &mut y);
            assert_bits(&y, &wa, &format!("Add n {n} {level:?}"));
            mul(level, &a, &b, &mut y);
            assert_bits(&y, &wm, &format!("Mul n {n} {level:?}"));
        }
    }
    // Every pair of special values, both ways round.
    let sp = specials();
    let a: Vec<f32> = sp.iter().flat_map(|&x| sp.iter().map(move |_| x)).collect();
    let b: Vec<f32> = sp.iter().flat_map(|_| sp.iter().copied()).collect();
    let n = a.len();
    let (mut wa, mut wm) = (vec![0.0f32; n], vec![0.0f32; n]);
    ref_add(&a, Operand::Rows(&b), n, 1, &mut wa).unwrap();
    ref_mul(&a, Operand::Rows(&b), n, 1, &mut wm).unwrap();
    for level in levels() {
        let mut y = vec![0.0f32; n];
        add(level, &a, &b, &mut y);
        // A NaN's payload from an IEEE addition is the hardware's choice, as in the reference;
        // the kinds must agree and every other value must be bit-equal.
        for (i, (&g, &w)) in y.iter().zip(&wa).enumerate() {
            assert!(
                g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                "Add {level:?} {:e} + {:e}: {g:e} vs {w:e}",
                a[i],
                b[i]
            );
        }
        mul(level, &a, &b, &mut y);
        for (i, (&g, &w)) in y.iter().zip(&wm).enumerate() {
            assert!(
                g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                "Mul {level:?} {:e} * {:e}: {g:e} vs {w:e}",
                a[i],
                b[i]
            );
        }
    }
}

const FLOATS: [(DType, GgmlType); 3] = [
    (DType::F32, GgmlType::F32),
    (DType::F16, GgmlType::F16),
    (DType::BF16, GgmlType::BF16),
];

/// `values` stored as `t`, with `t`'s own rounding.
fn stored(t: DType, values: &[f32]) -> Vec<u8> {
    match t {
        DType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::F16 => values
            .iter()
            .flat_map(|&v| f16_bits(v).to_le_bytes())
            .collect(),
        _ => values
            .iter()
            .flat_map(|&v| round_bf16(v).to_le_bytes())
            .collect(),
    }
}

#[test]
fn convert_equals_the_reference() {
    let mut rng = Rng(25);
    let mut values = specials();
    values.extend(mixed(&mut rng, 997));
    // Every F16 bit pattern, through F32, exercises each F16 and BF16 rounding case.
    values.extend((0..=u16::MAX).map(syntropy_kernels::reference::act_quant::f16_to_f32));
    for (from, gf) in FLOATS {
        let src = stored(from, &values);
        for (to, gt) in FLOATS {
            if from == to {
                continue;
            }
            let tw = if to == DType::F32 { 4 } else { 2 };
            let mut want = vec![0u8; values.len() * tw];
            ref_convert(gf, &src, gt, &mut want).unwrap();
            for level in levels() {
                let mut got = vec![0xA5u8; values.len() * tw];
                convert_rows(level, from, &src, to, &mut got);
                assert!(got == want, "{from:?} -> {to:?} {level:?}");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Random lengths and seeds: every level equals the reference for `Add`, `Mul` and every
    /// `Convert` pair (finite and non-finite values, NaN compared by kind for `Add` and `Mul`).
    #[test]
    fn elementwise_random(seed: u64, n in 0usize..300) {
        let mut rng = Rng(seed);
        let (a, b) = (mixed(&mut rng, n), mixed(&mut rng, n));
        let (mut wa, mut wm) = (vec![0.0f32; n], vec![0.0f32; n]);
        ref_add(&a, Operand::Rows(&b), n, 1, &mut wa).unwrap();
        ref_mul(&a, Operand::Rows(&b), n, 1, &mut wm).unwrap();
        for level in levels() {
            let mut y = vec![0.0f32; n];
            add(level, &a, &b, &mut y);
            prop_assert!(y.iter().zip(&wa).all(|(g, w)| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan())));
            mul(level, &a, &b, &mut y);
            prop_assert!(y.iter().zip(&wm).all(|(g, w)| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan())));
            for (from, gf) in FLOATS {
                let src = stored(from, &a);
                for (to, gt) in FLOATS {
                    if from == to {
                        continue;
                    }
                    let tw = if to == DType::F32 { 4 } else { 2 };
                    let mut want = vec![0u8; n * tw];
                    ref_convert(gf, &src, gt, &mut want).unwrap();
                    let mut got = vec![0u8; n * tw];
                    convert_rows(level, from, &src, to, &mut got);
                    prop_assert_eq!(&got, &want);
                }
            }
        }
    }

    /// Random prefill shapes: `Gemm` at every level equals the portable kernels, and the
    /// portable kernels are within the bound of the reference.
    #[test]
    fn gemm_random(seed: u64, t in 0usize..5, kb in 1usize..4, rows in 17usize..48, n_out in 1usize..9) {
        let ty = TYPES[t];
        let k = kb * ty.block_elems().max(32);
        let mut rng = Rng(seed);
        let w = weights(&mut rng, ty, n_out, k);
        let x = activations(&mut rng, rows, k);
        let want = run(Level::Portable, ty, &w, n_out, k, &x, rows);
        for level in levels() {
            let got = run(level, ty, &w, n_out, k, &x, rows);
            assert_same(&got, &want, &format!("{ty:?} {level:?}"));
        }
        check_against_reference(ty, &w, n_out, k, &x, rows, &want);
    }
}

/// `gemv` over a range of output rows equals the whole matrix for the rows the chunked
/// `GateUpAct` reads: `Weights::rows` keeps every row's bits.
#[test]
fn weight_row_views_keep_each_output() {
    let mut rng = Rng(26);
    for ty in TYPES {
        let k = widths(ty)[0];
        let (n, rows) = (20, 3);
        let w = weights(&mut rng, ty, n, k);
        let x = activations(&mut rng, rows, k);
        let whole = run(Level::Portable, ty, &w, n, k, &x, rows);
        let wm = Weights::new(ty, &w, n, k).unwrap();
        let mut a = Activations::new();
        a.quantize(ty.act(), &x, rows, k).unwrap();
        for (lo, hi) in [(0, 7), (7, 20), (19, 20)] {
            let mut y = vec![0.0f32; rows * (hi - lo)];
            gemv(
                Level::Portable,
                &wm.rows(lo..hi),
                &a.view(),
                0..hi - lo,
                &mut y,
            )
            .unwrap();
            for r in 0..rows {
                assert_same(
                    &y[r * (hi - lo)..(r + 1) * (hi - lo)],
                    &whole[r * n + lo..r * n + hi],
                    &format!("{ty:?} rows {lo}..{hi}"),
                );
            }
        }
    }
}
