// SPDX-License-Identifier: MIT OR Apache-2.0

//! The op-level comparison with declared activation quantisation (COMP-E13-T03 criterion 4;
//! RFC-0042 §6.4): for an op whose checkpoint declares its activation quantisation, the
//! comparison inserts the same fake-quant step into the baseline, so a W8A16 kernel on a
//! declared-W8A8 op fails `tol_op`, and the W8A8 kernel that applies the declared step passes.
//!
//! The op is a `Gemv` on E4M3 weights with a per-tensor scale, whose checkpoint declares
//! dynamic E4M3 activations in groups of 128 (RFC-0042 §4.3 rule a). ggml has no FP8 product,
//! so the baseline is the one RFC-0042 §6.2 gives FP8: ggml's F32 product over the `widen.f32`
//! weights. Here a sequential F32 dot stands in for it; the harness that runs ggml itself is
//! QUAL-E4-T06's. `tol_op` is computed from RFC-0042 §6.4's formula over the cases, not taken
//! from a table: the published values are pending calibration.

use syntropy_core::ActQuant;
use syntropy_kernels::reference::act_quant::{
    Q8Path, bf16_to_f32, fake_quant, round_bf16, round_e4m3,
};
use syntropy_kernels::reference::reduce;

const K: usize = 1024;
const ROWS: usize = 64;
const CASES: usize = 40;
const DECLARED: ActQuant = ActQuant::Fp8Dynamic { group: 128 };

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    /// Roughly normal, with an occasional outlier channel.
    fn activation(&mut self, i: usize) -> f32 {
        let v = self.next() + self.next() + self.next() + self.next();
        if i % 97 == 5 { v * 20.0 } else { v }
    }
}

/// One case: E4M3 weights already widened to F32 (row-major, `ROWS × K`; the per-tensor scale
/// 2^-4 is a power of two, so widening is exact) and one activation row.
fn case(seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng(seed | 1);
    let scale = 1.0 / 16.0;
    let w = (0..ROWS * K)
        .map(|_| round_e4m3(rng.next() * 300.0) * scale)
        .collect();
    let x = (0..K).map(|i| rng.activation(i)).collect();
    (w, x)
}

fn dot_f64(w: &[f32], x: &[f32]) -> f64 {
    w.iter()
        .zip(x)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum()
}

/// The F32Fixed order (RFC-0060 §4.1), from the reference's one implementation of it.
fn dot_fixed(w: &[f32], x: &[f32]) -> f32 {
    reduce::dot(w, x)
}

/// The stand-in for ggml's F32 product: one sequential accumulator.
fn dot_sequential(w: &[f32], x: &[f32]) -> f32 {
    w.iter().zip(x).fold(0.0f32, |s, (&a, &b)| s + a * b)
}

/// A candidate kernel's accumulation order: four interleaved accumulators, as a SIMD kernel
/// sums, added at the end.
fn dot_lanes(w: &[f32], x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 4];
    for (i, (&a, &b)) in w.iter().zip(x).enumerate() {
        acc[i % 4] += a * b;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

fn gemv(w: &[f32], x: &[f32], dot: impl Fn(&[f32], &[f32]) -> f32) -> Vec<f64> {
    w.chunks(K).map(|row| f64::from(dot(row, x))).collect()
}

fn gemv_f64(w: &[f32], x: &[f32]) -> Vec<f64> {
    w.chunks(K).map(|row| dot_f64(row, x)).collect()
}

/// RFC-0060 §4.2: `‖y − y_ref‖₂ / ‖y_ref‖₂`.
fn rel_l2(y: &[f64], y_ref: &[f64]) -> f64 {
    let num: f64 = y.iter().zip(y_ref).map(|(a, b)| (a - b) * (a - b)).sum();
    let den: f64 = y_ref.iter().map(|b| b * b).sum();
    (num / den).sqrt()
}

/// The nearest-rank p99.
fn p99(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let rank = (0.99 * v.len() as f64).ceil() as usize;
    v[rank.max(1) - 1]
}

fn bf16(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| bf16_to_f32(round_bf16(v))).collect()
}

fn fq(x: &[f32]) -> Vec<f32> {
    fake_quant(DECLARED, x, Q8Path::Reference).expect("whole groups")
}

struct Spreads {
    /// `rel_l2(F32Fixed reference with BF16-rounded inputs, F64)` per case.
    floor: Vec<f64>,
    /// `rel_l2(ggml, F32Fixed reference)` per case.
    ggml_vs_ref: Vec<f64>,
    /// `rel_l2(W8A8 kernel, ggml)` per case.
    w8a8: Vec<f64>,
    /// `rel_l2(W8A16 kernel, ggml)` per case, with the declared step in the baseline.
    w8a16: Vec<f64>,
}

fn spreads() -> Spreads {
    let mut s = Spreads {
        floor: Vec::new(),
        ggml_vs_ref: Vec::new(),
        w8a8: Vec::new(),
        w8a16: Vec::new(),
    };
    for c in 0..CASES {
        let (w, x) = case(0x9E37_79B9_7F4A_7C15 ^ (c as u64 * 0x1000_0001));
        let xq = fq(&x);
        // The declared step is part of the reference and of the baseline.
        let y64 = gemv_f64(&w, &xq);
        let reference = gemv(&w, &xq, dot_fixed);
        let ggml = gemv(&w, &xq, dot_sequential);
        // Both inputs BF16-rounded; the E4M3 weights are exact in BF16.
        let floor = gemv(&bf16(&w), &bf16(&xq), dot_fixed);
        s.floor.push(rel_l2(&floor, &y64));
        s.ggml_vs_ref.push(rel_l2(&ggml, &reference));
        // The W8A8 kernel applies the declared quantisation; W8A16 reads BF16 activations.
        s.w8a8.push(rel_l2(&gemv(&w, &xq, dot_lanes), &ggml));
        let w8a16 = gemv(&w, &bf16(&x), dot_lanes);
        s.w8a16.push(rel_l2(&w8a16, &ggml));
    }
    s
}

#[test]
fn a_w8a16_kernel_on_a_declared_w8a8_op_fails_tol_op() {
    let s = spreads();
    let eps_op = p99(s.floor.clone());
    let tol_op = (2.0 * p99(s.ggml_vs_ref.clone())).max(eps_op);
    let w8a8 = p99(s.w8a8.clone());
    let w8a16 = p99(s.w8a16.clone());
    eprintln!("eps_op {eps_op:e}, tol_op {tol_op:e}, W8A8 p99 {w8a8:e}, W8A16 p99 {w8a16:e}");
    assert!(eps_op > 0.0 && tol_op >= eps_op);
    // The kernel that applies the declared step passes.
    assert!(w8a8 <= tol_op, "W8A8 {w8a8:e} > tol_op {tol_op:e}");
    // The kernel that skips it fails, and not by a hair: E4M3 keeps 3 mantissa bits and BF16
    // keeps 8, so the skipped step is several times the BF16 floor.
    assert!(w8a16 > 4.0 * tol_op, "W8A16 {w8a16:e} vs tol_op {tol_op:e}");
    // Every case fails, not only the p99.
    let min = s.w8a16.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(min > tol_op, "W8A16 min {min:e} vs tol_op {tol_op:e}");
}
