// SPDX-License-Identifier: MIT OR Apache-2.0

//! GEMV against `kernels::reference` and across levels.
//!
//! - Each level the host can run equals the portable kernels bit for bit, for every type,
//!   rows 1–16 and widths with and without a partial float lane group.
//! - The portable kernels agree with `reference::gemv` on the dequantised weights and the
//!   quantised activations within a rounding bound derived from the operation count (below).
//! - The activation quantisers equal the reference quantisers bit for bit.
//! - Hand-computed cases, edge cases, refusals and the row partition.
//! - Regression: pinned output bits for one seeded case per type, unaligned weights, zero
//!   activations and F32 overflow, each the same at every level.
//! - Once `numerics/tolerances-v1.toml` publishes `Gemv`'s op-level `tol_ref` for a class, the
//!   kernel's `rel_l2` against the reference is checked against it as well.
//!
//! **The bound.** With `u = 2^-24`, a sum of `n` F32 terms rounded `n − 1` times and products
//! rounded once is within `(n + 1) · u · Σ|terms|` of the exact sum (standard first-order
//! analysis). Both the kernel and the reference are compared with the exact value computed in
//! F64 from the same quantised operands, each against its own operation count, and with each
//! other against the sum of the two. These are `[estimate]` bounds from the derivation, not
//! calibrated tolerances: op tolerances in `numerics/tolerances-v1.toml` are still pending
//! (QUAL-E4-T04 publishes them).

use proptest::prelude::*;
use syntropy_kernels::CpuIsa;
use syntropy_kernels::reference::act_quant::{
    Q8Path, bf16_to_f32, f16_bits, f16_to_f32, quantize_q8_0, quantize_q8_k, round_bf16,
};
use syntropy_kernels::reference::dequant::dequantize;
use syntropy_kernels::reference::{Matrix, gemv as ref_gemv};

use super::act::ActKind;
use super::*;

/// splitmix64: a seeded stream for the generators.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn byte(&mut self) -> u8 {
        self.next() as u8
    }

    /// Uniform in `[-1, 1)`.
    pub(crate) fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// A value with a spread of magnitudes: `unit · 2^e`, `e` in `[-6, 6]`.
    pub(crate) fn spread(&mut self) -> f32 {
        let e = (self.next() % 13) as i32 - 6;
        self.unit() * 2f32.powi(e)
    }

    /// A positive F16 scale in roughly `[2^-12, 2^-4]`, as bits.
    fn scale(&mut self) -> [u8; 2] {
        let e = (self.next() % 9) as i32 - 12;
        let v = (0.5 + self.unit().abs() * 0.5) * 2f32.powi(e);
        f16_bits(v).to_le_bytes()
    }
}

pub(crate) const TYPES: [WeightType; 5] = [
    WeightType::F16,
    WeightType::Bf16,
    WeightType::Q8_0,
    WeightType::Q4_0,
    WeightType::Q4K,
];

/// `n_out` random rows of `k` weights of `ty`, with finite scales and values.
pub(crate) fn weights(rng: &mut Rng, ty: WeightType, n_out: usize, k: usize) -> Vec<u8> {
    let blocks = n_out * k / ty.block_elems();
    let mut out = Vec::with_capacity(blocks * ty.block_bytes());
    for _ in 0..blocks {
        match ty {
            WeightType::F16 => out.extend(f16_bits(rng.spread()).to_le_bytes()),
            WeightType::Bf16 => out.extend(round_bf16(rng.spread()).to_le_bytes()),
            WeightType::Q8_0 | WeightType::Q4_0 => {
                out.extend(rng.scale());
                for _ in 2..ty.block_bytes() {
                    out.push(rng.byte());
                }
            }
            WeightType::Q4K => {
                out.extend(rng.scale());
                out.extend(rng.scale());
                for _ in 4..ty.block_bytes() {
                    out.push(rng.byte());
                }
            }
        }
    }
    out
}

/// `rows · k` activations with a spread of magnitudes.
pub(crate) fn activations(rng: &mut Rng, rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k).map(|_| rng.spread()).collect()
}

/// Every level this host can run, the portable one first.
pub(crate) fn levels() -> Vec<Level> {
    let mut v = vec![Level::Portable];
    #[cfg(target_arch = "x86_64")]
    for isa in [CpuIsa::X86V3, CpuIsa::X86V4] {
        let l = Level::for_isa(isa);
        if !v.contains(&l) {
            v.push(l);
        }
    }
    #[cfg(target_arch = "aarch64")]
    for isa in [CpuIsa::Arm82Dot, CpuIsa::Arm86I8mm] {
        let l = Level::for_isa(isa);
        if !v.contains(&l) {
            v.push(l);
        }
    }
    v
}

/// Runs one GEMV over every output row at `level`, into a fresh `y`.
pub(crate) fn run(
    level: Level,
    ty: WeightType,
    w: &[u8],
    n_out: usize,
    k: usize,
    x: &[f32],
    rows: usize,
) -> Vec<f32> {
    let w = Weights::new(ty, w, n_out, k).unwrap();
    let mut a = Activations::new();
    a.quantize(ty.act(), x, rows, k).unwrap();
    let mut y = vec![f32::from_bits(0x7FC0_DEAD); rows * n_out];
    gemv(level, &w, &a.view(), 0..n_out, &mut y).unwrap();
    y
}

/// Bit-equal, or both NaN.
pub(crate) fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

pub(crate) fn assert_same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            same(g, w),
            "{what}: output {i}: {g:e} ({:#x}) != {w:e} ({:#x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

/// The values the product multiplies, per activation row: `x` as the kernel quantised it.
pub(crate) fn seen(ty: WeightType, x: &[f32], k: usize) -> Vec<f32> {
    x.chunks_exact(k.max(1))
        .flat_map(|row| match ty.act() {
            ActKind::F32 => row.to_vec(),
            ActKind::Q8_0 => quantize_q8_0(row, Q8Path::Arm64Neon)
                .unwrap()
                .iter()
                .flat_map(|b| {
                    let d = f16_to_f32(b.d);
                    b.qs.map(|q| f32::from(q) * d)
                })
                .collect(),
            ActKind::Q8K => quantize_q8_k(row)
                .unwrap()
                .iter()
                .flat_map(|b| b.qs.map(|q| f32::from(q) * b.d))
                .collect(),
        })
        .collect()
}

/// The exact weight values in F64, as two parts whose sum is the weight: for `Q4_K`
/// `d·sc·q` and `−dmin·m`; for the other types the value and 0.
fn exact_weights(ty: WeightType, w: &[u8], k: usize) -> Vec<(f64, f64)> {
    if ty != WeightType::Q4K {
        return dequantize(ty.ggml(), w)
            .unwrap()
            .into_iter()
            .map(|v| (f64::from(v), 0.0))
            .collect();
    }
    let _ = k;
    let mut out = Vec::new();
    for blk in w.chunks_exact(ty.block_bytes()) {
        let (d, dmin) = (f64::from(half(blk)), f64::from(half(&blk[2..])));
        let q = &blk[16..];
        for g in 0..4 {
            for hi in [false, true] {
                let (sc, m) = scale_min_k4(2 * g + usize::from(hi), &blk[4..16]);
                for l in 0..32 {
                    let b = q[32 * g + l];
                    let n = if hi { b >> 4 } else { b & 0xF };
                    out.push((d * f64::from(sc) * f64::from(n), -dmin * f64::from(m)));
                }
            }
        }
    }
    out
}

/// Checks the kernel's output `y` and `reference::gemv`'s against the exact F64 value and each
/// other, within the bounds of the module docs.
pub(crate) fn check_against_reference(
    ty: WeightType,
    w: &[u8],
    n_out: usize,
    k: usize,
    x: &[f32],
    rows: usize,
    y: &[f32],
) {
    const U: f64 = 1.0 / (1u64 << 24) as f64;
    let xs = seen(ty, x, k);
    let wd = dequantize(ty.ggml(), w).unwrap();
    let m = Matrix::new(&wd, n_out, k).unwrap();
    let mut yr = vec![0.0f32; rows * n_out];
    ref_gemv(m, &xs, rows, &mut yr).unwrap();
    let we = exact_weights(ty, w, k);
    // Terms the kernel adds sequentially: blocks for the integer types, lane steps plus the
    // fold for the float types; each block term carries up to five roundings of its own.
    let kernel_n = match ty {
        WeightType::F16 | WeightType::Bf16 => k / LANES + 1 + 5,
        _ => k / ty.block_elems() + 5,
    } as f64;
    // The reference: 32 sequential terms per block, then the pairwise tree over blocks, and
    // for `Q4_K` the dequantiser's three roundings per weight.
    let ref_n = (32 + 64 + 3) as f64;
    for r in 0..rows {
        for o in 0..n_out {
            let (mut exact, mut mag, mut mag_ref) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..k {
                let (a, b) = we[o * k + i];
                let xv = f64::from(xs[r * k + i]);
                exact += (a + b) * xv;
                mag += (a.abs() + b.abs()) * xv.abs();
                mag_ref +=
                    f64::from(wd[o * k + i]).abs() * xv.abs() + (a.abs() + b.abs()) * xv.abs();
            }
            let (yk, yrv) = (f64::from(y[r * n_out + o]), f64::from(yr[r * n_out + o]));
            // An absolute floor for subnormal results.
            let floor = 1e-38;
            let bk = kernel_n * U * mag + floor;
            let br = ref_n * U * mag_ref + floor;
            assert!(
                (yk - exact).abs() <= bk,
                "{ty:?} kernel row {r} out {o}: {yk:e} vs exact {exact:e}, bound {bk:e}"
            );
            assert!(
                (yrv - exact).abs() <= br,
                "{ty:?} reference row {r} out {o}: {yrv:e} vs exact {exact:e}, bound {br:e}"
            );
            assert!(
                (yk - yrv).abs() <= bk + br,
                "{ty:?} kernel vs reference row {r} out {o}: {yk:e} vs {yrv:e}"
            );
        }
    }
}

/// Widths for `ty`: whole blocks, and for the float types widths that leave a partial lane
/// group.
pub(crate) fn widths(ty: WeightType) -> Vec<usize> {
    match ty {
        WeightType::F16 | WeightType::Bf16 => vec![1, 7, 31, 32, 33, 64, 100, 256],
        WeightType::Q4K => vec![256, 512, 768],
        _ => vec![32, 64, 96, 256],
    }
}

#[test]
fn every_level_equals_the_portable_kernels() {
    let mut rng = Rng(1);
    for ty in TYPES {
        for k in widths(ty) {
            for rows in [1, 2, 3, 5, 8, 9, 16] {
                let n_out = 7;
                let w = weights(&mut rng, ty, n_out, k);
                let x = activations(&mut rng, rows, k);
                let want = run(Level::Portable, ty, &w, n_out, k, &x, rows);
                for level in levels() {
                    let got = run(level, ty, &w, n_out, k, &x, rows);
                    assert_same(&got, &want, &format!("{ty:?} k {k} rows {rows} {level:?}"));
                }
            }
        }
    }
}

#[test]
fn portable_agrees_with_the_reference() {
    let mut rng = Rng(2);
    for ty in TYPES {
        for k in widths(ty) {
            for rows in [1, 4, 8] {
                let n_out = 5;
                let w = weights(&mut rng, ty, n_out, k);
                let x = activations(&mut rng, rows, k);
                let y = run(Level::Portable, ty, &w, n_out, k, &x, rows);
                check_against_reference(ty, &w, n_out, k, &x, rows, &y);
            }
        }
    }
}

#[test]
fn the_quantisers_equal_the_references() {
    let mut rng = Rng(3);
    for _ in 0..50 {
        let k = 512;
        let mut x = activations(&mut rng, 1, k);
        // Exact half-integer products and zero blocks, where quantizers disagree if wrong.
        x[..32]
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = i as f32 - 15.5);
        x[32..64].fill(0.0);
        let mut a = Activations::new();
        a.quantize(ActKind::Q8_0, &x, 1, k).unwrap();
        let want = quantize_q8_0(&x, Q8Path::Arm64Neon).unwrap();
        for (b, want) in want.iter().enumerate() {
            assert_eq!(&a.view().qs_row(0)[32 * b..32 * b + 32], &want.qs[..]);
            assert!(same(a.view().d_row(0)[b], f16_to_f32(want.d)));
        }
        a.quantize(ActKind::Q8K, &x, 1, k).unwrap();
        let want = quantize_q8_k(&x).unwrap();
        for (b, want) in want.iter().enumerate() {
            assert_eq!(&a.view().qs_row(0)[256 * b..256 * b + 256], &want.qs[..]);
            assert!(same(a.view().d_row(0)[b], want.d));
            for j in 0..8 {
                let s = i32::from(want.bsums[2 * j]) + i32::from(want.bsums[2 * j + 1]);
                assert_eq!(a.view().sums_row(0)[8 * b + j], s);
            }
        }
    }
}

/// Quantising into a workspace gives what the owned buffers hold, and a short or misaligned
/// workspace is refused.
#[test]
fn workspace_quantisation_equals_the_owned_buffers() {
    let mut rng = Rng(7);
    for kind in [ActKind::F32, ActKind::Q8_0, ActKind::Q8K] {
        let (rows, k) = (3, 512);
        let x = activations(&mut rng, rows, k);
        let mut owned = Activations::new();
        owned.quantize(kind, &x, rows, k).unwrap();
        let need = workspace_bytes(kind, rows, k);
        // Backed by `u32`s, so the bytes start on a 4-byte boundary.
        let mut words = vec![0u32; need.div_ceil(4) + 1];
        // SAFETY: test-only view of the `u32` buffer as bytes; `u8` has no invalid values.
        let ws = unsafe {
            std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), words.len() * 4)
        };
        if need > 0 {
            assert!(matches!(
                quantize_into(kind, &x, rows, k, &mut ws[..need - 1]),
                Err(GemvError::Len {
                    arg: "workspace",
                    ..
                })
            ));
            assert!(matches!(
                quantize_into(kind, &x, rows, k, &mut ws[1..]),
                Err(GemvError::Len {
                    arg: "workspace",
                    ..
                })
            ));
        }
        let view = quantize_into(kind, &x, rows, k, &mut ws[..need]).unwrap();
        let o = owned.view();
        for r in 0..rows {
            match kind {
                ActKind::F32 => assert_eq!(view.f32_row(r), o.f32_row(r)),
                _ => {
                    assert_eq!(view.qs_row(r), o.qs_row(r));
                    assert_eq!(
                        view.d_row(r)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        o.d_row(r).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    if kind == ActKind::Q8K {
                        assert_eq!(view.sums_row(r), o.sums_row(r));
                    }
                }
            }
        }
    }
}

#[test]
fn hand_computed_cases() {
    // Activations 127, 126, …, 96: amax 127, so every quantised code equals its value and the
    // Q8 scale is 1 (`Q8_0`). Weights with F16 scale 1.
    let x: Vec<f32> = (0..32).map(|i| 127.0 - i as f32).collect();
    let one = f16_bits(1.0).to_le_bytes();
    // Q8_0: codes 1, −1, 1, −1, …: Σ (−1)^i (127 − i) = 16.
    let mut w = one.to_vec();
    w.extend((0..32).map(|i| if i % 2 == 0 { 1u8 } else { 0xFF }));
    assert_eq!(
        run(Level::Portable, WeightType::Q8_0, &w, 1, 32, &x, 1),
        [16.0]
    );
    // Q4_0: every nibble 9, so every weight is 1: Σ (127 − i) = 3568.
    let mut w = one.to_vec();
    w.extend([0x99u8; 16]);
    assert_eq!(
        run(Level::Portable, WeightType::Q4_0, &w, 1, 32, &x, 1),
        [3568.0]
    );
    // F16 and BF16: weights 2, x = 1, 2, …, 33: 2 · Σ = 1122 (k = 33 leaves a partial group).
    let x: Vec<f32> = (1..=33).map(|i| i as f32).collect();
    let w: Vec<u8> = (0..33).flat_map(|_| f16_bits(2.0).to_le_bytes()).collect();
    assert_eq!(
        run(Level::Portable, WeightType::F16, &w, 1, 33, &x, 1),
        [1122.0]
    );
    let w: Vec<u8> = (0..33)
        .flat_map(|_| round_bf16(2.0).to_le_bytes())
        .collect();
    assert_eq!(
        run(Level::Portable, WeightType::Bf16, &w, 1, 33, &x, 1),
        [1122.0]
    );
    // Q4_K: d = 1, dmin = 1, sub-block 0's scale 1 and minimum 1, every other scale and minimum
    // 0, every code 2: sub-block 0's weights are 1·2 − 1 = 1, the rest 0. With x = 127 in the
    // first 32 elements and 0 elsewhere, the codes are 127 and the scale −1/(−127/127)… so
    // y = 32 · 127.
    let mut w = one.to_vec();
    w.extend(one);
    let mut scales = [0u8; 12];
    scales[0] = 1;
    scales[4] = 1;
    w.extend(scales);
    w.extend([0x22u8; 128]);
    let mut x = vec![0.0f32; 256];
    x[..32].fill(127.0);
    assert_eq!(
        run(Level::Portable, WeightType::Q4K, &w, 1, 256, &x, 1),
        [32.0 * 127.0]
    );
}

#[test]
fn edge_cases() {
    let mut rng = Rng(4);
    for ty in TYPES {
        let k = widths(ty)[0];
        let n_out = 4;
        let w = weights(&mut rng, ty, n_out, k);
        let wm = Weights::new(ty, &w, n_out, k).unwrap();
        for level in levels() {
            // No rows: nothing written.
            let mut a = Activations::new();
            a.quantize(ty.act(), &[], 0, k).unwrap();
            let mut y = [7.0f32; 4];
            gemv(level, &wm, &a.view(), 0..n_out, &mut y).unwrap();
            assert_eq!(y, [7.0; 4]);
            // An empty output range: nothing written.
            let x = activations(&mut rng, 2, k);
            a.quantize(ty.act(), &x, 2, k).unwrap();
            let mut y = [7.0f32; 8];
            gemv(level, &wm, &a.view(), 2..2, &mut y).unwrap();
            assert_eq!(y, [7.0; 8]);
            // A range writes only its outputs, and a longer `y` keeps its tail.
            let mut y = [7.0f32; 10];
            gemv(level, &wm, &a.view(), 1..3, &mut y).unwrap();
            for r in 0..2 {
                assert_eq!(y[r * 4], 7.0);
                assert_eq!(y[r * 4 + 3], 7.0);
                assert_ne!(y[r * 4 + 1], 7.0);
            }
            assert_eq!(y[8..], [7.0, 7.0]);
            // A NaN in an activation row makes that row's outputs NaN, and only those.
            let mut x = activations(&mut rng, 2, k);
            x[k + k / 2] = f32::NAN;
            a.quantize(ty.act(), &x, 2, k).unwrap();
            let mut y = [0.0f32; 8];
            gemv(level, &wm, &a.view(), 0..n_out, &mut y).unwrap();
            assert!(y[..4].iter().all(|v| !v.is_nan()), "{ty:?} {level:?}");
            assert!(y[4..].iter().all(|v| v.is_nan()), "{ty:?} {level:?}");
            // An infinity in an activation does too.
            x[k + k / 2] = f32::INFINITY;
            a.quantize(ty.act(), &x, 2, k).unwrap();
            gemv(level, &wm, &a.view(), 0..n_out, &mut y).unwrap();
            if ty.act() == ActKind::F32 {
                assert!(y[4..].iter().all(|v| v.is_nan() || v.is_infinite()));
            } else {
                assert!(y[4..].iter().all(|v| v.is_nan()));
            }
        }
        // A NaN weight scale (or value) makes that output NaN, and only that one.
        let mut w2 = w.clone();
        let row = wm.row_bytes();
        match ty {
            WeightType::F16 => w2[row..row + 2].copy_from_slice(&0x7E00u16.to_le_bytes()),
            WeightType::Bf16 => w2[row..row + 2].copy_from_slice(&0x7FC0u16.to_le_bytes()),
            _ => w2[row..row + 2].copy_from_slice(&0x7E00u16.to_le_bytes()),
        }
        let x = activations(&mut rng, 1, k);
        for level in levels() {
            let y = run(level, ty, &w2, n_out, k, &x, 1);
            assert!(y[1].is_nan(), "{ty:?} {level:?}");
            assert!(
                !y[0].is_nan() && !y[2].is_nan() && !y[3].is_nan(),
                "{ty:?} {level:?}"
            );
        }
    }
}

#[test]
fn zero_width_gives_positive_zero() {
    for ty in TYPES {
        let w = Weights::new(ty, &[], 3, 0).unwrap();
        let mut a = Activations::new();
        a.quantize(ty.act(), &[], 2, 0).unwrap();
        for level in levels() {
            let mut y = [7.0f32; 6];
            gemv(level, &w, &a.view(), 0..3, &mut y).unwrap();
            assert!(
                y.iter().all(|v| v.to_bits() == 0),
                "{ty:?} {level:?}: {y:?}"
            );
        }
    }
}

#[test]
fn refusals() {
    assert_eq!(
        Weights::new(WeightType::Q4_0, &[0; 18], 1, 16).unwrap_err(),
        GemvError::RowLength {
            ty: WeightType::Q4_0,
            k_in: 16,
            block: 32
        }
    );
    assert!(matches!(
        Weights::new(WeightType::Q8_0, &[0; 33], 1, 32),
        Err(GemvError::Len {
            arg: "weights",
            len: 33,
            need: 34
        })
    ));
    let w = [0u8; 34];
    let w = Weights::new(WeightType::Q8_0, &w, 1, 32).unwrap();
    let mut a = Activations::new();
    // Too few values, and a width that is not whole blocks.
    assert!(matches!(
        a.quantize(ActKind::Q8_0, &[0.0; 31], 1, 32),
        Err(GemvError::Len { .. })
    ));
    assert_eq!(
        a.quantize(ActKind::Q8K, &[0.0; 32], 1, 32),
        Err(GemvError::Activations)
    );
    // Activations for another type, another width; a range past the matrix; a short `y`.
    a.quantize(ActKind::F32, &[0.0; 32], 1, 32).unwrap();
    let mut y = [0.0f32; 1];
    assert_eq!(
        gemv(Level::Portable, &w, &a.view(), 0..1, &mut y),
        Err(GemvError::Activations)
    );
    a.quantize(ActKind::Q8_0, &[0.0; 64], 1, 64).unwrap();
    assert_eq!(
        gemv(Level::Portable, &w, &a.view(), 0..1, &mut y),
        Err(GemvError::Activations)
    );
    a.quantize(ActKind::Q8_0, &[0.0; 64], 2, 32).unwrap();
    assert_eq!(
        gemv(Level::Portable, &w, &a.view(), 0..2, &mut [0.0; 2]),
        Err(GemvError::Range { end: 2, n_out: 1 })
    );
    assert!(matches!(
        gemv(Level::Portable, &w, &a.view(), 0..1, &mut y),
        Err(GemvError::Len {
            arg: "y",
            len: 1,
            need: 2
        })
    ));
}

#[test]
fn levels_never_exceed_the_host() {
    // A level above the host's is lowered to the host's, and one at or below is kept.
    assert_eq!(
        Level::lowered(CpuIsa::X86V2, Level::Portable),
        Level::Portable
    );
    #[cfg(target_arch = "x86_64")]
    {
        assert_eq!(Level::lowered(CpuIsa::X86V4, Level::Avx2), Level::Avx2);
        assert_eq!(
            Level::lowered(CpuIsa::X86V4Amx, Level::Avx512),
            Level::Avx512
        );
        assert_eq!(Level::lowered(CpuIsa::X86V3, Level::Avx512), Level::Avx2);
        assert_eq!(
            Level::lowered(CpuIsa::X86V3, Level::Portable),
            Level::Portable
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert_eq!(Level::lowered(CpuIsa::Arm86I8mm, Level::Dot), Level::Dot);
        assert_eq!(Level::lowered(CpuIsa::Arm92Sme2, Level::I8mm), Level::I8mm);
        assert_eq!(Level::lowered(CpuIsa::Arm82Dot, Level::I8mm), Level::Dot);
        assert_eq!(Level::lowered(CpuIsa::Arm80, Level::I8mm), Level::Portable);
    }
}

#[test]
fn partition_covers_every_row_once() {
    for ty in TYPES {
        for k in widths(ty) {
            for n_out in [0, 1, 7, 64, 100, 4097] {
                let bytes = vec![0u8; n_out * k / ty.block_elems() * ty.block_bytes()];
                let w = Weights::new(ty, &bytes, n_out, k).unwrap();
                let align = row_align(w.row_bytes());
                for parts in [0, 1, 2, 3, 8, 13, 200] {
                    let runs: Vec<_> = partition(&w, parts).collect();
                    assert_eq!(runs.len(), parts.max(1));
                    let mut next = 0;
                    for r in &runs {
                        assert_eq!(r.start, next.min(n_out));
                        assert!(r.start <= r.end);
                        if r.start < n_out {
                            assert_eq!(r.start % align, 0);
                        }
                        next = r.end;
                    }
                    assert_eq!(next, n_out);
                    // As even as whole units allow: runs differ by at most one unit, plus the
                    // last unit's shortfall when `n_out` is not whole units.
                    let lens: Vec<_> = runs.iter().map(|r| r.len()).collect();
                    let (lo, hi) = (lens.iter().min().unwrap(), lens.iter().max().unwrap());
                    assert!(hi - lo < 2 * align, "{lens:?}");
                }
            }
        }
    }
}

#[test]
fn partitioned_runs_equal_one_run() {
    let mut rng = Rng(5);
    for ty in TYPES {
        let k = widths(ty)[1];
        let (n_out, rows) = (37, 3);
        let w = weights(&mut rng, ty, n_out, k);
        let x = activations(&mut rng, rows, k);
        let wm = Weights::new(ty, &w, n_out, k).unwrap();
        let mut a = Activations::new();
        a.quantize(ty.act(), &x, rows, k).unwrap();
        for level in levels() {
            let want = run(level, ty, &w, n_out, k, &x, rows);
            for parts in [2, 3, 5, 16] {
                let mut y = vec![0.0f32; rows * n_out];
                for r in partition(&wm, parts) {
                    gemv(level, &wm, &a.view(), r, &mut y).unwrap();
                }
                assert_same(&y, &want, &format!("{ty:?} {level:?} parts {parts}"));
            }
        }
    }
}

/// The bf16 helper the tests use agrees with the kernels' widening.
#[test]
fn bf16_widening_is_exact() {
    for v in [1.0f32, -3.5, 1e-30, f32::INFINITY] {
        let h = round_bf16(v);
        assert_eq!(bf16_to_f32(h).to_bits(), u32::from(h) << 16);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Random shapes and seeds: every level equals the portable kernels, and the portable
    /// kernels agree with the reference within the bound.
    #[test]
    fn random_shapes(seed: u64, t in 0usize..5, kb in 0usize..5, rows in 1usize..=12, n_out in 1usize..12) {
        let ty = TYPES[t];
        let k = match ty {
            WeightType::F16 | WeightType::Bf16 => kb * 29 + 1,
            _ => (kb + 1) * ty.block_elems(),
        };
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

/// Which levels this host runs, printed for the test log's evidence.
#[test]
fn report_levels() {
    eprintln!("gemv levels on this host: {:?}", levels());
}

/// The inputs of the pinned outputs below: seed 11, 2 activation rows, 3 output rows, the
/// second width of `widths`.
fn golden_case(ty: WeightType) -> Vec<f32> {
    let mut rng = Rng(11);
    let k = widths(ty)[1];
    let w = weights(&mut rng, ty, 3, k);
    let x = activations(&mut rng, 2, k);
    run(Level::Portable, ty, &w, 3, k, &x, 2)
}

/// Exact-mode output must not change unnoticed (ADR-0019): the portable kernels, and so every
/// level, give these bits for `golden_case`. A change to the block order, the lane fold or a
/// quantiser fails here. Update the table only in a change meant to alter the output.
#[test]
fn outputs_are_pinned() {
    let want: [(WeightType, [u32; 6]); 5] = [
        (
            WeightType::F16,
            [
                0x420a1b10, 0x454b51ba, 0xc3d1653b, 0x3fc76c70, 0xc028a29a, 0x3f9ab456,
            ],
        ),
        (
            WeightType::Bf16,
            [
                0x420a439a, 0x454ba9fa, 0xc3d0ee64, 0x3fc733f4, 0xc02a34ae, 0x3f9a739a,
            ],
        ),
        (
            WeightType::Q8_0,
            [
                0x40a0abe7, 0xc2ab4cc2, 0x41fc936a, 0x4094d860, 0xc1b6724a, 0xc089cc5f,
            ],
        ),
        (
            WeightType::Q4_0,
            [
                0xbe40fe75, 0xbf1a503e, 0x408f99d8, 0x3f152b45, 0xbfbc2046, 0x3f151262,
            ],
        ),
        (
            WeightType::Q4K,
            [
                0x43065438, 0xc5cb1506, 0x4353f838, 0x43517b96, 0xc596e5a0, 0x44910fbe,
            ],
        ),
    ];
    for (ty, bits) in want {
        let got: Vec<u32> = golden_case(ty).iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, bits, "{ty:?}");
    }
}

/// All-zero activations: every quantised block has scale 0, and every output is +0, not NaN.
#[test]
fn zero_activations_give_positive_zero() {
    let mut rng = Rng(12);
    for ty in TYPES {
        let k = widths(ty)[1];
        let w = weights(&mut rng, ty, 5, k);
        let x = vec![0.0f32; 3 * k];
        for level in levels() {
            let y = run(level, ty, &w, 5, k, &x, 3);
            assert!(
                y.iter().all(|v| v.to_bits() == 0),
                "{ty:?} {level:?}: {y:?}"
            );
        }
    }
}

/// Weights that start at an odd address, as a tensor inside a mapped file can: every level
/// reads them unaligned and gives the same bits as an aligned copy.
#[test]
fn unaligned_weights_equal_aligned() {
    let mut rng = Rng(13);
    for ty in TYPES {
        let k = widths(ty)[2];
        let (n_out, rows) = (9, 3);
        let w = weights(&mut rng, ty, n_out, k);
        let x = activations(&mut rng, rows, k);
        let want = run(Level::Portable, ty, &w, n_out, k, &x, rows);
        for shift in 1..4 {
            let mut buf = vec![0u8; w.len() + shift];
            buf[shift..].copy_from_slice(&w);
            for level in levels() {
                let got = run(level, ty, &buf[shift..], n_out, k, &x, rows);
                assert_same(&got, &want, &format!("{ty:?} {level:?} shift {shift}"));
            }
        }
    }
}

/// Activations large enough to overflow F32: every level overflows the same way, bit for bit.
#[test]
fn overflow_is_the_same_at_every_level() {
    let mut rng = Rng(14);
    for ty in TYPES {
        let k = widths(ty)[1];
        let (n_out, rows) = (4, 2);
        let w = weights(&mut rng, ty, n_out, k);
        let x: Vec<f32> = (0..rows * k)
            .map(|i| if i % 2 == 0 { 3.0e38 } else { 2.5e38 })
            .collect();
        let want = run(Level::Portable, ty, &w, n_out, k, &x, rows);
        assert!(
            want.iter().any(|v| !v.is_finite()),
            "{ty:?}: the case should overflow: {want:?}"
        );
        for level in levels() {
            let got = run(level, ty, &w, n_out, k, &x, rows);
            assert_same(&got, &want, &format!("{ty:?} {level:?}"));
        }
    }
}

/// The dtype class whose op-level tolerance applies to `ty` (RFC-0042 §6.4).
fn dtype_class(ty: WeightType) -> syntropy_core::quant::DTypeClass {
    use syntropy_core::quant::DTypeClass;
    match ty {
        WeightType::F16 | WeightType::Bf16 => DTypeClass::Float16,
        WeightType::Q8_0 => DTypeClass::Int8,
        WeightType::Q4_0 | WeightType::Q4K => DTypeClass::Bits4,
    }
}

/// Where the embedded tolerance table publishes `Gemv`'s op-level tolerance for a type's
/// class, the kernel's `rel_l2` against `reference::gemv` stays within its `tol_ref`
/// (RFC-0042 §6.4). Until QUAL-E4-T04 publishes the entries, none exists, the derived bound of
/// the module docs is the check, and this test has nothing to compare.
#[test]
fn published_op_tolerances_hold() {
    use syntropy_kernels::tolerances::ToleranceTable;
    let table = ToleranceTable::release().unwrap();
    let mut rng = Rng(15);
    for ty in TYPES {
        let Some(tol) = table
            .op("Gemv", dtype_class(ty))
            .and_then(|e| e.value())
            .map(|t| t.tol_ref)
        else {
            continue;
        };
        let k = widths(ty)[1];
        let (n_out, rows) = (16, 4);
        let w = weights(&mut rng, ty, n_out, k);
        let x = activations(&mut rng, rows, k);
        let y = run(Level::Portable, ty, &w, n_out, k, &x, rows);
        let wd = dequantize(ty.ggml(), &w).unwrap();
        let xs = seen(ty, &x, k);
        let mut yr = vec![0.0f32; rows * n_out];
        ref_gemv(Matrix::new(&wd, n_out, k).unwrap(), &xs, rows, &mut yr).unwrap();
        let (num, den) = y
            .iter()
            .zip(&yr)
            .fold((0.0f64, 0.0f64), |(n, d), (&a, &b)| {
                let (a, b) = (f64::from(a), f64::from(b));
                (n + (a - b) * (a - b), d + b * b)
            });
        let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        assert!(rel <= tol, "{ty:?}: rel_l2 {rel:e} > tol_ref {tol:e}");
    }
}
