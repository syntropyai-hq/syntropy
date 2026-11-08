// SPDX-License-Identifier: MIT OR Apache-2.0

//! The x86-64 kernels: AVX2 with FMA and F16C (`X86V3`), and AVX-512 (`X86V4`).
//!
//! The integer block sums use the sign trick: `|w|` as unsigned bytes times `x · sign(w)` as
//! signed bytes (`vpmaddubsw`), which cannot saturate because `|w| ≤ 128` and `|x| ≤ 127`
//! give pair sums of at most 32,512. They are exact, so they equal the portable sums. The
//! float kernels keep the 32 lanes in four 256-bit or two 512-bit registers, element `k` in
//! lane `k mod 32`, and fold them as the portable kernel does.
//!
//! `X86V4`'s integer kernels are the 256-bit ones compiled with the AVX-512 features: a block
//! of 32 codes is one 256-bit register.

use std::arch::x86_64::{
    __m256i, _mm_add_epi32, _mm_and_si128, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_set1_epi8,
    _mm_shuffle_epi32, _mm_srli_epi16, _mm256_and_si256, _mm256_castsi256_ps,
    _mm256_castsi256_si128, _mm256_cvtepu16_epi32, _mm256_cvtph_ps, _mm256_extracti128_si256,
    _mm256_fmadd_ps, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set_m128i,
    _mm256_set1_epi8, _mm256_set1_epi16, _mm256_setzero_ps, _mm256_sign_epi8, _mm256_slli_epi32,
    _mm256_srli_epi16, _mm256_storeu_ps, _mm256_sub_epi8, _mm512_castsi512_ps,
    _mm512_cvtepu16_epi32, _mm512_cvtph_ps, _mm512_fmadd_ps, _mm512_setzero_ps, _mm512_slli_epi32,
    _mm512_storeu_ps,
};

use super::portable::{BlockSums, int_rows};
use super::{Job, LANES, ROW_GROUP, WeightType, fold_lanes};

/// Runs `job` with the AVX2 kernels.
///
/// # Safety
///
/// The host has AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn run_avx2(job: &Job<'_, '_>, y: &mut [f32]) {
    match job.w.ty() {
        WeightType::F16 => float_rows_avx2::<true>(job, y),
        WeightType::Bf16 => float_rows_avx2::<false>(job, y),
        WeightType::Q8_0 | WeightType::Q4_0 | WeightType::Q4K => int_rows(job, y, &Avx2(())),
    }
}

/// Runs `job` with the AVX-512 kernels.
///
/// # Safety
///
/// The host has AVX-512 F, BW, VL and DQ (and therefore AVX2, FMA and F16C).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma,f16c")]
pub(super) unsafe fn run_avx512(job: &Job<'_, '_>, y: &mut [f32]) {
    match job.w.ty() {
        WeightType::F16 => float_rows_avx512::<true>(job, y),
        WeightType::Bf16 => float_rows_avx512::<false>(job, y),
        WeightType::Q8_0 | WeightType::Q4_0 | WeightType::Q4K => int_rows(job, y, &Avx2(())),
    }
}

/// The AVX2 block sums. Only `run_avx2` and `run_avx512` make one, inside functions compiled
/// with AVX2, so its methods are only ever inlined into code the host can run.
struct Avx2(());

/// The sum of the eight `i32` lanes of `v`.
#[inline(always)]
fn hsum(v: __m256i) -> i32 {
    // SAFETY: AVX2 and SSE2 are enabled in every caller (`Avx2`'s contract).
    unsafe {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
        _mm_cvtsi128_si32(s)
    }
}

/// `Σ w · x` over 32 signed bytes each, exact (the sign trick).
#[inline(always)]
fn dot_i8(w: __m256i, x: __m256i) -> i32 {
    // SAFETY: as `hsum`.
    unsafe {
        let p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
        hsum(_mm256_madd_epi16(p, _mm256_set1_epi16(1)))
    }
}

/// `Σ n · x` over 32 unsigned bytes `n ≤ 15` and 32 signed bytes `x`, exact.
#[inline(always)]
fn dot_u4(n: __m256i, x: __m256i) -> i32 {
    // SAFETY: as `hsum`.
    unsafe {
        hsum(_mm256_madd_epi16(
            _mm256_maddubs_epi16(n, x),
            _mm256_set1_epi16(1),
        ))
    }
}

/// 32 bytes from `p`.
#[inline(always)]
fn load32(p: &[u8]) -> __m256i {
    assert!(p.len() >= 32);
    // SAFETY: `p` holds at least 32 bytes (checked), and the load is unaligned.
    unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
}

/// 32 bytes from `p`, as signed bytes.
#[inline(always)]
fn load32_i8(p: &[i8]) -> __m256i {
    assert!(p.len() >= 32);
    // SAFETY: as `load32`.
    unsafe { _mm256_loadu_si256(p.as_ptr().cast()) }
}

impl BlockSums for Avx2 {
    #[inline(always)]
    fn q8_0(&self, w: &[u8], x: &[&[i8]], out: &mut [i32; ROW_GROUP]) {
        let wv = load32(w);
        for (o, x) in out.iter_mut().zip(x) {
            *o = dot_i8(wv, load32_i8(x));
        }
    }

    #[inline(always)]
    fn q4_0(&self, w: &[u8], x: &[&[i8]], out: &mut [i32; ROW_GROUP]) {
        assert!(w.len() >= 16);
        // SAFETY: AVX2 is enabled in every caller (`Avx2`'s contract), and `w` holds 16 bytes
        // (checked).
        let wv = unsafe {
            let b = _mm_loadu_si128(w.as_ptr().cast());
            let m = _mm_set1_epi8(0x0F);
            let n = _mm256_set_m128i(
                _mm_and_si128(_mm_srli_epi16::<4>(b), m),
                _mm_and_si128(b, m),
            );
            _mm256_sub_epi8(n, _mm256_set1_epi8(8))
        };
        for (o, x) in out.iter_mut().zip(x) {
            *o = dot_i8(wv, load32_i8(x));
        }
    }

    #[inline(always)]
    fn q4k(&self, w: &[u8], x: &[&[i8]], out: &mut [[i32; 8]; ROW_GROUP]) {
        for g in 0..4 {
            let q = load32(&w[32 * g..]);
            // SAFETY: AVX2 is enabled in every caller (`Avx2`'s contract).
            let (lo, hi) = unsafe {
                let m = _mm256_set1_epi8(0x0F);
                (
                    _mm256_and_si256(q, m),
                    _mm256_and_si256(_mm256_srli_epi16::<4>(q), m),
                )
            };
            for (o, x) in out.iter_mut().zip(x) {
                o[2 * g] = dot_u4(lo, load32_i8(&x[64 * g..]));
                o[2 * g + 1] = dot_u4(hi, load32_i8(&x[64 * g + 32..]));
            }
        }
    }
}

/// Eight F16 (`F16 = true`) or BF16 weights from 16 bytes, widened to F32.
#[inline(always)]
fn widen8<const F16: bool>(b: &[u8]) -> std::arch::x86_64::__m256 {
    assert!(b.len() >= 16);
    // SAFETY: AVX2 and F16C are enabled in every caller; `b` holds 16 bytes (checked).
    unsafe {
        let h = _mm_loadu_si128(b.as_ptr().cast());
        if F16 {
            _mm256_cvtph_ps(h)
        } else {
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(h)))
        }
    }
}

/// The float kernel with four 256-bit accumulators.
#[target_feature(enable = "avx2,fma,f16c")]
fn float_rows_avx2<const F16: bool>(job: &Job<'_, '_>, y: &mut [f32]) {
    let k = job.w.k_in();
    let full = k / LANES * LANES;
    for o in job.out.clone() {
        let w = job.w.row(o);
        for r in 0..job.x.rows() {
            let x = job.x.f32_row(r);
            let mut acc = [_mm256_setzero_ps(); 4];
            for c in (0..full).step_by(LANES) {
                for (j, a) in acc.iter_mut().enumerate() {
                    let e = c + 8 * j;
                    let wv = widen8::<F16>(&w[2 * e..]);
                    let xs = &x[e..e + 8];
                    // SAFETY: `xs` holds 8 values; the load is unaligned.
                    let xv = unsafe { std::arch::x86_64::_mm256_loadu_ps(xs.as_ptr()) };
                    *a = _mm256_fmadd_ps(wv, xv, *a);
                }
            }
            let mut lanes = [0.0f32; LANES];
            for (j, a) in acc.iter().enumerate() {
                // SAFETY: `lanes[8j..8j + 8]` holds 8 values; the store is unaligned.
                unsafe { _mm256_storeu_ps(lanes[8 * j..].as_mut_ptr(), *a) };
            }
            tail::<F16>(w, x, full, k, &mut lanes);
            y[r * job.n_out + o] = fold_lanes(lanes);
        }
    }
}

/// The float kernel with two 512-bit accumulators.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma,f16c")]
fn float_rows_avx512<const F16: bool>(job: &Job<'_, '_>, y: &mut [f32]) {
    let k = job.w.k_in();
    let full = k / LANES * LANES;
    for o in job.out.clone() {
        let w = job.w.row(o);
        for r in 0..job.x.rows() {
            let x = job.x.f32_row(r);
            let mut acc = [_mm512_setzero_ps(); 2];
            for c in (0..full).step_by(LANES) {
                for (j, a) in acc.iter_mut().enumerate() {
                    let e = c + 16 * j;
                    let wb = &w[2 * e..2 * e + 32];
                    let xs = &x[e..e + 16];
                    // SAFETY: `wb` holds 32 bytes and `xs` 16 values; the loads are unaligned.
                    let (wv, xv) = unsafe {
                        let h = _mm256_loadu_si256(wb.as_ptr().cast());
                        let wv = if F16 {
                            _mm512_cvtph_ps(h)
                        } else {
                            _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(h)))
                        };
                        (wv, std::arch::x86_64::_mm512_loadu_ps(xs.as_ptr()))
                    };
                    *a = _mm512_fmadd_ps(wv, xv, *a);
                }
            }
            let mut lanes = [0.0f32; LANES];
            for (j, a) in acc.iter().enumerate() {
                // SAFETY: `lanes[16j..16j + 16]` holds 16 values; the store is unaligned.
                unsafe { _mm512_storeu_ps(lanes[16 * j..].as_mut_ptr(), *a) };
            }
            tail::<F16>(w, x, full, k, &mut lanes);
            y[r * job.n_out + o] = fold_lanes(lanes);
        }
    }
}

/// The elements past the last whole group of [`LANES`], into their lanes, as the portable
/// kernel adds them.
#[inline(always)]
fn tail<const F16: bool>(w: &[u8], x: &[f32], from: usize, k: usize, lanes: &mut [f32; LANES]) {
    for e in from..k {
        let h = u16::from_le_bytes([w[2 * e], w[2 * e + 1]]);
        let wv = if F16 {
            syntropy_kernels::reference::act_quant::f16_to_f32(h)
        } else {
            f32::from_bits(u32::from(h) << 16)
        };
        let l = &mut lanes[e % LANES];
        *l = wv.mul_add(x[e], *l);
    }
}
