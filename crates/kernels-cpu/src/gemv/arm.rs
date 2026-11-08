// SPDX-License-Identifier: MIT OR Apache-2.0

//! The AArch64 kernels: NEON with SDOT (`Arm82Dot`), and with SMMLA (`Arm86I8mm`).
//!
//! The integer block sums are exact `i32` sums, `sdot` four products at a time and `smmla` a
//! 2 × 2 tile of 8-element dot products: two output rows by two activation rows. They equal
//! the portable sums. `Arm86I8mm` takes output rows and activation rows in pairs, and an odd
//! last output row or activation row with `sdot`. The float kernels keep the 32 lanes in eight
//! 128-bit registers, element `k` in lane `k mod 32`, and fold them as the portable kernel
//! does. F16 is widened by `fcvtl` (stable Rust has no F16 vector type yet, so it is one
//! `asm!` instruction); BF16 by a 16-bit shift. `smmla` is likewise one `asm!` instruction:
//! its intrinsic, `vmmlaq_s32`, is still unstable (`stdarch_neon_i8mm`) on the pinned
//! toolchain.

use std::arch::aarch64::{
    float32x4_t, int8x16_t, int32x4_t, uint16x4_t, vaddvq_s32, vandq_u8, vcombine_s8, vdotq_s32,
    vdupq_n_f32, vdupq_n_s8, vdupq_n_s32, vdupq_n_u8, vfmaq_f32, vgetq_lane_s32, vld1_s8, vld1_u16,
    vld1q_f32, vld1q_s8, vld1q_u8, vreinterpretq_f32_u32, vreinterpretq_s8_u8, vshll_n_u16,
    vshrq_n_u8, vst1q_f32, vsubq_s8,
};

use super::portable::{BlockSums, int_rows};
use super::{
    Job, LANES, QK_K, QK32, ROW_GROUP, WeightType, fold_lanes, half, q4k_term, q32_term,
    row_groups, scale_min_k4,
};

/// Runs `job` with the `sdot` kernels.
///
/// # Safety
///
/// The host has the dot-product instructions.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn run_dot(job: &Job<'_, '_>, y: &mut [f32]) {
    match job.w.ty() {
        WeightType::F16 => float_rows::<true>(job, y),
        WeightType::Bf16 => float_rows::<false>(job, y),
        WeightType::Q8_0 | WeightType::Q4_0 | WeightType::Q4K => int_rows(job, y, &Dot(())),
    }
}

/// Runs `job` with the `smmla` kernels.
///
/// # Safety
///
/// The host has the dot-product and I8MM instructions.
#[target_feature(enable = "neon,dotprod,i8mm")]
pub(super) unsafe fn run_i8mm(job: &Job<'_, '_>, y: &mut [f32]) {
    match job.w.ty() {
        WeightType::F16 => float_rows::<true>(job, y),
        WeightType::Bf16 => float_rows::<false>(job, y),
        WeightType::Q8_0 | WeightType::Q4_0 | WeightType::Q4K => int_rows_mm(job, y),
    }
}

/// The `sdot` block sums. Only `run_dot` and `run_i8mm` make one, inside functions compiled
/// with the dot-product feature, so its methods are only ever inlined into code the host can
/// run.
struct Dot(());

/// 16 signed bytes from `p`.
#[inline(always)]
fn ld16(p: &[i8]) -> int8x16_t {
    assert!(p.len() >= 16);
    // SAFETY: `p` holds at least 16 bytes (checked); NEON is always present on AArch64.
    unsafe { vld1q_s8(p.as_ptr()) }
}

/// 16 bytes from `p`, as signed bytes.
#[inline(always)]
fn ld16u(p: &[u8]) -> int8x16_t {
    assert!(p.len() >= 16);
    // SAFETY: as `ld16`.
    unsafe { vreinterpretq_s8_u8(vld1q_u8(p.as_ptr())) }
}

/// The low and high nibbles of 16 bytes, as bytes 0–15.
#[inline(always)]
fn nibbles(p: &[u8]) -> (int8x16_t, int8x16_t) {
    assert!(p.len() >= 16);
    // SAFETY: as `ld16`.
    unsafe {
        let b = vld1q_u8(p.as_ptr());
        let m = vdupq_n_u8(0x0F);
        (
            vreinterpretq_s8_u8(vandq_u8(b, m)),
            vreinterpretq_s8_u8(vshrq_n_u8::<4>(b)),
        )
    }
}

/// `Σ a · b` over two pairs of 16 signed bytes, exact.
#[inline(always)]
fn dot2(a0: int8x16_t, b0: int8x16_t, a1: int8x16_t, b1: int8x16_t) -> i32 {
    // SAFETY: the dot-product feature is enabled in every caller (`Dot`'s contract).
    unsafe {
        let s = vdotq_s32(vdupq_n_s32(0), a0, b0);
        vaddvq_s32(vdotq_s32(s, a1, b1))
    }
}

/// The `Q4_0` codes of one block minus 8, as elements 0–15 and 16–31.
#[inline(always)]
fn q4_0_codes(w: &[u8]) -> (int8x16_t, int8x16_t) {
    let (lo, hi) = nibbles(w);
    // SAFETY: NEON is always present on AArch64.
    unsafe {
        let eight = vdupq_n_s8(8);
        (vsubq_s8(lo, eight), vsubq_s8(hi, eight))
    }
}

impl BlockSums for Dot {
    #[inline(always)]
    fn q8_0(&self, w: &[u8], x: &[&[i8]], out: &mut [i32; ROW_GROUP]) {
        let (w0, w1) = (ld16u(w), ld16u(&w[16..]));
        for (o, x) in out.iter_mut().zip(x) {
            *o = dot2(w0, ld16(x), w1, ld16(&x[16..]));
        }
    }

    #[inline(always)]
    fn q4_0(&self, w: &[u8], x: &[&[i8]], out: &mut [i32; ROW_GROUP]) {
        let (w0, w1) = q4_0_codes(w);
        for (o, x) in out.iter_mut().zip(x) {
            *o = dot2(w0, ld16(x), w1, ld16(&x[16..]));
        }
    }

    #[inline(always)]
    fn q4k(&self, w: &[u8], x: &[&[i8]], out: &mut [[i32; 8]; ROW_GROUP]) {
        for g in 0..4 {
            let (l0, h0) = nibbles(&w[32 * g..]);
            let (l1, h1) = nibbles(&w[32 * g + 16..]);
            for (o, x) in out.iter_mut().zip(x) {
                let x = &x[64 * g..];
                o[2 * g] = dot2(l0, ld16(x), l1, ld16(&x[16..]));
                o[2 * g + 1] = dot2(h0, ld16(&x[32..]), h1, ld16(&x[48..]));
            }
        }
    }
}

/// One block's codes, unpacked to signed bytes in element order, for `smmla`'s 8-byte loads.
/// `Q4_K`'s are its 256 codes `0..=15`; `Q8_0`'s and `Q4_0`'s their 32 (offset by 8).
#[inline(always)]
fn unpack(ty: WeightType, blk: &[u8], out: &mut [i8; QK_K]) {
    let codes: &[u8] = match ty {
        WeightType::Q8_0 => {
            for (o, &b) in out.iter_mut().zip(&blk[2..2 + QK32]) {
                *o = b as i8;
            }
            return;
        }
        WeightType::Q4_0 => &blk[2..2 + QK32 / 2],
        _ => &blk[16..16 + QK_K / 2],
    };
    if ty == WeightType::Q4_0 {
        for (j, &b) in codes.iter().enumerate() {
            out[j] = (b & 0xF) as i8 - 8;
            out[j + 16] = (b >> 4) as i8 - 8;
        }
    } else {
        for g in 0..4 {
            for l in 0..32 {
                let b = codes[32 * g + l];
                out[64 * g + l] = (b & 0xF) as i8;
                out[64 * g + 32 + l] = (b >> 4) as i8;
            }
        }
    }
}

/// `[w0·x0, w0·x1, w1·x0, w1·x1]` over `n` elements (a multiple of 8), exact, by `smmla`.
#[inline(always)]
fn mm2x2(w0: &[i8], w1: &[i8], x0: &[i8], x1: &[i8], n: usize) -> [i32; 4] {
    assert!(w0.len() >= n && w1.len() >= n && x0.len() >= n && x1.len() >= n);
    // SAFETY: each slice holds `n` bytes (checked) and every 8-byte load is below `n`; I8MM
    // is enabled in the one caller, `int_rows_mm`, which only `run_i8mm` reaches.
    unsafe {
        let mut acc: int32x4_t = vdupq_n_s32(0);
        for k in (0..n).step_by(8) {
            let a = vcombine_s8(vld1_s8(w0[k..].as_ptr()), vld1_s8(w1[k..].as_ptr()));
            let b = vcombine_s8(vld1_s8(x0[k..].as_ptr()), vld1_s8(x1[k..].as_ptr()));
            // `vmmlaq_s32` is unstable on the pinned toolchain (`stdarch_neon_i8mm`), so the
            // one instruction is written out.
            std::arch::asm!(
                "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        [
            vgetq_lane_s32::<0>(acc),
            vgetq_lane_s32::<1>(acc),
            vgetq_lane_s32::<2>(acc),
            vgetq_lane_s32::<3>(acc),
        ]
    }
}

/// `Σ w · x` over `n` elements (a multiple of 16), exact, by `sdot`.
#[inline(always)]
fn dot_n(w: &[i8], x: &[i8], n: usize) -> i32 {
    let mut s = 0;
    for k in (0..n).step_by(32) {
        s += dot2(
            ld16(&w[k..]),
            ld16(&x[k..]),
            ld16(&w[k + 16..]),
            ld16(&x[k + 16..]),
        );
    }
    s
}

/// The integer kernel at `Arm86I8mm`: output rows in pairs by `smmla`, with an odd last output
/// row through [`int_rows`] and the `sdot` sums. Each output's terms are the portable
/// kernel's, added in the same order.
#[target_feature(enable = "neon,dotprod,i8mm")]
#[allow(
    clippy::needless_range_loop,
    reason = "the sub-block index addresses the codes of two weight rows and two activation rows at once"
)]
fn int_rows_mm(job: &Job<'_, '_>, y: &mut [f32]) {
    let ty = job.w.ty();
    let (e, bytes) = (ty.block_elems(), ty.block_bytes());
    let nb = job.w.k_in() / e;
    // Sub-blocks of one block: 1 for the 32-element types, 8 for `Q4_K`.
    let subs = e / QK32;
    let pairs_end = job.out.start + (job.out.len() / 2) * 2;
    let mut w0 = [0i8; QK_K];
    let mut w1 = [0i8; QK_K];
    for (r0, nr) in row_groups(job.x.rows()) {
        for o in (job.out.start..pairs_end).step_by(2) {
            let (row0, row1) = (job.w.row(o), job.w.row(o + 1));
            let mut acc = [[0.0f32; ROW_GROUP]; 2];
            for b in 0..nb {
                let (b0, b1) = (
                    &row0[b * bytes..(b + 1) * bytes],
                    &row1[b * bytes..(b + 1) * bytes],
                );
                unpack(ty, b0, &mut w0);
                unpack(ty, b1, &mut w1);
                // s[p][i][j]: output o + p, activation row r0 + i, sub-block j.
                let mut s = [[[0i32; 8]; ROW_GROUP]; 2];
                let mut i = 0;
                while i < nr {
                    let x0 = &job.x.qs_row(r0 + i)[b * e..(b + 1) * e];
                    if i + 1 < nr {
                        let x1 = &job.x.qs_row(r0 + i + 1)[b * e..(b + 1) * e];
                        for j in 0..subs {
                            let sl = j * QK32..(j + 1) * QK32;
                            let t = mm2x2(
                                &w0[sl.clone()],
                                &w1[sl.clone()],
                                &x0[sl.clone()],
                                &x1[sl],
                                QK32,
                            );
                            s[0][i][j] = t[0];
                            s[0][i + 1][j] = t[1];
                            s[1][i][j] = t[2];
                            s[1][i + 1][j] = t[3];
                        }
                        i += 2;
                    } else {
                        for j in 0..subs {
                            let sl = j * QK32..(j + 1) * QK32;
                            s[0][i][j] = dot_n(&w0[sl.clone()], &x0[sl.clone()], QK32);
                            s[1][i][j] = dot_n(&w1[sl.clone()], &x0[sl], QK32);
                        }
                        i += 1;
                    }
                }
                for (p, blk) in [b0, b1].into_iter().enumerate() {
                    for i in 0..nr {
                        let dx = job.x.d_row(r0 + i)[b];
                        let t = if ty == WeightType::Q4K {
                            let scales = &blk[4..16];
                            let xsum = &job.x.sums_row(r0 + i)[b * 8..b * 8 + 8];
                            let (mut s1, mut s2) = (0i32, 0i32);
                            for j in 0..8 {
                                let (sc, m) = scale_min_k4(j, scales);
                                s1 += i32::from(sc) * s[p][i][j];
                                s2 += i32::from(m) * xsum[j];
                            }
                            q4k_term(half(blk), half(&blk[2..]), dx, s1, s2)
                        } else {
                            q32_term(half(blk), dx, s[p][i][0])
                        };
                        acc[p][i] += t;
                    }
                }
            }
            for p in 0..2 {
                for i in 0..nr {
                    y[(r0 + i) * job.n_out + o + p] = acc[p][i];
                }
            }
        }
    }
    if pairs_end < job.out.end {
        let last = Job {
            w: job.w,
            x: job.x,
            out: pairs_end..job.out.end,
            n_out: job.n_out,
        };
        int_rows(&last, y, &Dot(()));
    }
}

/// Four weights from 8 bytes, widened to F32.
#[inline(always)]
fn widen4<const F16: bool>(b: &[u8]) -> float32x4_t {
    assert!(b.len() >= 8);
    // SAFETY: `b` holds 8 bytes (checked); the load is unaligned-safe for `u16` lanes because
    // `vld1_u16` has no alignment requirement beyond the element's, and a `u8` pointer cast is
    // read as four little-endian `u16`s. `fcvtl` is base AArch64 floating point and touches
    // only the two registers named.
    unsafe {
        let h: uint16x4_t = vld1_u16(b.as_ptr().cast());
        if F16 {
            let out: float32x4_t;
            std::arch::asm!(
                "fcvtl {o:v}.4s, {i:v}.4h",
                o = lateout(vreg) out,
                i = in(vreg) h,
                options(pure, nomem, nostack, preserves_flags)
            );
            out
        } else {
            vreinterpretq_f32_u32(vshll_n_u16::<16>(h))
        }
    }
}

/// The float kernel with eight 128-bit accumulators.
#[target_feature(enable = "neon")]
fn float_rows<const F16: bool>(job: &Job<'_, '_>, y: &mut [f32]) {
    let k = job.w.k_in();
    let full = k / LANES * LANES;
    for o in job.out.clone() {
        let w = job.w.row(o);
        for r in 0..job.x.rows() {
            let x = job.x.f32_row(r);
            let mut acc = [vdupq_n_f32(0.0); 8];
            for c in (0..full).step_by(LANES) {
                for (j, a) in acc.iter_mut().enumerate() {
                    let e = c + 4 * j;
                    let wv = widen4::<F16>(&w[2 * e..]);
                    let xs = &x[e..e + 4];
                    // SAFETY: `xs` holds 4 values.
                    *a = unsafe { vfmaq_f32(*a, wv, vld1q_f32(xs.as_ptr())) };
                }
            }
            let mut lanes = [0.0f32; LANES];
            for (j, a) in acc.iter().enumerate() {
                // SAFETY: `lanes[4j..4j + 4]` holds 4 values.
                unsafe { vst1q_f32(lanes[4 * j..].as_mut_ptr(), *a) };
            }
            for e in full..k {
                let h = u16::from_le_bytes([w[2 * e], w[2 * e + 1]]);
                let wv = if F16 {
                    syntropy_kernels::reference::act_quant::f16_to_f32(h)
                } else {
                    f32::from_bits(u32::from(h) << 16)
                };
                let l = &mut lanes[e % LANES];
                *l = wv.mul_add(x[e], *l);
            }
            y[r * job.n_out + o] = fold_lanes(lanes);
        }
    }
}
