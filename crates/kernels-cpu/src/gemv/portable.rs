// SPDX-License-Identifier: MIT OR Apache-2.0

//! The portable kernels: scalar Rust, and the definition of what every level computes (see
//! the parent module's docs).
//!
//! The integer kernels are written per block with one exact `i32` sum per (block, row), so the
//! SIMD kernels can share [`int_rows`] and supply only the block sums.

use super::{
    Job, LANES, QK_K, QK32, ROW_GROUP, WeightType, fold_lanes, half, q4k_term, q32_term,
    row_groups, scale_min_k4,
};

/// Runs `job` with the portable kernels.
pub(crate) fn run(job: &Job<'_, '_>, y: &mut [f32]) {
    match job.w.ty() {
        WeightType::F16 => float_rows(job, y, |b| {
            syntropy_kernels::reference::act_quant::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
        }),
        WeightType::Bf16 => float_rows(job, y, |b| {
            f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)
        }),
        WeightType::Q8_0 => int_rows(job, y, &Portable),
        WeightType::Q4_0 => int_rows(job, y, &Portable),
        WeightType::Q4K => int_rows(job, y, &Portable),
    }
}

/// The float kernel: [`LANES`] fused accumulators per row, folded pairwise.
fn float_rows(job: &Job<'_, '_>, y: &mut [f32], widen: impl Fn(&[u8]) -> f32) {
    let k = job.w.k_in();
    for o in job.out.clone() {
        let w = job.w.row(o);
        for r in 0..job.x.rows() {
            let x = job.x.f32_row(r);
            let mut lanes = [0.0f32; LANES];
            for (i, (wb, &xv)) in w.as_chunks::<2>().0.iter().zip(x).enumerate().take(k) {
                let l = &mut lanes[i % LANES];
                *l = widen(&wb[..]).mul_add(xv, *l);
            }
            y[r * job.n_out + o] = fold_lanes(lanes);
        }
    }
}

/// The exact block sums one level computes for the integer types. Each method returns the sum
/// for activation rows `rows` (at most [`ROW_GROUP`]) into `out[..rows.len()]`.
pub(crate) trait BlockSums {
    /// `Q8_0`: `Σ q_w · q_x` over one block (`w` its 32 codes, `x[r]` each row's 32 codes).
    fn q8_0(&self, w: &[u8], x: &[&[i8]], out: &mut [i32; ROW_GROUP]);
    /// `Q4_0`: `Σ (n − 8) · q_x` over one block (`w` its 16 code bytes).
    fn q4_0(&self, w: &[u8], x: &[&[i8]], out: &mut [i32; ROW_GROUP]);
    /// `Q4_K`: per sub-block `j` of one super-block (`w` its 128 code bytes), `Σ q · q_x`, into
    /// `out[r][j]`.
    fn q4k(&self, w: &[u8], x: &[&[i8]], out: &mut [[i32; 8]; ROW_GROUP]);
}

/// The scalar block sums.
pub(crate) struct Portable;

impl BlockSums for Portable {
    #[inline(always)]
    fn q8_0(&self, w: &[u8], x: &[&[i8]], out: &mut [i32; ROW_GROUP]) {
        for (o, x) in out.iter_mut().zip(x) {
            *o = w
                .iter()
                .zip(x.iter())
                .map(|(&w, &x)| i32::from(w as i8) * i32::from(x))
                .sum();
        }
    }

    #[inline(always)]
    fn q4_0(&self, w: &[u8], x: &[&[i8]], out: &mut [i32; ROW_GROUP]) {
        for (o, x) in out.iter_mut().zip(x) {
            let (lo, hi) = x.split_at(QK32 / 2);
            *o = w
                .iter()
                .zip(lo.iter().zip(hi))
                .map(|(&b, (&l, &h))| {
                    (i32::from(b & 0xF) - 8) * i32::from(l) + (i32::from(b >> 4) - 8) * i32::from(h)
                })
                .sum();
        }
    }

    #[inline(always)]
    fn q4k(&self, w: &[u8], x: &[&[i8]], out: &mut [[i32; 8]; ROW_GROUP]) {
        for (o, x) in out.iter_mut().zip(x) {
            // Four groups of 64 elements from 32 bytes each: low nibbles are sub-block 2g, high
            // nibbles 2g + 1.
            for g in 0..4 {
                let q = &w[32 * g..32 * g + 32];
                let xl = &x[64 * g..64 * g + 32];
                let xh = &x[64 * g + 32..64 * g + 64];
                o[2 * g] = q
                    .iter()
                    .zip(xl)
                    .map(|(&b, &v)| i32::from(b & 0xF) * i32::from(v))
                    .sum();
                o[2 * g + 1] = q
                    .iter()
                    .zip(xh)
                    .map(|(&b, &v)| i32::from(b >> 4) * i32::from(v))
                    .sum();
            }
        }
    }
}

/// The integer kernel over every output row of `job`, with `sums` computing the exact block
/// sums. Each output's terms are added in block order (parent module docs).
#[inline(always)]
pub(crate) fn int_rows(job: &Job<'_, '_>, y: &mut [f32], sums: &impl BlockSums) {
    let ty = job.w.ty();
    let nb = job.w.k_in() / ty.block_elems();
    let bytes = ty.block_bytes();
    for (r0, nr) in row_groups(job.x.rows()) {
        let mut xq: [&[i8]; ROW_GROUP] = [&[]; ROW_GROUP];
        let mut xd: [&[f32]; ROW_GROUP] = [&[]; ROW_GROUP];
        let mut xs: [&[i32]; ROW_GROUP] = [&[]; ROW_GROUP];
        for i in 0..nr {
            xq[i] = job.x.qs_row(r0 + i);
            xd[i] = job.x.d_row(r0 + i);
            if ty == WeightType::Q4K {
                xs[i] = job.x.sums_row(r0 + i);
            }
        }
        for o in job.out.clone() {
            let w = job.w.row(o);
            let mut acc = [0.0f32; ROW_GROUP];
            for b in 0..nb {
                let blk = &w[b * bytes..(b + 1) * bytes];
                let e = ty.block_elems();
                let mut xb: [&[i8]; ROW_GROUP] = [&[]; ROW_GROUP];
                for i in 0..nr {
                    xb[i] = &xq[i][b * e..(b + 1) * e];
                }
                let xb = &xb[..nr];
                match ty {
                    WeightType::Q8_0 => {
                        let mut s = [0i32; ROW_GROUP];
                        sums.q8_0(&blk[2..], xb, &mut s);
                        let dw = half(blk);
                        for i in 0..nr {
                            acc[i] += q32_term(dw, xd[i][b], s[i]);
                        }
                    }
                    WeightType::Q4_0 => {
                        let mut s = [0i32; ROW_GROUP];
                        sums.q4_0(&blk[2..], xb, &mut s);
                        let dw = half(blk);
                        for i in 0..nr {
                            acc[i] += q32_term(dw, xd[i][b], s[i]);
                        }
                    }
                    WeightType::Q4K => {
                        let mut s = [[0i32; 8]; ROW_GROUP];
                        sums.q4k(&blk[16..], xb, &mut s);
                        let (d, dmin) = (half(blk), half(&blk[2..]));
                        let scales = &blk[4..16];
                        let mut sc = [0i32; 8];
                        let mut mn = [0i32; 8];
                        for j in 0..8 {
                            let (a, m) = scale_min_k4(j, scales);
                            sc[j] = i32::from(a);
                            mn[j] = i32::from(m);
                        }
                        for i in 0..nr {
                            let xsum = &xs[i][b * (QK_K / QK32)..(b + 1) * (QK_K / QK32)];
                            let mut s1 = 0i32;
                            let mut s2 = 0i32;
                            for j in 0..8 {
                                s1 += sc[j] * s[i][j];
                                s2 += mn[j] * xsum[j];
                            }
                            acc[i] += q4k_term(d, dmin, xd[i][b], s1, s2);
                        }
                    }
                    WeightType::F16 | WeightType::Bf16 => {
                        unreachable!("float types use float_rows")
                    }
                }
            }
            for i in 0..nr {
                y[(r0 + i) * job.n_out + o] = acc[i];
            }
        }
    }
}
