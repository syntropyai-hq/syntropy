// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MoeGrouped` at the `CPU` site against `kernels::reference`, across levels, under row
//! permutations, and with the weight-byte counter (COMP-E3-T05's acceptance criteria).
//!
//! - **Conformance, per scheme** (`Q4_K`, `Q8_0`, and `Q4_K` gate and up with a `Q8_0` down).
//!   Every slot equals the expert's FFN of its row computed alone, bit for bit. That FFN's `h`
//!   is within a derived bound of `reference::gate_up_act` on the dequantised gate and up
//!   matrices and the quantised input, and each output is within a derived bound of
//!   `reference::gemv` on the dequantised down matrix and the quantised `h`. Every level equals
//!   the portable kernels bit for bit.
//! - **Counter.** One pass adds each routed expert's gate, up and down bytes to its counter
//!   exactly once, at every row count, and nothing for an expert no row chose.
//! - **Row order.** With every expert at the CPU site, permuting the rows permutes the
//!   per-(row, rank) outputs bit for bit, and the layer output (the outputs weighted by the
//!   router weights in ascending expert index, as `MoeCombine` adds them) with them.
//! - Site masks, refusals and edge cases.
//!
//! **The bounds** are the `gemv` tests' first-order bounds \[estimate\]: with `u = 2^-24`, a sum
//! of `n` F32 terms is within `n · u · Σ|terms|` of the exact sum, so a kernel and a reference
//! value of one dot product differ by at most `(n_kernel + n_ref) · u · Σ|w|·|x|`. For `h`,
//! SiLU's slope is below 1.1 in magnitude, so an error `δg` in the gate moves `silu(g)` by at
//! most `1.1 · δg`, plus the product's one rounding on each side. They are derived, not
//! calibrated: `numerics/tolerances-v1.toml` publishes no op tolerance for `MoeGrouped` yet
//! (QUAL-E4-T04), and the last test checks it once it does.

use syntropy_kernels::CpuIsa;
use syntropy_kernels::GatedAct;
use syntropy_kernels::reference::act_quant::{
    Q8Path, f16_bits, f16_to_f32, quantize_q8_0, quantize_q8_k,
};
use syntropy_kernels::reference::dequant::dequantize;
use syntropy_kernels::reference::math::silu;
use syntropy_kernels::reference::{Matrix, gate_up_act, gemv as ref_gemv};

use super::*;
use crate::gemv::{Activations, half, scale_min_k4};

const U: f64 = 1.0 / (1u64 << 24) as f64;

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// A positive F16 scale in roughly `[2^-9, 2^-5]`, as bits.
    fn scale(&mut self) -> [u8; 2] {
        let e = (self.next() % 5) as i32 - 9;
        f16_bits((0.5 + self.unit().abs() * 0.5) * 2f32.powi(e)).to_le_bytes()
    }
}

/// `n_out` random rows of `k` weights of `ty` (`Q8_0` or `Q4_K`).
fn matrix(rng: &mut Rng, ty: WeightType, n_out: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for _ in 0..n_out * k / ty.block_elems() {
        let scales = if ty == WeightType::Q4K { 2 } else { 1 };
        for _ in 0..scales {
            out.extend(rng.scale());
        }
        out.extend((2 * scales..ty.block_bytes()).map(|_| rng.next() as u8));
    }
    out
}

/// One layer's experts, stacked by expert as GGUF stores them.
struct Layer {
    shape: MoeShape,
    gate_ty: WeightType,
    down_ty: WeightType,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

impl Layer {
    fn new(rng: &mut Rng, shape: MoeShape, gate_ty: WeightType, down_ty: WeightType) -> Self {
        let (n, h, i) = (shape.n_experts, shape.hidden, shape.inter);
        Self {
            shape,
            gate_ty,
            down_ty,
            gate: matrix(rng, gate_ty, n * i, h),
            up: matrix(rng, gate_ty, n * i, h),
            down: matrix(rng, down_ty, n * h, i),
        }
    }

    fn gate_bytes(&self) -> usize {
        self.gate.len() / self.shape.n_experts
    }

    fn down_bytes(&self) -> usize {
        self.down.len() / self.shape.n_experts
    }

    fn expert(&self, e: u32) -> Expert<'_> {
        let MoeShape { hidden, inter, .. } = self.shape;
        let (e, gb, db) = (e as usize, self.gate_bytes(), self.down_bytes());
        Expert::new(
            Weights::new(
                self.gate_ty,
                &self.gate[e * gb..(e + 1) * gb],
                inter,
                hidden,
            )
            .unwrap(),
            Weights::new(self.gate_ty, &self.up[e * gb..(e + 1) * gb], inter, hidden).unwrap(),
            Weights::new(
                self.down_ty,
                &self.down[e * db..(e + 1) * db],
                hidden,
                inter,
            )
            .unwrap(),
        )
        .unwrap()
    }

    /// Runs the op at `level` over every expert, starting from an output of `fill`.
    fn run(&self, level: Level, x: &[f32], ids: &[i32], rows: usize) -> Vec<f32> {
        self.run_with(level, x, ids, rows, |_| true, None)
    }

    fn run_with(
        &self,
        level: Level,
        x: &[f32],
        ids: &[i32],
        rows: usize,
        here: impl Fn(u32) -> bool,
        reads: Option<&WeightReads>,
    ) -> Vec<f32> {
        let s = &self.shape;
        let mut out = vec![f32::from_bits(0x7FC0_DEAD); rows * s.top_k * s.hidden];
        let mut ws = vec![0u8; workspace_bytes(s, rows)];
        moe_grouped(
            level,
            s,
            |e| here(e).then(|| self.expert(e)),
            x,
            ids,
            rows,
            &mut out,
            &mut ws,
            reads,
        )
        .unwrap();
        out
    }
}

/// `rows` rows of `top_k` distinct experts each.
fn route(rng: &mut Rng, rows: usize, top_k: usize, n_experts: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity(rows * top_k);
    for _ in 0..rows {
        let mut row: Vec<i32> = Vec::new();
        while row.len() < top_k {
            let e = (rng.next() % n_experts as u64) as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids.extend(row);
    }
    ids
}

fn activations(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.unit() * 2.0).collect()
}

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn assert_same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            same(g, w),
            "{what}: value {i}: {g:e} ({:#x}) != {w:e} ({:#x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

/// Every level this host can run, the portable one first.
fn levels() -> Vec<Level> {
    let mut v = vec![Level::Portable];
    #[cfg(target_arch = "x86_64")]
    let isas = [CpuIsa::X86V3, CpuIsa::X86V4];
    #[cfg(target_arch = "aarch64")]
    let isas = [CpuIsa::Arm82Dot, CpuIsa::Arm86I8mm];
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let isas: [CpuIsa; 0] = [];
    for isa in isas {
        let l = Level::for_isa(isa);
        if !v.contains(&l) {
            v.push(l);
        }
    }
    v
}

/// The schemes under test: (gate and up, down).
const SCHEMES: [(WeightType, WeightType); 3] = [
    (WeightType::Q4K, WeightType::Q4K),
    (WeightType::Q8_0, WeightType::Q8_0),
    (WeightType::Q4K, WeightType::Q8_0),
];

/// A shape for the scheme whose matrices span several task blocks.
fn shape_for(gate_ty: WeightType, down_ty: WeightType) -> MoeShape {
    let q4k = gate_ty == WeightType::Q4K || down_ty == WeightType::Q4K;
    MoeShape {
        n_experts: 6,
        top_k: 2,
        hidden: 256,
        inter: if q4k { 512 } else { 160 },
    }
}

/// The values a product multiplies for one row: `x` as quantised for `ty`.
fn seen(ty: WeightType, x: &[f32]) -> Vec<f32> {
    match ty {
        WeightType::Q8_0 => quantize_q8_0(x, Q8Path::Arm64Neon)
            .unwrap()
            .iter()
            .flat_map(|b| {
                let d = f16_to_f32(b.d);
                b.qs.map(|q| f32::from(q) * d)
            })
            .collect(),
        WeightType::Q4K => quantize_q8_k(x)
            .unwrap()
            .iter()
            .flat_map(|b| b.qs.map(|q| f32::from(q) * b.d))
            .collect(),
        _ => unreachable!("the schemes under test"),
    }
}

/// The weights in F64 as two parts whose sum is the weight (`Q4_K`: `d·sc·q` and `−dmin·m`),
/// for magnitudes that see through `Q4_K`'s cancellation.
fn exact_parts(ty: WeightType, w: &[u8]) -> Vec<(f64, f64)> {
    if ty != WeightType::Q4K {
        return dequantize(ty.ggml(), w)
            .unwrap()
            .into_iter()
            .map(|v| (f64::from(v), 0.0))
            .collect();
    }
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

/// The bound on a kernel and a reference value of the dot product of row `o` of `parts` with
/// `x` (module docs).
fn dot_bound(ty: WeightType, parts: &[(f64, f64)], o: usize, x: &[f32]) -> f64 {
    let k = x.len();
    let kernel_n = (k / ty.block_elems() + 5) as f64;
    let ref_n = (32 + 64 + 3) as f64;
    let mag: f64 = parts[o * k..(o + 1) * k]
        .iter()
        .zip(x)
        .map(|(&(a, b), &xv)| (a.abs() + b.abs()) * f64::from(xv).abs())
        .sum();
    (kernel_n + ref_n) * U * mag + 1e-38
}

/// One expert's FFN of one row, alone, with the `gemv` module's portable kernels: the value
/// every slot must equal.
fn ffn_alone(ex: &Expert<'_>, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (inter, hidden) = (ex.gate.n_out(), ex.gate.k_in());
    let mut a = Activations::new();
    a.quantize(ex.gate.ty().act(), x, 1, hidden).unwrap();
    let (mut g, mut u) = (vec![0.0; inter], vec![0.0; inter]);
    gemv(Level::Portable, &ex.gate, &a.view(), 0..inter, &mut g).unwrap();
    gemv(Level::Portable, &ex.up, &a.view(), 0..inter, &mut u).unwrap();
    let h: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
    a.quantize(ex.down.ty().act(), &h, 1, inter).unwrap();
    let mut y = vec![0.0; hidden];
    gemv(Level::Portable, &ex.down, &a.view(), 0..hidden, &mut y).unwrap();
    (h, y)
}

#[test]
fn every_slot_equals_its_experts_ffn_of_the_row_alone() {
    let mut rng = Rng(1);
    for (gt, dt) in SCHEMES {
        let layer = Layer::new(&mut rng, shape_for(gt, dt), gt, dt);
        let s = layer.shape;
        for rows in [1, 2, 5, 8, 9, 17] {
            let x = activations(&mut rng, rows * s.hidden);
            let ids = route(&mut rng, rows, s.top_k, s.n_experts);
            let out = layer.run(Level::Portable, &x, &ids, rows);
            for r in 0..rows {
                for j in 0..s.top_k {
                    let ex = layer.expert(ids[r * s.top_k + j] as u32);
                    let (_, want) = ffn_alone(&ex, &x[r * s.hidden..(r + 1) * s.hidden]);
                    let at = (r * s.top_k + j) * s.hidden;
                    assert_same(
                        &out[at..at + s.hidden],
                        &want,
                        &format!("{gt:?}/{dt:?} rows {rows} row {r} rank {j}"),
                    );
                }
            }
        }
    }
}

#[test]
fn conforms_to_the_reference_for_each_scheme() {
    let mut rng = Rng(2);
    for (gt, dt) in SCHEMES {
        let layer = Layer::new(&mut rng, shape_for(gt, dt), gt, dt);
        let MoeShape {
            n_experts,
            hidden,
            inter,
            ..
        } = layer.shape;
        let rows = 3;
        let x = activations(&mut rng, rows * hidden);
        for e in 0..n_experts as u32 {
            let ex = layer.expert(e);
            let (gb, db) = (layer.gate_bytes(), layer.down_bytes());
            let e = e as usize;
            let (gw, uw, dw) = (
                &layer.gate[e * gb..(e + 1) * gb],
                &layer.up[e * gb..(e + 1) * gb],
                &layer.down[e * db..(e + 1) * db],
            );
            let (gd, ud, dd) = (
                dequantize(gt.ggml(), gw).unwrap(),
                dequantize(gt.ggml(), uw).unwrap(),
                dequantize(dt.ggml(), dw).unwrap(),
            );
            let (gp, up, dp) = (
                exact_parts(gt, gw),
                exact_parts(gt, uw),
                exact_parts(dt, dw),
            );
            for r in 0..rows {
                let xr = &x[r * hidden..(r + 1) * hidden];
                let (h, y) = ffn_alone(&ex, xr);
                // Stage 1: h against `GateUpAct` on the same quantised input.
                let xs = seen(gt, xr);
                let mut h_ref = vec![0.0f32; inter];
                gate_up_act(
                    GatedAct::Silu,
                    Matrix::new(&gd, inter, hidden).unwrap(),
                    Matrix::new(&ud, inter, hidden).unwrap(),
                    &xs,
                    1,
                    &mut h_ref,
                )
                .unwrap();
                for i in 0..inter {
                    let (bg, bu) = (dot_bound(gt, &gp, i, &xs), dot_bound(gt, &up, i, &xs));
                    let (mut gx, mut ux) = (0.0f64, 0.0f64);
                    for k in 0..hidden {
                        gx += f64::from(gd[i * hidden + k]) * f64::from(xs[k]);
                        ux += f64::from(ud[i * hidden + k]) * f64::from(xs[k]);
                    }
                    let sg = f64::from(silu(gx as f32)).abs() + 1.1 * bg;
                    let bound = 1.1 * bg * (ux.abs() + bu)
                        + sg * bu
                        + 4.0 * U * (f64::from(h[i]).abs() + f64::from(h_ref[i]).abs())
                        + 1e-38;
                    let d = (f64::from(h[i]) - f64::from(h_ref[i])).abs();
                    assert!(
                        d <= bound,
                        "{gt:?}/{dt:?} expert {e} row {r} h[{i}]: {:e} vs {:e}, bound {bound:e}",
                        h[i],
                        h_ref[i]
                    );
                }
                // Stage 2: the output against `Gemv` on the quantised h.
                let hs = seen(dt, &h);
                let mut y_ref = vec![0.0f32; hidden];
                ref_gemv(Matrix::new(&dd, hidden, inter).unwrap(), &hs, 1, &mut y_ref).unwrap();
                for o in 0..hidden {
                    let bound = dot_bound(dt, &dp, o, &hs);
                    let d = (f64::from(y[o]) - f64::from(y_ref[o])).abs();
                    assert!(
                        d <= bound,
                        "{gt:?}/{dt:?} expert {e} row {r} out {o}: {:e} vs {:e}, bound {bound:e}",
                        y[o],
                        y_ref[o]
                    );
                }
            }
        }
    }
}

#[test]
fn every_level_equals_the_portable_kernels() {
    let mut rng = Rng(3);
    for (gt, dt) in SCHEMES {
        let layer = Layer::new(&mut rng, shape_for(gt, dt), gt, dt);
        let s = layer.shape;
        for rows in [1, 3, 8, 9, 16] {
            let x = activations(&mut rng, rows * s.hidden);
            let ids = route(&mut rng, rows, s.top_k, s.n_experts);
            let want = layer.run(Level::Portable, &x, &ids, rows);
            for level in levels() {
                let got = layer.run(level, &x, &ids, rows);
                assert_same(&got, &want, &format!("{gt:?}/{dt:?} rows {rows} {level:?}"));
            }
        }
    }
}

#[test]
fn each_routed_expert_is_read_once_per_pass() {
    let mut rng = Rng(4);
    for (gt, dt) in SCHEMES {
        let layer = Layer::new(&mut rng, shape_for(gt, dt), gt, dt);
        let s = layer.shape;
        // From one row to many rows piling onto two experts (several row groups each).
        for rows in [1, 4, 8, 9, 33] {
            let mut ids = route(&mut rng, rows, s.top_k, s.n_experts);
            if rows > 8 {
                for (i, id) in ids.iter_mut().enumerate() {
                    *id = if i % 2 == 0 { 1 } else { 4 };
                }
            }
            let x = activations(&mut rng, rows * s.hidden);
            let reads = WeightReads::new(s.n_experts);
            for pass in 1..=2u64 {
                layer.run_with(Level::Portable, &x, &ids, rows, |_| true, Some(&reads));
                for e in 0..s.n_experts {
                    let routed = ids.contains(&(e as i32));
                    let want = if routed {
                        pass * layer.expert(e as u32).bytes() as u64
                    } else {
                        0
                    };
                    assert_eq!(
                        reads.get(e),
                        want,
                        "{gt:?}/{dt:?} rows {rows} pass {pass} expert {e}"
                    );
                }
            }
            reads.reset();
            assert!((0..s.n_experts).all(|e| reads.get(e) == 0));
        }
        // The tasks cover each matrix's rows exactly once, in blocks of at most BLOCK_BYTES.
        let ex = layer.expert(0);
        let tasks: Vec<Task> = expert_tasks(0, &ex).collect();
        for (phase, w) in [(Phase::GateUp, &ex.gate), (Phase::Down, &ex.down)] {
            let mut next = 0;
            for t in tasks.iter().filter(|t| t.phase == phase) {
                assert_eq!(t.out.start, next);
                assert!(t.out.len() * w.row_bytes() <= BLOCK_BYTES.max(w.row_bytes()));
                next = t.out.end;
            }
            assert_eq!(next, w.n_out(), "{phase:?}");
        }
        assert!(tasks.iter().filter(|t| t.phase == Phase::GateUp).count() > 1);
        assert!(tasks.iter().filter(|t| t.phase == Phase::Down).count() > 1);
    }
}

#[test]
fn row_order_does_not_change_the_layer_output() {
    let mut rng = Rng(5);
    for (gt, dt) in SCHEMES {
        let layer = Layer::new(&mut rng, shape_for(gt, dt), gt, dt);
        let s = layer.shape;
        for rows in [2, 7, 12] {
            let x = activations(&mut rng, rows * s.hidden);
            let ids = route(&mut rng, rows, s.top_k, s.n_experts);
            let weights: Vec<f32> = (0..rows * s.top_k).map(|_| rng.unit().abs()).collect();
            // A permutation: reversed, then rotated.
            let perm: Vec<usize> = (0..rows).rev().cycle().skip(rows / 3).take(rows).collect();
            let (mut xp, mut idp, mut wp) = (Vec::new(), Vec::new(), Vec::new());
            for &r in &perm {
                xp.extend_from_slice(&x[r * s.hidden..(r + 1) * s.hidden]);
                idp.extend_from_slice(&ids[r * s.top_k..(r + 1) * s.top_k]);
                wp.extend_from_slice(&weights[r * s.top_k..(r + 1) * s.top_k]);
            }
            for level in levels() {
                let out = layer.run(level, &x, &ids, rows);
                let outp = layer.run(level, &xp, &idp, rows);
                let (layer_out, layer_outp) = (
                    combine(&s, &out, &ids, &weights, rows),
                    combine(&s, &outp, &idp, &wp, rows),
                );
                let slot = s.top_k * s.hidden;
                for (i, &r) in perm.iter().enumerate() {
                    let what = format!("{gt:?}/{dt:?} rows {rows} {level:?} row {r}");
                    assert_same(
                        &outp[i * slot..(i + 1) * slot],
                        &out[r * slot..(r + 1) * slot],
                        &what,
                    );
                    assert_same(
                        &layer_outp[i * s.hidden..(i + 1) * s.hidden],
                        &layer_out[r * s.hidden..(r + 1) * s.hidden],
                        &what,
                    );
                }
            }
        }
    }
}

/// The layer output as `MoeCombine` adds it (RFC-0040 §3.4): per row, the slots' outputs
/// weighted by their router weights, in ascending expert index, in F32.
fn combine(s: &MoeShape, out: &[f32], ids: &[i32], w: &[f32], rows: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * s.hidden];
    for r in 0..rows {
        let mut order: Vec<usize> = (0..s.top_k).collect();
        order.sort_by_key(|&j| ids[r * s.top_k + j]);
        for j in order {
            let slot = &out[(r * s.top_k + j) * s.hidden..(r * s.top_k + j + 1) * s.hidden];
            for (yv, &v) in y[r * s.hidden..(r + 1) * s.hidden].iter_mut().zip(slot) {
                *yv += w[r * s.top_k + j] * v;
            }
        }
    }
    y
}

#[test]
fn slots_of_other_sites_keep_their_values() {
    let mut rng = Rng(6);
    let (gt, dt) = SCHEMES[1];
    let layer = Layer::new(&mut rng, shape_for(gt, dt), gt, dt);
    let s = layer.shape;
    let rows = 6;
    let x = activations(&mut rng, rows * s.hidden);
    let ids = route(&mut rng, rows, s.top_k, s.n_experts);
    let all = layer.run(Level::Portable, &x, &ids, rows);
    let reads = WeightReads::new(s.n_experts);
    let even = layer.run_with(
        Level::Portable,
        &x,
        &ids,
        rows,
        |e| e % 2 == 0,
        Some(&reads),
    );
    for (slot, &id) in ids.iter().enumerate() {
        let span = slot * s.hidden..(slot + 1) * s.hidden;
        if id % 2 == 0 {
            assert_same(&even[span.clone()], &all[span], "computed slot");
        } else {
            assert!(
                even[span].iter().all(|v| v.to_bits() == 0x7FC0_DEAD),
                "slot {slot} of expert {id} was written"
            );
        }
    }
    for e in (1..s.n_experts).step_by(2) {
        assert_eq!(reads.get(e), 0, "expert {e} of another site was read");
    }
}

#[test]
fn refusals_write_nothing() {
    let mut rng = Rng(7);
    let (gt, dt) = SCHEMES[1];
    let layer = Layer::new(&mut rng, shape_for(gt, dt), gt, dt);
    let s = layer.shape;
    let rows = 3;
    let x = activations(&mut rng, rows * s.hidden);
    let mut out = vec![1.5f32; rows * s.top_k * s.hidden];
    let mut ws = vec![0u8; workspace_bytes(&s, rows)];
    let call = |ids: &[i32], out: &mut [f32], ws: &mut [u8], s: &MoeShape| {
        moe_grouped(
            Level::Portable,
            s,
            |e| Some(layer.expert(e)),
            &x,
            ids,
            rows,
            out,
            ws,
            None,
        )
    };
    for bad in [-1, s.n_experts as i32] {
        let mut ids = route(&mut rng, rows, s.top_k, s.n_experts);
        ids[3] = bad;
        assert_eq!(
            call(&ids, &mut out, &mut ws, &s),
            Err(MoeError::ExpertId {
                row: 1,
                rank: 1,
                id: bad
            })
        );
    }
    let ids = route(&mut rng, rows, s.top_k, s.n_experts);
    let short = workspace_bytes(&s, rows) - 1;
    assert!(matches!(
        call(&ids, &mut out, &mut ws[..short], &s),
        Err(MoeError::Len {
            arg: "workspace",
            ..
        })
    ));
    assert!(matches!(
        call(&ids, &mut out[..10], &mut ws, &s),
        Err(MoeError::Len { arg: "out", .. })
    ));
    // An expert of another width.
    let wide = MoeShape {
        inter: s.inter + 32,
        ..s
    };
    let mut ws_wide = vec![0u8; workspace_bytes(&wide, rows)];
    assert_eq!(
        call(&ids, &mut out, &mut ws_wide, &wide),
        Err(MoeError::Expert)
    );
    assert!(out.iter().all(|&v| v == 1.5));
    // Gate and up of different types, and a down of the wrong shape, are not one expert.
    let ex = layer.expert(0);
    let q4k = matrix(&mut rng, WeightType::Q4K, s.inter, s.hidden);
    let q4k = Weights::new(WeightType::Q4K, &q4k, s.inter, s.hidden).unwrap();
    assert_eq!(
        Expert::new(ex.gate, q4k, ex.down).unwrap_err(),
        MoeError::Expert
    );
    assert_eq!(
        Expert::new(ex.gate, ex.up, ex.gate).unwrap_err(),
        MoeError::Expert
    );
}

#[test]
fn no_rows_and_misaligned_workspaces() {
    let mut rng = Rng(8);
    let (gt, dt) = SCHEMES[0];
    let layer = Layer::new(&mut rng, shape_for(gt, dt), gt, dt);
    let s = layer.shape;
    // No rows: nothing is read or written.
    let mut out = vec![2.0f32; 4];
    let reads = WeightReads::new(s.n_experts);
    moe_grouped(
        Level::Portable,
        &s,
        |e| Some(layer.expert(e)),
        &[],
        &[],
        0,
        &mut out,
        &mut [0u8; 64],
        Some(&reads),
    )
    .unwrap();
    assert!(out.iter().all(|&v| v == 2.0));
    assert!((0..s.n_experts).all(|e| reads.get(e) == 0));
    // A workspace at any byte offset gives the same output.
    let rows = 5;
    let x = activations(&mut rng, rows * s.hidden);
    let ids = route(&mut rng, rows, s.top_k, s.n_experts);
    let want = layer.run(Level::Portable, &x, &ids, rows);
    let need = workspace_bytes(&s, rows);
    let mut big = vec![0u8; need + 3];
    for skew in 0..4 {
        let mut out = vec![0.0f32; want.len()];
        moe_grouped(
            Level::Portable,
            &s,
            |e| Some(layer.expert(e)),
            &x,
            &ids,
            rows,
            &mut out,
            &mut big[skew..skew + need],
            None,
        )
        .unwrap();
        assert_same(&out, &want, &format!("skew {skew}"));
    }
}

/// The dtype class whose op-level tolerance applies to a scheme (RFC-0042 §6.4).
fn dtype_class(ty: WeightType) -> syntropy_core::quant::DTypeClass {
    use syntropy_core::quant::DTypeClass;
    match ty {
        WeightType::Q8_0 => DTypeClass::Int8,
        _ => DTypeClass::Bits4,
    }
}

/// Where the embedded tolerance table publishes `MoeGrouped`'s op-level tolerance for a
/// scheme's class, the kernel's `rel_l2` against the reference chain (`GateUpAct` then `Gemv`
/// on the dequantised weights, with the activations quantised as the kernel quantises them)
/// stays within its `tol_ref` (RFC-0042 §6.4). Until QUAL-E4-T04 publishes the entries, none
/// exists and this test has nothing to compare.
#[test]
fn published_op_tolerances_hold() {
    use syntropy_kernels::tolerances::ToleranceTable;
    let table = ToleranceTable::release().unwrap();
    let mut rng = Rng(9);
    for (gt, dt) in [SCHEMES[0], SCHEMES[1]] {
        let Some(tol) = table
            .op("MoeGrouped", dtype_class(gt))
            .and_then(|e| e.value())
            .map(|t| t.tol_ref)
        else {
            continue;
        };
        let layer = Layer::new(&mut rng, shape_for(gt, dt), gt, dt);
        let s = layer.shape;
        let rows = 4;
        let x = activations(&mut rng, rows * s.hidden);
        let ids = route(&mut rng, rows, s.top_k, s.n_experts);
        let out = layer.run(Level::Portable, &x, &ids, rows);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (slot, &e) in ids.iter().enumerate() {
            let r = slot / s.top_k;
            let e = e as usize;
            let (gb, db) = (layer.gate_bytes(), layer.down_bytes());
            let gd = dequantize(gt.ggml(), &layer.gate[e * gb..(e + 1) * gb]).unwrap();
            let ud = dequantize(gt.ggml(), &layer.up[e * gb..(e + 1) * gb]).unwrap();
            let dd = dequantize(dt.ggml(), &layer.down[e * db..(e + 1) * db]).unwrap();
            let xs = seen(gt, &x[r * s.hidden..(r + 1) * s.hidden]);
            let mut h = vec![0.0f32; s.inter];
            gate_up_act(
                GatedAct::Silu,
                Matrix::new(&gd, s.inter, s.hidden).unwrap(),
                Matrix::new(&ud, s.inter, s.hidden).unwrap(),
                &xs,
                1,
                &mut h,
            )
            .unwrap();
            let hs = seen(dt, &h);
            let mut y = vec![0.0f32; s.hidden];
            ref_gemv(Matrix::new(&dd, s.hidden, s.inter).unwrap(), &hs, 1, &mut y).unwrap();
            for (&a, &b) in out[slot * s.hidden..(slot + 1) * s.hidden].iter().zip(&y) {
                let (a, b) = (f64::from(a), f64::from(b));
                num += (a - b) * (a - b);
                den += b * b;
            }
        }
        let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        assert!(rel <= tol, "{gt:?}: rel_l2 {rel:e} > tol_ref {tol:e}");
    }
}

#[test]
fn a_row_listing_an_expert_twice_fills_both_slots() {
    let mut rng = Rng(10);
    let (gt, dt) = SCHEMES[0];
    let layer = Layer::new(&mut rng, shape_for(gt, dt), gt, dt);
    let s = layer.shape;
    let rows = 3;
    let x = activations(&mut rng, rows * s.hidden);
    let ids = vec![2, 2, 0, 2, 5, 5];
    let reads = WeightReads::new(s.n_experts);
    let out = layer.run_with(Level::Portable, &x, &ids, rows, |_| true, Some(&reads));
    for (slot, &e) in ids.iter().enumerate() {
        let r = slot / s.top_k;
        let (_, want) = ffn_alone(
            &layer.expert(e as u32),
            &x[r * s.hidden..(r + 1) * s.hidden],
        );
        assert_same(
            &out[slot * s.hidden..(slot + 1) * s.hidden],
            &want,
            &format!("slot {slot}"),
        );
    }
    for e in [0, 2, 5] {
        assert_eq!(
            reads.get(e),
            layer.expert(e as u32).bytes() as u64,
            "expert {e}"
        );
    }
}
