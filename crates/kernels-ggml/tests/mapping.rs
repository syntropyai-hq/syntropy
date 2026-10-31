// SPDX-License-Identifier: MIT OR Apache-2.0

//! The op mapping layer on ggml's CPU backend (COMP-E7-T02; RFC-0040 §8.2).
//!
//! **Conformance.** `kernels::reference` defines each op (RFC-0040 §3.3). Its op functions are
//! COMP-E1-T03's and not written yet, so each case here computes the op's definition itself, in
//! F64 over the same F32 inputs, and takes from `kernels::reference` what it has: the activation
//! quantisation ggml applies before a matrix product (`act_quant::fake_quant`) and the F16 and
//! BF16 roundings (`act_quant::f16_bits`, `round_bf16`). Then:
//! - ops whose result is one correctly rounded IEEE operation per element (`Add`, `Mul`,
//!   `RowGather`, `Convert`) must match bit for bit;
//! - the others must lie within the worst-case F32 rounding bound of the F64 result, which holds
//!   for any summation order and with or without fused multiply-adds, so it holds on x86-64 and
//!   arm64 alike: for a sum of k products, `γ(k)·Σ|wᵢxᵢ|` with `γ(k) = k·u / (1 − k·u)` and
//!   `u = 2⁻²⁴`. Each bound is derived where it is used.
//!
//! RFC-0042 §6.4's calibrated `tol_op` and `tol_ref`, and QUAL-E4-T04's runner, replace these
//! bounds once the tolerance table publishes them; every entry is pending today.

mod common;

use common::{Buffers, F32, Gen, Launch, batch, decode, f32_bytes, key, tensor, to_f32};
use syntropy_core::{ActQuant, Bytes, DType, GgmlType, QuantScheme};
use syntropy_kernels::reference::act_quant::{
    Q8Path, bf16_to_f32, f16_bits, f16_to_f32, fake_quant, round_bf16,
};
use syntropy_kernels::{
    ActKind, AttnKind, BufferSlice, Fusions, GatedAct, LayoutVersion, OpKind, PoolKind, PreparedOp,
    RopeKind, RouteNorm, Scalar, ShapeClass, ShapeSig, StepDesc,
};
use syntropy_kernels_ggml::{CpuOps, GgmlType as Ty, dequantize};

const U: f64 = 1.0 / (1u64 << 24) as f64;

fn gamma(k: usize) -> f64 {
    let ku = k as f64 * U;
    ku / (1.0 - ku)
}

/// The path ggml's CPU backend quantises 8-bit activation blocks with, in this crate's
/// baseline-ISA build: NEON on arm64, ggml's reference code on x86-64-v2, which has no AVX.
const PATH: Q8Path = if cfg!(target_arch = "aarch64") {
    Q8Path::Arm64Neon
} else {
    Q8Path::Reference
};

fn ops() -> CpuOps {
    CpuOps::new().unwrap()
}

fn q(t: GgmlType) -> QuantScheme {
    QuantScheme::Ggml(t)
}

// ---- What is mapped, and what is not ----

#[test]
fn unmapped_ops_return_none_from_query_and_are_refused_by_prepare() {
    let ops = ops();
    let cases = [
        (
            key(OpKind::Rope(RopeKind::Plain), &[F32]),
            decode(&[8, 64, 64]),
        ),
        (
            key(OpKind::AttnDecode(AttnKind::Gqa), &[F32]),
            decode(&[8, 2, 64, 16]),
        ),
        (
            key(OpKind::AttnPrefill(AttnKind::Gqa), &[F32]),
            batch(4, &[8, 2, 64, 16]),
        ),
        (
            key(
                OpKind::MoeRoute {
                    norm: RouteNorm::Softmax,
                },
                &[F32],
            ),
            decode(&[8, 2, 64]),
        ),
        (
            key(OpKind::MoeGrouped, &[q(GgmlType::Q8_0)]),
            decode(&[8, 2, 64, 128]),
        ),
        (key(OpKind::MoeCombine, &[F32]), decode(&[2, 0, 64])),
        (key(OpKind::Pool(PoolKind::Mean), &[F32]), decode(&[64])),
        (key(OpKind::EmbedInject, &[F32]), decode(&[64])),
        (key(OpKind::LogitsProcess, &[F32]), decode(&[1000])),
        (key(OpKind::TopKTopP, &[F32]), decode(&[1000])),
        (key(OpKind::SampleSeeded, &[F32]), decode(&[1000])),
        (key(OpKind::VerifyAccept, &[F32]), decode(&[1000])),
        (key(OpKind::TapGather, &[F32]), decode(&[1000, 8])),
        (key(OpKind::Scale, &[F32]), decode(&[64])),
        // GELU's form is not fixed by `kernels::reference` yet.
        (key(OpKind::Act(ActKind::Gelu), &[F32]), decode(&[64])),
        (
            key(
                OpKind::GateUpAct {
                    act: GatedAct::Gelu,
                },
                &[q(GgmlType::Q8_0)],
            ),
            decode(&[64, 64]),
        ),
        (
            key(
                OpKind::GateUpAct {
                    act: GatedAct::ClampedSilu,
                },
                &[q(GgmlType::Q8_0)],
            ),
            decode(&[64, 64]),
        ),
        // F16 weights: ggml rounds the activation to F16, which `ActQuant` cannot state.
        (
            key(OpKind::Gemv, &[QuantScheme::Plain(DType::F16)]),
            decode(&[64, 64]),
        ),
        // A block type this layer does not map.
        (key(OpKind::Gemv, &[q(GgmlType::IQ4_NL)]), decode(&[64, 64])),
        // `k_in` not a whole number of Q8_0 blocks.
        (key(OpKind::Gemv, &[q(GgmlType::Q8_0)]), decode(&[64, 48])),
        // Wrong number of dims, and an empty dim.
        (key(OpKind::Gemv, &[F32]), decode(&[64])),
        (key(OpKind::Add, &[F32]), decode(&[0])),
        // An activation that is not F32.
        (
            key(OpKind::Gemv, &[F32, QuantScheme::Plain(DType::BF16)]),
            decode(&[64, 64]),
        ),
        // Convert to the same type.
        (key(OpKind::Convert, &[F32, F32]), decode(&[64])),
    ];
    for (k, s) in &cases {
        assert_eq!(ops.query(k, s), None, "{k:?} {s:?}");
        let e = ops.prepare(k, s).unwrap_err();
        assert_eq!(
            e.code,
            syntropy_kernels::error::codes::OP_UNSUPPORTED,
            "{k:?}"
        );
    }
    // A fusion bit, or a repacked layout, unmaps an op that is otherwise mapped.
    let mut fused = key(OpKind::Gemv, &[F32]);
    fused.variant = Fusions::FUSE_BIAS.bits();
    assert_eq!(ops.query(&fused, &decode(&[64, 64])), None);
    let mut repacked = key(OpKind::Gemv, &[F32]);
    repacked.layout = LayoutVersion(7);
    assert_eq!(ops.query(&repacked, &decode(&[64, 64])), None);
    assert!(
        ops.query(&key(OpKind::Gemv, &[F32]), &decode(&[64, 64]))
            .is_some()
    );
}

#[test]
fn query_reports_the_activation_quantisation_ggml_applies() {
    let ops = ops();
    for (scheme, act, max_block) in [
        (F32, ActQuant::None, None),
        (QuantScheme::Plain(DType::BF16), ActQuant::Bf16, None),
        (q(GgmlType::Q8_0), ActQuant::Q8Block32, Some(32)),
        (q(GgmlType::Q4_0), ActQuant::Q8Block32, Some(32)),
        (q(GgmlType::Q4_1), ActQuant::Q8Block32, Some(32)),
        (q(GgmlType::Q4_K), ActQuant::Q8K256, Some(256)),
        (q(GgmlType::Q6_K), ActQuant::Q8K256, Some(256)),
    ] {
        let caps = ops
            .query(&key(OpKind::Gemv, &[scheme, F32, F32]), &decode(&[16, 256]))
            .unwrap();
        assert_eq!(caps.numerics.act_in, act, "{scheme:?}");
        assert_eq!(caps.numerics.out, DType::F32);
        match max_block {
            None => assert_eq!(caps.numerics.accumulate, syntropy_core::AccumKind::F32),
            Some(b) => assert_eq!(
                caps.numerics.accumulate,
                syntropy_core::AccumKind::I32ThenF32 { max_block: b }
            ),
        }
        assert_eq!(caps.layout, LayoutVersion::SOURCE);
        assert!(caps.capture_safe && caps.deterministic);
        // The plan's work buffer holds the converted activation, except for F32 weights.
        assert_eq!(caps.workspace.0 > 0, act != ActQuant::None, "{scheme:?}");
    }
}

#[test]
fn prepare_declares_the_workspace_query_declared() {
    let ops = ops();
    let cases = [
        (
            key(
                OpKind::GateUpAct {
                    act: GatedAct::Silu,
                },
                &[q(GgmlType::Q8_0)],
            ),
            batch(4, &[32, 64]),
        ),
        (key(OpKind::RmsNorm, &[F32]), batch(4, &[64])),
        (key(OpKind::Add, &[F32]), batch(4, &[64])),
        (
            key(OpKind::Gemm, &[q(GgmlType::Q4_K)]),
            batch(32, &[16, 256]),
        ),
    ];
    for (k, s) in &cases {
        let caps = ops.query(k, s).unwrap();
        let op = ops.prepare(k, s).unwrap();
        assert_eq!(op.workspace, caps.workspace, "{k:?}");
        op.check_workspace(&caps).unwrap();
    }
    // `GateUpAct` keeps its `[rows, 2·n_out]` product in the workspace, `RmsNorm` its normalised
    // rows; an elementwise op needs none.
    let (gu, s) = &cases[0];
    assert!(ops.query(gu, s).unwrap().workspace.0 >= 4 * 64 * 4);
    let (rms, s) = &cases[1];
    assert!(ops.query(rms, s).unwrap().workspace.0 >= 4 * 64 * 4);
    let (add, s) = &cases[2];
    assert_eq!(ops.query(add, s).unwrap().workspace, Bytes(0));
}

// ---- Elementwise ops: bit for bit ----

/// Inputs with the edge cases an elementwise op must carry through: signed zeros, infinities,
/// NaN, subnormals and the extremes.
fn edge_values(g: &mut Gen, n: usize) -> Vec<f32> {
    let edges = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE / 4.0,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1.0,
    ];
    let mut v = g.floats(n, 100.0);
    v[..edges.len()].copy_from_slice(&edges);
    v
}

fn same_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan());
        assert!(
            ok,
            "{what}[{i}]: got {g:e} ({:#010x}), want {w:e}",
            g.to_bits()
        );
    }
}

#[test]
fn add_and_mul_match_ieee_bit_for_bit() {
    let ops = ops();
    let (rows, n) = (3u32, 70u32);
    let mut g = Gen(1);
    let x = edge_values(&mut g, (rows * n) as usize);
    let y: Vec<f32> = edge_values(&mut g, (rows * n) as usize)
        .into_iter()
        .rev()
        .collect();
    for (kind, f) in [
        (OpKind::Add, (|a, b| a + b) as fn(f32, f32) -> f32),
        (OpKind::Mul, |a, b| a * b),
    ] {
        let b = Buffers::default();
        let shape = batch(rows, &[n]);
        let op = ops.prepare(&key(kind, &[F32]), &shape).unwrap();
        let (hx, hy) = (b.add(&f32_bytes(&x)), b.add(&f32_bytes(&y)));
        let ho = b.filled((rows * n * 4) as usize, 0);
        let l = Launch::new(
            &b,
            &op,
            vec![
                tensor(&b, hx, DType::F32, rows, n),
                tensor(&b, hy, DType::F32, rows, n),
            ],
            vec![],
            vec![tensor(&b, ho, DType::F32, rows, n)],
            vec![],
        );
        l.run(&ops, &op, &b).unwrap();
        let want: Vec<f32> = x.iter().zip(&y).map(|(&a, &c)| f(a, c)).collect();
        same_bits(
            &to_f32(&b.bytes(ho, (rows * n * 4) as usize)),
            &want,
            &format!("{kind:?}"),
        );
    }
}

#[test]
fn convert_rounds_as_the_reference_does() {
    let ops = ops();
    let (rows, n) = (2u32, 40u32);
    let mut g = Gen(2);
    let mut x = edge_values(&mut g, (rows * n) as usize);
    // Values that round, tie and overflow in F16 and BF16.
    x[10..16].copy_from_slice(&[
        65519.0,
        65520.0,
        1.000_488_3,
        1.000_976_6,
        3.0e-8,
        1.0 + 1.0 / 256.0,
    ]);
    let len = (rows * n) as usize;
    for (to, want) in [
        (
            DType::F16,
            x.iter().map(|&v| f16_bits(v)).collect::<Vec<u16>>(),
        ),
        (DType::BF16, x.iter().map(|&v| round_bf16(v)).collect()),
    ] {
        let b = Buffers::default();
        let k = key(OpKind::Convert, &[F32, F32, QuantScheme::Plain(to)]);
        let op = ops.prepare(&k, &batch(rows, &[n])).unwrap();
        let hx = b.add(&f32_bytes(&x));
        let ho = b.filled(len * 2, 0);
        let l = Launch::new(
            &b,
            &op,
            vec![tensor(&b, hx, DType::F32, rows, n)],
            vec![],
            vec![tensor(&b, ho, to, rows, n)],
            vec![],
        );
        l.run(&ops, &op, &b).unwrap();
        let got: Vec<u16> = b
            .bytes(ho, len * 2)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        assert_eq!(got, want, "F32 to {to:?}");
        // And back: widening is exact.
        let back = key(OpKind::Convert, &[QuantScheme::Plain(to), F32, F32]);
        let op = ops.prepare(&back, &batch(rows, &[n])).unwrap();
        let hw = b.filled(len * 4, 0);
        let l = Launch::new(
            &b,
            &op,
            vec![tensor(&b, ho, to, rows, n)],
            vec![],
            vec![tensor(&b, hw, DType::F32, rows, n)],
            vec![],
        );
        l.run(&ops, &op, &b).unwrap();
        let widen = if to == DType::F16 {
            f16_to_f32
        } else {
            bf16_to_f32
        };
        same_bits(
            &to_f32(&b.bytes(hw, len * 4)),
            &want.iter().map(|&h| widen(h)).collect::<Vec<_>>(),
            &format!("{to:?} to F32"),
        );
    }
}

/// A Q8_0 block: an F16 scale and 32 signed codes; returns its bytes and its values.
fn q8_0_block(g: &mut Gen) -> (Vec<u8>, Vec<f32>) {
    let d = f16_bits(0.01 + g.unit().abs() * 0.05);
    let codes: Vec<i8> = g.bytes(32).into_iter().map(|b| b as i8).collect();
    let mut bytes = d.to_le_bytes().to_vec();
    bytes.extend(codes.iter().map(|&c| c as u8));
    let df = f16_to_f32(d);
    (bytes, codes.iter().map(|&c| f32::from(c) * df).collect())
}

/// A Q4_0 block: an F16 scale and 32 nibbles offset by 8 (element i is the low nibble of byte i,
/// element 16 + i its high nibble).
fn q4_0_block(g: &mut Gen) -> (Vec<u8>, Vec<f32>) {
    let d = f16_bits(0.01 + g.unit().abs() * 0.05);
    let qs = g.bytes(16);
    let mut bytes = d.to_le_bytes().to_vec();
    bytes.extend(&qs);
    let df = f16_to_f32(d);
    let lo = qs.iter().map(|q| f32::from(i16::from(q & 0xF) - 8) * df);
    let hi = qs.iter().map(|q| f32::from(i16::from(q >> 4) - 8) * df);
    (bytes, lo.chain(hi).collect())
}

/// A Q4_K super-block of random codes, scales and minimums under finite F16 `d` and `dmin`. Its
/// values are ggml's dequantisation, whose layout COMP-E7-T01's tests check on a hand-built block.
fn q4_k_block(g: &mut Gen) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = f16_bits(0.002 + g.unit().abs() * 0.01)
        .to_le_bytes()
        .to_vec();
    bytes.extend(f16_bits(0.001 + g.unit().abs() * 0.01).to_le_bytes());
    bytes.extend(g.bytes(12 + 128));
    let values = dequantize(Ty::Q4K, &bytes).unwrap();
    (bytes, values)
}

/// Makes one block: its bytes and its values.
type BlockFn = fn(&mut Gen) -> (Vec<u8>, Vec<f32>);

/// A weight matrix of `rows` rows of `k` elements in `scheme`: its bytes and its values.
fn weight(g: &mut Gen, scheme: QuantScheme, rows: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let (mut bytes, mut values) = (Vec::new(), Vec::new());
    match scheme {
        QuantScheme::Plain(DType::F32) => {
            values = g.floats(rows * k, 1.0);
            bytes = f32_bytes(&values);
        }
        QuantScheme::Plain(DType::BF16) => {
            for v in g.floats(rows * k, 1.0) {
                let h = round_bf16(v);
                bytes.extend(h.to_le_bytes());
                values.push(bf16_to_f32(h));
            }
        }
        QuantScheme::Ggml(t) => {
            let (block, make): (usize, BlockFn) = match t {
                GgmlType::Q8_0 => (32, q8_0_block),
                GgmlType::Q4_0 => (32, q4_0_block),
                GgmlType::Q4_K => (256, q4_k_block),
                _ => unreachable!("{t:?}"),
            };
            for _ in 0..rows * k / block {
                let (b, v) = make(g);
                bytes.extend(b);
                values.extend(v);
            }
        }
        _ => unreachable!("{scheme:?}"),
    }
    (bytes, values)
}

/// `x` as the matrix product reads it: through the activation quantisation `act`.
fn act_rows(act: ActQuant, x: &[f32], k: usize) -> Vec<f32> {
    x.chunks_exact(k)
        .flat_map(|row| fake_quant(act, row, PATH).unwrap())
        .collect()
}

/// `W·xᵀ` in F64 per (row, output), with `Σ|wᵢxᵢ|` beside it.
fn matmul_f64(w: &[f32], x: &[f32], n_out: usize, k: usize) -> Vec<(f64, f64)> {
    x.chunks_exact(k)
        .flat_map(|xr| {
            w.chunks_exact(k).map(move |wr| {
                wr.iter().zip(xr).fold((0.0, 0.0), |(s, a), (&wi, &xi)| {
                    let p = f64::from(wi) * f64::from(xi);
                    (s + p, a + p.abs())
                })
            })
        })
        .take(x.len() / k * n_out)
        .collect()
}

fn run_matmul(kind: OpKind, scheme: QuantScheme, shape: ShapeSig, seed: u64) {
    let ops = ops();
    let (n_out, k) = (shape.dims[0] as usize, shape.dims[1] as usize);
    let rows = shape.rows as usize;
    let mut g = Gen(seed);
    let (wb, w) = weight(&mut g, scheme, n_out, k);
    let x = g.floats(rows * k, 2.0);
    let k_ = key(kind, &[scheme, F32, F32]);
    let caps = ops.query(&k_, &shape).unwrap();
    let op = ops.prepare(&k_, &shape).unwrap();
    let b = Buffers::default();
    let (hw, hx) = (b.add(&wb), b.add(&f32_bytes(&x)));
    let ho = b.filled(rows * n_out * 4, 0xFF);
    let l = Launch::new(
        &b,
        &op,
        vec![tensor(&b, hx, DType::F32, shape.rows, k as u32)],
        vec![b.view(hw)],
        vec![tensor(&b, ho, DType::F32, shape.rows, n_out as u32)],
        vec![],
    );
    l.run(&ops, &op, &b).unwrap();
    let got = to_f32(&b.bytes(ho, rows * n_out * 4));
    let xq = act_rows(caps.numerics.act_in, &x, k);
    // A sum of k products in F32, in any order: γ(k)·Σ|wᵢxᵢ|. Integer block sums are exact and
    // add no term; the block scale product of two F16 values is exact in F32. Two more roundings
    // (the scale times the integer sum, and the K-quant minimum term) are covered by γ(k + 2).
    let bound = gamma(k + 2);
    for (i, (&gv, &(exact, abs))) in got.iter().zip(&matmul_f64(&w, &xq, n_out, k)).enumerate() {
        let err = (f64::from(gv) - exact).abs();
        assert!(
            err <= bound * abs + 1e-30,
            "{kind:?} {scheme:?} [{}][{}]: got {gv}, exact {exact}, err {err:e} > {:e}",
            i / n_out,
            i % n_out,
            bound * abs
        );
    }
}

#[test]
fn gemv_and_logits_are_within_the_f32_bound_for_each_weight_type() {
    for (i, scheme) in [
        F32,
        QuantScheme::Plain(DType::BF16),
        q(GgmlType::Q8_0),
        q(GgmlType::Q4_0),
        q(GgmlType::Q4_K),
    ]
    .into_iter()
    .enumerate()
    {
        run_matmul(OpKind::Gemv, scheme, decode(&[24, 512]), 10 + i as u64);
        run_matmul(
            OpKind::Gemv,
            scheme,
            ShapeSig::new(ShapeClass::Verify { window: 5 }, &[24, 512]),
            20 + i as u64,
        );
        run_matmul(OpKind::Logits, scheme, decode(&[300, 256]), 30 + i as u64);
    }
}

#[test]
fn gemm_is_within_the_f32_bound() {
    for (i, scheme) in [F32, q(GgmlType::Q8_0), q(GgmlType::Q4_K)]
        .into_iter()
        .enumerate()
    {
        run_matmul(
            OpKind::Gemm,
            scheme,
            ShapeSig::new(ShapeClass::Prefill { chunk: 64 }, &[48, 256]),
            40 + i as u64,
        );
    }
}

fn silu64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

#[test]
fn gate_up_silu_is_within_its_bound() {
    let ops = ops();
    let (n_out, k, rows) = (40usize, 256usize, 3u32);
    let shape = batch(rows, &[n_out as u32, k as u32]);
    for (i, scheme) in [F32, q(GgmlType::Q8_0), q(GgmlType::Q4_K)]
        .into_iter()
        .enumerate()
    {
        let mut g = Gen(50 + i as u64);
        let (wb, w) = weight(&mut g, scheme, 2 * n_out, k);
        let x = g.floats(rows as usize * k, 2.0);
        let k_ = key(
            OpKind::GateUpAct {
                act: GatedAct::Silu,
            },
            &[scheme, F32, F32],
        );
        let caps = ops.query(&k_, &shape).unwrap();
        let op = ops.prepare(&k_, &shape).unwrap();
        let b = Buffers::default();
        let (hw, hx) = (b.add(&wb), b.add(&f32_bytes(&x)));
        let ho = b.filled(rows as usize * n_out * 4, 0);
        let l = Launch::new(
            &b,
            &op,
            vec![tensor(&b, hx, DType::F32, rows, k as u32)],
            vec![b.view(hw)],
            vec![tensor(&b, ho, DType::F32, rows, n_out as u32)],
            vec![],
        );
        l.run(&ops, &op, &b).unwrap();
        let got = to_f32(&b.bytes(ho, rows as usize * n_out * 4));
        let h = matmul_f64(&w, &act_rows(caps.numerics.act_in, &x, k), 2 * n_out, k);
        // Gate g and up v carry the product bound e_g, e_v. silu's derivative is at most 1.1 in
        // magnitude, so silu(g) is off by at most 1.1·e_g plus its own evaluation error, which
        // ggml's vectorised exp bounds at about 2 ulp, giving at most 8u·|silu(g)| with the add
        // and the divide. The final multiply adds one rounding.
        for r in 0..rows as usize {
            for o in 0..n_out {
                let (gv, ga) = h[r * 2 * n_out + o];
                let (uv, ua) = h[r * 2 * n_out + n_out + o];
                let (eg, eu) = (gamma(k + 2) * ga, gamma(k + 2) * ua);
                let s = silu64(gv);
                let exact = s * uv;
                let bound = (1.1 * eg + 8.0 * U * s.abs()) * (uv.abs() + eu)
                    + s.abs() * eu
                    + U * exact.abs()
                    + 1e-30;
                let got = f64::from(got[r * n_out + o]);
                assert!(
                    (got - exact).abs() <= bound,
                    "{scheme:?} [{r}][{o}]: got {got}, exact {exact}, bound {bound:e}"
                );
            }
        }
    }
}

#[test]
fn rms_norm_is_within_its_bound_and_reads_eps_from_the_statics() {
    let ops = ops();
    let (rows, n) = (4u32, 96usize);
    let shape = batch(rows, &[n as u32]);
    let op = ops.prepare(&key(OpKind::RmsNorm, &[F32]), &shape).unwrap();
    let mut g = Gen(60);
    let mut x = g.floats(rows as usize * n, 3.0);
    // A row of zeros: the result is 0 · (1/√eps) · gain, all zeros.
    x[n..2 * n].fill(0.0);
    let gain = g.floats(n, 2.0);
    for eps in [1e-6f32, 1e-5, 0.5] {
        let b = Buffers::default();
        let (hx, hg) = (b.add(&f32_bytes(&x)), b.add(&f32_bytes(&gain)));
        let ho = b.filled(rows as usize * n * 4, 0);
        let l = Launch::new(
            &b,
            &op,
            vec![tensor(&b, hx, DType::F32, rows, n as u32)],
            vec![b.view(hg)],
            vec![tensor(&b, ho, DType::F32, rows, n as u32)],
            vec![Scalar::F32(eps)],
        );
        l.run(&ops, &op, &b).unwrap();
        let got = to_f32(&b.bytes(ho, rows as usize * n * 4));
        for (r, xr) in x.chunks_exact(n).enumerate() {
            let ms = xr.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / n as f64;
            let scale = 1.0 / (ms + f64::from(eps)).sqrt();
            for (i, (&xi, &gi)) in xr.iter().zip(&gain).enumerate() {
                let exact = f64::from(xi) * scale * f64::from(gi);
                // Squares (u each, summed in F64), the mean's narrowing to F32, the eps add,
                // sqrt, the reciprocal, and two multiplies: under 8u relative.
                let bound = 8.0 * U * exact.abs() + 1e-30;
                let got = f64::from(got[r * n + i]);
                assert!(
                    (got - exact).abs() <= bound,
                    "eps {eps} [{r}][{i}]: got {got}, exact {exact}"
                );
            }
        }
    }
}

#[test]
fn silu_is_within_its_bound() {
    let ops = ops();
    let (rows, n) = (2u32, 200usize);
    let shape = batch(rows, &[n as u32]);
    let op = ops
        .prepare(&key(OpKind::Act(ActKind::Silu), &[F32]), &shape)
        .unwrap();
    let mut g = Gen(70);
    let x = g.floats(rows as usize * n, 30.0);
    let b = Buffers::default();
    let hx = b.add(&f32_bytes(&x));
    let ho = b.filled(rows as usize * n * 4, 0);
    let l = Launch::new(
        &b,
        &op,
        vec![tensor(&b, hx, DType::F32, rows, n as u32)],
        vec![],
        vec![tensor(&b, ho, DType::F32, rows, n as u32)],
        vec![],
    );
    l.run(&ops, &op, &b).unwrap();
    for (&gv, &xi) in to_f32(&b.bytes(ho, rows as usize * n * 4)).iter().zip(&x) {
        let exact = silu64(f64::from(xi));
        // exp within about 2 ulp (ggml's vectorised exp, or the C library's for a tail), then
        // an add and a divide: 8u relative.
        let bound = 8.0 * U * exact.abs() + 1e-37;
        assert!(
            (f64::from(gv) - exact).abs() <= bound,
            "silu({xi}): got {gv}, exact {exact}"
        );
    }
}

#[test]
fn row_gather_matches_the_table_rows_bit_for_bit() {
    let ops = ops();
    let mut g = Gen(80);
    let (table_rows, width, rows) = (7usize, 64usize, 5u32);
    let ids: Vec<i32> = vec![6, 0, 3, 3, 1];
    for scheme in [
        F32,
        QuantScheme::Plain(DType::BF16),
        q(GgmlType::Q8_0),
        q(GgmlType::Q4_0),
    ] {
        let (tb, t) = weight(&mut g, scheme, table_rows, width);
        let shape = batch(rows, &[table_rows as u32, width as u32]);
        let op = ops
            .prepare(&key(OpKind::RowGather, &[scheme, F32, F32]), &shape)
            .unwrap();
        let b = Buffers::default();
        let ht = b.add(&tb);
        let hi = b.add(&ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>());
        let ho = b.filled(rows as usize * width * 4, 0);
        let l = Launch::new(
            &b,
            &op,
            vec![tensor(&b, hi, DType::I32, rows, 1).with_rows_only()],
            vec![b.view(ht)],
            vec![tensor(&b, ho, DType::F32, rows, width as u32)],
            vec![],
        );
        l.run(&ops, &op, &b).unwrap();
        let want: Vec<f32> = ids
            .iter()
            .flat_map(|&i| {
                t[i as usize * width..(i as usize + 1) * width]
                    .iter()
                    .copied()
            })
            .collect();
        same_bits(
            &to_f32(&b.bytes(ho, rows as usize * width * 4)),
            &want,
            &format!("{scheme:?}"),
        );
    }
}

/// An ids tensor is one-dimensional: `[rows]`.
trait RowsOnly {
    fn with_rows_only(self) -> Self;
}

impl RowsOnly for syntropy_kernels::TensorArg {
    fn with_rows_only(self) -> Self {
        syntropy_kernels::TensorArg::packed(self.buf, self.dtype, [self.dims[0], 1, 1, 1]).unwrap()
    }
}

#[test]
fn row_gather_refuses_an_index_outside_the_table() {
    let ops = ops();
    let shape = batch(2, &[4, 32]);
    let op = ops
        .prepare(&key(OpKind::RowGather, &[F32]), &shape)
        .unwrap();
    for bad in [4i32, -1, i32::MAX] {
        let b = Buffers::default();
        let ht = b.filled(4 * 32 * 4, 0);
        let hi = b.add(
            &[0i32, bad]
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let ho = b.filled(2 * 32 * 4, 0x11);
        let l = Launch::new(
            &b,
            &op,
            vec![tensor(&b, hi, DType::I32, 2, 1).with_rows_only()],
            vec![b.view(ht)],
            vec![tensor(&b, ho, DType::F32, 2, 32)],
            vec![],
        );
        assert!(l.run(&ops, &op, &b).is_err(), "index {bad}");
        assert!(
            b.bytes(ho, 2 * 32 * 4).iter().all(|&v| v == 0x11),
            "nothing written"
        );
    }
}

// ---- Rows: the bucket and the active count ----

/// A step descriptor with `rows_active` set, in a buffer of `b`.
fn step(b: &Buffers, rows_active: u32) -> BufferSlice {
    let mut bytes = vec![0u8; std::mem::size_of::<StepDesc>()];
    let at = std::mem::offset_of!(StepDesc, rows_active);
    bytes[at..at + 4].copy_from_slice(&rows_active.to_le_bytes());
    b.slice(b.add(&bytes))
}

#[test]
fn a_launch_computes_only_the_active_rows() {
    let ops = ops();
    let (rows, n_out, k) = (6u32, 8usize, 64usize);
    let shape = batch(rows, &[n_out as u32, k as u32]);
    let k_ = key(OpKind::Gemm, &[q(GgmlType::Q8_0)]);
    let op = ops.prepare(&k_, &shape).unwrap();
    let mut g = Gen(90);
    let (wb, _) = weight(&mut g, q(GgmlType::Q8_0), n_out, k);
    let x = g.floats(rows as usize * k, 1.0);
    let row_bytes = n_out * 4;
    let mut full = None;
    for active in [6u32, 2, 0] {
        let b = Buffers::default();
        let (hw, hx) = (b.add(&wb), b.add(&f32_bytes(&x)));
        let ho = b.filled(rows as usize * row_bytes, 0x7F);
        let mut l = Launch::new(
            &b,
            &op,
            vec![tensor(&b, hx, DType::F32, rows, k as u32)],
            vec![b.view(hw)],
            vec![tensor(&b, ho, DType::F32, rows, n_out as u32)],
            vec![],
        );
        l.step = step(&b, active);
        l.run(&ops, &op, &b).unwrap();
        let out = b.bytes(ho, rows as usize * row_bytes);
        let split = active as usize * row_bytes;
        let full = full.get_or_insert_with(|| out.clone());
        assert_eq!(
            out[..split],
            full[..split],
            "active rows agree with the full bucket"
        );
        assert!(
            out[split..].iter().all(|&v| v == 0x7F),
            "rows past {active} untouched"
        );
    }
    // The full bucket again, after shorter launches: the row extents were restored.
    let b = Buffers::default();
    let (hw, hx) = (b.add(&wb), b.add(&f32_bytes(&x)));
    let ho = b.filled(rows as usize * row_bytes, 0);
    let l = Launch::new(
        &b,
        &op,
        vec![tensor(&b, hx, DType::F32, rows, k as u32)],
        vec![b.view(hw)],
        vec![tensor(&b, ho, DType::F32, rows, n_out as u32)],
        vec![],
    );
    l.run(&ops, &op, &b).unwrap();
    assert_eq!(Some(b.bytes(ho, rows as usize * row_bytes)), full);
    // More active rows than the bucket is refused.
    let mut l = l;
    l.step = step(&b, rows + 1);
    assert!(l.run(&ops, &op, &b).is_err());
}

// ---- Arguments that do not match the template are refused ----

#[test]
fn mismatched_arguments_are_refused_before_ggml_runs() {
    let ops = ops();
    let shape = batch(2, &[64]);
    let rms = ops.prepare(&key(OpKind::RmsNorm, &[F32]), &shape).unwrap();
    let b = Buffers::default();
    let hx = b.filled(2 * 64 * 4, 0);
    let hg = b.filled(64 * 4, 0);
    let ho = b.filled(2 * 64 * 4, 0x33);
    let good = || {
        Launch::new(
            &b,
            &rms,
            vec![tensor(&b, hx, DType::F32, 2, 64)],
            vec![b.view(hg)],
            vec![tensor(&b, ho, DType::F32, 2, 64)],
            vec![Scalar::F32(1e-6)],
        )
    };
    good().run(&ops, &rms, &b).unwrap();
    let ho2 = b.filled(2 * 64 * 4, 0x33);
    let mut bad: Vec<Launch> = Vec::new();
    // No eps, a NaN eps, an extra static.
    for statics in [
        vec![],
        vec![Scalar::F32(f32::NAN)],
        vec![Scalar::F32(1e-6), Scalar::F32(1.0)],
    ] {
        let mut l = good();
        l.statics = statics;
        bad.push(l);
    }
    // A wrong dtype, a wrong row count, a wrong width.
    for t in [
        tensor(&b, hx, DType::BF16, 2, 64),
        tensor(&b, hx, DType::F32, 1, 64),
        tensor(&b, hx, DType::F32, 2, 32),
    ] {
        let mut l = good();
        l.inputs = vec![t];
        bad.push(l);
    }
    // A slice past its buffer's end, an unknown buffer.
    let mut l = good();
    l.outputs[0].buf.offset = Bytes(64);
    bad.push(l);
    let mut l = good();
    l.outputs[0].buf.buffer = 99;
    bad.push(l);
    // A weight view smaller than the gain, or misaligned.
    let mut l = good();
    l.weights[0].len = Bytes(64 * 4 - 1);
    bad.push(l);
    let mut l = good();
    l.weights[0].addr += 2;
    bad.push(l);
    // An input slice misaligned for F32.
    let hx2 = b.filled(2 * 64 * 4 + 4, 0);
    let mut l = good();
    l.inputs = vec![tensor(&b, hx2, DType::F32, 2, 64)];
    l.inputs[0].buf.offset = Bytes(2);
    bad.push(l);
    // No workspace, or one too small.
    let mut l = good();
    l.workspace = None;
    bad.push(l);
    let mut l = good();
    l.workspace.as_mut().unwrap().bytes = Bytes(rms.workspace.0 - 1);
    bad.push(l);
    // Missing operands.
    let mut l = good();
    l.weights.clear();
    bad.push(l);
    for (i, l) in bad.iter_mut().enumerate() {
        if l.outputs[0].buf.buffer == ho {
            l.outputs[0].buf.buffer = ho2;
        }
        assert!(l.run(&ops, &rms, &b).is_err(), "case {i}");
    }
    assert!(
        b.bytes(ho2, 2 * 64 * 4).iter().all(|&v| v == 0x33),
        "nothing written"
    );
    // An unknown handle.
    let unknown = PreparedOp::with_handle(Bytes(0), 1000);
    assert!(good().run(&ops, &unknown, &b).is_err());
}

/// A NaN or an infinity in a row propagates; it does not stop the process.
#[test]
fn rms_norm_propagates_nan_and_infinity() {
    let ops = ops();
    let shape = batch(2, &[32]);
    let op = ops.prepare(&key(OpKind::RmsNorm, &[F32]), &shape).unwrap();
    let mut x = vec![1.0f32; 64];
    x[3] = f32::NAN;
    x[40] = f32::INFINITY;
    let b = Buffers::default();
    let (hx, hg) = (b.add(&f32_bytes(&x)), b.add(&f32_bytes(&[1.0; 32])));
    let ho = b.filled(64 * 4, 0);
    let l = Launch::new(
        &b,
        &op,
        vec![tensor(&b, hx, DType::F32, 2, 32)],
        vec![b.view(hg)],
        vec![tensor(&b, ho, DType::F32, 2, 32)],
        vec![Scalar::F32(1e-6)],
    );
    l.run(&ops, &op, &b).unwrap();
    let got = to_f32(&b.bytes(ho, 64 * 4));
    // Row 0's mean square is NaN, so every element is. Row 1's is infinite, so its scale is
    // 1/√∞ = 0: finite elements become 0 and the infinite one ∞ · 0 = NaN, as in F64.
    assert!(got[..32].iter().all(|v| v.is_nan()), "{:?}", &got[..32]);
    for (i, v) in got[32..].iter().enumerate() {
        if i == 8 {
            assert!(v.is_nan(), "[1][{i}] = {v}");
        } else {
            assert_eq!(v.to_bits(), 0, "[1][{i}] = {v}");
        }
    }
}
