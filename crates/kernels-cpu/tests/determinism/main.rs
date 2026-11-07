// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E3-T09: the CPU provider's deterministic variants (RFC-0040 §3.3; ADR-0019 decision 8).
//!
//! - `Gemv`, the CPU op built so far, gives bit-identical outputs when its output rows are
//!   split across 1, 2, 3, 4 and all of the host's cores, and across more threads than rows,
//!   at every level this host runs, for every weight type and for 1–8 activation rows. The
//!   split is the provider's own [`partition`], and each range runs on a real thread
//!   ([`split`]). Inputs holding NaN and infinities are included, compared by bit pattern.
//! - A device opened deterministic prepares only kernels whose capabilities declare
//!   `deterministic` at the key and shape. It refuses a kernel that declares nothing, so the
//!   fake provider's `Numeric` mode (QUAL-E3-T12), which opens the CPU provider deterministic,
//!   can never run an undeclared variant. A device opened without the option takes it.
//!
//! The other M0 ops reach the CPU provider with their own tasks; QUAL-E4-T06 runs this harness
//! over each of them.

#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

mod split;

use proptest::prelude::*;
use smallvec::smallvec;
use syntropy_core::{DType, DeviceId, ErrorClass, NodeId, QuantScheme};
use syntropy_kernels::{
    CpuIsa, DeviceCtx, KernelProvider, LayoutVersion, OpKey, OpKind, OpenOptions, PowerHint,
    ShapeClass, ShapeSig, StreamPlan,
};
use syntropy_kernels_cpu::gemv::{Activations, Level, WeightType, Weights, gemv, partition};
use syntropy_kernels_cpu::{CpuProvider, DispatchTable, KillSwitches};

use crate::split::check_thread_counts;

const TYPES: [WeightType; 5] = [
    WeightType::F16,
    WeightType::Bf16,
    WeightType::Q8_0,
    WeightType::Q4_0,
    WeightType::Q4K,
];

/// SplitMix64: deterministic test values.
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)` times `2^e`, `e` in `[-6, 6]`: a spread of magnitudes, so a
    /// changed summation order would change the rounding.
    fn spread(&mut self) -> f32 {
        let unit = ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
        unit * 2f32.powi((self.next() % 13) as i32 - 6)
    }
}

/// `n_out` rows of `k` weights of `ty`, with finite values and moderate F16 scales.
fn weights(g: &mut Gen, ty: WeightType, n_out: usize, k: usize) -> Vec<u8> {
    let f16 = |v: f32| syntropy_kernels::reference::act_quant::f16_bits(v).to_le_bytes();
    let scale = |g: &mut Gen| f16(0.005 + g.spread().abs().min(1.0) * 0.01);
    let mut out = Vec::new();
    for _ in 0..n_out * k / ty.block_elems() {
        match ty {
            WeightType::F16 => out.extend(f16(g.spread())),
            WeightType::Bf16 => out.extend(
                syntropy_kernels::reference::act_quant::round_bf16(g.spread()).to_le_bytes(),
            ),
            WeightType::Q8_0 | WeightType::Q4_0 => {
                out.extend(scale(g));
                out.extend((2..ty.block_bytes()).map(|_| g.next() as u8));
            }
            WeightType::Q4K => {
                out.extend(scale(g));
                out.extend(scale(g));
                out.extend((4..ty.block_bytes()).map(|_| g.next() as u8));
            }
        }
    }
    out
}

/// Every level this host runs, the portable one first.
fn levels() -> Vec<Level> {
    let mut v = vec![Level::Portable];
    for isa in [
        CpuIsa::X86V3,
        CpuIsa::X86V4,
        CpuIsa::Arm82Dot,
        CpuIsa::Arm86I8mm,
    ] {
        let l = Level::for_isa(isa);
        if !v.contains(&l) {
            v.push(l);
        }
    }
    v
}

/// Checks one GEMV at every thread count and level; returns nothing, panics on a difference.
fn check_gemv(ty: WeightType, w: &[u8], n_out: usize, k: usize, x: &[f32], rows: usize) {
    let weights = Weights::new(ty, w, n_out, k).expect("weights");
    let mut a = Activations::new();
    a.quantize(ty.act(), x, rows, k).expect("activations");
    let view = a.view();
    for level in levels() {
        check_thread_counts(
            &format!("{ty:?} {level:?} rows={rows} n_out={n_out} k={k}"),
            rows,
            n_out,
            |parts| partition(&weights, parts).collect(),
            |range, y| gemv(level, &weights, &view, range, y).expect("gemv"),
        );
    }
}

#[test]
fn gemv_is_bit_identical_across_thread_counts() {
    let mut g = Gen(0x5EED_0009);
    for ty in TYPES {
        let k = ty.block_elems().max(32) * 4;
        // Output counts that split evenly, unevenly, below the cache-line alignment, and below
        // the thread count.
        for n_out in [1, 7, 64, 131] {
            let w = weights(&mut g, ty, n_out, k);
            for rows in 1..=8 {
                let x: Vec<f32> = (0..rows * k).map(|_| g.spread()).collect();
                check_gemv(ty, &w, n_out, k, &x, rows);
            }
        }
    }
}

#[test]
fn gemv_with_nan_and_infinity_is_bit_identical_across_thread_counts() {
    let mut g = Gen(0xBAD_F00D);
    for ty in TYPES {
        let (n_out, k, rows) = (37, ty.block_elems().max(32) * 2, 3);
        let w = weights(&mut g, ty, n_out, k);
        let mut x: Vec<f32> = (0..rows * k).map(|_| g.spread()).collect();
        x[1] = f32::NAN;
        x[k + 5] = f32::INFINITY;
        x[2 * k + 9] = f32::NEG_INFINITY;
        check_gemv(ty, &w, n_out, k, &x, rows);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    /// Random shapes and seeds: the split never changes a bit.
    #[test]
    fn gemv_random_shapes_are_bit_identical_across_thread_counts(
        seed: u64,
        t in 0usize..5,
        blocks in 1usize..4,
        rows in 1usize..=8,
        n_out in 1usize..80,
    ) {
        let ty = TYPES[t];
        let k = ty.block_elems().max(32) * blocks;
        let mut g = Gen(seed);
        let w = weights(&mut g, ty, n_out, k);
        let x: Vec<f32> = (0..rows * k).map(|_| g.spread()).collect();
        check_gemv(ty, &w, n_out, k, &x, rows);
    }
}

fn open(p: &CpuProvider, deterministic: bool) -> DeviceCtx {
    p.open(
        DeviceId {
            node: NodeId([3; 32]),
            ordinal: 0,
        },
        &OpenOptions {
            streams: StreamPlan {
                compute: 1,
                copy_in: 0,
                copy_out: 0,
                collective: 0,
            },
            doorbell_slots: 4,
            capture: true,
            deterministic,
            power: PowerHint::Mains,
        },
    )
    .expect("opens")
}

fn gemv_key(weight: QuantScheme) -> OpKey {
    OpKey {
        op: OpKind::Gemv,
        schemes: smallvec![
            weight,
            QuantScheme::Plain(DType::F32),
            QuantScheme::Plain(DType::F32)
        ],
        layout: LayoutVersion::SOURCE,
        variant: 0,
    }
}

fn decode(rows: u32, dims: [u32; 2]) -> ShapeSig {
    ShapeSig {
        class: ShapeClass::Decode,
        rows,
        dims: smallvec![dims[0], dims[1]],
    }
}

/// Every `Gemv` scheme, at every level's table, declares a deterministic variant, and a
/// deterministic device prepares it.
#[test]
fn every_level_declares_gemv_deterministic_and_a_deterministic_device_prepares_it() {
    let schemes = [
        QuantScheme::Plain(DType::F16),
        QuantScheme::Plain(DType::BF16),
        QuantScheme::Ggml(syntropy_core::GgmlType::F16),
        QuantScheme::Ggml(syntropy_core::GgmlType::BF16),
        QuantScheme::Ggml(syntropy_core::GgmlType::Q8_0),
        QuantScheme::Ggml(syntropy_core::GgmlType::Q4_0),
        QuantScheme::Ggml(syntropy_core::GgmlType::Q4_K),
    ];
    for isa in [
        CpuIsa::X86V2,
        CpuIsa::X86V3,
        CpuIsa::X86V4,
        CpuIsa::X86V4Amx,
        CpuIsa::Arm80,
        CpuIsa::Arm82Dot,
        CpuIsa::Arm86I8mm,
        CpuIsa::Arm92Sme2,
    ] {
        let table = DispatchTable::for_level(isa);
        let kernel = table.get(&OpKind::Gemv).expect("every level has Gemv");
        for &scheme in &schemes {
            for rows in [1, 8] {
                let caps = (kernel.caps)(&gemv_key(scheme), &decode(rows, [64, 512]))
                    .unwrap_or_else(|| panic!("{isa:?} {scheme:?}: no caps"));
                assert!(caps.deterministic, "{isa:?} {scheme:?}");
            }
        }
    }
    let p = CpuProvider::new(KillSwitches::NONE);
    let ctx = open(&p, true);
    for scheme in schemes {
        p.prepare(&ctx, &gemv_key(scheme), &decode(4, [64, 512]))
            .unwrap_or_else(|e| panic!("{scheme:?}: {e}"));
    }
}

/// A kernel that declares no capabilities, or declares itself not deterministic.
fn noop(_: &syntropy_kernels_cpu::Launch<'_>) -> Result<(), syntropy_kernels::Error> {
    Ok(())
}

#[test]
fn a_deterministic_device_refuses_a_kernel_that_does_not_declare_it() {
    let p = CpuProvider::new(KillSwitches::NONE).with_kernel(OpKind::Add, noop);
    let shape = ShapeSig {
        class: ShapeClass::Decode,
        rows: 1,
        dims: smallvec![16],
    };
    let key = OpKey {
        op: OpKind::Add,
        schemes: smallvec![QuantScheme::Plain(DType::F32)],
        layout: LayoutVersion::SOURCE,
        variant: 0,
    };

    // Opened deterministic: the stand-in declares nothing, so it is refused, fail closed.
    let ctx = open(&p, true);
    let e = p.prepare(&ctx, &key, &shape).unwrap_err();
    assert_eq!(e.class, ErrorClass::Unsupported);
    assert_eq!(e.code, syntropy_kernels::error::codes::OP_UNSUPPORTED);
    assert!(e.to_string().contains("deterministic"), "{e}");

    // Opened without the option: the same kernel is prepared.
    let ctx = open(&p, false);
    p.prepare(&ctx, &key, &shape).expect("prepares");
}

/// A stand-in that replaces a real deterministic kernel is refused too: the check reads the
/// installed kernel's capabilities, not the op kind. A kernel whose capabilities say
/// `deterministic: false` is refused the same way (`provider.rs`'s unit test).
#[test]
fn a_stand_in_replacing_gemv_is_refused_on_a_deterministic_device() {
    let p = CpuProvider::new(KillSwitches::NONE).with_kernel(OpKind::Gemv, noop);
    let key = gemv_key(QuantScheme::Plain(DType::F16));
    let ctx = open(&p, true);
    let e = p.prepare(&ctx, &key, &decode(1, [8, 64])).unwrap_err();
    assert_eq!(e.class, ErrorClass::Unsupported);
    let ctx = open(&p, false);
    p.prepare(&ctx, &key, &decode(1, [8, 64]))
        .expect("prepares");
}

/// The harness catches a kernel whose value depends on the range it ran in: a sum that starts
/// from the range's first output, as a split reduction with per-thread partials would.
#[test]
#[should_panic(expected = "threads differ from 1")]
fn the_harness_catches_a_range_dependent_kernel() {
    let n_out = 16;
    check_thread_counts(
        "range-dependent",
        1,
        n_out,
        |parts| {
            let parts = parts.max(1);
            (0..parts)
                .map(|p| (p * n_out / parts)..((p + 1) * n_out / parts))
                .collect()
        },
        |range, y| {
            let mut acc = 0.1f32 * range.start as f32;
            for o in range {
                acc += 1.0e-3;
                y[o] = acc;
            }
        },
    );
}

/// And an output no range covers.
#[test]
#[should_panic(expected = "threads differ from 1")]
fn the_harness_catches_an_uncovered_output() {
    check_thread_counts(
        "uncovered",
        1,
        8,
        |parts| {
            let parts = parts.max(1);
            // With more than one thread the last output is left out.
            let end = if parts == 1 { 8 } else { 7 };
            (0..parts)
                .map(|p| (p * end / parts)..((p + 1) * end / parts))
                .collect()
        },
        |range, y| {
            for o in range {
                y[o] = o as f32;
            }
        },
    );
}
