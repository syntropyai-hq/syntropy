// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Gemv` through the CPU provider's `query`, `prepare` and `launch`, directly and from a
//! captured op list, for every weight type (COMP-E3-T03; RFC-0040 §7.3, §3.5).
//!
//! The provider's output equals the `gemv` module's portable kernels bit for bit (every level
//! does, `src/gemv/tests.rs`), which the module's tests compare with `kernels::reference`.

#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

use smallvec::{SmallVec, smallvec};
use syntropy_core::{
    AccumKind, ActQuant, BufferView, Bytes, DType, DeviceId, ErrorClass, GgmlType, HostRegion,
    NodeId, QuantScheme, ViewKind,
};
use syntropy_kernels::step::StepDesc;
use syntropy_kernels::{
    BufferSlice, DeviceCtx, KernelProvider, LayoutVersion, OpArgs, OpKey, OpKind, OpenOptions,
    PowerHint, ShapeClass, ShapeSig, Stream, StreamPlan, StreamPriority, TensorArg,
};
use syntropy_kernels_cpu::gemv::{Activations, Level, WeightType, Weights, gemv};
use syntropy_kernels_cpu::{CpuProvider, KillSwitches, Run};

const TYPES: [WeightType; 5] = [
    WeightType::F16,
    WeightType::Bf16,
    WeightType::Q8_0,
    WeightType::Q4_0,
    WeightType::Q4K,
];

fn open() -> (CpuProvider, DeviceCtx, Stream) {
    let p = CpuProvider::new(KillSwitches::NONE);
    let ctx = p
        .open(
            DeviceId {
                node: NodeId([9; 32]),
                ordinal: 0,
            },
            &OpenOptions {
                streams: StreamPlan {
                    compute: 1,
                    copy_in: 0,
                    copy_out: 0,
                    collective: 0,
                },
                doorbell_slots: 8,
                capture: true,
                deterministic: true,
                power: PowerHint::Mains,
            },
        )
        .unwrap();
    let s = p.stream(&ctx, StreamPriority::Compute).unwrap();
    (p, ctx, s)
}

fn key(weight: QuantScheme) -> OpKey {
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

fn shape(rows: u32, n_out: u32, k_in: u32) -> ShapeSig {
    ShapeSig {
        class: ShapeClass::Decode,
        rows,
        dims: smallvec![n_out, k_in],
    }
}

/// Deterministic test bytes and values (SplitMix64).
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Weights of `ty` with finite values: random codes, and F16 scales (or F16/BF16 values) of
/// moderate size.
fn weights(g: &mut Gen, ty: WeightType, n_out: usize, k: usize) -> Vec<u8> {
    let f16 = |v: f32| syntropy_kernels::reference::act_quant::f16_bits(v).to_le_bytes();
    let mut out = Vec::new();
    for _ in 0..n_out * k / ty.block_elems() {
        match ty {
            WeightType::F16 => out.extend(f16(g.unit())),
            WeightType::Bf16 => out
                .extend(syntropy_kernels::reference::act_quant::round_bf16(g.unit()).to_le_bytes()),
            WeightType::Q8_0 | WeightType::Q4_0 => {
                out.extend(f16(0.01 + g.unit().abs() * 0.01));
                out.extend((2..ty.block_bytes()).map(|_| g.next() as u8));
            }
            WeightType::Q4K => {
                out.extend(f16(0.01 + g.unit().abs() * 0.01));
                out.extend(f16(0.01 + g.unit().abs() * 0.01));
                out.extend((4..ty.block_bytes()).map(|_| g.next() as u8));
            }
        }
    }
    out
}

/// Host memory the test owns, in 64-byte-aligned `u64` words, imported into the device.
struct Host {
    words: Vec<u64>,
    handle: u64,
}

impl Host {
    fn new(p: &CpuProvider, ctx: &DeviceCtx, bytes: usize) -> Self {
        let mut words = vec![0u64; bytes.div_ceil(8).max(1)];
        let handle = p
            .import(
                ctx,
                HostRegion {
                    addr: words.as_mut_ptr() as u64,
                    len: Bytes((words.len() * 8) as u64),
                },
            )
            .unwrap()
            .handle;
        Self { words, handle }
    }

    fn bytes(&mut self) -> &mut [u8] {
        let n = self.words.len() * 8;
        // SAFETY: the words' bytes, for the test's own reads and writes between launches.
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast::<u8>(), n) }
    }

    fn slice(&self, bytes: usize) -> BufferSlice {
        BufferSlice {
            buffer: self.handle,
            offset: Bytes(0),
            bytes: Bytes(bytes as u64),
        }
    }

    fn f32s(&mut self, n: usize) -> Vec<f32> {
        self.bytes()[..4 * n]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }
}

/// The module's portable kernels on the same operands: what the provider must give.
fn expected(ty: WeightType, w: &[u8], n_out: usize, k: usize, x: &[f32], rows: usize) -> Vec<f32> {
    let w = Weights::new(ty, w, n_out, k).unwrap();
    let mut a = Activations::new();
    a.quantize(ty.act(), x, rows, k).unwrap();
    let mut y = vec![0.0f32; rows * n_out];
    gemv(Level::Portable, &w, &a.view(), 0..n_out, &mut y).unwrap();
    y
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn query_declares_each_types_numerics() {
    let (p, ctx, _) = open();
    for (scheme, accumulate, act_in) in [
        (
            QuantScheme::Plain(DType::F16),
            AccumKind::F32,
            ActQuant::None,
        ),
        (
            QuantScheme::Ggml(GgmlType::BF16),
            AccumKind::F32,
            ActQuant::None,
        ),
        (
            QuantScheme::Ggml(GgmlType::Q8_0),
            AccumKind::I32ThenF32 { max_block: 32 },
            ActQuant::Q8Block32,
        ),
        (
            QuantScheme::Ggml(GgmlType::Q4_0),
            AccumKind::I32ThenF32 { max_block: 32 },
            ActQuant::Q8Block32,
        ),
        (
            QuantScheme::Ggml(GgmlType::Q4_K),
            AccumKind::I32ThenF32 { max_block: 256 },
            ActQuant::Q8K256,
        ),
    ] {
        let caps = p.query(&ctx, &key(scheme), &shape(8, 64, 512)).unwrap();
        assert_eq!(caps.numerics.accumulate, accumulate, "{scheme:?}");
        assert_eq!(caps.numerics.act_in, act_in, "{scheme:?}");
        assert_eq!(caps.numerics.out, DType::F32);
        assert!(caps.deterministic && caps.capture_safe);
        let op = p.prepare(&ctx, &key(scheme), &shape(8, 64, 512)).unwrap();
        assert_eq!(op.workspace, caps.workspace);
        assert!(op.check_workspace(&caps).is_ok());
    }
    // A type with no kernel, a width that is not whole blocks, a fusion, another layout and
    // another output type are refused.
    let refused = [
        (key(QuantScheme::Ggml(GgmlType::Q4_1)), shape(1, 64, 512)),
        (key(QuantScheme::Ggml(GgmlType::Q4_K)), shape(1, 64, 288)),
        (
            OpKey {
                variant: 1,
                ..key(QuantScheme::Ggml(GgmlType::Q8_0))
            },
            shape(1, 64, 512),
        ),
        (
            OpKey {
                layout: LayoutVersion(7),
                ..key(QuantScheme::Ggml(GgmlType::Q8_0))
            },
            shape(1, 64, 512),
        ),
        (
            OpKey {
                schemes: SmallVec::from_slice(&[
                    QuantScheme::Ggml(GgmlType::Q8_0),
                    QuantScheme::Plain(DType::F32),
                    QuantScheme::Plain(DType::BF16),
                ]),
                ..key(QuantScheme::Ggml(GgmlType::Q8_0))
            },
            shape(1, 64, 512),
        ),
    ];
    for (k, s) in &refused {
        assert!(p.query(&ctx, k, s).is_none(), "{k:?}");
        let error = p.prepare(&ctx, k, s).unwrap_err();
        assert_eq!(error.class, ErrorClass::Unsupported, "{k:?}");
    }
}

/// One `Gemv` per type, launched directly and replayed from a capture, with a step descriptor
/// that makes fewer rows active than the bucket.
#[test]
fn launches_equal_the_portable_kernels() {
    let (p, ctx, s) = open();
    let mut g = Gen(11);
    let (bucket, n_out) = (8usize, 24usize);
    for ty in TYPES {
        let k = if ty == WeightType::Q4K { 512 } else { 96 };
        let scheme = QuantScheme::Ggml(ty.ggml());
        let op = p
            .prepare(
                &ctx,
                &key(scheme),
                &shape(bucket as u32, n_out as u32, k as u32),
            )
            .unwrap();
        let wbytes = weights(&mut g, ty, n_out, k);
        let mut w = Host::new(&p, &ctx, wbytes.len());
        w.bytes()[..wbytes.len()].copy_from_slice(&wbytes);
        let mut x = Host::new(&p, &ctx, bucket * k * 4);
        let xv: Vec<f32> = (0..bucket * k).map(|_| g.unit() * 4.0).collect();
        for (dst, v) in x.bytes().as_chunks_mut::<4>().0.iter_mut().zip(&xv) {
            *dst = v.to_le_bytes();
        }
        let mut y = Host::new(&p, &ctx, bucket * n_out * 4);
        let ws = Host::new(&p, &ctx, op.workspace.0 as usize);
        let mut step = Host::new(&p, &ctx, size_of::<StepDesc>());
        let at = std::mem::offset_of!(StepDesc, rows_active);
        let active = 5usize;
        step.bytes()[at..at + 4].copy_from_slice(&(active as u32).to_le_bytes());
        let inputs = [TensorArg::packed(
            x.slice(bucket * k * 4),
            DType::F32,
            [bucket as u32, k as u32, 1, 1],
        )
        .unwrap()];
        let outputs = [TensorArg::packed(
            y.slice(bucket * n_out * 4),
            DType::F32,
            [bucket as u32, n_out as u32, 1, 1],
        )
        .unwrap()];
        let weight = [BufferView {
            kind: ViewKind::Host,
            addr: w.words.as_ptr() as u64,
            len: Bytes(wbytes.len() as u64),
        }];
        let args = OpArgs {
            inputs: &inputs,
            outputs: &outputs,
            weights: &weight,
            step: step.slice(size_of::<StepDesc>()),
            workspace: Some(ws.slice(op.workspace.0 as usize)),
            statics: &[],
        };
        let want = expected(ty, &wbytes, n_out, k, &xv, active);

        // Directly: the active rows are written and the rest left as they were.
        y.bytes().fill(0xFF);
        p.launch(&ctx, &s, &op, &args).unwrap();
        assert_eq!(p.run(&ctx, &s).unwrap(), Run::Idle);
        let got = y.f32s(bucket * n_out);
        assert_eq!(bits(&got[..active * n_out]), bits(&want), "{ty:?}");
        assert!(
            got[active * n_out..]
                .iter()
                .all(|v| v.to_bits() == u32::MAX),
            "{ty:?}"
        );

        // From a capture, replayed twice.
        p.begin_capture(&ctx, &s).unwrap();
        p.launch(&ctx, &s, &op, &args).unwrap();
        let graph = p.end_capture(&ctx, &s).unwrap();
        for _ in 0..2 {
            y.bytes().fill(0);
            p.replay(&ctx, &s, &graph).unwrap();
            assert_eq!(p.run(&ctx, &s).unwrap(), Run::Idle);
            assert_eq!(bits(&y.f32s(active * n_out)), bits(&want), "{ty:?} replay");
        }
    }
}

#[test]
fn launch_refuses_mismatched_operands() {
    let (p, ctx, s) = open();
    let (rows, n_out, k) = (2u32, 4u32, 32u32);
    let op = p
        .prepare(
            &ctx,
            &key(QuantScheme::Ggml(GgmlType::Q8_0)),
            &shape(rows, n_out, k),
        )
        .unwrap();
    let x = Host::new(&p, &ctx, (rows * k * 4) as usize);
    let y = Host::new(&p, &ctx, (rows * n_out * 4) as usize);
    let ws = Host::new(&p, &ctx, op.workspace.0 as usize);
    let w = Host::new(&p, &ctx, (n_out * 34) as usize);
    let weight = [BufferView {
        kind: ViewKind::Host,
        addr: w.words.as_ptr() as u64,
        len: Bytes(u64::from(n_out * 34)),
    }];
    let good_x = TensorArg::packed(
        x.slice((rows * k * 4) as usize),
        DType::F32,
        [rows, k, 1, 1],
    )
    .unwrap();
    let good_y = TensorArg::packed(
        y.slice((rows * n_out * 4) as usize),
        DType::F32,
        [rows, n_out, 1, 1],
    )
    .unwrap();
    let ws_slice = Some(ws.slice(op.workspace.0 as usize));
    let launch =
        |inputs: &[TensorArg], outputs: &[TensorArg], weights: &[BufferView], workspace| {
            p.launch(
                &ctx,
                &s,
                &op,
                &OpArgs {
                    inputs,
                    outputs,
                    weights,
                    workspace,
                    ..OpArgs::none()
                },
            )
        };
    // The right operands are accepted.
    launch(&[good_x], &[good_y], &weight, ws_slice).unwrap();
    // A wrong shape, a missing weight, a short weight and a missing workspace are refused.
    let bad_x =
        TensorArg::packed(x.slice((rows * k * 4) as usize), DType::F32, [1, k, 1, 1]).unwrap();
    assert!(launch(&[bad_x], &[good_y], &weight, ws_slice).is_err());
    assert!(launch(&[good_x], &[good_y], &[], ws_slice).is_err());
    let short = [BufferView {
        len: Bytes(16),
        ..weight[0]
    }];
    assert!(launch(&[good_x], &[good_y], &short, ws_slice).is_err());
    assert!(launch(&[good_x], &[good_y], &weight, None).is_err());
    // Only the accepted launch was queued, and it ran without error.
    assert_eq!(p.run(&ctx, &s).unwrap(), Run::Idle);
    assert_eq!(p.health(&ctx), syntropy_kernels::DeviceHealth::Ok);
}
