// SPDX-License-Identifier: MIT OR Apache-2.0

//! The COMP-E3-T22 ops through the CPU provider's `query`, `prepare` and `launch`, directly and
//! from a captured op list, with fewer rows active than the bucket (RFC-0040 §7.3, §3.5).
//!
//! Each launch's output equals the module functions bit for bit (`gemv::gemv`,
//! `ops::gate_up_act`, `ops::row_gather`, `ops::add`, `ops::mul`, `ops::convert_rows`), which
//! `src/ops/tests.rs` compares with `kernels::reference` at every level.

#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

use smallvec::{SmallVec, smallvec};
use syntropy_core::{
    BufferView, Bytes, DType, DeviceId, GgmlType, HostRegion, NodeId, QuantScheme, ViewKind,
};
use syntropy_kernels::step::StepDesc;
use syntropy_kernels::{
    BufferSlice, DeviceCtx, DeviceHealth, GatedAct, KernelProvider, LayoutVersion, OpArgs, OpKey,
    OpKind, OpenOptions, PowerHint, PreparedOp, ShapeClass, ShapeSig, Stream, StreamPlan,
    StreamPriority, TensorArg,
};
use syntropy_kernels_cpu::gemv::{Activations, Level, WeightType, Weights, gemv};
use syntropy_kernels_cpu::ops::{add, convert_rows, gate_up_act, gate_up_scratch, mul, row_gather};
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
                node: NodeId([7; 32]),
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

fn f32s() -> SmallVec<[QuantScheme; 3]> {
    smallvec![QuantScheme::Plain(DType::F32)]
}

fn key(op: OpKind, schemes: &[QuantScheme]) -> OpKey {
    OpKey {
        op,
        schemes: SmallVec::from_slice(schemes),
        layout: LayoutVersion::SOURCE,
        variant: 0,
    }
}

fn shape(class: ShapeClass, rows: u32, dims: &[u32]) -> ShapeSig {
    ShapeSig {
        class,
        rows,
        dims: SmallVec::from_slice(dims),
    }
}

const PREFILL: ShapeClass = ShapeClass::Prefill { chunk: 512 };

const F32: QuantScheme = QuantScheme::Plain(DType::F32);

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

    fn f32s(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.unit() * 4.0).collect()
    }
}

/// Weights of `ty` with finite values.
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

/// Host memory the test owns, in 8-byte-aligned words, imported into the device.
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

    fn with(p: &CpuProvider, ctx: &DeviceCtx, data: &[u8]) -> Self {
        let mut h = Self::new(p, ctx, data.len());
        h.bytes()[..data.len()].copy_from_slice(data);
        h
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

    fn tensor(&self, dtype: DType, rows: usize, n: usize) -> TensorArg {
        let bytes = rows * n * (dtype.bits() as usize / 8);
        TensorArg::packed(self.slice(bytes), dtype, [rows as u32, n as u32, 1, 1]).unwrap()
    }

    fn view(&self, bytes: usize) -> BufferView {
        BufferView {
            kind: ViewKind::Host,
            addr: self.words.as_ptr() as u64,
            len: Bytes(bytes as u64),
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

fn le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|v| v.to_bits()).collect()
}

/// A step descriptor with `rows_active` set.
fn step(p: &CpuProvider, ctx: &DeviceCtx, active: usize) -> Host {
    let mut h = Host::new(p, ctx, size_of::<StepDesc>());
    let at = std::mem::offset_of!(StepDesc, rows_active);
    h.bytes()[at..at + 4].copy_from_slice(&(active as u32).to_le_bytes());
    h
}

/// Launches `op` with `args` directly and then twice from a capture, filling `out` with `fill`
/// before each run, and returns the output bytes after each of the three runs.
fn three_runs(
    p: &CpuProvider,
    ctx: &DeviceCtx,
    s: &Stream,
    op: &PreparedOp,
    args: &OpArgs<'_>,
    out: &mut Host,
    fill: u8,
) -> Vec<Vec<u8>> {
    let mut seen = Vec::new();
    out.bytes().fill(fill);
    p.launch(ctx, s, op, args).unwrap();
    assert_eq!(p.run(ctx, s).unwrap(), Run::Idle);
    seen.push(out.bytes().to_vec());
    p.begin_capture(ctx, s).unwrap();
    p.launch(ctx, s, op, args).unwrap();
    let graph = p.end_capture(ctx, s).unwrap();
    for _ in 0..2 {
        out.bytes().fill(fill);
        p.replay(ctx, s, &graph).unwrap();
        assert_eq!(p.run(ctx, s).unwrap(), Run::Idle);
        seen.push(out.bytes().to_vec());
    }
    seen
}

#[test]
fn query_and_prepare_accept_each_op_and_refuse_the_rest() {
    let (p, ctx, _) = open();
    let q8 = QuantScheme::Ggml(GgmlType::Q8_0);
    let silu = OpKind::GateUpAct {
        act: GatedAct::Silu,
    };
    let accepted = [
        (
            key(OpKind::Gemm, &[q8, F32, F32]),
            shape(PREFILL, 512, &[64, 512]),
        ),
        (
            key(
                OpKind::Logits,
                &[QuantScheme::Ggml(GgmlType::Q4_K), F32, F32],
            ),
            shape(ShapeClass::Decode, 1, &[1000, 512]),
        ),
        (key(silu, &[q8, F32, F32]), shape(PREFILL, 64, &[96, 256])),
        (
            key(OpKind::Add, &[F32, F32, F32]),
            shape(ShapeClass::Decode, 4, &[100]),
        ),
        (
            key(OpKind::Mul, &f32s()),
            shape(ShapeClass::Decode, 4, &[100]),
        ),
        (
            key(OpKind::RowGather, &[QuantScheme::Plain(DType::BF16), F32]),
            shape(ShapeClass::Decode, 4, &[50, 64]),
        ),
        (
            key(OpKind::RowGather, &[QuantScheme::Ggml(GgmlType::Q4_0), F32]),
            shape(ShapeClass::Decode, 4, &[50, 64]),
        ),
        (
            key(OpKind::Convert, &[F32, QuantScheme::Plain(DType::BF16)]),
            shape(ShapeClass::Decode, 4, &[100]),
        ),
        (
            key(OpKind::Convert, &[QuantScheme::Plain(DType::F16), F32]),
            shape(ShapeClass::Decode, 4, &[100]),
        ),
    ];
    for (k, sh) in &accepted {
        let caps = p.query(&ctx, k, sh).unwrap_or_else(|| panic!("{k:?}"));
        assert!(caps.deterministic && caps.capture_safe, "{k:?}");
        let op = p.prepare(&ctx, k, sh).unwrap();
        assert_eq!(op.workspace, caps.workspace, "{k:?}");
    }
    let refused = [
        (
            key(
                OpKind::GateUpAct {
                    act: GatedAct::Gelu,
                },
                &[q8, F32, F32],
            ),
            shape(ShapeClass::Decode, 1, &[64, 256]),
        ),
        (
            key(OpKind::Gemm, &[QuantScheme::Ggml(GgmlType::Q4_1), F32, F32]),
            shape(PREFILL, 32, &[64, 256]),
        ),
        (
            OpKey {
                variant: 1,
                ..key(OpKind::Logits, &[q8, F32, F32])
            },
            shape(ShapeClass::Decode, 1, &[64, 256]),
        ),
        (
            key(OpKind::Add, &[F32, QuantScheme::Plain(DType::BF16), F32]),
            shape(ShapeClass::Decode, 4, &[100]),
        ),
        (
            key(OpKind::Mul, &f32s()),
            shape(ShapeClass::Decode, 4, &[10, 10]),
        ),
        (
            key(OpKind::RowGather, &[QuantScheme::Ggml(GgmlType::Q4_1), F32]),
            shape(ShapeClass::Decode, 4, &[50, 64]),
        ),
        (
            key(OpKind::RowGather, &[q8, F32]),
            shape(ShapeClass::Decode, 4, &[50, 48]),
        ),
        (
            key(OpKind::Convert, &[F32, F32]),
            shape(ShapeClass::Decode, 4, &[100]),
        ),
        (
            key(OpKind::Convert, &[F32, q8]),
            shape(ShapeClass::Decode, 4, &[100]),
        ),
    ];
    for (k, sh) in &refused {
        assert!(p.query(&ctx, k, sh).is_none(), "{k:?}");
        assert!(p.prepare(&ctx, k, sh).is_err(), "{k:?}");
    }
}

/// `Gemm` at a prefill bucket and `Logits` at a vocabulary-sized output, each per type.
#[test]
fn gemm_and_logits_equal_gemv() {
    let (p, ctx, s) = open();
    let mut g = Gen(31);
    for ty in TYPES {
        let k = if ty == WeightType::Q4K { 256 } else { 64 };
        let scheme = QuantScheme::Ggml(ty.ggml());
        for (kind, class, bucket, active, n_out) in [
            (OpKind::Gemm, PREFILL, 40usize, 37usize, 9usize),
            (OpKind::Logits, ShapeClass::Decode, 2, 1, 300),
        ] {
            let op = p
                .prepare(
                    &ctx,
                    &key(kind, &[scheme, F32, F32]),
                    &shape(class, bucket as u32, &[n_out as u32, k as u32]),
                )
                .unwrap();
            let wb = weights(&mut g, ty, n_out, k);
            let xv = g.f32s(bucket * k);
            let w = Host::with(&p, &ctx, &wb);
            let x = Host::with(&p, &ctx, &le(&xv));
            let mut y = Host::new(&p, &ctx, bucket * n_out * 4);
            let ws = Host::new(&p, &ctx, op.workspace.0 as usize);
            let st = step(&p, &ctx, active);
            let (inputs, outputs, wv) = (
                [x.tensor(DType::F32, bucket, k)],
                [y.tensor(DType::F32, bucket, n_out)],
                [w.view(wb.len())],
            );
            let args = OpArgs {
                inputs: &inputs,
                outputs: &outputs,
                weights: &wv,
                step: st.slice(size_of::<StepDesc>()),
                workspace: Some(ws.slice(op.workspace.0 as usize)),
                statics: &[],
            };
            let wm = Weights::new(ty, &wb, n_out, k).unwrap();
            let mut a = Activations::new();
            a.quantize(ty.act(), &xv[..active * k], active, k).unwrap();
            let mut want = vec![0.0f32; active * n_out];
            gemv(Level::Portable, &wm, &a.view(), 0..n_out, &mut want).unwrap();
            for out in three_runs(&p, &ctx, &s, &op, &args, &mut y, 0xFF) {
                let got: Vec<f32> = out
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                assert_eq!(bits(&got[..active * n_out]), bits(&want), "{kind:?} {ty:?}");
                assert!(
                    got[active * n_out..bucket * n_out]
                        .iter()
                        .all(|v| v.to_bits() == u32::MAX)
                );
            }
        }
    }
}

#[test]
fn gate_up_act_equals_the_module() {
    let (p, ctx, s) = open();
    let mut g = Gen(32);
    let (bucket, active, n) = (24usize, 19usize, 70usize);
    for ty in TYPES {
        let k = if ty == WeightType::Q4K { 256 } else { 64 };
        let silu = OpKind::GateUpAct {
            act: GatedAct::Silu,
        };
        let op = p
            .prepare(
                &ctx,
                &key(silu, &[QuantScheme::Ggml(ty.ggml()), F32, F32]),
                &shape(PREFILL, bucket as u32, &[n as u32, k as u32]),
            )
            .unwrap();
        let wb = weights(&mut g, ty, 2 * n, k);
        let xv = g.f32s(bucket * k);
        let w = Host::with(&p, &ctx, &wb);
        let x = Host::with(&p, &ctx, &le(&xv));
        let mut y = Host::new(&p, &ctx, bucket * n * 4);
        let ws = Host::new(&p, &ctx, op.workspace.0 as usize);
        let st = step(&p, &ctx, active);
        let (inputs, outputs, wv) = (
            [x.tensor(DType::F32, bucket, k)],
            [y.tensor(DType::F32, bucket, n)],
            [w.view(wb.len())],
        );
        let args = OpArgs {
            inputs: &inputs,
            outputs: &outputs,
            weights: &wv,
            step: st.slice(size_of::<StepDesc>()),
            workspace: Some(ws.slice(op.workspace.0 as usize)),
            statics: &[],
        };
        let wm = Weights::new(ty, &wb, 2 * n, k).unwrap();
        let mut a = Activations::new();
        a.quantize(ty.act(), &xv[..active * k], active, k).unwrap();
        let mut want = vec![0.0f32; active * n];
        let mut scratch = vec![0.0f32; gate_up_scratch(active)];
        gate_up_act(Level::Portable, &wm, &a.view(), &mut want, &mut scratch).unwrap();
        for out in three_runs(&p, &ctx, &s, &op, &args, &mut y, 0xFF) {
            let got: Vec<f32> = out
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(bits(&got[..active * n]), bits(&want), "{ty:?}");
            assert!(
                got[active * n..bucket * n]
                    .iter()
                    .all(|v| v.to_bits() == u32::MAX)
            );
        }
        // A weight view that holds only the gate rows is refused at launch.
        let short = [w.view(wb.len() / 2)];
        assert!(
            p.launch(
                &ctx,
                &s,
                &op,
                &OpArgs {
                    weights: &short,
                    ..args
                }
            )
            .is_err()
        );
    }
}

#[test]
fn add_mul_and_convert_equal_the_module() {
    let (p, ctx, s) = open();
    let mut g = Gen(33);
    let (bucket, active, n) = (6usize, 4usize, 37usize);
    let av = g.f32s(bucket * n);
    let bv = g.f32s(bucket * n);
    let a = Host::with(&p, &ctx, &le(&av));
    let b = Host::with(&p, &ctx, &le(&bv));
    let st = step(&p, &ctx, active);
    for kind in [OpKind::Add, OpKind::Mul] {
        let op = p
            .prepare(
                &ctx,
                &key(kind, &[F32, F32, F32]),
                &shape(ShapeClass::Decode, bucket as u32, &[n as u32]),
            )
            .unwrap();
        let mut y = Host::new(&p, &ctx, bucket * n * 4);
        let (inputs, outputs) = (
            [
                a.tensor(DType::F32, bucket, n),
                b.tensor(DType::F32, bucket, n),
            ],
            [y.tensor(DType::F32, bucket, n)],
        );
        let args = OpArgs {
            inputs: &inputs,
            outputs: &outputs,
            step: st.slice(size_of::<StepDesc>()),
            ..OpArgs::none()
        };
        let mut want = vec![0.0f32; active * n];
        let f = if kind == OpKind::Add { add } else { mul };
        f(Level::Portable, &av, &bv, &mut want);
        for out in three_runs(&p, &ctx, &s, &op, &args, &mut y, 0xFF) {
            let got: Vec<f32> = out
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(bits(&got[..active * n]), bits(&want), "{kind:?}");
            assert!(
                got[active * n..bucket * n]
                    .iter()
                    .all(|v| v.to_bits() == u32::MAX)
            );
        }

        // In place: the output is the first input.
        let mut r = Host::with(&p, &ctx, &le(&av));
        let rt = r.tensor(DType::F32, bucket, n);
        let inplace = [rt, b.tensor(DType::F32, bucket, n)];
        p.launch(
            &ctx,
            &s,
            &op,
            &OpArgs {
                inputs: &inplace,
                outputs: &[rt],
                ..args
            },
        )
        .unwrap();
        assert_eq!(p.run(&ctx, &s).unwrap(), Run::Idle);
        let got = r.f32s(bucket * n);
        assert_eq!(bits(&got[..active * n]), bits(&want), "{kind:?} in place");
        assert_eq!(bits(&got[active * n..]), bits(&av[active * n..]));
    }
    // Convert, every pair.
    let floats = [DType::F32, DType::F16, DType::BF16];
    for from in floats {
        for to in floats {
            if from == to {
                continue;
            }
            let op = p
                .prepare(
                    &ctx,
                    &key(
                        OpKind::Convert,
                        &[QuantScheme::Plain(from), QuantScheme::Plain(to)],
                    ),
                    &shape(ShapeClass::Decode, bucket as u32, &[n as u32]),
                )
                .unwrap();
            let fw = from.bits() as usize / 8;
            let tw = to.bits() as usize / 8;
            let mut src = vec![0u8; bucket * n * fw];
            convert_rows(Level::Portable, DType::F32, &le(&av), from, &mut src);
            let x = Host::with(&p, &ctx, &src);
            let mut y = Host::new(&p, &ctx, bucket * n * tw);
            let (inputs, outputs) = ([x.tensor(from, bucket, n)], [y.tensor(to, bucket, n)]);
            let args = OpArgs {
                inputs: &inputs,
                outputs: &outputs,
                step: st.slice(size_of::<StepDesc>()),
                ..OpArgs::none()
            };
            let mut want = vec![0u8; active * n * tw];
            convert_rows(
                Level::Portable,
                from,
                &src[..active * n * fw],
                to,
                &mut want,
            );
            for out in three_runs(&p, &ctx, &s, &op, &args, &mut y, 0xEE) {
                assert_eq!(&out[..active * n * tw], &want[..], "{from:?} -> {to:?}");
                assert!(
                    out[active * n * tw..bucket * n * tw]
                        .iter()
                        .all(|&b| b == 0xEE)
                );
            }
            // An output of the wrong type is refused at launch.
            let wrong = [y.tensor(from, bucket, n)];
            assert!(
                p.launch(
                    &ctx,
                    &s,
                    &op,
                    &OpArgs {
                        outputs: &wrong,
                        ..args
                    }
                )
                .is_err()
            );
        }
    }
    assert_eq!(p.health(&ctx), DeviceHealth::Ok);
}

#[test]
fn row_gather_equals_the_module() {
    let (p, ctx, s) = open();
    let mut g = Gen(34);
    let (bucket, active, table_rows) = (8usize, 6usize, 13usize);
    for ty in TYPES {
        let width = if ty == WeightType::Q4K { 256 } else { 64 };
        let op = p
            .prepare(
                &ctx,
                &key(OpKind::RowGather, &[QuantScheme::Ggml(ty.ggml()), F32]),
                &shape(
                    ShapeClass::Decode,
                    bucket as u32,
                    &[table_rows as u32, width as u32],
                ),
            )
            .unwrap();
        let tb = weights(&mut g, ty, table_rows, width);
        let ids: Vec<i32> = (0..bucket)
            .map(|i| ((i * 5 + 3) % table_rows) as i32)
            .collect();
        let t = Host::with(&p, &ctx, &tb);
        let idh = Host::with(
            &p,
            &ctx,
            &ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>(),
        );
        let mut y = Host::new(&p, &ctx, bucket * width * 4);
        let st = step(&p, &ctx, active);
        let id_t =
            TensorArg::packed(idh.slice(bucket * 4), DType::I32, [bucket as u32, 1, 1, 1]).unwrap();
        let (inputs, outputs, wv) = (
            [id_t],
            [y.tensor(DType::F32, bucket, width)],
            [t.view(tb.len())],
        );
        let args = OpArgs {
            inputs: &inputs,
            outputs: &outputs,
            weights: &wv,
            step: st.slice(size_of::<StepDesc>()),
            ..OpArgs::none()
        };
        let mut want = vec![0.0f32; active * width];
        row_gather(
            Level::Portable,
            ty.ggml(),
            &tb,
            width,
            &ids[..active],
            &mut want,
        )
        .unwrap();
        for out in three_runs(&p, &ctx, &s, &op, &args, &mut y, 0xFF) {
            let got: Vec<f32> = out
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(bits(&got[..active * width]), bits(&want), "{ty:?}");
            assert!(
                got[active * width..bucket * width]
                    .iter()
                    .all(|v| v.to_bits() == u32::MAX)
            );
        }
    }
    assert_eq!(p.health(&ctx), DeviceHealth::Ok);
}

/// An ID outside the table fails the step: nothing is written, and the device's health turns
/// to the sticky error every kernel failure gives (COMP-E3-T02).
#[test]
fn row_gather_with_an_id_outside_the_table_fails() {
    let (p, ctx, s) = open();
    let (rows, width, table_rows) = (2usize, 32usize, 4usize);
    let op = p
        .prepare(
            &ctx,
            &key(OpKind::RowGather, &[QuantScheme::Ggml(GgmlType::Q8_0), F32]),
            &shape(
                ShapeClass::Decode,
                rows as u32,
                &[table_rows as u32, width as u32],
            ),
        )
        .unwrap();
    let t = Host::new(&p, &ctx, table_rows * 34);
    let idh = Host::with(
        &p,
        &ctx,
        &[1i32, 4]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let mut y = Host::new(&p, &ctx, rows * width * 4);
    y.bytes().fill(0xAB);
    let id_t = TensorArg::packed(idh.slice(rows * 4), DType::I32, [rows as u32, 1, 1, 1]).unwrap();
    let (inputs, outputs, wv) = (
        [id_t],
        [y.tensor(DType::F32, rows, width)],
        [t.view(table_rows * 34)],
    );
    p.launch(
        &ctx,
        &s,
        &op,
        &OpArgs {
            inputs: &inputs,
            outputs: &outputs,
            weights: &wv,
            ..OpArgs::none()
        },
    )
    .unwrap();
    assert!(p.run(&ctx, &s).is_err());
    assert!(y.bytes()[..rows * width * 4].iter().all(|&b| b == 0xAB));
    assert_ne!(p.health(&ctx), DeviceHealth::Ok);
}
