// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MoeGrouped` at the `CPU` site through the CPU provider's `query`, `prepare` and `launch`,
//! directly and from a captured op list, for `Q4_K` and `Q8_0` experts (COMP-E3-T05;
//! RFC-0040 §7.3, §3.4, §3.5).
//!
//! The provider's output equals the `moe` module's portable kernels bit for bit (every level
//! does, `src/moe/tests.rs`), which the module's tests compare with `kernels::reference`.

#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

use smallvec::smallvec;
use syntropy_core::{
    AccumKind, ActQuant, BufferView, Bytes, DType, DeviceId, ErrorClass, GgmlType, HostRegion,
    NodeId, QuantScheme, ViewKind,
};
use syntropy_kernels::step::StepDesc;
use syntropy_kernels::{
    BufferSlice, DeviceCtx, KernelProvider, LayoutVersion, OpArgs, OpKey, OpKind, OpenOptions,
    PowerHint, ShapeClass, ShapeSig, Stream, StreamPlan, StreamPriority, TensorArg,
};
use syntropy_kernels_cpu::gemv::{Level, WeightType, Weights};
use syntropy_kernels_cpu::moe::{Expert, MoeShape, moe_grouped, workspace_bytes};
use syntropy_kernels_cpu::{CpuProvider, KillSwitches, Run};

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
        op: OpKind::MoeGrouped,
        schemes: smallvec![
            weight,
            QuantScheme::Plain(DType::F32),
            QuantScheme::Plain(DType::F32)
        ],
        layout: LayoutVersion::SOURCE,
        variant: 0,
    }
}

fn shape(rows: u32, m: &MoeShape) -> ShapeSig {
    ShapeSig {
        class: ShapeClass::Verify { window: 4 },
        rows,
        dims: smallvec![
            m.n_experts as u32,
            m.top_k as u32,
            m.hidden as u32,
            m.inter as u32
        ],
    }
}

/// SplitMix64.
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

/// `n_out` rows of `k` weights of `ty`, with random codes and moderate F16 scales.
fn weights(g: &mut Gen, ty: WeightType, n_out: usize, k: usize) -> Vec<u8> {
    let f16 = |v: f32| syntropy_kernels::reference::act_quant::f16_bits(v).to_le_bytes();
    let scales = if ty == WeightType::Q4K { 2 } else { 1 };
    let mut out = Vec::new();
    for _ in 0..n_out * k / ty.block_elems() {
        for _ in 0..scales {
            out.extend(f16(0.01 + g.unit().abs() * 0.01));
        }
        out.extend((2 * scales..ty.block_bytes()).map(|_| g.next() as u8));
    }
    out
}

/// Host memory the test owns, in `u64` words, imported into the device.
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

    fn put<const N: usize>(&mut self, vals: impl IntoIterator<Item = [u8; N]>) {
        for (dst, v) in self.bytes().as_chunks_mut::<N>().0.iter_mut().zip(vals) {
            *dst = v;
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

    fn view(&self, len: usize) -> BufferView {
        BufferView {
            kind: ViewKind::Host,
            addr: self.words.as_ptr() as u64,
            len: Bytes(len as u64),
        }
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|v| v.to_bits()).collect()
}

const SCHEMES: [(WeightType, GgmlType); 2] = [
    (WeightType::Q4K, GgmlType::Q4_K),
    (WeightType::Q8_0, GgmlType::Q8_0),
];

fn moe_shape(ty: WeightType) -> MoeShape {
    MoeShape {
        n_experts: 5,
        top_k: 2,
        hidden: 256,
        inter: if ty == WeightType::Q4K { 256 } else { 96 },
    }
}

#[test]
fn query_declares_each_schemes_numerics() {
    let (p, ctx, _) = open();
    for (ty, g, accumulate, act_in) in [
        (
            WeightType::Q4K,
            GgmlType::Q4_K,
            AccumKind::I32ThenF32 { max_block: 256 },
            ActQuant::Q8K256,
        ),
        (
            WeightType::Q8_0,
            GgmlType::Q8_0,
            AccumKind::I32ThenF32 { max_block: 32 },
            ActQuant::Q8Block32,
        ),
    ] {
        let m = moe_shape(ty);
        let k = key(QuantScheme::Ggml(g));
        let caps = p.query(&ctx, &k, &shape(4, &m)).unwrap();
        assert_eq!(caps.numerics.accumulate, accumulate, "{g:?}");
        assert_eq!(caps.numerics.act_in, act_in, "{g:?}");
        assert_eq!(caps.numerics.out, DType::F32);
        assert!(caps.deterministic && caps.capture_safe);
        assert!(caps.workspace.0 >= workspace_bytes(&m, 4) as u64);
        let op = p.prepare(&ctx, &k, &shape(4, &m)).unwrap();
        assert_eq!(op.workspace, caps.workspace);
    }
    // Another scheme, a width that is not whole blocks, top_k above n_experts, a fusion and
    // another layout are refused.
    let m = moe_shape(WeightType::Q4K);
    let refused = [
        (key(QuantScheme::Ggml(GgmlType::Q4_0)), shape(1, &m)),
        (
            key(QuantScheme::Ggml(GgmlType::Q4_K)),
            shape(1, &MoeShape { inter: 96, ..m }),
        ),
        (
            key(QuantScheme::Ggml(GgmlType::Q8_0)),
            shape(1, &MoeShape { top_k: 6, ..m }),
        ),
        (
            OpKey {
                variant: 1,
                ..key(QuantScheme::Ggml(GgmlType::Q8_0))
            },
            shape(1, &m),
        ),
        (
            OpKey {
                layout: LayoutVersion(7),
                ..key(QuantScheme::Ggml(GgmlType::Q8_0))
            },
            shape(1, &m),
        ),
    ];
    for (k, s) in &refused {
        assert!(p.query(&ctx, k, s).is_none(), "{k:?} {s:?}");
        assert_eq!(
            p.prepare(&ctx, k, s).unwrap_err().class,
            ErrorClass::Unsupported,
            "{k:?}"
        );
    }
}

/// One launch per scheme, directly and replayed from a capture, with a step descriptor that
/// makes fewer rows active than the bucket, then with a site mask.
#[test]
fn launches_equal_the_portable_kernels() {
    let (p, ctx, s) = open();
    let mut g = Gen(21);
    let bucket = 6usize;
    for (ty, gg) in SCHEMES {
        let m = moe_shape(ty);
        let op = p
            .prepare(&ctx, &key(QuantScheme::Ggml(gg)), &shape(bucket as u32, &m))
            .unwrap();
        let (n, h, i) = (m.n_experts, m.hidden, m.inter);
        let (gw, uw, dw) = (
            weights(&mut g, ty, n * i, h),
            weights(&mut g, ty, n * i, h),
            weights(&mut g, ty, n * h, i),
        );
        let mut hosts = [&gw, &uw, &dw].map(|w| {
            let mut host = Host::new(&p, &ctx, w.len());
            host.bytes()[..w.len()].copy_from_slice(w);
            host
        });
        let wviews = [
            hosts[0].view(gw.len()),
            hosts[1].view(uw.len()),
            hosts[2].view(dw.len()),
        ];
        let xv: Vec<f32> = (0..bucket * h).map(|_| g.unit() * 2.0).collect();
        let ids: Vec<i32> = (0..bucket)
            .flat_map(|r| {
                let a = (g.next() % n as u64) as i32;
                let b = (a + 1 + (r as i32 % (n as i32 - 1))) % n as i32;
                [a, b]
            })
            .collect();
        let mut x = Host::new(&p, &ctx, xv.len() * 4);
        x.put(xv.iter().map(|v| v.to_le_bytes()));
        let mut idh = Host::new(&p, &ctx, ids.len() * 4);
        idh.put(ids.iter().map(|v| v.to_le_bytes()));
        let site_mask: Vec<i32> = (0..n as i32).map(|e| i32::from(e % 2 == 0)).collect();
        let mut site = Host::new(&p, &ctx, n * 4);
        site.put(site_mask.iter().map(|v| v.to_le_bytes()));
        let out_len = bucket * m.top_k * h;
        let mut out = Host::new(&p, &ctx, out_len * 4);
        let ws = Host::new(&p, &ctx, op.workspace.0 as usize);
        let mut step = Host::new(&p, &ctx, size_of::<StepDesc>());
        let at = std::mem::offset_of!(StepDesc, rows_active);
        let active = 4usize;
        step.bytes()[at..at + 4].copy_from_slice(&(active as u32).to_le_bytes());

        let tx = TensorArg::packed(
            x.slice(xv.len() * 4),
            DType::F32,
            [bucket as u32, h as u32, 1, 1],
        )
        .unwrap();
        let tid = TensorArg::packed(
            idh.slice(ids.len() * 4),
            DType::I32,
            [bucket as u32, m.top_k as u32, 1, 1],
        )
        .unwrap();
        let tsite = TensorArg::packed(site.slice(n * 4), DType::I32, [n as u32, 1, 1, 1]).unwrap();
        let tout = TensorArg::packed(
            out.slice(out_len * 4),
            DType::F32,
            [bucket as u32, m.top_k as u32, h as u32, 1],
        )
        .unwrap();

        // What the module's portable kernels give for the active rows.
        let expected = |here: &dyn Fn(u32) -> bool| {
            let (gb, db) = (gw.len() / n, dw.len() / n);
            let mut want = vec![f32::from_bits(u32::MAX); active * m.top_k * h];
            let mut scratch = vec![0u8; workspace_bytes(&m, active)];
            moe_grouped(
                Level::Portable,
                &m,
                |e| {
                    let e2 = e as usize;
                    here(e).then(|| {
                        Expert::new(
                            Weights::new(ty, &gw[e2 * gb..(e2 + 1) * gb], i, h).unwrap(),
                            Weights::new(ty, &uw[e2 * gb..(e2 + 1) * gb], i, h).unwrap(),
                            Weights::new(ty, &dw[e2 * db..(e2 + 1) * db], h, i).unwrap(),
                        )
                        .unwrap()
                    })
                },
                &xv,
                &ids,
                active,
                &mut want,
                &mut scratch,
                None,
            )
            .unwrap();
            want
        };

        for masked in [false, true] {
            let inputs_all = [tx, tid, tsite];
            let inputs = if masked {
                &inputs_all[..]
            } else {
                &inputs_all[..2]
            };
            let outputs = [tout];
            let args = OpArgs {
                inputs,
                outputs: &outputs,
                weights: &wviews,
                step: step.slice(size_of::<StepDesc>()),
                workspace: Some(ws.slice(op.workspace.0 as usize)),
                statics: &[],
            };
            let want = if masked {
                expected(&|e| site_mask[e as usize] != 0)
            } else {
                expected(&|_| true)
            };

            // Directly: the active rows are written and the rest left as they were.
            out.bytes().fill(0xFF);
            p.launch(&ctx, &s, &op, &args).unwrap();
            assert_eq!(p.run(&ctx, &s).unwrap(), Run::Idle);
            let got = out.f32s(out_len);
            let active_len = active * m.top_k * h;
            assert_eq!(
                bits(&got[..active_len]),
                bits(&want),
                "{ty:?} masked {masked}"
            );
            assert!(
                got[active_len..].iter().all(|v| v.to_bits() == u32::MAX),
                "{ty:?} masked {masked}: an inactive row was written"
            );

            // From a capture, replayed twice.
            p.begin_capture(&ctx, &s).unwrap();
            p.launch(&ctx, &s, &op, &args).unwrap();
            let graph = p.end_capture(&ctx, &s).unwrap();
            for _ in 0..2 {
                out.bytes().fill(0xFF);
                p.replay(&ctx, &s, &graph).unwrap();
                assert_eq!(p.run(&ctx, &s).unwrap(), Run::Idle);
                assert_eq!(
                    bits(&out.f32s(active_len)),
                    bits(&want),
                    "{ty:?} masked {masked} replay"
                );
            }
        }
        // Keep the weight hosts alive until here.
        hosts
            .iter_mut()
            .for_each(|h| assert!(!h.bytes().is_empty()));
    }
}

#[test]
fn launch_refuses_mismatched_operands() {
    let (p, ctx, s) = open();
    let m = moe_shape(WeightType::Q8_0);
    let rows = 2usize;
    let op = p
        .prepare(
            &ctx,
            &key(QuantScheme::Ggml(GgmlType::Q8_0)),
            &shape(rows as u32, &m),
        )
        .unwrap();
    let row = |k: usize| k / 32 * 34;
    let (gu, d) = (
        m.n_experts * m.inter * row(m.hidden),
        m.n_experts * m.hidden * row(m.inter),
    );
    let (wg, wu, wd) = (
        Host::new(&p, &ctx, gu),
        Host::new(&p, &ctx, gu),
        Host::new(&p, &ctx, d),
    );
    let weights = [wg.view(gu), wu.view(gu), wd.view(d)];
    let x = Host::new(&p, &ctx, rows * m.hidden * 4);
    let ids = Host::new(&p, &ctx, rows * m.top_k * 4);
    let out = Host::new(&p, &ctx, rows * m.top_k * m.hidden * 4);
    let ws = Host::new(&p, &ctx, op.workspace.0 as usize);
    let tx = TensorArg::packed(
        x.slice(rows * m.hidden * 4),
        DType::F32,
        [rows as u32, m.hidden as u32, 1, 1],
    )
    .unwrap();
    let tid = TensorArg::packed(
        ids.slice(rows * m.top_k * 4),
        DType::I32,
        [rows as u32, m.top_k as u32, 1, 1],
    )
    .unwrap();
    let tout = TensorArg::packed(
        out.slice(rows * m.top_k * m.hidden * 4),
        DType::F32,
        [rows as u32, m.top_k as u32, m.hidden as u32, 1],
    )
    .unwrap();
    let ws_slice = Some(ws.slice(op.workspace.0 as usize));
    let launch = |inputs: &[TensorArg], weights: &[BufferView], workspace| {
        p.launch(
            &ctx,
            &s,
            &op,
            &OpArgs {
                inputs,
                outputs: &[tout],
                weights,
                workspace,
                ..OpArgs::none()
            },
        )
    };
    // The right operands are accepted: the zeroed ids route every row to expert 0.
    launch(&[tx, tid], &weights, ws_slice).unwrap();
    // ids as F32, a missing ids input, a missing weight, a short down view and a missing
    // workspace are refused.
    let ids_f32 = TensorArg {
        dtype: DType::F32,
        ..tid
    };
    assert!(launch(&[tx, ids_f32], &weights, ws_slice).is_err());
    assert!(launch(&[tx], &weights, ws_slice).is_err());
    assert!(launch(&[tx, tid], &weights[..2], ws_slice).is_err());
    let short = [weights[0], weights[1], wd.view(d - 1)];
    assert!(launch(&[tx, tid], &short, ws_slice).is_err());
    assert!(launch(&[tx, tid], &weights, None).is_err());
    // Only the accepted launch was queued, and it ran without error.
    assert_eq!(p.run(&ctx, &s).unwrap(), Run::Idle);
    assert_eq!(p.health(&ctx), syntropy_kernels::DeviceHealth::Ok);
}
