// SPDX-License-Identifier: MIT OR Apache-2.0

//! GEMV allocates nothing once its activation buffers are sized: quantising a step's rows and
//! running every output row, at each level the host can run (RFC-0040 §2.4; RFC-0060 §3.5).

use syntropy_kernels::CpuIsa;
use syntropy_kernels_cpu::gemv::{Activations, Level, WeightType, Weights, gemv};
use syntropy_testkit::guards::{CountingAlloc, alloc_count_this_thread};

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

/// The allocator counts at all: without that, a zero below would prove nothing.
#[test]
fn the_allocator_counts() {
    let before = alloc_count_this_thread();
    let boxed = Box::new(1_u64);
    assert_eq!(alloc_count_this_thread() - before, 1);
    drop(boxed);
}

#[test]
fn sized_calls_allocate_nothing() {
    let (n_out, k, rows) = (16, 256, 8);
    let isas = [
        CpuIsa::X86V2,
        CpuIsa::X86V3,
        CpuIsa::X86V4,
        CpuIsa::Arm80,
        CpuIsa::Arm82Dot,
        CpuIsa::Arm86I8mm,
    ];
    let mut a = Activations::with_capacity(rows, k);
    let x: Vec<f32> = (0..rows * k).map(|i| (i % 17) as f32 - 8.0).collect();
    let mut y = vec![0.0f32; rows * n_out];
    for ty in [
        WeightType::F16,
        WeightType::Bf16,
        WeightType::Q8_0,
        WeightType::Q4_0,
        WeightType::Q4K,
    ] {
        // Zero bytes are valid weights of every type: zero scales and codes.
        let bytes = vec![0u8; n_out * k / ty.block_elems() * ty.block_bytes()];
        let w = Weights::new(ty, &bytes, n_out, k).unwrap();
        for isa in isas {
            let level = Level::for_isa(isa);
            let before = alloc_count_this_thread();
            a.quantize(ty.act(), &x, rows, k).unwrap();
            gemv(level, &w, &a.view(), 0..n_out, &mut y).unwrap();
            assert_eq!(alloc_count_this_thread(), before, "{ty:?} {level:?}");
        }
    }
}

/// A `Gemv` launched through the provider, directly and from a replayed capture, allocates
/// nothing in steady state: the operands go onto the ring by value and the activations are
/// quantised into the launch's workspace.
#[test]
fn provider_launches_allocate_nothing() {
    use smallvec::smallvec;
    use syntropy_core::{
        BufferView, Bytes, DType, DeviceId, HostRegion, NodeId, QuantScheme, ViewKind,
    };
    use syntropy_kernels::{
        BufferSlice, KernelProvider, LayoutVersion, OpArgs, OpKey, OpKind, OpenOptions, PowerHint,
        ShapeClass, ShapeSig, StreamPlan, StreamPriority, TensorArg,
    };
    use syntropy_kernels_cpu::{CpuProvider, KillSwitches, Run};

    let (rows, n_out, k) = (4u32, 32u32, 256u32);
    let p = CpuProvider::new(KillSwitches::NONE);
    let ctx = p
        .open(
            DeviceId {
                node: NodeId([2; 32]),
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
                deterministic: true,
                power: PowerHint::Mains,
            },
        )
        .unwrap();
    let s = p.stream(&ctx, StreamPriority::Compute).unwrap();
    let key = OpKey {
        op: OpKind::Gemv,
        schemes: smallvec![QuantScheme::Ggml(syntropy_core::GgmlType::Q4_K)],
        layout: LayoutVersion::SOURCE,
        variant: 0,
    };
    let shape = ShapeSig {
        class: ShapeClass::Decode,
        rows,
        dims: smallvec![n_out, k],
    };
    let op = p.prepare(&ctx, &key, &shape).unwrap();
    let import = |words: &mut Vec<u32>| {
        p.import(
            &ctx,
            HostRegion {
                addr: words.as_mut_ptr() as u64,
                len: Bytes((words.len() * 4) as u64),
            },
        )
        .unwrap()
        .handle
    };
    let mut x: Vec<u32> = (0..rows * k)
        .map(|i| ((i % 13) as f32 - 6.0).to_bits())
        .collect();
    let mut y = vec![0u32; (rows * n_out) as usize];
    let mut ws = vec![0u32; (op.workspace.0 as usize).div_ceil(4)];
    let w = vec![0u8; (n_out * k / 256 * 144) as usize];
    let (xb, yb, wsb) = (import(&mut x), import(&mut y), import(&mut ws));
    let slice = |buffer, bytes: u32| BufferSlice {
        buffer,
        offset: Bytes(0),
        bytes: Bytes(u64::from(bytes)),
    };
    let inputs = [TensorArg::packed(slice(xb, rows * k * 4), DType::F32, [rows, k, 1, 1]).unwrap()];
    let outputs = [
        TensorArg::packed(slice(yb, rows * n_out * 4), DType::F32, [rows, n_out, 1, 1]).unwrap(),
    ];
    let weights = [BufferView {
        kind: ViewKind::Host,
        addr: w.as_ptr() as u64,
        len: Bytes(w.len() as u64),
    }];
    let args = OpArgs {
        inputs: &inputs,
        outputs: &outputs,
        weights: &weights,
        workspace: Some(slice(wsb, op.workspace.0 as u32)),
        ..OpArgs::none()
    };
    p.begin_capture(&ctx, &s).unwrap();
    p.launch(&ctx, &s, &op, &args).unwrap();
    let graph = p.end_capture(&ctx, &s).unwrap();
    // One step outside the count, for first-use work.
    p.replay(&ctx, &s, &graph).unwrap();
    assert_eq!(p.run(&ctx, &s).unwrap(), Run::Idle);
    let before = alloc_count_this_thread();
    for _ in 0..1000 {
        p.replay(&ctx, &s, &graph).unwrap();
        p.launch(&ctx, &s, &op, &args).unwrap();
        assert_eq!(p.run(&ctx, &s).unwrap(), Run::Idle);
    }
    assert_eq!(alloc_count_this_thread(), before);
    assert_eq!(p.health(&ctx), syntropy_kernels::DeviceHealth::Ok);
}
