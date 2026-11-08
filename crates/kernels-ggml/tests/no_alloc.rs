// SPDX-License-Identifier: MIT OR Apache-2.0

//! A launch allocates nothing (COMP-E7-T02; RFC-0040 §8.2: "field writes, no allocation").
//!
//! `CountingAlloc` counts the Rust allocations of each thread, and the launch computes on the
//! calling thread. ggml's own C allocations do not pass through it. On the C side a launch does
//! not allocate by construction (unconfirmed static analysis of the pinned ggml): the template's
//! context is made at `prepare`, the compute plan's work buffer is the caller's workspace, and the
//! threadpool is the template's own, so `ggml_graph_compute` makes no disposable one.

mod common;

use common::{Buffers, F32, Gen, Launch, batch, f32_bytes, key, tensor};
use syntropy_core::{DType, GgmlType, QuantScheme};
use syntropy_kernels::{ActKind, GatedAct, OpKind, Scalar};
use syntropy_kernels_ggml::CpuOps;
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
fn launching_every_mapped_op_allocates_nothing() {
    let ops = CpuOps::new().unwrap();
    let b = Buffers::default();
    let mut g = Gen(5);
    let (rows, n, k) = (4u32, 32u32, 64u32);
    let rows_u = rows as usize;
    let q8 = QuantScheme::Ggml(GgmlType::Q8_0);
    // Q8_0 blocks with a finite F16 scale (0x2000 is 2⁻¹¹).
    let mut q8_weight = |out: u32| -> Vec<u8> {
        let mut w = Vec::new();
        for _ in 0..out * k / 32 {
            w.extend(0x2000u16.to_le_bytes());
            w.extend(g.bytes(32));
        }
        w
    };
    let x = b.add(&f32_bytes(&vec![0.5; rows_u * k as usize]));
    let y = b.add(&f32_bytes(&vec![1.5; rows_u * n as usize]));
    let wide = b.add(&f32_bytes(&vec![0.25; rows_u * n as usize]));
    let out = |len: u32| b.filled(rows_u * len as usize * 4, 0);
    let ids = b.add(
        &[3i32, 0, 1, 3]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect::<Vec<_>>(),
    );

    let mut launches = Vec::new();
    let mut add = |kind, schemes: &[QuantScheme], dims: &[u32], ins, ws, outs, statics| {
        let op = ops
            .prepare(&key(kind, schemes), &batch(rows, dims))
            .unwrap();
        let l = Launch::new(&b, &op, ins, ws, outs, statics);
        launches.push((op, l));
    };
    let w_mm = b.add(&q8_weight(n));
    let w_gu = b.add(&q8_weight(2 * n));
    add(
        OpKind::Gemv,
        &[q8],
        &[n, k],
        vec![tensor(&b, x, DType::F32, rows, k)],
        vec![b.view(w_mm)],
        vec![tensor(&b, out(n), DType::F32, rows, n)],
        vec![],
    );
    add(
        OpKind::GateUpAct {
            act: GatedAct::Silu,
        },
        &[q8],
        &[n, k],
        vec![tensor(&b, x, DType::F32, rows, k)],
        vec![b.view(w_gu)],
        vec![tensor(&b, out(n), DType::F32, rows, n)],
        vec![],
    );
    add(
        OpKind::RmsNorm,
        &[F32],
        &[n],
        vec![tensor(&b, y, DType::F32, rows, n)],
        vec![b.view(b.add(&f32_bytes(&vec![1.0; n as usize])))],
        vec![tensor(&b, out(n), DType::F32, rows, n)],
        vec![Scalar::F32(1e-6)],
    );
    add(
        OpKind::Act(ActKind::Silu),
        &[F32],
        &[n],
        vec![tensor(&b, y, DType::F32, rows, n)],
        vec![],
        vec![tensor(&b, out(n), DType::F32, rows, n)],
        vec![],
    );
    for kind in [OpKind::Add, OpKind::Mul] {
        add(
            kind,
            &[F32],
            &[n],
            vec![
                tensor(&b, y, DType::F32, rows, n),
                tensor(&b, wide, DType::F32, rows, n),
            ],
            vec![],
            vec![tensor(&b, out(n), DType::F32, rows, n)],
            vec![],
        );
    }
    let table = b.add(&f32_bytes(&vec![2.0; 4 * n as usize]));
    let ids_arg = tensor(&b, ids, DType::I32, rows, 1);
    let ids_arg =
        syntropy_kernels::TensorArg::packed(ids_arg.buf, DType::I32, [rows, 1, 1, 1]).unwrap();
    add(
        OpKind::RowGather,
        &[F32],
        &[4, n],
        vec![ids_arg],
        vec![b.view(table)],
        vec![tensor(&b, out(n), DType::F32, rows, n)],
        vec![],
    );
    let half = b.filled(rows_u * n as usize * 2, 0);
    add(
        OpKind::Convert,
        &[F32, F32, QuantScheme::Plain(DType::BF16)],
        &[n],
        vec![tensor(&b, y, DType::F32, rows, n)],
        vec![],
        vec![tensor(&b, half, DType::BF16, rows, n)],
        vec![],
    );
    // A warm-up launch of each, then the counted ones.
    for (op, l) in &launches {
        l.run(&ops, op, &b).unwrap();
    }
    let before = alloc_count_this_thread();
    for _ in 0..50 {
        for (op, l) in &launches {
            l.run(&ops, op, &b).unwrap();
        }
    }
    assert_eq!(alloc_count_this_thread() - before, 0);
    assert_eq!(launches.len(), 8);
}
