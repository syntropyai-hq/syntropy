// SPDX-License-Identifier: MIT OR Apache-2.0

//! The COMP-E3-T22 ops allocate nothing once their buffers are sized, at each level the host
//! can run (RFC-0040 §2.4; RFC-0060 §3.5): `GateUpAct` computes into the caller's scratch, and
//! `RowGather`, `Add`, `Mul` and `Convert` write only their outputs. `Gemm` and `Logits` are
//! the `Gemv` kernel, which `gemv_no_alloc.rs` covers.

use syntropy_core::{DType, GgmlType};
use syntropy_kernels::CpuIsa;
use syntropy_kernels_cpu::gemv::{Activations, Level, WeightType, Weights};
use syntropy_kernels_cpu::ops::{add, convert_rows, gate_up_act, gate_up_scratch, mul, row_gather};
use syntropy_testkit::guards::{CountingAlloc, alloc_count_this_thread};

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

const ISAS: [CpuIsa; 6] = [
    CpuIsa::X86V2,
    CpuIsa::X86V3,
    CpuIsa::X86V4,
    CpuIsa::Arm80,
    CpuIsa::Arm82Dot,
    CpuIsa::Arm86I8mm,
];

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
    let (n, k, rows) = (100, 256, 20);
    let mut a = Activations::with_capacity(rows, k);
    let x: Vec<f32> = (0..rows * k).map(|i| (i % 17) as f32 - 8.0).collect();
    let mut y = vec![0.0f32; rows * n];
    let mut scratch = vec![0.0f32; gate_up_scratch(rows)];
    let mut z = vec![0.0f32; rows * k];
    let mut bytes = vec![0u8; rows * k * 4];
    let ids: Vec<i32> = (0..rows as i32).map(|i| i % 7).collect();
    let f32_table = vec![0u8; 7 * k * 4];
    for ty in [
        WeightType::F16,
        WeightType::Bf16,
        WeightType::Q8_0,
        WeightType::Q4_0,
        WeightType::Q4K,
    ] {
        // Zero bytes are valid weights of every type: zero scales and codes.
        let wb = vec![0u8; 2 * n * k / ty.block_elems() * ty.block_bytes()];
        let w = Weights::new(ty, &wb, 2 * n, k).unwrap();
        let table = &wb[..7 * k / ty.block_elems() * ty.block_bytes()];
        for isa in ISAS {
            let level = Level::for_isa(isa);
            let before = alloc_count_this_thread();
            a.quantize(ty.act(), &x, rows, k).unwrap();
            gate_up_act(level, &w, &a.view(), &mut y, &mut scratch).unwrap();
            row_gather(level, ty.ggml(), table, k, &ids, &mut z).unwrap();
            assert_eq!(alloc_count_this_thread(), before, "{ty:?} {level:?}");
        }
    }
    for isa in ISAS {
        let level = Level::for_isa(isa);
        let before = alloc_count_this_thread();
        add(level, &x, &x, &mut z);
        mul(level, &x, &x, &mut z);
        row_gather(level, GgmlType::F32, &f32_table, k, &ids, &mut z).unwrap();
        let half = rows * k / 2;
        let (from, to) = bytes.split_at_mut(half * 2);
        convert_rows(level, DType::F16, from, DType::BF16, &mut to[..half * 2]);
        convert_rows(
            level,
            DType::BF16,
            &from[..half * 2],
            DType::F32,
            &mut to[..half * 4],
        );
        convert_rows(level, DType::F32, &to[..half * 4], DType::F16, from);
        assert_eq!(alloc_count_this_thread(), before, "{level:?}");
    }
}
