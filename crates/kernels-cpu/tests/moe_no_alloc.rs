// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MoeGrouped` at the `CPU` site allocates nothing given its workspace: grouping the slots,
//! gathering and quantising each expert's rows and running its tasks, at each level the host
//! can run (RFC-0040 §2.4; RFC-0060 §3.5; COMP-E3-T05).

use syntropy_kernels::CpuIsa;
use syntropy_kernels_cpu::gemv::{Level, WeightType, Weights};
use syntropy_kernels_cpu::moe::{Expert, MoeShape, WeightReads, moe_grouped, workspace_bytes};
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
fn calls_with_a_workspace_allocate_nothing() {
    let m = MoeShape {
        n_experts: 4,
        top_k: 2,
        hidden: 256,
        inter: 256,
    };
    let rows = 9;
    let x: Vec<f32> = (0..rows * m.hidden)
        .map(|i| (i % 17) as f32 - 8.0)
        .collect();
    let ids: Vec<i32> = (0..rows * m.top_k)
        .map(|i| (i % m.n_experts) as i32)
        .collect();
    let mut out = vec![0.0f32; rows * m.top_k * m.hidden];
    let mut ws = vec![0u8; workspace_bytes(&m, rows)];
    let reads = WeightReads::new(m.n_experts);
    for ty in [WeightType::Q8_0, WeightType::Q4K] {
        // Zero bytes are valid weights of both types: zero scales and codes.
        let gate = vec![0u8; m.inter * m.hidden / ty.block_elems() * ty.block_bytes()];
        let down = vec![0u8; m.hidden * m.inter / ty.block_elems() * ty.block_bytes()];
        let expert = |_| {
            Some(
                Expert::new(
                    Weights::new(ty, &gate, m.inter, m.hidden).unwrap(),
                    Weights::new(ty, &gate, m.inter, m.hidden).unwrap(),
                    Weights::new(ty, &down, m.hidden, m.inter).unwrap(),
                )
                .unwrap(),
            )
        };
        for isa in [
            CpuIsa::X86V2,
            CpuIsa::X86V3,
            CpuIsa::X86V4,
            CpuIsa::Arm80,
            CpuIsa::Arm82Dot,
            CpuIsa::Arm86I8mm,
        ] {
            let level = Level::for_isa(isa);
            let before = alloc_count_this_thread();
            moe_grouped(
                level,
                &m,
                expert,
                &x,
                &ids,
                rows,
                &mut out,
                &mut ws,
                Some(&reads),
            )
            .unwrap();
            assert_eq!(alloc_count_this_thread(), before, "{ty:?} {level:?}");
        }
    }
}
