// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tier manager's arena build (TIER-E1-T07; RFC-0020 §4.1, §4.4, §8.1) through the CUDA
//! provider on a real device. `tier` and `kernels-cuda` are both layer 2 (RFC-0001 §4.1), so this
//! test lives in the runner, which drives `Loading`, above both.
//!
//! The tests are `#[ignore]`d: run them with `cargo test -p syntropy-runner --test
//! tier_arenas_cuda -- --ignored` on a host with the NVIDIA driver. On a host without a CUDA
//! device each test returns after saying so.

use std::sync::{Arc, Mutex};

use smallvec::{SmallVec, smallvec};
use syntropy_core::error::codes;
use syntropy_core::{Bytes, DeviceId, ErrorClass, NodeId, TierId};
use syntropy_kernels::{DeviceCtx, KernelProvider, MemKind, OpenOptions, PowerHint, StreamPlan};
use syntropy_kernels_cuda::CudaProvider;
use syntropy_kernels_cuda::memory::MemEvent;
use syntropy_tier::{
    ArenaPlan, ArenaSpec, ContextReserve, PinClass, Purpose, Registrations, SegmentSpec,
    SlotGeometry, TierBudgetLine, VramFit, build_vram, open_measured,
};

const MIB: u64 = 1 << 20;

const VRAM: TierId = TierId {
    node: NodeId([0; 32]),
    ordinal: 0,
};

fn opts() -> OpenOptions {
    OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 1,
            collective: 0,
        },
        doorbell_slots: 8,
        capture: true,
        deterministic: true,
        power: PowerHint::Mains,
    }
}

fn arena(purpose: Purpose, bytes: u64, geometry: SlotGeometry) -> ArenaSpec {
    ArenaSpec {
        tier: VRAM,
        purpose,
        bytes: Bytes(bytes),
        max_bytes: Bytes(bytes),
        geometry,
        pin: PinClass::Locked,
        numa: None,
        registrations: Registrations::empty(),
        dma_devices: SmallVec::new(),
        pin_priority: 0,
        elastic: None,
    }
}

/// A small dense-and-MoE layout, 185 MiB in all, listed out of RFC-0020 §4.4's order.
fn plan(headroom: u64) -> ArenaPlan {
    let mut cache = arena(Purpose::ExpertCache, 64 * MIB, SlotGeometry::Bump);
    cache.elastic = Some(SegmentSpec {
        segment: Bytes(2 * MIB),
        min_segments: 4,
    });
    ArenaPlan {
        arenas: vec![
            cache,
            arena(
                Purpose::Weights,
                64 * MIB,
                SlotGeometry::Extents {
                    align: Bytes(2 * MIB),
                },
            ),
            arena(
                Purpose::Kv,
                32 * MIB,
                SlotGeometry::Pages {
                    page: Bytes(64 << 10),
                    count: 512,
                },
            ),
            arena(
                Purpose::ExpertStaging,
                16 * MIB,
                SlotGeometry::Ring {
                    slot: Bytes(4 * MIB),
                    count: 4,
                },
            ),
            arena(Purpose::Graphs, MIB, SlotGeometry::Bump),
            arena(Purpose::Scratch, 8 * MIB, SlotGeometry::Bump),
        ],
        nvme_extents: Vec::new(),
        per_tier: vec![(
            VRAM,
            TierBudgetLine {
                reserved: Bytes(185 * MIB),
                by_purpose: smallvec![],
                headroom: Bytes(headroom),
            },
        )],
    }
}

/// What [`open`] returns: the provider, the events its hook saw, the open device and its
/// measured context reserve.
type Opened = (
    CudaProvider,
    Arc<Mutex<Vec<MemEvent>>>,
    DeviceCtx,
    ContextReserve,
);

/// The provider with a hook that records every memory event, and device 0 opened through
/// `open_measured`; `None` on a host with no CUDA device.
fn open() -> Option<Opened> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&events);
    let p = CudaProvider::new()
        .with_alloc_hook(Arc::new(move |e: &MemEvent| seen.lock().unwrap().push(*e)));
    if p.enumerate().unwrap().is_empty() {
        eprintln!("no CUDA device: skipped");
        return None;
    }
    let dev = DeviceId {
        node: NodeId([0; 32]),
        ordinal: 0,
    };
    let (ctx, reserve) = open_measured(&p, dev, &opts()).unwrap();
    eprintln!(
        "device 0: {} bytes free before open, {} after; context reserve {} bytes",
        reserve.free_before.0,
        reserve.free_after.0,
        reserve.bytes().0
    );
    Some((p, events, ctx, reserve))
}

#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn vram_arenas_build_as_fixed_slabs_on_a_cuda_device() {
    let Some((p, events, ctx, reserve)) = open() else {
        return;
    };
    let plan = plan(256 * MIB);
    let built = build_vram(&p, &ctx, &plan, VRAM, reserve).unwrap();
    assert_eq!(built.report.fit, VramFit::All);
    assert_eq!(built.report.context_reserve, reserve);
    assert_eq!(built.report.needs.context, reserve.bytes());

    // The §4.4 order, one slab per arena.
    let order: Vec<Purpose> = built.arenas.iter().map(|a| a.purpose).collect();
    assert_eq!(
        order,
        [
            Purpose::Scratch,
            Purpose::Graphs,
            Purpose::Weights,
            Purpose::Kv,
            Purpose::ExpertStaging,
            Purpose::ExpertCache,
        ]
    );
    for a in &built.arenas {
        assert_eq!(a.buffer.kind, MemKind::Device);
        assert_eq!(
            a.buffer.bytes, plan.arenas[a.index].bytes,
            "{:?}",
            a.purpose
        );
        assert_ne!(a.buffer.addr, 0, "{:?}", a.purpose);
    }
    assert_eq!(
        built.get(1).unwrap().addr % (2 * MIB),
        0,
        "weights alignment"
    );
    for (i, a) in built.arenas.iter().enumerate() {
        for b in &built.arenas[i + 1..] {
            let (a, b) = (&a.buffer, &b.buffer);
            assert!(
                a.addr + a.bytes.0 <= b.addr || b.addr + b.bytes.0 <= a.addr,
                "{a:?} overlaps {b:?}"
            );
        }
    }

    // The provider allocated exactly the arenas, in that order, and nothing else.
    let seen: Vec<MemEvent> = events.lock().unwrap().clone();
    let handles: Vec<u64> = built.arenas.iter().map(|a| a.buffer.handle).collect();
    let allocated: Vec<u64> = seen
        .iter()
        .map(|e| match *e {
            MemEvent::Alloc {
                kind: MemKind::Device,
                bytes,
                handle,
            } => {
                let a = built.arenas.iter().find(|a| a.buffer.handle == handle);
                let a = a.expect("an allocation the build did not ask for");
                assert!(bytes >= a.buffer.bytes.0);
                handle
            }
            other => panic!("a memory event the build did not ask for: {other:?}"),
        })
        .collect();
    assert_eq!(allocated, handles);

    built.release(&p, &ctx);
    let seen: Vec<MemEvent> = events.lock().unwrap().clone();
    let released: Vec<u64> = seen[handles.len()..]
        .iter()
        .map(|e| match *e {
            MemEvent::Release { handle } => handle,
            other => panic!("unexpected event at release: {other:?}"),
        })
        .collect();
    assert_eq!(released, handles.into_iter().rev().collect::<Vec<_>>());
}

#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn a_layout_larger_than_the_device_fails_with_vram_short_and_allocates_nothing() {
    let Some((p, events, ctx, reserve)) = open() else {
        return;
    };
    // Headroom of all the free memory: the fixed arenas cannot fit.
    let plan = plan(reserve.free_before.0);
    let e = build_vram(&p, &ctx, &plan, VRAM, reserve).unwrap_err();
    assert_eq!(e.class, ErrorClass::Capacity);
    assert_eq!(e.code, codes::tier::VRAM_SHORT);
    assert!(
        events.lock().unwrap().is_empty(),
        "a short tier allocated memory"
    );
}

#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn a_tight_device_maps_fewer_expert_segments_and_reports_the_drift() {
    let Some((p, events, ctx, reserve)) = open() else {
        return;
    };
    // Leave room for the fixed arenas and 10 of the 32 expert segments of 2 MiB, plus 1 MiB.
    let fixed = 121 * MIB;
    let headroom = reserve.free_before.0 - reserve.bytes().0 - fixed - 21 * MIB;
    let plan = plan(headroom);
    let built = build_vram(&p, &ctx, &plan, VRAM, reserve).unwrap();
    assert_eq!(
        built.report.fit,
        VramFit::Reduced {
            expert: Bytes(20 * MIB)
        }
    );
    assert_eq!(built.get(0).unwrap().bytes, Bytes(20 * MIB));
    let d = built.report.drift().expect("a drift");
    assert_eq!(d.code, codes::tier::VRAM_EC_REDUCED);
    built.release(&p, &ctx);
    let seen = events.lock().unwrap();
    let allocs = seen
        .iter()
        .filter(|e| matches!(e, MemEvent::Alloc { .. }))
        .count();
    let releases = seen
        .iter()
        .filter(|e| matches!(e, MemEvent::Release { .. }))
        .count();
    assert_eq!((allocs, releases), (6, 6));
}
