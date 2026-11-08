// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the arena build (TIER-E1-T07), against a fake provider whose allocation hook sees
//! every `alloc` and `free`, and a fake OS adapter. The real-device run is
//! `crates/runner/tests/tier_arenas_cuda.rs`.

use std::ops::Range;
use std::sync::{Mutex, OnceLock};

use proptest::prelude::*;
use smallvec::{SmallVec, smallvec};
use syntropy_core::resource::{Backend, DeviceClass, DeviceKind};
use syntropy_core::{HostRegion, Nanos, NodeId};
use syntropy_kernels::{
    BufferSlice, CapturedGraph, DeviceFeatures, DeviceHealth, DeviceMemoryInfo, Event, HostFlag,
    LayoutVersion, MemoryDomain, OpArgs, OpCaps, OpKey, OpRegistry, PowerHint, PreparedOp,
    ProviderBuild, ProviderDevice, ProviderId, Registration, ShapeSig, Stream, StreamPlan,
    StreamPriority, unsupported,
};

use super::*;
use crate::arena::{PinClass, Registrations, SegmentSpec, TierBudgetLine};
use crate::os::{MemSnapshot, PinLimits};

const MIB: u64 = 1 << 20;

// ---- the fake provider -----------------------------------------------------------------------

/// What the fake provider's allocation hook saw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ev {
    Alloc { handle: u64, req: AllocRequest },
    Free { handle: u64 },
}

#[derive(Debug, Default)]
struct State {
    /// Every `alloc` and `free`, in order: the allocation hook.
    hook: Vec<Ev>,
    /// Free memory `enumerate` reports before and after `open`.
    free_before: u64,
    free_after: u64,
    /// Whether `memory_info` is implemented.
    memory_info: bool,
    opened: bool,
    next: u64,
    /// Fail the `alloc` call with this index (0-based).
    fail_on: Option<usize>,
    allocs: usize,
}

/// A provider that holds no memory: `alloc` hands out disjoint addresses at the requested
/// alignment and reports every call to its hook.
struct FakeProvider(Mutex<State>);

impl FakeProvider {
    fn new(free_before: u64, free_after: u64) -> Self {
        FakeProvider(Mutex::new(State {
            free_before,
            free_after,
            next: 0x1000_0000,
            ..State::default()
        }))
    }
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.0.lock().unwrap()
    }
    fn events(&self) -> Vec<Ev> {
        self.state().hook.clone()
    }
    fn allocs(&self) -> Vec<AllocRequest> {
        self.events()
            .into_iter()
            .filter_map(|e| match e {
                Ev::Alloc { req, .. } => Some(req),
                Ev::Free { .. } => None,
            })
            .collect()
    }
    /// Handles allocated and not yet freed.
    fn live(&self) -> Vec<u64> {
        let mut live = Vec::new();
        for e in self.events() {
            match e {
                Ev::Alloc { handle, .. } => live.push(handle),
                Ev::Free { handle } => {
                    let at = live.iter().position(|&h| h == handle);
                    live.remove(at.expect("freed a handle that is not live"));
                }
            }
        }
        live
    }
}

fn record(free: u64) -> ProviderDevice {
    ProviderDevice {
        local: 0,
        uuid: None,
        pci: None,
        kind: DeviceKind::DiscreteGpu,
        name: String::from("fake"),
        class: DeviceClass::Cuda { sm: 89 },
        features: DeviceFeatures::empty(),
        memory: DeviceMemoryInfo {
            total: Bytes(16 << 30),
            free: Bytes(free),
            budget: None,
            host_visible_device_local: None,
            max_alloc: Bytes(u64::MAX),
            unified: false,
        },
        numa: None,
    }
}

#[expect(
    clippy::result_large_err,
    reason = "the trait's methods return core::Error"
)]
fn not_used<T>() -> Result<T, Error> {
    Err(Error::new(
        ErrorClass::Internal,
        codes::kernels::OP_UNSUPPORTED,
        "not used by these tests",
    ))
}

impl KernelProvider for FakeProvider {
    fn id(&self) -> ProviderId {
        ProviderId::new("fake")
    }
    fn backend(&self) -> Backend {
        Backend::Cuda
    }
    fn build(&self) -> ProviderBuild {
        ProviderBuild {
            version: 1,
            module: None,
            toolchain: Default::default(),
            driver: None,
        }
    }
    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Cuda
    }
    fn enumerate(&self) -> Result<Vec<ProviderDevice>, Error> {
        let s = self.state();
        Ok(vec![record(if s.opened {
            s.free_after
        } else {
            s.free_before
        })])
    }
    fn open(&self, dev: DeviceId, _: &OpenOptions) -> Result<DeviceCtx, Error> {
        if dev.ordinal != 0 {
            return Err(unsupported());
        }
        self.state().opened = true;
        Ok(DeviceCtx::new(()))
    }
    fn memory_info(&self, _: &DeviceCtx) -> Result<DeviceMemoryInfo, Error> {
        let s = self.state();
        if s.memory_info {
            // One byte less than `enumerate` says, so a test sees which one was read.
            Ok(record(s.free_after - 1).memory)
        } else {
            Err(unsupported())
        }
    }
    fn ops(&self) -> &OpRegistry {
        static EMPTY: OnceLock<OpRegistry> = OnceLock::new();
        EMPTY.get_or_init(OpRegistry::new)
    }
    fn query(&self, _: &DeviceCtx, _: &OpKey, _: &ShapeSig) -> Option<OpCaps> {
        None
    }
    fn alloc(&self, _: &DeviceCtx, req: AllocRequest) -> Result<DeviceBuffer, Error> {
        let mut s = self.state();
        let n = s.allocs;
        s.allocs += 1;
        if s.fail_on == Some(n) {
            return Err(Error::new(
                ErrorClass::Capacity,
                codes::kernels::OP_UNSUPPORTED,
                "out of device memory",
            ));
        }
        let addr = s.next.next_multiple_of(req.align.0.max(1));
        s.next = addr + req.bytes.0;
        s.hook.push(Ev::Alloc { handle: addr, req });
        Ok(DeviceBuffer {
            addr,
            bytes: req.bytes,
            kind: req.kind,
            handle: addr,
        })
    }
    fn free(&self, _: &DeviceCtx, buf: DeviceBuffer) {
        self.state().hook.push(Ev::Free { handle: buf.handle });
    }
    fn register_host(&self, _: &DeviceCtx, _: HostRegion) -> Result<Registration, Error> {
        not_used()
    }
    fn import(&self, _: &DeviceCtx, _: HostRegion) -> Result<DeviceBuffer, Error> {
        not_used()
    }
    fn repack(&self, _: &OpKey, _: &[u8], _: &mut [u8]) -> Option<Result<LayoutVersion, Error>> {
        None
    }
    fn repacked_len(&self, _: &OpKey, _: Bytes) -> Option<Bytes> {
        None
    }
    fn stream(&self, _: &DeviceCtx, _: StreamPriority) -> Result<Stream, Error> {
        not_used()
    }
    fn event(&self, _: &DeviceCtx) -> Result<Event, Error> {
        not_used()
    }
    fn record(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> {
        not_used()
    }
    fn wait(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> {
        not_used()
    }
    fn signal_host(&self, _: &DeviceCtx, _: &Stream, _: HostFlag, _: u32) -> Result<(), Error> {
        not_used()
    }
    fn poll(&self, _: &DeviceCtx, _: &Event) -> Result<bool, Error> {
        not_used()
    }
    fn elapsed(&self, _: &DeviceCtx, _: &Event, _: &Event) -> Option<Nanos> {
        None
    }
    fn synchronize(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> {
        not_used()
    }
    fn copy(&self, _: &DeviceCtx, _: &Stream, _: BufferSlice, _: BufferSlice) -> Result<(), Error> {
        not_used()
    }
    fn prepare(&self, _: &DeviceCtx, _: &OpKey, _: &ShapeSig) -> Result<PreparedOp, Error> {
        not_used()
    }
    fn launch(
        &self,
        _: &DeviceCtx,
        _: &Stream,
        _: &PreparedOp,
        _: &OpArgs<'_>,
    ) -> Result<(), Error> {
        not_used()
    }
    fn begin_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> {
        not_used()
    }
    fn end_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<CapturedGraph, Error> {
        not_used()
    }
    fn replay(&self, _: &DeviceCtx, _: &Stream, _: &CapturedGraph) -> Result<(), Error> {
        not_used()
    }
    fn health(&self, _: &DeviceCtx) -> DeviceHealth {
        DeviceHealth::Ok
    }
}

// ---- plans -----------------------------------------------------------------------------------

const VRAM: TierId = TierId {
    node: NodeId([0; 32]),
    ordinal: 0,
};
const HOST: TierId = TierId {
    node: NodeId([0; 32]),
    ordinal: 2,
};
const DEV: DeviceId = DeviceId {
    node: NodeId([0; 32]),
    ordinal: 0,
};

fn opts() -> OpenOptions {
    OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 0,
            collective: 0,
        },
        doorbell_slots: 8,
        capture: false,
        deterministic: true,
        power: PowerHint::Mains,
    }
}

fn arena(tier: TierId, purpose: Purpose, bytes: u64, geometry: SlotGeometry) -> ArenaSpec {
    ArenaSpec {
        tier,
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

/// An expert cache of `segments` segments of `segment` MiB, at least `min` of them.
fn expert_cache(segments: u64, segment: u64, min: u32) -> ArenaSpec {
    let mut a = arena(
        VRAM,
        Purpose::ExpertCache,
        segments * segment * MIB,
        SlotGeometry::Bump,
    );
    a.elastic = Some(SegmentSpec {
        segment: Bytes(segment * MIB),
        min_segments: min,
    });
    a
}

/// A dense-and-MoE plan with the arenas in an order unlike RFC-0020 §4.4's, a `HostPageable`
/// weights arena on another tier, and 512 MiB of headroom on the VRAM tier.
fn plan() -> ArenaPlan {
    let extents = |a: u64| SlotGeometry::Extents { align: Bytes(a) };
    ArenaPlan {
        arenas: vec![
            expert_cache(16, 32, 2),
            arena(VRAM, Purpose::Weights, 1024 * MIB, extents(2 * MIB)),
            arena(
                VRAM,
                Purpose::Kv,
                256 * MIB,
                SlotGeometry::Pages {
                    page: Bytes(64 << 10),
                    count: 4096,
                },
            ),
            arena(HOST, Purpose::Weights, 64 * MIB, extents(4096)),
            arena(
                VRAM,
                Purpose::ExpertStaging,
                64 * MIB,
                SlotGeometry::Ring {
                    slot: Bytes(16 * MIB),
                    count: 4,
                },
            ),
            arena(VRAM, Purpose::Graphs, 32 * MIB, SlotGeometry::Bump),
            arena(VRAM, Purpose::Scratch, 128 * MIB, SlotGeometry::Bump),
        ],
        nvme_extents: Vec::new(),
        per_tier: vec![
            (
                VRAM,
                TierBudgetLine {
                    reserved: Bytes(2016 * MIB),
                    by_purpose: smallvec![],
                    headroom: Bytes(512 * MIB),
                },
            ),
            (HOST, TierBudgetLine::default()),
        ],
    }
}

/// Fixed arenas of `plan()`: 128 + 32 + 1024 + 256 + 64 MiB.
const FIXED: u64 = 1504 * MIB;
const EXPERT: u64 = 512 * MIB;
const HEADROOM: u64 = 512 * MIB;
const CTX: u64 = 300 * MIB;

/// A provider and an open device with a context reserve of 300 MiB and `room` bytes free
/// before `open`.
fn device(room: u64) -> (FakeProvider, DeviceCtx, ContextReserve) {
    let p = FakeProvider::new(room, room - CTX);
    let (ctx, reserve) = open_measured(&p, DEV, &opts()).unwrap();
    (p, ctx, reserve)
}

fn purpose_of(plan: &ArenaPlan, built: &VramArenas, handle: u64) -> Purpose {
    let a = built
        .arenas
        .iter()
        .find(|a| a.buffer.handle == handle)
        .unwrap();
    assert_eq!(plan.arenas[a.index].purpose, a.purpose);
    a.purpose
}

// ---- the context reserve ---------------------------------------------------------------------

#[test]
fn the_context_reserve_is_free_memory_before_open_less_after() {
    let (_p, _ctx, r) = device(8 << 30);
    assert_eq!(r.free_before, Bytes(8 << 30));
    assert_eq!(r.free_after, Bytes((8 << 30) - CTX));
    assert_eq!(r.bytes(), Bytes(CTX));
}

#[test]
fn memory_info_is_read_after_open_where_the_provider_has_it() {
    let p = FakeProvider::new(8 << 30, 7 << 30);
    p.state().memory_info = true;
    let (_ctx, r) = open_measured(&p, DEV, &opts()).unwrap();
    assert_eq!(r.free_after, Bytes((7 << 30) - 1));
    assert_eq!(r.bytes(), Bytes((1 << 30) + 1));
}

#[test]
fn memory_freed_by_another_program_during_open_gives_a_zero_reserve() {
    let p = FakeProvider::new(4 << 30, 5 << 30);
    let (_ctx, r) = open_measured(&p, DEV, &opts()).unwrap();
    assert_eq!(r.bytes(), Bytes(0));
}

#[test]
fn a_device_the_provider_does_not_have_is_refused_before_open() {
    let p = FakeProvider::new(1 << 30, 1 << 30);
    let dev = DeviceId { ordinal: 3, ..DEV };
    let e = open_measured(&p, dev, &opts()).unwrap_err();
    assert_eq!(e.class, ErrorClass::Unsupported);
    assert!(!p.state().opened, "open ran for a device that is not there");
}

// ---- Vram: order, sizes and the allocation hook ----------------------------------------------

#[test]
fn vram_arenas_are_allocated_in_the_loading_order_of_rfc_0020_4_4() {
    let plan = plan();
    let (p, ctx, reserve) = device(8 << 30);
    let built = build_vram(&p, &ctx, &plan, VRAM, reserve).unwrap();
    let order: Vec<Purpose> = p
        .events()
        .into_iter()
        .map(|e| match e {
            Ev::Alloc { handle, .. } => purpose_of(&plan, &built, handle),
            Ev::Free { .. } => panic!("nothing is freed while building"),
        })
        .collect();
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
    built.release(&p, &ctx);
}

#[test]
fn every_vram_arena_is_one_device_slab_of_its_planned_size_and_nothing_else_is_allocated() {
    let plan = plan();
    let (p, ctx, reserve) = device(8 << 30);
    let built = build_vram(&p, &ctx, &plan, VRAM, reserve).unwrap();
    // One allocation per VRAM arena of the plan, and none for the host arena.
    let vram: Vec<usize> = (0..plan.arenas.len())
        .filter(|&i| plan.arenas[i].tier == VRAM)
        .collect();
    let allocs = p.allocs();
    assert_eq!(allocs.len(), vram.len());
    for i in vram {
        let a = &plan.arenas[i];
        let buf = built.get(i).expect("every VRAM arena has a slab");
        assert_eq!(buf.bytes, a.bytes, "{:?}", a.purpose);
        assert_eq!(buf.kind, MemKind::Device);
        let req = allocs.iter().find(|r| r.bytes == a.bytes).unwrap();
        assert!(!req.reserve_only, "the fixed-slab path reserves nothing");
        assert_eq!(buf.addr % req.align.0, 0);
    }
    assert!(built.get(3).is_none(), "the HostPageable arena got a slab");
    // The weights slab keeps its extents' 2 MiB alignment; other geometries use SLAB_ALIGN.
    assert_eq!(built.get(1).unwrap().addr % (2 * MIB), 0);
    let kv = allocs.iter().find(|r| r.bytes.0 == 256 * MIB).unwrap();
    assert_eq!(kv.align, SLAB_ALIGN);
    assert_eq!(built.report.fit, VramFit::All);
    assert_eq!(built.report.expert_mapped, Bytes(EXPERT));
    assert!(built.report.drift().is_none());
    assert!(!built.report.segments);
    built.release(&p, &ctx);
}

#[test]
fn release_frees_every_slab_in_reverse_order() {
    let plan = plan();
    let (p, ctx, reserve) = device(8 << 30);
    let built = build_vram(&p, &ctx, &plan, VRAM, reserve).unwrap();
    let allocated: Vec<u64> = built.arenas.iter().map(|a| a.buffer.handle).collect();
    built.release(&p, &ctx);
    let freed: Vec<u64> = p
        .events()
        .into_iter()
        .filter_map(|e| match e {
            Ev::Free { handle } => Some(handle),
            Ev::Alloc { .. } => None,
        })
        .collect();
    assert_eq!(freed, allocated.into_iter().rev().collect::<Vec<_>>());
    assert!(p.live().is_empty());
}

#[test]
fn a_failed_allocation_frees_what_was_built_and_names_the_arena() {
    let plan = plan();
    let (p, ctx, reserve) = device(8 << 30);
    // The fourth allocation in §4.4 order is `Kv`.
    p.state().fail_on = Some(3);
    let e = build_vram(&p, &ctx, &plan, VRAM, reserve).unwrap_err();
    assert_eq!(e.class, ErrorClass::Capacity);
    let ctx_value = |k: &str| {
        e.context
            .iter()
            .find(|(key, _)| key.as_ref() == k)
            .map(|(_, v)| v.clone())
    };
    assert_eq!(ctx_value("purpose").as_deref(), Some("Kv"));
    assert_eq!(ctx_value("arena").as_deref(), Some("2"));
    assert_eq!(p.allocs().len(), 3);
    assert!(p.live().is_empty(), "slabs left after a failed build");
}

#[test]
fn a_zero_byte_arena_gets_no_slab() {
    let mut plan = plan();
    plan.arenas[5].bytes = Bytes(0); // Graphs
    plan.arenas[5].max_bytes = Bytes(0);
    let (p, ctx, reserve) = device(8 << 30);
    let built = build_vram(&p, &ctx, &plan, VRAM, reserve).unwrap();
    assert!(built.get(5).is_none());
    assert_eq!(p.allocs().len(), 5);
    assert!(p.allocs().iter().all(|r| r.bytes.0 > 0));
    built.release(&p, &ctx);
}

#[test]
fn arenas_of_another_vram_tier_are_left_to_their_own_device() {
    let mut plan = plan();
    let other = TierId { ordinal: 1, ..VRAM };
    plan.arenas[2].tier = other; // Kv
    let (p, ctx, reserve) = device(8 << 30);
    let built = build_vram(&p, &ctx, &plan, VRAM, reserve).unwrap();
    assert!(built.get(2).is_none());
    assert_eq!(built.report.needs.fixed, Bytes(FIXED - 256 * MIB));
    built.release(&p, &ctx);
}

// ---- Vram: the §8.1 check --------------------------------------------------------------------

#[test]
fn the_vram_needs_are_the_plans_sizes_with_the_measured_reserve() {
    let n = vram_needs(&plan(), VRAM, Bytes(CTX));
    assert_eq!(
        n,
        VramNeeds {
            context: Bytes(CTX),
            fixed: Bytes(FIXED),
            expert: Bytes(EXPERT),
            min_expert: Bytes(64 * MIB),
            segment: Bytes(32 * MIB),
            headroom: Bytes(HEADROOM),
        }
    );
    assert_eq!(n.need_fix(), Bytes(CTX + FIXED));
}

#[test]
fn everything_fits_with_exactly_enough_free_memory() {
    let free = CTX + FIXED + EXPERT + HEADROOM;
    let (p, ctx, reserve) = device(free);
    let built = build_vram(&p, &ctx, &plan(), VRAM, reserve).unwrap();
    assert_eq!(built.report.fit, VramFit::All);
    assert_eq!(built.report.context_reserve, reserve);
    built.release(&p, &ctx);
}

#[test]
fn one_byte_short_maps_the_expert_cache_in_fewer_whole_segments_and_reports_drift() {
    let plan = plan();
    let free = CTX + FIXED + EXPERT + HEADROOM - 1;
    let (p, ctx, reserve) = device(free);
    let built = build_vram(&p, &ctx, &plan, VRAM, reserve).unwrap();
    // 511 MiB + (1 MiB − 1) of room holds 15 segments of 32 MiB.
    assert_eq!(
        built.report.fit,
        VramFit::Reduced {
            expert: Bytes(15 * 32 * MIB)
        }
    );
    assert_eq!(built.get(0).unwrap().bytes, Bytes(15 * 32 * MIB));
    assert_eq!(built.arenas.last().unwrap().planned, Bytes(EXPERT));
    let d = built.report.drift().expect("a reduced cache is a drift");
    assert_eq!(d.class, ErrorClass::Drift);
    assert_eq!(d.code, codes::tier::VRAM_EC_REDUCED);
    let usable = d
        .context
        .iter()
        .find(|(k, _)| k.as_ref() == "vram_usable")
        .map(|(_, v)| v.clone());
    assert_eq!(usable, Some((CTX + FIXED + 15 * 32 * MIB).to_string()));
    built.release(&p, &ctx);
}

#[test]
fn the_expert_cache_never_drops_below_its_minimum_segments() {
    let free = CTX + FIXED + HEADROOM + 64 * MIB;
    let (p, ctx, reserve) = device(free);
    let built = build_vram(&p, &ctx, &plan(), VRAM, reserve).unwrap();
    assert_eq!(
        built.report.fit,
        VramFit::Reduced {
            expert: Bytes(64 * MIB)
        }
    );
    built.release(&p, &ctx);
}

#[test]
fn below_the_fixed_arenas_and_min_ec_loading_fails_with_vram_short_and_allocates_nothing() {
    let free = CTX + FIXED + HEADROOM + 64 * MIB - 1;
    let (p, ctx, reserve) = device(free);
    let e = build_vram(&p, &ctx, &plan(), VRAM, reserve).unwrap_err();
    assert_eq!(e.class, ErrorClass::Capacity);
    assert_eq!(e.code, codes::tier::VRAM_SHORT);
    assert_eq!(e.retry, RetryHint::OnNewPlan);
    assert!(p.events().is_empty(), "a short tier allocated memory");
}

#[test]
fn a_fixed_expert_cache_cannot_shrink() {
    let mut plan = plan();
    plan.arenas[0].elastic = None;
    let n = vram_needs(&plan, VRAM, Bytes(CTX));
    assert_eq!(n.min_expert, Bytes(EXPERT));
    assert_eq!(
        fit_vram(&n, Bytes(CTX + FIXED + EXPERT + HEADROOM - 1)),
        VramFit::Short
    );
}

#[test]
fn a_spec_without_a_minimum_defaults_to_one_segment_or_five_percent() {
    // 16 segments of 32 MiB: 5% of 512 MiB is 25.6 MiB, so one segment.
    let a = expert_cache(16, 32, 0);
    assert_eq!(min_expert(&a), (Bytes(32 * MIB), Bytes(32 * MIB)));
    // 100 segments of 2 MiB: 5% of 200 MiB is 10 MiB, five segments.
    let a = expert_cache(100, 2, 0);
    assert_eq!(min_expert(&a), (Bytes(10 * MIB), Bytes(2 * MIB)));
}

#[test]
fn two_expert_cache_arenas_are_trimmed_from_the_last_one_first() {
    let mut plan = plan();
    plan.arenas.push(expert_cache(8, 32, 2)); // index 7, 256 MiB
    let free = CTX + FIXED + EXPERT + 256 * MIB + HEADROOM - 100 * MIB;
    let (p, ctx, reserve) = device(free);
    let built = build_vram(&p, &ctx, &plan, VRAM, reserve).unwrap();
    // 668 MiB of room: 20 segments. The second arena loses 4 of its 8.
    assert_eq!(built.get(0).unwrap().bytes, Bytes(EXPERT));
    assert_eq!(built.get(7).unwrap().bytes, Bytes(4 * 32 * MIB));
    assert_eq!(built.report.expert_mapped, Bytes(20 * 32 * MIB));
    built.release(&p, &ctx);
}

#[test]
fn the_report_round_trips_through_serde() {
    let (p, ctx, reserve) = device(8 << 30);
    let built = build_vram(&p, &ctx, &plan(), VRAM, reserve).unwrap();
    let json = serde_json::to_string(&built.report).unwrap();
    let back: VramReport = serde_json::from_str(&json).unwrap();
    assert_eq!(back, built.report);
    assert_eq!(back.context_reserve.bytes(), Bytes(CTX));
    built.release(&p, &ctx);
}

proptest! {
    /// Whatever the figures, a reduced cache fits the room, keeps its minimum, is a whole number
    /// of segments up to the plan, and `All` and `Short` hold exactly when §8.1 says.
    #[test]
    fn fit_vram_follows_rfc_0020_8_1(
        context in 0u64..1 << 34,
        fixed in 0u64..1 << 36,
        segments in 1u64..256,
        segment in 1u64..1 << 28,
        min_segments in 0u64..256,
        headroom in 0u64..1 << 31,
        free in 0u64..1 << 38,
    ) {
        let min_segments = min_segments.min(segments);
        let n = VramNeeds {
            context: Bytes(context),
            fixed: Bytes(fixed),
            expert: Bytes(segments * segment),
            min_expert: Bytes(min_segments * segment),
            segment: Bytes(segment),
            headroom: Bytes(headroom),
        };
        let base = context + fixed + headroom;
        match fit_vram(&n, Bytes(free)) {
            VramFit::All => prop_assert!(base + n.expert.0 <= free),
            VramFit::Short => prop_assert!(base + n.min_expert.0 > free),
            VramFit::Reduced { expert } => {
                prop_assert!(base + n.expert.0 > free);
                prop_assert!(base + expert.0 <= free);
                prop_assert!(expert >= n.min_expert);
                prop_assert!(expert < n.expert);
                prop_assert_eq!(expert.0 % segment, 0);
                // No further segment would have fitted.
                prop_assert!(base + expert.0 + segment > free);
            }
        }
    }
}

// ---- HostPageable ----------------------------------------------------------------------------

/// An adapter that hands out fake addresses and records every call.
#[derive(Default)]
struct FakeOs {
    calls: Mutex<Vec<String>>,
    next: Mutex<u64>,
    /// Refuse the reservation with this index (0-based).
    refuse_reserve: Option<usize>,
    /// Refuse every commit.
    refuse_commit: bool,
    /// Bind fails for every range.
    no_numa: bool,
}

impl FakeOs {
    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl OsMemory for FakeOs {
    fn snapshot(&self) -> MemSnapshot {
        unimplemented!("not read by the build")
    }
    fn limits(&self) -> PinLimits {
        unimplemented!("not read by the build")
    }
    fn reserve(
        &self,
        bytes: Bytes,
        numa: Option<u16>,
        huge: HugePages,
    ) -> Result<HostRange, OsMemError> {
        let mut calls = self.calls.lock().unwrap();
        let n = calls.iter().filter(|c| c.starts_with("reserve")).count();
        calls.push(format!("reserve {} {numa:?} {huge:?}", bytes.0));
        if self.refuse_reserve == Some(n) {
            return Err(OsMemError::HostReserve {
                op: "mmap",
                errno: 12,
            });
        }
        let mut next = self.next.lock().unwrap();
        *next += 1;
        Ok(HostRange {
            base: std::ptr::null_mut(),
            len: bytes,
            numa: if self.no_numa { None } else { numa },
            os_handle: *next,
        })
    }
    fn commit(&self, r: &HostRange, sub: Range<u64>) -> Result<(), OsMemError> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("commit {} {sub:?}", r.os_handle));
        if self.refuse_commit {
            Err(OsMemError::HostReserve {
                op: "madvise",
                errno: 12,
            })
        } else {
            Ok(())
        }
    }
    fn lock(&self, _: &HostRange, _: Range<u64>) -> Result<(), OsMemError> {
        panic!("the build never locks: pinning does")
    }
    fn unlock(&self, _: &HostRange, _: Range<u64>) -> Result<(), OsMemError> {
        panic!("the build never unlocks")
    }
    fn release(&self, r: HostRange) {
        self.calls
            .lock()
            .unwrap()
            .push(format!("release {}", r.os_handle));
    }
}

/// Two `HostPageable` arenas, one elastic and NUMA-bound, beside the VRAM ones.
fn host_plan() -> ArenaPlan {
    let mut plan = plan();
    let mut cache = arena(HOST, Purpose::ExpertCache, 32 * MIB, SlotGeometry::Bump);
    cache.max_bytes = Bytes(96 * MIB);
    cache.numa = Some(1);
    plan.arenas.push(cache);
    plan
}

#[test]
fn host_pageable_arenas_are_reserved_to_their_maximum_and_committed_to_their_size() {
    let os = FakeOs::default();
    let built = build_host_pageable(&os, &host_plan(), HOST, HugePages::Auto).unwrap();
    assert_eq!(
        os.calls(),
        [
            format!("reserve {} None Auto", 64 * MIB),
            format!("commit 1 0..{}", 64 * MIB),
            format!("reserve {} Some(1) Auto", 96 * MIB),
            format!("commit 2 0..{}", 32 * MIB),
        ]
    );
    let summary: Vec<_> = built
        .arenas
        .iter()
        .map(|a| (a.index, a.purpose, a.range.len, a.committed, a.unbound))
        .collect();
    assert_eq!(
        summary,
        [
            (3, Purpose::Weights, Bytes(64 * MIB), Bytes(64 * MIB), false),
            (
                7,
                Purpose::ExpertCache,
                Bytes(96 * MIB),
                Bytes(32 * MIB),
                false
            ),
        ]
    );
    built.release(&os);
    assert_eq!(&os.calls()[4..], ["release 2", "release 1"]);
}

#[test]
fn a_failed_numa_bind_leaves_the_arena_unbound_and_says_so() {
    let os = FakeOs {
        no_numa: true,
        ..FakeOs::default()
    };
    let built = build_host_pageable(&os, &host_plan(), HOST, HugePages::Off).unwrap();
    let unbound: Vec<bool> = built.arenas.iter().map(|a| a.unbound).collect();
    // The weights arena asked for no node, so it is not "unbound".
    assert_eq!(unbound, [false, true]);
    built.release(&os);
}

#[test]
fn a_refused_reservation_releases_what_was_reserved() {
    let os = FakeOs {
        refuse_reserve: Some(1),
        ..FakeOs::default()
    };
    let e = build_host_pageable(&os, &host_plan(), HOST, HugePages::Auto).unwrap_err();
    assert_eq!(e.tier_code(), Some(4));
    assert_eq!(os.calls().last().map(String::as_str), Some("release 1"));
}

#[test]
fn a_refused_commit_releases_that_range_and_the_earlier_ones() {
    let os = FakeOs {
        refuse_commit: true,
        ..FakeOs::default()
    };
    let e = build_host_pageable(&os, &host_plan(), HOST, HugePages::Auto).unwrap_err();
    assert_eq!(e.class(), ErrorClass::Capacity);
    let calls = os.calls();
    assert_eq!(calls.last().map(String::as_str), Some("release 1"));
    assert_eq!(calls.iter().filter(|c| c.starts_with("reserve")).count(), 1);
}

#[test]
fn a_host_tier_with_no_arenas_builds_nothing() {
    let os = FakeOs::default();
    let other = TierId { ordinal: 9, ..HOST };
    let built = build_host_pageable(&os, &host_plan(), other, HugePages::Auto).unwrap();
    assert!(built.arenas.is_empty());
    assert!(os.calls().is_empty());
}

/// The Linux adapter: a real anonymous arena, reserved and committed, then released. The commit
/// is checked from `/proc/self/smaps`: the mapping is anonymous, and its resident size covers the
/// committed bytes. The adapter's own tests check `release`.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[test]
fn host_pageable_arenas_on_linux_are_anonymous_and_committed() {
    use crate::os::linux::LinuxMemory;

    let os = LinuxMemory::new(Bytes(0));
    let mut plan = ArenaPlan::default();
    let mut a = arena(HOST, Purpose::Weights, 4 * MIB, SlotGeometry::Bump);
    a.max_bytes = Bytes(16 * MIB);
    plan.arenas.push(a);
    let built = build_host_pageable(&os, &plan, HOST, HugePages::Off).unwrap();
    let r = &built.arenas[0].range;
    assert_eq!(r.len, Bytes(16 * MIB));
    let base = r.base as u64;
    // The smaps fields of the mapping that holds `base`. The kernel may merge the range with a
    // neighbouring mapping of the same flags, so the mapping is found by address, not by start.
    let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
    let mut ours = false;
    let mut fields: Vec<(String, u64)> = Vec::new();
    for line in smaps.lines() {
        if let Some((range, _)) = line.split_once(' ')
            && let Some((s, e)) = range.split_once('-')
            && let (Ok(s), Ok(e)) = (u64::from_str_radix(s, 16), u64::from_str_radix(e, 16))
        {
            ours = s <= base && base < e;
            continue;
        }
        if ours
            && let Some((k, v)) = line.split_once(':')
            && let Ok(kb) = v.trim().trim_end_matches("kB").trim().parse::<u64>()
        {
            fields.push((k.to_owned(), kb));
        }
    }
    let field = |name: &str| -> u64 {
        fields
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("no {name} for the mapping at {base:#x}"))
    };
    assert!(
        field("Rss") * 1024 >= 4 * MIB,
        "the commit did not fault the pages in"
    );
    assert_eq!(
        field("Anonymous"),
        field("Rss"),
        "the arena is not anonymous"
    );
    built.release(&os);
}

/// The Windows adapter: a real reservation, committed and released.
#[cfg(all(windows, any(target_arch = "x86_64", target_arch = "aarch64")))]
#[test]
fn host_pageable_arenas_on_windows_are_reserved_and_committed() {
    use crate::os::windows::WindowsMemory;

    let os = WindowsMemory::new(Bytes(0));
    let mut plan = ArenaPlan::default();
    let mut a = arena(HOST, Purpose::Weights, 4 * MIB, SlotGeometry::Bump);
    a.max_bytes = Bytes(16 * MIB);
    plan.arenas.push(a);
    let built = build_host_pageable(&os, &plan, HOST, HugePages::Off).unwrap();
    assert_eq!(built.arenas[0].range.len, Bytes(16 * MIB));
    assert_eq!(built.arenas[0].committed, Bytes(4 * MIB));
    built.release(&os);
}
