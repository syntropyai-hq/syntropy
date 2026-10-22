# RFC-0020: Tier manager and residency

- Status: Draft
- Area: tiers (tiered memory engine inside a node)
- Depends on: RFC-0001 (§2.3, §3.3, §5.2, §5.4, §5.5, §8, §9, §10.3, §11.3), RFC-0010, RFC-0011, RFC-0012, RFC-0021, RFC-0022, RFC-0023, RFC-0040, RFC-0054; ADR-0005, ADR-0006, ADR-0007, ADR-0012, ADR-0013, ADR-0014, ADR-0015, ADR-0020, ADR-0200, ADR-0201, ADR-0202
- Date: 2026-10-19
- Revised: 2026-10-20: streaming at decode where measured faster (§10); open question 4 answered. 2026-10-21: `ExtentRef` and `StepLayer` are defined in `syntropy-core` (§3). 2026-10-22: `BufferView` and `ViewKind` are defined in `syntropy-core` (§3)

## Summary

Each runner has one tier manager (`syntropy-tier`). It owns every byte the runner keeps in `Vram`, `Unified`, `HostPinned`, `HostPageable` and `Nvme`: weights, expert-cache slots, KV pages, staging rings, network pools and scratch (ADR-0007). This RFC specifies:

- the arenas the tier manager builds per tier kind and per purpose, and how each is allocated on CUDA, HIP, Metal, Vulkan and the CPU, on Linux, Windows, WSL2 and macOS;
- the per-OS formulas for how much host memory may be pinned, and the progressive pinning loop that gets there without freezing the machine;
- the residency map, tensor handles and leases, with their state machines and single-writer rules;
- the VRAM split (context, scratch, dense hot set, KV hot window, expert cache, headroom) and the elastic expert-cache segments that let VRAM shrink and grow at fixed addresses;
- unified-memory devices (Apple silicon, Strix Halo, GB10-class) as one `Unified` tier with no PCIe term;
- the runtime choice between execute-in-place and PCIe streaming per pass (the n* rule), refined from live timings;
- how the runner gives RAM and VRAM back when the user needs it, and what that costs in token time.

Three decisions carry most of the weight. Host memory that only the CPU reads is anonymous memory that need not be DMA-registered, so the scarce pinning budget goes to bytes a GPU or NIC actually DMAs (ADR-0200). VRAM arenas are reserved virtual ranges backed by separately mapped segments, so the expert cache can shrink and grow without moving a slot (ADR-0201). Memory is yielded along a fixed ladder whose token-time cost is predicted before each step is taken (ADR-0202).

## Motivation and goals / non-goals

The solver (RFC-0011) decides which tier each tensor group lives in. Something has to make that true on a real machine, keep it true while the user opens a browser or a game, and say precisely what it cost when it cannot. The research shows how this goes wrong when it is left to the OS or done naively (`plan/research/tiered-memory-inference.md` §10.1, §10.2, §18):

- File-backed weights read at ≈19 GB/s against 42.8 GB/s from an anonymous arena on Windows, and the page cache evicted expert pages first.
- One giant `cudaHostAlloc` got a Linux process OOM-killed with high MemAvailable; locking one huge range fails on Windows; WSL2 caps `cudaHostAlloc` at about 1 GiB; an unwired arena on macOS caused compressor panics.
- Loading and locking 35–55 GB froze a desktop for 1–3 minutes.
- Long context silently ate expert capacity: decode fell from ≈95 to ≈56 tok/s at 262K until KV was moved out of VRAM.
- Pageable host memory reaches ≈6 GB/s H2D against ≈24 GB/s pinned on PCIe 4.0 x16 (`hardware-and-performance-model.md` §2.3).

**Goals**

1. **The plan is what happens.** Every `TierAssignment`, `KvBudget`, `ExpertCacheBudget` and scratch entry becomes an arena of exactly the planned size, on the planned tier and NUMA domain, or the runner fails `Loading` with a `Capacity` error that corrects the profile (RFC-0001 §10.1). Predicted bytes per tier are within 5% of measured (SC-11).
2. **Never OOM, never freeze.** Zero OOM kills across the test matrix; pinning never takes free memory below the reserve (SC-11); no pinning step stalls the desktop.
3. **Fixed addresses.** Slots keep their device addresses for the life of a plan epoch so captured graphs stay valid (RFC-0001 §5.5); VRAM can still shrink.
4. **No allocation, lock or synchronising call on the token path** after `Warming` (RFC-0001 invariant 9).
5. **Yield on purpose, with a price.** Memory goes back to the user in a fixed order, and each rung's predicted Δt_tok is reported before and measured after.
6. **One behaviour across OSes**, with the per-OS differences confined to pinning adapters and pressure sources.

**Non-goals**

- Choosing home tiers, budgets or compute sites. That is the solver's job (RFC-0011). The tier manager executes the plan and reports when it cannot.
- Expert admission and eviction policy (RFC-0023), KV paging policy (RFC-0022), and the I/O engines themselves (RFC-0021). This RFC provides the arenas, handles and residency records they use.
- Changing OS limits on the user's behalf. Syntropy never raises `iogpu.wired_limit_mb`, `RLIMIT_MEMLOCK` or Windows privileges silently. The only ways a limit changes are the explicit `syntropy tune` opt-in (§9.1) and limits packaging owns in its own service unit, such as `LimitMEMLOCK` (RFC-0052).
- Byte-for-byte residency determinism across runs. The `deterministic` option (ADR-0019 §8) freezes residency per request; that is RFC-0012's.

## Design

### 1. Ownership and structure

`syntropy-tier` (layer 2, RFC-0001 §4.1) has these modules. `unsafe` is confined to `arena` and `os` (RFC-0001 §4.2 rule 8).

| Module | Contents | Owning RFC |
|---|---|---|
| `os` | `OsMemory` adapters: Linux, Windows, WSL2, macOS, Android/iOS, and a fake for `testkit` | 0020 |
| `budget` | pin ceilings, the VRAM split, the ledger check | 0020 |
| `arena` | arena construction per tier kind, segment mapping, slot allocators | 0020 |
| `pin` | progressive pinning | 0020 |
| `residency` | `ResidencyMap`, `TensorHandle`, leases, step fences | 0020 |
| `pressure` | pressure levels, the yield ladder, reacquisition | 0020 |
| `eip` | the runtime n* decision and its online fit | 0020 |
| `kv` | `KvStore`, page pools, prefix index, sessions | 0022 |
| `experts` | expert cache, slot table, prediction, miss balancer | 0023 |

**Threads and single writers** (RFC-0001 §2.3 invariant 4, §8.2):

- The tier manager is split into one **`TierShard` per device**, owned by that device's dispatcher thread. A GPU's shard holds its `Vram` arenas; a CPU device's shard holds the host arenas of its NUMA domain; an integrated GPU's shard holds the `Unified` arenas. Only the owning dispatcher allocates, frees or changes residency in its shard.
- A shared, read-mostly **`TierDirectory`** (one per runner, `Arc`) holds every arena's base address, slot geometry and the per-slot generation and fence words, so any thread can resolve a handle without a lock (§7).
- The **housekeeping thread** (RFC-0001 §8.2) drives progressive pinning, pressure response, digests and cache-policy decisions. It never writes a shard directly: it posts `TierCommand`s to each dispatcher over an SPSC ring, and the dispatcher applies them between steps.
- The **I/O engine** writes bytes into arena memory but never residency: completions go back to the owning dispatcher, which flips state (RFC-0001 §8.3).

Before `Warming` ends, shards may allocate. After it, the only allocations are slot reuse inside existing arenas, and the only mapping changes are elastic segment changes applied between steps (§8.4).

### 2. Tiers, arenas and purposes

A **tier** at runtime is one `TierInfo` from the resource graph (RFC-0001 §5.2) that this runner was granted budget on. Host RAM appears as one `HostPinned` and one `HostPageable` tier **per NUMA domain**, each with the affinity of that domain's CPU device and of the GPUs attached to it. RFC-0010 enumerates them that way; RFC-0011 places against them.

An **arena** is a contiguous reservation inside one tier for one **purpose**. Each purpose has its own budget so one cannot starve another (RFC-0001 §9.1):

| Purpose | Tiers | Size from | Allocator | Registered with |
|---|---|---|---|---|
| `Weights` | all except `Nvme` | Σ bytes of groups homed there (`TierAssignment`) | extents, laid out once at `Loading`, one slot per group | GPU provider (`HostPinned`), `MTLResidencySet` (`Unified` on Apple) |
| `ExpertCache` | `Vram`, `Unified`, `HostPinned`, `HostPageable` | `ExpertCacheBudget::bytes`, `per_layer`; over an `Nvme` home the RAM cache is split into a `DmaPinned` part (`HostPinned`, the copyable share and VRAM-admission sources) and an in-place part (`HostPageable`, `Locked` when allowed) | per-layer slot classes, striped across segments (§8.3) | GPU provider for the `HostPinned` part; io_uring fixed buffers for it where `io_fixed` allows; the `HostPageable` part needs none (NVMe direct reads need no registration) |
| `ExpertStaging` | `Vram`, `HostPinned` | RFC-0023 (copy, prefetch and demand staging slots) | slot classes; **fixed per epoch, not elastic** (it is small, and the copied share is clamped to its mapped slots, RFC-0023 §6) | as the `HostPinned` part of `ExpertCache` |
| `Kv` | `Vram`, `Unified`, `HostPinned` | `KvBudget::bytes` | fixed-size pages (RFC-0022, ADR-0205) | GPU provider; io_uring fixed buffers for spill pages |
| `Scratch` | `Vram`, `Unified`, `HostPinned` | `plan.scratch` + provider workspaces (`query`) | bump, frozen at `Warming` | — |
| `Graphs` | `Vram` | provider-reported capture size | provider-owned inside the reservation | — |
| `IoStaging` | `HostPinned` | RFC-0021 staging ring sizing | ring of fixed slots | GPU provider, io_uring/IoRing fixed buffers |
| `NetPool` | `HostPinned` | RFC-0031 pool sizing | ring of fixed slots | RDMA memory regions, GPU provider |
| `RowCache` | `HostPageable` by default (`HostPinned` only when the head stage's embedding rows are streamed to a GPU) | RFC-0023 for `RowLookup` groups homed on `Nvme` | slot classes | io_uring fixed buffers where `io_fixed` allows |
| `History` | `HostPageable` (locked when allowed; never DMA-registered) | `recovery.history_ram` (RFC-0033 §4.1), charged to `HostPageable` under the joint host row (§5) | 2 MiB segments handed to the activation-history store; the oldest spill through the I/O engine to RFC-0033's history file | none |

`Nvme` has no arena. A group homed on `Nvme` has an **extent map**: (pack file, offset, length) per group from the pack sidecar (RFC-0041), opened `O_DIRECT` or equivalent by RFC-0021.

**Pin classes.** Every host arena has one of three pin classes. They are the reason `HostPinned` and `HostPageable` are separate tiers:

| Class | Mechanism | Who reads it | Tier kind |
|---|---|---|---|
| `DmaPinned` | locked and registered with each provider that DMAs from it (`register_host`), and with RDMA or io_uring where needed | GPU copy engines, NICs, the I/O engine, CPU | `HostPinned` |
| `Locked` | anonymous, locked against paging (`mlock`, a hard working-set minimum, or wiring) but not DMA-registered | CPU only | `HostPageable`, locked opportunistically |
| `Pageable` | anonymous, never file-backed, not locked | CPU only | `HostPageable` |

A CPU kernel reads anonymous memory at the same rate whether it is locked or not; the 2.2× gap in the research is between anonymous and file-backed memory, not between locked and unlocked (`tiered-memory-inference.md` §10.1). Locking only stops the OS paging the bytes out, and DMA registration only matters for bytes a device copies. So the scarcest budget, DMA registration, goes only to bytes that are DMA'd: copied-miss sources, prefill streaming sources, staging rings, cache slots that receive NVMe reads, and network pools (ADR-0200). Groups that only the CPU computes are homed in `HostPageable` and locked when the lock budget allows; a `HostPageable` home is never a DMA source (RFC-0001 §9.4). The solver sees the budgets as `TierInfo::pin_limit` on the `HostPinned` tier and `usable` on both; RFC-0010 reports them from §5. The probe measures CPU `W_eff` over both tiers, so if anonymous `HostPageable` ever reads slower than `HostPinned` on some platform, the solver sees it.

### 3. Types

These extend RFC-0001 §5.5. Nothing there is renamed; fields added to RFC-0001 types carry serde defaults.

```rust
// ---- syntropy-tier::arena ----------------------------------------------------------

/// Why an arena exists. One budget per (tier, purpose).
pub enum Purpose { Weights, ExpertCache, ExpertStaging, Kv, Scratch, Graphs, IoStaging, NetPool, RowCache, History }

pub enum PinClass { DmaPinned, Locked, Pageable }

bitflags! {
    /// What the arena must be registered with before Loading ends.
    pub struct Registrations: u8 {
        const GPU_DMA    = 1 << 0;  // KernelProvider::register_host on every GPU in `dma_devices`
        const GPU_MAPPED = 1 << 1;  // also mapped into the GPU address space (doorbells, partial outputs)
        const IO_FIXED   = 1 << 2;  // io_uring / IoRing registered buffers, ≤ 1 GiB per buffer (ADR-0013)
        const RDMA_MR    = 1 << 3;  // Transport::register (RFC-0031)
        const RESIDENCY  = 1 << 4;  // MTLResidencySet / useHeap on Apple
    }
}

pub enum SlotGeometry {
    /// One slot per tensor group, laid out in plan order, each start aligned to `align`.
    Extents { align: Bytes },
    /// Fixed-size slots per class; class i holds `count` slots of `size` bytes.
    /// ExpertCache uses one class per MoE layer (RFC-0023).
    Classes(SmallVec<[SlotClass; 64]>),
    /// Fixed-size pages (KV, RFC-0022).
    Pages { page: Bytes, count: u32 },
    /// A ring of equal slots (staging, network pools).
    Ring { slot: Bytes, count: u32 },
    /// Bump allocation, frozen at Warming (scratch).
    Bump,
}
pub struct SlotClass { pub size: Bytes, pub count: u32, pub layer: Option<LayerIx> }

pub struct ArenaSpec {
    pub tier: TierId,
    pub purpose: Purpose,
    pub bytes: Bytes,                     // initial size
    pub max_bytes: Bytes,                 // reservation size; > bytes only for elastic arenas (§8.3)
    pub geometry: SlotGeometry,
    pub pin: PinClass,
    pub numa: Option<u16>,
    pub registrations: Registrations,
    pub dma_devices: SmallVec<[DeviceId; 4]>,
    /// Order in which this arena is pinned (§6). Lower first.
    pub pin_priority: u16,
    /// Elastic arenas are built from segments that can be mapped and unmapped (§8.3).
    pub elastic: Option<SegmentSpec>,
}
pub struct SegmentSpec { pub segment: Bytes, pub min_segments: u32 }

/// The tier manager's whole layout for one runner, derived from the plan before any byte
/// is allocated. Pure function of (plan, stages, census, profile, config); unit-testable.
pub struct ArenaPlan {
    pub arenas: Vec<ArenaSpec>,
    pub nvme_extents: Vec<(TensorGroupId, ExtentRef)>,
    pub per_tier: Vec<(TierId, TierBudgetLine)>,   // for the ledger check and SC-11
}
pub struct TierBudgetLine { pub reserved: Bytes, pub by_purpose: SmallVec<[(Purpose, Bytes); 9]>, pub headroom: Bytes }

// ---- syntropy-tier::os -------------------------------------------------------------

/// Everything OS-specific about host memory. One implementation per OS, plus a fake.
pub trait OsMemory: Send + Sync {
    fn snapshot(&self) -> MemSnapshot;
    fn limits(&self) -> PinLimits;
    /// Anonymous, never file-backed. `numa` binds the pages before first touch.
    fn reserve(&self, bytes: Bytes, numa: Option<u16>, huge: HugePages) -> Result<HostRange, Error>;
    fn commit(&self, r: &HostRange, sub: Range<u64>) -> Result<(), Error>;   // touch pages (§6)
    fn lock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), Error>;     // mlock / VirtualLock / wire
    fn unlock(&self, r: &HostRange, sub: Range<u64>) -> Result<(), Error>;
    fn release(&self, r: HostRange);
    fn pressure(&self) -> Box<dyn PressureSource>;
}

pub struct MemSnapshot {
    pub total: Bytes,
    pub available: Bytes,                     // MemAvailable / ullAvailPhys / free + inactive + purgeable
    pub numa_free: SmallVec<[(u16, Bytes); 4]>,      // reported only; never gated on (§6)
    /// Per-node estimate of what the node can supply after reclaiming clean cache, with
    /// numa_reserve already subtracted once (§5): MemFree + Active(file) + Inactive(file)
    /// + SReclaimable − numa_reserve, from /sys/devices/system/node/node<n>/meminfo.
    pub numa_available: SmallVec<[(u16, Bytes); 4]>,
    pub cgroup: Option<CgroupMem>,            // limit (memory.max / memory.high), usage, events.high/max deltas
    /// Windows: commit limit − commit charge. Other OSes: None (no commit accounting to respect).
    pub commit_headroom: Option<Bytes>,
    pub own_locked: Bytes,
    pub own_registered: Bytes,
    pub psi: Option<Psi>,                     // some/full avg10 from /proc/pressure/memory (Linux), when readable
    /// Linux: direct-reclaim stalls since the previous snapshot, the sum of the allocstall_*
    /// counters in /proc/vmstat. Used as the pressure signal where PSI is not available (§6).
    pub direct_reclaim_stalls: Option<u64>,
    pub os_level: Option<PressureLevel>,      // macOS dispatch source, Windows notification
    pub at: Instant,
}

pub struct PinLimits {
    pub memlock: Option<Bytes>,               // RLIMIT_MEMLOCK soft limit; None = unlimited
    pub cap_ipc_lock: bool,
    pub io_fixed_max: Option<Bytes>,          // what io_uring/IoRing registration may still take
    pub ws_hard_min_ok: bool,                 // Windows: hard working-set minimum allowed
    pub lockable_fraction: Option<f32>,       // Windows: measured fraction of RAM lockable for GPU DMA
    pub helper_gpu_cap: Option<Bytes>,        // Windows WDDM: 8 GiB host registration for helper GPUs
    pub host_alloc_cap: Option<Bytes>,        // WSL2: ≈1 GiB cudaHostAlloc cap; forces register path
    pub user_wire_limit: Option<Bytes>,       // macOS: vm.user_wire_limit / vm.global_user_wire_limit
    pub gpu_wired_cap: Option<Bytes>,         // macOS: recommendedMaxWorkingSetSize
}

// ---- syntropy-tier::residency ------------------------------------------------------

/// RFC-0001 §5.5 Residency, extended.
pub struct Residency {
    pub home: TierId,
    pub home_handle: TensorHandle,
    pub copies: SmallVec<[CachedCopy; 2]>,
    #[serde(default)] pub home_state: HomeState,   // added by this RFC
}
pub enum HomeState {
    Absent,                          // arena allocated, bytes not yet read
    Loading { ticket: IoTicket },
    /// Resident and at its planned pin class.
    Resident,
    /// Resident but below its planned pin class (pinning not done, or yielded).
    /// Costed as HostPageable; never a DMA source while in this state (ADR-0007 item 4).
    Underpinned { have: PinClass },
    /// Home is Nvme: an extent map, read on demand or by prefetch.
    Extent,
}

/// Completed-step counter per device. A slot freed at step s may be overwritten once the
/// device's completed-step counter reaches s. The counter is kept by the host: the dispatcher increments it
/// each time it consumes the step-done flag `O[p]` (RFC-0012 §3.2), in step order.
pub struct StepFence { pub device: u16, pub step: u64 }

pub struct ResidencyLease<'a> {
    pub handle: TensorHandle,
    pub fence: StepFence,          // eviction of the slot waits for this fence
    _dir: PhantomData<&'a TierDirectory>,
}

/// What the hot path and the I/O engine actually use.
pub struct BufferView { pub kind: ViewKind, pub addr: u64, pub len: Bytes }
pub enum ViewKind { Host, Device(DeviceId), Unified }

// ---- syntropy-tier::manager --------------------------------------------------------

pub struct TierManager { /* façade over TierShard per device and the TierDirectory */ }

impl TierManager {
    pub fn plan_arenas(plan: &PlacementPlan, stages: &[StageIx], census: &TensorCensus,
                       caps: &CapabilityProfile, layouts: &AcceptedLayouts, cfg: &TierConfig)
                       -> Result<ArenaPlan, Error>;   // `layouts` added 2026-10-22 (note below)
    /// Fetching: reserve host ranges and start progressive pinning (§6).
    pub fn begin_host(&mut self, ap: &ArenaPlan, os: Arc<dyn OsMemory>, grant: &LedgerGrant) -> Result<(), Error>;
    /// Loading, on each dispatcher thread: device arenas, registrations.
    pub fn build_device(&mut self, dev: &DeviceCtx, p: &dyn KernelProvider, ap: &ArenaPlan) -> Result<(), Error>;
    /// RFC-0001 §5.5.
    pub fn lease(&self, h: TensorHandle) -> Result<ResidencyLease<'_>, Error>;
    pub fn view(&self, lease: &ResidencyLease<'_>) -> BufferView;
    /// Slot allocation inside an existing arena; the only allocation allowed after Warming.
    pub fn take_slot(&mut self, arena: ArenaRef, class: u16) -> Option<TensorHandle>;
    pub fn release_slot(&mut self, h: TensorHandle, fence: StepFence);
    pub fn apply(&mut self, cmd: TierCommand) -> Result<TierOutcome, Error>;   // §12
    pub fn summary(&self) -> ResidencySummary;                                // §13
}

pub enum TierCommand {
    PinStep { arena: ArenaRef, range: Range<u64> },
    Unpin { arena: ArenaRef, range: Range<u64> },
    ShrinkElastic { arena: ArenaRef, segments: u32 },
    GrowElastic { arena: ArenaRef, segments: u32 },
    Yield { level: PressureLevel, reclaim: Bytes },
    ApplyDelta { delta: Arc<PlanDelta> },
}
```

**Where `BufferView` and `ViewKind` live** (revised 2026-10-22). Both are defined in `syntropy-core`, with serde derives, and `tier` re-exports them unchanged. `kernels` names `BufferView` in `OpArgs::weights` (RFC-0040 §3.5), and `kernels` is in layer 1, below `tier` (RFC-0001 §4.1, §4.2 rule 2), so the type must sit in `core`, as `ExtentRef` does.

*Note 2026-10-22 (OQ-IO-27, OQ-IO-28):* `plan_arenas` takes the accepted layouts as an argument, `AcceptedLayouts`: a map from (op key, device) to the `LayoutVersion`s each op accepts, which the runner builds from `KernelProvider::query` before the call. `plan_arenas` stays pure and makes §7.3's check itself. `LedgerGrant` is plain data in `core` (RFC-0001 §3.5), and the ledger check is a separate pure check of an `ArenaPlan` against it in `Fetching` (§12).

Supporting types used across RFC-0020 to RFC-0023:

```rust
pub struct ArenaRef { pub shard: u16, pub arena: u8 }                 // shard = device ordinal
pub struct ExtentRef { pub file: u16, pub offset: u64, pub len: u64 }  // multiples of the volume's DioAlign (RFC-0021 §6)
pub struct HostRange { pub base: *mut u8, pub len: Bytes, pub numa: Option<u16>, pub os_handle: u64 }  // !Send outside `os`
pub struct PinProgress { pub done: Bytes, pub target: Bytes, pub state: PinState, pub eta_ms: Option<u64> }
pub enum PinState { Running, Done, GaveUp { underpinned: Bytes } }
pub enum PressureLevel { Normal, Moderate, High, Critical }           // §11.2
/// A point in a device's execution: the step number and the layer within it. Consumers publish
/// their progress as one AtomicU64 per device, (step << 16) | layer; a StepLayer has passed when
/// the published word is greater than or equal to its encoding.
pub struct StepLayer { pub step: u64, pub layer: u16 }
```

**Where `ExtentRef` and `StepLayer` live** (revised 2026-10-21). Both are defined in `syntropy-core`, with serde derives, and `tier` and `io` use them from there. RFC-0021 §2's `ReadRequest` carries them, so they must sit below `io`, which is a layer below `tier` (RFC-0001 §4.1, §4.2 rule 2). `ArenaPlan` round-trips `ExtentRef` through serde, and `core` already depends on serde (§4.2 rule 1), so `io` needs no serde of its own. `ExtentRef` is `{ file: FileIx, offset: u64, len: Bytes }`, because interfaces take no bare integers for sizes (RFC-0001 §5.1); `FileIx`, the index the I/O engine's `open` returns, is in `syntropy-core` too.

### 4. Building arenas per tier kind and platform

All device memory comes from `KernelProvider::alloc` (RFC-0001 §5.7, §9.1). Host memory comes from `OsMemory::reserve` and is then registered with providers. Providers never allocate model memory on their own initiative.

#### 4.1 `Vram`

| Backend | Reservation | Physical backing | Elastic support | Fallback |
|---|---|---|---|---|
| CUDA | `cuMemAddressReserve`, one range per purpose, `max_bytes` long | `cuMemCreate` (pinned device allocation) per segment, rounded up to `cuMemGetAllocationGranularity(RECOMMENDED)`; `cuMemMap` and `cuMemSetAccess` for the owning device, plus P2P-verified devices in an `IntraNodeTp` group | yes, when the device attribute `VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` is set (ADR-0014) | one `cudaMalloc` per purpose; shrinking needs a runner reload |
| HIP | `hipMemAddressReserve` | `hipMemCreate`, `hipMemMap`, `hipMemSetAccess` | yes, when `hipDeviceAttributeVirtualMemoryManagementSupported` is set | one `hipMalloc` per purpose |
| Vulkan | none: Vulkan has no address reservation | one `VkDeviceMemory` per segment from the device-local heap, buffers bound per segment and addressed with buffer device addresses | yes: a segment is freed whole after its buffers are destroyed | — |
| ggml provider | — | the native provider's allocation, imported into a ggml buffer (RFC-0040 §buffer interop) | as the native provider | — |
| ggml-only device (the M2 Vulkan preview, RFC-0040 §7.5) | none | every buffer allocated through ggml's Vulkan buffer type, one allocation per purpose, each recorded in the device's arena directory with its bytes and lifetime | no: fixed per epoch; no `map_segment` | — |

Segment mapping needs two provider methods that RFC-0001 §5.7 does not list. RFC-0040 adds them with default bodies that return `Unsupported`:

```rust
fn map_segment(&self, ctx: &DeviceCtx, reservation: &DeviceBuffer, offset: Bytes, bytes: Bytes)
    -> Result<Segment, Error> { Err(unsupported()) }
fn unmap_segment(&self, ctx: &DeviceCtx, seg: Segment) -> Result<(), Error> { Err(unsupported()) }
```

**ggml-only devices.** A device whose only provider is ggml (the M2 Vulkan preview) owns no memory of its own: the tier manager allocates every purpose's buffer through ggml's buffer type at `Loading`, records it like any arena (handles, leases and the residency map are unchanged), and charges it against a fixed per-device budget, the plan's VRAM split for that device less `resource.vram_headroom`. Nothing on such a device is elastic: its expert cache keeps its `Loading` size for the epoch, so the VRAM rung of the yield ladder (§11.4) frees nothing there, and a capacity change on it is a re-plan, not a shrink. Allocations beyond the budget are refused at `Loading` with `Capacity`.

Only `ExpertCache` arenas are elastic. `ExpertStaging` is small and fixed per epoch, so the copied share can never exceed its mapped copy slots (RFC-0023 §6). `Weights`, `Kv`, `Scratch` and `Graphs` are mapped in full at `Loading` and never change within an epoch; a change to their sizes is a new `PlanDelta` that the runner applies by draining (RFC-0001 §7.3), except the KV hot window, which RFC-0022 moves within its fixed arena.

#### 4.2 `HostPinned` and `HostPageable`

| OS | Reserve | NUMA binding | Huge pages | Lock (`Locked`, `DmaPinned`) | DMA registration (`DmaPinned`) |
|---|---|---|---|---|---|
| Linux | `mmap(PROT_READ\|PROT_WRITE, MAP_PRIVATE\|MAP_ANONYMOUS\|MAP_NORESERVE)`, then `madvise(MADV_DONTFORK)` | `mbind(MPOL_BIND)` to the arena's NUMA node before first touch | `madvise(MADV_HUGEPAGE)` when THP is `madvise` or `always` (`tier.hugepages = auto`) | `mlock` on the committed sub-range, only if `RLIMIT_MEMLOCK` covers it or `CAP_IPC_LOCK` is held; otherwise the arena stays `Pageable` and `syntropy doctor` says why | `register_host` per pin step (`cudaHostRegister`/`hipHostRegister`, portable; mapped when `GPU_MAPPED`). The provider pins the pages; the tier manager does not also `mlock` them |
| Windows | `VirtualAllocExNuma(MEM_RESERVE)` at `Fetching`, one reservation per *lock range* (one per layer for weights, one per slot-class block for caches, each ≤ `tier.pin_step`); `MEM_COMMIT` per pin step inside the gate (§6), so no commit charge is taken before the gate allows it; `HostPageable` ranges are committed the same way during `Loading` | the `nndPreferred` argument | not used (large pages need `SeLockMemoryPrivilege`; see Open questions) | raise the working set with `SetProcessWorkingSetSizeEx(min, max, QUOTA_LIMITS_HARDWS_MIN_ENABLE)` first, then `VirtualLock` per range | `cudaHostRegister` per range. A DMA, such as one copied miss, never spans two ranges, so slot classes never straddle a range boundary |
| WSL2 (detected by `/dev/dxg`) | as Linux | as Linux | as Linux | as Linux | `cudaHostRegister` on the anonymous range, never `cudaHostAlloc` beyond its ≈1 GiB cap (`tiered-memory-inference.md` §10.2) |
| macOS | `mmap` anonymous, aligned to the 16 KiB page | none (one domain) | none | `mlock` (wires the pages) within `vm.user_wire_limit` | not applicable: Apple GPUs use the `Unified` path (§9) |
| Android, iOS (Tier 3) | `mmap` anonymous | none | none | none | none: `HostPageable` only (RFC-0001 §9.2) |

Registration with the I/O engine (`IO_FIXED`) and with RDMA (`RDMA_MR`) is done by RFC-0021 and RFC-0031 on arenas this RFC marks. io_uring fixed buffers are charged to `RLIMIT_MEMLOCK` and limited to 1 GiB each (ADR-0013), so only `IoStaging`, `ExpertCache` and `ExpertStaging` over an `Nvme` home, `RowCache` and KV spill pages are marked `IO_FIXED`. Bulk reads into `Weights` arenas during `Loading` use plain `O_DIRECT` reads into the same memory, which need no registration (RFC-0021 §4.1).

Registration failure is not fatal. A `DmaPinned` range whose registration fails stays resident as `Underpinned { have: Locked | Pageable }`, is costed as `HostPageable`, and is never used as a DMA source; the runner reports a `Capacity` drift with the registrable total it actually achieved, and RFC-0001 §7.2 decides whether to re-plan.

*Note 2026-10-22 (TIER-E2-T02, the Linux adapter as built):* `syntropy_tier::os::linux::LinuxMemory` implements the Linux row above. It is built for x86-64 and AArch64 only, the Linux targets ADR-0020 supports, because its constants and C signatures are those of these two; another Linux architecture has no adapter. The parts this section leaves open are settled as follows.
- **The trait.** `OsMemory` returns `OsMemError`, because `core` has no `Error` type yet. `reserve` and `commit` failures are `TIER_HOST_RESERVE` (4). §3's `pressure()` arrives with the yield ladder (§11).
- **NUMA binding.** A failed `mbind` is retried once. After the second failure, the range is returned with `HostRange::numa = None`, which is how the caller learns of `TIER_NUMA_BIND` (§15). Docker's default seccomp profile refuses `mbind` without `CAP_SYS_NICE`, so in such a container every arena is unbound.
- **Commit.** `MADV_POPULATE_WRITE`, retried on `EINTR`. Where the kernel answers `EINVAL`, every page is touched with an atomic no-op read-modify-write on one byte (`fetch_or(0)`), which keeps the bytes already there and never discards a concurrent store. A caller writing the range must not access the same bytes non-atomically while `commit` runs, which `OsMemory::commit` states.
- **Unlock** releases only the pages wholly inside the sub-range, because `mlock` is not counted per caller; pin steps are page-aligned (§6.1), where this is exact.
- **Lock.** `lock` checks `VmLck + len ≤ RLIMIT_MEMLOCK` (or `CAP_IPC_LOCK`) before calling `mlock`. A refusal is `OsMemError::LockRefused(PageableReason)`. `lock_or_pageable` turns that into `LockOutcome::Pageable(reason)`, the structured finding `syntropy doctor` renders (PROD-E9-T01).
- **Snapshot.** `own_locked` is `VmLck`. `own_registered` is `VmPin`. The cgroup record is the level, the process's own cgroup or an ancestor, with the least `min(max, high) − current`. A cgroup path with a `.` or `..` component is not followed. Event counts are deltas from the previous reading of the same level; when the binding level changes, they restart at zero.
- **`PinLimits::io_fixed_max`** is this process's view, `RLIMIT_MEMLOCK − VmPin`. It is an upper bound on the per-user `io_fixed`, which the daemon divides (§5).
- **`tier.hugepages`** is not in the config schema yet (TIER-E1-T09), so callers pass `HugePages`.
- **Range ownership.** The adapter records every range it reserves under a reservation id, carried in `HostRange::os_handle`. `commit`, `lock`, `unlock` and `release` refuse a range whose base, length or id differs, because `HostRange`'s fields are public. The registry lock is held across each system call, shared by `commit`, `lock` and `unlock` and exclusive in `release`, so a range is never unmapped while another call acts on it. A stale handle whose address a new reservation reused does not match the new id.

#### 4.3 `Nvme`

No arena. `plan_arenas` emits an `ExtentRef` per `Nvme`-homed group from the pack sidecar. Expert-cache and row-cache slots in faster tiers are filled from these extents by RFC-0021 reads. Session blobs and KV spill files on `Nvme` belong to the store and RFC-0022.

#### 4.4 Ordering inside `Loading`

1. Context reserve: each dispatcher measures device free memory before and after creating its context (§8.1).
2. `Vram` arenas in this order: `Scratch`, `Graphs`, `Weights`, `Kv`, `ExpertStaging`, `ExpertCache`. The elastic expert arenas are last, so a shortfall shrinks them first (§8.2).
3. Host arenas already exist from `Fetching` (§6). `Weights` arenas fill from the pack in pin-priority order, so pinned ranges fill first and later ranges are filled while pinning continues.
4. `Vram`-homed groups load by NVMe → `IoStaging` → provider `copy`, double-buffered (RFC-0021 §8); `Unified`-homed groups load by direct reads into the arena (§9).
5. Registrations; then residency map version 1 is published (§7).

### 5. Pinned-memory budgets per OS

These formulas produce `TierInfo::usable` and `TierInfo::pin_limit` for the host tiers. RFC-0010 evaluates them at probe time and publishes them in the profile; the tier manager evaluates them again before every pinning step (§6), because other programs change the inputs. `headroom` is `resource.ram_headroom`, default `max(4 GiB, 0.10 × RAM)`. `own` is what this node's runners already hold in that tier, so a re-plan of a running instance does not count its own memory against itself.

**Linux** (bare metal and containers):

```text
headroom    = max(4 GiB, 0.10 · MemTotal)
avail       = MemAvailable + own_resident − headroom
cgroup      = min(memory.max, memory.high) − memory.current + own_resident     (v2; v1: memory.limit_in_bytes − memory.usage_in_bytes + own)
host_usable = min(avail, cgroup)                                               → HostPageable.usable
lockable    = ∞                                    if CAP_IPC_LOCK, or the probe shows provider registration is not charged to RLIMIT_MEMLOCK
            = RLIMIT_MEMLOCK − VmLck               otherwise
pin_limit   = min(host_usable, max_pinned_fraction · MemTotal, lockable)        → HostPinned.pin_limit
per NUMA n: numa_available_n = MemFree_n + Active(file)_n + Inactive(file)_n + SReclaimable_n − numa_reserve   (reserve subtracted here, once)
            host_usable_n    = min(numa_available_n + own_n, host_usable)      (numa_reserve default 1 GiB; from /sys/devices/system/node/node<n>/meminfo)
io_fixed    = RLIMIT_MEMLOCK − locked_vm of the user   (∞ with CAP_IPC_LOCK)    caps IO_FIXED registrations (RFC-0021)
```

`io_fixed` is per user, not per process: io_uring charges registered buffers to the user's `locked_vm`, so the daemon's ledger divides it among the node's runners, and an `ENOMEM` from `io_uring_register_buffers` is treated as the real limit (RFC-0021 falls back to unregistered reads; efficiency, not correctness, is lost). `numa_available_n` is an estimate of what the node can supply after reclaiming clean cache; the pinning loop never waits on `MemFree_n`; it gates on available memory, cgroup headroom and pressure signals, and lets the commit itself reclaim (§6). Whether a provider's host registration is charged to `RLIMIT_MEMLOCK` is not documented for current drivers. RFC-0010's probe settles it per machine: it registers a 64 MiB range and reads `VmLck` and `VmPin` from `/proc/self/status` before and after. `max_pinned_fraction` defaults to 0.85 on Linux.

**Windows:**

```text
headroom    = max(4 GiB, 0.10 · ullTotalPhys)
host_usable = min(ullAvailPhys + own − headroom, job-object limit − usage + own)
pin_limit   = min(host_usable, max_pinned_fraction · ullTotalPhys, lockable_measured)     (max_pinned_fraction default 0.50)
helper GPUs: registration for any GPU after the first ≤ 8 GiB (WDDM, tiered-memory-inference.md §10.2)
```

`lockable_measured` comes from the probe, which locks and registers 1 GiB ranges up to `max_pinned_fraction` and records the first failure. It is cached in the profile by hardware fingerprint, driver version and OS build.

**WSL2:** as Linux, applied to the VM's `MemTotal` (set by `.wslconfig`), with `max_pinned_fraction` default 0.75 and the `cudaHostAlloc` cap recorded so that nothing uses it for more than 512 MiB. KV spill to `HostPinned` is off by default on WSL2, matching the surveyed engine, until RFC-0010 shows the pinned H2D rate within 80% of the same machine's native Windows rate (Open questions).

**macOS:** there is no `HostPinned` tier on Apple silicon; the GPU budget is the `Unified` tier (§9). The cap is the lower of the GPU working-set limit the OS reports and RAM minus the reserve (ADR-0007); `max_pinned_fraction` does not apply unless the user sets it. The reserve is always enforced, which together with wiring is what prevents the compressor panics that motivated ADR-0007:

```text
headroom       = max(4 GiB, 0.10 · hw.memsize)
unified_usable = min(recommendedMaxWorkingSetSize, hw.memsize − headroom, available + own − headroom)          (ADR-0007: no fixed percentage)
wired          ≤ vm.user_wire_limit and vm.global_user_wire_limit, read with sysctlbyname
```

**Android and iOS (Tier 3):** `host_usable = available − headroom` (`os_proc_available_memory` on iOS), `pin_limit = 0`.

`max_pinned_fraction` is `ResourcePolicy::max_pinned_fraction` (RFC-0001 §11.3). A user may raise it; Syntropy never raises it on its own.

**What the solver must enforce.** `HostPinned` and `HostPageable` of one NUMA domain are two tiers over the same RAM, so publishing each with its own `usable` would let a plan count that RAM twice. RFC-0010 therefore publishes, per NUMA domain n, `HostPinned_n.usable = pin_limit_n` (the DMA-registrable share, with `pin_limit_n = min(pin_limit, host_usable_n)`), `HostPageable_n.usable = host_usable_n`, and the pair as one shared pool. The solver must hold both rows:

```text
bytes(HostPinned_n)                          ≤ pin_limit_n
bytes(HostPinned_n) + bytes(HostPageable_n)  ≤ host_usable_n
```

Groups with no `StreamSite` and no copied-miss share are homed in `HostPageable`; RAM-resident experts are split into a `DmaPinned` copyable share and a pageable in-place share (RFC-0023 §3). The exact rows RFC-0011 needs are listed under this RFC's proposed amendments.

*Note 2026-10-22 (TIER-E2-T01, the rule as built):* the formulas above are `syntropy_core::budget` (RFC-0001 §5.13, §18.15). Probe-time evaluation and the check before each pinning step call the same pure functions:
- **Signature.** `ResourcePolicy` is not in `core` yet, and `HostMemInfo` does not carry every input. So the rule is `host_budget(&HostBudgetInputs, &HostBudgetPolicy) -> HostBudget { headroom, host_usable, pin_limit, io_fixed }`, with `host_usable`, `pin_limit` and `io_fixed` also callable on their own.
  - `HostBudgetInputs` holds `os: HostOs`, `mem: &HostMemInfo`, `own`, `own_locked`, `working_set` and `user_locked_vm`.
  - `HostBudgetPolicy` holds `ram_headroom`, already resolved to bytes, plus `max_pinned_fraction` and `numa_reserve`.
  - `host_pool` builds each domain's `HostPoolInfo` from `numa_available_n`.
- **WSL2 is recorded once.** It is `HostOs::Linux` with `HostMemInfo::wsl2` set. The flag selects the 0.75 default, so a caller cannot pick the Linux 0.85 default by mistake.
- **Linux `lockable` keeps the `own` convention:** `lockable = RLIMIT_MEMLOCK − VmLck + min(own_locked, VmLck)`. `own_locked` is what this node's runners already hold locked that `VmLck` counts. The add-back is clamped to `VmLck`, so `lockable` never exceeds `RLIMIT_MEMLOCK`. The caller passes a live `VmLck` together with its live `own_locked`, never a cached probe-time `VmLck`. Without it, a runner pinning 6 GiB under an 8 GiB limit would see a 2 GiB `pin_limit` and fail its own plan. `user_locked_vm` follows the same rule for `io_fixed`: the user's `locked_vm` less what the `io_fixed` ledger has granted.
- **An untested registration charge** (`LockAccounting::registration_charged = None`) counts as charged.
- **Windows before H5.** Until RFC-0010's step H5 measures `lockable_measured`, the static estimate is `HostMemInfo::windows_lockable`, the working-set ceiling obtainable without privilege. With neither value, `pin_limit` is 0: the rule never assumes a lock budget it was not given.
- **`io_fixed` off Linux.** This section defines `io_fixed` for Linux and WSL2 only. On Windows, macOS and Tier 3 it is 0, so no fixed-buffer budget is granted, and RFC-0021 uses unregistered reads. That loses efficiency, not correctness.
- **macOS without a working-set figure.** When `recommendedMaxWorkingSetSize` is not available (`working_set = None`), the term is left out. `unified_usable` is then `min(hw.memsize − headroom, available + own − headroom)`. The reserve that ADR-0007 relies on to prevent the compressor panics still holds. A GPU allocation past the real working-set limit fails at `Loading` with `Capacity`, and the profile is corrected from that (RFC-0010 §2.2). Neither path ever exceeds RAM minus the reserve, so the fallback stays within ADR-0007's rule.
- **macOS with a user-set `max_pinned_fraction`.** The fraction also caps `unified_usable`. `pin_limit` is 0, because there is no `HostPinned` tier.
- **The cgroup term.** `HostMemInfo::cgroup_limit` carries the cgroup headroom, or on Windows the job object's limit minus usage. On v2 the probe takes each level's limit as `min(memory.max, memory.high)`, either file `max` or absent meaning no limit, and keeps the level with the least headroom (issue #1650, PR #1670). v1 has no `memory.high`; its `memory.soft_limit_in_bytes` is not a budget term.
- **The `io_fixed` ledger** (`IoFixedLedger`) holds a limit fixed when it is built. To re-evaluate `io_fixed`, the daemon builds a new ledger over the new limit and replays the current grants into it.

### 6. Progressive pinning

Pinning starts in `Fetching`. It runs on the housekeeping thread. Registration needs a device context, so the provider hands the housekeeping thread a `HostRegistrar`, a `Send` capability that registers and unregisters host ranges off the dispatcher thread. RFC-0040 adds it with a default of `None`. Its flags are a provider-level set defined in `kernels`, because a layer-1 trait cannot name `syntropy-tier`'s `Registrations`; the tier manager maps `GPU_DMA` and `GPU_MAPPED` onto it and handles `IO_FIXED` and `RDMA_MR` itself:

```rust
// syntropy-kernels
bitflags! { pub struct HostRegFlags: u8 { const DMA = 1 << 0; const MAPPED = 1 << 1; } }
fn host_registrar(&self, ctx: &DeviceCtx) -> Option<Box<dyn HostRegistrar>> { None }
pub trait HostRegistrar: Send {
    fn register(&self, region: HostRegion, flags: HostRegFlags) -> Result<Registration, Error>;
    fn unregister(&self, reg: Registration) -> Result<(), Error>;
}
```

CUDA and HIP implement it by making the device's primary context current on the calling thread.

**When registration may run.** Whether a host-memory registration issued from another thread during captured-graph decode delays that decode is undocumented; older CUDA guides list page-locked host allocation among operations that serialise work across streams. Until the M1 spike in Open question 9 shows no effect on step-time p99, **every `DmaPinned` registration completes before the runner reports `Ready`**. Locking that involves no GPU driver (`mlock`, `VirtualLock` for `Locked` ranges) may continue after `Ready`. ADR-0007 item 4 allows `Ready` before pinning completes; this RFC narrows it for DMA registration (Proposed amendments). A provider without a registrar registers on its dispatcher thread during `Loading`.

**Registration and graph capture**. CUDA stream capture in global mode forbids calls such as host registration from other threads while a capture is open. So all `DmaPinned` registration completes before `Warming` starts capturing step graphs, not merely before `Ready`. If Open question 9 later allows registration after `Ready`, the registrar takes a capture lock: the dispatcher holds it for each capture window (initial capture and any re-capture between steps), and the housekeeping thread registers only while it is free.

**Order.** Arenas carry `pin_priority`; ranges within an arena are pinned in plan order:

| Priority | What | Why first |
|---|---|---|
| 0 | `IoStaging`, `NetPool` | `Loading` and the data plane need them; small |
| 1 | `ExpertCache`, `ExpertStaging`, `RowCache` slots in `HostPinned` over an `Nvme` home | every NVMe read lands in them |
| 2 | `DmaPinned` weight ranges: first the experts that the initial VRAM fill will **not** hold, by descending profile frequency; then groups with a `StreamSite`, in layer order | these are the copied-miss and prefill-streaming sources; pinning the experts outside the VRAM cache first gave +4.4% prefill in the surveyed engine (`tiered-memory-inference.md` §10.2) |
| 3 | `Kv` spill pages | needed only once contexts grow |
| 4 | `Locked` ranges of `HostPageable` weights, dense CPU groups first, then experts by frequency | protects against paging; no DMA depends on it |

**Loop:**

```text
step := tier.pin_step                               // default 1 GiB
backoff := 250 ms
for range in ordered_ranges:                        // whole registration units (§6.1), ≤ step, ≥ one unit
    loop:                                           // the gate: will committing this range leave the user enough?
        s := os.snapshot()
        ok := s.available − len(range) ≥ headroom                          // MemAvailable minus the reserve; never MemFree
          and cgroup_headroom(s) ≥ len(range)                              // memory.max − memory.current (and memory.high)
          and s.numa_available[range.numa] ≥ len(range)                    // numa_reserve already subtracted (§5)
          and (s.commit_headroom is None or s.commit_headroom − len(range) ≥ headroom)   // Windows
          and (range.class ≠ DmaPinned or own_pinned + len(range) ≤ pin_limit(s))
          and not pressured(s)                                             // below
          and s.os_level ∈ {None, Normal}
        if ok: break
        if pressured(s) has held continuously for tier.pin_give_up (default 120 s),
           or the capacity terms above have failed continuously for tier.pin_give_up:
            mark every remaining range Underpinned; report Drift { pin_limit: own_pinned }; return
        sleep(backoff); backoff := min(2 · backoff, 5 s)
    backoff := 250 ms
    t0 := now()
    os.commit(range)        // Linux: MADV_POPULATE_WRITE (≥ 5.14; else touch each page). The commit itself
                            // forces reclaim of clean cache through the normal allocator, at its own pace;
                            // Windows: MEM_COMMIT of the reserved range
    match range.class:      // registration pins pages that are now resident; it allocates nothing new
        DmaPinned: os.lock if Windows; registrar.register(range, DMA [| MAPPED]) once per device in dma_devices
        Locked:    os.lock(range)
    on error: range → Underpinned; report; continue with the next range
    post TierCommand::PinStep done to the owning dispatcher, which flips HomeState → Resident
    d := now() − t0
    if d > tier.pin_step_slow (250 ms): step := max(step / 2, one unit); slow_steps += 1 else slow_steps := 0
    if d < 50 ms:                       step := min(step · 2, tier.pin_step)
    if no sequence is live: continue                 // before Ready: no duty-cycle sleep (§Performance)
    sleep(d · (1 − tier.pin_duty) / tier.pin_duty)   // while sequences are live: at most pin_duty of wall time

pressured(s) :=   (PSI readable     and s.psi.some_avg10 ≥ tier.pin_psi_max)           // default 10 %
              or  (PSI not readable and s.direct_reclaim_stalls > 0 during the last step)
              or  slow_steps ≥ 3 at the minimum step size (one unit)                  // the step's own latency
```

**Why there is no free-page test.** On a normal Linux host clean page cache keeps `MemFree` near the zone watermarks while `MemAvailable` is high, and kswapd reclaims only up to the high watermark, which is well under a gigabyte even on large machines. Any wait for `MemFree` to exceed a fixed amount, before or after committing, can therefore stall forever. The design does not count free pages at all:
- **Capacity** is judged by `MemAvailable` minus the reserve, the cgroup's `memory.max − memory.current`, and the per-node available estimate, all of which count reclaimable cache.
- **Reclaim** is forced by the commit itself (`MADV_POPULATE_WRITE` allocates through the normal path); registration then pins pages that are already resident and takes no new memory, so the driver never takes pages faster than reclaim delivers them (`tiered-memory-inference.md` §10.2).
- **Distress** is read from the system's own signals: PSI `some avg10` from `/proc/pressure/memory`, and the step's own latency. When either crosses its threshold the loop backs off, and it gives up only when distress persists for `tier.pin_give_up`.
- **Kernels without PSI** (built without `CONFIG_PSI`, or booted with `psi=0`): the direct-reclaim stall counters (`allocstall_*` in `/proc/vmstat`, readable without privilege) replace PSI, and step latency is used as on every kernel. If neither is readable (some sandboxes), only step latency and the capacity terms gate pinning, and `syntropy doctor` says so.

#### 6.1 Registration units

One registration covers one **unit**, and no DMA source crosses a unit boundary on any OS:
- `Weights` arenas (`Extents`): a run of whole extents cut only between extents, close to `tier.pin_step`, never smaller than `tier.pin_unit_min` (256 MiB) unless a single extent is larger, in which case that extent is its own unit.
- Slot-class arenas (`ExpertCache`, `RowCache`, `Kv` spill pages): one elastic segment. Host elastic arenas use segments of `tier.pin_step` rounded down to whole slots, striped like VRAM segments (§8.3), so a host shrink unregisters whole units.
- `Ring` arenas (`IoStaging`, `NetPool`, `AttnHandoff`): one unit each.
- Windows: units are also the lock ranges (§4.2).

The registration count per arena is at most `⌈bytes / tier.pin_unit_min⌉` plus the number of extents larger than `tier.pin_step`: a 115 GB arena is ≈108 units at 1 GiB and at most ≈430 at the 256 MiB minimum [estimate]. RFC-0040's "one registration per sub-arena, in `Loading`" bullet is amended to this rule (Proposed amendments; follow-up to COMP).

**Overlap with `Fetching`.** Pages are committed and pinned empty while the shard downloads, so the pinning time hides behind the download and pack build; bytes are read into already-pinned memory during `Loading`. When the pack is already present (a restart), `Loading` reads into ranges in the same priority order, so DMA sources are ready first.

**What the solver sees meanwhile.** `Underpinned` ranges are costed as `HostPageable` (ADR-0007 item 4). The miss balancer (RFC-0023) never picks a copied miss whose source is not `DmaPinned`, and the n* switch (§10) never streams from it. The plan's predicted costs assume full pinning; the drift detector (RFC-0001 §7.4) ignores the gap until pinning has finished or given up, so slow pinning does not trigger a re-plan by itself.

### 7. Residency map, handles and leases

#### 7.1 Layout and writers

`ResidencyMap` (RFC-0001 §5.5) is the logical view. Physically it is split so that every field has one writer:

| Table | Indexed by | Holds | Writer |
|---|---|---|---|
| `HomeTable` | `TensorGroupId` | `home`, `home_handle`, `home_state` | the dispatcher of the device that owns the home tier: a GPU for `Vram`, the NUMA domain's CPU device for host tiers, the integrated GPU for `Unified`, the CPU device of the volume's NUMA domain for `Nvme` extents |
| `CopyTable[d]`, one per device | `TensorGroupId` (sparse, open addressing) | `CachedCopy` entries for copies in device d's tiers, **and** for copies in a RAM expert cache over an `Nvme` home that device d consumes | device d's dispatcher. A RAM expert cache over an `Nvme` home has one owner, the dispatcher of the GPU (or `Unified` device) whose misses it serves; the CPU device's shard holds only that arena's memory. On a CPU-only node, the CPU device's dispatcher owns it. Promotion, demotion and demand fills are therefore single-writer |
| `SlotState[arena]` | slot | generation (`AtomicU32`), lease count (`AtomicU16`), free-after fence (`AtomicU64`) | the dispatcher that owns the arena's residency (for the RAM expert cache over an `Nvme` home, the consuming GPU's dispatcher, though the arena lives in the CPU shard); lease counts by any thread |

Each table is published with a seqlock: the writer increments its version to an odd value, writes, then increments to even; a reader copies what it needs and retries if the version changed. `ResidencyMap::version` is a runner-wide counter that every writer increments after a publish, so the digest (§13) can tell whether anything changed. `TierManager::snapshot()` assembles the RFC-0001 struct for the debug surface; nothing on a hot path calls it.

The hot path does not look residency up per token. Dense groups' `BufferView`s are resolved once when the step program is built in `Warming` and stay valid for the epoch, because `Weights` arenas never move. Expert slots are found through the device-resident slot table (RFC-0023), and KV pages through the page table (RFC-0022). The residency map is the record those structures are derived from and checked against.

#### 7.2 State machines

```text
HomeState:  Absent ──read issued──▶ Loading ──completion──▶ Resident ◀──pin──▶ Underpinned
            Extent (static for Nvme homes)

CopyState (one cache or staging slot holding a copy of group g):
  Free ──admit/prefetch──▶ Inflight{ticket} ──completion (dispatcher)──▶ Resident
  Resident ──victim chosen──▶ Evicting ──slot-table MISS applied at step s, fence ≥ s, leases = 0──▶ Free (generation += 1)
  Inflight ──cancel or I/O error──▶ Free (generation += 1)
```

`Inflight` and `Evicting` copies are never read. A copy becomes readable only when the dispatcher, as the single writer, sees the completion and then publishes the slot-table entry (RFC-0023 §5.4). Generation 0 is never issued, so a zeroed handle never resolves.

#### 7.3 `TensorHandle` resolution

The 16-byte layout is RFC-0001's: `tier: u8` (node-local tier ordinal), `arena: u8`, `layout: u16` (pack layout variant), `slot: u32`, `generation: u32`, `pages: u32`. Resolution is lock-free and constant time:

```text
resolve(h):
    a := directory.arena(h.tier, h.arena)                // immutable for the epoch
    if a.slot_state[h.slot].generation.load(Acquire) ≠ h.generation: return Err(Stale)
    base := match a.geometry:
        Extents      → a.extent_offset[h.slot]
        Classes      → a.class_base[class_of(h.slot)] + index_in_class(h.slot) · a.class_size[...]
        Pages, Ring  → h.slot · a.unit
    return BufferView { kind: a.view_kind, addr: a.segment_addr(base), len: h.pages · 4096 }
```

**Layouts of groups read by several devices**. `layout` names the byte layout of the bytes a handle points at. Admissions, copied misses and streaming are raw byte copies, so a copy keeps the layout of the bytes it was copied from. A group that more than one device may compute (an expert under the miss split: GPU for hits and copied misses, CPU for in-place misses; a streamed group; a group with a level-F fallback device) is one of two kinds, chosen by RFC-0011 from the layout pass's per-reader times (RFC-0040 §6.1, RFC-0041 `PackBuild`):
- **Shared (the default):** one stored copy whose layout every consuming kernel accepts.
- **`per_site_layout`:** the pack holds a second host copy in the GPU's layout beside the home copy in the CPU's layout, and the solver charges its bytes to the tier that holds it. The CPU computes in-place misses from the home copy; **VRAM admissions and copied misses copy from the GPU-layout host copy**, which must be `DmaPinned`. The residency map records the GPU-layout copy as a `CachedCopy` of the group with its own `layout`, and the slot table points GPU kernels only at copies in a layout they accept.

`plan_arenas` checks both kinds: every copy's layout must be in the accepted set of each op that reads it (`KernelProvider::query`, passed to `plan_arenas` as its `AcceptedLayouts` argument; §3, note of 2026-10-22); otherwise `Loading` fails with `Unsupported` and the pack is rebuilt with the source layout, which every provider accepts.

A stale handle on a hot path is a bug (`Internal`, `TIER_STALE_HANDLE`); off the hot path, such as an I/O completion for a cancelled prefetch, it is expected and silently discarded.

#### 7.4 Leases and fences

RFC-0001's `lease()` pins a slot for one launch. There are two implementations, chosen by the caller's thread:

- **Step fences (hot path).** Steps on one device complete in order. The step program raises the constant-valued step-done flag `O[p]` (RFC-0040 §4.2, RFC-0012 §3.2); the dispatcher consumes it, resets it, and increments the device's **completed-step counter**, a host word, in step order. Fences compare against that host counter, never against a device-written step number. A lease taken by the dispatcher records the current step number as its fence; it costs nothing to release. A slot evicted at step s gets `free_after = s`; it may be overwritten once the device's completed-step counter is at least s. Copies into a slot on another stream wait on the event of step s on the device (`KernelProvider::wait`), never on the host.
- **Counted leases (off the hot path).** I/O completions, KV export, migration and debugging increment the slot's lease count and decrement it on drop. `release_slot` defers reuse until the count is zero and the fence has passed.

#### 7.5 Invariants

These are the property-test oracle (§Testing):

1. Every field has one writer thread, the one named in §7.1.
2. A slot's generation strictly increases; a handle resolves only while its generation matches.
3. No slot is written by I/O or a copy while a step that may read it is incomplete or a counted lease is held.
4. A kernel reads a copy only in state `Resident`.
5. Every DMA source is `DmaPinned` and `Resident`; `Underpinned` and `HostPageable` memory is never a DMA source.
6. For every tier, the sum of mapped arena bytes is at most the ledger grant, and pinned bytes never take `available` below `headroom` at the moment of pinning.

### 8. VRAM budget split and elastic segments

#### 8.1 The split

RFC-0001 §9.3 fixes the order: context reserve, scratch, dense hot set, KV hot window, expert cache (the remainder), headroom. The solver sets every size except the context reserve and passes them in the plan. At `Loading` each GPU dispatcher checks them against reality:

```text
free_0    := device free memory before context creation   (cuMemGetInfo / hipMemGetInfo / VK_EXT_memory_budget heapBudget − heapUsage / DXGI QueryVideoMemoryInfo.Budget − CurrentUsage)
ctx       := free_0 − free after context creation and provider init      (measured; the probe's value is only a prior)
need_fix  := ctx + scratch + graphs + dense_hot + kv_vram + staging
need_ec   := ExpertCacheBudget(d).bytes
headroom  := resource.vram_headroom   (default 512 MiB on a GPU that drives a display, 256 MiB otherwise)

if need_fix + need_ec + headroom ≤ free_0: map everything
elif need_fix + min_ec + headroom ≤ free_0: map need_fix, and only ⌊(free_0 − need_fix − headroom) / segment⌋ expert segments
     → report Capacity drift { vram_usable }  (the runner still reaches Ready; RFC-0001 §7.2 decides about re-planning)
else: fail Loading with Capacity (TIER_VRAM_SHORT); the daemon corrects the profile's usable VRAM and the coordinator re-plans
```

`min_ec` is `min_segments × segment` from the elastic spec, default the larger of one segment and 5% of the planned expert cache. The KV hot window is never shrunk at `Loading`; RFC-0022 sized it for the plan's admitted contexts, and KV admission (RFC-0001 §8.4) depends on it.

#### 8.2 Segment size

`segment = round_up(max(tier.vram_segment, 8 × largest slot in the arena), allocation granularity)`, with `tier.vram_segment` defaulting to 256 MiB. The factor of 8 bounds the bytes lost at the end of each segment to one slot in eight in the worst case. Examples (estimates from `backends-and-portability.md` §4.3): a DeepSeek-V3 expert at Q4_K is 23.6 MiB, so segments are 256 MiB; a Mistral Large 3 expert at Q4_K is 47.3 MiB, so segments are 384 MiB.

#### 8.3 Striped slot layout

If each layer's slots were contiguous, unmapping the tail segment would empty the last layers' caches. Instead, each segment holds a proportional share of every layer's slots:

```text
given per-layer slot counts n_l and sizes s_l (RFC-0023 §4), S := number of segments
for each segment k in 0..S:
    quota_l(k) := largest-remainder split of n_l over S segments
    pack the quota_l(k) slots of every layer into segment k, largest size first; no slot crosses a segment boundary
slot descriptor table: slot → (segment, offset, layer)   (in the TierDirectory; replaces the class formula of §7.3)
```

Removing the last segment removes about 1/S of every layer's slots, so per-layer hit rates degrade evenly instead of some layers losing their whole cache. Slot IDs are numbered segment by segment, so the slots of a removed segment are a contiguous ID range.

#### 8.4 Shrink and grow

```text
ShrinkElastic { arena, segments: k }       (from the pressure ladder §11, or a PlanDelta with a smaller ExpertCacheBudget)
  0. compaction: for each layer, move its hottest residents out of the last k segments into the
     slots of its coldest residents in the surviving segments, coldest first, while a tail resident is hotter
     than the coldest surviving one: an in-tier copy (device-to-device on a GPU, memcpy for a host arena)
     charged to the admission budget, published as a slot-table update after the copy lands. After compaction
     the tail holds each layer's coldest experts, which is what rung 2 is priced on (§11.4).
  1. victims := every slot in the last k mapped segments
  2. Resident copies → Evicting; post slot-table MISS for each, effective at the next step s (RFC-0023 §5.4)
     Inflight copies → cancel the read or copy; when it completes, drop it
  3. wait until completed_step ≥ s and every victim's lease count is 0     (checked between steps; no host sync)
  4. unmap_segment for each segment, between steps; generation += 1 for every victim slot
  5. ReportLedger to the daemon with the smaller grant; per-layer budgets shrink by the removed quotas
GrowElastic { arena, segments: k }
  1. map_segment for k segments past the current tail (fails cleanly if VRAM is not free: report, retry after 30 s)
     (host DmaPinned arenas: commit and lock the new units; register them only as §11.6 allows)
  2. new slots are Free; RFC-0023 admits into them by policy, off the critical path
```

Remaining slots keep their addresses, so captured graphs and the slot table stay valid (RFC-0001 §5.5). The only driver calls are segment map and unmap, issued between steps and never per token. Where a provider lacks `map_segment` (no VMM), a shrink only retires slots logically: no memory is returned until the runner reloads, and the runner reports that, so the dashboard does not claim VRAM was given back.

#### 8.5 Another program takes VRAM

The housekeeping thread watches each GPU at 1 Hz: NVML free memory on NVIDIA, the DXGI budget and its change notification (`RegisterVideoMemoryBudgetChangeNotificationEvent`) on Windows, `VK_EXT_memory_budget` on Vulkan, and amdgpu `mem_info_vram_used` on Linux AMD. When free memory stays below `vram_headroom / 2` for 2 s, or on Windows the OS budget falls below this process's usage, it shrinks the expert cache by enough segments to restore the headroom. That is rung 2 of the ladder in §11, applied to VRAM. If the non-elastic arenas alone exceed what is left, the runner reports `Capacity` drift and the coordinator re-plans. On Windows this matters more than elsewhere: past the budget, the OS may move the process's VRAM to system memory, and step times collapse without any error.

Growth back to the planned size waits until free memory has stayed above `2 × vram_headroom` for `tier.reacquire_after`, and then proceeds one segment per 5 s.

#### 8.6 KV against experts

The solver trades the KV hot window against the expert cache (RFC-0001 §9.3). Inside one epoch, the tier manager never moves bytes between the `Kv` and `ExpertCache` arenas on its own initiative. A different trade is a new plan whose budget-only changes arrive as a `PlanDelta`: the runner shrinks one elastic arena and RFC-0022 adjusts its hot window inside the fixed `Kv` arena. The `Kv` arena is sized for the largest hot window the plan admits.

### 9. Unified-memory devices

On Apple silicon, Strix Halo and GB10-class machines the tier graph is {`Unified`, `Nvme`} with no PCIe term (ADR-0014; cost model §2.3). One `Unified` tier per machine has the integrated GPU and the CPU device in its affinity set. Its `usable` is the GPU allocation cap, not RAM.

| Platform | `usable` | Arena construction | GPU visibility | Compute site rule |
|---|---|---|---|---|
| Apple silicon | §5 macOS formula: min(`recommendedMaxWorkingSetSize`, RAM − headroom, available − headroom) | `mmap` anonymous at 16 KiB alignment, length a multiple of 16 KiB; `mlock` (wire) within the user wire limit; one Metal buffer per 256 MiB segment with `newBufferWithBytesNoCopy` (page-aligned pointer and length are required) | added to an `MTLResidencySet` per queue on macOS 15+ (ADR-0014), or `useResource`/`useHeap` per encoder before that | the GPU by default; a group goes to the CPU only where RFC-0010's microbenchmark shows the CPU path faster for that kernel class |
| Strix Halo (Linux, amdgpu) | the GPU-visible heap the driver reports (carve-out plus GTT) | anonymous host memory imported into the GPU without copying: `hipHostRegister` on ROCm, `VK_EXT_external_memory_host` on Vulkan; the BIOS carve-out, if any, is allocated through the provider as device memory | imported memory is GPU-addressable; the carve-out is CPU-mapped where the driver exposes it | the iGPU, because the CPU side reads only ≈124 GB/s of the 256 GB/s peak (`hardware-and-performance-model.md` §2.1) |
| Strix Halo (Windows) | the "variable graphics memory" heap reported by DXGI or Vulkan | provider allocation from that heap | yes | the iGPU |
| GB10-class (Linux, CUDA) | MemAvailable − 6 GiB, because `cudaMemGetInfo` undercounts (`tiered-memory-inference.md` §10.5) | `KernelProvider::alloc(MemKind::Unified)`; RFC-0040 decides which CUDA allocation call gives full-rate access from both CPU and GPU on this platform | yes | the GPU |

Rules that differ from discrete GPUs:

- **No `HostPinned` tier and no copied misses.** The expert cache over a RAM home does not exist on these machines: every expert in `Unified` memory is already where the GPU computes it. The expert cache that does exist is `Unified` slots over an `Nvme` home (RFC-0023).
- **NVMe reads land in the final buffer.** Reads from the pack go straight into `Unified` arena slots that the GPU imports without copying (RFC-0001 §9.4). On Apple this uses `MacNoCache` or `MacMapped`, whichever RFC-0010 measured faster on that volume (ADR-0013, RFC-0021 §5).
- **Detection is conservative.** If the probe cannot confirm that CPU and GPU share memory at full rate, the device is modelled as discrete with a measured PCIe term (ADR-0014 feature matrix). A wrong guess then costs speed, never correctness.
- **The OS's GPU working-set limit is respected, and raised only by the user.** On Apple, `iogpu.wired_limit_mb` changes only through `syntropy tune` (§9.1). When a plan needs more `Unified` memory than the current limit allows, the model picker (RFC-0051) shows the plan that fits now, the plan that would fit after tuning, and the trade, and offers `syntropy tune` (ADR-0007, ADR-0014).
- **CPU work competes for the same bandwidth.** The CPU compute pool, the I/O engine's DMA and the GPU all read one memory system. RFC-0010's probe measures GPU `W_eff` with the I/O engine reading at the planned rate, so the profile already includes the contention.

#### 9.1 Raising the OS limit: `syntropy tune`

Some plans need more `Unified` memory than the OS lets the GPU wire by default: DeepSeek-V3-class 4-bit weights are about 74% of a 512 GB Mac (cost model scenario b1). ADR-0007 makes raising the limit an explicit opt-in. The flow (CLI owned by RFC-0051, records and uninstall by RFC-0052, the numbers by this RFC):

1. **Offer.** When a plan is infeasible only because of this limit, the solver's infeasibility report says so (RFC-0011), and the placement preview and model picker (RFC-0051) state that the limit is the cause and offer the opt-in. `syntropy tune` with no arguments shows the current limit (`iogpu.wired_limit_mb`, read with `sysctlbyname`, and `recommendedMaxWorkingSetSize`), the plan that fits today, the plan that would fit after tuning, and the proposed limit. The proposal is the smallest value that fits the plan's `Unified` bytes, and never more than RAM minus `resource.ram_headroom`, so the reserve survives tuning.
2. **Approve.** Applying it needs administrator approval (the system's authorization prompt); Syntropy holds no standing privilege.
3. **Record.** Before writing, the daemon records the original value, the new value, the time and the reason in its state directory. `syntropy tune --status` prints the record.
4. **Persist, if asked.** The sysctl does not survive a reboot by itself. If the user asks for persistence, packaging installs a launchd job that re-applies the recorded value at boot; it is listed in the record. Without it, the daemon notices at its next start that the limit is back to the default, re-probes, and plans for the lower limit.
5. **Re-probe.** After any change, RFC-0010 re-reads the limit and `recommendedMaxWorkingSetSize` and reports the new `usable`; the coordinator re-plans under RFC-0001 §7.2.
6. **Revert.** `syntropy tune --revert` restores the recorded original and removes the launchd job; uninstall does the same (RFC-0052). If the recorded original no longer matches the current value because the user changed it by hand, revert leaves it alone and says so.

No other code path writes the sysctl; the probe and the tier manager only read it. Tests check that with a syscall trace (Testing).

*Note 2026-10-23 (TIER-E2-T13):* the tier side, as built in `syntropy_tier::tune`.
- **The proposal.** `propose(plan_unified, current, ram, headroom)` returns `Fits`, or `Raise { limit_mb }` with the plan's `Unified` bytes rounded up to whole MiB. It returns `TooLarge` when they exceed RAM minus the reserve.
- **The record.** It is `<state>/tune.json`, a `TuneRecord` holding the original value, the new value, the Unix time, the reason, the version, a state (`Pending`, `Applied`, `Reverted` or `Abandoned`, as RFC-0052 §11.1's ledger names them) and the launchd job, if any. It is written whole, through a new file renamed over the old one, before the sysctl changes. A second tune keeps the first original while the value is still the one tune wrote.
- **The write is the caller's.** `tune` hands each change, `HostOp::SetWiredLimit` or `HostOp::RemoveBootJob`, to a `PrivilegedRunner` the caller supplies. The CLI supplies one that asks through the system's authorization prompt (PROD-E6-T14), so `tier` holds no privilege and makes no write. `tier` reads the sysctl only through `sysctlbyname` with no new value (`os::macos`).
- **The re-probe** is the `Reprobe` trait, told after every change. The probe (PLC-E1-T03) and the coordinator implement it.
- **The hand-edit check** compares the current value with the value tune *wrote*, the record's `new_mb`. Step 6's "the recorded original no longer matches the current value" can only mean that, because after a successful tune the current value never equals the original. A revert removes the launchd job even when it leaves a hand-set value alone, because the job would re-apply tune's value at the next boot. Uninstall calls the same revert.
- **The syscall trace.** It needs a system-call tracer such as DTrace, which System Integrity Protection limits. Whether a hosted runner allows one was not tried, so the trace stays with the macOS rig (TIER-E2-T24). Until then, a static check over every crate's sources stands in: every `sysctlbyname` call passes no new value, and only `tune` spells out the write.

### 10. Execute-in-place versus PCIe streaming at runtime (n*)

The plan names, per tensor group, `ComputeSite { decode, batched: Option<StreamSite { device, via, min_tokens }> }` (RFC-0001 §5.4). `min_tokens` is n* computed by the solver from the profile. The tier manager owns the runtime half of the rule: it refines n* from live timings and answers, for every pass, whether each group executes in place or is streamed. RFC-0012 asks the question when it builds a step; RFC-0023 applies the same model to the copied-miss share of MoE layers.

**Cost model per group g and pass size n** (cost model §3.2, with fitted intercepts):

```text
t_EIP(n)    = o_cpu + max( B_g / W_exec(home) ,  n · c_cpu(g) )        c_cpu(g) = 2·P_g / F_cpu(kernel class, dtype)
t_stream(n) = o_gpu + max( B_g / W_pcie       ,  n · c_gpu(g) )        perfect overlap with the previous group's compute
n*(g)       = min { n : t_stream(n) < t_EIP(n) }
            ≈ F_cpu · b / (2 · W_pcie)     when W_pcie < W_exec and the CPU is compute-bound at n*  (glossary: n*)
```

**Fit.** The per-step telemetry ring (RFC-0001 §13) records, per stage and kernel class, the CPU time of in-place groups and the copy and compute times of streamed groups, with the pass size. The `eip` module fits `c_cpu`, `o_cpu`, the PCIe rate and `c_gpu` by least squares with exponential forgetting (half-life `tier.eip_halflife`, default 60 s of samples), starting from the profile's values. Every `tier.eip_refit` (default 30 s) it recomputes n* per (kernel class, dtype) and adopts the new value only if it moved by more than 10%, so the switch does not oscillate on noise.

**Decision per pass:**

```text
stream(g, n) :=  g.compute.batched is Some(site)
             and n ≥ n*_live(g)
             and home_state(g) = Resident with pin class DmaPinned        // never from Underpinned or HostPageable
             and stream staging on site.device is free for this step
otherwise execute in place on g.compute.decode
```

**How the scheduler asks**. RFC-0012 selects a captured program per pass; it must not decide streaming from the plan's static `min_tokens` alone, because a group whose home is still `Underpinned` must not be streamed (invariant 5, RFC-0001 §9.4) and `n*_live` changes. The tier manager answers with a bitset:

```rust
impl TierManager {
    /// Groups of `stage` to stream for a pass of `n` tokens: `stream(g, n)` above, as a bitset over
    /// the stage's streamable groups, plus a version that changes only when the answer for some n changes.
    pub fn stream_set(&self, stage: StageIx, n: u32) -> (GroupBitset, u32);
}
```

RFC-0012 captures one `PrefillStream` program per distinct bitset it has seen (the set changes rarely: as pinning completes and when `n*_live` is refit), or uses per-group predicated copy nodes where the provider supports them. Groups outside the set run in place in the same program.

When the online fit has fewer than 32 samples for a kernel class, `n*_live` is the plan's `min_tokens`. If the fit says streaming never pays (`W_pcie ≥ W_exec` and the CPU is never compute-bound), `n*_live` is ∞ for that class until the next refit. If it says streaming always pays at n = 1, the decode path still executes in place unless the plan's `min_tokens` is already 1: changing the decode compute site is a solver decision (ADR-0006 item 7), reported as drift so RFC-0011 can make it. When the measurement holds, the solver streams at decode, automatically, from measurement: D1 describes a reference design, not a ban on streaming, and wherever streaming beats execute-in-place at decode the solver chooses it (OQ-PL-05, measured by spike S-05 on PCIe 5.0 with one CCD). The tier manager then serves `min_tokens = 1` exactly as any other stream site: same staging, same `DmaPinned` precondition, same parity requirement.

**Streaming mechanics.** Two stream buffers per streaming GPU, each the size of the largest streamed layer's groups, sit in that GPU's `Scratch` arena and are counted in `plan.scratch`. The copy stream fills buffer (l+1) mod 2 while the compute stream runs layer l from buffer l mod 2; the compute stream waits on the copy's event device-side. Prefill micro-batches are chosen by RFC-0012 to be at least `n_hide = F_gpu · b / (2 · W_pcie)` where VRAM allows, so the copy hides behind compute (SC-10). For MoE layers in prefill, the routed pool is streamed per chunk under RFC-0023's rules.

**Unified-memory devices** have no `batched` site: the compute site is chosen once per kernel class by microbenchmark (§9).

**Exactness.** In-place and streamed execution of the same group are both exact-mode paths and must both pass the parity harness (ADR-0019 item 4). Which one runs never depends on anything but measured timings and n, so a deterministic run (ADR-0019 item 8) fixes `n*_live` for its duration.

### 11. Memory-pressure response on personal devices

Pinned memory cannot be paged out, so the OS cannot take it back on its own. Giving it back is Syntropy's job (ADR-0007 item 10). The daemon detects pressure, decides how much each runner returns, and each runner returns it along the ladder of RFC-0001 §10.3, predicting the token-time cost of each rung before taking it (ADR-0202).

#### 11.1 Signals (daemon)

| OS | Sources | Read how |
|---|---|---|
| Linux | `/proc/pressure/memory` (`some`/`full avg10`); the cgroup's `memory.events` (`high`, `max` counters) and `memory.pressure`; `MemAvailable`; per-NUMA `MemFree` | a PSI trigger with `poll()` where the kernel allows unprivileged triggers (windows must be a multiple of 2 s); otherwise the files are read every second |
| Windows | `CreateMemoryResourceNotification(LowMemoryResourceNotification)`; `GlobalMemoryStatusEx`; commit charge against the commit limit | wait on the notification handle; poll the rest every second |
| macOS | a dispatch source of type `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` (normal, warn, critical); `host_statistics64` free + inactive + purgeable pages | event-driven plus a 1 s poll |
| Android, iOS (Tier 3) | the host app's trim-memory or memory-warning callbacks, forwarded through the in-process runner host; `os_proc_available_memory` on iOS | event-driven |

#### 11.2 Levels

```text
Linux:    Moderate  some avg10 ≥ 10 %  or available < 1.5 · headroom
          High      some avg10 ≥ 25 %  or available < headroom   or memory.events.high increased in the last 10 s
          Critical  full avg10 ≥ 5 %   or available < headroom / 2  or cgroup usage ≥ 0.95 · memory.max
Windows:  Moderate  available < 1.5 · headroom, or commit charge ≥ 0.90 · commit limit
          High      low-memory notification signalled, or available < headroom
          Critical  available < headroom / 2, or commit charge ≥ 0.95 · commit limit
macOS:    Moderate  available < 1.5 · headroom
          High      dispatch source "warn"
          Critical  dispatch source "critical"
```

A level must hold for 2 s before it is acted on, and the level falls one step after 10 s below its threshold. `reclaim` is the bytes that bring `available` back to `1.25 · headroom` at High and `1.5 · headroom` at Critical.

#### 11.3 Who gives back what

Each runner publishes a **yield menu** in its residency digest (§13): for every rung, the bytes it could release and the predicted Δt_tok for its instance. The daemon fills `reclaim` from the cheapest entries across all its runners, cheapest Δt_tok per byte first, and sends each runner `MemoryPressure { level, reclaim }` (RFC-0001 §3.5) with its share. So an idle instance's KV is released before an active instance's expert cache.

#### 11.4 The ladder

The order is RFC-0001 §10.3's. Rung 5 is proposed in this RFC's amendments.

| Rung | Releases | Mechanism | Predicted Δt_tok (shown before; measured 30 s after) |
|---|---|---|---|
| 1 | KV pages of idle sessions | RFC-0022 writes them to session blobs on `Nvme`, then frees the pages | 0 for live sequences. The cost lands on the next resume: `KV_bytes / W_nvme` added to its TTFT (SC-19) |
| 2 | RAM expert-cache slots over an `Nvme` home, in both parts (RFC-0023 §3) | `ShrinkElastic` on the pageable part first, then on the pinned part, lowest decayed counts first after compaction (§8.4) | pageable part: `(H(c) − H(c − Δ)) · B_routed / W_nvme`, with `H` the learned hit curve at the union size `c` (RAM cache plus VRAM cache, RFC-0023 §13), the pessimistic serial form of cost model §3.3; pinned part: the same term plus the lost copyable share, priced by the balancer's fit as the copied bytes that become in-place, `Δ_copy · (c_l − d_l − p_l)` per layer (RFC-0023 §8) |
| 3 | prefetch staging and other caches | spill `History` segments to their file (RFC-0033 §4.1); shrink `IoStaging`, `ExpertStaging` and `RowCache` to their minimums; lower `PrefetchWindow::max_inflight` to match | the cost model's prefetched share at the smaller window: misses no longer issued early are paid as demand reads |
| 4 | host copies of groups whose home is `Nvme`, or that have a deliberate replica elsewhere | free their RAM slots; their next use is a demand read (or, for a replica, the owner serves it, M3) | `Σ B_g · f_g / W_nvme`, with `f_g` the group's reads per token |
| 5 | (Critical only) lock on `HostPageable` homes | unlock `Locked` ranges so the OS can page them instead of killing the process | unbounded if the OS pages hot bytes; reported as `Ready { degraded }` |

When the ladder is exhausted, the runner reports `Drift` with its reduced usable capacity, and RFC-0001 §7.2 decides whether to re-plan. Groups homed in RAM are never freed without a new plan.

#### 11.5 Policy

`ResourcePolicy::memory_yield` (RFC-0001 §11.3; reloadable without restart):

| Value | Acts at | Rungs |
|---|---|---|
| `Never` | Critical only | 1–3; then `Drift`. Chosen by users who dedicate the machine |
| `UnderPressure` (default) | High and Critical | 1–4; 5 at Critical |
| `Aggressive` | Moderate and above | 1–5; also, after `tier.idle_release_after` (default 10 min) with no live sequence on an instance, rungs 1–3 are applied pre-emptively and reacquired when the next request is admitted, which then pays the refill in its TTFT |

#### 11.6 Reacquiring

After the level has been Normal for `tier.reacquire_after` (default 60 s), the runner grows back in reverse rung order, one segment or pin step every 5 s, through the same gate as progressive pinning (§6). Until Open question 9 is settled, regrown `DmaPinned` units are committed and locked but **not registered** while any sequence is live: they stay `Underpinned`, so nothing copies from them, and they are registered in the next window with no live sequence. VRAM segments are unaffected, since mapping device memory is not host registration. If pressure returns within 10 minutes of a reacquire, `reacquire_after` doubles, up to 15 minutes, so a machine that keeps needing its memory is not fought over.

#### 11.7 What the user sees

Every rung taken emits `YieldReport { rung, bytes, predicted_dt_tok, measured_dt_tok }` to telemetry. The dashboard shows memory returned and its price in tok/s (RFC-0051). `Ready { degraded }` is set while any rung is held.

### 12. Load, plan delta and teardown sequences

**Per runner state** (RFC-0001 §3.3):

| State | Tier manager work |
|---|---|
| `Fetching` | `plan_arenas`; ledger check against the `LedgerGrant`; reserve host ranges per NUMA domain; start progressive pinning (§6) |
| `Packing` | nothing new; pinning continues |
| `Loading` | measure context reserve; build `Vram` and `Unified` arenas in the §4.4 order; fill `Weights` arenas from the pack (RFC-0021) in pin-priority order; register arenas; publish residency map version 1 |
| `Warming` | hand `BufferView`s of dense groups to the step-program builder (RFC-0012); RFC-0023 fills the expert cache and RFC-0022 builds KV pools inside the arenas; spot-check `W_eff` per tier against the profile; then freeze `Scratch` and stop all allocation |
| `Ready` | serve leases; finish pinning; respond to pressure; apply plan deltas |
| `Draining` | stop admitting; on `persist`, RFC-0022 writes session KV; RFC-0023 writes the frequency profile |
| `Stopped` | process exit releases everything; the daemon checks the ledger is back to zero |

**Plan deltas adopted in place.** Budget-only changes arrive as `ApplyPlanDelta` (RFC-0001 §5.4) and are applied between steps:

| `DeltaItem` | Applied by |
|---|---|
| `ExpertCacheBudget` change on an elastic arena | `ShrinkElastic` / `GrowElastic` (§8.4), then RFC-0023 re-splits per-layer quotas; growth of a host `DmaPinned` arena registers its new units only as §11.6 allows |
| `KvBudget` hot window ≤ the `Kv` arena | RFC-0022 moves the window inside the arena |
| `PrefetchWindow` | RFC-0021 changes ring limits |
| `MissSplit` | RFC-0023 resets its balancer prior |
| `StreamSite::min_tokens` | resets `n*_live`'s prior (§10) |

Anything that changes a non-elastic arena's size, a home tier or a NUMA binding is a new shard and goes through drain-and-replace (RFC-0001 §7.3). In particular a delta may not grow `Scratch` or the KV provisional slots: `Warming` sizes them for the epoch's largest verify window and prefill chunk (`SpeculativeConfig::max_window`, `BatchingLimits`), deltas may lower those limits, and raising them needs a new epoch.

### 13. Telemetry and the residency digest

`ResidencyDigest { version, digest: ResidencySummary }` goes to the daemon at 1 Hz and on any change of level or degradation (RFC-0001 §3.5). It drives routing, KV admission and the dashboard, never the hot path.

```rust
pub struct ResidencySummary {
    pub tiers: SmallVec<[TierUsage; 8]>,          // per tier: budget, mapped, used, pinned, underpinned, by purpose
    pub pinning: PinProgress,                     // done / target bytes, state (Running | Done | GaveUp), ETA
    pub pressure: PressureLevel,
    pub yielded: SmallVec<[YieldReport; 5]>,      // rungs currently held
    pub yield_menu: SmallVec<[YieldOption; 5]>,   // §11.3
    pub kv_headroom: KvHeadroom,                  // RFC-0022 §11: tokens per tier; dashboard copy (admission reads the side channel)
    pub kv_spill: SmallVec<[KvSpillOutcome; 6]>,  // RFC-0022 §5.3: spill path in use per (stage, class, tier)
    pub expert: ExpertCacheDigest,                // RFC-0023: hit rates per tier, h(c) samples
    pub nstar: SmallVec<[(KernelClass, DType, u32); 8]>,   // n*_live (§10)
    pub degraded: Option<Degradation>,
}
pub struct YieldOption { pub rung: u8, pub bytes: Bytes, pub predicted_dt_tok: Nanos }
/// The certification gate's result for one (stage, class, tier), for the dashboard and `syntropy doctor`.
pub struct KvSpillOutcome {
    pub stage: StageIx, pub class: PageClass, pub tier: TierId, pub kind: AttnKind,
    pub planned: SpillCompute,                    // the plan's `spill_compute`
    pub outcome: SpillOutcome,
    pub sequences_affected: u32,                  // live sequences currently under this outcome
    pub since_ms: u64,
}
pub enum SpillOutcome {
    Certified,                                    // the planned path is certified and in use
    Substituted { path: SpillCompute },           // the other path, certified, is in use (performance degradation)
    Refused,                                      // no certified path: no spill to this tier (headroom excludes it)
    UncertifiedApproximate,                       // used only by sequences already Approximate (labelled)
}
```

Metrics (RFC-0001 §13 naming): `syntropy_tier_bytes{tier,purpose,state}`, `syntropy_tier_pinned_bytes{tier}`, `syntropy_tier_pin_step_seconds` (histogram), `syntropy_tier_pressure_level`, `syntropy_tier_yield_bytes_total{rung}`, `syntropy_tier_yield_dt_tok_seconds{rung,kind=predicted|measured}`, `syntropy_tier_vram_segments{device,state}`, `syntropy_tier_nstar_tokens{op_class}`, `syntropy_tier_stale_handle_total`.

### 14. Configuration

Keys are owned here and published in RFC-0054's schema. All are `node` scope unless marked.

| Key | Default | Reload | Meaning |
|---|---|---|---|
| `resource.ram_headroom` | `max(4 GiB, 10 % of RAM)` | hot | RAM always left for the user (§5) |
| `resource.vram_headroom` | 512 MiB on a display GPU, else 256 MiB | hot | VRAM left for other programs (§8) |
| `resource.max_pinned_fraction` (instance) | Linux 0.85, Windows 0.50, WSL2 0.75, macOS unset (no fixed fraction; §5), Tier 3 0 | restart | `ResourcePolicy::max_pinned_fraction` (§5) |
| `resource.memory_yield` (instance) | `UnderPressure` | hot | §11.5 |
| `tier.pin_step` | 1 GiB | restart | largest pinning step; Windows lock-range size (§6) |
| `tier.pin_unit_min` | 256 MiB | restart | smallest registration unit (§6.1) |
| `tier.pin_step_slow` | 250 ms | hot | a step slower than this halves the step |
| `tier.pin_duty` | 0.5 | hot | fraction of wall time pinning may use |
| `tier.pin_psi_max` | 10 % | hot | PSI `some avg10` above which pinning waits |
| `tier.pin_give_up` | 120 s | hot | stop pinning after the gate has failed this long |
| `tier.numa_reserve` | 1 GiB | restart | free memory kept per NUMA node |
| `tier.hugepages` | `auto` | restart | `auto` / `off`: THP advice on Linux host arenas |
| `tier.vram_segment` | 256 MiB | restart | minimum elastic segment (§8.2) |
| `tier.reacquire_after` | 60 s | hot | quiet time before growing back (§8.5, §11.6) |
| `tier.idle_release_after` | 10 min | hot | `Aggressive` only (§11.5) |
| `tier.eip_refit` | 30 s | hot | n* refit period (§10) |
| `tier.eip_halflife` | 60 s | hot | forgetting half-life of the n* fit |
| `tier.wsl_kv_spill` | `false` | restart | allow KV spill to `HostPinned` on WSL2 (§5) |

### 15. Failure handling and error codes

Domain `Tier`. Codes are never reused (RFC-0001 §10.4).

| Code | Class | Raised when | Action |
|---|---|---|---|
| `TIER_LEDGER_EXCEEDED` (1) | `Capacity` | the arena plan exceeds the ledger grant | fail `Fetching`; the coordinator re-plans with the ledger's figures |
| `TIER_VRAM_SHORT` (2) | `Capacity` | §8.1: non-elastic VRAM needs exceed free memory | fail `Loading`; the daemon corrects usable VRAM in the profile; re-plan |
| `TIER_VRAM_EC_REDUCED` (3) | `Drift` | §8.1: expert cache mapped below plan | `Ready { degraded }`; RFC-0001 §7.2 |
| `TIER_HOST_RESERVE` (4) | `Capacity` | `reserve` or `commit` fails (`ENOMEM`, `ERROR_NOT_ENOUGH_MEMORY`) | fail `Fetching` or `Loading`; profile correction; re-plan |
| `TIER_PIN_DEGRADED` (5) | `Drift` | pinning gave up, or registration failed for some ranges | ranges stay `Underpinned`, costed as `HostPageable`; `Ready { degraded }` |
| `TIER_REGISTER_UNSUPPORTED` (6) | `Unsupported` | the provider cannot register host memory | arena becomes `Locked`/`Pageable`; the filter in RFC-0011 stops planning copied misses or streaming on this device |
| `TIER_VMM_UNSUPPORTED` (7) | `Unsupported` | no `map_segment` | fixed slab (§4.1); shrink is logical only |
| `TIER_STALE_HANDLE` (8) | `Internal` | stale handle on a hot path | runner restarts (RFC-0001 §10.1); a counter is kept for off-path stale handles |
| `TIER_PRESSURE_EXHAUSTED` (9) | `Drift` | ladder exhausted and pressure persists | report reduced usable capacity; re-plan if it outlasts `replan.pressure_window` |
| `TIER_NUMA_BIND` (10) | `Transient` | `mbind` or `VirtualAllocExNuma` fails | retried once; then the arena is allocated unbound and the profile marks the domain as unbound, so the solver stops counting on locality |
| `TIER_VRAM_BUDGET_LOST` (11) | `Drift` | another program's use leaves the non-elastic arenas over the OS budget (§8.5) | re-plan |

A runner never reads model bytes from a slot whose state is not `Resident`, so none of these failures can produce wrong output: they cost time, or they stop the runner.

### 16. Per-platform matrix

| | Linux + NVIDIA | Linux + AMD (ROCm) | Windows + NVIDIA | WSL2 + NVIDIA | macOS (Apple silicon) | Strix Halo | GB10-class | Linux/Windows CPU-only | Android/iOS |
|---|---|---|---|---|---|---|---|---|---|
| Fast tier | `Vram` | `Vram` | `Vram` | `Vram` | `Unified` | `Unified` | `Unified` | — | `Unified` or none |
| Host tiers | `HostPinned`, `HostPageable` per NUMA | same | same | same | none | `HostPageable` (CPU) | `HostPageable` (CPU) | `HostPageable` (locked if allowed) | `HostPageable` |
| DMA registration | `cudaHostRegister` per step | `hipHostRegister` | `cudaHostRegister` per lock range; helper GPUs ≤ 8 GiB | `cudaHostRegister` only | — | `hipHostRegister` / Vulkan host import | provider `Unified` alloc | — | — |
| Lock | provider pins; `mlock` for `Locked` if memlock allows | same | working-set floor + `VirtualLock` | as Linux | `mlock` (wire) | `mlock` if allowed | `mlock` if allowed | `mlock`/`VirtualLock` if allowed | none |
| Pin ceiling default | 0.85 × RAM | 0.85 | 0.50 (measured) | 0.75 of VM RAM | min(OS working-set limit, RAM − reserve) | 0.85 | 0.85 | 0.85 (lock only) | 0 |
| Elastic VRAM | CUDA VMM | HIP VMM if supported | CUDA VMM | CUDA VMM if supported in WSL | Metal buffer per segment | HIP VMM / Vulkan memory per segment | CUDA VMM | — | — |
| Pressure source | PSI, cgroup events | same | memory resource notification | PSI inside the VM | dispatch source | per OS | PSI | per OS | host app callbacks |
| VRAM watch | NVML | amdgpu sysfs | DXGI budget | NVML | `currentAllocatedSize` vs working-set cap | driver heap budget | MemAvailable | — | — |

## Performance model and budgets

Notation follows cost model §1. Every number below is an **[estimate]** with its derivation, or a **[target]**. None is a Syntropy measurement.

**Why DMA sources must be pinned.** A copied miss of one DeepSeek-V3 expert at Q4_K_M (≈26.5 MB, cost model §2.4) takes 26.5 MB / 24 GB/s ≈ 1.1 ms from pinned memory and 26.5 MB / 6 GB/s ≈ 4.4 ms from pageable memory on PCIe 4.0 x16 (§2.3) [estimate]. A copied miss from pageable memory would cost more than computing it in place on most desktops, so the balancer must never pick one (§6, invariant 5).

**Why CPU-only groups need not be pinned.** CPU execute-in-place time is `B / W_exec(home)`. `W_exec` differs between anonymous and file-backed memory (42.8 vs ≈19 GB/s in the surveyed Windows case, `tiered-memory-inference.md` §10.1) and not, by any evidence found, between locked and unlocked anonymous memory. The probe measures both host tiers (§2), so this assumption is checked on every machine rather than trusted.

**VRAM is expert capacity.** Each GB of VRAM moved from the KV hot window or headroom to the expert cache holds `1 GB / s_expert` more experts: ≈38 DeepSeek-V3 Q4_K_M experts, or ≈700 of the surveyed 125B model's experts (`tiered-memory-inference.md` §5.2). Its value per token is `Δh_vram · B_routed · (1/W_miss − 1/W_vram)`, where `W_miss` is the in-place CPU rate. That is the term the solver trades against KV (§8.6).

**Segment striping waste.** At most one slot per segment is lost, and segments are at least 8 slots long, so waste is ≤ 12.5% of the elastic arena in the worst case and typically far less, because mixed slot sizes pack the tail [estimate from §8.2].

**n\*.** `n* ≈ F_cpu · b / (2 · W_pcie)`: ≈12 tokens at F_cpu = 1 TFLOPS and ≈24 at 2 TFLOPS, with b = 0.6 and W_pcie = 25 GB/s (cost model §3.2) [estimate]. `n_hide = F_gpu · b / (2 · W_pcie)` ≈ 1980 on an RTX 4090 (SC-10) [estimate]. [Target] SC-6: below `n*_live` no `HostPinned` group is streamed; at or above it every eligible group is. [Target] SC-10: streamed copies run at ≥ 0.90 × the probe's pinned rate.

**What fits in b2.** Scenario b2 of the cost model assumes ≈115 GB of RAM expert cache on a 128 GB machine. With this RFC's defaults the RAM cache is bounded by `host_usable = MemAvailable − 12.8 GB` (≤ 115.2 GB before the OS, desktop and other arenas take anything) on Linux, so the planned figure is an upper bound that a real desktop will undercut by whatever else is running [estimate]. The `DmaPinned` part is bounded by `pin_limit`: ≤ 0.85 × 128 = 108.8 GB on Linux and ≤ 0.50 × 128 = 64 GB on Windows; the in-place part has no pin limit because it lives in `HostPageable` (ADR-0200). Only the copyable share must fit under `pin_limit`, which is why the RAM cache is split.

**Cost of yielding RAM (example).** The same scenario, taking the RAM expert cache as ≈115 GB over an `Nvme` home, `B_routed` = 12.3 GB per token, `W_nvme` = 6 GB/s sustained. Yielding 10 GB at rung 2 under the uniform prior `h(c) = c / pool` costs `Δh = 10 / 394 ≈ 0.025`, so `Δt ≈ 0.025 × 12.3 GB / 6 GB/s ≈ 51 ms` per token: from the SC-5 example's ≈217 ms at h = 0.9 to ≈268 ms, or 4.6 → 3.7 tok/s [estimate]. The uniform prior is not a bound: under skew, a concave hit curve can be steeper than `1/pool` just below the current size, so the true price can be higher or lower than this example. The dashboard shows the price from the learned curve, and the shrink evicts the coldest experts because hot residents are first moved out of the tail segments (§8.4). [Target] predicted Δt_tok within ±25% of the measured value 30 s after the rung.

**Pinning time.** No measured commit-and-register rate per OS was found in the research, so none is assumed here. On a first load the pinning of a shard overlaps its download and pack build. On a restart with the pack present, `DmaPinned` registration is on the path to `Ready` until Open question 9 is settled (§6), so time to ready is `t_load` plus whatever of `t_pin = P / R` (P pinned bytes, R the commit-plus-register rate) does not overlap the reads. For b2 on Linux, P ≈ 104 GB; at an assumed R of 5–10 GB/s that is ≈ 10–21 s with no duty-cycle sleep [estimate; R is Open question 5]. Two rules keep it bounded: before `Ready` no sequence is live, so the loop does not apply the duty cycle (§6), and the dashboard's load estimate and RFC-0061's time-to-ready figure include `t_pin`. SC-23 counts pack and load only, so this term is reported next to it, not hidden in it. [Target] while sequences are live, pinning uses at most `tier.pin_duty` (50%) of wall time; it never takes `available` below `headroom` (SC-11). The M1 spike in Open questions measures R.

**Budgets.** [Target] predicted peak bytes per tier per node within 5% of measured, and zero OOM kills across the test matrix (SC-11). [Target] tier-manager metadata ≤ 32 bytes per slot and per group: DeepSeek-V3's 58 MoE layers × 256 = 14,848 routed experts (≈15.1K with the MTP layer) plus dense groups cost under 1 MB [estimate]. [Target] a steady-state step performs zero allocations and zero locks in `syntropy-tier` (ADR-0012 item 6).

## Alternatives considered

| Alternative | Why not |
|---|---|
| Register every host byte for DMA | Spends the scarcest budget (≈50% of RAM on Windows, the WSL2 cap, `RLIMIT_MEMLOCK` for io_uring) on bytes no device ever copies. CPU-only groups read at the same rate from anonymous memory (ADR-0200). |
| mmap of pack files as the RAM tier | Residency follows the page cache; measured 2.2× slower for a hot pool on Windows; prefetch–release thrash (ADR-0007). Kept only for cold read-once data and `MacMapped`. |
| One large `cudaHostAlloc` per runner | OOM-killed a Linux process with high MemAvailable; fails on Windows; capped at ≈1 GiB on WSL2 (`tiered-memory-inference.md` §10.2). |
| Driver-managed oversubscription (managed memory on discrete GPUs, system-memory fallback) | The driver pages on its own policy, which ADR-0007 forbids, and the surveyed docs report large slowdowns. |
| One contiguous VRAM slab per purpose | Kept as the no-VMM fallback. It cannot return memory without a reload, which fails goal 5 (ADR-0201). |
| Per-layer contiguous slot ranges with tail unmapping | Shrinking would empty the last layers' caches entirely. Striping keeps per-layer hit rates even (§8.3). |
| A global residency lock | Breaks single-writer-per-device (RFC-0001 invariant 4) and puts a lock on the hot path. |
| Leave memory pressure to the OS | Pinned memory cannot be reclaimed; the OS's only remedy is killing the process. |
| Yield in a fixed order with no cost prediction | The user and the daemon cannot choose the cheapest bytes across instances, and the dashboard cannot say what yielding costs (ADR-0202). |
| Windows large pages (`MEM_LARGE_PAGES`) for the host arena | Needs `SeLockMemoryPrivilege`, an administrator policy change. ADR-0200 already removes CPU-only groups from the lock budget, so v1 does not build it; Open questions. |

## Testing and validation

| Claim | Test | Level |
|---|---|---|
| Budget formulas (§5) | fixture snapshots of `/proc/meminfo`, cgroup v1/v2 files, rlimits, `GlobalMemoryStatusEx`, `sysctl` values per OS → expected `usable`, `pin_limit`, `io_fixed` | unit |
| `plan_arenas` is exact | golden `ArenaPlan`s for the cost model's scenarios (a), (b2) and (c); Σ arena bytes = plan bytes per tier; every DMA source is in a `DmaPinned` arena | unit, golden |
| Striping (§8.3) | for random per-layer counts and sizes: no slot crosses a segment; quotas sum to `n_l`; removing the tail segment removes ≤ ⌈n_l / S⌉ slots of each layer; waste ≤ one slot per segment | property |
| Residency invariants (§7.5) | model-based test: random interleavings of admit, complete, cancel, evict, fence advance, counted lease and drop, against a reference model; checks all six invariants after every event | property |
| Handle generations | a handle taken before a free never resolves after it; generation 0 never resolves | property |
| Progressive pinning never crosses headroom | `testkit` fake `OsMemory` scripting available memory, PSI and cgroup changes during pinning; asserts the gate, the give-up path, step halving and duty cycle | simulation |
| Pinning completes on a host full of page cache | fake `OsMemory` with `MemFree` held at a scripted zone watermark (below 1 GiB) before and after every commit and `MemAvailable` high, commits reclaiming cache at a scripted rate: pinning completes and never gives up; with PSI `some avg10` scripted above the threshold it backs off, and gives up only after sustained pressure; the same with PSI absent, using direct-reclaim stall counts and step latency; repeated on the Linux rig after reading a model file larger than RAM | simulation, integration |
| Windows commit charge | the Windows rig with a scripted committer: Syntropy never takes commit charge before the gate allows it, and Moderate fires at 0.90 of the commit limit | integration |
| Registration during decode | the Open question 9 spike: `cuMemHostRegister`/`hipHostRegister` of 64 MiB–1 GiB from a second thread during captured-graph decode; step-time p99 compared with the same run without it | benchmark |
| No DMA crosses a registration | property test over every arena layout: no extent or slot straddles a §6.1 unit; registration count within the §6.1 bound | property |
| Zero OOM kills (SC-11) | Linux CI container with `memory.max` set, a memory hog growing in a sibling process while a model loads and decodes; Windows and macOS soak with a scripted memory hog; WSL2 job | integration, Tier 1 hardware (RFC-0062) |
| Registration accounting | the probe's `VmLck`/`VmPin` check run under a low `RLIMIT_MEMLOCK` on each CI driver version | integration |
| Elastic VRAM keeps graphs valid | shrink and grow the expert cache by k segments during decode with captured graphs; the parity harness passes across the change (ADR-0019); no step exceeds 1.5 × predicted during the change | integration, parity |
| VRAM taken by another program | a CI helper allocates VRAM during decode; the runner shrinks within 5 s and reports | integration |
| n* switch (SC-6, SC-10) | sweep prefill chunk sizes on the CUDA rig; the switch happens within one chunk-size step of the measured crossover; streamed copies ≥ 0.90 × probe rate | benchmark (RFC-0061) |
| Yield cost prediction | apply each rung under a scripted pressure source; predicted Δt_tok within ±25% of measured | benchmark |
| Anonymous pageable reads as fast as pinned for CPU kernels | CPU `W_eff` over `HostPinned`, `HostPageable` and file-backed mmap on every Tier 1 OS | benchmark |
| `syntropy tune` is the only writer of OS limits | syscall trace over the whole macOS test suite: no write to `iogpu.wired_limit_mb` outside `tune`; `tune` then `tune --revert` restores the recorded value; uninstall reverts it and removes the launchd job | integration (macOS rig) |
| Teardown releases everything | kill the runner during pinning, `Loading` and decode; the daemon's ledger returns to zero and the OS reports the memory free | chaos (RFC-0060) |

## Rollout / milestones

Milestones follow RFC-0001 §16.

| Milestone | Lands |
|---|---|
| M0 | `TierManager` façade, `TierShard` per device, `TierDirectory`, `TensorHandle` and residency map with step fences; `Vram` as fixed slabs and `HostPageable` anonymous arenas; CUDA and CPU only; property tests for §7.5 |
| M1 | Pin classes and ADR-0200; per-OS budget formulas for Linux, Windows, WSL2 and macOS; progressive pinning with `HostRegistrar`; CUDA VMM elastic segments and striping; Apple `Unified` arenas with residency sets; VRAM watch; n* refit; pressure ladder rungs 1–4 (rung 1 uses RFC-0022's saved sessions, which also land in M1); residency digest and yield menu; the `syntropy tune` opt-in for the macOS GPU working-set limit (§9.1, with RFC-0051 and RFC-0052); SC-6, SC-10 and SC-11 on the single-device rigs |
| M2 | Several instances per node under one ledger; the daemon's cross-runner yield menu; GB10-class `Unified`; fixed, ggml-owned budgets for the Vulkan preview device (§4.1) |
| M3 | HIP VMM and Vulkan segment arenas with the M3 providers; Strix Halo `Unified`; rung 4 for replicas (with cross-node EP); `Aggressive` idle release |
| M4 | Android and iOS `HostPageable` adapters (Tier 3); the Windows large-page decision (Open questions); every SC row above on every Tier 1 platform |

## Open questions

1. **Is provider host registration charged to `RLIMIT_MEMLOCK`?** Not documented for current NVIDIA or AMD drivers; an old forum report says `cudaMallocHost` ignored `ulimit -l`. The probe's `VmLck`/`VmPin` check (§5) answers it per machine. A spike in M1 on every CI driver decides what `syntropy doctor` tells users about `LimitMEMLOCK`.
2. **Windows lock ceiling and large pages** (RFC-0001 §17 item 4). Answer for v1: no dedicated large-page or locked-section path. ADR-0200 and the split RAM expert cache (§2) take CPU-only bytes, including the in-place part of the RAM cache, out of the lock budget, so the ≈50% ceiling binds only the copyable share and staging. An M1 benchmark on a 128 GB Windows rig measures how much decode b2 loses because the copyable share is capped at 64 GB; only if the loss is material is an opt-in `MEM_LARGE_PAGES` path (requiring "Lock pages in memory") built in M4.
3. **Resolved: the macOS cap** (ADR-0007 as amended). No fixed percentage; the cap is the lower of the OS working-set limit and RAM minus the reserve, and raising the OS limit is the `syntropy tune` opt-in of §9.1. What remains open is evidence: an M1 soak on a ≥ 128 GB Mac with the wired arena at the new cap, under a scripted memory hog, must show no compressor panics and no OOM kills before 1.0.
4. **WSL2 KV spill. Answered 2026-10-20 (OQ-IO-13):** keep it off by default under WSL2 (ADR-0020). WSL2 caps pinned memory near 1 GiB, so spilled KV would compete with staging for that cap. TIER-E2-T11's measurement of pinned H2D inside WSL2 against native Windows is kept as evidence for a later revisit and no longer decides the default.
5. **Pinning throughput.** Commit-plus-register GB/s per OS is unmeasured. An M1 spike on the Linux, Windows and macOS rigs measures it at 64 MiB, 256 MiB and 1 GiB steps and sets `tier.pin_step` defaults.
6. **Elastic KV arena.** The `Kv` arena is fixed per epoch (§8.6), so a plan delta can grow the expert cache at KV's expense but not the reverse without a drain. If RFC-0011's deltas often want the reverse, make `Kv` elastic by migrating tail pages to `HostPinned` first.
7. **Rung 5 value.** Unlocking `HostPageable` homes at Critical may just move the failure from an OOM kill to a machine that pages heavily. M1 chaos runs compare it with an immediate re-plan.
8. **HIP VMM on consumer RDNA** and **Strix Halo carve-out against GTT** bandwidth: both need M3 measurements on the Tier 2 rigs.
9. **Host registration during decode**. Does `cuMemHostRegister` or `hipHostRegister` from the housekeeping thread delay captured-graph decode on the dispatcher? Until an M1 spike measures step-time p99 with 64 MiB–1 GiB registrations during decode, all `DmaPinned` registration finishes before `Ready` (§6). If there is no effect, registration after `Ready` is allowed; if there is, it stays before `Ready`, or runs between steps in slices that fit the step's slack.

## Proposed amendments to RFC-0001

1. **§9.1 and §9.2, `HostPinned` and `HostPageable` (meaning clarified, nothing renamed).** `HostPinned` is memory registered for DMA. Groups that only the CPU computes are homed in `HostPageable`, which is anonymous memory, never file-backed, and locked when the budget allows. The §9.1 row "`HostPageable`: heap, or mmap of pack extents" should read "anonymous memory, locked when the budget allows; mmap only for cold, read-once data". ADR-0200 records the decision; the glossary's "Pinned memory" entry should say that locking for CPU-only bytes is optional.
2. **§5.7 `KernelProvider`.** RFC-0040 should add `map_segment`, `unmap_segment` and `host_registrar`, each with a default body returning `Unsupported` or `None` (§4.1, §6). These are additions the owner may make; they are listed here because RFC-0020 depends on them from M1.
3. **§10.3 memory-pressure ladder.** Add rung 5, at Critical only: unlock `Locked` `HostPageable` homes so the OS can page them instead of killing the runner. Add the yield menu (§11.3), so the daemon fills `reclaim` from the cheapest bytes across runners rather than asking every runner for a share.
4. **§5.2 host tiers per NUMA domain.** State that each NUMA domain contributes one `HostPinned` and one `HostPageable` tier, matching "one CPU device per NUMA domain", so the solver can keep CPU execute-in-place NUMA-local. RFC-0010 should enumerate them that way.
5. **§9.2 macOS row.** State ADR-0007's amended rule: the cap is the lower of the OS-reported GPU working-set limit and RAM minus the RFC-0020 reserve, and `syntropy tune` is the only way to raise the OS limit.
6. **§9.3 expert-cache shrink.** RFC-0001 says the shrink "is applied as a PlanDelta". RFC-0020 also shrinks locally, from the pressure ladder and the VRAM watch (§8.5, §11), reporting the smaller ledger afterwards; a `PlanDelta` remains the path for planned budget changes. Record both.
7. **§9.2 WSL2 row.** "never stream KV" becomes "KV spill to `HostPinned` off by default; `tier.wsl_kv_spill` opts in after measurement" (§5).
8. **ADR-0007 item 4** (to be recorded as an ADR note). This RFC requires `DmaPinned` registration to finish before `Ready` until Open question 9 is settled (§6); locking without the GPU driver may continue after `Ready`.
9. **ADR-0007 item 10.** Yielding memory requests a re-plan only when the ladder is exhausted or pressure outlasts `replan.pressure_window`, not on every yield (§11.4), matching RFC-0001 §10.3.
10. **§11.3 `ResourcePolicy::max_pinned_fraction`.** Type it `Option<f32>`, so that "unset" (the macOS default, §5) is expressible.
11. **§5.2 resource graph, host tiers.** `HostPinned` and `HostPageable` of one NUMA domain share one RAM pool: the resource graph should carry that pool and its `host_usable`, so every consumer applies the joint row of §5.
