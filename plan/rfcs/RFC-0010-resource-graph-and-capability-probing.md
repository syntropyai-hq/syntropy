# RFC-0010: Resource graph and capability probing

- Status: Draft
- Area: placement and scheduling
- Depends on: RFC-0001, RFC-0020 (§2, §5, §9), RFC-0021 (§10, §11), RFC-0023 (§13), ADR-0100, ADR-0006, ADR-0007, ADR-0008, ADR-0011, ADR-0012, ADR-0013, ADR-0014, ADR-0015, ADR-0020
- Date: 2026-10-19
- Revised: 2026-10-20: link mean, hop histogram and periodic-stall flag for wireless hops (§6.2); battery presence for the solver's departure margin (§1.3); a read-only direct-path report (§1.3); open questions 4 (FileVault part) and 5 answered; 2026-10-21: Linux E1 sources as built and the system-call split to PLC-E1-T14 (§2.2 revision note)

## Summary

This RFC specifies the `syntropy-probe` crate and the `ResourceGraph` it feeds: how a node enumerates its devices, tiers and links on every supported OS, which microbenchmarks it runs at startup and in what order, how long each may take, how results are cached across restarts, how live timings from runners refine the numbers while the node serves, and when a measured change is large enough to trigger re-placement.

The design rests on one rule from the cost model: every number the solver uses is measured on the machine, with the kernel that will run and at the request size that will be issued (cost model §0 item 10, §5.2). Spec sheets are sanity bounds only. Enumeration that needs no driver context runs in the daemon; anything that opens a GPU context runs in a probe-mode runner (RFC-0001 §3.1, invariant 3). The startup suite targets 15 s per node (SC-22). The sustained, thermally throttled NVMe read and the device-path collective latency that gates cross-node tensor parallelism run later, in the background or on demand, and the solver treats their absence conservatively.

It extends RFC-0001 §5.1 and §5.2 (`DeviceId`, `TierId`, `LinkId`, `Measured`, `ResourceGraph`, `CapabilityProfile`), which this RFC owns, without renaming anything.

## Motivation and goals / non-goals

Placement quality dominates home-cluster performance: a cost-based layer-window solver ran a 70B model 3.3× faster than a memory-proportional split (`multi-device-inference.md` §3.3). A solver is only as good as its inputs, and the inputs vary more than any spec table suggests:
- achieved bandwidth is 0.43–0.79 of peak on GPU-class engines and 0.24–0.35 for CPU-hosted experts end to end (cost model §2.2);
- a single-CCD Ryzen reads DRAM no faster than its ≈64–70 GB/s fabric link, whatever DIMMs are fitted (§2.1);
- pinned versus pageable host-to-device copies differ by about 4× on the same PCIe 4.0 link (§2.3);
- a Gen5 NVMe drive without a heatsink throttles within about a dozen seconds of heavy load (§2.4);
- Wi-Fi's median hop is 3–7 ms but its p99 is ≈250 ms (§2.5);
- the effective cost of a collective on RDMA over Thunderbolt 5 was back-solved at ≈130–150 µs against a 10–21 µs wire latency (§3.8).

The surveyed systems failed here in specific ways: VRAM never queried and GPU shards sized against system RAM, made-up TFLOPS from VRAM size × clock, disk profiling by shelling out to an external benchmark tool, and link latency never measured at all (`multi-device-inference.md` §3.8, §13 items 5, 18, 19).

**Goals**
1. Enumerate every device, tier and link the solver may use, on Linux, Windows, macOS and the Tier 2/3 platforms, with no external tools and no elevated privileges.
2. Measure each first-order cost-model input (cost model §5.1 items 1–7) with the real kernel and request size: W_eff per (device, tier, kernel class, quant scheme), the batch curve that gives n* and verify-window cost, F_eff per (op class, dtype), t_fixed, t_switch, pinned and pageable PCIe, sustained throttled NVMe at expert extent sizes, link α/β/p99, and α_eff through the device-to-device path.
3. Finish the startup suite within 15 s per node (target, SC-22), all nodes in parallel.
4. Reuse a cached profile on restart after a ≈3 s confirmation (target), and re-probe fully only when hardware, drivers, build or power state change.
5. Refine profiles online from live step timings at ≤ 0.5% of step time on average (target), and detect drift that should trigger re-placement within `replan.drift_window` of its onset.
6. Give the solver an honest picture: every value carries spread, sample count, age and provenance (`Measured<T>`), and every unmeasured input is flagged in the plan's `assumptions`.

**Non-goals**
- Choosing kernels for a plan's exact shapes. That happens in `Warming` and belongs to RFC-0040 (RFC-0001 §5.7); this RFC consumes its timings as online samples.
- Learning expert hit-rate curves and speculative acceptance rates. Those are workload statistics owned by RFC-0023 and RFC-0012; this RFC only routes their drift signal.
- Transport selection and connection management (RFC-0031) and membership (RFC-0030). This RFC measures the links those RFCs set up.
- Pinned-budget policy (RFC-0020). This RFC reports the raw OS limits; RFC-0020's rule turns them into `TierInfo::usable`.

## Design

### 1. ResourceGraph data model

#### 1.1 What is fixed by RFC-0001

`ResourceGraph { version, nodes, devices, tiers, links, profiles, reservations }`, `NodeInfo`, `DeviceInfo`, `TierInfo`, `LinkInfo`, `CapabilityProfile`, `DeviceCaps`, `StorageCaps`, `LinkCaps` and `Measured<T>` keep their RFC-0001 names, fields and meaning. Everything below is an addition: new fields carry `#[serde(default)]`, new enum variants go at the end.

#### 1.2 Topology records (enumeration output)

```rust
/// RFC-0001 declares DeviceClass as a PackKey component. It is the pair that decides which
/// kernels and pack layouts apply. Two devices with equal DeviceClass share packs.
pub enum DeviceClass {
    Cuda { sm: u16 },                          // 75, 86, 89, 90, 100, 120, 121 ...
    Hip { gfx: u32 },                          // 0x1100, 0x1201, 0x942 ...
    Metal { family: AppleGpuFamily, cores: u16 },
    Vulkan { vendor: u32, device: u32, driver: VkDriverId },
    Cpu { arch: CpuArch, level: IsaLevel },    // x86-64-v2/v3/v4, aarch64-v8.0/v8.2-dotprod/v9-sve2
    Npu { vendor: u32, model: u32 },
}

/// Added to DeviceInfo.
pub struct DeviceInfoExt {
    pub driver: DriverInfo,                     // vendor driver version, runtime version, loader path
    pub pci: Option<PciInfo>,                   // None for Apple GPUs, CPU devices, most NPUs
    pub uuid: Option<[u8; 16]>,                 // NVML/HIP/Vulkan device UUID; dedupe key across providers
    pub luid: Option<u64>,                      // Windows adapter LUID (DXGI ↔ CUDA ↔ Vulkan match)
    pub cpu: Option<CpuTopology>,               // CPU devices only
    pub power: Option<PowerInfo>,               // power limit, current cap, thermal state
    pub problems: SmallVec<[DeviceProblem; 2]>, // why a present device is unusable or degraded
}

pub struct PciInfo {
    pub bdf: PciAddress,                        // domain:bus:device.function
    pub vendor: u16, pub device: u16, pub subsystem: u32,
    pub link_cur: PcieLink, pub link_max: PcieLink, // gen + width; cur is read at idle and under load
    pub upstream: PcieUpstream,                 // CpuRootPort | Chipset | Switch | ThunderboltTunnel | Unknown
    pub resizable_bar: Option<bool>,            // BAR1 or CPU-visible VRAM equals VRAM size
    pub numa: Option<u16>,
}
pub struct PcieLink { pub gen: u8, pub width: u8 }

pub struct CpuTopology {
    pub numa: u16,
    pub physical_cores: u16,
    pub logical_cpus: u16,
    pub perf_cores: u16,                        // hybrid parts: P-cores; homogeneous parts: all cores
    pub efficiency_cores: u16,
    pub llc: Bytes,                             // last-level cache per domain, summed over CCDs
    pub ccds: u8,                               // core complexes sharing one fabric link (0 = unknown)
    pub isa: IsaFlags,                          // AVX2, AVX-512 F/BW/VL/VNNI/BF16/FP16, AMX-BF16/INT8,
                                                // NEON dotprod/i8mm, SVE2, SME2
    pub amx_permitted: bool,                    // arch_prctl(ARCH_REQ_XCOMP_PERM) granted
    pub reserved_cpus: CpuSet,                  // cores ADR-0012 reserves: dispatchers, I/O, net, user
}

pub enum DeviceProblem {
    DriverTooOld { have: String, need: String },
    NoProvider,                                 // enumerated, but no KernelProvider drives it (e.g. most NPUs in v1)
    ProbeCrashed { suite: ProbeStep, class: ErrorClass },
    LinkDegraded { cur: PcieLink, max: PcieLink }, // e.g. x16 card training at x4
    BelowSpec { quantity: SpecQuantity, measured: f64, spec: f64 }, // < 0.5 × spec: misconfiguration flag
    Excluded { by: ExclusionSource },           // user config, or features.disable
}

/// Added to TierInfo.
pub struct TierInfoExt {
    pub numa: Option<u16>,
    /// Where `usable` came from, so the dashboard can explain a low number.
    pub budget_source: BudgetSource,
    /// Raw OS inputs for HostPinned and HostPageable tiers. RFC-0020's pure rule
    /// `budget::host_usable(&HostMemInfo, &ResourcePolicy)` turns them into `usable`.
    pub host: Option<HostMemInfo>,
    /// Context reserve measured at device open (RFC-0001 §9.3 item 1). Vram and Unified only.
    pub context_reserve: Option<Bytes>,
}
pub enum BudgetSource {
    AllocationProbe,                            // discrete VRAM: confirmed by reserve-and-map
    DxgiBudget,                                 // IDXGIAdapter3::QueryVideoMemoryInfo
    VkMemoryBudget,                             // VK_EXT_memory_budget heapBudget
    MetalWorkingSet,                            // recommendedMaxWorkingSetSize
    MemAvailableMinus { reserve: Bytes },       // GB10-class: MemAvailable − 6 GiB (ADR-0014)
    HostRules,                                  // RFC-0020 rule over HostMemInfo
    Statfs,                                     // Nvme: free bytes on the volume, capped by store.max_bytes
}
pub struct HostMemInfo {
    pub total: Bytes,
    pub available: Bytes,                       // MemAvailable / ullAvailPhys / vm_stat free+inactive+purgeable
    pub numa_free: Option<Bytes>,               // per-NUMA free for this tier's domain
    pub cgroup_limit: Option<Bytes>,            // cgroup v2 memory.max (v1 memory.limit_in_bytes) − current
    pub memlock_limit: Option<Bytes>,           // RLIMIT_MEMLOCK; None = unlimited or CAP_IPC_LOCK held
    pub windows_lockable: Option<Bytes>,        // working-set ceiling obtainable without privilege
    pub wsl2: bool,                             // /dev/dxg present: cudaHostAlloc cap applies (RFC-0001 §9.2)
    pub macos_wired_limit: Option<Bytes>,       // iogpu.wired_limit_mb if set, else None (default cap)
    pub large_pages: Option<Bytes>,             // huge/large page size usable without privilege
    /// Lock accounting measured by step H5 (§3.2), the inputs of RFC-0020 §5's `lockable` and
    /// `io_fixed` formulas.
    pub lock: Option<LockAccounting>,
}
pub struct LockAccounting {
    pub vmlck_before: Bytes, pub vmpin_before: Bytes,  // /proc/self/status before the test registrations
    /// Linux: whether a 64 MiB `register_host` raised VmLck or VmPin, i.e. whether provider host
    /// registration is charged to RLIMIT_MEMLOCK on this driver. None where not testable.
    pub registration_charged: Option<bool>,
    /// Linux: whether a 64 MiB io_uring fixed-buffer registration raised VmPin (RFC-0021 IO_FIXED).
    pub io_fixed_charged: Option<bool>,
    /// Windows: the largest lock-and-register total reached in 1 GiB ranges up to
    /// max_pinned_fraction before the first failure (RFC-0020 §5 `lockable_measured`).
    pub lockable_measured: Option<Bytes>,
    pub driver: String,                         // the driver version the result is valid for
}

/// Replaces nothing: RFC-0001's VolumeInfo is named but has no fields. RFC-0010 defines them.
pub struct VolumeInfo {
    pub mount: PathBuf,                         // the store directory's volume
    pub fs: FsKind,                             // ext4 | xfs | btrfs | zfs | ntfs | refs | apfs | other
    pub bus: StorageBus,                        // Nvme | Sata | Usb | Virtual | Unknown
    pub pci: Option<PciInfo>,                   // controller; `upstream` says CPU root port or chipset
    pub shared_lane: bool,                      // shares an uplink with other storage or NICs (AM5 chipset)
    pub direct_io: DirectIoSupport,             // Supported { mem_align, offset_align } | Unsupported
    pub logical_block: u32, pub physical_block: u32,
    pub encrypted: Option<bool>,                // FileVault, BitLocker, dm-crypt when detectable
    pub free: Bytes,
    pub temperature_source: TempSource,         // Hwmon | StorageQuery | None (macOS has no public interface)
    pub model: String, pub firmware: String,
}
```

#### 1.3 Capability records (measurement output)

```rust
/// RFC-0001 MeasureSource gains one variant. Derived values are scaled from a measured
/// neighbour (another quant scheme on the same device and tier, by bytes-per-weight); they
/// are better than Prior and are still listed in PlacementPlan::assumptions.
pub enum MeasureSource { Startup, Online, Prior, Derived { from: DerivedFrom } }

/// Added to CapabilityProfile.
pub struct CapabilityProfileExt {
    pub key: ProfileKey,                        // cache key (§5)
    pub status: ProfileStatus,                  // Quick | Full | FullWithSustained
    pub power: PowerSource,                     // the power state this profile was measured in
    /// The host has a battery (Linux power_supply `type` Battery, Windows `GetSystemPowerStatus`
    /// BatteryFlag ≠ 128, IOKit `IOPSCopyPowerSourcesInfo`, mobile always): a laptop-class node,
    /// departure-prone for RFC-0011 §4's margin whether or not it is on mains now.
    pub battery_present: bool,
    pub suite: ProbeSuiteVersion,               // bumped when a probe's method changes
    pub inventory: Option<NodeInventory>,       // set only when devices, tiers or links changed
    /// Origin download rate from completed origin fetches (§6.1); no active WAN probe runs.
    /// RFC-0011 turns it into the plan's WAN rate class.
    pub wan_down: Option<Measured<BytesPerSec>>,
}

/// Added to DeviceCaps.
pub struct DeviceCapsExt {
    /// W_eff per (tier, kernel class, quant scheme) with its batch curve. RFC-0001's
    /// `tier_bw` keeps one entry per (tier, class): the value for the scheme with the most
    /// bytes in the probe set, so readers that ignore schemes still get a measured number.
    pub quant_bw: Vec<QuantBw>,
    /// CPU↔GPU activation hand-off through host-mapped memory and a doorbell: one direction,
    /// 16 KiB payload, device-side wait. Cost model §3.2 `t_switch`.
    pub t_switch: Option<Measured<Nanos>>,
    /// Launch cost of one small kernel outside a captured graph, and per node inside one.
    pub launch: Option<LaunchCaps>,
    /// Measured in the presence of concurrent PCIe DMA reading the same DRAM: the CPU pool's
    /// slowdown per GB/s of DMA. The `d` coefficient of RFC-0023's three-site balance.
    pub dma_interference: Option<Measured<f32>>,
    /// Doorbell-to-running latency through the backend's device-triggered wake (H2 variant,
    /// RFC-0012 §3.3); sets `exec.inflight_park_threshold`.
    pub wake: Option<Measured<Nanos>>,
    /// Per host wait on backends without whole-step capture of host waits (M5, RFC-0012 §7.5).
    pub host_wait: Option<Measured<Nanos>>,
}

pub struct QuantBw {
    pub tier: TierId,
    pub class: KernelClass,
    pub scheme: QuantScheme,
    /// Weight bytes consumed per second at n tokens per pass, for n in {1, 2, 4, 8, 16, 32}.
    /// Point 0 (n = 1) is W_eff. The curve flattens into B/W while memory-bound and turns
    /// into 2·P·n/F once compute-bound, so the solver reads n* and t_verify(T) off it
    /// instead of assuming a roofline (cost model §3.1, §3.2, §3.10).
    pub curve: SmallVec<[(u16, Measured<BytesPerSec>); 6]>,
    pub extent: Bytes,                          // buffer size used: > 2 × LLC, ≥ 1 GiB
    /// Unified devices only: n = 1 rate while the I/O engine reads at the volume's S1 rate
    /// (RFC-0020 §9). The solver uses it for plans that stream from NVMe on that device.
    pub contended: Option<Measured<BytesPerSec>>,
}
pub struct LaunchCaps { pub eager: Measured<Nanos>, pub captured_node: Option<Measured<Nanos>> }

/// RFC-0001 names HostLinkCaps; RFC-0010 defines it.
pub struct HostLinkCaps {
    pub link: LinkId,                           // the Pcie link (or Thunderbolt tunnel)
    pub h2d_pinned: Measured<BytesPerSec>,      // 64 MiB copies from the registered arena
    pub d2h_pinned: Measured<BytesPerSec>,
    pub h2d_pageable: Measured<BytesPerSec>,    // for diagnosis only; streaming never uses it
    pub h2d_small: Measured<Nanos>,             // 16 KiB copy latency, submit to completion event
    pub bidirectional: Option<Measured<BytesPerSec>>, // H2D and D2H at once, summed
    pub p2p: Vec<(DeviceId, Measured<BytesPerSec>)>,  // verified peer copies (ADR-0006 item 5)
}

/// Added to StorageCaps. `sustained_read`, `burst_read`, `extent`, `queue_depth`,
/// `small_read` and `temperature_c` keep their RFC-0001 meaning.
pub struct StorageCapsExt {
    // Fields RFC-0021 §10 defines and relies on, recorded by the S1 engine selection:
    pub engine: EngineKind,                     // the engine S1 chose (RFC-0021 §4, §5, §7)
    pub chunk: Bytes,                           // best of 1, 2, 4, 8 MiB at qd_target
    pub dio: DioAlign,                          // memory and offset alignment for direct I/O
    pub buffered: bool,                         // RFC-0021 §4.2 flagged (non-direct) mode
    pub sqpoll: bool,                           // SQPOLL pays on this volume (RFC-0021 §4.1)
    pub read_around_efficiency: Option<f32>,    // RFC-0021 §6
    pub shared_lanes: bool,                     // copied from VolumeInfo::shared_lane (cost model §2.4)
    pub qd_target: u16,                         // smallest QD reaching ≥ 95% of the best rate at the
                                                // dominant request size (RFC-0021 §3.2); = queue_depth
    /// Read-only report of direct-to-GPU paths per GPU (RFC-0021 §9; none is built in v1):
    /// cuFile loads / opens / P2P or compatibility mode; DirectStorage runtime present.
    pub direct_paths: SmallVec<[(DeviceId, DirectPathReport); 2]>,
    // RFC-0010's own fields:
    /// The measured grid: engine × size × queue depth → throughput and latency spread.
    pub grid: Vec<IoPoint>,
    pub sustained: SustainedStatus,             // Pending | Running { since_ms } | Done
    pub throttle: Option<ThrottleInfo>,
    pub engines_blocked: SmallVec<[(EngineKind, BlockReason); 2]>, // e.g. io_uring under seccomp
}
pub struct IoPoint {
    pub engine: EngineKind, pub size: Bytes, pub qd: u16,
    pub cpu_per_gb: Option<Nanos>,              // I/O-thread CPU time per GB (RFC-0021 §10 tie-break)
    pub rate: Measured<BytesPerSec>,
    pub latency: Measured<Nanos>,               // per request
}
pub struct ThrottleInfo {
    pub onset_after: Nanos,                     // time under load before the rate fell
    pub ratio: f32,                             // sustained / burst
    pub temp_at_onset_c: Option<f32>,
}

/// Added to LinkCaps. One LinkCaps exists per (link, transport) pair.
pub struct LinkCapsExt {
    /// Ping-pong grid: 64 B, 16 KiB, 1 MiB, 64 MiB. α is fitted from the small sizes, β from the large.
    pub grid: SmallVec<[(Bytes, Measured<Nanos>); 4]>,
    pub p999: Option<Nanos>,                    // from the online histogram once it has ≥ 10,000 samples
    /// Mean one-way hop time at the decode frame size, from the same histogram: the value the
    /// solver costs `WiFi` and `Overlay` hops at (RFC-0011 §5).
    pub mean: Option<Measured<Nanos>>,
    /// The histogram itself, compacted (log buckets, ratio 1.02), so the solver can read
    /// P(t_hop > mean + x) for C14's stall rate (RFC-0011 §6).
    pub hop_hist: Option<HopHistogram>,
    /// Hop-time excess shows a periodic pattern (autocorrelation peak at a lag near 1 s), as the
    /// macOS peer-to-peer radio's off-channel scans produce; `doctor` names the cause.
    pub periodic_stall: bool,
    /// α_eff provenance: which buffers, which sync method, how many samples.
    pub alpha_eff_path: Option<DevicePath>,     // { src: DeviceId, dst: DeviceId, payload: Bytes, sync: SyncKind }
    /// `LinkCaps::alpha_eff` is the full one-way time of a collective-sized message through the real
    /// device path: wire latency, synchronisation on both ends, AND the transfer time S/β at the
    /// partial size each member sends (S/N for an all-reduce step). It is not ping latency.
    pub alpha_eff_includes_transfer: bool,       // always true; recorded so readers need not guess
    pub medium: NetMedium,                      // duplicated from LinkId::kind for quick filtering
}
/// Compacted hop-time histogram for one (link, transport): the live estimator's log buckets
/// (ratio 1.02) over `probe.link.window`, merged with the decayed all-time histogram (§6.2).
pub struct HopHistogram { pub first: Nanos, pub ratio: f32, pub counts: Vec<u32>, pub samples: u64 }
```

#### 1.4 Graph assembly and versioning

The coordinator builds the `ResourceGraph` from the event log; no node builds its own view.
- `NodeAdmitted { info }` adds a `NodeInfo` and the node's `DeviceInfo`, `TierInfo` and `LinkInfo` records, carried in a field this RFC adds: `NodeInfo::inventory: NodeInventory { devices, tiers, links }`. RFC-0001 gives no other path for device and tier records to reach the graph (amendment A1).
- `ReportProfile { profile }` replaces that node's `CapabilityProfile`. The coordinator accepts it only if `profile.version` is newer, and then emits `GraphUpdated`. A hardware change after admission (an eGPU unplugged, a volume unmounted) rides the same command in `CapabilityProfileExt::inventory: Option<NodeInventory>`.
- `ReportLedger` changes `reservations`.

`GraphDelta`, carried by `GraphUpdated`, is defined here:

```rust
pub struct GraphDelta {
    pub nodes_added: Vec<NodeId>,
    pub nodes_removed: Vec<NodeId>,
    pub profiles: Vec<(NodeId, u64 /* profile version */, Blake3 /* profile digest */)>,
    pub ledgers: Vec<(NodeId, LedgerSummary)>,
    pub links: Vec<(LinkId, LinkChange)>,       // Added | Removed | TransportsChanged
    pub cause: GraphChangeCause,                // Membership | StartupProfile | ModelProbe | Drift | Ledger
}
```

`GraphVersion` increases by one per `GraphUpdated`. The solver records the version it solved against in `PlacementPlan::graph`, so a re-plan can tell which inputs changed since the last commit. Profiles travel inline in `ReportProfile`: a profile is a few tens of KiB (estimate: 4 devices × ≈60 `QuantBw` entries × ≈100 B, plus 16 links × 3 transports × ≈200 B ≈ 34 KiB), small enough for the control plane at its rate of a few per hour.

**Ordinals.** `DeviceId::ordinal` follows RFC-0001 §5.1: discrete GPUs by PCI address, then integrated GPUs, then one CPU device per NUMA domain in node order, then NPUs. `TierId::ordinal` is assigned in this order: one `Vram` per discrete GPU in device order; one `Unified` per unified pool; one `HostPinned` and one `HostPageable` per NUMA domain; one `Nvme` per volume that holds the store or a configured extra tier path. A hardware change alters `DeviceInfo::fingerprint`, which invalidates the profile (§5); ordinals are recomputed and the graph changes, as it must.

**Co-location.** `NodeInfo::host` is BLAKE3 over the sorted set of GPU UUIDs, the permanent MAC addresses of physical NICs, and the CPU brand string. Two daemons on one host compute the same value. The claim is confirmed before the solver uses a shared-memory link between them: the two runners exchange a nonce through a named shared-memory segment (RFC-0031 owns the segment protocol). An unconfirmed claim is treated as two hosts.


### 2. Hardware enumeration per OS

#### 2.1 Who enumerates what

| Step | Process | Needs a driver context | Output |
|---|---|---|---|
| E1 Host facts: OS, CPU topology and ISA, RAM, NUMA, cgroups, power source, WSL2 or container detection | daemon | no | `NodeInfo`, CPU `DeviceInfo`, host `TierInfo` |
| E2 GPU inventory from management APIs: NVML, AMD SMI or amdgpu sysfs, DXGI/DXCore, IOKit | probe runner (revised 2026-10-22) | no | provisional GPU `DeviceInfo` with PCI, UUID/LUID, VRAM total and OS budget |
| E3 Storage volumes for the store and extra tier paths | daemon | no | `Nvme` `TierInfo` with `VolumeInfo` |
| E4 Network interfaces and RDMA devices | daemon (`net` crate) | no | NIC endpoints; `LinkInfo` records once peers are known |
| E5 Provider enumeration: `KernelProvider::enumerate` for every provider in the build, then `open` per device | probe runner | yes | `DeviceClass`, `features`, `backends`, context reserve, confirmed VRAM |

The daemon starts E1–E4 at once and spawns the probe runner (`syntropy __runner --probe`) in parallel. The runner's `ProbeResult` is merged with E2 by UUID, LUID or PCI address. A GPU that E2 lists but no provider opens keeps its `DeviceInfo` with `DeviceProblem::NoProvider` or `DriverTooOld`, so `syntropy doctor` can say why it is idle; the solver filters it out.

*Note 2026-10-22 (OQ-PRC-26, RFC-0001 §4.2 rule 8):* E2 runs in the probe runner, not the daemon. Its management APIs load vendor libraries and need `unsafe`, which the daemon may not do (RFC-0001 §4.2 rules 4 and 8). The probe runner calls them through `kernels-loader`, which rule 6 lets load them, and returns the records with its `ProbeResult`; the daemon merges them as above. E1's and E3's system calls that need `unsafe` go in `io`.

Vendor libraries are loaded with `dlopen`/`LoadLibraryExW` from the driver install (`libnvidia-ml.so.1`, `nvml.dll`, `libamd_smi.so`, `libcuda.so.1`, `nvcuda.dll`, `libamdhip64`, the Vulkan loader). A missing library means "no such device", never an error (RFC-0001 §4.2 rule 7).

#### 2.2 Per-OS sources

| What | Linux | Windows | macOS | Android / iOS (Tier 3) |
|---|---|---|---|---|
| CPU topology, hybrid cores, caches | `/sys/devices/system/cpu/*/topology`, `cache/index*`, CPUID leaf 0x1A (Intel hybrid core type), `cpu_capacity` (Arm big.LITTLE) | `GetLogicalProcessorInformationEx` (`RelationProcessorCore` with `EfficiencyClass`, `RelationCache`, `RelationNumaNode`) | `sysctl hw.perflevel{0,1}.physicalcpu`, `hw.l2cachesize`, `machdep.cpu.brand_string` | `/sys/devices/system/cpu`, `cpu_capacity`; iOS `sysctl hw.perflevel*` |
| ISA | CPUID; AMX needs `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`; Arm `getauxval(AT_HWCAP/AT_HWCAP2)` | CPUID; `IsProcessorFeaturePresent` on Arm64 | `sysctl hw.optional.arm.FEAT_*` (SME2, I8MM, BF16) | `getauxval`; iOS `sysctl hw.optional.arm.*` |
| RAM, available | `/proc/meminfo` `MemTotal`, `MemAvailable`; per node `/sys/devices/system/node/node*/meminfo` | `GlobalMemoryStatusEx`; `GetNumaAvailableMemoryNodeEx` | `sysctl hw.memsize`; `host_statistics64(HOST_VM_INFO64)` free + inactive + purgeable | `/proc/meminfo`; iOS `os_proc_available_memory()` |
| Locking limits | `getrlimit(RLIMIT_MEMLOCK)`, `capget` for `CAP_IPC_LOCK`, `VmLck`/`VmPin` from `/proc/self/status` (with the H5 test registrations), cgroup v2 `memory.max`/`memory.current`/`memory.high` (v1 `memory.limit_in_bytes`) | `GetProcessWorkingSetSizeEx`; the lock-and-register ceiling is measured by step H5 (§3.2) | wired limit: `sysctl iogpu.wired_limit_mb` read only, never written without opt-in (ADR-0014) | none: `HostPageable` only (RFC-0001 §9.2) |
| NUMA | `/sys/devices/system/node/node*/cpulist`, device `numa_node` in sysfs | `GetNumaNodeProcessorMaskEx`; SetupAPI `DEVPKEY_Numa_Node` for PCI devices | single domain | single domain |
| Environment | WSL2: `/dev/dxg` present; container: cgroup namespace, `/.dockerenv`; io_uring allowed: `io_uring_setup` succeeds and `/proc/sys/kernel/io_uring_disabled` is 0 | Windows build (IoRing needs 22000+) | macOS version (Metal 4 needs 26, RDMA over TB5 needs 26.2) | API level; app sandbox |
| Power | `/sys/class/power_supply/*/online`, `capacity` | `GetSystemPowerStatus`; active power scheme | IOKit `IOPSCopyPowerSourcesInfo` | `BatteryManager`; `UIDevice.batteryState` |
| NVIDIA GPU | NVML: `nvmlDeviceGetUUID`, `GetPciInfo_v3`, `GetMemoryInfo_v2`, `GetBAR1MemoryInfo`, `GetCurrPcieLinkGeneration/Width`, `GetMaxPcieLinkGeneration/Width`, `GetCudaComputeCapability`, `GetP2PStatus`, `GetNvLinkState`, `GetPowerManagementLimit`, `GetTemperature`; NUMA from sysfs | NVML (same calls); DXGI budget for WDDM devices; LUID match through `cuDeviceGetLuid` in the probe runner | – | – |
| AMD GPU | AMD SMI (`amdsmi_get_gpu_memory_total`, PCIe info) when the ROCm runtime is present; otherwise amdgpu sysfs `mem_info_vram_total`, `mem_info_vis_vram_total` (Resizable BAR when equal to VRAM), `mem_info_gtt_total`, `current_link_speed`, `current_link_width`, `unique_id`. ROCm SMI is not used; current ROCm removed it in favour of AMD SMI | DXGI and DXCore; HIP SDK runtime when installed | – | – |
| Apple GPU | – | – | IOKit `IOAccelerator` (`gpu-core-count`, model); Metal facts (`recommendedMaxWorkingSetSize`, `hasUnifiedMemory`, `supportsFamily`, `maxBufferLength`) are read in the probe runner because creating an `MTLDevice` opens the GPU | iOS: same, in the in-process runner |
| Any GPU through Vulkan | `vkEnumeratePhysicalDevices`; `VkPhysicalDeviceIDProperties` (UUID, LUID) for dedupe; `VK_EXT_pci_bus_info`; `VK_EXT_memory_budget`; `VK_KHR_cooperative_matrix` properties — all in the probe runner | same | MoltenVK is not used | Android: same |
| NPU | listed as `DeviceKind::Npu` with `NoProvider` in v1 unless a provider claims it | DXCore adapters with `DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML` | not exposed to apps | NNAPI/vendor runtimes: not enumerated in v1 |
| NVMe volume | `statfs`; `statx(STATX_DIOALIGN)`; `/sys/block/<dev>/queue/{logical_block_size,physical_block_size,nr_requests,rotational}`; controller PCI path from `/sys/class/nvme/<ctrl>/device`; temperature from the nvme hwmon (`temp1_input`) | `GetDiskFreeSpaceExW`; `IOCTL_STORAGE_QUERY_PROPERTY` with `StorageAdapterProperty` (bus type), `StorageAccessAlignmentProperty`, `StorageDeviceSeekPenaltyProperty`, `StorageDeviceTemperatureProperty`; BitLocker status from `FSCTL`/WMI where readable | `statfs`; IOKit `IONVMeController`/`IOBlockStorageDevice` properties; FileVault status from `fdesetup`-equivalent API where readable; no public temperature interface, so throttling is detected from throughput only (RFC-0021 §11) | app-private storage only; no device facts |
| Chipset-shared lane | Walk the PCI hierarchy from the NVMe controller to the root port. If the path passes a bridge that also parents SATA, USB or NIC functions and whose own link is narrower than the sum of its children, set `shared_lane` and `upstream = Chipset`. On boards with two daisy-chained chipset dies (AM5 X870E), every device behind either die shares one PCIe 4.0 x4 uplink (≈ 7.9 GB/s), and the rule treats them as one group (open question 5, answered) | SetupAPI parent chain (`DEVPKEY_Device_Parent`), same rule | n/a | n/a |

**Revision note, 2026-10-21 (PLC-E1-T02): Linux E1 sources as built.** The `probe` crate has no `unsafe` (RFC-0001 §4.2 rule 8), so Linux E1 reads files where this table names system calls:

- **Locking limits.** `RLIMIT_MEMLOCK` is the soft limit on `/proc/self/limits`' `Max locked memory` row, and `CAP_IPC_LOCK` is bit 14 of `CapEff` in `/proc/self/status`. For the process's own limits these are the values `getrlimit` and `capget` return, and any process can read its own files. One caveat applies to both sources: inside a user namespace, `CapEff` can show `CAP_IPC_LOCK` while the kernel checks locking against the initial namespace. The first `Loading` that falls short corrects the budget (§2.2, usable capacity).
- **Hybrid cores.** Intel's E-cores come from the `cpu_atom` PMU directory (`/sys/devices/cpu_atom/cpus`), which the kernel fills from CPUID leaf 0x1A. Reading the leaf directly would need the probe pinned to each CPU in turn. Elsewhere, the cores at the lowest `cpu_capacity` are efficiency cores.
- **ISA.** On the live host the standard library's runtime detection macros read CPUID and `getauxval(AT_HWCAP/AT_HWCAP2)` for every feature a stable macro covers. `lahf_lm`, the AMX bits and `sme2` have no stable macro, so they still come from the `/proc/cpuinfo` flags on the live path. Recorded fixtures use the `/proc/cpuinfo` flags throughout, which the kernel derives from the same sources.
- **Split to PLC-E1-T14.** The two checks that cannot be read from a file are the AMX permission request (`arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`) and the `io_uring_setup` half of the io_uring-allowed test. They wait for a maintainer decision on where that FFI lives. Until then E1 records `amx_permitted = false` and the `kernel.io_uring_disabled` sysctl with the process's seccomp mode.

**Host tiers per NUMA domain** (RFC-0020 §2). Each NUMA domain contributes one `HostPinned` and one `HostPageable` tier, matching the one CPU device per domain (RFC-0001 §5.1). Each tier's `affinity` is that domain's CPU device plus every GPU whose PCI `numa` is that domain (GPUs with unknown NUMA attach to domain 0). `HostMemInfo::numa_free` is that domain's free memory, and `budget::host_usable` evaluates RFC-0020 §5's per-NUMA cap (`pin_n ≤ MemFree_n + own_n − numa_reserve`) per tier. Single-domain machines get one of each. The probe measures CPU `W_eff` over both host tiers (M1), so if anonymous `HostPageable` memory reads slower than `HostPinned` on some platform the solver sees it (RFC-0020 §2).

The two host tiers of a domain draw on one pool of RAM, so the inventory also carries one **host pool record** per NUMA domain, linking them (RFC-0020 §5):

```rust
/// Added to NodeInventory. `HostPinned_n.usable = pin_limit_n`, `HostPageable_n.usable = host_usable_n`,
/// and the pool row below bounds their sum. RFC-0011 C9's joint rows read these values.
pub struct HostPoolInfo {
    pub numa: u16,
    pub tiers: [TierId; 2],                      // [HostPinned_n, HostPageable_n]
    pub usable: Bytes,                           // host_usable_n: both tiers together
    pub pin_limit: Bytes,                        // pin_limit_n: HostPinned alone
}
/// Added to NodeInventory: RDMA queue-pair capacity per device, from the daemon's ledger
/// (RFC-0031: Thunderbolt 5 allows 10 UC QPs per device, shared by every instance and epoch).
pub struct RdmaPortInfo { pub nic: u16, pub kind: RdmaKind, pub max_qp: u16, pub qp_in_use: u16 }
```

**Unified-memory devices** (ADR-0014 capacity rules): Apple → one `Unified` tier, `usable = min(recommendedMaxWorkingSetSize, RAM − resource.ram_headroom, available + own − resource.ram_headroom)` (RFC-0020 §5: the third term keeps memory other programs are using out of the budget), affinity {GPU, CPU}, and no `HostPinned` tier. GB10-class → `Unified`, `usable = MemAvailable − 6 GiB`, because `cudaMemGetInfo` undercounts. Strix Halo → `Unified` sized by the GPU-visible heap the driver reports (carve-out plus GTT on Linux, the DXGI budget on Windows), affinity {iGPU, CPU}. When detection is uncertain, the device is modelled as discrete with a measured PCIe term, which is correct and conservative.

**Usable capacity.** Discrete VRAM is confirmed by allocation in the probe runner: reserve address space, then map 256 MiB physical segments until mapping fails or `free − resource.vram_headroom` is reached, then release (CUDA VMM `cuMemCreate`/`cuMemMap`, HIP equivalents, Vulkan `vkAllocateMemory` against the budget). On Windows the probe never allocates past the DXGI local budget, because WDDM over-commits by demoting to system memory instead of failing. Host and unified tiers are not allocation-probed at startup: wiring or locking most of RAM would stall the user's machine (`tiered-memory-inference.md` §10.2). Their `usable` comes from the OS budget and RFC-0020's rule, and is corrected from the first `Loading` that falls short (`Capacity` in `Loading`, RFC-0001 §10.1), with the correction stored in the profile cache.

*Note 2026-10-22 (OQ-HW-13):* in M0a the Windows E2 path is checked on the agent pool's NVIDIA host booted into Windows (QUAL-E15-T16), an untimed host, because R-WIN arrives in M1b. R-WIN takes over from M1b.


### 3. Startup microbenchmark suite

#### 3.1 Suites

```rust
/// Carried by RFC-0001's IPC ProbeRequest { suite }. Results return as ProbeResult { profile }.
pub enum ProbeSuite {
    /// Daemon start with no valid cache: every step in §3.2 except S2 and L2.
    Full { schemes: SmallVec<[QuantScheme; 8]> },
    /// Daemon start with a cached profile: the confirmation subset (§5.2).
    Confirm { cached: ProfileKey },
    /// A model was registered: measure its real schemes and shapes on this node's devices.
    Model { model: ModelId, shapes: ModelShapes, schemes: SmallVec<[QuantScheme; 8]> },
    /// Sustained, throttled NVMe read on one volume (S2). Runs in the background.
    Sustained { tier: TierId, extent: Bytes, qd: u16, duration: Nanos },
    /// Device-to-device latency through the real collective path (L2), with a runner on the remote node.
    DevicePath { local: DeviceId, remote: RemoteEndpoint, remote_device: DeviceId, payloads: SmallVec<[Bytes; 3]> },
    /// One device or volume again, on request (`syntropy probe --device`, or after DeviceLost).
    Device { device: DeviceId },
}
pub struct ModelShapes { pub hidden: u32, pub ffn: u32, pub expert_ffn: Option<u32>, pub top_k: Option<u16>, pub head_dim: u16 }
```

**The default probe set** of quant schemes for `Full` is the union of (a) every scheme with ≥ 1% of read-per-token bytes in any model in the local store and (b) `{Q4_K, Q8_0, BF16, MXFP4}`, capped at 8 schemes. Schemes outside the set are measured by the `Model` suite before that model's first solve (§3.4), or derived from the nearest measured scheme until then.

#### 3.2 Steps

Every step uses the real kernel through `KernelProvider` (for CPU, the native CPU provider) and buffers from the arenas the tier manager will use. No step writes to the user's files.

| Step | What | Method | Samples and stop rule | Fills |
|---|---|---|---|---|
| D1 Open | context creation per device | free memory (management API) before and after `open`; the difference is the context reserve | once | `TierInfoExt::context_reserve` |
| D2 Capacity | discrete VRAM | reserve-and-map loop (§2.2) | once | `DeviceCaps::usable`, `TierInfo::usable` |
| M1 Memory W_eff | weight bytes per second per (device, tier, kernel class, scheme) | the provider's decode GEMV (`DenseGemv`) over a ≥ 1 GiB buffer that is > 2 × LLC, rows sized like the census (K = hidden); for MoE-capable devices also `ExpertGemv` on expert-sized blocks (≈ 1–26 MiB) gathered in random order with top-k = 8. GPU over `Vram`/`Unified`; CPU over `HostPinned` and `HostPageable` of its own NUMA domain, all compute-pool cores, NUMA-pinned, with ADR-0012's reserved cores excluded. On `Unified` devices a second pass runs while the I/O engine reads at the volume's S1 rate, because the GPU, the CPU pool and DMA share one memory system (RFC-0020 §9); the result is stored as `QuantBw::contended` | ≥ 5 timed passes after 2 warm-up passes; stop when the 95% CI half-width < 2% of the mean or at the step budget | `QuantBw` n = 1, `tier_bw` |
| M2 Batch curve | the same kernels at n ∈ {2, 4, 8, 16, 32} tokens per pass | GEMV → small GEMM at n columns; record weight bytes per second. **Runs in the `Model` probe** (§3.4) for the model's own schemes, not at startup. CPU curves run on one host tier (`HostPageable`); memory-bound points use the ≥ 1 GiB buffer, compute-bound points (n above the ridge) a buffer of 4 × LLC, since they no longer read at memory speed | ≥ 3 passes per point | `QuantBw::curve` |
| M3 Fixed cost | t_fixed per backend per device | synthetic layer stacks of 1, 2, 4 and 8 tiny layers (hidden 256), timed end to end through the executor's eager path and, where capture works, a captured graph; least-squares intercept | 20 runs per depth | `DeviceCaps::t_fixed`, `LaunchCaps` |
| M4 Compute F_eff | prefill-shaped throughput per (op class, dtype) | GEMM at M ∈ {512, 2048}, N = K = hidden-class sizes (4096, 7168, 8192) in each tensor-core dtype the device has (FP16/BF16, FP8, FP4, INT8) and in the dequant path for quantized schemes; attention at L = 4096 for each attention kind the provider implements | ≥ 3 passes | `DeviceCaps::throughput` |
| H1 PCIe | pinned H2D and D2H at 64 MiB; bidirectional; pageable H2D at 64 MiB; 16 KiB copy latency | provider `copy` on the copy stream from a registered arena region | ≥ 10 large copies, ≥ 200 small ones | `HostLinkCaps` |
| H2 Hand-off | t_switch, and the wake path | GPU kernel writes a 16 KiB activation to host-mapped memory and `signal_host`; a CPU thread spinning on the flag resets it, writes the activation back and raises a second flag; the stream waits device-side and resets that flag (RFC-0040 §4.2's constant-valued handshake, on the device's native path or its fallback); round trip / 2. A second variant parks the CPU thread on the backend's device-triggered wake (RFC-0012 §3.3: a host function on CUDA and HIP, a shared-event listener on Metal, a timeline-semaphore wait on Vulkan) and measures doorbell-to-running latency | ≥ 200 round trips each | `DeviceCapsExt::t_switch`, `DeviceCapsExt::wake` |
| H3 DMA interference | CPU pool slowdown under concurrent DMA | M1's CPU kernel alone, then with H2D copies of 64 MiB looping from the same NUMA node | 3 passes each | `DeviceCapsExt::dma_interference` |
| H4 Peer copies | GPU↔GPU on one host | `cudaDeviceCanAccessPeer` or HIP equivalent, then a test copy each way whose bytes are verified, then 64 MiB timed copies (ADR-0006 item 5) | ≥ 10 copies | `HostLinkCaps::p2p` |
| H5 Lock accounting | how locking and registration are charged | Linux: read `VmLck` and `VmPin` from `/proc/self/status`; `register_host` a 64 MiB anonymous range with each GPU provider and read them again; register a 64 MiB io_uring fixed buffer and read again; release all (≈0.2 s [estimate], in phase E). Windows: lock and register 1 GiB ranges with `VirtualLock` and the provider, one step at a time with the responsiveness check of RFC-0020 §6, up to `max_pinned_fraction`, recording the first failure; release all. The Windows part runs in the background on first start after admission, with its own budget (≤ 60 s [target]); until it finishes, `lockable_measured` is RFC-0020's static estimate, marked `Prior`. Never while a runner is loading or serving; cached by hardware fingerprint, driver version and OS build | once per key | `HostMemInfo::lock` (inputs to RFC-0020 §5 `lockable`, `io_fixed`, `lockable_measured`) |
| M5 Host-wait stack | per-wait cost of backends without whole-step capture of host waits | a k-layer stack (k ∈ {8, 32}) with one host wait per layer, built as RFC-0012 §7.5 builds steps: on Metal each wait path the OS offers (encoder-level shared-event waits inside one pre-encoded command buffer; queue-level waits between command buffers committed together, RFC-0040 open question 1); on Vulkan one submission with timeline waits, each released by one host `vkSignalSemaphore`; CUDA and HIP a captured graph with device-side waits for comparison; `t_wait` = slope of step time over k | ≥ 20 steps per k | `DeviceCapsExt::host_wait`, per Metal path |
| M6 CPU partial attention | the CPU's rate for attention over host-resident KV pages, per attention kind | the CPU provider's split-attention kernel (RFC-0022 §5.3) over KV pages in a `HostPinned` arena, where spilled KV lives (RFC-0022 §3), at T = 1 and T = 8 query rows at startup, and again at the plan's verify-window size `T_max` in the `Model` probe, per attention kind the model families use (GQA, MLA absorbed, sliding, sparse). It reports `W_cpu,attn(kind, T)`, the KV bytes per second consumed at T rows, which is the measured quantity of RFC-0022 §5.3–§5.4; `F_cpu,attn` and `T*_kv` are derived from the measured points as RFC-0022 states. MLA's many query heads per KV element make it compute-bound on the CPU at decode [estimate: ≈ 240 FLOP per byte of KV for many query heads per KV head], which this measures rather than assumes | ≥ 5 passes | `DeviceCaps::throughput` under `Attention(kind)`, one `W_cpu,attn(kind, T)` value per measured T, used by RFC-0011 C7's CPU-split path |
| S1 NVMe quick and engine selection | rate, latency and CPU cost per volume, and the engine choice | RFC-0021 §10 defines the measurement: every engine available on the OS (io_uring `READ_FIXED` + `O_DIRECT`, the `pread` pool, IoRing, IOCP, `F_NOCACHE` + `preadv`, `MacMapped`; GDS where RFC-0021 §9 allows) at the request mix (the census's expert extent, streamable unit, row bundle and KV page run; 4 KiB, 1 MiB, 8 MiB and 32 MiB before any model is known) and QD 1, 4 and 16; offsets random and aligned. Source: reads of an existing pack or source file on that volume; **no writes to the user's SSD**. A volume with no readable file yet defers S1 until the first source chunk lands there, and the profile carries `Derived` storage values until then. Then the RFC-0021 choice (highest rate at the dominant size; within 5%, the lower CPU time per GB) and tuning (`qd_target`, `chunk`, `sqpoll`) | ≈0.15 s per grid point, ≈5 s per volume in total (RFC-0021 §10 item 3); drop the first 50 ms | `StorageCaps::{burst_read, small_read, extent, queue_depth}`, `StorageCapsExt` |
| S2 NVMe sustained | thermally throttled steady state | the S1 engine at the pack's expert extent size and `qd_target`, for `probe.nvme.sustained_secs` (default 90 s), with temperature sampled every 1 s. Runs after the first model's `Loading` completes or in an idle window, never while `Packing` or `Loading` use that volume; one-second bins that overlap any other store I/O on the volume are discarded. Until it completes, `sustained_read` = 0.5 × S1's burst rate at the extent size, marked `Prior` and listed in the plan's assumptions | 1 s rate bins; sustained = median of the last 30 kept bins | `sustained_read`, `throttle`, `temperature_c` |
| L1 Links | α, β, p50/p99 per (link, transport) | ping-pong at 64 B, 16 KiB, 1 MiB and 64 MiB. At startup: α on every transport the pair supports (QUIC, TCP, shared memory, RDMA), and β only on the best-ranked transport, with 2 transfers per direction of 64 MiB on wired links and 16 MiB on wireless links. β for the other transports runs in the background after admission. α from the 64 B round trip / 2; β timed to the receiver's acknowledgement | ≥ 200 samples at 64 B and 16 KiB; 2 per direction at the β size | `LinkCaps::{alpha, beta, loss}`, `grid` |
| L2 Device path | α_eff, including S/β at the partial size | `DevicePath` suite: device buffer → transport → remote device buffer and back, including the doorbell or event sync on both ends, at payloads {64 B, h_model × 4 B, 4 × h_model × 4 B} | ≥ 200 round trips (ADR-0006 item 2) | `LinkCaps::alpha_eff` |

The RAM-side W_eff is the CPU's rate through the real quantized kernel, not a triad: the cost model separates a 90 GB/s DDR5 triad from a ≈70 GB/s matvec (§4a), and the solver needs the latter. A STREAM-style triad still runs once per NUMA domain (0.3 s) as a sanity bound, and a kernel rate below half of it raises `DeviceProblem::BelowSpec`.

#### 3.3 Measurement rules

1. **Isolation.** M1 on the CPU and H1 on any GPU never overlap, because DMA reads DRAM; H3 measures that overlap on purpose. GPU-only steps (M1/M2/M4 on `Vram`) overlap with each other across GPUs and with S1. L1 β probes wait until the CPU's M1 has finished, because QUIC encryption uses CPU cores.
2. **Clocks.** GPU timings use device events (or the provider's timestamp op) bracketing the work; CPU timings use the monotonic clock with the thread pinned. Each sample discards the first pass after any allocation.
3. **Robust statistics.** `p50` and `p99` come from the samples; rates report the median, not the mean. `samples` is the count after discarding warm-up.
4. **Sanity bounds.** A value below 0.5 × the spec bound of the narrowest element (cost model §5.2, final row) sets `DeviceProblem::BelowSpec` and is still used, since it is what the machine delivers. A value above 1.1 × spec indicates a broken measurement (a cached read, a timer fault): the step is retried once, then the spec bound is recorded with `MeasureSource::Prior` and the plan lists it as an assumption.
5. **Power state.** Every profile records `PowerSource`. On battery, M1–M4 run at a third of their sample counts; S2 is deferred until mains power.

#### 3.4 The model probe

When `ModelRegistered` appears in the log, every daemon's local `plan()` (ADR-0009) checks which (device, tier, kernel class, scheme) entries the model's census needs and which its profile lacks, counting only schemes with ≥ 1% of the model's read-per-token bytes. If any are missing, it runs `ProbeSuite::Model` with the model's real shapes (`hidden`, FFN and expert widths, top-k) and sends `ReportProfile`. Budget: ≤ 5 s per node (target), now that the model's batch curves (M2) run here: on the CPU, 3 schemes × 2 classes × ≈ 0.39 s per curve ≈ 2.3 s on one host tier, with GPU curves in parallel [estimate]. The coordinator, on `EnsureInstance`, waits up to `probe.model_wait` (default 5 s) for these reports from candidate nodes, then solves with whatever it has; missing entries are `Derived` (scaled by bytes per weight from the nearest measured scheme of the same kernel class on the same device and tier) and listed in `PlacementPlan::assumptions`.

#### 3.5 The device-path probe and the TP/EP gate

Host-level α from L1 is not α_eff: the cost model back-solves ≈130–150 µs per collective where the wire alone is 10–21 µs (§3.8). Cross-node TP and EP may be enabled only on a measured α_eff (ADR-0006 item 2), so L2 runs only where it can matter:

```text
pre_gate(pair a, b) =
    medium(a, b) not in {WiFi, Overlay}
    and alpha_L1(a, b).p99 + t_switch(a) + t_switch(b) < alpha_eff_max_ceiling
alpha_eff_max_ceiling = max over registered models M and N in {2, 4} of
    (B_w(M) / W_eff_min(island)) · (1 − 1/N) / (2 · n_layers(M))      // cost model §3.8
```

Pairs that pass the pre-gate get a `DevicePath` probe scheduled by both daemons when neither node has a `Ready` runner with live sequences. Results are cached with the profile and refreshed in `Warming` of any plan that uses the link for TP or EP (RFC-0001 §3.3). Until L2 has run, the solver treats the pair as failing the gate.


### 4. Time budgets and probe scheduling

All budgets are **targets**. The startup total derives from cost model §5.3, whose step budgets sum to ≈11 s, plus enumeration and context creation; SC-22 caps it at 15 s per node.

| Phase | Steps (parallel inside a phase) | Budget per node [target] | Derivation |
|---|---|---|---|
| A Enumerate | E1–E4 in the daemon; probe runner start | 0.5 s | management-API calls and sysfs reads only |
| B Open | D1, D2 per GPU, in parallel across GPUs | 1.5 s | context creation ≈0.3–1 s per CUDA/HIP device [estimate]; D2 maps 256 MiB segments, ≈128 maps on a 32 GB card |
| C Device bandwidth | M1 (n = 1), M4, M3, M5 on every GPU in parallel; S1 on every volume in parallel | 5.0 s | GPU: M1 4 schemes × 2 classes × ≈ 0.25 s = 2 s, M4 ≈ 1 s, M3 ≈ 0.5 s, M5 ≈ 0.5 s [estimates]; S1 ≈ 5 s per volume across engines, sizes and queue depths (RFC-0021 §10) sets the phase |
| D CPU | M1 (n = 1) on each CPU device over both host tiers, M6, then H3 | 3.0 s | M1: ≥ 7 passes of ≈ 15 ms (1 GiB at ≈ 70 GB/s) ≈ 0.11 s per (tier, class, scheme), × 4 schemes × 2 classes × 2 host tiers ≈ 1.8 s; M6 ≈ 0.4 s; H3 ≈ 0.3 s [estimates]. Batch curves are not in this phase (M2 runs in the `Model` probe) |
| E Host links | H1, H2 (both variants), H4 per GPU; H5 on Linux | 1.2 s | §5.3 item 3 (0.5 s), the hand-off loops, ≈ 0.2 s for Linux H5 |
| F Network | L1 α to each admitted node, overlapping B–E; β on the best transport after D, serialised per NIC | ≈ 0.9 s per wired 2.5 GbE pair, ≈ 1.3 s per Wi-Fi pair | β: 2 transfers per direction of 64 MiB at ≈ 0.29 GB/s ≈ 0.9 s on 2.5 GbE; 16 MiB at 400 Mbps ≈ 1.3 s on Wi-Fi [estimates] |
| **Total** | | **≈ 11 s critical path; ≤ 15 s hard cap** [estimate] | A + B + C + D + E = 0.5 + 1.5 + 5.0 + 3.0 + 1.2 ≈ 11.2 s. β starts after D (≈ 10 s) and, at four wired peers on one NIC, ends at ≈ 13.6 s; more peers, or Wi-Fi peers beyond two, continue in the background after the 15 s cap with their α already measured |
| Background | S2 per volume; L2 for pre-gated pairs; `Model` suites; β on secondary transports; Windows H5 | S2: 90 s default (60–120 s per §5.3 item 4); L2: ≤ 2 s per pair; `Model`: ≤ 5 s per node; Windows H5: ≤ 60 s | idle time or after `Loading`; never delays the first token (SC-22) |

**Enforcement.** The probe orchestrator gives each step a deadline equal to its share of the phase budget. A step that hits its deadline returns what it has, with the sample count it reached; a step with fewer than 3 valid samples returns nothing, and the value is `Derived` or `Prior`. At the 15 s hard cap the probe runner sends a partial `ProbeResult` with `status = Quick` and keeps going in the background at low priority, sending a second `ProbeResult` when it finishes. The node is admitted, and can host stages, on the partial profile.

**Busy machines.** A re-probe (cache invalid, power change, `syntropy probe`) never runs while a runner on the node is `Ready` with live sequences, because the probe and the stage would corrupt each other's numbers. It waits for an idle window of `probe.idle_window` (default 10 s with no live sequence), and online refinement (§6) covers the meantime.

**Scaling.** Phase C's budget is per device, and devices run in parallel, so a node with 4 GPUs stays within budget. The CPU phase is per NUMA domain and runs domains in parallel. A node with 8 or more volumes probes only the volumes that hold the store or a configured tier path.

### 5. Profile cache across restarts

#### 5.1 Key and storage

```rust
pub struct ProfileKey(pub Blake3);   // BLAKE3 over the canonical encoding of:
// - NodeInfo::host (HostFingerprint) and every DeviceInfo::fingerprint
//   (vendor/device/subsystem IDs, PCI address, VRAM size, VBIOS or firmware version, UUID)
// - DIMM population where SMBIOS is readable without privilege, else total RAM and NUMA layout
// - driver and runtime versions per device; OS kernel version; Syntropy BuildId
// - ProbeSuiteVersion and the kernel registry hash of every provider in the build
// - PowerSource (Mains | Battery) and, on Windows and macOS, the active power mode
// - Volume identity (filesystem UUID, controller model and firmware) for each Nvme tier
```

Profiles live at `<state_dir>/probe/<key>.profile` (postcard, mode 0600), one file per key, at most `probe.cache.max_profiles` (default 8, so a laptop keeps its mains and battery profiles for several driver versions). Writes go to a temporary file and are renamed into place. A file that fails to decode is deleted and the node re-probes. Every `Measured<T>` keeps its original `at_unix_ms`, so the dashboard can show how old each number is.

#### 5.2 Restart path

```text
on daemon start:
    key ← compute ProfileKey from E1–E4 and driver versions          // phase A only, ≈0.5 s
    cached ← load(<key>.profile)
    if cached is None:
        run Full; persist; report
    else:
        report cached immediately with status = cached.status         // node is admissible at once
        run Confirm (≈3 s target):
            D1, D2 per GPU; M1 at n = 1 for DenseGemv in one scheme per device and tier;
            M3; H1 pinned H2D; S1 at 32 MiB × QD 4; L1 α to each admitted node
        for each confirmed quantity q:
            if |q_new − q_cached| / q_cached > probe.cache.tolerance (default 0.10):
                invalidate(cached); schedule Full for the next idle window
                report the Confirm values for the quantities measured, Derived for the rest
                break
        else:
            refresh at_unix_ms of the confirmed entries; persist
    if cached is older than probe.cache.max_age (default 30 days):
        schedule Full for the next idle window
```

Online corrections (§6) are merged into the cached profile when they are stable (§6.3), so a restart begins from the latest knowledge, not the original probe. Capacity corrections from a short `Loading` are stored the same way.

**Power-state switch.** When `PowerSource` changes, the daemon loads the profile for the new key if one is cached and reports it as drift (§7); otherwise it keeps the current profile, marks the node's profile `power` mismatched, and lets online refinement and drift detection converge on the new numbers.


### 6. Online refinement from live timings

#### 6.1 Where samples come from

| Source | Carrier | Rate | Quantities |
|---|---|---|---|
| Plain steps | runner `telemetry` ring (RFC-0001 §3.5) | every step | step wall time, tokens in the step, bytes read per tier (from the step program's static byte counts plus the expert hits and misses actually taken), credit stalls |
| Profiled steps | `telemetry` ring | one in `probe.online.profile_every` (default 64) steps | timestamps at segment boundaries: stage entry; each run of consecutive layers on one compute site; each MoE layer's router doorbell, CPU-pool completion and GPU combine; hand-off. Gives per-term times for the §6.2 fits |
| I/O engine completions | `telemetry` ring | every read | size, queue depth at submit, latency, engine; NVMe temperature every 5 s |
| Channel statistics | `Channel::stats()` (RFC-0001 §5.8) | per frame | send-to-acknowledge latency per hop, bytes, retransmits |
| Origin fetches | the store's fetcher (RFC-0041 §8) | per completed origin range read | bytes and wall time, giving `CapabilityProfileExt::wan_down`; merged by §6.3's rule |
| Gossip heartbeats | daemon `cluster` | 1 s per node pair (ADR-0009) | control-path RTT per link, as a cheap continuous α sample for links without data traffic |
| Warming | runner, RFC-0040 op selection and the α_eff check (RFC-0001 §3.3) | per plan load | per-op times at the plan's real shapes; α_eff through the plan's real path |

A profiled step is a second captured variant of the same step program with timestamp ops inserted at segment boundaries; it computes the same values in the same order, so it changes no output. Each provider supplies the timestamp mechanism (a kernel that writes the device clock to host-mapped memory, or counter sampling at encoder boundaries on Metal); RFC-0040 owns it. CPU-pool spans use the monotonic clock. Device clocks are mapped to host time once per profiled step by a paired host read.

#### 6.2 Estimators

The live aggregator (`probe::live`, in the daemon, fed from the telemetry rings of local runners) keeps one estimator per profile quantity it can observe:

| Quantity | Model fitted | Estimator |
|---|---|---|
| W_eff per (device, tier, class, scheme) at the live n | `t_seg = c + B_seg / W` per segment, B_seg the segment's bytes | recursive least squares on (B_seg, t_seg) pairs with forgetting factor λ = 0.995 per sample; W = 1/slope. Requires ≥ 2 distinct B_seg values in the window (they differ across stages and MoE layers); otherwise W = B_seg / (t_seg − t_fixed) |
| t_fixed per device | intercept of the per-step fit `t_step = t_fixed + Σ_seg B_seg / W_seg` | same RLS, intercept term |
| NVMe sustained rate | busy-time throughput and the `Throttled`/`Normal` state that RFC-0021 §11's detector publishes from the I/O thread every second | median of the last 30 one-second values; throttle onset and recovery are RFC-0021's (below 0.80 × `sustained_read` for 20 s; normal after 60 s above 0.90 ×), recorded in `ThrottleInfo` |
| PCIe pinned rate | copy bytes / copy time on the copy stream (profiled steps) | median over a 60 s window |
| Hop latency per link | `t_hop = α + S/β` on (S, t) pairs from channel stats | RLS for α and β; p50/p99/p99.9 from a log-bucketed histogram (bucket ratio 1.02, ≈2% resolution) over `probe.link.window` (default 15 min) plus an all-time histogram decayed by half every 24 h |
| α (idle links) | heartbeat RTT / 2 | same histogram |

**Wi-Fi and other wireless media.** The solver costs these hops at their **mean** and bounds their tail with a stall rate (RFC-0011 C14, since 2026-10-20), so the estimator publishes three things per wireless link: the mean (`LinkCapsExt::mean`), the compacted histogram (`hop_hist`) from which the stall probability `P(t_hop > mean + solver.stall_threshold)` is read, and the p99 for the TP/EP gate (§3.5). Startup's 200 samples cannot pin a tail that is 50× the median (cost model §2.5). Until a wireless link's histogram holds 1,000 samples, the tail is the prior `max(measured p99, 3 × p50 + 10 ms)` placed at the 99th percentile, the stall probability is read from it, and the plan lists it as an assumption; the mean comes from the samples it has. The 10 ms floor comes from the cost model's p90 of ≈20 ms; the factor is a chosen conservative default for RFC-0061 to tune. The estimator also tests the hop-time excess for a periodic pattern (autocorrelation over lags of 0.5–2 s) and sets `periodic_stall`, because a stall that recurs about once a second is not a 1% tail.

#### 6.3 Merging into the profile

The aggregator holds live estimates locally; it does not stream them to the coordinator. An estimate is written into the node's cached profile (with `MeasureSource::Online`) when it has ≥ 200 samples, its 95% CI half-width is < 5%, and it has been stable within ±5% for 60 s. Merging into the cache and reporting to the coordinator are separate: the coordinator hears only about changes that pass the drift rule in §7, so the event log stays quiet under load (RFC-0001 §5.10).


### 7. Drift detection and re-placement triggers

#### 7.1 Two comparisons, two thresholds

Each runner compares live per-term timings with the plan's prediction (RFC-0001 §7.4), and the daemon's aggregator compares live estimates with the profile. The two answer different questions:

| Comparison | Question | Output |
|---|---|---|
| live term vs `PlacementPlan::predicted` | is the plan performing as solved? | a **drift event** in telemetry and the dashboard at `probe.drift.event_threshold` (default 10%, RFC-0001 §14 rule 2) |
| live estimate vs `CapabilityProfile` value | has the hardware changed? | a **profile drift**: `ReportProfile` with `Online` values at `replan.drift_threshold` (default 20%, ADR-0008 item 6), which leads to `GraphUpdated` and a deferred re-plan under RFC-0001 §7.2's hysteresis and payback rules |

A plan can be off while the hardware is not: expert hit rates, speculative acceptance or context lengths differ from what the solver assumed. Those **workload** terms are not profile quantities. Their drift goes to the coordinator as `RequestReplan { reason: ReplanReason::WorkloadDrift { term } }`, and the solver reads the current hit-rate curves (RFC-0023) and acceptance rates (RFC-0012) when it re-solves. The curves themselves travel in `ExpertWorkloadReport` (RFC-0023 §13) through `ClusterCommand::ReportWorkload` and its matching event, both proposed for the RFC-0001 amendment pass.

**Expert-cache tier shortfall.** RFC-0023's shadow caches measure what its policy would achieve at each capacity, but not the admission bandwidth a real tier has, so a tier can fall short of its curve (RFC-0023 §13). The aggregator compares each cache tier's measured hit rate with the curve's prediction at that tier's current budget, per layer and at fixed points: the VRAM tier's hit rate against `H(y)`, and the RAM tier's hit rate conditional on a VRAM miss against `(H(u + y) − H(y)) / (1 − H(y))`. The per-layer measured rates arrive in `ExpertWorkloadReport::tier_hit` (RFC-0023 adds the field). The comparison uses the same thresholds as every other term: a relative shortfall above `probe.drift.event_threshold` raises a drift event; one above `replan.drift_threshold` that holds for `replan.drift_window` with at least 200 samples and passes `probe.drift.min_impact` sends `WorkloadDrift { term: ExpertTierShortfall { tier } }`. A shortfall that coincides with a published `hull_gap > 0.03` at that budget is labelled curve over-prediction in the report rather than a workload change, so the dashboard can tell them apart. RFC-0011 §12.1 says what the solver does with it.

#### 7.2 The rule

```text
for each observed quantity q with profile value P_q and live estimator L_q:
    r_q(t) = (median of L_q over the last probe.drift.sample_window) / P_q − 1
    impact_q = |L_q − P_q| expressed as time per token at the live plan / predicted t_tok
    drifting_q = |r_q| > replan.drift_threshold and impact_q > probe.drift.min_impact
    if drifting_q has held continuously for replan.drift_window
       and the estimator has ≥ 200 samples in that window:
        mark q drifted
on any q drifted, at most once per probe.report_min_interval:
    build a profile with every drifted q replaced by its Online value (MeasureSource::Online)
    ReportProfile { profile }      // coordinator emits GraphUpdated { cause: Drift }
    reset drifted flags; P_q ← L_q for the reported quantities
```

- `impact_q` keeps tiny terms from triggering re-plans: a 50% change in a term worth 0.5% of the token is not worth a re-solve. Default `min_impact = 0.03`.
- The window rule and the sample floor together keep one slow step, a GC pause in another app or a single Wi-Fi burst from reporting. A wireless link reports when its mean or its stall probability at `solver.stall_threshold` moves past the threshold, because the solver costs wireless hops at the mean and bounds their tail with C14 (ADR-0008 item 4 as revised 2026-10-20); its p99 reports only for links inside a TP/EP island, whose gate uses it.
- Capacity is handled faster: a fall in an OS budget (DXGI budget notification, Vulkan `heapBudget`, NVML free memory, `MemAvailable` under the Linux pressure-stall signal, the macOS memory-pressure source) that takes `usable` below what running instances hold goes straight to the memory-pressure path (RFC-0001 §10.3) and, for VRAM, to the runner holding the device (amendment A2). It becomes `ReportProfile` only if it persists for `replan.pressure_window`.

#### 7.3 Typical drift causes and what each produces

| Cause | Observed as | Report |
|---|---|---|
| NVMe thermal throttling | NVMe 1 s bins fall below S2's sustained rate | `StorageCaps::sustained_read` (Online), `throttle` updated |
| Laptop unplugged | power source event; CPU and GPU W_eff fall | switch to the cached battery profile, or Online values |
| Another app takes GPU bandwidth or memory | W_eff on `Vram` falls; DXGI/NVML budget falls | W_eff (Online); capacity through §7.2's fast path |
| Wi-Fi degrades | hop mean or stall probability rises | `LinkCapsExt::mean` and `hop_hist` (Online); p99 too for island links |
| RDMA link drops to a fallback transport | `Link` error, transport change | `LinkInfo::transports` changed (inventory) |
| Hit rate below prior | MoE miss term above prediction | `WorkloadDrift { term: ExpertMiss }` |
| A cache tier below its hit curve (admission-bandwidth bound, or curve over-prediction) | `tier_hit` below the curve at the tier's budget (§7.1) | `WorkloadDrift { term: ExpertTierShortfall { tier } }` |
| Acceptance below prior | tokens per verify window below prediction | `WorkloadDrift { term: SpeculativeAcceptance }` |


### 8. Configuration knobs

All are `node` scope (RFC-0001 §11.1) except where marked. RFC-0054 owns the schema; these are the defaults.

| Key | Default | Meaning |
|---|---|---|
| `probe.startup_budget` | 15 s | hard cap on the startup suite (SC-22) |
| `probe.schemes` | auto (§3.1) | override the default probe set |
| `probe.nvme.sustained_secs` | 90 s | S2 duration (60–120 s range from cost model §5.3) |
| `probe.model_wait` | 7 s (cluster scope), above the `Model` probe's 5 s budget so a report that uses the whole budget still reaches the coordinator before it solves | how long the coordinator waits for `Model` probe reports before solving |
| `probe.idle_window` | 10 s | idle time required before a re-probe runs |
| `probe.cache.tolerance` | 0.10 | relative change in a `Confirm` quantity that invalidates the cache |
| `probe.cache.max_age` | 30 days | full re-probe after this, in an idle window |
| `probe.cache.max_profiles` | 8 | cached profiles kept per node |
| `probe.online.profile_every` | 64 steps | profiled-step period |
| `probe.link.window` | 15 min | link histogram window |
| `probe.drift.event_threshold` | 0.10 | plan-vs-live deviation that raises a drift event |
| `probe.drift.sample_window` | 30 s | median window for drift ratios |
| `probe.drift.min_impact` | 0.03 | minimum share of t_tok for a drifting term to count |
| `probe.report_min_interval` | 60 s | minimum gap between drift `ReportProfile`s from one node |
| `replan.drift_threshold` | 0.20 (cluster scope) | profile drift that reports and may re-plan (RFC-0001 §7.2; tuned by RFC-0033) |
| `replan.drift_window` | 60 s (cluster scope) | how long drift must persist |
| `probe.exclude` | [] | devices, volumes or interfaces never probed or used, by name, PCI address or UUID |

### 9. Failure handling

| Failure | Detection | Action |
|---|---|---|
| Probe runner crashes or hangs (driver fault) | process exit; no `ProbeResult` within its budget + 10 s | kill; mark the devices it was probing `DeviceProblem::ProbeCrashed`; retry those devices once, alone, in a new probe runner; if the retry fails, exclude them until the next daemon start or `syntropy probe --device`. Other devices' results stand |
| Context creation fails | `open` returns `Device` or `Unsupported` | `DriverTooOld` or `NoProvider`; the device stays visible and is filtered |
| VRAM allocation probe returns far less than the driver's free figure | D2 result < 0.8 × (free − headroom) | use the D2 figure; record `BudgetSource::AllocationProbe` |
| io_uring blocked (seccomp, sysctl, ChromeOS) | `io_uring_setup` returns `EPERM`/`ENOSYS` | record in `engines_blocked`; probe with the next engine RFC-0021 lists; the profile names the engine used |
| `O_DIRECT` unsupported on the volume (tmpfs, some ZFS configs) | `EINVAL` on open or `STATX_DIOALIGN` reports 0 | `DirectIoSupport::Unsupported`; S1 runs buffered with `posix_fadvise(POSIX_FADV_DONTNEED)` between passes, and the volume is flagged so the result is not mistaken for a direct-I/O rate; RFC-0021 decides whether the volume can be a tier |
| No readable pack or source file on a volume yet | store inventory | S1 deferred until the first source chunk lands on that volume; storage values are `Derived` from the volume's bus and link (marked in `assumptions`) |
| Volume full | `statfs` free below the shard's need | the `Nvme` tier's `usable` reflects it; the solver's C9 pack check fails on that node |
| No temperature source | `TempSource::None` | S2 still measures the rate; throttle detection uses rate only |
| A remote node disappears mid L1/L2 | transport error | discard partial samples for that pair; membership (RFC-0030) handles the node |
| Measured value outside sanity bounds | §3.3 rule 4 | flag, retry once, then use or clamp as the rule says |
| Profile cache unreadable or from an older suite | decode error, `ProbeSuiteVersion` mismatch | delete and re-probe |
| Clock anomaly (device timestamps go backwards or jump) | profiled-step timestamps non-monotonic | drop that profiled step; three in a row disable profiled steps on that device until the runner restarts and log a warning |

No probe failure blocks admission. A node with no usable accelerator is admitted as CPU-only; a node whose CPU probe also failed is admitted with `may_host_stages = false`, so it can still act as ingress.


## Performance model and budgets

**What the probe values feed.** The solver's per-token model is cost model §3.2–§3.10 (RFC-0001 §14):

```text
t_tok = Σ_s [ Σ_tiers B_{s,tier} / W_eff(s, tier, class, scheme)   ← QuantBw at n = 1
            + t_miss,s                                            ← StorageCaps sustained, h from RFC-0023
            + n_switch,s · t_switch                               ← DeviceCapsExt::t_switch
            + t_fixed,s ]                                         ← DeviceCaps::t_fixed
      + Σ_hops (α_e + S / β_e)                                    ← LinkCaps (mean for wireless; tail via C14)
n*      = smallest n where t_stream(n) < t_EIP(n), read off the CPU and GPU QuantBw curves
          and HostLinkCaps::h2d_pinned (cost model §3.2)
t_verify(T) = Σ_tiers B_tier(T) / curve_tier(T)                   ← QuantBw at n = T, with B(T)
          including the expert union U(T) for MoE (cost model §3.9, §3.10)
TP gate = LinkCaps::alpha_eff (mean and p99) < α_eff,max          ← cost model §3.8, SC-16
```

Because SC-30 requires predicted `t_tok` within ±15% of measured, and SC-3 leaves ≤ 5% unexplained, the probe's own error must be well below 15%. The budgets below are **targets** chosen to leave room for the cost model's remaining terms.

| Budget | Target | Derivation |
|---|---|---|
| Startup suite | ≤ 15 s per node, ≈ 11 s critical path | SC-22; §4 phase table |
| Confirmation on restart | ≤ 3 s | the `Confirm` subset with D2 replaced by a free-memory check against the cached `usable` (D2 re-runs only if free memory differs by > 5%): D1 ≈ 0.5 s per device in parallel, one M1 per (device, tier) ≈ 0.25 s × 2, M3 ≈ 0.5 s, H1 ≈ 0.3 s, one S1 point ≈ 0.4 s, L1 α overlapping; ≈ 2.2 s for a 2-device node [estimate] |
| Model probe | ≤ 5 s per node | M1 for ≤ 6 new (device, tier, class, scheme) entries × 0.25 s, plus M2 curves ≈ 0.39 s each on one CPU host tier (memory-bound points on 1 GiB, compute-bound on 4 × LLC), GPU in parallel [estimate] |
| Repeatability | coefficient of variation ≤ 3% across 5 consecutive `Full` runs for W_eff on mains power | chosen; a 3% spread in each of 2–3 dominant terms keeps the combined prediction error within a few percent, well inside SC-30's 15% |
| Probe vs steady state | `abs(W_probe − W_online) ≤ 10%` after 10 minutes of decode | chosen; matches the 10% drift-event threshold, so a correct probe never raises a drift event on its own |
| Online overhead | ≤ 0.5% of step time averaged | a profiled step adds ≤ 2 timestamp ops per segment; at ≈ 2 µs each [estimate] and ≤ 200 segments, ≤ 0.4 ms once per 64 steps; on a 10 ms step that is ≤ 0.06% |
| Link monitoring traffic | ≤ 1 KiB/s per link at idle | heartbeat-derived α plus one 64 B probe per second on idle data-plane links |
| Drift detection latency | ≤ `replan.drift_window` + `probe.drift.sample_window` + 1 s after onset | §7.2 rule |
| Profile size | ≤ 64 KiB per node | §1.4 estimate of ≈34 KiB, with headroom |

**Worked example (scenario a, cost model §4a).** On an RTX 4090 + DDR5-6000 desktop, the probe should report `W_eff(Vram, DenseGemv, Q4_K) ≈ 800 GB/s` and `W_eff(HostPinned, DenseGemv, Q4_K) ≈ 55–85 GB/s` (the example's assumptions, to be replaced by the measurement), pinned H2D ≈ 24 GB/s [measured on one forum system, §2.3], and a CPU batch curve that turns compute-bound at n ≈ 3–9 (§3.1). From those, n* = F_cpu · b / (2 · W_pcie) ≈ 12–24 [estimate], so prefill chunks stream and decode executes in place. If the probe instead found a single-CCD Ryzen at ≤ 60 GB/s, the RAM term would grow from 331 ms to ≥ 387 ms per token [estimate: 23.2 GB / 60 GB/s] and the solver would re-weigh a smaller quant, exactly the sensitivity the cost model warns about (§2.1).


## Alternatives considered

| Alternative | Why not |
|---|---|
| **Spec tables keyed by device name** | η spans 0.24–0.8 of peak by kernel, quant and architecture; DIMM population, CCD links, PCIe slot width and thermal state make spec values wrong in exactly the cases that decide placement (cost model §5.2). Kept only as sanity bounds |
| **Synthetic copy or triad bandwidth as W_eff** | the quantity the solver needs is weight bytes consumed by the kernel that will run; a DDR5 triad at 90 GB/s against a matvec at ≈70 GB/s is a 29% error on the dominant term of example (a). The triad stays as a sanity bound (§3.2) |
| **External tools** (a disk benchmark binary, a network throughput tool) | extra install dependencies, temp files in the working directory, and no access to the real I/O engine or registered buffers. The surveyed system that shelled out to one is listed as an anti-pattern (`multi-device-inference.md` §13 item 19) |
| **Probe inside the daemon** | violates invariant 3: a driver fault during a probe would take down the control plane. The probe runner costs one process spawn (≈ tens of ms) |
| **No startup probe; learn everything online** | the first plan would be solved blind, and a bad first plan costs a full re-shard (hundreds of GB, RFC-0001 §7.2) to undo. Online refinement is kept for what only live traffic shows (hit rates, Wi-Fi tails, throttling) |
| **Install-time calibration of the whole engine** (minutes of model runs) | a surveyed engine's 15–30 minute calibration bought 7% on one machine (`tiered-memory-inference.md` §4). The model probe (≤ 3 s at real shapes) plus online refinement gets the shape-specific numbers without the wait |
| **Continuous background probing while serving** | the probe and the stage would distort each other's measurements and slow the user's stream. Live timings from the stage itself are the cleaner signal |
| **One-way latency from synchronised clocks** | needs PTP-grade sync across consumer machines, which Syntropy cannot assume. RTT/2 plus per-direction β is accurate to within the asymmetry that β already captures |

The ADR this RFC adds (ADR-0100) records the contestable part: measuring W_eff with the real quantized kernel per scheme rather than a single bandwidth per tier.

## Testing and validation

| Claim | Test | Kind |
|---|---|---|
| Parsers read every OS source correctly | recorded fixtures: `/proc/meminfo`, cgroup v1/v2 trees, sysfs device trees (desktop, dual-socket, WSL2, Docker, Raspberry Pi 5), NVML and AMD SMI responses through a fake `dlopen` shim, DXGI/IOKit responses captured on lab machines | unit |
| Device dedupe across providers | the same GPU reported by CUDA, Vulkan and ggml-CUDA with matching UUID/LUID/BDF collapses to one `DeviceInfo` with ordered `backends` | unit, property (random permutations of report order give the same graph) |
| Ordinals are stable | the same fixture enumerated 1,000 times in shuffled order gives identical `DeviceId`/`TierId` | property |
| The probe recovers known values | `testkit` fake devices with configured W, F, t_fixed, PCIe rate, NVMe throttle profile and link α/β/jitter: the probe must recover W and β within 2%, t_fixed within 5%, and the throttled NVMe rate within 5% | unit with simulated devices (RFC-0060) |
| Wireless links are summarised for C14 | `testkit` links with configured hop-time distributions (a lognormal body, a 1% tail at 250 ms, and a 1 s periodic stall): `mean` within 5% of the configured mean once the window holds 1,000 samples; `P(t_hop > mean + 300 ms)` read from `hop_hist` within 0.002 of the configured value; `periodic_stall` set for the periodic link and not for the others; below 1,000 samples the prior tail is used and listed as an assumption | unit, property |
| Battery presence | recorded power-source fixtures for a desktop, a laptop on mains and a laptop on battery: `battery_present` is false, true and true | unit |
| Repeatability | 5 consecutive `Full` runs on each Tier 1 rig; CV ≤ 3% for W_eff | benchmark (RFC-0061) |
| Probe vs steady state | 10 minutes of decode on each reference rig; `abs(W_probe − W_online) ≤ 10%` | benchmark |
| The cost model built on the probe predicts t_tok | SC-30 on the reference rigs (±15%) | benchmark |
| Time budgets | wall time of `Full`, `Confirm` and `Model` on every Tier 1 rig, gated in CI at SC-22 and the §Performance targets | benchmark gate |
| Drift detector is quiet under noise | stationary simulated timings with CV up to 20% and 1% outliers at 10× for 24 simulated hours: zero drift reports | property, simulation |
| Drift detector catches real change | step changes of 25%, 50% and 2× on each quantity: report within `drift_window + sample_window + 1 s`; ramps of 1%/min: report once cumulative change passes 20% | property, simulation |
| Cache invalidation | changing any `ProfileKey` input (driver string, BuildId, power source, VBIOS) forces a re-probe; a 15% injected change in a `Confirm` quantity invalidates; a 5% change does not | unit |
| Probe crash isolation | fault injection: the probe runner aborts in each step; the daemon stays up, the failing devices are marked, other devices' results are used | integration (chaos, RFC-0060) |
| No writes to user data | the probe's file-system activity under `strace`/ETW/`fs_usage` touches only the state and store directories | integration |
| Battery behaviour | on battery, `Full` skips S2 and runs reduced sample counts; switching power source loads the cached profile for the new key | integration on laptops in the lab; unplugging and reconnecting the charger is a **Maintainer action** (ADR-0021) |

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M0 | `ResourceGraph` and profile types (§1); E1, E2, E5 for CPU and CUDA on Linux, the M0 backend (RFC-0001 §16); D1, D2, M1 (n = 1), M3, H1 and S1 plus S2 on the single node, within SC-22 (RFC-0001 §16 M0 lists W_eff, pinned PCIe, NVMe sustained read and t_fixed); profile cache with `ProfileKey` and `Confirm` |
| M1 | Windows and Metal enumeration and probes, as the Metal provider lands (RFC-0040 §7.2); full single-node suite: M2 batch curves in the `Model` probe, M4, M5, M6, H2 with the wake variant, H3, H4; `Model` probe; online refinement (§6) and drift detection (§7) for local quantities; budget watchers (DXGI, NVML, Vulkan, macOS pressure); power-state profiles and `battery_present`; Strix Halo, GB10 and AMD sysfs enumeration; host tiers per NUMA domain with CPU W_eff over both host tiers; H5 lock accounting (`VmLck`/`VmPin`, Windows `lockable_measured`); S1 engine selection with the RFC-0021 `StorageCaps` fields; the `Unified` contention pass; expert-tier shortfall drift |
| M2 | E4 and L1 over QUIC, TCP and shared memory; link histograms from channel stats and heartbeats, with the mean, the compacted histogram and the periodic-stall flag that RFC-0011's C14 reads for wireless links; multi-node graph assembly, `GraphDelta`; co-location confirmation; RDMA over Thunderbolt 5 links in L1 |
| M3 | L2 device-path probe and the pre-gate (§3.5) for cross-node TP/EP; RoCEv2 links; HIP and Vulkan enumeration through their native providers; NPU inventory records |
| M4 | Tier 2/3 enumeration (Android, iOS in-process runner, Linux arm64 boards); probe budgets enforced as CI regression gates on every Tier 1 rig |

## Open questions

1. **Default probe set.** Is `{Q4_K, Q8_0, BF16, MXFP4}` plus the store's schemes the right default, or should it follow the recommended-model list (RFC-0051)? Settled by measuring how often `Derived` values end up in first plans during M1 dogfooding.
2. **S1 before any download.** Resolved by RFC-0021 §10 item 3: the probe never writes to the user's SSD. Open part: how much worse first plans are when storage values are `Derived` because S1 was deferred. Measure in M0 dogfooding; if it matters, run S1 on the first downloaded chunks before the solve.
3. **NVMe temperature without privilege.** Linux hwmon covers kernels with `CONFIG_NVME_HWMON`; Windows and macOS sources vary by driver. How often is `TempSource::None`, and is rate-only throttle detection enough? Measure on the M1 lab machines.
4. **Apple internal SSD sustained read under FileVault** (cost model §6 item 5). **FileVault part answered 2026-10-20 (OQ-IO-01):** FileVault cannot change `W_nvme`. The internal volume is always encrypted, and the AES engine sits in the DMA path, so turning FileVault on only re-wraps the key (Apple Platform Security, checked 2026-10-20). The FileVault on/off A/B is dropped. Still open: the sustained rate and throttling of Apple internal SSDs, which S2's throttle soak measures on one MacBook and one Mac Studio.
5. **Chipset-lane detection. Answered 2026-10-20 (OQ-HW-11):** the heuristic holds if every device behind either chipset die is treated as sharing one PCIe 4.0 x4 uplink (≈ 7.9 GB/s), which §2.2 now does. PLC-E2-T15 desk-checks two boards' published lane maps instead of running a spike.
6. **Wireless tail prior.** `3 × p50 + 10 ms` is a guess. Since 2026-10-20 it sets only the prior tail behind C14's stall rate and the TP/EP gate, not the cost of every hop. RFC-0061 should replace it with a value fitted to home Wi-Fi traces collected in alpha, together with C14's 300 ms and 0.01 defaults.
7. **Windows lockable-memory probe.** H5 follows RFC-0020 §5: it locks and registers 1 GiB ranges up to `max_pinned_fraction` with RFC-0020 §6's responsiveness check between steps. Confirm on Windows 11 machines with 32 GB and 64 GB of RAM in M1 that the probe never makes the desktop unresponsive, and that the cached result stays valid across driver updates of the same major version.
8. **Timestamp op on Metal.** Counter sampling at encoder boundaries may force a profiled step to split encoders, changing its timing. RFC-0040 to measure the perturbation; if > 2%, Metal profiled steps run at a lower rate.

## Proposed amendments to RFC-0001

- **A1. Inventory transport.** `ClusterCommand::Admit { candidate: NodeInfo }` and `ReportProfile` are the only paths from a node to the graph, and neither carries `DeviceInfo`, `TierInfo` or `LinkInfo`. This RFC adds `NodeInfo::inventory` and `CapabilityProfileExt::inventory` (owner-permitted field additions). RFC-0001 §5.2 should say where those records come from, and RFC-0030 should note that `Admit` carries the inventory.
- **A2. VRAM pressure signal.** `MemoryPressure { level, reclaim }` (§3.5) has no tier, so a DXGI or NVML budget drop on one GPU cannot be addressed to the runner holding that GPU. Proposed: `MemoryPressure { level, reclaim, tier: Option<TierId> }`, with `None` meaning host RAM as today.
- **A3. `ReportProfile` cadence.** §5.10 says profiles are reported "at startup and on drift past the threshold only". The model probe (§3.4) adds entries when a model is registered. Proposed wording: "at startup, after a model probe, and on drift past the threshold".
- **A4. W_eff keyed by quant scheme.** §5.2 keys `DeviceCaps::tier_bw` by (tier, kernel class). The cost model requires the actual dequant kernel per quant (§5.3 item 1), and the CPU rate differs by scheme because some CPU kernels are compute-bound (`tiered-memory-inference.md` §5.4). This RFC adds `quant_bw` and keeps `tier_bw` as a per-class summary. Proposed: note in §5.2 that `quant_bw` is the solver's source of truth.
- **A5. Pure budget rule location.** RFC-0020's host budget rule must be callable from `probe` (layer 2) and `solver` (layer 2), which cannot depend on `tier` (layer 2). Proposed: the pure function `budget::host_usable(&HostMemInfo, &ResourcePolicy) -> Bytes` lives in `syntropy-core` alongside the policy objects; §4.1's description of `core` should allow pure policy functions.
- **A6. Drift thresholds.** §7.2 and §14 rule 2 use different numbers (20% and 10%). This RFC reads them as two thresholds with different outputs (§7.1). Proposed: §14 rule 2 should say "raises a drift event" (telemetry), and §7.2 keeps 20% for `ReportProfile` and re-planning.
