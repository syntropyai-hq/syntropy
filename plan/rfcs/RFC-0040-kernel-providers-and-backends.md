# RFC-0040: Kernel providers and backend integration

- Status: Draft
- Area: compute and models
- Depends on: RFC-0001 (§2.3, §4, §5.7, §8, §9), RFC-0010, RFC-0012, RFC-0020, RFC-0022, RFC-0023, RFC-0041, RFC-0042, RFC-0043, RFC-0052, RFC-0060, RFC-0062; ADR-0001, ADR-0002, ADR-0004, ADR-0005, ADR-0012, ADR-0014, ADR-0019, ADR-0020, ADR-0201, ADR-0205; ADR-0400 to ADR-0404
- Date: 2026-10-19
- Revised: 2026-10-20: OS GPU watchdogs (§4.5); the functional doorbell probe (§4.2); the ggml-only Vulkan preview at M2 (§7.5); NVIDIA's libraries fetched at `setup` rather than shipped (§10.3); encoder and reserved ops (§3.1); the router and penalty rules (§7.1); the GPUDirect Storage port not built (§7.7); upstream patches (§8.4); the M0 spike harnesses (Rollout, Open questions); 2026-10-22: one `DeviceClass` in `core`, decided at `enumerate`, and derived certificates on Vulkan (§2.1, §2.2, §6.2, §7.2, §7.3, §7.5; ADR-0405); 2026-10-22: SME2 detection on Linux through `AT_HWCAP2` (§7.3; ADR-0402)

## Summary

Every byte of compute in Syntropy runs through one trait, `KernelProvider` (RFC-0001 §5.7). This RFC fixes the full contract of that trait, the op registry it exposes, and how the executor picks one provider per op per device. It also specifies the six providers: native CUDA 13, Metal, CPU, HIP and Vulkan providers that Syntropy owns, and a ggml provider that links the ggml tensor library for coverage and as the correctness reference. Finally it covers how kernels are built and shipped, and which model families must run on which backend at which milestone.

The decisions that shape the rest:

1. **Per-step values live in device memory, never in launch arguments.** Each step's token IDs, positions, KV lengths, expert selections and cache slot assignments sit in a fixed-address *step descriptor* that the captured step graph reads. One captured graph per (shape class, verify-window size) then serves every step, and the host makes one launch per step (§4.4).
2. **The CPU↔GPU hand-off inside a MoE layer is a doorbell in host-mapped memory.** The GPU writes it with a stream memory operation and waits on the CPU's reply the same way. Where a backend lacks stream memory operations, a one-warp spin kernel stands in. The host never calls a synchronising driver function inside a step (§4.2).
3. **Selection is availability, then parity, then microbenchmark at the plan's real shapes.** A provider must beat the incumbent by a 3 % margin to replace it; ties go to the native provider. Results are cached per (device fingerprint, driver, build) (§6, ADR-0401).
4. **Kernels are compiled ahead of time through a content-addressed object cache** (ADR-0623), and every op carries the hash its `ConformanceManifest` certificate is keyed on. Selection uses only certified kernels (§6.2, ADR-0602). The CPU and Metal providers live in the core. The CUDA, HIP, Vulkan and SYCL providers ship as provider libraries in the backend modules of ADR-0507, behind a versioned C ABI with one exported entry symbol and a lockstep build-ID check (§10.4). The CUDA modules carry only Syntropy code; NVIDIA's libraries are fetched from NVIDIA at `setup` and checked against hashes pinned in the signed module manifest (§10.3). Users never need a compiler (ADR-0400).
5. **CPU kernels use stable `std::arch` for AVX2, AVX-512, AVX-VNNI and NEON.** AMX uses stable `asm!` because its intrinsics are still unstable. SME2 uses a statically linked assembly unit because streaming mode cannot be expressed in Rust. SVE2 gets no kernels in v1 (§7.3, ADR-0402).
6. **The ggml provider wraps one Syntropy op as a small, prebuilt ggml graph** over buffers the tier manager owns, launched on Syntropy's stream where a patched backend allows it (§8, ADR-0403).
7. **Every kernel fits the OS's default GPU watchdog.** Long prefill is split into launches or command buffers within a per-device budget, and host waits end before the watchdog would; Syntropy never changes `TdrDelay` or HAGS (§4.5, ADR-0014 rule 6).

## Motivation and goals / non-goals

Decode time is spent in kernels reading weights and KV (`hardware-and-performance-model.md` §0, §3.4). SC-1 sets the bar: dense GEMV at ≥ 0.80 of peak memory bandwidth on `Vram`/`Unified`, the grouped MoE expert kernel at ≥ 0.63, CPU execute-in-place at ≥ 0.75 of the probed DRAM rate. SC-2 caps the fixed per-token cost at `0.2 · B_ref / W_probe`, about 0.95 ms on an RTX 4090. An existing Metal engine's fitted `t_fixed` is ≈ 5 ms (cost model §2.2), so the host side of kernel dispatch matters as much as the kernels.

ADR-0002 commits Syntropy to its own runtime with kernels behind a trait. Without a precise contract, three things break: whole-step capture (each provider would launch differently), the residency-aware MoE split (the provider would need to know about tiers), and parity (providers would disagree silently).

### Goals

1. A trait precise enough that the executor (RFC-0012) contains no backend-specific code (RFC-0001 invariant 10).
2. Whole-step graph capture on CUDA, HIP and Metal, and pre-recorded command buffers on Vulkan, with no synchronising host call per token (invariant 9).
3. Native hot-op kernels on CUDA, Metal and CPU that meet SC-1 and SC-2, and on HIP and Vulkan by M3.
4. Full op coverage on every backend ggml supports, through one provider, from M0.
5. Selection by measurement, with the ggml provider as the op-level parity reference (ADR-0002 item 6).
6. One binary per OS; GPU kernel payloads loaded only on hosts that can use them.

### Non-goals

- A general tensor library or autograd. Ops exist because a family module (RFC-0043) uses them.
- Runtime kernel compilation or autotuning on the user's machine beyond choosing among prebuilt variants. PTX JIT exists only as the driver's forward-compatibility path (§7.1).
- NPU providers in v1 (`backends-and-portability.md` §1.3.3). The trait admits them later.
- Calling vendor math libraries (cuBLAS, hipBLAS, rocBLAS, hipBLASLt, MPS) from Syntropy's native providers. They allocate on their own and resist capture with Syntropy's buffers. ggml's CUDA and HIP backends still link cuBLAS/cuBLASLt and hipBLAS/rocBLAS (§8.1). The `rocm` module ships hipBLAS and rocBLAS, whose licences permit it; the `cuda13` module carries only Syntropy code, and `setup` fetches cuBLAS and cuBLASLt from NVIDIA beside its provider library (§10.3). Only ggml ops reach these libraries, under §8.2's capture rule. They may also appear as benchmark baselines in RFC-0061.
- Cross-node collectives. RFC-0032 builds them from `Transport` plus `copy`; this RFC provides intra-node collectives only.

## Design

### 1. Crates and source layout

| Crate | Contents | Languages | Ships in (ADR-0507, RFC-0052 §1) |
|---|---|---|---|
| `syntropy-kernels` | the trait, `OpKind`/`OpKey` registry, argument types, selection engine, microbenchmark harness, F32 reference semantics for every op (`kernels::reference`), the provider-library ABI's `#[repr(C)]` types and the `export!` macro that module crates expand (`kernels::abi`, §10.4). `#![forbid(unsafe_code)]` | Rust only | the core binary |
| `syntropy-kernels-loader` | the core side of the provider-library ABI: loading a module, §10.4's checks and the `ModuleProvider` proxy that calls through C function pointers; the driver census (§10.5), which loads vendor driver libraries at runtime for read-only enumeration. Layer 2, beside the provider crates (RFC-0001 §4.1). It is the only core crate that needs `unsafe` for the ABI, and the only one that touches a vendor driver library, so RFC-0001 §4.2 rules 6 and 8 name it (amendment 6) | Rust only | the core binary |
| `syntropy-kernels-cpu` | CPU provider; one module per ISA level; AMX microkernels in `asm!`; SME2 microkernels in `src/sme2/*.S` | Rust, AArch64 assembly | the core binary, every platform |
| `syntropy-kernels-metal` | Metal provider; `msl/*.metal`, compiled to two metallibs (Metal 4 and Metal 3.2 language revisions) embedded with `include_bytes!` | Rust, MSL | the core binary on macOS; the app on iOS |
| `syntropy-kernels-cuda` | CUDA provider (Rust); kernel sources `csrc/*.cu`; CUTLASS/CuTe GEMM launch shim `shim/*.cu`. Built, together with the ggml CUDA provider, as the provider library `libsyntropy_provider_cuda` | Rust, CUDA C++ | module `cuda13`; module `cuda12-legacy` from the same sources with the legacy profile |
| `syntropy-kernels-hip` | HIP provider; `hsrc/*.hip`, sharing headers with `csrc` through `common/portability.h`; built with the ggml HIP provider as `libsyntropy_provider_rocm` | Rust, HIP C++ | module `rocm` |
| `syntropy-kernels-vulkan` | Vulkan provider; `slang/*.slang` compiled to SPIR-V variants embedded in the provider library; built with the ggml Vulkan provider as `libsyntropy_provider_vulkan` | Rust, Slang | module `vulkan` |
| `syntropy-kernels-ggml` | ggml provider code; vendored ggml at a pinned commit (`third_party/ggml.lock`, RFC-0062 §4.2); `bindgen` bindings. Linked statically, with its backend, into each artefact that uses it (§8.1) | Rust, C (vendored) | the core (ggml CPU; ggml Metal on macOS); every module (its ggml GPU backend); module `sycl` (ggml SYCL only, no native provider) |

On iOS and Android every provider the platform uses is linked statically into the app, because neither platform allows loading downloaded code (ADR-0507 rule 1). The core binary never contains a vendor GPU runtime: the CUDA, HIP and Vulkan providers exist only inside their modules, which only runner and probe processes load (RFC-0001 §4.2 rule 4).

Rules beyond RFC-0001 §4.2:

- **Kernel argument structs have one source of truth.** Each GPU provider keeps a C header (`csrc/params.h`, `msl/params.h`, `slang/params.slang`) that declares every kernel's parameter block. `build.rs` runs `bindgen` (C, MSL) or a small generator (Slang) to emit `#[repr(C)]` Rust mirrors, plus a static layout test that compares `size_of`/`offset_of` against values the kernel compiler reports.
- **No allocation, scheduling or residency logic in C++, MSL or Slang.** The CUTLASS shim builds parameter structs and calls a launch function; that is all (ADR-0001 decision 3).
- **`kernels::reference`** implements every `OpKind` in plain Rust, F32 accumulation, fixed reduction order. It is the semantic definition of the op (§3.3), the oracle for property tests, and the op semantics that `syntropy_graph::reference` interprets for the reference forward pass (ADR-0019, RFC-0042 §7, RFC-0060 §4.1).

### 2. The `KernelProvider` contract

#### 2.1 The trait

RFC-0001 §5.7 fixes the trait's existing methods and their meaning. This RFC keeps every one of them unchanged and adds methods with default bodies, as RFC-0001 §5 allows. Additions are marked `// +0040`.

```rust
pub trait KernelProvider: Send + Sync {
    fn id(&self) -> ProviderId;
    fn backend(&self) -> Backend;
    /// +0040. Build identity: provider version, kernel-module BLAKE3, compiler versions,
    /// driver or runtime version seen at load. Part of every selection-cache key (§6.5).
    fn build(&self) -> ProviderBuild;
    /// +0040. Providers that can share buffers, streams and events on one device without
    /// copies form one memory domain (§2.3). Selection never mixes domains on a device.
    fn domain(&self) -> MemoryDomain;

    fn enumerate(&self) -> Result<Vec<ProviderDevice>, Error>;
    fn open(&self, dev: DeviceId, opts: &OpenOptions) -> Result<DeviceCtx, Error>;
    /// +0040. Free and budgeted memory now (DXGI budget, VK_EXT_memory_budget, cuMemGetInfo,
    /// recommendedMaxWorkingSetSize). Never called on a hot path.
    fn memory_info(&self, ctx: &DeviceCtx) -> Result<DeviceMemoryInfo, Error> { Err(unsupported()) }

    fn ops(&self) -> &OpRegistry;
    fn query(&self, ctx: &DeviceCtx, op: &OpKey, shape: &ShapeSig) -> Option<OpCaps>;

    fn alloc(&self, ctx: &DeviceCtx, req: AllocRequest) -> Result<DeviceBuffer, Error>;
    fn free(&self, ctx: &DeviceCtx, buf: DeviceBuffer);
    /// +0040. Back part of a `reserve_only` allocation with a physical segment, or release one.
    /// Addresses inside the reservation never move (elastic expert cache, RFC-0001 §9.3;
    /// RFC-0020 §4.1 names and uses these two methods).
    fn map_segment(&self, ctx: &DeviceCtx, reservation: &DeviceBuffer, offset: Bytes, bytes: Bytes) -> Result<Segment, Error> { Err(unsupported()) }
    fn unmap_segment(&self, ctx: &DeviceCtx, seg: Segment) -> Result<(), Error> { Err(unsupported()) }
    fn register_host(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<Registration, Error>;
    /// +0040. A `Send` capability that registers and unregisters host ranges from the
    /// housekeeping thread, so progressive pinning (RFC-0020 §6) never blocks the dispatcher.
    /// CUDA and HIP implement it by making the device's primary context current on the
    /// calling thread; Metal by creating no-copy buffers; Vulkan by importing host memory.
    fn host_registrar(&self, ctx: &DeviceCtx) -> Option<Box<dyn HostRegistrar>> { None }
    fn import(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<DeviceBuffer, Error>;
    /// +0040. Residency hint for backends with explicit residency (Metal residency sets,
    /// Vulkan priority). Default no-op: on CUDA and HIP every mapped allocation is resident.
    fn set_resident(&self, ctx: &DeviceCtx, bufs: &[&DeviceBuffer], resident: bool) -> Result<(), Error> { Ok(()) }
    fn repack(&self, op: &OpKey, src: &[u8], dst: &mut [u8]) -> Option<Result<LayoutVersion, Error>>;
    fn repacked_len(&self, op: &OpKey, src: Bytes) -> Option<Bytes>;

    fn stream(&self, ctx: &DeviceCtx, prio: StreamPriority) -> Result<Stream, Error>;
    fn event(&self, ctx: &DeviceCtx) -> Result<Event, Error>;
    fn record(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error>;
    fn wait(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error>;
    fn signal_host(&self, ctx: &DeviceCtx, s: &Stream, flag: HostFlag, value: u32) -> Result<(), Error>;
    /// +0040. Device-side wait until the host-mapped `flag` satisfies `cmp` against `value`.
    /// The other half of the doorbell (RFC-0001 §8.2). Unsupported means the executor splits
    /// the step at this point into two submissions (§4.2).
    /// `reset`, when set, is written to the flag by the device right after the wait passes: the
    /// consumer reset of §4.2's handshake. In captured regions `value` and `reset` are constants.
    fn wait_host(&self, ctx: &DeviceCtx, s: &Stream, flag: HostFlag, value: u32, cmp: WaitCmp, reset: Option<u32>) -> Result<(), Error> { Err(unsupported()) }
    /// +0040. Device-triggered wake for a host thread parked inside a step (RFC-0012 §3.3): after
    /// the work enqueued so far on `s` (normally a doorbell write), signal `waker` without the
    /// stream waiting for the signal. Captured like any other node. Unsupported means the
    /// waiter spins (§4.2).
    fn arm_wake(&self, ctx: &DeviceCtx, s: &Stream, waker: &HostWaker) -> Result<(), Error> { Err(unsupported()) }
    /// +0040. Non-blocking completion poll. Used by dispatchers instead of synchronising.
    fn poll(&self, ctx: &DeviceCtx, e: &Event) -> Result<bool, Error>;
    /// +0040. Device time between two completed events, for microbenchmarks and telemetry.
    fn elapsed(&self, ctx: &DeviceCtx, start: &Event, end: &Event) -> Option<Nanos> { None }
    /// +0040. Blocking drain of a stream. Allowed only outside `Ready`; debug builds assert.
    fn synchronize(&self, ctx: &DeviceCtx, s: &Stream) -> Result<(), Error>;
    fn copy(&self, ctx: &DeviceCtx, s: &Stream, src: BufferSlice, dst: BufferSlice) -> Result<(), Error>;
    /// +0040. Many copies in one submission (cudaMemcpyBatchAsync on CUDA 13; a loop elsewhere).
    fn copy_batch(&self, ctx: &DeviceCtx, s: &Stream, pairs: &[(BufferSlice, BufferSlice)]) -> Result<(), Error> {
        for (src, dst) in pairs { self.copy(ctx, s, *src, *dst)?; } Ok(())
    }

    fn prepare(&self, ctx: &DeviceCtx, op: &OpKey, shape: &ShapeSig) -> Result<PreparedOp, Error>;
    fn launch(&self, ctx: &DeviceCtx, s: &Stream, op: &PreparedOp, args: &OpArgs) -> Result<(), Error>;
    fn begin_capture(&self, ctx: &DeviceCtx, s: &Stream) -> Result<(), Error>;
    fn end_capture(&self, ctx: &DeviceCtx, s: &Stream) -> Result<CapturedGraph, Error>;
    fn replay(&self, ctx: &DeviceCtx, s: &Stream, g: &CapturedGraph) -> Result<(), Error>;

    fn collectives(&self) -> Option<&dyn CollectiveOps> { None }
    fn health(&self, ctx: &DeviceCtx) -> DeviceHealth;
}
```

`unsupported()` builds `Error { class: Unsupported, code: (Kernels, 1), .. }`. `poll` and `synchronize` have no defaults because every backend can implement them.

**The device class is decided at `enumerate`** (ADR-0405 decision 3, 2026-10-22). `enumerate` returns `ProviderDevice::class`, so every field of the class is known before `open`. Each provider computes the class with one pure function of the device's reported properties, the OS facts, the AMX permission result and `features.disable`. `open` applies the same function to the same inputs and gets the same class. It never loads a different variant set or function table than the class names. If `open` cannot obtain what the class assumes, it fails with an error that names the mismatch, and the node re-probes.

#### 2.2 Supporting types

```rust
pub struct ProviderId(pub Cow<'static, str>);      // "cuda", "metal", "cpu", "hip", "vulkan", "ggml-cuda", …
pub struct ProviderBuild { pub version: u32, pub module: Option<Blake3>, pub toolchain: SmallVec<[(Cow<'static, str>, String); 3]>, pub driver: Option<String> }
pub enum MemoryDomain { Cuda, Hip, Metal, Vulkan, Host, Ggml(GgmlBackend) }

/// One device as a provider sees it. The probe (RFC-0010) merges records from several
/// providers into one DeviceInfo by `uuid`, then PCI address, then (vendor, name, memory).
pub struct ProviderDevice {
    pub local: u32,                         // provider-local index
    pub uuid: Option<[u8; 16]>,             // CUDA/HIP UUID, VkPhysicalDeviceIDProperties, MTLDevice registryID hashed
    pub pci: Option<PciAddress>,
    pub kind: DeviceKind,
    pub name: String,
    pub class: DeviceClass,
    pub features: DeviceFeatures,
    pub memory: DeviceMemoryInfo,
    pub numa: Option<u16>,
}

/// syntropy-core types (ADR-0405): `kernels` re-exports DeviceClass, CpuIsa and DeviceFeatures and
/// defines none of its own. Part of PackKey (RFC-0001 §5.6), OpCert and SelectionKey (§6.5). Two
/// devices with equal classes run the same kernels and accept the same pack. A field belongs in the
/// class only when a change in it can change the pack layout or the set of kernel objects the
/// providers load for the device; other facts stay in DeviceInfoExt (RFC-0010 §1.2).
pub enum DeviceClass {
    Cuda { sm: u16 },                       // 75, 80, 86, 89, 90, 100, 120, 121, …
    Hip { gfx: u32 },                       // 0x1100, 0x1201, 0x942, …
    Metal { family: AppleGpuFamily, metal4: bool },
    /// `subgroup`: the subgroup size of the shader variant set the provider will load, 32 or 64.
    /// Computed at `enumerate` from the subgroup size control properties (minSubgroupSize,
    /// maxSubgroupSize, the required-size stages), never copied from a reported default: 32 if the
    /// device can run the variant set at 32, else 64 if it can at 64, else the native Vulkan
    /// provider does not claim the device. A device that offers both is class 32. `coopmat` is
    /// computed the same way, from the cooperative-matrix properties and features.disable.
    Vulkan { vendor: u32, coopmat: bool, subgroup: u8 },
    Cpu { isa: CpuIsa },
    Npu { vendor: u32, model: u32 },        // RFC-0010's addition; no provider produces it in v1
}
pub enum CpuIsa { X86V2, X86V3, X86V4, X86V4Amx, Arm80, Arm82Dot, Arm86I8mm, Arm92Sme2 }   // §7.3

bitflags! { pub struct DeviceFeatures: u64 {
    // GPU matrix and numeric paths
    const FP4_TC = 1 << 0;  const FP8_TC = 1 << 1;  const INT8_TC = 1 << 2;  const BF16_MMA = 1 << 3;
    const FP16_MMA = 1 << 4; const COOP_MATRIX = 1 << 5; const COOP_MATRIX2 = 1 << 6; const INT_DOT = 1 << 7;
    const METAL4_TENSOR = 1 << 8; const NEURAL_ACCEL = 1 << 9;
    // GPU execution and memory
    const GRAPH_CAPTURE = 1 << 16; const STREAM_MEM_OPS = 1 << 17; const VMM = 1 << 18; const HOST_REGISTER = 1 << 19;
    const HOST_IMPORT = 1 << 20; const P2P = 1 << 21; const UNIFIED = 1 << 22; const RESIDENCY_SETS = 1 << 23;
    const MEMORY_BUDGET = 1 << 24; const BATCHED_COPY = 1 << 25; const SHARED_EVENTS = 1 << 26; const REBAR = 1 << 27;
    // CPU ISA
    const AVX2 = 1 << 32; const AVX512 = 1 << 33; const AVX512_VNNI = 1 << 34; const AVX512_BF16 = 1 << 35;
    const AVX512_FP16 = 1 << 36; const AVX_VNNI = 1 << 37; const AMX_INT8 = 1 << 38; const AMX_BF16 = 1 << 39;
    const NEON_DOTPROD = 1 << 40; const NEON_I8MM = 1 << 41; const NEON_BF16 = 1 << 42; const SVE2 = 1 << 43;
    const SME2 = 1 << 44; const NEON_FP16 = 1 << 45;
}}

pub struct DeviceMemoryInfo {
    pub total: Bytes, pub free: Bytes,
    pub budget: Option<Bytes>,              // OS-granted budget where one exists (DXGI, Vulkan, Metal working set)
    pub host_visible_device_local: Option<Bytes>, // resizable-BAR window
    pub max_alloc: Bytes, pub unified: bool,
}

pub struct OpenOptions {
    pub streams: StreamPlan,                // per priority: compute, copy-in, copy-out, collective
    pub doorbell_slots: u32,                // host flags to pre-register (≥ MoE layers × 2 + 8)
    pub capture: bool,                      // false: kill switch features.disable = ["graph_capture"]
    pub deterministic: bool,                // ADR-0019 decision 8: deterministic kernel variants only
    pub power: PowerHint,                   // Mains | Battery: spin budgets, clock hints
}

/// Exists only on the device's dispatcher thread. Holds the backend context and the
/// provider's per-device state (module handles, pipeline caches, doorbell page).
pub struct DeviceCtx { inner: Box<dyn Any>, _not_send: PhantomData<*const ()> }

pub enum StreamPriority { Compute, CopyIn, CopyOut, Collective, Low }
pub enum WaitCmp { Eq, Geq }
/// A 4-byte slot in a doorbell page: pinned, host-mapped and registered with the device at
/// `open`. Visible to the host at `host_ptr` and to the device at a device address.
pub struct HostFlag { pub page: u16, pub slot: u16 }
/// A wake target for a host thread parked inside a step: a futex word, eventfd or Windows event
/// that RFC-0012's waiter announced. Created at `Warming`; its address is fixed for the epoch.
pub struct HostWaker { pub kind: WakeKind, pub handle: u64 }
pub enum WakeKind { Futex, EventFd, WinEvent, TimelineValue }
pub enum DeviceHealth { Ok, Throttled { reason: ThrottleReason }, Sticky { code: ErrorCode } }
/// RFC-0020 §6. Registrations made through it are valid for every stream of the device.
/// Units follow RFC-0020 §6.1 (aligned to extent, slot and elastic-segment boundaries).
/// Until the M1 spike settles RFC-0020 open question 9, every DMA registration completes
/// before `Ready`; none is issued while steps run.
pub trait HostRegistrar: Send {
    fn register(&self, region: HostRegion, flags: HostRegFlags) -> Result<Registration, Error>;
    fn unregister(&self, reg: Registration) -> Result<(), Error>;
}
bitflags! { pub struct HostRegFlags: u8 { const DMA = 1; const MAPPED = 2; } }   // the tier manager maps its Registrations onto these
pub struct Segment { pub reservation: u64, pub offset: Bytes, pub bytes: Bytes, pub handle: u64 }
```

*Note 2026-10-23 (#1525):* `ProviderId` and the toolchain names in `ProviderBuild` were `&'static str`. §6.5 postcard-encodes `SelectionKey` and `SelectionEntry`, which hold them, to `<state>/kernels/selection.v1`. Decoding a `&'static str` needs a buffer that lives for the whole program, so every cache load would have leaked one. `Cow<'static, str>` decodes owned, and a provider still names itself with a literal without allocating. A `Cow` cannot be `Copy`, because its owned form holds a `String`, so `ProviderId` departs from RFC-0001 §5's default that IDs are `Copy`: it is `Clone` only. RFC-0001 §5.12 makes the same change to `Error::context` (RFC-0001 §18.32). COMP-E13-T15 makes the change in code.

*Revised 2026-10-22 (ADR-0405):* `DeviceClass`, `CpuIsa` and `DeviceFeatures` live in `syntropy-core`, and `kernels` re-exports them (PLC-E1-T16). RFC-0010 §1.2 had its own `DeviceClass` with other variants, and now points here. `AppleFamily` is renamed `AppleGpuFamily`, the name `core` already uses. `Npu` comes from RFC-0010. `ProviderDevice` is unchanged.

#### 2.3 Memory domains and device ownership

Several providers can drive one physical GPU: `cuda` and `ggml-cuda`, or `vulkan` and `ggml-vulkan`, or even `cuda` and `vulkan`. Buffers, streams and events are shareable only inside a memory domain:

| Domain | Providers in it | Shared objects |
|---|---|---|
| `Cuda` | `cuda`, `ggml-cuda` (patched, §8.4) | primary context, `CUdeviceptr`, `CUstream`, `CUevent` |
| `Hip` | `hip`, `ggml-hip` (patched) | as CUDA |
| `Metal` | `metal`, `ggml-metal` | `MTLBuffer`, `MTLSharedEvent`, command queue |
| `Vulkan` | `vulkan`, `ggml-vulkan` (where import works, §8.3) | `VkBuffer` on one `VkDevice`, timeline semaphores |
| `Host` | `cpu`, `ggml-cpu` | host pointers |
| `Ggml(b)` | `ggml-sycl`, `ggml-opencl`, `ggml-cann`, … | ggml buffers only |

Rule: for each device in a plan, selection first picks **one domain** (§6.4), then picks a provider per op inside it. A device is never driven by two domains in one runner. This keeps one allocator and one queue per device, which the tier manager and graph capture both need.

#### 2.4 Threading rules

- `enumerate`, `ops`, `repack`, `repacked_len`, `build` and `domain` are callable from any thread. Packing runs `repack` on the runner's pack-builder threads (RFC-0041 §5).
- Every other method takes `&DeviceCtx`, which is `!Send`; it can be called only on the device's dispatcher thread, or on a CPU-pool thread for the `cpu` provider.
- After `Warming`, the dispatcher may call only `replay`, `launch` (on providers without capture), `record`, `wait`, `poll`, `signal_host`, `wait_host`, `copy`, `copy_batch` and `health` inside a step (`GdsPort::read_async`, §7.7, is not built in v1). Each of these must be allocation-free and lock-free in the provider, and must not make a synchronising driver call. `set_resident`, `map_segment` and `unmap_segment` may make driver calls that synchronise, so they run only between steps, when a `PlanDelta` or a memory-pressure command is applied (RFC-0001 §9.3, §10.3). CI enforces both rules with a counting allocator and a driver-call shim in `syntropy-testkit` (§Testing).

### 3. Op registry, op semantics and arguments

#### 3.1 Op kinds

`OpKey` is fixed by RFC-0001 §5.7: `{ op: OpKind, schemes: SmallVec<[QuantScheme; 3]>, layout: LayoutVersion, variant: u16 }`. `schemes` lists, in order, the weight scheme, the activation dtype and the output dtype. `layout` names the weight byte layout: `LayoutVersion::SOURCE` (0) for the source format's own layout, or a provider repack ID (§5.4). `variant` is a bitmask of **fusions** (`FUSE_BIAS`, `FUSE_RESIDUAL`, `FUSE_ACT`, `FUSE_ROPE`, `FUSE_NORM_IN`, `FUSE_SCALE_OUT`). Tile sizes and other tuning choices are private to a provider and chosen in `prepare`; they are never part of the key.

```rust
pub enum OpKind {
    // Weight-bound linear algebra
    Gemv,                         // rows ≤ 16; decode, verify windows, small batches
    Gemm,                         // rows > 16; prefill chunks, large batches
    GateUpAct { act: GatedAct },  // fused gate and up projections with SiLU/GELU gating
    MoeRoute { norm: RouteNorm }, // router logits → top-k ids and weights (softmax, sigmoid, grouped top-k, bias)
    MoeGrouped,                   // residency-aware grouped expert FFN (§3.4)
    MoeCombine,                   // weighted sum of per-expert outputs in ascending expert index, plus shared experts
    // Attention and sequence mixers
    AttnDecode(AttnKind),         // ≤ 16 query rows per sequence against paged KV
    AttnPrefill(AttnKind),        // causal chunk against paged KV; for Bidirectional, the whole input with no mask
    AttnPartial(AttnKind),        // attention over a subset of pages, returning (o, m, l) unnormalised (§3.6; RFC-0022's attention_partial)
    AttnMerge,                    // exact log-sum-exp merge of partials in a fixed order (§3.6)
    AttnIndex(IndexKind),         // indexer scores and top-k block, token or compressed-entry selection
    KvWrite(AttnKind),            // append K/V or latents into KV pages, RoPE applied where the family says
    ScanDecode(ScanKind),         // one-token recurrent state update
    ScanPrefill(ScanKind),        // chunked parallel scan over a prefill chunk
    CausalConv1d,                 // short convolution in Mamba- and DeltaNet-class layers
    // Normalisation and elementwise
    RmsNorm, LayerNorm, Rope(RopeKind), Act(ActKind), Add, Mul, Scale, HyperMix,
    // Tables, head and sampling
    RowGather,                    // embedding and n-gram rows by id
    EmbedInject,                  // reserved (RFC-0043 §2.4): encoder rows written over placeholder rows; reference only in v1
    Pool(PoolKind),               // embedding epilogue: mean, first or last row of each sequence, optional L2 norm (RFC-0043 §5.4)
    Logits,                       // final norm plus LM head (a Gemv/Gemm with vocab-sized output)
    LogitsProcess,                // temperature, penalties, grammar mask, logit bias
    TopKTopP, SampleSeeded, Argmax,
    VerifyAccept,                 // exact acceptance over a verify window or tree (RFC-0012)
    // Data movement
    Convert,                      // dtype conversion, including dequantisation
    GatherRows, ScatterRows,      // KV compaction, migration, prefix-cache copy-on-write
    KvPack, KvUnpack,             // layer-first device pages ↔ page-first host/NVMe records (RFC-0022 §3; §5.5)
    PartialReduce,                // sums per-site partial outputs in a fixed order
    ApplyTableDeltas,             // first node of every step: applies posted slot-table and KV block-table deltas (§3.4); RFC-0020/0023's apply_deltas
    Timestamp,                    // writes the device clock into a slot of a host-mapped timing buffer (§3.7)
    TapGather,                    // parity taps: candidate log-probs at reference top-K ids, logsumexp, argmax, margin (§3.8)
    HiddenTap,                    // parity variant only: copies tapped rows' residual stream at a layer boundary (§3.8)
}
pub enum AttnKind { Gqa, Mla, SlidingGqa { sinks: bool }, BlockSparse, TopKSparse,
                    Compressed { sinks: bool },  // DeepSeek-V4-class: raw window plus compressed entries, one softmax (revised 2026-10-23)
                    Bidirectional }       // encoder-only models: every row sees every key of its sequence (RFC-0043 §5.4)
pub enum PoolKind { Mean, Cls, LastToken }
pub enum IndexKind { BlockTopK, TokenTopK, CompressedTopK }
pub enum ScanKind { GatedDeltaNet, Kda, Mamba2 }
```

The list is closed per release. A family module (RFC-0043) may only use registered kinds; adding a kind needs a reference implementation, a ggml mapping or an explicit "native only" mark, and parity tests (§11.4). The 2026-10-20 additions follow that rule: `AttnKind::Bidirectional` and `Pool` serve the encoder-only embedding family from M2, with CPU and CUDA kernels; `EmbedInject` is reserved for multimodal input after 1.0 and has only its `kernels::reference` semantics in v1, which no provider is asked to implement. `OpKind` lines up with RFC-0001's `KernelClass` for profiling: `Gemv`, `GateUpAct` and `Logits` profile as `DenseGemv`; `MoeGrouped` as `ExpertGemv`; `AttnDecode` as `Attention(kind)`; `RowGather` as `RowGather`; `Convert`, `GatherRows` and `ScatterRows` as `Copy`.

*Revised 2026-10-23 (COMP-E16-T18; OQ-MF-16 (a)–(c)): compressed attention.* `Compressed` follows DeepSeek-V4's reference code, `inference/model.py` and `inference/kernel.py` in `deepseek-ai/DeepSeek-V4-Pro` at revision `b5968e9190ef611bbf34a7229255be88a0e937c1`, which COMP-E16-T15 read into RFC-0043 open question 2.
- **The sink.** `AttnKind::Compressed { sinks }` carries a sink flag, as `SlidingGqa { sinks }` does. With it set, a learned per-head logit sits inside the one softmax over the raw window and the entries the layer reads (`kernel.py` line 346). Every published V4 layer sets it.
- **The indexer.** `AttnIndex(CompressedTopK)` scores formed compressed entries, not tokens, and runs only on layers whose ratio has an indexer: r = 4 in the published configurations. A layer with r = 128 reads every formed entry and has no indexer. A layer with r = 0 forms no entries and reads its raw window alone. RFC-0043 §3.1 gives each layer kind's lowering.

#### 3.2 Shapes and buckets

```rust
pub struct ShapeSig {
    pub class: ShapeClass,          // Decode | Verify { window: u16 } | Batch { rows: u32 } | Prefill { chunk: u32 } | PrefillStream { chunk: u32 }
    pub rows: u32,                  // the bucket: the most token rows the prepared op handles
    pub dims: SmallVec<[u32; 6]>,   // op-specific static dims, in a fixed order per OpKind
}
```

`dims` per kind, for example: `Gemv`/`Gemm` `[n_out, k_in]`; `MoeGrouped` `[n_experts, top_k, hidden, inter]`; `AttnDecode(Gqa)` `[n_q_heads, n_kv_heads, d_head, page_tokens]`; `AttnDecode(Mla)` `[n_heads, d_latent, d_rope, page_tokens]`; `ScanDecode` `[n_heads, d_k, d_v]`.

**Rows are bucketed, and the active count is dynamic.** A prepared op handles any `rows_active ≤ rows`, read from the step descriptor (§3.5). The bucket set is RFC-0012 §2's: `{1}` for decode, `γ_max + 1` (or the tree size) for verify windows, `Batch` buckets in throughput mode only, a prefill ladder at {64, 256, 1024, 4096} tokens below the plan's `prefill_chunk` plus `prefill_chunk` itself, and `PrefillStream { prefill_chunk }` when a group streams at that chunk. One captured graph per bucket serves every step in it. A bucket wastes at most the padded rows' compute on compute-bound ops; memory-bound ops read weights once whatever the row count, so padding costs nothing there.

`query` answers with `OpCaps`:

```rust
pub struct OpCaps {
    pub workspace: Bytes,                // scratch the prepared op needs (§5.3)
    pub est: Option<Nanos>,              // the provider's own estimate; ordering hint only (§6.3)
    pub layout: LayoutVersion,           // weight layout the op reads (§5.4)
    pub capture_safe: bool,              // §4.4 rule 5
    pub deterministic: bool,             // §3.3
    pub numerics: Numerics,              // RFC-0042 §4: accumulation, activation quantisation, output dtype
    pub kernel_hash: Blake3,             // the hash its OpCert is keyed on (§10.2; RFC-0060 §6.4)
    pub source_key: Blake3,              // ADR-0623 SourceKey of its translation unit (§6.2, derived certificates)
}
```

#### 3.3 Op semantics

- **The definition of an op is its function in `kernels::reference`.** Each takes F32 inputs (weights dequantised exactly per RFC-0042 §2) and computes with F32 accumulation in a fixed order. Doc comments state the mathematical definition, edge cases (empty rows, `rows_active = 0`, masked positions, NaN propagation) and the accumulation contract.
- **Numerics contract.** RFC-0042 §4 fixes the accumulation precision per op and dtype class: for example, quantized GEMV accumulates in at least F32 (integer-dot paths accumulate exactly in I32 within a block, then scale in F32), attention keeps the softmax max and sum in F32, and scans keep their state in F32. A provider that accumulates in lower precision for an op is rejected by `query`'s `numerics` check (§6.2), not tolerated.
- **Deterministic variants.** When `OpenOptions::deterministic` is set, `prepare` must return a variant with a fixed reduction order that does not depend on the launch geometry or on timing (no split-K with atomics, no non-deterministic scatter-add). Each provider marks such variants in `OpCaps::deterministic`. This backs ADR-0019's `deterministic` option.

#### 3.4 The residency-aware grouped MoE op

This op is the kernel half of the three-site expert split (RFC-0001 §6.3 step 3; ADR-0002). RFC-0023 owns the policy, the slot table, the doorbell layout and the split rule (RFC-0023 §2, §5.4, §6, §8); RFC-0012 owns the thread roles. This section is the provider contract that implements them: what the kernels read, the one split function every site evaluates identically, and the order of operations in the captured graph.

**Device-resident tables, per device cache** (RFC-0023 §5.4):

| Table | Contents | Changed by |
|---|---|---|
| `slot_of[layer][expert] → u32` | cache slot index, or `MISS = u32::MAX` | table deltas (below) |
| `slot_addr[slot] → u64` | device address of each slot; fixed for the plan epoch, because slots are striped across elastic segments (RFC-0020 §4.1, §8.3) | never within an epoch |
| `copy_eligible[layer]` bitmap | 1 when a missed expert's current host copy is DMA-registered pinned memory, so the copy engine may fetch it (RFC-0020 §6) | table deltas |
| `copy_src[layer][expert] → u64` | host address of that copy, for the dispatcher's copies (host mirror only) | the dispatcher |
| balance block (in the step descriptor, §3.5) | per MoE layer, `B_copy(l)` in bytes: the copied-miss byte budget for this step, chosen by RFC-0023 §8's balancer | written before each launch |

**Doorbell.** The router op writes RFC-0023's `DoorbellLayer` for layer l into host-mapped memory: the step number (read from `StepDesc.step`; a payload check the host asserts, not the flag), the distinct experts selected in the pass, each as `RouteSlot { expert, tokens, cache_slot }`, and the predictions for later layers. It then raises the route flags `Rd[l]` (the dispatcher's) and `Rc[l][d]` (one per NUMA domain's CPU pool) with `signal_host`, using the constant-valued handshake of §4.2.

**Split function** (`kernels::moe::assign`; RFC-0023 §6's rule, implemented bit-identically in Rust, CUDA C++, HIP, MSL and Slang, and checked by a cross-implementation property test):

```text
input:  the layer's RouteSlot list; copy_eligible[l] at the step's table version; B_copy(l); per-expert bytes s_l (uniform within a layer)
misses  = route entries with cache_slot == MISS
order   = misses sorted by tokens descending, then expert index ascending
copied  = []; bytes = 0
for e in order:
    if copy_eligible[l][e] and bytes + s_l ≤ B_copy(l): copied.push(e); bytes += s_l
in_place = misses − copied
output: HIT for cache_slot ≠ MISS; COPY with copy-slot index i for the i-th expert of `copied`; CPU otherwise
```

The dispatcher, the CPU pools and the GPU's second expert kernel evaluate this function from the same doorbell and balance block, so no copy plan travels between them (RFC-0023 §6). The host sites evaluate `copy_eligible` from a host mirror held at exactly the step's table version, the `delta_head` the dispatcher wrote into that step's descriptor, never at the dispatcher's latest posted state; the dispatcher never retires a copy source before the last step that saw it as eligible has completed (RFC-0023 §5.4).

**One MoE layer on a GPU device, as captured in the step graph:**

| # | Stream | Operation | Writes |
|---|---|---|---|
| 1 | compute | router op (`MoeRoute`), fused with next-layer prediction where the family has it | the doorbell, plus ids and weights in device memory |
| 2 | compute | `signal_host` on the dispatcher's route flag `Rd[l]` and on each NUMA domain's route flag `Rc[l][d]` (constant 1, §4.2) | route flags |
| 3 | compute | `MoeGrouped` over `HIT` experts, reading `slot_addr[slot_of[l][e]]`, plus shared experts | per-(row, expert) outputs, F32 |
| 4 | copy-in | (issued by the dispatcher on seeing `Rd[l] = 1`, which it resets to 0; RFC-0012) `copy_batch` of the `COPY` experts from `copy_src` into layer l's half of the double-buffered copy slots, then `signal_host(F[l])` | copy slots, copy flag |
| 5 | compute | `wait_host(F[l] = 1)` and the reset of `F[l]` to 0, then `MoeGrouped` over `COPY` experts, reading copy slots | per-(row, expert) outputs |
| 6 | host | each NUMA domain's CPU pool sees `Rc[l][d] = 1` and resets it, computes its `CPU` experts in place (after any demand read lands), writes F32 outputs to its slice of a host-mapped block, and stores `Q[l][d] = 1` (release) | CPU outputs, CPU flags |
| 7 | compute | `wait_host(Q[l][d] = 1)` and its reset, for every domain d, then `MoeCombine` | the layer output |

`MoeCombine` adds, per row, the routed experts' outputs weighted by their router weights **in ascending expert index**, then the shared experts, in F32, whichever site produced each output (RFC-0023 §6). The sum therefore does not depend on how the split fell. The sites' own kernels still round differently, which ADR-0019 accepts and the parity harness measures. When the split has no `COPY` experts, the dispatcher stores `F[l] = 1` from the host directly; a domain with no `CPU` experts still raises its `Q[l][d]`. Every flag is raised every step, so the handshake of §4.2 never stalls.

**Table updates are deltas applied by the graph's first node.** The dispatcher posts `{ table, index, value }` entries to a host-mapped table-delta ring and, when it writes step s's descriptor, records the ring position it wants applied in `StepDesc.delta_head`. The first node of every captured step graph after the descriptor copy is `ApplyTableDeltas`: it applies every entry up to `delta_head`, in order, never reading a live ring head, and echoes the applied position to a host-mapped word. Every site therefore evaluates step s at one table version, which the dispatcher chose before launch (RFC-0023 §5.4). `copy_eligible` is maintained only by these deltas. RFC-0022 uses the same op for KV block tables. Providers without capture launch it as the step's first op. The ggml provider has no such op, so the dispatcher applies the deltas with a host-side `copy` before the step. An admitted expert becomes visible only through a delta posted after its own copy into its slot has completed (RFC-0023 §5.5).

On a `Unified` device there is no `COPY` class (`B_copy = 0`): every expert in the unified arena is computed in place by the GPU through its imported buffer, and only `Nvme`-homed misses wait for a demand read (RFC-0023 §3). On CPU-only stages every expert is `CPU`.

#### 3.5 Arguments and the step descriptor

```rust
pub struct OpArgs<'a> {
    pub inputs: &'a [TensorArg],    // activations, KV page pools, states
    pub outputs: &'a [TensorArg],
    pub weights: &'a [BufferView],  // from TierManager::view(lease); layout must equal OpKey.layout
    pub step: BufferSlice,          // the step descriptor (device memory)
    pub workspace: Option<BufferSlice>,
    pub statics: &'a [Scalar],      // per-op constants fixed at prepare: eps, rope theta, scales
}
pub struct TensorArg { pub buf: BufferSlice, pub dtype: DType, pub dims: [u32; 4], pub strides: [u32; 4] }
```

Every value that changes from step to step lives in the **step descriptor**, never in `OpArgs`. That is what lets one captured graph serve every step:

```c
struct StepDesc {                 // 64-byte aligned; sized at Warming from BatchingLimits
    u64 step;                     // ActivationFrame.step
    u32 rows_active, seqs_active;
    u32 moe_epoch, flags;         // flags: PREFILL_LAST, TREE, …
    u32 off_row_seq, off_row_pos, off_row_token, off_tree_parent;   // byte offsets of the arrays below
    u32 off_seq_kv_len, off_seq_page_table, off_seq_state_slot, off_seq_sampler;
    u32 off_moe_balance;          // per MoE layer of the stage: B_copy(l) in bytes (§3.4, RFC-0023 §8)
    u32 off_row_kv_write_from;    // per row: first position whose KV this step writes (RFC-0033 replay); row_pos[r] in normal steps
    u64 delta_head;               // table-delta ring position ApplyTableDeltas applies up to (§3.4)
    u64 in_payload;               // device-visible address of this step's inbound ActivationFrame payload
    u64 out_payload;              // device-visible address of the registered send buffer for the outbound frame
    u32 rows_tap, tap_k;          // parity taps (§3.8): tapped rows this step (0 in service steps); K reference ids per row
    u32 off_tap_rows, tap_flag;   // offset of the tapped row indices; doorbell slot the ids' arrival raises
    u64 tap_ids;                  // pinned buffer of reference token ids, rows_tap × tap_k (TapMode::Gather)
    u64 tap_out;                  // result block: per tapped row, K log-probs, logsumexp, argmax, top-2 margin
    u64 tap_full;                 // shared-memory region for F32 logits of tapped rows (TapMode::FullLocal), or 0
    // u32 row_seq[R]; u32 row_pos[R]; u32 row_token[R]; i32 tree_parent[R]; u32 row_kv_write_from[R];
    // u32 seq_kv_len[S]; u32 seq_page_table[S]; u32 seq_state_slot[S]; SamplerState seq_sampler[S];
    // u32 moe_balance[L_moe];
};
struct SamplerState {             // per sequence, 8-byte aligned; sampling parameters are RFC-0012 §8's
    u64 seed;                     // the client's seed, or the seed ingress generated (ADR-0107)
    u64 stream_hi, stream_lo;     // the key's sequence component in ADR-0107's versioned encoding: the
                                  // choice index when the client supplied the seed, else the SequenceId
    u64 rng_counter;              // draws taken so far; SampleSeeded/VerifyAccept report the value after
                                  // their last draw in the step's result block (u64 everywhere)
    /* temperature, top-k, top-p, penalties, grammar state slot: RFC-0012 §8 */
};
```

The host writes the descriptor into a pinned mirror. The first node of every captured step graph copies the mirror to its device-resident copy (a few KiB), and the second is `ApplyTableDeltas` (§3.4), so the host makes one compute-stream submission per step: the replay. Copy-stream submissions are separate: the dispatcher issues one `copy_batch` per MoE layer with copied misses, and one `CopyIn` before a prefill replay. They are budgeted under `t_switch` and the prefill term, not under `t_submit` (§Performance). KV writes go through `row_kv_write_from`: attention and cache-append kernels write KV only for positions at or above it, so a replay step (RFC-0033 §4.2) recomputes positions without overwriting KV that survived; in normal steps it equals `row_pos`. Receive and send buffers come from pools, so their addresses travel in the descriptor too. `in_payload` is the frame-in address RFC-0012 §3.1 requires: for decode and verify frames (16–224 KiB), the graph's first node after the descriptor copy reads the registered receive buffer through it and copies the frame into the stage's fixed input buffer. Prefill frames, which are MiB-scale, are copied instead by a `CopyIn` submission the dispatcher issues before the replay, behind a device-side event wait. On `Unified` devices the receive buffer is imported and read in place. `out_payload` is the registered send buffer the graph's last node writes. Mirrors come in pairs, and each bucket has one graph instance per mirror parity. Right after its descriptor copy, a graph raises its parity's "descriptor consumed" doorbell `DC[parity]` (constant 1, §4.2), and the host waits for it and resets it before rewriting that parity's mirror. With one step in flight per sequence the wait is always already satisfied; it matters only when throughput mode keeps two steps queued on one device. Paged-KV page tables live in device memory owned by the KV store (RFC-0022); the descriptor holds only offsets into them.

#### 3.6 Split attention over spilled KV pages

RFC-0022 §5.3 (`CpuSplit`) and ADR-0207 split one layer's decode attention between the GPU, over the sequence's VRAM pages, and the CPU pool of each NUMA domain holding host pages. The results are merged exactly. The provider contract:

```text
AttnPartial(kind): inputs  q (rows × heads × d, F32, from AttnHandoff on the CPU site);
                           the positions this site attends: on the GPU, its pages from the step descriptor's
                           page tables; on a CPU site, the host pages listed in the layer's AttnDoorbell
                           block (dense kinds: from the page tables; sparse kinds: the selection, below)
                   outputs o (rows × heads × d_v, F32, NOT divided by l),
                           m (rows × heads, F32: the running maximum score),
                           l (rows × heads, F32: Σ exp(score − m))
AttnMerge:         inputs  the partials (o_i, m_i, l_i), i = 0 … p−1, in a fixed order: the GPU's partial first,
                           then the CPU partials in ascending NUMA-domain index; optional per-head sink logit s
                   output  m = max(max_i m_i, s);
                           o = Σ_i e^(m_i − m) · o_i / (Σ_i e^(m_i − m) · l_i + e^(s − m))     (all F32;
                           s = −∞ when the family has no sinks, so its term is 0)
```

- `AttnPartial` exists for every `AttnKind` on the CPU provider, and on each GPU provider whose devices can spill KV to host memory (CUDA from M1; HIP and Vulkan from M3); `AttnMerge` on those GPU providers. Metal's unified devices have no spill path (RFC-0022), so Metal gets them only if that changes. Both are distinct from `PartialReduce`, which is a plain sum. Both have `kernels::reference` functions and parity tests on CUDA and CPU (Testing). For `Mla` the CPU works in the latent space with the absorbed query, and the value up-projection runs on the GPU after `AttnMerge`.
- **Sinks** (gpt-oss-class learned sinks) enter the denominator exactly once and contribute no value. They are applied in `AttnMerge` only, never inside a partial.
- **Empty side.** A site with no positions for a row returns `m = −∞`, `l = 0` and `o = 0`, and still raises its flag every step, so no wait depends on page counts (§4.2 rule 3). A row whose merged denominator is 0 (no positions anywhere and no sink) is a defect, not a NaN: it fails the sequence with `Correctness` (RFC-0042 §4.1 rule 6).
- **Hand-off buffers.** RFC-0022's `AttnHandoff`: a host `Scratch` sub-arena registered `GPU_MAPPED`, double-buffered across layers, holding per layer the F32 query (`rows × n_q_heads × d_q × 4` bytes) and one partials block per NUMA domain (`rows × n_q_heads × (d_v + 2) × 4` bytes). Its size is part of `plan.scratch`.
- **Doorbell.** RFC-0022's per-layer `AttnDoorbell` block, beside, not inside, the MoE `DoorbellLayer`. Its `step` field is a payload check the host asserts; the flags are separate and follow §4.2's handshake: `Ra[l][d]` (GPU → CPU pool of domain d, raised after the query, and for sparse kinds the selection, are written) and `Qa[l][d]` (CPU pool → GPU, raised after its partials are written). `AttnMerge` waits on every `Qa[l][d]` with `wait_host`.
- **Sparse kinds.** The indexer runs on the GPU inside the step, so the CPU site cannot read its positions from the page tables. The GPU's selection op writes, per row, the selected **positions** that live in host pages into the doorbell block's `selected` array, with their count, before raising `Ra[l][d]`. The bound is the layer's `selected` count per row (from its `KvFormula`), by construction of the indexer's top-k; a page count does not bound it for token-level selection, whose positions can fall in up to `selected` different pages. The CPU's `AttnPartial` attends exactly that list.
- The merge is the same rescaling a blocked attention kernel does between blocks, so the result equals one-site attention up to rounding. It is an exact-mode path, and RFC-0060 runs it at several split ratios (ADR-0207 decision 5).

#### 3.7 Timestamps

`Timestamp { slot }` writes the device's clock, in nanoseconds since an arbitrary device epoch, to slot `slot` of a host-mapped timing buffer owned by the step program. Profiled steps (RFC-0010 §6.1, RFC-0012 §11; one in `probe.online.profile_every`) are a second captured variant of the step program with `Timestamp` ops at segment boundaries. They compute the same values in the same order, so profiling never changes output.

| Backend | Mechanism | Host correlation |
|---|---|---|
| CUDA | a one-thread kernel reading the global nanosecond timer register and storing it to mapped memory; capturable | a device–host round trip at `Warming` fits the offset |
| HIP | a one-thread kernel reading the constant-rate wall-clock counter, scaled by its reported frequency | as CUDA |
| Metal | counter sample buffers sampled at encoder (stage) boundaries, which is where Apple GPUs support sampling | `sampleTimestamps` CPU/GPU pairs |
| Vulkan | `vkCmdWriteTimestamp2` into a query pool, copied to the timing buffer at step end; ticks scaled by `timestampPeriod` | calibrated timestamps where the driver exposes them, else a round-trip fit |
| CPU | the monotonic clock | identity |

Each `Timestamp` costs a launch or an encoder boundary. RFC-0010's overhead budget allows at most 2 per segment on profiled steps. Metal may have to split encoders to sample, which perturbs timing. RFC-0010 open question 8 asks this RFC to measure the perturbation; if it exceeds 2 %, Metal profiled steps run at a lower rate (Open questions).

#### 3.8 Parity taps

RFC-0012 §8.5 and RFC-0060 §5.5 measure a real instance by teacher forcing, and the warm-up canary uses the same mechanism on a short fixed sequence. The tail's step program reads the model's own distribution, the `Logits` output before `LogitsProcess`, so sampling settings never enter a measurement. The kernels involved:

```text
TapGather { k }: inputs  logits (rows × vocab, F32); tapped row indices (StepDesc::off_tap_rows, rows_tap of them);
                         reference ids (StepDesc::tap_ids, rows_tap × k)
                 output  per tapped row r, into StepDesc::tap_out:
                           lse_r    = logsumexp over the full vocabulary (max-subtracted, F32)
                           lp_r[i]  = logits[r][ids[r][i]] − lse_r          for i < k
                           argmax_r = index of the largest logit (ties to the lower id)
                           margin_r = top-1 log-prob − top-2 log-prob
FullLocal mode:  GatherRows copies the tapped rows' F32 logits to StepDesc::tap_full (a shared-memory region the
                 driver maps on the tail's host); no TapGather
```

- **One captured program for service and parity.** Tap ops sit after `Logits` in every tail step program and read `rows_tap` from the descriptor. In service steps `rows_tap = 0`, so they do no work, and parity never needs another graph (RFC-0012 §8.5).
- **Ids that have not arrived.** Reference ids stream to the tail over a `Tap` channel (RFC-0031). If a step's ids are not yet in `tap_ids`, the step waits device-side with `wait_host` on doorbell slot `tap_flag` (§4.2), the same mechanism grammar masks use. Parity runs are not latency-sensitive.
- **Numerics.** `TapGather` follows RFC-0042 §4.2's softmax rule (F32, max-subtracted). Its outputs are compared by the harness against the reference's stored blocks, so it is itself an op under conformance (RFC-0060 §4.2). Its argmax and index outputs are in the bit-exact class.
- **Memory.** Tap scratch (one prefill chunk of logits at vocabulary width, for `FullLocal`) is reserved at `Warming` only when `debug.parity_api` is set, plus a small fixed region for the canary.
- **Hidden-state taps (tap variant only).** `HiddenTap { boundary }` copies the residual stream of the tapped rows at a layer boundary into the host-mapped tap buffer. It never appears in a service program. When a parity sequence asks for layer taps and `debug.parity_api` is set, each affected stage builds a **tap variant** of its step programs in an idle window (RFC-0012 §8.5): the same ops, plus `HiddenTap` at the stage's input and output boundaries and at the layers RFC-0060 §5.7 samples, plus a `KvPack` of those layers' KV pages into the tap buffer when the request asks for the KV prefix. Tapped steps replay the variant; every other step replays the normal program, whose captured graph is unchanged (RFC-0060 checks its hash with the gate off). Release builds therefore carry the op, and layer-local parity gates the release artefacts themselves (RFC-0060 §5.7, RFC-0042 §7).

### 4. Streams, events, doorbells and graph capture

#### 4.1 Streams per device

`open` creates the streams named in `OpenOptions::streams`. The default plan is one `Compute` stream, one `CopyIn` (host → device: copied misses, prefetch, KV restore), one `CopyOut` (device → host: KV spill, activation hand-off to the wire) and, for `IntraNodeTp`, one `Collective` stream. CUDA and HIP map priorities to stream priorities (compute highest). Metal maps each stream to an `MTL4CommandQueue` (an `MTLCommandQueue` before macOS 26). Vulkan maps `Compute` to a compute queue and both copy streams to a transfer-only queue family where one exists, falling back to a second compute queue.

#### 4.2 Doorbells: host ↔ device without synchronising calls

At `open`, the provider allocates `doorbell_slots` 4-byte flags in one or more 4 KiB pinned pages, registers them with the device and keeps both addresses. `signal_host` and `wait_host` compile to the mechanism below. Each provider reports which one it uses through `DeviceFeatures::STREAM_MEM_OPS` and the `OpCaps` of a pseudo-op `Doorbell`.

| Backend | `signal_host` (device → host) | `wait_host` (host → device) | Fallback when the primary path is missing |
|---|---|---|---|
| CUDA | `cuStreamWriteValue32` writing the constant 1 to a host-registered flag | `cuStreamWaitValue32(EQ, 1)` followed by `cuStreamWriteValue32(flag, 0)`, the consumer's reset; both captured as memory-operation nodes whose values are constants | A **functional probe** at `open` decides (below). If it fails: a one-thread "store" kernel and a one-warp "spin" kernel that polls the mapped flag with `ld.volatile` and back-off, then resets it, with the same constants. Both are graph-capturable |
| HIP | `hipStreamWriteValue32`, constant 1 | `hipStreamWaitValue32(EQ, 1)`, then a constant-0 write | the same store and spin kernels |
| Metal | `MTLSharedEvent` signal, encoded after the producing encoder with the current step number; the host reads `signaledValue`. Event commands are encoded per step into a fresh command buffer (Metal 3) or attached at commit (Metal 4), so their values are per-step, not captured | the host sets `signaledValue` to the step number. The GPU waits for that value between compute encoders of one command buffer (Metal 3 path), or as a queue-level wait between the step's command buffers, which the host commits together up front (Metal 4 path) | none needed on macOS 15+; `SHARED_EVENTS` is required for the native Metal provider. A spike in M1 measures the per-wait cost on both paths (Open questions) |
| Vulkan | a 4-byte `vkCmdFillBuffer` of the constant 1 into a `HOST_VISIBLE \| HOST_COHERENT` flag buffer, pre-recorded; the host consumer resets it | **no in-command-buffer wait on a host value is portable.** The step is split at each `wait_host` into segments. All of a step's segments are submitted up front in one `vkQueueSubmit2`, each waiting on a timeline-semaphore value that the host signals later with `vkSignalSemaphore` (wait-before-signal). Wait values are supplied at submission, so they carry the step number | always the split; the host cost is one submission per step plus one `vkSignalSemaphore` per wait, measured by the probe (§6.6) |
| CPU | release store of 1 | acquire spin with `pause` back-off, then the reset. While a step is in flight the waiting thread parks only on a device-triggered wake armed for its next doorbell (`arm_wake`, below; RFC-0012 §3.3), never on a timer or a ring event; between steps it parks normally | — |

A spin kernel occupies one warp on one SM for at most the CPU's in-place time, which is the step's critical path anyway.

**The CUDA probe is functional, not an attribute query** (corrected 2026-10-20; OQ-KB-13). No device attribute for 32-bit stream wait and write operations exists in current CUDA. The only one left, the deprecated `CAN_USE_STREAM_MEM_OPS_V1`, reads 0 on WDDM devices where the operations work and capture, so an attribute test would send every Windows GeForce device to the spin path. At `open` the provider instead: registers a host flag; captures `cuStreamWaitValue32(EQ, 1)` and a constant-0 `cuStreamWriteValue32` on it into a graph in thread-local mode; instantiates the graph; replays it while a host thread writes 1 to the flag; and requires the replay to complete and leave the flag at 0 within a timeout. Any failure, or the timeout, selects the spin and store kernels. The result is recorded in `DeviceFeatures::STREAM_MEM_OPS` and in the capability profile. COMP-E4-T12 runs the same probe as a standalone check on a Windows GeForce machine in M0 week 1 and on R-WIN in M1.

**The handshake (ADR-0404 decision 6).** Values in captured CUDA and HIP memory-operation nodes, and in pre-recorded Vulkan fills, are fixed when the graph is captured or the command buffer recorded, so a doorbell cannot carry the step number on those backends. Every doorbell therefore uses a constant-valued, consumer-reset handshake:

1. Each flag has exactly one producer and one consumer. A doorbell with several consumers is several flags: the route doorbell is `Rd[l]` for the dispatcher plus one `Rc[l][d]` per NUMA domain's CPU pool.
2. The producer only ever writes 1. The consumer waits for 1, then writes 0, and only after that produces anything that can lead to the flag's next set.
3. Every flag is raised exactly once per pass of the program part that owns it, even when its work is empty (no copies, no CPU experts, no host pages). That is once per step for a step's layers, and once per drafted token for a chained MTP module, whose MoE layer repeats its doorbells in stream order.
4. All of a device's steps run through one compute stream in order (§4.1).
5. Step-level flags exist once per mirror parity: the descriptor-consumed flag `DC[p]`, the step-done flag `O[p]` (RFC-0012), and host-set step flags such as `tap_flag`. Each parity's graph instance captures its own slots. The host's next submission of parity p is what leads to their next set, so the host consumes `DC[p]` and `O[p]` from p's previous use before it resubmits p. Both start at 1.

*Why it is correct.* By rules 2 and 3, a set at step s+1 is causally after the consumer's reset at step s. The same argument holds per pass when a program runs a layer several times in one step (chained MTP), because those passes are also in stream order. Device → host flags (`Rd`, `Rc`, `DC`, the step-done flag): the GPU reaches layer l of step s+1 only after passing layer l's waits at step s, and those waits are satisfied only by replies the host sends after resetting the step-s flag. Host → device flags (`F`, `Q`, `tap_flag`, attention partials): the host sets them for step s+1 only after seeing that step's route doorbell, which the GPU raises after its step-s reset in stream order. A stale value therefore never satisfies a wait, and no wait depends on a value captured at another step. The release store and stream memory operations (which order prior work by default) make the payload visible before the flag. A producer that skips a set leaves the flag at 0, so the step stalls and RFC-0012's stall watchdog reports it; it never passes on old data. All flags are re-armed (0, with `DC` and `O` at 1) after each capture and after any abandoned step.

**The fallback, stated once.** The handshake governs every host-mapped flag: on CUDA and HIP, Vulkan's device-to-host fills, and the CPU. Metal shared events and Vulkan timeline-semaphore waits carry per-step values instead, set when the step's command buffer is encoded or its batch submitted. They are never captured and never reset (Metal and Vulkan rows above). For the host-mapped flags, only the device-side mechanism changes. Per device, in order of preference:

1. **Native operations.** CUDA and HIP stream wait-value and write-value operations, captured as graph nodes and running the handshake. On Metal and Vulkan, shared events and timeline values, set per step. None of them occupies an SM.
2. **Spin and store kernels**, only where the native operations are absent: the functional probe at `open` fails (CUDA, above), or memory-operation capture fails on the pinned ROCm release (HIP), or `features.disable` holds `stream_mem_ops`. A one-thread store kernel writes 1. A one-warp spin kernel polls the mapped flag for 1 with `ld.volatile` and back-off, then writes 0. They run the same constant-valued handshake. They never compare a flag with a step number, so the host side cannot tell which path a device uses. Step numbers appear only in payloads that the producing op writes from the descriptor (the router's `seq`, the attention doorbell's `step`), as checks the host asserts. A spin kernel holds one warp for the length of its wait (open question 3); `kernels.spin_kernel_max_us` bounds it.
3. **No device-side wait at all.** `wait_host` returns `Unsupported`, and the executor splits the step at each wait, launching the remainder after the host sees the flag. This costs launches but not correctness (ADR-0014 rule 2).

No host-mapped flag carries a monotonic step number on any path. Step numbers appear only in payload checks, and as the per-step values of Metal events and Vulkan timeline waits.

Other host-side rules, owned by RFC-0012 and restated because kernels depend on them: every flag lives on its own 64-byte line; a host store to a flag the device waits on is a release store followed by nothing else on that line; and a host thread waiting inside a step spins with `pause` back-off, or parks only on a wake armed for its next doorbell, until the step-done flag fires (CPU row above).

**Device-triggered wakes (`arm_wake`).** RFC-0012 §3.3 lets a waiting thread park when the predicted gap to its next doorbell exceeds `exec.inflight_park_threshold`, provided the device will wake it. Each provider implements the wake right after the doorbell write:

| Backend | `arm_wake` |
|---|---|
| CUDA, HIP | a host-function node (`cuLaunchHostFunc`/`hipLaunchHostFunc`, captured as a host node) on a forked side branch after the doorbell write, joined back at the step's end so the compute stream never waits for the callback; the callback signals the waiter's futex, eventfd or Windows event |
| Metal | `encodeSignalEvent` of a shared event after the doorbell's encoder, with an `MTLSharedEventListener` block that signals the waiter |
| Vulkan | the doorbell write ends its segment, whose timeline-semaphore signal the parked thread waits on with `vkWaitSemaphores` and a timeout at the predicted doorbell time |
| CPU | the writer's futex or event wake after its release store, when the waiter announced it is parking |

The plan arms wakes only for doorbells whose predicted gap exceeds the threshold, so programs for fast stages contain no host nodes at all. The probe measures each path's wake latency (RFC-0010's H2 wake variant). Metal, whose event values are set per step, keeps step numbers. On Vulkan, a pre-recorded fill is followed by a buffer barrier to the host domain; the probe checks once per driver that the host sees a fill while the submission is still running, and where it does not, `signal_host` ends the segment and signals the timeline semaphore instead, which the host reads with `vkGetSemaphoreCounterValue`.

**Programs without per-layer doorbells.** A stage whose MoE layers cannot miss (every expert of the stage resident in VRAM, or in a `Unified` device's arena) is captured without route flags or waits, and its routing statistics are read after the step from the doorbell's route block, which the router writes into host-mapped memory (§3.4), with plain loads and no copy (revised 2026-10-21). This removes per-layer host interaction where nothing needs it, which matters most on Metal and Vulkan, where every wait is a host-side command (RFC-0012 §7.3).

#### 4.3 Events

`event`, `record`, `wait` and `poll` follow the backend's native semantics (CUDA/HIP events, Metal shared events with monotonically increasing values, Vulkan timeline semaphores). `elapsed` uses timing-enabled events, which are created only for microbenchmarks and sampled telemetry, never for the steady-state graph. Events can cross streams on one device. Across devices of one domain in one runner (intra-node TP), CUDA and HIP events can be waited on from any device's stream. Metal and Vulkan do not support intra-node TP in v1.

#### 4.4 Graph capture

**What is captured.** For each (stage step program, bucket, mirror parity), the executor (RFC-0012) captures the full step: the descriptor copy, every layer's ops, the doorbell signals and waits, and the hand-off of the output activation into its registered send buffer. The capture is made once in `Warming` (`WarmStep::Capture`) and replayed every step.

**Capture rules every provider must honour:**

1. A captured graph references only buffers whose addresses are fixed for the plan epoch: weight homes, expert-cache slot ranges, staging rings, KV page pools, scratch, the step descriptor and the doorbell pages.
2. No host callbacks, no allocation, no host-visible synchronisation inside a captured region. The one exception is an `arm_wake` host node (§4.2), which sits on a forked side branch that the compute stream never waits on.
3. Every dynamic value is read from the step descriptor or from tables addressed through it (§3.5).
4. `prepare` must happen before `begin_capture`; providers may not lazily compile, allocate workspaces or load modules during capture.
5. A provider whose op cannot be captured reports `OpCaps::capture_safe = false`. The executor then ends the capture before that op, launches it directly and starts a new captured segment after it. A step therefore becomes a short list of graph segments and direct launches. Each break costs one more submission.

**Per backend:**

| Backend | Mechanism | Notes |
|---|---|---|
| CUDA | stream capture (`cuStreamBeginCapture` in thread-local mode) into `cuGraphInstantiate`; replay with `cuGraphLaunch` | thread-local mode lets the housekeeping thread register host memory (RFC-0020 §6) while the dispatcher captures. Stream memory operations are captured as memory-op nodes with constant values (§4.2). Instantiation may run on a helper thread with the device's primary context current; the executor swaps the new executable in between steps. Each instantiated graph is re-instantiated only when a plan delta changes an address, which by rule 1 means a new epoch |
| HIP | `hipStreamBeginCapture` / `hipGraphLaunch` | memory-op capture is verified per ROCm release in CI; when missing, the spin kernels stand in |
| Metal | Metal 4: command buffers built from an `MTL4CommandAllocator`, with compute dispatches encoded into an `MTLIndirectCommandBuffer` once at `Warming` and executed each step; macOS 15: an indirect command buffer executed from a fresh `MTLCommandBuffer` per step | argument buffers hold every resource address. Every buffer the ICB touches is in the device's `MTLResidencySet`, so the per-step command buffer needs no `useResource` calls. The per-step command buffer (one `executeCommandsInBuffer` per segment, with event waits and signals between segments carrying that step's values) is fixed in structure for the epoch, so the dispatcher encodes step s+1's buffer while step s runs or while it idles, off the critical path (RFC-0012 §7.3) |
| Vulkan | command buffers recorded once per (segment, bucket, parity) without `ONE_TIME_SUBMIT`, resubmitted each step | the split at each `wait_host` gives several command buffers per step; push constants are never used for per-step values |
| CPU | a pre-built op list replayed by the dispatcher | "capture" is a no-op by design |

**Invalidation.** A captured graph is dropped on a new plan epoch, on a provider health change, or when a `PlanDelta` changes a bucket set or a scratch size. Expert-cache contents, slot-table words, KV page contents and the step descriptor change freely without invalidation. Expert-cache shrinking unmaps tail segments only, so the remaining slot addresses stay valid (RFC-0001 §9.3).

**Graph count and memory.** Graph instances per stage = buckets × 2 parities, plus one profiled variant (§3.7) per profiled bucket, plus the parity variant (§3.8) when `debug.parity_api` is set. RFC-0012 §2 owns the count: with defaults, 14 instances in latency mode and up to 12 more in throughput mode, plus 2–8 profiled variants. Their device memory is part of `plan.scratch` and is measured at `Warming`, not predicted.

#### 4.5 OS GPU watchdogs

ADR-0014 rule 6: **kernels are always designed within the operating systems' default GPU watchdogs, and Syntropy never changes `TdrDelay`, hardware-accelerated GPU scheduling (HAGS) or any other machine-wide GPU setting, nor asks the user to.** Both settings are machine-wide, need a reboot and affect every application on the desktop. `syntropy doctor` reports them (RFC-0051), and the probe records each device's limit (RFC-0010).

The limits: on a WDDM display GPU, Windows resets the display driver when the GPU does not yield within `TdrDelay`, 2 s by default, which takes down every GPU application on the desktop; CUDA reports a kernel run-time limit on such devices (`CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT` = 1). macOS terminates a process whose command buffer runs past its GPU watchdog, reported at about 5 s per command buffer rather than per dispatch. A 4096-token prefill chunk is far above either: ≈ 22 s for a 70B dense stage at an assumed 26 TFLOPS, and ≈ 25 s for an 8B model on the M1-class floor Mac [estimate: 2 × 70e9 × 4096 ≈ 5.7e14 FLOP at 26 TFLOPS; 6.6e13 FLOP for the 8B chunk at the M1-class GPU's 2.6 TFLOPS FP32 peak].

1. **A segment budget per device.** The step-program builder (RFC-0012) keeps every unit of GPU work the OS times under the device's segment budget, RFC-0012's `exec.max_gpu_segment_ms` (suggested values ≤ 1 s on Metal and ≤ 0.5 s on WDDM display GPUs [target]; RFC-0012 sets the defaults). It plans with each op's `OpCaps::est` at the bucket shape, and providers make the split possible:
   - **CUDA and HIP on a device that reports a run-time limit:** prefill `Gemm`, `GateUpAct` and `AttnPrefill` accept a row range, so one prefill chunk becomes several launches inside one captured segment, each estimated within the budget.
   - **Metal:** a prefill step whose estimate exceeds the budget is encoded as several command buffers, one per layer group, each within the budget and committed in order with the step's event values; decode steps stay one command buffer per segment (§4.4).
   - **Vulkan:** the same rule per submission, since the Windows watchdog applies whatever the API.
   - **Compute-only GPUs** (no display attached, `KERNEL_EXEC_TIMEOUT` = 0) and the CPU have no OS budget.
2. **In-step host waits end before the watchdog.** A spin kernel or a device-side wait on a host flag (`cuStreamWaitValue32`, a Metal shared-event wait) is bounded by the device's segment budget. Past it the step fails with `DoorbellTimeout` (11), and the runner abandons the step before the OS acts; `kernels.spin_kernel_max_us` is capped at the budget (§12). Whether a blocked native wait counts toward the OS watchdog at all is measured in the lab on R-WIN and R-MACF (COMP-E4-T19, COMP-E6-T13), never on a user's machine.
3. **HAGS is reported, never requested.** No API lets an application request it; the provider reads its state into the capability profile, and `t_fixed` is measured both ways on R-WIN (§7.1 platform notes).

### 5. Memory

#### 5.1 Allocation kinds

`AllocRequest { bytes, kind, align, reserve_only }` from RFC-0001 §5.7. Provider semantics:

| `MemKind` | CUDA / HIP | Metal | Vulkan | CPU |
|---|---|---|---|---|
| `Device` | `cuMemCreate` + `cuMemMap` into a reserved range when `VMM` is present, else `cuMemAlloc` | `MTLHeap` (private storage) sub-allocated by the tier manager | `vkAllocateMemory` from a `DEVICE_LOCAL` type, bound to one `VkBuffer` with device address | — |
| `HostMapped` | `cuMemHostRegister` over tier-manager memory (via `register_host`), or `cuMemHostAlloc` for doorbell pages | `newBufferWithBytesNoCopy` over page-aligned tier-manager memory | `VK_EXT_external_memory_host` import over tier-manager memory where supported, else a `HOST_VISIBLE` allocation the tier manager adopts | the region itself |
| `Unified` | GB10-class: `cuMemHostRegister` over tier-manager memory with device-mapped pointer; Strix Halo under HIP: `hipHostRegister` | shared-storage buffer over tier-manager memory (`import`) | `HOST_VISIBLE \| DEVICE_LOCAL` type on integrated GPUs | — |

`reserve_only` reserves address space (`cuMemAddressReserve`, `hipMemAddressReserve`) with no backing; `map_segment`/`unmap_segment` add or remove physical segments whose size the tier manager rounds up to the granularity the driver reports (`cuMemGetAllocationGranularity`; queried, never assumed). Metal and Vulkan have no address reservation. There, `map_segment` with an empty reservation allocates a standalone segment (an `MTLHeap`, or a `VkDeviceMemory` with one bound buffer and its device address), as RFC-0020 §4.1 describes. Kernels reach expert slots through `slot_addr` (§3.4), so adding or freeing a segment never moves another segment's slots.

**`Unified` on CUDA integrated devices (GB10-class).** RFC-0020 §9 leaves to this RFC which CUDA allocation gives full-rate access from both CPU and GPU there. No vendor document found for this RFC settles it, so the provider measures. Three mechanisms are candidates:

1. `cuMemHostRegister` (with the device-map flag) over tier-manager memory: the default, because the tier manager keeps ownership and the memory stays ordinary pageable-then-locked RAM;
2. `cuMemHostAlloc` with the portable and device-map flags, adopted by the tier manager;
3. `cuMemAllocManaged`, with preferred-location and accessed-by advice set for the GPU and the CPU.

The probe (RFC-0010) runs the selected dense GEMV kernel and the CPU provider's GEMV over a 1–2 GiB buffer of each kind. It keeps the mechanism with the highest GPU W_eff among those whose CPU read rate is at least 0.9 × the best CPU rate measured [target threshold: on unified devices the GPU computes almost everything, but in-place CPU work must not collapse]. The choice is recorded in the capability profile, so the solver plans with the rates of the mechanism in use (Open questions).

#### 5.2 Registration and import

- `register_host` makes a range of the tier manager's pinned arena DMA-visible. One registration per RFC-0020 §6.1 unit, never per tensor: units are aligned to extent, slot and elastic-segment boundaries on every OS, so no DMA source crosses two registrations and a shrinking arena unregisters whole units. CUDA's registration cost grows with range count and WDDM limits registrations for some GPUs (RFC-0001 §9.2), which is why RFC-0020 bounds the unit count. Registrations are issued from `Fetching` onwards through `HostRegistrar`, under RFC-0020 §6's rule on when they may run (before `Ready` until the M1 spike shows they do not lengthen steps).
- `import` wraps memory the tier manager owns as a device buffer without copying. It is the only way bytes enter a `Unified` tier, and the way the ggml provider receives Syntropy memory on CPU and Metal (§8.3).
- On Metal, every allocation and import is added to the device's `MTLResidencySet` at creation. `set_resident(false)` removes it, which is how the tier manager tells Metal that a shrunk cache heap or an evicted KV range is no longer needed.

#### 5.3 Workspaces

`query` returns `OpCaps::workspace` per (op, shape). The executor sums the maximum over the ops in each concurrently live region of the step program and asks the tier manager for that much scratch. Providers never allocate workspaces themselves (RFC-0001 §9.1).

#### 5.4 Repacked layouts

`repack(op, src, dst)` converts one unit of source bytes (one expert's matrices, or one dense tensor) into the provider's preferred layout for `op`. It runs during `Packing` (RFC-0041 §5.3), is a pure function of its input, and must be deterministic so that pack extents can be content-addressed. Each provider publishes its layouts:

| Layout ID | Provider | Applies to | Purpose |
|---|---|---|---|
| `cpu.rows4`, `cpu.rows8` | CPU | k-quants, Q8_0, Q4_0, IQ4_NL/XS, MXFP4 | interleave 4 or 8 output rows block by block so one SIMD load feeds 4/8 dot products |
| `cpu.amx16` | CPU (AMX) | Q8_0, Q4_K, BF16 | 16-row × 64-byte tiles in AMX `tileloadd` order, with scales separated |
| `cuda.fp4sw` | CUDA (sm_120a/121a) | MXFP4, NVFP4 | element and scale-factor swizzle expected by the block-scaled MMA instruction |
| `cuda.fp8blk` | CUDA (sm_89+) | FP8 128×128 block-scaled | k-major tiles with per-block scales adjacent |
| `metal.t8` | Metal | k-quants, MXFP4 | layout matched to `simdgroup_matrix` 8×8 or tensor-op tiles |
| `vk.coop16` | Vulkan | Q8_0, k-quants | 16×16 cooperative-matrix tiles |
| `SOURCE` (0) | all | all | the source format's layout, unchanged |

Every repack is lossless: it reorders bytes and may widen scales, never requantise. `layout` is part of `OpKey`, and the pack sidecar records the layout of each extent (RFC-0041 §5.2), so a repacked weight can only be bound to an op prepared for that layout. When the pack's disk budget is exceeded, packing falls back to `SOURCE` and selection considers only kernels for `SOURCE` (RFC-0001 §3.3). **Coverage rule:** a provider that offers a weight-bound op for a scheme also offers a `SOURCE`-layout variant of it, so `SOURCE` is always a valid choice for every reader.

**One layout decision per tensor group, across all its readers.** A group's **readers** are every (device, provider) the plan may run an op on it with, reading stored bytes:

- for routed experts: the GPU (hits from VRAM slots and copied misses, both filled by verbatim copies of a host or `Nvme` copy) and the CPU pool of each NUMA domain that holds a home or RAM-cache copy (in-place misses). On a `Unified` device the GPU and the CPU miss site read the same bytes in place;
- for dense groups: the device of the group's stage, plus the GPU of any `StreamSite` that streams the group in `PrefillStream` (RFC-0012 §2; RFC-0020 §7.3), plus the fallback device of any level-F op on it (§11.2).

A reader is recorded as `ReaderId { device: DeviceId, provider: ProviderName }`, a `core` type (RFC-0041 §5.2). `ProviderName` is the owned text of the `ProviderId`, so the record postcard-encodes.

Because copies between sites are byte-for-byte, a slot always has the layout of the copy it was filled from. The layout pass (§6.1) therefore chooses per group, never per device, under one of two policies:

| Policy | Rule | When |
|---|---|---|
| **Shared** (default) | one stored layout that every reader's kernels accept: the candidate set is the intersection of the readers' layout sets, which always contains `SOURCE`. The pass picks the layout minimising `Σ_r f_r · t_r(layout)`, where `t_r` is reader r's benchmarked time per byte and `f_r` the share of the group's bytes per token the plan expects r to read (from its hit, copy and in-place split, RFC-0023 §8; equal shares when the plan gives none) | every group unless the plan says otherwise; always on `Unified` devices |
| **Per-site copies** | one stored copy per distinct reader layout; each reader reads only its own copy, and admissions and copied misses take the GPU-layout copy as source | only for groups the plan marks `per_site_layout`, because the extra copies' bytes must be charged to the tiers that hold them (RFC-0011) |

The layout pass reports both options to the solver per multi-reader group: the shared choice with its `Σ f_r · t_r`, and the per-site choices with their times and the extra bytes. The joint choice is recorded in `PackBuild` (RFC-0041 §5.3), one entry per stored layout with its readers. At `Warming`, each reader's selection considers only kernels for the layout its copy was stored in (§6.2 condition 3).

#### 5.5 KV page layouts (`KvLayoutId`)

RFC-0022 owns pages, page classes, pools and tiers; this RFC owns the order of elements inside one layer's page, because the attention kernels read it. A page holds `P` consecutive positions (`kv.page_tokens`, default 64, ADR-0205) of one layer in the layer's KV dtype (`KvFormula::dtype`, RFC-0042 §4.2).

```rust
pub struct KvLayoutId(pub u16);
```

| ID | Name | Used by | In-page order |
|---|---|---|---|
| 1 | `gqa.hpd` | `PerToken` GQA/MHA and sliding-window layers | a K plane then a V plane; each plane `[kv_head][position][d_head]`, so a decode kernel working on one KV head reads one contiguous `P × d_head` tile per plane |
| 2 | `mla.pt` | MLA layers | `[position][d_c + d_rope]`: the latent and the decoupled RoPE key together, shared by every head |
| 3 | `compressed.pe` | `Compressed` layers | `[entry][elems_per_entry]` |
| 4 | `state.f32` | `Fixed` (recurrent) layers | the family's state tensor in row-major order, then the short-convolution state; always F32 (RFC-0042 §4.2) |
| 5 | `indexer.pt` | `AuxKv::SparseIndexer` | `[position][elems_per_token]` |

Every provider reads and writes these canonical layouts in v1. Pages therefore move between tiers, between nodes (migration) and into saved sessions bit for bit, with no conversion (ADR-0019 decision 5). A provider that wants a swizzled in-page order must register a new ID, a `Convert` path both ways, and parity tests. Such IDs are reserved from 0x100 and are not used in v1. `KvPack` and `KvUnpack` gather a page index range from the per-layer device arrays (layer-first) into one page-first record per page, with each layer's slice padded to 4 KiB, and scatter it back (RFC-0022 §3). They run on the copy-in or copy-out stream. Where `copy_batch` maps to a batched copy in the driver, the provider may implement them as a batched copy instead of a kernel.

### 6. Per-op provider selection

Selection answers one question per (device, op occurrence in the step program): which provider, and which of its variants, runs it. ADR-0401 records the policy.

#### 6.1 When selection runs

Selection has two passes, because the weight layout must be known before packing but the final choice needs the device warmed:

1. **Layout pass, during `Fetching`.** For each weight-bound op (`Gemv`, `Gemm`, `GateUpAct`, `MoeGrouped`, `Logits`) on each reader of its tensor group (§5.4), the runner looks up the node's selection cache (§6.5). On a miss it benchmarks every candidate layout on **synthetic weights**: random but valid block encodings of the exact quant scheme and shape. Kernel speed depends on shapes and encoding, not on weight values, so this runs while the shard is still downloading. The group's layout is then chosen jointly across its readers by §5.4's rule, and the choice goes into the pack build request (RFC-0041 §5.3).
2. **Final pass, during `Warming` (`WarmStep::Prepare`).** Among candidates compatible with the packed layouts, the runner checks parity on each candidate (§6.2), benchmarks the ones the cache does not cover, and fixes the choice for the plan epoch. It writes the result to the pack's selection side record (RFC-0041 §5.2) and to the node cache (RFC-0001 §5.7).

#### 6.2 Candidates and certification

A candidate is a (provider, prepared variant) inside the device's memory domain (§6.4) for which:

1. `query(ctx, op, shape)` returns `Some(caps)`;
2. `caps.numerics` satisfies the RFC-0042 §4 accumulation contract for the op and dtype class;
3. `caps.layout` is the layout of the copy this reader reads (§5.4), or `SOURCE` when the pack keeps the source layout;
4. `caps.deterministic` is true, if the runner was opened deterministic;
5. it holds a **certificate** in the release's `ConformanceManifest` (ADR-0602; RFC-0060 §4.2, §6.4), at one of the levels below;
6. it passes the **runtime spot check**.

**Certificate levels:**

| Level | Condition | Use |
|---|---|---|
| **Certified** | an `OpCert` whose provider, device class, `OpKey` and shape class match, and whose `kernel` equals `caps.kernel_hash`. For a Vulkan-domain provider (the native Vulkan provider or ggml's Vulkan backend), the device and driver must also be the rig's that `OpCert::rig` records | selected normally |
| **Derived** | no `OpCert` for this device class, but one exists for the same provider, `OpKey`, shape class and `source_key` on another device class of the same vendor family | selected in exact mode only when (a) the `source_key` matches, (b) the binary is compatible under CUDA's same-major rule or is JIT-compiled from identical PTX, and (c) the first-load canary matches the reference digest. It is labelled `exact (derived certification)` in response metadata and the dashboard. A failed canary falls back to a fully certified provider, or marks the path approximate if none exists (RFC-0060 §6.4, ADR-0602 decision 7). **On a Vulkan-domain provider** (ADR-0405, V1): an `OpCert` exists for this device class, but the device or driver is not its rig's, or the device's identity is unknown. Condition (b) is replaced by: the device loads identical SPIR-V, with the `OpCert`'s `source_key` and `kernel` (the hash of the SPIR-V module), from the same shader-variant set as the rig. Conditions (a) and (c), the label and the fallback are as above |
| none | neither | never selected, in exact or approximate mode (RFC-0060 §4.2); selection falls through to the next provider holding a certificate, normally ggml |

Certificates are content-keyed: an `OpCert` survives every build in which the kernel's object bytes are unchanged (ADR-0602 decision 4). **The lab can certify more targets than it has.** CUDA runs a cubin built for compute capability X.y on any device of capability X.z with z ≥ y. RFC-0060's op conformance therefore loads the `sm_80` and `sm_86` objects explicitly on the `sm_89` rig, which certifies them without an Ampere card. Targets that no rig can execute (`sm_75`, `sm_90` and `sm_100` in today's lab) are the case derived certificates exist for. *Revised 2026-10-22 (ADR-0405):* on Vulkan they also cover the devices and drivers of a certified class other than the rig's. Each driver compiles the SPIR-V into its own machine code at pipeline creation, so a class-wide certificate would otherwise certify code no rig ran. QUAL-E6-T23 implements the rule. An `OpCert` that records no rig identity never certifies a Vulkan-domain op.

A manifest is valid when it is signed by a key the build's profile accepts (RFC-0060 §6.4's three signers). A build with no valid manifest selects ggml ops wherever ggml maps the op, plus the CPU provider's native ops for the ops ggml cannot run (§8.2: `LogitsProcess`, `TopKTopP`, `SampleSeeded`, `VerifyAccept`, `ApplyTableDeltas`, `KvPack`, `KvUnpack`, and the attention and mixing ops of families that need them), on single-device shapes, with the F32 wire, all labelled "uncertified" (ADR-0602 decision 6). Without those CPU ops the mode could not produce a token.

**Runtime spot check.** In `Warming`, each candidate runs once on fixed synthetic inputs at the plan's shape and is compared with `kernels::reference` on the CPU, within RFC-0042's per-op tolerance. It uses RFC-0042 §6.4's `tol_ref`, which equals the `tol_op` that RFC-0060's op conformance applies (its rule 3), so a certified kernel that passed in CI is not rejected on a user's machine for the same inputs. The comparison is against `kernels::reference` rather than ggml because the reference is always in the core. A failure is class `Correctness`: the candidate is struck for this (device fingerprint, build), and selection moves on. The cost is one reference evaluation per distinct (op, shape), milliseconds for decode shapes; large prefill GEMMs check a row subset.

#### 6.3 Microbenchmark protocol

| Item | Rule |
|---|---|
| Inputs | synthetic weights of the exact scheme and shape. For memory-bound ops, enough copies are rotated that the working set is at least 2× the device's last-level cache (L2 on GPUs, LLC per NUMA domain on CPUs, queried from the device), so every iteration reads from DRAM, VRAM or unified memory. A GEMV timed out of cache would overstate W_eff by the cache-to-DRAM ratio |
| Activations and KV | random F32 values cast to the op's activation dtype; KV page pools at the plan's expected context (`batching` prefill context, or 4096 tokens when unset) |
| Timing | device events around a captured loop of N launches; N grows until one batch lasts ≥ 2 ms; 3 warm-up batches, then the median of 5. The CPU uses `Instant` around N calls on the pool, NUMA-pinned |
| Launch overhead | measured once per provider as the cost of an empty captured node and of an uncaptured launch, so that `t_fixed` models get them (RFC-0010) |
| Shapes | every distinct (`OpKey`, `ShapeSig`) in the stage's step program, at its bucket, with `rows_active` at the bucket's typical value (1 for decode, `γ + 1` for verify) |
| Budget | `kernels.select_budget_ms` (default 3000 ms per runner) [target]. Ops are benchmarked in decreasing order of their share of the predicted step time. When the budget runs out, the rest take the provider's own `OpCaps::est` ordering, with ties to native. Every op is benchmarked at least once per (device fingerprint, build); later restarts hit the cache |

#### 6.4 Decision rule

**Domain per device first.** For each device the executor picks one memory domain (§2.3):

| Vendor | Order considered | How the choice is made |
|---|---|---|
| NVIDIA | `Cuda`, then `Vulkan`, then `Ggml(*)` | `Cuda` whenever the CUDA provider opens the device; else the first that opens |
| AMD | `Hip` and `Vulkan` | both benchmarked when both open; the domain with the lower predicted step time (sum over the step program of the per-op winners, plus segment-break costs) wins by a 5 % margin, else `Hip`. Research shows Vulkan winning decode on some AMD parts and HIP winning prefill on others (`backends-and-portability.md` §1.4), so the measurement decides |
| Apple | `Metal` | always |
| Intel GPU | `Vulkan`, then `Ggml(Sycl)` | as AMD when both open |
| CPU | `Host` | always |
| Other ggml-only (Adreno, Hexagon, CANN, …) | `Ggml(b)` | the only option |

**Op per domain second:**

```text
for each (op, shape) in the stage's step program, in decreasing predicted-time order:
    C = candidates passing §6.2
    if C is empty: the op is unsupported on this device → report to the solver (§11.3); stop
    for c in C: cost(c) = measured(c) + breaks(c) · t_submit + fences(c) · t_fence
        // breaks: graph segment breaks the candidate forces (capture_safe = false)
        // fences: cross-stream events it forces (unpatched ggml backends, §8.4)
    incumbent = the native candidate if present, else the cached choice, else argmin cost
    challenger = argmin cost
    choose challenger if cost(challenger) < 0.97 · cost(incumbent), else incumbent
```

The 3 % margin keeps choices stable across restarts despite benchmark noise. A choice flips only when a provider is clearly faster, and ties go to native as ADR-0002 requires.

#### 6.5 Selection cache

```rust
pub struct SelectionKey {
    pub device: Blake3,             // DeviceInfo::fingerprint
    pub class: DeviceClass,
    pub domain_builds: SmallVec<[ProviderBuild; 2]>,
    pub driver: String,
    pub op: OpKey,
    pub shape: ShapeSig,
}
pub struct SelectionEntry {
    pub provider: ProviderId, pub variant: u32,
    pub measured: Measured<Nanos>, pub weight_bytes: Bytes, // W_eff = weight_bytes / measured
    pub cert: CertLevel,            // Certified | Derived (§6.2)
    pub kernel_hash: Blake3,
    pub at_unix_ms: u64,
}
```

The node cache (`<state>/kernels/selection.v1`, postcard-encoded, one file per device) is shared by every model on the node. The pack's selection side record (`side/<pack-key>.select`, encoded by this crate, RFC-0041 §5.2) keeps the entries for its own ops, so a pack copied from another node with the same device class brings its choices with it (ADR-0016 decision 7), and the receiving node re-runs only the parity spot check. Any change in a key field is a miss. A cache hit still runs the runtime spot check, never the benchmark, and an entry whose `kernel_hash` no longer has a certificate in the current manifest is dropped.

#### 6.6 What selection feeds

- **Capability profile** (RFC-0010). `DeviceCaps::tier_bw` per (tier, `KernelClass`) and `throughput` per (`OpClass`, dtype) are measured with the kernel selection would choose, through `kernels::bench::measure(class, scheme, shape, tier)`. The probe calls this API, so the solver plans with the W_eff of the kernels that will actually run (cost model §5.3 item 1).
- **Cost model terms.** `t_submit` (one submission) and the doorbell round trip `t_switch` (one `signal_host` → host observes → host stores → `wait_host` releases) are measured per device in the probe through `kernels::bench`, and become `t_fixed` and `t_switch` inputs (cost model §3.2, §3.4).
- **Solver filter.** A device on which some op of a family has no candidate is reported per op, and the solver either moves that op's compute site or keeps the stage off the device (ADR-0002 item 7; §11.3).

### 7. Native providers

#### 7.1 CUDA (Tier 1; M0 hot ops, M1 complete)

**Runtime binding.** The provider runs inside `libsyntropy_provider_cuda` (§10.4). It loads `libcuda.so.1` / `nvcuda.dll` with `cudarc`'s dynamic loading, checks the driver version (≥ 580, ADR-0014), and retains each device's **primary context** (`cuDevicePrimaryCtxRetain`), so the CUTLASS shim, which uses the CUDA runtime, works in the same context. The CUDA runtime is linked dynamically from the module directory, where `setup` places NVIDIA's own redistributable archive; the provider library carries no NVIDIA code (§10.3). Syntropy's own kernels are loaded from the library's embedded fatbin with `cuLibraryLoadData` and launched with `cuLaunchKernelEx`, using parameter blocks from `csrc/params.h` (§1). The CUTLASS/CuTe GEMMs are launched through `extern "C"` shim functions linked into the same library (`syn_cuda_gemm_*`, hidden from its export table), which take a `CUstream`, buffer addresses, shape and a workspace, and do nothing else.

*Note 2026-10-31 (COMP-E4-T01): the driver is opened by its exact name.* The provider opens `libcuda.so.1` on Linux and `nvcuda.dll` on Windows itself, with `libloading`, and resolves the driver entry points it uses; it never asks the loader for the unversioned `libcuda.so`. `cudarc` supplies the driver API's types, but its own loader tries `libcuda.so` first. That name is a development link, and the loader may resolve it to a different file from the installed driver. On a WSL2 development host it resolved to a native-Linux driver copied into `/lib/x86_64-linux-gnu`, which cannot reach the WSL GPU, so `cuInit` reported no device, while `libcuda.so.1` resolved to WSL's driver. The same host's loader cache lists two `libcuda.so.1`, WSL's first and the copy second. So on that host the provider relies on the dynamic linker's search order for the soname, which puts WSL's directory first. The loader changes nothing beyond asking for the soname. On Windows it searches only `System32` for `nvcuda.dll` (`LOAD_LIBRARY_SEARCH_SYSTEM32`). Driver calls return their status as a plain integer and never as `cudarc`'s `CUresult` enum, so a code newer than the bindings stays a valid value. `doctor`'s DRV-008 (RFC-0053 §5) reports such a copy.

**Architectures** (ADR-0014: CUDA 13, sm_75 and newer):

| Target | Built as | Kernels that need it |
|---|---|---|
| sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120, sm_121 | cubin | every generic kernel (GEMV, attention, scans, elementwise, integer-dot GEMM, FP16/BF16 tensor-core GEMM) |
| compute_120 | PTX | forward compatibility for GPUs newer than the build; the driver JIT-compiles once and caches |
| sm_89, sm_120, sm_121 | cubin | FP8 E4M3/E5M2 tensor-core paths (`mma.sync` with FP8 operands) |
| sm_120f (sm_120a and sm_121a on toolkits without family targets) | cubin | MXFP4 and NVFP4 block-scaled MMA. The plain sm_120 target rejects these instructions, so they need the arch- or family-specific target; PTX for them is not forward compatible |

Datacenter-only paths (sm_90a warpgroup MMA, sm_100a tensor-memory MMA) are out of v1 scope: no Tier 1 lab hardware runs them (ADR-0020). Those GPUs run the generic cubins.

**Kernel design per hot op:**

- **Quantized GEMV** (`Gemv`, `GateUpAct`, `Logits`; every GGUF type, FP8, MXFP4, NVFP4, BF16, F16). One warp per group of output rows; 128-bit coalesced weight loads; dequantisation in registers; the activation vector staged once per block in shared memory. Integer-coded types (legacy quants, k-quants, i-quants) quantise the activation to 8-bit blocks per RFC-0042 §4.3 and use `dp4a` dot products. FP8 and FP4 types widen exactly to FP16 in registers and accumulate in F32. Rows 2–16 (verify windows, small batches) keep per-row accumulators in registers, so weights are read once whatever the row count.
- **Router** (`MoeRoute`). Fixed-order reductions only: no atomics and no split-K, so the same inputs give the same expert IDs and weights on every run and every member that replicates the router (a design rule, OQ-NET-22; RFC-0032). The same rule binds every provider's router (§7.2–§7.5); conformance runs each backend's router twice on identical inputs and requires identical output.
- **Grouped MoE** (`MoeGrouped`). Work items are (expert, output tile) pairs over the experts the assignment gives this site (§3.4). Each item reads its expert's weights once for every row that selected it, which is the property that makes a verify window cost one weight read per distinct expert, U(γ+1) of them (cost model §3.9).
- **Prefill GEMM** (`Gemm`, MoE prefill). Integer-coded weights: dequantise tiles to INT8 and use IMMA `mma.sync` (sm_75+), with activation tiles quantised per RFC-0042 §4.3. Float, FP8 and FP4 weights: CUTLASS/CuTe kernels through the shim, with BF16/FP16 tensor cores on every target, FP8 tensor cores on sm_89/120/121, and block-scaled FP4 on sm_120f. MoE prefill uses a grouped (pointer-array) GEMM over the experts touched by the chunk.
- **Attention decode** (`AttnDecode`). Split-KV: the context is cut into partitions processed by separate thread blocks, and a second pass merges the partial softmax statistics in partition order, so the result does not depend on scheduling. GQA groups all query heads of one KV head in one block, so KV is read once per KV head. MLA uses the absorbed form: queries are projected into the latent space and dotted with cached latents plus the decoupled RoPE key, so the latent cache is read once. Sliding-window kernels clamp the partition range and add sink logits. Sparse kernels read the indexer's selected blocks or tokens through the page table.
- **Attention prefill** (`AttnPrefill`). A tiled, causal, online-softmax kernel over paged KV with FP16/BF16 tensor cores, F32 softmax statistics.
- **Scans** (`ScanPrefill`, `ScanDecode`). Gated DeltaNet and KDA use a chunked formulation (chunk 64) for prefill and a fused recurrent update for decode; Mamba-2 uses a chunked state-space scan. States stay in F32.
- **Fusions.** RMSNorm fused with activation quantisation; RoPE fused into `KvWrite`; gated activation fused into `GateUpAct`; residual add fused into the output projection's epilogue.
- **Sampling at the tail.** `LogitsProcess`, `TopKTopP` (radix select), `SampleSeeded` (counter-based Philox per ADR-0107 and RFC-0042 §4.2: keyed by (client seed, choice index) whenever the request supplies a seed, else by (generated seed, `SequenceId`); indexed by the u64 `rng_counter`, read from the sequence's `SamplerState` in the step descriptor and reported with the step's tokens after the last draw, so the host carries it forward; draw order as ADR-0107 fixes it), so a sequence's random stream does not depend on batching and `VerifyAccept`, all on the GPU when the tail's head group lives there. Only accepted token IDs and optional log-probs cross to the host. Repetition and presence penalties are a sparse adjustment per row, applied before the softmax to the tokens the row has seen, never a dense pass over the vocabulary (OQ-PL-20; RFC-0012), confirmed by the `LogitsProcess` microbenchmark.

**Memory and capture features.** VMM for the expert cache when `VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` (else a fixed slab, ADR-0014); `cudaMemcpyBatchAsync` for `copy_batch` on CUDA 13 drivers; stream memory operations for doorbells (§4.2); graph capture (§4.4); peer access and a one-shot P2P all-reduce for `IntraNodeTp` (§7.6).

**Platform notes.** On Windows (WDDM) the provider reads whether hardware-accelerated GPU scheduling is on and records it in the capability profile; it never requests or changes it (§4.5). It treats graph replay as the only submission per step, because per-launch submission costs more under WDDM, and on display GPUs it honours the segment budget of §4.5. On WSL2 the provider reports the paravirtual device and leaves pinned-memory policy to RFC-0020.

**Health.** A sticky error (`CUDA_ERROR_ILLEGAL_ADDRESS`, `CUDA_ERROR_LAUNCH_FAILED` and the like) observed by `poll` or `health` makes every later call fail. The provider reports `DeviceHealth::Sticky`, and the runner exits with class `Device` (RFC-0001 §3.4).

#### 7.2 Metal (Tier 1; M1)

**Runtime binding.** `objc2-metal` bindings; `MTLCreateSystemDefaultDevice`. `DeviceClass::Metal { family, metal4 }` comes from `supportsFamily` (Apple7 and newer; macOS 15 floor) and the Metal 4 family check on macOS 26+. Both are computed at `enumerate` (§2.1): `metal4` from the family check, the OS version and `features.disable` (ADR-0405 decision 3).

**Two embedded libraries.** `metal4.metallib`, compiled with the Metal 4 language revision, uses `MTLTensor` resources and the Metal Performance Primitives tensor operations (`matmul2d`) for matrix tiles. On M5-class GPUs these run on the GPU neural accelerators; on older Apple GPUs they run on the shader cores. `metal3.metallib`, compiled with Metal 3.2, uses `simdgroup_matrix` 8×8 tiles. The provider loads the Metal 4 library when the OS and family allow it, and selection (§6) still benchmarks both where both load: a public characterisation of the M4-class tensor path found FP8 `matmul2d` emulated on the shader cores. Measurement, not the API version, decides.

**Kernel design.** The same op set as CUDA, adapted:

- GEMV and grouped MoE: simdgroup (32-lane) reductions, 16-byte loads, per-row accumulators for verify windows. Apple GPUs have no FP8 or FP4 matrix hardware in the published feature tables, so FP8, MXFP4 and NVFP4 widen exactly to FP16 or BF16 in registers (RFC-0042 §3).
- Prefill GEMM: `matmul2d` on Metal 4, `simdgroup_matrix` otherwise, with dequantisation into threadgroup tiles. Whether staging operands through threadgroup memory helps on the tensor-op path is not settled publicly, so both strategies exist as variants and selection picks one per device.
- Attention, scans, sampling: as CUDA, with simdgroup operations.

**Memory.** All Apple GPUs here are unified: weights, KV and caches are `Unified` tier memory owned by the tier manager and imported with `newBufferWithBytesNoCopy` (page-aligned, shared storage). Scratch comes from private heaps. Resources use `MTLHazardTrackingModeUntracked`; the executor knows every dependency and encodes explicit barriers. Every buffer joins the device's `MTLResidencySet` (macOS 15+) at creation. The usable size is `recommendedMaxWorkingSetSize`; Syntropy never raises the wired limit silently (ADR-0014, RFC-0020).

**Capture.** Indirect command buffers with argument buffers (§4.4); doorbells through `MTLSharedEvent` (§4.2). Long prefill steps are split into several command buffers so none runs past the macOS GPU watchdog (§4.5). On unified memory a MoE layer has no copy class (§3.4). Experts resident in unified memory are computed in place by the GPU. Misses whose home is `Nvme` are demand-read straight into unified memory (`F_NOCACHE`, RFC-0021) and then computed by the GPU, or by the CPU when the plan puts the miss site there.

**Health.** A command buffer that completes with `MTLCommandBufferStatusError` and a GPU fault error code is sticky for the runner: class `Device`.

#### 7.3 CPU (Tier 1; M0)

The CPU provider computes everything that runs on a CPU device: dense layers homed in `HostPinned`, expert misses in place (RFC-0001 §6.3), CPU-only nodes, and prefill on CPU stages. One CPU device exists per NUMA domain (RFC-0001 §5.1). Its "streams" are the CPU compute pool's work queues, its events are atomics, and capture is a no-op.

**ISA levels and detection.** `CpuIsa` is `core`'s type; this provider detects it (ADR-0405). It classifies the host at `enumerate`, which reports the level as the CPU device's class, and `open` applies the same detection and fills a per-op function-pointer table for that level (§2.1). Hot loops never test features.

*AMX at `enumerate`* (revised 2026-10-22, ADR-0405 decision 3). On Linux the AMX permission is granted per process, and `enumerate` (in the probe runner) and `open` (in the serving runner) may run in different processes. `enumerate` makes the request in its own process. It reports `X86V4Amx` only if the request is granted and `features.disable` does not list `amx`. `open` makes the request again. If it is denied there, `open` fails with `AmxDenied` (13), naming the mismatch, and the node re-probes, which yields `X86V4`. The provider never runs AVX-512 kernels under an `X86V4Amx` class, because the pack holds the `cpu.amx16` layout.

| `CpuIsa` | Requires | Detection |
|---|---|---|
| `X86V2` | SSE4.2, POPCNT | `is_x86_feature_detected!` (correctness floor only; not performance-gated) |
| `X86V3` | AVX2, FMA, F16C; AVX-VNNI used when present | `is_x86_feature_detected!`, which also checks OS support for the YMM state through XGETBV |
| `X86V4` | AVX-512 F/BW/VL/DQ; VNNI, BF16 and FP16 used when present | as above, including the OS ZMM/opmask state check |
| `X86V4Amx` | `X86V4` plus AMX-TILE and AMX-INT8 (AMX-BF16 when present) | CPUID leaf 7 (EDX bits for AMX-TILE, AMX-INT8, AMX-BF16), XCR0 bits 17–18 (tile config and data) through XGETBV, and on Linux a successful `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`; denied permission means `X86V4` |
| `Arm80` | NEON | always on AArch64 |
| `Arm82Dot` | NEON with SDOT/UDOT, FP16 arithmetic | `is_aarch64_feature_detected!` on Linux and Android; `sysctlbyname("hw.optional.arm.FEAT_DotProd")` and friends on macOS |
| `Arm86I8mm` | `Arm82Dot` plus I8MM (SMMLA) and BF16 | as above (`FEAT_I8MM`, `FEAT_BF16`) |
| `Arm92Sme2` | `Arm86I8mm` plus SME2 | `is_aarch64_feature_detected!("sme2")` on Linux; `hw.optional.arm.FEAT_SME2` on macOS. Syntropy reads the macOS sysctls itself because the standard library documents most AArch64 detection as Linux-only |

*Revised 2026-10-22 (SME2 detection on Linux):* on the toolchain `rust-toolchain.toml` pins (1.99.1), `is_aarch64_feature_detected!("sme")` and `("sme2")` do not compile: both names are unstable (E0658, `stdarch_aarch64_feature_detection`). The table above and the SME row below say otherwise. On Linux and Android the provider therefore reads `AT_HWCAP2` through `getauxval` and tests `HWCAP2_SME2` (bit 37), as ADR-0014's SME2 row names. On macOS it reads `hw.optional.arm.FEAT_SME2`, as before. Both read the same kernel-reported fact the macro reads, so switching to the macro once it is stable keeps the same result. Elsewhere (Windows on Arm) SME2 is reported absent. ADR-0402 item 3 carries the same note; RFC-0001 §18.14 records it.

**Stable Rust status of the intrinsic families** (checked 2026-10-19 against the Rust 1.99.1 standard library documentation, the Rust Reference's `target_feature` list, and the rust-lang tracking issues; this answers the follow-up recorded in ADR-0001):

| Family | Target feature on stable | Intrinsics on stable | Evidence | How Syntropy writes the kernels |
|---|---|---|---|---|
| AVX2, FMA, F16C | yes | yes (long-standing) | `core::arch::x86_64` stable docs | `std::arch` intrinsics |
| AVX-512 (F, BW, VL, DQ, CD, VNNI, BF16, FP16, VBMI, …) | yes, since 1.89.0 | yes, since 1.89.0 (for example `_mm512_dpbusd_epi32`, `_mm512_dpbf16_ps`) | Rust 1.89.0 release notes; stable intrinsic pages list "1.89.0" | `std::arch` intrinsics |
| AVX-VNNI (`avxvnni`, and the INT8/INT16 variants) | yes, since 1.89.0 | yes, since 1.89.0 (`_mm256_dpbusd_avx_epi32`) | stable intrinsic page | `std::arch` intrinsics |
| AMX (`amx-tile`, `amx-int8`, `amx-bf16`, …) | **no**: gated by `x86_amx_intrinsics` | **no**: nightly-only (`_tile_dpbssd` and the rest); tracking issue rust-lang/rust#126622 is open with no stabilisation PR | 1.99.1 docs mark them "nightly-only experimental" | stable `asm!`: one asm block per microkernel holding `ldtilecfg`, `tileloadd`, `tdpbssd`/`tdpbf16ps`, `tilestored` and `tilerelease`; tile registers never cross the block boundary, and every `tmm` register the block uses is declared clobbered |
| NEON, including dotprod, I8MM, BF16, FP16 | yes | yes, `core::arch::aarch64` | stable docs | `std::arch` intrinsics |
| SVE, SVE2 | yes (`sve`, `sve2` are listed in the Reference) | **no**: SVE intrinsics sit behind `stdarch_aarch64_sve`; tracking issue rust-lang/rust#145052 is open, and its RFC (rust-lang/rfcs#3838) is not accepted | tracking issue; Reference | **no SVE2 kernels in v1** (ADR-0402). The Tier 1 and Tier 2 Arm targets either lack non-streaming SVE (Apple silicon, Raspberry Pi 5-class) or use Arm-designed cores whose SVE vectors are 128 bits wide (GB10-class), where NEON with I8MM has the same width and needs no new code |
| SME, SME2 | **no**: absent from the Reference's stable list | **no** intrinsics; runtime detection of `sme`/`sme2` is stable (the detection macro has been stable since 1.60) | Reference; `is_aarch64_feature_detected!` docs | a statically linked AArch64 assembly unit (`src/sme2/*.S`, built by the `cc` crate with the platform assembler). Each exported function is a complete leaf routine that enters streaming mode with `smstart`, runs its outer-product loop on ZA, leaves with `smstop`, and preserves the AAPCS64 callee-saved registers. No Rust code ever runs in streaming mode |

ADR-0001's note asked for this record before M1. ADR-0402 turns it into policy, and it stays valid until AMX or SME intrinsics stabilise. Each Syntropy release re-checks the table (§Testing).

**Kernel design.**

- **GEMV over pinned weights** (the execute-in-place path). Output rows are partitioned across the NUMA-local pool, and each thread streams a contiguous run of weight bytes from its own node's memory. Integer-coded types use repacked `cpu.rows4`/`rows8` layouts (§5.4), quantise the activation to 8-bit blocks (RFC-0042 §4.3), and use `vpdpbusd` (AVX-512 VNNI or AVX-VNNI), `sdot`/`udot` (NEON), or `smmla` (I8MM, two rows at a time for verify windows). Float and FP8/FP4 types widen in registers and use FMA, or `vdpbf16ps` for BF16.
- **Expert misses** (`MoeGrouped` with every assigned expert at the `CPU` site). One task per (expert, output-row block), and all rows that chose the expert are processed in one pass over its weights. Outputs are written per (row, rank) in F32 to the host-mapped block (§3.4).
- **Compute-bound shapes** (prefill chunks, batches ≥ 16 rows on CPU stages). AMX tile kernels (`cpu.amx16` layout) on `X86V4Amx`, SME2 outer-product kernels on `Arm92Sme2`, otherwise blocked VNNI/I8MM GEMM. These matter where the plan computes prefill on a CPU stage below n* (cost model §3.2).
- **Attention, scans and sampling** for CPU-hosted layers and CPU-only nodes, with F32 softmax statistics and states.

The CPU provider is also a correctness tool: `kernels::reference` is separate code, so the CPU provider is itself checked against it and against ggml's CPU ops.

*Note 2026-10-23 (COMP-E3-T02):* the CPU provider's shell, as built in `syntropy_kernels_cpu::CpuProvider`.
- **Devices.** One device per NUMA node among the cores the process may run on (`syntropy_io::os::cpu_topology`). Each reports `DeviceClass::Cpu { isa }` from detection, and adds `GRAPH_CAPTURE`, `STREAM_MEM_OPS`, `HOST_REGISTER` and `HOST_IMPORT` to its ISA's features. The tier manager budgets RAM, so `ProviderDevice::memory` reports none.
- **Memory.** `alloc` takes only `HostMapped` and gives zeroed host memory aligned to at least 64 bytes. `register_host` records the region, and `import` wraps it without copying; the CPU sees host memory as it is. `Device`, `Unified` and `reserve_only` are `Unsupported`. A queued copy holds raw host addresses, so `free` defers deallocation while any stream has items left. The memory is released by the first `free` or `synchronize` that finds every stream drained, both between-step calls, and at the latest when the device drops. `poll` never deallocates, because a deallocation inside a step could take the allocator's lock (§2.4).
- **Streams.** Each stream is a preallocated `syntropy-spsc` ring of 4,096 items [target] that the provider's calls push onto. A `StreamRunner` runs the items in order on the thread that holds it. That can be a pool thread, after `take_runner`. It can be the device's own thread, through `CpuProvider::run`, which is the inline path behind `ComputePool::run_inline`. Or it can be `synchronize`. `run` never spins or parks: it returns at the first item that has to wait. `StreamRunner::drain` waits as a thread inside a step may, with an acquire spin and `pause` back-off, parking only on an armed `CpuWaker`. A full ring is refused, never grown.
- **Events and doorbells.** An event is a pair of atomic generation counters, and `poll` compares them. A record sets its generation with `fetch_max`, which gives timeline semantics, as §4.3 lets a backend's events follow: a wait for generation g passes once any record of g or later has run, from any stream. Doorbell flags are `AtomicU32`s, each on its own 64-byte line, 64 to a page. `signal_host` is a release store. `wait_host` is an acquire check, then the consumer's reset. `CpuProvider::doorbells` gives host threads the same flags.
- **The wake.** `arm_wake` takes a `CpuWaker` the waiter registered with `CpuProvider::register_waker`, whose `HostWaker` is `{ kind: Futex, handle: index }`. It queues the writer's wake: after the items before it, normally the doorbell's release store, the runner unparks the waiter if the waiter announced it is parking. The park is `std::thread::park_timeout`, which is a futex on Linux, `WaitOnAddress` on Windows, and a dispatch semaphore on Apple platforms.
- **Capture.** It records the items a stream's calls would push, and `replay` pushes them again. Each record and wait takes its event generation at that moment. A captured copy keeps its buffer handles, not their addresses, and each replay resolves them, so a buffer freed since the capture fails the replay rather than being written. A replay is all or nothing: every captured copy's buffers, and the ring's room for the whole list, are checked before any item is pushed. When COMP-E1-T05 gives launches buffer arguments, a captured launch keeps its buffers as handles under the same rules. A dropped device waits at most 30 s [target] for its taken runners to drain, then leaks their memory rather than free it under them.
- **Kernels.** `prepare` looks the op up in the ISA's dispatch table and returns `PreparedOp::with_handle`, whose handle is the kernel's index; `kernels` gained the handle field for this. `query` returns `None`, because each kernel task declares its op's capabilities. `OpArgs` is still COMP-E1-T05's placeholder, so a launch runs with `OpArgs::none()`, and tests add stand-in kernels with `CpuProvider::with_kernel`.
- **Health.** A kernel's error makes the device's health sticky with that error's code, and every later call fails with class `Device`. A caller's error that no §13 code names, such as a handle from another device or a full ring, is class `Internal` with the unassigned code 0.

*Note 2026-10-25 (COMP-E3-T07):* the SME2 assembly unit, as built in `syntropy_kernels_cpu::sme2`.
- **Build.** `build.rs` assembles `src/sme2/tiles.S` and `src/sme2/harness.S` through the `cc` crate for AArch64 macOS, Linux and Android only, and sets `cfg(syntropy_sme2)` there. Every other target builds no assembly; its hosts never report SME2.
- **Who calls the routines.** Only the holder of an `Sme2` token. `Sme2::for_level` gives one only for `Arm92Sme2` on a host whose detection inputs, read once per process with no kill switch, report SME2. `DispatchTable::for_level` stores the token, so a table below `Arm92Sme2`, which the `sme2` kill switch or a host without SME2 yields, has none.
- **Routines.** `syn_sme2_f32_tile` (one `FMOPA` per k step) and `syn_sme2_i8_tile` (one four-way widening `SMOPA` per group of four k values, exact `i32` sums) each compute a tile of up to 2·SVL_s × 2·SVL_s outputs in ZA tiles 0–3, with predicates at the edges. Each saves d8–d15, which `smstart` and `smstop` zero, and uses no other callee-saved register. The F32 tile equals a scalar chain of fused multiply-adds over k in increasing order bit for bit. `Sme2::gemm_f32` and `Sme2::gemm_i8` tile a whole product and allocate their packed panels; the `Gemm` op kernel will pack into its workspace instead.
- **ZA lazy saves.** A leaf routine cannot call `__arm_tpidr2_save`. Rust callers hold no ZA state, so TPIDR2_EL0 is zero on entry. If it is not, the routine returns 1 without touching memory or ZA, and the wrapper reports class `Internal`.

#### 7.4 HIP (Tier 2; M3)

- **Binding.** `libamdhip64` loaded dynamically; code objects for every gfx target on AMD's ROCm support list for the pinned release (RDNA3, RDNA4, CDNA2 and newer), plus Strix Halo's iGPU where AMD lists it, embedded in `libsyntropy_provider_rocm` (§10). Launch with `hipModuleLaunchKernel` and the shared `params.h`.
- **Source sharing.** Kernels live in `hsrc/*.hip` and include `common/portability.h`, which maps warp primitives, shared-memory declarations and vector types for both CUDA and HIP. Generic kernels (GEMV, attention decode, scans, elementwise) share one source; matrix-unit kernels are separate per family: WMMA on RDNA3/RDNA4 (FP16/BF16/INT8, FP8 on RDNA4) and MFMA on CDNA. Wave64 and wave32 variants exist where the code depends on wave size.
- **Features.** `hipStreamWaitValue32`/`hipStreamWriteValue32` for doorbells, HIP graphs, `hipHostRegister` for pinned arenas, peer access for intra-node TP. Where HIP graph capture of memory operations fails on a ROCm release, the spin kernels stand in (§4.2).
- **Not called by the native HIP provider:** rocBLAS, hipBLAS, hipBLASLt, MIOpen. Its GEMMs are Syntropy's own. The `rocm` module carries hipBLAS and rocBLAS for ggml's HIP backend only (§8.1); MIOpen and hipBLASLt are not shipped.

#### 7.5 Vulkan (Tier 2; ggml-only preview at M2, native at M3)

**The M2 preview.** Radeon, Arc, laptop integrated GPUs and Strix Halo would otherwise have no GPU path until M3. At M2 the `vulkan` module ships with `ggml-vulkan` as its only provider, labelled a **Tier 2 preview** in the API, CLI and dashboard (RFC-0050, RFC-0051):
- **Memory is ggml's.** Every buffer the preview device uses is allocated through ggml's Vulkan buffer type, and the tier manager records each allocation and owns its lifetime and budget (§8.3's fallback, ADR-0403 decision 2). There are no elastic arenas: no `map_segment`, no expert-cache segments that grow or shrink; the device's budget is fixed for the plan epoch (RFC-0020).
- **No capture and no doorbells.** ggml submits on its own queues, so every op is `capture_safe = false` and fenced with events (§8.4). `wait_host` is unsupported, so the executor splits a step at each host wait (§4.2 fallback 3); MoE layers with CPU misses therefore cost a split each, which the solver sees.
- **Coverage.** Level G wherever `ggml-vulkan` maps a family's required ops, F otherwise (§11.2), measured on the lab's Vulkan-capable rig. Its `OpCert`s come from the same conformance runs as any ggml op.
- **Not gated.** The Tier 2 coverage gate starts measuring at M3 (§11.4) and the Tier 2 promise applies from M4 (ADR-0020); the preview ships whatever level CI measures and says so.

**The native provider (M3)**, below, joins the same module.

- **Binding.** `ash` with the loader opened dynamically; Vulkan 1.3 minimum, 1.4 preferred (ADR-0014). Required: buffer device address, timeline semaphores, 8- and 16-bit storage, `shaderInt8`, `shaderFloat16`, subgroup size control. Used when exposed: `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`, `VK_KHR_shader_integer_dot_product`, `VK_KHR_shader_bfloat16`, `VK_EXT_memory_budget`, `VK_EXT_external_memory_host`.
- **Shaders.** Slang sources compiled offline by `slangc` to SPIR-V. Each kernel has variants over {subgroup 32, 64} × {cooperative matrix, subgroup arithmetic} × {integer dot, none}. `enumerate` fixes the subgroup and cooperative-matrix axes in the device's class by §2.2's rule, `open` loads the variant set that class names (ADR-0405 decision 3), and selection benchmarks the variants inside it. Pipeline caches persist per (device UUID, driver version) in the state directory.
- **Submission.** Pre-recorded command buffers per segment, split at every `wait_host` (§4.2). Each split is a submission, so Vulkan's `t_fixed` per MoE layer is higher than CUDA's or Metal's. That cost is measured and given to the solver, which can then prefer `Hip` on AMD or a coarser execute-in-place split.
- **Memory.** Device-local allocations bound to large buffers sub-allocated by the tier manager. Host registration uses `VK_EXT_external_memory_host` where the driver supports it; otherwise the tier manager adopts a `HOST_VISIBLE` allocation as its pinned arena for that device. Resizable-BAR memory (`HOST_VISIBLE | DEVICE_LOCAL` larger than 256 MiB) is reported as `REBAR` and may hold doorbell pages and step descriptors.

#### 7.6 Intra-node collectives (`IntraNodeTp`)

`CollectiveOps` exists on CUDA and HIP, the two domains whose events and peer access work across devices in one process:

```rust
pub trait CollectiveOps {
    /// Called once per member on its own dispatcher thread during Warming. `rendezvous` is a
    /// process-local object through which members exchange peer buffer addresses and flags.
    fn join(&self, ctx: &DeviceCtx, rendezvous: &Rendezvous, member: u8, members: u8) -> Result<CollectiveGroup, Error>;
    fn all_reduce(&self, ctx: &DeviceCtx, s: &Stream, g: &CollectiveGroup, buf: BufferSlice, dtype: DType) -> Result<(), Error>;
    fn all_gather(&self, ctx: &DeviceCtx, s: &Stream, g: &CollectiveGroup, src: BufferSlice, dst: BufferSlice) -> Result<(), Error>;
}
```

Decode-sized all-reduces (one hidden vector per row) use a **one-shot** kernel: each member writes its partial into its own buffer, raises a device flag, waits for every peer's flag with device-side polling, then reads all partials over NVLink or PCIe P2P and sums them **in member order**. Every member therefore computes a bit-identical result, which tensor-parallel layers need so that members never diverge (`multi-device-inference.md` §4.7). Larger messages use a two-shot reduce-scatter plus all-gather with the same fixed order. All of it is capturable. Without peer access, the solver does not choose `IntraNodeTp` (ADR-0014 fallback); the provider does not offer a host-staged collective.

#### 7.7 Kernel features other areas build on

These are part of the op contracts above; they are listed here so the owning RFCs can find them.

| Feature | Ops | Contract | For |
|---|---|---|---|
| Next-layer expert prediction | `MoeRoute` | with prediction enabled, the router op also applies the next `d` layers' router weights (exposed by the family, RFC-0043 §3.4) to the current hidden state, and writes the top `k + experts.predict_extra` experts per lookahead layer into the doorbell's prediction fields | RFC-0023 §9 |
| Tree masks | `AttnDecode`, `AttnPartial`, `KvWrite` | rows with `tree_parent` set (step descriptor, §3.5) attend to the committed context plus their ancestors in the window only | RFC-0012 tree verify, RFC-0022 §6 |
| Per-position recurrent-state snapshots | `ScanDecode` | in verify windows, the scan writes the state after each window position into snapshot slots the KV store names, so rejection rolls back to the accepted position without recomputation | RFC-0022 §6, RFC-0012 recurrent chains |
| Per-row penalties along tree paths | `LogitsProcess` | repetition and presence penalties count tokens along each row's path (committed tokens plus ancestors in the window), from per-sequence count tables plus the window's parent array | RFC-0012 §8 |
| Quantized-KV attention (approximate only) | `KvWrite`, `AttnDecode`, `AttnPartial` | variants that store and read KV below the model's KV dtype (Q8 and 4-bit blocks) under the `kv_lossy` knob; their `Numerics` declare the lowered storage, so selection never offers them in exact mode | RFC-0022 §9, RFC-0042 §10 |
| GPUDirect Storage port | — (not an op) | **Not built in v1** (RFC-0021 §9; ADR-0014): GDS direct mode is documented for data-centre and workstation GPUs only, so consumer cards would take the host path anyway. The design is kept for the revisit: `kernels-cuda` exposes `GdsPort`: `cuFileDriverOpen` and property checks, `cuFileHandleRegister` on `O_DIRECT` descriptors, `cuFileBufRegister` on VRAM expert-cache segments (all at `Loading`), `cuFileStreamRegister` on the copy-in stream, and `read_async`, which enqueues `cuFileReadAsync` into a slot on the copy-in stream and returns at once; completion is ordered on that stream like any copy. The blocking `cuFileRead` is never called on the dispatcher thread. `open` measures the enqueue cost once, and if it blocks (cuFile's compatibility mode) or the call is missing (cuFile older than CUDA 12.2's), `GdsPort` is not offered and reads take RFC-0021's host path. cuFile is loaded dynamically, only where present; it is opportunistic and never required (ADR-0013) | RFC-0021 §9 |
| Batched copies | `copy_batch` | `cudaMemcpyBatchAsync` on CUDA 13; loops elsewhere | RFC-0021 §8, RFC-0023 §6 |

### 8. The ggml provider

ADR-0002 gives the ggml provider two jobs: coverage on every backend ggml supports, and the op-level correctness reference for native kernels. ADR-0403 records the integration choices below.

#### 8.1 What is linked, and how

- **Only the tensor library.** `third_party/ggml/` holds ggml's `ggml/` tree (ggml-base, the CPU backend and the GPU backends) at the commit in `third_party/ggml.lock` (RFC-0062 §4.2). Its model library is never vendored or linked (ADR-0002). *Note 2026-10-23:* this bullet said `vendor/ggml`; RFC-0062 §4.2, the CI path rules and the repository use `third_party/ggml/`, which is the path.
- **Statically, with one backend per artefact.** ggml is built without its dynamic-backend loader. Each artefact links its own static copy of ggml-base together with the backends it serves:
  - the core: the CPU backend, plus the Metal backend on macOS;
  - each provider library: that module's GPU backend (CUDA, HIP, Vulkan or SYCL);
  - the iOS and Android apps: every backend they use (CPU and Metal on iOS; CPU, Vulkan, OpenCL and Hexagon on Android).
  All ggml symbols are hidden (§10.4), so copies in the core and in modules never interpose, and each copy registers only its own devices.
- **Vendor BLAS comes with ggml's GPU backends.** ggml's CUDA backend links cuBLAS (and cuBLASLt through it), and its HIP backend links hipBLAS and rocBLAS; `mul_mat` routes float weights and larger batches to them. The `rocm` module therefore ships hipBLAS and rocBLAS as unmodified redistributables, which their licences permit, while the `cuda13` provider library carries only Syntropy code and `setup` fetches cuBLAS and cuBLASLt from NVIDIA and places them, unmodified, beside it (§10.3). §8.2 marks the ggml ops that may reach these libraries as not capture-safe. Their size counts towards the module's download (RFC-0052 §12), as fetched bytes for `cuda13`; the M1 measurement in §10.6 answers RFC-0052 open question 3.
- **The core's ggml CPU backend is built for the baseline ISA** (x86-64-v2, ARMv8.0), because ggml selects CPU variants only through its dynamic loader. It is built with floating-point contraction off (`-ffp-contract=off`), so its dequantisation and row gathers match `kernels::reference` bit for bit on every architecture (RFC-0042 §4.1 rule 7). That is acceptable for its two jobs, the CPU fallback for rare ops and the calibration baseline, which ADR-0422 measures with exactly this build. Speed on CPUs comes from the native CPU provider.
- ggml is **not** needed at runtime for correctness: runtime parity spot checks use `kernels::reference` (§6.2). ggml is needed for op coverage, and in CI as the reference.

*Note, 2026-10-31 (COMP-E7-T01): the core's static build, as built.*
- **What is vendored.** `third_party/ggml/` holds only what the core builds today: ggml's licence, its public headers, ggml-base's sources and the CPU backend for x86-64 and AArch64. The Metal and GPU backends join with the tasks that build them. The tree is patched as RFC-0062 §11.3's note of this date describes.
- **The build.** `crates/kernels-ggml/build.rs` compiles a named file list with the `cc` crate, which the workspace already uses, rather than ggml's CMake. So the build adds no dependency, needs no CMake, and does not vendor ggml's CMake files. Its flags:
  - `-march=x86-64-v2` (MSVC: `/arch:SSE4.2`) or `-march=armv8-a`;
  - `-ffp-contract=off` (MSVC keeps its default `/fp:precise`, under which it does not contract);
  - `-fvisibility=hidden` and `-fvisibility-inlines-hidden`.

  An optional CPU sub-backend is left out, and the CPU backend's weight-repacking buffer type is not registered (`GGML_USE_CPU_REPACK` is not defined), so weights stay in their stored layout.
- **No C++ runtime library on Linux.** The CPU backend is C++. So that the core needs no `libstdc++` (rule 7 refuses it), `build.rs` links the GNU C++ runtime statically. The basis is the GCC Runtime Library Exception 3.1, which covers static linking of `libstdc++` into a program from an eligible compilation process. Whether `THIRD-PARTY-NOTICES` should still carry a notice for it is for the maintainers, under ADR-0018 (OQ-PKG-15). On macOS, `libc++` is the system C++ runtime, and the core loads it. On Windows, Rust's own code already imports the Visual C++ runtime (`vcruntime140.dll`) and the Universal CRT's API sets. Only ggml's C++ code adds `msvcp140.dll`, the same redistributable's C++ library, and three more CRT API sets (time, convert, environment), which rule 7 now lists with dated reasons. Linking the static runtime (`/MT`) instead would need `+crt-static` for the whole binary, which this task leaves alone.
- **The bindings.** `crates/kernels-ggml/src/bindings.rs` is generated once by a pinned `bindgen` and checked in, so a build needs no libclang. `ggml.lock` and the file's header record the version and the command.
- **Hidden symbols are checked on the core binary.** `syntropy` links the copy, and `--version` reports its commit. Rule 7 of `cargo xtask ci build` fails if the binary exports any `ggml_` or `gguf_` symbol (ELF `.dynsym`, the PE export directory, the Mach-O external symbols). `crates/cli/tests/ggml_symbols.rs` checks that the binary does hold ggml's functions, local to it, so that check is never vacuous.

#### 8.2 Mapping one Syntropy op onto ggml

`prepare` builds a small **template graph** for the (`OpKey`, `ShapeSig`): a `ggml_context` holding tensor descriptors with no data, the ops below, and a `ggml_cgraph`. It then asks the device's ggml backend whether every node is supported (`ggml_backend_supports_op`). `query` returns `Some` only when all are. `launch` binds the tensors' `data` pointers to the `OpArgs` buffers (field writes, no allocation) and calls the backend's asynchronous graph compute.

| `OpKind` | ggml template | Capture-safe on patched CUDA/HIP |
|---|---|---|
| `Gemv`, `Logits` over quantised weights at decode and verify buckets | `mul_mat` over a weight view on Syntropy memory | yes: these shapes use ggml's own quantised kernels |
| `Gemm`, and any `mul_mat` over float weights or at prefill and batch buckets | `mul_mat` | no on CUDA and HIP: ggml may route them to cuBLAS or hipBLAS, whose handles and workspaces it creates lazily outside patch P3. The executor launches them between captured segments (§4.4 rule 5) |
| `GateUpAct` | two `mul_mat` plus the gated activation op | as the `mul_mat` rows above |
| `MoeRoute` | `soft_max` or sigmoid, bias add, `top_k`, `get_rows` for weights | yes |
| `MoeGrouped` (GPU site) | one `mul_mat` triple per expert of this site, data pointers taken from the host mirror of `slot_of`/`slot_addr` after the dispatcher reads the doorbell | no: addresses depend on the step's routing, so the dispatcher launches it after `Rd[l]` |
| `MoeGrouped` (CPU site) | one `mul_mat` triple per assigned expert, from a template sized for `top_k × rows` experts | n/a (CPU) |
| `AttnDecode`/`AttnPrefill` (`Gqa`, `SlidingGqa`) | `get_rows` of the sequence's KV pages into contiguous scratch, then `flash_attn_ext` with mask and sinks | yes; the gather is an extra copy that native kernels avoid |
| `AttnDecode` (`Mla`) | absorbed-form `mul_mat` chain over gathered latents | yes |
| `ScanDecode`/`ScanPrefill` | ggml's scan ops for the kind, where the pinned commit has them | per op |
| `RmsNorm`, `Rope`, `Act`, `Add`, `Mul`, `RowGather`, `Convert` | `rms_norm`, `rope_ext`, unary ops, `add`, `mul`, `get_rows`, `cpy` | yes |
| `AttnPrefill(Bidirectional)` | `flash_attn_ext` with no mask over the step's own K and V | yes |
| `Pool(*)` | no mapping; native only (CPU, CUDA) | — |
| `EmbedInject` | no mapping; reserved, reference only in v1 | — |
| `LogitsProcess`, `TopKTopP`, `SampleSeeded`, `VerifyAccept`, `AttnIndex(*)`, `AttnPartial(*)`, `AttnMerge`, `HyperMix`, `Timestamp`, `TapGather` | no mapping; native only | — |
| `ApplyTableDeltas`, `KvPack`, `KvUnpack` | no ggml op: the dispatcher applies deltas with a host-side `copy` before the step, and page packing uses `copy_batch` | — |

Ops with no ggml mapping must have a native CPU implementation; the CPU provider is the universal fallback (§11.2).

*Note, 2026-10-31 (COMP-E7-T02): the mapping layer on the CPU backend, as built.* `crates/kernels-ggml/src/mapping.rs` holds it: `query`, `prepare` and `launch` over template graphs, which the provider's shell will call.
- **Mapped:** `Gemv`, `Gemm` and `Logits` (`mul_mat`); `GateUpAct` with SiLU (`mul_mat` over a fused weight, gate rows first, then `swiglu`); `RmsNorm` (`rms_norm`, then `mul` by the gain, with `eps` from `OpArgs::statics`); `Act(Silu)`, `Add`, `Mul`; `RowGather` (`get_rows`, each index checked against the table first, because ggml does not bound it); `Convert` between F32, F16 and BF16 (`cpy`). Matrix-product weights are F32, BF16 or the GGUF types `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` and `Q2_K`–`Q6_K`; `query` reports the activation quantisation ggml applies, read from ggml's CPU type traits. F16 weights are not mapped, because ggml rounds their activations to F16, which `ActQuant` cannot state.
- **Not mapped yet:** `Rope`, `AttnDecode`/`AttnPrefill`, `MoeRoute`, `MoeGrouped` and the GELU forms of `Act` and `GateUpAct`. Their operands, and the GELU form, come with `kernels::reference` (COMP-E1-T03); their mappings are COMP-E7-T19.
- **Memory.** Intermediate tensors and the work buffer of ggml's compute plan sit in the `OpArgs` workspace, which `OpCaps::workspace` declares (§5.3). Each template has its own one-thread ggml threadpool, so a launch allocates nothing.
- **Conformance.** Until `kernels::reference`'s op functions exist (COMP-E1-T03), the tests compare each mapped op with an F64 evaluation of its definition: bit for bit where an element is one IEEE operation, within a derived F32 rounding bound otherwise. COMP-E7-T19 re-runs them against `kernels::reference`. Results differ between x86-64 and arm64 within those bounds: the 8-bit activation rounding path and arm64's explicit fused multiply-adds differ. `deterministic` holds per run on one machine, so ggml baselines and `OpCert`s are not shared across architectures.
- **Rows.** A launch reads `rows_active` from the step descriptor and sets the row extent of each tensor to it, so only the active rows are computed (§3.2).
- **`NDEBUG`.** The core's ggml build now defines it, as ggml's own release builds do. Without it, a C `assert` in `rms_norm` aborts the process on a row holding a NaN or an infinity, which must propagate instead. `GGML_ASSERT` stays on.

#### 8.3 Buffer interop

ggml never decides what is resident (ADR-0002 item 4). How Syntropy memory reaches ggml, per backend (capability facts from `backends-and-portability.md` §1.3.1):

| ggml backend | Mechanism | If it fails |
|---|---|---|
| CPU | `buffer_from_host_ptr` (advertised) over tier-manager memory | — |
| Metal | `buffer_from_host_ptr` (advertised) over page-aligned unified memory | — |
| Vulkan, SYCL | the function is registered but not advertised. At `open`, the provider wraps a 1 MiB test region, runs a copy and a `mul_mat` through it, and compares the result. Pass: use it. The M2 Vulkan preview skips the probe and always takes the fallback (§7.5) | the tier manager asks ggml to allocate the buffer through the backend's buffer type, records the allocation, and owns its lifetime and budget |
| CUDA, HIP | not offered upstream. Local patch P1 adds a non-owning buffer over an existing device pointer | as Vulkan |
| others | buffer-type allocation, recorded by the tier manager | — |

Weight layouts for ggml ops are `SOURCE` (GGUF block types), RFC-0042 §3.5's exact transcriptions of safetensors tensors, or `widen.f32` copies (RFC-0042 §3.3); ggml's own repacked CPU layouts are not used, because they would make ggml own a second copy.

#### 8.4 Streams and the local patch queue

Unpatched ggml GPU backends submit on their own streams or queues. The provider then brackets each ggml op with an event wait and an event record against the executor's stream, and reports `capture_safe = false`, so selection charges a fence and a segment break (§6.4). Four patches, each sent upstream first (ADR-0002 item 8), remove that cost:

| Patch | Backend | Change | Size target |
|---|---|---|---|
| P1 | CUDA, HIP | non-owning buffer over an existing device pointer | ≤ 150 lines |
| P2 | CUDA, HIP | compute on a caller-provided stream, with ggml's internal graph capture disabled for that backend instance | ≤ 150 lines |
| P3 | CUDA, HIP | temporary-memory pool backed by a caller-provided workspace, so nothing is allocated during compute after warm-up | ≤ 250 lines |
| P4 | Metal | encode into a caller-provided command buffer and queue | ≤ 250 lines |

**Who sends them.** Agents write and test the patches like any other change (ADR-0021). Sending them upstream follows ggml's own contribution rules. Draft changes go upstream as soon as the S-04 spike settles the buffer contract in M0 (COMP-E7-T16; OQ-KB-03). Until a patch is accepted upstream, it is carried in the local queue within the budget below.

**Patch budget:** at most 6 local patches and 1000 changed lines in total, measured at every pin bump. Exceeding it for two consecutive bumps triggers ADR-0002's revisit clause (an explicit fork, kept as a standing stream of tasks with budgeted agent and CI capacity).

**Pin-bump cadence:** once per Syntropy minor release, and at most every 4 weeks. A bump is a tracked task: rebase patches, regenerate bindings, run the full op conformance and coverage matrix (§11; RFC-0060 §4.2), which re-issues every ggml `OpCert`, because ggml's kernel hash includes the commit (§10.2). Out-of-cycle bumps happen only for a security fix or for a new op that a family in the current milestone needs.

#### 8.5 The reference role

In CI on every Tier 1 device class, each native kernel's output is compared with the ggml op's output for the same inputs, within RFC-0042's per-op tolerance (ADR-0002 item 6). Where the device's ggml backend lacks the op, the ggml CPU op is the reference. Where ggml has no mapping at all (sampling, indexers, `HyperMix`), `kernels::reference` is the reference, and RFC-0060 requires a second, independent implementation in the test suite for those ops.

### 9. Hot-op kernel list and roofline targets

The roofline time of an op on device d is `t_roof = max(B / W_peak, F / F_peak)`, with B the bytes the op must read and F its arithmetic (cost model §3.1). Each target below is a **target**: the fraction of that bound the native kernel must reach on Tier 1 hardware, measured by the RFC-0061 microbenchmarks with weights out of cache (§6.3). W_peak and F_peak are vendor figures; SC-1 defines η against peak bandwidth.

| # | Op (shape class) | Bound at batch 1 | Target | Derivation |
|---|---|---|---|---|
| 1 | `Gemv`, `GateUpAct`, `Logits` on `Vram`/`Unified` (decode, rows = 1) | memory | η ≥ 0.80 | SC-1, dense |
| 2 | Same, verify windows and small batches (rows 2–8) | memory while rows < ridge (≈ 49 at b = 0.6 on an RTX 4090; ≈ 10 on M3 Ultra; cost model §3.1) | time ≤ 1.10 × time(rows = 1) | chosen: weight bytes dominate; activation bytes per row are `k × 4` |
| 3 | `MoeGrouped`, GPU site (decode and verify) | memory | η ≥ 0.63 over the distinct experts' bytes | SC-1, MoE |
| 4 | `Gemv`, `MoeGrouped` on the CPU over `HostPinned` (execute-in-place) | memory, or ALU for arithmetic-heavy schemes | ≥ 0.75 × min(the probe's all-core DRAM read rate for that NUMA domain, `R_alu(scheme)`), where `R_alu` is the same kernel's all-core rate in weight bytes per second on a matrix that fits in cache | SC-1, CPU. Gated only for schemes whose `R_alu` exceeds the DRAM rate on that device class (expected: Q4_K, Q8_0, MXFP4, BF16); i-quants and other ALU-bound schemes are measured and reported, not gated |
| 5 | Same, rows 2–8 on the CPU | memory up to the CPU ridge (≈ 3–9 rows at Q4, cost model §3.1), then compute | time ≤ max(row-1 time, 2 · P · rows / (0.6 · F_cpu,peak)) | chosen; F_cpu peak from the ISA (VNNI/I8MM int8 rate × cores × clock) |
| 6 | `AttnDecode` (`Gqa`, `Mla`, `SlidingGqa`) at context ≥ 4096 | memory (KV read) | η ≥ 0.70 over KV bytes read | chosen between rows 1 and 3: paged reads through a page table cost some efficiency against a dense GEMV, and less than scattered experts |
| 7 | `AttnDecode` (`BlockSparse`, `TopKSparse`, `Compressed`) | memory (selected KV) | η ≥ 0.55 over the selected bytes | chosen: gathers of selected blocks are less regular than paged dense reads |
| 8 | `Gemm` and grouped MoE `Gemm`, prefill chunks ≥ 512 rows, tensor-core dtypes | compute | ≥ 0.60 × vendor dense peak for the dtype | cost model §4a assumes 100–165 of 165 TFLOPS effective, so 0.6 is its lower end |
| 9 | `AttnPrefill` at 4K–32K | compute | ≥ 0.50 × vendor dense FP16/BF16 peak | chosen: causal masking idles part of each tile |
| 10 | `ScanDecode` | state read and write | time ≤ max(2 × (state bytes read + written) / W_peak, 10 µs) per layer, inside a captured graph | chosen: per-sequence states are small, so the floor covers latency-bound cases; revisited with M1 data |
| 11 | Sampling at the tail (`LogitsProcess` + `TopKTopP` + `SampleSeeded`) | memory (logits) | time ≤ 3 × vocab · 4 B / W_peak | chosen: three passes over F32 logits in the worst case |
| 12 | Doorbell round trip (`signal_host` → host sees it → host stores → `wait_host` releases) | latency | measured and reported, no gate before M1 data; it is `t_switch` in the cost model | cost model §3.2 estimates 5–20 µs; the probe replaces the estimate |

Rows 1, 3 and 4 are release gates on Tier 1 from M1 (RFC-0061). The rest are tracked and become gates once M1 measurements exist. Every target applies per (device class, quant scheme) on the native provider. The ggml provider has no targets; it is measured, and selection uses it only where it is faster.

### 10. Build system, kernel distribution and the module ABI

ADR-0400 records the decisions. ADR-0623 and RFC-0062 §5 own the kernel object cache and its trust rules; ADR-0507 and RFC-0052 §4 own how modules are packaged, chosen, fetched, verified and installed. This section owns how kernels are structured and built, and the in-process ABI between the core and a provider library.

#### 10.1 Toolchains

Every kernel toolchain is pinned by exact version (RFC-0062 §4.2), and the `toolchain_id` of ADR-0623 is its full version banner:

| Toolchain | Builds |
|---|---|
| CUDA 13.x (`nvcc`, `ptxas`, `fatbinary`) | `libsyntropy_provider_cuda` for module `cuda13`, including ggml CUDA |
| CUDA 12.x | the Tier 3 `cuda12-legacy` provider library (sm_60, sm_61, sm_70): ggml CUDA plus the native GEMV, MoE and attention-decode kernels |
| ROCm, current release (10.1.0 as of 2026-10; runtime minimum 7.2, ADR-0014) | `libsyntropy_provider_rocm`, including ggml HIP |
| Xcode (`metal`, `metallib`) | the two metallibs in the macOS core and the iOS app |
| Slang (`slangc`) | SPIR-V for `libsyntropy_provider_vulkan` |
| CMake, Ninja, platform C/C++ compiler | ggml in every artefact; the SME2 assembly unit |
| CUTLASS (header-only, vendored at a pinned tag) | the CUDA GEMM shim |
| oneAPI (for ggml SYCL) | `libsyntropy_provider_sycl` (Tier 3) |

#### 10.2 Translation units and the kernel object cache

Structure rules, which RFC-0062 §5.3 and ADR-0623 decision 2 require of the kernel crates:

1. **Many small translation units**, one per (op family, quant-scheme group): for example `csrc/gemv/kquant.cu`, `csrc/gemv/fp4.cu`, `csrc/moe/grouped_kquant.cu`, `csrc/attn/decode_gqa.cu`. A kernel change recompiles a few units × architectures, never a whole provider.
2. **Explicit instantiation lists.** Every template instantiation (tile shape, scheme, fusion bits) is listed in a generated file per unit, `instantiate/<unit>.inc`, produced by `cargo xtask kgen` from the op registry and each provider's support table (`kernels-*/support.toml`). The list is part of the object key, so adding an instantiation is a visible, cached change. No unit relies on implicit instantiation from another.
3. **Per-target objects, linked at the end.** Each unit compiles once per target (`sm_XX`, `gfxNNNN`, Metal language revision, SPIR-V variant). The per-architecture objects are linked into one fatbin, code-object bundle, metallib or SPIR-V set when the provider library or core is linked.
4. **Through the cache.** Each `kernels-*` crate's `build.rs` compiles through `cargo xtask kcache`, which computes `SourceKey` and `ObjectKey` exactly as ADR-0623 defines, looks the object up, and compiles only on a miss. Local builds use a local directory cache. CI uses the project bucket with ADR-0623's trust split: only `main`, nightly and release builds write `kcache/`. Contributors without device toolchains read `kcache/` and build everything else (`SYNTROPY_KERNELS=prebuilt`, the default); `SYNTROPY_KERNELS=none` builds providers that enumerate nothing.
5. **Architecture sets follow the build profile** (RFC-0062 §4.3): PR builds compile only the lab rigs' architectures, including the arch-specific (`a`/`f`) variants the rigs' kernels use, so FP4 block-scaled kernels are exercised on R-BW in PR† runs; nightly and release builds compile §7.1's full list.
6. **No fast-math flags** in any kernel compilation (RFC-0042 §4.1).
7. **Contraction off for bit-exact-class units.** Translation units holding bit-exact-class kernels (dequantisation, row gather, copies, KV page moves, the wire codec, index ops; RFC-0042 §4.1 rule 7, §6.4) are compiled with `--fmad=false` (CUDA, HIP), `-ffp-contract=off` (C++), safe math mode (Metal) or `precise` expressions (Slang), with flush-to-zero off and power-of-two scales built from exponent bits (RFC-0042 §4.1 rule 7, §6.4 "Bit-exact scope"). Those flags are part of the unit's `ObjectKey`, and L0 fails a bit-exact-class unit built without them (RFC-0060 §4.2 rule 4).

**Kernel hashes for certification.** Each prepared op reports the hash its `OpCert` is keyed on (`OpCert::kernel`, RFC-0060 §6.4), through `OpCaps::kernel_hash`:

| Provider | `kernel_hash` |
|---|---|
| CUDA, HIP, Metal, Vulkan | BLAKE3 of the compiled object (cubin, code object, AIR object, SPIR-V module) for the device's target that contains the kernel: the object the cache stored under its `ObjectKey` (ADR-0623 decision 1) |
| CPU (Rust, compiled into the core) | BLAKE3 over (the kernel's Rust source files ‖ `toolchain_id` ‖ ISA level ‖ `asm!`/assembly sources). Rust codegen output is not cached per kernel, so the source-plus-toolchain key is the stable identity |
| ggml | BLAKE3 over (`third_party/ggml.lock` commit ‖ the local patch set ‖ `toolchain_id` ‖ backend ‖ build flags) |

A provider library therefore carries a table of (`OpKey`, shape class, target) → kernel hash, generated at link time.

#### 10.3 Artefacts

| Artefact | Providers inside | Kernel payload | Where it ships |
|---|---|---|---|
| core binary, Linux and Windows | CPU native, ggml CPU | CPU kernels for every ISA level | every install |
| core binary, macOS | CPU native, ggml CPU, Metal native, ggml Metal | plus both metallibs | every install; no modules on macOS |
| module `cuda13`: `libsyntropy_provider_cuda` | `cuda`, `ggml-cuda` | Syntropy code only: fatbins for §7.1's targets and the CUTLASS shim. NVIDIA's libraries (the CUDA runtime, and cuBLAS and cuBLASLt for ggml) are not in the module; `setup` fetches them from NVIDIA (below) | after detection (RFC-0052 §4.2) |
| module `cuda12-legacy` | `cuda` (reduced set), `ggml-cuda` | sm_60/61/70 objects; CUDA 12 libraries fetched from NVIDIA the same way | after detection, Tier 3 |
| module `rocm`: `libsyntropy_provider_rocm` | `hip`, `ggml-hip` | code objects for every listed gfx target; ROCm user-space redistributables (MIT and NCSA): HIP and CLR, rocBLAS and hipBLAS for ggml. RCCL, ROCgdb and the binary-only pieces are excluded; hipCUB's notice ships with the library if its headers are compiled in (OQ-PKG-10) | after detection |
| module `vulkan`: `libsyntropy_provider_vulkan` | M2: `ggml-vulkan` only, a Tier 2 preview (§7.5). M3: `vulkan` and `ggml-vulkan` | M2: ggml's SPIR-V; M3: Syntropy's SPIR-V variants too; no vendor libraries | after detection |
| module `sycl`: `libsyntropy_provider_sycl` | `ggml-sycl` | — | on request, Tier 3 |
| iOS app, Android app | as the platform uses, all static | as above | the app |

**NVIDIA's libraries are fetched from NVIDIA, never shipped.** The fetch keeps installation one command with no toolkit step, and Syntropy's artefacts then redistribute none of NVIDIA's files. A CUDA module's signed `module.toml` (RFC-0052 §4.1) lists each NVIDIA redistributable archive it needs, from NVIDIA's own redistribution server, with the archive's SHA-256 pinned. `setup` shows NVIDIA's licence terms once, fetches each archive, checks it against the pinned hash and extracts its files unmodified beside the provider library (RFC-0052 §4.4; PROD owns the fetch route). Consequences for this RFC:
- the provider library loads NVIDIA's libraries through an `$ORIGIN` run path on Linux and from its own directory on Windows; it never edits NVIDIA's files (no `patchelf`) and never ships the driver library (`libcuda.so`), which always comes from the installed driver;
- NVIDIA's files stay outside the module's `MIT OR Apache-2.0` licence statement (ADR-0018 decision 5);
- the user's installed CUDA toolkit is the last fallback when the fetch is impossible (ADR-0507 rule 7);
- channels that would have to carry the libraries themselves (deb and rpm module packages, OCI images at M3, offline bundles at M4) are not built until NVIDIA's terms are confirmed to permit it for that channel (**Maintainer action**).

#### 10.4 The provider-library ABI

A provider library is a `cdylib` built in the same `cargo` invocation as the core, so the two always share one build ID (ADR-0507 rule 3). Inside, each provider is an ordinary Rust implementation of `KernelProvider`. The boundary is a C ABI generated from the trait. `kernels::abi::export!` emits the module side's `extern "C"` thunks inside the module's own `kernels-*` crate, and `kernels_loader::ModuleProvider` wraps an imported table back into a `KernelProvider` for the core. Statically linked providers (CPU, Metal, mobile) skip the ABI entirely.

**Exported symbols.** Exactly one: `syntropy_provider_entry_v<N>`, where N is the ABI major version, recorded as `provider_abi = N` in `module.toml` (RFC-0052 §4.1). The major version is in the symbol name, so a core never even finds the entry point of a library of another major version.

- Linux: a linker version script exports only that symbol, under the ELF version node `SYNTROPY_PROVIDER_<N>`. Everything else is local: Rust's standard library, the module's own copy of ggml, the CUTLASS shim and the static CUDA runtime. The library links with `-Bsymbolic`, so its internal references never bind to symbols of the core or of another module.
- Windows: the export table holds only that symbol (a `.def` file). Vendor DLLs resolve from the module directory only (RFC-0052 §4.3).
- macOS has no modules.

**Entry point and tables.**

```c
typedef struct { uint8_t magic[8];          /* "SYNPROV1" */
                 uint32_t abi_major, abi_minor;
                 uint8_t build_id[32];
                 uint32_t struct_size; } SynHeader;

typedef struct { SynHeader h;                /* the core's identity, checked by the module too */
                 void (*log)(uint8_t level, const char* target, const char* msg, size_t len);
                 void (*metric)(uint32_t id, uint64_t value);
                 const char* (*config)(const char* key);        /* features.disable, kernels.* (§12) */
               } SynHostApi;

typedef struct { uint32_t class, domain; uint16_t code; uint8_t retry; } SynStatus;   /* class = ErrorClass, 0 = ok */

typedef struct { uint32_t struct_size;       /* size of this vtable as the module built it */
                 const char* id;             /* "cuda", "ggml-cuda", … */
                 uint32_t backend, domain;
                 /* then one function pointer per KernelProvider method, in declaration order:
                    SynStatus (*enumerate)(void* self, SynProviderDevice* out, uint32_t cap, uint32_t* n);
                    SynStatus (*open)(void* self, uint64_t device, const SynOpenOptions*, uint64_t* ctx);
                    …
                    SynStatus (*replay)(void* self, uint64_t ctx, uint64_t stream, uint64_t graph);
                    SynStatus (*last_error)(void* self, uint64_t ctx, char* buf, size_t cap, size_t* len); */
                 void* self; } SynProviderVtbl;

typedef struct { SynHeader h;
                 uint32_t n_providers;
                 const SynProviderVtbl* providers;
                 SynStatus (*shutdown)(void); } SynModule;

const SynModule* syntropy_provider_entry_v1(const SynHostApi* host);   /* N = 1 at M1 */
```

**Compatibility check at load**, in this order, in the runner or probe process. The first failure refuses the module with `(Kernels, 4)` and a message naming the failed check:

1. The library is the file `module.toml` names, opened by absolute path (RFC-0052 §4.3).
2. The symbol `syntropy_provider_entry_v<N>` for the core's major version resolves. If not, the message gives the module's `provider_abi` from `module.toml` against the core's N.
3. Magic matches, and `struct_size` of every header and vtable is at least the size the core knows.
4. `abi_minor` is at least the core's minimum minor.
5. `build_id` equals the core's (lockstep). Developer builds may pass `--dev-modules` to skip this check; every result is then marked uncertified (RFC-0060 §6.4).
6. Every provider `id` the module returns is listed in `module.toml`'s `providers`.
7. The entry call itself returns, and the module's own check of the host header passed.

**Rules at the boundary:**

- Only `#[repr(C)]` structs, raw pointers with lengths, and `u64` opaque handles cross it. Device contexts, buffers, streams, events, prepared ops, captured graphs and registrars are handles. Rust `String`, `Vec`, `Box`, trait objects and panics never cross.
- Every thunk runs its body under `catch_unwind`. A caught panic returns `Internal` and makes the provider report `DeviceHealth::Sticky`, so the runner exits and restarts cleanly (RFC-0001 §3.4).
- Each side frees what it allocated. Messages are fetched with `last_error` into a caller-provided buffer.
- Thread affinity follows §2.4. Functions that take a context handle must be called on the dispatcher thread that opened it, and debug builds assert this inside the module. A `HostRegistrar` is its own handle with its own small vtable, because it is `Send`.
- **Versioning.** A minor version may only append fields to structs and entries to vtables; `struct_size` lets each side see what the other has. Anything else (reordering, a changed signature, a changed meaning) bumps the major version and with it the symbol name. Release builds always match exactly, because of the build-ID check. The minor version exists so that developer builds can mix compatible builds, and so that errors say what is wrong.
- **Cost.** One indirect call per crossing. A captured step crosses a handful of times (replay, records, signals), which is noise against SC-2's budget (§Performance).

**ggml inside a module.** Each provider library links its own static copy of ggml-base with its one ggml backend, with all symbols hidden. The core's ggml (CPU, and Metal on macOS) and a module's ggml are separate instances that never share objects. Each sees only its own devices, which is all the ggml provider needs (§8.1).

*Note, 2026-10-26 (COMP-E9-T04).* The first implementation of this section (`kernels::abi`, `kernels-loader`) fixes five details the text above leaves open.

- **Encoding.** Handles, integers and plain records (`OpenOptions`, `DeviceMemoryInfo`, `AllocRequest`, buffers, slices, segments, registrations, `OpArgs` with its tensors, views and scalars) cross as `#[repr(C)]` structs. Records that only cold paths use (`ProviderBuild`, the device list, the op registry, `OpKey`, `ShapeSig`, `OpCaps`) cross as UTF-8 JSON in a pointer and a length. Every variable-size result goes into a buffer the caller supplies (`SynOutBuf`): the callee reports the size it needs, and the caller retries once with a buffer of that size. The module therefore never hands the core memory to free. The JSON records are append-only like the structs: a minor version may add fields, and each side ignores fields it does not know.
- **Entry call.** The module checks only the host header's magic, ABI major and struct size, and returns its table whatever the build IDs. The build-ID comparison belongs to the core's load check (check 5), so `--dev-modules` can relax it.
- **Thread affinity.** A module keeps its device contexts in a per-thread table, so a context handle used on another thread is refused with `Internal`. The check is on in release builds too.
- **After a panic.** The thunk that caught it returns `Internal` with code 16, and the provider is `Sticky`. Later calls other than `health`, `free`, `close_ctx` and a registrar's `close` (the thunk `registrar_close`) fail with `DeviceSticky` (8). The proxy methods whose signatures carry no error (`free`, `query`, `repack`, `repacked_len`, `elapsed`, `host_registrar`) record the first `Internal` error, and `health` reports `Sticky`.
- **Linux export list.** rustc passes every `cdylib` link an anonymous version script that lists the `#[no_mangle]` symbols and makes the rest local, and GNU ld refuses to combine an anonymous version tag with a named one. A cargo-built library therefore exports exactly the entry symbol, but unversioned: the node `SYNTROPY_PROVIDER_N` needs a link step cargo does not give. COMP-E9-T19 adds it with a link step that replaces rustc's script; the single-export property does not depend on it. A panic in a registrar's destructor (`close`) makes the provider `Sticky` like any other thunk.

*Note, 2026-10-30 (COMP-E9-T19): the version node.* On Linux every link goes through `ci/link/provider-cc`, the C compiler driver `.cargo/config.toml` sets for `x86_64-unknown-linux-gnu` and `aarch64-unknown-linux-gnu`. It is a POSIX shell script that uses only built-ins and runs `cc`, or `$SYNTROPY_LINK_CC` when set (a cross build's compiler driver). When rustc's version script exports `syntropy_provider_entry_v1` and nothing else, it passes `ci/link/provider.map` in its place. That script exports the same one symbol under `SYNTROPY_PROVIDER_1` and makes the rest local, so the entry is `syntropy_provider_entry_v1@@SYNTROPY_PROVIDER_1`, and the loader's `dlsym` of the plain name finds it as the default version. A script that exports more is passed through, and every other link reaches the driver unchanged. It was checked with GNU ld and with rust-lld (`-fuse-ld=lld`), on aarch64 Linux; x86-64 Linux, where rustc links with rust-lld by default, is checked by the hosted Linux test job (`ubuntu-24.04`), which runs the exports test. GNU ld also lists the node as an absolute symbol in the dynamic table, which is not an export. A provider crate needs nothing of its own. Three things leave the entry unversioned, and the exports test then fails: a `CARGO_TARGET_<triple>_LINKER` or `-C linker` override, which bypasses the wrapper; rustc moving the link arguments into an `@file`, which it does only when the command line is too long for the OS; and a build outside this workspace's cargo config. The artefact assembly (COMP-E9-T08) checks the `cuda13` library with `readelf -V` for that reason. The "Linux export list" item above is superseded on this point.

*Note, 2026-10-30 (COMP-E13-T15): error domains on the boundary.* `kernels`' errors are `core::Error` (RFC-0001 §5.12), whose code may belong to any domain, so the ABI now carries the domain wherever it carries a code. This is a wire-format decision taken under the maintainers' standing instruction of 2026-10-22, applied on 2026-10-30.

- **Domain numbers.** A `SynStatus`'s `domain`, and a `Sticky` health's, is the domain's boundary number from one table that only grows: `Kernels` 1, the number the ABI published first, then `Cluster` 2, `Net` 3, `Dist` 4, `Recovery` 5, `Security` 6, `Exec` 7 and `Tier` 8 (added 2026-10-31 with `core`'s `Domain::Tier`, RFC-0020 §15). A domain added to `core` takes the next number. 0 is no domain. `kernels::abi::DOMAIN_CODES` holds the table, and a test fails unless every `core::Domain` variant has a number.
- **Unknown numbers.** A status whose domain number this build does not know decodes as an `Internal` error naming the number and the code, as an unknown class does. A health record with one decodes as no health, which the core's proxy reports as an unknown health.
- **`SynHealth`** is `{ uint32_t kind, domain, detail; }`, 12 bytes. `domain` is 0 for `Ok` and `Throttled`. For `Sticky`, `domain` is the code's domain number and `detail` the code.
- **`SynProviderVtbl`** gains a last field, `uint32_t health_size`: the size of the `SynHealth` the provider's `health` writes. The core refuses a table whose `health_size` differs from its own, with `(Kernels, 4)`. A library built before this change has a table 8 bytes shorter, which the size check refuses first.
- **The ABI stays version 1.** No provider library or core of the earlier layout has shipped, so the entry symbol, `SYNTROPY_PROVIDER_1` and `ABI_MAJOR` 1 are unchanged. `SynHealth` changed size in place only for that reason: a core of the 8-byte layout would accept a library of this one, whose table is larger, and hand its `health` an 8-byte buffer. Once v1 has shipped, resizing an existing record needs `ABI_MAJOR` 2; `health_size` guards a mismatch in either direction.

#### 10.5 Driver census and loading

- **Census without modules.** `kernels_loader::census`, in the core, identifies GPUs, vendors, architectures and driver versions in a probe-mode process (RFC-0001 §3.1), without loading any provider library and without creating a GPU context. It loads the driver's own libraries at runtime where installed, never linking them: the CUDA driver and NVML for version and attribute queries; `vkEnumeratePhysicalDevices` through the system Vulkan loader; DXGI/DXCore on Windows. AMD GPUs on Linux are identified from amdgpu and KFD sysfs (the gfx target per node), without initialising the ROCm runtime. `syntropy setup` resolves modules from this census (RFC-0052 §4.2), and the daemon re-runs it at start to report a GPU that has no matching module.
- **Loading.** Runner and probe processes load each installed module's provider library by absolute path (RFC-0052 §4.3), run §10.4's checks, and register the returned providers. Their devices then merge into `DeviceInfo` (§2.2). For a CUDA module, the loader first checks that every NVIDIA file `module.toml` lists is present beside the provider library with the size the manifest records; `setup` verified each archive's SHA-256 when it fetched it (§10.3), so loading does not re-hash. A missing file is reported as `ModuleMissing` (2) with the `setup` command that fetches it, or the toolkit fallback is used when configured.
- **A missing module is not an error.** That backend offers no devices, the probe says why, and `syntropy doctor` names the command that installs it (RFC-0051, RFC-0052).

#### 10.6 Size reporting

Each build reports artefact sizes (RFC-0062 §4.5) against RFC-0052 §12's budgets. Growth above 10 % over the previous release without a changelog entry fails the packaging gate. The report breaks each module down by component (Syntropy kernels, ggml, hipBLAS/rocBLAS, other redistributables). NVIDIA's libraries are not in the CUDA modules (§10.3), so the report lists them separately as bytes `setup` fetches, with the pinned archive sizes from `module.toml`. The first M1 report is the measurement RFC-0052 open question 3 asks for; until it exists, the size of the `cuda13` download including the fetched NVIDIA archives is unknown, and no size claim is made.

### 11. Op coverage policy per model × backend

#### 11.1 Required ops per family

RFC-0043 §3 lists the blocks; each block lowers to a fixed set of `OpKind`s. A family's **required op set** is the union over its blocks, for decode, verify and prefill shape classes, for each dtype class its v1 checkpoints ship in (RFC-0042 §2). RFC-0043 publishes the set per family as data (`graph::family::<f>::required_ops()`).

#### 11.2 Coverage levels

For each (family, backend domain, dtype class), CI computes a level from `query` results on lab hardware:

| Level | Meaning | Solver treatment |
|---|---|---|
| **N** (native) | every required op has a native candidate with an `OpCert` | normal |
| **G** (ggml) | every required op has a candidate on the device, and at least one is ggml | normal; selection may still pick ggml where it is faster |
| **F** (fallback) | some required ops have no candidate on the device; they run on the node's CPU device, with a `t_switch` hand-off each way per occurrence | the solver costs the hand-offs; the plan lists the ops in `assumptions`; the dashboard shows "partial on <backend>" |
| **U** (unsupported) | some required op has no candidate on any device of the node | the node cannot host stages of this family |

The **CPU domain must be at N or G for every family in every release**, because it is the universal fallback.

#### 11.3 What happens at plan time

The probe reports per device the `OpKey`s with candidates (a compact bitmap over the op registry, per scheme). The solver's filter (RFC-0011) drops devices at level U for the family, and charges level-F devices for their CPU hand-offs. A level-F plan never changes output: the fallback op runs the same op on another provider, which passes the same parity gate.

#### 11.4 Release gates

| Platform tier (ADR-0020) | Gate per v1 family and shipped dtype class |
|---|---|
| Tier 1 (CUDA, Metal, CPU) | level N for the M1 roster (Llama-class dense, DeepSeek-V3-class MLA MoE, Qwen MoE and hybrid, gpt-oss-class) at M1; level ≥ G for every v1 family at M3; level N for every v1 family at M4 |
| Tier 2 (HIP, Vulkan) | at M3: level ≥ G for the M1 roster and Gemma-class, level ≥ F for the other families; at M4: level N for dense GQA, level ≥ G for every v1 family |
| Tier 3 (ggml-only backends, CUDA 12 legacy) | measured and published; level F allowed; never gates a release |
| Tier 2 preview (the M2 `vulkan` module with `ggml-vulkan` only, §7.5) | measured and published at M2; labelled preview; never gates; the Tier 2 row above applies from M4, with its coverage gate running from M3 |

A family whose required ops include a new op kind ships on a backend only at the level that backend reaches; the coverage matrix is part of the release notes and of `syntropy doctor --coverage`.

#### 11.5 Planned coverage

| Family (RFC-0043) | CUDA | Metal | CPU | HIP | Vulkan |
|---|---|---|---|---|---|
| Llama/Mistral/Qwen dense (GQA) | N at M0 | N at M1 | N at M0 | N at M3 | N at M3 |
| Gemma-class sliding/global | N at M1 | N at M1 | N at M1 | G at M3, N at M4 | G at M3, N at M4 |
| gpt-oss-class (MXFP4, sinks, sliding) | N at M1 | N at M1 | N at M1 | G at M3 | G at M3 |
| DeepSeek-V3-class (MLA, MoE, FP8 originals) | N at M1 | N at M1 | N at M1 | G at M3 | G at M3 |
| Qwen MoE (`qwen_moe`, softmax top-k, GQA) | N at M0 (all experts resident), M1 (tiered) | N at M1 | N at M0 | G at M3 | G at M3 |
| Qwen hybrid (Gated DeltaNet, gated/sparse attention) | N at M1 | N at M1 | N at M1 | G at M3 | G at M3 (native scans if ggml-vulkan lacks them) |
| Kimi-class (MLA + KDA, latent MoE) | N at M3 | N at M3 | N at M3 | F at M3 | F at M3 |
| DeepSeek-V4-class (compressed attention, indexer) | N at M3 | N at M3 | N at M3 | F at M3 | F at M3 |
| GLM-class (MLA + indexed sparse attention) | N at M3 | N at M3 | N at M3 | F at M3 | F at M3 |

At M2, before the native Vulkan provider, the Vulkan column is the ggml-only preview of §7.5: whatever level `ggml-vulkan` reaches per family, measured and published, from G downwards. The encoder-only `bert` family (RFC-0043 §5.4) is planned at N on CPU and CUDA at M2 and promised nowhere else.

Metal tracks CUDA family for family, because ADR-0004 gives Metal kernels the same priority and multi-Mac clusters are the main platform for the largest families. Cells marked F depend on which new attention and scan ops ggml gains on that backend before the milestone. Level F on a Tier 2 backend at M3 is allowed by §11.4; reaching G or N there is M4 work. CI measures every cell, and the table in each release's notes replaces this plan.

### 12. Configuration knobs

RFC-0054 owns the schema and layering; these are the keys this RFC defines.

| Key | Default | Meaning |
|---|---|---|
| `features.disable` | `[]` | kill switches (ADR-0014 rule 4). Kernel-level entries: `graph_capture`, `stream_mem_ops`, `fp4_tc`, `fp8_tc`, `int8_tc`, `vmm`, `batched_copy`, `amx`, `sme2`, `avx512`, `metal4`, `coop_matrix`, `ggml`, `provider:<id>`. Each forces the documented fallback, and `syntropy doctor` lists what is active |
| `kernels.select_budget_ms` | 3000 | microbenchmark budget per runner (§6.3) |
| `kernels.select_margin` | 0.03 | challenger must beat the incumbent by this fraction (§6.4) |
| `kernels.domain_margin` | 0.05 | margin between memory domains on AMD and Intel GPUs (§6.4) |
| `kernels.parity_spot_check` | `true` | runtime spot check in `Warming` (§6.2). Turning it off is logged and labels the instance "unchecked" in its metadata |
| `kernels.force` | `{}` | debug override, `{ "<OpKind>": "<provider>" }`; the plan records it in `assumptions` and the dashboard flags it |
| `kernels.deterministic` | `false` | open devices with deterministic variants only (ADR-0019 decision 8); set per instance by the `deterministic` request option in M3 |
| `kernels.dev_modules` | `false` | developer builds only: skip the build-ID check of §10.4; every result is labelled uncertified |
| `kernels.spin_kernel_max_us` | derived: max(50 000, 20 × the plan's largest predicted per-layer CPU in-place time), capped at the device's GPU segment budget where the OS has a watchdog (§4.5) | a device spin kernel that waits longer than this raises `(Kernels, 11)`; the step is then failed as `Timeout`. A fixed value would fire on plans whose CPU segments legitimately take milliseconds per layer; the cap makes Syntropy abandon the step before the OS resets the GPU. A plan whose predicted per-layer CPU time would need a wait above the cap is not chosen for that device (RFC-0011) |
| `kernels.module_dir` | the running version's `modules/` directory (RFC-0052 §4.3, RFC-0054 §5.1) | where runner and probe processes load provider libraries from. `<store>/modules` and other paths are developer overrides; a non-default value labels results uncertified unless the modules there pass every §10.4 check |

### 13. Failure handling and error codes

Codes are `(Domain::Kernels, n)` and never reused (RFC-0001 §10.4).

| Code | Name | Class | Raised when | Action |
|---|---|---|---|---|
| 1 | `OpUnsupported` | `Unsupported` | no candidate for (op, scheme, shape) on a device | selection reports it; the solver re-plans the compute site (§11.3) |
| 2 | `ModuleMissing` | `Unsupported` | a backend's module (provider library) is not installed, or a vendor library its `module.toml` lists has not been fetched (§10.5) | the backend offers no devices; `doctor` explains and names the `setup` command |
| 3 | `ModuleIntegrity` | `Integrity` | module digest differs from the signed manifest | refuse the module; the store refetches it |
| 4 | `ModuleAbi` | `Unsupported` | ABI version or source key mismatch with the binary | refuse; the installer fetches the matching module |
| 5 | `DriverTooOld` | `Unsupported` | CUDA driver below 580, ROCm below 7.2, Vulkan below 1.3, macOS below 15 | the backend is skipped; the next domain is tried (§6.4) |
| 6 | `ParityFailed` | `Correctness` | a candidate fails the runtime spot check | strike the candidate for this (device, build); try the next; if none is left, the op is unsupported on the device |
| 7 | `CaptureFailed` | `Unsupported` | capture or instantiation fails | fall back to per-op launches for that segment (ADR-0014); `Ready { degraded: GraphCaptureOff }` |
| 8 | `DeviceSticky` | `Device` | a sticky device error | the runner exits; the supervisor restarts it (RFC-0001 §3.4) |
| 9 | `AllocFailed` | `Capacity` | `alloc`, `map_segment` or `import` fails | the tier manager corrects usable capacity and the plan is re-solved (RFC-0001 §10.1) |
| 10 | `RegisterFailed` | `Capacity` | `register_host` fails (pin limit, WDDM limit) | the tier manager shrinks the pinned arena for that device and reports `Drift` |
| 11 | `DoorbellTimeout` | `Timeout` | a spin kernel or a device-side wait exceeds `kernels.spin_kernel_max_us`, or the host waits on a device flag beyond the stall threshold | the step fails; the supervisor's stall rules apply |
| 12 | `GgmlImportFailed` | `Unsupported` | the import probe of §8.3 fails | fall back to ggml-allocated buffers recorded by the tier manager |
| 13 | `AmxDenied` | `Unsupported` | the OS refuses AMX tile permission | at `enumerate`, report the `X86V4` class; at `open` under an `X86V4Amx` class, fail and re-probe, never falling back to `X86V4` kernels (§7.3; revised 2026-10-22, ADR-0405 decision 3); record in the profile |
| 14 | `WorkspaceExceeded` | `Internal` | a prepared op asks for more workspace than `query` declared | a provider bug: the runner fails `Internal` with a diagnostic bundle |
| 15 | `NoCertifiedCandidate` | `Unsupported` | candidates exist for (op, shape) on a device but none holds a certificate (§6.2) | as code 1; the dashboard says the kernel exists but is uncertified on this device class |
| 16 | `ProviderPanicked` | `Internal` | a provider-library thunk caught a panic (§10.4) | the provider reports `Sticky`; the runner exits and restarts |
| 17 | `ManifestInvalid` | `Integrity` | the `ConformanceManifest` is missing, or not signed by a key the build's profile accepts | ggml ops plus the CPU provider's native ops for what ggml cannot run, on single-device shapes, with the F32 wire, labelled uncertified (§6.2; ADR-0602 decision 6) |

*Note 2026-10-28 (COMP-E13-T15): the domain's range.* `Domain::Kernels` holds 0x0001–0x00FF, so each code above keeps its published number: `(Domain::Kernels, 1)` is `OpUnsupported`, and so on to 17. `core::error::codes::kernels` lists them. The faults `exec` detects are not kernel faults, so they take a domain of their own, `Domain::Exec` (RFC-0012 §13, note of the same date). How these codes reach an API client is RFC-0050 §12.1's note of the same date.

Fallback chains are declared paths (RFC-0001 §10.2): native kernel → ggml op on the same device → the op on the node's CPU device (level F). None of them changes the numerics contract, because every candidate passed the same parity gate.

## Performance model and budgets

All figures here are **targets** or **estimates**, derived from `plan/research/hardware-and-performance-model.md` (cited as "cost model §n") and the vision document's SC rows. Nothing below is a Syntropy measurement.

**Per-op time.** For op o on device d, the cost model uses `t_o = B_o / W_eff(d, class(o)) + F_o / F_eff(d, class(o), dtype)` for overlapped kernels, with `max` replacing `+` where the kernel overlaps loads with arithmetic (cost model §3.1). W_eff and F_eff are measured by `kernels::bench` with the kernel selection will choose (§6.6), so the solver's inputs and the executed kernels are the same object.

**Per-step fixed cost on one device.** The kernel layer contributes four terms to `t_fixed` (cost model §3.4):

```text
t_fixed,kern(d) = t_submit(d) · n_submissions + n_nodes · t_node(d)
                + Σ_MoE layers on d  t_doorbell,exposed(d) + t_desc(d)
  n_submissions = compute-stream submissions: 1 with capture on CUDA and HIP (§4.4);
                  1 command buffer plus its per-step event commands on Metal;
                  1 vkQueueSubmit2 plus one vkSignalSemaphore per wait_host on Vulkan (§4.2)
  n_nodes · t_node = device-side gaps between captured nodes; t_node is the probe's measured
                  empty captured node (§6.3, LaunchCaps::captured_node in RFC-0012)
  t_doorbell,exposed = the part of the doorbell round trip not hidden behind GPU work on hits
  t_desc = the step-descriptor copy (a few KiB over PCIe, or none on unified memory)
```

Copy-stream submissions (one `copy_batch` per MoE layer with copied misses) belong to `t_switch`, not to this term.

**Node count is a budget.** On the SC-2 reference model (a 7B dense model, 32 layers) the §7.1 fusions leave about 8 nodes per layer, so ≈ 260 nodes per step [estimate]. At 1–2 µs per node [estimate, unmeasured], that is 0.26–0.52 ms, the same size as the whole kernel budget below. Every new op in a step program therefore has to justify its node, fusion is preferred over a separate node, and the M0 gate checks `n_nodes · t_node` from the probe against the target before the target is relied on. If the measured term exceeds half the budget, the target is restated, not assumed.

**SC-2 budget split.** SC-2 sets `t_fixed ≤ 0.2 · B_ref / W_probe`, ≈ 0.95 ms on an RTX 4090 and ≈ 1.2 ms on M3 Ultra. For a dense model (no doorbells) this RFC targets `t_fixed,kern ≤ 0.5 ×` the SC-2 budget, leaving the other half for the executor, sampler hand-off and the ring (RFC-0012). That gives ≈ 0.47 ms on the 4090 and ≈ 0.6 ms on M3 Ultra for one graph replay plus the descriptor copy [target]. For MoE layers with a CPU site, the doorbell cost is not part of SC-2. It is the `t_switch` term in SC-3's model, `n_switch · t_switch`, and its measured value goes to the solver.

**Why capture is mandatory on Tier 1.** Without capture, a 61-layer MoE stage launches on the order of 10–15 kernels per layer, ≈ 600–900 launches per token [estimate: route, gather, hit, copy-wait, copy-hit, combine, norms, attention parts, residuals]. At even a few µs of host submission each, that is milliseconds per token, above the whole SC-2 budget. One replay per step removes this term, which is why graph capture is a Tier 1 requirement with only a degraded-mode fallback (ADR-0014).

**The provider-library boundary.** A captured step crosses the C ABI of §10.4 a handful of times (replay, event records, doorbell signals), each one indirect call plus a `catch_unwind` frame, which is noise against SC-2's sub-millisecond budget [estimate]. Ops on providers without capture cross once per launch, and they already pay a launch each.

**Execute-in-place versus streaming: the kernel inputs to n\*.** The crossover `n* ≈ F_cpu · b / (2 · W_pcie)` (cost model §3.2) uses F_cpu for the actual CPU kernel and quant. Row 5 of the §9 table is how the CPU provider's compute rate enters. Integer-coded i-quant CPU kernels are arithmetic-bound and leave DRAM bandwidth unused, which is why the plan's `MissSplit` may copy part of the misses even at batch 1 (`tiered-memory-inference.md` §5.4, §14). The CPU provider therefore reports its per-scheme rate, not one number per device.

**Worked check: example (a), Llama-3.3-70B Q4_K_M on an RTX 4090 + DDR5** (cost model §4a). With the §9 targets: VRAM part 19.3 GB at η = 0.80 of 1008 GB/s → 23.9 ms; CPU part 23.2 GB at 0.75 × 90 GB/s (probed DRAM, [S22]) → 343.7 ms; KV at 8 K, 2.68 GB at η = 0.70 → 3.8 ms. Sum ≈ 371 ms + t_fixed → ≈ 2.7 tok/s [estimate]. SC-4 requires ≥ 2.5 tok/s for this case, so the kernel targets are consistent with the end-to-end gate.

**Worked check: example (b2), DeepSeek-V3 Q4_K_M at h = 1** (all routed experts in RAM). Hot set 9.7 GB on the GPU at η = 0.80 → 12 ms; routed 12.3 GB on the CPU at 0.75 × 90 GB/s → 182 ms. ≈ 5.1 tok/s [estimate], against the cost model's 5.3 at its assumed 70 GB/s CPU rate.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Per-step values as launch arguments, graphs updated each step (`cuGraphExecKernelNodeSetParams` and equivalents) | One host call per updated node per step: hundreds per token on MoE stages, and no equivalent on Metal ICBs or Vulkan. The step descriptor gives one copy per step on every backend |
| Doorbells carrying the step number, with captured memory-operation nodes patched each step (`cuGraphExecBatchMemOpNodeSetParams`) | A MoE layer with a CPU site has about three doorbell nodes (route signal, copy wait, one CPU wait per NUMA domain), so a 61-layer stage on one domain patches ≈ 180 nodes per step [estimate]. Even at 0.1 µs per call [estimate, unmeasured] that is ≈ 18 µs of the ≤ 20 µs host-work target (RFC-0012), before anything else the dispatcher does. The handshake of §4.2 needs no per-step host call |
| Constant values per parity instance (parity 0 waits for 1, parity 1 for 2, never reset) | Correct only while every flag is written at every step and parities alternate without a gap. A missed or abandoned step leaves a value that satisfies the same parity's wait two steps later, silently (an ABA case). Under the consumer reset the same fault leaves the flag at 0 and the step stalls, which the stall watchdog catches |
| Spin and store kernels that read `StepDesc.step` everywhere, keeping monotonic step numbers, as the protocol | Correct, but every doorbell then costs a kernel launch slot and a resident warp, and Vulkan and the mem-op path would differ in protocol. Rejected as the protocol. Spin and store kernels survive only as the fallback mechanism of §4.2, and there they run the constant-valued handshake, not step numbers |
| Host-side MoE split decided by the CPU and pushed to the GPU (dispatch block written by the host, GPU waits on it) | Adds a host round trip before the GPU can start on hits. The shared pure `assign` function lets hits start immediately and keeps every site in agreement without a message |
| GPU-driven copies of copied misses (SM gather kernels reading pinned memory) instead of the copy engine | Steals SMs from hit compute during the same window; the copy engine is otherwise idle. Kept as a variant to benchmark on GPUs without a free copy engine, not the default |
| Vendor GEMM libraries (cuBLAS/cuBLASLt, hipBLASLt, MPS) for native prefill | Their own allocation and stream behaviour, which breaks capture with Syntropy's buffers, and no quantised-weight support for GGUF types. CUTLASS-class templates compiled into our module give the tensor-core paths. The modules carry cuBLAS and hipBLAS/rocBLAS anyway for ggml (§8.1), so payload is not the reason |
| A patch that pre-creates ggml's BLAS handles and sets their workspace (`cublasSetWorkspace`) at `Warming`, making BLAS-routed ggml ops capturable | A fifth local patch against code ggml changes often. ggml is the fallback provider, so splitting its prefill GEMMs out of the graph costs little; revisit if ggml prefill is selected on Tier 1 devices in practice |
| Rust trait objects passed directly across a `cdylib` boundary | Relies on unstable layouts and on matching allocators and panic runtimes. The generated C ABI of §10.4 costs one indirect call per crossing |
| Rust provider code in the core, kernel objects alone in modules | Puts CUDA, HIP and Vulkan host code in a core that ADR-0507 keeps vendor-free, and splits one provider across two artefacts |
| Runtime compilation (NVRTC, Metal source compilation, runtime Slang) | Needs toolchains or large compiler libraries on user machines, slows first start, and complicates notarisation and iOS. PTX JIT remains only as the driver's forward-compatibility path |
| ggml's own dynamic backend loader on desktop | A second module system beside ADR-0507's, with a shared `ggml-base` that the core and every module would have to agree on; private static copies with hidden symbols avoid both (§8.1, ADR-0403) |
| Using ggml's scheduler to drive mixed-device graphs | Rejected by ADR-0002: its split is per tensor, not per expert subset per token |
| SVE2 kernels in v1 | Unstable intrinsics, and no target in the support matrix with vectors wider than NEON's (§7.3, ADR-0402) |
| A single fixed provider per device (no per-op selection) | Fails where ggml is faster for a specific op on a specific device, and gives no fallback when a native kernel fails parity on new hardware |

## Testing and validation

| Claim | How it is verified | Where |
|---|---|---|
| Every op's semantics | `kernels::reference` against hand-computed cases and against a second, independent implementation for ops ggml lacks; property tests on shapes, `rows_active` edges, masks and NaNs | unit and property tests, every PR (RFC-0060) |
| Native kernels are correct | op-level parity against the ggml op (or the ggml CPU op) on random inputs at every bucket shape and every scheme, within RFC-0042 per-op tolerances, run by RFC-0060 §4.2 against the F64 reference; produces the `OpCert`s of the `ConformanceManifest` | Tier 1 hardware jobs, path-filtered on `kernels-*` (ADR-0020); Tier 2 weekly |
| `assign` agrees across implementations | the Rust, CUDA, HIP, MSL and Slang versions run on the same 10⁶ random (route list, tables, `B_copy`) inputs and must return identical results | property test on each backend's CI hardware |
| Combine order makes the MoE result independent of the split | the same layer computed with `B_copy` at 0, 25 %, 50 % and 100 % of the missed bytes and with different cache contents gives bit-identical output when every site uses the same kernel (CPU-only run), and passes the parity tolerance when sites differ | integration test |
| No allocation, lock or synchronising driver call on the hot path | a counting global allocator and a driver-call shim (the provider's dynamic loader wraps every driver entry point in test builds) assert zero calls of the forbidden set during 1,000 steady-state steps | integration test on each Tier 1 backend |
| Capture covers the step | compute-stream submissions per step are counted: 1 on CUDA/HIP, 1 command buffer on Metal, 1 `vkQueueSubmit2` on Vulkan; copy-stream submissions are counted separately and must equal the number of MoE layers with copied misses plus prefill `CopyIn`s | integration test |
| Captured doorbells never see a stale value | one captured graph per (bucket, parity) replayed for 1,000 steps on a MoE fixture, with the CPU pool's in-place time drawn at random per layer from 0 to 10 ms and the dispatcher's copy issue delayed at random; every step's combined output is compared with an uncaptured run of the same steps, and a deliberate omission of one reset must deadlock the test (proving the test sees stale flags) | integration test on CUDA, HIP (mem-op and spin-kernel paths), Vulkan and Metal |
| Hot threads never sleep through a doorbell | a 10 ms CPU segment per MoE layer completes with step time within 5 % of the segment sum; inside a step, threads park only on an armed `arm_wake`, and no step waits for a timer or the stall watchdog; a program for a stage with no armed wakes contains no host nodes | hardware CI (RFC-0012) |
| Selection is stable and sound | replaying recorded benchmark results through the decision rule reproduces the cached choice; a deliberately broken kernel (fault-injected wrong output) is struck by the runtime spot check and never selected | unit test; fault-injection test |
| Roofline targets (§9) | RFC-0061 microbenchmarks per (device class, scheme): W_eff of rows 1, 3, 4 gated on Tier 1 from M1; the rest tracked | reference lab (RFC-0061) |
| Doorbell latency and `t_submit` | `kernels::bench` measures them in the probe; recorded per device class; compared against the cost model's 5–20 µs `t_switch` estimate | probe CI job |
| Provider libraries load only when they match | a module with another ABI major, another minor below the core's minimum, another build ID, or an unlisted provider ID is refused with the right check named; `--dev-modules` labels results uncertified | packaging CI (RFC-0062) |
| The ABI boundary is sealed | the exported dynamic symbol table of every provider library holds exactly one symbol (checked with the platform's symbol tools); a provider fault-injected to panic in each thunk leaves the runner with a `Sticky` provider and no undefined behaviour under Miri-style sanitizer runs of the core side | packaging CI and nightly |
| Certificates gate selection | a manifest without the `OpCert` for a kernel makes selection skip it; a changed kernel object invalidates exactly its own `OpCert`s; the cross-loaded `sm_80`/`sm_86` objects certify on the `sm_89` rig | conformance CI (RFC-0060 §4.2) |
| Split attention is exact | `AttnPartial` + `AttnMerge` against single-site attention at split ratios 0, 10 %, 50 %, 90 % and 100 %, every `AttnKind`, with and without a sink logit, with one and two NUMA-domain partials, within op-level tolerance; the empty-side rule at 0 % and 100 %; for sparse kinds, a CPU site attending exactly the GPU-written selection; end-to-end at several ratios through the harness (ADR-0207 decision 5) | op conformance; parity nightly |
| Joint layouts | a MoE group read by a GPU and a CPU site is stored in one layout both accept, or in per-site copies only when the plan says so; a fault-injected mismatched layout is refused at `Warming` (§6.2 condition 3) | integration test |
| Parity taps are correct and free when idle | `TapGather` against `kernels::reference` on random logits at real vocabulary sizes; a tail step with `rows_tap = 0` gives bit-identical output and timing within noise of a program without tap ops | op conformance; integration test |
| Timestamps do not change output | a profiled step variant and the plain step give bit-identical outputs on every backend; Metal's timing perturbation is measured (RFC-0010 open question 8) | integration test; probe CI |
| ggml interop | the import probe (§8.3) and the patched stream path are tested on every pin bump; with patches reverted, the fence fallback must still pass parity | pin-bump job |
| Stable Rust intrinsic status (§7.3) | a release checklist item re-reads the Rust release notes and stable docs; when AMX or SME intrinsics stabilise, ADR-0402 is revisited | release process |
| Fallback parity (ADR-0014 rule 4) | every `features.disable` entry is exercised in the parity suite on hardware that has the feature | Tier 1 nightly |
| The doorbell probe is functional | on a WDDM GeForce device and on Linux, the `open` probe's captured wait and reset replay with a host write and select native memory operations; with the operations made to fail (a fault-injected probe), the spin and store kernels are selected; no code path reads `CAN_USE_STREAM_MEM_OPS_V1` | probe CI; R-WIN |
| Work fits the OS watchdogs | on R-WIN with a display attached and on R-MACF, a 4096-token prefill of the largest fixture each rig runs never trips the OS watchdog; every launch (CUDA) and command buffer (Metal) is estimated, and measured, within the device's segment budget; a host wait held past the budget fails the step with code 11 before the OS acts; whether a blocked native wait counts toward the watchdog is recorded per OS | lab only, never on users' machines |
| Routers are deterministic | each backend's `MoeRoute` run twice on identical inputs at every bucket gives identical expert IDs and weights; no router kernel uses atomics or split-K (kernel review checklist) | op conformance |
| NVIDIA files come from NVIDIA | the `cuda13` module's file list contains no NVIDIA file; a runner with the fetched archives present loads, and one with a listed file missing reports code 2 naming the `setup` command; the provider library's run path resolves NVIDIA's libraries from the module directory | packaging CI |
| The Vulkan preview owns no memory of its own | on the lab's Vulkan rig, the allocation hook shows every buffer allocated through ggml's buffer type and recorded by the tier manager, and no `map_segment` call; coverage per family is published with the preview label | M2 Vulkan job |
| Encoder ops | `AttnPrefill(Bidirectional)` and `Pool` against `kernels::reference` on CPU and CUDA, and against ggml's unmasked attention | op conformance |

## Rollout / milestones

Milestones follow RFC-0001 and the vision document (M0–M4).

| Milestone | Delivered by this RFC |
|---|---|
| **M0** | the spike sprint's standalone harnesses in M0's first weeks (S-02, S-04, S-07, S-08, and the WDDM doorbell probe on day 1), whose results set the defaults the product tasks build; `syntropy-kernels` (trait, registry, reference, selection, bench); CPU provider at `X86V3`, `X86V4`, `Arm82Dot`, `Arm86I8mm` for `llama` and `qwen_moe`; CUDA provider hot ops for both families (GEMV for k-quants and Q8_0, attention decode, prefill GEMM, norms, sampling, and `MoeRoute`/`MoeGrouped`/`MoeCombine` with every expert resident in VRAM, the case RFC-0001 §16's M0 parity run and RFC-0023's M0 need) with graph capture and the step descriptor; ggml provider (CPU in the core; CUDA in the `cuda13` provider library); `OpCert`-gated selection against the M0 manifest (RFC-0060 §6.4); translation-unit structure and `cargo xtask kcache` (ADR-0623); the provider-library ABI v1 (§10.4), `kernels-loader` and the driver census |
| **M1** | CUDA complete for the M1 roster (MoE with copied and in-place misses, doorbells under §4.2's handshake with stream memory ops and the spin-kernel fallback, VMM expert cache, FP8 and FP4 paths, MLA, Gated DeltaNet, sliding/sinks); `AttnPartial` (`Gqa`, `SlidingGqa`, `Mla`) and `AttnMerge` with sinks on CUDA and CPU for RFC-0022's M1 split attention; the `HiddenTap` parity variant; Metal provider (both libraries, ICB capture, shared-event doorbells, residency sets); CPU MoE miss kernels and AMX/SME2 compute kernels; ggml Metal in the macOS core; `Timestamp` ops and profiled step variants on CUDA and Metal; the `cuda13` module carrying Syntropy code only, with NVIDIA's libraries fetched at `setup` (§10.3); the watchdog policy of §4.5 on WDDM display GPUs and Metal; patches P1–P4 submitted upstream; coverage matrix in CI; §9 rows 1, 3, 4 gated |
| **M2** | intra-node collectives (`IntraNodeTp`) on CUDA; selection cache shared through packs between same-class nodes; `kernels::bench` measures `α_eff` sync costs for RFC-0032; the `vulkan` module as a ggml-only Tier 2 preview with ggml-owned allocations (§7.5); `AttnKind::Bidirectional` and `Pool` on CPU and CUDA for the encoder-only embedding family |
| **M3** | HIP and Vulkan native providers in the `rocm` and `vulkan` modules; `AttnPartial` for the sparse and compressed kinds, and on HIP and Vulkan (on Metal only if RFC-0022 adds an Apple spill path); every v1 family at the §11.4 levels; deterministic variants for the `deterministic` option; the `cuda12-legacy` module |
| **M4** | Tier 2 backends at their CI bar; Tier 3 module builds (Android, iOS static, SYCL, WebGPU through ggml); the remaining §9 rows become gates |

## Open questions

1. **Metal doorbell cost on the Metal 4 path.** Queue-level waits between command buffers may cost more per wait than encoder-level waits on the Metal 3 path. Default: use whichever is measured as cheaper per MoE layer, per OS version. Settled by the S-07 spike, a standalone harness issuing 61 waits per step through both paths in M0 week 1 on an M4- or M5-class Mac (COMP-E6-T12), then re-measured on R-ULTRA-A in M1 (COMP-E6-T04). OQ-KB-12 is merged into that spike. If 61 waits cost more than SC-2's budget for the stage, Metal MoE stages default to whole-layer GPU placement.
2. **Do memory-operation nodes work natively on WDDM?** (rewritten 2026-10-20; the old wording asked about an attribute that no longer exists). Default: the functional probe of §4.2 decides per device. A run on a WDDM RTX 3090 captured a wait and a write into a two-node graph that instantiated and replayed 2,000 times [reported, not a Syntropy measurement], which suggests they do. Settled by the probe on day 1 of M0 on a Windows GeForce machine with the native `nvcuda.dll` (COMP-E4-T12), and confirmed on R-WIN in M1.
3. **Spin-kernel cost.** A one-warp spin occupies an SM slot for the CPU's in-place time. Default: one warp, back-off with `__nanosleep`. Settled by the S-02 harness in M0 weeks 2–3, measuring hit-kernel throughput with and without a concurrent spin kernel on R-ADA and R-BW (COMP-E2-T16).
4. ~~Derived certificates~~ **Resolved (2026-10-19):** allowed under the three conditions in §6.2, labelled `exact (derived certification)` (RFC-0060 §6.4, ADR-0602 decision 7). Coverage of the other targets: `sm_75` runs on the lab's Turing rig (R-TURING), and `sm_90` and `sm_100` are covered by a cloud canary run on every release candidate from M1; v1 writes no `sm_90a`- or `sm_100a`-only kernels (ADR-0014).
5. **ggml upstream acceptance of P1–P4.** Default: carry them locally within the patch budget. Settled at the first two pin bumps after M1. Submission follows ggml's contribution rules (§8.4).
6. **Activation quantisation inside integer-dot kernels in exact mode.** This RFC follows RFC-0042 §4.3, which allows 8-bit activation blocks for integer-coded weights because the cross-validation baseline computes that way. If the M0 harness shows this path outside the dtype-class ceiling for some family, those kernels need an F32-activation variant. The S-03 pre-calibration gives an early read in M0 week 2 (COMP-E14-T22); the M0 parity calibration decides.
7. **Graph memory.** Each captured instance costs device memory; RFC-0012 §2's 14–26 instances per stage is a count, not a size (§4.4). Settled by the S-15 spike in M1, on 8 GB cards including the Turing card (COMP-E4-T18), with instance sizes measured at `Warming` on the M1 roster.
8. **AMD domain choice.** Whether HIP or Vulkan wins on RDNA3/RDNA4 for decode once both have native kernels. Default: measure per device (§6.4). No decision needed, but the result decides where HIP kernel work goes in M4.
9. **`Unified` allocation on GB10-class devices** (§5.1). Default: `cuMemHostRegister` over tier-manager memory until the probe's comparison of the three mechanisms runs on the Tier 2 GB10 rig.
10. **Metal timestamp perturbation** (§3.7, RFC-0010 open question 8). Settled by comparing profiled and plain step times in the S-07 harness in M0 (COMP-E6-T12), then on the Apple rigs in M1; above 2 %, Metal profiles at a lower rate.

## Proposed amendments to RFC-0001

1. **§5.7, `KernelProvider`:** record the methods this RFC adds with default bodies (`build`, `domain`, `memory_info`, `map_segment`, `unmap_segment`, `host_registrar`, `set_resident`, `wait_host`, `arm_wake`, `poll`, `elapsed`, `synchronize`, `copy_batch`). The important one is `wait_host`: RFC-0001 §8.2 requires the GPU to wait on a host-written flag device-side, but the §5.7 trait has no method for it. `poll` and `synchronize` have no default body, because every backend implements them; adding them is a breaking change to a trait with no implementations yet, so it costs nothing now.
2. **§5.7, selection timing:** "During `Warming` the executor microbenchmarks them" should read "During `Fetching` (layout pass, on synthetic weights) and `Warming` (final pass)". The pack layout must be chosen before `Packing`, which comes before `Warming` (§6.1).
3. **§4.1 and §4.2 rule 7, how GPU providers ship.** Rule 7 says backends are cargo features of one binary, and the `runner` row lists every `kernels-*` crate as a dependency. Under ADR-0507 and §10 here, the CUDA, HIP, Vulkan and SYCL providers are built as separate provider libraries in backend modules, loaded by runner and probe processes through the C ABI of §10.4, while CPU and Metal (and every provider on mobile) stay static. Proposed: rule 7 reads "CPU and Metal providers are cargo features of the core; other GPU providers are provider libraries in backend modules (ADR-0507, RFC-0040 §10.4), loaded at runtime, so one core binary runs on hosts with and without them", and the `runner` row lists `kernels-cpu`, `kernels-metal` and `kernels-loader` instead of every `kernels-*` crate (see amendment 6).
4. **§4.1, `kernels-ggml` row:** ggml is linked statically, one private copy per artefact with its backends and hidden symbols (§8.1). "The only crates with … ggml" stays true.
5. **§5.1, `DType`:** RFC-0042 adds `F8E8M0` (MX block scales), `F4E2M1` (FP4 elements, used in `TensorArg` dtypes for unpacked tiles) and `F6E2M3`/`F6E3M2` (safetensors F6 types, read-only). It is an addition, listed here so readers of §5.1 know.
6. **§4.2 rules 6 and 8, the loader crate.** Rule 6 should read "vendor GPU runtimes only in their `kernels-*` crate; `kernels-loader` may load vendor driver libraries at runtime (CUDA driver, NVML, the Vulkan loader, DXGI/DXCore) for read-only device enumeration, never linking them at build time and never creating a GPU context". Rule 8's list should name `kernels-loader` explicitly among the crates allowed `unsafe`: "`unsafe` is confined to `kernels-*` (including `kernels-loader`: the provider-library loader and the census), `io`, `tier` (arenas), `net` … and `ipc`". Amendment 3's `runner` row then lists `kernels-cpu`, `kernels-metal` and `kernels-loader`. Both rules are CI-enforced, so the crate cannot land until they change.
7. **§5.7, selection paragraph.** Amendment 2 changes only timing; three more sentences of §5.7's closing paragraph disagree with §6 here. The whole paragraph should read: "For each op in a stage's step program, the executor considers every provider in the device's memory domain that holds a certificate for the op in the release's `ConformanceManifest`, chooses the layout per tensor group across its readers during `Fetching` (on synthetic weights), and in `Warming` checks each candidate against `kernels::reference` within RFC-0042's `tol_ref` and benchmarks it. It keeps the native candidate, or else the cached choice, unless another passing candidate is more than 3 % faster. The choice is cached per node and in the pack's selection record (RFC-0040 §6)."
