# ADR-0014: Hardware feature policy: use the current capability, always keep a fallback
- Status: Accepted (revised 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D3, D7), ADR-0001, ADR-0002, ADR-0005, ADR-0006, ADR-0007, ADR-0011, ADR-0013, ADR-0019, ADR-0020; RFC-0010, RFC-0040, RFC-0042, RFC-0052, RFC-0061, RFC-0062

## Context
D7 asks for the largest model at the highest tokens/s the hardware can physically deliver, with a fallback for every feature so Syntropy still runs on any supported device. In practice the features that matter fall into four groups.

1. **Bytes per weight.** Decode is a bytes-moved problem: `t ≈ Σ bytes/W_eff + Σ hops + t_fixed` (`plan/research/hardware-and-performance-model.md` §0). Native FP4 and FP8 formats halve or quarter the bytes against BF16. 2026 frontier MoE models ship natively in MXFP4 or FP4 (`backends-and-portability.md` §0 item 6). Tensor cores that consume those formats directly avoid a dequantisation pass.
2. **Host overhead.** t_fixed ≈ 5 ms per token in one fitted engine (§3.4, estimate). A surveyed engine captures a whole pass as one CUDA graph per verify-window size, so the host makes no synchronising call per token (`tiered-memory-inference.md` §5.1).
3. **Data paths.**
   - Unified-memory machines have no host↔device copy; their tier graph is {unified DRAM, NVMe} (`hardware-and-performance-model.md` §2.3).
   - RDMA over TB5 cuts the small-message cost from ≈300 µs (TCP over TB5) to 10–25 µs (§2.5). That is what makes cross-node TP possible (ADR-0006).
   - The direct-I/O paths are in ADR-0013.
4. **Kernel coverage lag.** New attention types land first on CUDA. Vulkan, Metal, SYCL and WebGPU follow later or never (`backends-and-portability.md` §7.2 risk 1). Without a fallback per op, "runs everywhere" fails.

Vendor facts this policy rests on, all checked against vendor documentation:
- **NVIDIA.** CUDA 13 removed offline compilation and library support for Maxwell, Pascal and Volta. Toolkits through 12.x still target them. Driver branch 580 is the last for those GPUs (`backends-and-portability.md` §5.3).
- **Apple.**
  - Metal 4 runs on Apple7-class GPUs and later (M1, A14 and newer).
  - Its tensor operations use the M5 GPU Neural Accelerators where present, and optimised shaders elsewhere.
  - `MTLResidencySet` makes residency explicit (`backends-and-portability.md` §6.1).
- **AMD.**
  - RDNA4 (gfx1200/gfx1201) adds FP8/BF8 WMMA. CDNA3 has FP8 MFMA. CDNA4 adds FP4/FP6 and MXFP block-scaled MFMA.
  - ROCm's support list is per architecture. Everything outside it needs Vulkan.

## Options considered
1. **Lowest common denominator** (FP16 or BF16 compute, plain streams, `pread`, TCP). One code path, but it leaves 2–4× in bytes per token and the RDMA latency advantage unused. Rejected by D7.
2. **Current features only.** Drop pre-Blackwell FP8/FP4 fallbacks, pre-macOS-26 Metal and non-RDMA TP. Fewer paths, but it strands most installed hardware. Rejected by D7's fallback clause.
3. **Compile-time feature selection** (one binary per feature set). It multiplies artefacts and breaks "one binary". It also cannot react to a disabled sysctl, a seccomp profile or an unenabled RDMA port. Rejected. Builds still carry per-ISA and per-architecture code objects (ADR-0001), but selection is at runtime.
4. **Chosen: prefer each current capability, detect it at runtime, and make a fallback with equal numerics (within ADR-0019 tolerance) mandatory, with selection per op per device by microbenchmark (ADR-0002).**

## Decision
### Rules
1. **Every feature is detected at runtime**, recorded as a capability bit in the resource graph, and shown by `syntropy probe`. No feature is assumed from a model name or OS version alone.
2. **Every feature has a fallback** that produces the same output within the ADR-0019 tolerance for its dtype. A path that changes output beyond that tolerance is an approximate-mode knob (ADR-0019), not a fallback.
3. **Selection**:
   - Among the implementations available on a device, the provider and kernel are picked per op by microbenchmark at the model's real shapes.
   - Data paths (I/O engine, transport) are picked per volume or per edge by measurement (ADR-0011, ADR-0013).
4. **Admission bar for a new feature**:
   - detection code
   - a named fallback
   - a parity test of the feature against its fallback on CI hardware (RFC-0060)
   - a capability flag
   - a config kill switch, `features.disable = [...]`, for triage
5. **Floors move only by a superseding ADR.** Moving a floor also moves a platform's tier in ADR-0020.
6. **Machine-wide OS settings.** Kernels are designed to finish within each OS's default GPU watchdog: the executor splits work so that no submitted GPU segment runs longer than `exec.max_gpu_segment_ms` (RFC-0012 §7.6). Syntropy never changes `TdrDelay` or hardware-accelerated GPU scheduling on Windows; `syntropy doctor` reports both. The only machine-wide settings it may change are opt-in, ledgered `syntropy tune` steps that uninstall reverts: the macOS wired limit (below) and, from M3 with the Strix Halo provider, the Linux TTM `pages_limit`.

### Feature matrix
| Feature | Minimum hardware / OS | Fallback chain |
|---|---|---|
| NVFP4 / MXFP4 tensor-core GEMM and GEMV | NVIDIA Blackwell (sm_100, sm_120, sm_121) with CUDA 13; AMD CDNA4 (MXFP4) with ROCm | in-register FP4→FP16 dequant feeding FP16/BF16 tensor cores (FP4 values are exact in FP16) → CPU dequant kernels |
| FP8 E4M3/E5M2 tensor cores | NVIDIA Ada (sm_89), Hopper (sm_90), Blackwell; AMD RDNA4 (gfx1200/1201), CDNA3+ | lossless FP8→FP16 widening in-kernel → FP32 FMA on GPUs without FP16 tensor cores → CPU |
| INT8 / k-quant integer dot paths | NVIDIA Turing+ (IMMA); AMD RDNA3+ WMMA; Vulkan integer dot product | dequant to FP16 in-kernel → scalar dp4a-class kernels |
| BF16 matrix units | NVIDIA Ampere+; AMD RDNA3+, CDNA2+; Apple GPUs where the Metal feature tables list `bfloat` support | FP32 accumulate from widened BF16 (exact widening) |
| CUDA graphs, one per verify-window size | any GPU supported by the CUDA 13 build | per-op stream launches when capture fails (e.g. an uncapturable provider op) |
| CUDA virtual memory management (`cuMemCreate`/`cuMemMap`) for resizable expert-cache segments | device attribute `VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` | a fixed `cudaMalloc` slab sized at load; resizing needs a runner reload |
| Batched async copies (`cudaMemcpyBatchAsync`) | CUDA 13 toolkit and driver | a loop of `cudaMemcpyAsync` on the copy stream |
| Metal 4 tensor ops (MTLTensor, TensorOps), M5 GPU Neural Accelerators | macOS 26+ on Apple7+ (M1 and later); accelerators on M5-class GPUs | MSL `simdgroup_matrix` kernels (macOS 15+) → ggml Metal provider |
| `MTLResidencySet` explicit residency | macOS 15+ | per-encoder `useResource`/`useHeap` |
| Vulkan cooperative matrix (`VK_KHR_cooperative_matrix`) | Vulkan 1.3+ driver exposing it; Vulkan 1.4 preferred | subgroup-arithmetic SPIR-V kernels → ggml Vulkan provider → CPU |
| Vulkan memory budget (`VK_EXT_memory_budget`) | driver exposing it | heap size minus 10% headroom, re-checked on allocation failure |
| Intel AMX (BF16/INT8 tiles) | Xeon Sapphire Rapids or newer; Linux 5.16+ with `arch_prctl` permission granted | AVX-512 → AVX2 |
| AVX-512 (F/BW/VL, VNNI, BF16, FP16) | AMD Zen 4/5; Intel parts reporting it | AVX2+FMA, with AVX-VNNI where present → x86-64-v2 baseline (correctness only) |
| Arm SME2 | CPU reporting FEAT_SME2 (`hw.optional.arm.FEAT_SME2` on macOS, `HWCAP2_SME2` on Linux) | NEON i8mm → NEON dotprod → ARMv8.0 NEON |
| Unified memory as one `Unified` tier with no PCIe term | Apple silicon; AMD Strix Halo; NVIDIA GB10 | if detection is uncertain, model it as a discrete device with a measured PCIe term (correct, conservative) |
| Pinned anonymous host arena, progressive pinning | CUDA/HIP `cudaHostRegister`/`hipHostRegister`; `mlock` on macOS/Linux | bounded pinned staging rings over pageable memory (slower, correct) |
| Intra-node P2P (NVLink, PCIe peer access) for TP | `cudaDeviceCanAccessPeer` / HIP equivalent true | staging through pinned host memory; the solver then usually prefers in-node PP |
| RDMA over Thunderbolt 5 | macOS 26.2+, Apple silicon Mac with TB5, `rdma_ctl enable` done in Recovery, matching macOS builds on both ends | QUIC over Thunderbolt IP → QUIC over Ethernet (ADR-0011) |
| RDMA over RoCEv2 / InfiniBand | Linux, RDMA NIC, rdma-core at runtime, `trusted_fabric` set | QUIC → TCP |
| io_uring with `O_DIRECT` and registered buffers | Linux 6.1+ (`STATX_DIOALIGN`), not blocked by seccomp or sysctl | 16-thread `O_DIRECT` `pread` pool (ADR-0013) |
| IoRing | Windows build 22000+ | IOCP with unbuffered overlapped reads |
| GPUDirect Storage | Not used in v1. `syntropy probe` reports, read-only, whether cuFile offers a peer-to-peer path (Linux, NVIDIA) | `io_uring` → pinned → device copy, the only v1 path |
| DirectStorage GPU upload | Not used in v1. `syntropy probe` reports, read-only, whether it is available (Windows, D3D12) | IoRing → pinned → device copy, the only v1 path |

Unified-memory capacity rules:
- **Apple.** Usable GPU memory is the lower of `recommendedMaxWorkingSetSize` (the OS's GPU working-set limit) and RAM minus the RFC-0020 reserve, further bounded by the memory available now (RFC-0010 §2.2). Raising the OS limit (`iogpu.wired_limit_mb`) is an explicit opt-in through `syntropy tune`, which explains the trade-off, needs administrator approval and reverts on uninstall (ADR-0007 item 3); Syntropy never raises it silently.
- **GB10.** `cudaMemGetInfo` undercounts, so usable memory is MemAvailable minus 6 GiB (`tiered-memory-inference.md` §10.5).
- **Strix Halo.** Usable memory is the GPU-visible heap the driver reports. Weights run on the iGPU, because the CPU side sees ≈124 GB/s of the 256 GB/s peak (`hardware-and-performance-model.md` §2.1). Raising the heap to the vision's sizes needs the Linux TTM `pages_limit` raised, which only rule 6's opt-in `tune` step does.

### Backend support targets
| Vendor | Primary | Range | Fallback |
|---|---|---|---|
| NVIDIA | CUDA 13.x, driver ≥ 580 | sm_75 (Turing) and newer. Cubins for every architecture the toolkit supports, plus PTX of the newest for forward compatibility. v1 has no kernel that only `sm_90a` or `sm_100a` can run | CUDA 12 legacy module for Pascal (sm_60/61) and Volta (sm_70): ggml provider plus a reduced native set, Tier 3 (ADR-0020). Then Vulkan, then CPU |
| AMD | ROCm ≥ 7.2 runtime; build and test against the current ROCm release (10.1.0 as of 2026-10) | GPUs on AMD's ROCm compatibility list for the shipped release, including RDNA3, RDNA4, CDNA2+ and supported APUs | Vulkan for every other AMD GPU and when no ROCm runtime is installed, then CPU |
| Apple | Metal, native kernels plus the ggml Metal provider | Apple silicon, macOS 15+; Metal 4 paths on macOS 26+ | CPU (NEON/SME2) |
| Intel GPU | Vulkan | Arc and Xe with Vulkan 1.3 drivers | SYCL through the ggml provider (Tier 3), then CPU |
| CPU | native `std::arch` kernels, ISA selected at runtime | x86-64-v2 and up; AArch64 ARMv8.0 and up | ggml CPU provider for ops without a native kernel |

## Consequences
- **Positive.**
  - Users on current hardware get native low-precision formats, graph launch, RDMA and direct I/O.
  - Users on older hardware get the same answers, more slowly.
  - Every capability is visible and can be switched off for triage.
- **Negative.**
  - The test matrix grows with each feature × fallback pair. RFC-0062 must budget hardware CI for it.
  - The CUDA payload needs one cubin per architecture. Dequant-fallback kernels double the hot-kernel count on pre-Blackwell GPUs.
- **Work created.**
  - RFC-0010: detection and capability bits.
  - RFC-0040: per-op selection and provider fallbacks.
  - RFC-0042: dtype tolerances per fallback.
  - RFC-0052: per-backend module packaging.
  - RFC-0060 and RFC-0062: feature-vs-fallback parity on CI hardware.
  - RFC-0061: performance gates per feature.

## What would make us revisit
- A fallback cannot be made to match its feature within the ADR-0019 tolerance on some dtype. That pair must then be reclassified as approximate, or the feature dropped.
- CUDA's next major release drops Turing. Then re-run the floor decision and move sm_75 to the legacy module.
- An Apple chip ships a FP4/FP8 GPU format with Metal support. Add a row.
- Resolved: AMD's release history lists ROCm 10.0.0 (2026-08-26) and 10.1.0 (2026-10-05) after the 7.2 and 7.14 lines (rocm.docs.amd.com, "ROCm release history"), and upstream ggml ships ROCm 10.0 builds (`backends-and-portability.md` §5.1, §5.3). The policy is ROCm 7.2 as the minimum runtime, with builds and tests against the current release. Revisit if AMD drops a GPU family in the current release that still has a meaningful user base.
- SME2 support on specific Apple CPUs was not confirmed from Apple documentation. The runtime feature check makes this safe, but the matrix names no Apple generation for it.

## Revisions

- **2026-10-19, RFC-0001 amendment pass.** The Apple capacity rule matches ADR-0007 item 3: the cap is the lower of the OS GPU working-set limit and RAM minus reserve, and raising the limit is an explicit opt-in through `syntropy tune`.
- **Revised 2026-10-20:** GPUDirect Storage and DirectStorage leave v1. NVIDIA documents GDS direct mode for data-centre and workstation GPUs only, so GeForce users take the bounce path Syntropy already builds, and DirectStorage's gain is GPU decompression, which uncompressed quantised weights do not use. Both rows stay as read-only capability reports, and the revisit is a wider GDS support list. D7 is still met by the other rows. The NVIDIA target writes no `sm_90a`- or `sm_100a`-only kernels in v1: architecture-specific kernels would need their own rigs to certify, while the portable cubins and PTX run on those GPUs. New rule 6 keeps kernels within the default GPU watchdogs, because long prefill on Metal and on display GPUs under WDDM otherwise collides with them. It forbids changing `TdrDelay` or HAGS, which are machine-wide and affect every other application, and allows the Linux TTM `pages_limit`, which Strix Halo needs to reach its usable memory, only as an opt-in, ledgered `tune` step, as the macOS wired limit already is.
