# Platform support

Syntropy is not released yet. This page describes the support each platform is designed to have at 1.0, and the milestone at which it first ships. The source is [ADR-0020](plan/decisions/ADR-0020-platform-support-tiers.md), which records the minimum runtimes, CI hardware and the rules for moving a platform between tiers.

A tier is a promise about testing. The gap between Tier 1 and Tier 3 is stated at install time and in the dashboard, so nobody finds it out later. A cluster can mix tiers: placement uses each machine's measured profile, whatever its tier.

## Tier 1: supported

Prebuilt, signed builds. Parity ([ADR-0019](plan/decisions/ADR-0019-exactness-contract.md)) and performance gates are enforced, and bugs are fixed before release. Builds and tests run on every pull request. Parity and performance run on real hardware for every pull request that touches kernels, execution, tiers, I/O or networking, nightly, and on every release candidate. Any failure blocks the release.

| Platform | Minimum | Backends | First ships |
|---|---|---|---|
| Linux x86_64 with NVIDIA | kernel 5.15, glibc 2.35, CUDA 13 with driver 580 or later, Turing (`sm_75`) or newer | native CUDA, ggml CUDA, native CPU | M1a (preview) |
| Linux x86_64, CPU only | as above, x86-64-v2 or newer | native CPU (AVX2, AVX-512, AMX where present), ggml CPU | M1a (preview) |
| Windows x86_64 with NVIDIA | Windows 11, CUDA 13 with driver 580 or later | native CUDA, ggml CUDA, native CPU | M1b |
| Windows x86_64, CPU only | Windows 11 | native CPU, ggml CPU | M1b |
| macOS on Apple silicon | macOS 15; Metal 4 paths on macOS 26; RDMA over Thunderbolt 5 on 26.2 | native Metal, ggml Metal, native CPU (NEON, SME2 where present) | M1b |
| macOS on Apple silicon, CPU only | macOS 15 | native CPU, ggml CPU | M1b |

## Tier 2: supported with known gaps

Prebuilt builds. Parity is enforced; performance is tracked but doesn't gate a release. Builds and unit tests run on every pull request, and real-hardware parity and performance runs happen weekly and on every release candidate. A build or parity failure blocks that backend module, and the known gaps are listed in the release notes.

| Platform | Minimum | Backends | First ships |
|---|---|---|---|
| Vulkan on AMD, Intel and NVIDIA GPUs (Linux and Windows) | Vulkan 1.3 driver | ggml Vulkan from M2; native Vulkan (Slang to SPIR-V) from M3 | M2 (preview) |
| AMD Strix Halo | Linux (ROCm where AMD lists the APU, otherwise Vulkan); Windows (Vulkan) | ggml Vulkan from M2; native HIP or Vulkan with a unified memory tier from M3 | M2 (preview) |
| Linux x86_64 with AMD ROCm | ROCm 7.2 or later, on a GPU from AMD's ROCm list | native HIP, ggml HIP, CPU | M3 |
| Linux arm64, GB10-class | as Linux; CUDA 13 for aarch64 | native CUDA with a unified memory tier, ggml CUDA, CPU | M4 |
| Linux arm64, Raspberry Pi 5-class (CPU) | ARMv8.2 or newer | native CPU (NEON dot product), ggml CPU | by M4 |
| RDMA NICs (RoCEv2, InfiniBand) on Linux | rdma-core at runtime, interfaces marked `trusted_fabric` | RDMA transport | M3 |
| Several GPUs in one machine | per backend | layer splits across GPUs from M2; tensor parallelism over peer-to-peer links on CUDA and HIP from M3 | M2 |

## Tier 3: best effort

Builds are published as experimental. They are built nightly and run on hardware when hardware is available. A Tier 3 platform never blocks a release, and a parity failure withdraws its build. Fixes from the community are welcome.

| Platform | Minimum | Backends |
|---|---|---|
| Android, as a node | Android 10 (API 29), arm64-v8a | native CPU, ggml Vulkan, OpenCL (Adreno), Hexagon |
| iOS, as a node for small slices | iOS 26, A14 or newer | native Metal and CPU, linked statically, with the runner in-process |
| Intel SYCL | the oneAPI runtime ggml needs | ggml SYCL, CPU |
| CUDA 12 for Pascal and Volta | CUDA 12, driver 580 branch | ggml CUDA and a reduced native set, CPU |
| WebGPU | native builds through Dawn or wgpu | ggml WebGPU, CPU |
| Windows 10 22H2 | out of vendor support, no IoRing | as Windows 11, with IOCP for I/O |
| Linux under WSL2 with an NVIDIA GPU | Windows 11 with WSL2, CUDA 13 through the Windows driver | native CUDA, ggml CUDA, CPU; KV doesn't spill to pinned RAM by default, because WSL2 caps pinned allocations |

Tier 3 builds arrive in M4.

## Not supported in 1.0

macOS on Intel, 32-bit operating systems, Windows on arm64, and a node hosted in a browser. A platform outside these tables may work, but `syntropy setup` reports it as unsupported and nothing on it counts as a regression.

## When each platform ships

| Milestone | Platforms first released |
|---|---|
| M0a, M0b | None. Linux x86_64 with CUDA and CPU is built and tested, but nothing is released. |
| M1a | Linux x86_64 with NVIDIA, and Linux x86_64 CPU-only. The first public release, labelled a **preview**. |
| M1b | Windows x86_64 with NVIDIA, macOS on Apple silicon, and CPU-only on both. Preview until the M2 launch. |
| M2 | A `vulkan` module for Radeon, Intel Arc, laptop integrated GPUs and Strix Halo, built on ggml's Vulkan backend without Syntropy's own Vulkan kernels or VRAM arenas: a **Tier 2 preview**. Its speed says nothing about the native provider that follows. |
| M3 | Native HIP and Vulkan providers in the `rocm` and `vulkan` modules, as a preview until M4 meets their Tier 2 CI bar. |
| M4 | ROCm, Vulkan and the remaining Tier 2 platforms reach their Tier 2 CI bar, and Tier 3 builds start. |

A platform's tier promise applies once it leaves preview: from the milestone it first ships, or, for a platform that ships first as a preview, from the milestone that ends the preview. Before that, `syntropy setup` reports it as not yet released rather than unsupported. When setup finds a GPU that no released module can use yet, it says so and names the milestone that brings it, instead of falling back to the CPU without telling you.

If native HIP and Vulkan kernels are cut from 1.0 under the scope-cut order in [RFC-0001 §16](plan/rfcs/RFC-0001-system-architecture.md#scope-cut-order), the ROCm and Vulkan rows stay on the ggml provider through 1.0, and the M2 `vulkan` module is that coverage.

## Moving between tiers

- **Tier 2 to Tier 1** needs a dedicated hardware CI runner, parity and performance gates green for three consecutive releases, those gates made required checks, and signed builds.
- **Tier 3 to Tier 2** needs periodic hardware access, parity green on the conformance set, and prebuilt builds.
- **A platform drops a tier** when its CI hardware has been unavailable for more than 30 days, when its vendor drops the runtime it depends on, or when parity fails across two releases.

Every move is recorded in ADR-0020 and in the release notes.
