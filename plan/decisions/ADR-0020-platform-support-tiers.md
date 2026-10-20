# ADR-0020: Platform support tiers
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0001, ADR-0002, ADR-0004, ADR-0011, ADR-0013, ADR-0014, ADR-0015, ADR-0018, ADR-0019, ADR-0021, ADR-0403; RFC-0001 §16, RFC-0040, RFC-0052, RFC-0060, RFC-0061, RFC-0062; vision §3

## Context
D7 promises that Syntropy installs and runs on any supported device. A promise only means something if it says which platforms are tested, how often, and on what hardware.

- **Demand.** The research finds the strongest user demand on Linux GPU and Windows, and treats Metal as one backend among several (`plan/research/multi-device-inference.md` §15.10).
- **Backend matrix.** ggml covers CUDA, HIP, Metal, Vulkan, SYCL, OpenCL (Adreno), Hexagon, WebGPU and CPU. Syntropy's native kernels cover the Tier 1 backends first (ADR-0002) (`backends-and-portability.md` §0, §1.2).
- **Coverage lag.** New attention types land on CUDA first, and other backends follow late or never (§7.2 risk 1). So model × backend support has to be stated per tier, with the CPU as the per-op fallback.
- **Platform constraints.** These shape what each tier can promise (`backends-and-portability.md` §5.3–§5.4, §6.1; ADR-0013, ADR-0014):
  - CUDA 13 drops Pascal and Volta.
  - ROCm supports a per-architecture list.
  - io_uring is blocked on Android and in Docker's default seccomp.
  - iOS forbids downloaded code and spawned processes, and caps per-process memory: about 6 GB was reported on both 8 GB and 12 GB iPhones, in a single report.
  - macOS requires notarisation of every nested binary.
  - Windows signing for individuals is limited to the USA and Canada.
  - Windows 10 lacks IoRing, which needs build 22000+.
- **Surveyed projects** that listed platforms as supported without CI on them later shipped regressions and "planned" support for most of a year (`multi-device-inference.md` §13 item 21; `tiered-memory-inference.md` §18).
- **CI hardware time is scarce.** Hardware jobs must be path-filtered and cancellable, not always on.

## Options considered
1. **One tier: everything ggml supports is "supported".** Honest only if every backend is tested on hardware, which no realistic lab covers. Rejected.
2. **Only the platforms we can test on every PR.** Honest, but it abandons Android, iOS, SYCL and legacy CUDA users who would get acceptable results through the ggml provider. Rejected by D7.
3. **Three tiers with explicit CI coverage and promotion rules.** Chosen. Every platform has a stated support level, and the gap between Tier 1 and Tier 3 is visible to users.

## Decision
### Tier definitions
| Tier | Promise | CI coverage | Release gating |
|---|---|---|---|
| **1** | Supported. Prebuilt, signed artefacts. Parity (ADR-0019) and performance gates enforced. Bugs fixed before release | Build, unit and integration tests on every PR. Real-hardware parity and performance jobs on every PR that touches `kernels-*`, `exec`, `tier`, `io` or `net` (path-filtered, concurrency-cancelled), plus nightly and on every release candidate | Any failure blocks the release |
| **2** | Supported with known gaps. Prebuilt artefacts. Parity enforced; performance tracked, not gated. Gaps listed in release notes | Build and unit tests on every PR (hosted or cross-compiled). Real-hardware parity and performance runs weekly and on every release candidate | Build or parity failure blocks that backend module; performance regressions are documented |
| **3** | Best effort. Builds published as experimental. Community fixes welcome | Nightly build; hardware runs when hardware is available | Never blocks a release. A parity failure withdraws that artefact |

A parity failure on any platform with hardware coverage withdraws the affected backend module from the release, rather than shipping it unverified.

### Platform matrix
| Platform | Minimum OS / runtime | Tier | Backends (native + ggml provider) | I/O engine (ADR-0013) | Data-plane transports (ADR-0011) | CI hardware |
|---|---|---|---|---|---|---|
| Linux x86_64 + NVIDIA | kernel 5.15+, glibc 2.35+; CUDA 13.x, driver ≥ 580; sm_75+ | 1 | CUDA native, ggml CUDA, CPU native | `UringDirect` on 6.1+, else `PreadPool`; GPUDirect Storage not in v1, the probe reports the capability only (RFC-0021 §9) | QUIC, TCP, shm; RoCE/IB Tier 2 | Ada-class and Blackwell-class consumer GPUs; a 2-node 10 GbE pair for multi-node jobs |
| macOS arm64 (Metal) | macOS 15+; Metal 4 paths on 26+; RDMA on 26.2+ | 1 | Metal native, ggml Metal, CPU native (NEON, SME2 where detected) | `MacNoCache` or `MacMapped` by probe | QUIC, TCP, shm; RDMA over TB5 nightly | an M1/M2-class 16 GB machine (floor) and a recent Max/Ultra-class machine; two TB5 Macs with RDMA enabled |
| Windows x86_64 + NVIDIA | Windows 11; CUDA 13.x, driver ≥ 580 | 1 | CUDA native, ggml CUDA, CPU native | `IoRing`, IOCP fallback; DirectStorage not in v1, the probe reports the capability only (RFC-0021 §9) | QUIC, TCP, shm | one Ada- or Blackwell-class consumer GPU |
| Linux x86_64 CPU-only | as Linux above; x86-64-v2+ (perf gates on AVX2 and AVX-512 hosts) | 1 | CPU native, ggml CPU | as Linux above | QUIC, TCP, shm | an AVX2 host and an AVX-512 host |
| Windows x86_64 CPU-only | Windows 11 | 1 | CPU native, ggml CPU | `IoRing`, IOCP | QUIC, TCP, shm | shared with the Windows NVIDIA runner, GPU disabled |
| macOS arm64 CPU-only | macOS 15+ | 1 | CPU native, ggml CPU | as macOS above | QUIC, TCP, shm | shared with the macOS floor runner, Metal disabled |
| Linux x86_64 + AMD ROCm | ROCm ≥ 7.2 (build against current, 10.1.0 as of 2026-10); GPUs on AMD's ROCm list | 2 | HIP native, ggml HIP, CPU | as Linux | QUIC, TCP, shm, RoCE | one RDNA3 and one RDNA4 card |
| Any OS + Vulkan (AMD, Intel, NVIDIA) | Vulkan 1.3+ driver | 2 | Vulkan native (Slang → SPIR-V), ggml Vulkan, CPU. M2 ships ggml Vulkan only, as a Tier 2 preview (see "Platform staging") | per OS | per OS | Intel Arc, an AMD card on the Vulkan path, an NVIDIA card on the Vulkan path (Linux and Windows) |
| Linux arm64: GB10-class | as Linux; CUDA 13 aarch64 | 2 | CUDA native with a `Unified` tier, ggml CUDA, CPU | as Linux | QUIC, TCP, shm, RoCE (ConnectX) | one GB10-class system |
| Linux arm64: Raspberry Pi 5-class CPU | as Linux; ARMv8.2+ | 2 | CPU native (NEON dotprod), ggml CPU | `UringDirect` or `PreadPool` | QUIC, TCP | one Pi 5-class board |
| AMD Strix Halo | Linux (ROCm where AMD lists the APU, else Vulkan); Windows (Vulkan) | 2 | HIP or Vulkan native with a `Unified` tier, ggml, CPU. In M2, the ggml Vulkan preview only, without the `Unified` tier's arenas | per OS | QUIC, TCP, USB4/TB IP | one Strix Halo system |
| Linux + RDMA NICs (RoCEv2/IB) | rdma-core at runtime, `trusted_fabric` | 2 (feature) | n/a | n/a | RDMA | a 2-node RoCE pair, weekly |
| Multi-GPU in one host (P2P TP) | per backend | 2 (feature) | CUDA, HIP | n/a | intra-node P2P | one dual-GPU host, weekly |
| Android (as a node) | Android 10+ (API 29), arm64-v8a | 3 | CPU native, ggml Vulkan, OpenCL (Adreno), Hexagon | `PreadPool` | QUIC, TCP | nightly NDK build; device runs when available |
| iOS (as a node, small slices only) | iOS 26+, A14+ | 3 | Metal native and CPU, statically linked; embedded runner (ADR-0015) | `MacNoCache` | QUIC, TCP | nightly Xcode build |
| Intel SYCL | oneAPI runtime per ggml | 3 | ggml SYCL, CPU | per OS | per OS | none |
| CUDA 12 legacy (Pascal, Volta) | CUDA 12.x, driver 580 branch | 3 | ggml CUDA plus a reduced native set, CPU | per OS | per OS | nightly build only |
| WebGPU | native builds through Dawn/wgpu | 3 | ggml WebGPU, CPU | per OS | per OS | nightly build only |
| Windows 10 22H2 x86_64 | out of vendor support; no IoRing | 3 | as Windows 11 | IOCP | QUIC, TCP, shm | none |
| Linux under WSL2 + NVIDIA paravirtual GPU | Windows 11 with WSL2; CUDA 13 through the Windows driver | 3 | CUDA native, ggml CUDA, CPU native | `UringDirect` or `PreadPool` | QUIC, TCP, shm | none |

**WSL2 (Tier 3).** `cudaHostAlloc` is capped at about 1 GiB there, so the tier manager registers ordinary memory with `cudaHostRegister` instead, and KV does not spill to `HostPinned` unless `tier.wsl_kv_spill` opts in after measurement (RFC-0001 §9.2, RFC-0020 §5). `setup` and `doctor` report the node as Tier 3 with that note (RFC-0052 §4.2).

**Not supported in v1:** macOS x86_64, 32-bit operating systems, Windows on arm64, and a browser-hosted node. A platform outside the matrix may work, but the tooling says it is unsupported and reports nothing as a regression.

### Platform staging by milestone
*Added 2026-10-20.* The matrix states each platform's support level at 1.0. Platforms reach users in stages, so that the first end-to-end integration and the first users arrive before all of M1 is done, and so that AMD and Intel GPUs get a GPU path before the native providers land in M3 (RFC-0001 §16):

| Milestone | Platforms first released | How it is labelled |
|---|---|---|
| M0a, M0b | none: Linux x86_64 with CUDA and CPU is built and tested, but nothing is released | – |
| M1a | Linux x86_64 + NVIDIA; Linux x86_64 CPU-only. The first public release | **preview** |
| M1b | Windows x86_64 + NVIDIA; macOS arm64 (Metal); Windows and macOS CPU-only. Scope gated on M1a's measurements | preview until the M2 launch |
| M2 | The launch: announcement and public benchmark page. Adds a ggml-only `vulkan` module for Radeon, Intel Arc, laptop iGPUs and Strix Halo: ggml's Vulkan backend through the ggml provider, allocations owned by ggml's buffer type (ADR-0403 decision 2's fallback), no elastic arenas and no native Vulkan kernels | `vulkan`: **Tier 2 preview** |
| M3 | Native HIP and Vulkan providers in the `rocm` and `vulkan` modules | preview until M4 meets their Tier 2 CI bar |
| M4 | Tier 2 platforms to their CI bar, and Tier 3 builds | per the matrix |

- A platform's tier promise applies from the milestone it is first released, except that the ROCm and Vulkan rows carry the Tier 2 promise from M4, when their CI bar is met (RFC-0001 §16). Before a platform is released, `setup` reports it as not yet released rather than unsupported.
- If the scope cut for native HIP and Vulkan is taken (RFC-0001 §16), the ROCm and Vulkan rows stay on the ggml provider through 1.0, and the M2 module is that coverage.

### Per-tier user experience
- Install and the dashboard show the node's tier and backend. Tier 2 known gaps and Tier 3 status are stated at install, not discovered later (RFC-0052, RFC-0051).
- When `setup` finds a GPU that no released module can use yet, it says so and names the milestone that brings it, rather than falling back to the CPU silently.
- A cluster may mix tiers. The solver uses each node's measured profile regardless of tier.

### Promotion and demotion
- **Tier 2 → 1.** All four conditions must hold:
  - a dedicated hardware CI runner in the lab, set up by a maintainer (**Maintainer action**)
  - parity and performance gates green for 3 consecutive releases
  - the platform's parity and performance gates are required checks, so no change that breaks the platform merges, and its platform tasks are in the epics (ADR-0021)
  - signed artefacts
- **Tier 3 → 2.** All three conditions must hold:
  - periodic hardware access
  - parity green on the conformance set
  - prebuilt artefacts
- **Demotion.** A platform drops one tier when its CI hardware has been unavailable for more than 30 days, when a vendor drops the runtime it depends on, or when parity fails across two releases.
- Each move is recorded by amending this ADR's matrix and noted in release notes.

## Consequences
- **Positive.**
  - Users know exactly what is tested on hardware like theirs.
  - CI hardware time is concentrated on Tier 1 paths and the platforms with the most demand.
  - The ggml provider gives Tier 3 platforms real reach without native kernel work.
- **Negative.**
  - Tier 1 needs self-hosted hardware runners on three OSes: at least two NVIDIA generations, two Apple machines plus a TB5 pair, and a 10 GbE pair. The lab has to keep all of them running.
  - Windows 10 users get Tier 3.
  - Mobile is best effort through 1.0.
  - Windows and Apple users get no public release until M1b, although Windows is scenario S1's largest audience.
  - The M2 `vulkan` preview runs at ggml's speed, without Syntropy's VRAM arenas or capture, so its numbers say nothing about the M3 native provider.
- **Work created.**
  - RFC-0062: runners, path filters, cadences and release gating per tier.
  - RFC-0060: parity jobs per platform.
  - RFC-0061: performance gates for Tier 1.
  - RFC-0052: per-platform packaging, signing and tier display.
  - RFC-0040: backend modules per platform.

## What would make us revisit
- Issue volume, release download counts per platform asset, and user reports (Syntropy collects no telemetry, ADR-0510) show a Tier 2 platform, such as ROCm or Vulkan on AMD, with comparable demand to a Tier 1 platform. Add a runner for it to the lab (**Maintainer action**), and promote it.
- Windows on arm64 or a browser-hosted node gains a credible backend and user demand.
- Self-hosted CI capacity for Apple RDMA pairs or 10 GbE pairs cannot be sustained in the lab. Move those features to Tier 2 cadence, and say so in the release notes.
- Settled on 2026-10-19:
  1. ROCm versioning: ROCm 7.2 is the minimum runtime, with builds against the current release (10.1.0). See ADR-0014.
  2. Windows 11 is the Tier 1 Windows floor, and Windows 10 is Tier 3. Accepted.
  3. "WebGPU" means a backend in native builds. A browser-hosted node is not supported in v1. Accepted.

## Revisions

- **Revised 2026-10-19:** WSL2 with an NVIDIA paravirtual GPU is added as Tier 3. RFC-0052 already installs on it and `doctor` already recognises it; without a matrix row, this ADR's rule that a platform outside the matrix is reported as unsupported contradicted both.
- **Revised 2026-10-20:** platforms are released in stages. M1a releases Linux CUDA and CPU as a preview, M1b adds Windows and Apple silicon, and the launch is at M2, so that integration and user feedback start before all of M1 is done. M2 also ships a ggml-only `vulkan` module as a Tier 2 preview, because Radeon, Arc, laptop iGPUs and Strix Halo would otherwise run CPU-only until M3. The Linux NVIDIA and Windows NVIDIA rows drop GPUDirect Storage and DirectStorage, which are not in v1 (RFC-0021 §9). The native ROCm and Vulkan modules ship in M3 and reach their Tier 2 bar in M4, as RFC-0001 §16 sets.
