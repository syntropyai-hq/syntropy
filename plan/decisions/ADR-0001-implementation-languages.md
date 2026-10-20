# ADR-0001: Implementation languages

- Status: Accepted
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D6, D7), ADR-0002, ADR-0012, ADR-0014, ADR-0015, ADR-0020, RFC-0001, RFC-0040, RFC-0052, RFC-0060, RFC-0062

## Context

ADR-0000 D6 ranks languages by delivered speed on this workload first, portability second and familiarity last. A component may use a different language when that is faster.

Where a decode step spends its time (`plan/research/hardware-and-performance-model.md` §0, §2.2, §3.2, §3.4, §3.7):

| Cost | Size | Decided by |
|---|---|---|
| Kernels reading weights and KV | almost all of `Σ B/W_eff`; η = 0.43–0.79 of peak for GPU-class engines, 0.24–0.35 end to end for CPU-hosted experts | kernel code generation and memory access patterns |
| Fixed per-token overhead `t_fixed` | ≈5 ms per token in one two-point fit on M3 Ultra [estimate] | host-side launch, synchronisation, graph replay |
| CPU↔GPU hand-off inside a layer | ≈1 µs wire + 5–20 µs sync [estimate] | doorbell latency, thread wake-ups |
| Network hop, one decode token | α ≈ 100–250 µs on 2.5 GbE, 10–25 µs on RDMA over TB5 [estimate] | transport stack, serialisation, sync on both ends |
| NVMe expert reads | one issuing thread peaks at ≈80K reads/s (`tiered-memory-inference.md` §10.3) | submission cost per request, queue depth |

Two measured details show how much host code matters. A driver sleep/wake round trip costs about 0.3 ms on Windows, longer than a small expert batch, so the surveyed single-GPU MoE engine spins on a pinned-memory doorbell instead (`tiered-memory-inference.md` §5.1). Moving its cache update off the token thread added 4–5% decode (§5.3). The host language must give direct control over threads, memory layout, pinning and atomics, with no collector or interpreter in the loop.

Other forces:
- The system layer parses untrusted input: GGUF and safetensors headers from the internet, wire frames from other nodes over invite links (ADR-0000 D4), HTTP from clients. Memory safety there is a correctness requirement.
- Every surveyed project that reaches phones is native code with a C ABI or uniffi on top; none uses Python or Go on device (`backends-and-portability.md` §2.1). The Rust path to iOS and Android is `staticlib` plus uniffi Swift/Kotlin bindings (§2.2).
- GPU kernel ecosystems are language-bound: CUDA C++ (and CUTLASS/CuTe templates), HIP, Metal Shading Language, SPIR-V. ggml, the kernel provider of ADR-0002, exposes C headers that `bindgen` consumes (§2.2).
- Rust bindings exist for the needed driver surfaces: `cudarc` for the CUDA driver API, `nvml-wrapper` (§2.2), plus Vulkan and Metal bindings.

## Options considered

| Option | Hot-path speed | Safety on network and parser code | Portability and packaging | Verdict |
|---|---|---|---|---|
| **A. Rust system layer; kernels in each backend's native language; CPU kernels in Rust `std::arch`** | Same LLVM back end as Clang; no GC; explicit SIMD; zero-cost `extern "C"` calls into kernel launch shims | Strong; `unsafe` is confined and auditable | `cargo` targets, one static binary, uniffi for mobile | **Chosen** |
| B. C++23 everywhere | Equal codegen; best interop with CUTLASS host code | Weak across the whole network, parser and concurrency surface | CMake presets per target; manual C ABI plus Swift/Kotlin wrappers for mobile | Rejected: no speed gain, larger unsafe surface |
| C. C++23 engine core inside the runner, Rust daemon around it | Equal codegen; one C ABI boundary per runner | Runner still parses activation frames and tier I/O completions in unsafe code | Two first-class host languages in one process | Rejected: the boundary buys nothing measurable and splits the executor, tier manager and transport across languages |
| D. Zig | Comparable codegen, strong cross-compilation | Moderate | No C++ interop for template kernels; little mobile precedent; small GPU and async ecosystem (§2.2) | Rejected: ecosystem immaturity |
| E. Mojo for kernels or system | Promising on NVIDIA/AMD | – | Narrow backend coverage (no Vulkan, early Apple GPU); toolchain licence and openness | Rejected |
| F. Go or Python system layer | GC or interpreter on the control path; Go has no io_uring in its runtime (§2.2) | Strong | Python cannot ship in phone apps; Go needs cgo for every GPU and kernel call (§2.2) | Rejected |
| G. A kernel DSL compiled at runtime | Competitive on CUDA/ROCm GEMM | – | No Metal, Vulkan or CPU targets; needs a compiler on the user's machine | Rejected for v1 |

## Decision

1. **Rust, 2024 edition, stable toolchain** pinned in `rust-toolchain.toml`, for the whole system layer: runtime and executor, scheduler, tier manager, I/O engines, transports, control plane, solver driver, model store, API and CLI. Release builds never use a nightly compiler.
2. **Kernels in each backend's native language**, compiled ahead of time:
   - **NVIDIA:** CUDA C++ with CUTLASS/CuTe-class templates for tensor-core FP4/FP8 GEMM and GEMV on Blackwell, Hopper and Ada. Ship cubins per architecture plus PTX for forward compatibility (support targets in ADR-0014).
   - **AMD:** HIP, sharing source with CUDA where portable, with separate WMMA paths for RDNA4/CDNA.
   - **Apple:** Metal Shading Language, compiled to `.metallib`; Metal 4 tensor APIs where present, `simdgroup_matrix` kernels otherwise.
   - **Vulkan:** Slang compiled offline to SPIR-V; `VK_KHR_cooperative_matrix` where the driver exposes it, subgroup kernels otherwise.
   - **CPU:** Rust with `std::arch` intrinsics: AVX2, AVX-512 (VNNI/BF16), AVX-VNNI, AMX, NEON (dotprod, i8mm), SVE2, SME2. One function variant per ISA level, selected once at startup from CPUID or `getauxval`/`sysctl`. Where stable `std::arch` lacks an intrinsic family, the kernel uses stable `core::arch::asm!`.
3. **The language boundary is a launch, never a decision.** Rust launches GPU kernels through the driver APIs (CUDA driver, HIP module, Metal, Vulkan). Where a template library needs host-side C++ to instantiate or launch a kernel, that code lives in a narrow `extern "C"` shim inside the provider crate (RFC-0040) and holds no scheduling, allocation or residency logic. ggml is reached through `bindgen` over its C headers (ADR-0002).
4. **`unsafe` policy.** `#![forbid(unsafe_code)]` in the control-plane, API, cluster, solver and manifest-parsing crates. `unsafe` is allowed only in kernel dispatch, FFI, DMA, pinned memory and I/O crates, each block with a written invariant. All untrusted bytes are parsed in safe Rust and fuzzed (RFC-0060).
5. **Hot-path rules.** On the steady-state token path: no heap allocation, no lock acquisition, no system call except deliberate I/O submission and doorbell waits, and no async runtime (ADR-0012).
6. **Packaging.** The system layer and CPU kernels link statically into the single `syntropy` binary (ADR-0015). GPU vendor runtimes and large kernel modules are loaded at runtime per RFC-0052. Mobile builds expose the Rust API through uniffi.

## Consequences

Positive:
- One host language from API socket to kernel launch. The executor, tier manager and transport share types and memory without a C ABI between them.
- Network-facing and concurrent code is memory-safe; the unsafe surface is a list of crates, not the codebase.
- Cross-compilation, a single binary and the mobile path follow established Rust practice.

Negative:
- CI needs every kernel toolchain: `nvcc`, `hipcc`, the Metal compiler, `slangc` (RFC-0062). Kernel builds dominate CI cost.
- CUTLASS-class host shims are C++ we still own. They must stay thin enough to review as glue.
- The Rust driver bindings (`cudarc`, Vulkan and Metal crates) are third-party dependencies with their own release cadence.
- The contributor pool that overlaps ggml is C/C++-heavy (`backends-and-portability.md` §2.2).

Work created:
- RFC-0001: crate layout and the `unsafe` boundary per crate.
- RFC-0040: kernel build pipeline per backend, ISA dispatch, shim rules.
- RFC-0052: binary and module packaging. RFC-0060: fuzz targets for every parser. RFC-0062: toolchain matrix in CI.

## What would make us revisit

- A profile on a Tier 1 platform shows more than 2% of decode step time in Rust host code that an equivalent C++ dispatch loop avoids (same driver calls, same graph replay). The first response is to fix the Rust code; a language change needs a repeatable gap.
- A Rust CPU kernel stays more than 10% below an equivalent C intrinsics kernel on the same quant type and machine after tuning. We would then allow C intrinsics kernels behind the same `KernelProvider` trait for that op.
- A kernel DSL ships ahead-of-time kernels at or above hand-written throughput on at least two Tier 1 backends, under a permissive licence, with no compiler on the user's machine.
- **Resolved (2026-10-19).** Where stable Rust lacks intrinsics for AMX, SVE2 or SME2, those kernels use stable `asm!`, or a small C or assembly translation unit linked statically. RFC-0040 records the stable status of each family before M1. The decision on Rust is unchanged.
