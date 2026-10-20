# ADR-0004: MLX is not a v1 engine

- Status: Accepted
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D1, D7), ADR-0002, ADR-0007, ADR-0011, ADR-0014, ADR-0019, ADR-0020, RFC-0020, RFC-0040, RFC-0061

## Context

Apple silicon is a Tier 1 platform (ADR-0020). MLX is Apple's MIT-licensed array framework, strong on Metal, and it now also runs on Linux CUDA (SM ≥ 7.5) and CPU (`plan/research/backends-and-portability.md` §0 item 3, §1.3.2). The question is whether Syntropy carries it as a second engine in v1.

**Placement.** MLX's memory API is process-wide limits only: `set_memory_limit`, `set_wired_limit`, `set_cache_limit`. It has no per-array placement and no explicit pinned/device split. On CUDA it uses managed memory where available and falls back to `cudaMallocHost` (§1.2, §1.3.2). Syntropy's tier manager has to decide where each tensor and expert lives (ADR-0007). Under MLX's allocator it could not place an expert in a chosen tier, stream it into a chosen buffer, or compute it in place where it sits. MLX can therefore be a compute backend, but not the tiering substrate (§0 item 3).

**Apple residency without MLX.** Metal 4 makes residency explicit through `MTLResidencySet`, attachable per queue (up to 32 sets) or per command buffer. On unified memory the "VRAM tier" is the wired working set, capped by `recommendedMaxWorkingSetSize` (about 2/3 to 3/4 of RAM by default), so Mac tiering is unified-memory-resident versus SSD-resident (§6.1). Syntropy reaches all of this from its own Metal code.

**Speed evidence on Apple** (§1.4; none reproduced by us):

| Comparison | Result | Caveat |
|---|---|---|
| M5 Max 128 GB, Qwen3.6-35B-A3B 8-bit, decode at 128-token context | ggml Metal 93 tok/s, MLX 85 | third-party, decode only |
| Same machine, Qwen3.6-27B | ggml Metal 18, MLX 17 | third-party |
| A surveyed desktop runner's MLX engine (NVFP4) versus its ggml engine (Q4_K_M), Qwen3.5-35B-A3B | 58 → 112 tok/s decode | different quantization, so not engine against engine |

Calibration anchors from `hardware-and-performance-model.md` §2.2 put ggml Metal at η = 0.43–0.70 on a 7B dense model across M-series chips, and MLX at η ≈ 0.53 on DeepSeek-V3 4-bit on M3 Ultra. Both lose efficiency on large chips and MoE. The research takeaway is that for a tiering engine, hit rate and I/O scheduling dominate kernel speed (`backends-and-portability.md` §1.4).

**Cost of a second engine.** A second engine doubles the parity surface (ADR-0019), splits Apple kernel work, and brings a second allocator that the tier manager cannot see.

**Apple features MLX is not needed for.** Metal 4 tensor APIs and the M5 GPU neural accelerators are reachable from MSL kernels (ADR-0014). RDMA over Thunderbolt 5 (macOS 26.2+) is an OS facility that Syntropy's transport uses directly (ADR-0011; `hardware-and-performance-model.md` §2.5).

## Options considered

| Option | For | Against |
|---|---|---|
| A. MLX as a full second engine on macOS | Mature Apple kernels; early access to new Apple features | Cannot implement D1 tiering; two engines in parity; Python-first tooling with a thinner C API (`mlx-c`) |
| B. MLX as an opaque whole-stage device on macOS | Some kernel speed for stages that fit entirely in unified memory | Still no tiering inside the stage; numerics and stage-boundary semantics differ from ours; doubles bring-up per family |
| C. MLX as an optional `KernelProvider` in v1 | Per-op choice by benchmark, like any provider (ADR-0002) | Needs MLX to compute on Syntropy-owned Metal buffers without copying, which its API does not document; unproven value |
| **D. No MLX in v1; native Metal kernels plus the ggml Metal provider** | One engine; tiering works the same on Apple as elsewhere | Apple kernel quality is entirely our job |

## Decision

1. **MLX is not a v1 engine, backend or provider.**
2. **Apple silicon is served by** native Metal kernels (MSL, Metal 4 tensor APIs where present, `simdgroup_matrix` otherwise) for the hot ops of ADR-0002, plus the ggml Metal provider for full op coverage. Residency is explicit through `MTLResidencySet` under the tier manager (ADR-0007, RFC-0020).
3. **MLX is a comparison point only.** RFC-0061 includes it in the Apple benchmark matrix.
4. **Revisit gate after M2.** MLX may come back as an optional `KernelProvider` if both hold:
   - **Speed:** MLX decodes at least **15% faster** than Syntropy's Metal path at batch 1, on at least two Tier 1 Apple machines (one Max-class and one Ultra-class chip), on at least two Tier 1 models (one dense, one MoE). The models must be fully resident in unified memory, so tiering does not confound the result. Quantizations must match to within 0.25 bits per weight, at 4K context, after a full release cycle of Metal kernel work.
   - **Interop:** MLX can execute on externally allocated Metal buffers without copying, so the tier manager keeps ownership.

## Consequences

Positive:
- One parity surface and one allocator. Tiering, KV paging, the expert cache and stage execution behave the same on Apple as on other platforms, with `Unified` as the fast tier.
- Mixed Apple and non-Apple clusters behave like homogeneous ones, because the transport and runtime are the same everywhere.

Negative:
- Syntropy has to match MLX-class kernel quality on Apple itself. RFC-0040 gives Metal kernels the same priority as CUDA kernels.
- New Apple GPU features reach Syntropy only when our Metal kernels adopt them.

Work created: RFC-0040 (Metal native provider, Metal 4 tensor paths), RFC-0020 (`MTLResidencySet` integration and wired-limit policy), RFC-0061 (MLX in the Apple benchmark matrix, with the gate measurement defined above).

## What would make us revisit

- The gate in Decision item 4 is met.
- MLX adds an allocation API with per-array placement into caller-provided buffers. That would remove the interop condition, and a provider evaluation would follow.
- Apple exposes a capability we need only through MLX or Core ML: neural-accelerator features unreachable from MSL, or RDMA features unreachable outside MLX's collectives.
