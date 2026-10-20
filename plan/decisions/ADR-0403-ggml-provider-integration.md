# ADR-0403: The ggml provider wraps single ops as prebuilt ggml graphs over Syntropy memory, statically linked with one backend per artefact

- Status: Accepted (2026-10-20); revised 2026-10-20 (decisions 2 and 3)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0002, ADR-0007, ADR-0014, ADR-0020, ADR-0507, ADR-0602; RFC-0040 §8, §10.4

## Context

- ADR-0002 adopts ggml, the tensor library only, as a provider for coverage on every backend ggml supports and as the correctness reference. ggml never decides residency. Patches are upstream-first, limited to backend hooks: stream and queue injection, buffer import, capture-safe launch. RFC-0040 sets the patch budget and cadence.
- ggml backends differ in interop (`backends-and-portability.md` §1.3.1): CPU and Metal advertise wrapping external host memory; Vulkan and SYCL register the function without advertising it; CUDA and HIP do not offer it.
- ggml's GPU backends can be built as dynamically loaded modules that link a shared `ggml-base`; its CPU backend can be built in all ISA variants and picked at load (§5.2). iOS forbids loading downloaded code (§5.4).
- ADR-0507 and RFC-0052 package each backend's native and ggml providers together in one provider library per module, and put the ggml CPU provider in the core.
- ggml is pre-1.0, with frequent releases, so its op API moves.
- Syntropy's executor needs whole-step capture and one allocator per device (RFC-0040 §2.3, §4.4).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Hand whole stages to ggml's scheduler | Least integration code | Rejected by ADR-0002: per-tensor splits, its own buffers, no residency control |
| B. Wrap single ops as ggml graphs over Syntropy buffers, prebuilt at `prepare` | Syntropy keeps scheduling, memory and capture; ggml supplies kernels | Per-op mapping code to maintain; some ops need an extra gather (paged KV) |
| C. Call ggml backend kernels directly, bypassing graphs | No graph overhead | Internal APIs, not part of ggml's interface; breaks at every bump |
| Link mode B1: ggml's own dynamic backend loader (shared `ggml-base`, per-backend modules) | Matches how ggml ships; CPU variants picked at load | A second module system beside ADR-0507's; a shared `ggml-base` the core and every module must agree on |
| **Link mode B2: a private static copy of ggml-base per artefact, with only that artefact's backend: CPU (and Metal on macOS) in the core, the module's GPU backend in each provider library; all symbols hidden** | Fits ADR-0507's one-library-per-module packaging; no shared ggml state between artefacts; iOS- and Android-compatible | Two or three ggml copies per machine; the core's ggml CPU is built for the baseline ISA, since CPU variants need ggml's dynamic loader |

## Decision

**Option B with link mode B2.**

1. `prepare` builds a template ggml graph per (`OpKey`, `ShapeSig`) and keeps it. `query` returns `Some` only when the device's ggml backend supports every node. `launch` binds tensor data pointers to `OpArgs` buffers and computes asynchronously.
2. Memory: CPU and Metal wrap Syntropy memory. Vulkan and SYCL wrap it only after an import probe at `open` passes. *Revised 2026-10-20:* the M2 Vulkan Tier 2 preview, which has no native provider, skips the probe and allocates every buffer through ggml's buffer type, recorded by the tier manager, with no elastic arenas (RFC-0040 §7.5). CUDA and HIP wrap it through local patch P1. Where none of these works, the tier manager allocates through ggml's buffer type and records and owns the allocation. ggml ops use source (GGUF) layouts, exact transcriptions of safetensors tensors to ggml types, or `widen.f32` copies (RFC-0042 §3.3, §3.5); ggml's own repacked CPU layouts are not used.
3. Streams: patches P2 (caller-provided CUDA/HIP stream), P3 (workspace-backed temporary pool) and P4 (caller-provided Metal command buffer) make ggml ops capture-safe on Syntropy's queue. Without them, each ggml op is fenced with events and breaks capture, and selection charges for that. ggml's CUDA and HIP backends link cuBLAS and hipBLAS/rocBLAS, which the `rocm` module ships and which `setup` fetches from NVIDIA for the CUDA modules (ADR-0400 decision 3, revised 2026-10-20); ggml ops that may route to them (float weights, prefill and batch buckets) are marked not capture-safe, because those libraries create handles and workspaces outside P3.
4. Patch budget: at most 6 local patches and 1000 changed lines. Each patch is submitted upstream before it is carried.
5. Pin bumps: once per Syntropy minor release, at most every 4 weeks, as a tracked task: rebase, regenerate bindings, full op conformance and coverage matrix. A bump changes every ggml op's kernel hash, so it re-issues every ggml `OpCert` (ADR-0602).
6. Each artefact links its own static ggml copy with its backends and hides every ggml symbol (RFC-0040 §10.4). The core's ggml CPU backend is built for the baseline ISA (x86-64-v2, ARMv8.0) with floating-point contraction off; it serves rare-op fallback and the calibration baseline (ADR-0422), and speed on CPUs comes from the native provider.
7. The runtime parity spot check uses `kernels::reference`, so ggml is never required for correctness at runtime; CI uses it as the op-level reference.

## Consequences

- Positive: every ggml backend is reachable from M0, through the same executor, allocator and capture model as native providers.
- Positive: upstream improvements to a ggml kernel reach Syntropy at the next bump, and selection adopts them only where measurably faster.
- Negative: mapping code per op kind, and per-bump maintenance.
- Negative: on backends without P1–P4 accepted, ggml ops cost a fence and a capture break each.
- Negative: paged KV costs an extra gather on ggml attention.

## What would make us revisit

- The patch queue exceeds the budget for two consecutive bumps, or upstream rejects buffer import or stream injection on a Tier 1 backend (ADR-0002's revisit clause: an explicit fork, kept as a standing stream of tasks).
- ggml adopts a stable C API for external memory and streams. Drop the patches.
- ggml gains native paged attention. Remove the gather.

## Log

- Revised 2026-10-19: decisions 2, 3 and 6, to account for the vendor BLAS libraries ggml's GPU backends link, to give non-GGUF schemes a baseline ggml can run, and to build ggml's CPU backend with contraction off.
- Revised 2026-10-20: decision 2 adds the M2 Vulkan preview's ggml-owned allocations, since that module has no native provider to own memory; decision 3 notes that NVIDIA's BLAS libraries are fetched at `setup`, not shipped.
