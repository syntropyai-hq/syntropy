# ADR-0002: Own the runtime, with kernel providers underneath

- Status: Accepted (revised 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D1, D3, D7), ADR-0001, ADR-0003, ADR-0004, ADR-0006, ADR-0007, ADR-0012, ADR-0019, RFC-0001, RFC-0012, RFC-0020, RFC-0023, RFC-0040, RFC-0042, RFC-0043, RFC-0060

## Context

Syntropy's speed comes from decisions made inside the forward pass. A runtime we do not own cannot express them:

- **A per-token, per-expert split of one MoE op across three execution sites.** VRAM-cache hits run on the GPU. A measured share of RAM-resident misses is copied over PCIe and computed on the GPU, and the rest is computed in place by the CPU. Routing reaches the host through a polled pinned-memory doorbell, and the whole pass replays as one captured CUDA graph per verify-window size (`plan/research/tiered-memory-inference.md` §5.1, §5.4).
- **Per-expert residency** across VRAM, pinned RAM and NVMe, with per-layer byte-sized cache slots and decayed-LFU admission (§5.2, §5.3).
- **Stage-range execution across nodes**: run layers `[a, b)`, take an activation frame in, emit one out, and hold only the shard (ADR-0010).
- **A custom tier I/O path**: direct reads into pinned registered buffers, not the OS page cache (ADR-0007).
- **Owned buffers**: every allocation comes from the tier manager, so placement decisions stay true.

What the candidate substrates offer (`plan/research/backends-and-portability.md`):

- **ggml, the tensor library,** covers the whole hardware brief: CUDA, HIP, Metal, Vulkan, SYCL, OpenCL (Adreno), Hexagon, CANN, MUSA, WebGPU, CPU variants. It has every GGUF quant type including MXFP4 and NVFP4, per-device buffer types, pinned host buffers, async set/get/copy and events (§0 item 1, §1.2, §1.3.1). It is MIT-licensed.
- **ggml's companion model library** adds graph builders for about 160 architecture families (§0 item 2). Its scheduler splits graphs by tensor, not by expert subset per token. Its expert cache is an LRU with no prefetch or predictor. It has no NVMe tier for arbitrary weights, no global placement optimiser and no cross-node scheduler beyond RPC, and its C API changed five times in one release (§1.3.1, §7.1).
- External buffer import (`buffer_from_host_ptr`) is advertised on Metal and CPU. On Vulkan and SYCL the function is registered but not advertised, so it must be tested before use; CUDA does not offer it (§1.3.1).
- **No engine is uniformly fastest.** Published cross-engine gaps on NVIDIA are within about ±25%, and Apple results trade places by model and context. For tiering, hit rate and I/O scheduling dominate: the expert cache alone gave 2.2× decode (§1.4).
- **New attention types land first on CUDA** and reach Vulkan, Metal, SYCL and WebGPU later or never (§7.2 risk 1). A runtime that targets every device needs a per-op fallback.

## Options considered

| Option | Performance ceiling | Coverage | Cost and risk |
|---|---|---|---|
| A. Bind ggml plus its model library; push hooks upstream | Capped by its scheduler, KV cache and buffer model; the three-way expert split and doorbell capture need deep changes in its most-churned code | Best on day one (§4.1: graphs for every 2026 architecture surveyed) | Recommended by the research (§7.1 stack A). Rejected: the cap is structural |
| B. Fork ggml and its model library | High | Starts high, decays with drift; a surveyed fork's model list froze at its fork point (§7.1 B) | Long-lived fork of a project with several releases a day |
| **C. Own the runtime; kernels through a `KernelProvider` trait with native and ggml providers** | High: every scheduling, memory and dispatch decision is ours; kernels are chosen per op | Full backend coverage through ggml from the start; model graphs are ours (ADR-0003) | Graph and kernel maintenance; ggml op-API churn |
| D. Own every kernel on every backend | Highest in theory | Poor: ops × dtypes × backends is too large a matrix | Years of kernel work before broad hardware runs |
| E. Build on a Rust tensor framework | Unknown | One surveyed framework has no ROCm or Vulkan; another has the backends but no LLM kernels for i-quants, MXFP4 MoE or linear attention (§7.1 C) | Weeks to months of kernel work per family (estimate, §7.1) |
| F. MLX as the substrate | Strong on Apple | Metal and Linux CUDA only; no per-array placement | See ADR-0004 |

## Decision

**Option C.**

1. **Syntropy owns** the model graph IR and the per-family graph definitions (ADR-0003), the executor and step loop including whole-step capture into CUDA graphs and Metal or Vulkan command buffers, the scheduler (RFC-0012), memory pools, pinned arenas, residency and the expert cache (ADR-0007, RFC-0020, RFC-0023), the KV store (RFC-0022), the I/O engines (RFC-0021) and stage input and output (ADR-0010).
2. **Kernels come through a `KernelProvider` trait** (RFC-0001 defines the signature, RFC-0040 the contract). A provider:
   - reports, per device, the (op, dtype, shape class) triples it supports;
   - executes on buffers it is handed as `TensorHandle`s and never allocates model memory on its own initiative;
   - records into the device's stream or queue owned by that device's dispatcher thread (ADR-0012), so whole-step capture works.
3. **Native providers**, owned by Syntropy, cover the hot ops on Tier 1 backends: CUDA, Metal and CPU first, then HIP and Vulkan. The hot ops are quantized GEMV and GEMM; the MoE grouped expert kernel with residency-aware dispatch (hit, PCIe-share and in-place-miss groups); attention (GQA, MLA, sliding-window, sparse or indexed); and hybrid linear-attention scans.
4. **A ggml provider** links ggml, the tensor library only, never its model library. It is vendored at a pinned commit and gives full op coverage on every backend ggml supports, including SYCL, CANN, OpenCL/Adreno, Hexagon and WebGPU. It wraps Syntropy memory through buffer import where the backend supports it. Where a backend cannot import, the ggml provider allocates through a ggml buffer type at the tier manager's request, and the tier manager records the allocation and owns its lifetime. ggml never decides what is resident.
5. **Op selection per device**: filter providers by availability, then pick by startup microbenchmark at the model's real shapes, separately for decode (GEMV) and prefill (GEMM) shape classes. Results are cached per hardware fingerprint, driver version and Syntropy version, and they feed the capability profile (RFC-0010). A native kernel must pass parity before it can be selected. Ties go to the native provider.
6. **The ggml provider is the correctness reference.** Every native kernel is tested against the ggml op on the same inputs and dtype within the per-dtype tolerances of ADR-0019. Where the device's ggml backend lacks the op, the ggml CPU op is the reference.
7. **Missing ops are a capability, not a crash.** If no provider on a device supports an op, the solver sees that device as unable to run the op. It either places the op's compute site elsewhere and costs the hand-off, or keeps the stage off that device (ADR-0008).
8. **ggml patches are upstream-first** and limited to backend-level hooks: external stream and queue injection, buffer import, capture-safe launch. Pin-bump cadence and the patch budget are set in RFC-0040.

## Consequences

Positive:
- No external scheduler, KV cache or allocator limits placement, the expert split or capture.
- Every backend ggml supports works from the start through one provider, so "installs on any supported device" (D7) does not wait for native kernels.
- Two kernel implementations per hot op give a built-in differential test.

Negative:
- Syntropy writes and maintains every model graph (ADR-0003); a new family needs Syntropy work before it runs.
- New attention types need native kernels per backend; until then they run on ggml ops, or on CPU fallback with a placement cost.
- ggml is pre-1.0, so its op API moves; every pin bump is a tracked task.

Work created: RFC-0040 (trait, provider contract, selection, patch policy), RFC-0043 (IR the providers execute), RFC-0012 (executor and capture), RFC-0042 and RFC-0060 (parity harness against the ggml reference), RFC-0010 (per-op microbenchmarks in the capability profile).

## What would make us revisit

- On a Tier 1 platform after M1, for a dense model that fits in VRAM, Syntropy decodes more than 10% slower than a ggml-based engine on the same file and backend. Find the cause in our executor before writing more native kernels. The single-device baseline comparison on R-ADA that gates the M1a exit (RFC-0001 §16) is the first measurement of this trigger.
- Upstream ggml rejects buffer import or stream injection on a Tier 1 backend, and the local patch queue exceeds the RFC-0040 budget for two consecutive pin bumps. Move to an explicit ggml fork, kept as a standing stream of tasks: rebases, parity against the upstream ops and the local patch set, each through the same gates as other changes, with agent and CI capacity set aside for it (ADR-0021).
- Owned graph definitions lag a Tier 1 family's public release by more than one Syntropy release twice in a row (see ADR-0003).
- **Reviewed (2026-10-19); decision stands.** The research recommended binding ggml's model library for coverage (`backends-and-portability.md` §7.1 stack A, §7.3 question 1). It also observed that one surveyed desktop runner dropped its own engine and now wraps the upstream server, apparently under model-release velocity (§2.1). This ADR keeps owned model graphs (ADR-0003). The coverage triggers above and in ADR-0003 are how we would notice the research being right.

## Revisions

- **Revised 2026-10-20:** the first revisit trigger now has a fixed first measurement, so a slow executor is found before more native kernels are written: the gating single-device baseline comparison on R-ADA under ADR-0612 at the M1a exit, with an early read at the M0b exit. It is a release gate from the first M1a release candidate. The provider split also sets the second step of RFC-0001 §16's scope-cut order: if the native HIP and Vulkan providers are cut, the ggml provider covers those backends, starting from M2's ggml-only `vulkan` module. The decision itself is unchanged.
