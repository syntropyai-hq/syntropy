# ADR-0207: Attention over spilled KV: the plan picks CPU-in-place or GPU streaming per attention kind, and splits merge exactly

- Status: Accepted (2026-10-20; revised 2026-10-19)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0022))
- Related: ADR-0006, ADR-0012, ADR-0019; RFC-0001 §6.3, §8.2, §9.3; RFC-0020 §10; RFC-0022 §5; RFC-0040

## Context

- KV competes with the expert cache for VRAM (RFC-0001 §9.3). Keeping only a hot window of KV in VRAM and spilling the rest to pinned RAM frees VRAM for experts; in the surveyed engine, doing so at 262K context let VRAM hold 3,872 experts instead of 1,589 and raised decode from 50.9 to 62.6 tok/s (`tiered-memory-inference.md` §12.1). That engine's model used sparse attention, so only a few thousand selected positions were read per step.
- For dense attention every decode step reads every page. Spilled pages must then be read every step, either by copying them over PCIe to the GPU or by computing that part of the attention on the CPU where they sit.
- The weights already face the same choice, and the rule is execute-in-place at decode, stream above n* (ADR-0006): dual-channel DDR5 at ≈70 GB/s beats pinned PCIe 4.0 x16 at ≈24 GB/s, so reading bytes once where they are wins at batch 1 (`hardware-and-performance-model.md` §3.2).
- Attention is not like a weight GEMV, because every stored KV element is read by all the query heads that share it. The CPU does `2 · g_eff` FLOP per KV byte-pair per query row, with `g_eff = n_q_heads / n_kv_heads` for GQA and ≈ 1.9 · n_heads for absorbed MLA. So the break-even is `T*_kv = b_kv · F_cpu / (2 · g_eff · W_pcie)` query rows, not the weights' n*. At F_cpu = 1 TFLOPS, BF16 and 25 GB/s it is ≈5 rows for Llama-70B (g = 8) and ≈0.16 for DeepSeek-V3's MLA (g_eff ≈ 243) [estimates; RFC-0022 §5.3].
- Attention over a split set of keys combines exactly: each part returns its unnormalised output, maximum score and softmax denominator, and a log-sum-exp rescale merges them. Blocked attention kernels already do this between blocks.

Worked cases [estimates]:
- Llama-3.3-70B at 32K context with 2 GiB of KV in VRAM and 8 GiB spilled, decode: streaming costs ≈358 ms per step at 24 GB/s; computing in place costs ≈123 ms at 70 GB/s (memory-bound), overlapped with ≈2.7 ms on the GPU; all in VRAM ≈13 ms.
- DeepSeek-V3 at 32K with its 2.25 GB of MLA KV spilled, decode: computing in place is compute-bound at ≈242 FLOP per byte, ≈0.28–0.56 s at 1–2 TFLOPS; streaming ≈94 ms; in VRAM ≈3 ms.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Never spill KV during decode | Fastest attention | Long contexts take expert capacity, which the solver may value more (the surveyed 62.6 vs 50.9 tok/s) |
| B. Always stream spilled pages to the GPU | One compute site | ≈3× slower than the CPU for GQA at decode (worked case) |
| C. Always split: CPU over host pages, exact merge (this ADR's first draft) | Reads each byte once | 3–6× slower than streaming for MLA, the flagship M1 family, because the CPU is compute-bound (worked case) |
| **E. The plan picks per (stage, class): `CpuSplit { max_rows }` when `T < T*_kv`, else `GpuStream`, from measured `F_cpu,attn` per attention kind; the runtime refines the crossover online; splits merge exactly, with sinks** | Each kind and pass size gets its cheaper path; the solver predicts the path that runs | Two execution paths and a probe per attention kind; the plan carries one more field |
| D. Approximate attention over spilled pages (attend only to selected positions) | Fast | Changes output; `approx_attention` is opt-in only (ADR-0019 §6) |

## Decision

**Option E** (supersedes this ADR's first draft, Option C).

1. `KvBudget` carries `spill_compute: SpillCompute` per (stage, class): `CpuSplit { max_rows }` or `GpuStream`. The solver chooses it from the terms in RFC-0022 §5.3–§5.4, with `F_cpu,attn` per attention kind probed by RFC-0010 at T = 1 and at the plan's verify window. Defaults when unmeasured: `GpuStream` for MLA and any kind with `g_eff ≥ 32`; `CpuSplit { max_rows: T*_kv }` for GQA and MHA.
2. `CpuSplit`: the GPU attends over VRAM pages and the CPU pool of the pages' NUMA domain over host pages; each returns F32 `(o, m, l)`; the GPU merges with the log-sum-exp rescale, including a per-head sink logit for families with attention sinks (`m = max(m_1, m_2, s)`, the sink adds `e^(s−m)` to the denominator once). An empty side returns `o = 0`, `m = −∞`, `l = 0` and always raises its flag.
3. The query and partials travel through a `GPU_MAPPED` host `Scratch` sub-arena and a per-layer attention doorbell beside the MoE doorbell; no host synchronisation (RFC-0001 invariant 9).
4. `GpuStream`: host pages are copied into double-buffered VRAM stream buffers on the copy stream and attended on the GPU. Prefill chunks always stream.
5. Sparse and indexed layers keep a frequency-ordered hot set of pages in VRAM; selected host pages follow the class's `spill_compute`.
6. RFC-0040 adds `AttnPartial(AttnKind)` and `AttnMerge` (with a sink operand) to its op list, each with a reference function and parity tests. Both paths are exact-mode paths and must pass the parity harness at every split ratio.

## Consequences

Positive:
- The solver and the runtime use the same cost terms, so plans predict the path that runs.
- GQA at decode keeps the CPU's ≈3× advantage; MLA streams, which is 3–6× better than the CPU would be.
- Spill is costed honestly: cheap for sparse attention and linear-attention hybrids, expensive for dense attention of every kind, especially dense MLA, which the solver will normally keep hot.

Negative:
- Two execution paths and the probe of `F_cpu,attn` per kind.
- The CPU pool serves both in-place experts and in-place attention under `CpuSplit`, and they compete for DRAM bandwidth; the solver must count both.
- Every attention kernel family needs an `AttnPartial` variant on the CPU provider.

## What would make us revisit

- The M1 probe of `F_cpu,attn` shows GQA decode on the CPU falling short of DRAM bandwidth (RFC-0022 Open question 2): then `GpuStream` becomes the default for every kind.
- PCIe 5.0 x16 pinned rates (estimated 48–52 GB/s, not measured) approach a machine's CPU DRAM rate: the probe's measured terms then decide per machine, as for weights.
- The parity harness shows the merge's rounding outside tolerance for some dtype: compute the merge in F32 or narrow the split to page boundaries that the reference also uses.
