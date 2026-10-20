# ADR-0102: Place experts as per-layer byte budgets over a learned hit-rate curve

- Status: Accepted (revised 2026-10-19)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the placement and scheduling area)
- Related: ADR-0007, ADR-0008, ADR-0006 item 8; RFC-0011 §3.2, §6 C8; RFC-0023

## Context

MoE models dominate the target list: DeepSeek-V3 has 61 × 256 routed experts, a Kimi K3-class model 93 × 896 (`backends-and-portability.md` §4.1). The solver has to decide how much of each layer's routed pool sits in VRAM, in pinned RAM and on NVMe, and how RAM-resident misses split between in-place CPU compute and PCIe copies.

ADR-0008 says "experts bucketed by profiled popularity, with group bytes per tier on the owner device and hit rates from a piecewise-linear fit of the learned h(c) curve". Facts that bear on how literally to take "buckets":
- Hit rates depend on the runtime cache policy as much as on capacity. A frequency-only baseline scored ≈0.35, and temporal-locality-aware policies passed 0.5; decayed LFU with hysteresis raised one engine from 0.50 to 0.72 (cost model §3.3; `tiered-memory-inference.md` §5.3). Which experts are resident changes every few rounds.
- Per-layer, byte-sized slots matter: one shared counter across layers gave a 2.97% hit rate against 70.4% with per-layer slots (`tiered-memory-inference.md` §5.2).
- The quantity the cost model needs is the fraction of routed reads served from each tier, `h(c)`, which is concave in resident bytes c and has a uniform prior `h = c / pool` (cost model §3.3).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Per-expert binary placement | exact control | 15,616 binaries for DeepSeek-V3, 83,328 for Kimi K3-class; decides what the runtime cache re-decides within seconds |
| B. Popularity buckets as integer groups per tier | fewer variables | bucket boundaries fixed by a static profile, not by the policy the runtime runs; integer choices for what is a continuous budget |
| **C. Continuous byte budgets per (layer block, tier) with the achieved hit curve of the runtime policy as a concave piecewise-linear function, homogenised bounds in the MILP, and the three sites (GPU, CPU, NVMe) as an epigraph max** | no binaries for experts; the curve is what RFC-0023 measures; the miss split comes out of the same LP; the segments are exactly ADR-0008's buckets | the solver never names individual experts; the curve must be learned before the plan is accurate (the uniform prior is pessimistic) |

## Decision

**Option C.**
- Per MoE block b and placement node p the MILP has continuous variables for VRAM cache bytes `y`, RAM-resident bytes `u`, the read fractions `z^V ≤ H_b(y)` and `z^R ≤ H_b(u + y)` as homogenised piecewise-linear upper bounds (8 segments by default). For NVMe-homed experts the VRAM and RAM caches are exclusive sets (a VRAM victim moves down to RAM, a promotion frees its RAM slot; RFC-0023 §3), so RAM is charged `u` and fast memory covers `u + y`; RAM-homed blocks (`u = Π`) keep the whole pool in RAM with VRAM copies. RAM-resident experts split into a DMA-pinned share `u^P ≤ u` in `HostPinned` (copy sources, with `z^P ≤ H(u^P + y)` bounding what can be copied) and a pageable rest in `HostPageable` computed in place, under RFC-0020's joint host rows. The model also has the PCIe and CPU shares of RAM-served misses, and a block time `τ ≥` each of the GPU, CPU and NVMe sites.
- Per-layer union curves come in two shapes from RFC-0023 §13: `window` for one sequence's verify window and `batch` for a throughput micro-batch; the solver uses each where its pass shape applies.
- A cache tier whose measured hit rate falls short of its curve past RFC-0010's drift thresholds is workload drift: a delta solve caps that tier's curve at the measured value.
- `H_b` is the slope-sorted merge of the block's per-layer curves, which is the optimal split of a block budget across its layers. The plan's per-layer budgets come from the same order.
- `H` is the achieved hit fraction of RFC-0023's cache policy at each capacity, learned online and persisted in the pack sidecar. Until learned, the uniform prior is used and listed as an assumption.
- A learned curve that differs from the one a plan used triggers a delta solve (budgets only, applied in place), not a re-shard.

## Consequences

- Positive: expert placement adds no binaries, so MoE models solve as fast as dense ones of the same block count. The first plan is safe (the prior underestimates hits), and the plan improves in place as statistics arrive.
- Negative: the solver depends on RFC-0023 publishing a monotone concave curve per layer; a non-concave measured curve must be replaced by its concave envelope, which overstates hits between breakpoints. Placement of hot experts on a specific GPU of a multi-GPU node is left to the runtime.
- Work created: RFC-0023 publishes `HitCurve` per layer from replayed routing traces at the cache's real policy; RFC-0060 tests that the concave envelope error is reported.

## What would make us revisit

- Simulation shows budget-level placement losing more than 10% decode against per-expert assignment on recorded traces (ADR-0008's criterion).
- Measured curves are routinely non-concave by more than 5 points of hit rate, for example because of cross-layer correlation the per-layer model misses.
- Cross-node EP becomes common enough that which node owns which expert matters to the critical path; then per-owner expert groups enter the MILP for EP islands only.
