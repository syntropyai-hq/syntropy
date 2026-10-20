# ADR-0201: VRAM arenas as reserved ranges with elastic, striped expert segments

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0020, RFC-0023))
- Related: ADR-0007 item 10, ADR-0012, ADR-0014; RFC-0001 §5.5, §9.3; RFC-0020 §8; RFC-0023 §4, §5.4; RFC-0040

## Context

- Captured step graphs (CUDA graphs, Metal indirect command buffers) bake in device addresses. RFC-0001 §5.5 requires expert-cache slots to keep fixed addresses for the life of a plan epoch, so the cache can change contents without re-capture.
- On a personal machine, other programs take VRAM: a game, a video call, a browser. Past the OS budget on Windows, the driver may silently move a process's VRAM to system memory, and step times collapse with no error. Syntropy must be able to give VRAM back (ADR-0007 item 10).
- The surveyed single-GPU engine reserved one virtual range for its expert cache, backed by 512 MiB physical segments allocated with CUDA's virtual memory management API, and unmapped tail segments to lend VRAM to another program, keeping slot addresses fixed (`tiered-memory-inference.md` §10.5).
- That design laid slots out contiguously. If each layer's slots are contiguous, unmapping the tail empties the last layers' caches entirely, and per-layer hit rates were the single most important property of the cache (2.97% vs 70.4% with per-layer slots, §5.2).
- ADR-0014 lists CUDA VMM with a fixed-slab fallback. HIP has an equivalent API on supported devices; Vulkan has none, but separately allocated `VkDeviceMemory` blocks addressed by buffer device address serve the same purpose.

## Options considered

| Option | For | Against |
|---|---|---|
| A. One fixed allocation per purpose | Simple; works everywhere | Cannot return VRAM without a reload; fails the yield requirement |
| B. Re-allocate and re-capture on resize | Exact sizes | Moves addresses; re-capture per change; needs a pause |
| C. Reserved range, segments mapped at the tail, slots laid out per layer | Fixed addresses; returns memory | Shrinking empties whole layers |
| **D. Reserved range per purpose; only the expert arenas elastic; slots striped across segments so every segment holds a proportional share of every layer's slots** | Fixed addresses; returns memory; a shrink lowers every layer's capacity evenly | A slot descriptor table instead of a closed-form address; up to one slot of waste per segment |

## Decision

**Option D.**

1. Each discrete GPU gets one reserved virtual range per purpose (`Scratch`, `Graphs`, `Weights`, `Kv`, `ExpertStaging`, `ExpertCache`), mapped in the RFC-0020 §4.4 order. Only `ExpertCache` is elastic; `ExpertStaging` is small and fixed per epoch, so the copied share is always bounded by its mapped slots.
2. Segment size is the larger of `tier.vram_segment` (default 256 MiB) and 8 × the largest slot, rounded to the allocation granularity. Waste is at most one slot per segment, ≤ 12.5%.
3. Slots are striped: segment k holds the largest-remainder share `n_l / S` of every layer's slots, packed largest first, no slot crossing a segment. A slot descriptor table (slot → segment, offset, layer) replaces closed-form addressing.
4. Shrink: first move each layer's hot residents out of the tail segments into the coldest surviving slots with in-tier copies charged to the admission budget, so the tail holds the coldest experts; then mark the tail segments' slots `Evicting`, publish `MISS` for them through the table-delta ring, wait for the step fence, unmap between steps, and report the smaller ledger grant. Grow maps new tail segments and lets the policy fill them.
5. Providers expose `map_segment` and `unmap_segment`, with default bodies returning `Unsupported` (RFC-0040). Without them the arena is a fixed slab, a shrink retires slots logically and returns no memory until reload, and the runner reports that honestly.
6. The KV arena is fixed per epoch in v1 (RFC-0020 §8.6).

## Consequences

Positive:
- VRAM can be returned to the user within a few steps without re-capturing graphs or reloading.
- A shrink degrades every layer's hit rate evenly instead of zeroing some layers.
- The same mechanism serves `PlanDelta` budget changes and the pressure ladder.

Negative:
- One more indirection (the descriptor table) when building `slot_addr`, at warm-up only.
- Segment map and unmap are driver calls; they happen between steps and never per token, but a shrink takes several steps.
- Vulkan and non-VMM devices get a weaker form of elasticity.

## What would make us revisit

- Unmapping a segment on a Tier 1 driver turns out to synchronise the device or stall the step loop by more than one step time. Then shrink only when no sequence is live.
- A KV-heavy workload shows plan deltas regularly wanting to grow the KV hot window at the expert cache's expense. Then make the `Kv` arena elastic too (RFC-0020 Open question 6).
- Measured striping waste above 5% on a v1 family. Then pack mixed slot sizes per segment more tightly or adapt segment sizes per model.
