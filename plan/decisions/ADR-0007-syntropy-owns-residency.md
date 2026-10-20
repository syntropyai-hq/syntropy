# ADR-0007: Syntropy owns residency

- Status: Accepted
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D1, D7), ADR-0002, ADR-0004, ADR-0005, ADR-0006, ADR-0008, ADR-0012, ADR-0013, ADR-0015, RFC-0010, RFC-0011, RFC-0020, RFC-0021, RFC-0022, RFC-0023, RFC-0054

## Context

ADR-0000 D1 places weights across VRAM, pinned RAM and NVMe with an adaptive expert cache. Someone has to decide which bytes are resident in which tier at each moment. The candidates are the operating system (mmap and the page cache) or Syntropy.

**What the OS page cache does as a residency manager:**
- File-backed pages are dropped first under pressure. Mapping a 34 GB expert file and a 26.8 GB table on a 63 GB machine made expert reads fault back in from disk. On Windows the CPU read mapped memory at **≈19 GB/s against 42.8 GB/s** from an anonymous arena, a 2.2× gap (`plan/research/tiered-memory-inference.md` §10.1).
- Row lookups through mmap on SATA ran at 2.1K rows/s against 14.4K with direct I/O (§10.4).
- In a multi-device study that leaves CPU-resident layers to the page cache, later prefetches evict earlier ones. A 16 GiB laptop reloaded about 25 GiB per token, the paper had to model four OS-specific memory regimes, and Metal could reload the whole model repeatedly (`multi-device-inference.md` §4.2, §8.1, §13 item 8).
- It has real advantages: reclaimable memory kept memory pressure under 6% on every device, against over 50% for runtimes that hold weights (§8.1). On Apple's 17.5 GB/s SSD, one engine's custom caches, prefetching and speculative routing all lost to the OS page cache (`tiered-memory-inference.md` §8).

**What explicit residency costs** (`tiered-memory-inference.md` §10.2):
- Windows cannot lock one huge range and allows locking only about half of RAM. WDDM caps helper-GPU host registration at 8 GiB.
- WSL2 caps `cudaHostAlloc` at about 1 GiB, so ordinary memory has to be registered with `cudaHostRegister` instead.
- On Linux, one giant `cudaHostAlloc` got the process OOM-killed despite high MemAvailable, because clean file cache cannot be reclaimed as fast as the driver takes pages. Registration has to be paced in steps against available memory, cgroup limits and per-NUMA memory. The surveyed engine paces against MemFree; RFC-0020 §6 does not, for the reason in item 3.
- On macOS an unwired arena caused three kernel panics through the memory compressor. The fix was wiring with `mlock` and capping the footprint at 60% of RAM.
- Loading and locking 35–55 GB froze the machine for 1–3 minutes on first start.

**Why pinning matters**: pageable host memory reaches about 6 GB/s H2D against about 24 GB/s pinned on PCIe 4.0 x16, because the driver bounces through a staging buffer (`hardware-and-performance-model.md` §2.3).

**What NVMe can do**: it is a capacity tier, not a bandwidth tier. Expert-sized random reads ran at an effective 1.5–2.6 GB/s on a Gen4 drive rated 7.45 GB/s. Every GB of dense overflow costs ≈170 ms per token at 6 GB/s, and Gen5 drives throttle under sustained load (`tiered-memory-inference.md` §11; `hardware-and-performance-model.md` §2.4, §3.3). One issuing thread caps at about 80K reads/s (`tiered-memory-inference.md` §10.3).

**What a good cache needs** (`tiered-memory-inference.md` §5.2, §5.3, §16):
- per-layer, byte-sized slots: one shared admission counter produced a 2.97% hit rate against 70.4% per layer;
- decayed LFU with hysteresis: hit rate rose from 0.50 to 0.72;
- cache replacements off the critical path;
- a single owner per expert: duplicate residency across GPUs wasted about 2,000 slots, and removing it gave +15.5%.

The OS cannot provide any of these.

## Options considered

| Option | For | Against |
|---|---|---|
| A. mmap everything; hint with `madvise`/`fadvise` | Zero code; low memory pressure; best on Apple's SSD in one engine | Residency follows kernel LRU; 2.2× slower CPU reads on Windows; prefetch-release thrash; no expert-level policy; behaviour differs per OS |
| B. Load everything into pinned RAM; no NVMe tier | Simple and fast when the model fits | Fails D1 for models above RAM; pinning limits cap it at about half of RAM on Windows |
| **C. An explicit tier manager owns every byte; pinned anonymous RAM tier; direct-I/O NVMe tier; mmap only where it measures better** | Predictable bandwidth; expert-level caching; the solver's plan is what actually happens | Per-OS pinning code; Syntropy must yield memory deliberately |

## Decision

**Option C.**

1. **One tier manager per runner owns every byte** the runner uses in every tier (`Vram`, `Unified`, `HostPinned`, `HostPageable`, `Nvme`): weights, expert cache slots, KV, staging rings and scratch. Arenas are created during `Loading` (ADR-0015). The token path never allocates.
2. **The residency map** is the live, versioned per-node record of which tier holds each tensor, expert and KV block. Every `TensorHandle` resolves through it. Kernel providers execute on handles and never allocate model memory (ADR-0002).
3. **The hot RAM tiers are anonymous memory**, never file-backed (revised 2026-10-19, ADR-0200). Bytes a device copies from live in `HostPinned`, which is locked and registered for DMA. Bytes only the CPU computes live in `HostPageable`, locked against paging when the budget allows and never DMA-registered. Locking and registration respect a per-OS budget computed from probes and policy (RFC-0020, RFC-0054):
   - Linux: committed in steps (RFC-0020 §6). There is no free-page (MemFree) test at any point, because clean page cache keeps MemFree near the zone watermarks on a normal host and a wait on it can stall forever.
     - Each step is gated on MemAvailable minus the reserve, the cgroup headroom (`memory.max − memory.current`, and `memory.high`) and the per-NUMA available estimate, all of which count reclaimable cache.
     - The commit itself (`MADV_POPULATE_WRITE`, or touching each page on older kernels) forces reclaim through the normal allocator at its own pace. Registration then pins only pages that are already resident and allocates nothing, so the driver never takes pages faster than reclaim delivers them.
     - Distress is read from PSI `some avg10` in `/proc/pressure/memory`, or, on kernels without PSI, from the `allocstall_*` direct-reclaim counters in `/proc/vmstat`, together with each step's own latency. Under distress the loop backs off, and slow steps halve the step size.
     - Pinning gives up only after sustained pressure: when distress, or a capacity shortfall, persists for `tier.pin_give_up`, the remaining ranges stay `Underpinned` and are costed as `HostPageable` (item 4).
   - Windows: per-range locks inside the OS lock limit, never one huge range; WSL2 registers ordinary memory instead of using `cudaHostAlloc`.
   - macOS: the arena is wired. The default footprint cap is the lower of the GPU working-set limit the OS reports for the device and RAM minus the RFC-0020 reserve. A fixed percentage is not used, because it would block plans the hardware supports: DeepSeek-V3-class 4-bit weights need about 74% of a 512 GB machine (cost-model scenario b1).
     - Raising the OS limit is an explicit user opt-in. `syntropy tune` explains the change, needs administrator approval, records the change, and reverts it on uninstall (RFC-0052). Syntropy never raises it silently.
     - When a plan is infeasible only because of this limit, the placement preview says so and offers the opt-in.
     - The panic history that motivated wiring (Context) is why the arena stays wired and the RFC-0020 reserve is always enforced.
4. **Pinning is progressive.** Ranges are pinned in the order the plan ranks them, in bounded steps, so the machine stays responsive. A runner may reach `Ready` before locking completes; unpinned ranges are costed as `HostPageable` until they are pinned. The GPU never DMAs from pageable memory directly.
   - *Note, 2026-10-19 (RFC-0020 §6).* DMA registration is narrower: every `DmaPinned` registration completes before the runner reports `Ready`, and before `Warming` captures step graphs, until RFC-0020 open question 9 shows that registering during captured decode leaves step time unaffected. Locking that involves no GPU driver may continue after `Ready`.
5. **NVMe is a capacity tier read with direct I/O** (ADR-0013) into pinned staging or straight into cache slots, one whole read unit from the pack at a time (ADR-0005). The cost model uses the sustained, throttled rate at the real request size.
6. **mmap is allowed in two cases only:** cold, read-once data, and as a per-machine fallback for a tensor class where a measured A/B (startup or online) shows the OS path faster, as on some Apple SSDs. Even then the bytes are accounted in the residency map and costed at their measured rate. **The OS page cache never decides residency** for anything the plan places in `HostPageable`, `HostPinned` or faster, apart from the mmap cases above.
7. **The expert cache** is per device, with per-layer byte-sized slots, decayed-LFU admission with hysteresis, and replacements that never stall the token path: evict at once, admit when the copy lands. The learned profile persists as a side record beside the pack (RFC-0023, RFC-0041 §5.2).
8. **Single ownership.** Each expert and tensor has exactly one owner node, and within a node one owning cache. Duplicates exist only when the plan deliberately replicates a hot expert.
9. **Prefetch windows are bounded** by what the receiving tier can hold, with sizes set by the plan (ADR-0008).
10. **Syntropy yields memory on purpose.** On an OS memory-pressure signal or a user policy change, the tier manager shrinks the host and expert-cache budgets in steps, along the ladder of RFC-0001 §10.3. It requests a re-plan only when the ladder is exhausted or pressure outlasts `replan.pressure_window`, not on every yield (revised 2026-10-19, RFC-0020 §11.4). On CUDA, VRAM expert-cache segments can be unmapped and remapped at fixed addresses so captured graphs stay valid (RFC-0020).

## Consequences

Positive:
- Placement is what the solver planned, so predicted and measured times can be compared and the profile refined.
- Expert-level caching, admission and ownership become possible; the page cache supports none of them.
- One behaviour across OSes, except the explicit pinning adapters.

Negative:
- Pinned memory cannot be paged out. Syntropy holds memory the OS cannot reclaim on its own, so the yield path in item 10 is a product requirement, not a nicety.
- Per-OS pinning code is fiddly, and failure modes (OOM kill, lock failure, compressor panic) are severe. Each adapter needs hardware CI (RFC-0062).
- First start still has to pin tens of GB. Progressive pinning spreads the cost but does not remove it.

Work created: RFC-0020 (tier manager, residency map, pinning adapters, yield), RFC-0021 (direct-I/O path into pinned buffers), RFC-0023 (expert cache), RFC-0022 (KV in the same tiers), RFC-0010 (pinning budget and mmap A/B probes), RFC-0054 (budget policy).

## What would make us revisit

- On a Tier 1 platform, the mmap fallback wins the A/B for the hot expert tier on most machines tested. Make it that platform's default, still accounted and costed.
- Field reports of OOM kills, lock failures or compressor panics persist after the RFC-0020 adapters ship. Lower the default budgets or stop pinning on that OS.
- Users report the machine becoming unresponsive under Syntropy's memory hold even with the yield path. Revisit the default budgets and the pressure thresholds.

## Revisions

- **Revised 2026-10-19.**
  - Item 3: the RAM tier is split by who reads it (ADR-0200): `HostPinned` is DMA-registered memory a device copies from; CPU-only bytes live in `HostPageable` anonymous memory, locked when the budget allows. The Linux gate is RFC-0020 §6's (corrected 2026-10-20, below). The macOS cap (the lower of the OS GPU working-set limit and RAM minus reserve, raised only through `syntropy tune`) was already in place.
  - Item 4: DMA registration finishes before `Ready` until RFC-0020 open question 9 is settled (RFC-0020 amendment 8).
  - Item 6: the page-cache rule covers `HostPageable` homes too.
  - Item 10: a yield requests a re-plan only when the ladder is exhausted or pressure persists (RFC-0020 amendment 9).
  - Item 7: the learned profile is a side record beside the pack, not part of the sidecar (RFC-0041 §5.2).
- **Revised 2026-10-20.**
  - Item 3 (Linux): the 2026-10-19 wording kept a MemFree check between commit and registration. Such a wait can deadlock when clean page cache keeps `MemFree` low, and RFC-0020 §6 now has no free-page test anywhere. Item 3 and the Context bullet follow it: capacity from MemAvailable minus the reserve, cgroup headroom and the per-NUMA estimate; reclaim forced by the commit; registration of resident pages only; distress from PSI or the `allocstall_*` counters plus step latency; give-up only after sustained pressure.
