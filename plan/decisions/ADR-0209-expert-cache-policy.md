# ADR-0209: Expert cache policy: per-layer byte slots, decayed LFU with hysteresis, a shadow-LRU guard, replacements off the critical path

- Status: Accepted (2026-10-20; revised 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0023))
- Related: ADR-0007 item 7, ADR-0008, ADR-0012, ADR-0201; RFC-0001 §5.5, §8.4 rule 7; RFC-0011; RFC-0020; RFC-0023 §4, §5; RFC-0040

## Context

The research gives unusually direct evidence on what an expert cache needs (`tiered-memory-inference.md` §5.2, §5.3, §5.5, §16):

- **Per-layer admission.** One shared arrival-order counter filled the cache from the first ≈26 layers: 2.97% hit rate against 70.4% with per-layer slots.
- **Byte-sized slots per layer.** Sizing each layer's slots for its own experts fitted 3,556 experts instead of 2,673, +13% decode.
- **Decayed LFU with hysteresis.** Replacing the least-routed resident experts with the most-routed missing ones every 4 rounds, decaying counts by 0.7, and requiring a 1.5× margin lifted the hit rate from 0.50 (static profile) to 0.72.
- **Off the critical path.** Evicting at once and admitting when the copy lands took decode from 91.7 to 94.4 tok/s.
- **A persisted profile** lets the next start begin where the last ended.
- **Other policies.** Plain LRU is the floor; learned recency/frequency and forecast-driven Belady policies reported higher hit rates in 2026 papers. Expert access has been reported as not consistent with LRU/LFU assumptions on some models.
- **Routing skew differs by family.** DeepSeek-class routers are trained for load balance; the surveyed numbers come from a 512-expert, top-10 model (`hardware-and-performance-model.md` §6 item 4).

SC-9 requires the hit rate to be at least max(cache fraction, LRU at equal capacity) on every recorded trace, so a policy that loses to LRU on some trace fails a success criterion.

## Options considered

| Option | For | Against |
|---|---|---|
| A. LRU | Simple; the generic baseline | Tens of points below better policies on skewed routing |
| B. Static fill from a profile | Deterministic | 0.50 against 0.72 adaptive in the surveyed engine |
| C. Decayed LFU with hysteresis, per layer | Best measured simple policy | No guarantee against LRU on traces with fast-shifting routing |
| D. A learned or forecast-driven policy | Highest reported hit rates | Needs trained predictors and traces Syntropy does not have yet; more compute on the housekeeping thread |
| **E. C as the default, plus per-layer shadows (decayed LFU, LRU, `Freeze`) and adoption of the leading one with a two-window dwell; `Static` for deterministic runs; D as a later `CachePolicy` variant** | Keeps C's measured gains; tracks the best of the three on any trace whose behaviour lasts longer than two windows | Two shadow structures per layer; a mode switch adds a parameter |

## Decision

**Option E**, specified in RFC-0023 §4–§5:

1. Slots are sized per MoE layer from the pack extents; per-layer budgets come from the solver or, without them, from greedy marginal value on the profile, with a floor of `min(k_l, ⌊bytes / Σ_l s_l⌋)` slots per layer and the rest by marginal value. Slots are striped across elastic segments (ADR-0201).
2. Every 4 decode steps: counts × 0.7; replace while a candidate's count ≥ 1.5 × the victim's, up to 96 per cache and within the per-step PCIe admission budget. Prefill accesses count at weight 0.1.
3. Per layer, three metadata-only shadows run on the same access stream: decayed LFU, LRU and `Freeze` (the resident set at the start of the window, unchanged). Every 1,024 decode accesses the layer adopts the mode of the shadow with the highest hit rate over the last two windows, with no margin and a two-window dwell. SC-9 is measured on the aggregate hit rate over every recorded trace after a 4,096-access warm-up per layer, with a 0.5-point tolerance [target]; that wording is an ARCH follow-up.
4. Replacements go through the device-resident slot table, updated by a host-mapped delta ring applied by the first node of each captured step; the victim becomes a miss at once, the newcomer becomes visible only after its copy lands, and a predicted expert already in a prefetch slot is promoted by an in-device copy into the victim's slot, so slot roles never change.
5. The profile (decayed counts and the resident set) is saved as the pack's `XPROFILE` side record (`side/<pack-key>.xprofile`, own schema version) at drain, every 10 minutes and at shutdown, and fills the cache at the next `Warming`. Without a profile, a sibling pack's profile of the same model on the node, whose counts the new pack's own replace after warm-up (OQ-MOE-14); without that, the router's bias where the family has one, else round-robin.
6. **Who may see a profile** (added 2026-10-20, below). Profiles and their curves are shared among the nodes one operator runs in a cluster, exchanged with a node run by another operator only when both operators opt in, and never published when derived from users' prompts. The project may publish profiles derived from its public corpus only if S-01 shows the cold start matters (RFC-0023 §13).

## Consequences

Positive:
- Carries over the measured gains (per-layer slots, decayed LFU, hysteresis, off-path replacement) without the surveyed engine's hard-coded geometry.
- SC-9 is enforced per layer rather than hoped for.
- Captured graphs stay valid while the cache changes.

Negative:
- The default parameters come from one model family; per-family tuning may be needed (RFC-0023 Open question 1).
- The shadow structures cost housekeeping CPU: O(k) per token per layer.
- `apply_deltas` must be implemented by every native provider.

## What would make us revisit

- On recorded traces of a v1 family, the guard keeps a layer in recency mode more than half the time: decayed LFU is the wrong default for that family.
- A learned or forecast policy beats this one by more than 5 points of hit rate on the trace corpus at equal capacity, with housekeeping cost under 2% of one core: add it as the default `CachePolicy` variant.
- Load-balanced routers show hit rates near the uniform prior at every practical capacity (RFC-0023 Open question 2): the cache's VRAM is then better spent on KV or a higher-precision hot set, which the solver will choose given the measured curve.

## Revisions

- **Revised 2026-10-20.** Decision 6 added: profiles derived from prompts can be personal data, while profiles derived from the project's public corpus carry none. Before any user trace exists, the picker shows corpus-derived hit curves as a labelled range. Decision 5 gains the sibling-profile seed (OQ-MOE-14).
