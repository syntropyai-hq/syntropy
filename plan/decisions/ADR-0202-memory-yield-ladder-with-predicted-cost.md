# ADR-0202: Yield memory along a fixed ladder, priced before each rung

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0020, RFC-0022))
- Related: ADR-0007 item 10, ADR-0009; RFC-0001 §3.5 `MemoryPressure`, §7.2, §10.3, §11.3 `ResourcePolicy::memory_yield`; RFC-0020 §8.5, §11; RFC-0022 §12; RFC-0051; RFC-0054

## Context

- Syntropy runs on machines people use for other things. Pinned and locked memory cannot be reclaimed by the OS, so if Syntropy does not give it back, the OS's only remedy is killing a process (ADR-0007, negative consequences).
- A surveyed system that left weights to the page cache kept memory pressure under 6% on every device, against over 50% for runtimes that hold weights; everyday background apps raised its TPOT by 5–18% and a game by 66% (`multi-device-inference.md` §8.1). The research recommends yielding RAM when the user needs it and modelling the TPOT cost of doing so (§15.6).
- RFC-0001 §10.3 fixes the order of what to give back: idle-session KV, RAM expert-cache entries over an `Nvme` home, prefetch staging and other caches, then host copies of `Nvme`-homed or replicated groups. It leaves open how much each runner gives, how the price is computed and shown, and what happens at the edge of an OOM kill.
- A node may run several instances. Asking each runner for an equal share returns expensive bytes from a busy instance while an idle one keeps cheap ones.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Ignore pressure; let the OS act | No code | The OS kills the runner; the user loses work and trust |
| B. Fixed order, equal share per runner, no prices | Simple | Returns expensive bytes before cheap ones across instances; the dashboard cannot say what it costs |
| C. Drop everything optional at the first signal | Maximally polite | Throws away caches at a brief spike; the next minutes run slowly for nothing |
| **D. Fixed rung order; each runner publishes a priced yield menu; the daemon fills the request from the cheapest entries across runners; levels with dwell and hysteresis; staged reacquisition with back-off; one Critical-only rung that unlocks CPU-only homes** | Cheapest bytes first; the user sees the price; no oscillation; avoids OOM kills | More moving parts; predictions can be wrong |

## Decision

**Option D.**

1. **Levels** (Moderate, High, Critical) from per-OS signals: PSI and cgroup events on Linux, the low-memory notification on Windows, the dispatch memory-pressure source on macOS, plus `available` against `resource.ram_headroom`. A level must hold 2 s before action and falls one step after 10 s below its threshold.
2. **The ladder** is RFC-0001 §10.3's four rungs plus a fifth, at Critical only: unlock `Locked` `HostPageable` homes so the OS may page them instead of killing the process (proposed as an RFC-0001 amendment).
3. **Prices.** Each runner publishes, in its residency digest, a yield menu: per rung, the bytes it can release and the predicted Δt_tok. Rung 2's price uses the learned `h(c)` curve: `(h(c) − h(c − Δ)) · B_routed / W_nvme`. Rung 1 costs live sequences nothing; its price is the future resume time. The price is honest only if the bytes released are the coldest: elastic segments are striped across layers, so before a shrink the tier manager moves each layer's hot residents out of the tail segments into the coldest surviving slots (RFC-0020 §8.4).
4. **Allocation.** The daemon fills `reclaim` from the cheapest Δt_tok per byte across all its runners, then sends each runner its share.
5. **Policy.** `memory_yield = Never` acts only at Critical, rungs 1–3; `UnderPressure` (default) acts at High and Critical, rungs 1–4, plus 5 at Critical; `Aggressive` acts from Moderate and also releases rungs 1–3 after 10 minutes idle.
6. **Reacquire** after 60 s at Normal, in reverse rung order, one segment or pin step per 5 s, through the progressive-pinning gate; the delay doubles (to 15 minutes at most) if pressure returns within 10 minutes.
7. **Visibility.** Every rung taken emits a report with predicted and measured Δt_tok; the runner is `Ready { degraded }` while any rung is held. VRAM pressure from other programs uses the same mechanism on the elastic expert arena (ADR-0201).

## Consequences

Positive:
- The cheapest memory goes first, across instances.
- Users see what yielding cost in tok/s, which turns an invisible slowdown into a choice they can change (`memory_yield`, headroom).
- Hysteresis and back-off stop Syntropy fighting the user's other programs for the same memory.

Negative:
- The price of rung 2 is only as good as the `h(c)` curve; early in a run it uses the uniform prior and overstates the cost.
- Rung 5 can turn an OOM kill into heavy paging, which is better for the user's other work but can make Syntropy very slow until it re-plans.
- More daemon logic: the cross-runner allocation and the per-OS signal adapters.

## What would make us revisit

- Measured Δt_tok after a rung differs from the prediction by more than 25% on the reference rigs (RFC-0020 target). Then improve the price model or stop showing it as a prediction.
- Rung 5 proves worse than an immediate re-plan in M1 chaos runs (RFC-0020 Open question 7). Then drop it and re-plan at Critical instead.
- Users on `UnderPressure` still report the machine becoming unresponsive. Then lower the default headroom thresholds or make `Aggressive` the laptop default.
