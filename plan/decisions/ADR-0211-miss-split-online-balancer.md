# ADR-0211: The split of expert misses between PCIe and the CPU is fitted online per layer

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0023))
- Related: ADR-0006 item 8, ADR-0012, ADR-0019 item 8; RFC-0001 §5.4 `MissSplit`, §6.3; RFC-0010; RFC-0011; RFC-0023 §6, §8

## Context

- ADR-0006 item 8 makes the MoE miss split a per-layer schedule: copy a share of RAM-resident misses over PCIe for the GPU while the CPU computes the rest in place, choosing the share to minimise `max(GPU, CPU)` from fitted per-layer costs, including the slowdown DMA reads cause the CPU pool.
- The right share depends on the weight format and the CPU kernel. In the surveyed engine, 0.2 suited a repacked 2-bit format whose CPU kernel already saturated RAM bandwidth, while 0.55 suited i-quants whose CPU kernels were arithmetic-bound, taking decode from 70 to 78 tok/s and from 56 to 65 tok/s (`tiered-memory-inference.md` §5.4, §14).
- That engine's default online rule was a closed form, `L / (L + 1.85 R)` clamped to 0.10–0.60, adjusted at most three times in the first ≈10 s; the 1.85 was fitted to three machines. Its opt-in balancer fitted `CPU = a + c·n_cpu + d·n_pcie` and `GPU = g0 + g·n_pcie` per layer by least squares with forgetting, and picked the split minimising `max(GPU, CPU)`.
- Syntropy runs on many CPUs, GPUs, PCIe configurations (x16, x8, x4 chipset slots, Thunderbolt tunnels) and formats; the research lists a fixed constant and per-architecture defaults among the open problems (§18).

## Options considered

| Option | For | Against |
|---|---|---|
| A. A fixed share per format | Simple | Wrong across machines; the constant would be guessed |
| B. The closed form with a fitted constant | Cheap; reasonable prior | The constant came from three machines; ignores per-layer differences and the DMA slowdown |
| C. A one-time calibration at install | Machine-specific | Misses workload and thermal changes; adds install time (the surveyed calibration took 15–30 minutes) |
| **D. Per-layer recursive least squares on live samples, with the closed form as the prior, guards against bad fits, a cap on step-to-step change, and a fixed mode for deterministic runs** | Adapts per machine, layer and workload; includes the DMA term; costs nothing on the token path | Needs device timestamps per layer; a bad fit could hurt until the guards catch it |

## Decision

**Option D**, as specified in RFC-0023 §8:

1. Per MoE layer, fit `T_cpu = a + c·B_inplace + d·B_copy` and `T_gpu = g + p·B_copy + k·B_hit` (bytes, not counts, because expert sizes can differ) by recursive least squares with forgetting factor 0.98 per sample.
2. Choose `B_copy = clamp((a + c·B_miss − g − k·B_hit) / (c − d + p), 0, B_copyable)`, using the last 8 steps' average miss and hit bytes, and write it into the step's parameter block before launch. A deterministic rule (most tokens first, then expert index) turns it into whole experts identically on the dispatcher, the CPU pool and the GPU.
3. Until a layer has 32 samples, use the plan's `MissSplit::copied_share`, which the solver derives from the closed form with the profile's measured `L` and `R`.
4. Reject fits with negative coefficients or residuals above 30% of the mean; limit the change per step to 25% of `B_miss`; freeze the split in deterministic runs.
5. Only `DmaPinned`, resident experts are copyable (ADR-0200).

## Consequences

Positive:
- The split follows each machine's actual RAM, PCIe and kernel behaviour, per layer, and tracks thermal and workload changes.
- The DMA slowdown term keeps the balancer from copying when copying would steal the CPU's bandwidth, the case where a fixed 0.55 would hurt.
- Telemetry exposes the fitted coefficients, which RFC-0010 can fold into the capability profile for the solver.

Negative:
- Kernels must write per-layer device timestamps into host-mapped telemetry (RFC-0040).
- A run's split varies over time, so results differ by rounding between runs; ADR-0019 already excludes byte-identical output, and deterministic runs freeze it.
- More state on the housekeeping thread: six coefficients and their covariance per layer.

## What would make us revisit

- On the reference rigs the fitted split's layer time is more than 5% worse than the best fixed split from a sweep (RFC-0023 target): the model is missing a term, likely PCIe contention with admission copies or NUMA effects.
- The closed-form prior alone is within 2% of the fit on every Tier 1 rig: drop the fit to save the per-layer timestamps.
- Multi-GPU helper caches (M3) need more than two compute sites: extend the fit to n sites as RFC-0023 §8 sketches.
