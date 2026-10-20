# ADR-0100: Probe W_eff with the real quantized kernel, per scheme, with a batch curve

- Status: Accepted (revised 2026-10-19)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the placement and scheduling area)
- Related: ADR-0006, ADR-0008, ADR-0012, ADR-0014; RFC-0010, RFC-0011, RFC-0012, RFC-0023, RFC-0040

## Context

The solver's dominant term is `Σ B / W_eff` (cost model §0 item 1). Three facts make a single bandwidth number per (device, tier) insufficient:
- **η depends on the kernel and the quant.** Achieved bandwidth ranges from 0.24 to 0.8 of peak across the calibration anchors, and MoE kernels run lower than dense ones on the same machine (0.53 against ≈0.73 on M3 Ultra; cost model §2.2).
- **CPU kernels can be compute-bound at batch 1 for some formats.** The surveyed single-GPU MoE engine found its repacked 2-bit CPU kernel saturating RAM bandwidth while its i-quant kernels were bound by arithmetic and left bandwidth spare. That difference changed the best PCIe share of misses from 0.2 to 0.55 (`tiered-memory-inference.md` §5.4, §14).
- **The cost of more tokens per pass is a curve, not a constant.** n* (cost model §3.2), the verify-window cost `t_target(γ+1)` (§3.10) and throughput-mode batching (§3.11) all need the time per pass as a function of tokens per pass. The ridge on a desktop CPU sits at ≈3–9 tokens at Q4 (§3.1), well inside the range speculation uses.

The probe budget is 15 s per node (SC-22). A triad or memcpy benchmark is cheap, but it measures the wrong quantity: 90 GB/s DDR5 triad against an assumed ≈70 GB/s CPU matvec in example (a), a 29% error on a term that is 93% of token time there (§4a).

## Options considered

| Option | For | Against |
|---|---|---|
| A. One copy/triad bandwidth per (device, tier), scaled by a fixed η per kernel class | cheapest probe (≈0.3 s per tier); simple | η is the quantity that varies; a table of η values is a spec sheet by another name. Misses compute-bound CPU formats entirely |
| B. Real kernel per (device, tier, kernel class) at one representative scheme | captures kernel quality; ≈0.5 s per tier | wrong for models whose schemes behave differently (FP8 vs Q4_K vs MXFP4); still no batch dependence |
| **C. Real kernel per (device, tier, kernel class, scheme), with a batch curve at n ∈ {1, 2, 4, 8, 16, 32}, for a bounded probe set; other schemes measured by a model probe at registration** | gives the solver W_eff, n*, t_verify(T) and batching cost directly from measurement; bounded cost | more probe time and a larger profile (≈34 KiB per node, estimate); needs the model probe path |
| D. Full model run at install time to calibrate everything | most faithful | minutes per model; a surveyed 15–30 minute calibration bought 7% on one machine (`tiered-memory-inference.md` §4) |

## Decision

**Option C.**
- The startup suite measures `QuantBw` at n = 1 (W_eff) for `DenseGemv` and `ExpertGemv` on every (device, tier) execute-in-place pair, over a probe set of at most 8 schemes: those with ≥ 1% of read-per-token bytes in any stored model, plus `{Q4_K, Q8_0, BF16, MXFP4}`. The batch curve (n ∈ {2 … 32}) moved to the `Model` probe because curves for the whole probe set cannot fit the 15 s startup budget (≈ 13 s on the CPU alone) — the move this ADR's revisit clause anticipated.
- A `Model` probe (≤ 5 s per node, target) runs on `ModelRegistered` for that model's schemes and real shapes before its first solve, including the batch curves (on one CPU host tier, with compute-bound points on a buffer of 4 × LLC). Until it reports, missing entries are `Derived` from the nearest measured scheme by bytes per weight and listed as plan assumptions.
- `DeviceCaps::tier_bw` (RFC-0001) keeps one value per (tier, class), the probe-set scheme with the most bytes, for readers that do not need the detail. The solver reads `quant_bw`.

## Consequences

- Positive: n*, verify-window cost and the CPU/PCIe miss split come from measurement on the user's machine, not from constants fitted elsewhere. Formats with compute-bound CPU kernels are recognised before placement.
- Negative: the startup suite spends ≈ 4 s of its 15 s on n = 1 bandwidth (RFC-0010 §4), and the first solve of a new model waits up to 5 s for its curves. The profile is larger. A new quant scheme needs its kernels in the probe before the first plan is accurate.
- Work created: RFC-0040 must expose each provider's decode GEMV and grouped expert kernel at arbitrary n for probing; RFC-0011 must read the curve as a piecewise-linear function.

## What would make us revisit

- On the RFC-0061 rigs, option B's single-scheme value predicts `t_tok` within 3% of option C's for every model in the conformance set. Then drop the per-scheme dimension and keep the curve.
- The startup suite cannot stay under SC-22 on 4-GPU nodes even with the batch curves in the `Model` probe. Then shrink the startup probe set to the schemes of stored models only.
- `Derived` values appear in more than 20% of first plans in M1 dogfooding. Then widen the default probe set.
