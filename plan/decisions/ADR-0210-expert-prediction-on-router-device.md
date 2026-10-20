# ADR-0210: Next-layer expert prediction runs fused into the router, on the router's device

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0023))
- Related: ADR-0003, ADR-0006, ADR-0012, ADR-0013; RFC-0001 §5.3 (`TensorClass::Router`), §5.8 side channel, §6.3 step 4; RFC-0023 §9, §10; RFC-0032; RFC-0040; RFC-0043

## Context

- Expert misses on slow tiers are only cheap if they are fetched before the layer needs them. Applying layer l+1's router to layer l's hidden state predicts layer l+1's experts well, because the residual stream changes little between layers. In the surveyed single-GPU engine, a helper thread applied a BF16 host copy of the next router to layer l's MoE input while the CPU pool ran layer l, predicted about half of the SSD reads, and gained 14% (6.9 → 7.9 tok/s), at the price of a host copy of every router (126 MB for that model) (`tiered-memory-inference.md` §6). SC-8 asks for at least half of NVMe misses to be issued before their layer's router runs.
- RFC-0001 §5.3 and the glossary describe exactly that design: "a host copy feeds next-layer expert prediction".
- On a discrete-GPU desktop running a large MoE model, the CPU's DRAM bandwidth is the bottleneck: it computes the in-place misses (scenario b2, `hardware-and-performance-model.md` §4 b2). A DeepSeek-V3 router is 256 × 7,168 BF16 weights ≈ 3.7 MB. Predicting one layer ahead for 58 MoE layers reads ≈213 MB per token: ≈3 ms of DRAM bandwidth on the CPU at 70 GB/s, taken from the misses, against ≈0.27 ms of VRAM bandwidth on the GPU at 800 GB/s, where the routers already live as part of the hot set [estimate].
- Some families know routing exactly and earlier: pre-attention routers, trained pre-gates, and token-hash-routed layers whose experts follow from the token ID (DeepSeek V4-class `num_hash_layers`, `backends-and-portability.md` §4.1).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Host router copies on a CPU thread (the surveyed design) | Works on every node; CPU sees the predictions immediately | Spends CPU DRAM bandwidth on the bottleneck resource; extra host memory for router copies |
| B. A separate predictor kernel on the GPU after the router | GPU bandwidth instead of CPU | An extra launch per layer; a second read of the hidden state |
| **C. Fused into the router op: extra GEMV rows for layers l+1..l+d over the same input, top `k + extra` per lookahead layer written into the same doorbell the router writes; host copies only where the router runs on the CPU; exact routing knowledge used instead wherever the family declares it** | Cheapest bandwidth; no extra launch; predictions arrive with the router's own result, the earliest moment possible | The router op becomes family-aware of lookahead; needs the next layers' routers on the same device (replicas across a stage boundary) |

## Decision

**Option C.**

1. RFC-0040's router op takes the routers of layers `l..l+d` and writes the actual selection for `l` and the top `k + experts.predict_extra` predictions for each of `l+1..l+d` into the layer's doorbell.
2. Depth comes from `PrefetchWindow::lookahead_layers`: 1 for RAM-homed experts, 2 for `Nvme`-homed ones by default; a depth whose recall stays below 0.3 is turned off per layer and re-probed later.
3. Across a stage boundary, the plan replicates the next stage's first `d` small `Router` groups on the previous stage, and predictions travel as `RouterHints` in the activation frame's side channel (M2).
4. Exact knowledge wins: `TokenHash` layers are requested one step ahead like row lookups; `PreGate` and `PreAttention` layers use the exact selection.
5. On CPU-only nodes the CPU computes the router, so it also computes the prediction, from host memory.

## Consequences

Positive:
- Prediction costs ≈0.1–0.25% of a b2 token instead of taking 1–3% of the bottleneck resource [estimate].
- No host memory for router copies on GPU nodes.
- Predictions are available as early as the router's own result.

Negative:
- RFC-0001 §5.3's `Router` comment and the glossary entry need amending (proposed in RFC-0023).
- The router op is more complex, and the plan must carry router replicas across stage boundaries.
- If the GPU is itself the bottleneck (all experts hot), the extra GEMV rows cost GPU time for no benefit; prediction is turned off for stages whose experts all sit in the fastest tier.

## What would make us revisit

- Measured recall of the fused predictor is materially lower than the host-copy design's on the same model (they compute the same thing, so a gap would indicate a numerical or input-selection difference).
- A family's router is too large for the extra rows to be cheap (for example a latent MoE with a very wide router input), making a smaller trained predictor worthwhile.
- A learned multi-step forecaster (`tiered-memory-inference.md` §6) shows enough gain to justify its own kernel.
