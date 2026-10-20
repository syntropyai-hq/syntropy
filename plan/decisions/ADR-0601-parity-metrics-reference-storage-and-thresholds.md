# ADR-0601: Parity metrics, reference storage and gate thresholds
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the quality and release engineering area)
- Related: ADR-0002, ADR-0019; RFC-0042, RFC-0060 §4, §5

## Context
ADR-0019 fixes what exact mode means:
- teacher-forced top-1 agreement and KL(P_ref ‖ P_cand), mean and p99, against a same-quantization reference executor, over a versioned corpus;
- a pass is within 2× the reference-to-ggml-CPU spread and within RFC-0042's absolute ceilings.

It leaves the implementation open:
- how the reference is computed and stored;
- how KL is computed when the candidate runs on another node;
- how per-category failures are caught;
- what happens when the spread is near zero.

The decisions here are contestable on cost:
- Full-vocabulary logits are about 608 KB per position at a 152K vocabulary, or about 40 GB for a 65,536-position corpus per model and quantization.
- Recomputing the reference per candidate run costs hours on 70B-class models (RFC-0060 "Performance model and budgets").

## Options considered
1. **Store full logits.** Exact KL. But storage runs to tens of GB per model and quant, and moving full logits from a remote tail costs 608 KB per position on the network.
2. **Recompute the reference live per run.** No storage, but reference cost is multiplied by paths × cadence, which is infeasible for large models.
3. **Store top-K reference distributions and compute a coarse-grained KL.** Small, and computable over the network with a gather tap. It gives a lower bound on KL, whose tightness must be shown.
4. **Store only reference argmax plus a few log-probs.** Smallest, but KL is unmeasurable, which violates ADR-0019.

On thresholds:
- **(a) 2× spread only.** Exactly ADR-0019, but fragile when the spread is near zero.
- **(b) Absolute ceilings only.** This loses the relative guard that catches regressions well under the ceiling.
- **(c) Both, plus floors that default to zero.**

## Decision
Revised 2026-10-19.

1. **Reference blocks.**
   - Per scored position, store:
     - the argmax and the top-1/top-2 margin;
     - the full-vocabulary logsumexp and `log r` (the logsumexp over the complement of the top-K set minus the total), both computed in F64;
     - the entropy;
     - the top-K = 256 reference logits;
     - M = 64 tail-sample logits, drawn with replacement from the complement in proportion to the reference's probabilities. Uniform draws would give each term a weight of about 2,370 and rarely sample the ranks just below K, where the tail's mass sits.

     That is 2,592 bytes per position.
   - Files are zstd-framed and keyed by `ModelId`, quantization, corpus version, reference-executor version and mode.
   - They are computed once per key and stored in the fixture store (RFC-0060 §5.3).
   - Corpus tiers count scored positions per category: `smoke` 4,096, `standard` 8,192 or the power minimum of decision 4 if larger, `calibration` ≥ 50,000. Every category holds at least 32 sequences (64 in `calibration`). `long` reaches its size through 32 or 64 documents scored on their last 128, 256 or 784 positions, not through more positions per document.
2. **Metric.** `KL_K = Σ_{i∈T} p_i (log p_i − log q_i) + r (log r − log s)`, computed in F64 from logits, with `log s` returned by the tap. It is the KL between distributions coarse-grained to T plus one tail bucket, and by the data-processing inequality a lower bound on the full KL.
   - A non-finite `log s` is a harness error, never clamped.
   - `KL_tail = (r/M) · Σ_{j∈S} (log p_j − log q_j) − r · (log r − log s)`, over the proportional tail sample S, is an unbiased estimate of what `KL_K` hides, with equal weights r/M on every term. `KL_K + KL_tail` is therefore an unbiased estimate of the full KL.
   - **M0 obligation:** compute exact KL from full logits on the nano and micro fixtures and on one real checkpoint per family, and show |KL − KL_K| ≤ max(0.01 · KL, 0.01 · C_kl_mean) at the mean and p99, per category. A planted defect in rare LM-head rows must be caught by the tail check.
   - If the bound fails, raise K and bump the block format.
   - The check repeats on every corpus version.
3. **Taps.**
   - A candidate on the driver's host writes full logits to shared memory.
   - A remote tail receives the reference top-K and tail-sample ids over a `Tap` channel. It returns the candidate's logits at those ids, its logsumexp and its complement logsumexp (in F64), its argmax and its margin.
   - Layer-output and stage-boundary taps ship in release builds behind `debug.parity_api`, as a separate step-program variant (RFC-0060 §1.3).
4. **Gate.**
   - **Thresholds** are RFC-0042 §6.2's, from its `calibration` run (≥ 50,000 positions per category), not from the run being judged:
     - `max(2 · S, φ)` pooled and `max(3 · S_c, φ)` per category, and ≤ the class ceiling C;
     - a category with zero calibration disagreements uses the rule-of-three bound 3/n_c.
   - **Decision.** A candidate fails a metric only when the evidence places it above its limit L = min(threshold, ceiling), one test per metric:
     - top-1: a one-sided exact binomial test on disagreement counts, with the sequence-clustering design effect measured in calibration;
     - `kl_mean`, the tail mean and `kl_p99`: one-sided lower confidence bounds from a bootstrap over whole sequences, with at least 20/α′ resamples;
     - family-wise α = 0.01 per report, spent hierarchically: 0.005 over the 4 pooled tests and 0.005 over the 24 per-category tests.
   - **Power target [target]:** at `standard` and `calibration`, a candidate whose true metric is 2 L fails with probability ≥ 95%, pooled and per category. `smoke` is a screen without one.
   - **Minimum sizes** per category follow from the power target (RFC-0060 §5.4): at least 32 sequences; for top-1, n_c ≥ deff_c · n₁(L_c), the smallest n at which the exact test at α′ rejects a rate of 2 L_c with probability ≥ 0.95; for KL means, m_c ≥ ((z_{1−α′} + z_{0.95}) · σ_c / L_c)². At an illustrative L_c = 0.6% and deff = 1, n₁ = 6,198. Calibration re-derives the minimum per corpus version, and a short `standard` category grows by sequences in the next one.
   - **Reading of ADR-0019:** "within 2× the spread" means "not shown above 2× the spread at the report's α". ARCH is asked to confirm it.
   - **False-reject target:** at most 1% of reports reject a candidate whose true metrics equal the spread [target], checked during calibration over ≥ 100 runs of correct-but-different candidates.
   - **Floor.** φ defaults to 0, which is ADR-0019's rule unchanged. RFC-0042 may set a non-zero φ for a dtype class only under ADR-0019's revisit clause, with the measured spread and the rejected correct kernel as evidence. **The one exception** is the `widen.f32` baseline below, where φ is set from the first calibration without a rejected kernel. The baseline per scheme follows RFC-0042 §6.2: the pack itself for GGUF types and plain floats, the exact ggml transcription for transcribable tensors, and a `widen.f32` copy for FP8, NVFP4 and non-transcribable `PackedInt`. For the `widen.f32` schemes the spread measures summation order only, so the calibration jobs apply `φ = F_m` from the first calibration, recorded as an ADR-0019 revisit event (ADR-0422 decision 2a).
   - **Starting defaults.** The per-category factor 3 and the near-tie margin τ_tie = 0.05 nats (diagnostic only) are starting defaults, calibrated at M0.
5. **Free-running greedy first-divergence** is reported, never gated (ADR-0019 §2).
6. **Large models.** Sampled-layer mode applies per (model, tier, category) part whose one-off reference is estimated above 48 h at the low end of the reference's measured throughput. That is the `long` category of 70B- and DeepSeek-V3-class models, and most parts for 1T-class models. Those parts are gated by layer-local parity on sampled layers, through the release-build taps of decision 3. The other parts run end to end with full references.
   - **Layer-local gate.** One procedure, shared with RFC-0042 §7: single-layer windows [a, a+1) at every sampled layer, recomputed by the reference from the candidate's tapped input and KV prefix. The last window runs through the LM head.
   - **Metrics:** mean and p99 `rel_l2` per sampled layer and category, plus top-1 and `KL_K` for the head window.
   - **Thresholds:** `max(2 · S_ll, φ_ll)` per (layer kind, dtype class, category, statistic), with S_ll the RFC-0042 baseline's layer-local spread against the reference on the calibration models, and never above RFC-0042 §5.2's class ceilings. Where the large model's own within-budget parts show a larger baseline spread, that value is used.
   - **Decision:** decision 4's evidence rule, with ≥ 32 sequences per category, α = 0.01 per report.

7. **Where references are computed** (added 2026-10-20). The calibration reference, and any reference block whose one-off CPU run would exceed decision 6's 48 h rule, is computed by `F32Fixed` on a GPU (`RefDevice::Gpu`, RFC-0060 §4.1). It uses `F32Fixed`'s reduction order, dequantisers and software transcendental functions, and runs on the untimed agent GPU pool or on-demand cloud instances, never on a timed rig (ADR-0610 decision 4).
   - **Cross-check.** On the nano and micro fixtures its logits must be bit-identical to the CPU path's. A mismatch is a reference defect, and it computes nothing that gates until fixed. Because the two paths agree bit for bit, their reference blocks share one key.
   - **Provisional ceilings.** Until this method has produced the calibration for the MoE families, their RFC-0042 ceilings are marked `provisional` in the M0 and M1 exit records, and an exit that relies on one says so.

## Consequences
- **Positive.**
  - Reference storage is about 127 MB per `standard` run per model and quant [estimate], so references for the whole roster fit in single-digit GB (calibration references add about 12 GB for the calibration models).
  - Parity works on real multi-node instances, not only on one host.
  - The gate is exactly ADR-0019's unless a floor is justified in public.
  - Per-category bounds stop an overall pass from hiding a failure concentrated in tool-call JSON or long-context positions.
- **Negative.**
  - `KL_K` is a lower bound. A candidate whose divergence lives entirely in the reference's tail beyond K could look better than it is. The M0 tightness check bounds this, and it must be repeated if K, the corpus or the families change.
  - Reference blocks must be regenerated when the reference executor, the dequantisers or the corpus change (RFC-0060 §12.5).
  - The per-category factor is a judgement until calibrated.

## What would make us revisit
- The tightness check fails at K = 512 on any family. Then store full logits for that family's `smoke` corpus, and use top-K only for `standard` and `calibration`.
- The per-category bound rejects candidates that pass every other check, more than once per release, on categories with fewer than 512 positions. Then increase category sizes rather than loosen the factor.
- RFC-0042's calibration finds a dtype class whose spread is zero to within float noise. Apply ADR-0019's revisit clause for that class, through φ.

## Revisions
- Revised 2026-10-20: decision 7 added. On a CPU the calibration reference does not fit its budget once attention is counted, so only dense models of about 1B could calibrate and the MoE ceilings would rest on dense evidence. A fixed-order F32 reference on a GPU computes the calibration reference and the parts over the 48 h rule, bit-identical to CPU `F32Fixed` on nano and micro fixtures, and the MoE ceilings are `provisional` in the M0 and M1 exits until it has calibrated them. RFC-0042 recomputes the calibration budget with attention included; RFC-0060 §4.1 and QUAL-E4 carry the executor change.
