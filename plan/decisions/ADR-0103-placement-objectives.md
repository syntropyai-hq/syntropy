# ADR-0103: Placement objectives: per-pass critical path with tail latency, speculation by outer loop, throughput by bottleneck

- Status: Accepted (revised 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the placement and scheduling area)
- Related: ADR-0000 (D2), ADR-0008 item 4, ADR-0017 item 5; RFC-0011 §7; SC-13, SC-15, SC-21

## Context

ADR-0008 fixed the default objective as batch-1 decode critical-path time with an alternate throughput objective. Three details are contestable:

1. **Which latency to charge a hop.** Wi-Fi's median hop is 3–7 ms, its p99 ≈250 ms; at p99 one hop quadruples token time in the mixed-cluster example (cost model §2.5, §4c). Wired links have tight tails.
2. **How speculative decoding enters.** The expected time per token is `(t_draft + t_pass(γ+1)) / E[tokens]` with `E = (1 − a^{γ+1}) / (1 − a)` (ADR-0017 item 5). The ratio is not linear, and `t_pass(γ+1)` depends on placement: a CPU tier turns compute-bound at ≈3–9 tokens at Q4 (cost model §3.1), so a window size can move the best placement of a layer from CPU to GPU.
3. **What throughput means.** With D stages and M sequences in flight, aggregate throughput is bounded by `1 / max_s t_s` (§3.7). D2 also requires a per-user floor. MoE union growth makes deep batches on slow tiers expensive (§3.9).

## Options considered

| Topic | Options | Chosen |
|---|---|---|
| Hop latency | mean; p50; p99 everywhere; p50 on wired media and p99 on `WiFi`/`Overlay` in latency mode, p50 in throughput mode; **p50 on wired media and the mean on `WiFi`/`Overlay`, with a stall-rate bound and a jitter badge** | the last, since 2026-10-20. The previous choice charged every token the tail and overstated the expected Wi-Fi hop about 20× |
| First token | decode only; a weighted sum of decode and TTFT; **decode subject to a TTFT cap at 2× the best candidate**; prefill on the compute-rich node | the cap, since 2026-10-20: it keeps D2's decode priority and rules out plans with a much slower first token |
| Speculation | ignore in placement; MINLP with the ratio; **fix the window, solve, scan γ on the result, re-solve at the best window once (Dinkelbach-style outer loop)** | outer loop: exact for a fixed window, at most two MILP solves |
| Throughput | maximise Σ tok/s directly (needs D in a denominator); **minimise the bottleneck stage θ at a fixed micro-batch b_μ, enumerate b_μ, enforce the per-sequence floor as a ring-time bound** | bottleneck at fixed b_μ: linear, and the expert union at b_μ enters through the coefficients |
| Stage count | explicit stage-count penalty; **no special rule: hops and `t_fixed` per stage are in the objective** | the objective's own terms already prefer dropping a weak device to adding a hop |

## Decision

1. **Latency mode** minimises one ring traversal at pass size T: dense and KV reads at the executing kernel's W_eff, per MoE block the max of GPU, CPU and NVMe work, device hand-offs, `t_fixed` per device used, hops at p50 (wired) or the measured mean (`WiFi`, `Overlay`; p99 until 2026-10-20), collectives inside TP/EP groups, the LM head and the sampler, and on re-plans a migration term `ω_move = t_tok / (β_fetch · payback_horizon)` per byte moved. *Revised 2026-10-20 (below):* it does so subject to three bounds:
   - **TTFT cap.** TTFT at `L_ttft` (the workload's p90 uncached prompt, prior 4096 tokens) ≤ 2 × the lowest TTFT among the candidate plans the solve compares. Both numbers are shown for every preview candidate. The cap binds from M2 (RFC-0011 §7.1).
   - **Stall rate.** On `WiFi` and `Overlay` edges, the summed probability that a hop exceeds its mean by `solver.stall_threshold` (300 ms) is ≤ `solver.max_stall_rate` (0.01), the example bound "p99 stall ≤ 300 ms" [target]. Plans with such edges carry a jitter badge: the stall rate and the p99 token time (RFC-0011 C14). p99 stays in the TP/EP gate (ADR-0006) only.
   - **Departure margin.** A departure-prone node (a laptop, a node on battery, or one reached only over wireless links) hosts a stage only for a predicted gain of ≥ 10% [target] over the best plan without such nodes, or when capacity needs it (RFC-0011 §4).
2. **Speculation** is chosen by an outer loop: solve at T = 1; scan drafters and γ on that placement with the evaluator; if the best γ* improves t_tok by ≥ 2%, re-solve at T = γ* + 1 warm-started; keep the better plan. The plan's `max_window` is γ* + 1, which bounds captured graphs and KV slots (RFC-0012).
3. **Throughput mode** minimises θ ≥ every stage's time and every hop's time at micro-batch b_μ, with ring time ≤ 1 / per-sequence floor; b_μ ∈ {1, 2, 4, 8, 16, 32} up to the KV limit, keeping the best `b_μ / θ`. Speculation is off unless the outer loop predicts a gain at that b_μ. Spare capacity produces a replica hint rather than a deeper batch.

## Consequences

- Positive: wireless members are used when they help and kept off the critical path when their tail would dominate; speculative window size follows the measured CPU and GPU batch curves; throughput plans see MoE union growth.
- Negative: two to six MILP solves per throughput plan (one per b_μ, warm-started), which RFC-0011 budgets inside SC-31 for latency mode only; throughput plans may take longer and are always background solves.
- Work created: RFC-0012 must keep runtime γ within the plan's window and report acceptance per drafter, model and request kind (greedy or sampled), with no task-class dimension (RFC-0012 §11); RFC-0061 measures the p99 assumption for wireless links in alpha.

## What would make us revisit

- Measured single-stream t_tok on a Wi-Fi member sits much closer to the median-costed prediction than the p99-costed one over long sessions. Then cost wireless hops at a lower quantile, or at the median plus a measured stall rate. *Acted on 2026-10-20: the mean plus a stall-rate bound (below).* Revisit the 300 ms and 0.01 defaults once alpha's Wi-Fi traces exist (RFC-0010 open question 6).
- Alpha data gives departure rates per node class. Then replace the 10% departure margin with the expected departure cost in the objective.
- The TTFT cap excludes the decode-best plan in most multi-node previews, or users report slow first tokens within the cap. Then revisit the factor of 2.
- The outer loop's second solve changes the placement in fewer than 5% of reference cases. Then drop it and scan γ only.
- Throughput users need fairness or latency SLOs beyond a per-sequence floor. Then add a TTFT constraint to the throughput objective.

## Revisions

- **Revised 2026-10-20.**
  - Decision 1, TTFT: latency mode minimises decode subject to a TTFT cap of 2× the best candidate's TTFT, with both numbers in the preview. Decode alone picked plans whose first token was several times slower than the best available, with nothing to trade the two (OQ-PL-14).
  - Decision 1, wireless hops: the measured mean for prediction and the objective, a stall-rate constraint, a jitter badge, and p99 kept for the TP/EP gate. A macOS periodic radio stall is flagged by the link estimator, because it recurs about once a second rather than sitting in a 1% tail.
  - Decision 1, device selection: the 10% departure margin, until alpha data supports pricing departure cost.
