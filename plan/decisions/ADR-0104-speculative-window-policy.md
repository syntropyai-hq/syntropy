# ADR-0104: Speculative window policy: plan-bounded, per-sequence adaptive, one verify bucket

- Status: Accepted (revised 2026-10-19)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the placement and scheduling area)
- Related: ADR-0017, ADR-0019, ADR-0103; RFC-0011 §7.3; RFC-0012 §9; RFC-0022 §6; RFC-0040 §3.2, §4.4; SC-21

## Context

ADR-0017 makes speculation first-class and assigns window sizing to the solver and runtime adjustment to the scheduler. The window size γ trades three things:
- **Expected tokens per pass**, `(1 − a^{γ+1}) / (1 − a)`, which saturates: at a = 0.7 it is 2.53 at γ = 3 and 3.14 at γ = 7 (cost model §3.10).
- **Verify cost**, flat while every tier stays memory-bound and rising once a CPU tier turns compute-bound at ≈3–9 tokens at Q4 (§3.1), and on MoE layers growing with the expert union U(γ+1) (§3.9). RFC-0011 §15.1 shows γ* = 3 on a 1 TFLOPS CPU and γ* = 7 on a 2 TFLOPS CPU for the same model.
- **Memory and graphs**: KV slots for provisional positions and one captured graph per window bucket (RFC-0022 §6, RFC-0040 §4.4).

Acceptance varies by prompt and task (code edits, chat, structured output) and between greedy and sampled requests. A surveyed engine drafts only while its MTP head is at least 50% confident (`tiered-memory-inference.md` §13).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Fixed γ per model family | simple | ignores the machine's verify-cost curve and the task's acceptance |
| B. γ chosen by the plan only | uses the cost model | one γ for every sequence; acceptance drift needs a re-plan |
| C. Fully runtime-adaptive γ with no bound | adapts best | KV slots and graphs must be sized for the worst case, or re-captured mid-stream |
| **D. The plan fixes T_max from the cost model; the scheduler picks each sequence's γ ≤ T_max − 1 every window by maximising predicted tokens per second from the plan's `t_pass(T)` table and a per-sequence Beta posterior on acceptance; drafting stops early below 0.5 confidence; one captured verify bucket at T_max serves every γ** | memory and graphs bounded by the plan; per-sequence adaptation within a window or two; γ = 0 is in the search, so drafting turns itself off | needs the plan to publish `t_pass(T)` for every T ≤ T_max; padding rows cost compute on compute-bound GPU ops (rare at T ≤ 16) |

## Decision

**Option D.**
- The plan records γ* as `SpeculativeConfig::max_window` (γ_max, RFC-0001 §5.4); `T_max = max_window + 1` positions. KV provisional slots and the `Verify { T_max }` bucket are sized to it in `Warming`, and a delta may lower it but not raise it (RFC-0012 §7.1).
- Per window, the controller chooses `γ = argmax_{0 ≤ γ ≤ T_max−1} E(â, γ) / (t_draft(γ) + t_pass(γ+1))`, with `t_pass` from `PredictedCost::pass_by_window` scaled by the instance's measured-to-predicted ratio, and `â` a Beta posterior seeded (weight 16 windows) from the plan's learned acceptance for the drafter and request kind (greedy or sampled; RFC-0012 §11, which drops the task-class dimension in v1).
- Drafting of a window stops at the first proposal below `speculative.min_confidence` (0.5) or rejected by the grammar.
- After 64 consecutive windows with γ = 0, drafting stops and a γ = 1 probe window runs every 256 tokens.
- Realised throughput below 0.85 × the controller's prediction over 1,000 windows raises workload drift so the solver can re-size T_max by a plan delta.

## Consequences

- Positive: speculation adapts to the user's task without re-plans; captured graphs and KV stay within plan budgets; drafting never runs at a loss for long.
- Negative: the controller relies on the plan's `t_pass(T)` table being accurate, which ties speculation quality to SC-30; the posterior's prior weight (16 windows) is a guess.
- Work created: RFC-0011 publishes `pass_by_window`; RFC-0012 records per-window acceptance for `WorkloadStats::acceptance`.

## What would make us revisit

- Measured γ choices on the reference rigs vary by less than 1 across tasks. Then fix γ per plan and drop the controller.
- Padding rows in `Verify { T_max }` measurably slow GPU kernels (more than 3% of step time at γ well below γ_max). Then capture a second, smaller verify bucket.
- A learned or per-position confidence threshold beats the fixed 0.5 by more than 10% in accepted tokens per second.
