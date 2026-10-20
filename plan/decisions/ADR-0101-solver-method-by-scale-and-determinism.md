# ADR-0101: Solver method by scale, ring structure without slots, and deterministic work limits

- Status: Accepted (revised 2026-10-19 and 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the placement and scheduling area)
- Related: ADR-0008, ADR-0006, ADR-0009; RFC-0001 §4.2 rule 5; RFC-0011 §5–§10; SC-31

## Context

ADR-0008 fixed a joint MILP in HiGHS with a DP fallback. Three choices inside that decision are contestable and shape the implementation:

1. **How ring order enters the model.** The obvious formulation uses stage slots: binaries for "node n is in slot s" and "layer l is in slot s", whose products give "layer l on node n". At 64 blocks, 16 placement nodes and 8 slots that is about 8,000 extra binaries plus linking rows. Cycle enumeration is ruled out (16,064 cycles at 8 nodes, 1.1 million at 10; `multi-device-inference.md` §3.2).
2. **When to run the MILP at all.** The model for an 8-node MoE cluster has about 2,350 binaries (more for small-pool MoE models), about 33,000 continuous columns and 66,000 rows after the builder's reductions (RFC-0011 §5, estimate; the first count of 12,000 and 25,000 was low by about 2.5×). A reported layer-window ILP solved in 10–12 ms, but it had `2M` integer variables and `1 + 3M` constraints for M devices (§3.3), two orders of magnitude smaller. SC-31 requires ≤ 2 s for 8 nodes and a feasible DP plan whenever one exists.
3. **How the search stops.** RFC-0001 §4.2 rule 5 requires the same inputs to give the same plan. HiGHS's `time_limit` is wall-clock, so a faster or busier coordinator would stop at a different incumbent. HiGHS exposes count-based limits (`mip_max_nodes`, `mip_max_leaves`, `simplex_iteration_limit`) and a `random_seed`, and runs single-threaded with `threads = 1`.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Slot-indexed MILP, wall-clock limit, MILP always | textbook formulation | ≈8,000 extra binaries; non-reproducible plans; no plan if the MILP stalls |
| B. Fixed ring order from a TSP heuristic, MILP per order | small MILPs | order and placement decoupled; the best order depends on which nodes hold which bytes |
| **C. Per-node contiguity by start indicators, successor binaries linked to block adjacency; bitmask DP over every subset and order as warm start and fallback; MILP only up to 8 candidate nodes; count-based work limits with a wall-clock safety net** | ring order emerges from layer order with #B · #P binaries (# for set size); the DP guarantees feasibility and seeds the MILP; reproducible | the DP's state space grows as 2^N, so N is capped at 8 after scoring; work limits need tuning per release |
| D. DP only | simplest; exact over contiguous plans under its stage cost | the greedy stage cost cannot trade KV against experts under the three-site max, cannot express islands or pins exactly, and ADR-0008 found coupled trade-offs first order |

## Decision

**Option C.**
- Each placement node holds one contiguous run of blocks (start indicators, `Σ_b π_{b,p} ≤ 1`). Successor binaries `f_{p,q}` are bounded below by block adjacency, including tail-to-head; positive hop costs make them exact. No slot variables, no cycle enumeration.
- The DP over (used subset, head, last node, next block) runs first for every solve; it is exact for contiguous plans under a greedy in-node stage cost and returns a feasible plan whenever one exists (SC-31).
- *Revised 2026-10-20 (below):* the DP is the production method from M1, and the MILP exists only in builds where S-09's study passed the gate. Where it exists, the MILP runs when there are at most 8 physical candidate nodes after scoring and `solver.enable_milp` is set; it is warm-started with the DP plan (and on re-plans the current plan) through `Highs_setSolution`. Above 8 candidates the score prunes to 8.
- HiGHS runs with `threads = 1`, `parallel = off`, a fixed `random_seed`, `mip_rel_gap = 0.01`, and count-based stopping: `mip_max_nodes` and `simplex_iteration_limit`. The wall-clock `time_limit` is a safety net that, if it fires, flags the plan as time-limited and is treated as a tuning fault.
- Repairs after a stage loss do not use a fresh DP over survivors: `solve_repair` keeps survivors' shards and evaluates `RestartInPlace`, `Replace` and `Absorb` by time to resume (RFC-0011 §12.4, RFC-0033 §6); the survivors-only DP and MILP run only for `Resolve`. There is no switch between plans during recovery.
- The DP's transition is factored, giving ≈ 7.5 × 10⁷ relaxations for 8 Local nodes and ≈ 3.4 × 10⁸ with islands [estimate], so its 100 ms and 400 ms targets are reachable; the unfactored form was ≈ 10× over them.
- The count limits are set by spike S1 to bind at ≤ 50% of the wall-clock safety limit on the slowest Tier 1 coordinator; the first defaults could not bind before the clock.
- **The MILP gate (2026-10-20).** S-09's standalone harness (PLC-E5-T02, M0) runs a prototype of RFC-0011 §5–§8 on HiGHS and a prototype of the DP on the same coefficients over the reference inputs: RFC-0011 §15's four scenarios, the RFC-0061 solver suite's inputs and random fabrics of 1–8 nodes. The production MILP is built only if `(t_tok(DP) − t_tok(MILP)) / t_tok(DP) > 0.05` on at least one input, both scored by the evaluator. The per-input result is recorded here (RFC-0011 §9.5).

## Consequences

- Positive: plans are reproducible across coordinators and in tests; the system always has a feasible plan within ≈ 100 ms for 8 Local nodes and ≈ 20 ms on the 32-block repair path (targets); ring order is optimised jointly with placement.
- Negative: the work limits must be retuned when HiGHS is upgraded or the formulation changes, by the RFC-0061 solver suite. The 8-node cap means clusters larger than 8 candidate nodes use only the best 8 for one instance; replicas can use the rest.
- Work created: RFC-0061's solver suite (solve time, gap, DP-vs-MILP distance per reference input); RFC-0060 property tests for DP feasibility against brute force.

## What would make us revisit

- Spike S1 (RFC-0011 open question 1) shows the MILP cannot reach a 1% gap within 2 s on 8 nodes with a Kimi K3-class census even with the corridor re-solve. Then reduce `max_blocks`, drop the `χ` binaries, or move expert budgets to a second-stage LP.
- The DP lands within 5% of the MILP on every reference input (ADR-0008's own criterion). *Since 2026-10-20 this is checked before the MILP is built, and decides whether it is built at all.* After M3's islands and throughput objective land, the study is re-run, and a gain above 5% on a new reference input builds the MILP then.
- A HiGHS release makes wall-clock-limited solves deterministic, or adds a deterministic work-unit limit. Then switch to it.
- Users routinely run more than 8 capable nodes for one model. Then replace the bitmask DP with an order-heuristic DP and revisit the cap.

## Revisions

- **Revised 2026-10-20.** Option D's "DP only" becomes the M1 production method, and option C's MILP becomes conditional on S-09's study (Decision, last bullet). Option D's stated weaknesses are now claims to measure: the study checks whether the greedy stage cost loses more than 5% to the joint model on any reference input, including the KV-against-experts trade and pins. Until the gate passes, HiGHS is only a test-only dependency of the DP's feasibility property test, and the determinism rules for HiGHS in this ADR apply only to builds that contain the MILP. The DP's determinism (pure arithmetic in a fixed order, RFC-0060 rules T3 and T4) is unchanged. Spike S1 (S-09) moves from M1 to M0 as a standalone harness, so the answer exists before M1's solver work starts.
