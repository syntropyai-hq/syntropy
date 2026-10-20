# ADR-0008: Placement solver

- Status: Accepted (revised 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D1, D2, D3), ADR-0003, ADR-0005, ADR-0006, ADR-0007, ADR-0009, ADR-0010, ADR-0017, RFC-0010, RFC-0011, RFC-0012, RFC-0022, RFC-0023, RFC-0033, RFC-0054

## Context

**Placement quality dominates everything else at home scale.** A published home-cluster study's per-device layer-window ILP ran Llama 3-70B 3.3× faster than a memory-proportional split and 1.6× faster than a compute-proportional one. Without the solver the model ran at 20,848 ms/token; with it, 674 ms. Prefetching was worth only 9–17% (`plan/research/multi-device-inference.md` §0 item 2, §3.3). The ILP solved in 10–12 ms on 4–32 devices with HiGHS [reported by a surveyed system].

**No surveyed solver covers Syntropy's problem.** None models per-device VRAM/RAM/NVMe residency, the network and MoE expert placement together (§3.7, §14 gap 1). The study's released code sets the communication coefficient to 0.0 and comments out the memory-bandwidth term. It also imposes an equal-window constraint the runtime does not need (§3.3, §4.2, §13 items 7 and 23).

**The cost model is known** (`hardware-and-performance-model.md` §3, §5):
- Latency objective: `t_tok ≈ Σ_tiers B/W_eff(exec) + Σ_hops (α + S/β) + t_fixed`.
- MoE tiers add `(1 − h)·B_routed/W_nvme` with the hit rate `h(c)` learned online, plus the PCIe and CPU terms of the execute-in-place versus stream choice (§3.2, §3.3).
- The throughput objective is a different placement: balance stage times with bytes in proportion to `W_eff`, account for MoE expert-union growth (8 tokens touch 7.2× the routed bytes of one at DeepSeek-V3 top-8), and admit up to the KV limit (§3.7, §3.9, §3.11).
- First-order inputs, ranked: usable capacity per (device, tier); `W_eff` of the real kernel (0.24–0.8 of peak); per-tensor bytes at mixed dtypes; KV bytes; per-link `α_eff` with p99; sustained NVMe rate; the hit-rate curve (§5.1). Wi-Fi hops are costed at p99 in latency mode (§5.4).

**Accounting gaps cause real failures:** no KV or activation reserve, TP checked only against the cluster sum, and embeddings and LM head duplicated on every pipeline stage but not counted (`multi-device-inference.md` §3.9, §13 item 6).

**Search must scale.** Enumerating every simple cycle of the topology grows from 16,064 cycles at 8 nodes to 1.1 million at 10 (§3.2). In contrast, a DP with water-filling plus a separate per-request routing DP ran at 6.63 ms per request at 256 GPUs (§3.5). Filter→score front ends prune candidates cheaply (§2.5). Rebalancing with a 20% gain threshold avoids churn (§3.6). A re-shard can move hundreds of GB: 377 GB at 3.5 GB/s is ≈108 s (`hardware-and-performance-model.md` §5.3).

**Device selection matters.** On a 10-node testbed the solver used three devices and beat "fill the fastest GPUs first" (146 against 156 ms/token); dropping a weak device beats adding a hop (`multi-device-inference.md` §3.3, §3.4).

**Model scale.** DeepSeek-V3 has 61 × 256 routed experts; Kimi K3 has 93 × 896 (`backends-and-portability.md` §4.1). Per-expert integer variables would make the MILP needlessly large; the expert cache handles individual experts at runtime (ADR-0007).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Proportional heuristics (memory, compute) | Trivial | 3.3× slower at 70B; ignore tiers and links |
| B. Separate solvers per concern (layers, then tiers, then experts, then KV) | Each piece small | Misses coupled trade-offs: KV competes with experts for VRAM; a hop is worth it only if it moves bytes to faster memory |
| C. Exhaustive enumeration of rings or cycles | Exact for tiny clusters | Combinatorial explosion past 8 nodes |
| **D. One joint MILP in HiGHS, with a DP fallback, a filter→score front end and two-phase solving** | Couples every first-order term; exact solvers are cheap at home scale | MILP modelling and solver dependency |
| E. Commercial MILP solvers | Faster on hard instances | Licence incompatible with an embedded MIT/Apache binary |

## Decision

**Option D.**

1. **Inputs**: the resource graph with capability profiles (RFC-0010); the model manifest and tensor census (ADR-0003, ADR-0005); policy (objective, exact or approximate mode, user pins, budgets; RFC-0054); and the current plan, for migration cost.
2. **Filter→score front end.** Filter out devices that lack required ops or dtypes (ADR-0002), devices with less than a minimum useful slice, unreachable or untrusted nodes, and version mismatches. Score survivors by `W_eff`, capacity and bytes already held locally. Only the pruned set enters the MILP.
3. **One joint MILP** over devices × tiers × links × experts × device selection, solved by **HiGHS**, embedded and statically linked through Rust bindings (MIT), in every build. *Revised 2026-10-20 (below): M1 ships the DP of item 5 with its greedy in-node stage cost as the production solver, and this MILP is built only if a simulation study shows it more than 5% better than the DP on a reference input. The formulation below stays the specification of every term.* The model contains:
   - binary selection per device; ordered stage slots with contiguous layer ranges at legal boundaries; ring order from stage-order variables, never from cycle enumeration;
   - home tier and compute site per dense tensor group, with execute-in-place as the default and stream variables per pass class (ADR-0006);
   - **expert groups**: per MoE layer, experts bucketed by profiled popularity, with group bytes per tier on the owner device and hit rates from a piecewise-linear fit of the learned `h(c)` curve;
   - KV bytes per tier per stage from the manifest formula × context × concurrent sequences;
   - parallelism mode per stage, with cross-node TP/EP only inside islands that pass the ADR-0006 gate;
   - per-device capacity constraints after OS reserve, pinning and wired limits, scratch, runtime overhead, and duplicated embedding and LM-head bytes on the head and tail stages. Every node is checked individually, never only the cluster sum.
4. **Objectives.** *Revised 2026-10-20 (below): the latency objective is subject to a TTFT cap, wireless hops are costed at their measured mean with a stall-rate bound, and departure-prone nodes need a 10% gain.* The default, per D2, is batch-1 decode critical-path time: per-tier `B/W_eff`, per-hop `α + S/β` (p99 for Wi-Fi until 2026-10-20; now the mean), PCIe terms for streaming and the miss split, the expected NVMe miss term at the sustained throttled rate, KV reads, `t_fixed` per stage, and the speculative window's expected tokens per pass (ADR-0017). The alternate throughput objective minimises the bottleneck stage time at the chosen batch, with expert-union growth and KV admission. In interactive mode, dense weights on `Nvme` are allowed only when no other feasible plan exists, and the user is told.
5. **DP fallback.** A solver-free DP over contiguous layer ranges, with greedy tier filling by `W_eff`, provides the warm start, the answer when HiGHS finds no incumbent within its time limit, and a plan in builds or environments where the MILP is disabled. *Revised 2026-10-20: the DP is the production method from M1 (below).*
6. **Two phases.**
   - **Slow re-placement** runs between requests: on model load, membership change, a policy change, or a measured term drifting more than 20% for longer than a window. A new plan is adopted only if its predicted gain is at least 20% after amortising its migration bytes over the configured horizon. It never runs mid-sequence except on node loss (RFC-0033).
   - **Fast per-request routing** runs on the ingress node over its replica of cluster state. It picks the instance or replica, the speculative window and the prefill chunking, and never moves weights.
7. **Plans are versioned by plan epoch** and published as cluster events (ADR-0009). Each plan carries its predicted per-stage and per-term times. The runtime compares them with live timings to refine profiles (RFC-0010), and the dashboard shows both.
8. **User overrides are constraints**, not post-edits: per-tensor or per-device pins enter the MILP. An infeasible pin set is reported as such.

## Consequences

Positive:
- Every first-order cost term from the research is in one model, including the network term and accounting the surveyed solvers dropped.
- Device selection, tiers, experts and KV are traded against each other directly.
- Predicted-versus-measured times make the cost model testable in production.

Negative:
- MILP modelling skill is needed, and solve time on large MoE models must be watched.
- Expert groups approximate per-expert placement; within-group placement is left to the runtime cache.
- Two code paths (MILP, DP) must agree on feasibility semantics.

Work created: RFC-0011 (formulation, linearisation, time limits, group bucketing, hysteresis), RFC-0010 (profile inputs and online refinement), RFC-0012 (fast routing), RFC-0022 and RFC-0023 (KV and expert-group semantics), RFC-0033 (re-placement on node loss), RFC-0054 (policy and pins), RFC-0061 (solver regression suite).

## What would make us revisit

- On the RFC-0061 benchmark suite, the DP fallback comes within 5% of the MILP objective and measured decode in every case. The MILP would then be optional. *Since 2026-10-20 this test runs first, in S-09's study, and decides whether the MILP is built (below).*
- Slow-phase solves exceed their time limit on realistic inputs (8 nodes, a Kimi K3-class model) after formulation work.
- Median predicted-versus-measured `t_tok` error stays above 20% after online refinement. A cost term is missing or wrong.
- Simulation shows expert-group aggregation losing more than 10% decode against per-expert assignment.

## Revisions

- **Revised 2026-10-20.**
  - Items 3 and 5: the DP with its greedy in-node stage cost is the production solver from M1. The joint MILP is built only if a simulation study in spike S-09's harness (PLC-E5-T02, M0) shows it beats the DP by more than 5% on a reference input: RFC-0011 §15's worked scenarios, the RFC-0061 solver suite's inputs, or random fabrics of 1–8 nodes, each scored by the one evaluator. Until then HiGHS is not linked into shipped binaries, and the shadow prices, delta solve and streaming-home pass use the greedy's closed forms (RFC-0011 §9.5). Reason: the evidence in Context shows that measured placement beats proportional splits, not that a MILP beats a good greedy (the device-selection gain over fill-fastest-first was 6.4%, within overlapping error bars), and every worked example is solved by filling the fastest memory first. The first revisit trigger now runs before the MILP's cost is paid. The MILP-shaped interfaces (`SolveInput`, `PredictedCost`, the explanation types) stay, so building it later redesigns nothing.
  - Item 4, TTFT: latency mode minimises decode subject to a TTFT cap of 2× the best candidate's TTFT at the workload's p90 uncached prompt, and the preview shows both numbers (RFC-0011 §7.1). The cap binds from M2. Reason: on S2 the decode-only objective picks a plan whose 4K-prompt TTFT is ≈ 22 s against 3.6–5.9 s on the PC alone [estimate: RFC-0032's chunked-prefill table and RFC-0011 §15.1].
  - Item 4, wireless links: `WiFi` and `Overlay` hops are costed at their measured mean for prediction and for the objective, a stall-rate constraint bounds their tail (RFC-0011 C14), plans with such hops carry a jitter badge, and p99 stays only in the TP/EP gate (ADR-0006). Reason: charging p99 on every token overstates the expected hop about 20× (≈ 12 ms mean against 250 ms p99 [estimate, cost model §2.5 quantiles]) and keeps Wi-Fi nodes out even when the model fits only across them.
  - Device selection: a departure-prone node (a laptop, a node on battery, or one reached only over wireless links) hosts a stage only for a predicted gain of ≥ 10% [target] over the best plan without such nodes, or when capacity needs it (RFC-0011 §4). A stall from a departing node costs every stream more than a few percent of speed is worth. Once alpha data gives departure rates, the expected departure cost moves into the objective instead.
