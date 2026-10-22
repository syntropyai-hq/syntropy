# RFC-0011: Joint placement solver

- Status: Draft
- Area: placement and scheduling
- Depends on: RFC-0001, RFC-0010, RFC-0023 (§3, §13), RFC-0060 (§2.4), ADR-0101, ADR-0102, ADR-0103, ADR-0006, ADR-0007, ADR-0008, ADR-0009, ADR-0010, ADR-0017, ADR-0019
- Date: 2026-10-19
- Revised: 2026-10-20: the DP with its greedy stage cost is the M1 method and the MILP sits behind a study (§9.5); a TTFT cap (§7.1); wireless hops at the mean with a stall bound (C14); a departure margin (§4); streaming at decode where measured faster (§3.1); an approximate-knob line in the preview (§14.1); open questions 5 and 8 answered; 2026-10-22: a note on §5's `c^coll`, which reads α for `α_eff` and gives the full mesh the one-shot form `α + S/β`

## Summary

The `syntropy-solver` crate turns a `ResourceGraph` (RFC-0010), a `ModelManifest` with its tensor census, an `InstancePolicy` and, on re-plans, the current plan into a `PlacementPlan` (RFC-0001 §5.4). It decides, in one joint model:
- which nodes and devices take part, and in what ring order;
- each node's contiguous layer range (its stage);
- for every dense tensor group, its home tier and compute site, including when it streams over PCIe above n*;
- for every MoE layer, how many of its routed-expert bytes sit in VRAM, in RAM and on NVMe, using the learned hit-rate curve, and how RAM-served misses split between in-place CPU compute and PCIe copies;
- the KV budget per tier per stage, including the duplicated embedding and LM-head bytes, scratch, and reserves;
- the parallelism mode per stage, with cross-node tensor or expert parallelism only on islands that passed the measured gate.

The default objective is the batch-1 decode critical path (D2), subject to a time-to-first-token cap of 2× the best candidate's TTFT; wireless hops are costed at their measured mean, with a stall-rate bound for their tail. The throughput objective minimises the bottleneck stage at a chosen micro-batch subject to a per-sequence floor. **The production method from M1 is a solver-free DP over contiguous layer ranges with a greedy in-node stage cost (§10).** The mixed-integer linear program of §5–§9, solved with HiGHS, is built only if S-09's simulation study shows it more than 5% better than the DP on a reference input (§9.5); until then §5–§9 are its formulation and the reference for the DP's coefficients, and the DP would be its warm start. A filter→score front end prunes devices before either. Solves are deterministic: the DP is pure arithmetic in a fixed order, and a built MILP runs HiGHS single-threaded with a fixed seed and count-based work limits, with wall-clock limits only as a safety net. Slow re-placement targets ≤ 2 s for up to 8 nodes, and fast per-request routing ≤ 1 ms (SC-31). Every plan carries a per-term prediction and an explanation that says why each tensor group is where it is.

## Motivation and goals / non-goals

The surveyed placement methods fail in ways the cost model predicts. Memory-proportional splits ran a 70B model 3.3× slower than a cost-based solver; partitioning mattered an order of magnitude more than prefetching (`multi-device-inference.md` §3.3). The one published tier-aware multi-device ILP zeroed its communication coefficient in released code and imposed an equal-window constraint the runtime did not need (§3.3, §13 items 7 and 23). Cycle enumeration grows to 1.1 million cycles at 10 nodes (§3.2). Accounting omitted KV, activations and the duplicated embedding and LM head, and tensor-parallel placement checked only the cluster total (§3.9). No surveyed solver models per-device VRAM/RAM/NVMe residency, the network and MoE expert placement together (§14 gap 1). ADR-0008 decided on one joint model of all these terms; this RFC is its formulation. Its 2026-10-20 revision ships the DP and its greedy stage cost as the production solver in M1: the cited studies show that measured placement beats proportional splits, not that a MILP beats a good greedy, and every worked example in §15 is solved by filling the fastest memory first.

**Goals**
1. Encode every first-order term of the cost model (§5.1 items 1–7) in one model, including the network term, per-tier W_eff of the real kernel, KV, the NVMe miss term at the sustained throttled rate, and duplicated embedding and head bytes.
2. Check every capacity per node and per tier, never only the cluster sum.
3. Choose devices: drop a weak device when that beats adding a hop (`multi-device-inference.md` §3.3–§3.4).
4. Meet SC-31: slow re-placement ≤ 2 s for up to 8 candidate nodes including expert variables, routing ≤ 1 ms, and a feasible DP plan whenever a contiguous plan exists.
5. Be reproducible: the same inputs give the same plan on any machine (RFC-0001 §4.2 rule 5).
6. Predict each term of `t_tok` for drift detection (RFC-0001 §7.4) and SC-30 (±15%).
7. Explain each decision in terms a user can act on: what a GB of VRAM is worth here, why a device is unused, what a smaller quant would buy.

**Non-goals**
- Per-expert placement. Individual experts move through the runtime cache (RFC-0023); the solver sets byte budgets per layer and tier.
- Kernel choice per op, which `Warming` makes by microbenchmark (RFC-0001 §5.7).
- Runtime scheduling decisions inside a plan: step composition, speculative window adjustments, the live miss split (RFC-0012, RFC-0023).
- Recovery mechanics after a stage loss (RFC-0033). This RFC supplies the repair solve that recovery calls (`solve_repair`, §12.4).
- Public-swarm scale. Beyond 8 candidate nodes the MILP is not run (§9.4).

## Design

### 1. Pipeline and phases

```text
            slow phase (coordinator, between requests)                        fast phase (every ingress)
 graph ─┐
 manifest ─┼─► filter ─► score/prune ─► preprocess ─► DP (warm start, fallback)   cluster state replica
 policy ─┤                                   │              │                         │
 current ┘                                   ▼              ▼                         ▼
                                        MILP (HiGHS) ◄── incumbent            route(request) → instance,
                                             │                                 prefill chunking, initial γ
                                             ▼
                          speculative outer loop (re-solve at T*)  ─►  round ─► validate (exact evaluator)
                                             │                                   │
                                             ▼                                   ▼
                                     hysteresis + payback (re-plans) ─► PlacementPlan + PlanExplanation
                                                                        ─► ClusterEvent::PlanCommitted
```

The MILP box runs only in builds where §9.5's gate has passed. Otherwise the DP's plan goes straight to the speculative outer loop, which then re-runs the DP at T* instead of the MILP.

| Phase | Runs on | When | Budget [target] | Output |
|---|---|---|---|---|
| Slow re-placement | coordinator, in the `ctl` runtime's blocking pool | model load (`EnsureInstance`), membership change, policy change, profile drift, workload drift (RFC-0001 §7.2) | ≤ 2 s up to 8 nodes (SC-31); a repair solve after a stage loss returns all its candidates within 2 s (§12.4, RFC-0033 §6.4) | a new `PlacementPlan` with a new `PlanEpoch`, or "keep the current plan" |
| Delta solve | coordinator | budget-only changes: KV vs expert cache split, miss split, prefetch windows, speculative settings, batching limits | ≤ 50 ms | a `PlanDelta` that runners adopt in place (RFC-0001 §5.4) |
| Fast routing | every ingress daemon | every request | ≤ 1 ms (SC-31) | `RouteDecision` |

The solver is a pure library (RFC-0001 §4.2 rule 5): no I/O, no clocks except HiGHS's optional safety timer, no threads beyond HiGHS's single thread. The coordinator calls it and commits the result through the event log.

### 2. Inputs and outputs

```rust
/// Everything a slow-phase solve reads. Borrowed from the coordinator's ClusterState.
pub struct SolveInput<'a> {
    pub graph: &'a ResourceGraph,
    pub manifest: &'a ModelManifest,
    pub policy: &'a InstancePolicy,               // RFC-0001 §11.3: objective, exactness, pins, overrides
    pub current: Option<&'a PlacementPlan>,       // present on re-plans
    pub workload: &'a WorkloadStats,              // learned hit curves, acceptance, context mix
    pub reason: SolveReason,                      // Load | Replan(ReplanReason) | Repair { constraints } | Delta (§12.4)
    pub limits: SolveLimits,
}

/// Learned statistics the solver uses instead of priors once they exist.
pub struct WorkloadStats {
    /// Per MoE layer: the runtime cache policy's achieved hit fraction of routed reads as a
    /// function of resident bytes in fast memory, H_l(c), where c counts the VRAM and RAM caches
    /// as exclusive sets for NVMe-homed experts (RFC-0023 §3, §13). Concave, piecewise linear,
    /// H_l(0) = 0, H_l(pool) = 1. Prior: H_l(c) = c / pool_l (uniform routing, cost model §3.3).
    /// Contents owned by RFC-0023 §13.
    pub hit_curves: Vec<HitCurve>,
    /// Expected distinct experts per layer for one pass of T tokens, in two shapes (RFC-0023 §13):
    /// `window` for T consecutive tokens of one sequence (verify windows) and `batch` for T tokens
    /// from different sequences (throughput micro-batches). Prior for both: the uniform-routing
    /// formula E · (1 − (1 − k/E)^T) (cost model §3.9).
    pub union: Vec<UnionCurve>,
    /// Measured hit rate per expert-cache tier (`ExpertCacheDigest::tier_hit`), compared with the
    /// curve at that tier's budget (§12.1, RFC-0010 §7.1).
    pub tier_hit: Vec<(TierId, LayerIx, Measured<f32>)>,
    /// Per (drafter, model) and request kind (greedy or sampled): acceptance a, from RFC-0012's
    /// `AcceptanceReport`; v1 has no task-class dimension (RFC-0012 §11). Prior 0.7 (ADR-0017 item 5).
    pub acceptance: Vec<AcceptanceReport>,        // per (drafter, model), greedy and sampled (RFC-0012 §11)
    /// Context length distribution of live and recent sequences (p50, p90), for KV sizing, and
    /// the p90 of uncached prompt tokens per request, the TTFT cap's reference length (§7.1).
    pub context: ContextStats,                    // RFC-0012 §11; prior p90 = solver.context_target (8192)
    pub concurrency: ConcurrencyStats,            // RFC-0012 §11; throughput mode sizing
}
pub struct HitCurve { pub layer: LayerIx, pub pool: Bytes, pub points: SmallVec<[(Bytes, f32); 9]>, pub source: MeasureSource }

pub struct SolveLimits {
    pub max_nodes: u16,                           // candidates after scoring; default 8
    pub mip_node_limit: u32,                      // deterministic work limits (§9.2)
    pub simplex_iteration_limit: u64,
    pub safety_time_limit: Nanos,                 // wall clock; default 2 s; a repair solve's `Resolve` gets what remains of §12.4's 2 s
    pub mip_rel_gap: f64,                         // default 0.01
}

pub struct SolveOutcome {
    pub plan: Option<PlacementPlan>,              // None: infeasible (see `infeasible`) or "keep current"
    pub keep_current: bool,                       // re-plan did not pass hysteresis and payback
    pub explanation: PlanExplanation,             // §14
    pub infeasible: Option<InfeasibilityReport>,  // §17
    pub stats: SolveStats,                        // which path won, gap, nodes, iterations, wall time
}

pub fn solve(input: &SolveInput) -> SolveOutcome;
pub fn solve_delta(input: &SolveInput, current: &PlacementPlan) -> Option<PlanDelta>;
pub fn solve_repair(input: &SolveInput, current: &PlacementPlan,
                    c: &RepairConstraints) -> Vec<(RepairStrategy, PlacementPlan, Nanos /*t_resume*/)>; // §12.4
pub fn preview(input: &PreviewInput) -> Preview;                                          // §14.1
pub fn route(state: &RoutingView, req: &RouteRequest) -> Result<RouteDecision, Error>;   // §13
pub fn evaluate(plan: &PlacementPlan, input: &SolveInput) -> Evaluation;                 // §11.3
```

**Additions to RFC-0001 §5.4 types** (owner-permitted, serde defaults):

```rust
/// Added to PlacementPlan.
pub struct PlacementPlanExt {
    pub transition: TransitionStrategy,           // MakeBeforeBreak | BreakBeforeMake (RFC-0001 §7.3)
    pub explanation: Option<Blake3>,              // digest of the PlanExplanation, fetched over bulk
    pub solve: SolveStats,                        // path (Milp | MilpIncumbent | Dp), gap, wall time
    pub degraded: Option<DegradedReason>,         // DenseOnNvme | MultiStageOnNode | AbsorbedStage
    pub move_bytes: Bytes,                        // bytes the transition must fetch or re-pack
    /// What the plan relies on, from the conformance manifest (RFC-0060 §6), so ingress and the
    /// scheduler can enforce and label it without asking runners.
    pub certification: PlanCertification,
    pub rdma_qps: SmallVec<[(NodeId, u16); 8]>,   // queue pairs the plan takes per node
    /// Groups stored once per reader layout instead of one shared layout (RFC-0040 §5.4; §11.2).
    pub per_site_layout: Vec<TensorGroupId>,
    /// Each measured node's WAN rate class, ⌊log2⌋ of its origin download rate in MB/s as this
    /// solve read it; unmeasured nodes have no entry (RFC-0041 §8.3; §11.2).
    pub wan_rate_class: SmallVec<[(NodeId, u8); 8]>,
}
pub struct PlanCertification {
    pub status: CertStatus,                       // Certified | Derived (pairwise inference) | Uncertified
    /// Certified run-time values: the runtime may use these and nothing beyond them (RFC-0012 §10.5).
    pub runtime: CertifiedRuntime,                // { prefix_reuse: bool, kv_spill_tiers: SmallVec<[TierKind; 3]>,
                                                  //   max_concurrency: u32, max_window: u16 }
    pub relies_on: Vec<CertRef>,                  // the OpCert, WireCert, FamilyCert and PathCert IDs used
}

/// RFC-0001 names PredictedCost; this RFC defines it. Per stage and per term, in the units
/// the runners' telemetry reports (RFC-0001 §13, §14).
pub struct PredictedCost {
    pub t_tok: Nanos,                             // per accepted token, speculation included
    pub t_pass: Nanos,                            // one ring traversal at the plan's verify window
    pub tokens_per_pass: f32,                     // E[tokens] = (1 − a^(γ+1)) / (1 − a)
    pub ttft: SmallVec<[(u32, Nanos); 4]>,        // TTFT per uncached prompt length; default {1K, 4K, 32K}
    /// TTFT at the cap's reference length L_ttft (§7.1) and the cap it was checked against.
    pub ttft_ref: (u32, Nanos),
    pub ttft_cap: Option<Nanos>,                  // None in M1 (reported only) and in throughput mode
    /// Present when the ring has a WiFi or Overlay edge: the stall rate of C14 and the p99 token
    /// time, which the views show as a jitter badge next to the mean.
    pub jitter: Option<Jitter>,                   // { stall_rate: f32, t_tok_p99: Nanos }
    pub context_length: u32,                      // longest context the plan holds KV for (§11.2, RFC-0054)
    /// t_pass(T) for T = 1 ..= SpeculativeConfig::max_window + 1, for RFC-0012's window controller.
    pub pass_by_window: SmallVec<[(u16, Nanos); 9]>,
    pub throughput: Option<f32>,                  // tok/s at the plan's batching, throughput mode
    pub stages: Vec<StageCost>,
}
pub struct StageCost {
    pub stage: StageIx,
    /// Keyed by RFC-0053's `CostTerm` (§2.2 there), the one key space for prediction and
    /// measurement: DenseVram, DenseUnified, DenseHostCpu, ExpertHitGpu, ExpertHitCpu (including
    /// the in-place share of misses), ExpertMissCopied, NvmeDemandWait, KvRead, RowLookup, Switch,
    /// Collective, HopSend, HopWait, CreditStall, Fixed, Sampler, ConstraintMask, Draft. Terms are
    /// critical-path attributions and sum to `StageCost`'s step time: where sites run in parallel
    /// (a MoE block's GPU, CPU and NVMe work, C8; KV's GPU and CPU-split parts, C7) the block's
    /// time is attributed to the binding site's term.
    pub terms: SmallVec<[(CostTerm, Nanos); 18]>,
    /// Each non-binding site of every parallel block, under its own term, summed over the step's
    /// blocks; not part of the step time. RFC-0053 §2.2 attributes measured time the same way,
    /// per RFC-0012's `ParallelBlock`s, so predicted and measured compare within each set.
    pub overlapped: SmallVec<[(CostTerm, Nanos); 4]>,
    /// The largest predicted per-layer CPU in-place time; RFC-0040 derives
    /// `kernels.spin_kernel_max_us` from it.
    pub cpu_layer_max: Nanos,
    pub bytes: SmallVec<[(TierId, Bytes); 6]>,    // predicted read bytes per token per tier (SC-11)
    pub resident: SmallVec<[ResidentBytes; 6]>,   // predicted peak resident bytes per tier, by purpose
    pub prefill_per_chunk: Nanos,                 // one prefill chunk of BatchingLimits::prefill_chunk
}

/// RFC-0001 names BatchingLimits; this RFC defines it.
pub struct BatchingLimits {
    pub max_sequences: u32,                       // KV admission ceiling for the instance
    pub prefill_chunk: u32,                       // tokens per chunk (§11.2)
    pub micro_batch: u16,                         // throughput mode: sequences per stage step; 1 in latency mode
    pub prefill_ladder: SmallVec<[u32; 5]>,       // captured prefill buckets (RFC-0012 §2)
}
pub struct ResidentBytes {
    pub tier: TierId,
    pub before: Bytes,                            // held before this plan (other instances, the old epoch)
    pub after: Bytes,
    pub by_purpose: SmallVec<[(Purpose, Bytes); 8]>,  // RFC-0001 §9.1/§9.3: ContextReserve, Scratch,
                                                  // DenseWeights, Kv, KvReserved, ExpertCache, Staging,
                                                  // NetPools, History, Headroom
}
```

The verify window is `SpeculativeConfig::max_window` = γ_max as RFC-0001 §5.4 defines it; `T_max = max_window + 1` is derived wherever it is needed, and no second field stores it.


### 3. Preprocessing: placement nodes, blocks and coefficients

Preprocessing turns graph and census into the small, dense problem the DP and the MILP need. Everything here is a pure function of `SolveInput`.

#### 3.1 Placement nodes

A **placement node** p is one way a physical node, or a fast island of nodes, can host one stage. Each has a mode and a set of execution pairs.

| Kind | Members | Mode | Created when |
|---|---|---|---|
| `Local(n)` | one node | `ParallelismMode::Local` | every node that survives the filter |
| `IntraTp(n, G)` | one node, GPU set G with \|G\| ≥ 2 | `IntraNodeTp` | G's peer copies verified (`HostLinkCaps::p2p`, ADR-0006 item 5) and the family has a TP sharding strategy |
| `Island(I, Tp)` | node set I, \|I\| ∈ {2, 3, 4} | `CrossNodeTp` | every pair in I passed the gate: measured `alpha_eff` mean and p99 below `α_eff,max(I)` (RFC-0010 §3.5, ADR-0006 item 2) |
| `Island(I, Ep)` | node set I | `CrossNodeEp` | same gate, MoE family |

Exclusivity: a physical node takes part in at most one selected placement node (§6, C2). With up to 8 candidate nodes and islands limited to the pairs that passed the gate, |P| stays ≤ 16 in practice; preprocessing keeps at most `solver.max_placement_nodes` (default 16), dropping islands with the lowest score first.

**Execution pairs.** For each placement node, the set K_p of (compute device, home tier) pairs on which a dense block can execute in place:

| Pair | Rate used at pass size T |
|---|---|
| (GPU d, `Vram` d) | `curve(d, Vram, class, scheme, T)` |
| (iGPU, `Unified`), (Apple GPU, `Unified`) | `curve(d, Unified, …, T)` |
| (CPU c, `HostPinned` of c's NUMA domain) | `max(curve(c, HostPinned, …, T), stream(T))`, where `stream(T) = min(h2d_pinned, curve(GPU, Vram, …, T))` for the node's best GPU: the pass streams when that is faster, which is the n* rule applied per pass (ADR-0006 item 7). This holds at T = 1 as well: where the probe measures streaming faster than execute-in-place at decode, n* = 1 and the solver streams at decode, automatically, from measurement. D1 describes a reference design, not a ban on streaming (OQ-PL-05) |
| (CPU c, `HostPageable` of c's NUMA domain) | `curve(c, HostPageable, …, T)`. No streaming: a pageable home is never a DMA source (RFC-0020 §2, ADR-0200). Only groups that are never copied stay here: no `StreamSite` and no copied-miss share. A dense group that a stream site would stream moves to `HostPinned` in C13's second stage, as far as the pin budget allows |
| (CPU, `Unified`) on unified machines | `curve(c, Unified, …, T)`; chosen only when it beats the GPU, e.g. a kernel the GPU lacks |
| (virtual device, island or TP group) | `Σ_m curve(m, …, T) · (1 − ε_split)`, uneven split weighted by each member's rate where the family's sharding allows it, else even; ε_split = 0.03 for load imbalance [default, tuned by RFC-0061]. A virtual pair maps to one tier per member: `tier_n(k)` is member n's tier of the same kind on the member device that executes its share, and every byte the pair holds is charged to `tier_n(k)` scaled by `ω_{n,p}` (C9) |
| (CPU, `Nvme` via `HostPinned` staging) | `min(sustained_read, curve(c, HostPinned, …, T))`. Present only in the degraded second pass (§6, C10) |

All rates come from `QuantBw` curves (RFC-0010, ADR-0100). A missing entry uses the `Derived` value and adds an `Assumption`.

#### 3.2 Blocks

The family declares legal stage boundaries (ADR-0006 item 1, RFC-0043); a **block** is the run of layers between two consecutive legal boundaries. When the model has more than `solver.max_blocks` (default 64) blocks, adjacent blocks merge pairwise until it fits. A 93-layer model with a cut after every layer becomes 47 blocks of ≤ 2 layers. After the solve, each stage boundary is refined at layer granularity (§11.1), so coarsening costs at most one block's imbalance per boundary before refinement and nothing after it.

Per block b, preprocessing computes:

| Symbol | Meaning | Source |
|---|---|---|
| `D^A_b`, `D^F_b` | dense bytes per quant scheme, split into the **mixer part** A (attention or linear-attention mixer, router, norms, hyper-connection weights: everything that reads KV or recurrent state) and the **FFN part** F (dense FFN, shared experts). `D_b = D^A_b + D^F_b` | census |
| `R_b(T)` | routed expert bytes read per pass of T tokens: `Σ_{l∈b} U_l(T) · e_l`, e_l the expert size. `U_l` is `UnionCurve::window` for one sequence's verify window (T = γ + 1), `UnionCurve::batch` for a throughput micro-batch (T = b_μ), and `batch(b_μ · T_w)` for b_μ sequences each verifying T_w positions, an upper bound | census, `WorkloadStats::union` (RFC-0023 §13) |
| `Π_b` | routed pool bytes | census |
| `H_b(c)` | block hit curve: the best hit fraction of `R_b` from c resident bytes spread over the block's layers. The sum of concave piecewise-linear curves under optimal allocation is their slope-sorted merge, computed once per block | `WorkloadStats::hit_curves` |
| `K^read_b(L)` | KV bytes read per pass at context L | `KvFormula::read_per_step` |
| `K^store_b(L, M)` | KV bytes stored for M sequences at context L | `KvFormula::stored` |
| `F_b(T)` | FLOPs per pass of T tokens, for compute-bound checks on the batch curve's flat end | hyperparameters |

Non-layer groups are handled separately: `RowLookup` tables (embeddings, n-gram and per-layer embedding tables), `Head` (LM head, MTP heads) and drafter groups.

#### 3.3 Coefficients

For block b on pair k at pass size T:
- dense time per part `δ^c_{b,k}(T) = Σ_{g ∈ part c of b} bytes_g / rate_k(class_g, scheme_g, T)` for c ∈ {A, F};
- expert time per byte at the hit tier `ε^V_{b,k}`, at RAM via CPU `ε^C_{b,n}`, via PCIe `ε^P_{b,n} = 1/h2d_pinned`, from NVMe `ε^N_{b,n} = 1/sustained_read` — the CPU rate is the `ExpertGemv` curve at `n = ⌈T·k/U(T)⌉` tokens per expert;
- KV read time per byte `λ_{k}` for the KV's tier as read by the layer's attention site: `1/rate(Vram)` when attention and KV share a GPU, `1/h2d_pinned` for KV in `HostPinned` read by a GPU (streamed KV, RFC-0022), `1/rate(HostPinned)` when the CPU runs attention.

The pass size T is fixed per solve (§7.3); latency-mode solves start at T = 1.

**Per-backend hand-off cost.** On CUDA and HIP a host-interacting layer costs `t_switch` (doorbell spin, RFC-0010 H2). On Metal and Vulkan, whole-step capture cannot include host waits, so each host-interacting layer instead costs the measured `t_wait` of RFC-0010's M5 probe (RFC-0012 §7.5). The solver uses `t_switch,p = max(t_switch, t_wait)` for the stage's backend in the `χ`, `ξ` and intra-stage hand-off terms, so it can prefer placements without CPU or NVMe misses on those backends. Unified stages with no CPU or NVMe miss site have no per-layer host wait.

**Scratch** per device, `σ_d`: activations for one prefill chunk at the chosen chunk size, the provider workspace bound for the family's ops (RFC-0040 publishes a per-DeviceClass bound used before `Warming`), captured graphs for each verify-window size, and two staging slots of the largest expert for copied misses. **Reserves** come from the graph: context reserve, `resource.vram_headroom`, `resource.ram_headroom` (RFC-0001 §9.3), I/O staging and network pools (a fixed share of the pinned budget, RFC-0020), and, on **every non-tail stage**, the activation history that stage keeps of its own outbound frames for replay: `hist_p = min(recovery.history_ram, M · L · h_model · streams · bytes(wire))` in `HostPageable` (RFC-0033 §4.1, §4.1.2: each non-tail stage keeps its own history, capped per runner at min(2 GiB, 5% of RAM) with disk spill beyond; never pinned). For scale: 4 K tokens at width 8192 in F32 is 128 MiB per sequence (SC-28). Because the reserve depends only on whether the stage is the tail, not on the stage count, a range's feasibility still depends only on the range and its position (§10).

### 4. Filter and score

**Filter.** A device or node is dropped, with the reason kept for the explanation (§14), when:

| Rule | Check |
|---|---|
| F1 Coverage | the device has no provider covering every op the family needs at the census's schemes (RFC-0040 publishes coverage per `DeviceClass` × family; the ggml provider counts) |
| F2 Health | `DeviceProblem` is `NoProvider`, `DriverTooOld`, `ProbeCrashed` or `Excluded`; the node's build is incompatible; the node is suspected by gossip |
| F3 Roles and labels | `RolePrefs::may_host_stages` is false; the policy's pins or label rules exclude it (RFC-0054) |
| F4 Power | the node is on battery and `ResourcePolicy::power` is not `Battery` or `Performance` |
| F5 Minimum slice | usable fast memory below `max(bytes of the smallest block + its KV + σ_d, solver.min_slice)`, default 1 GiB |
| F6 Reachability | no measured link to any other surviving node, unless the plan is single-node |
| F7 Exactness and certification | the instance is `Exact` and the device's only provider for some op has failed parity for that op (RFC-0042 records it), or the conformance manifest (RFC-0060 §6, proposed amendment A2) lacks cover for a **plan-time** dimension the candidate needs: an `OpCert` for the device class, a `WireCert` for a BF16 hop, a `FamilyCert` for a split or TP group, a `PathCert` for the placement shape. Runtime dimensions (prefix reuse, KV spill tier, concurrency) cannot be known at plan time; the plan records which of them lack cover, and RFC-0012 §10.5 and the tier manager enforce them per request |

**Score.** Survivors are ranked by the time they would save per token if they held the bytes they can hold:

```text
score(n) = Σ_{k ∈ K_n} min(C_k, B_model) · (1 / W_ref − 1 / W_k)⁺    − α_planning(ingress, n) · 1{wireless}
W_ref    = the slowest execution rate among survivors
```

with ties broken by bytes of the model already held locally (fewer bytes to fetch). The top `solver.max_nodes` (default 8) enter the DP (and the MILP where it is built); the rest are listed as "not considered" in the explanation. Score order also seeds the DP's node orders (§10).

**Departure margin.** A node is **departure-prone** when it has a battery (RFC-0010 `CapabilityProfileExt::battery_present`, so every laptop), is on battery power now, or reaches every other candidate only over `WiFi` or `Overlay` links. Such a node may host a stage only when the plan that uses it is at least `solver.departure_margin` (default 0.10 [target]) faster than the best plan that uses no departure-prone node, by the gain measure of §12.2 (`(t_tok(P₀) − t_tok(P)) / t_tok(P₀)`, relative tok/s in throughput mode), or when no feasible plan without departure-prone nodes exists (capacity). The DP's table already holds the best plan for every node subset (§10), so the check costs one minimum over the subsets without departure-prone nodes; where the MILP is built, it re-solves once with `v_p = 0` forced for them. A rejected node is listed in the explanation as `Unused` with its predicted gain and the reason. The margin is the first step: once alpha data gives departure rates per node class, the expected departure cost (rate × stall time) moves into the objective and replaces it. Filter F4 still removes nodes on battery outright unless `ResourcePolicy::power` allows them.


### 5. The MILP: sets, parameters and variables

**Sets**

| Set | Members |
|---|---|
| `N` | physical candidate nodes after filter and score, \|N\| ≤ 8 |
| `P` | placement nodes (§3.1); `phys(p) ⊆ N` its members; `prim(p)` its primary member |
| `B` | blocks in layer order `0 … β_L`; `B_moe ⊆ B` the blocks containing MoE layers |
| `K_p` | execution pairs of p (§3.1); `dev(k)`, `tier(k)` |
| `K^kv_p` | KV placements of p, each with an explicit **site** for its reads: (GPU, `Vram`) read at GPU memory speed; (GPU stream, `HostPinned`) read over PCIe at `h2d_pinned`; (CPU split, `HostPinned`) read by the CPU pool at `W_cpu,attn(kind, T)`, the KV bytes per second RFC-0010's M6 measures at T rows, in parallel with the GPU's part, merged exactly (RFC-0022 §5.3); and, for reserved sequences only, `HostPinned` or NVMe parking. KV's host tier is always `HostPinned`, never `HostPageable` (RFC-0022 §3). The plan records the site per stage and KV class as `KvBudget::spill_compute` (`GpuStream` or `CpuSplit`, RFC-0022 §5.3), costed with RFC-0022 §5.4's two canonical terms (C7), whose CPU rate is the measured `W_cpu,attn(kind, T)` and from which `F_cpu,attn` and `T*_kv` are derived; on Apple there is no spill path. For dense MLA the CPU split is compute-bound, so `GpuStream` wins unless the probe shows otherwise |
| `G_p` | GPUs (or the unified GPU) of p that can hold an expert cache |
| `I_n` | tiers of physical node n |
| `J_b` | segments of the block hit curve: `H_b(c) = min_{j∈J_b} (a_{b,j} + s_{b,j} · c)`, with `a_{b,1} = 0` |

**Parameters** (all from §3, at the solve's pass size T): `δ^A_{b,k}`, `δ^F_{b,k}`, `ε^V_{b,k}`, `ε^C_{b,p}`, `ε^P_p`, `ε^N_p`, `d_p` (DMA interference, seconds of CPU time per byte copied), `λ_k`, `D^A_b`, `D^F_b`, `R_b`, `Π_b`, `K^read_b`, `K^store_b`, `σ_d`, `t_fixed,d`, `t_switch,p`, `usable_i`, `reserve_i`, `c^hop_{p,q}`, `c^coll_{b,p}`, the non-layer group sizes (embedding `E`, LM head `Hd`, drafter bytes `Dr`), and on re-plans `cur_{b,p} ∈ {0,1}` (block b is on p in the current plan).

`c^hop_{p,q} = α^plan(prim(p) → prim(q)) + S(T)/β(prim(p) → prim(q))` on the cheapest transport the edge can actually get. RDMA counts only if both endpoints have queue pairs left: each RDMA edge of the plan takes QPs for the decode lane, the prefill lane and, on the return edge, tokens, and the solver reserves headroom for one overlapping epoch and two replay lanes against `RdmaPortInfo::max_qp` minus what other instances hold (RFC-0031 QP budget). An edge without QPs is costed on QUIC. Here `S(T) = T · h_model · streams · bytes(wire dtype)` (F32 in exact mode unless the family has passed BF16, ADR-0019 item 5) and `α^plan` is the p50 on wired media and the **measured mean** on `WiFi` and `Overlay` (ADR-0008 item 4, revised 2026-10-20). The mean is what tokens per second depend on; the tail of those links enters through the stall-rate row C14, and p99 is used only by the TP/EP gate (§3.1, RFC-0010 §3.5) and as a tie-breaker between plans whose predicted `t_tok` is equal. Co-located placement nodes on one host use the shared-memory link's α. The ring-return edge (tail → head) carries a `TokenFrame` (token IDs plus any draft), not activations, so its cost is `α^plan + S_token/β` with `S_token` a few hundred bytes; C4's return row uses `c^ret_{p,q}`.

`c^coll_{b,p}` for an island or intra-node TP group of N members: `2 · layers(b) · t_AR(N, S(T))`, with `t_AR = 2(N−1)·α_eff + 2(N−1)/N · S/β` for ring all-reduce or `α_eff + (N−1)·S/β` for one-shot on a full mesh, whichever RFC-0032 selects for that group (cost model §3.8). For EP islands: `2 · moe_layers(b) · (α_eff + k · h_model · 4 B / β)` per pass (dispatch and combine, §3.9).

*Note, 2026-10-22 (DIST-E13-T12), a correction for the maintainers to confirm:* the formulas above follow RFC-0032 §5.3's table, which a note of the same date reads with α, the size-independent per-message cost α = α_eff,probe − S₀/β. ADR-0310's `α_eff` already includes S/β at the probe partial, and RFC-0001 §5.2's `LinkCaps::alpha_eff` is used alone, never with S/β added again. Two changes follow:
- read α for `α_eff` in `t_AR` and in the EP term. Otherwise the size term is counted twice;
- one-shot on a full mesh is `α + S/β`, since its N−1 sends go out in parallel on separate links. `α + (N−1)·S/β` is the one-shot form when the sends share one NIC.

`syntropy-solver`'s `collective` module prices `t_AR` this way (`one_shot`, `ring`, `choose`), and derives α from a probe with `alpha_from_probe`. PLC-E5-T09's MILP term should take its `t_AR` from it.

**Variables**

| Variable | Domain | Meaning |
|---|---|---|
| `v_p` | binary | placement node p hosts a stage |
| `x_{b,p}` | binary | block b belongs to p's stage |
| `π_{b,p}` | [0, 1] | block b starts p's layer range (start indicator; integral at any vertex where x is integral) |
| `f_{p,q}` | binary, p ≠ q | q's stage follows p's in the ring (activation hop) |
| `r_{p,q}` | [0, 1], p ≠ q | p is the tail and q the head (token return) |
| `g_{p,d}` | binary | device d of p executes something (charges `t_fixed` and scratch) |
| `w^A_{b,k}`, `w^F_{b,k}` | [0, 1] | share of block b's mixer part and FFN part on pair k |
| `ξ_{b,d}` | [0, 1] | share of block b's layers whose mixer runs on device d but whose FFN does not (each such layer pays two hand-offs) |
| `w^H_k`, `w^R_k`, `w^Dr_k` | [0, 1] | share of the LM head, of `RowLookup` tables, and of drafter bytes on pair k |
| `ρ^hot_{b,k}`, `ρ^res_{b,k}` | [0, 1] | share of block b's KV for the **active** sequences (read every step) and for **reserved** sequences (queued or background in latency mode: capacity only, no read cost) on KV placement k |
| `τ^kv_{b,p}` | ≥ 0 s | time of block b's attention KV reads on p: the max of its GPU part and its CPU-split part |
| `y_{b,d}` | ≥ 0 bytes | block b's routed bytes resident in GPU d's expert cache |
| `u^P_{b,p}` | ≥ 0 bytes | the part of `u` held DMA-pinned in `HostPinned`: the copy source for copied misses and VRAM admissions. The rest, `u − u^P`, sits in `HostPageable` and is computed in place only (RFC-0020 §2, ADR-0200) |
| `u^G_{b,p}` | ≥ 0 bytes | the pageable share, `u^G = u − u^P`, charged to `HostPageable` |
| `σ_{b,p}` | {0, 1} | block b's active KV on p uses the CPU-split site, so its per-layer hand-off `t_switch` is paid (C7) |
| `z^RP_{b,p}` | [0, 1] | share of routed reads served from VRAM ∪ pinned RAM; the hottest RAM-resident experts are the pinned ones |
| `u_{b,p}` | ≥ 0 bytes | block b's routed bytes resident in RAM. When `u = Π_b` the experts are homed in RAM and the VRAM cache holds copies of RAM-resident experts. Otherwise the experts are homed on NVMe and the RAM cache is **exclusive** of the VRAM cache: a VRAM victim moves down to RAM, and promoting an expert to VRAM frees its RAM slot (RFC-0023 §3) |
| `z^V_{b,p}`, `z^R_{b,p}` | [0, 1] | share of routed reads served from VRAM, and from VRAM ∪ RAM |
| `m^P_{b,p}`, `m^C_{b,p}` | ≥ 0 bytes | RAM-served routed bytes per pass sent over PCIe to the GPU, and computed by the CPU in place |
| `τ_{b,p}` | ≥ 0 s | time of block b's routed-expert work on p (epigraph of a max) |
| `χ_{b,p}` | binary, only where `Π_b` fits in one GPU of p | block b on p hands off to the CPU pool (two `t_switch` per MoE layer) |
| `e_p` | [0, 1] | p is the tail but not the head (pays duplicated embedding rows for tail-side drafters, and tied LM-head copies) |
| `θ` | ≥ 0 s | bottleneck stage time (throughput objective only) |

**Size** [estimate, counted at the bounds |B| = 64, |P| = 16, |K_p| ≤ 5, 8 curve segments, ≈ 20 GPUs]. Binaries: `x` 1,024, `f` 240, `g` ≈ 50, `v` 16, C7's split indicator `σ` up to 1,024, about 2,350, plus up to 1,024 `χ` on small-pool MoE models. Continuous columns: `π` 1,024, `w` 10,240, `ξ` 3,200, `ρ^hot` and `ρ^res` 8,192, `y` 1,280, per-block expert and KV variables (`u, u^P, z^V, z^P, z^R, m^P, m^C, τ, τ^kv`) 9,216: about 33,000. Rows: C4 ≈ 15,400; hit-curve bounds 3 × 8 × 1,024 ≈ 24,600; C5 ≈ 13,400; C7 ≈ 9,200 (with σ's indicator rows); other C8 rows ≈ 11,000; C1–C3, C6, C9 and joint host rows ≈ 3,000: about 76,000. Two reductions apply in the builder: `w ≤ g` is aggregated to `Σ_b w_{b,k} ≤ |B| · g` (≈ 10,000 rows fewer), and blocks whose pool fits one GPU use 4 curve segments. That leaves about 33,000 continuous and 2,350 binary columns, and 66,000 rows. Presolve cannot remove C4 rows, because C1–C3 fix no `x` before branching. These counts set spike S1's question (open question 1).

### 6. The MILP: constraints

Each constraint is written for every index it mentions, over the sets above.

**C1 Every block placed once.** `Σ_p x_{b,p} = 1`.

**C2 Placement-node use and exclusivity.** `x_{b,p} ≤ v_p`; `v_p ≤ Σ_b x_{b,p}`; for every physical node n, `Σ_{p : n ∈ phys(p)} v_p ≤ 1`. A node hosts one stage per instance, except in a degraded plan (§17).

**C3 Contiguous ranges.** `π_{0,p} ≥ x_{0,p}`; `π_{b,p} ≥ x_{b,p} − x_{b−1,p}` for b ≥ 1; `Σ_b π_{b,p} ≤ 1`. Each selected placement node holds exactly one run of consecutive blocks. Together with C1 this fixes the ring order: the stage holding block 0 is the head, and the stage holding the next block after p's range is p's successor. No slot variables and no cycle enumeration are needed.

**C4 Ring successors.** Forward edges: `f_{p,q} ≥ x_{b,p} + x_{b+1,q} − 1` for every adjacent block pair, `Σ_q f_{p,q} ≤ v_p`, `Σ_p f_{p,q} ≤ v_q`. The ring return, tail to head, has its own variable because it carries a `TokenFrame`, not activations: `r_{p,q} ≥ x_{β_L,p} + x_{0,q} − 1`, `r ∈ [0, 1]`, costed at `c^ret_{p,q}`. Because every hop has positive cost, the optimum sets `f` and `r` to exactly the ring's edges. A single-stage plan has no `f` or `r` at 1 and pays no hop.

**C5 Dense placement.** For c ∈ {A, F}: `Σ_{k∈K_p} w^c_{b,k} = x_{b,p}` and `w^c_{b,k} ≤ g_{p,dev(k)}`. Split layers: `ξ_{b,d} ≥ Σ_{k : dev(k)=d} (w^A_{b,k} − w^F_{b,k})`. The split lets attention and its KV sit on the GPU while FFNs execute in RAM. At batch 1 and moderate context it changes little, because a byte of KV read from RAM costs the same as a byte of weights read from RAM. It pays when KV traffic dominates: in throughput mode each sequence reads its own KV while weights are shared, and at 32 K context Llama-70B reads 10.7 GB of KV per token, a quarter of its Q4 weights (cost model §3.6). Per-group pins (`PlacementPin`, `TensorOverride`, RFC-0001 §11.3) fix the corresponding shares: a pin of group g of part c of block b to tier i adds `w^c_{b,k} ≥ bytes_g / D^c_b · x_{b,p}` for the pair with `tier(k) = i`, or forbids pairs.

**C6 Non-layer groups.** LM head on the tail: `Σ_{k∈K_p} w^H_k = x_{β_L,p}`. `RowLookup` tables on the head: `Σ_k w^R_k = x_{0,p}`; tables larger than `solver.lookup_ram_max` (default 2 GiB) are pre-assigned an `Nvme` home with a RAM cache budget, because their reads are a few KiB per token at addresses known a step ahead (`tiered-memory-inference.md` §3.1, §9). Tail-side drafters (MTP heads, draft heads) when speculation is on: `Σ_{k : tier(k) ∉ {Nvme}} w^Dr_k = x_{β_L,p}`; a draft model: the same on the head, `= x_{0,p}`. A drafter never lands on `Nvme` (ADR-0017 item 2). An MTP module that embeds the next token has its own `RowLookup` group, linked to the prologue's by `shared_with` (RFC-0041 §3.1); it is placed on the tail with the MTP layers, like a tied LM head's `Head` group, and its bytes are C9's duplicated-row term `E_dup · e_p`, so a one-stage plan counts the shared rows once. `e_p ≥ x_{β_L,p} − x_{0,p}` marks a tail that is not the head.

**C7 KV placement.** `Σ_{k∈K^kv_p} ρ^hot_{b,k} = x_{b,p}` and `Σ_{k∈K^kv_p} ρ^res_{b,k} = x_{b,p}`. Hot KV in `Vram` needs the block's mixer on that GPU: `ρ^hot_{b,(d,Vram)} ≤ Σ_{k : dev(k)=d} w^A_{b,k}`. GPU-streamed and CPU-split placements are readable whatever device runs the mixer. The read time is the epigraph variable `τ^kv_{b,p}`, with one row per path and `h = t_switch · layers(b) · σ_{b,p}`: the GPU path `τ^kv_{b,p} ≥ K^read_b · (ρ^hot_{(GPU,Vram)} + ρ^hot_{(GPU stream)}) / W_gpu + h` (VRAM pages plus the staged copies); the PCIe path `τ^kv_{b,p} ≥ K^read_b · ρ^hot_{(GPU stream)} · (1 / h2d_pinned + 1 / W_gpu) + h` (copy, then read the staged copy); and the CPU path `τ^kv_{b,p} ≥ K^read_b · ρ^hot_{(CPU split)} / W_cpu,attn(kind, T) + h`. The binary `σ_{b,p}`, with `ρ^hot_{(CPU split)} ≤ σ_{b,p}` (ρ is a share in [0, 1]), charges the hand-off exactly when the split is used. These rows are RFC-0022 §5.4's two canonical forms: with σ = 0 they give `GpuStream: max(B_vram/W_gpu, B_host/W_pcie) + B_host/W_gpu`, and with no streamed share `CpuSplit: max(B_vram/W_gpu, B_host/W_cpu,attn) + t_switch`. Reserved sequences' KV has no read term. In latency mode the active set is the interactive sequence and the reserved set is the other `max_sequences − 1` (RFC-0054); in throughput mode every admitted sequence is active. Lossless KV dtype unless the plan is approximate with `KV_LOSSY` (ADR-0019 item 5).

**C8 Experts.** For every b ∈ B_moe and p:
- Caches only on a GPU that runs the block's router: `y_{b,d} ≤ Π_b · Σ_{k : dev(k)=d} w^A_{b,k}`.
- Coverage: `u_{b,p} ≤ Π_b · x_{b,p}`, and `Σ_d y_{b,d} ≤ Π_b · x_{b,p}`.
- Hit curve, homogenised so it vanishes when the block is elsewhere: `z^V_{b,p} ≤ a_{b,j} · x_{b,p} + s_{b,j} · Σ_d y_{b,d}` and `z^R_{b,p} ≤ a_{b,j} · x_{b,p} + s_{b,j} · (u_{b,p} + Σ_d y_{b,d})` for every segment j; `z^V_{b,p} ≤ z^R_{b,p} ≤ x_{b,p}`. The fast-memory set is VRAM plus RAM counted exclusively, which is what RFC-0023's curve measures. For RAM-homed blocks (`u = Π_b`) the second bound already gives `z^R = x`, so they are unchanged: RAM holds the whole pool and the VRAM cache copies from it. RAM is charged `u` in C9 in both cases. Exclusive caches save the VRAM cache's bytes of RAM on NVMe-homed blocks: about 10 GB on scenario (b2), worth ≈51 ms per token under the uniform prior (§15.2).
- Pinned share: `u^P_{b,p} ≤ u_{b,p}`; `z^V_{b,p} ≤ z^RP_{b,p} ≤ z^R_{b,p}`; `z^RP_{b,p} ≤ a_{b,j} · x_{b,p} + s_{b,j} · (u^P_{b,p} + Σ_d y_{b,d})` for every segment j. Only pinned bytes can be copied: `m^P_{b,p} ≤ R_b · (z^RP_{b,p} − z^V_{b,p})`. Misses that land from NVMe go through the I/O engine's registered staging, so they may be copied too; the runtime places the hottest RAM-resident experts in the pinned share, which is what the curve order assumes.
- Everything not hit in VRAM is computed from RAM, after landing from NVMe if it missed RAM too: `m^P_{b,p} + m^C_{b,p} = R_b · (x_{b,p} − z^V_{b,p})`.
- Three sites in parallel, so the block's routed time is their maximum (cost model §3.3 overlapped form with SC-5's correction that RAM-side compute covers NVMe-served bytes; `tiered-memory-inference.md` §5.1, §5.4):
  - GPU: `τ_{b,p} ≥ ε^V_{b,p} · R_b · z^V_{b,p} + ε^P_p · m^P_{b,p}`
  - CPU: `τ_{b,p} ≥ ε^C_{b,p} · m^C_{b,p} + d_p · m^P_{b,p}`
  - NVMe: `τ_{b,p} ≥ ε^N_p · R_b · (x_{b,p} − z^R_{b,p})`
- Hand-off: `m^C_{b,p} ≤ R_b · χ_{b,p}` where `χ` exists; elsewhere the hand-off is a constant `2 · t_switch · moe_layers(b) · x_{b,p}`.
- On `Unified` nodes there is one cache level and no PCIe term: `y = 0`, `m^P = 0`, `z^V = z^R ≤ H_b(u)`, and the GPU line becomes `τ ≥ ε^V · R_b · z^R`.
- EP islands: routed work splits across members in proportion to their expert rates, a preprocessing constant from RFC-0032's balanced owner assignment. `ε^V` and `ε^C` above are then the island's aggregate per-byte times, each member's capacity is checked with its share `ω_{n,p}` in C9, and `c^coll` adds dispatch and combine in the objective.

**C9 Capacity, per tier, per physical node.** For every n and tier i ∈ I_n:

```text
  Σ_b Σ_{k : tier(k)=i} (D^A_b · w^A_{b,k} + D^F_b · w^F_{b,k})        dense homes
+ Σ_b Σ_{k ∈ K^kv : tier(k)=i} (K^store_b(L, M_act) · ρ^hot_{b,k} + K^store_b(L, M_res) · ρ^res_{b,k})   KV
+ [i is Vram or Unified of GPU d]   Σ_b y_{b,d}                         expert cache
+ [i is HostPinned]                  Σ_b u^P_{b,p}                       pinned RAM experts (copy sources)
+ [i is HostPageable]                Σ_b (u_{b,p} − u^P_{b,p})           pageable RAM experts (in place only)
+ [i is Unified]                     Σ_b u_{b,p}
+ Σ_k (Hd · w^H_k + E · w^R_k + Dr · w^Dr_k) [tier(k) = i]                head, lookups, drafters
+ [i is HostPageable of the tail's domain]  E_dup · e_p                   duplicated rows on a tail
+ Σ_d σ_d · g_{p,d} [d's scratch tier is i]  +  reserve_i                  scratch, fixed reserves
+ [i is HostPageable]  hist_p · (v_p − x_{β_L,p})                         replay history on every non-tail stage
— every term above, for placement node p on member n, multiplied by ω_{n,p} and charged to tier_n(k)
≤ usable_i − reserved_i(other instances)

joint host rows, per NUMA domain δ of node n (RFC-0020 §5):
  bytes(HostPinned_δ)                         ≤ pin_limit_δ − reserved
  bytes(HostPinned_δ) + bytes(HostPageable_δ) ≤ host_usable_δ − reserved
```

`ω_{n,p}` is member n's share of p's bytes: 1 for `Local`, the split weight for islands and TP groups, and the family's shard fraction for KV in TP groups. Every capacity term is scaled by it and charged to the member's own tier, so each member is checked against its own tiers, never against the island's sum. The joint host rows replace checking each host tier against `host_usable` on its own, which would let a plan count the same RAM twice: `HostPinned` draws on the lock budget `pin_limit`, and both host tiers together draw on the RAM that is actually available. `M_act`, `M_res` and `L` are from C7 and §11.2. On the `Nvme` tier, the whole shard must fit in the pack directory: `Σ_b x_{b,p} · (D^A_b + D^F_b + Π_b + K^park_b) + lookups ≤ min(usable_nvme, store.max_bytes − other packs)`, where `K^park` is session KV the policy asks to park on NVMe (RFC-0022).

**C10 Dense on NVMe.** Pairs whose tier is `Nvme` are absent in the first pass. If the first pass is infeasible and the policy does not forbid it, a second pass adds them with their real cost, and the plan is marked `degraded = DenseOnNvme` and reported to the user (ADR-0008 item 4).

**C11 Symmetry.** For Local placement nodes p < q (by `NodeId`) whose coefficients are exactly equal (same device classes, the same usable bytes per tier and the same link costs to every other node, so swapping them cannot change the objective): `v_p ≥ v_q`, and `Σ_b b · π_{b,p} ≤ Σ_b b · π_{b,q} + |B| · (1 − v_q)`.

**C12 Throughput-mode floor.** In the throughput objective only: ring time `Σ (latency terms) ≤ 1 / per_sequence_floor_tps` (RFC-0001 §11.3), and `θ` bounds each stage and each hop (§7.2).

**C13 Streaming homes, a second stage.** A dense group that the plan executes on the CPU of a node with a GPU, and that its stream site would stream (n* ≤ `prefill_chunk`, §11.2), needs a `HostPinned` home: prefill and batched passes stream it, and a pageable home is never a DMA source (RFC-0020 §2). At decode pass sizes the two host pairs have the same rate, because a CPU reads pinned and pageable memory alike, so the latency objective cannot tell them apart. After the MILP or the DP returns, an LP with every integer fixed and `t_tok` held at the solution's value moves such groups' bytes from (CPU, `HostPageable`) to (CPU, `HostPinned`) in layer order, maximising the pinned bytes subject to C9's pin row and joint rows. On the DP path (§9.5) this is a greedy pass in layer order that stops at the first row to bind; it reaches the LP's result because those capacity rows are the LP's only constraints. The pin row therefore charges them after the pinned demands that come first: those `t_tok` needs (`u^P`, staging) and the KV of `max_sequences` sequences at `solver.context_target`, which lives in `HostPinned` (RFC-0022 §3). `context_length` above the target then grows into what remains (§11.2). Bytes that do not fit stay pageable and prefill in place on the CPU. The TTFT model charges them at their CPU prefill time (§11.2 item (a)), `PredictedCost::ttft` and the preview's `ttft` figures include it (§14.1), and each such group's `GroupReason` carries `binding = CapacityExhausted { tier: HostPinned }`, which the explanation states with the TTFT it costs. `HostPageable` holds only groups that are never copied. Giving up decode time for pin budget, such as a smaller `u^P` so that prefill can stream, is governed by the TTFT cap (§7.1, open question 8, answered): when the plan's TTFT after this stage exceeds the cap, pinned expert bytes `u^P` move to `HostPageable` coldest first, and the freed pin budget goes to streamable dense groups in layer order, until the cap holds or no `u^P` remains; each step costs the least decode time per byte of pin budget freed. In M1 the step is computed and reported but not applied; it applies with the cap from M2.

**C14 Stall rate on wireless and overlay hops.** For each ring edge e on a `WiFi` or `Overlay` link (forward hops and the token return), `q_e = P(t_hop,e > mean_e + solver.stall_threshold)`, read from the link's hop-time histogram (RFC-0010 §6.2), or from its prior tail until the histogram holds 1,000 samples. The plan's stall rate, `Σ_e q_e · f_e` (or `· r_e` on the return edge), is a union bound on the share of tokens that wait more than `stall_threshold` beyond the mean on some hop. It must not exceed `solver.max_stall_rate`. The defaults, 300 ms and 0.01, express "p99 stall ≤ 300 ms" [target]. The row is linear in `f` and `r`. In the DP, an edge whose own `q_e` exceeds the bound costs ∞, and a completed plan whose summed rate exceeds it is dropped in favour of the best plan without its worst wireless edge (at most one retry per wireless edge). A plan with any wireless or overlay edge carries its stall rate and its p99 token time, `PredictedCost::jitter` (§2), which the preview and the dashboard show as a jitter badge (§14.1).


### 7. The MILP: objectives

#### 7.1 Latency (default, D2): the batch-1 critical path

One ring traversal at pass size T (T = 1 without speculation) costs the sum of every stage's work plus every hop, because a single sequence visits stages in turn (cost model §3.7):

```text
minimise  t_pass(T) =
      Σ_b Σ_k  (δ^A_{b,k} · w^A_{b,k} + δ^F_{b,k} · w^F_{b,k})      dense reads at W_eff of the executing kernel
    + Σ_{b,d}  2 · t_switch · layers(b) · ξ_{b,d}                     layers split between mixer and FFN devices
    + Σ_{b,p}  τ^kv_{b,p}                                           KV reads of the active sequences (C7), at L_read
    + Σ_{b,p}  τ_{b,p}                                              routed experts: max(GPU, CPU, NVMe) per block
    + Σ_{b,p}  2 · t_switch,p · moe_layers(b) · χ_{b,p}             CPU-pool hand-offs (or the constant form)
    + Σ_{p,d}  t_fixed,d · g_{p,d}                                  launch and sync per device used
    + Σ_p      2 · t_switch,p · (Σ_d g_{p,d} − v_p)                 hand-offs between devices inside a stage
    + Σ_{p≠q}  (c^hop_{p,q} · f_{p,q} + c^ret_{p,q} · r_{p,q})        ring hops and the token return; mean on wireless (C14 bounds the tail)
    + Σ_{b,p}  c^coll_{b,p} · x_{b,p}                               collectives inside TP/EP groups
    + Σ_k      δ^H_k · w^H_k  +  t_sample                           LM head and sampler on the tail
    + ω_move · Σ_{b,p} bytes_b · (1 − cur_{b,p}) · x_{b,p}          migration, re-plans only (§7.4)
```

Two context lengths enter. `L_read = max(solver.context_target, WorkloadStats::context.p90)` prices KV reads, so a session that has grown long gets its KV costed at its real size. `L` in C9's capacity rows is the plan's `context_length`, the longest context it reserves KV for (§11.2). The intra-stage hand-off term counts `2 · (devices used − 1)` crossings: §11.1 orders each stage's layers so every device's layers are consecutive, which is what makes that count exact.

This objective prefers fewer stages without a special rule: every extra stage adds a hop and a `t_fixed`, so dropping a weak device beats adding a hop whenever its bytes save less than those costs (ADR-0006 item 1).

**TTFT cap.** Decode alone would pick plans whose first token is several times slower than the best available: on S2 the decode-optimal plan's 4K-prompt TTFT is ≈ 22 s against 3.6–5.9 s on the PC alone [estimate: RFC-0032's chunked-prefill table and §15.1]. Latency mode therefore minimises `t_pass` **subject to**

```text
TTFT(P, L_ttft) ≤ solver.ttft_cap_factor · TTFT_best          default factor 2.0
L_ttft    = WorkloadStats::context.uncached_prompt_p90, prior 4096 tokens (the 4K prompt §11.2 and §15.1 use)
TTFT_best = the lowest TTFT(·, L_ttft) among the candidate plans the solve compares
```

TTFT is the routing formula of §13 at `L_ttft` with no cached prefix and no queue: pipelined chunks plus hops, with each stage's prefill time per chunk from its stage cost. The **candidates** are the DP's best decode plan for each physical-node subset and head (its answer table, §10), each with its TTFT, which the stage cost returns beside the decode time, so the cap costs O(stages) arithmetic per candidate. The chosen plan is the lowest `t_tok` among candidates within the cap; `TTFT_best`'s own plan is always within it, so the cap never makes a solve infeasible. Where the MILP is built, the cap is one more row with an epigraph for the pipelined `max` term, and `TTFT_best` comes from the DP's candidates. Both numbers, decode tok/s and TTFT at `L_ttft`, are shown for every preview candidate (§14.1) and in the explanation's headline. The cap binds from M2, with the multi-node solver; M1's single-node solves compute and report it (`PredictedCost::ttft_cap`), and PLC-E6-T23 measures how often it would bind. Throughput mode has no cap. This answers open question 8.

#### 7.2 Throughput: the bottleneck stage at a chosen micro-batch

For a micro-batch of `b_μ` sequences per stage step, with D stages and `M = D · b_μ` sequences in flight, each stage processes one micro-batch per step, so aggregate throughput is `b_μ / θ` where θ is the slowest stage or hop (cost model §3.7, §3.11):

```text
minimise  θ  +  η_ring · t_pass(b_μ)
subject to  θ ≥ stage_time_p(b_μ)          for every p: the latency terms restricted to p's blocks and devices,
                                           with coefficients at T = b_μ (batch curves, `UnionCurve::batch` at b_μ)
            θ ≥ c^hop_{p,q}(b_μ) · f_{p,q}  every hop is its own pipeline stage on its link
            t_pass(b_μ) ≤ 1 / per_sequence_floor_tps            (C12)
            KV capacity at M = D̂ · b_μ sequences (C9)
```

`η_ring = 10⁻³` keeps ring latency as a tie-breaker so `f` stays tight. D is not known before the solve, so the DP runs once per D (§10), and the MILP for a given `b_μ` sizes KV at `D̂ · b_μ` with D̂ the best DP plan's stage count for that `b_μ`, repeating once if the MILP's D differs. A throughput plan therefore costs up to 12 warm-started MILP solves (6 values of `b_μ`, ×2), plus the speculative loop when it is on. It runs in the background with a 10 s target (Performance), and SC-31's 2 s applies to latency mode only (ADR-0103). The outer loop tries `b_μ ∈ {1, 2, 4, 8, 16, 32}` up to the KV limit and keeps the best `b_μ / θ`. The expert union makes large `b_μ` expensive on tiered MoE models (8 tokens touch 7.2× the routed bytes of one at DeepSeek-V3 top-8, cost model §3.9), and the objective sees that through `R_b(b_μ)`. When capacity is left after the best plan, the outcome carries a replica hint: a second instance on the unused nodes avoids every hop (cost model §5.4), and the coordinator may submit `EnsureInstance` for it.

Speculation is off in throughput mode unless the outer loop of §7.3 predicts a gain at the chosen `b_μ` (ADR-0017 item 8).

#### 7.3 Speculative window: outer loop

The speculative objective `(t_draft(γ) + t_pass(γ+1)) / E[tokens(γ)]`, with `E = (1 − a^{γ+1}) / (1 − a)` (ADR-0017 item 5), is not linear in the placement, but for a fixed γ the denominator is a constant and the numerator is the latency objective at T = γ + 1. The solver therefore fixes T, solves, and iterates:

```text
P1 ← DP(T = 1)                                          // or the MILP, where §9.5's gate built it
best ← (P1, γ = 0, t_tok = t_pass(P1, 1))
for each drafter s available (ADR-0017 item 1 order), a = acceptance(s, task) or 0.7:
    for γ in 1 ..= speculative.max_window (RFC-0054; `auto` = 7):
        t = (t_draft(s, γ, P1) + evaluate(P1, T = γ + 1).t_pass) / E(a, γ)
        track the minimum (s*, γ*)
if γ* > 0 and the improvement over best is ≥ 2%:
    P2 ← DP(T = γ* + 1) (the MILP warm-started from P1 where built)   // CPU tiers may turn compute-bound at T, so
                                                            // bytes may move to the GPU
    re-run the γ scan on P2; keep whichever of P1, P2 gives the lower t_tok
```

`t_draft(s, γ, P)` is the drafter's chain cost on its placed device: γ sequential draft steps of `Dr / rate + t_fixed` for MTP heads (and draft heads, where one is supplied; v1 produces none, RFC-0012 §9.1) on the tail, or for a draft model on the head. Tree shapes are RFC-0012's choice inside the window budget `T* = γ* + 1`. The plan records γ* as `SpeculativeConfig::max_window` (γ_max, RFC-0001 §5.4), and graphs and KV slots are sized for `max_window + 1` positions. At most two solves happen per latency-mode plan (throughput mode: §7.2).

#### 7.4 Migration cost on re-plans

Moving a shard is paid once; a better plan pays back per token. Converting one into the other: moving `X` bytes over the fetch path takes `X / β_fetch`, and over the payback horizon the instance serves `replan.payback_horizon / t_tok,current` tokens, so

```text
ω_move = t_tok,current / (β_fetch · replan.payback_horizon)       seconds per token per byte moved
```

with `β_fetch` the measured rate of the links or local reads that would supply the bytes (pack rebuilds count `Packing` throughput). Moving a recurrent-state layer of a hybrid family also moves its live state: the migration term charges those state bytes ÷ β of the path that carries them, plus the replay from the last state checkpoint, and there is no separate stickiness term (open question 5, answered; OQ-PL-12). This term only biases the MILP towards plans that keep shards in place; adoption itself still uses RFC-0001 §7.2's explicit rule (§12.2).

### 8. Linearisation techniques

| # | Nonlinear structure | Linear form | Why it is exact or safe |
|---|---|---|---|
| 1 | "each node holds one contiguous layer range" | start indicators `π` (C3) instead of slot-by-node products | at most one 0→1 transition per placement node; LP-integral given integral x |
| 2 | ring order | successor binaries `f` bounded below by adjacency of block owners (C4) | the costs are positive, so f equals the adjacency at the optimum; no subtour can arise, because the order comes from the layer sequence |
| 3 | hit rate `h(c)` (concave) | homogenised piecewise-linear upper bounds on `z` (C8) | maximising a concave function inside a minimisation with positive coefficients is convex; the epigraph is exact at the breakpoints and within the curve-fit error between them |
| 4 | three sites in parallel (`max`) | epigraph `τ` with one row per site | `τ` is minimised, so it equals the max |
| 5 | bottleneck stage | epigraph `θ` | same |
| 6 | tier split of a block's dense bytes | continuous shares `w` with post-rounding to whole tensor groups (§11.1) | the LP relaxation of a fractional knapsack is exact for the split; rounding error is bounded by the largest group in the block and is re-checked by the evaluator |
| 7 | KV placement | continuous shares `ρ` | same; pages are 4 KiB-granular, so a fractional split is physically realisable |
| 8 | alternative configurations of the same hardware (Local vs intra-node TP vs island) | separate placement nodes with an exclusivity row (C2) | no big-M; each configuration has its own exact coefficients |
| 9 | duplicated embedding rows only when head ≠ tail | `e_p ≥ x_{β_L,p} − x_{0,p}` | with contiguity, head = tail only in the one-stage plan |
| 10 | speculative ratio objective | outer loop over γ with T fixed (§7.3) | Dinkelbach-style: for fixed T the ratio's denominator is constant |
| 11 | throughput objective at unknown D | outer loop over `b_μ`, KV sized at `D̂ · b_μ`, one repeat | exact once D matches |
| 12 | migration cost | linear in x because `cur` is constant | — |
| 13 | mean for wireless hops, tail bounded by C14 | coefficient choice, plus one row `Σ_e q_e · f_e ≤ max_stall_rate` | the mean is linear in `f`; the stall row is a union bound, linear in `f` and `r`, and conservative |
| 14 | TTFT cap (§7.1) | one row over stage prefill times and hops, with an epigraph for the pipelined `max` term; `TTFT_best` a constant from the DP's candidates | the epigraph is exact because the row bounds the max from above |


### 9. HiGHS integration

#### 9.1 Linking

HiGHS (MIT) is built from source and linked statically through the `highs-sys` crate's default `build` feature, which compiles the vendored HiGHS with CMake and a C++ compiler at build time; target hosts need only the C++ standard library, which the binary already carries for other components (RFC-0062 owns the toolchain). `syntropy-solver` wraps the C API directly (`Highs_create`, `Highs_passModel`/`Highs_passMip`, `Highs_setStringOptionValue`/`Highs_setIntOptionValue`/`Highs_setDoubleOptionValue`, `Highs_setSolution` or `Highs_setSparseSolution`, `Highs_run`, `Highs_getSolution`, `Highs_getModelStatus`, `Highs_getInfo*`, `Highs_setCallback`) behind a small safe module; it is the only crate that links HiGHS (RFC-0001 §4.2 rule 6). The model is assembled column-wise in compressed sparse form in one pass and handed over in one call.

#### 9.2 Options and determinism

| Option | Value | Why |
|---|---|---|
| `threads` | 1 | RFC-0001 §4.2 rule 5; also keeps the coordinator's `ctl` runtime responsive |
| `parallel` | `"off"` | same |
| `random_seed` | 0x5359_4e54 (fixed) | reproducibility |
| `presolve` | `"on"` | removes empty and singleton rows; it cannot remove C4 rows, which bind only once `x` is fixed in branching |
| `mip_rel_gap` | `solver.mip_rel_gap`, default 0.01 | a 1% gap is well inside the ±15% prediction budget (SC-30) and the 20% adoption threshold |
| `mip_max_nodes` | `solver.mip_node_limit`, set by spike S1 | deterministic work limit |
| `simplex_iteration_limit` | `solver.simplex_iteration_limit`, set by spike S1 | deterministic work limit; whether HiGHS applies it per LP or cumulatively in MIP is confirmed by S1 |
| `time_limit` | `safety_time_limit`, default 2 s; in a repair solve, what remains of §12.4's 2 s budget | safety net only |
| `mip_feasibility_tolerance` | 10⁻⁶ (HiGHS default) | see Scaling below |

**Determinism rule.** Count-based limits (nodes, simplex iterations) give the same result on any machine for the same input. The wall-clock limit does not, so it is set above the work limits' typical cost on the slowest Tier 1 coordinator and treated as a fault when it fires: the outcome is flagged `SolveStats::time_limited = true`, the incumbent is used, and the event is counted in telemetry. The count limits must bind first. The first defaults (20,000 nodes and 2 × 10⁶ simplex iterations) could not: at 10⁴–10⁵ dual-simplex iterations per second for an LP of this size [estimate], 2 × 10⁶ iterations take 20–200 s, so the 2 s clock would have fired first on every hard input. Spike S1 therefore measures iterations per second and nodes per second on the slowest Tier 1 coordinator and sets both limits so that they bind at no more than 50% of `safety_time_limit` on that machine. RFC-0061's solver suite checks this on every release: the time limit must never fire on its 8-node, Kimi K3-class reference inputs. In the deterministic simulator (RFC-0060) the wall-clock limit is disabled and the count limits alone stop the search; a firing outside the simulator is counted as a fault. Model building follows RFC-0060's rules T3 and T4: the solver uses `BTreeMap`/`BTreeSet` or `syntropy_core::DetHashMap`, never a randomly seeded hash container, and sorts at every boundary, so variable and row order (and therefore HiGHS's path) depend only on the input.

**Scaling.** One unit pair inside the model: **microseconds and MiB**. Hit fractions are scaled by the bytes they multiply: the builder uses `z̃ = R_b · z` (MiB read per pass) instead of `z`, so a τ row's coefficient is a per-MiB time (≈ 175 µs/MiB at 6 GB/s NVMe) and a hit-curve row's coefficient is `R_b · s_{b,j}` (≈ 0.03 for a DeepSeek-V3 layer: 191 MiB of routed reads times a slope of 1/6,480 MiB). Coefficients then span about 10⁻²…10⁴ [estimate from those two extremes]; without the rescaling they span about 10⁸. S1 reports HiGHS's numerical warnings on the reference inputs.

**Builds and architectures.** HiGHS is compiled with `-ffp-contract=off` and without target-specific `-march` flags, so the same input yields the same plan on any machine of one architecture. Identical plans across x86_64 and arm64 are tested, not promised: only the coordinator solves, and other nodes apply the committed plan, so cross-architecture identity matters only for tests. If the test finds a difference, the guarantee is documented as per architecture.

#### 9.3 Warm start and search control

- The DP solution (§10) is mapped to a full assignment of every column and passed with `Highs_setSolution` before `Highs_run`, so branch-and-bound starts with an incumbent and can prune at once. Before that, the DP and current-plan solutions are canonicalised within each C11 symmetry class (identical nodes renumbered so used ones come first and start blocks increase), so the symmetry cuts never reject the warm start. On re-plans the current plan (re-evaluated on the new graph, if still feasible) is a second candidate, and the better of the two is passed. Whether HiGHS uses a supplied MIP solution as its starting incumbent in every release we pin is checked by a test in RFC-0060's solver suite (open question 1).
- A callback (`Highs_setCallback`) watches improving solutions and the gap. It stops the search early when the gap is below `mip_rel_gap`, and it records the incumbent trajectory for `SolveStats`.
- **Corridor re-solve.** If the full MILP stops at a work limit with a gap above 5%, a second MILP fixes every block more than `solver.corridor` blocks (default 4) away from the incumbent's stage boundaries to its incumbent owner (`x = 1` there and 0 elsewhere, which keeps C1 satisfiable), leaves the corridors around each boundary free, and solves again with the remaining work budget. This keeps the coupled trade-offs while limiting the branching to plausible boundary moves.

#### 9.4 Which method runs, by scale

| Situation | Method |
|---|---|
| Default from M1, and every build until §9.5's gate passes | DP with the greedy stage cost (§10); the speculative outer loop re-runs the DP at T* |
| Where the MILP is built: ≤ 8 candidate nodes (after scoring), MILP enabled | DP warm start, then MILP; the result is the better of the two by the evaluator |
| MILP disabled (`solver.enable_milp = false`), or HiGHS reports an error or numerical trouble | DP only |
| MILP hits its work limit without an incumbent better than the DP | DP (the DP plan was the incumbent anyway) |
| Delta solve (§12.3) | LP with structure fixed (no binaries) |
| Repair after a stage loss, runner failure or edge loss (RFC-0033 §6) | `solve_repair` (§12.4): keep-shards evaluation of `RestartInPlace`, `Replace` and `Absorb` with survivors' assignments fixed, scored by `t_resume`; the survivors-only DP and MILP run only for `Resolve`. No switch between plans mid-recovery |
| Fast routing | no optimisation model (§13) |

ADR-0101 records this split.

#### 9.5 The MILP gate

M1 ships the DP and its greedy stage cost, and the production MILP of §5–§9 is built only if a simulation study shows it beats the DP by more than 5% on a reference input. The cited studies show that measured placement beats proportional splits, not that a MILP beats a good greedy, and the MILP brings a solver dependency, determinism tuning and crash-isolated previews. ADR-0008's own revisit test ("DP within 5% of the MILP") therefore runs before the MILP's cost is paid, not after.

- **The study.** It runs in S-09's standalone harness in M0 (PLC-E5-T02), beside S-09's scale measurements (open question 1). The harness builds a prototype of §5–§8 on HiGHS and a prototype of §10's DP from the same coefficients. **Reference inputs:** §15's four worked scenarios with their stated inputs, the RFC-0061 solver suite's inputs (1, 2, 4 and 8 nodes × dense 70B, DeepSeek-V3-class and Kimi K3-class censuses), and random fabrics of 1–8 nodes from `testkit`'s generator, latency and throughput objectives both. **Measure:** for each input, `gain = (t_tok(DP) − t_tok(MILP)) / t_tok(DP)` with both plans scored by one evaluator; relative tok/s in throughput mode. The MILP must reach its 1% gap on the input for the comparison to count.
- **The gate.** The production MILP (PLC-E5's model and HiGHS integration tasks) is built only if `gain > 0.05` on at least one reference input. The result, with every input's gain, is recorded in ADR-0101 and here before M1's solver work starts.
- **If the gate does not pass.** No shipped binary links HiGHS; it stays a test-only dependency of the DP's feasibility property test (Testing). `solver.enable_milp` is not registered. Shadow prices (§14), the delta solve (§12.3) and C13's second stage use the closed forms those sections give, because each is a separable allocation that the greedy solves exactly. The study is re-run when M3's islands and throughput objective land, and whenever ADR-0008's revisit triggers fire.
- **What stays MILP-shaped either way.** `SolveInput`, `SolveOutcome`, `SolveStats::path`, `PredictedCost`, `PlanExplanation` and the evaluator are the same for both methods, so building the MILP later redesigns nothing. §5–§8 remain the specification of every coefficient the DP and the evaluator use.

### 10. DP: the production method, and the MILP's warm start

The DP is exact over contiguous plans for a simplified per-stage cost, and it covers every node order and subset without enumerating cycles. It is the production method from M1 (§9.5). Where the MILP is built, the DP is its warm start and fallback.

**Stage cost.** `stage(n, i, j)` is the time of blocks `[i, j)` on placement node n under a greedy in-node placement, or ∞ if they cannot fit:

```text
stage(n, i, j):
    need ← σ (scratch for the devices used) + reserves
         + [i = 0]          (lookup tables, draft model)
         + [j = β_L+1]      (LM head, tail-side drafters, + duplicated rows if i ≠ 0)
         + [j ≠ β_L+1]      hist_n in HostPageable (replay history on every non-tail stage, §3.3)
    0. pins: apply PlacementPin and TensorOverride first, as fixed allocations; a pin that does not
       fit returns ∞ naming the pin
    1. dense: fill pairs fastest first with every block's mixer part D^A_b, then its FFN part D^F_b
       (fractional knapsack). Host-resident dense groups go to HostPageable before HostPinned here,
       so the pin budget is kept for the DMA sources decode needs; C13's second stage then moves
       those a stream site would stream into whatever pin budget remains
    2. KV, after the mixers: a block's active KV goes to VRAM only on the GPU that runs its mixer
       and only while room remains; otherwise to the cheaper of GPU-streamed (HostPinned) and
       CPU-split (HostPinned) reads for its attention kind. Reserved sequences' KV goes to
       HostPinned while room remains, then to NVMe parking where the policy allows
    3. experts: for each MoE block, remaining VRAM to the expert cache, then pinned RAM up to the
       copy share's needs, then pageable RAM, by the merged hit-curve slopes across blocks (a
       separable concave allocation, so slope order is optimal); the rest stays on the NVMe home
    4. miss split per block: the closed form that equalises GPU and CPU time,
       m^P = R_miss·(ε^C − g_V) / (ε^C + ε^P − d)  clamped to [0, min(R_miss, pinned-served bytes)],
       where g_V is the GPU's hit work per miss byte (RFC-0023 refines it at runtime)
    5. local search: shift VRAM between KV and the expert cache in steps of 1/8 of free VRAM,
       keep the best
    6. check every tier and the joint host rows (C9); if anything does not fit: return ∞
    return Σ dense + τ^kv + max(GPU, CPU, NVMe) per block + t_fixed + switches
```

`stage` is evaluated once for every (n, i, j), so `|P| · |B|² / 2` evaluations (≈ 33,000 at |P| = 16, |B| = 64) with prefix sums over blocks; each is O(tiers + curve segments).

**Recurrence (latency), factored.** The state is (the set S of physical nodes used, the head placement node h, the last placement node ℓ, the next uncovered block). The transition is split in two so that each state does O(|P| + |B|) work instead of O(|P| · |B|):

```text
G[S][h][ℓ][i] = min over ℓ' in S \ {ℓ}:  F[S \ {ℓ}][h][ℓ'][i] + c^hop(ℓ', ℓ)      arrive at ℓ with [0, i) done
F[S][h][ℓ][j] = min over i < j:          G[S][h][ℓ][i] + stage(ℓ, i, j)          ℓ covers [i, j)
F[{h}][h][h][j] = stage(h, 0, j)
answer = min over (S, h, ℓ):  F[S][h][ℓ][β_L+1] + c^ret(ℓ, h)        (0 if S holds one node)
```

Physical exclusivity is a mask check on S. Cost [estimate]: at 8 physical nodes, |P| = 16 and |B| = 64 there are 2^8 · 16 · 16 · 64 ≈ 4.2 M states and ≈ 80 relaxations each, ≈ 3.4 × 10⁸ in total, about 0.34 s at ≈ 1 ns per relaxation; with Local placement nodes only (|P| = 8) it is 2^8 · 8 · 8 · 64 ≈ 1.05 M states × 72 ≈ 7.5 × 10⁷, about 75 ms. Targets: ≤ 100 ms for 8 Local nodes and ≤ 400 ms with islands, measured by RFC-0061's solver suite. Repair solves (§12.4) coarsen to 32 blocks, which brings the Local-only case to ≈ 2 × 10⁷ relaxations, about 20 ms [estimate]. Pruning against the best complete plan found so far may cut further, but no target relies on it.

**Throughput.** The same recurrence with `max` instead of `+` for stage and hop terms, per `b_μ`. KV for `D · b_μ` sequences depends on the stage count D, so the DP runs once per D = 1 … N with S restricted to D nodes and every stage's KV sized for `D · b_μ` sequences, and keeps the best `b_μ / θ` over D.

**Guarantee.** A contiguous plan that fits exists if and only if some `F[…][β_L+1]` is finite. The DP considers every subset, order and range per node, and a range's feasibility on a node depends only on the range and on whether it is first or last: the history reserve is per non-tail stage, not proportional to the stage count. Inside a range, the greedy returns ∞ only when the fractional relaxation is infeasible too. Every byte class has a host-tier option: dense groups in `HostPageable` or `HostPinned`, active KV through GPU-streamed or CPU-split reads from `HostPinned`, reserved KV in `HostPinned` or parked on NVMe, and experts behind an NVMe home with any cache size. The one coupling, active KV in VRAM only behind its own mixer, is respected by placing mixers first (step 1 before step 2). So the greedy never leaves capacity unused while a compatible byte remains, and it fails only when the mandatory bytes exceed the capacity of the tiers they may use. Pins are applied first and reduce capacity; they do not change the argument. This is the SC-31 guarantee. The test checks it against an exact LP feasibility check per range on random small instances (Testing), not against the greedy itself.

### 11. Plan assembly, rounding and validation

#### 11.1 Rounding

1. **Stage boundaries at layer granularity.** For each boundary between consecutive stages, try moving it by up to ⌈block size / 2⌉ layers in each direction, only to legal cut points; evaluate each candidate with the evaluator (§11.3) and keep the best feasible one. Boundaries are refined one at a time, head to tail, twice.
2. **Whole tensor groups.** In each stage, for each block and part, take the fractional shares `w^c_{b,k}` and assign whole groups (mixer, router and hyper-connection weights in part A; dense FFN and shared expert in part F) to pairs: sort the part's groups by time saved per byte on the faster pair and fill the faster pair until the next group would exceed `w^c_{b,k} · D^c_b` by more than half its size. The LM head, lookups and drafters are rounded the same way.
3. **Contiguous device runs.** Inside a stage, layers execute in order, so the set of layers homed on each device is chosen to be one consecutive run per device. Among the assignments with the same bytes per device (which have equal byte time), pick the one with the fewest device crossings; on the tail, the GPU run includes the last layer so the LM head needs no extra crossing.
4. **Experts.** Per-layer budgets come from `y_{b,d}` split across the block's layers by the same slope order that built `H_b`. A block's `u_{b,p}` equal to its pool means its experts are homed in `HostPinned` and the VRAM cache copies from them; otherwise they are homed on `Nvme` with a `HostPinned` cache budget of `u`, exclusive of the VRAM cache (RFC-0023 §3). `miss_split = m^P / (m^P + m^C)` starts the runtime's balance (RFC-0023).
5. **KV.** `ρ` becomes a `KvBudget` per stage: the hot window in the attention device's tier, the rest in `HostPinned` with `spill` set; KV parking on `Nvme` when the policy enables saved sessions (RFC-0022).

#### 11.2 Assembly into a `PlacementPlan`

| Field | From |
|---|---|
| `stages` | one `StageAssignment` per selected placement node, ring order from `f`; `role` HEAD for block 0, TAIL for the last, DRAFT where the drafter runs; `mode` from the placement node kind; `next_hop.link` and `.transport` = the first transport in `LinkInfo::transports` with a measured `LinkCaps` (lowest α); `.wire` = F32, or BF16 where the family passed it (ADR-0019 item 5) or the plan is approximate with `WIRE_COMPRESS`; `predicted_step` from the evaluator |
| `tensors` | one `TierAssignment` per census group: `owner` = the stage's node (an island's member by the TP/EP split), `home` from rounding, `compute.decode` = the home tier's executing device, `compute.batched = Some(StreamSite { device: GPU, via: Pcie link, min_tokens: n* })` for `HostPinned` homes on nodes with a GPU when n* ≤ `prefill_chunk` (C13 homes every such group the pin budget can hold there), where `n* = min { n : stream(n) > curve_cpu(n) }` from the probed curves (SC-6); `replicas` (M2): one copy of each of a stage's first `d` MoE layers' `Router` groups on the previous stage, `d` being RFC-0023 §9's prediction lookahead, with their bytes charged to that stage's tier in C9 and the evaluator |
| `expert_cache` | `ExpertCacheBudget` per GPU (VRAM over a `HostPinned` home) and per RAM tier (over an `Nvme` home), with `per_layer`, `policy` = RFC-0023's default, `miss_split` |
| `kv` | §11.1 item 5, with `spill_compute` (`GpuStream` or `CpuSplit`) per stage and KV class from C7's shares; RFC-0022 executes what the plan says, falling back only to a certified path |
| `per_site_layout` | for each multi-reader group with a layout-pass report (RFC-0040 §5.4: the shared layout's `Σ f_r · t_r`, and each per-site layout's time and extra bytes), the evaluator compares the plan with the shared layout against the plan with per-site copies, the extra bytes charged to the tiers that hold them; the group is marked only when per-site copies lower `t_tok` and every tier still fits. Shared is the default, the choice whenever no report exists yet, and always the choice on `Unified` devices |
| `wan_rate_class` | per node, `⌊log2⌋` of `CapabilityProfileExt::wan_down` in MB/s (RFC-0010 §6.1) as this solve read it; a node without a measurement gets no entry, and RFC-0041 §8.3's consumers apply their median rule. Consumers read it from the committed plan only, so a class changes at the next epoch |
| `prefetch` | `Nvme → HostPinned`: `lookahead_layers = 1`, `max_inflight = min(RAM slack, qd* · expert bytes)` with qd* the smallest queue depth reaching 95% of sustained read in the probe grid; `HostPinned → Vram`: `max_inflight` = staging slots × expert bytes (RFC-0021, RFC-0023 refine both) |
| `scratch` | `σ_d` per used device |
| `speculative` | §7.3; `None` when γ* = 0 |
| `batching` | `max_sequences` from RFC-0054's instance key (latency default 2: the active sequence and one queued); the plan's **`context_length`** = the largest L ≤ the model's maximum (and ≤ RFC-0054's `context_length` when set, and never below `solver.context_target`) for which C9 holds with the active sequence's KV at L and the reserved sequences' KV at L in their tiers, after C13's second stage (host KV is always `HostPinned`, so the pin row can bind); the committed plan publishes it; published in `PredictedCost::context_length` and checked by the API (§13). **`prefill_chunk`** is chosen with the TTFT model: for each candidate c from 256 up to the activation-scratch and frame-size limit, evaluate the predicted TTFT of 4K- and 32K-token prompts and keep the c that minimises it. Per chunk, the model charges (a) dense streamed groups at `max(compute, bytes / W_pcie)`, which hides once `c ≥ n_hide = F_gpu · b / (2 · W_pcie)` (SC-10), and dense groups that C13 left pageable at their CPU prefill time `2 · params · c / F_cpu`; (b) streamed **expert** groups by the tokens each expert actually sees, `c · k / E`, so hiding needs `c ≥ n_hide · E / k` (≈ 63,000 tokens on DeepSeek-V3 [estimate: 1,980 × 32], above any practical chunk, so tiered MoE prefill is copy-bound per chunk); (c) the NVMe-resident share of the routed pool at `W_nvme`, re-read every chunk because a chunk of ≥ 256 tokens touches essentially every expert (P(untouched) = (1 − 8/256)^256 ≈ 3 · 10⁻⁴ for DeepSeek-V3); (d) prefill hops, `c · S_tok / β`, against the next stage's compute; and (e) for EP islands, each owner's dispatch and combine frames. A row goes to every owner holding at least one of its k experts, so an owner of m experts receives `rows_m = c · (1 − C(E − m, k) / C(E, k))` rows per chunk, not `c · k / E`. That is `rows_m · S_tok` bytes, split into `⌈rows_m · S_tok / part_limit⌉` parts under the transport's maximum frame or part limit (RFC-0031), each part charged its α. For DeepSeek-V3 (E = 256, k = 8) on two owners of 128 experts, `C(128, 8) / C(256, 8)` ≈ 0.0035, so at c = 4096 each owner receives ≈ 4,082 rows × 28 KiB ≈ 112 MiB per dispatch [estimate]. There is no fixed 8192 cap. For stages with `Nvme`-homed pools, prefill runs in **layer-major groups** (RFC-0012 §4.4): G chunks share one pass over the pool, so the model charges `t_pool` = NVMe-resident pool bytes / `W_nvme` + RAM-resident pool bytes / `W_pcie` once per group rather than once per chunk, with G the largest group whose activations fit the stage's scratch. `prefill_ladder` holds the captured sizes {64, 256, 1024, 4096} below the chosen chunk plus the chunk itself (RFC-0012 §2) |
| `predicted`, `assumptions` | evaluator; every `Prior` or `Derived` input used |
| `transition` | `MakeBeforeBreak` when, on every node and tier, the old plan's reservation plus the new plan's requirement fit `usable`; else `BreakBeforeMake` (RFC-0001 §7.3) |

**Encoding.** A plan is serialised compactly: interned tables of the `NodeId`s, `TierId`s, `DeviceId`s and `LinkId`s it uses, then the fixed-size records (stages, budgets, prefetch windows), then one 8-byte record per tensor group in census order (owner index u8, home index u16, compute index u16, stream-site index u16 with 0 for none, replica count u8, with replicas in a side table). A plan whose encoding exceeds 64 KiB travels by digest over the bulk plane, and `PlanCommitted` carries the digest and a summary (amendment B2).

#### 11.3 Evaluator and repair

`evaluate(plan, input)` recomputes every term of `PredictedCost` from the plan's integral assignments with the same coefficients the DP and the MILP use, and checks every tier on every node at its predicted peak. It is the single source of truth for predictions, routing, previews and tests: the DP and the MILP are only search methods, and a plan's numbers always come from the evaluator.

If rounding produced a violation, a repair loop moves the group with the lowest time saved per byte on the violated tier to the next slower tier, re-evaluates, and repeats (at most 200 moves). If repair fails, the DP plan is used. Both events are counted in `SolveStats`.

### 12. Incremental re-solve on churn

#### 12.1 Triggers

| Trigger (RFC-0001 §7.2) | Warm start | Limits | Adoption |
|---|---|---|---|
| `NodeFailed`, `NodeLeft`, `NodeRevoked`, an edge lost, or a runner failure affecting the instance | `solve_repair` with RFC-0033's `RepairConstraints` (§12.4) | repair (≤ 2 s total, RFC-0033 §6.4) | the strategy with minimal `t_resume`, chosen by RFC-0033 §6.2; no switch to another plan during recovery |
| Restart budget spent on a runner | `solve_repair` with the faulty devices removed from the candidates | repair | as above |
| `NodeAdmitted` | the current plan, re-evaluated (the new node unused, so still feasible) | normal | hysteresis and payback (§12.2) |
| `GraphUpdated { cause: Drift }` | current plan re-evaluated on the new graph | normal | §12.2 |
| `RequestReplan { WorkloadDrift }` | delta solve first (§12.3); a full solve only if the delta solve cannot recover half the predicted loss | delta, then normal | delta: in place; full: §12.2 |
| `WorkloadDrift { ExpertTierShortfall { tier } }`: a cache tier's measured hit rate falls short of its curve past RFC-0010 §7.1's thresholds. The comparison points are fixed: the VRAM tier's hit rate is compared with `H(y)`, and the RAM tier's hit rate *conditional on a VRAM miss* with `(H(u + y) − H(y)) / (1 − H(y))`, per layer, from `WorkloadStats::tier_hit` | delta solve with that tier's curve capped at the measured hit rate at its budget, so the solver stops crediting hits the tier cannot deliver (for example under throttled admission, RFC-0021 §8); the cap is lifted when the shortfall clears | delta | in place |
| Policy change, pins, load or unload | DP | normal | as requested |

#### 12.2 Adoption rule for deferred re-plans

A deferred plan P' replaces the current plan P only if all of these hold (RFC-0001 §7.2):

```text
gain      = (t_tok(P) − t_tok(P')) / t_tok(P)               throughput mode: relative tok/s gain
move_time = Σ over moved bytes / β of the path that supplies them  + downtime(P → P')
downtime  = 0 for MakeBeforeBreak; drain + load + warm time of the slowest new runner otherwise
gain ≥ replan.min_gain                                       default 0.20
(t_tok(P) − t_tok(P')) · (replan.payback_horizon / t_tok(P)) ≥ move_time      the move pays back in time
no adopted re-plan of this instance in the last replan.cooldown                   default 10 min
```

In latency mode an adopted deferred plan also waits for an idle window (no live sequence on the instance) of 2 s, for at most `replan.max_defer` (default 5 min); after that it is applied with make-before-break if possible, otherwise at the next idle window regardless of the deferral limit. Gains below `min_gain` are kept as a "pending improvement" in the explanation, so the dashboard can show what an operator could get by forcing a re-plan.

#### 12.3 Delta solve

Workload drift (hit curves, acceptance, context mix) often needs only budget changes, which runners adopt in place without reloading (RFC-0001 §5.4 `PlanDelta`). The delta solve fixes every structural variable to its current value — `v`, `x`, `f`, `g`, the dense shares `w` (home tiers), and `u` where it equals the pool (experts homed in RAM) — and re-optimises the rest: KV shares `ρ`, expert cache sizes `y`, RAM cache sizes `u` over `Nvme` homes, miss splits `m^P`/`m^C`, plus the speculative outer loop and batching limits. On the DP path (§9.5) it re-runs steps 2–5 of the stage cost (§10) on each stage with that stage's range and homes fixed: they are separable concave allocations, which the greedy's slope order solves exactly. Where the MILP is built, the same re-optimisation is an LP with no binaries, solved in milliseconds [estimate: ≈ 10⁴ columns, no branching]. A delta that improves predicted `t_tok` by ≥ 2% is committed as a new epoch with `ApplyPlanDelta`; the 20% threshold does not apply because nothing moves.

```rust
/// RFC-0001 names DeltaItem; this RFC defines it. Every variant except ExtendShard leaves the
/// runner's shard unchanged; ExtendShard keeps that shard and adds another stage's layers.
/// Relink and ExtendShard are RFC-0032's (§4.1, amendment 1); the delta solve never emits them.
pub enum DeltaItem {
    ExpertCache { device: DeviceId, budget: ExpertCacheBudget },   // bytes, per-layer split, miss split
    Kv { stage: StageIx, budgets: SmallVec<[KvBudget; 3]> },
    Prefetch { window: PrefetchWindow },
    Speculative { config: Option<SpeculativeConfig> },
    Batching { limits: BatchingLimits },
    Predicted { cost: PredictedCost },                             // always present in a delta
    /// Same shard; a different neighbour, hop transport, relay or stage index (RFC-0032 §4.4).
    /// The runner stays Ready { degraded: Relinking } while it applies.
    Relink { stage: StageIx, prev: NodeId, next: NodeId, hop: HopPlan },
    /// The runner keeps its shard and stage and loads the listed stages' layers: a survivor
    /// absorbing a lost stage after a repair solve (§12.4, RFC-0033 §7.2). It reports
    /// Ready { degraded: Absorbing } until they are resident.
    ExtendShard { stages: SmallVec<[StageIx; 2]> },                // layers and tiers from the new plan
}

/// RFC-0001 names Assumption; this RFC defines it.
pub struct Assumption {
    pub input: AssumedInput,          // e.g. QuantBw { device, tier, scheme } | HitCurve { layer } | LinkP99 { link }
    pub source: MeasureSource,        // Prior | Derived
    pub sensitivity: Nanos,           // change in predicted t_tok if the input were 20% worse
}
```


#### 12.4 Repair solves

RFC-0033 calls the solver after a stage loss, runner failure or edge loss with `RepairConstraints { keep_shards, candidates, objective, max_move_bytes }` and picks a `RepairStrategy` by time to resume (RFC-0033 §6.1–§6.2). `solve_repair` returns one candidate plan per feasible strategy with its `t_resume`:

```text
solve_repair(input, current, c):
    if c.keep_shards:                                  // default: survivors keep KV and history
        fix every surviving placement node's x, w^A, w^F, u, u^P and homes to `current`
        RestartInPlace:  if the failed runner's node is healthy and its pack is valid: the current
                         plan unchanged; t_resume = t_load + t_warm + t_replay
        Replace { n }:   for each candidate node n outside the ring, or with spare capacity: assign the
                         lost range [i, j) whole to n; place it with stage(n, i, j) at full block
                         granularity; add the hops to its ring neighbours; check C9 on n
        Absorb { into }: split the lost range between its ring neighbours (the predecessor extends
                         forward, the successor backward), trying every legal split point; absorbed
                         groups go to the neighbours' remaining capacity, including HostPageable and,
                         as a degraded plan, dense on NVMe (C10)
                         if the neighbours cannot hold it: a non-adjacent survivor may take the range as
                         a second stage (`MultiStageOnNode`): for that node only, Σ_b π_{b,p} ≤ 2 in C3,
                         its in- and out-degree bounds in C4 become 2, and the DP charges both ranges
                         to its capacity jointly
    if no keep_shards strategy is feasible, or c.keep_shards is false:
        Resolve:         a fresh solve over survivors (DP; then the MILP within the remaining budget,
                         where it is built, §9.5)
    for each candidate: t_resume per RFC-0033 §6.2 = t_solve + missing bytes / β_fetch
                        + t_pack + t_load + t_warm + t_replay, with t_replay from the evaluator
                        (history bytes / β + the lost layers' prefill over the affected contexts)
    drop candidates that exceed c.max_move_bytes; return the rest
```

- **Budget.** All candidates together within RFC-0033 §6.4's ≤ 2 s [target]. `Replace` and `Absorb` evaluate one or two stages each with the 32-block stage cost, which is milliseconds; `Resolve` uses whatever budget remains.
- **No mid-recovery switch.** The chosen candidate stands until the instance is `Ready`. A better plan found later arrives through an ordinary deferred re-plan under §12.2's hysteresis.
- **Before RFC-0033 lands (M2).** Node loss uses `Resolve` with re-prefill, which RFC-0033 also uses as its universal fallback.

### 13. Fast per-request routing

Routing runs on the ingress node from its replica of cluster state (RFC-0001 §6.1, invariant 2). It never moves weights. This section owns `RouteRequest`, `RouteDecision`, `RoutingView` and the chunk rule; RFC-0032 §8.2 refers here rather than defining its own (`solver::route`, RFC-0001 §6.1).

```rust
pub struct RouteRequest {
    pub model: ModelRef,
    pub prompt_tokens: u32,
    pub prefix_keys: SmallVec<[PrefixKey; 4]>,   // candidate prefix hashes at block boundaries (RFC-0022)
    pub session: Option<SessionId>,
    pub exactness: ExactnessPolicy,              // resolved per RFC-0001 §11.3
    pub greedy: bool,                            // greedy sampling accepts more drafts than sampled
    pub priority: Priority,
    pub deadline: Option<Deadline>,
}
pub struct RouteDecision {
    pub instance: InstanceId,
    pub epoch: PlanEpoch,
    pub prefill_chunk: u32,                      // one of the instance's captured ladder sizes
    pub initial_window: u16,                     // initial γ for RFC-0012's controller (≤ max_window)
    pub predicted_ttft: Nanos,
    pub admitted: Admission,                     // Now | Queued { position } | Rejected { retry_after }
}
/// What `route()` reads on the ingress node. RFC-0032 §8.2 supplies the inputs.
pub struct RoutingView<'a> {
    pub instances: &'a [RoutableInstance],       // routable instances, from the replica's committed plans
    pub load: &'a [LoadDigest],                  // per (instance, stage), with age (RFC-0030 §5.6)
    pub ingress_alpha: &'a [(NodeId, Nanos)],    // ingress ↔ node α from LinkCaps, for heads and tails
    pub sessions: &'a [(SessionId, InstanceId)], // session pins (RFC-0022)
    pub prefix: &'a [(InstanceId, PrefixHints)], // prefix hints per instance
    pub now: MonoTime,
}
pub struct RoutableInstance {
    pub instance: InstanceId,
    pub epoch: PlanEpoch,
    pub model: ModelRef,
    pub head: NodeId,
    pub tail: NodeId,
    pub stages: u8,                              // D
    pub exact: bool,                             // the plan's exactness
    pub context_length: u32,
    pub prefill_chunk: u32,
    pub ladder: SmallVec<[u32; 5]>,              // captured prefill sizes (RFC-0012 §2)
    pub prefill_per_chunk: SmallVec<[Nanos; 8]>, // StageCost::prefill_per_chunk, per stage
    pub gamma_star: u16,
    pub sampled_acceptance: Option<f32>,         // learned; lowers the initial window (below)
    pub kv_capacity: Bytes,                      // static capacity, used when digests are stale
}
pub struct PrefixHints { /* the instance's published filter of prefix keys, RFC-0022 */ }
```

For each routable instance i of the model whose plan's exactness is compatible and whose `context_length` covers `prompt_tokens + max_tokens` (RFC-0054; if none does, routing returns `context_length_exceeded` with the largest planned limit, RFC-0050 §4.2, and never re-plans for one request: the plan already holds the largest context its memory allows for `max_sequences`, §11.2, and the live context p90 in `WorkloadStats::context` feeds the next re-plan's `L_read` instead):

```text
u      = prompt_tokens − cached_prefix(i)             cached prefix from i's published prefix digest
                                                      (a compact filter of prefix keys, RFC-0022), or the
                                                      session's KV if i holds it
c      = prefill chunk: plan.batching.prefill_chunk, or for short prompts the smallest captured
         ladder size ≥ ⌈u / D⌉, so that all D stages work on the prompt at once
m      = ⌈u / c⌉
ttft_i = Σ_s t_s(c) + (m − 1) · max_s t_s(c) + Σ_hops (α + c · S_tok / β)    pipelined chunks (cost model §3.7)
         + α(ingress → head) + α(tail → ingress)
wait_i = queued tokens at i's head / i's recent prefill rate              from the last telemetry digest
score_i = ttft_i + wait_i − session_bonus · [i holds the session]
```

`t_s(c)` is each stage's prefill time for a chunk, which the plan carries as `StageCost::prefill_per_chunk` at the plan's chunk size and scales linearly in c between the evaluator's sample points. The best score wins; ties go to the instance holding the session, then the lower `InstanceId`. KV admission (RFC-0001 §8.4 rule 5) is checked on the winner: the sequence's expected KV (prompt plus `max_tokens`, capped at the context target) against the instance's advertised headroom; otherwise the request queues (bounded) or is rejected with `Capacity`. The initial window is the plan's γ*, lowered by one when the request is sampled rather than greedy and the learned sampled acceptance is below the greedy one. The decision reaches the head in `SequenceOpen` and `SequenceMeta`, which gain `prefill_chunk` and `initial_window` fields (amendment B6), so the head applies exactly what the router chose.

Routing is O(instances) arithmetic on the replica, well under the 1 ms target (SC-31). If no instance is routable, ingress submits `EnsureInstance` and holds the request (RFC-0001 §6.1 step 3).


### 14. Plan explainability

Every committed plan has a `PlanExplanation`, stored by digest (`PlacementPlanExt::explanation`) and served by the debug surface and the dashboard (RFC-0001 §13, RFC-0051). It answers four questions a user asks: why is this tensor here, why is that device unused, what is the bottleneck, and what would make it faster.

```rust
pub struct PlanExplanation {
    pub epoch: PlanEpoch,
    pub headline: Headline,                       // predicted tok/s, TTFT, bottleneck term, path used
    /// Marginal value of capacity: seconds per token saved by one more GiB in each tier on
    /// each node, from the LP duals of C9 with the integers fixed at the solution.
    pub shadow_prices: Vec<(TierId, f64 /* ms per token per GiB */)>,
    pub groups: Vec<GroupReason>,                 // one per census group, aggregated per block in the UI
    pub devices: Vec<DeviceReason>,               // used, unused, filtered, not considered
    pub stages: Vec<StageReason>,                 // why this range, why this order
    pub bottlenecks: SmallVec<[(StageIx, CostTerm, f32 /* share of t_tok */); 4]>,
    pub alternatives: Vec<Alternative>,           // computed what-ifs, bounded (below)
    pub assumptions: Vec<Assumption>,             // the unmeasured inputs and their effect
}
pub struct GroupReason {
    pub group: TensorGroupId,
    pub home: TierId, pub compute: DeviceId,
    /// Time per token this group costs where it is, and where it would cost next best.
    pub cost_here: Nanos, pub next_best: Option<(TierId, Nanos)>,
    /// cost_next − cost_here, per GiB of the group, compared with the shadow price of the tier
    /// it occupies. A group is in a fast tier because its saving per GiB beats that price.
    pub saving_per_gib: f64,
    pub binding: Option<BindingReason>,           // Pinned | CapacityExhausted { tier } | NoKernel | Policy
}
pub enum DeviceReason {
    Used { stage: StageIx, share_of_t_tok: f32 },
    Unused { cost_if_forced: Option<Nanos> },     // marginal t_tok if forced in, when computed
    Filtered { rule: FilterRule, detail: String },// F1–F7 of §4
    NotConsidered { score: f64, rank: u16 },      // below the top max_nodes
}
pub enum Alternative {
    SmallerQuant { sibling: VariantSummary, predicted_tok_s: f32, fits: bool },   // ModelManifest::siblings
    MoreRam { node: NodeId, gib: u32, predicted_tok_s: f32 },
    DropDevice { device: DeviceId, predicted_tok_s: f32 },
    WiredLink { link: LinkId, predicted_tok_s: f32 },                           // when a wireless hop binds
    PendingReplan { predicted_gain: f32, blocked_by: AdoptionBlock },           // §12.2
}
```

**How reasons are computed.**
- *Shadow prices.* On the DP path (§9.5), the price of a tier is the time saved per byte by the marginal group or hit-curve segment on it, the one the greedy placed last or only partly: in a fractional knapsack that critical ratio is the LP dual of the tier's capacity row. Where a joint host row binds instead (C9), its price is shared by both host tiers it covers. Where the MILP is built, the integer variables are fixed after the solve and the LP is re-solved once; the row duals of C9 (`Highs_getSolution`'s `row_dual`) give the same quantity. Converted to ms per token per GiB, it is the number the dashboard shows as, for example, "a GiB of VRAM on this GPU is worth 14 ms per token here" (scenario (a), §15.1).
- *Group reasons.* From coefficients only: `cost_here = bytes · coefficient(home pair)`, `next_best` the cheapest other pair with capacity. Text templates turn them into sentences, for example (illustrative numbers): "Layer 37 FFN (412 MiB) is in RAM: in VRAM it would save 0.47 ms per token, 1.14 ms per GiB, below this GPU's VRAM price of 2.1 ms per GiB, which the KV hot window and the expert cache pay more for."
- *Device reasons.* Filtered and not-considered devices come from §4. For unused devices that passed the filter, `cost_if_forced` is computed lazily when the dashboard asks: on the DP path, the best entry of the DP table whose node subset contains the device's node; where the MILP is built, a re-solve with `v_p = 1` forced, run in the background on the coordinator at low priority with the normal limits. A departure-prone node left out by §4's margin is `Unused` with this figure and the margin as its reason.
- *Alternatives.* At commit time, the solver evaluates (without new MILP solves) the plan's structure with each sibling quantization's byte totals, and the DP with each single device dropped. At most 8 alternatives are kept, ordered by predicted gain. The quant comparison matters most on bandwidth-poor machines: on the cost model's (a) and (b2) rigs a smaller quant is the largest lever (2.5× and ≈1.6× respectively, §4a step 5, §4b2), and the user should see it next to the plan.
- *Assumptions.* Each `Prior` or `Derived` input is listed with the change in predicted `t_tok` if it were 20% worse, so the user can see which missing measurement matters.

The explanation is computed after the plan is chosen and never changes it.

#### 14.1 Placement preview

The dashboard's preview view, `syntropy plan`, the first-run model picker (SC-24) and automatic placement all show what a placement would do before anything loads. RFC-0051 §3.1 fixes the contract, field by field; this section implements it, and RFC-0051's `FitEstimate` and RFC-0050 §13.3's `NodeDelta` are projections of `Preview`. `preview` is pure and runs on the ingress node against its replica of the live resource graph, including current reservations.

```rust
pub fn preview(input: &PreviewInput) -> Preview;

pub struct PreviewInput<'a> {
    pub graph: &'a ResourceGraph,                   // live, with reservations
    pub census: CensusSource<'a>,                   // Full(&ModelManifest) | Summary(&CensusSummary)
    pub policy: &'a InstancePolicy,
    pub workload: &'a WorkloadStats,                // priors where nothing is learned yet
    pub evictable: Vec<InstanceId>,                 // idle instances the caller would unload, LRU first
    pub node_overrides: Vec<(NodeId, ResourceOverlay)>, // what-if, e.g. the headroom slider
    pub ttft_prompts: SmallVec<[u32; 4]>,           // default [1024, 4096, 32768]
    pub candidates: u8,                             // 1..=3
    pub effort: PreviewEffort,                      // DpOnly | Full (islands included; plus the MILP where built, §9.5)
}
/// Defined in RFC-0041 §3.3 (with `LayerSummary` and `LookupSummary`), produced by one pure
/// `census::summarize`, so the catalogue's copy equals a node's: exact sizes, uniform prior for
/// popularity. Enough for the DP, not for per-group assignment, so a summary-based preview
/// carries `precision: Summary`.
pub struct CensusSummary { /* RFC-0041 */ }

pub struct Preview {
    pub verdict: FitVerdict,                        // Fits | Degraded { why } | DoesNotFit(InfeasibilityReport)
    pub candidates: Vec<PreviewCandidate>,          // ranked by decode_tok_s, at most `candidates`
    pub assumptions: Vec<Assumption>,               // MeasureSource::Prior or Derived inputs: "not yet measured"
    pub precision: PreviewPrecision,                // Exact (full census) | Summary (± one tensor group per block)
    /// One informational line about an approximate knob, or None. Never a named preset.
    pub approx_hint: Option<ApproxHint>,            // { knob: ApproxKnobs, predicted_gain: Figure<f32>, cost: KnobCostRef }
}
impl Preview {
    /// The top-ranked candidate; RFC-0050 §4.3 reads `best().evicts` and `best().time_to_ready`.
    pub fn best(&self) -> Option<&PreviewCandidate> { self.candidates.first() }
}
pub struct PreviewCandidate {
    pub kind: CandidateKind,                        // Best | NoEvictions | FewerNodes | SmallerQuant { sibling }
    pub decode_tok_s: Figure<f32>,                  // batch 1; Estimate; basis names the plan's PredictedCost
    pub predicted: PredictedCost,                   // per stage, per CostTerm, the figures came from
    pub ttft: Vec<(u32, Figure<Nanos>)>,            // one per ttft_prompts length; Estimate
    pub dominant_term: CostTermShare,               // { term: CostTerm, node: NodeId, share: f32 }
    pub per_node: Vec<NodeDelta>,                   // per node and tier: before (Measured) and after (Estimate),
                                                    // each split by RFC-0001 §9.1/§9.3 purpose
    pub fetch: Vec<NodeFetch>,                      // per node: bytes by source (local store, named cluster node,
                                                    // origin) and its ETA; Estimate
    pub time_to_ready: Figure<Nanos>,               // Estimate, with parts below
    pub time_to_ready_parts: ReadyParts,            // { fetch, pack_load, warm }: each a Figure<Nanos>
    pub disk_needed: Vec<(NodeId, Figure<Bytes>)>,  // chunks plus pack
    pub ttft_ref: (u32, Figure<Nanos>),             // TTFT at the cap's L_ttft, shown beside decode_tok_s (§7.1)
    pub jitter: Option<Jitter>,                     // stall rate and p99 token time on wireless rings (C14)
    pub evicts: Vec<InstanceId>,                    // the prefix of `evictable` this plan needs unloaded
    pub transition: TransitionStrategy,             // MakeBeforeBreak | BreakBeforeMake
    pub plan: PlacementPlan,                        // uncommitted; commit re-solves (RFC-0051 §5.4)
}
/// RFC-0051 A2: every predicted or measured number crosses the API as a Figure.
pub struct Figure<T> { pub value: T, pub kind: FigureKind /* Measured | Estimate | Target */, pub basis: Option<String> }
```

- **Labels.** `before` bytes are `Measured` (the live ledger); everything predicted is `Estimate`, with `basis` naming the formula (for example "SC-23 bound with node pc's measured W_read"); a number that is a goal rather than a prediction is `Target`. Inputs that are `Prior` or `Derived` are listed in `assumptions`, so the views can say "not yet measured".
- **Candidates.** `Best` is the solve at the requested effort, within the TTFT cap from M2 (§7.1). `NoEvictions` is added when `Best` evicts an instance; `FewerNodes` when a plan with one node fewer is within 20% of `Best`'s tok/s; `SmallerQuant` when `Best` does not fit or is slower than the best fitting sibling. Candidates are ranked by `decode_tok_s`.
- **Decode and TTFT together.** Every candidate carries `decode_tok_s` and `ttft_ref`, and the views show both side by side, so a plan that buys decode speed with a slower first token shows that trade. When the cap excluded a faster-decoding plan, the explanation says so with that plan's two numbers.
- **Jitter badge.** A candidate whose ring has a `WiFi` or `Overlay` edge carries `jitter`: the predicted stall rate (C14) and the p99 token time. `decode_tok_s` is the mean, and the badge shows the tail next to it. RFC-0051 §5.4 draws the badge in the preview, and §3.1 carries the field.
- **Approximate-knob line.** Exact output is the default (D3), and there is no named approximate preset. `approx_hint` is set only when an approximate knob that applies to this model has a `KnobCost` report (RFC-0060 §6) and its predicted decode gain on this plan is ≥ 1.5× [target]; it carries the knob, the gain and the report's measured cost. `syntropy plan` and the dashboard preview show it as one informational line (RFC-0051, PROD), and nothing is applied unless the user sets the knob.
- **Verdict.** `Fits` when `Best` needs no degraded mode; `Degraded { why }` when it needs one (dense on NVMe, two stages on one node); `DoesNotFit` with the `InfeasibilityReport` below otherwise.
- **Time to ready.** `fetch` is the slowest node's bytes per source over that source's measured β; `pack_load` is SC-23's bound `1.25 · (2 · S / W_read + S / W_write)` with the node's measured rates; `warm` is the measured capture and canary time from the node's last `Warming` of this model, else a 10 s prior marked as an assumption.
- **Dominant term.** The `CostTerm` and node with the largest share of predicted token time, which gives the picker's one-phrase reasons ("limited by RAM bandwidth on pc").
- **Census sources.** A registered manifest or Hub headers give an exact census; the catalogue's summary gives exact sizes and prior expert popularity (RFC-0051 §3.1). For MoE models that need NVMe, `decode_tok_s.basis` states that speed depends on routing locality, and the views show the range over hit rates. Before any trace from the user's own machine exists, the range runs from the uniform prior to the project's corpus-derived hit curve for the family (RFC-0023 §13, from S-01's traces), labelled as corpus-derived.
- **What-if.** `node_overrides` change a copy of the graph the preview solves on, so the headroom slider and device toggles show their effect without touching the live graph.
- **Budget.** `DpOnly` previews run the island-free DP (`Local` and `IntraTp` placement nodes; islands only in `Full`), and candidates that share a reservation and a census share one DP table: `FewerNodes` is a minimum over smaller node subsets, which `Best`'s table already holds; `NoEvictions` needs a table on its own reservation, and each `SmallerQuant` sibling one on its own census. At ≈ 75 ms per table for 8 nodes [estimate, §10] the target is ≤ 100 ms per table: ≤ 200 ms for `Best`, `FewerNodes` and `NoEvictions`, plus ≤ 100 ms per sibling [target]. RFC-0051 §3.3's `recommend()` therefore costs one table per catalogue variant. This is what lets the picker order the whole catalogue. `Full` costs a slow-phase solve and is used for the picker's shortlist and the dashboard's "Preview" button. §17 guards both.

```rust
/// Returned by solve, solve_repair and preview when no plan fits (§17).
pub struct InfeasibilityReport {
    /// Per node and tier, for the closest DP attempt: bytes needed against usable (RFC-0051 §3.1).
    pub per_tier: Vec<(NodeId, TierId, Figure<Bytes> /* needed */, Figure<Bytes> /* usable */)>,
    /// The C9 rows, joint host rows or pins that bind, each with a sentence for the user,
    /// e.g. "HostPinned on pc: the OS lock limit (pin_limit) is 64 GB".
    pub binding: Vec<BindingConstraint>,               // { id: ConstraintId, words: String }
    pub conflicting_pins: Vec<PlacementPin>,
    pub smallest_fitting_sibling: Option<(VariantSummary, Figure<f32> /* decode tok/s */)>,
    pub extra_memory: Vec<(NodeId, TierKind, Figure<Bytes>)>,  // what would make this variant fit
    pub degraded_option: Option<DegradedReason>,       // e.g. fits only with dense bytes on NVMe
}
```


### 15. Worked examples

All numbers are **[estimate]**, computed with the cost model's assumed inputs for each scenario (`hardware-and-performance-model.md` §4). In production the probes replace every input. `t_fixed` is taken as 1 ms per stage (about the SC-2 target) unless stated.

#### 15.1 Scenario (a): Llama-3.3-70B Q4_K_M (42.5 GB) on one RTX 4090 + 64 GB DDR5 + PCIe 4.0 NVMe

Inputs: W_eff 800 GB/s (VRAM), 70 GB/s (CPU over DDR5; 55–85), pinned PCIe 25 GB/s, KV 2.68 GB per sequence at 8 K, context 0.5 GB, scratch 1.5 GB, and the RFCs' defaults: `vram_headroom` 512 MiB on a display GPU (RFC-0020 §8), `max_sequences` = 2 in latency mode (RFC-0054: the active sequence and one queued), and 6 GB of RAM in use by the OS and desktop [assumption].

| Decision | Solver result | Reason it shows |
|---|---|---|
| Placement nodes | one, `Local` | single node |
| VRAM | 0.5 context + 1.5 scratch + 0.54 headroom + 2.68 active KV + **18.8 GB dense** | VRAM's shadow price is 1/70 − 1/800 s/GB ≈ **13.0 ms per token per GB**; every dense byte saves exactly that by moving up, so VRAM fills |
| RAM | 23.7 GB dense in `HostPinned` (C13), computed in place by the CPU at decode and streamed to the GPU for prefill; ≈ 1 GB of staging and the queued sequence's KV (2.68 GB, reserved with no read cost, C7), also pinned, since KV's host tier is always `HostPinned` (RFC-0022 §3) | streaming at n = 1 would cost 1/25 s/GB, 2.9× the CPU's 1/70, so decode runs in place, where a pinned home costs nothing (RFC-0020 §2: a CPU reads locked and unlocked memory at the same rate). The home is pinned so that prefill can stream: the pin row holds 23.7 + 1 + 2.68 ≈ 27.4 GB within `pin_limit` = min(`host_usable` ≈ 51.6, 0.85 × 64 = 54.4, `lockable`) ≈ 51.6 GB on Linux when the probe finds provider registration uncharged to `RLIMIT_MEMLOCK` or the daemon holds `CAP_IPC_LOCK`, and within Windows' 0.50 × 64 = 32 GB cap. On Linux the pin row cannot bind before the joint row, so `context_length` is unchanged; on Windows it binds first (next row) |
| `context_length` | ≈ 45 K tokens on Linux, ≈ 15 K on Windows [estimate] | KV capacity for 2 sequences: the 2.68 GB VRAM window plus the pinned RAM left for KV, divided by 2 × 320 KiB per token. Linux: ≈ 26.9 GB (host_usable ≈ 64 − 6 − 6.4 headroom ≈ 51.6 GB, minus 23.7 GB dense and ≈ 1 GB staging). Windows: the 32 GB pin cap leaves ≈ 7.3 GB, so ≈ 15 K, still above the 8 K `solver.context_target` that C13 leaves room for |
| `compute.batched` | `StreamSite { min_tokens: n* }`, n* ≈ 12 at F_cpu = 1 TFLOPS, ≈ 24 at 2 TFLOPS | CPU curve `min(70, F_cpu · 0.6 / (2n))` GB/s crosses 25 GB/s there; streamed groups need a `HostPinned` home, so groups the solver streams in prefill are homed there |
| `prefill_chunk` | 2048, from the TTFT model (§11.2) | dense streamed copies hide once c ≥ n_hide = 165 TFLOPS · 0.6 / (2 · 25 GB/s) ≈ 1980 (SC-10) |
| Predicted TTFT, 4 K prompt | ≈ 3.6–5.9 s [estimate] | two 2048-token chunks with every prefill FLOP on the GPU: 591 TFLOP (cost model §4a step 4) at 100–165 TFLOPS effective. Each chunk streams the 23.7 GB of pinned dense groups in 23.7 / 25 ≈ 0.95 s, under the chunk's 1.8–3.0 s of compute, so the copies hide. If `pin_limit` could not hold them (for example, a small `RLIMIT_MEMLOCK` that charges provider registration), C13 leaves them pageable and they prefill on the CPU: ≈ 38.8 B parameters × 2 × 4096 ≈ 318 TFLOP at the cost model's 1.5 TFLOPS ≈ 212 s, which `PredictedCost::ttft`, the preview and the explanation show |
| Predicted t_tok | 23.5 + 338.9 + 3.4 + 1 ≈ **367 ms → 2.73 tok/s** | the cost model's 2.8 tok/s (§4a) used 19.3 GB in VRAM, with no headroom and one sequence's KV |

Speculative outer loop with a 1B Q4 draft model (≈0.7 GB) on the GPU at a = 0.7: the draft displaces 0.7 GB of dense bytes to RAM, which is charged at the pass's RAM rate (rate 70 at T ≤ 4, 37.5 at T = 8 on a 1 TFLOPS CPU), and it costs ≈1.9 ms per draft step. CPU throughput decides γ through the batch curve:

| γ | T = γ+1 | E[tokens] | CPU RAM term at 1 TFLOPS | t_tok at 1 TFLOPS | CPU RAM term at 2 TFLOPS | t_tok at 2 TFLOPS |
|---|---|---|---|---|---|---|
| 0 | 1 | 1.00 | 339 ms | 367 ms | 339 ms | 367 ms |
| 3 | 4 | 2.53 | 349 ms (rate 70) | **151 ms (6.6 tok/s)** | 349 ms | 151 ms |
| 4 | 5 | 2.77 | 407 ms (rate 60) | 159 ms | 349 ms | 138 ms |
| 7 | 8 | 3.14 | 651 ms (rate 37.5) | 220 ms | 349 ms (rate 70) | **124 ms (8.1 tok/s)** |

So γ* = 3 on a 1 TFLOPS CPU and γ* = 7 on a 2 TFLOPS CPU, within the cost model's 6–9 tok/s (§4a step 5). The difference comes entirely from the probed batch curve (ADR-0100). The explanation's top alternative is the IQ3_XXS sibling (27.5 GB): RAM part 8.7 GB, ≈152 ms → 6.6 tok/s without speculation, with its quality cost stated. The golden test (PLC-E6-T09) uses exactly these inputs and defaults.

#### 15.2 Scenario (b2): DeepSeek-V3 Q4_K_M (404 GB) on RTX 4090 + 128 GB DDR5 + PCIe 4.0 NVMe

Hot set 9.7 GB, routed pool 394 GB, routed reads 12.3 GB per token, KV at 32 K 2.25 GB per sequence, NVMe sustained 6 GB/s. Host budgets follow RFC-0020 §5 with 8 GB of RAM in use by the OS and desktop [assumption]: headroom = max(4 GiB, 0.10 × 128 GB) = 12.8 GB, so `host_usable` ≈ 120 − 12.8 ≈ 107 GB; `pin_limit` = min(host_usable, 0.85 × 128) on Linux and min(host_usable, 0.50 × 128 = 64 GB) on Windows.

| Decision | Solver result |
|---|---|
| VRAM | 9.7 GB hot set (all mixers, routers, shared experts, LM head) + 2.25 active KV + 2.0 context and scratch + 0.54 headroom + **≈9.5 GB expert cache** |
| RAM | **≈104 GB** RAM cache of routed experts (`u`), homed on NVMe since the pool does not fit, exclusive of the VRAM cache; plus 2.25 GB of reserved KV for the queued sequence and ≈1 GB of staging. Linux pins all of it if the copy share wants; Windows pins ≈61 GB (`u^P`, the hottest experts), which with the pinned staging and reserved KV fills its 64 GB cap, and keeps the rest in `HostPageable` for in-place compute (C8, C9 joint rows). The reserved KV is in `HostPinned` on both OSes (RFC-0022 §3) |
| NVMe | the full pack; next-layer prediction prefetches into RAM |
| Miss split | closed form with d = 0: `m^P / R_miss ≈ (14.3 − 0.03) / (14.3 + 40) ≈ 0.26` of RAM-served misses copied over PCIe; on Windows the copy share is capped by the pinned share's reads (C8), which moves work to the CPU but stays below the NVMe term |

| Case | z^R (VRAM ∪ RAM) | NVMe GB per token | routed τ = max(GPU, CPU, NVMe) | t_tok | tok/s |
|---|---|---|---|---|---|
| Linux, or Windows with the pageable share, uniform prior | (104 + 9.5) / 394 = 0.288 | 8.8 | 1460 ms (NVMe) | ≈1.48 s | **0.68** |
| Windows with expert caches confined to `HostPinned` (if RFC-0020 keeps that rule) | (61 + 9.5) / 394 = 0.179 | 10.1 | 1683 ms (NVMe) | ≈1.70 s | **0.59** |
| learned curve giving h = 0.9 at this cache size (113.5 GB) | 0.90 | 1.23 | 205 ms (NVMe) | ≈220 ms | **4.5** |

The cost model's overlapped column gives 0.68 and 4.6 tok/s (§4b2) with a 115 GB inclusive RAM cache that predates RFC-0020's budget rules; it is kept only as a comparison. Counting the VRAM cache exclusively (C8) adds its 9.5 GB to fast memory, worth ≈49 ms per token under the uniform prior (0.024 × 12.3 GB / 6 GB/s), or the same hit rate with 9.5 GB less RAM. The first plan, before any routing statistics exist, predicts 0.68 tok/s on Linux and lists the hit curve and the OS memory use as assumptions; RFC-0023's learned curve then triggers a delta solve (§12.3), not a re-shard, because only cache budgets and the split change. The explanation offers the Q2_K sibling (244 GB, 7.4 GB routed per token): 4.4–7.6 tok/s at h = 0.9 per the cost model, with its quality cost.

#### 15.3 Scenario (c1): Llama-3.3-70B on a gaming PC (RTX 4090), a MacBook Pro M4 Pro 48 GB and a Strix Halo mini PC

W_eff: 4090 VRAM 800, MacBook unified 191 (≈32 GB usable), Strix unified 179 (≈96 GB usable), PC DRAM 70. 2.5 GbE hop ≈0.3 ms; Wi-Fi 5 ms median, 250 ms p99, and ≈ 12 ms mean [estimate: 0.9 · 5 + 0.09 · ~60 + 0.01 · 250 from the cost model's quantiles].

Each stage holds only its own layers' KV (33.5 MB per layer per sequence at 8 K; the active sequence's in VRAM, the queued one's reserved in host memory, as in §15.1). With §15.1's context, scratch and 0.54 GB headroom, the 4090 takes 38 layers: 20.2 GB of weights at ≈ 0.53 GB per layer plus 1.27 GB of active KV, rather than (a)'s 18.8 GB of weights.

| Candidate | Stage times | Hops | t_tok (t_fixed 1 ms) | tok/s |
|---|---|---|---|---|
| PC alone (VRAM + CPU) | 366 | 0 | 367 | 2.7 |
| Strix alone | 237 + 15 (KV) | 0 | 253 | 4.0 |
| PC → Strix | 26.8 + 132.5 | 2 × 0.3 | 161.9 | 6.2 |
| **PC → MacBook (wired)** | 26.8 + 124.1 | 2 × 0.3 | **153.5** | **6.5** |
| PC → MacBook → Strix (time-balanced) | 26.8 + 64.1 + 64.1 | 3 × 0.3 | 158.8 | 6.3 |
| PC → MacBook on Wi-Fi, costed at the mean (C14 checks the tail) | 26.8 + 124.1 | 2 × 12 | 176.9 | 5.7 |
| *Before 2026-10-20:* PC → MacBook on Wi-Fi, costed at p99 | 26.8 + 124.1 | 2 × 250 | 653 | 1.5 |

- **Chosen: PC → Strix (6.2 tok/s).** Byte time alone favours PC → MacBook: in latency mode bytes go to the fastest memory with room, and the MacBook's 191 GB/s beats the Strix's 179 GB/s while its 32 GB holds the remaining 42 layers (22.3 GB, with both sequences' KV for them, ≈ 2.8 GB). But the MacBook is departure-prone (§4: it has a battery), and its gain over the best plan without it is (161.9 − 153.5) / 161.9 ≈ 5.2%, below the 10% margin, so the plan is PC → Strix. With `solver.departure_margin = 0` the plan is PC → MacBook at 153.5 ms (6.5 tok/s). A third stage adds a hop and a `t_fixed` for no byte-time gain (cost model §3.7). Against the best single machine (Strix alone, 4.0 tok/s) the chosen plan is 1.55× faster.
- On Wi-Fi the MacBook costs ≈ 176.9 ms at the mean hop, slower than PC → Strix, so the plan is unchanged; the old p99 costing mispredicted that candidate by ≈ 3.7× (653 against 176.9 ms). Whether C14 admits it depends on the measured hop histogram. On battery, filter F4 removes it first.
- `t_fixed` matters: at the 5 ms fitted for one existing Metal engine (cost model §2.2), every extra stage costs 4 ms more, which is why SC-2 is a target for every backend.
- The PC's DRAM is unused in every multi-node candidate: at 70 GB/s it is the slowest place to execute (cost model §4c).

#### 15.4 Scenario (b1): two M3 Ultra 512 GB Mac Studios with RDMA over Thunderbolt 5

DeepSeek-V3 4-bit (377 GB) fits on one node: 20.5 GB per token at W_eff ≈ 426 GB/s gives 48.1 ms (20.8 tok/s). The pair forms an `Island(Tp)` placement node only if the device-path probe (RFC-0010 §3.5) measured α_eff below α_eff,max ≈ 198 µs (cost model §3.8). With TP = 2 each node reads 10.3 GB (24.1 ms) plus 122 collectives:

| Measured α_eff | TP t_tok | vs one node | Initial plan | Re-plan after the α_eff probe lands (min_gain 20%) |
|---|---|---|---|---|
| 40 µs | 29.0 ms (34.5 tok/s) | −40% | TP | adopted |
| 150 µs | 42.4 ms (23.6 tok/s) | −12% | TP | **not adopted**: 12% < 20%, the single-node plan stays |
| 300 µs | 60.7 ms | +26% | single node (gate fails) | — |

The middle row shows the hysteresis at work: the same measurement leads to TP when it is known before the first plan, but not to a re-shard afterwards, because the gain does not clear `replan.min_gain`. The explanation lists it as a pending improvement. Kimi-K2-Thinking (594 GB) does not fit one node, so the choice is PP = 2 at 76.7 ms (13 tok/s) or TP = 2 at 43–57 ms (17–23 tok/s) depending on α_eff (cost model §4b1).


### 16. Configuration knobs

Keys are cluster scope except `solver.context_target`, which is instance scope; the solver's `ConfigSection` declares each key's scopes so RFC-0054's generated registry and `InstancePolicy.domain` pick them up. RFC-0054 owns the schema. Other instance-level settings the solver reads (`max_sequences`, `context_length`, `speculative.max_window`, the objective and pins) are RFC-0054's keys carried in `InstancePolicy`; `solver.latency_kv_sequences` is merged into `max_sequences` and `solver.max_spec_window` into `speculative.max_window`.

| Key | Default | Meaning |
|---|---|---|
| `solver.enable_milp` | false; registered only in builds that contain the MILP (§9.5) | true runs the MILP after the DP |
| `solver.max_nodes` | 8 | candidate physical nodes after scoring |
| `solver.max_placement_nodes` | 16 | including TP groups and islands |
| `solver.max_blocks` | 64 | block coarsening limit (§3.2) |
| `solver.hit_curve_segments` | 8 | segments per block hit curve |
| `solver.mip_rel_gap` | 0.01 | HiGHS stop gap |
| `solver.mip_node_limit` | set by spike S1 so it binds at ≤ 50% of `safety_time_limit` (§9.2) | deterministic work limit; checked by RFC-0061 |
| `solver.simplex_iteration_limit` | set by spike S1, as above | deterministic work limit |
| `solver.safety_time_limit` | 2 s (in a repair solve, `Resolve` gets what remains of §12.4's 2 s budget) | wall-clock safety net (SC-31) |
| `solver.corridor` | 4 blocks | corridor re-solve width (§9.3) |
| `solver.min_slice` | 1 GiB | filter F5 |
| `solver.lookup_ram_max` | 2 GiB | `RowLookup` tables above this are homed on NVMe |
| `solver.context_target` | 8192 tokens (**instance** scope, declared so in the solver's `ConfigSection`) | the KV sizing floor behind `context_length = auto`: a plan that cannot reserve KV for `max_sequences` at this length is infeasible; also the context that prices KV reads until `WorkloadStats::context` has data (§7.1) |
| `solver.wall_clock_limit` | `on` (`off` in `testkit`) | when `off`, HiGHS stops on the deterministic count limits only, so tests and the simulator are reproducible regardless of machine speed |
| `solver.preview_cache_ttl` | 30 s | §17 preview cache |
| `solver.preview_full_concurrency` | 1 per node | §17: concurrent `Full` previews |
| `solver.preview_candidates` | 3 | default candidate count for `preview` (§14.1) |
| `solver.stall_threshold` | 300 ms [target] | C14: a hop this far above its mean counts as a stall (replaces `solver.wireless_quantile`, removed 2026-10-20) |
| `solver.max_stall_rate` | 0.01 [target] | C14: the largest share of tokens a plan may expect to stall on wireless and overlay hops |
| `solver.ttft_cap_factor` | 2.0 [target] | latency mode: TTFT at `L_ttft` ≤ this × the best candidate's (§7.1); applies from M2 |
| `solver.departure_margin` | 0.10 [target] | a departure-prone node hosts a stage only for at least this gain, or when capacity needs it (§4) |
| `replan.min_gain` | 0.20 | adoption threshold (RFC-0001 §7.2; tuned by RFC-0033) |
| `replan.payback_horizon` | 600 s | migration payback window |
| `replan.max_defer` | 5 min | longest wait for an idle window in latency mode (kept 2026-10-22, note below) |
| `replan.cooldown` | 10 min | minimum time between adopted deferred re-plans |
| `replan.delta_min_gain` | 0.02 | adoption threshold for delta solves |

*Note 2026-10-22 (#1678):* this RFC declares `replan.min_gain`, `replan.payback_horizon` and `replan.max_defer` (RFC-0054 §5.1 gives `replan.*` to it); RFC-0033 §12 now refers to them and declares only `replan.degraded_min_gain`. RFC-0033 had given `replan.max_defer` 10 min. The 5 min here holds because RFC-0054 §5.1 gives `replan.*` to this RFC, and RFC-0001 §7.2 names the key without a value: the default was settled at 5 min (RFC-0011's value) under the maintainer's standing instruction of 2026-10-22.

### 17. Failure handling

| Failure | Detection | Action |
|---|---|---|
| Model does not fit anywhere | DP returns ∞ and the degraded second pass (C10) is also infeasible | `Capacity` error to the requester with an `InfeasibilityReport`: per node and tier, the shortfall in bytes; the smallest sibling quantization that fits and its predicted tok/s; the extra RAM or VRAM that would make this variant fit. The dashboard shows it before load (RFC-0051 placement preview) |
| Fits only with dense bytes on NVMe | first pass infeasible, second feasible | plan with `degraded = DenseOnNvme`; the response metadata and dashboard say so with the predicted tok/s; latency-mode users are offered the sibling that avoids it first |
| Fits only with two stages on one node | DP finds no plan with one stage per node; a variant allowing one node two non-adjacent ranges (C2 relaxed for that node) is feasible | `degraded = MultiStageOnNode`; used only in recovery when a survivor must absorb a lost range that is not adjacent to its own (RFC-0033) |
| Pins infeasible | the MILP and DP are infeasible with pins, feasible without | reject with the list of conflicting pins and the capacity each needs; never silently drop a pin (ADR-0008 item 8) |
| HiGHS returns an error, numerical trouble or an unknown status (builds with the MILP, §9.5) | `Highs_getModelStatus` | DP plan; `SolveStats::path = Dp`, the status logged; repeated failures on the same input disable the MILP for that instance until the next build |
| HiGHS panics or aborts the process (builds with the MILP) | on the coordinator: the daemon restarts and the supervisor counts it; for previews: the probe-mode child exits abnormally | **The coordinator** solves that input with the DP only after one crash (a crash-loop guard keyed by the input digest), and reports the bug with the input dumped. **Ingress previews** with `effort: Full` run the MILP in a short-lived probe-mode child process, so a crash never takes down the daemon, its API or its membership. Each crash records the input digest in the state directory; later previews of that input on that node use `DpOnly`, across restarts. The digest list is also announced to the coordinator, so every node's DP-only fallback list converges |
| Previews requested on every request (automatic placement, the API's estimate endpoint) | preview rate per node | A preview never triggers an expensive solve per request. Results are cached by (graph version, census digest, policy digest, overrides digest, effort) and reused while the graph version is unchanged, for at most `solver.preview_cache_ttl` (default 30 s). At most `solver.preview_full_concurrency` (default 1) `Full` previews run per node; further requests for the same key wait for it, and requests for other keys get the `DpOnly` answer at once with `effort` reported. Automatic placement on the request path uses `DpOnly` and never waits for a `Full` preview; the committed plan is solved by the coordinator anyway |
| Safety time limit fires | `SolveStats::time_limited` | incumbent used; counted; if it fires twice in a row, the work limits for that instance size are halved |
| Evaluator finds a rounding violation | §11.3 | repair, else DP plan |
| Input profile has `Prior` values for first-order terms | `assumptions` non-empty | plan proceeds; the dashboard shows the assumptions and their sensitivity; a later profile triggers a re-plan through drift |
| Graph changes during a solve | `GraphVersion` moved | the coordinator discards the result if the change affects candidate nodes and re-solves; otherwise commits with the solved-against version recorded |


## Performance model and budgets

The solver's own cost model is cost model §3 (RFC-0001 §14); its accuracy is the subject of SC-3, SC-13 and SC-30. Its run-time budgets are **targets**:

| Budget | Target | Derivation |
|---|---|---|
| Slow re-placement, 8 nodes, MoE with expert variables | ≤ 2 s end to end | SC-31 (chosen budget) |
| — of which preprocessing, filter, score | ≤ 50 ms | O(groups + nodes²) arithmetic over a census of ≤ 10⁵ tensors |
| — DP | ≤ 100 ms (8 Local nodes), ≤ 400 ms with islands | §10's factored recurrence: ≈ 7.5 × 10⁷ relaxations Local only and ≈ 3.4 × 10⁸ with islands, at ≈ 1 ns each [estimate]; RFC-0061 benchmarks it |
| — MILP, only where §9.5's gate built it | ≤ 1.2 s, within the deterministic work limits | ≈ 2,350 binaries (up to ≈ 3,380 with `χ`), ≈ 33,000 continuous columns and ≈ 66,000 rows after the builder's reductions (§5). A smaller layer-window ILP solved in 10–12 ms at 4–32 devices [reported by a surveyed system], but this model is far larger, so spike S1 decides whether the budget holds and sets the work limits (open question 1) |
| — speculative outer loop | ≤ 1 extra solve (DP, or the MILP warm-started where built) | §7.3; latency mode |
| — rounding, evaluator, explanation | ≤ 100 ms | linear passes; one LP re-solve for duals |
| Repair solve | all candidates ≤ 2 s (RFC-0033 §6.4); `Replace` and `Absorb` evaluations in milliseconds with 32-block stage costs; `Resolve` uses the rest | §12.4 |
| Throughput-mode solve | ≤ 10 s, in the background | up to 6 values of `b_μ` × 2 solves each (the D repeat), each warm-started from the previous; plus the speculative loop when it is on; ADR-0103 exempts throughput plans from SC-31's 2 s |
| Preview, fast mode | ≤ 100 ms per DP table for up to 8 nodes: ≤ 200 ms for `Best`, `FewerNodes` and `NoEvictions`, plus ≤ 100 ms per `SmallerQuant` sibling [target] | island-free DP at ≈ 75 ms per table [estimate, §10], one table shared by candidates with the same reservation and census (§14.1) |
| Delta solve | ≤ 50 ms | LP with ≈10⁴ columns, no binaries |
| Fast routing | ≤ 1 ms, typically ≤ 20 µs | O(instances) arithmetic (SC-31) |
| Plan size | ≈ 0.7 MiB for a Kimi K3-class census | compact encoding (§11.2): interned tables of node, tier, device and link IDs, then per group, in census order, an 8-byte record (owner index u8, home index u16, compute index u16, stream site index u16, replica count u8); 93 × 896 experts ≈ 83 K groups × 8 B ≈ 0.66 MiB. Without interning a group is ≈ 100 B (32-byte node IDs), ≈ 8 MiB per plan. `PlanCommitted` carries the plan by digest whenever its encoding exceeds 64 KiB (amendment B2) |
| Prediction accuracy | predicted t_tok within ±15% of measured on the reference rigs | SC-30; MILP gap ≤ 1% and rounding error ≤ one tensor group per block keep the solver's own contribution small |

**Solution quality.** The DP is exact for contiguous plans under its greedy stage cost. How far it lands from a MILP with a 1% gap is measured before the MILP is built, by S-09's study on the reference inputs (§9.5): a gain above 5% on any of them builds the MILP; otherwise the DP is the production method. Where the MILP is built, it cannot be more than 1% worse than the best plan under its model, and RFC-0061 keeps measuring the DP's distance from it on every release.

## Alternatives considered

| Alternative | Why not |
|---|---|
| **Slot-indexed formulation** (stage slots × nodes × layers, with order from slot index) | needs products of slot-to-node and layer-to-slot binaries, ≈ #B · #P · #S = 64 × 16 × 8 ≈ 8,000 more binaries (# for set size); the start-indicator and successor formulation (§8 items 1–2) gets ring order from layer order with #B · #P binaries |
| **Separate solves**: layers first, then tiers, then experts, then KV | misses the coupled trade-offs ADR-0008 exists for: KV competes with experts for VRAM; a hop pays only if it moves bytes to faster memory |
| **Per-expert binaries** | 61 × 256 = 15,616 binaries for DeepSeek-V3, 83,328 for a Kimi K3-class model, for decisions the runtime cache makes better from live routing. Budgets per layer with a learned hit curve capture what the solver needs (ADR-0102) |
| **Popularity buckets as separate integer groups** (ADR-0008's wording) | equivalent to the hit-curve segments with continuous fill, which needs no binaries; the segments are the buckets |
| **Fixed ring order from a TSP heuristic, then MILP per order** | separates order from placement; the DP already explores every order and subset, and the MILP's successor variables keep order and placement coupled |
| **Wall-clock time limits as the main stopping rule** | breaks reproducibility (RFC-0001 §4.2 rule 5): two coordinators of different speed would commit different plans for the same input |
| **p99 on every wireless hop** (the choice until 2026-10-20) | it charges every token the tail: predictions on Wi-Fi are off by about 20×, and Wi-Fi nodes are left out even when the model fits only across them. Replaced by the mean plus C14 and a jitter badge |
| **Excluding wireless links from latency-mode stages** | simple, but it refuses models that fit only across wireless nodes |
| **The MILP as M1's production method** (the choice until 2026-10-20) | its gain over a good greedy is unmeasured, and it brings a solver dependency, determinism tuning and crash-isolated previews into M1. Built only behind §9.5's gate |
| **Decode-only latency objective** (the choice until 2026-10-20) | nothing trades decode against the first token: on S2 it picks a 4K TTFT of ≈ 22 s against 3.6–5.9 s on the PC alone. Replaced by the TTFT cap |
| **A weighted sum of decode time and TTFT** | needs a weight users cannot reason about; the cap keeps D2's decode priority and rules out only plans with a much slower first token |
| **Pricing departure cost now** (rate × stall time in the objective) | no measured departure rates exist yet; the 10% margin is used until alpha data supplies them |
| **Stochastic or robust optimisation over hit-rate uncertainty** | heavier solves for a term that online learning corrects within minutes; the delta solve absorbs it without a re-shard |
| **Learned placement policy** (a model trained to output plans) | no training data at the start, no feasibility guarantee, and nothing to explain to the user |
| **Commercial MILP solvers** | licence incompatible with an embedded MIT/Apache binary (ADR-0008) |

## Testing and validation

| Claim | Test | Kind |
|---|---|---|
| Every plan fits every tier on every node | for random graphs (1–8 nodes, 1–4 devices each, random capacities and rates) and every conformance census, `evaluate(plan)` shows peak bytes ≤ usable on every (node, tier); predicted peaks within 5% of measured on the rigs (SC-11) | property; benchmark |
| The DP finds a plan whenever one exists | on random small instances (≤ 5 nodes, ≤ 12 blocks, with pins), the DP is feasible exactly when an exact LP feasibility check over all contiguous splits and node orders is (HiGHS LP, test-only), not merely when the greedy is; and its cost equals the brute-force optimum under its own stage cost. Instances include full VRAM, where mixers must spill to the CPU and their KV must go to host tiers | property |
| Repair keeps survivors' shards | random stage losses on simulated fabrics: every `keep_shards` candidate leaves each survivor's `x`, `w` and `u` unchanged, `t_resume` matches RFC-0033 §6.2's formula, and `Resolve` appears only when no keep-shards candidate is feasible | property, simulation |
| The DP meets its time targets | factored recurrence at 8 Local nodes × 64 blocks ≤ 100 ms and with islands ≤ 400 ms on the slowest Tier 1 coordinator; 32-block repair path ≤ 30 ms | benchmark (RFC-0061) |
| Preview | for random inputs, every `PreviewCandidate` evaluates to the predicted cost it reports; a summary-based preview's prediction is within one tensor group per block of the full-manifest preview | property |
| The MILP gate is decided on evidence (§9.5) | S-09's harness reports, per reference input, both plans' evaluator `t_tok` and the gain; re-running it on the same inputs gives the same gains; the recorded decision matches the rule `gain > 0.05` on some input | spike (M0), recorded |
| Where the MILP is built, it is never worse than the DP | on every test input, `evaluate(milp) ≤ evaluate(dp) · (1 + mip_rel_gap)`; the committed plan is the better of the two | property |
| Without the MILP, the closed forms match the LP | on random single-stage instances, the greedy's shadow prices, delta-solve budgets and C13 pinned bytes equal a test-only HiGHS LP's duals and solutions within 10⁻⁶ relative | property (test-only HiGHS) |
| TTFT cap (§7.1) | on random multi-node fabrics, the chosen plan's TTFT at `L_ttft` is ≤ 2 × the minimum over the DP's candidates, and no candidate within the cap has lower predicted `t_tok`; on scenario S2's inputs the cap rules out the ≈ 22 s 4K-TTFT plan | property; golden |
| Wireless costing (C14) | links in `testkit` with configured hop histograms: `α^plan` equals the histogram mean; the plan's stall rate equals `Σ q_e` over its wireless edges; no committed plan exceeds `max_stall_rate`; plans with a wireless edge carry `jitter`, wired-only plans do not | property |
| Departure margin (§4) | on random fabrics with laptop and battery nodes, a departure-prone node hosts a stage only when the plan's gain over the best plan without such nodes is ≥ `departure_margin`, or no plan without them is feasible; scenario (c1) chooses PC → Strix, and PC → MacBook with the margin set to 0 | property; golden |
| One-node plans win when the model fits one node and hops add only cost | SC-14: when the model fits on one node, the chosen plan is never slower (predicted) than the best single-node plan | property over simulated fabrics |
| Determinism | the same `SolveInput` solved 100 times gives byte-identical plans on one architecture, including under deliberate CPU contention (RFC-0060 §2.1); x86_64 against arm64 is tested and, if it differs, documented as a per-architecture guarantee (§9.2); in the simulator the wall-clock limit is disabled | property, CI on two architectures |
| Linearisations are exact | for each technique in §8, a unit test compares the MILP's value at fixed integer choices with the evaluator's direct computation (e.g. τ equals max of the three sites; f equals ring adjacency; z equals H at breakpoints) | unit |
| Worked examples | §15's four scenarios as golden tests with the stated inputs and defaults (headroom, `max_sequences`, RFC-0020 budgets and the OS-memory assumptions, the departure margin): the production method's choices and predicted t_tok must match the tables within 1% | unit (golden) |
| Device selection | the 10-device mixed testbed from the surveyed home-cluster study, rebuilt in `testkit` with the network term restored (`multi-device-inference.md` §15.3 item 13): check whether the solver also uses a small subset of devices, and compare with fill-fastest-first | simulation |
| Re-plan hysteresis | simulated membership churn (join, leave, drift) over 24 simulated hours: no deferred re-plan with gain < `min_gain` is adopted, no two adopted re-plans within `cooldown`, and repair solves always return their candidates within §12.4's 2 s budget | simulation (RFC-0060) |
| Solve-time budgets | RFC-0061 solver suite: 1, 2, 4 and 8 nodes × dense 70B, DeepSeek-V3-class, Kimi K3-class censuses; p99 solve time ≤ 2 s on the slowest Tier 1 coordinator; time limit never fires | benchmark gate |
| Prediction accuracy | SC-30 on the reference rigs: predicted t_tok within ±15% of measured, per stage and per term | benchmark |
| Routing | 10⁵ random requests over 16 instances: ≤ 1 ms p99; the chosen instance minimises the score; exact requests never go to an approximate plan | property, benchmark |
| Explanations are consistent | for every group, `saving_per_gib` ≥ the shadow price of its home tier, or the group is pinned or capacity-bound; for every unused device, forcing it in never improves predicted t_tok by more than the MILP gap | property |
| Warm start is used | HiGHS log shows the supplied solution accepted as the first incumbent, for every pinned HiGHS version | unit (HiGHS upgrade gate) |

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M0 | `PlacementPlan` construction for one device by a trivial rule (all dense in the fastest tier that fits, else fail) so the M0 pipeline runs end to end; `PredictedCost` and the evaluator for single-tier plans; S-09's standalone harness (PLC-E5-T02): the formulation, HiGHS's scale numbers and the DP-against-MILP study that decides §9.5's gate |
| M1 | solver v1 on one node (RFC-0001 §16 M1): filter, preprocessing, the DP with its greedy stage cost as the production method (the MILP over the node's devices × tiers only if §9.5's gate passed), expert budgets from hit curves, KV in the objective, n* and `StreamSite` (streaming at decode where measured faster), the speculative outer loop for MTP and draft models, the delta solve, explanations with shadow prices, the infeasibility report, the placement preview's numbers with decode and TTFT side by side |
| M2 | joint solver across nodes: placement nodes, ring structure with the token-return edge, hop terms at the mean with C14's stall bound and jitter figures for wireless, the TTFT cap (§7.1), the departure margin (§4), duplicated head and tail bytes, per-stage history reserves, migration term, hysteresis and payback, `Resolve`-only node-loss handling (re-prefill), fast per-request routing across instances with `context_length` checks, intra-node TP placement nodes, cluster preview |
| M3 | `solve_repair` with keep-shards `RestartInPlace`, `Replace` and `Absorb` (with `MultiStageOnNode`) and `t_resume`, together with RFC-0033; islands for cross-node TP and EP behind the measured gate; throughput objective with micro-batch search, per-sequence floor, the per-D DP and replica hints; §9.5's study re-run with islands and the throughput objective; corridor re-solve (where the MILP is built); the approximate-knob line once `KnobCost` reports exist |
| M4 | solver regression suite gating every release (RFC-0061); work limits tuned on the reference inputs |

## Open questions

1. **MILP solve time at scale (spike S1, S-09; moved to M0 as a standalone harness).** S-09's harness now also runs §9.5's DP-against-MILP study, which decides whether the MILP is built at all; the measurements below matter only if it is. The size estimate in §5 is about 33,000 continuous and 2,350 binary columns and 66,000 rows. S1 measures, on the slowest Tier 1 coordinator: simplex iterations per second and MIP nodes per second at this size; whether `simplex_iteration_limit` is per LP or cumulative in HiGHS's MIP; whether `Highs_setSolution` reliably seeds the MIP incumbent; the gap reached within the budget for 8 nodes and a Kimi K3-class census; and numerical warnings with the §9.2 scaling. It then sets the count limits to bind at ≤ 50% of `safety_time_limit`. If the gap cannot close, the fallbacks are the corridor re-solve (§9.3), fewer blocks, dropping `χ`, or moving expert budgets to a second-stage LP. ADR-0101 revisits on the outcome.
2. **Serial versus overlapped NVMe misses.** C8 uses the overlapped form (SC-5). In exact mode a demand read stalls its layer; if measured `t_tok` sits closer to the serial form, the NVMe row needs an additional serial share fitted from telemetry. Settle with the (b2) rig in M1.
3. **DMA interference `d`.** Measured per node by RFC-0010's H3 at one load level. Is it linear in DMA rate? RFC-0023's online fit will tell; if not, C8's CPU row needs a piecewise term.
4. **KV sizing in latency mode.** `max_sequences = 2` (RFC-0054) at the context target is a guess at how much KV headroom a single user needs. Agent workloads with parked sessions may need more (RFC-0022).
5. **Stage-boundary rules for hybrid families. Answered 2026-10-20 (OQ-PL-12):** add no stickiness term. The migration term (§7.4) charges the real cost of moving recurrent-state layers: their state bytes ÷ β of the path that carries them, plus replay from the last state checkpoint (RFC-0022 §3, §6). PLC-E6-T12 implements it.
6. **Large plans on the control plane.** A Kimi K3-class plan is ≈1 MiB. RFC-0030 to confirm whether `PlanCommitted` carries it inline or by digest.
7. **Prefill-optimised placements.** The cost model suggests routing long prefills to the most compute-rich node and shipping KV when the link is fast (§5.4). v1 does not plan separate prefill sites; revisit after M3 with measured KV transfer rates.
8. **Pin budget between decode and prefill. Answered 2026-10-20 (OQ-PL-14): not "decode first".** Latency mode minimises decode subject to the TTFT cap (§7.1), and C13 gives `u^P` to streamable dense groups when the cap needs it. M1's single-node plans report TTFT and the cap; PLC-E6-T23 measures how often the pin row binds and how often the cap would, as telemetry and a regression row rather than an M1 gate; the cap binds from M2.

## Proposed amendments to RFC-0001

- **B1. Compute site for `Nvme`-homed groups.** §5.4 says `ComputeSite::decode` is "a device in the home tier's affinity set", but `Nvme` has no compute affinity, and experts homed on NVMe are computed after landing in a RAM or VRAM cache (ADR-0006 item 6). Proposed: "for groups homed on `Nvme`, `decode` is the device that computes the group from its staging tier (the `HostPinned` cache's CPU, or the GPU when copied)".
- **B2. `PlanCommitted` size.** §5.10 carries `Arc<PlacementPlan>` inline. A Kimi K3-class plan is ≈ 0.7 MiB even in the compact encoding (§11.2), and ≈ 8 MiB without it. Proposed: `PlanCommitted { plan_digest, summary }` with the body on the bulk plane whenever the encoding exceeds 64 KiB, as `RegisterModel` already does for manifests.
- **B3. Definitions for named-but-undefined types.** `PredictedCost`, `BatchingLimits`, `Assumption`, `DeltaItem` and `TransitionStrategy` are referenced in §5.4 and defined in this RFC (§2, §12.3). RFC-0001 §5.4 should point here.
- **B4. Throughput-mode concurrency.** §6.2 describes throughput mode as filling each step "up to the batching limits". This RFC adds `BatchingLimits::micro_batch` and plans for `M = D · micro_batch` sequences in flight; RFC-0012 schedules them. No semantic change, but §6.2 could reference it.
- **B5. Workload statistics on the control plane.** `WorkloadStats::hit_curves`, `union` and `tier_hit` reach the coordinator in RFC-0023's `ExpertWorkloadReport` through `ClusterCommand::ReportWorkload` and its matching event, both proposed as RFC-0001 amendments (RFC-0023 Proposed amendments, item 5). The coordinator assembles `WorkloadStats` per model from them, together with acceptance statistics from RFC-0012.
- **B6. Routing decisions reach the head.** `SequenceOpen` (§5.8) and RFC-0032's `SequenceMeta` gain `prefill_chunk: u32` and `initial_window: u16`, so the head applies the router's choice (§13).
- **B7. Instance context length.** `InstanceRoutable` (or the instance's telemetry digest) carries the plan's `context_length` (§11.2), which ingress checks against `prompt + max_tokens` (§13).
- **B8. Separate host tiers in the solver's view of RAM.** §5.2's `TierInfo` should say that `HostPinned.pin_limit` and `host_usable` per NUMA domain are joint budgets across `HostPinned` and `HostPageable` (C9's joint rows; RFC-0020 §5).
