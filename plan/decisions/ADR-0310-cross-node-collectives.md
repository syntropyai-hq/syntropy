# ADR-0310: Cross-node collectives are one-shot all-reduces reduced in fixed member order in F32, driven by primary-authored step descriptors

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0006, ADR-0010, ADR-0011, ADR-0019, ADR-0309, RFC-0032 §5, RFC-0031 §6.8

## Context

ADR-0006 allows cross-node tensor and expert parallelism only inside fast islands that pass a measured gate. Such parallelism pays only when the effective per-collective latency is below `α_eff,max ≈ 200–300 µs` for 60–80-layer models (cost model §3.8). On a surveyed RDMA cluster the back-solved cost is about 130–150 µs per collective, while the wire costs 10–21 µs. Most of the cost is synchronisation and collective implementation. SC-17 targets `α_eff ≤ α_wire + 2 · t_switch`.

Further constraints:
- Islands are small: 2–4 nodes, typically Macs on Thunderbolt 5, a full mesh of point-to-point cables. They may be heterogeneous.
- Exact mode requires F32 at boundaries where the reference holds F32 (ADR-0010, ADR-0019).
- Members must never take different branches. Surveyed runtimes all-gather task IDs and seeds every step to prevent deadlock (`multi-device-inference.md` §4.7).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Ring all-reduce always | Bandwidth-optimal; standard | 2(N−1)·α per collective: at N = 4, 6α instead of 1α, which is latency-bound at decode |
| B. Star (gather to one member, reduce, broadcast) | Simple | 2α plus serialisation at the root |
| **C. One-shot: every member sends its partial to every other in parallel, and each sums all N partials in member-index order in F32; ring above 4 members or 1 MiB** | 1α + S/β on a full mesh; bitwise-identical results on every member; no root | (N−1) sends per member; needs a full mesh for the parallel form |
| D. A vendor collective library | Mature | Homogeneous, static, vendor-specific (ADR-0011) |

And for keeping members consistent:

| Option | For | Against |
|---|---|---|
| E. All-gather decisions each step | Symmetric | An extra collective per step; members still compute routers that may differ on heterogeneous devices |
| **F. The stage primary forwards the inbound frame as a step descriptor and broadcasts every structure-changing decision (routed experts). A verified local router is allowed only on homogeneous islands that pass a bitwise check** | At most +1α per MoE layer, and zero on verified homogeneous pairs; a single author removes divergence | The primary does slightly more work |

## Decision

**C with F** (RFC-0032 §5).
1. All-reduce algorithm:
   - `OneShot` for N ≤ 4 and S ≤ 1 MiB, which covers every decode and verify-window collective;
   - `Ring` (reduce-scatter then all-gather) otherwise.
2. Partials travel in F32 in exact mode. The reduction kernel sums in member-index order with no reassociation or FMA contraction, so all members hold bitwise-identical residual streams. BF16 partials are allowed only under `WIRE_COMPRESS`.
3. The synchronisation path has no host synchronisation:
   - the GPU writes its partial into a host-mapped registered slot and rings a doorbell;
   - the net poller posts the sends;
   - on the last receive, the poller sets a host flag;
   - the GPU stream waits on the flag device-side, then reduces.

   On unified memory there are no copies at all.
4. Every collective is driven by the primary's step descriptor. Routed experts are broadcast (`ROUTER_BCAST`), unless the island is homogeneous and the routers matched bitwise on the canary at `Warming`, with periodic re-verification.
5. Expert parallelism uses dispatch and combine messages to owners. Combines are summed in member order.

**Revised 2026-10-19:**
- **`α_eff` definition**. `α_eff` is the per-collective cost RFC-0010's L2 probe measures at the `h_model × 4 B` partial, with S/β included. It is used alone in the SC-16 gate and in cost tables.
- **Corrected example.** For DeepSeek-V3 TP = 2 at α_eff = 40 µs: 122 × 40 µs = 4.9 ms per token, 29.0 ms in total, ≈ 34.5 tok/s, matching the cost model.
- **Router broadcast layout.** `ROUTER_BCAST` puts the routing weights before the expert IDs, so they are naturally aligned.

**Revised 2026-10-20:**
1. **The Mac-pair rule is fixed before the measurement.** A raw-path RDMA probe runs on a TB5 Mac pair late in M1 (RFC-0031 open question 1). If its `α_eff` exceeds 65 µs, SC-17 becomes report-only at M3, and the Mac work of cross-node tensor parallelism (DIST-E13's TB5 tasks) moves to M4 or after 1.0. Linux RoCE and InfiniBand islands are unaffected. A rule set in advance turns the measurement straight into a schedule change; it replaces the revisit clause's "drops in priority".
2. **Router determinism is a design rule** (OQ-NET-22), not a spike. Router kernels use fixed-order reductions, with no atomics and no split-K, and RFC-0040 flags them as deterministic. The router GEMV is tiny, so the rule costs nothing measurable. The 10,000-step hash re-check of decision 4 stays as a guard, and a conformance test runs each backend's router kernel twice on fixed inputs and requires bitwise-equal outputs (RFC-0032 open question 4).

## Consequences

- Positive:
  - The per-collective cost approaches wire latency plus two hand-offs. That is the regime where TP pays on Thunderbolt 5 (an estimated 30–33 tok/s for DeepSeek-V3 TP = 2 against 20.7 on one node, RFC-0032 §Performance).
  - There is no deadlock from divergent branches, and no residual drift between members.
- Negative:
  - Exact-mode F32 partials cost about 2 % over BF16 in the TP = 2 example.
  - One-shot needs a full mesh to realise its 1α form.
  - Homogeneous local routing depends on kernel determinism (RFC-0032 open question 4).
- Work created:
  - RFC-0040: a deterministic-kernel flag and the fixed-order reduction kernel;
  - RFC-0043: legal split granularities per family;
  - RFC-0060: bitwise member-hash checks.

## What would make us revisit

- The M3 measurement shows parallel Thunderbolt RDMA sends serialising in the RDMA engine. Then `Ring` wins at lower N.
- The measured `α_eff` on the two-Mac rig stays above 65 µs after this design. The rule in the 2026-10-20 revision above applies (RFC-0001 §17 question 3).
- Families appear whose reference holds BF16 residuals. Their partials may then travel in BF16 in exact mode, per ADR-0010 item 5.
