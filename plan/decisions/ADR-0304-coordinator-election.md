# ADR-0304: Coordinator election uses beat-renewed leases, rank-staggered claims that abort on a live lease, and node-tagged terms

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0009, ADR-0301, RFC-0001 §5.4, RFC-0032 §7

## Context

ADR-0009 fixes the shape of election:
- the coordinator holds an epoch-fenced lease;
- candidates are ranked by an operator flag, then capability class, then lowest node ID;
- no majority quorum, so a single survivor elects itself;
- no preemption, and no election on connection changes;
- back-off on repeated elections;
- split brain during a network split is accepted and reconciled on heal.

Three things are left open:

1. **What "lease" means without quorum.** With no grantors, nobody can grant one.
2. **Partial partitions.** If Y cannot reach coordinator C but Z can, a naive timeout makes Y a second coordinator inside one connected cluster.
3. **Collisions.** `PlanEpoch { term, seq }` (RFC-0001 §5.4) and coordinator-allocated `InstanceId`s are derived from the term. If both sides of a split pick `max_term + 1`, they mint identical epochs and instance IDs for different plans.

The surveyed bully election re-ran on every connection change, with a 3 s campaign and back-off for only some triggers, and in-flight requests failed on takeover (`multi-device-inference.md` §2.2).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Randomised timeouts and votes, as quorum protocols use, without the quorum | Familiar | Any node may win; contested claims; still collides terms across a split |
| B. Bully election triggered by connectivity | Simple | Flapping; the observed failure mode |
| **C. Followers renew a local lease deadline from each coordinator beat (suspend-aware clock). On expiry, eligible nodes claim after `rank position × 300 ms`. A claim aborts if any reachable member answers that it still hears a live coordinator. Terms are `(counter << 24) \| tag24(node)`. The winner pulls missing entries from peers before appending `CoordinatorElected`. Planned handover skips the lease** | Deterministic winner in the common case; partial partitions don't split; terms and IDs never collide across a split (except a negligible 24-bit tag collision); fast planned handover | More message types than a timeout; tag collisions possible in theory |

## Decision

**Option C** (RFC-0032 §7).
1. Lease defaults:
   - 4 s on LAN, which is 4 missed 1 s beats, and 12 s on WAN paths;
   - claim stagger 300 ms per rank position;
   - claim window 500 ms;
   - back-off from 1 s doubling to 30 s after 3 elections in 60 s.
2. `rank_key = (operator preference, capability class, node ID ascending)`. Capability classes run from 3 down to 0:
   - 3: mains and wired;
   - 2: mains on Wi-Fi only;
   - 1: battery;
   - 0: ineligible, meaning a mobile OS by default, `may_coordinate = false`, an ingress-only certificate, or only Wan paths.
3. Voters reply `LeaseLive`, `Better` or `Stale` to reject a claim, and `Ack` to accept one. No quorum is counted. With no rejections, the claimant wins.
4. `term = (counter << 24) | tag24`, with `tag24` taken from BLAKE3 of the node ID. Derived from it:
   - `InstanceId = (counter << 40) | (tag24 << 16) | n16`;
   - `PlanEpoch.seq` counts within the term.
5. Fencing: every append, beat and reply carries the term, and lower terms are refused. Data-plane frames are fenced by plan epoch, not term, so runners keep serving across coordinator changes.
6. Heal: the lower term steps down, and diverged followers take a snapshot. Facts are re-reported, and runners from the losing side are recorded as orphans, then adopted or drained by the next plan.

**Revised 2026-10-19:**
1. **Eligibility is bound to the certificate**. Only a node whose admission certificate carries `COORDINATOR` may claim. Voters and followers check it. Ingress-only and invite-joined nodes default to not eligible.
2. **Counter rule.** A claim or `CoordinatorElected` is accepted only at `counter = counter(last elected term) + 1`. No node can jump the counter or overflow it.
3. **Pre-vote**. Voters reply without changing their term, and adopt a term only when they accept the elected node's `CoordinatorElected`. An aborted claim therefore never deposes a live coordinator. `Better` compares claims at the same counter by rank.
4. **Fencing.** Beats and appends are accepted only from the node elected at that term. Every entry passes `validate()` (RFC-0032 §6.3). A follower that sees an entry fail stops following that coordinator.

**Revised 2026-10-19, second revision:**
1. **Signed elections.** Claims, pre-vote `Ack`s and `CoordinatorElected` (a `SignedElection`) are signed by the claimant's or voter's node key. The claimant must hold a valid `COORDINATOR` certificate. Followers return a signed `Endorse` when they accept an election.
2. **Proofs, not term numbers.** Fencing replies and beats carry the coordinator's `ElectionProof`, never a bare term. A bare or unverifiable term is ignored, so no member can depose a coordinator by fabricating one.
3. **Heal order** `(supported, counter, tag)`, then lower node ID. A proof is supported when another member acked or endorsed it, so an isolated self-electing node cannot win a heal by inflating its counter. Every node applies the order independently, and the counter rule governs claims within one log only, so a heal at equal counters works.
4. **Heal procedure.** The losing side switches on a verified winning proof, even within a live lease. It finds the fork, adopts the winner's validated snapshot, endorses the winner, and re-submits its facts, journal and ledger entries (RFC-0032 §7.5).
5. **Handover** uses the next counter after the last elected term, and the outgoing coordinator's signed handover counts as an `Ack`.

**Revised 2026-10-20:**
1. **Rank tie-break** (OQ-NET-25). Decision 2's `rank_key` becomes `(operator preference, capability class, hosts_no_stage, node ID ascending)`, where `hosts_no_stage` is true for a node that hosts no stage of any instance in the committed plan state at the moment the lease expires. A coordinator that also hosts a stage makes that stage's recovery wait for an election, and the tie-break avoids that whenever a capability class offers a choice. Only nodes holding a `COORDINATOR` certificate are ranked (ADR-0009, revised 2026-10-20).
2. **The model check comes first** (OQ-NET-08, correctness part). Heal correctness is an M2 risk, so the explicit-state model of lease, pre-vote, signed proofs and heal order is checked at the start of M2, before the election code is written, with every RT-DIST scenario as a named case (DIST-E10-T08). An optional quorum mode for clusters of three or more always-on nodes stays an M4 question, taken up only if the checker or the simulator finds a heal that cannot be repaired.

## Consequences

- Positive:
  - Typical unplanned failover takes about 4.5–5 s to a new coordinator; a planned handover about 0.5 s [targets].
  - No request fails because of an election.
  - The cluster's preferred node becomes coordinator.
- Negative:
  - Entries the old coordinator appended but never replicated are lost. ADR-0009's rule that state must be rebuildable from facts makes that recoverable.
  - Term values are not consecutive, which affects only humans reading logs.
- Work created:
  - RFC-0060: a model-checked election and replication model;
  - RFC-0051: displaying the coordinator and its rank reason.

## What would make us revisit

- The model checker or field reports find two coordinators accepted by one node at the same term.
- Failover above 6 s blocks re-placement after a stage loss for too long in RFC-0033's chaos suite. Consider shortening the lease on wired-only clusters.
- An optional quorum mode is added for clusters of 3 or more always-on nodes (ADR-0009 revisit).
