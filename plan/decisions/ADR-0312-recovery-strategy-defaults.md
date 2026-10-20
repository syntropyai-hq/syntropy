# ADR-0312: Recovery keeps surviving shards in place, replays first, and picks strategies by predicted time to resume

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0008, ADR-0009, ADR-0010, ADR-0016, ADR-0311, RFC-0001 §7.2–§7.5, RFC-0032 §4, RFC-0033 §5–§7

## Context

RFC-0001 §7.5 lists four ways to carry a sequence across a plan change or failure: finish, KV migration, activation-history replay and re-prefill. It says the choice is made per sequence from the cost model. It also routes any change of a node's neighbours through drain-and-replace (§7.3). In a recovery, that would throw away the survivors' KV and history, which are the state recovery depends on.

The surveyed systems re-placed only by hand, deleted instances on node loss, or restarted whole groups (`multi-device-inference.md` §10). The research recommends automatic re-placement after every loss and never killing slow-but-alive stages (§15.8).

A worked example (RFC-0033 §Performance) shows the spread of time to resume after losing a 12 GB stage:
- about 21–25 s when the bytes are already on the absorbing node;
- about 66 s from a cluster holder over 2.5 GbE;
- about 220 s from the Hub, because the lost node was the only holder of its shard (ADR-0016).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Always full re-solve and drain-and-replace | Best steady state; one path | Discards survivors' KV; every recovery is a full re-prefill and may move hundreds of GB |
| B. Always restore the same plan shape (replace the lost node with a spare) | Predictable | Often no spare exists in a home cluster |
| **C. Repair plans keep surviving shards and relink them in place. The solver evaluates `RestartInPlace`, `Replace` and `Absorb` (degraded), and `Resolve` only as a fallback. The strategy with the least predicted time to resume wins, preferring better steady state within a 20 % slack. Sequences replay via the cascading handshake. Planned transitions choose finish, then KV migration, replay or re-prefill per sequence by predicted pause. Degraded plans are left at a halved gain threshold** | Survivors keep state; fastest resume; degraded but serving within seconds when bytes are local; one replay mechanism covers re-prefill as its limiting case | New in-place delta types (`Relink`, `ExtendShard`) need an RFC-0001 amendment; the time-to-resume model must be right |

## Decision

**Option C** (RFC-0033 §5–§7; RFC-0032 §4).
1. Repair constraints default to `keep_shards = true` and `MinTimeToResume`. Survivors receive `DeltaItem::Relink`, and absorbers receive `DeltaItem::ExtendShard`, staying `Ready` with degradation flags.
2. `t_resume = t_solve + missing bytes / β_fetch + t_pack + t_load + t_warm + t_replay`. Choose the minimum. Within 20 %, prefer the better steady-state `t_tok`. A healthy node with a valid pack restarts its runner in place.
3. The repair plan is solved speculatively from data-plane evidence (`EdgeDown` from both neighbours and a failed coordinator probe). It is committed only on `NodeFailed`, or after 4 s of continuous unreachability.
4. Planned transitions:
   - finish if the sequence's remaining tokens fit the drain deadline (60 s latency mode, 300 s throughput mode);
   - otherwise take the cheapest of KV migration (make-before-break only), replay and re-prefill.

   A sequence still running at the deadline ends with a retryable `EpochRetired` error.
5. A degraded plan exits when a re-plan gains ≥ 10 %, half the normal 20 % threshold, subject to payback.
6. Stragglers raise `Drift` and are never killed mid-sequence.

**Revised 2026-10-20:**
7. **An evictable second copy on the predicted absorber.** Each stage shard's source chunks get one more copy on the node the repair solve would absorb that stage into. It travels over the LAN at background priority and never from the Hub, is admitted only while that node keeps 2 × `store.min_free_bytes` free and within `store.standby_max_bytes` (RFC-0041 §4.4, §9), is evicted before any other chunk, and is never placed on battery-class nodes. The dashboard shows the disk it uses. Pre-packing (`Standby::Warm`) stays on non-ring nodes unless DIST-E14-T16's repair timings show it pays on ring nodes (RFC-0033 §9.3). The proposal it replaces, automatic `Warm` on a non-ring node, fired only when such a node existed, and most two- and three-node home clusters have none. A full replication factor of 2 was rejected because it doubles cluster disk for the largest models.

## Consequences

- Positive:
  - Typical recoveries resume in tens of seconds with no client-visible error (SC-28).
  - Runner crashes cost a reload and a replay, not a re-placement.
  - The cluster keeps serving at reduced speed rather than refusing.
- Negative:
  - The amendment to RFC-0001 §5.4 and §7.3 (RFC-0032 amendment 1) is required.
  - Cold recoveries were slow when the lost node held the only copy of its shard: ≈ 220 s from the Hub in RFC-0033's example. Item 7's second copy brings the common case down to a local read plus packing, at the cost of spare disk the user can cap.
- Work created:
  - RFC-0011: repair-mode solving with `RepairConstraints` and time-to-resume;
  - RFC-0041: fetch priority for replacements;
  - RFC-0060: the chaos suite.

## What would make us revisit

- Chaos-suite measurements show time-to-resume predictions off by more than 50 %, leading to wrong strategy choices.
- Degraded plans persist for long periods in field telemetry, opted in, because the exit threshold is never met. Lower it further, or force exit when the lost node returns.
- Cold recoveries from the Hub still dominate field reports, because second copies are often evicted or never made for lack of disk. Revisit the margin and the cap, or pre-packing on ring nodes.
