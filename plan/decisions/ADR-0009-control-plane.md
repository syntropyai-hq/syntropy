# ADR-0009: Control plane

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D2, D4), ADR-0008, ADR-0010, ADR-0011, ADR-0012, ADR-0015, RFC-0001, RFC-0030, RFC-0032, RFC-0033, RFC-0034, RFC-0053

## Context

The control plane holds cluster membership, model instances, placement plans and runner lifecycle. It must stay correct while home devices sleep, roam and reboot, and it must never become part of the token path.

**What worked in surveyed designs** (`plan/research/multi-device-inference.md` §2.1, §2.7):
- An event-sourced single-writer state: nodes emit events, one writer assigns a global index, `apply(state, event)` is a pure function, followers apply in order and NACK gaps.
- Snapshots for laggards, a whole-state debug endpoint, and a pure per-node `plan()` that emits at most one action per tick.
- Placement, `apply` and election are unit-tested as pure functions.
- Snapshot catch-up was retrofitted late, after late joiners broke clusters, so it must exist from day one.

**What failed** (§2.2, §2.3, §10.5, §13 items 1–2):
- **Tokens on the control plane.** Every token was JSON-encoded, indexed by the single writer and broadcast to every node. Parsing took about half of a follower's time under load, and throughput capped at 800–1,000 events/s. Under load the queues filled and pipes broke, which triggered re-elections and runner teardown.
- **Re-election on every connection change**, with a 3 s campaign and back-off of 15–300 s only for some triggers.
- **A coordinator takeover failed in-flight requests**, because the API ended them.

**Failure detection** (§1.5): surveyed timeouts are 15 s node silence and 30 s coordinator silence, tuned for slow networks. On a LAN, a 1–2 s heartbeat notices a failure within one or two tokens. A gossip mesh with a peer-down message and grace period is the other surveyed pattern (§2.5).

**Cluster sizes and quorum.** Home clusters are often two or three devices (`hardware-and-performance-model.md` §4b, §4c). A majority-quorum protocol on two nodes cannot make progress after losing either one, which contradicts the goal of serving from whatever survives.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Majority-quorum replicated log (Raft-class) | Linearisable; no split brain | A two-node cluster stops after losing one node; needs three always-on nodes to be useful |
| B. Coordinator-free gossip with mergeable (CRDT) state | No election; always available | Concurrent placement decisions conflict; plan epochs and runner lifecycles need one ordering |
| C. Event-sourced single writer with bully election on connection changes, as surveyed | Proven state model | Flapping elections and failed in-flight requests, as observed |
| D. No control plane; the head stage drives the ring | Minimal | No discovery, auth, multi-model or recovery (§2.4) |
| **E. Event-sourced single writer; SWIM membership; epoch-fenced coordinator lease; no quorum** | Keeps the proven state model; survives any single loss; churn does not reset requests | Split brain is possible during a network split and must be reconciled on heal |

## Decision

**Option E.**

1. **Event-sourced cluster state** (type summaries in RFC-0001 §5.10; the event log, `apply()`, snapshots, election and terms in RFC-0032; membership and admission in RFC-0030):
   - Nodes submit typed `ClusterCommand`s with idempotency IDs. The **coordinator** is the single writer: it validates commands and appends typed `ClusterEvent`s to the log with a monotonically increasing index.
   - `apply(state, event) -> state` is pure and deterministic, and every node applies the same log. Every node also runs a pure `validate()` on each entry and never applies one that fails it; authority events carry keyholder signatures, so the coordinator cannot forge membership or keys (RFC-0032 §6.3, RFC-0034 §8.3). `plan()` per node is a pure function of state that emits local actions (fetch, pack, load, drain) for the node daemon to perform.
   - Snapshots are taken periodically. A node that falls behind the retained log tail catches up from a snapshot plus the tail. A whole-state endpoint serves the current state for debugging.
2. **What the log holds**: membership and admission, capability-profile digests, model instances, placement plans by plan epoch (ADR-0008), runner lifecycle states, and fetch progress. **It never holds** tokens, activations, KV, per-token timings or request payloads. Those use the data plane (ADR-0010) or the metrics pipeline (RFC-0053).
3. **Membership and failure detection** use SWIM-style gossip: direct and indirect probes, then suspicion, then confirmed failure. Heartbeats run every 1 s on LAN. Suspicion timeouts default to a few seconds on LAN and are longer for invite-link members on WAN paths, set in RFC-0030. Control traffic uses bounded queues that drop rather than block on a slow peer. NACK and resend recover gaps.
4. **Coordinator election uses an epoch-fenced lease.**
   - The coordinator holds a lease renewed by heartbeat. Every event and command carries the coordinator epoch `(term, coordinator node ID)`, ordered lexicographically. Nodes reject anything carrying an older epoch.
   - An election starts only when the lease expires. Candidates are ranked by a fixed priority tuple: an operator preference flag first, then a capability class (mains-powered, wired and not mobile ranks above battery or mobile; iOS and Android nodes are ineligible by default), then the lowest stable node ID.
   - There is **no majority-quorum requirement**. A single surviving node elects itself.
5. **Anti-flap and back-off.** No preemption: a returning higher-priority node does not depose a coordinator with a valid lease. Connection changes alone never trigger an election. Repeated elections within a window back off exponentially. Defaults are set in RFC-0030.
6. **A coordinator change never fails an in-flight request.** Stages talk directly (ADR-0010), and tokens stream to the ingress node without the coordinator. Runners keep executing the current plan epoch while the coordinator is absent. Fast per-request routing runs on the ingress node from its state replica (ADR-0008). A new coordinator resumes from its replica plus a snapshot, keeps the current plan epoch, and re-plans only if membership actually changed. Losing a node that hosts a stage is a separate fault path (RFC-0033).
7. **Network splits and heal.** Without quorum, each side of a network split may elect its own coordinator and keep serving the instances it can run. On heal, the higher epoch wins. The other side's events since the fork are discarded, and its nodes re-submit their current facts (membership, residency, running instances) as commands. Running instances on the losing side are adopted or drained by the winner's next plan, never killed by the merge itself. Design rule: **no event is the only record of anything a node needs**. Cluster state must be reconstructible from node-reported facts.
8. **Wire format.** Control messages are a versioned binary schema over the control transport (ADR-0011). Cluster-key admission and node identity are specified in RFC-0034. Mixed incompatible versions are refused at admission, with the reason shown to the user.

## Consequences

Positive:
- Two-node clusters survive the loss of either node, and any cluster survives coordinator churn without failing requests.
- The control plane's load is independent of tokens/s, so it cannot collapse under generation load.
- A pure `apply()`, a pure `plan()` and election ranking are unit-testable and replayable from logs.

Negative:
- Split brain happens by design during network splits. The fact re-reporting path must be correct, and every event type needs a reconciliation story.
- The "reconstructible from facts" rule constrains what may live only in the log. For example, user-defined model aliases must also persist on the nodes that use them.
- SWIM tuning differs between LAN and WAN members, so mixed clusters need per-member timeouts.

Work created: RFC-0030 (SWIM parameters, admission, anti-flap defaults), RFC-0032 (event log, `apply()`, snapshots, lease, election and terms), RFC-0001 (`ClusterEvent`, `ClusterCommand`, epoch type summaries), RFC-0033 (stage loss and re-placement), RFC-0034 (admission and identity), RFC-0053 (metrics kept off the log), RFC-0060 (simulated network-split and heal tests on the testkit fabric).

## What would make us revisit

- Heals after a network split cause request failures or double-booked devices in the simulated-fabric suite or in field reports, and fixing reconciliation proves intractable. Offer an optional quorum mode for clusters with three or more always-on nodes.
- Control-plane event rate exceeds a few hundred events/s in a steady cluster. Something per-token has leaked into the log.
- Failure detection takes more than 3 s on wired LAN in measurements, so a stage loss costs more than a few tokens. Retune SWIM, or add data-plane liveness signals.

## Revisions

- **Revised 2026-10-19:** the event-log types, `apply()`, snapshots, election and terms are owned by RFC-0032, not RFC-0030; RFC-0030 owns membership and admission. Item 1 records that every node validates every entry, which closes the gap a single trusted writer left.
- **Revised 2026-10-20:** item 4's priority tuple gains a tie-break: operator preference, then capability class, then **"hosts no stage" before "hosts a stage"**, then lowest node ID. Only nodes whose admission certificate carries `COORDINATOR` are ranked at all (ADR-0304, RFC-0034 §5.2). A coordinator that also hosts a stage makes that stage's recovery wait for an election (RFC-0033 §9.1), and the tie-break avoids that whenever a class offers a choice. It never ranks a weaker node higher, because it acts only inside a class, and it is deterministic. Two-node clusters, where both nodes host stages, are unaffected. Stage hosting is read from the committed plan when the election starts, so the rank does not move during an election (OQ-NET-25).
