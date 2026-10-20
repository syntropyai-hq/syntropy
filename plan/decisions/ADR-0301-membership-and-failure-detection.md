# ADR-0301: Membership is SWIM-derived with full-mesh heartbeats up to 16 nodes, suspicion, and suspend-aware clocks

- Status: Accepted
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0009, ADR-0011, RFC-0030 §5, RFC-0032 §7, RFC-0033 §3

## Context

ADR-0009 fixes SWIM-style gossip with 1 s heartbeats on LAN and no quorum. This ADR fixes the parameters and the variations.

- **Cluster size.** Home clusters are 2–6 nodes. Classic SWIM probes one random member per period. With round-robin, a given node is first probed after about N/2 periods on average and N periods at worst. At N = 8 that misses SC-25 ("a dead node is detected within 3 missed heartbeats").
- **Surveyed timeouts** of 15 s and 30 s suit volunteer WANs, not a LAN where a 1–2 s heartbeat notices a failure within one or two tokens (`multi-device-inference.md` §1.5).
- **False positives are expensive.** `NodeFailed` triggers re-placement, and a re-shard can move hundreds of GB (`hardware-and-performance-model.md` §5.3). A node pegged at 100 % GPU with a busy decode ring, or a Wi-Fi link with a 250 ms p99, must not be declared dead.
- **Laptops sleep.** A waking node that compares wall time sees every peer as silent for minutes. Stale sessions after a wake caused one-way splits in a surveyed runtime, which added a 12 s ignore window as a fix (`multi-device-inference.md` §1.2).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Classic SWIM at every size | Constant per-node load at any N | Detection about N/2 periods at small N; misses SC-25 at N ≥ 6 |
| B. Fixed all-to-all heartbeats with a hard timeout | Simplest; fast | No way to tell an edge failure from a node failure; false positives under jitter |
| C. Leader-only heartbeats (every node ↔ coordinator) | O(N) messages | Coordinator loss blinds detection; no indirect evidence |
| **D. Full-mesh heartbeats up to 16 nodes, SWIM indirect probes and suspicion with incarnation refutation, Lifeguard-style local health and confirmation-scaled suspicion timeouts, coordinator confirmation before `NodeFailed`, and suspend-aware clocks** | Meets SC-25 for suspicion; separates edge from node failure; resists false positives; handles sleep | More states than B; randomised SWIM still needed above 16 |

## Decision

**Option D** (RFC-0030 §5).
1. Beats travel on QUIC DATAGRAM frames on the control connection, every 1 s on LAN and 2 s on WAN, to every member while N ≤ 16. Above that, randomised SWIM probing takes over.
2. After 2 missed periods, 3 helpers probe indirectly, preferring other interfaces. After 3 missed periods with no ack, the node is Suspect, gossiped with its incarnation, and refutable. If a helper acks, the edge is reported down, not the node.
3. The suspicion timeout shrinks with independent confirmers: 2–4 s on LAN, 10–20 s on WAN. A node whose own health is degraded multiplies its timeouts by up to 9.
4. Only the coordinator records `NodeFailed`, after its own confirming probe.
5. Suspend is detected by comparing a clock that stops in suspend with one that keeps running. Affected pairs are `CLOCK_MONOTONIC`/`CLOCK_BOOTTIME`, `mach_absolute_time`/`mach_continuous_time`, and `QueryUnbiasedInterruptTime`/`QueryInterruptTime`. A resumed node bumps its incarnation and enters an 8 s grace period in which it accuses nobody.
6. Will-sleep notifications produce a `Leave{Sleep}`, so peers recover without waiting out suspicion. Idle sleep is inhibited while a mains-powered node serves live sequences, never on battery, and lid close is never blocked.
7. Load digests for per-request routing ride the same beats. Nothing per-request enters the event log.

**Revised 2026-10-19:** A planned daemon restart, for an update or `syntropy restart`, sends `Leave { reason: Restart { back_within_ms } }`, capped at `membership.restart_grace` (30 s), with `back_within_ms` taken from RFC-0052's `update.restart_budget`. Peers suppress suspicion of the node, and the coordinator starts no recovery for it, until the deadline. The node's runners keep serving remote-ingress sequences in orphan mode (RFC-0033 §8.2). If no beat arrives by the deadline, normal failure handling applies (RFC-0030 §5.9).

## Consequences

- Positive: suspicion within about 3.5 s and confirmation within about 8.5 s on LAN [target]. Control traffic stays a few KB/s per node at home scale. Sleep and roaming cause no election or failure storms.
- Negative: confirmation takes longer than SC-25's 3 s. Recovery therefore relies on data-plane evidence (RFC-0033) to start earlier. Full-mesh beats cost O(N²) messages, which is why the cap is 16.
- Work created:
  - RFC-0060: simulated-fabric tests for detection time and false positives, with an injected suspend clock;
  - RFC-0033: data-plane liveness as a fast path.

## What would make us revisit

- Measured confirmation time on wired LAN is above 3 s and recovery cannot start earlier from data-plane evidence. Shorten `t_max`, or let a stage loss start recovery on Suspect.
- More than 1 false `NodeFailed` per node-week on wired LAN under the RFC-0060 load suite.
- Typical clusters exceed 16 nodes. Validate randomised SWIM at that size, or raise the cap.
