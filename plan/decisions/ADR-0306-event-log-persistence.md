# ADR-0306: Every node persists a hash-chained event log with periodic snapshots, and diverged or lagging nodes catch up by snapshot

- Status: Accepted
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0009, ADR-0304, ADR-0305, RFC-0032 §6

## Context

ADR-0009 requires snapshots and catch-up from day one. The surveyed runtime retrofitted them after late joiners broke clusters (`multi-device-inference.md` §2.1). Home clusters power-cycle entirely, for example after a house power cut, and must come back with their admissions, invites and plans intact. Without quorum, a network split yields divergent log suffixes that must be detected and discarded on heal (ADR-0304). The log carries plans of up to a few MB compressed and is otherwise low-volume.

## Options considered

| Option | For | Against |
|---|---|---|
| A. In-memory log only; rebuild from re-reported facts after restart | Nothing on disk | Invites, revocations, certificates and plans are not all re-reportable facts; a full power cycle loses the cluster's history |
| B. Only the coordinator persists | Less disk I/O | The coordinator's disk becomes special; a new coordinator after a restart starts empty |
| C. Embedded database (SQLite, an LSM store) | Queries, transactions | A dependency for an append-only log plus one snapshot; more failure modes |
| **D. Every node appends CRC-framed CBOR entries to segment files with a BLAKE3 hash chain; durable-class events fsynced by the coordinator before acknowledging; group commit otherwise; snapshots every 4,096 entries or 15 minutes; two snapshots retained; divergence or lag beyond the retained tail is repaired by snapshot transfer over the blob plane** | Simple, append-only, crash-safe; divergence detected in one comparison; any node can serve as coordinator after a power cycle | Each node uses some disk (MB-scale); snapshots must be produced deterministically |

## Decision

**Option D** (RFC-0032 §6.6, §6.7).
1. Record format: `len u32 | crc32c u32 | cbor(LogEntry)`, with segments rolling at 16 MiB. Each entry carries `chain = BLAKE3(prev.chain ‖ index ‖ term ‖ cbor(event))`.
2. Durability:
   - The coordinator fsyncs before replying for admissions, revocations, key rotations, invite creation and use, certificate issuance, keyholder changes, plan commits, cluster configuration changes and elections.
   - Everything else, on every node, is group-committed every 100 ms.
3. Snapshots:
   - The snapshot is `zstd(cbor(ClusterState))`, named by BLAKE3 and written by atomic rename.
   - A new one is written every 4,096 entries, or every 15 minutes if anything was appended.
   - Each node keeps its two newest snapshots and the entries after the older one.
4. Sync: a node presents its last `(index, chain)`. The coordinator replies with the following entries if the chain matches and the tail is retained. Otherwise it replies with a snapshot digest, fetched over `syntropy/blob/1`.
5. Restart: load the newest valid snapshot, replay the local segments while verifying CRCs and the chain, truncate a torn tail, then sync.

**Revised 2026-10-19:**
1. **Verbatim bytes**. Entries store and hash the event's CBOR exactly as the coordinator encoded it (`LogEntry::event_bytes`). Nodes never re-encode, so followers one minor behind compute the same chain.
2. **Durability means replication**. The coordinator replies to a command only after one other coordinator-eligible member has acknowledged it (fsynced for durable-class events), or after `log.ack_timeout` with `replicated = false`.
3. **Command journal.** Every originator journals its commands and re-submits any whose `CommandId` is absent from its log after each term change or snapshot replacement (RFC-0032 §6.10). Every event type has a stated reconciliation path.
4. **Configuration after a heal**. A diverged node finds the fork point by chain comparison (`ForkProbe`). It re-proposes its configuration, including RFC-0054 ledger entries accepted on the discarded branch, with that fork attached. A re-proposal applies only if the key has not changed in the surviving log since the fork. Otherwise the surviving value stands and both sides report the dropped change.

**Revised 2026-10-19, second revision:**
- **Validated snapshots.** Snapshots keep the signed record behind every authority-derived field. `validate_snapshot()` re-verifies them, and the coordinator's election proof, before a node adopts a snapshot. A failing snapshot is refused, and the node fetches the same index from another member (RFC-0032 §6.7).
- **Deny lists and rotation statements** survive snapshot replacement.

## Consequences

- Positive:
  - A cluster survives a full power cycle.
  - Late joiners start from a snapshot.
  - Divergence after a split is found by comparing one hash.
  - `syntropy cluster state --diff` can pinpoint disagreements.
- Negative:
  - The state must serialise deterministically, which ADR-0305's deterministic CBOR rules provide.
  - Disk use is small, in the tens of MB, but non-zero on every node.
- Work created:
  - RFC-0060: power-cycle and torn-write tests;
  - RFC-0053: log and snapshot metrics.

## What would make us revisit

- Snapshot transfer or restore time exceeds 5 s on realistic clusters. Move plan bodies out of the state into digest-addressed blobs.
- Log growth between snapshots exceeds 100 MB, meaning something high-rate has leaked into the log (ADR-0009 revisit).
