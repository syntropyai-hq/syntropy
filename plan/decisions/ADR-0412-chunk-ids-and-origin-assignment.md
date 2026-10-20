# ADR-0412: Chunk IDs are learned from the first fetch and committed by digest through the coordinator; origin fetches are assigned by weighted rendezvous hashing

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0000 (D4), ADR-0009, ADR-0011, ADR-0016; RFC-0001 §5.10; RFC-0041 §6, §8; RFC-0030, RFC-0031, RFC-0034

## Context

- ADR-0016: chunk boundaries at tensor and expert starts and every 4 MiB; chunk ID = BLAKE3 of the bytes, computed by the first node to fetch the chunk; the coordinator records the completed table through the event log; every chunk from a cluster node is verified before use; for a chunk no cluster node holds, the coordinator assigns the origin fetch to exactly one node, ties broken by WAN throughput. Its revisit clause: move tables out of the log if bookkeeping exceeds 1 % of the log's steady-state size.
- A 1.4 TB model has ≈ 500 K chunks, ≈ 24 MB of table, once a boundary is forced at every expert slice (RFC-0041 §6.2; ADR-0016's ≈ 335 K assumed no forced boundaries). One log event per chunk, for IDs or for assignments, would dominate the log.
- No node may ever hold a whole file, so the origin's file SHA-256 cannot check partial ranges. TLS authenticates the origin's bytes; the Hub's chunked storage adds its own hashes where used.
- Cluster members are trusted with the weights they handle (ADR-0000 D4). Untrusted swarms are out of scope.
- RFC-0001 §2.3: the coordinator is not on the request path; coordinator changes must not disrupt work.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Each receiver trusts the hash its sender declares | No coordination | A faulty or compromised member can serve wrong bytes with a matching declared hash |
| B. One log event per learned chunk ID and per origin assignment | Literal reading of ADR-0016 | Hundreds of thousands of events per model; replay and snapshot cost |
| **C. IDs proposed in batches whose bodies travel over the bulk plane by digest; the coordinator accepts first-writer-wins and logs only batch digests; origin assignment is a pure function of replicated state (weighted rendezvous hashing over nodes that need the chunk and have origin access)** | Every node verifies against one agreed table; ≈ 80 log events per 1.4 TB model; no assignment messages; coordinator changes alter nothing | A short delay before others can verify a newly fetched chunk; trust in the first fetcher's hash |

## Decision

**Option C.**

1. Chunk boundaries are a pure function of the file header (RFC-0041 §6.1).
2. A holder (the indexing node for local files, the origin fetcher otherwise) computes `ChunkId`s and proposes them with `ProposeChunkIds { file, range, body, evidence }` in batches flushed every 250 ms or 4096 entries. A batch binds the cluster only with evidence in one of RFC-0034 §10.1's forms: origin chunk hashes that every receiver checks against origin metadata it fetched itself; two `STAGE_HOST` witnesses that each read the bytes themselves; or a keyholder's whole-file verification. Without evidence the batch is provisional, and its chunks are never served to peers for packing. Where the origin publishes no chunk metadata, the second node that needs a chunk reads it from the origin and co-signs, so such chunks cross the WAN up to twice. The coordinator fetches the body by digest, accepts entries that are empty or equal, rejects batches that conflict, and appends `ChunkIdsCommitted { file, range, body }`.
3. Conflicts trigger refetch from the origin by the proposer; a persisting conflict quarantines the file until one node downloads it whole and checks its SHA-256.
4. Every chunk received from a cluster node is verified against its committed ID; a mismatch marks the source suspect for that model and is reported to the coordinator.
5. The origin fetcher of an unheld chunk is `argmax_n w(n) / −ln(u(c, n))` over nodes whose current shard needs it and that have origin access, with `u` from BLAKE3 of (`FileDigest`, chunk index, node ID) and `w = 2^k`, k being the node's WAN rate class (log2 of MB/s, rounded down) recorded in the newest committed plan; an unmeasured node takes the median class of the measured candidates. `w` is never read from a live profile. Every node computes the same assignee. A node waiting longer than `p2p.origin_wait_s` with no progress fetches from the origin itself.

## Consequences

- Positive: cluster-wide verification of every byte with a log cost of a few dozen events per model.
- Positive: origin load splits across nodes in proportion to their WAN rate; membership changes move only the affected nodes' share.
- Negative: chunk-ID authenticity rests on the first fetcher (TLS to the origin, plus the Hub's chunk hashes where used) until some node holds the full file and checks its SHA-256. This is the trust-on-first-fetch ADR-0016 already accepted.
- Negative: the liveness fallback can fetch a chunk from the origin twice when an assignee is slow, not dead. Accepted: a little extra WAN beats a stalled load.

## What would make us revisit

- Commitment latency shows up in load times (for example on many-node clusters, where batches queue at the coordinator). Let nodes accept provisional IDs from the fetcher for chunks not yet committed, verified later.
- RFC-0034 introduces a member trust level below "trusted with weights". Then the first-fetcher rule needs a second, independent confirmation.
- ADR-0016's 1 % log-size trigger fires anyway; move the batch digests into a separate replicated object.

## Log

- Revised 2026-10-19: decision 5's weight changed from the live measured rate to a quantised class recorded in the committed plan, because online profile refinement made the assignment differ between nodes.
- Revised 2026-10-19: decision 2 requires chunk-table evidence instead of trusting the first holder, so one member cannot poison every other node's chunk IDs.
