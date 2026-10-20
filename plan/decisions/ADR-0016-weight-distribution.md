# ADR-0016: Weight distribution: shard-only range fetch, cluster-first content-addressed chunks
- Status: Accepted; revised 2026-10-20 (decisions 7 and 8)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0005, ADR-0008, ADR-0009, ADR-0011, ADR-0015, ADR-0018; RFC-0011, RFC-0031, RFC-0033, RFC-0034, RFC-0041, RFC-0051, RFC-0052

## Context
- **Model sizes.** 2026 frontier MoE models run from 1.6T to 2.8T total parameters, natively 4-bit (`plan/research/backends-and-portability.md` §0 item 6). That is hundreds of GB to more than a terabyte per model. A node running a stage needs only its shard: the byte ranges of its layers, the experts it owns, plus embeddings or LM head if it is the head or tail stage (ADR-0006, ADR-0010).
- **Downloading everything everywhere is the observed failure.** A surveyed event-sourced runtime has every node download the whole model from the Hub. Its per-layer file filter was written and then disabled, because tensor parallelism needed every file and there were no sticky sessions. Users asked why, and streaming from other devices stayed a to-do (`multi-device-inference.md` §7, §13 item 13).
- **Other designs seen** (§7):
  - a coordinator that streams slices from a model it holds whole
  - a hash-first remote tensor cache
  - content-addressed peer-to-peer blocks verified on receipt
  - per-stage packages published to the Hub

  The research concludes that content-addressed transfer from any holder scales best and lets a placement change move only the deltas. The coordinator should never need the whole file.
- **LAN beats WAN.**
  - Measured LAN goodput: 2.35 Gbps on 2.5 GbE and 9.42 Gbps on 10 GbE. RDMA over TB5 measured 3.5–3.8 GB/s file transfer on M3 Ultra (`hardware-and-performance-model.md` §2.5).
  - A re-shard of 377 GB takes ≈108 s at 3.5 GB/s, or ≈5.4 min at 10 GbE goodput (§5.3, estimate).
  - Home WAN links are usually slower than the LAN, so each byte should cross the WAN once per cluster.
- **Byte ranges come straight from the formats.**
  - GGUF stores MoE experts as `{n_embd, n_ff, n_expert}` tensors, so each expert is a contiguous byte range at an offset given in the header (`backends-and-portability.md` §0 item 5).
  - safetensors gives per-tensor offsets in a JSON header, plus an index file for multi-file checkpoints.
  - Either way, a shard can be computed from headers alone, fetched with HTTP range requests.
- **Integrity.** Hub files carry a whole-file SHA-256. That cannot be checked for a byte range unless someone holds the whole file. BLAKE3 is a tree hash: chunks hash independently and in parallel, and a subrange can be verified against a root.
- **Downloader basics worth keeping** (§7): parallel transfers, resume, rate-limit handling, multiple model directories, read-only shared directories, a true offline mode.

## Options considered
1. **Every node downloads the full model.** Simplest, but it costs N× WAN bytes and N× disk, and is bounded by the WAN. Rejected.
2. **The coordinator downloads the full model and streams slices.**
   - It needs the whole model on one node's disk.
   - It is a single-source bottleneck.
   - A coordinator change in the middle of a transfer stalls everything.

   Rejected.
3. **A shared network filesystem** (NFS or SMB). It needs setup the user may not have, and it adds no integrity. Supported as an optional read-only source, not as the mechanism.
4. **A public content-addressed swarm protocol** (DHT-based). Public swarms are out of scope for v1 (D4), and its discovery and trust model is wrong for a private cluster. We take the content addressing, not the protocol.
5. **Per-stage packages published per split.** Someone must publish each (model, split) pair, it cannot follow the solver's arbitrary placements, and Syntropy publishes no interchange format (ADR-0005). Rejected.
6. **Chosen: each node fetches only its shard's chunks, from cluster nodes first and the origin second, with BLAKE3 verification and chunk-level resume.**

## Decision
1. **Chunk table per model.**
   - On first sight of a model revision, a node fetches only the file headers (range requests) and builds the chunk table.
   - Chunk boundaries fall at every tensor and every expert start, and every 4 MiB within them. A shard therefore maps to whole chunks with no over-fetch.
   - Each entry is `(file, offset, len, blake3)`. The chunk ID is the BLAKE3 of its bytes, computed by a node that holds the chunk.
   - The coordinator records the table in batches through the event log (ADR-0009), so every node verifies against the same table. The first fetcher is never the sole authority (revised 2026-10-19): a batch binds the cluster only with evidence, either origin chunk hashes that every receiver checks itself, two independent witnesses that each read the bytes, or a keyholder's signature (ADR-0412, RFC-0041 §6.3, RFC-0034 §10.1). Until then the batch is provisional and its chunks are never served to another node for packing.
   - Estimated table size: ≈1.4 TB in ≈500K entries once chunk boundaries are forced at expert-slice boundaries (about two chunks per ≈5.6 MB expert record), about 24 MB, in ≈122 batches. That is an estimate, from the 2.8T-parameter 4-bit case (RFC-0041 §6.2; revised 2026-10-19 from ≈335K at a flat 4 MiB per chunk).
2. **Shard = chunks.**
   - The placement plan's shard for a node resolves to a chunk set.
   - A re-placement fetches only the chunks the node lacks and frees chunks no longer in any of its shards, after the new plan is `Ready` (ADR-0015).
   - The solver costs a re-placement with the measured transfer time of the missing chunks (RFC-0011), so it can trade plan quality against movement.
3. **Source selection, per chunk.**
   1. The local store, including read-only shared directories.
   2. Cluster nodes whose store inventory lists the chunk, ranked by the measured bulk β of the link, with up to 4 sources striped in parallel.
   3. The origin: Hugging Face Hub, or any HTTPS URL that supports `Range`.

   For a chunk no cluster node holds, the coordinator assigns the origin fetch to exactly one node: the one that needs it, ties broken by measured WAN throughput. Other nodes then fetch it from that node, so each origin byte crosses the WAN once per cluster where the origin publishes chunk metadata, and at most twice where it does not, because the second node that needs a single-witness chunk reads it from the origin to witness it. Where the Hub serves a file through its chunk-based storage protocol (Xet), the fetcher requests only the chunks covering the needed ranges.
4. **Transport.**
   - Cluster transfers use the iroh blob plane (`syntropy/blob/<major>`, ADR-0011), paced so that live stage edges keep their p99.
   - Chunks are pushed into the store through the direct-I/O write path (ADR-0013), never through the page cache.
5. **Verification.**
   - Every chunk received from a cluster node is verified against its BLAKE3 entry before it is stored or served. A mismatch discards the chunk, marks the source suspect for that model, and refetches from another source.
   - Origin bytes are authenticated by TLS and, where the origin offers them, by its chunk hashes.
   - The whole-file SHA-256 from the Hub is checked whenever a node holds a complete file. Any mismatch revokes the chunk table and forces a refetch.
6. **Resume.**
   - Resume works per chunk, and within a chunk through HTTP `Range` or the blob protocol's offset.
   - Partial chunks live in a staging area and are promoted only after verification.
   - A daemon restart resumes from the store's inventory.
7. **Store.**
   - A content-addressed local store keyed by BLAKE3, spread over multiple writable model directories (the first with enough free space wins) and with optional read-only directories.
   - Free space is checked per volume before any fetch is admitted.
   - Packs (ADR-0005) are derived locally. A node may fetch a pack from a cluster node with the same device-class key instead of rebuilding it, verified the same way.
   - *Revised 2026-10-20:* that shortcut stays off by default at every milestone, because a hash check guards transit, not a dishonest repack. A user who turns it on gets the label "trusts members", rebuild-and-compare of at least 1% of received extents, and the warm-up canary on every model served from them (RFC-0041 §8.5).
8. **Credentials and offline mode.**
   - Hub tokens for gated models stay on the node that holds them and are never sent between nodes (RFC-0034).
   - `--offline` makes no origin calls and fails with the list of missing chunks.
   - *Revised 2026-10-20:* for gated and custom-licence models, a node receives no chunk, from the origin or a peer, until its operator has accepted the model's licence on that node, recorded locally. Across owners, a peer transfer is redistribution by the token holder, so each recipient accepts the terms itself. Holders serve, and origin assignment counts, only nodes that announce acceptance (RFC-0041 §7.5). From M3, before invites ship.

## Consequences
- **Positive.**
  - A node joining a cluster downloads only its shard, and mostly over the LAN.
  - WAN bytes are about one model per cluster, not one per node.
  - Re-placement moves deltas.
  - Corrupt or malicious chunks from a compromised node are caught before use.
  - A coordinator change only redirects origin assignments.
- **Negative.**
  - Chunk tables and store inventories add state to the event log.
  - Bytes fetched by range from the origin are trusted on first fetch through TLS until a whole-file check is possible.
  - Nodes that do origin fetches use extra disk for chunks they serve but do not run. These are reclaimed by an LRU over unreferenced chunks.
- **Work created.**
  - RFC-0041: chunk table, store, fetch planner, Xet support.
  - RFC-0031: the blob plane.
  - RFC-0011: transfer cost in the re-placement objective.
  - RFC-0033: refetch on source failure.
  - RFC-0034: token handling and suspect sources.
  - RFC-0051: per-node transfer progress and sources in the dashboard.
  - RFC-0052: offline mode and the model directory configuration.

## What would make us revisit
- The Hub stops honouring `Range` on resolved files, or changes its chunk protocol incompatibly. Shard-only origin fetches would then need a full-file fallback.
- Measured cluster distribution is slower than direct origin fetch on a typical home setup, such as a multi-gigabit WAN with a Wi-Fi LAN. In that case, order sources by measured throughput only.
- Chunk-table bookkeeping exceeds 1% of the event log's steady-state size or slows catch-up measurably. Then store tables outside the log and keep only their root hashes in it.

## Revisions

- Revised 2026-10-19: the chunk-table estimate follows RFC-0041 §6.2: forcing chunk boundaries at expert-slice boundaries gives about two chunks per expert record, so ≈500K entries rather than ≈335K.
- Revised 2026-10-19: items 1 and 3: a chunk ID committed on one fetcher's word let a single member poison every other node's verification. Batches now need evidence, and the WAN bound is once with origin chunk metadata, up to twice without.
- Revised 2026-10-20: decision 8 adds licence acceptance on the receiving node before chunks of a gated or custom-licence model flow to it. Decision 7's same-class pack shortcut stays off by default, with rebuild-and-compare sampling when a user enables it.
