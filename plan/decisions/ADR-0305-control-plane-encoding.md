# ADR-0305: Control-plane messages are deterministic CBOR via serde, gated by a cluster feature level

- Status: Accepted
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0009 (item 8), ADR-0011, ADR-0015, RFC-0001 §3.5 and §5, RFC-0031 §9, RFC-0032 §6.4

## Context

ADR-0009 requires a versioned binary schema for control messages, and mixed incompatible versions refused at admission. Within one protocol major, different minor releases must interoperate: a rolling upgrade has nodes on N and N−1 for minutes to days.

The control types are Rust serde types defined in RFC-0001 (`ClusterCommand`, `ClusterEvent`, `PlacementPlan` and others). The IPC between daemon and runner already uses postcard, which is safe there because both ends are the same build (RFC-0001 §3.5). Cross-node messages have no such guarantee.

Control volume is low: events per second, KiB-scale (RFC-0001 §2.2). The exception is `PlanCommitted`, which can carry tens of thousands of `TierAssignment` entries for a large MoE model. One surveyed runtime's JSON-encoded control plane, with tokens on it, spent half a follower's time parsing (`multi-device-inference.md` §2.3). Here the control plane carries no tokens, so parse cost is not on any hot path.

## Options considered

| Option | For | Against |
|---|---|---|
| A. postcard (as IPC) | Compact, fast, reuses serde types | Not self-describing: adding a field breaks older decoders; unusable across minors |
| B. Protocol Buffers (prost) | Robust field evolution; unknown fields skipped; language-neutral | A second definition of every RFC-0001 type in `.proto`, plus conversion code; drift between the two |
| C. FlatBuffers / Cap'n Proto | Zero-copy reads; evolution rules | Separate schemas again; zero-copy buys nothing at control rates |
| D. JSON | Debuggable | Large and slow; float and binary handling awkward; failure history in the survey |
| **E. CBOR (RFC 8949) via serde, deterministic encoding, `#[serde(default)]` on new fields, unknown keys ignored, new enum variants emitted only when the cluster's minimum feature level allows them, zstd above 64 KiB** | Reuses the RFC-0001 types directly; tolerates additive changes; self-describing for debugging tools; compact enough with compression | Field names on the wire cost bytes; variant gating needs discipline in the coordinator |

## Decision

**Option E** (RFC-0031 §9.1).
1. Envelope: `len: u32 | enc: u8 (0 CBOR, 1 zstd CBOR) | body`. The maximum is 64 MiB. Larger state, such as snapshots, moves on the blob plane by digest.
2. The encoding is deterministic: shortest integers, definite lengths, declaration-ordered struct fields, sorted map keys. Identical state then produces identical bytes, which the log hash chain relies on (RFC-0032 §6.1).
3. Evolution:
   - fields are added only with defaults;
   - fields are never removed or retyped within a major;
   - decoders never use `deny_unknown_fields`;
   - an enum variant introduced at feature level L is emitted only when every recipient advertises ≥ L. The coordinator uses the cluster minimum; peers use the level in their `Hello`.
4. Data-plane and blob frames do not use CBOR. They use the fixed layouts of ADR-0307. Only rare control bodies on the data plane, `SequenceOpen` and its kin, embed CBOR.

## Consequences

- Positive: one type definition per message; rolling upgrades within a major; `syntropy debug decode` can pretty-print any captured control message without knowing its version.
- Negative:
  - Plan bodies are larger than with protobuf before compression.
  - A coordinator bug that emits a gated variant early breaks older followers. A CI mixed-version test guards against it (RFC-0031 §Testing).
- Work created:
  - RFC-0032: feature-level tracking in `ClusterState`;
  - RFC-0062: mixed-minor CI job.

## What would make us revisit

- Compressed `PlanCommitted` events exceed 4 MiB for v1 families on 8 nodes, or snapshot catch-up time is dominated by CBOR decode. Move plan bodies to a columnar binary form on the blob plane, and log only their digest.
- A non-Rust tool needs to produce control messages. Generate a CDDL schema from the serde types, or move to protobuf for those messages.
