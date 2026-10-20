# ADR-0208: Saved sessions as per-stage, per-layer, 4 KiB-aligned blobs bound to model, KV settings and transcript

- Status: Accepted (2026-10-20; revised 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0022))
- Related: ADR-0005, ADR-0009, ADR-0013, ADR-0019, ADR-0205, ADR-0206; RFC-0001 §5.9, §5.10 `PinSession`; RFC-0021 §13; RFC-0022 §8; RFC-0041; RFC-0050

## Context

- A surveyed engine saved 63,025 tokens of state in 1.2 GB, wrote it in 710 ms and restored it in 898 ms, against ≈30 s to re-read the chat; each file was bound to an identity hash of the model files and every setting that changes meaning, including the resolved RoPE configuration (`tiered-memory-inference.md` §12.1). SC-19 asks for restore within 1.25 × the sequential read time plus the new turn's prefill.
- In Syntropy a session's KV is spread over stages, one per runner, each holding its own layers. A re-placement may change the stage split between a session's turns, and the node that saved part of a session may be offline.
- Pages move bit-for-bit in exact mode (ADR-0019 item 5), and direct I/O needs 4 KiB-aligned extents (ADR-0013).
- Sessions are the only KV state recorded in the cluster event log, as `PinSession { session, instance, holders }` (RFC-0001 §5.10).

## Options considered

| Option | For | Against |
|---|---|---|
| A. One opaque blob per stage (a memory dump of the page pools) | Fastest to write | Cannot resume under a different split; not direct-I/O friendly; tied to pool layout |
| B. Re-export as one model-wide file at the head | One file | Every stage's KV must cross the network to the head on every save |
| C. Store only the token list and re-prefill on resume | Tiny | ≈30 s instead of ≈1 s; defeats the purpose |
| **D. One blob per (session, stage); a fixed header with the identity fields; a section table; one 4 KiB-aligned section per (layer, class, token range), page-first, each with its own BLAKE3; the token list in the head's blob; durable by write, data-sync and atomic rename; logged with `PinSession` only once every stage's blob is durable** | Restores with sequential direct reads; any stage split can fetch exactly the layers it needs; integrity per section; incremental appends for long sessions | Stages must coordinate the log entry; a format to version |

## Decision

**Option D**, with the byte layout in RFC-0022 §8.2 (magic `SYNKVSB1`, version 1).

1. **Identity.** A blob restores only if `model_id`, `kv_settings_hash` (KV dtype, layout version, page size, resolved RoPE configuration, window sizes, compression ratios, KV-affecting approximate knobs), `page_tokens` and `kv_format` match the instance, and `token_hash` matches the transcript ingress holds. Any mismatch falls back to re-prefill; nothing partial is used.
2. **Sections** are per (layer, class, token range), so a different split reads or fetches only its layers, and a long live session appends sections incrementally every sealed page past `kv.checkpoint_tokens`.
3. **Durability.** A stage's saved session is a set of segment files plus a manifest listing them with their BLAKE3 hashes. Every segment, then the manifest, is written to a temp name, data-synced and renamed; nothing is ever appended to a renamed file. Long live sessions add a segment every `kv.checkpoint_tokens`. Each stage sends `SessionSaved { session, stage, manifest }` to the head over the ring's side channel; the head submits `PinSession` only when it holds one per stage, within `kv.save_timeout`.
4. **Lossy KV** sets a flag; such a blob never resumes on an exact instance.
5. **Retention** is store-managed: 7 days since last use and at most min(10% of the store budget, 50 GiB) per node by default, least recently used first.
6. **Encryption and saves** (added 2026-10-20, below). From M2, each segment is sealed under a fresh content key that its holder wraps to the session's X25519 public key (`SequenceOpen::session_pk`) and then drops; only the session's ingress, which holds the private key, can unwrap it, and it does so for one blob at a time during a restore (RFC-0034 §13). The blob carries flag bit 2 and a plaintext key section with the wrapped key and the `session_pk` fingerprint. Every re-save on release is incremental: one new segment of the pages sealed since the last save. Saves obey a per-volume endurance budget read from SMART (RFC-0022 §8.6).

## Consequences

Positive:
- SC-19 is a sequential-read problem: 1.2 GB from NVMe at 6 GB/s is ≈0.2 s [estimate].
- Sessions survive re-placement and, in M3, the loss of the node that ran a stage, if its blob was fetched or replicated.
- A restored session's pages enter the prefix index, so the following turn is an ordinary prefix hit.

Negative:
- Disk use grows with agent activity; retention defaults may need tuning.
- The header and section table are a versioned format that RFC-0062's compatibility policy must cover.
- Blobs reveal conversation content to anyone who can read the store directory; the store's 0700 permissions (RFC-0001 §12) are the protection, and encryption at rest is not in v1.

## What would make us revisit

- Agent workloads in M1 show sessions resumed mostly under the same split, with per-layer sections costing measurable write time: coalesce sections per stage.
- Users need sessions to move between machines outside a cluster (export and import by hand): add a portable, model-wide export format next to this one.
- A security review (RFC-0034) requires encryption at rest for session blobs. *Acted on 2026-10-20: from M2, content keys wrapped to a per-session key pair whose private half stays on the session's ingress node (below).* Revisit turning autosave off on nodes that are not the session's ingress after M2's disk data.

## Revisions

- **Revised 2026-10-20.** Decision 6 added. Session KV and transcripts persisted for 7 days on several machines, unencrypted, including laptops run by other people, and the M4 holder-key option never protected against the holder. From M2 a holder seals what it writes under a content key wrapped to the session's public key, so it can never read its own blob later; only the session's ingress can unwrap one, during a restore, which bounds a holder's exposure to the time it processes the session, while autosave stays on for fast agent resumes. The endurance part lands by M1: every re-save is incremental, and a write budget is read from SMART, because an agent workload can approach a consumer drive's rated endurance within a year. RFC-0034 §13 owns the cipher and keystore; RFC-0022 §8.2 and §8.6 own the format and the budget.
