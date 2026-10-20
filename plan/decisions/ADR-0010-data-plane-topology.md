# ADR-0010: Data plane topology

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D2, D3), ADR-0003, ADR-0006, ADR-0008, ADR-0009, ADR-0011, ADR-0012, ADR-0017, ADR-0019, RFC-0001, RFC-0012, RFC-0031, RFC-0032, RFC-0033, RFC-0042, RFC-0050

## Context

The data plane carries everything on the token path between nodes: prompt token IDs to the first stage, hidden states between stages, sampled tokens back to the start, and generated tokens to the client. Each hop on the critical path costs `α + S/β`, and in pipeline decode a single stream pays the sum of all hops every token (`plan/research/hardware-and-performance-model.md` §3.7).

**Topologies seen** (`plan/research/multi-device-inference.md` §2.3, §2.6, §13):
- **Client-holds-the-graph star.** A copy between two remote servers fails over to a host copy, so activations go server → client → server and every stage boundary pays two hops.
- **Coordinator-forward star.** Every boundary returns to the coordinator.
- **Direct ring.** Stages send to the next stage, and the last stage sends back to the first. This is the only shape that pays one hop per boundary.
- **Tokens through the control plane.** Every token was JSON-encoded, sent to the single writer, indexed and broadcast to all nodes. This capped throughput at 800–1,000 events/s and contributed to cluster collapse under load (ADR-0009).
- **End-of-step all-gather.** In one runtime every stage all-gathers the final hidden state so each can run the LM head and sample with the same seed. That costs a collective per token and duplicates the LM head on every stage (§4.1, §3.9).

**Message sizes** (§5.4, §6; `hardware-and-performance-model.md` §3.7) [estimates]:

| Case | Size | Wire time |
|---|---|---|
| Decode hop, BF16, hidden 8192 | 16 KiB | 56 µs serialisation on 2.5 GbE, plus α ≈ 100–250 µs |
| Decode hop, F32, hidden 8192 | 32 KiB | 0.26 ms on 1 GbE |
| Sampled token, tail to head | a token ID plus a small header | latency only |
| Prefill, 4K prompt, F32 | 128 MiB | ≈2.7 s at 400 Mb/s |
| Prefill, 4K prompt, BF16 | 64 MiB | 0.23 s on 2.5 GbE, 27 ms over RDMA on TB5 |

At batch-1 decode on a LAN, activation size is small next to link latency. Prefill is where size matters, and chunked, pipelined prefill hides transfer whenever a chunk's compute takes longer than its transfer (`multi-device-inference.md` §4.3, §6).

**Precision on the wire** (§6). One surveyed runtime sends raw F32 by policy after F16 and Q8 attempts failed in production, and requires any compression to be a separately versioned protocol with a safety contract. Another syncs Q8_0 buffers successfully for its supported models. The sources disagree, so precision must be measured per model family, for decode and prefill separately. ADR-0000 D3 makes exact output the default.

**Speculative decoding** lets one verify window cross each stage boundary once instead of once per token (`tiered-memory-inference.md` §16, §20 item 6).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Star through a coordinator or client | Central control; simple bring-up | Two hops per stage boundary; a central bottleneck |
| B. Tokens and activations over the control plane | One transport | Observed collapse under load; couples the token rate to election health |
| C. Every stage holds the LM head and samples after an all-gather | No tail-to-head dependency | A collective per token; duplicated LM-head bytes on every stage |
| **D. Direct stage-to-stage ring: the tail samples and returns the token to the head; tokens stream from the tail to the ingress node** | One hop per boundary plus one small return hop; no central path; LM head held once | The head and tail must both be reachable on the data plane; ring order matters to the solver |

## Decision

**Option D.**

1. **Stages connect directly.** Each adjacent stage pair holds one persistent bidirectional data-plane stream. The transport is chosen per edge from measured profiles (ADR-0011). No activation or token ever passes through the coordinator or the control plane (ADR-0009).
2. **Pipeline decode runs as a ring**: head → … → tail → head.
   - The **head stage** holds the embeddings and starts each step.
   - The **tail stage** holds the LM head and runs the sampler. It sends the sampled token ID back to the head: for a verify window, the accepted count and the next token (ADR-0017). Hidden states never travel from tail to head.
   - On a single node, head and tail are the same runner and no network hop exists.
3. **Ingress and output.** The ingress node tokenizes the prompt and sends token IDs to the head over the data plane. The tail streams sampled token IDs, plus logprobs when requested, point-to-point to the ingress node, which detokenizes and serves the API stream (RFC-0050). Stop conditions and grammar constraints are evaluated at the tail, which also holds the tokenizer.
4. **Frames.** `ActivationFrame` (RFC-0001, RFC-0031) carries a small header (plan epoch, session ID, sequence ID, position range, verify-window ID, dtype, shape) and the payload, written into preallocated buffers. Plans and captured graphs are reused, so a per-token frame carries only activations. A frame with a stale plan epoch is rejected. Senders retain the per-sequence boundary history that RFC-0033 needs for replay after stage loss.
5. **Precision in exact mode** (revised 2026-10-19). F32 is the exact-mode default. BF16 is allowed for a model family only where the parity harness has passed it for that family, for decode and prefill separately (ADR-0019 decision 5), which the conformance manifest records as a `WireCert` and the solver checks before choosing it (RFC-0060 §6.4). Without a certificate, exact mode never sends below F32.
6. **Approximate mode.** Activations below the family's passed precision (BF16 where no `WireCert` exists, FP8, Q8) are opt-in per model or per request and labelled approximate wherever they appear (D3, ADR-0019). Each compression is a versioned protocol variant, enabled per family only after the parity harness measures it for decode and prefill separately (RFC-0042).
7. **Prefill is chunked and pipelined** across stages with staggered starts. Chunk size comes from the plan and fast routing (ADR-0008).
8. **Collectives inside a fast island** (ADR-0006) run peer-to-peer among the island's members, never through a coordinator.
9. **Throughput mode** keeps several sequences in flight around the ring and batches the frames of sequences that share a hop.

## Consequences

Positive:
- Each stage boundary costs exactly one hop, and the return path carries a token ID instead of a hidden state.
- Token throughput is independent of control-plane health, so coordinator changes cannot stall generation.
- The LM head and embeddings are held once each, which the solver counts on the tail and head (ADR-0008).

Negative:
- The tail-to-head return is one more hop per token in multi-node plans. The solver must count it (`D` hops for `D` stages).
- Exact mode doubles prefill wire bytes against BF16 for families the harness has not passed at BF16.
- The head and tail need data-plane reachability to each other, and the tail to the ingress node. Off-LAN members may need relayed paths (ADR-0011).

Work created: RFC-0031 (frame format, streams, versioned compression variants), RFC-0032 (ring execution, pipelined prefill, island collectives), RFC-0012 (head and tail roles, sampler placement), RFC-0033 (sender-side boundary history), RFC-0042 (per-family precision measurements), RFC-0050 (ingress streaming).

## What would make us revisit

- F32 prefill transfer exceeds 25% of TTFT on 2.5 GbE for Tier 1 models after chunked pipelining. Evaluate lossless activation compression, which keeps exact mode, before any lossy default.
- The tail-to-head return hop measures as a significant share of per-token time on supported links. Evaluate co-locating head and tail, or duplicating the embedding table on the tail.
- **Settled 2026-10-19 (its BF16 reading is superseded by the revision below, so BF16 follows ADR-0019 decision 5).** F32 is the exact-mode wire default. BF16 is allowed only where the reference itself holds that tensor in BF16, so it is never a down-conversion. The original requirement's wording is superseded by item 5. The research recommends F32 by default and cites production failures of F16 and Q8 (`multi-device-inference.md` §6, §15.4 item 4). Item 5 reads BF16 as allowed only where it is not a down-conversion from the reference. If the original requirement meant BF16 as a general exact-mode option, that contradicts D3 for models whose reference keeps the residual stream in F32.

## Revisions

- **Revised 2026-10-19:** item 5 and the settled note above read BF16 as allowed only where the reference itself holds BF16. The reference holds the residual stream in F32 (ADR-0019 decision 1), so that reading never allowed BF16, while ADR-0019 decision 5 allows it per family once the harness passes it. Item 5 now follows ADR-0019: F32 by default, BF16 where certified for the family and phase. The note's point stands that BF16 is never a silent default.
- **Revised 2026-10-20:** both revisit triggers were judged against the cost model, and neither fires for v1. The tail-to-head return is one ≈ 128 B token frame, about 0.1–0.2 % of per-token time on 2.5 GbE, so the direct return stays; its share is recorded as a column of the SC-13 benchmark instead of a standalone benchmark (RFC-0032 §3.4). F32 prefill transfer is about 2 % of per-chunk compute in example (c1), so no lossless activation compression is built for v1, and a nightly row keeps measuring it (RFC-0032 §3.8) (OQ-NET-09, OQ-NET-10).
