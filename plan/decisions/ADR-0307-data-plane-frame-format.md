# ADR-0307: Data-plane frames use a fixed 64-byte header, CRC32C, and credit flow control with sender retention

- Status: Accepted
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0010, ADR-0011 (items 5–7), ADR-0012, RFC-0001 §5.8 and §8.4, RFC-0031 §6–§8, RFC-0033

## Context

Every data-plane frame sits on the decode critical path, and every byte of software around the wire adds to `Z` in SC-12 (≤ 2·t_switch, roughly 10–40 µs). Requirements:

- **Receive buffers.** Frames are written into preallocated, registered buffers with the payload 64-byte aligned (RFC-0001 §5.8). No per-frame allocation is allowed (ADR-0012).
- **Transports with no delivery guarantee.** Thunderbolt 5 RDMA offers only unreliable-connected queue pairs, and its send completions prove neither delivery nor integrity (TN3205, ADR-0011). So frames need sequence numbers, checksums and acknowledgements at the application level.
- **Failover.** ADR-0011 item 7 demotes a failing edge to another backend in-process. If in-flight frames vanish with the old connection, the stage must replay activation history (RFC-0033), which is far costlier than resending a few frames.
- **Fuzzability.** Formats must be fuzzable and stable within a protocol major, with additive minors.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Self-describing per-frame encoding (protobuf, CBOR) | Easy evolution | Builder allocations; no alignment control; parse cost per frame |
| B. Fixed header, rely on the transport's reliability | Simplest | Impossible on UC; failover loses frames, so every link fault becomes a stage-level replay |
| C. Fixed header and per-frame acks with timers per frame | Reliable | Timer per frame; complexity |
| **D. Fixed 64-byte little-endian base header (magic, minor, type, flags, instance, epoch, step, per-channel sequence number, extension and payload lengths, payload CRC32C, header CRC32C), a bounded type-specific extension padded so the payload is 64-byte aligned, and receiver-granted credits that double as cumulative acks; the sender retains every frame until acked and resends on NACK, timeout or reconnect** | One mechanism on every backend; exactly-once in-order delivery across failover; bounded memory; header built in place | Frame overhead of 128–256 B; a credit frame per data frame on decode channels |

## Decision

**Option D** (RFC-0031 §6, §7).
1. The base header and every extension are defined byte for byte in RFC-0031. Golden frames are checked into the repository, and changing any of them requires a wire-minor bump.
2. CRC32C covers the header and the payload on every frame, on every backend, ADR-0011 item 5. On QUIC and TCP it is defence in depth against buffer-reuse bugs. On RDMA and shared memory it is the only integrity check.
3. Credits:
   - `limit_seq − ack_seq` equals the receiver's posted slots;
   - a sender at its limit gets `WouldBlock` and the stage holds (RFC-0001 §8.4 rule 1);
   - decode channels credit every frame, which also yields an α sample per frame.
4. Retention:
   - `Channel::completed` fires only when the peer has cumulatively acked;
   - the retention window is sized at `Warming`;
   - reconnect, on the same or another backend, resends `(ack_seq, last_sent]` in order with the original sequence numbers.
5. Activation-history replay is reserved for stage loss. Link loss never needs it.

**Revised 2026-10-19:**
- **Control frames**. `Hello`, `HelloAck`, `Credit`, `Ping` and `Close` carry `chan_seq = 0`, consume no credit and are never NACKed. A credit is a cumulative snapshot. Data frames start at `chan_seq = 1`, and the initial `limit_seq` equals `recv_slots`.
- **RDMA receives.** RDMA keeps receives posted for control frames as well as data.
- **Byte-identical resends**. The `RETRANSMIT` flag is removed, and a resend is byte-identical to the original.
- **AEAD.** Under AEAD, the tag is a 16-byte trailer counted in `payload_len`, and sealed bytes are never resealed.
- **Parts.** `Token` frames and expert-parallel frames gain part fields.

## Consequences

- Positive: UC queue pairs are safe; per-edge demotion is invisible to compute; bounded, preallocated memory; one parser to fuzz.
- Negative:
  - About 8 µs of serial CRC work per hop at 8 GB/s, which RFC-0031 overlaps with the H2D copy.
  - Retention holds prefill-sized slots on prefill channels: 96 MiB per side at 16 MiB chunks. The solver charges this to the pinned budget.
- Work created:
  - RFC-0060: fault-injection property tests;
  - RFC-0061: Z microbenchmark;
  - RFC-0020: network sub-arena sizing.

## What would make us revisit

- Measured Z on Tier 1 rigs exceeds 2·t_switch, with CRC as the dominant term even after overlap. Move CRC into the output kernel, or skip payload CRC verification on AEAD transports with an explicit ADR.
- Per-frame credits cost measurable α on Wi-Fi or relayed edges (RFC-0031 open question 3).
