# ADR-0011: Transport stack: iroh/QUIC for control and bulk, a ranked `Transport` trait for the data plane
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0006, ADR-0009, ADR-0010, ADR-0012, ADR-0016, ADR-0018; RFC-0010, RFC-0030, RFC-0031, RFC-0032, RFC-0033, RFC-0034

## Context
Syntropy moves three kinds of traffic between nodes, and they want different things from a transport.

| Class | Size and rate | What matters |
|---|---|---|
| Control: cluster commands and events, gossip, invites, plan distribution | small, bursty, off the token path | authentication, NAT traversal for invite links (D4), ordered delivery |
| Data plane: activation frames between stages, sampled tokens, cross-node TP/EP collectives | 8–32 KiB per hop per decode token; up to 32 MiB per prefill chunk; 120–160 collectives per token for TP | one-way latency α and its p99, because every hop sits on the decode critical path |
| Bulk: weight shards, pack deltas, saved KV sessions | tens to hundreds of GB per re-placement | throughput, resumability, and not disturbing the data plane |

Message sizes come from `plan/research/multi-device-inference.md` §5.4: a decode hop carries `hidden × 2 B` in BF16 (14 KiB for a 7168-wide model, 16 KiB at 8192) or twice that in F32.

Link latencies, from `plan/research/hardware-and-performance-model.md` §2.5:
- 2.5 GbE: ping min 0.17–0.18 ms RTT, average 0.28–0.55 ms [verified].
- 10 GbE through the kernel stack: 210.9 µs per send/recv on a Mac Studio [verified].
- TCP over TB5: ≈300 µs [verified].
- RDMA over TB5: a small-message floor of ≈10 µs and 21.3 µs per send/recv. Plan on 2.5–4 GB/s per link [verified].
- Wi-Fi: 3–7 ms median, p99 ≈250 ms [verified].

Cross-node TP pays only when the effective per-collective latency, including GPU synchronisation, is below α_eff,max ≈ 200–300 µs (§3.8, estimate). Only RDMA-class links get there. For pipeline decode, each hop adds α + S/β, and a token crosses each stage boundary once per window (ADR-0010).

Observed failure modes this decision must avoid (`multi-device-inference.md` §13):
- plaintext data planes (item 12)
- tokens riding the control plane (item 1)
- RDMA errors that are not recovered in-process and need full reboots (item 15)
- clusters that silently refuse to form when version strings do not match (item 24)

Facts about RDMA over Thunderbolt 5, from Apple TN3205:
- It is available from macOS 26.2 on Apple silicon Macs with TB5. It is enabled per machine from Recovery with `rdma_ctl enable`.
- It exposes the Verbs API (`infiniband/verbs.h`, `librdma.tbd`).
- It supports send/receive only, at most 10 unreliable-connection (UC) queue pairs and 4095 outstanding work requests, and messages up to 16,773,120 bytes. The sender and receiver must post the same frame count.
- Thunderbolt does no hardware acknowledgement, so a send completion proves neither delivery nor integrity.
- Links are point-to-point and are not routed. Rings work only if the application forwards data. The Thunderbolt Bridge must be disabled when the cables form a loop.

`multi-device-inference.md` §5.5 adds that peers may fail to discover each other unless both run identical macOS builds.

iroh gives a Rust QUIC endpoint with TLS 1.3, an Ed25519 public key as the endpoint identity, hole punching and relay fallback, under MIT/Apache-2.0 (ADR-0018). Its identity model matches the node ID in the brief's vocabulary directly.

## Options considered
1. **iroh/QUIC for every plane.** One stack, one identity, encryption everywhere.
   - QUIC's userspace packet processing and per-packet AEAD put its latency floor at or above kernel TCP's: hundreds of µs on Ethernet, never the 10–25 µs RDMA reaches. This is an estimate, which RFC-0010's probes will check.
   - Cross-node TP on RDMA-capable meshes becomes impossible by construction. Rejected as the only transport.
2. **Raw TCP (`TCP_NODELAY`) plus TLS for everything.** Mature, and it works where UDP is blocked.
   - No NAT traversal or relays for invite links, and no key-based identity unless we build one.
   - Multiplexing planes on one connection puts weight bulk and activations behind the same head-of-line. Rejected as primary; kept as a fallback.
3. **libp2p.** It has NAT traversal and identity, but it is built around DHTs and public swarms (out of scope by D4), with a heavier stack and more moving parts per connection. Rejected.
4. **Collective libraries (NCCL, RCCL, MPI) as the cross-node data plane.**
   - These are vendor-specific and assume homogeneous devices and a static set of participants fixed at launch.
   - They cannot span a Mac, a CUDA box and a Strix Halo host, or survive membership change. Rejected across nodes. Intra-node multi-GPU uses the device runtime's peer copies under ADR-0006.
5. **Overlay-only networking** (Tailscale or WireGuard, then plain sockets). D4 allows an overlay for off-LAN joins, but an overlay is an IP network, not a transport. Syntropy runs over it like any other IP path and does not depend on it.
6. **Chosen: iroh/QUIC for control, invites, off-LAN data and bulk, plus a `Transport` trait for the data plane, with backends ranked per edge by measured cost.**

## Decision
1. **Planes and connections.**
   - Control and bulk use iroh. The node ID is the iroh endpoint key.
   - Each plane gets its own ALPN, `syntropy/ctl/<major>` and `syntropy/blob/<major>`, and its own connection, so bulk congestion state never delays control.
   - A node whose protocol major differs is refused at the handshake, and the refusal is shown in CLI and dashboard, not only logged.
2. **Data-plane backends behind `Transport`** (signature in RFC-0001, semantics in RFC-0031). Each edge picks the first available backend in this order:
   1. `Shm`: same host (co-located runners, ADR-0015).
   2. `Rdma`: RDMA over TB5 on macOS 26.2+, or ibverbs over RoCEv2 or InfiniBand on Linux. libibverbs is loaded at runtime (dlopen), not linked.
   3. `Quic`: iroh on a direct LAN path, ALPN `syntropy/data/<major>`.
   4. `Tcp`: TLS 1.3 over TCP with the same Ed25519 identity, used when UDP is blocked.
   5. `Quic` over a relay: the off-LAN last resort.
3. **Selection is measured, not assumed.** RFC-0010 probes every candidate on every edge: 64 B, 16 KiB, 1 MiB and 64 MiB ping-pong, at least 200 samples, recording α, β and p99, through the real GPU-buffer → wire → GPU-buffer path. The edge then runs on the candidate that minimises `α + S/β` at its actual frame size S, with a p99 penalty. So a fast QUIC path can beat a misconfigured RDMA path. The chosen backend and its numbers are recorded per edge in the resource graph and shown in the dashboard.
4. **Authentication is never delegated to the link.**
   - RDMA and shared-memory endpoints are paired only by exchanging GID/QPN (or segment handles) over the authenticated control connection.
   - RDMA over TB5 runs on a physical cable between two admitted nodes, so it is enabled automatically.
   - RoCEv2 crosses a switch and carries activations unencrypted, so it is used only on interfaces marked `trusted_fabric` in config. RFC-0034 owns this policy.
5. **Integrity on every frame.** `ActivationFrame` headers carry the plan epoch, session/sequence ID, position range, dtype, shape, a per-edge sequence number and a CRC32C of the payload. Receivers acknowledge with credits. This is mandatory on RDMA over TB5, which has no hardware acks. Frames are preallocated and registered once; nothing is allocated per token.
6. **Limits respected by construction.**
   - Queue pairs are a per-device budget (revised 2026-10-19). Thunderbolt 5 allows 10 per device, shared by every instance, every overlapping plan epoch and every recovery lane on the node. The daemon's device ledger tracks them (`RdmaPortInfo::max_qp`), the solver counts them per plan, and at most 4 per neighbour (data in and out, plus collectives) is a cap within that budget. A channel that would exceed its grant opens on the next transport with `TransportDemoted`; a plan never fails for lack of queue pairs (RFC-0031 §5.2).
   - Prefill chunks above 16,773,120 bytes are split.
   - Ring forwarding over non-adjacent TB links is done by the stage runtime (RFC-0032).
7. **In-process recovery.**
   - A queue pair entering the error state is torn down and recreated without restarting the runner.
   - After 3 failures on an edge within 60 s, the edge demotes to the next backend in the list for the rest of the plan epoch. The sender resends its retained, unacknowledged frames on the new backend, byte-identical and in order (ADR-0307); this is not RFC-0033's activation-history replay, which is for a lost stage (revised 2026-10-19).
   - Demotion and its cause are emitted as cluster events.
8. **Bulk yields to the data plane.** On a link that carries a live stage edge, bulk transfers are paced so that the edge's measured p99 stays within 1.2× its idle p99. This is the default and is tunable.

## Consequences
- **Positive.**
  - One identity across all planes, encryption by default on every IP path, and NAT traversal and relays for invite links without new code.
  - The fastest link is used where it exists, and the cluster keeps working when it doesn't.
  - Cross-node TP/EP (ADR-0006) has a physical path that can reach its break-even α.
- **Negative.**
  - There are four data-plane backends to build, fuzz and test. The `Rdma` backend has two variants (macOS TB5 and Linux verbs) with different queue-pair types and limits.
  - RDMA over TB5 needs a manual Recovery step on each Mac, and it is sensitive to macOS build mismatches. Syntropy can only detect and explain these, not fix them.
  - There is no RDMA backend on Windows in v1.
- **Work created.**
  - RFC-0031: the trait, frame format, credits and ALPNs.
  - RFC-0010: the link probes and continuous re-measurement.
  - RFC-0030: discovery and invites on iroh.
  - RFC-0034: the trust rules, including `trusted_fabric`.
  - RFC-0033: failover and replay.
  - RFC-0060: a simulated fabric in `syntropy-testkit` with injected loss, latency and queue-pair errors.
  - RFC-0051: the dashboard states "RDMA hardware present but not enabled" and "macOS build mismatch".

## What would make us revisit
- On Tier 1 CI hardware, LAN QUIC's α at p50 is more than 50 µs worse than kernel TCP's on the same link. Order TCP before QUIC on the LAN and keep QUIC for off-LAN.
- RDMA over TB5 still needs node reboots to recover after 6 months of soak testing. Make it opt-in instead of auto-selected.
- Apple adds reliable-connection queue pairs or one-sided operations, which would change the collective design in RFC-0032.
- A consumer Windows RDMA path (Network Direct on mainstream NICs) becomes available. Add a backend.
- iroh changes licence, drops relays or breaks its endpoint-identity model.
- Settled 2026-10-19: the `trusted_fabric` opt-in for RoCEv2 refines the original requirement, which put RDMA first without qualification. It follows the research recommendation never to ship a plaintext data plane by default (`multi-device-inference.md` §15.1).

## Revisions

- **Revised 2026-10-19:** item 6: "4 per neighbour" is a per-neighbour cap inside a per-device queue-pair budget that the ledger tracks and the solver plans against. Item 7: a demotion resends retained frames under ADR-0307's acknowledgement rule; RFC-0033 replay is not involved.
- **Revised 2026-10-20:** item 2's fifth backend, `Quic` over a relay, uses a hosted shared relay service as the built-in set. The iroh project's free public relays are documented by their operator as not for production use, with unpublished rate limits, so they serve internal and alpha testing only. The relay list is a TUF target, `relays.json`, signed by the targets role, so the relay provider can change without a release; `custom` and `disabled` remain. Bulk weight transfer never uses a relayed path (RFC-0031 §5.4). Installs stay on one iroh version for months, so the relay set must keep supporting the relay protocol versions inside the compatibility window. The revisit trigger "iroh drops relays" also covers the hosted service ending.
- **Revised 2026-10-20:** RDMA over TB5 (item 2) gets a raw-path probe on a TB5 Mac pair late in M1, before the M2 frame-path run. If the raw path's `α_eff` exceeds 65 µs, SC-17 becomes report-only at M3 and cross-node tensor parallelism on Mac pairs moves to M4 or after 1.0. If the probe's one-day fault smoke test needs a reboot to recover, TB5 RDMA ships opt-in at M2, the revisit condition above brought forward (RFC-0031 open question 1).
