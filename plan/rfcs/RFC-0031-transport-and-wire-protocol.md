# RFC-0031: Transport and wire protocol

- Status: Draft
- Area: distributed runtime
- Depends on: RFC-0001 (§2.2, §3.2, §5.8, §8.3, §8.4, §9.4, §12, §14), RFC-0010, RFC-0012, RFC-0020, RFC-0022, RFC-0023, RFC-0030, RFC-0060, RFC-0032, RFC-0033, RFC-0034, RFC-0041, RFC-0042, ADR-0010, ADR-0011, ADR-0012, ADR-0016, ADR-0019, ADR-0305, ADR-0307, ADR-0308, ADR-0309, ADR-0314
- Date: 2026-10-19

## Summary

This RFC defines how bytes move between Syntropy processes on different nodes, and on one host between nodes:
- the `Transport` trait and its five backends;
- the rule that picks a backend per edge from measured numbers;
- the byte layout of every data-plane and blob-plane frame;
- the control-plane envelope;
- flow control, acknowledgement and retransmission;
- the zero-copy receive paths;
- versioning.

- **Backends.**
  - `SharedMemory` for co-located nodes.
  - `Rdma` in two variants: ibverbs over RoCEv2 or InfiniBand on Linux, and RDMA over Thunderbolt 5 on macOS 26.2 and later.
  - `Quic` on iroh for direct LAN and WAN paths.
  - `Tcp` with TLS 1.3 when UDP is blocked.
  - `QuicRelay` as the off-LAN last resort.

  Each data-plane edge runs on the backend that minimises `α + S/β` plus a p99 penalty at its real frame sizes, from RFC-0010's probes. ADR-0011's rank order breaks ties and covers unmeasured cases.
- **One frame format on every backend.** A fixed 64-byte little-endian header carries magic, version, frame type, instance, plan epoch, step, a per-channel sequence number, lengths and two CRC32C checksums. A type-specific extension follows, padded so the payload starts 64-byte aligned. Every message uses it:
  - `ActivationFrame` and `TokenFrame`, with side-channel items such as per-stage KV headroom, router hints and the prefix-cache messages;
  - sequence control, credits and KV chunks;
  - collective parts, and expert-parallel dispatch and combine;
  - recovery handshakes;
  - weight chunks and peer tensor-group reads;
  - parity taps.
- **Flow control is credit-based** and doubles as acknowledgement. A sender keeps every frame until it is cumulatively acked. That makes the unreliable Thunderbolt 5 queue pairs safe, and lets any edge fail over to another backend by resending its unacked frames, with no activation-history replay.
- **Zero-copy.**
  - RDMA lands frames in registered pinned slots, or in device memory where GPUDirect RDMA is confirmed.
  - IP transports land frames with exactly one host copy.
  - Shared memory writes the payload once.
- **Encodings by plane.** Control and join messages are CBOR with feature-level gating (ADR-0305). Data and blob frames use the fixed layouts here (ADR-0307). The protocol major is in each ALPN. Minors negotiate capabilities in the channel `Hello`.

## Motivation and goals / non-goals

Every hop sits on the batch-1 decode critical path: `t_tok = Σ_s t_s + D · t_hop` (cost model §3.7). On RDMA over Thunderbolt 5 the wire costs 10–25 µs (cost model §2.5). The effective collective cost back-solved from a surveyed RDMA cluster is about 130–150 µs (§3.8), so most of the cost sits in the software path around the wire. That path is what this RFC controls.

The surveyed failures to avoid (`multi-device-inference.md` §5, §13):
- plaintext data planes (item 12);
- tokens on the control plane (item 1);
- RDMA errors that need node reboots (item 15);
- JSON-encoded per-token messages;
- star topologies doubling hops (item 11).

Goals:
1. Meet SC-12: added time per hop ≤ `α_probe + S/β_probe + 2 · t_switch`.
2. No allocation, lock or blocking call per frame on hot threads (RFC-0001 invariant 9). Frames are written into preallocated registered buffers.
3. Lossless, ordered, authenticated, integrity-checked channels on every backend, including ones whose hardware gives no delivery guarantee.
4. In-process recovery from any transport fault, and per-edge demotion without restarting a runner (ADR-0011 item 7).
5. Byte layouts precise enough to implement and fuzz from this document, and stable within a protocol major.

Non-goals:
- Datacenter collective libraries (NCCL, MPI) across nodes (ADR-0011 option 4).
- Lossy activation formats in exact mode (ADR-0010). Their containers are defined here for approximate mode; their numerics belong to RFC-0042.
- RDMA on Windows in v1.
- One-sided RDMA operations in v1. Send/receive only, because that is all Thunderbolt 5 offers, and one code path serves both RDMA variants.

## Design

### 1. Planes, ALPNs and connections

| Plane | ALPN | Endpoint (both ends) | Connection granularity | Streams | Encoding |
|---|---|---|---|---|---|
| Control | `syntropy/ctl/1` | daemon ↔ daemon | one per node pair | log replication, commands, RPCs, election; membership beats as QUIC datagrams | CBOR envelope (§9.1) |
| Join | `syntropy/join/1` | candidate ↔ keyholder daemon | per attempt | one request stream | CBOR envelope |
| Bulk | `syntropy/blob/1` | daemon ↔ daemon | one per node pair, separate from control so its congestion state never delays control | one bidi stream per request | blob frames (§6.10) |
| Data, sequence channels | `syntropy/data/1` | ingress daemon ↔ head runner; tail runner ↔ ingress daemon | one per (daemon, runner) pair | one bidi stream per channel | data frames (§6) |
| Data, stage channels | `syntropy/data/1`, or RDMA, shared memory or TCP per §3 | runner ↔ runner | one per runner pair (QUIC, TCP) or one queue pair per channel (RDMA) | one stream per channel | data frames |

- The protocol major (`1` above) appears in every ALPN. A peer offering only another major fails the handshake. Discovery has already shown it as incompatible (RFC-0030 §4.1).
- Runners have their own iroh endpoints, keyed by a per-runner Ed25519 key that the daemon delegates with a `DataPlaneCredential` (RFC-0034 §6). The node's secret key never enters a runner (RFC-0001 §3.2). Ring neighbours exchange runner endpoint addresses through their daemons (RFC-0030 §8.3).
- When the head runner and the ingress daemon share a node, sequence channels use the IPC rings `local_sequences` and `local_tokens` (RFC-0001 §3.5). Those rings carry the **same frame bytes** defined in §6, so the head and tail code paths do not branch on locality.

*Note 2026-10-27 (QUAL-E3-T18): the rings' records until M2.* Until DIST-E11-T03 (M2), `local_sequences` and `local_tokens` carry one postcard-encoded `SequenceControl` or `TokenFrame` per slot, the `core` types themselves, as RFC-0001 §5.8's note of 2026-10-23 already has M0 parity taps do. §6 has no sequence-control frames until DIST-E8-T05 (M2). `syntropy_ipc::shm::records` is the one encoder and decoder, which ingress, the runner and `syntropy-testkit`'s fake runner share. DIST-E11-T03 moves both rings to the frame bytes above. RFC-0001 §3.5's note and §18.39 record this, for the maintainers to confirm.

### 2. The `Transport` trait and `Channel` in full

RFC-0001 §5.8 fixes the trait's methods. This section adds types and default-bodied methods only.

```rust
/// RFC-0001: Rdma(RdmaKind) | Quic | Tcp | SharedMemory. Added: QuicRelay, so a relayed
/// path is never confused with a direct one in plans, metrics and the dashboard.
pub enum TransportKind { SharedMemory, Rdma(RdmaKind), Quic, QuicRelay, Tcp }

pub struct TransportCaps {
    pub zero_copy_recv: bool,          // frames land in a registered slot with no host copy
    pub device_memory_recv: bool,      // can land in device memory (GPUDirect RDMA, Unified import)
    pub max_frame: Bytes,              // RDMA over TB5: 16,773,120; others: net.max_frame (64 MiB)
    pub needs_registration: bool,      // RDMA, shared memory
    pub encrypted: bool,               // QUIC, TCP: true; RDMA: per ADR-0314; shared memory: false
    pub reliable: bool,                // RC, QUIC, TCP: true; UC (TB5): false; app-level reliability fills the gap
    pub max_channels_per_peer: u16,    // TB5: 4 (ADR-0011 item 6)
    pub inline_max: u16,               // RDMA inline-send limit for small frames; 0 if none
}

/// RFC-0001 lists the first six. `Tap` (RFC-0060 amendment A4) carries teacher tokens and reference
/// top-K ids from the parity driver to the tail, and to the head when it drafts (§6.13).
pub enum ChannelPurpose { Activations, Tokens, Sequence, Collective, Replay, Bulk, Tap }

/// Identifies a channel. Two channels with equal keys never coexist on one endpoint pair.
pub struct ChannelKey {
    pub instance: InstanceId,
    pub epoch: PlanEpoch,             // the epoch the channel was opened for; Relink (RFC-0032 §4.4) opens new ones
    pub purpose: ChannelPurpose,
    pub from: StageIx,                // Sequence: StageIx::INGRESS (0xFFFF) → head; Tokens to ingress: tail → INGRESS
    pub to: StageIx,
    pub lane: u8,                     // parallel channels on one edge (collective member pairs, replay streams)
}

pub struct ChannelSpec {
    pub key: ChannelKey,
    pub pool: PoolRegistration,       // where this side's receive slots come from (§8)
    pub slot_size: Bytes,             // largest frame this side accepts, header included
    pub recv_slots: u16,              // receive slots posted = initial credit granted to the peer
    pub priority: ChannelPriority,    // Data > Collective > Replay > Bulk (QUIC stream priority, RDMA CQ poll order)
    pub wire: WireDType,              // checked against HopPlan::wire (RFC-0001 §5.4)
    pub retention: u16,               // frames kept for retransmission; ≥ the peer's recv_slots
}

pub struct RemoteEndpoint {
    pub node: NodeId,
    pub key: PublicKey,               // runner key, or the node key for a daemon
    pub addrs: SmallVec<[SocketAddr; 4]>,
    pub relay: Option<RelayUrl>,
    pub rdma: Option<RdmaPairing>,    // §5.2
    pub shm: Option<ShmOffer>,        // §5.1
}

pub trait Transport: Send + Sync {
    // RFC-0001 §5.8: kind, caps, register, connect, accept.
    /// Cost of a frame of `size` on this edge, from the profile (§3). Default: α_p50 + size/β.
    fn cost(&self, edge: &LinkCaps, size: Bytes, objective: Objective) -> Nanos { /* §3 */ }
    /// Re-establishes a broken channel with the same key and resends unacked frames (§7.4).
    async fn reconnect(&self, ch: &mut Channel) -> Result<(), Error> { Err(Error::unsupported()) }
}

impl Channel {
    // RFC-0001 §5.8: try_send, try_recv, completed, stats.
    pub fn key(&self) -> &ChannelKey;
    /// Frames this side may still send before WouldBlock.
    pub fn credit(&self) -> u32;
    /// Ask the peer to retransmit from `seq` (UC loss, CRC failure).
    pub fn nack(&mut self, seq: u64);
    /// Graceful close after the peer acks everything sent; reason travels in a Close frame.
    pub fn close(&mut self, reason: CloseReason);
}

pub struct OutFrame {
    pub ftype: FrameType,
    pub step: u64,
    pub flags: FrameFlags,
    pub ext: ExtBuf,                  // type-specific header, written in place in the slot's header area
    pub payload: PayloadSlice,        // registered slot + offset + len; never copied by the transport on RDMA/shm
}
pub struct InFrame {
    pub header: FrameHeader,          // parsed and CRC-checked base header (§6.1)
    pub ext: ExtRef,                  // borrowed from the receive slot
    pub payload: PayloadRef,          // borrowed; dropping the InFrame returns the slot and may emit a credit (§7)
}
pub struct SendTicket { pub chan_seq: u64 }
```

Semantics, all binding on every backend:
- `try_send` assigns the next `chan_seq` to a data frame, writes the base header and CRCs, and hands the frame to the backend. It returns `WouldBlock` when `chan_seq > limit_seq` (§7.1). It never blocks and never allocates. Control frames (`Hello`, `HelloAck`, `Credit`, `Ping`, `Close`) carry `chan_seq = 0`, consume no credit and are never NACKed (§7.1).
- `completed` yields a ticket only when the peer has **cumulatively acknowledged** that `chan_seq`. A local send completion is not enough. Until then the payload slot stays in the retention window (§7.3). This is stronger than "the NIC finished": Thunderbolt 5 send completions prove neither delivery nor integrity (TN3205, ADR-0011).
- `try_recv` yields frames strictly in `chan_seq` order, after header and payload CRC checks. A gap or a CRC failure is never surfaced to the consumer. The channel NACKs and waits (§7.2).
- A frame whose `(instance, epoch)` does not match the channel key is dropped and counted (RFC-0001 invariant 7). So is a frame whose epoch the runner no longer serves.
- `stats()` returns α samples from ack echoes (§7.1), bytes, retransmits, NACKs and credit-stall time. It feeds the telemetry ring (RFC-0001 §13) and online link refinement (RFC-0010).

**Note, 2026-10-22 (DIST-E7-T01): what the code above leaves open, as `net::transport` defines it.**
- **Async methods.** `connect`, `accept` and `reconnect` return `impl Future<Output = …> + Send`, and implementations write them as `async fn`. The pinned toolchain does not allow `dyn` on a trait with `async fn`, so `Transport` is used through generics. A boxed wrapper for choosing a transport at run time is DIST-E7-T10's or T11's call. `cost` comes with link selection (DIST-E7-T11), which defines `LinkCaps`.
- **`Channel` is concrete.** It wraps a `ChannelBackend`, which each backend implements, and so does the simulator's `SimTransport`, at one virtual call per frame. On demotion the channel's backend is replaced and its retained frames are sent again (ADR-0011 item 7).
- **Buffers.** `BufferPool` divides memory into equal, 64-byte-aligned slots. It reaches the memory through an `unsafe trait PoolMemory`. `tier` cannot depend on `net` (RFC-0001 §4.2 rule 3), so its pinned `NetPool` arena hands out owned leases that meet the contract (TIER-E2-T21), and a `runner` newtype implements `PoolMemory` over a lease and registers it with the transports (TIER-E2-T25). The `unsafe impl` sits in the one `runner` module that RFC-0001 §4.2 rule 8's note of 2026-10-23 lets hold `unsafe`. The contract covers a dispatcher's device copies: whoever issues a copy into or out of a slot keeps the slot's writer or reader until the copy completes. Each slot is in one of three states: free, held by one writer, or held by readers.
  - `PayloadSlice` and `ExtBuf` are `Copy` descriptors: pool, slot, offset and length, into a frozen slot. `ExtBuf`'s length includes padding.
  - `PayloadRef` and `ExtRef` are leases on a slot; dropping the last one frees it. A sending channel holds such leases on every unacknowledged frame, and has released them by the time `completed` hands the ticket back.
  - `PoolRegistration` holds the pool, the transport and the backend's handle.
  - §8's header areas, registration tables and lock-free slot handout are DIST-E7-T12's.
- **Small types.**
  - `SendError`: `WouldBlock`, `TooLarge`, `Invalid`, `Closed` or `Link`.
  - `CloseReason`: an optional §12 code and up to 256 bytes of text.
  - `ChannelStats` counts, each way, the data frames and bytes `try_send` accepted and `try_recv` handed over. It also counts retransmits, NACKs sent, credit stalls and their time, α samples, CRC failures and stale frames.
- **Placeholders.**
  - Until `core` has RFC-0001 §5.12's `Error`, `net::Error` stands in: a class, a message and an optional §12 code. The code is optional because §12 gives some refusals none.
  - `StageIx::INGRESS` is `net::transport::INGRESS` (0xFFFF) until `core` adds it.
  - `RemoteEndpoint`'s `rdma` and `shm` fields come with the tasks that define their types: DIST-E7-T07 and T06. Its `relay` is DIST-E7-T23's `endpoint::RelayUrl`.
  - DIST-E7-T23's `PathKind` converts to `TransportKind` (`Direct` → `Quic`, `Relay` → `QuicRelay`), and its `EndpointError` to `net::Error`. None of its errors has a §12 code, so each keeps a class: `InvalidRequest` for bad input or configuration, `Link` for a failed connect, bind or accept, and `Cancelled` for an endpoint this node closed. `EndpointError::Connect` covers every connect failure, a refused ALPN as well as a timeout, so it is not `ConnectTimeout` (0x020A). Telling a timeout from a refusal, and giving only the timeout 0x020A, is DIST-E7-T03's.
- **Conformance.** `net::transport::conformance` checks the binding semantics above on any backend through a `Fabric`. It is the suite DIST-E7-T03, T05 to T08 and QUAL-E8-T04 run.

### 3. Link selection

The solver (RFC-0011) picks one transport per hop and records it in `HopPlan::transport`. This section defines the cost function the solver uses, and what the runtime does when the planned transport is unavailable.

**Cost per frame.** For a candidate transport t on directed edge e, with probed `α_p50`, `α_p99` and `β` (RFC-0010 L1), and `α_eff` where a device-path probe exists (L2):

```text
cost_t(S) = α_p50 + S / β + λ · (α_p99 − α_p50) + z_t(S)
  λ = 1.0   for Wi-Fi or relayed edges in latency mode   (cost model §5.4: cost Wi-Fi at p99)
      0.25  for other edges in latency mode
      0     in throughput mode, where the tail is amortised over batches
  z_t(S) = the backend's software path at the ends: measured as α_eff − α_p50 where L2 exists;
           otherwise the estimate in §Performance, until Warming measures it (RFC-0001 §3.3)
```

**Frame sizes per hop.**

| Frame | Size S |
|---|---|
| Decode | `S_dec = 64 + ext + rows · round_up(streams · hidden · elem, 64)` with rows = 1, or γ + 1 for verify windows |
| Prefill | `S_pf = the same with rows = prefill_chunk` |
| Token | about 128 B |

The hop's planned transport minimises `cost_t(S_dec)` in latency mode. When `cost_t(S_pf)` differs by more than 2× between the two best candidates, the solver may give the hop a second transport for prefill only (`HopPlan::prefill_transport`, added by RFC-0032 §3.3). This lets a Wi-Fi-plus-Ethernet pair, for example, use the faster medium for MiB-scale chunks without moving decode.

**Rank order when unmeasured.**
- Order: `SharedMemory` > `Rdma` > `Quic` (direct) > `Tcp` > `QuicRelay`.
- A transport without a probe result is used only when every measured candidate is unavailable. Such a hop is flagged `MeasureSource::Prior` in the plan's assumptions (RFC-0001 §5.4).

**Runtime fallback.** If the planned transport cannot connect within `link.connect_timeout` (2 s LAN, 10 s WAN), the runner connects on the next candidate by measured cost, marks the stage `Ready { degraded: TransportDemoted }`, and reports `Drift` with the new hop cost. The coordinator decides whether to re-plan (RFC-0001 §7.2). Demotion during operation is in §7.4.

*Note 2026-10-23 (DIST-E7-T11; RFC-0001 §18.31): what `cost_t` is for, its defaults, and what the fallback returns.*
- **Selection, not the objective.** `cost_t` ranks the transports of one edge. It picks `HopPlan::transport` at `S_dec`, and `prefill_transport` at `S_pf`. The solver's hop term in the objective stays RFC-0011 §5's `α^plan + S/β`, which since 2026-10-20 costs a wireless hop at its mean and bounds its tail with C14's stall rate. So λ = 1.0 on Wi-Fi weighs the tail when choosing between that edge's transports. It does not enter the plan's predicted time.
- **Where it lives.** RFC-0001 §5.2's `LinkCaps` and `link_cost(caps, size, objective, λ) -> Nanos` are in `core::resource::link`, so the solver computes `cost_t` without depending on `net`. `Transport::cost` (§2) delegates to it.
- **λ** is `link.p99_weight` (§11).
  - **Defaults, in latency mode:** 1.0 on `WiFi` and `Overlay` links and on `QuicRelay`, and 0.25 otherwise [target]. Overlay links join Wi-Fi because RFC-0011's C14 groups the two as wireless-like hops.
  - **Throughput mode:** 0.
  - **A configured value** is one number that replaces both defaults.
  - **Bounds:** validation keeps it within 0–10 [target: a chosen bound; 1.0 already prices a hop at its p99], so NaN and infinity are refused.
- **Measured means both α and β.** A transport counts as measured only when both its α and its β come from a probe. Startup probes β on the best-ranked transport only (RFC-0010 §3), so until the background probe reaches the others, a transport with α alone ranks with the unmeasured ones, in the rank order above.
- **z before an L2 probe.** Until `alpha_eff` exists, `z_t` is a per-transport constant for a 32 KiB F32 decode frame [estimate]:
  - 25 µs on IP paths: `Quic`, `Tcp` and `QuicRelay`, the upper end of §Performance's ≈15–25 µs with polling;
  - 20 µs on `Rdma`, the upper end of its ≈12–20 µs;
  - 2 µs on `SharedMemory`. §Performance gives no z for it, so it comes from the per-hop table instead: the shared-memory hop of ≈5–10 µs, less its α of ≈1–5 µs and S/β of ≈3 µs, leaves ≈1–2 µs. The same subtraction on the RDMA and 10 GbE rows gives ≈12–22 µs and ≈17–22 µs, which matches their z ranges. The 2 µs assumes host-resident payloads, as the shared-memory row does. Activations that start and end in GPU memory add a device-to-host and a host-to-device copy, plus a doorbell at each end, which comes nearer RDMA's figure, so an L2 probe's `alpha_eff` replaces it on such a path.

  The constant does not scale with S. A prefill-sized frame's CRC is in the measured `alpha_eff` once it exists.
- **What the fallback returns.** The fallback is `net::transport::connect_hop`. It returns a `Demotion { edge: LinkId, from: TransportKind, to: TransportKind }`, plain data in `core` and the data `DegradeDetail::Demoted` (RFC-0033 §7.1) carries, together with the new transport's `cost_t` at `S_dec`.
- **How the runner reports it.**
  - **The message.** It sends the R→D message `EdgeDemoted { demotion, cause, cost }` (RFC-0001 §3.5), whose `cause` is a `DemotionCause` in `core`: `ConnectFailed` for this fallback, and `RepeatedFaults` for §7.4's demotion after 3 faults in 60 s. The one message serves both paths. DIST-E7-T11 adds it, and DIST-E7-T10 sends it from §7.4.
  - **Forwarding.** The daemon forwards it as RFC-0033 §11's `ReportEdgeDemoted { edge, from, to, cause }`, as it forwards `StateChanged` as `ReportRunner`.
  - **No second report.** `EdgeDemoted`'s `cost` is the "`Drift` with the new hop cost" of the runtime-fallback paragraph above. The daemon shows it, and no separate drift report is sent. The coordinator's solver costs the new transport from the resource graph (RFC-0033 §11's `EdgeDemoted` event).
- **The flag.** DIST-E14-T13 sets `Ready { degraded }` with `TRANSPORT_DEMOTED` from the `Demotion`.
- **A re-plan's reason.** If the coordinator re-plans for the demotion, the reason is RFC-0033 §6.1's `ReplanReason::Drift { node, term: DriftTerm }`, with a `Hop { link, transport }` term. DIST-E14-T21 builds `ReplanReason` and `DriftTerm` in `core` (RFC-0033 §6.1's note of 2026-10-23).

### 4. QUIC backend (iroh)

#### 4.1 Endpoints and paths

- Each daemon and each runner owns one iroh endpoint. iroh supplies:
  - QUIC (TLS 1.3) with the Ed25519 endpoint key as identity;
  - a set of direct addresses gathered from interfaces;
  - NAT traversal;
  - relay fallback;
  - path migration when addresses change.
- Dialling hands iroh every known address for the peer, including link-local IPv6 with scope, LAN, Tailscale and static addresses, plus its relay URL. iroh races the paths and keeps the best.
- The backend reports the selected path as `Quic` (direct) or `QuicRelay`, so plans and metrics never mistake a relayed edge for a direct one.
- Address hints come from RFC-0030 (`members.cache`, discovery, `RunnerEndpointOffer`). Syntropy does not publish addresses to any public discovery service (RFC-0030 §6.4).

#### 4.2 Connection and stream mapping (ADR-0308)

- **Two QUIC connections per runner pair**:
  - The **latency connection** carries the decode lane, the prefill lane, ring-return tokens and collectives.
  - The **bulk connection** carries `Replay` and `Bulk` channels: replay, KV migration and tensor-group reads.

  Each has its own congestion controller, so a replay burst's bytes in flight never sit in the decode stream's congestion window. Bulk-connection sends are paced by the p99 rule of §7.5 whenever a latency connection shares the path.
- **One bidirectional stream per channel.** Frames flow in the channel direction and control frames flow back on the same stream. The Activations hop has two channels, the decode lane (`lane = 0`) and the prefill lane (`lane = 1`, RFC-0032 §3.2). Stream priority is `ChannelSpec::priority`, with the decode lane above the prefill lane.
  - Prefill bytes already sent still share the latency connection's congestion window. The head therefore applies RFC-0012's interleave limit, and prefill-lane sends are paced by the §7.5 rule while decode sequences are live.
  - RFC-0061 measures decode p99 during prefill and during replay.
- **Datagrams carry no data-plane frames.** Activations need lossless, ordered delivery, and retransmission would rebuild what streams already give. QUIC datagrams are used only for membership beats on the control connection (RFC-0030 §5.2).

#### 4.3 Transport configuration

| Setting | Value | Why |
|---|---|---|
| Congestion controller | Cubic; BBR selectable with `quic.congestion = bbr` once RFC-0061 shows a gain | standard, and well tested in the QUIC stack iroh uses |
| Initial window on paths with RTT < 2 ms | `quic.lan_initial_window` = 256 KiB | a 1–16 MiB prefill frame would otherwise spend several RTTs in slow start on an idle LAN connection |
| Stream receive window | ≥ 2 × `slot_size` of the channel | two frames in flight per channel |
| Connection receive window | the sum of credited bytes, `Σ recv_slots × slot_size`, over the connection's open channels, recomputed at each channel open; `quic.max_conn_window` (default 256 MiB) only caps it as a sanity bound | connection-level `MAX_DATA` must never be smaller than what credits allow |
| Keep-alive | 500 ms while any channel of a `Ready` instance is open | path loss shows up within the idle timeout |
| Idle timeout | `link.failover_timeout`: 2 s LAN, 10 s WAN | §7.4 |
| MTU discovery | on; initial 1,200 bytes | iroh default behaviour |

*Note 2026-10-23 (DIST-E7-T21, OQ-NET-19): what the pinned iroh reaches.* The workspace pins `iroh = "=1.4.1"`. Its QUIC stack is `noq` 1.4.0 (`noq-proto`, `noq-udp`), and its UDP sockets come from `netwatch` 0.19.3. This is a reading of those crates' source, with paths relative to each crate's root.
- **Where the settings go.** Every row above goes through `iroh::endpoint::QuicTransportConfigBuilder`, set in one of two ways:
  - endpoint-wide, with `Builder::transport_config` (`iroh src/endpoint.rs:678`);
  - per connection: on the dial side with `ConnectOptions::with_transport_config` (`src/endpoint.rs:1813`) through `connect_with_opts` (line 1106), and on the accept side with `create_server_config_builder` (line 1772), `ServerConfigBuilder::set_transport_config` (`src/endpoint/quic.rs:625`) and `Incoming::accept_with` (`src/endpoint/connection.rs:165`).

  Left unset, noq-proto's defaults apply (`noq-proto src/config/transport.rs:575–615`): Cubic, a 1,250,000-byte stream receive window, an unlimited connection window, MTU discovery from 1,200 bytes, and segmentation offload on.
- **Initial congestion window: reachable, but not as a rule on the path's RTT.** `congestion_controller_factory` (`iroh src/endpoint/quic.rs:420`) takes a controller factory. The window itself is set with `noq_proto::congestion::CubicConfig::initial_window` (`noq-proto src/congestion/cubic.rs:293`) or `Bbr3Config::initial_window` (`src/congestion/bbr3/mod.rs:2019`). The default is 12,000 bytes: `14720.clamp(2 × 1200, 10 × 1200)` (`cubic.rs:302`, `congestion.rs:158`). Two things follow.
  1. iroh re-exports only the `Controller`, `ControllerFactory` and `ControllerMetrics` traits (`iroh src/endpoint/quic.rs:69–73`), not the configs. So `net` names `noq-proto` directly, pinned to the exact version iroh pins.
  2. Each path of a connection gets its own controller from the connection's one factory, built before the path has an RTT sample (`noq-proto src/connection/paths.rs:298–301`, `congestion.rs:155`). An initial window of 256 KiB set this way would also apply to a relay path that multipath adds.

  "Paths with RTT < 2 ms" is therefore a controller of `net`'s own:
  - it wraps Cubic, or BBR3 under `quic.congestion = bbr`;
  - each `on_ack` carries the path's `RttEstimator`, and while the path is in slow start and its first RTT sample is below 2 ms, the wrapper reports a window of at least `quic.lan_initial_window`;
  - it returns control to the inner controller at the first congestion event.
- **Stream receive window: reachable, per connection.** `stream_receive_window` (`iroh src/endpoint/quic.rs:225`) sets one value for every stream of a connection when the connection is set up. There is no per-stream setter, so the row's "≥ 2 × `slot_size` of the channel" is met by twice the largest slot of any channel the connection carries. The connection window is `receive_window` (`quic.rs:236`). `Connection::set_receive_window` (`src/endpoint/connection.rs:1139`) changes it on a live connection, as the recompute at each channel open needs.
- **GSO: reachable on Linux and Windows, absent on macOS.**
  - `enable_segmentation_offload` (`iroh src/endpoint/quic.rs:438`) is on by default and can be turned off. When on, noq-proto batches up to the socket's `max_gso_segments` (`noq-proto src/connection/mod.rs:1049–1052`). iroh takes that from each IP transport's socket (`iroh src/socket/transports/ip.rs:237`) and uses the minimum over transports (`src/socket/transports.rs:422–425`).
  - noq-udp probes `UDP_SEGMENT` on Linux (64 segments, `noq-udp src/linux.rs:19–37`) and `UDP_SEND_MSG_SIZE` on Windows (512, `src/windows.rs:708–725`). Every other Unix, macOS included, gets 1 (`src/unix.rs:898–904`).
  - On Apple, batched sends need `UdpSocketState::set_apple_fast_path` (`unix.rs:388`). iroh's default `fast-apple-datapath` feature compiles it in (`iroh Cargo.toml:70`, `74`), but nothing calls it: not iroh 1.4.1, not noq 1.4.0, and not netwatch 0.19.3, which owns the socket (`netwatch src/udp.rs:736`, `801`). iroh does not expose its sockets, so `net` cannot call it either.
  - No upstream issue tracks enabling it. Searches of the iroh and noq trackers on 2026-10-23 found noq #204, which exposed the feature, and none that enables it.
  - iroh #4555 (open) reports that existing connections keep sending GSO batches after noq-udp turns GSO off. That bears on a Linux host whose probe passes but whose driver drops the batches.
- **Keep-alive and idle timeout** are reachable through `keep_alive_interval` (`quic.rs:366`; iroh's default is 5 s) and `max_idle_timeout` (`quic.rs:212`). iroh clamps a per-path idle timeout to 15 s and ignores a per-path keep-alive above 5 s (`quic.rs:492`, `514`). Each path also has its own keep-alive and idle timeout, 5 s and 15 s by default (`quic.rs:157–159`), and iroh abandons a path when its own idle timeout expires. So `net` sets the table's values per path too, through `default_path_keep_alive_interval` (`quic.rs:514`) and `default_path_max_idle_timeout` (`quic.rs:492`), as well as connection-wide, and `link.failover_timeout` is declared with a 15 s maximum (§11), so a larger value is refused at load rather than clamped to 15 s per path with only a log warning (added 2026-10-23 after #1773's and #1781's reviews; the line references were `src/socket.rs:109`, `117`, which hold the constants). The table's values lie inside those bounds.
- **Who does what.** `net`'s endpoint (DIST-E7-T23) sets none of these yet. DIST-E7-T03 sets them, and DIST-E7-T16 measures what the initial window and GSO gain, macOS included.

*Note 2026-10-23 (DIST-E7-T03): what the backend sets, and the connection window's cap.* For the maintainers to confirm the first bullet.
- **`quic.max_conn_window` defaults to 64 MiB.** The table above says 256 MiB; §11 and the registered key (`net::config::QuicSection`) say 64 MiB, and 64 MiB is the default. The row's other rule wins over the cap: the connection window is never smaller than the credited sum. When the sum passes the cap, the backend still raises the window to the sum, and reports it (`QuicTransport::connection_windows`, `over_cap`), because `net` has no logger until RFC-0053's lands. The cap is a sanity bound that shows a large credit, never one that stalls it.
- **Stream receive window: `2 × net.max_frame`.** iroh sets one value for every stream of a connection when it is set up, so twice the largest slot any channel can have meets the row. A channel whose slot is larger than `net.max_frame` is refused at open.
- **Connection receive window.** It starts at room for one channel's control frames, and is raised before each channel's handshake by that channel's `recv_slots × slot_size` plus four control frames (`CONTROL_RECEIVES × MAX_CONTROL_FRAME`).
- **The LAN window's first sample.** noq-proto calls `on_ack` for each acknowledged packet before it updates the round-trip estimate with that acknowledgement, so the first `on_ack` sees the configured initial round trip. The wrapper takes the first estimate that differs from it as the path's first sample.
- **Keep-alive and idle timeout.** These follow the table, connection-wide and per path. A dialled connection takes `link.*`'s LAN defaults when every address the peer gave is loopback, link-local or private, and the WAN defaults otherwise, a relay-only peer among them. An accepted connection takes the WAN idle timeout; QUIC uses the smaller of the two sides', so a LAN dialler brings its 2 s.
- **Windows.** The runtime cannot wait on a `syntropy-spsc` ring's event handle, so a parked channel's task re-checks its rings every 1 ms. While live sequences exist it busy-polls, as on other platforms (§4.4). DIST-E7-T24 adds the wait bridge.
- **Early streams.** A stream whose `Hello` names a channel the acceptor has not yet called `accept` for is held for `QuicSettings::early_hold`: half the shorter of the LAN and WAN connect timeouts, 1 s with the defaults [target]. It is then refused with `NotANeighbour`. A connector on the same settings waits for `HelloAck` for its link's connect timeout, at least twice the hold, so it receives that refusal, and learns that the two plans disagree, rather than timing out. So `early_hold` also bounds how late an acceptor may call `accept`: within `early_hold` of the peer's `Hello`. At most 16 early streams are held from one peer key [target], a quarter of the 64 held in all [target], so one peer cannot fill the table for the others. A stream over either is closed without a `HelloAck`. The connector sees a link failure (`Link`), which is retried, and not `NotANeighbour`, which RFC-0001 §10 never retries, since nothing about the plan was wrong. A second stream for a held key replaces the first when it comes from the same peer, as a reconnect would, and is refused with `NotANeighbour` when it comes from another. An `accept` whose caller stops waiting leaves nothing registered, so the caller may call it again. The TCP fallback holds a connection the same way (§5.3).

#### 4.4 Hot-path integration

QUIC is asynchronous. The runner's `net` domain is a single-threaded tokio runtime on its own reserved core (RFC-0001 §8.1, ADR-0012). Per channel:

```text
dispatcher thread ──OutFrame──▶ SPSC ring ──▶ net task: stream.write(header ‖ ext ‖ payload)
                                                     (TLS encrypts into packet buffers: the one send-side pass)
net task: stream.read_chunk() ──decrypt──▶ copy into a registered receive slot (§8) ──InFrame──▶ SPSC ring ──▶ dispatcher
                                            (the one receive-side host copy)
credit frames: produced by the receiving dispatcher when it drops an InFrame, sent by the net task
```

- **Polling**. While the runner has a `Ready` instance with live sequences, the net task busy-polls its outbound SPSC rings on its reserved core, under ADR-0012's spin budget: 50 µs, or 10 µs on battery, then it parks. The dispatcher pre-wakes it at the predicted send time minus `sched.prewake_us`, the same pre-wake RFC-0012 §3.3 uses for dispatchers, so it is spinning, not parked, when the frame arrives. A decode frame's hand-off then costs a cache-line transfer, not a cold wake.
- When idle, the rings use eventfd or event-handle signalling into tokio (ADR-0012 item 4).
- The dispatcher never awaits. It polls its inbound ring between launches. The net task never touches device memory.
- The H2D copy of a received activation into its device buffer is issued by the dispatcher (RFC-0012).

#### 4.5 Sequence channels

- The ingress daemon opens a `Sequence` channel to each head runner it routes to. The head runner accepts `Sequence` channels from any member daemon whose admission certificate verifies (§6.9, `cred_kind = AdmissionCert`).
- The tail runner opens a `Tokens` channel to each ingress node named in a sequence's metadata, lazily on the first such sequence, and keeps it for the epoch.
- Both channels use the daemon's iroh endpoint on the daemon side.

### 5. Other backends: shared memory, RDMA, TCP

#### 5.1 Shared memory (co-located nodes)

This backend is used only between nodes whose `HostFingerprint` matches and whose co-location has been confirmed (RFC-0010 §1.4). Typical cases are test clusters on one machine and a container next to a host daemon. A single node never needs it: one runner hosts all of its node's stages (RFC-0001 §3.2).

**Segment.** There is one segment per channel, created by the **receiver**:
- Linux: `shm_open("/syntropy-<128-bit random hex>", O_CREAT | O_EXCL, 0600)`.
- macOS: the same call, with the name truncated to the platform limit as `/sy-<96-bit base32>`.
- Windows: `CreateFileMapping` named `Local\syntropy-<random>` with an owner-only DACL.

Layout:

```text
page 0:   magic "SYNS" | version u16 | slot_size u32 | n_slots u16 | nonce [16] | creator pid u32 | pad
          producer index u64 (own cache line) | consumer index u64 (own cache line)
          producer wake word u32 (own cache line) | consumer wake word u32 (own cache line)
page 1..: n_slots × slot_size, each slot 64-byte aligned, holding a full frame (§6)
```

**Pairing.**
1. The receiver sends the name and the nonce in `ShmOffer`, inside the authenticated `RunnerEndpointOffer` (RFC-0030 §8.3).
2. The sender maps the segment, reads the nonce, and echoes it in the channel's `Hello` over QUIC.
3. The receiver checks the nonce, which confirms co-location, then unlinks the name, so the mapping lives only as long as the two processes.

If the open fails because the nodes run as different OS users, the edge falls back to QUIC over loopback (ADR-0314).

**Signalling.**
- Indices are advanced with release/acquire atomics.
- The consumer spins for the hot-thread budget (ADR-0012: 50 µs, or 10 µs on battery), then sleeps on the wake word:
  - Linux: `futex` on the shared mapping, non-private;
  - macOS: `os_sync_wait_on_address` with the shared flag;
  - Windows: a named event, because `WaitOnAddress` does not work across processes.
- The producer wakes the consumer only if the consumer's "sleeping" bit is set.

**Copies.** Frames are written straight into the slot. The receiver uses the slot as its receive buffer, so the payload is written once (RFC-0001 §9.4).

#### 5.2 RDMA (ADR-0309)

Two variants share one implementation:

| | Linux verbs | macOS RDMA over Thunderbolt 5 |
|---|---|---|
| Library | `libibverbs.so.1`, loaded with `dlopen` at startup (ADR-0011) | the system RDMA library behind `infiniband/verbs.h` (macOS 26.2 and later), loaded dynamically so the binary still runs on older macOS |
| Fabrics | RoCEv2 (only on `trusted_fabric` interfaces, ADR-0314), InfiniBand | one TB5 cable per pair; no switch, no routing |
| Queue pair | RC (reliable connected) | UC (unreliable connected); at most 10 per device; 4,095 outstanding work requests; messages up to 16,773,120 B (TN3205) |
| Operations used | SEND/RECV only | SEND/RECV (the only operations offered) |
| Reliability | hardware, plus Syntropy's (§7) after a QP reset | Syntropy's only (§7): sequence numbers, CRC32C, cumulative acks, NACKs, retransmission from retention |

**Per channel.** One QP, one send CQ and one receive CQ.

**QP budget**. Thunderbolt 5 allows 10 UC QPs per device. That budget is shared by every instance, every overlapping epoch (make-before-break, `Relink`) and every recovery lane on the node.
- The daemon's device ledger tracks QPs per RDMA device as a capacity, `RdmaPortInfo::max_qp`, reported to the solver (RFC-0010, RFC-0011).
- Runners take QPs from their ledger grant.
- A channel that would exceed the grant opens on the next transport by measured cost, normally QUIC, with `TransportDemoted`. A plan never fails for lack of QPs.
- The solver counts QPs per plan: decode lane, prefill lane, ring-return tokens, collectives. It reserves headroom for one overlapping epoch and two replay lanes.
- ADR-0011 item 6's "at most 4 per neighbour" becomes a per-neighbour cap *within* the device budget, as ADR-0011's 2026-10-19 revision records.

**Receive side.**
- The receiver posts `recv_slots` receive work requests, each pointing at one registered slot of `slot_size`.
- The credit it grants equals the number posted.
- A sender never exceeds its credit (§7.1). The TB5 requirement that the receiver has a receive posted for every message sent is therefore met by construction.

**Send side.** The sender posts one SEND with a two-entry scatter-gather list: the header area (base plus extension) and the payload slot. The payload is never copied to join it to its header. Frames up to `inline_max` bytes, such as credit, token and control frames, use inline sends where the device supports them.

**Memory registration.**
- Each buffer pool (§8) is registered once as one large memory region.
- A driver limit of about 100 memory regions has been reported for Thunderbolt RDMA. It is unconfirmed (`hardware-and-performance-model.md` §2.5). Syntropy uses fewer than 8 memory regions per runner whatever the instance count.
- **GPUDirect RDMA** on Linux, NVIDIA or AMD with a capable NIC: device memory is exported as a dma-buf and registered with `ibv_reg_dmabuf_mr`. It is used only when RFC-0010's device-path probe confirms a working direct path. Decode frames then go from GPU to wire to GPU with no host hop.
- **Unified memory**, on Apple silicon and APUs: the registered pinned slot is also GPU-visible through `KernelProvider::import`, so a Metal stage consumes RDMA frames with zero copies.

**Pairing.** Over the authenticated `RunnerEndpointOffer` (RFC-0030 §8.3), the two runners exchange:

```rust
pub struct RdmaPairing {
    pub kind: RdmaKind,
    pub device: String,          // local RDMA device name (diagnostics)
    pub gid: [u8; 16],           // GID for the interface's address (RoCEv2: IPv4-mapped or IPv6 GID)
    pub qpn: u32,
    pub psn: u32,                // random initial packet sequence number
    pub mtu: u16,
    pub channel_token: u64,      // random; carried in the RDMA Hello frame (§6.9)
}
```

QPs are moved INIT → RTR → RTS with the peer's GID, QPN and PSN. The first frame on the QP is a `Hello` (§6.9) carrying the credential and the channel token. Until it verifies, the receiver releases nothing to the dispatcher.

**Completion handling.**
- CQs are busy-polled by the RDMA net poller thread (RFC-0001 §8.2) or by the dispatcher itself when no poller exists.
- After the hot-thread spin budget with no work, the poller arms `ibv_req_notify_cq` and blocks on the completion channel's fd. No event channel is used on the hot path.

**Errors** are handled in §7.4. In outline: a QP in the error state is destroyed, recreated and re-paired over QUIC with a fresh QPN and PSN, and unacked frames are resent, all without restarting the runner.

**Diagnostics.**
- "RDMA hardware present but not enabled": a TB5 port exists, `rdma_ctl` is disabled, and no RDMA device is listed.
- "macOS build mismatch": both nodes report TB5 RDMA and a cable joins them (`IfaceInfo::tb_peer`, RFC-0030 §7.1), yet pairing or the probe fails, and their OS build strings differ.

Both appear in the dashboard and `doctor` (ADR-0011 work item for RFC-0051).

#### 5.3 TCP fallback

This backend is used when UDP is blocked, which shows up as QUIC dials to a peer failing while TCP connects.
- TLS 1.3 (rustls) authenticates both ends with the same Ed25519 key form iroh uses, verified against the expected node ID or runner key.
- There is **one TCP connection per channel**, so channels never share a byte stream and suffer head-of-line blocking.
- `TCP_NODELAY` is on. `SO_SNDBUF` and `SO_RCVBUF` are at least 2 × `slot_size`. On Linux, `TCP_QUICKACK` is re-armed after each read.
- Framing is identical to QUIC streams (§6). Credits and acks are Syntropy's, as on every backend.

*Note 2026-10-23 (DIST-E7-T05): the TCP fallback as built.* `syntropy_net::transport::tcp` is the backend, and the points this section leaves open are fixed as follows:
- **Keys.** Each end presents its key as an Ed25519 raw public key (RFC 7250), which is iroh's form, and signs the handshake with it. Only the Ed25519 signature scheme and TLS 1.3 are offered. The connector accepts only `RemoteEndpoint::key`. The acceptor accepts only the keys its owner allowed, the plan's neighbours, which the runner keeps current. The ALPN is `syntropy/data/1` (§1).
- **Channel opening.** The connector dials each of the endpoint's addresses in turn and sends `Hello` over TLS. The acceptor's listener checks the `Hello`'s protocol major and its credential on arrival, and refuses a connection that fails either at once. It holds the others for the `accept` of their channel key, for `TcpConfig::early_hold`, half the handshake timeout, and then refuses them with `NotANeighbour`, as the QUIC backend does (§4.3). At most 16 are held from one peer key [target], a quarter of the 64 held in all [target], so one allowed peer cannot fill the table for the others. A connection over either is closed without a `HelloAck`. The connector sees a link failure (`Link`), which is retried, and not `NotANeighbour`, which RFC-0001 §10 never retries, since nothing about the plan was wrong. A second connection for a held key replaces the first when it comes from the same peer, as a reconnect would, and is refused with `NotANeighbour` when it comes from another. `accept` checks the peer's key against the allowed set again, and the whole `Hello` with its credential, since either may have stopped passing while the connection waited, then answers `HelloAck` (§6.9). The credentials and their check come from the owner's `ChannelAuth`. Runner credentials and their verification are DIST-E7-T04's.
- **The driver.** The channel is `transport::stream`'s, the same driver a QUIC stream uses (§4.4). One task per channel on the `net` domain runs the `ChannelCore` over the TLS stream. The dispatcher's side is SPSC rings and atomics, with no lock. A frame's body is read straight into a receive slot.
- **Buffers.** The rule is: `SO_SNDBUF` and `SO_RCVBUF` are at least 2 × `slot_size` where the OS allows it, and otherwise autotuned. An unprivileged process gets at most the kernel's cap: on Linux `net.core.wmem_max` and `rmem_max`, 212 992 bytes by default [measured, `/proc/sys/net/core` on a WSL2 Linux host]. A larger request is clamped to the cap, and setting either buffer also turns off Linux's autotuning for the socket. That autotuning grows to `tcp_wmem` and `tcp_rmem`'s maximum, 4 MiB and 6 MiB on the same host [measured, `/proc/sys/net/ipv4`]. So the backend reads the two caps once and requests 2 × `slot_size` only where it fits. Here `slot_size` is the larger of the channel's two sides'. The connector sets its receive buffer before `connect`, from its own slot, since the SYN fixes the window scale, and sets its send buffer once `HelloAck` gives the peer's slot. Where it does not, the backend leaves the socket to autotuning and counts it in `TcpStats::buffers_autotuned`. Raising the sysctls restores the literal rule (RFC-0052 §2.4, note of 2026-10-23).

*Revised 2026-10-23 (#1835): the backend moved onto `transport::stream`.* "Channel opening" and "The driver" above describe it since then. As DIST-E7-T05 built it, each channel had a reading task and a writing task that shared the `ChannelCore` with the channel's methods under a mutex, held only while the core ran. The listener checked the credential on arrival and held a connection until `accept` or the handshake timeout, with no cap on how many it held.

#### 5.4 Relayed QUIC

This is the same backend as §4 over an iroh relay. Relayed edges are costed with their measured α and p99. In latency mode the solver therefore keeps them off the ring unless nothing else connects the nodes. They remain the normal path for sequence channels to off-LAN ingress nodes (`00-vision-and-scope.md` S3).

- **Which relays.** The built-in set is a hosted shared relay service, delivered as the signed `relays.json` TUF target; `custom` and `disabled` remain (RFC-0030 §6.4).
- **No bulk weight transfer over a relay.** A blob-plane request of kind `CHUNK` or `TENSOR_GROUP` (§6.10) is never sent over, or served on, a path the backend reports as `QuicRelay`; the server closes it with `NotFound` and the requester picks another source (RFC-0041). A relayed peer would usually lose to the origin in RFC-0041's source ranking anyway, because hosted relays are rate-limited, but the rule makes sure one model pull can never load the relay with a model's worth of bytes. Control traffic, `BY_DIGEST` snapshot and manifest bodies, and sequence channels may use a relay.

### 6. Data-plane wire format

All integers are little-endian. All offsets are in bytes from the start of the frame. Reserved fields are zero on send and ignored on receive. A frame is:

```text
| base header (64 B) | extension (ext_len B, padded so 64 + ext_len is a multiple of 64) | payload (payload_len B) |
```

The payload therefore starts 64-byte aligned within its slot (RFC-0001 §5.8). Slots are 64-byte aligned (§8). Multi-byte fields are naturally aligned within the header and extensions.

#### 6.1 Base header

```text
off  len  field         notes
0    4    magic         "SYNF" (0x53 0x59 0x4E 0x46)
4    1    wire_minor    minor version of this layout (major is in the ALPN or Hello)
5    1    ftype         FrameType, §6.2
6    2    flags         bit0 HAS_TRACE (a TRACE TLV is present), bit1 reserved (was RETRANSMIT; a resend is
                        byte-identical to the original, §7.3), bit2 ENCRYPTED
                        (RDMA AEAD, ADR-0314), bit3 FIRST_OF_SEQ (first frame of a sequence on this edge),
                        bit4 KEY_PHASE (AEAD key phase, flips on re-keying), bits 5–15 reserved
8    8    instance      InstanceId; 0 for blob-plane frames
16   8    epoch_term    PlanEpoch::term
24   8    epoch_seq     PlanEpoch::seq
32   8    step          per ActivationFrame semantics (RFC-0032 §3.2); on sealed control frames, `ctl_seq`;
                        0 where unused
40   8    chan_seq      data frames: per-channel sequence number, starting at 1; control frames (Hello, HelloAck,
                        Credit, Ping, Close): 0
48   4    ext_len       bytes of extension, including its padding; always a multiple of 64
52   4    payload_len   bytes; frames larger than the channel's slot_size are a protocol error
56   4    payload_crc   CRC32C (Castagnoli) of the payload; 0 when payload_len = 0
60   4    header_crc    CRC32C of bytes 0..60 followed by the extension bytes
```

- Receivers check `magic`, `wire_minor ≤ negotiated`, `header_crc`, `instance` and epoch against the channel key, `chan_seq` continuity, and then `payload_crc`. Only then do they release the frame (§2).
- When `ENCRYPTED` is set, the payload and the extension past its first 16 bytes are AEAD-sealed (ADR-0314, RFC-0034 §9.3), with these rules:
  - The 16-byte AEAD tag is a trailer after the last payload row, counted in `payload_len` and outside the row stride.
  - The associated data is bytes 0..56 of the base header, followed by the extension's first 16 bytes. Those 16 bytes stay in plaintext for routing, but they are authenticated. All of it is fixed for the frame's lifetime (the same rule as RFC-0034 §9.3).
  - **Nonce spaces**. The 96-bit nonce is `space u8, three zero bytes, counter u64`:
    - data frames use space 0 with `chan_seq`;
    - control frames, whose `chan_seq` is 0, use space 1 with `ctl_seq`, a per-direction count of sealed control frames that starts at 1. `ctl_seq` is carried in `step`, which control frames do not otherwise use, so it is covered by the associated data.
    - A receiver drops a control frame whose `ctl_seq` is not above the highest it has accepted.
  - On an encrypted channel every frame is sealed, control frames included, and an unsealed frame is dropped.
  - A resend retransmits the stored sealed bytes unchanged, so no `(key, nonce)` pair is ever used twice.
  - **Re-keying.** A re-pair exports new keys and flips flags bit4 `KEY_PHASE`. Retained frames keep their original phase and sealed bytes. The receiver keeps the previous key until it has acknowledged every `chan_seq` sealed under it, and `ctl_seq` restarts at 1 under each new key.
  - The CRCs cover the ciphertext, so corruption is caught before decryption is attempted.

#### 6.2 Frame types

| ftype | Name | Channels | Extension | Payload |
|---|---|---|---|---|
| 0x01 | `Activation` | Activations, Replay | §6.3 | hidden states |
| 0x02 | `Token` | Tokens | §6.5 (8 B) | token entries |
| 0x03 | `SeqControl` | Sequence | §6.6 | CBOR body |
| 0x04 | `Credit` | all, reverse direction | §6.7 | none |
| 0x05 | `KvChunk` | Replay, Bulk | §6.11 | KV pages |
| 0x06 | `BlobData` | blob plane | §6.10 | chunk bytes |
| 0x07 | `BlobRequest` | blob plane | §6.10 | none |
| 0x08 | `Collective` | Collective | §6.8 | partial sums or gathered slices |
| 0x09 | `StepDescriptor` | Collective (primary → members) | the `Activation` extension, §6.3 | the stage's input hidden states |
| 0x0A | `Hello` | first frame of every channel | §6.9 | credential |
| 0x0B | `HelloAck` | reply to `Hello` | §6.9 | the acceptor's credential |
| 0x0C | `Close` | any | §6.9 | UTF-8 reason (≤ 256 B) |
| 0x0D | `Recover` | Activations, Tokens, during recovery (RFC-0033 §4) | §6.12 | entries |
| 0x0E | `EpDispatch` | Collective (primary → expert owner) | §6.8 | routed hidden states |
| 0x0F | `EpCombine` | Collective (owner → primary) | §6.8 | weighted expert outputs |
| 0x10 | `Ping` | any | 16 B: `origin_ns u64, reply u8, pad` | none |
| 0x11 | `TapIds` | Tap (parity driver → tail or drafting head) | §6.13 | teacher tokens and reference ids |
| 0x12 | `EmbeddingResult` | Tokens (tail → ingress) | §6.14 | pooled vectors |

An unknown `ftype` within a negotiated minor is a protocol error: the receiver sends `Close { UnknownFrameType }`. New types are only ever sent when the peer advertised the capability that introduced them (§10).

#### 6.3 `Activation` extension

```text
off  len  field
0    2    from_stage
2    2    to_stage
4    1    kind          1 Prefill | 2 Decode | 3 Verify | 4 Replay | 5 Canary
5    1    dtype         WireDType, §6.4
6    1    streams       residual streams (1, or more for hyper-connection models)
7    1    reserved
8    4    hidden
12   4    n_rows        token positions carried in the payload
16   2    n_spans
18   2    part_index    0-based; a chunk larger than max_frame is split at row boundaries
20   2    part_count    ≥ 1
22   2    reserved
24   4    kind_arg      Prefill: chunk index; Verify: window (γ+1 or tree nodes); Replay: replay id; else 0
28   4    kind_flags    Prefill: bit0 LAST_CHUNK; Verify: bit0 TREE; Replay: bit0 LAST, bit1 FORWARD (RFC-0033 §4.2)
32   4    tree_len      bytes of the tree area
36   4    side_len      bytes of the side-channel area
40   8    payload_rows_crc  reserved (0); room for per-row CRCs in a later minor
48   16   reserved
64   …    spans: n_spans × 32 B
          tree area: tree_len B
          side-channel area: side_len B
          padding to the 64-byte boundary
```

**Span** (32 bytes, one per sequence segment in the frame, in row order):

```text
0   8   seq.ingress   SequenceId (RFC-0001 §5.9)
8   4   seq.boot
12  4   seq.n
16  4   first_position
20  4   n_positions   rows belonging to this span
24  4   row_offset    index of the span's first row in the payload
28  2   tree_offset   index of the span's first node in the tree area, 0xFFFF if not a tree
30  2   tree_nodes
```

**Tree area** (verify windows with `TREE`): 4 bytes per node, `parent: u16` (0xFFFF for the root, which is the last accepted token), `depth: u8`, `flags: u8`. The bit0 `PROMPT_LOOKUP` flag marks a node from the prompt-lookup drafter. Rows of a tree span are in node order. Attention masks are derived from parents (RFC-0012).

**Side-channel area.** A sequence of TLVs, each 8-byte aligned:

```text
0  2  type    1 TOKEN_IDS | 2 ROUTER_HINT | 3 SEQ_META | 4 SEQ_CTRL | 5 TRACE | 6 TRUNCATE | 7 DRAFT_CONF
              | 8 KV_HEADROOM | 9 PAGE_KEYS | 10 PREFIX_EVICT | 11 PREFIX_DROPPED | 12 PREFIX_PRESSURE
              | 13 PREFIX_MISS | 14 RESTART | 15 SESSION_SAVED
2  2  span    index of the span it applies to; 0xFFFF for all spans
4  4  len     value bytes (excluding padding)
8  …  value, padded to 8
```

| TLV | Value | Who needs it |
|---|---|---|
| `TOKEN_IDS` | u32 per row of the span | stages that hold `RowLookup` tables, such as per-layer embedding or n-gram tables, and the tail for repetition penalties |
| `ROUTER_HINT` | `layer: u16, k: u8, pad: u8`, then per row `k × u16` predicted experts | the next stage's expert prefetch (RFC-0023) |
| `SEQ_META` | CBOR `SequenceMeta`: `SequenceOpen` without the prompt tokens (RFC-0032 §3.6) | every stage on the sequence's first frame (`FIRST_OF_SEQ`) |
| `SEQ_CTRL` | `code: u8` (1 Cancel, 2 Release, 3 ReleasePersist; 4 and 5 reserved, since Pause and Resume stop at the head, RFC-0032 §3.1), `pad[7]`, `session: [u8; 16]` for ReleasePersist | `Cancel` and `Release` riding the ring (RFC-0001 §6.2). Receivers ignore codes 4 and 5 |
| `TRACE` | 26 bytes of W3C trace context: version, trace-id, span-id, flags | sampled frames (RFC-0001 §13) |
| `TRUNCATE` | `to: u32` | explicit KV truncation where implicit truncation does not apply (RFC-0032 §3.5) |
| `DRAFT_CONF` | f16 per tree node | the tail's branch pruning (ADR-0017 item 5) |
| `KV_HEADROOM` | `n: u8, pad[7]`, then `n` entries of `{ stage: u16, flags: u16, tokens: u32 × 3 }` (16 B). `tokens` is RFC-0022's `KvHeadroom { tokens: [u32; 3] }` as the stage's `KvStore::headroom()` reports it: the tokens the stage can still admit per tier, in the order `Vram`/`Unified`, `HostPinned`, `Nvme`, already the minimum over its token-indexed KV classes (RFC-0022 §11). Flag bit0 is `STALE` (the entry was copied forward without being refreshed) | the head's binding admission decision (RFC-0012 §10.1). Each stage replaces its own entry and keeps the others, and the tail returns the full list on the ring-return `Token` frame (RFC-0032 §3.10) |
| `PAGE_KEYS` | `first_page: u32, n: u16, pad[2]`, then `n × 32`-byte page keys (RFC-0022 §7.4) | every stage, which files its copy of each sealed page under the key |
| `PREFIX_EVICT` | `n: u16, pad[6]`, then `n × 32`-byte keys | every stage drops those index pages (RFC-0022 §7.5 rule 1) |
| `PREFIX_DROPPED` | `stage: u16, n: u16, pad[4]`, then `n × 32`-byte keys | relayed to the head (RFC-0022 §7.5 rule 3) |
| `PREFIX_PRESSURE` | `stage: u16, pad[2], pages: u32` | relayed to the head, which picks evictions (rule 1) |
| `PREFIX_MISS` | `stage: u16, pad[2], have_tokens: u32` (span = the sequence) | on the sequence's first frame; the tail converts it into `RESTART` (rule 4) |
| `RESTART` | `from: u32, pad[4]` (span = the sequence) | ring-return `Token` frames only: the head re-issues prefill from `from` (rule 4) |
| `SESSION_SAVED` | `session: [u8; 16], stage: u16, pad[6], manifest: [u8; 32]` (BLAKE3 of the stage's checkpoint manifest), span = 0xFFFF | `SessionSaved { session, stage, manifest }` (RFC-0022 §8.1–§8.2): a stage that has written its session checkpoint and its manifest reports it; relayed to the head, which submits `PinSession` once every stage has reported (RFC-0032 §3.10) |

**Payload.** `n_rows` rows, in row order. Each row is `streams × hidden` elements of `dtype`, laid out `[stream][hidden]`, with the row stride rounded up to 64 bytes. `payload_len = n_rows · stride` for plain dtypes.

**Revision note, 2026-10-22 (DIST-E8-T02): what each part carries, and tree parents.**
- **Parts.** A sender splits a frame larger than `max_frame` into the fewest parts. Each part holds as many rows as fit with its own extension, which counts against the part's row limit (§8). Each part is a whole `Activation` frame for its rows:
  - Its fixed fields are the chunk's, except `n_rows`, `n_spans`, `part_index`, `part_count`, `tree_len` and `side_len`.
  - It carries the spans of its rows, cut at the part's edges. `row_offset` counts from the part's first row, and `first_position` moves on by the rows the cut left in earlier parts.
  - `TOKEN_IDS` and `ROUTER_HINT` are cut with their rows. Every other item travels once: in the first part that carries its span, or in part 0 when it applies to every span.
  - A span without rows, which an empty frame uses to name the sequence of its `SEQ_CTRL` (RFC-0032 §3.1), goes in the part that holds its `row_offset`, or in the last part when that offset is `n_rows`.
  - The receiver lands the parts at consecutive offsets of the stage's input (§8). A verify frame with `TREE` is never split: its parents and `DRAFT_CONF` are per span, and its window is small (RFC-0032 §3.5).
- **Tree parents.** A node's `parent` is the index of an earlier node of the same span, counted from the span's first node, or 0xFFFF for the root. A receiver refuses any other value, so the parents always form a tree.

#### 6.4 Wire dtypes

| Code | `WireDType` | Exact mode | Container |
|---|---|---|---|
| 1 | `F32` | yes | IEEE binary32 |
| 2 | `BF16` | only where the reference holds BF16 at that boundary (ADR-0010 item 5, ADR-0019 item 5) | bfloat16 |
| 3 | `F16` | no; `WIRE_COMPRESS` | IEEE binary16 |
| 4 | `Q8B32` | no; `WIRE_COMPRESS` | per 32 elements: f16 scale, then 32 × i8 (34 B); row stride rounded to 64 |
| 5 | `F8E4M3B32` | no; `WIRE_COMPRESS` | per 32 elements: f16 scale, then 32 × E4M3 (34 B) |

The numerics of codes 3–5, meaning scale choice, rounding and the families each is approved for, belong to RFC-0042. A receiver rejects a frame whose `dtype` differs from the channel's `ChannelSpec::wire`. A plan change is the only way to change it.

#### 6.5 `Token` frame

Tokens travel tail → head (ring return, all of the step's sequences) and tail → ingress (only the sequences whose ingress is that node).
- **Extension.** `n_entries: u16`, `flags: u16` (bit0 `RING_RETURN`, bit1 `TO_INGRESS`), `side_len: u32`, `part_index: u16`, `part_count: u16`, `reserved: u32`, followed at offset 16 by a side-channel area of `side_len` bytes in the TLV encoding of §6.3, then padding.
- **Parts**. A step's entries that exceed the channel's slot, such as large logprob batches or parity tap blocks, are split across parts at entry boundaries. A single entry larger than a slot splits its tap block across parts. Receivers act on a step's tokens only when all its parts have arrived. Only ring-return frames carry TLVs, and only `KV_HEADROOM` (the accumulated per-stage list plus the tail's own entry), `PREFIX_DROPPED`, `PREFIX_PRESSURE`, `RESTART` and `SESSION_SAVED` are allowed there (RFC-0032 §3.10).
- **Payload.** `n_entries` entries, each 8-byte aligned:

```text
0   16  seq               SequenceId
16  4   first_position
20  2   n_tokens          accepted tokens: 1 for plain decode, 1..=γ+1 after a verify window
22  1   finish            0 none | 1 Stop | 2 Length | 3 TokenStop | 4 Cancelled | 5 Error
23  1   logprob_k         top-k alternatives per token; 0 = no logprobs
24  4   approx            ApproxKnobs bits that touched these tokens (RFC-0001 §11.3); 0 in exact mode
28  2   error_domain      when finish = Error
30  2   error_code
32  2   n_draft           drafted tokens for the next window (tail-side drafters, RFC-0032 §3.5)
34  2   draft_nodes       tree nodes; = n_draft for a chain
36  4   tap_len           bytes of the parity tap block; 0 unless the sequence is a parity sequence (RFC-0060 §5.5)
40  8   rng_counter       u64 sampler draw counter after these tokens (RFC-0032 §3.7); lets a replacement tail continue
48  …   tokens:     u32 × n_tokens
        draft:      u32 × n_draft
        draft tree: (parent u16, depth u8, flags u8) × draft_nodes   (absent for a chain)
        logprobs:   per token: chosen f32, then logprob_k × (id u32, logprob f32)
        tap block:  tap_len bytes (below)
        padding to 8
```

- *Note 2026-10-23 (DIST-E1-T14, as built; review-1493's N2 on #1780):* `core`'s `TokenEntry::rng_counter` has no serde default, though the item below calls it an addition with a default. A missing counter defaulting to 0 would restart a resumed sequence's random stream (RFC-0032 §3.7), so a later step's tokens would differ from the reference, which breaks exact mode (D3). The binary entry always carries it at offset 40, and `core`'s type is new, so no older encoding exists to need a default. A serde encoding without it is refused rather than read as 0. `draft`, `draft_tree` and `tap` keep their defaults.
- *Note 2026-10-23:* `logprob_k = 0` means no logprobs, so the entry has no form for the chosen tokens' logprobs without alternatives. OQ-NET-35 tracks it, with `logprob_k = 0x80` recommended behind capability bit 16 `CHOSEN_LOGPROBS` (§10): a new value of an existing field is not additive, so it is sent only on channels whose two ends advertise the bit, and ingress asks for it only when both of the tail's channels the entry travels, to the ingress and to the head, have it (RFC-0050 §3.1's note). DIST-E8-T18 settles it.
- `TokenEntry::draft`, `rng_counter` and `tap` are additions to RFC-0001's `TokenEntry`, made under the extension rules as fields with defaults. `tap` is RFC-0060 amendment A4's `TokenEntry::tap: Option<LogitTap>`.
- The draft travels only on the ring-return frame. The ingress copy omits it (`n_draft = 0`).
- The tap block travels only on the ingress copy, which goes to the parity driver.

**Revision note, 2026-10-22 (DIST-E8-T03): chains and trees, and what each copy carries.**
- **Chain or tree.** A tree has one node per drafted token, so `draft_nodes` equals `n_draft` for a tree as for a chain, and the entry cannot say which it holds. The extension's flags bit2 `DRAFT_TREE` says it for the frame:
  - When it is set, every entry with a draft carries its tree. A chain in such a frame is written as a tree whose nodes each have the one before as parent.
  - When it is clear, no entry carries a tree, and each draft is a chain.
  - A node's `parent` is an earlier node of the same draft, or 0xFFFF for the last accepted token, as in §6.3. Its `depth` is 1 under the last accepted token and its parent's plus one otherwise. A receiver refuses any other `parent` or `depth`, and an entry whose `draft_nodes` is not `n_draft`.
  - Bits 3–15 are reserved.
- **Copies.** A frame sets exactly one of `RING_RETURN` and `TO_INGRESS`. A receiver refuses a ring-return frame that carries a tap block, and an ingress frame that carries a draft. Of the `Activation` items (types 1–7), only `TRACE` may travel on a `Token` frame, so a receiver refuses the rest.
- **Trace context.** `TRACE` (§6.3) joins the types a `Token` frame may carry, on both copies, so the return path from the tail to the head and to ingress appears as a hop span (RFC-0053 §3.2, RFC-0001 §13).
  - The tail writes one `TRACE` per sampled entry, with `span` the entry's index in the frame, and sets flags bit0 `HAS_TRACE`. A receiver refuses a `TRACE` that names no entry (`span` 0xFFFF), which would carry one sequence's trace context to every ingress, and a second `TRACE` for the same entry.
  - An ingress copy carries only its own entries' `TRACE`, indexed among them, and no other item. That amends "only ring-return frames carry TLVs" above.
  - A receiver refuses a frame whose `HAS_TRACE` does not match its `TRACE` items.

**Tap block** (`LogitTap`, RFC-0060 §5.5 `TapMode::Gather { k, tail }` and RFC-0050 §13.6; RFC-0012 owns its semantics, RFC-0060 §5.3–§5.4 the arithmetic that consumes it):

```text
0   1   mode        1 Gather | 2 FullLocal
1   1   reserved
2   2   n_pos       tapped positions in this entry (every position of a verify window or prefill chunk)
4   2   k           reference top-K ids per position, the set T (Gather); 0 for FullLocal
6   2   tail        tail-sample ids per position, the set S drawn from V∖T (Gather); 0 for FullLocal
8   …   per position, 8-aligned:
  0   4   position
  4   4   argmax           candidate's argmax token id (ties to the lower id)
  8   4   margin           f32, candidate's top-1 minus top-2 logit
  12  4   reserved
  16  8   lse              f64, candidate's logsumexp over the full vocabulary, computed in F64
  24  8   lse_complement   f64, candidate's logsumexp over V∖T, computed in F64 directly, never as
                           1 − Σ p (Gather); 0 for FullLocal, where the driver computes it from the row
  32  …   Gather:    k × f32 candidate logits at T, then tail × f32 candidate logits at S, each in the
                     order the driver sent the ids (§6.13), padded to 8
          FullLocal: shm_offset u64, the row's offset in the shared-memory region the driver maps
```

- Values are the candidate's raw F32 logits, not log-probabilities. The driver forms `log q_i = y_i − lse` and `log s = lse_complement − lse` in F64 (RFC-0060 §5.4).
- Non-finite values travel as their IEEE bit patterns and are never clamped. The driver treats them as a harness error (RFC-0060 §5.4).
- Size for Gather is `32 + 4 · (k + tail)` bytes per position, rounded up to 8. At RFC-0060's `K = 256, M = 64` that is 1,312 B per position, so a verify window of 9 positions (γ = 8) carries 11,808 B, about 11.5 KiB, of tap. Larger entries use parts (above).

#### 6.6 `SeqControl` frame

The extension is `kind: u8` (1 Open, 2 Cancel, 3 Pause, 4 Resume, 5 Release, 6 ResumeOpen, 7 Reconcile), `body_enc: u8` (0 CBOR, 1 zstd-compressed CBOR), `reserved[6]`. The payload is the CBOR body of the matching `SequenceControl` variant (RFC-0001 §5.8, extended by RFC-0032 §3.6). The prompt in `Open` is a CBOR byte string of u32 token IDs, so a 128K-token prompt is 512 KiB. Bodies above `slot_size` are a protocol error, and the ingress daemon sizes the head's sequence slots for the instance's maximum context.

*Note 2026-10-23 (DIST-E8-T05): what the codec fixes.* The paragraph above leaves four points open, and DIST-E8-T05 fixes them as follows:
- **`ResumeOpen` (kind 6) is an `Open` with `resume` set.** `SequenceControl` has no variant of its own for it. The re-open after migration or recovery is `SequenceOpen { resume: Some(..) }` (RFC-0033 §4.4), so the encoder sends kind 6 exactly when `resume` is set, and kind 1 otherwise. The decoder refuses kind 1 with `resume` set and kind 6 without it (`InvalidRequest`). The head can tell a re-open from the kind byte without reading the body.
- **`Reconcile` (kind 7)** carries RFC-0033 §11's `{ seq, emitted, rng_counter, recent }`, with at most 8 tokens in `recent`. `core` gains the variant (RFC-0001 §5.8, note of 2026-10-23).
- **The body** is the variant's fields as a CBOR map, written with §9.1's deterministic encoder. Decoders ignore map keys they do not know, as on the control plane (ADR-0305 item 3). The prompt is a byte string of little-endian `u32`s, which `core`'s `TokenSlice` writes in every binary format.
- **Size.** The sender passes the receiver's `slot_size`. It writes plain CBOR (`body_enc` 0) when the frame fits, and compresses with zstd at level 3 (`body_enc` 1) only when it does not. A message that does not fit even compressed is refused before it is sent (`Capacity`). The receiver refuses a frame over its slot with `FrameTooLarge`.
- **Declared body sizes** (RFC-0060 §10.2). Each kind's body has a ceiling, before and after decompression, which both ends check. A small compressed frame therefore cannot make the receiver hold a large body. Every kind but `Open` and `ResumeOpen` has a ceiling of 1 KiB [target], since their bodies are small fixed maps; a `Reconcile` with 8 recent tokens is under 100 bytes. `Open` and `ResumeOpen` have the receiver's `slot_size` plus 1 MiB [target] for a grammar sent inline, within §9.1's 64 MiB. The sender refuses a body over its ceiling before it compresses anything (`Capacity`), and the receiver refuses one with `InvalidRequest`. These are the sizes the format declares, so §10.2's bound for this decoder is the ceiling, not 4 × the input. The ingress daemon sizes the head's sequence slots as `sequence_slot_size(max_context)` = 64 (header) + 64 (extension) + 4 × `max_context` + 64 KiB [target] for the open's other fields, rounded up to 64 bytes. That is 589 952 bytes [target] for a 128K-token context, inside the 1 MiB slot limit of the `local_sequences` ring, which carries the same frames (§4). An inline grammar can exceed the allowance. It is then compressed, and refused if it still does not fit.

#### 6.7 `Credit` frame

```text
0   8   ack_seq      highest chan_seq such that every frame ≤ ack_seq was received and validated
8   8   limit_seq    the sender may send frames with chan_seq ≤ limit_seq
16  8   echo_seq     the chan_seq whose arrival triggered this credit (0 if none)
24  4   echo_delay   ns between that arrival and this credit being sent
28  2   n_nack       ≤ 16
30  2   flags        bit0 SACK (the nack list is selective), bit1 RESET (receiver lost state; resend from ack_seq+1)
32  8×n nack list    chan_seq values the receiver is missing or rejected
pad to 64
```

`limit_seq − ack_seq` equals the number of receive slots the receiver currently has posted. §7 defines when credits are sent.

#### 6.8 Collective and expert-parallel frames

`Collective` extension (32 bytes, padded to 64):

```text
0   8   step
8   2   layer
10  1   phase        0 ATTN_ALLREDUCE | 1 FFN_ALLREDUCE | 2 ENTRY_BCAST | 3 EXIT_GATHER | 4 ROUTER_BCAST
11  1   algo         0 ONE_SHOT | 1 RING_RS (reduce-scatter step) | 2 RING_AG (all-gather step)
12  2   src_member
14  2   n_members
16  2   ring_step
18  1   dtype        WireDType; F32 in exact mode (RFC-0032 §5.3)
19  1   reserved
20  4   elems        elements in this part
24  4   elem_offset  offset of this part in the full vector (ring algorithms)
28  4   rows
```

`ROUTER_BCAST` payload: per row, `k: u8, pad[3]`, then `k × f32` routing weights, then `k × (expert u16)`, padded to 4 bytes (every field naturally aligned). `StepDescriptor` reuses the `Activation` extension and payload verbatim (RFC-0032 §5.2).

`EpDispatch` and `EpCombine` extension:

```text
0   8   step
8   2   layer
10  2   n_rows        rows (token positions) in this message
12  1   dtype
13  1   reserved
14  2   part_index    0-based
16  2   part_count    ≥ 1; a dispatch or combine larger than max_frame is split at row boundaries
18  6   reserved
24  …   per row (dispatch only), 4-byte aligned: row u16, k_local u8, pad u8, k_local × weight f32,
        k_local × expert u16, pad to 4
```

The payload is `n_rows` hidden-state rows (dispatch), or `n_rows` rows of summed, weighted expert outputs (combine). A part's row limit counts the extension with the rows: the largest r with `64 + ⌈(24 + r · m_row) / 64⌉ · 64 + r · row_bytes ≤ max message`, where `m_row` is the per-row dispatch metadata. At hidden 7168 in F32 on Thunderbolt 5, that is 584 rows at `k_local = 1` (m_row = 12 B) and 583 at `k_local = 8` (m_row = 52 B), so prefill-sized dispatches travel in parts [derivation at k_local = 8: 583 rows take 64 + 30,400 + 16,715,776 = 16,746,240 B ≤ 16,773,120; 584 rows take 16,774,912 B].

**Revision note, 2026-10-22 (DIST-E8-T08): how the codec reads the fields the tables leave open.**
- **`Collective` payload.** Every phase but `ROUTER_BCAST` carries `elems` elements of `dtype`, packed with no row stride. A ring step's slice starts at element `elem_offset` of the vector, so it need not start on a row. `rows` is the token positions the whole vector covers.
- **`ROUTER_BCAST`.** The primary, member 0 (RFC-0032 §5.1), sends it with `algo` ONE_SHOT and `elem_offset` 0.
  - `rows` counts its rows, and `elems` the experts they name in all. A row may name none.
  - The weights are F32 whatever `dtype` holds. `dtype` is still the channel's wire dtype, which every frame is checked against (§6.4).
- **Members.** `n_members` is at least 2, `src_member` is below it, and a ONE_SHOT frame has `ring_step` 0.
- **The step.** The `step` in these extensions equals the base header's. The frames carry no `TRACE`, so `HAS_TRACE` is clear.
- **EP rows.** `EpDispatch` and `EpCombine` rows are `hidden` elements of `dtype` at the `Activation` payload's stride (§6.3), each row's bytes rounded up to 64.
  - `hidden` is the stage's, from the plan, because neither extension carries it. At hidden 7168 in F32 the stride is the row, 28,672 B, as the derivation above uses.
  - A message holds at least one row, and every dispatch row names at least one expert (`k_local ≥ 1`).
- **EP parts.** Each part is a whole frame for its rows, with their routing rows, and holds as many rows as the row limit above allows. A combine part has no routing rows, so its `m_row` is 0.
- **`StepDescriptor`.** The primary forwards the inbound frame's extension and payload byte for byte, under ftype 0x09 and the `Collective` channel's `chan_seq`. The base header keeps the minor, flags, instance, epoch and step.

#### 6.9 `Hello`, `HelloAck`, `Close`

`Hello` is the first frame on every channel and every blob stream. Like every control frame, it has `chan_seq = 0`:

```text
0   1   purpose         ChannelPurpose code (1..7, with 7 = Tap), or 8 for a blob stream
1   1   wire_minor_max
2   2   proto_major
4   4   reserved
8   8   capabilities    §10
16  8   instance
24  16  epoch           term u64, seq u64
40  2   from_stage
42  2   to_stage
44  1   lane
45  1   wire_dtype
46  2   reserved
48  4   slot_size       largest frame the sender of Hello will accept on the reverse direction
52  2   recv_slots      initial credit granted to the peer
54  2   cred_kind       1 DataPlaneCredential | 2 AdmissionCert (RFC-0034 §5, §6)
56  8   channel_token   RDMA and shared memory: from pairing; otherwise 0
64  …   pad to 64
```

The payload is the credential: 168 bytes for `DataPlaneCredential`, 192 for `AdmissionCert`.

`HelloAck` extension: `status: u16` (0 OK, otherwise a §12 code), `wire_minor: u8` (the minimum of both), `reserved[5]`, `capabilities: u64` (the intersection), `limit_seq: u64`, `slot_size: u32`, `recv_slots: u16`, `cred_kind: u16`. The payload is the acceptor's own credential, so authentication is mutual on every channel (RFC-0034 §6). After `HelloAck`, both directions are open.

`Close` extension: `code: u16`, `reserved[6]`, `last_seq: u64` at offset 8 (the last data `chan_seq` the closer sent). The payload is a UTF-8 reason. A graceful close waits until `last_seq` is acked.

A reconnect `Hello` (§7.4) adds `resume_from: u64` at extension offset 64: the first data `chan_seq` the sender will resend. The matching `HelloAck` carries the receiver's current `ack_seq` in `limit_seq − recv_slots`.

**Revision note, 2026-10-22 (the DIST-E8-T01 codec's review): telling the cases apart.**
- **`resume_from`.** Data `chan_seq` starts at 1, so a reconnect's `resume_from` is never 0. A `Hello` whose extension is longer than 64 bytes with 0 at offset 64 is a fresh `Hello` carrying a later minor's trailing fields (§10), not a reconnect; one shorter than 72 bytes but longer than 64 is refused. A sender never writes 0 as a reconnect. Offsets 64–72 therefore sit in every `Hello` longer than 64 bytes, and a later minor's trailing `Hello` fields start at offset 72, after a zero `resume_from` (maintainer decision, 2026-10-22).
- **A refusing `HelloAck`.** The receiver reads `status` first. A refusal is read whatever its `cred_kind` and payload hold: its status is the message, so neither is checked, since a refusal may come before the acceptor has a credential to show. A refusing acceptor may write `cred_kind = 0` with no credential. An acceptance (`status = 0`) must name a known `cred_kind` and carry its credential.

#### 6.10 Blob-plane frames (weight chunks and digests)

Blob streams on `syntropy/blob/1` carry `Hello` (with `cred_kind = AdmissionCert`), then one request, then its response. The chunk store, source selection and resume policy belong to RFC-0041 (ADR-0016). This section fixes the bytes.

`BlobRequest` extension:

```text
0   1   kind         1 CHUNK | 2 BY_DIGEST (manifest bodies, snapshots, plan bodies) | 3 SESSION_KV (RFC-0022)
                     | 4 TENSOR_GROUP (RFC-0023: an expert group homed on the serving node's RAM, M3)
                     | 5 CANARY_BOUNDARY (RFC-0041 §5.3: a stage's warm-up canary boundary hidden states, M2)
1   1   priority     0 demand (a runner is Fetching, or a step waits on the bytes) | 1 replacement (recovery,
                     RFC-0033) | 2 prefetch (RFC-0023 peer → local prefetch) | 3 background
2   6   reserved
8   32  model        ModelId (zero for BY_DIGEST)
40  32  id           chunk BLAKE3, digest, or session blob id; TENSOR_GROUP: TensorGroupId (u32) in the first
                     4 bytes, zeros after; CANARY_BOUNDARY: InstanceId u64, PlanEpoch (term u64, seq u64),
                     producing StageIx u16, zeros after
72  8   offset       resume offset within the object
80  8   len          bytes wanted; 0 = to the end
```

*Note 2026-10-23 (RFC-0001 §18.28): a shard fetch's class.* A runner's `FetchShard` carries RFC-0041 §7's `FetchPriority`, and the store's chunk requests for it use these classes:
- `Urgent`, recovery after a node loss, uses `replacement`, as RFC-0033 §6.3 runs a replacement's fetch.
- `Normal`, a load, uses `demand`: a runner is `Fetching`.
- `Background`, pre-staging a sibling or a replica, uses `background`.

`prefetch` is RFC-0023's peer-to-local expert prefetch, which no shard fetch uses.

The response is a sequence of `BlobData` frames with payloads of at most `net.blob_frame` (1 MiB). Their extension is `id[32], offset u64, total u64, flags u16` (bit0 `LAST`), followed by padding. Errors use `Close` with `NotFound`, `Busy` (with a `retry_after_ms` in the reason) or `Throttled`. The receiver writes data through the store's direct-I/O path and verifies the object's BLAKE3 when it completes (RFC-0041). A mismatch discards it and marks the source suspect for that model (ADR-0016).

`TENSOR_GROUP` requests are served from a *runner's* resident tier, and daemons hold no model bytes (RFC-0001 §2.1). They therefore travel on a runner-to-runner `Bulk`-purpose data-plane channel, which carries the same `BlobRequest` and `BlobData` frames and is paced like the blob plane (§7.5). The bytes are verified against the group's tensor hashes from the manifest (RFC-0001 §5.3).

`CANARY_BOUNDARY` requests carry the multi-node warm-up canary (RFC-0041 §5.3) during `Loading`, before data-plane channels exist:
- The daemon hosting stage k requests stage k−1's boundary states from stage k−1's daemon, with `model` set to the plan's `ModelId` and `priority = demand`.
- The payload is the 32 canary tokens' boundary hidden states, `32 × hidden × streams` F32 values in row order, about 0.9 MB at hidden 7168. These are activations, not model bytes, so the runner hands them to its daemon through its shared-memory region and the daemon serves them.
- The server holds the stream open until its stage's layer range has run, then sends the rows. If the instance and epoch are unknown, or the plan is retired, it closes with `NotFound`.
- Plain `BLAKE3` verification does not apply, because nobody knows the hash in advance. Integrity rests on frame CRCs and the authenticated stream, and a wrong result shows up as a canary digest mismatch at the tail.
- The kind is an additive wire-minor change (§10). A server on an older minor closes with `NotFound`, and the requester reports the multi-node canary computation as unavailable for that load. The canary then depends on the other two digest sources RFC-0041 §5.3 lists, the digest index and the store's cached record.

#### 6.11 `KvChunk` frame

This frame carries one RFC-0022 `KvChunk` (RFC-0022 §10): one layer, one page class, one contiguous token range, page-first. It is used for KV migration, session transfer and recovery (RFC-0033 §5.2). Instance and exporter epoch are in the base header. Page contents are opaque here: RFC-0040 owns the in-page order (`KvLayoutId`), and RFC-0022 owns the classes and dtypes.

```text
0   16  seq
16  2   layer            LayerIx
18  1   class            1 Full | 2 Windowed | 3 Compressed | 4 State | 5 Aux   (RFC-0022 PageClass)
19  1   kv_dtype         1 Native | 2 Q8 | 3 K8V4   (RFC-0022 KvDType; the block size and rotation flag are in class_param)
20  4   class_param      Windowed: window; Compressed: ratio; Q8/K8V4: block (low 16 bits), rotate (bit 16)
24  2   layout           KvLayoutId (RFC-0040)
26  2   page_tokens
28  4   first_token
32  4   n_tokens
36  2   flags            bit0 LAST_FOR_LAYER, bit1 LAST (last chunk of the whole transfer)
38  2   reserved
40  32  kv_settings_hash RFC-0022: binds the KV dtype, RoPE configuration and every setting that changes KV contents
72  32  hash             BLAKE3 over this chunk's payload
104 …   pad to 128
```

- **Payload.** Page-first bytes, with each page slice padded to 4 KiB (RFC-0022 §10).
- **Checks.** The importer checks `kv_settings_hash`, `page_tokens`, `layout` and `hash` before allocating. Any mismatch rejects the import, and the caller falls back to replay or re-prefill.
- **Streaming.** Senders stream layer by layer, so the receiver can inject pages as they arrive and overlap transfer with the next layer's copy-in.
- **Channels.** Runner to runner, chunks travel on a `Replay`-purpose channel with Bulk priority and bulk pacing (§7.5): below the data plane, as RFC-0022 requires. From a store, session blobs travel on the blob plane with `SESSION_KV` (§6.10).

#### 6.12 `Recover` frame

RFC-0033 §4.2 owns the protocol. The bytes:
- **Extension.** `phase: u8` (1 QUERY | 2 REPORT | 3 PLAN | 4 DONE | 5 KV_READY), `reserved[3]`, `n: u32`, `from_stage: u16`, `reserved[6]`.
- **QUERY, REPORT and DONE entries** are 56 bytes each:

  | Field | Type |
  |---|---|
  | `seq` | `[16]` |
  | `resume_pos` | u32 |
  | `have_pos` | u32 |
  | `history_from` | u32 |
  | `history_to` | u32 |
  | `flags` | u32: bit0 `NEEDS_REPLAY`, bit1 `CAN_SERVE_HISTORY`, bit2 `ENDED`, bit3 `HAS_STATE` |
  | `state_pos` | u32 |
  | `state_restore` | u32 |
  | `reserved` | u32 |
  | `max_step` | u64, at offset 48 |

  `state_pos` and `state_restore` cover State layers (RFC-0033 §4.2.1). `max_step` lets a replacement head continue the step counter.
- **PLAN entries** are 40 bytes, one per (sequence, affected stage): `seq[16]`, `stage u16`, `source u16`, `replay_from u32`, `need u32`, `drop_below u32`, `flags u32`, `state_restore_to u32`. The last field is the planned restore point of a stage with State layers (RFC-0033 §4.2), and 0xFFFFFFFF for other stages. Each stage acts on the entries whose `stage` matches its own.
- **KV_READY entries** use the QUERY/REPORT layout with `have_pos` = the position up to which imported KV is ready (RFC-0033 §5.2).

#### 6.13 `TapIds` frame (parity taps)

This frame exists for RFC-0060 §5.5 and amendment A4.
- **Channel.** A `Tap`-purpose channel runs from the parity driver's daemon to the tail runner. It also runs to the head runner when the sequence's drafter sits there (RFC-0012 §8).
- **Authentication.** It is authenticated like a sequence channel, with `cred_kind = AdmissionCert`.
- **Gating.** The channel exists only while `debug.parity_api` is enabled on the driver's node (RFC-0060 §1.3).

Extension:

```text
0   16  seq
16  4   first_position
20  2   n_pos           positions covered by this frame
22  2   k               reference top-K ids per position (TapMode::Gather { k, tail }); 0 for FullLocal
24  2   tail            tail-sample ids per position; 0 for FullLocal
26  …   pad to 64
```

- **Payload.**
  - `n_pos × u32` teacher tokens, the tokens the tail sends around the ring in place of sampled ones;
  - then, for `Gather`, `n_pos × (k + tail) × u32` ids, position by position: the position's `k` reference top-K ids (T, from the RefBlock), then its `tail` tail-sample ids (S, drawn uniformly without replacement from V∖T, seeded by `(ModelId, corpus position)`, RFC-0060 §5.3). The tail returns logits in exactly this order (§6.5 tap block). It does not check that S is disjoint from T; the driver owns the sampling.
- **Flow.** The driver sends ahead of the positions in flight, within normal credit flow control. A tail that reaches a position whose ids have not arrived waits device-side (RFC-0012 §8).
- **Capability.** Bit 12 `TAP` (§10).

#### 6.14 `EmbeddingResult` frame

This frame implements RFC-0050 amendment A5 for prefill-only embedding sequences (`SequenceOpen::limits.embed`, RFC-0032 §3.6).
- **Channel.** The tail sends one result per sequence, in one or more parts, on its `Tokens` channel to the sequence's ingress, in place of tokens, after the last prefill chunk. No ring-return frame follows.
- **Extension.** `seq[16]`, `dims: u32`, `dtype: u8` (1 = F32, always in v1), `pooling: u8` (1 last_token | 2 mean | 3 cls), `normalised: u8`, `reserved: u8`, `prompt_tokens: u32`, `part_index: u16`, `part_count: u16`, padded to 64.
- **Payload.** This part's elements of the `dims × f32` vector, little-endian. A vector larger than the channel's slot is split at element boundaries, each part a whole frame holding as many elements as fit beside the header and extension. The receiver acts on the result only when every part has arrived, as with `Token` (§6.5).
- **Capability.** Bit 14 `EMBEDDING` (§10). A sender never writes the frame on a channel that did not negotiate it.

*Note 2026-10-23 (DIST-E8-T15):* the part fields took the extension's `reserved[4]` at offset 28. The text sent a larger vector in `Token`'s parts, but the extension had no part fields, and a 4096-dimension F32 vector (16 KiB) does not fit a 16 KiB `Tokens` slot with its header. `part_count = 0`, what those reserved bytes held, is refused, as `Token` refuses it: no sender ever wrote the reserved form, so nothing needs it read as one part. A result in several parts needs only bit 14 `EMBEDDING`; bit 15 `TOKEN_PARTS` gates `Token`'s parts alone. Recorded under the maintainers' standing instruction of 2026-10-22 on wire-format choices.

### 7. Flow control, acknowledgement, retransmission and failover

One mechanism runs on every backend (ADR-0307), so failover between backends needs no translation.

#### 7.1 Credits

- **Control frames carry no sequence number**. `Hello`, `HelloAck`, `Credit`, `Ping` and `Close` carry `chan_seq = 0`. They consume no credit and are never NACKed or held for ordering. A credit is a cumulative snapshot, so a lost credit is repaired by the next one, or by the sender's ack timeout (§7.3), which provokes a fresh credit.
- **On RDMA**, each side keeps receives posted for incoming control frames as well as data:
  - On a unidirectional channel, the data sender's receive queue sees only control frames, so it posts `recv_slots + 4` small (4 KiB) receives. At most one credit is outstanding per in-flight data frame, plus `Ping` and `Close`.
  - The data receiver's queue also takes the sender's `Hello`, `Ping` and `Close`. Receives are consumed in arrival order whatever their size, so it posts `recv_slots + 4` receives of the data slot size. Credits count only `recv_slots` of them, and the other 4 absorb control frames, so a `Ping` never takes a data frame's receive.
  - On a bidirectional `Collective` channel, both sides post `2 × recv_slots + 4` receives of the data slot size.
  - Receives are re-posted as they complete, so UC's "a receive is posted for every message sent" holds for control frames too.
- **At `Hello`** the receiver grants `limit_seq = recv_slots`, with `ack_seq = 0`, since data frames start at `chan_seq = 1`. The invariant `limit_seq − ack_seq = posted data slots` holds from the start.
- The sender may send frame `chan_seq` only while `chan_seq ≤ limit_seq`. Otherwise `try_send` returns `WouldBlock`, and the stage holds before its next step. The held time is reported as `credit_stall` (RFC-0001 §8.4 rule 1).
- The receiver advances `limit_seq` as it releases slots, which happens when an `InFrame` is dropped and the slot is re-posted.
- **When credits are sent.**
  - **Activations and Tokens channels:** a `Credit` follows every received *data* frame (control frames never trigger credits), with `echo_seq` set to it. A 64-byte credit per 16–32 KiB frame is 0.2–0.4 % overhead. It gives the sender an RTT sample per frame and releases its retention slot as early as possible.
  - **Replay, Bulk and the prefill lanes of Collective:** a credit every `max(1, recv_slots / 4)` frames, or when the remaining credit drops below half.
- **α sampling.** `rtt = now − sent_at(echo_seq) − echo_delay`. The one-way α estimate is `rtt / 2`, refined by RFC-0010 into `LinkCaps::alpha` with `MeasureSource::Online`. These samples come free on every edge carrying traffic.

**Sizing [derivation].** A channel needs enough credit to cover one round trip of frames in flight:

```text
recv_slots ≥ ⌈ (2·α_p99 + S/β) / t_issue ⌉ + 1
  t_issue = interval between frames on the channel:
            single-stream decode: one frame per ring traversal (≫ 2α), so 2 slots suffice
            throughput mode with M steps in flight: M + 2
            prefill: chunk compute time per stage; 3 slots allow double buffering plus one retransmission
```

Defaults: Activations 4, or M + 2 in throughput mode; Tokens 4; prefill (the same Activations channel, larger slots) 3; Collective 4 per member pair; Replay 4; blob streams 8. Slot sizes come from the plan (§8).

#### 7.2 Integrity failures and gaps

| Receiver observes | Action |
|---|---|
| `header_crc` fails | drop the frame; NACK `expected_seq`; count `crc_header_fail` |
| `payload_crc` fails | drop; NACK that seq; count `crc_payload_fail`. 3 payload CRC failures on one channel within 10 s → treat as a link fault (§7.4) |
| `chan_seq` > expected (a gap: UC loss, or reordering after failover) | hold out-of-order frames in their slots up to the credit window; NACK each missing seq once per `nack_interval` = max(2 · α_p99, 100 µs) |
| `chan_seq` < expected (a duplicate, after retransmission) | drop silently; re-send the current credit so the sender learns the ack |
| epoch or instance mismatch | drop; count; never NACK |

#### 7.3 Retention and retransmission

- The sender keeps each sent frame in its slot until `ack_seq` covers it. That is the **retention window**, at most `retention` frames, and never less than the peer's `recv_slots`.
- On NACK, it resends the listed frames **byte-identical** to the originals: the same header, CRCs and, under AEAD, the same sealed bytes and tag. Receivers recognise duplicates by `chan_seq`.
- If no credit advances `ack_seq` within `ack_timeout = max(4 · α_p99, 200 µs)` while frames are outstanding, it resends the oldest unacked frame once. Repeated timeouts escalate to §7.4.
- Retention bounds memory exactly: `retention × slot_size` per channel, allocated at `Warming`.
- On RC queue pairs, QUIC and TCP, the transport already retransmits lost packets, so Syntropy's retransmission fires only on CRC failures (memory or bus corruption) and after failover. On UC it is the reliability layer.

**Note, 2026-10-23 (DIST-E7-T02): the channel core, as built, and the credit probe.** `net::transport::core::ChannelCore` is §7.1–§7.3 for one side of one channel, sans I/O: every backend wraps one per side, moves its frames and passes the time in. Where this section leaves a choice open, the core takes these:
- **The credit probe, which amends §7.1's "control frames never trigger credits".**
  - The gap it closes: a credit that only grants more, sent when the receiver's caller drops a frame, can be lost while the sender has nothing unacknowledged. No later credit follows, since no data arrives, and no ack timeout fires, since nothing is outstanding.
  - So a sender whose `try_send` was refused for credit, with nothing unacknowledged, sends a `Ping` after one `ack_timeout`, then after 2, 4, … up to 64 `ack_timeout`s [target], until a credit lets it send. A data receiver answers a `Ping` with its reply and its current credit.
  - Only a refused send arms the probe, so an idle channel sends nothing.
- **The receive checks** run `magic`, the lengths and `header_crc`, then the channel key, then `wire_minor ≤ negotiated` and the frame rules (§6.2), then `chan_seq` and `payload_crc`. §6.1 lists `wire_minor` before `header_crc` and the key.
  - Reading the minor after `header_crc` means a damaged minor byte counts as damage, which is NACKed, and does not close the channel.
  - Reading it after the key means a frame of another instance or epoch is dropped and counted as stale whatever its minor or type, and never closes the channel.
  - An intact frame of the channel gets the same verdict either way. A minor above the negotiated one closes the channel as `InvalidRequest`, with no code of its own.
- **The credit granted** is `ack_seq + recv_slots −` the frames handed over or waiting to be, so it never shrinks and `limit_seq − ack_seq` is the data slots free. The receiver refuses a frame past the credit it granted as a protocol error. The sender takes no `limit_seq` above its `ack_seq` plus the peer's `recv_slots`, and stops at `retention` unacknowledged frames, so §7.3's memory bound holds against any peer.
- **α samples** come from credit echoes and from replies to `Ping`s the core itself sent; a reply carrying any other `origin_ns` gives none, so a peer cannot raise α and with it the ack timeout. An echo of a frame sent more than once gives none, since its arrival cannot be matched to one send. The core's α p99 is the largest of its last 128 samples [target], or the link profile's before the first.
- **Escalation.** Three ack timeouts in a row are `AckTimeout`, and an ack resets the count (§7.4's "ack timeout × 3"). Payload CRC failures of control frames count towards `CrcStorm` as data frames' do; a control frame is dropped and never NACKed.
- **A stream backend** (QUIC, TCP) gives the core each frame's base header before reading its body. A data frame longer than the slot size, or a control frame longer than 512 bytes, is `FrameTooLarge` (§12), and the channel closes.

#### 7.4 Link faults, reconnection and demotion (ADR-0011 item 7)

```text
on fault(ch):                        // QP error, QUIC connection lost, TCP reset, CRC storm, ack timeout × 3
    ch.state = Reconnecting
    failures[ch.edge].push(now)
    if failures[ch.edge].count_within(60 s) >= 3:
        demote(ch.edge)              // next transport by measured cost (§3), for the rest of the plan epoch
        emit ReportEdgeDemoted{edge, from, to, cause}   // control plane; shown in dashboard
    t = transport_for(ch.edge)
    loop until link.failover_timeout (2 s LAN, 10 s WAN):
        if t.reconnect(ch).is_ok():          // new QP + re-pair, or new QUIC stream/connection, or new TCP conn
            resend frames (ack_seq, last_sent] from retention, in order
            ch.state = Open; return
        backoff(10 ms, ×2, max 200 ms)
    ch.state = Down
    report EdgeDown to daemon → membership indirect probe (RFC-0030 §5.3) and recovery (RFC-0033 §3)
```

- Reconnection runs in the runner's `net` domain. The hot thread sees only `WouldBlock`, plus stalled completions while it lasts.
- Because the sender retains every unacked frame, a successful reconnect, even on a different transport, delivers each frame exactly once, in order, with no help from the compute layer. Activation-history replay (RFC-0033) is needed only when a *stage* is lost, never for a lost link.
- A reconnected channel keeps its `ChannelKey` and data `chan_seq` space. The reconnect `Hello` (`chan_seq = 0`) carries `resume_from = ack_seq + 1` as the sender believes it. The receiver's `HelloAck` carries its current `ack_seq` as `limit_seq − recv_slots`, so the sender resends from there. A receiver that restarted, and so lost its state, answers with the `RESET` flag. That ends the channel, because a restarted runner is a stage loss (RFC-0033).

#### 7.5 Bulk pacing

Bulk transfers yield to the data plane (RFC-0001 §8.4 rule 6; ADR-0011 item 8):
- Every 250 ms, a node that carries both a live data channel and a blob stream on the same physical edge compares the data channel's α p99 over that window with its idle p99, which RFC-0010 records.
- If the ratio exceeds `net.bulk_p99_ratio` (1.2), the blob send rate halves, and growth then stops two steps, 10 % of β [target], below the rate that broke the bound. Otherwise it grows by 5 % of the edge's β, up to that ceiling, or β.
- The ceiling rises one step after 64 windows (16 s) within the bound [target]. That wait doubles after each breach that follows a rise, up to 1,024 windows [target].
- The rate starts at the floor and never drops below it: `net.bulk_floor`, 5 % of β. Progress is guaranteed, and a 100 GB fetch cannot starve.

The limiter is a token bucket in the daemon's `bulk` runtime. It applies to `BlobData` sends and to `KvChunk` sends on Bulk channels.

**Note, 2026-10-23 (DIST-E7-T13): the ceiling, the start at the floor, and where the pacer lives.** The task lead took these under the maintainer's standing instruction.
- **Why a ceiling.** As first written, the rule climbs back into saturation every few seconds. In the simulator, the p99 of each window but the probe windows stays within 1.2 × idle, but each probe window holds enough samples above the bound that the p99 over a long transfer exceeds it.
  - The figures are [estimate]s from RFC-0060 §8.3's `lan_2_5g`, seed 1. The data channel sends 16 KiB every 2 ms; bulk goes in 4 KiB frames; the transfer is 10 GiB, 47 s simulated.
  - Without the ceiling: 1.515 × idle over the transfer, with 17 halvings.
  - With a ceiling that re-probes every 8 windows: 1.333 ×, with 4 halvings.
  - With the rule above: 1.046 ×, with 2 halvings, and the rate settles at 90 % of β. Seed 3 gives 1.076 ×, with 3 halvings.
  - **Caveat, for the maintainers.** The first climb from the floor always breaks the bound once, in one window. That window drops below 1 % of a transfer's samples only beyond about 100 windows, 25 s [estimate], which a 10 GB transfer on 2.5 GbE reaches. A transfer shorter than that can show a p99 over the transfer above 1.2 × idle, while every window but that one stays within it. A non-probing pacer, held at β less the measured data rate, would avoid the breach but needs that rate measured.
- **The start at the floor.** A transfer that starts beside a live ring never begins by taking the whole link.
- **The pacer** is `net::transport::pacing::BulkPacer`, sans I/O. It holds the token bucket, in whole bytes and nanoseconds. Its AIMD takes the nearest-rank p99 of each window's α samples from the edge's other channels; a window with no sample, when no data channel is live, counts as within the bound.
  - 1,024 such windows in a row, 256 s [target], the longest re-probe wait, also reset the ceiling to β and the re-probe wait to 64 windows. The data channel has gone away or gone idle, so its old breaches no longer hold the rate down. A ring that pauses that long meets the rule as first written when it comes back, and the first window after its pause can break the bound.
  - A shorter pause keeps them, because the ring that comes back saturates the link where the last breach found. On the unit tests' model edge, over 4,000 windows [estimate]: a reset after one empty window let the first window after each pause break the bound, 444 times with 8 windows on and 1 off, against 6 without a reset. A reset after 64 did the same for longer pauses: 33 times with 40 on and 200 off, against 5.
  - In the simulator, `SimNet::pace` gates `Bulk` and `Replay` sends on an edge with it.
  - In the QUIC and TCP backends, DIST-E7-T25 gates their bulk sends with it, and holds the rig check of ADR-0011 item 8's bound, the first task where it can run.
  - ADR-0011 item 8 states only the 1.2 × bound, which this keeps, so it needs no revision.

### 8. Zero-copy receive and buffer pools

**Pools.** Every registered buffer comes from the tier manager's `HostPinned` network sub-arena, sized by the plan (RFC-0001 §9.1, RFC-0020). Each runner gets one pool per (device NUMA node, purpose class). Slots are fixed-size and 64-byte aligned. Each pool is registered once with every transport that needs it: RDMA memory regions, io_uring fixed buffers where the store reuses them, and `KernelProvider::register_host` so the GPU can DMA from and to them.

| Pool | Slot size | Count per channel | Notes |
|---|---|---|---|
| decode activations | `S_dec` at the plan's max verify window and batch, rounded up to 4 KiB | recv_slots + 4 + retention: the 4 are the control receives of §7.1 (note of 2026-10-23) | KiB-scale |
| prefill activations | `S_pf` at `batching.prefill_chunk`, capped at `max_frame` | 3 + retention | MiB-scale; chunks above `max_frame` split into parts (§6.3) |
| tokens | from the plan: `max_batch × (48 B + 4 B × T_max + logprob bytes + draft bytes)`, rounded up to 4 KiB; on a parity sequence's ingress channel, plus the tap block for one prefill chunk at the spec's k | 8 | larger steps travel in parts (§6.5) |
| collective | partial-sum size × rows, at the max verify window | 4 per member pair | |
| replay | prefill slot size | 4 | only while recovering (RFC-0033) |
| headers | from the plan: 64 B base header plus the extension at the plan's maximum rows per frame × per-row metadata (spans and tree, `TOKEN_IDS`, `ROUTER_HINT` at the plan's k, EP row metadata), rounded up to 64 B. A 512-row prefill frame with `TOKEN_IDS` and `ROUTER_HINT` at k = 8 needs over 10 KiB [derivation: 512 × 4 B + 512 × 16 B] | as above | base and extension built in place; RDMA sends them as the first scatter-gather entry; the extension counts against each part's row limit |

**Receive paths** (RFC-0001 §9.4):

| Backend | Lands in | Host copies before the consumer | Device copy |
|---|---|---|---|
| RDMA with GPUDirect (Linux, probe-confirmed) | a slot in device memory | 0 | 0 |
| RDMA (pinned) | a registered pinned slot | 0 | 1 H2D by the dispatcher, or 0 when a CPU stage consumes in place or the device is `Unified` |
| Shared memory | the segment slot | 0 (the sender wrote it there) | as above |
| QUIC, TCP | the transport's decrypt buffer, then a registered slot | 1 memcpy (unavoidable on IP paths) | as above |

**Send paths.** The GPU writes the stage's output hidden state straight into a host-mapped registered slot: the output kernel's destination is the slot, or a D2H copy into it. Completion is signalled with the doorbell (`signal_host`, RFC-0001 §5.7). The net side never calls a device synchronise. On RDMA the slot is posted as is. On QUIC and TCP, TLS reads it once while encrypting.

**Note, 2026-10-22 (DIST-E7-T12): a slot's generation.** A buffer descriptor (`PayloadSlice`, `ExtBuf`; §2) names a slot by pool, slot index, offset and length, and is `Copy`. If its holder drops the slot's last reader before `try_send`, the slot can be freed, handed out and frozen again, and a lease taken through the old descriptor would carry the new bytes without any error.
- Each slot's state therefore carries a generation, which advances whenever the slot is freed. Each descriptor carries the generation of the freeze it was made from.
- Leasing a slot through a descriptor checks the generation, and a stale descriptor is refused with `SendError::Invalid`, never leased. Misuse becomes a refusal instead of a frame with the wrong payload.
- DIST-E7-T12 builds this with the lock-free slot handout.

**Note, 2026-10-23 (DIST-E7-T12): as built.**
- **The generation.** A slot's state is one 64-bit word: the generation in the high 32 bits, and in the low 32 bits free, writing, or the count of readers. A writer dropped unfrozen and the last reader dropped both free the slot at the next generation. `PayloadSlice` and `ExtBuf` carry `generation`, and `BufferPool::retain_payload` and `retain_ext` lease only at it.
- **Handing out slots.** `BufferPool::take` claims any free slot with no lock: each taker starts one slot further round than the last and claims the first free slot with a compare-and-swap. A channel core takes its receive slots this way.
- **Registration tables.** `RegistrationTable` holds one registration per pool per registrant: a transport kind, or a kernel provider by its `ProviderId` name, which the runner records, since `net` cannot depend on `kernels`. Registering the same pool again with the same registrant returns its one registration, and another pool under a registered `PoolId` is refused. The QUIC and TCP backends keep their pools in one. The channel-conformance suite's `Registration` rule checks the repeat on every backend whose fabric can register.
- **A channel side's slots** are `recv_slots + 4 + retention`, and the table's decode row now says so; it gave "recv_slots + retention". The four are the receives §7.1 reserves for control frames beside the data receives, which credit never counts, so a full data window never stops a `Ping` or a `Credit`, and a side that only receives, with no retention, still has them. The `recv_slots + 4` are the side's receive pool, which the channel core takes; the `retention` frames sit in the side's send pool, where their payloads were written. The four apply to every row of the table, not only decode: a `ChannelCore` takes a receive pool of `recv_slots + 4` on every channel, so the prefill, tokens, collective and replay counts are four short in the same way, as far as they count `recv_slots`. The pinned-memory derivation under the performance model, `(recv_slots + retention) × slot_size`, leaves them out too.
- **Header areas** count every side item's TLV header and up to 7 bytes of its padding: a per-row item's in every span, so a frame whose rows spread over its spans still fits, and the plan's further items per span (`SEQ_META`, `SEQ_CTRL`, `TRUNCATE`, `TRACE`) and per frame by their count and their values' bytes. The 512-row frame with `TOKEN_IDS` and `ROUTER_HINT` at k = 8 needs a 10,496-byte area. The codec's own extension for it, with one span, is 10,368 bytes [derivation: 64 B of fixed fields, a 32 B span, (8 + 2,048) B of `TOKEN_IDS` and (8 + 4 + 8,192) B of `ROUTER_HINT`, padded to 64].
- **Token slots** count each entry as 8-aligned, with `T_max × (4 + 8 × logprob_k)` logprob bytes, 4 B per drafted token and 4 B more per tree node. The tap block is 8 B plus `32 + 4 × (k + tail)` per position, padded to 8, for `Gather`, or 40 B per position for `FullLocal`. Like the formula above, they leave out the `Token` extension's side items: a sampled entry's `TRACE` (40 B each) and the ring-return items. A step over its slot travels in parts (§6.5), so those cost a part, never a frame that does not fit.
- **Landing parts.** `Landing` gives each part of a chunk, in order, the byte range `rows_before × stride` of the stage's input its payload goes to. A part out of order, a part count or layout that changes inside the chunk, or a part whose range would end past the input's bytes, is refused (`InvalidRequest`): the ranges are a device copy's destination, which no Rust bounds check guards. The bound is in bytes, because a part's stride comes from the peer's header, and every product of rows and stride is checked.

**Multi-part prefill frames** land part by part in separate slots. The dispatcher issues one H2D per part into consecutive offsets of the stage's device input buffer. That copy is needed anyway, so reassembly costs no extra host copy.

### 9. Control-plane and blob-plane encoding

#### 9.1 Control and join envelope (ADR-0305)

On `syntropy/ctl/1` and `syntropy/join/1` streams, each message is:

```text
| len: u32 (bytes after this field) | enc: u8 (0 = CBOR, 1 = zstd(CBOR)) | body |
```

- **Body.** CBOR (RFC 8949) produced by serde from the Rust types in RFC-0001, RFC-0030 and RFC-0032.
  - Encoding is deterministic: shortest-form integers, definite lengths, struct fields in declaration order, map keys sorted for `BTreeMap`.
  - Messages over 64 KiB are zstd-compressed at level 3. `PlanCommitted` bodies with tens of thousands of `TierAssignment` entries compress well.
  - The maximum message is 64 MiB. Larger state, such as snapshots, travels on the blob plane by digest (`BY_DIGEST`).
- **Evolution within a major.**
  - New struct fields carry `#[serde(default)]`, and decoders ignore unknown map keys. Structs never use `deny_unknown_fields`.
  - New enum variants are emitted only when every recipient's advertised **feature level** includes them. The coordinator tracks the cluster's minimum feature level (RFC-0032 §6.4). Peer-to-peer messages check the peer's level from its `Hello`.
- **Datagrams.** `| tag: u8 | CBOR body |`, with no length field. A datagram exceeding the path's datagram size limit is not sent. The beat builder trims piggybacked updates to fit (RFC-0030 §5.2).

```rust
/// Control-stream messages. Owner: this RFC (envelope); payload types: RFC-0030, RFC-0032.
pub enum CtlMsg {
    Hello { cert: AdmissionCert, proto_major: u16, feature_level: u32, build: BuildId, cluster: ClusterId },
    HelloAck { status: u16, cert: Option<AdmissionCert>, feature_level: u32 },  // cert: the acceptor's (mutual auth)
    Command { id: CommandId, cmd: ClusterCommand },
    CommandReply { id: CommandId, result: Result<LogIndex, CommandRejected> },
    LogAppend { term: u64, prev: (u64, u64), entries: Vec<LogEntry> },      // RFC-0032 §6.7
    LogAck { term: u64, applied: u64 },
    Sync { last: (u64, u64) },
    SyncReply(SyncReply),                                                    // entries, or a snapshot digest
    Election(ElectionMsg),                                                   // RFC-0032 §7
    Rpc { id: u64, body: NodeRpc },                                          // RFC-0030 §8.3
    RpcReply { id: u64, body: Result<NodeRpcReply, Error> },
    Bye { reason: ByeReason },
}
```

A control connection opens a dedicated stream for each of: log replication (coordinator to follower), commands (one stream per in-flight command), RPCs, and election. Membership beats are QUIC datagrams on the same connection.

#### 9.2 Blob plane

Blob streams use the data-plane frame layout (§6.10). Weight chunks are bulky and hot, and a fixed binary layout lets the receiver write straight into an aligned direct-I/O buffer.

### 10. Versioning and compatibility

| Layer | Identifier | Rule |
|---|---|---|
| Protocol major | `1` in every ALPN; `proto_major` in beacons and `Hello` | incompatible changes bump it. Nodes of different majors never connect, and discovery lists them as incompatible (RFC-0030 §4.1). Upgrading across a major is cluster-wide (RFC-0052 drains and upgrades node by node; mixed-major nodes form separate clusters until all are upgraded) |
| Data wire minor | `wire_minor` in every frame header and in `Hello` | additive only: new frame types, new TLV types, new flag bits, new trailing extension fields (receivers use `ext_len` to skip them). A channel runs at the lower of the two minors |
| Capabilities | 64-bit set in `Hello`/`HelloAck` | bit0 VERIFY_TREE, bit1 Q8_WIRE, bit2 F8_WIRE, bit3 KV_CHUNK_V1, bit4 SACK, bit5 RDMA_AEAD, bit6 MULTI_STREAM, bit7 EP_FRAMES, bit8 RING_ALLREDUCE, bit9 RECOVER_V1, bit10 PREFIX_ITEMS (the RFC-0022 items, TLVs 9–15), bit11 KV_HEADROOM, bit12 TAP, bit13 TENSOR_GROUP, bit14 EMBEDDING, bit15 TOKEN_PARTS, bit16 CHOSEN_LOGPROBS (added 2026-10-23: the `Token` entry's `logprob_k = 0x80`, OQ-NET-35, DIST-E8-T18), bits 17–63 reserved. A frame or TLV type tied to a capability is sent only when both sides advertise it |
| Control feature level | `feature_level: u32` in the control `Hello` | the coordinator emits event variants only up to the cluster minimum (RFC-0032 §6.4) |
| Beacon version | `beacon_version` byte | a receiver ignores beacon versions it does not know, and still lists the node as "unknown beacon version" |

**The compatibility promise**:
- Within one protocol major, a release interoperates with releases one minor older and one minor newer (N−1 and N+1), on all planes. That is the window RFC-0062 and ADR-0622 test, and all a rolling upgrade needs.
- Each release declares `[min_supported, current]` for `wire_minor` and for the control `feature_level` (`ci/protocol-versions.toml`, RFC-0062).
- The control `Hello` carries both ranges. A node is admitted to a connection only when its ranges overlap the peer's, and refused with `ProtoRangeDisjoint` otherwise. The cluster's effective level is the minimum over members (RFC-0032 §6.4).
- Pack and session formats are versioned separately (RFC-0041, RFC-0022).

### 11. Configuration

| Key | Scope | Default | Notes |
|---|---|---|---|
| `net.max_frame` | node | 64 MiB | capped per transport by `TransportCaps::max_frame` |
| `net.blob_frame` | node | 1 MiB | |
| `net.bulk_p99_ratio` / `net.bulk_floor` | cluster | 1.2 / 5 % of β | §7.5 |
| `link.connect_timeout` | cluster | 2 s LAN, 10 s WAN | §3 |
| `link.failover_timeout` | cluster | 2 s LAN, 10 s WAN; at most 15 s, iroh's per-path idle cap, so a larger value is refused at load with RFC-0054's `out_of_range` (revised 2026-10-23, §4.3's note) | §7.4 |
| `link.demote_after` | cluster | 3 faults in 60 s | ADR-0011 item 7 |
| `link.ack_timeout_min` | cluster | 200 µs | §7.3 |
| `link.p99_weight` | cluster | 1.0 Wi-Fi, overlay or relay, 0.25 otherwise (latency mode); within 0–10 | §3 λ (note of 2026-10-23) |
| `quic.congestion` | node | `cubic` | `cubic` \| `bbr` |
| `quic.lan_initial_window` | node | 256 KiB | §4.3 |
| `quic.max_conn_window` | node | 64 MiB | |
| `rdma.enable` | node | `auto` | `auto` \| `off`; `auto` uses TB5 and InfiniBand when present, RoCEv2 only on `trusted_fabric` interfaces |
| `rdma.poller_thread` | node | `auto` | one poller per RDMA NIC when ≥ 2 RDMA channels are open; otherwise the dispatcher polls |
| `rdma.gpudirect` | node | `auto` | used only after the RFC-0010 device-path probe passes |
| `shm.enable` | node | `auto` | |

Per-hop transport choice is not a knob. It comes from the plan. An operator who wants to forbid a medium marks the interface in `discovery.ignore_interfaces` (RFC-0030), or removes `trusted_fabric` (RFC-0034).

### 12. Error codes

`Domain::Net`, range 0x0200–0x02FF. These codes travel in `HelloAck::status`, in `Close::code` and in `Error::code`.

| Code | Name | Class |
|---|---|---|
| 0x0201 | `ProtoMajorMismatch` | `Unsupported` |
| 0x0202 | `CredentialInvalid` (signature, expiry, wrong node, wrong instance or epoch) | `Auth` |
| 0x0203 | `NotANeighbour` (credential valid, but the plan does not connect these stages) | `Auth` |
| 0x0204 | `UnknownFrameType` | `Unsupported` (was `Internal`; see the revision note below) |
| 0x0205 | `FrameTooLarge` | `InvalidRequest` (was `Internal`; see the revision note of 2026-10-23 below) |
| 0x0206 | `WireDTypeMismatch` | `Unsupported` (was `Internal`; see the revision note below) |
| 0x0207 | `CrcStorm` (3 payload CRC failures in 10 s) | `Link` |
| 0x0208 | `AckTimeout` | `Link` |
| 0x0209 | `QpError` (RDMA QP entered the error state) | `Link` |
| 0x020A | `ConnectTimeout` | `Link` |
| 0x020B | `PeerReset` (receiver lost channel state) | `Link` |
| 0x020C | `EdgeDemoted` | `Link` (informational; becomes a `Ready { degraded }` reason) |
| 0x020D | `RdmaNotEnabled` | `Unsupported` |
| 0x020E | `RdmaOsBuildMismatch` | `Unsupported` |
| 0x020F | `ShmUserMismatch` | `Unsupported` |
| 0x0210 | `BlobNotFound` / `BlobBusy` / `BlobThrottled` | `Transient` |
| 0x0211 | `EpochNotServed` | `Transient` (retry after `OnNewPlan`) |
| 0x0212 | `ProtoRangeDisjoint` (no overlap of declared wire-minor or feature-level ranges) | `Unsupported` |
| 0x0213 | `QpBudgetExhausted` (informational: channel opened on the next transport) | `Link` |

**Revision note, 2026-10-22 (DIST-E8-T01): the class of `UnknownFrameType`.** A receiver that refuses a frame because its `ftype` is not a type of the negotiated minor (§6.2) classes the refusal `Unsupported`, not `Internal`. RFC-0060 §10.2 requires a decoder of untrusted bytes to class every refusal `Integrity`, `InvalidRequest`, `Auth` or `Unsupported`, never `Internal`, and the fuzz targets assert it. The code is unchanged: the receiver still sends `Close { UnknownFrameType }` with 0x0204. The table now gives `Unsupported` for 0x0204, so the two RFCs agree.

**Revision note, 2026-10-22 (DIST-E8-T02): the class of `WireDTypeMismatch`.** A receiver that refuses an `Activation` because its `dtype` is not the channel's `ChannelSpec::wire` (§6.4) classes the refusal `Unsupported`, not `Internal`, for the reason the note above gives. It is not `Integrity` either: the frame's CRCs passed, and a resend carries the same bytes (§7.3), so no NACK can repair it. The code is unchanged: the receiver reports 0x0206. A dtype container the build does not carry is refused as `Unsupported` too, with no code of its own. The table now gives `Unsupported` for 0x0206.

**Revision note, 2026-10-23 (DIST-E7-T02): the class of `FrameTooLarge`.**
- A receiver that refuses a frame longer than its channel's slot size classes the refusal `InvalidRequest`, not `Internal`, for the reason the note on 0x0204 gives: it refuses untrusted bytes, so RFC-0060 §10.2 rules out `Internal`.
- It is not `Integrity`, since the frame's CRCs passed, nor `Unsupported`, since the frame type is one the channel carries. The peer broke the size the channel negotiated.
- The code is unchanged: the receiver sends `Close { FrameTooLarge }` with 0x0205. A sender's own refusal is `SendError::TooLarge`. An encoder that cannot fit a part into a frame still reports `FrameError::TooLong`, which is `Internal` with this code, because it is the sender's failure and not a refusal of received bytes.
- RFC-0050 §12.1 adds 0x0205 to its `Net` exempt list in the same change, so the code never takes a 400 shape.

## Performance model and budgets

The governing target is SC-12: added time per hop ≤ `α_probe + S/β_probe + Z`, with `Z ≤ 2 · t_switch(probe)`. RFC-0010's L1 probe measures α and β through the real transport stack, so the QUIC and TLS per-packet costs are already inside them. `Z` is what this RFC adds at the two ends: from the sender's GPU to the wire, and from the wire to the receiver's GPU.

**Z breakdown for a 32 KiB F32 decode frame between two discrete GPUs [estimate].**

| Step | Cost | Basis |
|---|---|---|
| Sender: output kernel writes into the host-mapped slot (no separate D2H) | ≈1.4 µs | 32 KiB at ≈24 GB/s pinned PCIe 4.0 (cost model §2.3) |
| Sender: doorbell observed by the spinning CPU | 1–5 µs | one PCIe posted write plus a polling interval; inside the cost model's 5–20 µs `t_switch` range (§3.2) |
| Sender: header build and CRC32C of the payload | ≈4 µs | ≥ 8 GB/s per core [estimate: one 8-byte `crc32` instruction per 3 cycles at 3 GHz, without interleaving]; 3-way interleaving would be about 3× faster |
| Sender: dispatcher → net-task hand-off, IP paths | ≈ 0.1–1 µs while the net task busy-polls; 20–50 µs for a cold eventfd wake | a cache-line transfer between cores when polling [estimate]. RFC-0012's estimate for a cold futex wake is tens of µs, which is why the net task polls while a `Ready` instance has live sequences (§4.4) |
| Receiver: net task → dispatcher hand-off | ≈ 0.1–1 µs | dispatchers poll their inbound rings (RFC-0012 §3.3) |
| Receiver: copy from the decrypt buffer into the registered slot (IP paths only) | ≈3 µs | memcpy at ≥ 10 GB/s [estimate] |
| Receiver: CRC check | ≈4 µs | as above |
| Receiver: ring hand-off and H2D copy, released by a device-side wait on a host flag | ≈1.4 µs + 1–5 µs | PCIe as above; wait release like the doorbell |
| **Total Z, IP path** | **≈15–25 µs with polling** (35–73 µs if the net task parks) | |
| **Total Z, RDMA (no memcpy)** | **≈12–20 µs** | |

- SC-12 allows Z of 10–40 µs, depending on the probed `t_switch`. At the low end of `t_switch`, the serial CRC work is the largest term.
- **Mitigation.** The receiver verifies the payload CRC in parallel with the H2D copy, and the compute launch waits on both flags, so the receive-side cost is `max(CRC, H2D)` rather than the sum. A 3-way interleaved CRC kernel cuts each CRC to about 1.5 µs.
- On `Unified` devices the H2D and D2H rows vanish.
- **Verification.** RFC-0061 measures Z per transport on the Tier 1 rigs and gates on SC-12.

**Per-hop cost by medium [estimate, from cost model §2.5 and §3.7; Z as above].**

| Medium, transport | α | S/β, 16 KiB BF16 | S/β, 32 KiB F32 | Hop, F32, p50 |
|---|---|---|---|---|
| RDMA over TB5 | 10–25 µs | 6.6 µs at 2.5 GB/s | 13 µs | ≈35–60 µs |
| 10 GbE, QUIC | ≈105–210 µs (half to all of the measured 210.9 µs send/recv) | 14 µs at 1.18 GB/s | 28 µs | ≈150–260 µs |
| 2.5 GbE, QUIC | ≈85–275 µs (half of the 0.17–0.55 ms ping RTT) | 56 µs at 294 MB/s | 111 µs | ≈215–410 µs |
| Wi-Fi 6, QUIC | 3–7 ms median, ≈250 ms p99 | 0.33 ms at 50 MB/s | 0.66 ms | ≈3.7–7.7 ms (p99 ≈ 250 ms) |
| Shared memory | ≈1–5 µs (spin wake) | ≈2 µs memcpy | ≈3 µs | ≈5–10 µs |

At batch-1 decode the hop is latency-bound everywhere except Wi-Fi's serialisation term. The F32 exact-mode payload adds 55 µs per hop on 2.5 GbE compared with BF16. Over example (c1)'s two hops that is 110 µs, or 0.07 % of its 153.8 ms per token, which is why exact mode costs essentially nothing at decode (ADR-0010).

**Frame overhead [derivation].**
- A decode frame carries 64 B of base header plus an extension of 64 B and one 32-byte span, which pads to 128 B. That is 192 B, or 256 B with side-channel data, so 0.6–0.8 % of a 32 KiB payload on the wire.
- Credits add 64 B per frame in the reverse direction.
- A 512-token prefill chunk at hidden 8192 in F32 is 16,777,216 B. That exceeds Thunderbolt 5's 16,773,120-byte message limit by 4 KiB, so on TB5 it travels as 2 parts (§6.3). Chunk sizes are plan decisions, and RFC-0011 may prefer 480 or 500 tokens on TB5 hops to avoid the split.

**Pinned memory per channel [derivation].** `(recv_slots + retention) × slot_size`.
- A decode Activations channel at batch 1: (4 + 4) × 36 KiB ≈ 288 KiB.
- A prefill-capable channel with 16 MiB slots: (3 + 3) × 16 MiB = 96 MiB per side.

The plan charges these to the `HostPinned` network sub-arena (RFC-0001 §9.1). The solver trades prefill chunk size against them on memory-tight nodes.

**Encryption cost [estimate].** AES-128-GCM with hardware AES is assumed to sustain ≥ 4 GB/s per core; the probe measures the real figure. At 10 GbE line rate (1.25 GB/s) that is under one-third of a core per direction, on the `net` core. At 2.5 GbE it is below 10 %. Relevant only for prefill and blob traffic.

**Credit round trips** never sit on the decode critical path. The sender starts with `recv_slots` credits, and a single stream consumes one per ring traversal.

## Alternatives considered

- **gRPC or HTTP/2 with protobuf messages for the data plane.** Per-message allocation, framing copies and an async runtime on the token path (ADR-0012). The star-topology and RPC-layer designs surveyed show what per-op RPC costs. Rejected.
- **FlatBuffers or Cap'n Proto for data frames.** Both are zero-copy with schema evolution. But neither guarantees 64-byte payload alignment or lets the header be built in place inside a registered DMA slot without a builder allocation. The data frames have few fields that change, so a fixed layout with additive minors is simpler to fuzz. Rejected for data frames. CBOR covers the control plane (ADR-0305).
- **Rely on the transport's reliability only.** Impossible on UC queue pairs. It would also force activation-history replay on every link failover, because a fresh connection loses in-flight frames. The retention window costs a few hundred KiB per decode channel. Rejected.
- **QUIC datagrams for decode frames.** These would avoid stream head-of-line blocking. But Syntropy would then re-implement ordering and retransmission, and on an uncongested LAN stream delivery has the same latency. Rejected for v1. Revisit if RFC-0061 shows a stream-induced p99 gap above 50 µs.
- **One QUIC connection per channel.** Separate congestion state per channel, but more handshakes and idle connections. A decode stream at tens of frames per second does not need its own controller. Rejected (ADR-0308).
- **RDMA WRITE-with-immediate into receiver-advertised slots.** Fewer receive work requests on Linux RC, but not available on Thunderbolt 5, which would split the code path. Deferred until Linux RoCE clusters show a measured gain.
- **Datacenter collective libraries across nodes.** Homogeneous, static membership, vendor-specific (ADR-0011). Rejected.
- **A per-frame CRC only on non-AEAD transports.** It saves about 8 µs per hop on QUIC and TCP, but loses detection of our own buffer-reuse bugs, and ADR-0011 item 5 makes CRC mandatory. Kept mandatory. Its cost is overlapped (§Performance).

## Testing and validation

| Claim | Verification |
|---|---|
| Parsers reject malformed input without panics or out-of-bounds reads | `cargo fuzz` targets per frame type, the TLV area, the token payload, the CBOR envelope and the beacon; corpus seeded from golden frames; run in CI per RFC-0060's fuzzing tier |
| Layouts match this document | golden-frame tests: byte arrays checked into `syntropy-net/tests/golden/` for every frame type, generated from the spec tables and decoded by the implementation; a change to any golden file needs a wire-minor bump |
| Exactly-once, in-order delivery under loss, duplication, reordering, corruption and failover | `testkit` simulated fabric (RFC-0060) with injected faults at configurable rates; property test: the received sequence equals the sent sequence for random fault schedules, including a mid-stream backend switch (QUIC → TCP, RDMA → QUIC); a UC loss schedule that drops **credits specifically** still delivers exactly once; golden frames for a resend (byte-identical) and an AEAD resend that never reuses a nonce with different associated data |
| No allocation or lock per frame on hot threads | the `testkit` counting allocator and lock instrumentation over 10,000 decode steps (ADR-0012 item 6) |
| RDMA verbs path | Linux CI with the kernel's software RoCE (`rdma_rxe`) on a loopback pair; nightly on a two-node RoCEv2 rig (Tier 2) |
| RDMA over TB5 path | two-Mac TB5 rig (RFC-0001 §16 M2 exit rig): QP error injection by unplugging the cable mid-stream must recover in-process (ADR-0011 revisit trigger); the cable pull is a **Maintainer action** on the lab rig (ADR-0021) |
| SC-12 per transport | RFC-0061 microbenchmark, **runner to runner through the real dispatcher → net-task rings**: GPU buffer → frame → wire → frame → GPU buffer at 64 B, `h_model × 2`, `h_model × 4` and 16 MiB; report Z against `2 · t_switch`, with the net task polling and parked |
| Bulk pacing keeps data-plane p99 within 1.2× idle | rig test: a 10 GB blob transfer on the same 2.5 GbE edge as a live decode ring; assert the hop's p99 |
| Mixed-version interop inside the §10 window | RFC-0062 §10.4's rolling-upgrade test: a 3-node emulated cluster on release N−1 upgraded node by node to the candidate under load, with a split and heal between releases, `exactly_once_tokens`, and the cluster level rising only after the last node; its golden-vector row decodes every frame version in the window. A node outside the declared ranges is refused with `ProtoRangeDisjoint` (unit test on `Hello`) |
| Parity tap layout and values | golden `Token` frames with Gather and FullLocal tap blocks, and golden `TapIds` frames, including non-finite values carried bit-exact; the values themselves are judged by RFC-0060's tests: the `KL_K` tightness check (RFC-0060 Testing item 3, §5.4) and the planted rare-row defect that only the tail sample catches (§5.4), both driven through this tap |
| Encryption on every IP path | integration test asserting `TransportCaps::encrypted` for QUIC and TCP, and packet captures in CI showing no plaintext activation bytes |

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M2 | `Transport` trait; QUIC (direct), TCP and shared-memory backends; RDMA over Thunderbolt 5 on macOS (RFC-0001 §16 M2); frame types `Activation`, `Token`, `SeqControl`, `Credit`, `Hello`, `HelloAck`, `Close`, `Ping`, `BlobRequest`, `BlobData`, `TapIds`; side-channel TLVs including `KV_HEADROOM`, `ROUTER_HINT` and the prefix-cache items (RFC-0022, RFC-0023), and the `TokenEntry` tap block for pipeline parity runs (RFC-0060); F32 and BF16 wire dtypes; credits, retention, NACK, reconnection and demotion; bulk pacing; control envelope (CBOR) and feature levels; golden frames and fuzz targets |
| M3 | Linux verbs (RoCEv2 behind `trusted_fabric`, InfiniBand); `Collective`, `StepDescriptor`, `EpDispatch`/`EpCombine` (cross-node TP and EP; on Mac pairs only if the raw-path probe passed, open question 1); `Recover` and `KvChunk` (RFC-0033, RFC-0022); `TENSOR_GROUP` peer reads (RFC-0023); relayed-QUIC costing for off-LAN members, with no weight chunks over a relay (§5.4); `prefill_transport`; Q8 and F8 wire containers for approximate mode; verify-tree spans |
| M4 | GPUDirect RDMA where probed; RDMA AEAD capability (ADR-0314); fuzz coverage gate for every frame type. The §10 window and declared ranges are tested from M2 by RFC-0062 §10.4 |

## Open questions

1. **α_eff on Thunderbolt 5 RDMA through this frame path** (RFC-0001 §17 question 3). SC-17 assumes the gap between the 10–25 µs wire latency and the back-solved 130–150 µs per collective is synchronisation and implementation, not the wire. *Split into two runs, with the rule fixed in advance:*
   - **A raw-path probe late in M1** (DIST-E7-T20) on a pair of TB5 Macs: R-MBP plus one Mac mini, or two minis, once the current mini is confirmed to have TB5 and to be on Apple's list of RDMA-capable models. It measures Metal buffer → shared-event wait → verbs SEND on a UC queue pair → shared event → Metal buffer at 64 B and 28 KiB, with no Syntropy codec, and reports the wire, doorbell and flag components against SC-17's 31.5–76.5 µs. The same session confirms the memory-region figure (question 4) and runs a one-day fault smoke test: cable pull, sleep and wake, peer reboot.
   - **The rule.** If the raw path's `α_eff` exceeds 65 µs, SC-17 becomes report-only at M3, and DIST-E13's Mac tensor-parallel work moves to M4 or after 1.0. If any smoke-test case needs a reboot to recover, TB5 RDMA ships opt-in at M2 (ADR-0309).
   - **The frame-path run in M2** (DIST-E7-T15), only if the raw path passes, measures GPU buffer → `Collective` frame → GPU buffer and sets the `α_eff` the SC-16 gate uses.
2. **QUIC versus TCP α on LAN.** ADR-0011 orders QUIC before TCP on the LAN and says to revisit if LAN QUIC's p50 α is more than 50 µs worse than kernel TCP's. Measure on the M2 rigs before freezing the rank order.
3. **Credit per frame.** A credit after every activation frame doubles packets per hop. If RFC-0061 shows a measurable α cost on Wi-Fi, credit every second frame on single-stream decode, where credit is never short.
4. **Thunderbolt RDMA memory-region limit.** *Answered (OQ-NET-18): not binding.* TB5 allows 10 UC queue pairs per device and each RDMA runner pair needs several, so queue pairs cap RDMA runners at about 2–3 per Mac long before a limit of about 100 regions would, at fewer than 8 regions per runner. The figure is confirmed in one line of the raw-path session (question 1).
5. **iroh transport knobs.** Whether the initial congestion window, stream receive windows and GSO are reachable through iroh's configuration API in the version pinned at M2. If they are not, the LAN prefill path loses some first-chunk throughput until upstream support lands. *Now a half-day desk check in M0 against the pinned version (DIST-E7-T21), rather than an M2 spike; DIST-E7-T16 measures what the knobs gain.* *Answered (OQ-NET-19), against iroh 1.4.1 (§4.3 note of 2026-10-23):*
   - **the initial window** is reachable through a congestion-controller factory, on `noq-proto`'s configs, and "RTT < 2 ms" is a wrapping controller of `net`'s;
   - **the stream receive window** is reachable per connection, with no per-stream setter;
   - **GSO** is on by default where noq-udp finds it, on Linux and Windows. On macOS it is absent: iroh never enables Apple's batched-send path, and no upstream issue tracks it.
6. **CRC and AEAD throughput on Raspberry Pi 5-class CPUs** (Tier 2). *Answered (OQ-NET-20).* Arm uses the hardware CRC32C instructions with 3-way interleaving, as x86 does. The Pi 5's Cortex-A76 has the Armv8 CRC32 extension; its CPU flags are checked in `/proc/cpuinfo` at R-PI5's bring-up. DIST-E7-T17 stays as a measurement of the resulting `Z` share, not a decision.

## Proposed amendments to RFC-0001

1. **"Fixed-size header" (§5.8).** RFC-0001 says the `ActivationFrame` byte layout "must keep the header fixed-size". This RFC uses a fixed 64-byte base header followed by a bounded extension. The extension holds the row spans, tree and side channel. Its size is bounded by the plan's maximum rows per frame times the per-row metadata (§8), not by a fixed constant, and its length is in the base header. A receiver therefore parses in two fixed steps, and the payload is still 64-byte aligned. Proposed wording: "a fixed-size base header and a bounded, length-prefixed extension; payload 64-byte aligned in a registered buffer."
2. **`TransportKind` gains `QuicRelay`**, so plans and metrics distinguish relayed paths, as ADR-0011's rank order already does. This is an extension under RFC-0001's rules. It is listed here because §5.8's comment enumerates the variants.
3. **`Channel::completed` semantics (§5.8).** "A completed send; its payload buffer may be reused" is specified here as *acknowledged by the peer*, not merely sent. This is required for Thunderbolt 5 UC queue pairs and for failover without replay. No signature change.
