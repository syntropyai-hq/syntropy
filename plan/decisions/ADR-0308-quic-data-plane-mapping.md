# ADR-0308: QUIC data plane: per-runner endpoints, one connection per pair, one stream per channel, no datagrams for data

- Status: Accepted
- Date: 2026-10-19
- Deciders: maintainers (drafted by the distributed runtime area)
- Related: ADR-0011, ADR-0012, ADR-0015, RFC-0001 §3.2 and §8.1, RFC-0031 §4, RFC-0034 §6

## Context

ADR-0011 makes QUIC on iroh the data-plane backend for LAN and off-LAN edges that have no RDMA. RFC-0001 §3.2 has two consequences here:
- the daemon owns the node key and the iroh endpoint for control and bulk;
- runners own every stage-to-stage channel, and receive only a delegated, epoch-scoped credential.

The open questions are three:
- how runners get QUIC identities;
- how channels map onto connections and streams;
- whether unreliable datagrams should carry decode frames.

Decode frames are 8–32 KiB at tens of frames per second per edge. Prefill frames are MiB-scale. Collectives on QUIC run only where links pass the TP/EP gate (ADR-0006).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Runners use the daemon's endpoint through IPC | One identity | Every frame crosses the daemon process: violates ADR-0015's "token path never crosses a process boundary" |
| B. Runners hold the node key | Simple | Node secret in a crash-prone process that parses model files; violates RFC-0001 §3.2 |
| **C. Each runner has its own iroh endpoint with an ephemeral Ed25519 key, certified by a node-signed `DataPlaneCredential`; one connection per runner pair; one bidirectional stream per channel; datagrams not used for data** | Token path stays inside runners; node key stays in the daemon; per-channel head-of-line isolation; one handshake per pair | Credential verification on every channel open; ephemeral keys must be rotated |
| D. One connection per channel | Separate congestion state per channel | More handshakes; little benefit at decode rates |
| E. Datagrams for decode frames | No stream head-of-line blocking | Must rebuild ordering and retransmission; no latency gain on uncongested LANs |

## Decision

**Option C** (RFC-0031 §4, RFC-0034 §6).
1. At spawn, the daemon gives each runner an ephemeral key pair and a `DataPlaneCredential`: runner key, node ID, instance, starting epoch and expiry, signed by the node key. The runner's iroh endpoint uses that key.
2. Peers verify each channel `Hello`:
   - the TLS peer key equals the credential's runner key;
   - the signature verifies under the node ID;
   - the node is a non-revoked member that the plan puts at the expected stage;
   - the epoch is current.
3. There is one QUIC connection per runner pair per ALPN, `syntropy/data/1`, and one bidirectional stream per channel: Activations, Tokens, Collective lanes, Replay lanes. Credit frames return on the same stream. Stream priority follows the channel's priority class.
4. QUIC datagrams carry only membership beats on daemon control connections, never data-plane frames.
5. The QUIC runtime is a single-threaded tokio runtime on a reserved core. It meets hot threads only through SPSC rings (ADR-0012).

**Revised 2026-10-19:**
- **Two QUIC connections per runner pair**:
  - the latency connection carries the decode lane, the prefill lane, ring-return tokens and collectives;
  - the bulk connection carries replay, KV migration and tensor-group reads, paced by the p99 rule.
- **Connection flow-control windows** equal the sum of credited bytes.
- **Two lanes per Activations hop.** Each hop has a decode lane and a prefill lane (RFC-0032 §3.2), and stages take decode frames first.
- **Net-task polling**. The runner's net task busy-polls its rings while a `Ready` instance has live sequences, with a pre-wake, so the dispatcher hand-off costs a cache-line transfer, not a cold wake.

## Consequences

- Positive: a runner crash never exposes the node key; a decode stream never queues behind a prefill or replay stream; connection state stays small.
- Negative:
  - Credentials need renewal: 24 h expiry, renewed by the daemon over IPC.
  - Ring neighbours exchange runner endpoint addresses through their daemons at every epoch (RFC-0030 §8.3).
- Work created:
  - RFC-0034: credential format and revocation push to runners;
  - RFC-0030: the `RunnerEndpointOffer` RPC.

## What would make us revisit

- RFC-0061 shows stream delivery adding more than 50 µs at p99 over datagram delivery on LAN for decode frames.
- iroh gains a supported way to derive per-process sub-identities from one endpoint key with equivalent isolation.
