# Multi-device inference: techniques, measurements and failure modes

Research input for the Syntropy plan. Survey date: 2026-10-19.

Scope: running one model's inference across several networked devices, with extra attention to designs that also use RAM or disk inside each device. Single-host tiered offload is covered in `tiered-memory-inference.md`, and the hardware numbers and cost equations are in `hardware-and-performance-model.md`. This note is organised by technique. Sources are described by what they are (an event-sourced cluster runtime, a published home-cluster study, a datacenter serving stack), not by name.

## How to read this

Every load-bearing claim carries a tag:

- **[verified: code]**: read in the source of a surveyed system. Clones were shallow and read-only; nothing was built or run.
- **[verified: paper/docs]**: stated in a surveyed system's paper, README or official docs. The tag means the claim was made, not that it is true.
- **[reported by a surveyed system]**: a performance number from a surveyed system's own paper, blog or benchmark thread. Test conditions are given with it.
- **[reported: third-party review]**: a number from an independent hardware review.
- **[inferred]**: our reasoning from code or docs. Anything computed is labelled **Estimate** and shows its arithmetic.

Abbreviations: PP = pipeline parallelism, TP = tensor parallelism, EP = expert parallelism, P/D = prefill/decode, TPOT = time per output token, TTFT = time to first token. KiB/MiB/GiB are binary; link rates are line rates.

---

## 0. Summary

1. **On home links PP wins and TP loses, unless the link is RDMA-class.** TP over Wi-Fi measured about 150 ms per all-reduce, which puts a 70B model at about 24 s/token. TP over 10GbE spent 34 ms/token in transfer at 4 nodes on 70B. On Thunderbolt 5 with RDMA, 2-node TP decoded 13.69 tok/s against 7.70 for 2-node PP on a 128B dense model. At batch 1, PP adds capacity and no decode speed (§4).
2. **Placement quality dominates everything else at home scale.** A per-device ILP over layer windows and GPU/CPU splits was 3.3× faster than a memory-proportional split on a 70B model. Without it the same model ran at 20,848 ms/token; with it, 674 ms. The solve takes 10–12 ms. The open-source release of that solver sets the communication term to zero (§3.3).
3. **Most runtimes size placement on free system RAM alone.** They have no VRAM query, no KV or activation reserve, and measure neither compute nor links, even where their docs say they do (§3.8, §13).
4. **The one shipped multi-device design with per-device VRAM/RAM/disk tiering uses the OS page cache as its residency manager.** Residency therefore follows kernel LRU heuristics, which differ per OS (§8).
5. **The 2026 runtimes are the best reference for networking and UX:** QUIC with public-key identity, invite tokens, per-stage layer packages, and a single ~18 MB binary with installers for all three desktop OSes. None of them has tier-aware placement or mid-generation recovery (§1, §7, §12).
6. **Mid-generation recovery exists in two forms.** One is activation-history replay to a replacement stage, implemented only by a 2023 internet-scale pipeline swarm. The other is request migration by re-prefill, used by datacenter stacks. Every home-cluster runtime aborts or stalls when a stage dies (§10).
7. **Some control planes route every generated token through a central leader, JSON-encoded, and broadcast it to all nodes.** Under load this has collapsed clusters (§2.3, §13).
8. **The opening:** zero-config discovery; a three-tier-per-device placement solver with device selection and a real network term; MoE-expert-aware tiering; a direct, encrypted stage-to-stage transport; mid-generation recovery; and one binary for macOS, Linux and Windows. No surveyed system does more than three of these (§14).

---

## 1. Cluster membership and discovery

### 1.1 LAN discovery mechanisms observed

| Mechanism | How it works | Notes |
|---|---|---|
| IPv6 multicast hello/reply | A transient, link-local-scope group (`ff12::/16`) is joined on every interface that has a non-loopback IPv6 address, with interface changes tracked. Every 1 s a node sends `Hello{nonce, namespace[8]}` with `namespace = blake3(namespace_string)[..8]`. Peers in the same namespace reply unicast `{nonce, peer_id[16], pubsub_port}`. Replies to any of the last 8 nonces are accepted. **[verified: code]** | Confined to one L2 multicast domain. Tailscale does not carry multicast. |
| IPv4 UDP broadcast | Subnet broadcast plus 255.255.255.255, 2.5 s interval, 30 s timeout **[verified: code]** | Earlier generation of the same runtime; it also had Tailscale-API and manual-JSON-topology discovery, with node-id and interface-type filters. |
| UDP broadcast query, JSON reply | The coordinator broadcasts a query carrying a 4-byte magic on a fixed port. Workers answer with JSON listing name, port, GPUs, backend and OS. Default timeout is 10 s; with no workers found, the coordinator runs alone **[verified: code]** | The project's docs say mDNS; the code uses broadcast. |
| mDNS plus libp2p DHT | A shared token defines a private network with a gossip ledger, carried on a libp2p-based overlay VPN. The DHT is on by default; turning it off leaves mDNS only. Custom bootstrap multiaddrs are supported **[verified: paper/docs]** | Gives zero-config across NATs. |
| Kademlia DHT | Runs over a bundled libp2p daemon shipped as a downloaded binary. Servers announce the block spans they host and their measured throughput **[verified: code + paper/docs]** | Built for an internet-scale public swarm. |
| QUIC endpoints with public-key identity | Every node is an iroh endpoint: QUIC hole punching with relay fallback. Private meshes join by **invite token**; public meshes advertise over Nostr **[verified: paper/docs]** | The cleanest model surveyed. |
| P2P library with DHT and relays | libp2p-style multiaddrs, bootstrap peers, relay servers, DCUtR hole punching, a DHT with an announce prefix **[verified: code]** | |
| Manual | `--world N --rank r --master <ip> --next <ip>`, with two port ranges offset by rank that must be open; or a root given `--workers ip:port ...`; or a JSON topology file **[verified: code]** | The most common setup among home-cluster designs. |
| SSH plus hostfile | `mlx.launch` uses SSH and a JSON hostfile (`ssh`, `ips`, `rdma` per rank). `mlx.distributed_config` can discover a Thunderbolt ring and write the hostfile; `--auto-setup` needs passwordless sudo ([MLX distributed docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)) **[verified: paper/docs]** | Framework-level, not a membership service. |

### 1.2 Connection setup and liveness in one multicast-based runtime

All **[verified: code]**.

- zenoh runs in router mode with multicast scouting off and **gossip scouting with multihop on**. On discovery, only the peer with the greater id dials, which avoids duplicate sessions; slow dials run in a spawned task.
- Liveness uses a zenoh liveliness token per node. A wildcard subscriber turns token put/delete into connect and disconnect events.
- **Stall guard.** If the 1 s discovery tick fires ≥ 5 s late (laptop sleep or suspend), discovery ignores peers for 12 s so that stale sessions (10 s lease) close before re-linking. This prevents one-way splits.
- **Topology edges come from a separate HTTP probe.** Every 10 s each worker GETs a node-id endpoint on every IP that every other node advertises (3 attempts, 5 s timeout) and emits edge-created or edge-deleted events per reachable IP. An edge therefore means "TCP reachable on this IP", not a measured link quality.
- **Version-hashed namespace.** The default namespace is the version string, so mixed versions never cluster. A log line now says so. When the package version and the release label differ, which string the shipped app uses decides interop.
- A namespace flag isolates clusters on the same LAN.
- Ports: API and pub/sub on fixed TCP ports, discovery on a fixed UDP port. Collective rendezvous (TCP ring and RDMA coordinator) uses a random ephemeral port per model instance, and the P/D prefill server port is recorded in cluster state.

### 1.3 Identity, static peers and remote membership

- **Ephemeral identity.** One runtime generates a random 128-bit node id at every process start; persistence is commented out "once we figure out how to deal with duplicates" **[verified: code]**. No node keeps a stable identity across restarts.
- **Static peers removed.** The same runtime's bootstrap-peers flag makes the process refuse to start ("temporarily removed"), and its rewrite dropped the earlier Tailscale and manual-JSON modes **[verified: code]**. Clusters are limited to one L2 multicast domain, so routed, VPN and cloud setups cannot join. A maintainer suggested Tailscale support may need "manual configuration overrides" **[verified: paper/docs]**.
- **Cross-LAN through a proxy.** The home-cluster study builds cross-LAN rings by inserting a zero-work, VPN-connected proxy device into the ring **[verified: paper/docs]**.
- **Public-key identity.** In the 2026 QUIC runtime a node's identity is its public key, private meshes join by invite token, and relay plus hole punching handle NATs **[verified: paper/docs]**.
- **Token leakage.** In the token-gated libp2p federation, publishing a swarm on the public explorer exposes its token, and containers need host networking **[verified: paper/docs]**.

### 1.4 Authentication and encryption

| Approach | Where seen |
|---|---|
| None: no auth, no encryption | The home-cluster ring study, the TP-only CPU runtime, and the remote-backend RPC layer of a ggml-based runtime, whose README says "Never run the RPC server on an open network" **[verified: code + paper/docs]** |
| Mutual HMAC-SHA256 challenge-response from a cluster key before any protocol data; only a SHA-256 prefix of the key is advertised; **traffic not encrypted** | A Rust star-topology runtime **[verified: paper/docs]** |
| Shared-token private libp2p network | The libp2p federation layer **[verified: paper/docs]** |
| QUIC, end-to-end encrypted, activations included | The 2026 QUIC runtime **[verified: paper/docs]** |
| Unencrypted; docs say to bind to a private address | Datacenter serving engines **[verified: paper/docs]** |

### 1.5 Failure-detection timings

| Setting | Value | Where |
|---|---|---|
| Node silence before the leader times the node out | 15 s (heartbeat every 5 s) | event-sourced cluster runtime **[verified: code]** |
| Leader silence before an election | 30 s | same **[verified: code]** |
| Default heartbeat timeout | 30 s, tuned for volunteer WANs | volunteer-GPU pipeline scheduler **[verified: code]** |
| Decode → prefill heartbeat | 5 s; decode aborts on transfer timeout | datacenter P/D deployment guide **[verified: paper/docs]** |
| Peer-down gossip | grace period, then withdraw the split | 2026 QUIC runtime **[verified: paper/docs]** |
| Socket receive timeout | 1000 ms on metadata receive; 3.5 s linger during topology rebuild | home-cluster study **[verified: code]** |

On a LAN, a 1–2 s heartbeat is enough to notice a failure within one or two tokens **[inferred]**.

### 1.6 Host-network side effects

- **Thunderbolt Bridge loops cause broadcast storms.** One runtime detects cycles of bridge-enabled nodes. Its macOS app installs a dedicated network location that disables the bridge and sets DHCP on each Thunderbolt port **[verified: code]**. Whether uninstall reverts this in all cases was not verified.
- **macOS Local Network permission** blocks discovery when denied, and that has been a known failure **[verified: paper/docs]**.
- Users have reported partial discovery: only 3 of 4 TB5-connected Mac minis found, and one-way visibility **[verified: paper/docs]**.

---

## 2. Control-plane designs

### 2.1 Event-sourced single-writer leader

One cluster runtime runs this process tree on every node **[verified: code]**:

```
node process (one per machine, asyncio event loop, multiprocessing start method "spawn")
├── Router          zenoh pub/sub through Rust bindings (tokio runtime inside the Python process)
├── EventRouter     orders leader-indexed events for local consumers; NACK/resend; snapshots
├── Election        bully-style leader election on a high-priority topic
├── Leader          every node constructs one; promoted or demoted by election results
├── Worker          per-node planner; owns runners; gathers memory, NICs, Thunderbolt, RDMA state
├── Downloads       Hugging Face Hub downloads (can be disabled)
├── HTTP API        also serves the web dashboard
└── Runner subprocesses   one per (model instance, node): framework, weights, KV cache, generation
        ↕ multiprocessing channels (tasks in, events out, cancel ids)
```

Worker, API and downloads are optional, so a coordinator-only node is possible.

**Topics** (all **[verified: code]**):

| Topic | Payload | Direction |
|---|---|---|
| global events | leader-indexed events | leader → all |
| local events | node-originated events | any node → leader |
| state snapshots | full state | leader → laggard |
| commands | API and worker commands | → leader |
| election | election messages, high priority | all ↔ all |
| connection | liveliness put/delete (local only) | router → election |
| download commands | download requests | leader/API → nodes |

**Mechanics** (all **[verified: code]**):

- **Serialisation.** Every control message is a pydantic model encoded as JSON. A payload ≥ 1 MiB logs a warning.
- **Congestion.** Publishers drop on congestion; receivers use a bounded 16,384-message queue that drops when full. Loss is recovered by NACK and resend. A code comment estimates "~1,000 events/s of a loaded cluster".
- **Ordering.** Nodes emit `{origin_id, origin_idx, session}`. The leader merges per-origin sequences, assigns `idx = last_applied + 1`, applies `apply(state, event)` and rebroadcasts. Followers apply in order and NACK gaps with an event-log request. Unacknowledged local events are resent after 5 s by a 1–2 s retry loop.
- **Catch-up.** The leader keeps the last 10,000 events (capped at 256 MiB of image data). A node further behind gets a full snapshot. Snapshot catch-up was retrofitted late, after clusters with late joiners broke.
- **Sessions.** `SessionId(leader_id, election_clock)` tags every event. On leader change, followers drop old-session events and wait for the new leader's snapshot.
- **State.** A frozen model holding instances, runners, downloads, tasks, last-seen times, the topology, per-node maps (identity, memory, disk, system, network, Thunderbolt, Thunderbolt bridge, RDMA control, backends), bridge cycles, P/D instance links, prefill server ports, custom model cards and the last applied index. The whole state is served as JSON over HTTP, which makes the cluster debuggable from one request.
- **Pure planning.** `plan()` on each worker returns at most one task per tick, in priority order: cancel, shutdown, create runner, download, connect to group, load, warm up, then forward pending generation. Rank ordering is encoded there: rank `n−1` connects only after all others are connecting, and rank 0 warms up last. This avoids collective deadlocks at bring-up.
- **Runner lifecycle.** `Idle → Connecting → Connected → Loading → Loaded → WarmingUp → Ready ↔ Running → ShuttingDown → Shutdown`, plus `Failed`.
- **Supervision.** The supervisor kills stuck runners: prefill timeout 60 s, decode timeout 5 s, stall timeout 90 s.
- **Testability.** Placement, `apply` and election are pure functions and are unit-tested as such. Strict type checking, frozen models and effect-handler-style pure functions make the code easy to verify.

### 2.2 Bully-style leader election

All **[verified: code]**.

- Election messages sort by `(clock, seniority, commands_seen, node_id)`. A campaign broadcasts its status, sleeps 3.0 s, rebroadcasts, then picks the maximum. The winner's seniority becomes `max(seniority, number of candidates)`. A force-leader flag sets seniority to 1,000,000.
- **Triggers:**
  - every connection change: `clock += 1` and a new campaign after a 0.2 s debounce
  - a higher clock in an incoming message
  - a heartbeat from a rival leader
  - leader silence > 30 s (heartbeats every 5 s)

  Elections triggered by heartbeats or silence back off exponentially from 15 s to 300 s.
- **Anti-flap.** If a strict majority of the round's candidates follow a leader that is itself standing, that leader stays. This stops a returning senior leader from forcing a second reset.
- **Takeover.** The new leader starts from its own last-known state minus in-flight generation tasks. Model instances survive, but **in-flight requests fail** because the API ends them. A docstring still said "every instance is stopped", a stale comment left from before the takeover change.

### 2.3 The request path in that design, and why it hurts

All **[verified: code]** unless tagged.

1. A client POSTs to **any** node's HTTP API, and an adapter converts the request to an internal task.
2. If no instance of the model exists, the API returns 404. **Requests never trigger placement**; a model must be placed beforehand from the dashboard or with curl.
3. The API publishes a generation command. It resends the command if it sees no acceptance, and the leader dedups by command id.
4. The leader picks the matching, non-prefill-only instance with the fewest in-flight tasks, attaches a prefill endpoint if the instance is P/D-linked, and emits `TaskCreated`.
5. Every worker applies `TaskCreated`. On each node of that instance, `plan()` forwards the task to its runner once all runners are ready or running.
6. Runners agree task order across ranks with an all-gather of task ids, so all ranks enqueue identically; this prevents TP collective deadlocks. They then prefill (locally or remotely) and decode with continuous batching, up to 8 concurrent requests.
7. Only rank 0 emits token events. The other ranks compute the same token, using the same seed (42) and the all-gathered final hidden state, and drop it.
8. Each token event goes runner → supervisor (multiprocessing channel) → node event router → local-events topic → **leader indexes it** → global-events topic **to every node** → the API node filters by command id and streams SSE.
9. On finish the API reports completion and the leader deletes the task.

**Consequence [inferred].** Each token crosses the pub/sub plane twice, JSON-encoded, through the leader, and is delivered to every node. A code comment says parsing these "took about half of a follower's time under load". Per-token latency includes a leader hop even when the API node, rank 0 and the leader are different machines, and throughput is capped by the leader's event loop (code comments cite 800–1,000 events/s).

### 2.4 Head-driven ring with no separate control plane

All **[verified: code]**.

- In the home-cluster study, device profiles travel around the ring to rank 0. Rank 0 solves placement and broadcasts the layer setup, and the ring is rewired around devices that receive no layers; those exit with "No layer is assigned to me".
- KV-cache operations (clear, sequence remove/add/copy/keep/divide) travel the ring as metadata so every stage keeps its KV shard consistent.
- The head does tokenisation, sampling and output. The last stage sends back to rank 0.
- There is no discovery, authentication or encryption.

### 2.5 Leaderless, client-driven and scheduler-driven designs

- **DHT plus client routing** (2023 internet swarm) **[verified: paper/docs]**. Servers announce block spans and throughput in a DHT. The client holds embeddings and the LM head and computes a shortest path over (server compute time + network latency), using D* Lite so the path can be patched quickly when servers are banned.
- **Gossip mesh** (2026 QUIC runtime) **[verified: paper/docs]**. Membership spreads by gossip with a peer-down message and a grace period. The runtime is single-machine first: if one node fits the model it serves locally; otherwise requests route by model name to a peer that has it, or use a stage split. Stage 0 becomes routable only when all stages are ready.
- **Two-phase scheduling** (2025 volunteer-GPU scheduler) **[verified: paper/docs]**. A slow model-allocation phase decides which GPUs host which layers. A fast per-request routing phase picks a path through them (§3.5).
- **Filter → score** (a GPU cluster manager) **[verified: paper/docs]**:
  - A filter chain (cluster, GPU match, labels, READY state, backend compatibility, local path) feeds a resource-fit stage.
  - Resource fit stops at the first strategy that yields candidates: single GPU → multi-GPU on one worker → distributed across workers → partial offload or CPU.
  - Scoring is binpack or spread, plus a model-file locality score. GGUF requirements come from parsing the file's metadata.
  - Distributed execution is delegated to datacenter serving engines. Limits: model files must sit at the same path on every worker; the actor-framework mode allows one distributed instance per worker; auto-scheduling needs the same GPU count on every worker; one engine's distributed mode is Linux-only.
- **Federation by replication** (the libp2p federation layer) **[verified: paper/docs]**. In federated mode each request goes to one node chosen by a load balancer: replication, no model split. Its worker mode is the remote-backend RPC layer ("weights are split proportional to the memory"), for models that runtime supports only. Its docs call it "still experimental" and "tech preview quality", and "Only a single model is supported currently".
- **Datacenter disaggregated stacks** **[verified: paper/docs]**. Discovery is Kubernetes-native (CRDs, EndpointSlices), file-based locally, or etcd/NATS under Slurm. An SLA planner sizes pools, and a KV-aware router places requests.

### 2.6 Star versus ring data paths

- **Client-holds-the-graph star.** In the remote-backend RPC layer, each remote host runs a server exposing its local kernel backends (CUDA, Metal, Vulkan, CPU) as remote devices, and the client runs the whole graph scheduler. Copying a tensor between two different servers returns false ("falling back to a host copy"), so activations between two remote stages go server → client → server **[verified: code]**. That doubles hop latency at every stage boundary **[inferred]**. This RPC layer is the substrate under the federation layer's worker mode, under one cluster manager's older GGUF path and under many hobby setups.
- **Coordinator-forward star.** In the Rust star-topology runtime, the coordinator sends a single-op or batched-layers request to a worker and waits for the tensor back, so every stage boundary returns to the coordinator. Contiguous layers on one worker are batched into one message **[verified: code]**.
- **Ring.** In the home-cluster study, stages send directly to the next stage and the last stage sends to rank 0 **[verified: code]**.
- **Ring with token broadcast.** The earlier generation of the event-sourced runtime used a gRPC node service (send prompt, send tensor, send example, collect topology, send result, opaque status, health check), with tensors sent as `bytes + shape + dtype`. Prompts and tensors were pushed node to node around the ring. The node holding the last layer sampled, forwarded the token back to partition 0 for the next step, and broadcast it to all peers. At most 2 KV caches were held per request **[verified: code]**.
- **Full mesh.** In the TP-only CPU runtime, every node writes its slice to every other node's socket (a full-mesh all-gather), with a root sync at the start of each token and a logits gather at the end **[verified: code]**.

### 2.7 Data structures worth keeping

All **[verified: code]**.

- **Topology as a typed directed multigraph.** One socket edge per reachable IP of the sink (multiaddr), plus RDMA edges `(source_iface, sink_iface)` derived from Thunderbolt hardware data. RDMA edges exist only when both ends report RDMA enabled.
- **Cycle** = ordered list of node ids; the order becomes rank order.
- **Model card fields:**
  - sizing and shape: model id, storage size, layer count, hidden size, KV-head count, supports-tensor flag
  - capabilities: tasks, components, backends (Metal, CUDA, CPU)
  - runtime: context length, CFG use, trust-remote-code, vision, sampling defaults

  Example: Llama-3.3-70B 4-bit, 80 layers, hidden size 8192, 40,652,242,944 B. Around 120 built-in cards ship as TOML, covering Llama 3.x, Nemotron, Qwen3/3.5/3.6 (including Coder and Next), Qwen3-VL, DeepSeek V3.1/V3.2/V4, Kimi K2.x, GLM 4.5–5.1, MiniMax M2.x, Step-3.5-Flash, Gemma 4 and gpt-oss, at quantizations from 3-bit to bf16 including MXFP4 and NVFP4. Custom models are added by fetching `config.json` and the safetensors size from Hugging Face Hub; custom cards get trust-remote-code off, while built-in cards default to on.
- **Shards:** pipeline | tensor | CFG. Each carries rank, world size, a half-open `[start_layer, end_layer)` and the layer count. Tensor shards always cover all layers.
- **Instance variants:** TCP ring (hosts per node, ephemeral port) and RDMA (an N×N matrix of RDMA interface names plus coordinator addresses). A P/D link lists prefill instances and decode instances.

---

## 3. Placement and partitioning

### 3.1 Proportional heuristics

- **Memory-proportional, contiguous.** Sort nodes by memory (descending) and give each a contiguous fraction `[start, end)` of `[0, 1)` proportional to its memory, then map fractions to layer ranges **[verified: code]**. The careful version:
  - allocate layers in proportion to free RAM with largest-remainder rounding and at least 1 layer per node
  - check each node: `storage × layers_i / n_layers ≤ ram_available_i`, else error
  - subtract the bytes of runners already placed but not yet loaded
  - shard bytes = `storage × layers/n_layers` for PP, `storage / world_size` for TP

  The remote-backend RPC layer uses the same default ("in proportion to each device's available memory", overridable with a manual split), and so does the libp2p federation layer that sits on it **[verified: paper/docs]**.
- **Compute-proportional with OOM migration**, used by an earlier edge system **[verified: paper/docs]**.
- **"TFLOPS"-proportional, capped by VRAM.** The TFLOPS figure is made up: for NVIDIA, `vram_gb × clock_GHz × 1.5` (or `vram_gb × 3`) from `nvidia-smi`; for anything else, system RAM in GB × 0.4, or a constant 2.0 **[verified: code]**.
- **Static FLOPS table** keyed by chip name, taken from a public CPU-spec site. It fed a terminal UI only, not partitioning **[verified: code]**.
- **Uniform TP slicing.** The constraints seen are: node count must be a power of two and at most the KV-head count; or `hidden_size % n == 0` and `kv_heads % n == 0`, next to a code comment saying the condition "is not correct, but it works good enough for now" **[verified: code]**. Some quantized models cannot be split, and 3-node TP rarely works **[verified: paper/docs]**.
- **Failure mode.** Weighting by RAM gives a slow, RAM-rich device more layers, and it becomes the bottleneck **[inferred from code]**. The home-cluster study measured memory-proportional as 3.3× slower than its solver at 70B (§3.3).

### 3.2 Cycle enumeration

One cluster runtime places a model like this (all **[verified: code]**):

1. **Memory accounting:** subtract bytes of runners placed but not yet loaded.
2. **Candidate groups** = every simple directed cycle of the topology graph (`rustworkx.simple_cycles`) plus every singleton.
3. **Filters:** cycle length ≥ minimum nodes; required nodes ⊆ cycle; **sum** of available RAM ≥ model storage size.
4. **TP filter:** the card supports tensor, `hidden_size % n == 0` and `kv_heads % n == 0`, with one architecture exempt.
5. **Model-specific vetoes:** DeepSeek-V3.1-8bit cannot use PP; Gemma 4 PP is allowed only on a single node.
6. **Fewest nodes wins:** keep only the smallest qualifying cycles.
7. **Backend filter:** every node needs a backend allowed by the transport. TCP ring = {Metal, CUDA, CPU}; RDMA = {Metal} only.
8. **RDMA check:** every pair of cycle members that are graph neighbours has at least one RDMA edge, and every node has RDMA enabled. The device-matrix builder then demands **all-to-all** RDMA anyway.
9. **Tie-breaks:** prefer cycles containing a leaf node (≤ 1 neighbour), then maximise `(Σ download fraction, Σ free RAM)`.
10. **Single node:** forced to PP over the TCP ring.
11. **Shard assignment:**
    - PP: proportional to free RAM, as in §3.1; rank = position in the cycle.
    - TP: every node gets all layers and 1/N of every sharded weight. **There is no per-node memory check**; only the cycle sum from step 3.
    - CFG-parallel (image models with classifier-free guidance, even world size): two pipelines in mirrored order, so the two last stages are ring neighbours.
12. **Communicator config:**
    - TCP ring: a per-node hostfile in which only left and right neighbours are real; non-neighbours get the placeholder `198.51.100.1:0` (RFC 5737 documentation range). Interface priority: Thunderbolt > probable Ethernet > Ethernet > Wi-Fi > unknown.
    - RDMA: an N×N interface matrix plus a coordinator `ip:port` on rank 0, chosen with priority Ethernet > Wi-Fi > unknown > probable Ethernet > Thunderbolt.

A preview endpoint enumerates every valid (sharding × transport) placement with per-node memory deltas, and the user picks one in the dashboard **[verified: code + paper/docs]**.

**Scalability [inferred].** `simple_cycles` on a fully connected digraph of `n` nodes yields `Σₖ₌₂ⁿ C(n,k)(k−1)!` directed cycles:

| n | Cycles |
|---|---|
| 4 | 20 |
| 6 | 409 |
| 8 | 16,064 |
| 10 | 1,112,073 |
| 12 | 1.19 × 10⁸ |

The enumeration runs in Python on every placement and every preview call. The topology is a multigraph with one edge per IP, which may add work depending on how rustworkx handles parallel edges (not verified). Beyond about 10 well-connected nodes it becomes impractical; the runtime in practice targets 2–8 Macs.

### 3.3 Layer-window ILP with tiers and device selection

This comes from a published home-cluster study (2025 preprint, 2026 conference paper) with an open-source implementation on a ggml-based runtime.

**Formulation [verified: paper/docs]:**

- Decision variables per device `m`: `w_m`, the layer window, and `n_m`, the GPU-resident layers inside the window. The remaining `w_m − n_m` layers are CPU-resident and backed by mmap, so they may stream from disk.
- The objective minimises a TPOT model that sums, per device:
  - compute: FLOPs per layer per quant type ÷ profiled FLOPS per backend per quant type
  - memory access: KV copy, RAM↔VRAM copy, register-load bandwidth
  - disk reload: bytes beyond available RAM ÷ profiled disk throughput, with an OS-specific formula
  - communication: `t_comm` per hop × number of windows
- It minimises the **sum** over devices rather than the max, because the authors could not predict when the OS starts prefetching. They call this a worst-case assumption.
- The full problem is an integer linear-fractional program (NP-hard), because `W = Σ w_m` sits in the denominator.
- Two steps make it tractable:
  1. Enumerate the divisors `k` of `L` (at most 11 for `L ≤ 100`). Each fixed `k` gives a small ILP.
  2. Iterate the device → set assignment (four sets, by which devices overload RAM) to a fixed point. A calibration step forces the slowest-disk device into "no overload" when a GPU sits under-used.
- Reported solve time: 10–12 ms on 4–32 devices **[reported by a surveyed system]**.
- **Device selection.** Any device assigned exactly one layer and no GPU layers is dropped and the problem re-solved. A dropped device can stay in the ring as a zero-work forwarder if it is the only route.

**Released code [verified: code]:**

- The solver is HiGHS, linked only into rank 0's build.
- `2M` integer variables and `1 + 3M` constraints: `Σw = W`; `n ≤ w`; one RAM constraint per device, signed by set membership; one VRAM constraint per device.
- A priority factor scales the head device's cost.
- **Discrepancy 1: the network term is zero.** The code sets the per-device communication coefficient to 0.0 with the comment "the ram-vram and vram-ram transfer time and the communication time are less than 1 ms". The released objective has no communication or RTT term. The paper explains one counter-intuitive decision (dropping a 1080 Ti node) by "an extra RTT per token on the high-latency Wi-Fi ring"; in the released code that decision can only come from the one-layer drop rule **[inferred]**.
- **Discrepancy 2: the memory-bandwidth term is commented out** (the paper's register-load term).
- Per-device arrays are fixed at 32 entries.
- Without the solver compiled in, the fallback is memory-proportional.
- **Profiler:**
  - times quantized matmuls per backend and per quant type, memory and VRAM bandwidth, and KV and host↔device copies
  - measures disk throughput by **shelling out to an external `fio` binary**, writing a temp config into the working directory
  - reads cgroup limits so containers report correctly

**Heterogeneity inputs [verified: paper/docs].** The profiles cover CPU, CUDA and Metal FLOPS for Q2K–Q8_0, IQ1 and IQ4NL, plus disk read speed (sequential versus random, by OS), available RAM, swap (Android), Metal's recommended working set and unified memory. The resulting preference order is "powerful GPUs > weak GPUs > powerful CPUs > fast disks". Backends are CUDA, Metal and CPU only (no Vulkan or AMD), the code asserts it is not on Windows, and phones run under Termux.

**Results against heuristics [reported by a surveyed system]:**

- Llama 3-70B: 3.3× faster than memory-proportional and 1.6× faster than compute-proportional with OOM migration.
- 8–45B models: 1.1–1.5× over compute-proportional.
- 10-node testbed: 146 ± 5 ms/token against 156 ± 7 ms for "fill the fastest GPUs first". The solver used only three of the ten devices.
- **Partitioning dominates.** 70B TPOT without the solver was 20,848 ms; with the solver but no prefetch, 755 ms; with both, 674 ms. Partitioning matters about an order of magnitude more than prefetch, which is worth 9–17%.

**Testbeds and end-to-end results [reported by a surveyed system].**

Main testbed: four consumer devices whose combined RAM + VRAM (37 GiB) cannot hold a 70B Q4K model. All runs Q4K over Wi-Fi at 320–610 Mb/s with 3–7 ms link latency.

| Device | Hardware | Available memory | Disk read |
|---|---|---|---|
| D1 | Mac M1 | 2.4 GiB | 0.7 GB/s |
| D2 | Linux i9 + RTX 3070 | 4.1 GiB RAM + 8 GiB VRAM | 3.0 GB/s |
| D3 | Linux i9 + RTX 2080 Ti | 9.7 GiB RAM + 11 GiB VRAM | 3.0 GB/s |
| D4 | Huawei Mate 40 Pro (Termux) | 1.9 GiB | 1.4 GB/s |

| Model | Single-device baseline on D3 (mmap), TPOT | Memory-proportional cluster runtime | TP-only CPU runtime | Layer-window ILP + multi-round ring, TPOT / TTFT |
|---|---|---|---|---|
| Llama 3-8B | 15 ms | 263 ms | 1150 ms | 15 / 18 ms (all on D3) |
| Llama 1-30B | 202 ms | – | – | 72 / 214 ms |
| Llama 3-45B | 328 ms | – | 6235 ms | 233 / 440 ms |
| Llama 3-70B | 10,120 ms | OOM | OOM | 674 / 1793 ms |

10-node testbed: a mix of RTX 4060 Ti, 3090, two 1080 Ti and a 2080 Ti, two Metal Macs and CPU-only boxes, with disks from 0.32 to 3.88 GB/s. On Llama-3.3-70B the solver reached 146 ms/token using only three devices. The best single node took 24,037 ms; the TP-only runtime 8,415 ms, with one device upgraded to 128 GiB; the memory-proportional runtime ran out of memory.

**Transfer to MoE.** A small 2026 open-source project re-implements this formulation in Python as an MILP over **layer and expert assignment**, with profilers for CPU/GPU throughput, memory, disk I/O and communication **[verified: paper/docs]**. No numbers are published. It suggests the formulation carries over to experts **[inferred]**.

### 3.4 Stage count under a latency budget

- **Rule.** The 2026 QUIC runtime chooses the number of physical stages from a decode-latency budget, not by maximising the stage count, because every hop sits on the critical path. Ranges are contiguous, downstream stages load first, and stage 0 becomes routable only when all stages are ready **[verified: paper/docs]**.
- **Sticky owners.** Recurrent and hybrid layers (Qwen3-Next, Falcon-H1) are pinned to one owner, so the roughly 76–79 MB of recurrent state never moves **[verified: paper/docs]**.
- **Certified roster.** Split mode fails closed unless the architecture is on a certified roster tied to the engine version pin **[verified: paper/docs]**.
- **Reported run.** A 296.5 GiB 2-bit model package on 4× 96 GB nodes plus one 48 GB node over a roughly 5 ms QUIC path: 14.98 tok/s. The authors' own note says the 65/1 layer split "is not ... optimal" **[reported by a surveyed system]**.
- **WAN bound.** A 2025 write-up on inference over public networks: a 2-stage synchronous pipeline at 100 ms one-way latency caps single-stream decode at `1/(2 × 0.1 s) ≈ 5 tok/s`, regardless of GPU speed **[verified: paper/docs]**. This is the clearest statement of why stage count must be minimised over WAN.
- **Dropping a device beats adding a hop.** See the 146 versus 156 ms result in §3.3.

### 3.5 DP with water-filling, and per-request routing

From a 2025 volunteer-GPU pipeline scheduler.

- **Phase 1, model allocation [verified: paper/docs]:**
  - A DP over GPUs decides, for each GPU, whether to skip it, extend a pipeline or start a new one.
  - Water-filling assigns layers: `x_i = min(c_i, λF_i)`, with `λ` found by binary search and largest-remainder rounding to contiguous intervals.
  - The objective trades pipeline count against stage count: `Z(k) = k^α / (T_comp + (s*(k)/k)·r_RTT)`.
  - The code has greedy and DP allocators plus an in-place water-filling rebalance **[verified: code]**.
- **Phase 2, request routing [verified: paper/docs]:**
  - A DP over a layer-indexed DAG with node latency plus pairwise RTT, `O(L·R̄²)`.
  - 6.63 ms per request at 256 GPUs **[reported by a surveyed system]**.
  - The code includes DP, randomised and round-robin routers **[verified: code]**.
- **Membership changes [verified: code].** Joins greedily take the bottleneck layers. A leave triggers a **global rebalance by re-bootstrapping all nodes** once load variance passes a threshold.
- **Performance [reported by a surveyed system].** Qwen3-32B on 5× RTX 5090 + 2× RTX 4090 across geographically separated datacentres, about 10 ms average latency, compared against one earlier heterogeneous-GPU system only: up to 3.2× lower latency (1.66× average) and up to 3.6× higher throughput (1.58× average).

### 3.6 Bottleneck-first greedy with hysteresis

From the 2023 internet-scale pipeline swarm **[verified: paper/docs]**.

- Each server measures `min(network throughput, GPU throughput)` in tokens/s.
- It picks the block interval that most improves the lexicographically sorted per-block throughput, i.e. it fills the worst bottleneck.
- Servers re-check periodically and move only if total throughput would rise by at least 20%. This avoids churn that would reset caches.

### 3.7 Solver survey

| Approach (source) | Problem shape | Method | Tiers modelled | Network modelled | Reported solve cost / result |
|---|---|---|---|---|---|
| Layer-window ILP (2025 home-cluster study) | layer windows + GPU/CPU split per device, device selection | ILFP → enumerate `k` → ILP per `k` (HiGHS) + set-assignment fixed point | VRAM, RAM, disk (through an OS model) | paper yes; **released code no** | 10–12 ms on 4–32 devices; 3.3× over memory-proportional at 70B |
| Layer + expert MILP (small 2026 re-implementation) | layers **and experts** | MILP | CPU/GPU, memory, disk I/O | yes (profiled) | none published |
| DP + water-filling (2025 volunteer-GPU scheduler) | pipelines × stages over volunteer GPUs; per-request path | DP + water-filling; per-request DP over a layer DAG | VRAM only | RTT per edge | 6.63 ms/request at 256 GPUs |
| Bottleneck-first greedy (2023 internet swarm) | each server picks a block span | greedy, 20% hysteresis; D* Lite client routing | VRAM only | latency, in routing | up to 10× over offloading |
| Max-flow + MILP (2025 heterogeneous-GPU paper) | placement + request flow on heterogeneous GPUs and links | max-flow graph + MILP | VRAM | edge capacities | up to 3.3× throughput, 24–42 GPU nodes |
| Asymmetric TP×PP (2024 heterogeneous-GPU paper) | asymmetric TP×PP over heterogeneous GPUs | constrained optimisation | VRAM | yes | up to 2.3× lower latency deadline, Llama-2 70B |
| P/D graph partitioning (2025 paper) | disaggregated P/D on heterogeneous GPUs | graph partitioning + max-flow | VRAM | KV transfer links | up to 2.0× throughput |
| Inter-op DP + intra-op ILP (2022 training paper) | stage slicing + intra-op sharding | **DP** for stage → device mesh, **ILP** for intra-op | VRAM | mesh bandwidth | – |
| Placement LP (2023 single-GPU offload paper) | where weights, KV and activations live on one GPU | **LP** over GPU/CPU/disk placement percentages | VRAM, RAM, disk | – | OPT-175B at 1 tok/s on one 16 GB GPU (batch 144) |
| Edge PP partitioning (several 2023–2024 edge papers) | edge PP partition | DP / LP | RAM | yes | all assume aggregate memory holds the model |
| Re-parallelisation on preemption (2024 paper) | re-parallelise when instances are preempted | Kuhn-Munkres bipartite matching for migration | VRAM | yes | P99 2.4–9.1× better, 54% cost saving |
| Proportional heuristics (four home-scale runtimes) | layer split | proportional to TFLOPS or memory | one tier | no | 3.3× slower than the ILP at 70B |

Results are **[reported by a surveyed system]**; shapes and methods are **[verified: paper/docs]**.

**Takeaways:**

- Exact solvers are cheap at home scale: a few devices, about 100 layers, or a few hundred expert groups.
- Every datacenter solver ignores disk. Only the layer-window ILP and the single-GPU LP model it, and the LP is single-device.
- **No published solver models per-device three-tier residency, the network and MoE expert placement together [inferred].**
- Hysteresis on rebalancing (20%) and a split between slow placement and fast per-request routing are the right dynamics for a changing membership.
- A fixed fallback order (fit in VRAM, then distribute, then offload), as in the filter → score scheduler, is a decision a solver should make by cost **[inferred]**.

Other recent literature **[verified: paper/docs]**: a 2026 paper claims elastic multi-device edge execution, but its abstract gives no mechanism. A 2026 single-SoC paper co-schedules CPU, iGPU and NPU; that is out of scope for multi-device work but relevant to per-device backends.

### 3.8 How device capabilities are measured, or not

| Input | Event-sourced cluster runtime **[verified: code]** | Layer-window ILP study **[verified: code + paper/docs]** |
|---|---|---|
| Memory | `psutil.virtual_memory().available` at 1 s; on macOS, `macmon` (`ram_total − ram_usage`); an override env var can fake it. Swap is collected and **never used** | available RAM, swap (Android), Metal recommended working set, unified memory, cgroup limits |
| GPU | Only `nvmlDeviceGetCount() > 0`, to decide whether a CUDA backend is listed. **No VRAM query anywhere**, so on a discrete-GPU Linux box shards are sized against system RAM. It works on a unified-memory GB10 only because memory there is shared **[inferred]** | VRAM capacity and bandwidth, host↔device copy speed |
| Compute | Not measured. GPU usage, temperature, power and CPU usage are collected for display; a `flops_fp16` field is a TODO | quantized matmul timing per backend × quant type |
| Disk | – | external `fio` run; sequential versus random by OS |
| Network | interface type from the OS hardware-port list, Thunderbolt ids from the system profiler, reachability from HTTP probes. No latency or bandwidth (both on the project's own TODO list) | link latency and throughput are profiled; the released solver ignores them |

Two other patterns: the internet swarm and the volunteer-GPU scheduler re-measure throughput and RTT continuously **[verified: paper/docs]**, and the star-topology runtime invents TFLOPS from VRAM and clock (§3.1).

The cluster runtime's README says placement "takes into account device resources and network latency/bandwidth between each link", and a vendor blog post from the same project says it "profiles each device for compute throughput, memory bandwidth, memory capacity, and network characteristics" and assigns P/D roles automatically **[verified: paper/docs]**. Neither is in the public code: P/D links are created by hand through an API call **[verified: code]**.

### 3.9 Memory-accounting gaps

- **No KV or activation reserve.** Users observed real usage exceeding the model size, and placement refusing near-limit fits. A maintainer said they were "pretty hesitant to force out other memory at the moment after encountering some ooms" **[verified: paper/docs]**. That project's tracker has 32 hits for "OOM", 24 for "out of memory" and 22 for "swap".
- **TP checks only the cycle sum,** never per node **[verified: code]**.
- **Duplicated non-layer parameters.** Embeddings, final norm and LM head are evaluated on **every** PP rank, and the per-node estimate `storage × layers/n_layers` does not count them **[verified: code; impact inferred]**.

---

## 4. Parallelism on consumer links

### 4.1 Pipeline at batch 1

- At batch 1 only one stage works at a time, so PP adds capacity and no speed. Continuous batching (up to 8 concurrent requests in one runtime) is what fills the pipeline **[verified: code]**.
- **Measured [reported by a surveyed system]**, Llama 3.2 3B on M4 Pro 24 GB Mac minis (2024 benchmark series):

  | Setup | Single-request tok/s | Multi-request tok/s |
  |---|---|---|
  | 1 node | 49.3 | 49.3 |
  | 2 nodes | 44.4 | 95.7 |
  | 3 nodes | 39.7 | 108.8 |

  The authors: "adding more devices will actually decrease single-request performance due to network overhead", and "the bottleneck here is generally the latency between devices, not the bandwidth".
- **Large models across 8 nodes [reported by a surveyed system].** 8× M4 Pro 64 GB Mac mini (512 GB total), all 4-bit, interconnect not stated:

  | Model | TTFT | tok/s |
  |---|---|---|
  | DeepSeek V3 671B | 2.91 s | 5.37 |
  | Llama 3.1 405B | 29.71 s | 0.88 |
  | Llama 3.3 70B | 3.14 s | 3.89 |
- **Apple's MLX numbers** confirm it: 2-node PP decodes at the single-node rate (§4.5). Pipelined prefill reaches "up to ~1.8× faster" than one node at 65,536 tokens **[verified: paper/docs]**.
- **Stage hand-off.** Rank `r > 0` receives a tensor shaped like its input from `r−1` before its first layer, and rank `r < N−1` sends its output to `r+1` after its last layer **[verified: code]**.
- **End-of-step all-gather.** In one runtime's PP decode, every rank all-gathers the final output so each can run the norm and LM head and sample locally with the same seed; only rank 0 emits. Each rank also holds the embedding and LM head (§3.9) **[verified: code]**.

### 4.2 Multi-round ring pipelining and the prefetch-release conflict

From the home-cluster study **[verified: paper/docs]**.

- Devices form a logical ring. With layer window `w_m` per device and `k` rounds per token, the model is cut into `L/W` segments (`W = Σ w_m`) assigned round-robin, so each token circulates the ring `k` times. `k = 1` is plain PP.
- **Why rounds matter.** Plain PP with mmap offload prefetches a device's whole slice, but the page cache can hold only part of it. Later prefetches evict earlier ones, so every layer is loaded twice. Smaller windows (more rounds) keep each prefetch at or below available RAM.
  - An RTX 4090 laptop on Llama-3-70B Q4K with 16 GiB RAM reloaded about 25 GiB per token during compute, even with prefetch.
  - Multiple rounds cut one device's per-token latency to 20–23% of plain PP.
- **Measured effect [reported by a surveyed system]:**
  - On a homogeneous 4× (8-core, 8 GiB, 2 GB/s SSD) CPU cluster, `k ≥ 2` halves TPOT for models above 60B.
  - On a heterogeneous testbed the drop is 51–84% for models above 45B.
  - For small models, extra rounds add only kernel-launch and communication overhead.
- **Batching.** Continuous batching runs at the token step, but one micro-batch is processed at a time to avoid disk and memory contention between micro-batches. Throughput peaks around 8–16 concurrent requests on memory-tight testbeds **[reported by a surveyed system]**.
- **The constraint `L = k·W`** (equal windows across rounds) is a modelling convenience; the runtime does not need it **[inferred]**.
- **Independent adoption.** A small 2026 MLX-only runtime also implements the pipelined ring for "models that exceed total cluster memory", with automatic discovery, Thunderbolt detection and a gRPC/protobuf transport. TP and long context are unfinished, and it publishes no benchmarks **[verified: paper/docs]**.

### 4.3 Pipelined chunked prefill

From one cluster runtime **[verified: code]**.

- Prefill is chunked at `4096 // min(4, world_size)` tokens.
- Rank `r` runs `r` leading dummy iterations, then the real chunks, then `N−1−r` trailing dummies, so stage `i` processes chunk `k` while stage `i+1` processes chunk `k−1`.
- Sends are queued and flushed after each chunk.

This is real pipelined prefill, and it is the main place where PP improves a single request.

### 4.4 Tensor-parallel arithmetic

- **Layout.** Column-parallel q/k/v/gate/up and row-parallel o/down give one all-sum after attention and one after the MLP, **per layer**. MoE layers wrap the experts with an all-sum **[verified: code]**.
- **Per-architecture code.** TP needs an explicit sharding strategy for each architecture, and unknown architectures run PP only. One runtime has strategies for Llama/Ministral3, DeepSeek V3/V3.2/V4, Kimi K2.5, MiniMax, GLM4-MoE and GLM4-MoE-Lite, GPT-OSS, Step 3.5, Nemotron-H (Mamba2 mixer sharding), Qwen3/3.5/Next and Gemma 4 **[verified: code]**.
- **TP-only CPU runtime.** Two all-gathers per layer (attention output and FFN output), a root sync at the start and a logits gather at the end. MoE (Qwen3 30B-A3B) slices each expert's matmuls by row or column, i.e. TP inside experts, not EP **[verified: code]**.
- **Estimate: syncs per token.** A 70B dense model has 80 layers × 2 = 160 sequential syncs per token.
  - On 10GbE, 34.41 ms of measured transfer per token ÷ 160 ≈ 0.22 ms per sync.
  - On Wi-Fi, about 150 ms per all-reduce gives 160 × 150 ms ≈ 24 s/token (the home-cluster study's own arithmetic).
  - So TP is viable only on sub-millisecond links.
- **Estimate: latency sensitivity.** A third-party review cites an MLX change that lowered RDMA memory-access latency from 300 µs to under 50 µs. If each of the 160 all-sums costs roughly that:

  | Per-all-sum cost | Communication per token | Decode ceiling from communication alone |
  |---|---|---|
  | 300 µs | 48 ms | ≈ 21 tok/s |
  | 50 µs | 8 ms | ≈ 125 tok/s |

  This illustrates the "order of magnitude" claim in the MLX docs; it is not a measurement.
- A maintainer of one runtime: TP "requires low double digit microsecond latency to give a speedup", while PP is fine at "tens of milliseconds" **[verified: paper/docs]**.
- **Topology constraint.** MLX's RDMA backend "supports only fully connected topologies" ([MLX distributed docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)) **[verified: paper/docs]**.
- **Even splits only.** The runtimes surveyed split TP evenly; node counts that do not divide `hidden_size` and `kv_heads` are rejected. Arbitrary splits are described as "a blocker for smarter tensor parallelism on heterogeneous devices" **[verified: paper/docs]**.
- **Star all-reduce.** A 2024 edge TP paper from the same group as the home-cluster study pairs TP with a sliding-window disk scheduler and uses a **star all-reduce, because latency rather than bandwidth dominates**. It claims "3.1 GB of memory for 70B-scale models"; the later study quotes it at about 30 s/token **[verified: paper/docs]**.
- **Direct pairwise all-reduce.** The remote-backend RPC layer added (2026) a pairwise all-reduce over a direct server-to-server connection, so "the tensor data ... never passes through the client" **[verified: code]**.

### 4.5 Measured TP versus PP

**TP-only CPU runtime [reported by a surveyed system]**, CPU only, 4× Mac mini M4 Pro 24 GB, q40 weights:

| Model | 1 node | 2 nodes | 4 nodes |
|---|---|---|---|
| Llama 3.3 70B over TB5 | doesn't fit | 3.31 tok/s | 6.51 tok/s |
| Llama 3.3 70B over 10GbE | doesn't fit | 3.25 tok/s | 6.04 tok/s |
| Llama 3.1 8B over TB5 | 16.91 tok/s | 30.27 tok/s | 38.83 tok/s |

The logs give an average transfer time of 34.41 ms per token for 70B on 4 nodes over 10GbE.

**Apple's MLX numbers [verified: paper/docs]** ([Apple ML Research, 2026-10-15](https://machinelearning.apple.com/research/mlx-distributed-inference)). Mac Studio M5 Ultra 512 GB, MLX RDMA over TB5 with two cables per pair, decode over 128 tokens after context:

| Model | 1× M5 Ultra | 2× PP | 2× TP | 4× TP |
|---|---|---|---|---|
| Mistral Medium 3.5 (128B dense FP8) | 7.70 tok/s | 7.70 | 13.69 | 23.37 |
| Qwen 3.8 (27B BF16) | 19.69 | 19.44 | 33.27 | 50.89 |
| Muse Spark Open (816B MoE, 42B active, NVFP4) | – | 15.74 | 24.99 | 36.73 |

**RDMA cluster review [reported: third-party review]**, 4× M3 Ultra Mac Studio (two 512 GB, two 256 GB). The cluster runtime used TP over RDMA; the comparison is a layer split over TCP through the remote-backend RPC layer. Qwen3-235B and DeepSeek V3.1 labelled 8-bit, Kimi K2 Thinking at native 4-bit. Decode tok/s:

| Model | Nodes | TP over RDMA | Layer split over TCP |
|---|---|---|---|
| Qwen3 235B A22B | 1 | 19.5 | 20.4 |
| | 2 | 26.2 | 17.2 |
| | 4 | 31.9 | 15.2 |
| DeepSeek V3.1 671B | 1 | 21.1 | 14.8 |
| | 2 | 27.8 | 16.0 |
| | 4 | 32.5 | 14.6 |
| Kimi K2 Thinking 1T A32B | 2 | 21.6 | 18.5 |
| | 4 | 28.3 | 16.4 |

Caveats: an 8-bit DeepSeek-V3.1 is 765.6 GB by the runtime's own model card, so the single-node bar on a 512 GB machine cannot have been 8-bit; either the quantization differed per bar or the label is loose **[inferred]**. Prompt length and TTFT were not published. The reviewer called it "prerelease software" with model-load failures: "When it works, it works great. When it doesn't…".

**Summary of data points:**

| Data point | Link | Result |
|---|---|---|
| TP, 70B, 4 CPU nodes | 10GbE | 34 ms/token in transfer, 6.04 tok/s |
| TP all-reduce | Wi-Fi | ≈150 ms per all-reduce → ≈24 s/token at 70B |
| TP versus PP, 2 nodes | TB5 RDMA | TP 13.69 versus PP 7.70 tok/s (128B dense) |
| Multi-round ring PP | Wi-Fi 320–610 Mb/s | insensitive to bandwidth above ≈50 Mb/s; +68% TPOT for 100 → 200 ms RTT |
| Internet-swarm PP | 100 Mb/s, 100 ms RTT | 2.29 → 1.57 steps/s (Llama 2 70B) |

### 4.6 Expert parallelism

- No home-scale runtime does EP. The TP-only runtime runs TP inside each expert (§4.4) **[verified: code]**.
- Datacenter engines use "Data Parallel attention with Expert or Tensor Parallel MoE layers", with elastic EP (§10.4) **[verified: paper/docs]**.
- Their general layout is TP inside a node and PP across nodes ("tensor_parallel_size = GPUs per node, pipeline_parallel_size = number of nodes"). PP "supports uneven splits" and is the recommendation when there is no NVLink **[verified: paper/docs]**.
- EP all-to-all has the same latency profile as TP collectives, so it belongs inside fast islands only **[inferred]**.

### 4.7 Cluster-consistent decisions inside collectives

All **[verified: code]**.

- All-gather the task ids so every rank enqueues the same requests in the same order. This prevents TP collective deadlocks.
- Sample with the same seed on every rank from the all-gathered final hidden state.
- All-gather memory-pressure flags so every rank evicts the same prefix-cache entries.

The same pattern applies to any per-rank decision that changes which collectives run, including tier-migration decisions **[inferred]**.

### 4.8 Rule for consumer links [inferred from the above]

- Use TP, or EP all-to-all, only inside a node or across RDMA/TB5 or wired ≥10GbE links with sub-millisecond syncs.
- Use a PP ring with the minimum number of stages everywhere else.
- Every extra stage costs one RTT per token per round.
- TP should support uneven splits, and the solver should be able to combine PP across slow links with TP inside fast islands.

---

## 5. Transports and wire formats

### 5.1 The planes in one cluster runtime

| Plane | Transport | Notes |
|---|---|---|
| Control: events, commands, election, **tokens** | zenoh over TCP, router mode, gossip | JSON payloads; drop on congestion; ≥1 MiB payload warns **[verified: code]** |
| Discovery | UDP, IPv6 multicast | 1 Hz hello **[verified: code]** |
| Topology probing | HTTP GET of a node-id endpoint every 10 s per IP | **[verified: code]** |
| Activations and collectives, TCP | MLX `ring` backend: TCP sockets, neighbours only, configured by hostfile and rank env vars | MLX docs: "Ring all reduce and all gather over TCP sockets", rank `i` talks only to `i±1` **[verified: code + paper/docs]** |
| Activations and collectives, RDMA | MLX `jaccl` backend: RDMA over Thunderbolt 5 with an ibverbs-style API, configured by device-matrix and coordinator env vars | MLX docs: macOS 26.2+, TB5, "supports only fully connected topologies", latency "an order of magnitude lower than the ring backend", "necessary for things like tensor parallelism" **[verified: code + paper/docs]** |
| KV transfer for P/D | raw TCP, 4-byte length-prefixed msgpack frames, `TCP_NODELAY`, 4 MiB `SO_SNDBUF` | **[verified: code]** |
| Weights | HTTPS from Hugging Face Hub, **every node independently** | §7 **[verified: code]** |

MLX's NCCL backend is not used there; CUDA nodes run the TCP ring. Linux RDMA is not supported **[verified: code]**. MLX also offers MPI, and NCCL on CUDA; its ring is "usually faster than MPI" **[verified: paper/docs]**.

### 5.2 Data-plane transports elsewhere

- **ZeroMQ PUSH/PULL over TCP** (home-cluster ring). Each node has a send socket to `next`, a receive socket, a socket to rank 0 (from the last stage) and a signal socket **[verified: code]**.
- **Raw TCP full mesh** (TP-only runtime): `TCP_NODELAY`, `TCP_QUICKACK`, non-blocking I/O **[verified: code]**.
- **QUIC with three ALPNs** (2026 runtime): data plane, control and stage traffic on separate protocol ids, end-to-end encrypted **[verified: paper/docs]**.
- **Protobuf over a P2P stream** (volunteer-GPU scheduler): `bytes hidden_states` in a pipeline-forward RPC stream, with ZeroMQ IPC between the peer process and the executor **[verified: code]**.
- **Remote-backend RPC protocol** (ggml-based runtime, protocol 8.0.0) **[verified: code]**:
  - commands for buffer allocation; tensor set, get and copy (including 2D variants); hash-first tensor set; graph compute; graph recompute by id; and communicator init, all-reduce and free
  - **graph reuse:** a graph with a known id is replayed with a fire-and-forget recompute command, so per-token graph serialisation disappears
  - **weight cache:** the hash-first set is tried for large tensors, so with the server's cache flag on, weights are cached on the server's disk and not resent
  - TCP by default; RDMA is auto-negotiated when both peers support it, either RoCEv2 through libibverbs or macOS 26.2+ RDMA over Thunderbolt 5 **[verified: paper/docs]**
  - the README still calls it "proof-of-concept"
- **Compact binary serialisation** (star-topology runtime): `RawTensor { data, dtype tag (u8), shape }` in the model's native dtype **[verified: code]**.
- **Datacenter**: NCCL, InfiniBand plus GPUDirect RDMA recommended **[verified: paper/docs]**.

### 5.3 Datacenter transfer-engine abstraction

From a datacenter KV transfer engine **[verified: paper/docs]**.

- Segments are DRAM, VRAM or NVMe-oF. Transports: TCP, RDMA, EFA, NVMe-oF, NVLink, intra-node NVLink, HIP and shared memory.
- A batch read/write API. Each server broadcasts a NIC/CPU/GPU **topology matrix** used for path selection.
- Multi-NIC pooling with retry. Failed endpoints are evicted and reclaimed by a monitor running at roughly 1 Hz.
- The next-generation engine:
  - picks the transport per request at runtime, and stages through host memory when no direct path exists
  - sprays slices across rails using completion-time and queue-depth telemetry
  - **handles partial link failures inside the runtime** by stopping use of degraded paths, retrying slices and re-admitting recovered paths
- Its PyTorch distributed backend is described as "capable of continuing operation seamlessly in the presence of rank failures", and its EP variant lets experts "be dynamically routed or recovered".

The datacenter assumptions (RDMA, a metadata server) do not fit home clusters. The interface does: a declarative "what to move, not how" API with per-slice path scheduling and in-runtime failover. The same interface could move disk → RAM, RAM → VRAM and node → node **[inferred]**.

### 5.4 Message sizes

All **Estimates**, bf16 = 2 B unless stated.

| Case | Formula | Result |
|---|---|---|
| PP decode, per hop per token | `hidden × 2 B` | Llama-3.3-70B (8192): 16 KiB; Qwen3-235B (4096): 8 KiB; DeepSeek-V3.x (7168): 14 KiB |
| PP decode, f32 on the wire | `hidden × 4 B` | 32 KiB at `e = 8192`: 0.26 ms on 1 GbE |
| PP decode, end-of-step all-gather | `N × hidden × 2 B` per rank | 64 KiB for N = 4 on Llama-70B |
| PP prefill, per hop per chunk | 2 ranks: 2048 tok × 8192 × 2 B | 32 MiB (Llama-70B) |
| PP prefill, whole prompt in f32 | 4096 × 8192 × 4 B | 128 MiB: ≈2.7 s at 400 Mb/s |
| TP decode, collectives per token (dense, 80 layers) | 2 all-sums/layer × 80 | 160 all-sums of 16 KiB each; latency-bound, not bandwidth-bound |
| P/D KV volume, Llama-3.1-8B | 32 layers × 2 (K,V) × 8 KV heads × 128 dim × 2 B | 128 KiB/token; an 8,192-token prompt ≈ 1 GiB ≈ 0.86 s on 10 GbE at line rate (1.25 GB/s) if not overlapped |
| P/D KV volume, Llama-3-70B (GQA) | 80 × 2 × 8 × 128 × 2 B | 327,680 B/token; 4K tokens ≈ 1.25 GiB ≈ 27 s at 400 Mb/s |

**Estimate: per-token network cost on the home-cluster study's main testbed.**

- `e = 8192` (Llama-3-70B) gives a 32 KiB f32 activation. At 400 Mb/s, within the reported 320–610 Mb/s, serialisation is `32768 · 8 / 4e8 ≈ 0.66 ms`.
- Add the reported 3–7 ms link latency: about 3.7–7.7 ms per hop.
- With `M = 4` devices and `k = 2` rounds, a token makes `k·M = 8` hops, about 30–60 ms per token. That is under 10% of the 674 ms TPOT, which matches the measured bandwidth insensitivity: TPOT flat above about 50 Mb/s, and +68% when RTT goes from 100 to 200 ms. The authors conclude that latency, not bandwidth, is the limit **[reported by a surveyed system]**.

### 5.5 Physical links

All **[verified: paper/docs]** unless tagged.

- **RDMA over Thunderbolt 5 requires:**
  - macOS ≥ 26.2 and TB5 hardware: M4 Pro Mac mini, M4 Max Mac Studio and MacBook Pro, M3 Ultra Mac Studio
  - `rdma_ctl enable`, run from Recovery on each machine
  - a **full mesh** of TB5 cables (there are no TB5 switches)
  - avoiding the Mac Studio TB5 port next to the Ethernet port
  - **identical macOS versions, down to beta build numbers**, or RDMA ports may not discover each other
- **Thunderbolt networking throughput:** "something like 50–60 Gbps real-world" **[reported: third-party review]**; that review's Ethernet was 2.5 GbE.
- **RDMA fragility in practice.** Users of one runtime report receive failures (errno 2/60) and queue-pair state-change failures (errno 22) needing a full reboot of all nodes, "multiple times per day"; models failing to load with RDMA on M4 Max laptops; M3 Ultra TB5 ports unable to use RDMA; and loads stuck in loading or warm-up with TP over RDMA. About 130 issues on that project mention RDMA **[verified: paper/docs]**.
- **RDMA claims.** One runtime's README claims "99% reduction in latency" for RDMA. The only number found, the MLX change from 300 µs to under 50 µs, is a reduction of at least 83% **[verified: paper/docs]**.

---

## 6. Activation precision on the wire

| Format | Where | Outcome |
|---|---|---|
| Raw f32 hidden state + int32 positions, ZeroMQ multipart | home-cluster ring | works; `n_embd × n_tokens × 4` bytes **[verified: code]** |
| Raw little-endian f32, **by policy** | 2026 QUIC runtime | "Historical f16 and q8 failures remain compression research evidence"; any compression must be "a separately versioned protocol with a model-agnostic safety contract" **[verified: paper/docs]** |
| Q8_0 buffers (32 values + 1 scale) when weights are q40; f32 models sync in f32 | TP-only CPU runtime | works for its supported models **[verified: paper/docs]** |
| Native model dtype (bf16/f16/f32/f8) + shape | star-topology runtime | **[verified: code]** |
| bf16 (framework native) | event-sourced cluster runtime | **[verified: code]** |
| Protobuf bytes | volunteer-GPU scheduler | **[verified: code]** |
| Kernel-library tensors through the client | remote-backend RPC layer | **[verified: code]** |

**Analysis [inferred]:**

- At batch 1 on a LAN, activation bytes per token are tiny next to link latency: 32 KiB in f32 at `e = 8192` is 0.26 ms on 1 GbE. Compression buys little and carries correctness risk.
- Prefill is where size matters: a 4K-token prompt is 128 MiB in f32 at `e = 8192`, about 2.7 s at 400 Mb/s. Prefill should be chunked and pipelined across stages (§4.3), and bf16 on the wire for prefill only is worth testing per model family.
- The sources disagree (f16/q8 failures in one runtime against Q8_0 success in another), so precision has to be measured per model family, for decode and prefill separately.

---

## 7. Weight distribution

| Approach | How it works | Seen in |
|---|---|---|
| Every node holds or downloads the full model | each node pulls the whole repo, or the same file must exist at the same path on every node | the event-sourced cluster runtime, the home-cluster study, the GPU cluster manager and datacenter engines, `mlx.launch` setups |
| Coordinator streams slices | the root loads the model and **streams each worker its slices over the network**, so workers need no model file | TP-only CPU runtime (raw) **[verified: code]** |
| Coordinator streams files with resume | required safetensors are sent in chunks, zstd-compressed with CRC32, resumable; workers cache them locally | star-topology runtime **[verified: code]** |
| Hash-addressed server cache | a hash-first tensor upload; with the cache flag on, weights are cached on the server's disk and not resent | remote-backend RPC layer **[verified: code]** |
| Content-addressed P2P blocks | blocks fetched by CID from any peer, verified on receipt | volunteer-GPU scheduler **[verified: code]** |
| Per-stage layer packages | Hugging Face Hub repos holding a package manifest plus GGUF fragments, so **each peer downloads only its stage's layers**; integrity checked by artifact digests | 2026 QUIC runtime **[verified: paper/docs]** |

**Full download on every node, in detail** **[verified: code]**. In the event-sourced runtime the per-layer file filter (mapping the safetensors weight map to files) exists but is dead code: the allow-pattern function returns `["*"]` first. A code comment gives the reasons: "(i) We don't handle all kinds of files; (ii) We don't have sticky sessions. (iii) Tensor parallel requires all files." Users asked why every node must download from Hugging Face, and the project lists "faster model loads by streaming model from other devices in cluster" as a TODO **[verified: paper/docs]**. Its downloader is otherwise solid:

- 8 parallel file downloads
- resume through `.partial` files
- rate-limit handling
- multiple writable model directories (the first with enough free space wins)
- read-only directories, e.g. NFS
- an offline mode (the offline install still needs one internet check)

**Format support.** That runtime loads MLX-format safetensors only. The downloader returns a `.gguf` path if one exists, but the loader cannot read it **[verified: code]**.

**Analysis [inferred].** The content-addressed P2P variant scales best and lets placement changes move only the deltas. The coordinator should never need the whole file in RAM, which the slice-streaming root does.

---

## 8. Per-device memory tiers inside multi-device runtimes

### 8.1 OS page cache as residency manager

From the home-cluster study.

- **GPU layers** (`n_m` of `w_m`) are pinned in VRAM. VRAM is deliberately not paged, because RAM↔VRAM paging would compete with disk → RAM traffic on PCIe **[verified: paper/docs]**.
- **CPU layers** are mmapped and left to the OS page cache. Prefetch is a hint **[verified: code]**:
  - after sending its activations, a device calls `posix_madvise(..., POSIX_MADV_WILLNEED)` on the page-aligned, merged tensor ranges of its *next* subgraph
  - an optional force flag also touches every fourth page to force the read
  - Linux sets `POSIX_FADV_SEQUENTIAL`; macOS uses `POSIX_MADV_RANDOM`
- **There is no explicit NVMe I/O**: no `O_DIRECT`, no io_uring. Only madvise and fadvise appear in the code **[verified: code]**.
- **Low memory pressure as a feature.** Weights live in reclaimable page cache, so the OS takes RAM back when the user opens other apps. Reported memory pressure is below 6% on every device for 70B, against over 50% for the memory-proportional cluster runtime and the TP-only runtime. Everyday background apps (video, conferencing, downloads) raised TPOT 5–18%; a 3D game raised it 66% **[reported by a surveyed system]**.
- **The cost.** Residency follows kernel LRU heuristics that differ by OS. The paper has to model four OS memory regimes (macOS with and without Metal, Linux, Android), and the Metal case "may cause the entire model weights to be repeatedly reloaded" **[verified: paper/docs]**.
- **Context length.** 70B TPOT stays flat up to about 4K tokens on the low-memory testbed and about 16K on the medium one, then climbs steeply or OOMs as KV crowds out page cache **[reported by a surveyed system]**.

### 8.2 Disk-streamed MoE experts

- The star-topology runtime `pread`s individual expert matrices on demand from safetensors, relying on the OS page cache plus a small LRU of dequantized experts **[verified: code]**. It cites a published SSD-streamed-MoE study in which the page cache beat a custom cache by 38%.
- This is the only shipped system found that combines multi-device sharding with SSD-streamed experts. The 2026 QUIC runtime has an SSD-expert backend as a separate plugin, but its split placement does not model RAM or disk **[verified: paper/docs]**.

### 8.3 Weights and KV in the event-sourced runtime

All **[verified: code]** unless tagged.

- **Load path.** A lazy graph over the safetensors is built first. PP evaluates `layers[start:end]` one by one with progress events, then trims the rest; TP evaluates each layer and then shards it in place. A barrier follows. On Metal the wired limit is raised to the recommended working-set size, with a warning if the shard exceeds 90% of it. There is no other residency control: no streaming, no mmap paging policy, no offload.
- **No tiers.** Weights go only into available RAM. Swap is reported and ignored, VRAM is never measured, and nothing is staged RAM → VRAM or streamed from NVMe.
- **KV per request.** One unbounded, unquantized KV cache per layer by default. Rotating and quantized cache code paths exist but are not the default, and a max-KV-size constant (3200) is defined and unused on the main path.
- **KV location.** Under TP each rank holds `n_kv_heads/N` heads; under PP each rank holds its own layers.
- **Prefix cache.** An in-memory LRU of whole-request caches keyed by token prefix, with SSM-state snapshots for hybrid models (Qwen3-Next/3.5, Nemotron-H). Limits: 64 entries, 10% of RAM, and a memory-pressure threshold of 0.70–0.85 depending on RAM size. Eviction is cluster-consistent (§4.7).
- **Gaps.** No KV spill to CPU or SSD, and no KV or activation reserve in placement (§3.9).

### 8.4 KV tiering in datacenter stacks

- A datacenter stack's KV block manager tiers KV across GPU, CPU, SSD and remote storage **[verified: paper/docs]**.
- A Kubernetes-based serving stack reports "13.9x throughput" from hierarchical KV offload at 250 users on 4× H100 **[reported by a surveyed system]**.

---

## 9. Prefill/decode disaggregation

### 9.1 Layer-streamed KV over TCP (consumer hardware)

From one cluster runtime **[verified: code]**:

- A prefill instance and a decode instance are linked **by hand** through an API call.
- Wire messages: `Header{request_id, model_id, num_layers, dtype, start_pos}`; then per layer either `KVChunk{layer_idx, num_tokens, n_heads, head_dim, dtype, keys, values}` or an arrays-state message (SSM state for hybrid models); then `Done{total_tokens}`. The decode side injects the chunks into its framework caches.
- Per-layer streaming overlaps prefill compute with transfer **[verified: paper/docs]**.

**Overlap condition [verified: paper/docs].** The project's blog gives `P/B < (c1/c2) · s/q`, with `K = c1/c2` = 2 for Llama-2-7B, 8 for Llama-3-8B and 16 for Llama-3-70B / Qwen-2.5-72B. With 8-bit KV and `P/B = 10,000`, full overlap needs `s` > 5k, 10k and 40k tokens respectively. The same form works as a cost-model term for deciding when to disaggregate, and an analogous term applies to streaming KV or weights from NVMe while computing **[inferred]**.

**Measured [reported by a surveyed system].** Llama-3.1 8B FP16, 8,192-token prompt, 32 generated tokens, 10 GbE link. Hardware as stated: DGX Spark "~100 TFLOPs FP16" with 273 GB/s; M3 Ultra "~26 TFLOPs FP16" with 819 GB/s.

| Config | Prefill | Generation | Total | Speedup |
|---|---|---|---|---|
| DGX Spark | 1.47 s | 2.87 s | 4.34 s | 1.9× |
| M3 Ultra | 5.57 s | 0.85 s | 6.42 s | 1.0× |
| Spark prefill + M3 Ultra decode | 1.47 s | 0.85 s | 2.32 s | 2.8× |

The post's headline says 4×; its own table shows 2.8×. The post also says roles are assigned automatically from device profiles, which the public code does not do (§3.8). Mac–Spark connectivity was described by the developers as "still super dodgy", and depended on a Linux kernel patch **[verified: paper/docs]**.

### 9.2 Datacenter disaggregation

All **[verified: paper/docs]** unless tagged.

- A 2024 disaggregation paper: "up to 7.4× more requests or 12.6× tighter SLO" by placing prefill and decode according to cluster bandwidth **[reported by a surveyed system]**.
- A 2025 heterogeneous-GPU paper: disaggregation via graph partitioning plus max-flow, up to 2.0× throughput, 1.3× on average **[reported by a surveyed system]**.
- A 2024 KV-centric serving paper: "up to 525%" throughput in simulation and "75% more requests" in the authors' production service **[reported by a surveyed system]**.
- Datacenter engines run P/D with an RDMA KV transfer engine. Decode → prefill heartbeats default to 5 s, decode aborts on transfer timeout, and prefill-side reclaim exists for some transfer backends but not all.
- A Kubernetes-based stack combines prefix-cache-aware routing, P/D, wide EP and KV tiering to CPU and disk. Vendor numbers include "50k tokens/sec" with wide EP on 16×16 B200 **[reported by a surveyed system]**.

### 9.3 Fit for home clusters [inferred]

- Disaggregation pays off for multi-user throughput with RDMA-class KV transfer.
- For a home cluster at batch 1–8, prefill on the strongest GPU and decode across the pipeline could help TTFT. The obstacle is moving KV: a 70B GQA cache is 320 KiB/token, so 4K tokens take about 27 s at 400 Mb/s (§5.4). It is only viable on wired 10GbE+ or Thunderbolt, and only when the overlap condition holds.

---

## 10. Fault tolerance

### 10.1 Behaviours observed

| Approach (source) | Detection | Mid-generation behaviour | Rebalance |
|---|---|---|---|
| Activation-history replay (2023 internet swarm) | RPC failure | **ban the server, replay cached stage inputs to a replacement, continue** **[verified: code]** | servers move spans on ≥ 20% gain |
| Request migration (datacenter disaggregated stack) | request-path failure | **move the request to a healthy worker**; failed worker excluded for 5 s **[verified: paper/docs]** | planner |
| Heartbeat then abort (volunteer-GPU scheduler) | heartbeat, 30 s default | request times out and is aborted **[verified: code]** | global re-bootstrap on leave |
| Topology withdrawal (2026 QUIC runtime) | gossip peer-down, grace period | "If a locked stage is lost, the topology becomes unavailable and is withdrawn after the normal stage-loss grace period" **[verified: paper/docs]** | – |
| Group restart (datacenter engine on an actor framework) | actor failure | the whole DP group restarts atomically **[verified: paper/docs]** | elastic EP, manual or by API |
| Abort on transfer timeout (datacenter P/D) | 5 s heartbeat, transfer timeout | decode aborts the request **[verified: paper/docs]** | router |
| In-runtime link failover (datacenter transfer engine) | per-path telemetry | link failover inside transfers; collectives survive rank failure **[verified: paper/docs]** | – |
| Instance deletion (event-sourced cluster runtime) | 15 s node silence | any instance including the node is deleted at once; in-flight requests fail **[verified: code]** | none: the user must re-place |
| Stall (home-cluster study) | socket receive timeouts only | stall **[verified: code; outcome inferred]** | solver re-run when the queue is empty |
| Abort (TP-only runtime, star runtime, RPC layer, libp2p federation) | socket error | abort **[inferred]** | none; the federation layer "cannot add workers once inference has begun" |

### 10.2 Activation-history replay

- **Mechanism [verified: code + paper/docs].** The client keeps the inputs it sent to each span for the current sequence. On failure it bans the server, picks a replacement covering the same blocks, replays the history to rebuild that server's KV cache, and continues. Cost is O(t) data for the failed span only.
- **Estimate: memory cost.** `n_ctx × e × bytes` per stage boundary. At 4K tokens, `e = 8192` and f16: `4096 × 8192 × 2 = 64 MiB` per boundary, cheap on the head.
- **Performance of that swarm [reported by a surveyed system]**, sequential steps/s for 128-token generation:
  - Llama 2 70B (NF4) on 3× T4: 2.29 steps/s at 1 Gb/s and under 5 ms; 1.57 at 100 Mb/s and 100 ms RTT; offloading baseline 0.139.
  - BLOOM-176B on 3× A100: 1.71; local NVLink PP: 2.46.
  - Headline: "up to 10× faster than offloading".
- It is the only surveyed system that implements mid-generation recovery for pipelines. Its stack (Python, a bundled Go libp2p daemon, a public-swarm trust model, `pip install` on CUDA Linux only) is not reusable; the replay mechanism and the hysteresis rule are.

### 10.3 Request migration

- Off by default (migration limit 0). When enabled, a request whose worker dies mid-stream moves to a healthy worker and "continue[s] from the exact point of failure". The failed worker is removed from routing for 5 s. Migration is disabled for guided decoding and beyond a sequence-length cap **[verified: paper/docs]**.
- The docs do not state the mechanism. Re-sending prompt plus generated tokens for re-prefill is the obvious implementation **[inferred]**.
- It is simpler than replay and always available, at the cost of a full prefill.

### 10.4 Elastic EP and group restart

All **[verified: paper/docs]**.

- An API call changes the data-parallel size. New ranks start with placeholder weights and get non-expert weights copied; an expert load balancer reshuffles experts; a two-stage barrier keeps ranks aligned.
- Unplanned failure is **not** handled automatically yet; it is described as "a core building block for fault tolerance". An EP transfer library's connect/disconnect-ranks calls can detect and recover EP-side failures.
- Limited to data parallel with TP = 1.
- With the actor framework, when one member of a DP group fails, the whole group restarts atomically.

### 10.5 Runtime-level failure handling in the event-sourced runtime

All **[verified: code]** unless tagged.

- A node silent for 15 s is timed out, and any instance that includes it is deleted immediately. If the leader's own loop stalled for more than 5 s, it first grants 15 s of reconnect allowance.
- **No automatic re-placement.** Placement is only ever triggered from the API, so after a node loss the user must re-place the model.
- If any runner of an instance fails, every runner of that instance is shut down. Recreation uses keyed exponential backoff.
- RDMA failures are not recovered in-process (§5.5) **[verified: paper/docs]**.
- **Control-plane collapse under load.** On a 2-node M5 Max + M2 Ultra running Qwen3.5-397B, logs show "All peer queues full, dropping message on local_events → broken pipe → re-election → worker teardown". A maintainer attributed it to a non-leader dying and rejoining **[verified: paper/docs]**. This is consistent with tokens riding the control plane and one election per connection change **[inferred]**. A later batch of fixes targeted the same class: discovery after stalls, one-way splits, split-brain leaders, late-joiner catch-up **[verified: code]**.

### 10.6 Stragglers and rebalancing

- None of the home systems detect slow nodes at runtime. The layer-window solver re-plans only when idle, from profiles; the internet swarm and the volunteer-GPU scheduler re-measure throughput and RTT continuously **[verified: paper/docs]**.
- Rebalancing rules seen: move only on ≥ 20% gain (swarm); joins take the bottleneck layers and leaves trigger a global re-bootstrap past a load-variance threshold (volunteer scheduler); re-run the solver "whenever the task queue is empty" because nothing migrates (home-cluster study) **[verified: paper/docs]**.
- The home-cluster study lists "device join, dropout" with neighbours reconnecting as a feature, but the code has no heartbeat, no re-solve on failure and no request recovery. In practice a lost device stalls the ring and the job restarts **[verified: code; outcome inferred]**.

---

## 11. Speculative decoding across devices

- **Draft on the head, verify through the pipeline** (home-cluster study) **[reported by a surveyed system]**. The draft runs as a separate process on the head device; 25–45% lower TPOT for 14–70B models. Testbed: 4 Linux nodes with 8 GiB RAM and 600 MB/s disks, two of them with an RTX 4090 capped at 11 GiB:
  - Qwen-2.5-32B: 55 → 38 ms/token with a 0.5B draft on GPU, about 26 tok/s.
  - Llama 3.3-70B: 803 → 442 ms/token with a 3B draft.
- **Why it fits PP [inferred].** Verifying `γ` drafted tokens costs about one pipeline traversal, the same as one token, so speculation amortises the per-hop latency that dominates home clusters.
- **Asynchronous pipelined speculation** (a 2024 paper on a ggml-based runtime with MPI between nodes). The target and draft pipelines run loosely coupled, with early cancellation of invalidated runs; "up to a 2.15× improvement" over standard speculative inference **[reported by a surveyed system]**.
- **Tree-based speculation for sparse edge pipelines** (a 2025 paper): keeps the pipeline busy, 1.28–1.79× **[reported by a surveyed system]**.
- **Large draft trees for offloaded targets** (a 2024 paper): "up to 20 tokens per target model iteration"; 50B+ models on consumer GPUs with RAM offload at 4–6 tok/s (4-bit) **[reported by a surveyed system]**. When each target pass is I/O-bound, verify wide trees **[inferred]**.
- The 2026 QUIC runtime records speculative settings per model family in its certification record; none are enabled by default yet **[verified: paper/docs]**.

---

## 12. Install, packaging, API and UX patterns

### 12.1 Install and packaging

| Pattern | Details |
|---|---|
| Single small binary with OS installers | `curl ... \| bash`, Homebrew, a PowerShell installer for Windows, Ubuntu and Arch packages, OCI images; about 18 MB (2026 QUIC runtime) **[verified: paper/docs]** |
| Menu-bar app wrapping a bundled interpreter | DMG or Homebrew cask, macOS 26.2+ only. A SwiftUI menu-bar process launches a PyInstaller-bundled node, asks for permission to modify system settings, installs a network location and a LaunchDaemon, and needs Local Network permission (event-sourced runtime) **[verified: code + paper/docs]** |
| Source build with many prerequisites | Xcode Metal toolchain, `uv`, `node`, `rust`, a pinned fork of a monitoring tool (the packaged one crashed on M5), then a dashboard build and a Python sync; or a Nix flake. Linux is source-only with CPU or CUDA extras; Windows is unsupported. Source builds have broken on Metal toolchain changes **[verified: code + paper/docs]** |
| Source build, per-rank flags | gcc, make, cmake, `fio`, `libzmq` and HiGHS (built from source on Linux). Rank 0 needs a different build flag to link the solver; CUDA needs another. Every device runs a different command line with hand-set IPs and ranks and downloads the model itself. A Docker image simulates a cluster on one host (home-cluster study) **[verified: paper/docs]** |
| Launcher script | `python launch.py <model>` downloads the converted model and builds the binary (TP-only runtime) **[verified: paper/docs]** |
| pip on CUDA Linux | Python/PyTorch (internet swarm) **[verified: paper/docs]** |
| venv with pinned datacenter engines | an install script creates a venv; the GPU stack pins exact datacenter-engine versions and the Mac stack pins `mlx==0.31.2` (volunteer scheduler) **[verified: code]** |
| Mobile | the star-topology runtime ships iOS and Android apps; the home-cluster study runs phones under Termux; the memory-proportional cluster runtime failed on a phone because it needed root **[verified: paper/docs]** |
| Datacenter engines | the home-cluster study's authors "tried to deploy" the two mainstream datacenter serving engines and "they failed to run on our home cluster"; both also assume NCCL-class networking **[verified: paper/docs]** |

**Dependency fragility.** One runtime builds on personal forks of its framework (for Linux wheels), of the framework's LM package (carrying its own patches for batch generation, detokenizer reuse and YaRN RoPE) and of an image-generation package, plus a pinned fork of a monitoring tool. It dropped its earlier portable tensor framework and became single-framework **[verified: code]**. Its stack is a Python asyncio node (FastAPI on hypercorn, pydantic), Rust networking through PyO3 bindings, MLX as the only compute framework, a Svelte dashboard and a Swift menu-bar app **[verified: code]**.

**Platform coverage.** In that runtime, Apple Silicon macOS is the only tier-1 platform. Linux CUDA exists in code and packaging (custom framework wheels, NVML backend detection, TCP ring only, no VRAM accounting) but is listed as "planned"; Linux Vulkan on AMD desktops is tier 2; Windows is "longer term"; AMD ROCm, Vulkan and Raspberry Pi are unsupported **[verified: code + paper/docs]**. User demand matches: the four most-requested items are Linux CUDA, AMD APU desktops, Linux RDMA and Raspberry Pi, followed by Vulkan and Linux CPU. Issue-tracker keyword hits: "cuda" 85, "windows" 48, "rocm" 10 **[verified: paper/docs]**. Other user requests: GGUF, embeddings, Docker, GPU offload for prompt processing, CXL memory, and multimodal input. Of the other systems, the home-cluster study lacks Windows, Vulkan and ROCm; the star-topology runtime covers CUDA, Metal, Vulkan, CPU and mobile; the TP-only runtime covers CPU (ARM NEON, x86 AVX2) and experimental Vulkan **[verified: paper/docs]**.

### 12.2 API surface

From the event-sourced runtime, served by every node **[verified: code]**:

- **Inference:** OpenAI Chat Completions (streaming SSE with keepalive), OpenAI Responses, Anthropic Messages, the common local-runner REST API (chat, generate, tags, show, ps, version), and image generation and edits. Tool calling and reasoning parsers exist per model family. Supporting these let third-party client tools work against the cluster unchanged **[verified: paper/docs]**.
- **Cluster and instance:**
  - placement previews, placement, place instance, create instance
  - await instance (SSE "ready"), delete instance, P/D instance links
  - full state as JSON, events
  - models, model search, add model
  - downloads
  - distributed traces (when enabled)
  - a benchmark endpoint used by the bench scripts
- There is no CLI for placing or chatting; both go through the HTTP API. CLI flags: verbosity, force-leader, no-API, API port, no-worker, no-downloads, offline, namespace, pub/sub and discovery ports.

### 12.3 Dashboard and operator UX

- Web dashboard on every node: topology view, model picker with fit estimates, placement chooser (PP/TP × TCP/RDMA), downloads matrix, chat with a logprob visualiser, prefill progress **[verified: code + paper/docs]**.
- Single-machine first, route by model name, stage 0 routable only when all stages are ready, and split mode fails closed for uncertified architectures (2026 QUIC runtime) **[verified: paper/docs]**.
- Placement is explicit and user-driven in the event-sourced runtime: requests never trigger it (§2.3).

---

## 13. Observed failure modes and anti-patterns

Each item: what was done, what went wrong, and the evidence.

1. **Tokens routed through a central leader.** Every generated token is JSON-encoded, sent to the leader, indexed, and broadcast to every node over the control plane. Parsing those events took about half of a follower's time under load; throughput is capped at 800–1,000 events/s; and under load the queues fill, pipes break and elections cascade (§2.3, §10.5) **[verified: code + paper/docs]**.
2. **Re-election on every connection change, and in-flight requests failing on leader change.** Membership churn resets the request path (§2.2) **[verified: code]**.
3. **Placement that ignores compute and bandwidth.** Memory-proportional splits give slow, RAM-rich devices more layers, and "fewest nodes wins" ignores link quality entirely. Memory-proportional was 3.3× slower than a cost-based solver at 70B (§3.1, §3.3).
4. **Exhaustive cycle enumeration.** `simple_cycles` grows from 16,064 cycles at 8 nodes to 1.1 million at 10, on every placement and preview call (§3.2) **[inferred from code]**.
5. **Sizing GPU shards against system RAM.** VRAM is never queried; on a discrete-GPU Linux box the GPU path is sized by `psutil` free RAM (§3.8) **[verified: code]**.
6. **Memory accounting that omits KV, activations and duplicated embedding/LM-head,** and TP placement that checks only the cluster sum. The result is OOMs and refused near-limit fits (§3.9) **[verified: code + paper/docs]**.
7. **Communication costs zeroed in the solver.** The paper's objective has a per-hop RTT term; the released code sets its coefficient to 0.0 and comments out the memory-bandwidth term. Decisions the paper attributes to RTT can only come from a side rule in the code (§3.3) **[verified: code]**.
8. **The OS page cache as residency manager.** Kernel LRU decides what stays in RAM, which produces the prefetch-release conflict, four OS-specific memory models and Metal reload storms. There is no `O_DIRECT` or io_uring path (§8.1) **[verified: code + paper/docs]**.
9. **Manual IP configuration.** World size, rank, coordinator IP, next-hop IP and open port ranges set by hand on every device; or a worker list on the root; or the same file path on every worker (§1.1) **[verified: code]**.
10. **Discovery confined to one multicast domain,** with an ephemeral node id on every start and static peers removed. Tailscale, routed and cloud setups cannot join (§1.3) **[verified: code]**.
11. **Star topologies.** Activations between two remote stages return through the client or coordinator, doubling hop latency at every boundary (§2.6) **[verified: code]**.
12. **No authentication or encryption** on several data planes, and an HMAC handshake that still sends plaintext (§1.4).
13. **Full-model download on every node from the internet,** with the per-shard filter written and then disabled (§7) **[verified: code]**.
14. **Even-split-only TP,** power-of-two node counts, and TP limited to node counts that divide hidden size and KV heads. Three-node TP rarely works (§3.1, §4.4).
15. **RDMA that needs identical OS builds and full reboots.** Errors are not recovered in-process (§5.5) **[verified: paper/docs]**.
16. **No recovery when a stage dies.** Instances are deleted, rings stall, splits are withdrawn, requests abort (§10.1).
17. **Claims ahead of code:**
    - README and blog describe latency/bandwidth-aware placement, device profiling and automatic P/D roles that the public code does not do (§3.8).
    - A "4× faster" headline over a table showing 2.8× (§9.1).
    - A "99% latency reduction" with no measurement behind it (§5.5).
    - "Up to 1.8× speedup on 2 devices and 3.2× on 4" for TP, with no model or methodology given **[verified: paper/docs]**.
    - A GPU-serving backend value advertised whenever NVML sees a GPU, with no code path that uses it **[verified: code]**.
    - Docs that say mDNS over code that uses UDP broadcast (§1.1).
    - A README saying Linux runs on CPU only while the code ships CUDA extras and backend detection **[verified: code + paper/docs]**.
    - A paper whose objective and decisions the released code does not implement (item 7).
    - Benchmark bars whose stated quantization cannot fit the stated machine (§4.5).
18. **Made-up capability numbers.** "TFLOPS" computed from VRAM size and clock speed, or from system RAM × 0.4 (§3.1) **[verified: code]**.
19. **Shelling out to `fio`** for disk profiling, writing a temp config into the working directory; and HiGHS built from source as an install prerequisite (§3.3, §12.1) **[verified: code]**.
20. **Fixed limits baked in:** 32-device arrays (§3.3), a single model per cluster and no adding workers once inference has begun in the federation layer (§10.1) **[verified: code + paper/docs]**.
21. **Single-framework lock-in on personal forks,** with Linux GPU support "planned" for most of a year (§12.1) **[verified: code + paper/docs]**.
22. **Raw f32 on every hop without measurement,** and the opposite, compression without a correctness gate. One runtime's f16/q8 attempts failed in production (§6).
23. **An equal-window constraint `L = k·W`** that the runtime does not need, which narrows the solver's choices (§4.2) **[inferred]**.
24. **Cycle counts, version strings and RDMA builds that must all match** for a cluster to form, with only a log line to tell the user (§1.2, §5.5) **[verified: code + paper/docs]**.

---

## 14. Gaps across the field

1. **Tier-aware, multi-device placement with a real network term.**
   - The layer-window ILP is the only multi-device solver that models disk, and its released code drops communication.
   - Datacenter solvers model links but not RAM or disk.
   - Nobody models VRAM, RAM, NVMe and links together per device and solves for device selection.
2. **An explicit tier manager instead of the OS page cache.**
   - The home-cluster study, the disk-streamed-expert runtime and plain mmap loaders all delegate residency to kernel LRU. Hence the prefetch-release conflict, the four OS-specific memory models and Metal reload storms.
   - No distributed system manages its own pinned-buffer residency with direct NVMe reads (`O_DIRECT`, io_uring, `F_NOCACHE`) and a prefetch window sized by the solver.
3. **MoE expert tiering across devices.** One small project formulates it as an MILP, and two runtimes stream experts from SSD on one node. Nobody places hot experts in VRAM, warm ones in RAM and cold ones on NVMe *across* devices with a solver, with runtime re-tiering from routing statistics.
4. **Mid-generation recovery on consumer clusters.** History replay exists only in a 2023 internet swarm; request migration exists only on Kubernetes datacenter stacks. Every home-cluster system aborts or stalls.
5. **Zero-config plus secure plus single binary plus heterogeneous OS.** The 2026 QUIC runtime is closest on UX but lacks tiering and recovery. The home-cluster study has the placement but is the hardest to install (manual IPs, `fio`, HiGHS from source, no Windows, CUDA and Metal only). The RPC layer and the home-cluster ring have no auth or encryption.
6. **Direct stage-to-stage forwarding with a coordinator-free ring on LAN.** The RPC layer and the star-topology runtime pay double hops; the home-cluster ring has no security or discovery.
7. **Speculation designed around pipeline latency.** Only draft-on-head ships. Asynchronous pipelined speculation and wide trees for I/O-bound targets have not been combined with tiered multi-device inference.

---

## 15. Recommendations for Syntropy

Where ADR-0000 already settles a question, the directive is cited.

### 15.1 Discovery, membership and identity

1. **LAN first, zero config** (ADR-0000 D4). Advertise with mDNS/DNS-SD plus IPv6 multicast, with a UDP-broadcast fallback for networks where mDNS is blocked **[inferred]**.
2. **Static peers and a rendezvous/Tailscale mode from day one,** so routed, VPN and cloud nodes can join. Off-LAN devices join with an invite token over Tailscale or a relayed or hole-punched QUIC path (D4).
3. **Persistent node identity = a public key** (iroh-style QUIC endpoint). Encrypt by default; never ship a plaintext data plane.
4. **Mutual authentication from a cluster secret before any model traffic.** An HMAC challenge-response is the minimum bar, and it must sit under encryption.
5. **Version-hashed namespace,** so mixed versions never join, and report the mismatch to the user in the UI, not only in a log.
6. **Stall guard:** a node waking from suspend backs off long enough for stale sessions to expire before re-linking.
7. **Gossip membership plus a 1–2 s heartbeat on LAN,** which notices a failure within one or two tokens. A 30 s timeout suits volunteer WANs, not a home LAN.
8. **Store the IP and interface each topology edge uses,** with per-edge transport type (socket or RDMA) and interface priority, so the data plane picks the fastest link. Detect Thunderbolt Bridge loops, and revert any host network changes on uninstall.
9. Public swarms of untrusted machines stay out of scope for the first release (D4).

### 15.2 Control plane

1. **Event-sourced, single-writer cluster state with a pure `apply()`,** typed commands and events, a whole-state debug endpoint, and snapshot catch-up for laggards from day one. Unit-test placement, apply and election as pure functions.
2. **Keep tokens and activations off the control plane.** Stream tokens point-to-point from the sampling rank to the API node.
3. **Membership churn must not reset the request path.** No re-election on every connection change; a leader change must not fail in-flight requests. Use anti-flap rules and back-off.
4. **One runner process per model instance under a supervisor,** with stall timeouts, crash isolation from the node daemon, and a lifecycle state machine (connect → load → warm up → ready) driven by a pure `plan()` over cluster state. Bring ranks up in a fixed order to avoid collective deadlocks.
5. **Drop rather than block on a slow peer** in control traffic, and recover by NACK/resend.
6. **Two-phase scheduling:** slow placement between requests, fast per-request routing that is RTT-aware.
7. **A filter → score front end** before the solver (backend compatibility, labels, model-file locality), with the VRAM/distribute/offload choice made by cost, not a fixed order.
8. **Cluster-consistent decisions inside collectives:** all-gather task order, sampling state and eviction or tier-migration flags so every rank takes identical branches.
9. **Treat the head as a single point of failure in v1,** and say so explicitly in the RFC.

### 15.3 Placement and cost model

1. **Extend the layer-window ILP.** Per device `m`, the decision variables are: layers resident in VRAM, layers resident in RAM (computed in place, per D1), layers streamed from NVMe, expert groups per tier for MoE, and a binary "selected" variable. Device selection then comes out of the solve.
2. **Include the per-hop RTT and serialisation term and the per-backend memory-bandwidth term** that the released solver drops.
3. **Drop the equal-window constraint.** Model the batch-1 decode critical path and the bottleneck stage for batched throughput as two objectives, with batch-1 latency as the default (D2) and throughput as a selectable mode.
4. **Solve with HiGHS** (MIT, embeddable, 10–12 ms at 4–32 devices in the study), linked into every build, not a special rank-0 build. Keep a DP over contiguous layer ranges as a solver-free fallback. Never enumerate cycles exhaustively; use a ring-order heuristic.
5. **Minimise stage count under a latency budget.** Prefer dropping a weak device to adding a hop (146 versus 156 ms in the study; 5 tok/s ceiling at 2 stages × 100 ms over WAN).
6. **Measure, don't look up.** Profile FLOPS per backend and quant type (fp16, int8), memory bandwidth per tier (VRAM, RAM, PCIe, NVMe sustained read), and link latency and bandwidth per edge. Bootstrap from profiles, then feed live per-stage timings back in. Profile disk in-process, not by shelling out to `fio`.
7. **Discrete-GPU correctness:** query VRAM capacity, cost PCIe transfers, and support several GPUs per node (NCCL inside the node, the network across nodes).
8. **Budget every byte per device and per tier:** weights + KV(context, batch) + activations + runtime overhead, plus duplicated embedding and LM-head on every stage that holds them. Check each node, never only the cluster sum, and account for instances still loading.
9. **Split layers by throughput subject to per-tier capacity,** not by RAM. Keep the careful mechanics: largest-remainder rounding, at least one layer per selected node, per-node feasibility checks.
10. **TP and EP only inside fast islands** (same host, RDMA over TB5, RDMA NICs, or wired ≥10GbE with sub-millisecond syncs). The solver treats an island as one virtual device with internal TP. Support uneven TP and PP+TP hybrids from the start.
11. **Re-plan between requests only,** when the measured critical path deviates from the model by more than about 20%, with hysteresis. Never re-plan mid-sequence unless a node dies.
12. **Pin recurrent and hybrid-layer state to a sticky owner,** and keep a certified-architecture roster so split mode fails closed for untested architectures.
13. **Verify before adopting.** Re-run the study's 10-node example in simulation with the network term restored, and check whether its reported device-selection decisions still come out.

### 15.4 Transport and wire format

1. **A ring of direct stage-to-stage streams.** The head does tokenisation, sampling, drafting and output. No star through a coordinator.
2. **One persistent stream per neighbour:** QUIC streams, or TCP with `TCP_NODELAY`. Fixed, preallocated activation frames with a small header (sequence id, position range, dtype, shape).
3. **Graph and plan reuse,** so per-token messages carry only activations, as with replay-by-id in RPC designs.
4. **f32 on the wire by default** (D3: lossy activation compression is opt-in and labelled approximate). bf16 or Q8 can be enabled per model family behind a versioned protocol and a correctness gate. The only place it clearly pays is chunked prefill on slow links, so measure f32, bf16 and Q8 per model family for decode and prefill separately.
5. **One data-movement abstraction for disk → RAM, RAM → VRAM and node → node,** in the "describe what to move" style with per-slice path scheduling, telemetry-driven path choice and in-runtime failover, so the tier manager and the network share scheduling and back-pressure.
6. **RDMA where available** (MLX-style RDMA over TB5 on Apple meshes, RoCE/libibverbs on Linux), with TCP as the fallback. Ring-over-TCP TP is not a fast path.

### 15.5 Weights

1. **Content-addressed layer and expert chunks, pulled from any peer that has them and verified by digest.** A new node fetches only its assigned ranges; a placement change moves only the deltas.
2. **Download once from Hugging Face Hub, then distribute over the LAN.** The coordinator never needs the full file in RAM. Each node keeps a hash-addressed local cache, so a re-placement or restart never resends weights it already holds.
3. **Keep the good downloader mechanics:** parallel files, resume, rate-limit handling, multiple model directories, read-only shared directories, a true offline mode. Streaming with resume and checksums is the minimum.
4. **Support formats beyond one framework's safetensors,** GGUF at minimum, so one model artifact works across engines and tiers.

### 15.6 Memory tiers and KV

1. **Own residency explicitly;** do not rely on the OS page cache. Use pinned buffers and direct NVMe reads (`O_DIRECT`/io_uring on Linux, `F_NOCACHE` on macOS).
2. **Bound the in-flight prefetch window to what the RAM tier can actually hold,** sized by the solver.
3. **Low memory pressure on personal devices is a product requirement.** Yield RAM when the user needs it, and model the TPOT cost of doing so.
4. **Page KV to RAM and NVMe,** and reuse prefix KV across restarts.
5. **MoE expert tiering across devices:** hot experts in VRAM, warm in RAM, cold on NVMe, placed by the solver and re-tiered at runtime from routing statistics.

### 15.7 Prefill/decode disaggregation

1. **Pipelined chunked prefill** across stages, with staggered start.
2. **Layer-streamed KV transfer for P/D,** with the overlap condition `P/B < K·s/q` as a cost-model term deciding when to disaggregate, and the analogous term for streaming KV or weights from NVMe while computing.
3. **Assign P/D roles automatically from the cost model,** not by hand. Expect it to pay only on wired 10GbE+ or Thunderbolt links.

### 15.8 Fault tolerance

1. **Activation-history replay.** The head keeps, for every active sequence, the activations it sent across each stage boundary (≈64 MiB per boundary at 4K tokens in f16). On stage loss:
   1. re-run the solver with the survivors
   2. assign the lost range to the best candidate, which fetches its layers by content address if needed, or absorb the lost layers into RAM/NVMe tiers on survivors as a degraded mode
   3. replay the boundary history to rebuild KV
   4. resume
2. **Request migration as the universal fallback.** If replay is impossible (not enough memory, or the head died), re-prefill prompt plus generated tokens on the new plan.
3. **Re-place automatically** after node loss; never wait for the user.
4. **Stragglers:** publish per-stage step time every token and feed it into the next placement. Never kill a slow-but-alive stage mid-sequence.
5. **Recover transport errors in-process,** RDMA included, without rebooting nodes.

### 15.9 Speculative decoding

1. **Ship draft-on-head first** (25–45% lower TPOT in the study).
2. **Then evaluate asynchronous pipelined speculation and wider trees when the target is I/O-bound.** Pipeline latency and disk latency are exactly the regimes where verification is nearly free.

### 15.10 Engines, platforms and packaging

1. **An engine abstraction with first-class CUDA, ROCm, Vulkan and CPU, with Metal as one backend among several.** No dependence on personal forks.
2. **Linux RDMA/NCCL and a Windows story** from the start; Linux GPU and Windows are where user demand is.
3. **One small binary per OS** with curl, Homebrew, PowerShell and distro-package installers. No interpreter, solver or `fio` to install separately.

### 15.11 API and UX

1. **API breadth as table stakes:** OpenAI Chat Completions and Responses, Anthropic Messages and the common local-runner API, with streaming, tool calling and reasoning parsing (API shapes are permitted by D5).
2. **Single-machine first:** if one device fits the model, serve locally; route by model name; a split becomes routable only when every stage is ready.
3. **Placement previews with per-node memory deltas and fit estimates** in the dashboard, but requests should trigger placement automatically rather than returning 404.
4. **Docs track what ships.** Benchmarks are tied to a commit, with prompt length, TTFT, quantization and interconnect stated. No capability claim ahead of code, and code that implements a published design (RFC or paper) must not silently diverge from it.

### 15.12 Open questions

1. **Target workload.** Settled by D2: batch-1 latency (ring PP, minimum stages) is the default; small multi-user throughput (more pipelines, per-request routing) is a second solver objective.
2. **Engine base.** Build on ggml (GGUF, quant kernels and backends for free, but fighting its scheduler for tiering) or write our own runtime? This decides how portable a layer-window solver implementation is.
3. **Activation precision.** Defaults are settled by D3 (f32, lossy opt-in). The per-model-family measurement of f32 versus bf16 versus Q8, for decode and prefill, is still needed.
4. **Windows, Vulkan and ROCm.** If "installs on any device" is literal, the GPU backend matrix drives the engine choice.
5. **Head failure and multi-head.** Is a single head acceptable for v1, or should the per-sequence boundary history be replicated to a standby?
6. **Cross-network meshes.** Settled for v1 by D4: LAN plus invite links over an overlay; no public swarms.
7. **Unverified items carried from the survey:**
   - how rustworkx `simple_cycles` treats parallel edges in a per-IP multigraph (§3.2)
   - whether host network-location changes made by a macOS installer are reverted on uninstall in all cases (§1.6)
   - whether framework CUDA builds run TP correctly on multi-node Linux; no public evidence either way
   - whether vendor claims of automatic heterogeneous placement and device profiling exist in unpublished code; treat them as unverified (§3.8)
