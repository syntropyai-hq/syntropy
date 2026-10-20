# Glossary

- **Status:** Accepted (2026-10-20)
- **Date:** 2026-10-19

This is the canonical vocabulary for every Syntropy plan document, RFC and ADR, and for the code. Where a term has a type in code, its name is given in `code` font. Names marked *(proposed)* are placeholders until RFC-0001 fixes them. "Owner" is the RFC or ADR that defines the behaviour behind the term; it names a document, never a person (§16). The words in [§15, Rejected synonyms](#15-rejected-synonyms) are not used anywhere.

## 1. Cluster, nodes and processes

**Host.** One physical machine. In normal operation a host runs one node. *Owner:* RFC-0010.

**Node.** One running Syntropy daemon (`syntropy serve`) on a host, identified by its node ID. The daemon runs the control plane, API, discovery and model store, and it supervises runners. *Owner:* RFC-0030.

**Node ID.** A node's persistent Ed25519 public key. It is also the node's QUIC endpoint identity, so authenticating a connection authenticates the node. *Code:* `NodeId`. *Owner:* RFC-0034.

**Cluster.** The set of nodes admitted under one cluster key. A node becomes a member through LAN discovery or an invite token. *Owner:* RFC-0030.

**Cluster key.** The Ed25519 signing key, held by keyholders, under which a cluster admits nodes: each member holds an admission certificate signed with it (ADR-0313). Every member is trusted with the weights and activations it handles (D4); trust among members is tiered by certificate role. *Owner:* RFC-0034.

**Invite token.** A credential plus rendezvous information that lets a device on another network join a cluster over an overlay path: a relay, a hole-punched QUIC connection or Tailscale. Lifetime and scope are set by RFC-0034. *Owner:* RFC-0030, RFC-0034.

**Coordinator.** The elected single writer of the cluster event log. It is elected with an epoch-fenced lease, prioritised by stable node ID and capability. No majority quorum is needed, because a two-node cluster must survive losing one node. Only nodes whose certificate carries the `COORDINATOR` role are eligible. The coordinator is never on the token path, and a change of coordinator fails no in-flight request (ADR-0009). *Owner:* RFC-0032 (election, lease, terms); failover in RFC-0033.

**Cluster event log.** The append-only sequence of `ClusterEvent`s that the coordinator writes. Every node validates each entry and then folds it into cluster state with a pure `apply()`. Snapshots bound replay, and a lagging node catches up from a snapshot plus the tail of the log. *Code:* `LogEntry`, `ClusterState`. *Owner:* RFC-0032.

**Cluster command.** A typed request to change cluster state, sent to the coordinator, which validates it and records the resulting events. *Code:* `ClusterCommand`. *Owner:* RFC-0032 (the type and catalogue); variants from RFC-0030, RFC-0011, RFC-0033, RFC-0034 and others.

**Cluster event.** A typed, immutable record of one change to cluster state. *Code:* `ClusterEvent`. *Owner:* RFC-0032 (the type and catalogue); variants from RFC-0030, RFC-0011, RFC-0033, RFC-0034 and others.

**Lease epoch.** The number attached to the coordinator's lease. Commands and events that carry an older epoch are rejected, so a deposed coordinator cannot write ("epoch fencing"). Not to be confused with the plan epoch. *Owner:* RFC-0032.

**Membership.** Which nodes the cluster believes are alive. It uses SWIM-style gossip with 1 s heartbeats on a LAN, plus anti-flap and back-off rules so that one flapping connection cannot trigger repeated elections or re-placements. *Owner:* RFC-0030.

**Ingress node.** The node that received an API request. It holds the client connection and receives the token stream point to point from the tail stage. *Owner:* RFC-0032, RFC-0050.

**Runner.** The per-model-instance process on a node that executes that node's stages. The daemon spawns it by re-executing the same binary in runner mode. It owns the devices, tiers and data plane for its model, and talks to the daemon over shared memory plus a Unix-domain socket or named pipe. A GPU driver fault kills the runner, not the daemon (ADR-0015). *Owner:* RFC-0012.

**Runner lifecycle.** The states a runner moves through: `Fetching → Packing → Loading → Warming → Ready → Draining`. A runner is routable only in `Ready`, and a split model is routable only when every stage's runner is `Ready`. The daemon's supervisor applies stall timeouts in every state. RFC-0001 adds two end states, `Failed` and `Stopped`. *Code:* `RunnerState`, with the field-less mirror `RunnerStateKind` used in the event log. *Owner:* RFC-0001, RFC-0012.

**Degraded mode.** Any plan that the solver accepts only because the preferred one is impossible. Examples: one node hosting several stages, a dense model with NVMe overflow in latency mode, or a lost stage absorbed into a survivor's slower tiers. Degraded modes are reported in the dashboard. *Owner:* RFC-0011, RFC-0033.

## 2. Hardware and resources

**Device.** A compute engine on a node: a discrete GPU, an integrated or APU GPU, a CPU complex (one device per NUMA domain) or an NPU. *Code:* `DeviceId`, `DeviceKind`, `DeviceInfo`. *Owner:* RFC-0010.

**NUMA domain.** A set of cores with uniform access to one bank of memory. Each NUMA domain's CPU complex is a separate device, so CPU execute-in-place stays NUMA-local. *Owner:* RFC-0010.

**Tier.** A memory or storage level where bytes can reside. Every tier has an owning node and an affinity to the devices that can execute against it at speed. The tier kinds are:

- **`Vram`**: device-local memory of one discrete GPU.
- **`Unified`**: memory shared by CPU and GPU on APUs and Apple silicon (Apple M-series, Strix Halo, GB10-class). There is no PCIe term. A GPU allocation cap (on macOS, the wired limit) takes the place of a VRAM size.
- **`HostPinned`**: DMA-registered host RAM: locked, and registered with each GPU provider, NIC or I/O ring that copies from it. It holds only bytes some device copies (groups streamed at n*, copied-miss and VRAM-admission sources, staging and network pools). Every OS limits how much can be registered (`pin_limit`).
- **`HostPageable`**: anonymous host RAM, never file-backed, that only the CPU reads. CPU-only weights and the in-place share of a RAM expert cache live here, locked against paging when the budget allows, never DMA-registered (ADR-0200). A device never DMAs from it.

Each NUMA domain has one `HostPinned` and one `HostPageable` tier, and the two share that domain's RAM: `HostPinned ≤ pin_limit` and `HostPinned + HostPageable ≤ host_usable`.
- **`Nvme`**: a storage volume used as a capacity tier and read with direct I/O.

*Code:* `TierKind`; a tier is identified by `TierId` and described by `TierInfo`. *Owner:* RFC-0020.

**Link.** A measured path between two endpoints, with latency α, bandwidth β and a p99 latency. The kinds are memory bus (device ↔ its tier), PCIe (host ↔ device), storage path (NVMe ↔ host) and network (node ↔ node: Ethernet, Wi-Fi, Thunderbolt/USB4 IP, RDMA). Links are directed, because Wi-Fi and other asymmetric paths differ by direction. *Code:* `LinkId`, `LinkKind`, `LinkCaps`. *Owner:* RFC-0010; network kinds RFC-0031.

**Resource graph.** The nodes, devices, tiers and links of a cluster, with their capability profiles. It is the solver's input. *Code:* `ResourceGraph`. *Owner:* RFC-0010.

**Capability profile.** The measured numbers for one device:

- effective bandwidth per (device, tier, kernel class, quant scheme);
- throughput per (device, op class, dtype);
- fixed per-token overhead;
- sustained NVMe read rate at the real request size, after thermal throttling;
- α, β and p99 for each link.

The startup probes produce it, and live timings refine it. *Code:* `CapabilityProfile`. *Owner:* RFC-0010.

**Probe.** A startup microbenchmark that fills part of a capability profile, using the real kernels and request sizes rather than synthetic copies. The budget is 15 s per node (SC-22). The 60–120 s sustained NVMe run continues in the background. *Owner:* RFC-0010.

**Op class.** A group of kernels with the same performance shape, such as quantized GEMV, grouped expert GEMV, attention or linear-attention scan. Throughput is profiled per op class and dtype. *Owner:* RFC-0010, RFC-0040.

**Fast island.** A set of devices whose links pass the TP/EP gate (`α_eff < α_eff,max`). Examples are GPUs in one host over NVLink or PCIe P2P, and nodes joined by RDMA. The solver may treat an island as one virtual device with internal tensor parallelism. *Owner:* RFC-0011.

**Support tier (Tier 1 / Tier 2 / Tier 3).** The support level of a platform under ADR-0020. Tier 1 has CI on real hardware every release, and a failure blocks the release. Tier 2 has a CI build plus periodic hardware runs. Tier 3 is best effort. This is a different sense of "tier" from a memory tier. Plan documents write "Tier" with a number only for support tiers, and name memory tiers by kind. *Owner:* RFC-0052, RFC-0062.

## 3. Describing a model

**Model manifest.** Syntropy's description of a model. It holds:

- the architecture family and its hyperparameters;
- the tensor census;
- byte sizes per quant type;
- the KV bytes-per-token formula.

It is built from GGUF or safetensors metadata. *Code:* `ModelManifest`. *Owner:* RFC-0041, with family rules from RFC-0043.

**Tensor census.** The classification of every tensor in a model by access pattern, with its bytes per quant type. It is computed before anything is placed. *Code:* `TensorCensus`. *Owner:* RFC-0041.

**Census class.** One of five classes in the tensor census:

- **`DenseEveryToken`**: read on every token, such as attention projections, dense FFN layers, shared experts and norms. It belongs in the fastest tier its device can execute from, and it is never streamed during decode.
- **`Expert{layer, id}`**: one routed expert of one MoE layer, read only when the router selects it. It is the unit of expert caching, ownership and NVMe reads.
- **`RowLookup`**: a table addressed by row from token ids, such as an embedding, n-gram or lookup table. Each token reads a few KB, and the addresses are known as soon as the ids are, so these tables can be prefetched and are natural `Nvme` residents.
- **`Head`**: the LM head, the projection from the hidden state to the vocabulary. It lives on the **tail** stage. The head stage holds the embeddings, which are `RowLookup`.
- **`Router`**: the gate that picks experts for each token. It is small. Next-layer prediction is fused into the router op on the router's device, from that device's copy; a host copy is used only where the router runs on the CPU (ADR-0210).

*Code:* `TensorClass` (`Router { layer }` carries its layer). *Owner:* RFC-0041.

**Hot set.** Every byte that a token reads whatever the routing: `DenseEveryToken`, `Head` and `Router` tensors. For DeepSeek-V3 at Q4_K_M it is ≈9.7 GB. Its dtype matters: Kimi-K2-Thinking's BF16 hot set is 20.8 GB per token. *Owner:* RFC-0011.

**Routed pool.** All `Expert` tensors of a model. *Owner:* RFC-0023.

**Routed bytes per token.** `k × n_moe_layers × bytes per expert`, before any caching. For DeepSeek-V3 at Q4_K_M it is ≈12.3 GB. *Owner:* RFC-0023.

**Shared expert.** An expert that every token uses alongside its routed experts. It is `DenseEveryToken`, not `Expert`.

**Latent MoE.** An MoE layer whose experts work in a reduced inner dimension (3584 in Kimi K3), which changes the bytes per expert. *Owner:* RFC-0043.

**Architecture family.** A Rust module that builds the graph for a family of models (for example DeepSeek-V3-class or Gemma-class) from shared blocks. Per-model differences come from manifest hyperparameters. The v1 families are listed in ADR-0003. *Owner:* RFC-0043.

**Shared block.** A reusable graph component: GQA/MHA attention, MLA, sliding-window attention, sparse or indexed attention, linear attention (Gated-DeltaNet- and Mamba-class), dense FFN, routed plus shared MoE, MTP head. *Owner:* RFC-0043.

**Conformance suite.** The tests that a new architecture family must pass, including parity against a reference, before it is enabled. Split execution across stages fails closed for any family that has not passed. *Owner:* RFC-0060, RFC-0043.

**Pack.** The per-device execution layout of a node's shard in the local cache. Every expert and every streamable unit starts on a 4 KiB boundary for direct I/O, and backend-specific repacked layouts are optional. A sidecar manifest records offsets, sizes, hashes and the plan epoch that produced the pack; the learned expert-frequency profile, op-selection entries and canary digest are separately versioned side records that survive a rebuild (RFC-0041 §5.2). It is keyed by (model content hash, device class, shard digest, layout version) (ADR-0005, RFC-0001 §5.6), so an epoch change that leaves a node's shard unchanged reuses the pack instead of rebuilding it. The runner builds it during `Packing`. A pack can be rebuilt and is safe to delete (ADR-0005). *Code:* `PackKey`. *Owner:* RFC-0041.

**Model content hash.** The hash that identifies a model variant (one quantization of one model) by content: BLAKE3 over the family tag and, for each source file the census reads, its role, its SHA-256 (`FileDigest`, the Hub's object ID where published) and its length (ADR-0411). It is computable from headers and file metadata before any weight bytes move, and the same files keep the same ID across releases. Byte-level dedupe across GGUF and safetensors happens at the chunk level, not here. Packs, saved sessions and prefix caches are bound to it, so changed weights invalidate them. The tokenizer and chat template have their own `TokenizerId`. *Code:* `ModelId`. *Owner:* RFC-0041.

## 4. Placement

**Placement solver.** The component that turns a resource graph and a model manifest into a placement plan (ADR-0008). A filter→score front end first drops devices that cannot serve the model. A joint MILP over devices, tiers, links, experts and device selection, solved with HiGHS, then picks the plan, warm-started by a dynamic program over contiguous layer ranges that is also the fallback. The solver is pure: the same inputs give the same plan on every node. It runs in two phases: slow re-placement between requests or on membership change, and fast per-request routing on the ingress node. *Code:* crate `syntropy-solver`; routing is `solver::route`. *Owner:* RFC-0011.

**Latency mode / throughput mode.** The two solver objectives (D2). Latency mode, the default, minimises the batch-1 decode critical path, the time per token of one stream. Throughput mode maximises aggregate tokens per second over concurrent sequences: it balances stage times so each device holds bytes in proportion to its W_eff, and admits sequences up to the KV limit. *Code:* `Objective`. *Owner:* RFC-0011, RFC-0054.

**Instance.** One committed ring of stages serving one model. The coordinator allocates its ID and never reuses it, and the instance keeps that ID across plan epochs. *Code:* `InstanceId`. *Owner:* RFC-0011.

**Placement plan.** The solver's output for one instance, versioned by plan epoch and immutable once committed. It holds:

- the ordered stages, each with its node, contiguous layer range and parallelism mode;
- for every tensor group, its owner, home tier and compute site;
- expert-cache budgets per device, KV budgets per tier, prefetch windows and scratch sizes;
- the speculative decoding configuration and batching limits;
- the cost model's per-term prediction, which runners compare with live timings to detect drift, and the list of inputs that were priors rather than measurements.

It is committed through the event log (`PlanCommitted`), so every node holds the same plan. *Code:* `PlacementPlan`. *Owner:* RFC-0011.

**Plan epoch.** The version of a placement plan: the pair (lease term, sequence number within the term), ordered lexicographically, so a plan committed by a newer coordinator always supersedes one from an older coordinator. Terms embed a 24-bit node tag, so they are unique per coordinator rather than consecutive, and the two sides of a network split never mint the same epoch. Every committed plan gets a new epoch, even for a budget change, while the instance keeps its ID. Every data-plane frame carries its plan epoch, and a runner rejects frames for an epoch it is not serving. Not to be confused with the lease epoch. *Code:* `PlanEpoch { term, seq }`. *Owner:* RFC-0011; terms from RFC-0032.

**Plan delta.** The part of a plan change that a runner adopts in place between steps, without reloading: budgets, prefetch windows, speculative settings, miss split and batching limits; a relink to a different neighbour, transport or stage index (`Relink`); and the extra layers a survivor takes on when it absorbs a lost stage (`ExtendShard`). Relinks and extensions keep the runner's KV and activation history. Any other change to layers, owners or home tiers is a new shard and goes through drain-and-replace. *Code:* `PlanDelta`, `DeltaItem`. *Owner:* RFC-0011; `Relink` and `ExtendShard` RFC-0032.

**Stage.** A contiguous range of layers executed by one node; one position in an instance's ring. A node hosts several stages only in degraded modes. In the cross-node TP and EP modes a stage spans a fast island, and the stage's node is the island's primary member, which holds the stage's ring endpoints. *Code:* `StageAssignment`, indexed by `StageIx`. *Owner:* RFC-0011, RFC-0032.

**Head stage.** Stage 0, the first stage of the ring. It holds the embeddings, receives each new sequence from the ingress node, and runs the draft model when the drafter is a separate model. Like every sending stage it keeps its outbound activation history; its own inputs are token IDs, which the ingress node holds, so a lost head re-prefills only its own layers. *Code:* `StageRole::HEAD`. *Owner:* RFC-0032.

**Tail stage.** The last stage of the ring. It holds the LM head, runs the sampler and any MTP or draft-head drafting, and sends each token both to the head (the ring return) and to the ingress node. In a one-stage plan the same stage is head and tail. *Code:* `StageRole::TAIL`. *Owner:* RFC-0032, RFC-0012.

**Tensor group.** The unit of placement: the solver assigns groups, never single tensors. There is one group per (layer, mixer), per (layer, dense FFN) and per (layer, expert), the last holding the expert's gate, up and down matrices; one per lookup table; one per router; and one for the LM head. *Code:* `TensorGroup`, `TensorGroupId`. *Owner:* RFC-0041.

**Owner.** The single node responsible for a tensor group, and so for each expert, in the cluster. It holds the authoritative copy in its home tier and computes the group there. Experts are never duplicated across nodes unless the plan replicates them deliberately because they are hot. *Code:* `TierAssignment::owner`. *Owner:* RFC-0011.

**Replica.** A deliberate copy of hot data on a node other than its owner, listed in the plan. Replicas are rare and never accidental. *Code:* `TierAssignment::replicas`. *Owner:* RFC-0011.

**Home tier.** The tier that holds the authoritative copy of a tensor group on its owner, fixed by the plan. Expert-cache copies in faster tiers are runtime state in the residency map, not plan state. *Code:* `TierAssignment::home`. *Owner:* RFC-0011, RFC-0020.

**Compute site.** The device that executes a tensor group. At decode and small batch sizes it is a device in the home tier's affinity set, so the group runs in place. From n* tokens up, the plan may name a second device to which the group is streamed over a link, normally PCIe, and computed there. *Code:* `ComputeSite { decode, batched: Option<StreamSite> }`. *Owner:* RFC-0011.

**Shard.** The tensor byte ranges one node must hold locally to run its stages, expressed as tensor groups plus the content-addressed chunks that cover them. A node fetches only its shard (ADR-0016). A shard is bytes held, never a stage. *Code:* `ShardSpec`. *Owner:* RFC-0041.

**Re-placement.** Solving and committing a new plan epoch for a running instance. It is immediate after a node fails, leaves or is revoked, or a runner spends its restart budget. It is deferred after a node joins, after capability or workload drift and under sustained memory pressure, and a deferred plan is adopted only when it clears the hysteresis and payback rules. A path that is lost while both nodes stay reachable is relinked in place. *Owner:* RFC-0011, RFC-0033.

**Drain-and-replace.** How a node moves to a plan epoch that changes its shard or stages beyond what a plan delta covers: a new runner loads for the new epoch and the old runner drains. It is make-before-break when the device ledgers have room for both, and break-before-make, the usual case, when they do not; ingress holds new requests during the gap. *Owner:* RFC-0011, RFC-0033.

## 5. Parallelism and the data plane

**Pipeline parallelism (PP).** Splitting a model into contiguous layer ranges, one per stage, and passing activations from stage to stage. It is the default across nodes (ADR-0006). PP never makes a single stream faster than the same bytes on the same devices; it helps by moving bytes off a slow tier onto another device's faster memory. *Code:* `PlacementPlan::stages`. *Owner:* RFC-0011, RFC-0032.

**Tensor parallelism (TP).** Splitting each layer's matrices across devices, which costs about two collectives (all-reduce or all-gather) per layer per step. Inside a node it runs over NVLink or PCIe P2P when the cost model says it pays. Across nodes it is allowed only inside a fast island whose measured α_eff is below the solver's break-even threshold. *Code:* `ParallelismMode::IntraNodeTp`, `ParallelismMode::CrossNodeTp`. *Owner:* RFC-0032; intra-node execution RFC-0012.

**Expert parallelism (EP).** Placing the experts of an MoE layer on different devices, with a dispatch and a combine per MoE layer. In Syntropy it is a cross-node mode inside a fast island, with the same α_eff gate as TP: attention runs on the stage's primary member and each expert on its owner. Inside one node, the split of experts between GPU and CPU is handled by execute-in-place and the expert cache instead. *Code:* `ParallelismMode::CrossNodeEp`. *Owner:* RFC-0032.

**Parallelism mode.** How a stage runs its layers. `Local` is the default: no layer is split, each tensor group runs on its compute site, and inside an MoE layer cache hits run on the GPU while misses run in place on the CPU, concurrently. The other modes are `IntraNodeTp`, `CrossNodeTp` and `CrossNodeEp`. *Code:* `ParallelismMode`. *Owner:* RFC-0032, RFC-0012.

**Ring.** The directed cycle of an instance's stages. Each stage sends activations to the next, and the tail sends the sampled token back to the head to start the next step. Stages connect directly, never through the coordinator (ADR-0010). Two other rings appear in the plan and are always qualified: the ring all-reduce used by TP, and the SPSC handoff rings between threads. *Owner:* RFC-0032.

**Hop.** One stage-to-stage edge of the ring, with the link and transport the plan chose for it, the wire dtype and its predicted time, α + S/β at batch 1. *Code:* `HopPlan`. *Owner:* RFC-0011, RFC-0031.

**Step.** One unit of work that travels the ring: one token per sequence in decode (or one verify window), or one chunk in prefill. Steps are numbered per instance, and every stage processes them in order. *Code:* `ActivationFrame::step`. *Owner:* RFC-0012, RFC-0032.

**Activation frame.** The stage-to-stage message: the hidden state of a batch of token positions at a stage boundary. It carries the instance, plan epoch, step, the sequence and positions of each row, the wire dtype, and side-channel data such as token IDs for stages that hold `RowLookup` tables, router hints for prefetch, and a new sequence's metadata on its first frame. Kinds are prefill, decode, verify, replay and canary. In exact mode the wire dtype is F32, or BF16 for a family the parity harness has certified for it (a `WireCert`); F16 and Q8 are approximate only. *Code:* `ActivationFrame`, `FrameKind`. *Owner:* RFC-0031 (layout), RFC-0032 (semantics).

**Token frame.** The tail's message carrying each sequence's accepted tokens, sent to the head (the ring return) and to the ingress node. *Code:* `TokenFrame`. *Owner:* RFC-0031 (layout), RFC-0012 (semantics).

**Control, data and bulk planes.** The three kinds of traffic between nodes, kept apart:

- **control**: commands, events, gossip, election and plan distribution, over iroh/QUIC; the same connection carries the **node-query channel**, request/response reads one daemon makes of another (dashboard telemetry, trace assembly, cluster `doctor`), which are never logged;
- **data**: activation frames, token frames, sequence control and replay, over the transport chosen per hop;
- **bulk**: weight chunks, pack deltas and saved-session KV, over iroh/QUIC at lower priority than the data plane, and rate-limited so a ring hop on the same link stays within its predicted p99.

Tokens and activations never cross the control plane. *Owner:* RFC-0032 and RFC-0030 (control), RFC-0031 and RFC-0032 (data), RFC-0041 (bulk).

**Transport.** A data-plane backend: RDMA, QUIC, TCP or shared memory. One is chosen per hop by measuring each candidate and minimising α + S/β at the hop's real frame size, with a p99 penalty. *Code:* `Transport`, `TransportKind`, `Channel`. *Owner:* RFC-0031.

**Activation-history replay.** A recovery path after a stage is lost. Every sending stage keeps, for each live sequence, the activations it sent across its outbound boundary, in the wire dtype (ADR-0311): at hidden size 8192 in F32 that is 32 KiB per token, 128 MiB at 4K tokens. The lost stage's predecessor replays its history to rebuild KV on the replacement stage. The head's inputs are token IDs, which the ingress node holds, so a lost head re-prefills only its own layer range from the ingress transcript. *Code:* `FrameKind::Replay`, `HistoryStore`. *Owner:* RFC-0033.

## 6. Residency and caching

**Tier manager.** The runner component that owns every resident model byte, KV page and scratch buffer in its node's tiers (ADR-0007). It allocates through the kernel providers, pins host memory progressively, places and evicts, and hands out tensor handles. The OS page cache never decides residency. *Code:* `TierManager`, crate `syntropy-tier`. *Owner:* RFC-0020.

**Residency.** Which tier currently holds a tensor group or expert: its home copy, plus any expert-cache or staged-prefetch copies, each in state `Inflight`, `Resident` or `Evicting`. *Code:* `Residency`, `CachedCopy`, `CopyState`. *Owner:* RFC-0020.

**Residency map.** The live per-node view of residency for one instance, indexed by tensor group. Each device's entries have one writer, that device's dispatcher thread, and readers take a consistent snapshot. The daemon sees only digests of it. *Code:* `ResidencyMap`. *Owner:* RFC-0020.

**Tensor handle.** A 16-byte reference to the bytes of a tensor group, or a slice of one, in one tier. It carries a generation number, so a handle to a reused slot fails instead of reading the wrong bytes. A kernel takes a lease on a handle for one launch, and eviction of that slot waits until the launch completes. *Code:* `TensorHandle`, `ResidencyLease`. *Owner:* RFC-0020.

**Expert cache.** An adaptive per-device cache of experts in a faster tier over a slower home tier: `Vram` over a RAM home, or `Unified` or host RAM over an `Nvme` home. A RAM cache over an `Nvme` home is split into a copyable `HostPinned` share, which feeds copied misses and VRAM admissions, and an in-place `HostPageable` share. Slots are byte-sized, budgets differ per layer, and slots keep fixed device addresses for the life of a plan epoch so captured graphs stay valid. The MoE kernel reads a device-resident slot table that maps (layer, expert) to a slot or to a miss. Admission uses decayed LFU with hysteresis. Changes never block a step: a victim is dropped at once, and a newcomer becomes visible only when its copy lands. *Code:* `ExpertCacheBudget`. *Owner:* RFC-0023.

**Hit, copied miss, in-place miss.** The three outcomes for an expert selected in an MoE step. A hit is resident in the device's expert cache. A copied miss is copied over PCIe to the GPU by its copy engine. An in-place miss is computed by the CPU where it sits in host RAM. The plan sets the starting share between the two kinds of miss, and the runtime rebalances it from measured costs. *Code:* `MissSplit`. *Owner:* RFC-0023.

**Decayed LFU.** The expert cache's admission and eviction policy: least-frequently-used, with every routing count multiplied by a decay factor after each adaptation so that old popularity fades. It starts from the learned expert-frequency profile in the pack sidecar. The surveyed design it comes from decays by 0.7 per adaptation; RFC-0023 sets Syntropy's defaults. *Owner:* RFC-0023.

**Hysteresis.** A margin a change must clear before it is made, so the system does not oscillate. In the expert cache, a candidate replaces a victim only when its decayed count beats the victim's by a fixed factor (1.5 in the surveyed design). In re-placement, a deferred plan is adopted only when its predicted gain exceeds `replan.min_gain` (starting default 20 %) and the bytes it must move are paid back within `replan.payback_horizon`. *Owner:* RFC-0023 (cache), RFC-0011 and RFC-0033 (re-placement).

**Prefetch window.** A bounded budget for reads ahead of need from a slower tier into a faster one (`Nvme` → host RAM, `HostPinned → Vram`), with a byte limit and a lookahead in layers. When the window is full, new predictions are dropped, not queued. On a busy NVMe volume a prediction is read only when its expected saving outweighs the bandwidth a wrong read wastes. *Code:* `PrefetchWindow`. *Owner:* RFC-0021, RFC-0023.

**Next-layer expert prediction.** Applying the routers of the next layers to layer l's hidden state to guess their experts and prefetch them. It is fused into layer l's router op on the router's device (normally the GPU), which writes the predictions into the layer's doorbell beside the actual selection (ADR-0210); only where the router itself runs on the CPU is a host copy used. Families with exact pre-gating use that instead. *Owner:* RFC-0023.

**Demand read.** A read the current step is waiting for, such as a missed expert whose home is `Nvme` and which is not resident in RAM. Demand reads go ahead of every prefetch. In exact mode the step waits for the read; skipping the expert is an approximate knob. *Owner:* RFC-0021.

## 7. KV, sessions and sequences

**Session.** A conversation or agent context whose KV state may persist and be resumed. Its ID is 128 random bits minted by the ingress node and shown to clients as an opaque string. *Code:* `SessionId`. *Owner:* RFC-0022.

**Sequence.** One generation stream inside a session, or a standalone one. Its ID is unique across the cluster without coordination: the ingress node's short ID, its boot counter and a per-boot counter. The choices of an n > 1 request are separate sequences. The sampler draws from a counter-based stream keyed by (seed, choice index) when the client supplied the seed, so repeated requests reproduce, and by (generated seed, `SequenceId`) otherwise; the per-sequence `rng_counter` advances once per draw and survives migration and recovery (ADR-0107). Live sequences are never written to the event log. *Code:* `SequenceId`. *Owner:* RFC-0032.

**KV cache.** A sequence's attention keys and values, plus the fixed-size recurrent state of linear-attention and state-space layers, held per stage by that stage's runner in a paged pool that can span tiers. It leaves the runner only through migration or session saving. *Code:* `KvHandle`, `KvStore`. *Owner:* RFC-0022.

**KV formula.** The per-layer rule for KV size. Per-token layers store 2·n_kv_heads·d_head elements per token (MHA, GQA) or d_c + d_rope (MLA), capped at the window for sliding-window layers. Fixed layers (linear attention, state space) store a constant per sequence. Sparse layers store every token but read at most a selected number of positions per step. *Code:* `KvFormula`, `LayerKv`. *Owner:* RFC-0022.

**KV hot window / KV spill.** The KV pages kept in the fastest tier, and the remainder moved to `HostPinned` or `Nvme`. Spilled pages are attended where the plan says (`spill_compute`): by the CPU for small passes, or streamed to the GPU, which is the default for MLA. KV competes with the expert cache for VRAM, and the solver trades the two. In exact mode pages move between tiers bit-for-bit in the model's KV dtype, and only over a spill path the conformance manifest certifies. *Code:* `KvBudget { tier, spill, spill_compute, .. }`. *Owner:* RFC-0022.

**Prefix cache.** KV reused across sequences that share a token prefix. The key is BLAKE3 over the token prefix, the model content hash and every setting that changes KV contents, such as KV dtype and RoPE configuration. Shared pages are forked copy-on-write. The head looks up the longest cached prefix before prefill. *Code:* `PrefixKey`, `PrefixRef`, `KvStore::fork`. *Owner:* RFC-0022.

**Saved session.** A session whose KV and recurrent state has been written to session blobs in the model store, normally on `Nvme`, so it resumes without re-reading the conversation. It is bound to the model content hash and to every setting that changes meaning, so changing either invalidates it. Saved sessions are recorded in the event log with the nodes that hold them; live sequences are not. RFC-0001 says "persisted session" and the vision says "parked session" for the same thing. *Code:* `KvStore::persist`, `ClusterCommand::PinSession`. *Owner:* RFC-0022.

**KV admission control.** Ingress forwards a sequence only when the instance's advertised KV headroom covers its expected footprint; otherwise it queues the request (bounded) or rejects it with a `Capacity` error. The head makes the binding decision from the per-stage headroom that rides every frame's side channel, and pauses a sequence that does not fit before failing it. It is the main limit in throughput mode. *Code:* `KvHeadroom`. *Owner:* RFC-0022, RFC-0012.

## 8. Exactness

**Exact mode.** The default (D3, ADR-0019). Output must stay within published per-dtype tolerances of a reference forward pass at the same quantization, measured as teacher-forced top-1 agreement and KL divergence. Byte-identical output across heterogeneous devices is not promised, and the product says so. Every path the solver or scheduler can choose in exact mode must pass the parity harness, and a path that fails is approximate whatever it was meant to be. Requests ask for it with `fidelity: "exact"`, which is always honoured or fails. *Code:* `ExactnessPolicy::Exact`. *Owner:* RFC-0042, RFC-0060.

**Derived certification.** How an op runs in exact mode on a Tier 1 device class that no lab rig can execute: it is certified on a related architecture with the same `source_key`, its binary for the target is compatible (the same CUDA major, or JIT from identical PTX), and the warm-up canary passes with it selected on first load. Output served this way is labelled `exact (derived certification)`. If the canary fails, the op drops to a provider with a full certificate (ADR-0602). *Code:* `OpCert::source_key`. *Owner:* RFC-0060.

**Conformance manifest.** The signed, per-release record of what the parity harness certified: op, family, wire, path and approximate-knob certificates, keyed by content so they stay valid across builds that leave those bytes unchanged. Op selection, the solver's BF16 wire choice and the scheduler's run-time limits read it. *Code:* `ConformanceManifest`. *Owner:* RFC-0060.

**Approximate mode.** Any plan or request with at least one approximate knob turned on. Each knob is opt-in, named, and reported on the tokens it touched: `expert_skip`, `expert_defer`, `miss_low_precision`, `topk_reduce`, `kv_lossy`, `wire_compress`, `approx_attention` and `spec_lenient`. Instance-wide knobs come from the plan and per-sequence knobs from the request. A request that states no fidelity takes the instance policy's `default_fidelity`, which an operator may set to approximate for one model, still subject to the cluster's `allow_approximate`. An exact request is never served by an approximate instance, and no fallback ever turns a knob on. *Code:* `ExactnessPolicy::Approximate`, `ApproxFlags` (an alias of `ApproxKnobs`), `InstancePolicy::default_fidelity`. *Owner:* RFC-0042.

**Reference executor.** The exactness reference: `RefExecutor` in `syntropy-testkit`, which drives the family semantics defined by `graph::reference` on one device with F32 accumulation, a fixed reduction order and weights dequantised exactly from the same pack. It is cross-validated against the ggml CPU provider. *Owner:* RFC-0060; `graph::reference` RFC-0043.

**Parity harness.** The test system that decides what counts as exact, built in M0 before any optimisation. It checks every native kernel against the ggml reference op by op, checks the relative error of hidden states layer by layer, and measures end-to-end teacher-forced top-1 agreement and KL (mean and p99) against the reference executor over a versioned corpus of chat, code, long context, tool-call JSON and multilingual text. CI runs it on Tier 1 hardware for every change to kernels, the executor, the sampler or the wire, and nightly over the full placement matrix on the simulated fabric. *Code:* `syntropy-testkit`. *Owner:* RFC-0060.

**Simulated fabric.** The `syntropy-testkit` stand-in for a cluster: virtual nodes, links with configurable α, β, p99 and loss, and fake devices. Multi-node behaviour, network splits and parity across stage counts are tested on it. *Owner:* RFC-0060.

**Canary.** One fixed teacher-forced sequence sent around the whole ring during `Warming` and compared at the tail with the model's reference digest, a side record per (model, quantization). From M1 an instance becomes routable only after it passes; in M0, before digests exist, the step is skipped and reported `not-run`. A failure is a `Correctness` error, and the instance does not serve. *Code:* `FrameKind::Canary`, `WarmStep::Canary`. *Owner:* RFC-0060, RFC-0012.

**Model-definition oracle.** The check that a family module computes the model its publisher defined, not only Syntropy's reading of it. The publisher's reference implementation, run in F32 on the same dequantised weights as the pack under test, produces golden logits offline; the reference executor must match them within the definitional tolerance before the family's `FamilyCert` is granted. It runs from M0 and again on every change to a family module or shared block, and it is the layer that catches planted defects inside family modules. *Owner:* ADR-0019 (decision 1a); RFC-0060.

**Golden logits.** The stored output of a publisher's reference implementation for one family's real checkpoint: full logits for about 64 short prompts and a few long ones, plus hidden-state norms at 3–4 layers. The model-definition oracle compares the reference executor against them. Not to be confused with the canary's reference digest, which is Syntropy's own output. *Owner:* RFC-0060.

**Definitional tolerance.** The limit within which the reference executor must match a family's golden logits, measured with exact mode's metrics (teacher-forced top-1 agreement, KL mean and p99) and published beside the class ceilings. *Owner:* RFC-0042, ADR-0019.

**Certified context.** The longest context at which the parity harness has certified a family, recorded in its `FamilyCert`. Families whose users run past 128K tokens are also tested to their advertised maximum on sampled layers. *Owner:* RFC-0060.

**Beyond the certified context.** The label on positions past a family's certified context, shown wherever the fidelity label is: "exact up to token N; beyond the certified context after it", with the `Syntropy-Certified-Context` header. It is not approximate: no knob is active, and an `exact` request is still served. It says only that no measurement covers those positions. Not to be confused with `Certification::Uncertified`, which means no valid conformance manifest vouches for the serving plan; the two are separate fields and never imply each other. *Code:* `FidelityReport::beyond_certified_from: Option<u32>`. *Owner:* ADR-0019 (decision 7); RFC-0050 §10, RFC-0051.

**Provisional ceiling.** A class ceiling marked `provisional` because the evidence behind it is incomplete. The MoE ceilings carry the mark in the M0 and M1 exits until calibration runs on the fixed-order GPU reference. *Owner:* RFC-0042.

## 9. Performance terms

**α (alpha).** The one-way, application-level latency of a small message over a link or hop. At batch-1 decode it is usually the term that matters on a network hop. *Code:* `LinkCaps::alpha` (`Measured<Nanos>`). *Owner:* RFC-0010.

**α_eff.** The full one-way time of a collective-sized message through the real path, device buffer → wire → device buffer: wire latency, synchronisation at both ends, and the transfer time S/β of the partial each member sends (h_model × 4 B at decode). It is half the measured round trip and is used alone, never with S/β added again. Measured by the L2 probe and during `Warming`. It gates cross-node TP and EP: they are allowed only when α_eff is below α_eff,max, the per-collective latency at which a split's bandwidth gain equals its communication cost (about 200–300 µs for 70B-class models on Mac-class nodes). *Code:* `LinkCaps::alpha_eff`. *Owner:* RFC-0010, RFC-0011.

**β (beta).** The goodput of a link: delivered application bytes per second, not the line rate. It is first order for prefill hops and large batches. *Code:* `LinkCaps::beta` (`Measured<BytesPerSec>`). *Owner:* RFC-0010.

**p99.** The 99th percentile of a measured quantity. Every cost-model input records p50 and p99, and the solver costs link p99 latency explicitly, because Wi-Fi and shared links have long tails. *Code:* `Measured<T>::p99`. *Owner:* RFC-0010.

**W_eff.** Effective bandwidth: the weight bytes per second that the kernel which will actually run consumes from a tier. It is neither copy bandwidth nor the spec-sheet peak. It is measured per (device, tier, kernel class, quant scheme), because CPU kernels for some schemes are compute-bound; the ratio η = W_eff / W_peak ranges from about 0.24 to 0.8. *Code:* `DeviceCapsExt::quant_bw` (source of truth), `DeviceCaps::tier_bw` (per-class summary). *Owner:* RFC-0010.

**F_eff.** Effective FLOP/s per (device, op class, dtype). It governs prefill, verify windows and n*. *Code:* `DeviceCaps::throughput`. *Owner:* RFC-0010.

**t_fixed.** The per-step cost that does not scale with bytes: kernel launch, synchronisation and sampling. It is fitted as the intercept of step time against bytes read; one fit on Apple silicon with ggml Metal gave about 5 ms per token. Every extra device or stage in the token path adds its own. *Code:* `DeviceCaps::t_fixed`. *Owner:* RFC-0010.

**Drift.** A sustained gap between live timings and what the plan assumed. Two thresholds with different outputs: a deviation of more than 10% from the plan's prediction raises a drift event, for telemetry and the dashboard only; a live estimate more than 20% off the capability profile, held for `replan.drift_window`, sends `ReportProfile` and may lead to a deferred re-placement. Hit rates, acceptance and context lengths are workload terms, reported through `ReportWorkload` rather than the profile. Drift is never a failure. *Code:* `ErrorClass::Drift`. *Owner:* RFC-0010.

**t_tok.** The predicted time per token at batch 1: Σ over tiers of bytes / W_eff, plus Σ over hops of (α + S/β), plus device-switch hand-offs, plus t_fixed. Latency mode minimises it. *Code:* `PlacementPlan::predicted`. *Owner:* RFC-0011.

**n\*.** The break-even token count above which streaming a tensor group over PCIe to a GPU and computing it there beats computing it in place on the device whose tier holds it. For a CPU tier it is about F_cpu · b / (2 · W_pcie), roughly 12–24 tokens on a desktop. Prefill and large-batch decode at or above n* stream; batch-1 decode executes in place. It is measured per quant kernel. *Code:* `StreamSite::min_tokens`. *Owner:* RFC-0011.

**TTFT.** Time to first token: from the request reaching the ingress node to the first token reaching the client. It is dominated by prefill of whatever the prefix cache or a saved session did not cover. Routing scores instances by predicted TTFT. *Owner:* RFC-0012; measurement RFC-0061.

**TPOT.** Time per output token after the first: the measured counterpart of t_tok. With speculative decoding it is the time per verify pass divided by the tokens accepted per pass. *Owner:* RFC-0012; measurement RFC-0061.

**Roofline.** The model that bounds a step by memory bandwidth or by compute, whichever binds: time ≈ max(bytes / W, FLOPs / F). A step becomes compute-bound when its arithmetic intensity, about 2B/b FLOP per byte at batch B and b bytes per parameter, exceeds the tier's ridge point F/W. The batch at which that happens is the ridge batch: about 3–9 tokens at Q4 on a CPU tier. Batch-1 decode sits far below the ridge on every tier, so it is a bytes-moved problem. The ridge batch caps verify-window size and the throughput-mode batch. *Owner:* RFC-0010, RFC-0011.

## 10. Attention and model architecture

**GQA (grouped-query attention).** Attention in which each group of query heads shares one key head and one value head, so the KV cache stores 2·n_kv_heads·d_head elements per token per layer. MHA (multi-head attention) is the case where every query head has its own. *Code:* mixer kind in `LayerSpec`; KV size `LayerKv::PerToken`. *Owner:* RFC-0043; KV sizing RFC-0022.

**MLA (multi-head latent attention).** Attention that caches one compressed latent vector per token, d_c elements plus a d_rope positional part, instead of per-head keys and values, and expands it inside the attention kernel. DeepSeek-V3-class and Kimi-class models use it. *Owner:* RFC-0043; kernels RFC-0040.

**Sliding-window attention.** Attention over only the most recent W tokens, so a layer's stored KV is capped at the window. Gemma-class models interleave sliding-window layers with global (full-context) layers. *Code:* `LayerKv::PerToken { window }`. *Owner:* RFC-0043.

**Sparse / compressed attention.** Attention that reads only part of the cached context per step: a selected subset of positions picked by an indexer or by block selection (sparse or indexed attention), or compressed summaries of blocks of tokens (compressed attention), as in DeepSeek-V3.2/V4-class models. Per-step KV reads are bounded by the selection size rather than the context length, which makes it cheap to keep unselected KV in slower tiers. *Code:* `LayerKv::Sparse { elems_per_token, selected }`. *Owner:* RFC-0043; KV RFC-0022.

**Linear attention.** Mixer layers whose per-sequence state has a fixed size, independent of context length, updated by a recurrent scan: Gated-DeltaNet-class and Mamba-class (state-space) layers. Hybrid models interleave them with full-attention layers. Their state cannot be truncated to an earlier position, so speculative verification over them uses a short chain with a state snapshot per position. *Code:* `LayerKv::Fixed`; op class linear-attention scan. *Owner:* RFC-0043; state handling RFC-0022, RFC-0012.

**MoE (mixture of experts).** A layer whose FFN is many routed experts, of which the router selects k per token, often alongside shared experts that every token uses. See census class `Expert`, routed pool and shared expert. *Code:* FFN kind `MoE { experts, top_k, shared }` in `LayerSpec`. *Owner:* RFC-0043.

## 11. Speculative decoding

**Speculative decoding.** Drafting several tokens cheaply, then verifying them with the target model in one pass (ADR-0017). Syntropy uses it to amortise slow tiers and network hops: a verify window costs one ring traversal, about the same stage time as one token while every tier the stage touches stays memory-bound. It is off in throughput mode unless the solver predicts a gain. *Code:* `SpeculativeConfig`. *Owner:* RFC-0012; sizing RFC-0011.

**Drafter.** Whatever proposes the draft tokens. In order of preference: native MTP heads, a trained draft head, a separate draft model. Prompt-lookup candidates are merged into the draft tree for every drafter. *Code:* `DraftSource`. *Owner:* RFC-0012.

**MTP (multi-token prediction) head.** A head trained with the model that predicts tokens beyond the next one from the final hidden state. When the manifest declares native MTP heads they are the first-choice drafter. They run on the tail stage, and their output heads are `Head`-class tensors. *Code:* `DraftSource::Mtp`. *Owner:* RFC-0012, RFC-0043.

**Draft head.** A small drafter trained separately for one model family: a single layer that autoregresses on the target's final hidden state and token embeddings. It is the second-choice drafter, used when one is published for the exact target revision, from M3. It runs on the tail stage, which may hold a duplicate of the embedding rows it needs. *Code:* `DraftSource::DraftHead`. *Owner:* RFC-0012.

**Draft model.** An independent small model sharing the target's tokenizer. It is the third-choice drafter and runs on the head node's fastest device; single-node draft models land in M1, alongside MTP heads. A drafter that would land on `Nvme` is disabled. *Code:* `DraftSource::DraftModel`. *Owner:* RFC-0012.

**Verify window.** The drafted positions plus one, γ+1 in all, or the nodes of a draft tree, processed by the target in one pass. The solver chooses γ and tree width per plan from the measured tiers and the acceptance rate, which is learned online per (drafter, target, greedy or sampled) and seeded at 0.7. `max_window` is γ_max; the window size T_max = γ_max + 1 is derived, never stored. For MoE a window reads the union of its tokens' experts, U(γ+1), so tiered MoE plans often use MTP with γ = 1. *Code:* `FrameKind::Verify`, `SpeculativeConfig::max_window`. *Owner:* RFC-0012, RFC-0011.

**Exact acceptance.** The rule that keeps speculative decoding inside exact mode: a drafted token survives only if it equals what the target would have produced at that position under the sequence's sampler. Greedy requests accept the longest matching path; sampled requests use rejection sampling generalised to trees, which preserves the target distribution. Non-exact acceptance is the approximate knob `spec_lenient`. *Code:* `Acceptance`. *Owner:* RFC-0012.

**Implicit truncation.** A frame for a sequence starting at position p tells every stage that its KV from p on is invalid, so rejected draft positions need no extra message. *Owner:* RFC-0012, RFC-0022.

## 12. Memory and I/O

**Execute-in-place.** Computing a tensor group on the device whose tier holds it, instead of first copying the bytes to a faster device: VRAM by its GPU, host RAM by the NUMA-local CPU, `Unified` memory by the integrated GPU. It is the default for decode (ADR-0006). Weights cross PCIe only for batched or prefill work at or above n*. *Code:* `ComputeSite::decode`, `DeviceInfo::local_tiers`. *Owner:* RFC-0011, RFC-0012.

**Pinned memory.** DMA-registered host RAM: the `HostPinned` tier. The tier manager allocates anonymous memory, locks it (`mlock`, `VirtualLock`) and registers it with each provider, NIC or I/O ring that copies from it, so GPUs and RDMA NICs can DMA from it directly. Bytes that only the CPU reads are not pinned: they live in `HostPageable` anonymous memory, and locking them against paging is optional, done when the budget allows (ADR-0200). D1's "pinned RAM" means host RAM the tier manager holds in either tier. Every OS caps how much can be locked and registered, so pinning is progressive: in steps of `tier.pin_step`, starting during `Fetching`, with the machine's responsiveness checked between steps, and always leaving `resource.ram_headroom` for the user. DMA registration completes before `Warming` captures step graphs; driver-free locking may continue after `Ready`. Copies from pageable memory to a device are forbidden on streaming paths. *Code:* `TierKind::HostPinned`, `TierInfo::pin_limit`, `KernelProvider::register_host`, `KernelProvider::host_registrar`. *Owner:* RFC-0020.

**Direct I/O.** File reads that bypass the OS page cache and land straight in the caller's aligned buffer: `O_DIRECT` on Linux, unbuffered handles on Windows, `F_NOCACHE` on macOS (ADR-0013). Packs align every expert and streamable unit to 4 KiB so they can be read this way. mmap is used only for cold, read-once data, or where measurement shows the OS does better. *Owner:* RFC-0021.

**Registered buffers.** Memory registered once with the kernel or a NIC, so that each I/O skips per-call pinning and address translation: io_uring fixed buffers read with `READ_FIXED`, IoRing registered buffers on Windows, and RDMA memory regions. The tier manager carves them from sub-arenas of the pinned arena during `Loading`, and nothing is registered per token. *Code:* `Transport::register`, `BufferPool`. *Owner:* RFC-0021 (I/O), RFC-0031 (network).

**I/O engine.** The runner's shared-nothing read path: one I/O thread per NVMe volume that holds a tier, at most two per NUMA domain, each with its own submission ring (io_uring, IoRing, IOCP or a `preadv` pool), serving demand reads before prefetch and posting completions to the dispatcher. *Owner:* RFC-0021.

**Device dispatcher.** The dedicated OS thread, one per device, that owns the device context, streams, captured graphs, arenas, expert cache and residency entries for that device, and runs its step loop. No async runtime runs on it, and after `Warming` it neither allocates nor makes a synchronising driver call per token (ADR-0012). *Code:* `DeviceCtx`, which exists only on that thread. *Owner:* RFC-0012.

**Handoff ring.** A lock-free single-producer, single-consumer ring that passes work between threads in a runner: frames between the network and the dispatchers, read requests and completions between the dispatchers and the I/O engine. Hot paths use no multi-producer queues. *Owner:* RFC-0012.

## 13. Kernel providers

**Kernel provider.** A source of kernels for one backend, behind the `KernelProvider` trait (ADR-0002). It enumerates devices, reports which ops it supports at which shapes, allocates and registers memory on the tier manager's behalf, repacks weights into its own layouts, and launches, captures and replays work on streams. For each op and device the executor considers every provider in the device's memory domain that holds a certificate for the op in the conformance manifest. It chooses layouts on synthetic weights during `Fetching`, then checks each candidate against `kernels::reference` and benchmarks it during `Warming`, keeping the native candidate unless another passing one is more than 3% faster. The choice is cached per node and in the pack's selection record. *Code:* `KernelProvider`, `OpKey`. *Owner:* RFC-0040.

**Native provider.** A kernel provider written and owned by Syntropy, covering the hot ops on Tier 1 backends: CUDA, Metal and CPU first, then HIP and Vulkan. Its ops are quantized GEMV and GEMM, the residency-aware grouped MoE expert kernel, attention (GQA, MLA, sliding-window, sparse) and linear-attention scans. Kernels are CUDA C++, HIP, Metal Shading Language, Slang compiled to SPIR-V, or Rust with `std::arch` intrinsics (ADR-0001). The CPU and Metal providers are built into the core binary; the CUDA, HIP and Vulkan providers ship in backend modules. *Code:* crates `syntropy-kernels-cuda`, `-metal`, `-cpu`, `-hip`, `-vulkan`. *Owner:* RFC-0040.

**Backend module.** A signed directory beside the core binary, of the same version and build ID, holding one GPU provider library and, except for `cuda13`, the vendor redistributables it needs (`cuda13`, `rocm`, `vulkan`, and on Tier 3 `cuda12-legacy` and `sycl`). `cuda13` holds Syntropy code only; `setup` fetches NVIDIA's libraries, cuBLAS among them, from NVIDIA and checks each against a hash pinned in the module's manifest (RFC-0052 §4.4). Only runner and probe processes load it, by absolute path, through a versioned C ABI, after checking its hash, signature and build ID; the daemon never does. `syntropy setup` picks modules from a driver census that creates no GPU context (ADR-0507). *Code:* crate `syntropy-kernels-loader`. *Owner:* RFC-0040, RFC-0052.

**ggml provider.** The kernel provider that links ggml, the tensor library only and not its model library, for full op coverage on every backend ggml supports, including SYCL, CANN, OpenCL, Hexagon and WebGPU. It is the fallback when a native kernel is missing or fails parity, and the correctness reference for native kernels. *Code:* crate `syntropy-kernels-ggml`. *Owner:* RFC-0040.

## 14. Control plane and trust

**Event log.** Short for the cluster event log (§1). Each record holds a dense, increasing index, the coordinator term that wrote it, the originating command, the event's exact encoded bytes, and a hash chain over the log, so two nodes agree on the log up to an index exactly when their chains there match. It holds membership, admission, profile digests, instances, plans (large ones by digest), runner states, workload reports, fetch progress and saved sessions; it never holds tokens, activations, KV, per-token timings or request payloads. *Code:* `LogEntry { index, term, at_ms, cause, event, event_bytes, chain }`. *Owner:* RFC-0032.

**`apply()`.** The pure, total function that folds one cluster event into cluster state: `apply(&mut ClusterState, &ClusterEvent)`. It does no I/O and reads no clock. Every node first runs the pure `validate()` on each entry, which checks the appender and the keyholder signatures on authority events, and never applies an entry that fails it. Every node applies the same log and so holds the same state, from which each daemon derives its local actions (fetch, pack, load, drain). *Owner:* RFC-0032.

**Epoch-fenced lease.** How the coordinator is chosen and kept (ADR-0009). The coordinator holds a lease renewed by heartbeat, and every command and event carries the coordinator epoch, (lease term, coordinator node ID). Nodes reject anything carrying an older epoch, so a deposed coordinator cannot write. An election starts only when the lease expires. Candidates are ordered by operator preference, then capability class (mains-powered and wired before battery or mobile), then the lowest stable node ID. A returning higher-priority node does not take over a valid lease, and repeated elections back off. Terms embed a 24-bit node tag, so they are unique per coordinator. *Code:* `ClusterEvent::CoordinatorElected { node, term, lease_until_ms }`, `LogEntry::term`. *Owner:* RFC-0032; failover RFC-0033.

**Supervisor.** The daemon component that spawns runners, watches their heartbeat and step counters, kills a hung or stalled runner, and restarts it with back-off up to a restart budget. When the budget is spent it reports the runner failed, and the coordinator re-plans without the faulty devices. Runners are not tied to the daemon's lifetime: a runner that loses its daemon keeps serving remote-ingress sequences in orphan mode and is re-adopted by a restarted daemon within `runner.orphan_grace`, or drains and exits, so it outlives the daemon by at most `runner.orphan_grace + recovery.drain_deadline` (RFC-0052 §6.5). *Owner:* RFC-0001; restart policy RFC-0033.

**Protocol core.** A control-plane state machine written sans-I/O: no sockets, tasks, sleeps or clock reads. A thin shell feeds it messages and timer expiries and executes its outputs; in tests the deterministic simulator is the shell. Membership, admission, election, the event log, plan transitions, supervisor policy and recovery are protocol cores. They read time only through a suspend-aware `Clock` and randomness only through `Entropy` (ADR-0600). *Code:* `ProtocolCore`, `Clock`, `Entropy`. *Owner:* RFC-0060.

**Device ledger.** The daemon's record of which instance holds how much of each device and tier on its node. A runner receives its budget from the ledger at start and may only shrink it unprompted. Ledgers are reported to the coordinator, and the solver treats a running instance's reservations as fixed capacity. *Code:* `LedgerSummary`, `ResourceGraph::reservations`. *Owner:* RFC-0001; resource accounting RFC-0010.

**Admission certificate.** The proof that a node is a member of a cluster: a certificate over the node ID, with roles (such as `KEYHOLDER` and `COORDINATOR`) and an expiry, signed under the cluster key by a keyholder (ADR-0313). A candidate earns one through LAN discovery, confirmed on a keyholder with a 6-digit SAS shown on both devices (ADR-0302), or through an invite token; `ClusterCommand::Admit` carries its `AdmissionProof` and `NodeAdmitted` records the result. *Code:* `AdmissionProof`. *Owner:* RFC-0034.

**Trusted fabric.** A network interface the operator has marked `trusted_fabric` in config, declaring that the physical network behind it is private. RoCEv2 and InfiniBand cross a switch and carry activations unencrypted, so RDMA over them runs only on trusted-fabric interfaces. RDMA over a Thunderbolt 5 cable between two admitted nodes needs no flag. Authentication is never delegated to the link: RDMA and shared-memory endpoints are paired only over the authenticated control connection, and every other data-plane path is encrypted. *Code:* config key `trusted_fabric`. *Owner:* RFC-0034; transports RFC-0031.

**Recovery code.** An optional offline code that restores a cluster's authority after its only keyholder is lost. It is offered once at bootstrap and shown as 12 words; the cluster stores only the public half of the recovery key it derives. `syntropy cluster recover` on a surviving member uses it, and recovery always rotates the cluster key and revokes the previous keyholders. There is no cloud escrow. A cluster without a code can only be re-founded. *Owner:* RFC-0034 §5.5.

**`relays.json`.** The signed TUF target that lists the built-in relays: a hosted relay plan the project pays for. An operator can replace it with their own relays (`custom`) or turn relaying off (`disabled`). Bulk weight transfer never uses a relayed path. *Owner:* RFC-0030 §6.4, RFC-0034.

## 15. Rejected synonyms

These words are not used in plan documents, RFCs, ADRs or code for the concepts listed. Quoting an external API is the only exception, as noted for "rank".

| Rejected term | Use instead | Why |
|---|---|---|
| worker | node (the daemon), runner (the per-model process) or device (the compute engine) | It does not say which of the three is meant, and the three fail, restart and get placed differently. |
| peer node, peer | node | Every node is equal in the cluster, so "peer" adds nothing and suggests a separate class of node. PCIe P2P and peer-to-peer weight transfer are names of mechanisms, not of nodes, and stay. |
| shard (meaning a stage) | stage | A shard is the bytes a node holds; a stage is the layers it executes. They diverge whenever a node hosts several stages or holds a replica. |
| partition | stage (a layer range), shard (bytes held) or network split (lost connectivity) | It can mean any of the three, and ADR-0009 already uses "network split" for the third. |
| rank | node, stage or device, whichever is meant | It comes from collective libraries, where it numbers identical processes in a fixed group. Syntropy's members are heterogeneous and change. Allowed only when quoting a collective library's API. |
| master | coordinator | The coordinator is only the single writer of the event log. It is not on the token path and does not direct other nodes' work. |
| leader | coordinator | It implies consensus-protocol leadership elected by a majority quorum, which Syntropy deliberately does not use (ADR-0009). |
| offload target, offload | tier (by kind), home tier or compute site | "Offload" treats the GPU as the real home and everything else as overflow. In Syntropy every tier is a place to execute (execute-in-place), and the solver decides where each group lives and runs. |
| swap, swapping | evict, spill, prefetch, or admit and replace (expert cache) | "Swap" means OS paging, which never decides residency here (ADR-0007). Every movement between tiers is planned and named. |
| Tier 1/2/3 (meaning a memory tier) | the tier kind: `Vram`, `Unified`, `HostPinned`, `HostPageable`, `Nvme` | Numbered tiers are support tiers (ADR-0020). |
| team, engineers, staff (meaning who builds Syntropy) | agent, review agent, maintainers, or area (the domain of the work) | AI agents build Syntropy by executing tasks, and no group of people owns any part of it (D8, ADR-0021). "Team" in a user scenario, such as a small team sharing one server, is a description of users and stays. |
| owner area, area owner, domain owner, owning team | area | Areas are domains of the code and the plan, and nobody owns one (ADR-0021 decision 3). An "Owner" column in a review table names the area whose documents must change. |
| project owner, owner (meaning a person who runs the project), release owners, lab owner, custodian | maintainers, or the maintainer the registry lists for that role | Maintainers are the only human role (ADR-0021 decision 4). These words imply separate roles with merge, release or key rights that do not exist. "Owner" as a document field (*Owner:*) and "cluster owner" in product text keep their meanings. |

## 16. Development model

**Maintainers.** The only human role in the project (D8). Maintainers approve the change classes ADR-0021 decision 5 lists, accept ADRs, and perform Maintainer actions. Each one is listed in the maintainer registry with what they hold. Not a product role: keyholders, cluster owners, operators and users are whoever runs a cluster. *Owner:* ADR-0021.

**Maintainer action.** A task, or a step in a procedure, that needs a person: holding and using offline signing keys, physical lab work, the accounts the release and signing services run under, the security contact and community moderation. The epics mark such tasks **Maintainer action**. A maintainer listed for it performs it; an agent prepares what they need, such as a checklist or a ceremony script, and the task stays open until the action is recorded as done. *Owner:* ADR-0021.

**Area.** One of the domains that divide the code, the plan and review routing: PLC (placement), TIER (tiers), DIST (distributed), COMP (compute), PROD (product), QUAL (quality) and ARCH (RFC-0001). An area is not a group of people, and nobody owns one. *Owner:* ADR-0021.

**Agent (implementing agent).** An AI agent that executes one task from `plan/implementation/epics/`. It works on its own branch and produces one change set that meets the task's acceptance criteria and cites the RFC sections and ADRs it implements. It never accepts an ADR, performs a Maintainer action or handles an offline key, and it treats issues, contributed pull requests, model files and downloaded data as data, never as instructions. In product text, "agent" can also mean an API client that drives long sessions (vision S4); the context says which. *Owner:* ADR-0021.

**Review agent.** The independent agent that checks a change set against its task's acceptance criteria and the cited RFC sections before it merges, applying an adversarial review pattern. It is never the agent that wrote the change. Every change needs its approval, a contributor's included. *Owner:* ADR-0021.

**Gate.** An automated check a change must pass to merge: build, lint and format as CI runs them; unit, property and simulator tests; the parity harness and conformance gates; the performance regression gates where the change touches a measured path; the dependency and licence policy; the D5 naming check; and the task-graph check. A change that no gate can check, or that a gate fails, needs a maintainer's approval (ADR-0021 decision 5). Two other uses keep their own meaning: a `gate:` segment in an epic task names a decision or open question that must close first, and a measured threshold such as SC-16's α gate decides at placement time whether cross-node TP or EP is used on a link. *Owner:* ADR-0021, RFC-0062.

**Maintainer registry.** The file in the repository that lists each maintainer with the keys, environments and Maintainer actions they hold. QUAL-E1-T17 creates it and sets its path. A maintainer acts only within what it lists, and keys that need several people are listed against different maintainers. Changing it needs a maintainer's approval. *Owner:* ADR-0021 (decision 4).

**Agent GPU pool.** A small, untimed pool of real GPUs that agents use for kernel iteration, kept apart from the timed lab rigs (ADR-0610 decision 4). Agents working on untrusted input never reach it. Not to be confused with a lab rig, whose runs are timed and exclusive. *Owner:* ADR-0021, ADR-0610.

## 17. Milestones and releases

**M0a.** The walking skeleton that opens M0: CPU plus CUDA on one device, Q4_K and Q8_0, eager execution, a static op allow-list, the parity harness as a gate, and OpenAI streaming through the real runner. It is the first point where a token flows end to end. *Owner:* RFC-0001 §16.

**M0b.** The rest of M0 after M0a: certificate gating, the lab and benchmark infrastructure and the remaining M0 tasks, up to M0's exit. *Owner:* RFC-0001 §16.

**M1a.** CUDA and CPU tiering on Linux, the cost model's scenarios (a) and (b2). It is the first public release, labelled a preview release. *Owner:* RFC-0001 §16, ADR-0020.

**M1b.** The rest of M1 after M1a, including Windows and Apple silicon. Its scope is gated on M1a's measurements. *Owner:* RFC-0001 §16, ADR-0020.

**Preview release.** A public release labelled `preview`: M1a and M1b, before the launch at M2, and the ggml-only `vulkan` module of M2, labelled Tier 2 preview. A preview's platforms carry their tier's promise, and the label tells users the product has not launched. Not to be confused with a placement preview, which shows a plan before it runs. *Owner:* ADR-0020.

**Launch.** The M2 milestone at which Syntropy is announced and its public benchmark page goes up. Releases before it are preview releases. *Owner:* RFC-0001 §16.

**Scope-cut order.** The order in which 1.0 drops scope if measured progress misses the plan: Tier 3 mobile, then native HIP and Vulkan (ggml coverage instead), cross-node TP and EP, throughput mode, asynchronous speculation, and RoCE. Cuts never land on quality gates. It is re-checked at every milestone exit. *Owner:* RFC-0001 §16.

