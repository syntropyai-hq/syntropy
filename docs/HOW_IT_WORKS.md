# How Syntropy works

This is the longer explanation behind the [README](../README.md). It describes the design: none of it is built yet, and nothing here is a measurement of Syntropy. Each section links the RFCs and ADRs that specify it in full.

## The idea: compute moves to the bytes

When a model generates one token for one user, it reads every active weight and the whole live KV cache once, and does little arithmetic per byte it reads. A 4-bit decode step does about three floating-point operations per byte, far below the point where a GPU or a CPU becomes compute-bound ([vision §1.1](../plan/00-vision-and-scope.md#11-decode-is-a-bytes-moved-problem)). So the time per token is close to a sum over the places the bytes live:

```
time per token ≈ Σ over tiers (bytes read from the tier ÷ bandwidth of the processor reading it)
               + Σ over network hops (latency + bytes sent ÷ link bandwidth)
               + fixed per-token cost (launches, synchronisation)
```

The rungs of that ladder are far apart on consumer hardware. These are published or independently measured hardware figures from the plan's research notes, not Syntropy results:

| Where the bytes sit | Read by | Bandwidth |
|---|---|---|
| GDDR6X or GDDR7 VRAM | the GPU | 936–1792 GB/s peak, RTX 3090 to RTX 5090 |
| Apple unified memory | the GPU | 120–819 GB/s, M4 to M3 Ultra |
| Dual-channel DDR5 | the CPU | 76–122 GB/s measured |
| PCIe 4.0 x16, pinned host memory to the GPU | the copy engine | about 24 GB/s measured |
| NVMe Gen4 or Gen5 | the host | 7.45 or 14.7 GB/s sequential, per spec |
| 2.5 GbE | the network | 2.35 Gb/s of goodput measured |

Two consequences shape everything else. First, the job is to decide which rung each byte lands on, because that sets the speed. Second, a weight should be computed by the processor that can read it fastest where it already sits. A desktop CPU reads dual-channel DDR5 at 76–122 GB/s, about 3–5× the ≈24 GB/s that PCIe 4.0 x16 carries to the GPU (PCIe 5.0 is expected at 48–52 GB/s [estimate], still below DRAM), so for a single stream it is faster to let the CPU compute a RAM-resident weight than to ship it to the GPU for every token.

## Inside one machine: the tiers

<p align="center"><img src="media/tiers.svg" width="760" alt="One machine: VRAM holds the hot weights and an expert cache and is read by the GPU; RAM holds every weight and the CPU computes cache misses there; NVMe holds cold experts and tables and is prefetched into RAM"></p>

Each model instance on a machine runs in a runner process with one **tier manager**, which owns every byte the runner keeps resident: weights, expert-cache slots, KV pages, staging buffers and scratch ([RFC-0020](../plan/rfcs/RFC-0020-tier-manager-and-residency.md), [ADR-0007](../plan/decisions/ADR-0007-syntropy-owns-residency.md)). The operating system's page cache never decides what is hot. The tiers are:

- **VRAM** (`Vram`), the GPU's own memory. It holds the bytes every token touches: attention, dense layers, shared experts, the router, the LM head, and the hot window of the KV cache. The rest of the VRAM budget is an expert cache.
- **Unified memory** (`Unified`) on Apple silicon, Strix Halo and GB10-class machines, where the GPU and CPU share one pool and there is no PCIe term at all.
- **RAM**, split into `HostPinned`, registered for DMA, and `HostPageable`, locked but not registered. Only bytes that a GPU or a network card actually copies are registered, because registration is scarce and slow, and the bytes the CPU computes in place need none of it ([ADR-0200](../plan/decisions/ADR-0200-host-ram-dma-registration-only-for-dma-sources.md)). Pinning is progressive, so it never takes free memory below a reserve kept for the rest of the system.
- **NVMe** (`Nvme`). It holds a per-device **pack**: every expert and large table laid out in 4 KiB-aligned extents, one per expert, read with direct I/O (io_uring on Linux, IoRing on Windows, `F_NOCACHE` reads on macOS) so that a read moves exactly the bytes needed ([RFC-0021](../plan/rfcs/RFC-0021-weight-io-path.md), [RFC-0041](../plan/rfcs/RFC-0041-model-formats-packs-weight-distribution.md)). NVMe is never a compute site; its bytes are read into RAM or VRAM first.

**Execute-in-place.** Every tensor group has a home tier and a compute site, and the compute site is the device whose tier holds it: VRAM is computed by its GPU, RAM by the CPU on the same NUMA node ([ADR-0006](../plan/decisions/ADR-0006-parallelism.md)). The GPU and the CPU work on different parts of the same step at the same time, and hand activations to each other, not weights.

**When streaming wins.** Copying weights to the GPU pays off only when one pass uses them for many tokens, as a long prompt chunk or a large batch does. The break-even token count, `n* = F_cpu · b / (2 · W_pcie)`, comes from the machine's measured CPU throughput `F_cpu`, bytes per parameter `b` and PCIe rate `W_pcie`, never from a constant. Below `n*` RAM-resident weights are computed in place; above it, the groups the plan marks for streaming are copied in micro-batches while the GPU works.

**Giving memory back.** The machine still belongs to its user. When the OS or another program needs RAM or VRAM, the runner gives it back along a fixed ladder of steps, and the cost of each step in token time is predicted before it is taken ([ADR-0202](../plan/decisions/ADR-0202-memory-yield-ladder-with-predicted-cost.md)).

**KV and sessions.** The KV cache is paged across VRAM, RAM and NVMe, and its bandwidth is part of the placement objective, because at long context it can rival the weights in bytes read per token. A prefix cache means a repeated prefix (a system prompt, tool schemas, files an agent keeps sending) is never prefilled twice. A parked session goes to NVMe, bound to a hash of the model files and every setting that changes meaning, so it is restored rather than recomputed ([RFC-0022](../plan/rfcs/RFC-0022-kv-cache-prefix-cache-and-sessions.md)).

## The MoE expert cache and prefetch

A mixture-of-experts model has hundreds of experts per layer, and each token uses a handful of them. On a machine where the model is larger than VRAM, each routed expert lives in RAM or on NVMe, and the design makes those reads cheap ([RFC-0023](../plan/rfcs/RFC-0023-moe-expert-cache-prediction-prefetch.md)).

- **The cache.** Each GPU keeps per-layer expert slots in its fastest memory. Admission uses a decayed least-frequently-used count with hysteresis, and a shadow least-recently-used policy runs alongside as a guard, so the hit rate stays at or above what LRU would give. The learned frequency profile is saved across restarts, so a restarted runner starts warm ([ADR-0209](../plan/decisions/ADR-0209-expert-cache-policy.md)). Replacements never stall a step, and the cache changes without invalidating captured GPU graphs, because the slot table lives on the device and is updated through a delta ring.
- **A cache miss is computed where it sits.** In one MoE layer, cache hits run on the GPU, some RAM-resident misses are copied over PCIe for the GPU, and the rest are computed by the CPU in place, all at the same time and with no host synchronisation. The share that is copied is chosen per layer, each step, by an online fit of the CPU, PCIe and GPU costs, including the slowdown that DMA traffic causes the CPU ([ADR-0211](../plan/decisions/ADR-0211-miss-split-online-balancer.md)). Both sites compute the exact expert.
- **Prediction.** The guess at the next layer's experts is fused into the router kernel, on the device that runs the router, so it costs no extra launch. Where a model family knows its routing ahead of time, the exact routing is used instead ([ADR-0210](../plan/decisions/ADR-0210-expert-prediction-on-router-device.md)).
- **Prefetch at every slow boundary.** The predictions start NVMe-to-RAM and RAM-to-VRAM transfers before the router asks for those experts. From M3, the same applies to an expert held in another machine's RAM. A prefetch is issued only when its expected value beats its cost on a busy drive, because wasted reads compete with demand reads.
- **One home per expert.** Every expert has exactly one home in the cluster and inside each machine. A hot expert is copied elsewhere only as an explicit entry in the plan, never by accident.
- **Telemetry for the planner.** The cache reports per-layer hit-rate curves, which the planner uses to decide how many expert bytes each tier should hold ([ADR-0102](../plan/decisions/ADR-0102-expert-budgets-from-learned-hit-curves.md)).

## The joint placement

All of these choices are made together, by one planner, in one plan ([RFC-0011](../plan/rfcs/RFC-0011-joint-placement-solver.md), [ADR-0008](../plan/decisions/ADR-0008-placement-solver.md)).

**Inputs.** The first is the cluster's resource graph: each machine's devices, tiers and links, with the bandwidths and latencies its probes measured using the real kernels, the real I/O path and the real network path ([RFC-0010](../plan/rfcs/RFC-0010-resource-graph-and-capability-probing.md)). Spec sheets are only sanity bounds. The second is the model's byte census: every tensor classed as dense-every-token, routed expert, row lookup, head or router, with its dtype and size.

**What it decides:**
- which machines and devices take part, and in what order around the ring;
- each machine's contiguous range of layers, its stage;
- the home tier and compute site of every dense tensor group, including whether it streams above `n*`;
- for every MoE layer, how its expert bytes divide between VRAM, RAM and NVMe, read off the learned hit-rate curve;
- the KV budget per tier per stage, with scratch and reserves;
- the parallelism mode of each stage, and the speculative window.

**Objective.** By default, the shortest decode critical path for one stream, with a cap on time to first token. A throughput objective, for a household sharing one cluster, minimises the slowest stage instead, with a per-user floor. Budgets are checked per machine: a plan that fits only in the cluster's total memory doesn't fit.

**Method.** From M1 the production planner is a dynamic program over contiguous layer ranges, with a greedy cost inside each machine that fills the fastest memory first. The full mixed-integer formulation over devices, tiers, links and experts, solved with HiGHS, is specified as the reference for every term. It will be built only if a simulation study shows it beating the dynamic program by more than 5% on reference inputs. Either way, a plan is deterministic: the same inputs give the same plan on every machine.

**Explanation and drift.** Every plan carries a prediction for each cost term and an explanation of why each tensor group is where it is, which the CLI and dashboard show before the model loads. Running stages record the same cost terms as they work, so predicted and measured time sit side by side. A sustained gap raises a drift event, and a large one triggers a new probe and possibly a new plan. A committed plan never changes in place; every change, even a budget tweak, is a new plan epoch.

## Across machines

<p align="center"><img src="media/ring.svg" width="760" alt="Three machines in a ring, each running a layer range: hidden states pass from one to the next, the last samples the token and returns it to the first and streams it to the machine that received the request"></p>

**The ring.** Across machines the model is a pipeline, cut by layer range at the boundaries each model family declares. The **head** stage embeds the input and starts each step. Each stage runs its layers and sends its hidden state directly to the next stage. The **tail** stage samples the token, sends it back to the head to start the next step, and streams it point to point to the **ingress** machine, the one holding the client's connection ([RFC-0032](../plan/rfcs/RFC-0032-distributed-execution.md), [ADR-0010](../plan/decisions/ADR-0010-data-plane-topology.md)). Only activations cross the network during decode, never weights. Prefill is chunked and staggered across stages so they work in parallel on a long prompt.

**What a pipeline buys.** For one stream, the stages run one after another, so a pipeline never makes the same bytes faster. It adds capacity, and it lets the planner move bytes from slow memory, such as a PC's DRAM, onto faster memory elsewhere, such as a Mac's unified memory. Every stage adds a hop and a fixed cost, so a device joins the ring only if it lowers the total, and a weak device is left out. When the model fits on one machine, the chosen plan is never slower than the best single-machine plan [target, SC-14].

**When a layer is split across machines.** Tensor or expert parallelism across machines needs a collective per layer per token, and each one pays the link's full latency. The planner allows it only between machines whose measured collective latency, timed device buffer to wire to device buffer over many samples, falls below the break-even point for that model; machines that pass pairwise form a fast island that runs as one virtual device. In practice that admits RDMA-class links such as Thunderbolt 5 between Macs, RoCE, or fast low-jitter Ethernet. Wi-Fi never qualifies. Partial sums are reduced in a fixed member order in F32, so every member holds identical hidden states ([ADR-0006](../plan/decisions/ADR-0006-parallelism.md)). Inside one machine with several GPUs, tensor parallelism over NVLink or PCIe peer-to-peer is used when the cost model says it pays.

**Transports.** Each link gets the transport its measurements favour: RDMA where present, QUIC on the LAN, shared memory between processes on one host, and TCP as the fallback ([ADR-0011](../plan/decisions/ADR-0011-transport-stack.md), [RFC-0031](../plan/rfcs/RFC-0031-transport-and-wire-protocol.md)). Every frame carries its plan epoch, and a runner rejects frames for a plan it isn't serving.

**The control plane.** Cluster state is an event log written by one elected **coordinator** and validated and applied by every machine. The coordinator never sees a token and isn't on the request path: the ingress machine routes requests from its own replica of cluster state, so a coordinator change fails no request in flight. Election needs no majority quorum, because a two-machine cluster has to survive losing either one ([ADR-0009](../plan/decisions/ADR-0009-control-plane.md)).

**Losing a machine.** Every sending stage keeps a short history of the activations it sent. When a stage machine is lost and the survivors have capacity, the planner re-places the lost layers, the new stage replays that history, and the sequence resumes without a client-visible error. A lost head re-prefills only its own layers from the ingress machine's transcript ([RFC-0033](../plan/rfcs/RFC-0033-fault-tolerance-and-elasticity.md), [ADR-0311](../plan/decisions/ADR-0311-sender-side-activation-history.md)).

**Weights.** Each machine fetches only the byte ranges of the model that its stage needs, from another cluster machine that already holds them where possible, in BLAKE3-verified, resumable chunks ([ADR-0016](../plan/decisions/ADR-0016-weight-distribution.md)).

## Speculative decoding with exact acceptance

Slow tiers and network hops cost the same per pass whether the pass checks one token or several. Speculative decoding uses that: a cheap drafter proposes a short tree of likely next tokens, and the model verifies the whole tree in one pass ([ADR-0017](../plan/decisions/ADR-0017-speculative-decoding.md), [RFC-0012](../plan/rfcs/RFC-0012-execution-scheduler.md)).

- **Drafters, in order of preference:** the model's own MTP heads where its manifest declares them; a trained draft head published for that exact model revision; or a separate draft model with the same tokenizer. Candidates from earlier text in the prompt are merged into the tree at no weight cost.
- **Placement.** MTP heads and draft heads read the model's final hidden state, so they run on the tail stage beside the sampler. A separate draft model runs on the head machine's fastest device. A drafter that would land on NVMe is turned off.
- **One trip round the ring per window.** The tail verifies, samples, drafts the next window and sends the accepted tokens and the new tree to the head. Each stage processes the whole tree in one pass.
- **Exact acceptance.** A greedy request accepts the longest path in the tree that matches the model's own top choice at each position. A sampled request uses rejection sampling generalised to trees, which preserves the model's output distribution. Speculative and plain decoding must still agree within the exactness tolerance, and the parity harness tests both.
- **Sized by the planner, off when it doesn't pay.** The planner picks the window and tree width that minimise expected time per token, from the cost of a verify pass on each tier and the acceptance rate measured online. Speculation switches itself off when the prediction says it won't help.

Single-machine speculation with MTP heads and draft models is planned for M1. Speculation across stages, loading trained draft heads that a model ships with, wide trees for I/O-bound models and asynchronous speculation follow in M3.

## Discovery, pairing and invites

Machines find each other without configuration ([RFC-0030](../plan/rfcs/RFC-0030-discovery-membership-invites.md), [RFC-0034](../plan/rfcs/RFC-0034-security-and-trust.md)).

- **Identity.** Each machine's daemon is identified by a persistent Ed25519 key, which is also its QUIC endpoint identity.
- **Discovery.** On a LAN, three mechanisms run at once: DNS-SD over mDNS, signed beacons on an IPv6 link-local multicast group, and signed IPv4 broadcast beacons. A beacon only prompts a connection attempt, which is then authenticated. Nodes on an incompatible protocol version are shown to the user rather than ignored.
- **Pairing.** A new machine joins after a person approves it on a machine holding the cluster key, comparing a six-digit code shown on both screens. A machine that was admitted before rejoins on its own, with no interaction.
- **Invites.** A machine on another network joins with an invite token of about 115 characters. The token names the inviting machine's key, so the joiner authenticates the cluster even through a relay, and it is single-use by default, expires, and carries scopes. The connection goes over hole-punched QUIC, a relay or a Tailscale address, with no port forwarding.
- **Membership.** Members exchange heartbeats over QUIC datagrams. A silent member is probed indirectly, then suspected, then declared failed. Clocks are suspend-aware, so a laptop waking from sleep isn't declared dead, and doesn't declare everyone else dead.
- **A laptop away from home** usually doesn't host a stage, because a WAN round trip in every token's path would cap the cluster's speed. It becomes the ingress machine: it sends the request, and the tail stage at home streams tokens to it. Back on the LAN, it becomes eligible for a stage at the next re-placement.

## Exactness

With default settings, output must match a reference forward pass at the same quantization, within published tolerances ([ADR-0019](../plan/decisions/ADR-0019-exactness-contract.md), [RFC-0042](../plan/rfcs/RFC-0042-quantization-and-numerical-fidelity.md), [RFC-0060](../plan/rfcs/RFC-0060-testing-and-correctness.md)).

- **Two links in the chain.** Syntropy's reference executor runs on one device with F32 accumulation, a fixed reduction order, and weights dequantised exactly from the same pack. It is checked against golden logits produced offline by the model publisher's own implementation, so a mistake in Syntropy's reading of a model is caught as well as a mistake in a kernel. Then every candidate path is checked against the reference executor.
- **Same quantization.** A 4-bit pack is judged against the 4-bit reference, not against the BF16 original. Quantization error belongs to the model file you chose and is reported separately.
- **The metrics.** Teacher-forced top-1 agreement and KL divergence (mean and p99), over a versioned corpus of chat, code, long context, tool-call JSON and multilingual text. A candidate passes when it is within twice the spread measured between the reference executor and the ggml CPU provider on the same pack, and within absolute ceilings per dtype class. The tolerances are versioned and stamped into every release.
- **Everything the planner can pick is tested:** every kernel, provider and fallback; every placement shape; every transport precision; speculative and plain decoding; prefix-cache reuse, KV paging and session resume. A path that fails is approximate by measurement, whatever it was meant to be.
- **On the wire and in storage**, exact mode sends activations in F32 (BF16 only for a family after the harness passes it), moves KV between tiers bit for bit, and never lowers stored KV precision.
- **Shortcuts are named knobs:** `expert_skip`, `expert_defer`, `miss_low_precision`, `topk_reduce`, `kv_lossy`, `wire_compress`, `approx_attention` and `spec_lenient`. Each is opt-in per model or per request, and a knob is added only with a harness report of its measured cost. Every response carries its fidelity and the list of active knobs, and the CLI and dashboard show them.
- **Not byte-identical.** Different devices round differently within the tolerance, so identical bytes across devices, placements or speculative settings are not promised. A slower `deterministic` mode, planned for M3, gives identical tokens for a fixed seed on the same build, plan and devices.
- **Built first.** The parity harness and the model-definition check are built in M0, before any optimisation, and CI runs them on Tier 1 hardware for every change to kernels, the executor, the sampler or the wire.

## Where to read more

| Topic | Documents |
|---|---|
| The whole architecture: processes, crates, core types, lifecycles, milestones | [RFC-0001](../plan/rfcs/RFC-0001-system-architecture.md) |
| Probes and the resource graph | [RFC-0010](../plan/rfcs/RFC-0010-resource-graph-and-capability-probing.md) |
| Placement | [RFC-0011](../plan/rfcs/RFC-0011-joint-placement-solver.md), [ADR-0008](../plan/decisions/ADR-0008-placement-solver.md) |
| Step scheduling and speculation | [RFC-0012](../plan/rfcs/RFC-0012-execution-scheduler.md), [ADR-0017](../plan/decisions/ADR-0017-speculative-decoding.md) |
| Tiers, I/O, KV and the expert cache | [RFC-0020](../plan/rfcs/RFC-0020-tier-manager-and-residency.md), [RFC-0021](../plan/rfcs/RFC-0021-weight-io-path.md), [RFC-0022](../plan/rfcs/RFC-0022-kv-cache-prefix-cache-and-sessions.md), [RFC-0023](../plan/rfcs/RFC-0023-moe-expert-cache-prediction-prefetch.md) |
| Discovery, transport, distributed execution, recovery, security | [RFC-0030](../plan/rfcs/RFC-0030-discovery-membership-invites.md) to [RFC-0034](../plan/rfcs/RFC-0034-security-and-trust.md) |
| Kernels, model formats, numerics, model graphs | [RFC-0040](../plan/rfcs/RFC-0040-kernel-providers-and-backends.md) to [RFC-0043](../plan/rfcs/RFC-0043-model-graph-ir-and-architecture-families.md) |
| The cost model behind every figure | [research/hardware-and-performance-model.md](../plan/research/hardware-and-performance-model.md) |
