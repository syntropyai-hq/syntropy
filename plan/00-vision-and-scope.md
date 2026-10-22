# Syntropy: vision and scope

- **Status:** Accepted (2026-10-20)
- **Date:** 2026-10-19
- **Builds on:** [ADR-0000](decisions/ADR-0000-product-directives.md) (product directives D1–D8) and the foundation ADRs 0001–0022
- **Evidence:** the four notes in [research/](research/). Section references such as "cost model §3.3" point to [research/hardware-and-performance-model.md](research/hardware-and-performance-model.md).
- **Companions:** [glossary.md](glossary.md) defines every term used here; [open-questions.md](open-questions.md) tracks what is not yet known.

Numbers in this document carry one of three labels. **[measured]** figures come from the research notes, with the source tag they carry there. **[estimate]** figures are arithmetic on measured inputs, with the formula shown. **[target]** figures are commitments Syntropy has to meet, each derived from a formula in the cost model or stated as a chosen budget. Nothing here is a Syntropy benchmark result: none exist yet.

## 1. The problem

### 1.1 Decode is a bytes-moved problem

Generating one token at batch 1 reads every active weight byte and the whole live KV cache once. The cost model (§0, §3.2, §3.7) puts the time for that token at

```
t_tok ≈ Σ_tiers B_i / W_i(exec)  +  Σ_hops (α_h + S_h / β_h)  +  t_fixed
```

`B_i` is the bytes read from tier *i* (weights plus KV), `W_i(exec)` the effective bandwidth of whichever processor executes against that tier, `α_h` and `β_h` the latency and goodput of each network hop, and `t_fixed` the per-token cost that does not scale with bytes. Layers run in order, so the tier terms add; they do not overlap into a max.

Compute hardly enters. A Q4 decode step performs about `2/b ≈ 3.3` FLOPs per byte read (b = 0.6 bytes per parameter), and the ridge point where a device turns compute-bound is ≈164 FLOP/byte on an RTX 4090 and ≈11–29 on a desktop CPU [estimate, cost model §3.1]. Single-stream decode is memory-bound on every device Syntropy targets. Two things set the ceiling: **capacity**, meaning which tiers the bytes can live in, and **bandwidth**, meaning how fast the device that executes them can read them where they live.

### 1.2 The bandwidth ladder on consumer hardware

| Where the bytes sit | Read by | Figure | Label (cost model §) |
|---|---|---|---|
| GDDR6X / GDDR7 VRAM | discrete GPU | 936–1792 GB/s peak (RTX 3090 to RTX 5090) | [measured: spec, §2.1] |
| Apple unified memory | Apple GPU | 120–819 GB/s (M4 to M3 Ultra); 1.2 TB/s on M5 Ultra | [measured: spec, §2.1] |
| APU unified memory (Strix Halo, GB10) | iGPU | 256–273 GB/s peak; Strix Halo CPU side reads only 124 GB/s | [measured, §2.1] |
| Dual-channel DDR5 | CPU | 76–122 GB/s measured; a single-CCD Ryzen stops at ≈64–70 GB/s whatever DIMMs are fitted | [measured, §2.1] |
| PCIe 4.0 x16, pinned host → GPU | copy engine | ≈24 GB/s; PCIe 5.0 not measured, 48–52 GB/s expected | [measured, §2.3]; [estimate] |
| NVMe Gen4 / Gen5 | host | 7.45 / 14.7 GB/s sequential spec; sustained rate under heat not published | [measured: spec, §2.4] |
| RDMA over Thunderbolt 5 | network | plan on 2.5–4 GB/s per link and 10–25 µs per small message | [measured, §2.5] |
| 2.5 GbE | network | 2.35 Gb/s goodput; α ≈ 100–250 µs per application-level hop | [measured, §2.5] |
| Wi-Fi 6/6E | network | 320–610 Mb/s; 3–7 ms median hop, ≈250 ms at p99 | [measured, §2.5] |

From the top rung to NVMe is two orders of magnitude, and the network sits another order below that. Every placement decision is a decision about which rung each byte lands on.

### 1.3 What the formula says about real machines

These come from the cost model's worked examples (§4). All are [estimate], built from measured inputs and stated efficiency factors.

- **70B dense on one gaming PC.** Llama-3.3-70B at Q4_K_M is 42.5 GB. On an RTX 4090 with 64 GB of DDR5-6000, 19.3 GB fits in VRAM once KV and scratch are reserved. The other 23.2 GB is read by the CPU from DRAM at ≈70 GB/s, and that half takes 93% of every token: ≈2.8 tok/s. Copying those 23.2 GB over PCIe each token instead would give 1.05 tok/s. At 100% of both peak bandwidths the ceiling is 3.8 tok/s.
- **Frontier MoE on the same PC.** DeepSeek-V3 (671B total, 36.5B active) at Q4_K_M is 404 GB. With 128 GB of RAM, the always-touched 9.7 GB fits in VRAM, about 115 GB of the 394 GB routed-expert pool fits in RAM, and the rest has to come from NVMe. Decode runs from ≈0.7 tok/s with no routing locality to ≈5.3 tok/s with every expert in RAM. The 5.3 is a hard cap: 12.3 GB of routed bytes per token over dual-channel DRAM at 70 GB/s.
- **One large Mac.** An M3 Ultra with 512 GB holds DeepSeek-V3 at 4-bit. Its bound at peak bandwidth is ≈40 tok/s; a community run of an existing engine measured ≈21 tok/s [measured, community].
- **A mixed home cluster.** A gaming PC, a MacBook Pro M4 Pro and a Strix Halo mini PC on 2.5 GbE run Llama-70B as a pipeline at ≈6.5 tok/s, against 2.8 tok/s on the PC alone. They run Qwen3-235B-A22B at ≈12–16 tok/s, because its 13 GB per token is a third of the dense model's.

So "the largest model at the highest tokens/s the hardware can physically deliver" (D7) has a precise meaning. The ceiling is `1 / Σ_i (B_i / W_i)`. The work is to choose each `B_i` so that the sum is as small as capacity allows, and then to run within a few percent of the bound that the measured bandwidths allow (SC-3).

### 1.4 Where existing approaches lose that speed

The survey (`research/multi-device-inference.md` §0 and §14, `research/tiered-memory-inference.md` §18) found these losses repeatedly. Per D5 they are described by technique, not by system.

- **Placement sized on free system RAM.** No VRAM query, no KV or activation reserve, no measured bandwidth, compute or link figures. In one published study a per-device ILP that modelled these ran a 70B model 3.3× faster than a memory-proportional split [reported by a surveyed system].
- **Residency left to the OS page cache.** File-backed pages are evicted first. One engine's CPU read a mapped hot tier at ≈19 GB/s against 42.8 GB/s from a pinned anonymous arena on the same machine [reported by a surveyed system].
- **Weights copied over PCIe during decode,** when the CPU could have computed them in place over DRAM at 3–5× the bandwidth.
- **Tensor parallelism on Ethernet or Wi-Fi.** A 60–80-layer model needs 120–160 collectives per token, and each one pays the full link latency.
- **Tokens routed through a central control process,** JSON-encoded and broadcast to every node. Under load this has collapsed clusters.
- **No recovery** when a stage dies mid-generation on a home cluster: the request aborts or stalls.
- **A large fixed per-token cost.** One fitted Metal engine spends ≈5 ms per token outside its byte reads. A 7B Q4 model on M3 Ultra reads its 3.79 GB in ≈5.9 ms, so that engine spends almost as long on overhead as on the model [estimate, §2.2].
- **Speed from hard-coded geometry.** A fast single-GPU MoE engine bakes one model's shapes into its kernels and refuses every other model.

No surveyed system models VRAM, RAM, NVMe and network links together for each device and also decides which devices to use. That is the space Syntropy is built for.

## 2. What Syntropy is

Syntropy is a local-first inference system that treats every byte of a model as something to place. Each host runs one daemon, `syntropy serve`. On startup the daemon measures its machine: the bandwidth each device actually achieves over each memory tier with the real dequantising kernels, the pinned PCIe rate, the sustained NVMe read rate at expert-sized requests once the drive is warm, and the latency, bandwidth and p99 of every link to every other node. Those measurements go into one mixed-integer program, together with a byte census of the model in which every tensor is classed as dense-every-token, routed expert, row lookup, head or router. The program decides which nodes take part, which contiguous layer range each node runs, which tier holds each tensor group, and which device computes it.

The plan keeps compute next to the bytes. Inside a machine, the always-touched tensors and the hottest experts sit in VRAM. The full expert set sits in host RAM, where the CPU computes VRAM-cache misses in place; only the bytes a GPU copies from RAM are registered for DMA. Cold experts and large lookup tables sit on NVMe, read with direct I/O in whole-expert units and prefetched from next-layer routing predictions. Across machines, the model is split into pipeline stages by layer range. Stages pass hidden states directly to one another in a ring, so only activations cross the network during decode, never weights. Tensor and expert parallelism across nodes are used only where a link's measured per-collective latency is low enough for them to pay, which in practice means RDMA. Speculative decoding, using native MTP heads first, spreads the cost of slow tiers and network hops over a verify window.

By default, output stays within published tolerances of a reference forward pass at the same quantization, and that reference is itself checked against the model publisher's own implementation (ADR-0019). Anything that trades accuracy for speed is opt-in and labelled. Syntropy ships as one binary for Linux, macOS and Windows. It finds other nodes on the LAN with no configuration, and admits a device on another network through an invite token. Each node fetches only the weight ranges it needs, from a cluster node that already holds them where possible. Whichever node a request arrives at serves it through OpenAI- and Anthropic-shaped APIs.

## 3. Goals and non-goals

"v1" in this plan means the 1.0 release at the end of M4. The feature set is complete at M3, and M4 hardens it (§8). Users get it in stages. The first public release is M1a, a Linux preview with CUDA and CPU tiering. M1b adds Windows and Apple silicon, and the launch, with its announcement and public benchmark page, is at M2 (RFC-0001 §16, ADR-0020).

### 3.1 v1 goals

1. **Capacity.** Run the largest model that the cluster's combined VRAM, RAM and NVMe can hold, as long as the cost model predicts a usable rate for the mode the user picked.
2. **Speed near the physical bound.** Decode within the margins of §7 of `1 / Σ B_i / W_i`, with each `W_i` measured on the user's own hardware.
3. **One stream first** (D2). Batch-1 decode on the critical path is the default objective. A throughput objective for a few concurrent users can be selected without a redesign.
4. **Exact by default** (D3, ADR-0019). "Exact" is defined against the publisher's reference implementation of each model, run in F32 on the same dequantised weights: golden logits and hidden-state norms from it gate every family's certification. The label covers positions up to the longest context the parity harness has certified for the family. Positions beyond that are labelled "beyond the certified context", which is not approximate but says that no measurement covers them. Every approximate setting is opt-in and visible wherever it is in effect.
5. **One machine is a first-class case.** A single PC or Mac with VRAM, RAM and NVMe tiers is milestone M1 (M1a on Linux, M1b on Windows and macOS), not a degenerate cluster.
6. **Mixed clusters.** NVIDIA, Apple, AMD, Intel and CPU-only nodes, on Linux, macOS and Windows, can join one cluster.
7. **Zero configuration on a LAN, and invite links off it** (D4).
8. **Survive node loss.** A sequence resumes when a stage node is lost and the survivors have capacity, and a coordinator change never fails an in-flight request.
9. **Long sessions.** Agents keep KV across turns and restarts, and a repeated prefix is never prefilled twice.
10. **Model-generic.** Model families are graph modules over shared blocks, and no model's geometry is hard-coded anywhere in the runtime.
11. **One command to install,** and no configuration file before the first token.

### 3.2 After v1

- NPUs as devices. Which NPU stacks, if any, is open (OQ-K6 in [open-questions.md](open-questions.md)).
- MLX as an optional kernel provider, if it beats Syntropy's Metal decode by 15% or more after M2 (ADR-0004).
- Phones and WebGPU nodes beyond Tier 3 small slices.
- KV pooled in other nodes' RAM and NVMe.
- Prefill/decode roles assigned automatically from the cost model.
- Learned expert predictors beyond the next-layer router, and asynchronous pipelined or wide-tree speculation.
- Image and other multimodal input, through the hooks §3.3 reserves.
- Languages other than English in the CLI and dashboard.

### 3.3 Non-goals

- **Public swarms of untrusted machines** (D4). Every node in a cluster is admitted under the cluster key and is trusted with the weights and activations it handles.
- **Training of any kind,** including fine-tuning. v1 produces no draft heads for speculative decoding (ADR-0017); the runtime loads one only where a model ships it, and never trains one.
- **Datacenter serving.** That means hundreds of concurrent users, autoscaling, multi-tenant isolation and orchestration operators. Throughput mode is sized for a household or a small team.
- **Byte-identical output across different device types** (ADR-0019). The product states this rather than implying it.
- **A new model interchange format** (ADR-0005). Packs are a local cache that is safe to delete.
- **Dense weights streamed from NVMe at interactive speed.** Every GB of dense overflow costs ≈170 ms per token at 6 GB/s [estimate, cost model §3.3]. The solver refuses such placements in latency mode unless nothing else fits, and then labels the result.
- **Dependence on paths users cannot install,** such as GPUDirect Storage on consumer GPUs (ADR-0013). GPUDirect Storage and DirectStorage are not in v1; the probe reports the capability only (RFC-0021 §9).
- **Neuron-level activation sparsity as a core mechanism.** It needs ReLU-sparse models, so it is a plug-in at most.
- **A full chat client, IDE plugin or model hub.** Syntropy serves APIs and a dashboard. From M2 the dashboard has a minimal chat view, so someone who installed from an app icon can talk to a loaded model: one conversation at a time, on native sessions, with no history sync, attachments or plugins. Full clients come from elsewhere.
- **Multimodal input in v1.** v1 is text-only. Image, audio and video inputs are refused with `unsupported_content_type`, and vision and audio towers are not loaded. The graph IR, the census and the API reserve the hooks a later release needs (an `EmbedInject` node fed by an encoder segment, an encoder node class, and census classes for tower tensors), so adding an encoder needs no IR break.
- **Localisation in v1.** v1 is English-only. Every human-facing CLI and dashboard string is externalised from M1, so translations can follow without a rewrite.

### 3.4 How Syntropy is built

Syntropy is open source from its first commit, and AI agents build it by executing the tasks in `implementation/` (D8, ADR-0021). Each task carries testable acceptance criteria and the ADR and RFC sections it implements, so a task is a unit an agent can finish and a reviewer can check. The areas that divide the plan (placement, tiers, distributed, compute, product, quality) are domains of the code, not groups of people, and nobody owns one.

Maintainers are the only human role. They approve the change classes ADR-0021 decision 5 lists (security-critical paths, CI and release configuration, agent instruction files, new model families, new direct dependencies, and any change a gate cannot check), and they do the steps that need a person: running the reference rigs, holding offline signing keys, and answering security reports. Keys that need several people are held by different maintainers. Every change passes automated gates and an independent review agent, and the parity harness, the conformance gates and the check against each publisher's reference output exist in M0, before the code they guard.

For this document that means two things. The schedule is bounded by rig time and by how fast changes pass the gates, not by headcount, and the milestones in §8 carry no dates. And a target in §7 counts as met only when its RFC-0061 measurement passes on a reference rig at a recorded commit. A simulation run or an agent's report does not meet it.

### 3.5 If delivery falls behind

If measured progress misses the plan, 1.0 drops features one step at a time in the order RFC-0001 §16 sets, starting with Tier 3 mobile and never touching a quality gate. Mac-pair tensor parallelism follows the same direction on its own trigger: if the raw-path probe measures `α_eff` above 65 µs, SC-17 becomes report-only at M3 (§7.3).

## 4. Users and scenarios

The people Syntropy is for own hardware that they already use for something else: a gaming PC, a Mac for work, a mini PC under the TV. They want a model larger than any one of those machines holds, and they don't want to become cluster operators to run it. Each scenario below gives the hardware, the placement the solver should arrive at, what the cost model predicts, and what decides success. Every rate is [estimate] from the cost model; none is a Syntropy measurement.

### S1. One gaming PC, a frontier MoE beyond its VRAM

- **Hardware.** RTX 4090 (24 GB) or RTX 5090 (32 GB), 128 GB of DDR5-6000 in two channels, a Gen4 or Gen5 NVMe drive in a CPU-attached slot.
- **Models.** DeepSeek-V3-class at Q4_K_M (404 GB) or Q2_K (244 GB), or Qwen3-235B-A22B at Q4_K_M (142 GB).
- **Placement.** VRAM holds the hot set (attention, dense layers, shared experts and LM head; 9.7 GB for DeepSeek-V3 at Q4), the KV cache and a VRAM expert cache. Host RAM holds routed experts in order of measured popularity, and the CPU computes VRAM misses in place there; the share the GPU copies is DMA-registered, the rest is locked anonymous memory. The NVMe pack holds every expert in 4 KiB-aligned extents, one per expert. Next-layer prediction issues NVMe → RAM prefetches, and MTP heads drive speculation.
- **Prediction** (cost model §3.3, with misses read from NVMe while the CPU computes, as in §7.2):
  - DeepSeek-V3 Q4: 0.7 tok/s at h = 0.29 (no locality), 2.4 at h = 0.8, 4.6 at h = 0.9, capped at 5.3 by DRAM.
  - DeepSeek-V3 Q2_K: 4.4–9.3 tok/s at h = 0.9–0.95.
  - Qwen3-235B Q4: 4.3 tok/s at its uniform prior h = 0.84, and 7.8 at h ≥ 0.95.
- **What decides it.** The RAM hit rate and the CPU's DRAM bandwidth decide it, not the GPU. A Gen5 drive at ≈12 GB/s sustained lifts h = 0.85 from 2.1 to ≈3.2 tok/s, but it cannot pass the DRAM cap. The model picker has to show the user this trade before the download starts.

### S2. A Mac and a PC on one desk

- **Hardware.** The gaming PC from S1 with 64 GB of RAM, a MacBook Pro M4 Pro with 48 GB, and optionally a Strix Halo mini PC with 128 GB. They connect over 2.5 GbE (the MacBook through a USB-C adapter or Thunderbolt) or over Wi-Fi.
- **Joining.** The nodes find each other over mDNS. No IP addresses are typed.
- **Placement.** In latency mode the solver fills the fastest memory first, keeps layers contiguous and keeps each MoE layer's attention with its experts. The PC's DRAM is used last, because at 70 GB/s it is the slowest place to execute. A departure-prone node (anything with a battery, anything on battery power, or a node reachable only over Wi-Fi or an overlay) hosts a stage only when the plan gains at least 10% [target] from it, or when capacity needs it, because such a node is the likeliest to stall everyone's stream by leaving (RFC-0011 §4). Wi-Fi hops are costed at their measured mean, with a bound on the stall rate (RFC-0011 C14), and there is no tensor parallelism on this fabric.
- **Prediction** (§4c, refined in RFC-0011 §15.3 with per-stage KV and a 1 ms `t_fixed` per stage):
  - Llama-70B Q4: the default plan is a pipeline from the 4090 to the Strix Halo at 6.2 tok/s, against 2.7 on the PC alone and 4.0 on the Strix Halo alone. A PC → MacBook pipeline would reach 6.5 tok/s, but that is a 5.2% gain from a laptop, below the 10% margin, so the MacBook stays out of the ring. A third stage adds a hop and a `t_fixed` for no byte-time gain.
  - Qwen3-235B-A22B Q4: 15.8 tok/s at η = 0.7 and 12.4 at η = 0.55 on 2.5 GbE. At the Wi-Fi median it drops to 12.9 and 10.5, and a single p99 Wi-Fi hop quadruples the token time.
- **What decides it.** On 2.5 GbE the network is 0.3% of token time. The gain comes from moving bytes off the PC's DRAM onto faster unified memory, so the model recommendations for clusters like this should favour MoE.

### S3. A laptop joining from away

- **Situation.** The model runs on the home cluster. The user is travelling with a laptop and joins with an invite token created at home, over iroh/QUIC with hole punching or a relay, or over Tailscale.
- **Placement.** The solver costs the WAN link with its measured α and p99. A stage on the laptop would put two WAN crossings into every token's ring. At an illustrative α of 50 ms (to be measured), that caps decode at 10 tok/s whatever the laptop computes. So the laptop becomes the ingress node only. It sends the request, and the tail stage at home streams tokens to it point to point.
- **What decides it.** Joining takes one token and no port forwarding. The WAN adds one round trip to TTFT and one-way latency to each displayed token, but nothing to the decode critical path. Back on the LAN, the laptop becomes eligible for a stage at the next re-placement.

### S4. An agent with long sessions

- **Situation.** A coding or research agent sends many turns that share a long prefix (system prompt, tool schemas, files), at 32K–128K context, with tool calls and reasoning.
- **Physics** (§3.6; backends note §4.2):
  - Llama-70B's GQA cache is 320 KiB per token. That is 10 GiB at 32K, which adds 10.7 GB of reads to every token, a quarter of its Q4 weight bytes.
  - DeepSeek-V3's MLA cache is 68.6 KiB per token, or 2.1 GiB at 32K.
  - Hybrid linear-attention models cache 20–30 KiB per token, or 2.5–3.75 GiB at 128K.
- **Placement.** KV goes in the fastest tier, and its bandwidth is part of the objective. Paged KV spans VRAM and pinned RAM, and it competes with experts for VRAM. A prefix cache is keyed by token-prefix hash. Parked sessions go to NVMe, bound to a hash of the model files and of every setting that changes meaning. Recurrent state for hybrid layers moves with its sequence.
- **What decides it.** The agent's next turn prefills only its new suffix. A resumed session restores its KV instead of re-reading the chat: a surveyed system restored 63K tokens of state (1.2 GB) in under a second, against about 30 s to re-prefill [reported by a surveyed system].

### S5. A small household server

- **Situation.** Three to six people share a Mac Studio or a Strix Halo box, joined by the gaming PC when nobody is playing on it. They select the throughput objective, with a per-user floor (D2's second mode).
- **Physics** (§3.6, §3.9):
  - KV capacity binds first: 32 users × 8K × 320 KiB is 80 GiB of KV for Llama-70B.
  - MoE batching amortises poorly. Uniform routing touches `U(B) = E·(1 − (1 − k/E)^B)` experts per layer, so 8 concurrent DeepSeek-V3 tokens touch 7.2× the expert bytes of one.
  - On one M3 Ultra, aggregate decode goes from 14 to 161 tok/s for dense Llama-70B as B goes from 1 to 32, but only from 20.7 to 56 for DeepSeek-V3 at 4-bit.
- **Placement.** The solver balances stage times so that bytes are proportional to W_eff. It batches up to the ridge point of the bottleneck tier or the KV limit, adds replicas where capacity allows, and applies admission control on KV.
- **Prediction** (§4c2). The three-machine cluster from S2 gives ≈16 tok/s aggregate on Llama-70B with three or more streams in flight. Wi-Fi is acceptable in this mode.
- **What decides it.** The per-user floor holds under load. A new request waits in the queue rather than evicting a running session's KV.

## 5. Platform matrix

Support tiers are fixed by ADR-0020. Inside this plan, "Tier 1/2/3" with a number always means a support level. A memory tier is always written as a tier kind (`Vram`, `HostPinned`, …) or a phrase such as "the RAM tier". The glossary records this collision.

| Support tier | What it promises | Platforms | Backend and I/O notes |
|---|---|---|---|
| **Tier 1** | CI on real hardware every release; a failure blocks the release | Linux x86_64 + NVIDIA (CUDA 13, Turing and newer) | native CUDA provider; io_uring with registered buffers; RDMA via ibverbs/RoCE where present |
| | | macOS arm64 (Metal) | native Metal provider (Metal 4 tensor ops where present); `F_NOCACHE` + `preadv` pool; RDMA over Thunderbolt 5 on macOS 26.2+; GPU wired limit applies |
| | | Windows x86_64 + NVIDIA | CUDA 13; IoRing on Windows 11, IOCP fallback; about half of RAM can be pinned; WSL2 caps `cudaHostAlloc` at ≈1 GiB |
| | | Linux, Windows, macOS CPU-only | native CPU provider: AVX2, AVX-512, AMX, AVX-VNNI on x86; NEON, SVE2, SME2 on arm64 |
| **Tier 2** | CI build every release, plus periodic hardware runs | Linux + AMD ROCm | HIP provider; ROCm version under review (OQ-P5) |
| | | Any OS + Vulkan (AMD, Intel, NVIDIA) | Vulkan provider (Slang → SPIR-V, cooperative matrix where present) |
| | | Linux arm64 (GB10-class, Raspberry Pi 5-class CPU) | GB10 runs as one `Unified` tier |
| | | AMD Strix Halo | iGPU over the `Unified` tier; the CPU side sees under half the DRAM bandwidth |
| **Tier 3** | Best effort; builds may lag a release | Android (as a node) | ggml provider (Vulkan, OpenCL, Hexagon); `pread` pool, since io_uring is blocked for apps |
| | | iOS (as a node, small slices only) | statically linked Metal and CPU; per-process memory cap (OQ-P8) |
| | | Intel SYCL, CUDA 12 legacy (Pascal, Volta), WebGPU | ggml provider |
| | | Linux under WSL2 with an NVIDIA paravirtual GPU | CUDA provider; `cudaHostAlloc` capped at ≈1 GiB, so ordinary memory is registered instead and KV does not spill to `HostPinned` by default (ADR-0020) |

Every node in a cluster can run a different row of this table. The resource graph records which providers and transports each node has, and the solver never assumes a feature that a node did not report.

## 6. Target model classes

Syntropy targets dense models from 7B to about 130B parameters, and MoE models up to 2026 frontier scale: Kimi K3 (2.8T total, 104B active, 896 routed experts, top-16), DeepSeek V4-Pro (1.6T / 49B), Qwen3.8-2.4T-A95B, Kimi K2.5 (≈1T / 32B), GLM-5 (744B / 40B) and Mistral Large 3 (675B / 41B) (backends note §4.1). The v1 families are set by ADR-0003. Many frontier models ship natively in 4-bit (MXFP4, NVFP4, INT4) with FP8 or BF16 elsewhere, so per-tensor dtypes in the census matter as much as parameter counts. Kimi-K2-Thinking's BF16 hot set alone is 20.8 GB per token, against 11.9 GB of INT4 routed bytes (§3.9).

The rule that sizes everything below is `tok/s ≤ W_eff / bytes_per_token`, with `bytes_per_token ≈ b·N_act + KV`. Each cell is [estimate] from the cost model at the W_eff stated, excluding KV and `t_fixed` unless the note says otherwise. Each cell is an upper bound on what a good runtime could reach, not a promise.

### 6.1 Dense

| Model (bytes per token) | CPU-only, 2-ch DDR5 (W 70) | RTX 4090 + DDR5 | Strix Halo 128 GB (W 179) | M5 Max 128 GB (W ≈430) | M3 Ultra (W 645) |
|---|---|---|---|---|---|
| 7B Q4_0 (3.79 GB) | ≈18 | ≈210 (all in VRAM) | ≈47 | ≈113 | ≈170 from bandwidth alone; one existing engine measured 92, and the cost model's two-point fit puts the gap down to ≈5 ms of `t_fixed` |
| 70B Q4_K_M (42.5 GB) | ≈1.6 (64 GB RAM) | ≈2.8 (§4a); ≈7.1 at IQ3_XXS; ≈6–9 with a draft model | ≈4.2 | ≈10 | ≈15; a third-party run of an existing engine measured 14.1 |
| 128B at Q4 (≈77 GB, `b` = 0.6) | ≈0.9 (128 GB RAM) | ≈1.2 (128 GB RAM; 58 GB of it CPU-executed) | ≈2.3 | ≈5.6, if it fits under the GPU wired limit | ≈8.4 |

For a 128B dense model at FP8 on M5 Ultra, vendor-published results over RDMA on Thunderbolt 5 give 7.70 tok/s on one node, the same 7.70 on a two-node pipeline, 13.69 with two-node tensor parallelism and 23.37 with four nodes [measured: vendor-published, multi-device note §4.5]. That is the cost model's prediction made visible: a pipeline adds capacity and no single-stream speed, and tensor parallelism over an RDMA-class link adds both.

### 6.2 MoE

| Model (bytes per token) | Gaming PC: 24 GB GPU + 128 GB DDR5 + NVMe | One 512 GB Ultra-class Mac | Two or more 512 GB Macs over RDMA on TB5 | Mixed home cluster (S2) |
|---|---|---|---|---|
| Qwen3-235B-A22B Q4 (≈13 GB) | 4.3 at h = 0.84; 7.8 at h ≥ 0.95 | ≈33 (W 426) | – | 15.8 (η 0.7) / 12.4 (η 0.55) |
| DeepSeek-V3 671B, 4-bit (20.5 GB at 4.5 bpw) | Q4: 0.7–5.3 by RAM hit rate; Q2_K: 4.4–9.3 | bound ≈40; ≈21 measured (community); ≈31 on M5 Ultra at η = 0.53 | PP ≈20.7; TP = 2 at α_eff = 40 µs ≈34.5 | needs NVMe tiers on the nodes |
| Kimi-K2-Thinking 594 GB (32.7 GB) | ≈1–2; needs a 32 GB card or an 8-bit hot set, which counts as approximate (D3) | does not fit | PP ≈13; TP = 2 ≈17–23; four nodes measured 28.3 in a third-party review | – |
| Kimi K3 2.8T, MXFP4 (≈1.49 TB; ≥55 GB) | RAM holds ≈8% of the routed pool, so ≈24 GB per token comes from NVMe: ≈0.25 tok/s, background use only | does not fit | four nodes, PP: ≈7.7 (M3 Ultra) to ≈11.7 (M5 Ultra); TP/EP subject to the §3.8 gate | – |

The Kimi K3 row assumes every weight is MXFP4 at 4.25 bpw (2.8T × 0.531 B ≈ 1.49 TB; 104B active × 0.531 B ≈ 55 GB per token) and W_eff = 0.53 × peak. The PC cell uses the routed bytes from the backends note §4.3 (16 × 92 × 16.73 MiB ≈ 25.8 GB) at 6 GB/s sustained.

### 6.3 What the tables say

- **Dense models up to 8B** run interactively on every Tier 1 device, including CPU-only ones. Their speed on fast devices is set by `t_fixed` as much as by bandwidth, which is why §7 puts a hard budget on `t_fixed`.
- **Dense 70B–130B** needs unified memory, a cluster, or a lower-bit quant. On one discrete-GPU PC the CPU's DRAM sets the speed, and NVMe can't help: each GB of dense overflow costs ≈170 ms per token.
- **MoE from 100B to 400B** is the class that tiering serves best. It is ≈2× faster than a dense 70B on the same three-machine cluster (§4c3), and on a 128 GB PC its speed depends on the RAM hit rate.
- **MoE from 600B to 1T** is reachable on one PC at Q2–Q4 (1–9 tok/s), on one 512 GB Mac (≈20–30), or on two Macs over RDMA (≈20–35).
- **MoE beyond 1.5T** needs four or more large-memory nodes. One PC can hold it on NVMe but not run it interactively, and Syntropy says so before the user downloads 1.5 TB.
- **Model recommendations follow the same arithmetic.** The model picker computes these cells from the user's own resource graph instead of looking them up.

## 7. Success criteria

Every number in this section is a **[target]**. Each row names the formula it comes from, or says that it is a chosen budget. Illustrations are [estimate] from the cost model's worked examples. They show the scale of a target on a known machine, and they are not results.

Conventions:
- `W_probe` is the effective bandwidth that the RFC-0010 probes measure on the machine under test, using the real kernel.
- `W_peak` is the published limit of the narrowest element. On an AM5 Ryzen that may be the CCD link rather than the DIMMs.
- `B_i` always includes KV bytes.
- The reference rigs are the cost model's examples (a), (b1), (b2) and (c). RFC-0061 fixes the exact machines.

### 7.1 Decode on one device

| ID | Target | Derived from | Illustration |
|---|---|---|---|
| SC-1 | Kernel efficiency `η = W_probe / W_peak`: ≥ 0.80 for dense GEMV on `Vram` or `Unified`; ≥ 0.63 for the grouped MoE expert kernel; CPU execute-in-place ≥ 0.75 × min(the probe's all-core DRAM read rate, the scheme's in-cache ALU-bound rate), gated for schemes whose ALU-bound rate exceeds the DRAM rate and reported, not gated, for the others | Dense: the best calibration anchor (0.79, M3 Ultra fit) and example (a)'s 0.8 assumption. MoE: halve the gap between MoE (0.53) and dense (0.73) on the same Mac, giving (0.53 + 0.73)/2. CPU: example (a)'s 70 of 90 GB/s. Some CPU kernels, the i-quants among them, are arithmetic-bound below the DRAM rate on mid-range Tier 1 CPUs, so a DRAM-only bar could not pass for them (RFC-0040 §9) | – |
| SC-2 | `t_fixed ≤ 0.2 · B_ref / W_probe`, with `B_ref` = 3.79 GB (7B Q4_0). Applies as written on CUDA and HIP, and to dense stages on every backend. **Provisional for MoE stages on Metal and Vulkan:** those backends need one host wait per host-interacting MoE layer, because they cannot capture host waits into one step (RFC-0012 §7.5). Their value is set from the M0b spike PLC-E7-T20 and recorded here with the measurement; until then no number applies | Keeps a 7B token's time outside byte reads under 1/6 | ≈0.95 ms on an RTX 4090 at 800 GB/s; ≈1.2 ms on M3 Ultra at 645 GB/s. One existing Metal engine's fitted value is ≈5 ms. RFC-0012 estimates 0.6–2.9 ms of host-wait cost on a 58-layer Metal MoE stage [estimate], which is why that case waits for measurement |
| SC-3 | Measured `t_tok ≤ 1.05 · t_model`, where `t_model = Σ B_i / W_i(probe) + n_switch·t_switch + t_fixed`, all probed | Chosen budget: at most 5% of token time that the cost model does not explain | – |
| SC-4 | Decode ≥ X × the peak bound `1 / (Σ B_i / W_i,peak + t_fixed,SC-2)`. X = 0.76 for one execution tier (all `Vram` or all `Unified`); 0.67 for dense split across `Vram` and CPU DRAM; 0.60 for MoE with every expert in `Vram` or `Unified` | SC-1 ÷ SC-3. For the CPU split, also × 0.94, the measured-to-peak ratio of DDR5 DRAM (90 of 96 GB/s) | (a) 70B on a 4090 + DDR5: bound 3.8 → ≥ 2.5 tok/s (the cost model estimates 2.8). M3 Ultra 70B: bound 18.8 → ≥ 14.3 (an existing engine measured 14.1). M3 Ultra DeepSeek-V3 4-bit: bound 38.1 → ≥ 22.9 (≈21 measured) |
| SC-5 | MoE over RAM and NVMe: `t_tok ≤ 1.10 · t_overlap(h)` with h measured, where `t_overlap = B_hot/W_hot + max(B_routed/W_ram_exec, ((1 − h)·B_routed + B_waste)/W_nvme)`, and `B_waste` is the measured bytes per token prefetched from NVMe and never used | Cost model §3.3, optimistic form, with two corrections. A missed expert must still be computed once it lands, so the RAM-side term covers all routed bytes; without it the §4b2 table predicts h = 0.95 to be faster than h = 1. Mispredicted prefetches share the drive with demand reads, so their bytes count on the NVMe side; RFC-0023 §10's expected-value gate keeps them small on a busy drive. The 1.10 is a chosen budget. RFC-0023 adds the VRAM-cache term | (b2) DeepSeek-V3 Q4 at h = 0.9: 217 ms → ≥ 4.2 tok/s |
| SC-6 | At batch or chunk size n < n\*, weights homed in host RAM are computed in place and never streamed. At n ≥ n\* the groups the plan streams (homed in `HostPinned`, the DMA-registered tier) are streamed in micro-batches | `n* = F_cpu · b / (2 · W_pcie)` (§3.2) | n\* ≈ 12–24 on a desktop CPU at 1–2 TFLOPS. Streaming 70B's RAM half at n = 1 is 2.7× slower |

### 7.2 Storage, memory and the expert cache

| ID | Target | Derived from | Illustration |
|---|---|---|---|
| SC-7 | Expert reads from NVMe at ≥ 0.85 × the probe's sustained, throttled rate at the same request size and queue depth | The cost model expects 80–95% of sequential spec for contiguous expert extents at QD 2–8 (§2.4). The probe already includes the drive's limits, so the remaining 15% covers submission and alignment | Probe at 6 GB/s → ≥ 5.1 GB/s of expert bytes. Page-cache paths in surveyed engines delivered 1.5–2.6 GB/s on Gen4 |
| SC-8 | ≥ 50% of NVMe expert misses issued before that layer's router runs, where "issued" means submitted to the device, not queued in a ring | Matches the share of SSD misses that a next-layer router predicted in one surveyed engine ("about half") | – |
| SC-9 | Aggregate expert-cache hit rate over every recorded routing trace, after a warm-up of 4,096 accesses per layer, ≥ max(cache fraction, LRU hit rate at equal capacity) − 0.5 points | The uniform prior `h = cache fraction` (§3.3) and the LRU baseline. The 0.5-point tolerance and the warm-up are chosen budgets: the cache follows whichever of its shadow policies led over the last two adaptation windows (RFC-0023 §5.3), so a trace that changes faster than that can trail the best shadow briefly | – |
| SC-10 | Streaming host → GPU at ≥ 0.90 × the probe's pinned rate. Prefill micro-batch ≥ `n_hide = F_gpu · b / (2 · W_pcie)` where VRAM allows | Copy hidden behind compute when `2Pn/F_gpu ≥ Pb/W_pcie`. The 0.90 is a chosen budget | RTX 4090 at 165 TFLOPS, b = 0.6, 25 GB/s: n_hide ≈ 1980, consistent with §4a's 2048-token micro-batches |
| SC-11 | Predicted peak bytes per tier per node within 5% of measured. Zero OOM kills across the test matrix. Progressive pinning never takes free memory below the RFC-0020 reserve | Chosen budgets | – |

### 7.3 Across devices

| ID | Target | Derived from | Illustration |
|---|---|---|---|
| SC-12 | Added time per hop ≤ `α_probe + S/β_probe + Z`, with `Z ≤ 2 · t_switch(probe)` | One device → host hand-off at the sender and one host → device at the receiver; the cost model puts `t_switch` at 5–20 µs (§3.2), so Z ≤ 10–40 µs | 16 KiB on 2.5 GbE: 100–250 µs + 56 µs + ≤ 40 µs |
| SC-13 | Single-stream pipeline: `t_tok ≤ 1.05 · (Σ_s t_s + D · t_hop)`, D hops in the ring | §3.7 with SC-3's budget | (c1) 4090 → Strix Halo, the default plan: 26.8 + 132.5 ms of stage time, 1 ms of `t_fixed` per stage and 2 × ≈0.3 ms of hops, 161.9 ms in all → ≥ 5.8 tok/s (RFC-0011 §15.3 predicts 6.2) |
| SC-14 | When the model fits on one node, the chosen plan is never slower, predicted or measured, than the best single-node plan | §3.7: a pipeline never speeds up a single stream with the same bytes | Property test over simulated fabrics (RFC-0060) |
| SC-15 | Throughput mode: aggregate ≥ 0.85 × `min(M, D) / (Σ_s t_s + D · t_hop)`, capped at `1 / max_s t_s` | §3.7; the 0.85 is a chosen budget | (c2) ≈16 tok/s → ≥ 13.6 |
| SC-16 | Cross-node TP or EP only when `α_eff(probe) < α_eff,max = (B_w/W_eff)(1 − 1/N) / (2 · n_layers)`. `α_eff` is the full one-way time of one collective through the real device path, including the transfer time S/β of each member's partial (h_model × 4 B at decode): half the measured round trip, used alone, with no S/β added again | §3.8 | Llama-70B Q4 on M3 Ultra-class nodes: α_eff,max ≈ 206 µs at N = 2 |
| SC-17 | Per-collective cost on RDMA over Thunderbolt 5: `α_eff ≤ α_wire + S/β + 2 · t_switch`, with `α_eff` as defined in SC-16 and S the member's partial. **Report-only at M3** if the raw-path probe on a TB5 Mac pair, run in late M1, measures `α_eff` above 65 µs; cross-node TP on Mac pairs then moves to M4 or after 1.0 | No host round trip beyond one hand-off per side. With 10–25 µs wire latency, 5–20 µs per hand-off and S/β ≈ 11.5 µs for a 28 KiB partial at 2.5 GB/s, this gives 31.5–76.5 µs (RFC-0032 Performance) | (b1) DeepSeek-V3 TP = 2 on two M3 Ultras: ≈34.5 tok/s at 40 µs, against 20.7 on one node and ≈23 at the ≈150 µs back-solved for a surveyed system |

### 7.4 First token, sessions and speculation

| ID | Target | Derived from | Illustration |
|---|---|---|---|
| SC-18 | `TTFT ≤ 1.15 · max(FLOPs/F_eff, Σ_streamed B · n_pass / W(B), activation bytes / β_hop)` + queueing. `W(B)` is `W_link` for streamed bytes resident in RAM and `W_nvme` for bytes resident on NVMe; `n_pass` is `⌈L/ubatch⌉`, or the number of layer-major groups on a stage that prefills layer-major over an `Nvme`-homed pool (RFC-0012 §4.4) | §3.5; the 1.15 is a chosen budget. Bytes that live on NVMe stream at the drive's rate, not the link's; the cost model's b2 prefill figure (394 GB at 25 GB/s) omits that term, as its 2026-10-20 note says | (a) 4K prompt on 70B, estimated at 4–6 s → ≤ 4.6–6.9 s |
| SC-19 | Resumed session: `TTFT ≤ 1.25 · KV_bytes / W_read(tier) + t_prefill(new tokens)` | Restore is a sequential read; the 1.25 is a chosen budget | 1.2 GB of parked state from NVMe at 6 GB/s → ≤ 0.25 s plus the new turn's prefill. A surveyed re-prefill took ≈30 s |
| SC-20 | Tokens prefilled = tokens after the longest cached-prefix match, exactly | Prefix-cache contract (RFC-0022) | Agent benchmark |
| SC-21 | Realised speculative speedup ≥ 0.85 × the predicted `E[tok] · t_target(1) / (γ·t_draft + t_target(γ+1))`, with `E[tok] = (1 − a^(γ+1)) / (1 − a)` and a measured. Speculation switches itself off when the prediction is below 1 | §3.10; the 0.85 is a chosen budget | a = 0.8, γ = 4 → 3.36 tokens per pass |

### 7.5 Install, discovery and reliability

| ID | Target | Derived from | Illustration |
|---|---|---|---|
| SC-22 | Startup probe ≤ 15 s per node (≈11 s critical path), with nodes probing in parallel. A model's own probe takes ≤ 5 s per node, and the coordinator waits at most `probe.model_wait` (7 s) for it before solving. The 60–120 s sustained NVMe run starts after the first model's load completes, or in an idle window, never while that volume is packing or loading, and does not delay the first token; until it finishes, plans list the sustained rate as an assumption | §5.3, whose step budgets sum to ≈11 s; RFC-0010 §3.4 and §4's phase table | – |
| SC-23 | `t_pack + t_load ≤ 1.25 · (2·S / W_read + S / W_write)` for a shard of S bytes, with download time `S / W_download` counted separately | The pack build reads the source once and writes the pack once, and load reads the pack once. The 1.25 is a chosen budget | 42.5 GB on a Gen4 drive (7.45 GB/s read; 6 GB/s write assumed) ≈ 18.5 s → ≤ 23 s |
| SC-24 | From install to first token: one install command or installer, one question and one model choice, no configuration file, no typed address. The question asks what the model is for, chat or agent, and sets the planned context length and a TTFT ceiling. The model choice has a quick-start default computed from the resource graph, bounded by its download time, and at `Bits4` or above. **Exceptions:** on a Linux GPU host installing from deb or rpm, it is two commands, the package and then the backend module package that `setup` names, unless RFC-0052 open question 7 adopts a package-manager hook. On a channel where the CUDA redistribution fallback applies (ADR-0507 rule 7), the user also installs the CUDA toolkit, which `setup` names | D4; the install recommendations of the research notes. Package hooks run without a terminal, so `setup --package` asks nothing and leaves consent decisions to the first `syntropy run` (RFC-0052 §3). The use question exists because a default sized for chat, at the picker's 8K context (RFC-0054 §5.2), fails a coding agent's first long request (S4) | – |
| SC-25 | A previously paired node rejoins within 2 heartbeats of finishing its probe, with no interaction. A new LAN node needs one pairing confirmation (ADR-0302) and is admitted within 2 heartbeats of approval. A dead node is detected within 3 missed heartbeats, which raises the informational `Suspect`. `NodeFailed`, the event that triggers re-placement, follows within 8 s of the node's last heartbeat on a LAN | ADR-0009's 1 s heartbeat; the 3 is a chosen budget. `NodeFailed` comes at `Suspect` + 2–4 s + one coordinator probe period, about 6–8 s (RFC-0030 §5.4), so 8 s bounds it | ≤ 2 s and ≤ 3 s; `NodeFailed` ≤ 8 s |
| SC-26 | An off-LAN device joins with one invite token, with no port forwarding | D4, ADR-0011 | – |
| SC-27 | A coordinator change fails zero in-flight requests | ADR-0009 | – |
| SC-28 | When a stage node other than the ingress is lost and the survivors have capacity, the sequence resumes with no client-visible error, within `t_detect + t_solve + S_delta/β_fetch + t_replay`, where `t_replay = history bytes / β + prefill of the lost layers`. For a lost head the history term is zero: the new head re-prefills its own layers from the ingress transcript | Activation-history replay from every sending stage (ADR-0311; multi-device note §10.2) | 4K tokens of boundary history at width 8192 in F32 is 128 MiB per boundary |
| SC-29 | Exact mode passes the parity harness within the RFC-0042 per-dtype tolerances on every Tier 1 platform, every release, up to each family's longest certified context. Every certified family's reference matches the publisher's golden logits and hidden-state norms within the definitional tolerance. Every approximate setting in effect appears in the API response, the CLI and the dashboard, and so does the "beyond the certified context" label on positions past it | D3, ADR-0019 (decisions 1a and 7) | – |
| SC-30 | Predicted `t_tok` within ±15% of measured on the reference rigs | Must stay below the ≈20% drift that triggers re-placement (§5.3), or Syntropy re-plans on its own error | – |
| SC-31 | Slow re-placement ≤ 2 s for up to 8 nodes, including expert-group variables. Fast per-request routing ≤ 1 ms. The DP fallback returns a feasible contiguous plan whenever one exists | Chosen budgets. For scale, a smaller layer-window ILP solved in 10–12 ms for 4–32 devices [reported by a surveyed system] | – |
| SC-32 | No Tier 1 reference benchmark moves outside the RFC-0061 noise band between releases. Every published number names its commit, prompt length, quantization and interconnect | The survey found decode regressions of ≈16–20% shipped between nearby releases of one engine | – |

## 8. What "feature complete" means

Syntropy is feature complete when every box through M3 is ticked and its SC rows pass on the reference rigs. A box is ticked when the tasks behind it have merged through the gates and the independent review agent and meet the definition of done in `implementation/README.md` (§3.4). M4 turns that into 1.0. Each item names the RFC that owns it. RFC-0001 defines the shared types that all of them use.

RFC-0001 §16 and the epics are authoritative for milestone scope, and this checklist is derived from them: `implementation/tools/roadmap.py` writes it from §16's deliverables and exit criteria, with each milestone's task count from the epics. To change a milestone, change RFC-0001 §16 and the epics, then re-run the script; text between the markers is overwritten.

<!-- BEGIN GENERATED: vision-milestones -->
### M0a: walking skeleton

182 tasks in the epics (`implementation/roadmap.md`, M0a).

- [ ] The workspace builds on every Tier 1 OS, and CI enforces the §4.2 dependency rules (RFC-0062).
- [ ] The merge gates of ADR-0021 decision 5 run from the first commit: build, lint and tests; the DCO, licence and dependency checks; the workflow security lint; the independent review-agent job; and the sandbox in which agent runs read untrusted content. The repository's security baseline, the private vulnerability-reporting channel and the agent trust tiers are in place from the first commit too (RFC-0062 §12.5, §12.6, §13; ADR-0021; ADR-0624).
- [ ] GGUF reader, census and manifest for `llama` at Q4_K and Q8_0. Such a model runs end to end through the OpenAI Chat Completions API with streaming, through the daemon and a real runner, on the CPU provider and on CUDA (RFC-0041, RFC-0043, RFC-0050, RFC-0051).
- [ ] Execution is eager; this exit needs no captured graphs (RFC-0012).
- [ ] Op selection takes native CPU and CUDA ops from a static allow-list for the `llama` op set and falls back to the ggml provider. Certificate-gated selection arrives in M0b (RFC-0040).
- [ ] Conformance is a gate. Every op `llama` uses passes on the CPU provider in hosted CI and on CUDA on R-ADA, against the ggml provider and the F64 reference, as a required check on every change that touches a provider (RFC-0060 §4).
- [ ] The parity harness is a gate. Teacher-forced parity of `llama` against the `testkit` reference executor is a required check on every change to a kernel, a provider, the graph or a family module, on the CPU in hosted CI and on R-ADA. Until M0b's calibration it uses the tolerance table's provisional ceilings (RFC-0060 §5, RFC-0042).
- [ ] The model-definition oracle gates `FamilyCert`. Golden logits and hidden-state norms for `llama` come from the publisher's reference implementation, run in F32 on the same dequantised weights, and are stored offline. `llama`'s `FamilyCert` is issued only when they pass, and they re-run on every change to the family module or a shared block (RFC-0042 §7, RFC-0060 §7, ADR-0019).
- [ ] Hardware jobs on R-ADA run for changes on this repository's branches. Fork pull requests get the hosted checks only, until M1b (RFC-0062 §3).

**Exit:** `llama` at Q4_K and Q8_0 runs end to end through the OpenAI streaming endpoint on the CPU provider and on R-ADA, eagerly, with parity `smoke` passing on both, every conformance and parity gate required and green at the exit commit, and `llama`'s `FamilyCert` issued against the publisher's reference. No SC is required at this exit.

### M0b: parity, certification and the lab

131 tasks in the epics (`implementation/roadmap.md`, M0b).

- [ ] `qwen_moe` (GQA with softmax top-k MoE, every expert resident in VRAM), its `FamilyCert` gated on golden logits from the publisher's F32 reference as in M0a (RFC-0043, RFC-0042 §7).
- [ ] The other M0 schemes (Q4_0, BF16, F16) and captured execution on CUDA (RFC-0040, RFC-0012).
- [ ] Calibration on the M0 families replaces the provisional parity ceilings. The MoE ceilings stay `provisional` until the fixed-order F32 calibration reference on a GPU lands, cross-checked against the CPU `F32Fixed` reference on the nano and micro fixtures (RFC-0042 §6).
- [ ] A minimal signed `ConformanceManifest` (op and family certificates) gates op selection in place of M0a's allow-list; the CUDA provider loads as a provider library through `kernels-loader` (RFC-0060 §6.4, RFC-0040).
- [ ] The `Warming` canary step is skipped and reported `not-run` (no reference digests or packs yet).
- [ ] Probe suite v1 measures `W_eff`, pinned PCIe, NVMe sustained read and `t_fixed` within SC-22's budget (RFC-0010).
- [ ] The lab dispatcher with just-in-time runners, the benchmark harness tied to a commit, ABBA block statistics, noise bands and the PR† regression gate in report-only mode (RFC-0061, RFC-0062).
- [ ] The deterministic simulator core, the fuzz targets for the M0 parsers, and release automation (RFC-0060, RFC-0062).
- [ ] The release-signing hierarchy: offline root and targets keys on hardware tokens, online keys in an HSM-backed key service, and a dry-run signing ceremony before the first public release candidate (RFC-0062 §7, RFC-0034).

**Exit:** parity passes for `llama` and `qwen_moe` on the CPU and on R-ADA (SC-29 holds from here on, every release), and single-tier decode meets SC-4 for the all-`Vram` case on the CUDA rig. The MoE ceilings are reported `provisional`. The single-device baseline comparison under ADR-0612 gets its early read on R-ADA (spike S-10), recorded with the exit.

### M1a: Linux preview (CUDA and CPU tiering)

481 tasks in the epics (`implementation/roadmap.md`, M1a).

- [ ] Tier manager with DMA-registered `HostPinned` and locked `HostPageable` arenas per NUMA domain, the Linux budgets, and progressive locking and registration; packs with 4 KiB alignment; the io_uring and thread-pool I/O engines (RFC-0020, RFC-0021, RFC-0041).
- [ ] Expert cache with decayed LFU, CPU execute-in-place for misses, next-layer prediction fused into the router op, and prefetch. How much of the NVMe expert tier M1a takes on follows the hit curves from S-01's routing traces. Until those exist, the model class that fits in RAM leads, and DeepSeek-V3 at Q4 is a capacity demonstration (RFC-0023).
- [ ] Solver v1 on one node (devices × tiers): the DP with greedy in-node placement, and a placement preview in the CLI. The MILP is built only if S-09's simulation study shows it beating the DP by more than 5% on a reference input (RFC-0011, ADR-0008).
- [ ] Single-node KV: paging across `Vram` → host RAM → `Nvme` with the `spill_compute` choice (CPU split attention on the CUDA and CPU providers, streaming to the GPU), the prefix cache on a single-stage instance, saved sessions on one node with incremental saves, and recurrent state in the same store (RFC-0022). Cross-node KV follows in M2 (prefix cache across stages, KV export and import for re-placement) and M3 (KV migration for recovery, session restore under a changed split or from remote holders).
- [ ] The CPU provider's AVX-512 and AMX paths, and CUDA conformance on R-BW (RFC-0040).
- [ ] Single-node speculative decoding with model-native MTP heads and with independent draft models (ADR-0017 item 7). Latency mode runs up to 4 concurrent decode rows for dense models, and for RAM-resident MoE when KV fits; NVMe-tiered MoE stays serial (RFC-0012).
- [ ] The `Warming` canary gate is enforced from M1a (in M0 it is reported `not-run`).
- [ ] The preview's API: OpenAI Chat Completions with tools, reasoning and structured output; stateless Responses and fill-in-the-middle; Anthropic Messages; the native `generate` API; one-node placement previews and automatic placement on first request. Positions past the longest certified context are labelled beyond the certified context (`FidelityReport::beyond_certified_from`, RFC-0050 §10). `admin`, `sessions` and keyholder routes answer only over the local socket or with a key, and every non-loopback bind needs a key and TLS (RFC-0050, RFC-0051, RFC-0054).
- [ ] Hooks reserved for multimodal input: an `EmbedInject` node, an encoder node class and census classes. v1 is text-only (RFC-0043).
- [ ] One-command install on Linux, with the `cuda13` module fetching NVIDIA's redistributable libraries at setup; a first-run picker that asks one use question; the model catalogue (RFC-0052, RFC-0051).
- [ ] What a public build needs: imported `cargo vet` audits for dependency updates, the expedited security-release path, signed Linux artefacts and the update channels, and header parsing of newly downloaded models in a resource-limited child process (RFC-0062, RFC-0034, RFC-0041).
- [ ] A standalone network probe for alpha testers' homes, shipped mid-M1a, whose traces set M2's Wi-Fi priors, wireless timeouts and relay sizing (RFC-0030, RFC-0031).

**Exit:** SC-1 to SC-11 and SC-18 to SC-21 on the Linux single-device rigs, including scenario (a) and (b2) of the cost model. The single-device baseline comparison on R-ADA under ADR-0612 passes; it is a release gate from the first M1a release candidate. The MoE ceilings stay `provisional` until the GPU calibration reference lands.

### M1b: Windows and Apple silicon

92 tasks in the epics (`implementation/roadmap.md`, M1b).

- [ ] The Metal native provider and the Metal probes; `Unified` arenas, the wired limit and the macOS I/O engines (`F_NOCACHE` with `preadv`, and mapped reads); the NEON and SME2 CPU paths (RFC-0040, RFC-0020, RFC-0021).
- [ ] Windows: working-set locking, the IoRing and IOCP engines, and the WDDM and WSL2 specifics (RFC-0020, RFC-0021, RFC-0040).
- [ ] One-command install on Windows (MSI and winget) and on macOS (app bundle and Homebrew cask), and a non-root LaunchDaemon for headless Macs, tested with FileVault on. Signing and notarisation follow RFC-0062 §7 (RFC-0052).
- [ ] Hardware runs for fork pull requests start, each run approved by a maintainer (RFC-0062 §3).
- [ ] The Thunderbolt 5 RDMA raw-path probe on a pair of Macs, which decides whether Mac-pair TP and SC-17 stay in M3 (RFC-0031, RFC-0032).
- [ ] The pairing protocol's commit-then-reveal exchange, modelled and tested against a grinding relay before `PairRequest` freezes (RFC-0030 §6).

**Exit:** SC-1 to SC-11 and SC-18 to SC-21 on the Windows and Apple single-device rigs, wherever each applies. SC-2 on Metal and Vulkan MoE stages is provisional until the M0b spike PLC-E7-T20 sets it (`00-vision-and-scope.md` SC-2). If the Ultra-class Mac in the test matrix has 256 GB, scenario (b1)'s single-node case is measured on a model that fits it (ADR-0610).

### M2: multiple devices on a LAN

292 tasks in the epics (`implementation/roadmap.md`, M2).

- [ ] Discovery with blinded beacons sent only on networks where a member has been seen, SWIM membership, coordinator lease and event log, with the election and split-heal design model-checked before its code; admission with commit-then-reveal and number matching, and encryption (RFC-0034 core). Coordination is keyholders-only by default, offered with a second-keyholder grant at pairing, with an optional offline recovery code at bootstrap, a guard against removing the last keyholder, and re-founding (RFC-0030, RFC-0032, RFC-0034).
- [ ] QUIC, TCP and shared-memory transports; RDMA over Thunderbolt 5 on macOS (RFC-0031).
- [ ] Ring pipeline across nodes with chunked pipelined prefill; joint solver over devices × tiers × links. Latency mode minimises decode time subject to a TTFT cap of 2× the best candidate's TTFT (RFC-0032, RFC-0011).
- [ ] Shard-only fetch and P2P chunk distribution (RFC-0041).
- [ ] Session data held on other nodes is stored as ciphertext under a per-session key that the session's own node holds (RFC-0022, RFC-0034).
- [ ] The simulated fabric for multi-node tests (RFC-0060).
- [ ] Dashboard v1: topology, tiers, plan explanation, live rates, multi-node previews, and a minimal chat view with one conversation over native sessions; routing across nodes and cluster-wide API keys (RFC-0051, RFC-0050, RFC-0034).
- [ ] The embeddings API with one embedding family, on a single node (RFC-0050, RFC-0043).
- [ ] A ggml-only `vulkan` module for AMD and Intel GPUs, labelled a Tier 2 preview (RFC-0040, ADR-0020).
- [ ] Syntropy-only nightly benchmark results are public from M2 (RFC-0061).

**Exit:** SC-12 to SC-14, SC-22 to SC-25 and SC-27 on the cost model's scenario (c) rig and, if M1b's raw-path probe passed, a two-Mac Thunderbolt rig. SC-15 (throughput mode) is an M3 exit, because throughput mode lands in M3.

### M3: feature complete

212 tasks in the epics (`implementation/roadmap.md`, M3).

- [ ] Invite links with relays and hole punching, the built-in relay set read from a signed `relays.json`; Tailscale paths (RFC-0030, RFC-0034).
- [ ] Fault tolerance: activation-history replay from every sending stage, request migration, in-place relink and shard extension, KV migration on re-placement and recovery, an evictable second copy of each stage's source chunks on the node a repair would absorb it into, a 60 s detach grace that lets a client re-attach to a paused stream, degraded modes, automatic re-placement (RFC-0033, RFC-0022, RFC-0050).
- [ ] Cross-node TP/EP on links that pass the α gate; RoCEv2 behind `trusted_fabric` (RFC-0032, RFC-0031). If M1b's probe measures a raw-path `α_eff` above 65 µs between the Macs, SC-17 is report-only at this exit and Mac-pair TP moves to M4 or after 1.0.
- [ ] Speculative decoding across stages; trained draft heads loaded where a model ships them, though v1 produces none; wide trees for I/O-bound targets; asynchronous speculation behind a flag (RFC-0012, ADR-0017).
- [ ] Runner re-adoption across daemon updates (§3.4).
- [ ] The `Compressed` KV class with the DeepSeek-V4-class family (RFC-0022, RFC-0043).
- [ ] HIP and Vulkan native providers; ROCm and Vulkan modules (RFC-0040).
- [ ] Throughput mode with continuous batching (RFC-0011, RFC-0012).
- [ ] Stored Responses, kept on the ingress only, and the local-runner API shape (RFC-0050).
- [ ] Dashboard and API surfaces for invites, admissions and recovery events, and the opt-in approximate settings, each labelled (RFC-0051, RFC-0050, RFC-0054).
- [ ] Every v1 family from ADR-0003, each through the conformance suite (RFC-0043, RFC-0060).

**Exit:** SC-15, SC-16, SC-17, SC-26 and SC-28, with SC-17 report-only under the Mac-pair rule above; the chaos suite of RFC-0060 passes (node kill mid-generation completes the request).

### M4: hardening and 1.0

51 tasks in the epics (`implementation/roadmap.md`, M4).

- [ ] Tier 2 and Tier 3 platforms per ADR-0020; signed installers and update channels for every Tier 1 OS (RFC-0052, RFC-0062).
- [ ] Performance regression gates on every Tier 1 rig; published benchmarks tied to commits, comparisons with other engines included (RFC-0061).
- [ ] Wire-protocol and pack-format compatibility policy in force (RFC-0062).
- [ ] Complete observability, and an external security audit of admission, transport credentials, the update client, the local API CA and API authentication, required before the 1.0 release candidate (RFC-0053, RFC-0034, RFC-0062 §13).

**Exit:** every SC target met on its rig, or waived in writing by a maintainer with the measurement attached; zero open P0 defects.
<!-- END GENERATED -->

M3 delivers the HIP and Vulkan providers. The Linux ROCm and Vulkan platforms reach their Tier 2 bar in M4.

## 9. Principles

1. **Compute moves to the bytes.** During decode, the device whose tier holds a weight computes it. Across nodes, only activations move. Weights are streamed only for batches above n\*.
2. **Measure, don't look up.** Every bandwidth, latency and rate the solver uses comes from a probe on that machine or from a live timing. Spec sheets are sanity bounds: a probe below about half of spec flags a misconfiguration.
3. **Exact by default.** A setting that changes output is opt-in, carries the word "approximate", and appears wherever it is in effect.
4. **No single model geometry hard-coded.** Models are manifests over shared blocks. A kernel tuned for one model is a provider optimisation with a generic fallback.
5. **Every modern fast path has a fallback.** FP4 tensor cores fall back to dequantise-and-GEMM, CUDA graphs to stream launches, io_uring to a `pread` pool, RDMA to QUIC to TCP.
6. **Tokens never cross the control plane.** The event log carries membership, plans and commands. Activations go from stage to stage, and tokens go from the tail to the ingress node.
7. **Syntropy owns residency.** The OS page cache never decides which bytes are hot.
8. **Budgets are checked per node.** A plan that fits only in the cluster's total memory does not fit.
9. **A device joins the critical path only if it lowers the sum.** Every stage adds a hop and a `t_fixed`, so a weak device stays out of the ring rather than slowing it, and a departure-prone node joins only for a gain of ≥ 10% [target] or for capacity (RFC-0011 §4).
10. **A number without a commit is not a result.** Benchmarks state the commit, prompt length, quantization and interconnect. Targets stay labelled as targets until a benchmark replaces them. A published benchmark report may name a baseline engine, at its exact version and configuration, with only ADR-0612 figures and the raw data, factual wording, no logos and a no-endorsement line, after inviting the baseline's maintainers to check the configuration 14 days before first publication. Syntropy-only nightly numbers go public from M2, and nightly comparisons stay private until M4. This plan itself names no baseline (D5).
