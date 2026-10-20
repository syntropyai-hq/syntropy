# ADR-0006: Parallelism

- Status: Accepted (revised 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D1, D2, D3), ADR-0002, ADR-0003, ADR-0007, ADR-0008, ADR-0010, ADR-0011, ADR-0017, RFC-0010, RFC-0011, RFC-0012, RFC-0023, RFC-0031, RFC-0032

## Context

Batch-1 decode is memory-bound on every tier, so per-token time is a sum over the bytes each device reads, the hops between devices and a fixed overhead: `t_tok ≈ Σ B/W_eff + Σ_hops (α + S/β) + t_fixed` (`plan/research/hardware-and-performance-model.md` §0, §3.1, §3.2). Each parallelism mode changes different terms.

**Pipeline parallelism (PP)** puts contiguous layer ranges on different devices (§3.7). One token visits every stage in turn, so PP never makes a single stream faster than the same bytes on the same devices. It helps when it moves bytes off a slow tier onto another device's faster memory. A decode hop carries one hidden state (16 KiB for Llama-70B, 14 KiB for DeepSeek-V3 at BF16):

| Hop, 16 KiB | Cost [estimate, §3.7] |
|---|---|
| 2.5 GbE | 56 µs serialisation + ≈100–250 µs |
| Wi-Fi at 400 Mbps | 328 µs + 3–7 ms median, up to ≈250 ms at p99 |
| RDMA over TB5 | 6.6 µs + 10–25 µs |

In the mixed home cluster example, PP of Llama-3.3-70B across an RTX 4090 PC and a Strix Halo box reaches ≈6.5 tok/s against ≈2.8 tok/s on the PC alone. The network is 0.3% of token time on 2.5 GbE (§4c) [estimate].

**Tensor parallelism (TP)** needs two all-reduces per layer, 120–160 sequential collectives per token for 60–80-layer models (§0 item 5, §3.8). It pays only while the effective per-collective latency, wire plus GPU→host→GPU sync, stays below `α_eff,max ≈ (B_w/W_eff)(1 − 1/N)/(2·n_layers)`: ≈198–206 µs at N = 2 and ≈296–309 µs at N = 4 for Llama-70B and DeepSeek-V3 on M3 Ultra-class nodes [estimate]. Evidence:
- 2-node TP over RDMA on TB5 decoded a 128B dense model at 13.69 tok/s against 7.70 for 2-node PP (`multi-device-inference.md` §4.5).
- TP over Wi-Fi measured about 150 ms per all-reduce, about 24 s/token at 70B. TP over 10 GbE spent 34 ms/token in transfer on 4 nodes (§4.4, §4.5).
- Over 2.5 GbE, TP of Llama-70B gives 5.9–6.5 tok/s against 6.5 for PP (`hardware-and-performance-model.md` §3.8) [estimate].
- For DeepSeek-V3 TP = 2, 20 µs per collective gives ≈37.7 tok/s, while 300 µs gives 16.5 tok/s, slower than one node (§4b) [estimate].
- Effective costs back-solved from a surveyed 4-node RDMA cluster are ≈130–150 µs per collective, far above the 10–21 µs wire latency, so synchronisation dominates (§3.8).

**Expert parallelism (EP)** needs a dispatch and a combine per MoE layer, about 116 small messages per token for DeepSeek-V3. It is latency-bound like TP (§3.9).

**Inside a device**, weights outside VRAM can be computed in place by the CPU or streamed over PCIe to the GPU (§3.2):
- `t_EIP = max(P·b/W_ram, 2·P·n/F_cpu)` and `t_stream = max(P·b/W_pcie, 2·P·n/F_gpu)`, for a block of P params at b bytes each and n tokens per pass.
- At n = 1, dual-channel DDR5 (76–122 GB/s measured) beats pinned PCIe 4.0 x16 (≈24 GB/s). In example (a), execute-in-place decodes at 2.8 tok/s and streaming at 1.05 tok/s (§4a) [estimate].
- The crossover is `n* ≈ F_cpu·b/(2·W_pcie)`: ≈12 tokens at 1 TFLOPS of CPU and ≈24 at 2 TFLOPS [estimate].
- For MoE misses, using the PCIe link in parallel with the CPU helps. Sending 55% of i-quant misses over PCIe while the CPU computes the rest raised decode from 70 to 78 tok/s (`tiered-memory-inference.md` §5.4).

**Inside a node with several GPUs**, a layer split crosses each card boundary once per verify window. A helper-GPU expert tier synchronises on every layer. Peer access was trusted only after a test copy in each direction (`tiered-memory-inference.md` §16).

## Options considered

| Option | For | Against |
|---|---|---|
| A. TP everywhere, the datacenter default | Speeds up a single stream | 2 collectives per layer; loses on every link above ≈200–300 µs effective latency |
| B. PP everywhere, never TP or EP | Simple; one crossing per stage per token | Leaves the measured 1.8× TP gain on RDMA meshes unused |
| **C. PP across nodes by default; TP and EP only on links that pass a measured break-even gate; execute-in-place inside a device** | Each mode only where the cost model says it pays | A solver gate and per-mode execution paths to build |
| D. Stream weights to the fastest GPU for all work | One compute site per node | 2.7× slower decode than execute-in-place in example (a) |

## Decision

**Option C.** The solver (ADR-0008) picks among the modes below by cost. These rules bound what it may pick.

1. **Across nodes, PP by contiguous layer ranges is the default.** Stage boundaries fall only at the legal boundaries a family declares (ADR-0003). In latency mode the solver minimises stage count and prefers dropping a weak device to adding a hop.
2. **Cross-node TP is gated.** A set of nodes may form a TP group only if all of these hold:
   - the family has a TP sharding strategy (ADR-0003);
   - for every link in the group, the effective collective latency measured through the real collective path (device buffer → wire → device buffer, at least 200 samples; RFC-0010) has a mean and a p99 below the group's break-even `α_eff,max`;
   - the solver predicts a lower critical-path time with TP than without it.

   TP splits may be uneven, weighted by `W_eff`, where the sharding strategy allows it. In practice this admits RDMA over Thunderbolt 5, RoCE, or ≥25 GbE with low jitter. That list is an expectation; the gate is the measurement. Wi-Fi links never qualify.
3. **Cross-node EP passes the same gate**, costed at 2 messages per MoE layer per token. Every expert has a single owner. Replicas exist only when the solver deliberately replicates a hot expert.
4. **Fast islands.** Nodes that pairwise pass the gate form an island, which the solver may treat as one virtual device with internal TP or EP. PP runs between islands.
5. **Inside a node with several GPUs**, TP over NVLink or PCIe P2P is allowed when the cost model says it pays. P2P is used only after a verified test copy in each direction. The alternatives are a layer split across GPUs through pinned host memory, or disjoint expert ownership.
6. **Inside a device, execute in place.** A tensor group's compute site is the device whose tier holds it: `Vram` → its GPU, host RAM (`HostPageable`, or `HostPinned` for groups that are also streamed) → the NUMA-local CPU, `Unified` → GPU or CPU by microbenchmark. `Nvme` is never a compute site; its bytes are read into a compute tier first (ADR-0007).
7. **Streaming over PCIe is chosen per pass, from measured terms.** A tensor group is streamed to the GPU only when `t_stream < t_EIP` for the pass's token count n. With typical desktop numbers, that means only for prefill chunks and batched or verify passes above `n*`. `n*` is computed from the device's capability profile, never a constant. Unified-memory devices have no PCIe term.
8. **The MoE miss split is a per-layer schedule, not a stream.** For RAM-resident experts that miss the VRAM cache, the executor may copy a share over PCIe for GPU compute while the CPU computes the rest in place. The share minimises `max(GPU, CPU)` using fitted per-layer costs, including the slowdown that DMA reads cause the CPU pool (RFC-0023). Both sites compute the exact expert. Rounding differences fall under the ADR-0019 tolerances.

## Consequences

Positive:
- Single-stream decode runs on the fastest memory that holds each byte, and only pays network latency where a hop moves bytes to faster memory.
- RDMA meshes get TP's measured speed-up without slower links ever paying for it.

Negative:
- Three execution paths to build and test: PP stages, TP/EP collectives, and the in-device three-way MoE layer.
- The gate depends on a collective-path probe, which must run per link and be refreshed as links change.
- Uneven TP needs per-family sharding code beyond the even splits most runtimes use.

Work created: RFC-0011 (gate, islands, n* and the stream/EIP choice in the objective), RFC-0032 (PP stages, TP/EP across nodes), RFC-0012 (executor paths), RFC-0023 (miss split), RFC-0010 (α_eff probe through the collective path), RFC-0031 (collective transport).

## What would make us revisit

- Measured Syntropy TP over a link that passed the gate is slower than PP on the same nodes for the same model. The gate formula is then missing a term.
- A TCP or QUIC path over ≥10 GbE measures `α_eff` below the break-even. The expectation list in Decision item 2 grows; the rule does not change.
- Execute-in-place stops winning at n = 1 on Tier 1 desktops. **Resolved (2026-10-19):** the choice between streaming and execute-in-place uses measured terms (item 7). D1 describes the reference design, not a ban on streaming. If measurements show streaming winning at decode, the solver streams, and D1 needs no change. Background: pinned PCIe 5.0 x16 is estimated at 48–52 GB/s (`hardware-and-performance-model.md` §2.3, no measurement found), while a single-CCD Ryzen's CPU path is put at ≤60 GB/s (§4a). On such machines `n*` approaches 1, and the brief's "stream only above n*" holds only because item 7 uses measured terms. **Settled (2026-10-20):** D1 is a reference design. Wherever measurement shows streaming beating execute-in-place at decode, the solver streams, and the D1 wording needs no further review.

## Revisions

- **2026-10-19, RFC-0001 amendment pass (ADR-0200).** Item 6 names both host tiers: CPU-only groups are homed in `HostPageable`, and `HostPinned` holds only bytes a device copies.
- **Revised 2026-10-20:** item 7 stands without a further review when measurement shows streaming beating execute-in-place at decode. D1 describes a reference design, and the choice is already made from measured terms, so the solver streams wherever they say so; this closes the third revisit item. Two schedule rules now bound cross-node TP and EP without changing their gate. First, if M1b's Thunderbolt 5 raw-path probe measures `α_eff` above 65 µs between two Macs, SC-17 is report-only at the M3 exit and Mac-pair TP moves to M4 or after 1.0. Second, cross-node TP and EP are the third entry in RFC-0001 §16's scope-cut order. No rule in Decision changes.
