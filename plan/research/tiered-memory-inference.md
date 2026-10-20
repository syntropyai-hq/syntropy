# Tiered memory inference: one model across VRAM, RAM and NVMe

Research notes for Syntropy, written 2026-10-19. This file covers every technique we found for running a model larger
than VRAM inside one machine, organised by technique: dense offload, partial offload, CPU execute-in-place, neuron
and expert sparsity, NVMe-resident lookup tables, I/O paths and KV tiering. It ends with recommendations.

Sources are described generically, as [ADR-0000 D5](../decisions/ADR-0000-product-directives.md#d5-the-plan-is-an-original-design)
requires. Three kinds of source were used:
- **A 2026 single-GPU MoE engine.** It runs a 125B MoE model (Qwen3.8-Flash-Next) on one consumer PC, with hot experts
  in VRAM, every expert in pinned RAM and the model's n-gram table on NVMe. Its source, paper and docs were read in
  depth, but nothing was built or run. It is the deepest single source here, and many measurements below come from it.
- **Published offload, expert-caching and KV-tiering papers, 2022-2026.**
- **Docs and source of widely used general-purpose engines and libraries**, including a ggml-based single-node engine
  and a datacenter serving engine.

Evidence tags:
- **[verified: code]**: read in the surveyed system's source.
- **[verified: paper/docs]**: a mechanism stated in a surveyed paper or in project docs.
- **[reported by a surveyed system]** / **[reported in a surveyed paper]**: a measurement made by that system's or
  paper's authors. None of them has been reproduced for this plan.
- **[inferred]**: our own arithmetic or reading, with the working shown.
- **[unverified]**: comes only from a secondary write-up.

Speedups are against each source's own baselines, so **they cannot be compared across rows**. Distribution across
machines is covered in `multi-device-inference.md`. The bandwidth arithmetic is in
`hardware-and-performance-model.md`.

---

## 1. The hardware envelope

At batch 1, decode is limited by bandwidth. To first order, a token takes at least the largest of
`bytes read from tier t / bandwidth of tier t` across the tiers when transfers overlap perfectly. When they don't
overlap, it takes the sum.

| Link or tier | Figure | Basis |
| --- | --- | --- |
| VRAM (RTX 5070 12 GB) | 672 GB/s | **[reported in a surveyed paper]** |
| PCIe 4.0 x16, host → GPU | ~26 GB/s measured; 31.5 GB/s line rate | measured **[reported in a surveyed paper]**; line rate = 16 GT/s × 16 lanes × 128/130 / 8 **[inferred]** |
| DDR5-5200, 2 channels | 52 GB/s for the CPU alone, 41 GB/s while the GPU also DMAs; 83.2 GB/s theoretical | **[reported in a surveyed paper]**; theoretical = 5,200 MT/s × 8 B × 2 **[inferred]** |
| NVMe Gen4 (Samsung 990 PRO) | 7,450 MB/s sequential read | Samsung 9100 PRO brochure, comparison chart ([pdf](https://download.semiconductor.samsung.com/resources/brochure/9100_PRO_Series_Brochure.pdf)) **[verified: docs]** |
| NVMe Gen5 (Samsung 9100 PRO) | 14,700-14,800 MB/s sequential; 1,850K-2,200K random-read IOPS at QD32 | same brochure **[verified: docs]** |
| Apple M3 Max internal SSD | 17.5 GB/s sequential, measured | **[reported by a surveyed system]** |
| **Effective SSD rate for expert-sized random reads** | **≈1.5-2.6 GB/s** (Windows, Gen4); **≈2 GB/s** (M3 Max, through the page cache) | **[inferred]**; working in §11 |
| 1 GbE / 10 GbE / USB4-TB4 | 0.125 / 1.25 / 5 GB/s | line rates **[inferred]** |

**[inferred]** VRAM is about 100× faster than NVMe. For random expert-sized reads the practical gap is closer to 300×.
Every technique below is a way to make the bytes a token reads from the slow tiers small, predictable or amortised.

---

## 2. Taxonomy

| Approach | Unit placed | What moves per token | Latency regime | Needs | Verdict for Syntropy |
| --- | --- | --- | --- | --- | --- |
| **Layer or weight streaming** (§2.1) | Layer, or a fraction of each tensor class | Every non-resident weight, on every forward pass | Seconds per token at batch 1. Useful for throughput only, at batch 16-300 | Large batch; large DRAM or SSD | Worst-case fallback only |
| **Tensor pinning / static partial offload** (§2.2) | Layer, or tensor selected by regex | Activations only; each tensor is computed where it lives | Interactive if the CPU part is small, or the model is MoE | CPU SIMD/AMX; RAM for the CPU part | The baseline and the override layer |
| **Execute-in-place on CPU** (§2.3, §5) | Expert or FFN | Activations to the weights, not weights to the GPU | Interactive; bounded by the CPU's RAM bandwidth | Fast CPU kernels per weight format | Core mechanism |
| **Expert caching and prediction** (§5, §6) | Expert (one per layer and index) | Missed experts' bytes (PCIe or SSD), *or* activations to where the expert lives | Interactive; bounded by hit rate × lower-tier bandwidth | MoE model, replacement policy, next-layer predictor | Core mechanism |
| **Neuron sparsity** (§7) | Neuron or neuron cluster | Predicted-active cold neurons (flash → DRAM) | Interactive on ReLU-sparse models | Activation-sparse models and predictors | Niche plug-in |
| **Flash-resident weights** (§8) | Expert or neuron bundle, laid out by read unit | Missed bytes from SSD | Usable only with MoE, caches above it and prefetch | Read-unit layout, fast SSD | Capacity tier |
| **Deterministic lookup tiering** (§9) | Table row | A few KB of rows, with addresses known in advance | Effectively free once prefetched | n-gram or conditional-memory tables | First use of the SSD tier |
| **KV tiering** (§12) | KV page or block (prefix-addressed) | KV pages on reuse (load) or eviction (store); only hot tokens in sparse variants | Cuts TTFT on reuse; frees VRAM for weights | Paged KV, prefix hashing, layout conversion | Core mechanism, mainly for agents |

### 2.1 Layer and weight streaming (dense offload)

The whole model, or each layer, is fetched to the GPU on every forward pass. This only pays when a large batch
amortises the fetch.

- **A 2023 throughput-oriented offload paper (GPU, CPU and disk)** **[verified: paper/docs]**:
  - Weights and the KV cache are compressed to 4 bits.
  - The CPU can compute attention over a CPU-resident KV cache.
  - Each tensor class gets a fractional placement per tier (§4).
  - A **zig-zag block schedule** walks a block of batches column by column, so one layer's weights serve many
    micro-batches before eviction. The authors prove it within 2× of optimal I/O complexity.
  - Reported: OPT-175B on one T4 16 GB with 208 GB DRAM and a 1.5 TB SSD (about 2 GB/s read) reaches **1 token/s of
    throughput at an effective batch of 144**, and **100× the maximum throughput** of earlier offload systems
    **[reported in a surveyed paper]**.
  - The schedule is throughput-only, with seconds-per-token latency, so it is not worth copying. The cost model and LP
    are (§4).
- **A 2022 datacenter offload engine for inference.** It is built on a training system that used GPU, CPU and NVMe
  memory together, for example to fine-tune trillion-parameter models on one DGX-2 **[verified: paper/docs]**.
  - Weights are pinned in CPU memory or on NVMe and streamed into the GPU **one layer at a time**. Activations stay on
    the GPU.
  - Prefetch gave nothing for CPU offload. For NVMe it gave 1.13× (OPT-30B), 1.14× (OPT-175B) and 1.21× (BLOOM-176B).
  - OPT-30B on one V100-32GB: 43 tok/s with CPU offload and 30 tok/s from NVMe, both at large batch. The authors say it
    does not suit latency-sensitive or small-batch work **[reported by a surveyed system]**.
  - A 2023 refresh added 4-bit weights (about 4× less PCIe volume) and KV offload to the CPU. On one A6000 48 GB with
    252 GB DRAM and NVMe at 5.6 GB/s (prompt 512, generating 32): **Llama-2-70B at 24.05 tok/s (batch 96, CPU, 4-bit)**
    and OPT-175B at 2.26 tok/s (batch 24). The 2023 offload paper's system got 1.84 tok/s on OPT-175B in the same table
    **[reported by a surveyed system]**.
  - Lesson: layer streaming needs batch ≥ 16-96 before throughput is usable. The parts worth keeping are the
    pinned-buffer NVMe engine and the fact that 4-bit weights cut PCIe bytes by 4×.
- **A mainstream Python model-loading library's big-model path** **[verified: docs]**:
  - The model is first built with no storage allocated (a meta device).
  - An automatic device map fills the GPUs, then the CPU, then disk (memory-mapped tensors), **sequentially by
    module**. Per-device memory caps apply, and a list of module classes that are never split keeps residual blocks
    whole.
  - Hooks move each offloaded module's weights to the GPU just before its forward pass and free them after.
  - Stated limits: only one GPU works at a time, there is no prefetch, disk offload is very slow without NVMe, and a
    first layer bigger than the GPU sends everything to CPU or disk.
  - Lesson: a module-level device map is the right *user-facing* abstraction to expose and let users override. Greedy
    sequential packing without prefetch is the baseline to beat.
- **A layer-at-a-time disk loader** **[verified: docs]**:
  - It splits the checkpoint into one file per layer and loads one layer at a time from disk. Next-layer prefetch is
    on by default for one model family (about 10% faster). Optional 4- or 8-bit block-wise compression cuts disk reads
    by up to 3×.
  - It claims 70B on about 4 GB of VRAM and 405B on about 8 GB. **No tokens/s are published.**
  - **[inferred]** A 70B model at 4 bits reads about 35-40 GB per token. On NVMe Gen4 that is about 5 s/token at best
    (40 GB / 7.45 GB/s). At FP16 (~140 GB) it is about 19 s/token. It is a demo, useful only as the worst-case fallback
    ("it runs, slowly") for a single node with too little RAM.
- **A datacenter serving engine's weight offload** **[verified: docs]**. It takes a per-GPU GiB budget to offload to
  the CPU and offers auto, prefetch and UVA backends. The prefetch backend groups layers, offloads N per group and
  prefetches a configurable number of layers ahead. It is a useful comparison as a server-side, throughput-oriented
  layer streamer with explicit prefetch depth, but it is designed for the datacenter.
- **A 2024 MoE offload paper** **[verified: paper abstract]**. It pipelines CPU, GPU and I/O work over paged weights,
  and searches the policy space with a **hierarchical roofline model** (§4). It reports up to 10.3× the throughput of
  earlier offload systems for Mixtral 8x7B on one T4 16 GB. Where GPU memory bounds throughput, it reaches that bound
  with 2-3× less CPU memory **[reported in a surveyed paper]**. The pipeline is batch-oriented.

### 2.2 Tensor pinning and static partial offload

Each tensor is placed once. Then it is either computed where it lives or copied for each use, by a static rule.
A ggml-based single-node engine offers the most complete set of controls **[verified: code]** unless noted:

| Mechanism | What it does | Notes |
| --- | --- | --- |
| **mmap by default** | The GGUF is memory-mapped, so the OS page cache *is* the RAM tier and pages fault in on first touch. Other load modes: no mmap, mlock, mmap+mlock, and direct I/O (`O_DIRECT`). The auto mode avoids mmap on iGPUs, where it doubled peak RAM during load | No `io_uring` anywhere in the tree **[verified: code search]** |
| **N whole layers in VRAM** | The rest run on the CPU from RAM | Layer granularity: the classic partial offload |
| **Regex placement per tensor** | A regex over GGUF tensor names maps tensors to a buffer, for example every `ffn_*_exps` expert tensor to the CPU, so experts stay in RAM while attention and shared tensors go to the GPU | Added in 2025 |
| **MoE and FFN shorthands** | Put all MoE weights in CPU memory, or only those of the first N layers; the same for dense FFNs | Added in 2025 |
| **Auto-fit** | Sets every unset parameter so the model fills device memory | Added late 2025 |
| **LRU GPU cache of host-resident experts** | For batches of 32 tokens or fewer, the MoE matmul over host experts runs on the GPU and only misses are uploaded. Larger batches use the normal offload path. The cache can be split across GPUs | Added 2026-10. Numbers in §5.5 **[verified: docs]** |
| **Rows read lazily from disk** | Per-layer embedding rows are read from disk on demand instead of kept resident (needs mmap). The auto setting does this for tensors over 4 GiB | Added for Qwen3.8-Flash-Next's 97.7 GiB n-gram table (§9) **[verified: code/docs]** |
| **Unified-memory fallback** | On Linux, a build flag lets VRAM exhaustion spill into system RAM instead of crashing. On Windows the driver's System Memory Fallback does the same | The docs say it hurts performance on discrete GPUs **[verified: docs]** |
| **Remote devices** | Remote CUDA, Metal or CPU devices take a share of weights and KV in proportion to their free memory | §16 |

Lessons:
- Regex placement plus the LRU expert cache is the generic baseline a Syntropy node must beat.
- Given the same VRAM, an expert cache beats pinning whole layers by **1.6-2.2×** (§5.5). For MoE, per-expert caching
  beats per-layer placement.

### 2.3 Execute-in-place on the CPU

Weights that miss VRAM are computed where they sit in RAM, and only activations cross PCIe.

- **The founding result (a 2024 paper)** **[verified: paper abstract]**. It uses the CPU's compute to minimise data
  movement between CPU and GPU: cache-missed experts are computed on the CPU instead of being copied over PCIe, and the
  execution strategy is chosen per situation.
  - Reported: uncompressed Mixtral-8x7B (over 90 GB) at **over 3 tok/s on one 24 GB GPU**, which the authors call an
    order of magnitude better than before. The final version reports 1.26× at single batch, 1.30× on long prefill and
    11.57× on beam search over its baselines **[reported in a surveyed paper]**.
- **A 2025 CPU/GPU hybrid MoE system** **[verified: paper/docs]**:
  - **Placement.** Attention (MLA), the KV cache and dense or shared parts run on the GPU. **Routed experts stay in
    DRAM and run on the CPU**, with an option to keep some hot experts on the GPU.
  - **CPU kernels.** Intel AMX kernels run prefill at up to 21.3 TFLOPS on one Xeon socket, 3.9× PyTorch native.
    AVX-512 handles decode.
  - **NUMA-aware tensor parallelism.** Each NUMA node holds its expert slices in local memory, for up to 63% more
    decode on dual-socket machines.
  - **Expert deferral**, an approximation, is covered in §17.
  - Paper totals: prefill 4.62-19.74× and decode 1.25-4.09× over existing systems **[reported in a surveyed paper]**.
  - Project docs, for DeepSeek-V3 q4_K_M on 2× Xeon Gold 6454S with DDR5-4800 (1 TB across both sockets) and an
    RTX 4090D 24 GB: **decode 13.69 tok/s and prefill 97.32 tok/s routing to 6 experts** instead of the model's 8,
    against 4.51 and 10.31 for a generic CPU/GPU engine routing to all 8. A preview with AMX int8 reached 286.55 tok/s
    prefill on a 2K prompt **[reported by a surveyed system]**. Routing to 6 of 8 is itself an approximation.
  - Inside a datacenter serving engine (DeepSeek-V3 int4 on 8× L20), adding GPUs gave about 26% more throughput at
    concurrency 1, but 264% at concurrency 8 **[reported by a surveyed system]**.
  - AMX matters only on recent Xeons. Consumer CPUs top out at AVX2 or AVX-512.
- **The single-GPU MoE engine** computes RAM-resident misses on the CPU, in place in a pinned arena, with
  AVX-512/AVX2 kernels, and copies only a measured share over PCIe (§5).

**[inferred]** When experts sit in DRAM, adding devices helps aggregate throughput far more than single-stream
latency. That holds inside one server, and it should hold across a cluster.

---

## 3. Per-tensor-class placement and byte accounting

### 3.1 A worked byte census

The single-GPU engine only makes sense given its model's byte budget. Qwen3.8-Flash-Next **[verified: paper/docs]**:
- 125B total and about 6B active parameters per token, across 48 layers.
- 36 of the 48 layers use **Gated DeltaNet**, a recurrent layer with fixed-size state. The other 12 use **Qwen Sparse
  Attention (QSA)**, with 2 KV heads, which attends to at most about 2,048 selected positions however long the
  context is.
- Every layer is MoE with **512 experts, top-10**, plus a gated shared expert. That is 24,576 routed experts, and a
  token touches about 2% of the expert bytes.
- Hyper-connections (a gated residual, `hc_count = 4`).
- A **51B-parameter n-gram embedding table**, also called PLE or per-layer embeddings. It is addressed by the last
  three token ids, so the rows are known before they are needed. A ggml-based port puts it at 97.7 GiB unquantized.
- A one-layer MTP (multi-token prediction) head, used for speculative decoding (§13).

Byte census of the 2-bit Q2_0 file **[verified: paper]**:

| Part | Stored | Read per token | Where the engine keeps it |
| --- | ---: | ---: | --- |
| Routed experts (48 × 512) | 34.0 GB | 0.66 GB (2%) | RAM (all of them) plus a VRAM cache of the most-used ~4,500 |
| Mixers, attention, shared experts, routers | 1.8 GB | 1.8 GB | VRAM |
| Gated residual (hyper-connections) | 1.3 GB | 1.3 GB | VRAM |
| Output head | 0.44 GB | 0.44 GB | VRAM |
| N-gram embedding table | 28.8 GB | ≤ 23 KB | SSD plus the OS page cache |
| MTP layer | 0.8 GB | per draft | VRAM |

**[inferred]** One model, three access patterns:
- **Dense, every token.** About 3.5 GB of dense weights are read for every token, so they must live in the fastest tier.
- **Sparse, with skewed popularity.** 34 GB of experts are read at 2% per token, and their popularity is skewed, so
  they can be cached.
- **Deterministic sparse lookups.** 28.8 GB of table rows are read a few KB per token, at addresses known in advance,
  so they can be prefetched.

KV is a fourth class (§12). A general tiering engine has to classify tensors by access pattern, not by size or layer
index, and compute a census like this for each model before it places anything.

### 3.2 Static placement by class

The single-GPU engine **[verified: docs]**:

| Tier | Contents |
| --- | --- |
| **VRAM** | Attention and DeltaNet mixers, gated-residual weights, routers, shared experts, output head, MTP draft layer, KV cache and recurrent state (above 64K context only the hot part, §12), prompt buffers, and an **expert cache that fills whatever VRAM is left** |
| **RAM** | All 24,576 experts in **one pinned arena**. The CPU computes misses there in place, and the GPU DMAs from it |
| **SSD** | The n-gram table (28.8 GB, IQ4_NL), read with unbuffered direct I/O by default. In low-RAM mode, also the experts that don't fit the RAM budget |

- Placement follows **tensor class and is fixed per model**; no optimiser chooses it. Only the VRAM expert-cache size
  and the RAM budget adapt to the machine **[verified: code]**.
- In the default configuration **the SSD tier holds no expert weights**, only the n-gram table, read 16 rows per token.
  Experts reach the SSD only in low-RAM mode or with the larger 4-8-bit quants, and decode there falls to 7-8.5 tok/s
  (§11) **[reported by a surveyed system]**.
- The engine is single-model. Its kernels hard-code this geometry (`H = 2560`, `FF = 640`, a 1,382,400-byte expert
  blob) and refuse any other shape **[verified: code]**. Qwen3.6-35B-A3B runs on a separate, experimental code path, and
  other models fail.

### 3.3 Overrides as the user-facing layer

The module-level device map (§2.1) and regex placement per tensor (§2.2) are the two override forms users already
know. A planner should emit its placement in a form that either one can override.

---

## 4. Cost-model and LP-based policy search

- **A 9-variable LP (the 2023 offload paper)** **[verified: paper]**:
  - Placement is a fraction of each tensor class per tier: `w_g/w_c/w_d` for weights, `h_g/h_c/h_d` for activations and
    `c_g/c_c/c_d` for KV cache.
  - A cost model estimates per-layer prefill and decode latency as the maximum of five overlapped streams:
    CPU→GPU reads, GPU→CPU writes, disk→CPU, CPU→disk, and compute. For example
    `T_gen = max(ctog, gtoc, dtoc, ctod, comp)`. Byte counts come from the hidden sizes, block size and sequence length.
  - Peak-memory constraints apply per tier.
  - The search enumerates a few (GPU batch size, block size) pairs, then **solves an LP over the 9 placement
    fractions**. That is fast because there are only 9 variables.
- **A hierarchical roofline model (the 2024 MoE offload paper)** adds per-device arithmetic intensity, so it models
  mixed CPU/GPU compute better than a pure byte-stream model **[verified: paper abstract]**.
- **Online fitted cost models (the single-GPU engine).** The split of expert misses between the CPU and the PCIe link
  is set from measured rates, either with a closed-form share or a per-layer least-squares fit that minimises
  `max(GPU, CPU)`. Formulas are in §5.4.
- **Priced layer split across GPUs.** The engine places layer ranges using free VRAM and a priced decode window
  (§16). Changing the pricing rule alone moved an unequal GPU pair by +52% prefill and +31% decode.
- **Install-time calibration.** Measuring engine settings on the user's PC takes 15-30 minutes. It made the engine 7%
  faster on one machine **[reported by a surveyed system]**.
- **Greedy baselines.** Sequential packing by module (§2.1) and fill-to-memory auto-fit (§2.2) are what the
  general-purpose tools do.

**[inferred]** The LP extends naturally to Syntropy's problem. It needs more tiers (VRAM, RAM and NVMe on every
device), a network-link term for each device pair, roofline compute terms per device, and constants calibrated at
runtime rather than fitted once.

---

## 5. Adaptive VRAM hot-expert cache over pinned RAM, with CPU execute-in-place for misses

All mechanisms in §5.1-§5.4 are from the single-GPU MoE engine.

### 5.1 Decode pipeline: one layer, three workers

From the paper **[verified: paper]**, matching code comments **[verified: code]**:
1. The GPU runs the layer's mixer and router, then writes the 10 chosen expert ids into a small pinned host buffer
   that acts as a **doorbell**.
2. A CPU thread **spins on the doorbell** instead of waiting on the driver, and splits the experts into three groups:
   - **hits**, already in the VRAM cache, which the GPU computes;
   - **a share of the misses copied over PCIe** by the GPU's copy engine, which the GPU computes once they land;
   - **the remaining misses, computed by the CPU in place** in the pinned RAM arena with AVX-512/AVX2 kernels.
3. The CPU writes its results into mapped host memory. The GPU sums the partial outputs, and the next layer starts.
4. The whole 48-layer pass is **one captured CUDA graph per verify-window size**, so the host makes no synchronising
   driver call on the token path.

The spin exists because a driver sleep/wake round trip costs about 0.3 ms on Windows, which is longer than a small
expert batch takes **[verified: code]**.

One verify window at 4K context on an RTX 5070 **[reported in a surveyed paper]**:

| Pack | Round (ms) | GPU wait | CPU experts | Drafting | Tokens/round | VRAM hit rate | CPU GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2_0 | 34.3 | 14.6 | 13.4 | 2.0 | 3.23 | 0.72 | 41 |
| IQ2_XS | 42.0 | 14.7 | 21.1 | 2.2 | 3.28 | 0.78 | 20 |
| IQ3_XXS | 49.4 | 15.3 | 27.1 | 2.1 | 3.24 | 0.71 | 24 |

The GPU (reading 3.5 GB of dense weights) and the CPU (reading missed experts) each take about half of every round.
The authors conclude that speeding up only one side cannot help by more than about a third.

### 5.2 Sizing the cache

- **Sized from what is actually free.** The cache gets the VRAM left after the dense arena, KV, MTP and prompt
  buffers. The engine checks the allocation against what the card really has instead of trusting its planner
  **[verified: code]**.
- **Slots sized in bytes per layer, not uniformly.** IQ3_XXS experts range from 1.3 to 2.3 MB. Sizing each layer's
  slots for that layer's experts fits 3,556 experts instead of 2,673, for +13% decode at 1K
  **[reported in a surveyed paper]**.
- **Admission per layer.** Each layer owns its own slot range. With one shared arrival-order counter, the first
  distinct (layer, expert) pairs filled the whole cache within the first ~26 layers at position 0. The measured hit
  rate was 2.97%, against 70.4% for 64 slots per layer in an offline replay **[verified: code comment; reported by a
  surveyed system]**.
- **VRAM is expert capacity.** Each extra GB of VRAM holds about 700 more experts. Long context costs VRAM before it
  costs compute: at 262K the KV cache leaves room for about 1,600 experts instead of 4,500, and Q2_0 decode falls from
  about 95 to about 56 tok/s **[reported in a surveyed paper]**. §12 shows KV streaming recovering part of this.

### 5.3 Fill and adaptation: an offline profile plus decayed LFU

**Initial fill.** A shipped profile ranks every (layer, expert) pair by how often a corpus routes to it.
- Ten-fold leave-one-out hit rate at 4,105 slots: **0.6447 [0.6110, 0.6750]**. In-sample was 0.6573, and filling by
  compulsory misses scored only 0.4864 **[verified: code comment]**.
- The paper reports 0.50 held-out at about 4,500 slots, on a different corpus split **[reported in a surveyed paper]**.

**Adaptive tier** **[verified: code]**:
- Every 4 rounds, up to 96 experts are swapped: the most-routed missing experts replace the least-routed resident ones.
- After each adaptation, routing counts are multiplied by 0.7. This is **LFU with exponential decay**.
- An expert is swapped in only if its decayed count beats the victim's by a factor of 1.5. This **hysteresis** prevents
  thrashing.
- Result: **hit rate about 0.72, against 0.50 for the static profile**, on a 12 GB card at 4K
  **[reported in a surveyed paper]**.

**Swaps never stall the token path.** The old expert is evicted at once, and the new one is admitted only when its copy
has landed. That took Q2_0 from 91.7 to 94.4 tok/s **[reported in a surveyed paper]**. Moving the cache update to a
helper thread added 4-5% decode on an RTX 5070 **[reported by a surveyed system]**.

**The learned ranking is saved.** It is written out with resident experts first, then by decayed heat, then by the
prior. The next start begins where the last one ended **[verified: code]**.

### 5.4 Splitting misses between the PCIe link and the CPU

- The PCIe share defaults to 0.2 of misses for the repacked Q2_0 pack and 0.55 for native i-quant packs
  **[verified: code]**.
- Why they differ: the Q2_0 CPU kernel already saturates RAM bandwidth, so DMA would steal bandwidth from it. The
  i-quant CPU kernels are bound by arithmetic and leave RAM bandwidth spare. Sending 55% of i-quant misses to the copy
  engine took IQ2_XS from 70 to 78 tok/s and IQ3_XXS from 56 to 65 **[reported in a surveyed paper]**.
- **Online refinement** (single CUDA GPU, no explicit share set) **[verified: docs]**:
  - Let R be expert bytes divided by the CPU pool's measured time per missed expert, and L the probed link speed.
  - Share = `L / (L + 1.85 R)`, rounded to 0.05 and clamped to 0.10-0.60.
  - The share changes at most three times, within the first ~10 s of decoding.
  - The 1.85 is a fit to three machines.
- **Opt-in balancer** **[verified: code]**. It fits each layer's costs by least squares with forgetting:
  - `CPU = a + c·n_cpu + d·n_pcie`
  - `GPU = g0 + g·n_pcie`
  - *d* captures the PCIe DMA's DRAM reads slowing the CPU pool.
  - It picks the split that minimises `max(GPU, CPU)`.

### 5.5 Other replacement and caching policies

| Policy | Mechanism | Reported (their conditions) |
| --- | --- | --- |
| **Per-layer LRU of k experts** (2023 paper) | k=2 on 12 GB, k=4 on 16 GB. 4-bit attention, 2-3-bit experts. Paired with next-layer speculative loading (§6) | Mixtral-8x7B, 2-bit experts: A100 3.06, RTX 3080 Mobile 2.66, RTX 3060 2.28, T4 2.09 tok/s. The 3060 nearly matches the 3080M because both are PCIe-bound **[reported in a surveyed paper]** |
| **Trace-guided cache** (2024 paper) | Traces expert activation; a representative trace guides **both replacement and prefetch**. Aimed at personal machines at batch 1, where decode reuses a few experts heavily | 3.1-16.7× per-token latency over general-purpose engines (DeepSeek and Mixtral) **[reported in a surveyed paper]** |
| **Mixed-precision misses** (2024 paper, built on ggml) | A less critical missed expert is loaded as a *low-precision copy*. Token-level dynamic loading, layer-level adaptive prefetch, sequence-level multi-dimensional caching | Up to 9.93× decode over other MoE offload systems on edge devices **[reported in a surveyed paper]**. An approximation (§17) |
| **Learned recency/frequency** (2026-01 paper, SSD-offloaded experts) | An ML replacement policy blends recency and frequency, because plain LRU and LFU fit routing poorly | Hit rate up to 51% above LRU/LFU; up to 2.6× on a user-grade desktop **[reported in a surveyed paper]** |
| **Forecast-driven Belady** (2026-09 paper) | Probabilistic Belady eviction from multi-step expert forecasts (§6); designed for graph capture | **96.97% hit rate at 45% expert residency, 80.22% of full-load performance** **[reported in a surveyed paper]** |
| **Nonuniform per-layer budgets** (2026-10 paper, NPU appliance) | Cache capacity differs per layer; domain-informed initial fill; replacement aware of routing history; cross-layer prefetch; optional approximate substitution for low-score misses | 84 GB cap, DeepSeek-V4-Flash W4A8: TPOT 38.0-44.8 ms against 1,269 ms for a datacenter engine's prefetch offloader, **but with different output caps (1K vs 128)** **[reported in a surveyed paper]** |
| **Decayed LFU with hysteresis** (single-GPU engine) | §5.3 | 0.50 → 0.72 hit rate |
| **Generic LRU** (ggml-based engine, 2026-10) | §2.2 | Table below |

Generic LRU expert cache against pinning whole layers in the same VRAM. Qwen3.8-Flash-Next Q4_0 (65.4 GiB of experts),
a public speed benchmark, 512 output tokens, no MTP, EPYC 7742 with 16 cores on the GPU's NUMA node, PCIe 4.0 x16
**[reported by a surveyed system]**:

| GPU | Config | Experts in VRAM | Decode tok/s | Speed-up | Hit rate |
| --- | --- | --- | ---: | ---: | ---: |
| RTX 4090 | auto-fit, whole layers | 16.5 GB as whole layers | 25.0 | 1.00× | - |
| RTX 4090 | experts in RAM + LRU cache | 10.7 GB cache | 40.7 | 1.62× | 77% |
| RTX 5090 | auto-fit, whole layers | 24.5 GB as whole layers | 30.8 | 1.00× | - |
| RTX 5090 | experts in RAM + LRU cache | 18.6 GB cache | 67.8 | 2.20× | 89% |

Its authors suggest sizing the cache at about 10% of total expert bytes.

**[inferred]** Takeaways:
- The replacement policy is worth tens of points of hit rate. Plain LRU is the floor; decayed LFU, learned policies and
  forecast-driven Belady sit above it.
- Since the generic LRU cache appeared, the single-GPU engine's remaining lead comes mostly from speculative decoding,
  the adaptive policy, its CPU kernels and the doorbell pipeline. The tiering idea itself is no longer the
  differentiator.

---

## 6. Expert prediction and prefetch

**Prediction is reliable one layer ahead.** Every successful system uses one of three approaches: the next layer's
router applied to the current hidden state, a trained pre-gate, or a learned predictor.

- **The next layer's router, run early.** The residual stream changes little from one layer to the next, so layer
  l+1's router applied to layer l's hidden state predicts layer l+1's experts well.
  - A 2023 paper applies the next layer's gate to the current hidden state and fetches its top 1-2 experts into the
    per-layer LRU (§5.5) **[verified: paper]**.
  - In the single-GPU engine, for SSD-resident experts, a worker thread applies **layer l+1's router (a BF16 host copy)
    to layer l's MoE input** while the CPU pool runs layer l **[verified: code]**:
    - it takes the top-k, drops whatever the GPU or RAM already holds, and warms the rest's pages;
    - it only warms pages and never changes what is computed;
    - depth is configurable from 1 to 4 layers;
    - about half of the SSD reads were predicted, for +14% (6.9 → 7.9 tok/s) **[reported by a surveyed system]**;
    - the price is a host copy of every router, 126 MB for this model.
  - The same engine can also copy predicted experts that no cache holds into VRAM ahead of the layer **[verified:
    code]**. The RAM → VRAM path does not use the router lookahead; the paper proposed it as the next step.
- **Pre-gating (trained).** A 2024 paper has block N's pre-gate choose block N+1's experts, so their fetch overlaps
  block N's compute. The lookahead is exact by construction, but the pre-gate **must be fine-tuned** (2,048 steps from
  pretrained weights) **[verified: paper]**. Switch Transformer on an A100 80 GB: about 1.5× throughput over on-demand
  fetching, peak GPU memory about 23% of GPU-only, and about 81% of GPU-only throughput **[reported in a surveyed
  paper]**.
- **A pre-attention router (model design).** A 2025 family of models built for local devices routes *before*
  attention, so the engine knows the expert ids early enough to load them from storage during attention. The models
  also combine fine-grained MoE with sparse FFNs, and use a NoPE/RoPE hybrid to shrink the KV cache. The 4B-A0.6B and
  21B-A3B models both exceed 20 tok/s on consumer CPUs at Q4_0, in about 1 GB and 8 GB **[reported in a surveyed
  paper]**. Models with an early router turn prediction into certainty.
- **Learned predictors.** A 2024 paper uses intermediate results to fetch experts several layers ahead, removing
  passive cache misses from the critical path: prefill 2.20× on average (3.21× max) and decode 2.07× on average (5.02×
  max) over offload baselines **[reported in a surveyed paper]**.
- **Multi-step forecasts.** A 2026-09 paper forecasts experts sequence-to-sequence across future steps and layers. It
  schedules prefetch as a job-sequencing-with-deadlines problem and evicts by probabilistic Belady (§5.5)
  **[verified: paper abstract]**.
- **Trace-guided prefetch.** The 2024 trace-guided cache (§5.5) drives prefetch from the same activation traces.
- **Prefill is a different regime** **[verified: docs]**:
  - Prompts run in chunks of up to 8,192 tokens (32,768 opt-in). Uncached experts stream from the pinned arena over
    PCIe while the GPU computes the previous ones.
  - Staging buffers are lent from the tail of the expert cache and refilled afterwards.
  - The ring is sized in bytes, and the auto setting picks the largest chunk that keeps it full: IQ3_XXS gained +18.5%
    at 32K **[reported by a surveyed system]**.
  - A 2026-10 paper stages whole expert layers through alternating buffers during prefill **[verified: paper
    abstract]**.
- **Negative results:**
  - An Apple Silicon engine that relies on the OS page cache tried custom caches, prefetch and speculative routing.
    All three were *slower*, and were dropped (§8) **[reported by a surveyed system]**.
  - Splitting each verify window into two token groups, so one group's CPU experts overlap the other's GPU work, was
    exact but **7% slower**. Dense weights were read twice, and fewer experts were shared within each group
    **[reported in a surveyed paper]**.

**[inferred]** Prefetch targets for a tiered, distributed engine: SSD → RAM pages, RAM → VRAM slots, and peer → local
where a peer holds the weights. Syntropy's scheduler should treat "routing known at time t" as a model capability
flag. Pre-gates, pre-attention routers and deterministic lookup tables (§9) all belong to that family.

---

## 7. Hot/cold neuron partitioning

Here the unit placed is a single FFN row or column, chosen by how often it activates, rather than a whole expert.

- **A 2024 hot/cold neuron paper** **[verified: paper abstract]**:
  - Activations follow a power law. **Hot neurons are preloaded on the GPU and cold neurons are computed on the CPU.**
  - Adaptive predictors guess which neurons will fire, and neuron-aware sparse operators skip the rest.
  - Reported: up to 11.69× faster than a generic CPU/GPU engine on one RTX 4090, with OPT-30B at 82% of an A100's
    generation rate **[reported in a surveyed paper]**.
  - It needs **ReLU-sparse models**: OPT, ReLU-fied Llama and the authors' own sparse models. Modern SwiGLU MoE models
    are not activation-sparse at the neuron level without retraining **[inferred from the model lists]**.
- **A 2024 mobile follow-up** **[verified: paper abstract]**:
  - "Neuron clusters" are the unit of scheduling and I/O. Dense clusters go to the NPU and sparse ones to the CPU.
  - A storage engine pipelines cluster compute with flash I/O, behind a segmented neuron cache.
  - Reported: a 47B model on a smartphone at **11.68 tok/s**, up to 27.8× over existing frameworks **[reported in a
    surveyed paper]**.
  - This is the right shape for a phone as a Syntropy node: an NPU/CPU split with a flash I/O pipeline. It still needs
    sparse-activation models.

**[inferred]** For Syntropy, expert-level sparsity is the mainstream equivalent. Neuron-level placement is a plug-in
for models that support it.

---

## 8. Flash-resident weights

- **A 2023-24 mobile-focused study** keeps parameters in flash and brings them into DRAM on demand
  **[verified: paper abstract]**. A flash cost model drives two techniques, and a sparsity predictor decides what to
  load:
  - **Windowing.** Neurons activated for the last few tokens stay resident, and only the difference is loaded. This is
    a recency cache.
  - **Row-column bundling.** A neuron's up-projection row and down-projection column are stored together, so one read
    fetches both and reads get larger.
  - Reported: models up to 2× the size of DRAM, 4-5× faster than naive loading on the CPU and 20-25× on the GPU
    **[reported in a surveyed paper]**.
- **Bundling generalises to "lay the file out in the unit you read".** The MoE versions:
  - The single-GPU engine on Apple Silicon can write a repacked expert file with each expert's three matrices
    contiguous, so one SSD request fetches one expert. On an M4 Mac mini 32 GB this took decode from 7.37 to 8.62 tok/s
    (+17%) and prefill from 64.7 to 84.1 tok/s on a 6K prompt **[reported by a surveyed system]**.
  - The Apple Silicon page-cache engine below issues one `pread` per expert.
- **An Apple Silicon engine that relies on the OS page cache (2026)** runs Qwen3.5-397B-A17B in pure C and Metal
  **[verified: docs]**:
  - Experts (about 6.75 MB each; 4 of 512 active per layer) are read on demand with parallel `pread`.
  - It has **no custom cache**. The page cache (about 35 GB) gives a 71% hit rate. The non-expert weights (5.5 GB) are
    mmap'd.
  - M3 Max 48 GB, SSD at 17.5 GB/s: **4.36 tok/s at 4-bit**. At 2-bit it reaches 5.74 tok/s, but that breaks JSON
    output and tool calls **[reported by a surveyed system]**.
  - Its custom caches, prefetching and speculative routing were all slower than the OS, and were dropped.
- **The single-GPU engine's Metal path** **[verified: docs]**:
  - When the experts exceed 60% of Metal's share of memory, experts stream from SSD into an anonymous arena.
  - The arena is **wired with `mlock`**, and on Macs with 64 GB or more the whole anonymous footprint is capped at 60%
    of RAM. The cap came after **three kernel panics** in which the memory compressor consumed an unwired arena. A
    resident pack of clean file pages behaves differently: macOS drops those pages rather than compressing them.
  - The per-layer handshake (read the router's choice, make sure the experts are resident) costs more than the SSD
    reads themselves. A 2-token window takes 38 ms streaming against about 17 ms resident. Speculation with 3 drafts is
    therefore the default for streaming on Macs with 96 GB or more.
- **A 2026-09 SSD-native MoE paper** builds an I/O pipeline for expert delivery across three tiers (SSD → RAM →
  VRAM), balances CPU and GPU execution, and runs every selected expert, with no pruning. It reports a
  **trillion-parameter MoE at over 1 tok/s on one RTX 5090 with ≤32 GB RAM**, and decode 2.10-15.58× over unnamed
  baselines **[reported in a surveyed paper]**.
- **Energy is the hidden cost.** A 2025 paper estimates that serving experts from SSD raises decode energy per token by
  up to about 12× over HBM **[verified: paper abstract]**. That matters for laptops and phones as nodes.

Windows SSD-streaming numbers and the effective-bandwidth arithmetic are in §11.

---

## 9. NVMe-resident lookup tables

- **Conditional memory (a 2026-01 model-architecture paper)** **[verified: paper abstract]**. It adds a second sparsity
  axis next to MoE: modernised n-gram embeddings with O(1) lookup. **Deterministic addressing lets the runtime prefetch
  rows from host memory with negligible overhead.** The authors scale it to 27B parameters, with gains over MoE
  baselines at equal parameters and equal FLOPs.
- **Qwen3.8-Flash-Next** ships a 51B-parameter n-gram table addressed by the last three token ids (§3.1).
  - The single-GPU engine keeps it **on SSD** (28.8 GB, IQ4_NL). Each token needs 16 direct-I/O row reads, at most
    about 23 KB, and **they are issued one step ahead** of the layer that uses them, since the addresses come from
    tokens already emitted **[verified: paper]**.
  - The ggml-based engine reads such rows from disk on demand, through mmap, for any tensor over 4 GiB
    **[verified: code/docs]**.
- **Reader options and probes** **[verified: docs]**:
  - The default is direct I/O through a thread pool (§10.3). The alternatives are mmap through the page cache, and
    holding the whole table in RAM (for rotational disks).
  - A startup probe warns when direct reads come in under 15K rows/s.
  - SATA SSDs top out at 10-14K rows/s with 16 reads in flight.
  - On one SATA machine, mmap managed 2.1K rows/s against 14.4K for direct I/O **[reported by a surveyed system]**.
  - On Windows a slow mapped file stalled prompts.
- **[inferred]** Even the 15K rows/s warning threshold supports about 940 tok/s of row reads (15,000 / 16), so the
  table never limits decode when it is prefetched.

**[inferred]** This is the **ideal SSD-tier workload**: large, cold, a few KB per token, with addresses known in
advance. The planner should detect tensors that look like embeddings, addressed by row from token ids, and send them
to SSD (or a peer's RAM) with prefetch. That frees RAM for experts.

---

## 10. I/O paths

### 10.1 RAM tier: pinned anonymous arena, not mmap

This was the single-GPU engine's most consequential I/O decision **[verified: code comment; reported by a surveyed
system]**:
- Mapping both the 34 GB expert file and the 26.8 GB n-gram file put 60.8 GB of file-backed pages on a 63 GB machine.
- The OS drops file-backed pages first, so expert reads faulted back in from disk.
- On Windows the CPU pool read mapped memory at **about 19 GB/s, against 42.8 GB/s from an anonymous heap arena**: a
  2.2× difference.
- The fix: read the expert file once at startup into anonymous memory, then register that memory with CUDA
  (`cudaHostRegister` or `cudaHostAlloc`) so the GPU can DMA from it. Registration is best-effort, and its result is
  reported.

### 10.2 Platform limits on pinned memory

All **[verified: code/docs]**:
- **Windows.**
  - Page-locking 34-43 GB as one range fails. The engine locks per-layer ranges plus a working-set lock
    (`SetProcessWorkingSetSize`, `VirtualLock`), and a PCIe read of a missed expert must never span two ranges.
  - Windows can lock only about half of RAM for the GPU. Pinning first the experts the GPU cache will *not* hold gave
    +4.4% prefill **[reported by a surveyed system]**.
  - Under WDDM, host-arena registration for helper GPUs is capped at 8 GiB.
- **WSL2.**
  - The driver caps `cudaHostAlloc` at about 1 GiB. The engine detects `/dev/dxg` and uses `cudaHostRegister` on
    ordinary memory instead.
  - It never streams KV under WSL.
- **Linux.**
  - One `cudaHostAlloc` covering every expert could get the process OOM-killed even with high MemAvailable, because
    clean file cache cannot be reclaimed as fast as the driver takes pages.
  - Registration is now paced in steps, checked against MemFree, cgroup v1/v2 limits and free memory on each NUMA node.
- **macOS.**
  - The arena is wired with `mlock`, with a 60% footprint cap after the compressor panics (§8).
- **First start.**
  - Loading 35-55 GB into RAM and locking part of it makes the PC slow or unresponsive for 1-3 minutes.

### 10.3 SSD reads

- **Direct I/O thread pool**, the engine's default for the n-gram table **[verified: code]**:
  - Linux: `O_DIRECT`, with a pool of 16 threads each doing a blocking `pread`, so the thread count is the queue depth.
  - Windows: `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` on an I/O completion port, with 4 submitting threads.
  - The thread count can be set from 1 to 64. Each in-flight read gets an aligned slab two pages long.
  - Why a pool: one thread issuing every read peaked at about 80K reads/s (~12 µs of kernel time per read), and an
    NVMe drive serves several times that at depth.
- **SSD-resident experts through the page cache** **[verified: code/docs]**:
  - Experts are read from the GGUF in place, through the OS file cache.
  - Linux: `madvise` (`MADV_WILLNEED`, `MADV_RANDOM`, `MADV_HUGEPAGE`) and `posix_fadvise`. Windows:
    `PrefetchVirtualMemory`.
  - The prompt path uses 32 threads, with up to 128 expert blobs in flight.

### 10.4 Mechanisms compared

| Mechanism | Seen in | Verdict for consumer machines |
| --- | --- | --- |
| **mmap + OS page cache** | The ggml-based engine (default), the Python library's disk offload, the Apple page-cache engine, the single-GPU engine's fallback | Zero-copy and simple, but the OS evicts file-backed pages first. Measured 19 vs 42.8 GB/s against an anonymous arena (Windows) and 2.1K vs 14.4K rows/s against direct I/O (SATA). Fine for cold, read-once data; risky for a hot tier; the best option on Apple's SSD **[reported by surveyed systems]** |
| **Pinned anonymous arena** (`cudaHostAlloc` / `cudaHostRegister`) | The single-GPU engine, the 2022 datacenter offload engine, a vendor KV block manager's CPU tier | Needed for DMA and for steady CPU bandwidth. The per-platform limits are hard (§10.2) **[verified: docs]** |
| **`O_DIRECT` / `FILE_FLAG_NO_BUFFERING` with a thread pool** | The single-GPU engine's table reader; the ggml-based engine's direct-I/O load mode | Predictable, keeps the cache clean, and reaches queue depth with plain threads. One thread peaks at about 80K reads/s **[verified: code]** |
| **io_uring** | None of the engines read at source level **[verified: code search]** | Linux-only. The obvious way to reach queue depth without 16-64 threads. We found no published numbers for this workload |
| **GPUDirect Storage (cuFile)** | A vendor KV block manager's NVMe tier (optional); datacenter loaders | **Linux-only, and only Tesla/Quadro-class GPUs.** Elsewhere, compat mode falls back to POSIX `pread` ([NVIDIA GDS docs](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html)). Not a consumer path **[verified: docs]** |
| **GPU-initiated storage** (2023 research) | Research only | GPU threads issue storage requests through high-throughput queues, with a software cache that coalesces them. Needs custom NVMe drivers, so users can't install it **[verified: paper abstract; inferred applicability]** |
| **DirectStorage (Windows)** | Games | An API for loading game assets from fast NVMe drives at full speed, built around D3D12 resources ([repo](https://github.com/microsoft/DirectStorage)). No surveyed inference engine uses it **[verified: docs; inferred applicability]** |
| **On-disk layout in read units** | Row-column bundling (§8); the repacked per-expert file (+17% decode); one `pread` per expert; the SSD-native MoE paper | The cheapest win available: **one aligned request per expert** **[reported by surveyed systems]** |

### 10.5 GPU memory management

- **Elastic VRAM** (opt-in, NVIDIA, single GPU) **[verified: code/docs]**:
  - The expert cache is one reserved virtual range, backed by 512 MiB physical segments allocated with the CUDA VMM
    API (`cuMemCreate`).
  - An HTTP endpoint unmaps tail segments to give VRAM to another program, or maps them back.
  - Slot addresses stay fixed, so captured CUDA graphs remain valid.
- **Batched DMA** (experimental). A verify group's expert uploads are submitted through CUDA 13's
  `cudaMemcpyBatchAsync`. It is presented as an experiment in host overhead, with no claimed speedup
  **[verified: docs]**.
- **Unified-memory systems.** On DGX Spark (GB10), `cudaMemGetInfo` undercounts, so free memory is taken as
  MemAvailable minus a 6 GiB headroom **[verified: code]**.
- **Unified-memory fallback on discrete GPUs** spills to system RAM instead of crashing, at a performance cost (§2.2).
- **Datacenter layout conversion** (§12.2). GPU-assisted I/O kernels, run on a few SMs, gather small KV pages at up to
  3× the speed of `cudaMemcpyAsync`. Using a layer-first GPU layout with a page-first host layout gives up to 2× with
  zero-copy.

---

## 11. Consumer NVMe: bandwidth and IOPS in practice

**Measured SSD streaming on Windows** (4-bit dynamic quant, RTX 5070, 64 GB, RAM budget 40 GiB) **[reported by a
surveyed system]**:
- Decode 7-8.5 tok/s, mean 7.9 over 6 runs. About 0.33 GB per token comes from SSD.
- Each verify round of about 3.5 tokens spends 550-750 ms waiting on SSD, about 75 ms in CPU expert kernels and about
  13 ms on the GPU.
- With a 24 GiB budget, before routing prefetch (§6): 5.1-5.2 tok/s.
- 16K prefill: 160 tok/s.
- **Effective SSD throughput is about 1.5-2.6 GB/s** either way it is computed **[inferred]**:
  - from the stall breakdown: 1.1 GB per round / 0.55-0.75 s = 1.5-2.0 GB/s;
  - from the decode rate: 7.9 tok/s × 0.33 GB/token ≈ 2.6 GB/s.

  The two figures probably come from different runs, since they don't quite reconcile. Either way, the rate is far
  below a Gen4 drive's ~7 GB/s sequential rating. These are megabyte-sized random reads through the page cache, on
  Windows.
- **RAM capacity is the gate.** Once the experts no longer all fit in RAM, decode drops roughly 6-8× (53 → 7-8 tok/s).
  The engine's installer refuses or downgrades model sizes accordingly.

**Apple page-cache engine, effective rate [inferred]:** 4 experts × 6.75 MB × 60 layers × (1 − 0.71 page-cache hit) ×
4.36 tok/s ≈ 2.0 GB/s, against 17.5 GB/s sequential.

**IOPS.** A Gen5 drive is rated at 1.85-2.2M random-read IOPS at QD32. Megabyte-sized expert reads are bound by
bandwidth, not IOPS. Table-row reads are bound by submission cost: one thread caps at about 80K reads/s, and a SATA SSD
manages 10-14K rows/s at 16 in flight (§10.3, §9).

What SSD tiers can deliver per workload. All figures are **[inferred]** from the ~2 GB/s effective random-read rate,
with Gen4 sequential (7.45 GB/s) and Gen5 sequential (14.7 GB/s) as the best case:

| Workload | Bytes from SSD per token | Time per token from SSD alone | Viable? |
| --- | ---: | --- | --- |
| Dense 70B at 4-bit, layer streaming | ~35-40 GB | 2.4-5.4 s best case (Gen5 to Gen4 sequential); ~20 s at 2 GB/s | **No** for interactive use; batch only |
| Qwen3.8-Flash-Next Q2_0, every expert from SSD (no RAM tier) | 0.66 GB | ~330 ms at 2 GB/s, so ≤3 tok/s before speculation | Marginal |
| Same model, RAM holding 2/3 of the experts (40 GiB budget, 4-bit) | ~0.33 GB (measured) | ~165 ms at 2 GB/s; measured 7-8.5 tok/s *with* MTP | **Usable but slow**: about 15% of the speed with every expert in RAM (53 tok/s) |
| Qwen3.5-397B-A17B 4-bit on an M3 Max | ~0.47 GB after a 71% page-cache hit | ~27 ms at its 17.5 GB/s peak; measured 4.36 tok/s overall | Usable on Apple's fast SSD; serial compute dominates |
| N-gram or conditional-memory rows | ≤23 KB (16 rows) | ~16 random reads, hidden by prefetching one step ahead | **Yes.** The ideal SSD workload |
| Parked KV for a 63K-token chat | 1.2 GB per restore, once | 0.9 s, against ~30 s to re-prefill | **Yes.** Pays off at the first reuse |
| Model load / cold start | Whole model, once | 70 GB / 7 GB/s ≈ 10 s best case; a 1-3 minute first start is reported, including pinning | Yes; this is load time, not per-token time |

Conclusions:
1. For weights read on every token, consumer NVMe is a **capacity tier**, not a bandwidth tier.
2. It is viable for weights only when (a) the model is MoE, (b) the RAM and VRAM caches above it absorb most accesses,
   and (c) predicted misses are prefetched and read in contiguous, expert-sized units.
3. It is clearly good for deterministic lookups, parked KV and cold start.
4. Energy per token rises sharply when weights come from SSD (§8).

---

## 12. KV cache tiering, prefix caches and saved sessions

### 12.1 On one consumer machine

From the single-GPU engine **[verified: docs, paper]**:
- **Where KV lives.** The KV cache and the DeltaNet recurrent state are in VRAM. Only the 12 QSA layers have a KV cache
  (2 KV heads); the 36 DeltaNet layers keep a fixed-size state.
- **KV formats** **[reported by a surveyed system]**:
  - 8-bit by default above 4-8K context.
  - Opt-in 4-bit with a Hadamard rotation before rounding: half the memory, perplexity +8-12% on long documents.
  - 8-bit K with rotated 4-bit V: 23% less memory than 8-bit. On an RTX 3090 at 198K context, a code-specialised variant
    decoded 99 tok/s against 85.
- **KV streaming at 64K and above.** The full KV lives in pinned RAM, and the 32,768 most-read cells stay in VRAM.
  Attention reads identical values; only their location changes.
  - Q2_0 at 262K: **50.9 → 62.6 tok/s**, because the VRAM this frees holds 3,872 experts instead of 1,589
    **[reported by a surveyed system]**.
  - Cost: about 13.7 KB of RAM per context token.
  - Overlapping the next attention layer's KV upload with compute (opt-in) cut prefill time by 5.3% at 32K, with no
    change to decode.
  - **[inferred]** KV tiering here is a lever for weight caching, because KV and experts compete for the same VRAM.
    Sparse attention (QSA reads at most ~2,048 positions) is what makes it cheap: only the selected blocks need to be in
    VRAM.
- **Conversation cache.** The server keeps conversations and their checkpoints, and reuses them when the token, image
  and steering-mode identity match **[verified: code]**.
- **Session files on disk.** These use an existing slot save/restore format. 63,025 tokens of state takes 1.2 GB on
  disk, saves in 710 ms and restores in 898 ms, against about 30 s to re-read the chat. Each file is bound to an
  identity hash of the model files and of every setting that changes meaning, including the resolved RoPE
  configuration **[verified: code/docs]**.
- **Gaps.** Parallel slots share neither KV nor prefix. Agent clients trigger full prompt re-reads, which is the
  largest open pain point among its users.
- The ggml-based engine also has a slot save/restore API **[verified: code]**.

### 12.2 Datacenter techniques worth taking

- **A pluggable connector between the engine and a KV block store.** A datacenter serving engine exposes nine connector
  types, including CPU offload with a byte budget and connectors chained together **[verified: docs]**. It is the
  right seam: the KV store sits behind the engine and moves blocks between GPU, CPU, disk and network.
- **A KV cache layer across GPU, CPU, storage and network (a 2025 paper)** **[verified: paper abstract]**. It provides
  prefix reuse across queries and engines, prefill/decode disaggregation, batched movement pipelined with compute, and
  a control API. Reported: up to 15× throughput on multi-round QA and document analysis. Context truncation can halve
  prefix-cache hit rates **[reported in a surveyed paper]**.
- **A disaggregated KV pool in idle CPU DRAM and SSD (a 2024 paper)** **[verified: paper abstract]**. Prefill and decode
  run on separate clusters, with the KV cache held in the GPU servers' otherwise idle CPU, DRAM and SSD. It adds a
  KV-cache-centric scheduler and early rejection under overload. Reported: up to 525% more throughput in simulation and
  75% more requests handled on a production chat service's real workloads **[reported in a surveyed paper]**. The
  transfer engine is datacenter RDMA, but the idea of pooling other machines' spare DRAM and SSD maps directly onto a
  cluster of consumer devices.
- **A radix tree used as a page table across HBM, host DRAM and storage (2025)** **[verified: docs]**:
  - Storage backends are pluggable and implement `get`, `exist` and `set`.
  - Write policies: write-through, write-through-selective (only entries hot by hit count) and write-back.
  - Prefetch from storage into host memory runs in best-effort, cancel-to-protect-TTFT or wait mode.
  - **The GPU pool is layer-first and the host pool is page-first**, to make transfers large: up to 2× with zero-copy.
  - GPU-assisted I/O kernels give up to 3× over `cudaMemcpyAsync`.
  - Reported: up to 6× throughput and 80% lower TTFT. Production users reported average TTFT −56% on an agentic coding
    workload with 25K-token sessions, and TTFT −84% on cache hits for DeepSeek-R1 **[reported by a surveyed system]**.
- **Hierarchical context caching with cache-aware scheduling (a 2026 paper)** **[verified: paper]**:
  - It tiers context caches over GPU HBM, CPU DRAM and disk.
  - **GPU-assisted I/O**: a few SMs run copy kernels that gather small KV pages, decoupling a layer-first GPU layout
    from a page-first host and storage layout.
  - **Cache-aware scheduling**: "delay hits" for requests waiting on a cache that is still being computed, batches
    balanced by load-to-compute ratio, and **bubble filling**, which runs decode batches while loads stall.
  - Reported: up to 5× throughput over two earlier tiered-KV stacks on Llama-70B, and 3.75× over a vendor-optimised
    engine (8× H200, PCIe 5.0, 1.6 TB DRAM) **[reported in a surveyed paper]**.
- **Speculative KV prefetch (a 2024 paper)** **[verified: paper abstract]**. The KV cache lives in host memory. A cheap
  "rehearsal" combines the current layer's input with part of the next layer's query weights and key cache to **predict
  the important tokens**, then prefetches only those entries. Reported: up to 3.00× over earlier KV management for
  offload-based inference, with better accuracy **[reported in a surveyed paper]**.
- **A low-rank key cache on the GPU (a 2024 paper)** **[verified: paper abstract]**. The value cache is offloaded, and
  a small sparse KV set is rebuilt each decode step. Reported: up to 6× larger batches and 3.04× throughput on an A100
  **[reported in a surveyed paper]**.
- **A four-tier KV block manager from a GPU vendor** **[verified: docs]**: GPU; pinned CPU; local NVMe, through a
  transfer library with optional GDS; and remote blob storage.

**[inferred]** On a home cluster, KV tiering is mostly about **prefix reuse for agents** (avoiding re-prefill) and
**freeing VRAM for weights**. Fitting the context at all is the lesser concern. From the datacenter systems, take:
- a paged, content-addressed KV store (radix tree or prefix hash);
- a layer-first GPU layout with a page-first host layout;
- write-through-selective;
- bubble filling.

---

## 13. Speculative decoding as an amortiser for slow tiers

From the single-GPU engine **[verified: paper/docs]**:
- The model's own MTP layer drafts up to 3 tokens. A verify window of up to 4 tokens passes through all 48 layers once.
  A draft enters only while the MTP is at least 50% confident.
- The result is 2.4-3.2 tokens per pass and **1.6-1.8× decode** over plain decoding (47-57 → 82-92 tok/s at 4K)
  **[reported in a surveyed paper]**. Speculative output is token-for-token identical to greedy decoding.
- That gain is smaller than the 2.5-3× typical of all-VRAM engines, because each extra token in the window routes to
  its own CPU experts. Dense weights are amortised across the window; expert misses are not.
- When the reply repeats the context, prompt lookup drafts up to 5 tokens. Code edits get 6-11% faster
  **[reported by a surveyed system]**.
- The MTP layer is taken from the BF16 checkpoint (about a 5-6 GB download), with 2-bit experts and a head reduced to
  40,525 likely tokens.
- On streaming Macs, speculation hides the per-layer handshake: 3 drafts is the default there (§8).
- Batching two requests in parallel slows each one on 12 GB cards **[verified: docs]**.

**[inferred]** Speculation matters more than usual in a tiered engine. It turns one read of the tier-1 weights per
token into one per window. It does nothing for tier-2 and tier-3 bytes that differ from token to token. Across devices,
a window also amortises network round trips: one crossing per window per pipeline stage.

---

## 14. Weight format and CPU kernels decide the bottleneck

The single-GPU engine does not quantise models; it consumes published GGUFs **[verified: docs/paper]**:

| Pack | Experts in RAM | Notes |
| --- | ---: | --- |
| Q2_0 | 34 GB | A 2.25-bpw grid {-1,0,1,2} × scale. **Repacked once** at setup (about 40 GB written, on AVX-512 CPUs) for a hand-written **AVX-512 VNNI** kernel that runs at about 42 GB/s, close to RAM bandwidth |
| IQ2_XS / IQ3_XXS / IQ3_S | 35.5 / 43 / 50 GB | ggml i-quants (codebook lattice), kept native. GPU kernels are transcribed from ggml's; the CPU uses ggml's dot products. **Arithmetic-bound on the CPU: about 5 GB/s per core**, 23-26 GB/s on 6 cores |
| Code-specialised IQ1_M | 23 GB | **Half the experts pruned** (256 of 512 kept, chosen on code). Its makers report 91% on SWE-bench Verified. Weak outside code and on CJK text |
| Dynamic 4-8-bit quants (IQ4_XS to Q8_0) | 55-120 GiB | Top-1 agreement with Q8_0 of 88.9-94.6%, against 87.7% for IQ3_S. Stream from SSD when RAM is short |
| MTP layer | - | From the BF16 checkpoint (§13) |

- **Quality.** Mean KL against a reference ggml implementation on the same file is 0.022. That is below the 0.058 by
  which the reference's own CPU and GPU builds differ **[reported in a surveyed paper]**. Users still complain about
  2-bit quality.
- **Prefill bound.** In the paper's build, prompts were limited by per-expert kernel launches and FP16 dequantisation,
  using about 7 GB/s of a 26 GB/s link. Later fused int8 tensor-core kernels lifted Q2_0 prefill at 32K from 571 to
  2,653 tok/s **[reported by a surveyed system]**.

**[inferred]** The format decides which tier is the bottleneck. Q2_0's simple grid makes the CPU RAM-bound, so PCIe
offload hurts it. The i-quants make the CPU compute-bound, so PCIe offload helps them (§5.4). A tiered scheduler needs
CPU throughput per format, not just byte counts.

---

## 15. End-to-end numbers for the single-GPU MoE engine

All **[reported by a surveyed system]** unless marked. Engine versions differ between tables.

Greedy, MTP on, 4K answers, 32K prompts:

| Machine | Pack | Decode (tok/s) | Prefill (tok/s) |
| --- | --- | ---: | ---: |
| RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200, Windows | Q2_0 | 94 | 2,650 |
| same | IQ2_XS | 79 | 2,090 |
| same | IQ3_XXS | 62 | 1,750 |
| same | IQ3_S | 53 | 1,620 |
| RX 9070 XT 16 GB, Ryzen 9 3900X, 47 GB, Linux | Q2_0 | 60 | 1,160 |
| RX 7900 XTX, Ryzen 9 7900X, 96 GB, ROCm nightly | IQ3_S, 4K / 128K | 60.4 / 59.2 | 1,275 / 1,494 |

- **Context scaling, RTX 5070, Q2_0.** Decode is 87.3 / 93.0 / 81.8 / 76.2 / 73.7 / 60.3 tok/s at 1K / 4K / 32K / 64K /
  128K / 262K. Prefill is roughly flat at 2,100-2,170 tok/s from 32K to 128K.
- **Paper, early engine** **[reported in a surveyed paper]**. Decode 88.7-94.6 tok/s for Q2_0 at 1K-4K, and 56.3 at
  262K. Prefill was only 389-571 tok/s, a TTFT of 55 s at 32K. **The paper's prefill figures were 4-5× out of date
  within weeks.**
- **Paper estimates, labelled ±20% until measured.** RTX 3090 24 GB: about 128-140 tok/s decode at 1K-4K for Q2_0.
  RTX 5060 Ti 16 GB: about 80-87.
- **Mac** (M5 Max 128 GB, streaming 5-6-bit dynamic quants). Story decode is 51 tok/s for the 6-bit quant, and a 15K
  prompt reads at 1,020 tok/s. On the same Mac with Q2_0, a generic ggml-based engine decodes at 56.6 tok/s and
  prefills at 887 / 809 tok/s (4K / 32K), against the engine's own 1,430 / 1,460 tok/s prefill.
- **Multi-GPU community reports:**
  - 2× RTX 3090 with 165 GiB RAM: 64-78 tok/s split, against 31 on one card under a RAM budget.
  - 2× P40: IQ2_XS at 34.8 tok/s split, against 19.6 on one card.
- **Third-party comparison** **[unverified]**. A secondary write-up describes an independent benchmark on one A100
  capped to 12 GiB VRAM, with the same IQ2_XS file in both engines. Decode was **2.7× faster than a generic CPU/GPU
  engine with 64 CPU cores, and 4.9× with 8 cores**, with no quality difference (GSM8K 93.0% vs 93.3%, HumanEval
  157/164 each). The original benchmark and its absolute numbers were not found. It predates the generic LRU expert
  cache (§5.5).

---

## 16. Multi-device: inside one machine and across machines

**Inside one PC** (the single-GPU engine with extra cards) **[verified: docs]**:
- **Layer split (pipeline parallelism).**
  - Each card gets a contiguous range of layers and caches experts only for its own layers.
  - A token crosses a card boundary **once per verify window**, a few hundred KB through pinned RAM. No NVLink or P2P is
    needed, and x4 or x1 slots work.
  - Auto placement prices a decode window against free VRAM. It puts the fastest card last, because the last card
    runs the head, the MTP and verification.
  - Tuning matters a lot. On an unequal pair, the old rule gave the second card 3 layers, and the new rule (K=20) gave
    **+52% prefill and +31% decode** **[reported by a surveyed system]**.
- **Helper GPUs as expert tiers.**
  - Each helper GPU holds a disjoint expert cache, so each expert has exactly one owner: the main GPU, one helper or
    the CPU.
  - Outputs are packed before they return over PCIe or USB4.
  - About 1,900-2,000 duplicate slots had been observed. Stopping the main GPU from promoting experts a helper already
    holds gave +15.5% generation on 2× RTX 3080 20 GB **[reported by a surveyed system]**.
  - The helper tier synchronises on every layer, whereas the layer split crosses once per window.
- **Peer access.** P2P is used only after a test copy in each direction arrives exactly.
- **Scope.** The engine is single-machine by design, and its authors do not plan to split a model across machines.

**Across machines and clusters:**
- **Offload against a decentralised pipeline** (the 2023 offload paper). Offload on one T4 had higher per-GPU
  throughput than a decentralised pipeline over 4× T4, in every simulated network condition. The authors judge
  collaborative inference more viable where latency matters and the network is fast enough **[reported in a surveyed
  paper]**. Offload and distribution are complements, which is Syntropy's premise.
- **The ggml-based engine's remote-device backend** **[verified: docs]**:
  - A server process exposes remote CUDA, Metal or CPU devices.
  - The main host distributes **weights and KV in proportion to each device's free memory**, with an adjustable split,
    and keeps a local tensor cache.
  - Its docs call it a fragile, insecure proof of concept that must never run on an open network.
  - It is static and proportional, with no per-layer awareness of network latency.
- **CPU/GPU hybrid MoE inside a datacenter engine.** More GPUs gave +264% throughput at concurrency 8 but only +26% for a
  single stream (§2.3).

**[inferred]** No surveyed system **combines per-device tiering (VRAM/RAM/SSD with adaptive expert caching) with
network distribution and per-device placement optimisation.** That combination is Syntropy's open space.

---

## 17. Exactness hazards

The single-GPU engine's paper headlines exact speculative output, but **its current defaults are no longer exact**
**[verified: docs]**:
- **Skipping the routed tail on misses.** In a verify window, a *missed* expert that every token ranks 7th-10th is
  neither copied nor computed, and the remaining weights are not renormalised. It is on by default on CUDA. It turns
  off when ≥90% of experts are cached, when batch slots are used, and on most HIP parts. Disabling it restores exact
  output.
- **CPU share in short prefill chunks.** For chunks under 1,024 tokens on a single CUDA GPU, part of the work goes to
  the CPU, which rounds differently from the GPU. Reported mean KL 0.004 (max 0.025), for 20-34% less prompt time.
- **Results vary from run to run by default.** The GPU and CPU round the same expert differently, and the adaptive tier
  moves experts between them across requests. Byte-identical repeats need the prompt cache, adaptive swaps and the PCIe
  share all switched off, which costs decode speed.

Other approximations in the field:
- Low-precision copies of missed experts (§5.5).
- **Expert deferral**: some experts' compute is deferred to overlap CPU work with GPU attention. CPU utilisation rises
  from under 75% to almost 100%, for up to 1.45× throughput and an average accuracy drop of 0.5% or less **[reported in
  a surveyed paper]**.
- Routing to 6 experts instead of the model's 8 (§2.3).
- Approximate substitution for low-score misses (§5.5).
- Pruning half the experts (§14), and 2-bit quants that break JSON and tool calls (§8).
- 4-bit KV: perplexity +8-12% on long documents (§12.1).

**[inferred]** Engines that mix device types drift toward "exact by flag, approximate by default" for the sake of
speed. Syntropy's policy is set in ADR-0000 D3: exact by default, with every shortcut opt-in and labelled. The
surveyed engine's test approach (teacher-forced KL and top-k agreement against a reference implementation) is a good
template for the parity harness. Syntropy will mix devices across a network and diverge more, so it needs that harness
from the first milestone.

---

## 18. Observed failure modes and anti-patterns

**Memory and I/O**
- mmap for a hot tier: file-backed pages are evicted first, causing re-faults and 19 vs 42.8 GB/s (§10.1).
- mmap on SATA for row lookups: 2.1K vs 14.4K rows/s; on Windows, slow mapped files stalled prompts (§9).
- mmap on iGPUs doubled peak RAM during load (§2.2).
- One giant `cudaHostAlloc` on Linux: OOM-kill with high MemAvailable, because clean file cache can't be reclaimed fast
  enough (§10.2).
- Locking one huge range on Windows fails; the WSL2 pinned cap is about 1 GiB; macOS compressor panics on an unwired
  arena (§10.2, §8).
- A 1-3 minute near-freeze on first start while 35-55 GB loads and part of it is locked.
- A single issuing thread caps at about 80K reads/s (§10.3).
- Unified-memory fallback on discrete GPUs trades a crash for a large slowdown (§2.2).

**Caching and scheduling**
- One shared arrival-order admission counter filled the cache from the first ~26 layers, for a 2.97% hit rate against
  70.4% per layer (§5.2).
- Uniform slot sizes wasted capacity (2,673 vs 3,556 experts) (§5.2).
- Duplicate residency across GPUs cost about 2,000 slots (§16).
- Swaps on the critical path cost ~3% decode (91.7 vs 94.4 tok/s) (§5.3).
- Splitting a verify window into two token groups was 7% slower (§6).
- Custom caches, prefetch and speculative routing lost to the OS page cache on Apple's 17.5 GB/s SSD (§8).
- A fixed CPU/PCIe split constant (1.85) fitted to three machines; per-architecture defaults; NUMA and x16+x4 topology
  pricing remain open problems.
- Long context silently costs expert capacity: decode ~95 → ~56 tok/s at 262K before KV streaming (§5.2).
- Batching two requests slows each one on 12 GB cards (§13).

**Product and process**
- Single-model specialisation: geometry hard-coded into kernels, so any other model is refused (§3.2).
- Dozens of environment switches with per-GPU-architecture defaults.
- Fast daily releases shipped measurable decode regressions (about 69 → 55-58 tok/s on an RTX 5060 Ti between nearby
  versions). Published paper numbers went 4-5× stale within weeks (§15).
- Prompt and prefix reuse with agent clients, multi-GPU split pricing and stability, device-lost and hang reports on
  multi-GPU AMD and Intel, garbage output on an M2 Max, model coverage and 2-bit quality are the main open complaint
  themes.
- Layer streaming at batch 1: seconds per token, whatever the prefetch (§2.1).
- A greedy device map that packs sequentially with no prefetch, keeping only one GPU busy at a time (§2.1).
- Proportional, static remote placement with no latency model, and no security on an open network (§16).

---

## 19. Install, API and platform surface

From the single-GPU engine **[verified: docs/code]**. This is not tiering technique, but it shaped how widely the engine
spread.

- **Process model.** A Python HTTP server starts the C++ engine and talks to it over stdin/stdout pipes, with requests
  serialised through a FIFO. The server binds `127.0.0.1`. Without an API key it accepts only requests whose `Host` is a
  local name, and checks `Origin` (a DNS-rebinding fix).
- **Install.**
  - One entry point per OS runs a single large setup script. It detects GPUs, RAM and disk, picks model and size by RAM,
    and asks three questions (model and size, context, images), each with a recommended default.
  - It installs a user-local Python and venv without admin rights.
  - It fetches a **prebuilt engine** for RTX 20-50 plus NVIDIA CUDA libraries from pip (about 0.4 GB). If none fits, it
    offers to install the toolchain and compile, which takes 20-40 minutes.
  - It downloads about 70 GB of model, resumable, with pinned revisions and SHA-256 checks.
  - The only prerequisite is the GPU driver (NVIDIA 580 or newer).
  - Extras: an optional calibration pass (15-30 minutes, §4), a shareable benchmark report, an updater, and an MCP
    server so external tools can drive install, start and stop.
- **Friction.** Driver versions; about 70-80 GB of disk, plus about 40 GB more for the Q2_0 repack; the near-freeze on
  first start; RAM-size gates on which models can run; release regressions.
- **API.**
  - OpenAI-compatible chat completions and Responses endpoints, and Anthropic-style `/v1/messages`.
  - Slot save/restore, an endpoint that lends or reclaims VRAM (§10.5), and Prometheus metrics.
  - A web UI with chat and a live GPU/CPU/RAM monitor; reasoning effort; JSON response formats; logit bias; MCP tools
    injected into chat; image input.
  - One request at a time by default, with opt-in batching (§13).
- **Platforms:**

| Platform | Status |
| --- | --- |
| NVIDIA CUDA, RTX 20-50 (sm_75+) | Primary, with prebuilt engines. Pascal/Volta experimental (CUDA 12 build) |
| AMD HIP: RDNA2 (6800/6900), RDNA3, RDNA4, Radeon AI PRO R9700 | Validated or community-tested; ROCm on Linux. Container images not yet supported on Windows |
| Intel Arc (SYCL) | Linux build from source, experimental |
| Apple Silicon M1-M5 (Metal) | Builds from source in about 7 minutes; community-supported |
| AMD Strix Halo, Radeon 780M iGPU | Experimental |
| CPU | AVX2 required for speed, AVX-512 preferred; no-AVX2 is experimental and slow |
| OS | Windows 10/11, Linux (Ubuntu 22.04/24.04, Debian 12/13 automated), Docker. WSL2 works with limits (§10.2) |

---

## 20. Recommendations for Syntropy

Ordered by expected impact for a cluster of consumer devices. Merged from both earlier notes; nothing dropped.

1. **Build a placement planner on a byte census and a cost model.**
   - Classify every tensor by access pattern, not by layer or size:
     - *dense, every token*: keep in the fastest tier, never stream;
     - *sparse with skewed popularity* (MoE experts): cache with adaptive LFU;
     - *deterministic row lookups* (embedding and n-gram rows): prefetch a step ahead from SSD;
     - *KV*.
   - Compute a §3.1-style census per model before placing anything.
   - Solve placement over devices and tiers as an LP/ILP, in the style of the 9-variable LP (§4). Extend it with roofline
     compute terms per device and a network-link term per device pair, and calibrate constants at runtime.
   - Outputs: pinned dense weights, the expert-cache size per device, and the window of KV kept resident.
   - Expose the result as an overridable module-to-device map plus regex overrides per tensor (§3.3).
2. **At the slow tiers, move compute to the bytes, not bytes to the compute.**
   - Compute RAM-resident misses on the CPU in place, and send only a measured share over PCIe (§5.1, §5.4).
   - Across a network the rule becomes absolute. One expert is 1.38 MB, while a token's hidden state is 2,560 × 2 B ≈
     5 KB in BF16. The cross-device path should ship activations, never weights, during decode **[inferred: sizes from
     §3.2]**.
3. **Treat the CPU pool, the PCIe link and the GPU as three workers, and balance them online.**
   - Use the per-layer least-squares cost model (§5.4), including the term for DMA traffic slowing the CPU pool.
   - Calibrate at runtime rather than shipping constants; the 1.85 constant was fitted to three machines.
   - Add network peers as a fourth worker class, with their own fitted latency and bandwidth terms.
4. **Give each device an expert tier: an adaptive VRAM cache over a pinned RAM arena, with misses computed in place.**
   - Per-layer, byte-sized slots, with admission per layer (§5.2).
   - Decayed LFU with hysteresis (§5.3). It took hit rate from 0.50 to 0.72, and per-expert caching beats whole-layer
     pinning by 1.6-2.2× (§5.5).
   - Swaps off the critical path: evict now, admit when the copy lands.
   - A learned profile persisted across restarts.
   - CPU-side miss compute, NUMA-local on workstations (§2.3).
   - **Disjoint expert ownership across devices**: no expert is duplicated in the cluster unless it is deliberately
     replicated because it is hot. Duplicate residency cost about 2,000 slots in one machine (§16).
   - Keep learned or forecast-driven replacement (§5.5) as the upgrade path beyond decayed LFU.
5. **Feed prefetch at every slow boundary from next-layer expert prediction.**
   - Apply layer l+1's router to layer l's hidden state (§6). It predicted about half of SSD misses for +14%, at the
     cost of a host copy of the routers (126 MB for the §3.1 model). It is cheap and works for any MoE.
   - Use exact routing where the model offers it (pre-gate, pre-attention router), exposed as a "routing known at time
     t" capability flag.
   - Keep a learned or forecast predictor as the upgrade path.
   - Prefetch targets: SSD → RAM pages, RAM → VRAM slots, and peer → local where a peer holds the weights.
6. **Make speculative decoding (MTP or a draft model) a first-class tier feature.**
   - It amortises the read of tier-1 weights across a verify window (1.6-1.8×), and hides per-layer handshakes on
     streaming tiers (3 drafts is best when streaming on a Mac).
   - Across devices, a window also amortises network round trips: one crossing per window per pipeline stage
     **[inferred]**.
7. **Across devices, prefer a pipeline split by layer ranges over per-layer expert parallelism.**
   - Each stage owns its layers' experts. Ship activations (about 5 KB per token in BF16 for a 2,560-wide model), and
     cross each link once per window (§16).
   - A per-layer synchronising design pays a round trip on every layer. Over Ethernet, 48 layers × 0.2 ms = 9.6 ms per
     window at a typical LAN RTT, which is about the whole GPU budget of a 4K Q2_0 window (14.6 ms, §5.1)
     **[inferred: the RTT is an assumption to measure in Syntropy's benchmarks]**.
   - Reserve expert parallelism across machines for fast links (USB4/TB, 10 GbE and up), and only after measuring
     per-layer RTT.
   - Expect extra devices to raise aggregate throughput far more than single-stream latency when experts sit in DRAM
     (§2.3).
8. **Plan KV and weights together, because they share the scarce tier.**
   - Above a context threshold, stream KV from RAM to free VRAM for experts. Above 64K this fitted 2.4× more experts and
     raised decode by 23% (§12.1).
   - Build a tiered, content-addressed KV store: GPU → pinned RAM → SSD, and optionally a peer's RAM, pooling spare
     DRAM and SSD across the cluster.
   - Use radix-tree or prefix-hash addressing, a layer-first GPU layout with a page-first host layout, write-through-
     selective, and bubble filling (§12.2). Put the store behind a connector-style seam.
   - Persist session files, bound to an identity hash of the model files and every setting that changes meaning, so
     agents never re-prefill. Share prefix KV across parallel slots; agent prefix reuse is the main pain point users
     report.
9. **Use pinned anonymous memory for the hot RAM tier, not mmap.**
   - Measured: 19 vs 42.8 GB/s, plus eviction from the file cache (§10.1).
   - Budget each platform's pinning limits from day one: about half of RAM on Windows, about 1 GiB on WSL2, wiring and
     a footprint cap on macOS, and paced registration with cgroup and NUMA awareness on Linux (§10.2).
   - Stage pinning progressively so the first start doesn't freeze the machine.
   - mmap stays right only for cold, read-once tiers.
10. **Make the SSD tier a capacity tier, laid out by read unit and read with direct I/O.**
    - Lay out by expert: one contiguous, aligned read per expert (+17% decode on a Mac).
    - Read through an `O_DIRECT`/IOCP thread pool with queue depth of at least 16 (about 7× faster than mmap on SATA).
      Benchmark io_uring as a Linux upgrade.
    - Fall back to mmap where measurement says the OS does better, as on Apple's SSD.
    - Send deterministic lookup tables to SSD (or a peer's RAM) first: detect tensors that look like embeddings,
      addressed by row from token ids, and prefetch them.
    - Never depend on GDS; it does not apply to consumer GPUs.
    - Plan on 1.5-2.5 GB/s effective for expert-sized random reads, not the drive's sequential rating (§11).
    - Track energy per token on laptops and phones when weights come from SSD.
11. **Decide the exactness policy and build the parity harness first.**
    - Low-precision misses, expert deferral or skipping, and top-k reduction are opt-in knobs. Report each with
      teacher-forced KL and top-1 agreement against a reference.
    - Rounding differences across CPU, GPU and peers make byte-exact output unlikely by default. Say so in the product.
12. **Schedule by CPU throughput per format, not just bytes.** The format decides whether the CPU is RAM-bound or
    compute-bound, and so whether PCIe offload helps or hurts (§14).
13. **Install: one command.** Detect hardware, choose the model by RAM, make downloads resumable and checksummed, and
    ship prebuilt engines. Avoid the surveyed engine's costs: a multi-GB repack at setup, a 1-3 minute freeze at load,
    and a sprawl of environment flags per architecture.
14. **Don't become a single-model engine.** The surveyed engine's speed comes partly from hard-coding one geometry, which
    is also why it can't run anything else. A generic LRU expert cache already reaches 1.6-2.2× its own baseline. Get
    generic placement right first, then add kernel specialisation per model as a plug-in.
15. **Phones as nodes:** an NPU/CPU split with a pipelined flash I/O path (§7), for models that are activation-sparse.

**Not recommended as core mechanisms:**
- Streaming dense layers from SSD (seconds per token). Layer-at-a-time disk loading is only the worst-case fallback for
  a node with too little RAM.
- Neuron-level sparsity, which needs ReLU-sparse models; a plug-in at most.
- GDS or GPU-initiated storage, neither of which consumers can install.
- Static whole-layer partitioning for MoE, which is 1.6-2.2× worse than per-expert caching.
