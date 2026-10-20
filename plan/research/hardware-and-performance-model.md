# Hardware data paths and the inference cost model

Research input for the Syntropy placement solver and scheduler. Written 2026-10-19.

The research notes `multi-device-inference.md` and `tiered-memory-inference.md` cover how surveyed systems behave. This note supplies the numbers and the arithmetic those design choices rest on. The solver RFC should be able to take the equations in §3 as its objective function and the measurement list in §5 as its probe set.

## How to read this

- **[verified: [Sn]]** means the number appears in source Sn (list at the end). Where a source is a forum post or a single community run, the text says so.
- **[reported by a surveyed system]** means the number comes from a surveyed inference system's own published results (paper, README, docs or blog). The test conditions are kept next to it. Per ADR-0000 D5 the system is not named or linked, so these figures carry no Sn.
- **[estimate: …]** means I derived it. The derivation is in the tag or next to it. Every estimate is reproducible from the verified or reported inputs.
- Units: GB = 10^9 bytes, GiB = 2^30. Bandwidth is per direction unless stated. "Peak" is the vendor or spec figure; "effective" (W_eff) is what a real kernel achieves.
- All worked-example outputs are estimates: they are upper bounds under stated efficiency factors, not benchmarks.
- Source IDs are kept stable across revisions of this note, so the numbering has gaps.

---

## 0. What drives the design

1. **Single-stream decode is a bytes-moved problem.** Per token, `t ≈ Σ_tiers bytes/W_eff + Σ_hops (α + S/β) + t_fixed`. Achieved bandwidth is 0.43–0.79 of peak for GPU-class engines in the anchors below (§2.2), and 0.24–0.35 end-to-end for CPU-hosted experts. It also varies with model architecture (MoE runs lower than dense on the same Mac), so η has to be measured per device, kernel and model, not looked up.
2. **Weights outside VRAM should be computed where they sit during decode**, on the CPU or iGPU over DRAM, not streamed over PCIe to the GPU. Desktop DRAM (70–120 GB/s) beats PCIe 4.0 x16 pinned (≈24 GB/s). Streaming to the GPU wins only once a batch or prompt chunk exceeds roughly 12–24 tokens on a desktop CPU (§3.2).
3. **Dense model + NVMe tier is unusable**: every overflowed byte is re-read every token. **MoE + NVMe is viable only with a high RAM hit rate.** Even at a 100 % RAM hit rate, dual-channel DDR5 caps a DeepSeek-V3-class Q4 model at ≈5 tok/s (§4b). Memory-channel count is the real limit on a desktop.
4. **Pipeline parallelism (PP) across devices doesn't speed up one stream.** It adds capacity and multi-stream throughput. A single stream only gets faster when bytes move onto a higher-bandwidth device.
5. **Tensor and expert parallelism (TP/EP) need 120–160 collectives per token** for 60–80-layer models. They pay off only when the *effective* per-collective latency (wire plus GPU sync) is below ≈200–300 µs on Mac-class nodes (§3.8). In practice that means RDMA over Thunderbolt 5 or real RDMA NICs: never Wi-Fi, and marginal at best on TCP over 2.5/10 GbE.
6. **Wi-Fi's median hop (3–7 ms) is tolerable for PP of large models. Its tail is not.** One large study puts p99 at ≈250 ms. Wi-Fi links must be costed with tail latency, or kept off the critical path.
7. **MoE batching amortises poorly.** With top-8 of 256 experts and uniform routing, 8 concurrent tokens touch 7.2× the expert bytes of one token. Throughput mode for MoE wants experts in fast memory on more nodes, not deeper batches on slow tiers.
8. **KV cache**: Llama-3-70B (GQA) is 320 KiB/token and DeepSeek-V3/Kimi-K2 (MLA) is 68.6 KiB/token. In multi-user mode, KV capacity binds before weights do.
9. **Speculative decoding amortises slow tiers for dense models**, because the target pass costs about the same for 1 or γ+1 tokens while memory-bound. For MoE, the verify pass touches the union of experts across the drafted tokens, which erodes the gain.
10. **The solver must measure** at startup: W_eff per (device, tier, kernel), PCIe pinned H2D, NVMe *sustained, throttled* read at the real request size, link α and β including sync overhead plus a p99, t_fixed and F_eff. Expert hit-rate curves must be measured online (§5).

---

## 1. Notation

| Symbol | Meaning | Unit |
|---|---|---|
| `B_{d,i}` | bytes read per decode step from tier *i* on device *d* (weights + KV) | GB |
| `W_{d,i}` | effective read bandwidth of tier *i* as seen by the kernel that consumes it | GB/s |
| `η` | W_eff / W_peak | – |
| `C_{d,i}` | usable capacity (after OS, wired limits, KV and scratch reservations) | GB |
| `α_h`, `β_h` | one-way application-level latency and goodput of hop *h* | s, GB/s |
| `F_d` | effective FLOP/s of device *d* for the matmul precision in use | FLOP/s |
| `t_fixed` | per-token cost that does not scale with bytes (launch, sync, sampling) | s |
| `N`, `N_act` | total and per-token-active parameters | – |
| `b` | bytes per parameter at the chosen quant (Q4_K_M ≈ 0.60, MLX 4-bit ≈ 0.5625, BF16 = 2) | B |
| `h_model` | hidden size | – |
| `L` | context length (tokens) | – |
| `B` | concurrent sequences (batch) | – |
| `E`, `k` | routed experts per MoE layer, experts per token | – |
| `h` | expert-cache hit rate for a tier | – |
| `γ`, `a` | speculative draft length, per-token acceptance probability | – |

---

## 2. Reference data

### 2.1 Memory bandwidth

**Discrete GPUs (consumer)**

| GPU | Memory | Capacity | Bus | Peak BW | Host link | Tag |
|---|---|---|---|---|---|---|
| RTX 3090 | GDDR6X 19.5 Gbps | 24 GB | 384-bit | 936 GB/s | PCIe 4.0 x16 | [verified: [S12]] |
| RTX 4090 | GDDR6X 21 Gbps | 24 GB | 384-bit | 1008 GB/s | PCIe 4.0 x16 | [verified: [S10]] |
| RTX 5080 | GDDR7 30 Gbps | 16 GB | 256-bit | 960 GB/s | PCIe 5.0 x16 | [verified: [S11]] |
| RTX 5090 | GDDR7 28 Gbps | 32 GB | 512-bit | 1792 GB/s | PCIe 5.0 x16 | [verified: [S11]] |
| RX 7900 XTX | GDDR6 20 Gbps | 24 GB | 384-bit | 960 GB/s | PCIe 4.0 x16 | [verified: [S13]] |
| RX 9070 XT | GDDR6 ≈20 Gbps | 16 GB | 256-bit | 640 GB/s | PCIe 5.0 x16 | [verified: [S14]] |

RTX 4090 compute, from the Ada whitepaper's comparison table (dense/sparse): FP16 tensor 165.2/330.4 TFLOPS with FP32 accumulate, 330.3/660.6 with FP16 accumulate; FP8 660.6/1321.2; INT8 660.6/1321.2 TOPS; FP32 non-tensor 82.6; L2 72 MB [verified: [S10]]. RTX 50-series compute figures were not pulled from a primary source here. Treat them as unknown until probed.

**Datacenter HBM (for scale only; Syntropy's targets are consumer)**

| Part | Memory | BW | Tag |
|---|---|---|---|
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | [verified: [S15]] |
| H200 | 141 GB HBM3e | 4.8 TB/s | [verified: [S15]] |
| B200 | 180–192 GB HBM3e (sources differ) | 8 TB/s | [verified: [S15]] |
| MI300X | 192 GB HBM3 | 5.3 TB/s | [verified: [S15]] |
| MI355X | 288 GB HBM3e | 8.0 TB/s | [verified: [S15], [S16]] |

**Apple silicon unified memory**

| Chip | Peak BW (GB/s) | Max memory | Memory | Tag |
|---|---|---|---|---|
| M1 | 68.25 | 16 GB | LPDDR4X-4266 | [verified: [S5]] |
| M1 Pro / Max / Ultra | 204.8 / 409.6 / 819.2 | 32 / 64 / 128 GB | LPDDR5-6400 | [verified: [S5]] |
| M2 / Pro / Max / Ultra | 102.4 / 204.8 / 409.6 / 819.2 | 24 / 32 / 96 / 192 GB | LPDDR5-6400 | [verified: [S6]] |
| M3 / Pro | 102.4 / 153.6 | 24 / 36 GB | LPDDR5-6400 | [verified: [S7]] |
| M3 Max (30c / 40c GPU) | 307.2 / 409.6 | 96 / 128 GB | LPDDR5-6400 | [verified: [S7]] |
| M3 Ultra | ≈819 | 512 GB | LPDDR5-6400 | [verified: [S7], [S17a]] |
| M4 / Pro | 120 / 273 | 32 / 64 GB | LPDDR5X-7500 / 8533 | [verified: [S8]] |
| M4 Max (32c / 40c GPU) | 410 / 546 | 36 / 128 GB | LPDDR5X-8533 | [verified: [S8]] |
| M5 | 153.6 | 32 GB | LPDDR5X-9600 | [verified: [S9]] |
| M5 Pro | 307 | 64 GB | LPDDR5X-9600 | [verified: [S1], [S2]] |
| M5 Max (32c / 40c GPU) | 460 / 614 | 128 GB (40c) | LPDDR5X-9600 | [verified: [S1], [S4]] |
| M5 Ultra (Mac Studio, Aug 2026) | 1200 ("1.2 TB/s"; 1228.8 per [S9]) | 512 GB | LPDDR5X | [verified: [S3], [S4]] |
| M6 (Mac mini, Aug 2026) | 170 | 32 GB | – | [verified: [S3]] |

The Mac Studio (M5 Max/Ultra) has four rear Thunderbolt 5 ports at 120 Gb/s, plus two more on the front for M5 Ultra, 10 GbE and Wi-Fi 7 [verified: [S4]]. The M3 Ultra Mac Studio has six TB5 ports [verified: [S17a], [S52]].

The GPU on macOS cannot wire all of unified memory. The default cap is reported at roughly 65–75 % of RAM, adjustable with `sysctl iogpu.wired_limit_mb`. This is undocumented by Apple and may change [verified (community, not Apple): [S76]].

**CPU DRAM (desktop, workstation, server)**

Peak = channels × 8 B × MT/s; a DDR5 DIMM channel is 64 bits [verified formula: [S22]].

| Config | Peak | Measured | Tag |
|---|---|---|---|
| DDR4-3200, 2 ch | 51.2 GB/s | – | [estimate: 2×8×3200] |
| DDR5-5600, 2 ch (Core Ultra 7 265K) | 89.6 | 75.97 | [verified: [S22]] |
| DDR5-6000 CL28, 2 ch (265K) | 96.0 | 90.01 | [verified: [S22]] |
| DDR5-7800, 2 ch (265K) | 124.8 | 117.8–122.4 | [verified: [S22]] |
| Ryzen (AM5), per CCD | link 32 B/clk read × FCLK: 64 GB/s at 2.0 GHz, 70.4 at 2.2 GHz | 70.08 GB/s read on a 9950X at DDR5-8000, 55 % of DRAM | [verified: [S21]] |
| Xeon Gold 6454S, 8 ch DDR5-4800 per socket (testbed of a surveyed CPU/GPU hybrid MoE engine) | 307.2 per socket | see §2.2 | [estimate: 8×8×4800] |
| Threadripper, 8 ch DDR5-6400 | 409.6 | – | [estimate: formula] |
| EPYC 9005, 12 ch DDR5-6400 | 614.4 per socket | server Zen 5 CCD (GMI-Wide) 99.8 GB/s read vs 62.5 GB/s on a desktop 9900X CCD | [verified channels/speed: [S23]; CCD figures: [S21]] |

**AM5 trap:** a single-CCD Ryzen (8 cores or fewer) cannot read faster than its CCD link (≈64–70 GB/s), whatever DIMMs are fitted. Dual-CCD parts have two links. The solver must measure, not compute from DIMM speed.

**LPDDR5X laptops, phones, unified-memory boxes**

| Device | Memory | Peak BW | Tag |
|---|---|---|---|
| Snapdragon X Elite laptops | LPDDR5X-8448 | 135 GB/s | [verified: [S24]] |
| Intel Lunar Lake | LPDDR5X-8533, 128-bit | 136.5 GB/s | [estimate: 16 B × 8533] |
| Snapdragon 8 Elite Gen 5 (phone) | LPDDR5X, 64-bit | 84.8 GB/s | [verified: [S25]] |
| Apple A19 Pro (phone) | not found | 60–77 GB/s | [estimate: 64-bit LPDDR5X at 7500–9600 MT/s] |
| AMD Ryzen AI Max+ 395 (Strix Halo) | 256-bit LPDDR5X-8000, up to 128 GB, up to 96 GB as GPU memory on Windows | 256 GB/s | [verified: [S19]] |
| — Strix Halo CPU side | reads limited by CCD links: 64 GB/s per CCD, 124 GB/s measured across both | 124 GB/s | [verified: [S20]] |
| — Strix Halo GPU side to DRAM | not found | expected well above the CPU's 124 GB/s; measure | [estimate] |
| NVIDIA DGX Spark (GB10) | 128 GB LPDDR5x, 256-bit, 16 ch | 273 GB/s | [verified: [S17]] |

DGX Spark also has 6144 CUDA cores, "up to 1 PFLOP FP4 with sparsity", a ConnectX-7 NIC with 2× QSFP, 10 GbE, Wi-Fi 7, and a 240 W PSU [verified: [S17]]. Two units link at up to 200 Gbps [verified: [S18]].

Consequence for Strix Halo: run weights on the iGPU, not the CPU. The CPU sees under half the DRAM bandwidth.

### 2.2 Achieved versus peak: calibration anchors

The η column is [estimate], derived from the measurement in the first column.

| Measurement | Bytes/token assumed | Achieved | η |
|---|---|---|---|
| ggml Metal, LLaMA-7B Q4_0, TG128 [S26]: M4 Pro 50.74 t/s | 6.74 B params × 4.5 bpw = 3.79 GB | 192 GB/s of 273 | 0.70 |
| same, M4 Max 40c 83.06 t/s | 3.79 GB | 315 of 546 | 0.58 |
| same, M2 Ultra 76c 94.27 t/s | 3.79 GB | 357 of 800 | 0.45 |
| same, M3 Ultra 80c 92.14 t/s | 3.79 GB | 349 of 819 | 0.43 |
| M3 Ultra, Llama 3.1 70B, 14.1 t/s [S49] (third-party review; quant not stated, assume Q4_K_M 42.5 GB) | 42.5 GB | ≈600 of 819 | ≈0.73 |
| M3 Ultra, DeepSeek-V3-0324 MLX 4-bit, ≈21 t/s generation [S75] | 36.5 B active × 4.5 bpw = 20.5 GB | ≈431 of 819 | ≈0.53 |
| CPU/GPU hybrid MoE engine, single Xeon Gold 6454S socket (8 ch DDR5-4800) + RTX 4090D, DeepSeek-V3 Q4_K_M, 8 experts, 8.73 t/s [reported by a surveyed system] | 12.3 GB routed on CPU | ≈107 of 307 | ≈0.35 (includes GPU time and CPU–GPU sync) |
| same, dual socket (1 TB DDR5-4800), 12.21 t/s [reported by a surveyed system] | 12.3 GB | ≈150 of 614 | ≈0.24 |

Two lessons:

- **Large chips are under-used by small models.** A two-point fit on M3 Ultra (7B at 92.14 t/s, 70B at 14.1 t/s, assuming Q4_K_M) gives `t = B / 645 GB/s + 5.0 ms` [estimate]. So W_eff ≈ 0.79 × peak, and t_fixed ≈ 5 ms per token for ggml Metal. The intercept matters for small models and for every extra pipeline stage, since each stage has its own launch and sync cost.
- **MoE kernels run at lower η than dense** (0.53 vs ≈0.73 on the same Mac). Many small expert GEMVs and gathers do worse than a few large ones. CPU-hosted experts in hybrid systems come in lower still (0.24–0.35 end-to-end). The hybrid engine's own paper names CPU compute limits and CPU–GPU synchronisation as the bottlenecks [reported by a surveyed system].

### 2.3 Host↔device links

| Link | Raw per direction | Measured / notes | Tag |
|---|---|---|---|
| PCIe 3.0 lane / x4 / x8 / x16 | 0.985 / 3.94 / 7.88 / 15.75 GB/s | 128b/130b at 8 GT/s | [verified: [S28]] |
| PCIe 4.0 lane / x4 / x8 / x16 | 1.97 / 7.88 / 15.75 / 31.5 GB/s | 4090 pinned H2D ≈24 GB/s (76 % of raw); pageable ≈6 GB/s, ≈11 GB/s with E-cores disabled | [verified raw: [S28]; measured (one forum system): [S30]] |
| PCIe 5.0 lane / x4 / x8 / x16 | 3.94 / 15.75 / 31.5 / 63.0 GB/s | no measured pinned figure found; expect 48–52 GB/s | [verified raw: [S28]; estimate: 0.76–0.83 × 63] |
| Pinned vs pageable (older hardware) | – | laptop: 2.31 pageable vs 5.77 GB/s pinned H2D; desktop: 5.37 vs 6.19 | [verified: [S29]] |
| AM5 chipset uplink (X870/X870E) | PCIe 4.0 x4 ≈ 7.9 GB/s, shared by every chipset-attached M.2, NIC and USB device | – | [verified: [S31]] |
| Thunderbolt PCIe tunnel (eGPU) | TB4 32 Gb/s, TB5 64 Gb/s | eGPU H2D ≈3 GB/s (TB4) to ≈6.5 GB/s (TB5) | [verified: [S47]; estimate: raw ÷ 8 × ~0.8] |

Why pageable is slow: the driver first copies pageable data into a pinned staging buffer, so the effective rate is bounded by a single-threaded host memcpy [verified: [S29], [S30]]. **Rule:** every streaming path (NVMe → host → GPU, network → GPU) uses pre-pinned ring buffers (`cudaHostAlloc`/`cudaHostRegister`, or HIP equivalents). Budget them, because pinned memory is not swappable.

**Unified-memory machines** (Apple, Strix Halo, GB10) have no host↔device copy. The tier graph collapses to {unified DRAM, NVMe}, with a GPU allocation cap instead of a VRAM size.

### 2.4 Storage

| Drive | Interface | Seq read | 4K random read QD1 | 4K random read deep QD | Tag |
|---|---|---|---|---|---|
| Samsung 970 EVO Plus | PCIe 3.0 x4 | 3500 MB/s | 19K IOPS ≈ 78 MB/s | 480K–620K IOPS (QD32, by capacity) | [verified: [S34]] |
| Samsung 990 PRO | PCIe 4.0 x4 | 7450 MB/s | 22K IOPS ≈ 90 MB/s ≈ 45 µs/IO | 1.2M (1 TB) / 1.4M (2 TB) IOPS QD32 ≈ 4.9–5.7 GB/s | [verified: [S32]] |
| Samsung 9100 PRO | PCIe 5.0 x4 | 14,700–14,800 MB/s | not published | 1.85M–2.2M IOPS at QD256 | [verified: [S33]] |
| Apple internal (M1 Ultra / M4 Max Mac Studio, 1 TB) | – | ≈5.3 / ≈5.1 GB/s (Blackmagic, user reports) | – | – | [verified (user reports): [S42]] |
| Apple M1 Max 1 TB (measured in a published SSD-offloading study) | – | >6 GiB/s, 1 GiB linear uncached read | – | needs ≥32 KiB chunks and several threads to be useful; a sparse row-load workload ran 1.25 GiB/s vs 6.1 GiB/s dense | [reported by a surveyed system] |

**Queue depth and request size.** At QD1 with 4 KiB reads, an NVMe drive delivers ≈1 % of its sequential rate (990 PRO: 90 MB/s vs 7450 MB/s). Throughput comes from depth and size. One DeepSeek-V3 expert at Q4_K_M is 44.0 M params × 4.82 bpw ≈ **26.5 MB** (three matrices of ≈8.8 MB) [estimate]. That is a large contiguous read, so expert loads behave like sequential I/O if the file layout keeps each expert contiguous and a few are in flight (QD 2–8). Expect 80–95 % of sequential spec [estimate; measure]. Neuron-level sparsity (offloading individual FFN rows rather than whole experts) is the opposite regime: KB-scale rows, where bundling to ≥32 KiB is mandatory [reported by a surveyed system].

**Thermal throttling.** A Gen5 drive without a heatsink (Crucial T700) hits its 81 °C throttle point within about a dozen seconds of heavy load and shuts down at about 90 °C. Writes have been measured falling to ≈1 GB/s after 15 minutes and ≈100 MB/s in another test, and long file transfers throttle [verified: [S35]]. No read-only sustained figure was found, so the read-side drop must be measured. MoE expert streaming is a sustained, hours-long read load, so the solver must cost NVMe at the **throttled steady-state rate**, measured with a 60–120 s read and re-checked from SMART temperature during operation. Laptop and mini-PC drives with no airflow are the worst case. No sustained-read figures were found for Apple internal SSDs [open question].

**Shared lanes.** On AM5, every chipset-attached M.2 shares a PCIe 4.0 x4 uplink (≈7.9 GB/s) [verified: [S31]]. RAID0 of two Gen4 drives on the chipset will not exceed one drive's worth. Prefer CPU-attached slots for the expert store.

**I/O paths by OS**

| OS | Path | What matters | Tag |
|---|---|---|---|
| Linux | `io_uring` + `O_DIRECT` into registered (fixed), pinned buffers; then `cudaMemcpyAsync` | Bypasses the page cache, so no double buffering of weights; `IOPOLL` is available only with O_DIRECT on block devices. Registered buffers mainly save CPU rather than raise device IOPS. | [verified: [S41]] |
| Linux + NVIDIA | GPUDirect Storage (cuFile) | Needs O_DIRECT for the direct path. Falls back to "compatibility mode" (CPU bounce buffer) when unsupported. Since CUDA 12.8, NVMe can use upstream PCI P2PDMA (kernel ≥6.2 on Ubuntu, open kernel driver ≥570). NVIDIA's "≥2× peak bandwidth" claim applies only to PCIe-switch topologies. The overview guide does not document GeForce support, so treat consumer cards as compat mode unless `gdscheck` says otherwise. | [verified: [S37]] |
| Windows | IoRing (`ioringapi.h`, Windows build 22000+) | Read, registered buffers, registered handles, cancel. Pair with `FILE_FLAG_NO_BUFFERING` for unbuffered reads [estimate: standard Win32 practice]. | [verified API surface: [S39]] |
| Windows | DirectStorage | GPU GDeflate decompression. 1.2 added an optional buffered mode. Microsoft's 1.1 post measured uncompressed streaming bus-limited at ≈3 GB/s on Gen3. Of interest only if weights are stored GDeflate-compressed; quantised weights compress poorly [estimate]. | [verified: [S38]] |
| macOS | `fcntl(F_NOCACHE)` per descriptor (nearest to O_DIRECT) | Does not purge pages already cached, so set it before the first read. Needs aligned buffers. On Apple silicon NVMe lands directly in unified memory, with no second hop to the GPU. | [verified: [S40]]; unified-memory point [reported by a surveyed system] |
| macOS | `mmap` of weights | Competes with the unified buffer cache. A surveyed home-cluster system reports a prefetch/release conflict, where later prefetches evict earlier ones. Prefer an explicit cache with F_NOCACHE reads. | [reported by a surveyed system] |
| macOS / APFS | – | No public data found on APFS or FileVault overhead for large sequential reads [open question; measure]. | – |

### 2.5 Network

Latency is quoted as one-way application-level where possible. Ping figures are round-trip and say so.

| Link | Nominal | Measured goodput | Small-message latency | Tag |
|---|---|---|---|---|
| Wi-Fi 6/6E (home, peer to peer via router) | – | 320–610 Mbps (four-device home testbed); 0.52 Gbps in a second setup | 3–7 ms link latency (same home testbed); ping min 1.9 / avg 4.1 ms RTT | [verified: [S46]]; home-testbed figures [reported by a surveyed system] |
| Wi-Fi tail (large campus study) | – | – | p90 ≈ 20 ms, p99 ≈ 250 ms | [verified: [S44]] |
| Wi-Fi 7, 6 GHz 320 MHz (phone, one router) | – | 3.68 Gbps at 2 ft, 2.13 Gbps at 40 ft, 0.92 Gbps at 120 ft; 5 GHz 160 MHz 2.08→1.53 Gbps | not reported | [verified: [S43]] |
| 1 GbE | 1 Gb/s | ≈0.94 Gbps | as 2.5 GbE | [estimate: 1500-MTU TCP overhead] |
| 2.5 GbE | 2.5 Gb/s | 2.35 Gbps (iperf3) | ping min 0.17–0.18 ms, avg 0.28–0.55 ms RTT | [verified: [S46]] |
| 10 GbE | 10 Gb/s | 9.42 Gbps | 210.9 µs send/recv over the Mac Studio's built-in 10 GbE (same study as the RDMA figures); 3–10 µs only with tuned or kernel-bypass stacks | [verified: [S46], [S52], [S56]] |
| 25/40/100 GbE | line rate | needs multiple streams or RDMA to approach line rate | kernel TCP tens of µs; RoCE a few µs | [estimate; no consumer measurement found] |
| DGX Spark ConnectX-7 | 200 Gb/s between two units | – | – | [verified: [S18]] |
| TB4/USB4 IP networking (Linux) | 40 Gb/s link | ≈9–12 Gbps single stream typical; 26 Gbps on a NUC 13 | ping avg ≈0.5 ms RTT | [verified (forum): [S53], [S46]] |
| USB4/TB between two Strix Halo boxes (Linux, link 20 Gb/s per direction) | 20 Gb/s | 8.85–9.41 Gbps single stream; 10.1 + 10.2 bidirectional | ping min 0.035–0.12 ms, avg 0.42–0.56 ms RTT; Soft-RoCE 66 µs (2 B), 858 µs worst | [verified: [S46]] |
| Thunderbolt 5 (spec) | 80 Gb/s each way; 120/40 boost; PCIe tunnel 64 Gb/s | – | – | [verified: [S47]] |
| TB5 IP (TCP) between Macs | – | "50–60 Gbps real-world" | ≈300 µs (the TCP path's latency, as quoted alongside the RDMA result) | [verified: [S48], [S49]] |
| **RDMA over TB5** (macOS 26.2+, JACCL) | – | ≈20 Gbps per link on one path, ≈100 Gbps over 6 trunked links; 3.5–3.8 GB/s file transfer on M3 Ultra; 7.4 GB/s sustained across 3 nodes | small-message floor ≈10 µs; send/recv 21.3 µs vs 210.9 µs on 10 GbE; "<50 µs vs 300 µs" | [verified: [S52], [S51], [S48]] |
| InfiniBand HDR | 200 Gb/s | – | 1.07 µs osu_latency (0–16 B); 1.2–1.4 µs ib_send_lat | [verified: [S54], [S55]] |

Operational facts for RDMA over TB5 [verified: [S48], [S50], [S51]]:

- JACCL supports only a fully connected topology, so every pair needs its own cable.
- RDMA is enabled per machine from Recovery with `rdma_ctl enable`; it cannot be done remotely.
- Clusters of four have been demonstrated.
- The MLX docs describe JACCL latency as "an order of magnitude lower than the ring backend" and "necessary for things like tensor parallelism".
- TCP over TB5 crashed both Macs under an HPL load in one test.
- A reported 100-memory-region limit in the `AppleThunderboltRDMA` driver may cap multi-node scaling [reported in [S51b]; unconfirmed].

A 2026 paper measuring RDMA over TB on M3 Ultra found bandwidth "far below nominal TB specifications". The paper's 64 MiB all-reduce algorithm bandwidth falls from 53.6 Gbps (2 nodes, 6 trunks, pooled) to 11.6 Gbps (4 nodes) [verified: [S52]]. **Plan on 2.5–4 GB/s per link and 10–25 µs per small message, not 10 GB/s and 1 µs.**

---

## 3. Cost model

### 3.1 Roofline and why decode is memory-bound

A decode step at batch B performs ≈2·N_act·B FLOPs and reads ≈b·N_act bytes of weights, plus KV. Arithmetic intensity is 2B/b FLOP/byte. A tier is compute-bound once 2B/b exceeds its ridge point F/W [Williams et al., [S74]].

| Device | F (assumed) | W | Ridge (FLOP/B) | Batch where Q4 decode turns compute-bound (b = 0.6) | Tag |
|---|---|---|---|---|---|
| RTX 4090, FP16 tensor, FP32 accumulate | 165 TFLOPS | 1008 GB/s | 164 | ≈49 | [estimate from [S10]] |
| Desktop CPU, AVX-512/VNNI int8 dot | 1–2 TFLOPS (not measured here) | 70–90 GB/s | 11–29 | ≈3–9 | [estimate] |
| M3 Ultra GPU | ≈28 TFLOPS FP32 (third-party derived) | 819 GB/s | ≈34 | ≈10 | [estimate; Apple publishes no FLOPS] |

Single-user decode (B = 1) is memory-bound everywhere. CPU tiers become compute-bound at small batches, and that matters for speculative verification and for multi-user mode on CPU-hosted layers.

### 3.2 Single-stream decode on one device with tiers

Each layer's weights live in exactly one tier: VRAM, host RAM, or NVMe with a RAM cache. Layers depend on each other in order, so the time for one token is a **sum** over tiers, not a max:

```
t_tok = Σ_i  B_i / W_i(exec)  +  n_dev_switch · t_switch  +  t_fixed
```

- `W_i(exec)` is the bandwidth of the processor that executes against tier i: GPU over VRAM, CPU over DRAM, or an iGPU over unified memory.
- `t_switch` is a CPU↔GPU activation hand-off. It is small: a 16 KiB activation over PCIe takes ≈1 µs of wire time plus ≈5–20 µs of synchronisation [estimate].

**Execute-in-place versus stream-to-GPU.** Tier-i weights can be computed on the CPU where they sit (EIP), or copied over PCIe each token and computed on the GPU (stream). For a block of P params at b bytes/param processing n tokens:

```
t_EIP    = max( P·b / W_ram ,   2·P·n / F_cpu )
t_stream = max( P·b / W_pcie ,  2·P·n / F_gpu )      (perfect prefetch overlap)
```

- **At n = 1**, the memory terms dominate. EIP wins whenever W_ram > W_pcie. Dual-channel DDR5 (76–122 GB/s measured) is 3–5× pinned PCIe 4.0 x16 (≈24 GB/s), so decode should always execute in place. The surveyed CPU/GPU hybrid MoE engines are built on exactly this rule [reported by a surveyed system].
- **As n grows**, CPU compute rises linearly while the PCIe copy is paid once per batch. The crossover is roughly
  `n* ≈ F_cpu · b / (2 · W_pcie)`, which is ≈12 tokens at F_cpu = 1 TFLOPS and ≈24 tokens at 2 TFLOPS (b = 0.6, W_pcie = 25 GB/s) [estimate].
  Above n*, which covers prefill and large-batch decode, stream the CPU-resident layers to the GPU in micro-batches. The bigger the micro-batch, the better the PCIe copy is amortised (§4a).

**Overlap.** A copy can overlap compute only when the bytes are known before they are needed:

- Dense weights for later layers: always known in advance.
- MoE experts: known only after that layer's router runs, unless predicted. A cheap predictor applies the next layer's gate to the current hidden state to guess next-layer experts [reported by a surveyed system].

With overlap, a streamed stage costs `max(compute, Σ streamed / W_link)`. Without it, the two times add.

### 3.3 The NVMe tier

For dense models, every byte that does not fit in RAM or VRAM must be read once per token:

```
t_nvme = B_overflow / W_nvme(sustained)
```

At W_nvme ≈ 6 GB/s, each GB of overflow adds ≈170 ms per token [estimate]. **The solver should refuse dense NVMe placements in interactive mode** unless the model is otherwise impossible to run. A smaller quant is almost always better: example (a) shows IQ3_XXS running 2.5× faster than Q4_K_M offloaded.

For MoE, only the routed experts that miss the RAM cache are read:

```
B_nvme = (1 − h_ram) · B_routed_per_token
t ≈ B_hot / W_gpu + h_ram · B_routed / W_ram_exec + (1 − h_ram) · B_routed / W_nvme      (serial, pessimistic)
t ≈ B_hot / W_gpu + max(h_ram · B_routed / W_ram_exec, (1 − h_ram) · B_routed / W_nvme)  (misses overlap hit compute, optimistic)
```

Where h_ram comes from:

- **Locality.** Adjacent tokens reuse some experts, and some experts recur in 2–4-token runs [reported by a surveyed system].
- **Measured LRU hit rates.** Community traces on a draft upstream libllama change (a GPU-resident LRU for host-offloaded experts) show top-10-of-512 routing at ≈67–69 % with 32–64 slots per layer, ≈81–90 % at 64–128, and ≈96 % at 216 slots (≈42 % of experts) [verified (draft change, one model): [S65]].
- **Policy matters as much as size.** A domain-aware two-stage offloading study reports a frequency-only baseline at ≈0.35 hit rate in decoding, rising past 0.5 once temporal locality is used [reported by a surveyed system (search summary)]. A study of speculative expert prefetching finds expert access "not consistent with temporal locality assumptions (e.g. LRU, LFU)". Its least-stale eviction policy cuts collision misses by up to 85× versus LRU and reaches >88 % hit rate with a cache of ≈5 % of VRAM (0.6 GB) on its test models [reported by a surveyed system]. Hit rates depend on workload, domain and eviction policy.

**h(c) is a runtime measurement, not a constant.** The solver should start from a conservative prior, h = cache fraction (uniform routing), and update it online.

### 3.4 Fixed per-token overhead and η

```
t_tok = Σ B / (η · W_peak) + t_fixed
```

On M3 Ultra (ggml Metal) the two-point fit gives η ≈ 0.79 and t_fixed ≈ 5 ms (§2.2) [estimate]. Each additional device or backend in the token path adds its own launch and sync overhead, which is part of why chaining many small devices hurts.

### 3.5 Prefill (time to first token)

Prefill is compute-bound once the prompt chunk exceeds the ridge batch.

```
FLOPs_prefill ≈ 2 · N_act · L  +  2 · n_layers · d_attn · L²        (causal attention: QKᵀ and AV, summed over positions)
t_prefill ≈ max( FLOPs / F_eff ,  Σ_streamed_bytes · ⌈L / ubatch⌉ / W_link ,  activation_bytes / β_hop )
```

The 2·N per token term is the standard forward cost [verified: Kaplan et al. [S72]]. The attention term is [estimate: per position t, 4·d_attn·t FLOPs per layer, summed over t ≤ L].

For Llama-3.3-70B (d_attn = 64 × 128 = 8192, 80 layers, 69.5 B params touched per token) [estimate]:

| L | Linear FLOPs | Attention FLOPs | Attention share |
|---|---|---|---|
| 4096 | 569 T | 22 T | 3.7 % |
| 8192 | 1139 T | 88 T | 7.2 % |
| 32768 | 4555 T | 1407 T | 23.6 % |

For MoE prefill, a chunk of ≥64 tokens touches nearly every expert in each layer (§3.9). That means:

- Prefill streams the whole routed pool, so prefill on a tiered MoE is bounded by `pool_bytes / W_link` per chunk.
- The chunk size has to be large to amortise it.

Mac M3 Ultra community runs on DeepSeek-V3-class 4-bit models show 58–189 tok/s prefill against ≈21 tok/s decode [verified (community): [S75]]. DGX Spark on gpt-oss-120B MXFP4 with ggml CUDA shows the same split from the compute-rich side: pp2048 1956 tok/s against tg32 60.6 tok/s [verified (community): [S27]]. Prefill is where compute-poor devices lose.

### 3.6 KV cache

Per token, per sequence:

```
MHA: 2 · n_layers · n_heads    · d_head · b_kv
GQA: 2 · n_layers · n_kv_heads · d_head · b_kv
MLA: n_layers · (d_c + d_rope) · b_kv          (only the compressed latent and the decoupled RoPE key are cached)
```

Sources: GQA [S61]; MLA [S60], [S59]. Each decode step reads the whole cache: `B_kv = L · KV_per_token`.

All values below are [estimate] from HF configs [S57], at BF16/FP16 KV:

| Model | Config | KV per token | 8 K | 32 K | 128 K |
|---|---|---|---|---|---|
| Llama-3.3-70B | 80 layers, 8 KV heads × 128 | 320 KiB | 2.5 GiB | 10 GiB | 40 GiB |
| Qwen3-235B-A22B | 94 layers, 4 KV heads × 128 | 188 KiB | 1.47 GiB | 5.9 GiB | 23.5 GiB |
| DeepSeek-V3 / Kimi-K2 | 61 layers, d_c = 512, d_rope = 64 | 68.6 KiB | 0.54 GiB | 2.1 GiB | 8.6 GiB |
| DeepSeek-V3 if it were MHA | 128 heads × (192 K + 128 V) | 4.77 MiB | 38 GiB | 153 GiB | – |

Implications:

- **Single user, long context.** At 32 K context, Llama-70B reads 10.7 GB of KV per token, a quarter of its Q4 weight bytes. KV belongs in the fastest tier, and its bandwidth cost has to sit in the objective.
- **Multi-user.** 32 users × 8 K × 320 KiB = 80 GiB of KV for Llama-70B, nearly twice its Q4 weights. KV capacity sets the concurrency limit. Unlike weights, KV is not shared between users, so batching does not amortise it.

### 3.7 Pipeline parallelism across devices

Stage *s* runs a contiguous layer range on device *d_s*. The hop carries the hidden state:

```
S_hop = h_model · b_act · B_tokens          (Llama-70B: 8192 × 2 B = 16 KiB per token; DeepSeek/Kimi: 7168 × 2 = 14 KiB)
t_hop = α + S_hop / β
```

| Hop, 16 KiB (one decode token) | Serialisation | Plus latency | Tag |
|---|---|---|---|
| 2.5 GbE | 56 µs | +≈100–250 µs | [estimate from [S46], [S52]] |
| Wi-Fi at 400 Mbps | 328 µs | +3–7 ms median, up to ≈250 ms at p99 | [estimate from [S44] and the home-testbed figures in §2.5] |
| RDMA TB5 at 2.5 GB/s | 6.6 µs | +10–25 µs | [estimate from [S52]] |

**Single stream.** One token visits every stage in turn, and the sampled token goes back to the head:

```
t_tok = Σ_s t_s + D · t_hop          (D hops in a ring)
```

Each device is busy only t_s / t_tok of the time. **PP never makes a single stream faster than the same bytes on the same devices.** It helps when it moves bytes from a slow tier (CPU DRAM, NVMe) onto another device's faster memory. For single-stream latency the optimum is greedy: fill devices in descending W_eff up to capacity, keeping layers contiguous.

**Multi-stream.** With M independent sequences in flight and D stages:

```
throughput ≈ min(M, D) / (Σ_s t_s + D·t_hop),   bounded above by  1 / max_s t_s(B_s)
```

So for throughput the solver **balances stage times**: give each device bytes in proportion to its W_eff, subject to capacity. That is a different placement from the latency-optimal one. With M ≥ D and batched stages, the bubble disappears. With micro-batching of a single long prefill, the pipeline bubble fraction is (D−1)/(m+D−1) for m micro-batches [verified: standard result; each stage idles for D−1 of the m+D−1 slots].

**Prefill hops** carry L × 16 KiB: 67 MB for a 4 K prompt. That takes 0.23 s on 2.5 GbE, 1.34 s on Wi-Fi at 400 Mbps and 27 ms on TB5 RDMA [estimate]. Chunked prefill hides this behind compute whenever a chunk's compute time exceeds its transfer time, which is true for slow devices (§4c).

### 3.8 Tensor parallelism

Standard transformer tensor parallelism splits each layer's matmuls across N devices: column-parallel for the QKV and MLP up projections, row-parallel for the attention output and MLP down projections. The forward pass then needs **2 all-reduces per transformer layer**, after the attention output projection and after the MLP down projection [verified: standard result]. Each all-reduce is `S = h_model · b_act · B` bytes.

```
ring all-reduce:  t_AR = 2(N−1)·α + 2(N−1)/N · S/β        (each rank moves 2(N−1)/N · S; [S71])
one-shot (full mesh, small N): t_AR ≈ α + (N−1)·S/β_link (+ local reduce)
per token: t_TP_comm = 2 · n_layers · t_AR
gain:      Δt = (B_weights / W_eff) · (1 − 1/N)
TP pays if Δt > t_TP_comm   →   α_eff,max ≈ (B_w/W_eff)(1 − 1/N) / (2·n_layers)   (decode, small S)
```

`α_eff,max` values below are [estimate]:

| Case | α_eff,max (N = 2) | α_eff,max (N = 4) |
|---|---|---|
| Llama-70B Q4 on M3 Ultra-class nodes (W_eff 645) | ≈206 µs | ≈309 µs |
| DeepSeek-V3 4-bit on M3 Ultra (W_eff ≈ 426) | ≈198 µs | ≈296 µs |

Read α_eff as wire latency **plus** the GPU→host→GPU synchronisation at every layer boundary, not ping time.

A surveyed multi-Mac cluster system published decode results on 4× M3 Ultra Mac Studio (two 512 GB, two 256 GB), MLX, RDMA over TB5, runs labelled tensor parallel (Dec 2025): DeepSeek-V3.1 at 32.5 tok/s and Kimi-K2-Thinking (native 4-bit) at 28.3 tok/s [reported by a surveyed system]. Back-solving those figures gives effective per-collective costs [estimate]:

- DeepSeek-V3.1 at 32.5 tok/s: 30.8 ms per token, of which ≈12.1 ms is weight reads at W_eff 426 GB/s ÷ 4. That leaves ≈18.7 ms over 122 collectives, **≈150 µs each**.
- Kimi-K2-Thinking at 28.3 tok/s gives **≈130 µs** each.

These are far above the ≈10–21 µs wire latency of [S52]. Most of the cost is synchronisation and collective implementation, which is where Syntropy can win: fuse the two collectives per layer where possible, and avoid host round-trips. The back-solve assumes that system ran TP over all layers with two collectives each; its code does not confirm this (see `multi-device-inference.md`).

**Over Ethernet or Wi-Fi.** Llama-70B TP=2 across a gaming PC and a Strix Halo box on 2.5 GbE, at α ≈ 150–250 µs, adds 24–40 ms per token. The decode time is still bounded by the slower device (Strix 23.2 GB / 179 GB/s ≈ 130 ms), so TP gives 5.9–6.5 tok/s against 6.5 tok/s for PP. On Wi-Fi (α ≈ 5 ms) TP adds 0.8 s per token [estimate]. **Rule: TP only between devices with RDMA-class links and similar W_eff.**

### 3.9 MoE: active bytes, routing locality, expert parallelism

Parameter split, from HF configs [S57], is [estimate]. "Hot" means attention, dense FFN layers, shared experts and the LM head, all touched every token.

| Model | Total | Active/token | Hot/token | Routed/token | Routed pool | One expert |
|---|---|---|---|---|---|---|
| DeepSeek-V3 (61 layers, 3 dense, 256 + 1 experts, top-8) | 671.0 B | 36.5 B | 16.1 B | 20.4 B | 653.9 B | 44.0 M |
| Kimi-K2 (61 layers, 1 dense, 384 + 1 experts, top-8) | 1026 B | 31.5 B | 10.4 B | 21.1 B | 1014.7 B | 44.0 M |
| Qwen3-235B-A22B (94 layers, 128 experts, top-8, no shared) | 235.1 B | 21.5 B | 7.3 B | 14.2 B | 227.1 B | 18.9 M |

Why MoE makes RAM and NVMe tiers viable:

- **Bytes per token track active, not total.** DeepSeek-V3 touches 5.4 % of its parameters per token.
- **The hot set is small and fits in VRAM.** For DeepSeek-V3 it is ≈9.7 GB at Q4_K_M, or ≈16 GB at 8-bit. Every token's always-touched bytes then run at VRAM speed, and only the routed bytes run at RAM or NVMe speed. The surveyed hybrid engines use exactly this split: attention, shared experts and KV on the GPU, routed experts on the CPU [reported by a surveyed system].
- **Experts are large contiguous objects** (≈25–27 MB at 4-bit), which suit sequential NVMe reads (§2.4).
- **Routing has exploitable skew and temporal locality** (§3.3).

Kimi-K2-Thinking is released natively as INT4 (group size 32) for routed experts only. Attention, shared experts, dense MLP and LM head stay BF16, for 594 GB total [verified: [S57]]. In that format the hot bytes are 10.4 B × 2 = **20.8 GB per token**, against 11.9 GB of routed bytes [estimate]. So the BF16 hot set dominates per-token traffic and does not fit beside a useful KV cache on a 24 GB card. A solver that treats "MoE = small active set" without looking at per-tensor dtypes gets this wrong.

**Batching and the union of experts.** With B tokens each choosing k of E experts, uniform routing touches `U(B) = E · (1 − (1 − k/E)^B)` distinct experts per layer. Uniform routing maximises U for given marginals, so this is a pessimistic bound. Values for DeepSeek-V3 (E = 256, k = 8) are [estimate]:

| B | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Distinct experts/layer | 8 | 15.8 | 30.5 | 57.4 | 102 | 163 | 222 |
| Routed bytes vs B = 1 | 1× | 1.97× | 3.8× | 7.2× | 12.8× | 20.4× | 27.8× |

On one M3 Ultra (W_eff ≈ 426 GB/s, MLX 4-bit), aggregate decode goes 20.7 → 32 → 37 → 44 → 56 tok/s at B = 1, 4, 8, 16, 32. Dense Llama-70B on the same machine goes 14 → 49 → 81 → 121 → 161 [estimate; KV at 8 K included for Llama]. **MoE throughput mode scales with nodes holding experts in fast memory, not with batch depth**, and batching on RAM- or NVMe-tier experts multiplies slow-tier traffic.

**Expert parallelism (EP)** puts different experts on different devices. Each MoE layer then needs a dispatch and a combine. These are small messages (k × h_model × b_act per token), so they are latency-bound like TP: ≈2 × n_moe_layers × α_eff per token, about 116 hops for DeepSeek-V3. DeepSeek's own deployment limits each token to experts on at most 4 nodes, uses redundant copies of hot experts, and uses direct point-to-point IB transfers in decode. That is on 50 GB/s IB against 160 GB/s NVLink [verified: [S59]]. **For home clusters, prefer layer-contiguous placement (PP) with each layer's attention co-located with its experts.** EP over Ethernet or Wi-Fi turns every layer into a network round-trip. EP is viable only over RDMA-class links (TB5 mesh).

### 3.10 Speculative decoding as slow-tier amortisation

With per-token acceptance probability `a` and draft length γ, the expected tokens per target pass is `(1 − a^{γ+1}) / (1 − a)` [verified formula: Leviathan et al. [S67]]. Then:

```
speedup ≈ E[tokens] · t_target(1) / ( γ · t_draft + t_target(γ+1) )
```

- **Dense target, memory-bound:** `t_target(γ+1) ≈ t_target(1)` while (γ+1) stays under the ridge batch of every tier the target touches. For a CPU-hosted tier that ridge is small, ≈3–9 tokens at Q4 (§3.1), so γ ≈ 4–6 is the useful range on hybrid desktops. The draft must live in VRAM so that `t_draft ≪ t_target`.
  - a = 0.8, γ = 4 → 3.36 tokens per pass; γ = 8 → 4.33 [estimate]. A surveyed speculative-offloading system reports 4–6 tok/s for 50 B+ models at 4-bit with RAM offload on consumer GPUs [reported by a surveyed system]. A surveyed home-cluster system reports Llama 3.3-70B going from 803 to 442 ms/token with a 3B draft on its head device (4 Linux nodes, 8 GiB RAM and 600 MB/s disk each, two with an RTX 4090 capped at 11 GiB) [reported by a surveyed system].
- **MoE target:** verifying γ+1 tokens touches U(γ+1) experts. For DeepSeek-V3, 5 tokens touch ≈4.7× the routed bytes of one, so tiered MoE gains far less. Native MTP heads (DeepSeek-V3: 85–90 % second-token acceptance, 1.8× tokens/s) [verified: [S59]] are the cheap case, with γ = 1 and a 2-token union of ≈1.97× routed bytes. That is best used when routed experts are in fast memory.

### 3.11 Mode summary

| Quantity | Single-user latency mode | Multi-user throughput mode |
|---|---|---|
| Objective | min t_tok (and TTFT) | max Σ tok/s subject to a per-user floor |
| Device fill | greedy by W_eff, contiguous layers | stage times balanced (bytes ∝ W_eff) |
| Batch | 1 (+ speculative γ) | up to the ridge batch of the bottleneck tier, or the KV capacity limit |
| KV | small, fastest tier | capacity-binding; often decides concurrency |
| TP / EP | only with RDMA-class α and balanced devices | rarely; PP with M ≥ D streams removes bubbles more cheaply |
| MoE on slow tiers | yes, with an expert cache | poor: union growth multiplies slow-tier bytes |
| Speculative decoding | yes for dense; MTP for MoE | usually off; batch already fills compute |
| Wi-Fi nodes | avoid on the critical path (tail latency) | acceptable as extra replicas or stages if p99 is costed |

---

## 4. Worked examples

All results are **[estimate]**. The inputs are verified or reported figures from §2. The efficiency factors are stated per example and are the main uncertainty.

### (a) 70B dense Q4 on one RTX 4090 + 64 GB DDR5 + PCIe 4.0 NVMe

**Inputs**

- Model: Llama-3.3-70B Q4_K_M, 42.5 GB [verified: [S58]]. That is 70.55 B params and 4.82 bpw. Per token it touches all layers plus the LM head, 69.5 B params [estimate].
- GPU: 1008 GB/s peak [verified: [S10]]. Assume W_eff = 800 GB/s (η ≈ 0.8, in line with the dense-model anchors in §2.2).
- RAM: DDR5-6000 dual channel, 96 GB/s peak, 90 GB/s measured on an Intel platform [verified: [S22]]. Assume CPU matvec W_eff = 70 GB/s, with a range of 55–85. On a single-CCD Ryzen use ≤60.
- PCIe 4.0 x16 pinned: 25 GB/s [verified: ≈24 in [S30]].
- NVMe: 990 PRO-class, 7.45 GB/s spec [verified: [S32]], ≈6 GB/s sustained [estimate].

**Step 1, VRAM budget.**

- KV at 8 K context: 327,680 B × 8192 = 2.68 GB.
- Compute buffers ≈1.5 GB and CUDA context ≈0.5 GB [estimate].
- Weights in VRAM: 24 − 2.68 − 1.5 − 0.5 = **19.3 GB**. That is 45 % of the weights, ≈36 of 80 layers at ≈0.53 GB per layer.
- Weights in RAM: 42.5 − 19.3 = **23.2 GB**. This fits in 64 GB alongside the OS, so NVMe is used only for loading.

**Step 2, decode (execute in place).**

- GPU: 19.3 / 800 = 24.1 ms.
- CPU: 23.2 / 70 = 331.4 ms.
- KV: 2.68 GB / 800 = 3.4 ms at full 8 K context.
- t_tok ≈ **359 ms → 2.8 tok/s**. The range is 2.2–3.4 for W_ram of 55–85.
- Hard upper bound at 100 % of both peaks: 19.3/1008 + 23.2/96 → 3.8 tok/s.
- **Bottleneck: host DRAM, 93 % of the time.**

**Step 3, the alternative (stream RAM layers to the GPU).** 23.2 / 25 = 928 ms → 1.05 tok/s. That is 2.7× worse, so streaming is the wrong choice for decode.

**Step 4, prefill of a 4 K prompt.**

- FLOPs: 2 × 69.5 B × 4096 + 2 × 80 × 8192 × 4096² = 569 + 22 = 591 TFLOP.
- GPU compute at 100–165 TFLOPS effective (165 = FP16/FP32-accumulate dense peak): 3.6–5.9 s.
- PCIe streaming of the RAM-resident 23.2 GB, per micro-batch: 0.93 s. With 512-token micro-batches that is 8 × 0.93 = 7.4 s, which **exceeds** compute. With 2048-token micro-batches it is 1.9 s, which is hidden.
- Running the 44 CPU layers' prefill on the CPU instead (38.5 B params × 2 × 4096 FLOPs at 1.5 TFLOPS) takes ≈210 s.
- **TTFT ≈ 4–6 s**, with large micro-batches and every prefill FLOP on the GPU.

**Step 5, levers in order of payoff.**

1. **Smaller quant.** IQ3_XXS is 27.5 GB [verified: [S58]]. RAM part = 8.2 GB, so t = 24.1 + 117 = 141 ms → **7.1 tok/s** (2.5×). This is a quality trade-off the user must accept.
2. **Speculative decoding** with a 1–3 B draft in VRAM, γ = 4–6. The CPU tier can verify 5 tokens nearly for free: 5 × 2 × 38.5 B = 0.38 TFLOP ≈ 0.2–0.4 s at 1–2 TFLOPS, on the order of the 0.33 s memory time. At a = 0.7–0.8, expect 2.8–3.4 tokens per pass → ≈6–9 tok/s [estimate; acceptance is model- and prompt-dependent].
3. **Shrink the KV reservation** with a q8 KV cache to free ≈1.3 GB of VRAM for ≈2.5 more layers (+≈3 % speed).
4. **NVMe:** irrelevant here. At Q8_0 (75.0 GB [verified: [S58]]) the RAM side would need ≈56 GB plus the OS in 64 GB, which is fragile. Each GB that spills to NVMe costs ≈170 ms per token.

**Placement a good solver should choose:** the first ~36 layers' worth of bytes in VRAM, filled by layer with KV and scratch reserved first; the remaining layers on the CPU, executing in place; micro-batch ≥2048 for prefill, streamed to the GPU; a speculative draft model in VRAM if available. Never mmap from NVMe at decode time.

### (b) DeepSeek-V3 / Kimi-K2-class MoE: two Mac Studios over TB5 vs one PC with 1 GPU + 128 GB RAM + NVMe

**b1. Two M3 Ultra 512 GB Mac Studios, RDMA over TB5**

**Model bytes.** DeepSeek-V3 MLX 4-bit at ≈4.5 bpw is 671 B × 0.5625 ≈ 377 GB, so it fits on one 512 GB node. Per token it reads 36.5 B × 0.5625 = **20.5 GB**.

**Single node, no network.**

- Bound: 819 / 20.5 = 39.9 tok/s.
- Community measurement: ≈21 tok/s on V3-0324 MLX 4-bit [verified (community): [S75]], so η ≈ 0.53.
- Add KV at 32 K context (2.3 GB per token read): +5.4 ms, about 11 %.

**TP = 2** (each node reads 10.3 GB/token at W_eff = 426 GB/s, so 24.1 ms; plus 122 collectives × α_eff):

| α_eff per collective | Comm per token | Total | tok/s | Note |
|---|---|---|---|---|
| 20 µs | 2.4 ms | 26.5 ms | 37.7 | wire latency only |
| 40 µs | 4.9 ms | 29.0 ms | 34.5 | – |
| 150 µs | 18.3 ms | 42.4 ms | ≈23 | as back-solved from the surveyed cluster's results, §3.8 |
| 300 µs | 36.6 ms | 60.7 ms | 16.5 | TCP-class; slower than one node |

**PP = 2**: ≈20.7 tok/s single stream (no gain), with 2× capacity for KV and for concurrent streams.

**Kimi-K2-Thinking**, native INT4 routed + BF16 hot, 594 GB [verified: [S57]]. It does not fit on one 512 GB node.

- Per token it reads 20.8 + 11.9 = **32.7 GB**.
- PP = 2: 32.7 / 426 = 76.7 ms → **13 tok/s**.
- TP = 2: 38.4 ms + 4.9–18.3 ms → **17–23 tok/s**.
- Reference: the surveyed cluster system reported 28.3 tok/s on **4** nodes (M3 Ultra, MLX, RDMA over TB5, labelled tensor parallel) [reported by a surveyed system].
- Re-quantising the BF16 hot set to 8-bit would cut per-token bytes by ≈10 GB, so the gain from hot-set precision is larger than from TP.

**M5 Ultra (1.2 TB/s, shipping since Sept 2026 [verified: [S3], [S4]])** scales every bandwidth term by 1.5×. Single-node DeepSeek-4-bit bound ≈ 60 tok/s, ≈31 tok/s at η = 0.53 [estimate; no measurement found].

**Bottleneck:** unified-memory bandwidth on one node, then per-collective latency and sync once TP is used.

**Placement:**

- Latency mode: one node if the model fits; TP = 2 over RDMA only when the probed α_eff is ≤≈100 µs. If the model doesn't fit, use TP over RDMA when α_eff is measured below α_eff,max, else PP.
- Throughput mode: PP or replicas, with M ≥ D concurrent streams.

**b2. One PC: RTX 4090 24 GB + 128 GB DDR5-6000 dual channel + PCIe 4.0 NVMe**

**Model:** DeepSeek-V3 Q4_K_M, 404.4 GB [verified: [S58]], 4.82 bpw.

**Step 1, split by role.**

- Hot set: 16.1 B × 0.6027 B = **9.7 GB** on the GPU, plus KV (32 K = 2.1 GiB) and buffers. This fits in 24 GB.
- Routed pool: 653.9 B × 0.6027 = **394 GB**.
- RAM available for the expert cache: ≈115 GB, i.e. 29 % of the pool. The rest lives on NVMe.
- Routed bytes per token: 20.4 B × 0.6027 = **12.3 GB**.

**Step 2, decode as a function of RAM hit rate h** (W_gpu 800, W_ram_exec 70, W_nvme 6 GB/s sustained):

| h | NVMe GB/token | Serial (tok/s) | Misses overlapped (tok/s) |
|---|---|---|---|
| 0.29 (uniform, no locality) | 8.7 | 0.66 | 0.68 |
| 0.60 | 4.9 | 1.07 | 1.2 |
| 0.80 | 2.5 | 1.8 | 2.4 |
| 0.90 | 1.2 | 2.7 | 4.6 |
| 0.95 | 0.6 | 3.6 | 5.6 |
| 1.00 (everything in RAM) | 0 | **5.3** | 5.3 |

**Bottleneck:** NVMe at h ≤ 0.9, then **dual-channel DRAM even at h = 1**: 12.3 GB / 70 GB/s = 176 ms. A PCIe 5.0 drive at ≈12 GB/s sustained lifts h = 0.85 from 2.1 to ≈3.2 tok/s, but it cannot pass the DRAM cap.

**Calibration.** A surveyed CPU/GPU hybrid MoE engine, with *all* experts in 1 TB of 8-channel DDR5-4800 (two Xeon Gold 6454S sockets) plus an RTX 4090D 24 GB, running DeepSeek-V3 Q4_K_M, reported 12.2 tok/s (8 experts) and 13.7 tok/s (6 experts) [reported by a surveyed system]. The difference from this desktop is memory channels (8 per socket vs 2), not the GPU. The same report gives 4.5 tok/s for a stock ggml/libllama build on that machine [reported by a surveyed system].

**Q2_K (244 GB) [verified: [S58]].**

- Routed per token: 7.4 GB. Cache covers 48 % of the pool.
- h = 0.9 → 4.4–7.6 tok/s; h = 0.95 → 5.9–9.3 tok/s.
- This is the realistic interactive setting on a 128 GB desktop, at a quality cost.

**Kimi-K2 on this PC:** 594 GB native, cache 20 % of pool. The BF16 hot set (20.8 GB) won't fit next to KV on 24 GB, so it needs a 32 GB card or an 8-bit hot set. Expect ≈1–2 tok/s. Not a recommended target.

**Prefill** on b2 is dominated by streaming the expert pool through PCIe per chunk: 394 GB / 25 GB/s = 16 s per chunk if every expert is touched. The alternative is computing on CPU AMX/AVX-512; the surveyed hybrid engine reports 186–287 tok/s prefill this way on the same dual-socket Xeon testbed [reported by a surveyed system]. A desktop without AMX should expect far less [estimate]. Large chunks (≥2–4 K tokens) are mandatory.

*Note, 2026-10-20.* The 394 GB / 25 GB/s figure treats the whole pool as if it streamed from RAM. Only the RAM-cached share does. The NVMe-resident pool bytes stream at `W_nvme` (6 GB/s sustained here), not at the PCIe rate, once per chunk, or once per layer-major group on a stage that prefills layer-major over an `Nvme`-homed pool. RFC-0012 §4.4 puts this at ≈281 GB, ≈47 s per chunk [estimate], and vision SC-18 counts these bytes at `W_nvme`.

**Placement a good solver should choose for b2:**

- GPU holds the hot set (attention, dense layers, shared experts, LM head, possibly at 8-bit), the KV cache, and a small VRAM expert cache for the hottest experts.
- RAM holds routed experts ordered by measured popularity.
- NVMe holds everything, read with O_DIRECT in whole-expert extents.
- Next-layer expert prediction issues prefetches.
- Single-user mode only; batch > 1 multiplies NVMe traffic (§3.9).

**Comparison:** two M3 Ultras run DeepSeek-V3-class at **≈20–35 tok/s**; the single PC at **≈1–3 tok/s (Q4)** or **≈4–9 tok/s (Q2)**. The 10× gap is memory bandwidth per byte of model (819 GB/s per node vs 70 GB/s DDR5 plus NVMe), not compute.

### (c) Mixed home cluster: gaming PC + MacBook + mini PC, 2.5 GbE vs Wi-Fi

**Devices (assumed)**

| Device | Usable fast memory | W_eff | Tag |
|---|---|---|---|
| Gaming PC: RTX 4090 + 64 GB DDR5-6000 | 19.3 GB weights in VRAM (as in (a)); RAM W_eff 70 | VRAM W_eff 800 | as (a) |
| MacBook Pro M4 Pro 48 GB | ≈32 GB GPU-usable after the wired cap | 273 × 0.7 = 191 GB/s | [verified peak: [S8]; estimate η] |
| Mini PC: Ryzen AI Max+ 395, 128 GB | ≈96–100 GB GPU-usable | 256 × 0.7 = 179 GB/s, iGPU path | [verified peak: [S19]; estimate η] |

Links:

- 2.5 GbE: α ≈ 0.2 ms per hop.
- Wi-Fi: α ≈ 5 ms median, ≈250 ms p99 [verified ranges: [S44], [S46]; median also from the home-testbed figures in §2.5].
- The MacBook needs a USB-C 2.5 GbE adapter or a TB cable to be on the wired fabric.

**c1. Llama-3.3-70B Q4_K_M (42.5 GB), single stream**

| Placement | Compute per token | 2.5 GbE | Wi-Fi median | Wi-Fi p99 hop |
|---|---|---|---|---|
| Gaming PC alone (VRAM + CPU, as (a)) | 359 ms | 2.8 tok/s | – | – |
| Strix Halo alone (42.5/179) | 237 ms | 4.2 tok/s | – | – |
| PP: 4090 (19.3 GB) → Strix (23.2 GB) | 24.1 + 129.6 = 153.8 ms | 2 hops → **6.5 tok/s** | 6.1 tok/s | 1.5 tok/s |
| PP: 4090 → MacBook (12.0) → Strix (11.2) | 24.1 + 62.6 + 62.6 = 149.4 ms | 3 hops → **6.7 tok/s** | 6.1 tok/s | – |

- Adding the MacBook barely helps a single stream (6.5 → 6.7). It takes bytes from a device of about the same W_eff, which confirms §3.7.
- **Bottleneck:** the unified-memory devices' bandwidth (≈180–190 GB/s). The network is 0.3 % of the per-token time on 2.5 GbE and ≈6–9 % at Wi-Fi median, but a p99 Wi-Fi hop quadruples the token time.
- TP over 2.5 GbE gives 5.9–6.5 tok/s, no better than PP (§3.8); over Wi-Fi it is ≈1 tok/s.

**c2. The same cluster, throughput mode, Llama-70B**

- Balanced stages: 4090 VRAM 19.3 GB (24 ms); MacBook 12.0 GB (63 ms); Strix 11.2 GB (63 ms).
- With ≥3 streams in flight: 1 / 63 ms ≈ **16 tok/s aggregate** at batch 1 per stage.
- Batching each stage further raises this until the Strix iGPU becomes compute-bound. At an assumed 15 TFLOPS effective (third-party FP16 peaks of 24–30 TFLOPS [S19b]), that happens around B ≈ 25 [estimate].
- KV caps concurrency first: each 8 K stream costs 2.7 GB of KV, spread across the stages that hold its layers.
- Activation hops at B = 32 are 512 KiB: 1.8 ms on 2.5 GbE, ≈15 ms on Wi-Fi. Still small against 63 ms stages, so **Wi-Fi is acceptable for throughput mode and poor for interactive latency.**

**c3. A MoE model fits this cluster better: Qwen3-235B-A22B Q4_K_M (142.2 GB [verified: [S58]])**

- Per layer: attention 43 MB plus 91 MB of routed bytes per token, so 134 MB per token per layer. Full layer footprint is 1.50 GB [estimate].
- Placement (latency mode, greedy by W_eff, layers contiguous, attention co-located with experts):
  - 4090: 13 layers + LM head, 3.0 ms per token
  - MacBook: 21 layers, 14.8 ms
  - Strix: 60 layers, 45.0 ms
- At η = 0.7: 62.8 ms → **15.8 tok/s on 2.5 GbE**, 12.9 tok/s at Wi-Fi median.
- At the MoE-typical η = 0.55: 12.4 and 10.5 tok/s.
- **A 235 B MoE runs about 2× faster than a 70 B dense model on the same three machines**, because per-token bytes (13 GB) are a third of Llama-70B's. For heterogeneous, bandwidth-poor clusters, the solver (and the model recommendations Syntropy shows users) should favour MoE models.

**Placement a good solver should choose for (c):**

- Latency mode: fill the highest-W_eff memory first. Keep layers contiguous, and keep each MoE layer's attention with its experts.
- Use the PC's CPU DRAM only as a last resort, since it is the slowest exec tier at 70 GB/s.
- Keep Wi-Fi devices off the critical path in latency mode, or include them only when their p99 is costed.
- Use TP nowhere on this fabric.
- Throughput mode: rebalance stages so bytes are proportional to W_eff, and admit streams up to the KV limit.

---

## 5. Implications for the placement solver

### 5.1 Variables, ranked

**First order (they decide the placement):**

1. `C_{d,i}`: usable capacity per device and tier. This means GPU-wired caps on macOS, the Strix Halo GPU allocation, VRAM minus KV, scratch and context, and RAM minus OS minus pinned buffers.
2. `W_{d,i}(exec)`: effective bandwidth of the kernel that will actually run, per model architecture (dense GEMV vs MoE expert GEMV). The spread is 0.24–0.8 × peak (§2.2).
3. Per-tensor bytes at the chosen quantisation, including mixed dtypes (Kimi-K2's BF16 hot set), split into hot and routed.
4. KV bytes per token × expected context × concurrency.
5. `α_eff` per link, including the sync cost on each side, plus a **p99**. β is first order only for prefill hops and large batches.
6. `W_nvme` sustained (throttled) at the real extent size and queue depth; PCIe pinned H2D.
7. For MoE: the expert hit-rate curve h(cache size) per tier, learned online.

**Second order (correct for them, but they rarely flip a decision):** CPU↔GPU activation hand-offs (≈µs), sampling and tokenisation, DRAM latency, pageable-vs-pinned for small control messages, NIC interrupt coalescing, and the attention-FLOP share below ≈8 K context.

**Third order or ignore:** link β for single-token decode hops (16 KiB) on anything ≥1 GbE.

### 5.2 Measure at runtime versus assume

| Quantity | Measure or assume | Why |
|---|---|---|
| W_eff per device/tier/kernel | **Measure** | η spans 0.24–0.8 and depends on kernel, quant and architecture; DIMM population, single-CCD fabric limits and power states make spec values wrong |
| t_fixed per backend | **Measure** | 5 ms on ggml Metal in one fit; different on CUDA, ROCm and Vulkan |
| PCIe H2D pinned/pageable | **Measure** | x8/x4 slots, chipset-attached GPUs and eGPU tunnels |
| NVMe sustained read | **Measure, then re-check** | thermal throttling; chipset-shared lanes; FileVault/APFS unknowns |
| α_eff, β, p99 per link | **Measure continuously** | Wi-Fi varies minute to minute; RDMA availability depends on OS state |
| F_eff (prefill) | Measure briefly, assume between probes | prefill planning only |
| Expert hit curve h(c) | **Learn online** | workload-dependent (§3.3) |
| Spec acceptance a | Learn online per (draft, target, task) | – |
| Peak specs (§2 tables) | Assume, as sanity bounds | a probe result under ≈50 % of spec flags misconfiguration |

### 5.3 Startup microbenchmarks (target: under 15 s per node, all parallel across nodes)

1. **Memory W_eff per exec engine** (≈2 s): run the *actual* dequant-GEMV kernel for the chosen quant over a 1–2 GB buffer that exceeds the last-level cache, on the GPU over VRAM, the CPU over DRAM (all threads, NUMA-pinned) and the iGPU over unified memory. Report GB/s of weight bytes consumed, not raw copy bandwidth. On MoE models, also run an expert-sized (≈25 MB) gather-GEMV.
2. **t_fixed** (≈0.5 s): time a tiny-model forward of 1 layer × n, and fit the intercept.
3. **PCIe** (≈0.5 s): pinned H2D and D2H at 64 MiB, with a pageable read for comparison.
4. **NVMe** (≈5 s quick, then a background 60–120 s sustained run during model load):
   - O_DIRECT / F_NOCACHE / unbuffered reads at 4 KiB QD1, then 1, 8 and 32 MiB at QD 1, 4 and 16, against the model file itself (also warms the cache).
   - Record the SMART temperature before and after.
   - Re-measure during the first minutes of real load, and store the throttled rate.
5. **Links** (≈2 s per pair, pairs in parallel):
   - Ping-pong at 64 B, 16 KiB, 1 MiB and 64 MiB. α comes from the small sizes; β from the large ones.
   - Send ≥200 samples for a p50/p99.
   - Repeat through the actual collective path (GPU buffer → wire → GPU buffer) to get α_eff including sync.
   - Probe RDMA (JACCL / ibverbs) when present, and fall back to TCP otherwise.
6. **Compute F_eff** (≈1 s): a GEMM at prefill shapes in the chosen precision.
7. **Continuous:** per-hop latency histograms, NVMe throughput and temperature, and expert hit rates. Re-plan when a measured term moves more than ≈20 % for longer than a window. Use hysteresis: a re-shard moves tens to hundreds of GB, e.g. 377 GB at 3.5 GB/s ≈ 108 s, or ≈5.4 min at 10 GbE goodput (9.4 Gbps).

### 5.4 How the decision changes by mode

- **Latency mode:** minimise `Σ B/W + Σ hops + t_fixed`.
  - Greedy fill by W_eff with capacity constraints and contiguous layers. This is a knapsack-like ILP; a surveyed home-cluster system solves a similar ILP with HiGHS in ≈10–12 ms for 4–32 devices [reported by a surveyed system].
  - Allow TP only on links with α_eff < α_eff,max (§3.8).
  - Use speculative decoding for dense models, and MTP for MoE.
  - Penalise Wi-Fi hops by p99, not mean.
- **Throughput mode:** maximise aggregate tokens/s.
  - Minimise the max stage time, so bytes go proportional to W_eff.
  - Choose batch per stage up to the ridge point or the KV limit.
  - Account for MoE union growth.
  - Add replicas when capacity allows, since a replica avoids every hop.
  - Admission control on KV capacity.
- **Prefill vs decode:** prefill may want a different execution site, streaming CPU-resident weights to the GPU in big chunks. On clusters it may be worth routing prefill to the most compute-rich node and shipping KV, if the link carries KV faster than the slow node can compute it. The KV for a 4 K prompt is 1.3 GB on Llama-70B, ≈4.5 s at 2.5 GbE, so mostly not worth it off RDMA [estimate].

---

## 6. Open questions

1. **α_eff on Apple RDMA.** Is the ≈130–150 µs per collective back-solved from the surveyed cluster's results (§3.8) wire, Metal sync, or implementation? A direct `all_sum` latency sweep on two Macs settles it, and it sets whether TP over TB5 is worth building first.
2. **Strix Halo GPU-side DRAM bandwidth.** CPU-side is measured at 124 GB/s; GPU-side was not found.
3. **RTX 50-series pinned H2D on PCIe 5.0 x16.** No measured figure found; the estimate is 48–52 GB/s.
4. **Expert hit-rate curves for DeepSeek/Kimi** under real chat workloads, given DeepSeek's load-balanced routing. The curves cited are from a 512-expert Qwen-family model and are contested in the literature.
5. **Sustained NVMe read under thermal load** in laptops and mini PCs, and on Apple internal SSDs with FileVault.
6. **macOS GPU wired-limit behaviour** across macOS 26.x, and whether raising it is acceptable for an end-user product.
7. **GPUDirect Storage on GeForce.** The documentation is silent. Probe with `gdscheck` on target machines before designing around it.
8. **Windows USB4/Thunderbolt networking throughput and latency.** No data found.
9. **Wi-Fi 7 MLO latency tails** in homes. Only vendor-grade claims were found; measure in alpha testing.

---

## Sources

Hardware, standards, model and measurement sources only. Figures from surveyed inference systems are tagged [reported by a surveyed system] in the text and have no entry here.

- [S1]: Apple Newsroom, *Apple introduces MacBook Pro with all-new M5 Pro and M5 Max* (Mar 2026). https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/
- [S2]: Apple Support, *MacBook Pro (14-inch, M5 Pro or M5 Max) – Tech Specs*. https://support.apple.com/en-us/126318
- [S3]: Apple Newsroom, *Apple introduces M6 and M5 Ultra* (Aug 2026). https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/
- [S4]: Apple Support, *Mac Studio (M5 Max or M5 Ultra) – Tech Specs*. https://support.apple.com/en-us/128107
- [S5]: Wikipedia, *Apple M1*. https://en.wikipedia.org/wiki/Apple_M1
- [S6]: Wikipedia, *Apple M2*. https://en.wikipedia.org/wiki/Apple_M2
- [S7]: Wikipedia, *Apple M3*. https://en.wikipedia.org/wiki/Apple_M3
- [S8]: Wikipedia, *Apple M4*. https://en.wikipedia.org/wiki/Apple_M4
- [S9]: Wikipedia, *Apple M5*. https://en.wikipedia.org/wiki/Apple_M5
- [S10]: NVIDIA, *Ada Lovelace GPU Architecture* whitepaper (RTX 4090 comparison table). https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
- [S11]: Wikipedia, *GeForce RTX 50 series*. https://en.wikipedia.org/wiki/GeForce_RTX_50_series
- [S12]: TechPowerUp GPU database, *GeForce RTX 3090*. https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622 ; Guru3D. https://www.guru3d.com/review/geforce-rtx-3080-and-3090-what-we-know/page-2/
- [S13]: Tom's Hardware, *AMD Radeon RX 7900 Series Deep Dive*. https://www.tomshardware.com/news/amd-radeon-rx-7900-series-deep-dive ; Notebookcheck. https://www.notebookcheck.net/AMD-Radeon-RX-7900-XTX-GPU-Benchmarks-and-Specs.674159.0.html
- [S14]: Wikipedia, *Radeon RX 9000 series*. https://en.wikipedia.org/wiki/Radeon_RX_9000_series
- [S15]: SemiAnalysis InferenceX chip pages. https://inferencex.semianalysis.com/chips/h200 , https://inferencex.semianalysis.com/chips/b200-vs-h100 , https://inferencex.semianalysis.com/chips/mi300x , https://inferencex.semianalysis.com/chips/mi355x
- [S16]: AMD, *Instinct MI350 series*. https://www.amd.com/en/products/accelerators/instinct/mi350.html
- [S17]: NVIDIA, *DGX Spark Hardware Overview*. https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- [S17a]: Apple Newsroom, *Apple reveals M3 Ultra* (Mar 2025). https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/ ; OWC on six TB5 ports. https://www.owc.com/blog/apple-redefines-the-pro-workstation-with-the-new-m3-ultra-mac-studio
- [S18]: StorageReview, *NVIDIA DGX Spark review*. https://storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops
- [S19]: Notebookcheck, *AMD Ryzen AI Max+ 395*. https://www.notebookcheck.net/AMD-Ryzen-AI-Max-395-Processor-Benchmarks-and-Specs.942323.0.html ; Evetech, *Strix Halo … 96GB of its 128GB into VRAM*. https://evezone.evetech.co.za/daily-drop/amd-strix-halo-arrives-how-the-ryzen-ai-max-395-turns-96gb-of-its-128gb-into-vram-for-70b
- [S19b]: TechPowerUp GPU database, *Radeon 8060S* (FP16 23.9 TFLOPS at 2335 MHz). https://techpowerup.com/gpu-specs/radeon-8060s.c4270
- [S20]: Chips and Cheese, *AMD's Chiplet APU: An Overview of Strix Halo*. https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix
- [S21]: Chips and Cheese, *AMD's Ryzen 9950X: Zen 5 on Desktop*. https://chipsandcheese.com/p/amds-ryzen-9950x-zen-5-on-desktop ; *Pushing AMD's Infinity Fabric to its Limits*. https://chipsandcheese.com/p/pushing-amds-infinity-fabric-to-its ; *AMD's EPYC 9355P*. https://chipsandcheese.com/p/amds-epyc-9355p-inside-a-32-core
- [S22]: Igor's Lab, Core Ultra 7 265K DDR5 scaling (theoretical vs measured bandwidth). https://www.igorslab.de/en/?p=322632
- [S23]: Puget Systems, *AMD EPYC 9005* (12 channels, DDR5-6400). https://www.pugetsystems.com/parts/CPU/AMD-EPYC-9555-3-2GHz-64-Core-360W-15884/
- [S24]: Qualcomm, *Snapdragon X Elite*. https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-x-elite
- [S25]: Notebookcheck, *Snapdragon 8 Elite Gen 5*. https://www.notebookcheck.net/SD-8-Elite-Gen-5-vs-SD-8-Gen-5_19966_19990.247596.0.html
- [S26]: ggml-org GitHub discussions, community benchmark thread for ggml's Metal backend on Apple Silicon M-series (LLaMA-7B Q4_0, TG128). URL omitted.
- [S27]: ggml-org GitHub discussions, community benchmark thread for ggml's CUDA backend on NVIDIA DGX Spark. URL omitted.
- [S28]: Techgage, PCIe per-lane rates after 128b/130b. https://techgage.com/?p=54226 ; GamersNexus, RTX 5090 PCIe 5.0 vs 4.0 vs 3.0. https://gamersnexus.net/gpus/nvidia-rtx-5090-pcie-50-vs-40-vs-30-x16-scaling-benchmarks
- [S29]: NVIDIA Technical Blog, *How to Optimize Data Transfers in CUDA C/C++*. https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
- [S30]: NVIDIA Developer Forums, *Bandwidth test of pageable memory is much different in 2 computers*. https://forums.developer.nvidia.com/t/bandwidth-test-of-pageable-memory-is-mush-different-in-2-computer/355919
- [S31]: MSI, *PCIe lane bifurcation and bandwidth sharing on X870E/X870*. https://www.msi.com/blog/pcie-lane-bifurcation-and-bandwidth-sharing-on-msi-x870e-x870-motherboards ; Guru3D X870E chipset overview. https://www.guru3d.com/review/colorful-igame-x870e-vulcan-w-oc-v14-review/page-2/
- [S32]: Samsung, *990 PRO datasheet*. https://download.semiconductor.samsung.com/resources/data-sheet/Samsung_NVMe_SSD_990_PRO_Datasheet_Rev.1.0_10129514072296.pdf
- [S33]: Samsung, *9100 PRO Series brochure*. https://download.semiconductor.samsung.com/resources/brochure/9100_PRO_Series_Brochure.pdf
- [S34]: Samsung Newsroom, *970 EVO Plus*. https://samsung-newsroom-production.barbariangroup.com/us/?p=25325 ; Samsung Community (QD1 19K IOPS). https://eu.community.samsung.com/t5/computers-it/samsung-970-evo-plus-slow-random-read-write/m-p/3727694/highlight/true
- [S35]: Guru3D, *Without a cooler Crucial's PCIe 5 SSD decelerates to HDD speeds*. https://www.guru3d.com/story/without-a-coolercrucials-pcie-5-ssd-decelerates-to-hdd-speeds/ ; ocinside T700 review. https://ocinside.de/review/crucial_t700_2tb_heatsink_vs_1tb_m2_ssd/6
- [S37]: NVIDIA, *GPUDirect Storage Overview Guide*. https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html
- [S38]: Guru3D, *DirectStorage 1.2*. https://www.guru3d.com/news-story/microsoft-directstorage-1-2-gets-faster-enhanced-features-and-improvements.html ; NVIDIA, *GDeflate for DirectStorage*. https://developer.nvidia.com/blog/accelerating-load-times-for-directx-games-and-apps-with-gdeflate-for-directstorage
- [S39]: Microsoft Learn, *ioringapi.h*. https://learn.microsoft.com/en-us/windows/win32/api/ioringapi/
- [S40]: ronomon/direct-io (F_NOCACHE semantics). https://github.com/ronomon/direct-io ; PostgreSQL hackers on F_NOCACHE vs O_DIRECT. https://www.postgresql.org/message-id/CA+hUKG+ADiyyHe0cun2wfT+SVnFVqNYPxoO6J9zcZkVO7+NGig@mail.gmail.com
- [S41]: Axboe, *Efficient IO with io_uring*. https://kernel.dk/io_uring.pdf ; *High-Performance DBMSs with io_uring* (IOPOLL needs O_DIRECT; registered buffers). https://arxiv.org/pdf/2512.04859
- [S42]: Apple Community thread, Mac Studio SSD read speeds. https://discussions-kr-prz.apple.com/thread/256035322 ; VI-Control, M1 Ultra SSD. https://vi-control.net/community/threads/nvme-and-ssd-speeds-on-mac-studio-m1-ultra-empirical-data.148444/
- [S43]: HighSpeedInternet.com, *Testing Wi-Fi 7*. https://www.highspeedinternet.com/resources/wi-fi-7-tested-benchmarks
- [S44]: Sui et al., *Characterizing and Improving WiFi Latency in Large-Scale Operational Networks* (MobiSys 2016). https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/mobisys_2016b.pdf
- [S46]: Level1Techs, *Ryzen AI Halo – USB4 Clustering with RDMA – Testing Notes*. https://forum.level1techs.com/t/ryzen-ai-halo-usb4-clustering-with-rdma-testing-notes/253117
- [S47]: Intel, *Thunderbolt 5 Technology Brief*. https://intel.com/content/dam/www/central-libraries/us/en/documents/2023-09/thunderbolt-5-technology-brief.pdf
- [S48]: Jeff Geerling, *1.5 TB of VRAM on Mac Studio – RDMA over Thunderbolt 5* (hardware review). https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5
- [S49]: GIGAZINE, summary of the same hardware tests. https://gigazine.net/gsc_news/en/20251222-mac-rdma-over-thunderbolt-1-5tb/
- [S50]: MLX documentation, *Distributed communication* (JACCL). https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- [S51]: ml-explore/mlx issue #3207, *RDMA file transfer over Thunderbolt 5 with JACCL*. https://github.com/ml-explore/mlx/issues/3207
- [S51b]: ml-explore/mlx issue #3162, *AppleThunderboltRDMA hard limit of 100 MRs*. https://github.com/ml-explore/mlx/issues/3162
- [S52]: *Towards Training Private LLMs: Fine-Tuning on Apple Silicon with RDMA over Thunderbolt* (arXiv 2609.18066), used here for its link measurements. https://arxiv.org/html/2609.18066
- [S53]: Framework Community, Thunderbolt networking threads. https://community.frame.work/t/how-come-no-one-seems-to-have-tried-using-thunderbolt-for-40gbps-networking/75470 ; Proxmox forum, NUC 13 Thunderbolt ring. https://forum.proxmox.com/goto/post?id=602583
- [S54]: HPC Advisory Council, *OSU Benchmark Tuning … HDR InfiniBand*. https://hpcadvisorycouncil.atlassian.net/wiki/spaces/HPCWORKS/pages/1284538459/OSU+Benchmark+Tuning+for+2nd+Gen+AMD+EPYC+using+HDR+InfiniBand+over+HPC-X+MPI
- [S55]: NVIDIA Developer Forums, *HDR InfiniBand and ConnectX-6*. https://forums.developer.nvidia.com/t/hdr-infiniband-and-connectx-6-vpi-interfaces/205934
- [S56]: Arista/Solarflare, *10Gb Ethernet: The Foundation for Low-Latency, Real-Time*. https://www.arista.com/assets/data/pdf/Whitepapers/Arista_Solarflare_Low_Latency_10GbE__1_.pdf
- [S57]: Hugging Face model configs: https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/config.json , https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json , https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json , https://huggingface.co/moonshotai/Kimi-K2-Thinking (594 GB, INT4 g32 routed experts; attention, shared experts, dense MLP and lm_head excluded from quantisation) , https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
- [S58]: GGUF file sizes via the Hugging Face API: https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF , https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF , https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF , https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF
- [S59]: DeepSeek-AI, *DeepSeek-V3 Technical Report* (arXiv 2412.19437). https://arxiv.org/abs/2412.19437
- [S60]: DeepSeek-AI, *DeepSeek-V2* (MLA) (arXiv 2405.04434). https://arxiv.org/abs/2405.04434
- [S61]: Ainslie et al., *GQA* (arXiv 2305.13245). https://arxiv.org/abs/2305.13245
- [S65]: ggml-org GitHub, draft libllama change *GPU-resident LRU cache for host-offloaded MoE expert weights*, with community hit-rate traces. URL omitted.
- [S67]: Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (arXiv 2211.17192). https://arxiv.org/abs/2211.17192
- [S71]: NVIDIA, *nccl-tests PERFORMANCE.md* (algorithm vs bus bandwidth, 2(n−1)/n). https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
- [S72]: Kaplan et al., *Scaling Laws for Neural Language Models* (arXiv 2001.08361). https://arxiv.org/abs/2001.08361
- [S74]: Williams, Waterman, Patterson, *Roofline* (CACM 2009). https://doi.org/10.1145/1498765.1498785
- [S75]: r/LocalLLaMA, DeepSeek-V3-0324 MLX 4-bit on M3 Ultra (prompt 58–82 tok/s, generation ≈21 tok/s, peak memory 380 GB). https://www.reddit.com/r/LocalLLaMA/comments/1jj6i4m/deepseek_v3/ ; DEV Community, R1-0528 on M3 Ultra (11 K prompt: ggml 77 vs MLX 189 tok/s prefill). https://dev.to/someoddcodeguy/running-deepseek-r1-0528-q4km-and-mlx-4-bit-on-a-mac-studio-m3-le4
- [S76]: devnote on overriding the macOS Metal VRAM cap (`iogpu.wired_limit_mb`; unofficial). https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap
