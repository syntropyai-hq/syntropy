# Compute backends, model formats and portability

Research notes for the Syntropy plan. Snapshot date 2026-10-19.

Scope: which kernel libraries Syntropy could sit on, which language the system layer should be written in, which on-disk model format to standardise on, which model architectures matter, how to ship one installer to every OS and GPU vendor, and how to detect hardware at runtime. Distributed-inference prior art is in `multi-device-inference.md`, and tiering prior art (including the single-GPU MoE engine studied in depth) is in `tiered-memory-inference.md`. This file only touches those topics where they constrain the backend choice.

## How to read this

- **[verified: code]**: read in source. Shallow clones, never built or executed: ggml/libllama `412a363c` (2026-10-19), safetensors HEAD, and research clones of the surveyed desktop runners, distributed runners and tiering engines (taken 2026-10-13 to 2026-10-19). Model facts come from each model's own `config.json` / `params.json` on the Hugging Face Hub, downloaded on 2026-10-19.
- **[verified: docs]**: stated by an official doc, model card, release note or maintainer README. Third-party benchmarks are marked "third-party" and were not reproduced.
- **[reported by a surveyed system]**: a figure from a surveyed system's own published results, kept with its test conditions. Per ADR-0000 D5 the system is not named.
- **[inferred]**: my reasoning or estimate. Arithmetic derived from configs is labelled "computed".
- No benchmarks were run for this document. Every number below is either quoted with its source or computed from a published config.
- Surveyed systems are referred to by role, consistently through the file: "the Go desktop runner", "the closed-source desktop app", "the single-file runner", "the Tauri desktop app", "the Python/MLX cluster runner", "the home-cluster ggml fork", "the Rust candle engine", "the Rust distributed runner", "the C++ distributed runner", "the single-GPU MoE engine" and "the CPU/GPU hybrid MoE engine".

---

## 0. Summary

1. **ggml is the only kernel library that covers the whole hardware brief today.** Its backends span CUDA, HIP, Metal, Vulkan, SYCL, OpenCL (Adreno), Hexagon, CANN, MUSA, WebGPU, OpenVINO (in progress), ZenDNN, zDNN, RPC and CPU, and it ships prebuilt binaries for Windows, macOS, Linux and Android. It also exposes nearly every primitive a tiering engine needs: per-device buffer types, pinned host buffers, async set/get/copy, events, a multi-backend scheduler with per-node backend assignment, a regex tensor-placement override, mmap/direct-I/O/lazy loading, and since 2026-10-07 a GPU LRU cache for host-resident MoE experts. **[verified: code]** The gaps are an SSD tier, a cross-node scheduler and API stability (ggml is still 0.27).
2. **Model graphs live in libllama, not ggml.** libllama has 164 `.cpp` files under `src/models/` (one per architecture family, plus a few helpers), including `deepseek4.cpp`, `kimi-k3.cpp`, `glm5-next.cpp`, `minimax-m3.cpp` and `qwen4exp.cpp` (Qwen3.8-Flash-Next). **[verified: code]** Building on raw ggml means writing those graphs again. Building on libllama means accepting its placement model and extending it upstream or in a fork. This is the biggest decision the design has to make.
3. **MLX is the strongest alternative on Apple silicon and now runs on Linux CUDA**, through the `mlx[cuda12]` / `mlx[cuda13]` extras on SM ≥ 7.5 with glibc ≥ 2.35. **[verified: docs]** It exposes memory limits but no per-array placement API **[verified: code]**, so it can be a compute backend but not the tiering substrate. A popular Go desktop runner already dlopens it as a second engine next to its ggml-based one. **[verified: code]**
4. **The system layer should be Rust** (recommended), linking ggml/libllama through their C headers and loading GPU backends as runtime modules (`GGML_BACKEND_DL`). C++ throughout is the fast-start alternative. Python+MLX (the cluster runner's choice) and Go (the desktop runner's choice) both lose on the "runs anywhere, including phones" requirement. **[inferred]**
5. **Canonical format: GGUF as published on the Hub, imported unchanged, plus a Syntropy-local repacked cache per device.** GGUF stores MoE experts as `{n_embd, n_ff, n_expert}` tensors, so each expert is a contiguous byte range. **[verified: code]** Neither GGUF (32-byte default alignment) nor safetensors (8-byte header padding) aligns individual experts to 4 KiB. The O_DIRECT/DMA path should therefore read around expert slices, or use a locally repacked file with padded experts. **[verified: code]** for the alignment facts, **[inferred]** for the design.
6. **2026 frontier MoE models are huge and natively 4-bit**: Kimi K3 is 2.8T total, 104B active, 896 experts top-16, with MXFP4 quantisation-aware training. DeepSeek V4-Pro is 1.6T/49B with FP4 experts. Qwen3.8-2.4T is A95B. Attention is now mostly hybrid: Gated DeltaNet, KDA, Mamba-2, DSA/QSA sparse attention, and compressed attention in DeepSeek V4. **[verified: docs]** Each new attention type needs kernels on every backend. That coverage lag is the main portability risk.
7. **Packaging reference: upstream ggml/libllama release assets plus the Go desktop runner's layout.** Ship separate CUDA 12 and CUDA 13 builds, because CUDA 13 dropped Maxwell, Pascal and Volta. Ship ROCm as a separate download (the Go desktop runner's Linux ROCm tarball is 1003 MiB). Use CPU variants selected at runtime. Sign and notarise on macOS, sign with Authenticode on Windows. **[verified: docs/code]** Reuse ggml's dynamic backend loader instead of writing one.

---

## 1. Kernel and runtime options

### 1.1 What a tiering + distribution engine needs from the kernel layer

These are the requirements every candidate below is checked against. **[inferred]**

| Need | Why |
|---|---|
| Per-tensor (ideally per-slice) device placement | Experts, attention, embeddings and KV cache live in different tiers |
| Pinned (page-locked) host buffers | Async PCIe DMA. Without pinning, CUDA/HIP copies stage through a driver bounce buffer |
| Async H2D/D2H copy with events/streams | Overlap expert upload with compute (every surveyed tiering engine depends on this) |
| Graph split across devices in one forward pass | GPU attention + CPU experts in the same layer |
| Wrap externally allocated memory (`buffer_from_host_ptr`) | Zero-copy from mmap or from our own arena |
| Op coverage for MoE (`mul_mat_id`), MLA, GDN/KDA/Mamba, sparse attention, every quant type | Model coverage |
| A C ABI | FFI from any system language |
| Mobile and Web targets | Brief includes Android, iOS and possibly browser |

### 1.2 Comparison matrix

| Engine | Hardware (2026) | Quant formats | Embedding API | Licence | Momentum | Placement hooks |
|---|---|---|---|---|---|---|
| **ggml / libllama** | CUDA, HIP, Metal, Vulkan, SYCL, OpenCL (Adreno), Hexagon, CANN, MUSA, WebGPU, OpenVINO (in progress), ZenDNN, zDNN, BLAS/BLIS, RPC, VirtGPU, CPU (x86 variants, ARM, RISC-V, s390x) **[verified: docs]** | GGUF: Q4_0…Q8_0, k-quants, i-quants, TQ1/TQ2, MXFP4, NVFP4, Q1_0, Q2_0, BF16/F16 **[verified: code]** | C (`ggml.h`: 379 `GGML_API`, `ggml-backend.h`: 119, `llama.h`: 260 `LLAMA_API`) **[verified: code]**. Rust: a community libllama bindings crate, 0.1.160 (2026-10-15) **[verified: docs]** | MIT | Very high. ggml.ai joined Hugging Face on 2026-02-20 ([Willison](https://simonwillison.net/2026/Feb/20/ggmlai-joins-hugging-face/)). Several builds a day (b11724 on 2026-10-19) **[verified: docs]** | Best in class (§1.3.1) |
| **MLX** | Metal; CUDA on Linux (SM ≥ 7.5, driver ≥ 550 for cuda12, ≥ 580 for cuda13); CPU-only Linux; Windows CUDA CI in progress **[verified: docs]** | affine 1–8 bit (group 32/64/128), mxfp4, mxfp8, nvfp4 **[verified: docs]** | C++, Python, Swift, `mlx-c` | MIT | High. 0.32.0 released 2026-07-07 | Limits only (`set_memory_limit`, `set_wired_limit`, `set_cache_limit`). On CUDA small allocations use `cudaMallocManaged`, large ones `cudaMallocAsync` **[verified: code]** |
| **tinygrad** | CPU, Metal, CUDA, NV (direct ioctl), AMD (incl. AM userspace driver for RDNA3/4 with amdgpu unloaded), QCOM, OpenCL, WebGPU; TinyGPU app for AMD/NVIDIA eGPUs on macOS over USB4/TB **[verified: docs]** | Its own (no GGUF k-quant kernels documented) | Python only | MIT | Pre-1.0 ("isn't 1.0 yet") **[verified: docs]** | Device strings per tensor. No pinned/async API documented **[inferred]** |
| **candle** | CPU (MKL/Accelerate), CUDA (+NCCL, cuDNN, opt-in cuTile), Metal (`candle-metal-kernels`), WASM. No ROCm or Vulkan **[verified: docs]** | GGUF/ggml quant types for quantised models **[verified: docs]** | Rust | MIT/Apache-2.0 | Medium. 0.11.0 on 2026-06-26 **[verified: docs]** | `Device` per tensor. Pinned/streams only by dropping to `cudarc` **[inferred]** |
| **burn / CubeCL** | CUDA, ROCm, Metal, Vulkan, WebGPU, CPU (CubeCL), Flex (no-std, Wasm); LibTorch deprecated in 0.22 **[verified: docs]** | Not documented on README | Rust | MIT/Apache-2.0 | Rising. 0.20 (Jan 2026) unified CPU+GPU kernels; 0.22.0 on 2026-10-06 **[verified: docs]** | Memory-management config per runtime. No LLM offload primitives found **[inferred]** |
| **IREE** | MLIR compiler + HAL runtime; topics list CUDA, ROCm, Vulkan/SPIR-V; LF AI & Data sandbox **[verified: docs]** | Compiler-driven; IRPA parameter archives **[verified: docs]** | C runtime API | Apache-2.0 + LLVM exception | No 2026 LLM-specific release found **[verified: docs]** (absence) | HAL allocators, buffer memory types, semaphores, IRPA parameter files. Flexible but compile-time model **[inferred]** |
| **ONNX Runtime GenAI** | CPU, CUDA, DirectML, NvTensorRtRtx, OpenVINO, QNN, WebGPU, AMD GPU; Linux, Windows, Mac, Android **[verified: docs]** | int4 (RTN etc. via model builder) | C/C++, C#, Python, Java | MIT | Microsoft-backed; arch list includes gpt-oss, DeepSeek, Gemma, Qwen, LFM2, Granite MoE Hybrid **[verified: docs]** | ORT allocators and IOBinding exist. No MoE offload or tiering in GenAI **[inferred]** |
| **TensorRT (LLM runtime)** | NVIDIA only; 1.4.0rc1 badge; "Architected on PyTorch" **[verified: docs]** | FP8, FP4, INT8, INT4 AWQ **[verified: docs]** | Python LLM API, C++ runtime | Apache-2.0 | Datacenter focus (DeepSeek V4 on Blackwell, NVL72) **[verified: docs]** | KV offload to host/disk through NVIDIA's separate KV block manager, PyTorch backend only, no CUDA graphs **[verified: docs]**. TensorRT weight streaming (device budget) **[verified: docs]** |
| **ExecuTorch** | XNNPACK, Vulkan, Qualcomm, MediaTek, Arm, Samsung on Android; Core ML, MPS on iOS; 1.4.0 adds Vulkan on Linux/Windows desktop **[verified: docs]** | AOT-quantised `.pte` + external `.ptd` weights | C++, Java, ObjC, Swift | BSD (LICENSE file) | 1.0 left beta; 1.4.0 latest found **[verified: docs]** | AOT memory planning. Weights can live in a separate `.ptd` named-data map **[verified: docs]**. No runtime tiering **[inferred]** |
| **Rust candle engine** (surveyed) | CUDA (FlashAttention v2/v3, optional cuTile), Metal, CPU. No ROCm or Vulkan **[verified: docs]** | GGUF 2–8 bit, GPTQ, AWQ, HQQ, FP8, BNB, in-situ quantisation, its own pre-quantised format **[verified: docs]** | Rust, Python | MIT | Active; 0.9.4 on master, built on upstream candle 0.11 at a pinned git rev **[verified: code]** | Device mapping, PagedAttention. No expert offload documented **[verified: docs]** |
| **WebGPU** (wgpu 30.0.1, Dawn) | Shipping in Chrome, Edge, Safari 26, Firefox 141+ (Windows) and 145+ (ARM macOS); Linux/Android Firefox in progress **[verified: docs]** | Via engine (ggml-webgpu, or TVM-compiled WGSL kernels) | wgpu (Rust), Dawn (C++) | MIT/Apache, BSD | Subgroups shipped only in Chromium (no Firefox, no Safari 26.5) **[verified: docs]** | ggml-webgpu caps: no async, no host buffer, no events **[verified: code]** |

Adjacent engines worth tracking, but not candidates as a base:

- **A CPU/GPU hybrid MoE engine** with AMX/AVX-512/AVX2 kernels. Its 2025 paper reports 4.62–19.74× prefill and 1.25–4.09× decode speedups over prior systems **[reported by a surveyed system]**. Python + C++, server-oriented. Covered in `tiered-memory-inference.md`.
- **A mobile-platform vendor's on-device LLM runtime**: a C++ LLM pipeline across CPU/GPU/NPU, shipped at scale in that vendor's browser, OS and wearables **[verified: docs]**. A strong mobile reference, but tied to the vendor's own model format.
- **A Zig + MLIR/OpenXLA framework**. Its v2 (2026-03-24) targets NVIDIA, AMD, TPU and Trainium. Its LLM server (alpha, 2026-07-08) is proprietary (secondary coverage) **[verified: docs]**. Datacenter-first, no consumer Metal/Vulkan story found.
- **A CUDA-only consumer engine** with its own QTIP-based trellis quant format and experimental CPU MoE offload **[verified: docs]**.

### 1.3 Per-engine notes

#### 1.3.1 ggml / libllama: placement primitives in detail

Device capability flags reported by each backend (`ggml_backend_dev_caps`) **[verified: code]**:

| Backend | async | pinned host buffer | buffer_from_host_ptr (cap flag / function present) | events | mmap load |
|---|---|---|---|---|---|
| CUDA (and HIP, MUSA: same source tree) | yes | yes (unless disabled by env) | no / no | yes | yes, except integrated GPUs |
| Metal | yes | no (unified memory) | **yes / yes** | yes | yes |
| Vulkan | yes | yes | no / yes | yes | yes, except integrated GPUs |
| SYCL | yes | yes (unless `GGML_SYCL_NO_PINNED`) | no / yes | build-dependent | yes |
| CPU | no | no | **yes / yes** | no | yes |
| RPC | yes | no | no / no | yes | yes |
| WebGPU | no | no | no / no | no | yes |

On Vulkan and SYCL the capability flag is `false` but the device interface registers a `buffer_from_host_ptr` function. Treat that as present but not advertised, and test it before relying on it. **[verified: code]**

APIs that a tiering layer can drive directly (`ggml/include/ggml-backend.h`) **[verified: code]**:
- `ggml_backend_dev_buffer_type()` / `ggml_backend_dev_host_buffer_type()`: device memory vs pinned host memory per device.
- `ggml_backend_tensor_set_async / get_async / set_2d_async / copy_async` plus `ggml_backend_event_{new,record,wait,synchronize}`: overlapped uploads.
- `ggml_backend_sched_new(backends, bufts, n, graph_size, parallel, op_offload)` and `ggml_backend_sched_set_tensor_backend(sched, node, backend)`: per-node backend assignment in one graph.
- `ggml_backend_dev_get_props()` returns free/total memory, device type (CPU/GPU/IGPU/ACCEL) and PCI bus id.
- `ggml_backend_load()` / `ggml_backend_load_all_from_path()`: runtime backend modules.

What libllama layers on top (CLI names; each has a `llama.h` parameter) **[verified: code]**:
- `-ot/--override-tensor <regex>=<buffer type>`, `--cpu-moe`, `--n-cpu-moe N`, `--n-cpu-ffn N` (dense FFN offload).
- `--moe-cache-mib N`: an LRU cache in VRAM for experts kept in host RAM. Only misses are uploaded. Small batches only (≤ 32 tokens). It was ported from a downstream fork and merged upstream on 2026-10-07; multi-GPU support followed on 2026-10-08. The change's own benchmark: on an RTX 5090 with Qwen3.8-Flash-Next Q4_0 (93.7 GiB, 65.4 GiB of experts), decode rose from 30.8 to 67.8 tok/s (2.20×) at an 89% hit rate, using `-cmoe --moe-cache-mib 19000` **[verified: docs]**. This is the core idea of the single-GPU MoE engine studied in `tiered-memory-inference.md`, now upstream.
- `--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}`. `dio` uses `O_DIRECT` and since 2026-10-01 streams through a ≤ 64 MB reusable buffer. **[verified: code/docs]**
- `--lazy-mode {on,auto,off}` reads rows of large tensors, such as per-layer or n-gram embeddings, from disk on demand. `auto` applies it to tensors over 4 GiB. **[verified: code]** This is the only SSD-tier mechanism in upstream. It is not general weight streaming.
- `--split-mode tensor` (experimental tensor parallelism through a "meta device" in `ggml-backend-meta.cpp`) and `--fit` (auto-sizes unset arguments to free memory; not supported with `tensor`). **[verified: code/docs]**
- RPC backend with tensor-parallel RPC (protocol v8) in libllama v0.7.0 **[verified: docs]**. Details in `multi-device-inference.md`.

Gaps for Syntropy **[inferred]**:
- No NVMe tier for arbitrary weights apart from mmap page cache and lazy rows.
- No global placement optimiser: `--fit` is greedy per device.
- No cross-node scheduler beyond RPC.
- `llama.h` changes often (the v0.7.0 notes list five `llama.h` changes).
- The expert cache is an LRU with no prefetch or predictor. A profile-prefilled, adaptive cache (as in the single-GPU MoE engine) is ahead of it; an open upstream proposal adds profile prefilling.

#### 1.3.2 MLX

- CUDA backend on Linux through `pip install mlx[cuda12]` (SM ≥ 7.5, driver ≥ 550.54.14, glibc ≥ 2.35) or `mlx[cuda13]` (driver ≥ 580). CPU-only Linux through `mlx[cpu]`. The install page still says Python packages target macOS ≥ 14 on Apple silicon for the Metal path ([install docs](https://ml-explore.github.io/mlx/build/html/install.html)) **[verified: docs]**.
- The Python/MLX cluster runner pins `mlx>=0.32.0` and on Linux uses `mlx-cuda-12/13==0.32.0` wheels from a patched fork release **[verified: code]**. The upstream Linux path is usable, but that runner carries patches.
- Memory API is process-wide limits only (`mlx/memory.h`). On CUDA it uses managed memory when `concurrentManagedAccess` is available and falls back to `cudaMallocHost` on Windows, WSL and Tegra (`mlx/backend/cuda/allocator.cpp` comments) **[verified: code]**. There is no explicit pinned/device split, so a tiering engine cannot place an expert in host RAM and stream it.
- The Go desktop runner builds MLX through `mlx-c`, loads it with `dlopen` (`mlx/dynamic.c` in its tree), and selects CUDA variants (`cuda_v13`) through an environment variable. On macOS arm64 MLX is its default for safetensors models **[verified: code]**.

#### 1.3.3 Others in brief

- **tinygrad**: the AM driver needs `amdgpu` unloaded and works best without IOMMU ([docs](https://docs.tinygrad.org/developer/am/)) **[verified: docs]**. That is fine for a lab and unacceptable for a consumer install. **[inferred]**
- **candle**: used by both surveyed Rust engines **[verified: code]**. The Rust distributed runner adds its own Vulkan path with `ash` and `gpu-allocator` because candle has none **[verified: code]**, which shows the coverage gap.
- **burn/CubeCL**: the broadest pure-Rust GPU coverage (CUDA, ROCm, Metal, Vulkan, WebGPU). Its README lists no LLM inference example beyond a toy text generator **[verified: docs]**. 0.21 reports 3.4× average Vulkan runtime improvement over 0.20 (vendor-reported, [blog](https://burn.dev/blog/release-0.21.0/)) **[verified: docs]**.
- **TensorRT**: NVIDIA-only and Linux/datacenter-oriented. Its useful idea is weight streaming: `setWeightStreamingBudgetV2` caps device memory for weights and TensorRT schedules the rest from host ([docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/weight-streaming.html)) **[verified: docs]**.
- **ExecuTorch / the mobile vendor runtime / ORT GenAI**: these are the realistic NPU routes (QNN, MediaTek, Core ML/ANE, OpenVINO NPU). None exposes runtime weight tiering. **[inferred]** If Syntropy wants NPUs, it would treat one of them as an opaque "device" running a whole sub-model, not a tensor-level backend.

### 1.4 Performance standing (quoted results only)

| Claim | Source | Notes |
|---|---|---|
| M5 Max 128 GB, Qwen3.6-35B-A3B 8-bit: ggml (Metal) 93 tok/s vs MLX 85 tok/s decode at 128-token context; 105 with MTP. Qwen3.6-27B: 18 vs 17 | [stared/benching-local-llms-on-apple-silicon](https://github.com/stared/benching-local-llms-on-apple-silicon), 2026-06-14, third-party | No prefill figures |
| The Go desktop runner, 0.19 MLX engine (NVFP4) vs 0.18 ggml engine (Q4_K_M), Qwen3.5-35B-A3B, Apple silicon: decode 58 → 112 tok/s, prefill 1154 → 1810 tok/s | **[reported by a surveyed system]**, via a secondary article | Different quantisation, so not engine-vs-engine |
| libllama `--moe-cache-mib`: RTX 4090 25.0 → 40.7 tok/s, RTX 5090 30.8 → 67.8 tok/s (Qwen3.8-Flash-Next Q4_0, experts in host RAM) | upstream change benchmarks, 2026-10-07/08 **[verified: docs]** | Prefill drops 9–27% with the cache on (multi-GPU change) |
| Single-GPU MoE engine (custom CUDA, same model family), RTX 5070 12 GB + 64 GB RAM: Q2_0 94 tok/s decode, 2650 tok/s prefill at 32K | **[reported by a surveyed system]** (maintainer README, read in clone) | Model-specific engine with MTP; see `tiered-memory-inference.md` |
| Rust candle engine v0.8.2 vs ggml/libllama, Gemma 4 E4B Q8 decode: 223.1 vs 183.0 tok/s (H100); 26B-A4B on GB10: 46.8 vs 46.4 | **[reported by a surveyed system]** | Self-reported, CUDA only |
| Radeon AI PRO R9700 (RDNA4): ggml Vulkan fastest token generation for Qwen3-8B and gpt-oss-20b; ggml ROCm/HIP faster on pp512 for Qwen3-8B | Phoronix review, Nov 2025, third-party | An Oct 2026 Phoronix review of an AMD-backed local server still finds Vulkan faster on several models |
| A browser engine on TVM-compiled WebGPU kernels keeps 71–80% of its native counterpart's decode throughput on M3 Max | **[reported by a surveyed system]**, Dec 2024 | Compiled kernels, not hand-written |
| Home-cluster ggml fork, 4 home devices: Llama 3-70B 674 ms/token vs 10120 ms for stock libllama on one box (mmap thrash) | **[reported by a surveyed system]** | Distributed + mmap; see `multi-device-inference.md` |

Takeaway **[inferred]**: no engine is uniformly fastest. On NVIDIA, the published gaps between ggml and the Rust/C++ alternatives are within roughly ±25% and come from self-reported benchmarks. On Apple, ggml and MLX trade places by model and context length. On AMD, ggml's Vulkan backend is often faster than its HIP backend for decode. For a tiering engine the dominant factor is hit rate and PCIe/SSD scheduling, not kernel speed (2.2× from the expert cache alone above). So coverage and hooks should drive the choice, not microbenchmarks.

---

## 2. Language and runtime for the system layer

### 2.1 What comparable projects chose

| Project | System layer | Kernel layer | Binding style | Evidence |
|---|---|---|---|---|
| ggml / libllama upstream | C/C++ | ggml | in-process | **[verified: code]** |
| **Go desktop runner** (v0.40.6, 2026-10-16) | Go 1.26 | upstream libllama HTTP server **run as a subprocess** (pinned build `b11669`, local compat patches) plus MLX through dlopen'd `mlx-c` | process boundary for libllama, cgo + dlopen for MLX | Its server wrapper is documented in-tree as wrapping the upstream server binary as a subprocess; GPU discovery spawns the server briefly to enumerate GPUs **[verified: code]** |
| Closed-source desktop app | desktop app, plus a CLI | ggml/libllama and MLX engines; its MLX engine is open source (MIT) **[verified: docs]**; runtimes ship as extension packs updated apart from the app | separate runtimes | 2025 changelog **[verified: docs]**. 2026 runtime packaging not documented publicly |
| Single-file runner 0.10.x | C/C++ on Cosmopolitan libc (one APE file for Linux/macOS/Windows/BSD) | ggml/libllama, rebuilt from scratch in 0.10.0 (2026-03) | GPU support as separately loaded `ggml-(cuda\|rocm\|vulkan)` libraries | support docs **[verified: docs]** |
| Python/MLX cluster runner | Python 3.13 (pinned `==3.13.*`) | MLX (Metal; CUDA on Linux through mlx-cuda wheels) | Rust networking crate (zenoh 1.9) through pyo3 | `pyproject.toml`, `Cargo.toml` **[verified: code]**. Tier 1 support is Apple Silicon only (platforms doc) |
| Home-cluster ggml fork | C/C++ (fork of libllama) | ggml | in-process; ZeroMQ, HiGHS ILP, external `fio` for disk benchmarks | README, profiler source **[verified: code]** |
| Rust candle engine | Rust | candle 0.11 (git pin) | in-process | `Cargo.toml` **[verified: code]** |
| Rust distributed runner | Rust (tokio) | candle 0.9 plus its own Vulkan (`ash`) and wgpu | in-process. Mobile through `uniffi` 0.28 (`cdylib` + `staticlib`) for Android/iOS | core and mobile `Cargo.toml` files **[verified: code]**. Licence FAIR v1.0.0 |
| Tauri desktop app | Rust (Tauri) + web UI | libllama through a Tauri plugin; engine variant chosen at build (`cpu`, `vulkan`, `metal`, `cuda12`, `cuda13`, `hip`) | plugin | README **[verified: docs]** |
| C++ distributed runner | C++ | own kernels (CPU, experimental Vulkan) | in-process | clone **[verified: code]** |
| Single-GPU MoE engine | C++ / CUDA / HIP, separate Metal and SYCL engines | own kernels, reuses ggml MMQ and ggml-cpu i-quant kernels | in-process | `CMakeLists.txt`, design notes in-tree **[verified: code]** |

Two observations **[inferred]**:
- **The Go desktop runner now wraps the upstream server instead of embedding ggml.** As of 2026-10 its tree has no ggml backend of its own (`ml/` holds only device and path helpers), and its release notes since mid-2026 refer to an MLX engine and a ggml-based engine. Keeping a parallel engine in Go apparently did not survive contact with libllama's model velocity. The causal reason is my reading, not documented. Syntropy should expect the same pressure.
- **Every project that reaches phones (the Rust distributed runner, ExecuTorch, the mobile vendor runtime, ggml/libllama) is native code with a C ABI or uniffi on top.** None of them uses Python or Go on device.

### 2.2 Trade-off matrix

| | Rust | C++ | Zig | Python + native | Go |
|---|---|---|---|---|---|
| FFI to ggml/libllama (C headers) | `bindgen`. An existing community bindings crate. CMake build through the `cmake` crate **[verified: docs]** | native | `@cImport`, trivial **[inferred]** | ctypes/pybind/nanobind. The Python cluster runner uses nanobind for its build extra **[verified: code]** | cgo (the Go desktop runner) or process boundary **[verified: code]** |
| FFI to CUDA driver/NVML directly | `cudarc` 0.19.10, `nvml-wrapper` 0.13.0 **[verified: docs]** | native | C headers | many | cgo |
| Async I/O (io_uring, IOCP, kqueue) | tokio (epoll/IOCP/kqueue); io_uring crates exist; can do raw `io_uring` or a thread pool of `pread` **[inferred]** | liburing, asio, IOCP direct | std.Io plus manual **[inferred]** | asyncio on top of native threads | netpoller; no io_uring in runtime **[inferred]** |
| Cross-compilation | `cargo` targets; the C++ parts still need per-target toolchains (NDK, Xcode, MSVC) **[inferred]** | CMake presets per target | best-in-class (`zig cc`) **[inferred]** | wheels per platform/arch/Python version; the Python cluster runner uses PyInstaller **[verified: code]** | trivial without cgo, per-target C toolchain with cgo **[inferred]** |
| Single-binary distribution | yes (static Rust + dlopen'd backend modules) | yes | yes | no (PyInstaller bundles: the Python cluster runner ships a signed `.app` built this way) **[verified: code]** | yes for the Go part; the Go desktop runner still ships 1.4 GiB archives because of GPU libraries **[verified: docs]** |
| iOS / Android | `staticlib` + uniffi Swift/Kotlin bindings (the Rust distributed runner does this) **[verified: code]** | native (upstream ggml/libllama ships an xcframework and Android builds) **[verified: docs]** | possible, little precedent **[inferred]** | not viable for shipping apps **[inferred]** | gomobile, limited; cgo + GPU libraries on mobile not seen in any LLM project **[inferred]** |
| Memory safety in networking, parsers, schedulers | strong | weak | moderate | strong at Python level, native code unsafe | strong |
| Contributor pool overlapping ggml | medium | high | low | high | medium |

Language recommendation **[inferred]**:
- **Rust for the orchestrator**: discovery, networking, the placement planner, the I/O scheduler (SSD → pinned RAM → VRAM), HTTP API and model management.
- **C/C++ only where it already exists**: ggml and libllama, vendored as a git submodule with a thin patch queue, plus any new kernels contributed upstream.
- **Keep the kernel boundary at the C ABI.** That keeps cross-compiling to Android/iOS on known paths (cargo-ndk / staticlib + uniffi on the Rust side, CMake on the ggml side).
- Avoid a process boundary on the hot path, as used by the Go desktop runner. A tiering engine has to control buffers and streams inside the forward pass, which a subprocess cannot do.

---

## 3. Model formats and quantisation

### 3.1 Formats

| Format | Layout | Alignment | Random access per tensor | mmap | Ecosystem |
|---|---|---|---|---|---|
| **safetensors** | `u64` header length, JSON header (≤ 100 MB), one packed byte buffer. No strides, no holes, little-endian, row-major **[verified: docs]** | Reference writer pads the header to 8 bytes and sorts tensors by descending dtype size, so each tensor is naturally aligned to its element size only **[verified: code]** (`safetensors/src/tensor.rs` L227–255) | yes (offsets in header) | yes | Default HF format. Dtypes include F4 (MXFP4), F6_E2M3/E3M2, F8_E4M3/E5M2/E8M0 and FNUZ variants **[verified: code]** |
| **GGUF** | header, KV metadata, tensor infos, padded data section | `general.alignment` (u32, any power of 2), default 32; alignment is relative to the GGUF start **[verified: code]** (`gguf.h`, `gguf.cpp` L622–633) | yes | yes (libllama default) | Dominant local format; split files `-0000N-of-0000M.gguf` |
| **MLX** | safetensors + `config.json` `quantization` block. Affine: packed `uint32` weights + per-group scales + biases; fp modes: weights + scales **[verified: docs]** | as safetensors | yes | yes | mlx-community and vendor repos (e.g. `ibm-granite/granite-4.2-*-mlx`) **[verified: docs]** |
| **ONNX** | protobuf graph + external data files | external data offsets; 2 GB protobuf limit forces external data for LLMs **[inferred]** | via external data | partly | ORT GenAI model builder |
| **Trellis-quant safetensors** | safetensors with QTIP-style trellis-coded weights (one CUDA-only engine's format) | as safetensors | yes | yes | CUDA-only consumer |
| **compressed-tensors** | safetensors + `quantization_config` (`quant_method: compressed-tensors`, e.g. `mxfp4-pack-quantized`) | as safetensors | yes | yes | Kimi K2.5/K3, Mistral Large 3 ship this way **[verified: docs]** |
| **IRPA** (IREE) | parameter archive, can reference external file ranges | archive-defined | yes | yes | IREE only |
| **.pte / .ptd** (ExecuTorch) | flatbuffer + appended segments; external named-data map | segment-defined | named lookup | yes | ExecuTorch only |

**MoE layout in GGUF.** libllama creates expert tensors as `{n_embd, n_ff_exp, n_expert}`, for example in `src/models/deepseek.cpp` **[verified: code]**. `n_expert` is the outermost (slowest) dimension, so expert *e* is one contiguous run of `nb[2]` bytes. That makes per-expert pread and upload possible without repacking. libllama's expert cache relies on this. However, `nb[2]` is a multiple of the quant block size (144 bytes for Q4_K per 256 weights), not of 4096. Expert slices therefore start at arbitrary 32-byte-aligned offsets. **[verified: code]**

**O_DIRECT / DMA constraints**:
- Linux exposes the required alignment per file through `statx(STATX_DIOALIGN)` → `stx_dio_mem_align` and `stx_dio_offset_align` (since 6.1). It is supported on ext4, XFS and NVMe block devices. btrfs did not support it at 6.1-rc3 ([LKML](https://lkml.iu.edu/hypermail/linux/kernel/2210.0/00853.html)) **[verified: docs]**.
- On Windows, `FILE_FLAG_NO_BUFFERING` needs sector-aligned offsets and buffers **[inferred from Win32 docs; not re-fetched]**.
- macOS `F_NOCACHE` has no alignment rule **[inferred]**.
- GPUDirect Storage (cuFile) falls back to "compatibility mode" (POSIX read + copy) when `nvidia_fs.ko` is absent or BAR1 is unavailable, and only the open kernel module is supported ([cuFile docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/topics/cufile-compatibility.html)) **[verified: docs]**. On consumer boxes, assume SSD → pinned RAM → VRAM. **[inferred]**

### 3.2 Quantisation schemes (2026)

Bits per weight below are computed from ggml block layouts (`ggml-common.h` static_asserts) **[verified: code]**.

| Scheme | bpw | Block / scale | Where it matters |
|---|---|---|---|
| Q8_0 | 8.50 | 32, f16 scale | quality reference, KV cache |
| Q6_K / Q4_K / Q2_K (k-quants) | 6.5625 / 4.50 / 2.625 | 256 super-blocks, 6-bit sub-scales | the bulk of Hub GGUFs |
| IQ4_XS / IQ3_S / IQ2_XS (i-quants) | 4.25 / 3.4375 / 2.3125 | lattice codebooks, importance-matrix calibrated | low-VRAM MoE (the single-GPU MoE engine ships IQ2_XS/IQ3_S) **[verified: code]** |
| Q2_0 / Q1_0 (new in ggml) | 2.25 / 1.125 | 64 / 128 weights, f16 scale | native 2-bit releases such as NVIDIA's "24GB" Nemotron 3.5 Bolt **[verified: docs]** |
| TQ1_0 / TQ2_0 | ternary | 256 | BitNet-style models (Prism Bonsai 2 27B added in libllama 0.7.0) **[verified: docs]** |
| **MXFP4** (OCP MX) | 4.25 | 32 × E2M1, E8M0 power-of-two scale | gpt-oss native; Kimi K3 (MXFP4 weights / MXFP8 activations, QAT); DeepSeek V4 `expert_dtype: fp4` **[verified: docs]** |
| **NVFP4** | 4.50 (ggml: 64-weight block, 16-weight sub-blocks with UE4M3 scales) | 16 × E2M1, E4M3 scale + FP32 per-tensor scale in NVIDIA's definition | Blackwell-native; NVIDIA says it is more accurate than MXFP4 at the same size ([NVIDIA](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference)) **[verified: docs]**. Mistral Large 3 and Nemotron ship NVFP4 variants |
| MXFP8, FP8 block (E4M3, 128×128 blocks) | 8 + scales | | DeepSeek V3/V4, GLM-5.x, MiniMax M2 ship FP8 block-scaled checkpoints **[verified: docs]**. **ggml has no FP8 tensor type**, so they are converted to GGUF types **[verified: code]** |
| AWQ / GPTQ | 4 (typically) | group-wise int4 + zeros | datacenter serving engines, the Rust candle engine, ORT GenAI; not GGUF |
| QTIP-style trellis quants | 1.6–8 (example: Llama-3.1-70B at 1.6 bpw in < 16 GB) | QTIP trellis, Hadamard incoherence | CUDA-only consumer path ([QTIP](https://arxiv.org/abs/2406.11235)) **[verified: docs]** |
| MLX affine | 1–8, group 32/64/128 | scale + bias | Apple path |

Implications **[inferred]**:
- The frontier models now *train* in MXFP4 or ship FP4 experts. Syntropy should carry MXFP4 end to end, with no requantisation, on every backend. ggml has the type. Kernel speed per backend for MXFP4 is unverified here.
- FP8 checkpoints must be converted at import. That is a one-time CPU cost and a reason to keep a local converted cache.
- For the SSD tier, the bits per weight of experts sets the bytes moved per token. See §4.3.

### 3.3 Hub conventions and tooling

- **HF originals**: `config.json`, sharded `model-0000N-of-0000M.safetensors`, `model.safetensors.index.json` (tensor → shard map). Quantised originals declare `quantization_config` (`fp8`, `mxfp4`, `compressed-tensors`) **[verified: docs]**, as in the configs in §4.
- **GGUF repos**: one file per quant type, split at ~50 GB. Unsloth "UD-" dynamic quants and ISTA-DASLab GSQ-RCO quants are what the single-GPU MoE engine ships with **[verified: code]**.
- **Storage backend**: the Hub's Xet storage uses content-defined ~64 KB chunks with deduplication ([Hub docs](https://huggingface.co/docs/hub/xet)) **[verified: docs]**. Re-downloading a sibling quant of the same model is cheap, and partial re-downloads are possible **[inferred]**.
- **Tooling**:
  - `convert_hf_to_gguf.py` plus the GGUF quantizer (with an importance matrix), in the libllama tree.
  - `mlx_lm.convert`.
  - the ORT GenAI model builder.
  - the trellis-quant engine's converter.
  - llm-compressor (compressed-tensors) and NVIDIA ModelOpt (NVFP4) **[inferred, names not re-verified]**.

### 3.4 Recommended canonical format strategy

**[inferred]** throughout.

1. **Accept, unchanged:**
   - GGUF from the Hub. This is the primary path. Files stay byte-identical, so a user's existing GGUFs keep working and interop with every ggml-based runner is preserved.
   - safetensors originals (BF16/FP8/MXFP4/compressed-tensors), with conversion to GGUF at import using the vendored converter.
2. **Do not invent a new interchange format.** GGUF already gives per-tensor random access, an alignment knob and contiguous experts.
3. **Keep a per-machine "placement cache"**: files Syntropy writes locally, keyed by (model hash, device class, placement plan):
   - experts repacked so each expert slice starts on a 4 KiB boundary (padded) for O_DIRECT reads straight into pinned staging buffers;
   - optional backend-specific repacked layouts (libllama already has CPU repack buffer types and `--repack`);
   - a sidecar manifest with per-expert offsets, sizes and an offline activation-frequency profile.
   - It is safe to delete and rebuildable.
4. **For distribution across nodes**, ship only the slices a node owns (tensor-range requests against the GGUF data offsets). Hub downloads support HTTP range reads; Xet makes partial fetch practical.
5. Writing `general.alignment = 4096` into converted GGUFs is legal and harmless, since gguf.cpp accepts any power of two. It only aligns tensor starts, not experts inside a tensor, so it does not replace step 3.

---

## 4. Model architectures to support in 2026

### 4.1 Dense and MoE: facts from configs and model cards

Total/active from model cards; the rest from `config.json`/`params.json` **[verified: docs]**.

| Model (release) | Total / active | Layers | Experts (routed + shared, top-k) | Attention | Native dtype / notes |
|---|---|---|---|---|---|
| Llama 4 Scout / Maverick (2025-04) | 109B / 17B; 400B / 17B | – | 16; 128 + 1 shared | iRoPE (interleaved no-RoPE layers) ([Meta](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)) | BF16 / FP8 |
| gpt-oss-120b / 20b (2025-08) | 117B / 5.1B; 21B / 3.6B | 36; 24 | 128 top-4; 32 top-4 | GQA 8 KV, head_dim 64; alternating sliding window 128 / full | MXFP4 experts |
| Gemma 4 31B / 26B-A4B / 12B / E4B / E2B (2026) | 30.7B dense; 26B / ~4B | 60; 30 | –; 128 top-8 | 5:1 sliding (1024) : global; global layers have 512-dim heads and K=V sharing (31B); E-models use per-layer embeddings | BF16; Google ships QAT Q4_0 GGUFs |
| Mistral Medium 3.5 (2026-03) | 128B dense | 88 | – | GQA 8 KV | FP8 |
| Mistral Large 3 (2025-12) | 675B / 41B | 61 (3 dense) | 128 + 1, top-4 | MLA (kv_lora 512, rope 64) | FP8 compressed-tensors; NVFP4 variant |
| DeepSeek V3 / R1 | 671B / 37B | 61 (3 dense) | 256 + 1, top-8 | MLA (kv_lora 512, rope 64) | FP8 128×128 |
| DeepSeek V4-Flash / V4-Pro (2026-04-24) | 284B / 13B; 1.6T / 49B | 43; 61 | 256 + 1 top-6; 384 + 1 top-6; `num_hash_layers: 3` | compressed attention: per-layer `compress_ratios` of 4 or 128, sliding window 128, indexer top-512/1024, 1 KV head of dim 512, 1M context | `expert_dtype: fp4`, FP8 rest |
| Kimi K2.5 (2026-01) | ~1T / 32B | 61 (1 dense) | 384 + 1, top-8 | MLA | compressed-tensors |
| **Kimi K3** (2026) | **2.8T / 104B** | 93 (1 dense) | **896 + 2 shared, top-16**, latent MoE (dim 3584) | **69 KDA (linear) + 24 gated MLA**, 1M context | **MXFP4 weights / MXFP8 activations, QAT** |
| Qwen3-235B-A22B (2025) | 235B / 22B | 94 | 128 top-8 | GQA 4 KV | BF16 |
| Qwen3.5-397B-A17B (2026-02) | 397B / 17B | 60 | 512 + 1, top-10 | 45 Gated DeltaNet + 15 gated attention (GQA 2 KV, head 256) | BF16 |
| Qwen3.6-35B-A3B (2026) | 35B / 3B | 40 | 256 top-8 | 30 GDN + 10 attention | BF16 |
| **Qwen3.8-Flash-Next** (2026-08-24) | 125B / 6B **+ 51B n-gram embedding** + 4B MTP | 48 | 512 + 1, top-10 | 36 GDN + 12 **Qwen Sparse Attention** (block-level, 2048-token budget) | 20M-entry bigram/trigram embedding at layer 2, "more amenable to offloading than MoE" (model card) |
| Qwen3.8-2.4T-A95B (2026-08) | 2.4T / 95B | 92 | 512 top-10 | 69 linear + 23 full | BF16 / FP8 |
| GLM-5 / 5.1 / 5.2 / 5.3 (2026-02…08) | GLM-5: 744B / 40B (5.3 config has the same shape) | 78 (3 dense) | 256 + 1, top-8 | MLA + DeepSeek Sparse Attention (indexer top-2048); 5.3: 1M context | FP8 |
| GLM-5.3-Flash (2026-08-25) | 320B / 18B | 45 | 288 + 1, top-8 | 34 linear + 11 DSA | FP8; natively multimodal |
| MiniMax M2 / M3 (2025-10 / 2026-06) | M3: ~428B / ~23B | 62; 60 | 256 top-8; 128 + 1, top-4 | GQA 8 / 4 KV; M3 adds block-sparse attention (top-16 blocks of 128) | FP8; M3 also MXFP8 repo |
| Nemotron 3.5 Bolt 74B-A9B (2026-10) | 74B / 9B | 88 blocks: 40 Mamba-2, 40 MoE, 8 attention | 512 + 1 | hybrid SSM | "native 2-bit" 24 GB variant |

libllama has graph files for every architecture in this table (`src/models/`) **[verified: code]**. That is the strongest single argument for building on libllama.

### 4.2 KV-cache and state size (computed from configs, BF16, per token)

Formulas **[inferred, standard]**:
- GQA: `2 · L_attn · n_kv · head_dim`
- MLA: `L · (kv_lora_rank + qk_rope_head_dim)`
- Linear/SSM layers: a fixed state per sequence, independent of length

| Model | Mechanism | KiB / token | GiB at 128K |
|---|---|---|---|
| GLM-4.6 (92 GQA layers, 8 KV) | GQA | 368.0 | 46.0 |
| Mistral Medium 3.5 128B | GQA | 352.0 | 44.0 |
| MiniMax M2 | GQA | 248.0 | 31.0 |
| Qwen3-235B-A22B | GQA | 188.0 | 23.5 |
| MiniMax M3 | GQA | 120.0 | 15.0 |
| Qwen3.8-2.4T (23 full) | hybrid | 92.0 | 11.5 |
| GLM-5 (latent; DSA indexer cache extra) | MLA | 87.8 | 11.0 |
| DeepSeek V3 / Kimi K2.5 / Mistral Large 3 | MLA | 68.6 | 8.6 |
| gpt-oss-120b (18 full; 18 SWA capped at 128 tokens) | GQA | 36.0 | 4.5 |
| Qwen3.5-397B (15 full) | hybrid | 30.0 | 3.75 |
| Kimi K3 (24 MLA; 69 KDA fixed state) | hybrid | 27.0 | 3.4 |
| Qwen3.8-Flash-Next (12 QSA) | hybrid | 24.0 | 3.0 |
| Qwen3.6-35B-A3B (10 full) | hybrid | 20.0 | 2.5 |

DeepSeek V4 compresses KV per layer by 4× or 128×. A third-party listing claims 10% of V3.2's KV cache at 1M context **[not verified]**.

Consequences for partitioning **[inferred]**:
- Hybrid models cut KV by roughly 4–10× against same-size GQA models, so long context becomes cheap. But their recurrent state is per-sequence and must move with the sequence when a pipeline stage migrates.
- Sparse attention (DSA, QSA, MiniMax block-sparse, DeepSeek CSA/HCA) adds an indexer and its own small cache, and reads only a top-k subset of KV per token. That makes **KV tiering to host RAM viable**: only the selected blocks need to be in VRAM. The single-GPU MoE engine prefetches KV this way from 64K context up (`tiered-memory-inference.md`).
- Pipeline splits must respect layer *type*. Linear and full-attention layers have very different per-token cost and state size, so splitting by equal layer counts is wrong.

### 4.3 Expert size and bytes moved per token (computed)

Params per expert = `3 · hidden · moe_intermediate`, using Kimi K3's latent dim of 3584 and Mistral Large 3's `expert_hidden_dim` of 4096. "Routed bytes/token" = top-k × MoE layers × expert bytes at Q4_K (4.5 bpw). That is the worst case, with no caching, no shared experts and no attention weights.

| Model | Params / expert | Expert @ Q4_K | Expert @ IQ2_XS | Routed bytes per token @ Q4_K |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 3.15M | 1.69 MiB | 0.87 MiB | 0.53 GiB |
| Gemma 4 26B-A4B | 5.95M | 3.19 MiB | 1.64 MiB | 0.75 GiB |
| Qwen3.8-Flash-Next | 4.92M | 2.64 MiB | 1.35 MiB | 1.24 GiB |
| gpt-oss-120b | 24.9M | 13.35 MiB (MXFP4 native: 12.61) | – | 1.88 GiB |
| DeepSeek V4-Flash (all 43 layers) | 25.2M | 13.50 MiB | 6.94 MiB | 3.40 GiB |
| MiniMax M2 | 14.2M | 7.59 MiB | 3.90 MiB | 3.68 GiB |
| Qwen3.5-397B-A17B | 12.6M | 6.75 MiB | 3.47 MiB | 3.96 GiB |
| GLM-5.3-Flash | 25.2M | 13.50 MiB | 6.94 MiB | 4.43 GiB |
| Qwen3-235B-A22B | 18.9M | 10.12 MiB | 5.20 MiB | 7.44 GiB |
| Mistral Large 3 | 88.1M | 47.25 MiB | – | 10.71 GiB |
| DeepSeek V3 | 44.0M | 23.62 MiB | 12.14 MiB | 10.71 GiB |
| Kimi K2.5 | 44.0M | 23.62 MiB | 12.14 MiB | 11.07 GiB |
| GLM-5 | 37.7M | 20.25 MiB | 10.41 MiB | 11.87 GiB |
| DeepSeek V4-Pro (all 61 layers) | 66.1M | 35.44 MiB | 18.21 MiB | 12.67 GiB |
| Qwen3.8-2.4T-A95B | 50.3M | 27.00 MiB | 13.88 MiB | 24.26 GiB |
| Kimi K3 | 33.0M | 17.72 MiB (MXFP4: 16.73) | 9.11 MiB | 25.47 GiB |

Reading this table **[inferred]**:
- A PCIe 4.0 x16 link (~25 GB/s practical, estimate) or a fast NVMe (~7 GB/s sequential, estimate) cannot feed 10–25 GiB per token. For the top half of the table, tiering only works with high cache hit rates plus CPU compute of misses in place. The two strongest surveyed tiering engines both do this; libllama's expert cache does not.
- Small-expert models (Qwen3.6/3.8, Gemma 4 MoE) have 1–3 MiB experts. SSD reads at that size hit IOPS and latency limits, not bandwidth limits, so I/O must be batched and queued deep (QD ≥ 32 estimate).
- New embedding-style tiers (Qwen3.8's 51B n-gram table, Gemma's per-layer embeddings) are row lookups of a few KB per token. They are the natural SSD residents. libllama's `--lazy-mode` and the single-GPU MoE engine's n-gram table on SSD already exploit this **[verified: code]**.

### 4.4 Support priority

**[inferred]**
- **Tier 1** (must run day one, every backend):
  - Qwen3.5/3.6/3.8 hybrid MoE (GDN + gated/sparse attention);
  - gpt-oss (MXFP4, sinks, SWA);
  - Gemma 4 dense + MoE;
  - Llama-architecture dense (Llama 3.x, Mistral Medium, Qwen dense);
  - DeepSeek-V3-family MLA MoE (also covers Kimi K2.5 and Mistral Large 3).
- **Tier 2**: GLM-5.x (MLA + DSA), MiniMax M2/M3, DeepSeek V4 (new compressed attention), Nemotron-H hybrids.
- **Tier 3** (distributed-only scale): Kimi K3 (KDA + latent MoE, 2.8T), Qwen3.8-2.4T, DeepSeek V4-Pro.

---

## 5. Packaging and zero-config install

### 5.1 How surveyed runners ship

| Project | Artefacts | GPU libraries | Signing | Update |
|---|---|---|---|---|
| **ggml / libllama upstream** (b11724) | Per-backend archives: win cpu/cuda-12.4/cuda-13.4/vulkan/sycl/rocm-10.0/openvino/opencl-adreno (x64 and arm64); ubuntu cpu/cuda-12.8/cuda-13.4/vulkan/rocm-10.0/sycl-fp16/fp32/openvino/s390x; macos arm64/x64; android arm64 (+snapdragon); xcframework. Separate `cudart-*` archives **[verified: docs]** | CUDA runtime redistributed as separate archive | – | manual |
| **Go desktop runner** (v0.40.6) | Linux amd64 `.tar.zst` 1371 MiB, ROCm variant 1003 MiB, MLX variant 1234 MiB; Windows installer 1511 MiB; macOS `.dmg` 204 MiB; Jetpack 5/6 builds **[verified: docs]** | per-variant library dirs (`cuda_v12`, `cuda_v13`, `rocm_v*`, `mlx_cuda_*`); a dedup script symlinks identical CUDA `.so` files between the MLX and ggml dirs **[verified: code]** | macOS: `codesign --options=runtime` + `notarytool`. Windows: `signtool` with timestamp, Inno Setup installer **[verified: code]** | own updater **[verified: code]** |
| **Single-file runner** 0.10.6 (2026-09-15) | One APE executable (Linux 2.6.18+, macOS 23.1+, Windows 10+ AMD64, Free/Net/OpenBSD). Windows can't run executables over 4 GB, so weights go external **[verified: docs]** | Looks for `ggml-(cuda\|rocm\|vulkan)` libraries next to the exe, in the bundle, in a versioned per-user cache dir, then `$HOME`. Windows releases include prebuilt CUDA/ROCm DLLs. Linux compiles if `nvcc`/`hipcc` is on PATH. macOS compiles a Metal module on first run with Xcode CLT **[verified: docs]** | – | manual |
| **Python/MLX cluster runner** | PyInstaller bundle in a Swift `.app`, DMG | MLX via pip wheels | Developer ID, hardened runtime, notarytool **[verified: code]** | Sparkle, appcast on S3 **[verified: code]** |
| **Tauri desktop app** 0.8.6 (2026-10-09) | `.exe`, Microsoft Store, universal `.dmg`, `.deb`, AppImage, Flatpak **[verified: docs]** | ggml/libllama engine variant per build (cpu/vulkan/metal/cuda12/cuda13/hip) **[verified: docs]** | not checked | Tauri (mechanism not checked) |
| **Closed-source desktop app** | AppImage (Linux x64/arm64), Windows x64/arm64, macOS 14+ arm64 only; AVX2 required on x64 **[verified: docs]** | runtimes as separately updated extension packs, least-recently-used ones auto-deleted (2025 release) **[verified: docs]** | not checked | in-app |
| **Single-GPU MoE engine** | git clone + a Windows batch file / `setup.sh`; Mac builds from source in ~7 min **[verified: code]** | relies on the installed driver | – | update script |

### 5.2 Runtime backend selection

- **ggml `GGML_BACKEND_DL`** (needs `BUILD_SHARED_LIBS`):
  - Each backend becomes a CMake `MODULE`, loaded at runtime with `ggml_backend_load_all_from_path()`.
  - `GGML_CPU_ALL_VARIANTS` builds CPU modules for x64, sse42, sandybridge, ivybridge, piledriver, haswell, skylakex, cannonlake, cascadelake, icelake, cooperlake, zen4, alderlake and sapphirerapids (AMX), and the best one is picked at load.
  - On macOS, backend modules are not versioned (CMake limitation).
  - (`ggml/src/CMakeLists.txt` L188–596) **[verified: code]**.
  - **This is the right mechanism for Syntropy: one core binary, backend modules downloaded or bundled per platform.** **[inferred]**
- **Subprocess discovery** (the Go desktop runner's pattern): run the backend once without a model, parse its device list (`CUDA0: … (16379 MiB, 14900 MiB free)`, compute capability, VMM, wave size, gfx target), then decide **[verified: code]**. In-process, `ggml_backend_dev_get_props` gives the same data without a subprocess **[verified: code]**.
- **Driver detection** without SDKs **[inferred]**:
  - NVIDIA: dlopen `libcuda.so.1` / `nvcuda.dll` and `libnvidia-ml.so.1` / `nvml.dll`, which ship with the driver.
  - AMD: `libamdhip64` (ROCm runtime or Windows driver) or amdgpu sysfs.
  - Vulkan: the loader is present on all mainstream desktops.
  - Metal: always on macOS arm64.
  - Fall back down the chain CUDA → HIP → SYCL → Vulkan → CPU, the order the single-file runner documents **[verified: docs]**.

### 5.3 Redistribution licensing

- **CUDA** (EULA v13.4, updated 2026) **[verified: docs]** ([EULA](https://docs.nvidia.com/cuda/eula/index.html)):
  - Attachment A redistributables include `cudart`, `cublas`, `cublasLt`, `nvrtc`, `nvrtc-builtins`, `nvJitLink`, `cufft`, `curand`, `cusparse`, `cusolver`, `nvblas`, `nvvm` and `cupti`, plus `libcuda.so` and `libcufile.so` on Linux.
  - Conditions: object code only, inside an application with "material additional functionality", not standalone, and Linux files unmodified.
  - Bundling cuBLAS is allowed. It is also why the Go desktop runner's archives exceed 1.3 GiB **[inferred]**.
- **CUDA 13 dropped offline compilation for Maxwell, Pascal and Volta** (CC < 7.5). Driver branch 580 is the last for those GPUs ([TechSpot](https://techspot.com/news/108523-nvidia-drop-driver-support-maxwell-pascal-volta-gpus.html), secondary) **[verified: docs]**. Upstream ggml/libllama and the Go desktop runner ship both CUDA 12 and CUDA 13 builds for this reason **[verified: docs/code]**. MLX's CUDA path requires SM ≥ 7.5 **[verified: docs]**.
- **ROCm**:
  - Licensing: almost all components are MIT/Apache/BSD/NCSA; hipCUB and RCCL are "custom"; a few profiler pieces are binary-only ([licence page](https://rocm.docs.amd.com/en/latest/about/license.html)) **[verified: docs]**.
  - Packaging (TheRock): ROCm moved to the TheRock build with `amdrocm-*` packages, current production 10.1.0. Each package comes in an all-architecture variant and per-family variants (`gfx110X-all`, `gfx120X-all`, …). AMD recommends the all-arch variant for redistribution. ROCm SMI and ROCm Bandwidth Test were removed (use AMD SMI) ([transition guide](https://rocm.docs.amd.com/en/docs-10.1.0/about/transition-guide-TheRock.html)) **[verified: docs]**.
  - Windows: upstream ggml/libllama ships `win-rocm-10.0` builds **[verified: docs]**.
- **Vulkan/Metal/DirectX**: system components, nothing to redistribute **[inferred]**.

### 5.4 Signing, notarisation, auto-update

- **macOS**:
  - Developer ID + hardened runtime + `notarytool`. Every nested binary and dylib must be signed before the app; both surveyed macOS runners sign each runtime file individually **[verified: code]**.
  - Since Sequoia, users can no longer Control-click to bypass Gatekeeper for unnotarised software and must go through System Settings ([Cult of Mac](https://www.cultofmac.com/864843/macos-sequoia-removes-handy-shortcut-bypass-gatekeeper-security/)) **[verified: docs]**.
  - **Notarisation is effectively mandatory.** **[inferred]**
  - Compiling a Metal module on first run (the single-file runner's approach) needs Xcode CLT and is incompatible with a notarised, no-toolchain install **[inferred]**.
- **Windows**:
  - Authenticode via `signtool`.
  - Azure Artifact Signing (formerly Trusted Signing). Individuals are eligible only in the USA and Canada. SmartScreen reputation still builds over time ([Microsoft Learn](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options)) **[verified: docs]**.
- **Linux**: no OS signing gate. Distribution through tarball + install script (the Go desktop runner), AppImage, Flatpak, `.deb` **[verified: docs]**.
- **Auto-update**: Sparkle (the Python cluster runner), a custom updater (the Go desktop runner), separately updated runtime packs (the closed-source desktop app).
  - Recommendation: update the app and the backend modules separately, with a signed manifest (minisign/ed25519) listing module hashes per platform and driver range **[inferred]**.
- **iOS**:
  - No downloaded executable code, so backends must be statically linked (Metal, CPU) **[inferred, App Store rule]**.
  - Usable memory is per-process limited. The `increased-memory-limit` entitlement helps on "some device models", and Apple says to size with `os_proc_available_memory()` ([Apple](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit)) **[verified: docs]**.
  - One developer reports a flat ~6 GB ceiling on both 8 GB and 12 GB iPhones even with the entitlement ([forum](https://developer.apple.com/forums/thread/834805)) **[verified: docs]** (single report).
- **Android**:
  - io_uring is blocked for apps by seccomp ([Phoronix](https://www.phoronix.com/news/Google-Restricting-IO_uring)) **[verified: docs]**.
  - GPU paths are Vulkan, OpenCL (Adreno) and Hexagon. Upstream ggml/libllama ships `android-arm64` and `-snapdragon` builds **[verified: docs]**.

---

## 6. Hardware capability detection

### 6.1 Enumeration APIs per platform

| What | Linux | Windows | macOS / iOS | Android |
|---|---|---|---|---|
| GPU list, VRAM total/free | `ggml_backend_dev_get_props` (all backends) **[verified: code]**. NVML `nvmlDeviceGetMemoryInfo(_v2)`, `nvmlDeviceGetBAR1MemoryInfo` ([NVML](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)) **[verified: docs]**. amdgpu sysfs `mem_info_vram_total/used`, `mem_info_vis_vram_total` (CPU-visible VRAM; reveals Resizable BAR), `mem_info_gtt_total` ([kernel docs](https://docs.kernel.org/gpu/amdgpu/driver-misc.html)) **[verified: docs]**. AMD SMI `amdsmi_get_gpu_memory_total` **[verified: docs]**. Vulkan `VkPhysicalDeviceMemoryBudgetPropertiesEXT.heapBudget/heapUsage` ([spec](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_memory_budget.html)) **[verified: docs]** | DXGI `IDXGIAdapter3::QueryVideoMemoryInfo` gives an OS **budget**, not just the total, plus a change notification ([docs](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo)) **[verified: docs]**. DXCore for compute-only adapters/NPUs **[inferred]**. NVML ships with the driver **[inferred]** | `MTLDevice.recommendedMaxWorkingSetSize`, `hasUnifiedMemory` (Metal docs). Default working-set cap is about 2/3–3/4 of RAM, raisable with `sysctl iogpu.wired_limit_mb` (not persistent, undocumented by Apple) (third-party guides) **[verified: docs]**. iOS: `os_proc_available_memory()` | Vulkan budget extension; `ActivityManager.MemoryInfo` **[inferred]** |
| Unified memory | `ggml` device type `IGPU`; ggml disables mmap loading on iGPUs for CUDA and Vulkan **[verified: code]** | integrated adapter flag in DXGI/DXCore **[inferred]** | always (Apple silicon) | always |
| System RAM | `/proc/meminfo` plus **cgroup v2 `memory.max`** in containers. The home-cluster ggml fork reads both, plus Docker detection and Termux swap **[verified: code]** | `GlobalMemoryStatusEx` **[inferred]** | `sysctl hw.memsize`, `vm_stat` **[inferred]** | `/proc/meminfo`, MemoryInfo **[inferred]** |
| PCIe link | `/sys/bus/pci/devices/<bdf>/current_link_speed`, `current_link_width` **[inferred]**. NVML `nvmlDeviceGetCurrPcieLinkGeneration/Width` **[verified: docs]**. amdgpu `pcie_bw` **[verified: docs]** | NVML; SetupAPI device properties **[inferred]** | n/a (fabric) | n/a |
| NVMe devices | `/sys/block/nvme*n*/queue/{rotational,logical_block_size,physical_block_size,nr_requests}`, controller PCI link through `/sys/class/nvme/*/device`; DIO alignment via `statx(STATX_DIOALIGN)` **[verified: docs for statx; rest inferred]** | `IOCTL_STORAGE_QUERY_PROPERTY` (`StorageAdapterProperty` → `BusTypeNvme`, `StorageDeviceSeekPenaltyProperty`) **[inferred]** | IOKit registry (`IONVMeController`) **[inferred]** | not exposed to apps **[inferred]** |
| NUMA | `/sys/devices/system/node`; pin CPU threads to the GPU's node (the upstream expert-cache benchmarks pin to the GPU's NUMA node) **[verified: docs]** | `GetNumaNodeProcessorMaskEx` **[inferred]** | n/a | n/a |
| Async disk API | io_uring (may be disabled: `io_uring_disabled` sysctl since 6.6, Docker's default seccomp blocks it, ChromeOS disables it) ([Phoronix](https://www.phoronix.com/news/Linux-6.6-sysctl-IO_uring)) **[verified: docs]**. So keep a `pread` thread-pool fallback, as the single-GPU MoE engine does (see `tiered-memory-inference.md`) | IoRing (`BuildIoRingReadFile`, Windows 11) ([docs](https://learn.microsoft.com/en-us/windows/win32/api/ioringapi/nf-ioringapi-buildioringreadfile)) **[verified: docs]** or IOCP | `F_NOCACHE` + `pread`/dispatch_io **[inferred]** | `pread` thread pool |

Apple residency note: Metal 4 makes residency explicit through `MTLResidencySet`, attachable per queue (up to 32 sets) or per command buffer. Sparse heaps are not supported in residency sets ([Apple](https://developer.apple.com/documentation/metal/mtlresidencyset)) **[verified: docs]**. On unified memory, the "VRAM tier" is really the wired working set. Tiering on Mac is RAM-resident vs SSD-resident, governed by `recommendedMaxWorkingSetSize`, not PCIe uploads. **[inferred]**

### 6.2 Startup microbenchmarks

Prior art:
- The home-cluster ggml fork measures CPU and GPU FLOPS per quant type through ggml, memory bandwidth, VRAM read bandwidth, host↔device copy, and disk sequential/random read by **shelling out to `fio`** **[verified: code]**. That is an external dependency, which conflicts with zero-config.
- libllama's `--fit` uses free memory, not measured bandwidth **[verified: code]**.
- Capability measurement in the surveyed cluster runners is covered in `multi-device-inference.md`.

Proposed set, all in-process, cached per (hardware fingerprint, driver version, Syntropy version), total budget ≤ 10–20 s on first run (estimate) **[inferred]**:
1. **Host memory bandwidth**: multi-threaded triad over a 1–2 GiB buffer, per NUMA node.
2. **H2D/D2H pinned copy bandwidth and latency** per GPU: `ggml_backend_tensor_set_async` from a pinned host buffer at 1, 4, 16 and 64 MiB. Detect PCIe gen/width and Resizable BAR from NVML/sysfs to sanity-check.
3. **NVMe read** per candidate model location: O_DIRECT at the measured `stx_dio_offset_align`, with block sizes matching the model's expert slices (from §4.3: 1–48 MiB) and 4 KiB random reads, at QD 1/8/32. Use io_uring if permitted, else a thread pool. Use a scratch file the size of the reads, or reads of the model file itself, which avoids writes on the user's SSD.
4. **Kernel throughput** per backend and quant type: decode-shaped mat-vec and prefill-shaped mat-mat (`mul_mat_id` with realistic top-k) for the actual model dims. That is the number the placement planner needs, not peak TFLOPS.
5. **Network**: RTT and bulk bandwidth to each peer (owned by the distribution design; see `multi-device-inference.md`).
6. **Re-check on change**: driver update, a new GPU, or measured budget drift (DXGI budget notification, Vulkan `heapBudget`).

---

## 7. Recommendations

### 7.1 Candidate stacks

**A. Rust orchestrator + vendored libllama/ggml through the C ABI (recommended)**
- **Shape**:
  - A Rust workspace owns discovery, the placement planner, the tier I/O scheduler, networking, the API and model management.
  - ggml/libllama is a git submodule built with `GGML_BACKEND_DL=ON` and `GGML_CPU_ALL_VARIANTS=ON`, called through `bindgen` on `llama.h` + `ggml-backend.h`.
  - Placement goes through `tensor_buft_overrides`, `moe_cache_size`, custom buffer types, and a small number of upstreamable patches: a pluggable expert-cache policy with prefetch hooks, SSD-backed buffer type, and a scheduler hook for cross-node splits.
- **Pros**:
  - Every backend and OS in the brief on day one.
  - Every 2026 architecture already has a graph file.
  - GGUF ecosystem compatibility.
  - Upstream is moving toward tiering anyway (expert cache, direct I/O, lazy rows, `--fit`, tensor split), so patches have a home.
  - Rust gives safe networking and concurrency, and a uniffi path to mobile.
- **Cons**:
  - Two build systems (cargo + CMake). libllama's API churn needs a pin-and-bump cadence: a pinned upstream build plus a local compat patch queue, as the Go desktop runner keeps.
  - Fine-grained control inside the forward pass is limited to what libllama exposes. Anything deeper needs a fork or upstream PRs.
  - Computing cache misses in place on the CPU, per expert, has to be built through ggml's scheduler, which today splits by tensor, not by expert subset per token.

**B. C++ end to end on a ggml/libllama fork (the approach of the home-cluster fork and the single-GPU MoE engine)**
- **Pros**: fastest path to top single-box performance, with full control of graph, scheduler and buffers. Two surveyed engines prove the approach.
- **Cons**:
  - Fork drift. The home-cluster fork's model list is still Llama/Qwen 2.5/QwQ/DeepSeek-R1-distill **[verified: code]**, which is what happens without upstream sync.
  - Memory-unsafe networking and discovery code.
  - Harder mobile bindings (manual C ABI + Swift/Kotlin wrappers).

**C. Pure Rust: candle or CubeCL kernels (the surveyed Rust engines' approach), MLX as the Apple backend**
- **Pros**: one toolchain, best safety, WebGPU/WASM reachable through wgpu/CubeCL, easiest mobile story.
- **Cons**:
  - candle has no ROCm or Vulkan; the Rust distributed runner had to write its own Vulkan path **[verified: code]**.
  - CubeCL has the backends but no LLM kernel library for i-quants, MXFP4 MoE, GDN, KDA or DSA. Weeks to months of kernel work per architecture family (estimate).
  - Smaller community than ggml (about 10M lifetime crate downloads for candle-core vs ggml/libllama's multiple daily releases **[verified: docs]**).

**Not recommended as the base: Python + MLX (the cluster runner's approach).**
- Strong on Apple and gaining Linux CUDA.
- But it pins Python 3.13, ships through PyInstaller, lists only Apple Silicon as Tier 1, and has no placement API **[verified: code]**.
- Use MLX as an optional backend on macOS through `mlx-c`, as the Go desktop runner does, if benchmarks justify it. The evidence is mixed (§1.4). **[inferred]**

### 7.2 Risks

1. **Kernel coverage lag for new attention types.** KDA (Kimi K3), DeepSeek V4 compressed attention, QSA (Qwen3.8), DSA (GLM-5.x) and Mamba-2 hybrids land first on CUDA. Vulkan, Metal, SYCL and WebGPU follow later or never. A "runs everywhere" promise has to be per model × backend, with a CPU fallback per op (ggml's scheduler already falls back per op) **[inferred]**.
2. **libllama API churn and governance.** ggml is 0.27 and libllama just versioned 0.7.0 **[verified: code]**. Hugging Face now employs the ggml team **[verified: docs]**, which helps continuity but means upstream priorities follow HF's.
3. **Binary size and redistribution.** CUDA 12 + 13 + ROCm push installs past 1–2.5 GiB (the Go desktop runner: 1371 MiB + 1003 MiB on Linux) **[verified: docs]**. Mitigation: download backend modules after detection, signed. A licence check of the CUDA "material additional functionality" condition is needed for a module-only download **[inferred]**.
4. **I/O APIs blocked in common environments.** io_uring is off in Docker's default seccomp, on Android and on ChromeOS. GDS is not available on consumer setups. Windows needs IoRing/IOCP **[verified: docs]**. A portable thread-pool `pread` path is mandatory.
5. **Mobile memory ceilings.** iOS per-process caps (~6 GB reported even on 12 GB phones) and no runtime code download. Phones are compute nodes for small models or layer slices, not hosts for large tiers **[inferred]**.
6. **Unified-memory budgets.** macOS caps the GPU working set at ~2/3–3/4 of RAM unless a sysctl is changed; an installer must not do that silently **[inferred]**.
7. **Expert sizes push the SSD tier toward IOPS limits.** 1–3 MiB experts (Qwen3.6/3.8, Gemma 4 MoE) and 10–25 GiB/token worst-case routed traffic for frontier MoE (§4.3) **[inferred, computed]**.
8. **FP8 checkpoints have no ggml type.** Import costs a conversion pass, and fidelity depends on the target quant **[verified: code]** for the type gap, **[inferred]** for impact.
9. **Benchmarks in this space are mostly self-reported.** None of the cross-engine numbers in §1.4 were reproduced. Syntropy should run its own matrix, using the ggml project's benchmark tool, MLX and candle, on 3–4 reference machines before locking the kernel choice.

### 7.3 Open questions for the design

1. **libllama or ggml?** Does Syntropy bind libllama (inherit ~160 architecture graphs, accept its placement model) or bind ggml only and own model graphs? This decides how much code Syntropy writes and keeps current, in tasks and upkeep, more than any other choice.
2. **Upstream or fork?** Should Syntropy upstream tiering hooks into ggml/libllama (expert-cache policy, prefetch API, SSD buffer type), or is a long-lived fork acceptable?
3. **MLX on macOS?** Is MLX worth carrying as a second backend for v1, given the mixed evidence (§1.4)?
4. **Which NPUs, if any?** QNN through ExecuTorch, a vendor mobile runtime or ORT, or ggml's Hexagon/OpenVINO backends? They imply different model formats.
5. **Minimum GPU generation?** Shipping only CUDA 13 drops Pascal/Volta users. Shipping both doubles the CUDA payload.
6. **Signing identity?** Azure Artifact Signing for individuals is US/Canada only.
