# ADR-0005: Model formats

- Status: Accepted; revised 2026-10-20 (decisions 4 and 6)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D3, D7), ADR-0002, ADR-0003, ADR-0007, ADR-0013, ADR-0015, ADR-0016, ADR-0019, RFC-0021, RFC-0023, RFC-0041, RFC-0042, RFC-0054

## Context

Users arrive with two kinds of model files (`plan/research/backends-and-portability.md` §3.1, §3.3):

- **GGUF** is the dominant local format: a header, KV metadata, tensor infos and a data section aligned by `general.alignment` (any power of two, default 32). MoE experts are stored as `{n_embd, n_ff_exp, n_expert}` tensors with the expert index outermost, so each expert's slice of each matrix is one contiguous byte run. That run is a multiple of the quant block size (144 bytes per 256 weights for Q4_K), not of 4096, so expert slices start at arbitrary 32-byte-aligned offsets. Hub GGUF repos split files at about 50 GB.
- **safetensors** is the default original format: a JSON header (≤ 100 MB) and one packed buffer, with tensors aligned only to their element size. Its dtypes include F4 (MXFP4), F6 and F8 E4M3/E5M2/E8M0. Large checkpoints shard into `model-0000N-of-0000M.safetensors` files with an index JSON. Quantized originals declare `quantization_config` (`fp8`, `mxfp4`, `compressed-tensors`).

What the 2026 models ship as (§3.2, §4.1):
- **FP8 block-scaled** (128×128 blocks): DeepSeek V3/V4, GLM-5.x, MiniMax M2. ggml has no FP8 tensor type, so a ggml-only runtime must convert these at import (§3.2, §7.2 risk 8).
- **MXFP4** natively or through quantization-aware training: gpt-oss, Kimi K3, DeepSeek V4 FP4 experts. **NVFP4** variants exist for Mistral Large 3 and Nemotron. Requantizing them costs fidelity and loses the tensor-core FP4 path (ADR-0014).
- **Mixed precision inside one checkpoint**: Kimi-K2-Thinking ships INT4 routed experts with a BF16 hot set, and the BF16 hot set dominates per-token bytes (`hardware-and-performance-model.md` §3.9). Per-tensor dtype must survive into the manifest.

What the I/O path needs (ADR-0007, ADR-0013):
- **One aligned request per expert.** Laying the file out in the unit that is read took decode from 7.37 to 8.62 tok/s (+17%) on an M4 Mac mini (`tiered-memory-inference.md` §8, §10.4).
- **Direct-I/O alignment.** Linux reports per-file alignment through `statx(STATX_DIOALIGN)` (since 6.1; not on btrfs at the time). Windows unbuffered reads need sector-aligned offsets and buffers. macOS `F_NOCACHE` has no alignment rule (`backends-and-portability.md` §3.1).
- **Partial fetch**, so a node reads and stores only its shard (ADR-0016).

Costs seen elsewhere: a surveyed engine's repack needed about 40 GB of extra disk, and a multi-GB repack at setup is a listed install anti-pattern (`tiered-memory-inference.md` §19, §20 item 13).

The research recommended GGUF unchanged plus conversion of safetensors to GGUF at import, mainly because ggml lacks FP8 (`backends-and-portability.md` §3.4). ADR-0002 removes that constraint, because Syntropy's own kernels can carry the missing types.

## Options considered

| Option | For | Against |
|---|---|---|
| A. A new Syntropy interchange format | Ideal layout | Fragments the ecosystem; users must convert; we own a spec |
| B. GGUF canonical; convert safetensors to GGUF at import | One parser for execution | FP8 has no GGUF type, so the conversion is lossy or widening; MXFP4 and NVFP4 scale layouts must map exactly; an import pass for every original |
| C. Read sources in place, no derived layout | No extra disk | Experts are not 4 KiB-aligned; read-around wastes bandwidth and needs bounce buffers; no backend repacks |
| **D. Read GGUF and safetensors natively; derive a per-device pack** | Users' files stay untouched; no lossy step; aligned, read-unit layout for the I/O path | Two source parsers; a local derived cache to manage |

## Decision

1. **Inputs.**
   - **GGUF, read unchanged**, is the primary path, including split files. Existing local files are indexed in place.
   - **safetensors originals are read natively**: BF16, FP16, FP8 E4M3/E5M2 (per-tensor and block-scaled), MXFP4, NVFP4 and compressed-tensors packings, with the sharded index.
   - **No lossy conversion when the runtime has the type.** Syntropy's native providers add the types ggml lacks: FP8 E4M3/E5M2 with block scales, and NVFP4 with block scales. A device with no kernel for a source type may get a widened copy in its pack (FP8 or FP4 to BF16, for example), which is exact. Requantization to a smaller type happens only on explicit user request and produces a new model identity with its own content hash.
2. **Both formats produce one model manifest** (RFC-0041): architecture family and hyperparameters (ADR-0003), the tensor census, byte sizes per quant type, per-tensor dtype, and the KV bytes-per-token formula. Header parsing happens in safe Rust with size limits (ADR-0001).
3. **The pack is the per-device execution layout**, stored in a local cache and keyed by (model content hash, device class, shard digest, layout version), matching `PackKey` in RFC-0001 §5.6. It holds only the node's shard, so a new plan epoch that leaves a node's shard unchanged reuses its pack as is.
   - Every expert and every streamable unit starts on a **4 KiB boundary**, or on the larger alignment that `STATX_DIOALIGN` or the sector size requires. An expert's matrices are contiguous, so one read fetches the whole expert.
   - **Optional backend-specific layouts**: CPU kernel interleaves, tensor-core FP4/FP8 swizzles, Metal tensor layouts.
   - A **sidecar manifest** records offsets, sizes and BLAKE3 hashes. The learned expert-frequency profile (RFC-0023), the op-selection entries and the canary's reference digest are separately versioned side records, so they survive a layout or container change that rebuilds the pack (RFC-0041 §5.2; revised 2026-10-19).
   - Units inside the pack are content-addressed. A new plan epoch reuses unchanged units by hash and writes only what changed.
4. **The pack is rebuildable and safe to delete.** It is built during the runner's `Packing` state (ADR-0015), from the shard's source bytes, with no whole-model blocking step. If a unit is not yet packed, or the pack exceeds its disk budget (RFC-0054), the I/O engine reads that unit from the source with an aligned read-around and reports the lower read efficiency to the capability profile.
   - *Revised 2026-10-20:* when free space would otherwise fall below the store's threshold, the store punches source ranges that a committed `SOURCE`-layout pack unit already holds byte for byte, after recording where they are; reads, re-plans and peer serving then take those ranges from the pack. The pack stays safe to delete: its punched ranges are written back when there is room, or refetched from the cluster or the origin when next needed (RFC-0041 §4.5). The picker always shows the true disk need.
5. **Packs stay local.** Nodes exchange source-format chunks (ADR-0016), never device-specific packs.
6. **No new interchange format is published.** *Revised 2026-10-20:* nor are packs or trained draft heads derived from third-party models, whose licences carry duties to anything derived from them; draft heads are dropped from v1.

## Consequences

Positive:
- Any GGUF or supported safetensors checkpoint runs without a conversion step, and users' files stay byte-identical for other tools.
- FP8, MXFP4 and NVFP4 checkpoints keep their native precision and their tensor-core fast paths.
- The NVMe tier gets one aligned request per expert, which is the cheapest I/O win measured.

Negative:
- Two source parsers and two quantization-metadata conventions to maintain and fuzz (RFC-0060).
- Packs cost disk, up to a shard's size per device class. RFC-0054 sets the budget and the per-model LRU eviction.
- Native FP8 and NVFP4 kernels are Syntropy's job on every backend that wants them fast (ADR-0002).

Work created: RFC-0041 (parsers, manifest, pack layout, content hash), RFC-0021 (read-around and aligned reads), RFC-0023 (frequency profile in the sidecar), RFC-0042 (FP8/NVFP4 numerics, widening rules), RFC-0054 (pack disk budget).

## What would make us revisit

- Packed reads beat aligned read-around from the source by less than 5% decode on Tier 1 machines. The pack would then be optional, not the default.
- Pack overhead exceeds 30% of shard size on typical Tier 1 devices. Store backend repacks in RAM at load instead of on disk.
- A successor format gains broad Hub adoption and natively aligns experts and carries FP4/FP8 block scales.
- Hub publishers start shipping GGUFs whose experts are already 4 KiB-aligned. The pack's alignment role would then disappear for those files.

## Revisions

- Revised 2026-10-19: decision 3: the expert-frequency profile moves from the sidecar into a separately versioned side record, so a pack rebuild does not lose what the cache has learned. The pack key was already (model content hash, device class, shard digest, layout version).
- Revised 2026-10-20: decision 4: under disk pressure the source ranges a `SOURCE`-layout pack unit duplicates are punched, so an NVMe-tiered MoE plan no longer needs about 1.7–2× the model's size on disk [estimate, RFC-0041 §4.5]. Decision 6: the project publishes no packs or draft heads derived from third-party models.
