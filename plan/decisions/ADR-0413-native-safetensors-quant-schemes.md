# ADR-0413: Safetensors quantization schemes map one-to-one onto native `QuantScheme`s; auxiliary tensors bind to their weight

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0005, ADR-0014, ADR-0019; RFC-0001 §5.1, §5.3; RFC-0041 §2.2; RFC-0042 §2, §3

## Context

- ADR-0005: safetensors originals are read natively (BF16, FP16, FP8 E4M3/E5M2 per-tensor and block-scaled, MXFP4, NVFP4, compressed-tensors), with no lossy conversion when the runtime has the type; a device without a kernel may get an exact widened copy in its pack.
- 2026 checkpoints store quantized weights as several tensors: FP8 block-scaled weights with a separate per-block scale tensor (DeepSeek-V3/V4, GLM-5.x, MiniMax M2); MXFP4 as packed blocks plus E8M0 scales (gpt-oss); compressed-tensors with packed INT4 plus scales and optional zero points (Kimi-K2-Thinking's INT4 group-32 experts), or NVFP4 with per-16 FP8 scales plus a per-tensor global scale (`backends-and-portability.md` §3.1–§3.2, §4.1; cost model §3.9).
- Mixed precision within one checkpoint is common (an `ignore` list leaves attention and heads in BF16), and per-tensor dtype must survive into the census because it decides per-token bytes.
- A shard that held a weight without its scales would be unusable.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Convert to GGUF types at import | One execution format | No GGUF FP8 type; lossy or widening; ADR-0005 rejected it |
| B. Dequantize to BF16 at import | Simple kernels | 2–4× the bytes per token, the opposite of what the cost model rewards |
| **C. Map each `quantization_config` (method, format, strategy) to one `QuantScheme`; bind scale, zero-point and global-scale tensors to their weight as auxiliary extents of one logical `TensorRecord`; refuse anything unmapped** | Native precision and tensor-core paths kept; groups are self-contained; unknown schemes fail closed | A mapping table to maintain as checkpoint conventions evolve |

## Decision

**Option C.**

1. The mapping is RFC-0041 §2.2's table: `fp8` (block 128×128, per-tensor, per-channel), `mxfp4`, and compressed-tensors `pack-quantized`, `float-quantized`, `nvfp4-pack-quantized` and `mxfp4-pack-quantized`, each to one `QuantScheme`. Modules on the `ignore` list keep their stored dtype.
2. Auxiliary tensors are bound by the naming convention of each method and stored in `TensorRecord::aux`. A group carries them, so chunk sets, shards and pack extents always include them.
3. An unknown method, format or strategy is `Unsupported (Formats, 4)` naming it. Nothing is guessed.
4. Where a device lacks a kernel for a scheme, the pack holds an exact widened copy (RFC-0042 §3), recorded as a non-`SOURCE` layout; the source stays untouched.
5. Requantization to a smaller type happens only through RFC-0042 §9's explicit tools and yields a new `ModelId`.

## Consequences

- Positive: frontier checkpoints run at their native precision; the solver sees true per-tensor bytes, including BF16 hot sets next to INT4 experts.
- Negative: every new checkpoint convention needs a mapping row and parity tests before it loads; until then it is refused.
- Negative: widened copies cost pack disk on devices without native kernels for the scheme.

## What would make us revisit

- A widely used checkpoint family ships a scheme that cannot be expressed as one `QuantScheme` plus auxiliary tensors (for example, codebooks shared across tensors). Extend the census model rather than convert.
- A successor container format standardises quantization metadata and gains Hub adoption (ADR-0005's trigger).
