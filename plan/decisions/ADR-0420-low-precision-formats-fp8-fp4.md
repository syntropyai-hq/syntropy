# ADR-0420: FP8 and FP4 weights: native where it is exact, exact widening everywhere else, scales on F32 partial sums

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0005, ADR-0014, ADR-0019, ADR-0413; RFC-0040 §7, RFC-0042 §2, §3

## Context

- Frontier checkpoints ship FP8 block-scaled (128×128) weights (DeepSeek-V3/V4, GLM-5.x, MiniMax M2), MXFP4 (gpt-oss, Kimi K3, DeepSeek V4's FP4 experts) and NVFP4 variants (`backends-and-portability.md` §3.2, §4.1). ggml has no FP8 type; Syntropy's native providers add FP8 and NVFP4 (ADR-0005).
- ADR-0014's fallback chains assume widening is lossless: "FP4 values are exact in FP16", "lossless FP8→FP16 widening".
- E2M1 and E4M3/E5M2 elements are exact in FP16 and BF16. The MX shared scale is a power of two from 2^−127 to 2^127, outside FP16's range. A BF16 or F16 group scale times a 4-bit integer can need 12 significant bits, one more than FP16 holds. NVFP4's E4M3 block scale times an E2M1 element needs at most 6 significant bits and stays within [2^−10, 2688].
- Decode is memory-bound, so a widened GEMV costs no more than a native-format one. Prefill is compute-bound, so tensor-core FP8/FP4 formats pay there. Those tensor-core paths need low-precision activations as well (RFC-0042 §4.3).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Convert FP8/FP4 checkpoints to a GGUF type at import | One kernel family | Lossy or widening; FP8 has no GGUF type; ADR-0005 rejected it |
| B. Widen by folding every scale into FP16 elements | Simple kernels | Inexact for MX scales and for INT4 group scales, so the "lossless" fallbacks of ADR-0014 would not be |
| **C. Elements widen exactly; a scale is folded into the element only when the product is provably exact in the target type (NVFP4's E4M3 block scale), otherwise applied to the F32 partial sum of its block; native tensor-core paths in prefill where their activation handling is allowed** | Every fallback exact by construction; tensor-core speed where it is honest | Kernels must handle per-block scaling of partial sums |
| D. Always use the native tensor-core format, quantising activations to match | Fastest prefill | Changes results for checkpoints that do not declare activation quantisation |

## Decision

**Option C.**

1. FP8 (E4M3, E5M2) and E2M1 elements widen to FP16, BF16 or F32 exactly.
2. NVFP4's E4M3 block scale may be folded into widened elements; its F32 global scale is applied once to the F32 result.
3. MX E8M0 scales, FP8 block scales (per 128-wide K-slice) and integer group scales multiply F32 partial sums per block, never narrower elements.
4. Decode kernels widen by default. MXFP4 kernels may instead use doubled E2M1 elements as INT8 with `Q8Block32` activation blocks, on the CPU and on GPUs: every OCP MXFP4 tensor has an exact transcription to ggml's `MXFP4` type, so this is ADR-0421 decision 4's baseline-equivalent path, not a separate policy (RFC-0042 §3.5).
5. Prefill defaults: W8A16 for FP8 (BF16 tensor cores, F32 accumulation); widen-to-BF16/FP16 tiles for MXFP4 and NVFP4. Native W8A8 FP8 (sm_89/120/121, RDNA4, CDNA3+) and block-scaled FP4 MMA (sm_120f, CDNA4) are used in exact mode only when the checkpoint declares the matching activation quantisation, or the family's harness run certifies the kernel (ADR-0421).
6. Devices with no kernel for a scheme get a widened copy in their pack: `widen.f32` (every scale folded, one rounding per element, equal to the reference's dequantised value), which ggml can run, or for native providers `widen.bf16` with unfoldable scales kept beside the elements.

## Consequences

- Positive: ADR-0014's fallbacks widen elements exactly and differ from the reference only by summation order and, for non-power-of-two scales applied after summation, F32-ulp rounding, so they pass parity without special tuning. They are not bit-identical to the reference (RFC-0042 §3.2).
- Positive: checkpoints keep their native bytes per token in decode on every backend.
- Negative: prefill on undeclared FP8 checkpoints forgoes FP8 tensor-core throughput until certified.
- Negative: per-block partial-sum scaling adds a multiply per block and complicates tile shapes on tensor-core paths.

## What would make us revisit

- A backend's widening path is measurably slower than native-format GEMV in decode (beyond the RFC-0040 §9 targets). Then add a native decode path for that backend.
- A frontier checkpoint ships a scale format whose product with elements is not exact under F32 partial sums (for example, scales that themselves need more than F32 range).
- Certification shows W8A8 or FP4 activation paths passing for most families. Then make them the default per family, with the harness kept as the check.

## Log

- Revised 2026-10-19: decisions 4 and 6 and the first consequence, because decision 4 conflicted with ADR-0421 decision 4, a widened copy with separate scales cannot run on ggml, and exactness was overstated for F32 scales.
