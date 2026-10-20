# ADR-0421: F32 accumulation always; activation quantisation inside matrix products only when declared, ggml-equivalent, or certified

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0002, ADR-0014, ADR-0019; RFC-0040 §6.2, RFC-0042 §4

## Context

- ADR-0019: exact mode means teacher-forced agreement with a same-quantization reference executor (F32 accumulation, fixed order, weights dequantised exactly), within 2× the spread between that reference and the ggml CPU provider, and within per-class ceilings. Classification is by measurement.
- The ggml CPU path, the calibration baseline, quantises activations to 8-bit blocks (its `Q8_0`/`Q8_1`/`Q8_K` layouts) for products with integer-coded weights. GPU integer-dot kernels do the same.
- Tensor cores on consumer GPUs run faster with FP16 accumulation (the RTX 4090's FP16 tensor rate doubles with FP16 accumulation, per the Ada whitepaper, cost model §2.1). FP8 and FP4 tensor-core instructions need FP8 or FP4 activations.
- Some checkpoints declare activation quantisation as part of the model (DeepSeek-V3's FP8 config: `activation_scheme: dynamic`; compressed-tensors `input_activations`), and some were trained with it (Kimi K3 with MXFP8 activations).
- Selection must be able to reject a kernel by contract before benchmarking (RFC-0040 §6.2).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Leave precision to each kernel; rely on the end-to-end harness | Least specification | Failures found late and attributed vaguely; no way to reject a kernel at `query` |
| B. F32 activations and F32 accumulation in every product | Closest to the reference | Rules out integer-dot kernels that the baseline itself uses; slow CPU paths |
| **C. F32 accumulation always (I32 exact within a scale block for integer dots); activation quantisation allowed when (a) the checkpoint declares it, in which case the reference does the same, (b) it is 8-bit blocks for GGUF integer-coded weights or BF16 rounding of matrix inputs, or (c) the family's harness certifies the kernel; anything else is approximate (`act_lowp`)** | Matches the reference's definition and the baseline's practice; enforceable through `Numerics` at selection time | Some fast tensor-core paths wait for certification |

## Decision

**Option C.**

1. Accumulators are F32; FP16 or BF16 accumulators are forbidden in exact mode. Integer dot products accumulate in I32 within one scale block, then scale in F32.
2. Softmax statistics, scan states and norms are computed in F32. The residual stream is F32 within a stage. No fast-math.
3. Declared activation quantisation is applied identically by the reference executor and by kernels.
4. `Q8Block32` and `Q8K256` activation blocks are allowed in exact mode for GGUF integer-coded weights and for safetensors tensors with an exact transcription to a ggml type (OCP MXFP4 always; symmetric compressed-tensors INT4 and INT8 whose group size is a multiple of 32 and whose scales convert to F16 exactly; RFC-0042 §3.5), using the block the ggml type uses. The justification is the same in both cases: the calibration baseline runs these tensors through ggml with those blocks. BF16 rounding of matrix-product inputs is allowed. Other integer and FP4 schemes widen with F32 activations by default. The M1 calibration set includes a MoE checkpoint in the CPU-expert scheme itself (compressed-tensors symmetric INT4, group 32), small enough for RFC-0042 §6.2's calibration budget, so the flagship in-place path is gated directly. The 1T-class model is judged against the class ceiling by RFC-0060 §5.7.
5. Other activation lowering (undeclared FP8, MXFP8, FP4) is allowed in exact mode only for (family, dtype class, shape class) entries certified in the release's `ConformanceManifest` (ADR-0602); otherwise only under the `act_lowp` approximate knob, which is added in M3 with its harness report (ADR-0019 decision 6).
6. Each kernel declares `Numerics { accumulate, act_in, out, softmax_f32, state_f32 }`; selection rejects any kernel whose declaration breaks these rules.

## Consequences

- Positive: precision violations are caught by declaration at selection, before any benchmark or harness run.
- Positive: integer-dot kernels on every backend remain exact-mode candidates.
- Negative: consumer GPUs run FP16-input tensor-core work at the FP32-accumulation rate in exact mode.
- Negative: certifications are per family and must be redone when kernels change.

## What would make us revisit

- Calibration shows the 8-bit activation path outside a class ceiling for some family. Then that family needs F32-activation integer kernels (RFC-0040 open question 6).
- A vendor documents FP16-accumulation error bounds tight enough to stay within the perturbation floor at the shapes Syntropy uses. Then allow it per (device class, shape) after certification.

## Log

- Revised 2026-10-19: decision 4 extended to transcribable safetensors tensors, with F32 activations the default for everything else, because the RFC-0042 matrix used 8-bit activations for compressed-tensors and MXFP4 paths that the old rule did not cover.
- Revised 2026-10-19: decision 4's calibration model narrowed to a checkpoint within the calibration budget.
