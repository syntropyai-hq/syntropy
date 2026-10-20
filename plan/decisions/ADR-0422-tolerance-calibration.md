# ADR-0422: Tolerance calibration: 2× the reference-versus-ggml spread, per-class ceilings, a BF16 perturbation floor measured always and applied end to end only on evidence

- Status: Accepted (2026-10-20); revised 2026-10-20 (decisions 1 and 7)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0002, ADR-0019, ADR-0421, ADR-0601; RFC-0042 §5, §6; RFC-0060, RFC-0062

## Context

- ADR-0019 decision 3: per dtype class, a candidate passes when its metrics are within 2× the spread between the reference executor and the ggml CPU provider on the same pack, and within the class's absolute ceilings, which RFC-0042 publishes after M0 calibration; a release without ceilings for a supported class is blocked. Its revisit clause: if a class's spread is so small that 2× of it rejects correct tensor-core kernels, move that class to absolute ceilings.
- ADR-0601 (quality area) fixes the end-to-end gate mechanics: stored top-256 reference blocks, the coarse-grained `KL_K` metric, pooled metrics within `max(2 × baseline, φ)`, per-category metrics within 3 × the category baseline, both within the RFC-0042 ceiling C, with φ = 0 unless RFC-0042 invokes ADR-0019's revisit clause with evidence.
- Op-level parity (RFC-0040 §6.2) compares single ops. There, the ggml op and the reference can compute identically, so their spread can be exactly zero, and 2 × 0 would reject every other correct kernel.
- Binomial noise on top-1 agreement at N positions is `√(p(1 − p)/N)`: about 0.001 at p = 0.95 and N = 50 000.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Fixed absolute thresholds chosen up front | Simple | No measured basis |
| B. ADR-0019 literally, at both levels | Faithful | Breaks at op level, where spreads can be zero |
| C. A BF16 perturbation floor applied at every level by default | Never tighter than an allowed operation | Loosens ADR-0019's end-to-end rule without the evidence its revisit clause asks for; conflicts with ADR-0601 |
| **D. End to end: ADR-0019's rule (2× pooled, 3× per category per ADR-0601, φ = 0), with per-class ceilings from calibration; the BF16 floor measured and published at every calibration, applied end to end for a class only on recorded evidence. Op level: the floor always applies** | Faithful to ADR-0019 and ADR-0601 where they speak; workable where they do not | Two floor policies to explain |

## Decision

**Option D.**

1. Calibration runs per dtype class on at least three families where checkpoints exist, with at least 50 000 scored positions per corpus category. Calibration models must fit the budget: the reference and baseline passes each within RFC-0060's 48 h rule at their measured rates, with the cost counted as RFC-0060 counts it, attention over the `long` documents included, and a `widen.f32` baseline within its host's memory. Larger models, 1T-class included, are judged against the class ceilings by RFC-0060 §5.7. At M0, when only `llama` and `qwen_moe` exist, it uses at least three models from those families, and the ceilings are published as provisional; M1 recalibrates over at least three families, and a rise then is logged here.
   - *Revised 2026-10-20:* the reference pass of calibration runs `F32Fixed` on one GPU: the CPU reference's operation order, contraction off, F32 subnormals kept and `kernels::reference`'s own transcendental functions. It must equal the CPU `F32Fixed` reference bit for bit on the `nano` and `micro` fixtures before any calibration uses it. A ceiling that gates a MoE model stays `provisional`, in the tables and in the M0 and M1 exit reports, until a MoE checkpoint of a v1 family has been calibrated directly with that reference (RFC-0042 §6.2). Counted with attention, the CPU reference could calibrate only dense models of about 1B, so the MoE ceilings would have rested on dense evidence.
2. Spread `S` = metrics(reference executor, baseline), where the baseline is the ggml CPU provider on the same graph and pack for GGUF types and plain floats, the ggml CPU provider over an exact transcription for transcribable safetensors tensors (RFC-0042 §3.5), and the ggml CPU provider over a `widen.f32` copy for FP8, non-transcribable `PackedInt` and NVFP4. Every baseline applies the checkpoint's declared activation quantisation exactly, through a CPU fake-quant step before each affected `mul_mat`, as the reference does. Floor `F` = metrics(reference, reference with BF16-rounded matrix-product inputs). Both are published.
2a. For the `widen.f32` baselines, `S` measures summation order only, so those schemes use `φ = F` from their first calibration (the ADR-0019 revisit case, known in advance). `Fp8` is calibrated this way at M1.
3. End-to-end tolerance per calibration model: pooled `max(2·S, φ)`, per category `max(3·S_c, φ)`, with `φ = 0`. A class gets `φ = F` only when a kernel shown correct by op-level parity and the independent review agent fails the gate while `2·S < F`, and a maintainer approves it (**Maintainer action**; ADR-0021 decision 5); the event and its evidence are logged here and in the release notes.
4. Ceiling per class = the maximum tolerance over its calibration models, rounded up to two significant digits. Every candidate must also be within the ceiling. Layer-local ceilings `C_ll` per (layer kind, class, statistic) are built the same way from RFC-0060 §5.7's `τ_ll` and cap it (RFC-0042 §5.2).
5. Op-level tolerance per (op, scheme, shape class): `max(2 · p99 rel_l2(ggml op, reference op), p99 rel_l2(BF16-input reference, reference))`. Integer-valued ops are bit-exact.
6. Tables are versioned (`numerics/tolerances-v<N>.toml`), published, stamped into releases, and recomputed on ggml pin bumps, reference changes and corpus changes. A ceiling rise above 10 % needs a release note and an entry in this ADR's log. A spread more than 2× its previous value stops calibration until a maintainer approves the investigation of which side changed (**Maintainer action**; ADR-0021 decision 5).
7. *Revised 2026-10-20:* the model-definition oracle's tolerance is calibrated per family from the same metrics: `τ_def(f) = 2 · max(S_pub, S_ref)`, where `S_pub` is the publisher's reference code in F32 against the same code in F64 and `S_ref` is `RefExecutor` in `F32Fixed` against F64, both on the oracle's golden prompts. It is published in the same tables, beside the ceilings (RFC-0042 §7.1).

## Consequences

- Positive: the end-to-end gate is exactly ADR-0019's, so "exact" means what the foundation says.
- Positive: op-level parity works for ops whose spread is zero.
- Negative: until evidence arrives, a class with a very small spread may reject a correct tensor-core kernel at first; the revisit procedure is the remedy, and it costs a release cycle.
- Negative: calibration costs GPU time for the reference and CPU time for the ggml baseline, per model per pin bump (RFC-0042 performance section), and the GPU reference is a second implementation of `F32Fixed` that must stay bit-identical to the CPU one.

## What would make us revisit

- The end-to-end floor has to be switched on for more than one class in the first two calibrations. Then the floor should be the default, with an amendment to ADR-0019.
- Ceilings drift upward over three consecutive recalibrations: pin a fixed baseline build for calibration.
- ADR-0601's per-category factor is recalibrated at M0 to a value that changes the ceilings materially. Update both ADRs together.

## Log

- Revised 2026-10-19: decision 1 gains provisional M0 ceilings; decision 2 defines the baseline per scheme and decision 2a sets `φ = F` from the start for schemes ggml cannot run natively. Recorded here as the first ADR-0019 revisit event, for the `widen.f32` baselines: FP8 of every scale kind, non-transcribable `PackedInt`, NVFP4.
- Revised 2026-10-19: decision 2 makes every baseline apply declared activation quantisation, so decision 2a's "summation order only" holds for declared-W8A8 checkpoints; decision 1 bounds calibration models by the 48 h rule and rig memory.
- Revised 2026-10-19: decision 4 adds layer-local ceilings `C_ll`; the `widen.f32` floor is the one exception ADR-0601 decision 4 names (RFC-0042 §6.3).
- Revised 2026-10-20: decision 1's budget counts attention over the `long` documents, which makes calibration about 10× costlier than the old ≈ 17 B-active bound assumed, so the CPU reference could calibrate only dense models of about 1B [estimate: RFC-0042 §6.2's worked figures]; the calibration reference moves to a fixed-order GPU run cross-checked bit for bit against the CPU reference, and MoE ceilings stay provisional until a MoE model is calibrated directly. Decision 7 added: the definitional tolerance of the model-definition oracle, published beside the ceilings.
