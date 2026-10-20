# ADR-0019: Exactness contract: teacher-forced agreement with a same-quantization reference
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D3), ADR-0002, ADR-0003, ADR-0006, ADR-0010, ADR-0014, ADR-0015, ADR-0017, ADR-0422, ADR-0601, ADR-0602; RFC-0012, RFC-0032, RFC-0040, RFC-0042, RFC-0043, RFC-0050, RFC-0051, RFC-0060, RFC-0061

## Context
D3 requires default output to match a reference within a stated numerical tolerance for the chosen quantization. Shortcuts that change output must be opt-in and labelled.

Syntropy's design creates more sources of numerical divergence than a single-device engine:
- **Execute-in-place** splits one layer between GPU and CPU kernels (ADR-0006). The two round differently.
- **The adaptive expert cache** moves an expert between devices across tokens and requests, so the same expert is computed by different kernels at different times.
- **Heterogeneous nodes** run consecutive stages on different backends: CUDA, Metal, CPU, Vulkan.
- **Shape-dependent kernels.** Speculative verification runs batches of T tokens rather than 1 (ADR-0017). Split-K and atomics change reduction order with shape.
- **Precision changes** on the wire (ADR-0010) and in fallbacks (ADR-0014).

Evidence from the field (`plan/research/tiered-memory-inference.md` §17):
- **A surveyed engine headlines exact output, but its defaults are not exact.**
  - Its default skips missed experts ranked 7th–10th without renormalising.
  - Its CPU share in short prefill chunks gives a mean KL of 0.004 nats (max 0.025) for 20–34% less prompt time.
  - Output varies run to run, because GPU and CPU round the same expert differently while the adaptive tier moves experts. Byte-identical repeats needed the prompt cache, adaptive moves and the PCIe share all switched off, at a cost in decode speed.
- **Other approximations in the field:**
  - low-precision copies of missed experts
  - expert deferral: up to 1.45× throughput for ≤0.5% average accuracy drop [reported]
  - routing 6 experts instead of 8
  - pruning half the experts
  - 2-bit quants that break JSON and tool calls
  - 4-bit KV: +8–12% perplexity on long documents
- **The research conclusion.** Mixed-device engines drift to "exact by flag, approximate by default". Teacher-forced KL and top-k agreement against a reference is the right test template. Syntropy needs that harness from the first milestone (§17, §20 item 11).
- **Wire precision results disagree.** One runtime saw f16 and q8 activation failures in production, while another succeeded with Q8_0. Precision must be measured per model family (`multi-device-inference.md` §6).

## Options considered
1. **Byte-identical output everywhere.** This needs identical algorithms and reduction orders on every backend, which rules out tensor-core accumulation, split-K and CPU/GPU splits. It is unattainable across heterogeneous devices at useful speed. Rejected.
2. **Benchmark-score parity** (Δperplexity or task scores within a margin). Coarse and slow, and it hides tail divergence at individual positions. Reported, but not a gate.
3. **Best effort, with no contract.** This is exactly the drift observed in the field. Rejected by D3.
4. **Chosen: teacher-forced top-1 agreement and KL divergence against a reference forward pass at the same quantization, within published per-dtype tolerances, measured by a harness built in M0. Byte-identical output is explicitly not promised.**

## Decision
1. **The reference.**
   - For each (model revision, pack quantization), the reference is the `syntropy-testkit` reference executor: one device, F32 accumulation, a fixed reduction order, and weights dequantised exactly from the same pack.
   - It is cross-validated against the ggml CPU provider on the conformance set (ADR-0002).
   - "Same quantization" means a 4-bit pack is judged against the 4-bit reference, not against BF16. Quantization error is a property of the chosen model file and is reported separately (RFC-0042).
   - *Revised 2026-10-20:* "one device" may be a GPU. The calibration reference is a fixed-order F32 reference on a GPU, cross-checked against the CPU `F32Fixed` reference on the nano and micro fixtures. On a CPU only dense models of about 1B fit the calibration run time once attention at 32K–128K is counted, so the MoE ceilings would rest on dense evidence; on a GPU a v1-family MoE model can be calibrated directly. The calibration run-time estimate (RFC-0042 §6.2, ADR-0422) is recomputed with attention included. Until the GPU method lands, the MoE ceilings are marked `provisional` in the M0 and M1 exits, and R-LABCPU's sizing (ADR-0610) is re-checked against it.
1a. **The model-definition oracle** (added 2026-10-20). The reference executor runs Syntropy's own family module, so on its own it shows only that kernels, placements and the wire agree with Syntropy's reading of the model. The oracle checks that reading against the model as published.
   - **The definition of "the model"** is the publisher's reference implementation, run in F32 on the same dequantised weights as the pack under test.
   - **Golden data.** For each family's real checkpoint, the publisher's implementation produces golden logits and hidden-state norms offline: full logits for about 64 short prompts and a few long ones, and hidden-state norms at 3–4 layers. They are stored per family.
   - **The gate.** `FamilyCert` is granted only when the reference executor matches the golden data within the definitional tolerance. This holds from M0, for `llama` and `qwen_moe`, and the check re-runs on every change to a family module or a shared block.
   - **The definitional tolerance** uses decision 2's metrics, teacher-forced top-1 agreement and KL (mean and p99), and is published beside the class ceilings (decision 3).
   - **Planted defects inside family modules**, such as a wrong YaRN factor or a router that is not renormalised, are caught by this oracle, and RFC-0060's planted-defect table names it as the catching layer.

   "Exact" therefore means two links: the candidate is within decision 3's tolerances of the reference executor, and the reference executor is within the definitional tolerance of the publisher's reference.
2. **The metrics.** Measured teacher-forced over a versioned evaluation corpus covering chat, code, long context (≥32K tokens), tool-call JSON and multilingual text:
   - **top-1 agreement:** the fraction of positions where the candidate's argmax equals the reference's
   - **KL(P_ref ‖ P_cand):** mean and p99 over positions, in nats

   Free-running greedy decodes also report the first divergence position. That figure is informative only, because near-ties legitimately cascade.
3. **The tolerances.**
   - For each dtype class (BF16/F16, FP8, 8-bit integer, 4-bit integer and FP4 families, below 4-bit), a candidate passes when both of its metrics are within 2× the spread measured between the reference executor and the ggml CPU provider on the same pack.
   - It must also stay within that class's absolute ceilings. RFC-0042 publishes them after M0 calibration.
   - Both are versioned, published in the docs, and stamped into every release.
   - A release with no published ceiling for a dtype class it supports is blocked.
   - *Note, 2026-10-20 (RFC-0060 §5.4).* "Within 2× the spread" means "not shown to exceed 2× the spread". The metrics come from a finite corpus, so a candidate fails a metric only when RFC-0060 §5.4's one-sided test rejects it against its limit L = min(τ, C) at the report's α (0.01 per report, spent hierarchically over the pooled and per-category tests). The corpus is sized to that RFC's power target: at the `standard` and `calibration` tiers, a candidate whose true metric is 2 L fails with probability ≥ 95%, pooled and in every category; `smoke` is a screen without one. The 2× is the pooled multiple; ADR-0601 and ADR-0422 set the per-category one at 3× the category's spread, read the same way, and the absolute ceilings are read the same way too.
4. **What must pass.** Every execution path the solver or scheduler can choose in exact mode:
   - every kernel and provider per op, and every fallback (ADR-0014)
   - every placement shape: single device, execute-in-place splits, cross-node pipelines, cross-node TP/EP
   - every transport precision
   - speculative versus non-speculative decoding (ADR-0017)
   - prefix-cache reuse, KV paging between tiers, and session save and resume

   Classification is by measurement, not intent. A path that fails the gate is approximate, whatever it was meant to be.
5. **Wire and storage rules in exact mode.**
   - Activations travel in F32. BF16 is allowed for a model family only after the harness passes it for that family, for decode and prefill separately.
   - KV pages move between tiers bit-for-bit in the model's KV dtype.
   - No exact-mode path lowers stored KV precision.
6. **Approximate knobs.** Each is opt-in per model configuration or per request, and named:
   - `expert_skip` (missed experts not computed)
   - `expert_defer`
   - `miss_low_precision` (lower-precision copies of missed experts)
   - `topk_reduce` (fewer routed experts than the model specifies)
   - `kv_lossy` (KV stored below the model's KV dtype)
   - `wire_compress` (activations below the family's passed precision)
   - `approx_attention` (sparsity not in the model's definition)
   - `spec_lenient` (non-exact speculative acceptance)

   A knob is added only with a harness report of its measured KL and top-1 cost.
7. **Labelling.**
   - Requests may set `fidelity: "exact" | "approximate"`. The default is `exact`. An `exact` request is never silently served by an approximate configuration; it fails with a clear error naming the active knobs.
   - *Revised 2026-10-19 (per-model default; see Revisions).* An instance policy may make approximate the default for requests that do not state a fidelity (`InstancePolicy::default_fidelity`, RFC-0054 §4.5). That is the operator's per-model opt-in under D3. An explicit `exact` is always honoured or fails, and the cluster's `allow_approximate` still gates every approximate response. Without such a policy the default stays `exact`.
   - Every response carries `fidelity` and the list of active knobs.
   - *Revised 2026-10-20:* the exact label applies only up to the longest context the parity harness has certified for the family, recorded in its `FamilyCert`. Positions beyond it are labelled **beyond the certified context**, wherever the fidelity label is shown: the response carries `FidelityReport::beyond_certified_from`, the position where that starts, and reads "exact up to token N; beyond the certified context after it" (RFC-0050 §10). The label is not approximate, because no knob is active and an `exact` request is still served; it states that no measurement covers those positions. It is distinct from `Certification::Uncertified`, which means no valid conformance manifest vouches for the serving plan.
   - For a family whose users run past 128K tokens, the harness also tests up to the family's advertised maximum context on sampled layers, and the certified context extends to what that run covers. Today the corpus's `long` category stops at 128K (RFC-0060 §5.2), while some v1 families advertise up to 1M (RFC-0043 §5.5).
   - The CLI prints them on start, and the dashboard badges each model and request (RFC-0050, RFC-0051).
8. **Byte-identical output is not promised** across devices, placements, runs or speculative settings, and the docs say so. A separate `deterministic` option (M3) freezes residency for the request, selects deterministic kernels and fixes CPU/GPU splits. It promises identical tokens for a fixed seed only on the same build, plan epoch and devices, and it is slower.
   - *Note, 2026-10-19 (sampler keying, ADR-0107).* When a request supplies its own seed, the sampler's random stream is keyed by (seed, choice index), so a repeated request draws the same random numbers across runs, and on the same build, plan epoch and devices reproduces its tokens. Without a client seed the key is (a seed ingress generates, `SequenceId`). Either way the draw counter is the per-sequence monotonic `rng_counter`, which survives migration and recovery. Identical draws do not make outputs byte-identical across devices; the paragraph above still holds.
9. **The parity harness is built in M0, before any optimisation,** together with the model-definition oracle for M0's families (decision 1a).
   - Op-level checks of every native kernel against the ggml reference.
   - Layer-level relative error of hidden states.
   - End-to-end teacher-forced metrics.
   - CI runs it on Tier 1 hardware for every change to kernels, the executor, the sampler or the wire, and nightly on the full placement matrix using the simulated fabric (RFC-0060).
   - Each `Warming` runner (ADR-0015) runs a short smoke test against stored reference logits.

## Consequences
- **Positive.**
  - "Exact" has a measurable meaning that users and CI can check.
  - Every optimisation has a numerical price tag before it ships.
  - Approximate speedups remain available without contaminating defaults.
- **Negative.**
  - The harness and corpus are M0 work that delays the first fast path.
  - Reference runs over long contexts on large models are slow. Nightly jobs need hardware budget, and the largest models are checked on sampled layers plus a reduced corpus.
  - The model-definition oracle needs each publisher's reference implementation run in F32, offline, for every family's checkpoint, and again when a module or shared block changes. That costs GPU time and storage for the golden data.
  - Long-context users see "beyond the certified context" past the certified context until sampled-layer testing covers their family.
  - Users will sometimes see different tokens across devices from the same prompt and seed, and the docs must explain why that is within contract.
- **Work created.**
  - RFC-0042: dtype classes, calibration and the ceiling tables.
  - RFC-0060: the reference executor, corpus, harness and CI gates, the golden-data gate on `FamilyCert`, and planted defects inside family modules.
  - RFC-0042: correct §7, which cites a publisher-output check that RFC-0043 §9 does not contain, and recompute the calibration budget with attention included.
  - RFC-0050 and RFC-0051: the "beyond the certified context" label (`beyond_certified_from`).
  - RFC-0040: op-level parity per provider.
  - RFC-0032: the wire precision rule.
  - RFC-0012: spec-versus-non-spec checks and the deterministic option.
  - RFC-0050 and RFC-0051: the fidelity fields and labels.
  - RFC-0061: performance reports always paired with fidelity.

## What would make us revisit
- The reference-versus-ggml spread is so small for some dtype class that 2× it rejects correct tensor-core kernels. Then switch that class to absolute ceilings only.
- Users need byte-identical cross-device output for a concrete use case (audit, caching keyed on output) at a cost they accept. Scope a fixed-order execution mode for it.
- A model family's reference behaviour is itself defined in a precision Syntropy does not run, such as an FP8 KV cache. Its reference is then that precision, and the knob list changes.
- A v1 family's publisher implementation cannot be run in F32 on dequantised weights. Then use, for that family only, the publisher's implementation in its native dtype with a looser tolerance, or a second independent open implementation.

## Revisions

- **Revised 2026-10-19:** decision 7 gains the per-model approximate default (RFC-0054 §4.5). D3 permits approximate behaviour "per model or per request"; before this revision only the per-request form was usable, so an operator who loaded a model usable only approximately could not serve stock clients that send no fidelity field. The labels are unchanged: a response served approximately says so and names its knobs, whichever side opted in. RFC-0001 §11.3 states the resolution rule, and RFC-0050 G4 counts the per-model default as the opt-in.
- **Revised 2026-10-19:** decision 8 gains a note on sampler keying (ADR-0107), so that a seeded request reproduces its random draws across runs. No decision changes.
- **Revised 2026-10-20:** decision 3 gains a note confirming RFC-0060 §5.4's reading of "within 2×": not shown to exceed the limit under that section's one-sided test at the report's α, with its power target of ≥ 95% rejection at twice the limit. The tolerances themselves do not change.
- **Revised 2026-10-20:**
  - Decision 1a (new), the model-definition oracle: the reference executor runs Syntropy's own family module, so a RoPE, norm or routing bug in that module would sit in the reference, the baseline and the candidate alike and pass every parity gate. The publisher's reference implementation, in F32 on the same dequantised weights, now defines the model, and golden logits and hidden-state norms gate `FamilyCert` from M0.
  - Decision 1: the single reference device may be a GPU, with a CPU cross-check on the small fixtures, because a CPU reference cannot calibrate MoE models in time; the MoE ceilings stay `provisional` until the method lands.
  - Decision 7: the exact label stops at the longest certified context, because parity is measured only up to 128K while some families advertise up to 1M; positions past it are labelled beyond the certified context, and families used past 128K are also tested to their advertised maximum on sampled layers.
