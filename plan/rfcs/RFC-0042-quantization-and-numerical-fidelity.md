# RFC-0042: Quantization and numerical fidelity

- Status: Draft
- Area: compute and models
- Depends on: RFC-0001 (§2.3 invariant 8, §5.1, §5.4, §5.8, §10.2), RFC-0012, RFC-0022, RFC-0031, RFC-0032, RFC-0040, RFC-0041, RFC-0043, RFC-0050, RFC-0051, RFC-0054, RFC-0060, RFC-0061; ADR-0000 (D3), ADR-0002, ADR-0005, ADR-0010, ADR-0014, ADR-0017, ADR-0019, ADR-0601; ADR-0420 to ADR-0422
- Date: 2026-10-19
- Revised: 2026-10-20: the model-definition oracle (§7.1); the calibration budget with attention counted and the GPU calibration reference (§6.2); certified context length (§5.7); per-model knobs only (§10); cross-device identity not promised (Non-goals); open question 7 answered

## Summary

This RFC defines what Syntropy's numbers mean. It lists every supported data type and quantization scheme with its exact dequantisation, assigns each to a dtype class, and fixes the kernel path and fallback per scheme and backend. It sets the accumulation rules every kernel must follow, and the rule for when a kernel may quantise activations inside a matrix product. It implements ADR-0019's exactness contract: the reference executor, the teacher-forced metrics, how per-dtype tolerances are calibrated, the activation precision on the wire, and the approximate-mode knobs with their labels. It also scopes the optional requantization tools.

Key decisions:

1. **Scales are applied to F32 partial sums**, never folded into a narrower element type, unless the product is provably exact there (NVFP4 elements times E4M3 block scales are; MX power-of-two scales on FP16 are not). Under that rule FP8, MXFP4, NVFP4 and INT4 elements widen exactly and MX scaling is exact; other scales differ from the reference only by F32 rounding (one ulp per product or partial sum), which the op tolerance absorbs. Every fallback in ADR-0014 is therefore within tolerance by construction, not bit-identical (ADR-0420).
2. **F32 accumulation everywhere; integer dot products accumulate exactly in I32 within a scale block.** FP16 accumulators are never used in exact mode (ADR-0421).
3. **Activation quantisation inside a matrix product is part of the reference only when the checkpoint declares it** (for example `activation_scheme: dynamic` FP8). 8-bit activation blocks are allowed for GGUF integer-coded weights, and for safetensors tensors with an exact transcription to a ggml type (§3.5), because the cross-validation baseline computes them that way. Every other integer or FP4 scheme widens with F32 activations by default; its 8-bit-activation kernels, like any other lowering, need a per-family harness pass, and otherwise are approximate (ADR-0421).
4. **Tolerances are 2× the reference-versus-ggml spread (3× per corpus category), capped by per-class ceilings from M0 calibration.** A BF16 perturbation floor is measured at every calibration. End to end it applies only under ADR-0019's revisit clause, with evidence; at op level it always applies, because an op's spread can be exactly zero (ADR-0422, consistent with ADR-0601).
5. **The model definition is checked against the publisher, not against Syntropy's own graph.** The reference executor runs the same family module as the product, so it cannot catch a module bug. A model-definition oracle closes that gap: golden logits and hidden-state norms produced offline by the publisher's reference code in F32 on the same dequantised weights, which gate every `FamilyCert` from M0 (§7.1).
6. **Calibration runs its reference on a GPU**, in the same fixed order as the CPU reference and cross-checked against it bit for bit, because the CPU reference cannot calibrate a MoE model of a v1 family within the budget once attention is counted (§6.2).

## Motivation and goals / non-goals

ADR-0000 D3 makes exact output the default. ADR-0019 turns that into a measurable contract and leaves this RFC the dtype classes, calibration, ceilings, wire rule details and knob definitions. Syntropy has more sources of divergence than a single-device engine: CPU and GPU kernels for the same expert, kernels that change with batch shape, heterogeneous stages, precision changes at the wire and in fallbacks (ADR-0019 context). A surveyed engine that headlines exact output shipped defaults that skip missed experts and vary run to run (`tiered-memory-inference.md` §17). The defence is a precise definition of every number, and a harness that checks it.

2026 checkpoints make this concrete. FP8 block-scaled originals (DeepSeek-V3/V4, GLM-5.x), MXFP4 (gpt-oss, Kimi K3 with MXFP8 activations in training), NVFP4 variants, INT4 group-32 experts beside a BF16 hot set (Kimi-K2-Thinking), and every GGUF block type (`backends-and-portability.md` §3.2, §4.1). Each needs a kernel on every backend, or an exact fallback.

### Goals

1. One normative dequantisation per scheme, bit-exact across implementations.
2. Every fallback in ADR-0014's matrix exact by construction, or demonstrably within tolerance.
3. Accumulation rules precise enough to reject a kernel at `query` time (RFC-0040 §6.2), before any measurement.
4. A calibration procedure that produces the per-class ceilings ADR-0019 requires before a release can ship.
5. Every approximate knob defined, scoped and labelled.

### Non-goals

- Byte-identical output across devices, placements or runs (ADR-0019 decision 8); the M3 `deterministic` option is the narrow exception, and it holds only on the same build, plan epoch and devices (§8.3). Cross-device identity stays unpromised until a concrete use asks for it, because it would rule out split-K and atomic reductions for every user.
- Model quality relative to the unquantised original. That is a property of the chosen file, reported separately (§5.6), not part of the exact-mode contract.
- The test infrastructure (corpus storage, CI jobs, simulated fabric): RFC-0060. This RFC defines what is measured and the thresholds.

## Design

### 1. Data types, schemes and dtype classes

#### 1.1 Types

RFC-0001 §5.1 defines `DType { F32, F16, BF16, F8E4M3, F8E5M2, I8, U8, I32 }` and `QuantScheme`. This RFC owns both and adds:

```rust
pub enum DType {
    F32, F16, BF16, F8E4M3, F8E5M2, I8, U8, I32,
    F8E8M0,          // +0042: MX power-of-two scale, 2^(e − 127); e = 255 is NaN
    F4E2M1,          // +0042: FP4 element (sign, 2-bit exponent with bias 1, 1-bit mantissa)
    F6E2M3, F6E3M2,  // +0042: read-only (safetensors F6); widened in packs (§3.3)
    I16, I64, F64,   // +0042: index tensors and the reference's diagnostic mode
}

pub enum QuantScheme {
    Plain(DType),
    Ggml(GgmlType),                          // every GGUF block type, §2.1
    Mxfp4 { block: u16 },                    // OCP MX: 32 × E2M1 + one E8M0 scale
    Nvfp4 { block: u16, scale: DType },      // 16 × E2M1 + one E4M3 scale; per-tensor F32 global scale as aux
    Fp8 { format: DType, scale: ScaleGranularity },
    CompressedTensors(CtScheme),
}
pub enum ScaleGranularity { Tensor, Channel, Block { rows: u16, cols: u16 }, Group { k: u16 } }
pub enum CtScheme {
    PackedInt { bits: u8, group: u16, symmetric: bool, zero_point: bool },
    Fp8 { strategy: ScaleGranularity },
}

/// +0042. Activation quantisation a matrix product performs on its input (§4.3).
pub enum ActQuant { None, Bf16, Q8Block32, Q8K256, Fp8Dynamic { group: u16 }, Mxfp8, Nvfp4 }
```

#### 1.2 Dtype classes

ADR-0019 sets tolerances per dtype class. Every scheme maps to exactly one class:

| Class | Schemes |
|---|---|
| `Float16` | `Plain(F32 \| F16 \| BF16)` |
| `Fp8` | `Fp8 { .. }`, `CompressedTensors(Fp8)` |
| `Int8` | `Q8_0`, `INT8` packed |
| `Bits4` ("4-bit integer and FP4 families", extended to 4–6 bits) | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, `IQ4_XS`, `Mxfp4`, `Nvfp4`, `PackedInt { bits: 4 }` |
| `Sub4` | `Q2_K`, `Q3_K`, `IQ3_XXS`, `IQ3_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ1_S`, `IQ1_M`, `TQ1_0`, `TQ2_0`, `Q2_0`, `Q1_0` |

ADR-0019 names "4-bit integer and FP4 families" as one class. Five- and six-bit types join it here, because they round at the same order of error and nothing calibrated separates them yet (Open questions).

**A model's class** is the class of the lowest-precision scheme that carries at least 10 % of the model's batch-1 `read_per_token` bytes (RFC-0041 §3.1). A Kimi-K2-Thinking-class checkpoint (INT4 routed experts, BF16 hot set) is therefore `Bits4`; a GGUF whose bulk is Q4_K with a Q6_K output head is `Bits4`.

### 2. Schemes and their exact dequantisation

The **dequantised value** of a weight element is defined per scheme below, and `kernels::reference` (RFC-0040 §1) computes it in F32. For GGUF block types the definition is ggml's own F32 dequantisation, including its rounding at each step (§2.1). For the FP4, FP8, compressed-tensors and plain schemes the value is `fl32(element × scale)` (times the NVFP4 global scale, with a second rounding), computed in F32 with one rounding per factor. Elements widen exactly. The product is exact when the scale is a power of two (MX) or when its significand and the element's together fit in 24 bits (NVFP4's E4M3 block scale, F16 or BF16 group scales on INT4); with an F32 scale (FP8 `weight_scale_inv`, F32 group scales) it can need up to 28 or 32 bits and is rounded once. Native kernels must produce the same values, or apply the same factors to F32 partial sums (§3.2), which differs from the reference by F32 rounding only.

#### 2.1 GGUF block types

GGUF types are defined by ggml's own reference dequantisation for the pinned ggml version (the `dequantize_row_*` functions). `kernels::reference` reimplements each, and a test checks it bit-exact against ggml's function on random blocks. Block geometry comes from ggml's type traits at build time; this table is informative, and a build-time test checks it:

| Type | Weights per block | Bytes per block | Bits per weight | Class |
|---|---|---|---|---|
| `F32`, `F16`, `BF16` | 1 | 4, 2, 2 | 32, 16, 16 | `Float16` |
| `Q8_0` | 32 | 34 | 8.5 | `Int8` |
| `Q4_0` / `Q4_1` | 32 | 18 / 20 | 4.5 / 5.0 | `Bits4` |
| `Q5_0` / `Q5_1` | 32 | 22 / 24 | 5.5 / 6.0 | `Bits4` |
| `Q2_K` / `Q3_K` / `Q4_K` / `Q5_K` / `Q6_K` | 256 | 84 / 110 / 144 / 176 / 210 | 2.625 / 3.4375 / 4.5 / 5.5 / 6.5625 | `Sub4` / `Sub4` / `Bits4` / `Bits4` / `Bits4` |
| `IQ4_NL` / `IQ4_XS` | 32 / 256 | 18 / 136 | 4.5 / 4.25 | `Bits4` |
| `IQ3_XXS` / `IQ3_S` | 256 | 98 / 110 | 3.0625 / 3.4375 | `Sub4` |
| `IQ2_XXS` / `IQ2_XS` / `IQ2_S` | 256 | 66 / 74 / 82 | 2.0625 / 2.3125 / 2.5625 | `Sub4` |
| `IQ1_S` / `IQ1_M` | 256 | 50 / 56 | 1.5625 / 1.75 | `Sub4` |
| `TQ1_0` / `TQ2_0` | 256 | 54 / 66 | 1.6875 / 2.0625 | `Sub4` |
| `Q2_0` / `Q1_0` | 64 / 128 | 18 / 18 | 2.25 / 1.125 | `Sub4` |
| `MXFP4` | 32 | 17 | 4.25 | `Bits4` (§2.2) |
| `NVFP4` (ggml's layout: 64 weights in 16-weight sub-blocks with E4M3 scales) | 64 | 36 | 4.5 | `Bits4` (§2.2) |

`Q8_1` and `Q8_K` are activation-only types in ggml; Syntropy uses their layouts for activation blocks (§4.3) and never as weight schemes.

#### 2.2 FP4: MXFP4 and NVFP4

- **E2M1 element values:** ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. Every one is exact in FP16, BF16 and F32. Doubled, they are the integers ±{0, 1, 2, 3, 4, 6, 8, 12}, exact in INT8.
- **MXFP4** (OCP Microscaling): a block of 32 elements shares one E8M0 scale `X = 2^(e − 127)`, with `e = 255` meaning NaN. `value = X · element`. `X` spans 2^−127 to 2^127, beyond FP16's range and into BF16's subnormals, so **the scale is never folded into an FP16 or BF16 element**: kernels multiply the block's F32 partial sum by `X` (an exponent adjustment, exact barring overflow).
- **NVFP4:** a block of 16 elements shares one E4M3 scale `s` (non-negative), and the tensor has one F32 global scale `g`. `value = g · s · element`. The product `s · element` has at most 6 significant bits and lies in [2^−10, 2688], so it is **exact in FP16 and BF16**. Kernels may fold `s` into the widened element, and must apply `g` to the F32 result.

#### 2.3 FP8

- **E4M3** (the OCP "FN" variant: no infinities, maximum 448, NaN = all mantissa and exponent bits set) and **E5M2** (IEEE-style, with infinities). Both widen exactly to FP16, BF16 and F32.
- **Scales** (`Fp8 { scale }`): `Tensor` and `Channel` scales are F32; `Block { 128, 128 }` gives one F32 scale per 128 × 128 tile of the weight (the `weight_scale_inv` convention: `value = element × scale`). Block scales vary along K, so a kernel multiplies each 128-wide K-slice's F32 partial sum by its block's scale before adding it to the accumulator.

#### 2.4 Integer schemes from compressed-tensors

`PackedInt { bits: 4 | 8, group, symmetric, zero_point }`: `value = scale_g · (q − z_g)` with `z_g = 0` when symmetric, groups of `group` consecutive elements along K. Unpacking order and offsets follow the compressed-tensors `pack-quantized` specification. Golden tests on fixture tensors from real checkpoints pin the interpretation (RFC-0041 §2.2). `q − z_g` is an exact small integer. The scale (F16, BF16 or F32) multiplies the group's F32 partial sum; it is never folded into an FP16 element, because a BF16 scale times a 4-bit integer can need 12 significant bits, one more than FP16 holds.

#### 2.5 Plain floats

`Plain(F16 | BF16 | F32)` weights are used as they are. F16 and BF16 widen to F32 exactly.

### 3. Kernel paths and fallbacks per scheme

#### 3.1 The matrix

Decode (`Gemv`, `MoeGrouped` at rows ≤ 16) is memory-bound on every device (cost model §3.1). Its kernels widen weights in registers and multiply them by F32 or 8-bit activations (§4.3), so matrix hardware does not matter for decode. Prefill and large batches are compute-bound, which is where tensor-core formats pay. ADR-0420 records the policy.

| Scheme | CUDA | Metal | CPU | HIP | Vulkan |
|---|---|---|---|---|---|
| GGUF integer-coded (`Q*`, k-quants, i-quants, `TQ*`) | decode: `dp4a` with `Q8Block32`/`Q8K256` activations; prefill: INT8 IMMA tiles | decode: widen to F32, simdgroup dot; prefill: widen to F16 tiles, `simdgroup_matrix` or `matmul2d` | VNNI / `sdot` / I8MM with 8-bit activation blocks; AMX or SME2 for prefill | as CUDA (`dot4`, INT8 WMMA/MFMA) | integer-dot extension where present, else widen |
| `Plain(BF16 \| F16)` | F32 FMA (decode); BF16/F16 tensor cores with F32 accumulators (prefill) | as CUDA with simdgroup or tensor ops | BF16 dot (`vdpbf16ps`, NEON BF16) or widen to F32 | WMMA / MFMA | cooperative matrix |
| `Fp8` (any scale) | decode: widen; prefill: **W8A16** (widen to BF16 tiles, BF16 tensor cores) by default; **W8A8** FP8 tensor cores on sm_89/120/121 only under §4.3 rule (a) or (c) | widen (no FP8 matrix hardware in Apple's published tables) | widen to F32 or BF16 | RDNA4 / CDNA3+: as CUDA with FP8 WMMA/MFMA; others widen | widen |
| `Mxfp4` | decode: widen, or `E2M1 × 2` as INT8 with `Q8Block32` activations (rule b: every MXFP4 tensor is transcribable, §3.5); prefill: widen to BF16 tiles by default; block-scaled FP4 MMA on sm_120f only under §4.3 (a) or (c) | widen | INT8 dot on doubled elements with `Q8Block32` activations (rule b), scale per block | CDNA4 MX MFMA under the same rule; else widen | widen |
| `Nvfp4` | decode: widen with `s` folded (exact, §2.2); prefill: FP16 tiles by default; FP4×FP4 block-scaled MMA only under §4.3 (a) or (c) | widen | widen | widen | widen |
| `PackedInt { 4 \| 8 }` | decode: widen with F32 activations, group scale on partial sums (default); unpack to INT8 and `dp4a` with `Q8Block32` activations under rule (b) for transcribable tensors (§3.5), otherwise only under rule (c); prefill: widen to BF16 tiles, IMMA under the same rules | widen | widen to F32 (default); INT8 dot with `Q8Block32` activations under rule (b) for transcribable tensors, otherwise rule (c) | as CUDA | widen |
| `F6E2M3`/`F6E3M2` | packs hold an exact BF16 copy (§3.3) | same | same | same | same |

#### 3.2 What "widen" means

**Widening** converts weight elements to a wider float type in registers or tiles (FP16, BF16 or F32), exactly, and applies scales by these rules:

1. A scale whose product with the element is exact in the widened type may be folded in: NVFP4's E4M3 block scale into FP16 or BF16; any scale into F32.
2. Every other scale multiplies the F32 partial sum over its scale block: the MX E8M0 scale per 32 elements; FP8 block scales per 128-wide K-slice; integer group scales per group; the NVFP4 global scale once at the end.
3. Accumulation is F32 (§4).

Under these rules every ADR-0014 fallback ("in-register FP4→FP16 dequant", "lossless FP8→FP16 widening", "FP32 accumulate from widened BF16") widens elements exactly. Products and sums then differ from the reference in two ways only: summation order, and, for a non-power-of-two scale applied after summation, `s · Σ fl(e·a)` in place of the reference's `Σ fl(fl(s·e)·a)`, an F32-ulp-level difference. Both are inside the op tolerance (§6.4), which is why the fallbacks pass parity by construction and ADR-0014 rule 2 holds for them. They are not bit-identical to the reference, and no test may assume they are. Scale application fused into a product, as in these fallbacks, is in the toleranced class (§6.4). Standalone dequantisation is different and is bit-exact (§6.4, "Bit-exact scope"): it is a fixed sequence of IEEE operations, so it reproduces bit for bit under the conditions stated there.

#### 3.3 Widened copies in packs

When a device has no kernel for a scheme at all (F6 types everywhere; FP8 or FP4 on a ggml-only backend without them), the pack stores a widened copy, a non-`SOURCE` layout recorded in the sidecar (RFC-0041 §5.2):

- `widen.f32`: every element times all its scales, `fl32(e × s)`, one rounding per element, which is exactly the reference's dequantised value (§2). ggml runs it as an F32 weight, so this is the layout for ggml-only devices.
- `widen.bf16`: BF16 elements with folded scales where the product fits BF16 exactly; unfoldable scales (MX, FP8 block) kept beside the elements and applied per §3.2. No ggml op applies per-K-slice scales, so this layout is only for native providers. It costs 2–4× the source bytes on that device, which the solver sees through the census's per-scheme byte totals.

#### 3.4 Emulation

Where a backend lacks even the integer-dot or F16 paths (some ggml-only targets), ggml's own kernels apply, and the parity gate decides. Syntropy writes no scalar emulation kernels of its own beyond the CPU provider's portable paths at `X86V2` and `Arm80`.

#### 3.5 Exact transcriptions to ggml types

Some safetensors schemes hold values that a ggml block type represents exactly. Where that is true for a whole tensor, the tensor is **transcribable**: the reader can rewrite its bytes into the ggml type without changing any dequantised value, and the ggml CPU provider can run it as the calibration baseline (§6.2). The census checks each tensor and records the result.

| Source scheme | ggml type | Condition, checked per tensor |
|---|---|---|
| OCP MXFP4 (32 × E2M1, one E8M0 scale) | `MXFP4` | always: the element codes and the scale are the same; only the nibble order within a block differs |
| `PackedInt { bits: 4, symmetric }` | `Q4_0` | group size a multiple of 32, and every group scale converts to F16 exactly |
| `PackedInt { bits: 8, symmetric }` | `Q8_0` | as above |
| asymmetric `PackedInt`, NVFP4, FP8 | none | not transcribable in v1 (NVFP4 is revisited at M1 once ggml's `NVFP4` global-scale handling is checked) |

A transcription is verified bit for bit: `kernels::reference` dequantises the source tensor and the transcribed tensor, and the two must agree in every element (RFC-0060's bit-exact class). Transcription is used by the baseline and by ggml fallbacks; native kernels read the source layout or their own repack.

### 4. Accumulation and activation precision

ADR-0421 records these rules. They are checked at `query` time through `OpCaps::numerics` (RFC-0040 §6.2): a kernel that declares a weaker contract than its op requires is never a candidate.

```rust
pub struct Numerics {
    pub accumulate: AccumKind,       // F32 | I32ThenF32 { max_block: u32 } | F64
    pub act_in: ActQuant,            // what the op does to its activation input (§4.3)
    pub out: DType,                  // F32 | BF16 | F16 (F16 only where §4.2 allows)
    pub softmax_f32: bool,           // attention: max and sum in F32
    pub state_f32: bool,             // scans: recurrent state in F32
}
```

#### 4.1 General rules

1. **Dot products and matrix products accumulate in F32.** FP16 or BF16 accumulators are never used in exact mode, including tensor-core instructions with half-precision accumulation, even where they run faster.
2. **Integer dot products** (`dp4a`, VNNI, `sdot`, I8MM, IMMA, AMX-INT8) accumulate exactly in I32 within one scale block, then convert to F32 and multiply by the block's scales. An 8-bit product (signed by signed, or unsigned by signed as in VNNI) is below 2^15 in magnitude, so I32 stays exact for sums of fewer than 2^16 products; no scale block in §2 is longer than 256.
3. **Reduction order is free** within F32, except when the runner is opened deterministic (RFC-0040 §3.3). Order differences are what the tolerances absorb.
4. **The residual stream is F32** inside a stage. Hops follow §8.
5. **No fast-math.** Kernels are compiled without flags that permit reassociation of user-visible results across ops, flush-to-zero on F32, or approximate transcendental functions beyond the backend's documented accuracy. `exp` in softmax and SiLU uses the backend's standard single-precision function.
6. **NaN and infinity propagate.** No kernel clamps them away. A NaN in logits fails the sequence with a `Correctness` error rather than sampling garbage.
7. **Bit-exact-class kernels: no contraction, subnormals kept, power-of-two scales from bits.** Kernels for RFC-0060's bit-exact class preserve F32 subnormals (no flush to zero) and build power-of-two scales (E8M0, 2^k factors) by setting exponent bits, never through `exp2` or `ldexp` approximations. They are compiled with contraction off (`--fmad=false` for CUDA and HIP translation units, `-ffp-contract=off` or `#pragma clang fp contract(off)` for C++, Metal's safe math mode, `precise` expressions in Vulkan shaders so SPIR-V carries `NoContraction`). Rust never contracts implicitly, so `kernels::reference` defines the two-rounding result. The core's ggml CPU backend is built with `-ffp-contract=off` (RFC-0040 §8.1). Other kernels may contract.

#### 4.2 Per-op rules

| Op | Rule |
|---|---|
| `Gemv`, `Gemm`, `GateUpAct`, `MoeGrouped`, `Logits` | §4.1; outputs F32. BF16 outputs only where the next op is a matrix product whose input may be BF16 (§4.3 rule b) |
| `AttnDecode`, `AttnPrefill` | scores `q·k` in F32; running max and sum in F32; `exp` in F32; the `p·v` product accumulated in F32; split-KV partials merged in F32 |
| `KvWrite` | K, V and latents rounded once to the model's KV dtype (round to nearest even); RoPE computed in F32 before the rounding |
| `ScanDecode`, `ScanPrefill` | state and gates in F32; chunked prefill must give the same state as the recurrent form within the per-op tolerance |
| `RmsNorm`, `LayerNorm` | sum of squares in F32; `eps` as the family defines it; scale applied in F32 |
| `Rope` | angles from positions computed in F32, or F64 for positions above 2^20; `sin`/`cos` single precision |
| `MoeRoute` | logits in F32; softmax or sigmoid in F32; top-k on F32 values with ties broken by lower expert index |
| `MoeCombine`, `PartialReduce` | F32, in rank order (RFC-0040 §3.4) |
| `LogitsProcess`, `TopKTopP`, `SampleSeeded`, `Argmax`, `VerifyAccept` | F32 logits; ties broken by lower token ID; sampling from a counter-based generator (ADR-0107). The key is (`client_seed`, choice index) whenever the request supplies a seed, and (`generated_seed`, `SequenceId`) when ingress generates it. Draws are indexed by `rng_counter: u64`, the sequence's monotonically increasing draw counter (RFC-0032 §3.7). The choices of an `n > 1` request draw distinct numbers either way, and a repeated request with the same client seed reproduces its draws. Draw order within a step is ADR-0107's. A draw does not depend on batching or placement |

**The model's KV dtype** is what the reference stores, recorded per layer in `KvFormula::dtype` by the family module (RFC-0043 §5; RFC-0022 §2): FP8 where the model's own configuration specifies an FP8 cache (ADR-0019's revisit case); otherwise BF16 for safetensors sources whose weights are BF16 or lower, F16 for GGUF sources (matching ggml's default cache type, so the calibration baseline agrees), and F32 only when a family module requires it. RFC-0022 stores KV in this dtype in exact mode; anything lower is the `kv_lossy` knob (§10).

#### 4.3 Activation quantisation inside a matrix product

A kernel that converts its activation input to a narrower type before multiplying changes the result. The rule (ADR-0421):

- **(a) Declared by the checkpoint: part of the reference.** When the checkpoint's quantization metadata declares an activation scheme (the FP8 config's `activation_scheme: dynamic`, compressed-tensors `input_activations`), the reference executor applies exactly that quantisation (§7), and kernels must apply the same one. For example, DeepSeek-V3's FP8 originals declare dynamic E4M3 activations per token per 128-element group, scale = amax / 448.
- **(b) 8-bit blocks for GGUF integer-coded weights and transcribable tensors: allowed.** `Q8Block32` (32 elements, one F16 scale, as ggml's `Q8_0`/`Q8_1` layouts) and `Q8K256` (as ggml's `Q8_K`) are how the ggml CPU path computes these products, and that path is the calibration baseline (§6.2). Their effect is therefore already inside the tolerance. The argument holds for a safetensors tensor only when the baseline runs it through ggml, which is exactly when it is transcribable (§3.5): MXFP4 always, symmetric INT4 and INT8 groups when the scales fit F16. The kernel must use the activation block the ggml type uses (`Q8Block32` for `Q4_0`, `Q8_0` and `MXFP4`). The M1 calibration set includes a MoE checkpoint in the CPU-expert scheme itself (compressed-tensors symmetric INT4, group 32), small enough for §6.2's calibration budget: a published one where it exists, otherwise one produced in that exact scheme for the fixture set. The path is therefore gated directly, not only by the argument. The 1T-class Kimi-K2-Thinking-class model is too large to calibrate; it is gated against the class ceiling by RFC-0060 §5.7's layer-local parity on the release artefacts. `Bf16` rounding of matrix inputs is allowed on the same grounds, because the perturbation floor (§6.3) is defined by it.
- **(c) Anything else needs a per-family harness pass.** FP8, MXFP8 or FP4 activations on a model that does not declare them may be used in exact mode only for a (family, dtype class, shape class) that passed the end-to-end gate (§6) with that kernel. Passing variants are recorded in the release's `ConformanceManifest` (RFC-0060 §6.4) as `PathCert`s for the (family, quantization class, path) with that kernel variant. If RFC-0060 prefers a dedicated entry type, this RFC follows it. Otherwise such a kernel is a candidate only under approximate mode, labelled `act_lowp` (§10). Prefill is where these paths matter, because decode is memory-bound and gains nothing from them.

`Numerics::act_in` carries the variant, so selection can enforce the rule mechanically.

### 5. Implementing the exactness contract: what is measured

ADR-0019 decision 9 asks for three levels of checks. RFC-0060 builds the harness; this section defines the measurements.

#### 5.1 Op level

For a candidate output `y` and a reference output `r` of the same op on the same inputs, with the definitions RFC-0060 §4.2 uses:

```text
rel_l2(y, r)         = ‖y − r‖₂ / ‖r‖₂
max_abs_scaled(y, r) = max_i |y_i − r_i| / (rms(r) + 1e−30)
```

Conformance measures both against the F64 reference; the runtime spot check measures `rel_l2` against the F32Fixed `kernels::reference` (§6.4).

Integer-valued ops (`Argmax`, `MoeRoute`'s indices, `TopKTopP`'s set, `SampleSeeded` with a fixed generator state, `VerifyAccept`, `assign` in RFC-0040 §3.4) must match **bit for bit** when given bit-identical inputs. Integer-dot kernels that consume the same quantised activations must also agree bit for bit with each other before scaling (an AMX kernel against the VNNI kernel, for example), because I32 accumulation is exact.

#### 5.2 Layer level

`rel_l2` of the residual stream at every layer boundary against the reference, teacher-forced. For models within the reference budget it is a **diagnostic**: it localises a failing end-to-end result to a layer and a block type, and it is reported in every harness run, but it is not a gate on its own. For models beyond the budget, including every 1T-class model, RFC-0060 §5.7's **layer-local parity on sampled layers gates the release**, measured on the release artefacts through RFC-0040 §3.8's tap variant (ADR-0601). Its threshold is RFC-0060 §5.7's: `τ_ll = max(2 · S_ll, φ_ll)` per (layer kind, dtype class, category, statistic), from the baseline's layer-local spread on the calibration models. This RFC supplies the ceilings that cap it. **Layer-local ceilings** `C_ll(kind, d, statistic)` are the maximum of `τ_ll` over the calibration models of class d that contain that layer kind, rounded up to two significant digits, as `C_class` is built (§6.2). There is one per layer kind (dense FFN, MoE FFN, full attention, sliding-window attention, linear attention, the head window) and per statistic (mean and p99 `rel_l2`; top-1 and `KL_K` for the head window). They are published in `tolerances-v<N>.toml` with the end-to-end ceilings, and they are provisional wherever a class has fewer than three calibration families with that layer kind.

#### 5.3 End to end (the gate)

For a corpus sequence `x₁ … x_T`, both executors run teacher-forced, so position t sees `x₁ … x_t` whatever either would have sampled. At each scored position:

```text
log P_ref = log_softmax(logits_ref)      in F64, over the full vocabulary
log P_cand = log_softmax(logits_cand)    in F64
KL_t  = Σ_v P_ref(v) · (log P_ref(v) − log P_cand(v))          nats
top1_t = [argmax P_cand = argmax P_ref]                         ties to the lower token ID
```

Aggregates, pooled and per corpus category: **mean KL**, **p99 KL** (nearest-rank over positions), and **top-1 disagreement** `1 − mean(top1_t)`. A candidate passes when every aggregate is within its tolerance (§6).

**Stored references.** When the reference's logits are not computed alongside the candidate (the usual CI case), the harness works from a stored reference block per position. RFC-0060 §5.3–§5.4 define the block and its estimators, and this RFC adopts them:
- the reference's logits at its top-K ids, its total logsumexp `lse` and the tail mass `log r` (both F64), and a tail sample of ids drawn in proportion to the reference's own probabilities;
- `KL_K`, the KL coarse-grained to the top-K plus one tail bucket, which is a lower bound on the full KL, and `KL_tail`, the unbiased estimate of the remainder from the tail sample;
- the tightness obligation at M0: on the fixtures and one real checkpoint per family, `|KL − KL_K| ≤ max(0.01 · KL, 0.01 · C_kl_mean(d))` at the mean and at p99, per category. This is relative plus absolute, so it stays defined as KL → 0, and it repeats on every corpus version. Also reported, never gated: free-running greedy decoding's first divergence position (near-ties legitimately cascade, ADR-0019 decision 2), and the full-depth layer diagnostics. The exception is layer-local parity for models beyond the reference budget, which gates (§5.2).

Speculative paths are checked the same way: the logits a verify window produces at each window position are compared with the reference's logits at that position. Acceptance itself is exact given the logits (ADR-0017).

#### 5.4 Placements under test

Every execution path the solver can choose in exact mode (ADR-0019 decision 4) is run against the same reference: single device per provider; execute-in-place splits with each `MissSplit` extreme; 2- and 3-stage pipelines on the simulated fabric at the family's legal boundaries; cross-node TP/EP; each wire dtype (§8); speculative and plain decoding; prefix-cache reuse, KV paging between tiers, and session save and resume. RFC-0060 owns the matrix and its schedule.

#### 5.5 Corpus requirements

The corpus is versioned (`corpus-vN`) and held by RFC-0060. Its categories are chat, code, long context (≥ 32 K tokens), tool-call JSON and multilingual text (ADR-0019 decision 2). Size rules:

- **Calibration** scores at least 50 000 positions per category. At a top-1 agreement near 0.95, the binomial standard error is `√(0.95 · 0.05 / 50 000)` ≈ 0.001 [estimate], so a calibrated spread is known to about ±0.2 percentage points.
- **Per-change CI** scores 4096 positions per category on the M0 fixtures and the Tier 1 quick set. Inputs are fixed, so run-to-run variation comes only from kernel non-determinism, and the gate is the same tolerance.
- **Long context** scores only the last positions of each long document, where long-range attention matters, to bound reference cost. RFC-0060 §5.2's table fixes the counts per tier: `smoke` scores 32 documents' last 128 positions, `standard` 32 documents' last 256, and `calibration` 64 documents' last 784. `long` therefore reaches its size through more documents, and every tier has at least 32 sequences per category for the sequence bootstrap.

These are minimums. RFC-0060 §5.2 holds the one corpus table, with positions per category for each corpus tier. Calibration uses its `calibration` tier, which meets the first rule, and every tolerance and ceiling in §6 comes from that tier, never from a per-change tier.

#### 5.6 Quantization error, reported separately

How far a 4-bit file is from its BF16 original is a property of the file, not of Syntropy (ADR-0019 decision 1). Where both exist, the harness also reports KL and top-1 of the quantized reference against the higher-precision reference, and the model card in the CLI and dashboard shows it (RFC-0051). It never gates.

#### 5.7 Certified context length

The exact contract covers only the positions the harness measured. Today's corpus stops at 128K tokens (`long`, RFC-0060 §5.2) and op cases at 131,072, while v1 families advertise contexts up to 1M (RFC-0043 §5.5).

- **What is certified.** Each family's `FamilyCert` records `max_ctx`: the longest context at which end-to-end or layer-local parity passed for that family (RFC-0043 §9.4). For a family whose checkpoints advertise no more than the `long` corpus reaches, `max_ctx` is the longest `long` document scored.
- **Families that advertise more than 128K.** Their conformance adds a sampled-layer `long` slice up to the advertised maximum: layer-local parity (§5.2, §7 "Large models") on single-layer windows at positions beyond 128K, covering every attention kind of the family. `max_ctx` is then the longest position that slice passed. RFC-0060 §5.2 and §5.7 own the slice's documents and schedule.
- **Labelling.** A position above the model's `max_ctx` is not covered by the contract. From the first such position the response is labelled beyond the certified context, in the same fidelity object that carries `exact` and `exact (derived certification)`: `FidelityReport::beyond_certified_from` gives that position and the `Syntropy-Certified-Context` header gives `max_ctx` (RFC-0050 §10, RFC-0051). This is distinct from `Uncertified`, which means the build has no valid conformance manifest (RFC-0060 §6.4). The label is not an approximate knob: nothing about the computation changes, and `TokenEntry::approx` stays empty. Ingress knows each sequence's positions, so it applies the label without help from the stages.
- **Plans.** The plan preview shows `max_ctx` beside the configured context, so a user who sets a longer context sees where certification ends before any request runs.

### 6. Tolerances

#### 6.1 Form

A tolerance is a triple per (dtype class, pooled or per corpus category): `τ_kl_mean`, `τ_kl_p99` (nats) and `τ_top1` (maximum top-1 disagreement). Op-level tolerances are a single `τ_op` on `rel_l2` per (op, scheme, shape class) (§6.4).

#### 6.2 Calibration (ADR-0422)

Calibration uses models from at least three families per dtype class where checkpoints exist, from RFC-0060's fixture set. At M0 only `llama` and `qwen_moe` exist, so M0 calibration uses at least three models drawn from those two families and publishes its ceilings marked `provisional`; M1 recalibrates over at least three families, and any rise at that point is logged in ADR-0422 like any other.

**Calibration models fit the budget, with attention counted.** The `calibration` tier does not score 250 000 positions on short sequences only. It prefills about 2.5e6 tokens per category set, most of them in `long` documents of 32K–128K tokens (RFC-0060 §5.2), and attention at those lengths dominates. Each pass therefore costs, per RFC-0060's formula:

```text
FLOP(pass) = 2 · N_act · P_prefilled  +  Σ over documents  2 · n_layers · d_attn · L²
```

Both passes, the reference and the baseline, must each fit RFC-0060's 48 h rule at the rate measured for that pass: `F_ref` for the reference (below) and the ggml CPU rate on R-LABCPU for the baseline. Two worked figures [estimate, from RFC-0060's token counts and formula]: Qwen3-30B-A3B (48 layers, `d_attn` 4096, 3.3B active) costs ≈ 5.6e16 FLOP per pass, ≈ 1.65e16 linear (2 × 3.3e9 × 2.5e6) plus ≈ 3.97e16 attention (62 × 4.22e14 + 2 × 6.76e15), which is 78–156 h at 100–200 GFLOPS; Llama-3.2-1B costs ≈ 1.3e16 FLOP, 18–36 h. A CPU reference therefore confines calibration to dense models of about 1B, and the ceilings that gate the MoE paths would rest on dense evidence. The earlier bound of ≈ 17 B active parameters counted 250 000 short positions and no attention; it is withdrawn.

**The reference runs on one GPU in calibration.** ADR-0019 requires one device and a fixed reduction order, not a CPU. Calibration therefore runs the reference executor's `F32Fixed` semantics on one GPU (`F32Fixed` on device `Gpu`, RFC-0060 §4.1): the same operation order as the CPU reference (pairwise reductions over fixed 32-element blocks in index order, softmax subtracting the row maximum), floating-point contraction off, F32 subnormals kept, IEEE-rounded division and square root, and the transcendental functions (`exp`, `log`, `tanh`, `erf`) taken from `kernels::reference`'s own software implementations rather than the device's libraries. Weights stay in their stored quantization in device memory and are dequantised to F32 at use (§7), so the device needs room for the quantised weights, the KV of the longest `calibration` document and scratch, never for F32 weights. It runs on an untimed GPU with enough memory, from the agent GPU pool or a cloud instance, never on a timed lab rig.
- **Cross-check.** On the `nano` and `micro` fixtures of every v1 family, the GPU reference must reproduce the CPU `F32Fixed` reference bit for bit: logits, every tapped hidden state and the stored reference blocks. A mismatch is a reference defect. It blocks calibration until fixed, like RFC-0060's reference-defect rule.
- **Its rate.** `F_ref` for calibration is the GPU reference's measured sustained rate on the model being calibrated. The budget rule above is recomputed from it before any calibration model is chosen, and the chosen models' recorded costs are published with the tables.
- **What stays on the CPU.** The CPU `F32Fixed` reference remains the reference for every per-change and nightly parity run, the warm-up canary digest and the F64 diagnostic mode (§7). Only calibration needs the GPU's throughput. R-LABCPU's sizing is re-checked against this split (RFC-0060).

A `widen.f32` baseline also needs 4 bytes per parameter of the model in memory or page cache, so the baseline host's memory bounds N_total for those schemes. Each family is calibrated on its smallest published checkpoint of the scheme within both bounds. Larger models, including every 1T-class model, are not calibration models. They are judged against `C_class` by RFC-0060 §5.7: end to end on the parts within budget, plus layer-local parity through RFC-0040 §3.8's tap variant.

**MoE ceilings stay provisional until a MoE model is calibrated directly.** A class ceiling that gates a MoE model is marked `provisional` in `tolerances-v<N>.toml`, and in the exit report of any milestone, until at least one MoE checkpoint of a v1 family has been calibrated with the GPU reference within the budget above. This applies to the M0 exit, and to the M1 exit if the GPU reference has not landed by then. A provisional ceiling still gates; the label tells readers it rests on dense calibration models only.

Per calibration model m and corpus category:

```text
S_m = metrics(reference executor, the baseline for m's schemes, below)                                    -- the spread
F_m = metrics(reference executor, reference with every matrix-product input rounded to BF16)            -- the floor (§6.3)
φ   = 0 by default; F_m for a class only under ADR-0019's revisit clause (§6.3)
τ_m = max(2 · S_m, φ) pooled;  max(3 · S_m,c, φ) per category c        per metric (top-1 as disagreement rate)
C_class = max over m in the class of τ_m, rounded up to two significant digits                          -- the ceiling
```

**The baseline per scheme.** The spread needs a second implementation that is independent of the reference but computes the same contract:

| Schemes | Baseline | Floor end to end |
|---|---|---|
| GGUF types and plain floats | the ggml CPU provider running the same family graph on the same pack | `φ = 0` (default rule) |
| transcribable safetensors tensors (§3.5) | the ggml CPU provider over the exact transcription | `φ = 0` |
| FP8 (every scale kind), non-transcribable `PackedInt`, NVFP4 | the ggml CPU provider over a `widen.f32` copy (§3.3) | `φ = F_m` from the start |

**Declared activation quantisation is part of every baseline.** Where the checkpoint declares activation quantisation (§4.3 rule a), the baseline graph applies exactly that quantisation before each affected ggml `mul_mat`. A Syntropy CPU fake-quant step quantises the F32 activation as declared and dequantises it back to F32; for DeepSeek-V3-class FP8 that is E4M3 per token per 128-element group, with scale `amax / 448`. The reference does the same (§7). Without this step, `S_m` would measure the effect of skipping the declaration, and the ceiling would pass a kernel that ignores it.

For the third row, the baseline's dequantised weights and quantised activations then equal the reference's exactly, so `S_m` measures summation order only, and `2 · S_m` would reject the BF16 rounding that rule (b) allows. That is ADR-0019's revisit case, known in advance: these schemes use `φ = F_m` from their first calibration, and the choice is recorded as a revisit event in ADR-0422's log. The `Fp8` class is calibrated this way at M1, on at least three FP8 checkpoints of different families within the calibration budget, where published. DeepSeek-V3- and GLM-class FP8 models above the budget are judged against the resulting ceiling by RFC-0060 §5.7.

A candidate passes on a calibration model m when it is within `τ_m`, and on any model of the class when it is within `C_class` (ADR-0019 decision 3: within 2× the spread, and within the class ceiling). The per-category factor 3 is ADR-0601's starting default for the noisier per-category estimates, recalibrated at M0.

**Published with the tolerances**, per (dtype class, category), beside `S` and `S_c`:
- the **design effect** `deff_c`: the variance of the category's top-1 disagreement rate under a bootstrap that resamples whole sequences, divided by the binomial variance at the same rate and position count. It measures how much clustering within sequences inflates the noise, and RFC-0060 §5.4's top-1 test divides counts by it. Values below 1 are published as 1;
- the **canary ceiling** `C_canary(d)`: the 99.9th percentile, over the calibration runs of correct candidates, of the canary statistic RFC-0060 §5.8 defines (the mean `KL_8` over the 32 canary positions whose reference margin is at least `τ_tie`). It is a separate statistic from `C_kl_p99`, so it gets its own ceiling;
- the **definitional tolerance** `τ_def(f)` per family, with the noise measurements it comes from (§7.1);
- a `provisional` flag on every ceiling that rests on dense calibration models only (above).

Outputs go to `numerics/tolerances-v<N>.toml`, versioned, published in the docs and stamped into each release (ADR-0019). Calibration re-runs when the ggml pin moves (the spread can change), when the reference executor changes, and when the corpus version changes. A ceiling that rises by more than 10 % needs a note in the release and in ADR-0422's log. Ceilings never loosen silently.

**Release blocker** (ADR-0019 decision 3): a release that supports a dtype class with no published ceiling cannot ship. M0's exit includes provisional ceilings for `Float16`, `Int8` and `Bits4`; M1's makes them final and adds `Fp8` and `Sub4`, except that a ceiling gating MoE models stays `provisional` until the condition above is met.

#### 6.3 The perturbation floor

ADR-0019's revisit clause anticipates a class whose reference-versus-ggml spread is so small that 2× of it rejects correct tensor-core kernels. `F_m` is the evidence for that case. It measures what rounding a matrix product's inputs to BF16 does to the output distribution, which is the smallest perturbation any tensor-core kernel introduces and one §4.3 rule (b) explicitly allows. Every calibration measures and publishes `F_m` next to `S_m`.

End to end, the floor stays at 0 (ADR-0019's rule unchanged, as ADR-0601 also fixes). It is switched on for a dtype class (`φ = F_m`) only when a kernel shown correct by op-level parity and the independent review agent fails the end-to-end gate while `2 · S_m < F_m`. Because a gate failed, a maintainer approves switching it on (**Maintainer action**; ADR-0021 decision 5). That event is recorded with its evidence in ADR-0422's log and the release notes. **The one exception** is §6.2's `widen.f32` baseline (FP8, non-transcribable `PackedInt`, NVFP4). There `S_m` measures summation order alone by construction, so `φ = F_m` is set from the first calibration without waiting for a rejected correct kernel. ADR-0601 decision 4 names the same exception, and ADR-0422 decision 2a records it. At op level (§6.4) the floor always applies, because op spreads can be exactly zero when ggml and the reference compute an op the same way.

#### 6.4 Op-level tolerances

RFC-0060 §4.2 runs op conformance as a three-way comparison (candidate, the ggml op, the F64 reference) and takes its thresholds from this section. Per op class c and dtype class d, with `rel_l2` and `max_abs_scaled` as RFC-0060 defines them against the F64 reference:

```text
ε_op(c, d)          = p99 over cases of rel_l2(F32Fixed reference with BF16-rounded inputs, F64)        the floor
ceiling_op(c, d)    = max over calibration cases of 2 · max(rel_l2(ggml, F64), ε_op), rounded up to 2 s.f.
ceiling_op_max(c,d) = the same construction on max_abs_scaled
tol_op(c, d)        = max(2 · p99 rel_l2(ggml, F32Fixed reference), ε_op)        candidate against ggml (RFC-0060 rule 3)
tol_ref(c, d)       = tol_op(c, d)                                                candidate against kernels::reference
```

- `tol_ref` is what the runtime spot check in `Warming` uses (RFC-0040 §6.2). It compares against `kernels::reference`, which is always present in the core, rather than ggml, which may not be present on that device. Using the same value as `tol_op` keeps a kernel that passed CI from failing on a user's machine.
- At op level the floor always applies, because an op's spread can be exactly zero when ggml and the reference compute it the same way (§6.3).
- Inputs follow RFC-0060's `InputRecipe`: normal and heavy-tailed activations, outlier channels, full-range and degenerate scales, near-tie attention scores, and real tensor slices nightly. The BF16 floor therefore reflects real ranges.
- For ops ggml lacks, the second independent implementation RFC-0040 §8.5 requires replaces the ggml op in every formula.
- For an op whose `OpKey` declares activation quantisation, the ggml comparison inserts the same fake-quant step as the end-to-end baseline. A kernel that skips the declared quantisation, such as a W8A16 kernel on a declared-W8A8 op, must fail `tol_op` (Testing).
- RFC-0060's bit-exact class has every tolerance at 0 for the providers listed below. Anything else on those ops is in the toleranced class with `tol_op`.

**Bit-exact scope (decided here; RFC-0060 §4.2 rule 4 cites it).** The class is defined by what a provider can reproduce exactly, not by format:
- **In the class:**
  - copies, row gather, KV page moves;
  - **standalone dequantisation of every weight format to F32**: GGUF block types (k-quant min terms and i-quant grids included), FP8 of every scale kind, MXFP4, NVFP4 with its global scale, compressed-tensors integers. Each element is a fixed sequence of IEEE operations: F16 or E8M0 scale decoding, integer subtraction, multiplication by scales, the k-quant `d·sc·q − dmin·m`. Performed in `kernels::reference`'s operation order, with contraction off, F32 subnormals kept and power-of-two scales built from exponent bits (§4.1 rule 7), it gives identical bits whether or not a scale is a power of two;
  - exact dtype widening (E2M1, E4M3, E5M2 to F16/BF16/F32; F16 and BF16 to F32);
  - BF16 narrowing with round to nearest even;
  - the wire codecs (§8.2), each against its reference codec;
  - argmax and top-k index selection on identical scores (lower index wins ties).
- **Not in the class:** any scale applied inside a fused product, that is, a scale multiplying a partial sum (`s · Σ fl(e · a)` against the reference's `Σ fl(fl(s · e) · a)`, §3.2); every matrix product and reduction; and a dequantiser on a provider that flushes F32 subnormals or cannot turn contraction off for it. Those are toleranced with `tol_op` on that provider, as the table below records.

This supersedes an earlier, narrower wording ("element widening and MX scaling only"), which was right about fused scale application and wrong about standalone dequantisation.

**Bit-exact ops per provider.** CI computes the list each build by running the bit-exact-class cases against `kernels::reference`; this table is the expected result:

| Provider | Bit-exact | Toleranced |
|---|---|---|
| CPU (Rust) | every bit-exact-class op | — |
| CUDA, HIP native | every bit-exact-class op (contraction off per §4.1 rule 7) | — |
| Metal native | every bit-exact-class op, if the safe math mode excludes contraction and keeps F32 subnormals as expected; any op that fails moves right | as measured |
| Vulkan native | byte moves, wire codec, index ops; dequantisation and row gather where the shader uses `precise` and the device reports F32 denormal preservation (`shaderDenormPreserveFloat32`) | the rest |
| ggml CPU (core build) | every bit-exact-class op ggml has (built with contraction off) | — |
| ggml CUDA, HIP, Metal, Vulkan | byte moves (copies, KV page moves) | dequantisation and row gather of quantised tables (upstream build flags, which may contract). Index ops have no ggml mapping (RFC-0040 §8.2) |

All five quantities are computed in CI on every ggml pin bump and published in `numerics/tolerances-v<N>.toml` next to the end-to-end table.

### 7. The reference executor

ADR-0019 decision 1 defines it, and this section specifies its numerics. Its code is split as RFC-0060 §4.1 describes: `syntropy_graph::reference` (RFC-0043) interprets the graph IR using `kernels::reference` op semantics (RFC-0040 §1), and `syntropy-testkit`'s `RefExecutor` (RFC-0060) drives it with weights, a numeric mode and taps. One production path calls `graph::reference`: the canary-digest computation during `Loading` (RFC-0041 §5.3), so it is release code and in the release's conformance scope (RFC-0060 §4.1).

| Aspect | Rule |
|---|---|
| Graph | the same family module and graph IR the product uses (RFC-0043), interpreted op by op with `kernels::reference` semantics, so the reference tests kernels, placement and wire, not graph construction. A bug in the family module is in the reference, the ggml baseline and the candidate alike, and every parity gate passes it. Graph construction is checked separately, by the model-definition oracle of §7.1 against the publisher's own code (corrected 2026-10-20: this row used to cite a conformance check against published expected outputs that no section defined) |
| Device | one CPU process, multithreaded across output rows, with every output element's reduction order fixed. Calibration runs the same semantics on one GPU (§6.2), cross-checked bit for bit against the CPU on the `nano` and `micro` fixtures |
| Weights | read from the same pack or source bytes the candidate uses; dequantised per §2 to F32 at use |
| Activations | F32 everywhere, except where the checkpoint declares activation quantisation (§4.3 rule a), which is applied exactly as declared |
| Accumulation | F32, with the order RFC-0060's `RefMode::F32Fixed` fixes: every reduction pairwise over fixed 32-element blocks in index order; softmax subtracts the row maximum |
| KV | stored in the model's KV dtype (§4.2) |
| Diagnostic mode | RFC-0060's `RefMode::F64`: F64 storage, compensated summation, F64 `exp` and `log`; used for op-level conformance and to check that the F32 reference itself sits well inside the floor |
| Large models | **layer-sampled mode**: the candidate's residual stream at a boundary `a` is fed to the reference for layers `[a, b)`, and the outputs are compared at `b`. **Gating uses single-layer windows, `b = a + 1`** (RFC-0060 §5.7 item 3), on the sampled layers, so each result isolates one layer's local error; wider windows are diagnostic only. The sampled layers cover every block type of the family (each attention kind, dense FFN, MoE, scan) in each run. The candidate's hidden states and KV prefixes come from RFC-0040 §3.8's tap variant, which a stage builds on request only when `debug.parity_api` is set (RFC-0012 §8.5), so the layer-sampled gate runs on the release artefacts themselves without changing their service programs (RFC-0060 §5.7). End-to-end metrics on the largest models use the reduced corpus (ADR-0019 consequences) |
| Cross-validation | the spread `S_m` (§6.2) is exactly the cross-validation against the ggml CPU provider. If a recalibration finds `S_m` more than 2× its previous value, calibration stops (ADR-0021 decision 5): either ggml or the reference changed behaviour, and a maintainer approves an agent's investigation of which one before calibration resumes (**Maintainer action**) |

**Warm-up canary data** (RFC-0001 §3.3). RFC-0060 §5.8 defines the `ReferenceDigest` and the canary's pass rule: 32 fixed canary tokens, top-1 checked at every position whose reference margin is at least `τ_tie`, and the mean `KL_8` over the checked positions within this RFC's canary ceiling `C_canary(d)` for the model's dtype class (§6.2). The reference that produces it is this section's F32Fixed reference, run during `Loading` when no published or cached digest exists (RFC-0041 §5.3). The canary is a smoke test that catches broken devices, kernels, layouts and wiring on the user's machine. The contract is checked by the harness, not by the canary.

#### 7.1 The model-definition oracle

"The model" is what the publisher's reference implementation computes, run in F32 on the same dequantised weights. The oracle checks Syntropy's family module against that, which no parity gate can do, because every parity gate compares two executions of the same module. RoPE and YaRN scaling, the attention scale, the norm form and epsilon, routing normalisation, sink handling, activation clamps and tied heads are the bugs it exists to catch.

| Aspect | Rule |
|---|---|
| Reference code | the publisher's own reference implementation for the family: the modelling code in the checkpoint's repository, or the inference code the publisher publishes, pinned by commit. It is run offline in a sandboxed environment with no network, never in CI's gating path and never shipped |
| Weights | the conformance checkpoint's weights (RFC-0043 §9.3), dequantised to F32 by `kernels::reference` (§2), so both sides compute on bit-identical F32 values and only the model definition differs. Loaded into the publisher's code in its own F32 checkpoint layout |
| Precision | F32 throughout on the publisher's side. Syntropy's side is `RefExecutor` in `F32Fixed` with declared activation quantisation switched off, because the publisher's F32 path does not apply it; declared activation quantisation is checked separately by the baselines' fake-quant step (§6.2) |
| Golden data | per family, stored offline: full-vocabulary logits at every position of about 64 short prompts and a few long ones, and the residual stream's per-row L2 norm at 3–4 layer boundaries spread through the depth. Prompts come from the corpus's licensed sources (RFC-0060 §5.2) |
| Storage | the fixture store (RFC-0060 §12.3), under its licence rule: public read only for golden data of permissively licensed models, private otherwise, because the data derives from the model and its licence travels with it |
| Families with no small checkpoint | DeepSeek-V4-class and Kimi-K3-class have no checkpoint the publisher's code can run in F32 on one host. Their golden data comes from (a) the publisher's code run on the family's `micro` fixture, with a configuration file written from its `FamilySpec` in the publisher's own format, and (b) single-layer outputs of the publisher's code on the full checkpoint for one layer of every block type, fed the same recorded input hidden states, compared with `RefExecutor`'s layer-local output (§7 "Large models") |
| When it runs | on every change to a family module or to a shared block it uses: exactly the changes that alter the family's `FamilyCert` code hash (RFC-0043 §9.4). The golden data is fixed per (family, checkpoint revision, reference-code commit); only `RefExecutor` runs per change, so the check costs one F32 reference pass over the golden prompts |
| What it gates | a `FamilyCert` is issued only when the oracle passes, from M0 for `llama` and `qwen_moe` (RFC-0043 §9.2) |

**The definitional tolerance.** It is set from the same metrics as the end-to-end gate (§5.3: mean KL, p99 KL and top-1 disagreement, top-1 counted only at positions whose golden margin is at least `τ_tie`), plus the relative difference of each tapped norm. Both sides are F32 renderings of one F64 computation, so a correct module differs from the publisher by at most the sum of the two sides' own F32 rounding. Per family f and metric:

```text
S_pub(f) = metric(publisher code in F32, the same code in F64)        on the golden prompts
S_ref(f) = metric(RefExecutor F32Fixed, RefExecutor F64)              on the same prompts
τ_def(f) = 2 · max(S_pub(f), S_ref(f))
```

`τ_def` is published in `tolerances-v<N>.toml` beside the ceilings (§6.2), with `S_pub` and `S_ref`. A family whose publisher code cannot run in F64 uses `S_ref` for both terms, and the tables say so. An oracle failure blocks the family's certificate. It is never waived by raising `τ_def`; a correct module that fails is a reference-code or weight-loading defect, investigated and recorded with its evidence.

**Planted defects at module level.** The oracle is the catching layer for defects inside a family module, which kernel-level planted defects cannot represent. `graph` carries test-only mutants of each M0 family module (wrong YaRN or RoPE scaling factor, routing weights not renormalised, wrong attention scale, RMSNorm epsilon or form changed, tied head not tied), built only with a test feature. Each mutant must pass every parity gate, which shows why the oracle is needed, and fail the oracle. RFC-0060 lists them in its planted-defect table.

### 8. Cross-device rounding and activation precision on the wire

#### 8.1 Where results legitimately differ

| Source | Why | Bound |
|---|---|---|
| Different kernels for the same op (CPU against GPU sites, native against ggml) | summation order, integer against float dot paths, 8-bit activation blocks | each kernel passes op-level parity; the combination passes end to end |
| Shape-dependent kernels (decode against verify windows, batch buckets) | split-K and tiling change with rows | the harness runs every bucket (§5.4) |
| The expert cache moving an expert between sites | a different kernel computes it on different tokens | `MoeCombine` sums in rank order, so only the sites' own rounding differs (RFC-0040 §3.4) |
| Heterogeneous stages | each backend's kernels | pipeline placements in the harness |
| Wire precision | rounding at a hop | §8.2 |

None of these is allowed to be large. All of them are allowed to be nonzero. That is ADR-0019's contract, and the product says so in its docs.

#### 8.2 The wire rule

`HopPlan::wire` (RFC-0001 §5.4) and `ActivationFrame::dtype` (§5.8):

1. **Exact mode defaults to F32.** The reference holds the residual stream in F32 (§7), so F32 on the wire adds no rounding.
2. **BF16 is certified per family and per shape class.** For each family, the harness runs the pipeline placements of §5.4 with BF16 hops, separately for decode frames and for prefill frames (ADR-0019 decision 5). A (family, shape class) that passes within the class tolerance becomes a `WireCert { family, phase, dtype: BF16 }` entry in the release's `ConformanceManifest` (RFC-0060 §6.4). The solver may then choose BF16 for those hops in exact mode. Conversion is round-to-nearest-even from F32, NaN preserved, applied once at the sender.
3. **Lossy wire formats** (F16 with its range risk; Q8 rows with per-row or per-32 scales) exist only under `wire_compress` (§10).
4. Hyper-connection models carry every residual stream at the same dtype.

**Wire codecs** (RFC-0031 §6.4 assigns the numerics of its codes here). The sender encodes once from the F32 residual stream; the receiver decodes to F32. Each codec is a pure function, and `kernels::reference` holds the defining implementation: every provider's codec must match it bit for bit (RFC-0060's bit-exact class). Non-finite values are never clamped away (§4.1 rule 6).

| Code | Encode (F32 → wire) | Decode (wire → F32) |
|---|---|---|
| 1 `F32` | identity | identity |
| 2 `BF16` | round to nearest even; NaN stays NaN with its sign; infinities kept; no flush to zero | exact widening |
| 3 `F16` | round to nearest even; subnormals kept; NaN stays NaN; a finite value of magnitude ≥ 65520 becomes ±∞, F16's range risk, so the overflow propagates and the tail fails the sequence (`NanInLogits`) rather than clamping it | exact widening |
| 4 `Q8B32` | blocks of 32 consecutive elements along `hidden` within one stream, the last block zero-padded when `hidden` is not a multiple of 32. Per block: `amax` = the largest finite magnitude; `d = amax / 127` in F32; scale `s = F16(d)`, rounded to nearest even; `q_i = clamp(round_half_away(x_i · (1/d)), −127, 127)`, with `q_i = 0` when `amax = 0`. These are ggml's `Q8_0` rules, so the codec equals the `Q8Block32` activation quantiser (§4.3) | `x_i = s · q_i`, padding dropped |
| 5 `F8E4M3B32` | the same blocks; `d = amax / 448`; `s = F16(d)`; `e_i = E4M3(x_i · (1/d))`, round to nearest even, saturating at ±448 (the OCP FN encoding has no infinity) | `x_i = s · e_i` |

For codes 4 and 5, a block holding a NaN or an infinity, or whose `d` rounds to infinity in F16 (`d ≥ 65520`, so `amax ≥ 65520 × 127` for `Q8B32` and `65520 × 448` for `F8E4M3B32`), is written with scale F16 NaN. The whole block then decodes to NaN, and the tail fails the sequence (`NanInLogits`) instead of serving a clamped value. A block whose `d` underflows F16's subnormal range decodes to zeros; that loss is part of what the knob's harness report measures.

A lossy codec is offered for a family and phase only after its `wire_compress` harness report is published (§10). The approved (family, phase, codec) list ships in `tolerances-v<N>.toml`, beside the BF16 `WireCert`s.

Decode frames are small either way: a 7168-wide hidden state is 28 KiB per token in F32 against 14 KiB in BF16, and at 2.5 GbE the difference is ≈ 49 µs of serialisation per hop [estimate: 14 KiB · 8 / 2.35 Gbps], against 100–250 µs of α (cost model §3.7). BF16 certification matters mostly for prefill frames, which are MiB-scale.

#### 8.3 The deterministic option (M3)

ADR-0019 decision 8: a per-request or per-instance `deterministic` option promises identical tokens for a fixed seed on the same build, plan epoch and devices. It is implemented by opening devices with deterministic kernel variants (RFC-0040 §3.3), freezing expert residency for the request (no cache admissions while it runs, so each expert keeps its site), fixing `MissSplit` (§3.4 of RFC-0040), and the sampler's counter-based generator, which keys on (client seed, choice index) whenever the request supplies a seed, so a repeated request reproduces its draws (§4.2, ADR-0107). It is slower and labelled as such. It is not an approximate knob: output stays within the exact contract.

### 9. Requantization tools (optional; M4)

Requantizing produces a **different model**, never an approximate mode of the same one (ADR-0005 decision 1). It exists for cases the cost model identifies, such as an 8-bit hot set that lets a Kimi-K2-Thinking-class model fit beside a useful KV cache on a 24 GB card (cost model §3.9, §4b2).

- `syntropy quantize <model> --scheme <ggml type | fp8 | …> [--tensors <class or regex>] [--imatrix <file> | --calibrate <corpus>]` writes a **standard GGUF** for ggml types, or a **safetensors compressed-tensors** checkpoint for FP8 and NVFP4. No Syntropy-specific format is produced (ADR-0005 decision 6).
- Quantization itself uses ggml's quantization functions for ggml types, so results match the ecosystem's tools. Importance statistics (`--calibrate`) are collected by Syntropy's own executor as per-channel activation sums of squares over the corpus.
- The output gets its own `ModelId` (RFC-0041 §1) and a quantization-error report against the source (§5.6). The model picker labels it with what was changed ("hot set requantized to Q8_0 from BF16"). Because it is a different model, the exact contract applies to it against its own reference.
- Writing a GGUF sets `general.alignment = 4096`. That is legal and harmless (`backends-and-portability.md` §3.4). It does not replace packs, because experts inside a stacked tensor stay unaligned.

### 10. Approximate-mode knobs and labelling

`ExactnessPolicy` is a policy object owned by RFC-0054 (RFC-0001 §5.13). This section defines each knob's meaning, scope and label. ADR-0019 decision 6 lists the knobs. A knob is added only with a harness report of its measured KL and top-1 cost, and that report is published with the release that adds it.

**Per-model knobs only, never a named preset.** Exact output is the default, and a preset that bundles several knobs would make leaving it a one-word choice. Knobs are set one at a time, per model in its configuration or per request where the table allows; there is no bundle such as a "fast" mode that turns on several at once. Each knob's harness report is its `KnobCost` entry in the release's `ConformanceManifest` (RFC-0001 §5.15): per (knob, family, dtype class), the measured mean and p99 KL and top-1 disagreement against the exact path. RFC-0051's single informational line in `syntropy plan` and the dashboard preview reads the cost from `KnobCost` and the predicted speed gain from the solver; a knob with no `KnobCost` entry is never offered there, and cannot be enabled (`KnobUnavailable`).

```rust
pub enum ExactnessPolicy { Exact, Approximate(ApproxKnobs) }
pub struct ApproxKnobs {
    pub expert_skip: Option<ExpertSkip>,        // per sequence
    pub expert_defer: bool,                     // per sequence
    pub miss_low_precision: Option<QuantScheme>,// instance
    pub topk_reduce: Option<u8>,                // per sequence: experts used per token
    pub kv_lossy: Option<KvDType>,              // instance
    pub wire_compress: Option<WireDType>,       // instance
    pub approx_attention: Option<ApproxAttn>,   // instance
    pub spec_lenient: Option<f32>,              // per sequence: acceptance relaxation
    pub act_lowp: Option<ActQuant>,             // instance; proposed in M3 (below)
}
bitflags! { pub struct ApproxFlags: u16 {      // TokenEntry::approx (RFC-0001 §5.8)
    const EXPERT_SKIP = 1; const EXPERT_DEFER = 2; const MISS_LOW_PRECISION = 4; const TOPK_REDUCE = 8;
    const KV_LOSSY = 16; const WIRE_COMPRESS = 32; const APPROX_ATTENTION = 64; const SPEC_LENIENT = 128;
    const ACT_LOWP = 256;
}}
```

| Knob | What it changes | Scope | Set by |
|---|---|---|---|
| `expert_skip` | routed experts that miss every fast tier, ranked at or below a threshold rank, are not computed; remaining weights are renormalised or not, as the knob states | per sequence | request or model config |
| `expert_defer` | missed experts are computed one step late, overlapped with the next layer | per sequence | request or model config |
| `miss_low_precision` | misses use a lower-precision copy of the expert (a sibling quantization from the pack) | instance | model config |
| `topk_reduce` | fewer routed experts per token than the model specifies | per sequence | request or model config |
| `kv_lossy` | KV stored below the model's KV dtype (§4.2), e.g. Q8 or 4-bit | instance | model config |
| `wire_compress` | activations below the family's certified wire dtype (§8.2) | instance | model config |
| `approx_attention` | sparsity or windowing not in the model's definition | instance | model config |
| `spec_lenient` | speculative acceptance that is not exact | per sequence | request |
| `act_lowp` | matrix-product activation quantisation not permitted by §4.3 rules (a) or (b) | instance | model config; added in M3 with its harness report |

Instance-wide knobs change the plan, and ingress never routes an `exact` request to an instance whose plan is approximate (RFC-0001 §5.8). Per-sequence knobs ride in `SequenceOpen::exactness`. Every token produced while a knob was active carries its flag in `TokenEntry::approx`. RFC-0050 and RFC-0051 put the flags in every response, in the CLI's start banner and as dashboard badges (ADR-0019 decision 7).

**A fallback never turns a knob on** (RFC-0001 §10.2). A path that would change output beyond tolerance is approximate by measurement, whatever it was meant to be (ADR-0019 decision 4), and is disabled in exact mode.

### 11. Configuration and error codes

| Key | Default | Meaning |
|---|---|---|
| `numerics.tolerances` | the release's `tolerances-v<N>.toml` | tolerance tables; overriding is a developer option that marks the build unofficial |
| `numerics.wire_bf16` | `certified` | `certified` follows the manifest's `WireCert`s; `never` forces F32 |
| `numerics.act_quant` | `certified` | `certified` follows the manifest's certificates for §4.3 rule (c); `strict` allows only rules (a) and (b) |
| `numerics.canary` | `true` | run the warm-up canary (§7) |

Codes are `(Domain::Numerics, n)`:

| Code | Name | Class | Raised when |
|---|---|---|---|
| 1 | `ToleranceMissing` | `Internal` | a supported dtype class has no published ceiling (release builds refuse to start a model of that class) |
| 2 | `CanaryFailed` | `Correctness` | the warm-up canary fails (§7); the runner does not reach `Ready` |
| 3 | `NanInLogits` | `Correctness` | NaN or infinity in logits at the tail; the sequence fails |
| 4 | `ExactRequestApproxInstance` | `InvalidRequest` | an `exact` request names an instance or model configured approximate; the message lists the active knobs |
| 5 | `KnobUnavailable` | `Unsupported` | a knob requested on a model or device that cannot honour it |
| 6 | `SchemeNoKernel` | `Unsupported` | no kernel or exact widened layout for a scheme on a device (RFC-0040 §11) |

## Performance model and budgets

All figures are **estimates** or **targets** with derivations; none is a Syntropy measurement.

**Widening does not slow decode.** Decode reads each weight once per token and is memory-bound (cost model §3.1), so dequantisation in registers is hidden while its arithmetic per byte stays below the device's ALU-to-bandwidth ratio. On GPUs that ratio is large. On CPUs it is not: integer-coded i-quant kernels are arithmetic-bound at about 5 GB/s per core in one surveyed engine (`tiered-memory-inference.md` §14). That is why the CPU provider reports a rate per scheme (RFC-0040 §9 row 5) and the solver plans with it.

**F32 accumulation costs prefill throughput on some GPUs.** The Ada whitepaper lists the RTX 4090's FP16 tensor throughput as 165.2 TFLOPS with FP32 accumulation against 330.3 with FP16 accumulation (cost model §2.1, [S10]). Exact mode uses the former. The cost model's prefill example (§4a) already assumes 100–165 TFLOPS, so SC-18's TTFT targets are consistent with this rule.

**W8A16 versus W8A8 in prefill.** For FP8 checkpoints that do not declare activation quantisation, exact mode's default W8A16 path runs prefill on BF16 tensor cores rather than FP8 ones, giving up the FP8 throughput multiple from the vendor tables until the family's W8A8 path is certified (§4.3 rule c). Checkpoints that declare dynamic FP8 activations (DeepSeek-V3-class) use W8A8 under rule (a) from M1. Decode is unaffected.

**Wire precision.** F32 frames are twice BF16's size: a 4096-token prefill at hidden 7168 is ≈ 117 MB in F32 against ≈ 59 MB in BF16, ≈ 0.40 s against ≈ 0.20 s at 2.35 Gbps goodput [estimate]. Chunked prefill hides this behind compute when a chunk's compute time exceeds its transfer time (cost model §3.7), and the solver checks that per hop. Decode adds ≈ 49 µs per hop at 2.5 GbE (§8.2).

**Harness cost.** A reference pass costs `2 · N_act` FLOPs per prefilled token plus the attention term of §6.2, and the `calibration` tier prefills ≈ 2.5e6 tokens, most of them in `long` documents, across all six categories (RFC-0060 §5.2). On a CPU at 100–200 GFLOPS that is 18–36 h for a 1B dense model and 78–156 h for a 30B-class MoE with 3B active, per pass [estimate, §6.2's worked figures]. That is why calibration runs its reference on a GPU (§6.2). The earlier figure here, ≈ 10 h for a 7B model, counted 5 categories × 50 000 short positions and no `long` prefill; it is withdrawn. Per-change CI scores 4096 positions per category on small fixtures with the CPU reference (§5.5). Large models use layer-sampled mode. The model-definition oracle (§7.1) costs one F32 reference pass over its golden prompts per family change.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Byte-identical output through fixed-order kernels everywhere | Rules out tensor-core accumulation, split-K and CPU/GPU splits (ADR-0019 option 1) |
| Absolute per-class ceilings only | Simpler, but spreads are expected to differ between families of one class (attention kinds, depth, vocabulary size); relative bounds on calibration models catch regressions that a ceiling set by the loosest family would let through |
| Fold every scale into FP16 elements | Inexact for MX power-of-two scales (range) and for BF16 group scales on INT4 (12 significant bits); §3.2's rule costs one multiply per block |
| Treat 8-bit activation blocks as approximate | The calibration baseline itself computes this way, and banning it would force slower F32-activation integer kernels on CPUs for no measurable fidelity gain |
| W8A8 FP8 by default for every FP8 checkpoint | Changes results for checkpoints that do not declare activation quantisation; allowed only by declaration or certification (ADR-0421) |
| F64 reference | Twice the cost or more; the F32 reference with a fixed order plus an F64 diagnostic mode gives the same confidence |
| KL over the top-k tokens only, as the gate's definition | It lower-bounds the true KL. Full-vocabulary KL is the definition; ADR-0601's top-256 `KL_K` is the stored-reference implementation, accepted because M0 must show it within 1 % of the full KL |
| Separate classes for 5- and 6-bit types now | No calibration data yet to justify a split (Open questions) |

## Testing and validation

| Claim | Verification |
|---|---|
| GGUF dequantisation matches ggml bit for bit | per type, random blocks plus edge cases (zero, maximum and NaN scales) against ggml's `dequantize_row_*` at the pinned commit |
| FP4/FP8 widening is exact | exhaustive: all 16 × 256 (E2M1, E4M3) products checked exact in FP16 and BF16; all 256 E4M3 and E5M2 values widen to their F32 values; MX scale application checked exact over the full exponent range |
| compressed-tensors interpretation | golden tensors from real checkpoints (headers and small slices stored as fixtures) |
| Accumulation contract | a static check that each kernel's declared `Numerics` meets its op's rule; a dynamic detector with crafted inputs on which FP16 accumulation or flush-to-zero produces a gross error |
| Tolerances are reproducible | two calibration runs on the same build give identical tables (reference and ggml CPU are deterministic) |
| The floor and gate discriminate | a known-good BF16-input tensor-core GEMM passes on every class; a deliberately FP16-accumulating variant fails; op-level floors never reject a kernel that is bit-identical to the reference up to summation order |
| Canary catches real faults | fault-injected kernels (wrong layout, wrong scale, dropped expert) fail RFC-0060 §5.8's canary; across the Tier 1 fixture matrix, correct builds never fail it |
| Wire and activation certifications | per family and shape class, recorded with the harness report that justified each entry |
| Deterministic option | 100 repeated runs with the same seed, build, epoch and devices give identical tokens |
| Sampling keys | the `n` choices of an `n > 1` request with one seed draw different numbers (distinctness test over 10⁴ draws); two requests with the same client seed draw identical numbers, and two with ingress-generated seeds do not; replaying a sequence from a reported `rng_counter` (u64) reproduces its remaining draws; batching the same sequences differently does not change any draw |
| Baselines apply declared activation quantisation | on a checkpoint declaring dynamic FP8 activations, the baseline's quantised activations equal the reference's bit for bit; a W8A16 kernel on that declared-W8A8 op fails `tol_op` (negative test) |
| Transcriptions are exact | per transcribable scheme (§3.5), the source and transcribed tensors dequantise to identical bits on random tensors and on fixture slices; a tensor that fails the census check is never transcribed |
| Bit-exact list per provider | the bit-exact-class cases run against `kernels::reference` on each provider each build, including every GGUF k-quant and i-quant dequantiser, FP8 block and channel scales, E8M0 scales at e = 0 (a subnormal scale) and NVFP4's global scale; the result must equal §6.4's table, and a change is a review item |
| Every knob is labelled | API-level tests: a request with a knob active gets the flag on every affected token and in the response metadata |
| The oracle catches module bugs | each module-level planted defect (§7.1) passes every parity gate and fails the model-definition oracle; the unmutated module passes within `τ_def`; a change to a shared block reruns the oracle for every family that uses it |
| The GPU reference equals the CPU reference | on the `nano` and `micro` fixtures of every v1 family, logits, tapped hidden states and stored reference blocks from the GPU `F32Fixed` reference are bit-identical to the CPU reference's |
| Calibration models fit the budget | each calibration model's recorded reference and baseline costs, attention included, are within the 48 h rule at the measured rates; a MoE ceiling without a directly calibrated MoE model carries the `provisional` flag |
| Positions beyond `max_ctx` are labelled | an API test with a context past the model's `max_ctx` is labelled beyond the certified context from the first position past it (`beyond_certified_from`) and `exact` before it; a family advertising more than 128K has a sampled-layer `long` result up to its advertised maximum in its certificate |
| No knob without a cost | a knob with no `KnobCost` entry cannot be enabled and never appears in the plan's informational line |

## Rollout / milestones

| Milestone | Delivered |
|---|---|
| **M0** | §1–§2 for GGUF types and plain floats; §4 rules enforced through `Numerics`; the reference executor on CPU, and on one GPU for calibration, cross-checked bit for bit; the model-definition oracle (§7.1) with golden data for `llama` and `qwen_moe`, gating their `FamilyCert`s, and the module-level planted defects; op, layer and end-to-end metrics; calibration over at least three models of `llama` and `qwen_moe` within the recomputed budget, and provisional ceilings for `Float16`, `Int8` and `Bits4`, with MoE ceilings flagged `provisional` until a MoE model is calibrated directly; the warm-up canary; F32 wire |
| **M1** | FP8, MXFP4, NVFP4 and compressed-tensors schemes with exact widening; §3.5's transcriptions and their census check; W8A8 for declared FP8 activations; final ceilings for the M0 classes over at least three families; `Fp8` and `Sub4` ceilings, `Fp8` with §6.2's `widen.f32` baseline and `φ = F_m`; declared activation quantisation in every baseline; a MoE checkpoint in the INT4 group-32 CPU-expert scheme among the calibration models, within §6.2's budget (the 1T-class model gated by RFC-0060 §5.7); layer-sampled reference mode; `PathCert`s for the 8-bit-activation kernels of non-transcribable schemes on the M1 roster; oracle golden data for every M1 family; `max_ctx` in every `FamilyCert`, with the beyond-the-certified-context label past it |
| **M2** | BF16 wire certification per family for pipelines; the placement matrix of §5.4 in nightly runs |
| **M3** | approximate knobs with harness reports (`expert_skip`, `topk_reduce`, `kv_lossy`, `wire_compress` first; then the rest as their reports land), each a per-model knob with its `KnobCost` entry and no named preset; the deterministic option; the `act_lowp` proposal with its report; sampled-layer `long` slices up to the advertised maximum for the families that advertise more than 128K |
| **M4** | requantization tools (§9); ceilings for every class on every Tier 1 platform; automated recalibration on pin bumps |

## Open questions

1. **Should 5- and 6-bit types form their own class?** Default: joined with 4-bit. Split if M0 calibration finds Q6_K's spread below half of Q4_K's, so that 6-bit kernels are not judged by a 4-bit tolerance.
2. **Canary sensitivity.** RFC-0060 §5.8's canary checks top-1 above `τ_tie` and the mean `KL_8` against `C_canary(d)`. Whether that catches the fault-injected kernels of the testing table without false positives on correct builds is settled by the M0 fixture matrix.
3. **Which families pass BF16 on the wire.** The research found one runtime failing with F16 and Q8 activations in production while another succeeded with Q8_0 (`multi-device-inference.md` §6). Settled per family in M2; the default stays F32.
4. **F16 KV for GGUF sources whose models were trained in BF16.** F16's range can overflow on families with large activations. Default: F16 for GGUF sources; a family module can require BF16 KV (RFC-0043 §5). Settled by M0/M1 calibration.
5. **FP8 checkpoints without declared activation schemes** (some GLM- and MiniMax-class FP8 configs). Default: W8A16. Reading each v1 FP8 checkpoint's quantisation config is an M0 desk check (COMP-E13-T09), so the list of families needing W8A8 certification exists before M1's FP8 work starts; certifying W8A8 per family where it passes stays M1. Kimi-K3-class is already known to declare no activation quantisation (RFC-0043 §5.5).
6. ~~Reference cost on 1T-class models~~ **Resolved:** 1T-class models are never calibration models (§6.2's budget rule). They are judged against the class ceilings by RFC-0060 §5.7, whose layer-sampled runs stay inside its 48 h rule.
7. ~~Op-level inputs~~ **Answered (2026-10-20; OQ-FID-13):** keep both kinds of input. Op cases use random inputs scaled by recorded activation statistics and replay recorded real activations from fixture models, because random inputs may miss real outliers.

## Proposed amendments to RFC-0001

1. **§5.4, `HopPlan::wire` comment.** It reads "F32, or BF16 where the reference itself is BF16". The reference holds the residual stream in F32 (ADR-0019 decision 1; §7), so that wording would never allow BF16, while ADR-0019 decision 5 allows BF16 per family after the harness passes. Proposed: "F32, or BF16 where the harness certified it for the family and shape class (ADR-0019 decision 5; RFC-0042 §8.2)".
2. **§5.1, additions** (listed for readers): `DType::{F8E8M0, F4E2M1, F6E2M3, F6E3M2, I16, I64, F64}`, `ScaleGranularity::Group`, the `CtScheme` variants, and `ActQuant`.
3. **§5.8, `TokenEntry::approx`.** `ApproxFlags` is defined in §10, including `ACT_LOWP` once that knob is added with its report.
