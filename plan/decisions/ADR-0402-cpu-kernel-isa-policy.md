# ADR-0402: CPU kernels: `std::arch` where stable, `asm!` for AMX, an assembly unit for SME2, no SVE2 in v1

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0001 (its note on AMX, SVE2 and SME2), ADR-0014, ADR-0020; RFC-0040 §7.3

## Context

ADR-0001 chose Rust with `std::arch` intrinsics for CPU kernels. It asked RFC-0040 to record, before M1, the stable status of each intrinsic family, and allowed stable `asm!` or a small statically linked C or assembly unit where intrinsics are missing.

Status on stable Rust, checked 2026-10-19 against the Rust 1.99.1 standard library documentation, the Rust Reference's `target_feature` list and the rust-lang tracking issues:

| Family | Stable target feature | Stable intrinsics |
|---|---|---|
| AVX2, FMA, F16C | yes | yes |
| AVX-512 (F, BW, VL, DQ, VNNI, BF16, FP16, …) | yes, since 1.89.0 | yes, since 1.89.0 (`_mm512_dpbusd_epi32`, `_mm512_dpbf16_ps`) |
| AVX-VNNI | yes, since 1.89.0 | yes, since 1.89.0 (`_mm256_dpbusd_avx_epi32`) |
| AMX | no (`x86_amx_intrinsics` gate) | no; tracking issue rust-lang/rust#126622 open, no stabilisation PR |
| NEON incl. dotprod, I8MM, BF16, FP16 | yes | yes |
| SVE, SVE2 | yes (listed in the Reference) | no; behind `stdarch_aarch64_sve`, tracking issue rust-lang/rust#145052 open, RFC rust-lang/rfcs#3838 not accepted |
| SME, SME2 | no (absent from the Reference's list) | no; runtime detection strings `sme`/`sme2` exist in the stable detection macro |

Other facts:

- AMX tile state needs OS support: XCR0 bits 17–18, and on Linux a per-process `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` request.
- SME2 work runs in streaming mode (`smstart`/`smstop`), which changes the vector length and the register state that compiled Rust code assumes.
- The Arm platforms in the support matrix are Apple silicon (no non-streaming SVE), Raspberry Pi 5-class boards (no SVE) and GB10-class systems with Arm-designed cores whose SVE vectors are 128 bits, the same width as NEON.
- The standard library documents most AArch64 runtime detection as Linux-only; macOS exposes `hw.optional.arm.FEAT_*` sysctls.
- Release builds never use a nightly compiler (ADR-0001 decision 1).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Nightly Rust for AMX and SVE intrinsics | Intrinsics with type checking | Violates ADR-0001; nightly APIs change |
| B. C intrinsics files for AMX, SVE2 and SME2 | Mature compiler support (ACLE, `immintrin.h`) | A C toolchain in every CPU build; a second language for the same kernel family |
| **C. AMX microkernels as single stable `asm!` blocks; SME2 microkernels as leaf routines in a statically linked AArch64 assembly unit; no SVE2 kernels; everything else in `std::arch`** | Stable toolchain only, except the platform assembler on AArch64; no Rust code ever runs in streaming mode or holds tile state across a block | Hand-written assembly for two kernel families; no compiler help with register allocation inside them |
| D. Skip AMX and SME2 entirely | Least code | Gives up the compute-bound CPU paths (prefill and verify on CPU stages) that ADR-0014 lists as current features |

## Decision

**Option C.**

1. **`std::arch` intrinsics** for AVX2/FMA, AVX-512 (including VNNI, BF16 and FP16), AVX-VNNI and NEON (dotprod, I8MM, BF16, FP16), in `#[target_feature]` functions selected once at `open`.
2. **AMX** microkernels are single `asm!` blocks that configure tiles, load, multiply, store and release within the block. Tile registers are declared clobbered and never live across a block boundary. Detection is CPUID plus XCR0 plus, on Linux, a successful permission request; refusal falls back to AVX-512.
3. **SME2** microkernels are complete leaf functions in `kernels-cpu/src/sme2/*.S`, assembled by the platform assembler through the `cc` crate and called through `extern "C"`. Each enters and leaves streaming mode itself and preserves the AAPCS64 callee-saved registers. Detection uses `is_aarch64_feature_detected!("sme2")` on Linux and `hw.optional.arm.FEAT_SME2` on macOS.

   *Revised 2026-10-22:* a clarification of the detection mechanism, not a change to this decision. On the pinned toolchain (1.99.1, `rust-toolchain.toml`) the macro's `sme` and `sme2` names are unstable (E0658), so on Linux and Android the provider reads `AT_HWCAP2` through `getauxval` and tests `HWCAP2_SME2` (bit 37), as ADR-0014's SME2 row names; macOS still reads `hw.optional.arm.FEAT_SME2`. Both give the same answer as the macro, so the provider switches to the macro once it is stable (item 5's re-check). RFC-0040 §7.3 and RFC-0001 §18.14 record it.

4. **No SVE2 kernels in v1.** NEON with I8MM covers the 128-bit SVE targets in the matrix.
5. Each release re-checks this table. A family whose intrinsics stabilise moves to `std::arch` at the next convenient change.

## Consequences

- Positive: stable toolchain for every release build; CPU kernels for every ISA the support matrix names, with the detection rules written down.
- Positive: streaming-mode and tile-state hazards are confined to blocks and routines that own their state completely.
- Negative: AMX and SME2 kernels are assembly, reviewed and tested as such (bit-exact against the `std::arch` VNNI/I8MM kernels on integer paths, RFC-0060).
- Negative: if a future Arm target ships SVE vectors wider than 128 bits, v1 leaves that width unused.

## What would make us revisit

- AMX or SME intrinsics stabilise. Rewrite the assembly in `std::arch`.
- A Tier 1 or Tier 2 Arm target ships SVE vectors wider than 128 bits, or the RFC-0061 lab shows SVE2 beating NEON I8MM by more than 10 % on GEMV at equal width.
- An AMX or SME2 kernel shows a correctness bug that the assembly form made hard to find, twice in one release cycle. Consider a C intrinsics unit for that family.
