# ggml activation quantizer vectors

`ggml-q8.txt` holds golden vectors for the activation quantisation references in
`syntropy_kernels::reference::act_quant` (RFC-0042 §4.3, COMP-E13-T03). `tests/ggml_act_quant.rs`
checks the references against it bit for bit.

## Provenance

- Generator: `gen_ggml_q8.c` in this directory, Syntropy's own code. It makes its inputs
  itself, calls ggml's quantizers on them, and prints inputs and outputs as hex. No ggml source
  is copied into this repository, and ggml is not vendored or linked by any crate.
- ggml: <https://github.com/ggml-org/ggml> at commit
  `da8e2a54b919557b816b1c2beb432ccb766f7ce0` (2026-10-20), MIT licence. The repository has no
  `third_party/ggml.lock` pin yet (COMP-E7-T01). When it has one, regenerate the vectors at that
  commit.
- Host: arm64, Apple M4 Pro, macOS, Apple clang 21.0.0. The `cpu` lines are therefore the CPU
  backend's arm64 NEON path, and the file says so (`host arm64-neon`).
- Output: `ggml-q8.txt`, SHA-256
  `3bf3b67b1beb5955b82a39947c1fd6521c6dc24c94ff7c334bdb92df1d8239ee`.

Each case records:

| Line | ggml function |
|---|---|
| `q8_0 ref`, `q8_1 ref`, `q8_K ref` | `quantize_row_q8_0_ref`, `quantize_row_q8_1_ref`, `quantize_row_q8_K_ref` (`src/ggml-quants.c`) |
| `q8_0 cpu`, `q8_1 cpu`, `q8_K cpu` | the CPU backend's `quantize_row_q8_0`, `quantize_row_q8_1`, `quantize_row_q8_K` for the host ISA (`src/ggml-cpu/arch/<isa>/quants.c`): the quantizers its matrix products run on their activations |
| `f16 ref` | `ggml_fp32_to_fp16` on each element |
| `bf16 ref`, `bf16 cpu` | `ggml_fp32_to_bf16_row_ref`, `ggml_fp32_to_bf16_row` |

The cases cover zeros of both signs, exact rounding ties at three power-of-two scales, near ties
under an inexact scale, F32's largest finite value and the F16 overflow boundary of the scale,
scales in F16's subnormal range and in F32's, F32 subnormal elements, the sign and tie rules of
`Q8_K`'s maximum, constant rows, four random rows (uniform, per-block scales over twelve
decades, outlier channels, heavy tails), and, for F16 and BF16 only, NaNs, infinities and their
rounding boundaries. Q8 inputs on which ggml's C is undefined (a non-finite element, or an
inverse scale that overflows F32) are not recorded; the module documentation says what the
reference does with them.

## Regenerating

The inputs are downloaded third-party code: keep them in their own empty directory, outside the
repository, and build with explicit paths. ggml needs no configure step for these files.

```sh
SRC=/path/to/empty/dir          # the ggml checkout
OUT=/path/to/another/empty/dir  # objects and the generator binary
PIN=da8e2a54b919557b816b1c2beb432ccb766f7ce0
HERE=crates/kernels/tests/data/act_quant
ISA=arm                          # arm or x86: the CPU backend's arch directory

git clone https://github.com/ggml-org/ggml "$SRC" && git -C "$SRC" checkout "$PIN"
mkdir -p "$OUT/inc"
printf '#pragma once\n#define GGML_VERSION "pinned"\n#define GGML_COMMIT "%s"\n' "$PIN" \
  > "$OUT/inc/ggml-version.h"
FLAGS="-O2 -std=c11 -ffp-contract=off -I$OUT/inc -I$SRC/include -I$SRC/src -I$SRC/src/ggml-cpu"
cc $FLAGS -c "$SRC/src/ggml.c" -o "$OUT/ggml.o"
cc $FLAGS -c "$SRC/src/ggml-quants.c" -o "$OUT/quants.o"
cc $FLAGS -c "$SRC/src/ggml-cpu/quants.c" -o "$OUT/cpu-quants.o"
cc $FLAGS -c "$SRC/src/ggml-cpu/arch/$ISA/quants.c" -o "$OUT/arch-quants.o"
cc $FLAGS -DGGML_PIN="\"$PIN\"" -c "$HERE/gen_ggml_q8.c" -o "$OUT/gen.o"
cc -Wl,-dead_strip "$OUT"/gen.o "$OUT"/ggml.o "$OUT"/quants.o "$OUT"/cpu-quants.o \
  "$OUT"/arch-quants.o -o "$OUT/gen_ggml_q8"     # on Linux: -Wl,--gc-sections
"$OUT/gen_ggml_q8" > "$HERE/ggml-q8.txt"
```

`-ffp-contract=off` matches the core's ggml build (RFC-0040 §8.1). On an x86 host with AVX2,
add `-mavx2 -mf16c` to `FLAGS` and set `ISA=x86`; the file then records `host x86-avx`, and the
test checks the `cpu` lines against `Q8Path::X86Avx`. No x86 vectors are committed yet.
