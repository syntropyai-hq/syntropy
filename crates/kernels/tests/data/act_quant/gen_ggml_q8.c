// SPDX-License-Identifier: MIT OR Apache-2.0

// Writes `ggml-q8.txt`, the golden vectors `syntropy-kernels` checks its activation quantisation
// references against (RFC-0042 §4.3; COMP-E13-T03). It is not part of any build: it links
// against ggml sources at a pinned commit, outside this repository, and its output is the only
// thing committed. `README.md` beside it gives the pinned commit and the exact commands.
//
// For every case it prints the input row as F32 bits, then each ggml quantizer's output bytes,
// in ggml's own block layouts:
// - `q8_0`, `q8_1`, `q8_K`: `quantize_row_q8_{0,1,K}_ref` from `ggml-quants.c` (`ref`), and the
//   CPU backend's `quantize_row_q8_{0,1,K}` for the host ISA (`cpu`), which is the path ggml's
//   matrix products take for their activations;
// - `f16`: `ggml_fp32_to_fp16` on each element;
// - `bf16`: `ggml_fp32_to_bf16_row_ref` (`ref`) and `ggml_fp32_to_bf16_row` (`cpu`).
//
// A case lists the quantizers it applies to. Inputs on which ggml's quantizers are undefined in
// C (a NaN or an infinity in a Q8 block, or a block whose `1/d` overflows F32, where a float to
// integer conversion is out of range) are not recorded for the Q8 types.

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"

#ifndef GGML_PIN
#error "build with -DGGML_PIN=\"<ggml commit>\""
#endif

#if defined(__ARM_NEON)
#define HOST_PATH "arm64-neon"
#elif defined(__AVX__) || defined(__AVX2__)
#define HOST_PATH "x86-avx"
#else
#define HOST_PATH "reference"
#endif

#define ROW_MAX 1024

static float row[ROW_MAX];
static int row_len;

static void put_hex(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) {
        printf("%02x", b[i]);
    }
    printf("\n");
}

static void put_input(void) {
    printf("in ");
    for (int i = 0; i < row_len; i++) {
        uint32_t u;
        memcpy(&u, &row[i], 4);
        printf("%08x", u);
    }
    printf("\n");
}

enum { Q8 = 1, BF16 = 2, F16 = 4, ALL = Q8 | BF16 | F16 };

static void emit(const char * name, int ops) {
    static block_q8_0 b0[ROW_MAX / QK8_0];
    static block_q8_1 b1[ROW_MAX / QK8_1];
    static block_q8_K bk[ROW_MAX / QK_K];
    static ggml_bf16_t h[ROW_MAX];
    static ggml_fp16_t f[ROW_MAX];

    printf("case %s %d\n", name, row_len);
    put_input();
    if (ops & Q8) {
        quantize_row_q8_0_ref(row, b0, row_len);
        printf("q8_0 ref ");
        put_hex(b0, sizeof(block_q8_0) * (row_len / QK8_0));
        memset(b0, 0, sizeof(b0));
        quantize_row_q8_0(row, b0, row_len);
        printf("q8_0 cpu ");
        put_hex(b0, sizeof(block_q8_0) * (row_len / QK8_0));

        quantize_row_q8_1_ref(row, b1, row_len);
        printf("q8_1 ref ");
        put_hex(b1, sizeof(block_q8_1) * (row_len / QK8_1));
        memset(b1, 0, sizeof(b1));
        quantize_row_q8_1(row, b1, row_len);
        printf("q8_1 cpu ");
        put_hex(b1, sizeof(block_q8_1) * (row_len / QK8_1));

        if (row_len % QK_K == 0) {
            quantize_row_q8_K_ref(row, bk, row_len);
            printf("q8_K ref ");
            put_hex(bk, sizeof(block_q8_K) * (row_len / QK_K));
            memset(bk, 0, sizeof(bk));
            quantize_row_q8_K(row, bk, row_len);
            printf("q8_K cpu ");
            put_hex(bk, sizeof(block_q8_K) * (row_len / QK_K));
        }
    }
    if (ops & F16) {
        for (int i = 0; i < row_len; i++) {
            f[i] = ggml_fp32_to_fp16(row[i]);
        }
        printf("f16 ref ");
        put_hex(f, 2 * (size_t) row_len);
    }
    if (ops & BF16) {
        ggml_fp32_to_bf16_row_ref(row, h, row_len);
        printf("bf16 ref ");
        put_hex(h, 2 * (size_t) row_len);
        ggml_fp32_to_bf16_row(row, h, row_len);
        printf("bf16 cpu ");
        put_hex(h, 2 * (size_t) row_len);
    }
    printf("end\n");
}

// splitmix64: the inputs are recorded as bits, so the generator only needs to be repeatable.
static uint64_t rng_state = 0x5EED0C0DE13703ULL;

static uint64_t next_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniform in [-1, 1).
static float uniform(void) {
    return (float) ((double) (next_u64() >> 40) * 0x1.0p-23 - 1.0);
}

static float bits(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void fill(int n, float v) {
    row_len = n;
    for (int i = 0; i < n; i++) {
        row[i] = v;
    }
}

// One block of 32 whose largest magnitude is `amax`, at element `at`, with the other elements
// spread over (-amax, amax).
static void block_with_amax(float * b, float amax, int at) {
    for (int j = 0; j < 32; j++) {
        b[j] = amax * uniform() * 0.999f;
    }
    b[at] = (at & 1) ? -amax : amax;
}

int main(void) {
    printf("# SPDX-License-Identifier: MIT OR Apache-2.0\n");
    printf("# Generated by gen_ggml_q8.c against ggml %s; see README.md. Do not edit.\n", GGML_PIN);
    printf("syntropy-ggml-q8-vectors 1\n");
    printf("ggml %s\n", GGML_PIN);
    // The CPU backend's Q8 path this build took: what the `cpu` lines record.
    printf("host %s\n", HOST_PATH);

    // Zeros, both signs, and a row of zeros with one non-zero element per Q8_K block.
    fill(256, 0.0f);
    emit("zeros", ALL);
    fill(256, -0.0f);
    emit("negative_zeros", ALL);
    fill(256, 0.0f);
    row[37] = 0.75f;
    emit("one_nonzero", ALL);
    fill(256, 0.0f);
    row[200] = -3.0f;
    emit("one_negative", ALL);

    // Rounding ties. With amax = 127 · 2^k the scale d = 2^k and 1/d = 2^-k are exact, so
    // x · id lands exactly on k + 0.5 and only the rounding rule decides.
    for (int s = 0; s < 3; s++) {
        const float scale = (s == 0) ? 1.0f : (s == 1) ? 0x1.0p-6f : 0x1.0p+9f;
        row_len = 256;
        for (int b = 0; b < 8; b++) {
            float * blk = row + 32 * b;
            for (int j = 0; j < 32; j++) {
                const int k = (j * 8 + b) - 128;  // -128 .. 127
                float t = (float) k + 0.5f;
                if (t > 126.5f) t = 126.5f;
                if (t < -126.5f) t = -126.5f;
                blk[j] = t * scale;
            }
            blk[b % 32] = ((b & 1) ? -127.0f : 127.0f) * scale;
        }
        char name[32];
        snprintf(name, sizeof name, "ties_scale_%d", s);
        emit(name, ALL);
    }

    // Near ties with an inexact scale: amax = 127 · 3, so d = 3 and 1/d is inexact, and
    // 1/d and 127/amax may round differently.
    row_len = 256;
    for (int i = 0; i < 256; i++) {
        row[i] = (float) ((i % 254) - 127) * 1.5f + 0.75f * (float) (i % 3);
    }
    for (int b = 0; b < 8; b++) {
        row[32 * b + 5] = 381.0f;
    }
    emit("near_ties_inexact_scale", ALL);

    // Largest magnitudes: F32's largest finite value, and the F16 overflow boundary of d.
    // d = amax / 127 is F16's largest finite value at amax = 65504 · 127, rounds down to it at
    // 65519 · 127, and rounds to infinity (a tie, to even) at 65520 · 127.
    {
        const float amaxes[8] = {
            FLT_MAX, -FLT_MAX, 65504.0f * 127.0f, 65519.0f * 127.0f,
            65520.0f * 127.0f, 65536.0f * 127.0f, 1.0e30f, 0x1.fffffep+100f,
        };
        row_len = 256;
        for (int b = 0; b < 8; b++) {
            block_with_amax(row + 32 * b, fabsf(amaxes[b]), (b * 7) % 32);
            if (amaxes[b] < 0) row[32 * b + (b * 7) % 32] = amaxes[b];
        }
        emit("max_magnitude", ALL);
    }

    // d in F16's subnormal range and below it: 2^-14 (smallest normal), 2^-15, 2^-24
    // (smallest subnormal), 2^-25 (a tie, to zero), 1.5 · 2^-25 (rounds to the smallest
    // subnormal), 3 · 2^-25 (a tie between 1 and 2 subnormal units, to 2), 2^-30, 2^-60.
    {
        const float ds[8] = {
            0x1.0p-14f, 0x1.0p-15f, 0x1.0p-24f, 0x1.0p-25f,
            0x1.8p-25f, 0x1.8p-24f, 0x1.0p-30f, 0x1.0p-60f,
        };
        row_len = 256;
        for (int b = 0; b < 8; b++) {
            block_with_amax(row + 32 * b, ds[b] * 127.0f, (b * 5 + 3) % 32);
        }
        emit("f16_subnormal_scale", ALL);
    }

    // d an F32 subnormal while 1/d stays finite (amax at or above about 3.73e-37), so every
    // quantizer is defined.
    {
        const float amaxes[8] = {1.0e-36f, 4.0e-37f, 5.0e-37f, 1.0e-35f,
                                 0x1.0p-120f, 0x1.0p-121f, 1.0e-30f, 1.0e-37f * 4.5f};
        row_len = 256;
        for (int b = 0; b < 8; b++) {
            block_with_amax(row + 32 * b, amaxes[b], (b * 3 + 1) % 32);
        }
        emit("f32_subnormal_scale", ALL);
    }

    // Subnormal elements in blocks with a normal amax.
    row_len = 256;
    for (int i = 0; i < 256; i++) {
        const uint32_t m = (uint32_t) (next_u64() & 0x7FFFFF);
        row[i] = bits(m | ((i & 1) ? 0x80000000u : 0));
    }
    for (int b = 0; b < 8; b++) {
        row[32 * b + 9] = (b & 1) ? -1.0f : 1.0f;
    }
    row[1] = bits(1);            // the smallest subnormal
    row[2] = bits(0x80000001u);  // its negative
    row[3] = bits(0x007FFFFF);   // the largest subnormal
    emit("subnormal_elements", ALL);

    // Q8_K: the element of largest magnitude sets the sign of the scale, and the first one
    // wins a tie in magnitude.
    row_len = 512;
    for (int i = 0; i < 512; i++) {
        row[i] = uniform() * 2.0f;
    }
    row[10] = -4.0f;
    row[11] = 4.0f;
    row[256 + 20] = 4.0f;
    row[256 + 21] = -4.0f;
    emit("q8k_sign_of_max", ALL);

    // Every element equal.
    fill(256, -0.3f);
    emit("constant_negative", ALL);
    fill(256, 1.0f / 3.0f);
    emit("constant_third", ALL);

    // Random rows: uniform; per-block scales over twelve decades; outlier channels;
    // heavy tails.
    for (int r = 0; r < 4; r++) {
        row_len = 1024;
        for (int i = 0; i < 1024; i++) {
            row[i] = uniform();
        }
        if (r == 1) {
            for (int b = 0; b < 32; b++) {
                const float scale = powf(10.0f, 12.0f * (uniform() * 0.5f + 0.5f) - 6.0f);
                for (int j = 0; j < 32; j++) row[32 * b + j] *= scale;
            }
        } else if (r == 2) {
            for (int i = 7; i < 1024; i += 64) row[i] *= 100.0f;
        } else if (r == 3) {
            for (int i = 0; i < 1024; i++) {
                const float u = fabsf(uniform()) + 0x1.0p-20f;
                row[i] = row[i] / (u * u);
            }
        }
        char name[32];
        snprintf(name, sizeof name, "random_%d", r);
        emit(name, ALL);
    }

    // BF16 and F16 only: values the Q8 quantizers leave undefined, and rounding boundaries.
    row_len = 32;
    {
        const uint32_t u[32] = {
            0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0xFF812345u,  // NaNs, quiet and signalling
            0x7F800000u, 0xFF800000u, 0x7F7FFFFFu, 0xFF7FFFFFu,  // infinities, +-FLT_MAX
            0x3F808000u, 0x3F818000u, 0x3F807FFFu, 0x3F808001u,  // BF16 ties and near ties
            0x00000001u, 0x80000001u, 0x007FFFFFu, 0x00008000u,  // F32 subnormals
            0x00018000u, 0x7F7F8000u, 0x7F7F7FFFu, 0x00000000u,  // BF16 subnormal tie, overflow tie
            0x80000000u, 0x477FE000u, 0x477FEFFFu, 0x477FF000u,  // -0, F16 max, near and at its tie
            0x33000000u, 0x33000001u, 0x33400000u, 0x387FC000u,  // F16 subnormal ties and edges
            0x38800000u, 0x3F800000u, 0x3FC00000u, 0x40490FDBu,
        };
        for (int i = 0; i < 32; i++) row[i] = bits(u[i]);
    }
    emit("special_values", BF16 | F16);

    return 0;
}
