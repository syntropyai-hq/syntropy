// SPDX-License-Identifier: MIT OR Apache-2.0
//
// SME2 outer-product GEMM tiles (RFC-0040 §7.3; ADR-0402 item 3).
//
// Every routine here is a complete leaf function with the AAPCS64 interface of a
// non-streaming function with no ZA state: it is called with PSTATE.SM = 0 and PSTATE.ZA = 0,
// enters streaming mode and enables ZA with `smstart`, runs its outer-product loop on the ZA
// tiles, and leaves both with `smstop` before it returns. No Rust code ever runs in streaming
// mode.
//
// Callee-saved registers: entering and leaving streaming mode sets every Z register to zero,
// and d8-d15 are the low halves of z8-z15, so each routine saves d8-d15 before `smstart` and
// restores them after `smstop`. The routines use none of x19-x29 and do not touch x30 or the
// frame pointer.
//
// ZA lazy saves: a caller with live ZA state could have set up a lazy save (TPIDR2_EL0 not
// zero). Committing it means calling `__arm_tpidr2_save`, which a leaf routine cannot do.
// Rust callers never hold ZA state, so TPIDR2_EL0 is always zero here; if it is not, the
// routine returns 1 without touching memory or ZA. Otherwise it returns 0.
//
// The tile is up to 2 x SVL_s rows by 2 x SVL_s columns, where SVL_s is the streaming vector
// length in 32-bit lanes (16 on a 512-bit implementation), held in ZA tiles 0-3:
//
//     za0: rows [0, SVL_s),        columns [0, SVL_s)
//     za1: rows [0, SVL_s),        columns [SVL_s, 2 SVL_s)
//     za2: rows [SVL_s, 2 SVL_s),  columns [0, SVL_s)
//     za3: rows [SVL_s, 2 SVL_s),  columns [SVL_s, 2 SVL_s)
//
// Edges are handled with predicates, so the routines never read or write outside the
// elements their arguments describe. The caller checks m, n <= 2 SVL_s and the slice bounds.

#include "common.h"

// uint64_t syn_sme2_svl_bytes(void)
//
// The streaming vector length in bytes. RDSVL is legal outside streaming mode.
FUNC_BEGIN SYM(syn_sme2_svl_bytes)
    rdsvl   x0, #1
    ret
FUNC_END SYM(syn_sme2_svl_bytes)

// Saves d8-d15 and checks TPIDR2_EL0; branches to `busy` when a lazy save is pending.
.macro ENTER busy
    mrs     x16, TPIDR2_EL0
    cbnz    x16, \busy
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    smstart
.endm

// Leaves streaming mode with ZA off, restores d8-d15 and returns 0.
.macro LEAVE
    smstop
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    mov     x0, #0
    ret
.endm

// Stores rows [0, m) of the result: za0/za1 for the first SVL_s rows, za2/za3 for the rest.
//   x2 = c, x4 = m, x7 = ldc (elements), x9 = SVL_s, p2/p3 = the column predicates.
// Clobbers x10-x12.
.macro STORE_ROWS
    mov     x10, x2
    cmp     x4, x9
    csel    x11, x4, x9, lo         // rows in the top half: min(m, SVL_s)
    mov     x12, #0
1:
    cmp     x12, x11
    b.hs    2f
    st1w    {za0h.s[w12, 0]}, p2, [x10]
    st1w    {za1h.s[w12, 0]}, p3, [x10, x9, lsl #2]
    add     x10, x10, x7, lsl #2
    add     x12, x12, #1
    b       1b
2:
    subs    x11, x4, x9             // rows in the bottom half: m - SVL_s
    b.ls    4f
    mov     x12, #0
3:
    st1w    {za2h.s[w12, 0]}, p2, [x10]
    st1w    {za3h.s[w12, 0]}, p3, [x10, x9, lsl #2]
    add     x10, x10, x7, lsl #2
    add     x12, x12, #1
    cmp     x12, x11
    b.lo    3b
4:
.endm

// uint64_t syn_sme2_f32_tile(const float *a, const float *b, float *c, uint64_t k,
//                            uint64_t m, uint64_t n, uint64_t ldb, uint64_t ldc)
//
// C[i][j] = sum over p in [0, k) of A[p][i] * B[p][j], in increasing p, for i < m, j < n.
//
//   a:   the A panel, k rows of m floats: a[p * m + i] (A transposed and packed).
//   b:   k rows of B with row stride ldb floats: b[p * ldb + j].
//   c:   the result, m rows with row stride ldc floats; overwritten, not accumulated.
//
// Each step is one FMOPA per ZA tile: a fused multiply-add of the rank-1 update into every
// element, one rounding per step, in the same order as a scalar loop of fused multiply-adds.
FUNC_BEGIN SYM(syn_sme2_f32_tile)
    ENTER   9f
    cntw    x9                      // SVL_s, in streaming mode
    whilelt p0.s, xzr, x4           // rows [0, SVL_s)
    whilelt p1.s, x9, x4            // rows [SVL_s, 2 SVL_s)
    whilelt p2.s, xzr, x5           // columns [0, SVL_s)
    whilelt p3.s, x9, x5            // columns [SVL_s, 2 SVL_s)
    zero    {za}
    lsl     x13, x4, #2             // A panel row stride in bytes
    lsl     x14, x6, #2             // B row stride in bytes
    cbz     x3, 6f
5:
    ld1w    {z0.s}, p0/z, [x0]
    ld1w    {z1.s}, p1/z, [x0, x9, lsl #2]
    ld1w    {z2.s}, p2/z, [x1]
    ld1w    {z3.s}, p3/z, [x1, x9, lsl #2]
    fmopa   za0.s, p0/m, p2/m, z0.s, z2.s
    fmopa   za1.s, p0/m, p3/m, z0.s, z3.s
    fmopa   za2.s, p1/m, p2/m, z1.s, z2.s
    fmopa   za3.s, p1/m, p3/m, z1.s, z3.s
    add     x0, x0, x13
    add     x1, x1, x14
    subs    x3, x3, #1
    b.ne    5b
6:
    STORE_ROWS
    LEAVE
9:
    mov     x0, #1
    ret
FUNC_END SYM(syn_sme2_f32_tile)

// uint64_t syn_sme2_i8_tile(const int8_t *a, const int8_t *b, int32_t *c, uint64_t groups,
//                           uint64_t m, uint64_t n, uint64_t ldb, uint64_t ldc)
//
// C[i][j] = sum over p in [0, 4 groups) of A[p][i] * B[p][j], exact in 32-bit integers
// (the caller bounds k so no sum overflows), for i < m, j < n.
//
//   a:   the A panel, one group of 4 values of k per step: a[g * 4m + 4i + q] = A[4g + q][i].
//   b:   the B panel, the same interleave with stride ldb bytes: b[g * ldb + 4j + q].
//   c:   the result, m rows with row stride ldc int32s; overwritten, not accumulated.
//
// Each step is one four-way widening SMOPA per ZA tile.
FUNC_BEGIN SYM(syn_sme2_i8_tile)
    ENTER   9f
    cntw    x9                      // SVL_s
    cntb    x15                     // SVL_b = 4 SVL_s
    lsl     x10, x4, #2             // 4m: bytes of A per group
    lsl     x11, x5, #2             // 4n: bytes of B per group
    whilelt p0.b, xzr, x10          // rows [0, SVL_s), four bytes each
    whilelt p1.b, x15, x10          // rows [SVL_s, 2 SVL_s)
    whilelt p4.b, xzr, x11          // columns [0, SVL_s)
    whilelt p5.b, x15, x11          // columns [SVL_s, 2 SVL_s)
    whilelt p2.s, xzr, x5           // the same columns, by 32-bit result element
    whilelt p3.s, x9, x5
    zero    {za}
    cbz     x3, 6f
5:
    ld1b    {z0.b}, p0/z, [x0]
    ld1b    {z1.b}, p1/z, [x0, x15]
    ld1b    {z2.b}, p4/z, [x1]
    ld1b    {z3.b}, p5/z, [x1, x15]
    smopa   za0.s, p0/m, p4/m, z0.b, z2.b
    smopa   za1.s, p0/m, p5/m, z0.b, z3.b
    smopa   za2.s, p1/m, p4/m, z1.b, z2.b
    smopa   za3.s, p1/m, p5/m, z1.b, z3.b
    add     x0, x0, x10
    add     x1, x1, x6
    subs    x3, x3, #1
    b.ne    5b
6:
    STORE_ROWS
    LEAVE
9:
    mov     x0, #1
    ret
FUNC_END SYM(syn_sme2_i8_tile)
