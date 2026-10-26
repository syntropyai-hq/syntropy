// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The callee-saved register harness for the SME2 routines (COMP-E3-T07; ADR-0402 item 3).
//
// `syn_sme2_harness_call` fills every AAPCS64 callee-saved register a routine must preserve
// (x19-x29 and d8-d15) with a distinct pattern, calls the routine with eight integer
// arguments, and records what those registers, x0, SVCR (PSTATE.SM in bit 0, PSTATE.ZA in
// bit 1) and SP hold on return. The test in `src/sme2.rs` compares the record with the
// patterns. The harness restores its own callee-saved registers before it returns.
//
// `syn_sme2_harness_clobber` is a deliberately wrong routine: it enters and leaves streaming
// mode without saving d8-d15. The negative test calls it through the harness to show that
// the harness detects the loss.

#include "common.h"

// Pattern n: 0xA5A5_0000_nnnn_nnnn-like, distinct for each register the harness fills.
.macro PAT reg, n
    movz    \reg, #0xa5a5, lsl #48
    movk    \reg, #\n, lsl #16
    movk    \reg, #\n
.endm

// void syn_sme2_harness_call(uint64_t report[23], const void *routine, const uint64_t args[8])
//
// report[0..11)  = x19..x29 after the call
// report[11..19) = d8..d15 after the call
// report[19]     = x0 (the routine's return value)
// report[20]     = SVCR after the call
// report[21]     = SP before the call
// report[22]     = SP after the call
FUNC_BEGIN SYM(syn_sme2_harness_call)
    stp     x29, x30, [sp, #-176]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8, d9, [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]
    str     x0, [sp, #160]
    mov     x16, x1
    mov     x17, x2

    PAT     x19, 0x19
    PAT     x20, 0x20
    PAT     x21, 0x21
    PAT     x22, 0x22
    PAT     x23, 0x23
    PAT     x24, 0x24
    PAT     x25, 0x25
    PAT     x26, 0x26
    PAT     x27, 0x27
    PAT     x28, 0x28
    PAT     x29, 0x29
    PAT     x9, 0x108
    fmov    d8, x9
    PAT     x9, 0x109
    fmov    d9, x9
    PAT     x9, 0x110
    fmov    d10, x9
    PAT     x9, 0x111
    fmov    d11, x9
    PAT     x9, 0x112
    fmov    d12, x9
    PAT     x9, 0x113
    fmov    d13, x9
    PAT     x9, 0x114
    fmov    d14, x9
    PAT     x9, 0x115
    fmov    d15, x9

    mov     x9, sp
    str     x9, [sp, #168]
    ldp     x0, x1, [x17]
    ldp     x2, x3, [x17, #16]
    ldp     x4, x5, [x17, #32]
    ldp     x6, x7, [x17, #48]
    blr     x16

    mrs     x9, SVCR
    mov     x10, sp
    ldr     x11, [sp, #160]
    stp     x19, x20, [x11]
    stp     x21, x22, [x11, #16]
    stp     x23, x24, [x11, #32]
    stp     x25, x26, [x11, #48]
    stp     x27, x28, [x11, #64]
    str     x29, [x11, #80]
    str     d8, [x11, #88]
    str     d9, [x11, #96]
    str     d10, [x11, #104]
    str     d11, [x11, #112]
    str     d12, [x11, #120]
    str     d13, [x11, #128]
    str     d14, [x11, #136]
    str     d15, [x11, #144]
    str     x0, [x11, #152]
    str     x9, [x11, #160]
    ldr     x12, [sp, #168]
    str     x12, [x11, #168]
    str     x10, [x11, #176]

    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     d8, d9, [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    ldp     x29, x30, [sp], #176
    ret
FUNC_END SYM(syn_sme2_harness_call)

// uint64_t syn_sme2_harness_clobber(void)
//
// Wrong on purpose: `smstart` and `smstop` zero z8-z15, and so d8-d15, which it never saves.
FUNC_BEGIN SYM(syn_sme2_harness_clobber)
    smstart
    smstop
    mov     x0, #0
    ret
FUNC_END SYM(syn_sme2_harness_clobber)
