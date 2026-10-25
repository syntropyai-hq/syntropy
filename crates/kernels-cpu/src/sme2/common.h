// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared by the SME2 assembly unit (RFC-0040 §7.3; ADR-0402 item 3): the symbol and
// function-boundary macros for Mach-O and ELF, and the assembler's feature set.

#ifndef SYNTROPY_SME2_COMMON_H
#define SYNTROPY_SME2_COMMON_H

// Mach-O prefixes C symbols with an underscore; ELF does not. Darwin's assembler takes `;` as
// a comment, so the function-boundary macros below are assembler macros, one statement a line.
#ifdef __APPLE__
#define SYM(name) _##name
#else
#define SYM(name) name
#endif

.macro FUNC_BEGIN sym
    .text
    .p2align 4
    .globl \sym
#ifndef __APPLE__
    .type \sym, %function
    .hidden \sym
#endif
\sym:
.endm

.macro FUNC_END sym
#ifndef __APPLE__
    .size \sym, . - \sym
#endif
.endm

// No Rust code runs with these features; only these routines use them, and only after the
// provider has detected SME2 (`Arm92Sme2`).
.arch armv9-a+sme2

#endif
