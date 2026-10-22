// SPDX-License-Identifier: MIT OR Apache-2.0

//! Assembly with no `unsafe` token, which only the lint level refuses.

core::arch::global_asm!("nop");
