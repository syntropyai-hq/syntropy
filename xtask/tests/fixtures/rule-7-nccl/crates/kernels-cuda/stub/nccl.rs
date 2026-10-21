// SPDX-License-Identifier: MIT OR Apache-2.0

//! A stand-in for NCCL: one entry point, never called.

#[unsafe(no_mangle)]
pub extern "C" fn ncclCommInitRank(_flags: u32) -> i32 {
    0
}
