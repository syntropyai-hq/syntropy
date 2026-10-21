// SPDX-License-Identifier: MIT OR Apache-2.0

//! A stand-in for the CUDA driver library: one entry point, never called.

#[unsafe(no_mangle)]
pub extern "C" fn cuInit(_flags: u32) -> i32 {
    0
}
