// SPDX-License-Identifier: MIT OR Apache-2.0

//! A stand-in for cuFFT: one entry point, never called.

#[unsafe(no_mangle)]
pub extern "C" fn cufftPlan1d(_flags: u32) -> i32 {
    0
}
