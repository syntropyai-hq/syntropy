// SPDX-License-Identifier: MIT OR Apache-2.0

//! A provider crate that links a stand-in for cuFFT.

unsafe extern "C" {
    fn cufftPlan1d(flags: u32) -> i32;
}

/// Calls `cufftPlan1d`. The fixture binary is built and read, never run.
pub fn init() -> i32 {
    // SAFETY: never executed; the fixture only needs the reference to `cufftPlan1d`.
    unsafe { cufftPlan1d(0) }
}
