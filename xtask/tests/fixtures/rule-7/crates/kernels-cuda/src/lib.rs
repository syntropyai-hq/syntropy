// SPDX-License-Identifier: MIT OR Apache-2.0

//! A provider crate that links a stand-in for the CUDA driver library.

unsafe extern "C" {
    fn cuInit(flags: u32) -> i32;
}

/// Calls `cuInit`. The fixture binary is built and read, never run.
pub fn init() -> i32 {
    // SAFETY: never executed; the fixture only needs the reference to `cuInit`.
    unsafe { cuInit(0) }
}
