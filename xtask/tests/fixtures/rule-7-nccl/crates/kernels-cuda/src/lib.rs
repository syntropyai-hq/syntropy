// SPDX-License-Identifier: MIT OR Apache-2.0

//! A provider crate that links a stand-in for NCCL.

unsafe extern "C" {
    fn ncclCommInitRank(flags: u32) -> i32;
}

/// Calls `ncclCommInitRank`. The fixture binary is built and read, never run.
pub fn init() -> i32 {
    // SAFETY: never executed; the fixture only needs the reference to `ncclCommInitRank`.
    unsafe { ncclCommInitRank(0) }
}
