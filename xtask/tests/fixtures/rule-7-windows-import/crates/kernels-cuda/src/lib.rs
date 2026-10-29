// SPDX-License-Identifier: MIT OR Apache-2.0

//! A provider crate that imports a stand-in for the CUDA driver library, `nvcuda.dll`. The
//! `raw-dylib` kind makes rustc write the import itself, so no import library or DLL is needed.

#[link(name = "nvcuda", kind = "raw-dylib")]
unsafe extern "C" {
    fn cuInit(flags: u32) -> i32;
}

/// Calls `cuInit`. The fixture binary is built and read, never run.
pub fn init() -> i32 {
    // SAFETY: never executed; the fixture only needs the import of `cuInit`.
    unsafe { cuInit(0) }
}
