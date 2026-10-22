// SPDX-License-Identifier: MIT OR Apache-2.0

//! Outside `src/`, so only the `#[path]` in `lib.rs` brings it in.

#![allow(unsafe_code)]

pub fn read(p: *const u8) -> u8 {
    unsafe { *p }
}
