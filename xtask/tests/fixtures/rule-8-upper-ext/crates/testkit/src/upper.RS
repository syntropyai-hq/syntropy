// SPDX-License-Identifier: MIT OR Apache-2.0

//! Loaded as `upper.rs` on a case-insensitive file system.

#![allow(unsafe_code)]

pub fn read(p: *const u8) -> u8 {
    unsafe { *p }
}
