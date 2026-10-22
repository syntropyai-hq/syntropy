// SPDX-License-Identifier: MIT OR Apache-2.0

//! A module of the bin target, with its own `allow`.

#![allow(unsafe_code)]

pub fn read(p: *const u8) -> u8 {
    unsafe { *p }
}
