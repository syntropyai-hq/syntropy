// SPDX-License-Identifier: MIT OR Apache-2.0

//! `unsafe` outside the allowed impl, which the check must refuse.

pub fn first(values: &[u8]) -> u8 {
    unsafe { *values.get_unchecked(0) }
}
