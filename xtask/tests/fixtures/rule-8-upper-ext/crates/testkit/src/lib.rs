// SPDX-License-Identifier: MIT OR Apache-2.0

//! A module whose file extension is `rs` only in another case.

#![deny(unsafe_code)]

pub mod guards;
#[cfg(windows)]
pub mod upper;
