// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `#[path]` module outside `src/`, whose own `allow` would turn `unsafe` back on.

#![deny(unsafe_code)]

pub mod guards;
#[path = "../outside.rs"]
pub mod outside_probe;
