// SPDX-License-Identifier: MIT OR Apache-2.0

//! The exception's shape is right, but another module uses `unsafe` too.

#![deny(unsafe_code)]

pub mod guards;
pub mod other;
