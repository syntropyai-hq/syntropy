// SPDX-License-Identifier: MIT OR Apache-2.0

//! The declarations are of the allowed shape, but other code uses `unsafe` too.

#![deny(unsafe_code)]

pub mod memory;
pub mod provider;
