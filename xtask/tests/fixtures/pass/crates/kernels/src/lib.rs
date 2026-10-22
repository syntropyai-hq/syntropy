// SPDX-License-Identifier: MIT OR Apache-2.0

//! `kernels` in this fixture. Its root denies `unsafe`, and its only `unsafe` is the listed
//! trait method declarations, each with no body, in `src/provider.rs` and `src/memory.rs`.

#![deny(unsafe_code)]

pub mod memory;
pub mod provider;
