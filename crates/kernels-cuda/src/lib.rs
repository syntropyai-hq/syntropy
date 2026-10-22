// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `KernelProvider` for NVIDIA GPUs, written in CUDA C++.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0040 (§7.1 for the provider, §10 for the build).
//! Built so far: the translation-unit structure of RFC-0040 §10.2 ([`kgen`]): one unit per (op
//! family, quant-scheme group) under `csrc/`, each with an explicit instantiation list
//! generated from `support.toml` by `cargo xtask kgen`; the parameter-block header
//! `csrc/params.h` and its generated Rust mirrors ([`params`]); and the compile flags and
//! kernel object cache keys, with contraction off for bit-exact-class units. No kernel, and no
//! build of the CUDA sources, exists yet.
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]

pub mod kgen;
pub mod params;

pub use kgen::support::registry;
