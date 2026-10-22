// SPDX-License-Identifier: MIT OR Apache-2.0

//! The model graph IR, shared blocks, architecture families, census classifiers and the reference
//! interpreter.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0043. It depends on `manifest` and, for the op
//! vocabulary, on `kernels` (RFC-0043 §1). It holds the IR types and their validator ([`ir`],
//! RFC-0043 §2.1, §2.2), and the legal stage boundaries and side data computed from value
//! liveness ([`boundary`], RFC-0043 §6.1). The blocks, families, census classification,
//! segments, sharding, tokenizers and the reference interpreter come with later tasks.
//!
//! `manifest` is still a stub, so nothing from it is used yet: the census facts that
//! [`ir::validate`] needs are passed as [`ir::CensusGroup`]s.

#![forbid(unsafe_code)]

pub mod boundary;
pub mod ir;
