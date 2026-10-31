// SPDX-License-Identifier: MIT OR Apache-2.0

//! The model graph IR, shared blocks, architecture families, census classifiers and the reference
//! interpreter.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0043. It depends on `manifest` and, for the op
//! vocabulary, on `kernels` (RFC-0043 §1). It holds the IR types and their validator ([`ir`],
//! RFC-0043 §2.1, §2.2), the M0 blocks ([`blocks`], RFC-0043 §3), the legal stage boundaries
//! and side data computed from value liveness ([`boundary`], RFC-0043 §6.1), the reference
//! interpreter's contract with its driver ([`reference`](mod@reference), RFC-0060 §4.1), the
//! family trait, its metadata view, the static registry and the frozen tag list ([`families`],
//! RFC-0043 §5.1), the `llama` family module ([`families::llama`], RFC-0043 §5.5), the fixture
//! spec a family supplies ([`spec`], RFC-0043 §9.1), and the census classification engine
//! ([`classify`], RFC-0043 §4). The other blocks and family modules, segments, sharding,
//! tokenizers and the interpreter itself come with later tasks.
//!
//! From `manifest`, the families use `HyperParams`, `TensorClass` and the metadata readers.
//! The census groups are not in `manifest` yet, so the facts [`ir::validate()`] needs are
//! passed as [`ir::CensusGroup`]s.

#![forbid(unsafe_code)]

pub mod blocks;
pub mod boundary;
pub mod classify;
pub mod families;
pub mod ir;
pub mod reference;
pub mod spec;
