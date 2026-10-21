// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tier manager: arenas, pinned budgets, the `ResidencyMap`, the expert cache, prefetch and the
//! KV store.
//!
//! Stub: layer 2 in RFC-0001 §4.1, designed in RFC-0020. Nothing is implemented yet.
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]
