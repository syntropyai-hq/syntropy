// SPDX-License-Identifier: MIT OR Apache-2.0

//! The lock-free single-producer, single-consumer ring of RFC-0001 §8.3, which `io`, `net` and
//! `ipc` share.
//!
//! Stub: layer 1 in RFC-0001 §4.1. DIST-E1-T01 builds the ring here. `io`, `net` and `ipc` may
//! depend on this crate inside layer 1, and it depends on no workspace crate and has no tokio
//! (RFC-0001 §4.2 rules 2 and 3, notes of 2026-10-22).
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]
