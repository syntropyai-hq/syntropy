// SPDX-License-Identifier: MIT OR Apache-2.0

//! I/O engines (io_uring, IoRing, IOCP, a `preadv` pool, mmap fallback) with per-core submission
//! rings and registered buffers.
//!
//! Stub: layer 1 in RFC-0001 §4.1, designed in RFC-0021. Nothing is implemented yet.
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]
