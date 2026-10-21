// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Transport` trait and its backends, frame codecs and delegated-credential checks.
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0031. So far it holds [`frame`]: the base header,
//! the control frames and their CRC32C (RFC-0031 §6); and the control plane's envelope
//! ([`ctl::envelope`], RFC-0031 §9.1).
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]

pub mod ctl;
pub mod frame;
