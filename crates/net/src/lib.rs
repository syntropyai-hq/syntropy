// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Transport` trait and its backends, frame codecs and delegated-credential checks.
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0031. So far it holds [`frame`]: the base header,
//! the control frames, the `Activation` and `Token` frames and their CRC32C (RFC-0031 §6);
//! [`crypto`]: the cryptographic helpers of RFC-0034 §3; [`words`]: the fingerprint and
//! recovery-code word list of RFC-0034 §6.5; [`fingerprint`]: the six-word identity fingerprints
//! of RFC-0034 §6.5; and the control plane's envelope ([`ctl::envelope`], RFC-0031 §9.1).
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]

pub mod config;
pub mod crypto;
pub mod ctl;
pub mod fingerprint;
pub mod frame;
pub mod words;
