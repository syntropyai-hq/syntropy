// SPDX-License-Identifier: MIT OR Apache-2.0

//! The daemon-runner protocol, the shared-memory layout and SPSC rings.
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0001 §3.5. So far it holds the shared-memory region
//! ([`shm`]): its header, its rings and the segment each OS provides. The command channel and its
//! messages are DIST-E1-T03's.
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(any(target_os = "linux", target_vendor = "apple", windows))]
pub mod shm;
