// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lowers the graph to per-stage step programs: per-device dispatchers, the CPU compute pool,
//! doorbells and graph capture.
//!
//! Layer 3 in RFC-0001 §4.1, designed in RFC-0012. Built so far:
//!
//! - [`thread`]: the dispatcher threads, one per GPU or NPU device and one per CPU device, pinned
//!   NUMA-local at the highest unprivileged priority, with their SPSC rings to the `net` domain and
//!   the I/O engine and their spin-then-park wait between steps (PLC-E7-T01).
//!
//! `exec` depends on no async runtime and no network crate: RFC-0001 §4.2 rule 3 and ADR-0012
//! item 6, which `cargo xtask check-deps` enforces.

#![forbid(unsafe_code)]

pub mod thread;

pub use thread::{
    Cores, Dispatcher, DispatcherCtx, DispatcherKind, DispatcherRings, DispatcherSpec,
    HotThreadReport, ThreadPriority, spawn_dispatcher,
};
