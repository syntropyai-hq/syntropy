// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lowers the graph to per-stage step programs: per-device dispatchers, the CPU compute pool,
//! doorbells and graph capture.
//!
//! Layer 3 in RFC-0001 §4.1, designed in RFC-0012. Built so far:
//!
//! - [`thread`]: the dispatcher threads, one per GPU or NPU device and one per CPU device, pinned
//!   NUMA-local at the highest unprivileged priority, with their SPSC rings to the `net` domain and
//!   the I/O engine and their spin-then-park wait between steps (PLC-E7-T01).
//! - [`dispatch`]: the pollable dispatcher. [`Dispatcher::poll_once`] runs one non-blocking pass
//!   of a stage's step loop, generic over the scheduler and device sides and timed on a
//!   statically dispatched `TickSource`. [`Dispatcher::run`] is the production thread's loop over
//!   it (PLC-E7-T17).
//! - [`pool`]: the CPU compute pool's inline path, [`ComputePool::run_inline`], which runs a
//!   doorbell's CPU work item on the calling thread (PLC-E7-T17).
//!
//! `exec` depends on no async runtime and no network crate: RFC-0001 §4.2 rule 3 and ADR-0012
//! item 6, which `cargo xtask check-deps` enforces.

#![forbid(unsafe_code)]
// `core::Error` is RFC-0001 §5.12's struct with its fields inline: 344 B [measured: `size_of`,
// debug build, x86-64 Linux, `core` at ce2044ae, unchanged by COMP-E13-T15], over clippy's
// 128-byte limit for an `Err`. An error is the cold path, and the Ok value of a `Result` is
// written in place.
#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

pub mod dispatch;
pub mod pool;
pub mod thread;

pub use dispatch::{
    Dispatcher, LoopConfig, NoDeviceWake, POLL_BUDGET, PollOutcome, PollStats, Serviced,
    StepDevice, StepProgress, StepScheduler, WakeSource, WakeWaiter,
};
pub use pool::{ComputePool, PoolProgram};
pub use thread::{
    Cores, DispatcherCtx, DispatcherKind, DispatcherRings, DispatcherSpec, DispatcherThread,
    HotThreadReport, ThreadPriority, spawn_dispatcher,
};
