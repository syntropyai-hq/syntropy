// SPDX-License-Identifier: MIT OR Apache-2.0

//! The step scheduler: continuous batching, chunked prefill, verify windows, drafting, acceptance
//! and the sampler.
//!
//! Layer 4 in RFC-0001 §4.1, designed in RFC-0012. Built so far: [`config`], the `sched.*`
//! configuration section of RFC-0012 §12.

#![forbid(unsafe_code)]

pub mod config;

pub use config::{Rejection, SchedSection};
