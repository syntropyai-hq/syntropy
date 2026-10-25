// SPDX-License-Identifier: MIT OR Apache-2.0

//! The expert cache, slot table, prediction and the miss balancer (RFC-0023).
//!
//! What exists so far (TIER-E8-T01):
//! - [`sizing`] computes the epoch's `MAX_UNION` and `MAX_PRED` from the plan's largest pass and
//!   lookahead, and from them the [`DoorbellLayout`] that `syntropy-kernels` defines
//!   (RFC-0023 §2, §11).
//! - [`stats`] is the statistics ring from a dispatcher to the housekeeping thread: RFC-0001
//!   §8.3's lossy hot-thread ring, one per dispatcher, whose first record kind is one pass's
//!   routing for one MoE layer (RFC-0023 §5.1).
//! - [`trace`] is the routing-trace format, its recorder and its reader (RFC-0060 §12.4).
//!
//! The cache itself, the policy and the balancer come in later TIER-E8 tasks.
//!
//! [`DoorbellLayout`]: syntropy_kernels::moe::DoorbellLayout

#![forbid(unsafe_code)]

pub mod sizing;
pub mod stats;
pub mod trace;

pub use sizing::{MoeRouting, SizingError, doorbell_layout, largest_pass, lookahead, plan_layout};
pub use stats::{RouteRecord, StatDrop, StatRecord, StatsConsumer, StatsProducer, stats_ring};
pub use trace::{RouteFault, TraceError, TraceReader, TraceWriter};
