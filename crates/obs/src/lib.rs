// SPDX-License-Identifier: MIT OR Apache-2.0

//! Span and metric naming, counters and histograms for hot threads, the telemetry ring codec, fault
//! points and exporters.
//!
//! Layer 0 in RFC-0001 §4.1, designed in RFC-0053. So far it holds:
//!
//! - [`metrics`]: the metric registry, label binding at registration, per-thread cells and
//!   log-linear histograms (RFC-0053 §2.1; PROD-E8-T01);
//! - [`telemetry`]: the `telemetry` ring's [`TelemetryRecord`] and its codec (RFC-0053 §1);
//! - [`drain`]: the 250 ms drain the runner drives, which turns cells into records (RFC-0001 §13);
//! - [`log`]: hot-thread [`log::LogRecord`]s with static callsites, which the daemon formats
//!   (RFC-0053 §4.1; PROD-E8-T04);
//! - [`ring`]: the one writer of each shared-memory ring, which drops and counts when the ring is
//!   full;
//! - [`timeline`]: hot-thread time through a `TickSource`, the only time `obs` reads (RFC-0060
//!   §2.1 rule T1);
//! - the `CostTerm` key space (RFC-0053 §2.2) and the fault points of RFC-0060 §9.2
//!   ([`faults`], [`fault_point!`]).
//!
//! **Hot threads.** [`metrics`]' cells, [`log`], [`ring`] and [`timeline`] run on hot threads: they
//! allocate nothing after `Warming`, take no lock and use no `tracing` (ADR-0012; the
//! `hot-threads` step of `cargo xtask ci lint` rejects it there).

#![forbid(unsafe_code)]

pub mod cost_term;
pub mod drain;
pub mod faults;
pub mod log;
pub mod metrics;
pub mod ring;
pub mod telemetry;
pub mod timeline;

pub use cost_term::CostTerm;
pub use drain::{DRAIN_PERIOD, Drain, DrainReport};
pub use telemetry::TelemetryRecord;
