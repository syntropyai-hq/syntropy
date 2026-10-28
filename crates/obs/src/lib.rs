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
//! - [`trace`]: local tracing: the span model, the 64-byte span record and its writer, the head's
//!   step sampling, `traceparent` in and out, the flight recorder and the `obs.trace.*` keys
//!   (RFC-0053 §3.1–§3.4; PROD-E8-T07);
//! - the `CostTerm` key space (RFC-0053 §2.2) and the fault points of RFC-0060 §9.2
//!   ([`faults`], [`fault_point!`]).
//!
//! **Hot threads.** [`metrics`]' cells, [`log`], [`ring`], [`timeline`] and [`trace`]'s span
//! writer and sampler run on hot threads: they
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
pub mod trace;

pub use cost_term::CostTerm;
pub use drain::{DRAIN_PERIOD, Drain, DrainReport};
pub use telemetry::TelemetryRecord;

/// The test-only features (RFC-0001 §4.2 rule 10) this build of the crate has on. The binary's
/// build metadata lists them (`cli`'s `build_id::features`, RFC-0062 §8.1), since cargo's
/// feature unification can turn one on without the binary's own crate seeing it.
pub const TEST_FEATURES: &[&str] = if cfg!(feature = "fault-injection") {
    &["fault-injection"]
} else {
    &[]
};

#[cfg(test)]
mod feature_tests {
    /// `TEST_FEATURES` names `fault-injection` exactly when it is on.
    #[test]
    fn test_features_name_what_is_on() {
        assert_eq!(
            super::TEST_FEATURES.contains(&"fault-injection"),
            cfg!(feature = "fault-injection")
        );
    }
}
