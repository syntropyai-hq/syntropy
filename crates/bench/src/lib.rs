// SPDX-License-Identifier: MIT OR Apache-2.0

//! Benchmark scenarios, the result schema and the regression gates (RFC-0061).
//!
//! A dev-only crate in RFC-0001 §4.1. It holds:
//! - [`schema`]: the benchmark key, the result record, its JSON form and the comparability rule
//!   of RFC-0061 §2;
//! - [`rigs`]: the lab inventory of RFC-0061 §6.3 and the `RigFingerprint` computed from it;
//! - [`scenario`]: the scenario files under `bench/scenarios/`, their ID forms, the standard grid
//!   and the `scenario_version` rule (RFC-0061 §4.1);
//! - [`sc_map`]: `bench/sc-map.toml`, which maps every success criterion to its scenarios, and
//!   renders RFC-0061 §4.2's table (RFC-0061 §4.2);
//! - [`specs`]: the spec database of RFC-0061 §5.3, the `W_peak` per path it gives a rig, and the
//!   probe sanity check.
//!
//! The `syntropy-bench` binary runs the checks on them that `lint` calls: `rigs check`,
//! `specs check` and `scenarios check`.

#![forbid(unsafe_code)]

pub mod rigs;
pub mod sc_map;
pub mod scenario;
pub mod schema;
pub mod specs;
