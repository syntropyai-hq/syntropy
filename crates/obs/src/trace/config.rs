// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tracing keys (RFC-0053 §3.2, §3.3, §3.6). RFC-0053 names them in its text and in no table;
//! their types, defaults, scopes and reload classes are chosen here and recorded in its note of
//! 2026-10-28. Every key is node scope.

use serde::Deserialize;
use syntropy_config::{ByteSize, ConfigSection};
use syntropy_core::policy::Fraction;

/// `obs.flight_recorder`: the flight recorder's size (RFC-0053 §3.6; Performance model).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "obs", shared, scope = Node, owner = "RFC-0053")]
pub struct ObsSection {
    /// The most bytes of spans and log records the flight recorder keeps.
    #[config(reload = Node, default = "64MiB")]
    pub flight_recorder: ByteSize,
}

/// `obs.trace.*`: sampling and remote trace context (RFC-0053 §3.2, §3.3).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "obs.trace", scope = Node, owner = "RFC-0053")]
pub struct ObsTraceSection {
    /// The probability a decode step after the first four is sampled, by a hash of (trace ID,
    /// step).
    #[config(reload = Hot, default = 0.015625)]
    #[serde(with = "syntropy_config::form")]
    pub step_sample: Fraction,
    /// Whether an incoming `traceparent`'s sampled bit forces a full trace. Off by default, so
    /// an untrusted client cannot make the node trace every step.
    #[config(reload = Hot, default = false)]
    pub honor_remote_sampled: bool,
}
