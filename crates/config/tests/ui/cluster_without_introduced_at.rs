// SPDX-License-Identifier: MIT OR Apache-2.0
// A cluster-scope key without `introduced_at`.
use serde::Deserialize;
use syntropy_config::ConfigSection;

/// Re-placement thresholds.
#[derive(Deserialize, ConfigSection)]
#[config(section = "replan", scope = Cluster, owner = "RFC-0011")]
struct ReplanSection {
    /// The smallest gain worth a re-plan.
    #[config(reload = Hot, default = 0.1)]
    min_gain: f64,
}

fn main() {}
