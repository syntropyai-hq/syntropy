// SPDX-License-Identifier: MIT OR Apache-2.0
// A key without a reload class.
use serde::Deserialize;
use syntropy_config::ConfigSection;

/// Probe settings.
#[derive(Deserialize, ConfigSection)]
#[config(section = "probe", scope = Node, owner = "RFC-0010")]
struct ProbeSection {
    /// How long the probe waits for a model.
    #[config(default = "7s")]
    model_wait: syntropy_config::Duration,
}

fn main() {}
