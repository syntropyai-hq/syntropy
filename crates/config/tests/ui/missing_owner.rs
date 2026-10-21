// SPDX-License-Identifier: MIT OR Apache-2.0
// A section without an owner.
use serde::Deserialize;
use syntropy_config::ConfigSection;

/// Probe settings.
#[derive(Deserialize, ConfigSection)]
#[config(section = "probe", scope = Node)]
struct ProbeSection {
    /// How long the probe waits for a model.
    #[config(reload = Node, default = "7s")]
    model_wait: syntropy_config::Duration,
}

fn main() {}
