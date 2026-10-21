// A key without a default.
use serde::Deserialize;
use syntropy_config::ConfigSection;

/// Probe settings.
#[derive(Deserialize, ConfigSection)]
#[config(section = "probe", scope = Node, owner = "RFC-0010")]
struct ProbeSection {
    /// How long the probe waits for a model.
    #[config(reload = Node)]
    model_wait: syntropy_config::Duration,
}

fn main() {}
