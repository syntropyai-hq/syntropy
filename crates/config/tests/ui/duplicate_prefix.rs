// Two sections that claim one prefix, in one registry.
use serde::Deserialize;
use syntropy_config::{ConfigSection, Registry};

/// Probe settings.
#[derive(Deserialize, ConfigSection)]
#[config(section = "probe", scope = Node, owner = "RFC-0010")]
struct ProbeSection {
    /// How long the probe waits for a model.
    #[config(reload = Node, default = "7s")]
    model_wait: syntropy_config::Duration,
}

/// More probe settings, from another crate.
#[derive(Deserialize, ConfigSection)]
#[config(section = "probe", scope = Node, owner = "RFC-0020")]
struct OtherProbeSection {
    /// A second key under the same prefix.
    #[config(reload = Hot, default = 1)]
    retries: u32,
}

static REGISTRY: Registry = Registry::new(&[&ProbeSection::SCHEMA, &OtherProbeSection::SCHEMA]);

fn main() {
    let _ = &REGISTRY;
}
