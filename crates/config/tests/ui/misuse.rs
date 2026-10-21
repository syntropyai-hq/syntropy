// Attributes in the wrong place or form.
use serde::Deserialize;
use syntropy_config::ConfigSection;

/// No scope, a malformed owner and a malformed prefix.
#[derive(Deserialize, ConfigSection)]
#[config(section = "Bad-Prefix", owner = "0054")]
struct Malformed {
    /// A key.
    #[config(reload = Hot, default = 1)]
    a: u32,
}

/// Cluster attributes on node keys, and an unknown attribute.
#[derive(Deserialize, ConfigSection)]
#[config(section = "node_keys", scope = Node, owner = "RFC-0054")]
struct NodeKeys {
    /// A node key with an authority.
    #[config(reload = Hot, default = 1, authority = Keyholder)]
    a: u32,
    /// A node key with a feature level.
    #[config(reload = Hot, default = 1, introduced_at = 1)]
    b: u32,
    /// An unknown attribute.
    #[config(reload = Hot, default = 1, units = "bytes")]
    c: u32,
    /// A renamed key.
    #[config(reload = Hot, default = 1)]
    #[serde(rename = "dee")]
    d: u32,
    /// A bad reload class and a bad default.
    #[config(reload = Sometimes, default = maybe)]
    e: u32,
}

fn main() {}
