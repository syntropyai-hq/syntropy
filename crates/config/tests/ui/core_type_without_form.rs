// SPDX-License-Identifier: MIT OR Apache-2.0
// Keys of `syntropy-core`'s policy value types without `serde(with = "syntropy_config::form")`,
// which would read the record encoding, not RFC-0054 §3's forms (PROD-E1-T23).
use serde::Deserialize;
use syntropy_config::ConfigSection;
use syntropy_core::policy::{Fraction, NodeSelector, Reserve};

/// Resource settings.
#[derive(Deserialize, ConfigSection)]
#[config(section = "resource", scope = Node, owner = "RFC-0020")]
struct Resource {
    /// RAM kept back.
    #[config(reload = Hot, default = "6GiB")]
    ram_headroom: Reserve,
    /// A share.
    #[config(reload = Hot, default = unset)]
    #[serde(default)]
    share: Option<Fraction>,
    /// Nodes.
    #[config(reload = Hot, default = [])]
    nodes: Vec<syntropy_core::policy::NodeSelector>,
    /// One node, read correctly.
    #[config(reload = Hot, default = "*")]
    #[serde(with = "syntropy_config::form")]
    node: NodeSelector,
}

fn main() {}
