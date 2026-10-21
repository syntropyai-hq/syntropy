// The keys RFC-0054 §2.3 step 10 names, declared `Member`.
use serde::Deserialize;
use syntropy_config::ConfigSection;

/// Top-level cluster keys.
#[derive(Deserialize, ConfigSection)]
#[config(section = "", scope = Cluster, owner = "RFC-0054")]
struct Root {
    /// Allowed.
    #[config(reload = Hot, default = true, introduced_at = 1, authority = Member)]
    allow_approximate: bool,
    /// The network scope.
    #[config(reload = Hot, default = "lan+invites", introduced_at = 1, authority = Member)]
    network_scope: String,
    /// Cluster aliases.
    #[config(reload = Hot, default = {}, introduced_at = 1, authority = Member)]
    cluster_aliases: std::collections::BTreeMap<String, String>,
}

/// Update keys.
#[derive(Deserialize, ConfigSection)]
#[config(section = "update", scope = Node, owner = "RFC-0052")]
struct Update {
    /// Coordinated updates.
    #[config(reload = Hot, default = true, scope = Cluster, introduced_at = 1, authority = Member)]
    cluster_coordinated: bool,
}

/// Instance keys.
#[derive(Deserialize, ConfigSection)]
#[config(section = "", scope = Instance, owner = "RFC-0054")]
struct Policy {
    /// Fidelity.
    #[config(reload = Instance, default = "exact", authority = Member)]
    fidelity: String,
    /// Default fidelity.
    #[config(reload = Instance, default = "exact", authority = Member)]
    default_fidelity: String,
    /// Pins.
    #[config(reload = Instance, default = [], authority = Member)]
    pin: Vec<String>,
    /// Overrides.
    #[config(reload = Instance, default = [], authority = Member)]
    tensor_override: Vec<String>,
    /// Routine tuning, which may be `Member`.
    #[config(reload = Instance, default = "latency", authority = Member)]
    objective: String,
}

/// Approximate knobs.
#[derive(Deserialize, ConfigSection)]
#[config(section = "approximate", scope = Instance, owner = "RFC-0042")]
struct Approximate {
    /// A knob.
    #[config(reload = Instance, default = false, authority = Member)]
    topk_reduce: bool,
}

fn main() {}
