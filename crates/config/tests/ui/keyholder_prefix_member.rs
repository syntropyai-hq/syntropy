// Keys under `security.*`, `invite.*`, `coordinator.*`, `swim.*` and `cluster.lan_admission`
// declared `Member` (RFC-0054 §2.3 step 10).
use serde::Deserialize;
use syntropy_config::ConfigSection;

/// Security.
#[derive(Deserialize, ConfigSection)]
#[config(section = "security", scope = Cluster, owner = "RFC-0034")]
struct Security {
    /// A key.
    #[config(reload = Hot, default = true, introduced_at = 1, authority = Member)]
    require_signatures: bool,
}

/// Invites.
#[derive(Deserialize, ConfigSection)]
#[config(section = "invite", scope = Cluster, owner = "RFC-0030")]
struct Invite {
    /// A key.
    #[config(reload = Hot, default = "24h", introduced_at = 1, authority = Member)]
    ttl: syntropy_config::Duration,
}

/// The coordinator.
#[derive(Deserialize, ConfigSection)]
#[config(section = "coordinator", scope = Cluster, owner = "RFC-0032")]
struct Coordinator {
    /// A key.
    #[config(reload = Hot, default = "10s", introduced_at = 1, authority = Member)]
    lease: syntropy_config::Duration,
}

/// Membership gossip.
#[derive(Deserialize, ConfigSection)]
#[config(section = "swim", scope = Cluster, owner = "RFC-0030")]
struct Swim {
    /// A key.
    #[config(reload = Hot, default = "1s", introduced_at = 1, authority = Member)]
    period: syntropy_config::Duration,
}

/// Cluster settings.
#[derive(Deserialize, ConfigSection)]
#[config(section = "cluster", scope = Cluster, owner = "RFC-0030")]
struct Cluster {
    /// LAN admission.
    #[config(reload = Hot, default = "confirm", introduced_at = 1, authority = Member)]
    lan_admission: String,
    /// A routine key, which may be `Member`.
    #[config(reload = Hot, default = 3, introduced_at = 1, authority = Member)]
    routine: u32,
}

fn main() {}
