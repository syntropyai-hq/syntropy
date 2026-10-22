// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0030 §9's keys: `cluster.*`, `discovery.*`, RFC-0030's four `net.*` keys, `swim.*`,
//! `membership.*`, `power.*`, `invite.*` and `pairing.*`.

use serde::Deserialize;
use syntropy_config::{ConfigSection, Duration, HoldsSecret};
use syntropy_net::config::{Backoff, CountInWindow};

use super::FEATURE_LEVEL_V1;

/// How a cluster admits a new LAN node (RFC-0030 §6.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum LanAdmission {
    /// One keyholder confirmation per node, with number matching (ADR-0302).
    Confirm,
    /// Any node of the namespace heard on a non-overlay LAN interface is admitted.
    Open,
    /// Only invite tokens admit.
    InviteOnly,
}

/// A LAN discovery mechanism (RFC-0030 §3, §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum Mechanism {
    /// mDNS.
    Mdns,
    /// IPv6 link-local multicast.
    V6Multicast,
    /// IPv4 broadcast.
    V4Broadcast,
}

/// Where a clustered node sends its blinded beacons (RFC-0030 §4.8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum BeaconScope {
    /// On the cluster's home networks.
    Home,
    /// On every network.
    All,
}

/// Which relays the node may use (RFC-0030 §6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum RelayMode {
    /// Off until the cluster has an invite-joined member, an outstanding invite or a member
    /// with `off_lan_reach`; then as `default`.
    Auto,
    /// The built-in set from the signed `relays.json`.
    Default,
    /// The relays in `net.relays`.
    Custom,
    /// No relays.
    Disabled,
}

/// When the daemon holds an idle-sleep inhibitor (RFC-0030 §5.9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum InhibitIdleSleep {
    /// While the node hosts a stage with live sequences.
    WhileServing,
    /// Never.
    Never,
    /// Always.
    Always,
}

/// What a pairing window opened without options allows (RFC-0030 §6.3). Written
/// `{ for = "10m", joins = 4, include_wifi = false }`, after `syntropy cluster open`'s flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(deny_unknown_fields)]
pub struct PairingWindow {
    /// How long the window stays open.
    #[serde(rename = "for")]
    pub open_for: Duration,
    /// How many nodes it admits before it closes.
    pub joins: u32,
    /// Whether it accepts pairing over Wi-Fi as well as wired and Thunderbolt interfaces.
    pub include_wifi: bool,
}

/// The node's discovery namespace and static peers, and the cluster's name and admission mode
/// (RFC-0030 §9).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "cluster", scope = Node, owner = "RFC-0030")]
pub struct ClusterSection {
    /// The discovery namespace; nodes in different namespaces never see each other (RFC-0030
    /// §4.1).
    #[config(reload = Node, default = "default")]
    pub namespace: String,
    /// The cluster's display name.
    #[config(reload = Hot, default = computed("\"<hostname>'s cluster\""), scope = Cluster,
             introduced_at = FEATURE_LEVEL_V1)]
    pub name: Option<String>,
    /// How the cluster admits a new LAN node. The pairing window is a command, not a mode.
    #[config(reload = Hot, default = "confirm", scope = Cluster,
             introduced_at = FEATURE_LEVEL_V1)]
    pub lan_admission: LanAdmission,
    /// Peers dialled directly: `host:port` or `host:port#<node-id>`.
    #[config(reload = Hot, default = [])]
    pub static_peers: Vec<String>,
}

/// LAN discovery (RFC-0030 §3, §4, §9).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "discovery", scope = Node, owner = "RFC-0030")]
pub struct DiscoverySection {
    /// The UDP port discovery listens on (RFC-0030 §3).
    #[config(reload = Node, default = 7434)]
    pub port: u16,
    /// The mechanisms in use; an empty list means static peers only.
    #[config(reload = Node, default = ["mdns", "v6_multicast", "v4_broadcast"])]
    pub mechanisms: Vec<Mechanism>,
    /// Interface globs discovery ignores.
    #[config(reload = Hot, default = computed("container and VM bridges, and loopback"))]
    pub ignore_interfaces: Option<Vec<String>>,
    /// Where a clustered node sends its blinded beacons (RFC-0030 §4.8).
    #[config(reload = Hot, default = "home")]
    pub beacon_scope: BeaconScope,
}

/// RFC-0030's keys inside RFC-0031's `net.*` (RFC-0054 §5.1).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "net", shared, scope = Node, owner = "RFC-0030")]
pub struct NetMembershipSection {
    /// UDP for iroh and TCP for the fallback (RFC-0030 §3).
    #[config(reload = Node, default = 7435)]
    pub listen_port: u16,
    /// Which relays the node may use (RFC-0030 §6.4).
    #[config(reload = Hot, default = "auto")]
    pub relay_mode: RelayMode,
    /// The relays used when `relay_mode = "custom"`.
    #[config(reload = Hot, default = [])]
    pub relays: Vec<String>,
    /// Whether bulk transfers run on metered links (RFC-0030 §7.2).
    #[config(reload = Hot, default = false)]
    pub bulk_on_metered: bool,
}

/// SWIM failure detection (RFC-0030 §5.4, §5.8). Cluster scope except `resume_detect`.
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "swim", scope = Cluster, owner = "RFC-0030")]
pub struct SwimSection {
    /// The probe period on the LAN.
    #[config(reload = Hot, default = "1s", introduced_at = FEATURE_LEVEL_V1)]
    pub period: Duration,
    /// The probe period across a WAN.
    #[config(reload = Hot, default = "2s", introduced_at = FEATURE_LEVEL_V1)]
    pub wan_period: Duration,
    /// Missed acks before indirect probes.
    #[config(reload = Hot, default = 2, introduced_at = FEATURE_LEVEL_V1)]
    pub miss_probe: u32,
    /// Missed probes before a LAN member is suspected.
    #[config(reload = Hot, default = 3, introduced_at = FEATURE_LEVEL_V1)]
    pub miss_suspect: u32,
    /// Missed probes before a WAN member is suspected.
    #[config(reload = Hot, default = 4, introduced_at = FEATURE_LEVEL_V1)]
    pub wan_miss_suspect: u32,
    /// The shortest suspicion timeout on the LAN.
    #[config(reload = Hot, default = "2s", introduced_at = FEATURE_LEVEL_V1)]
    pub suspect_min: Duration,
    /// The longest suspicion timeout on the LAN.
    #[config(reload = Hot, default = "4s", introduced_at = FEATURE_LEVEL_V1)]
    pub suspect_max: Duration,
    /// The shortest suspicion timeout across a WAN.
    #[config(reload = Hot, default = "10s", introduced_at = FEATURE_LEVEL_V1)]
    pub wan_suspect_min: Duration,
    /// The longest suspicion timeout across a WAN.
    #[config(reload = Hot, default = "20s", introduced_at = FEATURE_LEVEL_V1)]
    pub wan_suspect_max: Duration,
    /// Members asked to probe indirectly.
    #[config(reload = Hot, default = 3, introduced_at = FEATURE_LEVEL_V1)]
    pub k_indirect: u32,
    /// The largest cluster probed as a full mesh.
    #[config(reload = Hot, default = 16, introduced_at = FEATURE_LEVEL_V1)]
    pub full_mesh_max: u32,
    /// The gossip retransmit multiplier.
    #[config(reload = Hot, default = 3, introduced_at = FEATURE_LEVEL_V1)]
    pub retransmit_mult: u32,
    /// How soon after a resume from sleep the node re-checks its peers (RFC-0030 §5.8).
    #[config(reload = Hot, default = "5s", scope = Node)]
    pub resume_detect: Duration,
}

/// Flap damping and planned restarts (RFC-0030 §5.7, §5.9).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "membership", scope = Cluster, owner = "RFC-0030")]
pub struct MembershipSection {
    /// Failures within a window that mark a member as flapping (RFC-0030 §5.7).
    #[config(reload = Hot, default = { count = 3, window = "10m" },
             introduced_at = FEATURE_LEVEL_V1)]
    pub flap_threshold: CountInWindow,
    /// How long a flapping member is held back, doubling on each repeat.
    #[config(reload = Hot, default = { initial = "5m", max = "1h" },
             introduced_at = FEATURE_LEVEL_V1)]
    pub flap_backoff: Backoff,
    /// How long a member that announced a planned restart may be away (RFC-0030 §5.9).
    #[config(reload = Hot, default = "30s", introduced_at = FEATURE_LEVEL_V1)]
    pub restart_grace: Duration,
}

/// The idle-sleep inhibitor (RFC-0030 §5.9).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "power", scope = Node, owner = "RFC-0030")]
pub struct PowerSection {
    /// When the daemon holds an idle-sleep inhibitor. It never blocks lid-close or
    /// user-requested sleep.
    #[config(reload = Hot, default = computed("while_serving on mains power, never on battery"))]
    pub inhibit_idle_sleep: Option<InhibitIdleSleep>,
}

/// Invite tokens (RFC-0030 §6.5, §9). `invite.require_confirm` is RFC-0034's.
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "invite", scope = Cluster, owner = "RFC-0030")]
pub struct InviteSection {
    /// How long a new invite token is valid.
    #[config(reload = Hot, default = "24h", introduced_at = FEATURE_LEVEL_V1)]
    pub default_expiry: Duration,
    /// How many joins a new invite token allows.
    #[config(reload = Hot, default = 1, introduced_at = FEATURE_LEVEL_V1)]
    pub default_uses: u32,
    /// The longest validity a token can request.
    #[config(reload = Hot, default = "7d", introduced_at = FEATURE_LEVEL_V1)]
    pub max_expiry: Duration,
}

/// Pairing windows (RFC-0030 §6.3).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "pairing", scope = Cluster, owner = "RFC-0030")]
pub struct PairingSection {
    /// What a window opened without options allows: 10 minutes, 4 joins, wired and
    /// Thunderbolt interfaces only.
    #[config(reload = Hot, default = { for = "10m", joins = 4, include_wifi = false },
             introduced_at = FEATURE_LEVEL_V1)]
    pub window_default: PairingWindow,
}
