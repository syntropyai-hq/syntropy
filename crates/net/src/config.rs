// SPDX-License-Identifier: MIT OR Apache-2.0

//! The configuration sections of RFC-0031 §11: `net.*`, `link.*`, `quic.*`, `rdma.*` and
//! `shm.*` (RFC-0054 §5.1).
//!
//! The names, scopes and defaults are RFC-0031 §11's. Two keys inside these prefixes belong to
//! other RFCs and are declared by their own crates as shared sections: RFC-0030's `net.*` keys
//! (`net.listen_port`, `net.relay_mode`, `net.relays`, `net.bulk_on_metered`) and RFC-0034's
//! `rdma.encrypt_untrusted`.
//!
//! RFC-0031 gives no reload classes and no value syntax, so these sections choose them, by one
//! rule each:
//!
//! - **Reload.** `Node` for a value read once when the daemon builds a long-lived component (a
//!   QUIC endpoint, an RDMA poller, a shared-memory segment); `Hot` for a value read each time
//!   it is used (timers, thresholds, weights). RFC-0054 §9.1 gives the meaning of each class.
//! - **Units.** Times are [`Duration`]s and sizes [`ByteSize`]s, because RFC-0054 §3 never
//!   accepts a bare integer where a unit is meant.
//! - **Conditional defaults.** A default that depends on the link or the mode ("2 s LAN, 10 s
//!   WAN") is `computed`, with the RFC's rule as its text; the key is an `Option`, and a value
//!   set by the operator applies to every case.
//! - **Counts in a window** ("3 faults in 60 s") are a [`CountInWindow`] table, and the
//!   `auto`-or-`off` switches a [`Switch`].
//!
//! [`CountInWindow`], [`Backoff`] and [`Switch`] are shared with `cluster`'s sections.

use serde::Deserialize;
use syntropy_config::{ByteSize, ConfigSection, Duration, Fraction, HoldsSecret};

/// "`count` events within `window`": a threshold over a sliding window, such as RFC-0031's
/// `link.demote_after` ("3 faults in 60 s"). Written `{ count = 3, window = "60s" }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(deny_unknown_fields)]
pub struct CountInWindow {
    /// How many events.
    pub count: u32,
    /// Within how long.
    pub window: Duration,
}

/// A retry delay that starts at `initial` and doubles up to `max`, such as RFC-0033's
/// `runner.restart_backoff` ("1 s doubling to 60 s"). Written `{ initial = "1s", max = "60s" }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(deny_unknown_fields)]
pub struct Backoff {
    /// The first delay.
    pub initial: Duration,
    /// The longest delay.
    pub max: Duration,
}

/// A feature used where the node supports it (`auto`), or never (`off`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum Switch {
    /// Used when the node supports it.
    Auto,
    /// Never used.
    Off,
}

/// The QUIC congestion controller (RFC-0031 §4.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum Congestion {
    /// CUBIC.
    Cubic,
    /// BBR.
    Bbr,
}

/// Frame limits and bulk pacing (RFC-0031 §11).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "net", scope = Node, owner = "RFC-0031")]
pub struct NetSection {
    /// The largest frame, capped per transport by `TransportCaps::max_frame` (RFC-0031 §11).
    #[config(reload = Node, default = "64MiB")]
    pub max_frame: ByteSize,
    /// The frame size of the blob plane (RFC-0031 §11).
    #[config(reload = Node, default = "1MiB")]
    pub blob_frame: ByteSize,
    /// Bulk transfers back off when the activation path's p99 exceeds this multiple of its
    /// baseline (RFC-0031 §7.5).
    #[config(reload = Hot, default = 1.2, scope = Cluster, introduced_at = FEATURE_LEVEL_V1)]
    pub bulk_p99_ratio: f64,
    /// The share of the link's β bulk transfers always keep (RFC-0031 §7.5).
    #[config(reload = Hot, default = 0.05, scope = Cluster, introduced_at = FEATURE_LEVEL_V1)]
    pub bulk_floor: Fraction,
}

/// Link timeouts, demotion and the p99 weight (RFC-0031 §11). Cluster scope.
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "link", scope = Cluster, owner = "RFC-0031")]
pub struct LinkSection {
    /// How long a connection attempt waits (RFC-0031 §3).
    #[config(reload = Hot, default = computed("2s on LAN links, 10s on WAN links"),
             introduced_at = FEATURE_LEVEL_V1)]
    pub connect_timeout: Option<Duration>,
    /// How long a path may be silent before traffic fails over (RFC-0031 §7.4).
    #[config(reload = Hot, default = computed("2s on LAN links, 10s on WAN links"),
             introduced_at = FEATURE_LEVEL_V1)]
    pub failover_timeout: Option<Duration>,
    /// Faults within a window that demote a link (ADR-0011 item 7).
    #[config(reload = Hot, default = { count = 3, window = "60s" },
             introduced_at = FEATURE_LEVEL_V1)]
    pub demote_after: CountInWindow,
    /// The floor of the adaptive acknowledgement timeout (RFC-0031 §7.3).
    #[config(reload = Hot, default = "200us", introduced_at = FEATURE_LEVEL_V1)]
    pub ack_timeout_min: Duration,
    /// λ, the weight of a link's p99 in its cost, in latency mode (RFC-0031 §3).
    #[config(reload = Hot, default = computed("1.0 on Wi-Fi or relayed links, 0.25 otherwise"),
             introduced_at = FEATURE_LEVEL_V1)]
    pub p99_weight: Option<f64>,
}

/// QUIC transport settings (RFC-0031 §4.3, §11).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "quic", scope = Node, owner = "RFC-0031")]
pub struct QuicSection {
    /// The congestion controller.
    #[config(reload = Node, default = "cubic")]
    pub congestion: Congestion,
    /// The initial congestion window on LAN paths (RFC-0031 §4.3).
    #[config(reload = Node, default = "256KiB")]
    pub lan_initial_window: ByteSize,
    /// The connection-level flow-control window.
    #[config(reload = Node, default = "64MiB")]
    pub max_conn_window: ByteSize,
}

/// RDMA settings (RFC-0031 §11). `rdma.encrypt_untrusted` is RFC-0034's.
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "rdma", scope = Node, owner = "RFC-0031")]
pub struct RdmaSection {
    /// `auto` uses Thunderbolt 5 and InfiniBand when present, and RoCEv2 only on
    /// `trusted_fabric` interfaces.
    #[config(reload = Node, default = "auto")]
    pub enable: Switch,
    /// `auto` runs one poller per RDMA NIC when at least 2 RDMA channels are open; otherwise
    /// the dispatcher polls.
    #[config(reload = Node, default = "auto")]
    pub poller_thread: Switch,
    /// `auto` uses GPUDirect only after the RFC-0010 device-path probe passes.
    #[config(reload = Node, default = "auto")]
    pub gpudirect: Switch,
}

/// Shared-memory transport settings (RFC-0031 §11).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "shm", scope = Node, owner = "RFC-0031")]
pub struct ShmSection {
    /// Whether same-host stages use shared memory.
    #[config(reload = Node, default = "auto")]
    pub enable: Switch,
}

/// The cluster feature level that introduced every cluster-scope key of these sections
/// (RFC-0032 §6.4, RFC-0054 §2.3 step 9): the first.
pub const FEATURE_LEVEL_V1: u32 = 1;

/// Every section this crate declares (RFC-0031 §11).
pub const SECTIONS: &[&syntropy_config::SectionSchema] = &[
    &NetSection::SCHEMA,
    &LinkSection::SCHEMA,
    &QuicSection::SCHEMA,
    &RdmaSection::SCHEMA,
    &ShmSection::SCHEMA,
];
