// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0033 §12's keys: `recovery.*`, `replan.degraded_min_gain`, `straggler.*`, `runner.*`
//! (RFC-0033 §8.2) and `standby`.
//!
//! RFC-0033 §12 also lists `replan.min_gain`, `replan.payback_horizon` and `replan.max_defer`.
//! RFC-0054 §5.1 gives `replan.*` to RFC-0011, whose key table declares those three, so the
//! PLC lane registers them (RFC-0054 §5.1, note of 2026-10-22).

use serde::Deserialize;
use syntropy_config::{ByteSize, ConfigSection, Duration, Fraction, HoldsSecret, Reserve};
use syntropy_net::config::{Backoff, CountInWindow};

use super::FEATURE_LEVEL_V1;

/// Whether an instance keeps recovery history (RFC-0033 §4.1.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum History {
    /// Always.
    On,
    /// Never.
    Off,
    /// Off when the predicted footprint exceeds the budgets.
    Auto,
}

/// A standby for an instance (RFC-0033 §9.3). Written `"none"`, `"replica"` or
/// `{ warm = { nodes = ["pc"] } }`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum Standby {
    /// No standby.
    None,
    /// The named nodes pre-fetch and pack the shards of the stages they would replace.
    Warm {
        /// The nodes, by name.
        nodes: Vec<String>,
    },
    /// A second instance of the model on disjoint nodes where capacity allows.
    Replica,
}

/// When the supervisor re-probes a device before restarting its runner (RFC-0033 §8.2;
/// ADR-0015 item 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum DeviceReprobe {
    /// On `Device`-class failures.
    OnDeviceFailure,
}

/// Recovery of lost stages (RFC-0033 §3–§6). Mixed scopes, as RFC-0033 §12 gives them.
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "recovery", scope = Node, owner = "RFC-0033")]
pub struct RecoverySection {
    /// Whether the instance keeps recovery history (RFC-0033 §4.1.2).
    #[config(reload = Instance, default = "auto", scope = Instance)]
    pub history: History,
    /// The RAM recovery history may use per runner: min(2 GiB, 5 % of RAM).
    #[config(reload = Hot, default = { fraction = 0.05, max = "2GiB" })]
    pub history_ram: Reserve,
    /// The disk recovery history may use.
    #[config(reload = Hot, default = "16GiB")]
    pub history_disk: ByteSize,
    /// Tokens between recurrent-state checkpoints (RFC-0033 §4.2.1).
    #[config(reload = Instance, default = 1024, scope = Instance)]
    pub state_checkpoint_tokens: u32,
    /// How long a loss waits for confirmation before recovery starts (RFC-0033 §3.1). Computed
    /// from `link.failover_timeout`, so the window never ends before failover could save the
    /// link: 4s with the LAN default. [`confirm_after`](fn@confirm_after) applies the rule.
    #[config(reload = Hot, default = computed("2 x link.failover_timeout"), scope = Cluster,
             introduced_at = FEATURE_LEVEL_V1)]
    pub confirm_after: Option<Duration>,
    /// How long a drain may take (RFC-0033 §5.1).
    #[config(reload = Instance, scope = Instance,
             default = computed("60s in latency mode, 300s in throughput mode"))]
    pub drain_deadline: Option<Duration>,
    /// The p90 sequence length assumed before one is measured (RFC-0033 §5.1).
    #[config(reload = Hot, default = 512, scope = Cluster, introduced_at = FEATURE_LEVEL_V1)]
    pub p90_len_seed: u32,
    /// Prefer a better steady state within this slack of the fastest resume (RFC-0033 §6.2).
    #[config(reload = Hot, default = 0.2, scope = Cluster, introduced_at = FEATURE_LEVEL_V1)]
    pub strategy_slack: Fraction,
}

/// RFC-0033's key inside RFC-0011's `replan.*` (RFC-0054 §5.1).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "replan", shared, scope = Cluster, owner = "RFC-0033")]
pub struct ReplanRecoverySection {
    /// The predicted gain a re-plan needs while the instance is degraded (RFC-0033 §7.2).
    #[config(reload = Hot, default = 0.1, introduced_at = FEATURE_LEVEL_V1)]
    pub degraded_min_gain: Fraction,
}

/// `recovery.confirm_after`'s value: the one set, else 2 × `link.failover_timeout`
/// (RFC-0033 §3.1, §12). `failover_timeout` is the link's effective value, which depends on
/// whether it is a LAN or WAN link.
#[must_use]
pub fn confirm_after(set: Option<Duration>, failover_timeout: Duration) -> Duration {
    set.unwrap_or_else(|| Duration::from_nanos(failover_timeout.nanos().0.saturating_mul(2)))
}

/// Straggler detection (RFC-0033 §8.1).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "straggler", scope = Cluster, owner = "RFC-0033")]
pub struct StragglerSection {
    /// A stage slower than this multiple of its prediction is a straggler.
    #[config(reload = Hot, default = 1.25, introduced_at = FEATURE_LEVEL_V1)]
    pub factor: f64,
    /// The window the slowdown is measured over.
    #[config(reload = Hot, default = "30s", introduced_at = FEATURE_LEVEL_V1)]
    pub window: Duration,
}

/// Runner supervision (RFC-0033 §8.2; RFC-0001 §3.4; ADR-0015 item 5).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "runner", scope = Node, owner = "RFC-0033")]
pub struct RunnerSection {
    /// The interval at which the runner bumps its shared-memory heartbeat.
    #[config(reload = Hot, default = "100ms")]
    pub heartbeat: Duration,
    /// No heartbeat for this long, on the suspend-aware clock, means the runner is hung.
    #[config(reload = Hot, default = "3s")]
    pub liveness_timeout: Duration,
    /// A `Ready` runner with live sequences and no step progress for max(floor, factor × the
    /// predicted p99 step) is stalled.
    #[config(reload = Hot, default = 20.0)]
    pub stall_factor: f64,
    /// See `stall_factor`.
    #[config(reload = Hot, default = "10s")]
    pub stall_floor: Duration,
    /// The stall bound from `Fetching` to `Warming`.
    #[config(reload = Hot, default = computed("3 x (shard bytes / measured W_read) + 60s"))]
    pub load_stall: Option<Duration>,
    /// The delay before restarting a failed runner, per (instance, node), with jitter.
    #[config(reload = Hot, default = { initial = "1s", max = "60s" })]
    pub restart_backoff: Backoff,
    /// Failures within a window after which the node reports `ReportRunnerFailed`, and the
    /// repair plan excludes the device.
    #[config(reload = Hot, default = { count = 5, window = "10m" })]
    pub restart_budget: CountInWindow,
    /// No daemon heartbeat for this long, on the suspend-aware clock, puts the runner in
    /// orphan mode.
    #[config(reload = Hot, default = "2s")]
    pub daemon_silence: Duration,
    /// How long an orphaned runner waits for adoption (RFC-0052 §6.5).
    #[config(reload = Hot, default = "30s")]
    pub orphan_grace: Duration,
    /// When the device is re-probed before a restart; if it is gone, the resource graph
    /// changes.
    #[config(reload = Hot, default = "on_device_failure")]
    pub device_reprobe: DeviceReprobe,
}

/// The top-level `standby` instance key (RFC-0033 §9.3, §12).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "", scope = Instance, owner = "RFC-0033")]
pub struct StandbySection {
    /// The instance's standby. Second copies on the predicted absorber are separate and on by
    /// default, capped by `store.standby_max_bytes` (RFC-0041 §9).
    #[config(reload = Instance, default = "none")]
    pub standby: Standby,
}
