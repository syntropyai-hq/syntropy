// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0032 §9's keys: `coordinator.*`, `log.*`, `orphan.*`, `ring.*`, `tp.*` and `actions.*`.

use serde::Deserialize;
use syntropy_config::{ByteSize, ConfigSection, Duration, HoldsSecret};

use super::FEATURE_LEVEL_V1;

/// Elections in a window that start the election backoff, and the backoff itself (RFC-0032
/// §7.3): "3 elections in 60 s → 1 s, doubling to 30 s". Written
/// `{ count = 3, window = "60s", initial = "1s", max = "30s" }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(deny_unknown_fields)]
pub struct ElectionBackoff {
    /// How many elections.
    pub count: u32,
    /// Within how long.
    pub window: Duration,
    /// The first delay before the next claim.
    pub initial: Duration,
    /// The longest delay.
    pub max: Duration,
}

/// When the coordinator takes a log snapshot: after `entries` entries or `interval`, whichever
/// comes first (RFC-0032 §6.6). Written `{ entries = 4096, interval = "15m" }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(deny_unknown_fields)]
pub struct SnapshotEvery {
    /// Entries since the last snapshot.
    pub entries: u32,
    /// Time since the last snapshot.
    pub interval: Duration,
}

/// Where tensor-parallel islands route locally (RFC-0032 §5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum LocalRouter {
    /// Local routing on homogeneous islands after the bitwise check.
    Verify,
    /// Never.
    Never,
}

/// The coordinator lease and election (RFC-0032 §7.1, §7.3). Keyholder authority under
/// RFC-0054 §2.3 step 10.
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "coordinator", scope = Cluster, owner = "RFC-0032")]
pub struct CoordinatorSection {
    /// The coordinator lease on the LAN (RFC-0032 §7.1). RFC-0032 writes it as 4,000 ms.
    #[config(reload = Hot, default = "4s", introduced_at = FEATURE_LEVEL_V1)]
    pub lease_ms: Duration,
    /// The coordinator lease across a WAN. RFC-0032 writes it as 12,000 ms.
    #[config(reload = Hot, default = "12s", introduced_at = FEATURE_LEVEL_V1)]
    pub wan_lease_ms: Duration,
    /// The delay between successive candidates' claims (RFC-0032 §7.3).
    #[config(reload = Hot, default = "300ms", introduced_at = FEATURE_LEVEL_V1)]
    pub claim_stagger: Duration,
    /// How long a claim collects votes.
    #[config(reload = Hot, default = "500ms", introduced_at = FEATURE_LEVEL_V1)]
    pub claim_window: Duration,
    /// Repeated elections start a backoff.
    #[config(reload = Hot, default = { count = 3, window = "60s", initial = "1s", max = "30s" },
             introduced_at = FEATURE_LEVEL_V1)]
    pub backoff: ElectionBackoff,
    /// The operator's preferred coordinators, highest first, by node name.
    #[config(reload = Hot, default = [], introduced_at = FEATURE_LEVEL_V1)]
    pub prefer: Vec<String>,
}

/// The event log (RFC-0032 §6).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "log", scope = Cluster, owner = "RFC-0032")]
pub struct LogSection {
    /// When the coordinator takes a snapshot (RFC-0032 §6.6).
    #[config(reload = Hot, default = { entries = 4096, interval = "15m" },
             introduced_at = FEATURE_LEVEL_V1)]
    pub snapshot_every: SnapshotEvery,
    /// How long the log batches appends before one durable write.
    #[config(reload = Hot, default = "100ms", scope = Node)]
    pub group_commit: Duration,
    /// Entries a follower may lag before it syncs (RFC-0032 §6.7).
    #[config(reload = Hot, default = 256, introduced_at = FEATURE_LEVEL_V1)]
    pub lag_sync: u32,
    /// Entries queued for a follower; beyond it, the follower catches up by snapshot.
    #[config(reload = Hot, default = 4096, scope = Node)]
    pub follower_queue: u32,
    /// How long the coordinator waits for one eligible follower's ack before replying
    /// (RFC-0032 §6.3).
    #[config(reload = Hot, default = "500ms", introduced_at = FEATURE_LEVEL_V1)]
    pub ack_timeout: Duration,
    /// How long the command journal keeps a command after it is applied (RFC-0032 §6.10).
    #[config(reload = Hot, default = "24h", scope = Node)]
    pub command_retention: Duration,
}

/// Adoption of orphaned runners (RFC-0032 §8.3).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "orphan", scope = Cluster, owner = "RFC-0032")]
pub struct OrphanSection {
    /// How long the cluster waits for an orphaned runner's adoption.
    #[config(reload = Hot, default = "10m", introduced_at = FEATURE_LEVEL_V1)]
    pub adopt_deadline: Duration,
}

/// The ring (RFC-0032 §3).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "ring", scope = Cluster, owner = "RFC-0032")]
pub struct RingSection {
    /// Steps in flight around the ring (RFC-0032 §3.2). An instance key, set through
    /// `InstancePolicy.domain`.
    #[config(reload = Instance, scope = Instance,
             default = computed("D, the stage count, in throughput mode; 1 per sequence in latency mode"))]
    pub max_inflight_steps: Option<u32>,
    /// Whether tokens may travel through a relay node (RFC-0032 §3.4).
    #[config(reload = Instance, default = true, introduced_at = FEATURE_LEVEL_V1)]
    pub allow_token_relay: bool,
}

/// Tensor parallelism across nodes (RFC-0032 §5).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "tp", scope = Cluster, owner = "RFC-0032")]
pub struct TpSection {
    /// The largest all-reduce sent in one shot (RFC-0032 §5.3).
    #[config(reload = Instance, default = "1MiB", introduced_at = FEATURE_LEVEL_V1)]
    pub oneshot_max: ByteSize,
    /// How long a tensor-parallel group may take to come up (RFC-0032 §5.6).
    #[config(reload = Hot, default = "30s", introduced_at = FEATURE_LEVEL_V1)]
    pub bringup_timeout: Duration,
    /// Where islands route locally.
    #[config(reload = Instance, default = "verify", introduced_at = FEATURE_LEVEL_V1)]
    pub local_router: LocalRouter,
}

/// The daemon's action loop (RFC-0032 §8.3).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "actions", scope = Node, owner = "RFC-0032")]
pub struct ActionsSection {
    /// The action loop's tick.
    #[config(reload = Hot, default = "250ms")]
    pub tick: Duration,
}
