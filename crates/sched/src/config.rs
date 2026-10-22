// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `sched.*` configuration section (RFC-0012 §12; RFC-0054 §5.1).
//!
//! Every key is instance scope, as RFC-0012 §12 says of the `sched.*` keys. A node's files
//! cannot set one outside a `[policy."<model>"]` table (RFC-0054 §2.2 Rule 3). The names and
//! defaults are RFC-0012 §12's. RFC-0012 gives no reload classes or types, so this section
//! chooses them:
//!
//! - times are [`Duration`]s, because RFC-0054 §3 never accepts a bare integer where a unit is
//!   meant, so `sched.spin_us = "50us"`;
//! - counts are `u32`;
//! - a key whose RFC-0012 default depends on the instance or the power source has a computed
//!   default and is an `Option`, `None` until the scheduler applies the rule;
//! - a key the head reads each step is `Hot` (RFC-0054 §9.1 lists queue limits as hot); a key
//!   that changes the lane credits or the captured buckets is `Instance`.
//!
//! The section is built from an instance's policy by `config`'s builder, as RFC-0054 §5.4
//! says of every domain crate's section: [`InstanceSection::<SchedSection>::from_policy`]
//! takes the `sched` table, and [`InstanceSection::from_domain`] the `sched.*` entries of
//! `InstancePolicy::domain`. It rejects every invalid value with its RFC-0054 §8 code. A
//! computed default stays `None` until the scheduler applies its rule, such as
//! [`SchedSection::spin`]'s power source.
//!
//! [`InstanceSection::<SchedSection>::from_policy`]: syntropy_config::InstanceSection::from_policy
//! [`InstanceSection::from_domain`]: syntropy_config::InstanceSection::from_domain

use serde::Deserialize;
use syntropy_config::{ConfigSection, Duration};

/// Scheduler settings for one instance (RFC-0012 §12).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "sched", scope = Instance, owner = "RFC-0012")]
pub struct SchedSection {
    /// How long a dispatcher spins before it parks (ADR-0012 item 5; RFC-0012 §3.3).
    #[config(reload = Hot, default = computed("50us on mains power, 10us on battery"))]
    pub spin_us: Option<Duration>,
    /// How far ahead of a predicted frame a parked stage is woken (RFC-0012 §3.3).
    #[config(reload = Hot, default = "300us")]
    pub prewake_us: Duration,
    /// The cap on a prefill chunk's time while an interactive sequence decodes (RFC-0012 §4.3).
    #[config(
        reload = Hot,
        default = computed("0.5 x the active sequence's predicted t_tok, at least 64 tokens of chunk")
    )]
    pub decode_stall_budget: Option<Duration>,
    /// Prefill-lane chunks in flight per instance (RFC-0012 §4.3).
    #[config(reload = Instance, default = computed("D, the instance's stage count"), min = 1)]
    pub prefill_inflight: Option<u32>,
    /// Throughput mode: decode steps per prefill step, so one prefill step per this many
    /// decode steps (RFC-0012 §6.1).
    #[config(reload = Hot, default = computed("G, the instance's group count"), min = 1)]
    pub prefill_share: Option<u32>,
    /// The head's admission queue length (RFC-0012 §10.1).
    #[config(reload = Hot, default = 64)]
    pub max_queued: u32,
    /// Latency mode: the most sequences decoding in one step on dense and RAM-resident MoE
    /// instances; 1 forces serial decoding. NVMe-tiered MoE instances always use 1
    /// (RFC-0012 §4.6). Default 4 \[target\]. The step uses the `Batch { rows }` bucket at 2 or 4
    /// rows, so 4 is the largest value.
    #[config(reload = Instance, default = 4, min = 1, max = 4)]
    pub latency_decode_rows: u32,
    /// The longest scheduler-initiated pause (RFC-0012 §10.3).
    #[config(reload = Hot, default = "30s")]
    pub max_pause: Duration,
}

/// `sched.spin_us` on mains power when no policy sets it: 50 µs \[target\] (ADR-0012 item 5).
pub const SPIN_MAINS: Duration = Duration::from_nanos(50_000);
/// `sched.spin_us` on battery when no policy sets it: 10 µs \[target\] (ADR-0012 item 5).
pub const SPIN_BATTERY: Duration = Duration::from_nanos(10_000);

impl SchedSection {
    /// The spin budget: the policy's value, else the default for the power source.
    #[must_use]
    pub fn spin(&self, on_battery: bool) -> Duration {
        self.spin_us
            .unwrap_or(if on_battery { SPIN_BATTERY } else { SPIN_MAINS })
    }
}

#[cfg(test)]
mod tests;
