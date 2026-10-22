// SPDX-License-Identifier: MIT OR Apache-2.0

//! How hard Syntropy may lean on a machine: a node's [`ResourcePolicy`] and an instance's
//! [`ResourceOverlay`] (RFC-0001 §11.3, RFC-0054 §4.3–§4.5). Owner: RFC-0054 (types,
//! defaults), RFC-0020 (enforcement).

use serde::{Deserialize, Serialize};

use super::value::{DeviceSelector, Reserve};
use crate::units::Bytes;

/// What OS memory pressure makes Syntropy give back (RFC-0054 §4.3).
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum MemoryYield {
    /// Nothing: the operator accepts the OOM risk. Meant for dedicated machines.
    Never,
    /// The reclaim ladder's first steps at `warn`, more at `critical`. The default.
    #[default]
    UnderPressure,
    /// More of the ladder at `warn`, and a re-plan at once at `critical`.
    Aggressive,
}

/// What running on battery changes (RFC-0054 §4.3).
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum PowerPolicy {
    /// Nothing.
    Performance,
    /// Keep current stages but cap spin-waiting, and steer new stages elsewhere; below
    /// `min_battery_percent`, act as `Battery`. The default.
    #[default]
    Balanced,
    /// Move this node's stages elsewhere, and keep serving the API as ingress.
    Battery,
}

/// Which members a cluster, or one instance, may use (RFC-0054 §4.4). Default:
/// `LanAndInvites` (D4).
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum NetworkScope {
    /// Members on the LAN only. In an instance's policy: no stage on an invite-joined node.
    Lan,
    /// LAN members and members joined through an invite.
    #[default]
    LanAndInvites,
}

/// Which default set `resource.mode` selects (RFC-0054 §4.3).
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum UsageMode {
    /// A machine someone also uses for other things. The default.
    #[default]
    Personal,
    /// A machine that runs Syntropy and little else.
    Dedicated,
}

/// GPUs Syntropy may use: those matching an `allow` selector and no `deny` selector
/// (RFC-0054 §4.3, `resource.gpus.allow` and `resource.gpus.deny`). Default: every probed
/// device.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceFilter {
    /// Selectors a device must match one of.
    pub allow: Vec<DeviceSelector>,
    /// Selectors a device must match none of.
    pub deny: Vec<DeviceSelector>,
}

impl Default for DeviceFilter {
    fn default() -> DeviceFilter {
        DeviceFilter {
            allow: vec![DeviceSelector::Any],
            deny: Vec::new(),
        }
    }
}

const GIB: u64 = 1 << 30;

/// A node's own resolved resource policy (RFC-0001 §11.3, RFC-0054 §4.3). Node keys set it;
/// an instance's [`ResourceOverlay`] may only narrow it.
///
/// [`Default`] is the `personal` column of RFC-0054 §4.3 with `reserve_cores` at its lower
/// value, 1 \[target\], because the rule ("2 if ≥ 8 physical cores, else 1") needs the
/// host's core count. The defaults computed from probed host facts are PROD-E1-T07's. Each field decodes
/// to its default when absent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResourcePolicy {
    /// What OS memory pressure makes Syntropy give back.
    pub memory_yield: MemoryYield,
    /// Cap on `HostPinned` as a fraction of RAM, per-OS default (RFC-0020 §5). `None` means
    /// unset, the macOS default, where the `Unified` cap of RFC-0001 §9.2 applies instead.
    pub max_pinned_fraction: Option<f32>,
    /// What running on battery changes.
    pub power: PowerPolicy,
    /// Which members this node's instances may use.
    pub network_scope: NetworkScope,
    /// Which default set the fields below came from.
    pub mode: UsageMode,
    /// RAM never pinned or allocated by Syntropy.
    pub ram_headroom: Reserve,
    /// VRAM kept free on each discrete GPU (RFC-0001 §9.3 "headroom").
    pub vram_headroom: Reserve,
    /// Cores never given to dispatchers, I/O or the CPU pool.
    pub reserve_cores: u16,
    /// Below this battery percentage, `Balanced` acts as `Battery`. Default 30 \[target\].
    pub min_battery_percent: u8,
    /// The GPUs Syntropy may use.
    pub gpus: DeviceFilter,
    /// Whether Syntropy may raise macOS's wired-memory limit. Never silently (ADR-0014).
    pub macos_raise_wired_limit: bool,
}

impl Default for ResourcePolicy {
    fn default() -> ResourcePolicy {
        ResourcePolicy {
            memory_yield: MemoryYield::UnderPressure,
            max_pinned_fraction: None,
            power: PowerPolicy::Balanced,
            network_scope: NetworkScope::LanAndInvites,
            mode: UsageMode::Personal,
            // `{ min = "6GiB", fraction = 0.10 }` [target], RFC-0054 §4.3.
            ram_headroom: Reserve::min_fraction(Bytes(6 * GIB), 0.10),
            // `{ min = "1GiB", fraction = 0.06 }` [target], RFC-0054 §4.3.
            vram_headroom: Reserve::min_fraction(Bytes(GIB), 0.06),
            reserve_cores: 1,
            // 30 [target], RFC-0054 §4.3.
            min_battery_percent: 30,
            gpus: DeviceFilter::default(),
            macos_raise_wired_limit: false,
        }
    }
}

/// An instance's resource settings: each [`ResourcePolicy`] field an instance may narrow, as
/// an `Option` holding only a value set explicitly in the policy tables or in
/// `EnsureInstance` (RFC-0001 §11.3, RFC-0054 §4.5, amendment A9).
///
/// Each stage node applies it to its own `ResourcePolicy`, and it may only narrow it: more
/// headroom, a lower pinned fraction, a narrower network scope. Unset fields are left out of
/// the serialised form, so only explicitly set fields enter `InstancePolicy::digest`, and
/// node-computed defaults never do.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceOverlay {
    /// See [`ResourcePolicy::memory_yield`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_yield: Option<MemoryYield>,
    /// See [`ResourcePolicy::max_pinned_fraction`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_pinned_fraction: Option<f32>,
    /// See [`ResourcePolicy::power`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power: Option<PowerPolicy>,
    /// See [`ResourcePolicy::network_scope`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_scope: Option<NetworkScope>,
    /// See [`ResourcePolicy::ram_headroom`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ram_headroom: Option<Reserve>,
    /// See [`ResourcePolicy::vram_headroom`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_headroom: Option<Reserve>,
    /// See [`ResourcePolicy::reserve_cores`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserve_cores: Option<u16>,
    /// See [`ResourcePolicy::gpus`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpus: Option<DeviceFilter>,
}
