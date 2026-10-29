// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node's description: RFC-0001 §5.2's `NodeInfo` and the field types no other task defined
//! ([`HostFingerprint`], [`OsKind`], [`ProtocolRanges`], [`RolePrefs`]; DIST-E2-T10). Owner:
//! RFC-0030 (the record, carried by `ReportNodeInfo`), RFC-0010 (`inventory`).
//!
//! DIST-E4-T14 adds `staged_build`, `install` and `off_lan_reach` (RFC-0052 amendment A3,
//! RFC-0030 §8.5) with serde defaults, and DIST-E9-T03 replaces the cluster log's stand-in with
//! this type. The cluster carries it in deterministic CBOR (ADR-0305), so a newer node's
//! additions must not break an older node's decode: unknown fields are ignored, and an `OsKind`
//! this build does not know decodes as [`OsKind::Other`].
//!
//! This crate guarantees what the encoding's order depends on in these types: struct fields in
//! declaration order and `labels` as a `BTreeMap`, so one value always encodes to the same bytes
//! through one encoder. The canonical-form rules of the cluster's encoder, such as key order
//! across maps, are the cluster crate's (DIST-E9).

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::build::BuildId;
use crate::ids::{Blake3, NodeId};

use super::caps::PowerSource;
use super::device::CpuArch;
use super::inventory::NodeInventory;

/// One node's description (RFC-0001 §5.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeInfo {
    /// The node.
    pub id: NodeId,
    /// The same for every node on one host, so the solver knows which nodes share hardware
    /// (RFC-0010 §1.4).
    pub host: HostFingerprint,
    /// Its operating system.
    pub os: OsKind,
    /// Its CPU architecture.
    pub arch: CpuArch,
    /// The build it runs.
    pub build: BuildId,
    /// The protocol versions it speaks, in RFC-0031 §10's terms, so admission and the solver can
    /// check compatibility before any data channel opens (RFC-0062 §10).
    pub protocol: ProtocolRanges,
    /// Mains or battery. Required: an absent value has no safe default, since mains ranks above
    /// battery in the coordinator election (RFC-0032).
    pub power: PowerSource,
    /// The user's labels, for filter rules (RFC-0054).
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
    /// Which roles it may take.
    #[serde(default)]
    pub roles: RolePrefs,
    /// Its `DeviceInfo`, `TierInfo`, `LinkInfo` and `HostPoolInfo` records (RFC-0010 §2.2).
    #[serde(default)]
    pub inventory: NodeInventory,
}

/// The host's fingerprint: the one BLAKE3 digest RFC-0010 §1.4 defines over the host's sorted
/// raw values. Each daemon computes it (PLC-E1-T11). Until then it is the zero digest,
/// [`HostFingerprint::NOT_COMPUTED`], which never makes two nodes co-located.
///
/// Compare hosts with [`same_host`](Self::same_host), not `==`: every uncomputed fingerprint is
/// equal, so a map keyed by it would group uncomputed nodes as one host.
///
/// RFC-0030 §2.3's per-component hashes, which clone detection compares one by one, are a
/// different record (DIST-E2-T03).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct HostFingerprint(pub Blake3);

impl HostFingerprint {
    /// The zero digest: the fingerprint has not been computed.
    pub const NOT_COMPUTED: HostFingerprint = HostFingerprint(Blake3([0; 32]));

    /// Whether the fingerprint has been computed.
    #[must_use]
    pub fn is_computed(&self) -> bool {
        *self != Self::NOT_COMPUTED
    }

    /// Whether `self` and `other` are the same computed host. Two fingerprints that are not
    /// computed never match, so an uncomputed pair is never treated as co-located.
    #[must_use]
    pub fn same_host(&self, other: &HostFingerprint) -> bool {
        self.is_computed() && self == other
    }
}

/// A node's operating system: ADR-0020's platforms.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OsKind {
    /// Linux.
    Linux,
    /// macOS.
    MacOs,
    /// Windows.
    Windows,
    /// Android.
    Android,
    /// iOS.
    Ios,
    /// A platform this build does not know: what a newer node's platform decodes as here. It is
    /// lossy, so a node never re-encodes a record it decoded with `Other` in place of the original:
    /// a contract on the cluster's encoder (DIST-E9). ADR-0305's cluster feature level keeps a
    /// newer variant from reaching an older node in the log.
    #[serde(other)]
    Other,
}

/// A `[min_supported, current]` range of a version (RFC-0031 §10). `min_supported` is never
/// above `current`; decoding a range that breaks this fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RawRange<T>", into = "RawRange<T>")]
#[serde(bound(
    serialize = "T: Copy + Serialize",
    deserialize = "T: Copy + PartialOrd + Deserialize<'de> + fmt::Display"
))]
pub struct VersionRange<T> {
    min_supported: T,
    current: T,
}

/// [`VersionRange`]'s fields as they travel, before the check.
#[derive(Clone, Copy, Serialize, Deserialize)]
struct RawRange<T> {
    min_supported: T,
    current: T,
}

/// Why a [`VersionRange`] could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvertedRange<T> {
    /// The lowest version claimed.
    pub min_supported: T,
    /// The current version.
    pub current: T,
}

impl<T: fmt::Display> fmt::Display for InvertedRange<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "a version range's min_supported {} is above its current {}",
            self.min_supported, self.current
        )
    }
}

impl<T: fmt::Debug + fmt::Display> std::error::Error for InvertedRange<T> {}

impl<T: Copy + PartialOrd> VersionRange<T> {
    /// The range `[min_supported, current]`.
    ///
    /// # Errors
    ///
    /// [`InvertedRange`] when `min_supported` is above `current`.
    pub fn new(min_supported: T, current: T) -> Result<VersionRange<T>, InvertedRange<T>> {
        if min_supported > current {
            return Err(InvertedRange {
                min_supported,
                current,
            });
        }
        Ok(VersionRange {
            min_supported,
            current,
        })
    }

    /// The lowest version this node still accepts.
    #[must_use]
    pub fn min_supported(&self) -> T {
        self.min_supported
    }

    /// The version this node speaks.
    #[must_use]
    pub fn current(&self) -> T {
        self.current
    }
}

impl<T: Copy + PartialOrd + fmt::Display> TryFrom<RawRange<T>> for VersionRange<T> {
    type Error = String;

    fn try_from(raw: RawRange<T>) -> Result<VersionRange<T>, String> {
        VersionRange::new(raw.min_supported, raw.current).map_err(|e| e.to_string())
    }
}

impl<T: Copy> From<VersionRange<T>> for RawRange<T> {
    fn from(range: VersionRange<T>) -> RawRange<T> {
        RawRange {
            min_supported: range.min_supported,
            current: range.current,
        }
    }
}

/// The protocol versions a node speaks, in RFC-0031 §10's terms and at the widths RFC-0031's
/// `Hello` and frame header use. DIST-E8-T14's `Hello` and QUAL-E17-T07's
/// `ci/protocol-versions.toml` check reuse it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProtocolRanges {
    /// The protocol major: nodes of different majors never connect.
    pub proto_major: u16,
    /// The data plane's wire minor, `[min_supported, current]`.
    pub wire_minor: VersionRange<u8>,
    /// The control plane's feature level, `[min_supported, current]` (RFC-0032 §6.4).
    pub feature_level: VersionRange<u32>,
    /// The 64-bit capability set (RFC-0031 §10).
    pub capabilities: u64,
}

/// Which roles a node may take (RFC-0001 §5.2), from RFC-0054's `node.roles` keys and RFC-0032's
/// coordinator eligibility. An absent field takes its default, `true`. These are the node's
/// preferences only: the coordinator election also rules a mobile OS ineligible by default
/// whatever `may_coordinate` says (RFC-0032's capability class 0; ADR-0304).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct RolePrefs {
    /// May be elected coordinator.
    pub may_coordinate: bool,
    /// May accept requests (`node.roles.ingress`).
    pub may_ingress: bool,
    /// May host pipeline stages (`node.roles.host_stages`).
    pub may_host_stages: bool,
}

impl Default for RolePrefs {
    fn default() -> RolePrefs {
        RolePrefs {
            may_coordinate: true,
            may_ingress: true,
            may_host_stages: true,
        }
    }
}
