// SPDX-License-Identifier: MIT OR Apache-2.0

//! The plain-data forms of the configuration value types the policy objects hold
//! (RFC-0054 §3, §4; OQ-PRD-27). Owner: RFC-0054.
//!
//! `config` parses text into these and validates it: the accepted spellings, the ranges and
//! the scope rules (such as `"self"` only in node-scope values) are `config`'s. Here they are
//! data with the checks a reader of a decoded record needs. A pattern is held as its source
//! text; `config` keeps the compiled regex.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ids::LayerIx;
use crate::resource::device::DeviceKind;
use crate::resource::tier::TierKind;
use crate::units::Bytes;

/// A share in [0, 1] (RFC-0054 §3 `Fraction`). Negative zero is stored as zero. Decoding
/// rejects a value outside the range.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct Fraction(f32);

impl Fraction {
    /// Zero.
    pub const ZERO: Fraction = Fraction(0.0);
    /// One.
    pub const ONE: Fraction = Fraction(1.0);

    /// `value`, if it lies in [0, 1].
    pub fn new(value: f32) -> Option<Fraction> {
        // `+ 0.0` turns -0.0 into 0.0.
        (0.0..=1.0)
            .contains(&value)
            .then_some(Fraction(value + 0.0))
    }

    /// The share as a number in [0, 1].
    pub const fn get(self) -> f32 {
        self.0
    }
}

/// A value outside [0, 1] where a [`Fraction`] is meant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotAFraction(pub f32);

impl fmt::Display for NotAFraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} is not a fraction: it must lie between 0 and 1",
            self.0
        )
    }
}

impl std::error::Error for NotAFraction {}

impl TryFrom<f32> for Fraction {
    type Error = NotAFraction;

    fn try_from(value: f32) -> Result<Fraction, NotAFraction> {
        Fraction::new(value).ok_or(NotAFraction(value))
    }
}

impl From<Fraction> for f32 {
    fn from(fraction: Fraction) -> f32 {
        fraction.0
    }
}

/// Memory kept back from a tier: a floor, a share of capacity and an optional cap
/// (RFC-0054 §3 `Reserve`). It resolves against the tier's physical capacity as
/// `clamp(max(min, fraction × capacity), .., max)` ([`Reserve::resolve`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Reserve {
    /// The floor.
    pub min: Bytes,
    /// The share of capacity, in [0, 1].
    pub fraction: f32,
    /// The cap, if any.
    pub max: Option<Bytes>,
}

impl Reserve {
    /// A reserve of `min` bytes.
    pub const fn bytes(min: Bytes) -> Reserve {
        Reserve {
            min,
            fraction: 0.0,
            max: None,
        }
    }

    /// A reserve of `min` bytes or `fraction` of capacity, whichever is larger.
    pub const fn min_fraction(min: Bytes, fraction: f32) -> Reserve {
        Reserve {
            min,
            fraction,
            max: None,
        }
    }

    /// Whether `fraction` lies in [0, 1] and `max`, if set, is at least `min`.
    pub fn is_valid(&self) -> bool {
        Fraction::new(self.fraction).is_some() && self.max.is_none_or(|max| max >= self.min)
    }

    /// The bytes kept back from a tier of `capacity`: `clamp(max(min, fraction × capacity), ..,
    /// max)`. The share rounds up to a whole byte, since it is memory kept back.
    pub fn resolve(&self, capacity: Bytes) -> Bytes {
        let share = (f64::from(self.fraction) * capacity.0 as f64).ceil();
        // `as` saturates, and turns NaN into zero.
        let wanted = self.min.max(Bytes(share as u64));
        match self.max {
            Some(max) => wanted.min(max),
            None => wanted,
        }
    }
}

/// Selects nodes (RFC-0054 §3 `NodeSelector`): resolved against cluster membership where it
/// is used.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NodeSelector {
    /// Every node: `"*"`.
    Any,
    /// The node reading the value: `"self"`. `config` allows it only in node-scope values,
    /// so it never reaches an `InstancePolicy` (RFC-0054 §3, §4.5).
    This,
    /// The node with this name.
    Name(String),
    /// Nodes whose ID starts with these hex digits, lower case: `"id:3f9a…"`.
    IdPrefix(String),
    /// Nodes whose label `key` has `value`: `"label:gpu=rtx4090"`.
    Label {
        /// The label's name.
        key: String,
        /// The value it must have.
        value: String,
    },
}

/// A GPU API that numbers devices in a [`DeviceSelector`] (RFC-0054 §3).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DeviceApi {
    /// `cuda:<n>`.
    Cuda,
    /// `hip:<n>`.
    Hip,
    /// `metal:<n>`.
    Metal,
    /// `vulkan:<n>`.
    Vulkan,
}

/// Selects devices (RFC-0054 §3 `DeviceSelector`). Ordinals count each API's devices in
/// RFC-0001 §5.1 probe order.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DeviceSelector {
    /// Every device: `"*"`.
    Any,
    /// Every device of a kind: `"kind:dgpu"`.
    Kind(DeviceKind),
    /// The `index`th device of `api`, on the named node or, with no node, on each node:
    /// `"cuda:0"`, `"<node>/cuda:1"`.
    Gpu {
        /// A node name, or `None` for the node reading the value.
        node: Option<String>,
        /// The API whose devices are counted.
        api: DeviceApi,
        /// The device's position among that API's devices.
        index: u16,
    },
    /// The CPU device of NUMA domain `numa`: `"cpu:numa1"`.
    Cpu {
        /// A node name, or `None` for the node reading the value.
        node: Option<String>,
        /// The NUMA domain.
        numa: u16,
    },
}

/// Selects tiers (RFC-0054 §3 `TierSelector`): a tier kind, optionally on a node and
/// numbered, as in `"<node>/nvme:1"`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TierSelector {
    /// A node name, or `None` for every node.
    pub node: Option<String>,
    /// The tier kind.
    pub kind: TierKind,
    /// The tier's position among that node's tiers of this kind, or `None` for all of them.
    pub index: Option<u16>,
}

/// A half-open range of decoder layers, either end open (RFC-0054 §3 `LayerRange`): `"0..20"`,
/// `"40.."`, `"..8"`, or `"7"` as `7..8`. It is resolved against a model's layer count where
/// it is used, so a policy table can name it before the model is known.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct LayerRange {
    /// The first layer, or `None` for the model's first.
    pub start: Option<LayerIx>,
    /// One past the last layer, or `None` for one past the model's last.
    pub end: Option<LayerIx>,
}

/// A configuration key's dotted path, such as `recovery.history` (RFC-0054 §4.5, amendment
/// A4). `config` checks it against the registered keys.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KeyPath(pub String);

impl KeyPath {
    /// The path as written.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for KeyPath {
    fn from(path: &str) -> KeyPath {
        KeyPath(path.to_owned())
    }
}

/// A configuration value as TOML holds it, before a key's schema gives it a type (RFC-0054
/// §4.5 `domain`, amendment A4's `ConfigPatch`). `config` parses it into the key's type.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ConfigValue {
    /// A boolean.
    Bool(bool),
    /// An integer.
    Integer(i64),
    /// A float.
    Float(f64),
    /// A string.
    String(String),
    /// A TOML date-time, as its RFC 3339 text.
    Datetime(String),
    /// An array.
    Array(Vec<ConfigValue>),
    /// A table, by key.
    Table(BTreeMap<String, ConfigValue>),
}
