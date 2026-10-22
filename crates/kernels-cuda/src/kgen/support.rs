// SPDX-License-Identifier: MIT OR Apache-2.0

//! `support.toml`: the ops the CUDA provider implements, and the instantiations each needs
//! (RFC-0040 §10.2 rule 2).
//!
//! Each `[[op]]` names one [`OpKey`] in its serde form and the tile parameters its kernels are
//! instantiated at. Tiles are the provider's own tuning choices and are never part of the key
//! (RFC-0040 §3.1). The table builds the provider's [`OpRegistry`], so a key is listed once,
//! and every key must belong to exactly one translation unit.

use std::collections::BTreeMap;
use std::fmt;

use serde::Deserialize;
use syntropy_kernels::{OpKey, OpRegistry, RegistryError};

use super::units::{self, Unit};

/// The table, relative to the crate root.
pub const SUPPORT_PATH: &str = "support.toml";

/// The table as `support.toml` writes it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SupportFile {
    #[serde(default)]
    op: Vec<SupportEntry>,
}

/// One op the provider implements.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupportEntry {
    /// The op.
    pub key: OpKey,
    /// The tile parameters its kernels are instantiated at, one instantiation each. Empty for a
    /// kernel with no tile parameter, which is instantiated once.
    #[serde(default)]
    pub tiles: Vec<u32>,
}

/// The parsed table: the registry, and each unit's entries in `OpKey` order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Support {
    /// Every listed op.
    pub registry: OpRegistry,
    /// Each unit's entries, by unit name; every unit is present, with no entries if none.
    pub by_unit: BTreeMap<&'static str, Vec<SupportEntry>>,
}

/// Why `support.toml` was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SupportError {
    /// Not valid TOML, or not the table's shape.
    Parse(String),
    /// A key listed twice, or with an unknown fusion bit.
    Registry(RegistryError),
    /// No translation unit implements the key.
    NoUnit(OpKey),
    /// A tile parameter listed twice for one key.
    DuplicateTile(OpKey, u32),
}

impl fmt::Display for SupportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SupportError::Parse(e) => write!(f, "{SUPPORT_PATH}: {e}"),
            SupportError::Registry(e) => write!(f, "{SUPPORT_PATH}: {e}"),
            SupportError::NoUnit(key) => {
                write!(f, "{SUPPORT_PATH}: no translation unit implements {key:?}")
            }
            SupportError::DuplicateTile(key, tile) => {
                write!(f, "{SUPPORT_PATH}: tile {tile} listed twice for {key:?}")
            }
        }
    }
}

impl std::error::Error for SupportError {}

/// Parses `support.toml`.
///
/// # Errors
///
/// A [`SupportError`] for bad TOML, a duplicate key or tile, an unknown fusion bit, or a key
/// no unit implements.
pub fn parse(text: &str) -> Result<Support, SupportError> {
    let file: SupportFile = toml::from_str(text).map_err(|e| SupportError::Parse(e.to_string()))?;
    let mut registry = OpRegistry::new();
    let mut by_unit: BTreeMap<&'static str, Vec<SupportEntry>> =
        units::UNITS.iter().map(|u| (u.name, Vec::new())).collect();
    for mut entry in file.op {
        registry
            .register(entry.key.clone())
            .map_err(SupportError::Registry)?;
        let unit: &Unit = units::unit_for(&entry.key.op, entry.key.schemes.first())
            .ok_or_else(|| SupportError::NoUnit(entry.key.clone()))?;
        entry.tiles.sort_unstable();
        if let Some(w) = entry.tiles.windows(2).find(|w| w[0] == w[1]) {
            return Err(SupportError::DuplicateTile(entry.key.clone(), w[0]));
        }
        by_unit
            .get_mut(unit.name)
            .expect("every unit is in the map")
            .push(entry);
    }
    for entries in by_unit.values_mut() {
        entries.sort_by(|a, b| a.key.cmp(&b.key));
    }
    Ok(Support { registry, by_unit })
}

/// The CUDA provider's op registry, from the `support.toml` built into the crate: what the
/// provider's `ops()` will return.
///
/// # Panics
///
/// If the built-in table is invalid, which the crate's tests rule out.
#[must_use]
pub fn registry() -> OpRegistry {
    parse(include_str!("../../support.toml"))
        .expect("the built-in support.toml is valid")
        .registry
}
