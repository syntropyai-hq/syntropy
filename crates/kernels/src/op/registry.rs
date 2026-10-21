// SPDX-License-Identifier: MIT OR Apache-2.0

//! The op registry a provider exposes through `ops()` (RFC-0001 §5.7; RFC-0040 §3).

use std::collections::BTreeSet;
use std::fmt;

use super::key::OpKey;

/// Every op a provider implements, one [`OpKey`] each (RFC-0001 §5.7; RFC-0040 §3.1).
///
/// Keys are unique and iterate in `OpKey` order, so two registries with the same keys list
/// them the same way. Whether an op runs at a given shape on a given device is `query`'s
/// answer, not the registry's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpRegistry {
    keys: BTreeSet<OpKey>,
}

/// Why [`OpRegistry::register`] refused a key.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// The key is already registered.
    DuplicateKey(OpKey),
    /// The key's `variant` sets a bit no fusion owns (RFC-0040 §3.1).
    UnknownFusion(OpKey),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateKey(key) => write!(f, "op key registered twice: {key:?}"),
            Self::UnknownFusion(key) => {
                write!(f, "op key sets a variant bit no fusion owns: {key:?}")
            }
        }
    }
}

impl std::error::Error for RegistryError {}

impl OpRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `key`.
    ///
    /// # Errors
    ///
    /// [`RegistryError::DuplicateKey`] when `key` is already registered, and
    /// [`RegistryError::UnknownFusion`] when its `variant` sets a bit [`Fusions`] does not
    /// name. The registry is unchanged in both cases.
    ///
    /// [`Fusions`]: super::Fusions
    pub fn register(&mut self, key: OpKey) -> Result<(), RegistryError> {
        if key.fusions().is_none() {
            return Err(RegistryError::UnknownFusion(key));
        }
        if self.keys.contains(&key) {
            return Err(RegistryError::DuplicateKey(key));
        }
        self.keys.insert(key);
        Ok(())
    }

    /// Whether `key` is registered.
    #[must_use]
    pub fn contains(&self, key: &OpKey) -> bool {
        self.keys.contains(key)
    }

    /// The registered keys, in `OpKey` order.
    pub fn iter(&self) -> impl Iterator<Item = &OpKey> {
        self.keys.iter()
    }

    /// How many keys are registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether no key is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}
