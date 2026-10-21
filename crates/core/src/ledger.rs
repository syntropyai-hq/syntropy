// SPDX-License-Identifier: MIT OR Apache-2.0

//! The device-ledger grant a daemon gives a runner: how many bytes of each tier the runner may
//! hold (RFC-0001 §3.5, note of 2026-10-22; RFC-0020 §3, §12; OQ-IO-27).
//!
//! It lives in `core` because two crates with no edge between them use it: `ipc` carries it in
//! the daemon's `Init` message, and `tier` checks an `ArenaPlan`'s per-tier bytes against it in
//! `Fetching` (`TIER_LEDGER_EXCEEDED`, RFC-0020 §12). It is plain data; the check is `tier`'s.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ids::TierId;
use crate::units::Bytes;

/// The bytes of each tier a daemon grants one runner (RFC-0001 §3.5; RFC-0020 §3).
///
/// Each tier appears at most once, and the lines are kept in `TierId` order, so two grants of
/// the same bytes compare and encode equal whatever order they were built in. A tier with no
/// line is granted nothing: [`LedgerGrant::granted`] returns 0 for it.
///
/// The encoding is a record with one field, `per_tier`, a list of `(tier, bytes)` pairs. A
/// list with a tier twice is refused on decode, because the ledger check could not tell which
/// figure holds. Unknown fields are ignored, so a later build can add fields.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "Wire", into = "Wire")]
pub struct LedgerGrant {
    per_tier: Vec<(TierId, Bytes)>,
}

/// Why a [`LedgerGrant`] could not be built.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerGrantError {
    /// The tier has more than one line.
    DuplicateTier(TierId),
}

impl fmt::Display for LedgerGrantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LedgerGrantError::DuplicateTier(t) => write!(
                f,
                "tier {} of node {:016x} is granted more than once",
                t.ordinal,
                t.node.short()
            ),
        }
    }
}

impl std::error::Error for LedgerGrantError {}

/// The encoded form of a [`LedgerGrant`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct Wire {
    per_tier: Vec<(TierId, Bytes)>,
}

impl TryFrom<Wire> for LedgerGrant {
    type Error = LedgerGrantError;

    fn try_from(w: Wire) -> Result<Self, Self::Error> {
        LedgerGrant::new(w.per_tier)
    }
}

impl From<LedgerGrant> for Wire {
    fn from(g: LedgerGrant) -> Self {
        Wire {
            per_tier: g.per_tier,
        }
    }
}

impl LedgerGrant {
    /// A grant of `lines`, one `(tier, bytes)` pair per tier, in any order.
    ///
    /// # Errors
    ///
    /// [`LedgerGrantError::DuplicateTier`] when a tier appears twice, even with the same bytes.
    pub fn new(lines: impl IntoIterator<Item = (TierId, Bytes)>) -> Result<Self, LedgerGrantError> {
        let mut per_tier: Vec<(TierId, Bytes)> = lines.into_iter().collect();
        per_tier.sort_by_key(|&(t, _)| t);
        if let Some(w) = per_tier.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(LedgerGrantError::DuplicateTier(w[0].0));
        }
        Ok(Self { per_tier })
    }

    /// The bytes granted on `tier`; 0 when the grant has no line for it.
    #[must_use]
    pub fn granted(&self, tier: TierId) -> Bytes {
        self.per_tier
            .binary_search_by_key(&tier, |&(t, _)| t)
            .map_or(Bytes(0), |i| self.per_tier[i].1)
    }

    /// The lines, in `TierId` order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (TierId, Bytes)> + '_ {
        self.per_tier.iter().copied()
    }

    /// How many tiers have a line.
    #[must_use]
    pub fn len(&self) -> usize {
        self.per_tier.len()
    }

    /// Whether no tier has a line: the runner may hold nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.per_tier.is_empty()
    }
}
