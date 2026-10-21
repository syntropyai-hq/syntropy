// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive pinning (RFC-0020 §6). This task defines the progress types; the pinning loop
//! and its gate come in TIER-E2.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use syntropy_core::Bytes;

/// How far progressive pinning has got (RFC-0020 §3, supporting types).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PinProgress {
    /// Bytes pinned so far.
    pub done: Bytes,
    /// Bytes the plan asks to pin.
    pub target: Bytes,
    /// Where the loop is.
    pub state: PinState,
    /// Estimated time to finish, in milliseconds. Defaults to `None` (no estimate).
    #[serde(default)]
    pub eta_ms: Option<u64>,
}

/// The state of the pinning loop (RFC-0020 §3, §6).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinState {
    /// Pinning is in progress.
    Running,
    /// Every range is at its planned pin class.
    Done,
    /// Pressure or capacity failures lasted `tier.pin_give_up`; the remaining ranges are
    /// `Underpinned`.
    GaveUp {
        /// Bytes left below their planned pin class.
        underpinned: Bytes,
    },
}
