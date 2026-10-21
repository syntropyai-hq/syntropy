// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory-pressure levels, the yield ladder and reacquisition (RFC-0020 §11). This task defines
//! [`PressureLevel`]; the ladder comes in a later task.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// System memory pressure, ordered from none to worst (RFC-0020 §11.2).
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum PressureLevel {
    /// No pressure. The default.
    #[default]
    Normal,
    /// Moderate pressure.
    Moderate,
    /// High pressure.
    High,
    /// Critical pressure.
    Critical,
}
