// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory-pressure levels (RFC-0020 §11.2). The daemon's `MemoryPressure` message carries one to
//! a runner (RFC-0001 §3.5, §10.3), so the type lives in `core`, moved from `tier` (RFC-0001
//! §18.28; DIST-E1-T15). `tier` re-exports it, and its yield ladder uses it. Owner: RFC-0020.

use serde::{Deserialize, Serialize};

/// System memory pressure, ordered from none to worst (RFC-0020 §11.2).
///
/// Variants are only appended, as the command channel's postcard encodes a variant by its index
/// (RFC-0001 §3.5). The derived order follows variant order, so a new level between two others
/// needs a hand-written `Ord` or an `ipc_protocol` major bump.
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
