// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit newtypes (RFC-0001 §5.1). Interfaces never take bare integers for sizes, rates or
//! times.

use serde::{Deserialize, Serialize};

/// A size in bytes.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Bytes(pub u64);

/// A rate in bytes per second: effective (W_eff, β), never peak, unless the field says so.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BytesPerSec(pub f64);

/// A compute rate in floating-point operations per second.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FlopsPerSec(pub f64);

/// A duration in nanoseconds.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Nanos(pub u64);
