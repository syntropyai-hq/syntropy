// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error classes of RFC-0001 §5.12. RFC-0001 §10 defines what each class triggers.
//!
//! Only [`ErrorClass`] is here so far: `DeviceProblem::ProbeCrashed` records one
//! (RFC-0010 §1.2). The `Error` struct and its codes come with the task that builds them.

use serde::{Deserialize, Serialize};

/// What kind of failure an error is (RFC-0001 §5.12). Owner: RFC-0001.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ErrorClass {
    /// Bad client input.
    InvalidRequest,
    /// Op, dtype or feature missing on this device or build.
    Unsupported,
    /// VRAM, pinned RAM, disk, KV pool or admission limit.
    Capacity,
    /// Momentary: would-block, `EAGAIN`, one lost control message.
    Transient,
    /// Transport or network failure on one link.
    Link,
    /// Device fault or sticky driver error.
    Device,
    /// Hash mismatch, truncated or corrupt pack or chunk.
    Integrity,
    /// Parity or canary failure.
    Correctness,
    /// Measured costs no longer match the plan's prediction.
    Drift,
    /// Deadline passed or progress stalled.
    Timeout,
    /// Admission, credential or signature failure.
    Auth,
    /// Client or system cancellation.
    Cancelled,
    /// A bug.
    Internal,
}
