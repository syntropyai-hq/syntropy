// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error classes and codes of RFC-0001 §5.12. RFC-0001 §10 defines what each class
//! triggers.
//!
//! Here so far: [`ErrorClass`], which `DeviceProblem::ProbeCrashed` records (RFC-0010 §1.2);
//! [`ErrorCode`], the `(Domain, u16)` pair; the code tables of the domains whose RFCs publish
//! them ([`codes`]); and the map from each code to its RFC-0050 §12 API error shape ([`api`]).
//! The `Error` struct itself, with its scope, retry hint, context and source, comes with the
//! task that builds it.

use std::fmt;
use std::ops::RangeInclusive;

use serde::{Deserialize, Serialize};

pub mod api;
pub mod codes;

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

/// The area that publishes a code (RFC-0001 §5.12, §10.4). Each domain RFC owns the codes in
/// its range. Other domains are added with the tasks that build their tables. Owner: RFC-0001.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Domain {
    /// Discovery, membership and invites: 0x0100–0x01FF (RFC-0030 §10).
    Cluster,
    /// Transport and wire protocol: 0x0200–0x02FF (RFC-0031 §12).
    Net,
    /// Distributed execution and the cluster event log: 0x0300–0x03FF (RFC-0032 §10).
    Dist,
    /// Fault tolerance and elasticity: 0x0400–0x04FF (RFC-0033 §13).
    Recovery,
    /// Security and trust: 0x0500–0x05FF (RFC-0034 §15).
    Security,
}

impl Domain {
    /// The code range the domain's RFC publishes.
    pub const fn range(self) -> RangeInclusive<u16> {
        match self {
            Self::Cluster => 0x0100..=0x01FF,
            Self::Net => 0x0200..=0x02FF,
            Self::Dist => 0x0300..=0x03FF,
            Self::Recovery => 0x0400..=0x04FF,
            Self::Security => 0x0500..=0x05FF,
        }
    }

    /// The RFC section that publishes the domain's codes.
    pub const fn source(self) -> &'static str {
        match self {
            Self::Cluster => "RFC-0030 §10",
            Self::Net => "RFC-0031 §12",
            Self::Dist => "RFC-0032 §10",
            Self::Recovery => "RFC-0033 §13",
            Self::Security => "RFC-0034 §15",
        }
    }
}

/// A stable, machine-readable error code: `(Domain, u16)` (RFC-0001 §5.12). Codes are never
/// reused (RFC-0001 §10.4); [`codes::REGISTRY`] lists the published ones. Owner: RFC-0001
/// (the pair), each domain RFC (its codes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ErrorCode {
    /// The domain that publishes the code.
    pub domain: Domain,
    /// The code inside the domain, as its RFC writes it.
    pub code: u16,
}

impl ErrorCode {
    /// The code `code` of `domain`.
    pub const fn new(domain: Domain, code: u16) -> Self {
        Self { domain, code }
    }
}

/// Shows the domain and the code as its RFC writes it, e.g. `Net 0x0207`.
impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} 0x{:04X}", self.domain, self.code)
    }
}
