// SPDX-License-Identifier: MIT OR Apache-2.0

//! A measured quantity with its spread and provenance (RFC-0001 §5.1, RFC-0010 §1.3).
//! Every cost-model input is one. Owner: RFC-0010.

use serde::{Deserialize, Serialize};

use crate::quant::QuantScheme;

/// A measured quantity with its spread and provenance (RFC-0001 §5.1).
///
/// The default is a zero value with source [`MeasureSource::Prior`], so a field filled by
/// default is treated as unmeasured and is flagged in plans.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Measured<T> {
    /// Median.
    pub p50: T,
    /// 99th percentile.
    pub p99: T,
    /// Sample count after warm-up is discarded (RFC-0010 §3.3 rule 3).
    pub samples: u32,
    /// When the value was measured, in Unix milliseconds. Kept across restarts (RFC-0010 §5.1).
    pub at_unix_ms: u64,
    /// Where the value came from.
    pub source: MeasureSource,
}

/// Where a [`Measured`] value came from (RFC-0001 §5.1; RFC-0010 §1.3 adds `Derived`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasureSource {
    /// A probe microbenchmark.
    Startup,
    /// Refined from live timings.
    Online,
    /// A spec sheet or class default. Used only until measured, and flagged in plans. The
    /// serde default.
    #[default]
    Prior,
    /// Scaled from a measured neighbour: another quant scheme on the same device and tier,
    /// by bytes per weight. Better than `Prior`, and still listed in
    /// `PlacementPlan::assumptions` (RFC-0010 §1.3).
    Derived {
        /// The neighbour it was scaled from.
        from: DerivedFrom,
    },
}

/// The measured neighbour a [`MeasureSource::Derived`] value was scaled from (RFC-0010 §1.3).
///
/// RFC-0010 names the type without fields. The neighbour shares the derived value's device
/// and tier, so the scheme is enough to find it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DerivedFrom {
    /// The quant scheme whose measured rate was scaled.
    pub scheme: QuantScheme,
}
