// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metrics on hot threads (RFC-0053 §2.1).
//!
//! - [`MetricDesc`]: a metric, declared statically. [`Registry`] registers the metrics and binds
//!   each series' label values at `Warming`, returning the [`CounterIx`], [`HistIx`] and
//!   [`GaugeIx`] handles hot threads record through, and freezes them into a [`Layout`].
//! - [`LocalMetrics`]: one hot thread's cells, written by that thread alone with relaxed loads
//!   and stores, and read by the drain.
//! - [`HistScale`]: the log-linear buckets, [`BUCKETS`] = 256 per histogram.

mod desc;
mod histogram;
mod local;
mod registry;

pub use desc::{LabelKey, MetricDesc, MetricId, MetricKind, ThreadKind, Unit};
pub use histogram::{BUCKETS, HistScale};
pub use local::LocalMetrics;
pub use registry::{
    CounterIx, GaugeIx, HistIx, Layout, RING_DROPPED, Registry, RegistryError, SeriesId, SeriesInfo,
};
