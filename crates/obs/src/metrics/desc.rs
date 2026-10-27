// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a metric is (RFC-0053 §2.1): its static description, and the threads that record.

use serde::{Deserialize, Serialize};

/// A metric's number, fixed in its [`MetricDesc`]. Unique within a build. Owner: RFC-0053.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MetricId(pub u32);

/// How a metric's values combine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricKind {
    /// A count that only grows; the drain sends deltas.
    Counter,
    /// A value that is set; the drain sends the latest.
    Gauge,
    /// A distribution in [`BUCKETS`](super::BUCKETS) log-linear buckets.
    Histogram,
}

/// A metric's unit. A histogram's unit picks its bucket range (RFC-0053 §2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Unit {
    /// A plain count.
    Count,
    /// Bytes: a histogram covers 2¹⁰ to 2⁴² bytes.
    Bytes,
    /// Nanoseconds: a histogram covers 2⁶ to 2³⁸ ns.
    Nanos,
    /// A dimensionless ratio, for gauges.
    Ratio,
}

/// A label's name. RFC-0053 §2.4 fixes which names are allowed and how many values each may
/// take; the cardinality guard that enforces it is PROD-E8-T06's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabelKey(pub &'static str);

impl LabelKey {
    /// RFC-0001 §13's identity labels.
    pub const NODE: LabelKey = LabelKey("node");
    /// The instance.
    pub const INSTANCE: LabelKey = LabelKey("instance");
    /// The stage.
    pub const STAGE: LabelKey = LabelKey("stage");
    /// The device.
    pub const DEVICE: LabelKey = LabelKey("device");
    /// The memory tier; an NVMe volume is its `Nvme` tier.
    pub const TIER: LabelKey = LabelKey("tier");
    /// The link.
    pub const LINK: LabelKey = LabelKey("link");
    /// The kernel backend.
    pub const BACKEND: LabelKey = LabelKey("backend");
    /// The op class.
    pub const OP_CLASS: LabelKey = LabelKey("op_class");
}

/// A metric, declared statically and registered at `Warming` (RFC-0053 §2.1). Owner: RFC-0053.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MetricDesc {
    /// Its number.
    pub id: MetricId,
    /// Its name, such as `syntropy_exec_step_seconds`.
    pub name: &'static str,
    /// Counter, gauge or histogram.
    pub kind: MetricKind,
    /// Its unit.
    pub unit: Unit,
    /// Its labels' names, in the order a series binds their values.
    pub labels: &'static [LabelKey],
    /// One line of help text.
    pub help: &'static str,
    /// The crate that records it.
    pub owner: &'static str,
}

/// The kind of thread a set of cells belongs to, as telemetry names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ThreadKind {
    /// A stage dispatcher (RFC-0012), by index.
    Dispatcher(u16),
    /// A CPU-pool worker, by index.
    CpuPool(u16),
    /// An I/O thread, by index.
    Io(u16),
    /// The runner's housekeeping thread.
    Housekeeping,
    /// The telemetry drain itself: its own drop counters.
    Drain,
}
