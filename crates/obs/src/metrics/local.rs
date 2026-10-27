// SPDX-License-Identifier: MIT OR Apache-2.0

//! One thread's metric cells (RFC-0053 §2.1). A hot-thread module: the lint of `cargo xtask ci
//! lint`'s `hot-threads` step rejects `tracing` here.
//!
//! **Single writer.** The thread that builds a [`LocalMetrics`] is its only writer. Each write is
//! a relaxed load and a relaxed store of an `AtomicU64`, with no read-modify-write instruction,
//! so it takes no lock prefix and no exclusive-monitor loop. The drain reads the same atomics
//! from another thread and sees each value whole. A debug build asserts on every write that the
//! writer is the owning thread.
//!
//! **Layout.** A thread's cells are packed into 128-byte lines of its own, so no other thread's
//! cells share a cache line with them; a line is shared only with the drain's reads. A histogram
//! is 256 bucket cells and a sum, about 2 KiB per histogram per thread.
//!
//! **No allocation.** Everything is allocated when the cells are built, at `Warming`. Recording
//! allocates nothing, takes no lock and reads no clock: durations come from the caller, timed
//! through a [`TickSource`](syntropy_core::TickSource).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::desc::ThreadKind;
use super::histogram::{BUCKETS, HistScale};
use super::registry::{CounterIx, GaugeIx, HistIx, Layout};

/// Cells per 128-byte line.
const PER_LINE: usize = 16;

/// Sixteen cells on a cache line of their own.
#[repr(C, align(128))]
struct Line([AtomicU64; PER_LINE]);

impl Line {
    fn new() -> Line {
        Line(std::array::from_fn(|_| AtomicU64::new(0)))
    }
}

/// One histogram's cells.
#[repr(C, align(128))]
struct Hist {
    buckets: [AtomicU64; BUCKETS],
    sum: AtomicU64,
    scale: HistScale,
}

/// One thread's metric cells. Built once per hot thread at `Warming`, on that thread, and shared
/// with the drain.
pub struct LocalMetrics {
    thread: ThreadKind,
    layout: Arc<Layout>,
    counters: Box<[Line]>,
    gauges: Box<[Line]>,
    hists: Box<[Hist]>,
    #[cfg(debug_assertions)]
    owner: std::thread::ThreadId,
}

impl std::fmt::Debug for LocalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalMetrics")
            .field("thread", &self.thread)
            .field("counters", &self.layout.counters().len())
            .field("histograms", &self.hists.len())
            .field("gauges", &self.layout.gauges().len())
            .finish_non_exhaustive()
    }
}

impl LocalMetrics {
    /// Cells for every series of `layout`, owned by the calling thread, which is the only
    /// thread that may record through them.
    #[must_use]
    pub fn new(layout: &Arc<Layout>, thread: ThreadKind) -> Arc<LocalMetrics> {
        let lines = |cells: usize| (0..cells.div_ceil(PER_LINE)).map(|_| Line::new()).collect();
        Arc::new(LocalMetrics {
            thread,
            layout: Arc::clone(layout),
            counters: lines(layout.counters().len()),
            gauges: lines(layout.gauges().len()),
            hists: layout
                .histograms()
                .iter()
                .map(|&(_, scale)| Hist {
                    buckets: std::array::from_fn(|_| AtomicU64::new(0)),
                    sum: AtomicU64::new(0),
                    scale,
                })
                .collect(),
            #[cfg(debug_assertions)]
            owner: std::thread::current().id(),
        })
    }

    /// The kind of thread these cells belong to.
    #[must_use]
    pub fn thread(&self) -> ThreadKind {
        self.thread
    }

    /// The layout the cells were built from.
    #[must_use]
    pub fn layout(&self) -> &Arc<Layout> {
        &self.layout
    }

    /// Adds `value` to counter `counter`. Owning thread only.
    #[inline]
    pub fn add(&self, counter: CounterIx, value: u64) {
        self.check_owner();
        bump(cell(&self.counters, counter.0), value);
    }

    /// Records `value` in histogram `hist`: its bucket from the value's leading zeros, then one
    /// counter add, and the sum. Owning thread only.
    #[inline]
    pub fn record(&self, hist: HistIx, value: u64) {
        self.check_owner();
        let hist = &self.hists[hist.0 as usize];
        bump(&hist.buckets[hist.scale.bucket(value)], 1);
        bump(&hist.sum, value);
    }

    /// Sets gauge `gauge` to `value`. Owning thread only.
    ///
    /// A gauge series has one writing thread. `TelemetryRecord::Gauges` names no thread, so the
    /// drain reports the latest value it reads from each thread's cells, and two threads that set
    /// one series would overwrite each other's report.
    #[inline]
    pub fn set(&self, gauge: GaugeIx, value: f64) {
        self.check_owner();
        cell(&self.gauges, gauge.0).store(value.to_bits(), Ordering::Relaxed);
    }

    /// Counter slot `slot`'s value, for the drain.
    pub(crate) fn counter_value(&self, slot: usize) -> u64 {
        self.counters[slot / PER_LINE].0[slot % PER_LINE].load(Ordering::Relaxed)
    }

    /// Gauge slot `slot`'s value as bits, for the drain.
    pub(crate) fn gauge_bits(&self, slot: usize) -> u64 {
        self.gauges[slot / PER_LINE].0[slot % PER_LINE].load(Ordering::Relaxed)
    }

    /// Histogram slot `slot`'s bucket `bucket`, for the drain.
    pub(crate) fn bucket_value(&self, slot: usize, bucket: usize) -> u64 {
        self.hists[slot].buckets[bucket].load(Ordering::Relaxed)
    }

    /// Histogram slot `slot`'s sum, for the drain.
    pub(crate) fn sum_value(&self, slot: usize) -> u64 {
        self.hists[slot].sum.load(Ordering::Relaxed)
    }

    /// In a debug build, asserts that the calling thread owns the cells: a second writer would
    /// lose increments, since a write is a load and a store, not one atomic step.
    #[inline]
    fn check_owner(&self) {
        #[cfg(debug_assertions)]
        assert_eq!(
            std::thread::current().id(),
            self.owner,
            "{:?} metrics written from a thread that does not own them",
            self.thread
        );
    }
}

/// Counter cell `slot`.
#[inline]
fn cell(lines: &[Line], slot: u32) -> &AtomicU64 {
    let slot = slot as usize;
    &lines[slot / PER_LINE].0[slot % PER_LINE]
}

/// The single writer's add: a relaxed load, then a relaxed store.
#[inline]
fn bump(cell: &AtomicU64, value: u64) {
    cell.store(
        cell.load(Ordering::Relaxed).wrapping_add(value),
        Ordering::Relaxed,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::desc::{LabelKey, MetricDesc, MetricId, MetricKind, Unit};
    use crate::metrics::registry::Registry;

    const BYTES: MetricDesc = MetricDesc {
        id: MetricId(20),
        name: "syntropy_io_read_bytes",
        kind: MetricKind::Histogram,
        unit: Unit::Bytes,
        labels: &[LabelKey::TIER],
        help: "",
        owner: "io",
    };

    const LIVE: MetricDesc = MetricDesc {
        id: MetricId(21),
        name: "syntropy_exec_live_sequences",
        kind: MetricKind::Gauge,
        unit: Unit::Count,
        labels: &[],
        help: "",
        owner: "exec",
    };

    #[test]
    fn cells_hold_what_was_recorded() {
        let mut registry = Registry::new();
        registry.register(&BYTES).unwrap();
        registry.register(&LIVE).unwrap();
        let hist = registry.histogram(BYTES.id, &["Nvme"]).unwrap();
        let gauge = registry.gauge(LIVE.id, &[]).unwrap();
        let layout = registry.freeze();
        let drops = layout.step_drops();
        let local = LocalMetrics::new(&layout, ThreadKind::Io(0));
        local.add(drops, 3);
        local.add(drops, 4);
        local.record(hist, 1 << 20);
        local.record(hist, 1 << 20);
        local.record(hist, 10);
        local.set(gauge, 2.5);
        assert_eq!(local.counter_value(drops.0 as usize), 7);
        assert_eq!(local.bucket_value(0, 80), 2);
        assert_eq!(local.bucket_value(0, 0), 1);
        assert_eq!(local.sum_value(0), (2 << 20) + 10);
        assert_eq!(f64::from_bits(local.gauge_bits(0)), 2.5);
    }

    /// A debug build refuses a second writer.
    #[test]
    #[cfg(debug_assertions)]
    fn a_second_writer_fails_the_debug_assertion() {
        let layout = Registry::new().freeze();
        let local = LocalMetrics::new(&layout, ThreadKind::Dispatcher(0));
        let other = Arc::clone(&local);
        let result = std::thread::spawn(move || other.add(other.layout().step_drops(), 1)).join();
        assert!(result.is_err(), "the second writer was not refused");
    }
}
