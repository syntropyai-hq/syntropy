// SPDX-License-Identifier: MIT OR Apache-2.0

//! The telemetry drain (RFC-0053 §1, §2.1; RFC-0001 §13).
//!
//! Every [`DRAIN_PERIOD`] (250 ms) the drain reads every thread's cells, computes what changed
//! since its last read, and writes [`TelemetryRecord`]s into the `telemetry` ring through its
//! [`RingWriter`]. Hot threads never take part: they only write their own cells.
//!
//! **Who runs it.** RFC-0053 puts the drain on the runner's `net` runtime. `obs` is layer 0 and
//! cannot depend on `net` or `tokio`, so the drain is a value the runner drives: its `net`
//! runtime calls [`Drain::poll`] from a timer, at least once per period. `poll` does nothing
//! until a period has passed on the drain's [`TickSource`], so a timer that fires early or often
//! changes nothing. The drain reads no other clock.
//!
//! **What a crash loses.** Cells only grow, and the drain sends the difference from its last
//! read that reached the ring. A runner that dies loses only what changed since its last drain:
//! at most one period, 250 ms, while the ring has room.
//!
//! **A full ring.** A record that does not fit is dropped and counted in [`Layout::telemetry_drops`],
//! which the drain sends with the next records that fit. The deltas a dropped record carried are
//! not lost: the drain keeps its last read for those series, so the next drain sends them again.
//! Hot threads never see the ring, so its state never touches a step.
//!
//! **Order.** The first drain announces every series ([`TelemetryRecord::Series`]) before any
//! record uses it, and sends no deltas until every announcement is in the ring.

use std::sync::Arc;

use smallvec::SmallVec;
use syntropy_core::{MonoTime, Nanos, TickSource};

use crate::metrics::{BUCKETS, Layout, LocalMetrics, SeriesId, ThreadKind};
use crate::ring::{RecordSink, RingWriter};
use crate::telemetry::{HistDelta, SeriesEntry, TelemetryRecord, encode};

/// The drain's period (RFC-0001 §13): 250 ms \[target\].
pub const DRAIN_PERIOD: Nanos = Nanos(250_000_000);

/// Counters or gauges per record.
const PER_RECORD: usize = 16;

/// What one drain did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DrainReport {
    /// Records written to the ring.
    pub written: u32,
    /// Records the ring dropped.
    pub dropped: u32,
}

/// One part of one histogram's change: the delta to send, and the reads it covers, which become
/// the drain's last read once the record that carries it is in the ring.
struct Part {
    slot: usize,
    series: SeriesId,
    delta: HistDelta,
    buckets: Vec<(usize, u64)>,
    /// The sum read, on the part that carries the sum's change.
    sum: Option<u64>,
}

/// One thread's cells and the drain's last read of them that reached the ring.
struct Source {
    metrics: Arc<LocalMetrics>,
    counters: Vec<u64>,
    sums: Vec<u64>,
    buckets: Vec<[u64; BUCKETS]>,
    gauges: Vec<u64>,
}

/// The telemetry drain of one runner. See the module documentation.
pub struct Drain<T: TickSource, S: RecordSink> {
    ticks: T,
    layout: Arc<Layout>,
    writer: RingWriter<S>,
    sources: Vec<Source>,
    /// Series announced so far, in `SeriesId` order.
    announced: usize,
    /// Telemetry drops the daemon has been sent.
    drops_sent: u64,
    last: Option<MonoTime>,
    buffer: Vec<u8>,
}

impl<T: TickSource, S: RecordSink> std::fmt::Debug for Drain<T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drain")
            .field("sources", &self.sources.len())
            .field("announced", &self.announced)
            .field("last", &self.last)
            .finish_non_exhaustive()
    }
}

impl<T: TickSource, S: RecordSink> Drain<T, S> {
    /// The drain of `layout`'s series, writing through `writer`, the `telemetry` ring's one
    /// writer, on `ticks`' timeline.
    pub fn new(ticks: T, layout: Arc<Layout>, writer: RingWriter<S>) -> Drain<T, S> {
        Drain {
            ticks,
            layout,
            writer,
            sources: Vec::new(),
            announced: 0,
            drops_sent: 0,
            last: None,
            buffer: Vec::new(),
        }
    }

    /// Adds a thread's cells, at `Warming`. Its first drain sends everything recorded so far.
    pub fn add(&mut self, metrics: Arc<LocalMetrics>) {
        let layout = &self.layout;
        self.sources.push(Source {
            counters: vec![0; layout.counters().len()],
            sums: vec![0; layout.histograms().len()],
            buckets: vec![[0; BUCKETS]; layout.histograms().len()],
            gauges: (0..layout.gauges().len())
                .map(|slot| metrics.gauge_bits(slot))
                .collect(),
            metrics,
        });
    }

    /// The `telemetry` ring's writer.
    #[must_use]
    pub fn writer(&self) -> &RingWriter<S> {
        &self.writer
    }

    /// The `telemetry` ring's writer, for tests that read its sink.
    pub fn writer_mut(&mut self) -> &mut RingWriter<S> {
        &mut self.writer
    }

    /// Drains if a period has passed since the last drain, or if none has run; `None` otherwise.
    pub fn poll(&mut self) -> Option<DrainReport> {
        let now = self.ticks.now();
        if let Some(last) = self.last
            && now.saturating_duration_since(last) < DRAIN_PERIOD
        {
            return None;
        }
        Some(self.drain_now())
    }

    /// Drains now, whatever the time.
    pub fn drain_now(&mut self) -> DrainReport {
        let at = self.ticks.now();
        self.last = Some(at);
        let mut report = DrainReport::default();
        if !self.announce(&mut report) {
            self.flush_drops(at, &mut report);
            return report;
        }
        for at_source in 0..self.sources.len() {
            self.counters(at_source, at, &mut report);
            self.histograms(at_source, at, &mut report);
        }
        self.gauges(at, &mut report);
        self.flush_drops(at, &mut report);
        report
    }

    /// Writes one record; whether it went in.
    fn write(&mut self, record: &TelemetryRecord, report: &mut DrainReport) -> bool {
        // A record of these types always serializes; a failure would be a bug, counted as a drop.
        let fits = encode(record, &mut self.buffer).is_ok() && self.writer.push(&self.buffer);
        if fits {
            report.written += 1;
        } else {
            report.dropped += 1;
        }
        fits
    }

    /// Announces the series not yet announced; whether all are.
    fn announce(&mut self, report: &mut DrainReport) -> bool {
        let max = self.writer.max_record();
        while self.announced < self.layout.series().len() {
            let mut entries: SmallVec<[SeriesEntry; 8]> = SmallVec::new();
            let mut size = 4;
            for info in &self.layout.series()[self.announced..] {
                let bytes = 12
                    + info
                        .labels
                        .iter()
                        .map(|label| label.len() + 5)
                        .sum::<usize>();
                if entries.len() == 8 || (!entries.is_empty() && size + bytes > max) {
                    break;
                }
                size += bytes;
                entries.push(SeriesEntry {
                    id: info.id,
                    metric: info.metric,
                    labels: info.labels.clone(),
                });
            }
            let count = entries.len();
            if !self.write(&TelemetryRecord::Series { entries }, report) {
                return false;
            }
            self.announced += count;
        }
        true
    }

    fn counters(&mut self, at_source: usize, at: MonoTime, report: &mut DrainReport) {
        let source = &self.sources[at_source];
        let thread = source.metrics.thread();
        let changed: Vec<(usize, u64)> = (0..self.layout.counters().len())
            .filter_map(|slot| {
                let now = source.metrics.counter_value(slot);
                (now != source.counters[slot]).then_some((slot, now))
            })
            .collect();
        for chunk in changed.chunks(PER_RECORD) {
            let source = &self.sources[at_source];
            let deltas = chunk
                .iter()
                .map(|&(slot, now)| {
                    (
                        self.layout.counters()[slot],
                        now.wrapping_sub(source.counters[slot]),
                    )
                })
                .collect();
            let record = TelemetryRecord::Counters { at, thread, deltas };
            if self.write(&record, report) {
                let source = &mut self.sources[at_source];
                for &(slot, now) in chunk {
                    source.counters[slot] = now;
                }
            }
        }
    }

    fn histograms(&mut self, at_source: usize, at: MonoTime, report: &mut DrainReport) {
        let max = self.writer.max_record();
        let thread = self.sources[at_source].metrics.thread();
        // Each histogram's change, split so that no delta alone is larger than a record.
        let mut pending: Vec<Part> = Vec::new();
        for (slot, &(series, _)) in self.layout.histograms().iter().enumerate() {
            let source = &self.sources[at_source];
            let sum = source.metrics.sum_value(slot);
            let mut changed = Vec::new();
            for bucket in 0..BUCKETS {
                let now = source.metrics.bucket_value(slot, bucket);
                if now != source.buckets[slot][bucket] {
                    changed.push((bucket, now));
                }
            }
            if changed.is_empty() && sum == source.sums[slot] {
                continue;
            }
            // 11 bytes per bucket at most, and the sum and lengths in 24.
            let per_delta = ((max.saturating_sub(48)) / 11).clamp(1, BUCKETS);
            let mut first = true;
            let parts: Vec<&[(usize, u64)]> = if changed.is_empty() {
                vec![&[]]
            } else {
                changed.chunks(per_delta).collect()
            };
            for part in parts {
                let delta = HistDelta {
                    // The sum's change travels with the first part.
                    sum: if first {
                        sum.wrapping_sub(source.sums[slot])
                    } else {
                        0
                    },
                    buckets: part
                        .iter()
                        .map(|&(bucket, now)| {
                            (bucket as u8, now.wrapping_sub(source.buckets[slot][bucket]))
                        })
                        .collect(),
                };
                pending.push(Part {
                    slot,
                    series,
                    delta,
                    buckets: part.to_vec(),
                    sum: first.then_some(sum),
                });
                first = false;
            }
        }
        // Pack the deltas into records of at most four, and of at most `max` bytes.
        let mut start = 0;
        while start < pending.len() {
            let mut end = start;
            let mut size = 24;
            while end < pending.len() && end - start < 4 {
                let bytes = 24 + pending[end].delta.buckets.len() * 11;
                if end > start && size + bytes > max {
                    break;
                }
                size += bytes;
                end += 1;
            }
            let deltas = pending[start..end]
                .iter()
                .map(|part| (part.series, part.delta.clone()))
                .collect();
            let record = TelemetryRecord::Histograms { at, thread, deltas };
            if self.write(&record, report) {
                let source = &mut self.sources[at_source];
                for part in &pending[start..end] {
                    for &(bucket, now) in &part.buckets {
                        source.buckets[part.slot][bucket] = now;
                    }
                    if let Some(sum) = part.sum {
                        source.sums[part.slot] = sum;
                    }
                }
            }
            start = end;
        }
    }

    fn gauges(&mut self, at: MonoTime, report: &mut DrainReport) {
        let mut changed: Vec<(usize, usize, u64)> = Vec::new();
        for (at_source, source) in self.sources.iter().enumerate() {
            for slot in 0..self.layout.gauges().len() {
                let bits = source.metrics.gauge_bits(slot);
                if bits != source.gauges[slot] {
                    changed.push((at_source, slot, bits));
                }
            }
        }
        for chunk in changed.chunks(PER_RECORD) {
            let values = chunk
                .iter()
                .map(|&(_, slot, bits)| (self.layout.gauges()[slot], f64::from_bits(bits)))
                .collect();
            if self.write(&TelemetryRecord::Gauges { at, values }, report) {
                for &(at_source, slot, bits) in chunk {
                    self.sources[at_source].gauges[slot] = bits;
                }
            }
        }
    }

    /// Sends the telemetry drops not yet sent, as the drain's own counter.
    fn flush_drops(&mut self, at: MonoTime, report: &mut DrainReport) {
        if self.announced == 0 {
            return;
        }
        let dropped = self.writer.dropped();
        if dropped == self.drops_sent {
            return;
        }
        let series = self.layout.counters()[self.layout.telemetry_drops().0 as usize];
        let mut deltas = SmallVec::new();
        deltas.push((series, dropped - self.drops_sent));
        let record = TelemetryRecord::Counters {
            at,
            thread: ThreadKind::Drain,
            deltas,
        };
        if self.write(&record, report) {
            self.drops_sent = dropped;
        }
    }
}
