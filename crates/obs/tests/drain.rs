// SPDX-License-Identifier: MIT OR Apache-2.0

//! The telemetry drain on virtual time (PROD-E8-T01).
//!
//! A `SimClock` drives a `SimTicks`, so every time is exact and nothing waits. The `telemetry`
//! ring is an in-memory sink of bounded capacity, and the daemon's side is a fold over the
//! records it holds, as the daemon's intake folds them.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use syntropy_core::{Clock, DeviceClockSync, MonoTime, Nanos, TickRate, TickSource};
use syntropy_obs::metrics::{
    BUCKETS, CounterIx, GaugeIx, HistIx, HistScale, LabelKey, Layout, LocalMetrics, MetricDesc,
    MetricId, MetricKind, Registry, SeriesId, ThreadKind, Unit,
};
use syntropy_obs::ring::{RecordSink, RingClaims, RingName, RingWriter, SinkError};
use syntropy_obs::telemetry::{TelemetryRecord, decode};
use syntropy_obs::timeline::Timeline;
use syntropy_obs::{DRAIN_PERIOD, Drain};
use syntropy_testkit::sim::{SimClock, SimTicks};

const TOKENS: MetricDesc = MetricDesc {
    id: MetricId(200),
    name: "syntropy_exec_tokens_total",
    kind: MetricKind::Counter,
    unit: Unit::Count,
    labels: &[LabelKey::STAGE],
    help: "Tokens processed.",
    owner: "exec",
};

const STEP_TIME: MetricDesc = MetricDesc {
    id: MetricId(201),
    name: "syntropy_exec_step_seconds",
    kind: MetricKind::Histogram,
    unit: Unit::Nanos,
    labels: &[LabelKey::STAGE],
    help: "Stage step time.",
    owner: "exec",
};

const LIVE: MetricDesc = MetricDesc {
    id: MetricId(202),
    name: "syntropy_exec_live_sequences",
    kind: MetricKind::Gauge,
    unit: Unit::Count,
    labels: &[],
    help: "Live sequences.",
    owner: "exec",
};

/// The `telemetry` ring: `capacity` records, read by the test.
#[derive(Debug, Default)]
struct Ring {
    records: VecDeque<Vec<u8>>,
    capacity: usize,
}

impl RecordSink for Ring {
    fn push(&mut self, record: &[u8]) -> Result<(), SinkError> {
        if record.len() > self.max_record() {
            return Err(SinkError::TooLong);
        }
        if self.records.len() >= self.capacity {
            return Err(SinkError::Full);
        }
        self.records.push_back(record.to_vec());
        Ok(())
    }

    fn max_record(&self) -> usize {
        1024
    }
}

/// What the daemon has folded from the records it read.
#[derive(Debug, Default)]
struct Daemon {
    series: BTreeMap<SeriesId, (MetricId, Vec<String>)>,
    counters: BTreeMap<SeriesId, u64>,
    buckets: BTreeMap<SeriesId, [u64; BUCKETS]>,
    sums: BTreeMap<SeriesId, u64>,
    gauges: BTreeMap<SeriesId, f64>,
    /// Records that named a series not yet announced.
    unknown: usize,
}

impl Daemon {
    /// Reads and folds every record the ring holds.
    fn read(&mut self, ring: &mut Ring) {
        while let Some(bytes) = ring.records.pop_front() {
            match decode(&bytes).unwrap() {
                TelemetryRecord::Series { entries } => {
                    for entry in entries {
                        self.series.insert(entry.id, (entry.metric, entry.labels));
                    }
                }
                TelemetryRecord::Counters { deltas, .. } => {
                    for (series, delta) in deltas {
                        self.check(series);
                        *self.counters.entry(series).or_default() += delta;
                    }
                }
                TelemetryRecord::Histograms { deltas, .. } => {
                    for (series, delta) in deltas {
                        self.check(series);
                        *self.sums.entry(series).or_default() += delta.sum;
                        let buckets = self.buckets.entry(series).or_insert([0; BUCKETS]);
                        for (bucket, count) in delta.buckets {
                            buckets[bucket as usize] += count;
                        }
                    }
                }
                TelemetryRecord::Gauges { values, .. } => {
                    for (series, value) in values {
                        self.check(series);
                        self.gauges.insert(series, value);
                    }
                }
            }
        }
    }

    fn check(&mut self, series: SeriesId) {
        if !self.series.contains_key(&series) {
            self.unknown += 1;
        }
    }

    fn counter(&self, layout: &Layout, counter: CounterIx) -> u64 {
        let series = layout.counter_series(counter);
        self.counters.get(&series).copied().unwrap_or(0)
    }
}

struct Setup {
    clock: SimClock,
    layout: Arc<Layout>,
    tokens: CounterIx,
    step_time: HistIx,
    live: GaugeIx,
    local: Arc<LocalMetrics>,
    drain: Drain<SimTicks, Ring>,
}

fn setup(capacity: usize) -> Setup {
    let mut registry = Registry::new();
    registry.register(&TOKENS).unwrap();
    registry.register(&STEP_TIME).unwrap();
    registry.register(&LIVE).unwrap();
    let tokens = registry.counter(TOKENS.id, &["0"]).unwrap();
    let step_time = registry.histogram(STEP_TIME.id, &["0"]).unwrap();
    let live = registry.gauge(LIVE.id, &[]).unwrap();
    let layout = registry.freeze();
    let clock = SimClock::starting_at(MonoTime(1_000_000), 0);
    let ticks = SimTicks::new(clock.handle(), TickRate::from_hz(3_000_000_000).unwrap());
    let claims = RingClaims::new();
    let writer = RingWriter::claim(
        &claims,
        RingName::Telemetry,
        Ring {
            capacity,
            ..Ring::default()
        },
    );
    let mut drain = Drain::new(ticks, Arc::clone(&layout), writer);
    let local = LocalMetrics::new(&layout, ThreadKind::Dispatcher(0));
    drain.add(Arc::clone(&local));
    Setup {
        clock,
        layout,
        tokens,
        step_time,
        live,
        local,
        drain,
    }
}

/// The daemon's fold of the drained records equals what the hot thread recorded: every
/// counter, every histogram bucket and sum, and every gauge, each series announced before use.
#[test]
fn the_daemon_folds_what_the_thread_recorded() {
    let mut s = setup(1_000);
    let mut daemon = Daemon::default();
    let mut truth = [0u64; BUCKETS];
    let mut sum = 0;
    for n in 0..5_000u64 {
        s.local.add(s.tokens, 1 + n % 3);
        let value = 50 + n * 997 % 10_000_000;
        s.local.record(s.step_time, value);
        truth[HistScale::Nanos.bucket(value)] += 1;
        sum += value;
        s.local.set(s.live, (n % 7) as f64);
        s.clock.advance(Nanos(1_000_000));
        if s.drain.poll().is_some() {
            daemon.read(s.drain.writer_mut().sink_mut());
        }
    }
    s.drain.drain_now();
    daemon.read(s.drain.writer_mut().sink_mut());

    let expected: u64 = (0..5_000u64).map(|n| 1 + n % 3).sum();
    assert_eq!(daemon.counter(&s.layout, s.tokens), expected);
    let step_series = s.layout.histogram_series(s.step_time);
    assert_eq!(daemon.buckets[&step_series], truth);
    assert_eq!(daemon.sums[&step_series], sum);
    assert_eq!(
        daemon.gauges[&s.layout.gauge_series(s.live)],
        (4999 % 7) as f64
    );
    assert_eq!(
        daemon.unknown, 0,
        "a record used a series before its announcement"
    );
    assert_eq!(daemon.series.len(), s.layout.series().len());
    assert_eq!(s.drain.writer().dropped(), 0);
}

/// `poll` drains once per 250 ms of the tick source's time, however often it is called.
#[test]
fn poll_drains_once_per_period() {
    let mut s = setup(1_000);
    assert!(s.drain.poll().is_some(), "the first poll drains");
    let mut drains = 0;
    // 2 s in 1 ms steps.
    for _ in 0..2_000 {
        s.clock.advance(Nanos(1_000_000));
        if s.drain.poll().is_some() {
            drains += 1;
        }
    }
    assert_eq!(drains, 2_000_000_000 / DRAIN_PERIOD.0);
}

/// The criterion "a crash loses at most one drain period of records". The thread records a
/// token every millisecond for about 2 s; the runner's timer polls the drain every 10 ms; the
/// runner dies at an arbitrary instant. What the daemon holds misses only the tokens recorded in
/// the last 250 ms before the crash.
#[test]
fn a_crash_loses_at_most_one_drain_period() {
    for crash_ms in [1_337u64, 1_750, 1_999, 2_249] {
        let mut s = setup(1_000);
        let mut daemon = Daemon::default();
        let mut recorded_at = Vec::new();
        for ms in 0..crash_ms {
            s.local.add(s.tokens, 1);
            recorded_at.push(ms);
            s.clock.advance(Nanos(1_000_000));
            if ms % 10 == 0 && s.drain.poll().is_some() {
                daemon.read(s.drain.writer_mut().sink_mut());
            }
        }
        // The crash: no further drain. The ring's contents survive in shared memory.
        daemon.read(s.drain.writer_mut().sink_mut());
        let delivered = daemon.counter(&s.layout, s.tokens);
        // One token a millisecond, so the tokens lost are the milliseconds since the last drain
        // that reached the ring: at most one period, plus the timer's 10 ms.
        let lost = recorded_at.len() as u64 - delivered;
        let window_ms = DRAIN_PERIOD.0 / 1_000_000 + 10;
        assert!(
            lost <= window_ms,
            "a crash at {crash_ms} ms lost {lost} ms of tokens, over {window_ms} ms"
        );
        assert_eq!(
            &recorded_at[..delivered as usize],
            &(0..delivered).collect::<Vec<_>>()[..]
        );
    }
}

/// The criterion "ring overflow drops and counts, and step times are unaffected". With a ring
/// that holds two records and a daemon that does not read, the drain drops and counts. The hot
/// thread's recording never touches the ring and goes on unchanged. Once the daemon reads again,
/// the deltas the dropped records carried arrive, and the drop count with them.
#[test]
fn a_full_ring_drops_and_counts_and_loses_no_delta() {
    let mut s = setup(2);
    let mut daemon = Daemon::default();
    for _ in 0..20 {
        s.local.add(s.tokens, 1);
        s.local.record(s.step_time, 1_000);
        s.clock.advance(DRAIN_PERIOD);
        // Hot recording does not depend on the ring: these always land in the cells.
        let report = s.drain.poll().unwrap();
        assert!(report.written <= 2);
    }
    let dropped = s.drain.writer().dropped();
    assert!(dropped >= 18, "{dropped}");
    // The daemon reads again, every period, until the drain catches up.
    for _ in 0..20 {
        daemon.read(s.drain.writer_mut().sink_mut());
        s.clock.advance(DRAIN_PERIOD);
        s.drain.poll();
    }
    daemon.read(s.drain.writer_mut().sink_mut());
    assert_eq!(daemon.counter(&s.layout, s.tokens), 20);
    assert_eq!(daemon.sums[&s.layout.histogram_series(s.step_time)], 20_000);
    let drops = daemon.counter(&s.layout, s.layout.telemetry_drops());
    assert_eq!(
        drops,
        s.drain.writer().dropped(),
        "the daemon was told of every drop"
    );
    assert_eq!(daemon.unknown, 0);
}

/// The criterion "on the simulator's device timeline, step and span timestamps and GPU-event
/// mappings match the virtual clock exactly", as far as `obs` reaches before the simulator's
/// device timeline (QUAL-E3-T16) and the step and span records (PROD-E8-T02, PROD-E8-T07): on a
/// `SimTicks` at 3 GHz, a duration, a timestamp, a tick reading placed on the timeline and a
/// device event mapped through a `DeviceClockSync` each equal the virtual clock exactly.
#[test]
fn timestamps_and_device_events_match_the_virtual_clock() {
    let mut clock = SimClock::starting_at(MonoTime(5_000_000_000), 0);
    let ticks = SimTicks::new(clock.handle(), TickRate::from_hz(3_000_000_000).unwrap());
    let timeline = Timeline::new(ticks.clone());
    // A device clock at 1 GHz whose reading 40,000 corresponds to the host's ticks now.
    let sync = DeviceClockSync {
        device_ts: 40_000,
        host_ticks: ticks.ticks(),
        device_hz: 1_000_000_000,
    };
    let start = timeline.start();
    let begun = clock.now();
    clock.advance(Nanos(123_456));
    assert_eq!(timeline.elapsed(start), Nanos(123_456));
    assert_eq!(timeline.now(), MonoTime(5_000_123_456));
    assert_eq!(timeline.at(start), begun);
    assert_eq!(timeline.at(ticks.ticks()), clock.now());
    // A kernel that began 1 µs after the sync and ran 25 µs, in device time.
    let (kernel_begin, kernel_end) = (40_000 + 1_000, 40_000 + 26_000);
    assert_eq!(
        timeline.device_at(kernel_begin, &sync),
        MonoTime(5_000_001_000)
    );
    assert_eq!(
        timeline.device_elapsed(kernel_begin, kernel_end, &sync),
        Nanos(25_000)
    );
}
