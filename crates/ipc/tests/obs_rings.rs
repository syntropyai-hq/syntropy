// SPDX-License-Identifier: MIT OR Apache-2.0

//! `obs`'s writers on a real runner region's rings (PROD-E8-T01; RFC-0053 §1), in `ipc`'s tests
//! because `ipc` may depend on `obs` and not the other way round.
//!
//! `obs` is layer 0 and cannot name `ipc`'s rings, and `ipc` does not depend on `obs`, so the
//! runner joins them; this test does the joining, as the runner will. The region holds the
//! `telemetry` ring and one step ring per dispatcher. The drain is the `telemetry` ring's one
//! writer, and each dispatcher the one writer of its own step ring; the daemon side reads each
//! ring through its own consumer.

#![cfg(any(target_os = "linux", target_vendor = "apple", windows))]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use syntropy_core::{DeviceClockSync, Entropy, MonoTime, Nanos, TickSource};
use syntropy_ipc::cmd::PROTOCOL;
use syntropy_ipc::shm::{Region, RingRole, RingSpec, ShmLayout, Side};
use syntropy_obs::metrics::{
    LabelKey, LocalMetrics, MetricDesc, MetricId, MetricKind, Registry, ThreadKind, Unit,
};
use syntropy_obs::ring::{RecordSink, RingClaims, RingName, RingWriter, SinkError};
use syntropy_obs::telemetry::{TelemetryRecord, decode};
use syntropy_obs::{DRAIN_PERIOD, Drain};
use syntropy_spsc::shared::{SharedPopError, SharedProducer, SharedPushError};

/// A tick source the test moves by hand: one tick per nanosecond.
#[derive(Clone, Debug, Default)]
struct Ticks(Arc<AtomicU64>);

impl Ticks {
    fn advance(&self, by: Nanos) {
        self.0.fetch_add(by.0, Ordering::Relaxed);
    }
}

impl TickSource for Ticks {
    fn ticks(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    fn ticks_to_nanos(&self, ticks: u64) -> Nanos {
        Nanos(ticks)
    }

    fn now(&self) -> MonoTime {
        MonoTime(self.ticks())
    }

    fn device_to_ticks(&self, device_ts: u64, sync: &DeviceClockSync) -> u64 {
        sync.to_ticks(device_ts, 1_000_000_000)
    }
}

/// Segment names for the test's regions: global on Apple platforms and Windows, so drawn from
/// this process's ID and a seed per region.
struct Names(u64);

impl Entropy for Names {
    fn fill(&self, buf: &mut [u8]) {
        let mut x =
            (u64::from(std::process::id()) << 32 | self.0).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for byte in buf {
            x ^= x >> 29;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            *byte = (x >> 56) as u8;
        }
    }
}

/// An `ipc` producing end as `obs` sees it: what the runner provides.
struct Producer<'r>(SharedProducer<'r>);

impl RecordSink for Producer<'_> {
    fn push(&mut self, record: &[u8]) -> Result<(), SinkError> {
        self.0.push(record).map_err(|error| match error {
            SharedPushError::Full => SinkError::Full,
            SharedPushError::TooLong { .. } => SinkError::TooLong,
            SharedPushError::Closed | SharedPushError::Corrupt => SinkError::Closed,
        })
    }

    fn max_record(&self) -> usize {
        self.0.layout().slot_bytes() as usize
    }
}

const STEPS: MetricDesc = MetricDesc {
    id: MetricId(300),
    name: "syntropy_exec_steps_total",
    kind: MetricKind::Counter,
    unit: Unit::Count,
    labels: &[LabelKey::STAGE],
    help: "Steps run.",
    owner: "exec",
};

fn ring(role: RingRole, slots: u32, slot_bytes: u32) -> RingSpec {
    RingSpec {
        role,
        slots,
        slot_bytes,
    }
}

/// The region: `telemetry` with `telemetry_slots` slots, and two dispatchers' step rings of 8.
fn region(telemetry_slots: u32) -> (Region, Region, ShmLayout) {
    let layout = ShmLayout {
        rings: vec![
            ring(RingRole::Telemetry, telemetry_slots, 1024),
            ring(RingRole::Step { dispatcher: 0 }, 8, 256),
            ring(RingRole::Step { dispatcher: 1 }, 8, 256),
        ],
    };
    let names = Names(u64::from(telemetry_slots));
    let daemon = Region::create(&layout, [1; 32], PROTOCOL, &names).unwrap();
    let runner = Region::open(&daemon.handle(), Side::Runner, &layout, PROTOCOL).unwrap();
    (daemon, runner, layout)
}

/// The criteria "one producer per ring: each dispatcher owns an SPSC step ring separate from
/// the telemetry ring" and the telemetry codec end to end: two dispatchers write step records
/// into their own rings, the drain writes counters into `telemetry`, and the daemon reads each
/// ring apart.
#[test]
fn each_dispatcher_writes_its_own_step_ring_and_the_drain_writes_telemetry() {
    let (daemon, runner, _) = region(64);
    let claims = RingClaims::new();
    let mut registry = Registry::new();
    registry.register(&STEPS).unwrap();
    let stage0 = registry.counter(STEPS.id, &["0"]).unwrap();
    let stage1 = registry.counter(STEPS.id, &["1"]).unwrap();
    let layout = registry.freeze();

    let mut step0 = RingWriter::claim(
        &claims,
        RingName::Step { dispatcher: 0 },
        Producer(runner.producer(RingRole::Step { dispatcher: 0 }).unwrap()),
    );
    let mut step1 = RingWriter::claim(
        &claims,
        RingName::Step { dispatcher: 1 },
        Producer(runner.producer(RingRole::Step { dispatcher: 1 }).unwrap()),
    );
    let telemetry = RingWriter::claim(
        &claims,
        RingName::Telemetry,
        Producer(runner.producer(RingRole::Telemetry).unwrap()),
    );
    let mut drain = Drain::new(Ticks::default(), Arc::clone(&layout), telemetry);
    let cells0 = LocalMetrics::new(&layout, ThreadKind::Dispatcher(0));
    let cells1 = LocalMetrics::new(&layout, ThreadKind::Dispatcher(1));
    drain.add(Arc::clone(&cells0));
    drain.add(Arc::clone(&cells1));

    for step in 0..3u8 {
        assert!(step0.push(&[0, step]));
        cells0.add(stage0, 1);
        assert!(step1.push(&[1, step]));
        cells1.add(stage1, 1);
    }
    let report = drain.drain_now();
    assert_eq!(report.dropped, 0);

    let mut buffer = vec![0; 1024];
    let mut read = |role: RingRole| {
        let mut consumer = daemon.consumer(role).unwrap();
        let mut records = Vec::new();
        loop {
            match consumer.pop_into(&mut buffer) {
                Ok(len) => records.push(buffer[..len].to_vec()),
                Err(SharedPopError::Empty) => return records,
                Err(error) => panic!("{role:?}: {error}"),
            }
        }
    };
    assert_eq!(
        read(RingRole::Step { dispatcher: 0 }),
        [vec![0, 0], vec![0, 1], vec![0, 2]]
    );
    assert_eq!(
        read(RingRole::Step { dispatcher: 1 }),
        [vec![1, 0], vec![1, 1], vec![1, 2]]
    );
    let records: Vec<TelemetryRecord> = read(RingRole::Telemetry)
        .iter()
        .map(|bytes| decode(bytes).unwrap())
        .collect();
    assert!(matches!(records[0], TelemetryRecord::Series { .. }));
    let counted: Vec<(ThreadKind, u64)> = records
        .into_iter()
        .filter_map(|record| match record {
            TelemetryRecord::Counters { thread, deltas, .. } => {
                Some((thread, deltas.iter().map(|(_, delta)| delta).sum()))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        counted,
        [
            (ThreadKind::Dispatcher(0), 3),
            (ThreadKind::Dispatcher(1), 3)
        ]
    );
}

/// The criterion "ring overflow drops and counts": a full step ring drops its writer's records
/// and the writer counts them in its own cells, and a full `telemetry` ring makes the drain drop
/// and count. Neither write waits.
#[test]
fn full_rings_drop_and_count() {
    let (_daemon, runner, _) = region(2);
    let claims = RingClaims::new();
    let layout = Registry::new().freeze();
    let cells = LocalMetrics::new(&layout, ThreadKind::Dispatcher(0));
    let mut step = RingWriter::claim(
        &claims,
        RingName::Step { dispatcher: 0 },
        Producer(runner.producer(RingRole::Step { dispatcher: 0 }).unwrap()),
    );
    for n in 0..20u8 {
        if !step.push(&[n]) {
            cells.add(layout.step_drops(), 1);
        }
    }
    assert_eq!((step.written(), step.dropped()), (8, 12));

    let ticks = Ticks::default();
    let telemetry = RingWriter::claim(
        &claims,
        RingName::Telemetry,
        Producer(runner.producer(RingRole::Telemetry).unwrap()),
    );
    let mut drain = Drain::new(ticks.clone(), Arc::clone(&layout), telemetry);
    drain.add(Arc::clone(&cells));
    // Nobody reads `telemetry`, and the cells change every period.
    for _ in 0..4 {
        cells.add(layout.step_drops(), 1);
        drain.poll();
        ticks.advance(DRAIN_PERIOD);
    }
    assert_eq!(drain.writer().written(), 2, "the ring holds two");
    assert!(
        drain.writer().dropped() >= 2,
        "{}",
        drain.writer().dropped()
    );
}
