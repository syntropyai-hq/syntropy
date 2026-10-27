// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hot threads allocate nothing after `Warming` (RFC-0053 Testing; ADR-0012 §6; PROD-E8-T01).
//!
//! The binary's allocator is `testkit`'s `CountingAlloc`, which counts each thread's allocations.
//! The test does `Warming`'s work first: it registers metrics, binds series and builds the
//! thread's cells and its step-ring writer, all of which allocate. Then it records, writes step
//! records and times durations a hundred thousand times each, and panics if the thread allocated
//! even once.

use std::sync::Arc;

use syntropy_core::{MonoTime, Nanos, TickRate};
use syntropy_obs::metrics::{
    LabelKey, LocalMetrics, MetricDesc, MetricId, MetricKind, Registry, ThreadKind, Unit,
};
use syntropy_obs::ring::{RecordSink, RingClaims, RingName, RingWriter, SinkError};
use syntropy_obs::timeline::Timeline;
use syntropy_testkit::guards::{CountingAlloc, alloc_count_this_thread};
use syntropy_testkit::sim::{SimClock, SimTicks};

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

const STEPS: MetricDesc = MetricDesc {
    id: MetricId(100),
    name: "syntropy_exec_steps_total",
    kind: MetricKind::Counter,
    unit: Unit::Count,
    labels: &[LabelKey::STAGE],
    help: "Steps run.",
    owner: "exec",
};

const STEP_TIME: MetricDesc = MetricDesc {
    id: MetricId(101),
    name: "syntropy_exec_step_seconds",
    kind: MetricKind::Histogram,
    unit: Unit::Nanos,
    labels: &[LabelKey::STAGE],
    help: "Stage step time.",
    owner: "exec",
};

const LIVE: MetricDesc = MetricDesc {
    id: MetricId(102),
    name: "syntropy_exec_live_sequences",
    kind: MetricKind::Gauge,
    unit: Unit::Count,
    labels: &[],
    help: "Live sequences.",
    owner: "exec",
};

/// A step ring of fixed capacity, allocated up front: what a shared-memory ring is to a writer.
struct Fixed {
    slots: Vec<[u8; 256]>,
    used: usize,
}

impl RecordSink for Fixed {
    fn push(&mut self, record: &[u8]) -> Result<(), SinkError> {
        let Some(slot) = self.slots.get_mut(self.used) else {
            return Err(SinkError::Full);
        };
        slot[..record.len()].copy_from_slice(record);
        self.used += 1;
        Ok(())
    }

    fn max_record(&self) -> usize {
        256
    }
}

#[test]
fn hot_threads_allocate_nothing_after_warming() {
    // Warming: everything that allocates.
    let mut registry = Registry::new();
    registry.register(&STEPS).unwrap();
    registry.register(&STEP_TIME).unwrap();
    registry.register(&LIVE).unwrap();
    let steps = registry.counter(STEPS.id, &["0"]).unwrap();
    let step_time = registry.histogram(STEP_TIME.id, &["0"]).unwrap();
    let live = registry.gauge(LIVE.id, &[]).unwrap();
    let layout = registry.freeze();
    let local = LocalMetrics::new(&layout, ThreadKind::Dispatcher(0));
    let claims = RingClaims::new();
    let mut step_ring = RingWriter::claim(
        &claims,
        RingName::Step { dispatcher: 0 },
        Fixed {
            slots: vec![[0; 256]; 1024],
            used: 0,
        },
    );
    let mut clock = SimClock::starting_at(MonoTime(0), 0);
    let timeline = Timeline::new(SimTicks::new(clock.handle(), TickRate::NANOS));
    let drops = layout.step_drops();
    let record = [7u8; 256];
    let shared = Arc::clone(&local);

    // Ready: the hot loop.
    let before = alloc_count_this_thread();
    for n in 0..100_000u64 {
        let start = timeline.start();
        clock.advance(Nanos(1_000 + n % 4_096));
        local.add(steps, 1);
        local.record(step_time, timeline.elapsed(start).0);
        local.set(live, (n % 4) as f64);
        if !step_ring.push(&record) {
            local.add(drops, 1);
        }
        let _ = timeline.at(start);
    }
    let allocations = alloc_count_this_thread() - before;
    assert_eq!(
        allocations, 0,
        "a hot thread allocated {allocations} times after Warming"
    );
    drop(shared);
    // 1,024 step records fit; the other 98,976 were dropped and counted.
    assert_eq!(step_ring.dropped(), 100_000 - 1_024);
}
