// SPDX-License-Identifier: MIT OR Apache-2.0

//! The supervisor against a real runner machine and a real shared-memory region (DIST-E1-T06).
//!
//! The runner's `Lifecycle` reaches `Ready` as it does in production: with no live sequence.
//! Then it admits sequences in place, with no `StateChanged`, and the runner writes the count to
//! the region's header word. The daemon's `RunnerLink` samples the daemon's side of the same
//! region. A `Ready` runner whose header counts live sequences, and whose step counter stops, is
//! killed for a stall at the stall bound. Nothing here builds a `Ready { live_sequences: 2 }` by
//! hand: the only count the supervisor sees is the header's.
//!
//! It lives in `cli` because it needs both `runner` and `daemon`, which are both layer 5.

use std::sync::atomic::{AtomicU64, Ordering};

use syntropy_core::{
    AwakeClock, AwakeTime, Bytes, Clock, Entropy, InstanceId, MonoTime, Nanos, NodeId, PlanEpoch,
    RunnerState, RunnerStateKind,
};
use syntropy_daemon::report::{ReportSink, RunnerFailedReport, RunnerReport};
use syntropy_daemon::runner_link::{Effects, RunnerLink};
use syntropy_daemon::supervisor::{Input, KillReason, Policy, Supervisor};
use syntropy_ipc::cmd::{PROTOCOL, RunnerMsg};
use syntropy_ipc::shm::{Region, ShmLayout, Side};
use syntropy_runner::lifecycle::{Event, Lifecycle};

const S: u64 = 1_000_000_000;
const MS: u64 = 1_000_000;

/// A virtual clock whose two readings advance together.
struct Virtual(AtomicU64);

impl Clock for Virtual {
    fn now(&self) -> MonoTime {
        MonoTime(self.0.load(Ordering::Relaxed))
    }
    fn unix_ms(&self) -> u64 {
        0
    }
}

impl AwakeClock for Virtual {
    fn awake(&self) -> AwakeTime {
        AwakeTime(self.0.load(Ordering::Relaxed))
    }
    fn tick(&self) -> Nanos {
        Nanos(1)
    }
}

/// Fixed bytes: the region's name and the back-off's jitter need no real randomness here.
struct Fixed;

impl Entropy for Fixed {
    fn fill(&self, buf: &mut [u8]) {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = i as u8 ^ 0x5a;
        }
    }
}

#[derive(Default)]
struct Sink(Vec<RunnerReport>);

impl ReportSink for Sink {
    fn report_runner(&mut self, report: RunnerReport) {
        self.0.push(report);
    }
    fn report_runner_failed(&mut self, report: RunnerFailedReport) {
        panic!("no budget is spent here: {report:?}");
    }
}

#[derive(Default)]
struct Kills(Vec<KillReason>);

impl Effects for Kills {
    fn kill(&mut self, reason: KillReason) {
        self.0.push(reason);
    }
    fn spawn(&mut self) {}
    fn reprobe(&mut self) {}
    fn rejected(&mut self, _: Option<RunnerStateKind>, to: RunnerStateKind, _: PlanEpoch) {
        panic!("refused {to:?}");
    }
}

fn policy() -> Policy {
    Policy {
        liveness_timeout: Nanos(3 * S),
        stall_factor: 20.0,
        stall_floor: Nanos(10 * S),
        backoff_initial: Nanos(S),
        backoff_max: Nanos(60 * S),
        budget_count: 5,
        budget_window: Nanos(600 * S),
        reprobe_on_device: true,
        sample_period: Nanos(500 * MS),
    }
}

/// The count the runner writes to the header: its machine's live sequences.
fn live(machine: &Lifecycle) -> u32 {
    match machine.state() {
        RunnerState::Ready { live_sequences, .. }
        | RunnerState::Draining { live_sequences, .. } => *live_sequences,
        _ => 0,
    }
}

#[test]
fn a_real_runner_that_admits_sequences_and_stops_stepping_is_killed_for_a_stall() {
    let layout = ShmLayout::default();
    let daemon_side = Region::create(&layout, [7; 32], PROTOCOL, &Fixed).unwrap();
    let runner_side = Region::open(&daemon_side.handle(), Side::Runner, &layout, PROTOCOL).unwrap();
    let clock = Virtual(AtomicU64::new(S));
    let mut link = RunnerLink::new(
        InstanceId(3),
        NodeId([9; 32]),
        Supervisor::new(policy(), Fixed),
        Sink::default(),
    );
    let mut kills = Kills::default();
    link.handle(clock.now(), Input::Spawned, &mut kills);
    link.handle(
        clock.now(),
        Input::PredictedStep {
            p99: Nanos(100 * MS),
        },
        &mut kills,
    );

    // The runner's machine, from `Init` to `Ready`; each transition's message goes over to the
    // daemon's link.
    let epoch = PlanEpoch { term: 1, seq: 1 };
    let (mut machine, first) = Lifecycle::start(epoch, Bytes(1 << 20));
    let mut messages = vec![first];
    for event in [
        Event::ShardVerified {
            pack: Bytes(1 << 20),
        },
        Event::PackCommitted {
            resident: Bytes(1 << 20),
        },
        Event::Resident,
        Event::WarmPassed,
    ] {
        messages.push(machine.apply(event).unwrap().expect("a transition"));
    }
    for msg in &messages {
        link.on_runner_msg(clock.now(), msg, &mut kills);
    }
    assert!(matches!(
        messages.last(),
        Some(RunnerMsg::StateChanged {
            state: RunnerState::Ready {
                live_sequences: 0,
                ..
            },
            ..
        })
    ));

    // One step of the runner's loop and one sample of the daemon's, every 500 ms.
    let tick = |runner_steps: bool,
                machine: &Lifecycle,
                kills: &mut Kills,
                link: &mut RunnerLink<Fixed, Sink>| {
        clock.0.fetch_add(500 * MS, Ordering::Relaxed);
        runner_side.beat();
        if runner_steps {
            runner_side.step();
        }
        runner_side.set_live_sequences(live(machine));
        link.sample_region(&clock, &daemon_side, kills);
    };

    // `Ready` with no live sequence and no step for 30 s: no stall.
    for _ in 0..60 {
        tick(false, &machine, &mut kills, &mut link);
    }
    assert!(kills.0.is_empty());

    // The runner admits two sequences, in place: no message.
    assert_eq!(machine.apply(Event::Sequences { live: 2 }), Ok(None));
    assert_eq!(live(&machine), 2);
    // While it steps, no stall.
    for _ in 0..60 {
        tick(true, &machine, &mut kills, &mut link);
    }
    assert!(kills.0.is_empty());
    // It stops stepping with sequences live: killed when the silence reaches the stall bound,
    // max(10 s, 20 × 100 ms) = 10 s, which is the 20th sample.
    let mut samples = 0;
    while kills.0.is_empty() {
        tick(false, &machine, &mut kills, &mut link);
        samples += 1;
        assert!(samples <= 20, "no stall kill after {samples} samples");
    }
    assert_eq!(kills.0, [KillReason::Stalled]);
    assert_eq!(samples, 20);
    assert_eq!(daemon_side.live_sequences(), 2);

    // Every report the sink saw was a real transition of the runner's machine.
    let kinds: Vec<_> = link.sink().0.iter().map(|r| r.state).collect();
    assert_eq!(
        kinds,
        [
            RunnerStateKind::Fetching,
            RunnerStateKind::Packing,
            RunnerStateKind::Loading,
            RunnerStateKind::Warming,
            RunnerStateKind::Ready,
        ]
    );
}
