// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runner-channel loop over a real command channel (DIST-E1-T06): every `StateChanged` the
//! supervisor accepts reaches the `ReportSink` once, a refused one never does, and a spent budget
//! reaches `report_runner_failed`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use syntropy_core::{AwakeTime, Bytes, Clock, ErrorClass, Nanos, RunnerState, WarmStep};
use syntropy_ipc::cmd::{Endpoint, Listener, RunnerChannel, connect};

use super::*;
use crate::report::RunnerReport;
use crate::supervisor::Policy;

const S: u64 = 1_000_000_000;

#[derive(Default)]
struct Recorded {
    reports: Vec<RunnerReport>,
    failed: Vec<RunnerFailedReport>,
}

impl ReportSink for Recorded {
    fn report_runner(&mut self, report: RunnerReport) {
        self.reports.push(report);
    }

    fn report_runner_failed(&mut self, report: RunnerFailedReport) {
        self.failed.push(report);
    }
}

#[derive(Default)]
struct Seen {
    kills: Vec<KillReason>,
    spawns: u32,
    reprobes: u32,
    rejected: Vec<RunnerStateKind>,
}

impl Effects for Seen {
    fn kill(&mut self, reason: KillReason) {
        self.kills.push(reason);
    }
    fn spawn(&mut self) {
        self.spawns += 1;
    }
    fn reprobe(&mut self) {
        self.reprobes += 1;
    }
    fn rejected(&mut self, _: Option<RunnerStateKind>, to: RunnerStateKind, _: PlanEpoch) {
        self.rejected.push(to);
    }
}

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

struct Zero;

impl Entropy for Zero {
    fn fill(&self, buf: &mut [u8]) {
        buf.fill(0);
    }
}

fn policy() -> Policy {
    Policy {
        liveness_timeout: Nanos(3 * S),
        stall_factor: 20.0,
        stall_floor: Nanos(10 * S),
        backoff_initial: Nanos(S),
        backoff_max: Nanos(60 * S),
        budget_count: 2,
        budget_window: Nanos(600 * S),
        reprobe_on_device: true,
        sample_period: Nanos(S / 2),
    }
}

fn link() -> RunnerLink<Zero, Recorded> {
    let mut link = RunnerLink::new(
        InstanceId(7),
        NodeId([3; 32]),
        Supervisor::new(policy(), Zero),
        Recorded::default(),
    );
    link.handle(MonoTime(0), Input::Spawned, &mut Seen::default());
    link
}

fn endpoint() -> Endpoint {
    let name = format!("syntropy-daemon-runner-link-{}", std::process::id());
    if cfg!(windows) {
        Endpoint::Pipe(name)
    } else {
        Endpoint::Socket(std::env::temp_dir().join(name).join("cmd.sock"))
    }
}

/// The runner sends its forward path to `Ready`, one state outside §3.3 and a plan delta, then
/// closes the channel. The loop returns `Ok`, and the sink holds each accepted state once, in
/// order, with the instance, node, epoch and kind.
#[test]
fn every_accepted_state_changed_reaches_the_sink_once() {
    let endpoint = endpoint();
    let mut listener = Listener::bind(&endpoint).unwrap();
    let e1 = PlanEpoch { term: 1, seq: 1 };
    let e2 = PlanEpoch { term: 1, seq: 2 };
    let sent = vec![
        (
            RunnerState::Fetching {
                have: Bytes(0),
                need: Bytes(9),
            },
            e1,
        ),
        (
            RunnerState::Packing {
                done: Bytes(0),
                total: Bytes(9),
            },
            e1,
        ),
        (
            RunnerState::Loading {
                done: Bytes(0),
                total: Bytes(9),
            },
            e1,
        ),
        (
            RunnerState::Warming {
                step: WarmStep::Prepare,
            },
            e1,
        ),
        // `Warming → Draining` is no §3.3 edge.
        (
            RunnerState::Draining {
                reason: syntropy_core::DrainReason::Unload,
                deadline: syntropy_core::Deadline::within(Nanos(S)),
                live_sequences: 0,
            },
            e1,
        ),
        (
            RunnerState::Ready {
                live_sequences: 0,
                degraded: None,
            },
            e1,
        ),
        (
            RunnerState::Ready {
                live_sequences: 1,
                degraded: None,
            },
            e2,
        ),
    ];
    let runner = thread::spawn({
        let endpoint = endpoint.clone();
        let sent = sent.clone();
        move || {
            let mut channel = RunnerChannel::new(connect(&endpoint).unwrap());
            for (state, epoch) in sent {
                channel
                    .send(&RunnerMsg::StateChanged { state, epoch })
                    .unwrap();
            }
            channel.send(&RunnerMsg::Pong).unwrap();
        }
    });
    let mut channel = DaemonChannel::new(listener.accept().unwrap());
    let clock = Virtual(AtomicU64::new(S));
    let mut link = link();
    let mut seen = Seen::default();
    serve_channel(&mut channel, &clock, &mut link, &mut seen).expect("the runner closed");
    runner.join().unwrap();
    let want: Vec<RunnerReport> = sent
        .iter()
        .filter(|(s, _)| s.kind() != RunnerStateKind::Draining)
        .map(|(s, e)| RunnerReport {
            instance: InstanceId(7),
            node: NodeId([3; 32]),
            epoch: *e,
            state: s.kind(),
        })
        .collect();
    assert_eq!(link.sink().reports, want);
    assert_eq!(seen.rejected, [RunnerStateKind::Draining]);
    if let Endpoint::Socket(path) = &endpoint {
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}

/// A hang found by sampling kills, the exit reports `Failed` once, and the failure that spends
/// the budget (2 here) reaches `report_runner_failed` with the `Stopped` before it.
#[test]
fn a_spent_budget_reaches_report_runner_failed() {
    let clock = Virtual(AtomicU64::new(0));
    let mut link = link();
    let mut seen = Seen::default();
    for round in 0..2 {
        link.on_runner_msg(
            clock.now(),
            &RunnerMsg::StateChanged {
                state: RunnerState::Fetching {
                    have: Bytes(0),
                    need: Bytes(1),
                },
                epoch: PlanEpoch { term: 1, seq: 1 },
            },
            &mut seen,
        );
        link.sample(&clock, 0, 0, &mut seen);
        for _ in 0..7 {
            clock.0.fetch_add(S / 2, Ordering::Relaxed);
            link.sample(&clock, 0, 0, &mut seen);
        }
        assert_eq!(seen.kills.len(), round + 1);
        link.handle(clock.now(), Input::Exited, &mut seen);
        if round == 0 {
            let at = link.next_deadline().unwrap();
            clock.0.store(at.0, Ordering::Relaxed);
            link.handle(clock.now(), Input::Tick, &mut seen);
            assert_eq!(seen.spawns, 1);
            link.handle(clock.now(), Input::Spawned, &mut seen);
        }
    }
    let kinds: Vec<_> = link.sink().reports.iter().map(|r| r.state).collect();
    assert_eq!(
        kinds,
        [
            RunnerStateKind::Fetching,
            RunnerStateKind::Failed,
            RunnerStateKind::Fetching,
            RunnerStateKind::Failed,
            RunnerStateKind::Stopped,
        ]
    );
    assert_eq!(
        link.sink().failed,
        [RunnerFailedReport {
            instance: InstanceId(7),
            node: NodeId([3; 32]),
            class: ErrorClass::Timeout,
            failures: 2,
        }]
    );
    assert!(link.supervisor().stopped());
}
