// SPDX-License-Identifier: MIT OR Apache-2.0

//! The supervisor's laptop-sleep rig test (DIST-E1-T06): "a laptop sleep of 60 s causes no runner
//! kill on wake (macOS and Windows rig test)".
//!
//! It runs `daemon`'s supervisor on the production clocks, `io`'s `SystemClock`, sampling as the
//! daemon's shell does, while a thread bumps a heartbeat counter every 100 ms as a runner does.
//! The runner process and its shared-memory header are DIST-E1-T04's wiring, so the counter here
//! is in-process. The suspend is real: the thread, like a runner, stops while the host sleeps.
//!
//! It needs a person and a laptop, so it is ignored by default. To run it on a macOS or a Windows
//! rig:
//!
//! ```text
//! cargo test -p syntropy-cli --test rig_supervisor_sleep -- --ignored --nocapture
//! ```
//!
//! then, within the first 30 s, put the machine to sleep (close the lid, or the OS's Sleep
//! command), wait at least 60 s, and wake it. The test runs for `SYNTROPY_RIG_SECS` seconds
//! (default 240) and passes when the supervisor saw a suspend of at least 55 s and asked for no
//! kill. It fails, saying so, if no suspend was seen, so a run without a sleep cannot pass.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use syntropy_core::{Clock, InstanceId, MonoTime, Nanos, NodeId, PlanEpoch, SuspendDetector};
use syntropy_daemon::report::{ReportSink, RunnerFailedReport, RunnerReport};
use syntropy_daemon::runner_link::{Effects, RunnerLink};
use syntropy_daemon::supervisor::{Input, KillReason, Policy, Supervisor};
use syntropy_io::os::{OsEntropy, SystemClock};
use syntropy_ipc::cmd::RunnerMsg;

struct Sink;

impl ReportSink for Sink {
    fn report_runner(&mut self, report: RunnerReport) {
        println!("report: {report:?}");
    }
    fn report_runner_failed(&mut self, report: RunnerFailedReport) {
        panic!("the budget was spent: {report:?}");
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
    fn rejected(
        &mut self,
        _: Option<syntropy_core::RunnerStateKind>,
        to: syntropy_core::RunnerStateKind,
        _: PlanEpoch,
    ) {
        panic!("refused {to:?}");
    }
}

/// RFC-0033 §8.2's defaults, as `Policy::from_section` reads them from a node with no
/// `runner.*` keys.
fn policy() -> Policy {
    let s = 1_000_000_000;
    Policy {
        liveness_timeout: Nanos(3 * s),
        stall_factor: 20.0,
        stall_floor: Nanos(10 * s),
        backoff_initial: Nanos(s),
        backoff_max: Nanos(60 * s),
        budget_count: 5,
        budget_window: Nanos(600 * s),
        reprobe_on_device: true,
        sample_period: Nanos(s / 2),
    }
}

#[test]
#[ignore = "rig: put the host to sleep for 60 s while it runs (macOS and Windows; see the module docs)"]
fn a_laptop_sleep_of_60_s_kills_no_runner() {
    let secs: u64 = std::env::var("SYNTROPY_RIG_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(240);
    let clock = SystemClock::new();
    let mut link = RunnerLink::new(
        InstanceId(1),
        NodeId([0; 32]),
        Supervisor::new(policy(), OsEntropy::new()),
        Sink,
    );
    let mut kills = Kills::default();
    link.handle(clock.now(), Input::Spawned, &mut kills);
    // The runner's forward path to `Ready` with one live sequence, which also arms stall
    // detection; its step counter moves with the heartbeat.
    let epoch = PlanEpoch { term: 1, seq: 1 };
    for state in [
        syntropy_core::RunnerState::Fetching {
            have: syntropy_core::Bytes(0),
            need: syntropy_core::Bytes(1),
        },
        syntropy_core::RunnerState::Packing {
            done: syntropy_core::Bytes(0),
            total: syntropy_core::Bytes(1),
        },
        syntropy_core::RunnerState::Loading {
            done: syntropy_core::Bytes(0),
            total: syntropy_core::Bytes(1),
        },
        syntropy_core::RunnerState::Warming {
            step: syntropy_core::WarmStep::Prepare,
        },
        syntropy_core::RunnerState::Ready {
            live_sequences: 1,
            degraded: None,
        },
    ] {
        link.on_runner_msg(
            clock.now(),
            &RunnerMsg::StateChanged { state, epoch },
            &mut kills,
        );
    }
    link.handle(
        clock.now(),
        Input::PredictedStep {
            p99: Nanos(50_000_000),
        },
        &mut kills,
    );
    let beats = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let runner = thread::spawn({
        let beats = Arc::clone(&beats);
        let stop = Arc::clone(&stop);
        move || {
            while !stop.load(Ordering::Relaxed) {
                beats.fetch_add(1, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(100));
            }
        }
    });
    // A second detector, sampled with the supervisor, records the longest suspend it saw.
    let mut detector = SuspendDetector::new();
    let mut longest = Nanos(0);
    let start = clock.now();
    let end = start.saturating_add(Nanos(secs * 1_000_000_000));
    println!("sleep the host for at least 60 s within the next 30 s; running {secs} s");
    while clock.now() < end {
        let deadline = link.next_deadline().unwrap_or(MonoTime(0));
        let now = clock.now();
        if deadline > now {
            thread::sleep(Duration::from_nanos(deadline.0 - now.0));
        }
        let n = beats.load(Ordering::Relaxed);
        link.sample(&clock, n, n, &mut kills);
        let suspended = detector.observe_clock(&clock).suspended(clock.tick());
        if suspended > longest {
            longest = suspended;
            println!("suspend seen: {} ms", suspended.0 / 1_000_000);
        }
        assert!(
            kills.0.is_empty(),
            "the supervisor asked for a kill: {:?}",
            kills.0
        );
    }
    stop.store(true, Ordering::Relaxed);
    runner.join().unwrap();
    assert!(
        longest >= Nanos(55_000_000_000),
        "no suspend of 55 s or more was seen (longest {} ms): put the host to sleep for 60 s while the test runs",
        longest.0 / 1_000_000
    );
    println!("passed: a {} s suspend, no kill", longest.0 / 1_000_000_000);
}
