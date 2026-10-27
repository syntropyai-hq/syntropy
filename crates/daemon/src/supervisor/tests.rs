// SPDX-License-Identifier: MIT OR Apache-2.0

//! The supervisor's tests on virtual time (DIST-E1-T06).
//!
//! - A simulated hang, stall and crash each produce RFC-0033 §8.2's action at the configured
//!   threshold, with the `runner.*` defaults read from `RunnerSection` itself.
//! - The back-off doubles from 1 s to 60 s with jitter in [¾ · base, base], and the fifth failure
//!   in 10 min spends the budget.
//! - A 60 s suspend, and a tick late on both clocks, kill nothing.
//! - Reports outside RFC-0001 §3.3, and `Ready → Ready` without a later epoch, are refused.

use syntropy_config::{ConfigSection, Env, Flags, Layers, Os, Registry, ResolvedConfig, Value};
use syntropy_core::{RunnerState, WarmStep};

use super::*;

const S: u64 = 1_000_000_000;
const MS: u64 = 1_000_000;

static REGISTRY: Registry = Registry::new(&[&RunnerSection::SCHEMA]);

/// The policy of a node with no `runner.*` keys set: RFC-0033 §8.2's defaults.
fn policy() -> Policy {
    let flags = Flags::new();
    let env = Env::from_pairs(Os::Linux, std::iter::empty::<(&str, &str)>());
    // `runner.load_stall`'s default is computed from the shard and the disk, which the
    // supervisor does not use; any value lets the section resolve.
    let config = ResolvedConfig::resolve_with(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files: &[],
            env: &env,
            flags: &flags,
        },
        |key| (key.path == "runner.load_stall").then(|| Value::String(String::from("90s"))),
    )
    .expect("the defaults resolve");
    Policy::from_section(&config.section::<RunnerSection>().expect("the section"))
}

/// Entropy that repeats one byte, so the jitter is fixed.
struct Fixed(u8);

impl Entropy for Fixed {
    fn fill(&self, buf: &mut [u8]) {
        buf.fill(self.0);
    }
}

/// A supervisor on virtual time: both clocks advance together unless the test suspends the host.
struct Run {
    sup: Supervisor<Fixed>,
    now: u64,
    awake: u64,
    heartbeat: u64,
    steps: u64,
    /// The header's live-sequence count, which the runner writes and the shell samples.
    live: u32,
}

impl Run {
    fn new(jitter: u8) -> Run {
        Run::with(policy(), jitter)
    }

    fn with(policy: Policy, jitter: u8) -> Run {
        let mut run = Run {
            sup: Supervisor::new(policy, Fixed(jitter)),
            now: 0,
            awake: 0,
            heartbeat: 0,
            steps: 0,
            live: 0,
        };
        assert!(run.input(Input::Spawned).is_empty());
        run
    }

    fn input(&mut self, input: Input) -> Vec<Output> {
        let mut out = Outbox::new();
        self.sup.handle(MonoTime(self.now), input, &mut out);
        out.drain().collect()
    }

    /// Advances both clocks by `by` and samples.
    fn sample_after(&mut self, by: u64) -> Vec<Output> {
        self.now += by;
        self.awake += by;
        self.sample()
    }

    fn sample(&mut self) -> Vec<Output> {
        self.input(Input::Sample {
            heartbeat: self.heartbeat,
            steps: self.steps,
            live: self.live,
            awake: AwakeTime(self.awake),
        })
    }

    fn state(&mut self, state: RunnerState) -> Vec<Output> {
        self.input(Input::State {
            state,
            epoch: PlanEpoch { term: 1, seq: 1 },
        })
    }

    /// Reports the forward path up to `Ready`, as a runner's machine does: `Ready` with no live
    /// sequence. The runner then counts `live` sequences in the header, where the shell samples
    /// them.
    fn reach_ready(&mut self, live: u32) {
        self.live = live;
        for state in [
            RunnerState::Fetching {
                have: Bytes(0),
                need: Bytes(1),
            },
            RunnerState::Packing {
                done: Bytes(0),
                total: Bytes(1),
            },
            RunnerState::Loading {
                done: Bytes(0),
                total: Bytes(1),
            },
            RunnerState::Warming {
                step: WarmStep::Prepare,
            },
            RunnerState::Ready {
                live_sequences: 0,
                degraded: None,
            },
        ] {
            let out = self.state(state);
            assert!(matches!(out[..], [Output::Forward { .. }]), "{out:?}");
        }
    }

    /// Advances to the deadline and ticks.
    fn tick_at_deadline(&mut self) -> Vec<Output> {
        let at = self.sup.next_deadline().expect("a deadline").0;
        assert!(at >= self.now);
        self.awake += at - self.now;
        self.now = at;
        self.input(Input::Tick)
    }
}

use syntropy_core::Bytes;

#[test]
fn the_policy_is_rfc_0033s_defaults() {
    let p = policy();
    assert_eq!(p.liveness_timeout, Nanos(3 * S));
    assert_eq!(p.stall_floor, Nanos(10 * S));
    assert_eq!(p.stall_factor, 20.0);
    assert_eq!(
        (p.backoff_initial, p.backoff_max),
        (Nanos(S), Nanos(60 * S))
    );
    assert_eq!((p.budget_count, p.budget_window), (5, Nanos(600 * S)));
    assert!(p.reprobe_on_device);
    assert_eq!(p.sample_period, Nanos(500 * MS));
    // max(10 s, 20 × p99).
    assert_eq!(p.stall_after(Nanos(100 * MS)), Nanos(10 * S));
    assert_eq!(p.stall_after(Nanos(S)), Nanos(20 * S));
    // 1 s doubling to 60 s.
    let delays: Vec<_> = (1..=8).map(|n| p.backoff(n).0 / S).collect();
    assert_eq!(delays, [1, 2, 4, 8, 16, 32, 60, 60]);
}

/// A heartbeat that stops is a hang once it has not moved for more than 3 s: the sample at
/// 3.5 s kills, and the one at 3.0 s does not.
#[test]
fn a_hang_is_killed_after_the_liveness_timeout() {
    let mut r = Run::new(0);
    r.reach_ready(0);
    assert!(r.sample_after(0).is_empty());
    for _ in 0..5 {
        // 0.5 s … 2.5 s of silence.
        assert!(r.sample_after(500 * MS).is_empty(), "at {} ms", r.now / MS);
    }
    // 3.0 s: the timeout is reached.
    assert_eq!(r.sample_after(500 * MS), [Output::Kill(KillReason::Hung)]);
    // The kill's exit is a `Timeout` failure, reported for the runner, and a restart after 1 s.
    let out = r.input(Input::Exited);
    assert_eq!(
        out,
        [Output::Forward {
            state: RunnerState::Failed {
                class: ErrorClass::Timeout,
                during: RunnerStateKind::Ready,
                restarts: 0,
            },
            epoch: PlanEpoch { term: 1, seq: 1 },
        }]
    );
    assert_eq!(r.sup.next_deadline(), Some(MonoTime(r.now + S)));
    assert_eq!(r.tick_at_deadline(), [Output::Spawn]);
    // A heartbeat that keeps moving is never a hang.
    let mut r = Run::new(0);
    for _ in 0..100 {
        r.heartbeat += 1;
        assert!(r.sample_after(500 * MS).is_empty());
    }
}

/// `Ready` with live sequences and a step counter that stops is a stall after
/// max(10 s, 20 × p99): 10 s at a 100 ms p99, 20 s at 1 s. With no live sequences, or with no
/// prediction, it is never one.
#[test]
fn a_stall_is_killed_after_the_stall_bound() {
    for (p99, bound) in [(100 * MS, 10 * S), (S, 20 * S)] {
        let mut r = Run::new(0);
        r.reach_ready(2);
        assert!(r.input(Input::PredictedStep { p99: Nanos(p99) }).is_empty());
        r.sample_after(0);
        let mut quiet = 0;
        loop {
            r.heartbeat += 1;
            let out = r.sample_after(500 * MS);
            quiet += 500 * MS;
            if quiet < bound {
                assert!(out.is_empty(), "at {} ms", quiet / MS);
            } else {
                assert_eq!(out, [Output::Kill(KillReason::Stalled)], "p99 {p99}");
                assert_eq!(quiet, bound);
                break;
            }
        }
    }
    // The count is the header's: a `Ready` report's own count is ignored, and a count that
    // falls to zero stops the stall clock.
    let mut r = Run::new(0);
    r.reach_ready(0);
    r.input(Input::PredictedStep {
        p99: Nanos(100 * MS),
    });
    r.input(Input::State {
        state: RunnerState::Ready {
            live_sequences: 5,
            degraded: None,
        },
        epoch: PlanEpoch { term: 1, seq: 2 },
    });
    for _ in 0..40 {
        r.heartbeat += 1;
        assert!(r.sample_after(500 * MS).is_empty());
    }
    r.live = 1;
    for _ in 0..19 {
        r.heartbeat += 1;
        assert!(r.sample_after(500 * MS).is_empty());
    }
    r.live = 0;
    r.heartbeat += 1;
    assert!(
        r.sample_after(500 * MS).is_empty(),
        "no live sequence, no stall"
    );
    for (live, predicted) in [(0, true), (3, false)] {
        let mut r = Run::new(0);
        r.reach_ready(live);
        if predicted {
            r.input(Input::PredictedStep {
                p99: Nanos(100 * MS),
            });
        }
        for _ in 0..200 {
            r.heartbeat += 1;
            assert!(r.sample_after(500 * MS).is_empty());
        }
    }
}

/// A crash after a sticky device report is `Device`: re-probe, then restart if the device is
/// there, or stop with `Fatal(Device)` if it is gone. A crash without one is `Internal`.
#[test]
fn a_crash_is_classified_and_a_device_failure_re_probes() {
    for present in [true, false] {
        let mut r = Run::new(0);
        r.reach_ready(1);
        let out = r.state(RunnerState::Failed {
            class: ErrorClass::Device,
            during: RunnerStateKind::Ready,
            restarts: 0,
        });
        assert!(matches!(out[..], [Output::Forward { .. }]));
        // The runner reported its own `Failed`, so the exit adds none.
        assert_eq!(r.input(Input::Exited), [Output::Reprobe]);
        let out = r.input(Input::Reprobed { present });
        if present {
            assert!(out.is_empty());
            assert_eq!(r.tick_at_deadline(), [Output::Spawn]);
        } else {
            assert_eq!(
                out,
                [Output::Forward {
                    state: RunnerState::Stopped {
                        reason: StopReason::Fatal(ErrorClass::Device)
                    },
                    epoch: PlanEpoch { term: 1, seq: 1 },
                }]
            );
            assert!(r.sup.stopped());
            assert_eq!(r.sup.next_deadline(), None);
        }
    }
    // A runner that reports `Device` and then hangs on its way out is killed, and its failure
    // stays `Device`, so the device is still re-probed.
    let mut r = Run::new(0);
    r.reach_ready(0);
    r.state(RunnerState::Failed {
        class: ErrorClass::Device,
        during: RunnerStateKind::Ready,
        restarts: 0,
    });
    r.sample_after(0);
    let mut out = Vec::new();
    for _ in 0..6 {
        out.extend(r.sample_after(500 * MS));
    }
    assert_eq!(out, [Output::Kill(KillReason::Hung)]);
    assert_eq!(r.input(Input::Exited), [Output::Reprobe]);
    let mut r = Run::new(0);
    r.reach_ready(1);
    let out = r.input(Input::Exited);
    assert!(matches!(
        out[..],
        [Output::Forward {
            state: RunnerState::Failed {
                class: ErrorClass::Internal,
                ..
            },
            ..
        }]
    ));
    assert_eq!(r.tick_at_deadline(), [Output::Spawn]);
}

/// Each failure in the window doubles the delay, with jitter in [¾ · base, base]. The fifth
/// failure in 10 min stops the runner with `Drained(RestartBudgetSpent)` and asks for
/// `ReportRunnerFailed`. Failures spread over more than the window never spend it.
#[test]
fn the_back_off_doubles_and_the_fifth_failure_spends_the_budget() {
    for (jitter, scale) in [(0u8, 4u64), (0xFF, 0)] {
        let mut r = Run::new(jitter);
        let mut delays = Vec::new();
        for n in 1..=5 {
            r.enter_fetching(n);
            let out = r.input(Input::Exited);
            if n < 5 {
                assert_eq!(out.len(), 1, "{out:?}");
                let at = r.sup.next_deadline().unwrap().0;
                delays.push(at - r.now);
                assert_eq!(r.tick_at_deadline(), [Output::Spawn]);
                assert!(r.input(Input::Spawned).is_empty());
            } else {
                assert_eq!(
                    out,
                    [
                        Output::Forward {
                            state: RunnerState::Failed {
                                class: ErrorClass::Internal,
                                during: RunnerStateKind::Fetching,
                                restarts: 4,
                            },
                            epoch: PlanEpoch { term: 1, seq: 1 },
                        },
                        Output::Forward {
                            state: RunnerState::Stopped {
                                reason: StopReason::Drained(DrainReason::RestartBudgetSpent)
                            },
                            epoch: PlanEpoch { term: 1, seq: 1 },
                        },
                        Output::ReportFailed {
                            class: ErrorClass::Internal,
                            failures: 5,
                        },
                    ]
                );
            }
        }
        // No jitter: exactly 1, 2, 4, 8 s. Most jitter: the base less a quarter of it, and
        // never below ¾ of it.
        let bases = [S, 2 * S, 4 * S, 8 * S];
        for (d, base) in delays.iter().zip(bases) {
            assert!(*d <= base && *d >= base - base / 4, "{d} against {base}");
            if scale == 4 {
                assert_eq!(*d, base);
            }
        }
    }
    // Failures more than 10 min apart: the window holds one at a time.
    let mut r = Run::new(0);
    for n in 1..=8 {
        r.enter_fetching(n);
        let out = r.input(Input::Exited);
        assert_eq!(out.len(), 1, "failure {n}: {out:?}");
        r.now += 601 * S;
        r.awake += 601 * S;
        assert_eq!(r.input(Input::Tick), [Output::Spawn]);
        assert!(r.input(Input::Spawned).is_empty());
    }
}

impl Run {
    /// The first time, the forward path to `Ready`; after a restart, the runner's re-entry at
    /// `Fetching`.
    fn enter_fetching(&mut self, n: u32) {
        let out = self.state(RunnerState::Fetching {
            have: Bytes(0),
            need: Bytes(1),
        });
        assert!(matches!(out[..], [Output::Forward { .. }]), "{n}: {out:?}");
    }
}

/// A 60 s suspend, during which the runner's heartbeat cannot move either, kills nothing: the
/// two clocks diverge by the sleep, which is excused. A tick 10 s late on both clocks, from a
/// paused VM or a starved daemon, is excused the same way.
#[test]
fn a_suspend_or_a_late_tick_kills_nothing() {
    let mut r = Run::new(0);
    r.reach_ready(2);
    r.input(Input::PredictedStep {
        p99: Nanos(100 * MS),
    });
    r.sample_after(0);
    r.heartbeat += 1;
    assert!(r.sample_after(500 * MS).is_empty());
    // The host sleeps 60 s: running time advances, awake time barely does.
    r.now += 60 * S;
    r.awake += 10 * MS;
    assert!(r.sample().is_empty());
    // A starved daemon samples 10 s late; the runner did not move either.
    assert!(r.sample_after(10 * S).is_empty());
    // Then the runner resumes.
    for _ in 0..10 {
        r.heartbeat += 1;
        r.steps += 1;
        assert!(r.sample_after(500 * MS).is_empty());
    }
    // Without the excuse, the same silence is a hang: 4 s of awake time, sampled on time.
    let mut r = Run::new(0);
    r.sample_after(0);
    let mut out = Vec::new();
    for _ in 0..8 {
        out.extend(r.sample_after(500 * MS));
    }
    assert_eq!(out, [Output::Kill(KillReason::Hung)]);
}

/// Reports outside RFC-0001 §3.3 are refused and not forwarded: a first report other than
/// `Fetching`, a skipped state, `Ready → Ready` at the same or an older epoch (which
/// `can_become` alone allows), and anything after `Stopped`.
#[test]
fn reports_outside_section_3_3_are_refused() {
    let mut r = Run::new(0);
    let ready = RunnerState::Ready {
        live_sequences: 0,
        degraded: None,
    };
    assert_eq!(
        r.state(ready.clone()),
        [Output::Rejected {
            from: None,
            to: RunnerStateKind::Ready,
            epoch: PlanEpoch { term: 1, seq: 1 },
        }]
    );
    r.state(RunnerState::Fetching {
        have: Bytes(0),
        need: Bytes(1),
    });
    assert!(matches!(
        r.state(ready.clone())[..],
        [Output::Rejected {
            from: Some(RunnerStateKind::Fetching),
            ..
        }]
    ));
    let mut r = Run::new(0);
    r.reach_ready(0);
    for (seq, accepted) in [(1, false), (0, false), (2, true), (2, false), (3, true)] {
        let out = r.input(Input::State {
            state: ready.clone(),
            epoch: PlanEpoch { term: 1, seq },
        });
        assert_eq!(
            matches!(out[..], [Output::Forward { .. }]),
            accepted,
            "seq {seq}"
        );
        assert_eq!(
            matches!(out[..], [Output::Rejected { .. }]),
            !accepted,
            "seq {seq}"
        );
    }
    // A runner that stops by itself has ended, not failed.
    let stopped = RunnerState::Stopped {
        reason: StopReason::Drained(DrainReason::Unload),
    };
    let at = PlanEpoch { term: 1, seq: 3 };
    let draining = r.input(Input::State {
        state: RunnerState::Draining {
            reason: DrainReason::Unload,
            deadline: syntropy_core::Deadline::within(Nanos(S)),
            live_sequences: 0,
        },
        epoch: at,
    });
    assert!(matches!(draining[..], [Output::Forward { .. }]));
    let out = r.input(Input::State {
        state: stopped.clone(),
        epoch: at,
    });
    assert!(matches!(out[..], [Output::Forward { .. }]));
    assert!(r.input(Input::Exited).is_empty());
    assert!(r.sup.stopped());
    assert!(
        r.state(stopped).is_empty(),
        "a stopped supervisor takes no report"
    );
}
