// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runner supervisor (RFC-0001 §3.4; RFC-0033 §8.2; ADR-0015 item 5; DIST-E1-T06).
//!
//! [`Supervisor`] is the restart policy as a sans-I/O protocol core (RFC-0001 invariant 12): a
//! [`ProtocolCore`] that reads no clock and spawns, kills and probes nothing. A shell feeds it
//! what it sees, and carries out what it asks:
//!
//! - **Liveness.** At each of its deadlines the shell reads the runner's heartbeat, step counter
//!   and live-sequence count from the shared-memory header, and both clocks, and passes them as
//!   [`Input::Sample`]. The core measures how long each counter has not moved in *excused*
//!   time: running time less any interval the host spent suspended or paused, which a
//!   [`SuspendDetector`] finds from the two clocks (RFC-0030 §5.8, step 5). No heartbeat for
//!   `runner.liveness_timeout` is a hang. A `Ready` runner whose header counts live sequences and
//!   whose step counter
//!   has not moved for max(`runner.stall_floor`, `runner.stall_factor` × the predicted p99 step)
//!   is a stall. Either is [`Output::Kill`].
//! - **Reports.** Each `StateChanged` the runner sends is checked against RFC-0001 §3.3's edges
//!   ([`RunnerStateKind::can_become`]) and, for `Ready → Ready`, against the later-epoch rule,
//!   which the edge table does not hold. An accepted report is [`Output::Forward`], which the
//!   runner-channel loop hands to `report::forward_state_changed`. A report that is no §3.3
//!   transition is [`Output::Rejected`] and is not forwarded.
//! - **Crashes.** When the process exits, the core classifies the failure (§3.4): `Device` if the
//!   runner reported a `Failed { class: Device }` before it died, even if the core then killed
//!   it; `Timeout` \[chosen\] if the core killed it for a hang or a stall; and `Internal`
//!   otherwise. It reports `Failed` on the runner's behalf.
//! - **Restarts.** A failure inside the budget (`runner.restart_budget`, 5 in 10 min) restarts
//!   the runner after a back-off that doubles from `runner.restart_backoff`'s 1 s to 60 s, with
//!   jitter drawn from the [`Entropy`] it is given (rule T1). A `Device` failure is re-probed
//!   first ([`Output::Reprobe`]); a device that is gone stops the runner with
//!   `Fatal(Device)`. The failure that spends the budget stops it with
//!   `Drained(RestartBudgetSpent)` and asks the shell to submit `ReportRunnerFailed`.
//!
//! Not here, by the plan: `runner.load_stall` (the bound during `Fetching` to `Warming`, which
//! needs fetch progress no message carries yet) and a clean stop of a runner that is not yet
//! `Ready`, which RFC-0001 §3.3 has no edge for (the review of #1932, N3). A `Drain` or
//! `Shutdown` before `Ready` is refused by the runner's machine, so such a runner can end only
//! through `Failed`, which this core counts as a failure. That choice is the maintainers'.

use std::collections::VecDeque;

use syntropy_cluster::config::recovery::{DeviceReprobe, RunnerSection};
use syntropy_core::{
    AwakeTime, DrainReason, Entropy, ErrorClass, MonoTime, Nanos, Outbox, PlanEpoch, ProtocolCore,
    RunnerState, RunnerStateKind, StopReason, SuspendDetector,
};

/// The supervision settings, from RFC-0033 §8.2's `runner.*` keys.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    /// `runner.liveness_timeout`: no heartbeat for this long, in excused time, is a hang.
    pub liveness_timeout: Nanos,
    /// `runner.stall_factor`: the multiple of the predicted p99 step.
    pub stall_factor: f64,
    /// `runner.stall_floor`: the shortest stall bound.
    pub stall_floor: Nanos,
    /// `runner.restart_backoff`: the first delay.
    pub backoff_initial: Nanos,
    /// `runner.restart_backoff`: the longest delay.
    pub backoff_max: Nanos,
    /// `runner.restart_budget`: failures allowed in the window, the last of which spends it.
    pub budget_count: u32,
    /// `runner.restart_budget`: the window.
    pub budget_window: Nanos,
    /// `runner.device_reprobe`: re-probe the device before restarting after a `Device` failure.
    pub reprobe_on_device: bool,
    /// How often the shell samples the counters: a sixth of the liveness timeout \[chosen\]. A
    /// counter's silence is counted from the sample that saw it last move, up to one period
    /// after the move. A hang is killed at the first sample whose counted silence reaches the
    /// timeout, so between the timeout and the timeout plus one period after the last beat, and
    /// never before it.
    pub sample_period: Nanos,
}

impl Policy {
    /// The policy a node's `runner.*` section gives.
    #[must_use]
    pub fn from_section(section: &RunnerSection) -> Policy {
        let liveness_timeout = section.liveness_timeout.nanos();
        Policy {
            liveness_timeout,
            stall_factor: section.stall_factor,
            stall_floor: section.stall_floor.nanos(),
            backoff_initial: section.restart_backoff.initial.nanos(),
            backoff_max: section.restart_backoff.max.nanos(),
            budget_count: section.restart_budget.count,
            budget_window: section.restart_budget.window.nanos(),
            reprobe_on_device: match section.device_reprobe {
                DeviceReprobe::OnDeviceFailure => true,
            },
            sample_period: Nanos((liveness_timeout.0 / 6).max(1)),
        }
    }

    /// The stall bound for a predicted p99 step of `p99`: max(floor, factor × p99).
    #[must_use]
    pub fn stall_after(&self, p99: Nanos) -> Nanos {
        let scaled = (p99.0 as f64 * self.stall_factor).min(u64::MAX as f64) as u64;
        Nanos(scaled.max(self.stall_floor.0))
    }

    /// The delay before restart `n` in the window (1 for the first failure), before jitter:
    /// the initial delay doubled `n − 1` times, at most the longest delay.
    #[must_use]
    pub fn backoff(&self, n: u32) -> Nanos {
        let doublings = n.saturating_sub(1).min(63);
        let delay = self.backoff_initial.0.saturating_mul(1u64 << doublings);
        Nanos(delay.min(self.backoff_max.0))
    }
}

/// Why the supervisor killed a runner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KillReason {
    /// No heartbeat for `liveness_timeout`.
    Hung,
    /// `Ready` with live sequences and no step progress for the stall bound.
    Stalled,
}

/// What the shell tells the core.
#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    /// A runner process was started for the instance: the first, or the one an
    /// [`Output::Spawn`] asked for.
    Spawned,
    /// The shell's sample at a deadline: the shared-memory header's counters and the clock that
    /// stops while the host is suspended, read with `now`.
    Sample {
        /// The runner's heartbeat counter.
        heartbeat: u64,
        /// The runner's step counter.
        steps: u64,
        /// The runner's live-sequence count. The runner changes it in place, with no
        /// `StateChanged`, so it travels with the counters (RFC-0001 §3.5, note of 2026-10-28).
        live: u32,
        /// The clock that stops while the host is suspended.
        awake: AwakeTime,
    },
    /// A `StateChanged` the runner sent.
    State {
        /// The new state.
        state: RunnerState,
        /// The epoch it serves.
        epoch: PlanEpoch,
    },
    /// The plan's predicted p99 step time for the runner's stage.
    PredictedStep {
        /// The prediction.
        p99: Nanos,
    },
    /// The runner process exited.
    Exited,
    /// A deadline passed with nothing else to report, such as the end of a back-off.
    Tick,
    /// The result of an [`Output::Reprobe`].
    Reprobed {
        /// Whether the device is still there.
        present: bool,
    },
}

/// What the core asks the shell to do, in order.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    /// Kill the runner process. Its exit arrives as [`Input::Exited`].
    Kill(KillReason),
    /// Start a runner process for the instance.
    Spawn,
    /// Re-probe the runner's devices, and answer with [`Input::Reprobed`].
    Reprobe,
    /// Forward this state as `ReportRunner`, through `report::forward_state_changed`.
    Forward {
        /// The state.
        state: RunnerState,
        /// Its epoch.
        epoch: PlanEpoch,
    },
    /// A report that is no RFC-0001 §3.3 transition, or a `Ready → Ready` without a later epoch.
    /// It is not forwarded; the shell logs it.
    Rejected {
        /// The state the core held.
        from: Option<RunnerStateKind>,
        /// The state reported.
        to: RunnerStateKind,
        /// The epoch reported.
        epoch: PlanEpoch,
    },
    /// The restart budget is spent: submit `ClusterCommand::ReportRunnerFailed`.
    ReportFailed {
        /// The last failure's class.
        class: ErrorClass,
        /// The failures in the window, the last included.
        failures: u32,
    },
}

/// Where the runner is, as the supervisor sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// No process; waiting for the shell's `Spawned`.
    Starting,
    /// A process is running.
    Running,
    /// The core asked for a kill and waits for the exit.
    Killing(KillReason),
    /// Waiting for the re-probe's result.
    Reprobing,
    /// Waiting to restart at this time.
    Backoff(MonoTime),
    /// Terminal.
    Stopped,
}

/// The restart policy for one runner of one instance on this node (RFC-0001 §3.4).
pub struct Supervisor<E: Entropy> {
    policy: Policy,
    entropy: E,
    phase: Phase,
    /// The last accepted state and its epoch.
    state: Option<(RunnerState, PlanEpoch)>,
    /// The last failure class the runner reported in a `Failed` of its own.
    reported_failure: Option<ErrorClass>,
    detector: SuspendDetector,
    last_sample: Option<MonoTime>,
    heartbeat: Option<u64>,
    steps: Option<u64>,
    /// The live-sequence count from the last sample.
    live: u32,
    /// Excused time since the heartbeat last moved.
    heartbeat_quiet: Nanos,
    /// Excused time since the step counter last moved, while `Ready` with live sequences.
    step_quiet: Nanos,
    p99: Option<Nanos>,
    /// Failure times inside the budget window.
    failures: VecDeque<MonoTime>,
    restarts: u32,
}

impl<E: Entropy> Supervisor<E> {
    /// A supervisor that waits for the first `Spawned`.
    pub fn new(policy: Policy, entropy: E) -> Self {
        Self {
            policy,
            entropy,
            phase: Phase::Starting,
            state: None,
            reported_failure: None,
            detector: SuspendDetector::new(),
            last_sample: None,
            heartbeat: None,
            steps: None,
            live: 0,
            heartbeat_quiet: Nanos(0),
            step_quiet: Nanos(0),
            p99: None,
            failures: VecDeque::new(),
            restarts: 0,
        }
    }

    /// The last state accepted or reported, if any.
    pub fn state(&self) -> Option<&RunnerState> {
        self.state.as_ref().map(|(s, _)| s)
    }

    /// Whether the runner is stopped for good.
    pub fn stopped(&self) -> bool {
        self.phase == Phase::Stopped
    }

    fn reset_liveness(&mut self) {
        self.detector = SuspendDetector::new();
        self.last_sample = None;
        self.heartbeat = None;
        self.steps = None;
        self.live = 0;
        self.heartbeat_quiet = Nanos(0);
        self.step_quiet = Nanos(0);
    }

    /// Whether the runner is `Ready` with live sequences, so its step counter must move. The
    /// count is the header's, from the last sample: a `Ready` report carries the count only at
    /// the moment it was sent, and the runner changes it without one.
    fn must_step(&self) -> bool {
        self.live > 0 && matches!(self.state, Some((RunnerState::Ready { .. }, _)))
    }

    fn sample(
        &mut self,
        now: MonoTime,
        heartbeat: u64,
        steps: u64,
        live: u32,
        awake: AwakeTime,
        out: &mut Outbox<Output>,
    ) {
        if self.phase != Phase::Running {
            return;
        }
        let stepping_before = self.must_step();
        self.live = live;
        let interval = self.detector.observe(now, awake);
        let first = self.last_sample.is_none();
        self.last_sample = Some(now);
        // Excused time: the running interval less a suspend (any gap between the clocks) and
        // less lateness on both clocks past one sample period (a paused VM or a
        // starved daemon), so a host's sleep never counts against the runner.
        let suspended = interval.suspended(Nanos(0));
        let late = interval.late(self.policy.sample_period, self.policy.sample_period);
        let excused = Nanos(
            interval
                .running
                .0
                .saturating_sub(suspended.0)
                .saturating_sub(late.0),
        );
        if first || self.heartbeat != Some(heartbeat) {
            self.heartbeat_quiet = Nanos(0);
        } else {
            self.heartbeat_quiet = Nanos(self.heartbeat_quiet.0.saturating_add(excused.0));
        }
        if !stepping_before || !self.must_step() || first || self.steps != Some(steps) {
            self.step_quiet = Nanos(0);
        } else {
            self.step_quiet = Nanos(self.step_quiet.0.saturating_add(excused.0));
        }
        self.heartbeat = Some(heartbeat);
        self.steps = Some(steps);
        if self.heartbeat_quiet >= self.policy.liveness_timeout {
            self.kill(KillReason::Hung, out);
        } else if let Some(p99) = self.p99
            && self.must_step()
            && self.step_quiet >= self.policy.stall_after(p99)
        {
            self.kill(KillReason::Stalled, out);
        }
    }

    fn kill(&mut self, reason: KillReason, out: &mut Outbox<Output>) {
        self.phase = Phase::Killing(reason);
        out.push(Output::Kill(reason));
    }

    fn report(&mut self, state: RunnerState, epoch: PlanEpoch, out: &mut Outbox<Output>) {
        let from = self.state.as_ref().map(|(s, e)| (s.kind(), *e));
        let to = state.kind();
        let legal = match from {
            // A runner's first report is its entry into `Fetching` (`[*] → Fetching`).
            None => to == RunnerStateKind::Fetching,
            Some((RunnerStateKind::Ready, then)) if to == RunnerStateKind::Ready => epoch > then,
            Some((kind, then)) => kind.can_become(to) && epoch >= then,
        };
        if !legal {
            out.push(Output::Rejected {
                from: from.map(|(k, _)| k),
                to,
                epoch,
            });
            return;
        }
        if let RunnerState::Failed { class, .. } = state {
            self.reported_failure = Some(class);
        }
        if to == RunnerStateKind::Fetching {
            self.reported_failure = None;
        }
        self.state = Some((state.clone(), epoch));
        out.push(Output::Forward { state, epoch });
    }

    fn exited(&mut self, now: MonoTime, out: &mut Outbox<Output>) {
        let killed = match self.phase {
            Phase::Running => None,
            Phase::Killing(reason) => Some(reason),
            _ => return,
        };
        // A runner that stopped by itself has ended, not failed.
        if killed.is_none()
            && let Some((RunnerState::Stopped { .. }, _)) = self.state
        {
            self.phase = Phase::Stopped;
            return;
        }
        // A reported sticky device error wins over a kill: a runner that reported `Device` and
        // then hung on its way out still has its device re-probed. A kill for a hang or a stall
        // is `Timeout`, §5.12's class for "progress stalled" \[chosen\]: §3.4 names only the
        // classes of a crash.
        let class = match (killed, self.reported_failure) {
            (_, Some(ErrorClass::Device)) => ErrorClass::Device,
            (Some(_), _) => ErrorClass::Timeout,
            (None, _) => ErrorClass::Internal,
        };
        let epoch = self
            .state
            .as_ref()
            .map_or(PlanEpoch::default(), |(_, e)| *e);
        // Report `Failed` for the runner unless it reported one itself.
        if !matches!(self.state, Some((RunnerState::Failed { .. }, _))) {
            let during = self
                .state
                .as_ref()
                .map_or(RunnerStateKind::Fetching, |(s, _)| s.kind());
            let failed = RunnerState::Failed {
                class,
                during,
                restarts: self.restarts,
            };
            self.state = Some((failed.clone(), epoch));
            out.push(Output::Forward {
                state: failed,
                epoch,
            });
        }
        self.reset_liveness();
        // The budget: failures inside the window, this one included.
        let window = self.policy.budget_window;
        while let Some(&t) = self.failures.front() {
            if now.saturating_duration_since(t) > window {
                self.failures.pop_front();
            } else {
                break;
            }
        }
        self.failures.push_back(now);
        let failures = u32::try_from(self.failures.len()).unwrap_or(u32::MAX);
        if failures >= self.policy.budget_count {
            self.stop(
                StopReason::Drained(DrainReason::RestartBudgetSpent),
                epoch,
                out,
            );
            out.push(Output::ReportFailed { class, failures });
            return;
        }
        if class == ErrorClass::Device && self.policy.reprobe_on_device {
            self.phase = Phase::Reprobing;
            out.push(Output::Reprobe);
            return;
        }
        self.schedule_restart(now, failures);
    }

    fn schedule_restart(&mut self, now: MonoTime, failures: u32) {
        let base = self.policy.backoff(failures);
        // Jitter: a delay drawn from [¾ · base, base], so it never exceeds the longest delay
        // and never shrinks below the previous step's base [chosen].
        let mut bytes = [0u8; 8];
        self.entropy.fill(&mut bytes);
        let quarter = base.0 / 4;
        let jitter = if quarter == 0 {
            0
        } else {
            u64::from_le_bytes(bytes) % (quarter + 1)
        };
        let delay = Nanos(base.0 - jitter);
        self.phase = Phase::Backoff(now.saturating_add(delay));
    }

    fn stop(&mut self, reason: StopReason, epoch: PlanEpoch, out: &mut Outbox<Output>) {
        let stopped = RunnerState::Stopped { reason };
        self.state = Some((stopped.clone(), epoch));
        self.phase = Phase::Stopped;
        out.push(Output::Forward {
            state: stopped,
            epoch,
        });
    }
}

impl<E: Entropy> ProtocolCore for Supervisor<E> {
    type Input = Input;
    type Output = Output;

    fn handle(&mut self, now: MonoTime, input: Input, out: &mut Outbox<Output>) {
        match input {
            Input::Spawned => {
                if self.phase == Phase::Starting {
                    self.phase = Phase::Running;
                    self.reset_liveness();
                }
            }
            Input::Tick => {}
            Input::Sample {
                heartbeat,
                steps,
                live,
                awake,
            } => self.sample(now, heartbeat, steps, live, awake, out),
            Input::State { state, epoch } => {
                if !matches!(self.phase, Phase::Stopped) {
                    self.report(state, epoch, out);
                }
            }
            Input::PredictedStep { p99 } => self.p99 = Some(p99),
            Input::Exited => self.exited(now, out),
            Input::Reprobed { present } => {
                if self.phase == Phase::Reprobing {
                    if present {
                        let failures = u32::try_from(self.failures.len()).unwrap_or(u32::MAX);
                        self.schedule_restart(now, failures);
                    } else {
                        let epoch = self
                            .state
                            .as_ref()
                            .map_or(PlanEpoch::default(), |(_, e)| *e);
                        self.stop(StopReason::Fatal(ErrorClass::Device), epoch, out);
                    }
                }
            }
        }
        // A restart that is due, at the first input on or after its time.
        if let Phase::Backoff(at) = self.phase
            && now >= at
        {
            self.phase = Phase::Starting;
            self.restarts = self.restarts.saturating_add(1);
            out.push(Output::Spawn);
        }
    }

    fn next_deadline(&self) -> Option<MonoTime> {
        match self.phase {
            Phase::Running => Some(
                self.last_sample
                    .map_or(MonoTime(0), |t| t.saturating_add(self.policy.sample_period)),
            ),
            Phase::Backoff(at) => Some(at),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
