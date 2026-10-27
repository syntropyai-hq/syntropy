// SPDX-License-Identifier: MIT OR Apache-2.0

//! The supervisor's shell for one runner (RFC-0001 §3.4; DIST-E1-T06): what turns the runner's
//! command channel and the shared-memory counters into [`Input`]s, and the core's
//! outputs into reports and process actions.
//!
//! - [`RunnerLink`] owns a [`Supervisor`] and the [`ReportSink`]. Every `StateChanged` the core
//!   accepts, from the runner or on its behalf, goes to [`report::forward_state_changed`], so
//!   each transition reaches the sink once. A spent restart budget goes to
//!   [`ReportSink::report_runner_failed`]. Kills, spawns, re-probes and refused reports go to the
//!   caller's [`Effects`].
//! - [`serve_channel`] is the runner-channel loop: it reads the runner's messages until the
//!   runner closes the channel, and hands each to the link.
//!
//! [`RunnerLink::sample_region`] reads the counters from the daemon's side of the runner's
//! shared-memory region.
//!
//! The link reads no clock of its own: the caller passes `now`, and [`RunnerLink::sample`] takes
//! an [`AwakeClock`], which the daemon's wiring builds over `io`'s `SystemClock` (ADR-0022
//! decision 2). So the daemon never names `SystemClock`, and tests run on virtual time.

use syntropy_core::{
    AwakeClock, Clock, Entropy, InstanceId, MonoTime, NodeId, Outbox, PlanEpoch, ProtocolCore,
    RunnerStateKind,
};
use syntropy_ipc::cmd::{DaemonChannel, RecvError, RunnerMsg};
use syntropy_ipc::shm::Region;

use crate::report::{self, ReportSink, RunnerFailedReport};
use crate::supervisor::{Input, KillReason, Output, Supervisor};

/// The process actions the core asks for, which the daemon's wiring carries out.
pub trait Effects {
    /// Kill the runner process.
    fn kill(&mut self, reason: KillReason);
    /// Start a runner process.
    fn spawn(&mut self);
    /// Re-probe the runner's devices, then pass the result as `Input::Reprobed`.
    fn reprobe(&mut self);
    /// A report that is no RFC-0001 §3.3 transition; logged, not forwarded.
    fn rejected(&mut self, from: Option<RunnerStateKind>, to: RunnerStateKind, epoch: PlanEpoch);
}

/// One runner's supervisor, its identity and the sink its reports go to.
pub struct RunnerLink<E: Entropy, S: ReportSink> {
    instance: InstanceId,
    node: NodeId,
    supervisor: Supervisor<E>,
    sink: S,
}

impl<E: Entropy, S: ReportSink> RunnerLink<E, S> {
    /// A link for the runner of `instance` on `node`.
    pub fn new(instance: InstanceId, node: NodeId, supervisor: Supervisor<E>, sink: S) -> Self {
        Self {
            instance,
            node,
            supervisor,
            sink,
        }
    }

    /// The supervisor.
    pub fn supervisor(&self) -> &Supervisor<E> {
        &self.supervisor
    }

    /// The sink.
    pub fn sink(&self) -> &S {
        &self.sink
    }

    /// Hands `input` to the supervisor at `now`, then carries out its outputs in order.
    pub fn handle(&mut self, now: MonoTime, input: Input, effects: &mut impl Effects) {
        let mut out = Outbox::new();
        self.supervisor.handle(now, input, &mut out);
        for output in out.drain() {
            match output {
                Output::Forward { state, epoch } => {
                    let msg = RunnerMsg::StateChanged { state, epoch };
                    report::forward_state_changed(self.instance, self.node, &msg, &mut self.sink);
                }
                Output::ReportFailed { class, failures } => {
                    self.sink.report_runner_failed(RunnerFailedReport {
                        instance: self.instance,
                        node: self.node,
                        class,
                        failures,
                    });
                }
                Output::Kill(reason) => effects.kill(reason),
                Output::Spawn => effects.spawn(),
                Output::Reprobe => effects.reprobe(),
                Output::Rejected { from, to, epoch } => effects.rejected(from, to, epoch),
            }
        }
    }

    /// Hands a message from the runner to the supervisor. Only `StateChanged` concerns it; the
    /// other messages have their own handlers.
    pub fn on_runner_msg(&mut self, now: MonoTime, msg: &RunnerMsg, effects: &mut impl Effects) {
        if let RunnerMsg::StateChanged { state, epoch } = msg {
            self.handle(
                now,
                Input::State {
                    state: state.clone(),
                    epoch: *epoch,
                },
                effects,
            );
        }
    }

    /// Samples the runner's counters, read from its shared-memory header by the caller, with
    /// both readings of `clock`.
    pub fn sample(
        &mut self,
        clock: &impl AwakeClock,
        heartbeat: u64,
        steps: u64,
        live: u32,
        effects: &mut impl Effects,
    ) {
        let now = clock.now();
        let awake = clock.awake();
        self.handle(
            now,
            Input::Sample {
                heartbeat,
                steps,
                live,
                awake,
            },
            effects,
        );
    }

    /// Samples the counters of the runner's shared-memory header, through the daemon's side of
    /// its region: the runner's heartbeat, the step counter and the live-sequence count.
    pub fn sample_region(
        &mut self,
        clock: &impl AwakeClock,
        region: &Region,
        effects: &mut impl Effects,
    ) {
        let (heartbeat, steps, live) = (
            region.peer_heartbeat(),
            region.steps(),
            region.live_sequences(),
        );
        self.sample(clock, heartbeat, steps, live, effects);
    }

    /// When the supervisor next wants an input: a sample or a timer.
    pub fn next_deadline(&self) -> Option<MonoTime> {
        self.supervisor.next_deadline()
    }
}

/// The runner-channel loop: hands every message the runner sends to `link` until the runner
/// closes the channel, which returns `Ok`. Any other failure of the channel is returned.
///
/// # Errors
///
/// The [`RecvError`] that ended the loop, other than [`RecvError::Closed`].
pub fn serve_channel<E: Entropy, S: ReportSink>(
    channel: &mut DaemonChannel,
    clock: &impl Clock,
    link: &mut RunnerLink<E, S>,
    effects: &mut impl Effects,
) -> Result<(), RecvError> {
    loop {
        match channel.recv() {
            Ok(envelope) => link.on_runner_msg(clock.now(), &envelope.msg, effects),
            Err(RecvError::Closed) => return Ok(()),
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests;
