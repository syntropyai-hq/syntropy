// SPDX-License-Identifier: MIT OR Apache-2.0

//! Forwarding a runner's `StateChanged` as `ReportRunner` (RFC-0001 §3.5, §5.10; DIST-E1-T05).
//!
//! The daemon forwards every `StateChanged` its runner sends as RFC-0001 §5.10's
//! `ClusterCommand::ReportRunner { instance, node, epoch, state: RunnerStateKind }`. The cluster
//! log records the kind alone (`RunnerStateChanged`). Progress and degradation stay on the node.
//!
//! `ClusterCommand` and the client that submits it are DIST-E9's and not built yet, so the
//! forwarding target here is a seam: [`ReportSink`], which DIST-E9's command client implements.
//! [`RunnerReport`] has `ReportRunner`'s fields.

use syntropy_core::{InstanceId, NodeId, PlanEpoch, RunnerStateKind};
use syntropy_ipc::cmd::RunnerMsg;

/// `ClusterCommand::ReportRunner`'s fields (RFC-0001 §5.10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunnerReport {
    /// The runner's instance.
    pub instance: InstanceId,
    /// The node it runs on: this daemon's.
    pub node: NodeId,
    /// The plan epoch it serves.
    pub epoch: PlanEpoch,
    /// Its state, without the fields.
    pub state: RunnerStateKind,
}

/// Where the daemon submits runner reports: DIST-E9's `ClusterCommand` client.
pub trait ReportSink {
    /// Submits one `ReportRunner`.
    fn report_runner(&mut self, report: RunnerReport);
}

/// Forwards `msg` from the runner of `instance` on `node` to `sink` when it is a
/// `StateChanged`, and returns whether it was one. Every other message is left to its own
/// handler.
pub fn forward_state_changed(
    instance: InstanceId,
    node: NodeId,
    msg: &RunnerMsg,
    sink: &mut impl ReportSink,
) -> bool {
    let RunnerMsg::StateChanged { state, epoch } = msg else {
        return false;
    };
    sink.report_runner(RunnerReport {
        instance,
        node,
        epoch: *epoch,
        state: state.kind(),
    });
    true
}

#[cfg(test)]
mod tests {
    use syntropy_core::{Bytes, DrainReason, RunnerState, StopReason, WarmStep};
    use syntropy_ipc::cmd::{Envelope, RequestId, decode_frame, encode_frame};

    use super::*;

    #[derive(Default)]
    struct Recorded(Vec<RunnerReport>);

    impl ReportSink for Recorded {
        fn report_runner(&mut self, report: RunnerReport) {
            self.0.push(report);
        }
    }

    /// Each `StateChanged`, framed as the runner sends it and decoded as the daemon reads it,
    /// is forwarded once as a report of its kind and epoch; other messages are not.
    #[test]
    fn every_state_changed_is_forwarded_as_one_report() {
        let instance = InstanceId(9);
        let node = NodeId([0x42; 32]);
        let epoch = PlanEpoch { term: 2, seq: 5 };
        let states = [
            RunnerState::Fetching {
                have: Bytes(0),
                need: Bytes(10),
            },
            RunnerState::Warming {
                step: WarmStep::Prepare,
            },
            RunnerState::Ready {
                live_sequences: 0,
                degraded: None,
            },
            RunnerState::Stopped {
                reason: StopReason::Drained(DrainReason::Unload),
            },
        ];
        let mut sent: Vec<RunnerMsg> = states
            .iter()
            .map(|state| RunnerMsg::StateChanged {
                state: state.clone(),
                epoch,
            })
            .collect();
        sent.insert(1, RunnerMsg::Pong);
        let mut sink = Recorded::default();
        let mut forwarded = 0;
        for (n, msg) in sent.into_iter().enumerate() {
            let frame = encode_frame(&Envelope {
                id: RequestId(n as u64),
                reply_to: None,
                msg,
            })
            .unwrap();
            let (read, _) = decode_frame::<RunnerMsg>(&frame).unwrap();
            forwarded += usize::from(forward_state_changed(instance, node, &read.msg, &mut sink));
        }
        assert_eq!(forwarded, states.len());
        let want: Vec<RunnerReport> = states
            .iter()
            .map(|state| RunnerReport {
                instance,
                node,
                epoch,
                state: state.kind(),
            })
            .collect();
        assert_eq!(sink.0, want);
    }
}
