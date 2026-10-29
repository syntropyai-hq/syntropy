// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fault plans over the command channel (DIST-E1-T13; RFC-0060 §9.2).
//!
//! With `fault-injection`, this test binary plays a test daemon and runs itself again as the
//! runner, which connects to the daemon's endpoint and serves `SetFaultPlan` with
//! `install_fault_plan` on its clock, as the runner (DIST-E1-T04, T05) will. Each `Ping` makes
//! the runner reach the fault point `ipc.test.point` and report, in a `StateChanged` reply's
//! `Ready { live_sequences }`, what the point did. Without the feature, the binary checks that the frame such a daemon sends is
//! refused as an unknown variant.
//!
//! `cargo test --workspace` builds `ipc` without the feature, and `cargo xtask test unit` runs it
//! again with it, as it does `obs`'s fault points.

#![cfg(any(target_os = "linux", target_vendor = "apple", windows))]

/// One encoded frame of the test variant (wire index 2²⁰) carrying `toml`, built by hand: the
/// bytes a fault build's `DaemonFrame::SetFaultPlan` encodes to, as message 1 with no reply.
fn fault_frame_bytes(toml: &str) -> Vec<u8> {
    let mut payload = vec![0x01, 0x00];
    // 2²⁰ as a postcard varint.
    payload.extend_from_slice(&[0x80, 0x80, 0x40]);
    assert!(toml.len() < 128, "a one-byte length");
    payload.push(toml.len() as u8);
    payload.extend_from_slice(toml.as_bytes());
    let mut frame = (payload.len() as u32).to_le_bytes().to_vec();
    frame.extend(payload);
    frame
}

const PLAN_TEXT: &str = "rules = []\n";

/// The criterion "a build without `fault-injection` has no such message, and a frame carrying one
/// is refused as unknown": the production decoder reads the fault build's frame, and its reply,
/// as an unknown variant. (The type itself is absent too: `cmd`'s `compile_fail` doctest.)
#[cfg(not(feature = "fault-injection"))]
#[test]
fn a_build_without_fault_injection_refuses_the_frame_as_unknown() {
    use syntropy_ipc::cmd::{DaemonMsg, Envelope, FrameError, RunnerMsg, decode_frame};

    let frame = fault_frame_bytes(PLAN_TEXT);
    let refused = decode_frame::<DaemonMsg>(&frame).unwrap_err();
    // postcard's error for the serde error a derived enum raises at an index it has no variant
    // for. That the index is the cause: the same frame with `Ping`'s index, 6, decodes `Ping`
    // and then stops at the plan's bytes, which no longer belong to the message.
    assert!(
        matches!(
            refused,
            FrameError::Malformed(postcard::Error::SerdeDeCustom)
        ),
        "{refused:?}"
    );
    let mut as_ping = frame.clone();
    as_ping.splice(6..9, [0x06]);
    let len = (as_ping.len() - 4) as u32;
    as_ping[..4].copy_from_slice(&len.to_le_bytes());
    assert!(
        matches!(
            decode_frame::<DaemonMsg>(&as_ping),
            Err(FrameError::Trailing { .. })
        ),
        "{:?}",
        decode_frame::<DaemonMsg>(&as_ping)
    );
    // The reply's frame, `FaultPlanSet(Ok(()))`, is refused by a production daemon too.
    let mut payload = vec![0x01, 0x01, 0x01, 0x80, 0x80, 0x40, 0x00];
    let mut reply = (payload.len() as u32).to_le_bytes().to_vec();
    reply.append(&mut payload);
    assert!(matches!(
        decode_frame::<RunnerMsg>(&reply),
        Err(FrameError::Malformed(_))
    ));
    // A production frame still decodes.
    let ping = syntropy_ipc::cmd::encode_frame(&Envelope {
        id: syntropy_ipc::cmd::RequestId(1),
        reply_to: None,
        msg: DaemonMsg::Ping,
    })
    .unwrap();
    assert_eq!(
        decode_frame::<DaemonMsg>(&ping).unwrap().0.msg,
        DaemonMsg::Ping
    );
}

#[cfg(feature = "fault-injection")]
mod with_fault_injection {
    use std::process::{Child, Command};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use syntropy_core::{Clock, ErrorClass, MonoTime, NodeId, PlanEpoch, RunnerState};
    use syntropy_ipc::cmd::{
        DaemonFrame, DaemonMsg, Endpoint, Envelope, FaultDaemonChannel, FaultRunnerChannel,
        Listener, RequestId, RunnerFrame, RunnerMsg, TEST_VARIANT, connect, decode_frame,
        encode_frame, install_fault_plan,
    };
    use syntropy_obs::faults::{FaultEffect, FaultPlan, FaultRule, Trigger};

    use super::{PLAN_TEXT, fault_frame_bytes};

    /// The endpoint the runner connects to, set only in the child.
    const ENDPOINT: &str = "SYNTROPY_IPC_TEST_FAULT_ENDPOINT";
    /// The fault point the runner reaches on each `Ping`.
    const POINT: &str = "ipc.test.point";
    /// How long a child runner lives if nobody ends it.
    const RUNNER_LIFETIME: Duration = Duration::from_secs(60);

    /// The runner's clock: a fixed instant, which the plans here never read.
    struct Fixed;

    impl Clock for Fixed {
        fn now(&self) -> MonoTime {
            MonoTime(0)
        }

        fn unix_ms(&self) -> u64 {
            0
        }
    }

    /// What the fault point did, as the runner reports it in `StateChanged`'s
    /// `Ready { live_sequences }`.
    fn code(effect: FaultEffect) -> u32 {
        match effect {
            FaultEffect::None => 0,
            FaultEffect::Error(ErrorClass::Device) => 1,
            _ => 2,
        }
    }

    /// The runner's side, in the child process: serves fault plans and pings until the daemon
    /// closes the channel.
    fn run_runner(endpoint: &str) {
        thread::spawn(|| {
            thread::sleep(RUNNER_LIFETIME);
            std::process::exit(3);
        });
        let endpoint: Endpoint = endpoint.parse().unwrap();
        let mut channel = FaultRunnerChannel::new(connect(&endpoint).unwrap());
        let clock: Arc<dyn Clock> = Arc::new(Fixed);
        while let Ok(envelope) = channel.recv() {
            let reply = match envelope.msg {
                DaemonFrame::SetFaultPlan(toml) => {
                    RunnerFrame::FaultPlanSet(install_fault_plan(&toml, Arc::clone(&clock)))
                }
                DaemonFrame::Msg(DaemonMsg::Ping) => {
                    let effect = syntropy_obs::fault_point!("ipc.test.point");
                    RunnerFrame::Msg(RunnerMsg::StateChanged {
                        state: RunnerState::Ready {
                            live_sequences: code(effect),
                            degraded: None,
                        },
                        epoch: PlanEpoch::default(),
                    })
                }
                other => panic!("the runner was sent {other:?}"),
            };
            channel.reply(envelope.id, &reply).unwrap();
        }
    }

    fn endpoint(test: &str) -> Endpoint {
        let name = format!("syntropy-ipc-{test}-{}", std::process::id());
        if cfg!(windows) {
            Endpoint::Pipe(name)
        } else {
            Endpoint::Socket(std::env::temp_dir().join(name).join("cmd.sock"))
        }
    }

    fn spawn_runner(test: &str, endpoint: &Endpoint) -> (Listener, FaultDaemonChannel, Child) {
        let mut listener = Listener::bind(endpoint).unwrap();
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--test-threads", "1"])
            .env(ENDPOINT, endpoint.to_string())
            .spawn()
            .unwrap();
        let stream = listener.accept().unwrap();
        (listener, FaultDaemonChannel::new(stream), child)
    }

    fn rule(when: Trigger) -> FaultPlan {
        FaultPlan {
            rules: vec![FaultRule {
                point: POINT.to_owned(),
                when,
                filter: None,
                effect: FaultEffect::Error(ErrorClass::Device),
            }],
        }
    }

    /// Sends `frame` and returns the runner's answer to it.
    fn ask(channel: &mut FaultDaemonChannel, frame: &DaemonFrame) -> RunnerFrame {
        let id = channel.send(frame).unwrap();
        let answer = channel.recv().unwrap();
        assert_eq!(answer.reply_to, Some(id));
        answer.msg
    }

    /// What the runner's fault point does on the next `Ping`.
    fn ping(channel: &mut FaultDaemonChannel) -> u32 {
        match ask(channel, &DaemonFrame::Msg(DaemonMsg::Ping)) {
            RunnerFrame::Msg(RunnerMsg::StateChanged {
                state: RunnerState::Ready { live_sequences, .. },
                ..
            }) => live_sequences,
            other => panic!("a ping was answered with {other:?}"),
        }
    }

    /// The criteria "a round-trip test pushes a plan to a runner, and a fault point in the runner
    /// fires as the plan says" and "an invalid plan is refused in the reply, and the runner keeps
    /// its plan", over a real channel to a runner process.
    #[test]
    fn a_pushed_plan_fires_in_the_runner_and_an_invalid_one_is_refused() {
        const TEST: &str =
            "with_fault_injection::a_pushed_plan_fires_in_the_runner_and_an_invalid_one_is_refused";
        if let Ok(endpoint) = std::env::var(ENDPOINT) {
            run_runner(&endpoint);
            return;
        }
        let endpoint = endpoint("fault-plan");
        let (listener, mut channel, mut child) = spawn_runner(TEST, &endpoint);

        // No plan yet: the point does nothing.
        assert_eq!(ping(&mut channel), 0);
        // A plan that fires on every hit.
        let always = DaemonFrame::set_fault_plan(&rule(Trigger::Always)).unwrap();
        assert_eq!(
            ask(&mut channel, &always),
            RunnerFrame::FaultPlanSet(Ok(()))
        );
        assert_eq!((ping(&mut channel), ping(&mut channel)), (1, 1));

        // An invalid rule, `nth = 0`, and TOML that is no plan: each refused in the reply, and
        // the plan before stays.
        let zero = DaemonFrame::SetFaultPlan(
            "[[rules]]\npoint = \"ipc.test.point\"\nwhen = { nth = 0 }\neffect = \"panic\"\n"
                .to_owned(),
        );
        let RunnerFrame::FaultPlanSet(Err(why)) = ask(&mut channel, &zero) else {
            panic!("an invalid plan was not refused");
        };
        assert!(why.contains("nth"), "{why}");
        assert_eq!(ping(&mut channel), 1, "the plan before stays");
        let garbage = DaemonFrame::SetFaultPlan("rules = 7".to_owned());
        assert!(matches!(
            ask(&mut channel, &garbage),
            RunnerFrame::FaultPlanSet(Err(_))
        ));
        assert_eq!(ping(&mut channel), 1, "the plan before stays");

        // A new plan replaces it: the second hit after it fires, and only that one.
        let second = DaemonFrame::set_fault_plan(&rule(Trigger::Nth(2))).unwrap();
        assert_eq!(
            ask(&mut channel, &second),
            RunnerFrame::FaultPlanSet(Ok(()))
        );
        let fires: Vec<u32> = (0..3).map(|_| ping(&mut channel)).collect();
        assert_eq!(fires, [0, 1, 0]);

        drop(channel);
        let status = child.wait().unwrap();
        assert!(status.success(), "the runner failed: {status}");
        let endpoint = listener.endpoint().clone();
        drop(listener);
        if let Endpoint::Socket(path) = endpoint {
            let _ = std::fs::remove_dir(path.parent().unwrap());
        }
    }

    /// The fault build's frames: a production message encodes exactly as in a production build,
    /// and `SetFaultPlan` is the hand-built frame the production test refuses, so that test reads
    /// the real bytes.
    #[test]
    fn the_test_variant_sits_at_its_fixed_index_and_leaves_production_frames_alone() {
        assert_eq!(TEST_VARIANT, 1 << 20);
        fn envelope<M>(msg: M) -> Envelope<M> {
            Envelope {
                id: RequestId(1),
                reply_to: None,
                msg,
            }
        }
        for msg in [
            DaemonMsg::Ping,
            DaemonMsg::PeerRevoked {
                node: NodeId([7; 32]),
            },
        ] {
            assert_eq!(
                encode_frame(&envelope(DaemonFrame::Msg(msg.clone()))).unwrap(),
                encode_frame(&envelope(msg.clone())).unwrap()
            );
            let bytes = encode_frame(&envelope(msg.clone())).unwrap();
            assert_eq!(
                decode_frame::<DaemonFrame>(&bytes).unwrap().0.msg,
                DaemonFrame::Msg(msg)
            );
        }
        let plan = DaemonFrame::SetFaultPlan(PLAN_TEXT.to_owned());
        let bytes = encode_frame(&envelope(plan.clone())).unwrap();
        assert_eq!(bytes, fault_frame_bytes(PLAN_TEXT));
        assert_eq!(decode_frame::<DaemonFrame>(&bytes).unwrap().0.msg, plan);
        let reply = Envelope {
            id: RequestId(1),
            reply_to: Some(RequestId(1)),
            msg: RunnerFrame::FaultPlanSet(Ok(())),
        };
        let reply_bytes = encode_frame(&reply).unwrap();
        assert_eq!(
            &reply_bytes[4..],
            &[0x01, 0x01, 0x01, 0x80, 0x80, 0x40, 0x00],
            "the reply bytes the production test refuses"
        );
        assert_eq!(decode_frame::<RunnerFrame>(&reply_bytes).unwrap().0, reply);
        // An index above every production variant but not the test one is still unknown.
        let mut unknown = bytes.clone();
        unknown[8] = 0x41;
        assert!(decode_frame::<DaemonFrame>(&unknown).is_err());
    }
}
