// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runner integration suite (DIST-E1-T11): one scenario between a runner and the daemon's
//! side, run over each runner host's transport.
//!
//! - **With the in-process host**, `RunnerHost::InProcess`: the runner on a thread of its own,
//!   over an in-memory command channel, with a region shared in the process
//!   (`daemon::runner_host::start_in_process`).
//! - **Over an OS socket**, the transport a spawned runner uses: a Unix-domain socket or a named
//!   pipe, through `ipc`'s `Listener` and `connect`. The spawn itself, re-executing the binary, is
//!   DIST-E1-T04's, so here the runner is a thread on the other end of the socket.
//!
//! The scenario is generic over the transport and identical on both, so it checks that the two
//! hosts are interchangeable:
//! 1. the runner's `Hello` and the daemon's `Init`, with the build, protocol and process checks;
//! 2. a real `Lifecycle` from `Fetching` to `Ready`, each `StateChanged` reaching the supervisor's
//!    `RunnerLink` and its sink once;
//! 3. `Ping` and `Pong`, with the runner writing the shared-memory header's words (heartbeat,
//!    `RunnerState` word, live-sequence count), which `RunnerLink::sample_region` reads;
//! 4. the runner admitting two sequences in place and never stepping: the supervisor kills it
//!    for a stall at the stall bound, through the header alone;
//! 5. `Drain`: `Draining`, then `Stopped`, then the runner closes, and the runner-channel loop
//!    returns.
//!
//! It lives in `cli` because it needs both `runner` and `daemon`, which are both layer 5. It runs
//! in the `unit` suite (`cargo test --workspace`) on every hosted OS, Linux CI included.

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use syntropy_core::ledger::LedgerGrant;
use syntropy_core::{
    AwakeClock, AwakeTime, Blake3, Bytes, Clock, Deadline, DrainReason, Entropy, InstanceId,
    MonoTime, Nanos, NodeId, PlanEpoch, RunnerState, RunnerStateKind,
};
use syntropy_daemon::report::{ReportSink, RunnerFailedReport, RunnerReport};
use syntropy_daemon::runner_host::{RunnerHost, start_in_process};
use syntropy_daemon::runner_link::{Effects, RunnerLink, serve_channel};
use syntropy_daemon::supervisor::{Input, KillReason, Policy, Supervisor};
use syntropy_ipc::cmd::{
    DaemonChannel, DaemonMsg, Endpoint, Hello, Init, Listener, PROTOCOL, RecvError, RunnerChannel,
    RunnerMsg, Transport, connect,
};
use syntropy_ipc::shm::{Region, ShmLayout, Side};
use syntropy_runner::lifecycle::{Event, Lifecycle};

const BUILD: [u8; 32] = [0x7e; 32];
const S: u64 = 1_000_000_000;
const MS: u64 = 1_000_000;
const EPOCH: PlanEpoch = PlanEpoch { term: 1, seq: 1 };
/// The ping at which the runner admits its two sequences.
const ADMIT_AT: u32 = 2;

/// Fixed bytes: the region's name and the back-off's jitter need no real randomness here.
struct Fixed;

impl Entropy for Fixed {
    fn fill(&self, buf: &mut [u8]) {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(29) ^ 0x3c;
        }
    }
}

/// A virtual clock whose two readings advance together, which only the daemon's side reads.
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
struct Seen {
    kills: Vec<KillReason>,
    rejected: Vec<RunnerStateKind>,
}

impl Effects for Seen {
    fn kill(&mut self, reason: KillReason) {
        self.kills.push(reason);
    }
    fn spawn(&mut self) {}
    fn reprobe(&mut self) {}
    fn rejected(&mut self, _: Option<RunnerStateKind>, to: RunnerStateKind, _: PlanEpoch) {
        self.rejected.push(to);
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

/// The runner's live sequences, as its machine holds them.
fn live(machine: &Lifecycle) -> u32 {
    match machine.state() {
        RunnerState::Ready { live_sequences, .. }
        | RunnerState::Draining { live_sequences, .. } => *live_sequences,
        _ => 0,
    }
}

/// The runner writes its header words: a beat, the state word and the live count.
fn write_header(region: &Region, machine: &Lifecycle) {
    region.beat();
    region.set_runner_state(machine.state().kind().discriminant());
    region.set_live_sequences(live(machine));
}

/// The runner's side of the scenario: the handshake, the forward path, `Pong`s, and a drain.
fn runner<T: Transport>(mut channel: RunnerChannel<T>, region: Region) {
    let hello = Hello {
        pid: std::process::id(),
        build_id: BUILD,
        ipc_protocol: PROTOCOL,
        obs_tables: Blake3([0; 32]),
    };
    let init = channel.handshake(&hello).expect("the daemon's Init");
    assert_eq!(init.instance, InstanceId(5));
    let (mut machine, first) = Lifecycle::start(EPOCH, Bytes(1 << 20));
    let mut transitions = vec![first];
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
        transitions.push(machine.apply(event).unwrap().expect("a transition"));
    }
    for msg in transitions {
        write_header(&region, &machine);
        channel.send(&msg).unwrap();
    }
    let mut pings = 0;
    loop {
        match channel.recv() {
            Ok(envelope) => match envelope.msg {
                DaemonMsg::Ping => {
                    pings += 1;
                    if pings == ADMIT_AT {
                        // Two sequences open, in place: no `StateChanged`. The runner never
                        // steps in this scenario.
                        assert_eq!(machine.apply(Event::Sequences { live: 2 }), Ok(None));
                    }
                    write_header(&region, &machine);
                    channel.reply(envelope.id, &RunnerMsg::Pong).unwrap();
                }
                DaemonMsg::Drain { reason, deadline } => {
                    let draining = machine
                        .apply(Event::Drain { reason, deadline })
                        .unwrap()
                        .expect("Draining");
                    write_header(&region, &machine);
                    channel.send(&draining).unwrap();
                    let stopped = machine
                        .apply(Event::Sequences { live: 0 })
                        .unwrap()
                        .expect("Stopped");
                    write_header(&region, &machine);
                    channel.send(&stopped).unwrap();
                    // The runner exits, which closes its end.
                    return;
                }
                other => panic!("the runner was sent {other:?}"),
            },
            Err(RecvError::Closed) => panic!("the daemon closed before the drain"),
            Err(error) => panic!("the channel failed: {error}"),
        }
    }
}

/// The daemon's side: the handshake, the supervisor over the runner's reports and header, the
/// stall kill, and the drain. Returns the pings it took to see the stall.
fn daemon<T: Transport>(channel: &mut DaemonChannel<T>, region: &Region) -> u32 {
    let hello = channel
        .handshake_in_process(&BUILD)
        .expect("the runner's Hello");
    assert_eq!(hello.ipc_protocol, PROTOCOL);
    channel
        .send(&DaemonMsg::Init(Init {
            build_id: BUILD,
            instance: InstanceId(5),
            ledger: LedgerGrant::default(),
            shm: ShmLayout::default(),
        }))
        .unwrap();
    let clock = Virtual(AtomicU64::new(S));
    let mut link = RunnerLink::new(
        InstanceId(5),
        NodeId([2; 32]),
        Supervisor::new(policy(), Fixed),
        Sink::default(),
    );
    let mut seen = Seen::default();
    link.handle(clock.now(), Input::Spawned, &mut seen);
    link.handle(
        clock.now(),
        Input::PredictedStep {
            p99: Nanos(100 * MS),
        },
        &mut seen,
    );
    // The forward path, `Fetching` to `Ready`.
    for _ in 0..5 {
        let envelope = channel.recv().unwrap();
        link.on_runner_msg(clock.now(), &envelope.msg, &mut seen);
    }
    assert_eq!(region.runner_state(), RunnerStateKind::Ready.discriminant());
    assert_eq!(region.live_sequences(), 0);
    // A ping per sample: the runner beats on each, so it is never hung, but after the second it
    // has two live sequences and never steps.
    let mut pings = 0;
    while seen.kills.is_empty() {
        let id = channel.send(&DaemonMsg::Ping).unwrap();
        let pong = channel.recv().unwrap();
        assert_eq!((pong.reply_to, pong.msg), (Some(id), RunnerMsg::Pong));
        pings += 1;
        clock.0.fetch_add(500 * MS, Ordering::Relaxed);
        link.sample_region(&clock, region, &mut seen);
        assert!(pings <= 40, "no stall kill after {pings} pings");
    }
    assert_eq!(seen.kills, [KillReason::Stalled]);
    assert_eq!(region.live_sequences(), 2);
    assert_eq!(region.peer_heartbeat(), 5 + u64::from(pings));
    // The drain: `Draining` and `Stopped`, then the runner closes, and the loop returns.
    channel
        .send(&DaemonMsg::Drain {
            reason: DrainReason::Unload,
            deadline: Deadline::within(Nanos(30 * S)),
        })
        .unwrap();
    serve_channel(channel, &clock, &mut link, &mut seen).expect("the runner closed cleanly");
    let kinds: Vec<_> = link.sink().0.iter().map(|r| r.state).collect();
    assert_eq!(
        kinds,
        [
            RunnerStateKind::Fetching,
            RunnerStateKind::Packing,
            RunnerStateKind::Loading,
            RunnerStateKind::Warming,
            RunnerStateKind::Ready,
            RunnerStateKind::Draining,
            RunnerStateKind::Stopped,
        ]
    );
    assert!(
        link.sink()
            .0
            .iter()
            .all(|r| r.epoch == EPOCH && r.instance == InstanceId(5) && r.node == NodeId([2; 32]))
    );
    assert!(seen.rejected.is_empty());
    assert_eq!(
        region.runner_state(),
        RunnerStateKind::Stopped.discriminant()
    );
    pings
}

/// The pings to the stall kill: the second admits, and from it 20 samples of frozen steps reach
/// the 10 s bound, max(10 s, 20 × 100 ms).
const PINGS_TO_STALL: u32 = ADMIT_AT + 20;

#[test]
fn the_suite_passes_with_the_in_process_host() {
    assert_eq!(RunnerHost::InProcess, RunnerHost::InProcess);
    let mut hosted = start_in_process(&ShmLayout::default(), BUILD, &Fixed, |end| {
        runner(end.channel, end.region)
    })
    .unwrap();
    let pings = daemon(&mut hosted.end.channel, &hosted.end.region);
    assert_eq!(pings, PINGS_TO_STALL);
    hosted.join().expect("the runner thread ends cleanly");
}

#[test]
fn the_suite_passes_over_an_os_socket() {
    let name = format!("syntropy-cli-runner-integration-{}", std::process::id());
    let endpoint = if cfg!(windows) {
        Endpoint::Pipe(name)
    } else {
        Endpoint::Socket(std::env::temp_dir().join(name).join("cmd.sock"))
    };
    let mut listener = Listener::bind(&endpoint).unwrap();
    let layout = ShmLayout::default();
    let region = Region::create(&layout, BUILD, PROTOCOL, &Fixed).unwrap();
    let runner_region = Region::open(&region.handle(), Side::Runner, &layout, PROTOCOL).unwrap();
    let runner_thread = thread::spawn({
        let endpoint = endpoint.clone();
        move || {
            runner(
                RunnerChannel::new(connect(&endpoint).unwrap()),
                runner_region,
            )
        }
    });
    let mut channel = DaemonChannel::new(listener.accept().unwrap());
    let pings = daemon(&mut channel, &region);
    assert_eq!(pings, PINGS_TO_STALL);
    runner_thread
        .join()
        .expect("the runner thread ends cleanly");
    if let Endpoint::Socket(path) = &endpoint {
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
