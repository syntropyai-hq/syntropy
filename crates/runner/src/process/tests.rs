// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::atomic::AtomicU64;
use std::sync::mpsc;

use syntropy_core::ledger::LedgerGrant;
use syntropy_core::{Blake3, Bytes, Clock, Deadline, Entropy, InstanceId, MonoTime, PressureLevel};
use syntropy_ipc::cmd::memory::MemoryStream;
use syntropy_ipc::cmd::{DaemonChannel, Init, PROTOCOL, memory_channel};
use syntropy_ipc::shm::Side;

use super::*;

const S: u64 = 1_000_000_000;
const BUILD: [u8; 32] = [0x3c; 32];

#[test]
fn a_beating_daemon_is_never_silent() {
    let mut watch = DaemonWatch::new(Nanos(2 * S));
    for n in 0..100 {
        assert_eq!(watch.observe(n, AwakeTime(n * S)), None);
    }
}

#[test]
fn a_still_heartbeat_is_silent_after_the_silence_and_not_before() {
    let mut watch = DaemonWatch::new(Nanos(2 * S));
    assert_eq!(
        watch.observe(7, AwakeTime(10 * S)),
        None,
        "the first reading starts the count"
    );
    assert_eq!(watch.observe(7, AwakeTime(11 * S)), None);
    assert_eq!(watch.observe(7, AwakeTime(12 * S - 1)), None);
    assert_eq!(watch.observe(7, AwakeTime(12 * S)), Some(Nanos(2 * S)));
    assert_eq!(watch.observe(7, AwakeTime(15 * S)), Some(Nanos(5 * S)));
    // A beat restarts the count.
    assert_eq!(watch.observe(8, AwakeTime(15 * S)), None);
    assert_eq!(watch.observe(8, AwakeTime(16 * S)), None);
}

#[test]
fn awake_time_that_goes_back_counts_as_none() {
    let mut watch = DaemonWatch::new(Nanos(2 * S));
    assert_eq!(watch.observe(1, AwakeTime(10 * S)), None);
    assert_eq!(watch.observe(1, AwakeTime(3 * S)), None);
    assert_eq!(watch.observe(1, AwakeTime(12 * S)), Some(Nanos(2 * S)));
}

/// Bytes that differ on every call, so parallel tests never draw the same segment name.
struct Varying;

static DRAWS: AtomicU64 = AtomicU64::new(0);

impl Entropy for Varying {
    fn fill(&self, buf: &mut [u8]) {
        let n = DRAWS.fetch_add(1, Ordering::Relaxed);
        let mut x = (u64::from(std::process::id()) << 32 | n).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for byte in buf {
            x ^= x >> 29;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            *byte = (x >> 56) as u8;
        }
    }
}

/// An awake clock a test sets by hand.
#[derive(Clone)]
struct Manual(Arc<AtomicU64>);

impl Clock for Manual {
    fn now(&self) -> MonoTime {
        MonoTime(self.0.load(Ordering::Relaxed))
    }
    fn unix_ms(&self) -> u64 {
        0
    }
}

impl AwakeClock for Manual {
    fn awake(&self) -> AwakeTime {
        AwakeTime(self.0.load(Ordering::Relaxed))
    }
    fn tick(&self) -> Nanos {
        Nanos(1)
    }
}

/// A tick source that records when its rate was first needed, and whether the region had been
/// opened by then.
#[derive(Clone, Default)]
struct Ticks {
    calibrated: Arc<AtomicBool>,
}

impl syntropy_core::TickSource for Ticks {
    fn ticks(&self) -> u64 {
        0
    }
    fn ticks_to_nanos(&self, ticks: u64) -> Nanos {
        self.calibrated.store(true, Ordering::Release);
        Nanos(ticks)
    }
    fn now(&self) -> MonoTime {
        MonoTime(0)
    }
    fn device_to_ticks(&self, device_ts: u64, _: &syntropy_core::DeviceClockSync) -> u64 {
        device_ts
    }
}

/// A runner serving on a thread of its own, and the daemon's side of its channel and region.
struct Harness {
    daemon: DaemonChannel<MemoryStream>,
    region: Arc<Region>,
    clock: Manual,
    silent: mpsc::Receiver<Nanos>,
    runner: JoinHandle<Result<Ended, ServeError>>,
}

fn start() -> Harness {
    let layout = ShmLayout::default();
    let region = Arc::new(Region::create(&layout, BUILD, PROTOCOL, &Varying).unwrap());
    let handle = region.handle();
    let (mut daemon, runner) = memory_channel();
    let clock = Manual(Arc::new(AtomicU64::new(S)));
    let (silent_tx, silent) = mpsc::channel();
    let runner = thread::spawn({
        let clock = clock.clone();
        move || {
            serve(
                runner,
                Start {
                    ticks: Ticks::default(),
                    hello: Hello {
                        pid: std::process::id(),
                        build_id: BUILD,
                        ipc_protocol: PROTOCOL,
                        obs_tables: Blake3([0; 32]),
                    },
                    open_region: |layout: &ShmLayout| {
                        Region::open(&handle, Side::Runner, layout, PROTOCOL)
                    },
                    clock,
                    silence: DAEMON_SILENCE,
                    on_silent: move |still| silent_tx.send(still).unwrap(),
                },
            )
        }
    });
    daemon
        .handshake_in_process(&BUILD)
        .expect("the runner's Hello");
    daemon
        .send(&DaemonMsg::Init(Init {
            build_id: BUILD,
            instance: InstanceId(1),
            ledger: LedgerGrant::default(),
            shm: layout,
        }))
        .unwrap();
    Harness {
        daemon,
        region,
        clock,
        silent,
        runner,
    }
}

/// Waits for the runner's heartbeat to pass `beats`.
fn wait_for_beats(region: &Region, beats: u64) {
    for _ in 0..400 {
        if region.peer_heartbeat() > beats {
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!(
        "the runner's heartbeat stayed at {}",
        region.peer_heartbeat()
    );
}

#[test]
fn the_runner_answers_pings_beats_and_drains() {
    let mut h = start();
    let id = h.daemon.send(&DaemonMsg::Ping).unwrap();
    let pong = h.daemon.recv().unwrap();
    assert_eq!((pong.reply_to, pong.msg), (Some(id), RunnerMsg::Pong));
    // A message this build does not handle yet is logged and ignored.
    h.daemon
        .send(&DaemonMsg::MemoryPressure {
            level: PressureLevel::Moderate,
            reclaim: Bytes(0),
            tier: None,
        })
        .unwrap();
    let id = h.daemon.send(&DaemonMsg::Ping).unwrap();
    assert_eq!(h.daemon.recv().unwrap().reply_to, Some(id));
    wait_for_beats(&h.region, 1);
    h.region.beat();
    h.daemon
        .send(&DaemonMsg::Drain {
            reason: DrainReason::Unload,
            deadline: Deadline::within(Nanos(S)),
        })
        .unwrap();
    assert_eq!(
        h.runner.join().unwrap().unwrap(),
        Ended::Drained(DrainReason::Unload)
    );
    assert!(h.silent.try_recv().is_err(), "no silence was reported");
}

#[test]
fn a_shutdown_ends_the_runner() {
    let mut h = start();
    h.daemon
        .send(&DaemonMsg::Shutdown {
            deadline: Deadline::within(Nanos(S)),
        })
        .unwrap();
    assert_eq!(h.runner.join().unwrap().unwrap(), Ended::ShutDown);
}

/// The daemon's end closes, as when its process is killed: the runner reads end-of-file and
/// returns, and its beat thread stops.
#[test]
fn end_of_file_ends_the_runner() {
    let h = start();
    drop(h.daemon);
    assert_eq!(h.runner.join().unwrap().unwrap(), Ended::DaemonClosed);
    let beats = h.region.peer_heartbeat();
    thread::sleep(HEARTBEAT * 3);
    assert_eq!(h.region.peer_heartbeat(), beats, "the beat thread stopped");
}

/// The daemon's heartbeat stands still with its end open, as when it is frozen: once
/// `DAEMON_SILENCE` of awake time has passed with no beat, the beat thread reports it. While the
/// daemon beats, the same awake time passing reports nothing.
#[test]
fn a_still_daemon_heartbeat_is_reported_after_the_silence() {
    let h = start();
    wait_for_beats(&h.region, 0);
    // The daemon beats while the awake time passes the silence in steps: no report.
    for _ in 0..4 {
        h.region.beat();
        h.clock.0.fetch_add(DAEMON_SILENCE.0 / 2, Ordering::Relaxed);
        let seen = h.region.peer_heartbeat();
        wait_for_beats(&h.region, seen);
    }
    assert!(
        h.silent.try_recv().is_err(),
        "a beating daemon is not silent"
    );
    // The daemon stops beating, and the awake time passes the silence.
    let seen = h.region.peer_heartbeat();
    wait_for_beats(&h.region, seen);
    h.clock.0.fetch_add(DAEMON_SILENCE.0, Ordering::Relaxed);
    let still = h
        .silent
        .recv_timeout(Duration::from_secs(10))
        .expect("the silence was reported");
    assert!(still >= DAEMON_SILENCE, "reported after {still:?}");
    drop(h.daemon);
    assert_eq!(h.runner.join().unwrap().unwrap(), Ended::DaemonClosed);
}

/// An `Init` of another build ends the runner before it opens anything.
#[test]
fn an_init_of_another_build_fails_the_handshake() {
    let (mut daemon, runner) = memory_channel();
    let opened = Arc::new(AtomicBool::new(false));
    let thread = thread::spawn({
        let opened = Arc::clone(&opened);
        move || {
            serve(
                runner,
                Start {
                    ticks: Ticks::default(),
                    hello: Hello {
                        pid: std::process::id(),
                        build_id: BUILD,
                        ipc_protocol: PROTOCOL,
                        obs_tables: Blake3([0; 32]),
                    },
                    open_region: move |_: &ShmLayout| -> Result<Region, ShmError> {
                        opened.store(true, Ordering::Release);
                        unreachable!("the region is never opened")
                    },
                    clock: Manual(Arc::new(AtomicU64::new(0))),
                    silence: DAEMON_SILENCE,
                    on_silent: |_| {},
                },
            )
        }
    });
    daemon.handshake_in_process(&BUILD).unwrap();
    daemon
        .send(&DaemonMsg::Init(Init {
            build_id: [0; 32],
            instance: InstanceId(1),
            ledger: LedgerGrant::default(),
            shm: ShmLayout::default(),
        }))
        .unwrap();
    assert!(matches!(
        thread.join().unwrap(),
        Err(ServeError::Handshake(_))
    ));
    assert!(!opened.load(Ordering::Acquire));
}

/// Each lifecycle transition writes the state word in a real region's header, and the daemon's
/// side reads it; an event the state does not take leaves the word as it was.
#[test]
fn each_transition_writes_the_state_word_the_daemon_reads() {
    use syntropy_core::{PlanEpoch, RunnerStateKind};

    use crate::lifecycle::Event;

    let layout = ShmLayout::default();
    let daemon = Region::create(&layout, BUILD, PROTOCOL, &Varying).unwrap();
    let runner = Region::open(&daemon.handle(), Side::Runner, &layout, PROTOCOL).unwrap();
    let epoch = PlanEpoch { term: 1, seq: 1 };
    let (mut machine, first) = begin(&runner, epoch, Bytes(1 << 20));
    assert!(matches!(first, RunnerMsg::StateChanged { .. }));
    assert_eq!(
        daemon.runner_state(),
        RunnerStateKind::Fetching.discriminant()
    );
    for (event, kind) in [
        (
            Event::ShardVerified {
                pack: Bytes(1 << 20),
            },
            RunnerStateKind::Packing,
        ),
        (
            Event::PackCommitted {
                resident: Bytes(1 << 20),
            },
            RunnerStateKind::Loading,
        ),
        (Event::Resident, RunnerStateKind::Warming),
        (Event::WarmPassed, RunnerStateKind::Ready),
    ] {
        let msg = advance(&mut machine, event, &runner).unwrap();
        assert!(matches!(msg, Some(RunnerMsg::StateChanged { .. })));
        assert_eq!(daemon.runner_state(), kind.discriminant(), "{kind:?}");
    }
    assert!(advance(&mut machine, Event::Resident, &runner).is_err());
    assert_eq!(daemon.runner_state(), RunnerStateKind::Ready.discriminant());
}

/// `serve` reads the tick source's rate before anything else: before the handshake, so before the
/// region opens and before any thread it starts reads ticks.
#[test]
fn serve_calibrates_the_tick_source_first() {
    let layout = ShmLayout::default();
    let region = Region::create(&layout, BUILD, PROTOCOL, &Varying).unwrap();
    let handle = region.handle();
    let (mut daemon, runner) = memory_channel();
    let ticks = Ticks::default();
    let seen_at_open = Arc::new(AtomicBool::new(false));
    let thread = thread::spawn({
        let ticks = ticks.clone();
        let seen_at_open = Arc::clone(&seen_at_open);
        move || {
            let calibrated = Arc::clone(&ticks.calibrated);
            serve(
                runner,
                Start {
                    ticks,
                    hello: Hello {
                        pid: std::process::id(),
                        build_id: BUILD,
                        ipc_protocol: PROTOCOL,
                        obs_tables: Blake3([0; 32]),
                    },
                    open_region: move |layout: &ShmLayout| {
                        seen_at_open.store(calibrated.load(Ordering::Acquire), Ordering::Release);
                        Region::open(&handle, Side::Runner, layout, PROTOCOL)
                    },
                    clock: Manual(Arc::new(AtomicU64::new(S))),
                    silence: DAEMON_SILENCE,
                    on_silent: |_| {},
                },
            )
        }
    });
    daemon.handshake_in_process(&BUILD).unwrap();
    assert!(
        ticks.calibrated.load(Ordering::Acquire),
        "calibrated before Hello"
    );
    daemon
        .send(&DaemonMsg::Init(Init {
            build_id: BUILD,
            instance: InstanceId(1),
            ledger: LedgerGrant::default(),
            shm: layout,
        }))
        .unwrap();
    drop(daemon);
    assert_eq!(thread.join().unwrap().unwrap(), Ended::DaemonClosed);
    assert!(seen_at_open.load(Ordering::Acquire));
}

/// A daemon that reads `Hello` and then stands still, never sending `Init`, as one frozen between
/// the two would: the wait for `Init` counts as silence, and is reported after it.
#[test]
fn a_daemon_that_never_sends_init_is_reported_silent() {
    let (mut daemon, runner) = memory_channel();
    let clock = Manual(Arc::new(AtomicU64::new(S)));
    let (silent_tx, silent) = mpsc::channel();
    let thread = thread::spawn({
        let clock = clock.clone();
        move || {
            serve(
                runner,
                Start {
                    ticks: Ticks::default(),
                    hello: Hello {
                        pid: std::process::id(),
                        build_id: BUILD,
                        ipc_protocol: PROTOCOL,
                        obs_tables: Blake3([0; 32]),
                    },
                    open_region: |_: &ShmLayout| -> Result<Region, ShmError> {
                        unreachable!("no Init, so no region")
                    },
                    clock,
                    silence: DAEMON_SILENCE,
                    on_silent: move |still| silent_tx.send(still).unwrap(),
                },
            )
        }
    });
    daemon.handshake_in_process(&BUILD).unwrap();
    // Let the beat thread take its first reading, then pass the silence.
    thread::sleep(HEARTBEAT * 3);
    assert!(silent.try_recv().is_err(), "no report before the silence");
    clock.0.fetch_add(DAEMON_SILENCE.0, Ordering::Relaxed);
    let still = silent
        .recv_timeout(Duration::from_secs(10))
        .expect("the wait for Init was reported");
    assert!(still >= DAEMON_SILENCE, "{still:?}");
    drop(daemon);
    assert!(matches!(
        thread.join().unwrap(),
        Err(ServeError::Handshake(_))
    ));
}

#[test]
fn a_reset_or_broken_channel_means_the_daemon_is_gone() {
    use std::io::{Error, ErrorKind};
    for kind in [
        ErrorKind::ConnectionReset,
        ErrorKind::ConnectionAborted,
        ErrorKind::BrokenPipe,
        ErrorKind::UnexpectedEof,
    ] {
        assert!(gone(&Error::from(kind)), "{kind:?}");
    }
    assert!(!gone(&Error::from(ErrorKind::InvalidData)));
    assert_eq!(gone(&Error::from_raw_os_error(233)), cfg!(windows));
}
