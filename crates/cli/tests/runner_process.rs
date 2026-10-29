// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runners as processes (DIST-E1-T04): spawned through `daemon`'s process host, which re-executes
//! the `syntropy` binary as `__runner`, and ended by the daemon's loss with no parent-death signal
//! (RFC-0052 §6.5).
//!
//! The daemon-loss tests need a daemon that can be killed or frozen apart from the test, so the
//! test re-runs its own executable as a helper daemon (the ignored `helper_daemon` test, which
//! acts only with `SYNTROPY_TEST_HELPER` set). The helper spawns a runner, completes the handshake,
//! prints the runner's process ID and then waits, beating its heartbeat. The test then kills the
//! helper with `SIGKILL` or `TerminateProcess` (`Child::kill`), or freezes it with `SIGSTOP`, and
//! watches the runner's process from outside.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use syntropy_core::ledger::LedgerGrant;
use syntropy_core::{Deadline, DrainReason, Entropy, InstanceId, Nanos};
use syntropy_daemon::runner_host::{ProcessRunner, spawn_process};
use syntropy_ipc::cmd::{DaemonMsg, Init, RunnerMsg};
use syntropy_ipc::shm::ShmLayout;

/// The `syntropy` binary cargo built for this test.
fn binary() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_syntropy"))
}

/// Bytes that differ on every call, in every test and process, so no two regions or pipes share a
/// name. Production passes the OS entropy source.
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

/// The binary's 32-byte build ID, as it reports it, so the handshake pairs whatever features
/// cargo built the binary with.
fn build_id() -> [u8; 32] {
    let output = Command::new(binary())
        .args(["--version", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let id = json["id"].as_str().expect("the build's id, as hex");
    let mut bytes = [0; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&id[2 * i..2 * i + 2], 16).unwrap();
    }
    bytes
}

/// Spawns a runner, and completes the handshake and `Init`.
fn spawn_ready() -> ProcessRunner {
    let build = build_id();
    let layout = ShmLayout::default();
    let mut runner = spawn_process(binary(), &layout, build, &Varying).unwrap();
    let hello = runner.handshake().expect("the runner's Hello");
    assert_eq!(hello.pid, runner.id());
    runner
        .channel()
        .send(&DaemonMsg::Init(Init {
            build_id: build,
            instance: InstanceId(9),
            ledger: LedgerGrant::default(),
            shm: layout,
        }))
        .unwrap();
    runner
}

/// Reaps `runner`, failing after `limit` rather than hanging the suite; the reap goes through
/// `try_wait`, as a supervisor's polling does.
fn reap_within(runner: &mut ProcessRunner, limit: Duration) -> ExitStatus {
    let deadline = Instant::now() + limit;
    loop {
        if let Some(status) = runner.try_wait().unwrap() {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "runner {} still runs after {limit:?}",
            runner.id()
        );
        thread::sleep(Duration::from_millis(10));
    }
}

/// A spawned runner pairs, answers a ping, beats its heartbeat in the region, and exits 0 on a
/// drain; the host then reaps it and removes the region's name.
#[test]
fn a_spawned_runner_pairs_answers_beats_and_drains() {
    let mut runner = spawn_ready();
    let id = runner.channel().send(&DaemonMsg::Ping).unwrap();
    let pong = runner.channel().recv().unwrap();
    assert_eq!((pong.reply_to, pong.msg), (Some(id), RunnerMsg::Pong));
    let beats = runner.region().peer_heartbeat();
    let deadline = Instant::now() + Duration::from_secs(5);
    while runner.region().peer_heartbeat() <= beats + 1 {
        assert!(
            Instant::now() < deadline,
            "the runner's heartbeat stayed still"
        );
        thread::sleep(Duration::from_millis(20));
    }
    runner
        .channel()
        .send(&DaemonMsg::Drain {
            reason: DrainReason::Unload,
            deadline: Deadline::within(Nanos(1_000_000_000)),
        })
        .unwrap();
    assert!(!runner.region_unlinked());
    let status = reap_within(&mut runner, Duration::from_secs(30));
    assert!(status.success(), "{status}");
    assert!(
        runner.region_unlinked(),
        "reaping the runner removed the region's name"
    );
}

/// A runner whose channel closes before its `Init` exits without opening its region, and the host
/// removes the region's name when it reaps it. On Apple platforms the name is then gone; a Linux
/// memfd has no name, and a Windows mapping's name goes with its last handle, so there the host's
/// record of the call is what is checked.
#[test]
fn the_host_removes_the_region_name_when_the_runner_exits() {
    let mut runner = spawn_process(binary(), &ShmLayout::default(), build_id(), &Varying).unwrap();
    let handle = runner.region().handle();
    runner.close();
    let status = reap_within(&mut runner, Duration::from_secs(30));
    assert!(!status.success(), "a runner with no Init fails: {status}");
    assert!(runner.region_unlinked());
    if cfg!(target_vendor = "apple") {
        let reopened = syntropy_ipc::shm::Region::open(
            &handle,
            syntropy_ipc::shm::Side::Runner,
            &ShmLayout::default(),
            syntropy_ipc::cmd::PROTOCOL,
        );
        assert!(reopened.is_err(), "the region's name was removed");
    }
}

/// A program that is not there: the spawn is an error, and the region it built is not left named
/// (`spawn_process` removes the name on every error after building it).
#[test]
fn a_failed_spawn_is_an_error() {
    let missing = std::env::temp_dir().join("syntropy-no-such-binary");
    let error = spawn_process(&missing, &ShmLayout::default(), [0; 32], &Varying)
        .expect_err("no such program");
    assert!(
        matches!(error, syntropy_ipc::shm::ShmError::Os(_)),
        "{error}"
    );
}

/// The helper daemon's environment variable.
const HELPER: &str = "SYNTROPY_TEST_HELPER";

/// How long a helper daemon lives at most, so one a failed test leaves behind still ends.
const HELPER_LIFETIME: Duration = Duration::from_secs(120);

/// The helper daemon (module docs). Run by the tests below; ignored otherwise.
#[test]
#[ignore = "a helper process the daemon-loss tests start"]
fn helper_daemon() {
    if std::env::var_os(HELPER).is_none() {
        return;
    }
    let runner = spawn_ready();
    println!("runner-pid {}", runner.id());
    let end = Instant::now() + HELPER_LIFETIME;
    while Instant::now() < end {
        thread::park_timeout(Duration::from_secs(1));
    }
    // The runner reads end-of-file and exits; the drop does not wait for it past its bound.
    std::process::exit(0);
}

/// A helper daemon, killed when dropped, so a failing assertion never leaves one running.
struct Helper(Child);

impl Drop for Helper {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Starts the helper daemon, and returns it with its runner's process ID. Fails after 60 s
/// without one, so a broken helper never hangs the suite.
fn start_helper() -> (Helper, u32) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "helper_daemon",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(HELPER, "1")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = BufReader::new(child.stdout.take().unwrap());
    let helper = Helper(child);
    let (found, pid) = std::sync::mpsc::channel();
    thread::spawn(move || {
        for line in stdout.lines() {
            // libtest prints `test helper_daemon ... ` before the test runs, on the same line.
            if let Some((_, pid)) = line.unwrap().split_once("runner-pid ") {
                let _ = found.send(pid.trim().parse::<u32>().unwrap());
                return;
            }
        }
    });
    let pid = pid
        .recv_timeout(Duration::from_secs(60))
        .expect("the helper daemon printed its runner within 60 s");
    (helper, pid)
}

/// Whether process `pid` is running: listed, and not a zombie. Uses `ps` on Linux and Apple
/// platforms and `tasklist` on Windows, which every hosted runner has.
fn running(pid: u32) -> bool {
    if cfg!(windows) {
        let output = Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
            .output()
            .unwrap();
        String::from_utf8_lossy(&output.stdout).contains(&format!("\"{pid}\""))
    } else {
        let output = Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .unwrap();
        let stat = String::from_utf8_lossy(&output.stdout);
        let stat = stat.trim();
        !stat.is_empty() && !stat.starts_with('Z')
    }
}

/// How long after `since` process `pid` was first seen not running, polling until `limit` after
/// `since`; `None` if it still ran at the last poll. Each poll's time is counted at its start, so a
/// slow poll, as `tasklist` can be, never makes the process look later than it was.
fn ended_after(pid: u32, since: Instant, limit: Duration) -> Option<Duration> {
    loop {
        let at = since.elapsed();
        if !running(pid) {
            return Some(at);
        }
        if at > limit {
            return None;
        }
        thread::sleep(Duration::from_millis(10));
    }
}

/// Killing the daemon with `SIGKILL` (`TerminateProcess` on Windows) ends its runner within 1 s,
/// through end-of-file on the command channel.
#[test]
fn killing_the_daemon_ends_its_runner_within_a_second() {
    let (mut helper, pid) = start_helper();
    assert!(running(pid), "the runner runs while its daemon does");
    let killed = Instant::now();
    helper.0.kill().unwrap();
    helper.0.wait().unwrap();
    let took = ended_after(pid, killed, Duration::from_secs(1));
    eprintln!("runner {pid} ended {took:?} after its daemon was killed");
    let took =
        took.unwrap_or_else(|| panic!("runner {pid} still ran 1 s after its daemon was killed"));
    assert!(took <= Duration::from_secs(1), "{took:?}");
}

/// Freezing the daemon with `SIGSTOP` leaves its end of the channel open, so the runner finds it
/// lost by its still heartbeat, after `runner.daemon_silence` (2 s). The lower bound is loose: the
/// daemon's last beat can come up to 250 ms before the stop, and the stop itself takes a moment,
/// so only an end well before the silence shows a fault.
#[cfg(unix)]
#[test]
fn a_frozen_daemon_ends_its_runner_after_the_silence() {
    let signal = |name: &str, pid: u32| {
        let status = Command::new("kill")
            .args([name, &pid.to_string()])
            .status()
            .unwrap();
        assert!(status.success(), "kill {name} {pid}");
    };
    let (helper, pid) = start_helper();
    let stopping = Instant::now();
    signal("-STOP", helper.0.id());
    let took = ended_after(pid, stopping, Duration::from_secs(5));
    signal("-CONT", helper.0.id());
    drop(helper);
    eprintln!("runner {pid} ended {took:?} after its daemon froze");
    let took = took.unwrap_or_else(|| panic!("runner {pid} still ran 5 s after its daemon froze"));
    assert!(
        took >= Duration::from_millis(1_000),
        "the runner ended {took:?} after its daemon froze, well before runner.daemon_silence"
    );
}

/// A clock at a fixed wall time, for the crash record's timestamp.
struct FixedClock;

impl syntropy_core::Clock for FixedClock {
    fn now(&self) -> syntropy_core::MonoTime {
        syntropy_core::MonoTime(0)
    }

    fn unix_ms(&self) -> u64 {
        1_800_000_000_000
    }
}

/// A fresh state directory, removed on drop.
struct State(std::path::PathBuf);

impl State {
    fn new(tag: &str) -> State {
        let dir = std::env::temp_dir().join(format!(
            "syntropy-cli-runner-crash-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        State(dir)
    }
}

impl Drop for State {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A crash sink over `state`, as the daemon's wiring gives the host one.
fn sink(state: &State) -> syntropy_daemon::crash::CrashSink {
    use std::sync::Arc;
    syntropy_daemon::crash::CrashSink {
        store: syntropy_daemon::crash::CrashStore::new(&state.0),
        build_id: "ab".repeat(32),
        runner: "instance 9 stage 0".into(),
        clock: Arc::new(FixedClock),
        tail: Arc::new(syntropy_daemon::crash::LogTail::new()),
    }
}

/// A runner the host kills for the supervisor's reason leaves a crash record that names the
/// reason, on every OS, and a diagnostic bundle built afterwards holds it (PROD-E9-T12;
/// RFC-0053 §6). The record comes from the host's own reason, not from the exit status: on
/// Windows `TerminateProcess` leaves exit code 1, which is no crash by its status alone.
#[test]
fn a_runner_the_host_kills_leaves_a_crash_record_naming_the_reason_on_every_os() {
    use syntropy_daemon::bundle::{BundleOptions, Sources, build, verify};
    use syntropy_daemon::crash::{CrashCause, CrashProcess, CrashStore, HostKill};
    use syntropy_daemon::supervisor::KillReason;

    let state = State::new("host-kill");
    let mut runner = spawn_ready();
    let pid = runner.id();
    runner.record_crashes(sink(&state));
    runner.kill(HostKill::from(KillReason::Hung)).unwrap();
    let status = reap_within(&mut runner, Duration::from_secs(30));
    assert!(!status.success(), "{status}");

    let store = CrashStore::new(&state.0);
    let listing = store.list().unwrap();
    assert!(listing.unreadable.is_empty());
    assert_eq!(
        listing.records.len(),
        1,
        "one crash record for the killed runner"
    );
    let (name, record) = &listing.records[0];
    let signal = if cfg!(unix) { Some(9) } else { None };
    assert_eq!(
        record.cause,
        CrashCause::HostKilled {
            reason: HostKill::Supervisor(KillReason::Hung),
            signal
        }
    );
    assert_eq!(
        record.process,
        CrashProcess::Runner {
            pid,
            runner: "instance 9 stage 0".into()
        }
    );

    let sources = Sources {
        crashes: Some(&store),
        ..Sources::default()
    };
    let bundle = build(&sources, &BundleOptions::at(1_800_000_000_500)).unwrap();
    let files = verify(bundle.bytes()).unwrap();
    let file = format!("crashes/{name}");
    assert!(files.contains(&file), "{files:?}");
    assert!(bundle.manifest().contains(&file));
    drop(runner);
}

/// A freshly spawned runner whose `Hello` the host refuses, here for another build ID, is
/// killed (RFC-0001 §3.4) and leaves a crash record naming the refusal, on every OS
/// (PROD-E9-T12).
#[test]
fn a_runner_whose_handshake_is_refused_leaves_a_crash_record_on_every_os() {
    use syntropy_daemon::crash::{CrashCause, CrashStore, HostKill};

    let state = State::new("refused");
    let mut wrong = build_id();
    wrong[0] ^= 0xff;
    let mut runner = spawn_process(binary(), &ShmLayout::default(), wrong, &Varying).unwrap();
    runner.record_crashes(sink(&state));
    assert!(
        runner.handshake().is_err(),
        "a Hello of another build is refused"
    );
    let listing = CrashStore::new(&state.0).list().unwrap();
    assert_eq!(
        listing.records.len(),
        1,
        "one record for the refused runner"
    );
    match &listing.records[0].1.cause {
        CrashCause::HostKilled { reason, signal } => {
            assert_eq!(*reason, HostKill::Refused);
            if cfg!(windows) {
                assert_eq!(*signal, None);
            }
        }
        other => panic!("expected a host kill, got {other:?}"),
    }
    drop(runner);
}

/// Spawns this test binary as the runner: given the runner's arguments, it refuses them and
/// exits at once, without a `Hello`, standing in for `/bin/false` on every OS.
fn spawn_quitter(build: [u8; 32]) -> ProcessRunner {
    let quitter = std::env::current_exe().unwrap();
    spawn_process(&quitter, &ShmLayout::default(), build, &Varying).unwrap()
}

/// Reads the runner's channel until it ends: the runner has exited, and is not reaped yet.
fn until_closed(runner: &mut ProcessRunner) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while runner.channel().recv().is_ok() {
        assert!(
            Instant::now() < deadline,
            "the runner's channel stayed open"
        );
    }
}

/// The review of #1973, B1(a): a kill of a runner that has already exited, not yet reaped, is
/// not the host's kill. The kill reaps it, and its exit is recorded from its status: here an
/// ordinary exit, so no record at all.
#[test]
fn a_kill_after_the_runner_exited_is_not_recorded_as_the_hosts() {
    use syntropy_daemon::crash::{CrashStore, HostKill};
    use syntropy_daemon::supervisor::KillReason;

    let state = State::new("kill-after-exit");
    let mut runner = spawn_quitter(build_id());
    runner.record_crashes(sink(&state));
    until_closed(&mut runner);
    runner.kill(HostKill::from(KillReason::Hung)).unwrap();
    let status = reap_within(&mut runner, Duration::from_secs(30));
    assert!(
        !status.success(),
        "the quitter exits with an error code: {status}"
    );
    let listing = CrashStore::new(&state.0).list().unwrap();
    assert!(listing.records.is_empty(), "{:?}", listing.records);
    drop(runner);
}

/// The review of #1973, B1(a): a runner that crashed just before the host's kill is recorded
/// as that crash, not as the host's kill. Deterministic: the runner dies of `SIGABRT` from
/// outside, the test waits for its channel to close, and only then does the host kill it.
/// Unix only, for the signal.
#[cfg(unix)]
#[test]
fn a_runner_that_crashed_just_before_the_kill_is_recorded_as_its_crash() {
    use syntropy_daemon::crash::{CrashCause, CrashStore, HostKill};
    use syntropy_daemon::supervisor::KillReason;

    let state = State::new("crash-before-kill");
    let mut runner = spawn_ready();
    runner.record_crashes(sink(&state));
    let sent = Command::new("kill")
        .args(["-ABRT", &runner.id().to_string()])
        .status()
        .unwrap();
    assert!(sent.success());
    until_closed(&mut runner);
    runner.kill(HostKill::from(KillReason::Stalled)).unwrap();
    reap_within(&mut runner, Duration::from_secs(30));
    let listing = CrashStore::new(&state.0).list().unwrap();
    assert_eq!(listing.records.len(), 1, "{:?}", listing.records);
    assert_eq!(
        listing.records[0].1.cause,
        CrashCause::Signal {
            signal: 6,
            name: Some("SIGABRT".into())
        }
    );
    drop(runner);
}

/// A runner that drains and exits cleanly leaves no crash record, on every OS.
#[test]
fn a_runner_that_exits_cleanly_leaves_no_crash_record() {
    let state = State::new("clean");
    let mut runner = spawn_ready();
    runner.record_crashes(sink(&state));
    runner
        .channel()
        .send(&DaemonMsg::Drain {
            reason: DrainReason::Unload,
            deadline: Deadline::within(Nanos(1_000_000_000)),
        })
        .unwrap();
    let status = reap_within(&mut runner, Duration::from_secs(30));
    assert!(status.success(), "{status}");
    let listing = syntropy_daemon::crash::CrashStore::new(&state.0)
        .list()
        .unwrap();
    assert!(listing.records.is_empty(), "{:?}", listing.records);
    drop(runner);
}

/// A runner killed by a signal from outside the host, not by the host, is recorded from its
/// exit status, as `Signal` (RFC-0053 §6). Unix only: Windows has no signals.
#[cfg(unix)]
#[test]
fn a_runner_killed_by_another_process_is_recorded_from_its_signal() {
    use std::os::unix::process::ExitStatusExt as _;

    use syntropy_daemon::crash::{CrashCause, CrashStore};

    let state = State::new("external");
    let mut runner = spawn_ready();
    runner.record_crashes(sink(&state));
    let killed = Command::new("kill")
        .args(["-KILL", &runner.id().to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
    let status = reap_within(&mut runner, Duration::from_secs(30));
    assert_eq!(status.signal(), Some(9), "{status}");
    let listing = CrashStore::new(&state.0).list().unwrap();
    assert_eq!(listing.records.len(), 1);
    assert_eq!(
        listing.records[0].1.cause,
        CrashCause::Signal {
            signal: 9,
            name: Some("SIGKILL".into())
        }
    );
    drop(runner);
}
