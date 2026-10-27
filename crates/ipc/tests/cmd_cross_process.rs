// SPDX-License-Identifier: MIT OR Apache-2.0

//! The command channel's handshake across two processes (DIST-E1-T03): this test binary plays
//! the daemon, and runs itself again as the runner, which connects to the daemon's endpoint.
//!
//! - A runner of the daemon's build passes the handshake, gets `Init`, reports a state change and
//!   answers a `Ping`.
//! - A runner of another build is killed at the handshake (ADR-0015 item 1).

#![cfg(any(target_os = "linux", target_vendor = "apple", windows))]

use std::process::{Child, Command};
use std::thread;
use std::time::Duration;

use syntropy_core::ledger::LedgerGrant;
use syntropy_core::{Blake3, InstanceId, PlanEpoch};
use syntropy_ipc::cmd::{
    DaemonChannel, DaemonMsg, Endpoint, HandshakeError, Hello, Init, Listener, PROTOCOL,
    RunnerChannel, RunnerMsg, connect,
};
use syntropy_ipc::shm::ShmLayout;

/// The endpoint the runner connects to, set only in the child.
const ENDPOINT: &str = "SYNTROPY_IPC_TEST_CMD_ENDPOINT";
/// The runner's build ID, as one repeated byte in hex, set only in the child.
const RUNNER_BUILD: &str = "SYNTROPY_IPC_TEST_CMD_BUILD";
const BUILD: [u8; 32] = [0x5a; 32];
const OTHER_BUILD: [u8; 32] = [0xa5; 32];
/// A test value of a `RunnerState` discriminant.
const STATE: u32 = 3;
/// How long a child runner lives if nobody ends it, so a failed kill fails the test rather than
/// hanging it.
const RUNNER_LIFETIME: Duration = Duration::from_secs(60);

/// The runner's side, in the child process.
fn run_runner(endpoint: &str, build: &str) {
    thread::spawn(|| {
        thread::sleep(RUNNER_LIFETIME);
        std::process::exit(3);
    });
    let byte = u8::from_str_radix(build, 16).unwrap();
    let endpoint: Endpoint = endpoint.parse().unwrap();
    let mut channel = RunnerChannel::new(connect(&endpoint).unwrap());
    let hello = Hello {
        pid: std::process::id(),
        build_id: [byte; 32],
        ipc_protocol: PROTOCOL,
        obs_tables: Blake3([0x0b; 32]),
    };
    // A runner of another build waits here until the daemon kills it.
    let init = channel.handshake(&hello).unwrap();
    assert_eq!(init.instance, InstanceId(41));
    channel
        .send(&RunnerMsg::StateChanged {
            state: STATE,
            epoch: PlanEpoch { term: 1, seq: 1 },
        })
        .unwrap();
    let ping = channel.recv().unwrap();
    assert_eq!(ping.msg, DaemonMsg::Ping);
    channel.reply(ping.id, &RunnerMsg::Pong).unwrap();
    // The daemon closes the channel when it is done.
    assert!(channel.recv().is_err());
}

/// A fresh endpoint for one test.
fn endpoint(test: &str) -> Endpoint {
    let name = format!("syntropy-ipc-{test}-{}", std::process::id());
    if cfg!(windows) {
        Endpoint::Pipe(name)
    } else {
        Endpoint::Socket(std::env::temp_dir().join(name).join("cmd.sock"))
    }
}

/// Binds `endpoint`, spawns this binary as a runner of `build` running `test`, and accepts its
/// connection.
fn spawn_runner(test: &str, endpoint: &Endpoint, build: u8) -> (Listener, DaemonChannel, Child) {
    let mut listener = Listener::bind(endpoint).unwrap();
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--test-threads", "1"])
        .env(ENDPOINT, endpoint.to_string())
        .env(RUNNER_BUILD, format!("{build:02x}"))
        .spawn()
        .unwrap();
    let stream = listener.accept().unwrap();
    (listener, DaemonChannel::new(stream), child)
}

fn cleanup(listener: Listener) {
    let endpoint = listener.endpoint().clone();
    drop(listener);
    if let Endpoint::Socket(path) = endpoint {
        let _ = std::fs::remove_dir(path.parent().unwrap());
    }
}

#[test]
fn a_runner_of_the_daemons_build_passes_the_handshake() {
    const TEST: &str = "a_runner_of_the_daemons_build_passes_the_handshake";
    if let (Ok(endpoint), Ok(build)) = (std::env::var(ENDPOINT), std::env::var(RUNNER_BUILD)) {
        run_runner(&endpoint, &build);
        return;
    }
    let endpoint = endpoint("same-build");
    let (listener, mut channel, mut child) = spawn_runner(TEST, &endpoint, BUILD[0]);
    let hello = channel.handshake(&mut child, &BUILD).unwrap();
    assert_eq!(hello.pid, child.id());
    assert_eq!(hello.ipc_protocol, PROTOCOL);
    channel
        .send(&DaemonMsg::Init(Init {
            build_id: BUILD,
            instance: InstanceId(41),
            ledger: LedgerGrant::default(),
            shm: ShmLayout::default(),
        }))
        .unwrap();
    let state = channel.recv().unwrap();
    assert_eq!(
        state.msg,
        RunnerMsg::StateChanged {
            state: STATE,
            epoch: PlanEpoch { term: 1, seq: 1 }
        }
    );
    let ping = channel.send(&DaemonMsg::Ping).unwrap();
    let pong = channel.recv().unwrap();
    assert_eq!((pong.reply_to, pong.msg), (Some(ping), RunnerMsg::Pong));
    drop(channel);
    let status = child.wait().unwrap();
    assert!(status.success(), "the runner failed: {status}");
    cleanup(listener);
}

#[test]
fn a_runner_of_another_build_is_killed() {
    const TEST: &str = "a_runner_of_another_build_is_killed";
    if let (Ok(endpoint), Ok(build)) = (std::env::var(ENDPOINT), std::env::var(RUNNER_BUILD)) {
        run_runner(&endpoint, &build);
        panic!("the runner of another build outlived its handshake");
    }
    let endpoint = endpoint("other-build");
    let (listener, mut channel, mut child) = spawn_runner(TEST, &endpoint, OTHER_BUILD[0]);
    let refused = channel.handshake(&mut child, &BUILD).unwrap_err();
    assert!(
        matches!(
            refused.cause,
            HandshakeError::Build { theirs, ours } if theirs == OTHER_BUILD && ours == BUILD
        ),
        "{refused}"
    );
    let status = refused.exit.expect("the runner was killed and reaped");
    assert!(!status.success(), "{status}");
    // Killed, not exited: SIGKILL on Unix; `TerminateProcess` with exit code 1 on Windows, where
    // the runner's own exits would be 101 (a panic) or 3 (its lifetime).
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(status.signal(), Some(libc::SIGKILL), "{status}");
    }
    #[cfg(windows)]
    assert_eq!(status.code(), Some(1), "{status}");
    // The child is reaped: there is nothing left to wait for.
    assert_eq!(child.try_wait().unwrap(), Some(status));
    cleanup(listener);
}
