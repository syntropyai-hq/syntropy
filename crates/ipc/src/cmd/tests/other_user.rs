// SPDX-License-Identifier: MIT OR Apache-2.0

//! A connection from another OS user is refused (DIST-E1-T03; RFC-0034 §6.4).
//!
//! The test runs a copy of this test binary as a second account, through `sudo -n -u`, and has
//! it connect twice:
//!
//! 1. to a listener from [`Listener::bind`], whose directory has mode 0700. The OS refuses the
//!    connection before the daemon sees it.
//! 2. to a listener on a socket every user can reach, so the connection reaches the peer check,
//!    which refuses it and names the account's user ID.
//!
//! CI's hosted Linux and macOS jobs create the account and name it in `SYNTROPY_IPC_PEER_USER`.
//! Without it the test prints a notice and passes, except in CI, where it fails. Its files sit
//! under `/tmp`, not the temporary directory, which macOS gives each user as a private 0700
//! directory the other account could not enter.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::{ErrorKind, Read};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use super::super::{AcceptError, Endpoint, Listener, Peer};

/// The second account's name.
const USER: &str = "SYNTROPY_IPC_PEER_USER";
/// Set only in the copy that runs as the second account: the socket to connect to.
const CONNECT: &str = "SYNTROPY_IPC_PEER_CONNECT";
/// This test, which the copy runs.
const TEST: &str = "cmd::tests::other_user::a_connection_from_another_user_is_refused";

#[test]
fn a_connection_from_another_user_is_refused() {
    if let Some(socket) = env::var_os(CONNECT) {
        connect_as_peer(Path::new(&socket));
        return;
    }
    let Some(user) = env::var_os(USER).filter(|user| !user.is_empty()) else {
        let in_ci = env::var_os("CI").is_some_and(|v| !v.is_empty() && v != "false");
        assert!(
            !in_ci,
            "CI must run this test: create a second OS account and name it in {USER}"
        );
        eprintln!(
            "skipped: no second OS account named in {USER}, so the other-user refusal did not run"
        );
        return;
    };

    // Everything the other account touches sits under a directory it may enter.
    let base = Path::new("/tmp").join(format!("syntropy-ipc-peer-{}", std::process::id()));
    fs::create_dir(&base).unwrap();
    fs::set_permissions(&base, fs::Permissions::from_mode(0o755)).unwrap();
    let exe = base.join("peer");
    fs::copy(env::current_exe().unwrap(), &exe).unwrap();
    fs::set_permissions(&exe, fs::Permissions::from_mode(0o755)).unwrap();
    // SAFETY: no arguments, and it cannot fail.
    let our_uid = unsafe { libc::geteuid() };

    // 1. The endpoint as the daemon binds it: the directory keeps the other user out.
    let private = base.join("private").join("cmd.sock");
    let listener = Listener::bind(&Endpoint::Socket(private.clone())).unwrap();
    let result = run_as(&user, &exe, &private);
    assert!(
        result.outcome == "connect-error PermissionDenied",
        "the other user reached a socket in a 0700 directory: {}",
        result.outcome
    );
    assert_ne!(result.uid, our_uid, "{USER} names this test's own user");
    drop(listener);

    // 2. A socket anyone can reach: the peer check refuses the connection.
    let exposed = base.join("exposed");
    fs::create_dir(&exposed).unwrap();
    fs::set_permissions(&exposed, fs::Permissions::from_mode(0o755)).unwrap();
    let path = exposed.join("cmd.sock");
    let socket = UnixListener::bind(&path).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o777)).unwrap();
    socket.set_nonblocking(true).unwrap();
    let mut listener = Listener::from_std(socket, path.clone());
    let peer = {
        let (user, exe, path) = (user.clone(), exe.clone(), path.clone());
        thread::spawn(move || run_as(&user, &exe, &path))
    };
    let deadline = Instant::now() + Duration::from_secs(60);
    let accepted = loop {
        match listener.accept() {
            Err(AcceptError::Io(error)) if error.kind() == ErrorKind::WouldBlock => {
                assert!(Instant::now() < deadline, "the other user never connected");
                thread::sleep(Duration::from_millis(10));
            }
            other => break other,
        }
    };
    let result = peer.join().unwrap();
    match accepted {
        Err(AcceptError::Refused(Peer::Unix { uid, .. })) => assert_eq!(uid, result.uid),
        Err(error) => panic!("the connection failed in another way: {error}"),
        Ok(_) => panic!("the peer check accepted user {}", result.uid),
    }
    assert!(
        matches!(
            result.outcome.as_str(),
            "closed" | "read-error ConnectionReset"
        ),
        "the refused peer saw: {}",
        result.outcome
    );
    drop(listener);
    fs::remove_dir_all(&base).unwrap();
}

/// What the copy running as the other account saw.
struct PeerResult {
    uid: u32,
    outcome: String,
}

/// Runs this test's copy `exe` as `user`, connecting to `socket`, and reads what it saw.
fn run_as(user: &OsStr, exe: &Path, socket: &Path) -> PeerResult {
    let output = Command::new("sudo")
        .args(["-n", "-u"])
        .arg(user)
        .args(["--", "/usr/bin/env"])
        .arg(format!("{CONNECT}={}", socket.display()))
        .arg(exe)
        .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
        .output()
        .expect("run sudo");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the peer process failed ({}):\n{stdout}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    // libtest prints the test's name first, on the same line, so the result is found anywhere.
    let line = stdout
        .lines()
        .find_map(|line| line.split_once("PEER-RESULT ").map(|(_, result)| result))
        .unwrap_or_else(|| panic!("the peer process printed no result:\n{stdout}"));
    let (uid, outcome) = line.split_once(' ').unwrap();
    PeerResult {
        uid: uid.strip_prefix("uid=").unwrap().parse().unwrap(),
        outcome: outcome.to_owned(),
    }
}

/// The other account's side: connect to `socket`, wait for the daemon's answer, and print both.
fn connect_as_peer(socket: &Path) {
    // SAFETY: no arguments, and it cannot fail.
    let uid = unsafe { libc::geteuid() };
    let outcome = match UnixStream::connect(socket) {
        Err(error) => format!("connect-error {:?}", error.kind()),
        Ok(mut stream) => {
            stream
                .set_read_timeout(Some(Duration::from_secs(30)))
                .unwrap();
            let mut buf = [0u8; 16];
            match stream.read(&mut buf) {
                Ok(0) => "closed".to_owned(),
                Ok(n) => format!("read {n} bytes"),
                Err(error) => format!("read-error {:?}", error.kind()),
            }
        }
    };
    println!("\nPEER-RESULT uid={uid} {outcome}");
}
