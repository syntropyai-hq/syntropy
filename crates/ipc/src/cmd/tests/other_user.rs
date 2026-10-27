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
//! `an_async_connection_from_another_user_is_refused` makes the same two connections to the async
//! listener, the API's local socket (PROD-E2-T01; RFC-0050 §1.1).
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

#[cfg(feature = "async")]
use super::super::AsyncListener;
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
    let Some(user) = second_account() else {
        return;
    };
    let (base, exe) = peer_dir("peer");
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

#[cfg(feature = "async")]
#[test]
fn an_async_connection_from_another_user_is_refused() {
    // The copy runs only the test above.
    if env::var_os(CONNECT).is_some() {
        return;
    }
    let Some(user) = second_account() else {
        return;
    };
    let (base, exe) = peer_dir("async-peer");
    // SAFETY: no arguments, and it cannot fail.
    let our_uid = unsafe { libc::geteuid() };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .unwrap();
    let _context = rt.enter();

    // 1. The endpoint as the API binds it: the directory keeps the other user out.
    let private = base.join("private").join("api.sock");
    let listener = AsyncListener::bind(&Endpoint::Socket(private.clone())).unwrap();
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
    let path = exposed.join("api.sock");
    let socket = UnixListener::bind(&path).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o777)).unwrap();
    let mut listener = AsyncListener::from_std(socket, path.clone()).unwrap();
    let (refused, result) = thread::scope(|scope| {
        let peer = scope.spawn(|| run_as(&user, &exe, &path));
        let refused = rt.block_on(async {
            let accepted = tokio::time::timeout(Duration::from_secs(60), listener.accept()).await;
            // Name the arm at once, and drop an accepted stream at once, so a failure says what
            // happened and never holds the peer open.
            let refused = match accepted {
                Ok(Err(AcceptError::Refused(peer))) => Ok(peer),
                Ok(Err(AcceptError::Io(error))) => Err(format!("the accept failed: {error}")),
                Ok(Ok((stream, peer))) => {
                    drop(stream);
                    Err(format!("the peer check ACCEPTED {peer}"))
                }
                Err(_) => Err("no connection completed the accept within 60 s".to_owned()),
            };
            eprintln!("2. the async accept returned: {refused:?}");
            refused
        });
        (refused, peer.join().unwrap())
    });
    match refused {
        Ok(Peer::Unix { uid, .. }) => assert_eq!(uid, result.uid),
        Ok(peer) => panic!("a Windows peer on a Unix socket: {peer}"),
        Err(why) => panic!("{why}; the peer saw: {}", result.outcome),
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

/// The second account's name, or `None` after the notice that the test did not run. In CI the
/// account must exist.
fn second_account() -> Option<std::ffi::OsString> {
    let user = env::var_os(USER).filter(|user| !user.is_empty());
    if user.is_none() {
        let in_ci = env::var_os("CI").is_some_and(|v| !v.is_empty() && v != "false");
        assert!(
            !in_ci,
            "CI must run this test: create a second OS account and name it in {USER}"
        );
        eprintln!(
            "skipped: no second OS account named in {USER}, so the other-user refusal did not run"
        );
    }
    user
}

/// A directory under `/tmp` the other account may enter, named for `tag` and this process, with
/// a copy of this test binary in it that the account may run.
fn peer_dir(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let base = Path::new("/tmp").join(format!("syntropy-ipc-{tag}-{}", std::process::id()));
    fs::create_dir(&base).unwrap();
    fs::set_permissions(&base, fs::Permissions::from_mode(0o755)).unwrap();
    let exe = base.join("peer");
    fs::copy(env::current_exe().unwrap(), &exe).unwrap();
    fs::set_permissions(&exe, fs::Permissions::from_mode(0o755)).unwrap();
    (base, exe)
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
        Ok(mut stream) => match stream.set_read_timeout(Some(Duration::from_secs(30))) {
            // macOS refuses `SO_RCVTIMEO` with EINVAL on a socket whose peer has already closed
            // it: the daemon refused this connection before the timeout could be set.
            Err(error) if error.kind() == ErrorKind::InvalidInput => "closed".to_owned(),
            Err(error) => panic!("setting the read timeout failed: {error}"),
            Ok(()) => {
                let mut buf = [0u8; 16];
                match stream.read(&mut buf) {
                    Ok(0) => "closed".to_owned(),
                    Ok(n) => format!("read {n} bytes"),
                    Err(error) => format!("read-error {:?}", error.kind()),
                }
            }
        },
    };
    println!("\nPEER-RESULT uid={uid} {outcome}");
}
