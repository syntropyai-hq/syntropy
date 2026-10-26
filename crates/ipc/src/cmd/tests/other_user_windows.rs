// SPDX-License-Identifier: MIT OR Apache-2.0

//! A connection from another OS user is refused, on Windows (DIST-E1-T03; RFC-0034 §6.4).
//!
//! The Windows form of `other_user.rs`. The test runs a copy of this test binary as a second
//! account, through `CreateProcessWithLogonW`, and has it connect twice:
//!
//! 1. to a pipe from [`Listener::bind`], whose protected DACL grants this user alone. The OS
//!    refuses the connection before the daemon sees it.
//! 2. to a pipe whose DACL also lets every authenticated user open it, so the connection reaches
//!    the peer check, which refuses it and names the client's process.
//!
//! CI's hosted Windows job creates the account and names it in `SYNTROPY_IPC_PEER_USER`, with its
//! password in `SYNTROPY_IPC_PEER_PASSWORD`. Without them the test prints a notice and passes,
//! except in CI, where it fails. The copy cannot print to this process, so it writes what it saw
//! to a file. The copy and that file sit in a directory under `%PUBLIC%` that the test grants the
//! second account, not under the temporary directory, which is this user's alone.

use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::Read;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use windows_sys::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows_sys::Win32::System::Threading::{
    CREATE_NO_WINDOW, CREATE_UNICODE_ENVIRONMENT, CreateProcessWithLogonW, GetExitCodeProcess,
    LOGON_WITH_PROFILE, PROCESS_INFORMATION, STARTUPINFOW, WaitForSingleObject,
};

use super::super::{AcceptError, Endpoint, Listener, Peer, connect};
use crate::shm::UserSid;

/// The second account's name.
const USER: &str = "SYNTROPY_IPC_PEER_USER";
/// The second account's password.
const PASSWORD: &str = "SYNTROPY_IPC_PEER_PASSWORD";
/// Set only in the copy that runs as the second account: the pipe name to connect to.
const CONNECT: &str = "SYNTROPY_IPC_PEER_CONNECT";
/// Set only in the copy: the file it writes its result to.
const OUT: &str = "SYNTROPY_IPC_PEER_OUT";
/// This test, which the copy runs.
const TEST: &str = "cmd::tests::other_user::a_connection_from_another_user_is_refused";
/// How long the copy may take, in milliseconds. A first logon creates the account's profile.
const PEER_TIMEOUT_MS: u32 = 180_000;

#[test]
fn a_connection_from_another_user_is_refused() {
    if let Some(pipe) = env::var_os(CONNECT) {
        let out = env::var_os(OUT).expect("the parent names the result file");
        connect_as_peer(&pipe.to_string_lossy(), Path::new(&out));
        return;
    }
    let account = env::var_os(USER)
        .filter(|user| !user.is_empty())
        .zip(env::var_os(PASSWORD).filter(|password| !password.is_empty()));
    let Some((user, password)) = account else {
        let in_ci = env::var_os("CI").is_some_and(|v| !v.is_empty() && v != "false");
        assert!(
            !in_ci,
            "CI must run this test: create a second OS account and name it in {USER}, with its \
             password in {PASSWORD}"
        );
        eprintln!(
            "skipped: no second OS account named in {USER} and {PASSWORD}, so the other-user \
             refusal did not run"
        );
        return;
    };
    let our_sid = UserSid::current().unwrap().text().unwrap();

    // Everything the other account touches sits in a directory it is granted.
    let public =
        env::var_os("PUBLIC").map_or_else(|| PathBuf::from(r"C:\Users\Public"), PathBuf::from);
    let base = public.join(format!("syntropy-ipc-peer-{}", std::process::id()));
    fs::create_dir(&base).unwrap();
    let mut grant = OsString::from(&user);
    grant.push(":(OI)(CI)M");
    let icacls = Command::new("icacls")
        .arg(&base)
        .arg("/grant")
        .arg(&grant)
        .output()
        .expect("run icacls");
    assert!(
        icacls.status.success(),
        "icacls could not grant {} the test directory: {}",
        user.to_string_lossy(),
        String::from_utf8_lossy(&icacls.stdout)
    );
    let exe = base.join("peer.exe");
    fs::copy(env::current_exe().unwrap(), &exe).unwrap();
    let account = Account {
        user: &user,
        password: &password,
        exe: &exe,
        base: &base,
    };

    // 1. The pipe as the daemon creates it: its DACL keeps the other user out.
    let private = format!("syntropy-ipc-peer-{}-private", std::process::id());
    let listener = Listener::bind(&Endpoint::Pipe(private.clone())).unwrap();
    let result = account.run(&private, "private");
    assert_ne!(result.sid, our_sid, "{USER} names this test's own user");
    assert!(
        result.outcome == "connect-error PermissionDenied",
        "the other user reached a pipe whose DACL grants only {our_sid}: {}",
        result.outcome
    );
    eprintln!(
        "1. the owner-only pipe refused user {} (process {}): {}",
        result.sid, result.pid, result.outcome
    );
    drop(listener);

    // 2. A pipe any authenticated user may open: the peer check refuses the connection.
    let exposed = format!("syntropy-ipc-peer-{}-exposed", std::process::id());
    let endpoint = Endpoint::Pipe(exposed.clone());
    let mut listener = Listener::open_to_authenticated_users(&endpoint).unwrap();
    let (sent, accepted) = mpsc::channel();
    let acceptor = thread::spawn(move || {
        let _ = sent.send(listener.accept());
        listener
    });
    let result = account.run(&exposed, "exposed");
    let accepted = match accepted.recv_timeout(Duration::from_secs(30)) {
        Ok(accepted) => accepted,
        Err(_) => {
            // Unblock the acceptor with a connection of this user before failing.
            let _ = connect(&endpoint);
            panic!(
                "the other user never reached the peer check: {}",
                result.outcome
            );
        }
    };
    drop(acceptor.join().unwrap());
    match accepted {
        Err(AcceptError::Refused(Peer::Windows { pid, sid })) => {
            assert_eq!(pid, result.pid, "the refusal names another process");
            if let Some(sid) = &sid {
                assert_eq!(*sid, result.sid, "the refusal names another user");
            }
            let refusal = AcceptError::Refused(Peer::Windows { pid, sid });
            eprintln!("2. the peer check: {refusal}");
        }
        Err(error) => panic!("the connection failed in another way: {error}"),
        Ok(_) => panic!("the peer check accepted user {}", result.sid),
    }
    assert!(
        matches!(result.outcome.as_str(), "closed" | "read-error BrokenPipe"),
        "the refused peer saw: {}",
        result.outcome
    );
    eprintln!(
        "2. the refused peer, user {} (process {}), saw: {}",
        result.sid, result.pid, result.outcome
    );
    if let Err(error) = fs::remove_dir_all(&base) {
        eprintln!("could not remove {}: {error}", base.display());
    }
}

/// The second account, and where its copy of this test sits.
struct Account<'a> {
    user: &'a OsStr,
    password: &'a OsStr,
    exe: &'a Path,
    base: &'a Path,
}

/// What the copy running as the other account saw.
struct PeerResult {
    sid: String,
    pid: u32,
    outcome: String,
}

/// `text` as NUL-terminated UTF-16.
fn wide(text: &OsStr) -> Vec<u16> {
    text.encode_wide().chain(std::iter::once(0)).collect()
}

impl Account<'_> {
    /// Runs the copy as the account, connecting to `pipe`, and reads what it saw. `tag` names its
    /// result file.
    fn run(&self, pipe: &str, tag: &str) -> PeerResult {
        let out = self.base.join(format!("{tag}.txt"));
        // This process's environment, without the account's credentials, plus the copy's two
        // variables, sorted by name as `CreateProcessWithLogonW` expects.
        let mut vars: Vec<(OsString, OsString)> = env::vars_os()
            .filter(|(name, _)| {
                ![USER, PASSWORD, CONNECT, OUT]
                    .iter()
                    .any(|own| name.eq_ignore_ascii_case(own))
            })
            .collect();
        vars.push((CONNECT.into(), pipe.into()));
        vars.push((OUT.into(), out.clone().into()));
        vars.sort_by_key(|(name, _)| name.to_ascii_uppercase());
        let mut environment: Vec<u16> = Vec::new();
        for (name, value) in &vars {
            let mut entry = name.clone();
            entry.push("=");
            entry.push(value);
            environment.extend(wide(&entry));
        }
        environment.push(0);

        let mut command_line = OsString::from("\"");
        command_line.push(self.exe);
        command_line.push(format!("\" --exact {TEST} --nocapture --test-threads=1"));
        let mut command_line = wide(&command_line);
        let user = wide(self.user);
        let domain = wide(OsStr::new("."));
        let password = wide(self.password);
        let application = wide(self.exe.as_os_str());
        let directory = wide(self.base.as_os_str());
        let startup = STARTUPINFOW {
            cb: std::mem::size_of::<STARTUPINFOW>() as u32,
            ..Default::default()
        };
        let mut process = PROCESS_INFORMATION::default();
        // SAFETY: every string is NUL-terminated and outlives the call; the command line is a
        // writable buffer, as the call requires; the environment is a block of NUL-terminated
        // UTF-16 entries ending in an empty one, matching `CREATE_UNICODE_ENVIRONMENT`; the
        // startup information is zeroed apart from its size; the output pointer is live.
        let created = unsafe {
            CreateProcessWithLogonW(
                user.as_ptr(),
                domain.as_ptr(),
                password.as_ptr(),
                LOGON_WITH_PROFILE,
                application.as_ptr(),
                command_line.as_mut_ptr(),
                CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
                environment.as_ptr().cast(),
                directory.as_ptr(),
                &startup,
                &mut process,
            )
        };
        assert!(
            created != 0,
            "CreateProcessWithLogonW failed: {}",
            std::io::Error::last_os_error()
        );
        // SAFETY: the two handles the call just returned, each owned only here.
        let (handle, _thread) = unsafe {
            (
                OwnedHandle::from_raw_handle(process.hProcess.cast()),
                OwnedHandle::from_raw_handle(process.hThread.cast()),
            )
        };
        // SAFETY: a live process handle the call returned with full access.
        let waited = unsafe { WaitForSingleObject(handle.as_raw_handle().cast(), PEER_TIMEOUT_MS) };
        assert!(
            waited != WAIT_TIMEOUT,
            "the peer process {} ran past {PEER_TIMEOUT_MS} ms",
            process.dwProcessId
        );
        assert_eq!(waited, WAIT_OBJECT_0, "waiting for the peer process failed");
        let mut code = 0u32;
        // SAFETY: a live process handle and an output pointer.
        let got = unsafe { GetExitCodeProcess(handle.as_raw_handle().cast(), &mut code) };
        assert!(got != 0, "{}", std::io::Error::last_os_error());
        let text = fs::read_to_string(&out).unwrap_or_default();
        assert!(
            code == 0,
            "the peer process exited with {code:#x}; it wrote: {text:?}"
        );
        let line = text
            .lines()
            .find_map(|line| line.strip_prefix("PEER-RESULT "))
            .unwrap_or_else(|| panic!("the peer process wrote no result: {text:?}"));
        let (sid, rest) = line.split_once(' ').unwrap();
        let (pid, outcome) = rest.split_once(' ').unwrap();
        let result = PeerResult {
            sid: sid.strip_prefix("sid=").unwrap().to_owned(),
            pid: pid.strip_prefix("pid=").unwrap().parse().unwrap(),
            outcome: outcome.to_owned(),
        };
        assert_eq!(
            result.pid, process.dwProcessId,
            "the result came from another process"
        );
        result
    }
}

/// The other account's side: connect to `pipe`, wait for the daemon's answer, and write both,
/// with this process's user and ID, to `out`.
fn connect_as_peer(pipe: &str, out: &Path) {
    let sid = UserSid::current()
        .and_then(|sid| sid.text())
        .unwrap_or_else(|error| format!("unknown({error})"));
    let outcome = match connect(&Endpoint::Pipe(pipe.to_owned())) {
        Err(error) => format!("connect-error {:?}", error.kind()),
        Ok(mut stream) => {
            let mut buf = [0u8; 16];
            match stream.read(&mut buf) {
                Ok(0) => "closed".to_owned(),
                Ok(n) => format!("read {n} bytes"),
                Err(error) => format!("read-error {:?}", error.kind()),
            }
        }
    };
    let line = format!(
        "PEER-RESULT sid={sid} pid={} {outcome}\n",
        std::process::id()
    );
    fs::write(out, &line).unwrap();
    print!("{line}");
}
