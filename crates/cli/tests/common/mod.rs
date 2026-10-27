// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the CLI's integration tests share: running the built binary, a daemon served in-process,
//! and a fake daemon that answers with scripted responses.

#![allow(dead_code, reason = "each test binary uses a different part")]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::process::{Command, Output, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use syntropy_api::config::{ApiSection, TlsSource};
use syntropy_api::routes::{ApiState, NodeStatus};
use syntropy_api::server::Server;
use syntropy_api::version::{Build, ProtocolMajors, SupportTier, VersionInfo};
use syntropy_ipc::cmd::Endpoint;

/// The `syntropy` binary under test.
pub const BIN: &str = env!("CARGO_BIN_EXE_syntropy");

/// The longest any one command may run before the test calls it hung.
pub const HANG: Duration = Duration::from_secs(20);

/// A command for the binary with `args`, stdin closed, and no variable that could reach a real
/// daemon on this machine: `SYNTROPY_HOST` and `SYNTROPY_API_KEY` removed.
pub fn syntropy(args: &[&str]) -> Command {
    let mut command = Command::new(BIN);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_remove("SYNTROPY_HOST")
        .env_remove("SYNTROPY_API_KEY")
        .env_remove("NO_COLOR");
    command
}

/// Runs `command` to completion, failing the test if it runs past [`HANG`].
pub fn finish(command: &mut Command) -> Output {
    let mut child = command.spawn().expect("spawn syntropy");
    let start = Instant::now();
    loop {
        if child.try_wait().expect("wait").is_some() {
            return child.wait_with_output().expect("read the output");
        }
        if start.elapsed() > HANG {
            let _ = child.kill();
            let output = child.wait_with_output().expect("read the output");
            panic!(
                "{command:?} ran past {HANG:?}; stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The exit code of `output`.
pub fn code(output: &Output) -> i32 {
    output.status.code().expect("an exit code, not a signal")
}

/// A loopback address nothing listens on.
pub fn closed_addr() -> SocketAddr {
    let socket = TcpListener::bind("127.0.0.1:0").unwrap();
    socket.local_addr().unwrap()
}

/// The version a test daemon reports.
pub fn test_version() -> VersionInfo {
    VersionInfo {
        build: Build {
            version: "0.0.0-test".into(),
            id: [0x5a; 32],
        },
        protocols: ProtocolMajors::current(),
        support_tier: SupportTier::Tier2,
    }
}

struct Solo;

impl NodeStatus for Solo {
    fn has_state_replica(&self) -> bool {
        true
    }
}

/// A daemon's API, served in-process on loopback and, when `local` is given, its local socket.
pub struct Daemon {
    /// The loopback address.
    pub addr: SocketAddr,
    runtime: Option<tokio::runtime::Runtime>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Daemon {
    /// Serves the API.
    pub fn start(local: Option<&Endpoint>) -> Daemon {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let api = ApiSection {
            bind: vec!["127.0.0.1:0".parse().unwrap()],
            tls: TlsSource::Auto,
            allow_plaintext_lan: false,
            local_socket: local.is_some(),
        };
        let state = ApiState {
            version: test_version(),
            node: Arc::new(Solo),
        };
        let server = {
            let _context = runtime.enter();
            Server::bind(&api, local, false, state).expect("serve the API")
        };
        let addr = server.local_addrs()[0];
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        runtime.spawn(server.serve(async {
            let _ = stopped.await;
        }));
        Daemon {
            addr,
            runtime: Some(runtime),
            stop: Some(stop),
        }
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_timeout(Duration::from_secs(5));
        }
    }
}

/// A fake daemon on loopback that answers every request with `status` and `body`, and records
/// each request's head.
pub struct Fake {
    /// Its address.
    pub addr: SocketAddr,
    /// The request heads it received.
    pub heads: std::sync::mpsc::Receiver<String>,
}

impl Fake {
    /// Answers `answers` connections, one HTTP/1.1 response each, then stops.
    pub fn start(status: u16, body: String, answers: usize) -> Fake {
        let socket = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = socket.local_addr().unwrap();
        let (sent, heads) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for _ in 0..answers {
                let Ok((mut stream, _)) = socket.accept() else {
                    return;
                };
                let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    match stream.read(&mut byte) {
                        Ok(1) => head.push(byte[0]),
                        _ => break,
                    }
                }
                // A probe that only connects sends nothing.
                if head.is_empty() {
                    continue;
                }
                let _ = sent.send(String::from_utf8_lossy(&head).into_owned());
                let response = format!(
                    "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: \
                     {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        Fake { addr, heads }
    }
}
