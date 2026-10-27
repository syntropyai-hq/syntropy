// SPDX-License-Identifier: MIT OR Apache-2.0

//! The API on the daemon's `ctl` runtime (PROD-E2-T01; RFC-0001 §8.1, RFC-0050 §1.1): started
//! with `start_api`, it answers `/readyz` over loopback TCP and over the local socket in the
//! runtime location, with no model loaded, and stops cleanly.

use std::io::{Read, Write};
use std::sync::Arc;

use syntropy_api::config::{ApiSection, TlsSource};
use syntropy_api::routes::{ApiState, NodeStatus};
use syntropy_api::version::{Build, ProtocolMajors, SupportTier, VersionInfo};
use syntropy_config::Runtime as RuntimeLocation;
use syntropy_daemon::ctl::{runtime, start_api};

/// A solo node: it needs no replica.
struct Solo;

impl NodeStatus for Solo {
    fn has_state_replica(&self) -> bool {
        true
    }
}

/// `GET path` over `stream` with HTTP/1.1, and the whole response as text.
fn get(mut stream: impl Read + Write, path: &str) -> String {
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    String::from_utf8(response).unwrap()
}

#[test]
fn the_api_serves_on_the_ctl_runtime_over_tcp_and_the_local_socket() {
    let ctl = runtime(2).unwrap();
    let location = if cfg!(windows) {
        RuntimeLocation::NamedPipes
    } else {
        let dir = std::env::temp_dir().join(format!("syntropy-daemon-ctl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        RuntimeLocation::Dir(dir)
    };
    let api = ApiSection {
        bind: vec!["127.0.0.1:0".parse().unwrap()],
        tls: TlsSource::Auto,
        allow_plaintext_lan: false,
        local_socket: true,
    };
    let state = ApiState {
        version: VersionInfo {
            build: Build {
                version: "0.0.0-test".into(),
                id: [0; 32],
            },
            protocols: ProtocolMajors::current(),
            support_tier: SupportTier::Tier1,
        },
        node: Arc::new(Solo),
    };
    let served = start_api(&ctl, &api, Some(&location), false, state).unwrap();
    let addr = served.addrs()[0];
    let endpoint = served
        .local_socket()
        .expect("the local socket is on")
        .clone();

    let tcp = get(std::net::TcpStream::connect(addr).unwrap(), "/readyz");
    assert!(tcp.starts_with("HTTP/1.1 200 OK\r\n"), "{tcp}");
    assert!(tcp.ends_with(r#"{"ready":true}"#), "{tcp}");
    let local = get(syntropy_ipc::cmd::connect(&endpoint).unwrap(), "/readyz");
    assert!(local.starts_with("HTTP/1.1 200 OK\r\n"), "{local}");
    assert!(local.ends_with(r#"{"ready":true}"#), "{local}");

    ctl.block_on(served.stop());
    assert!(
        std::net::TcpStream::connect(addr).is_err(),
        "still listening"
    );
    if let RuntimeLocation::Dir(dir) = location {
        assert!(
            !dir.join("api.sock").exists(),
            "the socket outlived the API"
        );
        let _ = std::fs::remove_dir(dir);
    }
}

#[test]
fn without_a_runtime_location_the_api_serves_tcp_only() {
    let ctl = runtime(2).unwrap();
    let api = ApiSection {
        bind: vec!["127.0.0.1:0".parse().unwrap()],
        tls: TlsSource::Auto,
        allow_plaintext_lan: false,
        local_socket: true,
    };
    let state = ApiState {
        version: VersionInfo {
            build: Build {
                version: "0.0.0-test".into(),
                id: [0; 32],
            },
            protocols: ProtocolMajors::current(),
            support_tier: SupportTier::Tier1,
        },
        node: Arc::new(Solo),
    };
    let served = start_api(&ctl, &api, None, false, state).unwrap();
    assert!(served.local_socket().is_none());
    let tcp = get(
        std::net::TcpStream::connect(served.addrs()[0]).unwrap(),
        "/healthz",
    );
    assert!(tcp.starts_with("HTTP/1.1 200 OK\r\n"), "{tcp}");
    ctl.block_on(served.stop());
}
