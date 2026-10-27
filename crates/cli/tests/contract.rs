// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E6-T01's contract tests (RFC-0051 §2.2, Testing): every `--json` output deserialises
//! into the native API type it claims, over each way the client reaches a daemon.
//!
//! Commands with `--json` output so far: `version`, which prints `VersionInfo`. Each command its
//! task builds adds its case here.

mod common;

use common::{Daemon, code, finish, syntropy, test_version};
use syntropy_api::version::VersionInfo;
use syntropy_ipc::cmd::Endpoint;

#[test]
fn version_json_is_the_api_s_version_info_over_syntropy_host() {
    let daemon = Daemon::start(None);
    let output = finish(
        syntropy(&["version", "--json"]).env("SYNTROPY_HOST", format!("http://{}", daemon.addr)),
    );
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    // Exactly one JSON document, on one line.
    assert_eq!(text.lines().count(), 1, "{text}");
    let info: VersionInfo = serde_json::from_str(&text).expect("the API's VersionInfo");
    assert_eq!(info, test_version());
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn version_prints_the_daemon_s_build_protocols_and_tier_as_text() {
    let daemon = Daemon::start(None);
    let output =
        finish(syntropy(&["version"]).env("SYNTROPY_HOST", format!("http://{}", daemon.addr)));
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let info = test_version();
    assert!(
        text.contains(&format!(
            "daemon     syntropy 0.0.0-test (build {})",
            "5a".repeat(32)
        )),
        "{text}"
    );
    assert!(
        text.contains(&format!(
            "protocols  wire {}, ipc {}",
            info.protocols.wire, info.protocols.ipc
        )),
        "{text}"
    );
    assert!(text.contains("support    Tier 2"), "{text}");
    assert!(
        text.contains(&format!(
            "reached    http://{} (SYNTROPY_HOST)",
            daemon.addr
        )),
        "{text}"
    );
    // Not a terminal: no colour codes.
    assert!(!text.contains('\u{1b}'), "{text:?}");
}

/// The local socket comes first: the test points the runtime location at its own directory
/// (`XDG_RUNTIME_DIR` on Linux, `TMPDIR` on macOS) and serves the API there. On Windows the
/// socket is this user's named pipe, `syntropy-<SID>-api`.
#[test]
fn version_json_is_the_api_s_version_info_over_the_local_socket() {
    let runtime_dir =
        std::env::temp_dir().join(format!("syntropy-cli-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&runtime_dir);
    std::fs::create_dir_all(&runtime_dir).unwrap();
    let location = if cfg!(windows) {
        syntropy_config::Runtime::NamedPipes
    } else {
        syntropy_config::Runtime::Dir(runtime_dir.join("syntropy"))
    };
    let endpoint: Endpoint = syntropy_api::server::local_endpoint(&location).unwrap();
    let daemon = Daemon::start(Some(&endpoint));
    let mut command = syntropy(&["version", "--json"]);
    command
        .env("XDG_RUNTIME_DIR", &runtime_dir)
        .env("TMPDIR", &runtime_dir);
    let output = finish(&mut command);
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let info: VersionInfo = serde_json::from_slice(&output.stdout).expect("VersionInfo");
    assert_eq!(info, test_version());

    let mut command = syntropy(&["version"]);
    command
        .env("XDG_RUNTIME_DIR", &runtime_dir)
        .env("TMPDIR", &runtime_dir);
    let output = finish(&mut command);
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(
        text.contains(&format!("reached    the local socket ({endpoint})")),
        "{text}"
    );
    drop(daemon);
    let _ = std::fs::remove_dir_all(&runtime_dir);
}
