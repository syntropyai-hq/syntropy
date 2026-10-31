// SPDX-License-Identifier: MIT OR Apache-2.0

//! QUAL-E16-T03's tests: the `BuildId` this binary carries (RFC-0062 §8.1; ADR-0621 decision 1).
//!
//! - `syntropy --version --json` prints every `BuildId` field, with the features and the build ID.
//! - Every crate in `cli`'s dependency graph that declares a test-only feature reports it in the
//!   build's feature list.
//! - The daemon's handshake with a freshly spawned runner refuses and kills a runner whose
//!   `Hello` carries another build's ID, and accepts this build's (RFC-0001 §3.1; ADR-0015
//!   decision 1). The runner's side of the channel is DIST-E1-T04 and DIST-E1-T05; until it
//!   exists this test binary plays the runner, with the real build IDs.

#![cfg(any(target_os = "linux", target_vendor = "apple", windows))]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::{Child, Command};
use std::thread;
use std::time::Duration;

use common::{code, finish, syntropy};
use syntropy_api::version::Build;
use syntropy_cli::build_id;
use syntropy_core::{Blake3, InstanceId};
use syntropy_ipc::cmd::{
    DaemonChannel, DaemonMsg, Endpoint, HandshakeError, Hello, Init, Listener, PROTOCOL,
    RunnerChannel, connect,
};

#[test]
fn version_json_prints_every_field() {
    let output = finish(&mut syntropy(&["--version", "--json"]));
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let keys: BTreeSet<&str> = body
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        BTreeSet::from([
            "version",
            "commit",
            "profile",
            "target",
            "toolchain",
            "features",
            "id"
        ])
    );
    let build: Build = serde_json::from_value(body).expect("the API's Build");
    // The binary's build is the library's, and its ID is the digest of its fields.
    assert_eq!(build.build_id, *build_id::current());
    assert_eq!(build.features, build_id::features());
    assert_eq!(build.id, build_id::digest(&build.build_id, &build.features));
    // The version is `version.txt`'s.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let version = std::fs::read_to_string(root.join("version.txt")).unwrap();
    assert_eq!(build.build_id.version.to_string(), version.trim());
    // In a git checkout with no `SYNTROPY_BUILD_COMMIT`, the commit is HEAD's.
    if std::env::var_os("SYNTROPY_BUILD_COMMIT").is_none()
        && let Ok(head) = Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(["rev-parse", "HEAD"])
            .output()
        && head.status.success()
    {
        let head = String::from_utf8(head.stdout).unwrap();
        let commit: String = build
            .build_id
            .commit
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(commit, head.trim());
    }
    // `-V` is the same flag, and the text names each field.
    let text = finish(&mut syntropy(&["-V"]));
    assert_eq!(code(&text), 0);
    let text = String::from_utf8(text.stdout).unwrap();
    for line in [
        format!("syntropy {}", build.build_id.version),
        format!("profile    {}", build.build_id.profile),
        format!("target     {}", build.build_id.target),
        "features   none".to_owned(),
        format!("ggml       {}", syntropy_kernels_ggml::COMMIT),
        format!(
            "build      {}",
            build
                .id
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        ),
    ] {
        assert!(text.lines().any(|l| l == line), "{line:?} not in {text}");
    }
}

/// RFC-0001 §4.2 rule 10's test-only features, as `xtask`'s `policy::TEST_FEATURES` lists them.
const TEST_FEATURES: &[&str] = &["fault-injection", "debug-taps", "guards"];

/// A test-only feature further down reaches this binary through cargo's feature unification
/// without `cli`'s build script seeing it, so each crate that declares one must report it in
/// [`build_id::features`]. This reads `cargo metadata` for `cli`'s normal and build
/// dependencies on every platform.
#[test]
fn a_crate_with_a_test_feature_reports_it() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.toml");
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        // Not `--offline`: without `--filter-platform`, cargo resolves every platform's
        // dependencies, and a runner's cache holds only its own (as `check-deps` reads it).
        .args(["metadata", "--format-version", "1", "--locked"])
        .arg("--manifest-path")
        .arg(&manifest)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let members: BTreeSet<&str> = metadata["workspace_members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_str().unwrap())
        .collect();
    let packages: BTreeMap<&str, &serde_json::Value> = metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p["id"].as_str().unwrap(), p))
        .collect();
    let nodes: BTreeMap<&str, &serde_json::Value> = metadata["resolve"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| (n["id"].as_str().unwrap(), n))
        .collect();
    let cli = packages
        .values()
        .find(|p| p["name"] == "syntropy-cli")
        .unwrap()["id"]
        .as_str()
        .unwrap();
    let mut seen = BTreeSet::from([cli]);
    let mut queue = vec![cli];
    while let Some(id) = queue.pop() {
        for dep in nodes[id]["deps"].as_array().unwrap() {
            let shipped = dep["dep_kinds"]
                .as_array()
                .unwrap()
                .iter()
                .any(|kind| kind["kind"].is_null() || kind["kind"] == "build");
            let pkg = dep["pkg"].as_str().unwrap();
            if shipped && seen.insert(pkg) {
                queue.push(pkg);
            }
        }
    }
    let declaring: BTreeSet<String> = seen
        .iter()
        .filter(|id| members.contains(**id))
        .filter(|id| {
            packages[**id]["features"]
                .as_object()
                .unwrap()
                .keys()
                .any(|f| TEST_FEATURES.contains(&f.as_str()))
        })
        .map(|id| {
            let name = packages[*id]["name"].as_str().unwrap();
            name.strip_prefix("syntropy-").unwrap_or(name).to_owned()
        })
        .collect();
    let reporting: BTreeSet<String> = build_id::REPORTING_CRATES
        .iter()
        .map(|c| (*c).to_owned())
        .collect();
    assert_eq!(
        declaring, reporting,
        "the crates in cli's graph that declare a test-only feature must report it in \
         `build_id::features` and be listed in `build_id::REPORTING_CRATES`"
    );
}

/// The endpoint the runner connects to, set only in the child.
const ENDPOINT: &str = "SYNTROPY_CLI_TEST_RUNNER_ENDPOINT";
/// Which build the child runner claims: `this` or `other`, set only in the child.
const CLAIM: &str = "SYNTROPY_CLI_TEST_RUNNER_BUILD";
/// How long a child runner lives if nobody ends it, so a failed kill fails the test rather than
/// hanging it.
const RUNNER_LIFETIME: Duration = Duration::from_secs(60);

/// The build ID of a runner built from another commit, with every other field this build's.
fn other_commit_id() -> [u8; 32] {
    let mut other = build_id::current().clone();
    other.commit[19] ^= 0xff;
    build_id::digest(&other, &build_id::features())
}

/// The runner's side, in the child process.
fn run_runner(endpoint: &str, claim: &str) {
    thread::spawn(|| {
        thread::sleep(RUNNER_LIFETIME);
        std::process::exit(3);
    });
    let build = match claim {
        "this" => build_id::id(),
        _ => other_commit_id(),
    };
    let endpoint: Endpoint = endpoint.parse().unwrap();
    let mut channel = RunnerChannel::new(connect(&endpoint).unwrap());
    let hello = Hello {
        pid: std::process::id(),
        build_id: build,
        ipc_protocol: PROTOCOL,
        obs_tables: Blake3([0x0b; 32]),
    };
    // A runner of another build waits here until the daemon kills it.
    let init = channel.handshake(&hello).unwrap();
    assert_eq!(init.instance, InstanceId(7));
    // The daemon closes the channel when it is done.
    assert!(channel.recv().is_err());
}

/// A fresh endpoint for one test.
fn endpoint(test: &str) -> Endpoint {
    let name = format!("syntropy-cli-{test}-{}", std::process::id());
    if cfg!(windows) {
        Endpoint::Pipe(name)
    } else {
        Endpoint::Socket(std::env::temp_dir().join(name).join("cmd.sock"))
    }
}

/// Binds `endpoint`, spawns this binary as a runner claiming `claim` running `test`, and
/// accepts its connection.
fn spawn_runner(test: &str, endpoint: &Endpoint, claim: &str) -> (Listener, DaemonChannel, Child) {
    let mut listener = Listener::bind(endpoint).unwrap();
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--test-threads", "1"])
        .env(ENDPOINT, endpoint.to_string())
        .env(CLAIM, claim)
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
fn a_fresh_runner_of_this_build_passes_the_handshake() {
    const TEST: &str = "a_fresh_runner_of_this_build_passes_the_handshake";
    if let (Ok(endpoint), Ok(claim)) = (std::env::var(ENDPOINT), std::env::var(CLAIM)) {
        run_runner(&endpoint, &claim);
        return;
    }
    let endpoint = endpoint("this-build");
    let (listener, mut channel, mut child) = spawn_runner(TEST, &endpoint, "this");
    let hello = channel.handshake(&mut child, &build_id::id()).unwrap();
    assert_eq!(hello.build_id, build_id::id());
    channel
        .send(&DaemonMsg::Init(Init {
            build_id: build_id::id(),
            instance: InstanceId(7),
            ledger: Default::default(),
            shm: Default::default(),
        }))
        .unwrap();
    drop(channel);
    let status = child.wait().unwrap();
    assert!(status.success(), "the runner failed: {status}");
    cleanup(listener);
}

#[test]
fn a_fresh_runner_of_another_commit_is_refused_and_killed() {
    const TEST: &str = "a_fresh_runner_of_another_commit_is_refused_and_killed";
    if let (Ok(endpoint), Ok(claim)) = (std::env::var(ENDPOINT), std::env::var(CLAIM)) {
        run_runner(&endpoint, &claim);
        panic!("the runner of another build outlived its handshake");
    }
    let ours = build_id::id();
    let theirs = other_commit_id();
    assert_ne!(ours, theirs);
    let endpoint = endpoint("other-commit");
    let (listener, mut channel, mut child) = spawn_runner(TEST, &endpoint, "other");
    let refused = channel.handshake(&mut child, &ours).unwrap_err();
    assert!(
        matches!(
            refused.cause,
            HandshakeError::Build { theirs: t, ours: o } if t == theirs && o == ours
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
        assert_eq!(status.signal(), Some(9), "{status}");
    }
    #[cfg(windows)]
    assert_eq!(status.code(), Some(1), "{status}");
    cleanup(listener);
}
