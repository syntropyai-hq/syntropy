// SPDX-License-Identifier: MIT OR Apache-2.0

//! Probe mode (DIST-E1-T17): `daemon`'s process host spawns the `syntropy` binary as
//! `__runner --probe`, which answers `ProbeResult` and exits, with no region and no `Init`
//! (RFC-0001 §3.1, §3.5).

use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use syntropy_core::Entropy;
use syntropy_daemon::runner_host::probe_process;

/// The `syntropy` binary cargo built for this test.
fn binary() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_syntropy"))
}

/// Bytes that differ on every call, so no two pipes share a name.
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

/// The binary's build ID, as it reports it.
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

/// A probe process passes the handshake, returns its profile and exits 0. It names no device
/// until PLC-E1-T07's enumeration runs in it, and leaves `node` and `version` to the daemon.
#[test]
fn a_probe_process_returns_its_profile_and_exits() {
    let profile = probe_process(binary(), build_id(), &Varying).unwrap();
    assert_eq!(profile.version, 0);
    assert_eq!(profile.node.0, [0; 32]);
    assert!(profile.devices.is_empty() && profile.storage.is_empty() && profile.links.is_empty());
}

/// A probe of another build is refused at the handshake, and killed.
#[test]
fn a_probe_of_another_build_is_refused() {
    let err = probe_process(binary(), [0xee; 32], &Varying).unwrap_err();
    assert!(err.to_string().contains("refused"), "{err}");
}

/// Probe mode takes no region: `--probe` with `--region` is a usage error, and runner mode
/// without `--probe` still needs one.
#[test]
fn probe_mode_and_a_region_are_exclusive() {
    let both = Command::new(binary())
        .args(["__runner", "--probe", "--ipc", "fd:3", "--region", "name:x"])
        .output()
        .unwrap();
    assert_eq!(both.status.code(), Some(2), "{both:?}");
    let stderr = String::from_utf8_lossy(&both.stderr);
    assert!(stderr.contains("cannot be used with"), "{stderr}");
    let neither = Command::new(binary())
        .args(["__runner", "--ipc", "fd:3"])
        .output()
        .unwrap();
    assert_eq!(neither.status.code(), Some(2), "{neither:?}");
    let stderr = String::from_utf8_lossy(&neither.stderr);
    assert!(stderr.contains("required"), "{stderr}");
}
