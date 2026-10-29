// SPDX-License-Identifier: MIT OR Apache-2.0

//! The build-ID check through the real runner (QUAL-E16-T29; RFC-0001 §3.1; ADR-0015 decision
//! 1): a daemon that spawns `syntropy __runner` from a binary built with another
//! `SYNTROPY_BUILD_COMMIT` refuses and kills it at `Hello`, and the same binary as the daemon's
//! passes.
//!
//! The test is ignored: it needs a second build of `syntropy-cli`, which
//! `cargo xtask test runner-build-id` makes and names in `SYNTROPY_OTHER_BUILD_RUNNER`. This test
//! binary plays the daemon, through `daemon`'s real spawn path, with this build's ID; the suite
//! removes `SYNTROPY_BUILD_COMMIT` when it builds this test, so `CARGO_BIN_EXE_syntropy` is the
//! checkout's own commit.

#![cfg(any(target_os = "linux", target_vendor = "apple", windows))]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use syntropy_cli::build_id;
use syntropy_core::Entropy;
use syntropy_daemon::runner_host::spawn_process;
use syntropy_ipc::cmd::HandshakeError;
use syntropy_ipc::shm::ShmLayout;

/// This build's `syntropy` binary.
fn this_build() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_syntropy"))
}

/// Bytes that differ on every call, so no two regions or pipes share a name.
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

#[test]
#[ignore = "needs a second build: run `cargo xtask test runner-build-id`"]
fn a_runner_of_another_build_is_refused_and_this_build_passes() {
    let other = PathBuf::from(
        std::env::var_os("SYNTROPY_OTHER_BUILD_RUNNER")
            .expect("cargo xtask test runner-build-id names the other build"),
    );
    let ours = build_id::id();
    let layout = ShmLayout::default();

    // A runner built from another commit: its `Hello` carries its own build ID, which differs.
    let mut runner = spawn_process(&other, &layout, ours, &Varying).unwrap();
    let pid = runner.id();
    let refused = runner.handshake().unwrap_err();
    assert!(
        matches!(refused.cause, HandshakeError::Build { ours: o, .. } if o == ours),
        "{refused:?}"
    );
    let HandshakeError::Build { theirs, .. } = refused.cause else {
        unreachable!()
    };
    assert_ne!(theirs, ours, "the other build reports another ID");
    // The handshake killed and reaped it, and removed its region's name.
    let status = refused.exit.expect("the refused runner was reaped");
    assert!(!status.success(), "runner {pid}: {status}");
    // Killed, not exited: SIGKILL on Unix. On Windows `TerminateProcess` exits with code 1, which
    // the runner's own failure exit also uses, so there only the reap is shown.
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(status.signal(), Some(9), "runner {pid}: {status}");
    }
    assert!(runner.region_unlinked());

    // The daemon's own build passes, and is shut down.
    let mut runner = spawn_process(this_build(), &layout, ours, &Varying).unwrap();
    let hello = runner.handshake().unwrap();
    assert_eq!(hello.build_id, ours);
    assert_eq!(hello.pid, runner.id());
    runner.kill().unwrap();
    let status = runner.wait().unwrap();
    assert!(!status.success());
}
