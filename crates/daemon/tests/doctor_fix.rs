// SPDX-License-Identifier: MIT OR Apache-2.0

//! `doctor --fix` and `uninstall` (RFC-0053 §5.1, RFC-0052 §11.1): a fix applied through the
//! framework is reverted by the generated revert script, run for real with nothing but the OS's
//! tools, as after a user deletes the binary. The unit tests cover the in-process revert walk.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use syntropy_daemon::doctor::fix::AssumeYes;
use syntropy_daemon::doctor::{
    Area, AutoFix, DoctorCheck, DoctorCtx, EXIT_OK, Finding, Fix, FixSetup, HostEdit, Needs,
    Options, Reach, Registry, doctor,
};
use syntropy_daemon::host_changes::script::write_revert_script;
use syntropy_daemon::host_changes::{ChangeState, HostChangeLedger};

/// A new directory, private to this user (0700), as a real state directory is.
fn temp_dir(name: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "syntropy-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    // A directory left by an earlier run with the same process ID.
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
    }
    dir
}

/// Runs the script with nothing but the OS's tools, and returns its output.
fn run(script: &Path) -> String {
    #[cfg(windows)]
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
        ])
        .arg(script)
        .output()
        .expect("powershell runs");
    #[cfg(not(windows))]
    let output = Command::new("sh")
        .arg(script)
        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        // A UTF-8 locale, in which a start-up file's Latin-1 bytes are not valid characters: the
        // script must still rewrite it byte for byte.
        .env("LC_ALL", "C.UTF-8")
        .output()
        .expect("sh runs");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "the script failed: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    assert!(
        stderr.trim().is_empty(),
        "the script wrote errors:\n{stderr}\nstdout:\n{stdout}"
    );
    stdout
}

/// The host directory a check looks after: `host` beside the state directory.
fn host_dir(ctx: &DoctorCtx) -> PathBuf {
    ctx.state_dir.parent().unwrap().join("host")
}

/// Fails until `limits.conf` holds `fixed`, and offers an automatic fix that writes it.
fn limits(ctx: &DoctorCtx) -> Finding {
    let path = host_dir(ctx).join("limits.conf");
    if fs::read(&path).is_ok_and(|b| b == b"fixed\n") {
        return Finding::pass("limits.conf is set");
    }
    Finding::fail("limits.conf is not set").with_fix(Fix {
        automatic: Some(AutoFix {
            edits: vec![HostEdit::WriteFile {
                path,
                contents: b"fixed\n".to_vec(),
            }],
        }),
        ..Fix::manual("write limits.conf")
    })
}

/// Fails until `dropin.conf` exists, and offers an automatic fix that creates it.
fn dropin(ctx: &DoctorCtx) -> Finding {
    let path = host_dir(ctx).join("dropin.conf");
    if path.exists() {
        return Finding::pass("drop-in present");
    }
    Finding::fail("drop-in missing").with_fix(Fix {
        automatic: Some(AutoFix {
            edits: vec![HostEdit::WriteFile {
                path,
                contents: b"[Service]\n".to_vec(),
            }],
        }),
        ..Fix::manual("write the drop-in")
    })
}

#[test]
fn fixes_applied_by_doctor_are_reverted_by_the_generated_script() {
    let root = temp_dir("doctor-fix");
    let state = root.join("state");
    let host = root.join("host");
    fs::create_dir_all(&state).unwrap();
    fs::create_dir_all(&host).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&state, fs::Permissions::from_mode(0o700)).unwrap();
    }
    // The user's own file, which a fix replaces.
    fs::write(host.join("limits.conf"), b"user's own\n").unwrap();

    let registry = Registry::new(vec![
        DoctorCheck {
            id: "OS-001",
            area: Area::Os,
            needs: Needs::Offline,
            run: limits,
        },
        DoctorCheck {
            id: "OS-002",
            area: Area::Os,
            needs: Needs::Offline,
            run: dropin,
        },
    ])
    .unwrap();
    let mut ledger = HostChangeLedger::open(&state).unwrap();
    let mut out = Vec::new();
    let code = doctor(
        &registry,
        &DoctorCtx::new(&state, Reach::default()),
        Options {
            offline: true,
            json: false,
            fix: true,
        },
        Some(FixSetup {
            ledger: &mut ledger,
            consent: &mut AssumeYes,
            at: 0,
            by_version: "test",
        }),
        &mut out,
    )
    .unwrap();
    let out = String::from_utf8(out).unwrap();
    assert_eq!(code, EXIT_OK, "{out}");
    assert_eq!(fs::read(host.join("limits.conf")).unwrap(), b"fixed\n");
    assert!(host.join("dropin.conf").exists());

    // `doctor --fix` wrote the script; rendering it again from the saved ledger changes nothing.
    let script = if cfg!(windows) {
        state.join("revert-host-changes.ps1")
    } else {
        state.join("revert-host-changes.sh")
    };
    let written = fs::read(&script).unwrap();
    let ledger = HostChangeLedger::open(&state).unwrap();
    assert_eq!(write_revert_script(&ledger, "test").unwrap(), script);
    assert_eq!(fs::read(&script).unwrap(), written);

    let report = run(&script);
    assert_eq!(
        fs::read(host.join("limits.conf")).unwrap(),
        b"user's own\n",
        "{report}"
    );
    assert!(!host.join("dropin.conf").exists(), "{report}");

    // The script leaves markers, so the daemon's next ledger walk records both as reverted.
    let mut ledger = HostChangeLedger::open(&state).unwrap();
    let walked = syntropy_daemon::host_changes::revert::revert_all(
        &mut ledger,
        &mut syntropy_daemon::host_changes::revert::FsHost,
    )
    .unwrap();
    assert_eq!(walked.len(), 2, "{walked:?}");
    assert!(
        ledger
            .changes()
            .iter()
            .all(|c| c.state == ChangeState::Reverted)
    );
}
