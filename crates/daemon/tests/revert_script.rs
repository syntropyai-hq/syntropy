// SPDX-License-Identifier: MIT OR Apache-2.0

//! The generated revert script, run for real (RFC-0052 §11.1): `sh revert-host-changes.sh` on
//! Linux and macOS, `powershell -File revert-host-changes.ps1` on Windows. No Syntropy binary
//! exists anywhere in these runs, as after a user deletes it: the script needs only the OS's tools.
//!
//! Covered here: written files (removed, restored to the content they replaced, or left in place
//! and reported when the user edited them), shell start-up lines, symbolic links on Unix, and on
//! Windows the user `Path` entry and an `HKEY_CURRENT_USER` registry key. The kinds that need root
//! or a real service manager (services, lingering, firewall rules, system users, boot-time
//! settings) are checked as rendered scripts in the unit tests only.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use syntropy_daemon::host_changes::script::write_revert_script;
use syntropy_daemon::host_changes::{ChangeState, HostChangeKind, HostChangeLedger};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "syntropy-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
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

/// Records a change, makes it with `make`, and marks it applied.
fn record(ledger: &mut HostChangeLedger, kind: HostChangeKind, make: impl FnOnce()) -> u32 {
    let id = ledger.begin(kind, None, None, 0, "test").unwrap();
    make();
    ledger.applied(id).unwrap();
    id
}

#[test]
fn the_generated_script_reverts_every_entry_without_the_binary() {
    let state = temp_dir("state");
    let host = temp_dir("host");
    let mut ledger = HostChangeLedger::open(&state).unwrap();

    // A file Syntropy created: removed.
    let created = host.join("created.conf");
    ledger.write_file(&created, b"ours\n", 0, "test").unwrap();
    // A file Syntropy replaced: the old content comes back.
    let replaced = host.join("replaced.conf");
    fs::write(&replaced, b"theirs\n").unwrap();
    ledger.write_file(&replaced, b"ours\n", 0, "test").unwrap();
    // A file the user edited afterwards: left in place and reported.
    let edited = host.join("edited.conf");
    let edited_id = ledger.write_file(&edited, b"ours\n", 0, "test").unwrap();
    fs::write(&edited, b"ours\nand the user's line\n").unwrap();
    // A pending write that never happened: nothing to do.
    ledger
        .begin(
            HostChangeKind::FileWritten {
                path: host.join("never.conf"),
            },
            None,
            None,
            0,
            "test",
        )
        .unwrap();
    // A line added to a shell start-up file: removed, the user's lines kept.
    let rc = host.join("profile");
    let line = "export PATH=\"$HOME/.local/bin:$PATH\"";
    fs::write(&rc, "alias ll='ls -l'\n").unwrap();
    record(
        &mut ledger,
        HostChangeKind::PathEntry {
            shell_rc: Some(rc.clone()),
            line: line.into(),
        },
        || fs::write(&rc, format!("alias ll='ls -l'\n{line}\n")).unwrap(),
    );
    // A symbolic link: removed.
    #[cfg(unix)]
    let link = {
        let link = host.join("syntropy");
        let target = host.join("versions/1/syntropy");
        let (l, t) = (link.clone(), target.clone());
        record(
            &mut ledger,
            HostChangeKind::Symlink {
                path: link.clone(),
                target,
            },
            move || std::os::unix::fs::symlink(&t, &l).unwrap(),
        );
        link
    };
    #[cfg(windows)]
    let windows = windows::record_all(&mut ledger);

    let script = write_revert_script(&ledger, "test").unwrap();
    let out = run(&script);

    assert!(!created.exists(), "{out}");
    assert_eq!(fs::read(&replaced).unwrap(), b"theirs\n", "{out}");
    assert_eq!(
        fs::read(&edited).unwrap(),
        b"ours\nand the user's line\n",
        "{out}"
    );
    assert!(
        out.lines()
            .any(|l| l.contains("left in place") && l.contains(&*edited.to_string_lossy())),
        "the edited file is reported:\n{out}"
    );
    assert!(out.contains("1 change(s) left in place"), "{out}");
    let rc_now = fs::read_to_string(&rc).unwrap();
    assert!(rc_now.contains("alias ll='ls -l'"), "{rc_now}");
    assert!(!rc_now.contains(line), "{rc_now}");
    #[cfg(unix)]
    assert!(fs::symlink_metadata(&link).is_err(), "{out}");
    #[cfg(windows)]
    windows::check_all(&windows, &out);

    // The copies of reverted changes are deleted; the edited file's copy stays for the next run.
    let copies: Vec<String> = fs::read_dir(state.join("host-changes"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".written") || n.ends_with(".prior"))
        .collect();
    assert_eq!(copies, vec![format!("{edited_id}.written")], "{out}");

    // Running it again changes nothing and still reports the edited file.
    let again = run(&script);
    assert!(again.contains("1 change(s) left in place"), "{again}");
    assert_eq!(fs::read(&replaced).unwrap(), b"theirs\n");
    assert!(
        ledger
            .changes()
            .iter()
            .any(|c| c.state == ChangeState::Pending)
    );
}

#[cfg(windows)]
mod windows {
    use super::*;

    /// What the Windows-only changes touched.
    pub(super) struct Touched {
        /// The user `Path` before the test, and the entry added; `None` outside CI.
        path: Option<(String, String)>,
        /// The registry key created; `None` outside CI.
        key: Option<String>,
    }

    fn ps(command: &str) -> String {
        let output = Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", command])
            .output()
            .expect("powershell runs");
        assert!(
            output.status.success(),
            "{command}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_owned()
    }

    fn user_path() -> String {
        ps(
            "(Get-Item -Path 'HKCU:\\Environment').GetValue('Path', '', 'DoNotExpandEnvironmentNames')",
        )
    }

    /// The user `Path` and `HKEY_CURRENT_USER` belong to the account running the tests, so these
    /// cases run only on CI hosts, where that account is a throwaway runner. Elsewhere they are
    /// skipped, and the run says so.
    pub(super) fn record_all(ledger: &mut HostChangeLedger) -> Touched {
        if std::env::var_os("CI").is_none() {
            eprintln!("not on CI: skipping the user Path and registry cases");
            return Touched {
                path: None,
                key: None,
            };
        }
        let unique = format!(
            "SyntropyRevertTest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let before = user_path();
        let entry = format!("C:\\{unique}\\bin");
        record(
            ledger,
            HostChangeKind::WindowsUserPath {
                entry: entry.clone(),
            },
            || {
                ps(&format!(
                    "$k = Get-Item -Path 'HKCU:\\Environment'; \
                     $cur = $k.GetValue('Path', '', 'DoNotExpandEnvironmentNames'); \
                     $kind = if ($null -eq $k.GetValue('Path')) {{ 'ExpandString' }} else {{ $k.GetValueKind('Path') }}; \
                     $new = if ($cur) {{ $cur.TrimEnd(';') + ';{entry}' }} else {{ '{entry}' }}; \
                     New-ItemProperty -Path 'HKCU:\\Environment' -Name 'Path' -Value $new -PropertyType $kind -Force | Out-Null"
                ));
            },
        );
        let key = format!("Software\\{unique}");
        record(
            ledger,
            HostChangeKind::RegistryKey {
                hive: "HKEY_CURRENT_USER".into(),
                key: key.clone(),
            },
            || {
                ps(&format!(
                    "New-Item -Path 'HKCU:\\{key}' -Force | Out-Null; \
                     New-ItemProperty -Path 'HKCU:\\{key}' -Name 'DisplayName' -Value 'Syntropy test' | Out-Null"
                ));
            },
        );
        Touched {
            path: Some((before, entry)),
            key: Some(key),
        }
    }

    pub(super) fn check_all(touched: &Touched, out: &str) {
        if let Some((before, entry)) = &touched.path {
            let now = user_path();
            assert!(
                !now.contains(entry.as_str()),
                "{entry} is still in {now}\n{out}"
            );
            let kept = |p: &str| -> Vec<String> {
                p.split(';')
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect()
            };
            assert_eq!(
                kept(&now),
                kept(before),
                "the other entries are kept\n{out}"
            );
        }
        if let Some(key) = &touched.key {
            assert_eq!(
                ps(&format!(
                    "Test-Path -LiteralPath 'Registry::HKEY_CURRENT_USER\\{key}'"
                )),
                "False",
                "{out}"
            );
        }
    }
}
