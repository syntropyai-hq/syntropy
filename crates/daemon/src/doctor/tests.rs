// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unit tests for the doctor framework, with fake checks.

use std::fs;
use std::io::Cursor;
use std::path::PathBuf;

use syntropy_api::figure::Figure;

use super::fix::{AssumeYes, Consent, NoTerminal, Prompt, Targets};
use super::*;
use crate::host_changes::revert::{self, FsHost};
use crate::host_changes::{ChangeState, HostChangeKind};

const VERSION: &str = "0.0.0-test";

/// The fixable checks' files, under a test's directory: a `limits.d` file and a service drop-in,
/// both files `--fix` may write.
const LIMITS: &str = "host/etc/security/limits.d/limits.conf";
const DROPIN: &str = "host/etc/systemd/system/syntropy.service.d/dropin.conf";

/// A new directory per call, owner-only as a state directory is. Tests run in parallel, so a
/// counter keeps the names apart.
fn temp_dir() -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "syntropy-doctor-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("state")).unwrap();
    fs::create_dir_all(dir.join(LIMITS).parent().unwrap()).unwrap();
    fs::create_dir_all(dir.join(DROPIN).parent().unwrap()).unwrap();
    fs::create_dir_all(dir.join("home")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir.join("state"), fs::Permissions::from_mode(0o700)).unwrap();
    }
    dir
}

fn check(
    id: &'static str,
    area: Area,
    needs: Needs,
    run: fn(&DoctorCtx) -> Finding,
) -> DoctorCheck {
    DoctorCheck {
        id,
        area,
        needs,
        run,
    }
}

fn passes(_: &DoctorCtx) -> Finding {
    Finding::pass("fine")
}

fn fails(_: &DoctorCtx) -> Finding {
    Finding::fail("broken")
        .with_evidence("limit", Figure::measured("64 KiB".to_owned(), "getrlimit"))
        .with_fix(Fix {
            command: Some("edit /etc/security/limits.d/syntropy.conf".into()),
            ..Fix::manual("raise the limit")
        })
}

fn warns(_: &DoctorCtx) -> Finding {
    Finding::warn("slow")
}

fn informs(_: &DoctorCtx) -> Finding {
    Finding::new(Status::Info, "worth knowing")
}

fn panics(_: &DoctorCtx) -> Finding {
    panic!("a bug in a check")
}

/// The file the fixable checks look after, under the test's directory beside the state directory:
/// `limits.conf` and `never.conf` in `limits.d`, `dropin.conf` in the service drop-in directory.
fn host_file(ctx: &DoctorCtx, name: &str) -> PathBuf {
    let dir = ctx.state_dir.parent().unwrap();
    match name {
        "dropin.conf" => dir.join(DROPIN),
        _ => dir.join(LIMITS).with_file_name(name),
    }
}

/// The files `--fix` may write under the test's directory.
fn targets(dir: &std::path::Path) -> Targets {
    Targets::under(&dir.join("host"), Some(&dir.join("home")))
}

/// Fails until `limits.conf` holds `fixed`, and offers an automatic fix that writes it.
fn fixable(ctx: &DoctorCtx) -> Finding {
    let path = host_file(ctx, "limits.conf");
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

/// Warns until `dropin.conf` exists, with an automatic fix.
fn fixable_warning(ctx: &DoctorCtx) -> Finding {
    let path = host_file(ctx, "dropin.conf");
    if path.exists() {
        return Finding::pass("drop-in present");
    }
    Finding::warn("drop-in missing").with_fix(Fix {
        automatic: Some(AutoFix {
            edits: vec![HostEdit::WriteFile {
                path,
                contents: b"[Service]\n".to_vec(),
            }],
        }),
        ..Fix::manual("write the drop-in")
    })
}

/// Passes, but carries an automatic fix, which `--fix` must not offer.
fn passes_with_fix(ctx: &DoctorCtx) -> Finding {
    Finding::pass("fine").with_fix(Fix {
        automatic: Some(AutoFix {
            edits: vec![HostEdit::WriteFile {
                path: host_file(ctx, "never.conf"),
                contents: b"x".to_vec(),
            }],
        }),
        ..Fix::manual("never offered")
    })
}

/// Offers a fix that writes a read-only file in `limits.d`, which the ledger refuses.
fn read_only_fix(ctx: &DoctorCtx) -> Finding {
    Finding::fail("bad").with_fix(Fix {
        automatic: Some(AutoFix {
            edits: vec![HostEdit::WriteFile {
                path: host_file(ctx, "read-only.conf"),
                contents: b"x".to_vec(),
            }],
        }),
        ..Fix::manual("a fix the ledger refuses")
    })
}

/// A consent that fails the test if it is ever asked.
struct NeverAsked;

impl Consent for NeverAsked {
    fn ask(&mut self, id: &str, _fix: &Fix, _auto: &AutoFix) -> fix::Answer {
        panic!("{id}'s fix was offered")
    }
}

fn registry(checks: Vec<DoctorCheck>) -> Registry {
    Registry::new(checks).unwrap()
}

fn all_reach() -> Reach {
    Reach {
        daemon: true,
        probe: true,
        cluster: true,
    }
}

#[test]
fn every_area_has_its_own_prefix() {
    let prefixes: BTreeSet<_> = Area::ALL.iter().map(|a| a.prefix()).collect();
    assert_eq!(prefixes.len(), Area::ALL.len());
}

#[test]
fn a_check_id_must_be_its_areas_prefix_and_three_digits() {
    for bad in [
        "OS-1", "OS-0001", "OS003", "DRV-003", "os-003", "OS-00a", "OS-",
    ] {
        let err = Registry::new(vec![check(bad, Area::Os, Needs::Offline, passes)]).unwrap_err();
        assert_eq!(
            err,
            RegistryError::BadId {
                id: bad,
                area: Area::Os
            },
            "{bad}"
        );
    }
    assert!(Registry::new(vec![check("RDMA-004", Area::Rdma, Needs::Offline, passes)]).is_ok());
}

#[test]
fn a_check_id_is_used_once() {
    let err = Registry::new(vec![
        check("OS-001", Area::Os, Needs::Offline, passes),
        check("OS-001", Area::Os, Needs::Offline, fails),
    ])
    .unwrap_err();
    assert_eq!(err, RegistryError::Duplicate { id: "OS-001" });
}

#[test]
fn a_retired_check_id_is_never_reused() {
    assert!(RETIRED.iter().all(|id| {
        let area = Area::ALL
            .into_iter()
            .find(|a| id.starts_with(&format!("{}-", a.prefix())))
            .expect("a retired ID has an area's prefix");
        matches!(
            Registry::new(vec![check(id, area, Needs::Offline, passes)]),
            Err(RegistryError::Retired { .. })
        )
    }));
}

#[test]
fn the_doc_page_is_named_by_the_check_id() {
    let c = check("STO-002", Area::Storage, Needs::Offline, passes);
    assert_eq!(c.doc_page(), "docs/troubleshooting/STO-002.md");
}

#[test]
fn offline_runs_only_the_checks_that_need_no_daemon() {
    let r = registry(vec![
        check("INS-001", Area::Install, Needs::Offline, passes),
        check("RUN-001", Area::Runners, Needs::Daemon, fails),
        check("HW-001", Area::Hardware, Needs::Probe, fails),
        check("CLU-001", Area::Cluster, Needs::Cluster, fails),
    ]);
    let dir = temp_dir();
    // A daemon runs, but `--offline` keeps to the host.
    let ctx = DoctorCtx::new(&dir.join("state"), all_reach()).offline();
    let report = run(&r, &ctx);
    assert!(report.offline);
    assert_eq!(report.get("INS-001").unwrap().finding.status, Status::Pass);
    for id in ["RUN-001", "HW-001", "CLU-001"] {
        assert_eq!(
            report.get(id).unwrap().finding.status,
            Status::Skip {
                why: "--offline: this check needs the daemon".into()
            },
            "{id}"
        );
    }
    assert_eq!(report.exit_code(), EXIT_OK);
}

#[test]
fn a_check_that_cannot_reach_what_it_needs_is_skipped_with_the_reason() {
    let r = registry(vec![
        check("RUN-001", Area::Runners, Needs::Daemon, fails),
        check("HW-001", Area::Hardware, Needs::Probe, fails),
        check("CLU-001", Area::Cluster, Needs::Cluster, fails),
    ]);
    let dir = temp_dir();
    let why = |reach: Reach, id: &str| match &run(&r, &DoctorCtx::new(&dir, reach))
        .get(id)
        .unwrap()
        .finding
        .status
    {
        Status::Skip { why } => why.clone(),
        other => panic!("{id} ran: {other:?}"),
    };
    let none = Reach::default();
    assert_eq!(why(none, "RUN-001"), "the daemon is not running");
    assert_eq!(why(none, "CLU-001"), "the daemon is not running");
    let daemon = Reach {
        daemon: true,
        ..Reach::default()
    };
    assert_eq!(why(daemon, "HW-001"), "the daemon has no probe results yet");
    assert_eq!(why(daemon, "CLU-001"), "this node is not in a cluster");
    let report = run(&r, &DoctorCtx::new(&dir, all_reach()));
    assert!(
        report
            .results
            .iter()
            .all(|r| r.finding.status == Status::Fail)
    );
}

#[test]
fn a_failing_check_makes_the_exit_code_non_zero() {
    let dir = temp_dir();
    let ctx = DoctorCtx::new(&dir, all_reach());
    let ok = registry(vec![
        check("INS-001", Area::Install, Needs::Offline, passes),
        check("INS-002", Area::Install, Needs::Offline, warns),
        check("INS-003", Area::Install, Needs::Offline, informs),
        check("RUN-001", Area::Runners, Needs::Daemon, fails),
    ]);
    // The failing check is skipped offline, so nothing fails.
    assert_eq!(run(&ok, &ctx.clone().offline()).exit_code(), EXIT_OK);
    assert_eq!(run(&ok, &ctx).exit_code(), EXIT_FAILED);
    assert_ne!(EXIT_FAILED, EXIT_OK);

    let mut out = Vec::new();
    let code = doctor(&ok, &ctx, Options::default(), None, &mut out).unwrap();
    assert_eq!(code, EXIT_FAILED);
    let code = doctor(
        &ok,
        &ctx,
        Options {
            offline: true,
            ..Options::default()
        },
        None,
        &mut out,
    )
    .unwrap();
    assert_eq!(code, EXIT_OK);
}

#[test]
fn a_check_that_panics_fails_and_the_run_goes_on() {
    let r = registry(vec![
        check("INS-001", Area::Install, Needs::Offline, panics),
        check("INS-002", Area::Install, Needs::Offline, passes),
    ]);
    let dir = temp_dir();
    let report = run(&r, &DoctorCtx::new(&dir, Reach::default()));
    let first = &report.get("INS-001").unwrap().finding;
    assert_eq!(first.status, Status::Fail);
    assert_eq!(
        first.fix.as_ref().unwrap().doc.as_deref(),
        Some("docs/troubleshooting/INS-001.md")
    );
    assert_eq!(report.get("INS-002").unwrap().finding.status, Status::Pass);
    assert_eq!(report.exit_code(), EXIT_FAILED);
}

#[test]
fn a_fix_without_a_doc_page_gets_its_checks_page() {
    let r = registry(vec![check("OS-001", Area::Os, Needs::Offline, fails)]);
    let dir = temp_dir();
    let report = run(&r, &DoctorCtx::new(&dir, Reach::default()));
    let fix = report.get("OS-001").unwrap().finding.fix.clone().unwrap();
    assert_eq!(fix.doc.as_deref(), Some("docs/troubleshooting/OS-001.md"));
}

#[test]
fn json_output_carries_status_evidence_labels_and_fix() {
    let r = registry(vec![
        check("OS-001", Area::Os, Needs::Offline, fails),
        check("RUN-001", Area::Runners, Needs::Daemon, passes),
    ]);
    let dir = temp_dir();
    let mut out = Vec::new();
    let code = doctor(
        &r,
        &DoctorCtx::new(&dir, Reach::default()),
        Options {
            offline: true,
            json: true,
            fix: false,
        },
        None,
        &mut out,
    )
    .unwrap();
    assert_eq!(code, EXIT_FAILED);
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["schema"], JSON_SCHEMA);
    assert_eq!(v["offline"], true);
    let os = &v["results"][0];
    assert_eq!(os["id"], "OS-001");
    assert_eq!(os["area"], "os");
    assert_eq!(os["needs"], "offline");
    assert_eq!(os["status"], "fail");
    assert_eq!(os["summary"], "broken");
    assert_eq!(
        os["evidence"][0],
        serde_json::json!({
            "name": "limit",
            "value": "64 KiB",
            "kind": "measured",
            "basis": "getrlimit"
        })
    );
    assert_eq!(os["fix"]["description"], "raise the limit");
    assert_eq!(
        os["fix"]["command"],
        "edit /etc/security/limits.d/syntropy.conf"
    );
    assert_eq!(os["fix"]["doc"], "docs/troubleshooting/OS-001.md");
    assert_eq!(os["fix"]["automatic"], serde_json::Value::Null);
    let run = &v["results"][1];
    assert_eq!(run["status"], "skip");
    assert_eq!(run["why"], "--offline: this check needs the daemon");
}

#[test]
fn an_automatic_fix_lists_its_edits_in_json_but_not_the_content() {
    let r = registry(vec![check("OS-001", Area::Os, Needs::Offline, fixable)]);
    let dir = temp_dir();
    let report = run(&r, &DoctorCtx::new(&dir.join("state"), Reach::default()));
    let v: serde_json::Value = serde_json::from_str(&report.to_json()).unwrap();
    let edits = &v["results"][0]["fix"]["automatic"]["edits"];
    assert_eq!(edits.as_array().unwrap().len(), 1);
    let edit = edits[0].as_str().unwrap();
    assert!(
        edit.starts_with("write ") && edit.ends_with("limits.conf (6 bytes)"),
        "{edit}"
    );
    assert!(!report.to_json().contains("fixed\\n"));
}

#[test]
fn text_output_prints_each_check_with_its_evidence_and_fix() {
    let r = registry(vec![
        check("OS-001", Area::Os, Needs::Offline, fails),
        check("RUN-001", Area::Runners, Needs::Daemon, passes),
        check("OS-002", Area::Os, Needs::Offline, fixable),
    ]);
    let dir = temp_dir();
    let text = run(&r, &DoctorCtx::new(&dir.join("state"), Reach::default())).to_text();
    for line in [
        "FAIL  OS-001    broken",
        "      limit: 64 KiB [measured: getrlimit]",
        "      fix: raise the limit",
        "      run: edit /etc/security/limits.d/syntropy.conf",
        "      see: docs/troubleshooting/OS-001.md",
        "SKIP  RUN-001   not run",
        "      skipped: the daemon is not running",
        "      `syntropy doctor --fix` can apply this",
        "0 passed, 0 info, 0 warnings, 2 failed, 1 skipped",
    ] {
        assert!(text.contains(line), "missing {line:?} in:\n{text}");
    }
}

#[test]
fn fix_needs_the_ledger() {
    let r = registry(vec![check("OS-001", Area::Os, Needs::Offline, fixable)]);
    let dir = temp_dir();
    let err = doctor(
        &r,
        &DoctorCtx::new(&dir.join("state"), Reach::default()),
        Options {
            fix: true,
            ..Options::default()
        },
        None,
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(matches!(err, DoctorError::NoLedger));
}

/// Runs `doctor --fix` over `r` with `consent`, and returns the exit code and the output.
fn doctor_fix(
    r: &Registry,
    dir: &std::path::Path,
    consent: &mut dyn fix::Consent,
    json: bool,
) -> (i32, String) {
    let state = dir.join("state");
    let mut ledger = HostChangeLedger::open(&state).unwrap();
    let mut out = Vec::new();
    let code = doctor(
        r,
        &DoctorCtx::new(&state, Reach::default()),
        Options {
            offline: false,
            json,
            fix: true,
        },
        Some(FixSetup {
            ledger: &mut ledger,
            consent,
            targets: &targets(dir),
            at: 1_700_000_000_000,
            by_version: VERSION,
        }),
        &mut out,
    )
    .unwrap();
    (code, String::from_utf8(out).unwrap())
}

#[test]
fn fix_with_yes_applies_records_and_re_runs_and_uninstall_reverts_it() {
    let r = registry(vec![
        check("OS-001", Area::Os, Needs::Offline, fixable),
        check("OS-002", Area::Os, Needs::Offline, fixable_warning),
        check("OS-003", Area::Os, Needs::Offline, passes_with_fix),
    ]);
    let dir = temp_dir();
    let limits = dir.join(LIMITS);
    let dropin = dir.join(DROPIN);
    // The user's own file, which the fix replaces and the revert puts back.
    fs::write(&limits, b"user's own\n").unwrap();

    let (code, out) = doctor_fix(&r, &dir, &mut AssumeYes, false);
    assert_eq!(code, EXIT_OK, "{out}");
    assert!(
        out.contains("fixed OS-001: write limits.conf (host-change ledger entries 1)"),
        "{out}"
    );
    assert!(
        out.contains("fixed OS-002: write the drop-in (host-change ledger entries 2)"),
        "{out}"
    );
    assert!(
        !out.contains("OS-003:"),
        "a passing check's fix is never offered:\n{out}"
    );
    assert!(
        out.contains("PASS  OS-001    limits.conf is set"),
        "re-run:\n{out}"
    );
    assert!(
        out.contains("PASS  OS-002    drop-in present"),
        "re-run:\n{out}"
    );
    assert_eq!(fs::read(&limits).unwrap(), b"fixed\n");
    assert_eq!(fs::read(&dropin).unwrap(), b"[Service]\n");
    assert!(!dir.join(LIMITS).with_file_name("never.conf").exists());

    // Every change is in the ledger, applied, and the revert script was rewritten for it.
    let mut ledger = HostChangeLedger::open(&dir.join("state")).unwrap();
    let kinds: Vec<_> = ledger
        .changes()
        .iter()
        .map(|c| (c.kind.clone(), c.state))
        .collect();
    assert_eq!(
        kinds,
        [
            (
                HostChangeKind::FileWritten {
                    path: limits.clone()
                },
                ChangeState::Applied
            ),
            (
                HostChangeKind::FileWritten {
                    path: dropin.clone()
                },
                ChangeState::Applied
            ),
        ]
    );
    let script = fs::read_to_string(dir.join("state").join(if cfg!(windows) {
        "revert-host-changes.ps1"
    } else {
        "revert-host-changes.sh"
    }))
    .unwrap();
    assert!(script.contains("limits.conf") && script.contains("dropin.conf"));

    // `uninstall`'s revert walk undoes both: the user's file is back, the new one is gone.
    let reverted = revert::revert_all(&mut ledger, &mut FsHost).unwrap();
    assert!(
        reverted
            .iter()
            .all(|r| r.outcome == revert::Outcome::Reverted),
        "{reverted:?}"
    );
    assert_eq!(fs::read(&limits).unwrap(), b"user's own\n");
    assert!(!dropin.exists());
}

#[test]
fn fix_without_a_terminal_or_yes_changes_nothing() {
    let r = registry(vec![check("OS-001", Area::Os, Needs::Offline, fixable)]);
    let dir = temp_dir();
    let (code, out) = doctor_fix(&r, &dir, &mut NoTerminal, false);
    assert_eq!(code, EXIT_FAILED);
    assert!(
        out.contains("not fixed OS-001: no terminal to ask on; run on a terminal, or pass --yes"),
        "{out}"
    );
    assert!(!dir.join(LIMITS).exists());
    let ledger = HostChangeLedger::open(&dir.join("state")).unwrap();
    assert!(ledger.changes().is_empty());
}

#[test]
fn fix_asks_for_each_fix_and_applies_only_the_accepted_ones() {
    let r = registry(vec![
        check("OS-001", Area::Os, Needs::Offline, fixable),
        check("OS-002", Area::Os, Needs::Offline, fixable_warning),
    ]);
    let dir = temp_dir();
    let mut asked = Vec::new();
    let mut prompt = Prompt::new(Cursor::new(b"n\nYes\n".to_vec()), &mut asked);
    let (code, out) = doctor_fix(&r, &dir, &mut prompt, false);
    let asked = String::from_utf8(asked).unwrap();
    assert_eq!(
        asked.matches("Apply this fix? [y/N] ").count(),
        2,
        "{asked}"
    );
    assert!(
        asked.contains("OS-001: write limits.conf\n  write "),
        "{asked}"
    );
    assert!(
        asked.contains("`syntropy uninstall` reverts it."),
        "{asked}"
    );
    assert!(out.contains("not fixed OS-001: declined"), "{out}");
    assert!(out.contains("fixed OS-002"), "{out}");
    // OS-001 still fails.
    assert_eq!(code, EXIT_FAILED);
    assert!(!dir.join(LIMITS).exists());
    assert!(dir.join(DROPIN).exists());
    let ledger = HostChangeLedger::open(&dir.join("state")).unwrap();
    assert_eq!(ledger.changes().len(), 1);
}

#[test]
fn a_prompt_with_no_answer_declines() {
    let r = registry(vec![check("OS-001", Area::Os, Needs::Offline, fixable)]);
    let dir = temp_dir();
    let mut prompt = Prompt::new(Cursor::new(Vec::new()), Vec::new());
    let (code, out) = doctor_fix(&r, &dir, &mut prompt, false);
    assert_eq!(code, EXIT_FAILED);
    assert!(
        out.contains("not fixed OS-001: no answer (end of input)"),
        "{out}"
    );
}

#[test]
fn a_fix_the_ledger_refuses_is_reported_and_the_others_still_apply() {
    let r = registry(vec![
        check("OS-001", Area::Os, Needs::Offline, read_only_fix),
        check("OS-002", Area::Os, Needs::Offline, fixable_warning),
    ]);
    let dir = temp_dir();
    let read_only = dir.join(LIMITS).with_file_name("read-only.conf");
    fs::write(&read_only, b"theirs\n").unwrap();
    let mut perms = fs::metadata(&read_only).unwrap().permissions();
    perms.set_readonly(true);
    fs::set_permissions(&read_only, perms).unwrap();
    let (code, out) = doctor_fix(&r, &dir, &mut AssumeYes, true);
    assert_eq!(code, EXIT_FAILED);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    let fixes = v["fixes"].as_array().unwrap();
    assert_eq!(fixes[0]["id"], "OS-001");
    assert_eq!(fixes[0]["outcome"], "failed");
    let error = fixes[0]["error"].as_str().unwrap();
    assert!(
        error.starts_with("write ") && error.contains("read-only.conf (1 bytes): "),
        "{error}"
    );
    assert_eq!(fixes[0]["changes"], serde_json::json!([]));
    assert_eq!(fixes[1]["id"], "OS-002");
    assert_eq!(fixes[1]["outcome"], "applied");
    assert_eq!(fixes[1]["changes"], serde_json::json!([1]));
    assert_eq!(v["results"][1]["status"], "pass");
    assert_eq!(fs::read(&read_only).unwrap(), b"theirs\n");
}

/// A check whose automatic fix writes `path`, for the target tests.
fn writes(path: PathBuf) -> Finding {
    Finding::fail("needs a write").with_fix(Fix {
        automatic: Some(AutoFix {
            edits: vec![HostEdit::WriteFile {
                path,
                contents: b"x\n".to_vec(),
            }],
        }),
        ..Fix::manual("write a file")
    })
}

#[test]
fn only_limits_files_and_service_drop_ins_may_be_written() {
    let dir = temp_dir();
    let t = targets(&dir);
    let host = dir.join("host");
    let home = dir.join("home");
    let edit = |path: PathBuf| HostEdit::WriteFile {
        path,
        contents: Vec::new(),
    };
    for allowed in [
        host.join("etc/security/limits.d/syntropy.conf"),
        host.join("etc/systemd/system/syntropy.service.d/memlock.conf"),
        // A drop-in directory that does not exist yet.
        host.join("etc/systemd/system/other.service.d/memlock.conf"),
        home.join(".config/systemd/user/syntropy.service.d/memlock.conf"),
    ] {
        assert_eq!(
            t.refuse(&edit(allowed.clone())),
            None,
            "{}",
            allowed.display()
        );
    }
    for refused in [
        // Sysctls, module parameters (the TTM `pages_limit`), the boot loader, boot-time plists.
        host.join("etc/sysctl.d/99-syntropy.conf"),
        host.join("etc/sysctl.conf"),
        host.join("etc/modprobe.d/ttm.conf"),
        host.join("etc/default/grub"),
        host.join("etc/kernel/cmdline"),
        host.join("Library/LaunchDaemons/syntropy.wired.plist"),
        // Near misses of the patterns.
        host.join("etc/security/limits.conf"),
        host.join("etc/security/limits.d/syntropy"),
        host.join("etc/security/limits.d/.conf"),
        host.join("etc/security/limits.d/sub/syntropy.conf"),
        host.join("etc/systemd/system/syntropy.service"),
        host.join("etc/systemd/system/syntropy.service.d/memlock"),
        host.join("etc/systemd/system/syntropy.socket.d/memlock.conf"),
        home.join("etc/security/limits.d/syntropy.conf"),
        host.join(".config/systemd/user/syntropy.service.d/memlock.conf"),
        // `..` that would leave the pattern's directory.
        host.join("etc/security/limits.d/../../sysctl.d/x.conf"),
        PathBuf::from("etc/security/limits.d/syntropy.conf"),
    ] {
        assert!(
            t.refuse(&edit(refused.clone())).is_some(),
            "{}",
            refused.display()
        );
    }
    // Without a home, nothing under it may be written.
    let no_home = Targets::under(&host, None);
    assert!(
        no_home
            .refuse(&edit(
                home.join(".config/systemd/user/syntropy.service.d/memlock.conf")
            ))
            .is_some()
    );
}

#[cfg(unix)]
#[test]
fn a_limits_directory_that_links_elsewhere_is_refused() {
    let dir = temp_dir();
    let host = dir.join("host");
    let sysctl = host.join("etc/sysctl.d");
    fs::create_dir_all(&sysctl).unwrap();
    fs::remove_dir(host.join("etc/security/limits.d")).unwrap();
    std::os::unix::fs::symlink(&sysctl, host.join("etc/security/limits.d")).unwrap();
    let why = targets(&dir)
        .refuse(&HostEdit::WriteFile {
            path: host.join("etc/security/limits.d/99-syntropy.conf"),
            contents: Vec::new(),
        })
        .expect("refused");
    assert!(why.contains("resolves to"), "{why}");
}

/// Fails with an automatic fix that writes a sysctl file.
fn sysctl_fix(ctx: &DoctorCtx) -> Finding {
    writes(
        ctx.state_dir
            .parent()
            .unwrap()
            .join("host/etc/sysctl.d/99-syntropy.conf"),
    )
}

#[test]
fn a_fix_outside_the_targets_is_refused_without_asking_or_recording() {
    let r = registry(vec![
        check("OS-001", Area::Os, Needs::Offline, sysctl_fix),
        check("OS-002", Area::Os, Needs::Offline, passes),
    ]);
    let dir = temp_dir();
    let (code, out) = doctor_fix(&r, &dir, &mut NeverAsked, false);
    assert_eq!(code, EXIT_FAILED);
    assert!(out.contains("fix refused OS-001: "), "{out}");
    assert!(
        out.contains("is not a file doctor --fix may write"),
        "{out}"
    );
    assert!(!dir.join("host/etc/sysctl.d/99-syntropy.conf").exists());
    let ledger = HostChangeLedger::open(&dir.join("state")).unwrap();
    assert!(ledger.changes().is_empty());
    // Nothing changed, so no revert script was written.
    assert!(!dir.join("state/revert-host-changes.sh").exists());
    assert!(!dir.join("state/revert-host-changes.ps1").exists());
}

#[test]
fn the_prompt_shows_short_text_content_in_full() {
    let r = registry(vec![check("OS-001", Area::Os, Needs::Offline, fixable)]);
    let dir = temp_dir();
    let mut asked = Vec::new();
    let mut prompt = Prompt::new(Cursor::new(b"n\n".to_vec()), &mut asked);
    doctor_fix(&r, &dir, &mut prompt, false);
    let asked = String::from_utf8(asked).unwrap();
    assert!(asked.contains("(6 bytes)\n    | fixed\n"), "{asked}");
}
