// SPDX-License-Identifier: MIT OR Apache-2.0

//! The revert rules (RFC-0052 §11.1).
//!
//! - Revert only what still matches what was recorded. A file the user has since edited is left in
//!   place, and its path is reported.
//! - Restore the prior state where one was recorded: lingering stays on if it was already on.
//! - Never touch anything that is not in the ledger.
//!
//! [`decide`] applies the rules to one change and what the host shows for it. [`revert_all`]
//! walks the ledger in reverse order, asks a [`Host`] what it shows, acts on each decision and
//! records the outcome. [`FsHost`] is the host for the changes the binary reverts itself: written
//! files, symbolic links and shell start-up lines. The other kinds go through the OS's tools, which
//! the generated revert scripts ([`super::script`]) call.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::{
    ChangeState, HostChange, HostChangeKind, HostChangeLedger, LedgerError, PriorState, fingerprint,
};

/// What the host shows for a change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observed {
    /// Nothing of the change is there.
    Absent,
    /// The change is there as it was recorded.
    AsRecorded,
    /// Something is there, but not what was recorded: the user has edited it.
    Changed,
}

/// What a revert does with one change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Remove what the change added.
    Remove,
    /// Put the saved prior content back.
    RestorePrior {
        /// The saved copy.
        from: PathBuf,
    },
    /// Set the prior value again.
    RestoreValue {
        /// The value.
        value: String,
    },
    /// The host was already in the changed state before the change, so leave it.
    KeepPrior,
    /// The user has edited what was written: leave it in place and report it.
    LeaveEdited,
    /// Nothing of the change remains.
    AlreadyGone,
    /// A pending change the host does not show: it was never made.
    NeverMade,
    /// Already reverted or abandoned: nothing to do.
    Done,
}

/// The revert rules for one change, given what the host shows for it.
#[must_use]
pub fn decide(change: &HostChange, observed: Observed) -> Decision {
    match change.state {
        ChangeState::Reverted | ChangeState::Abandoned => return Decision::Done,
        // A pending change may or may not have happened. Only what matches the record is ours.
        ChangeState::Pending if observed != Observed::AsRecorded => return Decision::NeverMade,
        ChangeState::Pending | ChangeState::Applied => {}
    }
    match observed {
        Observed::Absent => Decision::AlreadyGone,
        Observed::Changed => Decision::LeaveEdited,
        Observed::AsRecorded => match &change.prior {
            None => Decision::Remove,
            Some(PriorState::AlreadySet) => Decision::KeepPrior,
            Some(PriorState::SavedAt { path }) => Decision::RestorePrior { from: path.clone() },
            Some(PriorState::Value { value }) => Decision::RestoreValue {
                value: value.clone(),
            },
        },
    }
}

/// What the host shows for a change, and how to undo it.
pub trait Host {
    /// What the host shows for `change`. `ledger` gives the copies the ledger keeps.
    ///
    /// # Errors
    ///
    /// An I/O error, or [`io::ErrorKind::Unsupported`] for a kind this host does not handle.
    fn observe(&self, ledger: &HostChangeLedger, change: &HostChange) -> io::Result<Observed>;

    /// Removes what `change` added.
    ///
    /// # Errors
    ///
    /// As [`Host::observe`].
    fn remove(&mut self, change: &HostChange) -> io::Result<()>;

    /// Puts the content saved at `from` back in place of what `change` wrote.
    ///
    /// # Errors
    ///
    /// As [`Host::observe`].
    fn restore(&mut self, change: &HostChange, from: &Path) -> io::Result<()>;

    /// Sets the setting `change` touched back to `value`.
    ///
    /// # Errors
    ///
    /// As [`Host::observe`].
    fn restore_value(&mut self, change: &HostChange, value: &str) -> io::Result<()>;
}

/// How a revert ended for one change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Undone: removed, or the prior content or value put back.
    Reverted,
    /// The host already had the setting before, so it was left: still recorded as reverted.
    KeptPrior,
    /// Left in place because the user edited it. The path or name is in the report.
    LeftEdited,
    /// Nothing of it remained.
    AlreadyGone,
    /// A pending change that was never made.
    NeverMade,
    /// Already reverted or abandoned before this run.
    Done,
    /// This host cannot revert this kind: the generated revert script does.
    Unsupported,
    /// The host failed to observe or undo it. The change stays in the ledger as it was.
    Failed(String),
}

/// One line of a revert report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reverted {
    /// The change's ID.
    pub id: u32,
    /// A short description of what it touched: a path, a service name.
    pub what: String,
    /// How it ended.
    pub outcome: Outcome,
}

/// A short description of what a change touched, for reports.
#[must_use]
pub fn describe(kind: &HostChangeKind) -> String {
    match kind {
        HostChangeKind::FileWritten { path } => format!("file {}", path.display()),
        HostChangeKind::Symlink { path, target } => {
            format!("symlink {} -> {}", path.display(), target.display())
        }
        HostChangeKind::PathEntry { shell_rc, line } => match shell_rc {
            Some(rc) => format!("line `{line}` in {}", rc.display()),
            None => format!("PATH line `{line}` (shown, not written)"),
        },
        HostChangeKind::WindowsUserPath { entry } => format!("user Path entry {entry}"),
        HostChangeKind::ServiceRegistered { manager, name } => {
            format!("{manager:?} service {name}")
        }
        HostChangeKind::LingerEnabled { user } => format!("lingering for {user}"),
        HostChangeKind::FirewallRule { platform, name } => format!("{platform:?} rule {name}"),
        HostChangeKind::RegistryKey { hive, key } => format!("registry key {hive}\\{key}"),
        HostChangeKind::SystemUser { name } => format!("system user {name}"),
        HostChangeKind::BootTimeSetting { name } => format!("boot-time setting {name}"),
    }
}

/// Reverts every change in the ledger, newest first, under the revert rules. Each change's new
/// state is saved before the next one is touched, so an interrupted revert can be run again.
///
/// # Errors
///
/// [`LedgerError`] if the ledger cannot be saved. A change the host cannot observe or undo is
/// reported as [`Outcome::Failed`] or [`Outcome::Unsupported`], and the walk goes on.
pub fn revert_all(
    ledger: &mut HostChangeLedger,
    host: &mut impl Host,
) -> Result<Vec<Reverted>, LedgerError> {
    let mut report = Vec::new();
    let ids: Vec<u32> = ledger.changes().iter().rev().map(|c| c.id).collect();
    for id in ids {
        let change = ledger.get(id).expect("the ID came from the ledger").clone();
        let what = describe(&change.kind);
        let open = matches!(change.state, ChangeState::Pending | ChangeState::Applied);
        if open && ledger.revert_marker(id).exists() {
            // A revert script already undid it.
            ledger.reverted(id)?;
            ledger.delete_copies(id)?;
            report.push(Reverted {
                id,
                what,
                outcome: Outcome::Done,
            });
            continue;
        }
        let outcome = match host.observe(ledger, &change) {
            Err(e) if e.kind() == io::ErrorKind::Unsupported => Outcome::Unsupported,
            Err(e) => Outcome::Failed(e.to_string()),
            Ok(observed) => act(ledger, host, &change, decide(&change, observed))?,
        };
        report.push(Reverted { id, what, outcome });
    }
    Ok(report)
}

fn act(
    ledger: &mut HostChangeLedger,
    host: &mut impl Host,
    change: &HostChange,
    decision: Decision,
) -> Result<Outcome, LedgerError> {
    let done = match decision {
        Decision::Done => return Ok(Outcome::Done),
        Decision::LeaveEdited => return Ok(Outcome::LeftEdited),
        Decision::NeverMade => {
            close(ledger, change.id, ChangeState::Abandoned)?;
            return Ok(Outcome::NeverMade);
        }
        Decision::AlreadyGone => {
            close(ledger, change.id, ChangeState::Reverted)?;
            return Ok(Outcome::AlreadyGone);
        }
        Decision::KeepPrior => {
            close(ledger, change.id, ChangeState::Reverted)?;
            return Ok(Outcome::KeptPrior);
        }
        Decision::Remove => host.remove(change),
        Decision::RestorePrior { from } => host.restore(change, &from),
        Decision::RestoreValue { value } => host.restore_value(change, &value),
    };
    match done {
        Ok(()) => {
            close(ledger, change.id, ChangeState::Reverted)?;
            Ok(Outcome::Reverted)
        }
        Err(e) if e.kind() == io::ErrorKind::Unsupported => Ok(Outcome::Unsupported),
        Err(e) => Ok(Outcome::Failed(e.to_string())),
    }
}

/// Records a closed change in the ledger, deletes its copies, and writes its revert marker so that
/// a revert script generated earlier skips it too.
fn close(ledger: &mut HostChangeLedger, id: u32, state: ChangeState) -> Result<(), LedgerError> {
    match state {
        ChangeState::Abandoned => ledger.abandoned(id)?,
        _ => ledger.reverted(id)?,
    }
    ledger.delete_copies(id)?;
    let marker = ledger.revert_marker(id);
    if let Some(dir) = marker.parent() {
        fs::create_dir_all(dir).map_err(|source| LedgerError::Io {
            path: dir.to_owned(),
            source,
        })?;
    }
    fs::write(&marker, b"").map_err(|source| LedgerError::Io {
        path: marker,
        source,
    })
}

/// The host as the file system shows it: written files, symbolic links and shell start-up lines.
/// Other kinds are [`io::ErrorKind::Unsupported`]; the generated revert script reverts them.
#[derive(Clone, Copy, Debug, Default)]
pub struct FsHost;

fn unsupported(change: &HostChange) -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        format!(
            "{} is reverted by the generated revert script",
            describe(&change.kind)
        ),
    )
}

impl Host for FsHost {
    fn observe(&self, ledger: &HostChangeLedger, change: &HostChange) -> io::Result<Observed> {
        match &change.kind {
            HostChangeKind::FileWritten { path } => {
                let bytes = match fs::read(path) {
                    Ok(bytes) => bytes,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Observed::Absent),
                    Err(e) => return Err(e),
                };
                let matches = match change.fingerprint {
                    Some(expected) => fingerprint(&bytes) == expected,
                    // An entry an installer script recorded: compare with its copy, if it kept one.
                    None => match fs::read(ledger.written_copy(change.id)) {
                        Ok(copy) => copy == bytes,
                        Err(e) if e.kind() == io::ErrorKind::NotFound => false,
                        Err(e) => return Err(e),
                    },
                };
                Ok(if matches {
                    Observed::AsRecorded
                } else {
                    Observed::Changed
                })
            }
            HostChangeKind::Symlink { path, target } => match fs::read_link(path) {
                Ok(points_to) if &points_to == target => Ok(Observed::AsRecorded),
                Ok(_) => Ok(Observed::Changed),
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Observed::Absent),
                // Something that is not a link is at the path.
                Err(_) if fs::symlink_metadata(path).is_ok() => Ok(Observed::Changed),
                Err(e) => Err(e),
            },
            HostChangeKind::PathEntry { shell_rc, line } => {
                let Some(rc) = shell_rc else {
                    return Ok(Observed::Absent);
                };
                match fs::read_to_string(rc) {
                    Ok(text) if text.lines().any(|l| l == line) => Ok(Observed::AsRecorded),
                    Ok(_) => Ok(Observed::Absent),
                    Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Observed::Absent),
                    Err(e) => Err(e),
                }
            }
            _ => Err(unsupported(change)),
        }
    }

    fn remove(&mut self, change: &HostChange) -> io::Result<()> {
        match &change.kind {
            HostChangeKind::FileWritten { path } | HostChangeKind::Symlink { path, .. } => {
                fs::remove_file(path)
            }
            HostChangeKind::PathEntry {
                shell_rc: Some(rc),
                line,
            } => {
                let text = fs::read_to_string(rc)?;
                let kept: Vec<&str> = text.lines().filter(|l| l != line).collect();
                let mut out = kept.join("\n");
                if text.ends_with('\n') && !kept.is_empty() {
                    out.push('\n');
                }
                fs::write(rc, out)
            }
            _ => Err(unsupported(change)),
        }
    }

    fn restore(&mut self, change: &HostChange, from: &Path) -> io::Result<()> {
        match &change.kind {
            HostChangeKind::FileWritten { path } => fs::copy(from, path).map(|_| ()),
            _ => Err(unsupported(change)),
        }
    }

    fn restore_value(&mut self, change: &HostChange, _value: &str) -> io::Result<()> {
        Err(unsupported(change))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{FirewallKind, ServiceManager};
    use super::*;

    fn change(kind: HostChangeKind, state: ChangeState, prior: Option<PriorState>) -> HostChange {
        HostChange {
            id: 1,
            kind,
            state,
            at: 0,
            by_version: "v".into(),
            fingerprint: None,
            prior,
        }
    }

    /// One example of every kind of change.
    fn every_kind() -> Vec<HostChangeKind> {
        vec![
            HostChangeKind::FileWritten {
                path: "/u/.config/systemd/user/syntropy.service".into(),
            },
            HostChangeKind::Symlink {
                path: "/u/.local/bin/syntropy".into(),
                target: "/u/.local/share/syntropy/current/syntropy".into(),
            },
            HostChangeKind::PathEntry {
                shell_rc: Some("/u/.profile".into()),
                line: "export PATH=\"$HOME/.local/bin:$PATH\"".into(),
            },
            HostChangeKind::WindowsUserPath {
                entry: r"C:\Users\u\AppData\Local\Syntropy\bin".into(),
            },
            HostChangeKind::ServiceRegistered {
                manager: ServiceManager::SystemdUser,
                name: "syntropy.service".into(),
            },
            HostChangeKind::LingerEnabled { user: "u".into() },
            HostChangeKind::FirewallRule {
                platform: FirewallKind::WindowsFirewall,
                name: "Syntropy discovery".into(),
            },
            HostChangeKind::RegistryKey {
                hive: "HKEY_CURRENT_USER".into(),
                key: r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Syntropy".into(),
            },
            HostChangeKind::SystemUser {
                name: "syntropy".into(),
            },
            HostChangeKind::BootTimeSetting {
                name: "iogpu.wired_limit_mb".into(),
            },
        ]
    }

    /// For every kind: an applied change that the host shows as recorded is removed; an edited one
    /// is left; one already gone is closed; a pending one the host does not show was never made.
    #[test]
    fn the_rules_apply_to_every_kind() {
        for kind in every_kind() {
            let applied = change(kind.clone(), ChangeState::Applied, None);
            assert_eq!(
                decide(&applied, Observed::AsRecorded),
                Decision::Remove,
                "{kind:?}"
            );
            assert_eq!(
                decide(&applied, Observed::Changed),
                Decision::LeaveEdited,
                "{kind:?}"
            );
            assert_eq!(
                decide(&applied, Observed::Absent),
                Decision::AlreadyGone,
                "{kind:?}"
            );

            let pending = change(kind.clone(), ChangeState::Pending, None);
            assert_eq!(
                decide(&pending, Observed::AsRecorded),
                Decision::Remove,
                "{kind:?}"
            );
            assert_eq!(
                decide(&pending, Observed::Changed),
                Decision::NeverMade,
                "{kind:?}"
            );
            assert_eq!(
                decide(&pending, Observed::Absent),
                Decision::NeverMade,
                "{kind:?}"
            );

            for state in [ChangeState::Reverted, ChangeState::Abandoned] {
                let closed = change(kind.clone(), state, None);
                assert_eq!(
                    decide(&closed, Observed::AsRecorded),
                    Decision::Done,
                    "{kind:?}"
                );
            }
        }
    }

    #[test]
    fn prior_state_is_restored() {
        let linger = change(
            HostChangeKind::LingerEnabled { user: "u".into() },
            ChangeState::Applied,
            Some(PriorState::AlreadySet),
        );
        assert_eq!(decide(&linger, Observed::AsRecorded), Decision::KeepPrior);

        let file = change(
            HostChangeKind::FileWritten {
                path: "/etc/x".into(),
            },
            ChangeState::Applied,
            Some(PriorState::SavedAt {
                path: "/s/host-changes/1.prior".into(),
            }),
        );
        assert_eq!(
            decide(&file, Observed::AsRecorded),
            Decision::RestorePrior {
                from: "/s/host-changes/1.prior".into()
            }
        );

        let setting = change(
            HostChangeKind::BootTimeSetting {
                name: "iogpu.wired_limit_mb".into(),
            },
            ChangeState::Applied,
            Some(PriorState::Value { value: "0".into() }),
        );
        assert_eq!(
            decide(&setting, Observed::AsRecorded),
            Decision::RestoreValue { value: "0".into() }
        );
        // An edited file is left even when a prior copy exists.
        assert_eq!(decide(&file, Observed::Changed), Decision::LeaveEdited);
    }

    /// A fake host that shows each change as `shown` and records what it was asked to do.
    struct FakeHost {
        shown: Observed,
        did: Vec<String>,
    }

    impl Host for FakeHost {
        fn observe(&self, _: &HostChangeLedger, _: &HostChange) -> io::Result<Observed> {
            Ok(self.shown)
        }
        fn remove(&mut self, c: &HostChange) -> io::Result<()> {
            self.did.push(format!("remove {}", c.id));
            Ok(())
        }
        fn restore(&mut self, c: &HostChange, from: &Path) -> io::Result<()> {
            self.did
                .push(format!("restore {} from {}", c.id, from.display()));
            Ok(())
        }
        fn restore_value(&mut self, c: &HostChange, value: &str) -> io::Result<()> {
            self.did.push(format!("set {} to {value}", c.id));
            Ok(())
        }
    }

    fn temp_state() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "syntropy-revert-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Every kind goes through `revert_all` newest first, and each reverted change is saved as
    /// reverted.
    #[test]
    fn revert_all_walks_every_kind_newest_first() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        for kind in every_kind() {
            let id = ledger.begin(kind, None, None, 0, "v").unwrap();
            ledger.applied(id).unwrap();
        }
        let mut host = FakeHost {
            shown: Observed::AsRecorded,
            did: Vec::new(),
        };
        let report = revert_all(&mut ledger, &mut host).unwrap();
        let ids: Vec<u32> = report.iter().map(|r| r.id).collect();
        assert_eq!(ids, (1..=10).rev().collect::<Vec<_>>());
        assert!(report.iter().all(|r| r.outcome == Outcome::Reverted));
        assert_eq!(host.did.len(), 10);
        let saved = HostChangeLedger::open(&state).unwrap();
        assert!(
            saved
                .changes()
                .iter()
                .all(|c| c.state == ChangeState::Reverted)
        );
    }

    /// The file-system host: a file the user edited after Syntropy wrote it is left in place and
    /// reported; an unedited one is removed; one that replaced a file gets the old content back.
    #[test]
    fn an_edited_file_is_left_in_place_and_reported() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let fresh = state.join("fresh.conf");
        let edited = state.join("edited.conf");
        let replaced = state.join("replaced.conf");
        fs::write(&replaced, b"theirs").unwrap();
        ledger.write_file(&fresh, b"ours", 0, "v").unwrap();
        let edited_id = ledger.write_file(&edited, b"ours", 0, "v").unwrap();
        ledger.write_file(&replaced, b"ours", 0, "v").unwrap();
        fs::write(&edited, b"ours, then the user's edit").unwrap();

        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert!(!fresh.exists());
        assert_eq!(fs::read(&replaced).unwrap(), b"theirs");
        assert_eq!(fs::read(&edited).unwrap(), b"ours, then the user's edit");
        let left: Vec<&Reverted> = report
            .iter()
            .filter(|r| r.outcome == Outcome::LeftEdited)
            .collect();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].id, edited_id);
        assert!(left[0].what.contains("edited.conf"), "{}", left[0].what);
        // The edited file's change stays applied, so a later revert reports it again.
        assert_eq!(
            HostChangeLedger::open(&state)
                .unwrap()
                .get(edited_id)
                .unwrap()
                .state,
            ChangeState::Applied
        );
    }

    #[test]
    fn the_file_system_host_reverts_path_lines_and_symlinks() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let rc = state.join("profile");
        let line = "export PATH=\"$HOME/.local/bin:$PATH\"";
        fs::write(&rc, "alias ll='ls -l'\n").unwrap();
        let id = ledger
            .begin(
                HostChangeKind::PathEntry {
                    shell_rc: Some(rc.clone()),
                    line: line.into(),
                },
                None,
                None,
                0,
                "v",
            )
            .unwrap();
        fs::write(&rc, format!("alias ll='ls -l'\n{line}\n")).unwrap();
        ledger.applied(id).unwrap();

        #[cfg(unix)]
        let link = {
            let link = state.join("syntropy");
            let target = state.join("versions/1/syntropy");
            let id = ledger
                .begin(
                    HostChangeKind::Symlink {
                        path: link.clone(),
                        target: target.clone(),
                    },
                    None,
                    None,
                    0,
                    "v",
                )
                .unwrap();
            std::os::unix::fs::symlink(&target, &link).unwrap();
            ledger.applied(id).unwrap();
            link
        };

        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert!(
            report.iter().all(|r| r.outcome == Outcome::Reverted),
            "{report:?}"
        );
        assert_eq!(fs::read_to_string(&rc).unwrap(), "alias ll='ls -l'\n");
        #[cfg(unix)]
        assert!(fs::symlink_metadata(&link).is_err());
    }

    #[test]
    fn the_file_system_host_leaves_os_kinds_to_the_script() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let id = ledger
            .begin(
                HostChangeKind::LingerEnabled { user: "u".into() },
                None,
                None,
                0,
                "v",
            )
            .unwrap();
        ledger.applied(id).unwrap();
        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert_eq!(report[0].outcome, Outcome::Unsupported);
        assert_eq!(
            HostChangeLedger::open(&state)
                .unwrap()
                .get(id)
                .unwrap()
                .state,
            ChangeState::Applied
        );
    }

    /// A change a revert script already undid is closed without touching the host, and a change
    /// the binary closes gets the marker, so an older script skips it too.
    #[test]
    fn revert_markers_are_honoured_and_written() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let by_script = ledger
            .write_file(&state.join("a.conf"), b"x", 0, "v")
            .unwrap();
        let by_binary = ledger
            .write_file(&state.join("b.conf"), b"x", 0, "v")
            .unwrap();
        fs::write(ledger.revert_marker(by_script), b"").unwrap();
        let mut host = FakeHost {
            shown: Observed::AsRecorded,
            did: Vec::new(),
        };
        let report = revert_all(&mut ledger, &mut host).unwrap();
        assert_eq!(host.did, vec![format!("remove {by_binary}")]);
        let outcome = |id| report.iter().find(|r| r.id == id).unwrap().outcome.clone();
        assert_eq!(outcome(by_script), Outcome::Done);
        assert_eq!(outcome(by_binary), Outcome::Reverted);
        assert!(ledger.revert_marker(by_binary).exists());
        let saved = HostChangeLedger::open(&state).unwrap();
        assert!(
            saved
                .changes()
                .iter()
                .all(|c| c.state == ChangeState::Reverted)
        );
    }

    /// A crash between `begin` and the change: the pending entry is closed as never made.
    #[test]
    fn a_pending_change_the_host_does_not_show_is_abandoned() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let id = ledger
            .begin(
                HostChangeKind::FileWritten {
                    path: state.join("never.conf"),
                },
                None,
                Some(fingerprint(b"x")),
                0,
                "v",
            )
            .unwrap();
        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert_eq!(report[0].outcome, Outcome::NeverMade);
        assert_eq!(
            HostChangeLedger::open(&state)
                .unwrap()
                .get(id)
                .unwrap()
                .state,
            ChangeState::Abandoned
        );
    }
}
