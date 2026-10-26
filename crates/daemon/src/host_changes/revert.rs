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
    ChangeState, Found, HostChange, HostChangeKind, HostChangeLedger, LedgerError, PriorState,
    fingerprint, read_regular, replace_file,
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
    /// The host shows what was there before the change: the content a written file replaced.
    /// Only a change with a saved prior copy can show this.
    AsPrior,
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
    /// A pending change with a saved prior copy, which the host shows neither as recorded nor as
    /// before. It may have partly happened, so it stays pending and its copies are kept: the
    /// prior copy may be the only one left of what was there.
    LeavePending,
    /// Already reverted or abandoned: nothing to do.
    Done,
}

/// The revert rules for one change, given what the host shows for it.
///
/// A pending change may or may not have happened. Only what matches the record is reverted.
/// What matches the state before the change was never made. Anything else was never made if
/// nothing was saved for it, and otherwise is left pending with its copies
/// ([`Decision::LeavePending`]): deleting the prior copy could lose the only copy of a file a
/// half-made change destroyed.
#[must_use]
pub fn decide(change: &HostChange, observed: Observed) -> Decision {
    let saved = matches!(change.prior, Some(PriorState::SavedAt { .. }));
    match (change.state, observed) {
        (ChangeState::Reverted | ChangeState::Abandoned, _) => return Decision::Done,
        (ChangeState::Pending, Observed::AsPrior) => return Decision::NeverMade,
        (ChangeState::Pending, Observed::Absent | Observed::Changed) if saved => {
            return Decision::LeavePending;
        }
        (ChangeState::Pending, Observed::Absent | Observed::Changed) => {
            return Decision::NeverMade;
        }
        (ChangeState::Pending | ChangeState::Applied, _) => {}
    }
    match observed {
        // The host already shows what was there before: nothing left to undo.
        Observed::Absent | Observed::AsPrior => Decision::AlreadyGone,
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
    /// A pending change that may have partly happened: left pending, with its copies, and
    /// reported with its path.
    LeftPending,
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
        HostChangeKind::TrustAnchor {
            store,
            location,
            label,
            node,
            ..
        } => format!(
            "{store:?} trust anchor {label} for node {node} in {}",
            location.display()
        ),
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
    revert_where(ledger, host, |_| true)
}

/// [`revert_all`] for only the changes `select` picks, newest first, under the same rules:
/// `syntropy trust --remove <node>` reverts that node's changes and leaves every other entry as
/// it is. `select` sees each change as the ledger holds it under the lock.
///
/// # Errors
///
/// As [`revert_all`].
pub fn revert_where(
    ledger: &mut HostChangeLedger,
    host: &mut impl Host,
    select: impl Fn(&HostChange) -> bool,
) -> Result<Vec<Reverted>, LedgerError> {
    check_private(ledger)?;
    // One writer at a time: the walk holds the ledger's lock, and starts from what is saved.
    let _lock = ledger.lock()?;
    ledger.reload()?;
    let mut report = Vec::new();
    let ids: Vec<u32> = ledger
        .changes()
        .iter()
        .rev()
        .filter(|c| select(c))
        .map(|c| c.id)
        .collect();
    for id in ids {
        let change = ledger.get(id).expect("the ID came from the ledger").clone();
        let what = describe(&change.kind);
        let open = matches!(change.state, ChangeState::Pending | ChangeState::Applied);
        if open && ledger.revert_marker(id).exists() {
            // A revert script already undid it.
            ledger.set_state_locked(id, ChangeState::Reverted)?;
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
        Decision::LeavePending => return Ok(Outcome::LeftPending),
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
        Decision::RestorePrior { from } => {
            // A saved copy is only ever the ledger's own; any other path is refused.
            if from != ledger.prior_copy(change.id) {
                return Ok(Outcome::Failed(format!(
                    "the prior copy {} is not the ledger's own copy for this change",
                    from.display()
                )));
            }
            host.restore(change, &from)
        }
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

/// Refuses to revert from state that someone else could have written: on Unix the state
/// directory, the copies directory and the ledger file must belong to the running user, be no
/// link, and be writable by no group or other user. A revert may run with more privilege than
/// whoever could edit these, so the ledger is trusted only when they are private.
fn check_private(ledger: &HostChangeLedger) -> Result<(), LedgerError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let state = ledger.state_dir();
        let probe = state.join(format!(".owner-probe-{}", std::process::id()));
        let insecure = |path: &Path, why: &str| LedgerError::Insecure {
            path: path.to_owned(),
            why: why.to_owned(),
        };
        // The probe is made only in a real directory, never through a link.
        let meta = fs::symlink_metadata(state).map_err(|source| LedgerError::Io {
            path: state.to_owned(),
            source,
        })?;
        if !meta.file_type().is_dir() {
            return Err(insecure(state, "it is not a directory: a link or a file"));
        }
        let me = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&probe)
            .and_then(|f| f.metadata())
            .map(|m| m.uid())
            .map_err(|source| LedgerError::Io {
                path: probe.clone(),
                source,
            });
        let _ = fs::remove_file(&probe);
        let me = me?;
        for path in [
            state.to_owned(),
            state.join(super::COPIES_DIR),
            state.join(super::LEDGER_FILE),
        ] {
            let meta = match fs::symlink_metadata(&path) {
                Ok(meta) => meta,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(source) => return Err(LedgerError::Io { path, source }),
            };
            if meta.file_type().is_symlink() {
                return Err(insecure(&path, "it is a symbolic link"));
            }
            if meta.uid() != me {
                return Err(insecure(&path, "it belongs to another user"));
            }
            if meta.mode() & 0o022 != 0 {
                return Err(insecure(&path, "a group or other users can write to it"));
            }
        }
    }
    #[cfg(not(unix))]
    let _ = ledger;
    Ok(())
}

/// Records a closed change in the ledger, deletes its copies, and writes its revert marker so that
/// a revert script generated earlier skips it too.
fn close(ledger: &mut HostChangeLedger, id: u32, state: ChangeState) -> Result<(), LedgerError> {
    match state {
        ChangeState::Abandoned => ledger.set_state_locked(id, ChangeState::Abandoned)?,
        _ => ledger.set_state_locked(id, ChangeState::Reverted)?,
    }
    ledger.delete_copies(id)?;
    // The marker sits with the copies, under the same rules: the directory owner-only and never
    // a link, the marker a new owner-only file.
    ledger.copies_dir()?;
    let marker = ledger.revert_marker(id);
    let file = super::new_private_file(&marker)?;
    file.sync_all().map_err(|source| LedgerError::Io {
        path: marker,
        source,
    })
}

/// Replaces the regular file at `path` with `bytes`, never in place and never through a link,
/// keeping its owner (where the caller may) and mode ([`replace_file`]).
fn replace_in_place(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.file_type().is_file() {
        return Err(io::Error::other(format!(
            "{} is no longer a regular file",
            path.display()
        )));
    }
    replace_file(path, bytes, Some(&meta))
}

/// Whether one line, without its `\n`, is `line`, or `line` and a carriage return.
fn is_line(segment: &[u8], line: &[u8]) -> bool {
    let body = segment.strip_suffix(b"\n").unwrap_or(segment);
    body == line || body.strip_suffix(b"\r") == Some(line)
}

/// `bytes` without every line that is `line` (with a `\n` or `\r\n` ending, or none at the end of
/// the file), and whether there was one. Every other byte is kept as it was: lines are compared as
/// bytes, never decoded, so a start-up file in any encoding, with any line endings, comes back
/// unchanged but for the removed line.
pub(crate) fn without_line(bytes: &[u8], line: &str) -> (Vec<u8>, bool) {
    let line = line.as_bytes();
    let mut kept = Vec::with_capacity(bytes.len());
    let mut found = false;
    for segment in bytes.split_inclusive(|&b| b == b'\n') {
        if is_line(segment, line) {
            found = true;
        } else {
            kept.extend_from_slice(segment);
        }
    }
    (kept, found)
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
                let bytes = match read_regular(path)? {
                    Found::Nothing => return Ok(Observed::Absent),
                    // A link or directory where Syntropy wrote a file is not what it wrote.
                    Found::Other => return Ok(Observed::Changed),
                    Found::File { bytes, .. } => bytes,
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
                if matches {
                    return Ok(Observed::AsRecorded);
                }
                // The content it replaced, from the ledger's own prior copy only.
                let own = ledger.prior_copy(change.id);
                if matches!(&change.prior, Some(PriorState::SavedAt { path }) if *path == own) {
                    match fs::read(&own) {
                        Ok(prior) if prior == bytes => return Ok(Observed::AsPrior),
                        Ok(_) => {}
                        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                        Err(e) => return Err(e),
                    }
                }
                Ok(Observed::Changed)
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
                match read_regular(rc)? {
                    Found::Nothing => Ok(Observed::Absent),
                    // A start-up file that is a link is left alone.
                    Found::Other => Ok(Observed::Changed),
                    Found::File { bytes, .. } => Ok(if without_line(&bytes, line).1 {
                        Observed::AsRecorded
                    } else {
                        Observed::Absent
                    }),
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
                let Found::File { bytes, meta } = read_regular(rc)? else {
                    return Err(io::Error::other(format!(
                        "{} is no longer a regular file",
                        rc.display()
                    )));
                };
                let (kept, _) = without_line(&bytes, line);
                replace_file(rc, &kept, Some(&meta))
            }
            _ => Err(unsupported(change)),
        }
    }

    fn restore(&mut self, change: &HostChange, from: &Path) -> io::Result<()> {
        match &change.kind {
            HostChangeKind::FileWritten { path } => replace_in_place(path, &fs::read(from)?),
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
            HostChangeKind::TrustAnchor {
                store: super::super::TrustStore::LinuxUserNss,
                location: "/u/.pki/nssdb".into(),
                label: "Syntropy CA box.local 0123456789abcdef".into(),
                node: "box.local".into(),
                certificate: "MIIBhjCCASygAwIBAgIU".into(),
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

    /// A private state directory, as a real one is (0700).
    /// A new state directory per call. Tests run in parallel and the clock can repeat between
    /// two calls (Windows counts in 100 ns), so a counter keeps the names apart, and
    /// `create_dir` fails rather than share a directory that already exists.
    fn temp_state() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "syntropy-revert-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        // A directory left by an earlier run with the same process ID.
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir(&dir).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
        }
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
        assert_eq!(ids, (1..=11).rev().collect::<Vec<_>>());
        assert!(report.iter().all(|r| r.outcome == Outcome::Reverted));
        assert_eq!(host.did.len(), 11);
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

    /// A saved copy outside the ledger's copies directory is never restored from: a tampered
    /// ledger cannot make a revert copy an arbitrary file over another.
    #[test]
    fn a_prior_copy_outside_the_ledger_is_refused() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let target = state.join("t.conf");
        let elsewhere = state.join("elsewhere");
        fs::write(&elsewhere, b"attacker's content").unwrap();
        let id = ledger
            .begin(
                HostChangeKind::FileWritten {
                    path: target.clone(),
                },
                Some(PriorState::SavedAt {
                    path: elsewhere.clone(),
                }),
                Some(fingerprint(b"ours")),
                0,
                "v",
            )
            .unwrap();
        fs::write(&target, b"ours").unwrap();
        ledger.applied(id).unwrap();
        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert!(
            matches!(&report[0].outcome, Outcome::Failed(why) if why.contains("not the ledger's own copy")),
            "{report:?}"
        );
        assert_eq!(fs::read(&target).unwrap(), b"ours");
    }

    /// State that another user, or a group, could have written is not trusted.
    #[cfg(unix)]
    #[test]
    fn state_others_can_write_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        ledger
            .write_file(&state.join("a.conf"), b"x", 0, "v")
            .unwrap();
        fs::set_permissions(&state, fs::Permissions::from_mode(0o777)).unwrap();
        let err = revert_all(&mut ledger, &mut FsHost).unwrap_err();
        assert!(matches!(err, LedgerError::Insecure { .. }), "{err}");
        assert!(state.join("a.conf").exists());
        fs::set_permissions(&state, fs::Permissions::from_mode(0o700)).unwrap();
        let copies = state.join(super::super::COPIES_DIR);
        fs::set_permissions(&copies, fs::Permissions::from_mode(0o770)).unwrap();
        let err = revert_all(&mut ledger, &mut FsHost).unwrap_err();
        assert!(matches!(err, LedgerError::Insecure { .. }), "{err}");
    }

    /// A link where Syntropy wrote a file is not what it wrote: left in place, and its target is
    /// never read or written.
    #[cfg(unix)]
    #[test]
    fn a_link_in_place_of_a_written_file_is_left() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let path = state.join("unit.service");
        let id = ledger.write_file(&path, b"ours", 0, "v").unwrap();
        let elsewhere = state.join("someone-elses-file");
        fs::write(&elsewhere, b"ours").unwrap();
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&elsewhere, &path).unwrap();
        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert_eq!(report[0].id, id);
        assert_eq!(report[0].outcome, Outcome::LeftEdited);
        assert_eq!(fs::read(&elsewhere).unwrap(), b"ours");
        assert!(
            fs::symlink_metadata(&path)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    /// Restoring replaces the file through a rename, keeping its permissions.
    #[cfg(unix)]
    #[test]
    fn a_restored_file_keeps_its_mode() {
        use std::os::unix::fs::PermissionsExt;
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let path = state.join("app.conf");
        fs::write(&path, b"theirs").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        ledger.write_file(&path, b"ours", 0, "v").unwrap();
        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert_eq!(report[0].outcome, Outcome::Reverted);
        assert_eq!(fs::read(&path).unwrap(), b"theirs");
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    /// Sets change `id` back to pending, as a crash after `begin` and before `applied` leaves it.
    fn crash_before_applied(ledger: &mut HostChangeLedger, id: u32) {
        let change = ledger.file.changes.iter_mut().find(|c| c.id == id).unwrap();
        change.state = ChangeState::Pending;
        ledger.save().unwrap();
    }

    /// A crash inside `write_file` leaves a pending entry. What the file holds decides what a
    /// revert does, and the copy of what it replaced is deleted only when nothing can need it.
    #[test]
    fn a_crash_inside_write_file_never_loses_the_replaced_content() {
        let state = temp_state();
        let path = state.join("a.conf");
        let original: &[u8] = b"ORIGINAL user content\n";

        // The file holds neither: a half-written file, as an in-place write could leave. It stays
        // pending, the file is left as it is, and the copy of the original is kept.
        fs::write(&path, original).unwrap();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let id = ledger
            .write_file(&path, b"NEW synced content\n", 0, "v")
            .unwrap();
        crash_before_applied(&mut ledger, id);
        fs::write(&path, b"NEW syn").unwrap();
        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert_eq!(report[0].outcome, Outcome::LeftPending);
        assert_eq!(ledger.get(id).unwrap().state, ChangeState::Pending);
        assert_eq!(fs::read(ledger.prior_copy(id)).unwrap(), original);
        assert_eq!(fs::read(&path).unwrap(), b"NEW syn");
        assert!(!ledger.revert_marker(id).exists());

        // The file is gone: still pending, and the copy is still kept.
        fs::remove_file(&path).unwrap();
        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert_eq!(report[0].outcome, Outcome::LeftPending);
        assert_eq!(fs::read(ledger.prior_copy(id)).unwrap(), original);

        // The old file, as a crash before the rename leaves it: never made, and the copy goes.
        fs::write(&path, original).unwrap();
        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert_eq!(report[0].outcome, Outcome::NeverMade);
        assert_eq!(ledger.get(id).unwrap().state, ChangeState::Abandoned);
        assert!(!ledger.prior_copy(id).exists());
        assert_eq!(fs::read(&path).unwrap(), original);

        // The new file, as a crash after the rename leaves it: reverted to the original.
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let id = ledger
            .write_file(&path, b"NEW synced content\n", 0, "v")
            .unwrap();
        crash_before_applied(&mut ledger, id);
        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert_eq!(
            report.iter().find(|r| r.id == id).unwrap().outcome,
            Outcome::Reverted
        );
        assert_eq!(fs::read(&path).unwrap(), original);
    }

    /// An applied change whose file is back to what it replaced has nothing left to undo.
    #[test]
    fn an_applied_file_already_back_to_its_earlier_content_is_closed() {
        let state = temp_state();
        let path = state.join("b.conf");
        fs::write(&path, b"theirs\n").unwrap();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let id = ledger.write_file(&path, b"ours\n", 0, "v").unwrap();
        fs::write(&path, b"theirs\n").unwrap();
        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert_eq!(report[0].outcome, Outcome::AlreadyGone);
        assert_eq!(ledger.get(id).unwrap().state, ChangeState::Reverted);
        assert_eq!(fs::read(&path).unwrap(), b"theirs\n");
    }

    /// Lines are compared and removed as bytes: every other byte of the start-up file comes
    /// back unchanged, whatever its encoding and line endings, and the marked line goes with or
    /// without a carriage return, at the end of the file or not.
    #[test]
    fn start_up_lines_are_removed_as_bytes() {
        let line = "export PATH=\"$HOME/.local/bin:$PATH\"";
        let cases: [(&[u8], &[u8]); 6] = [
            (
                b"# caf\xe9 alias\nexport PATH=\"$HOME/.local/bin:$PATH\"\nexport B=\xe9t\xe9\n",
                b"# caf\xe9 alias\nexport B=\xe9t\xe9\n",
            ),
            (
                b"a=1\r\nexport PATH=\"$HOME/.local/bin:$PATH\"\r\nb=2\r\n",
                b"a=1\r\nb=2\r\n",
            ),
            (
                b"a=1\nexport PATH=\"$HOME/.local/bin:$PATH\"\nb\0x\nc=3\n",
                b"a=1\nb\0x\nc=3\n",
            ),
            (
                b"a=1\nexport PATH=\"$HOME/.local/bin:$PATH\"",
                b"a=1\n",
            ),
            (
                b"export PATH=\"$HOME/.local/bin:$PATH\"\nx\nexport PATH=\"$HOME/.local/bin:$PATH\"\n",
                b"x\n",
            ),
            (
                b"  export PATH=\"$HOME/.local/bin:$PATH\"\nno final newline",
                b"  export PATH=\"$HOME/.local/bin:$PATH\"\nno final newline",
            ),
        ];
        for (i, (before, after)) in cases.iter().enumerate() {
            let (kept, found) = without_line(before, line);
            assert_eq!(kept, *after, "case {i}");
            assert_eq!(found, before != after, "case {i}");
        }

        // Through the host, on a file.
        let state = temp_state();
        let rc = state.join("profile");
        fs::write(&rc, cases[0].0).unwrap();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let kind = HostChangeKind::PathEntry {
            shell_rc: Some(rc.clone()),
            line: line.into(),
        };
        let id = ledger.begin(kind, None, None, 0, "v").unwrap();
        ledger.applied(id).unwrap();
        let report = revert_all(&mut ledger, &mut FsHost).unwrap();
        assert_eq!(report[0].outcome, Outcome::Reverted);
        assert_eq!(fs::read(&rc).unwrap(), cases[0].1);
    }

    /// The owner probe is made only in a real state directory, never through a link.
    #[cfg(unix)]
    #[test]
    fn a_state_directory_that_is_a_link_is_refused_before_anything_is_made() {
        let real = temp_state();
        let link = real.with_extension("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let mut ledger = HostChangeLedger::open(&link).unwrap();
        let err = revert_all(&mut ledger, &mut FsHost).unwrap_err();
        assert!(matches!(err, LedgerError::Insecure { .. }), "{err}");
        assert_eq!(
            fs::read_dir(&real).unwrap().count(),
            0,
            "nothing made in its target"
        );
        fs::remove_file(&link).unwrap();
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
