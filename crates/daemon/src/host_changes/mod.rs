// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host-change ledger (RFC-0052 §11.1).
//!
//! Every change Syntropy makes to the host outside its own directories is recorded in
//! `<state>/host-changes.json` **before** it is made, marked [`ChangeState::Pending`], and marked
//! [`ChangeState::Applied`] afterwards. Its writers are the installers, `setup`, `service`,
//! `doctor --fix` and opt-in features. Uninstall reverts the ledger in reverse order under the
//! revert rules ([`revert`]), and [`script`] renders the same rules as `revert-host-changes.sh`
//! and `.ps1`, which undo everything even when the binary is already deleted.
//!
//! **Copies of what was written.** For each [`HostChangeKind::FileWritten`] the ledger keeps the
//! bytes it wrote in `<state>/host-changes/<id>.written`, and the content it replaced, if any, in
//! `<id>.prior`. The binary checks a file against the entry's BLAKE3 `fingerprint`. The revert
//! scripts cannot compute BLAKE3 with standard tools, so they compare the file with the
//! `.written` copy instead (`cmp` on Unix, `Get-FileHash` on Windows). A written file can be a
//! config or credential file, so the copies directory is owner-only (0700) and each copy is too:
//! 0600, or the original's mode where that is stricter ([`copy_mode`]). Every written file is
//! copied, whatever its size; the files Syntropy writes outside its directories are units, plists,
//! drop-ins and completions, kilobytes each. A copy is deleted when its change is reverted,
//! abandoned or pruned ([`HostChangeLedger::prune_closed`]).
//!
//! **Never in place.** [`HostChangeLedger::write_file`] and every restore write a new file beside
//! the target and rename it over the target (`replace_file`), so a crash leaves the old file or the
//! new one, and a link at the target is replaced, never written through. A pending change with a
//! saved copy that the host shows neither as recorded nor as before is left pending with its
//! copies ([`revert::decide`]). Start-up lines are removed as bytes (`revert::without_line`).
//!
//! **Revert markers.** The scripts cannot edit the JSON ledger, so a script that undoes a change
//! writes `<state>/host-changes/<id>.reverted`, and checks for it first, which makes a second run
//! skip what the first undid. The binary writes the same marker when it closes a change, and treats
//! an existing marker as reverted.
//!
//! **One writer at a time, by a lock.** The writers are separate processes: the installers,
//! `setup`, `service`, `doctor --fix`, opt-in features, and `trust` on a client (ADR-0505 item 6).
//! Every change to the ledger takes an exclusive lock on `<state>/host-changes.lock` ([`LOCK_FILE`]),
//! re-reads the ledger under it, changes it and saves it, so no entry is lost and no ID, and so no
//! pair of copies, is handed out twice. [`HostChangeLedger::write_file`] holds the lock from the
//! choice of ID to `Applied`; a revert holds it for the whole walk, and the revert script is
//! rendered under it. A writer that cannot get the lock within [`LOCK_WAIT`] gets
//! [`LedgerError::Busy`], never a hang. The scripts take the same lock: the sh script with
//! `flock(1)` where it exists, and the PowerShell script by opening the lock file with no sharing.

pub mod revert;
pub mod script;

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use syntropy_core::Blake3;

/// The ledger file in the state directory.
pub const LEDGER_FILE: &str = "host-changes.json";

/// The directory, in the state directory, that holds the copies of written and replaced files.
pub const COPIES_DIR: &str = "host-changes";

/// The file whose exclusive lock every writer of the ledger holds, under the state directory.
pub const LOCK_FILE: &str = "host-changes.lock";

/// How long a writer waits for another to release the lock before it gives up with
/// [`LedgerError::Busy`]: 10 s [target], a chosen budget, longer than any one change takes and
/// short enough that a command never seems to hang.
pub const LOCK_WAIT: Duration = Duration::from_secs(10);

/// The ledger file's format version.
pub const SCHEMA: u16 = 1;

/// One change to the host, recorded before it is made (RFC-0052 §11.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostChange {
    /// Unique in the ledger, assigned in increasing order.
    pub id: u32,
    /// What was changed.
    pub kind: HostChangeKind,
    /// Where the change is in its life.
    pub state: ChangeState,
    /// When the change was recorded, in Unix milliseconds.
    pub at: u64,
    /// The Syntropy version that recorded it.
    pub by_version: String,
    /// A hash of what was written, to detect later user edits. For a written file, the BLAKE3 of
    /// its bytes.
    pub fingerprint: Option<Blake3>,
    /// The state before the change, which a revert restores.
    pub prior: Option<PriorState>,
}

/// What a change touched (RFC-0052 §11.1).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostChangeKind {
    /// A file written outside Syntropy's own directories: units, plists, `limits.d`, drop-ins,
    /// completions.
    FileWritten {
        /// The file.
        path: PathBuf,
    },
    /// A symbolic link at `path` pointing to `target`.
    Symlink {
        /// The link.
        path: PathBuf,
        /// What it points to.
        target: PathBuf,
    },
    /// A line added to a shell start-up file. `shell_rc` is `None` when the line was shown to the
    /// user rather than written, so there is nothing to revert.
    PathEntry {
        /// The start-up file, such as `~/.profile`.
        shell_rc: Option<PathBuf>,
        /// The exact line added.
        line: String,
    },
    /// An entry added to the Windows user `Path` variable.
    WindowsUserPath {
        /// The entry, a directory.
        entry: String,
    },
    /// A service or task registered with the OS.
    ServiceRegistered {
        /// The service manager.
        manager: ServiceManager,
        /// The service's name, label or task name.
        name: String,
    },
    /// systemd lingering enabled for a user.
    LingerEnabled {
        /// The user.
        user: String,
    },
    /// A firewall rule added.
    FirewallRule {
        /// The firewall.
        platform: FirewallKind,
        /// The rule's name.
        name: String,
    },
    /// A Windows registry key created: the uninstall entry, an event-log source.
    RegistryKey {
        /// The hive, by its full name, such as `HKEY_CURRENT_USER`.
        hive: String,
        /// The key's path under the hive.
        key: String,
    },
    /// An OS account created by a `--system` install.
    SystemUser {
        /// The account name.
        name: String,
    },
    /// A setting applied at boot, such as the macOS wired-limit LaunchDaemon's sysctl.
    BootTimeSetting {
        /// The setting's name.
        name: String,
    },
}

/// Where a change is in its life (RFC-0052 §11.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeState {
    /// Recorded, and about to be made or being made. After a crash the change may or may not have
    /// happened, so a revert checks the host.
    Pending,
    /// Made.
    Applied,
    /// Undone, or nothing of it remains on the host.
    Reverted,
    /// Never made: a pending change that the host does not show.
    Abandoned,
}

/// The host's state before a change, which a revert restores.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriorState {
    /// The setting was already in the changed state: lingering was already on. A revert leaves it.
    AlreadySet,
    /// The previous content was saved at `path`. A revert puts it back.
    SavedAt {
        /// The saved copy, normally `<state>/host-changes/<id>.prior`.
        path: PathBuf,
    },
    /// The previous value of a setting. A revert sets it again.
    Value {
        /// The value, as the OS reports it.
        value: String,
    },
}

/// A service manager (RFC-0052 §6).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceManager {
    /// A systemd user unit.
    SystemdUser,
    /// A systemd system unit.
    SystemdSystem,
    /// A launchd agent in the user's GUI domain.
    LaunchdAgent,
    /// A launchd daemon in the system domain.
    LaunchdDaemon,
    /// The Windows Service Control Manager.
    Scm,
    /// Windows Task Scheduler.
    TaskScheduler,
}

/// A host firewall.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FirewallKind {
    /// Windows Defender Firewall; the rule is named by its display name.
    WindowsFirewall,
    /// The macOS application firewall; the rule is the application's path.
    MacosApplicationFirewall,
    /// firewalld; the rule is a service name.
    Firewalld,
    /// ufw; the rule is an application profile name.
    Ufw,
}

/// The on-disk form of the ledger.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct LedgerFile {
    schema: u16,
    next_id: u32,
    changes: Vec<HostChange>,
}

/// Why the ledger could not be read or written.
#[non_exhaustive]
#[derive(Debug)]
pub enum LedgerError {
    /// A file could not be read or written.
    Io {
        /// The file.
        path: PathBuf,
        /// The error.
        source: io::Error,
    },
    /// The ledger file is not valid JSON of this format.
    Parse {
        /// The file.
        path: PathBuf,
        /// The error.
        source: serde_json::Error,
    },
    /// The ledger file is of a format version this build does not read.
    Schema {
        /// The version found.
        found: u16,
    },
    /// No change has this ID.
    UnknownId(u32),
    /// Another process holds the ledger's lock, and did not release it within [`LOCK_WAIT`].
    Busy {
        /// The lock file.
        path: PathBuf,
    },
    /// A start-up line holds a line break (`\n`, `\r`) or a NUL byte. The sh script would read a
    /// line break as two lines, and can't carry a NUL byte (a shell string ends at one), so either
    /// way its pattern would match a line of the user's that Syntropy never wrote. The ledger
    /// records only single lines of text.
    ControlCharacter {
        /// The line.
        line: String,
    },
    /// The file [`HostChangeLedger::write_file`] would replace is read-only. Its owner made it so,
    /// and an in-place write would have been refused, so the replacement is refused too.
    ReadOnly {
        /// The file.
        path: PathBuf,
    },
    /// A path in a change does not start at a root. The revert scripts would resolve a relative
    /// path against whatever directory they are run from, so the ledger records only rooted paths.
    NotRooted {
        /// The path.
        path: PathBuf,
    },
    /// A path the ledger would use could have been written or redirected by someone else: a
    /// state directory, copies directory or ledger that is not private to this user, or a
    /// symbolic link where a file or directory should be. It is refused rather than used.
    Insecure {
        /// The file or directory.
        path: PathBuf,
        /// Why it is not private.
        why: String,
    },
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LedgerError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            LedgerError::Parse { path, source } => {
                write!(
                    f,
                    "{} is not a host-change ledger: {source}",
                    path.display()
                )
            }
            LedgerError::Schema { found } => write!(
                f,
                "the host-change ledger has format version {found}; this build reads {SCHEMA}"
            ),
            LedgerError::UnknownId(id) => write!(f, "no host change has ID {id}"),
            LedgerError::Busy { path } => write!(
                f,
                "another Syntropy command is changing this host ({} is locked); try again when it \
                 has finished",
                path.display()
            ),
            LedgerError::ControlCharacter { line } => write!(
                f,
                "a start-up line must be a single line of text, but {line:?} holds a line break \
                 or a NUL byte"
            ),
            LedgerError::ReadOnly { path } => write!(
                f,
                "{} is read-only; Syntropy does not replace a file its owner made read-only",
                path.display()
            ),
            LedgerError::NotRooted { path } => write!(
                f,
                "{} is not an absolute path; the host-change ledger records only absolute paths",
                path.display()
            ),
            LedgerError::Insecure { path, why } => {
                write!(f, "refusing to use {}: {why}", path.display())
            }
        }
    }
}

impl std::error::Error for LedgerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LedgerError::Io { source, .. } => Some(source),
            LedgerError::Parse { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> LedgerError + '_ {
    move |source| LedgerError::Io {
        path: path.to_owned(),
        source,
    }
}

/// The BLAKE3 of `bytes`, as a ledger fingerprint.
#[must_use]
pub fn fingerprint(bytes: &[u8]) -> Blake3 {
    Blake3(*blake3::hash(bytes).as_bytes())
}

/// The host-change ledger of one state directory.
#[derive(Debug)]
pub struct HostChangeLedger {
    state_dir: PathBuf,
    file: LedgerFile,
    lock_wait: Duration,
}

impl HostChangeLedger {
    /// Opens the ledger in `state_dir`. A missing ledger file is an empty ledger.
    ///
    /// # Errors
    ///
    /// [`LedgerError`] if the file exists and cannot be read or parsed.
    pub fn open(state_dir: &Path) -> Result<Self, LedgerError> {
        Ok(Self {
            state_dir: state_dir.to_owned(),
            file: read_ledger(state_dir)?,
            lock_wait: LOCK_WAIT,
        })
    }

    /// The same ledger with a shorter wait for the lock, for tests.
    #[cfg(test)]
    pub(crate) fn with_lock_wait(mut self, wait: Duration) -> Self {
        self.lock_wait = wait;
        self
    }

    /// Takes the ledger's exclusive lock ([`LOCK_FILE`]), waiting up to [`LOCK_WAIT`] for another
    /// writer to release it. The lock is released when the guard is dropped. A writer re-reads the
    /// ledger under the lock ([`HostChangeLedger::reload`]) before it changes anything, so no
    /// entry is lost and no ID is handed out twice.
    pub(crate) fn lock(&self) -> Result<LedgerLock, LedgerError> {
        fs::create_dir_all(&self.state_dir).map_err(io_error(&self.state_dir))?;
        let path = self.state_dir.join(LOCK_FILE);
        let started = Instant::now();
        loop {
            match open_lock_file(&path) {
                Ok(file) => match file.try_lock() {
                    Ok(()) => return Ok(LedgerLock { file }),
                    Err(std::fs::TryLockError::WouldBlock) => {}
                    Err(std::fs::TryLockError::Error(e)) => return Err(io_error(&path)(e)),
                },
                // The PowerShell revert script holds the file open with no sharing.
                Err(e) if sharing_violation(&e) => {}
                Err(e) if e.kind() == io::ErrorKind::InvalidInput => {
                    return Err(LedgerError::Insecure {
                        path,
                        why: "it is not a regular file".into(),
                    });
                }
                Err(e) => return Err(io_error(&path)(e)),
            }
            if started.elapsed() >= self.lock_wait {
                return Err(LedgerError::Busy { path });
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Reads the ledger again from disk, as another writer may have saved it since. Callers hold
    /// the lock.
    pub(crate) fn reload(&mut self) -> Result<(), LedgerError> {
        self.file = read_ledger(&self.state_dir)?;
        Ok(())
    }

    /// The state directory the ledger lives in.
    #[must_use]
    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    /// Every change, in the order recorded.
    #[must_use]
    pub fn changes(&self) -> &[HostChange] {
        &self.file.changes
    }

    /// The change with `id`.
    #[must_use]
    pub fn get(&self, id: u32) -> Option<&HostChange> {
        self.file.changes.iter().find(|c| c.id == id)
    }

    /// Where the bytes written by change `id` are kept.
    #[must_use]
    pub fn written_copy(&self, id: u32) -> PathBuf {
        self.state_dir
            .join(COPIES_DIR)
            .join(format!("{id}.written"))
    }

    /// The marker that says change `id` has been reverted, which the revert scripts write and
    /// check, since they cannot edit the ledger.
    #[must_use]
    pub fn revert_marker(&self, id: u32) -> PathBuf {
        self.state_dir
            .join(COPIES_DIR)
            .join(format!("{id}.reverted"))
    }

    /// Where the content replaced by change `id` is kept.
    #[must_use]
    pub fn prior_copy(&self, id: u32) -> PathBuf {
        self.state_dir.join(COPIES_DIR).join(format!("{id}.prior"))
    }

    /// Records a change as [`ChangeState::Pending`] and saves the ledger, before the change is
    /// made. Returns its ID.
    ///
    /// # Errors
    ///
    /// [`LedgerError::Io`] if the ledger cannot be saved; then the change must not be made.
    /// [`LedgerError::NotRooted`] if a path in `kind` is relative, and [`LedgerError::ControlCharacter`]
    /// if its start-up line holds a line break.
    pub fn begin(
        &mut self,
        kind: HostChangeKind,
        prior: Option<PriorState>,
        fingerprint: Option<Blake3>,
        at: u64,
        by_version: &str,
    ) -> Result<u32, LedgerError> {
        let _lock = self.lock()?;
        self.reload()?;
        self.begin_locked(kind, prior, fingerprint, at, by_version)
    }

    /// [`HostChangeLedger::begin`], for a caller that holds the lock and has re-read the ledger.
    fn begin_locked(
        &mut self,
        kind: HostChangeKind,
        prior: Option<PriorState>,
        fingerprint: Option<Blake3>,
        at: u64,
        by_version: &str,
    ) -> Result<u32, LedgerError> {
        check_rooted(&kind)?;
        let id = self.file.next_id;
        self.file.next_id += 1;
        self.file.changes.push(HostChange {
            id,
            kind,
            state: ChangeState::Pending,
            at,
            by_version: by_version.to_owned(),
            fingerprint,
            prior,
        });
        self.save()?;
        Ok(id)
    }

    /// Marks change `id` as made, and saves the ledger.
    ///
    /// # Errors
    ///
    /// [`LedgerError::UnknownId`], or [`LedgerError::Io`] if the ledger cannot be saved.
    pub fn applied(&mut self, id: u32) -> Result<(), LedgerError> {
        self.set_state(id, ChangeState::Applied)
    }

    /// Marks change `id` as reverted, and saves the ledger.
    ///
    /// # Errors
    ///
    /// As [`HostChangeLedger::applied`].
    pub fn reverted(&mut self, id: u32) -> Result<(), LedgerError> {
        self.set_state(id, ChangeState::Reverted)
    }

    /// Marks change `id` as never made, and saves the ledger.
    ///
    /// # Errors
    ///
    /// As [`HostChangeLedger::applied`].
    pub fn abandoned(&mut self, id: u32) -> Result<(), LedgerError> {
        self.set_state(id, ChangeState::Abandoned)
    }

    fn set_state(&mut self, id: u32, state: ChangeState) -> Result<(), LedgerError> {
        let _lock = self.lock()?;
        self.reload()?;
        self.set_state_locked(id, state)
    }

    /// Sets change `id`'s state and saves, for a caller that holds the lock and has re-read the
    /// ledger.
    pub(crate) fn set_state_locked(
        &mut self,
        id: u32,
        state: ChangeState,
    ) -> Result<(), LedgerError> {
        let change = self
            .file
            .changes
            .iter_mut()
            .find(|c| c.id == id)
            .ok_or(LedgerError::UnknownId(id))?;
        change.state = state;
        self.save()
    }

    /// Writes `contents` to `path` under the ledger: the content it replaces is saved, the bytes
    /// are copied to the ledger, the change is recorded as pending, the file is written, and the
    /// change is marked applied. Returns the change's ID.
    ///
    /// The file is never written in place. The new content goes into a new file beside it,
    /// owner-only until it has the old file's owner and mode, and that file is synced and renamed
    /// over the old one. A crash therefore leaves the old file or the new one, never a mix, and a
    /// link at `path` is replaced, never followed. A new file, where none was, is 0644 on Unix: the
    /// files Syntropy writes outside its directories are units, plists, drop-ins and completions,
    /// which other users and the system read.
    ///
    /// # Errors
    ///
    /// [`LedgerError::Io`] if a copy, the ledger or the file cannot be written. A failure after
    /// the change was recorded leaves it pending, which a revert resolves by checking the host.
    /// [`LedgerError::Insecure`] if something other than a regular file, such as a link or a
    /// directory, is at `path`. [`LedgerError::ReadOnly`] if the file there is read-only.
    /// [`LedgerError::NotRooted`] if `path` is relative.
    pub fn write_file(
        &mut self,
        path: &Path,
        contents: &[u8],
        at: u64,
        by_version: &str,
    ) -> Result<u32, LedgerError> {
        let kind = HostChangeKind::FileWritten {
            path: path.to_owned(),
        };
        check_rooted(&kind)?;
        // The lock covers the ID, the copies written under it, the entry and the host write.
        let _lock = self.lock()?;
        self.reload()?;
        let id = self.file.next_id;
        // Syntropy replaces a regular file, or makes one where nothing is; it never writes through
        // a link someone left in its place, or over a directory.
        let old = match read_regular(path).map_err(io_error(path))? {
            Found::Nothing => None,
            Found::File { meta, .. } if meta.permissions().readonly() => {
                return Err(LedgerError::ReadOnly {
                    path: path.to_owned(),
                });
            }
            Found::File { bytes, meta } => Some((bytes, meta)),
            Found::Other => {
                return Err(LedgerError::Insecure {
                    path: path.to_owned(),
                    why: "it is not a regular file: a link, a directory or a device".into(),
                });
            }
        };
        self.copies_dir()?;
        // A replaced file keeps its mode when it is rewritten, so both copies take it.
        let mode = old.as_ref().and_then(|(_, meta)| unix_mode(meta));
        let prior = match &old {
            Some((bytes, _)) => {
                let saved = self.prior_copy(id);
                write_copy(&saved, bytes, mode)?;
                Some(PriorState::SavedAt { path: saved })
            }
            None => None,
        };
        write_copy(&self.written_copy(id), contents, mode)?;
        let recorded =
            self.begin_locked(kind, prior, Some(fingerprint(contents)), at, by_version)?;
        debug_assert_eq!(recorded, id);
        let like = old.as_ref().map(|(_, meta)| meta);
        replace_file(path, contents, like).map_err(io_error(path))?;
        self.set_state_locked(id, ChangeState::Applied)?;
        Ok(id)
    }

    /// Creates the copies directory, owner-only (0700 on Unix), and returns it. A link in its
    /// place is refused, never followed or given new permissions.
    pub(crate) fn copies_dir(&self) -> Result<PathBuf, LedgerError> {
        let dir = self.state_dir.join(COPIES_DIR);
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        match builder.create(&dir) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(io_error(&dir)(e)),
        }
        let meta = fs::symlink_metadata(&dir).map_err(io_error(&dir))?;
        if !meta.is_dir() {
            return Err(LedgerError::Insecure {
                path: dir,
                why: "it is not a directory".into(),
            });
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if meta.permissions().mode() & 0o777 != 0o700 {
                fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))
                    .map_err(io_error(&dir))?;
            }
        }
        Ok(dir)
    }

    /// Deletes the copies change `id` kept, the written bytes and the replaced content, once
    /// they are no longer needed: after the change is reverted, abandoned or pruned.
    ///
    /// # Errors
    ///
    /// [`LedgerError::Io`] if a copy exists and cannot be deleted.
    pub fn delete_copies(&self, id: u32) -> Result<(), LedgerError> {
        for copy in [self.written_copy(id), self.prior_copy(id)] {
            match fs::remove_file(&copy) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(io_error(&copy)(e)),
            }
        }
        Ok(())
    }

    /// Removes reverted and abandoned changes from the ledger, with their copies and revert
    /// markers, and saves it. Returns how many were removed. Run [`script::write_revert_script`]
    /// afterwards, as after any change to the ledger.
    ///
    /// # Errors
    ///
    /// [`LedgerError::Io`] if a file cannot be deleted or the ledger cannot be saved.
    pub fn prune_closed(&mut self) -> Result<usize, LedgerError> {
        let _lock = self.lock()?;
        self.reload()?;
        let closed: Vec<u32> = self
            .file
            .changes
            .iter()
            .filter(|c| matches!(c.state, ChangeState::Reverted | ChangeState::Abandoned))
            .map(|c| c.id)
            .collect();
        for &id in &closed {
            self.delete_copies(id)?;
            let marker = self.revert_marker(id);
            match fs::remove_file(&marker) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(io_error(&marker)(e)),
            }
        }
        self.file.changes.retain(|c| !closed.contains(&c.id));
        self.save()?;
        Ok(closed.len())
    }

    /// Saves the ledger atomically: a temporary file, synced, renamed over the ledger.
    fn save(&self) -> Result<(), LedgerError> {
        fs::create_dir_all(&self.state_dir).map_err(io_error(&self.state_dir))?;
        let path = self.state_dir.join(LEDGER_FILE);
        let mut json = serde_json::to_vec_pretty(&self.file).expect("the ledger always serialises");
        json.push(b'\n');
        replace_atomically(&path, &json)
    }
}

/// The ledger in `state_dir`; a missing file is an empty ledger.
fn read_ledger(state_dir: &Path) -> Result<LedgerFile, LedgerError> {
    let path = state_dir.join(LEDGER_FILE);
    match fs::read(&path) {
        Ok(bytes) => {
            let file: LedgerFile =
                serde_json::from_slice(&bytes).map_err(|source| LedgerError::Parse {
                    path: path.clone(),
                    source,
                })?;
            if file.schema != SCHEMA {
                return Err(LedgerError::Schema { found: file.schema });
            }
            Ok(file)
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(LedgerFile {
            schema: SCHEMA,
            next_id: 1,
            changes: Vec::new(),
        }),
        Err(e) => Err(io_error(&path)(e)),
    }
}

/// The ledger's exclusive lock, held until dropped ([`HostChangeLedger::lock`]).
#[derive(Debug)]
pub(crate) struct LedgerLock {
    file: File,
}

impl Drop for LedgerLock {
    fn drop(&mut self) {
        // Closing the file releases the lock too; unlocking first says so.
        let _ = self.file.unlock();
    }
}

/// Opens the lock file, creating it owner-only. It is never removed or replaced, since a writer
/// that locked a replaced file would not exclude one that locked the new one. Anything but a
/// regular file at its path is an [`io::ErrorKind::InvalidInput`] error.
fn open_lock_file(path: &Path) -> io::Result<File> {
    if let Ok(meta) = fs::symlink_metadata(path)
        && !meta.file_type().is_file()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the ledger's lock file is not a regular file",
        ));
    }
    let mut options = fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

/// Whether `e` is Windows' sharing violation: another process holds the file open with no
/// sharing, as the PowerShell revert script holds the lock file.
fn sharing_violation(e: &io::Error) -> bool {
    cfg!(windows) && e.raw_os_error() == Some(32)
}

/// The permission bits of a file's metadata (Unix only).
fn unix_mode(meta: &fs::Metadata) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        Some(meta.permissions().mode() & 0o7777)
    }
    #[cfg(not(unix))]
    {
        let _ = meta;
        None
    }
}

/// Refuses a change whose paths do not start at a root ([`LedgerError::NotRooted`]), or whose
/// start-up line holds a line break or a NUL byte ([`LedgerError::ControlCharacter`]). A symbolic link's target may be
/// relative: it is resolved from the link, not from where a script runs.
fn check_rooted(kind: &HostChangeKind) -> Result<(), LedgerError> {
    if let HostChangeKind::PathEntry { line, .. } = kind
        && line.contains(['\n', '\r', '\0'])
    {
        return Err(LedgerError::ControlCharacter { line: line.clone() });
    }
    let path = match kind {
        HostChangeKind::FileWritten { path } | HostChangeKind::Symlink { path, .. } => path,
        HostChangeKind::PathEntry {
            shell_rc: Some(path),
            ..
        } => path,
        _ => return Ok(()),
    };
    if path.has_root() {
        Ok(())
    } else {
        Err(LedgerError::NotRooted { path: path.clone() })
    }
}

/// What is at a path, read without following a link.
pub(crate) enum Found {
    /// Nothing.
    Nothing,
    /// Something other than a regular file: a link, a directory, a device. On Unix, also a file
    /// swapped in between the check and the read.
    Other,
    /// A regular file, its bytes, and the metadata of the file the bytes came from.
    File {
        /// The content.
        bytes: Vec<u8>,
        /// The metadata, from `lstat`.
        meta: fs::Metadata,
    },
}

/// Reads `path` without following a link. On Unix the file opened is checked to be the one
/// inspected (device and inode), so a swap between the check and the read is seen as [`Found::Other`].
pub(crate) fn read_regular(path: &Path) -> io::Result<Found> {
    let meta = match fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Found::Nothing),
        Err(e) => return Err(e),
    };
    if !meta.file_type().is_file() {
        return Ok(Found::Other);
    }
    let mut file = File::open(path)?;
    if !same_file(&meta, &file.metadata()?) {
        return Ok(Found::Other);
    }
    let mut bytes = Vec::new();
    io::Read::read_to_end(&mut file, &mut bytes)?;
    Ok(Found::File { bytes, meta })
}

#[cfg(unix)]
fn same_file(a: &fs::Metadata, b: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.dev() == b.dev() && a.ino() == b.ino()
}

#[cfg(not(unix))]
fn same_file(a: &fs::Metadata, b: &fs::Metadata) -> bool {
    a.len() == b.len() && b.file_type().is_file()
}

/// Creates a new, empty file next to `path`, owner-only (0600 on Unix) from the start, and
/// returns it with its path. It is never an existing path (`create_new`), and its name is unique
/// to this process and call.
pub(crate) fn create_beside(path: &Path) -> io::Result<(File, PathBuf)> {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .map_or_else(Default::default, |n| n.to_string_lossy());
    let tmp = dir.join(format!(
        ".{name}.syntropy-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(&tmp)?;
    Ok((file, tmp))
}

/// The mode of a file Syntropy creates where none was (Unix only).
#[cfg(unix)]
const NEW_FILE_MODE: u32 = 0o644;

/// Replaces whatever is at `path` with a regular file holding `bytes`, never writing in place
/// and never following a link: a new file is made beside it ([`create_beside`]), given `like`'s
/// owner (on Unix, when the caller may) and permissions, or [`NEW_FILE_MODE`] when there is no
/// `like`, written and synced, and renamed over `path`; then the directory is synced. The owner
/// and mode are set before any byte is written, so nobody the old file kept out can read the new
/// content through the new file.
pub(crate) fn replace_file(
    path: &Path,
    bytes: &[u8],
    like: Option<&fs::Metadata>,
) -> io::Result<()> {
    let (mut file, tmp) = create_beside(path)?;
    let result = (|| {
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            match like {
                Some(meta) => {
                    // Root gives the file back its owner. Anyone else can only keep their own, and
                    // a group they are not in is not worth failing over. The owner goes first,
                    // because a change of owner clears the set-user-ID and set-group-ID bits.
                    if let Err(e) =
                        std::os::unix::fs::fchown(&file, Some(meta.uid()), Some(meta.gid()))
                        && file.metadata()?.uid() != meta.uid()
                    {
                        return Err(e);
                    }
                    file.set_permissions(meta.permissions())?;
                }
                None => file.set_permissions(fs::Permissions::from_mode(NEW_FILE_MODE))?,
            }
        }
        #[cfg(not(unix))]
        if let Some(meta) = like {
            file.set_permissions(meta.permissions())?;
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&tmp, path)?;
        sync_dir(path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Makes a rename in `path`'s directory durable: on Unix the directory is synced.
fn sync_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// The mode of a copy: owner-only, and no looser than the file it copies (Unix only). A copy can
/// hold a config or credential file, so it is never readable by anyone the original was not.
#[must_use]
pub fn copy_mode(original: Option<u32>) -> u32 {
    original.map_or(0o600, |mode| mode & 0o600)
}

/// Writes a copy of a file with [`copy_mode`], synced.
fn write_copy(path: &Path, bytes: &[u8], original: Option<u32>) -> Result<(), LedgerError> {
    let mut file = new_private_file(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Narrower still if the original was, such as 0400; never wider than 0600.
        file.set_permissions(fs::Permissions::from_mode(copy_mode(original)))
            .map_err(io_error(path))?;
    }
    #[cfg(not(unix))]
    let _ = original;
    file.write_all(bytes).map_err(io_error(path))?;
    file.sync_all().map_err(io_error(path))
}

/// Creates `path` as a new file, owner-only (0600 on Unix) before any byte is written. Whatever
/// was at `path`, a stale file or a link someone planted, is removed first rather than opened.
pub(crate) fn new_private_file(path: &Path) -> Result<File, LedgerError> {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(io_error(path)(e)),
    }
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map_err(io_error(path))
}

/// Replaces `path` with `bytes` so a reader sees the old or the new file, never a mix.
/// The new file is owner-only (0600 on Unix), as the ledger is; a caller widens it after.
pub(crate) fn replace_atomically(path: &Path, bytes: &[u8]) -> Result<(), LedgerError> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut file = new_private_file(&tmp)?;
    file.write_all(bytes).map_err(io_error(&tmp))?;
    file.sync_all().map_err(io_error(&tmp))?;
    drop(file);
    fs::rename(&tmp, path).map_err(io_error(path))?;
    // The rename is durable once the directory is synced.
    sync_dir(path).map_err(io_error(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new state directory per call. Tests run in parallel and the clock can repeat between
    /// two calls (Windows counts in 100 ns), so a counter keeps the names apart, and
    /// `create_dir` fails rather than share a directory that already exists.
    fn temp_state() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "syntropy-ledger-{}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        dir
    }

    #[test]
    fn a_missing_ledger_is_empty() {
        let state = temp_state();
        let ledger = HostChangeLedger::open(&state).unwrap();
        assert!(ledger.changes().is_empty());
        assert!(!state.join(LEDGER_FILE).exists(), "opening writes nothing");
    }

    /// The pending record is on disk before `begin` returns, so a crash during the change still
    /// leaves it in the ledger.
    #[test]
    fn begin_writes_the_pending_change_ahead() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let id = ledger
            .begin(
                HostChangeKind::LingerEnabled { user: "u".into() },
                Some(PriorState::AlreadySet),
                None,
                1_700_000_000_000,
                "0.1.0",
            )
            .unwrap();
        let on_disk = HostChangeLedger::open(&state).unwrap();
        assert_eq!(on_disk.get(id).unwrap().state, ChangeState::Pending);
        ledger.applied(id).unwrap();
        let on_disk = HostChangeLedger::open(&state).unwrap();
        assert_eq!(on_disk.get(id).unwrap().state, ChangeState::Applied);
        assert_eq!(on_disk.changes(), ledger.changes());
    }

    #[test]
    fn ids_increase_and_survive_reopening() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let a = ledger
            .begin(
                HostChangeKind::SystemUser { name: "a".into() },
                None,
                None,
                1,
                "v",
            )
            .unwrap();
        let mut reopened = HostChangeLedger::open(&state).unwrap();
        let b = reopened
            .begin(
                HostChangeKind::SystemUser { name: "b".into() },
                None,
                None,
                2,
                "v",
            )
            .unwrap();
        assert!(b > a);
    }

    #[test]
    fn write_file_saves_prior_content_and_a_copy() {
        let state = temp_state();
        let target = state.join("unit.service");
        fs::write(&target, b"old").unwrap();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let id = ledger.write_file(&target, b"new", 5, "v").unwrap();
        let change = ledger.get(id).unwrap();
        assert_eq!(change.state, ChangeState::Applied);
        assert_eq!(change.fingerprint, Some(fingerprint(b"new")));
        assert_eq!(fs::read(&target).unwrap(), b"new");
        assert_eq!(fs::read(ledger.written_copy(id)).unwrap(), b"new");
        assert_eq!(
            change.prior,
            Some(PriorState::SavedAt {
                path: ledger.prior_copy(id)
            })
        );
        assert_eq!(fs::read(ledger.prior_copy(id)).unwrap(), b"old");
    }

    /// A copy of an owner-only file is owner-only, a copy of a world-readable one is still
    /// owner-only, and the copies directory is 0700.
    /// No link is followed: not at a file Syntropy writes, not in place of the copies directory,
    /// and not in place of the ledger's temporary file.
    #[cfg(unix)]
    #[test]
    fn links_are_refused_or_replaced_never_followed() {
        use std::os::unix::fs::PermissionsExt;
        let mode = |p: &Path| fs::symlink_metadata(p).unwrap().permissions().mode() & 0o777;
        let state = temp_state();
        let elsewhere = state.join("elsewhere");
        fs::create_dir(&elsewhere).unwrap();
        fs::set_permissions(&elsewhere, fs::Permissions::from_mode(0o755)).unwrap();
        let theirs = elsewhere.join("theirs");
        fs::write(&theirs, b"theirs\n").unwrap();

        // A link at the file to write.
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let target = state.join("unit.service");
        std::os::unix::fs::symlink(&theirs, &target).unwrap();
        let err = ledger.write_file(&target, b"ours\n", 0, "v").unwrap_err();
        assert!(matches!(err, LedgerError::Insecure { .. }), "{err}");
        assert_eq!(fs::read(&theirs).unwrap(), b"theirs\n");
        assert!(ledger.changes().is_empty());
        fs::remove_file(&target).unwrap();

        // A link in place of the copies directory: refused, and its target keeps its mode.
        std::os::unix::fs::symlink(&elsewhere, state.join(COPIES_DIR)).unwrap();
        let err = ledger.write_file(&target, b"ours\n", 0, "v").unwrap_err();
        assert!(matches!(err, LedgerError::Insecure { .. }), "{err}");
        assert_eq!(mode(&elsewhere), 0o755);
        assert!(!target.exists());
        fs::remove_file(state.join(COPIES_DIR)).unwrap();

        // A link in place of the ledger's temporary file: removed, not written through.
        let mut tmp = state.join(LEDGER_FILE).into_os_string();
        tmp.push(".tmp");
        std::os::unix::fs::symlink(&theirs, &tmp).unwrap();
        ledger.write_file(&target, b"ours\n", 0, "v").unwrap();
        assert_eq!(fs::read(&theirs).unwrap(), b"theirs\n");
        assert_eq!(mode(&state.join(LEDGER_FILE)), 0o600);
        assert_eq!(fs::read(&target).unwrap(), b"ours\n");
    }

    /// Several writers at once, each with its own ledger on one state directory, as separate
    /// processes would be: every change survives, every ID is handed out once, and each change's
    /// copy holds that change's bytes.
    #[test]
    fn concurrent_writers_lose_nothing() {
        const WRITERS: u32 = 4;
        const EACH: u32 = 12;
        let state = temp_state();
        let threads: Vec<_> = (0..WRITERS)
            .map(|w| {
                let state = state.clone();
                std::thread::spawn(move || {
                    let mut ids = Vec::new();
                    for i in 0..EACH {
                        let mut ledger = HostChangeLedger::open(&state).unwrap();
                        let path = state.join(format!("w{w}-{i}.conf"));
                        let id = if i % 2 == 0 {
                            let id = ledger
                                .begin(HostChangeKind::FileWritten { path }, None, None, 0, "v")
                                .unwrap();
                            ledger.applied(id).unwrap();
                            id
                        } else {
                            ledger
                                .write_file(&path, format!("{w}-{i}\n").as_bytes(), 0, "v")
                                .unwrap()
                        };
                        ids.push((id, w, i));
                    }
                    ids
                })
            })
            .collect();
        let handed_out: Vec<(u32, u32, u32)> = threads
            .into_iter()
            .flat_map(|t| t.join().unwrap())
            .collect();
        let ledger = HostChangeLedger::open(&state).unwrap();
        assert_eq!(ledger.changes().len(), (WRITERS * EACH) as usize);
        let mut ids: Vec<u32> = handed_out.iter().map(|(id, _, _)| *id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            (WRITERS * EACH) as usize,
            "an ID was handed out twice"
        );
        for (id, w, i) in handed_out {
            let change = ledger.get(id).unwrap();
            assert_eq!(change.state, ChangeState::Applied);
            assert_eq!(
                change.kind,
                HostChangeKind::FileWritten {
                    path: state.join(format!("w{w}-{i}.conf"))
                }
            );
            if i % 2 == 1 {
                assert_eq!(
                    fs::read(ledger.written_copy(id)).unwrap(),
                    format!("{w}-{i}\n").as_bytes()
                );
            }
        }
    }

    /// A writer that cannot get the lock within its wait gets a clear error, and gets on once
    /// the lock is free.
    #[test]
    fn a_held_lock_gives_a_clear_error_not_a_hang() {
        let state = temp_state();
        let holder = HostChangeLedger::open(&state).unwrap();
        let held = holder.lock().unwrap();
        let mut ledger = HostChangeLedger::open(&state)
            .unwrap()
            .with_lock_wait(Duration::from_millis(100));
        let kind = HostChangeKind::FileWritten {
            path: state.join("a.conf"),
        };
        let started = Instant::now();
        let err = ledger.begin(kind.clone(), None, None, 0, "v").unwrap_err();
        assert!(matches!(err, LedgerError::Busy { .. }), "{err}");
        assert!(
            err.to_string()
                .contains("another Syntropy command is changing this host"),
            "{err}"
        );
        assert!(started.elapsed() < Duration::from_secs(5));
        drop(held);
        ledger.begin(kind, None, None, 0, "v").unwrap();
    }

    /// `write_file` never writes in place, so a link swapped in after its check is replaced, not
    /// written through. A thread keeps switching the target between nothing and a link to
    /// another file while `write_file` runs; that file never changes.
    #[cfg(unix)]
    #[test]
    fn a_link_swapped_in_during_write_file_is_never_written_through() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let state = temp_state();
        let victim = state.join("victim.txt");
        fs::write(&victim, b"VICTIM\n").unwrap();
        let target = state.join("a.conf");
        let stop = Arc::new(AtomicBool::new(false));
        let swapper = {
            let (stop, victim, target) = (stop.clone(), victim.clone(), target.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let _ = std::os::unix::fs::symlink(&victim, &target);
                    let _ = fs::remove_file(&target);
                }
            })
        };
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        for _ in 0..50 {
            // Refused (a link was there when it looked) or written (as a new file): either way,
            // never through the link.
            let _ = ledger.write_file(&target, b"WRITTEN\n", 0, "v");
            assert_eq!(fs::read(&victim).unwrap(), b"VICTIM\n");
        }
        stop.store(true, Ordering::Relaxed);
        swapper.join().unwrap();
        assert_eq!(fs::read(&victim).unwrap(), b"VICTIM\n");
    }

    /// The new file every replacement starts from is 0600 before anything else happens to it, so
    /// nobody else can open it and read what is written next.
    #[cfg(unix)]
    #[test]
    fn a_file_made_beside_a_target_starts_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let state = temp_state();
        let (file, tmp) = create_beside(&state.join("target.conf")).unwrap();
        assert_eq!(file.metadata().unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(tmp.parent(), Some(state.as_path()));
        // Never an existing path, and a new name each time.
        let (_, again) = create_beside(&state.join("target.conf")).unwrap();
        assert_ne!(tmp, again);
    }

    /// A new file is 0644, and a replaced one keeps its mode.
    #[cfg(unix)]
    #[test]
    fn written_files_take_the_mode_of_what_they_replace() {
        use std::os::unix::fs::PermissionsExt;
        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let fresh = state.join("fresh.service");
        ledger.write_file(&fresh, b"[Unit]\n", 0, "v").unwrap();
        assert_eq!(mode(&fresh), 0o644);
        let kept = state.join("kept.conf");
        fs::write(&kept, b"theirs\n").unwrap();
        fs::set_permissions(&kept, fs::Permissions::from_mode(0o640)).unwrap();
        ledger.write_file(&kept, b"ours\n", 0, "v").unwrap();
        assert_eq!(mode(&kept), 0o640);
        assert_eq!(fs::read(&kept).unwrap(), b"ours\n");
    }

    /// A start-up line with a line break would reach the sh script as two lines, and one with a
    /// NUL byte would lose the byte there; either way the script's pattern could match a line of
    /// the user's. Neither is ever recorded.
    #[test]
    fn a_start_up_line_with_a_line_break_or_a_nul_byte_is_refused() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        for line in [
            "export A=1\nalias ll='ls -l'",
            "export A=1\r",
            "\nexport A=1",
            // The sh script drops the NUL byte, so its pattern would be the user's `keep1`.
            "keep1\0",
            "export A=1\0keep1",
        ] {
            let kind = HostChangeKind::PathEntry {
                shell_rc: Some(state.join("profile")),
                line: line.into(),
            };
            let err = ledger.begin(kind, None, None, 0, "v").unwrap_err();
            assert!(matches!(err, LedgerError::ControlCharacter { .. }), "{err}");
            assert!(
                err.to_string().contains("line break or a NUL byte"),
                "{err}"
            );
        }
        assert!(ledger.changes().is_empty());
    }

    /// A read-only file is not replaced: its owner made it so, and an in-place write would have
    /// been refused.
    #[test]
    fn a_read_only_file_is_not_replaced() {
        let state = temp_state();
        let path = state.join("pinned.conf");
        fs::write(&path, b"theirs\n").unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&path, permissions).unwrap();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let err = ledger.write_file(&path, b"ours\n", 0, "v").unwrap_err();
        assert!(matches!(err, LedgerError::ReadOnly { .. }), "{err}");
        assert_eq!(fs::read(&path).unwrap(), b"theirs\n");
        assert!(ledger.changes().is_empty());
    }

    /// The scripts would resolve a relative path from wherever they run, so none is recorded.
    #[test]
    fn relative_paths_are_refused() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let err = ledger
            .write_file(Path::new("relative.conf"), b"x", 0, "v")
            .unwrap_err();
        assert!(matches!(err, LedgerError::NotRooted { .. }), "{err}");
        for kind in [
            HostChangeKind::Symlink {
                path: "bin/syntropy".into(),
                target: "../share/syntropy".into(),
            },
            HostChangeKind::PathEntry {
                shell_rc: Some("-profile".into()),
                line: "x".into(),
            },
        ] {
            let err = ledger.begin(kind, None, None, 0, "v").unwrap_err();
            assert!(matches!(err, LedgerError::NotRooted { .. }), "{err}");
        }
        assert!(ledger.changes().is_empty());
        // A link's target may be relative: it resolves from the link.
        ledger
            .begin(
                HostChangeKind::Symlink {
                    path: state.join("syntropy"),
                    target: "../share/syntropy".into(),
                },
                None,
                None,
                0,
                "v",
            )
            .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn copies_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();

        let secret = state.join("credentials.conf");
        fs::write(&secret, b"token=old").unwrap();
        fs::set_permissions(&secret, fs::Permissions::from_mode(0o600)).unwrap();
        let id = ledger.write_file(&secret, b"token=new", 0, "v").unwrap();
        assert_eq!(mode(&ledger.written_copy(id)), 0o600);
        assert_eq!(mode(&ledger.prior_copy(id)), 0o600);

        let public = state.join("public.conf");
        fs::write(&public, b"a").unwrap();
        fs::set_permissions(&public, fs::Permissions::from_mode(0o644)).unwrap();
        let id = ledger.write_file(&public, b"b", 0, "v").unwrap();
        assert_eq!(mode(&ledger.written_copy(id)), 0o600);
        assert_eq!(mode(&ledger.prior_copy(id)), 0o600);

        let fresh = state.join("fresh.conf");
        let id = ledger.write_file(&fresh, b"c", 0, "v").unwrap();
        assert_eq!(mode(&ledger.written_copy(id)), 0o600);

        assert_eq!(mode(&state.join(COPIES_DIR)), 0o700);
        assert_eq!(copy_mode(Some(0o400)), 0o400, "a stricter original wins");
    }

    #[test]
    fn prune_removes_closed_changes_and_their_files() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        let closed = ledger
            .write_file(&state.join("a.conf"), b"x", 0, "v")
            .unwrap();
        let open = ledger
            .write_file(&state.join("b.conf"), b"x", 0, "v")
            .unwrap();
        ledger.reverted(closed).unwrap();
        fs::write(ledger.revert_marker(closed), b"").unwrap();
        assert_eq!(ledger.prune_closed().unwrap(), 1);
        assert!(ledger.get(closed).is_none());
        assert!(!ledger.written_copy(closed).exists());
        assert!(!ledger.revert_marker(closed).exists());
        assert!(ledger.get(open).is_some());
        assert!(ledger.written_copy(open).exists());
        assert_eq!(HostChangeLedger::open(&state).unwrap().changes().len(), 1);
    }

    #[test]
    fn an_unknown_id_is_an_error() {
        let state = temp_state();
        let mut ledger = HostChangeLedger::open(&state).unwrap();
        assert!(matches!(ledger.applied(9), Err(LedgerError::UnknownId(9))));
    }

    #[test]
    fn another_schema_is_refused() {
        let state = temp_state();
        fs::write(
            state.join(LEDGER_FILE),
            br#"{"schema":99,"next_id":1,"changes":[]}"#,
        )
        .unwrap();
        assert!(matches!(
            HostChangeLedger::open(&state),
            Err(LedgerError::Schema { found: 99 })
        ));
    }
}
