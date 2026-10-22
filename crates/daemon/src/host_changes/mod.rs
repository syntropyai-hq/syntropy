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
//! **Revert markers.** The scripts cannot edit the JSON ledger, so a script that undoes a change
//! writes `<state>/host-changes/<id>.reverted`, and checks for it first, which makes a second run
//! skip what the first undid. The binary writes the same marker when it closes a change, and treats
//! an existing marker as reverted.
//!
//! **One writer at a time: assumed, not enforced (open).** A [`HostChangeLedger`] reads the
//! whole file when it is opened and rewrites the whole file on every save, through
//! `host-changes.json.tmp` and a rename. The writers above are separate processes, and `trust`
//! records a change on the client too (ADR-0505 item 6). Nothing serialises them yet. Two writers
//! open at once each hold their own copy. Then the later save drops the entries the earlier one
//! added, which leaves a change on the host with no entry for uninstall to revert. Or one writer
//! removes the other's temporary file, and that save fails. An advisory lock on the state
//! directory, held from open to the last save, would close this. It is not built here, pending
//! review of PROD-E12-T01.

pub mod revert;
pub mod script;

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use syntropy_core::Blake3;

/// The ledger file in the state directory.
pub const LEDGER_FILE: &str = "host-changes.json";

/// The directory, in the state directory, that holds the copies of written and replaced files.
pub const COPIES_DIR: &str = "host-changes";

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
}

impl HostChangeLedger {
    /// Opens the ledger in `state_dir`. A missing ledger file is an empty ledger.
    ///
    /// # Errors
    ///
    /// [`LedgerError`] if the file exists and cannot be read or parsed.
    pub fn open(state_dir: &Path) -> Result<Self, LedgerError> {
        let path = state_dir.join(LEDGER_FILE);
        let file = match fs::read(&path) {
            Ok(bytes) => {
                let file: LedgerFile =
                    serde_json::from_slice(&bytes).map_err(|source| LedgerError::Parse {
                        path: path.clone(),
                        source,
                    })?;
                if file.schema != SCHEMA {
                    return Err(LedgerError::Schema { found: file.schema });
                }
                file
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => LedgerFile {
                schema: SCHEMA,
                next_id: 1,
                changes: Vec::new(),
            },
            Err(e) => return Err(io_error(&path)(e)),
        };
        Ok(Self {
            state_dir: state_dir.to_owned(),
            file,
        })
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
    pub fn begin(
        &mut self,
        kind: HostChangeKind,
        prior: Option<PriorState>,
        fingerprint: Option<Blake3>,
        at: u64,
        by_version: &str,
    ) -> Result<u32, LedgerError> {
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
    /// # Errors
    ///
    /// [`LedgerError::Io`] if a copy, the ledger or the file cannot be written. A failure after
    /// the change was recorded leaves it pending, which a revert resolves by checking the host.
    pub fn write_file(
        &mut self,
        path: &Path,
        contents: &[u8],
        at: u64,
        by_version: &str,
    ) -> Result<u32, LedgerError> {
        let id = self.file.next_id;
        // Syntropy writes a file, never through a link someone left in its place.
        if fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(LedgerError::Insecure {
                path: path.to_owned(),
                why: "it is a symbolic link".into(),
            });
        }
        self.copies_dir()?;
        // A replaced file keeps its mode when it is rewritten, so both copies take it.
        let mode = file_mode(path);
        let prior = match fs::read(path) {
            Ok(old) => {
                let saved = self.prior_copy(id);
                write_copy(&saved, &old, mode)?;
                Some(PriorState::SavedAt { path: saved })
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(io_error(path)(e)),
        };
        write_copy(&self.written_copy(id), contents, mode)?;
        let recorded = self.begin(
            HostChangeKind::FileWritten {
                path: path.to_owned(),
            },
            prior,
            Some(fingerprint(contents)),
            at,
            by_version,
        )?;
        debug_assert_eq!(recorded, id);
        write_synced(path, contents)?;
        self.applied(id)?;
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

/// The permission bits of the file at `path`, if it exists (Unix only).
fn file_mode(path: &Path) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path)
            .ok()
            .map(|m| m.permissions().mode() & 0o777)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
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

/// Writes `bytes` to the host file at `path`, in place, and syncs it.
fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), LedgerError> {
    let mut file = File::create(path).map_err(io_error(path))?;
    file.write_all(bytes).map_err(io_error(path))?;
    file.sync_all().map_err(io_error(path))
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
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        // The rename is durable once the directory is synced.
        File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(io_error(dir))?;
    }
    Ok(())
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
