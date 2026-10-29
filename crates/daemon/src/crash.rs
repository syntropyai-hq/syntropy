// SPDX-License-Identifier: MIT OR Apache-2.0

//! Crash records in the state directory (RFC-0053 §6; PROD-E9-T04).
//!
//! A panic in the daemon or a runner, and a runner's exit by signal or with a sticky device error
//! (RFC-0001 §3.4), write one [`CrashRecord`] to `<state>/crashes/`: the build ID, the panic
//! message and backtrace or the signal, the last [`TAIL_LINES`] log lines, and the device error
//! report if any. The [`CRASHES_KEPT`] most recent are kept. `doctor` shows them (RUN-001,
//! PROD-E9-T03, through [`CrashStore::list`]) and the diagnostic bundle includes them
//! ([`crate::bundle`]). Nothing sends them anywhere.
//!
//! - **Files.** One JSON object per record, `<unix ms, 20 digits>-<pid>-<n>.json`, so names sort
//!   oldest first. The directory and each file are owner-only (`syntropy_io::os::private_dir`,
//!   `create_private_file`), as the rest of the state directory is (RFC-0030 §2.1), and each file
//!   and the directory are synced.
//! - **Log lines carry no content.** [`LogTail`] keeps the last lines as the logger formats them,
//!   with every content field removed whatever `logging.include_content` says, so a crash record
//!   never holds a prompt, an output or token IDs.
//! - **A panic** is recorded by the hook [`install_panic_hook`] sets, which then runs the hook it
//!   replaced. **A runner's exit** is recorded by the supervisor's shell from the process's exit
//!   status: the process host does so when it reaps a runner it was given a [`CrashSink`] for
//!   ([`ProcessRunner::record_crashes`](crate::runner_host::ProcessRunner::record_crashes)).

use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use serde::{Deserialize, Serialize};
use syntropy_core::Clock;

use crate::logging::LogEvent;
use crate::logging::format::{self, WallClock};
use crate::supervisor::KillReason;

/// The directory under the state directory that holds the records.
pub const CRASH_DIR: &str = "crashes";

/// How many records are kept: the most recent 20.
pub const CRASHES_KEPT: usize = 20;

/// How many log lines a record holds: the last 200.
pub const TAIL_LINES: usize = 200;

/// The record format's version.
pub const CRASH_RECORD_VERSION: u16 = 1;

/// One crash.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrashRecord {
    /// [`CRASH_RECORD_VERSION`].
    pub version: u16,
    /// When, in milliseconds since the Unix epoch, by the daemon's `Clock`.
    pub at_unix_ms: u64,
    /// The build's 32-byte ID, as hex (ADR-0015 decision 1).
    pub build_id: String,
    /// The process that crashed.
    pub process: CrashProcess,
    /// What happened.
    pub cause: CrashCause,
    /// The last log lines before it, oldest first, with no content.
    pub last_logs: Vec<String>,
}

/// The process a crash record is about.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CrashProcess {
    /// The daemon.
    Daemon {
        /// Its process ID.
        pid: u32,
    },
    /// A runner.
    Runner {
        /// Its process ID.
        pid: u32,
        /// Which runner, as the supervisor names it (its instance and stage, for example).
        runner: String,
    },
}

/// What a crash was.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CrashCause {
    /// A panic.
    Panic {
        /// The panic's message.
        message: String,
        /// Where it panicked, `file:line:column`, if known.
        location: Option<String>,
        /// The backtrace.
        backtrace: String,
    },
    /// The process was killed by a signal (Unix).
    Signal {
        /// The signal's number.
        signal: i32,
        /// Its name, such as `SIGKILL`, when this build knows it.
        name: Option<String>,
    },
    /// The process ended with an exception status (Windows: an `NTSTATUS` error code, such as
    /// `0xC0000005` for an access violation).
    Exception {
        /// The status code.
        code: u32,
    },
    /// The runner reported a sticky device error before it exited (RFC-0001 §3.4).
    DeviceError {
        /// The device error report.
        report: String,
    },
    /// The process host killed the runner, for a reason RFC-0001 §3.4 gives (PROD-E9-T12). On
    /// every OS: a host kill is recorded from the host's own reason, never from how the exit
    /// status reads, since on Windows `TerminateProcess` leaves an ordinary exit code.
    HostKilled {
        /// Why the host killed it.
        reason: HostKill,
        /// The signal the exit status shows, on Unix.
        signal: Option<i32>,
    },
}

/// Why the process host killed a runner (RFC-0001 §3.4): the host's own type. These are the
/// only kills the host makes: a shutdown drains runners and never kills them, and dropping a
/// runner's host closes its channel without a kill. In a record it reads
/// `{"supervisor": "hung"}`, `{"supervisor": "stalled"}` or `"refused"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostKill {
    /// The supervisor's kill, with its reason: a hang or a stall. The supervisor shell's
    /// `Effects::kill` passes it to `ProcessRunner::kill`, which DIST-E1-T16 wires.
    Supervisor(KillReason),
    /// The freshly spawned runner's `Hello` was refused: another build, protocol or process.
    Refused,
}

impl From<KillReason> for HostKill {
    fn from(reason: KillReason) -> HostKill {
        HostKill::Supervisor(reason)
    }
}

impl CrashCause {
    /// The cause a process's exit status shows: a signal on Unix, an exception status on
    /// Windows, or `None` for an exit that is not a crash (a normal exit, with any code).
    #[must_use]
    pub fn from_exit(status: &ExitStatus) -> Option<CrashCause> {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt as _;
            status.signal().map(|signal| CrashCause::Signal {
                signal,
                name: signal_name(signal).map(str::to_owned),
            })
        }
        #[cfg(windows)]
        {
            // A system NTSTATUS error: severity bits 31–30 both set (error), and the customer
            // bit 29 and the reserved bit 28 clear. `exit(-1)` is `0xFFFF_FFFF`, which sets both
            // of those, so a program's own negative exit code is not taken for an exception.
            status
                .code()
                .map(i32::cast_unsigned)
                .filter(|&code| code & 0xF000_0000 == 0xC000_0000)
                .map(|code| CrashCause::Exception { code })
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = status;
            None
        }
    }
}

/// The names of the signals a crash is likely to name, for this OS.
fn signal_name(signal: i32) -> Option<&'static str> {
    let names: &[(i32, &str)] = if cfg!(target_os = "linux") || cfg!(target_os = "android") {
        &[
            (1, "SIGHUP"),
            (2, "SIGINT"),
            (3, "SIGQUIT"),
            (4, "SIGILL"),
            (5, "SIGTRAP"),
            (6, "SIGABRT"),
            (7, "SIGBUS"),
            (8, "SIGFPE"),
            (9, "SIGKILL"),
            (11, "SIGSEGV"),
            (13, "SIGPIPE"),
            (14, "SIGALRM"),
            (15, "SIGTERM"),
        ]
    } else if cfg!(any(target_os = "macos", target_os = "ios")) {
        &[
            (1, "SIGHUP"),
            (2, "SIGINT"),
            (3, "SIGQUIT"),
            (4, "SIGILL"),
            (5, "SIGTRAP"),
            (6, "SIGABRT"),
            (8, "SIGFPE"),
            (9, "SIGKILL"),
            (10, "SIGBUS"),
            (11, "SIGSEGV"),
            (13, "SIGPIPE"),
            (14, "SIGALRM"),
            (15, "SIGTERM"),
        ]
    } else {
        &[]
    };
    names
        .iter()
        .find(|(n, _)| *n == signal)
        .map(|(_, name)| *name)
}

/// The last [`TAIL_LINES`] log lines, as text, with no content. The logger feeds it every record
/// it writes (`Logger::tail`).
#[derive(Debug, Default)]
pub struct LogTail {
    lines: Mutex<VecDeque<String>>,
}

impl LogTail {
    /// An empty tail.
    #[must_use]
    pub fn new() -> LogTail {
        LogTail::default()
    }

    /// Adds `event`, formatted as text with its timestamp and every content field removed.
    pub fn push(&self, event: &LogEvent, wall: &WallClock) {
        let mut event = event.clone();
        event.fields.retain(|field| !field.content);
        if let Some(count) = event.key_fields {
            event.key_fields = Some(count.min(event.fields.len()));
        }
        let line = format::text(&event, wall, true, false);
        let mut lines = self.lines.lock().unwrap_or_else(PoisonError::into_inner);
        if lines.len() == TAIL_LINES {
            lines.pop_front();
        }
        lines.push_back(line);
    }

    /// The lines held, oldest first.
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        self.lines
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .cloned()
            .collect()
    }
}

/// The crash records of one state directory.
#[derive(Debug)]
pub struct CrashStore {
    dir: PathBuf,
    next: AtomicU64,
}

impl CrashStore {
    /// The store of `<state_dir>/crashes/`. Nothing is created until a record is written.
    #[must_use]
    pub fn new(state_dir: &Path) -> CrashStore {
        CrashStore {
            dir: state_dir.join(CRASH_DIR),
            next: AtomicU64::new(0),
        }
    }

    /// The directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Writes `record` as a new owner-only file, syncs it and the directory, then removes all but
    /// the [`CRASHES_KEPT`] most recent records. Returns the new file's path.
    ///
    /// # Errors
    ///
    /// The OS error when the directory or the file cannot be created or synced. A record that
    /// cannot be pruned is left in place.
    pub fn write(&self, record: &CrashRecord) -> io::Result<PathBuf> {
        syntropy_io::os::private_dir(&self.dir)?;
        let bytes = serde_json::to_vec_pretty(record).map_err(io::Error::other)?;
        let pid = match record.process {
            CrashProcess::Daemon { pid } | CrashProcess::Runner { pid, .. } => pid,
        };
        let path = loop {
            let n = self.next.fetch_add(1, Ordering::Relaxed);
            let path = self
                .dir
                .join(format!("{:020}-{pid}-{n}.json", record.at_unix_ms));
            match syntropy_io::os::create_private_file(&path, &bytes) {
                Ok(()) => break path,
                // A record of the same millisecond from an earlier start of this process.
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e),
            }
        };
        syntropy_io::os::sync_dir(&self.dir)?;
        self.prune();
        Ok(path)
    }

    /// The record files, oldest first.
    fn files(&self) -> io::Result<Vec<PathBuf>> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut files: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        files.sort();
        Ok(files)
    }

    fn prune(&self) {
        let Ok(files) = self.files() else { return };
        let excess = files.len().saturating_sub(CRASHES_KEPT);
        for path in &files[..excess] {
            let _ = fs::remove_file(path);
        }
        if excess > 0 {
            let _ = syntropy_io::os::sync_dir(&self.dir);
        }
    }

    /// The records held, newest first, with their file names. A file that cannot be read or
    /// parsed is left out and named in [`CrashListing::unreadable`].
    ///
    /// # Errors
    ///
    /// The OS error when the directory cannot be listed.
    pub fn list(&self) -> io::Result<CrashListing> {
        let mut records = Vec::new();
        let mut unreadable = Vec::new();
        for path in self.files()?.into_iter().rev() {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            match fs::read(&path)
                .ok()
                .and_then(|b| serde_json::from_slice::<CrashRecord>(&b).ok())
            {
                Some(record) => records.push((name, record)),
                None => unreadable.push(name),
            }
        }
        Ok(CrashListing {
            records,
            unreadable,
        })
    }
}

/// What [`CrashStore::list`] found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CrashListing {
    /// The records, newest first, each with its file name.
    pub records: Vec<(String, CrashRecord)>,
    /// The files that could not be read or parsed, by name.
    pub unreadable: Vec<String>,
}

/// What a runner host needs to record a runner's exit by signal: the store, the build, the
/// runner's name, the time and the log lines a record carries. Given to
/// [`ProcessRunner::record_crashes`](crate::runner_host::ProcessRunner::record_crashes), which
/// calls [`record_exit`](Self::record_exit) when it reaps the runner.
pub struct CrashSink {
    /// Where records go.
    pub store: CrashStore,
    /// The build's ID, as hex.
    pub build_id: String,
    /// Which runner, as the supervisor names it.
    pub runner: String,
    /// The time.
    pub clock: Arc<dyn Clock>,
    /// The log lines a record carries.
    pub tail: Arc<LogTail>,
}

impl std::fmt::Debug for CrashSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrashSink")
            .field("store", &self.store)
            .field("runner", &self.runner)
            .finish_non_exhaustive()
    }
}

impl CrashSink {
    /// Writes the record of runner process `pid`, which exited with `status`, when that exit is
    /// a crash ([`CrashCause::from_exit`]). Returns the record's path, or `None` for an exit that
    /// is not one.
    ///
    /// # Errors
    ///
    /// As [`CrashStore::write`].
    pub fn record_exit(&self, pid: u32, status: &ExitStatus) -> io::Result<Option<PathBuf>> {
        match CrashCause::from_exit(status) {
            Some(cause) => self.record(pid, cause).map(Some),
            None => Ok(None),
        }
    }

    /// Writes the record of runner process `pid`, which the process host killed for `reason`
    /// and which then exited with `status`. Always a record, on every OS (PROD-E9-T12).
    ///
    /// # Errors
    ///
    /// As [`CrashStore::write`].
    pub fn record_host_kill(
        &self,
        pid: u32,
        reason: HostKill,
        status: &ExitStatus,
    ) -> io::Result<PathBuf> {
        let signal = match CrashCause::from_exit(status) {
            Some(CrashCause::Signal { signal, .. }) => Some(signal),
            _ => None,
        };
        self.record(pid, CrashCause::HostKilled { reason, signal })
    }

    fn record(&self, pid: u32, cause: CrashCause) -> io::Result<PathBuf> {
        let record = CrashRecord {
            version: CRASH_RECORD_VERSION,
            at_unix_ms: self.clock.unix_ms(),
            build_id: self.build_id.clone(),
            process: CrashProcess::Runner {
                pid,
                runner: self.runner.clone(),
            },
            cause,
            last_logs: self.tail.lines(),
        };
        self.store.write(&record)
    }
}

/// What the panic hook needs to write a record.
pub struct PanicRecorder {
    /// Where records go.
    pub store: CrashStore,
    /// The build's ID, as hex.
    pub build_id: String,
    /// The process, as records name it.
    pub process: CrashProcess,
    /// The time.
    pub clock: Arc<dyn Clock>,
    /// The log lines a record carries.
    pub tail: Arc<LogTail>,
}

impl std::fmt::Debug for PanicRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PanicRecorder")
            .field("store", &self.store)
            .field("process", &self.process)
            .finish_non_exhaustive()
    }
}

impl PanicRecorder {
    /// The record of the panic `info` describes, with a backtrace captured now.
    #[must_use]
    pub fn record(&self, info: &std::panic::PanicHookInfo<'_>) -> CrashRecord {
        let payload = info.payload();
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a panic with a payload that is not text".to_owned());
        CrashRecord {
            version: CRASH_RECORD_VERSION,
            at_unix_ms: self.clock.unix_ms(),
            build_id: self.build_id.clone(),
            process: self.process.clone(),
            cause: CrashCause::Panic {
                message,
                location: info
                    .location()
                    .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())),
                backtrace: std::backtrace::Backtrace::force_capture().to_string(),
            },
            last_logs: self.tail.lines(),
        }
    }
}

/// Sets a panic hook that writes a crash record through `recorder`, then runs the hook it
/// replaced, so the panic still prints. A record that cannot be written is lost; the panic goes
/// on regardless.
pub fn install_panic_hook(recorder: PanicRecorder) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = recorder.store.write(&recorder.record(info));
        previous(info);
    }));
}

#[cfg(test)]
pub(crate) mod tests;
