// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where records go (RFC-0053 §4.2), and which sinks a daemon uses where.
//!
//! Each sink does two things, kept apart so the canary test can read exactly what a sink is
//! handed: [`SinkPlan::render`] turns a record into the bytes or line the sink writes, a pure
//! function, and the sink then delivers them. stderr, the rotating file and journald's socket
//! are delivered with the standard library. Unified logging on macOS and the Windows event log
//! go through `syntropy_io::os::log_sink`, the one crate that links those facilities.

#[cfg(target_os = "macos")]
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use syntropy_config::logging::{LogFile, LogFormat, LoggingSection};
use syntropy_obs::log::Level;

use super::event::LogEvent;
use super::format::{self, WallClock};

/// The rotating file's size before it rotates: 16 MiB \[target\] (RFC-0053 §4.2).
pub const FILE_BYTES: u64 = 16 << 20;

/// Files the rotation keeps: `daemon.log` and four older ones \[target\].
pub const FILE_COUNT: usize = 5;

/// journald's socket for the native protocol.
pub const JOURNAL_SOCKET: &str = "/run/systemd/journal/socket";

/// The event source name in the Windows Application log (RFC-0053 §4.2).
pub const EVENT_SOURCE: &str = "Syntropy";

/// How the daemon runs, as the sink choice reads it (RFC-0053 §4.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Situation {
    /// `syntropy serve` in a terminal or under another process.
    #[default]
    Foreground,
    /// A systemd service: `JOURNAL_STREAM` is set.
    Systemd,
    /// A launchd agent or daemon.
    Launchd,
    /// A Windows service or logon task.
    WindowsService,
}

impl Situation {
    /// The situation the environment shows: `JOURNAL_STREAM` for systemd, `XPC_SERVICE_NAME`
    /// (set by launchd, `0` outside it) for launchd. A Windows service is told by its own start
    /// path, so it is never inferred here.
    pub fn from_env(var: impl Fn(&str) -> Option<String>) -> Situation {
        if var("JOURNAL_STREAM").is_some_and(|v| !v.is_empty()) {
            Situation::Systemd
        } else if var("XPC_SERVICE_NAME").is_some_and(|v| !v.is_empty() && v != "0") {
            Situation::Launchd
        } else {
            Situation::Foreground
        }
    }
}

/// One sink a daemon writes to, before it is opened.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SinkPlan {
    /// stderr, as text (with colour on a terminal) or JSON Lines.
    Stderr {
        /// JSON Lines rather than text.
        json: bool,
        /// ANSI colour on the level.
        colour: bool,
    },
    /// The rotating file, as text or JSON Lines.
    File {
        /// `daemon.log`'s path.
        path: PathBuf,
        /// JSON Lines rather than text.
        json: bool,
    },
    /// journald, structured.
    Journald,
    /// Unified logging: the subsystem, one category per crate.
    UnifiedLog {
        /// The app bundle identifier (RFC-0052 §9).
        subsystem: String,
    },
    /// The Windows event log, warnings and errors only.
    EventLog,
}

/// The sinks for `situation` and `section` (RFC-0053 §4.2's table). `state_dir` holds
/// `logs/daemon.log`; `stderr_terminal` says whether stderr is a terminal.
pub fn plan(
    situation: Situation,
    section: &LoggingSection,
    state_dir: &Path,
    stderr_terminal: bool,
    subsystem: &str,
) -> Vec<SinkPlan> {
    let json = section.format == LogFormat::Json;
    let auto_file = state_dir.join("logs").join("daemon.log");
    // A file the user named is kept everywhere; `auto` keeps one only where the table does.
    let file = |auto: bool| match &section.file {
        LogFile::Path(path) => Some(SinkPlan::File {
            path: path.clone(),
            json,
        }),
        LogFile::Auto if auto => Some(SinkPlan::File {
            path: auto_file.clone(),
            json,
        }),
        LogFile::Auto | LogFile::Off => None,
    };
    let mut sinks = match situation {
        Situation::Foreground => vec![SinkPlan::Stderr {
            json,
            colour: !json && section.format == LogFormat::Auto && stderr_terminal,
        }],
        Situation::Systemd => vec![SinkPlan::Journald],
        Situation::Launchd => vec![SinkPlan::UnifiedLog {
            subsystem: subsystem.to_owned(),
        }],
        Situation::WindowsService => vec![SinkPlan::EventLog],
    };
    let auto = matches!(situation, Situation::Launchd | Situation::WindowsService);
    sinks.extend(file(auto));
    sinks
}

/// What a sink is handed for one record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rendered {
    /// One line, with its newline, for stderr and the file.
    Line(String),
    /// journald's native fields.
    Journal(Vec<u8>),
    /// One line for unified logging, in a category.
    UnifiedLog {
        /// The category: the record's crate.
        category: String,
        /// The line, without a timestamp: the OS stamps it.
        line: String,
    },
    /// One line for the event log.
    EventLog(String),
    /// Nothing: the sink does not take this record.
    Skip,
}

/// An open sink.
pub enum Sink {
    /// stderr, or a writer standing in for it.
    Stderr {
        /// Where the lines go.
        out: Box<dyn Write + Send>,
        /// JSON Lines rather than text.
        json: bool,
        /// ANSI colour.
        colour: bool,
    },
    /// The rotating file.
    File(RotatingFile),
    /// journald's socket, or a test's.
    #[cfg(unix)]
    Journald {
        /// The socket path.
        socket: PathBuf,
    },
    /// Unified logging, one log object per category.
    #[cfg(target_os = "macos")]
    UnifiedLog {
        /// The subsystem.
        subsystem: String,
        /// The log objects made so far.
        logs: BTreeMap<String, syntropy_io::os::log_sink::OsLog>,
    },
    /// The Windows event log.
    #[cfg(windows)]
    EventLog(syntropy_io::os::log_sink::EventLog),
    /// A sink that only records what its plan renders, for tests and for a facility this OS
    /// lacks.
    Memory {
        /// The sink it stands for.
        plan: SinkPlan,
        /// What it was handed.
        kept: Vec<Rendered>,
    },
}

impl std::fmt::Debug for Sink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Sink::Stderr { .. } => "Sink::Stderr",
            Sink::File(_) => "Sink::File",
            #[cfg(unix)]
            Sink::Journald { .. } => "Sink::Journald",
            #[cfg(target_os = "macos")]
            Sink::UnifiedLog { .. } => "Sink::UnifiedLog",
            #[cfg(windows)]
            Sink::EventLog(_) => "Sink::EventLog",
            Sink::Memory { .. } => "Sink::Memory",
        })
    }
}

/// The OS level a record goes under in unified logging and the event log.
#[cfg(any(target_os = "macos", windows))]
fn severity(level: Level) -> syntropy_io::os::log_sink::Severity {
    use syntropy_io::os::log_sink::Severity;
    match level {
        Level::Error => Severity::Error,
        Level::Warn => Severity::Warning,
        Level::Info => Severity::Info,
        Level::Debug | Level::Trace => Severity::Debug,
    }
}

/// The record's line without stamp or colour: what unified logging and the event log take.
fn bare(event: &LogEvent, wall: &WallClock) -> String {
    format::text(event, wall, false, false)
}

impl SinkPlan {
    /// What this sink would be handed for `event`: the pure half of writing it.
    pub fn render(&self, event: &LogEvent, wall: &WallClock) -> Rendered {
        match self {
            SinkPlan::Stderr { json, colour } => Rendered::Line(line(event, wall, *json, *colour)),
            SinkPlan::File { json, .. } => Rendered::Line(line(event, wall, *json, false)),
            SinkPlan::Journald => Rendered::Journal(format::journal(event, wall)),
            SinkPlan::UnifiedLog { .. } => Rendered::UnifiedLog {
                category: event.crate_name().to_owned(),
                line: bare(event, wall),
            },
            SinkPlan::EventLog => {
                if event.level <= Level::Warn {
                    Rendered::EventLog(bare(event, wall))
                } else {
                    Rendered::Skip
                }
            }
        }
    }

    /// Opens the sink. A facility this OS lacks opens as [`Sink::Memory`], which keeps what it
    /// would have been handed, so the plan stays testable everywhere.
    ///
    /// # Errors
    ///
    /// The OS error when the file cannot be opened or the event source registered.
    pub fn open(&self) -> io::Result<Sink> {
        Ok(match self {
            SinkPlan::Stderr { json, colour } => Sink::Stderr {
                out: Box::new(io::stderr()),
                json: *json,
                colour: *colour,
            },
            SinkPlan::File { path, json } => Sink::File(RotatingFile::open(
                path.clone(),
                *json,
                FILE_BYTES,
                FILE_COUNT,
            )?),
            #[cfg(unix)]
            SinkPlan::Journald => Sink::Journald {
                socket: PathBuf::from(JOURNAL_SOCKET),
            },
            #[cfg(target_os = "macos")]
            SinkPlan::UnifiedLog { subsystem } => Sink::UnifiedLog {
                subsystem: subsystem.clone(),
                logs: BTreeMap::new(),
            },
            #[cfg(windows)]
            SinkPlan::EventLog => Sink::EventLog(
                syntropy_io::os::log_sink::EventLog::register(EVENT_SOURCE)
                    .map_err(|error| io::Error::other(error.to_string()))?,
            ),
            #[allow(unreachable_patterns)]
            plan => Sink::Memory {
                plan: plan.clone(),
                kept: Vec::new(),
            },
        })
    }
}

/// A text or JSON line, with its newline.
fn line(event: &LogEvent, wall: &WallClock, json: bool, colour: bool) -> String {
    let mut line = if json {
        format::json(event, wall)
    } else {
        format::text(event, wall, true, colour)
    };
    line.push('\n');
    line
}

impl Sink {
    /// Writes `event`. A sink that fails drops the record: logging never fails the node.
    pub fn write(&mut self, event: &LogEvent, wall: &WallClock) {
        match self {
            Sink::Stderr { out, json, colour } => {
                let _ = out.write_all(line(event, wall, *json, *colour).as_bytes());
            }
            Sink::File(file) => {
                let _ = file.write(&line(event, wall, file.json, false));
            }
            #[cfg(unix)]
            Sink::Journald { socket } => {
                let _ = send_journal(socket, &format::journal(event, wall));
            }
            #[cfg(target_os = "macos")]
            Sink::UnifiedLog { subsystem, logs } => {
                let category = event.crate_name().to_owned();
                if !logs.contains_key(&category)
                    && let Ok(log) = syntropy_io::os::log_sink::OsLog::new(subsystem, &category)
                {
                    logs.insert(category.clone(), log);
                }
                if let Some(log) = logs.get(&category) {
                    log.write(severity(event.level), &bare(event, wall));
                }
            }
            #[cfg(windows)]
            Sink::EventLog(log) => {
                if event.level <= Level::Warn {
                    let _ = log.report(severity(event.level), &bare(event, wall));
                }
            }
            Sink::Memory { plan, kept } => kept.push(plan.render(event, wall)),
        }
    }

    /// Flushes buffered output.
    pub fn flush(&mut self) {
        match self {
            Sink::Stderr { out, .. } => {
                let _ = out.flush();
            }
            Sink::File(file) => {
                let _ = file.file.flush();
            }
            _ => {}
        }
    }
}

/// Creates `dir` and any missing parents, mode 0700 on Unix whatever the umask allows more of.
fn private_dir(dir: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir)
}

/// Opens `path` for appending: created mode 0600 on Unix, and narrowed to 0600 if it existed.
fn open_private(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let file = options.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    Ok(file)
}

/// Sends one datagram of journald's native protocol to `socket`.
///
/// # Errors
///
/// The OS error, such as a datagram too large for the socket, in which case the record is lost.
#[cfg(unix)]
pub fn send_journal(socket: &Path, payload: &[u8]) -> io::Result<()> {
    let sender = std::os::unix::net::UnixDatagram::unbound()?;
    sender.send_to(payload, socket).map(|_| ())
}

/// `daemon.log` and its older copies, rotated by size (RFC-0053 §4.2: 16 MiB × 5).
#[derive(Debug)]
pub struct RotatingFile {
    path: PathBuf,
    file: File,
    written: u64,
    max_bytes: u64,
    count: usize,
    json: bool,
}

impl RotatingFile {
    /// Opens `path` for appending, creating its directory, rotating at `max_bytes` and keeping
    /// `count` files in all. Records can hold what a user would not share, so on Unix a directory
    /// this creates is mode 0700, and every log file is 0600, an existing one narrowed to it. A
    /// directory that already exists, such as one a user named in `logging.file`, is left as it
    /// is.
    ///
    /// # Errors
    ///
    /// The OS error when the directory or the file cannot be made.
    pub fn open(
        path: PathBuf,
        json: bool,
        max_bytes: u64,
        count: usize,
    ) -> io::Result<RotatingFile> {
        if let Some(dir) = path.parent() {
            private_dir(dir)?;
        }
        let file = open_private(&path)?;
        let written = file.metadata()?.len();
        Ok(RotatingFile {
            path,
            file,
            written,
            max_bytes,
            count: count.max(1),
            json,
        })
    }

    /// The file `n` rotations old: `daemon.log` for 0, `daemon.log.1` for 1, and so on.
    pub fn nth(&self, n: usize) -> PathBuf {
        if n == 0 {
            self.path.clone()
        } else {
            let mut name = self.path.as_os_str().to_owned();
            name.push(format!(".{n}"));
            PathBuf::from(name)
        }
    }

    /// Appends `line`, rotating first when it would take the file past its size.
    ///
    /// # Errors
    ///
    /// The OS error.
    pub fn write(&mut self, line: &str) -> io::Result<()> {
        let bytes = line.len() as u64;
        if self.written > 0 && self.written + bytes > self.max_bytes {
            self.rotate()?;
        }
        self.file.write_all(line.as_bytes())?;
        self.written += bytes;
        Ok(())
    }

    fn rotate(&mut self) -> io::Result<()> {
        self.file.flush()?;
        let oldest = self.nth(self.count - 1);
        if oldest.exists() {
            fs::remove_file(&oldest)?;
        }
        for n in (0..self.count - 1).rev() {
            let from = self.nth(n);
            if from.exists() {
                fs::rename(&from, self.nth(n + 1))?;
            }
        }
        self.file = open_private(&self.path)?;
        self.written = 0;
        Ok(())
    }
}
