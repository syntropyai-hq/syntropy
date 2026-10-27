// SPDX-License-Identifier: MIT OR Apache-2.0

//! Starting a daemon when a command needs one and none answers (RFC-0051 §1).
//!
//! The rules:
//! - only a command that needs a daemon starts one (`run`, `chat`, `pull`); every other command
//!   reports that none is reachable, with exit 3;
//! - a remote daemon (`SYNTROPY_HOST`) is never started from here;
//! - if a service is installed (RFC-0052 §6), the CLI starts the service;
//! - otherwise it spawns `syntropy serve` detached, as the current user, with its log in the state
//!   directory, and tells the user it keeps running.
//!
//! An automatically started daemon is not tied to the CLI process, so ending a `run` session
//! does not unload a model that took minutes to load.
//!
//! [`ensure`] holds the rules. It reaches the daemon and starts one through a [`Launcher`], which
//! tests replace with a fake. [`SystemLauncher`] is the real one.

use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::{CliError, Exit};

/// Whether a command needs a daemon, and so may start one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Need {
    /// The command asks a daemon that must already run, such as `version` or `ls`.
    Running,
    /// The command needs a daemon and starts one if none answers: `run`, `chat` and `pull`.
    Start,
}

/// How a daemon was started.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Started {
    /// It was already running.
    No,
    /// The installed service was started.
    Service,
    /// `syntropy serve` was spawned, with this process ID and log file.
    Spawned {
        /// Its process ID.
        pid: u32,
        /// Where it logs.
        log: PathBuf,
    },
}

impl Started {
    /// What the CLI tells the user on stderr, if anything (RFC-0051 §1).
    #[must_use]
    pub fn notice(&self) -> Option<String> {
        match self {
            Started::No => None,
            Started::Service => Some("started the Syntropy service.".to_owned()),
            Started::Spawned { pid, .. } => Some(format!(
                "started a Syntropy daemon (pid {pid}); it keeps running so loaded models stay \
                 loaded.\nstop it with `syntropy stop`, or run it as a service with `syntropy \
                 service install`."
            )),
        }
    }
}

/// How the CLI starts a daemon.
pub trait Launcher {
    /// Whether a Syntropy service is installed for this user or system (RFC-0052 §6).
    fn service_installed(&self) -> bool;
    /// Starts the installed service.
    ///
    /// # Errors
    ///
    /// When the service manager refuses.
    fn start_service(&mut self) -> io::Result<()>;
    /// Spawns `syntropy serve` detached, logging to `log`, and returns its process ID.
    ///
    /// # Errors
    ///
    /// When the process cannot be spawned.
    fn spawn_serve(&mut self, log: &Path) -> io::Result<u32>;
    /// Waits before the next attempt to reach the new daemon.
    fn pause(&mut self, how_long: Duration);
}

/// How many times [`ensure`] tries to reach a daemon it started, and how long it waits between
/// tries \[target\]: 50 × 200 ms, a chosen budget of 10 s for a daemon to bind its listeners.
pub const START_ATTEMPTS: u32 = 50;
/// The wait between those tries \[target\].
pub const START_PAUSE: Duration = Duration::from_millis(200);

/// Reaches a daemon with `reach`, starting one by the rules above when `need` allows.
///
/// `remote` says whether `SYNTROPY_HOST` names the daemon; `state_dir` is where a spawned
/// daemon's log goes. Returns what `reach` returned and how the daemon came to run.
///
/// # Errors
///
/// `reach`'s exit-3 error when no start is allowed, or when the start fails or the new daemon
/// does not answer in time; any other error from `reach` as it is.
pub fn ensure<T>(
    need: Need,
    remote: bool,
    state_dir: Option<&Path>,
    launcher: &mut dyn Launcher,
    mut reach: impl FnMut() -> Result<T, CliError>,
) -> Result<(T, Started), CliError> {
    let unreachable = match reach() {
        Ok(client) => return Ok((client, Started::No)),
        Err(error) if error.exit == Exit::NoDaemon => error,
        Err(error) => return Err(error),
    };
    if need == Need::Running || remote {
        return Err(unreachable);
    }
    let started = if launcher.service_installed() {
        launcher.start_service().map_err(|error| {
            unreachable
                .clone()
                .detail(format!("starting the installed service failed: {error}"))
                .hint("syntropy service status")
        })?;
        Started::Service
    } else {
        let Some(state_dir) = state_dir else {
            return Err(unreachable.detail(
                "no daemon was started: there is no state directory for its log (set \
                 SYNTROPY_HOME or HOME)",
            ));
        };
        let log = state_dir.join("logs").join("daemon.log");
        let pid = launcher.spawn_serve(&log).map_err(|error| {
            unreachable
                .clone()
                .detail(format!("spawning `syntropy serve` failed: {error}"))
        })?;
        Started::Spawned { pid, log }
    };
    let mut last = unreachable;
    for _ in 0..START_ATTEMPTS {
        launcher.pause(START_PAUSE);
        match reach() {
            Ok(client) => return Ok((client, started)),
            Err(error) if error.exit == Exit::NoDaemon => last = error,
            Err(error) => return Err(error),
        }
    }
    let waited = START_PAUSE * START_ATTEMPTS;
    let mut error = last.detail(format!(
        "a daemon was started, but it did not answer within {}",
        crate::output::duration(waited)
    ));
    if let Started::Spawned { log, .. } = &started {
        error = error.detail(format!("its log is {}", log.display()));
    }
    Err(error)
}

/// The real launcher: the current executable, detached.
#[derive(Debug, Default)]
pub struct SystemLauncher;

impl Launcher for SystemLauncher {
    /// Service detection comes with the service integration of RFC-0052 §6, which records an
    /// installed service in the host-change ledger. Until then no service counts as installed.
    fn service_installed(&self) -> bool {
        false
    }

    fn start_service(&mut self) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "service management is not built yet",
        ))
    }

    fn spawn_serve(&mut self, log: &Path) -> io::Result<u32> {
        use std::process::{Command, Stdio};

        if let Some(dir) = log.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log)?;
        let mut command = Command::new(std::env::current_exe()?);
        command
            .arg("serve")
            .stdin(Stdio::null())
            .stdout(file.try_clone()?)
            .stderr(file);
        detach(&mut command);
        Ok(command.spawn()?.id())
    }

    fn pause(&mut self, how_long: Duration) {
        std::thread::sleep(how_long);
    }
}

/// Starts the child in a process group of its own, so the terminal's Ctrl+C and hang-up do not
/// reach it.
#[cfg(unix)]
fn detach(command: &mut std::process::Command) {
    use std::os::unix::process::CommandExt as _;
    command.process_group(0);
}

/// Starts the child detached from this console, in a process group of its own.
#[cfg(windows)]
fn detach(command: &mut std::process::Command) {
    use std::os::windows::process::CommandExt as _;
    /// `DETACHED_PROCESS`.
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    /// `CREATE_NEW_PROCESS_GROUP`.
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    command.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
}

#[cfg(not(any(unix, windows)))]
fn detach(_: &mut std::process::Command) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A launcher that records what it was asked, and a daemon that answers once started.
    #[derive(Default)]
    struct Fake {
        service: bool,
        service_fails: bool,
        spawn_fails: bool,
        calls: Vec<String>,
        /// How many reaches fail after a start before the daemon answers; `None`: never.
        answers_after: Option<u32>,
    }

    impl Launcher for Fake {
        fn service_installed(&self) -> bool {
            self.service
        }
        fn start_service(&mut self) -> io::Result<()> {
            self.calls.push("start_service".into());
            if self.service_fails {
                Err(io::Error::other("refused"))
            } else {
                Ok(())
            }
        }
        fn spawn_serve(&mut self, log: &Path) -> io::Result<u32> {
            self.calls.push(format!("spawn_serve {}", log.display()));
            if self.spawn_fails {
                Err(io::Error::other("no such file"))
            } else {
                Ok(4242)
            }
        }
        fn pause(&mut self, _: Duration) {
            self.calls.push("pause".into());
        }
    }

    fn no_daemon() -> CliError {
        CliError::cli(
            Exit::NoDaemon,
            "no_daemon",
            "no Syntropy daemon is reachable",
        )
    }

    /// Runs `ensure` with `fake`, where the daemon answers once started and after
    /// `fake.answers_after` failed reaches.
    fn run(need: Need, remote: bool, fake: &mut Fake) -> Result<((), Started), CliError> {
        let answers_after = fake.answers_after;
        let state = PathBuf::from("/state");
        let mut calls = 0u32;
        let mut misses = 0u32;
        let reach = move || {
            calls += 1;
            // The first reach is before any start.
            if calls == 1 {
                return Err(no_daemon());
            }
            match answers_after {
                Some(after) if misses >= after => Ok(()),
                _ => {
                    misses += 1;
                    Err(no_daemon())
                }
            }
        };
        ensure(need, remote, Some(&state), fake, reach)
    }

    #[test]
    fn a_running_daemon_is_used_and_nothing_starts() {
        let mut fake = Fake::default();
        let result = ensure(Need::Start, false, None, &mut fake, || Ok::<_, CliError>(7));
        assert_eq!(result.unwrap(), (7, Started::No));
        assert!(fake.calls.is_empty());
    }

    #[test]
    fn a_command_that_needs_a_running_daemon_never_starts_one() {
        let mut fake = Fake {
            answers_after: Some(0),
            ..Fake::default()
        };
        let error = run(Need::Running, false, &mut fake).unwrap_err();
        assert_eq!(error.exit, Exit::NoDaemon);
        assert!(fake.calls.is_empty(), "{:?}", fake.calls);
    }

    #[test]
    fn a_remote_daemon_is_never_started() {
        let mut fake = Fake {
            answers_after: Some(0),
            ..Fake::default()
        };
        let error = run(Need::Start, true, &mut fake).unwrap_err();
        assert_eq!(error.exit, Exit::NoDaemon);
        assert!(fake.calls.is_empty(), "{:?}", fake.calls);
    }

    #[test]
    fn an_installed_service_is_started_instead_of_spawning() {
        let mut fake = Fake {
            service: true,
            answers_after: Some(2),
            ..Fake::default()
        };
        let (_, started) = run(Need::Start, false, &mut fake).unwrap();
        assert_eq!(started, Started::Service);
        assert_eq!(fake.calls, ["start_service", "pause", "pause", "pause"]);
        assert_eq!(started.notice().unwrap(), "started the Syntropy service.");
    }

    #[test]
    fn without_a_service_serve_is_spawned_with_its_log_in_the_state_directory() {
        let mut fake = Fake {
            answers_after: Some(0),
            ..Fake::default()
        };
        let (_, started) = run(Need::Start, false, &mut fake).unwrap();
        let log = PathBuf::from("/state").join("logs").join("daemon.log");
        assert_eq!(
            started,
            Started::Spawned {
                pid: 4242,
                log: log.clone()
            }
        );
        assert_eq!(
            fake.calls,
            [format!("spawn_serve {}", log.display()), "pause".to_owned()]
        );
        let notice = started.notice().unwrap();
        assert!(notice.starts_with("started a Syntropy daemon (pid 4242); it keeps running"));
        assert!(notice.contains("syntropy stop") && notice.contains("syntropy service install"));
    }

    #[test]
    fn a_failed_start_or_a_silent_daemon_exits_3_with_the_reason() {
        let mut fake = Fake {
            spawn_fails: true,
            ..Fake::default()
        };
        let error = run(Need::Start, false, &mut fake).unwrap_err();
        assert_eq!(error.exit, Exit::NoDaemon);
        assert!(
            error
                .details
                .iter()
                .any(|d| d.contains("spawning `syntropy serve` failed"))
        );

        let mut fake = Fake {
            service: true,
            service_fails: true,
            ..Fake::default()
        };
        let error = run(Need::Start, false, &mut fake).unwrap_err();
        assert!(
            error
                .details
                .iter()
                .any(|d| d.contains("starting the installed service failed"))
        );
        assert_eq!(error.hints, ["syntropy service status"]);

        let mut fake = Fake {
            answers_after: None,
            ..Fake::default()
        };
        let error = run(Need::Start, false, &mut fake).unwrap_err();
        assert_eq!(error.exit, Exit::NoDaemon);
        assert_eq!(
            fake.calls.iter().filter(|c| *c == "pause").count(),
            START_ATTEMPTS as usize
        );
        assert!(
            error
                .details
                .iter()
                .any(|d| d.contains("did not answer within 10.0 s"))
        );
        assert!(error.details.iter().any(|d| d.contains("daemon.log")));
    }

    #[test]
    fn another_error_from_reach_is_returned_as_it_is() {
        let mut fake = Fake::default();
        let auth = CliError::cli(Exit::Auth, "denied", "no");
        let error =
            ensure::<()>(Need::Start, false, None, &mut fake, || Err(auth.clone())).unwrap_err();
        assert_eq!(error, auth);
        assert!(fake.calls.is_empty());
    }

    #[test]
    fn without_a_state_directory_no_daemon_is_spawned() {
        let mut fake = Fake::default();
        let error =
            ensure::<()>(Need::Start, false, None, &mut fake, || Err(no_daemon())).unwrap_err();
        assert_eq!(error.exit, Exit::NoDaemon);
        assert!(error.details[0].contains("no state directory"));
        assert!(fake.calls.is_empty());
    }
}
