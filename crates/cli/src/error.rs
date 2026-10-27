// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exit codes and error rendering (RFC-0051 §2.2).
//!
//! An error prints as `error[<domain>/<code>]: <message>`, then indented detail lines, then
//! `hint:` lines with the exact command that fixes it when one exists. An error from the daemon
//! keeps the API's code under the `api` domain. An error the CLI finds itself, such as no daemon
//! reachable, uses the `cli` domain and a named code. The API's numbered code registry is
//! PROD-E2-T05's.

use std::fmt;
use std::process::ExitCode;

use syntropy_core::error::ErrorClass;

/// The exit codes of RFC-0051 §2.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Exit {
    /// 0: success.
    Success = 0,
    /// 1: other failure (`Internal`, `Device`, `Link`, `Correctness`, and every class the table
    /// leaves out).
    Failure = 1,
    /// 2: a usage error, or a question needed an answer and stdin is not a terminal.
    Usage = 2,
    /// 3: no daemon reachable, and none could be started.
    NoDaemon = 3,
    /// 4: not found (model, instance, session, node).
    NotFound = 4,
    /// 5: invalid request or configuration (`InvalidRequest`, `Unsupported`).
    Invalid = 5,
    /// 6: does not fit, or no capacity (`Capacity`).
    Capacity = 6,
    /// 7: authentication or permission (`Auth`).
    Auth = 7,
    /// 8: integrity failure (`Integrity`).
    Integrity = 8,
    /// 9: timed out (`Timeout`).
    Timeout = 9,
    /// 10: partial success: some nodes failed in a cluster-wide command.
    Partial = 10,
    /// 130: interrupted (Ctrl+C, `Cancelled`).
    Interrupted = 130,
}

impl Exit {
    /// The code for an error of `class`. `not_found` marks an `InvalidRequest` whose code is a
    /// not-found code, which exits 4.
    #[must_use]
    pub fn for_class(class: ErrorClass, not_found: bool) -> Exit {
        match class {
            ErrorClass::InvalidRequest if not_found => Exit::NotFound,
            ErrorClass::InvalidRequest | ErrorClass::Unsupported => Exit::Invalid,
            ErrorClass::Capacity => Exit::Capacity,
            ErrorClass::Auth => Exit::Auth,
            ErrorClass::Integrity => Exit::Integrity,
            ErrorClass::Timeout => Exit::Timeout,
            ErrorClass::Cancelled => Exit::Interrupted,
            // `Internal`, `Device`, `Link`, `Correctness`, and those the table leaves out:
            // `Transient` and `Drift`, which the daemon retries or handles itself.
            _ => Exit::Failure,
        }
    }

    /// The process exit code.
    #[must_use]
    pub fn code(self) -> u8 {
        self as u8
    }
}

impl From<Exit> for ExitCode {
    fn from(exit: Exit) -> ExitCode {
        ExitCode::from(exit.code())
    }
}

/// An error a command returns, ready to render.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CliError {
    /// The exit code.
    pub exit: Exit,
    /// The domain: `api` for the daemon's errors, `cli` for the CLI's own.
    pub domain: &'static str,
    /// The code within the domain.
    pub code: String,
    /// The message for people.
    pub message: String,
    /// Lines that explain the message.
    pub details: Vec<String>,
    /// Commands that fix it.
    pub hints: Vec<String>,
}

impl CliError {
    /// An error the CLI finds itself, in the `cli` domain.
    pub fn cli(exit: Exit, code: &str, message: impl Into<String>) -> CliError {
        CliError {
            exit,
            domain: "cli",
            code: code.to_owned(),
            message: message.into(),
            details: Vec::new(),
            hints: Vec::new(),
        }
    }

    /// An error from the daemon's native error body. Its code and message are made safe to
    /// print on a terminal ([`crate::output::sanitize`]).
    pub fn api(status: u16, error: &syntropy_api::routes::NativeError) -> CliError {
        let not_found = status == 404 || error.code.ends_with("not_found");
        CliError {
            exit: Exit::for_class(error.class, not_found),
            domain: "api",
            code: crate::output::sanitize(&error.code).into_owned(),
            message: crate::output::sanitize(&error.message).into_owned(),
            details: Vec::new(),
            hints: Vec::new(),
        }
    }

    /// Adds a detail line.
    #[must_use]
    pub fn detail(mut self, line: impl Into<String>) -> CliError {
        self.details.push(line.into());
        self
    }

    /// Adds a hint: the command that fixes the error.
    #[must_use]
    pub fn hint(mut self, command: impl Into<String>) -> CliError {
        self.hints.push(command.into());
        self
    }

    /// The error for a command its task has not built yet: `Unsupported`, exit 5.
    pub fn not_built(command: &str) -> CliError {
        CliError::cli(
            Exit::Invalid,
            "not_built",
            format!("`syntropy {command}` is not available in this build yet"),
        )
    }
}

/// Renders the error as RFC-0051 §2.2 writes it, without colour.
impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}/{}]: {}", self.domain, self.code, self.message)?;
        for line in &self.details {
            write!(f, "\n  {line}")?;
        }
        for hint in &self.hints {
            write!(f, "\n  hint: {hint}")?;
        }
        Ok(())
    }
}

impl std::error::Error for CliError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every class of RFC-0001 §5.12 and its code in RFC-0051 §2.2's table.
    #[test]
    fn each_error_class_has_its_exit_code() {
        let table = [
            (ErrorClass::InvalidRequest, 5),
            (ErrorClass::Unsupported, 5),
            (ErrorClass::Capacity, 6),
            (ErrorClass::Transient, 1),
            (ErrorClass::Link, 1),
            (ErrorClass::Device, 1),
            (ErrorClass::Integrity, 8),
            (ErrorClass::Correctness, 1),
            (ErrorClass::Drift, 1),
            (ErrorClass::Timeout, 9),
            (ErrorClass::Auth, 7),
            (ErrorClass::Cancelled, 130),
            (ErrorClass::Internal, 1),
        ];
        for (class, code) in table {
            assert_eq!(Exit::for_class(class, false).code(), code, "{class:?}");
        }
        assert_eq!(
            Exit::for_class(ErrorClass::InvalidRequest, true),
            Exit::NotFound
        );
        assert_eq!(Exit::for_class(ErrorClass::Capacity, true), Exit::Capacity);
    }

    #[test]
    fn the_codes_without_a_class_are_the_table_s() {
        assert_eq!(
            [Exit::Success, Exit::Usage, Exit::NoDaemon, Exit::Partial].map(Exit::code),
            [0, 2, 3, 10]
        );
    }

    #[test]
    fn an_error_renders_with_its_details_and_hints() {
        let error = CliError::cli(
            Exit::NoDaemon,
            "no_daemon",
            "no Syntropy daemon is reachable",
        )
        .detail("tried the local socket")
        .hint("syntropy serve");
        assert_eq!(
            error.to_string(),
            "error[cli/no_daemon]: no Syntropy daemon is reachable\n  tried the local socket\n  \
             hint: syntropy serve"
        );
    }

    #[test]
    fn an_api_error_keeps_its_code_and_maps_its_class() {
        let native = |class, code: &str| syntropy_api::routes::NativeError {
            class,
            code: code.to_owned(),
            message: "m".to_owned(),
        };
        let error = CliError::api(
            404,
            &native(ErrorClass::Unsupported, "unsupported_endpoint"),
        );
        assert_eq!(
            (error.domain, error.code.as_str()),
            ("api", "unsupported_endpoint")
        );
        assert_eq!(error.exit, Exit::Invalid);
        let missing = CliError::api(404, &native(ErrorClass::InvalidRequest, "model_not_found"));
        assert_eq!(missing.exit, Exit::NotFound);
        let auth = CliError::api(401, &native(ErrorClass::Auth, "local_auth_required"));
        assert_eq!(auth.exit, Exit::Auth);
        // A daemon's control characters never reach the terminal.
        let mut hostile = native(ErrorClass::Internal, "c\u{1b}[2J");
        hostile.message = "m\u{1b}]0;owned\u{7}".to_owned();
        let error = CliError::api(500, &hostile);
        assert!(
            !error.to_string().chars().any(char::is_control),
            "{error:?}"
        );
    }
}
