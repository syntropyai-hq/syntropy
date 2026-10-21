// SPDX-License-Identifier: MIT OR Apache-2.0

//! Argument parsing for `cargo xtask`, with no dependencies beyond `std`.
//!
//! A new top-level command, such as `check-deps` (QUAL-E1-T03), is a variant of [`Command`], a
//! branch in [`parse`] and a line in [`USAGE`].

use std::fmt;

use crate::registry::{self, Entry};

pub const USAGE: &str = "\
Usage: cargo xtask <command>

Commands:
  ci <job>        run a CI job: the same command CI runs (RFC-0062 §1)
  ci --list       list the CI jobs
  test <suite>    run a test suite (RFC-0060 §1.2)
  test --list     list the test suites
  help            print this message

Exit status: 0 when every step passes, 1 when a step fails, 2 on a usage error.";

/// The top-level commands, in the order `USAGE` lists them.
const COMMANDS: &[&str] = &["ci", "test", "help"];

/// What the command line asks for. Names are resolved against the registry while parsing, so
/// every [`Command::Run`] names an entry that exists.
#[derive(Debug, PartialEq, Eq)]
pub enum Command<'r> {
    Help,
    List(Kind),
    Run { kind: Kind, entry: &'r Entry },
}

/// Which registry a name belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Job,
    Suite,
}

impl Kind {
    /// The command that runs this kind: `ci` or `test`.
    pub fn command(self) -> &'static str {
        match self {
            Kind::Job => "ci",
            Kind::Suite => "test",
        }
    }

    /// `job` or `suite`, for messages.
    pub fn noun(self) -> &'static str {
        match self {
            Kind::Job => "job",
            Kind::Suite => "suite",
        }
    }
}

/// A command line `parse` rejects. `main` prints it and exits with status 2.
#[derive(Debug, PartialEq, Eq)]
pub enum UsageError {
    NoCommand,
    UnknownCommand(String),
    MissingName {
        kind: Kind,
        valid: Vec<&'static str>,
    },
    UnknownName {
        kind: Kind,
        name: String,
        valid: Vec<&'static str>,
    },
    UnknownOption(String),
    UnexpectedArgument(String),
}

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UsageError::NoCommand => write!(
                f,
                "no command given; valid commands: {}",
                COMMANDS.join(", ")
            ),
            UsageError::UnknownCommand(command) => write!(
                f,
                "unknown command `{command}`; valid commands: {}",
                COMMANDS.join(", ")
            ),
            UsageError::MissingName { kind, valid } => write!(
                f,
                "`{}` needs a {} name; valid {}s: {}",
                kind.command(),
                kind.noun(),
                kind.noun(),
                valid.join(", ")
            ),
            UsageError::UnknownName { kind, name, valid } => write!(
                f,
                "unknown {} `{name}`; valid {}s: {}",
                kind.noun(),
                kind.noun(),
                valid.join(", ")
            ),
            UsageError::UnknownOption(option) => write!(f, "unknown option `{option}`"),
            UsageError::UnexpectedArgument(arg) => write!(f, "unexpected argument `{arg}`"),
        }
    }
}

impl std::error::Error for UsageError {}

/// Parses the arguments after `cargo xtask`, resolving job names against `jobs` and suite names
/// against `suites`.
pub fn parse<'r>(
    args: &[String],
    jobs: &'r [Entry],
    suites: &'r [Entry],
) -> Result<Command<'r>, UsageError> {
    let Some((command, rest)) = args.split_first() else {
        return Err(UsageError::NoCommand);
    };
    match command.as_str() {
        "help" | "-h" | "--help" => {
            no_more(rest)?;
            Ok(Command::Help)
        }
        "ci" => parse_run(Kind::Job, rest, jobs),
        "test" => parse_run(Kind::Suite, rest, suites),
        other if other.starts_with('-') => Err(UsageError::UnknownOption(other.to_owned())),
        other => Err(UsageError::UnknownCommand(other.to_owned())),
    }
}

/// `ci <job>`, `ci --list`, `test <suite>` and `test --list`.
fn parse_run<'r>(
    kind: Kind,
    args: &[String],
    entries: &'r [Entry],
) -> Result<Command<'r>, UsageError> {
    let Some((first, rest)) = args.split_first() else {
        return Err(UsageError::MissingName {
            kind,
            valid: registry::names(entries),
        });
    };
    match first.as_str() {
        "-h" | "--help" => {
            no_more(rest)?;
            Ok(Command::Help)
        }
        "--list" => {
            no_more(rest)?;
            Ok(Command::List(kind))
        }
        option if option.starts_with('-') => Err(UsageError::UnknownOption(option.to_owned())),
        name => {
            let entry = registry::find(entries, name).ok_or_else(|| UsageError::UnknownName {
                kind,
                name: name.to_owned(),
                valid: registry::names(entries),
            })?;
            no_more(rest)?;
            Ok(Command::Run { kind, entry })
        }
    }
}

fn no_more(rest: &[String]) -> Result<(), UsageError> {
    match rest.first() {
        None => Ok(()),
        Some(arg) => Err(UsageError::UnexpectedArgument(arg.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Action, JOBS, SUITES, Step};

    const STEP: &[Step] = &[Step {
        name: "noop",
        action: Action::Cargo(&["--version"]),
    }];

    /// A registry with more than one name, so error messages can be checked for the full list.
    const FIXTURE_JOBS: &[Entry] = &[
        Entry {
            name: "alpha",
            summary: "first",
            steps: STEP,
            pending: &[],
        },
        Entry {
            name: "beta-two",
            summary: "second",
            steps: STEP,
            pending: &[],
        },
    ];

    fn args(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_owned).collect()
    }

    fn parse_real(line: &str) -> Result<Command<'static>, UsageError> {
        parse(&args(line), JOBS, SUITES)
    }

    fn parse_fixture(line: &str) -> Result<Command<'static>, UsageError> {
        parse(&args(line), FIXTURE_JOBS, &[])
    }

    #[test]
    fn ci_lint_resolves_to_the_lint_job() {
        let lint = registry::find(JOBS, "lint").unwrap();
        assert_eq!(
            parse_real("ci lint"),
            Ok(Command::Run {
                kind: Kind::Job,
                entry: lint
            })
        );
    }

    #[test]
    fn test_unit_resolves_to_the_unit_suite() {
        let unit = registry::find(SUITES, "unit").unwrap();
        assert_eq!(
            parse_real("test unit"),
            Ok(Command::Run {
                kind: Kind::Suite,
                entry: unit
            })
        );
    }

    #[test]
    fn a_job_name_is_not_a_suite_name() {
        assert!(matches!(
            parse_real("test lint"),
            Err(UsageError::UnknownName {
                kind: Kind::Suite,
                ..
            })
        ));
        assert!(matches!(
            parse_real("ci unit"),
            Err(UsageError::UnknownName {
                kind: Kind::Job,
                ..
            })
        ));
    }

    #[test]
    fn unknown_job_lists_every_valid_name() {
        let err = parse_fixture("ci nosuchjob").unwrap_err();
        assert_eq!(
            err,
            UsageError::UnknownName {
                kind: Kind::Job,
                name: "nosuchjob".into(),
                valid: vec!["alpha", "beta-two"],
            }
        );
        assert_eq!(
            err.to_string(),
            "unknown job `nosuchjob`; valid jobs: alpha, beta-two"
        );
    }

    #[test]
    fn unknown_job_in_the_real_registry_lists_its_names() {
        let message = parse_real("ci nosuchjob").unwrap_err().to_string();
        for name in registry::names(JOBS) {
            assert!(message.contains(name), "`{message}` lacks `{name}`");
        }
    }

    #[test]
    fn unknown_suite_lists_every_valid_name() {
        let message = parse_real("test nosuchsuite").unwrap_err().to_string();
        assert!(message.starts_with("unknown suite `nosuchsuite`; valid suites: "));
        for name in registry::names(SUITES) {
            assert!(message.contains(name), "`{message}` lacks `{name}`");
        }
    }

    #[test]
    fn a_missing_name_lists_every_valid_name() {
        assert_eq!(
            parse_fixture("ci").unwrap_err().to_string(),
            "`ci` needs a job name; valid jobs: alpha, beta-two"
        );
        assert!(matches!(
            parse_real("test"),
            Err(UsageError::MissingName {
                kind: Kind::Suite,
                ..
            })
        ));
    }

    #[test]
    fn names_match_exactly() {
        assert!(parse_fixture("ci alph").is_err());
        assert!(parse_fixture("ci ALPHA").is_err());
        assert!(parse_fixture("ci beta-two").is_ok());
    }

    #[test]
    fn list_and_help() {
        assert_eq!(parse_real("ci --list"), Ok(Command::List(Kind::Job)));
        assert_eq!(parse_real("test --list"), Ok(Command::List(Kind::Suite)));
        for line in ["help", "-h", "--help", "ci --help", "test -h"] {
            assert_eq!(parse_real(line), Ok(Command::Help), "`{line}`");
        }
    }

    #[test]
    fn no_command_and_unknown_command() {
        assert_eq!(parse_real(""), Err(UsageError::NoCommand));
        assert_eq!(
            parse_real("frob"),
            Err(UsageError::UnknownCommand("frob".into()))
        );
        assert_eq!(
            parse_real("frob").unwrap_err().to_string(),
            "unknown command `frob`; valid commands: ci, test, help"
        );
        assert_eq!(
            parse_real("--frob"),
            Err(UsageError::UnknownOption("--frob".into()))
        );
    }

    #[test]
    fn options_are_not_taken_for_names() {
        assert_eq!(
            parse_real("ci --lint"),
            Err(UsageError::UnknownOption("--lint".into()))
        );
        assert_eq!(
            parse_real("test -x"),
            Err(UsageError::UnknownOption("-x".into()))
        );
    }

    #[test]
    fn extra_arguments_are_rejected() {
        for (line, extra) in [
            ("ci lint extra", "extra"),
            ("ci lint --fix", "--fix"),
            ("ci --list lint", "lint"),
            ("test unit unit", "unit"),
            ("help ci", "ci"),
        ] {
            assert_eq!(
                parse_real(line),
                Err(UsageError::UnexpectedArgument(extra.into())),
                "`{line}`"
            );
        }
    }

    #[test]
    fn an_unknown_name_is_reported_before_extra_arguments() {
        assert!(matches!(
            parse_real("ci nosuchjob extra"),
            Err(UsageError::UnknownName { .. })
        ));
    }

    #[test]
    fn usage_names_every_command() {
        for command in COMMANDS {
            assert!(
                USAGE.contains(&format!("  {command}")),
                "USAGE lacks `{command}`"
            );
        }
    }
}
