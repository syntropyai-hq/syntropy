// SPDX-License-Identifier: MIT OR Apache-2.0

//! Argument parsing for `cargo xtask`, with no dependencies beyond `std`.
//!
//! A top-level command, such as `check-deps` (QUAL-E1-T03), is a variant of [`Command`], a
//! branch in [`parse`] and a line in [`USAGE`]. A job's or suite's own options are declared in
//! its registry entry (`Entry::params`), and [`parse`] accepts exactly those.

use std::fmt;

use crate::registry::{self, Entry};

pub const USAGE: &str = "\
Usage: cargo xtask <command>

Commands:
  ci <job> [options]      run a CI job: the same command CI runs (RFC-0062 §1)
  ci --list               list the CI jobs and their options
  test <suite> [options]  run a test suite (RFC-0060 §1.2)
  test --list             list the test suites and their options
  check-deps [--manifest-path <path>]
                          check the crate dependency rules of RFC-0001 §4.2 (RFC-0062 §12.1)
                          on this workspace, or on the workspace of <path>; `lint` runs it
  help                    print this message

An option is `--name <value>` or `--name=<value>`, given at most once.

Exit status: 0 when every step passes, 1 when a step fails, 2 on a usage error or when xtask
was not started by `cargo xtask`.";

/// The top-level commands, in the order `USAGE` lists them.
const COMMANDS: &[&str] = &["ci", "test", "check-deps", "help"];

/// What the command line asks for. Names and options are resolved against the registry while
/// parsing, so every [`Command::Run`] names an entry that exists, with options it accepts.
#[derive(Debug, PartialEq, Eq)]
pub enum Command<'r> {
    Help,
    List(Kind),
    Run {
        kind: Kind,
        entry: &'r Entry,
        args: Args,
    },
    /// `check-deps`, on the workspace of `manifest_path` when given, else on this checkout.
    CheckDeps {
        manifest_path: Option<String>,
    },
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

/// The option values given to an entry, in command-line order. Each name is one of the entry's
/// `params`, and appears at most once.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Args(Vec<(&'static str, String)>);

impl Args {
    /// The value given for `name`, such as `--commits`.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(given, _)| *given == name)
            .map(|(_, value)| value.as_str())
    }
}

impl fmt::Display for Args {
    /// ` --name value` for each option, so a label reads like the command line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (name, value) in &self.0 {
            write!(f, " {name} {value}")?;
        }
        Ok(())
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
    /// An option before the job or suite name, or after `xtask`, that nothing accepts.
    UnknownOption(String),
    /// An option the entry does not declare.
    UnknownParam {
        kind: Kind,
        entry: &'static str,
        option: String,
        valid: Vec<&'static str>,
    },
    MissingValue(&'static str),
    RepeatedOption(&'static str),
    /// A required option left out.
    MissingOption {
        kind: Kind,
        entry: &'static str,
        option: &'static str,
    },
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
            UsageError::UnknownParam {
                kind,
                entry,
                option,
                valid,
            } if valid.is_empty() => write!(
                f,
                "{} `{entry}` takes no options, but got `{option}`",
                kind.noun()
            ),
            UsageError::UnknownParam {
                kind,
                entry,
                option,
                valid,
            } => write!(
                f,
                "unknown option `{option}` for {} `{entry}`; valid options: {}",
                kind.noun(),
                valid.join(", ")
            ),
            UsageError::MissingValue(option) => write!(f, "option `{option}` needs a value"),
            UsageError::MissingOption {
                kind,
                entry,
                option,
            } => write!(f, "{} `{entry}` needs `{option}`", kind.noun()),
            UsageError::RepeatedOption(option) => {
                write!(f, "option `{option}` is given more than once")
            }
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
        "check-deps" => parse_check_deps(rest),
        other if other.starts_with('-') => Err(UsageError::UnknownOption(other.to_owned())),
        other => Err(UsageError::UnknownCommand(other.to_owned())),
    }
}

/// `ci <job> [options]`, `ci --list`, `test <suite> [options]` and `test --list`.
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
            let args = parse_params(kind, entry, rest)?;
            Ok(Command::Run { kind, entry, args })
        }
    }
}

/// The options after an entry's name, checked against the entry's `params`.
fn parse_params(kind: Kind, entry: &Entry, args: &[String]) -> Result<Args, UsageError> {
    let mut values: Vec<(&'static str, String)> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        if !arg.starts_with('-') {
            return Err(UsageError::UnexpectedArgument(arg.clone()));
        }
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (arg.as_str(), None),
        };
        let param = entry
            .params
            .iter()
            .find(|param| param.name == name)
            .ok_or_else(|| UsageError::UnknownParam {
                kind,
                entry: entry.name,
                option: name.to_owned(),
                valid: entry.params.iter().map(|param| param.name).collect(),
            })?;
        let value = match inline {
            Some(value) => value,
            // A following option is not taken as a value: `--base --other` is a missing value.
            None => match rest.next() {
                Some(value) if !value.starts_with('-') => value.as_str(),
                _ => "",
            },
        };
        if value.is_empty() {
            return Err(UsageError::MissingValue(param.name));
        }
        if values.iter().any(|(seen, _)| *seen == param.name) {
            return Err(UsageError::RepeatedOption(param.name));
        }
        values.push((param.name, value.to_owned()));
    }
    if let Some(missing) = entry
        .params
        .iter()
        .find(|param| param.required && !values.iter().any(|(seen, _)| *seen == param.name))
    {
        return Err(UsageError::MissingOption {
            kind,
            entry: entry.name,
            option: missing.name,
        });
    }
    Ok(Args(values))
}

/// `check-deps [--manifest-path <path>]` and `check-deps --help`.
fn parse_check_deps(args: &[String]) -> Result<Command<'static>, UsageError> {
    const MANIFEST_PATH: &str = "--manifest-path";
    if let Some((first, rest)) = args.split_first()
        && matches!(first.as_str(), "-h" | "--help")
    {
        no_more(rest)?;
        return Ok(Command::Help);
    }
    let mut manifest_path = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (arg.as_str(), None),
        };
        if name != MANIFEST_PATH {
            return Err(if arg.starts_with('-') {
                UsageError::UnknownOption(name.to_owned())
            } else {
                UsageError::UnexpectedArgument(arg.clone())
            });
        }
        let value = match inline {
            Some(value) => value,
            None => match rest.next() {
                Some(value) if !value.starts_with('-') => value.as_str(),
                _ => "",
            },
        };
        if value.is_empty() {
            return Err(UsageError::MissingValue(MANIFEST_PATH));
        }
        if manifest_path.replace(value.to_owned()).is_some() {
            return Err(UsageError::RepeatedOption(MANIFEST_PATH));
        }
    }
    Ok(Command::CheckDeps { manifest_path })
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
    use crate::registry::{Action, JOBS, Param, SUITES, Step};

    const STEP: &[Step] = &[Step {
        name: "noop",
        action: Action::Cargo(&["--version"]),
    }];

    /// A registry with more than one name, so error messages can be checked for the full list,
    /// and with an entry that takes options.
    const FIXTURE_JOBS: &[Entry] = &[
        Entry {
            name: "alpha",
            summary: "first",
            defined_in: "RFC-0062 §2.3",
            params: &[],
            steps: STEP,
            pending: &[],
        },
        Entry {
            name: "beta-two",
            summary: "second, with options",
            defined_in: "RFC-0062 §2.2",
            params: &[
                Param {
                    name: "--base",
                    value: "<ref>",
                    help: "the merge base",
                    required: false,
                },
                Param {
                    name: "--head",
                    value: "<ref>",
                    help: "the head",
                    required: false,
                },
            ],
            steps: STEP,
            pending: &[],
        },
        Entry {
            name: "gamma",
            summary: "third, with a required option",
            defined_in: "RFC-0062 §8.3",
            params: &[Param {
                name: "--input",
                value: "<file>",
                help: "the input",
                required: true,
            }],
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

    /// The option values of a fixture command line that parses.
    fn fixture_args(line: &str) -> Vec<(&'static str, String)> {
        match parse_fixture(line) {
            Ok(Command::Run { args, .. }) => args.0,
            other => panic!("`{line}` gave {other:?}"),
        }
    }

    #[test]
    fn ci_lint_resolves_to_the_lint_job() {
        let lint = registry::find(JOBS, "lint").unwrap();
        assert_eq!(
            parse_real("ci lint"),
            Ok(Command::Run {
                kind: Kind::Job,
                entry: lint,
                args: Args::default(),
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
                entry: unit,
                args: Args::default(),
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
                valid: vec!["alpha", "beta-two", "gamma"],
            }
        );
        assert_eq!(
            err.to_string(),
            "unknown job `nosuchjob`; valid jobs: alpha, beta-two, gamma"
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
            "`ci` needs a job name; valid jobs: alpha, beta-two, gamma"
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
            "unknown command `frob`; valid commands: ci, test, check-deps, help"
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
    fn an_entry_takes_its_own_options_in_either_form() {
        assert_eq!(
            fixture_args("ci beta-two --base main"),
            [("--base", "main".to_owned())]
        );
        assert_eq!(
            fixture_args("ci beta-two --base=main"),
            [("--base", "main".to_owned())]
        );
        assert_eq!(
            fixture_args("ci beta-two --head abc123 --base=main"),
            [
                ("--head", "abc123".to_owned()),
                ("--base", "main".to_owned())
            ]
        );
        assert_eq!(fixture_args("ci beta-two"), []);
    }

    #[test]
    fn options_print_as_typed() {
        let Ok(Command::Run { args, .. }) = parse_fixture("ci beta-two --base=main --head x")
        else {
            panic!("did not parse");
        };
        assert_eq!(args.to_string(), " --base main --head x");
        assert_eq!(Args::default().to_string(), "");
    }

    #[test]
    fn an_option_another_entry_declares_is_rejected() {
        assert_eq!(
            parse_fixture("ci alpha --base main")
                .unwrap_err()
                .to_string(),
            "job `alpha` takes no options, but got `--base`"
        );
        assert_eq!(
            parse_fixture("ci beta-two --tip main")
                .unwrap_err()
                .to_string(),
            "unknown option `--tip` for job `beta-two`; valid options: --base, --head"
        );
        assert_eq!(
            parse_fixture("ci beta-two --tip=main"),
            Err(UsageError::UnknownParam {
                kind: Kind::Job,
                entry: "beta-two",
                option: "--tip".into(),
                valid: vec!["--base", "--head"],
            })
        );
    }

    #[test]
    fn an_option_needs_a_value() {
        for line in [
            "ci beta-two --base",
            "ci beta-two --base=",
            "ci beta-two --base --head x",
        ] {
            assert_eq!(
                parse_fixture(line),
                Err(UsageError::MissingValue("--base")),
                "`{line}`"
            );
        }
        assert_eq!(
            UsageError::MissingValue("--base").to_string(),
            "option `--base` needs a value"
        );
    }

    #[test]
    fn a_required_option_must_be_given() {
        assert_eq!(
            parse_fixture("ci gamma"),
            Err(UsageError::MissingOption {
                kind: Kind::Job,
                entry: "gamma",
                option: "--input",
            })
        );
        assert_eq!(
            parse_fixture("ci gamma").unwrap_err().to_string(),
            "job `gamma` needs `--input`"
        );
        assert_eq!(
            fixture_args("ci gamma --input x"),
            [("--input", "x".to_owned())]
        );
    }

    #[test]
    fn get_returns_a_given_value() {
        let Ok(Command::Run { args, .. }) = parse_fixture("ci beta-two --head h --base=b") else {
            panic!("did not parse");
        };
        assert_eq!(args.get("--base"), Some("b"));
        assert_eq!(args.get("--head"), Some("h"));
        assert_eq!(args.get("--tip"), None);
    }

    #[test]
    fn the_contribution_jobs_need_their_inputs() {
        assert_eq!(
            parse_real("ci dco --event e").unwrap_err().to_string(),
            "job `dco` needs `--commits`"
        );
        assert_eq!(
            parse_real("ci pr-title --event e --commits c")
                .unwrap_err()
                .to_string(),
            "job `pr-title` needs `--label-events`"
        );
        assert_eq!(
            parse_real("ci pr-title --event e --commits c --label-events l")
                .unwrap_err()
                .to_string(),
            "job `pr-title` needs `--roster`"
        );
        assert!(
            parse_real("ci pr-title --event e --commits c --label-events l --roster r").is_ok()
        );
    }

    #[test]
    fn an_option_is_given_at_most_once() {
        assert_eq!(
            parse_fixture("ci beta-two --base a --base=b"),
            Err(UsageError::RepeatedOption("--base"))
        );
    }

    #[test]
    fn extra_arguments_are_rejected() {
        for (line, extra) in [
            ("ci lint extra", "extra"),
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
        assert_eq!(
            parse_fixture("ci beta-two --base main extra"),
            Err(UsageError::UnexpectedArgument("extra".into()))
        );
    }

    #[test]
    fn lint_takes_no_options() {
        assert!(matches!(
            parse_real("ci lint --fix"),
            Err(UsageError::UnknownParam { entry: "lint", .. })
        ));
    }

    #[test]
    fn an_unknown_name_is_reported_before_extra_arguments() {
        assert!(matches!(
            parse_real("ci nosuchjob extra"),
            Err(UsageError::UnknownName { .. })
        ));
    }

    #[test]
    fn check_deps_takes_an_optional_manifest_path() {
        assert_eq!(
            parse_real("check-deps"),
            Ok(Command::CheckDeps {
                manifest_path: None
            })
        );
        for line in [
            "check-deps --manifest-path a/Cargo.toml",
            "check-deps --manifest-path=a/Cargo.toml",
        ] {
            assert_eq!(
                parse_real(line),
                Ok(Command::CheckDeps {
                    manifest_path: Some("a/Cargo.toml".into())
                }),
                "`{line}`"
            );
        }
        assert_eq!(parse_real("check-deps --help"), Ok(Command::Help));
    }

    #[test]
    fn check_deps_rejects_anything_else() {
        for (line, error) in [
            (
                "check-deps --fix",
                UsageError::UnknownOption("--fix".into()),
            ),
            (
                "check-deps --fix=yes",
                UsageError::UnknownOption("--fix".into()),
            ),
            (
                "check-deps Cargo.toml",
                UsageError::UnexpectedArgument("Cargo.toml".into()),
            ),
            (
                "check-deps --manifest-path",
                UsageError::MissingValue("--manifest-path"),
            ),
            (
                "check-deps --manifest-path=",
                UsageError::MissingValue("--manifest-path"),
            ),
            (
                "check-deps --manifest-path --fix",
                UsageError::MissingValue("--manifest-path"),
            ),
            (
                "check-deps --manifest-path a --manifest-path=b",
                UsageError::RepeatedOption("--manifest-path"),
            ),
            (
                "check-deps --help extra",
                UsageError::UnexpectedArgument("extra".into()),
            ),
        ] {
            assert_eq!(parse_real(line), Err(error), "`{line}`");
        }
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
