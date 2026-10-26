// SPDX-License-Identifier: MIT OR Apache-2.0

//! Argument parsing for `cargo xtask`, with no dependencies beyond `std`.
//!
//! A top-level command, such as `check-deps` (QUAL-E1-T03), is a variant of [`Command`], a
//! branch in [`parse`] and a line in [`USAGE`]. A job's or suite's own options are declared in
//! its registry entry (`Entry::params`), and [`parse`] accepts exactly those. The licence
//! commands (QUAL-E1-T04) declare theirs to [`parse_options`].

use std::fmt;

use crate::build::{self, Profile};
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
  build --profile <p> [--manifest-path <path>]
                          build in profile dev, test-release, chaos or release with its test
                          features, then check the build against the profile (RFC-0062 §4.4)
  deny [--manifest-path <path>] [--advisories]
                          run cargo-deny with deny.toml (ADR-0018 §3, RFC-0062 §11.2): the
                          licence, ban and source checks, or with --advisories the RustSec
                          advisories, which need network access; `lint` runs both
  spdx [--root <dir>]     check the SPDX header of every source file (ADR-0018 §1); `lint`
                          runs it
  third-party [--root <dir>]
                          check that third_party/licences.toml lists every vendored file
                          (ADR-0018 §3, RFC-0062 §12.2); `lint` runs it
  notices [--manifest-path <path>] [--out <file>]
                          generate THIRD-PARTY-NOTICES for the release build (ADR-0018 §6) and
                          write it to <file>; without --out, only check it can be generated,
                          which is what `lint` runs
  instructions --show <path> [--ref <rev>]
                          print one file ci/agent-instructions.toml lists, and refuse any other
                          (RFC-0062 §12.3); with --ref, the list and the file come from <rev>
  instructions --list [--ref <rev>]
                          print the files agents load as instructions or configuration
  fixtures fetch|import|verify [--manifest <file>] --store <dir> [options]
                          the fixture store (RFC-0060 §12.3): `fetch` [--name <fixture>]
                          --cache <dir> --cap <bytes> copies verified objects into a capped
                          cache; `import` --name <fixture> --file <file> --licence-file <file>
                          publishes a file and fills its `blake3` and `bytes`; `verify`
                          [--name <fixture>] checks the store against the manifest
  help                    print this message

An option is `--name <value>` or `--name=<value>`, given at most once.

Exit status: 0 when every step passes, 1 when a step fails, 2 on a usage error or when xtask
was not started by `cargo xtask`.";

/// The top-level commands, in the order `USAGE` lists them.
const COMMANDS: &[&str] = &[
    "ci",
    "test",
    "check-deps",
    "build",
    "deny",
    "spdx",
    "third-party",
    "notices",
    "instructions",
    "fixtures",
    "help",
];

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
    /// `build`, in `profile`, of the workspace of `manifest_path` when given, else of this
    /// checkout.
    Build {
        profile: &'static Profile,
        manifest_path: Option<String>,
    },
    /// `deny`: cargo-deny's licence, ban and source checks, or its advisories.
    Deny {
        manifest_path: Option<String>,
        advisories: bool,
    },
    /// `spdx`: the SPDX headers under `root`, else under this checkout.
    Spdx {
        root: Option<String>,
    },
    /// `third-party`: the vendored-file manifest under `root`, else under this checkout.
    ThirdParty {
        root: Option<String>,
    },
    /// `notices`: generate `THIRD-PARTY-NOTICES`, writing it to `out` when given.
    Notices {
        manifest_path: Option<String>,
        out: Option<String>,
    },
    /// `instructions`: one listed file when `show` is set, else the list, read from `rev` when
    /// given, else from the working tree.
    Instructions {
        show: Option<String>,
        rev: Option<String>,
    },
    /// `fixtures`: the fixture store client (QUAL-E7-T05).
    Fixtures {
        action: crate::fixtures::Action,
        args: crate::fixtures::Args,
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
    /// `build` without `--profile`.
    MissingProfile {
        valid: Vec<&'static str>,
    },
    /// `build --profile` with a name no profile has.
    UnknownProfile {
        name: String,
        valid: Vec<&'static str>,
    },
    /// A command that needs exactly one of its options.
    OneOf {
        command: &'static str,
        options: &'static [&'static str],
    },
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
            UsageError::MissingProfile { valid } => write!(
                f,
                "`build` needs `--profile <p>`; valid profiles: {}",
                valid.join(", ")
            ),
            UsageError::UnknownProfile { name, valid } => write!(
                f,
                "unknown profile `{name}`; valid profiles: {}",
                valid.join(", ")
            ),
            UsageError::OneOf { command, options } => write!(
                f,
                "`{command}` needs exactly one of {}",
                options.join(" or ")
            ),
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
        "build" => parse_build(rest),
        "deny" => parse_options(rest, &["--manifest-path"], &["--advisories"]).map(|o| {
            o.or_help(|o| Command::Deny {
                manifest_path: o.value("--manifest-path"),
                advisories: o.flag("--advisories"),
            })
        }),
        "spdx" => parse_options(rest, &["--root"], &[]).map(|o| {
            o.or_help(|o| Command::Spdx {
                root: o.value("--root"),
            })
        }),
        "third-party" => parse_options(rest, &["--root"], &[]).map(|o| {
            o.or_help(|o| Command::ThirdParty {
                root: o.value("--root"),
            })
        }),
        "notices" => parse_options(rest, &["--manifest-path", "--out"], &[]).map(|o| {
            o.or_help(|o| Command::Notices {
                manifest_path: o.value("--manifest-path"),
                out: o.value("--out"),
            })
        }),
        "instructions" => {
            let o = parse_options(rest, &["--show", "--ref"], &["--list"])?;
            if o.help {
                return Ok(Command::Help);
            }
            let show = o.value("--show");
            if show.is_some() == o.flag("--list") {
                return Err(UsageError::OneOf {
                    command: "instructions",
                    options: &["--show", "--list"],
                });
            }
            Ok(Command::Instructions {
                show,
                rev: o.value("--ref"),
            })
        }
        "fixtures" => parse_fixtures(rest),
        other if other.starts_with('-') => Err(UsageError::UnknownOption(other.to_owned())),
        other => Err(UsageError::UnknownCommand(other.to_owned())),
    }
}

/// `fixtures <fetch|import|verify> [options]`.
fn parse_fixtures<'r>(args: &[String]) -> Result<Command<'r>, UsageError> {
    use crate::fixtures::{Action, Args};
    let Some((word, rest)) = args.split_first() else {
        return Err(UsageError::OneOf {
            command: "fixtures",
            options: &["fetch", "import", "verify"],
        });
    };
    if matches!(word.as_str(), "-h" | "--help") {
        no_more(rest)?;
        return Ok(Command::Help);
    }
    let Some(action) = Action::parse(word) else {
        return Err(UsageError::OneOf {
            command: "fixtures",
            options: &["fetch", "import", "verify"],
        });
    };
    let o = parse_options(
        rest,
        &[
            "--manifest",
            "--store",
            "--cache",
            "--cap",
            "--name",
            "--file",
            "--licence-file",
        ],
        &[],
    )?;
    if o.help {
        return Ok(Command::Help);
    }
    Ok(Command::Fixtures {
        action,
        args: Args {
            manifest: o.value("--manifest"),
            store: o.value("--store"),
            cache: o.value("--cache"),
            cap: o.value("--cap"),
            name: o.value("--name"),
            file: o.value("--file"),
            licence_file: o.value("--licence-file"),
        },
    })
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

/// `build --profile <p> [--manifest-path <path>]` and `build --help`.
fn parse_build(args: &[String]) -> Result<Command<'static>, UsageError> {
    const PROFILE: &str = "--profile";
    const MANIFEST_PATH: &str = "--manifest-path";
    if let Some((first, rest)) = args.split_first()
        && matches!(first.as_str(), "-h" | "--help")
    {
        no_more(rest)?;
        return Ok(Command::Help);
    }
    let mut profile = None;
    let mut manifest_path = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (arg.as_str(), None),
        };
        let (option, slot) = match name {
            PROFILE => (PROFILE, &mut profile),
            MANIFEST_PATH => (MANIFEST_PATH, &mut manifest_path),
            _ if arg.starts_with('-') => return Err(UsageError::UnknownOption(name.to_owned())),
            _ => return Err(UsageError::UnexpectedArgument(arg.clone())),
        };
        let value = match inline {
            Some(value) => value,
            None => match rest.next() {
                Some(value) if !value.starts_with('-') => value.as_str(),
                _ => "",
            },
        };
        if value.is_empty() {
            return Err(UsageError::MissingValue(option));
        }
        if slot.replace(value.to_owned()).is_some() {
            return Err(UsageError::RepeatedOption(option));
        }
    }
    let Some(name) = profile else {
        return Err(UsageError::MissingProfile {
            valid: build::names(),
        });
    };
    let profile = build::find(&name).ok_or_else(|| UsageError::UnknownProfile {
        name,
        valid: build::names(),
    })?;
    Ok(Command::Build {
        profile,
        manifest_path,
    })
}

/// The options given to a top-level command: values and flags, each at most once, or `--help`.
#[derive(Debug, Default, PartialEq, Eq)]
struct Options {
    help: bool,
    values: Vec<(&'static str, String)>,
    flags: Vec<&'static str>,
}

impl Options {
    fn value(&self, name: &str) -> Option<String> {
        self.values
            .iter()
            .find(|(given, _)| *given == name)
            .map(|(_, value)| value.clone())
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.contains(&name)
    }

    /// [`Command::Help`] for `--help`, else the command `build` makes from the options.
    fn or_help(self, build: impl FnOnce(&Options) -> Command<'static>) -> Command<'static> {
        if self.help {
            Command::Help
        } else {
            build(&self)
        }
    }
}

/// Parses a top-level command's options: each of `values` as `--name <value>` or
/// `--name=<value>`, each of `flags` as a bare `--name`, each at most once; or a lone `--help`.
fn parse_options(
    args: &[String],
    values: &[&'static str],
    flags: &[&'static str],
) -> Result<Options, UsageError> {
    if let Some((first, rest)) = args.split_first()
        && matches!(first.as_str(), "-h" | "--help")
    {
        no_more(rest)?;
        return Ok(Options {
            help: true,
            ..Options::default()
        });
    }
    let mut options = Options::default();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (arg.as_str(), None),
        };
        if let Some(flag) = flags.iter().find(|flag| **flag == name) {
            if inline.is_some() {
                return Err(UsageError::UnexpectedArgument(arg.clone()));
            }
            if options.flags.contains(flag) {
                return Err(UsageError::RepeatedOption(flag));
            }
            options.flags.push(flag);
            continue;
        }
        let Some(option) = values.iter().find(|option| **option == name) else {
            return Err(if arg.starts_with('-') {
                UsageError::UnknownOption(name.to_owned())
            } else {
                UsageError::UnexpectedArgument(arg.clone())
            });
        };
        let value = match inline {
            Some(value) => value,
            None => match rest.next() {
                Some(value) if !value.starts_with('-') => value.as_str(),
                _ => "",
            },
        };
        if value.is_empty() {
            return Err(UsageError::MissingValue(option));
        }
        if options.values.iter().any(|(seen, _)| seen == option) {
            return Err(UsageError::RepeatedOption(option));
        }
        options.values.push((option, value.to_owned()));
    }
    Ok(options)
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
    fn instructions_takes_show_or_list_and_an_optional_ref() {
        assert_eq!(
            parse_real("instructions --show AGENTS.md"),
            Ok(Command::Instructions {
                show: Some("AGENTS.md".into()),
                rev: None
            })
        );
        assert_eq!(
            parse_real("instructions --list --ref origin/main"),
            Ok(Command::Instructions {
                show: None,
                rev: Some("origin/main".into())
            })
        );
        let one_of = Err(UsageError::OneOf {
            command: "instructions",
            options: &["--show", "--list"],
        });
        assert_eq!(parse_real("instructions"), one_of);
        assert_eq!(parse_real("instructions --show a --list"), one_of);
        assert_eq!(parse_real("instructions --help"), Ok(Command::Help));
        assert_eq!(
            parse_real("instructions --show").unwrap_err(),
            UsageError::MissingValue("--show")
        );
    }

    #[test]
    fn fixtures_takes_a_subcommand_and_its_options() {
        use crate::fixtures::{Action, Args};
        assert_eq!(
            parse_real("fixtures verify --store s --name n"),
            Ok(Command::Fixtures {
                action: Action::Verify,
                args: Args {
                    store: Some("s".into()),
                    name: Some("n".into()),
                    ..Args::default()
                },
            })
        );
        let one_of = Err(UsageError::OneOf {
            command: "fixtures",
            options: &["fetch", "import", "verify"],
        });
        assert_eq!(parse_real("fixtures"), one_of);
        assert_eq!(parse_real("fixtures frob"), one_of);
        assert_eq!(parse_real("fixtures --help"), Ok(Command::Help));
        assert_eq!(
            parse_real("fixtures fetch --frob"),
            Err(UsageError::UnknownOption("--frob".into()))
        );
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
            "unknown command `frob`; valid commands: ci, test, check-deps, build, deny, spdx, \
             third-party, notices, instructions, fixtures, help"
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
        assert_eq!(
            parse_real("ci pr-title --event e --commits c --label-events l --roster r")
                .unwrap_err()
                .to_string(),
            "job `pr-title` needs `--version-file`"
        );
        assert!(
            parse_real(
                "ci pr-title --event e --commits c --label-events l --roster r --version-file v"
            )
            .is_ok()
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
    fn build_takes_a_profile_and_an_optional_manifest_path() {
        let test_release = build::find("test-release").unwrap();
        for line in [
            "build --profile test-release",
            "build --profile=test-release",
        ] {
            assert_eq!(
                parse_real(line),
                Ok(Command::Build {
                    profile: test_release,
                    manifest_path: None
                }),
                "`{line}`"
            );
        }
        assert_eq!(
            parse_real("build --manifest-path a/Cargo.toml --profile release"),
            Ok(Command::Build {
                profile: build::find("release").unwrap(),
                manifest_path: Some("a/Cargo.toml".into())
            })
        );
        assert_eq!(parse_real("build --help"), Ok(Command::Help));
    }

    #[test]
    fn build_rejects_a_missing_or_unknown_profile_and_anything_else() {
        assert_eq!(
            parse_real("build").unwrap_err().to_string(),
            "`build` needs `--profile <p>`; valid profiles: dev, test-release, chaos, release"
        );
        assert_eq!(
            parse_real("build --profile pr").unwrap_err().to_string(),
            "unknown profile `pr`; valid profiles: dev, test-release, chaos, release"
        );
        for (line, error) in [
            ("build --profile", UsageError::MissingValue("--profile")),
            (
                "build --profile --manifest-path x",
                UsageError::MissingValue("--profile"),
            ),
            (
                "build --profile dev --profile=chaos",
                UsageError::RepeatedOption("--profile"),
            ),
            (
                "build --profile dev --fix",
                UsageError::UnknownOption("--fix".into()),
            ),
            ("build dev", UsageError::UnexpectedArgument("dev".into())),
        ] {
            assert_eq!(parse_real(line), Err(error), "`{line}`");
        }
    }

    #[test]
    fn the_licence_commands_take_their_options() {
        for (line, command) in [
            (
                "deny",
                Command::Deny {
                    manifest_path: None,
                    advisories: false,
                },
            ),
            (
                "deny --advisories --manifest-path=a/Cargo.toml",
                Command::Deny {
                    manifest_path: Some("a/Cargo.toml".into()),
                    advisories: true,
                },
            ),
            ("spdx", Command::Spdx { root: None }),
            (
                "spdx --root fixtures/x",
                Command::Spdx {
                    root: Some("fixtures/x".into()),
                },
            ),
            ("third-party", Command::ThirdParty { root: None }),
            (
                "third-party --root=x",
                Command::ThirdParty {
                    root: Some("x".into()),
                },
            ),
            (
                "notices",
                Command::Notices {
                    manifest_path: None,
                    out: None,
                },
            ),
            (
                "notices --out target/THIRD-PARTY-NOTICES --manifest-path m/Cargo.toml",
                Command::Notices {
                    manifest_path: Some("m/Cargo.toml".into()),
                    out: Some("target/THIRD-PARTY-NOTICES".into()),
                },
            ),
            ("deny --help", Command::Help),
            ("notices -h", Command::Help),
        ] {
            assert_eq!(parse_real(line), Ok(command), "`{line}`");
        }
    }

    #[test]
    fn the_licence_commands_reject_anything_else() {
        for (line, error) in [
            ("deny --fix", UsageError::UnknownOption("--fix".into())),
            (
                "deny licenses",
                UsageError::UnexpectedArgument("licenses".into()),
            ),
            (
                "deny --advisories=yes",
                UsageError::UnexpectedArgument("--advisories=yes".into()),
            ),
            (
                "deny --advisories --advisories",
                UsageError::RepeatedOption("--advisories"),
            ),
            ("spdx --root", UsageError::MissingValue("--root")),
            (
                "spdx --manifest-path x",
                UsageError::UnknownOption("--manifest-path".into()),
            ),
            (
                "third-party --root a --root b",
                UsageError::RepeatedOption("--root"),
            ),
            ("notices --out", UsageError::MissingValue("--out")),
            ("notices --out --check", UsageError::MissingValue("--out")),
            (
                "notices --help x",
                UsageError::UnexpectedArgument("x".into()),
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
