// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `syntropy` binary: argument parsing, mode dispatch and the client commands.
//!
//! Layer 6 in RFC-0001 §4.1, designed in RFC-0051 §1–§2. One binary runs the daemon, each
//! runner and every client command (ADR-0015 decision 1). Built so far (PROD-E6-T01):
//!
//! - [`args`]: the command tree of RFC-0051 §2.1 and the hidden `__runner` and `__selftest`
//!   modes;
//! - [`client`]: the typed native-API client, over the local socket, then loopback HTTP, or the
//!   remote daemon `SYNTROPY_HOST` names;
//! - [`autostart`]: when a command may start a daemon, and how;
//! - [`output`]: terminal detection, colour, `--json`, JSON Lines, units and labels;
//! - [`prompt`]: questions, which are never asked without a terminal;
//! - [`error`]: the exit-code table and error rendering with hints.
//!
//! Every command of the tree is declared. `version` is served: its `--json` output is the native
//! API's `VersionInfo`, unchanged. Each other command fails with "not available in this build
//! yet" (exit 5) until its task builds it. `__runner` and `__selftest` parse their arguments and
//! stop there, because runner mode and the self-test are not built.
//!
//! Dispatch reads no configuration, opens no GPU or network library and builds no async runtime
//! before it knows the command, so a client command prints within RFC-0051's 50 ms \[target\].

#![forbid(unsafe_code)]

use std::process::ExitCode;

use clap::FromArgMatches as _;

pub mod args;
pub mod autostart;
pub mod client;
pub mod error;
pub mod output;
pub mod prompt;

use args::{Cli, Command, RunnerArgs, SelftestArgs};
use error::{CliError, Exit};
use output::{Output, Terminal};

/// Runs the command line `args` (the program name first) and returns the exit code.
pub fn run<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let matches = match args::command().try_get_matches_from(args) {
        Ok(matches) => matches,
        // `--help` and `--version` print and exit 0; a usage error prints and exits 2.
        Err(error) => {
            let _ = error.print();
            return ExitCode::from(u8::try_from(error.exit_code()).unwrap_or(Exit::Usage.code()));
        }
    };
    let path = command_path(&matches);
    let cli = match Cli::from_arg_matches(&matches) {
        Ok(cli) => cli,
        Err(error) => {
            let _ = error.print();
            return Exit::Usage.into();
        }
    };
    let terminal = Terminal::detect();
    let output = Output::new(&cli.global, terminal);
    match dispatch(cli.command, &path, output) {
        Ok(()) => Exit::Success.into(),
        Err(error) => {
            eprintln!("{error}");
            error.exit.into()
        }
    }
}

/// The subcommand names `matches` went through, such as `cluster status`.
fn command_path(matches: &clap::ArgMatches) -> String {
    let mut names = Vec::new();
    let mut current = matches;
    while let Some((name, sub)) = current.subcommand() {
        names.push(name);
        current = sub;
    }
    names.join(" ")
}

/// Runs one command.
fn dispatch(command: Command, path: &str, output: Output) -> Result<(), CliError> {
    match command {
        Command::Version => version(output),
        Command::Runner(args) => runner(&args),
        Command::Selftest(args) => selftest(&args),
        _ => Err(CliError::not_built(path)),
    }
}

/// `syntropy version`: the daemon's build, protocol majors and support tier. With `--json` it
/// prints the native API's `VersionInfo` unchanged (RFC-0051 §2.2). The daemon must run: the
/// offline report of this binary alone comes with `version`'s own task (PROD-E6-T02).
fn version(output: Output) -> Result<(), CliError> {
    let os = syntropy_config::Os::current();
    let env = syntropy_config::Env::from_process(os);
    let client = client::Client::reach(client::targets(os, &env)?)?;
    let info = client.version()?;
    if output.json {
        return output::print_json(&info).map_err(write_failed);
    }
    let id: String = info.build.id.iter().map(|b| format!("{b:02x}")).collect();
    let tier = match info.support_tier {
        syntropy_api::version::SupportTier::Tier1 => "Tier 1",
        syntropy_api::version::SupportTier::Tier2 => "Tier 2",
        syntropy_api::version::SupportTier::Tier3 => "Tier 3",
        syntropy_api::version::SupportTier::Unsupported => "unsupported",
    };
    let text = format!(
        "daemon     syntropy {} (build {id})\nprotocols  wire {}, ipc {}\nsupport    {tier}\n\
         reached    {}\nclient     syntropy {}\n",
        info.build.version,
        info.protocols.wire,
        info.protocols.ipc,
        client.target(),
        env!("CARGO_PKG_VERSION"),
    );
    print!("{text}");
    Ok(())
}

/// `syntropy __runner --ipc <endpoint>`: runner mode (ADR-0015 decision 1). The runner process,
/// its lifecycle and the command channel's runner side are DIST-E1-T04 and DIST-E1-T05; until
/// they land, this dispatch stops after parsing the endpoint, and opens nothing.
fn runner(args: &RunnerArgs) -> Result<(), CliError> {
    Err(CliError::cli(
        Exit::Failure,
        "not_built",
        format!(
            "runner mode is not available in this build yet; the daemon's command channel at {} \
             was not opened",
            args.ipc
        ),
    ))
}

/// `syntropy __selftest --root <root>`: a staged build's self-test, which the updater runs
/// (RFC-0052 §7.3). Not built yet.
fn selftest(args: &SelftestArgs) -> Result<(), CliError> {
    Err(CliError::cli(
        Exit::Failure,
        "not_built",
        format!(
            "the self-test is not available in this build yet; {} was not checked",
            args.root.display()
        ),
    ))
}

/// The error for output that could not be written, such as a closed pipe.
fn write_failed(error: std::io::Error) -> CliError {
    CliError::cli(
        Exit::Failure,
        "output",
        format!("cannot write the output: {error}"),
    )
}
