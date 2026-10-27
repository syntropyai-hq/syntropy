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
//! - [`error`]: the exit-code table and error rendering with hints;
//! - [`build_id`]: this binary's `BuildId`, which `build.rs` builds, and its 32-byte build ID
//!   (QUAL-E16-T03).
//!
//! Every command of the tree is declared. `-V`/`--version` prints this binary's build with no
//! daemon; with `--json` it is the native API's `Build`. `version` is served: its `--json` output
//! is the native API's `VersionInfo`, unchanged. `config show --resolved`, `config check` and
//! `config path` are served without a daemon, over the binary's section registry ([`config`]). Each other command
//! fails with "not available in this build yet" (exit 5) until its task builds it. `__runner` and
//! `__selftest` parse their arguments and stop there, because runner mode and the self-test are
//! not built.
//!
//! Dispatch reads no configuration, opens no GPU or network library and builds no async runtime
//! before it knows the command, so a client command prints within RFC-0051's 50 ms \[target\].

#![forbid(unsafe_code)]

use std::process::ExitCode;

use clap::FromArgMatches as _;

pub mod args;
pub mod autostart;
pub mod build_id;
pub mod client;
pub mod config;
pub mod error;
pub mod output;
pub mod prompt;

use args::{Cli, Command, ConfigCommand, RunnerArgs, SelftestArgs};
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
        // `--help` prints and exits 0; a usage error prints and exits 2.
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
    let result = match cli.command {
        _ if cli.version => self_version(output).map(|()| Exit::Success),
        Some(command) => dispatch(command, &path, output),
        // No command at all is a usage error, as clap reports a missing subcommand.
        None => {
            let _ = args::command()
                .error(
                    clap::error::ErrorKind::MissingSubcommand,
                    "a command is required; see `syntropy --help`",
                )
                .print();
            return Exit::Usage.into();
        }
    };
    match result {
        Ok(exit) => exit.into(),
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
///
/// It returns the exit code: success, or a code a command chose to exit with after printing its
/// own diagnostics, as `config check` does on an invalid configuration. An error prints as
/// RFC-0051 §2.2 writes it.
fn dispatch(command: Command, path: &str, output: Output) -> Result<Exit, CliError> {
    match command {
        Command::Version => version(output).map(|()| Exit::Success),
        Command::Config(ConfigCommand::Show(args)) => config::show(&args, output),
        Command::Config(ConfigCommand::Check(args)) => config::check(&args, output),
        Command::Config(ConfigCommand::Path(args)) => config::path(&args, output),
        Command::Runner(args) => runner(&args).map(|()| Exit::Success),
        Command::Selftest(args) => selftest(&args).map(|()| Exit::Success),
        _ => Err(CliError::not_built(path)),
    }
}

/// `syntropy version`: the daemon's build, protocol majors and support tier. With `--json` it
/// prints the native API's `VersionInfo` unchanged (RFC-0051 §2.2). As text, the daemon's
/// strings are made safe for a terminal first ([`output::sanitize`]). The daemon must run: the
/// offline report of this binary alone comes with `version`'s own task (PROD-E6-T02).
fn version(output: Output) -> Result<(), CliError> {
    let os = syntropy_config::Os::current();
    let env = syntropy_config::Env::from_process(os);
    let client = client::Client::reach(client::targets(os, &env)?)?;
    let info = client.version()?;
    if output.json {
        return output::print_json(&info).map_err(write_failed);
    }
    let id = hex(&info.build.id);
    let tier = match info.support_tier {
        syntropy_api::version::SupportTier::Tier1 => "Tier 1",
        syntropy_api::version::SupportTier::Tier2 => "Tier 2",
        syntropy_api::version::SupportTier::Tier3 => "Tier 3",
        syntropy_api::version::SupportTier::Unsupported => "unsupported",
    };
    let text = format!(
        "daemon     syntropy {} (build {id})\nprotocols  wire {}, ipc {}\nsupport    {tier}\n\
         reached    {}\nclient     syntropy {}\n",
        output::sanitize(&info.build.build_id.version.to_string()),
        info.protocols.wire,
        info.protocols.ipc,
        client.target(),
        build_id::current().version,
    );
    print!("{text}");
    Ok(())
}

/// `syntropy --version`: this binary's own build, with no daemon (RFC-0062 §8.1). With `--json`
/// it prints the native API's `Build`, with every `BuildId` field.
fn self_version(output: Output) -> Result<(), CliError> {
    let build = build_id::build();
    if output.json {
        return output::print_json(&build).map_err(write_failed);
    }
    let id = &build.build_id;
    let commit = if id.commit_known() {
        hex(&id.commit)
    } else {
        "unknown".to_owned()
    };
    let features = if build.features.is_empty() {
        "none".to_owned()
    } else {
        build.features.join(", ")
    };
    print!(
        "syntropy {}\ncommit     {commit}\nprofile    {}\ntarget     {}\ntoolchain  {}\n\
         features   {features}\nbuild      {}\n",
        id.version,
        id.profile,
        id.target,
        hex(&id.toolchain.0),
        hex(&build.id),
    );
    Ok(())
}

/// `bytes` as lower-case hex.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
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
