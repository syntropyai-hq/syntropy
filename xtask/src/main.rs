// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask`: the one entry point for CI jobs and test suites (RFC-0062 §1, RFC-0060 §1.2).
//!
//! CI runs every job as `cargo xtask ci <job>` and every suite as `cargo xtask test <suite>`, so a
//! job run locally gives the same verdict as the same job in CI on the same commit. The alias in
//! `.cargo/config.toml` builds and runs this binary with `--locked`. Jobs and suites live in
//! [`registry`], which says how to add one.
//!
//! The tool sits outside the crate graph of RFC-0001 §4.1: no crate depends on it, and it depends
//! on none.

#![forbid(unsafe_code)]

mod cli;
mod registry;
mod run;

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use cli::{Command, Kind, USAGE};
use run::Verdict;

/// Every step passed, or help or a list was printed.
const EXIT_PASS: u8 = 0;
/// A step failed, or xtask could not write its log.
const EXIT_FAIL: u8 = 1;
/// The command line was not understood.
const EXIT_USAGE: u8 = 2;

fn main() -> ExitCode {
    let args: Result<Vec<String>, _> = std::env::args_os()
        .skip(1)
        .map(|a| a.into_string())
        .collect();
    let Ok(args) = args else {
        eprintln!("error: arguments must be valid UTF-8");
        return ExitCode::from(EXIT_USAGE);
    };
    let command = match cli::parse(&args, registry::JOBS, registry::SUITES) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!("Run `cargo xtask help` for usage.");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    ExitCode::from(match execute(command) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: could not write output: {error}");
            EXIT_FAIL
        }
    })
}

fn execute(command: Command<'_>) -> io::Result<u8> {
    match command {
        Command::Help => {
            let mut out = io::stdout().lock();
            writeln!(out, "{USAGE}")?;
            out.flush()?;
            Ok(EXIT_PASS)
        }
        Command::List(kind) => {
            let entries = match kind {
                Kind::Job => registry::JOBS,
                Kind::Suite => registry::SUITES,
            };
            run::list(entries, &mut io::stdout().lock())?;
            Ok(EXIT_PASS)
        }
        Command::Run { kind, entry } => {
            let verdict = run::run(
                kind,
                entry,
                &workspace_root(),
                &mut run::System,
                &mut io::stderr(),
            )?;
            Ok(exit_code(&verdict))
        }
    }
}

fn exit_code(verdict: &Verdict) -> u8 {
    match verdict {
        Verdict::Pass => EXIT_PASS,
        Verdict::Fail(_) => EXIT_FAIL,
    }
}

/// The workspace root: the parent of this crate's directory. Every step runs from there.
fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent().unwrap_or(manifest_dir).to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pass_exits_zero_and_a_failure_exits_one() {
        assert_eq!(exit_code(&Verdict::Pass), 0);
        assert_eq!(exit_code(&Verdict::Fail(vec!["fmt"])), 1);
    }

    #[test]
    fn the_workspace_root_holds_the_workspace_manifest() {
        let manifest = std::fs::read_to_string(workspace_root().join("Cargo.toml")).unwrap();
        assert!(manifest.contains("[workspace]"));
        assert!(manifest.contains("\"xtask\""));
    }
}
