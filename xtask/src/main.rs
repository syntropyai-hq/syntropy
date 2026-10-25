// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask`: the one entry point for CI jobs and test suites (RFC-0062 §1, RFC-0060 §1.2).
//!
//! CI runs every job as `cargo xtask ci <job>` and every suite as `cargo xtask test <suite>`, so a
//! job run locally gives the same verdict as the same job in CI on the same commit. The alias in
//! `.cargo/config.toml` builds and runs this binary with `--locked`. Jobs and suites live in
//! [`registry`], which says how to add one.
//!
//! Steps run in the checkout xtask was started from: the workspace root comes from the
//! `CARGO_MANIFEST_DIR` that `cargo run` sets at run time, never from the path the binary was
//! built at. One binary can serve several checkouts through a copied `target/`, and it must check
//! the one it was asked to. The alias builds xtask into `target/xtask/`, apart from the
//! workspace build its steps run, because Windows will not replace a running binary.
//!
//! `cargo xtask check-deps` checks the crate dependency rules of RFC-0001 §4.2 ([`check_deps`]).
//! `cargo xtask build --profile <p>` builds in a test build profile and checks the result
//! ([`build`]).
//! `cargo xtask deny`, `spdx`, `third-party` and `notices` apply the licence policy of ADR-0018
//! ([`licences`]).
//!
//! The tool sits outside the crate graph of RFC-0001 §4.1: no crate depends on it, and it depends
//! on none. `ci/layers.toml` lists it as a tool, and `check-deps` holds it to that.

#![forbid(unsafe_code)]

mod actionlint;
mod build;
mod cache_hit;
mod changes;
mod check_deps;
mod ci_gate;
mod cli;
mod codeowners;
mod contrib;
mod dep_diff;
mod determinism;
mod fixtures;
mod flake;
mod hot_threads;
mod instructions;
mod licences;
mod loom;
mod orphan_rule;
mod plan_tools;
mod properties;
mod quarantine;
mod registry;
mod repo_settings;
mod roster;
mod run;
mod runner_build_id;
mod sim_presets;
mod toolchain;
mod toolchains;
mod workflows;

use std::ffi::OsString;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use cli::{Command, Kind, USAGE};
use run::Verdict;

/// The exit codes, which [`flake`] defines so a runner can tell an `infra` failure from a `test`
/// failure without reading the log (RFC-0060 §13).
use flake::{EXIT_INFRA, EXIT_PASS, EXIT_TEST as EXIT_FAIL, EXIT_USAGE};

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
        Command::Run { kind, entry, args } => {
            let root = match workspace_root(std::env::var_os("CARGO_MANIFEST_DIR")) {
                Ok(root) => root,
                Err(error) => {
                    eprintln!("error: {error}");
                    return Ok(EXIT_USAGE);
                }
            };
            let context = run::Context { root, args };
            let verdict = run::run(kind, entry, &context, &mut run::System, &mut io::stderr())?;
            Ok(exit_code(&verdict))
        }
        Command::CheckDeps { manifest_path } => {
            let manifest = match manifest_path {
                Some(path) => PathBuf::from(path),
                None => match workspace_root(std::env::var_os("CARGO_MANIFEST_DIR")) {
                    Ok(root) => root.join("Cargo.toml"),
                    Err(error) => {
                        eprintln!("error: {error}");
                        return Ok(EXIT_USAGE);
                    }
                },
            };
            let pass = check_deps::run(&manifest, &mut io::stderr())?;
            Ok(if pass { EXIT_PASS } else { EXIT_FAIL })
        }
        Command::Build {
            profile,
            manifest_path,
        } => {
            let manifest = match manifest_path {
                Some(path) => PathBuf::from(path),
                None => match workspace_root(std::env::var_os("CARGO_MANIFEST_DIR")) {
                    Ok(root) => root.join("Cargo.toml"),
                    Err(error) => {
                        eprintln!("error: {error}");
                        return Ok(EXIT_USAGE);
                    }
                },
            };
            let pass = build::run(&manifest, profile, &mut io::stderr())?;
            Ok(if pass { EXIT_PASS } else { EXIT_FAIL })
        }
        Command::Deny {
            manifest_path,
            advisories,
        } => {
            let checkout = match checkout() {
                Ok(root) => root,
                Err(code) => return Ok(code),
            };
            let manifest = manifest_path.map_or_else(|| checkout.join("Cargo.toml"), PathBuf::from);
            let checks = if advisories {
                licences::deny::Checks::Advisories
            } else {
                licences::deny::Checks::Policy
            };
            let pass = licences::deny::run(&checkout, &manifest, checks, &mut io::stderr())?;
            Ok(if pass { EXIT_PASS } else { EXIT_FAIL })
        }
        Command::Instructions { show, rev } => {
            let root = match checkout() {
                Ok(root) => root,
                Err(code) => return Ok(code),
            };
            let source = match &rev {
                Some(rev) => instructions::Source::Commit { root: &root, rev },
                None => instructions::Source::Tree(&root),
            };
            let outcome = match &show {
                Some(path) => instructions::show(&source, path),
                None => instructions::load(&source).map(|entries| {
                    entries
                        .iter()
                        .map(|entry| {
                            let planned = entry
                                .planned
                                .as_ref()
                                .map_or(String::new(), |task| format!(" (planned: {task})"));
                            format!("{}\t{}\t{}{planned}\n", entry.path, entry.kind, entry.about)
                        })
                        .collect()
                }),
            };
            match outcome {
                Ok(text) => {
                    let mut out = io::stdout().lock();
                    out.write_all(text.as_bytes())?;
                    out.flush()?;
                    Ok(EXIT_PASS)
                }
                Err(error) => {
                    eprintln!("instructions: {error}");
                    Ok(EXIT_FAIL)
                }
            }
        }
        Command::Fixtures { action, args } => {
            let root = match checkout() {
                Ok(root) => root,
                Err(code) => return Ok(code),
            };
            match fixtures::run(action, &args, &root, &mut io::stdout()) {
                Ok(()) => Ok(EXIT_PASS),
                Err(error) => {
                    eprintln!("fixtures: {error}");
                    Ok(EXIT_FAIL)
                }
            }
        }
        Command::Spdx { root } => {
            let checkout = match checkout() {
                Ok(root) => root,
                Err(code) => return Ok(code),
            };
            let root = root.map_or(checkout, PathBuf::from);
            let pass = licences::headers::run(&root, &mut io::stderr())?;
            Ok(if pass { EXIT_PASS } else { EXIT_FAIL })
        }
        Command::ThirdParty { root } => {
            let checkout = match checkout() {
                Ok(root) => root,
                Err(code) => return Ok(code),
            };
            let root = root.map_or_else(|| checkout.clone(), PathBuf::from);
            let pass = licences::manifest::run(&checkout, &root, &mut io::stderr())?;
            Ok(if pass { EXIT_PASS } else { EXIT_FAIL })
        }
        Command::Workflows { root } => {
            let checkout = match checkout() {
                Ok(root) => root,
                Err(code) => return Ok(code),
            };
            let root = root.map_or(checkout, PathBuf::from);
            let pass = workflows::run(&root, &mut io::stderr())?;
            Ok(if pass { EXIT_PASS } else { EXIT_FAIL })
        }
        Command::Actionlint { root } => {
            let checkout = match checkout() {
                Ok(root) => root,
                Err(code) => return Ok(code),
            };
            let root = root.map_or(checkout, PathBuf::from);
            let pass = actionlint::run(&root, &mut io::stderr())?;
            Ok(if pass { EXIT_PASS } else { EXIT_FAIL })
        }
        Command::Notices { manifest_path, out } => {
            let checkout = match checkout() {
                Ok(root) => root,
                Err(code) => return Ok(code),
            };
            let manifest = manifest_path.map_or_else(|| checkout.join("Cargo.toml"), PathBuf::from);
            let out = out.map(PathBuf::from);
            let pass =
                licences::notices::run(&checkout, &manifest, out.as_deref(), &mut io::stderr())?;
            Ok(if pass { EXIT_PASS } else { EXIT_FAIL })
        }
    }
}

/// The root of the checkout xtask runs in, or the exit code for a usage error after printing it.
fn checkout() -> Result<PathBuf, u8> {
    workspace_root(std::env::var_os("CARGO_MANIFEST_DIR")).map_err(|error| {
        eprintln!("error: {error}");
        EXIT_USAGE
    })
}

fn exit_code(verdict: &Verdict) -> u8 {
    match verdict {
        Verdict::Pass => EXIT_PASS,
        Verdict::Fail(_) => EXIT_FAIL,
        Verdict::Infra(_) => EXIT_INFRA,
    }
}

/// The workspace root, from `manifest_dir`: the run-time value of `CARGO_MANIFEST_DIR`, which
/// `cargo run` sets to this crate's directory in the checkout it runs in. The root is its parent,
/// and must hold a `Cargo.toml`. There is no fallback: without the variable xtask cannot know
/// which checkout to check, so it refuses to run.
fn workspace_root(manifest_dir: Option<OsString>) -> Result<PathBuf, String> {
    let Some(manifest_dir) = manifest_dir.filter(|dir| !dir.is_empty()) else {
        return Err(
            "CARGO_MANIFEST_DIR is not set, so the workspace root is unknown; \
                    run xtask as `cargo xtask`, which sets it"
                .to_owned(),
        );
    };
    let manifest_dir = PathBuf::from(manifest_dir);
    let root = match manifest_dir.parent() {
        Some(root) if manifest_dir.is_absolute() => root,
        _ => {
            return Err(format!(
                "CARGO_MANIFEST_DIR `{}` is not an absolute path inside a workspace",
                manifest_dir.display()
            ));
        }
    };
    let manifest = root.join("Cargo.toml");
    if !manifest.is_file() {
        return Err(format!(
            "no workspace manifest at `{}` (from CARGO_MANIFEST_DIR `{}`)",
            manifest.display(),
            manifest_dir.display()
        ));
    }
    Ok(root.to_path_buf())
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
    fn an_infra_failure_exits_three_and_classifies_as_infra() {
        let code = exit_code(&Verdict::Infra(vec!["deny"]));
        assert_eq!(code, 3);
        assert_eq!(
            flake::classify(flake::End::Exited(i32::from(code))),
            flake::Outcome::Fail(flake::Class::Infra)
        );
        let code = exit_code(&Verdict::Fail(vec!["fmt"]));
        assert_eq!(
            flake::classify(flake::End::Exited(i32::from(code))),
            flake::Outcome::Fail(flake::Class::Test)
        );
    }

    /// A directory under the system temp directory, removed when dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let dir =
                std::env::temp_dir().join(format!("syntropy-xtask-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn under_cargo_the_root_is_this_workspace() {
        // `cargo test` sets CARGO_MANIFEST_DIR at run time, as `cargo run` does.
        let root = workspace_root(std::env::var_os("CARGO_MANIFEST_DIR")).unwrap();
        let manifest = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
        assert!(manifest.contains("[workspace]"));
        assert!(manifest.contains("\"xtask\""));
    }

    #[test]
    fn the_root_follows_the_run_time_value_not_the_build_path() {
        // Another checkout's layout: <scratch>/Cargo.toml and <scratch>/xtask.
        let other = Scratch::new("other-checkout");
        std::fs::write(other.0.join("Cargo.toml"), "[workspace]\n").unwrap();
        std::fs::create_dir(other.0.join("xtask")).unwrap();
        let root = workspace_root(Some(other.0.join("xtask").into_os_string())).unwrap();
        assert_eq!(root, other.0);
        let built_at = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        assert_ne!(Some(root.as_path()), built_at.parent());
    }

    #[test]
    fn without_cargo_manifest_dir_xtask_refuses_to_run() {
        for unset in [None, Some(OsString::new())] {
            let error = workspace_root(unset).unwrap_err();
            assert!(
                error.starts_with("CARGO_MANIFEST_DIR is not set"),
                "{error}"
            );
        }
    }

    #[test]
    fn a_relative_or_rootless_manifest_dir_is_refused() {
        for dir in ["xtask", "/"] {
            let error = workspace_root(Some(dir.into())).unwrap_err();
            assert!(error.contains("not an absolute path"), "`{dir}`: {error}");
        }
    }

    #[test]
    fn a_root_without_a_manifest_is_refused() {
        let empty = Scratch::new("no-manifest");
        std::fs::create_dir(empty.0.join("xtask")).unwrap();
        let error = workspace_root(Some(empty.0.join("xtask").into_os_string())).unwrap_err();
        assert!(error.starts_with("no workspace manifest at"), "{error}");
    }
}
