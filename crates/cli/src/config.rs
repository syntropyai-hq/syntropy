// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy config show --resolved`, `config check` and `config path` (RFC-0051 §2.1; RFC-0054
//! §1, §2.2, §2.4; RFC-0001 §11.2; PROD-E6-T23).
//!
//! Each calls `syntropy-config`'s `command` function over [`REGISTRY`], the binary's section
//! registry, with the process environment and the `--config` and `--set` flags. None needs a
//! daemon (RFC-0051 §1).
//!
//! Output follows RFC-0051 §2.2. The command's result goes to stdout: the resolved values,
//! `check`'s summary line, or the paths. Every diagnostic goes to stderr, in RFC-0054 §8's form,
//! as the configuration's own renderer writes it: an error, a warning such as an unknown
//! `SYNTROPY_*` variable, and its source lines. The renderer quotes the configuration's sources,
//! so each line is escaped on the way out ([`syntropy_api::printable`]). An invalid configuration
//! exits 5.
//!
//! The files are those a user's daemon reads (`Mode::User`). `--json` is refused until the API
//! has a configuration type to print (`/v1/syntropy/config`, RFC-0054 §10.2).

use std::path::Path;

use syntropy_api::config::ApiSection;
use syntropy_api::rules::API_RULES;
use syntropy_config::{
    ConfigSection, DebugSection, Env, Flags, HostEnvironment, Locations, Mode, Os, Problem,
    Registry, Scan, SectionSchema, command, load_and_validate, locate,
};

use crate::args::{ConfigCheckArgs, ConfigPathArgs, ConfigShowArgs, ConfigSource};
use crate::error::{CliError, Exit};
use crate::output::Output;

/// `syntropy-net`'s sections.
const NET: &[&SectionSchema] = syntropy_net::config::SECTIONS;
/// `syntropy-cluster`'s sections.
const CLUSTER: &[&SectionSchema] = syntropy_cluster::config::SECTIONS;
/// The sections of crates that declare one each: the API's listeners, the debug endpoints, the
/// node's logging and the scheduler's instance keys.
const SINGLES: [&SectionSchema; 4] = [
    &ApiSection::SCHEMA,
    &DebugSection::SCHEMA,
    &syntropy_config::logging::LoggingSection::SCHEMA,
    &syntropy_sched::SchedSection::SCHEMA,
];

/// How many sections the binary links.
const COUNT: usize = NET.len() + CLUSTER.len() + SINGLES.len();

/// Every section the binary links, in one array: `net`'s, then `cluster`'s, then the others.
const ALL: [&SectionSchema; COUNT] = {
    let mut all = [SINGLES[0]; COUNT];
    let mut i = 0;
    while i < NET.len() {
        all[i] = NET[i];
        i += 1;
    }
    let mut j = 0;
    while j < CLUSTER.len() {
        all[NET.len() + j] = CLUSTER[j];
        j += 1;
    }
    let mut k = 0;
    while k < SINGLES.len() {
        all[NET.len() + CLUSTER.len() + k] = SINGLES[k];
        k += 1;
    }
    all
};

/// The binary's section registry: every configuration section of the crates it links. A
/// `static`, so two sections that claim one prefix fail the build. A crate that adds a section
/// lists it in its own `SECTIONS`, or here.
pub static REGISTRY: Registry = Registry::new(&ALL);

/// `config show`: with `--resolved`, every key's effective value and where it came from.
pub fn show(args: &ConfigShowArgs, output: Output) -> Result<Exit, CliError> {
    refuse_json(output, "show")?;
    if !args.resolved {
        return Err(CliError::not_built("config show")
            .detail("only the resolved view is built")
            .hint("syntropy config show --resolved"));
    }
    let checked = load(&args.source)?;
    let diagnostics = checked.render(&REGISTRY);
    eprint!("{}", printable_lines(&diagnostics));
    match &checked.outcome {
        Ok(config) => {
            // The values only: stderr already carries each warning in RFC-0054 §8's form, so
            // the one-line copies `show_resolved` appends come out of stdout.
            let mut values = command::show_resolved(config);
            for warning in config.warnings() {
                values = values.replacen(&format!("\nwarning: {warning}\n"), "", 1);
            }
            print!("{values}");
            Ok(Exit::Success)
        }
        Err(_) => Err(CliError::cli(
            Exit::Invalid,
            "invalid_config",
            format!(
                "the configuration has {}, so nothing is shown",
                errors(checked.errors().len())
            ),
        )
        .hint("syntropy config check")),
    }
}

/// `config check`: every error and warning of RFC-0054 §8's phases 1–5, without a daemon.
pub fn check(args: &ConfigCheckArgs, output: Output) -> Result<Exit, CliError> {
    refuse_json(output, "check")?;
    let checked = load(&args.source)?;
    let diagnostics = checked.render(&REGISTRY);
    let report = command::check(&checked, &REGISTRY);
    eprint!("{}", printable_lines(&diagnostics));
    // The report is the diagnostics, then one summary line.
    print!(
        "{}",
        report
            .text
            .strip_prefix(&diagnostics)
            .unwrap_or(&report.text)
    );
    Ok(if report.exit == command::EXIT_OK {
        Exit::Success
    } else {
        Exit::Invalid
    })
}

/// `config path`: RFC-0054 §1's four orders, whether each file exists, and the state, store and
/// runtime locations.
pub fn path(args: &ConfigPathArgs, output: Output) -> Result<Exit, CliError> {
    refuse_json(output, "path")?;
    let (_, _, locations) = locations(args.config.as_deref());
    let scan = match locations.scan() {
        Ok(scan) => scan,
        // An explicit file that does not exist: the paths still print, with it absent.
        Err(problem) => {
            eprintln!(
                "warning: {}",
                syntropy_api::printable::printable(&problem.to_string())
            );
            Scan {
                system_file: None,
                dropins: Vec::new(),
                user_file: None,
                explicit: None,
            }
        }
    };
    print!("{}", command::path(&locations, &scan));
    Ok(Exit::Success)
}

/// The OS, its environment and where a user's daemon looks, with `config` as the explicit file.
fn locations(config: Option<&Path>) -> (Os, Env, Locations) {
    let os = Os::current();
    let env = Env::from_process(os);
    let locations = locate(os, Mode::User, &env, config);
    (os, env, locations)
}

/// Reads and checks the configuration `source` names: RFC-0054 §8's phases 1–5. Computed
/// defaults stay uncomputed, since computing them needs the probe, and `show` says so per key.
fn load(source: &ConfigSource) -> Result<syntropy_config::Checked, CliError> {
    let (os, env, locations) = locations(source.config.as_deref());
    let mut flags = Flags::new();
    for arg in &source.set {
        flags.set(arg).map_err(invalid)?;
    }
    let scan = locations.scan().map_err(invalid)?;
    Ok(load_and_validate(
        &REGISTRY,
        os,
        &scan,
        &env,
        &flags,
        |_| None,
        API_RULES,
        &HostEnvironment,
    ))
}

/// A problem before the configuration resolves, such as a `--set` without `=` or a missing
/// `--config` file: exit 5, in the configuration's own words.
fn invalid(problem: Problem) -> CliError {
    let code = problem.code.map_or("invalid", |code| code.name());
    CliError {
        exit: Exit::Invalid,
        domain: "config",
        code: code.to_owned(),
        message: problem.to_string(),
        details: Vec::new(),
        hints: Vec::new(),
    }
}

/// `--json` has no API type to print for these commands yet.
fn refuse_json(output: Output, command: &str) -> Result<(), CliError> {
    if output.json {
        return Err(CliError::cli(
            Exit::Invalid,
            "not_built",
            format!(
                "`syntropy config {command} --json` is not available in this build yet: the API's \
                 configuration type comes with `/v1/syntropy/config` (RFC-0054 §10.2)"
            ),
        ));
    }
    Ok(())
}

/// `text` safe to print on a terminal, a line at a time: each line through
/// `syntropy_api::printable`, with its newline kept. The renderer quotes the configuration's own
/// sources, a file's or the environment's values and the variables' names, and none of them is
/// text this binary wrote (RFC-0051 §2.2): a TOML `\u001b` escape, an environment value or a
/// variable's name can hold ESC or BEL.
fn printable_lines(text: &str) -> String {
    let mut safe = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let (body, end) = match line.strip_suffix('\n') {
            Some(body) => (body, "\n"),
            None => (line, ""),
        };
        safe.push_str(&syntropy_api::printable::printable(body));
        safe.push_str(end);
    }
    safe
}

/// "1 error" or "n errors".
fn errors(n: usize) -> String {
    format!("{n} error{}", if n == 1 { "" } else { "s" })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_holds_every_section_and_every_default_fits() {
        assert_eq!(REGISTRY.check(), vec![]);
        assert_eq!(REGISTRY.check_defaults(), vec![]);
        for prefix in [
            "api.bind",
            "debug.parity_api",
            "logging.include_content",
            "net.listen_port",
            "discovery.port",
            "swim.period",
            "quic.congestion",
        ] {
            assert!(REGISTRY.key(prefix).is_some(), "{prefix}");
        }
        assert_eq!(COUNT, NET.len() + CLUSTER.len() + 4);
    }

    /// The registry is the linked crates' lists, then the four single sections, in that order,
    /// by prefix and count.
    #[test]
    fn the_registry_is_net_then_cluster_then_the_singles() {
        let got: Vec<&str> = REGISTRY.sections().iter().map(|s| s.prefix).collect();
        let want: Vec<&str> = NET
            .iter()
            .chain(CLUSTER)
            .chain(SINGLES.iter())
            .map(|s| s.prefix)
            .collect();
        assert_eq!(got, want);
        assert_eq!(
            got.len(),
            33,
            "the sections the workspace declares on 2026-10-27, `logging` included"
        );
    }

    /// The drift guard: every configuration section a workspace crate `cli` depends on declares,
    /// outside its tests, is in the registry, and the registry holds nothing else. A crate that
    /// adds a section, or a new dependency of `cli` that has one, fails this test until the
    /// registry lists it. The crates come from `cli`'s own manifest, so the guard follows it; a
    /// section in a crate `cli` reaches only through another needs a direct edge to be listed.
    #[test]
    fn every_section_the_linked_crates_declare_is_in_the_registry() {
        let cli = Path::new(env!("CARGO_MANIFEST_DIR"));
        let manifest = std::fs::read_to_string(cli.join("Cargo.toml")).unwrap();
        let crates: Vec<&str> = manifest
            .lines()
            .filter_map(|line| line.split_once("path = \"../")?.1.split_once('"'))
            .map(|(name, _)| name)
            .collect();
        assert!(
            crates.contains(&"cluster") && crates.contains(&"api"),
            "{crates:?}"
        );
        let mut declared = std::collections::BTreeSet::new();
        for name in &crates {
            declared_sections(&cli.join("..").join(name).join("src"), &mut declared);
        }
        let registered: std::collections::BTreeSet<&str> =
            REGISTRY.sections().iter().map(|s| s.type_name).collect();
        let declared: std::collections::BTreeSet<&str> =
            declared.iter().map(String::as_str).collect();
        assert_eq!(
            declared, registered,
            "the linked crates' sections and the registry differ"
        );
    }

    /// The structs under a `#[config(section = …)]` attribute in the `.rs` files below `dir`,
    /// leaving out test modules (from `#[cfg(test)]` on), test files and doc examples.
    fn declared_sections(dir: &Path, out: &mut std::collections::BTreeSet<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.map(Result::unwrap) {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                if name != "tests" {
                    declared_sections(&path, out);
                }
                continue;
            }
            if !name.ends_with(".rs") || name == "tests.rs" {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let code = text.split("#[cfg(test)]").next().unwrap_or("");
            let mut pending = false;
            for line in code.lines().map(str::trim) {
                if line.starts_with("#[config(section") {
                    pending = true;
                } else if pending && let Some(rest) = line.strip_prefix("pub struct ") {
                    let ident: String = rest
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    out.insert(ident);
                    pending = false;
                }
            }
        }
    }
}
