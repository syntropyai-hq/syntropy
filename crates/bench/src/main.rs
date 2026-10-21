// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy-bench`: the benchmark harness's checks and tools (RFC-0061).
//!
//! Each command family is a word after the binary's name, so families can be added without
//! touching each other:
//!
//! - `rigs check [--file <path>]` validates the rig inventory, `bench/lab/rigs.toml` by default
//!   (RFC-0061 §6.3). `cargo xtask ci lint` runs it.
//! - `rigs fingerprint [--file <path>]` prints each rig's `RigFingerprint` (§2.1).
//! - `rigs epochs <old> <new>` lists the epoch changes between two revisions of the inventory,
//!   for the nightly report: for example the file at the last nightly's commit, from
//!   `git show <commit>:bench/lab/rigs.toml`, against the current one.
//! - `scenarios check [--base <ref>] [--root <dir>]` checks every scenario file under
//!   `bench/scenarios/` (format, ID form, rigs, and the `scenario_version` rule against the base
//!   commit), checks `bench/sc-map.toml` against the success criteria and the epics, and checks
//!   that RFC-0061 §4.2's generated table is current (RFC-0061 §4.1, §4.2). `lint` runs it.
//! - `scenarios render [--write] [--root <dir>]` prints that table, or writes it into RFC-0061.
//!   `--root` defaults to the working directory, which must be the repository root.
//! - `specs check [--dir <path>]` validates the spec database, `bench/specs/` by default
//!   (RFC-0061 §5.3): every figure has a vendor or standards source URL, its location in the
//!   document and the date it was checked. `cargo xtask ci lint` runs it.
//! - `specs peaks <rig> [--file <path>] [--dir <path>]` prints the rig's `W_peak` per path, with
//!   each path's elements and the narrowest one, from its inventory record and the database.
//!
//! Exit status: 0 on success, 1 when a check fails, 2 on a usage error.

#![forbid(unsafe_code)]

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use syntropy_bench::rigs::{self, Inventory};
use syntropy_bench::specs::{self, SpecDb};
use syntropy_bench::{sc_map, scenario};

const USAGE: &str = "\
Usage: syntropy-bench <command>

Commands:
  rigs check [--file <path>]        validate the rig inventory (RFC-0061 §6.3); `lint` runs it
  rigs fingerprint [--file <path>]  print each rig's RigFingerprint (RFC-0061 §2.1)
  rigs epochs <old> <new>           list the epoch changes between two inventory revisions
  scenarios check [--base <ref>] [--root <dir>]
                                    check bench/scenarios/, bench/sc-map.toml and RFC-0061
                                    §4.2's table (RFC-0061 §4); `lint` runs it
  scenarios render [--write] [--root <dir>]
                                    print RFC-0061 §4.2's table; --write puts it into the RFC
  specs check [--dir <path>]        validate the spec database (RFC-0061 §5.3); `lint` runs it
  specs peaks <rig> [--file <path>] [--dir <path>]
                                    print the rig's W_peak per path from its inventory record
  help                              print this message

--file defaults to bench/lab/rigs.toml and --dir to bench/specs, in the workspace. --base
defaults to `git merge-base HEAD origin/main`; --root to the working directory.";

const PASS: u8 = 0;
const FAIL: u8 = 1;
const USAGE_ERROR: u8 = 2;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    let code = match words.as_slice() {
        ["help" | "-h" | "--help"] => {
            println!("{USAGE}");
            PASS
        }
        ["rigs", "check", rest @ ..] => file_option(rest).map_or(USAGE_ERROR, |file| check(&file)),
        ["rigs", "fingerprint", rest @ ..] => {
            file_option(rest).map_or(USAGE_ERROR, |file| fingerprint(&file))
        }
        ["rigs", "epochs", old, new] => epochs(Path::new(old), Path::new(new)),
        ["scenarios", rest @ ..] => match parse_scenarios(rest) {
            Ok(Scenarios::Help) => {
                println!("{USAGE}");
                PASS
            }
            Ok(Scenarios::Check { base, root }) => scenarios_check(&root, base.as_deref()),
            Ok(Scenarios::Render { write, root }) => scenarios_render(&root, write),
            Err(error) => {
                eprintln!("error: {error}\n\n{USAGE}");
                USAGE_ERROR
            }
        },
        ["specs", "check", rest @ ..] => match rest {
            [] => specs_check(&workspace(specs::DIR)),
            ["--dir", dir] => specs_check(Path::new(dir)),
            _ => usage(rest),
        },
        ["specs", "peaks", rig, rest @ ..] => {
            let mut file = default_inventory();
            let mut dir = workspace(specs::DIR);
            let mut options = rest.iter();
            let mut ok = true;
            while let Some(option) = options.next() {
                match (*option, options.next()) {
                    ("--file", Some(path)) => file = PathBuf::from(path),
                    ("--dir", Some(path)) => dir = PathBuf::from(path),
                    _ => ok = false,
                }
            }
            if ok {
                specs_peaks(rig, &file, &dir)
            } else {
                usage(rest)
            }
        }
        _ => {
            eprintln!("error: unknown command `{}`\n\n{USAGE}", words.join(" "));
            USAGE_ERROR
        }
    };
    let _ = io::stdout().flush();
    ExitCode::from(code)
}

/// `[--file <path>]`, or the workspace's inventory.
fn file_option(rest: &[&str]) -> Option<PathBuf> {
    match rest {
        [] => Some(default_inventory()),
        ["--file", path] => Some(PathBuf::from(path)),
        [option] if option.starts_with("--file=") => {
            Some(PathBuf::from(&option["--file=".len()..]))
        }
        _ => {
            eprintln!(
                "error: unexpected arguments `{}`\n\n{USAGE}",
                rest.join(" ")
            );
            None
        }
    }
}

/// `bench/lab/rigs.toml` in the workspace.
fn default_inventory() -> PathBuf {
    workspace(rigs::PATH)
}

/// `rel` in the workspace `cargo run` started from: the run-time `CARGO_MANIFEST_DIR` is this
/// crate's directory, two levels below the root. Without it, the path is taken from the current
/// directory.
fn workspace(rel: &str) -> PathBuf {
    let root = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .and_then(|dir| dir.ancestors().nth(2).map(Path::to_path_buf));
    match root {
        Some(root) => root.join(rel),
        None => PathBuf::from(rel),
    }
}

fn usage(rest: &[&str]) -> u8 {
    eprintln!(
        "error: unexpected arguments `{}`\n\n{USAGE}",
        rest.join(" ")
    );
    USAGE_ERROR
}

fn specs_check(dir: &Path) -> u8 {
    let (db, problems) = SpecDb::load(dir);
    for problem in &problems {
        eprintln!("specs: {problem}");
    }
    if problems.is_empty() {
        eprintln!(
            "specs: PASS: {}: {} records, every figure with a vendor or standards source \
             (RFC-0061 §5.3)",
            dir.display(),
            db.records.len()
        );
        PASS
    } else {
        eprintln!(
            "specs: FAIL: {} problems in {} (RFC-0061 §5.3)",
            problems.len(),
            dir.display()
        );
        FAIL
    }
}

fn specs_peaks(id: &str, file: &Path, dir: &Path) -> u8 {
    let Some(inventory) = load(file) else {
        return FAIL;
    };
    let Some(rig) = inventory.rig(id) else {
        eprintln!("specs: FAIL: {} has no rig `{id}`", file.display());
        return FAIL;
    };
    let (db, problems) = SpecDb::load(dir);
    for problem in &problems {
        eprintln!("specs: {problem}");
    }
    if !problems.is_empty() {
        return FAIL;
    }
    match specs::peaks(rig, &db) {
        Ok(peaks) => {
            for peak in &peaks {
                println!("{}: W_peak {:.2} GB/s", peak.path, peak.gb_per_s());
                for element in &peak.chain {
                    println!("  {:.2} GB/s  {}", element.gb_per_s, element.name);
                }
            }
            PASS
        }
        Err(problems) => {
            for problem in problems {
                eprintln!("specs: {id}: {problem}");
            }
            FAIL
        }
    }
}

/// Reads and validates the inventory at `path`, printing why it fails.
fn load(path: &Path) -> Option<Inventory> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("rigs: FAIL: cannot read {}: {error}", path.display());
            return None;
        }
    };
    match Inventory::parse(&text) {
        Ok(inventory) => Some(inventory),
        Err(invalid) => {
            for problem in &invalid.0 {
                eprintln!("rigs: {}: {problem}", path.display());
            }
            eprintln!(
                "rigs: FAIL: {} is not a valid inventory (RFC-0061 §6.3)",
                path.display()
            );
            None
        }
    }
}

fn check(path: &Path) -> u8 {
    let Some(inventory) = load(path) else {
        return FAIL;
    };
    eprintln!(
        "rigs: PASS: {}: {} rig {}, format version {} (RFC-0061 §6.3)",
        path.display(),
        inventory.rigs.len(),
        if inventory.rigs.len() == 1 {
            "record"
        } else {
            "records"
        },
        inventory.version
    );
    PASS
}

fn fingerprint(path: &Path) -> u8 {
    let Some(inventory) = load(path) else {
        return FAIL;
    };
    for rig in &inventory.rigs {
        println!("{} {}", rig.id, rig.fingerprint().0);
    }
    PASS
}

fn epochs(old: &Path, new: &Path) -> u8 {
    let (Some(old), Some(new)) = (load(old), load(new)) else {
        return FAIL;
    };
    let changes = rigs::epochs(&old, &new);
    if changes.is_empty() {
        println!("no epoch changes");
    }
    for change in changes {
        println!("{change}");
    }
    PASS
}

#[derive(Debug, PartialEq, Eq)]
enum Scenarios {
    Help,
    Check { base: Option<String>, root: PathBuf },
    Render { write: bool, root: PathBuf },
}

/// Parses the words after `scenarios`.
fn parse_scenarios(args: &[&str]) -> Result<Scenarios, String> {
    let Some(action) = args.first() else {
        return Err(String::from("`scenarios` needs `check` or `render`"));
    };
    let mut base = None;
    let mut root = None;
    let mut write = false;
    let mut rest = args[1..].iter();
    while let Some(arg) = rest.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) => (name, Some(value.to_string())),
            None => (*arg, None),
        };
        let mut value = |name: &str| -> Result<String, String> {
            inline
                .clone()
                .or_else(|| rest.next().map(|v| (*v).to_string()))
                .ok_or_else(|| format!("`{name}` needs a value"))
        };
        match (*action, name) {
            ("check", "--base") if base.is_none() => base = Some(value(name)?),
            (_, "--root") if root.is_none() => root = Some(PathBuf::from(value(name)?)),
            ("render", "--write") if !write && inline.is_none() => write = true,
            (_, "--help" | "-h") => return Ok(Scenarios::Help),
            _ => return Err(format!("`{arg}` is not an option of `scenarios {action}`")),
        }
    }
    let root = root.unwrap_or_else(|| PathBuf::from("."));
    match *action {
        "check" => Ok(Scenarios::Check { base, root }),
        "render" => Ok(Scenarios::Render { write, root }),
        "help" | "--help" | "-h" => Ok(Scenarios::Help),
        other => Err(format!("unknown command `scenarios {other}`")),
    }
}

/// Everything `check` and `render` read, from `root`.
struct Repo {
    map: sc_map::Map,
    vision: Vec<u32>,
    tasks: std::collections::BTreeMap<String, sc_map::Task>,
    /// RFC-0061's text, with LF line endings.
    rfc: String,
    /// Whether RFC-0061 has CRLF line endings on disk, which `render --write` keeps.
    rfc_crlf: bool,
}

fn read_repo(root: &Path) -> Result<Repo, String> {
    let read_raw = |rel: &str| {
        std::fs::read_to_string(root.join(rel)).map_err(|e| {
            format!(
                "cannot read {rel} under {}: {e}; run from the repository root or pass --root",
                root.display()
            )
        })
    };
    let read = |rel: &str| read_raw(rel).map(|text| sc_map::to_lf(&text));
    let rfc = read_raw(sc_map::RFC_0061)?;
    let map =
        sc_map::parse(&read(sc_map::SC_MAP)?).map_err(|e| format!("{}: {e}", sc_map::SC_MAP))?;
    Ok(Repo {
        map,
        vision: sc_map::vision_ids(&read(sc_map::VISION)?),
        tasks: sc_map::tasks(&root.join(sc_map::EPICS)),
        rfc_crlf: rfc.contains("\r\n"),
        rfc: sc_map::to_lf(&rfc),
    })
}

fn scenarios_check(root: &Path, base: Option<&str>) -> u8 {
    let repo = match read_repo(root) {
        Ok(repo) => repo,
        Err(error) => {
            eprintln!("error: {error}");
            return USAGE_ERROR;
        }
    };
    let mut failures: Vec<String> = Vec::new();

    let rigs = scenario::rig_ids(&repo.rfc);
    let (scenarios, problems) = scenario::load(root, &rigs);
    failures.extend(problems.iter().map(ToString::to_string));
    match scenario::check_versions(root, base) {
        Ok(problems) => failures.extend(problems.iter().map(ToString::to_string)),
        Err(error) => failures.push(error),
    }

    failures.extend(
        sc_map::check(&repo.map, &repo.vision, &repo.tasks)
            .into_iter()
            .map(|p| format!("{}: {p}", sc_map::SC_MAP)),
    );

    let ids: Vec<String> = scenarios.iter().map(|s| s.id.0.clone()).collect();
    let table = sc_map::render(&repo.map, &repo.tasks, &ids);
    match sc_map::section(&repo.rfc) {
        None => failures.push(format!(
            "{}: no `{}` generated section",
            sc_map::RFC_0061,
            sc_map::SECTION
        )),
        Some(body) if body != table => {
            // Name each row whose owners the epics changed, then the fix.
            failures.extend(
                sc_map::owner_drift(&body, &repo.map, &repo.tasks)
                    .into_iter()
                    .map(|d| format!("{}: {d}", sc_map::RFC_0061)),
            );
            failures.push(format!(
                "{}: the `{}` section is stale; run `syntropy-bench scenarios render --write`, \
                 or record in {}'s `[[not_owner]]` why a task that cites a criterion doesn't \
                 own it",
                sc_map::RFC_0061,
                sc_map::SECTION,
                sc_map::SC_MAP
            ));
        }
        Some(_) => {}
    }

    for failure in &failures {
        eprintln!("{failure}");
    }
    eprintln!(
        "scenarios check: {} scenario files, {} SC rows, {} problems",
        scenarios.len(),
        repo.map.sc.len(),
        failures.len()
    );
    if failures.is_empty() { PASS } else { FAIL }
}

fn scenarios_render(root: &Path, write: bool) -> u8 {
    let repo = match read_repo(root) {
        Ok(repo) => repo,
        Err(error) => {
            eprintln!("error: {error}");
            return USAGE_ERROR;
        }
    };
    let rigs = scenario::rig_ids(&repo.rfc);
    let (scenarios, _) = scenario::load(root, &rigs);
    let ids: Vec<String> = scenarios.iter().map(|s| s.id.0.clone()).collect();
    let table = sc_map::render(&repo.map, &repo.tasks, &ids);
    if !write {
        print!("{table}");
        return PASS;
    }
    let Some(updated) = sc_map::replace_section(&repo.rfc, &table) else {
        eprintln!(
            "error: {} has no `{}` generated section",
            sc_map::RFC_0061,
            sc_map::SECTION
        );
        return FAIL;
    };
    if updated == repo.rfc {
        eprintln!("{}: current", sc_map::RFC_0061);
        return PASS;
    }
    // Keep the file's line endings, so a Windows checkout's RFC is not flipped to LF.
    let updated = if repo.rfc_crlf {
        updated.replace('\n', "\r\n")
    } else {
        updated
    };
    match std::fs::write(root.join(sc_map::RFC_0061), updated) {
        Ok(()) => {
            eprintln!("wrote {}", sc_map::RFC_0061);
            PASS
        }
        Err(e) => {
            eprintln!("error: cannot write {}: {e}", sc_map::RFC_0061);
            FAIL
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The words after `scenarios`.
    fn words(line: &str) -> Vec<&str> {
        line.split_whitespace().collect()
    }

    #[test]
    fn parses_the_two_scenario_actions_and_their_options() {
        assert_eq!(
            parse_scenarios(&words("check")),
            Ok(Scenarios::Check {
                base: None,
                root: PathBuf::from(".")
            })
        );
        assert_eq!(
            parse_scenarios(&words("check --base main --root /r")),
            Ok(Scenarios::Check {
                base: Some(String::from("main")),
                root: PathBuf::from("/r")
            })
        );
        assert_eq!(
            parse_scenarios(&words("check --base=abc")),
            Ok(Scenarios::Check {
                base: Some(String::from("abc")),
                root: PathBuf::from(".")
            })
        );
        assert_eq!(
            parse_scenarios(&words("render --write")),
            Ok(Scenarios::Render {
                write: true,
                root: PathBuf::from(".")
            })
        );
        assert_eq!(parse_scenarios(&words("help")), Ok(Scenarios::Help));
    }

    #[test]
    fn rejects_unknown_scenario_actions_and_misplaced_options() {
        for bad in [
            "",
            "frob",
            "check --write",
            "render --base main",
            "check --base",
            "check --base a --base b",
        ] {
            assert!(parse_scenarios(&words(bad)).is_err(), "`{bad}` parsed");
        }
    }
}
