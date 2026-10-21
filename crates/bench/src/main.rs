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
//!
//! Exit status: 0 on success, 1 when a check fails, 2 on a usage error.

#![forbid(unsafe_code)]

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use syntropy_bench::rigs::{self, Inventory};

const USAGE: &str = "\
Usage: syntropy-bench <command>

Commands:
  rigs check [--file <path>]        validate the rig inventory (RFC-0061 §6.3); `lint` runs it
  rigs fingerprint [--file <path>]  print each rig's RigFingerprint (RFC-0061 §2.1)
  rigs epochs <old> <new>           list the epoch changes between two inventory revisions
  help                              print this message

<path> defaults to bench/lab/rigs.toml in the workspace.";

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

/// `bench/lab/rigs.toml` in the workspace `cargo run` started from: the run-time
/// `CARGO_MANIFEST_DIR` is this crate's directory, two levels below the root. Without it, the
/// path is taken from the current directory.
fn default_inventory() -> PathBuf {
    let root = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .and_then(|dir| dir.ancestors().nth(2).map(Path::to_path_buf));
    match root {
        Some(root) => root.join(rigs::PATH),
        None => PathBuf::from(rigs::PATH),
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
