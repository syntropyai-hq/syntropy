// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0020 §9.1: no code path but `syntropy tune` writes `iogpu.wired_limit_mb`; the probe and
//! the tier manager only read it.
//!
//! The Testing row checks that with a syscall trace on the macOS rig. This test stands in for it
//! wherever the trace cannot run. It is a static check of the sources of every workspace crate:
//! - every `sysctlbyname` call passes a null new value and a new length of 0, so it only reads;
//! - the only production code that names a write of the sysctl, the `-w` command and the
//!   `name=value` argument, is `tier::tune::HostOp::set_wired_limit_argv`, which `tune` hands to
//!   the caller's approved runner.
//!
//! A static check sees only what the sources spell out, so it is evidence, not the trace.

use std::path::{Path, PathBuf};

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs")
            && path.file_name().is_some_and(|n| n != "tests.rs")
        {
            out.push(path);
        }
    }
}

/// The production sources: every crate's `src/`, without its unit-test modules (`tests.rs`).
fn production_sources() -> Vec<PathBuf> {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&crates).unwrap().flatten() {
        rust_sources(&entry.path().join("src"), &mut files);
    }
    files.sort();
    assert!(files.len() > 100, "found only {} sources", files.len());
    files
}

/// The arguments of the call whose opening parenthesis is at `open`, split at top-level commas.
fn call_args(text: &str, open: usize) -> Vec<String> {
    let mut depth = 0;
    let mut args = vec![String::new()];
    for c in text[open..].chars() {
        match c {
            '(' => {
                depth += 1;
                if depth == 1 {
                    continue;
                }
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            ',' if depth == 1 => {
                args.push(String::new());
                continue;
            }
            _ => {}
        }
        if let Some(last) = args.last_mut() {
            last.push(c);
        }
    }
    // A trailing comma leaves an empty last argument.
    args.into_iter()
        .map(|a| a.trim().to_owned())
        .filter(|a| !a.is_empty())
        .collect()
}

#[test]
fn every_sysctlbyname_call_only_reads() {
    let mut calls = 0;
    for path in production_sources() {
        let text = std::fs::read_to_string(&path).unwrap();
        for (at, _) in text.match_indices("sysctlbyname(") {
            let before = &text[..at];
            // The declaration, not a call.
            if before.trim_end().ends_with("fn") {
                continue;
            }
            let args = call_args(&text, at + "sysctlbyname".len());
            assert_eq!(args.len(), 5, "{}: {args:?}", path.display());
            assert!(
                args[3].contains("null"),
                "{} writes a sysctl: new value {}",
                path.display(),
                args[3]
            );
            assert_eq!(args[4], "0", "{}: new length {}", path.display(), args[4]);
            calls += 1;
        }
    }
    assert!(calls >= 1, "no sysctlbyname call found: the scan is broken");
}

#[test]
fn only_tune_spells_out_the_write() {
    let mut writers = Vec::new();
    for path in production_sources() {
        let text = std::fs::read_to_string(&path).unwrap();
        if text.contains("/usr/sbin/sysctl") || text.contains("WIRED_LIMIT_SYSCTL}=") {
            writers.push(path);
        }
    }
    let tune = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tune.rs");
    assert_eq!(
        writers
            .iter()
            .map(|p| p.canonicalize().unwrap())
            .collect::<Vec<_>>(),
        [tune.canonicalize().unwrap()]
    );
}

/// Other ways to write the sysctl that a source could spell out (review of #1822, N1): the
/// `sysctl` tool found on `PATH`, the MIB-form `sysctl()` or `sysctlnametomib()` calls, and the
/// sysctl's name as a string. None may appear outside `tier/src/tune.rs`.
#[test]
fn no_other_write_form_appears() {
    let tune = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/tune.rs")
        .canonicalize()
        .unwrap();
    let mut found = Vec::new();
    for path in production_sources() {
        if path.canonicalize().unwrap() == tune {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        for needle in ["Command::new(\"sysctl\")", "\"iogpu.wired_limit_mb"] {
            if text.contains(needle) {
                found.push(format!("{}: {needle}", path.display()));
            }
        }
        for call in ["sysctl(", "sysctlnametomib("] {
            for (at, _) in text.match_indices(call) {
                let before = &text[..at];
                let joined = before
                    .chars()
                    .last()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_');
                if !joined && !before.trim_end().ends_with("fn") {
                    found.push(format!("{}: a {call}…) call", path.display()));
                }
            }
        }
    }
    assert!(
        found.is_empty(),
        "sysctl write forms outside tune: {found:#?}"
    );
}
