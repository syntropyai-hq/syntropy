// SPDX-License-Identifier: MIT OR Apache-2.0

//! The core binary carries its own static ggml copy with every symbol hidden (RFC-0040 §8.1;
//! ADR-0403 item 6; COMP-E7-T01).
//!
//! `syntropy` must hold ggml's code, as functions of its own, so `cargo xtask ci build`'s rule 7,
//! which fails on an exported ggml symbol, checks something real. Its dynamic symbol table must
//! name none of them. On Linux and macOS the test reads the binary with the platform's `nm`; on
//! Windows an executable exports nothing unless asked, and rule 7 reads its export directory.

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::process::Command;

/// `nm`'s output for `args` and the `syntropy` binary.
fn nm(args: &[&str]) -> String {
    let output = Command::new("nm")
        .args(args)
        .arg(env!("CARGO_BIN_EXE_syntropy"))
        .output()
        .expect("run nm");
    assert!(output.status.success(), "nm {args:?}: {}", output.status);
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The ggml symbol names in an `nm` listing, with the leading `_` of a Mach-O name removed,
/// each with its `nm` type letter.
fn ggml_symbols(listing: &str) -> Vec<(char, String)> {
    listing
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace().rev();
            let name = fields.next()?;
            let kind = fields.next()?.chars().next()?;
            let name = name.strip_prefix('_').unwrap_or(name);
            (name.starts_with("ggml_") || name.starts_with("gguf_"))
                .then(|| (kind, name.to_owned()))
        })
        .collect()
}

#[test]
fn the_core_binary_holds_ggml_and_exports_none_of_it() {
    // Every symbol, so the binary's own (local) functions show: a lowercase type letter is a
    // local symbol, which no other object or library can bind.
    let all = ggml_symbols(&nm(&[]));
    let defined: Vec<&(char, String)> = all
        .iter()
        .filter(|(kind, _)| matches!(kind, 't' | 'T'))
        .collect();
    assert!(
        defined.iter().any(|(_, name)| name == "ggml_commit"),
        "the binary defines no ggml_commit, so it does not hold ggml: {all:?}"
    );
    let global: Vec<&(char, String)> = defined.iter().copied().filter(|(k, _)| *k == 'T').collect();
    assert!(
        global.is_empty(),
        "ggml functions are global, not hidden: {global:?}"
    );

    // What the dynamic linker sees.
    let exported = if cfg!(target_os = "linux") {
        nm(&["-D", "--defined-only"])
    } else {
        nm(&["-gU"])
    };
    let exported = ggml_symbols(&exported);
    assert!(
        exported.is_empty(),
        "the binary exports ggml symbols: {exported:?}"
    );
}
