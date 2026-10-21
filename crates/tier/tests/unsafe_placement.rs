// SPDX-License-Identifier: MIT OR Apache-2.0

//! `unsafe` placement in this crate (RFC-0020 §1, RFC-0001 §4.2 rule 8).
//!
//! The crate root denies `unsafe_code`. Every module except `arena` and `os` must also carry
//! `#![forbid(unsafe_code)]`, so that no scoped `allow` can bring `unsafe` into it. This test
//! stands in until `cargo xtask check-deps` checks `forbid(unsafe_code)` placement
//! (QUAL-E1-T03).

use std::path::{Path, PathBuf};

/// Modules where RFC-0020 §1 allows `unsafe`, and the crate root, which uses `deny`.
const EXEMPT: [&str; 3] = ["lib.rs", "arena.rs", "os.rs"];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read src dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn has_inner_attr(text: &str, attr: &str) -> bool {
    text.lines().any(|l| l.trim() == attr)
}

#[test]
fn every_module_but_arena_and_os_forbids_unsafe() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    files.sort();
    assert!(files.len() > EXEMPT.len(), "found only {files:?}");

    let mut missing = Vec::new();
    for path in &files {
        let rel = path.strip_prefix(&src).unwrap();
        // A module directory's files (arena/…, os/…) share their parent's exemption.
        let top = rel
            .components()
            .next()
            .unwrap()
            .as_os_str()
            .to_string_lossy();
        let top_file = if rel.components().count() > 1 {
            format!("{top}.rs")
        } else {
            top.into_owned()
        };
        if EXEMPT.contains(&top_file.as_str()) {
            continue;
        }
        let text = std::fs::read_to_string(path).unwrap();
        if !has_inner_attr(&text, "#![forbid(unsafe_code)]") {
            missing.push(rel.display().to_string());
        }
    }
    assert!(
        missing.is_empty(),
        "modules without #![forbid(unsafe_code)]: {missing:?}"
    );
}

#[test]
fn crate_root_denies_unsafe() {
    let lib = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    let text = std::fs::read_to_string(lib).unwrap();
    assert!(has_inner_attr(&text, "#![deny(unsafe_code)]"));
}
