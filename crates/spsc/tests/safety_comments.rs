// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every `unsafe` block and impl in this crate has a `// SAFETY:` comment directly above it,
//! stating its invariant (ADR-0001 item 4; `lib.rs` denies `clippy::undocumented_unsafe_blocks`).
//!
//! Clippy checks only the code it compiles, and the `lint` job runs on Linux, so the Apple and
//! Windows wake handles' `unsafe` blocks are never checked there. This test reads the sources as
//! text, every platform's code included, so any host that runs it checks all of them.

use std::fs;
use std::path::{Path, PathBuf};

/// Every `.rs` file under `dir`.
fn sources(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(sources(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            files.push(path);
        }
    }
    files
}

/// A line's code, without its comment.
fn code(line: &str) -> &str {
    line.split("//").next().unwrap_or("")
}

/// The `unsafe` sites in `text` (1-based lines), and those with no `// SAFETY:` in the comment
/// lines directly above them. Attribute lines between the comment and the site are skipped.
fn check(text: &str) -> (Vec<usize>, Vec<usize>) {
    let lines: Vec<&str> = text.lines().collect();
    let (mut sites, mut missing) = (Vec::new(), Vec::new());
    for (i, line) in lines.iter().enumerate() {
        let code = code(line);
        if !(code.contains("unsafe {") || code.contains("unsafe impl")) {
            continue;
        }
        sites.push(i + 1);
        let documented = lines[..i]
            .iter()
            .rev()
            .map(|l| l.trim_start())
            .take_while(|l| l.starts_with("//") || l.starts_with("#["))
            .any(|l| l.starts_with("// SAFETY:"));
        if !documented {
            missing.push(i + 1);
        }
    }
    (sites, missing)
}

#[test]
fn every_unsafe_block_has_a_safety_comment() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut sites = 0;
    let mut missing = Vec::new();
    for file in sources(&src) {
        let (found, undocumented) = check(&fs::read_to_string(&file).unwrap());
        sites += found.len();
        missing.extend(
            undocumented
                .iter()
                .map(|line| format!("{}:{line}", file.display())),
        );
    }
    assert!(
        missing.is_empty(),
        "unsafe without a SAFETY comment: {missing:?}"
    );
    // The ring's slots and every OS's wake handle: the scan is not checking nothing.
    assert!(sites >= 15, "found only {sites} unsafe sites");
}

#[test]
fn the_check_finds_a_missing_comment() {
    let (sites, missing) = check(
        "// SAFETY: fine.\nlet a = unsafe { f() };\nlet b = 1;\nlet c = unsafe { g() };\n\
         // A comment that is not one.\nunsafe impl Send for X {}\n",
    );
    assert_eq!(sites, [2, 4, 6]);
    assert_eq!(missing, [4, 6]);
}
