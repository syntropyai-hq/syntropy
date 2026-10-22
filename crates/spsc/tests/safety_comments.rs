// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every `unsafe` block and impl in this crate has a `// SAFETY:` comment stating its invariant
//! (ADR-0001 item 4; `lib.rs` denies `clippy::undocumented_unsafe_blocks`). As clippy reads it,
//! the comment sits in the comment lines directly above the site, attributes skipped, or above the
//! first line of the statement rustfmt wrapped the site into. The walk up to that first line stops
//! at another `unsafe` site, a match arm or a condition, so one comment never documents two sites.
//!
//! The scan is stricter than clippy in two shapes, and rejects both: a blank line between the
//! comment and the site, and a plain comment line between a wrapped statement's first line and its
//! `unsafe`.
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

/// A line's code, without its comment and with the contents of string and character literals
/// blanked, so a `//` inside a string such as `"http://…"` neither ends the code nor hides a later
/// `unsafe` on the line. Raw strings are not recognised; this crate's sources have none.
fn code(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut in_string = false;
    while let Some(c) = chars.next() {
        if in_string {
            match c {
                '\\' => {
                    chars.next();
                }
                '"' => {
                    in_string = false;
                    out.push('"');
                }
                _ => out.push(' '),
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push('"');
            }
            // A character literal such as '"' or '\'': skip it whole.
            '\'' if chars.clone().nth(1) == Some('\'') || chars.peek() == Some(&'\\') => {
                let mut rest = chars.clone();
                let close = if rest.next() == Some('\\') { 3 } else { 2 };
                for _ in 0..close {
                    chars.next();
                }
                out.push_str("' '");
            }
            '/' if chars.peek() == Some(&'/') => break,
            _ => out.push(c),
        }
    }
    out
}

/// Whether the code line `above` continues into the line below it, as a statement rustfmt wrapped
/// does: it ends neither a statement (`;`), nor a block (`}`), nor opens one (`{`).
fn continues(above: &str) -> bool {
    let code = code(above);
    let end = code.trim_end();
    !end.is_empty() && !end.ends_with([';', '{', '}'])
}

/// A line the walk up must not cross, as clippy does not: another `unsafe` site, whose comment is
/// its own; a match arm; or a condition.
fn stops(line: &str) -> bool {
    let code = code(line);
    let start = code.trim_start();
    code.contains("unsafe {")
        || code.contains("unsafe impl")
        || code.contains("=>")
        || ["if ", "while ", "match ", "} else if "]
            .iter()
            .any(|k| start.starts_with(k))
}

/// The `unsafe` sites in `text` (1-based lines), and those with no `// SAFETY:` in the comment
/// lines directly above them, or above the first line of the wrapped statement they belong to.
/// Attribute lines between the comment and the site are skipped.
fn check(text: &str) -> (Vec<usize>, Vec<usize>) {
    let lines: Vec<&str> = text.lines().collect();
    let (mut sites, mut missing) = (Vec::new(), Vec::new());
    for (i, line) in lines.iter().enumerate() {
        let code = code(line);
        if !(code.contains("unsafe {") || code.contains("unsafe impl")) {
            continue;
        }
        sites.push(i + 1);
        // As clippy does, the comment may sit above the line itself, or above the first line of
        // the statement the line belongs to, when rustfmt wrapped it (`let x =` then `unsafe {`, or
        // `unsafe` as one argument of a call laid out one argument per line).
        let mut first = i;
        let documented = loop {
            let above = lines[..first]
                .iter()
                .rev()
                .map(|l| l.trim_start())
                .take_while(|l| l.starts_with("//") || l.starts_with("#["));
            let mut above = above.peekable();
            if above.peek().is_some() {
                break above.any(|l| l.starts_with("// SAFETY:"));
            }
            match first.checked_sub(1) {
                Some(prev) if continues(lines[prev]) && !stops(lines[prev]) => first = prev,
                _ => break false,
            }
        };
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
fn the_check_reads_a_wrapped_statement_as_clippy_does() {
    // The comment above a statement's first line documents an `unsafe` rustfmt moved below it.
    let (sites, missing) = check(
        "// SAFETY: fine.\nlet x =\n    unsafe { f() };\n\
         // SAFETY: fine.\ncall(\n    a,\n    unsafe { g() },\n);\n\
         fn h() {\n    unsafe { i() };\n}\n",
    );
    assert_eq!(sites, [3, 7, 10]);
    // The last one is the first statement of a block: the line above opens it, so the comment
    // must be directly above.
    assert_eq!(missing, [10]);
    // A comment above the line that opens a block documents nothing inside it.
    let (sites, missing) = check("// SAFETY: fine.\nfn h() {\n    unsafe { i() };\n}\n");
    assert_eq!(sites, [3]);
    assert_eq!(missing, [3]);
}

#[test]
fn one_comment_documents_one_site_in_a_list_match_or_condition() {
    // Clippy flags the second site of each: the comment above the first is the first's alone.
    let (sites, missing) = check(
        "match v {\n    // SAFETY: fine.\n    0 => unsafe { f() },\n    _ => unsafe { f() },\n}\n\
         call(\n    // SAFETY: fine.\n    unsafe { f() },\n    unsafe { f() },\n);\n\
         let a = [\n    // SAFETY: fine.\n    unsafe { f() },\n    unsafe { f() },\n];\n\
         match w {\n    // SAFETY: fine.\n    0 => 1,\n    _ =>\n        unsafe { f() },\n}\n\
         // SAFETY: fine.\nif 1 == 2\n    || unsafe { f() } == 1\n{\n}\n",
    );
    assert_eq!(sites, [3, 4, 8, 9, 13, 14, 20, 24]);
    assert_eq!(missing, [4, 9, 14, 20, 24]);
}

#[test]
fn a_double_slash_in_a_string_hides_nothing() {
    let (sites, missing) = check(
        "let u = \"http://x\"; let y = unsafe { h() };\n\
         // SAFETY: fine.\nlet q = '\"'; let z = unsafe { k() };\n\
         let w = \"\\\"//\\\"\"; let v = unsafe { m() };\n\
         let t = \"unsafe { no }\";\n",
    );
    // An escaped quote does not end the string, so the `//` after it is inside it too. A string
    // that spells `unsafe {` is not a site.
    assert_eq!(sites, [1, 3, 4]);
    assert_eq!(missing, [1, 4]);
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

#[test]
fn an_attribute_between_comment_lines_is_skipped() {
    // The comment block above the site runs on past the attribute to the `// SAFETY:` line.
    let (sites, missing) = check(
        "// SAFETY: fine.\n#[cfg(unix)]\n// More about it.\nunsafe impl Send for Y {}\n\
         #[cfg(unix)]\n// More about it.\nunsafe impl Sync for Y {}\n",
    );
    assert_eq!(sites, [4, 7]);
    assert_eq!(missing, [7]);
}
