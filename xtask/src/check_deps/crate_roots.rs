// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rule 8 of RFC-0001 §4.2 (ADR-0001 item 4): every crate outside the `unsafe` allow-list sets
//! `#![forbid(unsafe_code)]`, except a crate with a module-scoped exception
//! ([`policy::UNSAFE_MODULES`]), which [`super::unsafe_scope`] checks instead.
//!
//! The check reads the crate root of each library and binary target and looks for the attribute
//! among its leading inner attributes, which is the only place an inner attribute can apply to the
//! whole crate. It skips comments and doc comments there. A `deny` can be overridden and a
//! `cfg_attr` is conditional, so neither counts. Tests, benches, examples and build scripts are
//! not production code, and are not checked.

use std::path::Path;

use super::graph::Graph;
use super::policy;
use super::{Finding, Rule};

/// Target kinds whose crate root is production code.
const PRODUCTION_KINDS: &[&str] = &[
    "lib",
    "rlib",
    "dylib",
    "cdylib",
    "staticlib",
    "proc-macro",
    "bin",
];

/// Every member outside the allow-list, every production target.
pub fn check(graph: &Graph) -> Vec<Finding> {
    let mut findings = Vec::new();
    for at in graph.members() {
        let name = graph.short(at);
        if policy::allows_unsafe(name) {
            continue;
        }
        if let Some(module) = policy::unsafe_module(name) {
            findings.extend(super::unsafe_scope::check(graph, at, module));
            continue;
        }
        for target in &graph.package(at).targets {
            if !target
                .kind
                .iter()
                .any(|kind| PRODUCTION_KINDS.contains(&kind.as_str()))
            {
                continue;
            }
            let shown = shown_path(&graph.root, &target.src_path);
            let kind = target.kind.join(", ");
            match std::fs::read_to_string(&target.src_path) {
                Ok(source) if forbids_unsafe_code(&source) => {}
                Ok(_) => findings.push(Finding::new(
                    Rule::Numbered(8),
                    format!(
                        "`{name}` does not set `#![forbid(unsafe_code)]` in {shown} ({kind} target \
                         `{}`); only `kernels-*`, `io`, `tier`, `net`, `ipc` and `spsc` may use `unsafe`, \
                         `testkit` only in its `GlobalAlloc` impl, and `kernels` only in declaring \
                         its listed `unsafe` trait methods",
                        target.name
                    ),
                )),
                Err(error) => findings.push(Finding::new(
                    Rule::Numbered(8),
                    format!("cannot read {shown}, the crate root of `{name}`: {error}"),
                )),
            }
        }
    }
    findings
}

/// `path` relative to the workspace root, with `/` between components on every OS, so a message
/// reads the same in every CI log. A path outside the root is shown whole.
pub fn shown_path(root: &Path, path: &Path) -> String {
    match path.strip_prefix(root) {
        Ok(relative) => {
            let parts: Vec<_> = relative
                .components()
                .map(|part| part.as_os_str().to_string_lossy())
                .collect();
            parts.join("/")
        }
        Err(_) => path.display().to_string(),
    }
}

/// Whether the crate root `source` sets `#![forbid(unsafe_code)]` among its leading inner
/// attributes.
pub fn forbids_unsafe_code(source: &str) -> bool {
    leading_attributes(source)
        .0
        .iter()
        .any(|body| sets_unsafe_code(body, "forbid"))
}

/// Whether the crate root `source` sets `#![deny(unsafe_code)]` or `#![forbid(unsafe_code)]`
/// among its leading inner attributes: the level a module-scoped exception needs.
pub fn denies_unsafe_code(source: &str) -> bool {
    leading_attributes(source)
        .0
        .iter()
        .any(|body| sets_unsafe_code(body, "deny") || sets_unsafe_code(body, "forbid"))
}

/// The insides of the leading inner attributes of `source`, and the byte offset where they end.
/// Reading stops at the first thing that is not one, or at one that does not close.
pub fn leading_attributes(source: &str) -> (Vec<&str>, usize) {
    let mut rest = source.strip_prefix('\u{feff}').unwrap_or(source);
    // A shebang line may open the file: `#!` whose next token, past whitespace and comments as
    // rustc skips them, is not `[`.
    if let Some(after) = rest.strip_prefix("#!")
        && !skip_trivia(after).is_some_and(|next| next.starts_with('['))
    {
        rest = after.split_once('\n').map_or("", |(_, next)| next);
    }
    let mut bodies = Vec::new();
    while let Some(next) = skip_trivia(rest) {
        let Some(after) = next.strip_prefix("#!") else {
            break;
        };
        let Some(after) = skip_trivia(after).and_then(|after| after.strip_prefix('[')) else {
            break;
        };
        let Some((body, after)) = split_bracketed(after) else {
            break;
        };
        bodies.push(body);
        rest = after;
    }
    (bodies, source.len() - rest.len())
}

/// `text` after any whitespace, line comments and (nested) block comments. `None` when a block
/// comment does not close.
fn skip_trivia(mut text: &str) -> Option<&str> {
    loop {
        text = text.trim_start();
        if let Some(comment) = text.strip_prefix("//") {
            text = comment.split_once('\n').map_or("", |(_, next)| next);
        } else if text.starts_with("/*") {
            let mut depth = 0usize;
            let mut chars = text.char_indices().peekable();
            let mut end = None;
            while let Some((at, c)) = chars.next() {
                match (c, chars.peek().map(|(_, next)| *next)) {
                    ('/', Some('*')) => {
                        depth += 1;
                        chars.next();
                    }
                    ('*', Some('/')) => {
                        depth -= 1;
                        chars.next();
                        if depth == 0 {
                            end = Some(at + 2);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            text = &text[end?..];
        } else {
            return Some(text);
        }
    }
}

/// Splits `text`, which follows an opening `[`, at its matching `]`, skipping brackets inside
/// string literals. Returns the inside and the text after the `]`.
fn split_bracketed(text: &str) -> Option<(&str, &str)> {
    let mut depth = 1usize;
    let mut chars = text.char_indices();
    while let Some((at, c)) = chars.next() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&text[..at], &text[at + 1..]));
                }
            }
            '"' => loop {
                match chars.next()? {
                    (_, '\\') => {
                        chars.next()?;
                    }
                    (_, '"') => break,
                    _ => {}
                }
            },
            _ => {}
        }
    }
    None
}

/// Whether an attribute's inside is `<level>(..)`, such as `forbid(..)`, with `unsafe_code` among
/// its lints.
fn sets_unsafe_code(attribute: &str, level: &str) -> bool {
    let Some(args) = attribute.trim().strip_prefix(level) else {
        return false;
    };
    let Some(args) = args
        .trim_start()
        .strip_prefix('(')
        .and_then(|args| args.trim_end().strip_suffix(')'))
    else {
        return false;
    };
    args.split(',').any(|lint| lint.trim() == "unsafe_code")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shebang_is_told_from_an_inner_attribute_past_comments() {
        // rustc skips whitespace and comments after `#!` before deciding.
        let attribute = "#! /* a note */ [forbid(unsafe_code)]\npub fn f() {}\n";
        assert_eq!(leading_attributes(attribute).0, ["forbid(unsafe_code)"]);
        let shebang = "#!/usr/bin/env run\n#![forbid(unsafe_code)]\n";
        assert_eq!(leading_attributes(shebang).0, ["forbid(unsafe_code)"]);
    }

    #[test]
    fn finds_the_attribute_after_comments_and_other_attributes() {
        for source in [
            "#![forbid(unsafe_code)]\n",
            "// SPDX-License-Identifier: MIT OR Apache-2.0\n\n//! Docs.\n\n#![forbid(unsafe_code)]\n",
            "/* a /* nested */ block */ #![forbid(unsafe_code)]",
            "#![doc = \"a ] in a string\"]\n#![forbid(unsafe_code)]\nfn main() {}",
            "#![deny(missing_docs)]\n#![forbid(missing_docs, unsafe_code)]\n",
            "#![forbid( unsafe_code , )]",
            "#! [forbid(unsafe_code)]",
            "\u{feff}#![forbid(unsafe_code)]",
            "#!/usr/bin/env run-cargo-script\n#![forbid(unsafe_code)]\n",
            "#![cfg_attr(docsrs, feature(doc_cfg))]\n#![forbid(unsafe_code)]",
        ] {
            assert!(forbids_unsafe_code(source), "{source:?}");
        }
    }

    #[test]
    fn rejects_a_missing_weak_conditional_or_misplaced_attribute() {
        for source in [
            "",
            "fn main() {}\n",
            "// #![forbid(unsafe_code)]\n",
            "/* #![forbid(unsafe_code)] */\n",
            "#![deny(unsafe_code)]\n",
            "#![cfg_attr(not(test), forbid(unsafe_code))]\n",
            "#![forbid(unsafe_op_in_unsafe_fn)]\n",
            "#[forbid(unsafe_code)]\nfn f() {}\n",
            "mod a;\n#![forbid(unsafe_code)]\n",
            "#![doc = \"#![forbid(unsafe_code)]\"]\n",
            "/* never closed #![forbid(unsafe_code)]",
            "#![forbid(unsafe_code\n",
            "#![doc = \"never closed]\n",
        ] {
            assert!(!forbids_unsafe_code(source), "{source:?}");
        }
    }

    #[test]
    fn deny_counts_only_for_the_module_scoped_exception() {
        assert!(denies_unsafe_code("#![deny(unsafe_code)]\n"));
        assert!(denies_unsafe_code(
            "//! Docs.\n#![deny(missing_docs, unsafe_code)]\n"
        ));
        assert!(denies_unsafe_code("#![forbid(unsafe_code)]\n"));
        assert!(!forbids_unsafe_code("#![deny(unsafe_code)]\n"));
        for source in [
            "",
            "#![deny(unsafe_op_in_unsafe_fn)]\n",
            "#![cfg_attr(not(test), deny(unsafe_code))]\n",
            "// #![deny(unsafe_code)]\n",
            "mod guards;\n#![deny(unsafe_code)]\n",
        ] {
            assert!(!denies_unsafe_code(source), "{source:?}");
        }
    }

    #[test]
    fn leading_attributes_end_after_the_last_one() {
        let source = "//! Docs.\n\n#![deny(unsafe_code)]\n#![doc = \"x\"]\n\npub mod guards;\n";
        let (bodies, end) = leading_attributes(source);
        assert_eq!(bodies, ["deny(unsafe_code)", "doc = \"x\""]);
        assert!(source[end..].trim_start().starts_with("pub mod guards;"));
    }

    #[test]
    fn paths_are_shown_relative_with_forward_slashes() {
        let root = Path::new("w");
        let path: std::path::PathBuf = ["w", "crates", "graph", "src", "lib.rs"].iter().collect();
        assert_eq!(shown_path(root, &path), "crates/graph/src/lib.rs");
        let outside = Path::new("elsewhere");
        assert_eq!(shown_path(root, outside), "elsewhere");
    }

    #[test]
    fn every_real_crate_root_outside_the_allow_list_forbids_unsafe_code() {
        // The same scan `check` runs, over the stubs' lib.rs and cli's main.rs, without cargo.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let mut checked = 0;
        for entry in std::fs::read_dir(root.join("crates")).unwrap() {
            let dir = entry.unwrap().path();
            let name = dir.file_name().unwrap().to_str().unwrap().to_owned();
            for file in ["src/lib.rs", "src/main.rs"] {
                let Ok(source) = std::fs::read_to_string(dir.join(file)) else {
                    continue;
                };
                checked += 1;
                let exception = policy::unsafe_module(&name).is_some();
                assert!(
                    policy::allows_unsafe(&name)
                        || forbids_unsafe_code(&source)
                        || (exception && denies_unsafe_code(&source)),
                    "{name}/{file}"
                );
            }
        }
        assert!(checked >= 30, "checked {checked} crate roots");
    }
}
