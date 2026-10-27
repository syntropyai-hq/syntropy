// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `hot-threads` step of `cargo xtask ci lint`: no `tracing` in hot-thread modules (RFC-0053
//! §2.1 and Testing; ADR-0012; PROD-E8-T01).
//!
//! Hot threads record into per-thread cells and SPSC rings. The `tracing` crate's spans and
//! events dispatch to subscribers, may allocate and take locks, so they stay out of every module
//! that runs on a hot thread (RFC-0053, Alternatives). [`MODULES`] lists those modules; a task
//! that adds a hot-thread module adds it here.
//!
//! The check reads each file's Rust tokens, so no comment or string literal trips it, and it
//! rejects:
//! - the identifier `tracing` anywhere: a path, a `use`, an attribute;
//! - an invocation of one of `tracing`'s macros, [`MACROS`], by its bare name, as a `use`
//!   would allow;
//! - `#[instrument]`, `tracing`'s attribute macro, by its bare name.

use std::path::Path;

use proc_macro2::{Delimiter, TokenStream, TokenTree};

use crate::run::Context;

/// The modules that run on hot threads, relative to the workspace root, each with why.
pub const MODULES: &[(&str, &str)] = &[
    (
        "crates/obs/src/metrics/local.rs",
        "per-thread metric cells (RFC-0053 §2.1)",
    ),
    (
        "crates/obs/src/metrics/histogram.rs",
        "histogram bucket selection (RFC-0053 §2.1)",
    ),
    ("crates/obs/src/ring.rs", "step-ring writes (RFC-0053 §1)"),
    (
        "crates/obs/src/log.rs",
        "hot-thread log records (RFC-0053 §4.1)",
    ),
    (
        "crates/obs/src/timeline.rs",
        "hot-thread timing (RFC-0053 §2.1)",
    ),
];

/// `tracing`'s function-like macros.
pub const MACROS: &[&str] = &[
    "trace",
    "debug",
    "info",
    "warn",
    "error",
    "event",
    "span",
    "trace_span",
    "debug_span",
    "info_span",
    "warn_span",
    "error_span",
    "enabled",
];

/// The `hot-threads` step.
pub fn step(context: &Context) -> Result<(), String> {
    let problems = check(&context.root, MODULES);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// Every problem in the hot-thread modules under `root`.
pub fn check(root: &Path, modules: &[(&str, &str)]) -> Vec<String> {
    let mut problems = Vec::new();
    for &(path, _) in modules {
        let text = match std::fs::read_to_string(root.join(path)) {
            Ok(text) => text,
            Err(error) => {
                problems.push(format!("{path}: a listed hot-thread module: {error}"));
                continue;
            }
        };
        match scan(&text) {
            Ok(found) => problems.extend(
                found
                    .into_iter()
                    .map(|(line, what)| format!("{path}:{line}: {what} on a hot thread")),
            ),
            Err(error) => problems.push(format!("{path}: not Rust tokens: {error}")),
        }
    }
    problems
}

/// Each use of `tracing` in `text`: its line and what it is.
pub fn scan(text: &str) -> Result<Vec<(usize, String)>, String> {
    let tokens: TokenStream = text.parse().map_err(|error| format!("{error}"))?;
    let mut found = Vec::new();
    walk(tokens, false, &mut found);
    Ok(found)
}

/// Scans `tokens`; `attribute` is set inside a `#[...]` group.
fn walk(tokens: TokenStream, attribute: bool, found: &mut Vec<(usize, String)>) {
    let tokens: Vec<TokenTree> = tokens.into_iter().collect();
    for (at, token) in tokens.iter().enumerate() {
        match token {
            TokenTree::Ident(ident) => {
                let name = ident.to_string();
                let line = ident.span().start().line;
                let bang =
                    matches!(tokens.get(at + 1), Some(TokenTree::Punct(p)) if p.as_char() == '!');
                if name == "tracing" {
                    found.push((line, "`tracing`".to_owned()));
                } else if bang && MACROS.contains(&name.as_str()) {
                    found.push((line, format!("`{name}!`, a `tracing` macro")));
                } else if attribute && at == 0 && name == "instrument" {
                    found.push((line, "`#[instrument]`, a `tracing` attribute".to_owned()));
                }
            }
            TokenTree::Group(group) => {
                let after_hash = at > 0
                    && matches!(&tokens[at - 1], TokenTree::Punct(p) if p.as_char() == '#')
                    || at > 1
                        && matches!(&tokens[at - 1], TokenTree::Punct(p) if p.as_char() == '!')
                        && matches!(&tokens[at - 2], TokenTree::Punct(p) if p.as_char() == '#');
                let inside = group.delimiter() == Delimiter::Bracket && after_hash;
                walk(group.stream(), inside, found);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<usize> {
        scan(text)
            .unwrap()
            .into_iter()
            .map(|(line, _)| line)
            .collect()
    }

    #[test]
    fn tracing_in_any_form_is_found() {
        let text = "use tracing::info;\n\
                    fn a() { info!(\"x\"); }\n\
                    #[instrument]\n\
                    fn b() { tracing::warn!(y = 1); }\n\
                    fn c() { let _s = debug_span!(\"s\"); }\n\
                    #[tracing::instrument(skip(self))]\n\
                    fn d() {}\n";
        // Line 4 has two: the path and the macro.
        assert_eq!(lines(text), [1, 2, 3, 4, 4, 5, 6]);
    }

    #[test]
    fn words_in_comments_strings_and_other_macros_are_not() {
        let text = "// tracing is not used here: info!(...)\n\
                    /// No `#[instrument]` either.\n\
                    fn a() { let s = \"tracing info! warn!\"; debug_assert!(s.len() > 0); }\n\
                    fn error() -> u8 { let info = 1; info }\n\
                    #[inline] fn b() { assert!(true); }\n";
        assert!(lines(text).is_empty(), "{:?}", scan(text));
    }

    /// The listed modules are clean, and every listed file exists.
    #[test]
    fn the_hot_thread_modules_have_no_tracing() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let problems = check(&root, MODULES);
        assert!(problems.is_empty(), "{problems:#?}");
    }

    #[test]
    fn a_missing_module_is_a_problem() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let problems = check(&root, &[("crates/obs/src/gone.rs", "")]);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].starts_with("crates/obs/src/gone.rs: a listed hot-thread module"));
    }
}
