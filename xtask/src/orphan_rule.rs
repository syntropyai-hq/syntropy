// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `orphan-rule` step of `cargo xtask ci lint`: no parent-death signal, exit watch or
//! kill-on-close job in the code that spawns runners or runs in them (RFC-0052 §6.5;
//! RFC-0001 §3.4; DIST-E1-T04).
//!
//! A runner finds its daemon gone by end-of-file on its command channel, or by the daemon's
//! heartbeat standing still, and then follows the bounded orphan rule. Each of these would end it
//! with the daemon instead, and so rules out re-adoption across a daemon restart (RFC-0052 §6.5,
//! superseding ADR-0015 decision 6):
//!
//! - `PR_SET_PDEATHSIG` (Linux `prctl`) and `PROC_PDEATHSIG_CTL` (FreeBSD `procctl`): a signal on
//!   the parent's death;
//! - `NOTE_EXIT`, `NOTE_EXITSTATUS` and `EVFILT_PROC` (kqueue), `pidfd_open` (Linux) and `getppid`
//!   polling: a watch on the parent's exit;
//! - `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (Windows): a job that kills its processes when its last
//!   handle closes.
//!
//! [`CRATES`] lists the crates whose production code spawns a runner or runs in one. The check
//! reads the Rust tokens of every `.rs` file under each crate's `src/`, so a name in a comment or
//! a string, as in this documentation, never trips it, and an identifier anywhere does: a path, a
//! `use`, a constant passed to a call. It checks these names and nothing else: a constant written
//! as a bare number, or a mechanism reached through another crate's code, is past what a token
//! check can see, and review covers it.

use std::path::{Path, PathBuf};

use proc_macro2::{TokenStream, TokenTree};

use crate::run::Context;

/// The crates checked, by directory under `crates/`, each with why.
pub const CRATES: &[(&str, &str)] = &[
    ("runner", "the runner process's own code"),
    (
        "daemon",
        "spawns runners and holds the daemon's side (RFC-0052 §6.5)",
    ),
    ("ipc", "the command channel and region a runner inherits"),
    ("io", "the runner's clocks and I/O"),
    ("cli", "the binary's wiring, including runner mode"),
];

/// The identifiers that set up a parent-death signal, an exit watch or a kill-on-close job.
pub const FORBIDDEN: &[(&str, &str)] = &[
    ("PR_SET_PDEATHSIG", "a parent-death signal"),
    ("PROC_PDEATHSIG_CTL", "a parent-death signal"),
    ("NOTE_EXIT", "an exit watch"),
    ("NOTE_EXITSTATUS", "an exit watch"),
    ("pidfd_open", "an exit watch"),
    ("getppid", "a watch on the parent"),
    ("EVFILT_PROC", "an exit watch"),
    ("JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE", "a kill-on-close job"),
];

/// The `orphan-rule` step.
pub fn step(context: &Context) -> Result<(), String> {
    let problems = check(&context.root, CRATES);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// Every problem in `crates` under `root`.
pub fn check(root: &Path, crates: &[(&str, &str)]) -> Vec<String> {
    let mut problems = Vec::new();
    for &(name, _) in crates {
        let src = root.join("crates").join(name).join("src");
        let mut files = Vec::new();
        if let Err(error) = rust_files(&src, &mut files) {
            problems.push(format!("crates/{name}/src: a listed crate: {error}"));
            continue;
        }
        files.sort();
        for file in files {
            let shown = file
                .strip_prefix(root)
                .unwrap_or(&file)
                .display()
                .to_string()
                .replace('\\', "/");
            let text = match std::fs::read_to_string(&file) {
                Ok(text) => text,
                Err(error) => {
                    problems.push(format!("{shown}: {error}"));
                    continue;
                }
            };
            match scan(&text) {
                Ok(found) => problems.extend(found.into_iter().map(|(line, name, what)| {
                    format!(
                        "{shown}:{line}: `{name}` sets up {what}; a runner must outlive its daemon \
                         under the bounded orphan rule instead (RFC-0052 §6.5)"
                    )
                })),
                Err(error) => problems.push(format!("{shown}: not Rust tokens: {error}")),
            }
        }
    }
    problems
}

/// Every `.rs` file under `dir`.
fn rust_files(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            rust_files(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

/// Each forbidden identifier in `text`: its line, its name and what it sets up.
pub fn scan(text: &str) -> Result<Vec<(usize, &'static str, &'static str)>, String> {
    let tokens: TokenStream = text.parse().map_err(|error| format!("{error}"))?;
    let mut found = Vec::new();
    walk(tokens, &mut found);
    Ok(found)
}

fn walk(tokens: TokenStream, found: &mut Vec<(usize, &'static str, &'static str)>) {
    for token in tokens {
        match token {
            TokenTree::Ident(ident) => {
                let name = ident.to_string();
                if let Some(&(name, what)) = FORBIDDEN.iter().find(|(n, _)| *n == name) {
                    found.push((ident.span().start().line, name, what));
                }
            }
            TokenTree::Group(group) => walk(group.stream(), found),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(text: &str) -> Vec<(usize, &'static str)> {
        scan(text)
            .unwrap()
            .into_iter()
            .map(|(line, name, _)| (line, name))
            .collect()
    }

    #[test]
    fn each_mechanism_is_found_in_any_form() {
        let text = "fn a() { unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) }; }\n\
                    use libc::{EVFILT_PROC, NOTE_EXIT};\n\
                    const K: u32 = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;\n\
                    fn b() { procctl(P_PID, 0, PROC_PDEATHSIG_CTL, &mut s); }\n";
        assert_eq!(
            names(text),
            [
                (1, "PR_SET_PDEATHSIG"),
                (2, "EVFILT_PROC"),
                (2, "NOTE_EXIT"),
                (3, "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE"),
                (4, "PROC_PDEATHSIG_CTL"),
            ]
        );
    }

    #[test]
    fn names_in_comments_and_strings_are_not() {
        let text = "// no PR_SET_PDEATHSIG here\n\
                    /// nor NOTE_EXIT\n\
                    fn a() -> &'static str { \"JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE\" }\n\
                    fn b() { let job = JOB_OBJECT_LIMIT_BREAKAWAY_OK; }\n";
        assert!(names(text).is_empty(), "{:?}", scan(text));
    }

    /// The listed crates are clean, and every listed crate exists.
    #[test]
    fn the_listed_crates_follow_the_orphan_rule() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let problems = check(&root, CRATES);
        assert!(problems.is_empty(), "{problems:#?}");
    }

    #[test]
    fn a_missing_crate_is_a_problem() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let problems = check(&root, &[("gone", "")]);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].starts_with("crates/gone/src: a listed crate"));
    }
}
