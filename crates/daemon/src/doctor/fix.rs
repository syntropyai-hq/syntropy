// SPDX-License-Identifier: MIT OR Apache-2.0
//! `doctor --fix` (RFC-0053 §5.1): only automatic fixes, asked for one at a time, every host change
//! recorded in the host-change ledger (RFC-0052 §11.1) before it is made, and the revert scripts
//! rewritten afterwards, so `uninstall` reverts it whether or not a Syntropy binary is left.

use std::fmt;
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Component, Path, PathBuf};

use serde::Serialize;

use super::{AutoFix, DoctorCtx, Fix, FixSetup, HostEdit, Registry, Report, Status, run_check};
use crate::host_changes::LedgerError;
use crate::host_changes::script::write_revert_script;

/// The answer to "apply this fix?".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answer {
    /// Apply it.
    Yes,
    /// Leave it, for this reason.
    No {
        /// Why, as the report prints it.
        why: String,
    },
}

/// Asks before each fix (RFC-0053 §5.1: on a terminal, `--yes` otherwise).
pub trait Consent {
    /// Whether to apply `fix`, the fix check `id` offers, whose edits are `auto`.
    fn ask(&mut self, id: &str, fix: &Fix, auto: &AutoFix) -> Answer;
}

/// `--yes`: every automatic fix is applied without asking.
#[derive(Clone, Copy, Debug, Default)]
pub struct AssumeYes;

impl Consent for AssumeYes {
    fn ask(&mut self, _id: &str, _fix: &Fix, _auto: &AutoFix) -> Answer {
        Answer::Yes
    }
}

/// No terminal to ask on and no `--yes`: every fix is left, and the report says how to apply it.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoTerminal;

impl Consent for NoTerminal {
    fn ask(&mut self, _id: &str, _fix: &Fix, _auto: &AutoFix) -> Answer {
        Answer::No {
            why: "no terminal to ask on; run on a terminal, or pass --yes".into(),
        }
    }
}

/// Asks on a terminal: prints the fix and its edits to `output` and reads one line from `input`.
/// Only `y` or `yes`, in any case, applies it; anything else, or the end of input, leaves it.
#[derive(Debug)]
pub struct Prompt<R, W> {
    input: R,
    output: W,
}

impl<R: BufRead, W: Write> Prompt<R, W> {
    /// A prompt that reads answers from `input` and writes questions to `output`.
    pub fn new(input: R, output: W) -> Self {
        Self { input, output }
    }
}

impl<R: BufRead, W: Write> Consent for Prompt<R, W> {
    fn ask(&mut self, id: &str, fix: &Fix, auto: &AutoFix) -> Answer {
        let mut question = format!("{id}: {}\n", fix.description);
        for edit in &auto.edits {
            question.push_str(&format!("  {edit}\n"));
            let HostEdit::WriteFile { contents, .. } = edit;
            // Short text is shown in full, so the user sees what is written.
            if let Ok(text) = std::str::from_utf8(contents)
                && contents.len() <= SHOWN_BYTES
            {
                for line in text.lines() {
                    question.push_str(&format!("    | {}\n", printable(line)));
                }
            }
        }
        question.push_str("  `syntropy uninstall` reverts it. Apply this fix? [y/N] ");
        let mut line = String::new();
        let asked = self
            .output
            .write_all(question.as_bytes())
            .and_then(|()| self.output.flush())
            .and_then(|()| self.input.read_line(&mut line));
        match asked {
            Ok(_) if matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes") => {
                Answer::Yes
            }
            Ok(0) => Answer::No {
                why: "no answer (end of input)".into(),
            },
            Ok(_) => Answer::No {
                why: "declined".into(),
            },
            Err(e) => Answer::No {
                why: format!("could not ask: {e}"),
            },
        }
    }
}

/// `line` with every control character but a tab escaped, so file content shown in the prompt
/// cannot move the cursor, clear the screen or hide text from the user.
fn printable(line: &str) -> String {
    line.chars()
        .map(|c| {
            if c.is_control() && c != '\t' {
                c.escape_default().to_string()
            } else {
                c.to_string()
            }
        })
        .collect()
}

/// The largest file content the prompt shows in full; a longer file is shown by its size only.
pub const SHOWN_BYTES: usize = 2048;

/// The files `--fix` may write, each a pattern of path components under a base directory: the
/// host's root, or the user's home. `*.conf` matches one component ending in `.conf`, and
/// `*.service.d` one ending in `.service.d`.
const UNDER_ROOT: &[&[&str]] = &[
    // A pluggable-authentication limits file, such as the memlock limit (OS-001).
    &["etc", "security", "limits.d", "*.conf"],
    // A drop-in for a systemd system service, such as `LimitMEMLOCK=` for the daemon's unit.
    &["etc", "systemd", "system", "*.service.d", "*.conf"],
];

/// As [`UNDER_ROOT`], under the user's home.
const UNDER_HOME: &[&[&str]] = &[
    // A drop-in for a systemd user service.
    &[".config", "systemd", "user", "*.service.d", "*.conf"],
];

/// Where `--fix` may write (RFC-0053 §5.1). Every [`HostEdit`] is checked against this list before
/// anyone is asked, and an edit outside it is refused with no ledger entry. Adding a pattern is a
/// reviewed change to this file.
#[derive(Clone, Debug)]
pub struct Targets {
    root: PathBuf,
    home: Option<PathBuf>,
}

impl Targets {
    /// The host's own files: under `/`, and under `home` when the user has one.
    ///
    /// The patterns are Linux paths, so on macOS (where `/etc` links to `/private/etc`) and on
    /// Windows every edit is refused: no automatic fix exists there yet.
    #[must_use]
    pub fn host(home: Option<&Path>) -> Self {
        Self::under(Path::new("/"), home)
    }

    /// The same patterns under another root, such as a test's directory.
    #[must_use]
    pub fn under(root: &Path, home: Option<&Path>) -> Self {
        Self {
            root: root.to_owned(),
            home: home.map(Path::to_owned),
        }
    }

    /// Why `edit` may not be made, or `None` when it may.
    #[must_use]
    pub fn refuse(&self, edit: &HostEdit) -> Option<String> {
        let HostEdit::WriteFile { path, .. } = edit;
        if !path.is_absolute() {
            return Some(format!("{} is not an absolute path", path.display()));
        }
        // `.` and `..` would let a path that matches a pattern name another file.
        if path
            .components()
            .any(|c| matches!(c, Component::CurDir | Component::ParentDir))
        {
            return Some(format!("{} has a `.` or `..` component", path.display()));
        }
        let bases = [
            (Some(&self.root), UNDER_ROOT),
            (self.home.as_ref(), UNDER_HOME),
        ];
        for (base, patterns) in bases {
            let Some(base) = base else { continue };
            let Ok(rest) = path.strip_prefix(base) else {
                continue;
            };
            let parts: Vec<_> = rest.components().map(|c| c.as_os_str()).collect();
            let matched = patterns.iter().any(|pattern| {
                pattern.len() == parts.len()
                    && pattern
                        .iter()
                        .zip(&parts)
                        .all(|(want, got)| matches_part(want, &got.to_string_lossy()))
            });
            if matched {
                return linked_parent(base, path);
            }
        }
        Some(format!(
            "{} is not a file doctor --fix may write: only a `limits.d` file or a systemd service \
             drop-in",
            path.display()
        ))
    }
}

/// Whether one path component matches one pattern component.
fn matches_part(want: &str, got: &str) -> bool {
    match want.strip_prefix('*') {
        Some(suffix) => got.len() > suffix.len() && got.ends_with(suffix),
        None => got == want,
    }
}

/// Why `path`'s directory is not where its name says, or `None` when it is: a link among the
/// directories under `base` would send the write somewhere the pattern does not allow, such as a
/// `limits.d` that links to `sysctl.d`. The nearest directory that exists is resolved and must be
/// the same directory under the resolved base.
fn linked_parent(base: &Path, path: &Path) -> Option<String> {
    let mut dir = path.parent()?;
    while !dir.exists() {
        dir = dir.parent()?;
    }
    let Ok(rest) = dir.strip_prefix(base) else {
        return Some(format!("{} does not exist", base.display()));
    };
    match (fs::canonicalize(base), fs::canonicalize(dir)) {
        (Ok(base_real), Ok(dir_real)) if dir_real == base_real.join(rest) => None,
        (Ok(_), Ok(dir_real)) => Some(format!(
            "{} resolves to {}, outside the files doctor --fix may write",
            dir.display(),
            dir_real.display()
        )),
        (Err(e), _) | (_, Err(e)) => Some(format!("{}: {e}", dir.display())),
    }
}

/// Why a fix could not finish: the revert scripts could not be written after a change. `lines`
/// reports the fixes, including the ones that were made.
#[derive(Debug)]
pub struct FixError {
    /// The fixes offered so far.
    pub lines: Vec<FixLine>,
    /// The ledger's error.
    pub error: LedgerError,
}

/// How one offered fix ended.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    /// Every edit was made; the ledger entries' IDs.
    Applied {
        /// The ledger entries, in edit order.
        changes: Vec<u32>,
    },
    /// Not offered: an edit writes a file [`Targets`] does not allow. Nothing was asked or
    /// recorded.
    Refused {
        /// Why.
        why: String,
    },
    /// Left, for this reason.
    Declined {
        /// The consent's reason.
        why: String,
    },
    /// An edit failed. The edits before it were made and are in the ledger.
    Failed {
        /// The error.
        error: String,
        /// The ledger entries of the edits that were made.
        changes: Vec<u32>,
    },
}

/// One offered fix in the `--fix` report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FixLine {
    /// The check that offered it.
    pub id: &'static str,
    /// The fix's description.
    pub description: String,
    /// How it ended.
    #[serde(flatten)]
    pub outcome: Outcome,
}

impl fmt::Display for FixLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ids = |changes: &[u32]| {
            changes
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        match &self.outcome {
            Outcome::Applied { changes } => write!(
                f,
                "fixed {}: {} (host-change ledger entries {})",
                self.id,
                self.description,
                ids(changes)
            ),
            Outcome::Refused { why } => write!(f, "fix refused {}: {why}", self.id),
            Outcome::Declined { why } => write!(f, "not fixed {}: {why}", self.id),
            Outcome::Failed { error, changes } if changes.is_empty() => {
                write!(f, "fix failed {}: {error}", self.id)
            }
            Outcome::Failed { error, changes } => write!(
                f,
                "fix failed {}: {error}; the edits made before it are host-change ledger entries {}",
                self.id,
                ids(changes)
            ),
        }
    }
}

/// Offers the automatic fix of every failing or warning check in `report`, applies the accepted
/// ones through the ledger, rewrites the revert scripts when anything changed, and runs the fixed
/// checks again, replacing their results in `report`. A fix with an edit outside `targets` is
/// refused before it is offered.
///
/// # Errors
///
/// [`FixError`] if the revert scripts cannot be written after a change. An edit that fails is
/// reported as [`Outcome::Failed`], and the other fixes are still offered.
pub fn apply(
    registry: &Registry,
    ctx: &DoctorCtx,
    report: &mut Report,
    setup: FixSetup<'_>,
) -> Result<Vec<FixLine>, FixError> {
    let FixSetup {
        ledger,
        consent,
        targets,
        at,
        by_version,
    } = setup;
    let mut lines = Vec::new();
    let mut changed = false;
    for result in &report.results {
        let finding = &result.finding;
        if !matches!(finding.status, Status::Warn | Status::Fail) {
            continue;
        }
        let Some(fix) = &finding.fix else { continue };
        let Some(auto) = &fix.automatic else { continue };
        let refused = auto.edits.iter().find_map(|edit| targets.refuse(edit));
        let answer = match refused {
            Some(why) => Err(Outcome::Refused { why }),
            None => Ok(consent.ask(result.id, fix, auto)),
        };
        let outcome = match answer {
            Err(refused) => refused,
            Ok(Answer::No { why }) => Outcome::Declined { why },
            Ok(Answer::Yes) => {
                let mut changes = Vec::new();
                let mut error = None;
                for edit in &auto.edits {
                    match edit.apply(ledger, at, by_version) {
                        Ok(id) => changes.push(id),
                        Err(e) => {
                            // An I/O error can come after the entry was recorded, leaving it
                            // pending, which the revert scripts must then cover.
                            changed |= matches!(e, LedgerError::Io { .. });
                            error = Some(format!("{edit}: {e}"));
                            break;
                        }
                    }
                }
                changed |= !changes.is_empty();
                match error {
                    None => Outcome::Applied { changes },
                    Some(error) => Outcome::Failed { error, changes },
                }
            }
        };
        lines.push(FixLine {
            id: result.id,
            description: fix.description.clone(),
            outcome,
        });
    }
    if changed && let Err(error) = write_revert_script(ledger, by_version) {
        return Err(FixError { lines, error });
    }
    // The fixed checks run again, so the report shows the host as it is now.
    for line in &lines {
        if matches!(
            line.outcome,
            Outcome::Declined { .. } | Outcome::Refused { .. }
        ) {
            continue;
        }
        let check = registry
            .get(line.id)
            .expect("a fixed check came from the registry");
        let rerun = run_check(check, ctx);
        if let Some(slot) = report.results.iter_mut().find(|r| r.id == line.id) {
            *slot = rerun;
        }
    }
    Ok(lines)
}

/// The `--json` output of a `--fix` run that stopped because the revert scripts could not be
/// written: the fixes offered so far, under `fixes`, and the error.
#[must_use]
pub fn json_error(fixes: &[FixLine], error: &LedgerError) -> String {
    #[derive(Serialize)]
    struct Stopped<'a> {
        fixes: &'a [FixLine],
        error: String,
    }
    serde_json::to_string_pretty(&Stopped {
        fixes,
        error: error.to_string(),
    })
    .expect("fix lines always serialize")
}

/// The `--json` output of a `--fix` run: the report, with the offered fixes under `fixes`.
#[must_use]
pub fn json_with_fixes(report: &Report, fixes: &[FixLine]) -> String {
    #[derive(Serialize)]
    struct WithFixes<'a> {
        #[serde(flatten)]
        report: &'a Report,
        fixes: &'a [FixLine],
    }
    serde_json::to_string_pretty(&WithFixes { report, fixes }).expect("a report always serializes")
}
