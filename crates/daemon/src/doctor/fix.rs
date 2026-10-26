// SPDX-License-Identifier: MIT OR Apache-2.0
//! `doctor --fix` (RFC-0053 §5.1): only automatic fixes, asked for one at a time, every host change
//! recorded in the host-change ledger (RFC-0052 §11.1) before it is made, and the revert scripts
//! rewritten afterwards, so `uninstall` reverts it whether or not a Syntropy binary is left.

use std::fmt;
use std::io::{BufRead, Write};

use serde::Serialize;

use super::{AutoFix, DoctorCtx, Fix, FixSetup, Registry, Report, Status, run_check};
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

/// How one offered fix ended.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    /// Every edit was made; the ledger entries' IDs.
    Applied {
        /// The ledger entries, in edit order.
        changes: Vec<u32>,
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
/// checks again, replacing their results in `report`.
///
/// # Errors
///
/// [`LedgerError`] if the revert scripts cannot be written after a change. An edit that fails is
/// reported as [`Outcome::Failed`], and the other fixes are still offered.
pub fn apply(
    registry: &Registry,
    ctx: &DoctorCtx,
    report: &mut Report,
    setup: FixSetup<'_>,
) -> Result<Vec<FixLine>, LedgerError> {
    let FixSetup {
        ledger,
        consent,
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
        let outcome = match consent.ask(result.id, fix, auto) {
            Answer::No { why } => Outcome::Declined { why },
            Answer::Yes => {
                let mut changes = Vec::new();
                let mut error = None;
                for edit in &auto.edits {
                    match edit.apply(ledger, at, by_version) {
                        Ok(id) => changes.push(id),
                        Err(e) => {
                            error = Some(format!("{edit}: {e}"));
                            break;
                        }
                    }
                }
                // A failed write leaves its entry pending, which the ledger's revert resolves, so
                // the scripts are rewritten after any attempt.
                changed = true;
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
    if changed {
        write_revert_script(ledger, by_version)?;
    }
    // The fixed checks run again, so the report shows the host as it is now.
    for line in &lines {
        if matches!(line.outcome, Outcome::Declined { .. }) {
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
