// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `syntropy doctor` framework (RFC-0053 §5.1): checks, their findings and fixes, the modes,
//! the exit code, and `--fix`, which applies only automatic fixes, asks before each one, and records
//! every host change in the host-change ledger so that `uninstall` reverts it.
//!
//! The checks themselves come later: the offline checks with PROD-E9-T02, the daemon checks with
//! PROD-E9-T03 and the cluster checks with PROD-E9-T07. The `syntropy doctor` command that calls
//! [`doctor`] is PROD-E6-T08's.
//!
//! A run is:
//!
//! 1. [`Registry::new`] checks the check list: every ID is `<prefix>-<three digits>` with its
//!    area's prefix, unique, and never one of [`RETIRED`].
//! 2. [`run`] runs every check the [`DoctorCtx`] can reach, and reports the others as
//!    [`Status::Skip`] with the reason: under `--offline`, every check that needs the daemon.
//! 3. With `--fix`, [`fix::apply`] offers each automatic fix of a failing or warning check, asks
//!    the [`fix::Consent`], applies the accepted ones through the ledger, rewrites the revert
//!    scripts, and runs those checks again.
//! 4. [`Report::exit_code`] is non-zero when any check fails.

pub mod fix;

use std::collections::BTreeSet;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use serde::Serialize;
use syntropy_api::figure::Figure;

use crate::host_changes::{HostChangeLedger, LedgerError};

/// The exit code when no check failed.
pub const EXIT_OK: i32 = 0;

/// The exit code when at least one check failed (RFC-0053 §5.1).
pub const EXIT_FAILED: i32 = 1;

/// The version of the `--json` output's format.
pub const JSON_SCHEMA: u16 = 1;

/// Check IDs that were used once and are retired. An ID is never reused (RFC-0053 §5.1), so
/// [`Registry::new`] refuses a check with one of these.
pub const RETIRED: &[&str] = &[];

/// The documentation directory that holds one page per check ID, `<id>.md` (PROD-E13-T06).
pub const DOCS_DIR: &str = "docs/troubleshooting";

/// What a check is about. Each area gives its checks' ID prefix (RFC-0053 §5.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Area {
    /// `INS`: the installation, the service and the host-change ledger.
    Install,
    /// `DRV`: GPU drivers and runtimes.
    Drivers,
    /// `HW`: the hardware against its peak.
    Hardware,
    /// `OS`: operating-system limits and state.
    Os,
    /// `STO`: the model store's volumes.
    Storage,
    /// `NET`: discovery and the data-plane links.
    Network,
    /// `RDMA`: RDMA hardware and its set-up.
    Rdma,
    /// `CLU`: the cluster.
    Cluster,
    /// `SEC`: listeners, keys and permissions.
    Security,
    /// `CFG`: configuration.
    Config,
    /// `RUN`: runners.
    Runners,
    /// `PRV`: what leaves this node.
    Privacy,
}

impl Area {
    /// Every area, in the order RFC-0053 §5.2 lists them.
    pub const ALL: [Self; 12] = [
        Self::Install,
        Self::Drivers,
        Self::Hardware,
        Self::Os,
        Self::Storage,
        Self::Network,
        Self::Rdma,
        Self::Cluster,
        Self::Security,
        Self::Config,
        Self::Runners,
        Self::Privacy,
    ];

    /// The ID prefix of this area's checks, such as `OS` in `OS-003`.
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Install => "INS",
            Self::Drivers => "DRV",
            Self::Hardware => "HW",
            Self::Os => "OS",
            Self::Storage => "STO",
            Self::Network => "NET",
            Self::Rdma => "RDMA",
            Self::Cluster => "CLU",
            Self::Security => "SEC",
            Self::Config => "CFG",
            Self::Runners => "RUN",
            Self::Privacy => "PRV",
        }
    }
}

/// What a check needs before it can run (RFC-0053 §5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Needs {
    /// Nothing but this host and its files: install, drivers, OS limits, storage, config. These
    /// run under `--offline`.
    Offline,
    /// The running daemon: runners, listeners, keys.
    Daemon,
    /// The daemon's probe results.
    Probe,
    /// The cluster state, through the daemon.
    Cluster,
}

/// What this run of `doctor` can reach, which decides the checks that run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reach {
    /// The daemon answers.
    pub daemon: bool,
    /// The daemon has probe results.
    pub probe: bool,
    /// The node is a cluster member and the daemon has the cluster state.
    pub cluster: bool,
}

impl Reach {
    /// `--offline`: only this host, even when a daemon runs.
    pub const OFFLINE: Self = Self {
        daemon: false,
        probe: false,
        cluster: false,
    };

    /// Why a check that needs `needs` cannot run, or `None` when it can.
    #[must_use]
    pub fn missing(self, needs: Needs, offline: bool) -> Option<&'static str> {
        let daemon_missing = if offline {
            "--offline: this check needs the daemon"
        } else {
            "the daemon is not running"
        };
        match needs {
            Needs::Offline => None,
            Needs::Daemon | Needs::Probe | Needs::Cluster if offline || !self.daemon => {
                Some(daemon_missing)
            }
            Needs::Probe if !self.probe => Some("the daemon has no probe results yet"),
            Needs::Cluster if !self.cluster => Some("this node is not in a cluster"),
            Needs::Daemon | Needs::Probe | Needs::Cluster => None,
        }
    }
}

/// What a check reads. PROD-E9-T02 and PROD-E9-T03 add the OS facts, the daemon's state and the
/// probe results their checks read; tests give fake ones.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct DoctorCtx {
    /// What this run can reach.
    pub reach: Reach,
    /// `--offline` was given.
    pub offline: bool,
    /// The daemon's state directory, which holds the host-change ledger.
    pub state_dir: PathBuf,
}

impl DoctorCtx {
    /// A context for a run that can reach `reach`, with `state_dir` as the state directory.
    #[must_use]
    pub fn new(state_dir: &Path, reach: Reach) -> Self {
        Self {
            reach,
            offline: false,
            state_dir: state_dir.to_owned(),
        }
    }

    /// The same context under `--offline`: only checks that need nothing but the host run.
    #[must_use]
    pub fn offline(mut self) -> Self {
        self.offline = true;
        self.reach = Reach::OFFLINE;
        self
    }
}

/// One check (RFC-0053 §5.1).
#[derive(Clone, Copy, Debug)]
pub struct DoctorCheck {
    /// Stable, such as `OS-003`: documented in [`DOCS_DIR`] and never reused.
    pub id: &'static str,
    /// What it is about; gives the ID's prefix.
    pub area: Area,
    /// What it needs before it can run.
    pub needs: Needs,
    /// The check. It reads `ctx` and never changes the host; a fix that changes the host is an
    /// [`AutoFix`] in its finding, which only `--fix` applies.
    pub run: fn(&DoctorCtx) -> Finding,
}

impl DoctorCheck {
    /// The check's documentation page, `docs/troubleshooting/<id>.md`.
    #[must_use]
    pub fn doc_page(&self) -> String {
        doc_page(self.id)
    }
}

/// The documentation page for check `id`.
#[must_use]
pub fn doc_page(id: &str) -> String {
    format!("{DOCS_DIR}/{id}.md")
}

/// How a check came out (RFC-0053 §5.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Status {
    /// Nothing to report.
    Pass,
    /// Worth knowing, nothing to do.
    Info,
    /// Works, but worse than it could.
    Warn,
    /// Broken: the exit code is non-zero.
    Fail,
    /// Not run.
    Skip {
        /// Why, such as `--offline`.
        why: String,
    },
}

impl Status {
    /// The status as the text output prints it.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
            Self::Skip { .. } => "SKIP",
        }
    }
}

/// What a check found (RFC-0053 §5.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Finding {
    /// How it came out.
    #[serde(flatten)]
    pub status: Status,
    /// One line on what was found.
    pub summary: String,
    /// The measured values, versions and limits behind it, by name.
    #[serde(serialize_with = "evidence_json")]
    pub evidence: Vec<(String, Figure<String>)>,
    /// What to do about it.
    pub fix: Option<Fix>,
}

impl Finding {
    /// A finding with `status` and `summary`, no evidence and no fix.
    pub fn new(status: Status, summary: impl Into<String>) -> Self {
        Self {
            status,
            summary: summary.into(),
            evidence: Vec::new(),
            fix: None,
        }
    }

    /// A passing finding.
    pub fn pass(summary: impl Into<String>) -> Self {
        Self::new(Status::Pass, summary)
    }

    /// A failing finding.
    pub fn fail(summary: impl Into<String>) -> Self {
        Self::new(Status::Fail, summary)
    }

    /// A warning.
    pub fn warn(summary: impl Into<String>) -> Self {
        Self::new(Status::Warn, summary)
    }

    /// Adds one piece of evidence.
    #[must_use]
    pub fn with_evidence(mut self, name: impl Into<String>, figure: Figure<String>) -> Self {
        self.evidence.push((name.into(), figure));
        self
    }

    /// Sets the fix.
    #[must_use]
    pub fn with_fix(mut self, fix: Fix) -> Self {
        self.fix = Some(fix);
        self
    }
}

/// The evidence as a JSON array of objects, each with its name beside the figure's fields.
fn evidence_json<S: serde::Serializer>(
    evidence: &[(String, Figure<String>)],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct Item<'a> {
        name: &'a str,
        #[serde(flatten)]
        figure: &'a Figure<String>,
    }
    serializer.collect_seq(evidence.iter().map(|(name, figure)| Item { name, figure }))
}

/// What to do about a finding (RFC-0053 §5.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Fix {
    /// What the fix does, in a sentence.
    pub description: String,
    /// The exact command to run, for this OS.
    pub command: Option<String>,
    /// The check's documentation page; [`run`] fills it in from the check ID when the check
    /// leaves it out.
    pub doc: Option<String>,
    /// The change `--fix` makes, for the fixes that are safe, reversible and host-ledgered.
    pub automatic: Option<AutoFix>,
}

impl Fix {
    /// A fix the user applies, described by `description`.
    pub fn manual(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            command: None,
            doc: None,
            automatic: None,
        }
    }
}

/// A fix that `--fix` may apply: host edits, each one recorded in the host-change ledger before it
/// is made, so `uninstall` reverts it (RFC-0053 §5.1, RFC-0052 §11.1).
///
/// The edits are a closed list of ledgered kinds, and [`fix::Targets`] closes the files they may
/// write: a pluggable-authentication `limits.d` file and a systemd service drop-in, nothing else.
/// A fix that writes anywhere else, such as `sysctl.d`, `modprobe.d` (the TTM `pages_limit`), the
/// boot loader's configuration or a launchd plist, is refused before anyone is asked. There is no
/// edit at all for firmware or BIOS settings, the Windows GPU watchdog (`TdrDelay`),
/// hardware-accelerated GPU scheduling or anything on another node. `doctor` never changes those,
/// and the only OS limits Syntropy changes go through `syntropy tune` (RFC-0051 §2.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoFix {
    /// The edits, made in order.
    pub edits: Vec<HostEdit>,
}

/// One host edit an [`AutoFix`] makes.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostEdit {
    /// Writes `contents` to `path`, such as a `limits.d` file or a unit drop-in, through
    /// [`HostChangeLedger::write_file`]: the content it replaces is saved, and a revert puts it
    /// back or removes the file.
    WriteFile {
        /// An absolute path.
        path: PathBuf,
        /// The file's new content.
        contents: Vec<u8>,
    },
}

impl HostEdit {
    /// Makes the edit through `ledger`, and returns the ledger entry's ID.
    ///
    /// # Errors
    ///
    /// As [`HostChangeLedger::write_file`].
    pub fn apply(
        &self,
        ledger: &mut HostChangeLedger,
        at: u64,
        by_version: &str,
    ) -> Result<u32, LedgerError> {
        match self {
            Self::WriteFile { path, contents } => ledger.write_file(path, contents, at, by_version),
        }
    }
}

impl fmt::Display for HostEdit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WriteFile { path, contents } => {
                write!(f, "write {} ({} bytes)", path.display(), contents.len())
            }
        }
    }
}

/// The JSON output lists each edit as text; a file's new content stays out of it.
impl Serialize for AutoFix {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Edits {
            edits: Vec<String>,
        }
        Edits {
            edits: self.edits.iter().map(ToString::to_string).collect(),
        }
        .serialize(serializer)
    }
}

/// Why a check list was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// The ID is not `<area prefix>-<three digits>`.
    BadId {
        /// The ID.
        id: &'static str,
        /// The check's area.
        area: Area,
    },
    /// Two checks have this ID.
    Duplicate {
        /// The ID.
        id: &'static str,
    },
    /// The ID was retired and may not be used again.
    Retired {
        /// The ID.
        id: &'static str,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadId { id, area } => write!(
                f,
                "check ID {id:?} is not {}-<three digits>, as its area {area:?} requires",
                area.prefix()
            ),
            Self::Duplicate { id } => write!(f, "check ID {id} is used twice"),
            Self::Retired { id } => write!(f, "check ID {id} was retired and is never reused"),
        }
    }
}

impl std::error::Error for RegistryError {}

/// A checked list of checks, in the order they run and report.
#[derive(Clone, Debug)]
pub struct Registry {
    checks: Vec<DoctorCheck>,
}

impl Registry {
    /// Checks the list.
    ///
    /// # Errors
    ///
    /// The first [`RegistryError`] found, in list order.
    pub fn new(checks: Vec<DoctorCheck>) -> Result<Self, RegistryError> {
        let mut seen = BTreeSet::new();
        for check in &checks {
            let digits = check
                .id
                .strip_prefix(check.area.prefix())
                .and_then(|rest| rest.strip_prefix('-'));
            let well_formed =
                digits.is_some_and(|d| d.len() == 3 && d.bytes().all(|b| b.is_ascii_digit()));
            if !well_formed {
                return Err(RegistryError::BadId {
                    id: check.id,
                    area: check.area,
                });
            }
            if RETIRED.contains(&check.id) {
                return Err(RegistryError::Retired { id: check.id });
            }
            if !seen.insert(check.id) {
                return Err(RegistryError::Duplicate { id: check.id });
            }
        }
        Ok(Self { checks })
    }

    /// The checks, in run order.
    #[must_use]
    pub fn checks(&self) -> &[DoctorCheck] {
        &self.checks
    }

    /// The check with ID `id`.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&DoctorCheck> {
        self.checks.iter().find(|c| c.id == id)
    }
}

/// One check's result in a report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CheckResult {
    /// The check's ID.
    pub id: &'static str,
    /// Its area.
    pub area: Area,
    /// What it needs.
    pub needs: Needs,
    /// What it found.
    #[serde(flatten)]
    pub finding: Finding,
}

/// The result of a run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Report {
    /// The `--json` format's version, [`JSON_SCHEMA`].
    pub schema: u16,
    /// `--offline` was given.
    pub offline: bool,
    /// One result per check, in run order.
    pub results: Vec<CheckResult>,
}

impl Report {
    /// [`EXIT_FAILED`] when any check failed, otherwise [`EXIT_OK`]. Warnings and skipped checks
    /// do not fail a run.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        if self
            .results
            .iter()
            .any(|r| r.finding.status == Status::Fail)
        {
            EXIT_FAILED
        } else {
            EXIT_OK
        }
    }

    /// The result of check `id`.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&CheckResult> {
        self.results.iter().find(|r| r.id == id)
    }

    /// The report as `--json` prints it.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a report always serializes")
    }

    /// The report as text, one block per check: the status, ID and summary, then each piece of
    /// evidence with its label, and the fix.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for r in &self.results {
            let f = &r.finding;
            out.push_str(&format!(
                "{:<4}  {:<8}  {}\n",
                f.status.label(),
                r.id,
                f.summary
            ));
            if let Status::Skip { why } = &f.status {
                out.push_str(&format!("      skipped: {why}\n"));
            }
            for (name, figure) in &f.evidence {
                out.push_str(&format!("      {name}: {figure}\n"));
            }
            if let Some(fix) = &f.fix {
                out.push_str(&format!("      fix: {}\n", fix.description));
                if let Some(command) = &fix.command {
                    out.push_str(&format!("      run: {command}\n"));
                }
                if fix.automatic.is_some() {
                    out.push_str("      `syntropy doctor --fix` can apply this\n");
                }
                if let Some(doc) = &fix.doc {
                    out.push_str(&format!("      see: {doc}\n"));
                }
            }
        }
        let count = |s: &str| {
            self.results
                .iter()
                .filter(|r| r.finding.status.label() == s)
                .count()
        };
        out.push_str(&format!(
            "{} passed, {} info, {} warnings, {} failed, {} skipped\n",
            count("PASS"),
            count("INFO"),
            count("WARN"),
            count("FAIL"),
            count("SKIP")
        ));
        out
    }
}

/// Runs one check, or skips it when `ctx` cannot reach what it needs. A check that panics fails,
/// and the run goes on.
#[must_use]
pub fn run_check(check: &DoctorCheck, ctx: &DoctorCtx) -> CheckResult {
    let finding = match ctx.reach.missing(check.needs, ctx.offline) {
        Some(why) => Finding::new(Status::Skip { why: why.into() }, "not run".to_owned()),
        None => match panic::catch_unwind(AssertUnwindSafe(|| (check.run)(ctx))) {
            Ok(mut finding) => {
                if let Some(fix) = &mut finding.fix
                    && fix.doc.is_none()
                {
                    fix.doc = Some(check.doc_page());
                }
                finding
            }
            Err(_) => Finding::fail("the check itself failed (it panicked); please report it")
                .with_fix(Fix {
                    doc: Some(check.doc_page()),
                    ..Fix::manual("re-run with `RUST_BACKTRACE=1` and report the output")
                }),
        },
    };
    CheckResult {
        id: check.id,
        area: check.area,
        needs: check.needs,
        finding,
    }
}

/// Runs every check in `registry` against `ctx` (RFC-0053 §5.1).
#[must_use]
pub fn run(registry: &Registry, ctx: &DoctorCtx) -> Report {
    Report {
        schema: JSON_SCHEMA,
        offline: ctx.offline,
        results: registry
            .checks()
            .iter()
            .map(|check| run_check(check, ctx))
            .collect(),
    }
}

/// The flags of `syntropy doctor` that this framework handles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// `--offline`: run only the checks that need no daemon.
    pub offline: bool,
    /// `--json`: print the report as JSON.
    pub json: bool,
    /// `--fix`: offer the automatic fixes.
    pub fix: bool,
}

/// What `doctor --fix` needs to record its changes.
pub struct FixSetup<'a> {
    /// The host-change ledger.
    pub ledger: &'a mut HostChangeLedger,
    /// Asks before each fix: a terminal prompt, `--yes`, or a refusal when there is no terminal.
    pub consent: &'a mut dyn fix::Consent,
    /// The files a fix may write: [`fix::Targets::host`] in production.
    pub targets: &'a fix::Targets,
    /// The time to record, in Unix milliseconds.
    pub at: u64,
    /// The Syntropy version to record.
    pub by_version: &'a str,
}

/// `syntropy doctor`: runs the checks, applies fixes under `--fix`, writes the report to `out` and
/// returns the exit code. `fix_setup` is used only under `--fix`.
///
/// # Errors
///
/// [`LedgerError`] if `--fix` cannot write the ledger or the revert scripts. A fix that fails
/// on its own is reported in the output, and the run goes on.
pub fn doctor(
    registry: &Registry,
    ctx: &DoctorCtx,
    options: Options,
    fix_setup: Option<FixSetup<'_>>,
    out: &mut dyn std::io::Write,
) -> Result<i32, DoctorError> {
    let ctx = if options.offline {
        ctx.clone().offline()
    } else {
        ctx.clone()
    };
    let mut report = run(registry, &ctx);
    let mut fixes = Vec::new();
    if options.fix {
        let setup = fix_setup.ok_or(DoctorError::NoLedger)?;
        fixes = match fix::apply(registry, &ctx, &mut report, setup) {
            Ok(lines) => lines,
            Err(fix::FixError { lines, error }) => {
                // The fixes that were made are still reported, before the error, in the format
                // the run asked for.
                let text = if options.json {
                    fix::json_error(&lines, &error) + "\n"
                } else {
                    lines.iter().map(|line| format!("{line}\n")).collect()
                };
                out.write_all(text.as_bytes())
                    .map_err(DoctorError::Output)?;
                return Err(DoctorError::Ledger(error));
            }
        };
    }
    let text = if options.json {
        if options.fix {
            fix::json_with_fixes(&report, &fixes)
        } else {
            report.to_json()
        }
    } else {
        let mut text = String::new();
        for line in &fixes {
            text.push_str(&format!("{line}\n"));
        }
        text.push_str(&report.to_text());
        text
    };
    out.write_all(text.as_bytes())
        .and_then(|()| {
            if options.json {
                out.write_all(b"\n")
            } else {
                Ok(())
            }
        })
        .map_err(DoctorError::Output)?;
    Ok(report.exit_code())
}

/// Why `doctor` could not finish.
#[derive(Debug)]
#[non_exhaustive]
pub enum DoctorError {
    /// `--fix` was given without a ledger to record changes in.
    NoLedger,
    /// The ledger or the revert scripts could not be written.
    Ledger(LedgerError),
    /// The report could not be written out.
    Output(std::io::Error),
}

impl From<LedgerError> for DoctorError {
    fn from(e: LedgerError) -> Self {
        Self::Ledger(e)
    }
}

impl fmt::Display for DoctorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLedger => f.write_str("--fix needs the host-change ledger"),
            Self::Ledger(e) => write!(f, "{e}"),
            Self::Output(e) => write!(f, "could not write the report: {e}"),
        }
    }
}

impl std::error::Error for DoctorError {}

#[cfg(test)]
mod tests;
