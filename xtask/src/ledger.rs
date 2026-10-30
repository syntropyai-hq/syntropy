// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CI cost ledger: `ci/cost.toml`, the monthly report and the lint checks around them
//! (RFC-0062 §3.1 and "Performance model and budgets", Ledger; QUAL-E1-T13).
//!
//! Three things live here:
//!
//! - [`lint_step`], the `cost` step of `lint`: `ci/cost.toml` is well formed; every larger-runner
//!   label a workflow uses has a `larger-runner` entry there; and nothing under `.github/`, `ci/`,
//!   `governance/` or `bench/lab/` reads the ledger, because the ledger stops nothing.
//! - [`report_step`], `cargo xtask ci cost-report`: the monthly report over recorded Actions API
//!   job listings and recorded bucket, cache and lab metrics. It fails when a line grew by more
//!   than 20% over the previous month and the ledger has no note for it, or when a larger runner
//!   or a bucket has no entry. It makes no network call.
//! - the shapes of the recorded inputs, described at [`Metrics`] and [`Previous`].
//!
//! A job's minutes are its run time rounded up to whole minutes, as GitHub bills them. Lab rigs
//! are owned hardware, so lab hours are reported by rig and job class but need no entry.
//!
//! Both lint checks are text scans, not proofs. The larger-runner check finds labels by their
//! shape (a size token or a core count), so a larger runner with a plain name or a runner group
//! gets past it, and a list containing `self-hosted` is a lab runner it skips. The reader check
//! looks for the ledger's name and the keywords that would branch on it, so a step that builds
//! the path from pieces gets past it. Review still owns both rules.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::run::Context;

/// The ledger, relative to the workspace root.
pub const COST: &str = "ci/cost.toml";

/// Growth over the previous month, beyond which a line needs a note: 20% \[target\], from
/// RFC-0062 "Ledger".
const GROWTH_NUMERATOR: f64 = 1.0;
const GROWTH_DENOMINATOR: f64 = 5.0;

/// The only files allowed to mention the ledger by name. The report job is the one place that
/// runs the report, and it is a report: it never gates another job. QUAL-E2-T11's agent sandbox
/// definition and the dispatcher must not be added here.
const READERS: &[&str] = &[COST, ".github/workflows/cost-report.yml"];

/// The directories whose files may not read the ledger: workflows, CI configuration, governance
/// and the lab. The agent run definitions of QUAL-E2-T11 join this list when that task adds them.
const SCANNED: &[&str] = &[".github", "ci", "governance", "bench/lab"];

/// The words by which a file reads the ledger: its path, the report command, or the name.
const LEDGER_WORDS: &[&str] = &["cost.toml", "cost-report", "ledger"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCost {
    #[serde(default)]
    resource: Vec<Resource>,
    #[serde(default)]
    note: Vec<Note>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Resource {
    kind: String,
    name: String,
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Note {
    line: String,
    month: String,
    text: String,
}

/// The parsed ledger.
#[derive(Debug)]
pub struct Cost {
    resource: Vec<Resource>,
    note: Vec<Note>,
}

impl Cost {
    /// Parses and validates the text of `ci/cost.toml`.
    pub fn parse(text: &str) -> Result<Cost, Vec<String>> {
        let raw: RawCost = toml::from_str(text).map_err(|e| vec![format!("{COST}: {e}")])?;
        let mut problems = Vec::new();
        for (i, r) in raw.resource.iter().enumerate() {
            if !matches!(r.kind.as_str(), "hosted" | "larger-runner" | "storage") {
                problems.push(format!(
                    "{COST}: `{}` has the unknown kind `{}` (hosted, larger-runner or storage)",
                    r.name, r.kind
                ));
            }
            if r.name.trim().is_empty() {
                problems.push(format!("{COST}: resource {} has no name", i + 1));
            }
            if r.reason.trim().is_empty() {
                problems.push(format!("{COST}: `{}` has no reason", r.name));
            }
            if raw.resource[..i]
                .iter()
                .any(|p| p.kind == r.kind && p.name == r.name)
            {
                problems.push(format!("{COST}: `{}` ({}) is listed twice", r.name, r.kind));
            }
        }
        for n in &raw.note {
            if n.text.trim().is_empty() {
                problems.push(format!(
                    "{COST}: the note for `{}` in {} has no text",
                    n.line, n.month
                ));
            }
        }
        if problems.is_empty() {
            Ok(Cost {
                resource: raw.resource,
                note: raw.note,
            })
        } else {
            Err(problems)
        }
    }

    fn has(&self, kind: &str, name: &str) -> bool {
        self.resource
            .iter()
            .any(|r| r.kind == kind && r.name == name)
    }

    fn has_note(&self, line: &str, month: &str) -> bool {
        self.note
            .iter()
            .any(|n| n.line == line && n.month == month && !n.text.trim().is_empty())
    }
}

// ---------------------------------------------------------------------------------------------
// lint
// ---------------------------------------------------------------------------------------------

/// The `cost` step of `lint`.
pub fn lint_step(context: &Context) -> Result<(), String> {
    let root = &context.root;
    let text =
        std::fs::read_to_string(root.join(COST)).map_err(|error| format!("{COST}: {error}"))?;
    let cost = Cost::parse(&text).map_err(|problems| problems.join("\n"))?;
    let mut problems = Vec::new();
    for (file, text) in read_dir_files(root, ".github/workflows")? {
        problems.extend(larger_runner_problems(&file, &text, &cost));
    }
    for (file, text) in scanned_files(root)? {
        problems.extend(reader_problems(&file, &text));
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// A runner label that names a larger (billed) runner: a token `large`, `xlarge`, `xl` or
/// `xxl`, a core or vCPU count such as `8core`, `4-cores` or `16vcpu`, or `gpu`. Standard hosted
/// labels such as `ubuntu-24.04`, `macos-26` and `windows-2025-arm` match none of them.
pub fn is_larger_label(label: &str) -> bool {
    let label = label.to_ascii_lowercase();
    if label == "self-hosted" {
        return false;
    }
    let tokens: Vec<&str> = label.split(['-', '_']).collect();
    tokens.iter().enumerate().any(|(i, t)| {
        matches!(*t, "large" | "xlarge" | "xl" | "xxl" | "gpu")
            || t.strip_suffix("cores")
                .or_else(|| t.strip_suffix("core"))
                .or_else(|| t.strip_suffix("vcpu"))
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
            || (t.bytes().all(|b| b.is_ascii_digit())
                && !t.is_empty()
                && matches!(
                    tokens.get(i + 1),
                    Some(&("core" | "cores" | "vcpu" | "cpu"))
                ))
    })
}

/// Every runner label a workflow names, with its line: the values of `runs-on`, `runner` and
/// `labels` (a scalar, an inline list or a block list), and any token that begins with an OS name
/// and has an OS-version shape, which catches matrix `os` lists. Comments are ignored; `${{ }}`
/// expressions are skipped, because the matrix values they read are found where they are written.
/// A list that contains `self-hosted` names a lab runner, whose labels are not billed, so none of
/// its labels is returned.
fn runner_labels(text: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut found: Vec<(usize, String)> = Vec::new();
    let mut push = |n: usize, raw: &str| {
        let label = raw.trim().trim_matches(['"', '\'']).trim();
        if !label.is_empty()
            && !label.contains("${{")
            && !found.iter().any(|(l, v)| *l == n && v == label)
        {
            found.push((n, label.to_owned()));
        }
    };
    for (i, raw) in lines.iter().enumerate() {
        let line = strip_comment(raw);
        let trimmed = line.trim().trim_start_matches("- ").trim();
        // Free tokens that look like a hosted-runner image name.
        for token in trimmed.split(|c: char| !(c.is_ascii_alphanumeric() || "-._".contains(c))) {
            if ["ubuntu-", "macos-", "windows-"]
                .iter()
                .any(|p| token.starts_with(p))
            {
                push(i + 1, token);
            }
        }
        let Some((key, value)) = trimmed.split_once(':') else {
            continue;
        };
        if !matches!(key.trim(), "runs-on" | "runner" | "labels") {
            continue;
        }
        let value = value.trim();
        let mut list: Vec<(usize, &str)> = Vec::new();
        if let Some(inline) = value.strip_prefix('[') {
            for item in inline.trim_end_matches(']').split(',') {
                list.push((i + 1, item));
            }
        } else if value.is_empty() {
            let indent = line.len() - line.trim_start().len();
            for (j, next) in lines.iter().enumerate().skip(i + 1) {
                let next = strip_comment(next);
                let item = next.trim();
                if item.is_empty() {
                    continue;
                }
                let deeper = next.len() - next.trim_start().len() > indent;
                match item.strip_prefix("- ") {
                    Some(label) if deeper => list.push((j + 1, label)),
                    _ => break,
                }
            }
        } else if !value.starts_with('{') {
            list.push((i + 1, value));
        }
        let lab = list
            .iter()
            .any(|(_, l)| l.trim().trim_matches(['"', '\'']).trim() == "self-hosted");
        if !lab {
            for (n, label) in list {
                push(n, label);
            }
        }
    }
    found
}

fn strip_comment(line: &str) -> &str {
    match line
        .find(" #")
        .or_else(|| line.starts_with('#').then_some(0))
    {
        Some(at) => &line[..at],
        None => line,
    }
}

/// Larger-runner labels in a workflow that `cost.toml` has no entry for.
pub fn larger_runner_problems(file: &str, text: &str, cost: &Cost) -> Vec<String> {
    runner_labels(text)
        .into_iter()
        .filter(|(_, label)| is_larger_label(label) && !cost.has("larger-runner", label))
        .map(|(line, label)| {
            format!(
                "{file}:{line}: the larger-runner label `{label}` has no `larger-runner` entry \
                 in {COST} (RFC-0062 §3.1)"
            )
        })
        .collect()
}

/// A file that reads the ledger when it may not. Only [`READERS`] may name it, and the report
/// workflow may not branch on it.
pub fn reader_problems(file: &str, text: &str) -> Vec<String> {
    let allowed = READERS.contains(&file);
    let mut problems = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.to_ascii_lowercase();
        if !LEDGER_WORDS.iter().any(|w| line.contains(w)) {
            continue;
        }
        let branches = line.trim_start().starts_with("if:")
            || line.trim_start().starts_with("- if:")
            || line.contains("exit ");
        if !allowed {
            problems.push(format!(
                "{file}:{}: reads the cost ledger, which only {COST} and the report workflow \
                 may name; the ledger stops nothing (RFC-0062 §3.1)",
                i + 1
            ));
        } else if branches && file != COST {
            problems.push(format!(
                "{file}:{}: branches on the cost ledger; the ledger is a report and stops \
                 nothing (RFC-0062 §3.1)",
                i + 1
            ));
        }
    }
    problems
}

fn read_dir_files(root: &Path, dir: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    let mut stack = vec![PathBuf::from(dir)];
    while let Some(relative) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(root.join(&relative)) else {
            continue; // an absent directory has no files
        };
        let mut names: Vec<_> = entries.flatten().map(|e| e.file_name()).collect();
        names.sort();
        for name in names {
            let path = relative.join(&name);
            let full = root.join(&path);
            if full.is_dir() {
                stack.push(path);
            } else if let Ok(text) = std::fs::read_to_string(&full) {
                out.push((path.to_string_lossy().replace('\\', "/"), text));
            }
        }
    }
    out.sort();
    Ok(out)
}

fn scanned_files(root: &Path) -> Result<Vec<(String, String)>, String> {
    let mut all = Vec::new();
    for dir in SCANNED {
        all.extend(read_dir_files(root, dir)?);
    }
    Ok(all)
}

// ---------------------------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------------------------

/// One page of `GET /repos/{owner}/{repo}/actions/runs/{id}/jobs`, or the whole answer when it
/// is one page. The fixtures are an array of pages.
#[derive(Debug, Deserialize)]
pub struct JobsPage {
    jobs: Vec<Job>,
}

#[derive(Debug, Deserialize)]
pub struct Job {
    name: String,
    #[serde(default)]
    conclusion: Option<String>,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    started_at: Option<String>,
    #[serde(default)]
    completed_at: Option<String>,
}

/// The recorded bucket, cache and lab metrics for the month.
#[derive(Debug, Deserialize)]
pub struct Metrics {
    #[serde(default)]
    lab_hours: Vec<LabHours>,
    #[serde(default)]
    cache: Vec<CacheRate>,
    #[serde(default)]
    storage: Vec<Bucket>,
}

#[derive(Debug, Deserialize)]
struct LabHours {
    rig: String,
    class: String,
    hours: f64,
}

#[derive(Debug, Deserialize)]
struct CacheRate {
    name: String,
    hits: u64,
    misses: u64,
}

#[derive(Debug, Deserialize)]
struct Bucket {
    bucket: String,
    bytes: u64,
}

/// The previous month's report totals, as `--out` writes them: `{"month": "...", "lines": {key:
/// value}}`.
#[derive(Debug, Deserialize)]
pub struct Previous {
    lines: BTreeMap<String, f64>,
}

/// One release-freeze window, as `cargo xtask ci freeze --events <log> --durations-out <file>`
/// writes it (QUAL-E17-T04). Times are unix seconds; an open window has no end. Fields the report
/// does not read are ignored.
#[derive(Debug, Deserialize)]
pub struct FreezeWindow {
    #[serde(default)]
    seconds: Option<u64>,
}

/// The "release freezes" line: the windows, the hours of the closed ones, the longest, and how
/// many are still open. The hours come from the event log, so they are as measured as the log.
pub fn freeze_line(windows: &[FreezeWindow]) -> String {
    let closed: Vec<u64> = windows.iter().filter_map(|w| w.seconds).collect();
    let total: u64 = closed.iter().sum();
    let longest = closed.iter().copied().max().unwrap_or(0);
    format!(
        "  {:<44} {:>10} windows, {:.1} h total, longest {:.1} h, {} open\n",
        "release freezes",
        windows.len(),
        total as f64 / 3600.0,
        longest as f64 / 3600.0,
        windows.len() - closed.len()
    )
}

/// One line of the report.
#[derive(Debug, PartialEq)]
pub struct Line {
    pub key: String,
    pub value: f64,
    pub unit: &'static str,
    /// Whether the 20% growth rule applies. A hit rate is a ratio, not a spend.
    pub growth_rule: bool,
}

/// The report of one month, with every problem found.
#[derive(Debug)]
pub struct Report {
    pub lines: Vec<Line>,
    pub text: String,
    pub problems: Vec<String>,
}

/// Seconds since the epoch of an `YYYY-MM-DDTHH:MM:SSZ` timestamp.
fn epoch_seconds(stamp: &str) -> Option<i64> {
    let b = stamp.as_bytes();
    if b.len() != 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[19] != b'Z' {
        return None;
    }
    let n = |r: std::ops::Range<usize>| stamp.get(r)?.parse::<i64>().ok();
    let (y, m, d) = (n(0..4)?, n(5..7)?, n(8..10)?);
    let (h, mi, s) = (n(11..13)?, n(14..16)?, n(17..19)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || h > 23 || mi > 59 || s > 60 {
        return None;
    }
    // Days from civil, after Howard Hinnant's algorithm.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + h * 3600 + mi * 60 + s)
}

/// Builds the report of `month` (`YYYY-MM`).
pub fn build_report(
    cost: &Cost,
    month: &str,
    pages: &[JobsPage],
    metrics: &Metrics,
    previous: Option<&Previous>,
) -> Report {
    let mut problems = Vec::new();
    let mut hosted: BTreeMap<String, f64> = BTreeMap::new();
    let mut larger: BTreeMap<String, f64> = BTreeMap::new();
    for job in pages.iter().flat_map(|p| p.jobs.iter()) {
        let (Some(start), Some(end)) = (&job.started_at, &job.completed_at) else {
            continue; // still queued or running: not billed yet
        };
        if !start.starts_with(month) || job.conclusion.as_deref() == Some("skipped") {
            continue;
        }
        if job.labels.iter().any(|l| l == "self-hosted") {
            continue; // a lab job: its hours come from the lab metrics
        }
        let (Some(a), Some(b)) = (epoch_seconds(start), epoch_seconds(end)) else {
            problems.push(format!("job `{}`: unreadable timestamps", job.name));
            continue;
        };
        let minutes = ((b - a).max(0) as f64 / 60.0).ceil();
        match job.labels.iter().find(|l| is_larger_label(l)) {
            Some(label) => *larger.entry(label.clone()).or_default() += minutes,
            None => *hosted.entry(job.name.clone()).or_default() += minutes,
        }
    }
    let mut lines = Vec::new();
    for (name, v) in hosted {
        lines.push(line(format!("hosted/{name}"), v, "min", true));
    }
    for (label, v) in larger {
        if !cost.has("larger-runner", &label) {
            problems.push(format!(
                "larger/{label}: used with no `larger-runner` entry in {COST}"
            ));
        }
        lines.push(line(format!("larger/{label}"), v, "min", true));
    }
    let mut lab: BTreeMap<String, f64> = BTreeMap::new();
    for l in &metrics.lab_hours {
        *lab.entry(format!("lab/{}/{}", l.rig, l.class)).or_default() += l.hours;
    }
    for (key, v) in lab {
        lines.push(line(key, v, "h", true));
    }
    for c in &metrics.cache {
        let total = c.hits + c.misses;
        let rate = if total == 0 {
            0.0
        } else {
            c.hits as f64 * 100.0 / total as f64
        };
        lines.push(line(format!("cache/{}", c.name), rate, "% hit", false));
    }
    for b in &metrics.storage {
        if !cost.has("storage", &b.bucket) {
            problems.push(format!(
                "storage/{}: a bucket with no `storage` entry in {COST}",
                b.bucket
            ));
        }
        lines.push(line(
            format!("storage/{}", b.bucket),
            b.bytes as f64 / 1e9,
            "GB",
            true,
        ));
    }
    let mut text = format!("CI cost ledger report for {month}\n");
    for l in &lines {
        let before = previous.and_then(|p| p.lines.get(&l.key)).copied();
        let mut row = format!("  {:<44} {:>10.1} {}", l.key, l.value, l.unit);
        if let Some(p) = before.filter(|p| *p > 0.0 && l.growth_rule) {
            let _ = write!(row, "  (was {p:.1}, {:+.0}%)", (l.value - p) * 100.0 / p);
            if (l.value - p) * GROWTH_DENOMINATOR > p * GROWTH_NUMERATOR {
                if cost.has_note(&l.key, month) {
                    row.push_str("  noted");
                } else {
                    row.push_str("  NEEDS NOTE");
                    problems.push(format!(
                        "{}: grew {:.0}% over the previous month (more than 20%) and {COST} \
                         has no `[[note]]` for it in {month}",
                        l.key,
                        (l.value - p) * 100.0 / p
                    ));
                }
            }
        }
        text.push_str(&row);
        text.push('\n');
    }
    Report {
        lines,
        text,
        problems,
    }
}

fn parse_pages(text: &str) -> Result<Vec<JobsPage>, String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let pages = match value {
        serde_json::Value::Array(pages) => pages,
        page => vec![page],
    };
    pages
        .into_iter()
        .map(|p| serde_json::from_value(p).map_err(|e| e.to_string()))
        .collect()
}

fn line(key: String, value: f64, unit: &'static str, growth_rule: bool) -> Line {
    Line {
        key,
        value,
        unit,
        growth_rule,
    }
}

/// The totals `--out` writes, for next month's `--previous`.
fn totals_json(month: &str, lines: &[Line]) -> String {
    let map: serde_json::Map<String, serde_json::Value> = lines
        .iter()
        .map(|l| (l.key.clone(), serde_json::json!(l.value)))
        .collect();
    let doc = serde_json::json!({ "month": month, "lines": map });
    serde_json::to_string_pretty(&doc).unwrap_or_default() + "\n"
}

/// `cargo xtask ci cost-report`.
pub fn report_step(context: &Context) -> Result<(), String> {
    let arg = |name: &str| context.args.get(name);
    let required = |name: &str| arg(name).ok_or_else(|| format!("{name} is required"));
    let month = required("--month")?;
    let well_formed = month.len() == 7
        && month.as_bytes()[4] == b'-'
        && month[..4].bytes().all(|b| b.is_ascii_digit())
        && month[5..]
            .parse::<u32>()
            .is_ok_and(|m| (1..=12).contains(&m));
    if !well_formed {
        return Err(format!("--month `{month}` is not YYYY-MM"));
    }
    let read = |path: &str| {
        std::fs::read_to_string(context.root.join(path)).map_err(|e| format!("{path}: {e}"))
    };
    let cost = Cost::parse(&read(arg("--cost").unwrap_or(COST))?).map_err(|p| p.join("\n"))?;
    let jobs_path = required("--jobs")?;
    let pages = parse_pages(&read(jobs_path)?).map_err(|e| format!("{jobs_path}: {e}"))?;
    let metrics_path = required("--metrics")?;
    let metrics: Metrics =
        serde_json::from_str(&read(metrics_path)?).map_err(|e| format!("{metrics_path}: {e}"))?;
    let previous = match arg("--previous") {
        Some(path) => Some(
            serde_json::from_str::<Previous>(&read(path)?).map_err(|e| format!("{path}: {e}"))?,
        ),
        None => None,
    };
    let mut report = build_report(&cost, month, &pages, &metrics, previous.as_ref());
    if let Some(path) = arg("--freeze") {
        let windows: Vec<FreezeWindow> =
            serde_json::from_str(&read(path)?).map_err(|e| format!("{path}: {e}"))?;
        report.text.push_str(&freeze_line(&windows));
    }
    print!("{}", report.text);
    if let Some(out) = arg("--out") {
        std::fs::write(context.root.join(out), totals_json(month, &report.lines))
            .map_err(|e| format!("{out}: {e}"))?;
    }
    if report.problems.is_empty() {
        Ok(())
    } else {
        Err(report.problems.join("\n"))
    }
}

#[cfg(test)]
mod tests;
