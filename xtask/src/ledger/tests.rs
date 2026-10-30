// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

const JOBS: &str = include_str!("jobs-2026-10.json");
const METRICS: &str = include_str!("metrics-2026-10.json");
const PREVIOUS: &str = include_str!("previous-2026-09.json");

const COST_TEXT: &str = r#"
[[resource]]
kind = "hosted"
name = "standard-hosted-runners"
reason = "free on a public repository"

[[resource]]
kind = "larger-runner"
name = "ubuntu-24.04-8core"
reason = "the nightly CUDA build"

[[resource]]
kind = "storage"
name = "artefacts"
reason = "bisection"

[[resource]]
kind = "storage"
name = "results"
reason = "traces"
"#;

fn report(cost_text: &str, previous: Option<&str>) -> Report {
    let cost = Cost::parse(cost_text).unwrap();
    let pages = parse_pages(JOBS).unwrap();
    let metrics: Metrics = serde_json::from_str(METRICS).unwrap();
    let previous: Option<Previous> = previous.map(|p| serde_json::from_str(p).unwrap());
    build_report(&cost, "2026-10", &pages, &metrics, previous.as_ref())
}

fn value(report: &Report, key: &str) -> f64 {
    report
        .lines
        .iter()
        .find(|l| l.key == key)
        .unwrap_or_else(|| panic!("no line {key}: {}", report.text))
        .value
}

#[test]
fn the_repository_ledger_parses_and_its_workflows_pass() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let context = Context {
        root,
        args: Default::default(),
    };
    lint_step(&context).unwrap();
}

#[test]
fn the_report_counts_whole_minutes_for_the_month_only() {
    let r = report(COST_TEXT, None);
    // 3 min 10 s rounds up to 4, and 2 min is 2. The 30 September job and the skipped job are out.
    assert_eq!(value(&r, "hosted/lint"), 6.0);
    assert_eq!(value(&r, "hosted/hosted-test (macos)"), 8.0);
    assert_eq!(value(&r, "larger/ubuntu-24.04-8core"), 10.0);
    // A self-hosted job is lab time, which comes from the lab metrics.
    assert!(r.lines.iter().all(|l| !l.key.contains("nightly-parity")));
    assert_eq!(value(&r, "lab/R-ADA/nightly"), 5.5);
    assert_eq!(value(&r, "cache/kcache"), 90.0);
    assert_eq!(value(&r, "storage/artefacts"), 450.0);
    assert!(r.problems.is_empty(), "{:?}", r.problems);
}

#[test]
fn growth_over_twenty_percent_without_a_note_fails() {
    let r = report(COST_TEXT, Some(PREVIOUS));
    // results: 90 -> 120 GB is +33%. lint: 5 -> 6 is exactly +20%, which passes.
    assert_eq!(r.problems.len(), 1, "{:?}", r.problems);
    assert!(
        r.problems[0].starts_with("storage/results: grew 33%"),
        "{:?}",
        r.problems
    );
    assert!(r.text.contains("NEEDS NOTE"), "{}", r.text);
    assert!(!r.problems.iter().any(|p| p.contains("hosted/lint")));
}

#[test]
fn a_note_for_the_line_and_month_clears_the_growth() {
    let with_note = format!(
        "{COST_TEXT}\n[[note]]\nline = \"storage/results\"\nmonth = \"2026-10\"\ntext = \"a new corpus\"\n"
    );
    let r = report(&with_note, Some(PREVIOUS));
    assert!(r.problems.is_empty(), "{:?}", r.problems);
    assert!(r.text.contains("noted"), "{}", r.text);
    // The same note for another month does not count.
    let wrong_month = with_note.replace("month = \"2026-10\"", "month = \"2026-09\"");
    assert_eq!(report(&wrong_month, Some(PREVIOUS)).problems.len(), 1);
}

#[test]
fn a_larger_runner_or_bucket_without_an_entry_fails_the_report() {
    let bare = COST_TEXT
        .replace("larger-runner", "hosted")
        .replace("name = \"ubuntu-24.04-8core\"", "name = \"other\"");
    let r = report(&bare, None);
    assert!(
        r.problems
            .iter()
            .any(|p| p.starts_with("larger/ubuntu-24.04-8core"))
    );
    let no_bucket = COST_TEXT.replace("name = \"results\"", "name = \"traces\"");
    let r = report(&no_bucket, None);
    assert!(r.problems.iter().any(|p| p.starts_with("storage/results")));
}

#[test]
fn the_totals_round_trip_into_the_next_months_previous() {
    let r = report(COST_TEXT, None);
    let written = totals_json("2026-10", &r.lines);
    let previous: Previous = serde_json::from_str(&written).unwrap();
    assert_eq!(previous.lines["hosted/lint"], 6.0);
}

#[test]
fn timestamps_convert_to_epoch_seconds() {
    assert_eq!(epoch_seconds("1970-01-01T00:00:00Z"), Some(0));
    assert_eq!(epoch_seconds("2000-03-01T00:00:01Z"), Some(951_868_801));
    assert_eq!(epoch_seconds("2026-10-03"), None);
}

#[test]
fn the_ledger_is_validated() {
    let bad = |text: &str| Cost::parse(text).unwrap_err().join("\n");
    assert!(
        bad("[[resource]]\nkind = \"gpu\"\nname = \"x\"\nreason = \"r\"").contains("unknown kind")
    );
    assert!(
        bad("[[resource]]\nkind = \"storage\"\nname = \"x\"\nreason = \" \"").contains("no reason")
    );
    let twice = "[[resource]]\nkind = \"storage\"\nname = \"x\"\nreason = \"r\"\n";
    assert!(bad(&format!("{twice}{twice}")).contains("listed twice"));
    assert!(
        bad("[[resource]]\nkind = \"storage\"\nnme = \"x\"\nreason = \"r\"").contains("cost.toml")
    );
}

#[test]
fn larger_labels_are_recognised() {
    for larger in [
        "ubuntu-24.04-8core",
        "ubuntu-latest-16-cores",
        "linux-large",
        "macos-26-xlarge",
        "windows-2025-gpu",
        "ubuntu-24.04-16vcpu",
    ] {
        assert!(is_larger_label(larger), "{larger}");
    }
    for standard in [
        "ubuntu-24.04",
        "macos-26",
        "windows-2025",
        "windows-2025-arm",
        "self-hosted",
    ] {
        assert!(!is_larger_label(standard), "{standard}");
    }
}

#[test]
fn lint_fails_a_larger_label_without_an_entry() {
    let cost = Cost::parse(COST_TEXT).unwrap();
    let covered = "jobs:\n  a:\n    runs-on: ubuntu-24.04-8core\n";
    assert!(larger_runner_problems("w.yml", covered, &cost).is_empty());
    for (workflow, label) in [
        ("    runs-on: ubuntu-24.04-32core\n", "ubuntu-24.04-32core"),
        ("    runs-on:\n      - macos-26-xlarge\n", "macos-26-xlarge"),
        ("        runner: \"windows-2025-gpu\"\n", "windows-2025-gpu"),
        (
            "        os: [ubuntu-24.04, ubuntu-24.04-16core]\n",
            "ubuntu-24.04-16core",
        ),
    ] {
        let problems = larger_runner_problems("w.yml", workflow, &cost);
        assert_eq!(problems.len(), 1, "{workflow}: {problems:?}");
        assert!(problems[0].contains(label), "{problems:?}");
    }
    // A comment is not a use, and a standard label needs no entry.
    let quiet = "    runs-on: ubuntu-24.04 # not ubuntu-24.04-32core\n";
    assert!(larger_runner_problems("w.yml", quiet, &cost).is_empty());
}

#[test]
fn lab_runner_labels_are_not_larger_runners() {
    // A list with `self-hosted` is a lab runner, which the report counts from the lab metrics
    // and which GitHub does not bill, so none of its labels needs a `larger-runner` entry.
    let cost = Cost::parse(COST_TEXT).unwrap();
    for workflow in [
        "    runs-on: [self-hosted, r-ada, gpu]\n",
        "    runs-on: [\"self-hosted\", linux-large]\n",
        "    runs-on:\n      - self-hosted\n      - r-ada\n      - gpu\n",
        "    runs-on:\n      labels: [self-hosted, gpu]\n",
    ] {
        let problems = larger_runner_problems("w.yml", workflow, &cost);
        assert!(problems.is_empty(), "{workflow}: {problems:?}");
    }
    // `self-hosted` in one list does not exempt a hosted larger label in another.
    let mixed = "    runs-on: [self-hosted, gpu]\n  b:\n    runs-on: ubuntu-24.04-32core\n";
    assert_eq!(larger_runner_problems("w.yml", mixed, &cost).len(), 1);
}

#[test]
fn nothing_but_the_ledger_and_the_report_workflow_may_name_it() {
    let reads = "      - run: test \"$(cat ci/cost.toml | wc -l)\" -gt 3 || exit 1\n";
    assert_eq!(reader_problems(".github/workflows/pr.yml", reads).len(), 1);
    assert_eq!(
        reader_problems("ci/jobs.toml", "# see the ledger\n").len(),
        1
    );
    assert_eq!(
        reader_problems("governance/dispatch.toml", "limit = \"cost-report\"").len(),
        1
    );
    assert!(reader_problems("ci/jobs.toml", "# nothing to see\n").is_empty());
    assert!(reader_problems(COST, "# the ledger stops nothing, so `exit 1` is no gate").is_empty());
    // The report workflow may run the report, and may not branch on it.
    let report_run = "      - run: cargo xtask ci cost-report --month 2026-10\n";
    assert!(reader_problems(".github/workflows/cost-report.yml", report_run).is_empty());
    let gate = "        if: steps.cost-report.outputs.total > 100\n";
    assert_eq!(
        reader_problems(".github/workflows/cost-report.yml", gate).len(),
        1
    );
}

/// Two `pr.yml` runs of this repository, recorded read-only from
/// `GET /repos/syntropyai-hq/syntropy/actions/runs/{id}/jobs` on 2026-10-26 and trimmed to the
/// fields the report reads. They are public metadata.
#[test]
fn the_report_reads_recorded_actions_api_jobs() {
    let cost = Cost::parse(COST_TEXT).unwrap();
    let pages = parse_pages(include_str!("jobs-recorded-2026-10.json")).unwrap();
    let metrics: Metrics = serde_json::from_str("{}").unwrap();
    let r = build_report(&cost, "2026-10", &pages, &metrics, None);
    // lint ran 51 s and 87 s, which bill as 1 and 2 minutes. Skipped jobs bill nothing.
    assert_eq!(value(&r, "hosted/lint"), 3.0);
    assert_eq!(value(&r, "hosted/changes"), 2.0);
    assert_eq!(value(&r, "hosted/ci-gate"), 2.0);
    assert_eq!(r.lines.len(), 3, "{}", r.text);
    assert!(r.problems.is_empty(), "{:?}", r.problems);
}

/// A durations file in the shape `ci freeze --durations-out` writes (QUAL-E17-T04): two closed
/// windows of 2 h and 5 h, and one still open. The file is a hand-written test input.
#[test]
fn the_freeze_line_counts_windows_hours_and_the_longest() {
    let windows: Vec<FreezeWindow> =
        serde_json::from_str(include_str!("freeze-2026-10.json")).unwrap();
    assert_eq!(
        freeze_line(&windows),
        format!(
            "  {:<44} {:>10} windows, 7.0 h total, longest 5.0 h, 1 open\n",
            "release freezes", 3
        )
    );
    assert!(freeze_line(&[]).contains("0.0 h total"));
}
