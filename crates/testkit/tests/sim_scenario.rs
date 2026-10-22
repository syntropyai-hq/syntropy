// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scenario file (QUAL-E8-T03; RFC-0060 §8.5).
//!
//! - RFC-0060 §8.5's example parses.
//! - An unknown checker name in `[expect]` is rejected at load.

use syntropy_config::Duration;
use syntropy_testkit::sim::scenario::{
    CHECKERS, LengthDist, Lengths, NumericMode, SAFETY, Scenario, ScenarioError, Seeds,
    WorkloadKind,
};

/// RFC-0060 §8.5's example, as `testdata/sim/` holds it.
const EXAMPLE: &str = include_str!("../../../testdata/sim/pp3-stage-loss.toml");

#[test]
fn the_rfc_0060_example_parses() {
    let scenario = Scenario::from_toml(EXAMPLE).expect("the §8.5 example loads");
    assert_eq!(scenario.name, "pp3-stage-loss-mid-decode");
    assert_eq!(scenario.numeric, NumericMode::Digest);
    assert_eq!(scenario.model, "fixture:deepseek-v3-class/full-shape");
    assert_eq!(
        scenario.seeds,
        Seeds {
            pr_cap: 64,
            nightly_cap: 4_096
        }
    );
    assert_eq!(
        scenario.virtual_duration,
        "120s".parse::<Duration>().unwrap()
    );
    let nodes: Vec<(&str, usize, Option<&str>)> = scenario
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node.devices.len(), node.nvme.as_deref()))
        .collect();
    assert_eq!(
        nodes,
        [
            ("pc", 2, Some("gen4-990pro-class")),
            ("mac", 1, Some("apple-internal")),
            ("halo", 1, Some("gen4-generic")),
        ]
    );
    assert_eq!(
        scenario.nodes[0].devices,
        ["gpu:ada-24g", "cpu:ddr5-2ch-128g"]
    );
    assert_eq!(scenario.links.len(), 1);
    assert_eq!(scenario.links[0].between, ["pc", "mac", "halo"]);
    assert_eq!(scenario.links[0].preset, "lan_2_5g");
    assert_eq!(scenario.workload.kind, WorkloadKind::ClosedLoop);
    assert_eq!(scenario.workload.streams, 2);
    assert_eq!(
        scenario.workload.prompt,
        Lengths::Drawn(LengthDist::Uniform { lo: 256, hi: 4_096 })
    );
    assert_eq!(scenario.workload.r#gen, Lengths::Fixed(256));
    assert_eq!(scenario.faults.len(), 2);
    assert_eq!(
        (
            scenario.faults[0].at.as_str(),
            scenario.faults[0].action.as_str(),
            scenario.faults[0].target.as_deref()
        ),
        ("after:decode_tokens>=64", "kill_runner", Some("stage:1"))
    );
    assert_eq!(
        scenario.faults[1]
            .params
            .get("down")
            .and_then(|v| v.as_str()),
        Some("2s")
    );
    assert_eq!(
        scenario.expect.resume_within.as_deref(),
        Some("formula:sc28")
    );
    // `safety` expands to its eight checkers; the two it repeats are run once.
    let mut want: Vec<&str> = SAFETY.to_vec();
    want.push("sc28_resume");
    let want: Vec<&str> = CHECKERS.into_iter().filter(|c| want.contains(c)).collect();
    assert_eq!(scenario.expect.resolved_checkers(), want);
}

/// The example with one line replaced.
fn with(old: &str, new: &str) -> String {
    assert_eq!(EXAMPLE.matches(old).count(), 1, "{old:?}");
    EXAMPLE.replace(old, new)
}

#[test]
fn an_unknown_checker_is_rejected_at_load() {
    let text = with(
        r#"checkers = ["safety", "exactly_once_tokens", "sc28_resume", "no_approx_in_exact"]"#,
        r#"checkers = ["safety", "exactly_once_token"]"#,
    );
    let error = Scenario::from_toml(&text).unwrap_err();
    assert_eq!(
        error,
        ScenarioError::UnknownChecker("exactly_once_token".into())
    );
    assert!(
        error
            .to_string()
            .contains("`exactly_once_token` is not a checker"),
        "{error}"
    );
}

#[test]
fn every_rfc_0060_checker_name_loads() {
    for checker in CHECKERS.into_iter().chain(["safety"]) {
        let text = with(
            r#"checkers = ["safety", "exactly_once_tokens", "sc28_resume", "no_approx_in_exact"]"#,
            &format!(r#"checkers = ["{checker}"]"#),
        );
        assert!(Scenario::from_toml(&text).is_ok(), "{checker}");
    }
}

#[test]
fn a_scenario_that_cannot_run_is_rejected_at_load() {
    let cases = [
        (
            with("pr_cap = 64", "pr_cap = 65"),
            ScenarioError::TooManySeeds {
                cadence: "PR",
                count: 65,
                cap: 64,
            },
        ),
        (
            with("nightly_cap = 4096", "nightly_cap = 5000"),
            ScenarioError::TooManySeeds {
                cadence: "nightly",
                count: 5_000,
                cap: 4_096,
            },
        ),
        (
            with(r#"{ id = "halo","#, r#"{ id = "pc","#),
            ScenarioError::DuplicateNode("pc".into()),
        ),
        (
            with(
                r#"between = ["pc", "mac", "halo"]"#,
                r#"between = ["pc", "studio"]"#,
            ),
            ScenarioError::UnknownNode("studio".into()),
        ),
        (
            with(r#"between = ["pc", "mac", "halo"]"#, r#"between = ["pc"]"#),
            ScenarioError::LinkTooShort,
        ),
        (
            with("lo = 256, hi = 4096", "lo = 4096, hi = 256"),
            ScenarioError::EmptyRange { lo: 4_096, hi: 256 },
        ),
        (with("streams = 2", "streams = 0"), ScenarioError::NoStreams),
        (
            with(
                r#"resume_within = "formula:sc28""#,
                r#"resume_within = "soon""#,
            ),
            ScenarioError::BadResumeWithin("soon".into()),
        ),
        (
            with(r#""gpu:ada-24g", "cpu:ddr5-2ch-128g""#, r#""gpu:ada-48g""#),
            ScenarioError::UnknownPreset("gpu:ada-48g".into()),
        ),
        (
            with(r#""gpu:strix-halo-128g-unified""#, r#""gen4-generic""#),
            ScenarioError::WrongPresetKind {
                name: "gen4-generic".into(),
                expected: "a device",
            },
        ),
        (
            with(r#"nvme = "apple-internal""#, r#"nvme = "gpu:ada-24g""#),
            ScenarioError::WrongPresetKind {
                name: "gpu:ada-24g".into(),
                expected: "a drive",
            },
        ),
    ];
    for (text, want) in cases {
        assert_eq!(Scenario::from_toml(&text).unwrap_err(), want);
    }
    // A duration is a bound too.
    let text = with(
        r#"resume_within = "formula:sc28""#,
        r#"resume_within = "5s""#,
    );
    assert!(Scenario::from_toml(&text).is_ok());
}

#[test]
fn a_file_of_the_wrong_shape_is_a_parse_error() {
    for (old, new) in [
        (r#"numeric = "digest""#, r#"numeric = "approximate""#),
        (r#"virtual_duration = "120s""#, r#"virtual_duration = 120"#),
        (
            r#"name = "pp3-stage-loss-mid-decode""#,
            "names = \"x\"\nname = \"y\"",
        ),
        ("kind = \"closed_loop\"", "kind = \"open_loop\""),
    ] {
        let error = Scenario::from_toml(&with(old, new)).unwrap_err();
        assert!(matches!(error, ScenarioError::Parse(_)), "{new}: {error:?}");
    }
    // Seeds default to the caps when the file leaves them out.
    let text = with(
        r#"seeds = { pr_cap = 64, nightly_cap = 4096 }   # counts are derived ("Simulator throughput") and recorded here by CI"#,
        "",
    );
    assert_eq!(Scenario::from_toml(&text).unwrap().seeds, Seeds::default());
}
