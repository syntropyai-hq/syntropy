// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scenario file (QUAL-E8-T03; RFC-0060 §8.5).
//!
//! - RFC-0060 §8.5's example parses.
//! - An unknown checker name in `[expect]` is rejected at load.

use syntropy_config::Duration;
use syntropy_testkit::sim::presets::{FigureError, Label};
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

/// A two-node scenario with `needs`, and each node's own lines, such as a `set`.
fn with_figures(needs: &str, pc: &str, mac: &str) -> String {
    format!(
        r#"
name = "figures"
numeric = "digest"
model = "fixture:nano"
virtual_duration = "1s"
needs = {needs}
workload = {{ kind = "closed_loop", streams = 1, prompt = 16, gen = 16 }}

[[nodes]]
id = "pc"
devices = ["gpu:ada-24g", "cpu:ddr5-2ch-128g"]
nvme = "gen4-990pro-class"
{pc}

[[nodes]]
id = "mac"
devices = ["gpu:apple-max-128g-unified"]
nvme = "apple-internal"
{mac}

[expect]
checkers = ["safety"]
"#
    )
}

/// A supplied figure: a value the scenario chooses, labelled `target`.
fn supplied(preset: &str, key: &str, value: u32) -> String {
    format!(
        "[nodes.set.\"{preset}\".\"{key}\"]\nvalue = {value}\nunit = \"µs\"\nlabel = \"target\"\nsource = \"chosen for this test\"\n"
    )
}

/// A scenario that needs a figure the research leaves `unknown` loads only if each node that has
/// such a device supplies it, labelled; and the figure it reads is the supplied one.
#[test]
fn a_needed_unknown_figure_is_supplied_or_the_scenario_does_not_load() {
    // Not supplied: the load refuses, naming the node, the preset and the key.
    let error = Scenario::from_toml(&with_figures(r#"["t_fixed"]"#, "", "")).unwrap_err();
    assert!(
        matches!(
            &error,
            ScenarioError::FigureNeeded {
                node,
                error: FigureError::Unknown { preset, key, .. },
            } if node == "pc" && preset == "gpu:ada-24g" && key == "t_fixed"
        ),
        "{error}"
    );
    // Supplied for the pc's GPU only: now the CPU's is missing.
    let pc_gpu = supplied("gpu:ada-24g", "t_fixed", 8);
    let error = Scenario::from_toml(&with_figures(r#"["t_fixed"]"#, &pc_gpu, "")).unwrap_err();
    assert!(error.to_string().contains("cpu:ddr5-2ch-128g"), "{error}");
    // Supplied for every device that has `t_fixed`: it loads, and the run reads the values.
    let pc = format!("{pc_gpu}{}", supplied("cpu:ddr5-2ch-128g", "t_fixed", 20));
    let mac = supplied("gpu:apple-max-128g-unified", "t_fixed", 5_000);
    let scenario = Scenario::from_toml(&with_figures(r#"["t_fixed"]"#, &pc, &mac))
        .expect("every needed figure is supplied");
    let figure = scenario.nodes[0].figure("gpu:ada-24g", "t_fixed").unwrap();
    assert_eq!((figure.value, figure.label), (Some(8.0), Label::Target));
    assert_eq!(
        scenario.nodes[1]
            .figure("gpu:apple-max-128g-unified", "t_fixed")
            .unwrap()
            .value,
        Some(5_000.0)
    );
    // A drive's unknown is needed only of drives: the mac's internal SSD has no sustained rate.
    let error = Scenario::from_toml(&with_figures(r#"["sustained_read"]"#, "", "")).unwrap_err();
    assert!(
        matches!(&error, ScenarioError::FigureNeeded { node, .. } if node == "mac"),
        "{error}"
    );
    // A known figure needs nothing supplied, and only devices with the key are asked for it.
    assert!(Scenario::from_toml(&with_figures(r#"["w_eff.vram.dense_gemv"]"#, "", "")).is_ok());
    // A need that is no figure is refused.
    assert_eq!(
        Scenario::from_toml(&with_figures(r#"["w_eff.vram.dense_gem"]"#, "", "")).unwrap_err(),
        ScenarioError::UnknownNeed("w_eff.vram.dense_gem".into())
    );
}

/// Without `needs`, nothing is refused at load, and a run still cannot read an unknown figure
/// silently: `figure` refuses it, and gives a supplied or known one.
#[test]
fn a_run_reads_figures_only_through_figure() {
    let scenario = Scenario::from_toml(EXAMPLE).unwrap();
    let pc = &scenario.nodes[0];
    assert!(matches!(
        pc.figure("gpu:ada-24g", "sigma_dev"),
        Err(FigureError::Unknown { .. })
    ));
    assert!(matches!(
        pc.figure("gpu:ada-24g", "no_such_figure"),
        Err(FigureError::Absent { .. })
    ));
    assert_eq!(
        pc.figure("gpu:ada-24g", "w_eff.vram.dense_gemv")
            .unwrap()
            .value,
        Some(800.0)
    );
    // A supplied figure takes the place of a known one on its node only.
    let set = supplied("gpu:ada-24g", "w_eff.vram.dense_gemv", 600);
    let scenario = Scenario::from_toml(&with_figures("[]", &set, "")).unwrap();
    assert_eq!(
        scenario.nodes[0]
            .figure("gpu:ada-24g", "w_eff.vram.dense_gemv")
            .unwrap()
            .value,
        Some(600.0)
    );
}

/// A node supplies figures only for its own presets, only keys they have, and only labelled and
/// sourced.
#[test]
fn a_supplied_figure_is_labelled_sourced_and_one_of_the_nodes_presets() {
    let cases = [
        (
            supplied("gpu:apple-max-128g-unified", "t_fixed", 1),
            ScenarioError::SetForAbsentPreset {
                node: "pc".into(),
                preset: "gpu:apple-max-128g-unified".into(),
            },
        ),
        (
            supplied("gpu:ada-24g", "w_eff.unified.dense_gemv", 1),
            ScenarioError::NotAFigure {
                node: "pc".into(),
                preset: "gpu:ada-24g".into(),
                key: "w_eff.unified.dense_gemv".into(),
            },
        ),
        (
            supplied("gpu:ada-24g", "t_fixed", 1).replace("chosen for this test", " "),
            ScenarioError::BadFigure {
                node: "pc".into(),
                preset: "gpu:ada-24g".into(),
                key: "t_fixed".into(),
            },
        ),
    ];
    for (pc, want) in cases {
        assert_eq!(
            Scenario::from_toml(&with_figures("[]", &pc, "")).unwrap_err(),
            want
        );
    }
    // No label, or a label outside the four, is not a figure.
    for pc in [
        supplied("gpu:ada-24g", "t_fixed", 1).replace("label = \"target\"\n", ""),
        supplied("gpu:ada-24g", "t_fixed", 1).replace("\"target\"", "\"guess\""),
    ] {
        assert!(matches!(
            Scenario::from_toml(&with_figures("[]", &pc, "")).unwrap_err(),
            ScenarioError::Parse(_)
        ));
    }
}
