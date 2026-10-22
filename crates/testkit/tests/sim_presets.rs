// SPDX-License-Identifier: MIT OR Apache-2.0

//! The device and drive presets (QUAL-E8-T03; RFC-0060 §8.5): every file loads, each figure is
//! labelled and sourced, and the compiled-in list is the directory.

use std::path::Path;

use syntropy_testkit::sim::presets::{self, Figure, Label, Preset, PresetError, PresetKind};

#[test]
fn the_compiled_in_presets_are_the_directory() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("sim/presets");
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .expect("the presets directory")
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    on_disk.sort();
    let compiled: Vec<&str> = presets::ALL.iter().map(|(file, _)| *file).collect();
    assert_eq!(on_disk, compiled);
    for (file, text) in presets::ALL {
        assert_eq!(
            std::fs::read_to_string(dir.join(file)).unwrap(),
            text,
            "{file}"
        );
    }
}

#[test]
fn every_preset_loads_with_the_names_rfc_0060s_example_uses() {
    let all = presets::by_name();
    let names: Vec<&str> = all.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        [
            "apple-internal",
            "cpu:ddr5-2ch-128g",
            "gen4-990pro-class",
            "gen4-generic",
            "gpu:ada-24g",
            "gpu:apple-max-128g-unified",
            "gpu:strix-halo-128g-unified",
        ]
    );
    for preset in all.values() {
        assert!(!preset.sources.is_empty());
        let drive = preset.kind == PresetKind::Nvme;
        assert_eq!(drive, !preset.name.contains(':'), "{}", preset.name);
    }
}

fn known(preset: &Preset, figure: &str) -> (Option<f64>, Option<[f64; 2]>, Label) {
    match &preset.figures[figure] {
        Figure::Known(known) => (known.value, known.range, known.label),
        Figure::Unknown { unknown } => panic!("{figure} is unknown: {unknown}"),
    }
}

#[test]
fn the_figures_are_the_research_notes() {
    let all = presets::by_name();
    let ada = &all["gpu:ada-24g"];
    assert_eq!(
        known(ada, "memory_peak_bw"),
        (Some(1008.0), None, Label::Published)
    );
    assert_eq!(
        known(ada, "host_link_pinned_h2d"),
        (Some(24.0), None, Label::Measured)
    );
    assert_eq!(known(ada, "w_eff"), (Some(800.0), None, Label::Estimate));
    let cpu = &all["cpu:ddr5-2ch-128g"];
    assert_eq!(
        known(cpu, "memory_peak_bw"),
        (Some(96.0), None, Label::Published)
    );
    assert_eq!(
        known(cpu, "memory_achieved_bw"),
        (Some(90.01), None, Label::Measured)
    );
    assert_eq!(
        known(cpu, "w_eff"),
        (Some(70.0), Some([55.0, 85.0]), Label::Estimate)
    );
    assert_eq!(known(cpu, "memory"), (Some(128.0), None, Label::Target));
    let mac = &all["gpu:apple-max-128g-unified"];
    assert_eq!(
        known(mac, "memory_peak_bw"),
        (Some(546.0), None, Label::Published)
    );
    assert_eq!(known(mac, "eta"), (Some(0.58), None, Label::Estimate));
    let drive = &all["gen4-990pro-class"];
    assert_eq!(
        known(drive, "seq_read"),
        (Some(7450.0), None, Label::Published)
    );
    assert_eq!(
        known(drive, "sustained_read"),
        (Some(6.0), None, Label::Estimate)
    );
    // What the research leaves open stays open.
    for (preset, figure) in [
        ("gpu:strix-halo-128g-unified", "w_eff"),
        ("gen4-990pro-class", "throttled_read"),
        ("apple-internal", "sustained_read"),
        ("gpu:ada-24g", "sigma_dev"),
    ] {
        assert!(
            matches!(all[preset].figures[figure], Figure::Unknown { .. }),
            "{preset} {figure}"
        );
    }
    // A generic Gen4 drive is known only by its link.
    let generic = &all["gen4-generic"];
    let known_figures: Vec<&str> = generic
        .figures
        .iter()
        .filter(|(_, figure)| matches!(figure, Figure::Known(_)))
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(known_figures, ["link_raw"]);
    assert_eq!(
        known(generic, "link_raw"),
        (Some(7.88), None, Label::Published)
    );
}

const GOOD: &str = r#"
name = "gpu:test"
kind = "gpu"
describes = "a test GPU"
sources = ["research §2.1 [S10]"]

[figures.memory]
value = 24
unit = "GB"
label = "published"
source = "research §2.1 [S10]"

[figures.t_fixed]
unknown = "research §2.2 gives none"
"#;

#[test]
fn a_preset_without_sources_or_with_an_unsourced_figure_does_not_load() {
    assert!(Preset::from_toml("gpu.test.toml", GOOD).is_ok());
    let cases = [
        (
            GOOD.replace(r#"sources = ["research §2.1 [S10]"]"#, "sources = []"),
            PresetError::NoSources("gpu:test".into()),
        ),
        (
            GOOD.replace(r#"source = "research §2.1 [S10]""#, r#"source = """#),
            PresetError::NoSource {
                preset: "gpu:test".into(),
                figure: "memory".into(),
            },
        ),
        (
            GOOD.replace(
                r#"unknown = "research §2.2 gives none""#,
                r#"unknown = " ""#,
            ),
            PresetError::NoSource {
                preset: "gpu:test".into(),
                figure: "t_fixed".into(),
            },
        ),
        (
            GOOD.replace("value = 24", "range = [30, 20]"),
            PresetError::NoValue {
                preset: "gpu:test".into(),
                figure: "memory".into(),
            },
        ),
    ];
    for (text, want) in cases {
        assert_eq!(Preset::from_toml("gpu.test.toml", &text).unwrap_err(), want);
    }
    // A figure with no label, or a label outside the four, is not a figure.
    for text in [
        GOOD.replace("label = \"published\"\n", ""),
        GOOD.replace(r#"label = "published""#, r#"label = "verified""#),
    ] {
        assert!(matches!(
            Preset::from_toml("gpu.test.toml", &text).unwrap_err(),
            PresetError::Parse { .. }
        ));
    }
    assert_eq!(
        Preset::from_toml("test.toml", GOOD).unwrap_err(),
        PresetError::FileName {
            file: "test.toml".into(),
            expected: "gpu.test.toml".into()
        }
    );
}
