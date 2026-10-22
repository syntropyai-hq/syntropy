// SPDX-License-Identifier: MIT OR Apache-2.0

//! The device and drive presets (QUAL-E8-T03; RFC-0060 §8.5): every file loads, each figure is
//! labelled and sourced, and the compiled-in list is the directory.

use std::path::Path;

use syntropy_testkit::sim::presets::{
    self, DevicePreset, Figure, FigureError, Label, MemoryTier, PresetError, PresetKind,
};

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

fn known(preset: &DevicePreset, figure: &str) -> (Option<f64>, Option<[f64; 2]>, Label) {
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
    assert_eq!(
        known(ada, "w_eff.vram.dense_gemv"),
        (Some(800.0), None, Label::Estimate)
    );
    let cpu = &all["cpu:ddr5-2ch-128g"];
    assert_eq!(
        known(cpu, "memory_peak_bw"),
        (Some(96.0), None, Label::Published)
    );
    assert_eq!(
        known(cpu, "memory_achieved_bw"),
        (Some(90.01), None, Label::Measured)
    );
    for tier in ["host_pinned", "host_pageable"] {
        assert_eq!(
            known(cpu, &format!("w_eff.{tier}.dense_gemv")),
            (Some(70.0), Some([55.0, 85.0]), Label::Estimate)
        );
    }
    assert_eq!(known(cpu, "memory"), (Some(128.0), None, Label::Target));
    let mac = &all["gpu:apple-max-128g-unified"];
    assert_eq!(
        known(mac, "memory_peak_bw"),
        (Some(546.0), None, Label::Published)
    );
    assert_eq!(known(mac, "eta"), (Some(0.58), None, Label::Estimate));
    assert_eq!(
        known(mac, "w_eff.unified.dense_gemv"),
        (Some(315.0), None, Label::Estimate)
    );
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
        ("gpu:strix-halo-128g-unified", "w_eff.unified.dense_gemv"),
        ("gen4-990pro-class", "throttled_read"),
        ("apple-internal", "sustained_read"),
        ("apple-internal", "throttled_read"),
        ("apple-internal", "qd32_4k_read_iops"),
        ("gen4-generic", "throttled_read"),
        ("gen4-generic", "qd32_4k_read_iops"),
        ("gpu:ada-24g", "sigma_dev"),
        ("gpu:ada-24g", "f_eff"),
        ("gpu:ada-24g", "w_eff.vram.expert_gemv"),
        ("gpu:ada-24g", "w_eff.host_pinned.dense_gemv"),
        ("cpu:ddr5-2ch-128g", "w_eff.host_pinned.expert_gemv"),
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

/// Every preset has every key of its kind: the devices their tiers' W_eff for both kernel
/// classes, the discrete GPU its host link, and the drives the drive keys.
#[test]
fn every_preset_has_its_kinds_keys() {
    let all = presets::by_name();
    let tiers = |name: &str| all[name].tiers().to_vec();
    assert_eq!(tiers("gpu:ada-24g"), ["vram", "host_pinned"]);
    assert_eq!(tiers("gpu:apple-max-128g-unified"), ["unified"]);
    assert_eq!(tiers("gpu:strix-halo-128g-unified"), ["unified"]);
    assert_eq!(tiers("cpu:ddr5-2ch-128g"), ["host_pinned", "host_pageable"]);
    assert_eq!(all["gpu:ada-24g"].memory_tier, Some(MemoryTier::Vram));
    assert_eq!(
        all["gpu:ada-24g"].required_keys(),
        [
            "memory",
            "f_eff",
            "t_fixed",
            "sigma_dev",
            "w_eff.vram.dense_gemv",
            "w_eff.vram.expert_gemv",
            "w_eff.host_pinned.dense_gemv",
            "w_eff.host_pinned.expert_gemv",
            "host_link_pinned_h2d",
        ]
    );
    for preset in all.values() {
        if preset.kind == PresetKind::Nvme {
            assert_eq!(
                preset.required_keys(),
                presets::figure_keys().drive,
                "{}",
                preset.name
            );
        }
        for key in preset.required_keys() {
            assert!(preset.figures.contains_key(&key), "{} {key}", preset.name);
        }
    }
}

const GOOD: &str = r#"
name = "gpu:test"
kind = "gpu"
memory_tier = "unified"
describes = "a test GPU"
sources = ["research §2.1 [S10]"]

[figures.memory]
value = 24
unit = "GB"
label = "published"
source = "research §2.1 [S10]"

[figures."w_eff.unified.dense_gemv"]
unknown = "research §2.1 gives none"

[figures."w_eff.unified.expert_gemv"]
unknown = "research §2.1 gives none"

[figures.f_eff]
unknown = "research §2.1 gives none"

[figures.t_fixed]
unknown = "research §2.2 gives none"

[figures.sigma_dev]
unknown = "research gives none"
"#;

/// A preset that leaves out a key of its kind, names a W_eff key outside its set, or has a
/// memory tier where it should not, or none where it should, does not load.
#[test]
fn a_missing_key_a_stray_w_eff_key_or_a_wrong_memory_tier_does_not_load() {
    let at = |figure: &str| ("gpu:test".to_owned(), figure.to_owned());
    let missing = GOOD.replace(
        "[figures.f_eff]\nunknown = \"research §2.1 gives none\"\n",
        "",
    );
    let (preset, figure) = at("f_eff");
    assert_eq!(
        DevicePreset::from_toml("gpu.test.toml", &missing).unwrap_err(),
        PresetError::MissingFigure { preset, figure }
    );
    for stray in [
        "w_eff",
        "\"w_eff.vram.dense_gemv\"",
        "\"w_eff.unified.dense_gemm\"",
    ] {
        let text = format!("{GOOD}\n[figures.{stray}]\nunknown = \"none\"\n");
        let (preset, figure) = at(stray.trim_matches('"'));
        assert_eq!(
            DevicePreset::from_toml("gpu.test.toml", &text).unwrap_err(),
            PresetError::NotAKey { preset, figure }
        );
    }
    assert_eq!(
        DevicePreset::from_toml(
            "gpu.test.toml",
            &GOOD.replace("memory_tier = \"unified\"\n", "")
        )
        .unwrap_err(),
        PresetError::MemoryTier {
            preset: "gpu:test".into(),
            kind: PresetKind::Gpu
        }
    );
    // Peak rates and the like may stand beside the kind's keys.
    let extra = format!(
        "{GOOD}\n[figures.memory_peak_bw]\nvalue = 1\nunit = \"GB/s\"\nlabel = \"published\"\nsource = \"S1\"\n"
    );
    assert!(DevicePreset::from_toml("gpu.test.toml", &extra).is_ok());
}

#[test]
fn a_preset_without_sources_or_with_an_unsourced_figure_does_not_load() {
    assert!(DevicePreset::from_toml("gpu.test.toml", GOOD).is_ok());
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
        (
            GOOD.replace(r#"unit = "GB""#, r#"unit = " ""#),
            PresetError::NoValue {
                preset: "gpu:test".into(),
                figure: "memory".into(),
            },
        ),
    ];
    for (text, want) in cases {
        assert_eq!(
            DevicePreset::from_toml("gpu.test.toml", &text).unwrap_err(),
            want
        );
    }
    // A figure with no label, or a label outside the four, is not a figure.
    for text in [
        GOOD.replace("label = \"published\"\n", ""),
        GOOD.replace(r#"label = "published""#, r#"label = "verified""#),
    ] {
        assert!(matches!(
            DevicePreset::from_toml("gpu.test.toml", &text).unwrap_err(),
            PresetError::Parse { .. }
        ));
    }
    assert_eq!(
        DevicePreset::from_toml("test.toml", GOOD).unwrap_err(),
        PresetError::FileName {
            file: "test.toml".into(),
            expected: "gpu.test.toml".into()
        }
    );
}

/// `figure` gives a known figure, and refuses an `unknown` or absent one, so a fake cannot read
/// either as a zero.
#[test]
fn figure_refuses_an_unknown_or_absent_figure() {
    let all = presets::by_name();
    let ada = &all["gpu:ada-24g"];
    let known = ada.figure("w_eff.vram.dense_gemv").expect("a known figure");
    assert_eq!((known.value, known.unit.as_str()), (Some(800.0), "GB/s"));
    assert_eq!(
        ada.figure("t_fixed"),
        Err(FigureError::Unknown {
            preset: "gpu:ada-24g".into(),
            key: "t_fixed".into(),
            open: "research §2.2 and §3.4 give t_fixed only for ggml Metal on M3 Ultra; none for this GPU"
                .into(),
        })
    );
    assert_eq!(
        ada.figure("w_eff.unified.dense_gemv"),
        Err(FigureError::Absent {
            preset: "gpu:ada-24g".into(),
            key: "w_eff.unified.dense_gemv".into(),
        })
    );
    // Every required key of every preset is either known or refused as unknown, never absent.
    for preset in all.values() {
        for key in preset.required_keys() {
            assert!(
                !matches!(preset.figure(&key), Err(FigureError::Absent { .. })),
                "{} {key}",
                preset.name
            );
        }
    }
}

/// The figure-key table parses, and is the one `crates/testkit/sim/figure-keys.toml` holds, which
/// the `sim-presets` lint reads.
#[test]
fn the_figure_key_table_parses() {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("sim/figure-keys.toml");
    assert_eq!(
        std::fs::read_to_string(file).unwrap(),
        presets::FIGURE_KEYS_FILE
    );
    let keys = presets::figure_keys();
    assert_eq!(keys.device, ["memory", "f_eff", "t_fixed", "sigma_dev"]);
    assert_eq!(keys.discrete_gpu, ["host_link_pinned_h2d"]);
    assert_eq!(keys.kernel_classes, ["dense_gemv", "expert_gemv"]);
    assert_eq!(keys.drive.len(), 7);
    assert_eq!(keys.tiers.gpu_vram, ["vram", "host_pinned"]);
    assert_eq!(keys.tiers.gpu_unified, ["unified"]);
    assert_eq!(keys.tiers.cpu, ["host_pinned", "host_pageable"]);
    // Every key some kind requires, each W_eff key once per tier and class.
    let all = presets::all_keys();
    assert_eq!(all.len(), 4 + 7 + 1 + 4 * 2);
    assert!(all.contains(&"w_eff.host_pageable.expert_gemv".to_owned()));
}
