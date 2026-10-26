// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `sim-presets` step of `lint`: every simulator device and drive preset cites the research
//! figures it encodes (QUAL-E8-T03; RFC-0060 §8.5).
//!
//! Each file in [`DIR`] must have:
//! - a file name of `<kind>.<name>.toml`, the name's own `<kind>:` prefix left out;
//! - `sources`, a list with at least one citation;
//! - `figures`, and for each figure either a `label` (`published`, `measured`, `estimate` or
//!   `target`), a non-empty `source`, a `unit` and a `value` or a `range`; or an `unknown` that
//!   says where the research leaves it open, and nothing else;
//! - every figure key of its kind, known or `unknown`, and no `w_eff.*` key outside them. A
//!   device has `memory`, `f_eff`, `t_fixed`, `sigma_dev` and `w_eff.<tier>.<kernel class>` for
//!   each tier it reads and each of `dense_gemv` and `expert_gemv`; a discrete GPU also has
//!   `host_link_pinned_h2d`. A drive has `seq_read`, `sustained_read`, `throttled_read`,
//!   `heating_time`, `qd1_4k_read_iops`, `qd32_4k_read_iops` and `read_latency`;
//! - a `memory_tier` of `vram` or `unified` if it is a `gpu`, and none otherwise.
//!
//! The keys come from [`KEYS`], the table `testkit::sim::presets` compiles in, so the loader and
//! this step check one table. `testkit` loads the same files and refuses the same faults. This
//! step reads them as plain TOML, so `lint` needs no `testkit` build: `xtask` depends on no
//! workspace crate (`ci/layers.toml`), but it reads their files.

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::run::Context;

/// The presets' directory, relative to the workspace root.
pub const DIR: &str = "crates/testkit/sim/presets";

/// The labels a figure may carry (RFC-0060 §8.5, note of 2026-10-23).
const LABELS: [&str; 4] = ["published", "measured", "estimate", "target"];

/// The kinds a preset may describe.
const KINDS: [&str; 3] = ["gpu", "cpu", "nvme"];

/// The figure-key table, relative to the workspace root: the file `testkit::sim::presets`
/// compiles in.
pub const KEYS: &str = "crates/testkit/sim/figure-keys.toml";

/// The figure keys each preset kind must have, as [`KEYS`] holds them.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Table {
    device: Vec<String>,
    discrete_gpu: Vec<String>,
    drive: Vec<String>,
    kernel_classes: Vec<String>,
    tiers: Tiers,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Tiers {
    gpu_vram: Vec<String>,
    gpu_unified: Vec<String>,
    cpu: Vec<String>,
}

/// Reads the figure-key table at `path`.
fn load_table(path: &Path) -> Result<Table, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    toml::from_str(&text).map_err(|error| format!("{KEYS}: not the figure-key table: {error}"))
}

/// The figure keys a preset of `kind` with `memory_tier` must have.
fn required_keys(table: &Table, kind: &str, memory_tier: Option<&str>) -> Vec<String> {
    let tiers: &[String] = match (kind, memory_tier) {
        ("gpu", Some("vram")) => &table.tiers.gpu_vram,
        ("gpu", Some("unified")) => &table.tiers.gpu_unified,
        ("cpu", _) => &table.tiers.cpu,
        ("nvme", _) => return table.drive.clone(),
        _ => &[],
    };
    let mut keys = table.device.clone();
    for tier in tiers {
        for class in &table.kernel_classes {
            keys.push(format!("w_eff.{tier}.{class}"));
        }
    }
    if kind == "gpu" && memory_tier == Some("vram") {
        keys.extend(table.discrete_gpu.iter().cloned());
    }
    keys
}

/// The `lint` step.
pub fn step(context: &Context) -> Result<(), String> {
    let table = load_table(&context.root.join(KEYS))?;
    let problems = check_dir(&context.root.join(DIR), &table)?;
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// Every problem in the presets under `dir`, one line each.
fn check_dir(dir: &Path, table: &Table) -> Result<Vec<String>, String> {
    let entries = fs::read_dir(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    let mut files: Vec<String> = entries
        .map(|entry| {
            entry
                .map_err(|error| format!("{}: {error}", dir.display()))
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
        })
        .collect::<Result<_, _>>()?;
    files.sort();
    if files.is_empty() {
        return Err(format!("{}: no presets", dir.display()));
    }
    let mut problems = Vec::new();
    for file in files {
        if !file.ends_with(".toml") {
            problems.push(format!(
                "{DIR}/{file}: not a preset; presets are .toml files"
            ));
            continue;
        }
        let text = fs::read_to_string(dir.join(&file))
            .map_err(|error| format!("{DIR}/{file}: {error}"))?;
        problems.extend(
            check_file(&file, &text, table)
                .into_iter()
                .map(|problem| format!("{DIR}/{file}: {problem}")),
        );
    }
    Ok(problems)
}

/// The problems in one preset file.
fn check_file(file: &str, text: &str, keys: &Table) -> Vec<String> {
    let table: toml::Table = match toml::from_str(text) {
        Ok(table) => table,
        Err(error) => return vec![format!("not TOML: {error}")],
    };
    let mut problems = Vec::new();
    let text_of = |key: &str| table.get(key).and_then(toml::Value::as_str);
    let (Some(name), Some(kind)) = (text_of("name"), text_of("kind")) else {
        return vec!["needs a `name` and a `kind`".to_owned()];
    };
    if !KINDS.contains(&kind) {
        problems.push(format!("kind `{kind}` is not one of {}", KINDS.join(", ")));
    }
    let bare = name.strip_prefix(&format!("{kind}:")).unwrap_or(name);
    let expected = format!("{kind}.{bare}.toml");
    if file != expected {
        problems.push(format!("preset `{name}` must be in {expected}"));
    }
    let cited = table
        .get("sources")
        .and_then(toml::Value::as_array)
        .is_some_and(|sources| {
            sources
                .iter()
                .any(|source| source.as_str().is_some_and(|text| !text.trim().is_empty()))
        });
    if !cited {
        problems.push(format!("preset `{name}` cites no source"));
    }
    let memory_tier = text_of("memory_tier");
    match (kind, memory_tier) {
        ("gpu", Some("vram" | "unified")) | ("cpu" | "nvme", None) => {}
        ("gpu", Some(tier)) => {
            problems.push(format!("memory_tier `{tier}` is not `vram` or `unified`"))
        }
        ("gpu", None) => problems.push("a gpu needs `memory_tier`, `vram` or `unified`".into()),
        (_, Some(_)) => problems.push("has `memory_tier`, which only a gpu has".into()),
        // A kind outside the three is already named above.
        (_, None) => {}
    }
    match table.get("figures").and_then(toml::Value::as_table) {
        Some(figures) if !figures.is_empty() => {
            let required = required_keys(keys, kind, memory_tier);
            for key in &required {
                if !figures.contains_key(key) {
                    problems.push(format!(
                        "has no figure `{key}`; write it as `unknown`, with where the research \
                         leaves it open, if the research gives none"
                    ));
                }
            }
            for key in figures.keys() {
                if (key == "w_eff" || key.starts_with("w_eff.")) && !required.contains(key) {
                    problems.push(format!(
                        "figure `{key}` is not one of its W_eff keys, `w_eff.<tier>.<kernel class>`"
                    ));
                }
            }
            for (figure, value) in figures {
                problems.extend(
                    check_figure(value)
                        .into_iter()
                        .map(|problem| format!("figure `{figure}` {problem}")),
                );
            }
        }
        _ => problems.push(format!("preset `{name}` has no figures")),
    }
    problems
}

/// What is wrong with one figure.
fn check_figure(figure: &toml::Value) -> Vec<String> {
    let Some(figure) = figure.as_table() else {
        return vec!["is not a table".to_owned()];
    };
    let filled = |key: &str| {
        figure
            .get(key)
            .and_then(toml::Value::as_str)
            .is_some_and(|text| !text.trim().is_empty())
    };
    if figure.contains_key("unknown") {
        let mut problems = Vec::new();
        if !filled("unknown") {
            problems.push("is unknown but does not say where the research leaves it open".into());
        }
        if figure.len() > 1 {
            problems.push("is unknown, so it has no value, label or source".into());
        }
        return problems;
    }
    let mut problems = Vec::new();
    match figure.get("label").and_then(toml::Value::as_str) {
        Some(label) if LABELS.contains(&label) => {}
        Some(label) => problems.push(format!(
            "has label `{label}`, which is not one of {}",
            LABELS.join(", ")
        )),
        None => problems.push("has no label".into()),
    }
    if !filled("source") {
        problems.push("cites no source".into());
    }
    if !filled("unit") {
        problems.push("has no unit".into());
    }
    let number = |value: &toml::Value| value.as_float().or(value.as_integer().map(|n| n as f64));
    let value = figure.get("value").and_then(number);
    let range = figure
        .get("range")
        .and_then(toml::Value::as_array)
        .and_then(|range| match range.as_slice() {
            [lo, hi] => Some((number(lo)?, number(hi)?)),
            _ => None,
        });
    if value.is_none() && range.is_none() {
        problems.push("has no value or range".into());
    }
    if range.is_some_and(|(lo, hi)| lo > hi) {
        problems.push("has a range that runs backwards".into());
    }
    problems
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    /// The repository's figure-key table.
    fn table() -> Table {
        load_table(&root().join(KEYS)).expect("the repository's table")
    }

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/sim-presets")
            .join(name)
    }

    #[test]
    fn the_presets_in_the_repository_pass() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        assert_eq!(check_dir(&root.join(DIR), &table()), Ok(Vec::new()));
    }

    /// The table is `testkit`'s file, and this step refuses a table it cannot read, rather than
    /// checking against nothing.
    #[test]
    fn the_keys_come_from_testkits_table() {
        let table = table();
        assert_eq!(
            required_keys(&table, "gpu", Some("vram")),
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
        assert_eq!(required_keys(&table, "nvme", None).len(), 7);
        let missing = load_table(&root().join("no/such/figure-keys.toml")).unwrap_err();
        assert!(missing.contains("no/such/figure-keys.toml"), "{missing}");
        let dir = std::env::temp_dir().join(format!("sim-presets-table-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("figure-keys.toml");
        std::fs::write(&bad, "device = [\"memory\"]\n").unwrap();
        let error = load_table(&bad).unwrap_err();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(error.contains("not the figure-key table"), "{error}");
    }

    #[test]
    fn a_well_formed_preset_passes() {
        assert_eq!(check_dir(&fixture("pass"), &table()), Ok(Vec::new()));
    }

    #[test]
    fn a_preset_with_no_source_fails() {
        let problems = check_dir(&fixture("no-source"), &table()).unwrap();
        assert_eq!(
            problems,
            [format!(
                "{DIR}/gpu.test.toml: preset `gpu:test` cites no source"
            )]
        );
    }

    #[test]
    fn a_figure_with_no_label_fails() {
        let problems = check_dir(&fixture("no-label"), &table()).unwrap();
        assert_eq!(
            problems,
            [format!("{DIR}/gpu.test.toml: figure `memory` has no label")]
        );
    }

    #[test]
    fn a_missing_key_a_stray_w_eff_key_or_a_misplaced_memory_tier_fails() {
        let problems = check_dir(&fixture("keys"), &table()).unwrap();
        let open = "; write it as `unknown`, with where the research leaves it open, if the research gives none";
        let want = [
            "cpu.tiered.toml: has `memory_tier`, which only a gpu has".to_owned(),
            format!("gpu.discrete.toml: has no figure `w_eff.host_pinned.expert_gemv`{open}"),
            format!("gpu.discrete.toml: has no figure `host_link_pinned_h2d`{open}"),
            "gpu.discrete.toml: figure `w_eff` is not one of its W_eff keys, `w_eff.<tier>.<kernel class>`".to_owned(),
            "gpu.discrete.toml: figure `w_eff.vram.dense_gemm` is not one of its W_eff keys, `w_eff.<tier>.<kernel class>`".to_owned(),
            "gpu.no-tier.toml: a gpu needs `memory_tier`, `vram` or `unified`".to_owned(),
        ];
        let got: Vec<&str> = problems
            .iter()
            .map(|problem| problem.strip_prefix(&format!("{DIR}/")).unwrap())
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn each_other_fault_is_named() {
        let problems = check_dir(&fixture("faults"), &table()).unwrap();
        let want = [
            "nvme.drive.toml: figure `seq_read` has label `verified`, which is not one of published, measured, estimate, target",
            "nvme.drive.toml: figure `seq_read` cites no source",
            "nvme.drive.toml: figure `sustained_read` is unknown but does not say where the research leaves it open",
            "nvme.drive.toml: figure `throttled_read` is unknown, so it has no value, label or source",
            "nvme.drive.toml: figure `iops` has no value or range",
            "nvme.drive.toml: figure `latency` has a range that runs backwards",
            "wrong-name.toml: preset `cpu:named` must be in cpu.named.toml",
            "wrong-name.toml: preset `cpu:named` has no figures",
        ];
        let got: Vec<&str> = problems
            .iter()
            .map(|problem| problem.strip_prefix(&format!("{DIR}/")).unwrap())
            .collect();
        for line in want {
            assert!(got.contains(&line), "missing {line:?} in {got:#?}");
        }
        assert_eq!(got.len(), want.len(), "{got:#?}");
    }
}
