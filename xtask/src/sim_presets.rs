// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `sim-presets` step of `lint`: every simulator device and drive preset cites the research
//! figures it encodes (QUAL-E8-T03; RFC-0060 §8.5).
//!
//! Each file in [`DIR`] must have:
//! - a file name of `<kind>.<name>.toml`, the name's own `<kind>:` prefix left out;
//! - `sources`, a list with at least one citation;
//! - `figures`, and for each figure either a `label` (`published`, `measured`, `estimate` or
//!   `target`), a non-empty `source`, a `unit` and a `value` or a `range`; or an `unknown` that
//!   says where the research leaves it open, and nothing else.
//!
//! `testkit::sim::presets` loads the same files and refuses the same faults. This step reads them
//! as plain TOML, so `lint` needs no `testkit` build.

use std::fs;
use std::path::Path;

use crate::run::Context;

/// The presets' directory, relative to the workspace root.
pub const DIR: &str = "crates/testkit/sim/presets";

/// The labels a figure may carry (RFC-0060 §8.5, note of 2026-10-23).
const LABELS: [&str; 4] = ["published", "measured", "estimate", "target"];

/// The kinds a preset may describe.
const KINDS: [&str; 3] = ["gpu", "cpu", "nvme"];

/// The `lint` step.
pub fn step(context: &Context) -> Result<(), String> {
    let problems = check_dir(&context.root.join(DIR))?;
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// Every problem in the presets under `dir`, one line each.
fn check_dir(dir: &Path) -> Result<Vec<String>, String> {
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
            check_file(&file, &text)
                .into_iter()
                .map(|problem| format!("{DIR}/{file}: {problem}")),
        );
    }
    Ok(problems)
}

/// The problems in one preset file.
fn check_file(file: &str, text: &str) -> Vec<String> {
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
    match table.get("figures").and_then(toml::Value::as_table) {
        Some(figures) if !figures.is_empty() => {
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

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/sim-presets")
            .join(name)
    }

    #[test]
    fn the_presets_in_the_repository_pass() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        assert_eq!(check_dir(&root.join(DIR)), Ok(Vec::new()));
    }

    #[test]
    fn a_well_formed_preset_passes() {
        assert_eq!(check_dir(&fixture("pass")), Ok(Vec::new()));
    }

    #[test]
    fn a_preset_with_no_source_fails() {
        let problems = check_dir(&fixture("no-source")).unwrap();
        assert_eq!(
            problems,
            [format!(
                "{DIR}/gpu.test.toml: preset `gpu:test` cites no source"
            )]
        );
    }

    #[test]
    fn a_figure_with_no_label_fails() {
        let problems = check_dir(&fixture("no-label")).unwrap();
        assert_eq!(
            problems,
            [format!("{DIR}/gpu.test.toml: figure `memory` has no label")]
        );
    }

    #[test]
    fn each_other_fault_is_named() {
        let problems = check_dir(&fixture("faults")).unwrap();
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
