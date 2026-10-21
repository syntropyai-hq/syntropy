// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rig inventory and `RigFingerprint` of QUAL-E15-T01 (RFC-0061 §2.1, §6.3), end to end.
//!
//! `data/` holds made-up inventories: `inventory.toml` has one rig with every field set and one
//! system-on-chip with the optional fields left out; `rev1.toml` and `rev2.toml` are two
//! revisions of another; `invalid*.toml` each break one rule. The binary tests run
//! `syntropy-bench rigs …` as `lint` and the nightly report do.

use std::path::{Path, PathBuf};
use std::process::Command;

use syntropy_bench::rigs::{self, Inventory};
use syntropy_bench::schema::BenchResult;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/rigs/data")
        .join(name)
}

fn inventory(name: &str) -> Inventory {
    Inventory::parse(&std::fs::read_to_string(data(name)).unwrap()).unwrap()
}

/// Runs `syntropy-bench` with `args`; returns its exit status, stdout and stderr.
fn bench(args: &[&str]) -> (Option<i32>, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_syntropy-bench"))
        .args(args)
        .output()
        .expect("syntropy-bench runs");
    (
        output.status.code(),
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

/// The fingerprints of `data/inventory.toml`'s rigs. A record hashes the same on every OS, so
/// these hold on each platform the tests run on. They were cross-checked against an independent
/// BLAKE3 implementation over the encoding rebuilt from its documentation in `rigs.rs`.
const GOLDEN: [(&str, &str); 2] = [
    (
        "R-FIXTURE-A",
        "d1cdec0efec3dd7c0e07a5345fce93aa375434646efb8f77a0ee970bfa280c73",
    ),
    (
        "R-FIXTURE-SOC",
        "28e11b774c289b46f6ff6cdc8240e3a9e07977d96279a8ee71638a9bacb0ae0d",
    ),
];

#[test]
fn the_fixture_fingerprints_match_their_golden_values() {
    let inventory = inventory("inventory.toml");
    let found: Vec<(String, String)> = inventory
        .rigs
        .iter()
        .map(|rig| (rig.id.clone(), rig.fingerprint().0.to_string()))
        .collect();
    let golden: Vec<(String, String)> = GOLDEN
        .iter()
        .map(|(id, hex)| ((*id).to_owned(), (*hex).to_owned()))
        .collect();
    assert_eq!(found, golden);
}

/// One step of a path into a parsed TOML document.
#[derive(Clone, Debug)]
enum Step {
    Key(String),
    Index(usize),
}

/// Every leaf of `value` (a string, integer or boolean), by its path.
fn leaf_paths(value: &toml::Value, path: &mut Vec<Step>, out: &mut Vec<Vec<Step>>) {
    match value {
        toml::Value::Table(table) => {
            for (key, child) in table {
                path.push(Step::Key(key.clone()));
                leaf_paths(child, path, out);
                path.pop();
            }
        }
        toml::Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                path.push(Step::Index(index));
                leaf_paths(child, path, out);
                path.pop();
            }
        }
        _ => out.push(path.clone()),
    }
}

fn at<'a>(value: &'a mut toml::Value, path: &[Step]) -> &'a mut toml::Value {
    path.iter().fold(value, |value, step| match step {
        Step::Key(key) => value.get_mut(key.as_str()).unwrap(),
        Step::Index(index) => value.get_mut(*index).unwrap(),
    })
}

/// The field name `rigs::diff` gives the leaf at `path` (which starts with `rig`, `<n>`): a
/// list element is named by its `slot`, and a `bios.other` setting by its name.
fn field_name(document: &toml::Value, path: &[Step]) -> String {
    let mut name = String::new();
    let mut node = document;
    for (depth, step) in path.iter().enumerate() {
        match step {
            Step::Key(key) => {
                node = &node[key.as_str()];
                if depth < 2 {
                    continue;
                }
                let in_other = name.ends_with("bios.other");
                if in_other {
                    name.push_str(&format!("[{key}]"));
                } else {
                    if !name.is_empty() {
                        name.push('.');
                    }
                    name.push_str(key);
                }
            }
            Step::Index(index) => {
                node = &node[*index];
                if depth >= 2 {
                    let slot = node["slot"].as_str().unwrap();
                    name.push_str(&format!("[{slot}]"));
                }
            }
        }
    }
    name
}

/// Changes the leaf to another valid value of its field.
fn change(value: &mut toml::Value, key: &str) {
    match value {
        toml::Value::String(text) => match key {
            "pcie" => *text = if text == "gen1x1" { "gen2x1" } else { "gen1x1" }.to_owned(),
            "attachment" => *text = if text == "cpu" { "chipset" } else { "cpu" }.to_owned(),
            "id" => text.push('X'),
            _ => text.push_str("-changed"),
        },
        toml::Value::Integer(n) => *n += 1,
        toml::Value::Boolean(b) => *b = !*b,
        other => panic!("not a leaf: {other:?}"),
    }
}

/// The criteria "changing any listed field changes the fingerprint" and "a fixture observation
/// differing from its record in any field is reported as a mismatch naming that field": every
/// leaf of every fixture rig, changed alone.
#[test]
fn changing_any_field_changes_the_fingerprint_and_is_named_as_a_mismatch() {
    let text = std::fs::read_to_string(data("inventory.toml")).unwrap();
    let document: toml::Value = toml::from_str(&text).unwrap();
    let records = Inventory::parse(&text).unwrap();
    let mut paths = Vec::new();
    leaf_paths(&document, &mut Vec::new(), &mut paths);
    let mut checked = 0;
    for path in &paths {
        let [
            Step::Key(rig_key),
            Step::Index(rig_index),
            ..,
            Step::Key(leaf),
        ] = path.as_slice()
        else {
            continue; // `version`
        };
        assert_eq!(rig_key, "rig");
        let record = &records.rigs[*rig_index];
        let mut changed = document.clone();
        change(at(&mut changed, path), leaf);
        let observed: Inventory = changed.clone().try_into().unwrap();
        let observed = &observed.rigs[*rig_index];
        assert_eq!(observed.problems(), Vec::<String>::new(), "{path:?}");
        assert_ne!(
            observed.fingerprint(),
            record.fingerprint(),
            "changing {path:?} kept the fingerprint"
        );
        let fields: Vec<String> = rigs::diff(record, observed)
            .into_iter()
            .map(|m| m.field)
            .collect();
        let field = field_name(&document, path);
        if leaf == "slot" {
            // The element is now another element: every field of both is named.
            let renamed = field_name(&changed, path);
            assert!(
                fields.contains(&field) && fields.contains(&renamed),
                "{path:?}: {fields:?}"
            );
        } else {
            assert_eq!(fields, std::slice::from_ref(&field), "{path:?}");
        }
        checked += 1;
    }
    // Every leaf of both rigs: 5 + 2 + 4 × 2 + 5 × 2 + 5 × 2 + 4 × 2 + 5 + 2 on R-FIXTURE-A,
    // 5 + 1 + 4 + 3 + 4 + 3 on R-FIXTURE-SOC.
    assert_eq!(checked, 50 + 20);
}

#[test]
fn a_mismatch_says_what_the_record_and_the_rig_have() {
    let record = inventory("rev1.toml").rigs[0].clone();
    let observed = inventory("rev2.toml").rigs[0].clone();
    let shown: Vec<String> = rigs::diff(&record, &observed)
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        shown,
        [
            "`gpu[PCIEX16_1].driver`: the record has `580.65.06`, the rig has `580.82.07`",
            "`memory[DIMM_B2].speed_mts`: the record has `6000`, the rig has `5600`",
        ]
    );
}

#[test]
fn a_bench_key_with_a_new_fingerprint_is_incomparable_with_the_old() {
    let text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/schema/data/result-v1.json"),
    )
    .unwrap();
    let before = inventory("rev1.toml").rigs[0].fingerprint();
    let after = inventory("rev2.toml").rigs[0].fingerprint();
    assert_ne!(before, after);
    let mut old = BenchResult::from_json(&text).unwrap();
    old.key.rig = before;
    let mut same = old.clone();
    same.key.build.commit.0[0] ^= 1;
    let mut new = old.clone();
    new.key.rig = after;
    assert!(old.comparable_with(&same), "only `build` differs");
    assert!(!old.comparable_with(&new), "a new epoch is not comparable");
}

#[test]
fn the_epoch_list_names_each_changed_rig_and_the_fields_that_changed() {
    let (old, new) = (inventory("rev1.toml"), inventory("rev2.toml"));
    let fingerprint =
        |inventory: &Inventory, id: &str| inventory.rig(id).unwrap().fingerprint().0.to_string();
    let expected = [
        format!(
            "R-FIXTURE-A: new epoch {} (was {}); changed: gpu[PCIEX16_1].driver, \
             memory[DIMM_B2].speed_mts",
            fingerprint(&new, "R-FIXTURE-A"),
            fingerprint(&old, "R-FIXTURE-A")
        ),
        format!(
            "R-FIXTURE-B: removed; last epoch {}",
            fingerprint(&old, "R-FIXTURE-B")
        ),
        format!(
            "R-FIXTURE-C: added; first epoch {}",
            fingerprint(&new, "R-FIXTURE-C")
        ),
    ];
    // R-FIXTURE-D only moved and reformatted: same fingerprint, no epoch.
    assert_eq!(
        fingerprint(&old, "R-FIXTURE-D"),
        fingerprint(&new, "R-FIXTURE-D")
    );
    let listed: Vec<String> = rigs::epochs(&old, &new)
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(listed, expected);

    let (code, out, err) = bench(&[
        "rigs",
        "epochs",
        data("rev1.toml").to_str().unwrap(),
        data("rev2.toml").to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "{err}");
    assert_eq!(out.lines().collect::<Vec<_>>(), expected);
    let (code, out, _) = bench(&[
        "rigs",
        "epochs",
        data("rev1.toml").to_str().unwrap(),
        data("rev1.toml").to_str().unwrap(),
    ]);
    assert_eq!((code, out.as_str()), (Some(0), "no epoch changes\n"));
}

#[test]
fn rigs_check_passes_valid_inventories_and_fails_invalid_ones() {
    // The workspace's own inventory, found as `lint` finds it.
    let (code, _, err) = bench(&["rigs", "check"]);
    assert_eq!(code, Some(0), "{err}");
    assert!(
        err.contains("rigs: PASS: ") && err.contains("format version 1"),
        "{err}"
    );

    let (code, _, err) = bench(&[
        "rigs",
        "check",
        "--file",
        data("inventory.toml").to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "{err}");
    assert!(
        err.contains(": 2 rig records, format version 1 (RFC-0061 §6.3)"),
        "{err}"
    );

    for (file, expected) in [
        (
            "invalid.toml",
            "rig `R-FIXTURE-BAD`: `memory`: the slot `DIMM_A1` appears twice",
        ),
        (
            "invalid-link.toml",
            "`gen4x12` is not a PCIe link such as `gen4x16`",
        ),
    ] {
        let (code, _, err) = bench(&["rigs", "check", "--file", data(file).to_str().unwrap()]);
        assert_eq!(code, Some(1), "{file}: {err}");
        assert!(err.contains(expected), "{file}: {err}");
        assert!(err.contains("rigs: FAIL: "), "{file}: {err}");
    }
}

/// Review B1 on #1568, as reproduced: a multi-line `power_plan` in an LF and in a CRLF copy of
/// the fixture would fingerprint differently, so `rigs check` refuses both. Without such a
/// value, the CRLF copy keeps every golden fingerprint.
#[test]
fn a_multi_line_value_fails_the_check_whatever_the_line_endings() {
    let text = std::fs::read_to_string(data("inventory.toml")).unwrap();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("rigs-line-endings");
    std::fs::create_dir_all(&dir).unwrap();
    let multi_line = text.replacen(
        "power_plan = \"performance\"",
        "power_plan = \"\"\"\nbalanced\nprofile\"\"\"",
        1,
    );
    assert_ne!(multi_line, text);
    for (name, body) in [
        ("lf.toml", multi_line.clone()),
        ("crlf.toml", multi_line.replace('\n', "\r\n")),
    ] {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        let (code, _, err) = bench(&["rigs", "check", "--file", path.to_str().unwrap()]);
        assert_eq!(code, Some(1), "{name}: {err}");
        assert!(
            err.contains("rig `R-FIXTURE-A`: `power_plan` is `"),
            "{name}: {err}"
        );
        assert!(err.contains("with a control character"), "{name}: {err}");
    }
    let crlf = Inventory::parse(&text.replace('\n', "\r\n")).unwrap();
    let found: Vec<(&str, String)> = crlf
        .rigs
        .iter()
        .map(|rig| (rig.id.as_str(), rig.fingerprint().0.to_string()))
        .collect();
    let golden: Vec<(&str, String)> = GOLDEN
        .iter()
        .map(|(id, hex)| (*id, (*hex).to_owned()))
        .collect();
    assert_eq!(found, golden);
}

#[test]
fn rigs_fingerprint_prints_one_line_per_rig() {
    let (code, out, err) = bench(&[
        "rigs",
        "fingerprint",
        "--file",
        data("inventory.toml").to_str().unwrap(),
    ]);
    assert_eq!(code, Some(0), "{err}");
    let expected: Vec<String> = GOLDEN
        .iter()
        .map(|(id, hex)| format!("{id} {hex}"))
        .collect();
    assert_eq!(out.lines().collect::<Vec<_>>(), expected);
}

#[test]
fn a_usage_error_exits_two() {
    for args in [
        &["rigs"][..],
        &["rigs", "check", "extra"],
        &["rigs", "epochs", "one"],
    ] {
        let (code, _, err) = bench(args);
        assert_eq!(code, Some(2), "{args:?}: {err}");
        assert!(err.contains("Usage: syntropy-bench"), "{args:?}: {err}");
    }
}

#[test]
fn the_workspace_inventory_is_valid() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rigs::PATH);
    let inventory = Inventory::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(inventory.version, rigs::VERSION);
}
