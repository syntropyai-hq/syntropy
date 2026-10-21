// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the tolerance-table loader, the release table, stamping and `ToleranceMissing`.
//!
//! Every number in these tests is a synthetic input chosen to exercise a rule. None is a
//! tolerance: the release table holds none until calibration (COMP-E14-T03) publishes them.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use proptest::prelude::*;
use syntropy_core::{DTypeClass, ErrorClass};

use super::*;

const PENDING: &str = r#"{ status = "pending" }"#;

/// Renders a complete table in which every entry is pending, except the entries named in
/// `set` (by dotted path, such as `end_to_end.bits4.pooled`), which take the given inline
/// value, and those named in `drop`, which are left out. `extra` is appended verbatim.
fn render(version: u32, set: &[(&str, &str)], drop: &[&str], extra: &str) -> String {
    let set: BTreeMap<&str, &str> = set.iter().copied().collect();
    let mut out = format!("schema = {SCHEMA}\nversion = {version}\n");
    let line = |out: &mut String, path: String, key: &str| {
        if drop.contains(&path.as_str()) {
            return;
        }
        let value = set.get(path.as_str()).copied().unwrap_or(PENDING);
        out.push_str(&format!("{key} = {value}\n"));
    };
    for (_, class) in CLASS_KEYS {
        out.push_str(&format!("[end_to_end.{class}]\n"));
        for scope in Scope::ALL {
            line(
                &mut out,
                format!("end_to_end.{class}.{}", scope.key()),
                scope.key(),
            );
        }
        out.push_str(&format!("[design_effect.{class}]\n"));
        for c in Category::ALL {
            line(
                &mut out,
                format!("design_effect.{class}.{}", c.key()),
                c.key(),
            );
        }
    }
    out.push_str("[canary]\n");
    for (_, class) in CLASS_KEYS {
        line(&mut out, format!("canary.{class}"), class);
    }
    for (_, class) in CLASS_KEYS {
        for kind in LayerKind::ALL {
            out.push_str(&format!("[layer_local.{class}.{}]\n", kind.key()));
            for s in kind.statistics() {
                let path = format!("layer_local.{class}.{}.{}", kind.key(), s.key());
                line(&mut out, path, s.key());
            }
        }
    }
    out.push_str(extra);
    out
}

fn pending_table() -> String {
    render(1, &[], &[], "")
}

/// A synthetic published triple for every scope of `class`.
fn publish_class(class: &str, status: &str) -> Vec<(String, String)> {
    Scope::ALL
        .iter()
        .map(|s| {
            (
                format!("end_to_end.{class}.{}", s.key()),
                format!(
                    r#"{{ status = "{status}", kl_mean = 0.0021, kl_p99 = 0.034, top1 = 0.0062 }}"#
                ),
            )
        })
        .collect()
}

fn with_published(classes: &[(&str, &str)]) -> String {
    let owned: Vec<(String, String)> = classes
        .iter()
        .flat_map(|(c, s)| publish_class(c, s))
        .collect();
    let set: Vec<(&str, &str)> = owned
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    render(1, &set, &[], "")
}

fn parse_err(text: &str) -> TableError {
    ToleranceTable::parse(text).expect_err("the table must be rejected")
}

fn invalid_entry(err: &TableError) -> &str {
    match err {
        TableError::Invalid { entry, .. } => entry,
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// A fresh directory under the system temporary directory.
fn temp_dir() -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "syntropy-tolerances-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("create a temporary directory");
    dir
}

// ---------------------------------------------------------------------------------------------
// The release table
// ---------------------------------------------------------------------------------------------

#[test]
fn release_table_loads_with_its_version() {
    let table = ToleranceTable::release().expect("the embedded table loads");
    assert_eq!(table.version(), RELEASE_VERSION);
}

/// The loader reads a `C_ll` entry for each (layer kind, class, statistic) RFC-0042 §5.2
/// lists: mean and p99 `rel_l2` for each of the six kinds, plus top-1 and `KL_K` for the head
/// window, for each of the five classes.
#[test]
fn release_table_has_a_layer_local_entry_for_every_kind_class_and_statistic() {
    let table = ToleranceTable::release().expect("the embedded table loads");
    let mut read = 0;
    for kind in LayerKind::ALL {
        for class in DTypeClass::ALL {
            for statistic in kind.statistics() {
                let entry = table.layer_local(kind, class, *statistic);
                assert!(
                    entry.is_some(),
                    "no C_ll entry for {kind:?} {class:?} {statistic:?}"
                );
                read += 1;
            }
        }
    }
    assert_eq!(read, 5 * (5 * 2 + 4));
    assert_eq!(table.layer_local.len(), read);
}

#[test]
fn layer_kinds_and_statistics_follow_rfc_0042_section_5_2() {
    let keys: Vec<&str> = LayerKind::ALL.iter().map(|k| k.key()).collect();
    assert_eq!(
        keys,
        [
            "dense_ffn",
            "moe_ffn",
            "full_attention",
            "sliding_window_attention",
            "linear_attention",
            "head_window"
        ]
    );
    for kind in LayerKind::ALL {
        let stats: Vec<&str> = kind.statistics().iter().map(|s| s.key()).collect();
        if kind == LayerKind::HeadWindow {
            assert_eq!(stats, ["rel_l2_mean", "rel_l2_p99", "top1", "kl_k"]);
        } else {
            assert_eq!(stats, ["rel_l2_mean", "rel_l2_p99"]);
        }
    }
    let table = ToleranceTable::release().expect("the embedded table loads");
    assert_eq!(
        table.layer_local(LayerKind::DenseFfn, DTypeClass::Bits4, LayerStatistic::Top1),
        None,
        "top-1 has a ceiling only for the head window"
    );
}

/// The loader reads a `τ_def` entry for each certified family: `llama` and `qwen_moe` from
/// M0 (RFC-0042 §7.1; RFC-0043 §9.2).
#[test]
fn release_table_has_a_definitional_entry_for_each_m0_family() {
    let table = ToleranceTable::release().expect("the embedded table loads");
    for family in ["llama", "qwen_moe"] {
        assert!(
            table.definitional(family).is_some(),
            "no tau_def entry for {family}"
        );
    }
    assert_eq!(table.families().collect::<Vec<_>>(), ["llama", "qwen_moe"]);
}

/// No value in the release table: calibration has not run, so every entry is pending.
#[test]
fn release_table_holds_no_values_before_calibration() {
    let table = ToleranceTable::release().expect("the embedded table loads");
    for class in DTypeClass::ALL {
        for scope in Scope::ALL {
            assert!(table.end_to_end(class, scope).is_pending());
        }
        for category in Category::ALL {
            assert!(table.design_effect(class, category).is_pending());
        }
        assert!(table.canary(class).is_pending());
        assert_eq!(table.class_ceiling(class), None);
    }
    assert!(table.layer_local.values().all(Entry::is_pending));
    assert!(table.definitional.values().all(Entry::is_pending));
    assert_eq!(table.ops().count(), 0);
}

/// Release stamping: the build embeds the newest `numerics/tolerances-v<N>.toml`, byte for
/// byte, and [`RELEASE_VERSION`] is its `N`.
#[test]
fn the_build_embeds_the_newest_table() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../numerics");
    let newest = std::fs::read_dir(&dir)
        .expect("read numerics/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter_map(|p| version_from_file_name(&p).map(|v| (v, p)))
        .max()
        .expect("numerics/ holds a table");
    assert_eq!(newest.0, RELEASE_VERSION);
    let text = std::fs::read_to_string(&newest.1).expect("read the newest table");
    assert_eq!(text, RELEASE_TEXT);
    let loaded = ToleranceTable::load(&newest.1).expect("the newest table loads by path");
    assert_eq!(
        loaded,
        ToleranceTable::release().expect("the embedded table loads")
    );
}

#[test]
fn class_keys_cover_every_class() {
    let classes: Vec<DTypeClass> = CLASS_KEYS.iter().map(|(c, _)| *c).collect();
    let mut all = DTypeClass::ALL.to_vec();
    all.reverse();
    assert_eq!(classes, all, "highest precision first");
    for class in DTypeClass::ALL {
        assert_eq!(class_from_key(class_key(class)), Some(class));
    }
}

// ---------------------------------------------------------------------------------------------
// The release blocker: `ToleranceMissing`
// ---------------------------------------------------------------------------------------------

#[test]
fn tolerance_missing_is_numerics_code_1_class_internal() {
    assert_eq!(ToleranceMissing::CODE, 1);
    assert_eq!(ToleranceMissing::NAME, "ToleranceMissing");
    assert_eq!(ToleranceMissing::CLASS, ErrorClass::Internal);
}

/// A release build refuses to start a model whose class has no published ceiling; with the
/// release table, that is every class.
#[test]
fn release_build_refuses_every_class_of_the_release_table() {
    let table = ToleranceTable::release().expect("the embedded table loads");
    for class in DTypeClass::ALL {
        let err = table
            .gate_model_start(class, BuildKind::Release)
            .expect_err("a pending class must not start in a release build");
        assert_eq!(err.classes(), [class]);
        assert_eq!(err.version(), RELEASE_VERSION);
        assert_eq!(
            table.gate_model_start(class, BuildKind::NonRelease),
            Ok(None),
            "a non-release build starts it without a ceiling"
        );
    }
}

#[test]
fn release_build_starts_a_class_with_published_ceilings() {
    let table = ToleranceTable::parse(&with_published(&[("bits4", "provisional")]))
        .expect("the table loads");
    let ceiling = table
        .gate_model_start(DTypeClass::Bits4, BuildKind::Release)
        .expect("bits4 has ceilings")
        .expect("bits4 has ceilings");
    assert!(ceiling.provisional);
    assert_eq!(ceiling.class, DTypeClass::Bits4);
    let t = Triple {
        kl_mean: 0.0021,
        kl_p99: 0.034,
        top1: 0.0062,
    };
    assert_eq!(ceiling.pooled, t);
    assert_eq!(ceiling.category(Category::Stress), t);
    assert_eq!(ceiling.scope(Scope::Category(Category::Chat)), t);
    assert!(
        table
            .gate_model_start(DTypeClass::Int8, BuildKind::Release)
            .is_err()
    );
}

#[test]
fn check_release_lists_every_supported_class_without_a_ceiling() {
    let table =
        ToleranceTable::parse(&with_published(&[("int8", "final")])).expect("the table loads");
    // M0's exit classes (RFC-0042 §6.2), in no particular order.
    let supported = [DTypeClass::Bits4, DTypeClass::Float16, DTypeClass::Int8];
    let err = table
        .check_release(&supported)
        .expect_err("two classes lack ceilings");
    assert_eq!(err.classes(), [DTypeClass::Float16, DTypeClass::Bits4]);
    assert_eq!(
        err.to_string(),
        "ToleranceMissing: tolerances-v1 has no published end-to-end ceiling for dtype classes \
         float16, bits4; a release build does not start a model of such a class"
    );
    assert_eq!(table.check_release(&[DTypeClass::Int8]), Ok(()));
    assert_eq!(table.check_release(&[]), Ok(()));
}

proptest! {
    /// For any set of published classes, a release build refuses exactly the pending ones,
    /// and the release check lists exactly the supported pending ones.
    #[test]
    fn release_gate_refuses_exactly_the_pending_classes(
        published in proptest::collection::btree_set(0usize..5, 0..=5),
        supported in proptest::collection::btree_set(0usize..5, 0..=5),
    ) {
        let classes: Vec<(&str, &str)> =
            published.iter().map(|&i| (CLASS_KEYS[i].1, "final")).collect();
        let table = ToleranceTable::parse(&with_published(&classes)).expect("the table loads");
        for (i, (class, _)) in CLASS_KEYS.iter().enumerate() {
            let started = table.gate_model_start(*class, BuildKind::Release).is_ok();
            prop_assert_eq!(started, published.contains(&i));
        }
        let supported_classes: Vec<DTypeClass> =
            supported.iter().map(|&i| CLASS_KEYS[i].0).collect();
        let expected: Vec<DTypeClass> = supported
            .iter()
            .filter(|i| !published.contains(i))
            .map(|&i| CLASS_KEYS[i].0)
            .collect();
        match table.check_release(&supported_classes) {
            Ok(()) => prop_assert!(expected.is_empty()),
            Err(e) => prop_assert_eq!(e.classes(), expected.as_slice()),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Stamping and overrides
// ---------------------------------------------------------------------------------------------

#[test]
fn the_embedded_table_is_stamped_official() {
    let (table, stamp) = ToleranceTable::for_build(None).expect("the embedded table loads");
    assert_eq!(table.version(), RELEASE_VERSION);
    assert!(stamp.is_official());
    assert_eq!(stamp.contract_ref(), "tolerances-v1");
}

#[test]
fn an_override_is_stamped_unofficial() {
    let dir = temp_dir();
    let path = dir.join("tolerances-v7.toml");
    std::fs::write(&path, render(7, &[], &[], "")).expect("write the override");
    let (table, stamp) = ToleranceTable::for_build(Some(&path)).expect("the override loads");
    assert_eq!(table.version(), 7);
    assert!(!stamp.is_official());
    assert_eq!(stamp.source, TableSource::Override(path.clone()));
    assert_eq!(stamp.contract_ref(), "tolerances-v7+unofficial");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn the_file_name_must_match_the_version() {
    let dir = temp_dir();
    let path = dir.join("tolerances-v2.toml");
    std::fs::write(&path, pending_table()).expect("write the table");
    assert_eq!(
        ToleranceTable::load(&path),
        Err(TableError::Version {
            expected: 2,
            found: 1
        })
    );
    for name in [
        "tolerances.toml",
        "tolerances-v0.toml",
        "tolerances-v01.toml",
        "tolerances-v1.json",
        "tolerances-vx.toml",
    ] {
        let path = dir.join(name);
        assert_eq!(
            ToleranceTable::load(&path),
            Err(TableError::FileName { path: path.clone() }),
            "{name}"
        );
    }
    let absent = dir.join("tolerances-v3.toml");
    assert!(matches!(
        ToleranceTable::load(&absent),
        Err(TableError::Read { .. })
    ));
    std::fs::remove_dir_all(dir).ok();
}

// ---------------------------------------------------------------------------------------------
// Completeness and keys
// ---------------------------------------------------------------------------------------------

#[test]
fn a_complete_pending_table_loads() {
    let table = ToleranceTable::parse(&pending_table()).expect("the table loads");
    assert_eq!(table, {
        let mut t = ToleranceTable::release().expect("the embedded table loads");
        t.definitional.clear();
        t
    });
}

#[test]
fn every_required_entry_is_required() {
    let mut paths = Vec::new();
    for (_, class) in CLASS_KEYS {
        for scope in Scope::ALL {
            paths.push(format!("end_to_end.{class}.{}", scope.key()));
        }
        for c in Category::ALL {
            paths.push(format!("design_effect.{class}.{}", c.key()));
        }
        paths.push(format!("canary.{class}"));
        for kind in LayerKind::ALL {
            for s in kind.statistics() {
                paths.push(format!("layer_local.{class}.{}.{}", kind.key(), s.key()));
            }
        }
    }
    assert_eq!(paths.len(), 5 * (7 + 6 + 1 + 14));
    for path in paths {
        let err = parse_err(&render(1, &[], &[&path], ""));
        assert_eq!(
            err,
            TableError::Missing {
                entry: path.clone()
            },
            "{path}"
        );
    }
}

#[test]
fn a_missing_section_is_a_parse_error() {
    let text = pending_table().replace("[canary]", "[canary_typo]");
    assert!(matches!(parse_err(&text), TableError::Parse { .. }));
}

#[test]
fn unknown_keys_are_rejected() {
    let cases = [
        (
            pending_table().replace("[end_to_end.bits4]", "[end_to_end.bits3]"),
            "end_to_end.bits3",
        ),
        (
            pending_table().replace("\nmulti = ", "\nmultilingual = "),
            "end_to_end.bits4.multilingual",
        ),
        (
            pending_table().replace("[layer_local.int8.moe_ffn]", "[layer_local.int8.moe]"),
            "layer_local.int8.moe",
        ),
    ];
    for (text, entry) in cases {
        assert_eq!(
            parse_err(&text),
            TableError::Unknown {
                entry: entry.to_owned()
            }
        );
    }
}

#[test]
fn a_statistic_without_a_ceiling_for_its_kind_is_rejected() {
    let text = pending_table().replace(
        "[layer_local.fp8.dense_ffn]\n",
        "[layer_local.fp8.dense_ffn]\nkl_k = { status = \"pending\" }\n",
    );
    assert_eq!(
        parse_err(&text),
        TableError::Unknown {
            entry: "layer_local.fp8.dense_ffn.kl_k".to_owned()
        }
    );
}

#[test]
fn unknown_fields_are_rejected() {
    let text = render(
        1,
        &[(
            "canary.int8",
            r#"{ status = "final", value = 0.5, note = "x" }"#,
        )],
        &[],
        "",
    );
    assert!(matches!(parse_err(&text), TableError::Parse { .. }));
    let text = format!("bogus = 1\n{}", pending_table());
    assert!(matches!(parse_err(&text), TableError::Parse { .. }));
}

#[test]
fn schema_and_version_are_checked() {
    let text = pending_table().replace("schema = 1", "schema = 2");
    assert_eq!(parse_err(&text), TableError::Schema { found: 2 });
    let text = render(0, &[], &[], "");
    assert_eq!(invalid_entry(&parse_err(&text)), "version");
}

// ---------------------------------------------------------------------------------------------
// Status rules
// ---------------------------------------------------------------------------------------------

#[test]
fn a_pending_entry_carries_no_value() {
    let text = render(
        1,
        &[("canary.fp8", r#"{ status = "pending", value = 0.5 }"#)],
        &[],
        "",
    );
    assert_eq!(invalid_entry(&parse_err(&text)), "canary.fp8");
}

#[test]
fn a_published_entry_carries_its_value() {
    let text = render(1, &[("canary.fp8", r#"{ status = "final" }"#)], &[], "");
    assert!(matches!(parse_err(&text), TableError::Missing { .. }));
    let text = render(
        1,
        &[(
            "end_to_end.int8.pooled",
            r#"{ status = "final", kl_mean = 0.01, kl_p99 = 0.02 }"#,
        )],
        &[],
        "",
    );
    assert_eq!(invalid_entry(&parse_err(&text)), "end_to_end.int8.pooled");
}

#[test]
fn a_class_shares_one_end_to_end_status() {
    let mut set = publish_class("int8", "final");
    set[3].1 = PENDING.to_owned();
    let set: Vec<(&str, &str)> = set.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let err = parse_err(&render(1, &set, &[], ""));
    assert_eq!(invalid_entry(&err), "end_to_end.int8");
    let mut set = publish_class("int8", "final");
    set[0].1 = set[0].1.replace("final", "provisional");
    let set: Vec<(&str, &str)> = set.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let err = parse_err(&render(1, &set, &[], ""));
    assert_eq!(invalid_entry(&err), "end_to_end.int8");
}

#[test]
fn only_ceilings_are_provisional() {
    let text = render(
        1,
        &[(
            "design_effect.bits4.code",
            r#"{ status = "provisional", value = 1.2 }"#,
        )],
        &[],
        "",
    );
    assert_eq!(invalid_entry(&parse_err(&text)), "design_effect.bits4.code");
    for path in ["canary.bits4", "layer_local.bits4.moe_ffn.rel_l2_p99"] {
        let text = render(
            1,
            &[(path, r#"{ status = "provisional", value = 0.13 }"#)],
            &[],
            "",
        );
        let table = ToleranceTable::parse(&text).expect("a provisional ceiling loads");
        let entry = if path.starts_with("canary") {
            table.canary(DTypeClass::Bits4)
        } else {
            table
                .layer_local(
                    LayerKind::MoeFfn,
                    DTypeClass::Bits4,
                    LayerStatistic::RelL2P99,
                )
                .expect("the statistic exists for the kind")
        };
        assert_eq!(entry, Entry::Provisional(0.13));
        assert!(entry.is_provisional());
        assert_eq!(entry.value(), Some(0.13));
    }
}

// ---------------------------------------------------------------------------------------------
// Value rules
// ---------------------------------------------------------------------------------------------

#[test]
fn ceilings_are_rounded_to_two_significant_digits() {
    for v in [0.0, 0.0042, 0.042, 1.0, 1.5, 12.0, 120.0, 9.9e-7, 1e-30] {
        assert!(two_significant_digits(v), "{v}");
    }
    for v in [0.00423, 1.25, 123.0, 0.1 + 0.2] {
        assert!(!two_significant_digits(v), "{v}");
    }
    let text = render(
        1,
        &[(
            "layer_local.sub4.head_window.kl_k",
            r#"{ status = "final", value = 0.0123 }"#,
        )],
        &[],
        "",
    );
    assert_eq!(
        invalid_entry(&parse_err(&text)),
        "layer_local.sub4.head_window.kl_k"
    );
    let mut set = publish_class("fp8", "final");
    set[2].1 = set[2].1.replace("0.034", "0.0345");
    let set: Vec<(&str, &str)> = set.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    assert_eq!(
        invalid_entry(&parse_err(&render(1, &set, &[], ""))),
        "end_to_end.fp8.code.kl_p99"
    );
}

#[test]
fn the_canary_ceiling_is_not_rounded() {
    let text = render(
        1,
        &[("canary.int8", r#"{ status = "final", value = 0.01234 }"#)],
        &[],
        "",
    );
    let table = ToleranceTable::parse(&text).expect("the canary ceiling loads");
    assert_eq!(table.canary(DTypeClass::Int8), Entry::Final(0.01234));
}

#[test]
fn ranges_are_checked() {
    let cases = [
        ("canary.int8", r#"{ status = "final", value = -0.1 }"#),
        ("canary.int8", r#"{ status = "final", value = nan }"#),
        ("canary.int8", r#"{ status = "final", value = inf }"#),
        (
            "design_effect.int8.long",
            r#"{ status = "final", value = 0.9 }"#,
        ),
        (
            "layer_local.int8.head_window.top1",
            r#"{ status = "final", value = 1.1 }"#,
        ),
    ];
    for (path, value) in cases {
        let err = parse_err(&render(1, &[(path, value)], &[], ""));
        assert_eq!(invalid_entry(&err), path, "{value}");
    }
    let mut set = publish_class("sub4", "final");
    set[1].1 = set[1].1.replace("top1 = 0.0062", "top1 = 1.5");
    let set: Vec<(&str, &str)> = set.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    assert_eq!(
        invalid_entry(&parse_err(&render(1, &set, &[], ""))),
        "end_to_end.sub4.chat.top1"
    );
}

#[test]
fn a_design_effect_of_one_loads() {
    let text = render(
        1,
        &[(
            "design_effect.int8.long",
            r#"{ status = "final", value = 1 }"#,
        )],
        &[],
        "",
    );
    let table = ToleranceTable::parse(&text).expect("deff = 1 loads");
    assert_eq!(
        table.design_effect(DTypeClass::Int8, Category::Long),
        Entry::Final(1.0)
    );
}

// ---------------------------------------------------------------------------------------------
// Op-level entries
// ---------------------------------------------------------------------------------------------

fn op_entry(fields: &str) -> String {
    format!("[[op]]\nop = \"mul_mat\"\nclass = \"bits4\"\n{fields}\n")
}

#[test]
fn op_entries_load() {
    let extra = op_entry(
        "status = \"final\"\neps_op = 0.003\nceiling_op = 0.008\nceiling_op_max = 0.05\ntol_op = 0.004\ntol_ref = 0.004",
    ) + &op_entry("status = \"pending\"").replace("bits4", "int8");
    let table = ToleranceTable::parse(&render(1, &[], &[], &extra)).expect("the ops load");
    let t = OpTolerance {
        eps_op: 0.003,
        ceiling_op: 0.008,
        ceiling_op_max: 0.05,
        tol_op: 0.004,
        tol_ref: 0.004,
    };
    assert_eq!(
        table.op("mul_mat", DTypeClass::Bits4),
        Some(Entry::Final(t))
    );
    assert_eq!(table.op("mul_mat", DTypeClass::Int8), Some(Entry::Pending));
    assert_eq!(table.op("mul_mat", DTypeClass::Fp8), None);
    assert_eq!(table.ops().count(), 2);
}

#[test]
fn op_rules_are_checked() {
    let cases = [
        // tol_ref differs from tol_op.
        "eps_op = 0.003\nceiling_op = 0.008\nceiling_op_max = 0.05\ntol_op = 0.004\ntol_ref = 0.005",
        // tol_op below eps_op.
        "eps_op = 0.003\nceiling_op = 0.008\nceiling_op_max = 0.05\ntol_op = 0.002\ntol_ref = 0.002",
        // ceiling_op below 2 · eps_op.
        "eps_op = 0.003\nceiling_op = 0.005\nceiling_op_max = 0.05\ntol_op = 0.004\ntol_ref = 0.004",
        // ceiling_op not rounded to two significant digits.
        "eps_op = 0.003\nceiling_op = 0.00812\nceiling_op_max = 0.05\ntol_op = 0.004\ntol_ref = 0.004",
        // ceiling_op_max not rounded to two significant digits.
        "eps_op = 0.003\nceiling_op = 0.008\nceiling_op_max = 0.0512\ntol_op = 0.004\ntol_ref = 0.004",
        // Only some fields.
        "eps_op = 0.003\ntol_op = 0.004",
    ];
    for fields in cases {
        let extra = op_entry(&format!("status = \"final\"\n{fields}"));
        let err = parse_err(&render(1, &[], &[], &extra));
        assert!(
            invalid_entry(&err).starts_with("op[0] (mul_mat, bits4)")
                || invalid_entry(&err) == "op[0]",
            "{fields}: {err:?}"
        );
    }
    let extra = op_entry(
        "status = \"provisional\"\neps_op = 0.003\nceiling_op = 0.008\nceiling_op_max = 0.05\ntol_op = 0.004\ntol_ref = 0.004",
    );
    assert!(matches!(
        parse_err(&render(1, &[], &[], &extra)),
        TableError::Invalid { .. }
    ));
    let extra = op_entry("status = \"pending\"").repeat(2);
    assert!(matches!(
        parse_err(&render(1, &[], &[], &extra)),
        TableError::Invalid { .. }
    ));
    let extra = op_entry("status = \"pending\"").replace("bits4", "q4");
    assert!(matches!(
        parse_err(&render(1, &[], &[], &extra)),
        TableError::Unknown { .. }
    ));
    let extra = op_entry("status = \"pending\"").replace("mul_mat", "mul mat");
    assert!(matches!(
        parse_err(&render(1, &[], &[], &extra)),
        TableError::Invalid { .. }
    ));
}

// ---------------------------------------------------------------------------------------------
// Definitional entries
// ---------------------------------------------------------------------------------------------

fn definitional_entry(family: &str, source: &str, kl_mean: &str) -> String {
    format!(
        "[definitional.{family}]\nstatus = \"final\"\ns_pub_source = \"{source}\"\n\
         kl_mean = {kl_mean}\n\
         kl_p99 = {{ s_pub = 0.002, s_ref = 0.003, tau_def = 0.006 }}\n\
         top1 = {{ s_pub = 0.001, s_ref = 0.001, tau_def = 0.002 }}\n\
         norm_rel = {{ s_pub = 1e-6, s_ref = 3e-7, tau_def = 2e-6 }}\n"
    )
}

#[test]
fn definitional_entries_load() {
    let extra = definitional_entry(
        "llama",
        "publisher",
        "{ s_pub = 0.1, s_ref = 0.05, tau_def = 0.2 }",
    ) + "[definitional.qwen_moe]\nstatus = \"pending\"\n";
    let table = ToleranceTable::parse(&render(1, &[], &[], &extra)).expect("the families load");
    let d = table
        .definitional("llama")
        .and_then(|e| e.value())
        .expect("llama is published");
    assert_eq!(d.s_pub_source, SPubSource::Publisher);
    assert_eq!(
        d.kl_mean,
        DefinitionalMetric {
            s_pub: 0.1,
            s_ref: 0.05,
            tau_def: 0.2
        }
    );
    assert_eq!(d.kl_p99.tau_def, 0.006);
    assert_eq!(table.definitional("qwen_moe"), Some(Entry::Pending));
    assert_eq!(table.definitional("gemma"), None);
}

#[test]
fn tau_def_is_twice_the_larger_spread() {
    let extra = definitional_entry(
        "llama",
        "publisher",
        "{ s_pub = 0.1, s_ref = 0.05, tau_def = 0.1 }",
    );
    assert_eq!(
        invalid_entry(&parse_err(&render(1, &[], &[], &extra))),
        "definitional.llama.kl_mean"
    );
}

#[test]
fn a_ref_source_uses_s_ref_for_both_terms() {
    let extra = definitional_entry(
        "llama",
        "ref",
        "{ s_pub = 0.05, s_ref = 0.05, tau_def = 0.1 }",
    );
    let err = parse_err(&render(1, &[], &[], &extra));
    // kl_mean is consistent; kl_p99 has s_pub ≠ s_ref.
    assert_eq!(invalid_entry(&err), "definitional.llama.kl_p99");
    let extra = extra
        .replace(
            "s_pub = 0.002, s_ref = 0.003",
            "s_pub = 0.003, s_ref = 0.003",
        )
        .replace(
            "s_pub = 1e-6, s_ref = 3e-7, tau_def = 2e-6",
            "s_pub = 3e-7, s_ref = 3e-7, tau_def = 6e-7",
        );
    let table =
        ToleranceTable::parse(&render(1, &[], &[], &extra)).expect("a ref-sourced entry loads");
    let d = table
        .definitional("llama")
        .and_then(|e| e.value())
        .expect("llama is published");
    assert_eq!(d.s_pub_source, SPubSource::Ref);
}

#[test]
fn definitional_rules_are_checked() {
    // A published entry without every metric.
    let extra = "[definitional.llama]\nstatus = \"final\"\ns_pub_source = \"publisher\"\n";
    assert!(matches!(
        parse_err(&render(1, &[], &[], extra)),
        TableError::Invalid { .. }
    ));
    // A pending entry with values.
    let extra = definitional_entry(
        "llama",
        "publisher",
        "{ s_pub = 0.1, s_ref = 0.05, tau_def = 0.2 }",
    )
    .replace("status = \"final\"", "status = \"pending\"");
    assert!(matches!(
        parse_err(&render(1, &[], &[], &extra)),
        TableError::Invalid { .. }
    ));
    // Provisional is for ceilings only.
    let extra = definitional_entry(
        "llama",
        "publisher",
        "{ s_pub = 0.1, s_ref = 0.05, tau_def = 0.2 }",
    )
    .replace("status = \"final\"", "status = \"provisional\"");
    assert!(matches!(
        parse_err(&render(1, &[], &[], &extra)),
        TableError::Invalid { .. }
    ));
    // A family name that is not a module name.
    let extra = "[definitional.Llama]\nstatus = \"pending\"\n";
    assert!(matches!(
        parse_err(&render(1, &[], &[], extra)),
        TableError::Invalid { .. }
    ));
    // top1 spreads are rates.
    let extra = definitional_entry(
        "llama",
        "publisher",
        "{ s_pub = 0.1, s_ref = 0.05, tau_def = 0.2 }",
    )
    .replace(
        "top1 = { s_pub = 0.001, s_ref = 0.001, tau_def = 0.002 }",
        "top1 = { s_pub = 2.0, s_ref = 0.001, tau_def = 4.0 }",
    );
    assert_eq!(
        invalid_entry(&parse_err(&render(1, &[], &[], &extra))),
        "definitional.llama.top1.s_pub"
    );
}
