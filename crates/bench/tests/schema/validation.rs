// SPDX-License-Identifier: MIT OR Apache-2.0

//! What fails validation, checked against both `BenchResult::from_json` and the schema file.

use serde_json::{Value, json};
use syntropy_bench::schema::{
    ApproxKnob, ApproxParams, BenchResult, ExactnessPolicy, KeyField, SchemaError,
};

use crate::checker;
use crate::support::{SCHEMA_FILE, other_key, sample_result};

fn schema() -> Value {
    serde_json::from_str(&std::fs::read_to_string(SCHEMA_FILE).expect("read schema file"))
        .expect("schema is JSON")
}

/// `ApproxParams` with every knob at its default.
fn null_params() -> Value {
    serde_json::to_value(ApproxParams::default()).expect("serialise params")
}

/// The sample result with `other_key()`, which uses a throughput objective and approximate
/// knobs with parameters.
fn policy_json() -> Value {
    let mut result = sample_result();
    result.key = other_key();
    serde_json::from_str(&result.to_json().expect("write")).expect("JSON")
}

fn sample_json() -> Value {
    serde_json::from_str(&sample_result().to_json().expect("write")).expect("JSON")
}

/// Asserts that both readers reject `value`, and returns the Rust reader's error.
fn rejected(value: &Value, what: &str) -> SchemaError {
    assert!(
        checker::check(&schema(), value).is_err(),
        "schema file accepts {what}"
    );
    match BenchResult::from_json(&value.to_string()) {
        Ok(_) => panic!("from_json accepts {what}"),
        Err(e) => e,
    }
}

#[test]
fn the_sample_passes_both_readers() {
    let value = sample_json();
    checker::check(&schema(), &value).expect("schema file accepts the sample");
    BenchResult::from_json(&value.to_string()).expect("from_json accepts the sample");
}

/// The criterion: a result missing any key field fails validation.
#[test]
fn result_missing_a_key_field_fails_validation() {
    for field in KeyField::ALL {
        let mut value = sample_json();
        value["key"]
            .as_object_mut()
            .expect("key object")
            .remove(field.name())
            .expect("field present");
        let error = rejected(&value, &format!("a key without {field}"));
        let message = error.to_string();
        assert!(
            message.contains(&format!("missing field `{field}`")),
            "{field}: {message}"
        );
    }
}

/// Every top-level field is required, including those whose value may be `null`.
#[test]
fn result_missing_any_top_level_field_fails_validation() {
    let names: Vec<String> = sample_json()
        .as_object()
        .expect("object")
        .keys()
        .cloned()
        .collect();
    assert_eq!(names.len(), 14);
    for name in names {
        let mut value = sample_json();
        value.as_object_mut().expect("object").remove(&name);
        let message = rejected(&value, &format!("a result without {name}")).to_string();
        assert!(
            message.contains(&format!("missing field `{name}`")),
            "{name}: {message}"
        );
    }
}

/// A nullable field nested in a record must be written too.
#[test]
fn nested_nullable_fields_must_be_written() {
    for (path, field) in [
        ("/env", "microcode"),
        ("/repetitions/0", "aggregate_tps"),
        ("/bounds", "overlap"),
    ] {
        let mut value = sample_json();
        value
            .pointer_mut(path)
            .and_then(Value::as_object_mut)
            .expect(path)
            .remove(field)
            .expect(field);
        let message = rejected(&value, &format!("{path} without {field}")).to_string();
        assert!(
            message.contains(&format!("missing field `{field}`")),
            "{field}: {message}"
        );
    }
}

#[test]
fn unknown_fields_fail_validation() {
    for path in [
        "",
        "/key",
        "/key/build",
        "/summary/ttft_s",
        "/env",
        "/key/thermal/sustained",
    ] {
        let mut value = sample_json();
        value
            .pointer_mut(path)
            .and_then(Value::as_object_mut)
            .expect(path)
            .insert("extra".into(), json!(0));
        let message = rejected(&value, &format!("an unknown field at {path:?}")).to_string();
        assert!(
            message.contains("unknown field `extra`"),
            "{path}: {message}"
        );
    }
}

/// An externally tagged enum holds exactly one variant.
#[test]
fn an_enum_with_two_variants_fails_validation() {
    let mut value = sample_json();
    value["key"]["thermal"] = json!({"sustained": {"preload_s": 600}, "cold": null});
    rejected(&value, "a thermal protocol naming two variants");
}

#[test]
fn other_schema_versions_fail_validation() {
    let mut value = sample_json();
    value["schema"] = json!(2);
    assert!(matches!(
        rejected(&value, "schema 2"),
        SchemaError::Version(2)
    ));
}

#[test]
fn digests_must_be_lowercase_hex_of_the_right_length() {
    let good = "0123456789abcdef".repeat(4);
    let mut value = sample_json();
    value["key"]["rig"] = json!(good);
    checker::check(&schema(), &value).expect("schema file accepts lowercase hex");
    BenchResult::from_json(&value.to_string()).expect("from_json accepts lowercase hex");
    for bad in [
        good.to_uppercase(),
        good[..62].to_owned(),
        format!("{good}00"),
        "g".repeat(64),
    ] {
        let mut value = sample_json();
        value["key"]["rig"] = json!(bad);
        rejected(&value, &format!("rig {bad:?}"));
    }
    let mut value = sample_json();
    value["key"]["build"]["commit"] = json!("0".repeat(64));
    rejected(&value, "a 64-digit commit");
}

#[test]
fn a_repeated_knob_fails_validation() {
    let mut value = sample_json();
    value["key"]["fidelity"] =
        json!({"approximate": {"knobs": ["kv_lossy", "kv_lossy"], "params": null_params()}});
    let message = rejected(&value, "a repeated knob").to_string();
    assert!(message.contains("listed twice"), "{message}");

    value["key"]["fidelity"] =
        json!({"approximate": {"knobs": ["kv_lossy", "expert_skip"], "params": null_params()}});
    let result = BenchResult::from_json(&value.to_string()).expect("knobs in any order");
    checker::check(&schema(), &value).expect("schema file accepts knobs in any order");
    assert_eq!(
        result.key.fidelity,
        ExactnessPolicy::Approximate {
            knobs: [ApproxKnob::ExpertSkip, ApproxKnob::KvLossy].into(),
            params: Box::default(),
        }
    );
}

#[test]
fn out_of_range_integers_and_wrong_types_fail_validation() {
    let cases = [
        ("/key/scenario_version", json!(65536)),
        ("/key/prompt_tokens", json!(-1)),
        ("/key/gen_tokens", json!(1.5)),
        ("/key/expert_cache", json!("warm")),
        ("/trust", json!("maybe")),
        ("/repetitions/0/ttft_s", json!("0.25")),
        ("/repetitions/0/ttft_s", Value::Null),
    ];
    for (path, bad) in cases {
        let mut value = sample_json();
        *value.pointer_mut(path).expect(path) = bad.clone();
        rejected(&value, &format!("{path} = {bad}"));
    }
}

/// RFC-0054 §7 resolves an approximate policy with no knob to `Exact`, so the key never holds one.
#[test]
fn an_empty_knob_list_fails_validation() {
    let mut value = sample_json();
    value["key"]["fidelity"] = json!({"approximate": {"knobs": [], "params": null_params()}});
    let message = rejected(&value, "an empty knob list").to_string();
    assert!(message.contains("at least one knob"), "{message}");
}

/// The policy fields RFC-0054 §4.1 and §4.2 add must be written too, as `null` when unset.
#[test]
fn policy_fields_from_rfc_0054_must_be_written() {
    let value = policy_json();
    checker::check(&schema(), &value).expect("schema file accepts the policy sample");
    BenchResult::from_json(&value.to_string()).expect("from_json accepts the policy sample");
    for (path, field) in [
        ("/key/objective/throughput", "target_concurrency"),
        ("/key/objective/throughput", "ttft_ceiling"),
        ("/key/fidelity/approximate", "params"),
        ("/key/fidelity/approximate/params", "kv_lossy"),
        ("/key/fidelity/approximate/params", "spec_lenient"),
    ] {
        let mut value = policy_json();
        value
            .pointer_mut(path)
            .and_then(Value::as_object_mut)
            .expect(path)
            .remove(field)
            .expect(field);
        let message = rejected(&value, &format!("{path} without {field}")).to_string();
        assert!(
            message.contains(&format!("missing field `{field}`")),
            "{field}: {message}"
        );
    }
}

/// An `f32` field takes any magnitude that rounds to a finite `f32`, and nothing larger.
#[test]
fn f32_fields_reject_values_beyond_the_f32_range() {
    let paths = [
        "/repetitions/0/itl_s/0",
        "/repetitions/1/floor_attainment",
        "/summary/floor_attainment",
        "/key/objective/throughput/per_sequence_floor_tps",
    ];
    for path in paths {
        for good in [json!(f32::MAX), json!(-f32::MAX), json!(1e-50)] {
            let mut value = policy_json();
            *value.pointer_mut(path).expect(path) = good.clone();
            checker::check(&schema(), &value)
                .unwrap_or_else(|e| panic!("schema file rejects {path} = {good}: {e}"));
            BenchResult::from_json(&value.to_string())
                .unwrap_or_else(|e| panic!("from_json rejects {path} = {good}: {e}"));
        }
        for bad in [json!(1e39), json!(-1e39), json!(3.4028236e38)] {
            let mut value = policy_json();
            *value.pointer_mut(path).expect(path) = bad.clone();
            let message = rejected(&value, &format!("{path} = {bad}")).to_string();
            assert!(
                message.contains("out of range"),
                "{path} = {bad}: {message}"
            );
        }
    }
}
