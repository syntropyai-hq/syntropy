// SPDX-License-Identifier: MIT OR Apache-2.0

//! P-bench-1 and the golden record.

use proptest::prelude::*;
use serde_json::Value;
use syntropy_bench::schema::{BenchResult, SchemaError};

use crate::checker;
use crate::support::{GOLDEN_FILE, SCHEMA_FILE, bench_result, sample_result};

fn schema() -> Value {
    serde_json::from_str(&std::fs::read_to_string(SCHEMA_FILE).expect("read schema file"))
        .expect("schema is JSON")
}

proptest! {
    /// P-bench-1: a result reads back as itself, and its JSON passes the schema file.
    #[test]
    fn p_bench_1_result_round_trips_through_json(result in bench_result()) {
        let text = result.to_json().expect("write");
        let back = BenchResult::from_json(&text).expect("read back");
        prop_assert_eq!(&back, &result);
        let value: Value = serde_json::from_str(&text).expect("JSON");
        if let Err(e) = checker::check(&schema(), &value) {
            return Err(TestCaseError::fail(format!("schema file rejects a written result: {e}")));
        }
    }
}

/// The wire format is pinned: field names, enum spellings and digest encoding.
#[test]
fn golden_result_matches_committed_json() {
    let golden_text = std::fs::read_to_string(GOLDEN_FILE).expect("read golden file");
    let golden: Value = serde_json::from_str(&golden_text).expect("golden is JSON");
    let written: Value =
        serde_json::from_str(&sample_result().to_json().expect("write")).expect("JSON");
    assert_eq!(
        written,
        golden,
        "the JSON form changed; if that is intended, bump SCHEMA_VERSION and add a new golden file\n{}",
        serde_json::to_string_pretty(&written).expect("pretty")
    );
    assert_eq!(
        BenchResult::from_json(&golden_text).expect("golden reads"),
        sample_result()
    );
    checker::check(&schema(), &golden).expect("golden passes the schema file");
}

#[test]
fn non_finite_floats_are_refused_on_write() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut result = sample_result();
        result.repetitions[0].tpot_s = bad;
        assert!(
            matches!(result.to_json(), Err(SchemaError::NotRepresentable)),
            "tpot_s = {bad}"
        );

        // In an optional field, JSON would turn the value into `null` and so into `None`.
        let mut result = sample_result();
        result.repetitions[0].energy_j_per_token = Some(bad);
        assert!(
            matches!(result.to_json(), Err(SchemaError::NotRepresentable)),
            "energy = {bad}"
        );

        let mut result = sample_result();
        result.repetitions[0].itl_s.push(bad as f32);
        assert!(
            matches!(result.to_json(), Err(SchemaError::NotRepresentable)),
            "itl_s += {bad}"
        );
    }
}

#[test]
fn a_result_of_another_schema_version_is_not_written() {
    let mut result = sample_result();
    result.schema = 2;
    assert!(matches!(result.to_json(), Err(SchemaError::Version(2))));
}
