// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serde properties of the parity tap types (RFC-0060 §5.5, amendment A4).
//!
//! Every type round-trips through a serde format unchanged, every record decodes when it carries
//! fields this build does not know, a spec built by a driver decodes identically at a tail, and a
//! tap's non-finite values survive in JSON as RFC-0050 §13.6 writes them. JSON stands in for any
//! self-describing format.

use std::fmt::Debug;

use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use syntropy_core::{
    ForceMode, LogitTap, ParityTapSpec, TapMode, TapRow, TapValues, TeacherStreamId,
};

/// Finite floats only: `PartialEq` cannot compare NaN, so non-finite values have their own tests.
fn finite_f32() -> impl Strategy<Value = f32> {
    prop::num::f32::NORMAL | prop::num::f32::SUBNORMAL | prop::num::f32::ZERO
}

fn finite_f64() -> impl Strategy<Value = f64> {
    prop::num::f64::NORMAL | prop::num::f64::SUBNORMAL | prop::num::f64::ZERO
}

fn force_mode() -> impl Strategy<Value = ForceMode> {
    prop_oneof![
        Just(ForceMode::Prefill),
        any::<u32>().prop_map(|prefill_len| ForceMode::Decode { prefill_len }),
        (any::<u32>(), any::<u16>(), 0.0f32..=1.0, any::<u64>()).prop_map(
            |(prefill_len, gamma, accept, seed)| ForceMode::Verify {
                prefill_len,
                gamma,
                accept,
                seed,
            }
        ),
    ]
}

fn tap_mode() -> impl Strategy<Value = TapMode> {
    prop_oneof![
        Just(TapMode::FullLocal),
        (any::<u16>(), any::<u16>()).prop_map(|(k, tail)| TapMode::Gather { k, tail }),
    ]
}

fn spec() -> impl Strategy<Value = ParityTapSpec> {
    (any::<u64>(), force_mode(), tap_mode()).prop_map(|(stream, mode, tap)| ParityTapSpec {
        stream: TeacherStreamId(stream),
        mode,
        tap,
    })
}

fn tap_values() -> impl Strategy<Value = TapValues> {
    prop_oneof![
        (
            prop::collection::vec(finite_f32(), 0..8),
            prop::collection::vec(finite_f32(), 0..8)
        )
            .prop_map(|(top, tail)| TapValues::Gather { top, tail }),
        any::<u64>().prop_map(|shm_offset| TapValues::FullLocal { shm_offset }),
    ]
}

fn tap_row() -> impl Strategy<Value = TapRow> {
    (
        any::<u32>(),
        any::<u32>(),
        finite_f32(),
        finite_f64(),
        finite_f64(),
        tap_values(),
    )
        .prop_map(
            |(pos, argmax, margin, lse, lse_complement, values)| TapRow {
                pos,
                argmax,
                margin,
                lse,
                lse_complement,
                values,
            },
        )
}

fn logit_tap() -> impl Strategy<Value = LogitTap> {
    (
        any::<u16>(),
        any::<u16>(),
        prop::collection::vec(tap_row(), 0..4),
    )
        .prop_map(|(k, tail, rows)| LogitTap { k, tail, rows })
}

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let json = serde_json::to_string(value).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(&back, value, "round trip through {json}");
}

/// Encodes `value`, adds an unknown field to the record, and decodes.
fn decodes_with_future_fields<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let mut json = serde_json::to_value(value).expect("serialize");
    json.as_object_mut().expect("a record").insert(
        "field_from_a_future_build".to_owned(),
        serde_json::json!({ "nested": [1, 2, 3], "flag": true }),
    );
    let back: T = serde_json::from_value(json.clone())
        .unwrap_or_else(|e| panic!("decode with unknown fields failed: {e}\n{json}"));
    assert_eq!(&back, value);
}

proptest! {
    #[test]
    fn rt_teacher_stream_id(value in any::<u64>().prop_map(TeacherStreamId)) {
        round_trip(&value);
    }

    #[test]
    fn rt_force_mode(value in force_mode()) {
        round_trip(&value);
    }

    #[test]
    fn rt_tap_mode(value in tap_mode()) {
        round_trip(&value);
    }

    #[test]
    fn rt_parity_tap_spec(value in spec()) {
        round_trip(&value);
        decodes_with_future_fields(&value);
    }

    #[test]
    fn rt_tap_values(value in tap_values()) {
        round_trip(&value);
    }

    #[test]
    fn rt_tap_row(value in tap_row()) {
        round_trip(&value);
        decodes_with_future_fields(&value);
    }

    #[test]
    fn rt_logit_tap(value in logit_tap()) {
        round_trip(&value);
        decodes_with_future_fields(&value);
    }

    /// Finite tap values come back bit for bit, subnormals and signed zeros included.
    #[test]
    fn finite_tap_floats_keep_their_bits(margin in finite_f32(), lse in finite_f64(), top in finite_f32()) {
        let row = TapRow {
            pos: 0,
            argmax: 0,
            margin,
            lse,
            lse_complement: lse,
            values: TapValues::Gather { top: vec![top], tail: vec![] },
        };
        let back: TapRow = serde_json::from_str(&serde_json::to_string(&row).unwrap()).unwrap();
        prop_assert_eq!(back.margin.to_bits(), margin.to_bits());
        prop_assert_eq!(back.lse.to_bits(), lse.to_bits());
        let TapValues::Gather { top: back_top, .. } = back.values else {
            panic!("expected Gather");
        };
        prop_assert_eq!(back_top[0].to_bits(), top.to_bits());
    }
}

/// The driver encodes a spec for each force mode and each tap mode; the tail decodes the same
/// bytes and gets the same spec, then acts on it. JSON stands in for the transport.
#[test]
fn a_spec_built_by_a_driver_decodes_identically_at_a_tail() {
    fn driver(stream: u64, mode: ForceMode, tap: TapMode) -> Vec<u8> {
        let spec = ParityTapSpec {
            stream: TeacherStreamId(stream),
            mode,
            tap,
        };
        serde_json::to_vec(&spec).expect("driver encodes")
    }
    /// What a stub tail derives from the spec: the positions it runs as prefill, and how many ids
    /// per position it expects from the driver.
    fn tail(bytes: &[u8]) -> (ParityTapSpec, Option<u32>, u32) {
        let spec: ParityTapSpec = serde_json::from_slice(bytes).expect("tail decodes");
        let prefill = match spec.mode {
            ForceMode::Decode { prefill_len } | ForceMode::Verify { prefill_len, .. } => {
                Some(prefill_len)
            }
            _ => None, // Prefill: every position
        };
        let ids = match spec.tap {
            TapMode::Gather { k, tail } => u32::from(k) + u32::from(tail),
            _ => 0, // FullLocal: the driver sends no ids
        };
        (spec, prefill, ids)
    }

    let modes = [
        ForceMode::Prefill,
        ForceMode::Decode { prefill_len: 512 },
        ForceMode::Verify {
            prefill_len: 512,
            gamma: 8,
            accept: 0.7,
            seed: 0x5EED,
        },
    ];
    let taps = [TapMode::FullLocal, TapMode::Gather { k: 256, tail: 64 }];
    for mode in modes {
        for tap in taps {
            let bytes = driver(42, mode, tap);
            let (spec, prefill, ids) = tail(&bytes);
            assert_eq!(
                spec,
                ParityTapSpec {
                    stream: TeacherStreamId(42),
                    mode,
                    tap,
                }
            );
            assert_eq!(
                serde_json::to_vec(&spec).unwrap(),
                bytes,
                "re-encodes identically"
            );
            assert_eq!(prefill, (mode != ForceMode::Prefill).then_some(512));
            assert_eq!(ids, if tap == TapMode::FullLocal { 0 } else { 320 });
        }
    }
}

/// RFC-0050 §13.6: a non-finite value is written as "nan", "inf" or "-inf" and never clamped
/// (RFC-0060 §5.4), so the driver sees it and treats it as a harness error.
#[test]
fn non_finite_tap_values_survive_as_named_strings() {
    let row = TapRow {
        pos: 7,
        argmax: 3,
        margin: f32::NAN,
        lse: f64::INFINITY,
        lse_complement: f64::NEG_INFINITY,
        values: TapValues::Gather {
            top: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.5],
            tail: vec![-0.25],
        },
    };
    let json = serde_json::to_string(&row).expect("serialize");
    assert_eq!(
        json,
        r#"{"pos":7,"argmax":3,"margin":"nan","lse":"inf","lse_complement":"-inf","values":{"Gather":{"top":["nan","inf","-inf",1.5],"tail":[-0.25]}}}"#
    );

    let back: TapRow = serde_json::from_str(&json).expect("deserialize");
    assert!(back.margin.is_nan());
    assert_eq!(back.lse, f64::INFINITY);
    assert_eq!(back.lse_complement, f64::NEG_INFINITY);
    let TapValues::Gather { top, tail } = &back.values else {
        panic!("expected Gather");
    };
    assert!(top[0].is_nan());
    assert_eq!(top[1..], [f32::INFINITY, f32::NEG_INFINITY, 1.5]);
    assert_eq!(tail, &[-0.25]);
    assert_eq!(
        serde_json::to_string(&back).unwrap(),
        json,
        "re-encodes identically"
    );
}

/// Only the three names are accepted in place of a number.
#[test]
fn other_strings_are_not_floats() {
    for bad in [r#""NaN""#, r#""infinity""#, r#""1.5""#, r#""""#] {
        let json = format!(
            r#"{{"pos":0,"argmax":0,"margin":{bad},"lse":0.0,"lse_complement":0.0,"values":{{"FullLocal":{{"shm_offset":0}}}}}}"#
        );
        assert!(
            serde_json::from_str::<TapRow>(&json).is_err(),
            "{bad} decoded as a float"
        );
    }
}

/// An integer where a float is expected decodes, as a JSON number would from another writer.
#[test]
fn integer_numbers_decode_as_floats() {
    let json = r#"{"pos":0,"argmax":0,"margin":2,"lse":-3,"lse_complement":0,"values":{"Gather":{"top":[1],"tail":[]}}}"#;
    let row: TapRow = serde_json::from_str(json).expect("deserialize");
    assert_eq!((row.margin, row.lse, row.lse_complement), (2.0, -3.0, 0.0));
    assert_eq!(
        row.values,
        TapValues::Gather {
            top: vec![1.0],
            tail: vec![]
        }
    );
}
