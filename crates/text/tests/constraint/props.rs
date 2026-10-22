// SPDX-License-Identifier: MIT OR Apache-2.0

//! Criterion 1 of PROD-E3-T04, generated half: for schemas from a generator, masked sampling over
//! random logits always produces JSON that the validator accepts.
//!
//! The generator covers RFC-0050 §7.2's keywords: every `type`, `properties`, `required`,
//! `additionalProperties`, `items`, `prefixItems`, `minItems`/`maxItems`, `enum`, `const`,
//! `anyOf`, `oneOf` (with disjoint options, so it means the same as `anyOf`), `allOf`, `$ref` and
//! `$defs` (recursive too), `pattern`, every §7.2 `format`, `minLength`/`maxLength`, and
//! `minimum`/`maximum` on integers. It runs on proptest's default configuration, so
//! `PROPTEST_CASES` sets the case count. The sampler walks at random for 48 tokens and then
//! steers toward an end (`support::sample`).

use proptest::prelude::*;
use serde_json::{Map, Value, json};
use syntropy_text::constraint::{ConstraintEngine, LlgEngine, OutputConstraint};

use super::support::{sample, vocabulary};
use super::validator::validate;

const NAMES: &[&str] = &[
    "id", "name", "type", "value", "tags", "when", "count", "note", "é",
];
const WORDS: &[&str] = &[
    "a",
    "beta",
    "Gamma",
    "x y",
    "é",
    "日本",
    "quote\"d",
    "back\\slash",
];
const PATTERNS: &[&str] = &["^[a-z]{1,4}$", "^[A-Z][0-9]{2}$", "^(yes|no)$", "^x+$"];
const FORMATS: &[&str] = &["date-time", "date", "time", "uuid", "email", "ipv4", "ipv6"];

fn leaf() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(json!({ "type": "string" })),
        (0u64..3, 0u64..4).prop_map(|(min, extra)| json!({
            "type": "string", "minLength": min, "maxLength": min + extra
        })),
        proptest::sample::subsequence(WORDS, 1..4).prop_map(|words| json!({ "enum": words })),
        proptest::sample::select(PATTERNS).prop_map(|p| json!({ "type": "string", "pattern": p })),
        proptest::sample::select(FORMATS).prop_map(|f| json!({ "type": "string", "format": f })),
        Just(json!({ "type": "integer" })),
        (-50i64..50, 0i64..100).prop_map(|(min, span)| json!({
            "type": "integer", "minimum": min, "maximum": min + span
        })),
        Just(json!({ "type": "number" })),
        Just(json!({ "type": "boolean" })),
        Just(json!({ "type": "null" })),
        Just(json!({ "type": ["string", "null"] })),
        prop_oneof![Just(json!("k")), Just(json!(3)), Just(json!(true))]
            .prop_map(|c| json!({ "const": c })),
        Just(json!({ "enum": [1, "two", null, false] })),
    ]
}

fn schema() -> impl Strategy<Value = Value> {
    leaf().prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            // An object with some of the names, some required, closed or open.
            (
                proptest::sample::subsequence(NAMES, 0..4),
                proptest::collection::vec(inner.clone(), 4),
                any::<u8>(),
                prop_oneof![Just(None), Just(Some(json!(false))), Just(Some(json!({ "type": "integer" })))],
            )
                .prop_map(|(names, schemas, required_bits, additional)| {
                    let mut properties = Map::new();
                    let mut required = Vec::new();
                    for (index, (name, schema)) in names.iter().zip(schemas).enumerate() {
                        properties.insert((*name).to_owned(), schema);
                        if required_bits & (1 << index) != 0 {
                            required.push(json!(name));
                        }
                    }
                    let mut object = json!({ "type": "object", "properties": properties, "required": required });
                    if let Some(additional) = additional {
                        object["additionalProperties"] = additional;
                    }
                    object
                }),
            // An array, bounded or not.
            (inner.clone(), 0u64..3, prop::option::of(0u64..3)).prop_map(|(items, min, extra)| {
                let mut array = json!({ "type": "array", "items": items, "minItems": min });
                if let Some(extra) = extra {
                    array["maxItems"] = json!(min + extra);
                }
                array
            }),
            // A tuple, then more items of one schema.
            (inner.clone(), inner.clone(), inner.clone()).prop_map(|(a, b, rest)| json!({
                "type": "array", "prefixItems": [a, b], "items": rest, "minItems": 2, "maxItems": 4
            })),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| json!({ "anyOf": [a, b] })),
            inner.clone().prop_map(|a| json!({ "allOf": [a] })),
            // `oneOf` over options of different types, so exactly one matches.
            Just(json!({ "oneOf": [{ "type": "string" }, { "type": "integer" }, { "type": "null" }] })),
        ]
    })
}

/// A schema, or at the root a definition that the schema uses by reference: `$ref` is resolved
/// from the root, so `$defs` lives there.
fn root() -> impl Strategy<Value = Value> {
    prop_oneof![
        3 => schema(),
        1 => schema().prop_map(|d| json!({
            "$defs": { "d": d },
            "type": "object",
            "properties": { "first": { "$ref": "#/$defs/d" }, "second": { "$ref": "#/$defs/d" } },
            "required": ["first"],
            "additionalProperties": false
        })),
    ]
}

/// A recursive schema: a tree whose nodes hold a value and child nodes.
fn tree() -> Value {
    json!({
        "$defs": {
            "node": {
                "type": "object",
                "properties": {
                    "value": { "type": "integer", "minimum": 0, "maximum": 9 },
                    "children": { "type": "array", "items": { "$ref": "#/$defs/node" }, "maxItems": 2 }
                },
                "required": ["value"],
                "additionalProperties": false
            }
        },
        "$ref": "#/$defs/node"
    })
}

fn check(engine: &LlgEngine, schema: &Value, seed: u64) -> Result<(), TestCaseError> {
    let constraint = OutputConstraint::JsonSchema {
        schema: schema.clone(),
        strict: true,
    };
    let output = engine
        .compile(&constraint)
        .map_err(|e| TestCaseError::fail(format!("compile: {e}\n{schema}")))?;
    prop_assert!(output.warnings.is_empty(), "{:?}", output.warnings);
    let sampled = sample(engine, &output.compiled, seed, 4000)
        .map_err(|e| TestCaseError::fail(format!("{e}\n{schema}")))?;
    let text = String::from_utf8(sampled.bytes)
        .map_err(|e| TestCaseError::fail(format!("not UTF-8: {e}\n{schema}")))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|e| TestCaseError::fail(format!("not JSON: {e}: {text}\n{schema}")))?;
    validate(schema, &value).map_err(|e| TestCaseError::fail(format!("{e}: {text}\n{schema}")))?;
    Ok(())
}

proptest! {
    #[test]
    fn masked_sampling_from_a_generated_schema_always_validates(schema in root(), seed in any::<u64>()) {
        let engine = LlgEngine::new(vocabulary()).expect("an engine");
        check(&engine, &schema, seed)?;
    }

    #[test]
    fn masked_sampling_from_a_recursive_schema_always_validates(seed in any::<u64>()) {
        let engine = LlgEngine::new(vocabulary()).expect("an engine");
        check(&engine, &tree(), seed)?;
    }
}
