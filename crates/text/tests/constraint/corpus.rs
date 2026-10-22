// SPDX-License-Identifier: MIT OR Apache-2.0

//! Criterion 1 of PROD-E3-T04, corpus half: for each tool schema in `tool-schemas/`, masked
//! sampling over random logits always produces JSON that the validator accepts.
//!
//! The schemas were written for this test in the shapes tool definitions take: titles and
//! descriptions, closed objects with required keys, enums, nested objects and arrays of objects,
//! `anyOf` with `null`, `oneOf`, formats, patterns, integer bounds, tuples, maps of
//! `additionalProperties`, and recursion through `$defs`. Each compiles under `strict`.

use std::path::PathBuf;

use serde_json::Value;
use syntropy_text::constraint::{ConstraintEngine, LlgEngine, OutputConstraint};

use super::support::{sample, vocabulary};
use super::validator::validate;

/// Seeds per schema.
const SEEDS: u64 = 24;

fn corpus() -> Vec<(String, Value)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/constraint/tool-schemas");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("the corpus directory")
        .map(|entry| entry.expect("an entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|path| {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&path).expect("a schema file");
            (name, serde_json::from_str(&text).expect("a JSON schema"))
        })
        .collect()
}

#[test]
fn masked_sampling_from_every_tool_schema_always_validates() {
    let corpus = corpus();
    assert!(
        corpus.len() >= 12,
        "the corpus holds {} schemas",
        corpus.len()
    );
    let engine = LlgEngine::new(vocabulary()).expect("an engine");
    for (name, schema) in &corpus {
        let constraint = OutputConstraint::JsonSchema {
            schema: schema.clone(),
            strict: true,
        };
        let output = engine
            .compile(&constraint)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(output.warnings.is_empty(), "{name}: {:?}", output.warnings);
        for seed in 0..SEEDS {
            let sampled = sample(&engine, &output.compiled, seed, 4000)
                .unwrap_or_else(|e| panic!("{name}, seed {seed}: {e}"));
            let text = String::from_utf8(sampled.bytes)
                .unwrap_or_else(|e| panic!("{name}, seed {seed}: not UTF-8: {e}"));
            let value: Value = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("{name}, seed {seed}: not JSON: {e}: {text}"));
            if let Err(reason) = validate(schema, &value) {
                panic!("{name}, seed {seed}: {reason}: {text}");
            }
        }
    }
}
