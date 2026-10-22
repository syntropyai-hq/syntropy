// SPDX-License-Identifier: MIT OR Apache-2.0

//! Criterion 2 of PROD-E3-T04: the first compile of a schema of up to 50 properties takes at most
//! 50 ms at p99 [target].
//!
//! A timing measurement, so it is ignored by default and run in release:
//!
//! ```sh
//! cargo test --release -p syntropy-text --test compile_time -- --ignored --nocapture
//! ```
//!
//! It compiles 200 different 50-property schemas, so none comes from a cache, against a
//! vocabulary of 131,072 tokens: the 256 bytes and generated words, the size of a large-vocabulary
//! tokenizer. Building the engine for the vocabulary, the one-off cost per tokenizer, is printed
//! but not counted.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};
use syntropy_text::constraint::{
    ConstraintEngine, LlgEngine, OutputConstraint, TokenId, VocabToken, Vocabulary,
};

const VOCABULARY_SIZE: usize = 131_072;
const SCHEMAS: usize = 200;
const PROPERTIES: usize = 50;
const TARGET: Duration = Duration::from_millis(50);

/// The bytes, then distinct generated words of 2 to 8 characters, then the end token.
fn vocabulary() -> Arc<Vocabulary> {
    const CHARS: &[u8] =
        b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.,:;\"'{}[]()";
    let mut tokens: Vec<VocabToken> = (0..=255u8).map(|b| VocabToken::Text(vec![b])).collect();
    let mut seen = std::collections::BTreeSet::new();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    while tokens.len() < VOCABULARY_SIZE - 1 {
        let len = 2 + (next() % 7) as usize;
        let word: Vec<u8> = (0..len)
            .map(|_| CHARS[(next() % CHARS.len() as u64) as usize])
            .collect();
        if seen.insert(word.clone()) {
            tokens.push(VocabToken::Text(word));
        }
    }
    tokens.push(VocabToken::Special("<|end|>".to_owned()));
    let eos = (tokens.len() - 1) as TokenId;
    Arc::new(Vocabulary::new(tokens, eos).expect("a vocabulary"))
}

/// Schema `n`: 50 properties of mixed kinds, every name unique to `n`.
fn schema(n: usize) -> Value {
    let mut properties = Map::new();
    for k in 0..PROPERTIES {
        let property = match k % 10 {
            0 => json!({ "type": "string", "maxLength": 20 + k }),
            1 => json!({ "type": "integer", "minimum": 0, "maximum": 1000 + n }),
            2 => json!({ "enum": [format!("a{n}"), format!("b{k}"), "c"] }),
            3 => json!({ "type": "boolean" }),
            4 => json!({ "type": "array", "items": { "type": "string" }, "maxItems": 5 }),
            5 => json!({
                "type": "object",
                "properties": { "x": { "type": "number" }, "y": { "type": "string" } },
                "required": ["x"],
                "additionalProperties": false
            }),
            6 => json!({ "type": "string", "format": "date-time" }),
            7 => json!({ "anyOf": [{ "type": "string" }, { "type": "null" }] }),
            8 => json!({ "type": "string", "pattern": "^[a-z]{2,8}$" }),
            _ => json!({ "type": "number" }),
        };
        properties.insert(format!("field_{n}_{k}"), property);
    }
    let required: Vec<Value> = properties.keys().map(|k| json!(k)).collect();
    json!({ "type": "object", "properties": properties, "required": required, "additionalProperties": false })
}

#[test]
#[ignore = "a timing measurement: run in release with --ignored (see the module docs)"]
fn the_first_compile_of_50_properties_is_within_50_ms_at_p99() {
    let vocab = vocabulary();
    let started = Instant::now();
    let engine = LlgEngine::new(vocab).expect("an engine");
    println!(
        "engine for {VOCABULARY_SIZE} tokens: {:?} (not counted)",
        started.elapsed()
    );

    let mut times = Vec::with_capacity(SCHEMAS);
    for n in 0..SCHEMAS {
        let constraint = OutputConstraint::JsonSchema {
            schema: schema(n),
            strict: true,
        };
        let started = Instant::now();
        let output = engine.compile(&constraint).expect("the schema compiles");
        times.push(started.elapsed());
        assert!(output.warnings.is_empty());
    }
    times.sort();
    let at = |q: f64| times[((times.len() as f64 * q).ceil() as usize).clamp(1, times.len()) - 1];
    let (p50, p99, max) = (at(0.50), at(0.99), times[times.len() - 1]);
    println!(
        "first compile of a {PROPERTIES}-property schema, {SCHEMAS} schemas: p50 {p50:?}, \
         p99 {p99:?}, max {max:?}"
    );
    assert!(p99 <= TARGET, "p99 {p99:?} is over the {TARGET:?} target");
}
