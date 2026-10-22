// SPDX-License-Identifier: MIT OR Apache-2.0

//! The compile cache, keyed by BLAKE3(constraint, tokenizer) (RFC-0050 §7.3).

use std::sync::Arc;

use serde_json::json;
use syntropy_text::constraint::{
    CompileCache, LlgEngine, OutputConstraint, VocabToken, Vocabulary, key,
};

use super::support::vocabulary;

fn schema(properties: serde_json::Value) -> OutputConstraint {
    OutputConstraint::JsonSchema {
        schema: json!({ "type": "object", "properties": properties }),
        strict: true,
    }
}

#[test]
fn the_key_follows_the_constraint_its_property_order_and_the_tokenizer() {
    let vocab = vocabulary();
    let ab = schema(json!({ "a": { "type": "string" }, "b": { "type": "integer" } }));
    let ba = schema(json!({ "b": { "type": "integer" }, "a": { "type": "string" } }));
    assert_eq!(key(&ab, &vocab), key(&ab.clone(), &vocab));
    assert_ne!(
        key(&ab, &vocab),
        key(&ba, &vocab),
        "property order is part of the output"
    );
    let lenient = OutputConstraint::JsonSchema {
        schema: json!({ "type": "object", "properties": { "a": { "type": "string" }, "b": { "type": "integer" } } }),
        strict: false,
    };
    assert_ne!(key(&ab, &vocab), key(&lenient, &vocab));
    assert_ne!(
        key(&OutputConstraint::Regex("a".into()), &vocab),
        key(&OutputConstraint::Choice(vec!["a".into()]), &vocab)
    );
    let mut tokens = vocab.tokens().to_vec();
    tokens.push(VocabToken::Text(b"extra".to_vec()));
    let other = Vocabulary::new(tokens, vocab.eos()).unwrap();
    assert_ne!(key(&ab, &vocab), key(&ab, &other));
}

#[test]
fn the_cache_compiles_once_and_drops_the_least_recently_used() {
    let engine = LlgEngine::new(vocabulary()).unwrap();
    let mut cache = CompileCache::new(2);
    let a = OutputConstraint::Regex("a+".into());
    let b = OutputConstraint::Regex("b+".into());
    let c = OutputConstraint::Regex("c+".into());

    let first = cache.get_or_compile(&engine, &a).unwrap();
    let again = cache.get_or_compile(&engine, &a).unwrap();
    assert!(Arc::ptr_eq(&first.compiled, &again.compiled));
    assert_eq!(cache.hits_and_misses(), (1, 1));

    cache.get_or_compile(&engine, &b).unwrap();
    cache.get_or_compile(&engine, &a).unwrap(); // `a` is now the most recently used
    cache.get_or_compile(&engine, &c).unwrap(); // full: drops `b`
    assert_eq!(cache.len(), 2);
    let a_still = cache.get_or_compile(&engine, &a).unwrap();
    assert!(Arc::ptr_eq(&first.compiled, &a_still.compiled));
    let misses = cache.hits_and_misses().1;
    cache.get_or_compile(&engine, &b).unwrap();
    assert_eq!(
        cache.hits_and_misses().1,
        misses + 1,
        "`b` was dropped and compiled again"
    );
}

#[test]
fn a_failed_compile_is_not_cached() {
    let engine = LlgEngine::new(vocabulary()).unwrap();
    let mut cache = CompileCache::new(4);
    let bad = OutputConstraint::Regex("[".into());
    assert!(cache.get_or_compile(&engine, &bad).is_err());
    assert!(cache.get_or_compile(&engine, &bad).is_err());
    assert!(cache.is_empty());
    assert_eq!(cache.hits_and_misses(), (0, 2));
}

#[test]
fn warnings_come_back_from_the_cache_too() {
    let engine = LlgEngine::new(vocabulary()).unwrap();
    let mut cache = CompileCache::new(4);
    let lenient = OutputConstraint::JsonSchema {
        schema: json!({ "type": "array", "uniqueItems": true }),
        strict: false,
    };
    let first = cache.get_or_compile(&engine, &lenient).unwrap();
    let cached = cache.get_or_compile(&engine, &lenient).unwrap();
    assert_eq!(first.warnings.len(), 1);
    assert_eq!(first.warnings, cached.warnings);
}
