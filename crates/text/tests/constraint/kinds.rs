// SPDX-License-Identifier: MIT OR Apache-2.0

//! The constraint kinds other than generated schemas, `strict` handling through the engine, and
//! the engine's contract at its edges.

use std::sync::Arc;

use regex::Regex;
use serde_json::{Value, json};
use syntropy_text::constraint::{
    ConstraintEngine, ConstraintError, ConstraintWarning, GrammarSyntax, LlgEngine,
    OutputConstraint, TokenId, TokenMask, VocabToken, Vocabulary,
};

use super::support::{Rng, sample_text, vocabulary};
use super::validator::validate;

fn engine() -> LlgEngine {
    LlgEngine::new(vocabulary()).expect("an engine")
}

fn invalid(constraint: OutputConstraint) -> String {
    match engine().compile(&constraint) {
        Err(ConstraintError::Invalid { reason }) => reason,
        Err(other) => panic!("{constraint:?}: {other:?}"),
        Ok(_) => panic!("{constraint:?} compiled"),
    }
}

#[test]
fn json_object_gives_an_object() {
    let engine = engine();
    for seed in 0..16 {
        let text = sample_text(&engine, &OutputConstraint::JsonObject, seed).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert!(value.is_object(), "{text}");
    }
}

#[test]
fn a_regex_constrains_the_whole_output() {
    let engine = engine();
    let pattern = "[a-c]{2,4}-[0-9]{2}(\\.[0-9])?";
    let whole = Regex::new(&format!("^(?:{pattern})$")).unwrap();
    for seed in 0..32 {
        let text = sample_text(&engine, &OutputConstraint::Regex(pattern.into()), seed).unwrap();
        assert!(whole.is_match(&text), "{text}");
    }
}

#[test]
fn a_choice_gives_exactly_one_of_its_strings() {
    let engine = engine();
    let choices = vec![
        "alpha".to_owned(),
        "beta".to_owned(),
        "a.b*(c)".to_owned(),
        "é".to_owned(),
    ];
    let constraint = OutputConstraint::Choice(choices.clone());
    let mut seen = std::collections::BTreeSet::new();
    for seed in 0..64 {
        let text = sample_text(&engine, &constraint, seed).unwrap();
        assert!(choices.contains(&text), "{text}");
        seen.insert(text);
    }
    assert!(seen.len() > 1, "only {seen:?} over 64 seeds");
}

#[test]
fn a_lark_grammar_constrains_the_output() {
    let engine = engine();
    let grammar = OutputConstraint::Grammar {
        syntax: GrammarSyntax::Lark,
        text: "start: answer \" \" number\nanswer: \"yes\" | \"no\"\nnumber: /[1-9][0-9]?/\n"
            .into(),
    };
    let whole = Regex::new("^(yes|no) [1-9][0-9]?$").unwrap();
    for seed in 0..32 {
        let text = sample_text(&engine, &grammar, seed).unwrap();
        assert!(whole.is_match(&text), "{text}");
    }
}

#[test]
fn malformed_constraints_are_constraint_invalid() {
    assert!(!invalid(OutputConstraint::Regex("[a-".into())).is_empty());
    assert!(
        !invalid(OutputConstraint::Grammar {
            syntax: GrammarSyntax::Lark,
            text: "start: undefined_rule\n".into(),
        })
        .is_empty()
    );
    assert!(invalid(OutputConstraint::Choice(Vec::new())).contains("at least one"));
    let schema = |schema| OutputConstraint::JsonSchema {
        schema,
        strict: true,
    };
    assert!(invalid(schema(json!(false))).contains("accepts no value"));
    assert!(!invalid(schema(json!({ "$ref": "#/$defs/missing" }))).is_empty());
    assert!(
        !invalid(schema(
            json!({ "type": "string", "minLength": 5, "maxLength": 2 })
        ))
        .is_empty()
    );
}

#[test]
fn strict_fails_unsupported_keywords_and_non_strict_warns() {
    let engine = engine();
    let schema = json!({
        "type": "object",
        "properties": {
            "tags": { "type": "array", "items": { "type": "string" }, "uniqueItems": true },
            "spec": { "$ref": "https://example.org/spec.json" }
        },
        "required": ["tags"],
        "additionalProperties": false
    });
    let strict = OutputConstraint::JsonSchema {
        schema: schema.clone(),
        strict: true,
    };
    match engine.compile(&strict) {
        Err(error @ ConstraintError::Unsupported { .. }) => {
            assert_eq!(error.code(), "constraint_unsupported");
            assert_eq!(
                error,
                ConstraintError::Unsupported {
                    keyword: "uniqueItems".into(),
                    at: "/properties/tags".into()
                }
            );
        }
        other => panic!("{:?}", other.map(|o| o.warnings)),
    }
    let lenient = OutputConstraint::JsonSchema {
        schema,
        strict: false,
    };
    let output = engine.compile(&lenient).unwrap();
    let warnings: Vec<String> = output.warnings.iter().map(ToString::to_string).collect();
    assert_eq!(
        warnings,
        [
            "unsupported schema keyword `uniqueItems` at /properties/tags was ignored",
            "unsupported schema keyword `$ref` at /properties/spec was ignored",
        ]
    );
    // The rest of the schema still constrains the output.
    for seed in 0..8 {
        let text = sample_text(&engine, &lenient, seed).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert!(value["tags"].is_array(), "{text}");
    }
}

#[test]
fn an_overlapping_one_of_is_refused_under_strict_and_labelled_without() {
    // `integer` and `number` overlap: an output such as `1` matches both, which `oneOf` forbids.
    // The `oneOf` under `a` is exclusive, so only the one under `n` is named.
    let engine = engine();
    let schema = json!({
        "type": "object",
        "properties": {
            "a": { "oneOf": [{ "type": "string" }, { "type": "null" }] },
            "n": { "oneOf": [{ "type": "integer" }, { "type": "number" }] }
        },
        "required": ["a", "n"],
        "additionalProperties": false
    });
    let strict = OutputConstraint::JsonSchema {
        schema: schema.clone(),
        strict: true,
    };
    match engine.compile(&strict) {
        Err(error @ ConstraintError::Unsupported { .. }) => {
            assert_eq!(error.code(), "constraint_unsupported");
            assert_eq!(
                error,
                ConstraintError::Unsupported {
                    keyword: "oneOf".into(),
                    at: "/properties/n".into()
                }
            );
        }
        other => panic!("{:?}", other.map(|output| output.warnings)),
    }
    let lenient = OutputConstraint::JsonSchema {
        schema,
        strict: false,
    };
    let output = engine.compile(&lenient).unwrap();
    assert_eq!(
        output.warnings,
        [ConstraintWarning::OneOfAsAnyOf {
            at: "/properties/n".into()
        }]
    );
    for seed in 0..8 {
        let text = sample_text(&engine, &lenient, seed).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert!(value["n"].is_number(), "{text}");
    }
}

#[test]
fn naming_an_overlapping_one_of_takes_a_bounded_number_of_compiles() {
    // Twelve exclusive `oneOf`s before an overlapping one. Finding which to name stops after
    // eight tries, so the first `oneOf` is named; `strict` still refuses, and the non-strict
    // compile is still labelled.
    let mut properties = serde_json::Map::new();
    for k in 0..12 {
        properties.insert(
            format!("p{k:02}"),
            json!({ "oneOf": [{ "type": "string" }, { "type": "null" }] }),
        );
    }
    properties.insert(
        "z".into(),
        json!({ "oneOf": [{ "type": "integer" }, { "type": "number" }] }),
    );
    let schema = json!({ "type": "object", "properties": properties });
    let engine = engine();
    let strict = OutputConstraint::JsonSchema {
        schema: schema.clone(),
        strict: true,
    };
    assert_eq!(
        engine.compile(&strict).map(|output| output.warnings),
        Err(ConstraintError::Unsupported {
            keyword: "oneOf".into(),
            at: "/properties/p00".into()
        })
    );
    let lenient = OutputConstraint::JsonSchema {
        schema,
        strict: false,
    };
    assert_eq!(
        engine.compile(&lenient).unwrap().warnings,
        [ConstraintWarning::OneOfAsAnyOf {
            at: "/properties/p00".into()
        }]
    );
}

#[test]
fn an_exclusive_one_of_compiles_exactly_under_strict() {
    // Options of different types, or different constants, cannot both match one value.
    let engine = engine();
    for schema in [
        json!({ "oneOf": [{ "type": "string" }, { "type": "integer" }, { "type": "null" }] }),
        json!({ "oneOf": [{ "const": "heat" }, { "const": "cool" }, { "const": "auto" }] }),
    ] {
        let constraint = OutputConstraint::JsonSchema {
            schema: schema.clone(),
            strict: true,
        };
        let output = engine
            .compile(&constraint)
            .unwrap_or_else(|error| panic!("{schema}: {error}"));
        assert!(
            output.warnings.is_empty(),
            "{schema}: {:?}",
            output.warnings
        );
        for seed in 0..16 {
            let text = sample_text(&engine, &constraint, seed).unwrap();
            let value: Value = serde_json::from_str(&text).unwrap();
            validate(&schema, &value).unwrap_or_else(|reason| panic!("{reason}: {text}"));
        }
    }
}

#[test]
fn a_schema_nothing_can_satisfy_says_why() {
    for schema in [
        // A cycle of references that never adds anything.
        json!({
            "$defs": { "a": { "$ref": "#/$defs/b" }, "b": { "$ref": "#/$defs/a" } },
            "$ref": "#/$defs/a"
        }),
        // A string that must hold a control character constrained output does not write.
        json!({ "const": "\u{1}" }),
    ] {
        let reason = invalid(OutputConstraint::JsonSchema {
            schema: schema.clone(),
            strict: true,
        });
        assert!(
            reason.starts_with("no output can satisfy it: not one token of this tokenizer"),
            "{schema}: {reason}"
        );
    }
}

#[test]
fn a_text_token_that_starts_with_byte_0xff_is_never_allowed() {
    // No UTF-8 text holds byte 0xFF, and the engine's trie marks special tokens with it, so such
    // a token, the test vocabulary's single byte 0xFF among them, is kept out of every mask; the
    // other tokens stay constrained. One of them spells the special token `<|tool|>`'s trie
    // entry.
    let base = vocabulary();
    let mut tokens = base.tokens().to_vec();
    let mut never: Vec<TokenId> = vec![0xFF];
    for bytes in [&b"\xFFa"[..], b"\xFF{", b"\xFF<|tool|>"] {
        tokens.push(VocabToken::Text(bytes.to_vec()));
        never.push((tokens.len() - 1) as TokenId);
    }
    let engine = LlgEngine::new(Arc::new(Vocabulary::new(tokens, base.eos()).unwrap())).unwrap();
    let eos = engine.vocabulary().eos();
    for constraint in [
        OutputConstraint::JsonObject,
        OutputConstraint::Regex("a+[{]?".into()),
        OutputConstraint::Grammar {
            syntax: GrammarSyntax::Lark,
            text: "start: \"{\" \"a\"* \"}\"\n".into(),
        },
        OutputConstraint::Choice(vec!["a{".into(), "<|tool|>".into()]),
    ] {
        let output = engine.compile(&constraint).unwrap();
        for seed in 0..16 {
            let mut state = engine.new_state(&output.compiled).unwrap();
            let mut mask = TokenMask::new(engine.vocabulary().len());
            let mut rng = Rng::new(seed);
            for _ in 0..48 {
                engine.mask_into(&mut state, &mut mask).unwrap();
                for token in &never {
                    assert!(!mask.allows(*token), "{constraint:?} allows {token}");
                }
                let allowed: Vec<TokenId> = mask.allowed().filter(|t| *t != eos).collect();
                if allowed.is_empty() {
                    break;
                }
                let token = allowed[(rng.next_u64() % allowed.len() as u64) as usize];
                engine.advance(&mut state, token).unwrap();
            }
        }
        let mut fresh = engine.new_state(&output.compiled).unwrap();
        for token in &never {
            match engine.advance(&mut fresh, *token) {
                Err(ConstraintError::Rejected { reason, .. }) => {
                    assert!(reason.contains("starts with byte 0xFF"), "{reason}");
                }
                other => panic!("{constraint:?} accepted {token}: {other:?}"),
            }
        }
    }
    // Constrained output from this vocabulary still validates.
    for seed in 0..16 {
        let text = sample_text(&engine, &OutputConstraint::JsonObject, seed).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert!(value.is_object(), "{text}");
    }
}

#[test]
fn an_additional_property_name_cannot_spell_a_declared_one() {
    // With additional properties open, a new name must differ from every declared name once
    // decoded. Strings carry no `\u` escapes, so `"é"` cannot stand in for the declared
    // `"é"` and leave its value unchecked.
    let engine = engine();
    let constraint = OutputConstraint::JsonSchema {
        schema: json!({ "type": "object", "properties": { "é": { "type": "string" } } }),
        strict: true,
    };
    let output = engine.compile(&constraint).unwrap();
    let mut state = engine.new_state(&output.compiled).unwrap();
    for byte in *b"{\"\\" {
        engine.advance(&mut state, u32::from(byte)).unwrap();
    }
    let mut mask = TokenMask::new(engine.vocabulary().len());
    engine.mask_into(&mut state, &mut mask).unwrap();
    assert!(!mask.allows(u32::from(b'u')), "a `\\u` escape is allowed");
    for escape in *b"\"\\bfnrt" {
        assert!(
            mask.allows(u32::from(escape)),
            "`\\{}` is refused",
            escape as char
        );
    }
    for seed in 0..64 {
        let text = sample_text(&engine, &constraint, seed).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        if let Some(declared) = value.get("é") {
            assert!(declared.is_string(), "{text}");
        }
    }
}

#[test]
fn properties_come_out_in_the_schema_order_with_fixed_separators() {
    let engine = engine();
    let schema = json!({
        "type": "object",
        "properties": { "zeta": { "const": 1 }, "alpha": { "const": 2 }, "mid": { "const": [true, null] } },
        "required": ["zeta", "alpha", "mid"],
        "additionalProperties": false
    });
    let text = sample_text(
        &engine,
        &OutputConstraint::JsonSchema {
            schema,
            strict: true,
        },
        1,
    )
    .unwrap();
    assert_eq!(text, r#"{"zeta": 1, "alpha": 2, "mid": [true, null]}"#);
}

#[test]
fn the_engine_refuses_a_forbidden_token_and_a_wrong_sized_mask() {
    let engine = engine();
    let output = engine
        .compile(&OutputConstraint::Regex("ab".into()))
        .unwrap();
    let mut state = engine.new_state(&output.compiled).unwrap();
    let mut mask = TokenMask::new(engine.vocabulary().len());
    engine.mask_into(&mut state, &mut mask).unwrap();
    assert!(mask.allows(u32::from(b'a')));
    assert!(!mask.allows(u32::from(b'b')));
    assert!(!mask.allows(engine.vocabulary().eos()));
    assert!(!engine.is_accepting(&mut state).unwrap());
    match engine.advance(&mut state, u32::from(b'z')) {
        Err(error @ ConstraintError::Rejected { .. }) => assert_eq!(error.code(), "internal_error"),
        other => panic!("{other:?}"),
    }

    let mut fresh = engine.new_state(&output.compiled).unwrap();
    let mut short = TokenMask::new(3);
    assert!(matches!(
        engine.mask_into(&mut fresh, &mut short),
        Err(ConstraintError::Invalid { .. })
    ));
    engine.advance(&mut fresh, u32::from(b'a')).unwrap();
    engine.advance(&mut fresh, u32::from(b'b')).unwrap();
    assert!(engine.is_accepting(&mut fresh).unwrap());
    engine.mask_into(&mut fresh, &mut mask).unwrap();
    assert_eq!(
        mask.allowed().collect::<Vec<_>>(),
        [engine.vocabulary().eos()]
    );
}

#[test]
fn special_tokens_are_never_matched_as_text() {
    let engine = engine();
    // A regex that matches the special token's name as text still allows only text tokens.
    let output = engine
        .compile(&OutputConstraint::Regex("<\\|tool\\|>".into()))
        .unwrap();
    let mut state = engine.new_state(&output.compiled).unwrap();
    let mut mask = TokenMask::new(engine.vocabulary().len());
    engine.mask_into(&mut state, &mut mask).unwrap();
    let vocabulary = engine.vocabulary();
    for token in mask.allowed() {
        assert!(
            matches!(
                vocabulary.token(token),
                Some(syntropy_text::constraint::VocabToken::Text(_))
            ),
            "{token} is special"
        );
    }
}

#[test]
fn states_of_one_compiled_constraint_are_independent() {
    let engine = engine();
    let output = engine
        .compile(&OutputConstraint::Choice(vec!["ab".into(), "cd".into()]))
        .unwrap();
    let mut one = engine.new_state(&output.compiled).unwrap();
    let mut two = engine.new_state(&output.compiled).unwrap();
    engine.advance(&mut one, u32::from(b'a')).unwrap();
    let mut mask = TokenMask::new(engine.vocabulary().len());
    engine.mask_into(&mut two, &mut mask).unwrap();
    assert!(mask.allows(u32::from(b'a')) && mask.allows(u32::from(b'c')));
    engine.mask_into(&mut one, &mut mask).unwrap();
    assert!(mask.allows(u32::from(b'b')) && !mask.allows(u32::from(b'd')));
}
