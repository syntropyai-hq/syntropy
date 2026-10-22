// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `llguidance` backend of [`ConstraintEngine`] (RFC-0050 §7.2, ADR-0504).
//!
//! JSON Schema, regular expressions and Lark-style grammars go to `llguidance` as they are, after
//! [`super::schema`] has checked a schema's coverage. A choice becomes a regular expression that
//! matches exactly one of its strings. A JSON schema compiles with these options:
//!
//! - `oneOf` exactly when the engine can prove its options exclusive, as when they differ in type
//!   or constant: it then compiles as `anyOf` with nothing lost. Otherwise an output could match
//!   two options, which `oneOf` forbids. Under `strict` such a `oneOf` is refused with
//!   `constraint_unsupported`, naming it, and without `strict` it is compiled as `anyOf` and named
//!   in a [`super::ConstraintWarning::OneOfAsAnyOf`]. It is never approximated without a label;
//! - fixed separators, `", "` between items and `": "` after a key, with no other whitespace. The
//!   output stays valid JSON, and a model cannot loop on whitespace inside a structure;
//! - no `\uXXXX` escapes in strings, only `\"`, `\\`, `\b`, `\f`, `\n`, `\r` and `\t`, so each
//!   string has one spelling. An additional property's name is any name but the declared ones,
//!   and `llguidance` excludes those by their spelling: with `\u` escapes allowed, `"\u00e9"`
//!   would pass as a new name and then decode to a declared `"é"`, with its value unchecked;
//! - property order as the schema lists it, which `llguidance` keeps by default.
//!
//! A text token that is empty, or that starts with byte `0xFF`, which no UTF-8 text holds and the
//! engine's trie uses to mark special tokens, has no bytes in the trie, so no mask allows it and
//! [`ConstraintEngine::advance`] refuses it. Every other token of the vocabulary stays
//! constrained as usual.

use std::sync::{Arc, Mutex, PoisonError};

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory, TokenParser};
use serde_json::{Value, json};

use super::{
    CompileOutput, ConstraintEngine, ConstraintError, ConstraintWarning, GrammarSyntax,
    OutputConstraint, TokenId, TokenMask, VocabToken, Vocabulary, schema,
};

/// The marker byte the engine's trie puts before a special token's name.
const SPECIAL_MARKER: u8 = TokTrie::SPECIAL_TOKEN_MARKER;

/// How `llguidance` refuses a `oneOf` it cannot prove exclusive when `coerce_one_of` is off.
const ONE_OF_REFUSED: &str = "oneOf constraints are not supported";

/// The grammar warning `llguidance` records when it compiles such a `oneOf` as `anyOf`.
const ONE_OF_COERCED: &str = "oneOf not fully supported, falling back to anyOf";

/// At most this many extra compiles go into finding which `oneOf` to name, so a schema of many
/// `oneOf`s cannot multiply the cost of a compile. Past them the first `oneOf` is named.
const LOCATE_LIMIT: usize = 8;

/// [`ConstraintEngine`] over `llguidance`, for one vocabulary.
pub struct LlgEngine {
    vocabulary: Arc<Vocabulary>,
    factory: ParserFactory,
    /// Text tokens with no bytes in the trie, by ID in increasing order: no constrained output
    /// holds one.
    unwritable: Vec<TokenId>,
}

/// A constraint compiled by [`LlgEngine`].
pub struct LlgCompiled {
    /// The parser at the start of the constraint. Each sequence starts from a copy.
    template: Mutex<TokenParser>,
}

/// One sequence in an [`LlgCompiled`].
pub struct LlgState {
    matcher: Matcher,
}

impl LlgEngine {
    /// An engine for `vocabulary`. Builds the vocabulary's trie, the cost a tokenizer pays once.
    pub fn new(vocabulary: Arc<Vocabulary>) -> Result<LlgEngine, ConstraintError> {
        let mut unwritable = Vec::new();
        let words: Vec<Vec<u8>> = vocabulary
            .tokens()
            .iter()
            .enumerate()
            .map(|(id, token)| match token {
                // Kept out of the trie: as bytes, it would read as a special token's name.
                VocabToken::Text(bytes) if bytes.first().is_none_or(|b| *b == SPECIAL_MARKER) => {
                    // `Vocabulary::new` keeps every ID within a `TokenId`.
                    let id = id as TokenId;
                    if id != vocabulary.eos() {
                        unwritable.push(id);
                    }
                    Vec::new()
                }
                VocabToken::Text(bytes) => bytes.clone(),
                VocabToken::Special(name) => {
                    let mut word = vec![SPECIAL_MARKER];
                    word.extend_from_slice(name.as_bytes());
                    word
                }
            })
            .collect();
        // `Vocabulary::new` keeps the length within a `TokenId`.
        let info = TokRxInfo::new(words.len() as TokenId, vocabulary.eos());
        let trie = TokTrie::from(&info, &words);
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&env).map_err(invalid)?;
        factory.quiet();
        Ok(LlgEngine {
            vocabulary,
            factory,
            unwritable,
        })
    }

    /// The parser for `grammar`, or `llguidance`'s reason it has none.
    fn parser(&self, grammar: TopLevelGrammar) -> Result<TokenParser, String> {
        self.factory
            .create_parser(grammar)
            .map_err(|error| error.to_string())
    }

    /// The parser for the checked JSON schema `checked`, with what the compile adds to its
    /// warnings.
    fn json_parser(
        &self,
        checked: schema::Checked,
        strict: bool,
        warnings: &mut Vec<ConstraintWarning>,
    ) -> Result<TokenParser, ConstraintError> {
        if strict {
            return match self.parser(json_grammar(checked.schema.clone(), false)) {
                Err(reason) if reason.contains(ONE_OF_REFUSED) => {
                    let at = self.overlapping(&checked).into_iter().next();
                    Err(ConstraintError::Unsupported {
                        keyword: "oneOf".to_owned(),
                        at: at.unwrap_or_default(),
                    })
                }
                other => other.map_err(compile_error),
            };
        }
        let mut parser = self
            .parser(json_grammar(checked.schema.clone(), true))
            .map_err(compile_error)?;
        if parser
            .grammar_warnings()
            .iter()
            .any(|warning| warning.contains(ONE_OF_COERCED))
        {
            warnings.extend(
                self.overlapping(&checked)
                    .into_iter()
                    .map(|at| ConstraintWarning::OneOfAsAnyOf { at }),
            );
        }
        Ok(parser)
    }

    /// The `oneOf`s of `checked` that `llguidance` cannot prove exclusive, in document order, or
    /// the first `oneOf` when none is found within [`LOCATE_LIMIT`] tries. Each is tried alone,
    /// with every other `oneOf` read as `anyOf`; this runs only once a compile has shown that one
    /// exists, so the answer decides only which location is named.
    fn overlapping(&self, checked: &schema::Checked) -> Vec<String> {
        let mut found = Vec::new();
        for at in checked.one_ofs.iter().take(LOCATE_LIMIT) {
            if found.contains(at) {
                continue;
            }
            let mut alone = checked.schema.clone();
            for other in checked.one_ofs.iter().filter(|other| *other != at) {
                if let Some(Value::Object(object)) = alone.pointer_mut(other) {
                    as_any_of(object);
                }
            }
            if let Err(reason) = self.parser(json_grammar(alone, false))
                && reason.contains(ONE_OF_REFUSED)
            {
                found.push(at.clone());
            }
        }
        if found.is_empty() {
            found.extend(checked.one_ofs.first().cloned());
        }
        found
    }

    /// What `constraint` lowers to, with the schema check's warnings.
    fn lower(constraint: &OutputConstraint) -> Result<Lowered, ConstraintError> {
        Ok(match constraint {
            OutputConstraint::JsonObject => {
                Lowered::Grammar(json_grammar(json!({ "type": "object" }), false))
            }
            OutputConstraint::JsonSchema { schema, strict } => Lowered::Json {
                checked: schema::check(schema, *strict)?,
                strict: *strict,
            },
            OutputConstraint::Regex(pattern) => {
                Lowered::Grammar(TopLevelGrammar::from_regex(pattern))
            }
            OutputConstraint::Grammar {
                syntax: GrammarSyntax::Lark,
                text,
            } => Lowered::Grammar(TopLevelGrammar::from_lark(text.clone())),
            OutputConstraint::Choice(choices) => {
                if choices.is_empty() {
                    return Err(ConstraintError::Invalid {
                        reason: "a choice needs at least one string".to_owned(),
                    });
                }
                let alternatives: Vec<String> = choices.iter().map(|c| regex_literal(c)).collect();
                let pattern = format!("(?:{})", alternatives.join("|"));
                Lowered::Grammar(TopLevelGrammar::from_regex(&pattern))
            }
        })
    }
}

/// What a constraint lowers to before it compiles.
enum Lowered {
    /// A checked JSON schema, compiled by [`LlgEngine::json_parser`].
    Json {
        checked: schema::Checked,
        strict: bool,
    },
    /// Any other grammar, compiled as it is.
    Grammar(TopLevelGrammar),
}

/// The `oneOf` of `object`, read as `anyOf`; beside an `anyOf` of its own, it joins `allOf`.
fn as_any_of(object: &mut serde_json::Map<String, Value>) {
    let Some(options) = object.shift_remove("oneOf") else {
        return;
    };
    if !object.contains_key("anyOf") {
        object.insert("anyOf".to_owned(), options);
        return;
    }
    let wrapped = json!({ "anyOf": options });
    match object.get_mut("allOf") {
        Some(Value::Array(all)) => all.push(wrapped),
        _ => {
            object.insert("allOf".to_owned(), json!([wrapped]));
        }
    }
}

/// A JSON-schema grammar with the options the module docs list; `coerce_one_of` compiles a `oneOf`
/// the engine cannot prove exclusive as `anyOf` instead of refusing it.
fn json_grammar(mut schema: Value, coerce_one_of: bool) -> TopLevelGrammar {
    if let Value::Object(object) = &mut schema {
        object.insert(
            "x-guidance".to_owned(),
            json!({
                "coerce_one_of": coerce_one_of,
                "whitespace_flexible": false,
                "item_separator": ", ",
                "key_separator": ": ",
                "json_allowed_escapes": "nrbtf\\\"",
            }),
        );
    }
    TopLevelGrammar::from_json_schema(schema)
}

/// A compile failure, with `llguidance`'s internal name for "nothing can start" replaced by what it
/// means.
fn compile_error(reason: String) -> ConstraintError {
    if reason.contains("NoExtensionBias") {
        return ConstraintError::Invalid {
            reason: "no output can satisfy it: not one token of this tokenizer can start an \
                     output that matches. A schema does this when it can never be met, such as a \
                     `$ref` cycle that adds nothing, or a string that would have to hold a \
                     control character other than \\b, \\f, \\n, \\r or \\t, which constrained \
                     output does not write"
                .to_owned(),
        };
    }
    ConstraintError::Invalid { reason }
}

/// `text` as a regular expression that matches exactly it.
fn regex_literal(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if "\\.+*?()|[]{}^$#&-~".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn invalid(error: impl std::fmt::Display) -> ConstraintError {
    ConstraintError::Invalid {
        reason: error.to_string(),
    }
}

impl ConstraintEngine for LlgEngine {
    type Compiled = LlgCompiled;
    type State = LlgState;

    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn compile(
        &self,
        constraint: &OutputConstraint,
    ) -> Result<CompileOutput<LlgCompiled>, ConstraintError> {
        let mut warnings = Vec::new();
        let parser = match Self::lower(constraint)? {
            Lowered::Grammar(grammar) => self.parser(grammar).map_err(compile_error)?,
            Lowered::Json { checked, strict } => {
                warnings.extend(checked.warnings.iter().cloned());
                self.json_parser(checked, strict, &mut warnings)?
            }
        };
        // A grammar that cannot start fails here, not at the first mask.
        let mut probe = Matcher::new(Ok(parser.deep_clone()));
        probe
            .compute_mask_or_eos()
            .map_err(|error| compile_error(error.to_string()))?;
        Ok(CompileOutput {
            compiled: Arc::new(LlgCompiled {
                template: Mutex::new(parser),
            }),
            warnings,
        })
    }

    fn new_state(&self, compiled: &LlgCompiled) -> Result<LlgState, ConstraintError> {
        let parser = compiled
            .template
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .deep_clone();
        Ok(LlgState {
            matcher: Matcher::new(Ok(parser)),
        })
    }

    fn advance(&self, state: &mut LlgState, token: TokenId) -> Result<(), ConstraintError> {
        if self.unwritable.binary_search(&token).is_ok() {
            return Err(ConstraintError::Rejected {
                token,
                reason: "the token is empty or starts with byte 0xFF, so no constrained output \
                         holds it"
                    .to_owned(),
            });
        }
        state
            .matcher
            .consume_token(token)
            .map_err(|error| ConstraintError::Rejected {
                token,
                reason: error.to_string(),
            })
    }

    fn mask_into(&self, state: &mut LlgState, mask: &mut TokenMask) -> Result<(), ConstraintError> {
        if mask.len() != self.vocabulary.len() {
            return Err(ConstraintError::Invalid {
                reason: format!(
                    "a mask of {} tokens for a vocabulary of {}",
                    mask.len(),
                    self.vocabulary.len()
                ),
            });
        }
        // Once the output cannot be extended the parser stops, and its mask is the end token
        // alone.
        let allowed = state.matcher.compute_mask_or_eos().map_err(invalid)?;
        mask.copy_from_words(allowed.as_slice());
        Ok(())
    }

    fn is_accepting(&self, state: &mut LlgState) -> Result<bool, ConstraintError> {
        state.matcher.is_accepting().map_err(invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}

    #[test]
    fn compiled_constraints_are_shared_and_states_move_between_threads() {
        send_sync::<LlgCompiled>();
        send::<LlgState>();
        send_sync::<LlgEngine>();
    }

    #[test]
    fn regex_literals_escape_every_metacharacter() {
        assert_eq!(regex_literal("a.b*c"), r"a\.b\*c");
        assert_eq!(regex_literal("(x|y)[z]{1}^$"), r"\(x\|y\)\[z\]\{1\}\^\$");
        assert_eq!(regex_literal("plain words"), "plain words");
    }
}
