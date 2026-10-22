// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `llguidance` backend of [`ConstraintEngine`] (RFC-0050 §7.2, ADR-0504).
//!
//! JSON Schema, regular expressions and Lark-style grammars go to `llguidance` as they are, after
//! [`super::schema`] has checked a schema's coverage. A choice becomes a regular expression that
//! matches exactly one of its strings. A JSON schema compiles with these options:
//!
//! - `coerce_one_of`, so `oneOf` is treated as `anyOf`, as RFC-0050 §7.2 says;
//! - fixed separators, `", "` between items and `": "` after a key, with no other whitespace. The
//!   output stays valid JSON, and a model cannot loop on whitespace inside a structure;
//! - no `\uXXXX` escapes in strings, only `\"`, `\\`, `\b`, `\f`, `\n`, `\r` and `\t`, so each
//!   string has one spelling. An additional property's name is any name but the declared ones,
//!   and `llguidance` excludes those by their spelling: with `\u` escapes allowed, `"\u00e9"`
//!   would pass as a new name and then decode to a declared `"é"`, with its value unchecked;
//! - property order as the schema lists it, which `llguidance` keeps by default.

use std::sync::{Arc, Mutex, PoisonError};

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory, TokenParser};
use serde_json::{Value, json};

use super::{
    CompileOutput, ConstraintEngine, ConstraintError, GrammarSyntax, OutputConstraint, TokenId,
    TokenMask, VocabToken, Vocabulary, schema,
};

/// The marker byte the engine's trie puts before a special token's name.
const SPECIAL_MARKER: u8 = TokTrie::SPECIAL_TOKEN_MARKER;

/// [`ConstraintEngine`] over `llguidance`, for one vocabulary.
pub struct LlgEngine {
    vocabulary: Arc<Vocabulary>,
    factory: ParserFactory,
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
        let words: Vec<Vec<u8>> = vocabulary
            .tokens()
            .iter()
            .map(|token| match token {
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
        })
    }

    /// The grammar `constraint` lowers to, with its warnings.
    fn lower(
        constraint: &OutputConstraint,
    ) -> Result<(TopLevelGrammar, Vec<super::ConstraintWarning>), ConstraintError> {
        Ok(match constraint {
            OutputConstraint::JsonObject => (json_grammar(json!({ "type": "object" })), Vec::new()),
            OutputConstraint::JsonSchema { schema, strict } => {
                let checked = schema::check(schema, *strict)?;
                (json_grammar(checked.schema), checked.warnings)
            }
            OutputConstraint::Regex(pattern) => (TopLevelGrammar::from_regex(pattern), Vec::new()),
            OutputConstraint::Grammar {
                syntax: GrammarSyntax::Lark,
                text,
            } => (TopLevelGrammar::from_lark(text.clone()), Vec::new()),
            OutputConstraint::Choice(choices) => {
                if choices.is_empty() {
                    return Err(ConstraintError::Invalid {
                        reason: "a choice needs at least one string".to_owned(),
                    });
                }
                let alternatives: Vec<String> = choices.iter().map(|c| regex_literal(c)).collect();
                let pattern = format!("(?:{})", alternatives.join("|"));
                (TopLevelGrammar::from_regex(&pattern), Vec::new())
            }
        })
    }
}

/// A JSON-schema grammar with the options the module docs list.
fn json_grammar(mut schema: Value) -> TopLevelGrammar {
    if let Value::Object(object) = &mut schema {
        object.insert(
            "x-guidance".to_owned(),
            json!({
                "coerce_one_of": true,
                "whitespace_flexible": false,
                "item_separator": ", ",
                "key_separator": ": ",
                "json_allowed_escapes": "nrbtf\\\"",
            }),
        );
    }
    TopLevelGrammar::from_json_schema(schema)
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
        let (grammar, warnings) = Self::lower(constraint)?;
        let parser = self.factory.create_parser(grammar).map_err(invalid)?;
        // A grammar that cannot start fails here, not at the first mask.
        let mut probe = Matcher::new(Ok(parser.deep_clone()));
        probe.compute_mask_or_eos().map_err(invalid)?;
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
