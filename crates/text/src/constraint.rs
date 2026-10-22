// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grammar-constrained decoding: constraints compiled into token masks (RFC-0050 §7, ADR-0504;
//! PROD-E3-T04).
//!
//! A request's [`OutputConstraint`] (a JSON schema, a regex, a grammar or a choice) compiles,
//! against a tokenizer's [`Vocabulary`], into an automaton that a [`ConstraintEngine`] steps token
//! by token. At each step [`ConstraintEngine::mask_into`] writes the [`TokenMask`] of tokens that
//! keep the output valid. [`LlgEngine`] is the engine, over `llguidance`, and [`CompileCache`]
//! keeps compiled constraints by BLAKE3(constraint, tokenizer).
//!
//! **Exactness (D3, RFC-0050 §7.4).** This module computes masks and never touches logits. The
//! sampler applies a mask only to a request that asked for a constraint (PROD-E3-T05), so a
//! request without one never reaches the engine and its output cannot change. A constraint is
//! part of the sampler the request asked for, and the reference applies the same mask, so a
//! constrained exact request stays exact.

mod cache;
mod engine;
mod kinds;
mod llg;
mod schema;
mod vocab;

pub use cache::{CompileCache, ConstraintKey, key};
pub use engine::{CompileOutput, ConstraintEngine, ConstraintError, ConstraintWarning, TokenMask};
pub use kinds::{GrammarSyntax, OutputConstraint};
pub use llg::{LlgCompiled, LlgEngine, LlgState};
pub use vocab::{TokenId, VocabToken, Vocabulary, VocabularyError};
