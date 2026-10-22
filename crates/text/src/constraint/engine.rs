// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `ConstraintEngine` trait (ADR-0504 decision 6), its token masks, and its errors.

use std::sync::Arc;

use syntropy_core::ErrorClass;

use super::{OutputConstraint, TokenId, Vocabulary};

/// A grammar engine that turns constraints into token masks for one vocabulary.
///
/// One engine serves one tokenizer: building the vocabulary's trie is the expensive part, so it is
/// done once, and every constraint compiled for that tokenizer reuses it (RFC-0050 §7.2). Any
/// engine that implements the trait can replace the `llguidance` one (ADR-0504).
pub trait ConstraintEngine {
    /// A compiled constraint, shared by every sequence that uses it.
    type Compiled: Send + Sync;
    /// One sequence's position in a compiled constraint.
    type State: Send;

    /// The vocabulary the engine compiles for.
    fn vocabulary(&self) -> &Vocabulary;

    /// Compiles `constraint`, with a warning for each keyword it ignored.
    fn compile(
        &self,
        constraint: &OutputConstraint,
    ) -> Result<CompileOutput<Self::Compiled>, ConstraintError>;

    /// A sequence at the start of `compiled`.
    fn new_state(&self, compiled: &Self::Compiled) -> Result<Self::State, ConstraintError>;

    /// Moves `state` past `token`. A token the last mask did not allow is an error.
    fn advance(&self, state: &mut Self::State, token: TokenId) -> Result<(), ConstraintError>;

    /// Writes into `mask` which tokens may come next. The end-of-sequence token is allowed exactly
    /// when the output so far is complete.
    fn mask_into(
        &self,
        state: &mut Self::State,
        mask: &mut TokenMask,
    ) -> Result<(), ConstraintError>;

    /// Whether the output so far is complete, so the sequence may end here.
    fn is_accepting(&self, state: &mut Self::State) -> Result<bool, ConstraintError>;
}

/// A compiled constraint and what compiling it ignored.
#[derive(Debug)]
pub struct CompileOutput<C> {
    pub compiled: Arc<C>,
    /// One per ignored keyword, for `Syntropy-Warning` (RFC-0050 §7.2). Empty under `strict`.
    pub warnings: Vec<ConstraintWarning>,
}

impl<C> Clone for CompileOutput<C> {
    fn clone(&self) -> Self {
        CompileOutput {
            compiled: Arc::clone(&self.compiled),
            warnings: self.warnings.clone(),
        }
    }
}

/// What a non-strict compile did to a schema that `strict` would have refused. Each is named in
/// the response's `Syntropy-Warning` (RFC-0050 §7.2). `at` is a JSON pointer into the schema, such
/// as `/properties/when`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintWarning {
    /// An unsupported keyword, such as `patternProperties`, or `$ref` for a reference outside the
    /// document, was ignored.
    Ignored { keyword: String, at: String },
    /// A `oneOf` whose options may overlap was compiled as `anyOf`: an output may match more than
    /// one option, which `oneOf` does not allow.
    OneOfAsAnyOf { at: String },
}

impl std::fmt::Display for ConstraintWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shown = |at: &str| {
            if at.is_empty() {
                "/".to_owned()
            } else {
                at.to_owned()
            }
        };
        match self {
            ConstraintWarning::Ignored { keyword, at } => write!(
                f,
                "unsupported schema keyword `{keyword}` at {} was ignored",
                shown(at)
            ),
            ConstraintWarning::OneOfAsAnyOf { at } => write!(
                f,
                "`oneOf` at {} was treated as `anyOf`: its options may overlap, so the output may \
                 match more than one of them",
                shown(at)
            ),
        }
    }
}

/// Why a constraint did not compile, or a sequence could not move on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintError {
    /// The constraint is malformed: a bad schema, regex or grammar, or one nothing can satisfy.
    Invalid { reason: String },
    /// A `strict` schema uses a keyword outside RFC-0050 §7.2's coverage.
    Unsupported { keyword: String, at: String },
    /// A token the mask did not allow was passed to [`ConstraintEngine::advance`]. The sampler
    /// applies the mask, so this is a bug.
    Rejected { token: TokenId, reason: String },
}

impl ConstraintError {
    /// The error class (RFC-0050 §11 maps it to a status).
    pub fn class(&self) -> ErrorClass {
        match self {
            ConstraintError::Invalid { .. } => ErrorClass::InvalidRequest,
            ConstraintError::Unsupported { .. } => ErrorClass::Unsupported,
            ConstraintError::Rejected { .. } => ErrorClass::Internal,
        }
    }

    /// The `syntropy.error.code` name (RFC-0050 §11).
    pub fn code(&self) -> &'static str {
        match self {
            ConstraintError::Invalid { .. } => "constraint_invalid",
            ConstraintError::Unsupported { .. } => "constraint_unsupported",
            ConstraintError::Rejected { .. } => "internal_error",
        }
    }
}

impl std::fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintError::Invalid { reason } => write!(f, "invalid constraint: {reason}"),
            ConstraintError::Unsupported { keyword, at } => {
                let at = if at.is_empty() { "/" } else { at };
                write!(
                    f,
                    "schema keyword `{keyword}` at {at} is not supported under `strict`"
                )
            }
            ConstraintError::Rejected { token, reason } => {
                write!(
                    f,
                    "token {token} does not continue the constraint: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for ConstraintError {}

/// The tokens that may come next: one bit per token ID, set when the token is allowed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenMask {
    words: Vec<u32>,
    len: usize,
}

impl TokenMask {
    /// A mask over `len` tokens that allows none.
    pub fn new(len: usize) -> TokenMask {
        TokenMask {
            words: vec![0; len.div_ceil(32)],
            len,
        }
    }

    /// How many tokens it covers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it covers no tokens.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether `token` is allowed. A token outside the mask is not.
    pub fn allows(&self, token: TokenId) -> bool {
        let token = token as usize;
        token < self.len && self.words[token / 32] & (1 << (token % 32)) != 0
    }

    /// Allows or forbids `token`.
    ///
    /// # Panics
    ///
    /// If `token` is outside the mask.
    pub fn set(&mut self, token: TokenId, allowed: bool) {
        let token = token as usize;
        assert!(
            token < self.len,
            "token {token} is outside a mask of {}",
            self.len
        );
        let bit = 1 << (token % 32);
        if allowed {
            self.words[token / 32] |= bit;
        } else {
            self.words[token / 32] &= !bit;
        }
    }

    /// How many tokens are allowed.
    pub fn count(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// The allowed tokens, in ID order.
    pub fn allowed(&self) -> impl Iterator<Item = TokenId> + '_ {
        (0..self.len as TokenId).filter(|&token| self.allows(token))
    }

    /// Forbids every token, then allows those whose bits are set in `words`, 32 tokens a word
    /// with the lowest bit first. Bits past `len` are ignored.
    pub(crate) fn copy_from_words(&mut self, words: &[u32]) {
        self.words.iter_mut().for_each(|word| *word = 0);
        for (into, from) in self.words.iter_mut().zip(words) {
            *into = *from;
        }
        let tail = self.len % 32;
        if tail != 0
            && let Some(last) = self.words.last_mut()
        {
            *last &= (1 << tail) - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mask_sets_reads_and_counts_bits() {
        let mut mask = TokenMask::new(70);
        assert_eq!(mask.len(), 70);
        assert_eq!(mask.count(), 0);
        for token in [0, 31, 32, 69] {
            mask.set(token, true);
        }
        assert!(mask.allows(31) && mask.allows(32) && mask.allows(69));
        assert!(!mask.allows(1) && !mask.allows(70) && !mask.allows(u32::MAX));
        assert_eq!(mask.allowed().collect::<Vec<_>>(), [0, 31, 32, 69]);
        mask.set(31, false);
        assert_eq!(mask.count(), 3);
        assert!(TokenMask::new(0).is_empty());
    }

    #[test]
    fn copying_words_clears_old_bits_and_ignores_bits_past_the_end() {
        let mut mask = TokenMask::new(34);
        mask.set(5, true);
        mask.copy_from_words(&[0b1010, u32::MAX, u32::MAX]);
        assert_eq!(mask.allowed().collect::<Vec<_>>(), [1, 3, 32, 33]);
    }

    #[test]
    #[should_panic(expected = "outside a mask")]
    fn setting_a_token_outside_the_mask_panics() {
        TokenMask::new(3).set(3, true);
    }

    #[test]
    fn errors_carry_rfc_0050_codes_and_classes() {
        let invalid = ConstraintError::Invalid { reason: "x".into() };
        let unsupported = ConstraintError::Unsupported {
            keyword: "not".into(),
            at: "/properties/a".into(),
        };
        assert_eq!(invalid.code(), "constraint_invalid");
        assert_eq!(invalid.class(), ErrorClass::InvalidRequest);
        assert_eq!(unsupported.code(), "constraint_unsupported");
        assert_eq!(unsupported.class(), ErrorClass::Unsupported);
        assert_eq!(
            unsupported.to_string(),
            "schema keyword `not` at /properties/a is not supported under `strict`"
        );
        let warning = ConstraintWarning::Ignored {
            keyword: "uniqueItems".into(),
            at: String::new(),
        };
        assert_eq!(
            warning.to_string(),
            "unsupported schema keyword `uniqueItems` at / was ignored"
        );
        let coerced = ConstraintWarning::OneOfAsAnyOf {
            at: "/properties/n".into(),
        };
        assert_eq!(
            coerced.to_string(),
            "`oneOf` at /properties/n was treated as `anyOf`: its options may overlap, so the \
             output may match more than one of them"
        );
    }
}
