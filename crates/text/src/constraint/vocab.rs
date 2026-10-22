// SPDX-License-Identifier: MIT OR Apache-2.0

//! The vocabulary a constraint compiles against: each token's bytes, by token ID.
//!
//! This is the one type the tokenizer side (COMP) hands the constraint engine. It holds what a
//! grammar needs and nothing about how text is split into tokens: the bytes each token stands for,
//! which tokens are special, and which token ends a sequence.

/// A token's position in its tokenizer's vocabulary.
pub type TokenId = u32;

/// One vocabulary entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VocabToken {
    /// A token that stands for these bytes of text. A byte-level token may hold part of a UTF-8
    /// character.
    Text(Vec<u8>),
    /// A special token, such as an end-of-turn or tool-call marker, by its name. Text never
    /// matches it; a grammar can only produce it as itself.
    Special(String),
}

/// A tokenizer's vocabulary as the constraint engine reads it.
#[derive(Clone, Debug)]
pub struct Vocabulary {
    tokens: Vec<VocabToken>,
    eos: TokenId,
    digest: [u8; 32],
}

/// Why a vocabulary was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VocabularyError {
    /// No tokens.
    Empty,
    /// More tokens than a `TokenId` can number.
    TooLarge(usize),
    /// The end-of-sequence token is not in the vocabulary.
    EosOutOfRange { eos: TokenId, len: usize },
    /// A text token of more than one byte starts with `0xFF`. No UTF-8 text holds that byte, and
    /// the engine's trie marks special tokens with it.
    MarkerByte(TokenId),
}

impl std::fmt::Display for VocabularyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VocabularyError::Empty => write!(f, "the vocabulary has no tokens"),
            VocabularyError::TooLarge(len) => {
                write!(f, "{len} tokens is more than a token ID numbers")
            }
            VocabularyError::EosOutOfRange { eos, len } => {
                write!(
                    f,
                    "the end-of-sequence token {eos} is outside a vocabulary of {len}"
                )
            }
            VocabularyError::MarkerByte(id) => {
                write!(
                    f,
                    "text token {id} starts with byte 0xFF and is longer than one byte"
                )
            }
        }
    }
}

impl std::error::Error for VocabularyError {}

impl Vocabulary {
    /// The vocabulary of `tokens`, by ID, whose sequences end with `eos`.
    pub fn new(tokens: Vec<VocabToken>, eos: TokenId) -> Result<Vocabulary, VocabularyError> {
        if tokens.is_empty() {
            return Err(VocabularyError::Empty);
        }
        if TokenId::try_from(tokens.len()).is_err() {
            return Err(VocabularyError::TooLarge(tokens.len()));
        }
        if eos as usize >= tokens.len() {
            return Err(VocabularyError::EosOutOfRange {
                eos,
                len: tokens.len(),
            });
        }
        for (id, token) in tokens.iter().enumerate() {
            if let VocabToken::Text(bytes) = token
                && bytes.len() > 1
                && bytes[0] == 0xFF
            {
                // `id` fits a `TokenId`: the length check above passed.
                return Err(VocabularyError::MarkerByte(id as TokenId));
            }
        }
        let digest = digest(&tokens, eos);
        Ok(Vocabulary {
            tokens,
            eos,
            digest,
        })
    }

    /// How many tokens it holds.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Never true: a vocabulary has at least one token.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The token that ends a sequence.
    pub fn eos(&self) -> TokenId {
        self.eos
    }

    /// Token `id`, if the vocabulary has it.
    pub fn token(&self, id: TokenId) -> Option<&VocabToken> {
        self.tokens.get(id as usize)
    }

    /// Every token, by ID.
    pub fn tokens(&self) -> &[VocabToken] {
        &self.tokens
    }

    /// BLAKE3 of the vocabulary's contents: the tokenizer half of a compiled constraint's cache
    /// key (RFC-0050 §7.3). Two vocabularies with the same tokens and end token share it.
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

/// BLAKE3 over a length-prefixed encoding, so no two vocabularies collide by concatenation.
fn digest(tokens: &[VocabToken], eos: TokenId) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"syntropy-text vocabulary v1\0");
    hasher.update(&(tokens.len() as u64).to_le_bytes());
    for token in tokens {
        let (tag, bytes): (u8, &[u8]) = match token {
            VocabToken::Text(bytes) => (0, bytes),
            VocabToken::Special(name) => (1, name.as_bytes()),
        };
        hasher.update(&[tag]);
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    hasher.update(&eos.to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> VocabToken {
        VocabToken::Text(s.as_bytes().to_vec())
    }

    #[test]
    fn a_vocabulary_checks_its_end_token_and_marker_bytes() {
        assert_eq!(
            Vocabulary::new(vec![], 0).unwrap_err(),
            VocabularyError::Empty
        );
        assert_eq!(
            Vocabulary::new(vec![text("a")], 1).unwrap_err(),
            VocabularyError::EosOutOfRange { eos: 1, len: 1 }
        );
        let marker = VocabToken::Text(vec![0xFF, b'x']);
        assert_eq!(
            Vocabulary::new(vec![text("a"), marker], 0).unwrap_err(),
            VocabularyError::MarkerByte(1)
        );
        // A single 0xFF byte is a byte-fallback token, not a marker.
        let byte = VocabToken::Text(vec![0xFF]);
        let vocab = Vocabulary::new(vec![byte, VocabToken::Special("end".into())], 1).unwrap();
        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab.eos(), 1);
        assert!(!vocab.is_empty());
        assert_eq!(vocab.token(1), Some(&VocabToken::Special("end".into())));
        assert_eq!(vocab.token(2), None);
    }

    #[test]
    fn the_digest_follows_every_part_of_the_vocabulary() {
        let base = || vec![text("ab"), text("c"), VocabToken::Special("end".into())];
        let one = Vocabulary::new(base(), 2).unwrap();
        assert_eq!(one.digest(), Vocabulary::new(base(), 2).unwrap().digest());

        let mut moved = base();
        moved.swap(0, 1);
        let split = vec![text("a"), text("bc"), VocabToken::Special("end".into())];
        let special = vec![
            text("ab"),
            VocabToken::Special("c".into()),
            VocabToken::Special("end".into()),
        ];
        for other in [
            Vocabulary::new(moved, 2).unwrap(),
            Vocabulary::new(split, 2).unwrap(),
            Vocabulary::new(special, 2).unwrap(),
            Vocabulary::new(base(), 1).unwrap(),
        ] {
            assert_ne!(one.digest(), other.digest());
        }
    }
}
