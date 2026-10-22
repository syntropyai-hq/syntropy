// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fingerprint and recovery-code word list (RFC-0034 §6.5, §5.5). Owner: RFC-0034.
//!
//! The list is the BIP-39 English list, which is MIT-licensed, frozen as version
//! [`WORDLIST_VERSION`] after a review for offensive words. The upstream file is vendored as
//! `third_party/bip39/english.txt`, with its licence beside it. It differs from upstream in one
//! line: line 259 holds `cairn` in place of a word the naming rule (AGENTS.md, D5) keeps out of
//! the repository. The review replaced six more words; [`REPLACED`] applies them here. Each
//! replacement takes the index of the word it replaces and sorts in its place, so the list stays
//! sorted and every other index keeps its word.
//!
//! An identity fingerprint is six words for the first 66 bits of a digest, and a recovery code
//! is twelve words for 132 bits; both read 11 bits per word, most significant bit first.
//! [`render`] shows them as `word-word-…`, never as a numbered list.

use std::sync::OnceLock;

/// The version of the frozen list. Changing any word is a new version.
pub const WORDLIST_VERSION: u32 = 1;

/// The number of words: one per 11-bit index.
pub const WORD_COUNT: usize = 2048;

/// The bits each word encodes.
pub const BITS_PER_WORD: usize = 11;

/// The upstream BIP-39 English list, one word per line.
const UPSTREAM: &str = include_str!("../../../third_party/bip39/english.txt");

/// The words the offensive-word review replaced (RFC-0034 §6.5, 2026-10-22): the index, the
/// BIP-39 word, and the word that takes its place. Each replaced word is mainly used to insult or
/// demean a person, and each replacement sorts between the same neighbours, so the list stays in
/// alphabetical order.
pub const REPLACED: [(usize, &str, &str); 6] = [
    (26, "addict", "addition"),
    (406, "crazy", "crayon"),
    (543, "dumb", "duet"),
    (935, "insane", "inscribe"),
    (1030, "liar", "lexicon"),
    (1888, "ugly", "ukulele"),
];

/// The frozen list, version [`WORDLIST_VERSION`]: the upstream list with [`REPLACED`] applied.
pub fn words() -> &'static [&'static str; WORD_COUNT] {
    static WORDS: OnceLock<[&'static str; WORD_COUNT]> = OnceLock::new();
    WORDS.get_or_init(|| {
        let mut words = [""; WORD_COUNT];
        let mut count = 0;
        for (slot, word) in words.iter_mut().zip(UPSTREAM.lines()) {
            *slot = word;
            count += 1;
        }
        // The asset is part of the binary, and the tests check its shape; a mismatch here means
        // the vendored file changed without a new list version.
        assert_eq!(
            count, WORD_COUNT,
            "the vendored word list has {count} words"
        );
        for (index, upstream, replacement) in REPLACED {
            assert_eq!(words[index], upstream, "word {index} of the vendored list");
            words[index] = replacement;
        }
        words
    })
}

/// The word for an 11-bit `index`, or `None` if `index` is 2048 or more.
pub fn word(index: usize) -> Option<&'static str> {
    words().get(index).copied()
}

/// `count` words for the first `count × 11` bits of `bits`, read most significant bit first, as
/// `word-word-…`. `None` if `bits` holds fewer bits than that.
pub fn render(bits: &[u8], count: usize) -> Option<String> {
    if bits.len().checked_mul(8)? < count.checked_mul(BITS_PER_WORD)? {
        return None;
    }
    let words = words();
    let rendered: Vec<&str> = (0..count)
        .map(|k| words[index_at(bits, k * BITS_PER_WORD)])
        .collect();
    Some(rendered.join("-"))
}

/// The 11-bit index that starts at bit `start` of `bits`, most significant bit first.
fn index_at(bits: &[u8], start: usize) -> usize {
    (start..start + BITS_PER_WORD).fold(0, |index, bit| {
        let set = (bits[bit / 8] >> (7 - bit % 8)) & 1;
        (index << 1) | usize::from(set)
    })
}
