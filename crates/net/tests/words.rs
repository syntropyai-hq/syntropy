// SPDX-License-Identifier: MIT OR Apache-2.0

//! The frozen word list (RFC-0034 §6.5, DIST-E2-T08's acceptance criteria).
//!
//! - The asset's licence is on ADR-0018's allow list: `cargo xtask third-party` checks the
//!   manifest entry in `lint`.
//! - The review result is recorded with any word replaced: [`REPLACED`] and RFC-0034 §6.5.
//! - The list has 2,048 distinct entries.
//! - A golden fingerprint renders in the `word-word-…` form.

use std::collections::HashSet;

use syntropy_net::words::{self, REPLACED, WORD_COUNT, WORDLIST_VERSION};

const UPSTREAM: &str = include_str!("../../../third_party/bip39/english.txt");

#[test]
fn the_list_has_2048_distinct_entries_in_bip_39_form() {
    let words = words::words();
    assert_eq!(words.len(), WORD_COUNT);
    let distinct: HashSet<&str> = words.iter().copied().collect();
    assert_eq!(distinct.len(), 2048);
    for word in words {
        assert!(
            (3..=8).contains(&word.len()) && word.bytes().all(|b| b.is_ascii_lowercase()),
            "{word}"
        );
    }
    // BIP-39's conventions hold after the replacements: the list is sorted, and the first four
    // letters name each word.
    assert!(words.windows(2).all(|pair| pair[0] < pair[1]));
    let prefixes: HashSet<&str> = words.iter().map(|w| &w[..w.len().min(4)]).collect();
    assert_eq!(prefixes.len(), 2048);
}

#[test]
fn the_list_is_upstream_with_only_the_reviewed_replacements() {
    let upstream: Vec<&str> = UPSTREAM.lines().collect();
    assert_eq!(upstream.len(), WORD_COUNT);
    for (index, (&ours, &theirs)) in words::words().iter().zip(&upstream).enumerate() {
        match REPLACED.iter().find(|&&(at, _, _)| at == index) {
            Some(&(_, old, new)) => {
                assert_eq!(theirs, old, "upstream word {index}");
                assert_eq!(ours, new, "word {index}");
                assert!(!upstream.contains(&new), "{new} is already a BIP-39 word");
            }
            None => assert_eq!(ours, theirs, "word {index} is not a reviewed replacement"),
        }
    }
    // Line 259 of the vendored file already differs from upstream, for the naming rule.
    assert_eq!(upstream[258], "cairn");
    assert_eq!(words::word(258), Some("cairn"));
    let replaced: Vec<(&str, &str)> = REPLACED.iter().map(|&(_, old, new)| (old, new)).collect();
    assert_eq!(
        replaced,
        [
            ("addict", "addition"),
            ("crazy", "crayon"),
            ("dumb", "duet"),
            ("insane", "inscribe"),
            ("liar", "lexicon"),
            ("ugly", "ukulele"),
        ]
    );
}

/// FNV-1a over 64 bits: enough to notice any change to the list, which is all this test asks.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn version_1_of_the_list_is_frozen() {
    // The digest of the words joined by '\n'. A change to any word fails here and needs a new
    // `WORDLIST_VERSION`: fingerprints and recovery codes already shown would read differently.
    assert_eq!(WORDLIST_VERSION, 1);
    let joined = words::words().join("\n");
    assert_eq!(fnv1a64(joined.as_bytes()), 0x8852_4cbf_d611_91a6);
}

/// A fixed 66-bit value: the 11-bit indices 0, 543, 1888, 2047, 1030 and 26, most significant
/// bit first, in 9 bytes whose last 6 bits are unused.
const GOLDEN_BITS: [u8; 9] = [0x00, 0x08, 0x7f, 0xb0, 0x7f, 0xf8, 0x0c, 0x06, 0x80];

#[test]
fn a_golden_fingerprint_renders_as_word_word() {
    let rendered = words::render(&GOLDEN_BITS, 6).unwrap();
    assert_eq!(rendered, "abandon-duet-ukulele-zoo-lexicon-addition");
    // Words joined by hyphens, not a numbered list.
    assert!(
        rendered
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b == b'-')
    );
    assert_eq!(rendered.split('-').count(), 6);
    // Only the first 66 bits count.
    let mut tail = GOLDEN_BITS;
    tail[8] |= 0x3f;
    assert_eq!(words::render(&tail, 6).unwrap(), rendered);
}

#[test]
fn rendering_needs_enough_bits() {
    assert_eq!(words::render(&GOLDEN_BITS[..8], 6), None);
    assert_eq!(words::render(&[], 0).as_deref(), Some(""));
    // A recovery code's shape: twelve words from 132 bits.
    let code = words::render(&[0xa5; 17], 12).unwrap();
    assert_eq!(code.split('-').count(), 12);
    assert_eq!(words::render(&[0xa5; 16], 12), None);
}

#[test]
fn words_are_found_by_index() {
    assert_eq!(words::word(0), Some("abandon"));
    assert_eq!(words::word(2047), Some("zoo"));
    assert_eq!(words::word(2048), None);
    assert_eq!(words::word(1888), Some("ukulele"));
}
