// SPDX-License-Identifier: MIT OR Apache-2.0

//! Identity fingerprints (RFC-0034 §6.5, DIST-E2-T07's acceptance criteria).
//!
//! - Stable golden renderings for test keys. The expected values were computed independently,
//!   with the `blake3` Python binding (checked against the official BLAKE3 vectors) and the
//!   vendored word list with the reviewed replacements applied.
//! - Distinct keys give distinct renderings over 10⁶ random keys.

use std::collections::HashSet;

use syntropy_net::fingerprint::{
    FINGERPRINT_BITS, FINGERPRINT_BYTES, FINGERPRINT_WORDS, fingerprint, fingerprint_bits,
};

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

/// (key, the first 66 bits of `BLAKE3("syntropy-fp-v1" ‖ key)`, the rendering).
const GOLDEN: [(&str, &str, &str); 4] = [
    // The public key of the Ed25519 seed 00..1f (the crypto goldens' key).
    (
        "03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8",
        "3b994f4d38fe2e2c00",
        "describe-skill-spy-impose-title-bid",
    ),
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "11ae34fa1efb092c40",
        "balance-immune-direct-digital-rack-normal",
    ),
    (
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        "04405c496fd0b6f040",
        "affair-actual-banner-text-arena-join",
    ),
    (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "62dccb4b981ce14d40",
        "glare-town-spray-coral-sock-pluck",
    ),
];

#[test]
fn golden_fingerprints_are_stable() {
    assert_eq!(
        (FINGERPRINT_WORDS, FINGERPRINT_BITS, FINGERPRINT_BYTES),
        (6, 66, 9)
    );
    for (key, bits, rendering) in GOLDEN {
        let key = unhex(key);
        assert_eq!(fingerprint_bits(&key).to_vec(), unhex(bits), "{rendering}");
        assert_eq!(fingerprint(&key), rendering);
    }
}

#[test]
fn a_fingerprint_is_six_hyphenated_words() {
    let shown = fingerprint(b"any key");
    assert_eq!(shown.split('-').count(), FINGERPRINT_WORDS);
    assert!(
        shown
            .split('-')
            .all(|word| !word.is_empty() && word.bytes().all(|b| b.is_ascii_lowercase()))
    );
    // Only the first 66 bits count: the last 6 bits of the last byte are always zero.
    assert_eq!(
        fingerprint_bits(b"any key")[FINGERPRINT_BYTES - 1] & 0x3f,
        0
    );
}

/// SplitMix64: a seeded source of test keys.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn key(&mut self) -> [u8; 32] {
        let mut key = [0u8; 32];
        for word in 0..4 {
            key[word * 8..(word + 1) * 8].copy_from_slice(&self.next().to_le_bytes());
        }
        key
    }
}

#[test]
fn distinct_keys_give_distinct_renderings_over_a_million_keys() {
    const KEYS: usize = 1_000_000;
    let mut source = SplitMix(0x5eed_0ff1_9e12);
    let mut keys = HashSet::with_capacity(KEYS);
    let mut renderings = HashSet::with_capacity(KEYS);
    while keys.len() < KEYS {
        let key = source.key();
        if keys.insert(key) {
            assert!(
                renderings.insert(fingerprint(&key)),
                "a second key renders the same"
            );
        }
    }
    assert_eq!(renderings.len(), KEYS);
}
