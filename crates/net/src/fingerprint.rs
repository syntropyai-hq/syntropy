// SPDX-License-Identifier: MIT OR Apache-2.0

//! Six-word identity fingerprints (RFC-0034 §6.5). Owner: RFC-0034.
//!
//! A fingerprint lets a person compare a node ID, a cluster ID or a self-signed API certificate
//! out of band. It is six words from the frozen list ([`crate::words`]) for the first 66 bits of
//! `BLAKE3("syntropy-fp-v1" ‖ key)`, read 11 bits per word, most significant bit first, and shown
//! as `word-word-word-word-word-word`, never as a numbered list. 66 bits are enough to resist an
//! offline second-preimage search; the pairing SAS (RFC-0030 §6.2) is a different check.

use crate::crypto::{self, Context};
use crate::words;

/// The words in a fingerprint.
pub const FINGERPRINT_WORDS: usize = 6;

/// The digest bits a fingerprint shows: 6 words of 11 bits.
pub const FINGERPRINT_BITS: usize = FINGERPRINT_WORDS * words::BITS_PER_WORD;

/// The bytes that hold [`FINGERPRINT_BITS`] bits.
pub const FINGERPRINT_BYTES: usize = FINGERPRINT_BITS.div_ceil(8);
const _: () = assert!(FINGERPRINT_BYTES * 8 >= FINGERPRINT_BITS && FINGERPRINT_BYTES <= 32);

/// The first [`FINGERPRINT_BITS`] bits of `BLAKE3("syntropy-fp-v1" ‖ key)`, most significant bit
/// first, in [`FINGERPRINT_BYTES`] bytes whose unused low bits are zero. Two keys have the same
/// fingerprint exactly when these bytes are equal.
pub fn fingerprint_bits(key: &[u8]) -> [u8; FINGERPRINT_BYTES] {
    let digest = crypto::hash(Context::Fingerprint, key);
    let mut bits = [0u8; FINGERPRINT_BYTES];
    bits.copy_from_slice(&digest[..FINGERPRINT_BYTES]);
    let unused = FINGERPRINT_BYTES * 8 - FINGERPRINT_BITS;
    bits[FINGERPRINT_BYTES - 1] &= 0xff << unused;
    bits
}

/// `key`'s fingerprint, as `word-word-word-word-word-word`.
pub fn fingerprint(key: &[u8]) -> String {
    words::render(&fingerprint_bits(key), FINGERPRINT_WORDS)
        .expect("FINGERPRINT_BYTES holds FINGERPRINT_WORDS words")
}
