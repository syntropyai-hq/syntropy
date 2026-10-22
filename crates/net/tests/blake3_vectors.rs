// SPDX-License-Identifier: MIT OR Apache-2.0

//! The official BLAKE3 test vectors (`third_party/blake3/test_vectors.json`) against the
//! `blake3` crate as `syntropy-net` builds it, with the `pure` backend (RFC-0034 §3).
//!
//! Each case is an input length, with the input filled with the repeating bytes 0, 1, …, 250,
//! and an extended output for each of the hash, keyed-hash and derive-key modes. The default
//! 32-byte output must be the extended output's first 32 bytes.

use serde_json::Value;

const VECTORS: &str = include_str!("../../../third_party/blake3/test_vectors.json");

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

/// The `len`-byte extended output of `hasher`.
fn extended(hasher: &blake3::Hasher, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    hasher.finalize_xof().fill(&mut out);
    out
}

#[test]
fn the_official_blake3_test_vectors_pass() {
    let vectors: Value = serde_json::from_str(VECTORS).unwrap();
    let key: [u8; 32] = vectors["key"]
        .as_str()
        .unwrap()
        .as_bytes()
        .try_into()
        .unwrap();
    let context = vectors["context_string"].as_str().unwrap();
    let cases = vectors["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 35);
    for case in cases {
        let len = usize::try_from(case["input_len"].as_u64().unwrap()).unwrap();
        let input: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let expected = |mode: &str| unhex(case[mode].as_str().unwrap());

        let hash = expected("hash");
        let mut hasher = blake3::Hasher::new();
        hasher.update(&input);
        assert_eq!(extended(&hasher, hash.len()), hash, "hash, {len} bytes");
        assert_eq!(
            blake3::hash(&input).as_bytes()[..],
            hash[..32],
            "hash, {len} bytes"
        );

        let keyed = expected("keyed_hash");
        let mut hasher = blake3::Hasher::new_keyed(&key);
        hasher.update(&input);
        assert_eq!(extended(&hasher, keyed.len()), keyed, "keyed, {len} bytes");
        assert_eq!(
            blake3::keyed_hash(&key, &input).as_bytes()[..],
            keyed[..32],
            "keyed, {len} bytes"
        );

        let derived = expected("derive_key");
        let mut hasher = blake3::Hasher::new_derive_key(context);
        hasher.update(&input);
        assert_eq!(
            extended(&hasher, derived.len()),
            derived,
            "derive, {len} bytes"
        );
        assert_eq!(
            blake3::derive_key(context, &input)[..],
            derived[..32],
            "derive, {len} bytes"
        );
    }
}
