// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cryptographic helpers (RFC-0034 §3, DIST-E2-T04's acceptance criteria).
//!
//! - Golden vectors for every context string: `tests/data/crypto-contexts.txt`.
//! - A signature valid in one context never verifies in another: property tests below.
//! - RFC 8032 test vectors pass: the unit tests in `src/crypto.rs`, which reach the signing
//!   primitive without a context.
//!
//! The strict-verification rules each have a negative vector here.

use proptest::prelude::*;
use syntropy_net::crypto::{
    self, Context, HASH_LEN, PUBLIC_KEY_LEN, SEED_LEN, SIGNATURE_LEN, SignatureError, SigningKey,
};

const GOLDEN: &str = include_str!("data/crypto-contexts.txt");
const GOLDEN_MESSAGE: &[u8] = b"syntropy golden vector";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn golden_key() -> SigningKey {
    SigningKey::from_seed(&std::array::from_fn(|i| i as u8))
}

#[test]
fn every_context_string_has_golden_vectors() {
    let key = golden_key();
    let keyed: [u8; HASH_LEN] = std::array::from_fn(|i| 0x20 + i as u8);
    let mut lines = GOLDEN.lines().filter(|line| !line.starts_with('#'));
    assert_eq!(
        lines.next(),
        Some(format!("public {}", hex(&key.public_key())).as_str())
    );
    let mut contexts = Vec::new();
    for line in lines {
        let fields: Vec<&str> = line.split(' ').collect();
        let [name, signature, hash, keyed_hash, derived] = fields[..] else {
            panic!("malformed line: {line}");
        };
        let context = Context::ALL
            .into_iter()
            .find(|context| context.as_str() == name)
            .unwrap_or_else(|| panic!("{name} is not in the registry"));
        assert_eq!(
            hex(&key.sign(context, GOLDEN_MESSAGE)),
            signature,
            "{name}: signature"
        );
        assert_eq!(
            hex(&crypto::hash(context, GOLDEN_MESSAGE)),
            hash,
            "{name}: hash"
        );
        assert_eq!(
            hex(&crypto::keyed_hash(&keyed, context, GOLDEN_MESSAGE)),
            keyed_hash,
            "{name}: keyed hash"
        );
        assert_eq!(
            hex(&crypto::derive_key(context, GOLDEN_MESSAGE)),
            derived,
            "{name}: derived key"
        );
        contexts.push(context);
    }
    // The file lists the registry exactly, in RFC-0034 §3's order: a context added to the
    // registry needs its golden line.
    assert_eq!(contexts, Context::ALL);
}

#[test]
fn the_registry_holds_rfc_0034_section_3s_strings() {
    let strings: Vec<&str> = Context::ALL.iter().map(|c| c.as_str()).collect();
    assert_eq!(
        strings,
        [
            "syntropy-beacon-v1",
            "syntropy-beacon-blind-v1",
            "syntropy-beacon-mac-v1",
            "syntropy-netkey-v1",
            "syntropy-cert-v1",
            "syntropy-dpcred-v1",
            "syntropy-sas-v1",
            "syntropy-sas-commit-v1",
            "syntropy-invite-id-v1",
            "syntropy-invite-verifier-v1",
            "syntropy-recovery-v1",
            "syntropy-ns-v1",
            "syntropy-term-tag",
            "syntropy-keyrotate-v1",
            "syntropy-release-v1",
            "syntropy-module-v1",
            "syntropy-rdma-aead-v1",
            "syntropy-fp-v1",
            "syntropy-session-wrap-v1",
            "syntropy-admit-v1",
            "syntropy-revoke-v1",
            "syntropy-authority-v1",
            "syntropy-apikey-v1",
        ]
    );
}

/// ℓ, the order of the base point, little-endian.
const ELL: [u8; 32] = [
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10,
];

/// `a + b` over 256-bit little-endian integers; the tests only add where it cannot overflow.
fn add_le(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut sum = [0u8; 32];
    let mut carry = 0u16;
    for i in 0..32 {
        let total = u16::from(a[i]) + u16::from(b[i]) + carry;
        sum[i] = total as u8;
        carry = total >> 8;
    }
    assert_eq!(carry, 0);
    sum
}

#[test]
fn a_signature_with_s_not_reduced_is_refused() {
    let key = golden_key();
    let signature = key.sign(Context::Cert, b"certificate bytes");
    let s: [u8; 32] = signature[32..].try_into().unwrap();
    // S + ℓ satisfies the verification equation as well as S does; only the range check
    // refuses it (RFC 8032 §5.1.7 step 1).
    let mut malleated = signature;
    malleated[32..].copy_from_slice(&add_le(&s, &ELL));
    assert_ne!(malleated, signature);
    assert_eq!(
        crypto::verify(
            &key.public_key(),
            Context::Cert,
            b"certificate bytes",
            &malleated
        ),
        Err(SignatureError::Invalid)
    );
}

/// The identity point, `y = 1`: a point of small order.
const IDENTITY: [u8; 32] = {
    let mut bytes = [0u8; 32];
    bytes[0] = 1;
    bytes
};

#[test]
fn a_small_order_key_is_refused_even_for_a_signature_that_would_otherwise_verify() {
    // With A and R both the identity and S = 0, `[S]B = R + [k]A` holds for every message: the
    // classic forgery against a small-order key.
    let mut forged = [0u8; SIGNATURE_LEN];
    forged[..32].copy_from_slice(&IDENTITY);
    let lenient = ed25519_dalek::VerifyingKey::from_bytes(&IDENTITY).unwrap();
    let message = b"any message at all";
    let prefixed = [Context::Admit.as_str().as_bytes(), message].concat();
    assert!(
        ed25519_dalek::Verifier::verify(
            &lenient,
            &prefixed,
            &ed25519_dalek::Signature::from_bytes(&forged)
        )
        .is_ok(),
        "the forgery verifies without the small-order check"
    );
    assert_eq!(
        crypto::verify(&IDENTITY, Context::Admit, message, &forged),
        Err(SignatureError::SmallOrderKey)
    );
    // Other small-order points: y = -1, and y = 0 with either sign of x.
    let mut minus_one = [0xff; 32];
    minus_one[0] = 0xec;
    minus_one[31] = 0x7f;
    let zero = [0u8; 32];
    let mut zero_negative = [0u8; 32];
    zero_negative[31] = 0x80;
    for key in [minus_one, zero, zero_negative] {
        assert_eq!(
            crypto::verify(&key, Context::Admit, message, &forged),
            Err(SignatureError::SmallOrderKey),
            "{}",
            hex(&key)
        );
    }
}

/// A signature by the golden key over `syntropy-cert-v1 ‖ "small-order R"` whose `R` is the
/// identity and whose `S` is `k·a mod ℓ`, with `a` the key's secret scalar and `k` the RFC 8032
/// challenge. It satisfies the cofactorless equation, so a verifier without the small-order
/// check on `R` accepts it (OpenSSL 3.6.4 does). Computed with Python's integers from
/// RFC 8032 §5.1.5–§5.1.7.
const SMALL_ORDER_R: &str = concat!(
    "0100000000000000000000000000000000000000000000000000000000000000",
    "4c9e9f7085cfeacd9ac3fb991fd3f4197e98ea7bd15e571b1a98ea77b3001d0d",
);

#[test]
fn a_signature_with_a_small_order_r_is_refused() {
    let key = golden_key();
    let signature: [u8; SIGNATURE_LEN] = (0..SMALL_ORDER_R.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&SMALL_ORDER_R[i..i + 2], 16).unwrap())
        .collect::<Vec<u8>>()
        .try_into()
        .unwrap();
    let lenient = ed25519_dalek::VerifyingKey::from_bytes(&key.public_key()).unwrap();
    let prefixed = [Context::Cert.as_str().as_bytes(), b"small-order R"].concat();
    assert!(
        ed25519_dalek::Verifier::verify(
            &lenient,
            &prefixed,
            &ed25519_dalek::Signature::from_bytes(&signature)
        )
        .is_ok(),
        "the signature verifies without the small-order check on R"
    );
    assert_eq!(
        crypto::verify(
            &key.public_key(),
            Context::Cert,
            b"small-order R",
            &signature
        ),
        Err(SignatureError::Invalid)
    );
}

#[test]
fn a_non_canonical_key_encoding_is_refused() {
    // A point whose y is below 19 also has the encoding y + p, which is below 2^255. Find one
    // that is a curve point of large order.
    let (y, canonical) = (2u8..19)
        .find_map(|y| {
            let mut bytes = [0u8; 32];
            bytes[0] = y;
            let key = ed25519_dalek::VerifyingKey::from_bytes(&bytes).ok()?;
            (!key.is_weak()).then_some((y, key))
        })
        .expect("a curve point with y below 19");
    let mut non_canonical = [0xff; PUBLIC_KEY_LEN];
    non_canonical[0] = 0xed + y;
    non_canonical[31] = 0x7f;
    // The library decodes the second encoding to the same point.
    let decoded = ed25519_dalek::VerifyingKey::from_bytes(&non_canonical).unwrap();
    assert_eq!(decoded.to_edwards(), canonical.to_edwards());
    let signature = [0u8; SIGNATURE_LEN];
    assert_eq!(
        crypto::verify(&non_canonical, Context::Cert, b"m", &signature),
        Err(SignatureError::NonCanonicalKey)
    );
    // The canonical encoding gets past the key checks, to the signature.
    assert_eq!(
        crypto::verify(&canonical.to_bytes(), Context::Cert, b"m", &signature),
        Err(SignatureError::Invalid)
    );
    // y = 1 with the sign bit set names x = -0: the identity again, not canonically encoded.
    let mut negative_zero_x = IDENTITY;
    negative_zero_x[31] = 0x80;
    assert!(crypto::verify(&negative_zero_x, Context::Cert, b"m", &signature).is_err());
}

#[test]
fn a_key_that_is_not_a_curve_point_is_refused() {
    // y = 2 is not on the curve if decoding fails; search for a y that is not, so the test does
    // not depend on which small values happen to be points.
    let off_curve = (2u8..=255)
        .map(|y| {
            let mut bytes = [0u8; 32];
            bytes[0] = y;
            bytes
        })
        .find(|bytes| ed25519_dalek::VerifyingKey::from_bytes(bytes).is_err())
        .expect("a y with no curve point");
    assert_eq!(
        crypto::verify(&off_curve, Context::Cert, b"m", &[0u8; SIGNATURE_LEN]),
        Err(SignatureError::MalformedKey)
    );
}

#[test]
fn constant_time_comparison_compares_bytes_and_lengths() {
    assert!(crypto::ct_eq(b"", b""));
    assert!(crypto::ct_eq(b"verifier", b"verifier"));
    assert!(!crypto::ct_eq(b"verifier", b"Verifier"));
    assert!(!crypto::ct_eq(b"verifier", b"verifiex"));
    assert!(!crypto::ct_eq(b"verifier", b"verifie"));
    assert!(!crypto::ct_eq(b"", b"v"));
}

fn arb_context() -> impl Strategy<Value = Context> {
    (0..Context::ALL.len()).prop_map(|i| Context::ALL[i])
}

fn arb_seed() -> impl Strategy<Value = [u8; SEED_LEN]> {
    prop::array::uniform32(any::<u8>())
}

proptest! {
    /// A signature valid in one context never verifies in another, over the same message or any
    /// other one.
    #[test]
    fn a_signature_valid_in_one_context_never_verifies_in_another(
        seed in arb_seed(),
        signed in arb_context(),
        other in arb_context(),
        message in prop::collection::vec(any::<u8>(), 0..256),
        other_message in prop::collection::vec(any::<u8>(), 0..256),
    ) {
        let key = SigningKey::from_seed(&seed);
        let public = key.public_key();
        let signature = key.sign(signed, &message);
        prop_assert_eq!(crypto::verify(&public, signed, &message, &signature), Ok(()));
        if other != signed {
            prop_assert_eq!(
                crypto::verify(&public, other, &message, &signature),
                Err(SignatureError::Invalid)
            );
            prop_assert_eq!(
                crypto::verify(&public, other, &other_message, &signature),
                Err(SignatureError::Invalid)
            );
            // Moving the first context's string into the message does not help either.
            let shifted = [signed.as_str().as_bytes(), &message].concat();
            prop_assert!(crypto::verify(&public, other, &shifted, &signature).is_err());
        }
    }

    /// The same holds for each BLAKE3 mode.
    #[test]
    fn a_hash_in_one_context_never_equals_one_in_another(
        key in prop::array::uniform32(any::<u8>()),
        first in arb_context(),
        second in arb_context(),
        message in prop::collection::vec(any::<u8>(), 0..256),
    ) {
        prop_assume!(first != second);
        prop_assert_ne!(crypto::hash(first, &message), crypto::hash(second, &message));
        prop_assert_ne!(
            crypto::keyed_hash(&key, first, &message),
            crypto::keyed_hash(&key, second, &message)
        );
        prop_assert_ne!(crypto::derive_key(first, &message), crypto::derive_key(second, &message));
    }

    /// A changed bit in the message or the signature fails verification.
    #[test]
    fn a_changed_bit_fails_verification(
        seed in arb_seed(),
        context in arb_context(),
        message in prop::collection::vec(any::<u8>(), 1..256),
        bit in any::<prop::sample::Index>(),
        in_signature in any::<bool>(),
    ) {
        let key = SigningKey::from_seed(&seed);
        let mut signature = key.sign(context, &message);
        let mut message = message;
        if in_signature {
            let at = bit.index(SIGNATURE_LEN * 8);
            signature[at / 8] ^= 1 << (at % 8);
        } else {
            let at = bit.index(message.len() * 8);
            message[at / 8] ^= 1 << (at % 8);
        }
        prop_assert!(crypto::verify(&key.public_key(), context, &message, &signature).is_err());
    }
}
