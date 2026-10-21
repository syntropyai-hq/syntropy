// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cryptographic helpers (RFC-0034 §3). Owner: RFC-0034.
//!
//! - **Ed25519 (RFC 8032) with strict verification.** [`verify`] refuses a public key that is not
//!   canonically encoded or is of small order, a signature whose `S` is not reduced (`S ≥ ℓ`) or
//!   whose `R` is of small order or not canonically encoded, and then checks the cofactorless
//!   equation `[S]B = R + [k]A`.
//! - **Domain separation.** Every signature, hash and derivation names a [`Context`] from the
//!   registry of RFC-0034 §3. The context string comes first in what is signed or hashed:
//!   [`SigningKey::sign`] signs `ctx ‖ message`, and [`hash`] is `BLAKE3(ctx ‖ message)`. No
//!   context string is a prefix of another, so `ctx_a ‖ m` and `ctx_b ‖ m'` differ for every `m`
//!   and `m'`, and a signature or hash valid in one context is never valid in another.
//! - **BLAKE3** in its three modes: [`hash`], [`keyed_hash`] and [`derive_key`]. A caller that
//!   needs fewer bytes, such as `BLAKE3(…)[0..16]`, truncates the result.
//! - **Constant-time comparison** of secrets, such as an invite verifier: [`ct_eq`].
//!
//! A verification failure is a [`SignatureError`]. Its error code and class depend on what was
//! signed (RFC-0034 §15: `Auth` for a certificate or an authority event, `Integrity` for a module
//! or an update), so the caller maps it.

use std::fmt;

use ed25519_dalek::Signer;
use subtle::ConstantTimeEq;

/// The bytes of an Ed25519 public key.
pub const PUBLIC_KEY_LEN: usize = 32;

/// The bytes of an Ed25519 secret seed.
pub const SEED_LEN: usize = 32;

/// The bytes of an Ed25519 signature.
pub const SIGNATURE_LEN: usize = 64;

/// The bytes of a BLAKE3 output, and of a BLAKE3 key.
pub const HASH_LEN: usize = 32;

/// A domain-separation context string (RFC-0034 §3), with the section that uses it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Context {
    /// `syntropy-beacon-v1`: a beacon's signature (RFC-0030 §4.2).
    Beacon,
    /// `syntropy-beacon-blind-v1`: a blinded beacon's tag (RFC-0030 §4.8).
    BeaconBlind,
    /// `syntropy-beacon-mac-v1`: a blinded beacon's MAC (RFC-0030 §4.8).
    BeaconMac,
    /// `syntropy-netkey-v1`: a network key (RFC-0030 §4.8).
    NetKey,
    /// `syntropy-cert-v1`: an admission certificate's signature (RFC-0034 §5.1).
    Cert,
    /// `syntropy-dpcred-v1`: a data-plane credential's signature (RFC-0034 §6.3).
    DpCred,
    /// `syntropy-sas-v1`: the pairing code (RFC-0030 §6.2).
    Sas,
    /// `syntropy-sas-commit-v1`: the pairing commitment (RFC-0030 §6.2).
    SasCommit,
    /// `syntropy-invite-id-v1`: an invite token's ID (RFC-0030 §6.4).
    InviteId,
    /// `syntropy-invite-verifier-v1`: an invite token's verifier (RFC-0030 §6.4).
    InviteVerifier,
    /// `syntropy-recovery-v1`: the recovery key's derivation (RFC-0034 §5.5).
    Recovery,
    /// `syntropy-ns-v1`: a namespace hash (RFC-0030 §4.1).
    Namespace,
    /// `syntropy-term-tag`: a coordinator claimant's term tag (RFC-0032 §7.2).
    TermTag,
    /// `syntropy-keyrotate-v1`: a cluster key rotation statement (RFC-0034 §8.2, §5.5).
    KeyRotate,
    /// `syntropy-release-v1`: a release statement (RFC-0034 §12.1).
    Release,
    /// `syntropy-module-v1`: backend modules (RFC-0034 §3).
    Module,
    /// `syntropy-rdma-aead-v1`: the TLS exporter label of RDMA frame keys (RFC-0034 §9.3).
    RdmaAead,
    /// `syntropy-fp-v1`: an identity fingerprint (RFC-0034 §6.5).
    Fingerprint,
    /// `syntropy-session-wrap-v1`: saved-session key wrapping (RFC-0034 §3, §13).
    SessionWrap,
    /// `syntropy-admit-v1`: an admission approval (RFC-0034 §8.3).
    Admit,
    /// `syntropy-revoke-v1`: a revocation (RFC-0034 §8.1).
    Revoke,
    /// `syntropy-authority-v1`: the other authority events (RFC-0034 §3, §8.3).
    Authority,
    /// `syntropy-apikey-v1`: cluster-wide API keys (RFC-0034 §3).
    ApiKey,
}

impl Context {
    /// Every context, in RFC-0034 §3's order.
    pub const ALL: [Context; 23] = [
        Context::Beacon,
        Context::BeaconBlind,
        Context::BeaconMac,
        Context::NetKey,
        Context::Cert,
        Context::DpCred,
        Context::Sas,
        Context::SasCommit,
        Context::InviteId,
        Context::InviteVerifier,
        Context::Recovery,
        Context::Namespace,
        Context::TermTag,
        Context::KeyRotate,
        Context::Release,
        Context::Module,
        Context::RdmaAead,
        Context::Fingerprint,
        Context::SessionWrap,
        Context::Admit,
        Context::Revoke,
        Context::Authority,
        Context::ApiKey,
    ];

    /// The context string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Context::Beacon => "syntropy-beacon-v1",
            Context::BeaconBlind => "syntropy-beacon-blind-v1",
            Context::BeaconMac => "syntropy-beacon-mac-v1",
            Context::NetKey => "syntropy-netkey-v1",
            Context::Cert => "syntropy-cert-v1",
            Context::DpCred => "syntropy-dpcred-v1",
            Context::Sas => "syntropy-sas-v1",
            Context::SasCommit => "syntropy-sas-commit-v1",
            Context::InviteId => "syntropy-invite-id-v1",
            Context::InviteVerifier => "syntropy-invite-verifier-v1",
            Context::Recovery => "syntropy-recovery-v1",
            Context::Namespace => "syntropy-ns-v1",
            Context::TermTag => "syntropy-term-tag",
            Context::KeyRotate => "syntropy-keyrotate-v1",
            Context::Release => "syntropy-release-v1",
            Context::Module => "syntropy-module-v1",
            Context::RdmaAead => "syntropy-rdma-aead-v1",
            Context::Fingerprint => "syntropy-fp-v1",
            Context::SessionWrap => "syntropy-session-wrap-v1",
            Context::Admit => "syntropy-admit-v1",
            Context::Revoke => "syntropy-revoke-v1",
            Context::Authority => "syntropy-authority-v1",
            Context::ApiKey => "syntropy-apikey-v1",
        }
    }

    /// `ctx ‖ message`: the bytes a signature or a plain hash covers.
    fn prefixed(self, message: &[u8]) -> Vec<u8> {
        let context = self.as_str().as_bytes();
        let mut bytes = Vec::with_capacity(context.len() + message.len());
        bytes.extend_from_slice(context);
        bytes.extend_from_slice(message);
        bytes
    }
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An Ed25519 signing key. The secret is zeroed when the key is dropped, and `Debug` shows only
/// the public key.
pub struct SigningKey(ed25519_dalek::SigningKey);

impl SigningKey {
    /// The key whose RFC 8032 secret is `seed`.
    pub fn from_seed(seed: &[u8; SEED_LEN]) -> SigningKey {
        SigningKey(ed25519_dalek::SigningKey::from_bytes(seed))
    }

    /// The public key, as RFC 8032 encodes it.
    pub fn public_key(&self) -> [u8; PUBLIC_KEY_LEN] {
        self.0.verifying_key().to_bytes()
    }

    /// The signature over `context ‖ message`.
    pub fn sign(&self, context: Context, message: &[u8]) -> [u8; SIGNATURE_LEN] {
        sign_raw(&self.0, &context.prefixed(message))
    }
}

impl fmt::Debug for SigningKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SigningKey")
            .field("public_key", &Hex(&self.public_key()))
            .finish_non_exhaustive()
    }
}

/// Why a signature did not verify. The caller maps it to an error code (RFC-0034 §15).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignatureError {
    /// The public key does not decode to a curve point.
    MalformedKey,
    /// The public key decodes, but is not the canonical encoding of its point.
    NonCanonicalKey,
    /// The public key is a point of small order, which can verify forged signatures.
    SmallOrderKey,
    /// The signature is malformed (`S ≥ ℓ`, or `R` not a canonical point of large order) or does
    /// not verify.
    Invalid,
}

impl fmt::Display for SignatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SignatureError::MalformedKey => "the public key is not a curve point",
            SignatureError::NonCanonicalKey => "the public key is not canonically encoded",
            SignatureError::SmallOrderKey => "the public key is of small order",
            SignatureError::Invalid => "the signature does not verify",
        })
    }
}

impl std::error::Error for SignatureError {}

/// Checks `signature` over `context ‖ message` under `public_key`, strictly (module docs).
pub fn verify(
    public_key: &[u8; PUBLIC_KEY_LEN],
    context: Context,
    message: &[u8],
    signature: &[u8; SIGNATURE_LEN],
) -> Result<(), SignatureError> {
    verify_raw(public_key, &context.prefixed(message), signature)
}

/// BLAKE3 of `context ‖ message`.
pub fn hash(context: Context, message: &[u8]) -> [u8; HASH_LEN] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(context.as_str().as_bytes());
    hasher.update(message);
    hasher.finalize().into()
}

/// BLAKE3 in keyed mode, under `key`, of `context ‖ message`.
pub fn keyed_hash(key: &[u8; HASH_LEN], context: Context, message: &[u8]) -> [u8; HASH_LEN] {
    let mut hasher = blake3::Hasher::new_keyed(key);
    hasher.update(context.as_str().as_bytes());
    hasher.update(message);
    hasher.finalize().into()
}

/// BLAKE3 in derive-key mode, with `context` as the context string and `key_material` as the
/// input, such as `derive-key("syntropy-recovery-v1", seed)` (RFC-0034 §5.5).
pub fn derive_key(context: Context, key_material: &[u8]) -> [u8; HASH_LEN] {
    blake3::derive_key(context.as_str(), key_material)
}

/// Whether `a` and `b` hold the same bytes, in time that does not depend on where they differ.
/// Slices of different lengths compare unequal; their lengths are not treated as secret.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.ct_eq(b).into()
}

/// The RFC 8032 signature over `message`, with no context.
fn sign_raw(key: &ed25519_dalek::SigningKey, message: &[u8]) -> [u8; SIGNATURE_LEN] {
    key.sign(message).to_bytes()
}

/// Strict RFC 8032 verification of `signature` over `message`, with no context.
fn verify_raw(
    public_key: &[u8; PUBLIC_KEY_LEN],
    message: &[u8],
    signature: &[u8; SIGNATURE_LEN],
) -> Result<(), SignatureError> {
    let key = ed25519_dalek::VerifyingKey::from_bytes(public_key)
        .map_err(|_| SignatureError::MalformedKey)?;
    // `from_bytes` decodes under ZIP-215 rules, which accept a non-canonical encoding of a
    // point. A key must re-encode to the bytes it came from.
    if key.to_edwards().compress().as_bytes() != public_key {
        return Err(SignatureError::NonCanonicalKey);
    }
    if key.is_weak() {
        return Err(SignatureError::SmallOrderKey);
    }
    // `verify_strict` refuses `S ≥ ℓ` and an `R` of small order, and compares the recomputed `R`
    // with the signature's `R` byte for byte, so a non-canonical `R` fails too.
    key.verify_strict(message, &ed25519_dalek::Signature::from_bytes(signature))
        .map_err(|_| SignatureError::Invalid)
}

/// Bytes shown as lowercase hex.
struct Hex<'a>(&'a [u8]);

impl fmt::Debug for Hex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(text: &str) -> Vec<u8> {
        assert!(text.len().is_multiple_of(2), "{text}");
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
            .collect()
    }

    /// RFC 8032 §7.1's Ed25519 test vectors, extracted from the RFC's text: one line per vector,
    /// `name secret public message signature`, with `-` for an empty message.
    const RFC_8032: &str = include_str!("../tests/data/rfc8032-ed25519.txt");

    #[test]
    fn rfc_8032_test_vectors_pass() {
        let mut names = Vec::new();
        for line in RFC_8032
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let fields: Vec<&str> = line.split(' ').collect();
            let [name, secret, public, message, signature] = fields[..] else {
                panic!("malformed line: {line}");
            };
            let secret: [u8; SEED_LEN] = unhex(secret).try_into().unwrap();
            let public: [u8; PUBLIC_KEY_LEN] = unhex(public).try_into().unwrap();
            let message = if message == "-" {
                Vec::new()
            } else {
                unhex(message)
            };
            let signature: [u8; SIGNATURE_LEN] = unhex(signature).try_into().unwrap();
            let key = SigningKey::from_seed(&secret);
            assert_eq!(key.public_key(), public, "{name}: public key");
            assert_eq!(sign_raw(&key.0, &message), signature, "{name}: signature");
            assert_eq!(verify_raw(&public, &message, &signature), Ok(()), "{name}");
            let mut flipped = signature;
            flipped[0] ^= 1;
            assert!(verify_raw(&public, &message, &flipped).is_err(), "{name}");
            names.push(name.to_owned());
        }
        assert_eq!(names, ["1", "2", "3", "1024", "SHA(abc)"]);
    }

    #[test]
    fn no_context_string_is_a_prefix_of_another() {
        for a in Context::ALL {
            for b in Context::ALL {
                if a != b {
                    assert!(
                        !b.as_str().starts_with(a.as_str()),
                        "{a} is a prefix of {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_signing_key_debug_shows_only_the_public_key() {
        let key = SigningKey::from_seed(&[7; SEED_LEN]);
        let shown = format!("{key:?}");
        let public: String = key
            .public_key()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(shown, format!("SigningKey {{ public_key: {public}, .. }}"));
    }
}
