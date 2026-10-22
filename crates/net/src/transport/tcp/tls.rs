// SPDX-License-Identifier: MIT OR Apache-2.0

//! TLS 1.3 for the TCP fallback (RFC-0031 §5.3): both ends authenticate with the Ed25519 key form
//! iroh's QUIC uses, a raw public key (RFC 7250), on the ring provider (RFC-0034 §3).
//!
//! Each end presents its key as an Ed25519 `SubjectPublicKeyInfo` and proves it holds the private
//! half by signing the handshake. Neither end trusts a certificate authority: the connector accepts
//! only the key it dialled for, the peer's node or runner key from its `RemoteEndpoint`, and the
//! acceptor only the keys its owner allowed, the plan's neighbours. Only the Ed25519 signature
//! scheme is offered, so a peer cannot bargain down to a weaker one.

use std::fmt;
use std::sync::{Arc, RwLock};

use ed25519_dalek::SigningKey as DalekKey;
use rustls::client::AlwaysResolvesClientRawPublicKeys;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{
    CryptoProvider, WebPkiSupportedAlgorithms, verify_tls13_signature_with_raw_key,
};
use rustls::pki_types::{
    CertificateDer, PrivatePkcs8KeyDer, ServerName, SubjectPublicKeyInfoDer, UnixTime,
};
use rustls::server::AlwaysResolvesServerRawPublicKeys;
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::sign::CertifiedKey;
use rustls::{
    CertificateError, ClientConfig, DigitallySignedStruct, DistinguishedName, Error as TlsError,
    PeerIncompatible, ServerConfig, SignatureScheme,
};

/// The data plane's ALPN (RFC-0031 §1).
pub const DATA_ALPN: &[u8] = b"syntropy/data/1";

/// The DER of an Ed25519 `SubjectPublicKeyInfo` before its 32 key bytes (RFC 8410 §4).
const SPKI_PREFIX: [u8; 12] = [
    0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00,
];

/// The DER of an Ed25519 `OneAsymmetricKey` (PKCS#8 v1) before its 32 seed bytes (RFC 8410 §7).
const PKCS8_PREFIX: [u8; 16] = [
    0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20,
];

/// The Ed25519 `SubjectPublicKeyInfo` of `key`: what each end presents.
pub fn spki(key: &[u8; 32]) -> Vec<u8> {
    let mut der = SPKI_PREFIX.to_vec();
    der.extend_from_slice(key);
    der
}

/// The Ed25519 public key a `SubjectPublicKeyInfo` holds, if it is one.
pub fn spki_key(der: &[u8]) -> Option<[u8; 32]> {
    der.strip_prefix(&SPKI_PREFIX)?.try_into().ok()
}

/// The ring provider with TLS 1.3's cipher suites, as iroh builds it.
fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

/// The algorithms each end verifies the other's handshake signature with.
fn algorithms() -> WebPkiSupportedAlgorithms {
    rustls::crypto::ring::default_provider().signature_verification_algorithms
}

/// `seed`'s key as a TLS certified key: its raw public key and its signer.
fn certified(seed: &[u8; 32]) -> Result<Arc<CertifiedKey>, TlsError> {
    let public = DalekKey::from_bytes(seed).verifying_key().to_bytes();
    let mut pkcs8 = PKCS8_PREFIX.to_vec();
    pkcs8.extend_from_slice(seed);
    let signer = rustls::crypto::ring::sign::any_eddsa_type(&PrivatePkcs8KeyDer::from(pkcs8))?;
    Ok(Arc::new(CertifiedKey::new(
        vec![CertificateDer::from(spki(&public))],
        signer,
    )))
}

/// Checks a handshake signature against the raw key `cert` holds, Ed25519 only.
fn verify_signature(
    message: &[u8],
    cert: &CertificateDer<'_>,
    dss: &DigitallySignedStruct,
) -> Result<HandshakeSignatureValid, TlsError> {
    if dss.scheme != SignatureScheme::ED25519 {
        return Err(TlsError::PeerIncompatible(
            PeerIncompatible::NoSignatureSchemesInCommon,
        ));
    }
    verify_tls13_signature_with_raw_key(
        message,
        &SubjectPublicKeyInfoDer::from(cert.as_ref()),
        dss,
        &algorithms(),
    )
}

/// A presented key that is not an Ed25519 key, or not the one expected.
fn wrong_key() -> TlsError {
    TlsError::InvalidCertificate(CertificateError::ApplicationVerificationFailure)
}

/// The connector's check of the acceptor: its raw key must be `expected`.
#[derive(Debug)]
struct ExpectKey {
    expected: [u8; 32],
}

impl ServerCertVerifier for ExpectKey {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        match spki_key(end_entity) {
            Some(key) if key == self.expected && intermediates.is_empty() => {
                Ok(ServerCertVerified::assertion())
            }
            _ => Err(wrong_key()),
        }
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        // Only TLS 1.3 is offered, so a 1.2 handshake never reaches this.
        Err(TlsError::General("TLS 1.2 is not offered".to_owned()))
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![SignatureScheme::ED25519]
    }

    fn requires_raw_public_keys(&self) -> bool {
        true
    }
}

/// The keys an acceptor admits: the plan's neighbours, which its owner sets.
#[derive(Clone, Default)]
pub struct AllowedKeys(Arc<RwLock<Vec<[u8; 32]>>>);

impl AllowedKeys {
    /// Admits `key`.
    pub fn allow(&self, key: [u8; 32]) {
        let mut keys = self
            .0
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !keys.contains(&key) {
            keys.push(key);
        }
    }

    /// Stops admitting `key`. Channels already open stay open.
    pub fn forbid(&self, key: &[u8; 32]) {
        self.0
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retain(|allowed| allowed != key);
    }

    /// Whether `key` is admitted.
    pub fn allows(&self, key: &[u8; 32]) -> bool {
        self.0
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(key)
    }
}

impl fmt::Debug for AllowedKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.0.read().map_or(0, |keys| keys.len());
        f.debug_struct("AllowedKeys").field("keys", &count).finish()
    }
}

/// The acceptor's check of the connector: its raw key must be one the owner allowed.
#[derive(Debug)]
struct AllowKeys {
    allowed: AllowedKeys,
}

impl ClientCertVerifier for AllowKeys {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }

    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<ClientCertVerified, TlsError> {
        match spki_key(end_entity) {
            Some(key) if self.allowed.allows(&key) && intermediates.is_empty() => {
                Ok(ClientCertVerified::assertion())
            }
            _ => Err(wrong_key()),
        }
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        // Only TLS 1.3 is offered, so a 1.2 handshake never reaches this.
        Err(TlsError::General("TLS 1.2 is not offered".to_owned()))
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![SignatureScheme::ED25519]
    }

    fn requires_raw_public_keys(&self) -> bool {
        true
    }
}

/// The connector's TLS configuration: TLS 1.3 only, `seed`'s key, and the acceptor's key pinned
/// to `expected`.
pub fn client_config(seed: &[u8; 32], expected: [u8; 32]) -> Result<ClientConfig, TlsError> {
    let mut config = ClientConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(ExpectKey { expected }))
        .with_client_cert_resolver(Arc::new(AlwaysResolvesClientRawPublicKeys::new(certified(
            seed,
        )?)));
    config.alpn_protocols = vec![DATA_ALPN.to_vec()];
    Ok(config)
}

/// The acceptor's TLS configuration: TLS 1.3 only, `seed`'s key, and a client key from `allowed`
/// required.
pub fn server_config(seed: &[u8; 32], allowed: AllowedKeys) -> Result<ServerConfig, TlsError> {
    let mut config = ServerConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .with_client_cert_verifier(Arc::new(AllowKeys { allowed }))
        .with_cert_resolver(Arc::new(AlwaysResolvesServerRawPublicKeys::new(certified(
            seed,
        )?)));
    config.alpn_protocols = vec![DATA_ALPN.to_vec()];
    Ok(config)
}

/// The server name a connector presents. A raw public key carries no name, and the acceptor
/// checks none, so it is a fixed placeholder.
pub fn server_name() -> ServerName<'static> {
    ServerName::try_from("syntropy.data").expect("a valid DNS name")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_spki_round_trips_its_key_and_refuses_anything_else() {
        let key = [7u8; 32];
        let der = spki(&key);
        assert_eq!(der.len(), 44);
        assert_eq!(spki_key(&der), Some(key));
        assert_eq!(spki_key(&der[..43]), None);
        let mut other = der.clone();
        other[8] ^= 1;
        assert_eq!(spki_key(&other), None);
    }

    #[test]
    fn a_seed_gives_a_signer_for_its_own_public_key() {
        let seed = [9u8; 32];
        let key = certified(&seed).unwrap();
        let public = DalekKey::from_bytes(&seed).verifying_key().to_bytes();
        assert_eq!(key.cert[0].as_ref(), spki(&public).as_slice());
        assert_eq!(key.key.algorithm(), rustls::SignatureAlgorithm::ED25519);
    }

    #[test]
    fn allowed_keys_add_remove_and_answer() {
        let allowed = AllowedKeys::default();
        assert!(!allowed.allows(&[1; 32]));
        allowed.allow([1; 32]);
        allowed.allow([1; 32]);
        assert!(allowed.allows(&[1; 32]));
        allowed.forbid(&[1; 32]);
        assert!(!allowed.allows(&[1; 32]));
    }
}
