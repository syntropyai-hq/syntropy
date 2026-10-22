// SPDX-License-Identifier: MIT OR Apache-2.0

//! The node's CA: its key and certificate in the state directory, its node-key binding, and the
//! leaves it issues (RFC-0050 §11.1.1; RFC-0034 §3, §4).

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rcgen::{
    BasicConstraints, CertificateParams, CidrSubnet, DistinguishedName, DnType,
    ExtendedKeyUsagePurpose, GeneralSubtree, IsCa, Issuer, KeyPair, KeyUsagePurpose,
    NameConstraints, PKCS_ECDSA_P256_SHA256, PublicKeyData, SanType,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::sign::CertifiedKey;
use serde::{Deserialize, Serialize};
use syntropy_net::crypto::{self, Context, PUBLIC_KEY_LEN, SIGNATURE_LEN, SigningKey};
use time::OffsetDateTime;
use x509_parser::extensions::{GeneralName, GeneralSubtree as StoredSubtree, ParsedExtension};
use zeroize::Zeroize;

use super::files;
use super::leaf::{LEAF_VALIDITY, Leaf, LeafNames};
use super::scope::{CaScope, ScopeError};

/// The CA's file in its directory: key, certificate, scope and binding together, so one atomic
/// write replaces a CA whole. Owner-only (0600) in an owner-only directory (0700).
pub const CA_FILE: &str = "ca.json";

/// The record of the last CA replacement, which `doctor` SEC-006 reads to tell the user that
/// clients must re-trust the node (PROD-E9-T03).
pub const REPLACED_FILE: &str = "ca-replaced.json";

/// How long a CA certificate is valid: 10 years [target, a chosen budget]. A CA is replaced when
/// the node's scope changes, not on a schedule (RFC-0050 §11.1.1).
pub const CA_VALIDITY: Duration = Duration::from_secs(3650 * 24 * 3600);

/// How far before now a certificate's validity starts, for clients whose clocks run behind:
/// 1 hour [target, a chosen budget].
const NOT_BEFORE_SKEW: Duration = Duration::from_secs(3600);

/// The version of [`CA_FILE`]'s layout.
const FORMAT: u32 = 1;

/// The CA certificate's key usages, keyCertSign and cRLSign: bits 5 and 6 of RFC 5280
/// §4.2.1.3's list, as x509-parser numbers them.
const CA_KEY_USAGE: u16 = 1 << 5 | 1 << 6;

/// Why the CA could not be opened or a leaf issued.
#[derive(Debug)]
pub enum TlsError {
    /// The scope given could not be built.
    Scope(ScopeError),
    /// A leaf was asked for a name or address outside the CA's scope: a public DNS name, a
    /// public address, or a local address outside the subnets the CA serves.
    OutsideScope(String),
    /// A leaf was asked for no names at all.
    NoNames,
    /// A file could not be read or written.
    Io { path: PathBuf, error: io::Error },
    /// The CA's file is not one this build wrote, or its parts disagree. It is left as it is:
    /// no key is overwritten without a reason the user can see.
    Corrupt { path: PathBuf, what: &'static str },
    /// The certificate library refused the certificate.
    Certificate(rcgen::Error),
    /// rustls refused the leaf's key.
    Key(rustls::Error),
    /// A time outside what a certificate can carry.
    Time,
}

impl fmt::Display for TlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TlsError::Scope(e) => write!(f, "TLS CA scope: {e}"),
            TlsError::OutsideScope(name) => {
                write!(f, "`{name}` is outside what this node's CA may issue for")
            }
            TlsError::NoNames => f.write_str("a leaf needs at least one name or address"),
            TlsError::Io { path, error } => write!(f, "{}: {error}", path.display()),
            TlsError::Corrupt { path, what } => write!(f, "{}: {what}", path.display()),
            TlsError::Certificate(e) => write!(f, "certificate: {e}"),
            TlsError::Key(e) => write!(f, "leaf key: {e}"),
            TlsError::Time => f.write_str("a time no certificate can carry"),
        }
    }
}

impl std::error::Error for TlsError {}

impl From<ScopeError> for TlsError {
    fn from(e: ScopeError) -> TlsError {
        TlsError::Scope(e)
    }
}

impl From<rcgen::Error> for TlsError {
    fn from(e: rcgen::Error) -> TlsError {
        TlsError::Certificate(e)
    }
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> TlsError + '_ {
    move |error| TlsError::Io {
        path: path.to_owned(),
        error,
    }
}

/// The node key's statement binding the CA to the node (RFC-0034 R5, §3 revised 2026-10-22):
/// a signature in context `syntropy-tls-ca-v1` over `0x01 ‖ the CA certificate, DER`.
///
/// It covers the whole certificate, its name constraints above all, not just the CA's public
/// key. A client that knows the node key, from its admission certificate (RFC-0034 §5), checks
/// that the certificate it is about to trust is the one the node minted, byte for byte: a
/// certificate re-minted over the same key, without its constraints, fails the check.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CaBinding {
    node_key: [u8; PUBLIC_KEY_LEN],
    signature: [u8; SIGNATURE_LEN],
}

impl CaBinding {
    /// The statement's version byte.
    pub const VERSION: u8 = 1;

    /// The bytes the node key signs for the CA certificate `certificate_der`.
    pub fn statement(certificate_der: &[u8]) -> Vec<u8> {
        let mut statement = Vec::with_capacity(1 + certificate_der.len());
        statement.push(CaBinding::VERSION);
        statement.extend_from_slice(certificate_der);
        statement
    }

    /// The node key's binding of the CA certificate `certificate_der`.
    pub fn sign(node_key: &SigningKey, certificate_der: &[u8]) -> CaBinding {
        CaBinding {
            node_key: node_key.public_key(),
            signature: node_key.sign(Context::TlsCa, &CaBinding::statement(certificate_der)),
        }
    }

    /// Checks the binding against the CA certificate `certificate_der`, exactly as served.
    pub fn verify(&self, certificate_der: &[u8]) -> Result<(), crypto::SignatureError> {
        crypto::verify(
            &self.node_key,
            Context::TlsCa,
            &CaBinding::statement(certificate_der),
            &self.signature,
        )
    }

    /// The node key that signed it.
    pub fn node_key(&self) -> &[u8; PUBLIC_KEY_LEN] {
        &self.node_key
    }

    pub fn signature(&self) -> &[u8; SIGNATURE_LEN] {
        &self.signature
    }
}

impl fmt::Debug for CaBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CaBinding")
            .field("node_key", &hex(&self.node_key))
            .field("signature", &hex(&self.signature))
            .finish()
    }
}

/// Why a CA was replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplaceReason {
    /// The node's names or served subnets changed, so the old name constraints no longer fit.
    ScopeChanged,
    /// The node key changed (`identity reset`), so the old binding names another identity.
    NodeKeyChanged,
}

/// A CA replacement, kept in [`REPLACED_FILE`] for `doctor` SEC-006: clients that trusted the
/// old CA must trust the new one (RFC-0050 §11.1.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaReplacement {
    /// When, in seconds since the Unix epoch.
    pub replaced_at: u64,
    pub reason: ReplaceReason,
    /// The old and new CAs' [`LocalCa::fingerprint`], in hex.
    pub old_fingerprint: String,
    pub new_fingerprint: String,
}

/// What [`LocalCa::open`] found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaOpen {
    /// There was no CA, so one was generated.
    Created,
    /// The stored CA fits the node's scope and key.
    Loaded,
    /// The stored CA no longer fit, so a new one replaced it.
    Replaced(CaReplacement),
}

/// [`CA_FILE`]'s layout. The key is hex of its PKCS#8 DER, and is cleared when this is dropped.
#[derive(Serialize, Deserialize)]
struct CaFile {
    format: u32,
    /// Seconds since the Unix epoch.
    created: u64,
    node_key: String,
    binding: String,
    scope: CaScope,
    /// Hex of the certificate's DER: the exact bytes the binding covers, with no PEM text
    /// around them that the binding would not cover.
    certificate: String,
    key: String,
}

impl Drop for CaFile {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

/// The node's local CA (RFC-0050 §11.1.1; ADR-0505 item 6): one per node, never shared, and
/// never handed to another node or the coordinator.
///
/// Its key is ECDSA P-256, generated with ring's `SystemRandom`, the OS CSPRNG RFC-0034 §3
/// names. ring's random source is sealed, so it cannot be fed from `core::Entropy`; tests use
/// keys loaded from fixed bytes instead. The key never leaves this type: no accessor returns it,
/// it is not in `Debug`, and it is cleared when the CA is dropped.
pub struct LocalCa {
    dir: PathBuf,
    key: KeyPair,
    certificate: CertificateDer<'static>,
    certificate_pem: String,
    spki: Vec<u8>,
    scope: CaScope,
    binding: CaBinding,
    created: u64,
}

impl Drop for LocalCa {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

impl fmt::Debug for LocalCa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalCa")
            .field("dir", &self.dir)
            .field("fingerprint", &hex(&self.fingerprint()))
            .field("scope", &self.scope)
            .field("created", &self.created)
            .finish_non_exhaustive()
    }
}

fn unix(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn datetime(time: Option<SystemTime>) -> Result<OffsetDateTime, TlsError> {
    let secs = time
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .ok_or(TlsError::Time)?
        .as_secs();
    OffsetDateTime::from_unix_timestamp(i64::try_from(secs).map_err(|_| TlsError::Time)?)
        .map_err(|_| TlsError::Time)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

pub(super) fn common_name(name: &str) -> DistinguishedName {
    let mut dn = DistinguishedName::new();
    dn.push(DnType::CommonName, name);
    dn
}

/// The CA certificate's common name, from the node key it is bound to.
pub(super) fn ca_name(node_key: &[u8; PUBLIC_KEY_LEN]) -> String {
    format!("Syntropy local CA {}", hex(&node_key[..8]))
}

/// `bytes` in base64, with padding (RFC 4648 §4).
pub(super) fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let group = chunk
            .iter()
            .enumerate()
            .fold(0u32, |group, (i, b)| group | u32::from(*b) << (16 - 8 * i));
        for i in 0..4 {
            text.push(if i <= chunk.len() {
                char::from(ALPHABET[(group >> (18 - 6 * i) & 63) as usize])
            } else {
                '='
            });
        }
    }
    text
}

/// `der` as one PEM `CERTIFICATE` block, in lines of 64 with LF line ends (RFC 7468 §2): built
/// from the DER, so it carries nothing the binding does not cover.
pub(super) fn certificate_pem(der: &[u8]) -> String {
    let body = base64(der);
    let mut text = String::from("-----BEGIN CERTIFICATE-----\n");
    for line in body.as_bytes().chunks(64) {
        text.push_str(std::str::from_utf8(line).expect("base64 is ASCII"));
        text.push('\n');
    }
    text.push_str("-----END CERTIFICATE-----\n");
    text
}

/// The parameters of the CA certificate for `scope` and the node key `node_key`, issued at
/// `now`: what [`LocalCa::with_key`] signs, and what [`check_certificate`] holds a stored
/// certificate to.
fn ca_params(
    scope: &CaScope,
    node_key: &[u8; PUBLIC_KEY_LEN],
    now: SystemTime,
) -> Result<CertificateParams, TlsError> {
    let mut params = CertificateParams::default();
    params.not_before = datetime(now.checked_sub(NOT_BEFORE_SKEW))?;
    params.not_after = datetime(now.checked_add(CA_VALIDITY))?;
    params.distinguished_name = common_name(&ca_name(node_key));
    // Path length 0: it signs leaves only. Not every client enforces that for a trust anchor;
    // every client enforces the name constraints, below any intermediate too.
    params.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
    params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    params.name_constraints = Some(name_constraints(scope));
    Ok(params)
}

/// Whether a stored certificate's subtrees are the `minted` ones, in the same order.
fn subtrees_match(minted: &[GeneralSubtree], stored: Option<&[StoredSubtree<'_>]>) -> bool {
    let stored = stored.unwrap_or_default();
    minted.len() == stored.len()
        && minted
            .iter()
            .zip(stored)
            .all(|(minted, stored)| match (minted, &stored.base) {
                (GeneralSubtree::DnsName(minted), GeneralName::DNSName(stored)) => minted == stored,
                (
                    GeneralSubtree::IpAddress(CidrSubnet::V4(addr, mask)),
                    GeneralName::IPAddress(stored),
                ) => *stored == [&addr[..], &mask[..]].concat(),
                (
                    GeneralSubtree::IpAddress(CidrSubnet::V6(addr, mask)),
                    GeneralName::IPAddress(stored),
                ) => *stored == [&addr[..], &mask[..]].concat(),
                _ => false,
            })
}

/// Holds a stored CA certificate to what this node mints for `scope` and `node_key`
/// ([`ca_params`]): the CA key's own certificate (`spki`), self-signed and named for the node,
/// with exactly the CA's extensions: a critical path length 0, critical certificate and CRL
/// signing, and critical name constraints that are the scope's. Anything else is refused. The
/// binding says the node signed these bytes; this says they are bytes the node would sign.
fn check_certificate(
    der: &[u8],
    spki: &[u8],
    scope: &CaScope,
    node_key: &[u8; PUBLIC_KEY_LEN],
) -> Result<(), &'static str> {
    let (rest, certificate) =
        x509_parser::parse_x509_certificate(der).map_err(|_| "the certificate is not X.509")?;
    if !rest.is_empty() {
        return Err("the certificate has bytes after it");
    }
    if certificate.tbs_certificate.subject_pki.raw != spki {
        return Err("the certificate is not the key's");
    }
    certificate
        .verify_signature(None)
        .map_err(|_| "the certificate's self-signature does not verify")?;
    let subject = certificate.subject();
    let name = ca_name(node_key);
    let named = subject.iter_attributes().count() == 1
        && subject
            .iter_common_name()
            .next()
            .and_then(|cn| cn.as_str().ok())
            == Some(name.as_str());
    if !named || certificate.issuer().as_raw() != subject.as_raw() {
        return Err("the certificate is not named for this node's CA");
    }
    let constraints = name_constraints(scope);
    // Subject key identifier, basic constraints, key usage, name constraints: each once.
    let mut seen = [false; 4];
    for extension in certificate.extensions() {
        let (slot, fits, what) = match extension.parsed_extension() {
            ParsedExtension::SubjectKeyIdentifier(_) => (
                0,
                !extension.critical,
                "the certificate's key identifier is critical",
            ),
            ParsedExtension::BasicConstraints(basic) => (
                1,
                extension.critical && basic.ca && basic.path_len_constraint == Some(0),
                "the certificate's basic constraints are not a leaf-only CA's",
            ),
            ParsedExtension::KeyUsage(usage) => (
                2,
                extension.critical && usage.flags == CA_KEY_USAGE,
                "the certificate's key usages are not the CA's",
            ),
            ParsedExtension::NameConstraints(stored) => (
                3,
                extension.critical
                    && subtrees_match(
                        &constraints.permitted_subtrees,
                        stored.permitted_subtrees.as_deref(),
                    )
                    && subtrees_match(
                        &constraints.excluded_subtrees,
                        stored.excluded_subtrees.as_deref(),
                    ),
                "the certificate's name constraints are not its scope's",
            ),
            _ => return Err("the certificate carries an extension this node does not mint"),
        };
        if !fits {
            return Err(what);
        }
        if std::mem::replace(&mut seen[slot], true) {
            return Err("the certificate repeats an extension");
        }
    }
    if seen != [true; 4] {
        return Err("the certificate lacks an extension this node mints");
    }
    Ok(())
}

/// The CA's critical name constraints (RFC-0050 §11.1.1): the scope's names and subnets
/// permitted. An address family with no permitted subnet is excluded whole, so the CA can issue
/// for no address of it.
pub(super) fn name_constraints(scope: &CaScope) -> NameConstraints {
    let mut permitted: Vec<GeneralSubtree> = scope
        .dns_names()
        .iter()
        .map(|name| GeneralSubtree::DnsName(name.clone()))
        .collect();
    permitted.extend(scope.subnets().iter().map(|subnet| {
        GeneralSubtree::IpAddress(CidrSubnet::from_addr_prefix(subnet.addr(), subnet.prefix()))
    }));
    let mut excluded = Vec::new();
    if !scope.subnets().iter().any(|s| s.addr().is_ipv4()) {
        excluded.push(GeneralSubtree::IpAddress(CidrSubnet::from_v4_prefix(
            [0; 4], 0,
        )));
    }
    if !scope.subnets().iter().any(|s| s.addr().is_ipv6()) {
        excluded.push(GeneralSubtree::IpAddress(CidrSubnet::from_v6_prefix(
            [0; 16], 0,
        )));
    }
    NameConstraints {
        permitted_subtrees: permitted,
        excluded_subtrees: excluded,
    }
}

impl LocalCa {
    /// Opens the CA in `dir`, the `tls` directory under the node's state directory, for a node
    /// whose scope is `scope` and whose node key is `node_key`.
    /// - With no CA there, it generates one (`Created`).
    /// - A stored CA that fits both is used (`Loaded`).
    /// - A stored CA with another scope or another node key is replaced (`Replaced`), and the
    ///   replacement is recorded in [`REPLACED_FILE`] first, for SEC-006.
    ///
    /// A file this build cannot read, or whose parts disagree, is an error, and is left as it is.
    pub fn open(
        dir: &Path,
        scope: &CaScope,
        node_key: &SigningKey,
        now: SystemTime,
    ) -> Result<(LocalCa, CaOpen), TlsError> {
        files::private_dir(dir).map_err(io_error(dir))?;
        let path = dir.join(CA_FILE);
        let Some(mut bytes) = files::read_optional(&path).map_err(io_error(&path))? else {
            let ca = LocalCa::generate(dir, scope, node_key, now)?;
            ca.save()?;
            return Ok((ca, CaOpen::Created));
        };
        let stored = LocalCa::load(dir, &bytes);
        bytes.zeroize();
        let stored = stored?;
        let reason = if stored.binding.node_key != node_key.public_key() {
            ReplaceReason::NodeKeyChanged
        } else if &stored.scope != scope {
            ReplaceReason::ScopeChanged
        } else {
            return Ok((stored, CaOpen::Loaded));
        };
        let ca = LocalCa::generate(dir, scope, node_key, now)?;
        let record = CaReplacement {
            replaced_at: unix(now),
            reason,
            old_fingerprint: hex(&stored.fingerprint()),
            new_fingerprint: hex(&ca.fingerprint()),
        };
        // The record first: a crash before the new CA is saved leaves the old one, which the
        // next open replaces again, with a new record.
        let record_path = dir.join(REPLACED_FILE);
        let text = serde_json::to_vec_pretty(&record).expect("the record serialises");
        files::write_private(&record_path, &text).map_err(io_error(&record_path))?;
        ca.save()?;
        Ok((ca, CaOpen::Replaced(record)))
    }

    /// The last replacement recorded in `dir`, if any: what SEC-006 reports.
    pub fn last_replacement(dir: &Path) -> Result<Option<CaReplacement>, TlsError> {
        let path = dir.join(REPLACED_FILE);
        let Some(bytes) = files::read_optional(&path).map_err(io_error(&path))? else {
            return Ok(None);
        };
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|_| TlsError::Corrupt {
                path,
                what: "not a CA replacement record",
            })
    }

    fn generate(
        dir: &Path,
        scope: &CaScope,
        node_key: &SigningKey,
        now: SystemTime,
    ) -> Result<LocalCa, TlsError> {
        let key = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256)?;
        LocalCa::with_key(dir, scope, node_key, now, key)
    }

    /// The CA for `key`; `generate` passes a fresh key, tests a fixed one.
    pub(super) fn with_key(
        dir: &Path,
        scope: &CaScope,
        node_key: &SigningKey,
        now: SystemTime,
        key: KeyPair,
    ) -> Result<LocalCa, TlsError> {
        let params = ca_params(scope, &node_key.public_key(), now)?;
        let certificate = params.self_signed(&key)?.der().clone();
        Ok(LocalCa {
            dir: dir.to_owned(),
            // The binding is signed over the finished certificate, so it covers every byte a
            // client installs.
            binding: CaBinding::sign(node_key, &certificate),
            certificate_pem: certificate_pem(&certificate),
            certificate,
            spki: key.subject_public_key_info(),
            key,
            scope: scope.clone(),
            created: unix(now),
        })
    }

    /// The CA in a [`CA_FILE`]'s `bytes`, refused unless every part agrees: the binding verifies
    /// over the certificate's exact bytes, and the certificate is the key's own, self-signed,
    /// and carries the constraints this node mints for the stored scope.
    fn load(dir: &Path, bytes: &[u8]) -> Result<LocalCa, TlsError> {
        let path = dir.join(CA_FILE);
        let corrupt = |what| TlsError::Corrupt {
            path: path.clone(),
            what,
        };
        let file: CaFile =
            serde_json::from_slice(bytes).map_err(|_| corrupt("not a CA file this build wrote"))?;
        if file.format != FORMAT {
            return Err(corrupt("a CA file format this build does not read"));
        }
        let scope = file
            .scope
            .clone()
            .checked()
            .ok_or_else(|| corrupt("the scope is not one this build writes"))?;
        let node_key: [u8; PUBLIC_KEY_LEN] = unhex(&file.node_key)
            .and_then(|k| k.try_into().ok())
            .ok_or_else(|| corrupt("the node key is not 32 bytes of hex"))?;
        let signature: [u8; SIGNATURE_LEN] =
            unhex(&file.binding)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(|| corrupt("the binding is not 64 bytes of hex"))?;
        let binding = CaBinding {
            node_key,
            signature,
        };
        let certificate =
            unhex(&file.certificate).ok_or_else(|| corrupt("the certificate is not hex"))?;
        // Over the exact bytes, so any edit to the certificate fails here.
        binding
            .verify(&certificate)
            .map_err(|_| corrupt("the node-key binding does not verify"))?;
        let mut der = unhex(&file.key).ok_or_else(|| corrupt("the key is not hex"))?;
        let key = KeyPair::from_pkcs8_der_and_sign_algo(
            &PrivatePkcs8KeyDer::from(&der[..]),
            &PKCS_ECDSA_P256_SHA256,
        );
        der.zeroize();
        let key = key.map_err(|_| corrupt("the key is not a P-256 PKCS#8 key"))?;
        let spki = key.subject_public_key_info();
        check_certificate(&certificate, &spki, &scope, &node_key).map_err(corrupt)?;
        Ok(LocalCa {
            dir: dir.to_owned(),
            key,
            certificate_pem: certificate_pem(&certificate),
            certificate: CertificateDer::from(certificate),
            spki,
            scope,
            binding,
            created: file.created,
        })
    }

    fn save(&self) -> Result<(), TlsError> {
        let mut der = self.key.serialize_der();
        let file = CaFile {
            format: FORMAT,
            created: self.created,
            node_key: hex(&self.binding.node_key),
            binding: hex(&self.binding.signature),
            scope: self.scope.clone(),
            certificate: hex(&self.certificate),
            key: hex(&der),
        };
        der.zeroize();
        let mut text = serde_json::to_vec_pretty(&file).expect("the CA file serialises");
        drop(file);
        let path = self.dir.join(CA_FILE);
        let written = files::write_private(&path, &text);
        text.zeroize();
        written.map_err(io_error(&path))
    }

    /// The CA certificate, DER.
    pub fn certificate_der(&self) -> &CertificateDer<'static> {
        &self.certificate
    }

    /// The CA certificate, PEM: what `GET /v1/syntropy/tls/ca` returns and `syntropy trust`
    /// installs. Public material only, encoded from [`LocalCa::certificate_der`], the bytes the
    /// binding covers.
    pub fn certificate_pem(&self) -> &str {
        &self.certificate_pem
    }

    /// The CA's public key, SubjectPublicKeyInfo DER: what the fingerprint covers.
    pub fn spki_der(&self) -> &[u8] {
        &self.spki
    }

    /// `BLAKE3("syntropy-fp-v1" ‖ SPKI DER)`, the identity fingerprint of RFC-0034 §6.5 over
    /// the CA's public key. Its six-word form is PROD-E5-T10's.
    pub fn fingerprint(&self) -> [u8; 32] {
        crypto::hash(Context::Fingerprint, &self.spki)
    }

    pub fn scope(&self) -> &CaScope {
        &self.scope
    }

    pub fn binding(&self) -> &CaBinding {
        &self.binding
    }

    /// Issues a 30-day leaf for `names`, at `now`. Every name must be one of the scope's names,
    /// exactly, and every address must lie in one of its subnets; anything else, a public DNS
    /// name or address above all, is refused before anything is signed.
    pub fn issue_leaf(&self, names: &LeafNames, now: SystemTime) -> Result<Leaf, TlsError> {
        if names.is_empty() {
            return Err(TlsError::NoNames);
        }
        if let Some(name) = names
            .dns_names()
            .iter()
            .find(|n| !self.scope.permits_name(n))
        {
            return Err(TlsError::OutsideScope(name.clone()));
        }
        if let Some(ip) = names
            .addresses()
            .iter()
            .find(|ip| !self.scope.permits_ip(**ip))
        {
            return Err(TlsError::OutsideScope(ip.to_string()));
        }
        self.sign_leaf(names, now)
    }

    /// Signs a leaf for `names` without checking them against the scope. Only
    /// [`LocalCa::issue_leaf`] calls it outside tests, which use it to show that a client refuses
    /// a leaf the name constraints do not allow.
    pub(super) fn sign_leaf(&self, names: &LeafNames, now: SystemTime) -> Result<Leaf, TlsError> {
        let mut params = CertificateParams::default();
        for name in names.dns_names() {
            params
                .subject_alt_names
                .push(SanType::DnsName(name.as_str().try_into()?));
        }
        params
            .subject_alt_names
            .extend(names.addresses().iter().map(|ip| SanType::IpAddress(*ip)));
        let cn = names
            .dns_names()
            .first()
            .cloned()
            .or_else(|| names.addresses().first().map(ToString::to_string))
            .ok_or(TlsError::NoNames)?;
        params.distinguished_name = common_name(&cn);
        params.is_ca = IsCa::ExplicitNoCa;
        params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
        params.use_authority_key_identifier_extension = true;
        let not_after = now.checked_add(LEAF_VALIDITY).ok_or(TlsError::Time)?;
        params.not_before = datetime(now.checked_sub(NOT_BEFORE_SKEW))?;
        params.not_after = datetime(Some(not_after))?;
        let mut key = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256)?;
        let issuer = Issuer::from_ca_cert_der(&self.certificate, &self.key)?;
        let certificate = params.signed_by(&key, &issuer)?;
        let mut der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.serialize_der()));
        let signing = rustls::crypto::ring::sign::any_ecdsa_type(&der);
        der.zeroize();
        key.zeroize();
        let chain = vec![certificate.der().clone(), self.certificate.clone()];
        Ok(Leaf {
            certified: Arc::new(CertifiedKey::new(chain, signing.map_err(TlsError::Key)?)),
            names: names.clone(),
            not_after,
        })
    }
}
