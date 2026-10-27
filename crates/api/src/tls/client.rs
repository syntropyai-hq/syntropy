// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node's CA certificate as a client receives it, for `syntropy trust` (RFC-0050 §11.1.1;
//! PROD-E5-T04).
//!
//! A client fetches the certificate before it trusts anything, so it checks what it got before it
//! shows the fingerprint and installs it. [`CaCertificate::from_pem`] accepts exactly one PEM
//! `CERTIFICATE` block that is a self-signed CA certificate carrying critical X.509 name
//! constraints that hold it to what a node mints: permitted DNS names of two or more labels,
//! permitted subnets each inside one local block (the ranges [`super::is_local`] accepts), and
//! each address family either permitted a subnet or excluded whole. Anything else is refused, because a CA without those constraints is a
//! grant over every name, and no Syntropy node mints one (ADR-0505 item 6). This check does not
//! say which node minted the certificate: the user's fingerprint comparison does, and from M2 the
//! node-key binding does too (`trust --cluster`, PROD-E5-T08).

use std::fmt;
use std::net::{Ipv4Addr, Ipv6Addr};

use rustls::pki_types::CertificateDer;
use rustls::pki_types::pem::PemObject;
use syntropy_net::crypto::{self, Context};
use x509_parser::extensions::{GeneralName, NameConstraints, ParsedExtension};

use super::ca::{base64, certificate_pem};
use super::scope::host_name;

/// Why a fetched certificate is not installed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaRefused(pub &'static str);

impl fmt::Display for CaRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not a Syntropy node's CA certificate: {}", self.0)
    }
}

impl std::error::Error for CaRefused {}

/// A node's CA certificate, checked as [`CaCertificate::from_pem`] says.
#[derive(Clone, PartialEq, Eq)]
pub struct CaCertificate {
    der: Vec<u8>,
    spki: Vec<u8>,
    subject: String,
    permitted: Vec<String>,
}

impl fmt::Debug for CaCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CaCertificate")
            .field("subject", &self.subject)
            .field("fingerprint", &self.fingerprint_hex())
            .finish_non_exhaustive()
    }
}

impl CaCertificate {
    /// Parses and checks the PEM text a node served: one `CERTIFICATE` block and nothing else but
    /// white space around it, holding a self-signed CA certificate whose critical name
    /// constraints hold it to what a node mints (see the module documentation).
    ///
    /// # Errors
    ///
    /// [`CaRefused`], naming the first check that failed.
    pub fn from_pem(text: &str) -> Result<CaCertificate, CaRefused> {
        let trimmed = text.trim();
        if !trimmed.starts_with("-----BEGIN CERTIFICATE-----")
            || !trimmed.ends_with("-----END CERTIFICATE-----")
            || trimmed.matches("-----BEGIN").count() != 1
            || trimmed.matches("-----END").count() != 1
        {
            return Err(CaRefused("the text is not exactly one PEM certificate"));
        }
        let der = CertificateDer::from_pem_slice(trimmed.as_bytes())
            .map_err(|_| CaRefused("the PEM certificate does not decode"))?;
        CaCertificate::from_der(der.as_ref())
    }

    /// As [`CaCertificate::from_pem`], from the certificate's DER bytes.
    ///
    /// # Errors
    ///
    /// [`CaRefused`], naming the first check that failed.
    pub fn from_der(der: &[u8]) -> Result<CaCertificate, CaRefused> {
        let (rest, certificate) = x509_parser::parse_x509_certificate(der)
            .map_err(|_| CaRefused("the certificate is not X.509"))?;
        if !rest.is_empty() {
            return Err(CaRefused("the certificate has bytes after it"));
        }
        if certificate.issuer().as_raw() != certificate.subject().as_raw() {
            return Err(CaRefused("the certificate is not self-signed"));
        }
        certificate
            .verify_signature(None)
            .map_err(|_| CaRefused("the certificate's self-signature does not verify"))?;
        let mut is_ca = false;
        let mut permitted = None;
        for extension in certificate.extensions() {
            match extension.parsed_extension() {
                ParsedExtension::BasicConstraints(basic) => is_ca = basic.ca,
                ParsedExtension::NameConstraints(constraints) => {
                    if !extension.critical {
                        return Err(CaRefused(
                            "the certificate's name constraints are not critical",
                        ));
                    }
                    permitted = Some(checked_permitted(constraints)?);
                }
                _ => {}
            }
        }
        if !is_ca {
            return Err(CaRefused("the certificate is not a CA certificate"));
        }
        let permitted = permitted.ok_or(CaRefused("the certificate has no name constraints"))?;
        // The subject is shown beside the fingerprint, so it must be exactly the name a node
        // gives its CA: one common name, `Syntropy local CA <16 lowercase hex digits>`.
        let subject = certificate
            .subject()
            .iter_common_name()
            .next()
            .and_then(|cn| cn.as_str().ok())
            .unwrap_or_default()
            .to_owned();
        if certificate.subject().iter_attributes().count() != 1 || !is_ca_name(&subject) {
            return Err(CaRefused(
                "the certificate is not named as a node's CA (`Syntropy local CA <16 hex digits>`)",
            ));
        }
        Ok(CaCertificate {
            der: der.to_vec(),
            spki: certificate.tbs_certificate.subject_pki.raw.to_vec(),
            subject,
            permitted,
        })
    }

    /// The certificate, DER.
    pub fn der(&self) -> &[u8] {
        &self.der
    }

    /// The certificate as one PEM block, encoded from [`CaCertificate::der`] in lines of 64 with
    /// LF line ends: the file `syntropy trust` writes.
    pub fn pem(&self) -> String {
        certificate_pem(&self.der)
    }

    /// The DER in base64 on one line, with padding: how the host-change ledger records it.
    pub fn base64(&self) -> String {
        base64(&self.der)
    }

    /// `BLAKE3("syntropy-fp-v1" ‖ SPKI DER)`, the same fingerprint as
    /// [`super::LocalCa::fingerprint`] on the node.
    pub fn fingerprint(&self) -> [u8; 32] {
        crypto::hash(Context::Fingerprint, &self.spki)
    }

    /// [`CaCertificate::fingerprint`] in lowercase hex, as the node records it for SEC-006.
    pub fn fingerprint_hex(&self) -> String {
        self.fingerprint()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// The fingerprint's six-word form (RFC-0034 §6.5), from the same digest.
    pub fn fingerprint_words(&self) -> String {
        syntropy_net::fingerprint::fingerprint(&self.spki)
    }

    /// The certificate's common name.
    pub fn subject(&self) -> &str {
        &self.subject
    }

    /// What the CA may issue for, from its permitted subtrees: DNS names, and subnets as
    /// `address/prefix`. Shown to the user beside the fingerprint.
    pub fn permitted(&self) -> &[String] {
        &self.permitted
    }
}

/// Whether `name` is a node CA's common name: `Syntropy local CA ` and 16 lowercase hex digits,
/// as `ca::ca_name` makes it from the node key.
fn is_ca_name(name: &str) -> bool {
    name.strip_prefix("Syntropy local CA ").is_some_and(|hex| {
        hex.len() == 16
            && hex
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

/// A name-constraint subtree this check understands.
enum Subtree {
    /// A DNS name and every name below it.
    Dns(String),
    /// An IPv4 network and its prefix length.
    V4(u32, u32),
    /// An IPv6 network and its prefix length.
    V6(u128, u32),
    /// Any other name form, which a TLS client does not match a server against.
    Other,
}

/// The local blocks a permitted subnet must lie inside, whole: what [`super::is_local`] accepts.
const LOCAL_V4: [(u32, u32); 5] = [
    (0x0a00_0000, 8),  // 10/8
    (0xac10_0000, 12), // 172.16/12
    (0xc0a8_0000, 16), // 192.168/16
    (0xa9fe_0000, 16), // 169.254/16, link-local
    (0x6440_0000, 10), // 100.64/10, shared
];
const LOCAL_V6: [(u128, u32); 2] = [
    (0xfc00 << 112, 7),  // fc00::/7, unique local
    (0xfe80 << 112, 10), // fe80::/10, link-local
];

/// Whether `prefix` bits of `network` lie inside one of `blocks`, whole.
fn inside<T>(network: T, prefix: u32, blocks: &[(T, u32)], width: u32) -> bool
where
    T: Copy + PartialEq + std::ops::Shr<u32, Output = T>,
{
    blocks
        .iter()
        .any(|&(block, len)| prefix >= len && network >> (width - len) == block >> (width - len))
}

/// The prefix length of a contiguous subnet mask, or `None` for a mask with a hole in it.
fn prefix_len(mask: &[u8]) -> Option<u32> {
    let ones: u32 = mask.iter().map(|b| b.count_ones()).sum();
    let contiguous = mask
        .iter()
        .flat_map(|b| (0..8).rev().map(move |i| b >> i & 1))
        .enumerate()
        .all(|(i, bit)| (bit == 1) == (i < ones as usize));
    contiguous.then_some(ones)
}

fn subtree(name: &GeneralName<'_>) -> Result<Subtree, CaRefused> {
    const MALFORMED: CaRefused = CaRefused("a name-constraint address is malformed");
    Ok(match name {
        GeneralName::DNSName(dns) => Subtree::Dns((*dns).to_owned()),
        GeneralName::IPAddress(bytes) if bytes.len() == 8 => {
            let prefix = prefix_len(&bytes[4..]).ok_or(MALFORMED)?;
            let addr = u32::from_be_bytes(bytes[..4].try_into().expect("4 bytes"));
            let mask = u32::MAX.checked_shl(32 - prefix).unwrap_or(0);
            Subtree::V4(addr & mask, prefix)
        }
        GeneralName::IPAddress(bytes) if bytes.len() == 32 => {
            let prefix = prefix_len(&bytes[16..]).ok_or(MALFORMED)?;
            let addr = u128::from_be_bytes(bytes[..16].try_into().expect("16 bytes"));
            let mask = u128::MAX.checked_shl(128 - prefix).unwrap_or(0);
            Subtree::V6(addr & mask, prefix)
        }
        GeneralName::IPAddress(_) => return Err(MALFORMED),
        _ => Subtree::Other,
    })
}

/// Holds the name constraints to what a node mints (RFC-0050 §11.1.1), and returns the
/// permitted subtrees for display.
/// - Every permitted subtree is a DNS name of two or more labels that the node's host-name rule
///   leaves unchanged, or a subnet that lies whole inside one local block: a mask such as
///   `10.0.0.0/0` is refused, not read as `10/8`.
/// - At least one DNS name is permitted, as every node's scope has one; otherwise DNS names
///   would be unconstrained.
/// - Each address family is constrained: some subnet of it is permitted, or the whole family is
///   excluded, as a node with no subnet of that family excludes it. Otherwise the CA could issue
///   for any address of that family.
fn checked_permitted(constraints: &NameConstraints<'_>) -> Result<Vec<String>, CaRefused> {
    let mut shown = Vec::new();
    let (mut dns, mut v4, mut v6) = (false, false, false);
    for entry in constraints
        .permitted_subtrees
        .as_deref()
        .unwrap_or_default()
    {
        match subtree(&entry.base)? {
            Subtree::Dns(name) => {
                // The node's own host-name rule, which every name it mints passed, and with the
                // name unchanged by it: lowercase letters, digits and hyphens in dot-separated
                // labels. No wildcard, space, slash or control character reaches the user's
                // screen beside the fingerprint.
                if host_name(&name).as_deref() != Ok(name.as_str()) {
                    return Err(CaRefused(
                        "a permitted name is not a host name a node mints",
                    ));
                }
                if !name.contains('.') {
                    return Err(CaRefused("the CA may issue for a whole top-level domain"));
                }
                shown.push(name);
                dns = true;
            }
            Subtree::V4(network, prefix) => {
                if !inside(network, prefix, &LOCAL_V4, 32) {
                    return Err(CaRefused("the CA may issue for a public address"));
                }
                v4 = true;
                shown.push(format!("{}/{prefix}", Ipv4Addr::from(network)));
            }
            Subtree::V6(network, prefix) => {
                if !inside(network, prefix, &LOCAL_V6, 128) {
                    return Err(CaRefused("the CA may issue for a public address"));
                }
                v6 = true;
                shown.push(format!("{}/{prefix}", Ipv6Addr::from(network)));
            }
            Subtree::Other => {
                return Err(CaRefused(
                    "a permitted subtree is neither a DNS name nor an address",
                ));
            }
        }
    }
    for entry in constraints.excluded_subtrees.as_deref().unwrap_or_default() {
        match subtree(&entry.base)? {
            Subtree::V4(_, 0) => v4 = true,
            Subtree::V6(_, 0) => v6 = true,
            _ => {}
        }
    }
    // Every node's scope has a DNS name (`ScopeError::NoNames`). Without a permitted one, DNS
    // names would be unconstrained.
    if !dns {
        return Err(CaRefused("the CA may issue for any DNS name"));
    }
    if !v4 {
        return Err(CaRefused("the CA may issue for any IPv4 address"));
    }
    if !v6 {
        return Err(CaRefused("the CA may issue for any IPv6 address"));
    }
    Ok(shown)
}
