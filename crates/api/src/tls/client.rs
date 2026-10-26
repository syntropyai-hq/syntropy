// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node's CA certificate as a client receives it, for `syntropy trust` (RFC-0050 §11.1.1;
//! PROD-E5-T04).
//!
//! A client fetches the certificate before it trusts anything, so it checks what it got before it
//! shows the fingerprint and installs it. [`CaCertificate::from_pem`] accepts exactly one PEM
//! `CERTIFICATE` block that is a self-signed CA certificate carrying critical X.509 name
//! constraints, with at least one permitted subtree and no permitted address outside the local
//! ranges ([`is_local`]). Anything else is refused, because a CA without those constraints is a
//! grant over every name, and no Syntropy node mints one (ADR-0505 item 6). This check does not
//! say which node minted the certificate: the user's fingerprint comparison does, and from M2 the
//! node-key binding does too (`trust --cluster`, PROD-E5-T08).

use std::fmt;
use std::net::IpAddr;

use rustls::pki_types::CertificateDer;
use rustls::pki_types::pem::PemObject;
use syntropy_net::crypto::{self, Context};
use x509_parser::extensions::{GeneralName, ParsedExtension};

use super::ca::{base64, certificate_pem};
use super::scope::is_local;

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
    /// constraints permit at least one subtree, and permit no address outside the local ranges.
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
                    permitted = Some(
                        constraints
                            .permitted_subtrees
                            .as_deref()
                            .unwrap_or_default()
                            .iter()
                            .map(|subtree| permitted_name(&subtree.base))
                            .collect::<Result<Vec<String>, CaRefused>>()?,
                    );
                }
                _ => {}
            }
        }
        if !is_ca {
            return Err(CaRefused("the certificate is not a CA certificate"));
        }
        let permitted = permitted.ok_or(CaRefused("the certificate has no name constraints"))?;
        if permitted.is_empty() {
            return Err(CaRefused(
                "the certificate's name constraints permit no subtree",
            ));
        }
        let subject = certificate
            .subject()
            .iter_common_name()
            .next()
            .and_then(|cn| cn.as_str().ok())
            .unwrap_or_default()
            .to_owned();
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

/// One permitted subtree for display, refused if it is not a DNS name or a local subnet.
fn permitted_name(name: &GeneralName<'_>) -> Result<String, CaRefused> {
    match name {
        GeneralName::DNSName(dns) if !dns.trim_start_matches('.').is_empty() => {
            Ok((*dns).to_owned())
        }
        GeneralName::IPAddress(bytes) => {
            let (addr, prefix) = match bytes.len() {
                8 => {
                    let addr: [u8; 4] = bytes[..4].try_into().expect("8 bytes");
                    (IpAddr::from(addr), prefix_len(&bytes[4..]))
                }
                32 => {
                    let addr: [u8; 16] = bytes[..16].try_into().expect("32 bytes");
                    (IpAddr::from(addr), prefix_len(&bytes[16..]))
                }
                _ => return Err(CaRefused("a permitted address is malformed")),
            };
            if !is_local(addr) {
                return Err(CaRefused("the CA may issue for a public address"));
            }
            Ok(format!("{addr}/{prefix}"))
        }
        _ => Err(CaRefused(
            "a permitted subtree is neither a DNS name nor an address",
        )),
    }
}

/// The number of leading one bits in a subnet mask.
fn prefix_len(mask: &[u8]) -> u32 {
    let mut ones = 0;
    for byte in mask {
        ones += byte.leading_ones();
        if *byte != 0xff {
            break;
        }
    }
    ones
}
