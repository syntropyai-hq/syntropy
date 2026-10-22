// SPDX-License-Identifier: MIT OR Apache-2.0

//! Leaf certificates and the resolver that serves the current one (RFC-0050 §11.1.1).

use std::fmt;
use std::net::IpAddr;
use std::sync::{Arc, PoisonError, RwLock};
use std::time::{Duration, SystemTime};

use rustls::pki_types::CertificateDer;
use rustls::server::{ClientHello, ResolvesServerCert};
use rustls::sign::CertifiedKey;

/// How long a leaf is valid: 30 days (RFC-0050 §11.1.1).
pub const LEAF_VALIDITY: Duration = Duration::from_secs(30 * 24 * 3600);

/// How long before expiry a leaf is reissued: 7 days (RFC-0050 §11.1.1).
pub const REISSUE_BEFORE: Duration = Duration::from_secs(7 * 24 * 3600);

/// The names and addresses a leaf carries: the node's current names and LAN addresses.
/// Names are lowercase without a trailing dot, addresses canonical (an IPv4-mapped IPv6 address
/// is its IPv4 address), and both sorted, so two sets of the same names compare equal.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LeafNames {
    dns_names: Vec<String>,
    addresses: Vec<IpAddr>,
}

impl LeafNames {
    /// The set of `dns_names` and `addresses`, normalised as the type says.
    pub fn new<'n>(
        dns_names: impl IntoIterator<Item = &'n str>,
        addresses: impl IntoIterator<Item = IpAddr>,
    ) -> LeafNames {
        let mut dns_names: Vec<String> = dns_names
            .into_iter()
            .map(|n| n.trim_end_matches('.').to_ascii_lowercase())
            .collect();
        let mut addresses: Vec<IpAddr> = addresses.into_iter().map(|a| a.to_canonical()).collect();
        dns_names.sort();
        dns_names.dedup();
        addresses.sort();
        addresses.dedup();
        LeafNames {
            dns_names,
            addresses,
        }
    }

    /// The DNS names, lowercase and sorted.
    pub fn dns_names(&self) -> &[String] {
        &self.dns_names
    }

    /// The addresses, canonical and sorted.
    pub fn addresses(&self) -> &[IpAddr] {
        &self.addresses
    }

    /// Whether there is neither a name nor an address.
    pub fn is_empty(&self) -> bool {
        self.dns_names.is_empty() && self.addresses.is_empty()
    }
}

/// A leaf certificate with its key, ready to serve: the chain is the leaf, then the CA.
pub struct Leaf {
    pub(super) certified: Arc<CertifiedKey>,
    pub(super) names: LeafNames,
    pub(super) not_after: SystemTime,
}

impl Leaf {
    /// The chain and signing key, as rustls serves them.
    pub fn certified_key(&self) -> Arc<CertifiedKey> {
        Arc::clone(&self.certified)
    }

    /// The leaf certificate.
    /// The leaf certificate, DER: the first certificate of the chain it is served with.
    pub fn certificate_der(&self) -> &CertificateDer<'static> {
        &self.certified.cert[0]
    }

    /// The names and addresses it carries.
    pub fn names(&self) -> &LeafNames {
        &self.names
    }

    /// When the leaf expires.
    pub fn not_after(&self) -> SystemTime {
        self.not_after
    }

    /// Whether the leaf must be reissued at `now` for the node's current `names`: they changed,
    /// or it expires within [`REISSUE_BEFORE`] (RFC-0050 §11.1.1).
    pub fn needs_reissue(&self, names: &LeafNames, now: SystemTime) -> bool {
        &self.names != names || now + REISSUE_BEFORE >= self.not_after
    }
}

/// Names and expiry only: the signing key never prints.
impl fmt::Debug for Leaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Leaf")
            .field("names", &self.names)
            .field("not_after", &self.not_after)
            .finish_non_exhaustive()
    }
}

/// Serves the current leaf to every TLS handshake, and swaps in a new one with
/// [`LeafResolver::rotate`]. A connection keeps the leaf it was handshaken with, so a rotation
/// drops none (RFC-0050 §11.1.1); the next handshake gets the new leaf.
pub struct LeafResolver {
    current: RwLock<Arc<CertifiedKey>>,
}

impl LeafResolver {
    /// A resolver that serves `leaf` until it is rotated.
    pub fn new(leaf: &Leaf) -> LeafResolver {
        LeafResolver {
            current: RwLock::new(leaf.certified_key()),
        }
    }

    /// Serves `leaf` from the next handshake on.
    pub fn rotate(&self, leaf: &Leaf) {
        *self.current.write().unwrap_or_else(PoisonError::into_inner) = leaf.certified_key();
    }

    /// The leaf served now.
    pub fn current(&self) -> Arc<CertifiedKey> {
        Arc::clone(&self.current.read().unwrap_or_else(PoisonError::into_inner))
    }
}

impl fmt::Debug for LeafResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeafResolver").finish_non_exhaustive()
    }
}

impl ResolvesServerCert for LeafResolver {
    fn resolve(&self, _client_hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        Some(self.current())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_names_compare_as_sets() {
        let a = LeafNames::new(
            ["Box.Local.", "llm.home.arpa"],
            [
                "192.168.1.5".parse().unwrap(),
                "::ffff:192.168.1.6".parse().unwrap(),
            ],
        );
        let b = LeafNames::new(
            ["llm.home.arpa", "box.local", "box.local"],
            [
                "192.168.1.6".parse().unwrap(),
                "192.168.1.5".parse().unwrap(),
            ],
        );
        assert_eq!(a, b);
        assert_eq!(a.dns_names(), ["box.local", "llm.home.arpa"]);
        assert!(!a.is_empty());
        assert!(LeafNames::new([], []).is_empty());
    }
}
