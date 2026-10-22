// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bind classes and the bind plan of RFC-0050 §11.1 (ADR-0505 item 2).
//!
//! | Class | Addresses | API key | Transport |
//! |---|---|---|---|
//! | [`BindClass::Loopback`] | `127.0.0.0/8`, `::1` | by route (RFC-0050 §11.1) | plain HTTP |
//! | [`BindClass::Lan`] | private, link-local and unique local addresses, and the wildcards `0.0.0.0` and `::` | required | TLS; plain HTTP only with `api.allow_plaintext_lan = true` |
//! | [`BindClass::Public`] | every other address | required | TLS, always |
//!
//! The daemon runs [`plan_binds`] at start. It refuses to start a non-loopback bind until at
//! least one API key exists, and its message names the command that creates one. TLS never
//! blocks start, because the local CA supplies a certificate without configuration.

use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use crate::config::{ApiSection, TlsSource};

/// The command a refusal to start names (RFC-0050 §11.1).
pub const CREATE_KEY_COMMAND: &str = "syntropy api-key create --name … --scope inference";

/// What an address's listener must do (RFC-0050 §11.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindClass {
    /// `127.0.0.0/8` and `::1`, and their IPv4-mapped forms: plain HTTP, with no key for
    /// `inference` and the read-only routes (RFC-0050 §11.1).
    Loopback,
    /// A LAN address, or a wildcard: an API key and TLS, or plain HTTP with
    /// `api.allow_plaintext_lan = true`. LAN addresses are IPv4 `10/8`, `172.16/12`,
    /// `192.168/16` and link-local `169.254/16`, and IPv6 unique local `fc00::/7` and link-local
    /// `fe80::/10`. The wildcards `0.0.0.0` and `::` count as LAN binds (ADR-0505 item 2).
    Lan,
    /// Any other address: an API key and TLS, with no plain-HTTP opt-in. That includes
    /// `100.64/10`, the shared range tailnets use, until PROD-E5-T09 classifies overlay
    /// interfaces.
    Public,
}

/// The class of a bind address (RFC-0050 §11.1).
#[must_use]
pub fn classify(ip: IpAddr) -> BindClass {
    let ip = ip.to_canonical();
    if ip.is_loopback() {
        return BindClass::Loopback;
    }
    if ip.is_unspecified() {
        return BindClass::Lan;
    }
    let lan = match ip {
        IpAddr::V4(v4) => v4.is_private() || v4.is_link_local(),
        IpAddr::V6(v6) => v6.is_unique_local() || v6.is_unicast_link_local(),
    };
    if lan {
        BindClass::Lan
    } else {
        BindClass::Public
    }
}

/// Where a TLS listener's certificate comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Certificate {
    /// A leaf the local CA issues (RFC-0050 §11.1.1, PROD-E5-T03).
    LocalCa,
    /// The user's files from `api.tls`.
    Files {
        /// The certificate chain.
        cert: PathBuf,
        /// The private key.
        key: PathBuf,
    },
}

/// How a listener serves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Transport {
    /// Plain HTTP: loopback, or a LAN bind with `api.allow_plaintext_lan = true`, which the
    /// dashboard and SEC-005 flag (PROD-E5-T06).
    Plain,
    /// TLS 1.3.
    Tls(Certificate),
}

/// One listener of the plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listener {
    /// The address from `api.bind`.
    pub addr: SocketAddr,
    /// Its class.
    pub class: BindClass,
    /// How it serves.
    pub transport: Transport,
}

/// Why the daemon does not start (RFC-0050 §11.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StartRefusal {
    /// A bind beyond loopback, and no API key exists.
    NoApiKey {
        /// The binds that need a key.
        binds: Vec<SocketAddr>,
    },
    /// `api.tls = "none"` on a bind that may not serve plain HTTP. Validation rejects such a
    /// configuration first ([`crate::rules`]); the plan refuses it again rather than serve
    /// plain HTTP where the policy forbids it.
    PlainNotAllowed {
        /// The bind.
        bind: SocketAddr,
        /// Its class.
        class: BindClass,
    },
}

impl fmt::Display for StartRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StartRefusal::NoApiKey { binds } => {
                let list: Vec<String> = binds.iter().map(|b| format!("`{b}`")).collect();
                write!(
                    f,
                    "refusing to start: `api.bind` listens beyond loopback on {}, which needs \
                     at least one API key, and none exists; create one with `{CREATE_KEY_COMMAND}`",
                    list.join(", ")
                )
            }
            StartRefusal::PlainNotAllowed { bind, class } => {
                let why = match class {
                    BindClass::Public => "plain HTTP is never allowed on a public address",
                    _ => "plain HTTP beyond loopback needs `api.allow_plaintext_lan = true`",
                };
                write!(
                    f,
                    "refusing to start: `api.tls = \"none\"` on `{bind}`, but {why}"
                )
            }
        }
    }
}

impl std::error::Error for StartRefusal {}

/// The listeners `api` asks for, or why the daemon must not start (RFC-0050 §11.1).
///
/// `has_api_key` says whether at least one API key exists (PROD-E5-T01's key store).
///
/// # Errors
///
/// - [`StartRefusal::NoApiKey`] when a bind is not loopback and no key exists.
/// - [`StartRefusal::PlainNotAllowed`] when `api.tls = "none"` would serve plain HTTP on a
///   public address, or on a LAN address without `api.allow_plaintext_lan`.
pub fn plan_binds(api: &ApiSection, has_api_key: bool) -> Result<Vec<Listener>, StartRefusal> {
    let beyond_loopback: Vec<SocketAddr> = api
        .bind
        .iter()
        .copied()
        .filter(|addr| classify(addr.ip()) != BindClass::Loopback)
        .collect();
    if !beyond_loopback.is_empty() && !has_api_key {
        return Err(StartRefusal::NoApiKey {
            binds: beyond_loopback,
        });
    }
    api.bind
        .iter()
        .map(|&addr| {
            let class = classify(addr.ip());
            let transport = match (class, &api.tls) {
                (BindClass::Loopback, _) => Transport::Plain,
                (_, TlsSource::Auto | TlsSource::LocalCa) => Transport::Tls(Certificate::LocalCa),
                (_, TlsSource::Files { cert, key }) => Transport::Tls(Certificate::Files {
                    cert: cert.clone(),
                    key: key.clone(),
                }),
                (BindClass::Lan, TlsSource::None) if api.allow_plaintext_lan => Transport::Plain,
                (_, TlsSource::None) => {
                    return Err(StartRefusal::PlainNotAllowed { bind: addr, class });
                }
            };
            Ok(Listener {
                addr,
                class,
                transport,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn addresses_fall_in_their_classes() {
        for loopback in ["127.0.0.1", "127.8.9.10", "::1", "::ffff:127.0.0.1"] {
            assert_eq!(classify(ip(loopback)), BindClass::Loopback, "{loopback}");
        }
        for lan in [
            "0.0.0.0",
            "::",
            "::ffff:0.0.0.0",
            "10.1.2.3",
            "172.16.0.1",
            "172.31.255.254",
            "192.168.1.10",
            "169.254.3.4",
            "fd12:3456::1",
            "fc00::1",
            "fe80::1",
            "::ffff:192.168.1.10",
        ] {
            assert_eq!(classify(ip(lan)), BindClass::Lan, "{lan}");
        }
        for public in [
            "8.8.8.8",
            "172.32.0.1",
            "100.64.0.1",
            "100.101.102.103",
            "2001:db8::1",
            "2606:4700::1111",
            "::ffff:8.8.8.8",
        ] {
            assert_eq!(classify(ip(public)), BindClass::Public, "{public}");
        }
    }
}
