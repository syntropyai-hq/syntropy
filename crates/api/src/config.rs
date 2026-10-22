// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `[api]` section's listener keys (RFC-0054 §5.2, RFC-0050 §11.1).
//!
//! The section holds the keys the bind policy reads. The other `api.*` keys of RFC-0054 §5.2
//! join it with the tasks that use them.

use std::fmt;
use std::net::SocketAddr;
use std::path::PathBuf;

use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, Visitor};
use syntropy_config::{ConfigSection, HoldsSecret};

/// The HTTP API's listeners (RFC-0050 §11.1). Node scope; every key reloads hot.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, ConfigSection)]
#[config(section = "api", scope = Node, owner = "RFC-0050")]
pub struct ApiSection {
    /// The addresses the HTTP API listens on. A loopback address serves plain HTTP; any other
    /// address needs an API key and TLS (RFC-0050 §11.1). The local socket is separate.
    #[config(reload = Hot, default = ["127.0.0.1:7433", "[::1]:7433"])]
    pub bind: Vec<SocketAddr>,
    /// Where non-loopback listeners get their certificate: `"auto"` (the local CA on LAN
    /// binds), `"local_ca"`, `{ cert = path, key = path }` (your own files) or `"none"`, which
    /// is allowed only on LAN binds with `api.allow_plaintext_lan = true` (RFC-0050 §11.1).
    #[config(reload = Hot, default = "auto")]
    pub tls: TlsSource,
    /// Allows `api.tls = "none"` on LAN binds. The dashboard and doctor check SEC-005 flag it
    /// as insecure for as long as it is set (RFC-0001 §12).
    #[config(reload = Hot, default = false)]
    pub allow_plaintext_lan: bool,
}

/// `api.tls`: where a non-loopback listener's certificate comes from (RFC-0050 §11.1). Loopback
/// listeners serve plain HTTP whatever it says.
///
/// Written `"auto"`, `"local_ca"`, `"none"` or `{ cert = "…", key = "…" }`. A table with only
/// one of `cert` and `key` is the wrong shape, rejected when the value is read, with a message
/// that names the missing one.
#[derive(Clone, Debug, PartialEq, Eq, HoldsSecret)]
pub enum TlsSource {
    /// The local CA on LAN binds. Overlay-restricted binds, which may omit TLS, come with
    /// PROD-E5-T09.
    Auto,
    /// The local CA on every non-loopback bind, overlay binds included.
    LocalCa,
    /// The user's certificate chain and private key, both PEM files.
    Files {
        /// The certificate chain.
        cert: PathBuf,
        /// The certificate's private key.
        key: PathBuf,
    },
    /// No TLS. Valid on LAN binds only with `api.allow_plaintext_lan = true`, and never on a
    /// public address ([`crate::rules`]).
    None,
}

/// The names `api.tls` takes as a string.
const NAMES: &[&str] = &["auto", "local_ca", "none"];

/// The keys of `api.tls`'s table form.
const FIELDS: &[&str] = &["cert", "key"];

impl<'de> Deserialize<'de> for TlsSource {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(TlsVisitor)
    }
}

struct TlsVisitor;

impl<'de> Visitor<'de> for TlsVisitor {
    type Value = TlsSource;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("\"auto\", \"local_ca\", \"none\" or { cert = \"…\", key = \"…\" }")
    }

    fn visit_str<E: de::Error>(self, name: &str) -> Result<TlsSource, E> {
        match name {
            "auto" => Ok(TlsSource::Auto),
            "local_ca" => Ok(TlsSource::LocalCa),
            "none" => Ok(TlsSource::None),
            other => Err(E::unknown_variant(other, NAMES)),
        }
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<TlsSource, A::Error> {
        let (mut cert, mut key): (Option<PathBuf>, Option<PathBuf>) = (None, None);
        while let Some(name) = map.next_key::<String>()? {
            let slot = match name.as_str() {
                "cert" => &mut cert,
                "key" => &mut key,
                other => return Err(de::Error::unknown_field(other, FIELDS)),
            };
            if slot.is_some() {
                return Err(de::Error::custom(format!("`{name}` is given twice")));
            }
            *slot = Some(map.next_value()?);
        }
        match (cert, key) {
            (Some(cert), Some(key)) => Ok(TlsSource::Files { cert, key }),
            (Some(_), None) => Err(de::Error::custom(
                "a `{ cert, key }` pair needs `key` too: the certificate's private key file",
            )),
            (None, Some(_)) => Err(de::Error::custom(
                "a `{ cert, key }` pair needs `cert` too: the certificate chain file",
            )),
            (None, None) => Err(de::Error::custom(
                "a `{ cert, key }` pair needs both `cert` and `key`",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<TlsSource, String> {
        #[derive(Deserialize)]
        struct Doc {
            tls: TlsSource,
        }
        toml::from_str::<Doc>(text)
            .map(|d| d.tls)
            .map_err(|e| e.message().to_owned())
    }

    #[test]
    fn every_form_parses() {
        assert_eq!(parse(r#"tls = "auto""#), Ok(TlsSource::Auto));
        assert_eq!(parse(r#"tls = "local_ca""#), Ok(TlsSource::LocalCa));
        assert_eq!(parse(r#"tls = "none""#), Ok(TlsSource::None));
        assert_eq!(
            parse(r#"tls = { cert = "/c.pem", key = "/k.pem" }"#),
            Ok(TlsSource::Files {
                cert: "/c.pem".into(),
                key: "/k.pem".into()
            })
        );
    }

    #[test]
    fn a_half_pair_names_what_is_missing() {
        let err = parse(r#"tls = { cert = "/c.pem" }"#).unwrap_err();
        assert!(err.contains("needs `key` too"), "{err}");
        let err = parse(r#"tls = { key = "/k.pem" }"#).unwrap_err();
        assert!(err.contains("needs `cert` too"), "{err}");
        let err = parse(r#"tls = {}"#).unwrap_err();
        assert!(err.contains("both `cert` and `key`"), "{err}");
    }

    #[test]
    fn other_names_and_fields_are_refused() {
        let err = parse(r#"tls = "off""#).unwrap_err();
        assert!(err.contains("unknown variant `off`"), "{err}");
        let err = parse(r#"tls = { cert = "/c", key = "/k", chain = "/x" }"#).unwrap_err();
        assert!(err.contains("unknown field `chain`"), "{err}");
    }
}
