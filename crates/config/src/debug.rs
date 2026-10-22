// SPDX-License-Identifier: MIT OR Apache-2.0

//! The debug-key gates of RFC-0054 §7.1 (PROD-E1-T05; ADR-0505 item 7; RFC-0060 §6.4).
//!
//! Three node keys open a test-only surface on the builds that ship, so the parity harness
//! (RFC-0060 §1.3) and the benchmark driver (RFC-0061 §10.2) can test the signed release
//! artefacts themselves:
//!
//! | Key | Enables |
//! |---|---|
//! | `debug.parity_api` | `POST /v1/syntropy/parity` |
//! | `debug.bench_api` | `BenchControls` on `POST /v1/syntropy/generate` |
//! | `debug.telemetry_api` | `GET /v1/syntropy/debug/telemetry` |
//!
//! This module holds what every user of the gates shares:
//! - [`DebugSection`], the three keys, each `false` by default in every build profile. Nothing
//!   here depends on a `cfg` or a feature, so `dev`, `test-release`, `chaos` and `release`
//!   builds have one schema. A binary that serves the API registers [`DebugSection::SCHEMA`].
//! - [`DebugSection::allows`], the loopback predicate the API applies to each gated request:
//!   the endpoint answers only while its key is on, and only on a loopback listener or the
//!   local socket to a loopback peer. Everything else gets 404, whatever key the caller holds.
//! - The warning a [`ResolvedConfig`] carries for each key that is on, in
//!   [`ResolvedConfig::warnings`]. A reload builds a new `ResolvedConfig`, so the warning comes
//!   with every reload too. The startup log (PROD-E6-T02), reloads (PROD-E1-T11), doctor
//!   CFG-003 (PROD-E9-T02) and the dashboard banner read it, or [`DebugSection::on`].
//! - [`DebugSection::ci_test_manifest`], the manifest-signer predicate the manifest loader
//!   (QUAL-E6-T04) applies to a `ConformanceManifest` signed by the CI test key.

use std::fmt;
use std::net::{IpAddr, SocketAddr};

use serde::Deserialize;

use crate::resolve::{Problem, ResolvedConfig};
// The trait and the derive, which share the name.
use crate::ConfigSection;

/// The test-only request controls (RFC-0054 §7.1). Each key is off by default in every build
/// profile, release included. Only an explicit setting in a file, the environment or a flag
/// turns one on, and its endpoint then answers on loopback and the local socket only.
///
/// ```
/// use syntropy_config::{DebugApi, DebugSection, Env, Flags, Layers, Os, Peer, Registry};
/// use syntropy_config::{ConfigSection, ResolvedConfig};
///
/// static REGISTRY: Registry = Registry::new(&[&DebugSection::SCHEMA]);
///
/// let env = Env::from_pairs(Os::Linux, [("SYNTROPY_DEBUG__PARITY_API", "true")]);
/// let layers = Layers { os: Os::Linux, files: &[], env: &env, flags: &Flags::new() };
/// let config = ResolvedConfig::resolve(&REGISTRY, layers).unwrap();
/// assert!(config.warnings()[0].message.starts_with("`debug.parity_api` is on"));
///
/// let gates = DebugSection::from_config(&config);
/// let lan = Peer::Tcp { listener: "0.0.0.0".parse().unwrap(), remote: "192.168.1.20".parse().unwrap() };
/// assert!(gates.allows(DebugApi::Parity, &Peer::LocalSocket));
/// assert!(!gates.allows(DebugApi::Parity, &lan));
/// assert!(!gates.allows(DebugApi::Telemetry, &Peer::LocalSocket));
///
/// // The CI test key, with the default `api.bind`, then with a LAN listener added.
/// let default_bind = ["127.0.0.1:7433".parse().unwrap(), "[::1]:7433".parse().unwrap()];
/// assert!(gates.ci_test_manifest(&default_bind).is_ok());
/// let refused = gates.ci_test_manifest(&["0.0.0.0:7433".parse().unwrap()]).unwrap_err();
/// assert_eq!(refused.non_loopback.len(), 1);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, ConfigSection)]
#[config(section = "debug", scope = Node, owner = "RFC-0054")]
pub struct DebugSection {
    /// Enables `POST /v1/syntropy/parity`: teacher-forced sequences with logit taps
    /// (RFC-0060 §5.5). Loopback listeners and the local socket only.
    #[config(reload = Hot, default = false)]
    pub parity_api: bool,
    /// Enables `BenchControls` on `POST /v1/syntropy/generate`: `ignore_eos = true` and an exact
    /// `gen_tokens` (RFC-0061 §3.1). Loopback listeners and the local socket only.
    #[config(reload = Hot, default = false)]
    pub bench_api: bool,
    /// Enables `GET /v1/syntropy/debug/telemetry`: the per-step record drain (RFC-0053 §2.5).
    /// Loopback listeners and the local socket only.
    #[config(reload = Hot, default = false)]
    pub telemetry_api: bool,
}

/// One gated part of the debug surface, named by the key that opens it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DebugApi {
    /// `POST /v1/syntropy/parity`, opened by `debug.parity_api`.
    Parity,
    /// `BenchControls` on `POST /v1/syntropy/generate`, opened by `debug.bench_api`.
    Bench,
    /// `GET /v1/syntropy/debug/telemetry`, opened by `debug.telemetry_api`.
    Telemetry,
}

impl DebugApi {
    /// Every gated part, in key order.
    pub const ALL: [DebugApi; 3] = [DebugApi::Parity, DebugApi::Bench, DebugApi::Telemetry];

    /// The key that opens it, such as `debug.parity_api`.
    pub const fn key(self) -> &'static str {
        match self {
            DebugApi::Parity => "debug.parity_api",
            DebugApi::Bench => "debug.bench_api",
            DebugApi::Telemetry => "debug.telemetry_api",
        }
    }

    /// What the key enables, as the warning names it.
    const fn enables(self) -> &'static str {
        match self {
            DebugApi::Parity => "`POST /v1/syntropy/parity`",
            DebugApi::Bench => "`BenchControls` on `POST /v1/syntropy/generate`",
            DebugApi::Telemetry => "`GET /v1/syntropy/debug/telemetry`",
        }
    }
}

impl fmt::Display for DebugApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.key())
    }
}

/// How a request reached the API: what [`DebugSection::allows`] checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Peer {
    /// The local socket: a Unix domain socket, or a named pipe on Windows. It is local by
    /// construction, and the API checks the OS peer credentials on it (RFC-0050 §11.1).
    LocalSocket,
    /// A TCP connection to one of the `api.bind` listeners.
    Tcp {
        /// The address the listener is bound to, as `api.bind` gives it.
        listener: IpAddr,
        /// The address of the peer that connected.
        remote: IpAddr,
    },
}

impl Peer {
    /// Whether the request came over the local socket, or over a loopback listener from a
    /// loopback peer. A wildcard listener (`0.0.0.0`, `::`) is not loopback, since ADR-0505
    /// item 2 counts it as a LAN bind, so a debug endpoint does not answer on it even to a
    /// local client. The peer is checked as well as the listener, so a host that accepts
    /// off-host packets addressed to `127.0.0.0/8` still gets 404 for them.
    pub fn is_loopback(&self) -> bool {
        match *self {
            Peer::LocalSocket => true,
            Peer::Tcp { listener, remote } => is_loopback(listener) && is_loopback(remote),
        }
    }
}

/// Whether `addr` is a loopback address: `127.0.0.0/8` or `::1`, including the IPv4-mapped
/// form (`::ffff:127.0.0.1`) a dual-stack socket reports for an IPv4 peer. The unspecified
/// addresses are not loopback.
pub fn is_loopback(addr: IpAddr) -> bool {
    addr.to_canonical().is_loopback()
}

impl DebugSection {
    /// Every key off: the default in every build profile.
    pub const OFF: DebugSection = DebugSection {
        parity_api: false,
        bench_api: false,
        telemetry_api: false,
    };

    /// The gates as `config` resolved them. A key is on only when it resolved to `true`. A key
    /// the registry does not hold, which no layer can then set, is off.
    pub fn from_config(config: &ResolvedConfig) -> Self {
        let on = |api: DebugApi| resolved_on(config, api).is_some();
        DebugSection {
            parity_api: on(DebugApi::Parity),
            bench_api: on(DebugApi::Bench),
            telemetry_api: on(DebugApi::Telemetry),
        }
    }

    /// Whether `api`'s key is on.
    pub const fn is_on(self, api: DebugApi) -> bool {
        match api {
            DebugApi::Parity => self.parity_api,
            DebugApi::Bench => self.bench_api,
            DebugApi::Telemetry => self.telemetry_api,
        }
    }

    /// Whether any key is on, which doctor CFG-003 and the dashboard banner report.
    pub fn any(self) -> bool {
        DebugApi::ALL.into_iter().any(|api| self.is_on(api))
    }

    /// The parts whose key is on, in key order.
    pub fn on(self) -> impl Iterator<Item = DebugApi> {
        DebugApi::ALL
            .into_iter()
            .filter(move |&api| self.is_on(api))
    }

    /// The loopback predicate: whether `api` answers a request that reached the API as `peer`.
    /// True only while `api`'s key is on and `peer` is the local socket or loopback
    /// ([`Peer::is_loopback`]). The API returns 404 whenever it is false, whatever key the
    /// caller holds (RFC-0054 §7.1, ADR-0505 item 7).
    pub fn allows(self, api: DebugApi, peer: &Peer) -> bool {
        self.is_on(api) && peer.is_loopback()
    }

    /// The manifest-signer predicate (RFC-0054 §7.1, RFC-0060 §6.4): whether the manifest
    /// loader may accept a `ConformanceManifest` signed by the CI test key, given the node's
    /// `api.bind` listeners. Both conditions must hold:
    /// - `debug.parity_api` or `debug.bench_api` is on (`debug.telemetry_api` does not count);
    /// - every listener in `api_bind` is loopback. The local socket always qualifies, so it is
    ///   not an argument, and an empty `api_bind` passes.
    ///
    /// The cluster plane's listeners are not API listeners and are not checked: multi-node
    /// parity binds them to the lab network. A release- or nightly-signed manifest does not
    /// go through this predicate.
    pub fn ci_test_manifest(self, api_bind: &[SocketAddr]) -> Result<(), ManifestSignerNotAllowed> {
        let no_gate = !(self.parity_api || self.bench_api);
        let non_loopback: Vec<SocketAddr> = api_bind
            .iter()
            .copied()
            .filter(|addr| !is_loopback(addr.ip()))
            .collect();
        if no_gate || !non_loopback.is_empty() {
            return Err(ManifestSignerNotAllowed {
                no_gate,
                non_loopback,
            });
        }
        Ok(())
    }
}

/// Why the manifest loader refuses a `ConformanceManifest` signed by the CI test key: error
/// `manifest_signer_not_allowed`, code 15 of the `Config` domain (RFC-0054 §8). Its
/// [`Display`](fmt::Display) form names each condition that failed, for the loader's log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestSignerNotAllowed {
    /// Neither `debug.parity_api` nor `debug.bench_api` is on.
    pub no_gate: bool,
    /// The `api.bind` listeners that are not loopback, in the order `api.bind` lists them.
    pub non_loopback: Vec<SocketAddr>,
}

impl ManifestSignerNotAllowed {
    /// The error's name in the `Config` domain (RFC-0054 §8).
    pub const NAME: &'static str = "manifest_signer_not_allowed";
    /// The error's code in the `Config` domain (RFC-0054 §8).
    pub const CODE: u16 = 15;
}

impl fmt::Display for ManifestSignerNotAllowed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: a conformance manifest signed by the CI test key is accepted only while \
             `debug.parity_api` or `debug.bench_api` is on and every `api.bind` listener is \
             loopback",
            Self::NAME
        )?;
        if self.no_gate {
            f.write_str("; neither `debug.parity_api` nor `debug.bench_api` is on")?;
        }
        if !self.non_loopback.is_empty() {
            f.write_str("; these `api.bind` listeners are not loopback: ")?;
            for (i, addr) in self.non_loopback.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{addr}")?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for ManifestSignerNotAllowed {}

/// The resolved entry of `api`'s key, if the key resolved to `true`.
fn resolved_on(config: &ResolvedConfig, api: DebugApi) -> Option<&crate::resolve::Resolved> {
    config
        .get(api.key())
        .filter(|resolved| resolved.value() == Some(&toml::Value::Boolean(true)))
}

/// One warning for each debug key that is on, at the layer that set it (RFC-0054 §7.1 "Visible
/// when on", §8 phase 4). [`ResolvedConfig::resolve_with`] adds them.
pub(crate) fn warnings(config: &ResolvedConfig) -> Vec<Problem> {
    DebugApi::ALL
        .into_iter()
        .filter_map(|api| {
            let resolved = resolved_on(config, api)?;
            Some(Problem::new(
                Some(resolved.origin.clone()),
                format!(
                    "`{}` is on: {} answers on loopback listeners and the local socket. It is a \
                     test-only surface (RFC-0054 §7.1); turn it off when the test is done",
                    api.key(),
                    api.enables()
                ),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    #[test]
    fn loopback_addresses() {
        let loopback = [
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            IpAddr::V4(Ipv4Addr::new(127, 1, 2, 3)),
            IpAddr::V6(Ipv6Addr::LOCALHOST),
            IpAddr::V6(Ipv4Addr::LOCALHOST.to_ipv6_mapped()),
        ];
        for addr in loopback {
            assert!(is_loopback(addr), "{addr}");
        }
        let not = [
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            IpAddr::V6(Ipv6Addr::UNSPECIFIED),
            IpAddr::V4(Ipv4Addr::new(192, 168, 1, 5)),
            IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            IpAddr::V4(Ipv4Addr::new(100, 64, 0, 1)),
            IpAddr::V6(Ipv4Addr::new(192, 168, 1, 5).to_ipv6_mapped()),
            // The deprecated IPv4-compatible form is not a loopback address.
            IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0x7f00, 1)),
            IpAddr::V6(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1)),
        ];
        for addr in not {
            assert!(!is_loopback(addr), "{addr}");
        }
    }
}
