// SPDX-License-Identifier: MIT OR Apache-2.0

//! Network interface enumeration and medium classification (RFC-0030 §7.1–§7.2). Owner:
//! RFC-0030.
//!
//! [`interfaces`] lists this host's interfaces as `core`'s [`IfaceInfo`] records. Each OS has
//! two parts:
//! - a gatherer, which reads the OS's own sources: on Linux sysfs, `/proc/net` and
//!   `ip -o addr`; on macOS `networksetup`, `ifconfig` and `route`; on Windows PowerShell's
//!   `Get-NetAdapter` family;
//! - a parser and classifier over what the gatherer read. These compile on every OS, so each
//!   OS's rules are tested on any host from recorded output.
//!
//! RFC-0030 §7.2's note of 2026-10-22 records these sources and the rules beyond its table.
//! Nothing here uses `unsafe`.

pub mod linux;
pub mod macos;
pub mod windows;

use std::collections::BTreeMap;
use std::fmt;
use std::net::IpAddr;

use serde::{Deserialize, Serialize};
use syntropy_core::NetMedium;
use syntropy_core::resource::{IfaceFlags, IfaceInfo, IpNet};

/// One interface as a gatherer saw and classified it, before it has an ordinal.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Observed {
    /// The OS's name for it.
    pub name: String,
    /// Its medium, by RFC-0030 §7.2.
    pub medium: NetMedium,
    /// The bits the OS reports. `TRUSTED_FABRIC` is never set here: the node's configuration
    /// sets it (RFC-0034 §9).
    pub flags: IfaceFlags,
    /// Its addresses.
    pub addrs: Vec<IpNet>,
    /// Its MTU in bytes, or 0 if the OS did not say.
    pub mtu: u32,
    /// The negotiated rate in Mbit/s, if the OS reports one.
    pub speed_mbps: Option<u32>,
    /// The bridge it is a member of.
    pub bridge: Option<String>,
}

/// Interface ordinals by name, for `Endpoint::Nic { node, ordinal }`: an ordinal is stable
/// while its name is, and never reused for another name. The daemon keeps one for its
/// lifetime, and may persist it (it is serde data).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ordinals {
    by_name: BTreeMap<String, u16>,
    next: u32,
}

impl Ordinals {
    /// No ordinals given yet.
    pub fn new() -> Ordinals {
        Ordinals::default()
    }

    /// The ordinal of `name`: the one it had, or the next unused one. `None` once all 65,536
    /// have been given out.
    pub fn ordinal(&mut self, name: &str) -> Option<u16> {
        if let Some(&ordinal) = self.by_name.get(name) {
            return Some(ordinal);
        }
        let ordinal = u16::try_from(self.next).ok()?;
        self.next += 1;
        self.by_name.insert(name.to_owned(), ordinal);
        Some(ordinal)
    }
}

/// Why enumeration failed. It names the source, never guesses past it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IfaceError {
    /// A source could not be read or run.
    Source {
        /// The file, directory or command.
        source: String,
        /// Why.
        why: String,
    },
    /// A source's output had a line this module does not read.
    Parse {
        /// The file, directory or command.
        source: String,
        /// The line, as read.
        line: String,
    },
    /// More than 65,536 interface names have been seen.
    OrdinalsExhausted,
}

impl IfaceError {
    pub(crate) fn source(source: impl Into<String>, why: impl fmt::Display) -> IfaceError {
        IfaceError::Source {
            source: source.into(),
            why: why.to_string(),
        }
    }

    pub(crate) fn parse(source: impl Into<String>, line: &str) -> IfaceError {
        IfaceError::Parse {
            source: source.into(),
            line: line.to_owned(),
        }
    }
}

impl fmt::Display for IfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IfaceError::Source { source, why } => write!(f, "cannot read {source}: {why}"),
            IfaceError::Parse { source, line } => write!(f, "{source}: unexpected line {line:?}"),
            IfaceError::OrdinalsExhausted => f.write_str("more than 65,536 interface names seen"),
        }
    }
}

impl std::error::Error for IfaceError {}

/// The address ranges a tailnet assigns (RFC-0030 §7.2's overlay column): its IPv4 range in
/// the shared address space, 100.64.0.0/10, and its IPv6 prefix, fd7a:115c:a1e0::/48.
pub fn is_tailnet(addr: IpAddr) -> bool {
    const V4: IpAddr = IpAddr::V4(std::net::Ipv4Addr::new(100, 64, 0, 0));
    const V6: IpAddr = IpAddr::V6(std::net::Ipv6Addr::new(
        0xfd7a, 0x115c, 0xa1e0, 0, 0, 0, 0, 0,
    ));
    [(V4, 10), (V6, 48)]
        .iter()
        .any(|&(net, prefix)| IpNet::new(net, prefix).is_some_and(|range| range.contains(addr)))
}

/// The records for `observed`, sorted by name, each with its ordinal from `ordinals`.
pub fn build(
    mut observed: Vec<Observed>,
    ordinals: &mut Ordinals,
) -> Result<Vec<IfaceInfo>, IfaceError> {
    observed.sort_by(|a, b| a.name.cmp(&b.name));
    observed
        .into_iter()
        .map(|o| {
            let ordinal = ordinals
                .ordinal(&o.name)
                .ok_or(IfaceError::OrdinalsExhausted)?;
            Ok(IfaceInfo {
                ordinal,
                name: o.name,
                medium: o.medium,
                addrs: o.addrs.into_iter().collect(),
                mtu: o.mtu,
                speed_mbps: o.speed_mbps,
                flags: o.flags,
                bridge: o.bridge,
                rdma: None,
                tb_peer: None,
            })
        })
        .collect()
}

/// This host's interfaces, classified by RFC-0030 §7.2, with ordinals from `ordinals`.
pub fn interfaces(ordinals: &mut Ordinals) -> Result<Vec<IfaceInfo>, IfaceError> {
    build(gather()?, ordinals)
}

#[cfg(target_os = "linux")]
fn gather() -> Result<Vec<Observed>, IfaceError> {
    linux::gather()
}

#[cfg(target_os = "macos")]
fn gather() -> Result<Vec<Observed>, IfaceError> {
    macos::gather()
}

#[cfg(target_os = "windows")]
fn gather() -> Result<Vec<Observed>, IfaceError> {
    windows::gather()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn gather() -> Result<Vec<Observed>, IfaceError> {
    Err(IfaceError::source(
        "interface enumeration",
        "not built for this OS",
    ))
}

/// Runs `program` with `args` and returns its standard output as text.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
pub(crate) fn run(program: &str, args: &[&str]) -> Result<String, IfaceError> {
    let shown = || format!("`{program} {}`", args.join(" "));
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .map_err(|e| IfaceError::source(shown(), e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let first = stderr.lines().next().unwrap_or("").trim().to_owned();
        return Err(IfaceError::source(
            shown(),
            format!("exited with {}: {first}", output.status),
        ));
    }
    String::from_utf8(output.stdout).map_err(|_| IfaceError::source(shown(), "output is not UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinals_are_stable_by_name_and_never_reused() {
        let mut ordinals = Ordinals::new();
        assert_eq!(ordinals.ordinal("eth0"), Some(0));
        assert_eq!(ordinals.ordinal("wlan0"), Some(1));
        assert_eq!(ordinals.ordinal("eth0"), Some(0));
        assert_eq!(ordinals.ordinal("tailscale0"), Some(2));
        let json = serde_json::to_string(&ordinals).unwrap();
        let mut back: Ordinals = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ordinal("wlan0"), Some(1));
        assert_eq!(back.ordinal("en9"), Some(3));
    }

    #[test]
    fn tailnet_ranges() {
        for addr in [
            "100.64.0.1",
            "100.101.102.103",
            "fd7a:115c:a1e0::5",
            "fd7a:115c:a1e0:ab::1",
        ] {
            assert!(is_tailnet(addr.parse().unwrap()), "{addr}");
        }
        for addr in [
            "100.63.255.255",
            "100.128.0.0",
            "192.168.1.4",
            "fd7a:115c:a1e1::1",
            "fe80::1",
        ] {
            assert!(!is_tailnet(addr.parse().unwrap()), "{addr}");
        }
    }

    #[test]
    fn build_sorts_by_name_and_assigns_ordinals() {
        let mut ordinals = Ordinals::new();
        ordinals.ordinal("zz9");
        let records = build(
            vec![
                Observed {
                    name: String::from("zz9"),
                    medium: NetMedium::Overlay,
                    ..Observed::default()
                },
                Observed {
                    name: String::from("aa0"),
                    medium: NetMedium::WiFi,
                    mtu: 1500,
                    ..Observed::default()
                },
            ],
            &mut ordinals,
        )
        .unwrap();
        let names: Vec<(&str, u16)> = records
            .iter()
            .map(|r| (r.name.as_str(), r.ordinal))
            .collect();
        assert_eq!(names, [("aa0", 1), ("zz9", 0)]);
        assert_eq!(records[0].medium, NetMedium::WiFi);
        assert_eq!((records[0].rdma.clone(), records[0].tb_peer), (None, None));
    }
}
