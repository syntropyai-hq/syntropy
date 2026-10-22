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
//! A listing fails only when a source it needs cannot be read at all. A line or interface a
//! parser cannot read costs that line or that interface, and becomes a warning in the
//! [`Listing`], so one odd interface never hides the others.
//!
//! RFC-0030 §7.2's note of 2026-10-22 records these sources and the rules beyond its table.
//! Nothing here uses `unsafe`.

pub mod linux;
pub mod macos;
pub mod windows;

use std::collections::{BTreeMap, BTreeSet};
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

/// What a gatherer read: the interfaces, and a warning for each line or interface it could not
/// read.
pub type Gathered = (Vec<Observed>, Vec<IfaceError>);

/// Interface ordinals by name, for `Endpoint::Nic { node, ordinal }` (RFC-0030 §7.1: stable
/// while the interface name is).
///
/// - A name keeps its ordinal for as long as the table holds the name.
/// - No two names held at once share an ordinal. A decoded table that would break this is
///   refused.
/// - A new name gets the next free ordinal after the last one given out, wrapping at 65,536.
///   So an ordinal freed by [`Ordinals::retire`] is given out again only after every other
///   free one has been.
/// - The table holds at most 65,536 names at once. A host that makes a new name for each
///   container (`veth…`) fills it unless its owner retires names it no longer sees. The daemon
///   keeps one table, and may persist it (it is serde data).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "OrdinalsData", into = "OrdinalsData")]
pub struct Ordinals {
    by_name: BTreeMap<String, u16>,
    held: BTreeSet<u16>,
    next: u16,
}

/// [`Ordinals`] as stored: the names and the next ordinal to try.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct OrdinalsData {
    by_name: BTreeMap<String, u16>,
    next: u16,
}

impl TryFrom<OrdinalsData> for Ordinals {
    type Error = String;

    fn try_from(data: OrdinalsData) -> Result<Ordinals, String> {
        let mut held = BTreeSet::new();
        for (name, &ordinal) in &data.by_name {
            if !held.insert(ordinal) {
                return Err(format!(
                    "ordinal {ordinal} is held by two names, one of them {name:?}"
                ));
            }
        }
        Ok(Ordinals {
            by_name: data.by_name,
            held,
            next: data.next,
        })
    }
}

impl From<Ordinals> for OrdinalsData {
    fn from(ordinals: Ordinals) -> OrdinalsData {
        OrdinalsData {
            by_name: ordinals.by_name,
            next: ordinals.next,
        }
    }
}

impl Ordinals {
    /// An empty table.
    pub fn new() -> Ordinals {
        Ordinals::default()
    }

    /// The ordinal of `name`: the one it holds, or the next free one. `None` when the table
    /// already holds 65,536 names.
    pub fn ordinal(&mut self, name: &str) -> Option<u16> {
        if let Some(&ordinal) = self.by_name.get(name) {
            return Some(ordinal);
        }
        if self.held.len() > usize::from(u16::MAX) {
            return None;
        }
        let mut candidate = self.next;
        while self.held.contains(&candidate) {
            candidate = candidate.wrapping_add(1);
        }
        self.next = candidate.wrapping_add(1);
        self.held.insert(candidate);
        self.by_name.insert(name.to_owned(), candidate);
        Some(candidate)
    }

    /// Lets go of `name`, so its ordinal can go to another name later. Returns the ordinal it
    /// held.
    pub fn retire(&mut self, name: &str) -> Option<u16> {
        let ordinal = self.by_name.remove(name)?;
        self.held.remove(&ordinal);
        Some(ordinal)
    }

    /// How many names the table holds.
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// Whether the table holds no names.
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

/// What a listing read and could not read.
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
    /// The ordinal table holds 65,536 names, so this interface got none and was left out.
    OrdinalsFull {
        /// The interface.
        name: String,
    },
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
            IfaceError::OrdinalsFull { name } => write!(
                f,
                "{name} is left out: the ordinal table holds 65,536 names"
            ),
        }
    }
}

impl std::error::Error for IfaceError {}

/// A listing: the interfaces, sorted by name, and a warning for each line or interface that
/// could not be read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Listing {
    /// The interfaces.
    pub ifaces: Vec<IfaceInfo>,
    /// What could not be read. Each costs only its own line or interface.
    pub warnings: Vec<IfaceError>,
}

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

/// The listing for `gathered`, sorted by name, each interface with its ordinal from
/// `ordinals`. An interface that gets no ordinal is left out, with a warning.
pub fn build((mut observed, mut warnings): Gathered, ordinals: &mut Ordinals) -> Listing {
    observed.sort_by(|a, b| a.name.cmp(&b.name));
    let mut ifaces = Vec::with_capacity(observed.len());
    for o in observed {
        let Some(ordinal) = ordinals.ordinal(&o.name) else {
            warnings.push(IfaceError::OrdinalsFull { name: o.name });
            continue;
        };
        ifaces.push(IfaceInfo {
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
        });
    }
    Listing { ifaces, warnings }
}

/// This host's interfaces, classified by RFC-0030 §7.2, with ordinals from `ordinals`. It
/// fails only when a source it needs cannot be read at all.
pub fn interfaces(ordinals: &mut Ordinals) -> Result<Listing, IfaceError> {
    Ok(build(gather()?, ordinals))
}

#[cfg(target_os = "linux")]
fn gather() -> Result<Gathered, IfaceError> {
    linux::gather()
}

#[cfg(target_os = "macos")]
fn gather() -> Result<Gathered, IfaceError> {
    macos::gather()
}

#[cfg(target_os = "windows")]
fn gather() -> Result<Gathered, IfaceError> {
    windows::gather()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn gather() -> Result<Gathered, IfaceError> {
    Err(IfaceError::source(
        "interface enumeration",
        "not built for this OS",
    ))
}

/// The text of a command's standard output. Output that is not UTF-8 is read with
/// replacement characters and a warning, so one adapter's name cannot fail a listing.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
pub(crate) fn output_text(
    command: &mut std::process::Command,
    shown: &str,
    warnings: &mut Vec<IfaceError>,
) -> Result<String, IfaceError> {
    let output = command.output().map_err(|e| IfaceError::source(shown, e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let first = stderr.lines().next().unwrap_or("").trim().to_owned();
        return Err(IfaceError::source(
            shown,
            format!("exited with {}: {first}", output.status),
        ));
    }
    match String::from_utf8(output.stdout) {
        Ok(text) => Ok(text),
        Err(e) => {
            warnings.push(IfaceError::source(
                shown,
                "output is not UTF-8; read with replacement characters",
            ));
            Ok(String::from_utf8_lossy(e.as_bytes()).into_owned())
        }
    }
}

/// Runs `program` with `args`, by name, and returns its standard output as text.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) fn run(
    program: &str,
    args: &[&str],
    warnings: &mut Vec<IfaceError>,
) -> Result<String, IfaceError> {
    let shown = format!("`{program} {}`", args.join(" "));
    output_text(
        std::process::Command::new(program).args(args),
        &shown,
        warnings,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinals_are_stable_by_name() {
        let mut ordinals = Ordinals::new();
        assert_eq!(ordinals.ordinal("eth0"), Some(0));
        assert_eq!(ordinals.ordinal("wlan0"), Some(1));
        assert_eq!(ordinals.ordinal("eth0"), Some(0));
        assert_eq!(ordinals.ordinal("tailscale0"), Some(2));
        let json = serde_json::to_string(&ordinals).unwrap();
        let mut back: Ordinals = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ordinal("wlan0"), Some(1));
        assert_eq!(back.ordinal("en9"), Some(3));
        assert_eq!(back.len(), 4);
    }

    /// A decoded table never hands out an ordinal a name already holds, whatever its cursor
    /// says, and a table with two names on one ordinal does not decode.
    #[test]
    fn a_decoded_table_never_reuses_a_held_ordinal() {
        let mut stale: Ordinals =
            serde_json::from_str(r#"{"by_name":{"eth0":0,"wlan0":1},"next":0}"#).unwrap();
        assert_eq!(stale.ordinal("en9"), Some(2));
        assert_eq!(stale.ordinal("eth0"), Some(0));
        let doubled =
            serde_json::from_str::<Ordinals>(r#"{"by_name":{"eth0":3,"wlan0":3},"next":0}"#);
        assert!(doubled.is_err(), "{doubled:?}");
    }

    /// A retired name's ordinal goes out again only after every other free one has.
    #[test]
    fn retired_ordinals_come_back_last() {
        let mut ordinals = Ordinals::new();
        for name in ["a", "b", "c"] {
            ordinals.ordinal(name);
        }
        assert_eq!(ordinals.retire("b"), Some(1));
        assert_eq!(ordinals.retire("b"), None);
        assert_eq!(ordinals.ordinal("d"), Some(3), "the cursor moves on past 1");
        assert_eq!(ordinals.ordinal("a"), Some(0));
        let mut wrapped: Ordinals =
            serde_json::from_str(r#"{"by_name":{"x":0,"y":65535},"next":65535}"#).unwrap();
        assert_eq!(
            wrapped.ordinal("z"),
            Some(1),
            "it wraps and skips held ordinals"
        );
    }

    /// When every ordinal is held, a new interface is left out with a warning and the rest of
    /// the listing stands.
    #[test]
    fn a_full_table_leaves_out_only_the_new_interface() {
        let mut ordinals = Ordinals::new();
        for i in 0..=u32::from(u16::MAX) {
            assert!(ordinals.ordinal(&format!("veth{i}")).is_some());
        }
        assert_eq!(ordinals.ordinal("eth9"), None);
        let observed = |name: &str| Observed {
            name: name.to_owned(),
            ..Observed::default()
        };
        let listing = build(
            (vec![observed("veth7"), observed("eth9")], Vec::new()),
            &mut ordinals,
        );
        assert_eq!(listing.ifaces.len(), 1);
        assert_eq!(listing.ifaces[0].name, "veth7");
        assert_eq!(
            listing.warnings,
            [IfaceError::OrdinalsFull {
                name: String::from("eth9")
            }]
        );
        ordinals.retire("veth0");
        assert_eq!(ordinals.ordinal("eth9"), Some(0));
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
        let warning = IfaceError::parse("test", "x");
        let listing = build(
            (
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
                vec![warning.clone()],
            ),
            &mut ordinals,
        );
        let names: Vec<(&str, u16)> = listing
            .ifaces
            .iter()
            .map(|r| (r.name.as_str(), r.ordinal))
            .collect();
        assert_eq!(names, [("aa0", 1), ("zz9", 0)]);
        assert_eq!(listing.ifaces[0].medium, NetMedium::WiFi);
        assert_eq!(
            (listing.ifaces[0].rdma.clone(), listing.ifaces[0].tb_peer),
            (None, None)
        );
        assert_eq!(listing.warnings, [warning]);
    }
}
