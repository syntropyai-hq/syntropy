// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a node's CA may issue for (RFC-0050 §11.1.1): the node's DNS names and the subnets it
//! serves when the CA is generated, written into the CA certificate as critical X.509 name
//! constraints.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// An IP subnet: a network address with its host bits zero, and a prefix length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub struct Subnet {
    addr: IpAddr,
    prefix: u8,
}

impl Subnet {
    /// The subnet of `addr` with a `prefix`-bit network part, or `None` if `prefix` is longer
    /// than the address.
    pub fn new(addr: IpAddr, prefix: u8) -> Option<Subnet> {
        let addr = addr.to_canonical();
        let addr = match addr {
            IpAddr::V4(a) if prefix <= 32 => {
                let mask = u32::MAX.checked_shl(32 - u32::from(prefix)).unwrap_or(0);
                IpAddr::V4(Ipv4Addr::from(u32::from(a) & mask))
            }
            IpAddr::V6(a) if prefix <= 128 => {
                let mask = u128::MAX.checked_shl(128 - u32::from(prefix)).unwrap_or(0);
                IpAddr::V6(Ipv6Addr::from(u128::from(a) & mask))
            }
            _ => return None,
        };
        Some(Subnet { addr, prefix })
    }

    /// The subnet a bound interface serves: its /24 for IPv4, its /64 for IPv6.
    pub fn interface(addr: IpAddr) -> Subnet {
        let prefix = if addr.to_canonical().is_ipv4() {
            24
        } else {
            64
        };
        Subnet::new(addr, prefix).expect("the prefix fits the address")
    }

    /// The single address `addr`: a /32 or a /128, as an overlay address is served.
    pub fn host(addr: IpAddr) -> Subnet {
        let prefix = if addr.to_canonical().is_ipv4() {
            32
        } else {
            128
        };
        Subnet::new(addr, prefix).expect("the prefix fits the address")
    }

    /// The network address.
    pub fn addr(&self) -> IpAddr {
        self.addr
    }

    /// The prefix length.
    pub fn prefix(&self) -> u8 {
        self.prefix
    }

    /// Whether `ip` lies in this subnet.
    pub fn contains(&self, ip: IpAddr) -> bool {
        Subnet::new(ip, self.prefix).is_some_and(|s| s.addr == self.addr)
    }
}

impl fmt::Display for Subnet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.addr, self.prefix)
    }
}

impl FromStr for Subnet {
    type Err = String;

    fn from_str(s: &str) -> Result<Subnet, String> {
        let (addr, prefix) = s
            .split_once('/')
            .ok_or_else(|| format!("`{s}` has no prefix"))?;
        let addr: IpAddr = addr.parse().map_err(|_| format!("`{s}` has no address"))?;
        let prefix: u8 = prefix.parse().map_err(|_| format!("`{s}` has no prefix"))?;
        let subnet =
            Subnet::new(addr, prefix).ok_or_else(|| format!("`{s}`'s prefix is too long"))?;
        if subnet.addr != addr.to_canonical() {
            return Err(format!("`{s}` has host bits set"));
        }
        Ok(subnet)
    }
}

impl From<Subnet> for String {
    fn from(subnet: Subnet) -> String {
        subnet.to_string()
    }
}

impl TryFrom<String> for Subnet {
    type Error = String;

    fn try_from(s: String) -> Result<Subnet, String> {
        s.parse()
    }
}

/// Whether a local CA may cover `ip`: a private (RFC 1918), shared (RFC 6598, as overlays use),
/// link-local or unique-local (RFC 4193) address. Never a public address, so a leaked CA key
/// cannot impersonate a public host, and never loopback, which the API serves without TLS.
pub fn is_local(ip: IpAddr) -> bool {
    match ip.to_canonical() {
        IpAddr::V4(a) => {
            let shared = a.octets()[0] == 100 && (a.octets()[1] & 0xC0) == 64;
            a.is_private() || a.is_link_local() || shared
        }
        IpAddr::V6(a) => {
            let first = a.segments()[0];
            (first & 0xFE00) == 0xFC00 || (first & 0xFFC0) == 0xFE80
        }
    }
}

/// Why a scope could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScopeError {
    /// A name that is not a DNS host name: labels of letters, digits and hyphens, at most 63
    /// bytes each and 253 in all.
    InvalidName(String),
    /// No name a CA may be constrained to: every name was left out.
    NoNames,
}

impl fmt::Display for ScopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScopeError::InvalidName(name) => write!(f, "`{name}` is not a DNS host name"),
            ScopeError::NoNames => f.write_str(
                "the node has no DNS name of two or more labels, such as `<host>.local`, to \
                 constrain its CA to",
            ),
        }
    }
}

impl std::error::Error for ScopeError {}

/// A DNS host name in the form certificates carry: lowercase, without a trailing dot.
pub(super) fn host_name(name: &str) -> Result<String, ScopeError> {
    let lower = name.trim_end_matches('.').to_ascii_lowercase();
    let label_ok = |label: &str| {
        !label.is_empty()
            && label.len() <= 63
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    };
    let numeric = lower
        .split('.')
        .all(|l| l.bytes().all(|b| b.is_ascii_digit()));
    if lower.is_empty() || lower.len() > 253 || numeric || !lower.split('.').all(label_ok) {
        return Err(ScopeError::InvalidName(name.to_owned()));
    }
    Ok(lower)
}

/// What a node's CA may issue for, fixed when the CA is generated (RFC-0050 §11.1.1).
///
/// A dNSName constraint also permits every name below it (RFC 5280 §4.2.1.10), so a CA
/// constrained to `box.local` may issue for `x.box.local` too. That is why single-label names are
/// left out: a node named `box` would otherwise get a CA for the whole `.box` top-level domain.
/// The leaves this crate issues carry the scope's names exactly.
///
/// A limit: a name of two labels can still cover a public zone. A node name or
/// `api.server_names` entry that is a public suffix (`co.uk`) or a bare registrable domain
/// (`example.com`) gives the CA that whole zone, so `api.server_names` should hold the node's own
/// host names only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaScope {
    dns_names: Vec<String>,
    subnets: Vec<Subnet>,
}

/// What [`CaScope::new`] left out, for the caller to report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeftOut {
    /// Single-label names, which would constrain the CA to a whole top-level domain.
    pub names: Vec<String>,
    /// Public or loopback addresses, which no local CA covers.
    pub addresses: Vec<IpAddr>,
}

impl CaScope {
    /// The scope of a CA generated now.
    /// - **Names:** the node's DNS names: `node.name`, the OS host name, `<host>.local` and
    ///   `api.server_names`.
    /// - **Subnets:** each bound interface address's /24 or /64, and each overlay address as a
    ///   /32 or /128.
    ///
    /// Single-label names and addresses no local CA covers ([`is_local`]) are left out, and
    /// returned in [`LeftOut`]. A name that is not a host name is an error, and so is a scope
    /// left with no name.
    pub fn new<'n>(
        names: impl IntoIterator<Item = &'n str>,
        interfaces: &[IpAddr],
        overlay: &[IpAddr],
    ) -> Result<(CaScope, LeftOut), ScopeError> {
        let mut left_out = LeftOut::default();
        let mut dns_names = Vec::new();
        for name in names {
            let name = host_name(name)?;
            if name.contains('.') {
                dns_names.push(name);
            } else {
                left_out.names.push(name);
            }
        }
        let mut subnets = Vec::new();
        let served = interfaces
            .iter()
            .map(|ip| (ip, Subnet::interface(*ip)))
            .chain(overlay.iter().map(|ip| (ip, Subnet::host(*ip))));
        for (ip, subnet) in served {
            if is_local(*ip) {
                subnets.push(subnet);
            } else {
                left_out.addresses.push(*ip);
            }
        }
        dns_names.sort();
        dns_names.dedup();
        subnets.sort();
        subnets.dedup();
        left_out.names.sort();
        left_out.names.dedup();
        if dns_names.is_empty() {
            return Err(ScopeError::NoNames);
        }
        Ok((CaScope { dns_names, subnets }, left_out))
    }

    /// The DNS names, lowercase and sorted.
    pub fn dns_names(&self) -> &[String] {
        &self.dns_names
    }

    /// The subnets, sorted. Empty when the node serves no local address, and then the CA may
    /// issue for no IP address at all.
    pub fn subnets(&self) -> &[Subnet] {
        &self.subnets
    }

    /// Whether a leaf may carry `name`: one of the scope's names, exactly.
    pub fn permits_name(&self, name: &str) -> bool {
        host_name(name).is_ok_and(|name| self.dns_names.binary_search(&name).is_ok())
    }

    /// Whether a leaf may carry `ip`: an address in one of the scope's subnets.
    pub fn permits_ip(&self, ip: IpAddr) -> bool {
        self.subnets.iter().any(|subnet| subnet.contains(ip))
    }

    /// A scope read back from storage: names and subnets as written, re-checked.
    pub(super) fn checked(self) -> Option<CaScope> {
        let names_ok = self
            .dns_names
            .iter()
            .all(|n| host_name(n).as_deref() == Ok(n) && n.contains('.'));
        let sorted = self.dns_names.windows(2).all(|w| w[0] < w[1])
            && self.subnets.windows(2).all(|w| w[0] < w[1]);
        let local = self.subnets.iter().all(|s| is_local(s.addr));
        (names_ok && sorted && local && !self.dns_names.is_empty()).then_some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn interfaces_serve_their_24_or_64_and_overlays_their_address() {
        assert_eq!(
            Subnet::interface(ip("192.168.1.37")).to_string(),
            "192.168.1.0/24"
        );
        assert_eq!(
            Subnet::interface(ip("fd12:3456:789a:1::5")).to_string(),
            "fd12:3456:789a:1::/64"
        );
        assert_eq!(
            Subnet::host(ip("100.101.102.103")).to_string(),
            "100.101.102.103/32"
        );
        assert_eq!(
            Subnet::host(ip("fd7a:115c:a1e0::1")).to_string(),
            "fd7a:115c:a1e0::1/128"
        );
        // An IPv4-mapped IPv6 address is its IPv4 address.
        assert_eq!(
            Subnet::interface(ip("::ffff:10.1.2.3")).to_string(),
            "10.1.2.0/24"
        );
        let subnet = Subnet::interface(ip("10.1.2.3"));
        assert!(subnet.contains(ip("10.1.2.250")));
        assert!(!subnet.contains(ip("10.1.3.1")));
        assert!(!subnet.contains(ip("fd00::1")));
        assert_eq!(Subnet::new(ip("10.0.0.1"), 33), None);
        assert_eq!("10.1.2.0/24".parse(), Ok(subnet));
        assert!("10.1.2.3/24".parse::<Subnet>().is_err());
    }

    #[test]
    fn only_local_addresses_are_covered() {
        for local in [
            "10.0.0.1",
            "172.16.5.4",
            "192.168.0.10",
            "100.64.0.1",
            "100.127.255.254",
            "169.254.1.1",
            "fd00::1",
            "fe80::1",
        ] {
            assert!(is_local(ip(local)), "{local}");
        }
        for other in [
            "8.8.8.8",
            "100.128.0.1",
            "172.32.0.1",
            "127.0.0.1",
            "::1",
            "2001:db8::1",
            "2606:4700::1111",
        ] {
            assert!(!is_local(ip(other)), "{other}");
        }
    }

    #[test]
    fn a_scope_leaves_out_single_labels_and_public_or_loopback_addresses() {
        let (scope, left_out) = CaScope::new(
            ["Box", "box.local", "BOX.local.", "llm.home.arpa"],
            &[ip("192.168.1.37"), ip("203.0.113.9"), ip("127.0.0.1")],
            &[ip("100.101.102.103")],
        )
        .unwrap();
        assert_eq!(scope.dns_names(), ["box.local", "llm.home.arpa"]);
        assert_eq!(
            scope
                .subnets()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["100.101.102.103/32", "192.168.1.0/24"]
        );
        assert_eq!(left_out.names, ["box"]);
        assert_eq!(left_out.addresses, [ip("203.0.113.9"), ip("127.0.0.1")]);
        assert!(scope.permits_name("BOX.LOCAL"));
        assert!(!scope.permits_name("x.box.local"));
        assert!(!scope.permits_name("example.com"));
        assert!(scope.permits_ip(ip("192.168.1.200")));
        assert!(!scope.permits_ip(ip("192.168.2.1")));
        assert!(!scope.permits_ip(ip("8.8.8.8")));
    }

    #[test]
    fn names_must_be_host_names_and_one_must_remain() {
        for bad in [
            "",
            "-box.local",
            "box_1.local",
            "*.box.local",
            "10.0.0.1",
            "a..b",
        ] {
            assert_eq!(
                CaScope::new([bad], &[], &[]),
                Err(ScopeError::InvalidName(bad.to_owned())),
                "{bad}"
            );
        }
        assert_eq!(CaScope::new(["box"], &[], &[]), Err(ScopeError::NoNames));
        assert_eq!(
            CaScope::new([&"a".repeat(64)[..]], &[], &[]),
            Err(ScopeError::InvalidName("a".repeat(64)))
        );
    }
}
