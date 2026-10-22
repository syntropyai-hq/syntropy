// SPDX-License-Identifier: MIT OR Apache-2.0

//! Network interface records: RFC-0030 §7.1's `IfaceInfo` and the types it holds. RFC-0001
//! §5.2's `LinkInfo::interface` names it. Owner: RFC-0030.
//!
//! `net` fills these records from the OS (DIST-E6-T01); RDMA ports (DIST-E6-T02) and
//! Thunderbolt peers (DIST-E6-T04) are filled by later tasks. Every field after `name` has a
//! serde default, so a record written before a field existed still decodes.

use std::net::IpAddr;

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ids::{NetMedium, NodeId, RdmaKind};

/// One network interface of a node (RFC-0030 §7.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IfaceInfo {
    /// The interface's ordinal in `Endpoint::Nic { node, ordinal }`, stable while its name is.
    pub ordinal: u16,
    /// The OS's name for it, such as `en0`, `eth0`, `Ethernet 2` or `thunderbolt0`.
    pub name: String,
    /// What carries it (RFC-0030 §7.2).
    #[serde(default)]
    pub medium: NetMedium,
    /// Its addresses, IPv6 link-local ones with their scope.
    #[serde(default)]
    pub addrs: SmallVec<[IpNet; 4]>,
    /// The MTU in bytes.
    #[serde(default)]
    pub mtu: u32,
    /// The rate the OS reports as negotiated, in Mbit/s. A sanity bound only: the solver uses
    /// measured rates (RFC-0010).
    #[serde(default)]
    pub speed_mbps: Option<u32>,
    /// State and role bits.
    #[serde(default)]
    pub flags: IfaceFlags,
    /// The bridge it is a member of, such as `bridge0`.
    #[serde(default)]
    pub bridge: Option<String>,
    /// The RDMA port on this interface, if any (DIST-E6-T02).
    #[serde(default)]
    pub rdma: Option<RdmaPortInfo>,
    /// For Thunderbolt IP or RDMA: the node at the other end of the cable (DIST-E6-T04).
    #[serde(default)]
    pub tb_peer: Option<NodeId>,
}

bitflags! {
    /// State and role bits of an [`IfaceInfo`] (RFC-0030 §7.1).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct IfaceFlags: u32 {
        /// The interface is up.
        const UP = 1 << 0;
        /// It supports multicast, which discovery's beacons need (RFC-0030 §4).
        const MULTICAST = 1 << 1;
        /// The OS reports it as metered. Unset both when the OS reports it unmetered and when
        /// the OS does not say, so an unset bit does not prove a link is free to use.
        const METERED = 1 << 2;
        /// It is a member of a bridge, named in [`IfaceInfo::bridge`].
        const BRIDGE_MEMBER = 1 << 3;
        /// It is a bridge with Thunderbolt members, such as macOS's `bridge0`.
        const TB_BRIDGE = 1 << 4;
        /// The node's configuration names it a trusted fabric (RFC-0034 §9).
        const TRUSTED_FABRIC = 1 << 5;
        /// It carries the default route.
        const DEFAULT_ROUTE = 1 << 6;
    }
}

/// An address with its prefix length, and for an IPv6 link-local address its scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IpNet {
    /// The address.
    pub addr: IpAddr,
    /// The prefix length: at most 32 for IPv4 and 128 for IPv6.
    pub prefix_len: u8,
    /// The scope (zone) of an IPv6 link-local address: the interface index it is reached
    /// through.
    #[serde(default)]
    pub scope_id: Option<u32>,
}

impl IpNet {
    /// An address and prefix, or `None` if the prefix is longer than the address.
    pub fn new(addr: IpAddr, prefix_len: u8) -> Option<IpNet> {
        (prefix_len <= max_prefix(addr)).then_some(IpNet {
            addr,
            prefix_len,
            scope_id: None,
        })
    }

    /// The same network with `scope_id` as its scope.
    pub fn with_scope(self, scope_id: u32) -> IpNet {
        IpNet {
            scope_id: Some(scope_id),
            ..self
        }
    }

    /// Whether `addr` lies in this network: the same family, and the same leading
    /// `prefix_len` bits. A /0 matches every address of its family; an over-long prefix
    /// matches nothing. It compares addresses only and ignores `scope_id`, so a link-local
    /// network matches its addresses on every interface; a caller that needs the zone compares
    /// `scope_id` itself.
    pub fn contains(&self, addr: IpAddr) -> bool {
        if self.prefix_len > max_prefix(self.addr) {
            return false;
        }
        match (self.addr, addr) {
            (IpAddr::V4(net), IpAddr::V4(a)) => same_prefix(
                u128::from(u32::from(net)),
                u128::from(u32::from(a)),
                32,
                self.prefix_len,
            ),
            (IpAddr::V6(net), IpAddr::V6(a)) => {
                same_prefix(u128::from(net), u128::from(a), 128, self.prefix_len)
            }
            _ => false,
        }
    }
}

fn max_prefix(addr: IpAddr) -> u8 {
    match addr {
        IpAddr::V4(_) => 32,
        IpAddr::V6(_) => 128,
    }
}

/// Whether `a` and `b`, each `width` bits wide, agree in their leading `prefix` bits.
fn same_prefix(a: u128, b: u128, width: u32, prefix: u8) -> bool {
    let prefix = u32::from(prefix);
    if prefix == 0 {
        return true;
    }
    let shift = width - prefix;
    (a >> shift) == (b >> shift)
}

/// An RDMA port and the interface it runs on (RFC-0030 §7.1, RFC-0031 §5.2). DIST-E6-T02
/// fills it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdmaPortInfo {
    /// The RDMA device, such as `mlx5_0` or `rdma_en2`.
    pub device: String,
    /// The port on that device.
    pub port: u8,
    /// The RDMA fabric.
    pub kind: RdmaKind,
    /// The GID table index used for connections.
    #[serde(default)]
    pub gid_index: u8,
    /// The port is active.
    #[serde(default)]
    pub active: bool,
    /// The link layer the port runs over.
    pub link_layer: RdmaLinkLayer,
}

/// The link layer under an RDMA port (RFC-0030 §7.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RdmaLinkLayer {
    /// Ethernet, for RoCE.
    Ethernet,
    /// InfiniBand.
    InfiniBand,
    /// Thunderbolt.
    Thunderbolt,
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, Ipv6Addr};

    use super::*;

    fn net(text: &str, prefix: u8) -> IpNet {
        IpNet::new(text.parse().expect("an address"), prefix).expect("a valid prefix")
    }

    #[test]
    fn prefixes_longer_than_the_address_are_rejected() {
        assert!(IpNet::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 33).is_none());
        assert!(IpNet::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 129).is_none());
        assert!(IpNet::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 32).is_some());
        assert!(IpNet::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 128).is_some());
    }

    #[test]
    fn contains_compares_the_leading_bits() {
        let cgnat = net("100.64.0.0", 10);
        assert!(cgnat.contains("100.64.0.1".parse().unwrap()));
        assert!(cgnat.contains("100.127.255.254".parse().unwrap()));
        assert!(!cgnat.contains("100.128.0.1".parse().unwrap()));
        assert!(!cgnat.contains("fd7a:115c:a1e0::1".parse().unwrap()));
        let ula = net("fd7a:115c:a1e0::", 48);
        assert!(ula.contains("fd7a:115c:a1e0:ab12::1".parse().unwrap()));
        assert!(!ula.contains("fd7a:115c:a1e1::1".parse().unwrap()));
        assert!(net("0.0.0.0", 0).contains("203.0.113.9".parse().unwrap()));
        assert!(net("192.0.2.7", 32).contains("192.0.2.7".parse().unwrap()));
        assert!(!net("192.0.2.7", 32).contains("192.0.2.8".parse().unwrap()));
        let over_long = IpNet {
            addr: "192.0.2.0".parse().unwrap(),
            prefix_len: 40,
            scope_id: None,
        };
        assert!(!over_long.contains("192.0.2.0".parse().unwrap()));
    }

    /// The prefix lengths at both ends of each family. A /0 shifts by the full width without
    /// its guard, which overflows for IPv6, so these pin the guard.
    #[test]
    fn contains_at_the_prefix_bounds() {
        let v4_any = net("0.0.0.0", 0);
        assert!(v4_any.contains("0.0.0.0".parse().unwrap()));
        assert!(v4_any.contains("255.255.255.255".parse().unwrap()));
        assert!(!v4_any.contains("::1".parse().unwrap()));
        let v4_host = net("198.51.100.7", 32);
        assert!(v4_host.contains("198.51.100.7".parse().unwrap()));
        assert!(!v4_host.contains("198.51.100.6".parse().unwrap()));
        let v6_any = net("::", 0);
        assert!(v6_any.contains("::".parse().unwrap()));
        assert!(v6_any.contains("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff".parse().unwrap()));
        assert!(v6_any.contains("fe80::1".parse().unwrap()));
        assert!(!v6_any.contains("0.0.0.0".parse().unwrap()));
        let v6_host = net("2001:db8::42", 128);
        assert!(v6_host.contains("2001:db8::42".parse().unwrap()));
        assert!(!v6_host.contains("2001:db8::43".parse().unwrap()));
    }

    /// `contains` ignores the scope: a link-local network matches the address whatever zone
    /// either side names.
    #[test]
    fn contains_ignores_the_scope() {
        let scoped = net("fe80::", 64).with_scope(3);
        assert!(scoped.contains("fe80::9".parse().unwrap()));
        assert!(net("fe80::", 64).contains("fe80::9".parse().unwrap()));
    }

    #[test]
    fn a_link_local_address_keeps_its_scope() {
        let link_local = net("fe80::1", 64).with_scope(7);
        assert_eq!(link_local.scope_id, Some(7));
        assert!(link_local.contains("fe80::2".parse().unwrap()));
    }

    #[test]
    fn flag_bits_are_fixed() {
        // The bits are part of the record's encoding, so they never move.
        assert_eq!(IfaceFlags::UP.bits(), 1);
        assert_eq!(IfaceFlags::MULTICAST.bits(), 2);
        assert_eq!(IfaceFlags::METERED.bits(), 4);
        assert_eq!(IfaceFlags::BRIDGE_MEMBER.bits(), 8);
        assert_eq!(IfaceFlags::TB_BRIDGE.bits(), 16);
        assert_eq!(IfaceFlags::TRUSTED_FABRIC.bits(), 32);
        assert_eq!(IfaceFlags::DEFAULT_ROUTE.bits(), 64);
    }
}
