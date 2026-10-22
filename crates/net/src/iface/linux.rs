// SPDX-License-Identifier: MIT OR Apache-2.0

//! Linux, RFC-0030 §7.2's Linux row. The sources are:
//! - sysfs (`/sys/class/net/<if>`) for each interface's type, flags, MTU, speed, wireless
//!   marker, driver and bridge;
//! - `/proc/net/if_inet6` for IPv6 addresses with their scope;
//! - `ip -o -4 addr show` for IPv4 addresses, which sysfs does not hold;
//! - `/proc/net/route` and `/proc/net/ipv6_route` for the default route;
//! - NetworkManager's `nmcli`, where it runs, for metered.
//!
//! Every parser takes text or a directory, so tests run them on any host.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::Path;

use syntropy_core::NetMedium;
use syntropy_core::resource::{IfaceFlags, IpNet};

use super::{IfaceError, Observed, is_tailnet};

/// `ARPHRD_ETHER`, the sysfs `type` of an Ethernet-framed interface.
const ARPHRD_ETHER: u32 = 1;
/// `ARPHRD_LOOPBACK`.
const ARPHRD_LOOPBACK: u32 = 772;
/// `IFF_UP` in sysfs `flags`.
const IFF_UP: u32 = 0x1;
/// `IFF_LOOPBACK`.
const IFF_LOOPBACK: u32 = 0x8;
/// `IFF_MULTICAST`.
const IFF_MULTICAST: u32 = 0x1000;
/// `RTF_UP` in the route tables' flags.
const RTF_UP: u32 = 0x1;
/// `RTF_REJECT`: an unreachable route, not a usable default.
const RTF_REJECT: u32 = 0x200;
/// The scope `/proc/net/if_inet6` gives a link-local address.
const SCOPE_LINK: u32 = 0x20;

/// What sysfs holds for one interface.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SysfsIface {
    /// Its name.
    pub name: String,
    /// Its `type`, an `ARPHRD_*` value.
    pub arphrd: u32,
    /// Its `flags`, `IFF_*` bits.
    pub iff: u32,
    /// Its `mtu`.
    pub mtu: u32,
    /// Its `speed`, when the driver reports a positive one.
    pub speed_mbps: Option<u32>,
    /// It has a `wireless` or `phy80211` entry, or its `uevent` says `DEVTYPE=wlan`.
    pub wireless: bool,
    /// The name of the driver its `device/driver` link points to.
    pub driver: Option<String>,
    /// The bridge it is a port of (it has `brport`, and `master` names the bridge).
    pub bridge: Option<String>,
    /// It is a bridge (it has a `bridge` entry).
    pub is_bridge: bool,
}

impl SysfsIface {
    fn thunderbolt(&self) -> bool {
        matches!(
            self.driver.as_deref(),
            Some("thunderbolt-net" | "thunderbolt_net")
        )
    }
}

/// Reads every interface under `root`, normally `/sys/class/net`. An interface that vanishes
/// while it is read is left out. One whose files cannot be read is left out with a warning.
/// Only an unreadable `root` fails.
pub fn read_sysfs(root: &Path) -> Result<(Vec<SysfsIface>, Vec<IfaceError>), IfaceError> {
    let shown = root.display().to_string();
    let entries = fs::read_dir(root).map_err(|e| IfaceError::source(&shown, e))?;
    let mut out = Vec::new();
    let mut warnings = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                warnings.push(IfaceError::source(&shown, e));
                continue;
            }
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        let dir = root.join(&name);
        let numbers = (|| -> Result<Option<(u32, u32, u32)>, IfaceError> {
            let Some(arphrd) = read_number(&dir.join("type"), 10)? else {
                return Ok(None);
            };
            let iff = read_number(&dir.join("flags"), 16)?.unwrap_or(0);
            let mtu = read_number(&dir.join("mtu"), 10)?.unwrap_or(0);
            Ok(Some((arphrd, iff, mtu)))
        })();
        let (arphrd, iff, mtu) = match numbers {
            Ok(Some(numbers)) => numbers,
            Ok(None) => continue,
            Err(e) => {
                warnings.push(e);
                continue;
            }
        };
        // `speed` is unreadable (EINVAL) or -1 when the link is down or the driver has none.
        let speed_mbps = fs::read_to_string(dir.join("speed"))
            .ok()
            .and_then(|text| text.trim().parse::<i64>().ok())
            .filter(|&speed| speed > 0)
            .and_then(|speed| u32::try_from(speed).ok());
        let uevent = fs::read_to_string(dir.join("uevent")).unwrap_or_default();
        let wireless = dir.join("wireless").exists()
            || dir.join("phy80211").exists()
            || uevent.lines().any(|line| line.trim() == "DEVTYPE=wlan");
        let link_name = |path: &Path| {
            fs::read_link(path)
                .ok()
                .and_then(|target| target.file_name().map(|n| n.to_string_lossy().into_owned()))
        };
        let driver = link_name(&dir.join("device").join("driver"));
        let bridge = if dir.join("brport").exists() {
            link_name(&dir.join("master"))
        } else {
            None
        };
        out.push(SysfsIface {
            name,
            arphrd,
            iff,
            mtu,
            speed_mbps,
            wireless,
            driver,
            bridge,
            is_bridge: dir.join("bridge").exists(),
        });
    }
    super::require_some(&shown, &out, &warnings)?;
    Ok((out, warnings))
}

/// A number in `path`, in `radix` (16 accepts a `0x` prefix). `None` if the file is gone.
fn read_number(path: &Path, radix: u32) -> Result<Option<u32>, IfaceError> {
    let shown = path.display().to_string();
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(IfaceError::source(shown, e)),
    };
    let digits = text.trim();
    let digits = if radix == 16 {
        digits.trim_start_matches("0x")
    } else {
        digits
    };
    u32::from_str_radix(digits, radix)
        .map(Some)
        .map_err(|_| IfaceError::parse(shown, text.trim()))
}

/// The addresses in `/proc/net/if_inet6`: one line per address, as `address ifindex prefix
/// scope flags name`, all in hex but the name. A link-local address keeps its interface
/// index as its scope. A line that does not read is skipped, with a warning.
pub fn parse_if_inet6(text: &str) -> (Vec<(String, IpNet)>, Vec<IfaceError>) {
    const SOURCE: &str = "/proc/net/if_inet6";
    let one = |line: &str| -> Option<(String, IpNet)> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let [addr, index, prefix, scope, _flags, name] = fields[..] else {
            return None;
        };
        let addr = u128::from_str_radix(addr, 16).ok()?;
        let index = u32::from_str_radix(index, 16).ok()?;
        let prefix = u8::from_str_radix(prefix, 16).ok()?;
        let scope = u32::from_str_radix(scope, 16).ok()?;
        let mut net = IpNet::new(IpAddr::V6(Ipv6Addr::from(addr)), prefix)?;
        if scope == SCOPE_LINK {
            net = net.with_scope(index);
        }
        Some((name.to_owned(), net))
    };
    lines_of(text, SOURCE, one)
}

/// Addresses by interface name.
pub type Addresses = Vec<(String, IpNet)>;

/// The addresses in `ip -o -4 addr show`'s text (`v4`) and `/proc/net/if_inet6`'s (`v6`), and
/// a warning for each line that does not read. Either source that printed lines of which none
/// read fails, naming it.
pub fn read_addresses(v4: &str, v6: &str) -> Result<(Addresses, Vec<IfaceError>), IfaceError> {
    let (mut addrs, mut warnings) = parse_ip_addr_v4(v4);
    super::require_some("`ip -o -4 addr show`", &addrs, &warnings)?;
    let (v6_addrs, v6_warnings) = parse_if_inet6(v6);
    super::require_some("/proc/net/if_inet6", &v6_addrs, &v6_warnings)?;
    addrs.extend(v6_addrs);
    warnings.extend(v6_warnings);
    Ok((addrs, warnings))
}

/// Each non-blank line of `text` read by `one`, and a warning for each line `one` refuses.
fn lines_of<T>(
    text: &str,
    source: &str,
    one: impl Fn(&str) -> Option<T>,
) -> (Vec<T>, Vec<IfaceError>) {
    let mut out = Vec::new();
    let mut warnings = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        match one(line) {
            Some(item) => out.push(item),
            None => warnings.push(IfaceError::parse(source, line)),
        }
    }
    (out, warnings)
}

/// The IPv4 addresses in `ip -o -4 addr show`'s output: one line per address, as
/// `index: name inet address/prefix …`, or for a point-to-point address (OpenVPN's `tun0`,
/// `ppp0`, `ip addr add A peer B`) `index: name inet LOCAL peer PEER/prefix …`, which is read
/// as `LOCAL` with that prefix, as the kernel holds it. A name such as `eth0.10@eth0` is cut at
/// the `@`. A line that does not read is skipped, with a warning.
pub fn parse_ip_addr_v4(text: &str) -> (Vec<(String, IpNet)>, Vec<IfaceError>) {
    const SOURCE: &str = "`ip -o -4 addr show`";
    let one = |line: &str| -> Option<(String, IpNet)> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let (Some(name), Some(&"inet"), Some(local)) =
            (fields.get(1), fields.get(2), fields.get(3))
        else {
            return None;
        };
        let name = name.split('@').next().unwrap_or(name);
        let (addr, prefix) = match local.split_once('/') {
            Some(pair) => pair,
            None => {
                // `LOCAL peer PEER/prefix`: the prefix is on the peer.
                let (Some(&"peer"), Some(peer)) = (fields.get(4), fields.get(5)) else {
                    return None;
                };
                (*local, peer.split_once('/')?.1)
            }
        };
        let addr: Ipv4Addr = addr.parse().ok()?;
        let prefix: u8 = prefix.parse().ok()?;
        Some((name.to_owned(), IpNet::new(IpAddr::V4(addr), prefix)?))
    };
    lines_of(text, SOURCE, one)
}

/// The interfaces that carry a default route: an up IPv4 route to 0.0.0.0/0 in
/// `/proc/net/route`, or an up, non-reject IPv6 route to ::/0 in `/proc/net/ipv6_route`.
pub fn parse_default_routes(route: &str, ipv6_route: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in route.lines().skip(1) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if let [name, dest, _gw, flags, _, _, _, mask, ..] = f[..] {
            let up = u32::from_str_radix(flags, 16).is_ok_and(|flags| flags & RTF_UP != 0);
            if dest == "00000000" && mask == "00000000" && up {
                out.insert(name.to_owned());
            }
        }
    }
    for line in ipv6_route.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if let [dest, prefix, _, _, _, _, _, _, flags, name] = f[..] {
            let flags = u32::from_str_radix(flags, 16).unwrap_or(0);
            let default = dest.bytes().all(|b| b == b'0') && prefix == "00";
            if default && flags & RTF_UP != 0 && flags & RTF_REJECT == 0 {
                out.insert(name.to_owned());
            }
        }
    }
    out
}

/// Metered, per device, from `nmcli -t -f GENERAL.DEVICE,GENERAL.METERED device show`. A
/// device whose value is neither yes nor no (`unknown`) is left out.
pub fn parse_nmcli_metered(text: &str) -> BTreeMap<String, bool> {
    let mut out = BTreeMap::new();
    let mut device: Option<String> = None;
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("GENERAL.DEVICE:") {
            device = Some(name.trim().to_owned());
        } else if let Some(value) = line.strip_prefix("GENERAL.METERED:") {
            let value = value.trim();
            let metered = if value.starts_with("yes") {
                Some(true)
            } else if value.starts_with("no") {
                Some(false)
            } else {
                None
            };
            if let (Some(name), Some(metered)) = (device.take(), metered) {
                out.insert(name, metered);
            }
        }
    }
    out
}

/// RFC-0030 §7.2's Linux row, in order:
/// - loopback;
/// - Wi-Fi when the interface has a wireless marker, whatever its type;
/// - Thunderbolt IP when its driver is `thunderbolt-net`;
/// - overlay when it holds a tailnet address;
/// - Ethernet when its type is `ARPHRD_ETHER`;
/// - an overlay for any other type, a tunnel (`tun`, `sit`, `gre`, `ipip`, …).
///
/// The table's `tailscale*`, `wg*` and `tun*` names are tunnels, so the type rule covers them;
/// no separate name rule is applied, and a name alone never makes an Ethernet-type interface
/// an overlay.
pub fn classify(iface: &SysfsIface, addrs: &[IpNet]) -> NetMedium {
    if iface.arphrd == ARPHRD_LOOPBACK || iface.iff & IFF_LOOPBACK != 0 {
        NetMedium::Loopback
    } else if iface.wireless {
        NetMedium::WiFi
    } else if iface.thunderbolt() {
        NetMedium::ThunderboltIp
    } else if addrs.iter().any(|a| is_tailnet(a.addr)) {
        NetMedium::Overlay
    } else if iface.arphrd == ARPHRD_ETHER {
        NetMedium::Ethernet
    } else {
        NetMedium::Overlay
    }
}

/// The observed records: each sysfs interface with its addresses, medium and flags. A bridge
/// with a Thunderbolt port is flagged `TB_BRIDGE`.
pub fn observe(
    sysfs: &[SysfsIface],
    addrs: &[(String, IpNet)],
    defaults: &BTreeSet<String>,
    metered: &BTreeMap<String, bool>,
) -> Vec<Observed> {
    sysfs
        .iter()
        .map(|iface| {
            let own: Vec<IpNet> = addrs
                .iter()
                .filter(|(name, _)| *name == iface.name)
                .map(|(_, net)| *net)
                .collect();
            let mut flags = IfaceFlags::empty();
            flags.set(IfaceFlags::UP, iface.iff & IFF_UP != 0);
            flags.set(IfaceFlags::MULTICAST, iface.iff & IFF_MULTICAST != 0);
            flags.set(
                IfaceFlags::METERED,
                metered.get(&iface.name).copied().unwrap_or(false),
            );
            flags.set(IfaceFlags::BRIDGE_MEMBER, iface.bridge.is_some());
            flags.set(
                IfaceFlags::TB_BRIDGE,
                iface.is_bridge
                    && sysfs.iter().any(|port| {
                        port.bridge.as_deref() == Some(iface.name.as_str()) && port.thunderbolt()
                    }),
            );
            flags.set(IfaceFlags::DEFAULT_ROUTE, defaults.contains(&iface.name));
            Observed {
                name: iface.name.clone(),
                medium: classify(iface, &own),
                flags,
                addrs: own,
                mtu: iface.mtu,
                speed_mbps: iface.speed_mbps,
                bridge: iface.bridge.clone(),
            }
        })
        .collect()
}

/// Reads this host's sources and observes its interfaces. It fails only when sysfs, `ip` or
/// `/proc/net/if_inet6` cannot be used.
#[cfg(target_os = "linux")]
pub(crate) fn gather() -> Result<super::Gathered, IfaceError> {
    let optional = |path: &str| -> Result<String, IfaceError> {
        match fs::read_to_string(path) {
            Ok(text) => Ok(text),
            // No IPv6, or no route table: nothing to read.
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(String::new()),
            Err(e) => Err(IfaceError::source(path, e)),
        }
    };
    let (sysfs, mut warnings) = read_sysfs(Path::new("/sys/class/net"))?;
    let v4_text = super::run("ip", &["-o", "-4", "addr", "show"], &mut warnings)?;
    let (addrs, address_warnings) = read_addresses(&v4_text, &optional("/proc/net/if_inet6")?)?;
    warnings.extend(address_warnings);
    let defaults = parse_default_routes(
        &optional("/proc/net/route")?,
        &optional("/proc/net/ipv6_route")?,
    );
    // Metered comes from NetworkManager when it is installed and running; otherwise it is not
    // known, and `METERED` stays unset (RFC-0030 §7.2: "when available").
    let metered = std::process::Command::new("nmcli")
        .args([
            "-t",
            "-f",
            "GENERAL.DEVICE,GENERAL.METERED",
            "device",
            "show",
        ])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| parse_nmcli_metered(&String::from_utf8_lossy(&output.stdout)))
        .unwrap_or_default();
    Ok((observe(&sysfs, &addrs, &defaults, &metered), warnings))
}

#[cfg(test)]
mod tests {
    //! The fixtures follow the formats of the kernel's sysfs and procfs and of iproute2's and
    //! NetworkManager's output. They are synthetic, written from those formats, not recorded
    //! on a rig.

    use super::*;

    fn iface(name: &str, arphrd: u32) -> SysfsIface {
        SysfsIface {
            name: name.to_owned(),
            arphrd,
            iff: IFF_UP | IFF_MULTICAST,
            mtu: 1500,
            ..SysfsIface::default()
        }
    }

    fn v4(text: &str, prefix: u8) -> IpNet {
        IpNet::new(text.parse().unwrap(), prefix).unwrap()
    }

    #[test]
    fn classifies_each_row_of_the_table() {
        let mut lo = iface("lo", ARPHRD_LOOPBACK);
        lo.iff |= IFF_LOOPBACK;
        assert_eq!(classify(&lo, &[]), NetMedium::Loopback);
        assert_eq!(
            classify(&iface("eth0", ARPHRD_ETHER), &[]),
            NetMedium::Ethernet
        );
        // Wi-Fi wins even when the type says Ethernet, as a USB Wi-Fi bridge's may.
        let wlan = SysfsIface {
            wireless: true,
            ..iface("wlan0", ARPHRD_ETHER)
        };
        assert_eq!(classify(&wlan, &[]), NetMedium::WiFi);
        let tb = SysfsIface {
            driver: Some(String::from("thunderbolt-net")),
            ..iface("thunderbolt0", ARPHRD_ETHER)
        };
        assert_eq!(classify(&tb, &[]), NetMedium::ThunderboltIp);
        let tb_underscore = SysfsIface {
            driver: Some(String::from("thunderbolt_net")),
            ..iface("tb0", ARPHRD_ETHER)
        };
        assert_eq!(classify(&tb_underscore, &[]), NetMedium::ThunderboltIp);
        for name in ["tailscale0", "wg0", "tun3"] {
            assert_eq!(
                classify(&iface(name, 65534), &[]),
                NetMedium::Overlay,
                "{name}"
            );
        }
        // An overlay by its address alone, even framed as Ethernet.
        let by_addr = iface("vpn7", ARPHRD_ETHER);
        assert_eq!(
            classify(&by_addr, &[v4("100.101.1.2", 32)]),
            NetMedium::Overlay
        );
        // Any other tunnel type.
        assert_eq!(classify(&iface("sit0", 776), &[]), NetMedium::Overlay);
        // A name alone does not make an Ethernet interface an overlay.
        for name in ["wgx-not-wg", "tunnelbridge0", "tailscale-lab"] {
            assert_eq!(
                classify(&iface(name, ARPHRD_ETHER), &[]),
                NetMedium::Ethernet,
                "{name}"
            );
        }
        assert_eq!(
            classify(&iface("docker0", ARPHRD_ETHER), &[]),
            NetMedium::Ethernet
        );
    }

    #[test]
    fn parses_if_inet6_with_link_local_scope() {
        let text = "\
fe8000000000000002155dfffed4df5b 02 40 20 80     eth0
00000000000000000000000000000001 01 80 10 80       lo
fd7a115ca1e0000000000000000000aa 05 80 00 00 tailscale0
";
        let (parsed, warnings) = parse_if_inet6(text);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].0, "eth0");
        assert_eq!(
            parsed[0].1.addr,
            "fe80::215:5dff:fed4:df5b".parse::<IpAddr>().unwrap()
        );
        assert_eq!(
            (parsed[0].1.prefix_len, parsed[0].1.scope_id),
            (64, Some(2))
        );
        assert_eq!((parsed[1].1.prefix_len, parsed[1].1.scope_id), (128, None));
        assert!(is_tailnet(parsed[2].1.addr));
        // A line that does not read costs only that line.
        let (parsed, warnings) = parse_if_inet6(&format!("not a line\n{text}"));
        assert_eq!((parsed.len(), warnings.len()), (3, 1));
    }

    #[test]
    fn parses_ip_addr_v4() {
        let text = "\
1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever preferred_lft forever
2: eth0    inet 172.18.109.241/20 brd 172.18.111.255 scope global eth0\\       valid_lft forever preferred_lft forever
7: eth0.10@eth0    inet 10.10.0.4/24 brd 10.10.0.255 scope global eth0.10\\       valid_lft forever preferred_lft forever
";
        let (parsed, warnings) = parse_ip_addr_v4(text);
        assert!(warnings.is_empty(), "{warnings:?}");
        let names: Vec<&str> = parsed.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["lo", "eth0", "eth0.10"]);
        assert_eq!(parsed[1].1, v4("172.18.109.241", 20));
        // A line that does not read is skipped with a warning; the others stand.
        let (parsed, warnings) =
            parse_ip_addr_v4(&format!("2: eth0 link/ether 00:15:5d:d4:df:5b\n{text}"));
        assert_eq!((parsed.len(), warnings.len()), (3, 1));
    }

    /// A point-to-point address prints its prefix on the peer, as OpenVPN's `tun0` and PPP's
    /// `ppp0` do; it reads as the local address with that prefix.
    #[test]
    fn parses_point_to_point_addresses() {
        let text = "\
5: tun0    inet 10.8.0.6 peer 10.8.0.5/32 scope global tun0\\       valid_lft forever preferred_lft forever
9: ppp0    inet 100.65.3.7 peer 10.64.64.64/32 scope global ppp0\\       valid_lft forever preferred_lft forever
11: gre1    inet 192.0.2.1 peer 192.0.2.2/30 scope global gre1\\       valid_lft forever preferred_lft forever
";
        let (parsed, warnings) = parse_ip_addr_v4(text);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            parsed,
            [
                (String::from("tun0"), v4("10.8.0.6", 32)),
                (String::from("ppp0"), v4("100.65.3.7", 32)),
                (String::from("gre1"), v4("192.0.2.1", 30)),
            ]
        );
        // `peer` with no prefix, or nothing after it, does not read.
        let (parsed, warnings) = parse_ip_addr_v4(
            "5: tun0    inet 10.8.0.6 peer 10.8.0.5 scope global tun0\n5: tun0    inet 10.8.0.6 peer\n",
        );
        assert_eq!((parsed.len(), warnings.len()), (0, 2));
    }

    /// An address source that printed lines of which none read fails the listing, naming it;
    /// one with a good line among bad ones does not.
    #[test]
    fn an_unreadable_address_source_fails() {
        let v6 = "fe8000000000000002155dfffed4df5b 02 40 20 80     eth0\n";
        let failed = read_addresses("garbage\nmore garbage\n", v6).unwrap_err();
        assert!(
            failed.to_string().contains("ip -o -4 addr show"),
            "{failed}"
        );
        let failed = read_addresses("", "not\nif_inet6\n").unwrap_err();
        assert!(
            failed.to_string().contains("/proc/net/if_inet6"),
            "{failed}"
        );
        let (addrs, warnings) = read_addresses(
            "garbage\n2: eth0    inet 10.0.0.2/24 scope global eth0\n",
            v6,
        )
        .unwrap();
        assert_eq!((addrs.len(), warnings.len()), (2, 1));
        // A host with no IPv4 address prints nothing, which is not a failure.
        assert_eq!(read_addresses("", "").unwrap(), (Vec::new(), Vec::new()));
    }

    #[test]
    fn finds_default_routes() {
        let route = "\
Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT
eth0\t00000000\t016012AC\t0003\t0\t0\t0\t00000000\t0\t0\t0
eth0\t006012AC\t00000000\t0001\t0\t0\t0\t00F0FFFF\t0\t0\t0
wlan0\t00000000\t0101A8C0\t0002\t0\t0\t600\t00000000\t0\t0\t0
";
        let ipv6 = "\
00000000000000000000000000000000 00 00000000000000000000000000000000 00 fe800000000000000000000000000001 00000400 00000001 00000000 00000003     wg0
00000000000000000000000000000000 00 00000000000000000000000000000000 00 00000000000000000000000000000000 ffffffff 00000001 00000000 00200200       lo
";
        let defaults = parse_default_routes(route, ipv6);
        // wlan0's route is not up; lo's IPv6 default is a reject route.
        assert_eq!(defaults.into_iter().collect::<Vec<_>>(), ["eth0", "wg0"]);
    }

    #[test]
    fn parses_nmcli_metered() {
        let text = "\
GENERAL.DEVICE:wlan0
GENERAL.METERED:yes (guessed)

GENERAL.DEVICE:eth0
GENERAL.METERED:no

GENERAL.DEVICE:lo
GENERAL.METERED:unknown
";
        let metered = parse_nmcli_metered(text);
        assert_eq!(metered.get("wlan0"), Some(&true));
        assert_eq!(metered.get("eth0"), Some(&false));
        assert_eq!(metered.get("lo"), None);
    }

    #[test]
    fn observes_flags_bridges_and_addresses() {
        let tb = SysfsIface {
            driver: Some(String::from("thunderbolt-net")),
            bridge: Some(String::from("br0")),
            ..iface("thunderbolt0", ARPHRD_ETHER)
        };
        let br = SysfsIface {
            is_bridge: true,
            ..iface("br0", ARPHRD_ETHER)
        };
        let wlan = SysfsIface {
            wireless: true,
            iff: IFF_UP,
            ..iface("wlan0", ARPHRD_ETHER)
        };
        let addrs = vec![
            (String::from("wlan0"), v4("192.168.1.9", 24)),
            (String::from("br0"), v4("10.0.0.1", 24)),
        ];
        let defaults: BTreeSet<String> = [String::from("wlan0")].into();
        let metered: BTreeMap<String, bool> = [(String::from("wlan0"), true)].into();
        let observed = observe(&[tb, br, wlan], &addrs, &defaults, &metered);
        let by = |name: &str| observed.iter().find(|o| o.name == name).unwrap();
        assert_eq!(by("thunderbolt0").medium, NetMedium::ThunderboltIp);
        assert!(by("thunderbolt0").flags.contains(IfaceFlags::BRIDGE_MEMBER));
        assert_eq!(by("thunderbolt0").bridge.as_deref(), Some("br0"));
        assert!(by("br0").flags.contains(IfaceFlags::TB_BRIDGE));
        assert_eq!(by("br0").addrs, [v4("10.0.0.1", 24)]);
        let w = by("wlan0");
        assert_eq!(w.medium, NetMedium::WiFi);
        assert!(
            w.flags
                .contains(IfaceFlags::UP | IfaceFlags::METERED | IfaceFlags::DEFAULT_ROUTE)
        );
        assert!(!w.flags.contains(IfaceFlags::MULTICAST));
        assert!(!w.flags.contains(IfaceFlags::TRUSTED_FABRIC));
    }

    /// A sysfs tree whose only interface does not read fails, naming the directory.
    #[cfg(unix)]
    #[test]
    fn an_unreadable_sysfs_fails() {
        let root = std::env::temp_dir().join(format!("syntropy-sysfs-bad-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("bad0")).unwrap();
        fs::write(root.join("bad0/type"), "not-a-number\n").unwrap();
        let failed = read_sysfs(&root).unwrap_err();
        assert!(
            failed.to_string().contains(&root.display().to_string()),
            "{failed}"
        );
        fs::remove_dir_all(&root).unwrap();
    }

    /// A sysfs tree as the kernel lays it out, built in a temporary directory.
    #[cfg(unix)]
    #[test]
    fn reads_a_sysfs_tree() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!("syntropy-sysfs-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let write = |path: &Path, text: &str| {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        };
        let net = root.join("class/net");
        let drivers = root.join("bus/drivers");
        fs::create_dir_all(drivers.join("thunderbolt-net")).unwrap();
        // eth0: Ethernet, 1 Gbit/s.
        write(&net.join("eth0/type"), "1\n");
        write(&net.join("eth0/flags"), "0x1003\n");
        write(&net.join("eth0/mtu"), "1500\n");
        write(&net.join("eth0/speed"), "1000\n");
        // wlan0: Ethernet-framed, with a phy80211 link and a speed of -1.
        write(&net.join("wlan0/type"), "1\n");
        write(&net.join("wlan0/flags"), "0x1003\n");
        write(&net.join("wlan0/mtu"), "1500\n");
        write(&net.join("wlan0/speed"), "-1\n");
        fs::create_dir_all(net.join("wlan0/phy80211")).unwrap();
        // thunderbolt0: a port of br0, with the thunderbolt-net driver.
        write(&net.join("thunderbolt0/type"), "1\n");
        write(&net.join("thunderbolt0/flags"), "0x1003\n");
        write(&net.join("thunderbolt0/mtu"), "65520\n");
        fs::create_dir_all(net.join("thunderbolt0/device")).unwrap();
        symlink(
            drivers.join("thunderbolt-net"),
            net.join("thunderbolt0/device/driver"),
        )
        .unwrap();
        fs::create_dir_all(net.join("thunderbolt0/brport")).unwrap();
        // br0: a bridge.
        write(&net.join("br0/type"), "1\n");
        write(&net.join("br0/flags"), "0x1003\n");
        write(&net.join("br0/mtu"), "1500\n");
        fs::create_dir_all(net.join("br0/bridge")).unwrap();
        symlink(net.join("br0"), net.join("thunderbolt0/master")).unwrap();
        // gone0: listed, but its files vanished.
        fs::create_dir_all(net.join("gone0")).unwrap();
        // bad0: a `type` that does not read; it is left out with a warning.
        write(&net.join("bad0/type"), "not-a-number\n");

        let (mut read, warnings) = read_sysfs(&net).unwrap();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        read.sort_by(|a, b| a.name.cmp(&b.name));
        let names: Vec<&str> = read.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, ["br0", "eth0", "thunderbolt0", "wlan0"]);
        let by = |name: &str| read.iter().find(|i| i.name == name).unwrap();
        assert_eq!(by("eth0").speed_mbps, Some(1000));
        assert_eq!(by("eth0").iff, 0x1003);
        assert!(by("wlan0").wireless);
        assert_eq!(by("wlan0").speed_mbps, None);
        assert_eq!(
            by("thunderbolt0").driver.as_deref(),
            Some("thunderbolt-net")
        );
        assert_eq!(by("thunderbolt0").bridge.as_deref(), Some("br0"));
        assert_eq!(by("thunderbolt0").mtu, 65520);
        assert!(by("br0").is_bridge);
        let observed = observe(&read, &[], &BTreeSet::new(), &BTreeMap::new());
        let medium = |name: &str| observed.iter().find(|o| o.name == name).unwrap().medium;
        assert_eq!(medium("eth0"), NetMedium::Ethernet);
        assert_eq!(medium("wlan0"), NetMedium::WiFi);
        assert_eq!(medium("thunderbolt0"), NetMedium::ThunderboltIp);
        fs::remove_dir_all(&root).unwrap();
    }
}
