// SPDX-License-Identifier: MIT OR Apache-2.0

//! macOS, RFC-0030 §7.2's macOS row. The sources are:
//! - `networksetup -listallhardwareports` for each device's hardware port: `Wi-Fi`,
//!   `Thunderbolt 1`, `Thunderbolt Bridge`, `Ethernet` and so on. These are the names
//!   SystemConfiguration holds, and the ones `SCNetworkInterfaceGetLocalizedDisplayName` returns;
//! - `ifconfig -a` for flags, MTU, addresses, link speed and bridge members;
//! - `route -n get default` and `route -n get -inet6 default` for the default route.
//!
//! Metered (`NWPath.isExpensive`) needs Network.framework, which this module does not call, so
//! `METERED` stays unset; DIST-E6-T08 adds it. Every parser takes text, so tests run on any
//! host.
//!
//! Known limit: the port names are the ones `networksetup` prints, which may be localised on a
//! system in another language (`WLAN` for `Wi-Fi`). Such a port is then classified Ethernet.
//! No non-English rig has checked this yet; if they are localised, DIST-E6-T08's
//! Network.framework and SystemConfiguration calls take the interface type instead
//! (RFC-0030 §7.2 note of 2026-10-22).

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use syntropy_core::NetMedium;
use syntropy_core::resource::{IfaceFlags, IpNet};

use super::{IfaceError, Observed, is_tailnet};

/// The device names and their hardware ports, from `networksetup -listallhardwareports`:
/// blocks of `Hardware Port: <port>` then `Device: <device>`.
pub fn parse_hardware_ports(text: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut port: Option<&str> = None;
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("Hardware Port:") {
            port = Some(name.trim());
        } else if let Some(device) = line.strip_prefix("Device:")
            && let Some(port) = port.take()
        {
            out.insert(device.trim().to_owned(), port.to_owned());
        }
    }
    out
}

/// One interface in `ifconfig -a`'s output.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IfconfigIface {
    /// Its name.
    pub name: String,
    /// The names in its `flags=…<…>` list, such as `UP` and `MULTICAST`.
    pub flags: Vec<String>,
    /// Its MTU.
    pub mtu: u32,
    /// Its `inet` and `inet6` addresses.
    pub addrs: Vec<IpNet>,
    /// The rate in its `media:` line, such as `1000baseT`, in Mbit/s.
    pub speed_mbps: Option<u32>,
    /// For a bridge, its `member:` interfaces.
    pub members: Vec<String>,
}

impl IfconfigIface {
    fn has(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }
}

/// The interfaces in `ifconfig -a`'s output. Each starts with an unindented
/// `name: flags=8863<UP,…> mtu 1500` line; its `inet`, `inet6`, `media` and `member` lines
/// follow, indented. Other indented lines are skipped. A header that does not read skips that
/// interface, and an `inet`, `inet6` or `member` line that does not read skips that line, each
/// with a warning.
pub fn parse_ifconfig(text: &str) -> (Vec<IfconfigIface>, Vec<IfaceError>) {
    const SOURCE: &str = "`ifconfig -a`";
    let mut out: Vec<IfconfigIface> = Vec::new();
    let mut warnings = Vec::new();
    // Whether the indented lines belong to the last interface in `out`: not before the first
    // header, and not after a header that did not read.
    let mut open = false;
    let mut seen_header = false;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if !line.starts_with(char::is_whitespace) {
            seen_header = true;
            match ifconfig_header(line) {
                Some(iface) => {
                    out.push(iface);
                    open = true;
                }
                None => {
                    warnings.push(IfaceError::parse(SOURCE, line));
                    open = false;
                }
            }
            continue;
        }
        match out.last_mut() {
            Some(current) if open => {
                if !ifconfig_detail(current, line) {
                    warnings.push(IfaceError::parse(SOURCE, line));
                }
            }
            // A skipped interface's lines go with it; a line before any header is noted.
            _ if !seen_header => warnings.push(IfaceError::parse(SOURCE, line)),
            _ => {}
        }
    }
    (out, warnings)
}

/// [`parse_ifconfig`] for a listing: `ifconfig -a`'s text that printed lines of which none
/// read fails, naming the command.
pub fn read_ifconfig(text: &str) -> Result<(Vec<IfconfigIface>, Vec<IfaceError>), IfaceError> {
    let (ifaces, warnings) = parse_ifconfig(text);
    super::require_some("`ifconfig -a`", &ifaces, &warnings)?;
    Ok((ifaces, warnings))
}

/// An `ifconfig` header line: `name: flags=8863<UP,…> mtu 1500`.
fn ifconfig_header(line: &str) -> Option<IfconfigIface> {
    let (name, rest) = line.split_once(": ")?;
    let flags = rest
        .split_once('<')
        .and_then(|(_, after)| after.split_once('>'))
        .map(|(list, _)| {
            list.split(',')
                .filter(|f| !f.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let mtu = match rest.split_whitespace().skip_while(|w| *w != "mtu").nth(1) {
        Some(mtu) => mtu.parse().ok()?,
        None => 0,
    };
    Some(IfconfigIface {
        name: name.to_owned(),
        flags,
        mtu,
        ..IfconfigIface::default()
    })
}

/// Reads one indented `ifconfig` line into `iface`. `false` for an `inet`, `inet6` or
/// `member` line that does not read.
fn ifconfig_detail(iface: &mut IfconfigIface, line: &str) -> bool {
    let words: Vec<&str> = line.split_whitespace().collect();
    let after = |key: &str| {
        words
            .iter()
            .position(|w| *w == key)
            .and_then(|i| words.get(i + 1))
            .copied()
    };
    let hex = |text: &str| u32::from_str_radix(text.trim_start_matches("0x"), 16).ok();
    let net = match words.first().copied() {
        Some("inet") => (|| {
            let addr: Ipv4Addr = words.get(1)?.parse().ok()?;
            let prefix = u8::try_from(hex(after("netmask")?)?.count_ones()).ok()?;
            IpNet::new(IpAddr::V4(addr), prefix)
        })(),
        Some("inet6") => (|| {
            let text = words.get(1)?;
            let addr: Ipv6Addr = text.split('%').next().unwrap_or(text).parse().ok()?;
            let prefix: u8 = after("prefixlen")?.parse().ok()?;
            let net = IpNet::new(IpAddr::V6(addr), prefix)?;
            match after("scopeid") {
                Some(scope) => Some(net.with_scope(hex(scope)?)),
                None => Some(net),
            }
        })(),
        Some("media:") => {
            iface.speed_mbps = media_speed(line);
            return true;
        }
        Some("member:") => {
            return match words.get(1) {
                Some(member) => {
                    iface.members.push((*member).to_owned());
                    true
                }
                None => false,
            };
        }
        _ => return true,
    };
    match net {
        Some(net) => {
            iface.addrs.push(net);
            true
        }
        None => false,
    }
}

/// The rate in a `media:` line: the first word in its parentheses, or after `media:`, as
/// `<n>base…` (Mbit/s) or `<n>Gbase…` (Gbit/s). `autoselect` with no rate gives `None`.
fn media_speed(line: &str) -> Option<u32> {
    let inner = line
        .split_once('(')
        .map(|(_, rest)| rest)
        .or_else(|| line.split_once("media:").map(|(_, rest)| rest))?;
    let word = inner.split_whitespace().next()?.to_ascii_lowercase();
    let (number, _) = word.split_once("base")?;
    let (digits, scale) = match number.strip_suffix('g') {
        Some(digits) => (digits, 1000),
        None => (number, 1),
    };
    digits.parse::<u32>().ok()?.checked_mul(scale)
}

/// The interface named in `route -n get …`'s `interface:` line.
pub fn parse_route_get(text: &str) -> Option<String> {
    text.lines()
        .find_map(|line| line.trim().strip_prefix("interface:"))
        .map(|name| name.trim().to_owned())
}

/// Whether `port` is a Thunderbolt hardware port (`Thunderbolt 1`, …), not the bridge.
fn thunderbolt_port(port: Option<&str>) -> bool {
    port.is_some_and(|p| p.starts_with("Thunderbolt ") && p != "Thunderbolt Bridge")
}

/// RFC-0030 §7.2's macOS row, in order:
/// - loopback;
/// - Wi-Fi for the `Wi-Fi` hardware port, and for Apple's Wi-Fi peer-to-peer interfaces
///   (`awdl*`, `llw*`);
/// - Thunderbolt IP for a `Thunderbolt N` port, or a bridge with such a port as a member;
/// - overlay for a tunnel interface (`utun*`, `gif*`, `stf*`, `ipsec*`) or one that holds a
///   tailnet address;
/// - Ethernet for every other port.
pub fn classify(iface: &IfconfigIface, port: Option<&str>, thunderbolt_members: bool) -> NetMedium {
    let name = iface.name.as_str();
    let tunnel = ["utun", "gif", "stf", "ipsec"]
        .iter()
        .any(|p| name.starts_with(p));
    if name.starts_with("lo") && iface.has("LOOPBACK") {
        NetMedium::Loopback
    } else if matches!(port, Some("Wi-Fi" | "AirPort"))
        || name.starts_with("awdl")
        || name.starts_with("llw")
    {
        NetMedium::WiFi
    } else if thunderbolt_port(port) || (!iface.members.is_empty() && thunderbolt_members) {
        NetMedium::ThunderboltIp
    } else if tunnel || iface.addrs.iter().any(|a| is_tailnet(a.addr)) {
        NetMedium::Overlay
    } else {
        NetMedium::Ethernet
    }
}

/// The observed records for `ifaces`, with hardware `ports` and `defaults` (the interfaces
/// carrying a default route).
pub fn observe(
    ifaces: &[IfconfigIface],
    ports: &BTreeMap<String, String>,
    defaults: &BTreeSet<String>,
) -> Vec<Observed> {
    let port = |name: &str| ports.get(name).map(String::as_str);
    let bridge_of = |name: &str| {
        ifaces
            .iter()
            .find(|b| b.members.iter().any(|m| m == name))
            .map(|b| b.name.clone())
    };
    ifaces
        .iter()
        .map(|iface| {
            let thunderbolt_members = iface.members.iter().any(|m| thunderbolt_port(port(m)));
            let bridge = bridge_of(&iface.name);
            let mut flags = IfaceFlags::empty();
            flags.set(IfaceFlags::UP, iface.has("UP"));
            flags.set(IfaceFlags::MULTICAST, iface.has("MULTICAST"));
            flags.set(IfaceFlags::BRIDGE_MEMBER, bridge.is_some());
            flags.set(IfaceFlags::TB_BRIDGE, thunderbolt_members);
            flags.set(IfaceFlags::DEFAULT_ROUTE, defaults.contains(&iface.name));
            Observed {
                name: iface.name.clone(),
                medium: classify(iface, port(&iface.name), thunderbolt_members),
                flags,
                addrs: iface.addrs.clone(),
                mtu: iface.mtu,
                speed_mbps: iface.speed_mbps,
                bridge,
            }
        })
        .collect()
}

/// Runs this host's tools and observes its interfaces. It fails only when `networksetup` or
/// `ifconfig` cannot be used.
#[cfg(target_os = "macos")]
pub(crate) fn gather() -> Result<super::Gathered, IfaceError> {
    use super::run;
    let mut warnings = Vec::new();
    let ports = parse_hardware_ports(&run(
        "networksetup",
        &["-listallhardwareports"],
        &mut warnings,
    )?);
    let (ifaces, ifconfig_warnings) = read_ifconfig(&run("ifconfig", &["-a"], &mut warnings)?)?;
    warnings.extend(ifconfig_warnings);
    // `route` fails when there is no default route of that family; that is no default.
    let mut defaults = BTreeSet::new();
    for args in [
        &["-n", "get", "default"][..],
        &["-n", "get", "-inet6", "default"][..],
    ] {
        let mut ignored = Vec::new();
        if let Some(name) = run("route", args, &mut ignored)
            .ok()
            .as_deref()
            .and_then(parse_route_get)
        {
            defaults.insert(name);
        }
    }
    Ok((observe(&ifaces, &ports, &defaults), warnings))
}

#[cfg(test)]
mod tests {
    //! The fixtures follow the output formats of macOS's `networksetup`, `ifconfig` and
    //! `route`. They are synthetic, written from those formats, not recorded on a rig.

    use super::*;

    const PORTS: &str = "\
Hardware Port: Ethernet
Device: en0
Ethernet Address: 3c:22:fb:00:00:01

Hardware Port: Wi-Fi
Device: en1
Ethernet Address: 3c:22:fb:00:00:02

Hardware Port: Thunderbolt 1
Device: en2
Ethernet Address: 82:6a:00:00:00:03

Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: 82:6a:00:00:00:04

VLAN Configurations
===================
";

    const IFCONFIG: &str = "\
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\toptions=1203<RXCSUM,TXCSUM,TXSTATUS,SW_TIMESTAMP>
\tinet 127.0.0.1 netmask 0xff000000
\tinet6 ::1 prefixlen 128
\tinet6 fe80::1%lo0 prefixlen 64 scopeid 0x1
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether 3c:22:fb:00:00:01
\tinet 192.168.1.20 netmask 0xffffff00 broadcast 192.168.1.255
\tinet6 fe80::1c2b:3a4d:5e6f:7081%en0 prefixlen 64 secured scopeid 0x4
\tmedia: autoselect (10Gbase-T <full-duplex>)
\tstatus: active
en1: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 192.168.1.21 netmask 0xffffff00 broadcast 192.168.1.255
\tmedia: autoselect
\tstatus: active
en2: flags=8963<UP,BROADCAST,SMART,RUNNING,PROMISC,SIMPLEX,MULTICAST> mtu 1500
\tmedia: autoselect <full-duplex>
\tstatus: active
bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet6 fe80::806a:ff:fe00:4%bridge0 prefixlen 64 scopeid 0x9
\tmember: en2 flags=3<LEARNING,DISCOVER>
\t        ifmaxaddr 0 port 7 priority 0 path cost 0
\tmedia: <unknown type>
\tstatus: active
utun3: flags=8051<UP,POINTOPOINT,RUNNING,MULTICAST> mtu 1280
\tinet 100.88.12.7 --> 100.88.12.7 netmask 0xffffffff
\tinet6 fd7a:115c:a1e0::ab12 prefixlen 48
awdl0: flags=8943<UP,BROADCAST,RUNNING,PROMISC,SIMPLEX,MULTICAST> mtu 1500
\tmedia: autoselect
";

    #[test]
    fn parses_hardware_ports() {
        let ports = parse_hardware_ports(PORTS);
        assert_eq!(ports.get("en0").map(String::as_str), Some("Ethernet"));
        assert_eq!(ports.get("en1").map(String::as_str), Some("Wi-Fi"));
        assert_eq!(ports.get("en2").map(String::as_str), Some("Thunderbolt 1"));
        assert_eq!(
            ports.get("bridge0").map(String::as_str),
            Some("Thunderbolt Bridge")
        );
        assert_eq!(ports.len(), 4);
    }

    #[test]
    fn parses_ifconfig() {
        let (ifaces, warnings) = parse_ifconfig(IFCONFIG);
        assert!(warnings.is_empty(), "{warnings:?}");
        let names: Vec<&str> = ifaces.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(
            names,
            ["lo0", "en0", "en1", "en2", "bridge0", "utun3", "awdl0"]
        );
        let en0 = &ifaces[1];
        assert_eq!(en0.mtu, 1500);
        assert_eq!(en0.speed_mbps, Some(10_000));
        assert!(en0.has("MULTICAST"));
        assert_eq!(
            en0.addrs[0],
            IpNet::new("192.168.1.20".parse().unwrap(), 24).unwrap()
        );
        assert_eq!(en0.addrs[1].scope_id, Some(4));
        assert_eq!(en0.addrs[1].prefix_len, 64);
        assert_eq!(ifaces[2].speed_mbps, None);
        assert_eq!(ifaces[4].members, ["en2"]);
        assert_eq!(ifaces[5].addrs[0].prefix_len, 32);
        let (none, warnings) = parse_ifconfig("\tinet 10.0.0.1 netmask 0xff000000");
        assert_eq!((none.len(), warnings.len()), (0, 1));
    }

    /// A line or interface that does not read costs only itself.
    #[test]
    fn a_bad_line_costs_only_itself() {
        let text = format!(
            "{IFCONFIG}bad0 flags=8863<UP> mtu 1500\n\tinet 10.9.9.9 netmask 0xffffff00\nen7: flags=8863<UP,MULTICAST> mtu 1500\n\tinet 10.0.0.7 netmask nonsense\n\tinet 10.0.0.8 netmask 0xffffff00\n"
        );
        let (ifaces, warnings) = parse_ifconfig(&text);
        // bad0's header has no `: `, so bad0 and its line are skipped; en7 keeps its good
        // address and loses the bad one.
        assert!(ifaces.iter().all(|i| i.name != "bad0"));
        let en7 = ifaces.iter().find(|i| i.name == "en7").unwrap();
        assert_eq!(
            en7.addrs,
            [IpNet::new("10.0.0.8".parse().unwrap(), 24).unwrap()]
        );
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert_eq!(ifaces.len(), 8);
    }

    /// `ifconfig` output of which nothing reads fails, naming the command.
    #[test]
    fn an_unreadable_ifconfig_fails() {
        let failed = read_ifconfig("no header here\nnor here\n").unwrap_err();
        assert!(failed.to_string().contains("ifconfig -a"), "{failed}");
        assert!(read_ifconfig(IFCONFIG).is_ok());
    }

    #[test]
    fn reads_link_rates() {
        assert_eq!(
            media_speed("\tmedia: autoselect (1000baseT <full-duplex>)"),
            Some(1000)
        );
        assert_eq!(
            media_speed("\tmedia: autoselect (2500Base-T <full-duplex>)"),
            Some(2500)
        );
        assert_eq!(media_speed("\tmedia: 100baseTX <full-duplex>"), Some(100));
        assert_eq!(media_speed("\tmedia: autoselect"), None);
        assert_eq!(media_speed("\tmedia: <unknown type>"), None);
    }

    #[test]
    fn parses_route_get() {
        let text = "   route to: default\ndestination: default\n    gateway: 192.168.1.1\n  interface: en0\n";
        assert_eq!(parse_route_get(text).as_deref(), Some("en0"));
        assert_eq!(
            parse_route_get("route: writing to routing socket: not in table"),
            None
        );
    }

    #[test]
    fn classifies_each_row_of_the_table() {
        let (ifaces, _) = parse_ifconfig(IFCONFIG);
        let ports = parse_hardware_ports(PORTS);
        let defaults: BTreeSet<String> = [String::from("en0")].into();
        let observed = observe(&ifaces, &ports, &defaults);
        let by = |name: &str| observed.iter().find(|o| o.name == name).unwrap();
        assert_eq!(by("lo0").medium, NetMedium::Loopback);
        assert_eq!(by("en0").medium, NetMedium::Ethernet);
        assert!(
            by("en0")
                .flags
                .contains(IfaceFlags::UP | IfaceFlags::MULTICAST | IfaceFlags::DEFAULT_ROUTE)
        );
        assert_eq!(by("en0").speed_mbps, Some(10_000));
        assert_eq!(by("en1").medium, NetMedium::WiFi);
        assert_eq!(by("en2").medium, NetMedium::ThunderboltIp);
        assert_eq!(by("en2").bridge.as_deref(), Some("bridge0"));
        assert!(by("en2").flags.contains(IfaceFlags::BRIDGE_MEMBER));
        assert_eq!(by("bridge0").medium, NetMedium::ThunderboltIp);
        assert!(by("bridge0").flags.contains(IfaceFlags::TB_BRIDGE));
        assert_eq!(by("utun3").medium, NetMedium::Overlay);
        assert_eq!(by("awdl0").medium, NetMedium::WiFi);
        // Metered is not known on macOS yet (DIST-E6-T08).
        assert!(
            observed
                .iter()
                .all(|o| !o.flags.contains(IfaceFlags::METERED))
        );
    }

    #[test]
    fn a_bridge_without_thunderbolt_members_is_ethernet() {
        let (ifaces, _) = parse_ifconfig(
            "bridge1: flags=8863<UP,MULTICAST> mtu 1500\n\tmember: en0 flags=3<LEARNING,DISCOVER>\n",
        );
        let ports: BTreeMap<String, String> =
            [(String::from("en0"), String::from("Ethernet"))].into();
        let observed = observe(&ifaces, &ports, &BTreeSet::new());
        assert_eq!(observed[0].medium, NetMedium::Ethernet);
        assert!(!observed[0].flags.contains(IfaceFlags::TB_BRIDGE));
    }
}
