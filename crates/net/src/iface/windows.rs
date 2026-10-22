// SPDX-License-Identifier: MIT OR Apache-2.0

//! Windows, RFC-0030 §7.2's Windows row. One PowerShell run of [`SCRIPT`] reads
//! `Get-NetAdapter` for each adapter's interface type (the `IfType` that
//! `GetAdaptersAddresses` reports), description, status, MTU and speed;
//! `Get-PnpDeviceProperty` for its PnP parent; `Get-NetIPAddress` for addresses; and
//! `Get-NetRoute` for the default route. It prints tab-separated lines, which [`parse`] reads.
//!
//! Metered (`INetworkCostManager`) is a COM interface, which this module does not call, so
//! `METERED` stays unset; DIST-E6-T08 adds it. The parser and classifier take text, so tests
//! run on any host.

use std::collections::BTreeSet;
use std::net::IpAddr;

use syntropy_core::NetMedium;
use syntropy_core::resource::{IfaceFlags, IpNet};

use super::{IfaceError, Observed, is_tailnet};

/// `IfType` 6, Ethernet (IANA ifType).
const IF_TYPE_ETHERNET: u32 = 6;
/// `IfType` 24, software loopback.
const IF_TYPE_LOOPBACK: u32 = 24;
/// `IfType` 53, a proprietary virtual interface.
const IF_TYPE_PROP_VIRTUAL: u32 = 53;
/// `IfType` 71, IEEE 802.11.
const IF_TYPE_WIFI: u32 = 71;
/// `IfType` 131, a tunnel.
const IF_TYPE_TUNNEL: u32 = 131;

/// The PowerShell that prints one line per adapter (`A`), address (`I`) and default route
/// (`R`), with tab-separated fields.
pub const SCRIPT: &str = r#"$ErrorActionPreference = 'Stop'
foreach ($a in Get-NetAdapter) {
  $parent = ''
  try { $parent = [string](Get-PnpDeviceProperty -InstanceId $a.PnPDeviceID -KeyName 'DEVPKEY_Device_Parent').Data } catch { }
  "A`t$($a.InterfaceIndex)`t$($a.Name)`t$($a.InterfaceDescription)`t$($a.InterfaceType)`t$($a.Status)`t$($a.MtuSize)`t$($a.Speed)`t$parent"
}
foreach ($i in Get-NetIPAddress) { "I`t$($i.InterfaceIndex)`t$($i.IPAddress)`t$($i.PrefixLength)" }
foreach ($r in Get-NetRoute -DestinationPrefix '0.0.0.0/0','::/0' -ErrorAction SilentlyContinue) { "R`t$($r.InterfaceIndex)" }
"#;

/// One adapter from `Get-NetAdapter`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Adapter {
    /// Its interface index.
    pub index: u32,
    /// Its name, such as `Ethernet 2`.
    pub name: String,
    /// Its driver's description, such as `Thunderbolt(TM) Networking`.
    pub description: String,
    /// Its IANA `IfType`.
    pub if_type: u32,
    /// Its status is `Up`.
    pub up: bool,
    /// Its MTU, or 0 if not reported.
    pub mtu: u32,
    /// Its speed in bit/s, if reported.
    pub speed_bps: Option<u64>,
    /// Its PnP parent's instance ID, or empty.
    pub parent: String,
}

/// What [`SCRIPT`] printed: the adapters, each address with its interface index, and the
/// interface indexes that carry a default route.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Parsed {
    /// The adapters.
    pub adapters: Vec<Adapter>,
    /// The addresses, by interface index.
    pub addrs: Vec<(u32, IpNet)>,
    /// Interface indexes with a default route.
    pub defaults: BTreeSet<u32>,
}

/// Reads [`SCRIPT`]'s output. A link-local address such as `fe80::1%12` keeps its zone as its
/// scope.
pub fn parse(text: &str) -> Result<Parsed, IfaceError> {
    const SOURCE: &str = "PowerShell Get-NetAdapter";
    let mut parsed = Parsed::default();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let bad = || IfaceError::parse(SOURCE, line);
        let fields: Vec<&str> = line.split('\t').collect();
        let number = |s: &str| s.trim().parse::<u32>().map_err(|_| bad());
        match fields[..] {
            [
                "A",
                index,
                name,
                description,
                if_type,
                status,
                mtu,
                speed,
                parent,
            ] => {
                parsed.adapters.push(Adapter {
                    index: number(index)?,
                    name: name.to_owned(),
                    description: description.to_owned(),
                    if_type: number(if_type)?,
                    up: status == "Up",
                    mtu: if mtu.trim().is_empty() {
                        0
                    } else {
                        number(mtu)?
                    },
                    speed_bps: speed.trim().parse::<u64>().ok().filter(|&s| s > 0),
                    parent: parent.to_owned(),
                });
            }
            ["I", index, addr, prefix] => {
                let (text, zone) = match addr.split_once('%') {
                    Some((text, zone)) => (text, Some(number(zone)?)),
                    None => (addr, None),
                };
                let addr: IpAddr = text.parse().map_err(|_| bad())?;
                let prefix: u8 = prefix.trim().parse().map_err(|_| bad())?;
                let mut net = IpNet::new(addr, prefix).ok_or_else(bad)?;
                if let Some(zone) = zone {
                    net = net.with_scope(zone);
                }
                parsed.addrs.push((number(index)?, net));
            }
            ["R", index] => {
                parsed.defaults.insert(number(index)?);
            }
            _ => return Err(bad()),
        }
    }
    Ok(parsed)
}

/// RFC-0030 §7.2's Windows row, in order:
/// - loopback for `IfType` 24;
/// - Wi-Fi for `IfType` 71;
/// - Thunderbolt IP when the description names Thunderbolt, or an `IfType` 6 adapter has a
///   Thunderbolt PnP parent;
/// - overlay when the description names Tailscale, the adapter holds a tailnet address, or its
///   `IfType` is 53 (virtual) or 131 (tunnel);
/// - Ethernet for every other adapter.
pub fn classify(adapter: &Adapter, addrs: &[IpNet]) -> NetMedium {
    let description = adapter.description.to_ascii_lowercase();
    let parent = adapter.parent.to_ascii_uppercase();
    match adapter.if_type {
        IF_TYPE_LOOPBACK => NetMedium::Loopback,
        IF_TYPE_WIFI => NetMedium::WiFi,
        _ if description.contains("thunderbolt")
            || (adapter.if_type == IF_TYPE_ETHERNET && parent.contains("THUNDERBOLT")) =>
        {
            NetMedium::ThunderboltIp
        }
        _ if description.contains("tailscale")
            || addrs.iter().any(|a| is_tailnet(a.addr))
            || matches!(adapter.if_type, IF_TYPE_PROP_VIRTUAL | IF_TYPE_TUNNEL) =>
        {
            NetMedium::Overlay
        }
        _ => NetMedium::Ethernet,
    }
}

/// The observed records. `Get-NetAdapter` reports no multicast flag, so `MULTICAST` is set
/// for the media that carry multicast (Ethernet, Wi-Fi, Thunderbolt IP) and left unset for
/// overlays and loopback.
pub fn observe(parsed: &Parsed) -> Vec<Observed> {
    parsed
        .adapters
        .iter()
        .map(|adapter| {
            let addrs: Vec<IpNet> = parsed
                .addrs
                .iter()
                .filter(|(index, _)| *index == adapter.index)
                .map(|(_, net)| *net)
                .collect();
            let medium = classify(adapter, &addrs);
            let mut flags = IfaceFlags::empty();
            flags.set(IfaceFlags::UP, adapter.up);
            flags.set(
                IfaceFlags::MULTICAST,
                matches!(
                    medium,
                    NetMedium::Ethernet | NetMedium::WiFi | NetMedium::ThunderboltIp
                ),
            );
            flags.set(
                IfaceFlags::DEFAULT_ROUTE,
                parsed.defaults.contains(&adapter.index),
            );
            Observed {
                name: adapter.name.clone(),
                medium,
                flags,
                addrs,
                mtu: adapter.mtu,
                speed_mbps: adapter
                    .speed_bps
                    .and_then(|bps| u32::try_from(bps / 1_000_000).ok())
                    .filter(|&mbps| mbps > 0),
                bridge: None,
            }
        })
        .collect()
}

/// Runs [`SCRIPT`] and observes this host's adapters.
#[cfg(target_os = "windows")]
pub(crate) fn gather() -> Result<Vec<Observed>, IfaceError> {
    let text = super::run(
        "powershell.exe",
        &["-NoProfile", "-NonInteractive", "-Command", SCRIPT],
    )?;
    Ok(observe(&parse(&text)?))
}

#[cfg(test)]
mod tests {
    //! The fixture follows the output [`SCRIPT`] prints from PowerShell's NetAdapter cmdlets.
    //! It is synthetic, written from that format, not recorded on a rig.

    use super::*;

    const OUTPUT: &str = "\
A\t12\tEthernet\tIntel(R) Ethernet Controller I225-V\t6\tUp\t1500\t2500000000\tPCI\\VEN_8086&DEV_15F3\\3&11583659&0&E8\r
A\t15\tWi-Fi\tIntel(R) Wi-Fi 6E AX211 160MHz\t71\tUp\t1500\t1200000000\tPCI\\VEN_8086&DEV_51F0\\3&11583659&0&A3\r
A\t21\tEthernet 3\tThunderbolt(TM) Networking\t6\tUp\t1500\t20000000000\t\r
A\t22\tEthernet 4\tUSB Network Adapter\t6\tDisconnected\t1500\t0\tTHUNDERBOLT\\0000\\1\r
A\t30\tTailscale\tTailscale Tunnel\t53\tUp\t1280\t100000000000\t\r
A\t31\tvEthernet (Lab)\tHyper-V Virtual Ethernet Adapter\t6\tUp\t\t10000000000\t\r
I\t12\t192.168.1.30\t24\r
I\t12\tfe80::a1b2:c3d4:e5f6:7788%12\t64\r
I\t15\t192.168.1.31\t24\r
I\t30\t100.71.3.9\t32\r
I\t30\tfd7a:115c:a1e0::dead\t128\r
I\t1\t127.0.0.1\t8\r
R\t12\r
R\t15\r
";

    #[test]
    fn parses_the_script_output() {
        let parsed = parse(OUTPUT).unwrap();
        assert_eq!(parsed.adapters.len(), 6);
        let eth = &parsed.adapters[0];
        assert_eq!(
            (eth.index, eth.name.as_str(), eth.if_type, eth.up),
            (12, "Ethernet", 6, true)
        );
        assert_eq!(eth.speed_bps, Some(2_500_000_000));
        assert_eq!(parsed.adapters[3].speed_bps, None);
        assert!(!parsed.adapters[3].up);
        assert_eq!(parsed.adapters[5].mtu, 0);
        let link_local = parsed
            .addrs
            .iter()
            .find(|(_, n)| n.scope_id.is_some())
            .unwrap();
        assert_eq!(
            (link_local.0, link_local.1.scope_id, link_local.1.prefix_len),
            (12, Some(12), 64)
        );
        assert_eq!(
            parsed.defaults.iter().copied().collect::<Vec<_>>(),
            [12, 15]
        );
        assert!(parse("X\t1\n").is_err());
        assert!(parse("A\tnot-a-number\tEthernet\td\t6\tUp\t1500\t0\t\n").is_err());
    }

    #[test]
    fn classifies_each_row_of_the_table() {
        let observed = observe(&parse(OUTPUT).unwrap());
        let by = |name: &str| observed.iter().find(|o| o.name == name).unwrap();
        assert_eq!(by("Ethernet").medium, NetMedium::Ethernet);
        assert_eq!(by("Ethernet").speed_mbps, Some(2500));
        assert!(
            by("Ethernet")
                .flags
                .contains(IfaceFlags::UP | IfaceFlags::MULTICAST | IfaceFlags::DEFAULT_ROUTE)
        );
        assert_eq!(by("Ethernet").addrs.len(), 2);
        assert_eq!(by("Wi-Fi").medium, NetMedium::WiFi);
        // The Thunderbolt networking driver, and an IfType 6 adapter under a Thunderbolt
        // parent.
        assert_eq!(by("Ethernet 3").medium, NetMedium::ThunderboltIp);
        assert_eq!(by("Ethernet 4").medium, NetMedium::ThunderboltIp);
        assert!(!by("Ethernet 4").flags.contains(IfaceFlags::UP));
        assert_eq!(by("Tailscale").medium, NetMedium::Overlay);
        assert!(!by("Tailscale").flags.contains(IfaceFlags::MULTICAST));
        assert_eq!(by("vEthernet (Lab)").medium, NetMedium::Ethernet);
        // Metered is not known on Windows yet (DIST-E6-T08).
        assert!(
            observed
                .iter()
                .all(|o| !o.flags.contains(IfaceFlags::METERED))
        );
    }

    #[test]
    fn a_virtual_adapter_with_a_tailnet_address_is_an_overlay() {
        let adapter = Adapter {
            index: 40,
            name: String::from("wg-home"),
            description: String::from("WireGuard Tunnel"),
            if_type: IF_TYPE_PROP_VIRTUAL,
            up: true,
            ..Adapter::default()
        };
        assert_eq!(classify(&adapter, &[]), NetMedium::Overlay);
        let ethernet = Adapter {
            if_type: IF_TYPE_ETHERNET,
            ..adapter
        };
        let tailnet = IpNet::new("100.64.9.9".parse().unwrap(), 32).unwrap();
        assert_eq!(classify(&ethernet, &[tailnet]), NetMedium::Overlay);
        assert_eq!(classify(&ethernet, &[]), NetMedium::Ethernet);
    }
}
