// SPDX-License-Identifier: MIT OR Apache-2.0

//! This machine's end of a hop: the medium of the interface a path leaves from, and for Wi-Fi
//! the generation and band.
//!
//! The probe asks the platform's own tools, rather than OS APIs, so it needs no `unsafe` (RFC-0001
//! §4.2 rule 8 and its note of 2026-10-22 send OS calls that need it to `io`):
//!
//! | OS | Interface for a destination | Medium | Wi-Fi generation and band |
//! |---|---|---|---|
//! | Linux | `ip -o route get <ip>` | `/sys/class/net/<if>` as RFC-0030 §7.2 reads it | `iw dev <if> link` |
//! | macOS | `route -n get <ip>` | `networksetup -listallhardwareports` | `system_profiler SPAirPortDataType` |
//! | Windows | `Find-NetRoute` and `Get-NetAdapter` (PowerShell) | the adapter's `InterfaceType` and description | `netsh wlan show interfaces` |
//!
//! The destination is the peer's address on a direct path, and 192.0.2.1 on a relayed path: a
//! documentation address (RFC 5737) whose route is the default route. A route lookup sends no
//! packet. A loopback peer is `Loopback` without a lookup, and a failed lookup for a peer is
//! `Unknown`, never the default route's medium. The parsers read only the lines they need: the tools also print the network's name,
//! the access point's and the interface's MAC addresses and the machine's addresses, and none of
//! that is kept. The Windows parsers read English output only; other languages give `Unknown`.

use std::io::Read;
use std::net::{IpAddr, Ipv4Addr};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::report::{EndInfo, Medium, Os, Wifi, WifiBand, WifiGeneration};

/// Finds this machine's end of a path.
pub trait Detect: Send + Sync + 'static {
    /// The end for a path to `peer`, or for a relayed path when `peer` is `None`. It may block
    /// while platform tools run, so async callers run it on a blocking thread.
    fn local_end(&self, peer: Option<IpAddr>) -> EndInfo;
}

/// The same end every time, for tests.
#[derive(Clone, Copy, Debug)]
pub struct Fixed(pub EndInfo);

impl Detect for Fixed {
    fn local_end(&self, _peer: Option<IpAddr>) -> EndInfo {
        self.0
    }
}

/// The platform's tools.
#[derive(Clone, Copy, Debug, Default)]
pub struct Platform;

/// The destination whose route is the default route.
const DEFAULT_ROUTE: IpAddr = IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1));

/// How long one tool may run.
const TOOL_TIMEOUT: Duration = Duration::from_secs(15);

impl Detect for Platform {
    fn local_end(&self, peer: Option<IpAddr>) -> EndInfo {
        let os = Os::current();
        // A loopback peer is on this host, whatever the tools would say about its route.
        if peer.is_some_and(|ip| ip.is_loopback()) {
            return EndInfo {
                os,
                medium: Medium::Loopback,
                wifi: None,
            };
        }
        let (medium, wifi) = match os {
            Os::Linux | Os::Android => linux(peer),
            Os::Macos => macos(peer),
            Os::Windows => windows(peer),
            Os::Ios | Os::Other => (Medium::Unknown, None),
        };
        let wifi = (medium == Medium::Wifi).then(|| wifi.unwrap_or(UNKNOWN_WIFI));
        EndInfo { os, medium, wifi }
    }
}

const UNKNOWN_WIFI: Wifi = Wifi {
    generation: WifiGeneration::Unknown,
    band: WifiBand::Unknown,
};

/// Runs `program` with `args` and returns its standard output if it succeeds within
/// [`TOOL_TIMEOUT`]. Its standard error is discarded.
fn run(program: &str, args: &[&str]) -> Option<String> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    // Read on another thread, so a large output cannot fill the pipe while this one waits.
    let mut stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut out = Vec::new();
        stdout.read_to_end(&mut out).map(|_| out)
    });
    let deadline = Instant::now() + TOOL_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    let out = reader.join().ok()?.ok()?;
    status
        .success()
        .then(|| String::from_utf8_lossy(&out).into_owned())
}

/// The route lookup for `peer`, or for the default route when there is no peer (a relayed path).
/// A failed lookup for a peer gives `None`, and so `Unknown`: the default route's interface need
/// not be the one the peer's path leaves from.
fn lookup<T>(peer: Option<IpAddr>, by: impl Fn(IpAddr) -> Option<T>) -> Option<T> {
    by(peer.unwrap_or(DEFAULT_ROUTE))
}

fn linux(peer: Option<IpAddr>) -> (Medium, Option<Wifi>) {
    let Some(device) = lookup(peer, |ip| {
        route_device_linux(&run("ip", &["-o", "route", "get", &ip.to_string()])?)
    }) else {
        return (Medium::Unknown, None);
    };
    let base = Path::new("/sys/class/net").join(&device);
    let wireless = base.join("wireless").exists() || base.join("phy80211").exists();
    let driver = std::fs::read_link(base.join("device/driver"))
        .ok()
        .and_then(|link| {
            link.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        });
    let arp_type = std::fs::read_to_string(base.join("type"))
        .ok()
        .and_then(|text| text.trim().parse().ok());
    let medium = classify_linux(&device, wireless, driver.as_deref(), arp_type);
    let wifi = (medium == Medium::Wifi)
        .then(|| run("iw", &["dev", &device, "link"]))
        .flatten()
        .and_then(|text| wifi_from_iw(&text));
    (medium, wifi)
}

fn macos(peer: Option<IpAddr>) -> (Medium, Option<Wifi>) {
    let Some(device) = lookup(peer, |ip| {
        route_device_macos(&run("route", &["-n", "get", &ip.to_string()])?)
    }) else {
        return (Medium::Unknown, None);
    };
    let port = run("networksetup", &["-listallhardwareports"])
        .and_then(|text| hardware_port_macos(&text, &device));
    let medium = classify_macos(&device, port.as_deref());
    let wifi = (medium == Medium::Wifi)
        .then(|| run("system_profiler", &["SPAirPortDataType"]))
        .flatten()
        .and_then(|text| wifi_from_system_profiler(&text));
    (medium, wifi)
}

fn windows(peer: Option<IpAddr>) -> (Medium, Option<Wifi>) {
    let adapter = lookup(peer, |ip| {
        // The address is formatted by `IpAddr`, so it holds only digits, hex letters, `.`, `:`
        // and `%`.
        // The route's interface index comes first: `Get-NetAdapter` does not list the loopback
        // pseudo-interface, so it fails for a loopback route, after the index is printed.
        let script = format!(
            "$ErrorActionPreference = 'Stop'; \
             $r = Find-NetRoute -RemoteIPAddress '{ip}' | Where-Object {{ $_.InterfaceIndex }} | \
             Select-Object -First 1; \
             'ifIndex=' + $r.InterfaceIndex; \
             try {{ $a = Get-NetAdapter -InterfaceIndex $r.InterfaceIndex; \
             'ifType=' + $a.InterfaceType; 'desc=' + $a.InterfaceDescription }} catch {{ }}"
        );
        adapter_windows(&run(
            "powershell",
            &["-NoProfile", "-NonInteractive", "-Command", &script],
        )?)
    });
    let medium = adapter.map_or(Medium::Unknown, |route| classify_windows(&route));
    let wifi = (medium == Medium::Wifi)
        .then(|| run("netsh", &["wlan", "show", "interfaces"]))
        .flatten()
        .and_then(|text| wifi_from_netsh(&text));
    (medium, wifi)
}

/// The interface after `dev` in `ip -o route get` output.
pub fn route_device_linux(output: &str) -> Option<String> {
    let mut words = output.split_whitespace();
    words.find(|&word| word == "dev")?;
    words.next().map(str::to_owned)
}

/// RFC-0030 §7.2's Linux rules, in its order: Wi-Fi even when the type says Ethernet.
pub fn classify_linux(
    name: &str,
    wireless: bool,
    driver: Option<&str>,
    arp_type: Option<u32>,
) -> Medium {
    if name == "lo" || arp_type == Some(772) {
        Medium::Loopback
    } else if wireless {
        Medium::Wifi
    } else if matches!(driver, Some("thunderbolt-net" | "thunderbolt_net")) {
        Medium::ThunderboltIp
    } else if ["tailscale", "wg", "tun"]
        .iter()
        .any(|prefix| name.starts_with(prefix))
    {
        Medium::Overlay
    } else if arp_type == Some(1) {
        Medium::Ethernet
    } else {
        Medium::Unknown
    }
}

/// The interface in `route -n get` output's `interface:` line.
pub fn route_device_macos(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "interface").then(|| value.trim().to_owned())
    })
}

/// The hardware port that `device` belongs to, from `networksetup -listallhardwareports`.
pub fn hardware_port_macos(output: &str, device: &str) -> Option<String> {
    let mut port = None;
    for line in output.lines() {
        if let Some(name) = line.strip_prefix("Hardware Port:") {
            port = Some(name.trim().to_owned());
        } else if let Some(name) = line.strip_prefix("Device:")
            && name.trim() == device
        {
            return port;
        }
    }
    None
}

/// RFC-0030 §7.2's macOS rules, by hardware port name.
pub fn classify_macos(device: &str, port: Option<&str>) -> Medium {
    let port = port.unwrap_or("");
    if device.starts_with("lo") {
        Medium::Loopback
    } else if ["utun", "ipsec", "ppp"]
        .iter()
        .any(|prefix| device.starts_with(prefix))
    {
        Medium::Overlay
    } else if port == "Wi-Fi" || port == "AirPort" {
        Medium::Wifi
    } else if port.contains("Thunderbolt") || device.starts_with("bridge") {
        Medium::ThunderboltIp
    } else if port.contains("Ethernet") || port.contains("LAN") {
        Medium::Ethernet
    } else {
        Medium::Unknown
    }
}

/// A route's interface on Windows: its index, and the adapter's `InterfaceType` and description
/// when `Get-NetAdapter` lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowsRoute {
    /// The route's interface index.
    pub if_index: u32,
    /// The adapter's `InterfaceType` (`IfType`).
    pub if_type: Option<u32>,
    /// The adapter's description.
    pub description: String,
}

/// The route's interface from the PowerShell lines `ifIndex=`, `ifType=` and `desc=`. `None`
/// without an interface index.
pub fn adapter_windows(output: &str) -> Option<WindowsRoute> {
    let mut if_index = None;
    let mut if_type = None;
    let mut description = String::new();
    for line in output.lines() {
        if let Some(value) = line.strip_prefix("ifIndex=") {
            if_index = value.trim().parse().ok();
        } else if let Some(value) = line.strip_prefix("ifType=") {
            if_type = value.trim().parse().ok();
        } else if let Some(value) = line.strip_prefix("desc=") {
            description = value.trim().to_owned();
        }
    }
    Some(WindowsRoute {
        if_index: if_index?,
        if_type,
        description,
    })
}

/// The index of Windows's loopback pseudo-interface.
const WINDOWS_LOOPBACK_INDEX: u32 = 1;

/// RFC-0030 §7.2's Windows rules: the loopback pseudo-interface, then the overlay and Thunderbolt
/// descriptions, then `InterfaceType` (`IfType`).
pub fn classify_windows(route: &WindowsRoute) -> Medium {
    let description = route.description.as_str();
    if route.if_index == WINDOWS_LOOPBACK_INDEX {
        Medium::Loopback
    } else if description.contains("Tailscale") || description.contains("WireGuard") {
        Medium::Overlay
    } else if description.contains("Thunderbolt") {
        Medium::ThunderboltIp
    } else {
        match route.if_type {
            Some(71) => Medium::Wifi,
            Some(6) => Medium::Ethernet,
            Some(24) => Medium::Loopback,
            Some(53) => Medium::Overlay,
            _ => Medium::Unknown,
        }
    }
}

/// The generation for an IEEE PHY mode such as `802.11ax`.
fn generation_from_phy(phy: &str) -> WifiGeneration {
    let phy = phy.to_ascii_lowercase().replace(' ', "");
    if phy.contains("802.11be") {
        WifiGeneration::Wifi7
    } else if phy.contains("802.11ax") {
        WifiGeneration::Wifi6
    } else if phy.contains("802.11ac") {
        WifiGeneration::Wifi5
    } else if phy.contains("802.11n") {
        WifiGeneration::Wifi4
    } else if ["802.11a", "802.11b", "802.11g"]
        .iter()
        .any(|old| phy.contains(old))
    {
        WifiGeneration::Older
    } else {
        WifiGeneration::Unknown
    }
}

fn band_from_mhz(mhz: f64) -> WifiBand {
    match mhz {
        f if (2_400.0..2_500.0).contains(&f) => WifiBand::Ghz2_4,
        f if (4_900.0..5_925.0).contains(&f) => WifiBand::Ghz5,
        f if (5_925.0..7_125.0).contains(&f) => WifiBand::Ghz6,
        _ => WifiBand::Unknown,
    }
}

/// `iw dev <if> link`: the band from `freq:`, the generation from the MCS kind in the bitrate
/// lines. `None` when not connected.
pub fn wifi_from_iw(output: &str) -> Option<Wifi> {
    if output.trim_start().starts_with("Not connected") {
        return None;
    }
    let mut band = WifiBand::Unknown;
    let mut generation = WifiGeneration::Unknown;
    for line in output.lines().map(str::trim) {
        if let Some(value) = line.strip_prefix("freq:") {
            if let Ok(mhz) = value.trim().parse::<f64>() {
                band = band_from_mhz(mhz);
            }
        } else if line.starts_with("rx bitrate:") || line.starts_with("tx bitrate:") {
            let found = if line.contains("EHT-MCS") {
                WifiGeneration::Wifi7
            } else if line.contains("HE-MCS") {
                WifiGeneration::Wifi6
            } else if line.contains("VHT-MCS") {
                WifiGeneration::Wifi5
            } else if line.split_whitespace().any(|word| word == "MCS") {
                WifiGeneration::Wifi4
            } else {
                WifiGeneration::Older
            };
            generation = newest(generation, found);
        }
    }
    Some(Wifi { generation, band })
}

/// The newer of two generations; `Unknown` loses to anything.
fn newest(a: WifiGeneration, b: WifiGeneration) -> WifiGeneration {
    let order = |generation| match generation {
        WifiGeneration::Unknown => 0,
        WifiGeneration::Older => 1,
        WifiGeneration::Wifi4 => 2,
        WifiGeneration::Wifi5 => 3,
        WifiGeneration::Wifi6 => 4,
        WifiGeneration::Wifi7 => 5,
    };
    if order(b) > order(a) { b } else { a }
}

/// `netsh wlan show interfaces`: the first connected interface's `Radio type` and `Band`. Without
/// a `Band` line, a channel up to 14 is 2.4 GHz and one from 32 is 5 GHz, unless the radio type
/// allows 6 GHz, whose channel numbers overlap both.
pub fn wifi_from_netsh(output: &str) -> Option<Wifi> {
    let mut blocks: Vec<Vec<(String, String)>> = Vec::new();
    for line in output.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let (key, value) = (key.trim().to_owned(), value.trim().to_owned());
        if key == "Name" || blocks.is_empty() {
            blocks.push(Vec::new());
        }
        if let Some(block) = blocks.last_mut() {
            block.push((key, value));
        }
    }
    let field = |block: &[(String, String)], name: &str| {
        block
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.clone())
    };
    let block = blocks
        .iter()
        .find(|block| field(block, "State").is_some_and(|state| state == "connected"))?;
    let generation =
        field(block, "Radio type").map_or(WifiGeneration::Unknown, |phy| generation_from_phy(&phy));
    let band = match field(block, "Band") {
        Some(band) => match band.replace(' ', "").to_ascii_lowercase().as_str() {
            "2.4ghz" => WifiBand::Ghz2_4,
            "5ghz" => WifiBand::Ghz5,
            "6ghz" => WifiBand::Ghz6,
            _ => WifiBand::Unknown,
        },
        None => {
            let six_possible = matches!(generation, WifiGeneration::Wifi6 | WifiGeneration::Wifi7);
            match field(block, "Channel").and_then(|channel| channel.parse::<u32>().ok()) {
                Some(1..=14) if !six_possible => WifiBand::Ghz2_4,
                Some(32..=177) if !six_possible => WifiBand::Ghz5,
                _ => WifiBand::Unknown,
            }
        }
    };
    Some(Wifi { generation, band })
}

/// `system_profiler SPAirPortDataType`: `PHY Mode` and the band in `Channel`'s parentheses, inside
/// the "Current Network Information" section.
pub fn wifi_from_system_profiler(output: &str) -> Option<Wifi> {
    let mut lines = output.lines();
    let header = lines.find(|line| line.trim() == "Current Network Information:")?;
    let indent = header.len() - header.trim_start().len();
    let mut generation = WifiGeneration::Unknown;
    let mut band = WifiBand::Unknown;
    for line in lines {
        let trimmed = line.trim_start();
        if !trimmed.is_empty() && line.len() - trimmed.len() <= indent {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("PHY Mode:") {
            generation = generation_from_phy(value);
        } else if let Some(value) = trimmed.strip_prefix("Channel:") {
            let value = value.to_ascii_lowercase();
            band = if value.contains("(2ghz") || value.contains("(2.4ghz") {
                WifiBand::Ghz2_4
            } else if value.contains("(5ghz") {
                WifiBand::Ghz5
            } else if value.contains("(6ghz") {
                WifiBand::Ghz6
            } else {
                WifiBand::Unknown
            };
        }
    }
    Some(Wifi { generation, band })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The tool outputs below are written for these tests in each tool's format, with
    // documentation addresses (RFC 5737), locally administered MAC addresses and a made-up network
    // name. They are not captured from a machine.

    const IW_WIFI6: &str = "Connected to 02:00:00:00:00:01 (on wlan0)
\tSSID: example-net
\tfreq: 5180.0
\tRX: 1234 bytes (10 packets)
\tsignal: -50 dBm
\trx bitrate: 1200.9 MBit/s 80MHz HE-MCS 11 HE-NSS 2 HE-GI 0 HE-DCM 0
\ttx bitrate: 864.8 MBit/s 80MHz HE-MCS 8 HE-NSS 2 HE-GI 0 HE-DCM 0
";

    #[test]
    fn iw_gives_generation_and_band() {
        assert_eq!(
            wifi_from_iw(IW_WIFI6),
            Some(Wifi {
                generation: WifiGeneration::Wifi6,
                band: WifiBand::Ghz5,
            })
        );
        let wifi4 = "Connected to 02:00:00:00:00:01 (on wlp2s0)\n\tfreq: 2437\n\
                     \ttx bitrate: 72.2 MBit/s MCS 7 short GI\n";
        assert_eq!(
            wifi_from_iw(wifi4),
            Some(Wifi {
                generation: WifiGeneration::Wifi4,
                band: WifiBand::Ghz2_4,
            })
        );
        let wifi7_6ghz = "Connected to 02:00:00:00:00:01 (on wlan0)\n\tfreq: 6115\n\
                          \ttx bitrate: 2882.4 MBit/s 160MHz EHT-MCS 13 EHT-NSS 2 EHT-GI 0\n";
        assert_eq!(
            wifi_from_iw(wifi7_6ghz),
            Some(Wifi {
                generation: WifiGeneration::Wifi7,
                band: WifiBand::Ghz6,
            })
        );
        assert_eq!(wifi_from_iw("Not connected.\n"), None);
    }

    const NETSH: &str = "
There is 1 interface on the system:

    Name                   : Wi-Fi
    Description            : Example Wireless Adapter
    GUID                   : 00000000-0000-0000-0000-000000000000
    Physical address       : 02:00:00:00:00:01
    Interface type         : Primary
    State                  : connected
    SSID                   : example-net
    AP BSSID               : 02:00:00:00:00:02
    Band                   : 6 GHz
    Channel                : 37
    Network type           : Infrastructure
    Radio type             : 802.11ax
    Authentication         : WPA3-Personal
";

    #[test]
    fn netsh_gives_the_connected_interfaces_generation_and_band() {
        assert_eq!(
            wifi_from_netsh(NETSH),
            Some(Wifi {
                generation: WifiGeneration::Wifi6,
                band: WifiBand::Ghz6,
            })
        );
        // An older Windows prints no `Band`; the channel decides when 6 GHz is impossible.
        let no_band = NETSH
            .replace("    Band                   : 6 GHz\n", "")
            .replace("802.11ax", "802.11ac")
            .replace(": 37", ": 44");
        assert_eq!(
            wifi_from_netsh(&no_band),
            Some(Wifi {
                generation: WifiGeneration::Wifi5,
                band: WifiBand::Ghz5,
            })
        );
        let ambiguous = NETSH.replace("    Band                   : 6 GHz\n", "");
        assert_eq!(wifi_from_netsh(&ambiguous).unwrap().band, WifiBand::Unknown);
        assert_eq!(
            wifi_from_netsh(&NETSH.replace(": connected", ": disconnected")),
            None
        );
    }

    const SYSTEM_PROFILER: &str = "Wi-Fi:

      Interfaces:
        en0:
          Card Type: Wi-Fi
          MAC Address: 02:00:00:00:00:01
          Supported PHY Modes: 802.11 a/b/g/n/ac/ax
          Status: Connected
          Current Network Information:
            example-net:
              PHY Mode: 802.11ac
              Channel: 149 (5GHz, 80MHz)
              Security: WPA2 Personal
          Other Local Wi-Fi Networks:
            other-net:
              PHY Mode: 802.11n
              Channel: 6 (2GHz, 20MHz)
";

    #[test]
    fn system_profiler_reads_the_current_network_only() {
        assert_eq!(
            wifi_from_system_profiler(SYSTEM_PROFILER),
            Some(Wifi {
                generation: WifiGeneration::Wifi5,
                band: WifiBand::Ghz5,
            })
        );
        assert_eq!(wifi_from_system_profiler("Wi-Fi:\n  Status: Off\n"), None);
    }

    #[test]
    fn route_lookups_name_the_interface() {
        assert_eq!(
            route_device_linux(
                "192.0.2.1 via 198.51.100.1 dev wlp3s0 src 198.51.100.7 uid 1000 \\    cache "
            ),
            Some("wlp3s0".to_owned())
        );
        assert_eq!(
            route_device_linux("RTNETLINK answers: Network is unreachable"),
            None
        );
        let macos = "   route to: 192.0.2.1\ndestination: default\n       mask: default\n    \
                     gateway: 198.51.100.1\n  interface: en0\n      flags: <UP,GATEWAY,DONE>\n";
        assert_eq!(route_device_macos(macos), Some("en0".to_owned()));
        let ports = "\nHardware Port: Ethernet\nDevice: en1\nEthernet Address: 02:00:00:00:00:03\n\n\
                     Hardware Port: Wi-Fi\nDevice: en0\nEthernet Address: 02:00:00:00:00:01\n";
        assert_eq!(hardware_port_macos(ports, "en0"), Some("Wi-Fi".to_owned()));
        assert_eq!(
            hardware_port_macos(ports, "en1"),
            Some("Ethernet".to_owned())
        );
        assert_eq!(hardware_port_macos(ports, "en9"), None);
        assert_eq!(
            adapter_windows("ifIndex=12\nifType=71\ndesc=Example Wireless Adapter\n"),
            Some(WindowsRoute {
                if_index: 12,
                if_type: Some(71),
                description: "Example Wireless Adapter".to_owned(),
            })
        );
        // The loopback route: `Get-NetAdapter` failed after the index was printed.
        assert_eq!(
            adapter_windows("ifIndex=1\n"),
            Some(WindowsRoute {
                if_index: 1,
                if_type: None,
                description: String::new(),
            })
        );
        assert_eq!(adapter_windows("error\n"), None);
    }

    #[test]
    fn interfaces_are_classified_as_rfc_0030_section_7_2_says() {
        assert_eq!(
            classify_linux("lo", false, None, Some(772)),
            Medium::Loopback
        );
        // Wi-Fi wins even where the type says Ethernet.
        assert_eq!(classify_linux("wlan0", true, None, Some(1)), Medium::Wifi);
        assert_eq!(
            classify_linux("thunderbolt0", false, Some("thunderbolt-net"), Some(1)),
            Medium::ThunderboltIp
        );
        assert_eq!(
            classify_linux("tailscale0", false, None, Some(65534)),
            Medium::Overlay
        );
        assert_eq!(
            classify_linux("eth0", false, Some("e1000e"), Some(1)),
            Medium::Ethernet
        );
        assert_eq!(
            classify_linux("can0", false, None, Some(280)),
            Medium::Unknown
        );

        assert_eq!(classify_macos("en0", Some("Wi-Fi")), Medium::Wifi);
        assert_eq!(classify_macos("en1", Some("Ethernet")), Medium::Ethernet);
        assert_eq!(
            classify_macos("en5", Some("USB 10/100/1000 LAN")),
            Medium::Ethernet
        );
        assert_eq!(
            classify_macos("bridge0", Some("Thunderbolt Bridge")),
            Medium::ThunderboltIp
        );
        assert_eq!(
            classify_macos("en2", Some("Thunderbolt 1")),
            Medium::ThunderboltIp
        );
        assert_eq!(classify_macos("utun3", None), Medium::Overlay);
        assert_eq!(classify_macos("lo0", None), Medium::Loopback);

        let windows = |if_index, if_type, description: &str| {
            classify_windows(&WindowsRoute {
                if_index,
                if_type,
                description: description.to_owned(),
            })
        };
        assert_eq!(
            windows(12, Some(71), "Example Wireless Adapter"),
            Medium::Wifi
        );
        assert_eq!(
            windows(12, Some(6), "Example Gigabit Controller"),
            Medium::Ethernet
        );
        assert_eq!(
            windows(12, Some(6), "Thunderbolt(TM) Networking"),
            Medium::ThunderboltIp
        );
        assert_eq!(windows(12, Some(53), "Tailscale Tunnel"), Medium::Overlay);
        assert_eq!(windows(12, Some(131), "Example Tunnel"), Medium::Unknown);
        // The loopback pseudo-interface, which `Get-NetAdapter` does not list.
        assert_eq!(windows(1, None, ""), Medium::Loopback);
        assert_eq!(windows(12, None, ""), Medium::Unknown);
    }

    #[test]
    fn a_failed_peer_lookup_is_unknown_not_the_default_route() {
        let peer = IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7));
        let asked = std::cell::RefCell::new(Vec::new());
        // The peer's lookup fails, and the default route's would succeed: no fallback.
        let found = lookup(Some(peer), |ip| {
            asked.borrow_mut().push(ip);
            (ip == DEFAULT_ROUTE).then_some("eth0")
        });
        assert_eq!(found, None);
        assert_eq!(*asked.borrow(), [peer]);
        // With no peer (a relayed path), the default route is the one asked.
        assert_eq!(
            lookup(None, |ip| (ip == DEFAULT_ROUTE).then_some("eth0")),
            Some("eth0")
        );
    }

    #[test]
    fn the_platform_finds_loopback_for_a_loopback_peer() {
        // On every host, before any tool runs: a loopback peer is on this host.
        for peer in [
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)),
            IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
        ] {
            let end = Platform.local_end(Some(peer));
            assert_eq!(end.medium, Medium::Loopback, "{peer}: {end:?}");
            assert_eq!(end.wifi, None);
            assert_eq!(end.os, Os::current());
        }
    }
}
