// SPDX-License-Identifier: MIT OR Apache-2.0

//! The no-identifier scan: a report must carry no IP address, MAC address or SSID.
//!
//! The report's schema leaves no room for an SSID: every string in it is an enum value or the
//! probe's version, and unknown keys are refused ([`crate::report`]). This scan checks the text
//! itself for addresses, so it also catches a number or a value that a future field would let
//! through. It splits the text into runs of hexadecimal digits, `.` and `:` (and, for MAC
//! addresses, `-`), and tries each run as an IPv4 address, an IPv6 address, a socket address and a
//! MAC address. The JSON around the values cannot form one of these: a key always ends with `"`
//! before its `:`.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

/// What the scan found: the kind and the byte offset, never the value itself, so the scan's own
/// output repeats no identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Finding {
    /// The kind of identifier.
    pub kind: Identifier,
    /// Where it starts in the text.
    pub offset: usize,
}

/// A kind of identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Identifier {
    /// An IPv4 or IPv6 address, with or without a port.
    IpAddress,
    /// A MAC address.
    MacAddress,
    /// An SSID or BSSID key.
    WifiName,
}

/// Every identifier in `text`.
pub fn scan(text: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    for (offset, token) in tokens(text, |c| c.is_ascii_hexdigit() || c == '.' || c == ':') {
        // A sentence's full stop or a stray colon may end a run. A leading colon may be compact
        // JSON's (`"k":192.0.2.1`) or part of the address (`::1`), so both readings are tried.
        let token = token.trim_end_matches(['.', ':']);
        if is_ip(token) || is_ip(token.trim_start_matches(':')) {
            findings.push(Finding {
                kind: Identifier::IpAddress,
                offset,
            });
        } else if is_mac(token, ':') {
            findings.push(Finding {
                kind: Identifier::MacAddress,
                offset,
            });
        }
    }
    for (offset, token) in tokens(text, |c| c.is_ascii_hexdigit() || c == '-') {
        if is_mac(token, '-') {
            findings.push(Finding {
                kind: Identifier::MacAddress,
                offset,
            });
        }
    }
    // A bracketed IPv6 socket address, `[::1]:80`, splits at the brackets; its inside is caught as
    // an address above. Wi-Fi names have no fixed form, so only a key that would carry one is
    // looked for: "ssid", which also covers "bssid".
    let lower = text.to_ascii_lowercase();
    let mut from = 0;
    while let Some(at) = lower[from..].find("ssid") {
        findings.push(Finding {
            kind: Identifier::WifiName,
            offset: from + at,
        });
        from += at + "ssid".len();
    }
    findings.sort_by_key(|finding| finding.offset);
    findings.dedup();
    findings
}

/// The maximal runs of characters that satisfy `keep`, with their byte offsets.
fn tokens(text: &str, keep: impl Fn(char) -> bool) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut start = None;
    for (index, c) in text.char_indices() {
        match (keep(c), start) {
            (true, None) => start = Some(index),
            (false, Some(from)) => {
                out.push((from, &text[from..index]));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(from) = start {
        out.push((from, &text[from..]));
    }
    out
}

fn is_ip(token: &str) -> bool {
    if token.parse::<Ipv4Addr>().is_ok() || token.parse::<SocketAddr>().is_ok() {
        return true;
    }
    // An IPv6 address needs at least two colons; a lone `a:b` is not one.
    token.matches(':').count() >= 2
        && (token.parse::<Ipv6Addr>().is_ok() || token.parse::<IpAddr>().is_ok())
}

fn is_mac(token: &str, separator: char) -> bool {
    let groups: Vec<&str> = token.split(separator).collect();
    groups.len() == 6
        && groups
            .iter()
            .all(|group| group.len() == 2 && group.chars().all(|c| c.is_ascii_hexdigit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(text: &str) -> Vec<Identifier> {
        scan(text).into_iter().map(|finding| finding.kind).collect()
    }

    #[test]
    fn addresses_in_any_form_are_found() {
        // Documentation ranges (RFC 5737, RFC 3849) and a locally administered MAC address.
        for text in [
            "\"x\": \"192.0.2.10\"",
            "at 198.51.100.7:4433,",
            "2001:db8::1",
            "[2001:db8::2]:443",
            "fe80::1%eth0",
            "::1",
            "{\"compact\":192.0.2.11}",
            "{\"v\":\"2001:0db8:0000:0000:0000:0000:0000:0003\"}",
        ] {
            assert_eq!(kinds(text), [Identifier::IpAddress], "{text}");
        }
        for text in ["02:00:00:aa:bb:cc", "02-00-00-AA-BB-CC"] {
            assert_eq!(kinds(text), [Identifier::MacAddress], "{text}");
        }
        assert_eq!(
            kinds("{\"ssid\": 1, \"bssid\": 2}"),
            [Identifier::WifiName, Identifier::WifiName]
        );
    }

    #[test]
    fn report_shaped_json_is_clean() {
        let text = r#"{
  "probe_version": "0.0.0",
  "settings": { "period_ms": 1000, "timeout_ms": 10000 },
  "ends": { "local": { "medium": "wifi", "wifi": { "generation": "wifi6", "band": "2.4ghz" } } },
  "rtt_us": [812, null, 1500, 99999],
  "stall": { "start_ms": 12000, "gap_ms": 3001 },
  "rtt": { "p50": 1, "p99": 2, "p999": 3 },
  "negative": -1, "dead": "beef"
}"#;
        assert_eq!(scan(text), []);
    }

    #[test]
    fn a_finding_reports_its_offset_not_its_value() {
        let text = "ok 192.0.2.1";
        let found = scan(text);
        assert_eq!(
            found,
            [Finding {
                kind: Identifier::IpAddress,
                offset: 3,
            }]
        );
        assert!(!format!("{found:?}").contains("192"));
    }
}
