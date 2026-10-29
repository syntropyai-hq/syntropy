// SPDX-License-Identifier: MIT OR Apache-2.0

//! Address redaction: interface addresses, endpoint addresses and relay URLs replaced by stable
//! per-bundle placeholders, unless `--with-addresses` is given (RFC-0053 §6).
//!
//! Two kinds of text are replaced, wherever they occur in a string the bundle writes:
//!
//! - **URLs**, `scheme://…` up to the next space, quote or bracket, which relay URLs and
//!   endpoint URLs are: `<url-N>`.
//! - **IP addresses**, alone or with a port, IPv4 or IPv6, bracketed or not, with or without a
//!   zone, and inside a longer token (`p2p.peers.2001:db8::5.x`, `remote_addr:203.0.113.7`):
//!   `<addr-N>`. Text is an address when the standard library parses it as one, or when it is
//!   four dotted decimal octets, zero-padded or not, with an optional port. A candidate with no
//!   digit counts only when no letter or digit touches it, so `Vec::new` stays.
//!
//! With `--with-addresses` both are kept, except a URL's userinfo (`user:password@`), which is a
//! credential and is replaced by `[redacted]@`.
//!
//! Host names that are not in a URL are kept, with or without a port, so a member's name stays
//! readable. The same text
//! gets the same placeholder throughout one bundle, and the numbering starts again in the next,
//! so placeholders say which records name one address and nothing about the address itself.
//! Every place a replacement was made is recorded, as `file: path`, for `MANIFEST.txt`.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, Ipv6Addr, SocketAddr};

/// Replaces addresses, or with `--with-addresses` keeps them, and records where it replaced.
#[derive(Debug, Default)]
pub(crate) struct Redactor {
    keep: bool,
    placeholders: BTreeMap<String, String>,
    addrs: usize,
    urls: usize,
    /// Every `file: path` a replacement was made in.
    pub(crate) fields: BTreeSet<String>,
    /// Every `file: path` a URL's userinfo (`user:password@`) was removed from. Userinfo is a
    /// credential, not an address, so it goes even with `--with-addresses`.
    pub(crate) credentials: BTreeSet<String>,
}

impl Redactor {
    /// A redactor; `keep` is `--with-addresses`.
    pub(crate) fn new(keep: bool) -> Redactor {
        Redactor {
            keep,
            ..Redactor::default()
        }
    }

    /// Whether addresses are kept.
    pub(crate) fn keeps(&self) -> bool {
        self.keep
    }

    /// How many distinct addresses and URLs were replaced.
    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.addrs, self.urls)
    }

    fn placeholder(&mut self, text: &str, url: bool) -> String {
        if let Some(p) = self.placeholders.get(text) {
            return p.clone();
        }
        let p = if url {
            self.urls += 1;
            format!("<url-{}>", self.urls)
        } else {
            self.addrs += 1;
            format!("<addr-{}>", self.addrs)
        };
        self.placeholders.insert(text.to_owned(), p.clone());
        p
    }

    /// `text` with every URL and IP address in it replaced; `at` names where it is, for the
    /// record of redacted fields.
    pub(crate) fn scan(&mut self, text: &str, at: &str) -> String {
        if self.keep {
            let out = self.scan_urls(text);
            if out != text {
                self.credentials.insert(at.to_owned());
            }
            return out;
        }
        let out = self.scan_urls(text);
        let out = self.scan_ips(&out);
        if out != text {
            self.fields.insert(at.to_owned());
        }
        out
    }

    fn scan_urls(&mut self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(at) = rest.find("://") {
            // The scheme: letters, digits, `+`, `-` and `.` before `://`, starting with a letter.
            let head = &rest[..at];
            let scheme_len = head
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
                .map(char::len_utf8)
                .sum::<usize>();
            let start = at - scheme_len;
            if scheme_len == 0 || !head[start..].starts_with(|c: char| c.is_ascii_alphabetic()) {
                out.push_str(&rest[..at + 3]);
                rest = &rest[at + 3..];
                continue;
            }
            let tail = &rest[at + 3..];
            let len = tail
                .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | '`'))
                .unwrap_or(tail.len());
            let mut url = &rest[start..at + 3 + len];
            // Punctuation that ends a sentence is not part of the URL.
            url = url.trim_end_matches(['.', ',', ';', ')', ']', '}']);
            out.push_str(&rest[..start]);
            if self.keep {
                out.push_str(&strip_userinfo(url));
            } else {
                let p = self.placeholder(url, true);
                out.push_str(&p);
            }
            rest = &rest[start + url.len()..];
        }
        out.push_str(rest);
        out
    }

    /// Every IP address in `text`, wherever it starts: alone, inside a longer token such as
    /// `p2p.peers.2001:db8::5.x` or `remote_addr:203.0.113.7:7433`, bracketed, with a zone or a
    /// port, and IPv4 with zero-padded octets. A run of characters an address can hold is
    /// searched from each place an address can start, taking the longest text there that is
    /// one; a run with no `:` and no `.` holds none and is skipped whole.
    fn scan_ips(&mut self, text: &str) -> String {
        let bytes = text.as_bytes();
        let in_run =
            |b: u8| b.is_ascii_alphanumeric() || matches!(b, b':' | b'.' | b'%' | b'[' | b']');
        let starts = |b: u8| b.is_ascii_hexdigit() || matches!(b, b'[' | b':');
        let mut out = String::with_capacity(text.len());
        let mut copied = 0;
        let mut i = 0;
        while i < bytes.len() {
            if !in_run(bytes[i]) {
                i += 1;
                continue;
            }
            // Runs are ASCII, so every index in one is a char boundary.
            let end = (i..bytes.len())
                .find(|&j| !in_run(bytes[j]))
                .unwrap_or(bytes.len());
            if !bytes[i..end].iter().any(|&b| b == b':' || b == b'.') {
                i = end;
                continue;
            }
            let mut at = i;
            while at < end {
                if !starts(bytes[at]) {
                    at += 1;
                    continue;
                }
                let limit = end.min(at + MAX_ADDRESS_LEN);
                // A candidate with no digit, such as `ec::` in `Vec::new`, counts only when no
                // letter or digit touches it, as for `dead::beef` standing alone.
                let clean = |j: usize| {
                    (at == 0 || !bytes[at - 1].is_ascii_alphanumeric())
                        && bytes.get(j).is_none_or(|b| !b.is_ascii_alphanumeric())
                };
                let found = (at + 2..=limit).rev().find(|&j| {
                    let t = &text[at..j];
                    is_address(t) && (t.bytes().any(|b| b.is_ascii_digit()) || clean(j))
                });
                match found {
                    Some(stop) => {
                        out.push_str(&text[copied..at]);
                        let p = self.placeholder(&text[at..stop], false);
                        out.push_str(&p);
                        copied = stop;
                        at = stop;
                    }
                    None => at += 1,
                }
            }
            i = end;
        }
        out.push_str(&text[copied..]);
        out
    }
}

/// `url` with its userinfo, the `user:password@` before the host, replaced by `[redacted]@`.
fn strip_userinfo(url: &str) -> String {
    let Some(after) = url.find("://").map(|i| i + 3) else {
        return url.to_owned();
    };
    let authority_len = url[after..]
        .find(['/', '?', '#'])
        .unwrap_or(url.len() - after);
    match url[after..after + authority_len].rfind('@') {
        Some(at) => format!("{}[redacted]{}", &url[..after], &url[after + at..]),
        None => url.to_owned(),
    }
}

/// The longest text an address with a zone and a port can take.
const MAX_ADDRESS_LEN: usize = 64;

/// Whether `token` is four dotted decimal octets, zero-padded or not, with an optional `:port`.
fn is_dotted_quad(token: &str) -> bool {
    let host = match token.split_once(':') {
        Some((host, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => host,
        Some(_) => return false,
        None => token,
    };
    let octets: Vec<&str> = host.split('.').collect();
    octets.len() == 4
        && octets.iter().all(|o| {
            (1..=3).contains(&o.len())
                && o.bytes().all(|b| b.is_ascii_digit())
                && o.parse::<u16>().is_ok_and(|n| n <= 255)
        })
}

/// Whether `token` is an IP address, alone or with a port, bracketed or with a zone.
fn is_address(token: &str) -> bool {
    if !token.contains(['.', ':']) {
        return false;
    }
    if is_dotted_quad(token) {
        return true;
    }
    if token.parse::<IpAddr>().is_ok() || token.parse::<SocketAddr>().is_ok() {
        return true;
    }
    // `fe80::1%eth0`, `[fe80::1%eth0]`, `[fe80::1%eth0]:80`, `[::1]`.
    let inner = match token.strip_prefix('[') {
        Some(rest) => match rest.split_once(']') {
            Some((inner, after)) => {
                if !(after.is_empty()
                    || after
                        .strip_prefix(':')
                        .is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())))
                {
                    return false;
                }
                inner
            }
            None => return false,
        },
        None => token,
    };
    let host = inner.split_once('%').map_or(inner, |(host, _)| host);
    host.parse::<Ipv6Addr>().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_and_urls_get_stable_placeholders() {
        let mut r = Redactor::new(false);
        let text = "peer 192.0.2.7:7433 via https://relay.example.net:443/x, then 192.0.2.7 and \
                    [2001:db8::5]:9000 or fe80::1%eth0; v6 2001:db8::5.";
        let out = r.scan(text, "f: a");
        assert_eq!(
            out,
            "peer <addr-1> via <url-1>, then <addr-2> and <addr-3> or <addr-4>; v6 <addr-5>."
        );
        // The same text, the same placeholder.
        assert_eq!(r.scan("again 192.0.2.7:7433", "f: b"), "again <addr-1>");
        assert_eq!(
            r.scan("p2p.peers.198.51.100.4:9.x", "f: c"),
            "p2p.peers.<addr-6>.x"
        );
        assert_eq!(r.counts(), (6, 1));
        assert_eq!(
            r.fields.iter().collect::<Vec<_>>(),
            ["f: a", "f: b", "f: c"]
        );
    }

    #[test]
    fn other_text_is_left_alone() {
        let mut r = Redactor::new(false);
        for text in [
            "syntropy_daemon::logging at 12:30:45.123",
            "node-a.local:7433 is a host name",
            "version 1.99.1, layers 0..32, ratio 3:1",
            "a hex id 9f86d081884c7d65",
        ] {
            assert_eq!(r.scan(text, "f: x"), text);
        }
        assert!(r.fields.is_empty());
    }

    #[test]
    fn with_addresses_keeps_addresses_but_not_url_credentials() {
        let mut r = Redactor::new(true);
        let text = "peer 192.0.2.7:7433 via https://relay.example.net";
        assert_eq!(r.scan(text, "f: a"), text);
        assert!(r.fields.is_empty());
        assert_eq!(
            r.scan(
                "pull https://user:tok3n@hub.example.org:443/x?y=1@z",
                "f: b"
            ),
            "pull https://[redacted]@hub.example.org:443/x?y=1@z"
        );
        assert_eq!(r.credentials.iter().collect::<Vec<_>>(), ["f: b"]);
        assert!(r.fields.is_empty());
    }

    /// The review of #1967, B1: IPv6 inside a longer token, after a `word:` prefix, bracketed,
    /// with a zone, and IPv4 with zero-padded octets.
    #[test]
    fn addresses_inside_longer_tokens_are_found() {
        let mut r = Redactor::new(false);
        for (text, want) in [
            ("p2p.peers.2001:db8::5.x", "p2p.peers.<addr-1>.x"),
            ("remote_addr:2001:db8::77", "remote_addr:<addr-2>"),
            ("203.000.113.077", "<addr-3>"),
            ("peer=203.000.113.077:7433;", "peer=<addr-4>;"),
            ("k.[2001:db8::9]:443.v", "k.<addr-5>.v"),
            ("via fe80::1%eth0, done", "via <addr-6>, done"),
            ("key.198.51.100.4.tail", "key.<addr-7>.tail"),
            ("dead::beef", "<addr-8>"),
        ] {
            assert_eq!(r.scan(text, "f: a"), want, "{text}");
        }
        // Paths, times, versions, ratios and digests stay.
        for text in [
            "Vec::new and syntropy_daemon::logging",
            "at 12:30:45.123 on 2026-10-29",
            "1.99.1 and 3:1 and 0..32",
            "id 9f86d081884c7d65f9f86d081884c7d65",
        ] {
            assert_eq!(r.scan(text, "f: b"), text, "{text}");
        }
    }
}
