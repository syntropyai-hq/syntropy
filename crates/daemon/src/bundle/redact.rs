// SPDX-License-Identifier: MIT OR Apache-2.0

//! Address redaction: interface addresses, endpoint addresses and relay URLs replaced by stable
//! per-bundle placeholders, unless `--with-addresses` is given (RFC-0053 §6).
//!
//! Two kinds of text are replaced, wherever they occur in a string the bundle writes:
//!
//! - **URLs**, `scheme://…` up to the next space, quote or bracket, which relay URLs and
//!   endpoint URLs are: `<url-N>`.
//! - **IP addresses**, alone or with a port, IPv4 or IPv6, bracketed or not, with or without a
//!   zone: `<addr-N>`. A token is an address when the standard library parses it as one.
//!
//! Host names that are not in a URL are kept, so a member's name stays readable. The same text
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
            return text.to_owned();
        }
        let out = self.scan_urls(text);
        let out = self.scan_ips(&out);
        let out = self.scan_ipv4_runs(&out);
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
            let p = self.placeholder(url, true);
            out.push_str(&p);
            rest = &rest[start + url.len()..];
        }
        out.push_str(rest);
        out
    }

    fn scan_ips(&mut self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut token_start: Option<usize> = None;
        let is_token =
            |c: char| c.is_ascii_alphanumeric() || matches!(c, ':' | '.' | '%' | '[' | ']');
        let flush = |this: &mut Redactor, out: &mut String, token: &str| {
            let trimmed = token.trim_end_matches(['.', ':']);
            if is_address(trimmed) {
                let p = this.placeholder(trimmed, false);
                out.push_str(&p);
                out.push_str(&token[trimmed.len()..]);
            } else {
                out.push_str(token);
            }
        };
        for (i, c) in text.char_indices() {
            if is_token(c) {
                token_start.get_or_insert(i);
            } else {
                if let Some(start) = token_start.take() {
                    flush(self, &mut out, &text[start..i]);
                }
                out.push(c);
            }
        }
        if let Some(start) = token_start {
            flush(self, &mut out, &text[start..]);
        }
        out
    }
}

impl Redactor {
    /// IPv4 addresses inside a longer token, such as `peers.203.0.113.7:7433` in a key path: each
    /// run of digits and dots that is four octets, with the port that follows it if any.
    fn scan_ipv4_runs(&mut self, text: &str) -> String {
        let bytes = text.as_bytes();
        let mut out = String::with_capacity(text.len());
        let mut i = 0;
        let mut copied = 0;
        while i < bytes.len() {
            if !bytes[i].is_ascii_digit() {
                i += 1;
                continue;
            }
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            let run = text[start..i].trim_end_matches('.');
            let mut end = start + run.len();
            if run.parse::<std::net::Ipv4Addr>().is_err() {
                continue;
            }
            // A port: `:` and digits.
            if bytes.get(end) == Some(&b':') {
                let digits = bytes[end + 1..]
                    .iter()
                    .take_while(|b| b.is_ascii_digit())
                    .count();
                if digits > 0 {
                    end += 1 + digits;
                }
            }
            out.push_str(&text[copied..start]);
            let p = self.placeholder(&text[start..end], false);
            out.push_str(&p);
            copied = end;
            i = end;
        }
        out.push_str(&text[copied..]);
        out
    }
}

/// Whether `token` is an IP address, alone or with a port, bracketed or with a zone.
fn is_address(token: &str) -> bool {
    if token.is_empty() || !token.contains(['.', ':']) {
        return false;
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
    fn with_addresses_keeps_everything() {
        let mut r = Redactor::new(true);
        let text = "peer 192.0.2.7:7433 via https://relay.example.net";
        assert_eq!(r.scan(text, "f: a"), text);
        assert!(r.fields.is_empty());
    }
}
