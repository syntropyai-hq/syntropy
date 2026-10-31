// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ticket the listener prints and the tester copies to the other machine: the listener's node
//! ID, and the relay and direct addresses it knows. A ticket holds addresses, so it is shown on
//! the terminal only and never written to a report.
//!
//! The form is `snp1:<node ID in hex>`, then `,relay=<URL>` and `,addr=<socket address>` as many
//! times as needed. With iroh's public relays, the node ID alone is enough: their online lookup
//! finds the rest.

use std::net::SocketAddr;

use syntropy_net::endpoint::{NodeAddr, RelayUrl};

const PREFIX: &str = "snp1:";

/// The ticket for `addr`.
pub fn encode(addr: &NodeAddr) -> String {
    let mut out = String::from(PREFIX);
    for byte in addr.node_id() {
        out.push_str(&format!("{byte:02x}"));
    }
    for url in addr.relay_urls() {
        out.push_str(&format!(",relay={url}"));
    }
    for direct in addr.direct_addrs() {
        out.push_str(&format!(",addr={direct}"));
    }
    out
}

/// Why a ticket does not parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TicketError(pub String);

impl std::fmt::Display for TicketError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "not a probe ticket: {}", self.0)
    }
}

impl std::error::Error for TicketError {}

/// The address in a ticket. Surrounding whitespace is ignored.
pub fn decode(ticket: &str) -> Result<NodeAddr, TicketError> {
    let rest = ticket
        .trim()
        .strip_prefix(PREFIX)
        .ok_or_else(|| TicketError(format!("it does not start with `{PREFIX}`")))?;
    let mut parts = rest.split(',');
    let hex = parts.next().unwrap_or_default();
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(TicketError("the node ID is not 64 hex digits".to_owned()));
    }
    let mut id = [0u8; 32];
    for (index, byte) in id.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * index..2 * index + 2], 16)
            .map_err(|_| TicketError("the node ID is not hex".to_owned()))?;
    }
    let mut addr = NodeAddr::new(id).map_err(|error| TicketError(error.to_string()))?;
    for part in parts {
        if let Some(url) = part.strip_prefix("relay=") {
            let url: RelayUrl = url
                .parse()
                .map_err(|error| TicketError(format!("{error}")))?;
            addr = addr.with_relay_url(url);
        } else if let Some(direct) = part.strip_prefix("addr=") {
            let direct: SocketAddr = direct
                .parse()
                .map_err(|_| TicketError("an address does not parse".to_owned()))?;
            addr = addr.with_direct_addr(direct);
        } else {
            return Err(TicketError("an unknown field".to_owned()));
        }
    }
    Ok(addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A valid node ID: the public key of an all-ones secret key, from a bound endpoint.
    async fn some_id() -> [u8; 32] {
        let endpoint = syntropy_net::endpoint::Endpoint::bind(
            [1; 32],
            syntropy_net::endpoint::Config::default(),
        )
        .await
        .unwrap();
        let id = endpoint.node_id();
        endpoint.close().await;
        id
    }

    #[tokio::test]
    async fn a_ticket_round_trips() {
        let id = some_id().await;
        let addr = NodeAddr::new(id)
            .unwrap()
            .with_relay_url("https://relay.example.test./".parse().unwrap())
            .with_direct_addr("192.0.2.4:4433".parse().unwrap())
            .with_direct_addr("[2001:db8::4]:4433".parse().unwrap());
        let ticket = encode(&addr);
        assert!(ticket.starts_with("snp1:"));
        let back = decode(&format!("  {ticket}\n")).unwrap();
        assert_eq!(back.node_id(), id);
        assert_eq!(back.relay_urls(), addr.relay_urls());
        let mut direct = back.direct_addrs();
        direct.sort();
        let mut expected = addr.direct_addrs();
        expected.sort();
        assert_eq!(direct, expected);
        assert_eq!(
            decode(&encode(&NodeAddr::new(id).unwrap()))
                .unwrap()
                .node_id(),
            id
        );
    }

    #[test]
    fn a_bad_ticket_is_refused() {
        for bad in [
            "",
            "snp2:00",
            "snp1:xyz",
            &format!("snp1:{}", "0".repeat(63)),
            &format!("snp1:{},port=1", "a".repeat(64)),
        ] {
            assert!(decode(bad).is_err(), "{bad}");
        }
    }
}
