// SPDX-License-Identifier: MIT OR Apache-2.0

//! The probe's datagrams, on the protocol [`ALPN`].
//!
//! An echo request and its echo are each [`PAYLOAD_BYTES`] long, padded with zeros: a magic
//! number, a kind byte and the request's sequence number, little-endian. The echo adds the
//! echoing machine's end: its operating system, medium and Wi-Fi details, one byte each, so the
//! report can name both ends of the hop. The size is a heartbeat's order of magnitude; RFC-0030
//! §5.2 caps a heartbeat body at 1,150 bytes ([`MAX_PAYLOAD`]).

use crate::report::{EndInfo, Medium, Os, Wifi, WifiBand, WifiGeneration};

/// The probe's application protocol.
pub const ALPN: &[u8] = b"syntropy/netprobe/1";

/// The size of each request and echo.
pub const PAYLOAD_BYTES: usize = 128;

/// RFC-0030 §5.2's cap on a heartbeat body.
pub const MAX_PAYLOAD: usize = 1_150;

const MAGIC: [u8; 4] = *b"SNP1";
const REQUEST: u8 = 1;
const ECHO: u8 = 2;

/// A decoded datagram.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Message {
    /// An echo request.
    Request {
        /// Its sequence number.
        seq: u64,
    },
    /// An echo.
    Echo {
        /// The request's sequence number.
        seq: u64,
        /// The echoing machine's end.
        end: EndInfo,
    },
}

impl Message {
    /// The datagram for this message.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![0u8; PAYLOAD_BYTES];
        out[..4].copy_from_slice(&MAGIC);
        match *self {
            Self::Request { seq } => {
                out[4] = REQUEST;
                out[5..13].copy_from_slice(&seq.to_le_bytes());
            }
            Self::Echo { seq, end } => {
                out[4] = ECHO;
                out[5..13].copy_from_slice(&seq.to_le_bytes());
                out[13..17].copy_from_slice(&encode_end(end));
            }
        }
        out
    }

    /// The message in `data`, or `None` if it is not one of the probe's.
    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() != PAYLOAD_BYTES || data[..4] != MAGIC {
            return None;
        }
        let seq = u64::from_le_bytes(data[5..13].try_into().ok()?);
        match data[4] {
            REQUEST => Some(Self::Request { seq }),
            ECHO => Some(Self::Echo {
                seq,
                end: decode_end(data[13..17].try_into().ok()?)?,
            }),
            _ => None,
        }
    }
}

const OSES: [Os; 6] = [
    Os::Linux,
    Os::Windows,
    Os::Macos,
    Os::Android,
    Os::Ios,
    Os::Other,
];
const MEDIA: [Medium; 6] = [
    Medium::Ethernet,
    Medium::Wifi,
    Medium::ThunderboltIp,
    Medium::Overlay,
    Medium::Loopback,
    Medium::Unknown,
];
const GENERATIONS: [WifiGeneration; 6] = [
    WifiGeneration::Wifi4,
    WifiGeneration::Wifi5,
    WifiGeneration::Wifi6,
    WifiGeneration::Wifi7,
    WifiGeneration::Older,
    WifiGeneration::Unknown,
];
const BANDS: [WifiBand; 4] = [
    WifiBand::Ghz2_4,
    WifiBand::Ghz5,
    WifiBand::Ghz6,
    WifiBand::Unknown,
];

/// An end as four bytes: indices into the tables above, plus one, with 0 for "no Wi-Fi".
fn encode_end(end: EndInfo) -> [u8; 4] {
    let index = |found: Option<usize>| found.map_or(0, |at| at as u8 + 1);
    [
        index(OSES.iter().position(|&os| os == end.os)),
        index(MEDIA.iter().position(|&medium| medium == end.medium)),
        index(end.wifi.and_then(|wifi| {
            GENERATIONS
                .iter()
                .position(|&generation| generation == wifi.generation)
        })),
        index(
            end.wifi
                .and_then(|wifi| BANDS.iter().position(|&band| band == wifi.band)),
        ),
    ]
}

fn decode_end(bytes: [u8; 4]) -> Option<EndInfo> {
    fn pick<T: Copy>(table: &[T], byte: u8) -> Option<T> {
        table.get(usize::from(byte).checked_sub(1)?).copied()
    }
    let wifi = match (bytes[2], bytes[3]) {
        (0, 0) => None,
        (generation, band) => Some(Wifi {
            generation: pick(&GENERATIONS, generation)?,
            band: pick(&BANDS, band)?,
        }),
    };
    Some(EndInfo {
        os: pick(&OSES, bytes[0])?,
        medium: pick(&MEDIA, bytes[1])?,
        wifi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_message_round_trips() {
        let mut ends = vec![];
        for os in OSES {
            for medium in MEDIA {
                ends.push(EndInfo {
                    os,
                    medium,
                    wifi: None,
                });
            }
        }
        for generation in GENERATIONS {
            for band in BANDS {
                ends.push(EndInfo {
                    os: Os::Linux,
                    medium: Medium::Wifi,
                    wifi: Some(Wifi { generation, band }),
                });
            }
        }
        for (seq, end) in ends.into_iter().enumerate() {
            let seq = seq as u64 * 0x0101_0101_0101;
            for message in [Message::Request { seq }, Message::Echo { seq, end }] {
                let bytes = message.encode();
                assert_eq!(bytes.len(), PAYLOAD_BYTES);
                assert_eq!(Message::decode(&bytes), Some(message));
            }
        }
    }

    #[test]
    fn other_datagrams_are_ignored() {
        let mut bytes = Message::Request { seq: 1 }.encode();
        assert_eq!(Message::decode(&bytes[..64]), None);
        bytes[0] = b'X';
        assert_eq!(Message::decode(&bytes), None);
        let mut bytes = Message::Request { seq: 1 }.encode();
        bytes[4] = 9;
        assert_eq!(Message::decode(&bytes), None);
        let mut bytes = Message::Echo {
            seq: 1,
            end: EndInfo {
                os: Os::Linux,
                medium: Medium::Ethernet,
                wifi: None,
            },
        }
        .encode();
        bytes[14] = 200;
        assert_eq!(Message::decode(&bytes), None);
    }
}
