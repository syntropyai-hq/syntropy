// SPDX-License-Identifier: MIT OR Apache-2.0

//! What `GET /v1/syntropy/version` reports: the build, the protocol majors and the support tier
//! (RFC-0050 §1.2).
//!
//! The protocol majors come from the crates that own each protocol. The build and the support
//! tier are given by the daemon: the `BuildId` type and its construction in the `cli` crate are
//! QUAL-E16-T03's, and the platform's tier is ADR-0020's matrix applied to the host.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

/// The body of `GET /v1/syntropy/version`, which `syntropy version --json` prints unchanged
/// (RFC-0051 §2.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionInfo {
    /// The build serving the request.
    pub build: Build,
    /// The major version of each protocol this build speaks.
    pub protocols: ProtocolMajors,
    /// This node's platform support level (ADR-0020).
    pub support_tier: SupportTier,
}

/// A build: its product version and its build ID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Build {
    /// The product version, such as `0.1.0`.
    pub version: String,
    /// The 32-byte build ID the daemon and its runners compare at the IPC handshake (ADR-0015
    /// decision 1), reported as 64 lower-case hex digits.
    pub id: [u8; 32],
}

impl Serialize for Build {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;
        let mut id = String::with_capacity(64);
        for byte in self.id {
            let _ = write!(id, "{byte:02x}");
        }
        let mut build = serializer.serialize_struct("Build", 2)?;
        build.serialize_field("version", &self.version)?;
        build.serialize_field("id", &id)?;
        build.end()
    }
}

impl<'de> Deserialize<'de> for Build {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Text {
            version: String,
            id: String,
        }
        let text = Text::deserialize(deserializer)?;
        let digits = text.id.as_bytes();
        if digits.len() != 64
            || !digits
                .iter()
                .all(|d| matches!(d, b'0'..=b'9' | b'a'..=b'f'))
        {
            return Err(serde::de::Error::custom(
                "a build ID is 64 lower-case hex digits",
            ));
        }
        let mut id = [0u8; 32];
        let (pairs, _) = digits.as_chunks::<2>();
        for (byte, [high, low]) in id.iter_mut().zip(pairs) {
            let hex = |d: u8| {
                if d.is_ascii_digit() {
                    d - b'0'
                } else {
                    d - b'a' + 10
                }
            };
            *byte = hex(*high) << 4 | hex(*low);
        }
        Ok(Build {
            version: text.version,
            id,
        })
    }
}

/// The protocol majors this build speaks (RFC-0062 §10.1). Two builds interoperate on a plane
/// only with equal majors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolMajors {
    /// `proto_major` of the data and control planes, which every ALPN, beacon and `Hello`
    /// carries (RFC-0031 §10).
    pub wire: u16,
    /// The daemon-runner IPC protocol's major (RFC-0001 §3.5).
    pub ipc: u16,
}

impl ProtocolMajors {
    /// The majors of this build: `syntropy_net`'s wire protocol and `syntropy_ipc`'s command
    /// channel.
    #[must_use]
    pub fn current() -> ProtocolMajors {
        ProtocolMajors {
            wire: syntropy_net::transport::stream::PROTO_MAJOR,
            ipc: syntropy_ipc::cmd::PROTOCOL.major,
        }
    }
}

/// A platform's support level (ADR-0020): what the project promises for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportTier {
    /// Tier 1: supported; parity and performance gates enforced.
    Tier1,
    /// Tier 2: supported with known gaps; parity enforced, performance tracked.
    Tier2,
    /// Tier 3: best effort.
    Tier3,
    /// Outside ADR-0020's matrix: it may work, and nothing is reported as a regression.
    Unsupported,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_body_names_the_build_protocols_and_tier() {
        let info = VersionInfo {
            build: Build {
                version: "0.1.0".into(),
                id: [0xab; 32],
            },
            protocols: ProtocolMajors { wire: 1, ipc: 1 },
            support_tier: SupportTier::Tier1,
        };
        let body = serde_json::to_value(&info).unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "build": { "version": "0.1.0", "id": "ab".repeat(32) },
                "protocols": { "wire": 1, "ipc": 1 },
                "support_tier": "tier1",
            })
        );
    }

    #[test]
    fn the_body_reads_back_and_a_bad_build_id_is_refused() {
        let info = VersionInfo {
            build: Build {
                version: "0.1.0".into(),
                id: core::array::from_fn(|i| i as u8 * 7),
            },
            protocols: ProtocolMajors { wire: 1, ipc: 2 },
            support_tier: SupportTier::Tier3,
        };
        let text = serde_json::to_string(&info).unwrap();
        assert_eq!(serde_json::from_str::<VersionInfo>(&text).unwrap(), info);
        for id in ["ab", &"AB".repeat(32), &"zz".repeat(32)] {
            let text = format!(
                r#"{{"build":{{"version":"0","id":"{id}"}},"protocols":{{"wire":1,"ipc":1}},"support_tier":"tier1"}}"#
            );
            assert!(serde_json::from_str::<VersionInfo>(&text).is_err(), "{id}");
        }
    }

    #[test]
    fn the_current_majors_are_the_owning_crates() {
        let majors = ProtocolMajors::current();
        assert_eq!(majors.wire, syntropy_net::transport::stream::PROTO_MAJOR);
        assert_eq!(majors.ipc, syntropy_ipc::cmd::PROTOCOL.major);
    }
}
