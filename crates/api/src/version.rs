// SPDX-License-Identifier: MIT OR Apache-2.0

//! What `GET /v1/syntropy/version` reports: the build, the protocol majors and the support tier
//! (RFC-0050 §1.2).
//!
//! The protocol majors come from the crates that own each protocol. The build and the support
//! tier are given by the daemon: the `BuildId` type and its construction in the `cli` crate are
//! QUAL-E16-T03's, and the platform's tier is ADR-0020's matrix applied to the host.

use std::fmt::Write as _;

use serde::Serialize;

/// The body of `GET /v1/syntropy/version`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
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

/// The protocol majors this build speaks (RFC-0062 §10.1). Two builds interoperate on a plane
/// only with equal majors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
    fn the_current_majors_are_the_owning_crates() {
        let majors = ProtocolMajors::current();
        assert_eq!(majors.wire, syntropy_net::transport::stream::PROTO_MAJOR);
        assert_eq!(majors.ipc, syntropy_ipc::cmd::PROTOCOL.major);
    }
}
