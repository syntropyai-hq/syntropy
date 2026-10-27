// SPDX-License-Identifier: MIT OR Apache-2.0

//! What `GET /v1/syntropy/version` reports: the build, the protocol majors and the support tier
//! (RFC-0050 §1.2).
//!
//! The protocol majors come from the crates that own each protocol. The build and the support
//! tier are given by the daemon: the build is the `BuildId` value the `cli` crate builds
//! (RFC-0062 §8.1), and the platform's tier is ADR-0020's matrix applied to the host.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use syntropy_core::build::{BuildId, BuildProfile, SemVer, TargetTriple};
use syntropy_core::ids::Blake3;

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

/// A build: its [`BuildId`], the features it was built with, and its 32-byte build ID.
///
/// In JSON the `BuildId`'s fields sit beside the other two, digests as lower-case hex:
/// `{"version", "commit", "profile", "target", "toolchain", "features", "id"}`. An all-zero
/// `commit` means the build had none to read (QUAL-E16-T03).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Build {
    /// The version, commit, profile, target and toolchain (RFC-0062 §8.1).
    pub build_id: BuildId,
    /// The features it was built with, as `crate/feature` (RFC-0062 §8.1's build metadata).
    pub features: Vec<String>,
    /// The 32-byte build ID the daemon and its runners compare at the IPC handshake (ADR-0015
    /// decision 1): `cli`'s digest of `build_id` and `features`.
    pub id: [u8; 32],
}

/// `bytes` as lower-case hex.
fn hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// `N` bytes from `2N` lower-case hex digits.
fn unhex<const N: usize, E: serde::de::Error>(text: &str, what: &str) -> Result<[u8; N], E> {
    let digits = text.as_bytes();
    if digits.len() != 2 * N
        || !digits
            .iter()
            .all(|d| matches!(d, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(E::custom(format_args!(
            "{what} is {} lower-case hex digits",
            2 * N
        )));
    }
    let value = |d: u8| {
        if d.is_ascii_digit() {
            d - b'0'
        } else {
            d - b'a' + 10
        }
    };
    let mut out = [0u8; N];
    let (pairs, _) = digits.as_chunks::<2>();
    for (byte, [high, low]) in out.iter_mut().zip(pairs) {
        *byte = value(*high) << 4 | value(*low);
    }
    Ok(out)
}

impl Serialize for Build {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;
        let id = &self.build_id;
        let mut build = serializer.serialize_struct("Build", 7)?;
        build.serialize_field("version", &id.version)?;
        build.serialize_field("commit", &hex(&id.commit))?;
        build.serialize_field("profile", &id.profile)?;
        build.serialize_field("target", &id.target)?;
        build.serialize_field("toolchain", &hex(&id.toolchain.0))?;
        build.serialize_field("features", &self.features)?;
        build.serialize_field("id", &hex(&self.id))?;
        build.end()
    }
}

impl<'de> Deserialize<'de> for Build {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Text {
            version: SemVer,
            commit: String,
            profile: BuildProfile,
            target: TargetTriple,
            toolchain: String,
            features: Vec<String>,
            id: String,
        }
        let text = Text::deserialize(deserializer)?;
        Ok(Build {
            build_id: BuildId {
                version: text.version,
                commit: unhex(&text.commit, "a commit")?,
                profile: text.profile,
                target: text.target,
                toolchain: Blake3(unhex(&text.toolchain, "a toolchain digest")?),
            },
            features: text.features,
            id: unhex(&text.id, "a build ID")?,
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

    fn build(id: [u8; 32]) -> Build {
        Build {
            build_id: BuildId {
                version: "0.1.0-rc.1".parse().unwrap(),
                commit: [0x0c; 20],
                profile: BuildProfile::TestRelease,
                target: TargetTriple("x86_64-unknown-linux-gnu".into()),
                toolchain: Blake3([0x70; 32]),
            },
            features: vec!["cli/x".into()],
            id,
        }
    }

    #[test]
    fn the_body_names_the_build_protocols_and_tier() {
        let info = VersionInfo {
            build: build([0xab; 32]),
            protocols: ProtocolMajors { wire: 1, ipc: 1 },
            support_tier: SupportTier::Tier1,
        };
        let body = serde_json::to_value(&info).unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "build": {
                    "version": "0.1.0-rc.1",
                    "commit": "0c".repeat(20),
                    "profile": "test-release",
                    "target": "x86_64-unknown-linux-gnu",
                    "toolchain": "70".repeat(32),
                    "features": ["cli/x"],
                    "id": "ab".repeat(32),
                },
                "protocols": { "wire": 1, "ipc": 1 },
                "support_tier": "tier1",
            })
        );
    }

    #[test]
    fn the_body_reads_back_and_a_bad_build_id_is_refused() {
        let info = VersionInfo {
            build: build(core::array::from_fn(|i| i as u8 * 7)),
            protocols: ProtocolMajors { wire: 1, ipc: 2 },
            support_tier: SupportTier::Tier3,
        };
        let text = serde_json::to_string(&info).unwrap();
        assert_eq!(serde_json::from_str::<VersionInfo>(&text).unwrap(), info);
        let good = serde_json::to_value(&info).unwrap();
        for (field, bad) in [
            ("id", "ab".to_owned()),
            ("id", "AB".repeat(32)),
            ("id", "zz".repeat(32)),
            ("commit", "ab".repeat(32)),
            ("toolchain", "ab".repeat(20)),
            ("version", "0".to_owned()),
        ] {
            let mut body = good.clone();
            body["build"][field] = bad.clone().into();
            assert!(
                serde_json::from_value::<VersionInfo>(body).is_err(),
                "{field} = {bad}"
            );
        }
        let mut body = good.clone();
        body["build"]["extra"] = 1.into();
        assert!(serde_json::from_value::<VersionInfo>(body).is_err());
    }

    #[test]
    fn the_current_majors_are_the_owning_crates() {
        let majors = ProtocolMajors::current();
        assert_eq!(majors.wire, syntropy_net::transport::stream::PROTO_MAJOR);
        assert_eq!(majors.ipc, syntropy_ipc::cmd::PROTOCOL.major);
    }
}
