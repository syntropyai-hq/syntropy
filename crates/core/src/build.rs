// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`BuildId`], the identity of a build (RFC-0062 §8.1; ADR-0621 decision 1; RFC-0062 amendment
//! C1 to RFC-0001 §5.2).
//!
//! Only the type lives here. The value is built in `cli`, the top binary crate, when the binary
//! is built, and passed down at run time, so `core` does not depend on the commit, and a new
//! commit does not invalidate every crate's compilation cache.
//!
//! In a human-readable format such as JSON, `commit` and `toolchain` are lower-case hex, and the
//! version and profile are their text. In a binary format such as postcard, the digests are raw
//! bytes.

use std::fmt;
use std::str::FromStr;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ids::Blake3;

/// A build's identity. Owner: RFC-0062.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BuildId {
    /// The product version, from `version.txt` (RFC-0062 §8.1).
    pub version: SemVer,
    /// The source commit, or all zeros when the build had none to read (a source archive with
    /// no git checkout and no `SYNTROPY_BUILD_COMMIT`). A release names the tag's commit.
    #[serde(with = "hex20")]
    pub commit: [u8; 20],
    /// The cargo profile it was built with (RFC-0062 §4.4).
    pub profile: BuildProfile,
    /// The target it was built for.
    pub target: TargetTriple,
    /// The BLAKE3 digest of the compiler's `rustc -vV`, so two toolchains never share an ID.
    #[serde(with = "hex32")]
    pub toolchain: Blake3,
}

impl BuildId {
    /// Whether the build knew its commit.
    #[must_use]
    pub fn commit_known(&self) -> bool {
        self.commit != [0; 20]
    }
}

/// A SemVer version, `MAJOR.MINOR.PATCH` with an optional pre-release (`-rc.1`). Written and
/// read as its text in every format.
///
/// It has no order: SemVer 2.0.0 §11's precedence puts `1.0.0-rc.1` below `1.0.0` and compares
/// numeric identifiers as numbers, which no derived order does. The task that first compares
/// versions adds it, with its own test. Build metadata (`+…`) is not part of a version here and
/// is refused.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SemVer {
    /// The major version.
    pub major: u64,
    /// The minor version.
    pub minor: u64,
    /// The patch version.
    pub patch: u64,
    /// The pre-release identifiers after `-`, if any: dot-separated, each non-empty and of
    /// `0-9`, `A-Z`, `a-z` and `-`, with no leading zero in a numeric one (SemVer 2.0.0 §9).
    pub pre: Option<String>,
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        match &self.pre {
            Some(pre) => write!(f, "-{pre}"),
            None => Ok(()),
        }
    }
}

/// Why a version did not parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemVerError(String);

impl fmt::Display for SemVerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}` is not a MAJOR.MINOR.PATCH version", self.0)
    }
}

impl std::error::Error for SemVerError {}

impl FromStr for SemVer {
    type Err = SemVerError;

    fn from_str(text: &str) -> Result<SemVer, SemVerError> {
        let error = || SemVerError(text.to_owned());
        let (core, pre) = match text.split_once('-') {
            Some((core, pre)) if pre_release(pre) => (core, Some(pre.to_owned())),
            Some(_) => return Err(error()),
            None => (text, None),
        };
        let number = |part: &str| {
            let leading_zero = part.len() > 1 && part.starts_with('0');
            if part.is_empty() || leading_zero || !part.bytes().all(|b| b.is_ascii_digit()) {
                return Err(error());
            }
            part.parse::<u64>().map_err(|_| error())
        };
        let mut parts = core.split('.');
        let (Some(major), Some(minor), Some(patch), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(error());
        };
        Ok(SemVer {
            major: number(major)?,
            minor: number(minor)?,
            patch: number(patch)?,
            pre,
        })
    }
}

/// Whether `pre` is a SemVer pre-release (SemVer 2.0.0 §9): dot-separated identifiers, each
/// non-empty and of `0-9`, `A-Z`, `a-z` and `-`, a numeric one without a leading zero.
fn pre_release(pre: &str) -> bool {
    pre.split('.').all(|id| {
        let numeric = id.bytes().all(|b| b.is_ascii_digit());
        !id.is_empty()
            && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            && !(numeric && id.len() > 1 && id.starts_with('0'))
    })
}

impl Serialize for SemVer {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for SemVer {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<SemVer, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(D::Error::custom)
    }
}

/// The cargo profile of a build: RFC-0062 §4.4's, or another a developer named. Written and read
/// as the profile's name in every format.
///
/// Two profiles are equal when they name the same profile, as [`BuildProfile::from_name`] reads
/// the name: a hand-built `Other("dev")` equals `Dev`, so every value reads back from its text as
/// a value equal to it.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BuildProfile {
    /// `dev`, cargo's own.
    Dev,
    /// `test-release`: release optimisation with debug assertions.
    TestRelease,
    /// `chaos`: release optimisation with `fault-injection`.
    Chaos,
    /// `pr`: the PR build.
    Pr,
    /// `release`: everything shipped.
    Release,
    /// Another profile, by its name. Build one with [`BuildProfile::from_name`], which gives the
    /// named variant for a name this enum has.
    Other(String),
}

impl BuildProfile {
    /// The name [`BuildProfile::from_name`] reads this profile's name as: `Other("debug")` is
    /// `dev`.
    fn canonical(&self) -> String {
        BuildProfile::from_name(self.name()).name().to_owned()
    }
}

impl PartialEq for BuildProfile {
    fn eq(&self, other: &BuildProfile) -> bool {
        self.canonical() == other.canonical()
    }
}

impl Eq for BuildProfile {}

impl std::hash::Hash for BuildProfile {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.canonical().hash(state);
    }
}

impl BuildProfile {
    /// The profile cargo names `name`. Cargo's build directory for `dev` is `debug`, which
    /// maps to [`BuildProfile::Dev`] too.
    #[must_use]
    pub fn from_name(name: &str) -> BuildProfile {
        match name {
            "dev" | "debug" => BuildProfile::Dev,
            "test-release" => BuildProfile::TestRelease,
            "chaos" => BuildProfile::Chaos,
            "pr" => BuildProfile::Pr,
            "release" => BuildProfile::Release,
            other => BuildProfile::Other(other.to_owned()),
        }
    }

    /// The profile's name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            BuildProfile::Dev => "dev",
            BuildProfile::TestRelease => "test-release",
            BuildProfile::Chaos => "chaos",
            BuildProfile::Pr => "pr",
            BuildProfile::Release => "release",
            BuildProfile::Other(name) => name,
        }
    }
}

impl fmt::Display for BuildProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Serialize for BuildProfile {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for BuildProfile {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<BuildProfile, D::Error> {
        Ok(BuildProfile::from_name(&String::deserialize(deserializer)?))
    }
}

/// A target triple, such as `x86_64-unknown-linux-gnu`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TargetTriple(pub String);

impl fmt::Display for TargetTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Lower-case hex in a human-readable format, bytes otherwise.
fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    text
}

/// `N` bytes from `2 × N` lower-case hex digits.
fn from_hex<const N: usize>(text: &str) -> Option<[u8; N]> {
    let digits = text.as_bytes();
    if digits.len() != N * 2 {
        return None;
    }
    let value = |d: u8| match d {
        b'0'..=b'9' => Some(d - b'0'),
        b'a'..=b'f' => Some(d - b'a' + 10),
        _ => None,
    };
    let mut out = [0u8; N];
    for (byte, pair) in out.iter_mut().zip(digits.as_chunks::<2>().0) {
        *byte = value(pair[0])? << 4 | value(pair[1])?;
    }
    Some(out)
}

/// Serde for `[u8; 20]`: hex when human-readable.
mod hex20 {
    use super::*;

    pub(super) fn serialize<S: Serializer>(value: &[u8; 20], s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            s.serialize_str(&to_hex(value))
        } else {
            value.serialize(s)
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 20], D::Error> {
        if d.is_human_readable() {
            let text = String::deserialize(d)?;
            from_hex(&text).ok_or_else(|| D::Error::custom("a commit is 40 lower-case hex digits"))
        } else {
            <[u8; 20]>::deserialize(d)
        }
    }
}

/// Serde for a [`Blake3`]: hex when human-readable.
mod hex32 {
    use super::*;

    pub(super) fn serialize<S: Serializer>(value: &Blake3, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            s.serialize_str(&to_hex(&value.0))
        } else {
            value.serialize(s)
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Blake3, D::Error> {
        if d.is_human_readable() {
            let text = String::deserialize(d)?;
            from_hex(&text)
                .map(Blake3)
                .ok_or_else(|| D::Error::custom("a digest is 64 lower-case hex digits"))
        } else {
            Blake3::deserialize(d)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> BuildId {
        BuildId {
            version: "1.2.3-rc.1".parse().unwrap(),
            commit: core::array::from_fn(|i| i as u8 * 13),
            profile: BuildProfile::TestRelease,
            target: TargetTriple("x86_64-unknown-linux-gnu".into()),
            toolchain: Blake3(core::array::from_fn(|i| 255 - i as u8)),
        }
    }

    #[test]
    fn versions_parse_and_print_back() {
        for text in ["0.0.0", "1.2.3", "10.20.30", "1.0.0-rc.1", "2.0.0-alpha"] {
            assert_eq!(text.parse::<SemVer>().unwrap().to_string(), text);
        }
        for bad in [
            "", "1", "1.2", "1.2.3.4", "01.2.3", "1.2.x", "1.2.3-", "v1.2.3", " 1.2.3",
        ] {
            assert!(bad.parse::<SemVer>().is_err(), "{bad}");
        }
    }

    /// The pre-release is SemVer 2.0.0 §9's: dot-separated identifiers of `0-9A-Za-z-`, none
    /// empty, a numeric one without a leading zero. Build metadata is refused (review of #1924,
    /// N1).
    #[test]
    fn a_pre_release_is_semver_s() {
        for good in [
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-0.3.7",
            "1.0.0-x.7.z.92",
            "1.0.0-x-y-z.--",
            "1.0.0-rc.10",
            "1.0.0-0",
            "1.0.0-0a",
        ] {
            assert_eq!(good.parse::<SemVer>().unwrap().to_string(), good);
        }
        for bad in [
            "1.0.0-a..b",
            "1.0.0-.a",
            "1.0.0-a.",
            "1.0.0-rc.01",
            "1.0.0-00",
            "1.0.0-a b",
            "1.0.0-a_b",
            "1.0.0-\u{1b}[2J",
            "1.0.0-rc.1+build.5",
            "1.0.0+build",
        ] {
            assert!(bad.parse::<SemVer>().is_err(), "{bad}");
        }
    }

    /// A hand-built `Other` with a named profile's name equals that profile, so it reads back
    /// from its text equal to itself (review of #1924, N2).
    #[test]
    fn every_profile_reads_back_equal_to_itself() {
        for profile in [
            BuildProfile::Dev,
            BuildProfile::TestRelease,
            BuildProfile::Chaos,
            BuildProfile::Pr,
            BuildProfile::Release,
            BuildProfile::Other("bench-local".into()),
            BuildProfile::Other("dev".into()),
            BuildProfile::Other("debug".into()),
            BuildProfile::Other("release".into()),
        ] {
            let json = serde_json::to_string(&profile).unwrap();
            let back: BuildProfile = serde_json::from_str(&json).unwrap();
            assert_eq!(back, profile, "{json}");
        }
        assert_eq!(BuildProfile::Other("dev".into()), BuildProfile::Dev);
        assert_ne!(BuildProfile::Other("bench".into()), BuildProfile::Dev);
        let set: crate::DetHashSet<BuildProfile> =
            [BuildProfile::Dev, BuildProfile::Other("debug".into())]
                .into_iter()
                .collect();
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn json_writes_hex_digests_and_text_names() {
        let id = sample();
        let json = serde_json::to_value(&id).unwrap();
        assert_eq!(json["version"], "1.2.3-rc.1");
        assert_eq!(json["profile"], "test-release");
        assert_eq!(json["target"], "x86_64-unknown-linux-gnu");
        assert_eq!(json["commit"].as_str().unwrap().len(), 40);
        assert_eq!(json["toolchain"].as_str().unwrap().len(), 64);
        assert_eq!(serde_json::from_value::<BuildId>(json).unwrap(), id);
    }

    #[test]
    fn postcard_carries_raw_bytes_and_reads_back() {
        let id = sample();
        let bytes = postcard::to_allocvec(&id).unwrap();
        assert_eq!(postcard::from_bytes::<BuildId>(&bytes).unwrap(), id);
    }

    #[test]
    fn profiles_map_cargo_s_names() {
        for (name, profile) in [
            ("dev", BuildProfile::Dev),
            ("debug", BuildProfile::Dev),
            ("test-release", BuildProfile::TestRelease),
            ("chaos", BuildProfile::Chaos),
            ("pr", BuildProfile::Pr),
            ("release", BuildProfile::Release),
            ("bench-local", BuildProfile::Other("bench-local".into())),
        ] {
            assert_eq!(BuildProfile::from_name(name), profile, "{name}");
        }
        assert_eq!(BuildProfile::Dev.to_string(), "dev");
    }

    #[test]
    fn a_zero_commit_is_unknown_and_bad_hex_is_refused() {
        let mut id = sample();
        assert!(id.commit_known());
        id.commit = [0; 20];
        assert!(!id.commit_known());
        let mut json = serde_json::to_value(sample()).unwrap();
        json["commit"] = "ABC".into();
        assert!(serde_json::from_value::<BuildId>(json).is_err());
    }
}
