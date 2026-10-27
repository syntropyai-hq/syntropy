// SPDX-License-Identifier: MIT OR Apache-2.0

//! This binary's [`BuildId`], built by `cli`'s build script and passed down at run time
//! (RFC-0062 §8.1; ADR-0621 decision 1; RFC-0062 amendment C1).
//!
//! `core` holds only the type. The value is made here, in the top binary crate, so a new commit
//! recompiles `cli` alone. The daemon and its runners get it from [`current`] and the 32-byte
//! build ID from [`digest`], which the IPC `Hello` and `Init` carry and the daemon compares on a
//! fresh spawn (RFC-0001 §3.1; ADR-0015 decision 1).
//!
//! Without a git checkout and without `SYNTROPY_BUILD_COMMIT`, the commit is all zeros:
//! [`BuildId::commit_known`] is false, and `syntropy --version` says "commit unknown". Such a
//! build still pairs only with itself, because the digest also covers the version, profile,
//! target, toolchain and features.

use std::sync::OnceLock;

use syntropy_api::version::Build;
use syntropy_core::build::{BuildId, BuildProfile, SemVer, TargetTriple};
use syntropy_core::ids::Blake3;

/// The constants `build.rs` writes.
mod generated {
    include!(concat!(env!("OUT_DIR"), "/build_id.rs"));
}

/// The domain tag of [`digest`]'s input, so the build ID never equals another BLAKE3 digest of
/// the same bytes.
const DOMAIN: &[u8] = b"syntropy build id v1\0";

/// This binary's build.
#[must_use]
pub fn current() -> &'static BuildId {
    static CURRENT: OnceLock<BuildId> = OnceLock::new();
    CURRENT.get_or_init(|| BuildId {
        version: generated::VERSION
            .parse::<SemVer>()
            .expect("build.rs checked version.txt"),
        commit: generated::COMMIT,
        profile: BuildProfile::from_name(generated::PROFILE),
        target: TargetTriple(generated::TARGET.to_owned()),
        toolchain: Blake3(*blake3::hash(generated::TOOLCHAIN.as_bytes()).as_bytes()),
    })
}

/// The features this binary was built with, as `crate/feature`.
///
/// The build script sees only `cli`'s own features. A test-only feature (RFC-0001 §4.2 rule 10)
/// of a crate further down would reach this binary through cargo's feature unification without
/// `cli` seeing it, so each crate in `cli`'s dependency graph that declares one reports it here.
/// None does yet: the `a_crate_with_a_test_feature_reports_it` test fails when one joins the
/// graph, and names it.
#[must_use]
pub fn features() -> Vec<String> {
    generated::FEATURES
        .iter()
        .map(|feature| format!("cli/{feature}"))
        .collect()
}

/// The crates in `cli`'s dependency graph whose test-only features [`features`] reports.
pub const REPORTING_CRATES: &[&str] = &[];

/// The 32-byte build ID of `build` with `features`: BLAKE3 over a domain tag and each field,
/// length-prefixed. Two builds pair at the IPC handshake only when every field is equal.
#[must_use]
pub fn digest(build: &BuildId, features: &[String]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    let mut field = |bytes: &[u8]| {
        let len = u64::try_from(bytes.len()).expect("a field fits in u64");
        hasher.update(&len.to_le_bytes());
        hasher.update(bytes);
    };
    field(DOMAIN);
    field(build.version.to_string().as_bytes());
    field(&build.commit);
    field(build.profile.name().as_bytes());
    field(build.target.0.as_bytes());
    field(&build.toolchain.0);
    field(&u64::try_from(features.len()).expect("fits").to_le_bytes());
    for feature in features {
        field(feature.as_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// This binary's 32-byte build ID.
#[must_use]
pub fn id() -> [u8; 32] {
    static ID: OnceLock<[u8; 32]> = OnceLock::new();
    *ID.get_or_init(|| digest(current(), &features()))
}

/// This binary's build as the native API reports it (RFC-0050 §1.2): what `syntropy --version
/// --json` prints, and what `GET /v1/syntropy/version` carries.
#[must_use]
pub fn build() -> Build {
    Build {
        build_id: current().clone(),
        features: features(),
        id: id(),
    }
}

/// The build script's version check, compiled here too so its tests run (`build/version.rs`).
#[cfg(test)]
#[path = "../build/version.rs"]
mod build_version;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_value_is_this_build_s() {
        let build = current();
        assert!(build.target.0.contains('-'), "{}", build.target);
        // The profile is the directory cargo built into, never the fallback.
        assert_ne!(build.profile.name(), "unknown");
        assert_ne!(build.toolchain.0, [0; 32]);
        assert!(
            generated::TOOLCHAIN.starts_with("rustc "),
            "{}",
            generated::TOOLCHAIN
        );
    }

    /// The build refuses exactly the `version.txt` `SemVer` refuses, through the same parser:
    /// a pre-release outside SemVer §9, such as `0.0.0-a_b`, fails the build rather than
    /// `syntropy --version` at run time (review of #1924).
    #[test]
    fn the_build_refuses_a_version_semver_refuses() {
        for text in [
            "0.0.0",
            "1.2.3\n",
            " 1.2.3-rc.1 ",
            "1.0.0-x-y.7",
            "0.0.0-a_b",
            "1.0.0-rc.01",
            "1.0.0-a..b",
            "1.0.0+build",
            "1.2",
            "v1.2.3",
            "",
        ] {
            let built = super::build_version::checked(text);
            let parsed = text.trim().parse::<SemVer>();
            assert_eq!(built.is_ok(), parsed.is_ok(), "{text:?}");
            if let Ok(version) = built {
                assert_eq!(version, text.trim());
            }
        }
        let refused = super::build_version::checked("0.0.0-a_b").unwrap_err();
        assert!(refused.contains("0.0.0-a_b"), "{refused}");
    }

    #[test]
    fn every_field_changes_the_build_id() {
        let base = current().clone();
        let id = digest(&base, &[]);
        assert_eq!(id, digest(&base, &[]));
        let mut changed = Vec::new();
        let mut other = base.clone();
        other.commit[0] ^= 1;
        changed.push(digest(&other, &[]));
        let mut other = base.clone();
        other.version.patch += 1;
        changed.push(digest(&other, &[]));
        let mut other = base.clone();
        other.profile = BuildProfile::Release;
        changed.push(digest(&other, &[]));
        let mut other = base.clone();
        other.target = TargetTriple("aarch64-apple-darwin-x".into());
        changed.push(digest(&other, &[]));
        let mut other = base.clone();
        other.toolchain.0[31] ^= 1;
        changed.push(digest(&other, &[]));
        changed.push(digest(&base, &["obs/fault-injection".to_owned()]));
        for (n, other) in changed.iter().enumerate() {
            assert_ne!(*other, id, "change {n} kept the build ID");
        }
        // Length prefixes keep two feature lists with the same bytes apart.
        assert_ne!(
            digest(&base, &["ab".to_owned(), "c".to_owned()]),
            digest(&base, &["a".to_owned(), "bc".to_owned()])
        );
    }
}
