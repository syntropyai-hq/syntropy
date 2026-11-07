// SPDX-License-Identifier: MIT OR Apache-2.0

//! The access rule of the project bucket (ADR-0603 decision 8; RFC-0060 §12.3; ADR-0018 §7),
//! from M0: **the bucket is private**. Public read is granted object by object, and only to
//!
//! - project-synthetic fixtures (`source = { synthetic = "..." }`, the generator's output), and
//! - fixtures whose licence is permissive, one of [`PERMISSIVE`],
//!
//! each published with its licence file beside it, which gets the same access. Every other
//! object stays private to the lab and is never redistributed: a gated fixture, a Llama-family
//! file whatever its recorded licence, and any fixture without a permissive licence record.
//!
//! `import` writes each object with the access this module decides ([`super::store::Access`]);
//! the S3 backend sends it as the object's ACL. The bucket's own policy, which keeps everything
//! else private, is a Maintainer action.

use super::manifest::Fixture;
use super::store::Access;

/// The permissive licences, by their Hub identifiers: ADR-0603 decision 8's list (public
/// domain, CC0, CC-BY, MIT, Apache-2.0, BSD) plus ISC, zlib and BSL-1.0, which grant the same
/// freedoms. Weak copyleft (MPL-2.0) is OSI-approved but not permissive, so it stays private.
pub const PERMISSIVE: &[&str] = &[
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "zlib",
    "bsl-1.0",
    "cc0-1.0",
    "cc-by-4.0",
    "unlicense",
];

/// Whether `f` is a Llama-family file: lab-only whatever licence is recorded (ADR-0603
/// decision 4). Matched on the licence identifier and on the Hub organisation.
pub fn llama_family(f: &Fixture) -> bool {
    let hub = f
        .source
        .get("hub")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    f.licence.to_ascii_lowercase().starts_with("llama") || hub.starts_with("meta-llama/")
}

/// Whether `f` is a project-synthetic fixture.
pub fn synthetic(f: &Fixture) -> bool {
    f.source.get("synthetic").is_some()
}

/// The access `f` and its licence file are published with.
pub fn access(f: &Fixture) -> Access {
    if f.gated || llama_family(f) {
        return Access::Private;
    }
    if synthetic(f) || PERMISSIVE.contains(&f.licence.as_str()) {
        Access::Public
    } else {
        Access::Private
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::manifest::Manifest;

    fn fixture(source: &str, licence: &str, gated: bool) -> Fixture {
        let text = format!(
            "[[fixture]]\nname = \"x\"\nsource = {source}\nlicence = \"{licence}\"\n\
             gated = {gated}\ntier = \"lab\"\n"
        );
        Manifest::parse(&text).unwrap().fixtures.remove(0)
    }

    const HUB: &str = "{ hub = \"Qwen/Qwen3-0.6B\", revision = \"r\", file = \"f\" }";

    #[test]
    fn synthetic_and_permissive_fixtures_are_public() {
        assert_eq!(access(&fixture(HUB, "apache-2.0", false)), Access::Public);
        assert_eq!(access(&fixture(HUB, "mit", false)), Access::Public);
        let synthetic = "{ synthetic = \"testkit nano llama3 seed 0x5eed\" }";
        assert_eq!(
            access(&fixture(synthetic, "mit OR apache-2.0", false)),
            Access::Public
        );
    }

    #[test]
    fn gated_llama_and_unlicensed_fixtures_are_private() {
        // Gated, even under a permissive licence.
        assert_eq!(access(&fixture(HUB, "apache-2.0", true)), Access::Private);
        // A Llama-family file, by licence or by organisation, even if ungated.
        assert_eq!(access(&fixture(HUB, "llama3.2", false)), Access::Private);
        let meta = "{ hub = \"meta-llama/Llama-3.2-1B\", revision = \"r\", file = \"f\" }";
        assert_eq!(access(&fixture(meta, "apache-2.0", false)), Access::Private);
        // No permissive licence record: copyleft, non-commercial, unknown or empty.
        for licence in [
            "mpl-2.0",
            "cc-by-nc-4.0",
            "cc-by-sa-4.0",
            "gemma",
            "other",
            "",
        ] {
            assert_eq!(
                access(&fixture(HUB, licence, false)),
                Access::Private,
                "{licence}"
            );
        }
    }

    /// The checked-in manifest: every Llama-family and gated entry is private.
    #[test]
    fn the_manifests_llama_and_gated_entries_are_private() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let text = std::fs::read_to_string(root.join(super::super::lint::MANIFEST)).unwrap();
        let m = Manifest::parse(&text).unwrap();
        let mut private = 0;
        for f in &m.fixtures {
            if f.gated || llama_family(f) {
                assert_eq!(access(f), Access::Private, "{}", f.name);
                private += 1;
            }
        }
        assert!(
            private > 0,
            "the manifest has no gated or Llama-family entry to check"
        );
    }
}
