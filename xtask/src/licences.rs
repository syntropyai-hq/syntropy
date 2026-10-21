// SPDX-License-Identifier: MIT OR Apache-2.0

//! The licence policy of ADR-0018 and RFC-0062 §11.2–§12.2 (QUAL-E1-T04): four commands, each
//! a step of the `lint` job.
//!
//! - `cargo xtask deny` runs cargo-deny with `deny.toml` ([`deny`]): the allowed licences, the
//!   bans and the sources, and with `--advisories` the RustSec advisories, the only check that
//!   needs the network.
//! - `cargo xtask spdx` checks the `SPDX-License-Identifier` header of every Syntropy-authored
//!   source file ([`headers`]).
//! - `cargo xtask third-party` checks that `third_party/licences.toml` lists every vendored file
//!   individually, with a licence on the allow list ([`manifest`]).
//! - `cargo xtask notices` generates `THIRD-PARTY-NOTICES` from the release crates' resolved
//!   graph and the manifest ([`notices`]); in `lint` it checks that generation succeeds.
//!
//! The policy each check applies is the one in the checkout xtask runs from: its `deny.toml`. A
//! `--manifest-path` or `--root` option points the check at another tree, such as a test
//! fixture, and the policy stays the same.

pub mod deny;
pub mod expr;
pub mod headers;
pub mod manifest;
pub mod notices;
pub mod toml;

use std::path::Path;

/// The cargo-deny configuration, relative to the workspace root.
pub const DENY_TOML: &str = "deny.toml";

/// Where vendored third-party code lives: `third_party/` at the root, and `vendor/` inside any
/// `kernels-*` crate (RFC-0062 §12.2).
pub const THIRD_PARTY: &str = "third_party";

/// The licence policy: the allow list of ADR-0018 §3, read from `deny.toml`.
#[derive(Debug)]
pub struct Policy {
    /// Each allowed licence, as `deny.toml` writes it, such as `Apache-2.0 WITH LLVM-exception`.
    pub allow: Vec<String>,
}

impl Policy {
    /// Reads `[licenses] allow` from the `deny.toml` under `checkout`. A missing or empty list is
    /// an error, so the checks that use it fail closed.
    pub fn load(checkout: &Path) -> Result<Policy, String> {
        let path = checkout.join(DENY_TOML);
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        match toml::string_array(&text, "licenses", "allow") {
            Ok(Some(allow)) if !allow.is_empty() => Ok(Policy { allow }),
            Ok(_) => Err(format!(
                "{DENY_TOML} has no `allow` list in `[licenses]`, so no licence is allowed"
            )),
            Err(error) => Err(format!("{DENY_TOML}:{error}")),
        }
    }
}

/// `1 file` or `2 files`: `n` and `noun`, plural with an `s` unless `n` is 1.
pub fn count(n: usize, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

/// `path` relative to `root`, with `/` between components, as messages and the manifest write
/// paths on every OS.
pub fn relative(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    parts.join("/")
}

/// Whether the root-relative path `rel` is vendored third-party code, or a directory that holds
/// it: `third_party` and everything under it, and `crates/kernels-*/vendor` and everything under
/// it.
pub fn is_vendored(rel: &str) -> bool {
    let mut parts = rel.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(THIRD_PARTY), _, _) => true,
        (Some("crates"), Some(krate), Some("vendor")) => krate.starts_with("kernels-"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendored_paths_are_third_party_and_provider_vendor_trees() {
        for rel in [
            "third_party",
            "third_party/licences.toml",
            "third_party/ggml/src/ggml.c",
            "crates/kernels-cuda/vendor",
            "crates/kernels-cuda/vendor/cutlass/include/cute/tensor.hpp",
            "crates/kernels-loader/vendor/x.h",
        ] {
            assert!(is_vendored(rel), "{rel}");
        }
        for rel in [
            "",
            "crates",
            "crates/kernels-cuda",
            "crates/kernels-cuda/src/vendor/x.rs",
            "crates/core/vendor/x.c",
            "crates/kernels/vendor/x.c",
            "xtask/tests/fixtures/third_party/x.c",
            "third_party_extra/x.c",
        ] {
            assert!(!is_vendored(rel), "{rel}");
        }
    }

    #[test]
    fn counts_are_plural_except_for_one() {
        assert_eq!(count(0, "file"), "0 files");
        assert_eq!(count(1, "file"), "1 file");
        assert_eq!(count(2, "NOTICE file"), "2 NOTICE files");
    }

    #[test]
    fn relative_paths_use_forward_slashes() {
        let root = Path::new("/repo");
        assert_eq!(
            relative(root, &root.join("third_party").join("a").join("b.c")),
            "third_party/a/b.c"
        );
        assert_eq!(relative(root, root), "");
    }

    #[test]
    fn the_checkout_policy_is_adr_0018_s_allow_list() {
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let policy = Policy::load(checkout).unwrap();
        assert_eq!(
            policy.allow,
            [
                "MIT",
                "Apache-2.0",
                "Apache-2.0 WITH LLVM-exception",
                "BSD-2-Clause",
                "BSD-3-Clause",
                "ISC",
                "Zlib",
                "CC0-1.0",
                "BSL-1.0",
                "Unicode-3.0",
                "CDLA-Permissive-2.0",
            ]
        );
    }

    #[test]
    fn a_policy_without_an_allow_list_fails_closed() {
        let dir =
            std::env::temp_dir().join(format!("syntropy-xtask-policy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for text in ["[licenses]\nallow = []\n", "[bans]\nallow = [\"MIT\"]\n"] {
            std::fs::write(dir.join(DENY_TOML), text).unwrap();
            let error = Policy::load(&dir).unwrap_err();
            assert!(error.contains("no `allow` list"), "{error}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(Policy::load(&dir).unwrap_err().starts_with("cannot read"));
    }
}
