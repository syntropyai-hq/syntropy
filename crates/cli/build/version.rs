// SPDX-License-Identifier: MIT OR Apache-2.0

//! The product version's check, shared by `cli`'s build script and its tests (RFC-0062 §8.1;
//! QUAL-E16-T03).
//!
//! The build script refuses a `version.txt` that `syntropy_core::build::SemVer` would refuse at
//! run time, by calling that parser: one rule, so a version that builds always parses, and
//! `syntropy --version` never fails on it.

/// `text`, trimmed, when it is a version `SemVer` reads; otherwise why not.
pub fn checked(text: &str) -> Result<String, String> {
    let version = text.trim();
    version
        .parse::<syntropy_core::build::SemVer>()
        .map(|_| version.to_owned())
        .map_err(|e| e.to_string())
}
