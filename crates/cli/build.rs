// SPDX-License-Identifier: MIT OR Apache-2.0

//! Builds the binary's `BuildId` (RFC-0062 §8.1; ADR-0621 decision 1; QUAL-E16-T03).
//!
//! It writes `$OUT_DIR/build_id.rs`, which `src/build_id.rs` includes:
//! - **version**: `version.txt` at the repository root, or `CARGO_PKG_VERSION` where the file
//!   is absent, as in a crate archive. A `version.txt` that `syntropy_core::build::SemVer` would
//!   refuse fails the build, through that parser itself ([`version::checked`]);
//! - **commit**: `SYNTROPY_BUILD_COMMIT` (40 hex digits) when set, as a release or a source
//!   archive's build sets it; else `git rev-parse HEAD`, when the repository's top level is this
//!   workspace's root; else all zeros, which the binary reports as unknown. There is no git
//!   dependency, only the `git` command when it exists;
//! - **profile**: the cargo profile, from the directory cargo builds it in (`OUT_DIR`), since
//!   cargo tells a build script only `debug` or `release`;
//! - **target**: `TARGET`;
//! - **toolchain**: `$RUSTC -vV`, which `src/build_id.rs` hashes with BLAKE3;
//! - **features**: `cli`'s own enabled features, `CARGO_FEATURE_*`.
//!
//! This script is `cli`'s, so a new commit or version reruns it and recompiles `cli` alone:
//! every other crate's compilation stays cached (`cargo xtask test cache-hit` checks it).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "build/version.rs"]
mod version;

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets it"));

    let version = version(&manifest);
    let commit = commit(&manifest);
    let profile = profile(&out);
    let target = env::var("TARGET").expect("cargo sets it");
    let toolchain = toolchain();
    let mut features: Vec<String> = env::vars()
        .filter_map(|(name, _)| {
            name.strip_prefix("CARGO_FEATURE_")
                .map(|f| f.to_ascii_lowercase().replace('_', "-"))
        })
        .collect();
    features.sort();

    let bytes = |b: &[u8]| {
        let list: Vec<String> = b.iter().map(|x| format!("0x{x:02x}")).collect();
        format!("[{}]", list.join(", "))
    };
    let code = format!(
        "/// The product version.\n\
         pub const VERSION: &str = {version:?};\n\
         /// The commit, or zeros when unknown.\n\
         pub const COMMIT: [u8; 20] = {commit};\n\
         /// The cargo profile's name.\n\
         pub const PROFILE: &str = {profile:?};\n\
         /// The target triple.\n\
         pub const TARGET: &str = {target:?};\n\
         /// What `rustc -vV` printed.\n\
         pub const TOOLCHAIN: &str = {toolchain:?};\n\
         /// `cli`'s enabled features.\n\
         pub const FEATURES: &[&str] = &{features:?};\n",
        commit = bytes(&commit),
    );
    fs::write(out.join("build_id.rs"), code).expect("write build_id.rs");
}

/// `version.txt`, or `CARGO_PKG_VERSION` when there is none.
fn version(manifest: &Path) -> String {
    let file = manifest.join("..").join("..").join("version.txt");
    println!("cargo:rerun-if-changed={}", file.display());
    match fs::read_to_string(&file) {
        // `core`'s own parser, so a version that builds is one `SemVer` reads at run time.
        Ok(text) => {
            version::checked(&text).unwrap_or_else(|why| panic!("{}: {why}", file.display()))
        }
        Err(_) => env::var("CARGO_PKG_VERSION").expect("cargo sets it"),
    }
}

/// The commit: `SYNTROPY_BUILD_COMMIT`, then git, then zeros.
fn commit(manifest: &Path) -> [u8; 20] {
    println!("cargo:rerun-if-env-changed=SYNTROPY_BUILD_COMMIT");
    if let Ok(text) = env::var("SYNTROPY_BUILD_COMMIT") {
        return hex20(text.trim()).unwrap_or_else(|| {
            panic!("SYNTROPY_BUILD_COMMIT=`{text}` is not 40 lower-case hex digits")
        });
    }
    let git = |args: &[&str]| -> Option<String> {
        let output = Command::new("git")
            .arg("-C")
            .arg(manifest)
            .args(args)
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    // A source tree unpacked inside another repository, such as a packager's, has no `.git` of
    // its own, and git would answer with the outer repository's HEAD. Trust HEAD only when the
    // repository's top level is this workspace's root.
    let root = manifest.join("..").join("..");
    let ours = |top: String| match (fs::canonicalize(top), fs::canonicalize(&root)) {
        (Ok(top), Ok(root)) => top == root,
        _ => false,
    };
    if !git(&["rev-parse", "--show-toplevel"]).is_some_and(ours) {
        return [0; 20];
    }
    let Some(head) = git(&["rev-parse", "HEAD"]) else {
        return [0; 20];
    };
    // Rerun when HEAD moves: the HEAD file, the branch it names, and the packed refs. A
    // worktree keeps HEAD in its own git directory and the refs in the common one, which
    // `--git-path` resolves.
    let mut watched = vec!["HEAD".to_owned(), "packed-refs".to_owned()];
    if let Some(branch) = git(&["symbolic-ref", "-q", "HEAD"]) {
        watched.push(branch);
    }
    for path in watched {
        if let Some(file) = git(&["rev-parse", "--path-format=absolute", "--git-path", &path]) {
            println!("cargo:rerun-if-changed={file}");
        }
    }
    hex20(&head).unwrap_or([0; 20])
}

/// 20 bytes from 40 lower-case hex digits.
fn hex20(text: &str) -> Option<[u8; 20]> {
    if text.len() != 40 {
        return None;
    }
    let mut out = [0u8; 20];
    for (i, byte) in out.iter_mut().enumerate() {
        let pair = text.get(i * 2..i * 2 + 2)?;
        if pair
            .bytes()
            .any(|b| !matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        {
            return None;
        }
        *byte = u8::from_str_radix(pair, 16).ok()?;
    }
    Some(out)
}

/// The profile's name: the directory under the target directory that holds `build/`, where
/// cargo's `dev` profile builds into `debug`.
fn profile(out: &Path) -> String {
    out.ancestors()
        .skip_while(|p| p.file_name().is_none_or(|n| n != "build"))
        .nth(1)
        .and_then(|p| p.file_name())
        .map_or_else(
            || env::var("PROFILE").unwrap_or_else(|_| "unknown".to_owned()),
            |n| n.to_string_lossy().into_owned(),
        )
}

/// What `$RUSTC -vV` prints: the release, its commit, the host and the LLVM version.
fn toolchain() -> String {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(&rustc)
        .arg("-vV")
        .output()
        .expect("run rustc -vV");
    assert!(output.status.success(), "rustc -vV failed");
    String::from_utf8(output.stdout).expect("rustc -vV prints UTF-8")
}
