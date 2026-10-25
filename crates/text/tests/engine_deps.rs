// SPDX-License-Identifier: MIT OR Apache-2.0

//! Criterion 2 of COMP-E17-T02: the tokenizer engine builds for every desktop target with no C
//! regex dependency (RFC-0043 §8.2, ADR-0431 decision 1).
//!
//! For each desktop triple of RFC-0062 §4.1 it reads `cargo metadata` for that platform and walks
//! the normal and build dependencies of `syntropy-text`. The Oniguruma bindings must be absent,
//! `tokenizers` must resolve with its pure-Rust regex backend and none of its C or C++ features,
//! and nothing it reaches may be a native build tool (`cc`, `cmake`, `pkg-config`, `bindgen`).
//! Features are read from the resolved graph, so a feature another workspace crate turns on is
//! caught too.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::Value;

/// The desktop artefacts of RFC-0062 §4.1 (ADR-0020).
const DESKTOP_TARGETS: [&str; 4] = [
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
    "aarch64-apple-darwin",
];

/// The Oniguruma bindings, the C regex library `tokenizers` links under its `onig` feature.
const C_REGEX_CRATES: [&str; 2] = ["onig", "onig_sys"];

/// Features of `tokenizers` that build C or C++ code, or are not wanted in the engine.
const BANNED_ENGINE_FEATURES: [&str; 4] = ["onig", "esaxx_fast", "progressbar", "http"];

/// Crates that build or bind native code. A `links` key is no test: `rayon-core` sets one only to
/// keep a single copy in the build.
const NATIVE_BUILD_CRATES: [&str; 4] = ["cc", "cmake", "pkg-config", "bindgen"];

/// The resolved graph for one platform, by package ID.
struct Resolved {
    names: HashMap<String, String>,
    features: HashMap<String, BTreeSet<String>>,
    deps: HashMap<String, Vec<String>>,
}

fn resolve(target: &str) -> Resolved {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--filter-platform",
            target,
        ])
        .current_dir(&workspace)
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .expect("cargo metadata starts");
    assert!(
        output.status.success(),
        "cargo metadata failed for {target}"
    );
    let metadata: Value = serde_json::from_slice(&output.stdout).expect("metadata is JSON");

    let packages = metadata["packages"].as_array().expect("packages");
    let names = packages
        .iter()
        .map(|package| {
            let id = package["id"].as_str().expect("package id").to_owned();
            let name = package["name"].as_str().expect("package name").to_owned();
            (id, name)
        })
        .collect();
    let mut features = HashMap::new();
    let mut deps = HashMap::new();
    for node in metadata["resolve"]["nodes"]
        .as_array()
        .expect("resolve nodes")
    {
        let id = node["id"].as_str().expect("node id").to_owned();
        let enabled = node["features"]
            .as_array()
            .expect("node features")
            .iter()
            .map(|feature| feature.as_str().expect("feature").to_owned())
            .collect();
        // Normal (`kind` null) and build edges; dev edges never reach an artefact.
        let linked = node["deps"]
            .as_array()
            .expect("node deps")
            .iter()
            .filter(|dep| {
                dep["dep_kinds"]
                    .as_array()
                    .expect("dep kinds")
                    .iter()
                    .any(|kind| kind["kind"].is_null() || kind["kind"] == "build")
            })
            .map(|dep| dep["pkg"].as_str().expect("dep pkg").to_owned())
            .collect();
        features.insert(id.clone(), enabled);
        deps.insert(id, linked);
    }
    Resolved {
        names,
        features,
        deps,
    }
}

impl Resolved {
    /// The package IDs `root` reaches, itself included, keyed by name.
    fn reachable(&self, root: &str) -> HashMap<String, String> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![root.to_owned()];
        while let Some(id) = stack.pop() {
            if seen.insert(id.clone()) {
                stack.extend(self.deps[&id].iter().cloned());
            }
        }
        seen.into_iter()
            .map(|id| (self.names[&id].clone(), id))
            .collect()
    }
}

#[test]
fn no_c_regex_dependency_on_any_desktop_target() {
    for target in DESKTOP_TARGETS {
        let resolved = resolve(target);
        let text = resolved
            .names
            .iter()
            .find(|(_, name)| *name == "syntropy-text")
            .map(|(id, _)| id.as_str())
            .expect("syntropy-text is in the workspace");
        let reachable = resolved.reachable(text);

        for name in C_REGEX_CRATES {
            assert!(
                !reachable.contains_key(name),
                "{target}: syntropy-text reaches the C regex crate {name}"
            );
        }

        let engine_id = reachable
            .get("tokenizers")
            .unwrap_or_else(|| panic!("{target}: syntropy-text does not reach tokenizers"));
        let engine = &resolved.features[engine_id];
        assert!(
            engine.contains("fancy-regex"),
            "{target}: tokenizers lacks its pure-Rust regex feature: {engine:?}"
        );
        for feature in BANNED_ENGINE_FEATURES {
            assert!(
                !engine.contains(feature),
                "{target}: tokenizers resolves with `{feature}`: {engine:?}"
            );
        }

        // The suffix-array crate the Unigram trainer uses; its `cpp` feature builds C++.
        if let Some(id) = reachable.get("esaxx-rs") {
            assert!(
                !resolved.features[id].contains("cpp"),
                "{target}: esaxx-rs builds its C++ code"
            );
        }

        // `blake3` lists `cc` as a build dependency even in its `pure` build, so the native-code
        // check covers the engine's own graph, not the whole crate's.
        for name in resolved.reachable(engine_id).keys() {
            assert!(
                !NATIVE_BUILD_CRATES.contains(&name.as_str()),
                "{target}: tokenizers reaches the native build crate {name}"
            );
        }
    }
}
