// SPDX-License-Identifier: MIT OR Apache-2.0

//! The solver is pure (RFC-0001 §4.2 rule 5; PLC-E4-T01): its normal dependencies reach no
//! async runtime, socket, filesystem or thread-pool crate and none of the workspace's I/O
//! crates, and its own sources call no filesystem, network, process, thread or clock API.
//!
//! `cargo xtask check-deps` checks the same crate lists over the whole workspace in `lint`
//! (`xtask/src/check_deps/policy.rs`). This test keeps the check with the crate, so a build of
//! the solver's tests fails on its own.
//!
//! `std` is not a package, so `cargo metadata` cannot see a dependency's own `std::fs` calls.
//! The source scan covers the solver's own sources only.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use serde_json::Value;

/// Workspace crates that do I/O: rule 5's list in `xtask/src/check_deps/policy.rs`, and the
/// crates above the solver that hold runtimes and sockets.
const IO_MEMBERS: &[&str] = &[
    "syntropy-io",
    "syntropy-net",
    "syntropy-ipc",
    "syntropy-store",
    "syntropy-cluster",
    "syntropy-api",
    "syntropy-daemon",
];

/// External crates the solver must not reach: async runtimes and sockets, filesystem crates and
/// thread pools (RFC-0062 §12.1).
const IO_EXTERNAL: &[&str] = &[
    "tokio",
    "async-std",
    "smol",
    "mio",
    "socket2",
    "walkdir",
    "tempfile",
    "notify",
    "fs-err",
    "rayon",
    "rayon-core",
    "threadpool",
];

/// `std` paths and names the solver's sources must not use.
const IO_APIS: &[&str] = &[
    "std::fs",
    "std::net",
    "std::process",
    "std::thread",
    "std::env",
    "std::io::stdin",
    "std::time::Instant",
    "std::time::SystemTime",
    "Instant::now",
    "SystemTime::now",
    "File::open",
    "File::create",
    "tokio",
];

/// The packages the solver reaches through normal dependencies, by name, each with the chain
/// that reaches it.
fn reached() -> BTreeMap<String, Vec<String>> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let output = Command::new(cargo)
        .args([
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--manifest-path",
        ])
        .arg(&manifest)
        .output()
        .expect("cargo metadata runs");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: Value = serde_json::from_slice(&output.stdout).expect("cargo metadata's JSON");

    let names: BTreeMap<&str, &str> = metadata["packages"]
        .as_array()
        .expect("packages")
        .iter()
        .map(|p| (p["id"].as_str().unwrap(), p["name"].as_str().unwrap()))
        .collect();
    let nodes: BTreeMap<&str, &Value> = metadata["resolve"]["nodes"]
        .as_array()
        .expect("resolve nodes")
        .iter()
        .map(|n| (n["id"].as_str().unwrap(), n))
        .collect();
    let root = names
        .iter()
        .find(|(_, name)| **name == "syntropy-solver")
        .map(|(id, _)| *id)
        .expect("the solver's package");

    let mut reached = BTreeMap::new();
    let mut seen = BTreeSet::from([root]);
    let mut queue = vec![(root, vec!["syntropy-solver".to_owned()])];
    while let Some((id, chain)) = queue.pop() {
        for dep in nodes[id]["deps"].as_array().expect("deps") {
            let normal = dep["dep_kinds"]
                .as_array()
                .expect("dep_kinds")
                .iter()
                .any(|kind| kind["kind"].is_null());
            let dep_id = dep["pkg"].as_str().unwrap();
            if !normal || !seen.insert(dep_id) {
                continue;
            }
            let name = names[dep_id].to_owned();
            let mut next = chain.clone();
            next.push(name.clone());
            reached.insert(name, next.clone());
            queue.push((dep_id, next));
        }
    }
    reached
}

#[test]
fn the_solver_reaches_no_io_crate() {
    let reached = reached();
    // The walk itself works: it reaches the solver's own dependencies, direct and indirect.
    for expected in ["syntropy-core", "syntropy-manifest", "serde", "smallvec"] {
        assert!(reached.contains_key(expected), "the walk missed {expected}");
    }
    // Dev-dependencies are not followed.
    assert!(!reached.contains_key("proptest"));
    let found: Vec<String> = IO_MEMBERS
        .iter()
        .chain(IO_EXTERNAL)
        .filter_map(|name| reached.get(*name))
        .map(|chain| chain.join(" → "))
        .collect();
    assert!(
        found.is_empty(),
        "the solver is pure (RFC-0001 §4.2 rule 5), but reaches: {found:#?}"
    );
}

fn sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("the source directory") {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn the_solver_sources_call_no_io_api() {
    let mut files = Vec::new();
    sources(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    assert!(files.len() >= 2, "the scan found the solver's sources");
    let mut found = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(&file).expect("a source file");
        for (number, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or_default();
            for api in IO_APIS.iter().filter(|api| code.contains(*api)) {
                found.push(format!("{}:{}: {api}", file.display(), number + 1));
            }
        }
    }
    assert!(
        found.is_empty(),
        "the solver is pure (RFC-0001 §4.2 rule 5), but its sources use: {found:#?}"
    );
}
