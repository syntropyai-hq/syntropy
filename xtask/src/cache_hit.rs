// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask test cache-hit`: a commit that changes only a comment in `cli` recompiles `cli`
//! alone (QUAL-E16-T03; RFC-0062 §8.1; ADR-0621 decision 1).
//!
//! The commit and the rest of the `BuildId` reach the binary only through `cli`'s build script,
//! so no other crate's fingerprint can see them. The suite checks that with cargo's own record:
//! 1. it builds `syntropy-cli`, so every crate is compiled;
//! 2. it stands in for a comment-only commit: it moves `cli`'s `src/lib.rs` modification time
//!    forward (cargo fingerprints workspace sources by modification time, so an edited comment
//!    and a touched file look the same to it) and builds again with `SYNTROPY_BUILD_COMMIT` set
//!    to another commit, which the build script reads in place of `git rev-parse HEAD`;
//! 3. it reads that build's `--message-format=json` lines: every `compiler-artifact` that is not
//!    `fresh` must be `syntropy-cli`'s, and at least one must be, so a build that rebuilt nothing
//!    cannot pass;
//! 4. it runs the binary it built, which must report the other commit in `--version --json`;
//! 5. it builds once more without the variable, so the binary left behind names the checkout's
//!    commit again.
//!
//! A real commit moves `HEAD` instead of setting the variable. The build script watches `HEAD`,
//! the branch's ref and `packed-refs` (`cargo:rerun-if-changed`), which reruns `cli`'s script
//! only: a build script's rerun rules cannot reach another package. This suite runs locally; no
//! hosted job runs it yet.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::SystemTime;

use crate::registry::cargo;
use crate::run::Context;

/// The package whose rebuild is allowed.
const PACKAGE: &str = "syntropy-cli";

/// The commit the second build claims: not a real one, and never this checkout's.
const OTHER_COMMIT: &str = "c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00";

/// The suite's one step.
pub fn suite_step(context: &Context) -> Result<(), String> {
    let root = &context.root;
    build(root, None).map_err(|e| format!("the first build failed: {e}"))?;
    let source = root.join("crates/cli/src/lib.rs");
    touch(&source)?;
    let lines =
        build(root, Some(OTHER_COMMIT)).map_err(|e| format!("the second build failed: {e}"))?;
    let verdict = check(&lines);
    let reported = verdict.as_ref().ok().map(|()| reported_commit(&lines));
    // Leave a binary of the checkout's own commit behind, whatever the verdict.
    let restored = build(root, None)
        .map(|_| ())
        .map_err(|e| format!("the last build failed: {e}"));
    verdict?;
    match reported {
        Some(Ok(commit)) if commit == OTHER_COMMIT => {}
        Some(Ok(commit)) => {
            return Err(format!(
                "the rebuilt binary reports commit {commit}, not {OTHER_COMMIT}"
            ));
        }
        Some(Err(e)) => return Err(e),
        None => {}
    }
    restored?;
    println!("cache-hit: only {PACKAGE} was recompiled after a comment-only change in it");
    Ok(())
}

/// Builds `syntropy-cli` with `--message-format=json`, with `SYNTROPY_BUILD_COMMIT` set to
/// `commit` or removed, and returns stdout's lines.
fn build(root: &Path, commit: Option<&str>) -> Result<Vec<String>, String> {
    let mut command = Command::new(cargo());
    command
        .current_dir(root)
        .args(["build", "--locked", "--package", PACKAGE])
        .arg("--message-format=json-render-diagnostics")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    match commit {
        Some(commit) => command.env("SYNTROPY_BUILD_COMMIT", commit),
        None => command.env_remove("SYNTROPY_BUILD_COMMIT"),
    };
    let output = command.output().map_err(|e| format!("cargo: {e}"))?;
    if !output.status.success() {
        return Err(format!("cargo build exited with {}", output.status));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_owned)
        .collect())
}

/// Moves `path`'s modification time to now, past the first build's outputs, so cargo sees it as
/// edited.
fn touch(path: &Path) -> Result<(), String> {
    let file = std::fs::File::options()
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.set_modified(SystemTime::now())
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// One `compiler-artifact` line: its package's name and whether cargo reused it.
#[derive(Debug, PartialEq, Eq)]
pub struct Artifact {
    pub package: String,
    pub target: String,
    pub fresh: bool,
}

/// The `compiler-artifact` messages among cargo's JSON `lines`.
pub fn artifacts(lines: &[String]) -> Result<Vec<Artifact>, String> {
    let mut out = Vec::new();
    for line in lines {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if message["reason"] != "compiler-artifact" {
            continue;
        }
        let package = package_name(message["package_id"].as_str().unwrap_or_default())
            .ok_or_else(|| format!("a compiler-artifact without a readable package_id: {line}"))?;
        out.push(Artifact {
            package,
            target: message["target"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
            fresh: message["fresh"].as_bool().unwrap_or(false),
        });
    }
    Ok(out)
}

/// The package name in a cargo package ID: `path+file:///…/crates/cli#syntropy-cli@0.0.0`,
/// `registry+…#serde@1.0.228`, or the older `name version (source)` form.
pub fn package_name(id: &str) -> Option<String> {
    if let Some((_, tail)) = id.rsplit_once('#') {
        // `#name@version`, or `#version` when the name is the path's last component.
        return match tail.split_once('@') {
            Some((name, _)) => Some(name.to_owned()),
            None => id
                .split_once('#')
                .and_then(|(url, _)| url.rsplit('/').next())
                .map(str::to_owned),
        };
    }
    id.split_whitespace().next().map(str::to_owned)
}

/// The verdict on the second build's `lines`.
pub fn check(lines: &[String]) -> Result<(), String> {
    let artifacts = artifacts(lines)?;
    if artifacts.is_empty() {
        return Err("cargo reported no compiler artifacts".to_owned());
    }
    let rebuilt: Vec<&Artifact> = artifacts.iter().filter(|a| !a.fresh).collect();
    let others: Vec<String> = rebuilt
        .iter()
        .filter(|a| a.package != PACKAGE)
        .map(|a| format!("{} ({})", a.package, a.target))
        .collect();
    if !others.is_empty() {
        return Err(format!(
            "a comment-only change in {PACKAGE} recompiled other crates: {}",
            others.join(", ")
        ));
    }
    if rebuilt.is_empty() {
        return Err(format!(
            "{PACKAGE} was not recompiled, so the check saw no change (were its sources touched?)"
        ));
    }
    Ok(())
}

/// The commit the `syntropy` binary among `lines`' artifacts reports in `--version --json`.
fn reported_commit(lines: &[String]) -> Result<String, String> {
    let binary = lines
        .iter()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|m| m["reason"] == "compiler-artifact" && m["target"]["name"] == "syntropy")
        .find_map(|m| m["executable"].as_str().map(PathBuf::from))
        .ok_or("cargo named no `syntropy` executable")?;
    let output = Command::new(&binary)
        .args(["--version", "--json"])
        .output()
        .map_err(|e| format!("{}: {e}", binary.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} --version --json exited with {}",
            binary.display(),
            output.status
        ));
    }
    let body: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("{} --version --json: {e}", binary.display()))?;
    body["commit"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| "--version --json printed no commit".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(id: &str, target: &str, fresh: bool) -> String {
        serde_json::json!({
            "reason": "compiler-artifact",
            "package_id": id,
            "target": { "name": target },
            "fresh": fresh,
        })
        .to_string()
    }

    const CLI: &str = "path+file:///w/crates/cli#syntropy-cli@0.0.0";
    const CORE: &str = "path+file:///w/crates/core#syntropy-core@0.0.0";

    #[test]
    fn package_ids_of_each_form_name_their_package() {
        assert_eq!(package_name(CLI).as_deref(), Some("syntropy-cli"));
        assert_eq!(
            package_name("registry+https://github.com/rust-lang/crates.io-index#serde@1.0.228")
                .as_deref(),
            Some("serde")
        );
        assert_eq!(
            package_name("path+file:///w/crates/syntropy-x#0.1.0").as_deref(),
            Some("syntropy-x")
        );
        assert_eq!(
            package_name("serde 1.0.228 (registry+https://github.com/rust-lang/crates.io-index)")
                .as_deref(),
            Some("serde")
        );
    }

    #[test]
    fn only_cli_rebuilt_passes() {
        let lines = vec![
            artifact(CORE, "syntropy_core", true),
            artifact(CLI, "build-script-build", false),
            artifact(CLI, "syntropy_cli", false),
            artifact(CLI, "syntropy", false),
            r#"{"reason":"build-finished","success":true}"#.to_owned(),
            "not json".to_owned(),
        ];
        assert_eq!(check(&lines), Ok(()));
    }

    #[test]
    fn another_crate_rebuilt_fails_and_names_it() {
        let lines = vec![
            artifact(CORE, "syntropy_core", false),
            artifact(CLI, "syntropy_cli", false),
        ];
        let error = check(&lines).unwrap_err();
        assert!(error.contains("syntropy-core (syntropy_core)"), "{error}");
    }

    #[test]
    fn nothing_rebuilt_fails() {
        let lines = vec![
            artifact(CORE, "syntropy_core", true),
            artifact(CLI, "syntropy_cli", true),
        ];
        assert!(check(&lines).unwrap_err().contains("not recompiled"));
        assert!(check(&[]).is_err());
    }
}
