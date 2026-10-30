// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask build --profile <p>`: builds the workspace in one of the build profiles of
//! RFC-0062 §4.4 and RFC-0060 §1.3, with that profile's test-only features, then checks the
//! build cargo reports against the profile (QUAL-E3-T04).
//!
//! | Profile | Cargo profile | Test features | Debug assertions | Builds |
//! |---|---|---|---|---|
//! | `dev` | `dev` | `fault-injection`, `debug-taps`, `guards` | on | the workspace |
//! | `test-release` | `test-release`: `release` with debug assertions | all three | on | the workspace |
//! | `chaos` | `chaos`: `release` | `fault-injection` | off | the production crates |
//! | `release` | `release` | none | off | each release crate on its own |
//!
//! The cargo profiles are in the workspace `Cargo.toml`. The `pr` profile is QUAL-E16-T21's.
//!
//! - **Features.** Any crate may declare a test-only feature (RFC-0001 §4.2 rule 10), where code
//!   sits behind it. The build passes each of the profile's features that a selected member
//!   declares, so cargo turns it on in every selected member that has it. A feature no selected
//!   member declares yet is named in the log and not passed.
//! - **`chaos`** builds every crate of the crate graph, but not the dev-only crates or the tools
//!   of `ci/layers.toml`: dev-only crates such as `testkit` may turn other test features on in the
//!   crates they depend on, and cargo would unify those into the chaos build.
//! - **`release`** builds each release crate on its own, with its default features, as its
//!   artefact is built: the build `check-deps` rule 10 resolves.
//! - **The check.** After the build, every unit of a workspace member that cargo reports must
//!   have debug assertions on or off as the profile says, and no test-only feature outside the
//!   profile's. Build scripts and proc macros are not checked, because they run on the host and
//!   cargo may build them with build-override settings. So a passing run shows what the profile
//!   built, not only what `Cargo.toml` declares.
//! - **The compiler** (`release` only; QUAL-E3-T10). Cargo records the `rustc -vV` of the compiler
//!   that built a target directory in its `.rustc_info.json`. After a release build, that
//!   compiler must be the stable release `rust-toolchain.toml` pins (ADR-0001 item 1), and not a
//!   nightly, such as the test toolchain of `ci/nightly-toolchain.toml`, nor a beta or a
//!   development build.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Deserialize;

use crate::check_deps::layers::{Layers, Place};
use crate::check_deps::policy::{self, PREFIX};

/// Which crates a profile builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Every workspace member: `--workspace`.
    Workspace,
    /// Every member but the dev-only crates and the tools of `ci/layers.toml`.
    Production,
    /// Each release crate, in a build of its own, with no test feature.
    ReleaseCrates,
}

/// A build profile `cargo xtask build` knows.
#[derive(Debug, PartialEq, Eq)]
pub struct Profile {
    /// The name on the command line.
    pub name: &'static str,
    /// The cargo profile it builds with (`cargo build --profile`).
    pub cargo_profile: &'static str,
    /// The test-only features it turns on.
    pub features: &'static [&'static str],
    /// Whether its units have debug assertions on.
    pub debug_assertions: bool,
    /// The crates it builds.
    pub scope: Scope,
}

/// The profiles of RFC-0062 §4.4 that QUAL-E3-T04 defines, in that table's order.
pub const PROFILES: &[Profile] = &[
    Profile {
        name: "dev",
        cargo_profile: "dev",
        features: policy::TEST_FEATURES,
        debug_assertions: true,
        scope: Scope::Workspace,
    },
    Profile {
        name: "test-release",
        cargo_profile: "test-release",
        features: policy::TEST_FEATURES,
        debug_assertions: true,
        scope: Scope::Workspace,
    },
    Profile {
        name: "chaos",
        cargo_profile: "chaos",
        features: &["fault-injection"],
        debug_assertions: false,
        scope: Scope::Production,
    },
    Profile {
        name: "release",
        cargo_profile: "release",
        features: &[],
        debug_assertions: false,
        scope: Scope::ReleaseCrates,
    },
];

/// The profile called `name`, if there is one.
pub fn find(name: &str) -> Option<&'static Profile> {
    PROFILES.iter().find(|profile| profile.name == name)
}

/// Every profile name, in table order.
pub fn names() -> Vec<&'static str> {
    PROFILES.iter().map(|profile| profile.name).collect()
}

#[derive(Deserialize)]
struct Metadata {
    packages: Vec<MetaPackage>,
    workspace_members: Vec<String>,
    workspace_root: PathBuf,
    /// Where cargo builds, which `CARGO_TARGET_DIR` can move.
    #[serde(default)]
    target_directory: Option<PathBuf>,
}

/// The `release:` values of every `rustc -vV` output cargo recorded in a target directory's
/// `.rustc_info.json`: the compiler the builds there used, as cargo itself saw it.
fn recorded_releases(json: &str) -> Result<BTreeSet<String>, String> {
    #[derive(Deserialize)]
    struct Info {
        outputs: BTreeMap<String, Output>,
    }
    #[derive(Deserialize)]
    struct Output {
        stdout: String,
    }
    let info: Info = serde_json::from_str(json)
        .map_err(|error| format!("cannot read .rustc_info.json: {error}"))?;
    let releases: BTreeSet<String> = info
        .outputs
        .values()
        .flat_map(|output| output.stdout.lines())
        .filter_map(|line| line.strip_prefix("release: "))
        .map(|release| release.trim().to_owned())
        .collect();
    if releases.is_empty() {
        return Err(".rustc_info.json records no `rustc -vV` output".to_owned());
    }
    Ok(releases)
}

/// The release check of QUAL-E3-T10 (ADR-0001 item 1): every compiler the release build
/// recorded is `pinned`, the stable release of `rust-toolchain.toml`, and none is a nightly, beta
/// or development build, such as the test nightly of `ci/nightly-toolchain.toml`.
fn check_release_compiler(releases: &BTreeSet<String>, pinned: &str) -> Vec<String> {
    releases
        .iter()
        .filter(|release| {
            release.contains('-') || !crate::toolchain::matches(pinned, release.as_str())
        })
        .map(|release| {
            format!(
                "the release build was compiled by rustc {release}, but rust-toolchain.toml pins \
                 the stable release {pinned}, and only that release may build a release \
                 (ADR-0001 item 1)"
            )
        })
        .collect()
}

/// Reads the compiler the build under `target` recorded and checks it against the pin that
/// applies to the workspace at `root`: the nearest `rust-toolchain.toml` at or above it, the file
/// rustup follows.
fn release_compiler(root: &Path, target: &Path) -> Result<Vec<String>, String> {
    let file = root
        .ancestors()
        .map(|dir| dir.join("rust-toolchain.toml"))
        .find(|file| file.is_file())
        .ok_or("no rust-toolchain.toml at or above the workspace")?;
    let toolchain = std::fs::read_to_string(&file)
        .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    let pinned = crate::toolchain::channel(&toolchain)
        .ok_or("rust-toolchain.toml has no `channel` in its `[toolchain]` table")?;
    let info = target.join(".rustc_info.json");
    let json = std::fs::read_to_string(&info)
        .map_err(|error| format!("cannot read {}: {error}", info.display()))?;
    Ok(check_release_compiler(&recorded_releases(&json)?, &pinned))
}

#[derive(Deserialize)]
struct MetaPackage {
    id: String,
    name: String,
    #[serde(default)]
    features: BTreeMap<String, Vec<String>>,
}

/// One `cargo build`.
#[derive(Debug, PartialEq, Eq)]
struct Build {
    /// `--workspace`, with `--exclude <crate>` for each crate left out, or `--package <crate>`.
    packages: Vec<String>,
    /// The test features passed with `--features`.
    features: Vec<&'static str>,
}

/// What a build of one profile runs, read from the workspace's metadata.
#[derive(Debug, PartialEq, Eq)]
struct Plan {
    builds: Vec<Build>,
    /// The profile's features that no selected member declares.
    undeclared: Vec<&'static str>,
    /// Package ID to name, for every workspace member.
    members: BTreeMap<String, String>,
}

/// The builds `profile` runs on the workspace `metadata` describes. `layers` is needed for
/// [`Scope::Production`] only.
fn plan(profile: &Profile, metadata: &Metadata, layers: Option<&Layers>) -> Result<Plan, String> {
    let member_ids: BTreeSet<&str> = metadata
        .workspace_members
        .iter()
        .map(String::as_str)
        .collect();
    let mut members: Vec<&MetaPackage> = metadata
        .packages
        .iter()
        .filter(|package| member_ids.contains(package.id.as_str()))
        .collect();
    members.sort_by(|a, b| a.name.cmp(&b.name));
    let short = |name: &'_ str| name.strip_prefix(PREFIX).unwrap_or(name).to_owned();
    // The members each build selects, and the arguments that select them.
    let selections: Vec<(Vec<&MetaPackage>, Vec<String>)> = match profile.scope {
        Scope::Workspace => vec![(members.clone(), vec!["--workspace".to_owned()])],
        Scope::Production => {
            let layers = layers.ok_or("the `chaos` scope needs `ci/layers.toml`")?;
            let (left_out, selected): (Vec<&MetaPackage>, Vec<&MetaPackage>) =
                members.iter().copied().partition(|member| {
                    matches!(
                        layers.place(&short(&member.name)),
                        Some(Place::DevOnly | Place::Tool)
                    )
                });
            let mut args = vec!["--workspace".to_owned()];
            for member in left_out {
                args.extend(["--exclude".to_owned(), member.name.clone()]);
            }
            vec![(selected, args)]
        }
        Scope::ReleaseCrates => {
            let crates: Vec<(Vec<&MetaPackage>, Vec<String>)> = policy::RELEASE_CRATES
                .iter()
                .filter_map(|release| {
                    members
                        .iter()
                        .find(|member| short(&member.name) == *release)
                })
                .map(|member| {
                    (
                        vec![*member],
                        vec!["--package".to_owned(), member.name.clone()],
                    )
                })
                .collect();
            if crates.is_empty() {
                return Err("the workspace has no release crate".to_owned());
            }
            crates
        }
    };
    let mut undeclared: BTreeSet<&'static str> = profile.features.iter().copied().collect();
    let builds = selections
        .into_iter()
        .map(|(selected, packages)| {
            let features: Vec<&'static str> = profile
                .features
                .iter()
                .copied()
                .filter(|feature| {
                    selected
                        .iter()
                        .any(|member| member.features.contains_key(*feature))
                })
                .collect();
            for feature in &features {
                undeclared.remove(feature);
            }
            Build { packages, features }
        })
        .collect();
    let undeclared = profile
        .features
        .iter()
        .copied()
        .filter(|feature| undeclared.contains(feature))
        .collect();
    let members = members
        .iter()
        .map(|member| (member.id.clone(), member.name.clone()))
        .collect();
    Ok(Plan {
        builds,
        undeclared,
        members,
    })
}

#[derive(Deserialize)]
struct Message {
    reason: String,
    package_id: Option<String>,
    target: Option<MessageTarget>,
    profile: Option<MessageProfile>,
    features: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct MessageTarget {
    name: String,
    kind: Vec<String>,
}

#[derive(Deserialize)]
struct MessageProfile {
    debug_assertions: bool,
}

/// What the build's own reports show.
#[derive(Debug, Default, PartialEq, Eq)]
struct Checked {
    /// Units of workspace members checked.
    units: usize,
    /// Each test feature found on, as `<member>/<feature>`.
    on: BTreeSet<String>,
    /// Each way the build differs from the profile.
    problems: Vec<String>,
}

impl Checked {
    /// Checks cargo's `compiler-artifact` reports in `stdout` against `profile`.
    fn add(&mut self, profile: &Profile, members: &BTreeMap<String, String>, stdout: &str) {
        for line in stdout.lines() {
            let Ok(message) = serde_json::from_str::<Message>(line) else {
                continue;
            };
            if message.reason != "compiler-artifact" {
                continue;
            }
            let Some(name) = message.package_id.as_ref().and_then(|id| members.get(id)) else {
                continue;
            };
            let (Some(target), Some(unit_profile)) = (message.target, message.profile) else {
                continue;
            };
            if target
                .kind
                .iter()
                .any(|kind| kind == "custom-build" || kind == "proc-macro")
            {
                continue;
            }
            self.units += 1;
            let short = name.strip_prefix(PREFIX).unwrap_or(name);
            let kind = target.kind.join(", ");
            if unit_profile.debug_assertions != profile.debug_assertions {
                self.problems.push(format!(
                    "`{short}` ({kind} `{}`) was built with debug assertions {}; the `{}` \
                     profile has them {}",
                    target.name,
                    on_off(unit_profile.debug_assertions),
                    profile.name,
                    on_off(profile.debug_assertions)
                ));
            }
            for feature in message.features.unwrap_or_default() {
                if !policy::TEST_FEATURES.contains(&feature.as_str()) {
                    continue;
                }
                self.on.insert(format!("{short}/{feature}"));
                if !profile.features.contains(&feature.as_str()) {
                    self.problems.push(format!(
                        "`{short}` ({kind} `{}`) was built with `{feature}`, which the `{}` \
                         profile does not turn on",
                        target.name, profile.name
                    ));
                }
            }
        }
    }

    /// Records that the builds reported no unit to check, if so.
    fn finish(&mut self) {
        if self.units == 0 {
            self.problems.push(
                "cargo reported no unit of a workspace crate, so nothing was checked".to_owned(),
            );
        }
    }
}

/// `a, b and c`, or `none`.
fn listed(items: &[String]) -> String {
    match items.split_last() {
        None => "none".to_owned(),
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
    }
}

fn quoted(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| format!("`{item}`")).collect()
}

fn on_off(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}

/// Reads the workspace of `manifest` with `cargo metadata --no-deps`.
fn metadata(manifest: &Path) -> Result<Metadata, String> {
    let output = Command::new(crate::registry::cargo())
        .args(["metadata", "--format-version", "1", "--no-deps", "--locked"])
        .arg("--manifest-path")
        .arg(manifest)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("could not start `cargo metadata`: {error}"))?;
    if !output.status.success() {
        return Err(format!("`cargo metadata` failed: {}", output.status));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("cannot read `cargo metadata` output: {error}"))
}

/// Builds the workspace of `manifest` in `profile` and checks the result. Progress, findings and
/// the verdict go to `log`. Returns whether every build and the check passed.
pub fn run(manifest: &Path, profile: &Profile, log: &mut dyn Write) -> io::Result<bool> {
    writeln!(
        log,
        "build: profile `{}`: cargo profile `{}`, debug assertions {}, test features {}",
        profile.name,
        profile.cargo_profile,
        on_off(profile.debug_assertions),
        listed(&quoted(profile.features))
    )?;
    log.flush()?;
    let planned = metadata(manifest).and_then(|metadata| {
        let layers = match profile.scope {
            Scope::Production => Some(Layers::load(&metadata.workspace_root)?),
            _ => None,
        };
        let target = metadata
            .target_directory
            .clone()
            .unwrap_or_else(|| metadata.workspace_root.join("target"));
        Ok((
            plan(profile, &metadata, layers.as_ref())?,
            metadata.workspace_root,
            target,
        ))
    });
    let (plan, root, target) = match planned {
        Ok(planned) => planned,
        Err(error) => {
            writeln!(log, "build: FAIL: {error}")?;
            return Ok(false);
        }
    };
    for feature in &plan.undeclared {
        writeln!(
            log,
            "build: no crate this profile builds declares `{feature}`, so it is not passed"
        )?;
    }
    let mut checked = Checked::default();
    for build in &plan.builds {
        let mut args: Vec<String> = ["build", "--locked", "--profile", profile.cargo_profile]
            .iter()
            .map(|arg| (*arg).to_owned())
            .collect();
        args.extend(build.packages.iter().cloned());
        if !build.features.is_empty() {
            args.extend(["--features".to_owned(), build.features.join(",")]);
        }
        writeln!(log, "build: cargo {}", args.join(" "))?;
        log.flush()?;
        let output = Command::new(crate::registry::cargo())
            .args(&args)
            .args(["--message-format", "json-render-diagnostics"])
            .arg("--manifest-path")
            .arg(manifest)
            .stderr(Stdio::inherit())
            .output();
        match output {
            Ok(output) if output.status.success() => {
                checked.add(
                    profile,
                    &plan.members,
                    &String::from_utf8_lossy(&output.stdout),
                );
            }
            Ok(output) => {
                writeln!(log, "build: FAIL: `cargo build` failed: {}", output.status)?;
                return Ok(false);
            }
            Err(error) => {
                writeln!(log, "build: FAIL: could not start `cargo build`: {error}")?;
                return Ok(false);
            }
        }
    }
    checked.finish();
    if profile.scope == Scope::ReleaseCrates {
        match release_compiler(&root, &target) {
            Ok(problems) if problems.is_empty() => writeln!(
                log,
                "build: the compiler cargo recorded in {} is the pinned stable release",
                target.join(".rustc_info.json").display()
            )?,
            Ok(problems) => checked.problems.extend(problems),
            Err(error) => checked.problems.push(error),
        }
    }
    for problem in &checked.problems {
        writeln!(log, "build: {problem}")?;
    }
    let on: Vec<String> = checked.on.iter().map(|f| format!("`{f}`")).collect();
    if checked.problems.is_empty() {
        writeln!(
            log,
            "build: PASS: {} units of workspace crates built in `{}`, debug assertions {} in \
             every one; test features on: {}",
            checked.units,
            profile.name,
            on_off(profile.debug_assertions),
            listed(&on)
        )?;
    } else {
        writeln!(
            log,
            "build: FAIL: the build differs from the `{}` profile in {} {}",
            profile.name,
            checked.problems.len(),
            if checked.problems.len() == 1 {
                "way"
            } else {
                "ways"
            }
        )?;
    }
    log.flush()?;
    Ok(checked.problems.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(json: &str) -> Metadata {
        serde_json::from_str(json).unwrap()
    }

    const WORKSPACE: &str = r#"{
        "workspace_root": "/w",
        "packages": [
            {"id": "cli-id", "name": "syntropy-cli", "features": {}},
            {"id": "obs-id", "name": "syntropy-obs",
             "features": {"fault-injection": [], "default": []}},
            {"id": "exec-id", "name": "syntropy-exec", "features": {"debug-taps": []}},
            {"id": "cuda-id", "name": "syntropy-kernels-cuda", "features": {}},
            {"id": "testkit-id", "name": "syntropy-testkit", "features": {"guards": []}},
            {"id": "xtask-id", "name": "syntropy-xtask", "features": {}}
        ],
        "workspace_members": ["cli-id", "obs-id", "exec-id", "cuda-id", "testkit-id", "xtask-id"]
    }"#;

    fn layers() -> Layers {
        Layers::parse(
            "[layers]\n0 = [\"obs\"]\n1 = [\"exec\"]\n2 = [\"kernels-cuda\"]\n3 = [\"cli\"]\n\
             [outside]\ndev-only = [\"testkit\"]\ntools = [\"xtask\"]\n",
        )
        .unwrap()
    }

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn the_profiles_are_rfc_0062_section_4_4s() {
        let all: &[&str] = &["fault-injection", "debug-taps", "guards"];
        let table: Vec<(&str, &str, &[&str], bool, Scope)> = PROFILES
            .iter()
            .map(|p| {
                (
                    p.name,
                    p.cargo_profile,
                    p.features,
                    p.debug_assertions,
                    p.scope,
                )
            })
            .collect();
        assert_eq!(
            table,
            [
                ("dev", "dev", all, true, Scope::Workspace),
                ("test-release", "test-release", all, true, Scope::Workspace),
                (
                    "chaos",
                    "chaos",
                    &["fault-injection"][..],
                    false,
                    Scope::Production
                ),
                ("release", "release", &[][..], false, Scope::ReleaseCrates),
            ]
        );
        assert_eq!(find("chaos").map(|p| p.name), Some("chaos"));
        assert_eq!(find("pr"), None);
        assert_eq!(names(), ["dev", "test-release", "chaos", "release"]);
    }

    #[test]
    fn a_workspace_profile_passes_each_declared_test_feature() {
        let plan = plan(find("test-release").unwrap(), &metadata(WORKSPACE), None).unwrap();
        assert_eq!(
            plan.builds,
            [Build {
                packages: strings(&["--workspace"]),
                features: vec!["fault-injection", "debug-taps", "guards"],
            }]
        );
        assert_eq!(plan.undeclared, Vec::<&str>::new());
        assert_eq!(plan.members.len(), 6);
    }

    #[test]
    fn chaos_leaves_out_dev_only_crates_and_tools() {
        let chaos = find("chaos").unwrap();
        let plan = plan(chaos, &metadata(WORKSPACE), Some(&layers())).unwrap();
        assert_eq!(
            plan.builds,
            [Build {
                packages: strings(&[
                    "--workspace",
                    "--exclude",
                    "syntropy-testkit",
                    "--exclude",
                    "syntropy-xtask"
                ]),
                features: vec!["fault-injection"],
            }]
        );
        assert!(super::plan(chaos, &metadata(WORKSPACE), None).is_err());
    }

    #[test]
    fn a_feature_no_selected_member_declares_is_not_passed() {
        let json = WORKSPACE.replace(r#""features": {"guards": []}"#, r#""features": {}"#);
        let plan = plan(find("dev").unwrap(), &metadata(&json), None).unwrap();
        assert_eq!(plan.builds[0].features, ["fault-injection", "debug-taps"]);
        assert_eq!(plan.undeclared, ["guards"]);
    }

    #[test]
    fn release_builds_each_release_crate_on_its_own_with_no_features() {
        let plan = plan(find("release").unwrap(), &metadata(WORKSPACE), None).unwrap();
        assert_eq!(
            plan.builds,
            [
                Build {
                    packages: strings(&["--package", "syntropy-cli"]),
                    features: vec![],
                },
                Build {
                    packages: strings(&["--package", "syntropy-kernels-cuda"]),
                    features: vec![],
                },
            ]
        );
        let none = r#"{"workspace_root": "/w",
                       "packages": [{"id": "obs-id", "name": "syntropy-obs"}],
                       "workspace_members": ["obs-id"]}"#;
        assert_eq!(
            super::plan(find("release").unwrap(), &metadata(none), None),
            Err("the workspace has no release crate".to_owned())
        );
    }

    fn artifact(id: &str, kind: &str, debug_assertions: bool, features: &[&str]) -> String {
        serde_json::json!({
            "reason": "compiler-artifact",
            "package_id": id,
            "target": {"name": "unit", "kind": [kind]},
            "profile": {"opt_level": "3", "debug_assertions": debug_assertions},
            "features": features,
        })
        .to_string()
    }

    fn members() -> BTreeMap<String, String> {
        [("obs-id", "syntropy-obs"), ("cli-id", "syntropy-cli")]
            .into_iter()
            .map(|(id, name)| (id.to_owned(), name.to_owned()))
            .collect()
    }

    fn checked(profile: &str, stdout: &str) -> Checked {
        let mut checked = Checked::default();
        checked.add(find(profile).unwrap(), &members(), stdout);
        checked.finish();
        checked
    }

    #[test]
    fn a_build_that_matches_its_profile_passes() {
        let stdout = [
            artifact("obs-id", "lib", true, &["default", "fault-injection"]),
            artifact("cli-id", "bin", true, &[]),
            // An external crate, a build script and a proc macro are not checked.
            artifact("serde-id", "lib", false, &["guards"]),
            artifact("obs-id", "custom-build", false, &[]),
            artifact("cli-id", "proc-macro", false, &["guards"]),
            r#"{"reason": "build-finished", "success": true}"#.to_owned(),
            "not json".to_owned(),
        ]
        .join("\n");
        let checked = checked("test-release", &stdout);
        assert_eq!(checked.units, 2);
        assert_eq!(
            checked.on.into_iter().collect::<Vec<_>>(),
            ["obs/fault-injection"]
        );
        assert_eq!(checked.problems, Vec::<String>::new());
    }

    #[test]
    fn debug_assertions_and_features_outside_the_profile_fail() {
        let stdout = [
            artifact("obs-id", "lib", true, &["guards"]),
            artifact("cli-id", "bin", false, &[]),
        ]
        .join("\n");
        assert_eq!(
            checked("release", &stdout).problems,
            [
                "`obs` (lib `unit`) was built with debug assertions on; the `release` profile \
                 has them off",
                "`obs` (lib `unit`) was built with `guards`, which the `release` profile does not \
                 turn on",
            ]
        );
        assert_eq!(
            checked(
                "chaos",
                &artifact("obs-id", "lib", false, &["fault-injection", "debug-taps"])
            )
            .problems,
            [
                "`obs` (lib `unit`) was built with `debug-taps`, which the `chaos` profile does not \
              turn on"
            ]
        );
        assert_eq!(
            checked("test-release", &artifact("cli-id", "bin", false, &[])).problems,
            [
                "`cli` (bin `unit`) was built with debug assertions off; the `test-release` profile \
              has them on"
            ]
        );
    }

    #[test]
    fn a_build_that_reports_no_member_unit_fails() {
        assert_eq!(
            checked("dev", "").problems,
            ["cargo reported no unit of a workspace crate, so nothing was checked"]
        );
    }

    #[test]
    fn lists_read_as_prose() {
        assert_eq!(listed(&[]), "none");
        assert_eq!(listed(&strings(&["a"])), "a");
        assert_eq!(listed(&strings(&["a", "b", "c"])), "a, b and c");
    }

    /// The shape cargo writes: one entry per query, keyed by a hash; the `-vV` query is the one
    /// with a `release:` line.
    fn rustc_info(version: &str) -> String {
        format!(
            r#"{{"rustc_fingerprint": 1, "outputs": {{
                "1": {{"success": true, "status": "", "code": 0,
                       "stdout": "rustc {version} (58cf8f9c3 2026-10-13)\nbinary: rustc\nhost: x86_64-unknown-linux-gnu\nrelease: {version}\nLLVM version: 23.1.1\n",
                       "stderr": ""}},
                "2": {{"success": true, "status": "", "code": 0,
                       "stdout": "___\nlib___.rlib\ndebug_assertions\n", "stderr": ""}}
            }}, "successes": {{}}}}"#
        )
    }

    #[test]
    fn a_release_build_by_the_pinned_stable_compiler_passes() {
        let releases = recorded_releases(&rustc_info("1.99.1")).unwrap();
        assert_eq!(releases, BTreeSet::from(["1.99.1".to_owned()]));
        assert!(check_release_compiler(&releases, "1.99.1").is_empty());
    }

    #[test]
    fn a_release_build_by_the_test_nightly_or_another_release_fails() {
        for version in ["1.101.0-nightly", "1.100.0-beta.3", "1.99.1-dev", "1.98.0"] {
            let releases = recorded_releases(&rustc_info(version)).unwrap();
            let problems = check_release_compiler(&releases, "1.99.1");
            assert_eq!(problems.len(), 1, "{version}");
            assert!(problems[0].contains(version), "{}", problems[0]);
        }
        // A two-part pin matches any patch of its release, so only the suffix check stops a
        // nightly of that release.
        assert!(crate::toolchain::matches("1.99", "1.99.1-nightly"));
        let nightly = BTreeSet::from(["1.99.1-nightly".to_owned()]);
        assert_eq!(check_release_compiler(&nightly, "1.99").len(), 1);
        assert!(check_release_compiler(&BTreeSet::from(["1.99.1".to_owned()]), "1.99").is_empty());
    }

    #[test]
    fn a_target_directory_without_a_recorded_compiler_fails() {
        let none = r#"{"rustc_fingerprint": 1, "outputs": {}, "successes": {}}"#;
        assert!(recorded_releases(none).is_err());
        assert!(recorded_releases("not json").is_err());
    }
}
