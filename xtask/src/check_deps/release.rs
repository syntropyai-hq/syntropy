// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rules that need cargo to resolve or build a release crate: rule 7 (the core binary links no
//! vendor GPU runtime) and rule 10 (test-only features never reach a release). Rule 10's second
//! half, [`default_features`], reads only `cargo metadata`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::graph::Graph;
use super::policy::{self, PREFIX};
use super::{Finding, Rule, elf, list};

/// What a rule that may not run produced.
pub enum Outcome {
    /// The rule ran; its findings, possibly none.
    Ran(Vec<Finding>),
    /// The rule could not run here, and why.
    NotRun(String),
}

/// Rule 10: resolves the features of each release crate's build on its own, as its artefact is
/// built (`cargo tree -e normal,build` for every target platform, so dev-dependencies do not
/// count), and reports any workspace crate built with a test-only feature.
pub fn rule_10(manifest: &Path, graph: &Graph) -> Outcome {
    let crates: Vec<&str> = policy::RELEASE_CRATES
        .iter()
        .copied()
        .filter(|name| graph.member(name).is_some())
        .collect();
    if crates.is_empty() {
        return Outcome::NotRun("the workspace has no release crate".to_owned());
    }
    let members: BTreeSet<&str> = graph
        .members()
        .into_iter()
        .map(|at| graph.package(at).name.as_str())
        .collect();
    let mut findings = Vec::new();
    for name in crates {
        let output = Command::new(crate::registry::cargo())
            .args([
                "tree",
                "--locked",
                "--edges",
                "normal,build",
                "--target",
                "all",
            ])
            .args(["--prefix", "depth", "--format", "{p}\t{f}"])
            .arg("--manifest-path")
            .arg(manifest)
            .arg("--package")
            .arg(format!("{PREFIX}{name}"))
            .stderr(Stdio::inherit())
            .output();
        let output = match output {
            Ok(output) if output.status.success() => output,
            Ok(output) => {
                findings.push(Finding::new(
                    Rule::Numbered(10),
                    format!("`cargo tree` for `{name}` failed: {}", output.status),
                ));
                continue;
            }
            Err(error) => {
                findings.push(Finding::new(
                    Rule::Numbered(10),
                    format!("could not start `cargo tree`: {error}"),
                ));
                continue;
            }
        };
        let tree = String::from_utf8_lossy(&output.stdout);
        let mut seen = BTreeSet::new();
        for line in tree.lines() {
            let Some((package, features)) = parse_tree_line(line) else {
                continue;
            };
            if !members.contains(package) {
                continue;
            }
            for feature in features {
                if policy::TEST_FEATURES.contains(&feature) && seen.insert((package, feature)) {
                    let short = package.strip_prefix(PREFIX).unwrap_or(package);
                    findings.push(Finding::new(
                        Rule::Numbered(10),
                        format!(
                            "the release build of `{name}` enables `{short}/{feature}`; release \
                             artefacts are built without {}",
                            list(policy::TEST_FEATURES, "or")
                        ),
                    ));
                }
            }
        }
    }
    Outcome::Ran(findings)
}

/// Rule 10, defaults: no member turns a test-only feature on by default (QUAL-E3-T04). A crate
/// that did would build every dependent that keeps default features with it, release crates
/// included. Each member's `default` is expanded through the `[features]` tables it reaches: its
/// own, and, through an `x/y` or `x?/y` entry, member dependency `x`'s, from feature `y` on
/// (#1556). A finding is a test feature anywhere on that walk, reported once per member with the
/// first path that reaches it. The walk visits each (crate, feature) pair once, so it ends on any
/// cycle. `dep:x` entries enable no feature, and a dependency outside the workspace is not
/// followed.
pub fn default_features(graph: &Graph) -> Vec<Finding> {
    /// A feature of a package: the package's index in the graph, and the feature's name.
    type Pair<'a> = (usize, &'a str);
    let members = graph.members();
    let member_named = |name: &str| {
        members
            .iter()
            .copied()
            .find(|at| graph.package(*at).name == name)
    };
    let mut findings = Vec::new();
    for &start in &members {
        if !graph.package(start).features.contains_key("default") {
            continue;
        }
        let short = graph.short(start);
        // Breadth first over (crate, feature) pairs. `parent` maps each pair to the pair whose
        // entry enabled it, and that entry as written.
        let root = (start, "default");
        let mut parent: BTreeMap<Pair, (Pair, &str)> = BTreeMap::new();
        let mut seen = BTreeSet::from([root]);
        let mut queue = VecDeque::from([root]);
        let mut reported = BTreeSet::new();
        while let Some(node @ (at, feature)) = queue.pop_front() {
            let package = graph.package(at);
            for entry in package.features.get(feature).into_iter().flatten() {
                let entry = entry.as_str();
                if entry.starts_with("dep:") {
                    continue;
                }
                // The pair the entry names: `y` of this crate, or `y` of member dependency `x`.
                let target = match entry.split_once('/') {
                    Some((key, dep_feature)) => {
                        let key = key.strip_suffix('?').unwrap_or(key);
                        package
                            .dependencies
                            .iter()
                            .find(|dep| dep.key() == key)
                            .and_then(|dep| member_named(&dep.name))
                            .map(|dep_at| (dep_at, dep_feature))
                    }
                    None => Some((at, entry)),
                };
                let Some(next @ (next_at, next_feature)) = target else {
                    continue;
                };
                if !policy::TEST_FEATURES.contains(&next_feature) {
                    if seen.insert(next) {
                        parent.insert(next, (node, entry));
                        queue.push_back(next);
                    }
                    continue;
                }
                if !reported.insert(next) {
                    continue;
                }
                let what = if next_at == start {
                    format!("`{next_feature}`")
                } else {
                    format!("`{}/{next_feature}`", graph.short(next_at))
                };
                let mut path = vec![entry];
                let mut up = node;
                while let Some((from, via)) = parent.get(&up) {
                    path.push(via);
                    up = *from;
                }
                path.push("default");
                path.reverse();
                findings.push(Finding::new(
                    Rule::Numbered(10),
                    format!(
                        "`{short}` turns on {what} by default ({}); test-only features are \
                         never default features, or every dependent that keeps default \
                         features would build with them",
                        path.join(" → ")
                    ),
                ));
            }
        }
    }
    findings
}

/// One line of `cargo tree --prefix depth --format "{p}\t{f}"`: the package name and its
/// enabled features. `None` for a blank line.
fn parse_tree_line(line: &str) -> Option<(&str, Vec<&str>)> {
    let line = line.trim_start_matches(|c: char| c.is_ascii_digit());
    // A repeated subtree ends in ` (*)`.
    let line = line.strip_suffix(" (*)").unwrap_or(line).trim_end();
    let (package, features) = line.split_once('\t').unwrap_or((line, ""));
    let name = package.split_whitespace().next()?;
    let features = features
        .split(',')
        .map(str::trim)
        .filter(|feature| !feature.is_empty())
        .collect();
    Some((name, features))
}

/// Rule 7: builds the core binary with no default features, so with no GPU feature, and reads
/// its dynamic section. Every library it needs must be on the system-library allow-list, which
/// fails closed on a vendor library no list names, and no symbol may be a vendor runtime's. The
/// reader handles ELF, so the rule runs on Linux, where `lint` runs; QUAL-E1-T28 adds Windows and
/// macOS.
pub fn rule_7(manifest: &Path, graph: &Graph) -> Outcome {
    let crate_name = policy::CORE_BINARY_CRATE;
    let Some(at) = graph.member(crate_name) else {
        return Outcome::NotRun(format!("the workspace has no `{PREFIX}{crate_name}` crate"));
    };
    if !cfg!(target_os = "linux") {
        return Outcome::NotRun(format!(
            "it reads the ELF dynamic section, and this host is {}; `lint` runs it on Linux, \
             and QUAL-E1-T28 adds Windows and macOS",
            std::env::consts::OS
        ));
    }
    let has_bin = graph.package(at).targets.iter().any(|target| {
        target.name == policy::CORE_BINARY && target.kind.iter().any(|kind| kind == "bin")
    });
    if !has_bin {
        return Outcome::Ran(vec![Finding::new(
            Rule::Numbered(7),
            format!(
                "`{crate_name}` has no bin target `{}`, the core binary",
                policy::CORE_BINARY
            ),
        )]);
    }
    let binary = match build_core_binary(manifest) {
        Ok(binary) => binary,
        Err(error) => return Outcome::Ran(vec![Finding::new(Rule::Numbered(7), error)]),
    };
    let dynamic = match std::fs::read(&binary)
        .map_err(|error| error.to_string())
        .and_then(|bytes| elf::read(&bytes))
    {
        Ok(dynamic) => dynamic,
        Err(error) => {
            return Outcome::Ran(vec![Finding::new(
                Rule::Numbered(7),
                format!("cannot read the core binary {}: {error}", binary.display()),
            )]);
        }
    };
    Outcome::Ran(vendor_findings(&dynamic))
}

/// `cargo build` of the core binary with no default features; returns the executable's path.
fn build_core_binary(manifest: &Path) -> Result<PathBuf, String> {
    let package = format!("{PREFIX}{}", policy::CORE_BINARY_CRATE);
    let output = Command::new(crate::registry::cargo())
        .args(["build", "--locked", "--quiet", "--no-default-features"])
        .args(["--message-format", "json-render-diagnostics"])
        .arg("--manifest-path")
        .arg(manifest)
        .args(["--package", &package, "--bin", policy::CORE_BINARY])
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("could not start `cargo build`: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "the core binary did not build with no default features: {}",
            output.status
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|message| message["reason"] == "compiler-artifact")
        .filter(|message| message["target"]["name"] == policy::CORE_BINARY)
        .find_map(|message| message["executable"].as_str().map(PathBuf::from))
        .ok_or_else(|| "`cargo build` reported no core binary".to_owned())
}

/// Rule 7's findings for a binary's dynamic section: one per `DT_NEEDED` entry that is not a
/// system library, and one per vendor whose symbols it names.
fn vendor_findings(dynamic: &elf::Dynamic) -> Vec<Finding> {
    let mut findings = Vec::new();
    for needed in &dynamic.needed {
        if policy::is_system_library(needed) {
            continue;
        }
        let vendor = policy::VENDOR_RUNTIMES.iter().find(|vendor| {
            vendor
                .libraries
                .iter()
                .any(|stem| policy::is_library(needed, stem))
        });
        let message = match vendor {
            Some(vendor) => format!(
                "the core binary, built with no default features, needs `{needed}` from the {} \
                 runtime; vendor GPU runtimes ship only in backend modules",
                vendor.name
            ),
            None => format!(
                "the core binary, built with no default features, needs `{needed}`, which is not \
                 a system library it may need; vendor GPU runtimes ship only in backend modules, \
                 and a system library that is not one goes in `SYSTEM_LIBRARIES` in \
                 xtask/src/check_deps/policy.rs, with a maintainer's review"
            ),
        };
        findings.push(Finding::new(Rule::Numbered(7), message));
    }
    for vendor in policy::VENDOR_RUNTIMES {
        let symbols: Vec<&str> = dynamic
            .symbols
            .iter()
            .map(String::as_str)
            .filter(|symbol| names_vendor_symbol(vendor, symbol))
            .collect();
        if !symbols.is_empty() {
            let mut shown = symbols[..symbols.len().min(5)].join(", ");
            if symbols.len() > 5 {
                shown.push_str(&format!(" and {} more", symbols.len() - 5));
            }
            findings.push(Finding::new(
                Rule::Numbered(7),
                format!(
                    "the core binary, built with no default features, has {} runtime symbols in \
                     its dynamic symbol table: {shown}; vendor GPU runtimes ship only in backend \
                     modules",
                    vendor.name
                ),
            ));
        }
    }
    findings
}

/// Whether `symbol` is one of `vendor`'s: it starts with one of the vendor's library families, or
/// with an entry-point prefix followed by an ASCII capital or `_`.
fn names_vendor_symbol(vendor: &policy::VendorRuntime, symbol: &str) -> bool {
    vendor
        .families
        .iter()
        .any(|family| symbol.starts_with(family))
        || vendor.entry_prefixes.iter().any(|prefix| {
            symbol
                .strip_prefix(prefix)
                .and_then(|rest| rest.chars().next())
                .is_some_and(|next| next.is_ascii_uppercase() || next == '_')
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults(graph: &Graph) -> Vec<String> {
        default_features(graph)
            .iter()
            .map(Finding::to_string)
            .collect()
    }

    #[test]
    fn a_test_feature_among_a_crates_defaults_is_reported() {
        let graph = Graph::test(&["obs", "exec"], &[], &[])
            .with_features("obs", &[("default", &["guards"]), ("guards", &[])], &[])
            .with_features(
                "exec",
                &[
                    ("default", &["extra"]),
                    ("extra", &["fault-injection", "dep:helper"]),
                    ("fault-injection", &[]),
                ],
                &[],
            );
        assert_eq!(
            defaults(&graph),
            [
                "rule 10: `exec` turns on `fault-injection` by default (default → extra → \
                 fault-injection); test-only features are never default features, or every \
                 dependent that keeps default features would build with them",
                "rule 10: `obs` turns on `guards` by default (default → guards); test-only \
                 features are never default features, or every dependent that keeps default \
                 features would build with them",
            ]
        );
    }

    #[test]
    fn a_default_that_turns_on_a_members_test_feature_is_reported() {
        let graph = Graph::test(&["obs", "testkit", "bench", "net"], &["serde"], &[])
            .with_features("obs", &[("guards", &[]), ("debug-taps", &[])], &[])
            .with_features(
                "testkit",
                &[
                    ("default", &["helpers"]),
                    ("helpers", &["syntropy-obs/guards"]),
                ],
                &[("syntropy-obs", "syntropy-obs")],
            )
            .with_features(
                "bench",
                &[("default", &["o?/debug-taps"])],
                &[("o", "syntropy-obs")],
            )
            // An external crate's feature of the same name is not a test feature.
            .with_features(
                "net",
                &[("default", &["serde/guards"])],
                &[("serde", "serde")],
            );
        let found = defaults(&graph);
        assert_eq!(found.len(), 2, "{found:#?}");
        assert!(found[0].starts_with(
            "rule 10: `bench` turns on `obs/debug-taps` by default (default → o?/debug-taps)"
        ));
        assert!(found[1].starts_with(
            "rule 10: `testkit` turns on `obs/guards` by default (default → helpers → \
             syntropy-obs/guards)"
        ));
    }

    #[test]
    fn a_default_that_reaches_a_test_feature_through_a_member_dependency_is_reported() {
        // #1556's reproduction: `bench` turns on `obs/extra`, and `extra` turns on
        // `fault-injection`. And two crates deep, through `exec`'s `taps`.
        let graph = Graph::test(&["obs", "exec", "bench"], &[], &[])
            .with_features(
                "obs",
                &[
                    ("extra", &["fault-injection"]),
                    ("fault-injection", &[]),
                    ("debug-taps", &[]),
                ],
                &[],
            )
            .with_features(
                "exec",
                &[("taps", &["syntropy-obs/debug-taps"])],
                &[("syntropy-obs", "syntropy-obs")],
            )
            .with_features(
                "bench",
                &[("default", &["syntropy-obs/extra", "syntropy-exec?/taps"])],
                &[
                    ("syntropy-obs", "syntropy-obs"),
                    ("syntropy-exec", "syntropy-exec"),
                ],
            );
        assert_eq!(
            defaults(&graph),
            [
                "rule 10: `bench` turns on `obs/fault-injection` by default (default → \
                 syntropy-obs/extra → fault-injection); test-only features are never default \
                 features, or every dependent that keeps default features would build with them",
                "rule 10: `bench` turns on `obs/debug-taps` by default (default → \
                 syntropy-exec?/taps → syntropy-obs/debug-taps); test-only features are never \
                 default features, or every dependent that keeps default features would build \
                 with them",
            ]
        );
    }

    #[test]
    fn a_feature_cycle_across_two_crates_ends() {
        // Cargo rejects a dependency cycle, but the walk must end on one all the same. The test
        // feature inside the cycle is reported once.
        let graph = Graph::test(&["a", "b"], &[], &[])
            .with_features(
                "a",
                &[("default", &["syntropy-b/y"]), ("x", &["syntropy-b/y"])],
                &[("syntropy-b", "syntropy-b")],
            )
            .with_features(
                "b",
                &[
                    ("y", &["syntropy-a/x", "guards", "z"]),
                    ("z", &["y", "guards"]),
                    ("guards", &[]),
                ],
                &[("syntropy-a", "syntropy-a")],
            );
        assert_eq!(
            defaults(&graph),
            [
                "rule 10: `a` turns on `b/guards` by default (default → syntropy-b/y → guards); \
              test-only features are never default features, or every dependent that keeps \
              default features would build with them"
            ]
        );
    }

    #[test]
    fn a_dependency_outside_the_workspace_is_not_followed() {
        // `serde`'s own `extra` turns on a feature named `guards`, but `serde` is not a member,
        // so neither its table nor its `guards` is a test feature of this workspace.
        let graph = Graph::test(&["net"], &["serde"], &[])
            .with_features("serde", &[("extra", &["guards"]), ("guards", &[])], &[])
            .with_features(
                "net",
                &[("default", &["serde/extra", "serde?/guards"])],
                &[("serde", "serde")],
            );
        assert_eq!(defaults(&graph), Vec::<String>::new());
    }

    #[test]
    fn test_features_outside_the_defaults_pass() {
        let graph = Graph::test(&["obs", "exec"], &[], &[])
            .with_features(
                "obs",
                &[
                    ("default", &["std"]),
                    ("std", &["default"]),
                    ("fault-injection", &[]),
                    ("guards", &["fault-injection"]),
                ],
                &[],
            )
            .with_features("exec", &[("debug-taps", &["syntropy-obs/guards"])], &[]);
        assert_eq!(defaults(&graph), Vec::<String>::new());
    }

    #[test]
    fn tree_lines_give_the_package_and_its_features() {
        assert_eq!(
            parse_tree_line("0syntropy-cli v0.0.0 (/w/crates/cli)\t"),
            Some(("syntropy-cli", vec![]))
        );
        assert_eq!(
            parse_tree_line("2syntropy-obs v0.0.0 (/w/crates/obs)\tdefault,fault-injection (*)"),
            Some(("syntropy-obs", vec!["default", "fault-injection"]))
        );
        assert_eq!(
            parse_tree_line("1serde_derive v1.0.228 (proc-macro)\tdefault"),
            Some(("serde_derive", vec!["default"]))
        );
        assert_eq!(parse_tree_line(""), None);
    }

    fn messages(dynamic: &elf::Dynamic) -> Vec<String> {
        vendor_findings(dynamic)
            .iter()
            .map(Finding::to_string)
            .collect()
    }

    #[test]
    fn a_plain_binary_has_no_vendor_findings() {
        let dynamic = elf::Dynamic {
            needed: vec![
                "libc.so.6".into(),
                "libgcc_s.so.1".into(),
                "libm.so.6".into(),
            ],
            symbols: vec![
                "malloc".into(),
                "clock_gettime".into(),
                "cupcake".into(),
                "_Unwind_Resume".into(),
                "_ZN4core3fmt5write17h0123456789abcdefE".into(),
            ],
        };
        assert_eq!(messages(&dynamic), Vec::<String>::new());
    }

    #[test]
    fn any_library_outside_the_allow_list_fails() {
        let dynamic = elf::Dynamic {
            needed: vec![
                "libc.so.6".into(),
                "ld-linux-x86-64.so.2".into(),
                "libcufft.so".into(),
                "libnccl.so.2".into(),
                "libMIOpen.so.1".into(),
                "libunknown-gpu.so".into(),
            ],
            symbols: vec![],
        };
        assert_eq!(
            messages(&dynamic),
            [
                "rule 7: the core binary, built with no default features, needs `libcufft.so` from \
                 the CUDA runtime; vendor GPU runtimes ship only in backend modules",
                "rule 7: the core binary, built with no default features, needs `libnccl.so.2` from \
                 the CUDA runtime; vendor GPU runtimes ship only in backend modules",
                "rule 7: the core binary, built with no default features, needs `libMIOpen.so.1` \
                 from the HIP runtime; vendor GPU runtimes ship only in backend modules",
                "rule 7: the core binary, built with no default features, needs \
                 `libunknown-gpu.so`, which is not a system library it may need; vendor GPU \
                 runtimes ship only in backend modules, and a system library that is not one goes \
                 in `SYSTEM_LIBRARIES` in xtask/src/check_deps/policy.rs, with a maintainer's \
                 review",
            ]
        );
    }

    #[test]
    fn library_families_match_without_a_capital_after_the_prefix() {
        let dynamic = elf::Dynamic {
            needed: vec![],
            symbols: vec![
                "cufftPlan1d".into(),
                "curand_init".into(),
                "ncclCommInitRank".into(),
                "hipfftPlan1d".into(),
                "rcclGetVersion".into(),
                "miopenCreate".into(),
            ],
        };
        assert_eq!(
            messages(&dynamic),
            [
                "rule 7: the core binary, built with no default features, has CUDA runtime symbols in \
                 its dynamic symbol table: cufftPlan1d, curand_init, ncclCommInitRank; vendor GPU \
                 runtimes ship only in backend modules",
                "rule 7: the core binary, built with no default features, has HIP runtime symbols in \
                 its dynamic symbol table: hipfftPlan1d, rcclGetVersion, miopenCreate; vendor GPU \
                 runtimes ship only in backend modules",
            ]
        );
    }

    #[test]
    fn vendor_libraries_and_symbols_are_reported() {
        let dynamic = elf::Dynamic {
            needed: vec![
                "libc.so.6".into(),
                "libcuda.so.1".into(),
                "libvulkan.so".into(),
            ],
            symbols: vec![
                "cuInit".into(),
                "cuMemAlloc_v2".into(),
                "cudaMalloc".into(),
                "__cudaRegisterFatBinary".into(),
                "cublasCreate_v2".into(),
                "nvmlInit_v2".into(),
                "vkCreateInstance".into(),
                "malloc".into(),
            ],
        };
        assert_eq!(
            messages(&dynamic),
            [
                "rule 7: the core binary, built with no default features, needs `libcuda.so.1` from \
                 the CUDA runtime; vendor GPU runtimes ship only in backend modules",
                "rule 7: the core binary, built with no default features, needs `libvulkan.so` from \
                 the Vulkan runtime; vendor GPU runtimes ship only in backend modules",
                "rule 7: the core binary, built with no default features, has CUDA runtime symbols in \
                 its dynamic symbol table: cuInit, cuMemAlloc_v2, cudaMalloc, \
                 __cudaRegisterFatBinary, cublasCreate_v2 and 1 more; vendor GPU runtimes ship \
                 only in backend modules",
                "rule 7: the core binary, built with no default features, has Vulkan runtime symbols \
                 in its dynamic symbol table: vkCreateInstance; vendor GPU runtimes ship only in \
                 backend modules",
            ]
        );
    }
}
