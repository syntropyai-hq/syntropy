// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rules that need cargo to resolve or build a release crate: rule 7 (the core binary links no
//! vendor GPU runtime) and rule 10 (test-only features never reach a release).

use std::collections::BTreeSet;
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
/// its dynamic section. It must need no vendor GPU runtime library and name no vendor runtime
/// symbol. The reader handles ELF, so the rule runs on Linux, where `lint` runs.
pub fn rule_7(manifest: &Path, graph: &Graph) -> Outcome {
    let crate_name = policy::CORE_BINARY_CRATE;
    let Some(at) = graph.member(crate_name) else {
        return Outcome::NotRun(format!("the workspace has no `{PREFIX}{crate_name}` crate"));
    };
    if !cfg!(target_os = "linux") {
        return Outcome::NotRun(format!(
            "it reads the ELF dynamic section, and this host is {}; `lint` runs it on Linux",
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

/// Rule 7's findings for a binary's dynamic section: one per vendor library it needs, and one per
/// vendor whose symbols it names.
fn vendor_findings(dynamic: &elf::Dynamic) -> Vec<Finding> {
    let mut findings = Vec::new();
    for vendor in policy::VENDOR_RUNTIMES {
        for needed in &dynamic.needed {
            let is_vendor = vendor.libraries.iter().any(|stem| {
                needed
                    .strip_prefix(stem)
                    .is_some_and(|rest| rest == ".so" || rest.starts_with(".so."))
            });
            if is_vendor {
                findings.push(Finding::new(
                    Rule::Numbered(7),
                    format!(
                        "the core binary, built with no default features, needs `{needed}` from the {} \
                         runtime; vendor GPU runtimes ship only in backend modules",
                        vendor.name
                    ),
                ));
            }
        }
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
                    "the core binary, built with no default features, has {} runtime symbols in its \
                     dynamic symbol table: {shown}; vendor GPU runtimes ship only in backend \
                     modules",
                    vendor.name
                ),
            ));
        }
    }
    findings
}

/// Whether `symbol` is one of `vendor`'s: a listed prefix followed by an ASCII capital or `_`.
fn names_vendor_symbol(vendor: &policy::VendorRuntime, symbol: &str) -> bool {
    vendor.symbol_prefixes.iter().any(|prefix| {
        symbol
            .strip_prefix(prefix)
            .and_then(|rest| rest.chars().next())
            .is_some_and(|next| next.is_ascii_uppercase() || next == '_')
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn vendor_libraries_and_symbols_are_reported() {
        let dynamic = elf::Dynamic {
            needed: vec![
                "libc.so.6".into(),
                "libcuda.so.1".into(),
                "libvulkan.so".into(),
                "libcudaxyz.so".into(),
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
                "rule 7: the core binary, built with no default features, has CUDA runtime symbols in \
                 its dynamic symbol table: cuInit, cuMemAlloc_v2, cudaMalloc, \
                 __cudaRegisterFatBinary, cublasCreate_v2 and 1 more; vendor GPU runtimes ship \
                 only in backend modules",
                "rule 7: the core binary, built with no default features, needs `libvulkan.so` from \
                 the Vulkan runtime; vendor GPU runtimes ship only in backend modules",
                "rule 7: the core binary, built with no default features, has Vulkan runtime symbols \
                 in its dynamic symbol table: vkCreateInstance; vendor GPU runtimes ship only in \
                 backend modules",
            ]
        );
    }
}
