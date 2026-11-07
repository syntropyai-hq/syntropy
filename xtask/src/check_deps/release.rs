// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rules that need cargo to resolve or build a release crate: rule 7 (the core binary links no
//! vendor GPU runtime) and rule 10 (test-only features never reach a release). Rule 10 has three
//! parts: [`rule_10`] resolves each release crate's features, [`default_features`] reads only
//! `cargo metadata`, and [`fault_point_symbols`] reads the core binary's symbol table.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::graph::Graph;
use super::policy::{self, PREFIX};
use super::{Finding, Rule, elf, list, macho, pe};

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

/// Rule 7: builds the core binary with no default features, so with no GPU feature, and reads the
/// libraries it makes the loader load: its ELF `DT_NEEDED` entries on Linux, its PE import and
/// delay-load directories on Windows, and its Mach-O `LC_LOAD_DYLIB`, `LC_LOAD_WEAK_DYLIB`,
/// `LC_REEXPORT_DYLIB`, `LC_LAZY_LOAD_DYLIB` and `LC_LOAD_UPWARD_DYLIB` commands on macOS, every
/// architecture of a universal file included (QUAL-E1-T28). Every library must be on the
/// allow-list of system libraries for its format, which fails closed on a vendor library no list
/// names, and no symbol it binds may be a vendor runtime's. Libraries `kernels-loader` opens at
/// run time are rule 6's, and no load command names them.
pub fn rule_7(manifest: &Path, graph: &Graph) -> Outcome {
    let crate_name = policy::CORE_BINARY_CRATE;
    let Some(at) = graph.member(crate_name) else {
        return Outcome::NotRun(format!("the workspace has no `{PREFIX}{crate_name}` crate"));
    };
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
    let binary = match build_core_binary(manifest, Features::None) {
        Ok(binary) => binary,
        Err(error) => return Outcome::Ran(vec![Finding::new(Rule::Numbered(7), error)]),
    };
    let linked = match std::fs::read(&binary)
        .map_err(|error| error.to_string())
        .and_then(|bytes| Linked::read(&bytes))
    {
        Ok(linked) => linked,
        Err(error) => {
            return Outcome::Ran(vec![Finding::new(
                Rule::Numbered(7),
                format!("cannot read the core binary {}: {error}", binary.display()),
            )]);
        }
    };
    Outcome::Ran(vendor_findings(&linked))
}

/// The executable formats rule 7 reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Elf,
    Pe,
    MachO,
}

/// What rule 7 reads of a binary, whatever its format.
#[derive(Debug)]
struct Linked {
    format: Format,
    /// Each library the loader loads for it, with how the binary names it, such as `needs` for a
    /// `DT_NEEDED` entry.
    libraries: Vec<(String, &'static str)>,
    /// The symbols the loader binds for it.
    symbols: Vec<String>,
    /// The symbols it exports to the loader: defined `.dynsym` entries, the PE export directory,
    /// or the Mach-O external symbols defined in a section.
    exports: Vec<String>,
}

impl Linked {
    /// Reads `bytes` as an ELF, PE or Mach-O file, by its magic number.
    fn read(bytes: &[u8]) -> Result<Linked, String> {
        if bytes.starts_with(b"\x7fELF") {
            elf::read(bytes).map(Linked::from)
        } else if bytes.starts_with(b"MZ") {
            pe::read(bytes).map(Linked::from)
        } else if macho::is_mach_o(bytes) {
            macho::read(bytes).map(Linked::from)
        } else {
            Err("not an ELF, PE or Mach-O file".to_owned())
        }
    }

    /// Whether the core binary may load `library`.
    fn allows(&self, library: &str) -> bool {
        match self.format {
            Format::Elf => policy::is_system_library(library),
            Format::Pe => policy::is_windows_system_library(library),
            Format::MachO => policy::is_macos_system_library(library),
        }
    }

    /// The vendor runtime `library` belongs to, if it is a known one.
    fn vendor(&self, library: &str) -> Option<&'static policy::VendorRuntime> {
        policy::VENDOR_RUNTIMES
            .iter()
            .find(|vendor| match self.format {
                Format::Elf => vendor
                    .libraries
                    .iter()
                    .any(|stem| policy::is_library(library, stem)),
                Format::Pe => vendor.dlls.iter().any(|stem| policy::is_dll(library, stem)),
                Format::MachO => vendor
                    .libraries
                    .iter()
                    .any(|stem| policy::is_dylib(library, stem)),
            })
    }

    /// The allow-list a finding points at, in xtask/src/check_deps/policy.rs.
    fn allow_list(&self) -> &'static str {
        match self.format {
            Format::Elf => "`SYSTEM_LIBRARIES`",
            Format::Pe => "`WINDOWS_SYSTEM_LIBRARIES`",
            Format::MachO => "`MACOS_SYSTEM_LIBRARIES` or `MACOS_FRAMEWORKS`",
        }
    }

    /// Where the symbols come from, for a finding.
    fn symbol_source(&self) -> &'static str {
        match self.format {
            Format::Elf => "its dynamic symbol table",
            Format::Pe => "its import name tables",
            Format::MachO => "its undefined symbols",
        }
    }
}

impl From<elf::Dynamic> for Linked {
    fn from(dynamic: elf::Dynamic) -> Linked {
        Linked {
            format: Format::Elf,
            libraries: dynamic
                .needed
                .into_iter()
                .map(|lib| (lib, "needs"))
                .collect(),
            symbols: dynamic.symbols,
            exports: dynamic.exports,
        }
    }
}

impl From<pe::Imports> for Linked {
    fn from(imports: pe::Imports) -> Linked {
        let imported = imports.dlls.into_iter().map(|dll| (dll, "imports"));
        let delayed = imports.delayed.into_iter().map(|dll| (dll, "delay-loads"));
        Linked {
            format: Format::Pe,
            libraries: imported.chain(delayed).collect(),
            symbols: imports.symbols,
            exports: imports.exports,
        }
    }
}

impl From<macho::Loads> for Linked {
    fn from(loads: macho::Loads) -> Linked {
        Linked {
            format: Format::MachO,
            libraries: loads
                .dylibs
                .into_iter()
                .map(|dylib| {
                    let how = match dylib.command {
                        macho::Command::Load => "loads",
                        macho::Command::Weak => "weakly loads",
                        macho::Command::Reexport => "re-exports",
                        macho::Command::Lazy => "lazily loads",
                        macho::Command::Upward => "loads upward",
                    };
                    (dylib.name, how)
                })
                .collect(),
            symbols: loads.symbols,
            exports: loads.exports,
        }
    }
}

/// The path, as both Rust manglings spell it, of `syntropy_obs::faults::inject`: the module that
/// holds every item only `obs`'s `fault-injection` feature compiles (RFC-0060 §9.2). The legacy
/// mangling gives `_ZN12syntropy_obs6faults6inject5check17h…E`, and v0 gives
/// `_RNv…12syntropy_obs6faults6inject5check`.
pub const FAULT_POINT_PATH: &str = "12syntropy_obs6faults6inject";

/// Rule 10, symbols (QUAL-E9-T01): builds the core binary with the features its release build
/// resolves, its own defaults with no test feature, and reads its full symbol table (`.symtab`).
/// No symbol may come from `syntropy_obs::faults::inject`, so the binary holds no fault point.
///
/// The build uses cargo's `dev` profile, which keeps every symbol and is fast enough for `lint`.
/// The features, and so the code compiled in, are the release build's. Scanning the nightly
/// `release` artefact itself is QUAL-E16-T15's. The reader handles ELF, so this part runs on
/// Linux, where `lint` runs; rule 7 reads Windows and macOS binaries too.
pub fn fault_point_symbols(manifest: &Path, graph: &Graph) -> Outcome {
    let crate_name = policy::CORE_BINARY_CRATE;
    let Some(at) = graph.member(crate_name) else {
        return Outcome::NotRun(format!("the workspace has no `{PREFIX}{crate_name}` crate"));
    };
    if !cfg!(target_os = "linux") {
        return Outcome::NotRun(format!(
            "its symbol-table half reads the ELF `.symtab`, and this host is {}; `lint` runs it \
             on Linux",
            std::env::consts::OS
        ));
    }
    let has_bin = graph.package(at).targets.iter().any(|target| {
        target.name == policy::CORE_BINARY && target.kind.iter().any(|kind| kind == "bin")
    });
    if !has_bin {
        // Rule 7 reports the missing core binary.
        return Outcome::Ran(Vec::new());
    }
    let binary = match build_core_binary(manifest, Features::Default) {
        Ok(binary) => binary,
        Err(error) => return Outcome::Ran(vec![Finding::new(Rule::Numbered(10), error)]),
    };
    match std::fs::read(&binary)
        .map_err(|error| error.to_string())
        .and_then(|bytes| elf::symtab(&bytes))
    {
        Ok(names) => Outcome::Ran(fault_point_findings(&names)),
        Err(error) => Outcome::Ran(vec![Finding::new(
            Rule::Numbered(10),
            format!("cannot read the core binary {}: {error}", binary.display()),
        )]),
    }
}

/// Rule 10's finding for a symbol table that names `syntropy_obs::faults::inject`, if it does.
fn fault_point_findings(names: &[String]) -> Vec<Finding> {
    let found: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|name| name.contains(FAULT_POINT_PATH))
        .collect();
    if found.is_empty() {
        return Vec::new();
    }
    let mut shown = found[..found.len().min(3)].join(", ");
    if found.len() > 3 {
        shown.push_str(&format!(" and {} more", found.len() - 3));
    }
    vec![Finding::new(
        Rule::Numbered(10),
        format!(
            "the core binary, built with its release features, holds {} {} of \
             `syntropy_obs::faults::inject`, which only `fault-injection` compiles: {shown}; \
             release artefacts carry no fault point",
            found.len(),
            if found.len() == 1 {
                "symbol"
            } else {
                "symbols"
            }
        ),
    )]
}

/// Which features [`build_core_binary`] builds with.
#[derive(Clone, Copy)]
enum Features {
    /// `--no-default-features`: no GPU feature, for rule 7.
    None,
    /// The crate's defaults, as its release build resolves them, for rule 10.
    Default,
}

/// `cargo build` of the core binary in the `dev` profile; returns the executable's path.
fn build_core_binary(manifest: &Path, features: Features) -> Result<PathBuf, String> {
    let package = format!("{PREFIX}{}", policy::CORE_BINARY_CRATE);
    let mut command = Command::new(crate::registry::cargo());
    command.args(["build", "--locked", "--quiet"]);
    if let Features::None = features {
        command.arg("--no-default-features");
    }
    let output = command
        .args(["--message-format", "json-render-diagnostics"])
        .arg("--manifest-path")
        .arg(manifest)
        .args(["--package", &package, "--bin", policy::CORE_BINARY])
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("could not start `cargo build`: {error}"))?;
    if !output.status.success() {
        let with = match features {
            Features::None => "no default features",
            Features::Default => "its default features",
        };
        return Err(format!(
            "the core binary did not build with {with}: {}",
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

/// Rule 7's findings for a binary's linkage: one per library that is not a system library, and
/// one per vendor whose symbols it binds.
fn vendor_findings(linked: &Linked) -> Vec<Finding> {
    let mut findings = Vec::new();
    // The core links its own static copy of ggml with every symbol hidden (ADR-0403 item 6;
    // COMP-E7-T01): none may reach its dynamic symbol table.
    let ggml: Vec<&str> = linked
        .exports
        .iter()
        .map(String::as_str)
        .filter(|symbol| policy::is_ggml_symbol(symbol))
        .collect();
    if !ggml.is_empty() {
        let mut shown = ggml[..ggml.len().min(5)].join(", ");
        if ggml.len() > 5 {
            shown.push_str(&format!(" and {} more", ggml.len() - 5));
        }
        findings.push(Finding::new(
            Rule::Numbered(7),
            format!(
                "the core binary exports ggml symbols: {shown}; its static ggml copy must keep \
                 every symbol hidden (ADR-0403 item 6)"
            ),
        ));
    }
    for (library, how) in &linked.libraries {
        if linked.allows(library) {
            continue;
        }
        let message = match linked.vendor(library) {
            Some(vendor) => format!(
                "the core binary, built with no default features, {how} `{library}` from the {} \
                 runtime; vendor GPU runtimes ship only in backend modules",
                vendor.name
            ),
            None => format!(
                "the core binary, built with no default features, {how} `{library}`, which is not \
                 a system library it may need; vendor GPU runtimes ship only in backend modules, \
                 and a system library that is not one goes in {} in \
                 xtask/src/check_deps/policy.rs, with a maintainer's review",
                linked.allow_list()
            ),
        };
        findings.push(Finding::new(Rule::Numbered(7), message));
    }
    for vendor in policy::VENDOR_RUNTIMES {
        let symbols: Vec<&str> = linked
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
                     {}: {shown}; vendor GPU runtimes ship only in backend modules",
                    vendor.name,
                    linked.symbol_source()
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
    fn fault_point_symbols_are_found_in_either_mangling() {
        let names = |list: &[&str]| list.iter().map(|name| name.to_string()).collect::<Vec<_>>();
        let clean = names(&[
            "main",
            "_ZN12syntropy_obs6faults11FaultEffect3fmt17h0123456789abcdefE",
            "_ZN12syntropy_obs9cost_term8CostTerm5as_u817h0123456789abcdefE",
        ]);
        assert!(fault_point_findings(&clean).is_empty());
        let messages = |list: &[String]| {
            fault_point_findings(list)
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        };
        let legacy = names(&[
            "main",
            "_ZN12syntropy_obs6faults6inject5check17h0123456789abcdefE",
        ]);
        assert_eq!(
            messages(&legacy),
            [
                "rule 10: the core binary, built with its release features, holds 1 symbol of \
              `syntropy_obs::faults::inject`, which only `fault-injection` compiles: \
              _ZN12syntropy_obs6faults6inject5check17h0123456789abcdefE; release artefacts carry \
              no fault point"
            ]
        );
        let v0 = names(&[
            "_RNvNtNtCs1a2b3c_12syntropy_obs6faults6inject5check",
            "_RNvNtNtCs1a2b3c_12syntropy_obs6faults6inject7install",
            "_RNvNtNtCs1a2b3c_12syntropy_obs6faults6inject5clear",
            "_RNvNtNtCs1a2b3c_12syntropy_obs6faults6inject8counters",
        ]);
        let found = messages(&v0);
        assert_eq!(found.len(), 1);
        assert!(found[0].contains("holds 4 symbols"), "{found:?}");
        assert!(
            found[0].contains("_RNvNtNtCs1a2b3c_12syntropy_obs6faults6inject5clear and 1 more")
        );
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

    fn messages(dynamic: elf::Dynamic) -> Vec<String> {
        vendor_findings(&Linked::from(dynamic))
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
            exports: vec![],
        };
        assert_eq!(messages(dynamic), Vec::<String>::new());
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
            exports: vec![],
        };
        assert_eq!(
            messages(dynamic),
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
            exports: vec![],
        };
        assert_eq!(
            messages(dynamic),
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
            exports: vec![],
        };
        assert_eq!(
            messages(dynamic),
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

    /// The core's static ggml copy keeps every symbol hidden: an exported `ggml_` or `gguf_`
    /// symbol fails, and an imported one is not an export (COMP-E7-T01).
    #[test]
    fn an_exported_ggml_symbol_fails() {
        let dynamic = elf::Dynamic {
            needed: vec!["libc.so.6".into()],
            symbols: vec![
                "malloc".into(),
                "ggml_init".into(),
                "gguf_init_empty".into(),
            ],
            exports: vec!["ggml_init".into(), "gguf_init_empty".into()],
        };
        assert_eq!(
            messages(dynamic),
            [
                "rule 7: the core binary exports ggml symbols: ggml_init, gguf_init_empty; its \
              static ggml copy must keep every symbol hidden (ADR-0403 item 6)"
            ]
        );
        let imported = elf::Dynamic {
            needed: vec!["libc.so.6".into()],
            symbols: vec!["ggml_init".into()],
            exports: vec![],
        };
        assert_eq!(messages(imported), Vec::<String>::new());
        let pe = pe::fixture::build_with_exports(true, &[], &[], true, &["ggml_free"]);
        assert_eq!(
            read_messages(&pe),
            [
                "rule 7: the core binary exports ggml symbols: ggml_free; its static ggml copy must \
              keep every symbol hidden (ADR-0403 item 6)"
            ]
        );
        assert!(policy::is_ggml_symbol("gguf_get_key") && !policy::is_ggml_symbol("myggml_x"));
    }

    fn read_messages(bytes: &[u8]) -> Vec<String> {
        vendor_findings(&Linked::read(bytes).unwrap())
            .iter()
            .map(Finding::to_string)
            .collect()
    }

    #[test]
    fn the_format_is_read_from_the_magic_number() {
        assert_eq!(
            Linked::read(b"#!/bin/sh\n").unwrap_err(),
            "not an ELF, PE or Mach-O file"
        );
        assert_eq!(
            Linked::read(b"").unwrap_err(),
            "not an ELF, PE or Mach-O file"
        );
        let pe = pe::fixture::build(true, &[], &[], true);
        assert_eq!(Linked::read(&pe).unwrap().format, Format::Pe);
        let macho = macho::fixture::executable(true, true, &[], &[]);
        assert_eq!(Linked::read(&macho).unwrap().format, Format::MachO);
        let fat = macho::fixture::universal(false, &[macho]);
        assert_eq!(Linked::read(&fat).unwrap().format, Format::MachO);
    }

    #[test]
    fn a_windows_binary_importing_only_system_dlls_passes() {
        let kernel = pe::fixture::Dll {
            name: "KERNEL32.dll",
            imports: &[Some("GetLastError"), Some("CloseHandle")],
        };
        let crt = pe::fixture::Dll {
            name: "api-ms-win-crt-runtime-l1-1-0.dll",
            imports: &[Some("_initterm"), None],
        };
        let vcruntime = pe::fixture::Dll {
            name: "VCRUNTIME140.dll",
            imports: &[Some("memcpy"), Some("__C_specific_handler")],
        };
        let file = pe::fixture::build(true, &[kernel, crt], &[vcruntime], true);
        assert_eq!(read_messages(&file), Vec::<String>::new());
    }

    #[test]
    fn a_windows_binary_that_imports_or_delay_loads_a_vendor_dll_fails() {
        let kernel = pe::fixture::Dll {
            name: "KERNEL32.dll",
            imports: &[Some("GetLastError")],
        };
        let cuda = pe::fixture::Dll {
            name: "nvcuda.dll",
            imports: &[Some("cuInit")],
        };
        let vulkan = pe::fixture::Dll {
            name: "vulkan-1.dll",
            imports: &[Some("vkCreateInstance")],
        };
        let unknown = pe::fixture::Dll {
            name: "gpuvendor.dll",
            imports: &[None],
        };
        let file = pe::fixture::build(true, &[kernel, cuda], &[vulkan, unknown], true);
        assert_eq!(
            read_messages(&file),
            [
                "rule 7: the core binary, built with no default features, imports `nvcuda.dll` \
                 from the CUDA runtime; vendor GPU runtimes ship only in backend modules",
                "rule 7: the core binary, built with no default features, delay-loads \
                 `vulkan-1.dll` from the Vulkan runtime; vendor GPU runtimes ship only in backend \
                 modules",
                "rule 7: the core binary, built with no default features, delay-loads \
                 `gpuvendor.dll`, which is not a system library it may need; vendor GPU runtimes \
                 ship only in backend modules, and a system library that is not one goes in \
                 `WINDOWS_SYSTEM_LIBRARIES` in xtask/src/check_deps/policy.rs, with a \
                 maintainer's review",
                "rule 7: the core binary, built with no default features, has CUDA runtime symbols \
                 in its import name tables: cuInit; vendor GPU runtimes ship only in backend \
                 modules",
                "rule 7: the core binary, built with no default features, has Vulkan runtime \
                 symbols in its import name tables: vkCreateInstance; vendor GPU runtimes ship \
                 only in backend modules",
            ]
        );
    }

    #[test]
    fn a_macos_binary_loading_only_system_frameworks_and_metal_passes() {
        let dylibs = [
            ("/usr/lib/libSystem.B.dylib", macho::Command::Load),
            ("/usr/lib/libobjc.A.dylib", macho::Command::Load),
            (
                "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
                macho::Command::Load,
            ),
            (
                "/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation",
                macho::Command::Weak,
            ),
            (
                "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation",
                macho::Command::Load,
            ),
            (
                "/System/Library/Frameworks/CoreGraphics.framework/Versions/A/CoreGraphics",
                macho::Command::Load,
            ),
        ];
        let symbols = [
            "malloc",
            "MTLCreateSystemDefaultDevice",
            "objc_msgSend",
            "CFRelease",
        ];
        let arm = macho::fixture::executable(true, true, &dylibs, &symbols);
        let intel = macho::fixture::executable(true, true, &dylibs[..1], &symbols[..1]);
        assert_eq!(read_messages(&arm), Vec::<String>::new());
        let fat = macho::fixture::universal(false, &[arm, intel]);
        assert_eq!(read_messages(&fat), Vec::<String>::new());
    }

    #[test]
    fn a_macos_binary_loading_a_vendor_library_in_any_architecture_fails() {
        let system = ("/usr/lib/libSystem.B.dylib", macho::Command::Load);
        let arm = macho::fixture::executable(true, true, &[system], &["malloc"]);
        let intel = macho::fixture::executable(
            true,
            true,
            &[
                system,
                ("@rpath/libvulkan.1.dylib", macho::Command::Load),
                ("/opt/gpu/libgpu.dylib", macho::Command::Weak),
                (
                    "/System/Library/Frameworks/OpenCL.framework/Versions/A/OpenCL",
                    macho::Command::Reexport,
                ),
            ],
            &["malloc", "vkCreateInstance"],
        );
        let fat = macho::fixture::universal(true, &[arm, intel]);
        assert_eq!(
            read_messages(&fat),
            [
                "rule 7: the core binary, built with no default features, loads \
                 `@rpath/libvulkan.1.dylib` from the Vulkan runtime; vendor GPU runtimes ship \
                 only in backend modules",
                "rule 7: the core binary, built with no default features, weakly loads \
                 `/opt/gpu/libgpu.dylib`, which is not a system library it may need; vendor GPU \
                 runtimes ship only in backend modules, and a system library that is not one goes \
                 in `MACOS_SYSTEM_LIBRARIES` or `MACOS_FRAMEWORKS` in \
                 xtask/src/check_deps/policy.rs, with a maintainer's review",
                "rule 7: the core binary, built with no default features, re-exports \
                 `/System/Library/Frameworks/OpenCL.framework/Versions/A/OpenCL`, which is not a \
                 system library it may need; vendor GPU runtimes ship only in backend modules, \
                 and a system library that is not one goes in `MACOS_SYSTEM_LIBRARIES` or \
                 `MACOS_FRAMEWORKS` in xtask/src/check_deps/policy.rs, with a maintainer's review",
                "rule 7: the core binary, built with no default features, has Vulkan runtime \
                 symbols in its undefined symbols: vkCreateInstance; vendor GPU runtimes ship \
                 only in backend modules",
            ]
        );
    }
}
