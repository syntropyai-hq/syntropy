// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask notices`: generates `THIRD-PARTY-NOTICES` (ADR-0018 §6, RFC-0062 §6.3, §12.2).
//!
//! The notices list every package `cargo metadata` resolves for the release build, and every
//! entry of `third_party/licences.toml`. The release build is the closure of the release crates
//! (`cli` and the backend providers, as `check-deps` rule 10 defines them) over normal and build
//! edges, with every feature on and for every platform, so the list covers any release artefact.
//! Dev-dependencies are left out; workspace crates are Syntropy's own. For each component the
//! notices give its licence and where it comes from, and they carry:
//!
//! - for each MPL-2.0 component, where its source code is available (MPL-2.0 §3.2): its crates.io
//!   page and download, or its git repository and commit. An MPL-2.0 package from a path has no
//!   published source to point to, and ADR-0018 §3 allows MPL-2.0 only unmodified, so it fails;
//! - every `NOTICE` file a package ships (Apache-2.0 §4(d));
//! - every licence file a package ships, and each manifest entry's notice file.
//!
//! Generation fails when a package declares no licence, when a licence is not an SPDX
//! expression, or when the manifest has a problem `cargo xtask third-party` reports. In `lint`
//! the command runs without `--out`, so it checks that generation succeeds and writes nothing.
//! With `--out <file>` it writes the notices there; the per-artefact files of a release are
//! QUAL-E16-T13's.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Deserialize;

use super::expr::Expr;
use super::manifest::{self, Entry};
use super::{Policy, count};
use crate::check_deps::policy::{PREFIX, RELEASE_CRATES};

/// The crates.io index, as `cargo metadata` names it in a package's `source`.
const CRATES_IO: &[&str] = &[
    "registry+https://github.com/rust-lang/crates.io-index",
    "sparse+https://index.crates.io/",
];

/// Where a package comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    CratesIo,
    /// Another registry, by its index URL.
    Registry(String),
    /// A git repository and the commit Cargo.lock pins.
    Git {
        url: String,
        commit: String,
    },
    /// A local path: not published.
    Path,
}

impl Origin {
    fn from_source(source: Option<&str>) -> Origin {
        let Some(source) = source else {
            return Origin::Path;
        };
        if CRATES_IO.contains(&source) {
            return Origin::CratesIo;
        }
        if let Some(git) = source.strip_prefix("git+") {
            let (url, commit) = git.split_once('#').unwrap_or((git, ""));
            let url = url.split_once('?').map_or(url, |(url, _)| url);
            return Origin::Git {
                url: url.to_owned(),
                commit: commit.to_owned(),
            };
        }
        let index = source
            .strip_prefix("registry+")
            .or_else(|| source.strip_prefix("sparse+"))
            .unwrap_or(source);
        Origin::Registry(index.to_owned())
    }

    /// Where the source code of `name` `version` is available, one location per line.
    fn locations(&self, name: &str, version: &str) -> Vec<String> {
        match self {
            Origin::CratesIo => vec![
                format!("https://crates.io/crates/{name}/{version}"),
                format!("https://static.crates.io/crates/{name}/{name}-{version}.crate"),
            ],
            Origin::Registry(index) => vec![format!("{name} {version} in the registry {index}")],
            Origin::Git { url, commit } => vec![format!("{url} at commit {commit}")],
            Origin::Path => Vec::new(),
        }
    }
}

/// One third-party package of the release build.
#[derive(Debug)]
pub struct Component {
    pub name: String,
    pub version: String,
    /// The SPDX expression of its `license` field.
    pub license: Option<String>,
    /// Its `license-file`, relative to [`Component::dir`].
    pub license_file: Option<String>,
    pub repository: Option<String>,
    pub origin: Origin,
    /// The package's directory, where its licence and NOTICE files are.
    pub dir: PathBuf,
}

impl Component {
    fn label(&self) -> String {
        format!("{} {}", self.name, self.version)
    }

    /// The licence as the notices print it.
    fn licence(&self) -> String {
        match (&self.license, &self.license_file) {
            (Some(license), _) => license.clone(),
            (None, Some(file)) => format!("the terms in its `{file}`"),
            (None, None) => "none declared".to_owned(),
        }
    }

    /// Whether the notices must say where its source is (MPL-2.0 §3.2).
    fn is_mpl(&self) -> bool {
        self.license
            .as_deref()
            .and_then(|l| Expr::parse(l).ok())
            .is_some_and(|expr| expr.mentions("MPL-2.0"))
    }
}

#[derive(Deserialize)]
struct Metadata {
    packages: Vec<RawPackage>,
    workspace_members: Vec<String>,
    resolve: Option<RawResolve>,
    workspace_root: PathBuf,
}

#[derive(Deserialize)]
struct RawPackage {
    id: String,
    name: String,
    version: String,
    license: Option<String>,
    license_file: Option<String>,
    repository: Option<String>,
    source: Option<String>,
    manifest_path: PathBuf,
}

#[derive(Deserialize)]
struct RawResolve {
    nodes: Vec<RawNode>,
}

#[derive(Deserialize)]
struct RawNode {
    id: String,
    deps: Vec<RawDep>,
}

#[derive(Deserialize)]
struct RawDep {
    pkg: String,
    dep_kinds: Vec<RawKind>,
}

#[derive(Deserialize)]
struct RawKind {
    kind: Option<String>,
}

/// The release build of a workspace: its root, the release crates found, and its third-party
/// packages sorted by name and version.
#[derive(Debug)]
pub struct Release {
    pub root: PathBuf,
    pub crates: Vec<String>,
    pub components: Vec<Component>,
}

/// Reads the release build from `cargo metadata --format-version 1` output.
pub fn from_metadata(json: &str) -> Result<Release, String> {
    let metadata: Metadata = serde_json::from_str(json)
        .map_err(|error| format!("cannot read `cargo metadata` output: {error}"))?;
    let resolve = metadata
        .resolve
        .ok_or("`cargo metadata` gave no resolved graph")?;
    let by_id: BTreeMap<&str, &RawPackage> = metadata
        .packages
        .iter()
        .map(|package| (package.id.as_str(), package))
        .collect();
    let members: BTreeSet<&str> = metadata
        .workspace_members
        .iter()
        .map(String::as_str)
        .collect();
    let edges: BTreeMap<&str, Vec<&str>> = resolve
        .nodes
        .iter()
        .map(|node| {
            let production = node
                .deps
                .iter()
                .filter(|dep| {
                    dep.dep_kinds
                        .iter()
                        .any(|kind| kind.kind.as_deref() != Some("dev"))
                })
                .map(|dep| dep.pkg.as_str())
                .collect();
            (node.id.as_str(), production)
        })
        .collect();
    let mut crates = Vec::new();
    let mut queue = VecDeque::new();
    for short in RELEASE_CRATES {
        let name = format!("{PREFIX}{short}");
        if let Some(id) = members
            .iter()
            .find(|id| by_id.get(*id).is_some_and(|p| p.name == name))
        {
            crates.push((*short).to_owned());
            queue.push_back(*id);
        }
    }
    if crates.is_empty() {
        return Err(format!(
            "the workspace has none of the release crates ({})",
            RELEASE_CRATES.join(", ")
        ));
    }
    let mut seen: BTreeSet<&str> = queue.iter().copied().collect();
    while let Some(id) = queue.pop_front() {
        for dep in edges.get(id).into_iter().flatten() {
            if seen.insert(*dep) {
                queue.push_back(*dep);
            }
        }
    }
    let mut components = Vec::new();
    for id in seen {
        if members.contains(id) {
            continue;
        }
        let package = by_id
            .get(id)
            .ok_or_else(|| format!("`cargo metadata` names unknown package `{id}`"))?;
        components.push(Component {
            name: package.name.clone(),
            version: package.version.clone(),
            license: package.license.clone(),
            license_file: package.license_file.clone(),
            repository: package.repository.clone(),
            origin: Origin::from_source(package.source.as_deref()),
            dir: package
                .manifest_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default(),
        });
    }
    components.sort_by(|a, b| (&a.name, &a.version).cmp(&(&b.name, &b.version)));
    Ok(Release {
        root: metadata.workspace_root,
        crates,
        components,
    })
}

/// Runs `cargo metadata` on `manifest`, from the workspace's directory so its
/// `.cargo/config.toml` applies, and reads the release build.
pub fn load(manifest: &Path) -> Result<Release, String> {
    let manifest = std::fs::canonicalize(manifest)
        .map_err(|error| format!("no manifest at {}: {error}", manifest.display()))?;
    let dir = manifest.parent().unwrap_or(Path::new("."));
    let output = Command::new(crate::registry::cargo())
        .args([
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--all-features",
        ])
        .arg("--manifest-path")
        .arg(&manifest)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("could not start `cargo metadata`: {error}"))?;
    if !output.status.success() {
        return Err(format!("`cargo metadata` failed: {}", output.status));
    }
    let json = String::from_utf8(output.stdout)
        .map_err(|_| "`cargo metadata` printed invalid UTF-8".to_owned())?;
    from_metadata(&json)
}

/// What stops the notices being complete and correct.
pub fn problems(components: &[Component]) -> Vec<String> {
    let mut problems = Vec::new();
    for component in components {
        let label = component.label();
        match (&component.license, &component.license_file) {
            (None, None) => problems.push(format!(
                "`{label}` declares no licence, so the notices cannot list it (ADR-0018 §6)"
            )),
            (Some(license), _) => {
                if let Err(error) = Expr::parse(license) {
                    problems.push(format!(
                        "`{label}` has the licence `{license}`, which is not an SPDX expression: \
                         {error}"
                    ));
                }
            }
            (None, Some(_)) => {}
        }
        if component.is_mpl() && component.origin == Origin::Path {
            problems.push(format!(
                "`{label}` is under MPL-2.0 and comes from a local path, so it has no published \
                 source to point to (MPL-2.0 §3.2); ADR-0018 §3 allows MPL-2.0 only for unmodified \
                 dependencies"
            ));
        }
    }
    problems
}

/// A file a package or the manifest carries into the notices.
struct Carried {
    /// Whose file, such as `apache-crate 1.0.0: NOTICE`.
    heading: String,
    text: String,
}

/// The licence and NOTICE files in a package's directory, by kind, each sorted by name.
fn package_files(component: &Component) -> (Vec<Carried>, Vec<Carried>) {
    let mut names: Vec<String> = std::fs::read_dir(&component.dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    if let Some(file) = &component.license_file
        && !names.contains(file)
    {
        names.push(file.clone());
    }
    names.sort();
    let mut licences = Vec::new();
    let mut notices = Vec::new();
    for name in names {
        let upper = name.to_ascii_uppercase();
        let is_notice = upper.starts_with("NOTICE");
        let is_licence = ["LICENSE", "LICENCE", "COPYING", "UNLICENSE"]
            .iter()
            .any(|prefix| upper.starts_with(prefix))
            || component.license_file.as_deref() == Some(name.as_str());
        if !is_notice && !is_licence {
            continue;
        }
        let text = match std::fs::read(component.dir.join(&name)) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(error) => format!("(cannot read this file: {error})"),
        };
        let carried = Carried {
            heading: format!("{}: {name}", component.label()),
            text,
        };
        if is_notice {
            notices.push(carried);
        } else {
            licences.push(carried);
        }
    }
    (licences, notices)
}

/// Counts for the summary line.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub crates: usize,
    pub files: usize,
    pub mpl: usize,
    pub notice_files: usize,
}

/// The text of `THIRD-PARTY-NOTICES` for `release` and the manifest `entries` under `root`.
pub fn render(release: &Release, entries: &[Entry], root: &Path) -> (String, Counts) {
    let mut out = String::new();
    let mut counts = Counts {
        crates: release.components.len(),
        files: entries.len(),
        ..Counts::default()
    };
    let _ = writeln!(out, "THIRD-PARTY-NOTICES");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Syntropy is licensed under MIT OR Apache-2.0. Its release artefacts also contain the\n\
         third-party components below, each under its own licence. `cargo xtask notices`\n\
         generates this file from the resolved dependency graph of the release crates ({})\n\
         and from {} (ADR-0018 §6). Do not edit it by hand.\n",
        release.crates.join(", "),
        manifest::PATH
    );

    section(
        &mut out,
        &format!("Rust crates ({})", release.components.len()),
    );
    for component in &release.components {
        let _ = writeln!(out, "{}", component.label());
        let _ = writeln!(out, "  Licence: {}", component.licence());
        match component
            .origin
            .locations(&component.name, &component.version)
            .first()
        {
            Some(location) => {
                let _ = writeln!(out, "  Source: {location}");
            }
            None => {
                let _ = writeln!(out, "  Source: a local path, not published");
            }
        }
        if let Some(repository) = &component.repository {
            let _ = writeln!(out, "  Repository: {repository}");
        }
        let _ = writeln!(out);
    }

    section(&mut out, &format!("Vendored files ({})", entries.len()));
    if entries.is_empty() {
        let _ = writeln!(out, "None.\n");
    }
    for entry in entries {
        let _ = writeln!(out, "{}", entry.path);
        let _ = writeln!(out, "  Licence: {}", entry.license);
        let _ = writeln!(
            out,
            "  Upstream: {}, version {}",
            entry.upstream, entry.version
        );
        let _ = writeln!(out, "  Notice: {}", entry.notice);
        let _ = writeln!(out);
    }

    section(&mut out, "MPL-2.0 source code");
    let mpl: Vec<&Component> = release.components.iter().filter(|c| c.is_mpl()).collect();
    counts.mpl = mpl.len();
    if mpl.is_empty() {
        let _ = writeln!(out, "No component above is under MPL-2.0.\n");
    } else {
        let _ = writeln!(
            out,
            "These components are under MPL-2.0. As MPL-2.0 §3.2 requires, their source code\n\
             is available at the locations below.\n"
        );
    }
    for component in mpl {
        let _ = writeln!(out, "{}", component.label());
        for location in component
            .origin
            .locations(&component.name, &component.version)
        {
            let _ = writeln!(out, "  {location}");
        }
        if let Some(repository) = &component.repository {
            let _ = writeln!(out, "  Repository: {repository}");
        }
        let _ = writeln!(out);
    }

    let mut licences = Vec::new();
    let mut notices = Vec::new();
    let mut without = Vec::new();
    for component in &release.components {
        let (found_licences, found_notices) = package_files(component);
        if found_licences.is_empty() {
            without.push(component.label());
        }
        licences.extend(found_licences);
        notices.extend(found_notices);
    }
    // One copy of each manifest notice file, naming the vendored files it covers.
    let mut by_notice: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for entry in entries {
        by_notice
            .entry(&entry.notice)
            .or_default()
            .push(&entry.path);
    }
    for (notice, paths) in by_notice {
        let text = match std::fs::read(root.join(notice)) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(error) => format!("(cannot read this file: {error})"),
        };
        licences.push(Carried {
            heading: format!("{notice} (for {})", paths.join(", ")),
            text,
        });
    }
    counts.notice_files = notices.len();

    section(&mut out, "NOTICE files");
    if notices.is_empty() {
        let _ = writeln!(out, "No component above ships a NOTICE file.\n");
    } else {
        let _ = writeln!(
            out,
            "Carried from the components that ship them, as Apache-2.0 §4(d) requires.\n"
        );
    }
    carry(&mut out, &notices);

    section(&mut out, "Licence texts");
    if !without.is_empty() {
        let _ = writeln!(
            out,
            "These packages ship no licence file; their licence is the one listed above:\n{}\n",
            without.join(", ")
        );
    }
    carry(&mut out, &licences);
    (out, counts)
}

fn section(out: &mut String, title: &str) {
    let _ = writeln!(out, "== {title} ==\n");
}

fn carry(out: &mut String, files: &[Carried]) {
    for file in files {
        let _ = writeln!(out, "-- {} --\n", file.heading);
        let _ = writeln!(out, "{}", file.text.trim_end());
        let _ = writeln!(out);
    }
}

/// Generates the notices for the workspace of `manifest`, with the manifest of vendored files
/// checked against the policy of `checkout`. Writes them to `out` when given; either way writes
/// the problems and the verdict to `log`. Returns whether generation succeeded.
pub fn run(
    checkout: &Path,
    manifest_path: &Path,
    out: Option<&Path>,
    log: &mut dyn Write,
) -> io::Result<bool> {
    writeln!(log, "notices: {}", manifest_path.display())?;
    log.flush()?;
    let release = match load(manifest_path) {
        Ok(release) => release,
        Err(error) => {
            writeln!(log, "notices: FAIL: {error}")?;
            return Ok(false);
        }
    };
    let mut found = problems(&release.components);
    let entries = match Policy::load(checkout) {
        Ok(policy) => {
            let (manifest_problems, _) = manifest::check(&release.root, &policy);
            found.extend(manifest_problems);
            manifest::load(&release.root).unwrap_or_default()
        }
        Err(error) => {
            found.push(error);
            Vec::new()
        }
    };
    for problem in &found {
        writeln!(log, "notices: {problem}")?;
    }
    if !found.is_empty() {
        writeln!(
            log,
            "notices: FAIL: {}; the notices would be incomplete (ADR-0018 §6)",
            count(found.len(), "problem")
        )?;
        return Ok(false);
    }
    let (text, counts) = render(&release, &entries, &release.root);
    if let Some(out) = out
        && let Err(error) = std::fs::write(out, &text)
    {
        writeln!(
            log,
            "notices: FAIL: cannot write {}: {error}",
            out.display()
        )?;
        return Ok(false);
    }
    writeln!(
        log,
        "notices: PASS: {} of the release build ({}) and {}; {}; {} carried{}",
        count(counts.crates, "crate"),
        release.crates.join(", "),
        count(counts.files, "vendored file"),
        count(counts.mpl, "MPL-2.0 source pointer"),
        count(counts.notice_files, "NOTICE file"),
        match out {
            Some(out) => format!("; written to {}", out.display()),
            None => "; nothing written (pass --out <file> to write them)".to_owned(),
        }
    )?;
    log.flush()?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `cargo metadata` document: `cli` and `bench` in the workspace; `cli` uses `a` (normal),
    /// `b` (build) and `d` (dev only); `a` uses `c`; `bench` uses `e`, which no release crate
    /// reaches.
    const METADATA: &str = r#"{
      "workspace_root": "/ws",
      "workspace_members": ["cli-id", "bench-id"],
      "packages": [
        {"id": "cli-id", "name": "syntropy-cli", "version": "0.0.0", "license": "MIT OR Apache-2.0",
         "license_file": null, "repository": null, "source": null, "manifest_path": "/ws/crates/cli/Cargo.toml"},
        {"id": "bench-id", "name": "syntropy-bench", "version": "0.0.0", "license": "MIT OR Apache-2.0",
         "license_file": null, "repository": null, "source": null, "manifest_path": "/ws/crates/bench/Cargo.toml"},
        {"id": "a-id", "name": "a", "version": "1.0.0", "license": "MPL-2.0", "license_file": null,
         "repository": "https://example.org/a", "source": "registry+https://github.com/rust-lang/crates.io-index",
         "manifest_path": "/reg/a-1.0.0/Cargo.toml"},
        {"id": "b-id", "name": "b", "version": "2.0.0", "license": "Apache-2.0", "license_file": null,
         "repository": null, "source": "git+https://example.org/b?rev=abc#abc123", "manifest_path": "/git/b/Cargo.toml"},
        {"id": "c-id", "name": "c", "version": "0.1.0", "license": null, "license_file": "COPYING",
         "repository": null, "source": "sparse+https://index.crates.io/", "manifest_path": "/reg/c-0.1.0/Cargo.toml"},
        {"id": "d-id", "name": "d", "version": "1.0.0", "license": "GPL-3.0-only", "license_file": null,
         "repository": null, "source": "registry+https://github.com/rust-lang/crates.io-index", "manifest_path": "/reg/d/Cargo.toml"},
        {"id": "e-id", "name": "e", "version": "1.0.0", "license": "MIT", "license_file": null,
         "repository": null, "source": "registry+https://github.com/rust-lang/crates.io-index", "manifest_path": "/reg/e/Cargo.toml"}
      ],
      "resolve": {"nodes": [
        {"id": "cli-id", "deps": [
          {"pkg": "a-id", "dep_kinds": [{"kind": null}]},
          {"pkg": "b-id", "dep_kinds": [{"kind": "build"}]},
          {"pkg": "d-id", "dep_kinds": [{"kind": "dev"}]}
        ]},
        {"id": "bench-id", "deps": [{"pkg": "e-id", "dep_kinds": [{"kind": null}]}]},
        {"id": "a-id", "deps": [{"pkg": "c-id", "dep_kinds": [{"kind": null}]}]},
        {"id": "b-id", "deps": []}, {"id": "c-id", "deps": []},
        {"id": "d-id", "deps": []}, {"id": "e-id", "deps": []}
      ]}
    }"#;

    #[test]
    fn the_release_build_is_the_normal_and_build_closure_of_the_release_crates() {
        let release = from_metadata(METADATA).unwrap();
        assert_eq!(release.crates, ["cli"]);
        let names: Vec<&str> = release.components.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"]);
        assert_eq!(release.root, PathBuf::from("/ws"));
    }

    #[test]
    fn origins_come_from_the_source_field() {
        assert_eq!(
            Origin::from_source(Some(
                "registry+https://github.com/rust-lang/crates.io-index"
            )),
            Origin::CratesIo
        );
        assert_eq!(
            Origin::from_source(Some("sparse+https://index.crates.io/")),
            Origin::CratesIo
        );
        assert_eq!(
            Origin::from_source(Some("git+https://example.org/b?rev=abc#abc123")),
            Origin::Git {
                url: "https://example.org/b".to_owned(),
                commit: "abc123".to_owned()
            }
        );
        assert_eq!(
            Origin::from_source(Some("sparse+https://registry.example.invalid/index/")),
            Origin::Registry("https://registry.example.invalid/index/".to_owned())
        );
        assert_eq!(Origin::from_source(None), Origin::Path);
    }

    #[test]
    fn mpl_components_get_their_source_location() {
        let release = from_metadata(METADATA).unwrap();
        let (text, counts) = render(&release, &[], Path::new("/ws"));
        assert_eq!(counts.mpl, 1);
        let mpl = text.split("== MPL-2.0 source code ==").nth(1).unwrap();
        assert!(
            mpl.contains("a 1.0.0\n  https://crates.io/crates/a/1.0.0\n"),
            "{mpl}"
        );
        assert!(
            mpl.contains("https://static.crates.io/crates/a/a-1.0.0.crate"),
            "{mpl}"
        );
        assert!(mpl.contains("Repository: https://example.org/a"), "{mpl}");
        assert!(
            text.contains(
                "b 2.0.0\n  Licence: Apache-2.0\n  Source: https://example.org/b at commit abc123\n"
            ),
            "{text}"
        );
        assert!(
            text.contains("c 0.1.0\n  Licence: the terms in its `COPYING`\n"),
            "{text}"
        );
    }

    #[test]
    fn a_package_without_a_licence_or_mpl_from_a_path_is_a_problem() {
        let component = |license: Option<&str>, origin: Origin| Component {
            name: "x".to_owned(),
            version: "1.0.0".to_owned(),
            license: license.map(str::to_owned),
            license_file: None,
            repository: None,
            origin,
            dir: PathBuf::new(),
        };
        assert_eq!(
            problems(&[component(Some("MIT"), Origin::Path)]),
            Vec::<String>::new()
        );
        assert_eq!(
            problems(&[component(Some("MPL-2.0"), Origin::CratesIo)]),
            Vec::<String>::new()
        );
        let found = problems(&[
            component(None, Origin::CratesIo),
            component(Some("MIT OR"), Origin::CratesIo),
            component(Some("MPL-2.0"), Origin::Path),
        ]);
        assert_eq!(found.len(), 3, "{found:#?}");
        assert!(found[0].contains("declares no licence"));
        assert!(found[1].contains("not an SPDX expression"));
        assert!(found[2].contains("no published source"));
    }

    #[test]
    fn a_workspace_without_a_release_crate_is_an_error() {
        let json = METADATA.replace("syntropy-cli", "syntropy-other");
        assert!(
            from_metadata(&json)
                .unwrap_err()
                .contains("none of the release crates")
        );
    }
}
