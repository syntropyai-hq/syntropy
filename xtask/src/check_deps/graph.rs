// SPDX-License-Identifier: MIT OR Apache-2.0

//! The dependency graph, read from `cargo metadata --format-version 1`.
//!
//! The graph is resolved with `--all-features` and without a platform filter, so an optional or
//! platform-specific dependency is an edge whether or not a given build enables it. Each edge
//! carries the kinds it is declared with: normal, build or dev.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Deserialize;

use super::policy::PREFIX;

/// One package in the resolved graph.
#[derive(Debug)]
pub struct Package {
    /// The package name, such as `syntropy-exec` or `serde`.
    pub name: String,
    /// Whether it is a workspace member.
    pub member: bool,
    /// The native library it declares with the `links` key, if any.
    pub links: Option<String>,
    pub targets: Vec<Target>,
    /// Its `[features]` table: each feature and the entries it enables, as written (`x`,
    /// `dep:x`, `x/y`, `x?/y`).
    pub features: BTreeMap<String, Vec<String>>,
    /// Its declared dependencies, every kind and platform.
    pub dependencies: Vec<Dependency>,
}

/// A dependency as the package declares it.
#[derive(Debug, Deserialize)]
pub struct Dependency {
    /// The package it names.
    pub name: String,
    /// The key the manifest gives it when that differs from `name`; feature entries such as
    /// `x/y` use the key.
    pub rename: Option<String>,
}

impl Dependency {
    /// The key feature entries use for this dependency: the rename, else the package name.
    pub fn key(&self) -> &str {
        self.rename.as_deref().unwrap_or(&self.name)
    }
}

/// One build target of a package, from `cargo metadata`.
#[derive(Debug, Deserialize)]
pub struct Target {
    pub name: String,
    /// `lib`, `bin`, `proc-macro`, `test`, `custom-build` and so on.
    pub kind: Vec<String>,
    /// The crate root.
    pub src_path: PathBuf,
}

/// The kinds a dependency is declared with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Kinds {
    pub normal: bool,
    pub build: bool,
    pub dev: bool,
}

impl Kinds {
    /// Normal or build: the edges that reach a built artefact.
    pub fn production(self) -> bool {
        self.normal || self.build
    }

    /// `normal`, `build`, `dev`, or several joined by `and`, for messages.
    pub fn describe(self) -> String {
        let names: Vec<&str> = [
            (self.normal, "normal"),
            (self.build, "build"),
            (self.dev, "dev"),
        ]
        .iter()
        .filter(|(set, _)| *set)
        .map(|(_, name)| *name)
        .collect();
        names.join(" and ")
    }
}

/// A dependency edge to package `to`.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub to: usize,
    pub kinds: Kinds,
}

/// The resolved graph. Packages are indexed by position.
#[derive(Debug)]
pub struct Graph {
    /// The workspace root, from `cargo metadata`.
    pub root: PathBuf,
    packages: Vec<Package>,
    edges: Vec<Vec<Edge>>,
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
    links: Option<String>,
    targets: Vec<Target>,
    #[serde(default)]
    features: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    dependencies: Vec<Dependency>,
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

impl Graph {
    /// Runs `cargo metadata` on `manifest` and reads its graph.
    pub fn load(manifest: &Path) -> Result<Graph, String> {
        let mut command = Command::new(crate::registry::cargo());
        command
            .args([
                "metadata",
                "--format-version",
                "1",
                "--locked",
                "--all-features",
            ])
            .arg("--manifest-path")
            .arg(manifest)
            .stderr(Stdio::inherit());
        let output = command
            .output()
            .map_err(|error| format!("could not start `cargo metadata`: {error}"))?;
        if !output.status.success() {
            return Err(format!("`cargo metadata` failed: {}", output.status));
        }
        let json = String::from_utf8(output.stdout)
            .map_err(|_| "`cargo metadata` printed invalid UTF-8".to_owned())?;
        Graph::from_metadata(&json)
    }

    /// Reads the graph from `cargo metadata --format-version 1` output.
    pub fn from_metadata(json: &str) -> Result<Graph, String> {
        let metadata: Metadata = serde_json::from_str(json)
            .map_err(|error| format!("cannot read `cargo metadata` output: {error}"))?;
        let resolve = metadata
            .resolve
            .ok_or("`cargo metadata` gave no resolved graph")?;
        let index: BTreeMap<&str, usize> = metadata
            .packages
            .iter()
            .enumerate()
            .map(|(at, package)| (package.id.as_str(), at))
            .collect();
        let lookup = |id: &str| {
            index
                .get(id)
                .copied()
                .ok_or_else(|| format!("`cargo metadata` names unknown package `{id}`"))
        };
        let mut edges = vec![Vec::new(); metadata.packages.len()];
        for node in &resolve.nodes {
            let from = lookup(&node.id)?;
            for dep in &node.deps {
                let mut kinds = Kinds::default();
                for kind in &dep.dep_kinds {
                    match kind.kind.as_deref() {
                        None => kinds.normal = true,
                        Some("build") => kinds.build = true,
                        Some("dev") => kinds.dev = true,
                        Some(other) => return Err(format!("unknown dependency kind `{other}`")),
                    }
                }
                edges[from].push(Edge {
                    to: lookup(&dep.pkg)?,
                    kinds,
                });
            }
        }
        let mut members = vec![false; metadata.packages.len()];
        for id in &metadata.workspace_members {
            members[lookup(id)?] = true;
        }
        let packages = metadata
            .packages
            .into_iter()
            .zip(members)
            .map(|(raw, member)| Package {
                name: raw.name,
                member,
                links: raw.links,
                targets: raw.targets,
                features: raw.features,
                dependencies: raw.dependencies,
            })
            .collect();
        Ok(Graph {
            root: metadata.workspace_root,
            packages,
            edges,
        })
    }

    pub fn package(&self, at: usize) -> &Package {
        &self.packages[at]
    }

    pub fn edges(&self, at: usize) -> &[Edge] {
        &self.edges[at]
    }

    /// The indices of the workspace members, in name order.
    pub fn members(&self) -> Vec<usize> {
        let mut members: Vec<usize> = (0..self.packages.len())
            .filter(|at| self.packages[*at].member)
            .collect();
        members.sort_by(|a, b| self.packages[*a].name.cmp(&self.packages[*b].name));
        members
    }

    /// How messages and rules name a package: a member without its `syntropy-` prefix, as
    /// RFC-0001 names it, and an external package by its own name.
    pub fn short(&self, at: usize) -> &str {
        let package = &self.packages[at];
        match package.name.strip_prefix(PREFIX) {
            Some(short) if package.member => short,
            _ => &package.name,
        }
    }

    /// The member RFC-0001 calls `short`, such as `exec`.
    pub fn member(&self, short: &str) -> Option<usize> {
        (0..self.packages.len()).find(|at| self.packages[*at].member && self.short(*at) == short)
    }

    /// `a → b → c`, naming each package as [`Graph::short`] does.
    pub fn path(&self, path: &[usize]) -> String {
        let names: Vec<&str> = path.iter().map(|at| self.short(*at)).collect();
        names.join(" → ")
    }

    /// Breadth-first search from `start` over the edges `follow` accepts. Returns the shortest
    /// path, `start` first, to each package `hit` accepts, nearest first. The search does not go
    /// past a hit, nor through a package `enter` refuses.
    pub fn search(
        &self,
        start: usize,
        follow: impl Fn(&Edge) -> bool,
        enter: impl Fn(usize) -> bool,
        hit: impl Fn(usize) -> bool,
    ) -> Vec<Vec<usize>> {
        let mut parent: Vec<Option<usize>> = vec![None; self.packages.len()];
        let mut seen = vec![false; self.packages.len()];
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut hits = Vec::new();
        while let Some(at) = queue.pop_front() {
            for edge in self.edges[at].iter().filter(|edge| follow(edge)) {
                if seen[edge.to] {
                    continue;
                }
                seen[edge.to] = true;
                parent[edge.to] = Some(at);
                if hit(edge.to) {
                    let mut path = vec![edge.to];
                    let mut step = at;
                    loop {
                        path.push(step);
                        match parent[step] {
                            Some(up) => step = up,
                            None => break,
                        }
                    }
                    path.reverse();
                    hits.push(path);
                } else if enter(edge.to) {
                    queue.push_back(edge.to);
                }
            }
        }
        hits
    }
}

#[cfg(test)]
impl Graph {
    /// A graph for tests. `members` are named as RFC-0001 names them (`exec`) and get the
    /// `syntropy-` prefix; `externals` keep their names. Each edge is `(from, to, kinds)`, with
    /// `kinds` a space-separated list of `normal`, `build` and `dev`, and the ends named the same
    /// way, members first.
    pub fn test(members: &[&str], externals: &[&str], edges: &[(&str, &str, &str)]) -> Graph {
        let mut packages: Vec<Package> = members
            .iter()
            .map(|short| Package {
                name: format!("{PREFIX}{short}"),
                member: true,
                links: None,
                targets: Vec::new(),
                features: BTreeMap::new(),
                dependencies: Vec::new(),
            })
            .collect();
        packages.extend(externals.iter().map(|name| Package {
            name: (*name).to_owned(),
            member: false,
            links: None,
            targets: Vec::new(),
            features: BTreeMap::new(),
            dependencies: Vec::new(),
        }));
        let mut graph = Graph {
            root: PathBuf::from("/fixture"),
            edges: vec![Vec::new(); packages.len()],
            packages,
        };
        let find = |graph: &Graph, name: &str| {
            graph
                .member(name)
                .or_else(|| {
                    (0..graph.packages.len())
                        .find(|at| !graph.packages[*at].member && graph.packages[*at].name == name)
                })
                .unwrap_or_else(|| panic!("no package `{name}`"))
        };
        for (from, to, kinds) in edges {
            let mut set = Kinds::default();
            for kind in kinds.split_whitespace() {
                match kind {
                    "normal" => set.normal = true,
                    "build" => set.build = true,
                    "dev" => set.dev = true,
                    other => panic!("unknown kind `{other}`"),
                }
            }
            let (from, to) = (find(&graph, from), find(&graph, to));
            graph.edges[from].push(Edge { to, kinds: set });
        }
        graph
    }

    /// Gives the member `name` a `[features]` table, each feature with its entries, and declared
    /// dependencies as `(key, package)`, the key being the package name unless renamed.
    pub fn with_features(
        mut self,
        name: &str,
        features: &[(&str, &[&str])],
        dependencies: &[(&str, &str)],
    ) -> Graph {
        let at = self
            .member(name)
            .unwrap_or_else(|| panic!("no member `{name}`"));
        let package = &mut self.packages[at];
        package.features = features
            .iter()
            .map(|(feature, entries)| {
                let entries = entries.iter().map(|e| (*e).to_owned()).collect();
                ((*feature).to_owned(), entries)
            })
            .collect();
        package.dependencies = dependencies
            .iter()
            .map(|(key, package)| Dependency {
                name: (*package).to_owned(),
                rename: (key != package).then(|| (*key).to_owned()),
            })
            .collect();
        self
    }

    /// Sets the `links` key of the package `name`.
    pub fn with_links(mut self, name: &str, links: &str) -> Graph {
        let at = self
            .member(name)
            .or_else(|| (0..self.packages.len()).find(|at| self.packages[*at].name == name))
            .unwrap_or_else(|| panic!("no package `{name}`"));
        self.packages[at].links = Some(links.to_owned());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_members_edges_and_kinds_from_metadata() {
        let json = r#"{
            "packages": [
                {"id": "a-id", "name": "syntropy-exec", "links": null,
                 "targets": [{"name": "syntropy_exec", "kind": ["lib"], "src_path": "/w/crates/exec/src/lib.rs"}]},
                {"id": "b-id", "name": "tokio", "links": null, "targets": []},
                {"id": "c-id", "name": "cc", "links": "ggml", "targets": []}
            ],
            "workspace_members": ["a-id"],
            "resolve": {"nodes": [
                {"id": "a-id", "deps": [
                    {"name": "tokio", "pkg": "b-id", "dep_kinds": [{"kind": null, "target": null}, {"kind": "dev", "target": null}]},
                    {"name": "cc", "pkg": "c-id", "dep_kinds": [{"kind": "build", "target": "cfg(unix)"}]}
                ]},
                {"id": "b-id", "deps": []},
                {"id": "c-id", "deps": []}
            ]},
            "workspace_root": "/w"
        }"#;
        let graph = Graph::from_metadata(json).unwrap();
        assert_eq!(graph.root, PathBuf::from("/w"));
        assert_eq!(graph.members(), [0]);
        assert_eq!(graph.short(0), "exec");
        assert_eq!(graph.short(1), "tokio");
        assert_eq!(graph.member("exec"), Some(0));
        assert_eq!(graph.member("tokio"), None);
        assert_eq!(graph.package(2).links.as_deref(), Some("ggml"));
        assert_eq!(graph.package(0).targets[0].kind, ["lib"]);
        let edges = graph.edges(0);
        assert_eq!(edges.len(), 2);
        assert_eq!(
            edges[0].kinds,
            Kinds {
                normal: true,
                build: false,
                dev: true
            }
        );
        assert_eq!(edges[0].kinds.describe(), "normal and dev");
        assert_eq!(edges[1].kinds.describe(), "build");
        assert!(edges[1].kinds.production());
    }

    #[test]
    fn metadata_without_a_resolve_or_with_unknown_ids_is_an_error() {
        let no_resolve =
            r#"{"packages": [], "workspace_members": [], "resolve": null, "workspace_root": "/w"}"#;
        assert!(
            Graph::from_metadata(no_resolve)
                .unwrap_err()
                .contains("no resolved graph")
        );
        let unknown = r#"{"packages": [], "workspace_members": ["x"],
            "resolve": {"nodes": []}, "workspace_root": "/w"}"#;
        assert!(
            Graph::from_metadata(unknown)
                .unwrap_err()
                .contains("unknown package `x`")
        );
        assert!(Graph::from_metadata("not json").is_err());
    }

    #[test]
    fn search_finds_the_shortest_path_and_stops_at_hits_and_refused_packages() {
        let graph = Graph::test(
            &["a", "b", "c", "wall"],
            &["x", "y"],
            &[
                ("a", "b", "normal"),
                ("b", "c", "normal"),
                ("c", "x", "normal"),
                ("a", "x", "dev"),
                ("a", "wall", "normal"),
                ("wall", "y", "normal"),
                ("x", "y", "normal"),
            ],
        );
        let a = graph.member("a").unwrap();
        let normal = |edge: &Edge| edge.kinds.normal;
        let externals = |at: usize| !graph.package(at).member;
        let paths = graph.search(a, normal, |at| graph.short(at) != "wall", externals);
        let shown: Vec<String> = paths.iter().map(|path| graph.path(path)).collect();
        // `x` through normal edges only, and `y` is never reached: `x` is a hit and `wall` is
        // not entered.
        assert_eq!(shown, ["a → b → c → x"]);
        let any = |_: &Edge| true;
        let shown: Vec<String> = graph
            .search(a, any, |_| true, |at| graph.short(at) == "y")
            .iter()
            .map(|path| graph.path(path))
            .collect();
        // Two paths of length 2; the first edge listed wins.
        assert_eq!(shown, ["a → x → y"]);
    }
}
