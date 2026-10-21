// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rules of RFC-0001 §4.2 that the dependency graph alone decides: the layer table, and rules
//! 1–6 and 9. Each check returns its findings; it prints nothing.

use super::graph::{Edge, Graph};
use super::layers::{self, Layers, Place};
use super::policy::{self, PREFIX};
use super::{Finding, Rule, list};

/// Every graph rule. Without a layer table, rules 2 and 9 cannot run, and the caller reports the
/// table's error instead.
pub fn check(graph: &Graph, layers: Option<&Layers>) -> Vec<Finding> {
    let mut findings = Vec::new();
    if let Some(layers) = layers {
        findings.extend(table(graph, layers));
        findings.extend(rule_2(graph, layers));
        findings.extend(rule_9(graph, layers));
    }
    findings.extend(rule_1(graph));
    findings.extend(rule_3(graph));
    findings.extend(rule_4(graph));
    findings.extend(rule_5(graph));
    findings.extend(rule_6(graph));
    findings
}

/// RFC-0001 §4.1: every member is named `syntropy-<name>` and sits in the layer table, and the
/// table lists only members.
fn table(graph: &Graph, layers: &Layers) -> Vec<Finding> {
    let mut findings = Vec::new();
    let members = graph.members();
    for &at in &members {
        let name = &graph.package(at).name;
        if !name.starts_with(PREFIX) {
            findings.push(Finding::new(
                Rule::Table,
                format!("workspace member `{name}` is not named `{PREFIX}<name>` (RFC-0001 §4.1)"),
            ));
        } else if layers.place(graph.short(at)).is_none() {
            findings.push(Finding::new(
                Rule::Table,
                format!(
                    "workspace member `{name}` is not in {}; add `{}` to its layer",
                    layers::PATH,
                    graph.short(at)
                ),
            ));
        }
    }
    for name in layers.names() {
        if graph.member(name).is_none() {
            findings.push(Finding::new(
                Rule::Table,
                format!(
                    "{} lists `{name}`, which is not a workspace member",
                    layers::PATH
                ),
            ));
        }
    }
    findings
}

/// Rule 1: `core` depends on no workspace crate, and its only external normal and build
/// dependencies are `serde`, `thiserror`, `smallvec` and `bitflags`.
fn rule_1(graph: &Graph) -> Vec<Finding> {
    let Some(core) = graph.member(policy::CORE) else {
        return Vec::new();
    };
    let mut findings = Vec::new();
    for edge in graph.edges(core).iter().filter(|e| e.kinds.production()) {
        let to = graph.package(edge.to);
        let kinds = edge.kinds.describe();
        if to.member {
            findings.push(Finding::new(
                Rule::Numbered(1),
                format!(
                    "`core` has a {kinds} dependency on the workspace crate `{}`; `core` depends \
                     on no workspace crate",
                    graph.short(edge.to)
                ),
            ));
        } else if !policy::CORE_EXTERNALS.contains(&to.name.as_str()) {
            findings.push(Finding::new(
                Rule::Numbered(1),
                format!(
                    "`core` has a {kinds} dependency on `{}`; its external dependencies are \
                     limited to {}",
                    to.name,
                    list(policy::CORE_EXTERNALS, "and")
                ),
            ));
        }
    }
    findings
}

/// Rule 2: every edge between members, dev edges included, goes to a lower layer. Every crate
/// except `core` may use `obs` and `config`, and every crate may use `core`. Edges into the
/// dev-only row are rule 9's; `core`'s normal and build edges are rule 1's. Tools have no
/// workspace edges either way.
fn rule_2(graph: &Graph, layers: &Layers) -> Vec<Finding> {
    let mut findings = Vec::new();
    for from in graph.members() {
        let from_name = graph.short(from);
        let Some(from_place) = layers.place(from_name) else {
            continue; // reported by `table`
        };
        for edge in graph.edges(from) {
            if !graph.package(edge.to).member {
                continue;
            }
            let to_name = graph.short(edge.to);
            let Some(to_place) = layers.place(to_name) else {
                continue;
            };
            let kinds = edge.kinds.describe();
            let message = match (from_place, to_place) {
                (Place::Tool, _) => format!(
                    "`{from_name}` is a tool outside the crate graph, but has a {kinds} \
                     dependency on `{to_name}`; tools depend on no workspace crate"
                ),
                (_, Place::Tool) => format!(
                    "`{from_name}` has a {kinds} dependency on `{to_name}`, a tool outside the \
                     crate graph; no crate depends on a tool"
                ),
                (_, Place::DevOnly) | (Place::DevOnly, _) => continue,
                (Place::Layer(from_layer), Place::Layer(to_layer)) => {
                    let rule_1_covers = from_name == policy::CORE && edge.kinds.production();
                    let to_core = to_name == policy::CORE;
                    let shared = policy::SHARED.contains(&to_name) && from_name != policy::CORE;
                    if rule_1_covers || to_core || shared || to_layer < from_layer {
                        continue;
                    }
                    format!(
                        "`{from_name}` ({}) has a {kinds} dependency on `{to_name}` ({}); a crate \
                         depends only on crates in lower layers",
                        layers::describe(from_place),
                        layers::describe(to_place)
                    )
                }
            };
            findings.push(Finding::new(Rule::Numbered(2), message));
        }
    }
    findings
}

/// Rule 3, with ADR-0012 §6: no compute crate reaches `net`, `cluster`, `store`, `api`, `daemon` or
/// `tokio` through normal dependencies. The search does not pass through another compute crate,
/// which gets its own finding, so each offending edge is reported once.
fn rule_3(graph: &Graph) -> Vec<Finding> {
    let mut forbidden = policy::COMPUTE_MUST_NOT_REACH.to_vec();
    forbidden.push(policy::TOKIO);
    let mut findings = Vec::new();
    for start in graph.members() {
        let name = graph.short(start);
        if !policy::is_compute(name) {
            continue;
        }
        let hits = graph.search(
            start,
            normal,
            |at| !(graph.package(at).member && policy::is_compute(graph.short(at))),
            |at| {
                let package = graph.package(at);
                if package.member {
                    policy::COMPUTE_MUST_NOT_REACH.contains(&graph.short(at))
                } else {
                    package.name == policy::TOKIO
                }
            },
        );
        for path in hits {
            let reached = graph.short(*path.last().expect("a path ends at its hit"));
            let adr = if reached == policy::TOKIO
                && (policy::is_provider(name) || policy::ADR_0012_NO_TOKIO.contains(&name))
            {
                " (and ADR-0012 §6)"
            } else {
                ""
            };
            findings.push(Finding::new(
                Rule::Numbered(3),
                format!(
                    "compute crate `{name}` reaches `{reached}` through normal dependencies: {}; \
                     compute crates never depend on {}{adr}",
                    graph.path(&path),
                    list(&forbidden, "or"),
                ),
            ));
        }
    }
    findings
}

/// Rule 4: `daemon` reaches no `kernels-*` crate, `kernels-loader` included, through normal
/// dependencies, so the daemon binary links no GPU provider.
fn rule_4(graph: &Graph) -> Vec<Finding> {
    let Some(daemon) = graph.member(policy::DAEMON) else {
        return Vec::new();
    };
    graph
        .search(
            daemon,
            normal,
            |_| true,
            |at| graph.package(at).member && policy::is_provider(graph.short(at)),
        )
        .into_iter()
        .map(|path| {
            let reached = graph.short(*path.last().expect("a path ends at its hit"));
            Finding::new(
                Rule::Numbered(4),
                format!(
                    "`daemon` reaches the provider crate `{reached}` through normal dependencies: \
                     {}; the daemon never links a `kernels-*` crate",
                    graph.path(&path)
                ),
            )
        })
        .collect()
}

/// Rule 5: `solver` reaches no I/O crate of the workspace, and no async runtime, socket,
/// filesystem or thread-pool crate, through normal dependencies.
fn rule_5(graph: &Graph) -> Vec<Finding> {
    let Some(solver) = graph.member(policy::SOLVER) else {
        return Vec::new();
    };
    graph
        .search(
            solver,
            normal,
            |_| true,
            |at| {
                let package = graph.package(at);
                if package.member {
                    policy::SOLVER_MUST_NOT_REACH.contains(&graph.short(at))
                } else {
                    policy::SOLVER_MUST_NOT_REACH_EXTERNAL.contains(&package.name.as_str())
                }
            },
        )
        .into_iter()
        .map(|path| {
            let reached = graph.short(*path.last().expect("a path ends at its hit"));
            Finding::new(
                Rule::Numbered(5),
                format!(
                    "`solver` reaches `{reached}` through normal dependencies: {}; the solver is \
                     pure, with no I/O, clocks or threads apart from HiGHS",
                    graph.path(&path)
                ),
            )
        })
        .collect()
}

/// Rule 6: each external system is reached by one workspace crate. A member other than the
/// system's own crate must not depend on one of its bindings, directly or through other external
/// packages, nor declare its native library with `links`. Reaching a binding through the
/// system's own crate is the intended path, so the search does not enter members.
fn rule_6(graph: &Graph) -> Vec<Finding> {
    let mut findings = Vec::new();
    for system in policy::SYSTEMS {
        let binding = |at: usize| {
            let package = graph.package(at);
            binds(system, &package.name, package.links.as_deref())
        };
        for start in graph.members() {
            let name = graph.short(start);
            if name == system.only_in {
                continue;
            }
            if let Some(links) = graph.package(start).links.as_deref()
                && system.links.contains(&links)
            {
                findings.push(Finding::new(
                    Rule::Numbered(6),
                    format!(
                        "`{name}` declares `links = \"{links}\"`, the native library of {}; only \
                         `{}` uses {}",
                        system.name, system.only_in, system.name
                    ),
                ));
            }
            let hits = graph.search(
                start,
                |_: &Edge| true,
                |at| !graph.package(at).member,
                |at| !graph.package(at).member && binding(at),
            );
            for path in hits {
                let reached = graph.short(*path.last().expect("a path ends at its hit"));
                findings.push(Finding::new(
                    Rule::Numbered(6),
                    format!(
                        "`{name}` reaches `{reached}`, a binding of {}: {}; only `{}` uses {}",
                        system.name,
                        graph.path(&path),
                        system.only_in,
                        system.name
                    ),
                ));
            }
        }
    }
    findings
}

/// Whether a package called `name`, declaring `links`, is a binding of `system`.
fn binds(system: &policy::System, name: &str, links: Option<&str>) -> bool {
    system.packages.contains(&name) || links.is_some_and(|links| system.links.contains(&links))
}

/// Rule 9: no production crate has a normal or build dependency on a dev-only crate (`testkit`,
/// and `bench` from the same row of RFC-0001 §4.1). Dev-only crates may depend on each other, and
/// edges from tools are rule 2's.
fn rule_9(graph: &Graph, layers: &Layers) -> Vec<Finding> {
    let mut findings = Vec::new();
    for from in graph.members() {
        let from_name = graph.short(from);
        if !matches!(layers.place(from_name), Some(Place::Layer(_))) {
            continue;
        }
        for edge in graph.edges(from).iter().filter(|e| e.kinds.production()) {
            let to_name = graph.short(edge.to);
            if graph.package(edge.to).member && layers.place(to_name) == Some(Place::DevOnly) {
                let mut kinds = edge.kinds;
                kinds.dev = false; // a dev edge alongside is allowed
                let kinds = kinds.describe();
                findings.push(Finding::new(
                    Rule::Numbered(9),
                    format!(
                        "`{from_name}` has a {kinds} dependency on `{to_name}`, which is dev-only; \
                         a production crate uses it only as a dev-dependency"
                    ),
                ));
            }
        }
    }
    findings
}

fn normal(edge: &Edge) -> bool {
    edge.kinds.normal
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule labels and messages of `findings`, sorted.
    fn shown(findings: Vec<Finding>) -> Vec<String> {
        let mut lines: Vec<String> = findings.iter().map(Finding::to_string).collect();
        lines.sort();
        lines
    }

    fn layers(text: &str) -> Layers {
        Layers::parse(text).unwrap()
    }

    const TABLE: &str = r#"
[layers]
0 = ["core", "obs", "config"]
1 = ["manifest", "kernels", "io", "net"]
2 = ["graph", "kernels-cpu", "kernels-loader", "tier", "solver", "store"]
3 = ["exec", "cluster"]
4 = ["sched", "api"]
5 = ["runner", "daemon"]
6 = ["cli"]
[outside]
dev-only = ["testkit", "bench"]
tools = ["xtask"]
"#;

    const ALL: &[&str] = &[
        "core",
        "obs",
        "config",
        "manifest",
        "kernels",
        "io",
        "net",
        "graph",
        "kernels-cpu",
        "kernels-loader",
        "tier",
        "solver",
        "store",
        "exec",
        "cluster",
        "sched",
        "api",
        "runner",
        "daemon",
        "cli",
        "testkit",
        "bench",
        "xtask",
    ];

    fn run(externals: &[&str], edges: &[(&str, &str, &str)]) -> Vec<String> {
        let graph = Graph::test(ALL, externals, edges);
        shown(check(&graph, Some(&layers(TABLE))))
    }

    #[test]
    fn the_designed_graph_passes() {
        // The "Depends on" column of RFC-0001 §4.1, for the crates in this table, plus the
        // patterns the rules allow.
        let edges = [
            ("core", "serde", "normal"),
            ("core", "bitflags", "normal"),
            ("core", "proptest", "dev"),
            ("core", "testkit", "dev"),
            ("obs", "core", "normal"),
            ("config", "core", "normal"),
            ("config", "obs", "normal"),
            ("manifest", "core", "normal"),
            ("kernels", "core", "normal"),
            ("kernels", "obs", "normal"),
            ("io", "core", "normal"),
            ("net", "core", "normal"),
            ("net", "tokio", "normal"),
            ("net", "iroh", "normal"),
            ("graph", "manifest", "normal"),
            ("graph", "kernels", "normal"),
            ("kernels-cpu", "kernels", "normal"),
            ("kernels-loader", "kernels", "normal"),
            ("tier", "manifest", "normal"),
            ("tier", "kernels", "normal"),
            ("tier", "io", "normal"),
            ("solver", "manifest", "normal"),
            ("solver", "highs", "normal"),
            ("store", "manifest", "normal"),
            ("store", "io", "normal"),
            ("store", "net", "normal"),
            ("exec", "graph", "normal"),
            ("exec", "kernels", "normal"),
            ("exec", "tier", "normal"),
            ("exec", "testkit", "dev"),
            ("cluster", "net", "normal"),
            ("cluster", "manifest", "normal"),
            ("cluster", "solver", "normal"),
            ("sched", "exec", "normal"),
            ("sched", "tier", "normal"),
            ("sched", "graph", "normal"),
            ("api", "cluster", "normal"),
            ("api", "solver", "normal"),
            ("api", "net", "normal"),
            ("runner", "kernels-cpu", "normal"),
            ("runner", "kernels-loader", "normal"),
            ("runner", "exec", "normal"),
            ("runner", "sched", "normal"),
            ("runner", "net", "normal"),
            ("runner", "store", "normal"),
            ("daemon", "kernels", "normal"),
            ("daemon", "net", "normal"),
            ("daemon", "store", "normal"),
            ("daemon", "cluster", "normal"),
            ("daemon", "api", "normal"),
            ("cli", "daemon", "normal"),
            ("cli", "runner", "normal"),
            ("cli", "config", "normal"),
            ("cli", "obs", "normal"),
            ("testkit", "exec", "normal"),
            ("testkit", "daemon", "normal"),
            ("bench", "cli", "normal"),
            ("bench", "testkit", "normal"),
            ("xtask", "serde_json", "normal"),
        ];
        let externals = [
            "serde",
            "bitflags",
            "proptest",
            "tokio",
            "iroh",
            "highs",
            "serde_json",
        ];
        assert_eq!(run(&externals, &edges), Vec::<String>::new());
    }

    #[test]
    fn the_table_must_match_the_members() {
        let graph = Graph::test(&["core", "extra"], &[], &[]);
        let found = shown(check(
            &graph,
            Some(&layers("[layers]\n0 = [\"core\", \"gone\"]\n")),
        ));
        assert_eq!(
            found,
            [
                "layer table: ci/layers.toml lists `gone`, which is not a workspace member",
                "layer table: workspace member `syntropy-extra` is not in ci/layers.toml; add \
                 `extra` to its layer",
            ]
        );
    }

    #[test]
    fn rule_1_limits_core_to_four_externals_and_no_workspace_crate() {
        let found = run(
            &["anyhow", "thiserror", "cc", "proptest"],
            &[
                ("core", "thiserror", "normal"),
                ("core", "anyhow", "normal"),
                ("core", "cc", "build"),
                ("core", "obs", "normal"),
                ("core", "proptest", "dev"),
            ],
        );
        assert_eq!(
            found,
            [
                "rule 1: `core` has a build dependency on `cc`; its external dependencies are \
                 limited to `serde`, `thiserror`, `smallvec` and `bitflags`",
                "rule 1: `core` has a normal dependency on `anyhow`; its external dependencies \
                 are limited to `serde`, `thiserror`, `smallvec` and `bitflags`",
                "rule 1: `core` has a normal dependency on the workspace crate `obs`; `core` \
                 depends on no workspace crate",
            ]
        );
    }

    #[test]
    fn rule_2_checks_every_member_edge_against_the_layers() {
        let found = run(
            &[],
            &[
                ("manifest", "solver", "normal"),
                ("graph", "tier", "normal"),
                ("kernels", "kernels-cpu", "dev"),
                ("core", "obs", "dev"),
                ("xtask", "core", "normal"),
                ("cli", "xtask", "build"),
            ],
        );
        assert_eq!(
            found,
            [
                "rule 2: `cli` has a build dependency on `xtask`, a tool outside the crate graph; \
                 no crate depends on a tool",
                "rule 2: `core` (layer 0) has a dev dependency on `obs` (layer 0); a crate depends \
                 only on crates in lower layers",
                "rule 2: `graph` (layer 2) has a normal dependency on `tier` (layer 2); a crate \
                 depends only on crates in lower layers",
                "rule 2: `kernels` (layer 1) has a dev dependency on `kernels-cpu` (layer 2); a \
                 crate depends only on crates in lower layers",
                "rule 2: `manifest` (layer 1) has a normal dependency on `solver` (layer 2); a \
                 crate depends only on crates in lower layers",
                "rule 2: `xtask` is a tool outside the crate graph, but has a normal dependency \
                 on `core`; tools depend on no workspace crate",
            ]
        );
    }

    #[test]
    fn rule_3_reports_each_offending_edge_once_with_its_path() {
        let found = run(
            &["tokio", "shim"],
            &[
                ("exec", "net", "normal"),
                ("sched", "exec", "normal"),
                ("kernels-cpu", "shim", "normal"),
                ("shim", "tokio", "normal"),
                ("graph", "tokio", "dev"),
                ("tier", "tokio", "normal"),
            ],
        );
        assert_eq!(
            found,
            [
                "rule 3: compute crate `exec` reaches `net` through normal dependencies: exec → \
                 net; compute crates never depend on `net`, `cluster`, `store`, `api`, \
                 `daemon` or `tokio`",
                "rule 3: compute crate `kernels-cpu` reaches `tokio` through normal dependencies: \
                 kernels-cpu → shim → tokio; compute crates never depend on `net`, `cluster`, \
                 `store`, `api`, `daemon` or `tokio` (and ADR-0012 §6)",
                "rule 3: compute crate `tier` reaches `tokio` through normal dependencies: tier → \
                 tokio; compute crates never depend on `net`, `cluster`, `store`, `api`, \
                 `daemon` or `tokio`",
            ]
        );
    }

    #[test]
    fn rule_4_follows_normal_edges_from_the_daemon_to_any_provider() {
        let found = run(
            &[],
            &[
                ("daemon", "kernels-loader", "normal"),
                ("daemon", "exec", "normal"),
                ("exec", "kernels-cpu", "normal"),
                ("daemon", "kernels", "normal"),
            ],
        );
        assert_eq!(
            found,
            [
                "rule 4: `daemon` reaches the provider crate `kernels-cpu` through normal \
                 dependencies: daemon → exec → kernels-cpu; the daemon never links a \
                 `kernels-*` crate",
                "rule 4: `daemon` reaches the provider crate `kernels-loader` through normal \
                 dependencies: daemon → kernels-loader; the daemon never links a `kernels-*` \
                 crate",
            ]
        );
        // A dev-dependency does not link into the daemon.
        assert_eq!(
            run(&[], &[("daemon", "kernels-cpu", "dev")]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn rule_5_keeps_io_runtimes_and_thread_pools_out_of_the_solver() {
        let found = run(
            &["tokio", "rayon", "highs"],
            &[
                ("solver", "net", "normal"),
                ("solver", "highs", "normal"),
                ("solver", "manifest", "normal"),
                ("manifest", "rayon", "normal"),
                ("solver", "tokio", "dev"),
            ],
        );
        assert_eq!(
            found,
            [
                "rule 5: `solver` reaches `net` through normal dependencies: solver → net; the \
                 solver is pure, with no I/O, clocks or threads apart from HiGHS",
                "rule 5: `solver` reaches `rayon` through normal dependencies: solver → manifest \
                 → rayon; the solver is pure, with no I/O, clocks or threads apart from HiGHS",
            ]
        );
    }

    #[test]
    fn rule_6_allows_each_system_in_its_own_crate_only() {
        let graph = Graph::test(
            ALL,
            &["iroh", "wrapper", "highs-sys", "ggml-build"],
            &[
                ("net", "iroh", "normal"),
                ("store", "net", "normal"),
                ("store", "iroh", "dev"),
                ("cluster", "wrapper", "normal"),
                ("wrapper", "iroh", "normal"),
                ("solver", "highs-sys", "build"),
                ("api", "highs-sys", "normal"),
                ("kernels-cpu", "ggml-build", "build"),
            ],
        )
        .with_links("ggml-build", "ggml")
        .with_links("kernels-loader", "ggml");
        assert_eq!(
            shown(check(&graph, Some(&layers(TABLE)))),
            [
                "rule 6: `api` reaches `highs-sys`, a binding of HiGHS: api → highs-sys; only \
                 `solver` uses HiGHS",
                "rule 6: `cluster` reaches `iroh`, a binding of iroh: cluster → wrapper → iroh; \
                 only `net` uses iroh",
                "rule 6: `kernels-cpu` reaches `ggml-build`, a binding of ggml: kernels-cpu → \
                 ggml-build; only `kernels-ggml` uses ggml",
                "rule 6: `kernels-loader` declares `links = \"ggml\"`, the native library of \
                 ggml; only `kernels-ggml` uses ggml",
                "rule 6: `store` reaches `iroh`, a binding of iroh: store → iroh; only `net` uses \
                 iroh",
            ]
        );
    }

    #[test]
    fn rule_9_allows_testkit_only_as_a_dev_dependency() {
        let found = run(
            &[],
            &[
                ("runner", "testkit", "normal dev"),
                ("store", "bench", "build"),
                ("api", "testkit", "dev"),
                ("bench", "testkit", "normal"),
                ("testkit", "runner", "normal"),
            ],
        );
        assert_eq!(
            found,
            [
                "rule 9: `runner` has a normal dependency on `testkit`, which is dev-only; a \
                 production crate uses it only as a dev-dependency",
                "rule 9: `store` has a build dependency on `bench`, which is dev-only; a \
                 production crate uses it only as a dev-dependency",
            ]
        );
    }

    #[test]
    fn without_a_table_the_other_rules_still_run() {
        let graph = Graph::test(&["core", "exec", "net"], &[], &[("exec", "net", "normal")]);
        let found = shown(check(&graph, None));
        assert_eq!(found.len(), 1);
        assert!(found[0].starts_with("rule 3: "), "{found:?}");
    }
}
