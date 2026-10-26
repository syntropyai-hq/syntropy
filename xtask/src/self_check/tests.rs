// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::*;

const RULES: &str = r#"
[paths]
workspace = ["Cargo.toml"]
no_crate = ["plan/**"]
[inputs]
"numerics/**" = ["kernels"]
[hardware]
status_creators = []
[job.lint]
runner = "hosted"
when = "always"
[job.build]
runner = "hosted"
when = "closure"
[job.hw-a]
runner = "lab"
when = "direct"
paths = ["crates/kernels/**", "crates/graph/src/blocks.rs", "numerics/**"]
[job.hw-b]
runner = "lab"
when = "direct"
paths = ["ci/images/b*/**"]
planned = "QUAL-E2-T08"
"#;

fn config() -> Config {
    Config::parse(RULES).unwrap()
}

fn workspace() -> Workspace {
    let members: BTreeMap<String, String> =
        [("crates/kernels", "kernels"), ("crates/graph", "graph")]
            .into_iter()
            .map(|(d, n)| (d.to_owned(), n.to_owned()))
            .collect();
    let dependents = BTreeMap::from([
        ("kernels".to_owned(), BTreeSet::from(["graph".to_owned()])),
        ("graph".to_owned(), BTreeSet::new()),
    ]);
    Workspace {
        members,
        dependents,
    }
}

fn files(list: &[&str]) -> Vec<String> {
    list.iter().map(|f| (*f).to_owned()).collect()
}

const TREE: &[&str] = &[
    "crates/kernels/src/lib.rs",
    "crates/graph/src/blocks.rs",
    "numerics/tolerances.toml",
    "plan/README.md",
];

#[test]
fn a_tree_that_meets_every_pattern_passes() {
    let out = check(&config(), &workspace(), &files(TREE));
    assert_eq!(out.problems, Vec::<String>::new());
    assert_eq!(out.diff.len(), 3);
    // The planned rule's pattern has no file yet: reported, not failed.
    assert_eq!(out.notices.len(), 1);
    assert!(out.notices[0].contains("hw-b") && out.notices[0].contains("QUAL-E2-T08"));
}

#[test]
fn a_removed_path_fails_and_names_the_rule_and_pattern() {
    let tree = files(&["crates/kernels/src/lib.rs", "numerics/tolerances.toml"]);
    let out = check(&config(), &workspace(), &tree);
    assert_eq!(out.problems.len(), 1, "{:?}", out.problems);
    assert!(out.problems[0].contains("rule `hw-a`"));
    assert!(out.problems[0].contains("crates/graph/src/blocks.rs"));
}

#[test]
fn a_moved_file_fails_the_same_way() {
    let tree = files(&[
        "crates/kernels/src/lib.rs",
        "crates/graph/src/blocks/mod.rs",
        "numerics/tolerances.toml",
    ]);
    let out = check(&config(), &workspace(), &tree);
    assert!(out.problems.iter().any(|p| p.contains("blocks.rs")));
}

#[test]
fn a_planned_rule_with_a_file_is_checked_like_any_other() {
    let mut tree = files(TREE);
    tree.push("ci/images/b1/Dockerfile".to_owned());
    let out = check(&config(), &workspace(), &tree);
    assert_eq!(out.problems, Vec::<String>::new());
    assert!(out.notices.is_empty());
    assert_eq!(out.diff.len(), 4);
}
