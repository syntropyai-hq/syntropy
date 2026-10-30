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

/// The real rules, then `tweak` applied to the answer for a diff of `len` paths.
fn tweaked(
    len: usize,
    tweak: impl Fn(&mut changes::Affected) + 'static,
) -> impl Fn(&Config, &Workspace, &[String], &BTreeSet<String>) -> changes::Affected {
    move |config, workspace, paths, lock| {
        let mut answer = changes::affected(config, workspace, paths, lock);
        if paths.len() == len {
            tweak(&mut answer);
        }
        answer
    }
}

fn run(apply: &Rules) -> Vec<String> {
    check_with(&config(), &workspace(), &files(TREE), apply).problems
}

#[test]
fn a_file_that_does_not_require_its_job_fails_and_names_the_rule_and_pattern() {
    let problems = run(&tweaked(1, |a| a.hw_required.retain(|j| j != "hw-a")));
    assert!(
        problems.iter().any(|p| p.contains("rule `hw-a`")
            && p.contains("does not select `hw-a`")
            && p.contains("crates/kernels/**")),
        "{problems:?}"
    );
}

#[test]
fn a_job_the_combined_diff_does_not_require_fails_and_names_it() {
    let problems = run(&tweaked(3, |a| a.hw_required.clear()));
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("rule `hw-a`") && problems[0].contains("combined diff"));
}

#[test]
fn a_job_the_combined_diff_requires_and_no_rule_expects_fails_and_names_it() {
    let problems = run(&tweaked(3, |a| a.hw_required.push("hw-x".to_owned())));
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("`hw-x`") && problems[0].contains("no checked rule expects"));
}

#[test]
fn a_hosted_job_that_did_not_run_fails_and_names_it() {
    let problems = run(&tweaked(3, |a| a.jobs.retain(|j| j != "build")));
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("`build`") && problems[0].contains("did not run"));
}

#[test]
fn a_hosted_job_that_ran_and_no_rule_expects_fails_and_names_it() {
    let problems = run(&tweaked(3, |a| a.jobs.push("extra".to_owned())));
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("`extra`") && problems[0].contains("no rule expects"));
}

const HOSTED_RULES: &str = r#"
[job.tool-check]
runner = "hosted"
when = "direct"
paths = ["ci/tools.toml", "xtask/src/tools.rs"]
[job.tool-later]
runner = "hosted"
when = "direct"
paths = ["ci/later.toml"]
planned = "QUAL-E9-T99"
"#;

fn hosted_config() -> Config {
    Config::parse(&format!("{RULES}{HOSTED_RULES}")).unwrap()
}

fn hosted_tree() -> Vec<String> {
    let mut tree = files(TREE);
    tree.extend(files(&[
        "ci/tools.toml",
        "xtask/src/tools.rs",
        "ci/later.toml",
    ]));
    tree
}

#[test]
fn a_hosted_direct_rule_passes_when_its_files_run_its_job() {
    let out = check(&hosted_config(), &workspace(), &hosted_tree());
    assert_eq!(out.problems, Vec::<String>::new());
    assert!(
        out.diff
            .iter()
            .any(|(f, r)| f == "xtask/src/tools.rs" && r == "tool-check")
    );
    assert!(
        out.diff
            .iter()
            .any(|(f, r)| f == "ci/later.toml" && r == "tool-later")
    );
}

#[test]
fn a_hosted_direct_rule_whose_file_does_not_run_its_job_fails() {
    // The engine no longer runs `tool-check` for a lone file: as if its pattern were dropped.
    let apply = tweaked(1, |a| a.jobs.retain(|j| j != "tool-check"));
    let problems = check_with(&hosted_config(), &workspace(), &hosted_tree(), &apply).problems;
    assert!(
        problems
            .iter()
            .any(|p| p.contains("rule `tool-check`") && p.contains("ci/tools.toml")),
        "{problems:?}"
    );
}

#[test]
fn a_hosted_direct_job_missing_from_the_combined_diff_fails() {
    let len = check(&hosted_config(), &workspace(), &hosted_tree())
        .diff
        .len();
    let apply = tweaked(len, |a| a.jobs.retain(|j| j != "tool-check"));
    let problems = check_with(&hosted_config(), &workspace(), &hosted_tree(), &apply).problems;
    assert_eq!(
        problems,
        ["hosted job `tool-check` did not run on the combined diff"]
    );
}

#[test]
fn a_planned_hosted_direct_rule_that_is_not_listed_as_planned_fails() {
    let apply = tweaked(1, |a| a.planned.retain(|(j, _)| j != "tool-later"));
    let problems = check_with(&hosted_config(), &workspace(), &hosted_tree(), &apply).problems;
    assert!(
        problems.iter().any(|p| p.contains("rule `tool-later`")),
        "{problems:?}"
    );
}
